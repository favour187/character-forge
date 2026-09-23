# The 512 MB ceiling — what happened, what it cost, and how it is enforced now

## The event

The Render API records exactly what killed the service (`GET /v1/services/<id>/events`):

```json
{"type":"server_failed","timestamp":"2026-09-23T12:17:04.8959Z",
 "details":{"oomKilled":{"memoryLimit":"512Mi"},"evicted":false}}
{"type":"server_failed","timestamp":"2026-09-23T12:18:18.366786Z",
 "details":{"oomKilled":{"memoryLimit":"512Mi"},"evicted":false}}
{"type":"server_available","timestamp":"2026-09-23T12:18:53.760529Z"}
```

So: not a crash in Python, not a build failure — the **kernel OOM killer** took the worker, twice, and the
free plan gives you no more than that. While the worker restarts, Render answers `502`, which is the only
symptom a user sees.

## Why it went over

The plan is **512 MB / 0.1 CPU**, and five things stacked against it:

1. **A 2048² atlas for every image build.** `run_pipeline` had `tex = 2048` for `source == "image"`, with the
   comment "concept art deserves the full atlas". It does — on a machine that can afford it. One atlas at
   2048² is not 16 MB, it is ~**320 MB**, because `paint_atlas` holds a float32 colour buffer, a height
   buffer, a smoothed copy for the normal map, and the Sobel/gradient temporaries of both at the same time.
2. **Concurrency inside one process.** `gunicorn -w 1 --threads 4` was chosen so the UI stays responsive
   during a build. Four requests then ran four pipelines *in the same address space*, so their peaks added
   up. Threads are the right call for latency and the wrong call for memory unless the heavy section is
   serialised — which it was not.
3. **Full-resolution decode.** A 12 MP phone photo (`3024×4032`) decoded to RGBA is 48 MB, and
   `np.array(img)` plus an `int32` copy of the RGB put ~150 MB of pixel buffers in play before the pipeline
   even started thinking. Cropping afterwards does not help; the peak already happened.
4. **float64 by default.** NumPy gives you float64 unless you ask. The atlas and its temporaries were
   `float64` early on, which doubles #1.
5. **No return of freed pages.** CPython + numpy free into glibc's arena, not back to the OS. So the
   process's high-water mark *persisted*, and the next build's peak sat on top of it: two big builds in a
   row looked like the sum of both.

## Measured

Single build, 768×1024 input, high-water RSS of the process (baseline = imports only, 105 MB), measured
with the current float32 / in-place-normal code:

| atlas | peak RSS | what it means |
|---|---|---|
| 512²  | **143 MB** | text builds, or a very tight instance |
| 1024² | **194 MB** | what the guard now picks on a 512 MB plan |
| 2048² | **424 MB** | what the old code asked for on *every* image build — over the line as soon as two people arrive |

(The float64 version these numbers replaced was not measured here; it is what the two `oomKilled` events
above came from, so all that can honestly be said is that it exceeded 512 MB on its own.)

Concurrency (the code *before* the guard, 3024×4032 photo, threads sharing one address space — this is
the shape of a normal traffic burst on a live URL):

| | 1 request | 2 concurrent | 4 concurrent |
|---|---|---|---|
| atlas 2048² | 302 MB | 492 MB | **864 MB ← dead** |
| atlas 1024² | 220 MB | 311 MB | 512 MB ← borderline |
| atlas 512²  | 177 MB | 313 MB | 520 MB ← borderline |

That is the whole story: `2 × 424 MB ≈ 850 MB > 512 MB`. Two people dropping an image at the same moment
was enough.

**After** the guard, the same 6-way concurrent image burst on the emulated 512 MB budget:
`VmHWM` (peak) **224 MB**, steady RSS 141 MB, all six requests 200 OK, `/health` still green.

**And on the real free instance**, after deploying `744f946`:

```
$ python3 tests/check.py https://character-forge.onrender.com --slow
done in 96s - 58 passed, 0 failed
concurrency burst: [200 x6] in 38s
$ curl -s https://character-forge.onrender.com/health | jq .memory
{"limit_mb": 512, "used_mb": 155, "headroom_mb": 357, "atlas_px": 1024, "queued": 0, "serialised": true}
$ curl -s .../services/srv-dapn30jbc2fs73bb6910/events   # build_ended, deploy_ended - no server_failed
```

Note `/health` reading `limit_mb: 512` on Render: that means `memguard` found the real cgroup ceiling,
not a fallback — so the atlas it picked is sized against the number that actually kills the box.

## What enforces it now — `memguard.py`

| lever | what it does |
|---|---|
| `memguard.slot()` | one pipeline at a time, per worker. Queued requests wait up to `FORGE_QUEUE_WAIT_S` (75 s) and then get a **429 with a message**, not a 502. The UI retries once on its own. |
| `memguard.tex_size()` | reads the *real* cgroup ceiling (`memory.max`, falling back to `RLIMIT_AS`, then 60 % of `MemTotal`), subtracts what the process is using right now, and picks the largest atlas that fits — 2048 on a Starter box, 1024 on free, 512 if it is tight. An atlas pixel is budgeted at 88 bytes because that is what it measured at. |
| `memguard.release()` | `gc.collect()` + `malloc_trim(0)` after every build, in a `finally`, so high-water marks do not stack across requests. |
| `FORGE_MEMORY_MB` / `FORGE_MAX_TEXTURE` | pin the budget or the atlas when you want deterministic behaviour. |

Plus, in the pipeline itself:

* `Image.draft("RGB", (768, 768))` before `convert()` — libjpeg downscales **during** decode, so a 12 MP
  photo costs ~2 MB instead of ~150 MB (both `engine.analyze_image` and `sculpt.load_subject`).
* A 60 MP pixel guard on decode, and `MAX_CONTENT_LENGTH = 16 MB`, so a decompression bomb cannot be used
  to OOM the service.
* float32 everywhere in the atlas and the normal-map bake (in-place ops, no stacked temporaries).
* The sculptor sizes its **voxel grid from the triangle budget** instead of marching cubes over the whole
  image: ~5k faces come out of a ~30×40 grid, not out of a 512² field (which once produced 529k triangles
  and 381 MB before decimation).
* `FORGE_KEEP_MODELS=24` prunes `models/` after each build, so the 512 MB of ephemeral disk cannot fill up
  (a full disk shows up as a failed build, which is confusing and worse).

## Checking it yourself

```bash
curl -s https://character-forge.onrender.com/health | jq .memory
# {"limit_mb":512,"used_mb":118,"headroom_mb":394,"atlas_px":1024,"queued":0,"serialised":true}

# the events API is the only place the OOM kill is recorded:
curl -s -H "Authorization: Bearer $RENDER_API_KEY" \
  "https://api.render.com/v1/services/srv-dapn30jbc2fs73bb6910/events?limit=40" \
  | jq '[.[] | .event | select(.type=="server_failed") | {timestamp, .details}]'

python tests/check.py https://character-forge.onrender.com --slow   # includes the 6-way burst
```

`/health` reporting `atlas_px: 1024` on the free plan is the fix working as designed.

## Getting 2048 textures back

Only one real answer: more RAM. `plan: starter` (2 GB, $7/mo) makes `tex_size()` choose 2048 by itself —
no code change, nothing to remember. Until then, a 1024² atlas on a mesh whose normal map is baked from the
depth field looks nearly identical in-engine at gameplay distance, and the polygon budget is the constraint
that actually matters for a mobile/indie game.

## Known limits

* **One build at a time per worker.** With 0.1 CPU that is not a real loss (parallel builds would each be
  ~4× slower and the total would be the same), but it does mean a burst of users see a queue. Scaling out is
  `numInstances: 2+` — each instance gets its own slot and its own 512 MB.
* **A build cannot be interrupted.** `gunicorn -t 180` is the ceiling; the longest observed build on the live
  free instance is ~12 s cold, ~1 s warm.
* **Postgres keeps the artifacts, not the disk.** `obj.zip` needs the folder that built it; GLB, STL and the
  base-colour atlas are mirrored to Neon so downloads survive a restart.
