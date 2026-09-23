# Character Forge — deployment record

| | |
|---|---|
| **Live app** | https://character-forge.onrender.com |
| **Health / API check** | https://character-forge.onrender.com/health |
| **Source (GitHub)** | https://github.com/favour187/character-forge |
| **Render dashboard** | https://dashboard.render.com/web/srv-dapn30jbc2fs73bb6910 |
| Render service id | `srv-dapn30jbc2fs73bb6910` |
| Render workspace | **My game** (`tea-d8nsc3bsq97s73bor410`) — *"My Workspace" had its free-tier quota exhausted* |
| Plan / region | **Free · Frankfurt** — 0.1 CPU · **512 MB** · ephemeral disk. See [docs/MEMORY.md](docs/MEMORY.md) |
| Runtime | Python 3.12.9 · `gunicorn -w 1 --threads 4 -t 180 -b 0.0.0.0:$PORT server:app` |
| Auto-deploy | on, triggered by every push to `main` (verified: build + health-check ~55 s) |
| AI director | **OpenRouter** — free-tier key (shared with the *Agent-ai* / *ai-screen-assistant-server* services), primary `nex-agi/nex-n2.5-mini:free`, then Gemma/Qwen fallbacks, then the local rule planner. ~50 req/day on $0 credits; past the cap the app degrades instead of failing. |
| Database | **Neon Postgres** (`ep-purple-forest-b17w47jm` pooler, `neondb`, shared with *gameforge-ai*), table `characters`, ≤80 rows: report + GLB + STL + base-colour atlas, so the gallery and its downloads survive a redeploy |
| First deployed | 2026-09-23 |

## Redeploy after a code change

Push to `main` and it is live in about a minute:

```bash
git push origin main                     # auto-deploy handles it
curl -s https://character-forge.onrender.com/health | jq   # wait for 200 + ok:true
python tests/check.py https://character-forge.onrender.com # full end-to-end check
```

Manual trigger, if auto-deploy ever misses it:

```bash
curl -X POST -H "Authorization: Bearer $RENDER_API_KEY" \
     https://api.render.com/v1/services/srv-dapn30jbc2fs73bb6910/deploys
```

## Set / change an environment variable

This is how you turn the AI director on and off, attach Neon, or pin the texture size. A change restarts the
service (that counts against the free-plan restart budget, so batch them).

```bash
curl -s -X PUT -H "Authorization: Bearer $RENDER_API_KEY" -H "Content-Type: application/json" \
  https://api.render.com/v1/services/srv-dapn30jbc2fs73bb6910/env-vars/OPENROUTER_API_KEY \
  -d '{"value":"sk-or-v1-…"}'
```

Available knobs are listed in the README ([Environment variables](README.md#environment-variables)).

## Checking the memory ceiling

The free plan's 512 MB is the one thing that has actually broken this service. Two commands tell you
everything:

```bash
# what the app thinks it can afford right now
curl -s https://character-forge.onrender.com/health | jq .memory

# whether the kernel has ever OOM-killed it
curl -s -H "Authorization: Bearer $RENDER_API_KEY" \
  "https://api.render.com/v1/services/srv-dapn30jbc2fs73bb6910/events?limit=40" \
  | jq '[.[] | .event | select(.type=="server_failed") | {timestamp, details}]'
```

`atlas_px: 1024` and an empty second list = healthy. Full write-up with measurements:
**[docs/MEMORY.md](docs/MEMORY.md)**.

## Upgrading

1. Dashboard → Characteristics → **Starter ($7/mo, 0.5 CPU / 2 GB)**. Nothing else to change: `memguard`
   notices the bigger cgroup and starts choosing 2048² atlases by itself.
2. Or `numInstances: 2` on the same plan — each instance gets its own build slot. Add
   `FORGE_QUEUE_WAIT_S=8` if you would rather shed load than queue it.
3. For serious image-to-3D quality, the seam is `engine.run_sculpt()` — return `(mesh, uv, textures, report)`
   from a GPU provider (Tripo / Meshy / a self-hosted Hunyuan3D) and the whole site keeps working.

## Free-tier notes

* Instance sleeps after 15 min idle → the first request pays ~30–50 s of cold start. The UI says so in the
  progress overlay instead of looking broken.
* 0.1 CPU → a build that takes 0.8 s locally takes ~5–12 s there. That is why builds are serialised.
* Disk is ephemeral: `models/` is wiped on redeploy and pruned to `FORGE_KEEP_MODELS=24`. GLB / STL /
  base-colour come back from Neon; `obj.zip` and `all.zip` need the instance that built them.
* `frankfurt` is the closest free region to Nigeria; `ohio`/`singapore` are also free. Latency to Lagos is
  ~90–120 ms either way, so region choice does not matter much for a 5–10 s build.

## Secrets

Nothing secret is in this repository. `git log -p | grep -i "ghp_\|rnd_\|sk-or"` must stay empty, and
`.env` is git-ignored.

> ⚠️ **Rotate the GitHub and Render tokens that were pasted into chat** — GitHub → Settings → Developer
> settings → Personal access tokens → *generate new token* → *delete* `ghp_…`; Render → Account → API keys →
> *create* → *delete* `rnd_…`. Also re-issue the OpenRouter key if it has been shared. Anything pasted into
> a chat, an email or a screenshot should be treated as leaked. The deployment keeps working after rotation:
> Render's auto-deploy uses its own GitHub app integration, not your PAT — you only need the new Render key
> for API calls like the ones above.

## History

| date | deploy | what |
|---|---|---|
| 2026-09-23 | `98ee95c` | text/image → 3D pipeline, web UI, GLB/OBJ/STL |
| 2026-09-23 | `4fdc409`, `fff7000` | OpenRouter analysis stage, 429 back-off |
| 2026-09-23 | `d154c82`, `76d2325` | chat UI + AI build director; explicit colours win |
| 2026-09-23 | `16a22de` | float32 atlas + in-place normal map — first response to the OOM kill |
| 2026-09-23 | `cc9670c` | docs: note the atlas fix |
| 2026-09-23 | `744f946` | **`sculpt.py`: any image → watertight 3D. `memguard.py`: build slot + memory-sized atlas + `malloc_trim`, mode router, `429/503` semantics, `Retry-After`, disk pruning, `tests/check.py`, rebuilt responsive UI. `docs/API.md`, `docs/MEMORY.md`, `tests/preview.py`** |
| 2026-09-23 | `744f946` (live) | verified on the free instance: 58/58 end-to-end checks pass, 6 concurrent image builds all 200 in 38 s, `/health` memory 155 MB used of 512, **no `oomKilled` since** |
