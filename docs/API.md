# HTTP API

Everything is `multipart/form-data` (or `application/x-www-form-urlencoded` when there is no file).
Responses are JSON. A build is one request: there is no job id to poll — the request holds until the model
is written, then returns the report. On the free instance that is ~1 s warm, ~10 s cold, and up to 75 s if
you are queued behind someone else.

## `POST /chat` — one conversational turn

Keeps a session so the next message revises the same subject.

| field | type | meaning |
|---|---|---|
| `session` | str | 12-char id; omit or send empty to start one. Also accepted as `?session=`. |
| `message` | str | the description, or the change to apply ("make the back flat", "add wings", "keep it under 2000 tris") |
| `image` | file | concept art. PNG (alpha helps), JPG, WEBP. ≤ 16 MB, ≤ 60 MP |
| `back_image` | file | optional back view → the rear silhouette is measured instead of inferred (sculpt mode) |
| `mode` | `auto`\|`character`\|`sculpt` | `auto` routes by what the image analysis finds |
| `budget` | `auto`\|`mobile`\|`game`\|`high` | triangle budget; `auto` lets the director choose (default `game`, 5k) |
| `relief` | `1`/`0` | sculpt with a flat back (plaques, coins, printing) |
| `roundness` | `0.25`–`1.6` | depth multiplier for the sculpted volume |
| `use_ai` | `1`/`0` | `0` skips the OpenRouter call and uses the rule planner (instant, offline) |

A text-only turn after a build is a **revision**: the plan from the last build is sent back to the director
with your change, and everything you did not mention stays identical. After a *sculpt* build the server also
keeps your uploaded pixels for the session, so "now flatten the back" re-runs the reconstruction on the same
image rather than building a fresh character.

→ `200 {"session", "turn", "report"}` · `400` bad input · `413` oversized upload · `429` queued (`Retry-After`) · `503` memory · `500` pipeline error

## `POST /generate` — same fields, no session

Good for scripts and CI. Returns the report directly.

```bash
curl -s -F image=@sprite.png -F mode=sculpt -F budget=mobile \
     https://character-forge.onrender.com/generate | jq '{pipeline, size_m, geometry}'
```

## `GET /chat/<session>`

`{"session","title","turns":[…],"reports":{id: report}}` — enough to redraw a conversation after a reload.

## `POST /reset`

`{"session","title","turns":[]}`.

## `GET /recent`

`{"items":[{id, source, prompt, style, triangles, created_at}], "persistent": bool}` — newest 12; straight
from Postgres when `DATABASE_URL` is set, from the local disk otherwise.

## Files

| route | notes |
|---|---|
| `GET /model/<id>/<file>` | inline, no `Content-Disposition` — this is what the 3D viewer loads |
| `GET /download/<id>/model.glb` | glTF 2.0 binary, PBR, `POSITION` + `TEXCOORD_0` + `COLOR_0`, three textures |
| `GET /download/<id>/model.obj` | + `material.mtl` (`GET /download/<id>/material.mtl`) |
| `GET /download/<id>/obj.zip` | OBJ + MTL + the three maps + a README (only on the instance that built it) |
| `GET /download/<id>/model.stl` | geometry only, for slicing |
| `GET /download/<id>/texture_atlas.png` \| `normal_atlas.png` \| `mr_atlas.png` | base colour, normal, metallic-roughness (B = metal, G = rough) |
| `GET /download/<id>/all.zip` | everything above + `report.json` |
| `GET /report/<id>` | the report, rebuilt from Postgres if the folder is gone |

## `GET /health`

```json
{"ok":true,"database":true,"ai":true,"ai_model":"nex-agi/nex-n2.5-mini:free",
 "planner":"ai-director","modes":["character","sculpt","relief"],
 "memory":{"limit_mb":512,"used_mb":118,"headroom_mb":394,"atlas_px":1024,"queued":0,"serialised":true}}
```

`memory` comes from `memguard` and is the fastest way to see what the instance can afford right now.

## Report shape

```json
{
  "id":"0fb00fbf033e", "pipeline":"sculpt",       // or "character"
  "source":"image", "prompt":"…", "style":"sculpted volume",
  "plan":{"brief":"…", "target_tris":5000, "reconstructor":"silhouette + shading -> implicit solid",
          "features":["cape","staff"], "palette":{"garment_a":"#5a46a0"}, "back_view":"…"},
  "ai":{"enabled":true,"used":false,"model":null,"plan_source":"measured from the image (no LLM needed)","error":null},
  "geometry":{"vertices":2604,"triangles":5000,"triangles_before_optimize":11172,"target_tris":5000,
              "decimated":true,"materials":1,"texture_size":1024,
              "maps":["baseColor","metallicRoughness","normal"],"watertight":true,"volumes":1},
  "size_m":[1.804,0.701,0.739], "height_m":1.79, "heads_tall":4.6,   // heads_tall: character only
  "watertight":true, "relief":false, "two_view":false,
  "tags":["258×345 px silhouette","depth 0.74 m","two-sided volume"],
  "notes":["Background removed by margin-colour segmentation","…"],
  "stages":[{"name":"Isolate silhouette + read depth","ms":320}, …],
  "build_ms":833, "memory":{"limit_mb":512,"headroom_mb":397,"atlas_px":1024},
  "downloads":{"glb":"/download/…/model.glb","obj":"…","stl":"…","texture":"…","normal":"…","mr":"…","all":"…"},
  "preview":"/model/…/model.glb", "persisted":true
}
```

## Errors

| code | when | body |
|---|---|---|
| `400` | no prompt and no image; unreadable image; subject too small; `mode=sculpt` with no image | `{"error":"…"}` |
| `413` | upload over `FORGE_MAX_UPLOAD_MB` | `{"error":"That file is over the 16 MB limit…"}` |
| `429` | another build holds the slot for longer than `FORGE_QUEUE_WAIT_S` | `{"error","busy":true,"retry_after":N}` + `Retry-After` header |
| `503` | the pipeline hit the memory ceiling | `{"error":"…512 MB…","memory":{…}}` |
| `500` | unexpected failure (the trace is in the service log, never in the response) | `{"error":"Pipeline failed: …"}` |

## Python / CLI

```bash
python forge_cli.py --text "chibi mage with a purple robe" --out out/mage
python forge_cli.py --image car.png --mode sculpt --depth 0.8 --out out/car
python forge_cli.py --image coin.png --mode sculpt --relief --out out/coin
python forge_cli.py --image front.png --back back.png --mode sculpt --out out/two_view
```

Or in code:

```python
import engine
rep = engine.run_pipeline(source="image", image_bytes=open("car.png","rb").read(),
                          budget="game", mode="sculpt", relief=False, roundness=1.0,
                          model_dir="out/car", prompt="car")
print(rep["size_m"], rep["geometry"]["watertight"], rep["downloads"]["glb"])
```
