# Character Forge — deployment record

| | |
|---|---|
| **Live app** | https://character-forge.onrender.com |
| **Health / API check** | https://character-forge.onrender.com/health |
| **Source (GitHub)** | https://github.com/favour187/character-forge |
| **Render dashboard** | https://dashboard.render.com/web/srv-dapn30jbc2fs73bb6910 |
| Render service id | `srv-dapn30jbc2fs73bb6910` |
| Render workspace | **My game** (`tea-d8nsc3bsq97s73bor410`) — *"My Workspace" had its free-tier quota exhausted* |
| Plan / region | Free · Frankfurt (closest region to Abuja) |
| Runtime | Python 3.12.9 · `gunicorn -w 1 --threads 4 -t 180 server:app` |
| AI analysis | **OpenRouter** — key reused from your *Agent-ai* / *ai-screen-assistant-server* services (`sk-or-v1-d…23de`, free-tier key, $0 credits → free models only, ~50 req/day) · primary model `nex-agi/nex-n2.5-mini:free` with fallbacks |
| Database | **Neon Postgres attached (2026-09-23)** — shared with the *gameforge-ai* project (`gameforge-api` service, `ep-purple-forest-b17w47jm` pooler, `neondb`); `/health` reports `database: true`, builds persist across deploys/restarts (table `characters`, kept to ≤80 rows) |
| Deployed | 2026-09-23 |

## API
```
POST /generate            form fields: text=<prompt> | image=<file>, budget=mobile|game|high
GET  /model/<id>/model.glb | obj.zip | model.stl | texture_atlas.png
GET  /report/<id>         analysis + geometry report (JSON)
GET  /recent              last 12 characters
GET  /health
```

## Redeploy after a code change
**Auto-deploy is on** — every push to `main` builds and goes live in ~1–2 min (verified).
Manual trigger if ever needed:
```bash
curl -X POST -H "Authorization: Bearer $RENDER_API_KEY" \
     https://api.render.com/v1/services/srv-dapn30jbc2fs73bb6910/deploys
```

## Attach Neon Postgres (persistent gallery)
1. Create a project at https://console.neon.tech (or hand me a working `napi_…` key / connection string).
2. Set the env var on the service — this triggers a redeploy automatically:
```bash
curl -X PUT -H "Authorization: Bearer $RENDER_API_KEY" -H "Content-Type: application/json" \
     https://api.render.com/v1/services/srv-dapn30jbc2fs73bb6910/env-vars/DATABASE_URL \
     -d '{"value":"postgresql://USER:PASSWORD@HOST/neondb?sslmode=require"}'
```
The schema is created on first boot (`db.init()`); `/health` then reports `"database": true`.

## OpenRouter notes
* The key is a **free-tier key with $0 credits**: only `:free` models work and OpenRouter caps them at ~50 requests/day (1000/day once the account holds ≥ $10 credits). Past the cap the app silently falls back to the heuristic analyzer — the analysis card shows an amber "fell back" notice.
* To use a stronger paid vision model after topping up: set `OPENROUTER_MODEL=google/gemini-2.5-flash-lite` (or `openai/gpt-4.1-mini`) on the Render service.
* Uncheck **AI analysis** in the UI to run the pure heuristic pipeline (instant, offline).

## Free-tier notes
* The instance sleeps after 15 min idle — the first request afterwards takes ~50 s to wake.
* 0.1 CPU: generation takes ≈ 10 s live (≈ 1 s locally).
* Disk is ephemeral: generated files survive until the next deploy/restart unless Neon is attached.

## Secrets
Kept **only** in `forge/.env` (git-ignored, chmod 600) in this workspace — never committed.
The tokens were shared in chat; rotate them when convenient (GitHub → Settings → Developer settings; Render → Account → API keys).

## Chat UI + AI director (latest)

* `GET /` is now a chat. One turn = one build; following text-only turns revise the same character
  (the session keeps `last_plan`, stored in `sessions/<id>.json` next to `models/<mid>/`).
* `director.py` = the planning stage (LLM → validated build plan: proportions, features, palette,
  materials, **target_tris**, texture size, detail resolution, `back_view`, `extra_parts`).
  Falls back to `engine.local_plan` / `local_revise` when the model is unavailable.
* Downloads: `GET /download/<mid>/{model.glb|obj.zip|model.stl|all.zip|report.json|*.png}`.
  Use these links outside the in-app preview iframe — the iframe blocks downloads by design.
* Env additions: `FORGE_PLAN_TIMEOUT_S=60`, `FORGE_PLAN_BUDGET_S=150`, `FORGE_PLAN_MAX_TOKENS=8000`
  (free models spend ~2.5k tokens thinking before the JSON, so keep max_tokens generous).
* Free-tier per-model limit is still ~50 requests/day on the cheap models; every build needs one
  plan call, so ~50 builds/day, then the rule planner takes over automatically.

## Quality pass (2026-09-23, later)

* **Viewer**: new *Anime* mode is the default — toon gradient shading, inverted-hull line art,
  soft contact shadow; *Material* mode keeps straight PBR.
* **Geometry**: face features (nose, brows, anime side-locks), pelvis/shoulder/ankle blend
  volumes (seam-free silhouette), higher base tessellation at game budget.
* **Concept-art texturing**: image builds sample the uploaded art onto the character (face via
  2-D skin-blob detection with a face-likeness gate; per-part body bands; 2048² atlas).
  Segmentation uses margin-LUT background estimation + flood + opening/erosion, so gradient
  skies and HUD clutter in game screenshots are handled far better.
* **Textures**: low-frequency normals (the raw Sobel grain read as noisy skin), cheek blush on
  chibi/stylized text builds.
* **Memory fix**: the 2048² atlas paint pass now runs float32 end-to-end and builds the normal
  map directly into the uint8 buffer — the original float64 + float-stack version spiked past
  the 512 MB free instance and OOM-killed the worker (502 on every request). Peak RSS is now
  ~307 MB for a full high-budget image build.
* **Verified live** with the shared Neon DB attached: `POST /chat` image build (15,015 tris,
  2048², persisted `true`), text build (5,470 tris, persisted `true`), `/recent`
  `persistent: true`, `all.zip` downloads.
