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
| Database | **not yet attached** — the Neon API key supplied returned `401`; app runs disk-only until `DATABASE_URL` is set |
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
Push to `main`, then (public repo without the GitHub-app link, so trigger it explicitly):
```bash
curl -X POST -H "Authorization: Bearer $RENDER_API_KEY" \
     https://api.render.com/v1/services/srv-dapn30jbc2fs73bb6910/deploys
```
or click **Manual Deploy → Deploy latest commit** in the dashboard.

## Attach Neon Postgres (persistent gallery)
1. Create a project at https://console.neon.tech (or hand me a working `napi_…` key / connection string).
2. Set the env var on the service — this triggers a redeploy automatically:
```bash
curl -X PUT -H "Authorization: Bearer $RENDER_API_KEY" -H "Content-Type: application/json" \
     https://api.render.com/v1/services/srv-dapn30jbc2fs73bb6910/env-vars/DATABASE_URL \
     -d '{"value":"postgresql://USER:PASSWORD@HOST/neondb?sslmode=require"}'
```
The schema is created on first boot (`db.init()`); `/health` then reports `"database": true`.

## Free-tier notes
* The instance sleeps after 15 min idle — the first request afterwards takes ~50 s to wake.
* 0.1 CPU: generation takes ≈ 10 s live (≈ 1 s locally).
* Disk is ephemeral: generated files survive until the next deploy/restart unless Neon is attached.

## Secrets
Kept **only** in `forge/.env` (git-ignored, chmod 600) in this workspace — never committed.
The tokens were shared in chat; rotate them when convenient (GitHub → Settings → Developer settings; Render → Account → API keys).
