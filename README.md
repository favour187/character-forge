# Character Forge

**Live:** https://character-forge.onrender.com · **Code:** https://github.com/favour187/character-forge


A self-contained **text / concept-image → game-ready 3D character** tool.
It implements the exact pipeline you described — analyze → reconstruct → UV →
texture → optimize → export — as runnable code, with a web UI, a live PBR
viewer and Unity-ready downloads.

```
IMAGE / TEXT
     ↓   engine.analyze_text / engine.analyze_image
AI understands the object        silhouette, proportions (heads-tall), palette, parts
     ↓   engine.build_character
Predicts 3D shape                 parametric volumes in A-pose, hidden side inferred
     ↓   engine.optimize_parts
Game topology                     quadric decimation to a polygon budget (mobile/game/hero)
     ↓   engine.assemble  (+ _part_uv)
Creates UVs / surface structure   per-part projection into a 4×4 atlas
     ↓   engine.paint_atlas
Generates textures                baseColor · metallic-roughness · normal
     ↓   engine.finalize
3D MODEL                          GLB (PBR) · OBJ+MTL · STL
```

## Run

```bash
pip install -r requirements.txt
python server.py            # → http://localhost:8000
```

The web viewer defaults to **Anime** mode (cel shading + line art + soft contact shadow);
switch to *Material* for straight PBR. Image builds run at 2048² textures.

CLI (no UI):

```bash
python forge_cli.py --text "chibi cat-girl mage with staff and pointy hat" --out out/mage
python forge_cli.py --image concept.png --budget mobile --out out/hero
```

## How each stage works

| Stage | What happens | Where |
|---|---|---|
| **1. Analyze (AI)** | When `OPENROUTER_API_KEY` is set, the prompt or image is sent to an LLM / vision model through **OpenRouter** (`ai.py`). It returns style, proportions, a full palette, accessories, species and a one-line brief as JSON. For images the pixel-measured colours and heads-tall proportions still win (they're exact); the model contributes semantics. Free-tier models by default (`nex-agi/nex-n2.5-mini:free` → Gemma 4 → Qwen 3.8 → `openrouter/free`); set `OPENROUTER_MODEL` to use a paid one (e.g. `google/gemini-2.5-flash-lite`). Any failure (rate-limit, timeout) falls back to the heuristic analyzers below. | `ai.py` |
| **1. Analyze (text, heuristic)** | Keyword semantics → species/build, style (chibi / stylized / realistic), accessories (backpack, wizard hat, sword, shield, cape, staff, horns, cat ears, tail, glasses), "`<colour> <part>`" grammar (e.g. *red armor*, *brown boots*). | `analyze_text` |
| **1. Analyze (image, heuristic)** | Subject isolated from alpha or by margin-LUT segmentation + flood + morphological cleanup (handles gradient skies, HUD text, game screenshots); row-width profile finds the **neck pinch** → *heads-tall* → proportion style; shoulder width → build; positional colour priors (crown → hair, face → skin, torso, legs, feet). Occluded back is inferred by symmetry + semantic defaults. | `analyze_image` |
| **2. Reconstruct geometry** | Character assembled as a volumetric primitive rig (head, hair cap, eyes, neck, torso, belt, arms, hands, legs, boots + accessories) in **Y-up, metres, A-pose**. | `build_character` |
| **3. UVs** | Each part is box-projected into its own material tile of a 4×4 atlas → one material / one draw call. | `_part_uv`, `assemble` |
| **4. Textures** | With image input the per-part tiles are **sampled from the concept art itself** (the character wears the reference design: face located by 2-D skin-blob detection, body bands per part), so the built model carries the art's colours and detail; text-only builds get procedural per-material detail (weave, leather grain, brushed metal, hair strands). Base colour + glTF metallic-roughness (B = metal, G = rough) + low-frequency normal map. | `paint_atlas` |
| **5. Optimize** | Per-part quadric-error decimation (`fast_simplification`) to the budget: **Mobile ≈1.5k**, **Game ≈5k**, **Hero ≈15k** triangles; UVs are recomputed after decimation so seams never tear. | `optimize_parts` |
| **6. Export** | `model.glb` (embedded PBR textures), `model.obj` + `.mtl` + PNGs, `model.stl`, plus `report.json`. | `finalize` |

## Environment variables

| Var | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | enables the AI analysis stage (optional) |
| `OPENROUTER_MODEL` / `OPENROUTER_FALLBACK_MODELS` | model chain (defaults are free vision models) |
| `DATABASE_URL` | Neon/Postgres mirror for the gallery (optional) |
| `FORGE_KEEP_ROWS` | max rows kept in Postgres (default 80) |

## Unity import

* **GLB** – install *glTFast* (`com.unity.cloud.gltfast` via Package Manager),
  drop the file in `Assets/`. Materials, normal and metal/rough maps import automatically.
* **OBJ** – native import; assign `material_0.png` (base), `normal_atlas.png`, `mr_atlas.png`.
* Scale is 1 unit = 1 m; the mesh is Y-up and A-posed for rigging (Mixamo / Blender → Humanoid rig).

## Layout

```
forge/
├── engine.py        the 6-stage pipeline (pure Python, no GPU)
├── server.py        Flask API:  POST /generate  ·  GET /model/<id>/<file>
├── forge_cli.py     command-line front end
├── static/
│   ├── index.html   UI + three.js PBR viewer (lit / wireframe / normals / UV-check)
│   └── vendor/      three.js r160 (vendored – works offline)
├── models/          generated results (one folder per job)
└── docs/            sample renders
```

## Where the "AI" is – and how to upgrade it

The analyzer is a deterministic vision/NLP heuristic and the reconstruction is
parametric, so it runs in ~1 s on CPU. Every stage is a function with a clean
contract, so you can swap in a neural model per stage:

* **Analyze** → already pluggable: `ai.py` talks to any OpenRouter model; swap `OPENROUTER_MODEL`
  for a stronger vision model when credits allow.
* **Reconstruct** → replace `build_character` with an image-to-3D network (TripoSR,
  InstantMesh, Hunyuan3D…) returning a `trimesh.Trimesh`; the UV / texture / optimize /
  export stages work unchanged.
* **Texture** → feed the UV layout to a diffusion texture painter instead of `paint_atlas`.
