#!/usr/bin/env python3
"""Command-line entry point for the Character Forge pipeline.

  python forge_cli.py --text "chibi cat-girl mage with staff" --out out/mage
  python forge_cli.py --image concept.png --budget mobile --out out/hero
  python forge_cli.py --image car.png --mode sculpt --depth 0.7 --out out/car
  python forge_cli.py --image coin.png --mode sculpt --relief --out out/coin   # flat back
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine

ap = argparse.ArgumentParser(description="Text / concept image -> game-ready 3D character")
src = ap.add_mutually_exclusive_group(required=True)
src.add_argument("--text", help="character description")
src.add_argument("--image", help="concept image (PNG/JPG)")
ap.add_argument("--budget", choices=list(engine.BUDGETS), default="game")
ap.add_argument("--tex", type=int, default=None, help="atlas resolution (512/1024/2048; default: 2048 for image input)")
ap.add_argument("--mode", choices=["auto", "character", "sculpt"], default="auto",
                help="auto: a standing figure is rigged, anything else is sculpted from pixels")
ap.add_argument("--back", help="optional back-view image, so the rear is measured too")
ap.add_argument("--relief", action="store_true", help="sculpt with a flat back (plaques, printing)")
ap.add_argument("--depth", type=float, default=None, help="sculpt depth multiplier, e.g. 0.6 or 1.3")
ap.add_argument("--out", default="out/character", help="output directory")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
img = open(a.image, "rb").read() if a.image else None
back = open(a.back, "rb").read() if a.back else None
rep = engine.run_pipeline(source="image" if a.image else "text", text=a.text,
                          image_bytes=img, budget=a.budget, tex_size=a.tex,
                          model_dir=a.out, prompt=a.text or a.image, mode=a.mode,
                          relief=True if a.relief else None, roundness=a.depth,
                          back_image=back)
json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=2)
g = rep["geometry"]
print(f"{rep.get('pipeline', 'character')} | style {rep['style']} | tags {', '.join(rep['tags'])} | parts {', '.join(rep['accessories']) or '-'}")
extra = f" | {rep['size_m'][0]}×{rep['size_m'][1]}×{rep['size_m'][2]} m" if rep.get("size_m") else ""
extra += f" | watertight={rep.get('watertight')}" if "watertight" in rep else ""
print(f"{g['triangles']:,} tris / {g['vertices']:,} verts "
      f"(from {g['triangles_before_optimize']:,}) | 1 material, {g['texture_size']}² atlas{extra}")
for s in rep["stages"]:
    print(f"  {s['ms']:5d} ms  {s['name']}")
print("wrote:", ", ".join(sorted(os.listdir(a.out))))
