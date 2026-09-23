#!/usr/bin/env python3
"""Command-line entry point for the Character Forge pipeline.

  python forge_cli.py --text "chibi cat-girl mage with staff" --out out/mage
  python forge_cli.py --image concept.png --budget mobile --out out/hero
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine

ap = argparse.ArgumentParser(description="Text / concept image -> game-ready 3D character")
src = ap.add_mutually_exclusive_group(required=True)
src.add_argument("--text", help="character description")
src.add_argument("--image", help="concept image (PNG/JPG)")
ap.add_argument("--budget", choices=list(engine.BUDGETS), default="game")
ap.add_argument("--tex", type=int, default=1024, help="atlas resolution (512/1024/2048)")
ap.add_argument("--out", default="out/character", help="output directory")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
img = open(a.image, "rb").read() if a.image else None
rep = engine.run_pipeline(source="image" if a.image else "text", text=a.text,
                          image_bytes=img, budget=a.budget, tex_size=a.tex,
                          model_dir=a.out, prompt=a.text or a.image)
json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=2)
g = rep["geometry"]
print(f"style {rep['style']} | tags {', '.join(rep['tags'])} | parts {', '.join(rep['accessories']) or '-'}")
print(f"{g['triangles']:,} tris / {g['vertices']:,} verts "
      f"(from {g['triangles_before_optimize']:,}) | 1 material, {g['texture_size']}² atlas")
for s in rep["stages"]:
    print(f"  {s['ms']:5d} ms  {s['name']}")
print("wrote:", ", ".join(sorted(os.listdir(a.out))))
