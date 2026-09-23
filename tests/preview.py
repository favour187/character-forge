#!/usr/bin/env python3
"""Render shaded previews of built models into docs/ — no GPU, no browser.

    python tests/preview.py                # sculpt + character samples
    python tests/preview.py --base .       # where docs/ lives

A 40-line painter's-algorithm rasteriser (matplotlib PolyCollection) is enough to eyeball a mesh: it
shows the silhouette, the baked texture and the vertex tint, which is exactly what a reviewer needs to
judge "does this look like my image".  These are *renders of the exported mesh*, not screenshots of the web
viewer — the caption in the README says so.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixtures import buf, car, face, knight, logo      # noqa: E402

VIEWS = [(0, 0, "front"), (35, 12, "3/4"), (90, 8, "side"), (180, 0, "back")]


def _rotation(azim, elev):
    a, e = np.deg2rad(azim), np.deg2rad(elev)
    ca, sa, ce, se = np.cos(a), np.sin(a), np.cos(e), np.sin(e)
    return np.array([[ca, 0.0, sa], [se * sa, ce, -se * ca], [-ce * sa, se, ce * ca]])


def draw(ax, mesh, light=(0.4, 0.5, 0.85)):
    import matplotlib.pyplot as plt                    # noqa: F401 (kept for clarity of deps)
    from matplotlib.collections import PolyCollection

    v = np.asarray(mesh.vertices, float)
    f = np.asarray(mesh.faces, int)
    n = np.asarray(mesh.face_normals, float)
    uv = getattr(mesh.visual, "uv", None)
    tex = None
    mat = getattr(mesh.visual, "material", None)
    if mat is not None and getattr(mat, "baseColorTexture", None) is not None:
        tex = np.asarray(mat.baseColorTexture.convert("RGB"), np.uint8)
    vc = None
    va = getattr(mesh.visual, "vertex_attributes", None)
    try:
        if va is not None and "color" in va:
            vc = np.asarray(va["color"], float)[:, :3] / 255.0
    except Exception:                                    # noqa: BLE001
        vc = None

    R = _rotation(*ax._forge_view)
    p, nn = v @ R.T, n @ R.T
    L = np.array(light) / np.linalg.norm(light)
    shade = np.clip(nn @ L, -0.30, 1.0) * 0.80 + 0.30
    if tex is not None and uv is not None:
        h, w = tex.shape[:2]
        uvc = np.asarray(uv, float)[f].mean(axis=1)
        uu = np.clip((uvc[:, 0] * (w - 1)).astype(int), 0, w - 1)
        vv = np.clip(((1 - uvc[:, 1]) * (h - 1)).astype(int), 0, h - 1)
        base = tex[vv, uu][:, :3] / 255.0
    else:
        base = np.full((len(f), 3), 0.72)
    if vc is not None:
        base = base * vc[f].mean(axis=1)
    col = np.clip(base * shade[:, None], 0, 1)
    order = np.argsort(p[f, 2].mean(axis=1))
    ax.add_collection(PolyCollection(p[f, :2][order], facecolors=col[order],
                                     edgecolors=col[order] * 0.9, linewidths=0.18))
    lo, hi = p[:, :2].min(axis=0), p[:, :2].max(axis=0)
    pad = (hi - lo) * 0.10 + 1e-6
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect("equal")
    ax.axis("off")


def show(mesh, title, path, views=VIEWS, figsize=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(views), figsize=figsize or (3.5 * len(views), 3.6), dpi=110)
    for ax, (az, el, nm) in zip(np.atleast_1d(axs), views):
        ax._forge_view = (az, el)
        draw(ax, mesh)
        ax.set_title(nm, color="#8e97b0", fontsize=10)
    fig.suptitle(title, color="#e9ecf5", fontsize=12, y=0.985)
    fig.patch.set_facecolor("#0a0c12")
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", path)


def main(base):
    import engine
    import sculpt

    docs = os.path.join(base, "docs")
    os.makedirs(docs, exist_ok=True)

    for nm, im, extra in (("sculpt-face", face(), {}),
                          ("sculpt-car", car(), {}),
                          ("sculpt-logo", logo(), {}),
                          ("sculpt-relief", car(), {"relief": True}),
                          ("sculpt-mobile", car(), {"budget": "mobile"})):
        mesh, _uv, _tx, info = sculpt.reconstruct(buf(im), budget=extra.get("budget", "game"),
                                                  tex_size=1024, **{k: v for k, v in extra.items()
                                                                    if k != "budget"})
        g = info["geometry"]
        show(mesh, f"{nm.replace('-', ' ')} · {info['size_m'][0]}×{info['size_m'][1]}×{info['size_m'][2]} m"
                   f" · {g['triangles']:,} tris · watertight={g['watertight']}",
             os.path.join(docs, f"{nm}.png"))

    # a character-rig sample for the README, through the real pipeline
    out = os.path.join(base, "models", "_preview_char")
    os.makedirs(out, exist_ok=True)
    rep = engine.run_pipeline(source="image", image_bytes=buf(knight()), budget="game",
                              mode="character", use_ai=False, model_dir=out, prompt="knight")
    import trimesh
    mesh = trimesh.load(os.path.join(out, "model.glb"), file_type="glb")
    mesh = list(mesh.geometry.values())[0] if hasattr(mesh, "geometry") else mesh
    show(mesh, f"character rig from a concept image · {rep['geometry']['triangles']:,} tris · "
               f"{rep['heads_tall']} heads tall", os.path.join(docs, "ui-character-rig.png"),
         figsize=(4.0 * 4, 4.0))
    print(f"character-rig sample built in {rep['build_ms'] if 'build_ms' in rep else 0} ms -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main(ap.parse_args().base)
