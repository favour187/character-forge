"""Sculpt — turn *any* image into a real 3D mesh on CPU, in about a second.

The character rig in `engine.py` is great for humanoids and useless for a car, a
sword, a logo or a bust.  This module is the general case: a classical
reconstruction pipeline that needs no GPU and no network, so it fits inside
Render's 512 MB free instance.

    image ──► 1. isolate subject          alpha, or margin-LUT + morphology
                2. shape from silhouette  per-row ellipse lofting -> half-depth field
                3. shape from shading     luminance high-pass -> real surface detail
                4. implicit solid         f(x,y,z) = min(Zfront - z, Zback + z)
                5. marching cubes         closed, watertight manifold
                6. optimise               decimate to the polygon budget
                7. unwrap + bake          planar UV; colour/normal/roughness from the art

Step 4 is what keeps the result clean instead of "AI soup": the field is *linear*
in z, so marching cubes places both surfaces exactly on their zero crossings, the
rim closes itself wherever the depth field reaches 0, and a flat back is free
(Zback = 0).  No skirt to sew, no open edges, no floating junk.

Modes
  volume (default)  rounded front and back, like a molded figurine
  relief            front bulge, flat back — plaques, coins, 3D printing
  two-view          an extra "back" image drives the rear silhouette (visual hull)

Colour comes from the uploaded art itself, so the model looks like the input.
Surface detail is baked into a normal map at texture resolution, so even a
1.5k-triangle mobile mesh reads as a detailed one.
"""

from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image, ImageFilter

log = logging.getLogger("forge.sculpt")

WORK_MAX_SIDE = 512          # reconstruction happens at <=512 px; more is pointless
MAX_INPUT_PIXELS = 60_000_000
BUDGET_TRIS = {"mobile": 1500, "game": 5000, "high": 15000, "hero": 15000,
                  "auto": 5000}   # matches engine.BUDGETS so both paths report the same thing


# ---------------------------------------------------------------------------
# 1. subject isolation
# ---------------------------------------------------------------------------
def load_subject(image_bytes, max_side=WORK_MAX_SIDE):
    """Decode -> (rgb float32 (H,W,3) 0..255, mask bool, notes).

    Decoding is the most dangerous thing an upload does to a small box, so
    `Image.draft` lets libjpeg downscale *during* decode (a 12 MP phone photo
    costs ~1 MB, not 48 MB) and the pixel count is capped before convert().
    """
    notes = []
    try:
        im = Image.open(io.BytesIO(image_bytes))
        im.load()
    except Exception as e:                                          # noqa: BLE001
        raise ValueError("That file could not be read as an image "
                         f"({type(e).__name__}: {str(e)[:90]}). PNG, JPG or WEBP please.") from e
    w, h = im.size
    if w * h > MAX_INPUT_PIXELS:
        raise ValueError(f"image too large ({w}×{h} px) — keep it under {MAX_INPUT_PIXELS/1e6:.0f} MP")
    if im.format == "JPEG":
        im.draft("RGB", (max_side, max_side))
    im = im.convert("RGBA")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.uint8)
    alpha, rgb = arr[:, :, 3], arr[:, :, :3].astype(np.float32)
    H, W = rgb.shape[:2]

    if int(alpha.min()) < 235 and int(alpha.max()) > 30:
        mask = alpha > 96
        notes.append("Subject isolated from the image's alpha channel")
    else:
        row_bg = np.median(rgb[:, int(W * 0.92):, :], axis=1)      # right margin LUT
        col_bg = np.median(rgb[int(H * 0.92):, :, :], axis=0)      # bottom margin LUT
        dr = np.abs(rgb - row_bg[:, None, :]).max(axis=2)
        dc = np.abs(rgb - col_bg[None, :, :]).max(axis=2)
        mask = (dr > 26) & (dc > 26)
        notes.append("Background removed by margin-colour segmentation")

    from scipy import ndimage
    if int(mask.sum()) < 64:
        mask = np.ones((H, W), bool)
        mask[:2, :2] = False
        notes.append("No clear subject found - reconstructing the whole frame")
    else:
        opened = ndimage.binary_opening(mask, iterations=2)        # strips HUD text/lines
        if int(opened.sum()) > 64:
            mask = opened
        lab, n = ndimage.label(mask)
        if n > 1:                                                  # keep the main blobs
            sizes = ndimage.sum(np.ones_like(lab, np.int32), lab, range(1, n + 1))
            keep = [i + 1 for i in np.argsort(sizes)[::-1][:3] if sizes[i] > 0.08 * sizes.max()]
            mask = np.isin(lab, keep)
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), bool))
        mask = ndimage.binary_fill_holes(mask)                     # no see-through arms
        if int(mask.sum()) < 64:
            raise ValueError("Could not find a clear subject (a PNG with a transparent "
                             "background works best).")
    ys, xs = np.nonzero(mask)
    if len(ys) < 16:
        raise ValueError("Subject is too small to reconstruct — use a bigger crop.")
    return rgb, mask, [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], notes


# ---------------------------------------------------------------------------
# 2+3. the half-depth field
# ---------------------------------------------------------------------------
def half_depth(rgb, mask, box, roundness=1.0, shading=1.0, softness=1.4, depth_fit=1.0):
    """Half-thickness Z (metres) over the cropped box; the surface is z = ±Z(x,y).

    Two classical cues, combined with a min() so neither can run away:

    shape from silhouette - every horizontal run of the mask is a chord of a
    circular cross-section, so a run of width w contributes (w/2)*sqrt(1-u^2).
    Separate runs in one row (an arm off the torso) each get their own ellipse,
    which is what a real body does.

    maximal-inscribed-disc bound - the depth can never exceed the local radius of
    the largest disc fitting inside the silhouette. Without it a side view of a
    car comes out as deep as it is long; with it, thickness follows the *short*
    axis, as it must. This is the single biggest quality lever on the result.

    shape from shading - the luminance high-pass is folded in as a bounded bump
    term, which is what puts a nose on a face and a bulge on a pauldron.
    """
    from scipy import ndimage
    x0, y0, x1, y1 = box
    m = mask[y0:y1, x0:x1]
    h, w = m.shape
    z = np.zeros((h, w), np.float32)
    idx = np.arange(w, dtype=np.float32)
    row = m.astype(np.int8)
    pad0 = np.zeros((h, 1), np.int8)
    d = np.diff(np.concatenate([pad0, row, pad0], axis=1), axis=1)   # +1 run start, -1 end
    for y in range(h):
        starts = np.flatnonzero(d[y] == 1)
        ends = np.flatnonzero(d[y] == -1) - 1
        for s, e in zip(starts, ends):
            width = e - s + 1
            if width < 1:
                continue
            u = (idx[s:e + 1] - (s - 1.0)) / (width + 1.0) * 2.0 - 1.0
            prof = np.sqrt(np.clip(1.0 - u * u, 0.0, None)) * (width * 0.5)
            np.maximum(z[y, s:e + 1], prof, out=z[y, s:e + 1])

    # --- inscribed-disc bound, low-passed ------------------------------------
    # The raw distance field would force a knife edge at the top of a head (small
    # radius locally, even though the head is deep). Blurring it first keeps the
    # crown dome-shaped while still stopping a side view of a car from puffing out
    # as deep as it is long. A 2 px ramp on the raw field then closes the surface
    # exactly on the drawn silhouette, so the outline stays faithful.
    dt = ndimage.distance_transform_edt(m)
    if depth_fit:
        dt_lp = ndimage.gaussian_filter(dt, sigma=max(4.0, 0.07 * min(h, w)))
        cap = float(depth_fit) * np.maximum(dt_lp, dt * 0.35)
        np.minimum(z, cap, out=z)
        z *= np.clip(dt / 2.2, 0.0, 1.0)

    # A narrow run (a roofline, a crown of hair) would otherwise loft into a fin.
    k = int(np.clip(round(min(h, w) * 0.085) | 1, 3, 31))
    if k > 1:
        z = ndimage.maximum_filter1d(z, size=k, axis=0, mode="nearest")
        z = ndimage.gaussian_filter1d(z, sigma=k / 4.0, axis=0, mode="nearest")
        if depth_fit:
            np.minimum(z, cap, out=z)

    if shading:
        base = z.copy()
        rgb_ = rgb[y0:y1, x0:x1]
        r, g, b = rgb_[:, :, 0], rgb_[:, :, 1], rgb_[:, :, 2]
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        big = ndimage.gaussian_filter(lum, sigma=max(3.0, min(h, w) / 90.0))
        hi = np.clip((lum - big) * 2.6, -1.0, 1.0).astype(np.float32)
        hi = ndimage.gaussian_filter(hi, sigma=1.1)
        rim = ndimage.distance_transform_edt(m)
        weight = np.clip(rim / 3.0, 0.0, 1.0)               # no bumps on the outline
        z = base + hi * (base * (0.30 * float(shading))) * weight
        np.maximum(z, base * 0.30, out=z)                    # a shadow never pinches through
        np.minimum(z, base * 1.9, out=z)                     # nor bulges past a highlight

    z = z * float(roundness)                                   # user depth / plumpness
    if softness:
        z = ndimage.gaussian_filter(z, sigma=float(softness))
        z[~ndimage.binary_dilation(m, iterations=1)] = 0.0
    return np.maximum(z, 0).astype(np.float32)


# ---------------------------------------------------------------------------
# 4+5. implicit solid -> mesh
# ---------------------------------------------------------------------------
def mesh_from_field(zf, zb, step_m, relief=False, max_slices=90):
    """Marching cubes over f(x,y,z) = min(zf(x,y) - z, zb(x,y) + z), in metres.

    Zero crossings sit exactly on z = +zf and z = -zb (the field is linear in z,
    so there is no interpolation bias); outside the silhouette both terms are
    <= 0, which pinches the surface shut at the outline.  `zb == 0` (relief)
    gives a flat back for free.  Voxels are cubic so MC normals stay honest.
    """
    from skimage import measure
    # two rings of empty voxels around the field: marching cubes can only close a
    # surface that ends *inside* the volume, so an unpadded grid leaves open edges
    zf = np.pad(zf, 2); zb = np.pad(zb, 2)
    span = float(max(zf.max(), zb.max(), 1e-3)) * 1.02 + step_m
    nz = int(max(10, min(max_slices, round(2 * span / step_m) + 2)))
    zc = np.linspace(-span, span, nz, dtype=np.float32)
    if relief:
        zb = np.zeros_like(zb)
    # fields arrive in image layout [row=y, col=x]; marching cubes assigns array
    # axes to model axes in order, so transpose to [x, y, z] here (cheap: coarse)
    fx = np.ascontiguousarray(zf.T)[:, :, None]
    bx = np.ascontiguousarray(zb.T)[:, :, None]
    # the tiny margin makes the exterior strictly negative (not exactly 0 along the
    # outline at z=0, which left marching cubes an ambiguous -> non-watertight rim)
    field = np.minimum(fx - zc[None, None, :], bx + zc[None, None, :]) - step_m * 0.012
    del fx, bx
    del zc
    verts, faces, normals, _ = measure.marching_cubes(
        field, level=0.0, spacing=(step_m, step_m, max(step_m, 1e-6)))
    verts = verts - np.array([2.0, 2.0, 0.0]) * step_m      # undo the padding offset
    del field
    return verts, faces, normals


def downsample(field, target_tris):
    """Coarsen the depth field so marching cubes lands near the polygon budget.

    Generating a half-million triangle mesh and smashing it down to 1.6k wastes a
    second and ~200 MB of RAM - and RAM is what OOMs a 512 MB instance.  Nothing
    is lost visually: fine detail lives in the baked normal map and texture.
    Fields are image layout [row, col]; returns (coarse field, step in pixels).
    """
    ny, nx = field.shape
    px_w, px_h = max(int(nx), 2), max(int(ny), 2)
    # marching cubes emits ~6.8 faces per grid cell on this kind of closed blob, so
    # sizing the grid from the budget means decimation has almost nothing to do
    # (and heavy decimation is what puts non-manifold edges in the mesh)
    cells = float(np.clip(target_tris / 6.8, 650.0, 9000.0))
    g = max(1.0, float(np.sqrt(px_w * px_h / cells)))
    gx, gy = max(8, int(round(px_w / g))), max(8, int(round(px_h / g)))
    img = Image.fromarray(field.astype(np.float32))      # PIL size == (nx, ny)
    small = np.asarray(img.resize((gx, gy), Image.BILINEAR), np.float32)   # [y, x]
    return small, g


# ---------------------------------------------------------------------------
# 6. optimise
# ---------------------------------------------------------------------------
def decimate(verts, faces, target_tris):
    if len(faces) <= target_tris:
        return verts, faces, False
    try:
        import fast_simplification
        v, f = fast_simplification.simplify(np.ascontiguousarray(verts, np.float32),
                                            np.ascontiguousarray(faces, np.int32),
                                            target_count=int(target_tris), agg=7.0)
        return np.asarray(v, np.float32), np.asarray(f, np.int64), True
    except Exception as e:                                              # noqa: BLE001
        log.info("decimation skipped (%s)", e)
        return verts, faces, False


# ---------------------------------------------------------------------------
# 7. unwrap + bake
# ---------------------------------------------------------------------------
def _nearest_fill(img, mask):
    """Copy the nearest inside pixel into every outside texel (one vectorised step).

    UV padding and the rim then sample plausible art instead of black fringes.
    """
    from scipy import ndimage
    out = np.array(img, np.float32)
    keep = np.asarray(mask, bool)
    if keep.all():
        return out
    idx = ndimage.distance_transform_edt(~keep, return_distances=False, return_indices=True)
    out[~keep] = out[idx[0][~keep], idx[1][~keep]]
    return out


def bake(rgb, mask, zf, size=1024, normal_strength=2.4, step_m=1.0):
    # zf is in image layout [row, col] so it lines up with the resized art
    """baseColor / metallicRoughness / normal, all derived from the uploaded art."""
    from scipy import ndimage
    h, w = mask.shape
    art = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB").resize(
        (size, size), Image.BILINEAR)
    msk = Image.fromarray((mask * 255).astype(np.uint8), "L").resize((size, size),
                                                                     Image.BILINEAR)
    art_np = _nearest_fill(np.asarray(art, np.float32), np.asarray(msk, np.float32) > 120)
    base = Image.fromarray(np.clip(art_np, 0, 255).astype(np.uint8), "RGB")

    r, g, b = art_np[:, :, 0], art_np[:, :, 1], art_np[:, :, 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    sat = art_np.max(axis=2) - art_np.min(axis=2)
    metal = np.clip((34 - sat) * 7.5, 0, 255) * (lum > 118)          # bright + grey = metal
    rough = np.clip(212 - lum * 0.42 - np.maximum(0, 40 - sat) * 0.25, 50, 245)
    mr = np.zeros((size, size, 3), np.uint8)
    mr[:, :, 1] = rough.astype(np.uint8)
    mr[:, :, 2] = (metal * 0.9).astype(np.uint8)

    # normal map from the same field the geometry used, at full texture detail:
    # this is why a low-poly sculpt still shows folds, eyes and panel lines
    zimg = Image.fromarray(np.clip(zf / max(float(zf.max()), 1e-6) * 255,
                                   0, 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    zt = np.asarray(zimg, np.float32) / 255.0
    zt = ndimage.gaussian_filter(zt, sigma=max(0.6, size / 900.0))
    dy, dx = np.gradient(zt)
    k = normal_strength * (h / size)
    inv = 1.0 / np.sqrt((dx * k) ** 2 + (dy * k) ** 2 + 1.0)
    nrm = np.empty((size, size, 3), np.uint8)
    nrm[:, :, 0] = ((-dx * k * inv) * 0.5 + 0.5) * 255
    nrm[:, :, 1] = ((dy * k * inv) * 0.5 + 0.5) * 255      # v axis is flipped vs image rows
    nrm[:, :, 2] = (inv * 0.5 + 0.5) * 255
    return base, Image.fromarray(mr, "RGB"), Image.fromarray(nrm, "RGB").filter(
        ImageFilter.GaussianBlur(0.35))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def reconstruct(image_bytes, back_bytes=None, budget="game", tex_size=1024,
                roundness=1.0, shading=1.0, relief=False, target_height=1.8,
                depth_fit=1.0, target_tris=None):
    """Any image -> (trimesh.Trimesh with UVs+PBR material, report dict).

    `relief` = flat back (plaques / printing), `back_bytes` = feed a second image
    of the back so the rear silhouette is measured instead of inferred.
    `roundness` scales the depth, `depth_fit` tightens the inscribed-disc bound
    (lower = flatter / more relief-like), `shading` = how much surface detail is
    taken from the image's own light and shade, `target_tris` overrides `budget`.
    """
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    rgb, mask, box, notes = load_subject(image_bytes)
    x0, y0, x1, y1 = box
    px_h, px_w = y1 - y0, x1 - x0
    m_per_px = float(target_height) / max(px_h, px_w, 1)

    zf_img = half_depth(rgb, mask, box, roundness, shading,
                        depth_fit=depth_fit) * m_per_px                   # already cropped
    target = int(target_tris) if target_tris else BUDGET_TRIS.get(budget, 5200)
    target = max(300, min(60000, target))
    zfc, g_px = downsample(zf_img, target)        # coarse grid => cheap, small marching cubes
    step_m = m_per_px * g_px

    if back_bytes:
        rgb_b, mask_b, box_b, _ = load_subject(back_bytes)
        scale_h = max(px_h, 1) / max(box_b[3] - box_b[1], 1)             # align subject sizes
        zbf = half_depth(rgb_b, mask_b, box_b, roundness, 0.2,
                         depth_fit=depth_fit) * m_per_px * scale_h
        zb, _ = downsample(zbf, target)
        zb *= 0.85                                                        # backs read flatter
        notes.append("Back view supplied -> rear silhouette measured (two-view hull)")
    elif relief:
        zb = np.zeros_like(zfc)
        notes.append("Flat back (relief mode)")
    else:
        zb = zfc.copy()
        notes.append("Back inferred by front/back symmetry")

    verts, faces, _ = mesh_from_field(zfc, zb, step_m)
    tris_before = int(len(faces))
    verts, faces, decimated = decimate(verts, faces, target)

    V = np.asarray(verts, np.float64)
    V[:, 1] = -V[:, 1]                                 # image rows grow downward -> Y up
    V -= (V.min(axis=0) + V.max(axis=0)) * 0.5          # centre x / z
    V[:, 1] += (V.max(axis=0)[1] - V.min(axis=0)[1]) * 0.5   # rest the base on y=0
    mesh = trimesh.Trimesh(vertices=V, faces=faces, process=False)
    try:
        from trimesh import smoothing
        smoothing.filter_laplacian(mesh, lamb=0.38, iterations=2,
                                   volume_constraint=True)            # relax voxel stairs
    except Exception as e:                                             # noqa: BLE001
        log.debug("smoothing skipped: %s", e)
    try:      # marching cubes + decimation leave a few zero-area faces: they are what
        mesh.update_faces(mesh.nondegenerate_faces())   # trips the watertight test
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
    except Exception:                                   # noqa: BLE001
        pass
    try:
        if not mesh.is_watertight:
            mesh.fill_holes()
        mesh.fix_normals()
    except Exception:                                                  # noqa: BLE001
        pass

    ext = mesh.bounds[1] - mesh.bounds[0]
    v = np.asarray(mesh.vertices, np.float64)
    lo, hi = mesh.bounds
    span = np.maximum(hi - lo, 1e-6)
    pad = 0.022      # inset: the rim samples just inside the outline,
                   # not the background-blended edge texels
    uv = np.zeros((len(v), 2), np.float64)
    uv[:, 0] = (v[:, 0] - lo[0]) / span[0]
    uv[:, 1] = (v[:, 1] - lo[1]) / span[1]
    uv = pad + uv * (1 - 2 * pad)

    base, mr_img, nrm_img = bake(rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1], zf_img,
                                 size=tex_size, step_m=step_m)
    mesh.visual = TextureVisuals(uv=uv, material=PBRMaterial(
        baseColorFactor=[1, 1, 1, 1.0], baseColorTexture=base,
        metallicRoughnessTexture=mr_img, normalTexture=nrm_img,
        metallicFactor=1.0, roughnessFactor=1.0))

    # glTF multiplies COLOR_0 over baseColorTexture: darken back-facing / rim
    # vertices so the rear reads as a sculpted back, not a mirrored decal.
    zq = v[:, 2]
    back = np.clip((zq - np.percentile(zq, 52)) / (np.ptp(zq) * 0.20 + 1e-6), 0.0, 1.0)
    back = back ** 0.75                                  # reach the flat-back quickly
    tint = np.clip(1.0 - back * 0.62, 0.0, 1.0)          # darken ...
    tint = 0.30 + tint * 0.70                            # ... then lift off pure black
    tint = np.stack([tint * 0.86, tint * 0.90, tint], axis=1)   # cool grey, desaturated
    cols = np.concatenate([(tint * 255).astype(np.uint8),
                           np.full((len(tint), 1), 255, np.uint8)], axis=1)
    mesh.visual.vertex_attributes["color"] = cols          # -> glTF COLOR_0 (x4 baseColor)

    notes.append("Texture UV = front planar projection (1 material, 1 draw call)")
    notes.append("Normal map baked from the depth field -> detail survives decimation")
    info = {
        "source": "image",
        "reconstructor": "sculpt",
        "style": "relief" if relief else "sculpted volume",
        "notes": notes,
        "tags": [f"{px_w}×{px_h} px silhouette", f"depth {ext[2]:.2f} m",
                 "flat-back relief" if relief else "two-sided volume",
                 "watertight" if mesh.is_watertight else "open edges"],
        "accessories": [],
        "palette": {},
        "relief": bool(relief),
        "two_view": bool(back_bytes),
        "watertight": bool(mesh.is_watertight),
        "height_m": round(float(ext[1]), 2),
        "size_m": [round(float(s), 3) for s in ext],
        "geometry": {
            "vertices": int(len(mesh.vertices)), "triangles": int(len(mesh.faces)),
            "triangles_before_optimize": tris_before, "target_tris": int(target),
            "decimated": bool(decimated), "materials": 1,
            "texture_size": int(tex_size),
            "maps": ["baseColor", "metallicRoughness", "normal"],
            "watertight": bool(mesh.is_watertight), "volumes": 1,
        },
    }
    return mesh, uv, (base, mr_img, nrm_img), info
