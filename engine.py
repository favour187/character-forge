"""
Character Forge engine — the full image/text -> game-ready 3D pipeline.

Stages (mirrors the Tripo/Meshy-style flow, implemented locally):
  1. analyze input      (text semantics, or image: silhouette / proportions / palette)
  2. reconstruct geometry (volumetric primitive mesh assembly)
  3. unwrap UVs          (per-part box-projection into a baked atlas)
  4. generate textures   (base color + metallic-roughness + normal maps)
  5. optimize topology   (quadric decimation that respects UV seams)
  6. export              (GLB / OBJ+MTL / STL)
"""

import io
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import trimesh
from trimesh.visual.texture import TextureVisuals
from trimesh.visual.material import PBRMaterial

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------

TILE_NAMES = ["skin", "hair", "garment_a", "garment_b", "pants",
              "leather", "metal", "accent", "boots", "eye"]
TILE = {n: i for i, n in enumerate(TILE_NAMES)}
ATLAS_COLS, ATLAS_ROWS = 4, 4

NAMED_COLORS = {
    "red": (200, 45, 45), "crimson": (160, 25, 40), "maroon": (110, 35, 40),
    "orange": (225, 130, 40), "gold": (218, 165, 50), "golden": (218, 165, 50),
    "yellow": (230, 200, 70), "green": (70, 150, 80), "emerald": (40, 160, 110),
    "teal": (45, 150, 150), "cyan": (70, 190, 200), "blue": (60, 95, 190),
    "navy": (35, 50, 110), "azure": (70, 140, 220), "purple": (115, 70, 170),
    "violet": (130, 80, 190), "lavender": (170, 150, 215), "magenta": (200, 60, 150),
    "pink": (230, 140, 170), "brown": (125, 85, 50), "tan": (190, 160, 120),
    "beige": (210, 190, 160), "black": (35, 35, 40), "white": (235, 235, 235),
    "grey": (130, 130, 135), "gray": (130, 130, 135), "silver": (190, 195, 200),
    "bronze": (170, 120, 60),
}

BUDGETS = {
    "mobile": {"tris": 1500, "sphere_sub": 2, "cyl_sec": 8, "cap_cnt": (4, 8)},
    "game":   {"tris": 5000, "sphere_sub": 3, "cyl_sec": 14, "cap_cnt": (6, 14)},
    "high":   {"tris": 15000, "sphere_sub": 4, "cyl_sec": 24, "cap_cnt": (10, 24)},
}

STYLE_PROPS = {
    "chibi":     {"heads": 3.1, "build": 1.08, "arm_len": 0.72},
    "stylized":  {"heads": 4.6, "build": 1.00, "arm_len": 0.92},
    "realistic": {"heads": 6.4, "build": 0.94, "arm_len": 1.05},
}

# ----------------------------------------------------------------------------
# 1A. text analysis
# ----------------------------------------------------------------------------

def default_params():
    return dict(
        style="stylized", height=1.70, build=1.0, head_boost=1.0,
        skin=(232, 178, 142), hair=(74, 50, 35),
        garment_a=(106, 74, 158), garment_b=(236, 228, 210),
        pants=(58, 62, 84), boots=(90, 70, 50), leather=(128, 90, 56),
        metal=(176, 181, 190), accent=(216, 168, 60), eye=(40, 42, 52),
        accessories=set(), tags=[],
    )


def _set_color(p, target_key, rgb):
    p[target_key] = tuple(int(c) for c in rgb)


def analyze_text(text):
    p = default_params()
    t = " " + (text or "").lower() + " "
    tags = p["tags"]

    def has(*words):
        return any(w in t for w in words)

    # species / body
    if has("orc", "ogre", "brute"):
        p["skin"] = (120, 155, 85); p["build"] = 1.28; tags.append("orc build")
    if has("dwarf"):
        p["style"] = "chibi"; p["build"] = 1.22; tags.append("dwarf proportions")
    if has("elf", "elven"):
        p["build"] = 0.90; p["hair"] = (214, 196, 130); tags.append("elf")
    if has("robot", "mech", "android", "cyborg"):
        p["skin"] = (150, 158, 168); p["garment_a"] = (90, 98, 112)
        tags.append("robotic")
    if has("vampire"):
        p["skin"] = (232, 224, 216); tags.append("vampire")

    # style / proportions
    if has("chibi", "cute", "super deformed", " sd ", "kid", "child", "young student"):
        p["style"] = "chibi"; tags.append("chibi proportions")
    elif has("realistic", "real proportions", "adult"):
        p["style"] = "realistic"; tags.append("realistic proportions")
    if has("tall"):
        p["height"] = 1.85
    if has("short", "small"):
        p["height"] = 1.55
    if has("muscular", "buff", "stocky", "heavy"):
        p["build"] *= 1.18
    if has("slim", "slender", "lean", "skinny"):
        p["build"] *= 0.88

    # accessories
    if has("backpack", "satchel", "school bag", "rucksack", "student", "school"):
        p["accessories"].add("backpack"); tags.append("backpack")
    if has("wizard hat", "witch hat", "pointy hat", "pointed hat", "sorcerer", "wizard", "mage"):
        p["accessories"].add("wizard_hat"); tags.append("wizard hat")
    if has("staff"):
        p["accessories"].add("staff"); tags.append("staff")
    if has("sword", "blade", "knight", "warrior", "paladin"):
        p["accessories"].add("sword"); tags.append("sword")
    if has("shield"):
        p["accessories"].add("shield"); tags.append("shield")
    if has("cape", "cloak"):
        p["accessories"].add("cape"); tags.append("cape")
    if has("horn", "demon", "tiefling"):
        p["accessories"].add("horns"); tags.append("horns")
    if has("cat", "neko", "feline", "beastkin"):
        p["accessories"].add("cat_ears"); tags.append("cat ears")
    if has("tail") or has("cat", "neko", "demon", "tiefling"):
        p["accessories"].add("tail"); tags.append("tail")
    if has("glasses", "spectacles"):
        p["accessories"].add("glasses"); tags.append("glasses")
    if has("armor", "armour", "plate"):
        p["garment_a"] = (150, 156, 168); tags.append("armored")

    # colors: "<color> <target>"
    targets = {
        "robe": "garment_a", "dress": "garment_a", "outfit": "garment_a",
        "armor": "garment_a", "armour": "garment_a", "shirt": "garment_a",
        "tunic": "garment_a", "jacket": "garment_a", "uniform": "garment_a",
        "cloak": "garment_b", "cape": "garment_b", "scarf": "garment_b",
        "trim": "garment_b", "hood": "garment_b",
        "pants": "pants", "trousers": "pants", "skirt": "pants",
        "boots": "boots", "shoes": "boots",
        "hair": "hair", "skin": "skin",
        "backpack": "leather", "bag": "leather", "belt": "leather",
        "strap": "leather", "satchel": "leather",
        "sword": "metal", "blade": "metal", "staff": "leather",
        "hat": "accent", "gem": "accent", "eyes": "eye",
    }
    for cname, rgb in NAMED_COLORS.items():
        for tok in (" " + cname + " ", " " + cname + "-"):
            idx = 0
            while True:
                i = t.find(tok, idx)
                if i < 0:
                    break
                idx = i + 1
                after = t[i + len(tok): i + len(tok) + 24].split()
                key = None
                for w in after[:2]:
                    w = w.strip(".,!?;:")
                    if w in targets:
                        key = targets[w]
                        break
                if key:
                    _set_color(p, key, rgb)
                else:
                    _set_color(p, "garment_a", rgb)
                tags.append(f"{cname} tint")
    # dedupe tags, keep order
    seen = set(); p["tags"] = [x for x in tags if not (x in seen or seen.add(x))]
    p["source_notes"] = ["Hidden surfaces built from semantic defaults (a full 3D body is "
                         "reconstructed, not just the visible side)."]
    return p


# ----------------------------------------------------------------------------
# 1B. image analysis
# ----------------------------------------------------------------------------

def _background_mask(fg_guess):
    """Background = non-subject regions connected to the image border."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(~fg_guess)
        border_labels = np.unique(np.concatenate(
            [lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        border_labels = border_labels[border_labels != 0]
        return np.isin(lab, border_labels)
    except Exception:
        h, w = fg_guess.shape
        bg = np.zeros((h, w), bool)
        stack = [(y, x) for x in range(w) for y in (0, h - 1) if not fg_guess[y, x]]
        stack += [(y, x) for y in range(h) for x in (0, w - 1) if not fg_guess[y, x]]
        while stack:
            y, x = stack.pop()
            if bg[y, x]:
                continue
            bg[y, x] = True
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not bg[ny, nx] and not fg_guess[ny, nx]:
                    stack.append((ny, nx))
        return bg


def _dominant_colors(px, k=4, min_share=0.04):
    """Coarse-bin histogram clustering: robust for flat-shaded concept art,
    reasonable for painted/photographic input. Returns colors sorted by area."""
    if len(px) < 8:
        return []
    px = px.astype(np.int32)
    bins = (px >> 4)                                   # 16 levels / channel
    key = bins[:, 0] * 256 + bins[:, 1] * 16 + bins[:, 2]
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    order = np.argsort(counts)[::-1]
    clusters = []                                      # [sum_rgb, count]
    for oi in order[:40]:
        members = px[inv == oi]
        mean = members.mean(axis=0)
        cnt = int(counts[oi])
        merged = False
        for c in clusters:
            centre = c[0] / c[1]
            if np.linalg.norm(centre - mean) < 38:
                c[0] += members.sum(axis=0); c[1] += cnt
                merged = True
                break
        if not merged:
            clusters.append([members.sum(axis=0).astype(np.float64), cnt])
    total = len(px)
    clusters.sort(key=lambda c: -c[1])
    out = []
    for s, cnt in clusters:
        if cnt >= total * min_share:
            out.append(tuple(int(v) for v in np.round(s / cnt)))
        if len(out) >= k:
            break
    return out


def _head_fraction(widths):
    """Find the neck pinch in the row-width profile -> head height fraction."""
    H = len(widths)
    if H < 20:
        return 0.22
    k = max(3, H // 40)
    sm = np.convolve(widths, np.ones(k) / k, mode="same")
    lo, hi = int(H * 0.08), int(H * 0.45)
    seg = sm[lo:hi]
    if len(seg) < 3:
        return 0.22
    best, best_score = None, 0.0
    for i in range(1, len(seg) - 1):
        left_peak = seg[:i].max()
        right_peak = seg[i + 1:].max()
        pinch = min(left_peak, right_peak) - seg[i]
        score = pinch / (min(left_peak, right_peak) + 1e-9)
        if score > best_score:
            best_score, best = score, i
    if best is None or best_score < 0.08:
        return 0.22
    return float(np.clip((lo + best) / H, 0.12, 0.42))


def analyze_image(png_bytes):
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    img.thumbnail((768, 768))
    arr = np.array(img)
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3].astype(np.int32)

    if int(alpha.min()) < 200 and int(alpha.max()) > 30:
        mask = alpha > 60
        seg_note = "Subject isolated from PNG alpha channel"
    else:
        border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        bg = np.median(border, axis=0)
        fg_guess = np.linalg.norm(rgb - bg, axis=2) > 30
        mask = ~_background_mask(fg_guess)
        seg_note = "Subject isolated by border-colour flood segmentation"

    ys, xs = np.where(mask)
    if len(ys) < 200:
        raise ValueError("Could not find a clear subject in the image "
                         "(try a cleaner background or a PNG with transparency).")

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    H = max(y1 - y0, 1); W = max(x1 - x0, 1)
    aspect = W / H

    widths = mask[y0:y1 + 1].sum(axis=1).astype(float)
    head_frac = _head_fraction(widths)
    heads_tall = 1.0 / head_frac

    # shoulder width relative to height -> build
    sh_band = widths[int(H * head_frac):int(H * min(0.95, head_frac + 0.22))]
    shoulder_w = (np.percentile(sh_band, 90) if len(sh_band) else W) / H

    def band_pixels(f0, f1):
        yy0 = y0 + int(H * f0); yy1 = max(y0 + int(H * f1), yy0 + 2)
        sm = mask[yy0:yy1, x0:x1 + 1]
        return rgb[yy0:yy1, x0:x1 + 1][sm]

    head_end = head_frac
    torso_end = head_frac + (1 - head_frac) * 0.50
    # positional priors: crown of head -> hair, face -> skin, torso, legs, feet
    crown = _dominant_colors(band_pixels(0.0, head_end * 0.42), k=3)
    face = _dominant_colors(band_pixels(head_end * 0.42, head_end), k=4)
    mid = _dominant_colors(band_pixels(head_end, torso_end), k=4)
    low = _dominant_colors(band_pixels(torso_end, 0.9), k=4)
    feet = _dominant_colors(band_pixels(0.88, 1.0), k=3)

    def dist(a, b):
        return float(np.linalg.norm(np.array(a, float) - np.array(b, float)))

    def skinness(c):
        r, g, b = [float(v) for v in c]
        return r >= 60 and r >= g >= b and 15 < (r - b) < 140

    p = default_params()
    p["source_notes"] = [seg_note]
    hair = crown[0] if crown else None
    skin = next((c for c in face if skinness(c) and (hair is None or dist(c, hair) > 30)), None)
    if skin is None and face:
        skin = next((c for c in face if hair is None or dist(c, hair) > 30), None)
    if skin:
        p["skin"] = skin
        p["source_notes"].append("Skin tone sampled from face region")
    else:
        p["source_notes"].append("No skin tone found - using default")
    if hair:
        p["hair"] = hair
        p["source_notes"].append("Hair colour sampled from crown of head")
    if mid:
        p["garment_a"] = mid[0]
        p["source_notes"].append("Primary garment colour sampled from torso")
        second = next((c for c in mid[1:] if dist(c, mid[0]) > 40 and
                       (skin is None or dist(c, skin) > 30)), None)
        p["garment_b"] = second or tuple(int(min(255, v * 1.15 + 20)) for v in mid[0])
    if low:
        p["pants"] = low[0]
        p["source_notes"].append("Lower garment colour sampled from legs")
    if feet:
        p["boots"] = feet[0]
    top = crown + face

    if heads_tall < 3.9:
        p["style"] = "chibi"
    elif heads_tall > 5.8:
        p["style"] = "realistic"
    p["build"] = float(np.clip(0.55 + shoulder_w * 1.9, 0.82, 1.3))
    p["height"] = float(np.clip(1.25 + heads_tall * 0.085, 1.3, 1.9))
    p["tags"] = [f"{heads_tall:.1f} heads tall", f"silhouette aspect {aspect:.2f}",
                 f"{p['style']} proportions"]

    def greyish(c):
        r, g, b = c
        return max(r, g, b) - min(r, g, b) < 22 and 90 < min(r, g, b) < 235
    if mid and greyish(mid[0]):
        p["tags"].append("metallic surfaces detected")

    p["palette_read"] = {"top": top, "mid": mid, "low": low, "skin": skin, "hair": hair}
    p["source_notes"].append("Back / occluded geometry inferred by symmetry + semantic defaults")
    return p


# ----------------------------------------------------------------------------
# 2-3. geometry + UVs
# ----------------------------------------------------------------------------

class Part:
    __slots__ = ("mesh", "tile")

    def __init__(self, mesh, tile):
        self.mesh = mesh
        self.tile = tile


def _rot(angle_deg, axis, point=None):
    return trimesh.transformations.rotation_matrix(np.radians(angle_deg), axis, point)


# trimesh primitives are Z-aligned; game characters are Y-up.  These helpers
# hand back Y-aligned versions (capsule/cylinder centred at origin, cone with
# its base at the origin pointing +Y).
_Z_TO_Y = _rot(-90, [1, 0, 0])


def cap_y(height, radius, count):
    m = trimesh.creation.capsule(height=height, radius=radius, count=count)
    m.apply_transform(_Z_TO_Y)
    return m


def cyl_y(radius, height, sections):
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    m.apply_transform(_Z_TO_Y)
    return m


def cone_y(radius, height, sections):
    m = trimesh.creation.cone(radius=radius, height=height, sections=sections)
    m.apply_transform(_Z_TO_Y)
    return m


def build_character(p, budget):
    """Assemble the character from parametric volumes. Returns (parts, notes).

    Layout: Y-up, metres, origin between the feet, facing +Z, A-pose."""
    sp = STYLE_PROPS[p["style"]]
    H = p["height"]
    heads = sp["heads"]
    build = p["build"] * sp["build"]
    head_r = H / (2 * heads) * p.get("head_boost", 1.0)
    head_c = H - head_r * 1.02
    neck_y = H - head_r * 1.95
    hip_y = H * 0.47
    torso_len = neck_y - hip_y
    torso_mid = (neck_y + hip_y) / 2
    torso_r = H * 0.085 * build
    sh_x = torso_r * 1.25 * build                # shoulder x-offset
    sub = budget["sphere_sub"]; sec = budget["cyl_sec"]; cnt = budget["cap_cnt"]
    S = trimesh.creation.icosphere
    BOX = trimesh.creation.box
    parts = []

    def add(mesh, tile):
        parts.append(Part(mesh, tile))

    # ---- head + hair + eyes -------------------------------------------------
    head = S(subdivisions=sub, radius=head_r)
    head.apply_transform(np.diag([1.0, 1.06, 0.98, 1.0]))
    head.apply_translation([0, head_c, 0])
    add(head, TILE["skin"])

    hair = S(subdivisions=sub, radius=head_r * 1.09)
    hair = trimesh.intersections.slice_mesh_plane(
        hair, plane_normal=[0, 1, 0], plane_origin=[0, -head_r * 0.15, 0])
    hair.apply_translation([0, head_c + head_r * 0.06, -head_r * 0.07])
    if len(hair.faces) > 4:
        add(hair, TILE["hair"])

    for s in (-1, 1):
        e = S(subdivisions=max(1, sub - 1), radius=head_r * 0.11)
        e.apply_translation([s * head_r * 0.36, head_c + head_r * 0.05, head_r * 0.88])
        add(e, TILE["eye"])

    # ---- neck / torso / hips ------------------------------------------------
    neck = cyl_y(head_r * 0.32, head_r * 0.6, sections=sec)
    neck.apply_translation([0, neck_y + head_r * 0.1, 0])
    add(neck, TILE["skin"])

    torso = cap_y(torso_len * 0.78, torso_r, count=cnt)
    torso.apply_transform(np.diag([1.35, 1.0, 0.85, 1.0]))
    torso.apply_translation([0, torso_mid, 0])
    add(torso, TILE["garment_a"])

    belt = cyl_y(torso_r * 1.05, hip_y * 0.12, sections=sec)
    belt.apply_transform(np.diag([1.3, 1.0, 0.9, 1.0]))
    belt.apply_translation([0, hip_y + hip_y * 0.02, 0])
    add(belt, TILE["leather"])

    # ---- arms (A-pose, ~24 degrees from the body) ----------------------------
    arm_len = torso_len * 1.05 * sp["arm_len"]
    arm_r = torso_r * 0.34
    ang = 24.0
    for s in (-1, 1):
        shoulder = np.array([s * sh_x * 1.1, neck_y - head_r * 0.25, 0.0])
        arm = cap_y(arm_len, arm_r, count=cnt)
        arm.apply_translation(shoulder - [0, arm_len / 2, 0])
        arm.apply_transform(_rot(s * ang, [0, 0, 1], point=shoulder))
        add(arm, TILE["garment_b"])
        dirv = np.array([s * np.sin(np.radians(ang)), -np.cos(np.radians(ang)), 0.0])
        hand = S(subdivisions=max(1, sub - 1), radius=arm_r * 1.2)
        hand.apply_translation(shoulder + dirv * (arm_len + arm_r * 0.9))
        add(hand, TILE["skin"])

    # ---- legs / boots ---------------------------------------------------------
    leg_r = torso_r * 0.44
    leg_len = hip_y * 0.86
    for s in (-1, 1):
        x = s * torso_r * 0.55
        leg = cap_y(leg_len, leg_r, count=cnt)
        leg.apply_translation([x, hip_y * 0.05 + leg_len / 2, 0])
        add(leg, TILE["pants"])
        boot = cyl_y(leg_r * 1.15, hip_y * 0.18, sections=sec)
        boot.apply_transform(np.diag([1.0, 1.0, 1.45, 1.0]))
        boot.apply_translation([x, hip_y * 0.09, leg_r * 0.35])
        add(boot, TILE["boots"])

    acc = p["accessories"]
    back_z = -(torso_r * 0.85)

    # ---- backpack ---------------------------------------------------------------
    if "backpack" in acc:
        bp = BOX(extents=[torso_r * 1.7, torso_len * 0.85, torso_r * 0.8])
        bp.apply_translation([0, torso_mid + torso_len * 0.05, back_z - torso_r * 0.4])
        add(bp, TILE["leather"])
        pocket = BOX(extents=[torso_r * 1.1, torso_len * 0.32, torso_r * 0.28])
        pocket.apply_translation([0, torso_mid - torso_len * 0.15, back_z - torso_r * 0.92])
        add(pocket, TILE["accent"])
        for s in (-1, 1):
            strap = BOX(extents=[torso_r * 0.22, torso_len * 0.9, torso_r * 0.35])
            strap.apply_translation([s * torso_r * 0.6, torso_mid + torso_len * 0.05,
                                     torso_r * 0.75])
            add(strap, TILE["leather"])

    # ---- wizard hat ---------------------------------------------------------------
    if "wizard_hat" in acc:
        top_y = head_c + head_r * 0.78
        brim = cyl_y(head_r * 1.5, head_r * 0.1, sections=sec)
        brim.apply_translation([0, top_y, 0])
        add(brim, TILE["garment_a"])
        cone = cone_y(head_r * 0.92, head_r * 1.9, sections=sec)
        cone.apply_transform(_rot(-10, [1, 0, 0]))
        cone.apply_translation([0, top_y, 0])
        add(cone, TILE["garment_a"])
        band = cyl_y(head_r * 0.94, head_r * 0.16, sections=sec)
        band.apply_translation([0, top_y + head_r * 0.12, 0])
        add(band, TILE["accent"])

    # ---- cape --------------------------------------------------------------------
    if "cape" in acc:
        cape = BOX(extents=[torso_r * 2.7, H * 0.6, torso_r * 0.1])
        cape.apply_translation([0, -H * 0.3, 0])
        cape.apply_transform(_rot(-8, [1, 0, 0]))
        cape.apply_translation([0, neck_y, back_z])
        add(cape, TILE["garment_b"])

    # ---- sword (on the right hip) -------------------------------------------------
    if "sword" in acc:
        blade = BOX(extents=[torso_r * 0.18, H * 0.48, torso_r * 0.05])
        blade.apply_translation([0, H * 0.24, 0])
        guard = BOX(extents=[torso_r * 0.6, torso_r * 0.1, torso_r * 0.14])
        grip = cyl_y(torso_r * 0.08, H * 0.09, sections=max(6, sec // 2))
        grip.apply_translation([0, -H * 0.05, 0])
        for m, tl in ((blade, TILE["metal"]), (guard, TILE["accent"]), (grip, TILE["leather"])):
            m.apply_transform(_rot(-25, [0, 0, 1]))
            m.apply_transform(_rot(180, [1, 0, 0]))       # point the blade down
            m.apply_translation([sh_x * 1.35, hip_y * 1.05, back_z * 0.6])
            add(m, tl)

    # ---- staff (in the left hand) -------------------------------------------------
    if "staff" in acc:
        sx = -(sh_x * 1.1 + np.sin(np.radians(ang)) * (arm_len + arm_r))
        rod = cyl_y(torso_r * 0.08, H * 0.95, sections=max(6, sec // 2))
        rod.apply_translation([sx, H * 0.475, torso_r * 0.15])
        add(rod, TILE["leather"])
        gem = S(subdivisions=max(1, sub - 1), radius=torso_r * 0.24)
        gem.apply_translation([sx, H * 0.99, torso_r * 0.15])
        add(gem, TILE["accent"])

    # ---- shield (left forearm) -----------------------------------------------------
    if "shield" in acc:
        sx = -(sh_x * 1.1 + np.sin(np.radians(ang)) * arm_len * 0.75)
        disc = trimesh.creation.cylinder(radius=torso_r * 1.0, height=torso_r * 0.14,
                                         sections=sec)
        disc.apply_transform(_rot(90, [0, 1, 0]))      # Z-axis -> X-axis
        disc.apply_translation([sx - arm_r * 1.2, torso_mid - torso_len * 0.15, torso_r * 0.2])
        add(disc, TILE["metal"])
        boss = S(subdivisions=max(1, sub - 1), radius=torso_r * 0.2)
        boss.apply_translation([sx - arm_r * 1.2 - torso_r * 0.1,
                                torso_mid - torso_len * 0.15, torso_r * 0.2])
        add(boss, TILE["accent"])

    # ---- horns / cat ears / tail / glasses ------------------------------------------
    if "horns" in acc:
        for s in (-1, 1):
            horn = cone_y(head_r * 0.16, head_r * 0.7, sections=max(6, sec // 2))
            horn.apply_transform(_rot(-s * 28, [0, 0, 1]))
            horn.apply_translation([s * head_r * 0.55, head_c + head_r * 0.7, 0])
            add(horn, TILE["accent"])
    if "cat_ears" in acc:
        for s in (-1, 1):
            ear = cone_y(head_r * 0.24, head_r * 0.55, sections=max(6, sec // 2))
            ear.apply_transform(_rot(-s * 14, [0, 0, 1]))
            ear.apply_translation([s * head_r * 0.52, head_c + head_r * 0.85, 0])
            add(ear, TILE["hair"])
    if "tail" in acc:
        tail = cap_y(head_r * 1.8, head_r * 0.15, count=(3, 6))
        tail.apply_translation([0, -head_r * 0.9, 0])
        tail.apply_transform(_rot(-50, [1, 0, 0]))
        tail.apply_translation([0, hip_y * 0.95, back_z])
        add(tail, TILE["hair"])
    if "glasses" in acc:
        for s in (-1, 1):
            ring = trimesh.creation.annulus(r_min=head_r * 0.16, r_max=head_r * 0.22,
                                            height=head_r * 0.05)   # Z-axis = facing forward
            ring.apply_translation([s * head_r * 0.36, head_c + head_r * 0.05, head_r * 0.95])
            add(ring, TILE["metal"])

    notes = [f"{len(parts)} volumes assembled in A-pose (rig-friendly), Y-up, 1 unit = 1 m"]
    return parts, notes


def _part_uv(mesh, tile_idx):
    """Box-projection style planar UV into the part's atlas tile."""
    v = mesh.vertices
    mn, mx = v.min(axis=0), v.max(axis=0)
    ext = mx - mn
    order = np.argsort(ext)[::-1]
    a, b = order[0], order[1]
    uv = np.zeros((len(v), 2), dtype=np.float64)
    uv[:, 0] = (v[:, a] - mn[a]) / (ext[a] + 1e-9)
    uv[:, 1] = (v[:, b] - mn[b]) / (ext[b] + 1e-9)
    pad = 0.03
    uv = pad + uv * (1 - 2 * pad)
    tx = tile_idx % ATLAS_COLS
    ty = tile_idx // ATLAS_COLS
    uv[:, 0] = (tx + uv[:, 0]) / ATLAS_COLS
    uv[:, 1] = (ty + uv[:, 1]) / ATLAS_ROWS
    return uv


def assemble(parts):
    """Merge parts into one mesh + uv array."""
    vs, fs, uvs = [], [], []
    off = 0
    for pt in parts:
        m = pt.mesh
        vs.append(m.vertices)
        fs.append(m.faces + off)
        uvs.append(_part_uv(m, pt.tile))
        off += len(m.vertices)
    verts = np.vstack(vs)
    faces = np.vstack(fs)
    uv = np.vstack(uvs)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # drop degenerate / duplicate faces (vertex array untouched -> UVs stay aligned)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    return mesh, uv


# ----------------------------------------------------------------------------
# 4. textures
# ----------------------------------------------------------------------------

def _tile_detail(kind, rng, n):
    """Return (height map, roughness scalar field) per material kind."""
    h = np.zeros((n, n), dtype=np.float64)
    rough = np.ones((n, n), dtype=np.float64) * 0.85
    if kind in ("garment_a", "garment_b", "pants"):
        h += rng.normal(0, 0.6, (n, n))
        weave = np.sin(np.arange(n) * np.pi / 2)[None, :] * 0.25
        h += weave + weave.T * 0.5
        rough *= 0.95
    elif kind == "skin":
        h += rng.normal(0, 0.25, (n, n))
        rough *= 0.7
    elif kind == "hair":
        h += rng.normal(0, 0.4, (n, n))
        strands = np.sin(np.arange(n) * np.pi / 1.5)[:, None] * 0.5
        h += strands
        rough *= 0.6
    elif kind == "leather":
        h += rng.normal(0, 0.5, (n, n))
        dots = rng.random((n // 8, n // 8)) > 0.7
        big = np.kron(dots, np.ones((8, 8))) * 0.8
        h += big
        rough *= 0.75
    elif kind == "metal":
        streaks = rng.normal(0, 0.35, (n, 1)) * np.ones((1, n))
        h += streaks
        rough *= 0.35
    elif kind == "accent":
        h += rng.normal(0, 0.3, (n, n))
        rough *= 0.5
    elif kind == "boots":
        h += rng.normal(0, 0.45, (n, n))
        rough *= 0.8
    elif kind == "eye":
        rough *= 0.15
    return h, np.clip(rough, 0.05, 1.0)


def paint_atlas(p, size=1024):
    """Base color + metallic-roughness + normal atlases."""
    n = size // ATLAS_COLS
    rng = np.random.default_rng(42)

    color = np.zeros((n * ATLAS_ROWS, n * ATLAS_COLS, 3), dtype=np.float64)
    mr = np.zeros((n * ATLAS_ROWS, n * ATLAS_COLS, 3), dtype=np.uint8)
    height = np.zeros((n * ATLAS_ROWS, n * ATLAS_COLS), dtype=np.float64)
    mr[:, :, 0] = 255  # occlusion slot (unused) -> white

    palette = {
        "skin": p["skin"], "hair": p["hair"], "garment_a": p["garment_a"],
        "garment_b": p["garment_b"], "pants": p["pants"], "leather": p["leather"],
        "metal": p["metal"], "accent": p["accent"], "boots": p["boots"],
        "eye": p["eye"],
    }
    metallic = {"metal": 0.92, "accent": 0.35}

    for name, i in TILE.items():
        tx, ty = i % ATLAS_COLS, i // ATLAS_COLS
        y0, x0 = (ATLAS_ROWS - 1 - ty) * n, tx * n   # v=0 is the bottom row of the image
        h, rough = _tile_detail(name, rng, n)
        base = np.array(palette[name], dtype=np.float64)
        shade = 1.0 + h * 0.06
        color[y0:y0 + n, x0:x0 + n] = base[None, None, :] * shade[:, :, None]
        mr[y0:y0 + n, x0:x0 + n, 1] = (rough * 255).astype(np.uint8)
        mr[y0:y0 + n, x0:x0 + n, 2] = int(metallic.get(name, 0.04) * 255)
        height[y0:y0 + n, x0:x0 + n] = h

    color = np.clip(color, 0, 255).astype(np.uint8)
    color_img = Image.fromarray(color, "RGB").filter(ImageFilter.GaussianBlur(0.6))

    # normal map from height (Sobel)
    dy, dx = np.gradient(height)
    strength = 2.2
    nrm = np.stack([-dx * strength, -dy * strength, np.ones_like(height)], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm_img = Image.fromarray(((nrm * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")

    mr_img = Image.fromarray(mr, "RGB")
    return color_img, mr_img, nrm_img


# ----------------------------------------------------------------------------
# 5. optimization
# ----------------------------------------------------------------------------

def optimize_parts(parts, target_tris, min_faces=48):
    """Game-topology pass: quadric-error decimation applied per volume so every
    part keeps a clean, seam-free UV projection afterwards. The polygon budget
    is distributed proportionally to each part's density."""
    total = sum(len(pt.mesh.faces) for pt in parts)
    if total <= target_tris:
        return parts, total, False
    ratio = target_tris / total
    try:
        import fast_simplification
    except Exception:
        fast_simplification = None
    out = []
    for pt in parts:
        m = pt.mesh
        n = len(m.faces)
        goal = max(min_faces, int(n * ratio))
        if n <= goal:
            out.append(pt)
            continue
        try:
            if fast_simplification is not None:
                v, f = fast_simplification.simplify(
                    np.asarray(m.vertices, dtype=np.float32),
                    np.asarray(m.faces, dtype=np.int32),
                    target_count=goal, agg=7)
                m2 = trimesh.Trimesh(vertices=np.asarray(v, dtype=np.float64),
                                     faces=np.asarray(f, dtype=np.int64), process=False)
            else:
                m2 = m.simplify_quadric_decimation(face_count=goal)
            m2.update_faces(m2.nondegenerate_faces())
            m2.remove_unreferenced_vertices()
            if len(m2.faces) >= 4:
                out.append(Part(m2, pt.tile))
                continue
        except Exception:
            pass
        out.append(pt)
    return out, total, True


# ----------------------------------------------------------------------------
# 6. export
# ----------------------------------------------------------------------------

def finalize(mesh, uv, p, model_dir, tex_size=1024):
    color_img, mr_img, nrm_img = paint_atlas(p, size=tex_size)
    material = PBRMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        baseColorTexture=color_img,
        metallicRoughnessTexture=mr_img,
        normalTexture=nrm_img,
        metallicFactor=1.0,
        roughnessFactor=1.0,
    )
    mesh.visual = TextureVisuals(uv=uv, material=material)

    glb = mesh.export(file_type="glb")
    with open(f"{model_dir}/model.glb", "wb") as f:
        f.write(glb)
    stl = mesh.export(file_type="stl")
    with open(f"{model_dir}/model.stl", "wb") as f:
        f.write(stl)
    try:
        trimesh.exchange.export.export_mesh(mesh, f"{model_dir}/model.obj")
    except Exception:
        mesh.export(file_obj=f"{model_dir}/model.obj", file_type="obj")
    color_img.save(f"{model_dir}/texture_atlas.png")
    mr_img.save(f"{model_dir}/mr_atlas.png")
    nrm_img.save(f"{model_dir}/normal_atlas.png")
    return color_img


def swatch(rgb):
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


# ----------------------------------------------------------------------------
# full pipeline
# ----------------------------------------------------------------------------

def run_pipeline(source="text", text=None, image_bytes=None,
                 budget="game", tex_size=1024, model_dir=".", prompt=""):
    t0 = time.time()
    stages = []

    def done(name):
        stages.append({"name": name, "ms": int((time.time() - t0) * 1000)})

    if source == "image":
        params = analyze_image(image_bytes)
    else:
        params = analyze_text(text)
    done("Analyze input")

    b = BUDGETS[budget]
    parts, geo_notes = build_character(params, b)
    tris_before = sum(len(pt.mesh.faces) for pt in parts)
    done("Reconstruct geometry")

    parts, _, decimated = optimize_parts(parts, b["tris"])
    mesh, uv = assemble(parts)
    done("Optimize topology")

    color_img = finalize(mesh, uv, params, model_dir, tex_size=tex_size)
    done("Unwrap UVs + bake textures")
    done("Export GLB / OBJ / STL")

    report = {
        "stages": stages,
        "source": source,
        "prompt": prompt,
        "style": params["style"],
        "tags": params.get("tags", []),
        "notes": geo_notes + params.get("source_notes", []),
        "palette": {k: swatch(params[k]) for k in
                    ("skin", "hair", "garment_a", "garment_b", "pants",
                     "boots", "leather", "metal", "accent")},
        "accessories": sorted(params["accessories"]),
        "geometry": {
            "vertices": int(len(mesh.vertices)),
            "triangles": int(len(mesh.faces)),
            "triangles_before_optimize": int(tris_before),
            "decimated": bool(decimated),
            "materials": 1,
            "texture_size": tex_size,
            "maps": ["baseColor", "metallicRoughness", "normal"],
        },
        "height_m": round(float(params["height"]), 2),
    }
    return report
