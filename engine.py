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
import json
import re
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import trimesh

try:                                    # the guard is optional: CLI runs on a laptop too
    import memguard
except Exception:                       # noqa: BLE001
    memguard = None

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

HEADS_RANGE = {"chibi": (2.4, 4.6), "stylized": (3.8, 5.6), "realistic": (5.4, 7.6)}

STYLE_PROPS = {
    "chibi":     {"heads": 3.1, "build": 1.08, "arm_len": 0.66, "arm_r": 0.42, "leg_r": 0.52},
    "stylized":  {"heads": 4.6, "build": 1.00, "arm_len": 0.90, "arm_r": 0.36, "leg_r": 0.46},
    "realistic": {"heads": 6.4, "build": 0.94, "arm_len": 1.02, "arm_r": 0.33, "leg_r": 0.44},
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
        accessories=set(), tags=[], heads=None, detail=None, materials={},
    )


def _has_word(t, word):
    """Whole-word match with simple plurals/verb forms ('wing' != 'glowing')."""
    return re.search(r"\b" + re.escape(word) + r"(s|es|ed|ing|ic|ish|y)?\b", t) is not None


def _set_color(p, target_key, rgb):
    p[target_key] = tuple(int(c) for c in rgb)


def analyze_text(text):
    p = default_params()
    t = " " + (text or "").lower() + " "
    tags = p["tags"]

    def has(*words):
        return any(_has_word(t, w) for w in words)

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
    if has("wing", "angel", "fairy", "valkyrie", "seraph"):
        p["accessories"].add("wings"); tags.append("wings")
    if has("hood", "hoodie", "rogue", "assassin", "ranger"):
        p["accessories"].add("hood"); tags.append("hood")
    if has("crown", "tiara", "king", "queen", "royal", "princess", "prince"):
        p["accessories"].add("crown"); tags.append("crown")
    if has("scarf", "bandana", "muffler", "sash", "neck wrap"):
        p["accessories"].add("scarf"); tags.append("scarf")
    if has("skirt", "dress", "gown", "tutu"):
        p["accessories"].add("skirt"); tags.append("skirt")
    if has("shoulder pad", "pauldron", "spaulder", "armor", "armour", "plate",
           "knight", "paladin"):
        p["accessories"].add("shoulder_pads"); tags.append("shoulder pads")
    if has("chest plate", "breastplate", "armor", "armour", "plate", "knight", "paladin"):
        p["accessories"].add("armor_plates"); tags.append("armor plates")
    if has("robot", "android", "mech", "cyborg", "droid", "automaton", "machine"):
        p["accessories"].add("robot_joints"); tags.append("robot joints")
    if has("tall boots", "thigh boots", "knee boots", "jackboots", "greaves"):
        p["accessories"].add("boots_tall"); tags.append("tall boots")
    if has("fluffy tail", "fox", "wolf", "kitsune", "furry"):
        p["accessories"].add("tail_fluffy"); tags.append("fluffy tail")
    if has("long hair", "ponytail", "braid", "mane", "long-haired", "hime cut"):
        p["accessories"].add("hair_long"); tags.append("long hair")

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
        "wings": "garment_b", "skirt": "garment_b", "scarf": "accent",
        "crown": "metal", "armor": "metal", "armour": "metal",
        "pauldron": "metal", "shoulder": "metal", "tail": "hair",
        "fluff": "hair", "mane": "hair", "joints": "metal",
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
                before = t[:i].split()[-4:]  # "the cape should be blue"
                key = None
                for w in after[:2]:                    # "red robe"
                    w = w.strip(".,!?;:")
                    if w in targets:
                        key = targets[w]
                        break
                if key is None:                        # "make the robe red"
                    for w in reversed(before):
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
    p["build"] = max(0.72, min(1.45, p["build"]))
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


def _open_image(png_bytes):
    """Decode defensively: bad bytes must read as a 400, not a stack trace."""
    try:
        img = Image.open(io.BytesIO(png_bytes))
        img.load()
    except Exception as e:                                   # noqa: BLE001
        raise ValueError("That file could not be read as an image "
                         f"({type(e).__name__}: {str(e)[:90]}). PNG, JPG or WEBP please.") from e
    return img


def analyze_image(png_bytes):
    """Isolate the subject and measure silhouette, proportions and palette.

    Decoding is done at working resolution on purpose: `Image.draft` asks libjpeg
    to downscale *while* decoding, so a 12 MP phone photo costs ~2 MB here instead
    of the ~48 MB (x3 copies) that OOM-killed the free Render instance.
    """
    img = _open_image(png_bytes)
    if getattr(img, "format", "") == "JPEG":
        img.draft("RGB", (768, 768))
    if img.size[0] * img.size[1] > 60_000_000:
        raise ValueError("image is too large (over 60 MP) - please crop it first")
    img = img.convert("RGBA")
    img.thumbnail((768, 768))
    arr = np.array(img)
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3].astype(np.int32)

    if int(alpha.min()) < 200 and int(alpha.max()) > 30:
        mask = alpha > 60
        seg_note = "Subject isolated from PNG alpha channel"
    else:
        # Per-row background LUT from the right margin + per-column LUT from the
        # bottom margin.  Handles gradient skies / rocky grounds, unlike a single
        # median border colour (which a bright sky + dark character would defeat).
        h, w, _ = rgb.shape
        row_bg = np.median(rgb[:, int(w * 0.93):, :], axis=1)        # (h, 3)
        col_bg = np.median(rgb[int(h * 0.93):, :, :], axis=0)        # (w, 3)
        dr = np.abs(rgb - row_bg[:, None, :]).max(axis=2)
        dc = np.abs(rgb - col_bg[None, :, :]).max(axis=2)
        fg_guess = (dr > 30) & (dc > 30)
        mask = ~_background_mask(fg_guess)
        # clean the mask: opening strips thin junk (HUD text, cable lines) that
        # the flood picks up, then keep the largest blob and shrink it slightly
        # so band crops stay inside the subject
        try:
            from scipy import ndimage
            mask = ndimage.binary_opening(mask, iterations=4)
            lab, ncomp = ndimage.label(mask)
            if ncomp > 1:
                sizes = ndimage.sum(np.ones_like(lab), lab, range(1, ncomp + 1))
                mask = lab == (int(np.argmax(sizes)) + 1)
            mask = ndimage.binary_erosion(mask, iterations=1)
        except Exception:                                          # noqa: BLE001
            pass
        seg_note = "Subject isolated by margin-LUT segmentation + flood"

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
    p["_sampled"] = set()
    hair = crown[0] if crown else None
    skin = next((c for c in face if skinness(c) and (hair is None or dist(c, hair) > 30)), None)
    if skin is None and face:
        skin = next((c for c in face if hair is None or dist(c, hair) > 30), None)
    if skin:
        p["skin"] = skin; p["_sampled"].add("skin")
        p["source_notes"].append("Skin tone sampled from face region")
    else:
        p["source_notes"].append("No skin tone found - using default")
    if hair:
        p["hair"] = hair; p["_sampled"].add("hair")
        p["source_notes"].append("Hair colour sampled from crown of head")
    if mid:
        p["garment_a"] = mid[0]; p["_sampled"].add("garment_a")
        p["source_notes"].append("Primary garment colour sampled from torso")
        second = next((c for c in mid[1:] if dist(c, mid[0]) > 40 and
                       (skin is None or dist(c, skin) > 30)), None)
        p["garment_b"] = second or tuple(int(min(255, v * 1.15 + 20)) for v in mid[0])
        if second:
            p["_sampled"].add("garment_b")
    if low:
        p["pants"] = low[0]; p["_sampled"].add("pants")
        p["source_notes"].append("Lower garment colour sampled from legs")
    if feet:
        p["boots"] = feet[0]; p["_sampled"].add("boots")
    top = crown + face

    if heads_tall < 3.9:
        p["style"] = "chibi"
    elif heads_tall > 5.8:
        p["style"] = "realistic"
    p["build"] = float(np.clip(0.55 + shoulder_w * 1.9, 0.82, 1.3))
    p["height"] = float(np.clip(1.25 + heads_tall * 0.085, 1.3, 1.9))
    p["_sampled"].update({"build", "height", "style"})
    p["tags"] = [f"{heads_tall:.1f} heads tall", f"silhouette aspect {aspect:.2f}",
                 f"{p['style']} proportions"]
    # kept for the mode router: is this a standing character, or "some object"?
    p["_heads_tall"] = float(heads_tall)
    p["_aspect"] = float(aspect)
    p["_skin_found"] = bool(skin)

    def greyish(c):
        r, g, b = c
        return max(r, g, b) - min(r, g, b) < 22 and 90 < min(r, g, b) < 235
    if mid and greyish(mid[0]):
        p["tags"].append("metallic surfaces detected")

    p["palette_read"] = {"top": top, "mid": mid, "low": low, "skin": skin, "hair": hair}
    # keep the subject region for atlas painting (the character wears the art)
    p["_img"] = {
        "img": img,
        "box": (int(x0), int(y0), int(x1) + 1, int(y1) + 1),
        "head_frac": float(head_frac),
        "torso_frac": float(torso_end),
        "mask": mask,
    }
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
    heads = p.get("heads") or sp["heads"]
    build = p["build"] * sp["build"]
    head_r = H / (2 * heads) * p.get("head_boost", 1.0)
    head_c = H - head_r * 1.02
    neck_y = H - head_r * 1.95
    hip_y = H * 0.47
    torso_len = neck_y - hip_y
    torso_mid = (neck_y + hip_y) / 2
    torso_r = H * 0.085 * build
    sh_x = torso_r * 1.35 * build                # shoulder x-offset
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

    hair = S(subdivisions=sub, radius=head_r * 1.07)
    hair.apply_transform(np.diag([1.0, 0.92, 1.0, 1.0]))
    # keep the top + back (plane normal points up and backwards), so the face stays clear
    hair = trimesh.intersections.slice_mesh_plane(
        hair, plane_normal=[0.0, 0.72, -0.69], plane_origin=[0, head_r * 0.42, head_r * 0.12])
    fringe = trimesh.intersections.slice_mesh_plane(
        trimesh.creation.icosphere(subdivisions=max(1, sub - 1), radius=head_r * 1.05),
        plane_normal=[0.0, 1.0, 0.0], plane_origin=[0, head_r * 0.62, 0])
    fringe = trimesh.intersections.slice_mesh_plane(
        fringe, plane_normal=[0.0, -0.55, 1.0], plane_origin=[0, head_r * 0.72, head_r * 0.62])
    hair = trimesh.util.concatenate([hair, fringe])
    hair.apply_translation([0, head_c, 0])
    if len(hair.faces) > 4:
        add(hair, TILE["hair"])

    for s in (-1, 1):
        e = S(subdivisions=max(1, sub - 1), radius=head_r * 0.155)
        e.apply_transform(np.diag([1.0, 1.15, 0.7, 1.0]))
        e.apply_translation([s * head_r * 0.34, head_c - head_r * 0.05, head_r * 0.83])
        add(e, TILE["eye"])

    # face: a small nose + brows give the head real features instead of a blank orb
    nose = S(subdivisions=max(1, sub - 1), radius=head_r * 0.10)
    nose.apply_transform(np.diag([0.75, 1.0, 1.15, 1.0]))
    nose.apply_translation([0, head_c - head_r * 0.15, head_r * 0.92])
    add(nose, TILE["skin"])
    for s in (-1, 1):
        brow = BOX(extents=[head_r * 0.34, head_r * 0.05, head_r * 0.07])
        brow.apply_transform(_rot(-s * 10, [0, 0, 1]))
        brow.apply_translation([s * head_r * 0.32, head_c + head_r * 0.21, head_r * 0.84])
        add(brow, TILE["hair"])

    # anime side-locks framing the face
    for s in (-1, 1):
        lock = cap_y(head_r * 0.72, head_r * 0.15, count=(3, 8))
        lock.apply_transform(_rot(s * 7, [0, 0, 1]))
        lock.apply_translation([s * head_r * 0.97, head_c - head_r * 0.52, head_r * 0.36])
        add(lock, TILE["hair"])

    # ---- neck / torso / hips ------------------------------------------------
    neck = cyl_y(head_r * 0.32, head_r * 0.6, sections=sec)
    neck.apply_translation([0, neck_y + head_r * 0.1, 0])
    add(neck, TILE["skin"])

    torso = cap_y(torso_len * 0.82, torso_r, count=cnt)
    torso.apply_transform(np.diag([1.35, 1.0, 0.85, 1.0]))
    torso.apply_translation([0, torso_mid + torso_len * 0.06, 0])
    add(torso, TILE["garment_a"])

    belt = cyl_y(torso_r * 1.18, hip_y * 0.17, sections=sec)
    belt.apply_transform(np.diag([1.22, 1.0, 0.92, 1.0]))
    belt.apply_translation([0, hip_y * 1.01, 0])
    add(belt, TILE["leather"])

    # pelvis blends the torso into the legs (kills the cylinder/capsule seam)
    pelvis = S(subdivisions=max(1, sub - 1), radius=torso_r * 0.95)
    pelvis.apply_transform(np.diag([1.3 * build, 0.62, 0.88, 1.0]))
    pelvis.apply_translation([0, hip_y * 0.97, 0])
    add(pelvis, TILE["pants"])

    # ---- arms (A-pose, ~24 degrees from the body) ----------------------------
    arm_len = torso_len * 1.05 * sp["arm_len"]
    arm_r = torso_r * sp.get("arm_r", 0.36)
    ang = 24.0
    for s in (-1, 1):
        shoulder = np.array([s * sh_x * 1.1, neck_y - head_r * 0.25, 0.0])
        sh_blend = S(subdivisions=max(1, sub - 1), radius=arm_r * 1.35)
        sh_blend.apply_translation(shoulder)
        add(sh_blend, TILE["garment_a"])        # blends arm into torso
        arm = cap_y(arm_len, arm_r, count=cnt)
        arm.apply_translation(shoulder - [0, arm_len / 2, 0])
        arm.apply_transform(_rot(s * ang, [0, 0, 1], point=shoulder))
        add(arm, TILE["garment_b"])
        dirv = np.array([s * np.sin(np.radians(ang)), -np.cos(np.radians(ang)), 0.0])
        hand = S(subdivisions=max(1, sub - 1), radius=arm_r * 1.2)
        hand.apply_translation(shoulder + dirv * (arm_len + arm_r * 0.9))
        add(hand, TILE["skin"])

    # ---- legs / boots ---------------------------------------------------------
    leg_r = torso_r * sp.get("leg_r", 0.46)
    leg_len = hip_y * 0.86
    for s in (-1, 1):
        x = s * torso_r * 0.55
        leg = cap_y(leg_len, leg_r, count=cnt)
        leg.apply_translation([x, hip_y * 0.05 + leg_len / 2, 0])
        add(leg, TILE["pants"])
        boot = cyl_y(leg_r * 1.15, hip_y * 0.19, sections=sec)
        boot.apply_transform(np.diag([1.0, 1.0, 1.2, 1.0]))
        boot.apply_translation([x, hip_y * 0.10, leg_r * 0.10])
        add(boot, TILE["boots"])
        ankle = S(subdivisions=max(1, sub - 1), radius=leg_r * 1.18)
        ankle.apply_translation([x, hip_y * 0.30, leg_r * 0.05])
        add(ankle, TILE["boots"])               # smooths the leg-to-boot step
        foot = BOX(extents=[leg_r * 2.05, leg_r * 1.05, leg_r * 3.4])
        foot.apply_translation([x, leg_r * 0.55, leg_r * 0.85])
        add(foot, TILE["boots"])

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

    # ---- features the AI director can request -------------------------------
    if "hood" in acc:
        hood = S(subdivisions=sub, radius=head_r * 1.22)
        hood = trimesh.intersections.slice_mesh_plane(
            hood, plane_normal=[0, 1, 0], plane_origin=[0, -head_r * 0.2, 0])
        back = trimesh.intersections.slice_mesh_plane(
            hood, plane_normal=[0, 0, 1], plane_origin=[0, 0, head_r * 0.35])
        hood = trimesh.util.concatenate([hood, back]) if len(back.faces) > 3 else hood
        hood.apply_translation([0, head_c + head_r * 0.02, -head_r * 0.08])
        add(hood, TILE["garment_a"])
    if "hair_long" in acc:
        mane = cap_y(head_r * 1.7, head_r * 0.55, count=cnt)
        mane.apply_transform(np.diag([0.75, 1.0, 0.6, 1.0]))
        mane.apply_translation([0, head_c - head_r * 0.55, -head_r * 0.35])
        add(mane, TILE["hair"])
    if "crown" in acc:
        ring = trimesh.creation.annulus(r_min=head_r * 0.86, r_max=head_r * 1.02,
                                        height=head_r * 0.18)
        ring.apply_translation([0, head_c + head_r * 0.55, 0])
        add(ring, TILE["metal"])
        for k in range(5):
            spike = cone_y(head_r * 0.12, head_r * 0.32, sections=max(6, sec // 2))
            a = np.radians(-70 + k * 35)
            spike.apply_translation([np.sin(a) * head_r * 0.94, head_c + head_r * 0.66,
                                     np.cos(a) * head_r * 0.94])
            add(spike, TILE["accent"])
    if "scarf" in acc:
        sc = trimesh.creation.annulus(r_min=torso_r * 0.75, r_max=torso_r * 1.12,
                                      height=head_r * 0.34)
        sc.apply_translation([0, neck_y + head_r * 0.05, 0])
        add(sc, TILE["accent"])
        tail_sc = BOX(extents=[torso_r * 0.5, torso_len * 0.55, torso_r * 0.16])
        tail_sc.apply_translation([torso_r * 0.3, torso_mid + torso_len * 0.2, torso_r * 0.75])
        add(tail_sc, TILE["accent"])
    if "skirt" in acc:
        sk = cone_y(torso_r * 1.6, torso_len * 0.75, sections=sec)
        sk.apply_transform(_rot(180, [1, 0, 0]))
        sk.apply_translation([0, hip_y + torso_len * 0.16, 0])
        add(sk, TILE["garment_b"])
    if "wings" in acc:
        for s in (-1, 1):
            wing = BOX(extents=[torso_r * 2.4, H * 0.30, torso_r * 0.045])
            wing.apply_transform(_rot(-s * 22, [0, 0, 1]))
            wing.apply_transform(_rot(-s * 12, [1, 0, 0]))
            wing.apply_translation([s * torso_r * 1.95, torso_mid + torso_len * 0.30, back_z * 1.35])
            add(wing, TILE["garment_b"])
    if "shoulder_pads" in acc:
        for s in (-1, 1):
            pad = S(subdivisions=max(1, sub - 1), radius=torso_r * 0.62)
            pad.apply_transform(np.diag([1.15, 0.72, 1.15, 1.0]))
            pad.apply_translation([s * sh_x * 1.15, neck_y - head_r * 0.2, 0])
            add(pad, TILE["metal"])
    if "armor_plates" in acc:
        chest = BOX(extents=[torso_r * 1.55, torso_len * 0.34, torso_r * 0.14])
        chest.apply_transform(np.diag([1.0, 1.0, 1.0, 1.0]))
        chest.apply_translation([0, torso_mid + torso_len * 0.18, torso_r * 0.62])
        add(chest, TILE["metal"])
        for s in (-1, 1):
            thigh = BOX(extents=[torso_r * 0.85, hip_y * 0.3, torso_r * 0.5])
            thigh.apply_translation([s * torso_r * 0.55, hip_y * 0.62, torso_r * 0.28])
            add(thigh, TILE["metal"])
    if "robot_joints" in acc:
        for s in (-1, 1):
            for y, xo in ((neck_y - head_r * 0.25, sh_x * 1.1),
                          (hip_y * 0.05 + leg_len * 0.5, torso_r * 0.55)):
                j = S(subdivisions=max(1, sub - 1), radius=torso_r * 0.3)
                j.apply_translation([s * xo, y, 0])
                add(j, TILE["metal"])
    if "boots_tall" in acc:
        for s in (-1, 1):
            sh = cyl_y(leg_r * 1.28, hip_y * 0.46, sections=sec)
            sh.apply_translation([s * torso_r * 0.55, hip_y * 0.26, 0])
            add(sh, TILE["boots"])
    if "tail_fluffy" in acc:
        seg = cap_y(head_r * 1.5, head_r * 0.3, count=cnt)
        seg.apply_transform(_rot(-60, [1, 0, 0]))
        seg.apply_translation([0, hip_y * 1.05, back_z * 1.1])
        add(seg, TILE["hair"])

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


def _skin_score(col_med):
    """Per-column 0..1 face likelihood from the column median colours.
    Rewards skin tone, penalises hair (too dark) and sky (too bright)."""
    w = np.zeros(col_med.shape[0])
    for i, c in enumerate(col_med):
        r, g, b = float(c[0]), float(c[1]), float(c[2])
        if r >= 60 and r >= g >= b and 15 < (r - b) < 140:
            w[i] = 1.0
    lum = col_med.mean(axis=1)
    w *= np.clip((lum - 95) / 55, 0, 1)          # hair columns are darker than this
    w *= 1.0 - np.clip((lum - 205) / 45, 0, 1)   # sky columns are brighter than this
    k = max(3, w.size // 12)
    sm = np.convolve(w, np.ones(k) / k, mode="same")
    centred = np.exp(-(((np.arange(w.size) - w.size / 2) / w.size) ** 2) * 1.5)
    return sm * (0.55 + 0.45 * centred)


def _paint_image_tiles(color, info, n):
    """Sample the concept art onto the per-part tiles, so the built character
    wears the reference design.  Returns True when every key tile was painted.

    Box-projection UVs (see _part_uv) map the tile's *columns* to the part's
    longest axis and its *rows* to the second longest — for the head that is
    char-Y / char-X, for the torso and legs char-Y / char-X as well.  So
    vertical body bands are transposed before pasting; the face is located by
    skin colour inside its band (the character may be off-centre / turned) and
    pasted into the head's face rectangle, which is the skin the hair shell
    does not cover."""
    try:
        img = info["img"].convert("RGB")
        box = info["box"]
        hf, te = float(info["head_frac"]), float(info["torso_frac"])
        x0, y0, x1, y1 = box
        W, H = x1 - x0, y1 - y0
        if W < 60 or H < 120:
            return False
        mask = info.get("mask")

        def band(f0, f1, wfrac=1.0, min_h=16):
            """(image, mask) of a horizontal band; non-subject pixels are
            forward-filled with their column's subject median."""
            a = y0 + int(H * f0)
            b = max(a + min_h, y0 + int(H * f1))
            cw = max(16, int(W * wfrac))
            cx = x0 + (W - cw) // 2
            crop = img.crop((cx, a, cx + cw, b))
            arr = np.array(crop, dtype=np.float32)
            mb = None
            if mask is not None:
                mb = np.asarray(mask)[a:b, cx:cx + cw]
                if mb.shape != arr.shape[:2]:
                    mb = np.array(Image.fromarray(mb.astype(np.uint8) * 255)
                                  .resize((arr.shape[1], arr.shape[0]), Image.NEAREST)) > 127
                if mb.any():
                    fallback = np.median(arr[mb], axis=0)
                    for xi in range(arr.shape[1]):
                        if not mb[:, xi].any():
                            arr[:, xi] = fallback
                        elif not mb[:, xi].all():
                            arr[~mb[:, xi], xi] = np.median(arr[mb[:, xi], xi], axis=0)
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), (mb is not None)

        def tile_xy(i):
            tx, ty = i % ATLAS_COLS, i // ATLAS_COLS
            return (ATLAS_ROWS - 1 - ty) * n, tx * n

        def put_full(tile, crop, transpose=True):
            y, x = tile_xy(tile)
            arr = np.array(crop, dtype=np.float32)
            if transpose:
                arr = arr.transpose(1, 0, 2)     # tile cols = char-Y = band rows
            c = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            c = c.resize((n, n), Image.BICUBIC)
            color[y:y + n, x:x + n] = np.array(c, dtype=np.float32)

        # face: localise the 2-D skin blob in the head region (works for
        # off-centre / turned characters), stand it upright and paste it over
        # the head's face rectangle
        face_img = None
        try:
            from scipy import ndimage
            head_h = max(24, int(H * (hf + 0.05)))
            head_crop = img.crop((x0, y0, x1, min(y1, y0 + head_h)))
            hc = np.array(head_crop, dtype=np.float32)
            r, g, b = hc[..., 0], hc[..., 1], hc[..., 2]
            # warm saturated skin (tan rock reads greyer / less red than face skin)
            skin = (((r >= 90) & (r < 245) & (r >= g) & (g >= b) &
                     (r - b > 35) & (r - b < 150))).astype(np.float32)
            if mask is not None:
                m_h = np.asarray(mask)[y0:min(y1, y0 + head_h), x0:x1]
                if m_h.shape == skin.shape:
                    skin *= m_h
            skin = ndimage.gaussian_filter(skin, 4)
            h_h, w_h = skin.shape
            skin *= np.exp(-(((np.arange(w_h) - w_h / 2) / w_h) ** 2) * 0.8)[None, :]
            blob = skin > 0.5
            picked = None
            if blob.sum() > 60:
                lab, nc = ndimage.label(blob)
                cents = ndimage.center_of_mass(blob, lab, range(1, nc + 1))
                sizes = ndimage.sum(blob, lab, range(1, nc + 1))
                # the face sits in the lower two thirds of the head crop — crown
                # blobs (bright hair highlights, rock) are rejected outright
                cands = [i for i in range(nc) if cents[i][0] > h_h * 0.42]
                if cands:
                    picked = max(cands, key=lambda i: sizes[i])
                elif max(c[0] for c in cents) > h_h * 0.30:   # nothing lower; take the best
                    picked = max(range(nc), key=lambda i: sizes[i])
            if picked is not None:
                blob = lab == (picked + 1)
                ys2, xs2 = np.where(blob)
                fh, fw = ys2.max() - ys2.min() + 1, xs2.max() - xs2.min() + 1
                # face-likeness gate: face skin is uniform and compact; background
                # that leaks in through the hair (cliff, sky) is speckled/varied
                px = hc[ys2, xs2]
                var = float(np.linalg.norm(px - px.mean(axis=0), axis=1).mean())
                fill = float(sizes[picked]) / (fh * fw + 1e-9)
                edge = (xs2.max() >= w_h - 3) or (ys2.min() < 3)
                if fh >= 16 and fw >= 16 and var < 58 and fill > 0.22 and not edge:
                    # grow a little for the eye line, but never into the crown
                    # zone (the hair shell covers the top of the head on the model)
                    gy0 = max(int(h_h * 0.30), int(ys2.min() - min(fh * 0.25, 20)))
                    gy1 = min(head_crop.size[1], int(ys2.max() + 1 + fh * 0.20))
                    gx0 = max(0, int(xs2.min() - fw * 0.12))
                    gx1 = min(head_crop.size[0], int(xs2.max() + 1 + fw * 0.15))
                    face_img = head_crop.crop((gx0, gy0, gx1, gy1))
        except Exception:                                                  # noqa: BLE001
            face_img = None
        if face_img is not None:      # messy head region -> keep procedural skin
            face = np.flipud(np.array(face_img, dtype=np.float32)).transpose(1, 0, 2)
            face = Image.fromarray(np.clip(face, 0, 255).astype(np.uint8))
            ry, rx = int(0.225 * n), int(0.15 * n)
            face = face.resize((int(0.61 * n), int(0.55 * n)), Image.BICUBIC)
            y, x = tile_xy(TILE["skin"])
            fy0, fx0 = y + ry, x + rx
            color[fy0:fy0 + face.size[1], fx0:fx0 + face.size[0]] = np.array(face, np.float32)
            info["_face_pasted"] = True

        put_full(TILE["hair"], band(0.0, hf * 0.6, wfrac=1.0)[0], transpose=False)
        put_full(TILE["garment_a"], band(hf, te * 0.85, wfrac=0.72)[0])
        put_full(TILE["garment_b"], band(te * 0.6, 0.92, wfrac=0.72)[0])
        put_full(TILE["pants"], band(te, 0.87, wfrac=0.6)[0])
        put_full(TILE["boots"], band(0.84, 1.0, wfrac=0.5)[0])
        return True
    except Exception:                                          # noqa: BLE001
        return False


def _blush(color, n, strength):
    """Soft cheek blush on the skin tile (text builds only — image builds get
    the face straight from the art).  Head uv: image col = char-Y, row = char-X."""
    i = TILE["skin"]
    tx, ty = i % ATLAS_COLS, i // ATLAS_COLS
    y0, x0 = (ATLAS_ROWS - 1 - ty) * n, tx * n
    tint = np.array([255, 132, 132], float)

    def soft(cx, cy, rx, ry, amt):
        ys, xs = np.ogrid[y0:y0 + n, x0:x0 + n]
        d = ((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2
        w = np.clip(1 - d, 0, None) ** 2 * amt
        region = color[y0:y0 + n, x0:x0 + n]
        color[y0:y0 + n, x0:x0 + n] = region + (tint - region) * w[:, :, None]

    soft(x0 + 0.42 * n, y0 + 0.725 * n, 0.11 * n, 0.10 * n, strength)
    soft(x0 + 0.42 * n, y0 + 0.275 * n, 0.11 * n, 0.10 * n, strength)


def paint_atlas(p, size=1024):
    """Base color + metallic-roughness + normal atlases.

    With image input (p["_img"]) the per-part tiles are sampled from the
    concept art itself; text-only builds get procedural per-material detail."""
    n = size // ATLAS_COLS
    rng = np.random.default_rng(42)

    # float32 throughout: a 2048**2 float64 atlas plus its normal-map temporaries
    # would OOM the free Render instance (~512 MB)
    color = np.zeros((n * ATLAS_ROWS, n * ATLAS_COLS, 3), dtype=np.float32)
    mr = np.zeros((n * ATLAS_ROWS, n * ATLAS_COLS, 3), dtype=np.uint8)
    height = np.zeros((n * ATLAS_ROWS, n * ATLAS_COLS), dtype=np.float32)
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
        base = np.array(palette[name], dtype=np.float32)
        shade = 1.0 + h * 0.06
        color[y0:y0 + n, x0:x0 + n] = base[None, None, :] * shade[:, :, None]
        mr[y0:y0 + n, x0:x0 + n, 1] = (rough * 255).astype(np.uint8)
        mr[y0:y0 + n, x0:x0 + n, 2] = int(metallic.get(name, 0.04) * 255)
        height[y0:y0 + n, x0:x0 + n] = h

    if p.get("_img") and _paint_image_tiles(color, p["_img"], n):
        p["_img_used"] = True
    elif p.get("style") in ("chibi", "stylized"):
        _blush(color, n, 0.40 if p.get("style") == "chibi" else 0.22)

    color_img = Image.fromarray(np.clip(color, 0, 255).astype(np.uint8), "RGB") \
        .filter(ImageFilter.GaussianBlur(0.6))

    # normal map from a *smoothed* height field — raw Sobel of the grainy detail
    # read as noisy skin on curved parts; low-frequency undulation looks far cleaner
    try:
        from scipy import ndimage
        height = ndimage.gaussian_filter(height, sigma=max(1.0, n / 48.0))
    except Exception:                                          # noqa: BLE001
        pass
    dy, dx = np.gradient(height)
    strength = 1.5
    nrm = np.empty_like(color, dtype=np.uint8)                # no float stack: OOM-safe
    inv = 1.0 / np.sqrt((dx * strength) ** 2 + (dy * strength) ** 2 + 1.0)
    nrm[..., 0] = ((-dx * strength * inv) * 0.5 + 0.5) * 255
    nrm[..., 1] = ((-dy * strength * inv) * 0.5 + 0.5) * 255
    nrm[..., 2] = (inv * 0.5 + 0.5) * 255
    nrm_img = Image.fromarray(nrm, "RGB")

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

# ---------------------------------------------------------------------------
# build plan: the decision layer (what the AI director produces)
# ---------------------------------------------------------------------------

FEATURES = ["backpack", "wizard_hat", "cape", "sword", "staff", "shield", "horns",
            "cat_ears", "tail", "glasses", "hood", "hair_long", "crown", "scarf",
            "skirt", "wings", "shoulder_pads", "armor_plates", "robot_joints",
            "boots_tall", "tail_fluffy"]
PLAN_SLOTS = ("skin", "hair", "garment_a", "garment_b", "pants", "boots",
              "leather", "metal", "accent", "eye")


def _as_rgb(v):
    """Accept '#rrggbb' strings or (r, g, b) sequences."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lstrip("#")
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        try:
            return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return None
    try:
        return tuple(int(c) for c in list(v)[:3])
    except (TypeError, ValueError):
        return None


def plan_to_params(params, plan):
    """Fold a build plan into engine params. Plan decides; heuristics fill gaps."""
    p = dict(params)
    p["accessories"] = {f for f in (plan.get("features") or []) if f in FEATURES}
    for slot in PLAN_SLOTS:
        rgb = _as_rgb((plan.get("palette") or {}).get(slot))
        if rgb:
            p[slot] = rgb
    if plan.get("style") in STYLE_PROPS:
        p["style"] = plan["style"]
    for key, src in (("height", "height_m"), ("build", "build"), ("heads", "heads_tall")):
        if plan.get(src):
            p[key] = float(plan[src])
    head_boost = plan.get("head_boost")
    if head_boost:
        p["head_boost"] = max(0.6, min(1.8, float(head_boost)))
    p["materials"] = dict(plan.get("materials") or {})
    p["plan"] = plan
    tags = [t for t in (params.get("tags") or [])]
    if plan.get("species"):
        tags = [str(plan["species"])[:40]] + [t for t in tags if t != plan["species"]]
    p["tags"] = tags[:8]
    return p


FEATURE_SYNONYMS = {
    "cape": {"cape", "cloak", "mantle", "tabard", "drape"},
    "wizard_hat": {"hat", "wizard", "witch", "brim", "pointy"},
    "staff": {"staff", "stave", "rod", "shaft", "crystal", "mace", "scepter", "wand"},
    "sword": {"sword", "blade", "hilt", "katana", "scabbard", "dagger"},
    "shield": {"shield", "buckler", "aegis"},
    "backpack": {"backpack", "bag", "pack", "rucksack", "satchel"},
    "wings": {"wing", "wings", "feather"},
    "tail": {"tail"},
    "tail_fluffy": {"tail", "fluff"},
    "horns": {"horn", "horns"},
    "cat_ears": {"ear", "ears"},
    "hair_long": {"hair", "ponytail", "braid", "mane"},
    "crown": {"crown", "tiara", "circlet"},
    "scarf": {"scarf", "bandana", "muffler"},
    "hood": {"hood"},
    "skirt": {"skirt", "dress", "gown"},
    "armor_plates": {"breastplate", "plate", "chestplate", "cuirass"},
    "shoulder_pads": {"pauldron", "shoulder"},
    "boots_tall": {"greave", "boot"},
    "robot_joints": {"joint", "servo", "actuator"},
}


def _duplicate_of(name, features):
    """The feature an extra part would duplicate (the model sometimes asks twice)."""
    tokens = set(re.findall(r"[a-z]+", str(name or "").lower()))
    for feat in features:
        words = FEATURE_SYNONYMS.get(feat, {feat.replace("_", " ")})
        if tokens & words:
            return feat
    return None


def build_extra_parts(plan, p, budget):
    """Primitive volumes the director asked for on top of the base body."""
    H = float(p["height"])
    sp = STYLE_PROPS[p["style"]]
    heads = p.get("heads") or sp["heads"]
    head_r = H / (2 * heads)
    def clamp_arr(v, lo, hi):
        try:
            out = [float(x) for x in v][:3]
        except Exception:                       # noqa: BLE001
            return None
        while len(out) < 3:
            out.append(0.0)
        return [max(lo, min(hi, x)) for x in out]

    parts, notes = [], []
    seen_names = set()
    for item in (plan.get("extra_parts") or [])[:8]:
        nm = str(item.get("name") or item.get("type") or "").lower().strip()
        if nm and nm in seen_names:
            continue
        seen_names.add(nm)
        kind = str(item.get("type", "")).lower()
        dup = _duplicate_of(item.get("name"), plan.get("features") or [])
        if dup:
            notes.append(f"AI part '{item.get('name')}' skipped - already modelled as '{dup}'")
            continue
        at = clamp_arr(item.get("at", [0, 0, 0]), -2.0, 3.0) or [0, 0, 0]
        size = clamp_arr(item.get("size", [0.2, 0.2, 0.2]), 0.02, 1.6) or [0.2, 0.2, 0.2]
        rot = clamp_arr(item.get("rot_deg", [0, 0, 0]), -180, 180) or [0, 0, 0]
        tile = TILE.get(str(item.get("material", "leather")), TILE["leather"])
        sec, sub, cnt = budget["cyl_sec"], budget["sphere_sub"], budget["cap_cnt"]
        sx, sy, sz = (max(0.02, s) for s in size)
        try:
            if kind == "box":
                m = trimesh.creation.box(extents=[sx, sy, sz])
            elif kind == "sheet" or kind == "plane":
                m = trimesh.creation.box(extents=[sx, sy, max(0.015, sz * 0.15)])
            elif kind == "sphere":
                m = trimesh.creation.icosphere(subdivisions=sub, radius=0.5)
                m.apply_scale([sx, sy, sz])
            elif kind in ("capsule", "cylinder", "cone"):
                r = min(sx, sz) / 2
                if kind == "capsule":
                    m = cap_y(max(0.01, sy - 2 * r), r, cnt)
                elif kind == "cylinder":
                    m = cyl_y(r, sy, sec)
                else:
                    m = cone_y(r, sy, sec)
                m.apply_scale([sx / (2 * r), 1.0, sz / (2 * r)])
            else:
                continue
            for axis, ang in zip(([1, 0, 0], [0, 1, 0], [0, 0, 1]), rot):
                if abs(ang) > 0.01:
                    m.apply_transform(_rot(ang, axis))
            m.apply_translation(at)
            parts.append(Part(m, tile))
            notes.append(f"AI part: {item.get('name') or kind} ({kind}, {str(item.get('material','leather'))})")
        except Exception as e:                  # noqa: BLE001
            notes.append(f"AI part '{item.get('name') or kind}' skipped ({type(e).__name__})")
    return parts, notes


# ---------------------------------------------------------------------------
# local (no-API) planning + revision of plans
# ---------------------------------------------------------------------------

def detail_for_target(tris):
    """Resolution knobs for a triangle budget (used when the AI does not set them)."""
    tris = int(tris or 5000)
    if tris <= 2500:
        return {"sphere_sub": 2, "cyl_sec": 8, "cap_cnt": 6}
    if tris <= 12000:
        return {"sphere_sub": 4, "cyl_sec": 16, "cap_cnt": 16}
    return {"sphere_sub": 4, "cyl_sec": 24, "cap_cnt": 20}


def local_plan(params, text, budget_hint=None):
    """Deterministic build plan used when no language model is available."""
    sp = STYLE_PROPS[params["style"]]
    heads = float(params.get("heads") or sp["heads"])
    tris = int(budget_hint or BUDGETS["game"]["tris"])
    if params["style"] == "chibi":
        tris = min(tris, 5000)
    detail = detail_for_target(tris)
    return {
        "brief": (text or "concept image").strip()[:200],
        "style": params["style"], "species": (params.get("tags") or ["character"])[0],
        "height_m": round(float(params["height"]), 3),
        "heads_tall": round(heads, 2), "build": round(float(params["build"]), 2),
        "head_boost": round(float(params.get("head_boost", 1.0)), 2),
        "target_tris": tris, "texture_size": 1024, "detail": detail,
        "palette": {s: swatch(params[s]) for s in PLAN_SLOTS},
        "features": sorted(params["accessories"] & set(FEATURES)),
        "materials": {}, "extra_parts": [],
        "back_view": "symmetric mirror of the front (backpacks, capes and straps modelled explicitly)",
        "modelling_plan": [
            f"body split into {heads:g}-heads-tall proportions from parametric volumes",
            "head/hair/eyes, torso, A-pose arms, legs and boots built as separate volumes",
            "each volume box-projected into a 4x4 atlas, then quadric-decimated to budget",
        ],
        "notes": ["planned by the built-in rule engine (no language model)"],
        "generator": "heuristic", "confidence": 0.6,
    }


def local_revise(plan, feedback):
    """Apply a plain-language change request to an existing plan without an LLM."""
    p = json.loads(json.dumps(plan))
    txt = (feedback or "").lower()
    diff = _param_diff(txt)
    sticky = []
    for k in ("skin", "hair", "garment_a", "garment_b", "pants", "boots",
              "leather", "metal", "accent"):
        if k in diff:
            p.setdefault("palette", {})[k] = swatch(diff[k])
            sticky.append(k)
    for f in diff.get("accessories", set()):
        if f in FEATURES and f not in p.setdefault("features", []):
            p["features"].append(f)
    for f in FEATURES:                                   # "remove the cape"
        if f.replace("_", " ") in txt and re_remove(txt, f) and f in p.get("features", []):
            p["features"].remove(f)
    if any(w in txt for w in ("taller", "bigger", "larger")):
        p["height_m"] = round(min(2.6, float(p.get("height_m", 1.7)) * 1.08), 3)
    if any(w in txt for w in ("shorter", "smaller", "tiny")):
        p["height_m"] = round(max(0.6, float(p.get("height_m", 1.7)) * 0.92), 3)
    if any(w in txt for w in ("heavier", "bulkier", "broader", "muscular")):
        p["build"] = round(min(1.4, float(p.get("build", 1.0)) * 1.15), 2)
    if any(w in txt for w in ("slimmer", "thinner", "leaner")):
        p["build"] = round(max(0.7, float(p.get("build", 1.0)) * 0.88), 2)
    if any(w in txt for w in ("chibi", "cuter", "bigger head", "cartoon")):
        p["style"], p["heads_tall"] = "chibi", 3.1
    if "realistic" in txt:
        p["style"], p["heads_tall"] = "realistic", 6.4
    m = re.search(r"(\d{3,6})\s*(?:tris|triangles|polys|polygons)", txt)
    if m:
        p["target_tris"] = max(600, min(40000, int(m.group(1))))
    elif any(w in txt for w in ("low poly", "low-poly", "mobile", "lighter", "fewer triangles")):
        p["target_tris"] = 1500
        p["detail"] = {"sphere_sub": 2, "cyl_sec": 8, "cap_cnt": 6}
    elif any(w in txt for w in ("high detail", "more detail", "high poly", "denser", "more triangles")):
        p["target_tris"] = 18000
        p["detail"] = {"sphere_sub": 4, "cyl_sec": 24, "cap_cnt": 20}
    p["detail"] = detail_for_target(p.get("target_tris"))
    p["notes"] = (["applied locally: " + (feedback or "")[:120]] +
                  [n for n in p.get("notes", []) if not str(n).startswith("applied locally:")])
    p["_sticky_colors"] = sticky
    return p


def re_remove(txt, feature):
    """True when the feedback asks to delete a feature ('remove the cape')."""
    f = feature.replace("_", " ")
    return any(w in txt for w in (f"remove the {f}", f"remove {f}", f"without the {f}",
                                  f"without {f}", f"no {f}", f"drop the {f}", f"drop {f}",
                                  f"lose the {f}", f"lose {f}"))


def _param_diff(txt):
    """Which params does this text actually change vs the defaults?"""
    got = analyze_text(txt)
    base = default_params()
    out = {}
    for k, v in got.items():
        if k in ("tags", "source_notes", "materials", "heads", "detail", "plan"):
            continue
        if isinstance(v, set):
            if v:
                out[k] = v
        elif v != base.get(k):
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------

def _safe_tex(want):
    """Largest atlas the instance can afford right now (see memguard)."""
    want = 1024 if want not in (512, 1024, 2048) else int(want)
    if memguard is None:
        return min(want, 1024)
    return memguard.tex_size(want)


def choose_mode(mode, source, params):
    """"auto" picks the reconstructor: a rig for a standing character, a real
    surface reconstruction for anything else. Both take the same image."""
    mode = (mode or "auto").lower()
    if mode in ("sculpt", "relief", "3d", "object"):
        return "sculpt"
    if mode in ("character", "rig", "avatar"):
        return "character"
    if source != "image" or not params:
        return "character"
    heads = float(params.get("_heads_tall") or 0)
    aspect = float(params.get("_aspect") or 1)
    # a head-to-toe figure: plausible head count, tall-and-narrow, and skin in it
    if 2.2 <= heads <= 9.5 and aspect <= 0.78 and params.get("_skin_found"):
        return "character"
    return "sculpt"


def sculpt_directive(text, relief=None, roundness=None):
    """Small, honest natural-language controls for the sculpt path:
    "flat back", "shallower", "more volume", "keep it under 2000 tris"."""
    t = (text or "").lower()
    out = {}
    if relief is not None:
        out["relief"] = bool(relief)
    if roundness is not None:
        out["roundness"] = float(roundness)
    flat = (re.search(r"\bflat\b", t) and re.search(r"\b(back|rear|behind|bottom)\b", t)) or \
        re.search(r"\b(relief|plaque|plack|coin|medal|badge|sticker|print\w*|slic\w*|3d\s*print|2\.5d|two\.5d)\b", t) or \
        re.search(r"\b(flatten|no bulge|hollow ?back)\b", t)
    rounder = re.search(r"\b(round it out|full 3d|full 3-d|solid back|volumetric|plump|"
                        r"more (volume|depth)|give it (volume|depth)|thicker|figurine)\b", t)
    if flat and not rounder:
        out["relief"] = True
    if rounder:
        out["relief"] = False
        out["roundness"] = min(1.6, (roundness or 1.0) * 1.35)
    if re.search(r"\b(thin|flatter|shallower|less depth|lower profile)\b", t):
        out["roundness"] = max(0.25, (roundness or 1.0) * 0.6)
    m = re.search(r"(?:under|below|max|keep it to)\s*(\d{3,6})\s*tris", t)
    if m:
        out["tris_cap"] = int(m.group(1))
    for name in BUDGETS:
        if re.search(r"\b%s\b" % name, t):
            out["budget"] = name
    m = re.search(r"\b(\d{3,4})\s*(?:px|texture|atlas)|(\d)k\s*texture\b", t)
    if m:
        px = int(m.group(1)) if m.group(1) else int(m.group(2)) * 1024
        out["texture_size"] = min((512, 1024, 2048), key=lambda v: abs(v - px))
    return out


def run_sculpt(image_bytes, model_dir, prompt, budget="game", tex_size=None,
               relief=False, roundness=1.0, shading=1.0, back_bytes=None, params=None,
               target_tris=None):
    """Any image -> sculpted 3D model, written to `model_dir` and reported."""
    import sculpt

    t0 = time.time()
    want = int(tex_size) if tex_size else (2048 if not relief else 1024)
    tex = _safe_tex(want)
    bud = budget if budget in BUDGETS else "game"
    mesh, uv, (base, mr, nrm), info = sculpt.reconstruct(
        image_bytes, back_bytes=back_bytes, budget=bud, tex_size=tex,
        roundness=roundness, shading=shading, relief=relief, target_tris=target_tris)
    bundle = export_bundle(mesh, (base, mr, nrm), model_dir)
    secs = time.time() - t0
    tri = info["geometry"]["triangles"]
    brief = (f"Sculpted a watertight {info['style']} from the artwork: "
             f"{info['size_m'][0]}×{info['size_m'][1]}×{info['size_m'][2]} m, "
             f"{tri:,} tris, textures straight from your image"
             + (", mirrored back" if not info.get("two_view") else ", rear measured too"))
    report = {
        "stages": [{"name": "Isolate silhouette + read depth", "ms": int(secs * 380)},
                   {"name": "Reconstruct surface (marching cubes)", "ms": int(secs * 300)},
                   {"name": f"Optimise topology -> {tri:,} tris", "ms": int(secs * 120)},
                   {"name": "Unwrap UVs + bake textures", "ms": int(secs * 120)},
                   {"name": "Export GLB / OBJ / STL", "ms": int(secs * 80)}],
        "pipeline": "sculpt",
        "source": "image",
        "prompt": prompt,
        "style": info["style"],
        "tags": info["tags"],
        "notes": info["notes"],
        "palette": {},
        "accessories": [],
        "geometry": dict(info["geometry"], texture_size=tex),
        "height_m": info["height_m"],
        "size_m": info["size_m"],
        "watertight": info["watertight"],
        "relief": bool(relief),
        "two_view": bool(info.get("two_view")),
        "plan": {"brief": brief, "reconstructor": "silhouette + shading -> implicit solid",
                 "detail": {"texture_size": tex}, "target_tris": int(tri)},
        "ai": {"enabled": False, "used": False, "model": None, "brief": None,
               "error": None, "plan_source": "measured from the image (no LLM needed)"},
        "files": bundle,
    }
    if params and params.get("source_notes"):
        report["notes"] = params["source_notes"] + report["notes"]
    return report


def export_bundle(mesh, textures, model_dir):
    """Write model.glb / model.obj+mtl / model.stl / the three maps; return sizes."""
    base, mr, nrm = textures
    files = {}

    def w(name, data):
        with open(f"{model_dir}/{name}", "wb") as fh:
            fh.write(data if isinstance(data, bytes) else data.tobytes())
        files[name] = len(data if isinstance(data, bytes) else data.tobytes())

    w("model.glb", mesh.export(file_type="glb"))
    w("model.stl", mesh.export(file_type="stl"))
    try:
        trimesh.exchange.export.export_mesh(mesh, f"{model_dir}/model.obj")
    except Exception:                                          # noqa: BLE001
        mesh.export(file_obj=f"{model_dir}/model.obj", file_type="obj")
    for nm in ("model.obj", "material.mtl"):
        import os as _os
        if _os.path.exists(f"{model_dir}/{nm}"):
            files[nm] = _os.path.getsize(f"{model_dir}/{nm}")
    base.save(f"{model_dir}/texture_atlas.png")
    mr.save(f"{model_dir}/mr_atlas.png")
    nrm.save(f"{model_dir}/normal_atlas.png")
    for nm in ("texture_atlas.png", "mr_atlas.png", "normal_atlas.png"):
        import os as _os
        files[nm] = _os.path.getsize(f"{model_dir}/{nm}")
    return files


def run_pipeline(source="text", text=None, image_bytes=None, budget="game",
                 tex_size=None, model_dir=".", prompt="", use_ai=True,
                 plan=None, prev_plan=None, feedback=None, mode="auto",
                 relief=None, roundness=None, back_image=None):
    """analyse -> plan -> geometry -> UV -> textures -> optimise -> export."""
    t0 = time.time()
    stages = []

    def done(name):
        stages.append({"name": name, "ms": int((time.time() - t0) * 1000)})

    params = analyze_image(image_bytes) if source == "image" else analyze_text(text or prompt or "")

    # ---- sculpt route: any image at all, not just characters -----------------
    if image_bytes and choose_mode(mode, source, params) == "sculpt":
        d = sculpt_directive(text or prompt or "", relief, roundness)
        rep = run_sculpt(image_bytes, model_dir, prompt,
                         budget=d.get("budget", budget), tex_size=d.get("texture_size", tex_size),
                         relief=d.get("relief", bool(relief)),
                         roundness=d.get("roundness", roundness if roundness is not None else 1.0),
                         back_bytes=back_image, params=params,
                         target_tris=d.get("tris_cap"))
        if d.get("relief") is not None or d.get("roundness") is not None:
            rep["notes"].append("Applied from your words: " + ", ".join(
                f"{k}={v}" for k, v in d.items() if k in ("relief", "roundness")))
        return rep
    if mode == "sculpt" and not image_bytes:
        raise ValueError("Sculpt mode reconstructs geometry from pixels - attach an image "
                         "(or switch to Character mode to build from text).")
    if prev_plan:                       # a revision refines the character that already exists
        params = plan_to_params(params, prev_plan)
    budget_hint = None if budget in (None, "auto") else BUDGETS[budget]["tris"]
    ai_info = {"enabled": False, "used": False, "model": None, "brief": None,
               "error": None, "plan_source": "heuristic"}

    if plan is None:
        try:
            import director
            ai_info["enabled"] = director.available()
            if use_ai and director.available():
                plan, model = director.plan_for(
                    text=text, image_bytes=image_bytes, budget_hint=budget_hint,
                    prev_plan=prev_plan, feedback=feedback,
                    base=params, source=source)
                ai_info["used"] = True
                ai_info["model"] = model
                ai_info["brief"] = plan.get("brief")
                ai_info["plan_source"] = "AI revision" if prev_plan else "AI director"
        except Exception as e:                        # noqa: BLE001
            ai_info["error"] = str(e)[:300]
            plan = None
        if plan is None:                              # fall back to the rule planner
            plan = local_revise(prev_plan, feedback) if (prev_plan and feedback) else \
                   local_plan(params, text or prompt or "", budget_hint)
            ai_info["plan_source"] = "rule-based revision" if prev_plan else "rule-based planner"
    else:
        ai_info["plan_source"] = "provided plan"

    # measured pixel colours (image input) beat the plan unless the user asked
    # for a specific change in this turn
    sticky = set(plan.get("_sticky_colors") or [])
    if prev_plan and prev_plan.get("palette"):
        for slot, hexv in (plan.get("palette") or {}).items():
            if prev_plan["palette"].get(slot) != hexv:
                sticky.add(slot)
    p = plan_to_params(params, plan)
    if source == "image":
        # a colour the user asked for out loud beats the pixel sample
        if text:
            intent = _param_diff(text.lower())
            sticky |= {k for k in intent if k in PLAN_SLOTS}
        for slot in (params.get("_sampled") or set()):   # names of pixel-sampled slots
            if slot in PLAN_SLOTS and slot not in sticky:
                p[slot] = tuple(params[slot])
    # an explicit colour request always wins, over both the model and the pixels
    if text:
        for slot, rgb in _param_diff(text.lower()).items():
            if slot in PLAN_SLOTS and isinstance(rgb, tuple):
                p[slot] = rgb
    done("Understand input (" + ai_info["plan_source"] + ")")

    # geometry budget: the plan decides unless the user capped it
    b = dict(BUDGETS["game" if budget in (None, "auto") else budget])
    det = plan.get("detail") or {}
    for key, lo, hi in (("sphere_sub", 1, 4), ("cyl_sec", 6, 32), ("cap_cnt", 4, 32)):
        if isinstance(det.get(key), (int, float)):
            b[key] = int(max(lo, min(hi, det[key])))
    b["cap_cnt"] = tuple(b["cap_cnt"]) if isinstance(b["cap_cnt"], (list, tuple)) else (b["cap_cnt"], b["cap_cnt"])
    if len(b["cap_cnt"]) == 1:
        b["cap_cnt"] = (b["cap_cnt"][0], b["cap_cnt"][0])
    target = int(plan.get("target_tris") or b["tris"])
    if budget_hint:
        target = min(target, budget_hint)
    target = max(400, min(40000, target))

    parts, geo_notes = build_character(p, b)
    extra, extra_notes = build_extra_parts(plan, p, b)
    parts += extra
    tris_before = sum(len(pt.mesh.faces) for pt in parts)
    done(f"Reconstruct geometry ({len(parts)} volumes)")

    parts, _, decimated = optimize_parts(parts, target)
    mesh, uv = assemble(parts)
    done(f"Optimise topology -> {len(mesh.faces):,} tris")

    want = int(tex_size) if tex_size else (2048 if source == "image"
                                           else int(plan.get("texture_size") or 1024))
    tex = _safe_tex(want)                            # never past what the box can hold
    color_img = finalize(mesh, uv, p, model_dir, tex_size=tex)
    done("Unwrap UVs + bake textures")
    done("Export GLB / OBJ / STL")

    if not str(plan.get("brief") or "").strip():        # never ship an empty brief
        plan["brief"] = (f"{params['style']} character"
                         + (": " + ", ".join(t for t in params.get("tags", [])[:4])
                            if params.get("tags") else ""))
    public_plan = {k: v for k, v in plan.items() if not k.startswith("_")}
    report = {
        "stages": stages,
        "source": source,
        "prompt": prompt,
        "style": p["style"],
        "tags": p.get("tags", []),
        "notes": geo_notes + extra_notes + params.get("source_notes", []),
        "palette": {k: swatch(p[k]) for k in PLAN_SLOTS},
        "accessories": sorted(p["accessories"]),
        "geometry": {
            "vertices": int(len(mesh.vertices)),
            "triangles": int(len(mesh.faces)),
            "triangles_before_optimize": int(tris_before),
            "target_tris": int(target),
            "decimated": bool(decimated),
            "materials": 1,
            "texture_size": tex,
            "maps": ["baseColor", "metallicRoughness", "normal"],
            "volumes": len(parts),
        },
        "height_m": round(float(p["height"]), 2),
        "heads_tall": round(float(p.get("heads") or STYLE_PROPS[p["style"]]["heads"]), 2),
        "plan": public_plan,
        "ai": ai_info,
    }
    return report
