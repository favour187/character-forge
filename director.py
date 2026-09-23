"""Build director — the language model decides *how* to turn the input into 3D.

The director returns a **build plan**: a JSON contract that the geometry engine
executes.  The model is in charge of the creative/technical decisions the user
asked it to make — proportions, silhouette breakdown, materials, the vertex /
triangle budget, texture resolution, what the hidden back of the character
looks like, and any extra primitives the parametric body does not cover.

Everything is validated and clamped here, so a hallucinating model can never
produce an unbuildable plan: unknown features are dropped, numbers are clamped,
and any field the model leaves out falls back to the deterministic rule plan.
"""

import os
import re

import ai

# keep in sync with engine.FEATURES / engine.PLAN_SLOTS / engine.TILE_NAMES
FEATURES = ["backpack", "wizard_hat", "cape", "sword", "staff", "shield", "horns",
            "cat_ears", "tail", "glasses", "hood", "hair_long", "crown", "scarf",
            "skirt", "wings", "shoulder_pads", "armor_plates", "robot_joints",
            "boots_tall", "tail_fluffy"]
SLOTS = ["skin", "hair", "garment_a", "garment_b", "pants", "boots",
         "leather", "metal", "accent", "eye"]
MATERIALS = ["skin", "hair", "garment_a", "garment_b", "pants", "boots",
             "leather", "metal", "accent", "eye"]
PRIMITIVES = ["box", "sphere", "capsule", "cylinder", "cone", "sheet"]

SCHEMA = """{
  "brief": "one sentence describing the character you are building",
  "style": "chibi | stylized | realistic",
  "species": "orc / elf / robot / human student / ...",
  "height_m": 1.7,
  "heads_tall": 4.5,
  "build": 1.0,
  "head_boost": 1.0,
  "target_tris": 4500,
  "texture_size": 1024,
  "detail": {"sphere_sub": 3, "cyl_sec": 14, "cap_cnt": 12},
  "palette": {"skin": "#e8b28e", "hair": "#4a3223", "garment_a": "#8a4f9e",
              "garment_b": "#ece4d2", "pants": "#3a3e54", "boots": "#5a4632",
              "leather": "#805a38", "metal": "#b0b5be", "accent": "#d8a83c",
              "eye": "#282a34"},
  "materials": {"garment_a": "matte woven cotton", "metal": "scratched steel",
                "leather": "worn brown leather"},
  "features": ["cape", "hood"],
  "back_view": "what the hidden parts (back, underside, unseen side) look like, and how you reconstruct them",
  "modelling_plan": ["head: sphere subdiv 3 + clipped hair cap",
                     "torso: capsule scaled 1.35x wide",
                     "cape: thin box sheet angled 8 deg back"],
  "extra_parts": [
    {"name": "floating orb", "type": "sphere", "at": [0.3, 1.3, 0.2],
     "size": [0.12, 0.12, 0.12], "material": "accent", "rot_deg": [0, 0, 0]}
  ],
  "notes": ["anything the modeller should know"],
  "confidence": 0.8
}"""

RULES = """RULES
- Output ONLY the JSON object. Do NOT explain, do NOT think out loud, do NOT use markdown.
- Units are metres, Y is up, the character stands on y=0 and faces +Z.
- "target_tris" is the triangle budget YOU choose for this character and use case; pick 1200-2500 for very simple mobile props, 3000-6000 for normal game characters, 8000-20000 only when the brief demands fine detail or hair strands.
- "detail" sets construction resolution: sphere_sub 1-4, cyl_sec/  cap_cnt 6-32. Keep them consistent with target_tris.
- palette entries must be 6-digit hex colours sampled from the description/image.
- "features" may only use: """ + ", ".join(FEATURES) + """.
- Never duplicate: if something is in "features" (cape, staff, sword, shield, wings, hat, backpack, tail, horns) do NOT also build it as an extra part.
- "extra_parts" are extra primitives for things the parametric body does not cover (wings, tails, floating objects, weapons, mech parts). Positions are in metres: x = left/right, y = height above the feet, z = depth (positive = front, negative = back). size = [width, height, depth] in metres, max 6 parts.
- For an input image, study silhouette, proportions, depth cues, materials, clothing and accessories, then describe in "back_view" how you reconstruct what the camera cannot see (mirroring, symmetry, standard garment construction).
- Reply with ONE JSON object and nothing else."""


def available():
    return ai.available()


def _messages(text, image_bytes, hint, budget_hint, prev_plan, feedback, base, source):
    import json
    parts = []
    if source == "image":
        head = ("You are the build director of a 3D character generator. The user uploaded a "
                "concept image. Turn it into a build plan for a game-ready 3D character.")
    else:
        head = ("You are the build director of a 3D character generator. Turn the user's "
                "description into a build plan for a game-ready 3D character.")
    if prev_plan:
        head += ("\n\nThe character already exists (plan below). Apply the user's change "
                 "request and return the COMPLETE updated plan, keeping everything they did "
                 "not ask to change identical.")
    if budget_hint:
        head += (f"\n\nThe user capped the budget at {budget_hint} triangles: target_tris "
                 f"must be <= {budget_hint}.")
    body = [head, "", "SCHEMA", SCHEMA, "", RULES,
            "- Keep every string under 14 words. No trailing commentary.", ""]
    if prev_plan:
        import json as _json
        body += ["CURRENT PLAN", _json.dumps(prev_plan, ensure_ascii=False)[:4000], ""]
    if text:
        body += ["CHARACTER DESCRIPTION", text[:1200], ""]
    if hint and hint != text:
        body += ["EXTRA HINT FROM THE USER", hint[:400], ""]
    body += ["MEASURED ANALYSIS (trust these numbers and colours)",
             f"style={base.get('style')} height={base.get('height')} build={base.get('build')}",
             "palette=" + str({k: base.get(k) for k in SLOTS}),
             "detected=" + str(sorted(base.get("accessories", [])))]
    if prev_plan and feedback:
        body += ["", "CHANGE REQUEST", feedback[:600]]
    content = [{"type": "text", "text": "\n".join(body)}]
    if image_bytes:
        b64 = ai._b64_image(image_bytes)          # noqa: SLF001  (shared helper)
        if b64:
            content.append({"type": "image_url", "image_url": {"url": b64}})
    parts.append({"role": "user", "content": content})
    return parts


def plan_for(text=None, image_bytes=None, budget_hint=None, prev_plan=None,
             feedback=None, base=None, source="text"):
    """Ask the model for a build plan; returns (plan, model_name)."""
    if base is None:
        import engine
        base = engine.default_params()
    raw, model = ai._chat(_messages(text or feedback, image_bytes, text or "",
                                    budget_hint, prev_plan, feedback, base, source),
                          max_tokens=int(os.environ.get("FORGE_PLAN_MAX_TOKENS", "8000")),
                          temperature=0.3,
                          timeout=float(os.environ.get("FORGE_PLAN_TIMEOUT_S", "60")),
                          budget=float(os.environ.get("FORGE_PLAN_BUDGET_S", "150")))
    return clean_plan(raw, base, budget_hint, fallback_text=text), model


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _num(v, lo, hi, default=None):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:                                        # NaN
        return default
    return max(lo, min(hi, f))


def _hex(v):
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        try:
            return "#%02x%02x%02x" % tuple(max(0, min(255, int(c))) for c in v[:3])
        except (TypeError, ValueError):
            return None
    s = str(v or "").strip().lower()
    if re.fullmatch(r"#?[0-9a-f]{6}", s):
        return "#" + s.lstrip("#")
    if re.fullmatch(r"#?[0-9a-f]{3}", s):
        s = s.lstrip("#")
        return "#" + "".join(c * 2 for c in s)
    named = {"red": "#c82d2d", "blue": "#3c5fbe", "green": "#469650", "black": "#232328",
             "white": "#ebebeb", "grey": "#828287", "gray": "#828287", "brown": "#7d5532",
             "gold": "#daa532", "silver": "#bec3c8", "purple": "#7346aa", "pink": "#e68caa",
             "teal": "#2d9696", "orange": "#e18228", "yellow": "#e6c846"}
    return named.get(s)


def clean_plan(raw, base, budget_hint=None, fallback_text=None):
    """Validate an LLM plan against the engine contract; fill gaps from the rules."""
    import engine
    seed = (fallback_text or "").strip() or str(raw.get("species") or "").strip() or \
        ", ".join(str(t) for t in (base.get("tags") or [])[:4]) or "a game character"
    rule = engine.local_plan(base, seed, budget_hint)
    if not isinstance(raw, dict):
        raw = {}
    plan = dict(rule)

    plan["brief"] = str(raw.get("brief") or rule["brief"])[:400]
    plan["species"] = str(raw.get("species") or rule["species"])[:60]
    if str(raw.get("style", "")).lower() in ("chibi", "stylized", "realistic"):
        plan["style"] = str(raw["style"]).lower()
    plan["height_m"] = round(_num(raw.get("height_m"), 0.6, 2.6, rule["height_m"]), 3)
    heads_range = {"chibi": (2.6, 3.4), "stylized": (4.0, 5.4), "realistic": (5.8, 7.4)}
    lo, hi = heads_range.get(plan["style"], (2.6, 7.4))
    plan["heads_tall"] = round(_num(raw.get("heads_tall"), lo, hi, min(max(rule["heads_tall"], lo), hi)), 2)
    plan["build"] = round(_num(raw.get("build"), 0.7, 1.45, rule["build"]), 2)
    plan["head_boost"] = round(_num(raw.get("head_boost"), 0.6, 1.25,
                                    rule.get("head_boost", 1.0)), 2)

    tris = _num(raw.get("target_tris"), 400, 40000, rule["target_tris"])
    if budget_hint:
        tris = min(tris, budget_hint)
    plan["target_tris"] = int(tris)

    ts = _num(raw.get("texture_size"), 512, 2048, 1024)
    plan["texture_size"] = 512 if ts < 768 else (2048 if ts > 1400 else 1024)

    det = raw.get("detail") if isinstance(raw.get("detail"), dict) else {}
    from_rules = engine.detail_for_target(plan["target_tris"])
    plan["detail"] = {k: int(_num(det.get(k), lo, hi, from_rules[k]))
                      for k, lo, hi in (("sphere_sub", 1, 4), ("cyl_sec", 6, 32),
                                        ("cap_cnt", 4, 32))}

    palette = dict(rule["palette"])
    for slot, val in (raw.get("palette") or {}).items():
        slot = str(slot).lower().strip()
        if slot in SLOTS:
            h = _hex(val)
            if h:
                palette[slot] = h
    plan["palette"] = palette

    mats = {}
    if isinstance(raw.get("materials"), dict):
        for slot, desc in list(raw["materials"].items())[:10]:
            slot = str(slot).lower().strip()
            if slot in MATERIALS and isinstance(desc, (str, int, float)):
                mats[slot] = str(desc)[:60]
    plan["materials"] = mats

    feats = [f for f in (raw.get("features") or [])
             if isinstance(f, str) and f.lower().strip() in FEATURES]
    plan["features"] = sorted({f.lower().strip() for f in feats})

    for key in ("back_view",):
        if isinstance(raw.get(key), str) and raw[key].strip():
            plan[key] = raw[key].strip()[:600]

    mp = [str(x)[:120] for x in (raw.get("modelling_plan") or [])
          if isinstance(x, (str, int, float))][:8]
    plan["modelling_plan"] = mp or rule["modelling_plan"]

    notes = [str(x)[:160] for x in (raw.get("notes") or [])
             if isinstance(x, (str, int, float))][:6]
    plan["notes"] = notes

    extras = []
    for item in (raw.get("extra_parts") or [])[:6]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type", "")).lower().strip()
        if kind not in PRIMITIVES:
            continue
        at = item.get("at") if isinstance(item.get("at"), (list, tuple)) else [0, 0, 0]
        size = item.get("size") if isinstance(item.get("size"), (list, tuple)) else [0.2] * 3
        rot = item.get("rot_deg") if isinstance(item.get("rot_deg"), (list, tuple)) else [0, 0, 0]
        extras.append({
            "name": str(item.get("name") or kind)[:40],
            "type": kind,
            "at": [round(_num(v, -2.0, 3.0, 0.0), 3) for v in list(at)[:3] + [0, 0, 0]][:3],
            "size": [round(_num(v, 0.02, 1.6, 0.2), 3) for v in list(size)[:3] + [0.2, 0.2, 0.2]][:3],
            "rot_deg": [round(_num(v, -180, 180, 0.0), 1) for v in list(rot)[:3] + [0, 0, 0]][:3],
            "material": str(item.get("material", "leather")).lower().strip()
                        if str(item.get("material", "leather")).lower().strip() in MATERIALS
                        else "leather",
        })
    plan["extra_parts"] = extras
    plan["confidence"] = round(_num(raw.get("confidence"), 0, 1, 0.7), 2)
    plan["generator"] = "ai"
    return plan
