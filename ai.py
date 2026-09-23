"""OpenRouter-powered analysis stage (text LLM + vision LLM).

Turns a prompt or a concept image into the same `params` dict the heuristic
analyzers produce, so the rest of the pipeline is untouched.  Everything here
is best-effort: any failure (no key, 402/429, bad JSON, timeout) falls back to
the heuristics and is reported in `report["ai"]`.
"""

import base64
import io
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request

from PIL import Image

log = logging.getLogger("forge.ai")

API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
APP_URL = os.environ.get("OPENROUTER_APP_URL", "https://character-forge.onrender.com")
APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "Character Forge")
TIMEOUT = float(os.environ.get("OPENROUTER_TIMEOUT_MS", "45000")) / 1000.0

# Primary model + ordered fallbacks.  Defaults are free, vision-capable models
# (the key on this account is a free-tier key); override with OPENROUTER_MODEL.
PRIMARY = os.environ.get("OPENROUTER_MODEL", "nex-agi/nex-n2.5-mini:free").strip()
FALLBACKS = [m.strip() for m in os.environ.get(
    "OPENROUTER_FALLBACK_MODELS",
    "google/gemma-4-31b-it:free,qwen/qwen3.8-27b:free,google/gemma-4-26b-a4b-it:free,"
    "openrouter/free"
).split(",") if m.strip()]
PER_MODEL_TIMEOUT = min(TIMEOUT, 30.0)
TOTAL_BUDGET = float(os.environ.get("OPENROUTER_TOTAL_BUDGET_S", "70"))

ACCESSORIES = ["backpack", "wizard_hat", "sword", "shield", "cape", "staff",
               "horns", "cat_ears", "tail", "glasses"]
COLOR_KEYS = ["skin", "hair", "garment_a", "garment_b", "pants", "boots",
              "leather", "metal", "accent", "eye"]

SYSTEM = f"""You are the analysis stage of a text/image-to-3D character pipeline.
Return ONLY a JSON object (no prose, no markdown) with exactly these keys:
{{
  "style": "chibi" | "stylized" | "realistic",      // body proportions: chibi ~3 heads tall, stylized ~4.5, realistic ~6.5
  "height_m": number,                                   // 1.3 - 2.1
  "build": number,                                      // 0.8 slim ... 1.0 average ... 1.3 heavy/muscular
  "colors": {{ {", ".join(f'"{k}": "#rrggbb"' for k in COLOR_KEYS)} }},
  "accessories": [ subset of {json.dumps(ACCESSORIES)} ],
  "species": string,                                    // e.g. human, elf, orc, robot, cat-girl
  "tags": [ up to 8 short descriptive tags ],
  "brief": string                                       // one vivid sentence describing the character
}}
Colour semantics: garment_a = main torso garment / armour, garment_b = sleeves/cape/secondary cloth,
pants = lower garment, boots = footwear, leather = belts/straps/bags, metal = weapons/armour trim,
accent = gems/badges/hat band, eye = iris colour. Use the ACTUAL colours described or visible.
Only list accessories that are clearly present or described. For images, infer the unseen back side sensibly."""


def available():
    return bool(API_KEY)


def _post(payload):
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json",
                 "HTTP-Referer": APP_URL, "X-Title": APP_NAME})
    with urllib.request.urlopen(req, timeout=PER_MODEL_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def _extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
    raise ValueError("model returned no JSON object")


def _chat(messages):
    """Try the primary model then each fallback in turn. Returns (data, model)."""
    chain = [PRIMARY] + [m for m in FALLBACKS if m != PRIMARY]
    errors = []
    t0 = time.time()
    for model in chain:
        if time.time() - t0 > TOTAL_BUDGET:
            errors.append("time budget exhausted")
            break
        payload = {"model": model, "messages": messages,
                   "temperature": 0.2, "max_tokens": 700}
        try:
            res = _post(payload)
            if "error" in res:
                raise RuntimeError(str(res["error"].get("message", res["error"]))[:160])
            msg = res["choices"][0]["message"]
            content = msg.get("content")
            if isinstance(content, list):            # multi-part content
                content = "".join(p.get("text", "") for p in content)
            if not content:                           # reasoning-only models
                content = msg.get("reasoning") or ""
            return _extract_json(content), res.get("model", model)
        except urllib.error.HTTPError as e:
            body = e.read()[:160].decode("utf-8", "ignore")
            errors.append(f"{model}: HTTP {e.code} {body}")
            if e.code in (401, 403):
                break                                 # key problem: stop trying
        except Exception as e:  # noqa: BLE001
            errors.append(f"{model}: {str(e)[:120]}")
        time.sleep(0.3)
    raise RuntimeError(" | ".join(errors)[-400:])


def _hex_to_rgb(h):
    h = str(h).strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        return None
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _clean(raw):
    """Validate/normalise the model output into a partial params dict."""
    out = {}
    st = str(raw.get("style", "")).lower()
    if st in ("chibi", "stylized", "realistic"):
        out["style"] = st
    try:
        h = float(raw.get("height_m"))
        if 1.0 <= h <= 2.5:
            out["height"] = h
    except Exception:  # noqa: BLE001
        pass
    try:
        b = float(raw.get("build"))
        if 0.6 <= b <= 1.6:
            out["build"] = b
    except Exception:  # noqa: BLE001
        pass
    colors = {}
    for k, v in (raw.get("colors") or {}).items():
        if k in COLOR_KEYS:
            rgb = _hex_to_rgb(v)
            if rgb:
                colors[k] = rgb
    out["colors"] = colors
    acc = raw.get("accessories") or []
    out["accessories"] = {str(a).lower().replace(" ", "_").replace("-", "_")
                          for a in acc if str(a).lower().replace(" ", "_").replace("-", "_") in ACCESSORIES}
    out["tags"] = [str(t)[:32] for t in (raw.get("tags") or [])][:8]
    if raw.get("species"):
        out["species"] = str(raw["species"])[:32]
    if raw.get("brief"):
        out["brief"] = str(raw["brief"])[:280]
    return out


def analyze_text(text):
    data, model = _chat([{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Character description:\n{text}"}])
    return _clean(data), model


def analyze_image(image_bytes, hint=""):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((640, 640))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    user = [{"type": "text", "text": "Analyse this character concept image." +
             (f" Extra context: {hint}" if hint else "")},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
    data, model = _chat([{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}])
    return _clean(data), model


def merge(params, llm, source):
    """Overlay the LLM reading onto the heuristic params.

    Text: the LLM wins wherever it is confident (it understands language far
    better than keyword matching); the heuristic colours stay for anything the
    model did not specify.
    Image: pixel-measured colours and heads-tall proportions are exact, so they
    win; the LLM contributes semantics (accessories, species, style hints,
    colours for parts the sampler could not locate) and the brief."""
    notes = []
    if source == "text":
        for k in ("style", "height", "build"):
            if k in llm:
                params[k] = llm[k]
        for k, rgb in llm["colors"].items():
            params[k] = rgb
        params["accessories"] |= llm["accessories"]
        notes.append("Semantics, proportions and palette read by the language model")
    else:
        sampled = set(params.get("_sampled", ()))
        for k, rgb in llm["colors"].items():
            if k not in sampled:
                params[k] = rgb
        if "build" in llm and "build" not in sampled:
            params["build"] = llm["build"]
        params["accessories"] |= llm["accessories"]
        notes.append("Accessories / species read by the vision model; colours + proportions "
                     "measured from pixels")
    if llm.get("species"):
        params["tags"].insert(0, llm["species"])
    for t in llm.get("tags", []):
        if t not in params["tags"]:
            params["tags"].append(t)
    params["source_notes"] = notes + params.get("source_notes", [])
    return params
