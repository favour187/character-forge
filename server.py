"""Character Forge — web server exposing the text/image -> 3D pipeline.

Routes
  GET  /                    chat UI
  POST /chat                one conversational turn (message and/or image)
  GET  /chat/<sid>          session transcript (for reload)
  POST /reset               new empty session
  POST /generate            single-shot build (no session), kept for scripts
  GET  /download/<mid>/<f>  attachments: model.glb, obj.zip, model.stl, *.png, all.zip
  GET  /model/<mid>/<f>     inline file (used by the 3-D viewer)
  GET  /recent              recent builds
  GET  /health
"""

import io
import json
import logging
import shutil
import time
import os
import traceback
import uuid
import zipfile

from flask import Flask, jsonify, request, send_file, send_from_directory

import ai
import db
import engine

try:
    import memguard
except Exception:                                       # noqa: BLE001
    memguard = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("forge")

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "models")
SESSIONS = os.path.join(ROOT, "sessions")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(SESSIONS, exist_ok=True)

app = Flask(__name__, static_folder=os.path.join(ROOT, "static"),
            static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024     # one photo is plenty
DB_READY = db.init()
KEEP_DIRS = int(os.environ.get("FORGE_KEEP_MODELS", "24"))   # ephemeral disk on Render
MAX_UPLOAD = int(os.environ.get("FORGE_MAX_UPLOAD_MB", "16"))

MIME = {"glb": "model/gltf-binary", "gltf": "model/gltf+json", "obj": "text/plain",
        "mtl": "text/plain", "stl": "model/stl", "png": "image/png",
        "json": "application/json", "zip": "application/zip",
        "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}


def _mime(name):
    return MIME.get(name.rsplit(".", 1)[-1].lower(), "application/octet-stream")


def _downloads(mid):
    return {
        "glb": f"/download/{mid}/model.glb",
        "obj": f"/download/{mid}/obj.zip",
        "stl": f"/download/{mid}/model.stl",
        "texture": f"/download/{mid}/texture_atlas.png",
        "normal": f"/download/{mid}/normal_atlas.png",
        "mr": f"/download/{mid}/mr_atlas.png",
        "report": f"/download/{mid}/report.json",
        "all": f"/download/{mid}/all.zip",
    }


def _attach(mid, name, mimetype=None):
    """Attachment response with a friendly filename."""
    d = os.path.join(MODELS, mid)
    path = os.path.join(d, os.path.basename(name))
    if os.path.exists(path):
        return send_file(path, mimetype=mimetype or _mime(name), as_attachment=True,
                         download_name=f"character_{mid}_{name}")
    hit = db.get_file(mid, name)
    if hit:
        data, mime = hit
        return send_file(io.BytesIO(data), mimetype=mime, as_attachment=True,
                         download_name=f"character_{mid}_{name}")
    return jsonify({"error": "file not found — generate again on this instance"}), 404


def _zip_dir(mid, only=None, readme=None):
    d = os.path.join(MODELS, mid)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(d)):
            if fn == "report.json" or (only and not fn.endswith(only)):
                continue
            z.write(os.path.join(d, fn), f"character_{mid}/{fn}")
        if readme:
            z.writestr("character_%s/README.txt" % mid, readme)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------

def _session_path(sid):
    return os.path.join(SESSIONS, f"{os.path.basename(sid)}.json")


def load_session(sid):
    if not sid:
        return None
    p = _session_path(sid)
    if os.path.exists(p):
        try:
            with open(p) as fh:
                return json.load(fh)
        except Exception:                              # noqa: BLE001
            return None
    return None


def save_session(sess):
    try:
        with open(_session_path(sess["id"]), "w") as fh:
            json.dump(sess, fh)
    except Exception as e:                             # noqa: BLE001
        log.warning("session save failed: %s", e)


def _img_path(sid):
    return os.path.join(SESSIONS, f"{os.path.basename(sid)}_ref.img")


def _write_session_image(sid, data):
    try:
        with open(_img_path(sid), "wb") as fh:
            fh.write(data)
    except Exception as e:                                # noqa: BLE001
        log.warning("session image save failed: %s", e)


def _read_session_image(sid):
    p = _img_path(sid)
    try:
        if os.path.exists(p) and os.path.getsize(p) < 16 * 1024 * 1024:
            with open(p, "rb") as fh:
                return fh.read()
    except Exception:                                     # noqa: BLE001
        pass
    return None


def new_session(first_message=""):
    sid = uuid.uuid4().hex[:12]
    return {"id": sid, "created": None, "title": (first_message or "new character")[:60],
            "turns": [], "last_plan": None, "last_mid": None}


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


def _prune_models():
    """Keep only the newest builds on disk (the gallery mirrors them to Postgres)."""
    try:
        dirs = [d for d in os.listdir(MODELS) if os.path.isdir(os.path.join(MODELS, d))]
        dirs.sort(key=lambda d: os.path.getmtime(os.path.join(MODELS, d)), reverse=True)
        for d in dirs[KEEP_DIRS:]:
            shutil.rmtree(os.path.join(MODELS, d), ignore_errors=True)
    except Exception as e:                                # noqa: BLE001
        log.warning("prune failed: %s", e)


def _build(sess, source, text, image_bytes, budget, use_ai, prev_plan, feedback, prompt,
           mode="auto", relief=None, roundness=None, back_image=None):
    """One pipeline run, inside the single-slot gate that keeps RSS under the plan
    ceiling.  Raises memguard.Busy when the forge is already working."""
    if memguard is not None:
        with memguard.slot():
            return _build_inner(sess, source, text, image_bytes, budget, use_ai, prev_plan,
                                feedback, prompt, mode, relief, roundness, back_image)
    return _build_inner(sess, source, text, image_bytes, budget, use_ai, prev_plan, feedback,
                        prompt, mode, relief, roundness, back_image)


def _build_inner(sess, source, text, image_bytes, budget, use_ai, prev_plan, feedback, prompt,
                 mode, relief, roundness, back_image):
    mid = uuid.uuid4().hex[:12]
    model_dir = os.path.join(MODELS, mid)
    os.makedirs(model_dir, exist_ok=True)
    t0 = time.time()
    try:
        report = engine.run_pipeline(
            source=source, text=text, image_bytes=image_bytes, budget=budget,
            model_dir=model_dir, prompt=prompt, use_ai=use_ai,
            prev_plan=prev_plan, feedback=feedback, mode=mode, relief=relief,
            roundness=roundness, back_image=back_image)
    finally:
        if memguard is not None:
            memguard.release()          # never stack one build's high-water on the next
    report["build_ms"] = int((time.time() - t0) * 1000)
    report["memory"] = memguard.status() if memguard else None
    report["id"] = mid
    report["budget"] = budget
    report["downloads"] = _downloads(mid)
    report["preview"] = f"/model/{mid}/model.glb"
    with open(os.path.join(model_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    report["persisted"] = db.save(mid, report, model_dir)
    _prune_models()
    if sess is not None:
        sess["last_mid"] = mid
        sess["last_mode"] = report.get("pipeline") or "character"
        if report.get("plan"):
            sess["last_plan"] = report["plan"]
    return report


@app.route("/chat", methods=["POST"])
def chat():
    try:
        budget = request.form.get("budget", "auto")
        if budget not in ("auto",) and budget not in engine.BUDGETS:
            budget = "auto"
        message = (request.form.get("message") or "").strip()
        use_ai = request.form.get("use_ai", "1") not in ("0", "false", "off")
        mode = request.form.get("mode", "auto")
        relief = request.form.get("relief")
        relief = None if relief in (None, "") else relief in ("1", "true", "on")
        try:
            roundness = float(request.form.get("roundness") or 0) or None
        except ValueError:
            roundness = None
        image_bytes = None
        back_bytes = None
        if "image" in request.files and request.files["image"].filename:
            image_bytes = request.files["image"].read() or None
        if "back_image" in request.files and request.files["back_image"].filename:
            back_bytes = request.files["back_image"].read() or None

        sess = load_session(request.form.get("session"))
        if sess is None:
            sess = new_session(message)

        prev_plan = sess.get("last_plan") if (message and image_bytes is None
                                              and sess.get("last_plan")) else None
        # a sculpt rebuild needs the original pixels; keep the last image per session
        prior_image = None
        if image_bytes is None and sess.get("last_mode") == "sculpt":
            prior_image = _read_session_image(sess["id"])

        if image_bytes is not None or prior_image or not prev_plan:
            source = "image" if image_bytes is not None else "text"
            if source == "text" and not message and not prior_image:
                return jsonify({"error": "Type a description or attach a concept image."}), 400
            prompt = message or "(concept image)"
            if image_bytes is not None:
                _write_session_image(sess["id"], image_bytes)
            report = _build(sess, source if not prior_image else "image", message,
                            image_bytes or prior_image, budget, use_ai,
                            prev_plan=None, feedback=None, prompt=prompt, mode=mode,
                            relief=relief, roundness=roundness, back_image=back_bytes)
            kind = "build"
        else:
            report = _build(sess, "text", message, None, budget, use_ai,
                            prev_plan=prev_plan, feedback=message,
                            prompt=message, mode=mode, relief=relief, roundness=roundness)
            kind = "revision"

        turn = {
            "role": "assistant",
            "kind": kind,
            "text": message or "(concept image)",
            "reply": (report.get("plan", {}).get("brief") or "").strip(),
            "plan_source": report.get("ai", {}).get("plan_source"),
            "model": report.get("ai", {}).get("model"),
            "ai_error": report.get("ai", {}).get("error"),
            "mid": report["id"],
            "has_image": image_bytes is not None,
            "pipeline": report.get("pipeline") or "character",
            "ts": None,
        }
        sess["turns"].append(turn)
        save_session(sess)
        return jsonify({"session": sess["id"], "turn": turn, "report": report})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:                             # noqa: BLE001
        code, payload = _forge_error(e)
        if code:
            return _err(code, payload)
        traceback.print_exc()
        return jsonify({"error": f"Pipeline failed: {e}"}), 500


@app.route("/chat/<sid>")
def chat_state(sid):
    sess = load_session(sid)
    if not sess:
        return jsonify({"error": "unknown session"}), 404
    reports = {}
    for t in sess.get("turns", []):
        rp = os.path.join(MODELS, t.get("mid", ""), "report.json")
        if os.path.exists(rp):
            with open(rp) as fh:
                reports[t["mid"]] = json.load(fh)
    return jsonify({"session": sess["id"], "title": sess.get("title"),
                     "turns": sess.get("turns", []), "reports": reports})


@app.route("/reset", methods=["POST"])
def reset():
    sess = new_session()
    save_session(sess)
    return jsonify({"session": sess["id"], "title": sess["title"], "turns": []})


@app.route("/generate", methods=["POST"])
def generate():
    """Single-shot build (no chat session) — handy for scripts and the CLI."""
    try:
        budget = request.form.get("budget", "auto")
        if budget not in ("auto",) and budget not in engine.BUDGETS:
            budget = "auto"
        source = "image" if "image" in request.files else "text"
        text = (request.form.get("text") or "").strip()
        image_bytes = None
        if source == "image":
            image_bytes = request.files["image"].read()
            if not image_bytes:
                return jsonify({"error": "Empty image upload."}), 400
        elif not text:
            return jsonify({"error": "Enter a text prompt or upload a concept image."}), 400
        use_ai = request.form.get("use_ai", "1") not in ("0", "false", "off")
        mode = request.form.get("mode", "auto")
        relief = request.form.get("relief")
        relief = None if relief in (None, "") else relief in ("1", "true", "on")
        try:
            roundness = float(request.form.get("roundness") or 0) or None
        except ValueError:
            roundness = None
        back_bytes = None
        if "back_image" in request.files and request.files["back_image"].filename:
            back_bytes = request.files["back_image"].read() or None
        report = _build(None, source, text, image_bytes, budget, use_ai, None, None,
                        text or "(concept image)", mode=mode, relief=relief,
                        roundness=roundness, back_image=back_bytes)
        return jsonify(report)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:                             # noqa: BLE001
        code, payload = _forge_error(e)
        if code:
            return _err(code, payload)
        traceback.print_exc()
        return jsonify({"error": f"Pipeline failed: {e}"}), 500


@app.route("/recent")
def recent():
    items = db.recent(12)
    if not items:                                      # disk fallback (single instance)
        for mid in sorted(os.listdir(MODELS),
                          key=lambda m: os.path.getmtime(os.path.join(MODELS, m)),
                          reverse=True)[:12]:
            rp = os.path.join(MODELS, mid, "report.json")
            if os.path.exists(rp):
                with open(rp) as fh:
                    r = json.load(fh)
                items.append(dict(id=mid, source=r["source"], prompt=r.get("prompt"),
                                  style=r.get("style"),
                                  triangles=r["geometry"]["triangles"], created_at=None))
    return jsonify({"items": items, "persistent": DB_READY})


@app.route("/report/<mid>")
def report_route(mid):
    mid = os.path.basename(mid)
    rp = os.path.join(MODELS, mid, "report.json")
    if os.path.exists(rp):
        return send_file(rp, mimetype="application/json")
    r = db.get_report(mid)
    if r:
        r["downloads"] = _downloads(mid)
        r["preview"] = f"/model/{mid}/model.glb"
        return jsonify(r)
    return jsonify({"error": "not found"}), 404


@app.route("/download/<mid>/<name>")
def download(mid, name):
    mid, name = os.path.basename(mid), os.path.basename(name)
    if name == "all.zip":
        d = os.path.join(MODELS, mid)
        if os.path.isdir(d):
            buf = _zip_dir(mid, readme=(
                "Character Forge export\n"
                "----------------------\n"
                "model.glb   -> drag into Unity (glTFast), Godot, Blender, web viewers\n"
                "model.obj   -> + material.mtl (+ textures) for Unity/Blender/Maya\n"
                "model.stl   -> geometry only, for printing/CAD\n"
                "*.png       -> baseColor / normal / metallicRoughness atlases\n"
                "report.json -> full pipeline report incl. the AI build plan\n"
                "Y-up, metres, A-pose.\n"))
            return send_file(buf, mimetype="application/zip", as_attachment=True,
                             download_name=f"character_{mid}.zip")
        return jsonify({"error": "build not found on this instance"}), 404
    if name == "obj.zip":
        d = os.path.join(MODELS, mid)
        if os.path.isdir(d):
            buf = _zip_dir(mid, only=(".obj", ".mtl", ".png"), readme=(
                "model.obj + material.mtl -> import into Unity/Blender\n"
                "material_0.png           -> base colour (referenced by the MTL)\n"
                "normal_atlas.png         -> normal map\n"
                "mr_atlas.png             -> metallic/roughness (B=metal, G=rough)\n"))
            return send_file(buf, mimetype="application/zip", as_attachment=True,
                             download_name=f"character_{mid}_obj.zip")
        return jsonify({"error": "OBJ package is only on the instance that built it — "
                                 "generate again."}), 404
    return _attach(mid, name)


@app.route("/model/<mid>/<name>")
def model_file(mid, name):
    """Inline file serving for the 3-D viewer (no attachment header)."""
    mid, name = os.path.basename(mid), os.path.basename(name)
    d = os.path.join(MODELS, mid)
    if os.path.exists(os.path.join(d, name)):
        return send_from_directory(d, name, mimetype=_mime(name))
    return jsonify({"error": "not found"}), 404


def _err(code, payload):
    """JSON error with the headers a client needs to back off correctly."""
    resp = jsonify(payload)
    resp.status_code = code
    if payload.get("retry_after"):
        resp.headers["Retry-After"] = str(payload["retry_after"])
    return resp


def _forge_error(e):
    """Translate the two expected operational failures into honest HTTP."""
    if memguard is not None and isinstance(e, memguard.Busy):
        return 429, {"error": str(e), "busy": True, "retry_after": e.retry_after}
    if isinstance(e, (MemoryError,)) or "Allocating" in str(e) or "memory" in str(e).lower():
        log.error("out of memory during build: %s", e)
        return 503, {"error": "That build needed more memory than this free instance has "
                              "(512 MB). Try a smaller texture size or a lower polygon "
                              "budget, or move the service to a Starter plan.",
                     "memory": memguard.status() if memguard else None}
    return 0, {}


@app.errorhandler(413)
def too_large(_e):
    return _err(413, {"error": f"That file is over the {MAX_UPLOAD} MB limit. "
                               f"A screenshot or a photo under {MAX_UPLOAD} MB works fine."})


@app.route("/health")
def health():
    try:
        import neural3d
        neural = neural3d.status()
    except Exception:
        neural = {"available": False, "loaded": False, "device": "unavailable"}
    return jsonify({"ok": True, "database": DB_READY, "ai": ai.available(),
                    "ai_model": ai.PRIMARY if ai.available() else None,
                    "planner": "ai-director",
                    "neural_3d": neural,
                    "modes": ["character", "sculpt", "relief"],
                    "memory": memguard.status() if memguard else None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
