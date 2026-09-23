"""Character Forge — web server exposing the text/image -> 3D pipeline."""

import io
import json
import logging
import os
import traceback
import uuid
import zipfile

from flask import Flask, jsonify, request, send_file, send_from_directory

import db
import engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("forge")

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "models")
os.makedirs(MODELS, exist_ok=True)

app = Flask(__name__, static_folder=os.path.join(ROOT, "static"),
            static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
DB_READY = db.init()


def _downloads(mid):
    return {
        "glb": f"/model/{mid}/model.glb",
        "obj": f"/model/{mid}/obj.zip",
        "stl": f"/model/{mid}/model.stl",
        "texture": f"/model/{mid}/texture_atlas.png",
    }


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        budget = request.form.get("budget", "game")
        if budget not in engine.BUDGETS:
            budget = "game"
        source = "image" if "image" in request.files else "text"
        text = request.form.get("text", "").strip()
        image_bytes = None
        if source == "image":
            image_bytes = request.files["image"].read()
            if not image_bytes:
                return jsonify({"error": "Empty image upload."}), 400
        elif not text:
            return jsonify({"error": "Enter a text prompt or upload a concept image."}), 400

        mid = uuid.uuid4().hex[:12]
        model_dir = os.path.join(MODELS, mid)
        os.makedirs(model_dir, exist_ok=True)

        report = engine.run_pipeline(
            source=source, text=text, image_bytes=image_bytes,
            budget=budget, model_dir=model_dir,
            prompt=text or "(concept image)")
        report["id"] = mid
        report["budget"] = budget
        report["downloads"] = _downloads(mid)
        report["preview"] = f"/model/{mid}/model.glb"
        with open(os.path.join(model_dir, "report.json"), "w") as fh:
            json.dump(report, fh, indent=2)
        report["persisted"] = db.save(mid, report, model_dir)
        return jsonify(report)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:                       # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": f"Pipeline failed: {e}"}), 500


@app.route("/recent")
def recent():
    items = db.recent(12)
    if not items:                                # disk fallback (single instance)
        for mid in sorted(os.listdir(MODELS),
                          key=lambda m: os.path.getmtime(os.path.join(MODELS, m)),
                          reverse=True)[:12]:
            rp = os.path.join(MODELS, mid, "report.json")
            if os.path.exists(rp):
                r = json.load(open(rp))
                items.append(dict(id=mid, source=r["source"], prompt=r.get("prompt"),
                                  style=r.get("style"),
                                  triangles=r["geometry"]["triangles"], created_at=None))
    return jsonify({"items": items, "persistent": DB_READY})


@app.route("/report/<mid>")
def report(mid):
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


@app.route("/model/<mid>/<name>")
def model_file(mid, name):
    mid, name = os.path.basename(mid), os.path.basename(name)
    d = os.path.join(MODELS, mid)
    if os.path.isdir(d):
        if name == "obj.zip":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for fn in sorted(os.listdir(d)):
                    if fn.endswith((".obj", ".mtl", ".png")):
                        z.write(os.path.join(d, fn), f"character/{fn}")
                z.writestr("character/README.txt",
                           "model.obj + material.mtl  -> import into Unity/Blender\n"
                           "material_0.png            -> base color (referenced by the MTL)\n"
                           "normal_atlas.png          -> normal map (assign in the material)\n"
                           "mr_atlas.png              -> glTF metal/rough (B=metallic, G=roughness)\n"
                           "Y-up, metres, A-pose.\n")
            buf.seek(0)
            return send_file(buf, mimetype="application/zip",
                             download_name=f"character_{mid}_obj.zip")
        if os.path.exists(os.path.join(d, name)):
            return send_from_directory(d, name)
    # not on this instance's disk -> database mirror
    hit = db.get_file(mid, name)
    if hit:
        data, mime = hit
        return send_file(io.BytesIO(data), mimetype=mime, download_name=name)
    if name == "obj.zip":
        return jsonify({"error": "OBJ package is only available on the instance that "
                                 "generated it - re-generate to download OBJ."}), 404
    return jsonify({"error": "not found"}), 404


@app.route("/health")
def health():
    return jsonify({"ok": True, "database": DB_READY})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
