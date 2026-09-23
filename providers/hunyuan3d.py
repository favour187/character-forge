"""Real 2D-image -> 3D geometry provider for Character Forge.

This module is deliberately separate from the old primitive modeller.  The
normal AI/director can describe the character, but a dedicated 3D model must
produce the actual mesh.  Character Forge talks to a Hunyuan3D-compatible
HTTP API and receives a real GLB.

Set:
    HY3D_API_URL=http://your-hunyuan-host:8081
Optional:
    HY3D_TIMEOUT_S=600
    HY3D_RESOLUTION=256
    HY3D_STEPS=8
    HY3D_GUIDANCE=5
    HY3D_FACE_COUNT=40000
    HY3D_TEXTURE=1

The official Hunyuan3D server accepts a base64 image at POST /generate and
returns the generated GLB.
"""

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request

import trimesh


def configured():
    return bool(os.environ.get("HY3D_API_URL", "").strip())


def _data_uri(image_bytes):
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")


def _post(url, payload, timeout):
    req = urllib.request.Request(
        url.rstrip("/") + "/generate",
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "model/gltf-binary"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read(), response.headers.get("Content-Type", "")


def generate(image_bytes, model_dir, target_tris=12000, texture=True):
    """Generate and save a real 3D GLB through a Hunyuan3D-compatible API.

    Returns compact geometry metadata for Character Forge's existing report/UI.
    """
    if not image_bytes:
        raise ValueError("A source image is required for neural 3D reconstruction.")

    base = os.environ.get("HY3D_API_URL", "").strip().rstrip("/")
    if not base:
        raise RuntimeError(
            "HY3D_API_URL is not configured. Set it to a running Hunyuan3D API server."
        )

    timeout = float(os.environ.get("HY3D_TIMEOUT_S", "600"))
    resolution = int(os.environ.get("HY3D_RESOLUTION", "256"))
    steps = int(os.environ.get("HY3D_STEPS", "8"))
    guidance = float(os.environ.get("HY3D_GUIDANCE", "5"))
    face_count = int(os.environ.get("HY3D_FACE_COUNT", str(max(1000, min(100000, target_tris)))))

    payload = {
        "image": _data_uri(image_bytes),
        "remove_background": True,
        "texture": str(os.environ.get("HY3D_TEXTURE", "1")).lower() not in ("0", "false", "off"),
        "seed": int(time.time()) & 0xFFFFFFFF,
        "octree_resolution": max(64, min(512, resolution)),
        "num_inference_steps": max(1, min(20, steps)),
        "guidance_scale": max(0.1, min(20.0, guidance)),
        "face_count": max(1000, min(100000, face_count)),
        "type": "glb",
    }

    try:
        data, content_type = _post(base, payload, timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", "ignore")
        raise RuntimeError(f"Hunyuan3D API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Hunyuan3D API: {exc.reason}") from exc

    # The official endpoint returns the GLB as a binary response. Give a useful
    # error if a proxy/service returned JSON instead.
    if data[:1] in (b"{", b"["):
        try:
            obj = json.loads(data.decode("utf-8"))
            raise RuntimeError(obj.get("message") or obj.get("text") or str(obj))
        except UnicodeDecodeError:
            pass

    if not data or len(data) < 16:
        raise RuntimeError("Hunyuan3D returned an empty or invalid model.")

    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "model.glb")
    with open(path, "wb") as fh:
        fh.write(data)

    try:
        mesh = trimesh.load(io.BytesIO(data), file_type="glb", force="scene")
        if isinstance(mesh, trimesh.Scene):
            vertices = sum(len(g.vertices) for g in mesh.geometry.values())
            triangles = sum(len(g.faces) for g in mesh.geometry.values())
            volumes = len(mesh.geometry)
        else:
            vertices = len(mesh.vertices)
            triangles = len(mesh.faces)
            volumes = 1
    except Exception as exc:
        raise RuntimeError(f"Hunyuan3D returned data that could not be read as GLB: {exc}") from exc

    return {
        "path": path,
        "vertices": int(vertices),
        "triangles": int(triangles),
        "volumes": int(volumes),
        "provider": "hunyuan3d",
        "content_type": content_type,
        "textured": bool(payload["texture"]),
        "settings": {
            "octree_resolution": payload["octree_resolution"],
            "num_inference_steps": payload["num_inference_steps"],
            "face_count": payload["face_count"],
        },
    }
