"""Neural image-to-3D backend for Character Forge.

This module keeps the 3D reconstruction model inside the Character Forge
pipeline: the model code is an optional dependency and the pretrained weights
are cached locally by Hugging Face. It does not call a hosted 3D-generation API.

Backend: Stability AI / Tripo AI TripoSR (MIT).
"""

from __future__ import annotations

import io
import os
import threading
import time
from typing import Any

import numpy as np
from PIL import Image

_MODEL = None
_DEVICE = None
_LOCK = threading.Lock()


def configured() -> bool:
    return os.environ.get("FORGE_3D_ENGINE", "auto").strip().lower() in {
        "auto", "triposr", "neural", "ai"
    }


def available() -> bool:
    if not configured():
        return False
    try:
        import torch
        from tsr.system import TSR
        return True
    except Exception:
        return False


def status() -> dict[str, Any]:
    requested = os.environ.get("FORGE_3D_ENGINE", "auto").strip().lower()
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        device = "cuda:0" if cuda else "cpu"
    except Exception:
        cuda, device = False, "unavailable"
    return {
        "requested": requested,
        "available": available(),
        "loaded": _MODEL is not None,
        "device": _DEVICE or device,
        "model": os.environ.get("TRIPOSR_MODEL", "stabilityai/TripoSR"),
        "mc_resolution": int(os.environ.get("TRIPOSR_MC_RESOLUTION", "256")),
        "gpu": cuda,
    }


def _load():
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL, _DEVICE

    import torch
    from tsr.system import TSR

    device = os.environ.get("TRIPOSR_DEVICE", "").strip()
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_name = os.environ.get("TRIPOSR_MODEL", "stabilityai/TripoSR")
    model = TSR.from_pretrained(
        model_name,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    try:
        model.renderer.set_chunk_size(int(os.environ.get("TRIPOSR_CHUNK_SIZE", "8192")))
    except Exception:
        pass

    model.to(device)
    model.eval()
    _MODEL, _DEVICE = model, device
    return _MODEL, _DEVICE


def _prepare(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    if os.environ.get("TRIPOSR_REMOVE_BG", "1").strip().lower() not in {"0", "false", "off"}:
        try:
            from tsr.utils import remove_background
            import rembg
            image = remove_background(image, rembg.new_session())
        except Exception:
            pass
    max_side = int(os.environ.get("TRIPOSR_INPUT_MAX", "1024"))
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def generate(image_bytes: bytes, output_path: str, *, back_bytes: bytes | None = None) -> dict[str, Any]:
    """Generate a genuine neural 3D mesh and write model.glb.

    TripoSR's inference path is single-image reconstruction. A supplied back
    image is recorded but not falsely claimed to be fused into the mesh.
    """
    if not available():
        raise RuntimeError(
            "TripoSR is not installed. Install requirements-neural3d.txt on a "
            "GPU-capable worker to enable the neural image-to-3D engine."
        )

    import torch

    image = _prepare(image_bytes)
    model, device = _load()
    started = time.time()

    with _LOCK, torch.inference_mode():
        scene_codes = model(image, device=device)
        resolution = int(os.environ.get("TRIPOSR_MC_RESOLUTION", "256"))
        threshold = float(os.environ.get("TRIPOSR_THRESHOLD", "25"))
        meshes = model.extract_mesh(
            scene_codes,
            has_vertex_color=True,
            resolution=resolution,
            threshold=threshold,
        )

    mesh = meshes[0]
    try:
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
    except Exception:
        pass

    ext = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=float)
    if np.any(ext > 0):
        target_height = 1.8
        scale = target_height / max(float(ext[1]), 1e-6)
        mesh.apply_scale(scale)
        mesh.apply_translation(-mesh.bounds.mean(axis=0))

    glb = mesh.export(file_type="glb")
    with open(output_path, "wb") as fh:
        fh.write(glb)

    return {
        "pipeline": "neural-triposr",
        "reconstructor": "TripoSR",
        "device": device,
        "model": os.environ.get("TRIPOSR_MODEL", "stabilityai/TripoSR"),
        "inference_ms": int((time.time() - started) * 1000),
        "geometry": {
            "vertices": int(len(mesh.vertices)),
            "triangles": int(len(mesh.faces)),
            "materials": 1,
            "texture_size": None,
            "maps": ["vertexColor"],
            "watertight": bool(getattr(mesh, "is_watertight", False)),
        },
        "two_view": False,
        "back_image_ignored": bool(back_bytes),
        "note": (
            "Neural single-image reconstruction. Hidden surfaces are inferred "
            "by the learned 3D model."
        ),
    }
