"""Character Forge startup hook.

Routes image builds in Auto mode through the existing native sculpt reconstructor.
The existing parametric character engine remains available for explicit Character
mode, and the hook is fail-safe: if anything is unavailable, normal engine
startup continues unchanged.
"""

from __future__ import annotations

import os


def _enabled() -> bool:
    return os.environ.get("FORGE_NATIVE_IMAGE_RECON", "1").strip().lower() not in {
        "0", "false", "off", "no"
    }


def _install() -> None:
    if not _enabled():
        return

    try:
        import engine
    except Exception:
        # Never make Python/server startup fail because of this optional routing.
        return

    current = getattr(engine, "run_pipeline", None)
    if current is None or getattr(current, "_native_image_recon", False):
        return

    def run_pipeline(*args, **kwargs):
        source = kwargs.get("source", args[0] if args else "text")
        image_bytes = kwargs.get("image_bytes", args[2] if len(args) > 2 else None)
        mode = kwargs.get("mode", "auto")

        # Auto + image means actual arbitrary-surface reconstruction.
        # Explicit Character mode keeps the original parametric rig untouched.
        if source == "image" and image_bytes and str(mode).lower() == "auto":
            kwargs["mode"] = "sculpt"

        return current(*args, **kwargs)

    run_pipeline._native_image_recon = True
    run_pipeline._native_image_recon_original = current
    engine.run_pipeline = run_pipeline


_install()
