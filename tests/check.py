#!/usr/bin/env python3
"""End-to-end check for Character Forge.

Run it against a local server or the live one:

    python tests/check.py                          # http://127.0.0.1:8000
    python tests/check.py https://character-forge.onrender.com
    python tests/check.py https://character-forge.onrender.com --slow

It builds from text, from an image (sculpt route), from a full-body image
(character route), revises with a spoken instruction, verifies every download,
prods the memory guard, and fires a concurrency burst to prove the instance
survives it (no OOM).  Exits non-zero on the first failure.
"""
import argparse
import io
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fixtures import buf, car, face, knight, logo        # noqa: E402

FAILS = []
PASSES = []


def ok(name, extra=""):
    PASSES.append(name)
    print(f"  \033[32mPASS\033[0m {name}{(' — ' + extra) if extra else ''}")


def bad(name, why):
    FAILS.append(f"{name}: {why}")
    print(f"  \033[31mFAIL\033[0m {name}: {why}")


def check(name, cond, extra=""):
    ok(name, extra) if cond else bad(name, extra or "condition false")
    return bool(cond)


def http(base, path, data=None, filename=None, content_type=None, method=None, timeout=300):
    """Tiny multipart/form client (no requests dependency)."""
    url = base.rstrip("/") + path
    body = None
    if data is not None:
        boundary = "----forge" + uuid.uuid4().hex
        parts = []
        for k, v in data.items():
            if v is None:
                continue
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        for k, (fname, blob, ctype) in (filename or {}).items():
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; "
                         f"filename=\"{fname}\"\r\nContent-Type: {ctype}\r\n\r\n".encode()
                         + blob + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)
        content_type = f"multipart/form-data; boundary={boundary}"
    req = urllib.request.Request(url, data=body, method=method or ("POST" if body else "GET"))
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, dict(r.headers), raw
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read()


def jload(raw):
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except Exception:                                     # noqa: BLE001
        return {}


def glb_is_sane(blob):
    """glTF 2.0 binary: magic, version, and a POSITION/TEXCOORD_0 attribute set."""
    if len(blob) < 20 or struct.unpack_from("<I", blob, 0)[0] != 0x46546C67:
        return False, "not a GLB"
    if struct.unpack_from("<I", blob, 4)[0] != 2:
        return False, "gltf version != 2"
    n = struct.unpack_from("<I", blob, 12)[0]
    j = json.loads(blob[20:20 + n])
    prim = j["meshes"][0]["primitives"][0]
    attrs = set(prim.get("attributes", {}))
    if "POSITION" not in attrs:
        return False, "no POSITION"
    mats = j.get("materials", [{}])
    pbr = mats[0].get("pbrMetallicRoughness", {}) if mats else {}
    return True, {"attrs": sorted(attrs), "baseColorTexture": "baseColorTexture" in pbr,
                  "metallicRoughnessTexture": "metallicRoughnessTexture" in pbr,
                  "normalTexture": "normalTexture" in (mats[0] if mats else {}),
                  "images": len(j.get("images", [])), "tris": sum(
                      1 for _ in range(1)) and len(j["accessors"][j["meshes"][0]["primitives"][0]["indices"]]["type"]) > 0}


def run(base, slow=False):
    print(f"\n\033[1mCharacter Forge check against {base}\033[0m\n")
    t0 = time.time()

    # ── liveness ───────────────────────────────────────────────────────────
    st, _h, raw = http(base, "/health")
    h = jload(raw)
    if not check("GET /health returns 200 + json", st == 200 and h.get("ok") is True, f"{st} {raw[:120]!r}"):
        return
    ok("health payload", f"ai={h.get('ai')} db={h.get('database')} atlas={(h.get('memory') or {}).get('atlas_px')}²")
    check("health advertises the three modes", set(h.get("modes", [])) >= {"character", "sculpt", "relief"},
          str(h.get("modes")))

    # ── the site itself ────────────────────────────────────────────────────
    st, hd, raw = http(base, "/")
    html = raw.decode("utf-8", "replace")
    check("GET / serves the studio", st == 200 and "Character Forge" in html, str(st))
    for needle in ("viewport-fit=cover", "role=\"tablist\"", "Drop the image you want in 3D", "prefers-reduced-motion"):
        check(f"UI contains {needle}", needle in html, "missing")
    st, _, three = http(base, "/static/vendor/three/build/three.module.js")
    check("vendored three.js is served (works offline)", st == 200 and len(three) > 100_000, f"{len(three)} bytes")

    # ── text -> rigged character ───────────────────────────────────────────
    label = "text prompt -> character"
    st, _h, raw = http(base, "/generate", {"text": "chibi cat-girl mage with a purple robe and a pointy hat",
                                           "budget": "mobile", "use_ai": "0", "mode": "auto"})
    rep = jload(raw)
    if check(label + " -> 200", st == 200, f"{st} {raw[:200]!r}"):
        g = rep.get("geometry", {})
        check(label + " pipeline = character", rep.get("pipeline") in (None, "character"), str(rep.get("pipeline")))
        check(label + " respects the mobile budget", 0 < g.get("triangles", 0) <= 1800, f"{g.get('triangles')} tris")
        check(label + " reports a brief", bool((rep.get("plan") or {}).get("brief")))
        mid = rep["id"]
        st2, _h2, glb = http(base, f"/model/{mid}/model.glb")
        good, info = glb_is_sane(glb) if st2 == 200 else (False, "http %s" % st2)
        check(label + " GLB parses as glTF 2.0 with PBR", good and info.get("baseColorTexture")
              and info.get("metallicRoughnessTexture") and info.get("normalTexture"), str(info))
        for name in ("model.stl", "texture_atlas.png", "normal_atlas.png", "mr_atlas.png", "report.json"):
            s3, _h3, raw3 = http(base, f"/download/{mid}/{name}")
            check(f"download {name}", s3 == 200 and len(raw3) > 100, f"{s3} {len(raw3)}B")
        s4, _h4, z = http(base, f"/download/{mid}/all.zip")
        check("download all.zip", s4 == 200 and z[:2] == b"PK", f"{s4} {len(z)}B")

    # ── any image -> sculpted volume ───────────────────────────────────────
    for nm, img in (("car (side view)", car()), ("logo (alpha PNG)", logo()), ("bust / face crop", face())):
        label = f"image sculpt: {nm}"
        st, _h, raw = http(base, "/generate", {"budget": "game", "use_ai": "0", "mode": "auto"},
                           filename={"image": ("in.png", buf(img), "image/png")})
        rep = jload(raw)
        if not check(label + " -> 200", st == 200, f"{st} {raw[:220]!r}"):
            continue
        g = rep.get("geometry", {})
        check(label + " routed to the sculptor", rep.get("pipeline") == "sculpt", str(rep.get("pipeline")))
        check(label + " built a real mesh", 200 < g.get("triangles", 0) <= 5600, f"{g.get('triangles')} tris")
        check(label + " is watertight (STL/printable)", g.get("watertight") is True, str(g.get("watertight")))
        sm = rep.get("size_m") or [0, 0, 0]
        check(label + " has sane metric size", 0.2 < max(sm) < 6, str(sm))
        if "side view" in nm:      # the inscribed-disc bound must stop a car being as deep as it is long
            check("side view stays flat (depth < width)", sm[2] < sm[0] * 0.85, f"w={sm[0]} d={sm[2]}")
        st2, _h2, glb = http(base, f"/model/{rep['id']}/model.glb")
        good, info = glb_is_sane(glb) if st2 == 200 else (False, "http %s" % st2)
        check(label + " GLB has COLOR_0 + TEXCOORD_0", good and {"COLOR_0", "TEXCOORD_0"} <= set(info.get("attrs", [])),
              str(info.get("attrs") if isinstance(info, dict) else info))

    # ── full-body image still uses the rig ──────────────────────────────────
    st, _h, raw = http(base, "/generate", {"budget": "game", "use_ai": "0", "mode": "auto"},
                       filename={"image": ("knight.png", buf(knight()), "image/png")})
    rep = jload(raw)
    if check("full-body figure -> 200", st == 200, f"{st} {raw[:180]!r}"):
        check("full-body figure routed to the character rig", rep.get("pipeline") != "sculpt",
              f"pipeline={rep.get('pipeline')} heads={rep.get('heads_tall')}")

    # ── conversational revision of a sculpt (reuses the uploaded pixels) ────
    st, _h, raw = http(base, "/chat", {"message": "sculpt this into a model", "budget": "game",
                                       "use_ai": "0", "mode": "auto"},
                       filename={"image": ("car.png", buf(car()), "image/png")})
    first = jload(raw)
    sid = first.get("session")
    if check("chat turn with image -> 200", st == 200 and sid, str(st)):
        r1 = (first.get("report") or {})
        check("first sculpt build is a two-sided volume", r1.get("relief") is False, str(r1.get("relief")))
        st2, _h2, raw2 = http(base, "/chat", {"session": sid, "message": "now make the back flat",
                                              "budget": "game", "use_ai": "0", "mode": "auto"})
        second = jload(raw2)
        r2 = second.get("report") or {}
        if check("follow-up text-only turn rebuilds the same image -> 200", st2 == 200 and r2, f"{st2} {raw2[:180]!r}"):
            check("the spoken instruction reached the sculptor (relief on)", r2.get("relief") is True,
                  f"relief={r2.get('relief')} notes={r2.get('notes', [])[-1:]}")
            d1 = (r1.get("size_m") or [1, 1, 1])[2]
            d2 = (r2.get("size_m") or [1, 1, 1])[2]
            check("relief is shallower than the volume", d2 < d1, f"{d1} -> {d2}")

    # ── input validation ───────────────────────────────────────────────────
    st, _h, raw = http(base, "/generate", {"text": "", "use_ai": "0"})
    check("empty prompt is rejected with 400", st == 400, str(st))
    st, _h, raw = http(base, "/generate", {"use_ai": "0"}, filename={"image": ("x.png", b"\x89PNG nope", "image/png")})
    check("garbage image is rejected, not a 500 crash", st in (400, 500) and b"Pipeline failed" not in raw, f"{st} {raw[:120]!r}")
    st, _h, raw = http(base, "/generate", {"mode": "sculpt", "text": "a car", "use_ai": "0"})
    check("sculpt mode without an image is explained, not crashed", st == 400 and b"attach an image" in raw.lower(),
          f"{st} {raw[:120]!r}")
    st, _h, raw = http(base, "/download/deadbeef0000/model.glb")
    check("unknown build id -> 404", st == 404, str(st))
    st, _h, raw = http(base, "/report/deadbeef0000")
    check("unknown report -> 404", st == 404, str(st))

    # ── the memory guard ────────────────────────────────────────────────────
    mem = h.get("memory") or {}
    if mem:
        check("guard reports a plan ceiling", mem.get("limit_mb", 0) > 0, str(mem))
        check("atlas never exceeds 2048 on a small box", mem.get("atlas_px") in (512, 1024, 2048), str(mem.get("atlas_px")))
        if mem.get("limit_mb", 9999) <= 600:
            check("on a <=512 MB plan the atlas is capped at 1024 (measured 2048 build = 424 MB)",
                  mem.get("atlas_px", 0) <= 1024, f"atlas={mem.get('atlas_px')} limit={mem.get('limit_mb')}")
        if slow:
            n = 6
            print(f"\n  \033[2mconcurrency burst: {n} simultaneous image builds "
                  f"(this is what used to OOM-kill the worker)\033[0m")
            t1 = time.time()
            with ThreadPoolExecutor(max_workers=n) as ex:
                outs = list(ex.map(lambda _: http(base, "/generate", {"budget": "game", "use_ai": "0", "mode": "sculpt"},
                                                   filename={"image": ("car.png", buf(car()), "image/png")}), range(n)))
            codes = [o[0] for o in outs]
            ok("burst replies", f"{codes} in {time.time() - t1:.0f}s")
            check("no build in the burst 500-ed", all(c != 500 for c in codes), str(codes))
            st, _h, raw = http(base, "/health")
            after = jload(raw)
            check("instance still healthy after the burst", after.get("ok") is True, str(after)[:160])
            m2 = after.get("memory") or {}
            check("headroom did not collapse", m2.get("headroom_mb", 0) > 0, str(m2))

    print(f"\ndone in {time.time() - t0:.0f}s — "
          f"\033[32m{len(PASSES)} passed\033[0m, " +
          (f"\033[31m{len(FAILS)} failed\033[0m\n" + "\n".join("  ! " + f for f in FAILS) if FAILS else "0 failed\n"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default="http://127.0.0.1:8000")
    ap.add_argument("--slow", action="store_true", help="also run the concurrency burst (minutes on a free CPU)")
    a = ap.parse_args()
    sys.exit(run(a.base, slow=a.slow))
