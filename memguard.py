"""memguard — keeps the pipeline inside the instance's memory ceiling.

Why this exists: Render's free plan is 0.1 CPU / **512 MB** and kills the whole
gunicorn worker when it goes over (the events API reports it as
`server_failed: {"oomKilled": {"memoryLimit": "512Mi"}}`, and on a free service
that shows up as a 502 for up to a minute while it restarts).  Measured peaks on
the atlas at `2048²`:

    1 build, 1024 atlas   311 MB     4 builds at once, 1024 atlas   512 MB
    1 build, 2048 atlas   488 MB     4 builds at once, 2048 atlas   864 MB  <- dead

Measured peaks (this repo, single request): 512 atlas -> 143 MB, 1024 -> 194 MB,
2048 -> 424 MB.  Render free = 512 MB, so 2048 must never be chosen there.

Three levers, all here:

  1. `slot()`      one build at a time (0.1 CPU cannot run four anyway)
  2. `tex_size()`  pick the biggest atlas that fits in the *actual* free memory
  3. `release()`   hand pages back to the OS after every build, so the high-water
                   mark of the next build is not stacked on top of the last one
"""

from __future__ import annotations

import ctypes
import gc
import logging
import math
import os
import threading
import time

log = logging.getLogger("forge.mem")

# Measured on this codebase (single build, 768x1024 input, RSS high-water):
#     imports only            105 MB
#     atlas   512            143 MB   ->  +38 MB
#     atlas  1024            194 MB   ->  +89 MB
#     atlas  2048            424 MB   -> +319 MB   <- this is what OOMs a 512 MB box
# so an atlas pixel really costs ~88 bytes all-in (colour + height + normal + the
# float temporaries of Sobel/gradient), and a build carries ~100 MB of mesh,
# decoded photo and export buffers around it.
ATLAS_BYTES_PER_PX = 88
EXTRA_BUILD_MB = 100             # everything the atlas is not
SAFETY = 0.8                     # plan to use at most this share of what is free
CANDIDATES = (2048, 1024, 512)   # powers of two keep the 4x4 atlas tiles integral
FORCED = int(os.environ.get("FORGE_MAX_TEXTURE", "0") or 0)   # 0 = auto, else pinned

_LIMIT_CACHE: list = []


def _read_int(path):
    try:
        with open(path) as fh:
            txt = fh.read().strip()
        if txt in ("max", ""):
            return 0
        return int(txt)
    except Exception:                                          # noqa: BLE001
        return 0


def _stat_key(path, key):
    try:
        with open(path) as fh:
            for line in fh:
                p = line.split()
                if len(p) == 2 and p[0] == key:
                    return int(p[1])
    except Exception:                                          # noqa: BLE001
        pass
    return 0


def limits():
    """(limit_mb, current_mb, headroom_mb) for this process, cgroup-aware.

    Render runs each service in its own cgroup, so `memory.max` is the plan's
    real ceiling (512 MB on free).  Where there is no cgroup (a laptop, this
    sandbox) we fall back to RLIMIT_AS, then to 60 % of MemTotal, and only then
    to a conservative 512 MB - so the guard never *invents* a limit on a dev
    box, and never trusts a missing one on Render.
    """
    if _LIMIT_CACHE:
        limit = _LIMIT_CACHE[0]                          # the ceiling is fixed; usage is not
        return _report(limit)
    unbounded = 64 * 1024 ** 3
    limit = _read_int("/sys/fs/cgroup/memory.max") or _read_int(
        "/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if limit <= 0 or limit > unbounded:                 # "max" / 0 / RLIM_INFINITY
        limit = 0
        try:
            import resource
            soft = resource.getrlimit(resource.RLIMIT_AS)[0]
            if soft and soft > 0:
                limit = soft if soft <= unbounded else 0
        except Exception:                               # noqa: BLE001
            pass
    if limit <= 0:
        total = 0
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) * 1024
                        break
        except Exception:                               # noqa: BLE001
            total = 0
        limit = int(total * 0.6) if total else 512 * 1024 ** 2
        log.info("no cgroup memory limit - assuming %.0f MB", limit / 1024 ** 2)
    override = os.environ.get("FORGE_MEMORY_MB")
    if override:                                        # local testing / stricter caps
        limit = int(float(override)) * 1024 ** 2

    _LIMIT_CACHE[:] = [limit]
    return _report(limit)


def _report(limit):
    """(limit_mb, current_mb, headroom_mb) with usage measured right now."""
    used = _read_int("/sys/fs/cgroup/memory.current") or _read_int(
        "/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if used:
        # page cache counts against the cgroup but is reclaimable, so leaving it in
        # would shrink the atlas for no reason - subtract the inactive file cache
        used = max(0, used - _stat_key("/sys/fs/cgroup/memory.stat", "inactive_file"))
    else:
        try:
            with open(f"/proc/{os.getpid()}/statm") as fh:
                used = int(fh.read().split()[1]) * 4096
        except Exception:                                 # noqa: BLE001
            used = 0
    mb = 1024.0 ** 2
    return limit / mb, used / mb, max(0.0, (limit - used) / mb)


def tex_size(desired=1024, extra_mb=EXTRA_BUILD_MB):
    """Largest atlas that fits, given what is free *right now*.

    `extra_mb` reserves room for the mesh, the decoded photo and the export
    buffers that live alongside the atlas.
    """
    if FORCED:
        return max([c for c in CANDIDATES if c <= FORCED] or [CANDIDATES[-1]])
    _limit, _used, head = limits()
    free = max(0.0, head - extra_mb) * 1024.0 ** 2 * SAFETY
    want = int(desired)
    out = CANDIDATES[-1]                                   # smallest, always allowed
    for cand in CANDIDATES:                                # descending: 2048, 1024, 512
        if cand <= want and cand * cand * ATLAS_BYTES_PER_PX <= free:
            out = cand
            break
    if out < want:
        log.info("texture %d -> %d (headroom %.0f MB)", want, out, head)
    return out


# ---------------------------------------------------------------------------
# one build at a time
# ---------------------------------------------------------------------------
class Busy(RuntimeError):
    """Raised when the forge is already building; the HTTP layer turns it into 429."""

    def __init__(self, wait_s):
        super().__init__(
            "The forge is busy building another character (the free instance has "
            "512 MB and one CPU, so builds run one at a time). "
            f"Try again in a few seconds - usually under {max(2, int(wait_s))} s.")
        self.retry_after = max(3, int(wait_s + 2))


_LOCK = threading.Lock()
_WAITERS = [0]
SLOT_WAIT_S = float(os.environ.get("FORGE_QUEUE_WAIT_S", "75"))
SLOT_ENABLED = os.environ.get("FORGE_SERIALISE_BUILDS", "1") not in ("0", "false", "off")


class slot:
    """`with slot():` — hold the single build slot, or raise Busy quickly.

    Queuing is better than refusing: a browser retry storm on a cold, sleeping
    free instance would otherwise 429 everybody.  After FORGE_QUEUE_WAIT_S we
    give up and the caller gets a clear, retryable error instead of an OOM kill.
    """

    def __init__(self, timeout=None):
        self.timeout = SLOT_WAIT_S if timeout is None else float(timeout)
        self.have = False

    def __enter__(self):
        if not SLOT_ENABLED:
            return self
        t0 = time.time()
        _WAITERS[0] += 1
        try:
            if not _LOCK.acquire(timeout=self.timeout):
                raise Busy(time.time() - t0)
            self.have = True
        finally:
            _WAITERS[0] -= 1
        return self

    def __exit__(self, *exc):
        if self.have:
            _LOCK.release()
        return False


def status():
    """For /health: what the guard currently sees."""
    limit, used, head = limits()
    return {"limit_mb": round(limit), "used_mb": round(used), "headroom_mb": round(head),
            "atlas_px": tex_size(2048), "queued": _WAITERS[0],
            "serialised": SLOT_ENABLED}


def release():
    """Return freed pages to the allocator; RSS does not shrink on its own.

    Without this the next build starts from the previous build's high-water mark
    and one big export after another walks straight into the OOM killer.
    """
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:                                          # noqa: BLE001
        pass


def footprint_mb(arrays=()):
    return sum(a.nbytes for a in arrays if a is not None) / 1024.0 ** 2


def budget_check(need_mb):
    """True if `need_mb` looks safe right now (used by callers that can degrade)."""
    _l, _u, head = limits()
    return head * SAFETY >= need_mb
