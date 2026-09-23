"""Optional Postgres persistence (Neon) for generated characters.

Render's free disk is ephemeral, so finished models + reports are mirrored into
Postgres when DATABASE_URL is set.  Everything degrades gracefully to disk-only
when it is not (or when the database is unreachable).
"""

import json
import logging
import os

log = logging.getLogger("forge.db")
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
KEEP = int(os.environ.get("FORGE_KEEP_ROWS", "80"))   # stay well inside a free tier

_SCHEMA = """
CREATE TABLE IF NOT EXISTS characters (
    id          TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    source      TEXT NOT NULL,
    prompt      TEXT,
    style       TEXT,
    triangles   INTEGER,
    report      JSONB NOT NULL,
    glb         BYTEA NOT NULL,
    atlas       BYTEA,
    stl         BYTEA
);
CREATE INDEX IF NOT EXISTS characters_created_idx ON characters (created_at DESC);
"""


def enabled():
    return bool(DATABASE_URL)


def _connect():
    import psycopg
    return psycopg.connect(DATABASE_URL, connect_timeout=10)


def init():
    if not enabled():
        log.info("DATABASE_URL not set - running disk-only")
        return False
    try:
        with _connect() as con:
            con.execute(_SCHEMA)
        log.info("database ready")
        return True
    except Exception as e:                       # noqa: BLE001
        log.warning("database unavailable (%s) - running disk-only", e)
        return False


def save(mid, report, model_dir):
    if not enabled():
        return False
    try:
        def rd(name):
            p = os.path.join(model_dir, name)
            return open(p, "rb").read() if os.path.exists(p) else None
        with _connect() as con:
            con.execute(
                """INSERT INTO characters
                   (id, source, prompt, style, triangles, report, glb, atlas, stl)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (id) DO NOTHING""",
                (mid, report["source"], report.get("prompt"), report.get("style"),
                 report["geometry"]["triangles"], json.dumps(report),
                 rd("model.glb"), rd("texture_atlas.png"), rd("model.stl")))
            # keep the table bounded
            con.execute(
                """DELETE FROM characters WHERE id IN (
                     SELECT id FROM characters ORDER BY created_at DESC OFFSET %s)""",
                (KEEP,))
        return True
    except Exception as e:                       # noqa: BLE001
        log.warning("db save failed: %s", e)
        return False


def recent(limit=12):
    if not enabled():
        return []
    try:
        with _connect() as con:
            rows = con.execute(
                """SELECT id, created_at, source, prompt, style, triangles
                   FROM characters ORDER BY created_at DESC LIMIT %s""", (limit,)).fetchall()
        return [dict(id=r[0], created_at=r[1].isoformat(), source=r[2],
                     prompt=r[3], style=r[4], triangles=r[5]) for r in rows]
    except Exception as e:                       # noqa: BLE001
        log.warning("db recent failed: %s", e)
        return []


def get_report(mid):
    if not enabled():
        return None
    try:
        with _connect() as con:
            row = con.execute("SELECT report FROM characters WHERE id=%s", (mid,)).fetchone()
        return row[0] if row else None
    except Exception as e:                       # noqa: BLE001
        log.warning("db get_report failed: %s", e)
        return None


def get_file(mid, name):
    """Return (bytes, mimetype) for a stored artifact, or None."""
    col = {"model.glb": ("glb", "model/gltf-binary"),
           "texture_atlas.png": ("atlas", "image/png"),
           "model.stl": ("stl", "application/octet-stream")}.get(name)
    if not enabled() or col is None:
        return None
    try:
        with _connect() as con:
            row = con.execute(f"SELECT {col[0]} FROM characters WHERE id=%s", (mid,)).fetchone()
        if row and row[0] is not None:
            return bytes(row[0]), col[1]
        return None
    except Exception as e:                       # noqa: BLE001
        log.warning("db get_file failed: %s", e)
        return None
