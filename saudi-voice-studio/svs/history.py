"""Generation history (SQLite, local)."""
from __future__ import annotations

import sqlite3
import time
from contextlib import closing

from . import config


def _conn():
    c = sqlite3.connect(config.HISTORY_DB)
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS gens(
        id TEXT PRIMARY KEY, ts REAL, voice TEXT, dialect TEXT, emotion TEXT, speed REAL,
        text TEXT, seconds REAL, took REAL, device TEXT, engine TEXT)""")
    return c


def add(rec: dict) -> None:
    # `with sqlite3.connect()` only commits — it does NOT close the connection (file handle stays open until GC). Close explicitly.
    with closing(_conn()) as c, c:
        c.execute("INSERT OR REPLACE INTO gens VALUES(:id,:ts,:voice,:dialect,:emotion,:speed,:text,:seconds,:took,:device,:engine)",
                  {**rec, "ts": time.time()})


def recent(limit: int = 20) -> list[dict]:
    with closing(_conn()) as c:
        return [dict(r) for r in c.execute("SELECT * FROM gens ORDER BY ts DESC LIMIT ?", (limit,))]


def get(gid: str) -> dict | None:
    with closing(_conn()) as c:
        r = c.execute("SELECT * FROM gens WHERE id=?", (gid,)).fetchone()
        return dict(r) if r else None
