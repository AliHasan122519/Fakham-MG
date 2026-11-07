"""Background generation jobs (CPU generation takes minutes, so MCP/web poll instead of blocking)."""
from __future__ import annotations

import threading
import time
import uuid

from . import studio
from .engine import EngineUnavailable

JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def start(**kwargs) -> str:
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"id": jid, "state": "running", "progress": 0.0, "message": "بدء", "started": time.time()}

    def run():
        job = JOBS[jid]
        try:
            res = studio.generate(progress=lambda p, m: job.update(progress=round(p, 2), message=m), **kwargs)
            job.update(state="done", result=res, progress=1.0, message="تم")
        except (studio.StudioError, EngineUnavailable, KeyError, ValueError) as e:
            job.update(state="error", error=str(e))
        except Exception as e:  # pragma: no cover
            job.update(state="error", error=f"خطأ غير متوقع: {type(e).__name__}: {e}")
        job["elapsed"] = round(time.time() - job["started"], 1)

    threading.Thread(target=run, daemon=True).start()
    return jid


def get(jid: str) -> dict | None:
    return JOBS.get(jid)


def wait(jid: str, timeout: float) -> dict:
    end = time.time() + timeout
    while time.time() < end and JOBS[jid]["state"] == "running":
        time.sleep(0.5)
    return JOBS[jid]
