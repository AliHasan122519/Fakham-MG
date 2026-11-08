"""Background generation jobs (CPU generation takes minutes, so MCP/web poll instead of blocking)."""
from __future__ import annotations

import threading
import time
import uuid

from . import studio
from .engine import EngineUnavailable

JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
MAX_RUNNING = 2     # a leaked link must not be able to queue unlimited work on your GPU/CPU
MAX_KEPT = 200


def running_count() -> int:
    return sum(1 for j in JOBS.values() if j["state"] == "running")


def start(fn=None, **kwargs) -> str:
    """Run `fn(progress=..., **kwargs)` (default: studio.generate) in a background thread; poll with get()."""
    fn = fn or studio.generate
    with _LOCK:
        if len(JOBS) >= MAX_KEPT:
            for k in [k for k, j in JOBS.items() if j["state"] != "running"][: MAX_KEPT // 2]:
                JOBS.pop(k, None)
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"id": jid, "state": "running", "progress": 0.0, "message": "بدء", "started": time.time()}

    def run():
        job = JOBS[jid]
        try:
            res = fn(progress=lambda p, m: job.update(progress=round(p, 2), message=m), **kwargs)
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
