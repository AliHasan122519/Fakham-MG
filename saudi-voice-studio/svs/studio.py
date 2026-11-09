"""High-level entry points shared by the web UI and the MCP server (v2: plan-based)."""
from __future__ import annotations

from typing import Callable

from . import config, delivery, plan as planmod, render


class StudioError(RuntimeError):
    pass


def analyze(text: str, *, voice: str = "narrator_deep", dialect: str | None = None, style: str | None = None,
            mode: str = "auto", controls: dict | None = None, emotion: str | None = None) -> dict:
    try:
        return planmod.build(text, voice=voice, dialect=dialect, style=style or None, mode=mode,
                             controls=controls, emotion=emotion)
    except (ValueError, KeyError) as e:
        raise StudioError(str(e))


def generate(text: str = "", voice_id: str = "narrator_deep", dialect: str | None = None, emotion: str = "auto",
             speed: float | None = None, fmt: str = "both", progress: Callable | None = None, seed: int | None = None,
             plan: dict | None = None, style: str | None = None, controls: dict | None = None,
             compare_raw: bool | None = None, candidates: int | None = None) -> dict:
    """Render either an explicit (edited) plan or text analysed on the fly.

    emotion='auto' → detect per sentence; any concrete emotion key → manual mode with that emotion everywhere."""
    try:
        if plan is None:
            ctrl = dict(controls or {})
            if speed:
                ctrl["speed"] = speed
            auto = emotion in ("auto", "", None)
            plan = planmod.build(text, voice=voice_id, dialect=dialect, style=style or None,
                                 mode="auto" if auto else "manual", controls=ctrl,
                                 emotion=None if auto else emotion)
        take = render.render(plan, fmt=fmt, progress=progress, compare_raw=compare_raw, candidates=candidates)
    except (ValueError, KeyError, render.RenderError) as e:
        raise StudioError(str(e))
    return take


def take_view(take: dict) -> dict:
    """Compact, UI/MCP-friendly summary of a take."""
    return {"id": take["id"], "parent": take.get("parent"), "seconds": take["metrics"]["after"]["duration_s"],
            "took": take["took_s"], "device": take["device"], "engine": take["engine"],
            "realtime_factor": take["realtime_factor"], "dialect": take["plan"]["dialect"],
            "dialect_native": take["dialect_native"], "dialect_note": take["dialect_note"],
            "voice": take["plan"]["voice"], "style": take["plan"]["style"], "mode": take["plan"]["mode"],
            "warnings": take["warnings"], "metrics": take["metrics"], "cache_hits": take["cache_hits"],
            "intelligibility": take.get("intelligibility", {}), "intelligibility_compare": take.get("intelligibility_compare"),
            "baseline_kind": take.get("baseline_kind"), "qc": take.get("qc"), "draft": bool(take.get("draft")), "voice_match": take.get("voice_match"),
            "segments": [{"id": s["id"], "text": s["text"], "corrected": s.get("corrected"), "vocalized": s.get("vocalized"), "spoken": s["spoken"], "emotion": s["emotion"],
                          "intensity": s["intensity"], "params": s["params"], "seed": s["seed"]}
                         for s in take["plan"]["segments"]]}
