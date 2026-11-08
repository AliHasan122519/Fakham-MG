"""One-call automatic voice-over: text in, finished MP3 out, using only free local AI models (no API, no login).

Stages (all automatic): meaning analysis (semantic embeddings) → pronunciation + context diacritics → emotions →
delivery plan → synthesis in YOUR voice (default voice) with best-of-N selection by speaker similarity (when compute
allows) and speech-recognition safety net → adaptive mastering.
"""
from __future__ import annotations

from typing import Callable

from . import enhance, render, voices


def run(text: str = "", *, voice: str | None = None, dialect: str | None = None, max_quality: bool = False,
        progress: Callable[[float, str], None] | None = None, plan: dict | None = None) -> dict:
    progress = progress or (lambda p, m: None)
    voice = voice or voices.default_voice_id()
    if plan is None:
        progress(0.02, "تحليل المعنى وتجهيز الإلقاء")
        plan = enhance.smart_enhance(text, voice=voice, dialect=dialect, mode="auto", diacritics="selective", compare_raw=False)
    cand = render.default_candidates()
    if max_quality:
        cand = max(cand, 3)
    return render.render(plan, progress=progress, candidates=cand)
