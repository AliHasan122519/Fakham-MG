"""One-call automatic voice-over: text in, finished MP3 out, using only free local AI models (no API, no login).

Stages (all automatic): meaning analysis (semantic embeddings) → pronunciation + context diacritics → emotions →
delivery plan → synthesis in YOUR voice (default voice) with best-of-N selection by speaker similarity (when compute
allows) and speech-recognition safety net → adaptive mastering.
"""
from __future__ import annotations

from typing import Callable

from . import enhance, render, voices

# genre → narrator (only when that narrator is available; otherwise the default voice reads it)
GENRE_NARRATOR = {"ad": "promo_energetic", "news": "anchor_formal", "announcement": "anchor_formal",
                  "documentary": "documentary_cinematic", "story": "narrator_deep"}


def run(text: str = "", *, voice: str | None = None, dialect: str | None = None, max_quality: bool = False,
        progress: Callable[[float, str], None] | None = None, plan: dict | None = None) -> dict:
    progress = progress or (lambda p, m: None)
    voice = voice or voices.default_voice_id()
    if voice == "auto":                    # pick the narrator from the detected type of text
        progress(0.02, "اختيار الراوي المناسب لنوع النص")
        base = enhance.smart_enhance(text, voice=voices.default_voice_id(), dialect=dialect, mode="auto",
                                     diacritics="none", compare_raw=False)
        cand = GENRE_NARRATOR.get(base["genre"])
        avail = {v["id"]: v for v in voices.list_voices()}
        voice = cand if cand in avail and avail[cand]["reference_source"] in ("derived", "own") else voices.default_voice_id()
    if plan is None:
        progress(0.02, "تحليل المعنى وتجهيز الإلقاء")
        plan = enhance.smart_enhance(text, voice=voice, dialect=dialect, mode="auto", diacritics="selective", compare_raw=False)
    cand = render.default_candidates()
    if max_quality:
        cand = max(cand, 3)
    return render.render(plan, progress=progress, candidates=cand)
