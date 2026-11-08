"""Optional speech-to-text (faster-whisper, MIT) used as a *safety net*, never as a dependency of synthesis:
  * verifies that a reference clip really contains the transcript the user typed (mismatch = garbled voice),
  * transcribes a clip when the user has no transcript,
  * checks that generated speech is intelligible.
Everything degrades gracefully when faster-whisper is not installed."""
from __future__ import annotations

import difflib
import os
import re
import threading

import numpy as np

from . import config

_LOCK = threading.Lock()
_MODEL = None
MODEL_NAME = os.environ.get("SVS_WHISPER_MODEL", "small")   # small: CPU-friendly; medium/large-v3 on a GPU give better dialect transcripts (untested here)


def available() -> bool:
    if os.environ.get("SVS_ASR", "auto") == "off":
        return False
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def _model():
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            from faster_whisper import WhisperModel
            try:
                import torch
                gpu = torch.cuda.is_available()
            except Exception:
                gpu = False
            _MODEL = WhisperModel(MODEL_NAME, device="cuda" if gpu else "cpu",
                                  compute_type="float16" if gpu else "int8",
                                  download_root=str(config.MODEL_DIR / "whisper"))
    return _MODEL


def transcribe(wave16k: np.ndarray, *, words: bool = False) -> dict:
    """wave16k: float32 mono @16 kHz. Returns {text, words:[{w,start,end}]}."""
    segs, _ = _model().transcribe(np.asarray(wave16k, dtype=np.float32), language="ar", beam_size=3,
                                  word_timestamps=words, vad_filter=False, condition_on_previous_text=False)
    segs = list(segs)
    out = {"text": " ".join(s.text.strip() for s in segs).strip(), "words": []}
    if words:
        out["words"] = [{"w": w.word.strip(), "start": float(w.start), "end": float(w.end)} for s in segs for w in (s.words or [])]
    return out


_DIAC = re.compile("[ً-ٰٟـ]")


def norm(s: str) -> str:
    s = _DIAC.sub("", s)
    s = re.sub("[إأآٱ]", "ا", s).replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return re.sub(r"[^ء-ي ]", "", s).strip()


def similarity(a: str, b: str) -> float:
    """Character-level similarity of two Arabic strings (diacritics/punctuation ignored). autojunk off on purpose:
    the default heuristic makes difflib wildly inaccurate on strings longer than ~200 chars."""
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    return float(difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio())
