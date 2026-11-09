"""Automatic quality control after every generation (measured, not assumed).

Per-sentence and whole-file checks; flagged sentences are re-synthesised with new seeds by the renderer (see render.py)
instead of re-running the whole text. Every check reports ok / warn / fixed / skipped with the numbers behind it.
"""
from __future__ import annotations

import difflib
import re

import numpy as np

SR = 24000
_DIAC = re.compile("[ً-ٰٟـ]")


def arabic_chars(text: str) -> int:
    return len(re.findall(r"[ء-ي]", _DIAC.sub("", text)))


def _frames(x: np.ndarray, ms: float = 10.0):
    n = int(SR * ms / 1000)
    k = len(x) // n
    return x[: k * n].reshape(k, n) if k else np.zeros((0, n))


def _db(v):
    return 20 * np.log10(np.maximum(v, 1e-9))


def speech_seconds(x: np.ndarray) -> float:
    """Duration between the first and last frame above -40 dB of the peak (ignores leading/trailing silence)."""
    fr = _frames(x)
    if len(fr) == 0:
        return 0.0
    lv = _db(np.sqrt((fr ** 2).mean(axis=1)))
    idx = np.where(lv > lv.max() - 40)[0]
    return float((idx[-1] - idx[0] + 1) * 0.01) if len(idx) else 0.0


def chars_per_second(wave: np.ndarray, text: str) -> float | None:
    n = arabic_chars(text)
    d = speech_seconds(wave)
    return (n / d) if (n >= 10 and d > 0.5) else None


def reverb_tail_db(x: np.ndarray) -> float | None:
    """Median level of 80-250 ms after sharp speech offsets relative to the speech just before them (dB).
    A dry studio take falls to the noise floor (≲ -30); a reverberant room keeps a decaying tail (≳ -18).
    None if no clean offset (sharp drop followed by a pause, not by more speech) is found."""
    fr = _frames(x, 10.0)
    if len(fr) < 45:
        return None
    lv = _db(np.sqrt((fr ** 2).mean(axis=1)))
    peak = lv.max()
    vals = []
    i = 6
    while i < len(lv) - 27:
        before = np.mean(lv[i - 5:i])
        if before > peak - 25 and lv[i - 5:i].min() > before - 6 and lv[i + 1] < before - 5:
            seg = lv[i + 3:i + 26]
            if seg.max() < before - 3:                  # a pause, not the next syllable
                vals.append(float(np.mean(lv[i + 8:i + 25]) - before))
                i += 30
                continue
        i += 1
    return float(np.median(vals)) if vals else None


def noise_floor_db(x: np.ndarray) -> float | None:
    """Level of the quietest non-silent frames (digital silence excluded)."""
    fr = _frames(x)
    if len(fr) < 20:
        return None
    r = np.sqrt((fr ** 2).mean(axis=1))
    r = r[r > 1e-5]
    return float(_db(np.percentile(r, 10))) if len(r) > 10 else None


def word_diff(target: str, heard: str) -> dict:
    """Missing/duplicated words between the text that should be spoken and what a recognizer heard."""
    def toks(s):
        s = _DIAC.sub("", s)
        s = re.sub("[إأآٱ]", "ا", s).replace("ى", "ي").replace("ة", "ه")
        return re.sub(r"[^ء-ي ]", " ", s).split()
    a, b = toks(target), toks(heard)
    if not a:
        return {"missing": 0.0, "dup": 0}
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(bl.size for bl in sm.get_matching_blocks())
    # near matches (recognizer spelling noise): count a missing word as matched if a close word exists in the heard text
    rest = [w for w in a if w not in b]
    close = sum(1 for w in rest if difflib.get_close_matches(w, b, n=1, cutoff=0.75))
    missing = max(0.0, (len(a) - matched - close) / len(a))
    dup = sum(1 for i in range(1, len(b)) if b[i] == b[i - 1] and len(b[i]) > 2 and not any(
        a[j] == a[j + 1] == b[i] for j in range(len(a) - 1)))
    return {"missing": round(missing, 3), "dup": dup}


def status(ok: bool, warn_only: bool = False) -> str:
    return "ok" if ok else ("warn" if warn_only else "fail")
