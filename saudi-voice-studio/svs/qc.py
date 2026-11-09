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


def pause_noise_db(x: np.ndarray) -> float | None:
    """Noise level in genuine low-level pauses: runs ≥150 ms whose level is 35-70 dB under the speech peak (digital silence and the
    short fade edges of syllables are ignored). None when the audio has no such pause (nothing measurable)."""
    fr = _frames(x, 10.0)
    if len(fr) < 40:
        return None
    r = np.sqrt((fr ** 2).mean(axis=1))
    lv = _db(r)
    peak = lv.max()
    cand = (lv < peak - 35) & (lv > peak - 75) & (r > 1e-6)
    vals, run = [], []
    for i, c in enumerate(cand):
        if c:
            run.append(lv[i])
        else:
            if len(run) >= 15:
                vals.append(float(np.median(run)))
            run = []
    if len(run) >= 15:
        vals.append(float(np.median(run)))
    return float(np.median(vals)) if vals else None


F0_SPREAD_LIMIT = 1.5      # robust spread (semitones) allowed between sentence pitches when there are enough sentences
F0_ANCHOR_LIMIT = 2.5      # with few sentences: max distance (semitones) of any sentence from this voice's LEARNED typical narration pitch
F0_MIN_N = 5               # fewer sentences than this cannot give a trustworthy spread estimate


def f0_assess(f0s: list[float], baseline_f0: float | None) -> dict:
    """Statistically sound pitch-consistency assessment.
    * n >= 5: robust spread (1.4826 x MAD around the median) — outlier-resistant, unlike a plain std on a handful of values.
    * n < 5: a spread of 2-4 numbers is not meaningful, so each sentence is compared with a baseline instead: the voice's typical
      narration pitch learned from its previous takes (NOT the reference clip — measured: the model narrates 3-4 semitones above a calm
      reference reading, so the clip itself is a wrong anchor). With no baseline yet the check is honestly 'insufficient'.
    Returns {mode, n, center, spread, limit, ok, outliers: indices beyond limit}."""
    v = np.asarray([x for x in f0s if x is not None], dtype=float)
    n = len(v)
    if n >= F0_MIN_N:
        med = float(np.median(v))
        sd = float(1.4826 * np.median(np.abs(v - med)))
        out = [i for i, x in enumerate(f0s) if x is not None and abs(x - med) > max(2.0, 2.0 * sd)]
        return {"mode": "spread", "n": n, "center": med, "spread": round(sd, 2), "limit": F0_SPREAD_LIMIT,
                "ok": sd <= F0_SPREAD_LIMIT, "outlier_idx": out, "outlier_thr": 2.0}
    if n >= 1 and baseline_f0 is not None:
        dev = [None if x is None else abs(x - baseline_f0) for x in f0s]
        worst = max(d for d in dev if d is not None)
        out = [i for i, d in enumerate(dev) if d is not None and d > F0_ANCHOR_LIMIT]
        return {"mode": "anchor", "n": n, "center": float(baseline_f0), "spread": round(worst, 2), "limit": F0_ANCHOR_LIMIT,
                "ok": worst <= F0_ANCHOR_LIMIT, "outlier_idx": out, "outlier_thr": F0_ANCHOR_LIMIT}
    return {"mode": "insufficient", "n": n, "center": None, "spread": None, "limit": None, "ok": True, "outlier_idx": [], "outlier_thr": None}


def _eq_intent_max_db(filters: list[str] | None) -> float:
    import re as _re
    g = [abs(float(m)) for f in (filters or []) for m in _re.findall(r":g=(-?\d+(?:\.\d+)?)", f)]
    return float(max(g)) if g else 0.0


def _eq_intent_db(filters: list[str] | None) -> float:
    """Mean absolute gain over the 8 analysis bands implied by a list of ffmpeg `equalizer=...:g=X` filters (the INTENDED tonal change)."""
    import re as _re
    g = [abs(float(m)) for f in (filters or []) for m in _re.findall(r":g=(-?\d+(?:\.\d+)?)", f)]
    return float(sum(g) / 8.0) if g else 0.0


def transparency(pre: np.ndarray, post: np.ndarray, *, ref: np.ndarray | None = None, intended_db: float = 0.0,
                 intended_max_db: float | None = None, sim: float | None = None) -> dict:
    """Is the mastered audio still the same voice, only polished? Separates INTENDED tonal change (declared via `intended_db`,
    e.g. reference-matching EQ) from UNWANTED change, and checks artefacts that speaker similarity cannot see
    (muffling, hiss, clipping, crushed dynamics, added reverb). Speaker similarity alone is NOT used as the verdict.
    Every limit is fixed and documented; `intended_db` only raises the tonal budget by what was deliberately applied."""
    from . import audio
    n = min(len(pre), len(post))
    pre, post = np.asarray(pre[:n], dtype=np.float32), np.asarray(post[:n], dtype=np.float32)
    mid = [i for i, (c, _) in enumerate(audio._EQ_BANDS) if c in (500, 1000)]

    def norm(x):
        b = audio._band_levels(x, SR)
        return b - b[mid].mean()
    bp, bq = norm(pre), norm(post)
    td = float(np.abs(bp - bq).mean())
    reasons, info = [], {"tonal_change_db": round(td, 2), "intended_db": round(float(intended_db), 2)}
    limit = 1.5 + float(intended_db)
    mx = float(np.abs(bp - bq).max())
    imax = float(intended_max_db if intended_max_db is not None else intended_db * 3.0)
    mlimit = 4.0 + imax
    info["max_band_change_db"] = round(mx, 2)
    if td > limit:
        reasons.append(f"تغيّر جرس غير مبرَّر: {td:.1f} dB (المسموح {limit:.1f} = 1.5 + المقصود {intended_db:.1f})")
    elif mx > mlimit:
        reasons.append(f"تغيّر حادّ في نطاق واحد: {mx:.1f} dB (المسموح {mlimit:.1f} = 4.0 + المقصود {imax:.1f})")
    if ref is not None:
        br = norm(np.asarray(ref, dtype=np.float32))
        d_pre, d_post = float(np.abs(bp - br).mean()), float(np.abs(bq - br).mean())
        info.update(dist_to_reference_before=round(d_pre, 2), dist_to_reference_after=round(d_post, 2))
        if d_post > d_pre + 0.5:
            reasons.append(f"ابتعد الجرس عن مقطعك المرجعي ({d_pre:.1f} ← {d_post:.1f} dB)")
    # noise relative to speech (loudness changes must not count as noise changes)
    def snr(x):
        nz = pause_noise_db(x)
        return None if nz is None else float(audio.active_rms_db(x) - nz)
    s0, s1 = snr(pre), snr(post)
    if s0 is not None and s1 is not None:
        info.update(snr_before=round(s0, 1), snr_after=round(s1, 1))
        if s0 - s1 > 3.0:
            reasons.append(f"زاد الضجيج نسبةً للكلام ({s0:.0f} ← {s1:.0f} dB)")
    r0, r1 = reverb_tail_db(pre), reverb_tail_db(post)
    if r0 is not None and r1 is not None and r1 - r0 > 6.0:
        reasons.append(f"زاد ذيل الصدى ({r0:.0f} ← {r1:.0f} dB)")
    clip = float(np.mean(np.abs(post) > 0.99))
    if clip > 0:
        reasons.append(f"تقطيع في {clip * 100:.3f}% من العينات")
    crest = lambda x: float(20 * np.log10(max(np.abs(x).max(), 1e-9)) - audio.active_rms_db(x))
    c0, c1 = crest(pre), crest(post)
    info.update(crest_before=round(c0, 1), crest_after=round(c1, 1))
    if c0 - c1 > 6.0:
        reasons.append(f"انسحقت الديناميكية (عامل الذروة {c0:.0f} ← {c1:.0f} dB)")
    if sim is not None:
        info["speaker_sim"] = round(float(sim), 3)
        if sim < 0.90:
            reasons.append(f"تغيّرت بصمة المتحدث ({sim:.2f} < 0.90)")
    return {"ok": not reasons, "reasons": reasons, **info}


def chain_intent(chain: str | None) -> tuple[float, float]:
    """(mean |gain| over the 8 analysis bands, max |gain|) that the ffmpeg filter chain was DESIGNED to apply:
    equalizer bands count for one band each, bass/treble shelves for two bands each. Everything else (loudnorm, gentle
    compression, high-pass) is not tonal intent and is judged by the artefact checks."""
    import re as _re
    gains, per_band = [], 0.0
    for m in _re.finditer(r"equalizer=[^,]*?:g=(-?\d+(?:\.\d+)?)", chain or ""):
        g = abs(float(m.group(1)))
        gains.append(g)
        per_band += g
    for m in _re.finditer(r"(?:bass|treble)=g=(-?\d+(?:\.\d+)?)", chain or ""):
        g = abs(float(m.group(1)))
        gains.append(g)
        per_band += 2 * g
    return (per_band / 8.0, max(gains) if gains else 0.0)
