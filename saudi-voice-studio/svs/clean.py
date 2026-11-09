"""Final-stage cleanup: de-reverberation + denoise, measured and GUARDED.

Both steps are always *evaluated*; they are only *applied* when the measurement shows a problem, and an applied result is kept only
if it passes guards (tonal balance preserved, level not collapsed). A clean, dry model output therefore passes through unchanged and
keeps its warmth; a reverberant or noisy one (e.g. a processed/echoey reference) is repaired and the report says exactly what changed.

De-reverberation = late-reverberation spectral suppression (Lebart et al. 2001 / Habets 2007 statistical model): the late tail of
each STFT bin is predicted from the same bin ~50 ms earlier with an exponential decay set by an estimated T60, and subtracted with a
gain floor. It is a classic, dependency-free (numpy+scipy) method — NOT a neural dereverb — so its limits are stated in the report.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft

from . import qc

SR = 24000
REVERB_APPLY_DB = -26.0       # tail above this (re speech level) → treat as reverberant and de-reverberate
NOISE_APPLY_DB = -50.0        # pause noise above this (dBFS) → denoise (speech peak ≈ -14 dBFS, i.e. SNR ≲ 36 dB)


def estimate_t60(x: np.ndarray) -> float | None:
    """T60 from the decay of the level after sharp speech offsets (slope of dB over 30-250 ms). None if not measurable."""
    fr = qc._frames(x, 10.0)
    if len(fr) < 45:
        return None
    lv = qc._db(np.sqrt((fr ** 2).mean(axis=1)))
    peak = lv.max()
    slopes = []
    i = 6
    while i < len(lv) - 27:
        before = np.mean(lv[i - 5:i])
        if before > peak - 25 and lv[i - 5:i].min() > before - 6 and lv[i + 1] < before - 5 and lv[i + 3:i + 26].max() < before - 3:
            t = np.arange(3, 26) * 0.01
            y = lv[i + 3:i + 26]
            ok = y > before - 55
            if ok.sum() >= 8:
                k = np.polyfit(t[ok], y[ok], 1)[0]
                if k < -5:
                    slopes.append(k)
            i += 30
            continue
        i += 1
    if not slopes:
        return None
    return float(np.clip(-60.0 / np.median(slopes), 0.15, 1.5))


def dereverb(x: np.ndarray, t60: float, *, strength: float = 2.2, late_ms: float = 60.0, floor: float = 0.10) -> np.ndarray:
    """Late-reverberation suppression with a Wiener-style gain."""
    n_fft, hop = 1024, 256
    f, t, X = stft(x, SR, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros")
    P = np.abs(X) ** 2
    d = max(1, int(round(late_ms / 1000 * SR / hop)))
    delta = 3.0 * np.log(10) / max(t60, 0.1)
    decay = np.exp(-2.0 * delta * (d * hop / SR))
    late = np.zeros_like(P)
    late[:, d:] = decay * P[:, :-d]
    # smooth the late-tail estimate over a few frames so it does not follow single syllables
    k = np.ones(5) / 5
    late = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, late) * strength
    gain = np.clip(1.0 - late / np.maximum(P, 1e-12), floor ** 2, 1.0)
    gain = np.sqrt(gain)
    # temporal smoothing of the gain avoids musical-noise warbling
    gain = np.apply_along_axis(lambda r: np.convolve(r, np.ones(3) / 3, mode="same"), 1, gain)
    _, y = istft(X * gain, SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=True)
    y = y[: len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float32)


def _tonal_distance(a: np.ndarray, b: np.ndarray) -> float:
    from . import audio
    la, lb = audio._band_levels(a, SR), audio._band_levels(b, SR)
    mid = [i for i, (c, _) in enumerate(audio._EQ_BANDS) if c in (500, 1000)]
    return float(np.abs((la - la[mid].mean()) - (lb - lb[mid].mean())).mean())


def polish(x: np.ndarray, *, force_dereverb: bool = False, force_denoise: bool = False) -> tuple[np.ndarray, dict]:
    """Returns (cleaned, report). Report keys: reverb_before/after (dB), noise_before/after (dBFS), applied[], skipped[]."""
    x = np.asarray(x, dtype=np.float32)
    rep: dict = {"applied": [], "skipped": []}
    rt0 = qc.reverb_tail_db(x)
    nf0 = qc.pause_noise_db(x)
    rep.update(reverb_before=None if rt0 is None else round(rt0, 1), noise_before=None if nf0 is None else round(nf0, 1))
    y = x
    if force_dereverb or (rt0 is not None and rt0 > REVERB_APPLY_DB):
        t60 = estimate_t60(x) or 0.45
        cand = dereverb(x, t60)
        rt1 = qc.reverb_tail_db(cand)
        dist = _tonal_distance(x, cand)
        if rt1 is not None and rt0 is not None and rt1 < rt0 - 3 and dist <= 2.0:
            y = cand
            rep["applied"].append(f"إزالة الصدى (T60≈{t60:.2f}ث)")
        else:
            rep["skipped"].append("إزالة الصدى لم تجتز الحراسة (لا تحسّن كافٍ أو تغيّر الجرس)")
    if force_denoise or (nf0 is not None and nf0 > NOISE_APPLY_DB):
        from . import audio
        try:
            cand = audio.ffmpeg_wave(y, "afftdn=nr=10:nf=-50:tn=1,highpass=f=60")
            nf1 = qc.pause_noise_db(cand)
            dist = _tonal_distance(y, cand)
            if nf1 is not None and nf0 is not None and nf1 < nf0 - 2 and dist <= 1.5:
                y = cand
                rep["applied"].append("تقليل الضجيج")
            else:
                rep["skipped"].append("تقليل الضجيج لم يجتز الحراسة")
        except Exception as e:                                     # pragma: no cover
            rep["skipped"].append(f"تقليل الضجيج تعذّر: {e}")
    rt2, nf2 = qc.reverb_tail_db(y), qc.pause_noise_db(y)
    rep.update(reverb_after=None if rt2 is None else round(rt2, 1), noise_after=None if nf2 is None else round(nf2, 1))
    return y, rep
