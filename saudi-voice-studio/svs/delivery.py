"""Delivery engine: turns (emotion, intensity, style, controls, context) into concrete prosody parameters.

What the current TTS (F5-TTS family) can really do per sentence:
  * speed       — model-level duration control (natural, no resampling)         -> REAL
  * pauses      — exact silence before/after each sentence                      -> REAL
  * gain/energy — per-sentence level (emphasis vs. softness)                    -> REAL
  * pitch       — small formant-preserving shift (rubberband), clamped ±2 st    -> REAL but limited
  * timbre/emotion colour — only from the reference clip (per-emotion references) -> REAL only if you provide them
It has NO built-in emotion switch. Everything below is explicit prosody shaping, not "emotion synthesis".
"""
from __future__ import annotations

import hashlib
import random

EMOTIONS = {
    #            ar        speed  pitch  gain  p_before p_after  nfe  var
    "neutral":    ("محايد",   1.00,  0.0,  0.0, 0.00, 0.35, 0, 1.0),
    "joy":        ("الفرح",   1.06,  0.5,  1.0, 0.00, 0.30, 0, 1.2),
    "sadness":    ("الحزن",   0.86, -0.7, -2.0, 0.10, 0.65, 0, 0.8),
    "excitement": ("الحماس",  1.12,  0.7,  1.5, 0.00, 0.22, 0, 1.4),
    "calm":       ("الهدوء",  0.92, -0.3, -1.0, 0.05, 0.55, 0, 0.7),
    "confidence": ("الثقة",   0.97, -0.2,  0.5, 0.00, 0.40, 0, 0.8),
    "grandeur":   ("الفخامة", 0.88, -0.8,  0.5, 0.15, 0.70, 0, 0.8),
    "empathy":    ("التعاطف", 0.93, -0.2, -1.0, 0.10, 0.50, 0, 0.8),
    "suspense":   ("التشويق", 0.90, -0.5, -1.5, 0.20, 0.65, 0, 1.0),
    "warning":    ("التحذير", 1.02,  0.2,  2.0, 0.00, 0.45, 0, 0.9),
    "formal":     ("الرسمية", 0.98, -0.2,  0.0, 0.00, 0.45, 0, 0.6),
}
ALIASES = {"enthusiastic": "excitement", "serious": "formal", "dramatic": "grandeur", "warm": "empathy"}

STYLES = {
    #             ar               speed pause variation  default_expressiveness
    "narration":    ("سرد",           0.97, 1.00, 1.0, 1.0),
    "ad":           ("إعلان",         1.08, 0.75, 1.2, 1.2),
    "news":         ("نشرة أخبار",    1.02, 0.85, 0.6, 0.7),
    "documentary":  ("وثائقي سينمائي", 0.90, 1.30, 0.8, 1.0),
    "story":        ("حكاية",         0.95, 1.15, 1.3, 1.3),
    "conversation": ("حوار",          1.03, 0.80, 1.4, 1.1),
}

DEFAULT_CONTROLS = {
    "speed": 1.0,            # global speed multiplier (0.7-1.3)
    "expressiveness": 1.0,   # how strongly emotion shapes delivery (0-1.5)
    "tone_strength": 1.0,    # strength of pitch/energy colouring (0-1.5)
    "pause": 1.0,            # pause length multiplier (0.3-2.0)
    "formality": 0.5,        # 0 casual … 1 formal (calms variation, lengthens closings)
    "variation": 1.0,        # anti-monotony jitter (0 = identical cadence … 1.5)
    "light_dialect": False,  # optional light MSA->dialect word swaps (see lexicon)
    "stability": 0.6,        # rhythm stability: how strongly sentence speeds are pulled to one steady pace (0..1)
    "energy": 0.0,           # -1 calm … +1 enthusiastic (speed, loudness, shorter pauses; never pitch)
    "clarity": 0.5,          # articulation clarity: presence EQ and a slightly slower pace (0..1)
    "warmth_db": 0.0,        # depth/warmth: low-shelf at 180 Hz, -3..+3 dB (does not change pitch)
    "reverb": 0.0,           # EXPLICIT only. 0 = dry studio sound (default). Never added automatically.
    "denoise": False,        # extra noise reduction on the finished audio (model output is already clean)
}
# Smart defaults per delivery style (used when the user has not set a control explicitly)
STYLE_AUTO = {
    "narration":    {"stability": 0.65, "clarity": 0.55, "energy": 0.0, "warmth_db": 0.5},
    "ad":           {"stability": 0.5, "clarity": 0.6, "energy": 0.4, "warmth_db": 0.0},
    "news":         {"stability": 0.9, "clarity": 0.8, "energy": 0.0, "warmth_db": 0.0},
    "documentary":  {"stability": 0.8, "clarity": 0.6, "energy": -0.1, "warmth_db": 1.0},
    "story":        {"stability": 0.6, "clarity": 0.5, "energy": 0.0, "warmth_db": 1.0},
    "conversation": {"stability": 0.45, "clarity": 0.5, "energy": 0.1, "warmth_db": 0.5},
}
LIMITS = {"speed": (0.7, 1.3), "expressiveness": (0.0, 1.5), "tone_strength": (0.0, 1.5), "pause": (0.3, 2.0),
          "formality": (0.0, 1.0), "variation": (0.0, 1.5), "stability": (0.0, 1.0), "energy": (-1.0, 1.0),
          "clarity": (0.0, 1.0), "warmth_db": (-3.0, 3.0), "reverb": (0.0, 1.0)}
OVERRIDE_LIMITS = {"speed_mul": (0.6, 1.5), "pitch_st": (-2.0, 2.0), "gain_db": (-6.0, 6.0),
                   "pause_before": (0.0, 3.0), "pause_after": (0.0, 4.0)}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def canon_emotion(e: str) -> str:
    e = ALIASES.get(e, e)
    return e if e in EMOTIONS else "neutral"


def auto_controls(style: str | None) -> dict:
    return dict(STYLE_AUTO.get(style or "narration", STYLE_AUTO["narration"]))


def clean_controls(c: dict | None) -> dict:
    out = dict(DEFAULT_CONTROLS)
    for k, v in (c or {}).items():
        if k in LIMITS:
            try:
                out[k] = clamp(float(v), *LIMITS[k])
            except (TypeError, ValueError):
                pass
        elif k in ("light_dialect", "denoise"):
            out[k] = bool(v)
    return out


def _jitter(seed_text: str, amp: float) -> tuple[float, float]:
    """Deterministic pseudo-random (speed, pitch) wobble so repeated sentences are not delivered identically."""
    h = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
    r = random.Random(h)
    return r.uniform(-1, 1) * 0.03 * amp, r.uniform(-1, 1) * 0.25 * amp


def compute(seg: dict, controls: dict, style: str, *, index: int, total: int, voice_style: str | None = None,
            lock_identity: bool = False) -> dict:
    """Concrete parameters for one segment. `seg` needs: emotion, intensity, sentence_type, spoken/text,
    first_in_paragraph/last_in_paragraph, overrides{}. Returns dict with speed_mul, pitch_st, gain_db, pause_*."""
    c = clean_controls(controls)
    st = STYLES.get(style, STYLES["narration"])
    # The persona already carries its own base speed/pause character (voice.speed, fx). Style multipliers are applied
    # RELATIVE to the persona's native style so that e.g. a documentary persona in documentary style is not slowed twice.
    vs = STYLES.get(voice_style or style, st)
    st = (st[0], st[1] / vs[1], st[2] / vs[2], st[3], st[4])
    emo = canon_emotion(seg.get("emotion", "neutral"))
    _n, e_speed, e_pitch, e_gain, e_pb, e_pa, e_nfe, e_var = EMOTIONS[emo]
    inten = clamp(float(seg.get("intensity", 0.5)), 0.0, 1.0)
    k = inten * c["expressiveness"]                     # emotional depth
    speed = st[1] * (1 + (e_speed - 1) * k)
    pitch = e_pitch * k * c["tone_strength"]
    gain = e_gain * k * c["tone_strength"]
    p_before = e_pb * c["pause"] * st[2]
    p_after = e_pa * c["pause"] * st[2]

    text = seg.get("spoken") or seg.get("text", "")
    n_words = max(1, len(text.split()))
    stype = seg.get("sentence_type", "statement")
    # sentence-shape effects (meaning-driven, not emotion-driven)
    if stype == "question":
        pitch += 0.35 * c["tone_strength"]
        p_after *= 1.15
    elif stype == "exclamation":
        gain += 0.8 * c["tone_strength"]
        speed *= 1.02
    elif stype == "trailing":
        speed *= 0.95
        p_after = max(p_after, 0.9 * c["pause"])
    if n_words <= 3:                                    # very short lines land with weight
        speed *= 0.95
        p_after *= 1.2
    elif n_words >= 18:                                 # long lines: keep momentum, breathe after
        speed *= 1.03
        p_after *= 1.15
    # narrative arc: open a little slower, close with a settled tail, breathe between paragraphs
    if index == 0:
        speed *= 0.98
    if index == total - 1:
        speed *= 0.97
        p_after = max(p_after, 0.5 * c["pause"]) + 0.15 * c["formality"]
    if seg.get("last_in_paragraph") and index != total - 1:
        p_after += 0.35 * c["pause"] * st[2]
    # formality: steadier, a touch slower, slightly longer closings
    speed *= 1 - 0.04 * c["formality"]
    p_after *= 1 + 0.25 * c["formality"]
    # energy (calm ↔ enthusiastic) and articulation clarity: tempo/loudness only, never pitch
    speed *= 1 + 0.06 * c["energy"]
    gain += 1.5 * c["energy"]
    p_after *= max(0.6, 1 - 0.2 * c["energy"])
    speed *= 1 - 0.04 * c["clarity"]
    # anti-monotony jitter (a stable rhythm damps it)
    amp = c["variation"] * st[3] * e_var * (1 - 0.5 * c["formality"]) * (1 - 0.6 * c["stability"])
    js, jp = _jitter(f"{index}|{text}", amp)
    speed *= 1 + js
    pitch += jp
    speed *= c["speed"]

    ov = seg.get("overrides") or {}
    if lock_identity:
        # Measured (ECAPA speaker similarity vs the reference): a -0.5 semitone shift alone cost 0.07 of similarity, so
        # voices built from YOUR reference keep their pitch untouched; emotion is carried by speed/pauses/loudness.
        pitch = 0.0
    out = {"speed_mul": speed, "pitch_st": pitch, "gain_db": gain, "pause_before": p_before, "pause_after": p_after}
    for key, (lo, hi) in OVERRIDE_LIMITS.items():
        if ov.get(key) is not None:
            out[key] = ov[key]
        out[key] = clamp(float(out[key]), lo, hi)
    out["nfe_add"] = e_nfe
    out["emotion"] = emo
    return {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in out.items()}
