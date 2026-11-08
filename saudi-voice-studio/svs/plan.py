"""Performance plan: build from text (auto), recompute after edits, validate. The plan is the single source of
truth that the renderer follows, and is fully editable (spoken text, emotion, intensity, per-segment overrides)."""
from __future__ import annotations

import hashlib
import uuid

from . import analysis, delivery, lexicon_ar, pronounce, voices

MAX_SEGMENTS = 60
MAX_CHARS = 6000
MODES = ("auto", "manual")


def _seg_seed(plan_id: str, idx: int) -> int:
    return int(hashlib.sha256(f"{plan_id}:{idx}".encode()).hexdigest()[:7], 16)


def build(text: str, *, voice: str = "narrator_deep", dialect: str | None = None, style: str | None = None,
          mode: str = "auto", controls: dict | None = None, emotion: str | None = None) -> dict:
    """Analyse text and produce an editable plan. mode='manual' => every segment starts at `emotion` (or neutral)
    instead of the detected emotion (detections are still shown as suggestions)."""
    if mode not in MODES:
        raise ValueError("mode must be auto|manual")
    text = (text or "").strip()
    if not text:
        raise ValueError("النص فارغ.")
    if len(text) > MAX_CHARS:
        raise ValueError(f"النص طويل ({len(text)}). الحد {MAX_CHARS} حرف للخطة الواحدة.")
    v = voices.get_voice(voice)
    dialect = dialect or v["default_dialect"]
    if dialect not in voices.dialects():
        raise ValueError(f"لهجة غير معروفة: {dialect}")
    an = analysis.analyze(text)
    if len(an["sentences"]) > MAX_SEGMENTS:
        raise ValueError(f"عدد الجمل كبير ({len(an['sentences'])}). الحد {MAX_SEGMENTS}.")
    ctrl = delivery.clean_controls(controls)
    style = style if style in delivery.STYLES else an["style"]
    plan_id = uuid.uuid4().hex[:10]
    segs = []
    for i, s in enumerate(an["sentences"]):
        sp = pronounce.to_spoken(s["text"], dialect, light_dialect=ctrl["light_dialect"])
        emo = delivery.canon_emotion(emotion or "neutral") if mode == "manual" else s["emotion"]
        segs.append({"id": f"s{i + 1}", "text": s["text"], "spoken": sp["spoken"], "pronunciation": sp,
                     "paragraph": s["paragraph"], "first_in_paragraph": s["first_in_paragraph"],
                     "last_in_paragraph": s["last_in_paragraph"], "sentence_type": s["sentence_type"],
                     "emotion": emo, "intensity": s["intensity"] if mode == "auto" else 0.6,
                     "detected": {"emotion": s["emotion"], "intensity": s["intensity"], "cues": s["cues"],
                                  "scores": s["scores"]},
                     "overrides": {}, "seed": _seg_seed(plan_id, i), "spoken_edited": False})
    plan = {"version": 2, "id": plan_id, "voice": voice, "dialect": dialect, "style": style, "mode": mode,
            "genre": an["genre"], "genre_name": an["genre_name"], "controls": ctrl, "segments": segs,
            "voice_style": v.get("style", "narration")}
    return recompute(plan)


def recompute(plan: dict) -> dict:
    """Recompute computed prosody ('params') for every segment from emotion/intensity/controls/overrides."""
    plan["controls"] = delivery.clean_controls(plan.get("controls"))
    n = len(plan["segments"])
    for i, s in enumerate(plan["segments"]):
        s["emotion"] = delivery.canon_emotion(s.get("emotion", "neutral"))
        s["intensity"] = delivery.clamp(float(s.get("intensity", 0.5)), 0.0, 1.0)
        s["params"] = delivery.compute(s, plan["controls"], plan.get("style", "narration"), index=i, total=n,
                                       voice_style=plan.get("voice_style"))
    return plan


def validate(plan: dict) -> dict:
    """Validate an externally supplied/edited plan (from the UI or from Claude) and return a sanitised copy."""
    if not isinstance(plan, dict) or not isinstance(plan.get("segments"), list) or not plan["segments"]:
        raise ValueError("خطة غير صالحة: segments مطلوبة.")
    if len(plan["segments"]) > MAX_SEGMENTS:
        raise ValueError(f"عدد الجمل كبير (الحد {MAX_SEGMENTS}).")
    voices.get_voice(str(plan.get("voice", "")))
    if plan.get("dialect") not in voices.dialects():
        raise ValueError("لهجة غير معروفة.")
    out = {k: plan.get(k) for k in ("version", "id", "voice", "dialect", "style", "mode", "genre", "genre_name")}
    out["voice_style"] = voices.get_voice(str(plan["voice"])).get("style", "narration")
    out["id"] = str(out.get("id") or uuid.uuid4().hex[:10])[:20]
    out["style"] = out["style"] if out["style"] in delivery.STYLES else "narration"
    out["mode"] = out["mode"] if out["mode"] in MODES else "auto"
    out["controls"] = delivery.clean_controls(plan.get("controls"))
    segs, total_chars = [], 0
    for i, s in enumerate(plan["segments"]):
        spoken = str(s.get("spoken") or s.get("text") or "").strip()
        if not spoken:
            raise ValueError(f"الجملة {i + 1} فارغة.")
        total_chars += len(spoken)
        ov = {}
        for k, (lo, hi) in delivery.OVERRIDE_LIMITS.items():
            if (s.get("overrides") or {}).get(k) is not None:
                ov[k] = delivery.clamp(float(s["overrides"][k]), lo, hi)
        segs.append({"id": str(s.get("id") or f"s{i + 1}")[:12], "text": str(s.get("text", spoken))[:1000],
                     "spoken": spoken[:1000], "paragraph": int(s.get("paragraph", 0)),
                     "first_in_paragraph": bool(s.get("first_in_paragraph")),
                     "last_in_paragraph": bool(s.get("last_in_paragraph")),
                     "sentence_type": s.get("sentence_type") if s.get("sentence_type") in
                     ("statement", "question", "exclamation", "trailing") else "statement",
                     "emotion": delivery.canon_emotion(s.get("emotion", "neutral")),
                     "intensity": delivery.clamp(float(s.get("intensity", 0.5)), 0.0, 1.0),
                     "overrides": ov, "seed": int(s.get("seed") or _seg_seed(out["id"], i)) & 0x7FFFFFFF,
                     "emotion_ref": s.get("emotion_ref") if s.get("emotion_ref") in delivery.EMOTIONS else None})
    if total_chars > MAX_CHARS:
        raise ValueError(f"النص المنطوق طويل (الحد {MAX_CHARS}).")
    out["segments"] = segs
    return recompute(out)


def respell(plan: dict, idx: int, spoken: str) -> dict:
    """User edited the spoken text of one segment."""
    plan["segments"][idx]["spoken"] = spoken.strip()
    plan["segments"][idx]["spoken_edited"] = True
    return recompute(plan)
