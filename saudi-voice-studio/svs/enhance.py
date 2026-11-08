"""Smart Voice Enhancement ("التحسين الذكي"): one call that prepares everything BEFORE synthesis.

Pipeline (each step is real and measurable — nothing here is decorative):
  1. analysis      — sentences/paragraphs, genre, per-sentence emotion from MEANING (semantic embeddings) + lexical cues
  2. pronunciation — numbers/units/abbreviations/brands → speakable text, your dictionary, optional light dialect
  3. diacritics    — context-resolved readings of ambiguous words (علم/ذهب/كتب/شعر…), waqf forms before pauses
  4. delivery plan — per-sentence speed, pitch, loudness, pauses, anti-monotony, narrative arc
  5. review        — the plan is returned for you to edit/approve (original, corrected and vocalised text side by side)
  6. synthesis + adaptive mastering + (optionally) a TRUE raw baseline for an honest A/B
"""
from __future__ import annotations

import time

from . import analysis, asr, delivery, plan as planmod, semantic


def smart_enhance(text: str, *, voice: str = "narrator_deep", dialect: str | None = None, style: str | None = None,
                  mode: str = "auto", controls: dict | None = None, emotion: str | None = None,
                  diacritics: str = "selective", compare_raw: bool = True, use_semantic: bool = True) -> dict:
    t0 = time.time()
    steps = []
    t = time.time()
    plan = planmod.build(text, voice=voice, dialect=dialect, style=style, mode=mode, controls=controls, emotion=emotion,
                         diacritics=diacritics, use_semantic=use_semantic)
    segs = plan["segments"]
    emo_counts: dict[str, int] = {}
    for s in segs:
        emo_counts[s["emotion"]] = emo_counts.get(s["emotion"], 0) + 1
    changed = sum(1 for s in segs if (s.get("corrected") or s["text"]).strip() != s["text"].strip())
    voc_changed = sum(1 for s in segs if s.get("vocalized") and s["vocalized"] != s.get("corrected"))
    resolved = sum(len((s.get("tashkeel") or {}).get("resolved", [])) for s in segs)
    unresolved = sum(len((s.get("tashkeel") or {}).get("unresolved", [])) for s in segs)
    method = plan.get("analysis_method", "lexical")
    steps = [
        {"id": "analysis", "name": "تحليل النص", "ok": True, "detail": f"{len(segs)} جملة · النوع: {plan['genre_name']} · الطريقة: "
         + ("دلالية (معنى الجملة) + معجمية" if method == "semantic+lexical" else "معجمية فقط (النموذج الدلالي غير متاح)")},
        {"id": "pronunciation", "name": "تصحيح النطق", "ok": True, "detail": f"{changed} جملة عُدّل نطقها (أرقام/اختصارات/قاموس)"},
        {"id": "diacritics", "name": "التشكيل السياقي", "ok": diacritics == "selective",
         "detail": (f"حُسمت {resolved} كلمة ملتبسة من السياق، و{unresolved} تحتاج اختيارك" if diacritics == "selective" else "معطّل")},
        {"id": "emotions", "name": "تحديد المشاعر", "ok": True, "detail": "، ".join(f"{delivery.EMOTIONS[k][0]}: {v}" for k, v in emo_counts.items())},
        {"id": "plan", "name": "خطة الإلقاء", "ok": True, "detail": "سرعة/نبرة/جهارة/وقفات لكل جملة — قابلة للتعديل"},
    ]
    plan["enhancement"] = {"steps": steps, "compare_raw": bool(compare_raw), "method": method,
                           "semantic_model": semantic.MODEL_ID if method == "semantic+lexical" else None,
                           "asr_available": asr.available(), "seconds": round(time.time() - t0, 1)}
    return plan
