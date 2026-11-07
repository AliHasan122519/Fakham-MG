"""High-level generation pipeline used by both the web UI and the MCP server."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio, config, history, text_ar, voices
from .engine import ENGINE


class StudioError(RuntimeError):
    pass


def generate(text: str, voice_id: str = "narrator_deep", dialect: str | None = None, emotion: str = "neutral",
             speed: float | None = None, fmt: str = "mp3", progress: Callable[[float, str], None] | None = None,
             seed: int | None = None) -> dict:
    """Generate a voice-over. Returns metadata incl. file names under config.OUTPUTS."""
    progress = progress or (lambda p, m: None)
    t0 = time.time()
    text = text_ar.normalize(text)
    if not text:
        raise StudioError("النص فارغ.")
    if len(text) > config.MAX_CHARS:
        raise StudioError(f"النص طويل ({len(text)} حرفًا). الحد الحالي {config.MAX_CHARS}. قسّمه إلى أجزاء.")
    if not text_ar.has_arabic(text):
        raise StudioError("هذا المحرك مخصص للنص العربي — لم أجد حروفًا عربية.")

    v = voices.get_voice(voice_id)
    d_key = dialect or v["default_dialect"]
    dialect_cfg = voices.dialects().get(d_key)
    if not dialect_cfg:
        raise StudioError(f"لهجة غير معروفة: {d_key}")
    try:
        ref_path, ref_text, ref_src = voices.reference_for(v, d_key)
    except KeyError:
        raise StudioError("لا يوجد مقطع مرجعي ولا مقطع تجريبي. ارفع مقطعًا تملك حق استخدامه (set_voice_reference).")
    emo = voices.emotions().get(emotion)
    if not emo:
        raise StudioError(f"عاطفة غير معروفة: {emotion}")

    base_speed = float(speed if speed is not None else v["speed"])
    eff_speed = max(0.5, min(1.6, base_speed * emo["speed_mul"]))
    nfe = int(v["nfe_step"]) + int(emo["nfe_add"])
    if config.DEFAULT_NFE:
        nfe = config.DEFAULT_NFE
    elif ENGINE.pick_device() == "cpu":
        nfe = 16  # CPU is ~10x slower than real time at 16 steps; raise with SVS_NFE=32 if you can wait
    engine_key = dialect_cfg["engine"]

    chunks = text_ar.split_chunks(text)
    parts: list[np.ndarray] = []
    sr = audio.SR
    for i, ch in enumerate(chunks):
        progress(i / len(chunks), f"توليد الجزء {i + 1} من {len(chunks)}")
        wave, sr = ENGINE.synth(ch, str(ref_path), ref_text, engine=engine_key,
                                speed=eff_speed, nfe_step=nfe, seed=seed)
        parts.append(wave)
        if i < len(chunks) - 1:
            parts.append(audio.silence(emo["pause_s"], sr))
    wave = np.concatenate(parts)

    gid = uuid.uuid4().hex[:12]
    raw = config.OUTPUTS / f"{gid}.raw.wav"
    wav = config.OUTPUTS / f"{gid}.wav"
    mp3 = config.OUTPUTS / f"{gid}.mp3" if fmt in ("mp3", "both") else None
    progress(0.95, "المعالجة الصوتية")
    audio.write_wav(raw, wave, sr)
    audio.process(raw, wav, mp3 if mp3 else config.OUTPUTS / f"{gid}.mp3", audio.build_filter(v.get("fx"), emo["pitch_st"], emo["gain_db"]))
    raw.unlink(missing_ok=True)

    secs = audio.duration(wav)
    took = time.time() - t0
    rec = dict(id=gid, voice=voice_id, dialect=d_key, emotion=emotion, speed=eff_speed, text=text[:500],
               seconds=round(secs, 2), took=round(took, 1), device=ENGINE.device, engine=engine_key)
    history.add(rec)
    progress(1.0, "تم")
    return {**rec, "files": {"mp3": f"{gid}.mp3", "wav": f"{gid}.wav"},
            "reference_source": ref_src,
            "warnings": ([] if ref_src == "own" else ["يستخدم مقطعًا تجريبيًا مرفقًا بحزمة habibi-tts؛ ارفع مقطعك المصرّح به لتصبح الشخصية صوتك الخاص (set_voice_reference)."]),
            "dialect_native": dialect_cfg["native"], "dialect_note": dialect_cfg["note"],
            "realtime_factor": round(took / max(secs, 0.1), 1)}


def output_path(gid: str, ext: str) -> Path:
    if not gid.isalnum() or ext not in ("mp3", "wav"):
        raise StudioError("معرّف أو صيغة غير صالحة.")
    p = config.OUTPUTS / f"{gid}.{ext}"
    if not p.is_file():
        raise StudioError("الملف غير موجود (قد تكون الجلسة أُعيد تشغيلها).")
    return p
