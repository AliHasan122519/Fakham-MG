"""Renderer: plan -> per-sentence synthesis (cached) -> prosody DSP -> natural assembly -> mastering -> take.

A "take" is one rendered version of a plan (raw baseline + mastered output + per-sentence clips + metrics). Takes are
immutable and kept on disk so versions can be compared, and single sentences can be regenerated cheaply because the
other sentences are served from the segment cache.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from . import audio, config, expressive, history, plan as planmod, voices
from .engine import ENGINE

CACHE = config.DATA / "cache"
TAKES = config.DATA / "takes"
CACHE.mkdir(parents=True, exist_ok=True)
TAKES.mkdir(parents=True, exist_ok=True)
_TAKE_ID = re.compile(r"^[a-f0-9]{12}$")
_SEG_ID = re.compile(r"^[A-Za-z0-9_-]{1,12}$")
CACHE_MAX_BYTES = 1_500_000_000


class RenderError(RuntimeError):
    pass


def _ref_hash(path: Path) -> str:
    st = Path(path).stat()
    return hashlib.sha256(f"{path}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:10]


def _seg_key(engine: str, ref: Path, spoken: str, speed: float, nfe: int, seed: int) -> str:
    return hashlib.sha256(f"{engine}|{_ref_hash(ref)}|{spoken}|{speed:.3f}|{nfe}|{seed}".encode()).hexdigest()[:24]


def _prune_cache() -> None:
    files = sorted(CACHE.glob("*.wav"), key=lambda p: p.stat().st_atime)
    total = sum(p.stat().st_size for p in files)
    while files and total > CACHE_MAX_BYTES:
        p = files.pop(0)
        total -= p.stat().st_size
        p.unlink(missing_ok=True)


def default_synth(spoken, ref_path, ref_text, *, engine, speed, nfe, seed):
    return ENGINE.synth(spoken, str(ref_path), ref_text, engine=engine, speed=speed, nfe_step=nfe, seed=seed)


def _nfe(voice: dict, add: int) -> int:
    if config.DEFAULT_NFE:
        return config.DEFAULT_NFE
    if ENGINE.pick_device() == "cpu":
        return 16        # CPU: ~10x slower than real time already; 32 steps would double that
    return int(voice.get("nfe_step", 32)) + int(add)


def _process_segment(w: np.ndarray, params: dict, fx: dict) -> np.ndarray:
    w = audio.trim_silence(w)
    cur = audio.active_rms_db(w)
    w = w * (10 ** ((-23.0 - cur) / 20))                         # equalise sentence loudness first
    chain = audio.pitch_chain(float(fx.get("pitch_st", 0.0)), float(params.get("pitch_st", 0.0)))
    if chain:
        w = audio.ffmpeg_wave(w, chain)
    w = w * (10 ** (float(params.get("gain_db", 0.0)) / 20))
    return audio.fade(np.clip(w, -1.0, 1.0), 12.0)


def _assemble(clips: list[np.ndarray], pauses: list[tuple[float, float]], sr: int = audio.SR) -> np.ndarray:
    """Join clips with exact pauses. Where the requested gap is tiny, cross-fade instead of butting waveforms."""
    out = np.zeros(0, dtype=np.float32)
    for i, c in enumerate(clips):
        pre = pauses[i][0] + (pauses[i - 1][1] if i > 0 else 0.0)
        if i == 0:
            out = np.concatenate([audio.silence(pauses[0][0], sr), c])
        elif pre < 0.06:                                          # cross-fade 40 ms
            n = min(int(0.04 * sr), len(out), len(c))
            ramp = np.linspace(0, 1, n, dtype=np.float32)
            out[-n:] = out[-n:] * (1 - ramp) + c[:n] * ramp
            out = np.concatenate([out, c[n:]])
        else:
            out = np.concatenate([out, audio.silence(pre, sr), c])
    return np.concatenate([out, audio.silence(pauses[-1][1], sr)])


def render(plan_in: dict, *, fmt: str = "both", progress: Callable[[float, str], None] | None = None,
           synth_fn=default_synth, parent: str | None = None, clean: bool = True) -> dict:
    progress = progress or (lambda p, m: None)
    t0 = time.time()
    plan = planmod.validate(plan_in)
    v = voices.get_voice(plan["voice"])
    dcfg = voices.dialects()[plan["dialect"]]
    engine = dcfg["engine"]
    if not dcfg.get("available", True):
        raise RenderError("هذه اللهجة/المحرك غير مفعّل في هذه البيئة (المحرك التعبيري اختياري).")
    segs = plan["segments"]
    raw_clips, proc_clips, pauses, seg_meta = [], [], [], []
    warnings: list[str] = []
    cache_hits = 0
    base_speed = float(v.get("speed", 1.0))
    for i, s in enumerate(segs):
        progress(0.04 + 0.78 * i / len(segs), f"توليد الجملة {i + 1} من {len(segs)}")
        P = s["params"]
        emo_for_ref = s.get("emotion_ref") or s["emotion"]
        try:
            ref, ref_text, ref_src = voices.reference_for(v, plan["dialect"], emo_for_ref)
        except KeyError:
            raise RenderError("لا يوجد مقطع مرجعي لهذا الصوت. ارفع مقطعًا تملك حق استخدامه.")
        if ref_src == "demo" and "demo" not in warnings:
            warnings.append("demo")
        speed = float(np.clip(base_speed * P["speed_mul"], 0.75, 1.5))   # below ~0.75 the model starts to smear words
        nfe = _nfe(v, P.get("nfe_add", 0))
        exag = cfg_w = None
        if engine == "expressive":
            exag, cfg_w = expressive.exaggeration_for(s["emotion"], s["intensity"], plan["controls"]["expressiveness"])
            nfe = int(exag * 100) * 10 + int(cfg_w * 10)        # folded into the cache key
        key = _seg_key(engine, ref, s["spoken"], speed, nfe, s["seed"])
        cp = CACHE / f"{key}.wav"
        if cp.is_file():
            wave = sf.read(str(cp), dtype="float32")[0]
            cp.touch()
            cache_hits += 1
        else:
            if engine == "expressive":
                wave, sr = expressive.WORKER_POOL.synth(s["spoken"], ref, exaggeration=exag, cfg=cfg_w, seed=s["seed"])
                if abs(speed - 1.0) > 0.03 and audio.have_filter("rubberband"):      # this model has no speed input
                    wave = audio.ffmpeg_wave(wave, f"rubberband=tempo={float(np.clip(speed, 0.8, 1.25)):.4f}:pitchq=quality", sr)
            else:
                wave, sr = synth_fn(s["spoken"], ref, ref_text, engine=engine, speed=speed, nfe=nfe, seed=s["seed"])
            sf.write(str(cp), np.asarray(wave, dtype=np.float32), sr)
            wave = np.asarray(wave, dtype=np.float32)
        raw_clips.append(wave)
        proc_clips.append(_process_segment(wave, P, v.get("fx") or {}))
        pauses.append((P["pause_before"], P["pause_after"]))
        seg_meta.append({"id": s["id"], "ref_source": ref_src, "speed": round(speed, 3), "nfe": nfe,
                         "seconds": round(len(proc_clips[-1]) / audio.SR, 2), "cached": cp.exists() and cache_hits > 0})

    progress(0.84, "التجميع والانتقالات")
    take_id = uuid.uuid4().hex[:12]
    d = TAKES / take_id
    d.mkdir(parents=True)
    # --- BEFORE baseline: model output exactly as produced, fixed 0.35 s gaps, only loudness-matched (fair A/B)
    base_raw = _assemble(raw_clips, [(0.0, 0.35)] * len(raw_clips))
    audio.write_wav(d / "raw_in.wav", base_raw)
    # --- AFTER: processed sentences, planned pauses, adaptive master
    mix = _assemble(proc_clips, pauses)
    audio.write_wav(d / "mix_in.wav", mix)
    progress(0.90, "المعالجة الصوتية النهائية")
    profile = audio.spectral_profile(mix)
    af = audio.master_chain(v.get("fx"), profile, clean=clean)
    audio.process(d / "raw_in.wav", d / "raw.wav", None, audio.leveled_only())
    audio.process(d / "mix_in.wav", d / "master.wav", None, af)
    audio.encode(d / "raw.wav", d / "raw.mp3")
    audio.encode(d / "master.wav", d / "master.mp3")
    (d / "raw_in.wav").unlink(missing_ok=True)
    (d / "mix_in.wav").unlink(missing_ok=True)
    # per-sentence clips for audition (same chain, individually leveled)
    for s, c in zip(segs, proc_clips):
        tmp = d / f"_{s['id']}.wav"
        audio.write_wav(tmp, c)
        audio.encode(tmp, d / f"seg_{s['id']}.mp3", bitrate="128k")
        tmp.unlink(missing_ok=True)
    progress(0.96, "قياس الجودة")
    m_raw, m_master = audio.metrics(d / "raw.wav"), audio.metrics(d / "master.wav")
    took = time.time() - t0
    secs = m_master["duration_s"]
    take = {"id": take_id, "parent": parent, "created": time.time(), "plan": plan, "segments": seg_meta,
            "files": {"master_mp3": "master.mp3", "master_wav": "master.wav", "raw_mp3": "raw.mp3",
                      "segments": {s["id"]: f"seg_{s['id']}.mp3" for s in segs}},
            "metrics": {"before": m_raw, "after": m_master}, "profile": profile, "mastering": af,
            "device": ENGINE.device if ENGINE.device != "uninitialised" else ENGINE.pick_device(),
            "engine": engine, "dialect_native": dcfg["native"], "dialect_note": dcfg["note"],
            "warnings": warnings, "took_s": round(took, 1), "realtime_factor": round(took / max(secs, 0.1), 1),
            "cache_hits": cache_hits}
    (d / "take.json").write_text(json.dumps(take, ensure_ascii=False), encoding="utf-8")
    history.add({"id": take_id, "voice": plan["voice"], "dialect": plan["dialect"],
                 "emotion": plan["mode"], "speed": plan["controls"]["speed"],
                 "text": " ".join(s["text"] for s in segs)[:500], "seconds": secs, "took": round(took, 1),
                 "device": take["device"], "engine": engine})
    _prune_cache()
    progress(1.0, "تم")
    return take


def load_take(take_id: str) -> dict:
    if not _TAKE_ID.match(take_id):
        raise RenderError("معرّف النسخة غير صالح.")
    p = TAKES / take_id / "take.json"
    if not p.is_file():
        raise RenderError("النسخة غير موجودة (قد تكون الجلسة أُعيد تشغيلها).")
    return json.loads(p.read_text(encoding="utf-8"))


def list_takes(limit: int = 30) -> list[dict]:
    out = []
    for p in sorted(TAKES.glob("*/take.json"), key=lambda q: q.stat().st_mtime, reverse=True)[:limit]:
        try:
            t = json.loads(p.read_text(encoding="utf-8"))
            out.append({"id": t["id"], "parent": t.get("parent"), "created": t["created"], "voice": t["plan"]["voice"],
                        "seconds": t["metrics"]["after"]["duration_s"], "sentences": len(t["plan"]["segments"]),
                        "preview": t["plan"]["segments"][0]["text"][:60]})
        except Exception:
            continue
    return out


def take_file(take_id: str, name: str) -> Path:
    if not _TAKE_ID.match(take_id):
        raise RenderError("معرّف غير صالح.")
    allowed = {"master.mp3", "master.wav", "raw.mp3"}
    if name not in allowed and not re.fullmatch(r"seg_[A-Za-z0-9_-]{1,12}\.mp3", name):
        raise RenderError("اسم ملف غير مسموح.")
    p = TAKES / take_id / name
    if not p.is_file():
        raise RenderError("الملف غير موجود.")
    return p


def regenerate(take_id: str, segment_id: str, *, edits: dict | None = None, new_seed: bool = True,
               progress=None, synth_fn=default_synth) -> dict:
    """New take = same plan, with one sentence re-rolled (new seed) and/or edited. All other sentences come from the
    segment cache, so this costs one sentence of synthesis instead of the whole text."""
    if not _SEG_ID.match(segment_id):
        raise RenderError("معرّف الجملة غير صالح.")
    old = load_take(take_id)
    plan = json.loads(json.dumps(old["plan"]))
    seg = next((s for s in plan["segments"] if s["id"] == segment_id), None)
    if seg is None:
        raise RenderError("الجملة غير موجودة في هذه النسخة.")
    for k, val in (edits or {}).items():
        if k in ("spoken", "emotion", "intensity", "emotion_ref"):
            seg[k] = val
        elif k == "overrides" and isinstance(val, dict):
            seg["overrides"] = val
    if new_seed:
        seg["seed"] = (int(seg["seed"]) * 1103515245 + 12345 + int(time.time())) & 0x7FFFFFFF
    plan = planmod.recompute(plan)
    return render(plan, progress=progress, synth_fn=synth_fn, parent=take_id)
