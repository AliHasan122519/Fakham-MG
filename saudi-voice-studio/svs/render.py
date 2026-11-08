"""Renderer: plan -> per-sentence synthesis (cached) -> prosody DSP -> natural assembly -> mastering -> take.

A "take" is one rendered version of a plan (raw baseline + mastered output + per-sentence clips + metrics). Takes are
immutable and kept on disk so versions can be compared, and single sentences can be regenerated cheaply because the
other sentences are served from the segment cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from . import asr, audio, config, expressive, history, plan as planmod, refaudio, speaker, voices
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


VERIFY = os.environ.get("SVS_VERIFY", "auto")        # auto: on when faster-whisper is installed
MIN_INTELLIGIBILITY = 0.5
MAX_RETRIES = 2


def audio_resample(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, timeout=120).stdout
    return np.frombuffer(raw, dtype=np.float32)


_REF_EMB: dict = {}


def _ref_embedding(ref: Path):
    key = (str(ref), int(Path(ref).stat().st_mtime))
    if key not in _REF_EMB:
        x = sf.read(str(ref), dtype="float32")[0]
        x = x if x.ndim == 1 else x[:, 0]
        _REF_EMB[key] = speaker.embed(refaudio.audio_resample16(x))
    return _REF_EMB[key]


def _spk_sim(wave: np.ndarray, ref: Path) -> float | None:
    if not speaker.available() or len(wave) < int(1.5 * audio.SR):
        return None
    try:
        return float(np.dot(_ref_embedding(ref), speaker.embed(refaudio.audio_resample16(wave))))
    except Exception:
        return None


def default_candidates() -> int:
    """Candidates per sentence. Default 1; the "max quality" option or SVS_CANDIDATES raises it (best-of-N by speaker similarity)."""
    env = os.environ.get("SVS_CANDIDATES")
    if env:
        return max(1, min(5, int(env)))
    return 1      # measured: best-of-3 raised voice similarity by only ~0.01 at 3.4x the time, so it is opt-in ("جودة قصوى")


def _verify_on() -> bool:
    return VERIFY != "off" and asr.available()


def _intelligibility(wave: np.ndarray, spoken: str) -> float:
    heard = asr.transcribe(refaudio.audio_resample16(np.asarray(wave, dtype=np.float32)))["text"]
    return asr.similarity(spoken, heard)


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
           synth_fn=default_synth, parent: str | None = None, clean: bool = False, compare_raw: bool | None = None,
           candidates: int | None = None) -> dict:
    progress = progress or (lambda p, m: None)
    t0 = time.time()
    plan = planmod.validate(plan_in)
    v = voices.get_voice(plan["voice"])
    dcfg = voices.dialects()[plan["dialect"]]
    engine = dcfg["engine"]
    if not dcfg.get("available", True):
        raise RenderError("هذه اللهجة/المحرك غير مفعّل في هذه البيئة (المحرك التعبيري اختياري).")
    segs = plan["segments"]
    if compare_raw is None:
        compare_raw = bool((plan_in.get("enhancement") or {}).get("compare_raw"))
    raw_clips, proc_clips, pauses, seg_meta = [], [], [], []
    warnings: list[str] = []
    intel: dict[str, float] = {}
    stats = {"cache_hits": 0}
    base_speed = float(v.get("speed", 1.0))
    n_pass = 2 if compare_raw else 1
    if candidates is None:
        candidates = int((plan_in.get("enhancement") or {}).get("candidates") or 1)
    candidates = max(1, min(5, candidates))
    match: dict[str, float] = {}

    def wave_for(spoken, ref, ref_text, speed, nfe, seed, *, verify, label, frac, exag=None, cfg_w=None):
        """Synthesise one sentence with cache, optional intelligibility guard and seed re-rolls."""
        key = _seg_key(engine, ref, spoken, speed, nfe, seed)
        cp = CACHE / f"{key}.wav"
        if cp.is_file():
            cp.touch()
            stats["cache_hits"] += 1
            return sf.read(str(cp), dtype="float32")[0], None
        def once(sd):
            if engine == "expressive":
                w, sr_ = expressive.WORKER_POOL.synth(spoken, ref, exaggeration=exag, cfg=cfg_w, seed=sd)
                if abs(speed - 1.0) > 0.03 and audio.have_filter("rubberband"):      # this model has no speed input
                    w = audio.ffmpeg_wave(w, f"rubberband=tempo={float(np.clip(speed, 0.8, 1.25)):.4f}:pitchq=quality", sr_)
                return np.asarray(w, dtype=np.float32), sr_
            w, sr_ = synth_fn(spoken, ref, ref_text, engine=engine, speed=speed, nfe=nfe, seed=sd)
            return np.asarray(w, dtype=np.float32), sr_
        wave, sr = once(seed)
        sim = None
        if candidates > 1 and verify and engine != "expressive" and speaker.available():
            # Best-of-N: keep the candidate closest to YOUR reference voice (speaker embedding) that is also intelligible.
            def score(w):
                sp_ = _spk_sim(w, ref)
                it_ = _intelligibility(w, spoken) if (_verify_on() and len(spoken) >= 12) else 1.0
                return (0.65 * (sp_ if sp_ is not None else 0.5) + 0.35 * it_), sp_, it_
            best, (bs, _, _) = (wave, sr), score(wave)
            for k in range(1, candidates):
                progress(frac, f"تجربة نسخة {k + 1} من {candidates} للجملة (اختيار الأقرب لصوتك)")
                w2, sr2 = once((int(seed) + 104729 * k) & 0x7FFFFFFF)
                sc2 = score(w2)[0]
                if sc2 > bs:
                    best, bs = (w2, sr2), sc2
            wave, sr = best
        if verify and _verify_on() and len(spoken) >= 12 and engine != "expressive":
            # Output guard: a bad/noisy reference yields babble. Check intelligibility; re-roll the seed up to twice.
            best, best_sim, tries = wave, _intelligibility(wave, spoken), 0
            while best_sim < MIN_INTELLIGIBILITY and tries < MAX_RETRIES:
                tries += 1
                progress(frac, f"إعادة محاولة {label} (وضوح {int(best_sim * 100)}%)")
                w2, sr = once((int(seed) + 7919 * tries) & 0x7FFFFFFF)
                sim2 = _intelligibility(w2, spoken)
                if sim2 > best_sim:
                    best, best_sim = w2, sim2
            wave, sim = best, best_sim
        sf.write(str(cp), wave, sr)
        return wave, sim

    for i, s in enumerate(segs):
        progress(0.04 + 0.78 / n_pass * i / len(segs), f"توليد الجملة {i + 1} من {len(segs)}")
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
        wave, sim = wave_for(s["spoken"], ref, ref_text, speed, nfe, s["seed"], verify=True, label=f"الجملة {i + 1}",
                             frac=0.04 + 0.78 / n_pass * i / len(segs), exag=exag, cfg_w=cfg_w)
        diac_note = None
        if (sim is not None and engine != "expressive" and s.get("vocalized") and s.get("corrected")
                and s["spoken"] == s["vocalized"] and s["vocalized"] != s["corrected"]):
            # Verification-gated diacritics: measured on real generations, a diacritised word can make the model slower or
            # less clear (e.g. «ذَهَبَ» → ASR 0.81-0.93 vs plain 0.97-1.0). Synthesize the plain text too and keep the
            # diacritised reading only if it is not clearly less intelligible.
            w_alt, _ = wave_for(s["corrected"], ref, ref_text, speed, nfe, s["seed"], verify=False, label="", frac=0.5)
            sim_alt = _intelligibility(w_alt, s["corrected"])
            if sim_alt > sim + 0.05:
                wave, sim, diac_note = w_alt, sim_alt, "dropped"
            else:
                diac_note = "kept"
        if sim is not None:
            intel[s["id"]] = round(sim, 3)
            if sim < MIN_INTELLIGIBILITY:
                warnings.append(f"low_intelligibility:{s['id']}:{int(sim * 100)}")
        vm = _spk_sim(wave, ref)
        if vm is not None:
            match[s["id"]] = round(vm, 3)
        raw_clips.append(wave)
        proc_clips.append(_process_segment(wave, P, v.get("fx") or {}))
        pauses.append((P["pause_before"], P["pause_after"]))
        seg_meta.append({"id": s["id"], "ref_source": ref_src, "speed": round(speed, 3), "nfe": nfe, "diacritics": diac_note,
                         "seconds": round(len(proc_clips[-1]) / audio.SR, 2)})

    baseline_kind = "model_output_same_text"
    if compare_raw:
        # TRUE baseline: the original text (numbers/units only normalised), neutral delivery, no diacritics, no processing.
        baseline_kind = "true_raw"
        raw_clips = []
        for i, s in enumerate(segs):
            progress(0.45 + 0.37 * i / len(segs), f"توليد النسخة الخام للمقارنة {i + 1} من {len(segs)}")
            ref, ref_text, _ = voices.reference_for(v, plan["dialect"], None)
            speed_b = float(np.clip(base_speed, 0.75, 1.5))
            nfe_b = _nfe(v, 0)
            text_b = s.get("corrected") or s["text"]
            w, _sim = wave_for(text_b, ref, ref_text, speed_b, nfe_b, s["seed"], verify=False, label="", frac=0.5)
            raw_clips.append(w)
    cache_hits = stats["cache_hits"]

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
    final_match = None
    if speaker.available() and segs:
        try:   # the HEADLINE number: the finished, mastered file against your reference (per-sentence numbers are noisy when short)
            ref0, _, _ = voices.reference_for(v, plan["dialect"], None)
            mw, _msr = sf.read(str(d / "master.wav"), dtype="float32")
            mw = mw if mw.ndim == 1 else mw[:, 0]
            if len(mw) >= int(2.5 * audio.SR):
                final_match = round(float(np.dot(_ref_embedding(ref0), speaker.embed(refaudio.audio_resample16(mw)))), 3)
        except Exception:
            final_match = None
    cmp_intel = None
    if compare_raw and _verify_on():
        progress(0.97, "قياس وضوح النطق قبل/بعد")
        target = " ".join((s.get("corrected") or s["text"]) for s in segs)
        def _sim(wav):
            x16 = audio_resample(wav)
            return round(asr.similarity(target, asr.transcribe(x16)["text"]), 3)
        cmp_intel = {"before": _sim(d / "raw.wav"), "after": _sim(d / "master.wav"),
                     "note": "تطابق تفريغ الصوت مع النص المصحّح (1.0 = مطابق تمامًا) — مقياس آلي لا يغني عن الاستماع."}
    took = time.time() - t0
    secs = m_master["duration_s"]
    if intel and all(v < 0.3 for v in intel.values()) and len(intel) == len(segs):
        raise RenderError("الصوت الناتج غير مفهوم في كل الجمل — غالبًا المقطع المرجعي ضعيف أو مشوّش. "
                          "أعد رفع مقطع أنظف (٥–١٠ ثوانٍ، بلا ضجيج) من تبويب الشخصيات.")
    take = {"id": take_id, "parent": parent, "intelligibility": intel, "baseline_kind": baseline_kind,
            "voice_match": ({"final": None, "mean": round(float(np.mean(list(match.values()))), 3), "min": round(float(min(match.values())), 3),
                             "per_sentence": match, "candidates": candidates,
                             "scale": "غير المتشابهين ≈ 0.0–0.3 · نفس المتحدث ≈ 0.7+"} if match else None),
            "intelligibility_compare": cmp_intel, "created": time.time(), "plan": plan, "segments": seg_meta,
            "files": {"master_mp3": "master.mp3", "master_wav": "master.wav", "raw_mp3": "raw.mp3",
                      "segments": {s["id"]: f"seg_{s['id']}.mp3" for s in segs}},
            "metrics": {"before": m_raw, "after": m_master}, "profile": profile, "mastering": af,
            "device": ENGINE.device if ENGINE.device != "uninitialised" else ENGINE.pick_device(),
            "engine": engine, "dialect_native": dcfg["native"], "dialect_note": dcfg["note"],
            "warnings": warnings, "took_s": round(took, 1), "realtime_factor": round(took / max(secs, 0.1), 1),
            "cache_hits": cache_hits}
    if take.get("voice_match") is not None:
        take["voice_match"]["final"] = final_match
    elif final_match is not None:
        take["voice_match"] = {"final": final_match, "mean": final_match, "min": final_match, "per_sentence": {}, "candidates": candidates,
                               "scale": "غير المتشابهين ≈ 0.0–0.3 · نفس المتحدث ≈ 0.7+"}
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
