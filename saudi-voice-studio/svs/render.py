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

from . import asr, audio, config, expressive, history, plan as planmod, qc, refaudio, speaker, voices
from . import clean as _cleanup
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


def _f0_median_st(w: np.ndarray) -> float | None:
    """Median pitch of a sentence in semitones re (100 Hz); None when too few voiced frames."""
    try:
        f0 = audio._f0_track(refaudio.audio_resample16(np.asarray(w, dtype=np.float32)))
    except Exception:
        return None
    return float(np.median(12 * np.log2(f0 / 100.0))) if len(f0) >= 25 else None


_F0_BASE_FILE = config.DATA / "f0_baseline.json"


def _f0_baseline(ref_path) -> float | None:
    """Typical narration pitch (semitones re 100 Hz) of THIS voice/reference, learned from previous takes; None until >= 10 sentences seen."""
    try:
        d = json.loads(_F0_BASE_FILE.read_text(encoding="utf-8"))
        e = d.get(_ref_hash(Path(ref_path)))
        return float(e["median"]) if e and int(e.get("n", 0)) >= 10 else None
    except Exception:
        return None


def _f0_baseline_update(ref_path, f0s: list) -> None:
    """Running estimate (bounded memory of the last 60 sentence pitches) — used only to judge future takes with too few sentences."""
    vals = [float(x) for x in f0s if x is not None]
    if len(vals) < 3:
        return
    try:
        try:
            d = json.loads(_F0_BASE_FILE.read_text(encoding="utf-8"))
        except Exception:
            d = {}
        k = _ref_hash(Path(ref_path))
        hist = (d.get(k) or {}).get("hist", []) + vals
        hist = hist[-60:]
        d[k] = {"median": float(np.median(hist)), "n": len(hist), "hist": hist}
        if len(d) > 50:
            d = dict(list(d.items())[-50:])
        _F0_BASE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _F0_BASE_FILE.write_text(json.dumps(d), encoding="utf-8")
    except Exception:
        pass


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
           candidates: int | None = None, draft: bool = False) -> dict:
    """draft=True: fast preview — no verification-gated second synthesis and no automatic re-rolls (QC still measures and
    reports). Same voice, same model, same processing; only the repair work is skipped."""
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
    enh_waves, seg_ctx = [], []
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
        if (not draft and sim is not None and engine != "expressive" and s.get("vocalized") and s.get("corrected")
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
        enh_waves.append(wave)
        seg_ctx.append({"ref": ref, "ref_text": ref_text, "speed": speed, "nfe": nfe, "exag": exag, "cfg": cfg_w, "spoken": s["spoken"]})
        pauses.append((P["pause_before"], P["pause_after"]))
        seg_meta.append({"id": s["id"], "ref_source": ref_src, "speed": round(speed, 3), "nfe": nfe, "diacritics": diac_note})

    # ================= QC pass: detect → re-synthesise ONLY the affected sentences → harmonise pace → process
    qc_notes: dict = {"fixed": [], "unresolved": []}
    use_asr = _verify_on() and engine != "expressive"
    def _asr_check(w, spoken):
        heard = asr.transcribe(refaudio.audio_resample16(np.asarray(w, dtype=np.float32)))["text"]
        return {"sim": asr.similarity(spoken, heard), **qc.word_diff(spoken, heard)}
    def _bad(c):
        return c["sim"] < MIN_INTELLIGIBILITY or c["missing"] > 0.25 or c["dup"] > 0
    def _score(c, spk):
        return c["sim"] - c["missing"] - 0.5 * min(c["dup"], 2) + 0.5 * (spk if spk is not None else 0.5)
    word_chk: dict = {}
    if use_asr:
        progress(0.80, "فحص الكلمات المفقودة/المكررة")
        for i, s in enumerate(segs):
            if len(s["spoken"]) >= 12:
                word_chk[i] = _asr_check(enh_waves[i], s["spoken"])
    f0_raw = {i: _f0_median_st(w) for i, w in enumerate(enh_waves) if len(w) >= 1.5 * audio.SR}
    ref_f0 = _f0_baseline(seg_ctx[0]["ref"]) if seg_ctx else None     # learned typical narration pitch (None until enough history)
    f0_a = qc.f0_assess([f0_raw.get(i) for i in range(len(segs))], ref_f0)        # robust statistics; reference anchor when few sentences
    med_f0 = f0_a["center"]
    f0_thr = f0_a["outlier_thr"] or 2.0
    def _f0pen(w):
        if med_f0 is None:
            return 0.0
        f = _f0_median_st(w) if len(w) >= 1.5 * audio.SR else None
        return 0.0 if f is None else 0.15 * min(abs(f - med_f0), 4.0)
    spk = {i: _spk_sim(enh_waves[i], seg_ctx[i]["ref"]) for i in range(len(segs))} if speaker.available() else {}
    med_spk = float(np.median([x for x in spk.values() if x is not None])) if any(x is not None for x in spk.values()) else None
    flagged = [i for i in range(len(segs)) if (i in word_chk and _bad(word_chk[i])) or
               (med_spk is not None and spk.get(i) is not None and spk[i] < med_spk - 0.18 and len(enh_waves[i]) >= 1.5 * audio.SR and len(segs) >= 3) or
               (med_f0 is not None and f0_raw.get(i) is not None and abs(f0_raw[i] - med_f0) > f0_thr)]
    if draft:
        flagged = []
    early_stop = os.environ.get("SVS_QC_EARLY_STOP", "1") != "0"     # 0 = always try every attempt (used to compare cost/quality)
    def _resolved(w, c, sp):
        """A candidate is accepted only if it has no word problem AND is not a pitch/identity outlier."""
        if c is not None and _bad(c):
            return False
        if med_f0 is not None and len(w) >= 1.5 * audio.SR:
            f = _f0_median_st(w)
            if f is not None and abs(f - med_f0) > f0_thr:
                return False
        if med_spk is not None and sp is not None and len(w) >= 1.5 * audio.SR and len(segs) >= 3 and sp < med_spk - 0.18:
            return False
        return True
    qc_notes["attempts"] = 0
    for i in flagged:
        c0 = seg_ctx[i]
        best_w, best_c, best_spk = enh_waves[i], word_chk.get(i), spk.get(i)
        best_sc = (_score(best_c, best_spk) if best_c else (best_spk or 0)) - _f0pen(best_w)
        for k in (1, 2):                    # attempt 2 happens only if attempt 1 did not resolve the problem (never lowered for word errors)
            progress(0.82, f"إصلاح الجملة {i + 1} (محاولة {k})")
            w2, _ = wave_for(c0["spoken"], c0["ref"], c0["ref_text"], c0["speed"], c0["nfe"],
                             (int(segs[i]["seed"]) + 15485863 * k) & 0x7FFFFFFF, verify=False, label="", frac=0.82,
                             exag=c0["exag"], cfg_w=c0["cfg"])
            qc_notes["attempts"] += 1
            c2 = _asr_check(w2, c0["spoken"]) if (use_asr and len(c0["spoken"]) >= 12) else None
            sp2 = _spk_sim(w2, c0["ref"]) if speaker.available() else None
            sc2 = (_score(c2, sp2) if c2 else (sp2 or 0)) - _f0pen(w2)
            if sc2 > best_sc + 0.02:
                best_w, best_c, best_spk, best_sc = w2, c2, sp2, sc2
            if early_stop and _resolved(best_w, best_c, best_spk):
                break
        improved = best_w is not enh_waves[i]
        enh_waves[i] = best_w
        if best_c:
            word_chk[i] = best_c
        spk[i] = best_spk
        (qc_notes["fixed"] if improved and _resolved(best_w, best_c, best_spk) else qc_notes["unresolved"]).append(segs[i]["id"])
    # pace harmonisation: pull every sentence toward ONE steady pace (relative to its planned speed), within ±12 %
    rates = [qc.chars_per_second(w, s["spoken"]) for w, s in zip(enh_waves, segs)]
    norm = [(r / c["speed"]) if r else None for r, c in zip(rates, seg_ctx)]
    valid = [x for x in norm if x]
    cv_before = cv_after = None
    stab = float(plan["controls"].get("stability", 0.6))
    if len(valid) >= 2:
        med = float(np.median(valid))
        cv_before = round(float(np.std(valid) / np.mean(valid)), 3)
        after = []
        for i, n_ in enumerate(norm):
            if not n_:
                continue
            f = float(np.clip(med / n_, 1 / 1.12, 1.12))
            f = 1 + (f - 1) * stab
            if abs(f - 1) > 0.015:
                try:
                    enh_waves[i] = audio.ffmpeg_wave(enh_waves[i], f"rubberband=tempo={f:.4f}:pitchq=quality" if audio.have_filter("rubberband") else f"atempo={f:.4f}")
                except Exception:
                    f = 1.0
            after.append(n_ * f)
        cv_after = round(float(np.std(after) / np.mean(after)), 3)
    # timbre/pitch harmonisation: sentences synthesised with different seeds (and re-rolled ones) must sound like ONE speaker.
    # Pull each sentence's pitch (unintended deviation only: planned pitch shifts are applied later) and long-term spectrum toward the
    # take's own median. Small, bounded corrections; skipped for short sentences where the estimates are unreliable.
    timbre = {"band_before": None, "band_after": None, "pitch_fixed": [], "eq_fixed": []}
    long_ix = [i for i, w in enumerate(enh_waves) if len(w) >= 1.5 * audio.SR]
    if len(long_ix) >= 3:
        f0s = {i: _f0_median_st(enh_waves[i]) for i in long_ix}
        have = [i for i in long_ix if f0s[i] is not None]
        a0 = qc.f0_assess([f0s[i] for i in have], ref_f0)
        timbre["f0_assess_before"] = {k: a0[k] for k in ("mode", "n", "spread", "limit", "ok")}
        if a0["mode"] != "insufficient":
            centre = a0["center"]
            for j, i in enumerate(have):
                dev = f0s[i] - centre
                # spread mode: pull toward the take median; anchor mode: only correct a sentence clearly away from the reference pitch
                if (a0["mode"] == "spread" and abs(dev) > 0.8) or (a0["mode"] == "anchor" and abs(dev) > qc.F0_ANCHOR_LIMIT - 0.5):
                    shift = float(np.clip(-dev * 0.8, -1.2, 1.2))
                    try:
                        enh_waves[i] = audio.ffmpeg_wave(enh_waves[i], audio.pitch_chain(0.0, shift) or "anull")
                        timbre["pitch_fixed"].append({"id": segs[i]["id"], "dev_st": round(dev, 2), "shift_st": round(shift, 2)})
                    except Exception:
                        pass
            f2 = [_f0_median_st(enh_waves[i]) for i in have]
            a1 = qc.f0_assess(f2, ref_f0)
            timbre["f0_assess_after"] = {k: a1[k] for k in ("mode", "n", "spread", "limit", "ok")}
        bands = {i: audio._band_levels(enh_waves[i], audio.SR) for i in long_ix}
        mid = [k for k, (c, _) in enumerate(audio._EQ_BANDS) if c in (500, 1000)]
        norm_b = {i: b - b[mid].mean() for i, b in bands.items()}
        target = np.median(np.stack(list(norm_b.values())), axis=0)
        timbre["band_before"] = round(float(np.mean([np.abs(norm_b[i] - target).mean() for i in long_ix])), 2)
        for i in long_ix:
            diff = target - norm_b[i]
            fl = []
            for (c, q), d_ in zip(audio._EQ_BANDS, diff):
                g_ = float(np.clip(d_ * 0.7, -2.5, 2.5))
                if abs(g_) >= 0.4:
                    fl.append(f"equalizer=f={c}:t=q:w={q}:g={g_:.1f}")
            if fl:
                try:
                    enh_waves[i] = audio.ffmpeg_wave(enh_waves[i], ",".join(fl))
                    timbre["eq_fixed"].append(segs[i]["id"])
                except Exception:
                    pass
        after_b = {i: (lambda b: b - b[mid].mean())(audio._band_levels(enh_waves[i], audio.SR)) for i in long_ix}
        timbre["band_after"] = round(float(np.mean([np.abs(after_b[i] - target).mean() for i in long_ix])), 2)
    if seg_ctx and len(f0_raw) >= 3:
        _f0_baseline_update(seg_ctx[0]["ref"], [f0_raw.get(i) for i in range(len(segs))])
    for i, s in enumerate(segs):
        proc_clips.append(_process_segment(enh_waves[i], s["params"], v.get("fx") or {}))
        seg_meta[i]["seconds"] = round(len(proc_clips[-1]) / audio.SR, 2)
    raw_clips = list(enh_waves)

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
    # final cleanup is a standard stage: measured, applied only where needed, guarded (see svs/clean.py)
    mix, clean_rep = _cleanup.polish(mix, force_denoise=bool(plan["controls"].get("denoise")))
    audio.write_wav(d / "mix_in.wav", mix)
    progress(0.90, "المعالجة الصوتية النهائية")
    profile = audio.spectral_profile(mix)
    ref_eq = None
    if plan.get("lock_identity"):
        # Your own voice: match the finished audio's tonal balance (warmth/brightness) to YOUR reference clip.
        try:
            ref0, _t0, _s0 = voices.reference_for(v, plan["dialect"], None)
            rw, rsr = sf.read(str(ref0), dtype="float32")
            rw = rw if rw.ndim == 1 else rw[:, 0]
            if rsr != audio.SR:
                rw = np.interp(np.linspace(0, len(rw) - 1, int(len(rw) * audio.SR / rsr)), np.arange(len(rw)), rw).astype(np.float32)
            ref_eq = audio.reference_eq(mix, rw)
        except Exception:
            ref_eq = None
    C = plan["controls"]
    af = audio.master_chain(v.get("fx"), profile, clean=bool(C.get("denoise")) or clean, ref_eq=ref_eq,
                            warmth_db=C.get("warmth_db", 0.0), clarity=C.get("clarity", 0.5), reverb=C.get("reverb", 0.0))
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
    # ---------- QC report (every number below is measured on THIS take)
    checks = []
    def add(cid, name, st, detail, **extra):
        checks.append({"id": cid, "name": name, "status": st, "detail": detail, **extra})
    mw, _ = sf.read(str(d / "master.wav"), dtype="float32")
    mw = mw if mw.ndim == 1 else mw[:, 0]
    clipped = float(np.mean(np.abs(mw) > 0.99))
    tp = m_master.get("true_peak_db")
    add("clipping", "التقطيع والتشويه", "ok" if (clipped == 0 and (tp is None or tp <= -1.0)) else "fail",
        f"ذروة {tp} dBTP · عيّنات مقصوصة {clipped * 100:.3f}%")
    nfv = qc.pause_noise_db(mw)
    cl_note = (" · عولج تلقائيًا: " + "، ".join(clean_rep["applied"])) if clean_rep["applied"] else ""
    add("noise", "الضوضاء والوشوشة", "ok" if (nfv is None or nfv <= -50) else "warn",
        (f"ضجيج الوقفات {round(nfv, 1)} dBFS (الحد −50)" if nfv is not None else "لا توجد وقفات منخفضة تُقاس (الأرضية رقمية/نظيفة)") + cl_note,
        cleanup=clean_rep)
    rtv = qc.reverb_tail_db(mw)
    add("reverb", "الصدى والارتداد", "ok" if (rtv is None or rtv <= -26 or C.get("reverb", 0) > 0) else "warn",
        f"ذيل ما بعد الكلام {round(rtv, 1) if rtv is not None else '—'} dB (الجاف ≲ −26)"
        + ("" if C.get("reverb", 0) == 0 else " · أُضيف صدى بطلبك") + (cl_note if "الصدى" in cl_note else ""))
    rms_resid = [audio.active_rms_db(c) - float(s["params"]["gain_db"]) for c, s in zip(proc_clips, segs)]
    sd = float(np.std(rms_resid)) if len(rms_resid) > 1 else 0.0
    add("level", "ثبات مستوى الصوت بين الجمل", "ok" if sd <= 1.2 else "warn", f"الانحراف {sd:.2f} dB بعد استبعاد الجهارة المخططة (الحد 1.2)")
    if cv_after is not None:
        off = [segs[i]["id"] for i, n_ in enumerate(norm) if n_ and abs(n_ - med) / med > 0.15]
        add("rate", "اتزان سرعة الحديث", "ok" if cv_after <= 0.10 else "warn",
            f"تذبذب السرعة بين الجمل (CV) {cv_before} ← {cv_after} بعد التوحيد (الحد 0.10)", before=cv_before, after=cv_after, off_by_15pct=off)
    else:
        add("rate", "اتزان سرعة الحديث", "skipped", "تحتاج جملتين طويلتين على الأقل للقياس")
    maxp = max([p_[1] for p_ in pauses] + [p_[0] for p_ in pauses])
    exp_len = sum(len(c) for c in proc_clips) / audio.SR + sum(a_ + b_ for a_, b_ in pauses)
    gap_err = abs(m_master["duration_s"] - exp_len)
    add("pauses", "الوقفات والفراغات والتداخل", "ok" if (maxp <= 3.0 and gap_err <= 0.6) else "warn",
        f"أطول وقفة {maxp:.2f} ث · فرق المدة عن المخطط {gap_err:.2f} ث")
    if word_chk:
        bad_now = [segs[i]["id"] for i, c in word_chk.items() if _bad(c)]
        add("words", "الكلمات المفقودة/المكررة والنطق الواضح", "ok" if not bad_now else "warn",
            ("كل الجمل سليمة" if not bad_now else f"جمل بها مشكلة: {', '.join(bad_now)}") + (f" · أُصلحت تلقائيًا: {', '.join(qc_notes['fixed'])}" if qc_notes["fixed"] else ""),
            per_sentence={segs[i]["id"]: {"similarity": round(c["sim"], 3), "missing": c["missing"], "dup": c["dup"]} for i, c in word_chk.items()})
    else:
        add("words", "الكلمات المفقودة/المكررة والنطق الواضح", "skipped", "التعرف الآلي غير متاح أو الجمل قصيرة")
    sv = [x for x in spk.values() if x is not None]
    if len(sv) >= 2:
        spread = max(sv) - min(sv)
        add("identity", "ثبات هوية الراوي بين الجمل", "ok" if spread <= 0.25 else "warn",
            f"تفاوت تشابه البصمة بين الجمل {spread:.2f} (الحد 0.25)" + (f" · أُعيدت: {', '.join(qc_notes['fixed'])}" if qc_notes["fixed"] else ""))
    else:
        add("identity", "ثبات هوية الراوي بين الجمل", "skipped", "نموذج البصمة غير متاح أو الجمل قصيرة")
    # processing transparency: the master must still be the same voice, only polished. INTENDED tonal change (reference-matching EQ,
    # persona/warmth/clarity filters actually present in the chain) is separated from UNWANTED change; artefacts that speaker similarity
    # cannot see (muffling, hiss, clipping, crushed dynamics, added reverb) are checked explicitly. Similarity alone is never the verdict.
    try:
        mi = np.asarray(mix, dtype=np.float32)               # the unmastered mix kept in memory (the file is removed after mastering)
        n_ = min(len(mi), len(mw))
        sp_pm = None
        if speaker.available():
            sp_pm = speaker.similarity(refaudio.audio_resample16(mi[:n_][: 24000 * 20]), refaudio.audio_resample16(mw[:n_][: 24000 * 20]))
        i_mean, i_max = qc.chain_intent(af)
        ref_arr = None
        if plan.get("lock_identity"):
            try:
                r0_, _t, _s = voices.reference_for(v, plan["dialect"], None)
                ref_arr, rsr_ = sf.read(str(r0_), dtype="float32")
                ref_arr = ref_arr if ref_arr.ndim == 1 else ref_arr[:, 0]
                if rsr_ != audio.SR:
                    ref_arr = np.interp(np.linspace(0, len(ref_arr) - 1, int(len(ref_arr) * audio.SR / rsr_)), np.arange(len(ref_arr)), ref_arr).astype(np.float32)
            except Exception:
                ref_arr = None
        tr = qc.transparency(mi[:n_], mw[:n_], ref=ref_arr, intended_db=i_mean, intended_max_db=i_max, sim=sp_pm)
        add("mastering", "المعالجة لم تغيّر طبيعة الصوت", "ok" if tr["ok"] else "warn",
            ("سليمة: " if tr["ok"] else "تحذير: " + "؛ ".join(tr["reasons"]) + " — ") +
            f"تغيّر الجرس {tr['tonal_change_db']} dB (المقصود {tr['intended_db']})"
            + (f" · تطابق البصمة {tr['speaker_sim']}" if "speaker_sim" in tr else ""), transparency=tr)
    except Exception as e:
        add("mastering", "المعالجة لم تغيّر طبيعة الصوت", "skipped", f"تعذّر القياس ({type(e).__name__})")
    fa_b, fa_a = timbre.get("f0_assess_before"), timbre.get("f0_assess_after")
    if fa_a and fa_a["mode"] != "insufficient":
        what = (f"انحراف النغمة القوي (MAD) بين {fa_a['n']} جملة {fa_b['spread']} ← {fa_a['spread']} نصف نغمة (الحد {fa_a['limit']})"
                if fa_a["mode"] == "spread" else
                f"عيّنة صغيرة ({fa_a['n']} جمل): لا يصح حساب انحراف بينها، فقُورنت كل جملة بنغمة الراوي المعتادة المتعلَّمة من نسخه السابقة — أبعد جملة {fa_a['spread']} نصف نغمة (الحد {fa_a['limit']})")
        add("pitch", "ثبات طبقة الصوت بين الجمل", "ok" if fa_a["ok"] else "warn",
            what + (f" · صُحّحت {len(timbre['pitch_fixed'])} جملة" if timbre["pitch_fixed"] else ""), pitch_stats=fa_a)
    else:
        add("pitch", "ثبات طبقة الصوت بين الجمل", "skipped", "عيّنة صغيرة ولا توجد بعد نغمة معتادة متعلَّمة لهذا الصوت (تتكوّن بعد ~10 جمل) — لم يُقدَّر انحراف غير موثوق")
    if timbre["band_before"] is not None:
        add("tone", "ثبات جرس الصوت بين الجمل", "ok" if timbre["band_after"] <= 1.5 else "warn",
            f"فرق الطيف بين الجمل {timbre['band_before']} ← {timbre['band_after']} dB (الحد 1.5)" + (f" · عُدّلت {len(timbre['eq_fixed'])} جملة" if timbre["eq_fixed"] else ""))
    else:
        add("tone", "ثبات جرس الصوت بين الجمل", "skipped", "الجمل قصيرة أو قليلة")
    st_all = "fail" if any(c["status"] == "fail" for c in checks) else ("warn" if any(c["status"] == "warn" for c in checks) else "ok")
    qc_report = {"status": st_all, "checks": checks, "fixed_sentences": qc_notes["fixed"], "unresolved_sentences": qc_notes["unresolved"], "repair_attempts": qc_notes.get("attempts", 0),
                 "rate_cv": {"before": cv_before, "after": cv_after}, "timbre": timbre, "draft": bool(draft)}
    took = time.time() - t0
    secs = m_master["duration_s"]
    if intel and all(v < 0.3 for v in intel.values()) and len(intel) == len(segs):
        raise RenderError("الصوت الناتج غير مفهوم في كل الجمل — غالبًا المقطع المرجعي ضعيف أو مشوّش. "
                          "أعد رفع مقطع أنظف (٥–١٠ ثوانٍ، بلا ضجيج) من تبويب الشخصيات.")
    take = {"id": take_id, "parent": parent, "intelligibility": intel, "baseline_kind": baseline_kind, "qc": qc_report,
            "voice_match": ({"final": None, "mean": round(float(np.mean(list(match.values()))), 3), "min": round(float(min(match.values())), 3),
                             "per_sentence": match, "candidates": candidates,
                             "scale": "غير المتشابهين ≈ 0.0–0.3 · نفس المتحدث ≈ 0.7+"} if match else None),
            "intelligibility_compare": cmp_intel, "created": time.time(), "plan": plan, "segments": seg_meta,
            "files": {"master_mp3": "master.mp3", "master_wav": "master.wav", "raw_mp3": "raw.mp3",
                      "segments": {s["id"]: f"seg_{s['id']}.mp3" for s in segs}},
            "metrics": {"before": m_raw, "after": m_master}, "profile": profile, "mastering": af, "ref_eq": ref_eq,
            "device": ENGINE.device if ENGINE.device != "uninitialised" else ENGINE.pick_device(),
            "engine": engine, "dialect_native": dcfg["native"], "dialect_note": dcfg["note"],
            "draft": bool(draft), "warnings": warnings, "took_s": round(took, 1), "realtime_factor": round(took / max(secs, 0.1), 1),
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


def finalize(take_id: str, *, progress=None, synth_fn=default_synth) -> dict:
    """Turn a DRAFT into a deliverable take: same plan and seeds (sentences come from the cache), but with the full quality
    pipeline — verification-gated diacritics, QC repairs/re-rolls, harmonisation. A draft is never exported as final without this."""
    old = load_take(take_id)
    plan = json.loads(json.dumps(old["plan"]))
    return render(plan, progress=progress, synth_fn=synth_fn, parent=take_id, draft=False)


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
