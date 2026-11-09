"""Reference-voice ingestion: decode → analyse → (guarded) clean → validate transcript → preview → save.

Why this module exists (root cause of the "hiss / noise, no speech" reports, reproduced with real clips):
  1. F5-family models need the reference AUDIO and its TRANSCRIPT to correspond exactly. The old code cut every clip
     to 10 s (and F5 itself cuts to 12 s) but kept the full transcript → audio/text mismatch → the model cannot align
     the prompt and produces babble/hiss instead of speech.
  2. A noisy or band-limited clip is *cloned as is*: the model reproduces the hiss.
  3. Aggressive loudness normalisation (loudnorm on a clip with a high noise floor) lifts that hiss.
  4. Nothing ever verified that the clip contained speech, nor that the transcript matched it.
This module analyses and verifies BEFORE saving, rejects damaged clips with an explanation, never silently trims
speech without trimming its transcript, and only keeps a denoised version if it provably still contains the voice.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from . import asr, audio, config

SR = audio.SR
MAX_UPLOAD = 25 * 1024 * 1024
MAX_INPUT_S = 120.0
MIN_SPEECH_S = 3.0
IDEAL = (5.0, 10.0)
MAX_REF_S = 11.5           # F5 hard-clips at 12 s; stay below
ALLOWED_FORMATS = {"wav", "mp3", "flac", "ogg", "matroska", "webm", "mov", "mp4", "m4a", "3gp", "3g2", "mj2",
                   "aac", "amr", "caf", "aiff", "opus", "w64"}
PREVIEWS = config.DATA / "ref_preview"
PREVIEWS.mkdir(parents=True, exist_ok=True)
TEST_SENTENCE = "مرحبا بكم في استوديو الصوت السعودي، نسعد بخدمتكم دائما."


class RefError(ValueError):
    """User-facing rejection (message is Arabic and actionable)."""


# ------------------------------------------------------------------ decode
def _probe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-show_entries",
                        "format=format_name,duration:stream=codec_type,codec_name,sample_rate,channels",
                        "-of", "json", str(path)], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RefError("تعذّر قراءة الملف: ليس ملفًا صوتيًا صالحًا أو أنه تالف. جرّب تسجيلًا جديدًا بصيغة WAV أو MP3 أو M4A.")
    return json.loads(r.stdout or "{}")


def decode(path: Path) -> tuple[np.ndarray, dict]:
    """Decode any allow-listed audio container to mono float32 @24 kHz. Containers outside the allow-list
    (HLS/concat playlists, images, …) are refused before ffmpeg ever decodes them."""
    if Path(path).stat().st_size > MAX_UPLOAD:
        raise RefError("الملف كبير جدًا (الحد 25MB). قصّ المقطع إلى 5–10 ثوانٍ.")
    info = _probe(path)
    fmts = set(str(info.get("format", {}).get("format_name", "")).split(","))
    if not fmts or not fmts <= ALLOWED_FORMATS:
        raise RefError("صيغة الملف غير مدعومة. ارفع WAV أو MP3 أو M4A أو OGG أو WebM أو FLAC.")
    if not any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
        raise RefError("الملف لا يحتوي مسارًا صوتيًا.")
    meta = {"container": ",".join(sorted(fmts)),
            "codec": next((s.get("codec_name") for s in info["streams"] if s.get("codec_type") == "audio"), None),
            "src_sample_rate": int(next((s.get("sample_rate") or 0 for s in info["streams"] if s.get("codec_type") == "audio"), 0) or 0),
            "src_channels": int(next((s.get("channels") or 0 for s in info["streams"] if s.get("codec_type") == "audio"), 0) or 0)}
    r = subprocess.run(["ffmpeg", "-v", "error", "-protocol_whitelist", "file", "-t", str(MAX_INPUT_S), "-i", str(path),
                        "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                       capture_output=True, timeout=180)
    x = np.frombuffer(r.stdout, dtype=np.float32).copy()
    if r.returncode != 0 or len(x) < SR * 0.5:
        raise RefError("تعذّر فك ترميز الصوت (الملف تالف أو قصير جدًا). سجّل المقطع من جديد.")
    meta["input_seconds"] = round(len(x) / SR, 2)
    return np.nan_to_num(x), meta


# ------------------------------------------------------------------ analysis
def _frames(x: np.ndarray, win: int = int(SR * 0.02)) -> np.ndarray:
    n = len(x) // win
    return x[: n * win].reshape(n, win)


def _db(v):
    return 20 * np.log10(np.maximum(v, 1e-9))


def voiced_fraction(x: np.ndarray) -> float:
    """Share of the LOUDER half of the frames that carry a clear pitch (harmonic structure). Speech: 0.5-0.95.
    Broadband hiss/noise: ~0. (Calibrated on 6 real clips + added noise, see docs/REFERENCE-AUDIO.md.)"""
    x16 = audio_resample16(x)
    n, h = 640, 320
    if len(x16) < n * 3:
        return 0.0
    lv = np.array([np.sqrt((x16[i:i + n] ** 2).mean()) for i in range(0, len(x16) - n, h)])
    thr = np.percentile(lv, 50)
    voiced = total = 0
    for i, l in zip(range(0, len(x16) - n, h), lv):
        if l < thr or l < 1e-4:
            continue
        total += 1
        sg = x16[i:i + n] - x16[i:i + n].mean()
        ac = np.correlate(sg, sg, "full")[n - 1:]
        k = 40 + int(np.argmax(ac[40:228]))
        if ac[0] > 0 and ac[k] / ac[0] > 0.5:
            voiced += 1
    return voiced / max(1, total)


def analyse(x: np.ndarray) -> dict:
    """Objective quality indicators for a reference clip.
    `snr_db` is estimated as the gap between loud (p90) and quiet (p10) frames; on real clips with added white noise
    it tracks the true SNR within ~3 dB (true 20→19-25, 12→15-17, 8→11-13, 5→9-10, 3→7-9, 0→5-6)."""
    fr = _frames(x)
    lvl = _db(np.sqrt((fr ** 2).mean(axis=1)))
    p10, p50, p90 = (float(v) for v in np.percentile(lvl, [10, 50, 90]))
    snr = p90 - p10
    peak_db = float(_db(np.abs(x).max()))
    speech = lvl > (p10 + max(6.0, min(12.0, 0.5 * snr)))
    spec = np.abs(np.fft.rfft(fr[~speech] * np.hanning(fr.shape[1]), axis=1)) ** 2 + 1e-12 if (~speech).any() else None
    flat_quiet = float(np.mean(np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1))) if spec is not None else 0.0
    S = np.abs(np.fft.rfft(x[: SR * 30] * np.hanning(min(len(x), SR * 30)))) ** 2
    f = np.fft.rfftfreq(min(len(x), SR * 30), 1 / SR)
    band = 10 * np.log10(np.convolve(S, np.ones(41) / 41, "same") + 1e-12)
    ok = np.where(band > band.max() - 55)[0]
    cutoff = float(f[ok[-1]]) if len(ok) else 0.0
    return {"duration_s": round(len(x) / SR, 2), "speech_ratio": round(float(speech.mean()), 2), "snr_db": round(snr, 1),
            "noise_floor_dbfs": round(p10, 1), "peak_dbfs": round(peak_db, 1),
            "active_rms_dbfs": round(audio.active_rms_db(x), 1), "clipping_pct": round(float(100 * np.mean(np.abs(x) > 0.985)), 2),
            "voiced_ratio": round(float(voiced_fraction(x)), 2), "quiet_flatness": round(flat_quiet, 3),
            "bandwidth_hz": int(cutoff), "dc_offset": round(float(abs(np.mean(x))), 4)}


def audio_resample16(x: np.ndarray) -> np.ndarray:
    n = int(len(x) * 16000 / SR)
    if n < 2:
        return x
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def speech_bounds(x: np.ndarray, pad_s: float = 0.15) -> tuple[int, int]:
    fr = _frames(x)
    lvl = _db(np.sqrt((fr ** 2).mean(axis=1)))
    p10, p90 = np.percentile(lvl, [10, 90])
    if p90 - p10 < 11:                    # noise dominates: no reliable pauses, keep everything
        return 0, len(x)
    idx = np.where(lvl > p10 + max(6.0, min(12.0, 0.5 * (p90 - p10))))[0]
    if len(idx) == 0:
        return 0, len(x)
    win = fr.shape[1]
    return max(0, int(idx[0] * win - pad_s * SR)), min(len(x), int((idx[-1] + 1) * win + pad_s * SR))


# ------------------------------------------------------------------ cleaning (guarded)
def _ltas(x: np.ndarray) -> np.ndarray:
    n = 2048
    fr = [x[i:i + n] * np.hanning(n) for i in range(0, len(x) - n, n // 2)]
    fr = [s for s in fr if np.sqrt((s ** 2).mean()) > 10 ** (-48 / 20)]
    if not fr:
        return np.zeros(n // 2 + 1)
    S = np.mean([np.abs(np.fft.rfft(s)) ** 2 for s in fr], axis=0)
    f = np.fft.rfftfreq(n, 1 / SR)
    m = (f >= 150) & (f <= 5000)
    out = np.zeros_like(S)
    out[m] = 10 * np.log10(S[m] + 1e-12)
    return out[m]


def level(x: np.ndarray, target_rms_db: float = -23.0, max_gain_db: float = 24.0) -> np.ndarray:
    g = float(np.clip(target_rms_db - audio.active_rms_db(x), -20.0, max_gain_db))
    y = x * (10 ** (g / 20))
    pk = np.abs(y).max()
    if pk > 0.79:                                     # keep ≥2 dB headroom, no hard clipping
        y = y * (0.79 / pk)
    return audio.fade(y.astype(np.float32), 10.0)


def denoise(x: np.ndarray, noise_floor_db: float, snr: float) -> tuple[np.ndarray, str]:
    """Conservative noise reduction. Strength follows the measured SNR; never more than ~12 dB."""
    nr = 6 if snr >= 28 else 9 if snr >= 18 else 12
    chain = f"highpass=f=60,afftdn=nr={nr}:nf={max(-70, min(-30, noise_floor_db - 3)):.0f}:tn=1:tr=1"
    return audio.ffmpeg_wave(x, chain), chain


def guarded_clean(x: np.ndarray, rep: dict, transcript: str | None) -> tuple[np.ndarray, dict]:
    """Return (clean_clip, guard_report). The denoised clip is used ONLY if it provably still contains the voice:
    its long-term spectrum in the speech band must stay close to the original's (correlation ≥ 0.93), speech-band level
    must not drop by more than 3 dB relative to the full-band level, and (when ASR is available) the transcript match
    must not get worse."""
    base = level(x)
    g = {"denoise_applied": False, "reason": "", "ltas_corr": None, "speech_band_drop_db": None, "asr_before": None, "asr_after": None}
    if rep["snr_db"] >= 32 and rep["quiet_flatness"] < 0.3:
        g["reason"] = "المقطع نظيف أصلًا؛ لم يُطبَّق تنظيف."
        return base, g
    den, chain = denoise(x, rep["noise_floor_dbfs"], rep["snr_db"])
    g["chain"] = chain
    den = level(den)
    a, b = _ltas(base), _ltas(den)
    corr = float(np.corrcoef(a, b)[0, 1]) if len(a) == len(b) and a.std() > 0 and b.std() > 0 else 0.0
    # speech-band (300–3400 Hz) level relative to the rest, before vs after
    def band_level(w):
        S = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2
        f = np.fft.rfftfreq(len(w), 1 / SR)
        return 10 * np.log10(S[(f > 300) & (f < 3400)].sum() / (S.sum() + 1e-12) + 1e-12)
    drop = float(band_level(base[: SR * 15]) - band_level(den[: SR * 15]))
    g["ltas_corr"], g["speech_band_drop_db"] = round(corr, 3), round(drop, 2)
    ok = corr >= 0.93 and drop <= 3.0
    if ok and transcript and asr.available():
        sb = asr.similarity(transcript, asr.transcribe(audio_resample16(base))["text"])
        sa = asr.similarity(transcript, asr.transcribe(audio_resample16(den))["text"])
        g["asr_before"], g["asr_after"] = round(sb, 3), round(sa, 3)
        ok = sa >= sb - 0.08
    if ok:
        g.update(denoise_applied=True, reason="تم تنظيف ضجيج خفيف وتأكدتُ أن الصوت لم يتأثر.")
        return den, g
    g["reason"] = "التنظيف كان سيؤذي صوت المتحدث، فأبقيتُ المقطع الأصلي دون تنظيف."
    return base, g


def guarded_dereverb(x: np.ndarray, transcript: str | None) -> tuple[np.ndarray, dict]:
    """Late-reverberation suppression on a reference clip, kept only if it provably helps and does not hurt the voice
    (tail drops ≥3 dB, tonal balance within 2 dB, ASR match not worse by more than 0.08)."""
    from . import clean as _clean, qc as _qc
    g: dict = {"applied": False, "tail_before": None, "tail_after": None, "t60": None}
    t0 = _qc.reverb_tail_db(x)
    g["tail_before"] = None if t0 is None else round(t0, 1)
    if t0 is None or t0 <= _clean.REVERB_APPLY_DB:
        return x, g
    t60 = _clean.estimate_t60(x) or 0.45
    g["t60"] = round(t60, 2)
    cand = level(_clean.dereverb(x, t60))
    t1 = _qc.reverb_tail_db(cand)
    g["tail_after"] = None if t1 is None else round(t1, 1)
    dist = _clean._tonal_distance(x, cand)
    g["tonal_distance_db"] = round(dist, 2)
    ok = t1 is not None and t1 < t0 - 3 and dist <= 2.0
    if ok and transcript and asr.available():
        sb = asr.similarity(transcript, asr.transcribe(audio_resample16(x))["text"])
        sa = asr.similarity(transcript, asr.transcribe(audio_resample16(cand))["text"])
        g["asr_before"], g["asr_after"] = round(sb, 3), round(sa, 3)
        ok = sa >= sb - 0.08
    if ok:
        g["applied"] = True
        return cand, g
    return x, g


# ------------------------------------------------------------------ verdict
def verdict(rep: dict, speech_s: float) -> dict:
    """status: ok | warn | reject, with Arabic reasons and tips."""
    reasons: list[tuple[str, str]] = []   # (severity, message)
    def add(sev, msg):
        reasons.append((sev, msg))
    if rep["active_rms_dbfs"] < -50:
        add("reject", "المقطع صامت تقريبًا. سجّل بصوت أعلى وأقرب للميكروفون.")
    if speech_s < MIN_SPEECH_S:
        add("reject", f"الكلام في المقطع قصير جدًا ({speech_s:.1f} ث). المطلوب 5–10 ثوانٍ من كلام متصل.")
    if rep["voiced_ratio"] < 0.15:
        add("reject", "لا أجد كلامًا واضحًا في المقطع (يبدو ضوضاء/شرشرة/موسيقى). هذا بالضبط ما ينتج صوتًا مشوّشًا.")
    if rep["clipping_pct"] > 3:
        add("reject", f"المقطع مشوَّه بقطع الذروة ({rep['clipping_pct']}%). خفّض مستوى التسجيل وأعد المحاولة.")
    elif rep["clipping_pct"] > 0.5:
        add("warn", f"فيه قطع طفيف في الذروة ({rep['clipping_pct']}%). يُفضّل تسجيل أهدأ.")
    if rep["snr_db"] < 11:
        add("reject", f"الضجيج عالٍ جدًا (نسبة الإشارة للضجيج ≈ {rep['snr_db']} dB؛ المطلوب ≥ 11). "
                      "النموذج يستنسخ الضجيج فيخرج صوت شرشرة بلا كلام. سجّل في مكان هادئ وقريبًا من الميكروفون.")
    elif rep["snr_db"] < 20:
        add("warn", f"فيه ضجيج ملحوظ (≈ {rep['snr_db']} dB). النموذج سيستنسخ جزءًا منه؛ الأفضل تسجيل أنظف.")
    if rep["bandwidth_hz"] and rep["bandwidth_hz"] < 3800:
        add("warn", "المقطع ضيّق النطاق (يشبه تسجيل مكالمة هاتفية). سيخرج الصوت مكتومًا.")
    if speech_s > MAX_REF_S:
        add("info", f"المقطع طويل ({speech_s:.0f} ث)؛ سأقتطع أفضل جزء بحدود الكلمات مع نصه.")
    elif speech_s < IDEAL[0]:
        add("warn", f"المقطع أقصر من المثالي ({speech_s:.1f} ث). المثالي 5–10 ثوانٍ.")
    order = {"reject": 2, "warn": 1, "info": 0}
    status = {2: "reject", 1: "warn", 0: "ok"}[max([order[s] for s, _ in reasons] or [0])]
    score = 100
    score -= max(0, 30 - rep["snr_db"]) * 1.4
    score -= min(30, rep["clipping_pct"] * 6)
    score -= 15 if (rep["bandwidth_hz"] and rep["bandwidth_hz"] < 3800) else 0
    score -= 12 if speech_s < IDEAL[0] else 0
    score -= 40 if status == "reject" else 0
    return {"status": status, "score": int(max(0, min(100, score))), "messages": [m for _, m in reasons],
            "reject": [m for s, m in reasons if s == "reject"], "warn": [m for s, m in reasons if s == "warn"]}


# ------------------------------------------------------------------ main entry points
def analyze_upload(src: Path, transcript: str = "", *, use_asr: bool = True) -> dict:
    """Full ingestion. Returns a preview dict (never writes into the voice store). Raises RefError on unusable input."""
    t0 = time.time()
    x, meta = decode(src)
    rep_full = analyse(x)
    a, b = speech_bounds(x)
    speech = x[a:b]
    rep = analyse(speech) if len(speech) > SR else rep_full
    speech_s = (b - a) / SR
    v = verdict(rep, speech_s)
    transcript = (transcript or "").strip()
    asr_ok = use_asr and asr.available()
    res = {"meta": meta, "quality": rep, "verdict": v, "speech_seconds": round(speech_s, 2), "asr": {"available": asr_ok},
           "transcript_in": transcript, "transcript": transcript, "auto_trimmed": False, "guard": {}}
    if v["status"] == "reject":
        res["seconds"] = round(time.time() - t0, 1)
        return res
    clip = speech
    # --- long clip: cut at a word boundary AND cut the transcript with it (needs ASR alignment)
    if speech_s > MAX_REF_S:
        if not asr_ok:
            v["status"] = "reject"
            v["reject"].append(f"المقطع أطول من 12 ثانية ({speech_s:.0f} ث) ولا يمكنني اقتطاعه مع نصه دون التعرف الآلي. "
                               "اقصصه إلى 5–10 ثوانٍ مع نصه.")
            v["messages"] = v["reject"] + v["warn"]
            return res
        tw = asr.transcribe(audio_resample16(speech), words=True)
        words = tw["words"]
        if len(words) < 3:
            raise RefError("لم أتعرف على كلام كافٍ في المقطع الطويل. قصّه وأعد المحاولة.")
        end_i = 0
        for i, w in enumerate(words):
            if w["end"] <= 10.0:
                end_i = i
        cut = words[end_i]["end"]
        if cut < 4.0:
            raise RefError("لم أجد وقفة مناسبة لاقتطاع المقطع. قصّه يدويًا إلى 5–10 ثوانٍ.")
        clip = speech[: int((cut + 0.12) * SR)]
        transcript = " ".join(w["w"] for w in words[: end_i + 1])
        res.update(auto_trimmed=True, transcript=transcript,
                   note="اقتُطع المقطع عند حدّ كلمة ونصه مُفرَّغ آليًا للجزء المقصوص — راجعه.")
    clip_rep = analyse(clip)
    cleaned, guard = guarded_clean(clip, clip_rep, transcript if transcript else None)
    cleaned, guard["dereverb"] = guarded_dereverb(cleaned, transcript if transcript else None)
    if guard["dereverb"].get("tail_before") is not None and guard["dereverb"]["tail_before"] > -26:
        msg = (f"المقطع فيه صدى غرفة (ذيل {guard['dereverb']['tail_before']} dB)؛ النموذج يستنسخ الصدى مع الصوت. "
               + ("عالجتُه تلقائيًا وتأكدتُ أن الصوت لم يتأثر." if guard["dereverb"].get("applied") else
                  "لم أستطع معالجته بأمان؛ الأفضل تسجيل المقطع في غرفة مفروشة أو قرب الميكروفون."))
        if v["status"] == "ok":
            v["status"] = "warn"
        v["warn"].append(msg)
        v["messages"] = v["reject"] + v["warn"]
    res["guard"] = guard
    # --- transcript ↔ audio verification (the guard against garbled output)
    if asr_ok:
        heard = asr.transcribe(audio_resample16(cleaned))["text"]
        res["asr"].update(text=heard)
        if not transcript:
            res.update(transcript=heard, auto_transcribed=True,
                       note=(res.get("note", "") + " النص مُفرَّغ آليًا — راجعه وصحّحه قبل الحفظ.").strip())
            if not heard.strip():
                v["status"] = "reject"
                v["reject"].append("لم أسمع كلامًا يمكن التعرف عليه في المقطع.")
        else:
            sim = asr.similarity(transcript, heard)
            res["asr"]["similarity"] = round(sim, 3)
            if sim < 0.45:
                v["status"] = "reject"
                v["reject"].append("النص الذي كتبتَه لا يطابق ما في المقطع (نسبة التطابق "
                                   f"{int(sim * 100)}%). هذا سبب الشرشرة: النموذج يحتاج النص المطابق حرفيًا. "
                                   f"سمعتُ: «{heard[:80]}»")
            elif sim < 0.7:
                if v["status"] == "ok":
                    v["status"] = "warn"
                v["warn"].append(f"النص يطابق المقطع جزئيًا ({int(sim * 100)}%). تحقق منه. سمعتُ: «{heard[:80]}»")
        v["messages"] = v["reject"] + v["warn"]
    else:
        if not transcript:
            v["status"] = "reject"
            v["reject"].append("النص المنطوق في المقطع مطلوب (التعرف الآلي غير مثبّت).")
        else:
            v["warn"].append("لم أتحقق من تطابق النص مع المقطع (التعرف الآلي غير مثبّت).")
        v["messages"] = v["reject"] + v["warn"]
    pid = uuid.uuid4().hex[:12]
    d = PREVIEWS / pid
    d.mkdir(parents=True)
    orig = level(clip)
    sf.write(str(d / "original.wav"), orig, SR, subtype="PCM_16")
    sf.write(str(d / "processed.wav"), cleaned, SR, subtype="PCM_16")
    (d / "meta.json").write_text(json.dumps({"created": time.time(), "transcript": res["transcript"]}), encoding="utf-8")
    after = analyse(cleaned)
    res.update(preview_id=pid, quality_after=after, seconds=round(time.time() - t0, 1),
               final_seconds=round(len(cleaned) / SR, 2))
    _prune_previews()
    return res


def _prune_previews(max_keep: int = 20, max_age_s: int = 3600) -> None:
    ds = sorted([p for p in PREVIEWS.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    for p in ds[:-max_keep] + [p for p in ds if time.time() - p.stat().st_mtime > max_age_s]:
        shutil.rmtree(p, ignore_errors=True)


def preview_file(pid: str, name: str) -> Path:
    import re
    if not re.fullmatch(r"[a-f0-9]{12}", pid) or name not in ("original.wav", "processed.wav", "test.wav"):
        raise RefError("ملف المعاينة غير صالح.")
    p = PREVIEWS / pid / name
    if not p.is_file():
        raise RefError("انتهت صلاحية المعاينة؛ أعد رفع المقطع.")
    return p


def commit(voice_id: str, preview_id: str, transcript: str, emotion: str | None = None, *, use: str = "processed") -> Path:
    """Save a previewed clip as the reference of a voice. `use`: processed | original."""
    from . import voices
    voices.check_id(voice_id)
    if use not in ("processed", "original"):
        raise RefError("اختيار غير صالح.")
    src = preview_file(preview_id, f"{use}.wav")
    transcript = (transcript or "").strip()
    if not transcript:
        raise RefError("النص المنطوق في المقطع مطلوب.")
    if len(transcript) > 600:
        raise RefError("النص طويل جدًا (الحد 600 حرف؛ المقطع المرجعي قصير).")
    if emotion is not None and not __import__("re").match(r"^[a-z]{3,12}$", emotion):
        raise RefError("اسم العاطفة غير صالح.")
    suffix = f".{emotion}" if emotion else ""
    dst = config.USER_REFS / f"{voice_id}{suffix}.wav"
    shutil.copyfile(src, dst)
    (config.USER_REFS / f"{voice_id}{suffix}.txt").write_text(transcript, encoding="utf-8")
    if not emotion:
        try:                              # estimate gender from pitch so derived narrators pick a matching base voice
            g = estimate_gender(sf.read(str(dst), dtype="float32")[0])
            jp = config.USER_VOICES / f"{voice_id}.json"
            if jp.is_file():
                vj = json.loads(jp.read_text(encoding="utf-8"))
                vj["gender"] = g
                jp.write_text(json.dumps(vj, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    if not emotion:                       # the first voice you add with your own reference becomes the default voice
        try:
            if not voices._DEFAULT_FILE.exists():
                voices.set_default_voice(voice_id)
        except Exception:
            pass
    return dst


def estimate_gender(x: np.ndarray) -> str:
    """male/female from median pitch (≥165 Hz → female). A rough rule, used only to pick a matching base voice."""
    f0 = audio._f0_track(audio_resample16(x))
    return "female" if len(f0) > 10 and float(np.median(f0)) >= 165 else "male"


def looks_like_speech(wave: np.ndarray) -> dict:
    """WEAK check on a generated clip: is there harmonic (voiced) content at all? It catches pure hiss only. In the
    diagnosis the garbled output of a 3 dB-SNR reference still looked voiced (0.64) while its transcript match was 0.00,
    so the real guard for generated audio is asr-based (see render._verify_segment)."""
    if len(wave) < SR // 2:
        return {"ok": False, "voiced_ratio": 0.0, "reason": "قصير جدًا"}
    vr = float(voiced_fraction(wave))
    return {"ok": vr >= 0.15, "voiced_ratio": round(vr, 2), "reason": "" if vr >= 0.15 else "لا يبدو كلامًا (ضوضاء/شرشرة)"}


def test_voice(preview_id: str, dialect: str = "najdi", progress=None) -> dict:
    """End-to-end check BEFORE saving: synthesise a standard sentence with the candidate reference and verify with ASR that
    the result is intelligible speech. This is the definitive test for the hiss/noise failure."""
    from . import voices
    from .engine import ENGINE
    progress = progress or (lambda p, m: None)
    d = PREVIEWS / preview_id
    meta = json.loads((preview_file(preview_id, "processed.wav").parent / "meta.json").read_text(encoding="utf-8"))
    progress(0.1, "توليد جملة اختبار بالمقطع المرجعي")
    engine = voices.dialects().get(dialect, {}).get("engine", "najdi")
    w, sr = ENGINE.synth(TEST_SENTENCE, str(preview_file(preview_id, "processed.wav")), meta["transcript"], engine=engine,
                         speed=1.0, nfe_step=16, seed=11)
    sf.write(str(d / "test.wav"), np.asarray(w, dtype=np.float32), sr, subtype="PCM_16")
    out = {"seconds": round(len(w) / sr, 2), "voiced_ratio": looks_like_speech(w)["voiced_ratio"], "asr_available": asr.available()}
    if asr.available():
        heard = asr.transcribe(audio_resample16(np.asarray(w, dtype=np.float32)))["text"]
        sim = asr.similarity(TEST_SENTENCE, heard)
        out.update(heard=heard, similarity=round(sim, 3), passed=bool(sim >= 0.6))
        out["message"] = ("اجتاز الاختبار: الصوت الناتج مفهوم." if out["passed"] else
                          f"فشل الاختبار: الصوت الناتج غير مفهوم (تطابق {int(sim * 100)}%). لا تحفظ هذا المقطع؛ سجّل مقطعًا أنظف.")
    else:
        out.update(passed=None, message="تم التوليد؛ استمع للنتيجة (التحقق الآلي غير مثبّت).")
    progress(1.0, "تم")
    return out
