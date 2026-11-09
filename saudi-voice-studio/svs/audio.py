"""Audio post-processing with FFmpeg (free, local): speed-neutral pitch, EQ, reverb, loudness, MP3/WAV."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 24000


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def build_filter(fx: dict | None, pitch_st: float = 0.0, gain_db: float = 0.0) -> str:
    """Build an ffmpeg -af chain from a voice 'fx' dict.

    fx keys (all optional): bass_db, treble_db, reverb (0..1), compress (bool), warmth (bool),
    highpass_hz, lowpass_hz.
    """
    fx = fx or {}
    f: list[str] = []
    st = float(fx.get("pitch_st", 0.0)) + float(pitch_st)
    if abs(st) > 0.01:
        r = 2 ** (st / 12)
        f += [f"asetrate={int(SR * r)}", f"aresample={SR}", f"atempo={1 / r:.5f}"]
    if fx.get("highpass_hz"):
        f.append(f"highpass=f={int(fx['highpass_hz'])}")
    if fx.get("lowpass_hz"):
        f.append(f"lowpass=f={int(fx['lowpass_hz'])}")
    if fx.get("bass_db"):
        f.append(f"bass=g={float(fx['bass_db'])}:f=110:w=0.6")
    if fx.get("treble_db"):
        f.append(f"treble=g={float(fx['treble_db'])}:f=4500:w=0.7")
    if fx.get("compress"):
        f.append("acompressor=threshold=-20dB:ratio=3:attack=15:release=250:makeup=2")
    rv = float(fx.get("reverb", 0) or 0)
    if rv > 0:
        # Light "cinematic hall": two decaying echoes. Scales with rv.
        f.append(f"aecho=0.8:{0.25 + 0.35 * rv:.2f}:{int(60 + 80 * rv)}|{int(140 + 160 * rv)}:{0.18 * rv + 0.05:.2f}|{0.12 * rv + 0.03:.2f}")
    if gain_db:
        f.append(f"volume={gain_db}dB")
    f.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    return ",".join(f)


def write_wav(path: Path, wave: np.ndarray, sr: int = SR) -> None:
    sf.write(str(path), np.asarray(wave, dtype=np.float32), sr)


def process(src_wav: Path, dst_wav: Path, dst_mp3: Path | None, af: str) -> None:
    if not have_ffmpeg():
        raise RuntimeError("ffmpeg غير مثبت — ثبّته مجانًا: sudo apt install ffmpeg")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_wav), "-af", af, "-ar", str(SR),
                    "-ac", "1", "-c:a", "pcm_s24le", str(dst_wav)], check=True)
    if dst_mp3:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(dst_wav), "-codec:a", "libmp3lame",
                        "-b:a", "192k", str(dst_mp3)], check=True)


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


MAX_REF_BYTES = 25 * 1024 * 1024


def prepare_reference(src: Path, dst: Path, max_seconds: float = 10.0) -> Path:
    """Convert an uploaded clip to mono 24 kHz WAV (<= max_seconds, loudness-normalised).

    Security: the untrusted file is fed through stdin and ffmpeg is limited to the `pipe` protocol, so a
    crafted playlist (HLS/concat) cannot make ffmpeg read other local files or fetch network URLs.
    """
    data = Path(src).read_bytes()
    if len(data) > MAX_REF_BYTES:
        raise ValueError("المقطع المرجعي كبير جدًا (الحد 25MB).")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-protocol_whitelist", "pipe", "-t", "60", "-i", "pipe:0",
                        "-t", str(max_seconds), "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(SR),
                        "-af", "loudnorm=I=-20:TP=-2", str(dst)], input=data, capture_output=True, timeout=120)
    if r.returncode != 0 or not Path(dst).is_file():
        raise ValueError("تعذّر قراءة المقطع الصوتي (صيغة غير صالحة؟).")
    return dst


# ===================================================================== v2: segment DSP, mastering, metrics
def ffmpeg_wave(wave: np.ndarray, af: str, sr: int = SR) -> np.ndarray:
    """Run a float32 mono array through an ffmpeg -af chain (pipes only; no files, no protocols)."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0", "-af", af,
                        "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1"],
                       input=np.asarray(wave, dtype="<f4").tobytes(), capture_output=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg: " + r.stderr.decode("utf-8", "replace")[-300:])
    return np.frombuffer(r.stdout, dtype="<f4").astype(np.float32)


def have_filter(name: str) -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, timeout=20).stdout
        return f" {name} " in out
    except Exception:
        return False


def pitch_chain(voice_st: float, seg_st: float) -> str:
    """voice_st: identity shift (formants move with pitch => a different-sounding speaker).
    seg_st: expressive shift (formants preserved => same speaker, different intonation)."""
    parts = []
    if abs(voice_st) > 0.05:
        r = 2 ** (voice_st / 12)
        parts.append(f"rubberband=pitch={r:.5f}:formant=shifted:pitchq=quality" if have_filter("rubberband")
                     else f"asetrate={int(SR * r)},aresample={SR},atempo={1 / r:.5f}")
    if abs(seg_st) > 0.05:
        r = 2 ** (seg_st / 12)
        parts.append(f"rubberband=pitch={r:.5f}:formant=preserved:pitchq=quality" if have_filter("rubberband")
                     else f"asetrate={int(SR * r)},aresample={SR},atempo={1 / r:.5f}")
    return ",".join(parts)


def trim_silence(w: np.ndarray, sr: int = SR, thresh_db: float = -42.0, keep_ms: float = 25.0) -> np.ndarray:
    if len(w) < sr // 10:
        return w
    win = int(sr * 0.01)
    n = len(w) // win
    e = 20 * np.log10(np.sqrt((w[: n * win].reshape(n, win) ** 2).mean(axis=1)) + 1e-9)
    loud = np.where(e > (e.max() + thresh_db if e.max() > -60 else -80))[0]
    if len(loud) == 0:
        return w
    keep = int(keep_ms / 10)
    a, b = max(0, loud[0] - keep), min(n, loud[-1] + 1 + keep)
    return w[a * win: b * win]


def active_rms_db(w: np.ndarray, sr: int = SR) -> float:
    win = int(sr * 0.02)
    n = len(w) // win
    if n == 0:
        return -60.0
    r = np.sqrt((w[: n * win].reshape(n, win) ** 2).mean(axis=1))
    act = r[r > r.max() * 0.1] if r.max() > 0 else r
    return float(20 * np.log10(np.sqrt((act ** 2).mean()) + 1e-9))


def fade(w: np.ndarray, ms: float = 10.0, sr: int = SR) -> np.ndarray:
    n = min(int(sr * ms / 1000), len(w) // 2)
    if n > 0:
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        w = w.copy()
        w[:n] *= ramp
        w[-n:] *= ramp[::-1]
    return w


def spectral_profile(w: np.ndarray, sr: int = SR) -> dict:
    """Cheap tonal profile used for adaptive EQ: share of energy above 4 kHz and 8 kHz."""
    if len(w) < 2048:
        return {"hf4": 0.0, "hf8": 0.0}
    x = w[: sr * 30] * np.hanning(min(len(w), sr * 30))
    S = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    tot = S.sum() + 1e-12
    return {"hf4": float(S[f > 4000].sum() / tot), "hf8": float(S[f > 8000].sum() / tot)}


_EQ_BANDS = [(125, 1.0), (250, 1.0), (500, 1.0), (1000, 1.0), (2000, 1.0), (3150, 1.0), (5000, 0.9), (7000, 0.8)]


def _band_levels(x: np.ndarray, sr: int) -> np.ndarray:
    n = 2048
    fr = [x[i:i + n] * np.hanning(n) for i in range(0, len(x) - n, n // 2)]
    fr = [f for f in fr if np.sqrt((f ** 2).mean()) > 10 ** (-46 / 20)]
    if not fr:
        return np.zeros(len(_EQ_BANDS))
    S = np.mean([np.abs(np.fft.rfft(f)) ** 2 for f in fr], axis=0)
    f = np.fft.rfftfreq(n, 1 / sr)
    L = 10 * np.log10(S + 1e-12)
    return np.array([float(np.mean(L[(f >= c / 2 ** 0.5) & (f < c * 2 ** 0.5)])) for c, _ in _EQ_BANDS])


def reference_eq(generated: np.ndarray, reference: np.ndarray, sr: int = SR, max_db: float = 4.0, strength: float = 0.8) -> list[str]:
    """Tonal-balance matching: ffmpeg `equalizer` filters that nudge the GENERATED voice's long-term spectrum toward the
    reference clip's (warmth, brightness), at most ±max_db per band. Measured on a real reference: generated audio was
    ~4 dB thin below 300 Hz and ~4 dB brighter at 6-8 kHz than the reference."""
    g, r = _band_levels(generated, sr), _band_levels(reference, sr)
    mid = [i for i, (c, _) in enumerate(_EQ_BANDS) if c in (500, 1000)]
    diff = (r - r[mid].mean()) - (g - g[mid].mean())          # anchor both spectra at the speech-fundamental region
    out = []
    for (c, q), d in zip(_EQ_BANDS, diff):
        gain = float(np.clip(d * strength, -max_db, max_db))
        if abs(gain) >= 0.4:
            out.append(f"equalizer=f={c}:t=q:w={q}:g={gain:.1f}")
    return out


def master_chain(fx: dict | None, profile: dict, *, clean: bool = False, target_lufs: float = -16.0,
                 ref_eq: list[str] | None = None, warmth_db: float = 0.0, clarity: float = 0.5, reverb: float = 0.0) -> str:
    """Mastering chain, deliberately gentle so the voice keeps its character.

    Adaptive part: the model's vocoder output is often dull above 4 kHz (low `hf4`) or harsh/metallic around 7-9 kHz
    (high `hf8`). We pick a small presence/air lift or a small cut from the measured spectrum instead of a fixed preset.
    """
    fx = fx or {}
    f = ["highpass=f=70"]
    if clean:   # off by default: model output is already clean and denoising measurably reduced speaker similarity (-0.03)
        f.append("afftdn=nr=9:nf=-48:tn=0")
    f.append("deesser=i=0.15:m=0.5:f=0.5")
    hf4, hf8 = profile.get("hf4", 0.03), profile.get("hf8", 0.005)
    if ref_eq is not None:                                        # follow YOUR reference's tonal balance instead of presets
        f += ref_eq
    elif hf4 < 0.02:                                                # dull → a LIGHT presence/air lift (identity-safe)
        f += ["equalizer=f=3200:t=q:w=1.0:g=1.0", "treble=g=1.0:f=5500:w=0.7"]
    elif hf4 < 0.04:
        f += ["equalizer=f=3200:t=q:w=1.0:g=0.6", "treble=g=0.4:f=5500:w=0.7"]
    if ref_eq is None and hf8 > 0.006:                                               # harsh / metallic band → tame it
        f.append(f"equalizer=f=8000:t=q:w=1.2:g={-min(3.0, 1.0 + hf8 * 200):.1f}")
    if fx.get("bass_db"):
        f.append(f"bass=g={float(fx['bass_db'])}:f=110:w=0.6")
    if fx.get("treble_db"):
        f.append(f"treble=g={float(fx['treble_db'])}:f=4500:w=0.7")
    if fx.get("highpass_hz"):
        f.append(f"highpass=f={int(fx['highpass_hz'])}")
    if abs(warmth_db) >= 0.2:
        f.append(f"bass=g={float(warmth_db):.1f}:f=180:w=0.7")      # depth/warmth without touching pitch
    cl = (float(clarity) - 0.5) * 4.0                              # clarity 0..1 → -2..+2 dB presence at 3.2 kHz
    if abs(cl) >= 0.3:
        f.append(f"equalizer=f=3200:t=q:w=1.0:g={cl:.1f}")
    if fx.get("compress", True):
        f.append("acompressor=threshold=-21dB:ratio=2.2:attack=20:release=220:makeup=1.5")
    rv = float(reverb or 0)                                       # EXPLICIT only; the default sound is dry
    if rv > 0:
        f.append(f"aecho=0.8:{0.25 + 0.35 * rv:.2f}:{int(60 + 80 * rv)}|{int(140 + 160 * rv)}:{0.18 * rv + 0.05:.2f}|{0.12 * rv + 0.03:.2f}")
    f.append(f"loudnorm=I={target_lufs}:TP=-1.5:LRA=9")
    f.append("alimiter=limit=0.89:level=disabled")
    return ",".join(f)


def leveled_only(target_lufs: float = -16.0) -> str:
    """The 'before' baseline: loudness matched so A/B is not fooled by volume, nothing else touched."""
    return f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"


def encode(src_wav: Path, dst_mp3: Path, *, bitrate: str = "320k") -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_wav), "-ar", "44100", "-ac", "1",
                    "-codec:a", "libmp3lame", "-b:a", bitrate, str(dst_mp3)], check=True, timeout=300)


def _f0_track(x16: np.ndarray) -> np.ndarray:
    fr, hop, vals = 640, 320, []
    for i in range(0, len(x16) - fr, hop):
        s = x16[i:i + fr]
        if np.sqrt((s ** 2).mean()) < 0.02:
            continue
        s = s - s.mean()
        ac = np.correlate(s, s, "full")[fr - 1:]
        lo, hi = 40, 228
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[0] > 0 and ac[k] / ac[0] > 0.5:
            vals.append(16000 / k)
    return np.array(vals)


def metrics(path: Path) -> dict:
    """Objective, reproducible measurements (not a quality score): loudness, noise floor, tonal balance, pauses, pitch variation."""
    p = str(path)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", p, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=300).stderr
    def g(label):
        import re as _re
        m = _re.findall(label + r":\s+(-?[\d.]+)", r)
        return float(m[-1]) if m else None
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, timeout=300).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    dur = len(x) / 16000
    win = 160
    n = len(x) // win
    rms = np.sqrt((x[: n * win].reshape(n, win) ** 2).mean(axis=1)) if n else np.zeros(1)
    sil = rms < 0.01
    # pauses = silent runs > 0.25 s
    runs, cur = [], 0
    for s_ in sil:
        if s_:
            cur += 1
        else:
            if cur * 0.01 > 0.25:
                runs.append(cur * 0.01)
            cur = 0
    f0 = _f0_track(x)
    st_std = float(np.std(12 * np.log2(f0 / np.median(f0)))) if len(f0) > 20 else 0.0
    S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 if len(x) > 1024 else np.ones(2)
    fq = np.fft.rfftfreq(len(x), 1 / 16000) if len(x) > 1024 else np.array([0, 1.0])
    voiced = rms[rms > 0.02]
    return {"duration_s": round(dur, 2), "lufs": g("I"), "lra": g("LRA"), "true_peak_db": g("Peak"),
            "noise_floor_db": float(round(20 * np.log10(max(float(np.percentile(rms[rms > 1e-5], 10)) if (rms > 1e-5).sum() > 10 else 1e-6, 1e-6)), 1)),   # digital-silence frames (exact pauses) excluded
            "energy_above_4k_pct": round(100 * float(S[fq > 4000].sum() / (S.sum() + 1e-12)), 2),
            "pauses_over_250ms": len(runs), "pause_total_s": round(float(sum(runs)), 2),
            "pause_mean_s": round(float(np.mean(runs)), 2) if runs else 0.0,
            "pitch_variation_semitones": round(st_std, 2),
            "loudness_dynamics_db": round(float(np.std(20 * np.log10(voiced + 1e-6))), 2) if len(voiced) else 0.0}
