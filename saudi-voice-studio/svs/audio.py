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
                    "-ac", "1", str(dst_wav)], check=True)
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
