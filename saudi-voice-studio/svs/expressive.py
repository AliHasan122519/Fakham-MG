"""Adapter for the optional expressive engine (Chatterbox Multilingual V2 weights, MIT license).

Why it exists: F5-family models (NAMAA/Habibi) have no emotion control. Chatterbox has a model-level
`exaggeration` (emotion intensity) input, so emotion intensity from the plan maps to something the model really uses.
Trade-offs measured here: Modern Standard Arabic only (no Saudi dialect), separate 3.3 GB weights and a separate
virtualenv; ~4-6x real time on CPU; outputs carry an inaudible Resemble watermark.
Disabled unless SVS_EXPRESSIVE_PY (python of its venv) and SVS_EXPRESSIVE_MODELS exist.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config

PY = Path(os.environ.get("SVS_EXPRESSIVE_PY", ""))
MODELS = Path(os.environ.get("SVS_EXPRESSIVE_MODELS", config.MODEL_DIR / "chatterbox"))
WORKER = Path(__file__).with_name("expressive_worker.py")
NEEDED = ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json"]
# how strongly each emotion should drive the model's exaggeration input (0.5 ≈ model default)
EMOTION_ENERGY = {"neutral": 0.5, "calm": 0.35, "formal": 0.5, "confidence": 0.75, "empathy": 0.7, "sadness": 0.8,
                  "grandeur": 0.9, "suspense": 0.9, "warning": 1.0, "joy": 1.0, "excitement": 1.2}


class ExpressiveUnavailable(RuntimeError):
    pass


def available() -> bool:
    return bool(str(PY)) and PY.is_file() and all((MODELS / f).is_file() for f in NEEDED)


def exaggeration_for(emotion: str, intensity: float, expressiveness: float) -> tuple[float, float]:
    """(exaggeration, cfg_weight). Per Chatterbox guidance: high exaggeration pairs with a lower cfg."""
    k = max(0.0, min(1.0, intensity)) * max(0.0, expressiveness)
    ex = 0.30 + 0.55 * k * EMOTION_ENERGY.get(emotion, 0.5) / 0.75
    ex = float(max(0.25, min(1.0, ex)))
    return round(ex, 2), (0.3 if ex > 0.7 else 0.5)


class _Worker:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.device = "n/a"

    def _start(self) -> None:
        if not available():
            raise ExpressiveUnavailable("المحرك التعبيري غير مثبّت (فعّل ENABLE_EXPRESSIVE_ENGINE في الدفتر).")
        env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
        env["MPLBACKEND"] = "Agg"
        self.proc = subprocess.Popen([str(PY), str(WORKER), str(MODELS)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
        ready = json.loads(self.proc.stdout.readline() or "{}")
        if not ready.get("ready"):
            self.stop()
            raise ExpressiveUnavailable("تعذّر تشغيل المحرك التعبيري.")
        self.device = ready.get("device", "cpu")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None

    def synth(self, text: str, ref: Path, *, exaggeration: float, cfg: float, seed: int) -> tuple[np.ndarray, int]:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "o.wav"
                self.proc.stdin.write(json.dumps({"text": text, "ref": str(ref), "out": str(out), "lang": "ar",
                                                  "exaggeration": exaggeration, "cfg": cfg, "seed": seed}) + "\n")
                self.proc.stdin.flush()
                reply = json.loads(self.proc.stdout.readline() or '{"ok": false, "error": "worker died"}')
                if not reply.get("ok"):
                    self.stop()
                    raise ExpressiveUnavailable("فشل المحرك التعبيري: " + reply.get("error", ""))
                w, sr = sf.read(str(out), dtype="float32")
                return (w if w.ndim == 1 else w[:, 0]), int(sr)


WORKER_POOL = _Worker()
