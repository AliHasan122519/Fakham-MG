"""Speaker-similarity measurement and automatic best-of-N selection (SpeechBrain ECAPA-TDNN, Apache-2.0, runs locally).

Why: "make it sound exactly like my reference" needs a NUMBER, not an impression. The speaker embedding gives cosine
similarity between two voices. Useful anchors measured here (see docs/VOICE-MATCH.md): same speaker (two halves of one
clip) ≈ high; different speakers ≈ low. Zero-shot TTS lands in between; the app re-rolls the seed and keeps the closest
candidate (best-of-N) when it has the compute budget.
"""
from __future__ import annotations

import os
import threading

import numpy as np

from . import config

_LOCK = threading.Lock()
_M = {}


def available() -> bool:
    if os.environ.get("SVS_SPEAKER", "auto") == "off":
        return False
    try:
        import speechbrain  # noqa: F401
        return True
    except Exception:
        return False


def _model():
    with _LOCK:
        if "m" not in _M:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
            _M["m"] = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                     savedir=str(config.MODEL_DIR / "ecapa"),
                                                     run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})
            _M["torch"] = torch
    return _M["m"], _M["torch"]


def embed(wave16k: np.ndarray) -> np.ndarray:
    """Unit-norm ECAPA embedding of a mono 16 kHz float clip (needs ≥ ~1 s of speech)."""
    m, torch = _model()
    w = torch.tensor(np.asarray(wave16k, dtype=np.float32))[None, :]
    with torch.no_grad():
        e = m.encode_batch(w).squeeze().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def similarity(a16: np.ndarray, b16: np.ndarray) -> float:
    return float(np.dot(embed(a16), embed(b16)))
