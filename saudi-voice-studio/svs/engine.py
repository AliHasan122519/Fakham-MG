"""TTS engine wrapper: NAMAA Saudi TTS V2 (F5-TTS DiT, Najdi) and Habibi-TTS Saudi base.

Both are F5-TTS flow-matching models that clone the voice of a short reference clip.
Weights are downloaded on first use from Hugging Face (free, public, no token).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from . import config


class EngineUnavailable(RuntimeError):
    pass


class Engine:
    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._vocoder = None
        self._lock = threading.Lock()
        self.device = "uninitialised"

    # ---- environment -------------------------------------------------
    def pick_device(self) -> str:
        import torch
        want = config.DEVICE
        if want == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return want

    def status(self) -> dict:
        info: dict = {"engines": {}, "device": None, "gpu": False, "ffmpeg": None, "loaded": list(self._models)}
        try:
            import torch
            info["gpu"] = torch.cuda.is_available()
            info["device"] = self.device if self._models else self.pick_device()
            info["torch"] = torch.__version__
            info["threads"] = torch.get_num_threads()
        except Exception as e:  # torch missing
            info["error"] = f"torch غير مثبت: {e}"
        from .audio import have_ffmpeg
        info["ffmpeg"] = have_ffmpeg()
        for key, spec in config.ENGINES.items():
            ckpt = self._local(spec["file"])
            info["engines"][key] = {"repo": spec["repo"], "downloaded": ckpt.is_file(), "license": "CC-BY-NC-SA-4.0 (غير تجاري)"}
        return info

    @staticmethod
    def _local(rel: str) -> Path:
        return config.MODEL_DIR / rel

    def _fetch(self, spec: dict) -> tuple[str, str]:
        ckpt, vocab = self._local(spec["file"]), self._local(spec["vocab"])
        if ckpt.is_file() and vocab.is_file():
            return str(ckpt), str(vocab)
        from huggingface_hub import hf_hub_download
        p1 = hf_hub_download(spec["repo"], spec["file"], local_dir=config.MODEL_DIR)
        p2 = hf_hub_download(spec["repo"], spec["vocab"], local_dir=config.MODEL_DIR)
        return p1, p2

    # ---- loading -----------------------------------------------------
    def _load(self, key: str):
        if key in self._models:
            return self._models[key]
        try:
            import torch
            from f5_tts.infer.utils_infer import load_model, load_vocoder
            from f5_tts.model import DiT
        except Exception as e:
            raise EngineUnavailable(f"مكتبات المحرك غير مثبتة (pip install -r requirements.txt): {e}")
        self.device = self.pick_device()
        ckpt, vocab = self._fetch(config.ENGINES[key])
        model = load_model(DiT, config.DIT_CFG, ckpt, vocab_file=vocab, device=self.device)
        model = model.to(torch.float32).eval()
        self._models[key] = model
        if self._vocoder is None:
            self._vocoder = load_vocoder(device=self.device)
        return model

    # ---- synthesis ---------------------------------------------------
    def synth(self, text: str, ref_audio: str, ref_text: str, *, engine: str = "najdi",
              speed: float = 1.0, nfe_step: int = 32, seed: int | None = None) -> tuple[np.ndarray, int]:
        with self._lock:  # one generation at a time: CPU/GPU memory safety
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text
            try:
                from habibi_tts.infer.utils_infer import infer_process
            except Exception:
                from f5_tts.infer.utils_infer import infer_process
            model = self._load(engine)
            if seed is not None:
                import torch
                torch.manual_seed(seed)
            ref_a, ref_t = preprocess_ref_audio_text(ref_audio, ref_text)
            wave, sr, _ = infer_process(ref_a, ref_t, text, model, self._vocoder,
                                        nfe_step=int(nfe_step), speed=float(speed))
            return np.asarray(wave, dtype=np.float32), int(sr)


ENGINE = Engine()
