"""Round-trip check: transcribe a generated file with free faster-whisper and print the text.
usage: python scripts/verify_asr.py out.mp3"""
import subprocess, sys
import numpy as np
from faster_whisper import WhisperModel
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", sys.argv[1], "-ar", "16000", "-ac", "1", "-f", "f32le", "-"],
                     capture_output=True, check=True).stdout
segs, _ = WhisperModel("small", device="cpu", compute_type="int8").transcribe(np.frombuffer(raw, np.float32), language="ar")
print(" ".join(s.text for s in segs))
