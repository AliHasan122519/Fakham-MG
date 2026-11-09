"""Saudi Voice Studio — free, open-source Saudi Arabic voice-over studio."""
import os

__version__ = "1.0.0-rc.3"

# The server is headless. Notebook environments (Colab/Jupyter) export
# MPLBACKEND=module://matplotlib_inline.backend_inline, which only exists inside an IPython kernel;
# our isolated venv has no matplotlib_inline, so `import matplotlib.pylab` (pulled in by f5-tts/librosa)
# would crash with "Key backend: ... is not a valid value for backend". Force a non-interactive backend
# BEFORE anything can import matplotlib. Unconditional on purpose: the inherited value is the bug.
os.environ["MPLBACKEND"] = "Agg"

# Memory: with bf16 autocast on CPU, oneDNN JIT-compiles and CACHES a kernel set for every new tensor shape (each sentence has a
# different length). Measured on the real engine: RSS grew ~0.2 GB per sentence without limit (2.9 -> 4.3 GB in 7 sentences);
# fp32 stayed flat (2.41 GB over 8 sentences). Capping the primitive cache bounds it (plateau ~3.1 GB) at no measurable speed cost.
# Must be set before torch is imported; a user-provided value always wins.
os.environ.setdefault("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "8")

try:                                  # cap glibc malloc arenas before any worker thread exists (see svs/memory.py)
    from . import memory as _memory
    _memory.tune()
except Exception:                     # pragma: no cover - never block start-up on an optimisation
    pass
