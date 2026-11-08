"""Saudi Voice Studio — free, open-source Saudi Arabic voice-over studio."""
import os

__version__ = "0.1.0"

# The server is headless. Notebook environments (Colab/Jupyter) export
# MPLBACKEND=module://matplotlib_inline.backend_inline, which only exists inside an IPython kernel;
# our isolated venv has no matplotlib_inline, so `import matplotlib.pylab` (pulled in by f5-tts/librosa)
# would crash with "Key backend: ... is not a valid value for backend". Force a non-interactive backend
# BEFORE anything can import matplotlib. Unconditional on purpose: the inherited value is the bug.
os.environ["MPLBACKEND"] = "Agg"
