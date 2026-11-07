"""Central configuration. Everything is overridable with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("SVS_HOME", ROOT / "data"))          # outputs, history, user voices (git-ignored)
OUTPUTS = DATA / "outputs"
USER_VOICES = DATA / "voices"
USER_REFS = DATA / "refs"
HISTORY_DB = DATA / "history.sqlite3"
PRESETS = ROOT / "voices" / "presets.json"
BUNDLED_REFS = ROOT / "voices" / "refs"

# Model files. Defaults are downloaded on demand from Hugging Face (free, no key needed).
# Override SVS_MODEL_DIR to point at already-downloaded files (e.g. on Google Drive in Colab).
MODEL_DIR = Path(os.environ.get("SVS_MODEL_DIR", DATA / "models"))
ENGINES = {
    # Najdi fine-tune (CC-BY-NC-SA-4.0 — NON-COMMERCIAL).
    "najdi": dict(repo="NAMAA-Space/NAMAA-Saudi-TTS-V2", file="model_2000.safetensors", vocab="vocab.txt"),
    # Habibi-TTS Saudi-specialised base (CC-BY-NC-SA-4.0 — NON-COMMERCIAL).
    "saudi": dict(repo="SWivid/Habibi-TTS", file="Specialized/SAU/model_200000.safetensors",
                  vocab="Specialized/SAU/vocab.txt"),
}
DIT_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

DEVICE = os.environ.get("SVS_DEVICE", "auto")          # auto | cpu | cuda
DEFAULT_NFE = int(os.environ.get("SVS_NFE", "0"))        # 0 = auto (16 on CPU, voice default on GPU); 16 fast, 32 good, 64 best
MAX_CHARS = int(os.environ.get("SVS_MAX_CHARS", "2000"))

# Web / MCP access
TOKEN = os.environ.get("SVS_TOKEN", "")                # if set, API + downloads require it
PUBLIC_URL = os.environ.get("SVS_PUBLIC_URL", "http://127.0.0.1:8765").rstrip("/")
HOST = os.environ.get("SVS_HOST", "127.0.0.1")
PORT = int(os.environ.get("SVS_PORT", "8765"))

for _d in (OUTPUTS, USER_VOICES, USER_REFS, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)
