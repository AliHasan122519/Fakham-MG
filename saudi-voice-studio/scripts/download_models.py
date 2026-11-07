"""Download the free open weights from Hugging Face (no key, no account). ~1.3 GB per engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from huggingface_hub import hf_hub_download
from svs import config

which = sys.argv[1:] or ["najdi", "saudi"]
for key in which:
    spec = config.ENGINES[key]
    for f in (spec["file"], spec["vocab"]):
        print("->", hf_hub_download(spec["repo"], f, local_dir=config.MODEL_DIR))
print("تم. النماذج في", config.MODEL_DIR)
