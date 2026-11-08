"""Download the free open weights from Hugging Face (no key, no account).
usage: download_models.py [najdi] [saudi] [expressive]   (~1.3 GB, ~1.3 GB, ~3.3 GB)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from huggingface_hub import hf_hub_download
from svs import config

EXPRESSIVE = ("ResembleAI/chatterbox", ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
                                        "grapheme_mtl_merged_expanded_v1.json", "conds.pt"])
which = sys.argv[1:] or ["najdi", "saudi"]
for key in which:
    if key == "expressive":
        for f in EXPRESSIVE[1]:
            print("->", hf_hub_download(EXPRESSIVE[0], f, local_dir=config.MODEL_DIR / "chatterbox"))
        continue
    spec = config.ENGINES[key]
    for f in (spec["file"], spec["vocab"]):
        print("->", hf_hub_download(spec["repo"], f, local_dir=config.MODEL_DIR))
print("تم. النماذج في", config.MODEL_DIR)
