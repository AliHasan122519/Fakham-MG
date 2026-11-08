"""Download the free open weights from Hugging Face (no key, no account).
usage: download_models.py [najdi] [saudi] [semantic] [whisper] [expressive]   (~1.3, ~1.3, ~0.5, ~0.5, ~3.3 GB)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from huggingface_hub import hf_hub_download
from svs import config

EXPRESSIVE = ("ResembleAI/chatterbox", ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
                                        "grapheme_mtl_merged_expanded_v1.json", "conds.pt"])
which = sys.argv[1:] or ["najdi", "saudi"]
for key in which:
    if key == "semantic":                      # sentence-embedding model for the smart analysis (MIT, ~470 MB)
        from transformers import AutoModel, AutoTokenizer
        for cls in (AutoTokenizer, AutoModel):
            cls.from_pretrained("intfloat/multilingual-e5-small", cache_dir=str(config.MODEL_DIR / "hf"))
        print("-> semantic model ready")
        continue
    if key == "whisper":                       # speech recognizer used as a safety net (MIT, ~460 MB)
        from faster_whisper.utils import download_model
        print("->", download_model("small", cache_dir=str(config.MODEL_DIR / "whisper")))
        continue
    if key == "expressive":
        for f in EXPRESSIVE[1]:
            print("->", hf_hub_download(EXPRESSIVE[0], f, local_dir=config.MODEL_DIR / "chatterbox"))
        continue
    spec = config.ENGINES[key]
    for f in (spec["file"], spec["vocab"]):
        print("->", hf_hub_download(spec["repo"], f, local_dir=config.MODEL_DIR))
print("تم. النماذج في", config.MODEL_DIR)
