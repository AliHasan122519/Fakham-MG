"""Worker for the OPTIONAL expressive engine (Chatterbox Multilingual, MIT). Runs in its OWN virtualenv
(torch 2.6.0) so its pins can never break the main engine. Protocol: one JSON object per stdin line, one JSON
reply per stdout line. Everything else (library chatter, progress bars) is sent to stderr."""
import json
import os
import sys

_proto = os.fdopen(os.dup(1), "w", buffering=1)      # keep the real stdout for the protocol only
os.dup2(2, 1)                                         # anything libraries print() goes to stderr

import torch  # noqa: E402
import torchaudio as ta  # noqa: E402
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402


def main(model_dir: str) -> None:
    torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxMultilingualTTS.from_local(model_dir, dev)
    print(json.dumps({"ready": True, "device": dev, "sr": model.sr}), file=_proto, flush=True)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            torch.manual_seed(int(req.get("seed", 0)))
            wav = model.generate(req["text"], language_id=req.get("lang", "ar"), audio_prompt_path=req["ref"],
                                 exaggeration=float(req["exaggeration"]), cfg_weight=float(req["cfg"]))
            ta.save(req["out"], wav.cpu(), model.sr)
            print(json.dumps({"ok": True, "sr": model.sr, "seconds": wav.shape[-1] / model.sr}), file=_proto, flush=True)
        except Exception as e:                       # report, keep serving
            print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}), file=_proto, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
