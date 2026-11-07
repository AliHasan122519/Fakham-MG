#!/usr/bin/env bash
# One-time setup. Everything here is free.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v ffmpeg >/dev/null || { echo "ثبّت FFmpeg: sudo apt install ffmpeg  |  brew install ffmpeg"; exit 1; }
python3 -m venv .venv
.venv/bin/pip install -q -U pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_models.py "${@:-najdi}"
echo "✔ جاهز. شغّل: scripts/start.sh"
