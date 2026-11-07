#!/usr/bin/env bash
# Installs this folder as a Claude Code skill (symlink, so updates apply instantly).
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p ~/.claude/skills
ln -sfn "$SRC" ~/.claude/skills/saudi-voice-studio
echo "✔ المهارة مثبتة: ~/.claude/skills/saudi-voice-studio"
echo "أضف خادم MCP:  claude mcp add saudi-voice -- $SRC/.venv/bin/python -m svs.mcp_server"
