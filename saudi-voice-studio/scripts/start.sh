#!/usr/bin/env bash
# Starts web UI + MCP (/mcp) + downloads on one port. If `cloudflared` is installed, also opens a FREE
# public HTTPS tunnel (no account) so Claude on your phone can reach it. Secret token protects everything.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
export SVS_PORT="${SVS_PORT:-8765}"
if [ -z "${SVS_TOKEN:-}" ]; then
  [ -f .svs_token ] || $PY -c "import secrets;print(secrets.token_urlsafe(24))" > .svs_token
  export SVS_TOKEN="$(cat .svs_token)"
fi
if command -v cloudflared >/dev/null && [ -z "${SVS_PUBLIC_URL:-}" ]; then
  cloudflared tunnel --url "http://127.0.0.1:$SVS_PORT" --no-autoupdate > /tmp/svs_tunnel.log 2>&1 &
  for _ in $(seq 1 30); do
    URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/svs_tunnel.log | head -1 || true)
    [ -n "$URL" ] && break; sleep 1
  done
  export SVS_PUBLIC_URL="${URL:-http://127.0.0.1:$SVS_PORT}"
fi
export SVS_PUBLIC_URL="${SVS_PUBLIC_URL:-http://127.0.0.1:$SVS_PORT}"
echo "=============================================="
echo "الواجهة     : $SVS_PUBLIC_URL/?token=$SVS_TOKEN"
echo "رابط MCP    : $SVS_PUBLIC_URL/mcp?token=$SVS_TOKEN"
echo "=============================================="
SVS_HOST="${SVS_HOST:-127.0.0.1}" exec $PY -m svs.mcp_server --http
