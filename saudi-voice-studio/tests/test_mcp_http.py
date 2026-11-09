"""Real MCP protocol over HTTP (streamable HTTP, JSON-RPC) against the production ASGI app — not just calling tool functions."""
import json
import os

os.environ.setdefault("SVS_TOKEN", "t")

from starlette.testclient import TestClient

from svs import mcp_server

HDR = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def _rpc(c, path, method, params=None, id_=1, token="t"):
    body = {"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}}
    r = c.post(f"{path}?token={token}", headers=HDR, content=json.dumps(body))
    return r


def _parse(r):
    txt = r.text.strip()
    if r.headers["content-type"].startswith("text/event-stream"):
        txt = [l[5:].strip() for l in txt.splitlines() if l.startswith("data:")][-1]
    return json.loads(txt)


def test_mcp_initialize_list_and_call_over_http_with_and_without_trailing_slash():
    app = mcp_server.build_http_app()
    with TestClient(app, base_url="http://127.0.0.1") as c:
        for path in ("/mcp", "/mcp/"):                                  # both URL spellings must work
            r = _rpc(c, path, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                              "clientInfo": {"name": "pytest", "version": "1"}})
            assert r.status_code == 200, (path, r.status_code, r.text[:200])
            assert "tools" in _parse(r)["result"]["capabilities"]
            r = _rpc(c, path, "tools/list", id_=2)                        # stateless: works without any session header
            assert r.status_code == 200, r.text[:200]
            names = {t["name"] for t in _parse(r)["result"]["tools"]}
            assert {"generate_voiceover", "smart_enhance", "analyze_text", "regenerate_sentence", "get_job", "engine_status",
                    "list_voices"} <= names, names
        r = _rpc(c, "/mcp", "tools/call", {"name": "analyze_text", "arguments": {"text": "ذهب القوم إلى السوق."}}, id_=3)
        res = _parse(r)["result"]
        assert not res.get("isError"), res
        r = _rpc(c, "/mcp", "tools/call", {"name": "engine_status", "arguments": {}}, id_=4)
        assert not _parse(r)["result"].get("isError")


def test_mcp_requires_token_and_blocks_foreign_host():
    app = mcp_server.build_http_app()
    with TestClient(app, base_url="http://127.0.0.1") as c:
        assert _rpc(c, "/mcp", "tools/list", token="wrong").status_code == 401
        assert c.post("/mcp", headers=HDR, content="{}").status_code == 401          # no token at all
        r = c.post("/mcp?token=t", headers={**HDR, "Host": "evil.example.com"},
                   content=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}))
        assert r.status_code in (400, 403, 421)                                       # DNS-rebinding protection
