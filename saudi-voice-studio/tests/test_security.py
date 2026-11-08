"""Security regression tests (no model, no network)."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["SVS_HOME"] = str(Path(__file__).resolve().parent / "_tmp_home_sec")
os.environ["SVS_TOKEN"] = "t"
os.environ["SVS_VERIFY"] = "off"
os.environ["SVS_ASR"] = "off"
os.environ["SVS_SEMANTIC"] = "off"   # fast, offline unit tests (semantic checks live in test_semantic_slow.py)   # tests never load the speech recognizer
os.environ["SVS_SPEAKER"] = "off"

from svs import audio, config, mcp_server, voices  # noqa: E402


@pytest.mark.parametrize("bad", ["../x", "..", "a/b", "a\\b", "X", "", "a" * 41, "ab\x00", "/etc/passwd"])
def test_voice_id_traversal_rejected(bad):
    for fn in (voices.get_voice, voices.delete_voice):
        with pytest.raises(ValueError):
            fn(bad)


def test_delete_does_not_touch_outside_files(tmp_path):
    victim = config.USER_VOICES.parent / "victim.json"
    victim.write_text("{}")
    with pytest.raises(ValueError):
        voices.delete_voice("../victim")
    assert victim.exists()


@pytest.mark.parametrize("url", ["http://example.com/a.wav", "https://127.0.0.1/a", "https://169.254.169.254/latest",
                                 "https://localhost/a", "https://10.0.0.1/a", "https://[::1]/a", "file:///etc/passwd",
                                 "ftp://example.com/a"])
def test_ssrf_guard(url):
    with pytest.raises((ValueError, OSError)):
        mcp_server._public_https(url)


def test_base64_cap_and_validation(tmp_path):
    with pytest.raises(ValueError):
        mcp_server._fetch_audio("A" * (mcp_server.MAX_AUDIO * 2), "", str(tmp_path))
    with pytest.raises(ValueError):
        mcp_server._fetch_audio("not base64!!", "", str(tmp_path))


def test_reference_decoder_refuses_playlists_and_non_audio(tmp_path):
    """Defense in depth: HLS/concat playlists could make ffmpeg read other local files. The decoder only accepts
    allow-listed audio containers (checked with ffprobe BEFORE decoding) and whitelists the `file` protocol only."""
    from svs import refaudio
    secret = tmp_path / "secret.wav"
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=330:d=1", str(secret)], check=True)
    for name, body in {
        "hls": f"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nfile://{secret}\n#EXT-X-ENDLIST\n",
        "concat": f"ffconcat version 1.0\nfile '{secret}'\n",
        "text": "not audio at all",
    }.items():
        src = tmp_path / f"{name}.bin"
        src.write_text(body)
        with pytest.raises(refaudio.RefError):
            refaudio.decode(src)


def test_valid_wav_decodes_to_24k_mono(tmp_path):
    import subprocess
    from svs import refaudio
    src = tmp_path / "t.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=220:d=2", "-ac", "2", "-ar", "44100", str(src)], check=True)
    x, meta = refaudio.decode(src)
    assert abs(len(x) / 24000 - 2.0) < 0.1 and meta["src_channels"] == 2 and meta["src_sample_rate"] == 44100


def test_refuse_public_bind_without_token(monkeypatch):
    monkeypatch.setattr(config, "TOKEN", "")
    monkeypatch.setattr(config, "HOST", "0.0.0.0")
    with pytest.raises(SystemExit):
        config.require_safe_bind()
    monkeypatch.setattr(config, "HOST", "127.0.0.1")
    config.require_safe_bind()


def test_headers_auth_and_upload_cap(monkeypatch):
    from fastapi.testclient import TestClient
    from svs import web
    c = TestClient(web.app)
    r = c.get("/")
    assert r.headers["referrer-policy"] == "no-referrer" and "default-src 'none'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    h = {"x-svs-token": "t"}
    assert c.post("/api/generate", json={"text": "x"}).status_code == 401
    assert c.get("/files/abc.mp3").status_code == 401
    assert c.get("/files/..%2f..%2fetc%2fpasswd.mp3", headers=h).status_code in (404, 400)
    monkeypatch.setattr(web, "MAX_UPLOAD", 10)
    r = c.post("/api/voices", headers=h, data={"voice_id": "up_test", "name_ar": "x", "ref_text": "نص"},
               files={"ref": ("a.wav", b"x" * 100)})
    assert r.status_code == 400
    voices_left = [v["id"] for v in voices.list_voices()]
    if "up_test" in voices_left:
        voices.delete_voice("up_test")


def test_wrong_and_empty_token_rejected():
    from fastapi.testclient import TestClient
    from svs import web
    c = TestClient(web.app)
    assert c.get("/api/voices", headers={"x-svs-token": "wrong"}).status_code == 401
    assert c.get("/api/voices", headers={"x-svs-token": ""}).status_code == 401
    assert c.get("/api/voices?token=t").status_code == 200


def test_job_cap():
    from svs import jobs
    jobs.JOBS.clear()
    for i in range(jobs.MAX_RUNNING):
        jobs.JOBS[f"r{i}"] = {"state": "running"}
    assert jobs.running_count() >= jobs.MAX_RUNNING
    jobs.JOBS.clear()


def test_notebook_matplotlib_backend_is_forced_headless():
    """Regression: Colab exports MPLBACKEND=module://matplotlib_inline.backend_inline, invalid in our isolated venv."""
    import subprocess
    bad = {**os.environ, "MPLBACKEND": "module://matplotlib_inline.backend_inline"}
    r = subprocess.run([sys.executable, "-c",
                        "import sys; sys.path.insert(0, %r); import svs, os; print(os.environ['MPLBACKEND']);"
                        "\ntry:\n import matplotlib.pyplot as p, matplotlib; print(matplotlib.get_backend())\nexcept ImportError: print('agg')"
                        % str(Path(__file__).resolve().parent.parent)],
                       env=bad, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[0] == "Agg" and r.stdout.split()[-1].lower() == "agg"


def test_cell_overrides_inherited_backend():
    cell = (Path(__file__).resolve().parent.parent / "colab" / "run_studio.py").read_text(encoding="utf-8")
    assert 'env["MPLBACKEND"] = "Agg"' in cell


def test_v2_endpoints_require_token_and_validate():
    from fastapi.testclient import TestClient
    from svs import web
    c = TestClient(web.app)
    for method, url in [("post", "/api/analyze"), ("post", "/api/plan/recompute"), ("post", "/api/pronounce"),
                        ("get", "/api/lexicon"), ("put", "/api/lexicon"), ("get", "/api/takes"), ("get", "/api/meta"),
                        ("get", "/files/abcdefabcdef/master.mp3"), ("post", "/api/takes/abcdefabcdef/regenerate")]:
        assert getattr(c, method)(url).status_code == 401, url
    h = {"x-svs-token": "t"}
    assert c.post("/api/analyze", headers=h, json={"text": "احذر! خطر."}).status_code == 200
    assert c.post("/api/analyze", headers=h, json={"text": "x" * 25000}).status_code == 422
    assert c.post("/api/plan/recompute", headers=h, json={"segments": []}).status_code == 400
    assert c.put("/api/lexicon", headers=h, json={"entries": [{"word": "a", "say": "b"}] * 501}).status_code == 422
    assert c.get("/files/..%2f..%2fetc/passwd.mp3", headers=h).status_code in (400, 404)
    assert c.get("/files/abcdefabcdef/..%2fmaster.mp3", headers=h).status_code in (400, 404)
    assert c.get("/files/abcdefabcdef/take.json", headers=h).status_code == 404       # only audio names are served


def test_lexicon_entries_are_capped_and_cannot_inject_regex():
    from svs import pronounce
    n = pronounce.save_user_entries([{"word": "(a+)+$", "say": "x"}, {"word": "w" * 100, "say": "y"}, {"word": "ok", "say": "s" * 500}])
    assert n == 1
    pronounce.to_spoken("aaaaaaaaaaaaaaaaaaaaaaaa!")          # metacharacters in a word are escaped, so no ReDoS
    pronounce.save_user_entries([])
