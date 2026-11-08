"""Security regression tests (no model, no network)."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["SVS_HOME"] = str(Path(__file__).resolve().parent / "_tmp_home_sec")
os.environ["SVS_TOKEN"] = "t"

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


def test_ffmpeg_playlist_cannot_read_local_files(tmp_path):
    """Defense in depth: with the ffmpeg build tested the nameless-file HLS trick was not exploitable anyway,
    but other builds/formats may differ, so references are fed via stdin with a pipe-only protocol whitelist."""
    secret = tmp_path / "secret.wav"
    secret.write_bytes(b"RIFF....WAVE")
    for name, body in {
        "hls": f"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nfile://{secret}\n#EXT-X-ENDLIST\n",
        "concat": f"ffconcat version 1.0\nfile '{secret}'\n",
    }.items():
        src = tmp_path / f"{name}.bin"
        src.write_text(body)
        with pytest.raises(ValueError):
            audio.prepare_reference(src, tmp_path / f"{name}.wav")


def test_valid_wav_still_converts(tmp_path):
    import subprocess
    src = tmp_path / "t.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=220:d=1", str(src)], check=True)
    out = audio.prepare_reference(src, tmp_path / "o.wav")
    assert out.is_file() and audio.duration(out) > 0.5


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
