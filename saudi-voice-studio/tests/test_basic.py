"""Fast tests (no model download, no GPU)."""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["SVS_HOME"] = str(Path(__file__).resolve().parent / "_tmp_home")
os.environ["SVS_TOKEN"] = "t"
os.environ["SVS_VERIFY"] = "off"
os.environ["SVS_ASR"] = "off"
os.environ["SVS_SEMANTIC"] = "off"   # fast, offline unit tests (semantic checks live in test_semantic_slow.py)   # tests never load the speech recognizer

from svs import text_ar  # noqa: E402


def test_numbers():
    assert text_ar.number_to_words(0) == "صفر"
    assert text_ar.number_to_words(15) == "خمسة عشر"
    assert text_ar.number_to_words(2025) == "ألفان وخمسة وعشرون"
    assert "مليون" in text_ar.number_to_words(1234567)


def test_normalize():
    out = text_ar.normalize("في عام ٢٠٢٥ نما 12.5% تطويـــل")
    assert "٢" not in out and "%" not in out and "ـ" not in out
    assert "بالمائة" in out


def test_chunks():
    chunks = text_ar.split_chunks("جملة قصيرة. " * 60, 100)
    assert all(len(c) <= 110 for c in chunks) and len(chunks) > 3


def test_voices_and_dialects():
    from svs import voices
    ids = {v["id"] for v in voices.list_voices()}
    assert {"narrator_deep", "promo_energetic", "anchor_formal", "narrator_female", "documentary_cinematic"} <= ids
    assert {"najdi", "hijazi", "white", "khaleeji", "msa"} == set(voices.dialects())
    assert {"neutral", "joy", "sadness", "excitement", "calm", "confidence", "grandeur", "empathy", "suspense", "warning", "formal"} <= set(voices.emotions())


def test_create_delete_voice():
    from svs import voices
    v = voices.create_voice("test_voice", "تجربة", base_on="narrator_deep")
    base = {k: x for k, x in voices.get_voice("narrator_deep")["fx"].items() if k != "pitch_st"}
    assert v["id"] == "test_voice" and v["fx"] == base      # identity shift is never inherited
    voices.delete_voice("test_voice")
    try:
        voices.get_voice("test_voice")
        assert False
    except KeyError:
        pass


def test_filter_builds():
    from svs import audio
    af = audio.build_filter({"pitch_st": -1.5, "bass_db": 4, "reverb": 0.3, "compress": True})
    assert "asetrate" in af and "aecho" in af and af.endswith("loudnorm=I=-16:TP=-1.5:LRA=11")


def test_web_auth_and_voices():
    from fastapi.testclient import TestClient
    from svs.web import app
    c = TestClient(app)
    assert c.get("/api/voices").status_code == 401
    r = c.get("/api/voices", headers={"x-svs-token": "t"})
    assert r.status_code == 200 and len(r.json()["voices"]) >= 5
    assert c.get("/files/../../etc.passwd", headers={"x-svs-token": "t"}).status_code in (404, 400)


def test_mcp_tools_registered():
    import asyncio
    from svs import mcp_server
    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert {"generate_voiceover", "list_voices", "create_voice", "clone_voice", "set_voice_reference",
            "get_download_links", "list_history", "engine_status", "get_job"} <= names


def test_clone_requires_rights():
    from svs import mcp_server
    r = mcp_server.clone_voice("x1", "س", "نص", False, audio_base64="AAAA")
    assert "error" in r
