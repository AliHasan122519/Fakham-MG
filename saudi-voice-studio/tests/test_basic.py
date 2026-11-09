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
os.environ["SVS_SPEAKER"] = "off"

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


def test_onednn_cache_is_capped_before_torch_import():
    """Regression for the measured bf16 memory growth: the cap must be set by the package itself, in a fresh process, before torch loads."""
    import subprocess, sys
    code = ("import os,sys; os.environ.pop('ONEDNN_PRIMITIVE_CACHE_CAPACITY',None); import svs; "
            "assert os.environ['ONEDNN_PRIMITIVE_CACHE_CAPACITY']=='8'; assert 'torch' not in sys.modules; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(__import__('pathlib').Path(__file__).resolve().parents[1]))
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_history_does_not_leak_file_handles():
    import os
    from svs import history
    def n_fds():
        return len(os.listdir("/proc/self/fd"))
    history.add({"id": "h0", "voice": "v", "dialect": "d", "emotion": "e", "speed": 1.0, "text": "t", "seconds": 1.0, "took": 1.0,
                 "device": "cpu", "engine": "najdi"})
    before = n_fds()
    for i in range(60):
        history.add({"id": f"h{i}", "voice": "v", "dialect": "d", "emotion": "e", "speed": 1.0, "text": "t", "seconds": 1.0, "took": 1.0,
                     "device": "cpu", "engine": "najdi"})
        history.recent(5)
        history.get(f"h{i}")
    assert n_fds() - before <= 1, (before, n_fds())


def test_memory_hygiene_is_safe_and_applies_on_linux():
    from svs import memory
    memory.release()                                   # must never raise
    if __import__("sys").platform.startswith("linux"):
        assert memory.tune() is True                   # glibc: arena cap accepted


def test_notebook_payload_is_current_with_the_working_tree():
    """The Colab notebook embeds the code. A stale payload once shipped WITHOUT the memory fix; every embedded file must equal the repo file."""
    import base64, io, json, re, tarfile
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "colab" / "saudi_voice_studio.ipynb").read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    payload = re.search(r'PAYLOAD\s*=\s*"([A-Za-z0-9+/=]+)"', src).group(1)
    t = tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode="r:gz")
    stale = [n for n in t.getnames() if t.getmember(n).isfile() and (root / n).exists() and t.extractfile(n).read() != (root / n).read_bytes()]
    assert not stale, f"rebuild the notebook (python colab/build_nb.py): {stale}"
    assert "svs/memory.py" in t.getnames()
