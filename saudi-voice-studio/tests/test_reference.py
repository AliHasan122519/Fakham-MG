"""Reference-audio pipeline tests on REAL human recordings (the sample clips shipped inside the habibi-tts package)."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["SVS_HOME"] = str(Path(__file__).resolve().parent / "_tmp_home_ref")
os.environ["SVS_TOKEN"] = "t"
os.environ["SVS_VERIFY"] = "off"
os.environ["SVS_ASR"] = "off"
os.environ["SVS_SEMANTIC"] = "off"
os.environ["SVS_SPEAKER"] = "off"

from svs import asr, refaudio  # noqa: E402

try:
    from importlib.resources import files
    ASSETS = Path(str(files("habibi_tts").joinpath("assets")))
except Exception:           # pragma: no cover
    ASSETS = Path("/nonexistent")
pytestmark = pytest.mark.skipif(not (ASSETS / "Najdi.wav").is_file(), reason="habibi-tts sample clips not installed")
TEXT = {"Najdi.wav": "تكفى طمني انا اليوم ماني بنايم ولا هو بداخل عيني النوم الين اتطمن عليه.",
        "Hijazi.wav": "ابغاك تحقق معاه بس بشكل ودي لانه سلطان يمر بظروف صعبة شوية."}


def noisy(x, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(len(x))
    n *= np.sqrt(np.mean(x ** 2) / 10 ** (snr_db / 10)) / np.sqrt(np.mean(n ** 2))
    return (x + n).astype(np.float32)


def write(x, path):
    import soundfile as sf
    sf.write(str(path), x, refaudio.SR)
    return path


@pytest.mark.parametrize("name", ["Najdi.wav", "Hijazi.wav", "Gulf.wav", "UAE.wav", "MSA.mp3", "EGY.mp3"])
def test_real_clips_decode_and_pass(name):
    x, meta = refaudio.decode(ASSETS / name)
    a, b = refaudio.speech_bounds(x)
    rep = refaudio.analyse(x[a:b])
    v = refaudio.verdict(rep, (b - a) / refaudio.SR)
    assert v["status"] in ("ok", "warn"), v
    assert rep["voiced_ratio"] > 0.3 and rep["snr_db"] > 15 and 5 < rep["duration_s"] < 12


@pytest.mark.parametrize("fmt,args", [("m4a", ["-c:a", "aac", "-b:a", "64k"]), ("ogg", ["-c:a", "libvorbis"]),
                                      ("webm", ["-c:a", "libopus", "-b:a", "32k"]), ("flac", ["-c:a", "flac"])])
def test_phone_container_formats_decode(tmp_path, fmt, args):
    dst = tmp_path / f"c.{fmt}"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(ASSETS / "Najdi.wav"), *args, str(dst)], check=True)
    x, meta = refaudio.decode(dst)
    assert abs(len(x) / refaudio.SR - 5.5) < 0.4


@pytest.mark.parametrize("snr", [3, 5])
def test_noisy_reference_is_rejected_with_clear_reason(tmp_path, snr):
    """This is the failure that produced hiss: a 3-5 dB-SNR reference makes the model output babble."""
    x, _ = refaudio.decode(ASSETS / "Najdi.wav")
    res = refaudio.analyze_upload(write(noisy(x, snr), tmp_path / "n.wav"), TEXT["Najdi.wav"], use_asr=False)
    assert res["verdict"]["status"] == "reject"
    assert any("الضجيج" in m or "شرشرة" in m for m in res["verdict"]["reject"])


def test_moderately_noisy_reference_warns_but_is_accepted(tmp_path):
    x, _ = refaudio.decode(ASSETS / "Najdi.wav")
    res = refaudio.analyze_upload(write(noisy(x, 15), tmp_path / "n.wav"), TEXT["Najdi.wav"], use_asr=False)
    assert res["verdict"]["status"] in ("warn", "ok")


def test_silent_short_and_clipped_are_rejected(tmp_path):
    sil = write(np.zeros(refaudio.SR * 6, dtype=np.float32) + 1e-5, tmp_path / "s.wav")
    assert refaudio.analyze_upload(sil, "نص", use_asr=False)["verdict"]["status"] == "reject"
    x, _ = refaudio.decode(ASSETS / "Najdi.wav")
    short = write(x[: refaudio.SR * 2], tmp_path / "short.wav")
    assert refaudio.analyze_upload(short, "نص", use_asr=False)["verdict"]["status"] == "reject"
    clipped = write(np.clip(x * 12, -1, 1), tmp_path / "clip.wav")
    assert refaudio.analyze_upload(clipped, "نص", use_asr=False)["verdict"]["status"] == "reject"


def test_pure_hiss_is_rejected(tmp_path):
    rng = np.random.default_rng(3)
    res = refaudio.analyze_upload(write((0.1 * rng.standard_normal(refaudio.SR * 7)).astype(np.float32), tmp_path / "h.wav"),
                                  "نص", use_asr=False)
    assert res["verdict"]["status"] == "reject"


def test_corrupt_and_non_audio_files(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF" + os.urandom(2000))
    with pytest.raises(refaudio.RefError):
        refaudio.analyze_upload(bad, "نص", use_asr=False)


def test_preview_pipeline_saves_original_and_processed_and_commit_roundtrip(tmp_path):
    from svs import config, voices
    res = refaudio.analyze_upload(ASSETS / "Hijazi.wav", TEXT["Hijazi.wav"], use_asr=False)
    assert res["verdict"]["status"] in ("ok", "warn") and res["preview_id"]
    for n in ("original.wav", "processed.wav"):
        assert refaudio.preview_file(res["preview_id"], n).stat().st_size > 10000
    voices.create_voice("ref_unit", "اختبار", base_on="narrator_deep")
    refaudio.commit("ref_unit", res["preview_id"], res["transcript"])
    v = voices.get_voice("ref_unit")
    assert v["_ref_path"] and v["reference_source"] == "own"
    voices.delete_voice("ref_unit")


def test_denoise_guard_never_destroys_the_voice():
    """A denoised clip is accepted only if its speech spectrum stays correlated with the original (≥0.93) and the
    speech band is not attenuated by more than 3 dB; otherwise the original is kept."""
    x, _ = refaudio.decode(ASSETS / "Najdi.wav")
    for snr in (12, 20):
        xx = noisy(x, snr)
        a, b = refaudio.speech_bounds(xx)
        clip = xx[a:b]
        out, g = refaudio.guarded_clean(clip, refaudio.analyse(clip), None)
        if g["denoise_applied"]:
            assert g["ltas_corr"] >= 0.93 and g["speech_band_drop_db"] <= 3.0
        assert len(out) == len(clip) and np.abs(out).max() <= 1.0


def test_transcript_similarity_function():
    assert asr.similarity("مَرْحَبًا بِكُمْ", "مرحبا بكم") > 0.95
    assert asr.similarity("مرحبا بكم في استوديو الصوت", "السلام عليكم ورحمة الله وبركاته") < 0.5
    long_a = "في عالم تتسارع فيه التقنية وتكبر فيه الطموحات " * 8
    assert asr.similarity(long_a, long_a) == 1.0                   # autojunk bug regression: long strings must still match


def test_waqf_and_tashkeel_rules():
    from svs import tashkeel
    assert tashkeel.waqf("ذَهَبَ إِلَى الْمَدْرَسَةِ، وَاشْتَرَى كِتَابًا!") == "ذَهَبَ إِلَى الْمَدْرَسَهْ، وَاشْتَرَى كِتَابا!"
    assert tashkeel.waqf("قَرَأَ الدَّرْسَ.") == "قَرَأَ الدَّرْسْ."
    assert tashkeel.waqf("ذهب الولد إلى المدرسة.") == "ذهب الولد إلى المدرسة."      # undiacritised words untouched


def test_user_voice_does_not_inherit_identity_shift():
    from svs import voices
    v = voices.create_voice("ident_unit", "هوية", base_on="documentary_cinematic")
    assert "pitch_st" not in v["fx"] and v["fx"].get("bass_db") == 4      # mastering style kept, identity shift dropped
    voices.delete_voice("ident_unit")


# ---------------------------------------------------------------- web API (no model needed)
def test_reference_api_upload_preview_commit_and_guards():
    from fastapi.testclient import TestClient
    from svs import voices, web
    c = TestClient(web.app)
    H = {"x-svs-token": "t"}
    assert c.post("/api/reference/analyze", files={"ref": ("a.wav", b"x")}).status_code == 401
    data = (ASSETS / "Najdi.wav").read_bytes()
    r = c.post("/api/reference/analyze", headers=H, data={"ref_text": TEXT["Najdi.wav"]}, files={"ref": ("najdi.wav", data)})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["verdict"]["status"] in ("ok", "warn") and rep["preview_id"]
    pid = rep["preview_id"]
    for n in ("original.wav", "processed.wav"):
        assert c.get(f"/files/ref/{pid}/{n}", headers=H).status_code == 200
    assert c.get(f"/files/ref/{pid}/meta.json", headers=H).status_code == 404            # only audio names are served
    assert c.get(f"/files/ref/{pid}/..%2f..%2fmeta.json", headers=H).status_code in (400, 404)
    assert c.get("/files/ref/abc/original.wav", headers=H).status_code == 404
    # commit creates the voice and it becomes usable, with no inherited identity shift
    r = c.post("/api/voices/api_unit/reference/commit", headers=H,
               json={"preview_id": pid, "ref_text": rep["transcript"], "name_ar": "اختبار", "base_on": "documentary_cinematic"})
    assert r.status_code == 200, r.text
    assert r.json()["reference_source"] == "own" and "pitch_st" not in r.json()["fx"]
    assert c.post("/api/voices/../x/reference/commit", headers=H, json={"preview_id": pid, "ref_text": "x"}).status_code in (400, 404, 405)
    voices.delete_voice("api_unit")
    # a damaged upload gets a clear Arabic rejection, not a silent save
    r = c.post("/api/reference/analyze", headers=H, files={"ref": ("bad.wav", b"RIFF" + os.urandom(900))})
    assert r.status_code == 400 and "ملف" in r.json()["detail"]
    # uploads are size-capped
    big = c.post("/api/reference/analyze", headers=H, files={"ref": ("big.wav", b"0" * (26 * 1024 * 1024))})
    assert big.status_code == 400


def test_enhance_endpoint_returns_reviewable_plan_with_three_text_versions():
    from fastapi.testclient import TestClient
    from svs import web
    c = TestClient(web.app)
    H = {"x-svs-token": "t"}
    assert c.post("/api/enhance", json={"text": "x"}).status_code == 401
    r = c.post("/api/enhance", headers=H, json={"text": "اشتريت 5 كم من Google. ثم ذهب الطالب إلى المدرسة!", "voice": "narrator_deep"})
    assert r.status_code == 200, r.text
    p = r.json()
    assert [s["id"] for s in p["enhancement"]["steps"]] == ["analysis", "pronunciation", "diacritics", "emotions", "plan"]
    s0 = p["segments"][0]
    assert s0["text"] != s0["corrected"] and "قوقل" in s0["corrected"]                      # original vs corrected differ
    assert all("params" in s and s["vocalized"] is not None for s in p["segments"])          # diacritics step ran


def test_notebook_payload_contains_every_data_file_the_code_reads():
    """Regression: the first fresh-install run failed because voices/semantic_prototypes.json was missing from the
    notebook payload. Every data file referenced by the package must be shipped."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_nb", Path(__file__).resolve().parent.parent / "colab" / "build_nb.py")
    src = (Path(__file__).resolve().parent.parent / "colab" / "build_nb.py").read_text(encoding="utf-8")
    root = Path(__file__).resolve().parent.parent
    included = eval(src.split("INCLUDE = ", 1)[1].split("\n", 1)[0])
    pkg = "\n".join(p.read_text(encoding="utf-8") for p in (root / "svs").glob("*.py"))
    import re
    referenced = set(re.findall(r'"voices"\s*/\s*"([\w.]+\.json)"', pkg)) | set(re.findall(r'config\.ROOT / "voices" / "([\w.]+)"', pkg))
    referenced |= {"presets.json", "lexicon_default.json", "semantic_prototypes.json"}
    for name in referenced:
        assert f"voices/{name}" in included, f"voices/{name} is not shipped in the notebook payload"
    assert "web/index.html" in included and "scripts/download_models.py" in included


# ---------------------------------------------------------------- one-button automatic flow
def test_default_voice_follows_first_own_reference_and_web_auto_review():
    from fastapi.testclient import TestClient
    from svs import config, voices, web
    (config.DATA / "default_voice.txt").unlink(missing_ok=True)
    assert voices.default_voice_id() == "narrator_deep"                  # nothing of yours yet → built-in persona
    res = refaudio.analyze_upload(ASSETS / "Najdi.wav", TEXT["Najdi.wav"], use_asr=False)
    voices.create_voice("my_voice", "صوتي")
    refaudio.commit("my_voice", res["preview_id"], res["transcript"])
    assert voices.default_voice_id() == "my_voice"                       # becomes the default automatically
    c = TestClient(web.app)
    H = {"x-svs-token": "t"}
    assert c.post("/api/auto", json={"text": "x"}).status_code == 401
    r = c.post("/api/auto", headers=H, json={"text": "احذر! الطريق خطير ممنوع الاقتراب.", "review": True})
    assert r.status_code == 200 and r.json()["voice"] == "my_voice" and r.json()["segments"]   # plan for review, in YOUR voice
    assert c.post("/api/auto", headers=H, json={"text": ""}).status_code == 422
    assert c.post("/api/auto", headers=H, json={"text": "نص", "voice": "nope_voice"}).status_code == 400
    assert c.get("/api/voices", headers=H).json()["voices"][0] is not None
    assert any(v["is_default"] for v in c.get("/api/voices", headers=H).json()["voices"])
    voices.delete_voice("my_voice")
    assert voices.default_voice_id() == "narrator_deep"                  # deleted → falls back safely


def test_auto_run_uses_default_voice_and_best_of_n_is_wired(monkeypatch):
    from svs import auto, render, voices
    seen = {}
    def fake_render(plan, **kw):
        seen["voice"], seen["cand"] = plan["voice"], kw.get("candidates")
        return {"ok": True}
    monkeypatch.setattr(render, "render", fake_render)
    auto.run("احذر! الطريق خطير.", max_quality=True)
    assert seen["voice"] == voices.default_voice_id() and seen["cand"] >= 3


def test_own_voice_locks_identity_no_pitch_shift():
    """Measured with ECAPA speaker similarity: a -0.5 semitone shift cost 0.07. Voices built from YOUR reference are
    never pitch-shifted; personas with demo references keep their expressive pitch."""
    from svs import plan as planmod, voices
    res = refaudio.analyze_upload(ASSETS / "Najdi.wav", TEXT["Najdi.wav"], use_asr=False)
    voices.create_voice("lock_unit", "قفل")
    refaudio.commit("lock_unit", res["preview_id"], res["transcript"])
    own = planmod.build("احذر! الطريق خطير. ثم ساد الهدوء وعاد السلام إلى المكان.", voice="lock_unit")
    assert own["lock_identity"] and all(s["params"]["pitch_st"] == 0 for s in own["segments"])
    persona = planmod.build("احذر! الطريق خطير. ثم ساد الهدوء وعاد السلام إلى المكان.", voice="narrator_deep")
    assert not persona["lock_identity"] and any(s["params"]["pitch_st"] != 0 for s in persona["segments"])
    again = planmod.validate(own)                                          # survives the edit→recompute round trip
    assert again["lock_identity"] and all(s["params"]["pitch_st"] == 0 for s in again["segments"])
    voices.delete_voice("lock_unit")


def test_gated_diacritics_prefers_plain_when_clearer(monkeypatch):
    """When the vocalised sentence is clearly less intelligible than the plain one, the plain text is used."""
    from svs import plan as planmod, render, voices
    monkeypatch.setattr(render, "_verify_on", lambda: True)
    sims = {}
    def fake_intel(wave, spoken):
        return 0.5 if "َ" in spoken else 0.97
    monkeypatch.setattr(render, "_intelligibility", fake_intel)
    import numpy as np
    def synth(spoken, ref, ref_text, *, engine, speed, nfe, seed):
        return (0.2 * np.sin(2 * np.pi * 150 * np.arange(24000 * 2) / 24000) * (1 + 0.1 * len(spoken) % 3)).astype("float32"), 24000
    p = planmod.build("ذهب القوم إلى السوق في الصباح الباكر.", voice="narrator_deep", diacritics="none")
    s0 = p["segments"][0]
    s0["corrected"], s0["vocalized"] = "ذهب القوم إلى السوق في الصباح الباكر.", "ذَهَبَ القوم إلى السوق في الصباح الباكر."
    s0["spoken"] = s0["vocalized"]
    take = render.render(p, synth_fn=synth)
    assert take["segments"][0]["diacritics"] == "dropped" and take["intelligibility"]["s1"] == 0.97


def test_narrators_derive_from_own_voice_and_no_recursion():
    """Regression: list_voices → own_base → default_voice_id → list_voices used to recurse forever (hang)."""
    import signal
    from svs import voices
    res = refaudio.analyze_upload(ASSETS / "Najdi.wav", TEXT["Najdi.wav"], use_asr=False)
    voices.create_voice("my_voice", "صوتي")
    refaudio.commit("my_voice", res["preview_id"], res["transcript"])
    signal.alarm(20)                                       # a hang would kill the test instead of freezing the suite
    try:
        lv = {v["id"]: v for v in voices.list_voices()}
    finally:
        signal.alarm(0)
    assert lv["my_voice"]["reference_source"] == "own" and lv["my_voice"]["gender"] == "male"
    for pid in ("narrator_deep", "promo_energetic", "anchor_formal", "documentary_cinematic"):
        assert lv[pid]["reference_source"] == "derived"
        assert voices.reference_for(voices.get_voice(pid), "najdi")[2] == "derived"
    assert lv["narrator_female"]["reference_source"] == "demo"          # a male base is never used for the female narrator
    voices.delete_voice("my_voice")
    assert {v["reference_source"] for v in voices.list_voices() if v["builtin"]} == {"demo"}


def test_sample_endpoint_auth_cache_and_name_guards():
    from fastapi.testclient import TestClient
    from svs import web
    c = TestClient(web.app)
    H = {"x-svs-token": "t"}
    assert c.post("/api/voices/narrator_deep/sample").status_code == 401
    assert c.post("/api/voices/..%2fx/sample", headers=H).status_code in (400, 404)
    assert c.post("/api/voices/nope_voice/sample", headers=H).status_code == 400
    assert c.get("/files/sample/narrator_deep.mp3", headers=H).status_code == 404       # nothing generated yet
    assert c.get("/files/sample/../x.mp3", headers=H).status_code in (400, 404)
    assert c.get("/files/sample/narrator_deep.mp3").status_code == 401


def test_auto_voice_picks_narrator_by_genre(monkeypatch):
    from svs import auto, enhance, render
    seen = {}
    monkeypatch.setattr(render, "render", lambda plan, **kw: seen.setdefault("voice", plan["voice"]) and {"ok": 1})
    auto.run("لا تفوت العرض الحصري، احجز الآن واحصل على خصم خاص!", voice="auto")
    assert seen["voice"] in ("promo_energetic", "narrator_deep")        # ad → promo narrator when available, else default


def test_api_auto_accepts_auto_narrator():
    from fastapi.testclient import TestClient
    from svs import jobs, web
    c = TestClient(web.app)
    H = {"x-svs-token": "t"}
    started = {}
    import svs.auto as auto_mod
    orig = jobs.start
    jobs.start = lambda fn=None, **kw: started.setdefault("kw", kw) and "job123"
    try:
        r = c.post("/api/auto", headers=H, json={"text": "لا تفوت العرض الحصري!", "voice": "auto"})
    finally:
        jobs.start = orig
    assert r.status_code == 200 and started["kw"]["voice"] == "auto"
    assert c.post("/api/auto", headers=H, json={"text": "نص", "voice": "auto", "review": True}).status_code == 200
