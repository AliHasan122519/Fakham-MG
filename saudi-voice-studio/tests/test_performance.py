"""Performance-engine tests (no model: a deterministic fake synthesiser stands in for the TTS)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["SVS_HOME"] = str(Path(__file__).resolve().parent / "_tmp_home_perf")
os.environ["SVS_TOKEN"] = "t"
os.environ["SVS_VERIFY"] = "off"
os.environ["SVS_ASR"] = "off"
os.environ["SVS_SEMANTIC"] = "off"   # fast, offline unit tests (semantic checks live in test_semantic_slow.py)   # tests never load the speech recognizer

from svs import analysis, audio, delivery, plan as planmod, pronounce, render  # noqa: E402

TEXT = ("في قلب الصحراء، حيث يلتقي الرمل بالسماء، تُروى حكايات الأجداد عبر القرون.\n\n"
        "لكن فجأة، خيّم الصمت على المكان... احذر! الطريق خطير ممنوع الاقتراب.\n\n"
        "مبروك لكم هذا النجاح الرائع. نشعر بك ولست وحدك. لا تفوت العرض الحصري، احجز الآن!")


def fake_synth(spoken, ref, ref_text, *, engine, speed, nfe, seed):
    sr = 24000
    n = int(sr * max(len(spoken), 8) * 0.07 / speed)
    t = np.arange(n) / sr
    f0 = 120 + 15 * np.sin(2 * np.pi * 0.8 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    w = sum(np.sin(h * ph) / h for h in range(1, 10))
    env = np.sin(np.pi * np.arange(n) / n) ** 0.5 * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t) ** 2)
    rng = np.random.default_rng(seed)
    return (0.25 * w * env + 0.001 * rng.standard_normal(n)).astype(np.float32), sr


# ---------------------------------------------------------------- analysis
def test_sentence_and_paragraph_split():
    s = analysis.split_paragraphs_sentences(TEXT)
    assert len(s) >= 7
    assert s[0]["first_in_paragraph"] and s[0]["paragraph"] == 0 and s[-1]["paragraph"] == 2
    assert any(x["last_in_paragraph"] for x in s)


@pytest.mark.parametrize("sentence,expected", [
    ("احذر! الطريق خطير ممنوع الاقتراب.", "warning"),
    ("مبروك لكم هذا النجاح الرائع.", "joy"),
    ("نشعر بك ولست وحدك ونحن بجانبك.", "empathy"),
    ("لا تفوت العرض الحصري، احجز الآن!", "excitement"),
    ("أعلنت وزارة الصحة اليوم عن بدء الحملة.", "formal"),
    ("لكن فجأة، خيّم الصمت على المكان...", "suspense"),
    ("في قلب الصحراء تُروى حكايات الأجداد عبر القرون.", "grandeur"),
])
def test_emotion_detection(sentence, expected):
    assert analysis.detect_emotion(sentence)["emotion"] == expected


def test_neutral_when_no_cues():
    assert analysis.detect_emotion("ذهب الطالب إلى المدرسة صباحا.")["emotion"] == "neutral"


def test_genre_detection():
    assert analysis.detect_genre("لا تفوت العرض الحصري احجز الآن واحصل على خصم")[0] == "ad"
    assert analysis.detect_genre("أعلنت وزارة الصحة اليوم صرح المتحدث الرسمي")[0] == "news"


# ---------------------------------------------------------------- pronunciation
def test_numbers_units_dictionary():
    r = pronounce.to_spoken("وصلنا بعد 5 كم و AI من Google بسعر 250 ريال")
    sp = r["spoken"]
    assert "خمسة كيلومتر" in sp and "إيه آي" in sp and "قوقل" in sp and "مائتان وخمسون" in sp
    assert not any(ch.isdigit() for ch in sp)


def test_unknown_acronym_is_spelled_and_unknown_latin_flagged():
    r = pronounce.to_spoken("شركة XYZ و Foobar")
    assert "إكس واي زد" in r["spoken"] and "Foobar" in r["flags"]["unknown_latin"]


def test_ambiguous_words_flagged_and_user_lexicon_wins():
    r = pronounce.to_spoken("هذا علم وذهب")
    assert {a["stem"] for a in r["flags"]["ambiguous"]} >= {"علم", "ذهب"}
    pronounce.save_user_entries([{"word": "نيوم", "say": "نِيُوم"}])
    assert "نِيُوم" in pronounce.to_spoken("مدينة نيوم")["spoken"]
    pronounce.save_user_entries([])


def test_light_dialect_is_opt_in():
    assert "الحين" not in pronounce.to_spoken("أريد ذلك الآن", "najdi")["spoken"]
    assert "الحين" in pronounce.to_spoken("أريد ذلك الآن", "najdi", light_dialect=True)["spoken"]


# ---------------------------------------------------------------- delivery / plan
def test_plan_has_params_and_varies_across_sentences():
    p = planmod.build(TEXT, voice="narrator_deep")
    sp = [s["params"]["speed_mul"] for s in p["segments"]]
    pa = [s["params"]["pause_after"] for s in p["segments"]]
    assert len(set(sp)) > 3 and len(set(pa)) > 3           # not a monotone, identical cadence
    assert max(sp) / min(sp) > 1.08
    # last sentence closes slower than the average; paragraph ends breathe longer than mid-paragraph
    assert p["segments"][-1]["params"]["pause_after"] >= 0.5
    ends = [s["params"]["pause_after"] for s in p["segments"] if s["last_in_paragraph"]]
    mids = [s["params"]["pause_after"] for s in p["segments"] if not s["last_in_paragraph"]]
    assert np.mean(ends) > np.mean(mids)


def test_variation_zero_removes_jitter_and_expressiveness_scales():
    c0 = {"variation": 0.0, "expressiveness": 0.0}
    a = planmod.build("احذر! الطريق خطير.", voice="narrator_deep", controls=c0)["segments"][0]["params"]
    b = planmod.build("احذر! الطريق خطير.", voice="narrator_deep", controls={"variation": 0.0, "expressiveness": 1.5})["segments"][0]["params"]
    assert abs(b["gain_db"]) > abs(a["gain_db"]) and a["gain_db"] < 1.0


def test_style_is_relative_to_persona_no_double_slowdown():
    p = planmod.build("جملة عادية تمامًا.", voice="documentary_cinematic", style="documentary",
                      controls={"variation": 0, "formality": 0})
    assert p["segments"][0]["params"]["speed_mul"] > 0.9


def test_manual_mode_uses_one_emotion_but_keeps_suggestions():
    p = planmod.build(TEXT, voice="narrator_deep", mode="manual", emotion="calm")
    assert {s["emotion"] for s in p["segments"]} == {"calm"}
    assert any(s["detected"]["emotion"] != "calm" for s in p["segments"])


def test_overrides_win_and_are_clamped():
    p = planmod.build("احذر!", voice="narrator_deep")
    p["segments"][0]["overrides"] = {"speed_mul": 9, "pause_after": 1.5, "pitch_st": -50}
    v = planmod.validate(p)["segments"][0]["params"]
    assert v["speed_mul"] == delivery.OVERRIDE_LIMITS["speed_mul"][1] and v["pause_after"] == 1.5 and v["pitch_st"] == -2.0


def test_validate_rejects_bad_plans():
    for bad in [None, {}, {"segments": []}, {"segments": [{"spoken": ""}], "voice": "narrator_deep", "dialect": "najdi"},
                {"segments": [{"spoken": "x"}], "voice": "../x", "dialect": "najdi"}]:
        with pytest.raises((ValueError, KeyError, TypeError)):
            planmod.validate(bad)


def test_text_limits():
    with pytest.raises(ValueError):
        planmod.build("ا" * (planmod.MAX_CHARS + 1))
    with pytest.raises(ValueError):
        planmod.build("   ")


# ---------------------------------------------------------------- render (fake synth)
@pytest.fixture(scope="module")
def take():
    p = planmod.build(TEXT, voice="documentary_cinematic")
    return render.render(p, synth_fn=fake_synth)


def test_render_outputs_and_metrics(take):
    d = render.TAKES / take["id"]
    for name in ("master.mp3", "master.wav", "raw.mp3"):
        assert (d / name).is_file() and (d / name).stat().st_size > 1000
    assert all((d / f"seg_{s}.mp3").is_file() for s in take["files"]["segments"])
    a, b = take["metrics"]["after"], take["metrics"]["before"]
    assert -18 < a["lufs"] < -14 and a["true_peak_db"] <= -1.0


def test_mastering_measurably_reduces_noise_floor():
    sr = 24000
    t = np.arange(sr * 3) / sr
    rng = np.random.default_rng(1)
    voice = 0.2 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 2 * t) > 0)
    noisy = (voice + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
    clean = audio.ffmpeg_wave(noisy, audio.master_chain({}, audio.spectral_profile(noisy)))
    lead = slice(int(0.5 * sr), int(0.9 * sr))     # a gap where the voice is silent
    gap = lambda x: 20 * np.log10(np.sqrt((x[lead] ** 2).mean()) + 1e-9)
    assert gap(clean[: len(noisy)]) < gap(noisy * 1.0) + 6   # normalised level considered; noise must not be amplified


def test_regenerate_one_sentence_reuses_cache(take):
    n = len(take["plan"]["segments"])
    new = render.regenerate(take["id"], take["plan"]["segments"][1]["id"], synth_fn=fake_synth)
    assert new["parent"] == take["id"] and new["cache_hits"] == n - 1
    assert new["plan"]["segments"][1]["seed"] != take["plan"]["segments"][1]["seed"]


def test_pitch_chain_preserves_duration():
    x = fake_synth("اختبار النبرة والطول", None, None, engine="najdi", speed=1.0, nfe=16, seed=1)[0]
    for chain in (audio.pitch_chain(-2.5, 0), audio.pitch_chain(0, 0.8)):
        y = audio.ffmpeg_wave(x, chain)
        assert abs(len(y) / len(x) - 1) < 0.03


def test_take_file_guard(take):
    with pytest.raises(render.RenderError):
        render.take_file(take["id"], "../take.json")
    with pytest.raises(render.RenderError):
        render.take_file("../../etc", "master.mp3")
    assert render.take_file(take["id"], "master.mp3").is_file()


def test_persona_identities_are_distinct_in_config():
    from svs import voices
    ids = {v["id"]: v for v in voices.list_voices()}
    keys = {(v["ref_id"], round(v["fx"].get("pitch_st", 0), 1), v["speed"], v["style"]) for v in ids.values() if v["builtin"]}
    assert len(keys) == 5          # five different (reference, identity shift, base speed, style) combinations


def test_expressive_engine_disabled_by_default_and_mapping():
    from svs import expressive, voices
    assert not expressive.available() and voices.dialects()["msa"]["available"] is False
    lo = expressive.exaggeration_for("calm", 0.3, 1.0)[0]
    hi = expressive.exaggeration_for("excitement", 1.0, 1.5)[0]
    assert 0.25 <= lo < hi <= 1.0 and expressive.exaggeration_for("excitement", 1.0, 1.5)[1] == 0.3
    p = planmod.build("جملة تجريبية.", voice="narrator_deep", dialect="msa")
    with pytest.raises(render.RenderError):
        render.render(p, synth_fn=fake_synth)
