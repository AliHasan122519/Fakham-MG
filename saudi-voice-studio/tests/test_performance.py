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
os.environ["SVS_SPEAKER"] = "off"

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
    assert p["segments"][0]["params"]["speed_mul"] > 0.85      # was 0.69 when persona and style speeds stacked


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


def test_reference_eq_moves_spectrum_toward_reference_and_is_clamped():
    rng = np.random.default_rng(0)
    sr = 24000
    t = np.arange(sr * 3) / sr
    base = sum(np.sin(2 * np.pi * 150 * h * t) / h for h in range(1, 40)) * (np.sin(2 * np.pi * 3 * t) ** 2)
    thin = audio.ffmpeg_wave((0.2 * base).astype(np.float32), "highpass=f=400")           # no body below 400 Hz
    full = (0.2 * base).astype(np.float32)
    eq = audio.reference_eq(thin, full)
    gains = {int(f.split("f=")[1].split(":")[0]): float(f.split("g=")[1]) for f in eq}
    assert gains.get(125, 0) > 0 and gains[125] <= 4.0 and all(abs(g) <= 4.0 for g in gains.values())
    before = np.abs(audio._band_levels(thin, sr) - audio._band_levels(full, sr)).mean()
    fixed = audio.ffmpeg_wave(thin, ",".join(eq))
    after = np.abs(audio._band_levels(fixed, sr) - audio._band_levels(full, sr)).mean()
    assert after < before


# ---------------------------------------------------------------- v1.0: dry sound, pro controls, QC, pace harmonisation, export
def test_default_sound_is_dry_no_reverb_anywhere():
    import json
    from svs import config
    d = json.loads((config.ROOT / "voices" / "presets.json").read_text(encoding="utf-8"))
    assert all("reverb" not in v["fx"] for v in d["voices"])
    assert "aecho" not in audio.master_chain({}, {"hf4": 0.03, "hf8": 0.001})
    assert "aecho" in audio.master_chain({}, {"hf4": 0.03, "hf8": 0.001}, reverb=0.3)       # only on explicit request
    p = planmod.build("نص تجريبي.", voice="documentary_cinematic")
    assert p["controls"]["reverb"] == 0.0 and p["controls"]["denoise"] is False


def test_style_auto_defaults_and_user_override():
    a = planmod.build("نص تجريبي.", voice="narrator_deep", style="news")["controls"]
    assert a["stability"] == 0.9 and a["clarity"] == 0.8                                      # smart defaults for news
    b = planmod.build("نص تجريبي.", voice="narrator_deep", style="news", controls={"stability": 0.2})["controls"]
    assert b["stability"] == 0.2 and b["clarity"] == 0.8                                      # explicit value wins, others stay smart


def test_pro_controls_change_the_result():
    base = planmod.build("احذر! الطريق خطير ممنوع الاقتراب.", voice="narrator_deep", controls={"variation": 0, "energy": 0, "clarity": 0.5})
    hot = planmod.build("احذر! الطريق خطير ممنوع الاقتراب.", voice="narrator_deep", controls={"variation": 0, "energy": 1.0, "clarity": 0.5})
    calm = planmod.build("احذر! الطريق خطير ممنوع الاقتراب.", voice="narrator_deep", controls={"variation": 0, "energy": -1.0, "clarity": 0.5})
    pb, ph, pc = (x["segments"][0]["params"] for x in (base, hot, calm))
    assert ph["speed_mul"] > pb["speed_mul"] > pc["speed_mul"] and ph["gain_db"] > pb["gain_db"] > pc["gain_db"]
    assert ph["pitch_st"] == pb["pitch_st"] == pc["pitch_st"]                                  # energy never touches pitch
    clear = planmod.build("جملة.", voice="narrator_deep", controls={"clarity": 1.0, "variation": 0})["segments"][0]["params"]["speed_mul"]
    mush = planmod.build("جملة.", voice="narrator_deep", controls={"clarity": 0.0, "variation": 0})["segments"][0]["params"]["speed_mul"]
    assert clear < mush
    assert "bass=g=2.0" in audio.master_chain({}, {}, warmth_db=2.0) and "bass=g" not in audio.master_chain({}, {}, warmth_db=0.0)
    assert "equalizer=f=3200:t=q:w=1.0:g=2.0" in audio.master_chain({}, {}, clarity=1.0)


def test_qc_helpers():
    from svs import qc
    assert qc.word_diff("ذهب القوم إلى السوق", "ذهب القوم إلى السوق") == {"missing": 0.0, "dup": 0}
    assert qc.word_diff("ذهب القوم إلى السوق", "ذهب السوق")["missing"] >= 0.4
    assert qc.word_diff("ذهب القوم إلى السوق", "ذهب القوم القوم إلى السوق")["dup"] == 1
    assert qc.word_diff("أنا أحبك حبا جما", "انا احبك حبا جما")["missing"] == 0.0         # spelling variants are not "missing"
    sr = 24000
    t = np.arange(sr * 2) / sr
    voice = (0.3 * np.sin(2 * np.pi * 150 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0)).astype(np.float32)
    dry = voice
    rng = np.random.default_rng(1)
    ir = (rng.standard_normal(int(sr * 0.5)) * np.exp(-np.arange(int(sr * 0.5)) / sr / 0.12)).astype(np.float32)
    ir[0] = 120.0                                                                                     # direct sound + room tail
    wet = np.convolve(voice, ir)[: len(voice)].astype(np.float32)
    rd, rw = qc.reverb_tail_db(dry), qc.reverb_tail_db(wet)
    assert rd is not None and rw is not None and rw > rd + 20 and rw > -35, (rd, rw)               # a room tail is detected


def test_take_has_qc_report_pace_harmonisation_and_24bit_wav():
    import soundfile as sf
    text = ("وصلنا إلى المدينة في الصباح الباكر. ثم ذهبنا إلى السوق القديم لنشتري بعض الهدايا لأهلنا وأصدقائنا. "
            "وفي المساء عدنا إلى البيت متعبين لكن سعداء. كانت رحلة جميلة لا تُنسى أبدًا.")
    def odd_pace(spoken, ref, ref_text, *, engine, speed, nfe, seed):
        sr = 24000
        factor = [0.9, 1.1, 0.94, 1.06][len(spoken) % 4]                  # the model "randomly" speaks too fast/slow
        n = int(sr * len(spoken) * 0.07 * factor / speed)
        t = np.arange(n) / sr
        w = sum(np.sin(2 * np.pi * 140 * h * t) / h for h in range(1, 8)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t) ** 2)
        return (0.2 * w).astype(np.float32), sr
    p = planmod.build(text, voice="narrator_deep", controls={"stability": 1.0})
    take = render.render(p, synth_fn=odd_pace)
    q = take["qc"]
    rate = next(c for c in q["checks"] if c["id"] == "rate")
    assert rate["before"] > 0.05 and rate["after"] < rate["before"] * 0.5, rate
    ids = {c["id"] for c in q["checks"]}
    assert {"clipping", "noise", "reverb", "level", "rate", "pauses", "words", "identity", "mastering", "pitch", "tone"} <= ids
    assert next(c for c in q["checks"] if c["id"] == "clipping")["status"] == "ok"
    d = render.TAKES / take["id"]
    assert sf.info(str(d / "master.wav")).subtype == "PCM_24"
    import subprocess, json as _j
    br = int(_j.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=bit_rate", "-of", "json", str(d / "master.mp3")],
                                     capture_output=True, text=True).stdout)["format"]["bit_rate"])
    assert br >= 300000


def test_personas_colour_emotions_differently():
    """Same sentence + same emotion + same intensity: each persona must deliver it differently (not just a speed change)."""
    txt = "ارتفع الغبار حتى حجب السماء وضاع الطريق."
    out = {}
    for vid in ("documentary_cinematic", "promo_energetic", "anchor_formal", "narrator_deep", "narrator_female"):
        p = planmod.build(txt, voice=vid, emotion="excitement")
        seg = p["segments"][0]
        seg["emotion"], seg["intensity"] = "excitement", 0.9
        p = planmod.recompute(p)
        out[vid] = p["segments"][0]["params"]
    # promo gets the strongest excitement, anchor/documentary the most restrained
    assert out["promo_energetic"]["gain_db"] > out["narrator_deep"]["gain_db"] > out["anchor_formal"]["gain_db"]
    assert out["promo_energetic"]["gain_db"] > out["documentary_cinematic"]["gain_db"]
    rates = {k: round(v["speed_mul"], 3) for k, v in out.items()}
    assert len(set(rates.values())) >= 4, rates
    # grandeur is the documentary persona's signature
    g = {}
    for vid in ("documentary_cinematic", "promo_energetic"):
        p = planmod.build(txt, voice=vid)
        p["segments"][0]["emotion"], p["segments"][0]["intensity"] = "grandeur", 0.9
        from svs import voices as _v
        g[vid] = planmod.recompute(p)["segments"][0]["params"]["speed_mul"] * _v.get_voice(vid)["speed"]   # effective speed
    assert g["documentary_cinematic"] < g["promo_energetic"]


def test_every_emotion_has_a_recording_prompt():
    from svs import delivery
    for k in delivery.EMOTIONS:
        if k == "neutral":
            continue
        d, t = delivery.EMOTION_PROMPTS[k]
        assert d and len(t.split()) >= 8, k                      # a sentence long enough for a 5-10 s clip
