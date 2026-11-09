"""The QC must tell healthy mastering from deliberately damaged audio — on REAL speech (bundled demo clips), not synthetic tones."""
import numpy as np
import pytest
import soundfile as sf

from svs import audio, qc

try:
    from importlib.resources import files
    ASSETS = files("habibi_tts").joinpath("assets")
    _ = ASSETS.joinpath("Najdi.wav")
except Exception:                                   # pragma: no cover
    ASSETS = None

pytestmark = pytest.mark.skipif(ASSETS is None, reason="habibi_tts demo assets not installed")


def _load(name):
    x, sr = sf.read(str(ASSETS.joinpath(name)), dtype="float32")
    x = x if x.ndim == 1 else x[:, 0]
    x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * 24000 / sr)), np.arange(len(x)), x).astype(np.float32)
    return x / np.abs(x).max() * 0.5


@pytest.fixture(scope="module")
def pre():
    return _load("Najdi.wav")


def _f(x, chain):
    return audio.ffmpeg_wave(x, chain)


DECLARED = ["equalizer=f=125:t=q:w=1.0:g=3.0", "equalizer=f=250:t=q:w=1.0:g=2.0", "equalizer=f=7000:t=q:w=0.8:g=-2.5"]


def test_healthy_mastering_passes(pre):
    chain = ",".join(DECLARED) + ",loudnorm=I=-16:TP=-1.5:LRA=11"
    mean, mx = qc.chain_intent(chain)
    r = qc.transparency(pre, _f(pre, chain), intended_db=mean, intended_max_db=mx)
    assert r["ok"], r
    r2 = qc.transparency(pre, _f(pre, "loudnorm=I=-16:TP=-1.5:LRA=11"))
    assert r2["ok"], r2


@pytest.mark.parametrize("name,chain", [
    ("muffled", "lowpass=f=3000"),
    ("harsh", "equalizer=f=3000:t=q:w=1.0:g=8"),
    ("crushed", "acompressor=threshold=-40dB:ratio=20:attack=1:release=50:makeup=12"),
    ("reverb", "aecho=0.8:0.88:60|120|240:0.4|0.3|0.2"),
    ("undeclared_double_eq", "equalizer=f=125:t=q:w=1.0:g=6.0,equalizer=f=250:t=q:w=1.0:g=4.0,equalizer=f=7000:t=q:w=0.8:g=-5"),
])
def test_filter_defects_are_caught(pre, name, chain):
    r = qc.transparency(pre, _f(pre, chain))
    assert not r["ok"], (name, r)


def test_hiss_and_clipping_are_caught(pre):
    rng = np.random.default_rng(0)
    hiss = pre + (rng.standard_normal(len(pre)) * 0.01).astype(np.float32)
    assert not qc.transparency(pre, hiss)["ok"]
    clipped = np.clip(pre * 6, -1, 1).astype(np.float32)
    r = qc.transparency(pre, clipped)
    assert not r["ok"] and any("تقطيع" in x for x in r["reasons"]), r


def test_compressor_that_raises_a_noisy_sources_floor_is_flagged(pre):
    # a physical side effect, not a false alarm: 2:1 compression lifts the noise floor of this (noisy) demo clip by ~7 dB re speech
    r = qc.transparency(pre, _f(pre, "acompressor=threshold=-24dB:ratio=2:attack=20:release=250"))
    assert not r["ok"] and any("الضجيج" in x for x in r["reasons"]), r


def test_reference_distance_must_not_grow(pre):
    ref = _load("Hijazi.wav")
    chain = ",".join(DECLARED)
    mean, mx = qc.chain_intent(chain)
    post = _f(pre, chain)
    r = qc.transparency(pre, post, ref=ref, intended_db=mean, intended_max_db=mx)
    assert "dist_to_reference_after" in r and r["ok"] == (r["dist_to_reference_after"] <= r["dist_to_reference_before"] + 0.5 and not r["reasons"])


def test_chain_intent_parses_equalizers_and_shelves():
    m, mx = qc.chain_intent("equalizer=f=125:t=q:w=1.0:g=3.0,equalizer=f=7000:t=q:w=0.8:g=-2,bass=g=4:f=180:w=0.7,loudnorm=I=-16")
    assert mx == 4.0 and abs(m - (3 + 2 + 2 * 4) / 8) < 1e-6
    assert qc.chain_intent("loudnorm=I=-16,highpass=f=60") == (0.0, 0.0)


def test_pitch_assessment_is_statistically_sound():
    a = qc.f0_assess([9.0, 9.4, 8.7], baseline_f0=8.5)                  # 3 sentences: no spread estimate, anchor on the learned baseline
    assert a["mode"] == "anchor" and a["ok"]
    b = qc.f0_assess([9.0, 12.9, 8.7], baseline_f0=8.5)
    assert b["mode"] == "anchor" and not b["ok"] and b["outlier_idx"] == [1]
    c = qc.f0_assess([9.0, 9.3, 8.8, 9.1, 12.4, 8.9, 9.2], baseline_f0=8.5)  # n>=5: one outlier must not inflate the robust spread
    assert c["mode"] == "spread" and c["ok"] and c["outlier_idx"] == [4]
    d = qc.f0_assess([6.0, 9.0, 12.0, 7.0, 11.0, 8.0], baseline_f0=8.5)       # genuinely unstable narrator
    assert d["mode"] == "spread" and not d["ok"]
    assert qc.f0_assess([], None)["mode"] == "insufficient"
