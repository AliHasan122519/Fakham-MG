"""Narrator preview samples: a short sentence read by each narrator, cached so the 🔊 button is instant after the first time."""
from __future__ import annotations

import hashlib
import json
import re
import shutil

from . import auto, config, render, voices

SAMPLES = config.DATA / "samples"
SAMPLES.mkdir(parents=True, exist_ok=True)
OWN_TEXT = "مرحبًا بكم، هذا صوتي، وأقرأ لكم النصوص بإلقاء واضح وطبيعي."
_ID = re.compile(r"^[a-z0-9_\-]{2,40}$")


def _text(v: dict) -> str:
    return v.get("sample_text") or OWN_TEXT


def _key(v: dict) -> str:
    ref, _t, src = voices.reference_for(v, v.get("default_dialect", "najdi"))
    st = ref.stat()
    return hashlib.sha256(json.dumps([str(ref), st.st_size, int(st.st_mtime), v.get("fx"), v.get("speed"), _text(v), src,
                                      v.get("style")], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def sample_file(voice_id: str):
    if not _ID.match(voice_id):
        raise ValueError("معرّف غير صالح.")
    p = SAMPLES / f"{voice_id}.mp3"
    if not p.is_file():
        raise FileNotFoundError("لا توجد عينة بعد.")
    return p


def cached(voice_id: str) -> bool:
    try:
        v = voices.get_voice(voice_id)
        m = SAMPLES / f"{voice_id}.json"
        return (SAMPLES / f"{voice_id}.mp3").is_file() and m.is_file() and json.loads(m.read_text())["key"] == _key(v)
    except Exception:
        return False


def make(voice_id: str, progress=None) -> dict:
    """Generate (or reuse) the narrator sample with the full automatic pipeline. Returns {cached, seconds}."""
    v = voices.get_voice(voice_id)
    if cached(voice_id):
        return {"cached": True}
    take = auto.run(_text(v), voice=voice_id, progress=progress)
    shutil.copyfile(render.take_file(take["id"], "master.mp3"), SAMPLES / f"{voice_id}.mp3")
    (SAMPLES / f"{voice_id}.json").write_text(json.dumps({"key": _key(v), "take": take["id"]}), encoding="utf-8")
    return {"cached": False, "seconds": take["metrics"]["after"]["duration_s"], "took": take["took_s"]}
