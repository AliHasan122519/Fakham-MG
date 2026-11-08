"""Voice registry: bundled presets + user-created voices (JSON in SVS_HOME/voices)."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from . import audio, config

_ID_RE = re.compile(r"^[a-z0-9_\-]{2,40}$")


def check_id(voice_id: str) -> str:
    """Voice ids become file names: only [a-z0-9_-]{2,40} is accepted (blocks '../' traversal)."""
    if not isinstance(voice_id, str) or not _ID_RE.match(voice_id):
        raise ValueError("المعرّف: أحرف إنجليزية صغيرة وأرقام و _ - فقط (2-40).")
    return voice_id


def _presets() -> dict:
    return json.loads(config.PRESETS.read_text(encoding="utf-8"))


def dialects() -> dict:
    return _presets()["dialects"]


def emotions() -> dict:
    return _presets()["emotions"]


def _resolve_ref(v: dict) -> Path | None:
    """Reference clip lookup order: explicit path > user refs > bundled refs."""
    cands = []
    if v.get("ref_audio"):
        p = Path(v["ref_audio"])
        cands.append(p if p.is_absolute() else config.ROOT / p)
    cands += [config.USER_REFS / f"{v['id']}.wav", config.BUNDLED_REFS / f"{v['id']}.wav"]
    for c in cands:
        if c.is_file():
            return c
    return None


def _ref_text(v: dict) -> str | None:
    if v.get("ref_text"):
        return v["ref_text"]
    for d in (config.USER_REFS, config.BUNDLED_REFS):
        t = d / f"{v['id']}.txt"
        if t.is_file():
            return t.read_text(encoding="utf-8").strip()
    return None


# Reference prompts shipped inside the MIT-licensed `habibi-tts` pip package (model authors' own
# inference examples). Used ONLY as a demo fallback so the studio works on first run.
# Not a substitute for an authorised reference of your own; the model weights are non-commercial.
DEMO_REFS = {
    "najdi": ("Najdi.wav", "تكفى طمني انا اليوم ماني بنايم ولا هو بداخل عيني النوم الين اتطمن عليه."),
    "hijazi": ("Hijazi.wav", "ابغاك تحقق معاه بس بشكل ودي لانه سلطان يمر بظروف صعبة شوية."),
    "khaleeji": ("Gulf.wav", "وين تو الناس متى تصحى ومتى تفطر وتغير يبيلك ساعة يعني بالله تروح الشغل الساعة عشره."),
    "white": ("Najdi.wav", "تكفى طمني انا اليوم ماني بنايم ولا هو بداخل عيني النوم الين اتطمن عليه."),
}


def demo_reference(dialect: str) -> tuple[Path, str] | None:
    try:
        from importlib.resources import files
        name, text = DEMO_REFS.get(dialect, DEMO_REFS["najdi"])
        p = Path(str(files("habibi_tts").joinpath(f"assets/{name}")))
        return (p, text) if p.is_file() else None
    except Exception:
        return None


def list_voices() -> list[dict]:
    voices = {v["id"]: dict(v) for v in _presets()["voices"]}
    for f in sorted(config.USER_VOICES.glob("*.json")):
        try:
            v = json.loads(f.read_text(encoding="utf-8"))
            voices[v["id"]] = {**v, "builtin": False}
        except (json.JSONDecodeError, KeyError):
            continue
    out = []
    for v in voices.values():
        ref = _resolve_ref(v)
        v["has_reference"] = bool(ref and _ref_text(v))
        v["reference_source"] = "own" if v["has_reference"] else ("demo" if demo_reference(v.get("default_dialect", "najdi")) else "none")
        v["ready"] = v["reference_source"] != "none"
        v.pop("ref_audio", None)
        out.append(v)
    return out


def get_voice(voice_id: str) -> dict:
    check_id(voice_id)
    for v in list_voices():
        if v["id"] == voice_id:
            full = dict(v)
            raw = {x["id"]: x for x in _presets()["voices"]}.get(voice_id) or json.loads(
                (config.USER_VOICES / f"{voice_id}.json").read_text(encoding="utf-8"))
            full["_ref_path"] = _resolve_ref(raw)
            full["_ref_text"] = _ref_text(raw)
            return full
    raise KeyError(f"صوت غير موجود: {voice_id}")


def reference_for(v: dict, dialect: str) -> tuple[Path, str, str]:
    """(path, transcript, source) — own reference if present, else the dialect demo clip."""
    if v.get("_ref_path") and v.get("_ref_text"):
        return v["_ref_path"], v["_ref_text"], "own"
    d = demo_reference(dialect)
    if d:
        return d[0], d[1], "demo"
    raise KeyError("no reference")


def save_reference(voice_id: str, src_audio: Path, transcript: str) -> Path:
    """Attach an authorised reference clip (5-10 s) + its exact transcript to a voice."""
    check_id(voice_id)
    if not transcript.strip():
        raise ValueError("النص المنطوق في المقطع المرجعي (ref_text) مطلوب وبدقة.")
    if len(transcript) > 1000:
        raise ValueError("النص المرجعي طويل جدًا (الحد 1000 حرف).")
    dst = config.USER_REFS / f"{voice_id}.wav"
    audio.prepare_reference(Path(src_audio), dst)
    (config.USER_REFS / f"{voice_id}.txt").write_text(transcript.strip(), encoding="utf-8")
    return dst


def create_voice(voice_id: str, name_ar: str, *, gender: str = "male", description: str = "",
                 default_dialect: str = "najdi", speed: float = 1.0, nfe_step: int = 32,
                 fx: dict | None = None, base_on: str | None = None) -> dict:
    check_id(voice_id)
    if default_dialect not in dialects():
        raise ValueError(f"لهجة غير معروفة: {default_dialect}")
    base = {}
    if base_on:
        base = {k: v for k, v in get_voice(base_on).items() if k in ("fx", "speed", "nfe_step", "gender")}
    v = dict(id=voice_id, name_ar=name_ar, gender=gender, description=description,
             default_dialect=default_dialect, speed=speed, nfe_step=nfe_step,
             fx=fx if fx is not None else base.get("fx", {}), ref_audio=None, ref_text=None)
    (config.USER_VOICES / f"{voice_id}.json").write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")
    if base_on:  # inherit reference too, so the new persona is immediately usable
        b = get_voice(base_on)
        if b["_ref_path"] and b["_ref_text"]:
            shutil.copy(b["_ref_path"], config.USER_REFS / f"{voice_id}.wav")
            (config.USER_REFS / f"{voice_id}.txt").write_text(b["_ref_text"], encoding="utf-8")
    return get_voice(voice_id)


def delete_voice(voice_id: str) -> None:
    check_id(voice_id)
    p = config.USER_VOICES / f"{voice_id}.json"
    if not p.is_file():
        raise KeyError("يمكن حذف الأصوات التي أنشأتها فقط.")
    p.unlink()
    for ext in ("wav", "txt"):
        (config.USER_REFS / f"{voice_id}.{ext}").unlink(missing_ok=True)
