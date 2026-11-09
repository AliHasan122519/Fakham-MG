"""FastAPI web app: Arabic mobile UI + JSON API + file downloads (also reused by MCP for links)."""
from __future__ import annotations

import hmac
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from . import auto, config, delivery, enhance, samples, history, jobs, plan as planmod, pronounce, refaudio, render, studio, voices
from .engine import ENGINE, EngineUnavailable

app = FastAPI(title="Saudi Voice Studio", docs_url=None, redoc_url=None)
MAX_UPLOAD = 25 * 1024 * 1024


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"          # the token lives in the URL: never leak it via Referer
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; "
        "media-src 'self' blob:; connect-src 'self'; form-action 'none'; base-uri 'none'; frame-ancestors 'none'")
    return resp


def _save_upload(up: UploadFile, dst: Path) -> None:
    size = 0
    with dst.open("wb") as f:
        while chunk := up.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                raise ValueError("الملف كبير جدًا (الحد 25MB).")
            f.write(chunk)
WEB = config.ROOT / "web"


def auth(request: Request) -> None:
    if not config.TOKEN:
        return
    given = request.headers.get("x-svs-token") or request.query_params.get("token") or ""
    if not hmac.compare_digest(given, config.TOKEN):
        raise HTTPException(401, "رمز الدخول غير صحيح (SVS_TOKEN).")


class AnalyzeReq(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    voice: str = "narrator_deep"
    dialect: str | None = None
    style: str | None = None
    mode: str = "auto"
    emotion: str | None = None
    controls: dict | None = None


class GenReq(BaseModel):
    plan: dict | None = None            # an analysed/edited plan (preferred)
    text: str | None = Field(default=None, max_length=20000)
    voice: str = "narrator_deep"
    dialect: str | None = None
    emotion: str = "auto"
    style: str | None = None
    speed: float | None = Field(default=None, ge=0.5, le=1.6)
    controls: dict | None = None


class RegenReq(BaseModel):
    segment_id: str
    new_seed: bool = True
    edits: dict | None = None


class PronReq(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    dialect: str = "najdi"
    light_dialect: bool = False


class LexReq(BaseModel):
    entries: list[dict] = Field(max_length=500)


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"ok": True, "auth_required": bool(config.TOKEN)}


@app.get("/api/status", dependencies=[Depends(auth)])
def status():
    return ENGINE.status()


@app.get("/api/voices", dependencies=[Depends(auth)])
def api_voices():
    return {"voices": voices.list_voices(), "dialects": voices.dialects(), "emotions": voices.emotions()}


@app.post("/api/analyze", dependencies=[Depends(auth)])
def api_analyze(req: AnalyzeReq):
    try:
        return studio.analyze(req.text, voice=req.voice, dialect=req.dialect, style=req.style, mode=req.mode,
                              controls=req.controls, emotion=req.emotion)
    except (studio.StudioError, KeyError) as e:
        raise HTTPException(400, str(e))


class EnhanceReq(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    voice: str = "narrator_deep"
    dialect: str | None = None
    style: str | None = None
    mode: str = "auto"
    emotion: str | None = None
    controls: dict | None = None
    diacritics: str = "selective"
    compare_raw: bool = True


@app.post("/api/enhance", dependencies=[Depends(auth)])
def api_enhance(req: EnhanceReq):
    """التحسين الذكي: analysis + pronunciation + context diacritics + emotions + delivery plan (for review)."""
    try:
        return enhance.smart_enhance(req.text, voice=req.voice, dialect=req.dialect, style=req.style or None, mode=req.mode,
                                     controls=req.controls, emotion=req.emotion, diacritics=req.diacritics,
                                     compare_raw=req.compare_raw)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


class AutoReq(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    voice: str | None = None
    dialect: str | None = None
    max_quality: bool = False
    review: bool = False
    style: str | None = None
    controls: dict | None = None          # only the keys the user touched; everything else stays automatic
    draft: bool = False                   # fast preview: skip automatic repairs/re-rolls


@app.post("/api/auto", dependencies=[Depends(auth)])
def api_auto(req: AutoReq):
    """The one-button path: smart enhancement + generation in the default voice. review=true returns the plan instead."""
    voice = req.voice or voices.default_voice_id()
    if voice == "auto" and req.review:
        voice = voices.default_voice_id()
    if req.review:
        try:
            return enhance.smart_enhance(req.text, voice=voice, dialect=req.dialect, compare_raw=False,
                                         style=req.style or None, controls=req.controls)
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e))
    if jobs.running_count() >= jobs.MAX_RUNNING:
        raise HTTPException(429, "توجد مهام توليد قيد التنفيذ؛ انتظر حتى تنتهي.")
    try:
        if voice != "auto":                # "auto" = narrator chosen from the detected text type
            voices.get_voice(voice)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))
    return {"job": jobs.start(lambda progress=None, **kw: auto.run(progress=progress, **kw),
                              text=req.text, voice=voice, dialect=req.dialect, max_quality=req.max_quality,
                              style=req.style or None, controls=req.controls, draft=req.draft)}


@app.post("/api/voices/{voice_id}/sample", dependencies=[Depends(auth)])
def api_sample(voice_id: str):
    """🔊 narrator preview. Cached → instant; otherwise a job (first press only)."""
    try:
        voices.check_id(voice_id)
        voices.get_voice(voice_id)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    if samples.cached(voice_id):
        return {"cached": True, "url": f"/files/sample/{voice_id}.mp3"}
    if jobs.running_count() >= jobs.MAX_RUNNING:
        raise HTTPException(429, "توجد مهام قيد التنفيذ؛ انتظر.")
    return {"job": jobs.start(lambda progress=None, **kw: samples.make(progress=progress, **kw), voice_id=voice_id)}


@app.get("/files/sample/{voice_id}.mp3", dependencies=[Depends(auth)])
def api_sample_file(voice_id: str):
    try:
        return FileResponse(samples.sample_file(voice_id), media_type="audio/mpeg")
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))


class DefaultReq(BaseModel):
    voice_id: str


@app.post("/api/default_voice", dependencies=[Depends(auth)])
def api_default_voice(req: DefaultReq):
    try:
        voices.set_default_voice(req.voice_id)
        return {"ok": True, "default": req.voice_id}
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/plan/recompute", dependencies=[Depends(auth)])
def api_recompute(plan: dict):
    try:
        return planmod.validate(plan)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/pronounce", dependencies=[Depends(auth)])
def api_pronounce(req: PronReq):
    return pronounce.to_spoken(req.text, req.dialect, light_dialect=req.light_dialect)


@app.get("/api/lexicon", dependencies=[Depends(auth)])
def api_lexicon():
    lx = pronounce.lexicon()
    return {"user_entries": lx["user_entries"], "builtin_count": len(lx["entries"]) - len(lx["user_entries"])}


@app.put("/api/lexicon", dependencies=[Depends(auth)])
def api_lexicon_put(req: LexReq):
    return {"saved": pronounce.save_user_entries(req.entries)}


@app.get("/api/meta", dependencies=[Depends(auth)])
def api_meta():
    return {"emotions": {k: {"name_ar": v[0]} for k, v in delivery.EMOTIONS.items()},
            "styles": {k: {"name_ar": v[0]} for k, v in delivery.STYLES.items()},
            "controls": delivery.DEFAULT_CONTROLS, "limits": delivery.LIMITS,
            "override_limits": delivery.OVERRIDE_LIMITS,
            "emotion_prompts": {k: {"direction": v[0], "text": v[1]} for k, v in delivery.EMOTION_PROMPTS.items()}}


@app.post("/api/generate", dependencies=[Depends(auth)])
def api_generate(req: GenReq):
    if jobs.running_count() >= jobs.MAX_RUNNING:
        raise HTTPException(429, "توجد مهام توليد قيد التنفيذ؛ انتظر حتى تنتهي.")
    if not req.plan and not req.text:
        raise HTTPException(400, "أعطِ plan أو text.")
    jid = jobs.start(text=req.text or "", voice_id=req.voice, dialect=req.dialect, emotion=req.emotion,
                     speed=req.speed, plan=req.plan, style=req.style, controls=req.controls)
    return {"job": jid}


def _regen(progress=None, **kw):
    return render.regenerate(progress=progress, **kw)


@app.post("/api/takes/{take_id}/regenerate", dependencies=[Depends(auth)])
def api_regen(take_id: str, req: RegenReq):
    if jobs.running_count() >= jobs.MAX_RUNNING:
        raise HTTPException(429, "توجد مهام توليد قيد التنفيذ؛ انتظر حتى تنتهي.")
    try:
        render.load_take(take_id)
    except render.RenderError as e:
        raise HTTPException(404, str(e))
    return {"job": jobs.start(_regen, take_id=take_id, segment_id=req.segment_id, edits=req.edits, new_seed=req.new_seed)}


@app.get("/api/jobs/{jid}", dependencies=[Depends(auth)])
def api_job(jid: str):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "مهمة غير موجودة")
    view = {k: v for k, v in job.items() if k != "result"}
    if job["state"] == "done":
        r = job["result"]
        if isinstance(r, dict) and "plan" in r and "metrics" in r:
            view["take"] = studio.take_view(r)
        else:
            view["result"] = r
    return view


@app.get("/api/takes", dependencies=[Depends(auth)])
def api_takes(limit: int = 30):
    return {"items": render.list_takes(min(limit, 100))}


@app.get("/api/takes/{take_id}", dependencies=[Depends(auth)])
def api_take(take_id: str):
    try:
        return studio.take_view(render.load_take(take_id))
    except render.RenderError as e:
        raise HTTPException(404, str(e))


@app.get("/api/history", dependencies=[Depends(auth)])
def api_history(limit: int = 20):
    return {"items": history.recent(min(limit, 100))}


@app.get("/files/{take_id}/{name}", dependencies=[Depends(auth)])
def files_take(take_id: str, name: str, download: int = 0):
    try:
        p = render.take_file(take_id, name)
    except render.RenderError as e:
        raise HTTPException(404, str(e))
    return FileResponse(p, media_type="audio/mpeg" if name.endswith(".mp3") else "audio/wav",
                        filename=f"saudi-voice-{take_id}-{name}" if download else None)


@app.get("/files/{take_id}.{ext}", dependencies=[Depends(auth)])
def files_legacy(take_id: str, ext: str, download: int = 0):
    """Stable short link used by MCP: /files/<take>.mp3 → mastered output."""
    if ext not in ("mp3", "wav"):
        raise HTTPException(404, "غير موجود")
    return files_take(take_id, f"master.{ext}", download)


@app.post("/api/reference/analyze", dependencies=[Depends(auth)])
def api_ref_analyze(ref: UploadFile = File(...), ref_text: str = Form("")):
    """Upload → decode/analyse/clean/verify → preview (nothing is saved to any voice yet)."""
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "ref_in"
            _save_upload(ref, tmp)
            return refaudio.analyze_upload(tmp, ref_text)
    except (refaudio.RefError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.get("/files/ref/{pid}/{name}", dependencies=[Depends(auth)])
def api_ref_file(pid: str, name: str):
    try:
        return FileResponse(refaudio.preview_file(pid, name), media_type="audio/wav")
    except refaudio.RefError as e:
        raise HTTPException(404, str(e))


class RefTestReq(BaseModel):
    preview_id: str
    dialect: str = "najdi"


@app.post("/api/reference/test", dependencies=[Depends(auth)])
def api_ref_test(req: RefTestReq):
    if jobs.running_count() >= jobs.MAX_RUNNING:
        raise HTTPException(429, "توجد مهام قيد التنفيذ؛ انتظر.")
    try:
        refaudio.preview_file(req.preview_id, "processed.wav")
    except refaudio.RefError as e:
        raise HTTPException(404, str(e))
    return {"job": jobs.start(lambda progress=None, **kw: refaudio.test_voice(progress=progress, **kw),
                              preview_id=req.preview_id, dialect=req.dialect)}


class RefCommitReq(BaseModel):
    preview_id: str
    ref_text: str = Field(min_length=1, max_length=600)
    emotion: str | None = None
    use: str = "processed"
    # when creating a NEW voice together with its reference:
    name_ar: str | None = None
    gender: str = "male"
    base_on: str | None = None
    default_dialect: str = "najdi"


@app.post("/api/voices/{voice_id}/reference/commit", dependencies=[Depends(auth)])
def api_ref_commit(voice_id: str, req: RefCommitReq):
    try:
        voices.check_id(voice_id)
        if req.name_ar and not any(v["id"] == voice_id for v in voices.list_voices()):
            voices.create_voice(voice_id, req.name_ar, gender=req.gender, default_dialect=req.default_dialect,
                                base_on=req.base_on or None)
        refaudio.commit(voice_id, req.preview_id, req.ref_text, req.emotion, use=req.use)
        v = voices.get_voice(voice_id)
        return {k: x for k, x in v.items() if not k.startswith("_")}
    except (refaudio.RefError, ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/voices", dependencies=[Depends(auth)])
def api_create_voice(voice_id: str = Form(...), name_ar: str = Form(...), gender: str = Form("male"),
                     description: str = Form(""), default_dialect: str = Form("najdi"), speed: float = Form(1.0),
                     base_on: str = Form(""), ref_text: str = Form(""), ref: UploadFile | None = File(None)):
    """Create a voice (optionally with a reference that goes through the full verification pipeline)."""
    try:
        if ref is not None and ref.filename:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / "ref_in"
                _save_upload(ref, tmp)
                res = refaudio.analyze_upload(tmp, ref_text)
            if res["verdict"]["status"] == "reject":
                raise ValueError(" ".join(res["verdict"]["reject"]))
        v = voices.create_voice(voice_id, name_ar, gender=gender, description=description,
                                default_dialect=default_dialect, speed=speed, base_on=base_on or None)
        if ref is not None and ref.filename:
            refaudio.commit(voice_id, res["preview_id"], res["transcript"])
        v = voices.get_voice(voice_id)
        return {k: x for k, x in v.items() if not k.startswith("_")}
    except (ValueError, KeyError, refaudio.RefError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/voices/{voice_id}/reference", dependencies=[Depends(auth)])
def api_set_ref(voice_id: str, ref_text: str = Form(""), ref: UploadFile = File(...), emotion: str = Form("")):
    try:
        voices.get_voice(voice_id)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "ref_in"
            _save_upload(ref, tmp)
            res = voices.save_reference(voice_id, tmp, ref_text, emotion=emotion or None)
        return {"ok": True, "quality": res["quality"], "verdict": res["verdict"]}
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


@app.delete("/api/voices/{voice_id}", dependencies=[Depends(auth)])
def api_delete_voice(voice_id: str):
    try:
        voices.delete_voice(voice_id)
        return {"ok": True}
    except KeyError as e:
        raise HTTPException(400, str(e))


def main() -> None:
    import uvicorn
    config.require_safe_bind()
    uvicorn.run(app, host=config.HOST, port=config.PORT, access_log=False)  # access log would print ?token=


if __name__ == "__main__":
    main()
