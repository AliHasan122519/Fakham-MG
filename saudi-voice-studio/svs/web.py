"""FastAPI web app: Arabic mobile UI + JSON API + file downloads (also reused by MCP for links)."""
from __future__ import annotations

import hmac
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from . import config, history, jobs, studio, voices
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


class GenReq(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    voice: str = "narrator_deep"
    dialect: str | None = None
    emotion: str = "neutral"
    speed: float | None = Field(default=None, ge=0.5, le=1.6)
    format: str = "mp3"


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


@app.post("/api/generate", dependencies=[Depends(auth)])
def api_generate(req: GenReq):
    if jobs.running_count() >= jobs.MAX_RUNNING:
        raise HTTPException(429, "توجد مهام توليد قيد التنفيذ؛ انتظر حتى تنتهي.")
    jid = jobs.start(text=req.text, voice_id=req.voice, dialect=req.dialect, emotion=req.emotion,
                     speed=req.speed, fmt=req.format)
    return {"job": jid}


@app.get("/api/jobs/{jid}", dependencies=[Depends(auth)])
def api_job(jid: str):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "مهمة غير موجودة")
    return job


@app.get("/api/history", dependencies=[Depends(auth)])
def api_history(limit: int = 20):
    return {"items": history.recent(min(limit, 100))}


@app.get("/files/{gid}.{ext}", dependencies=[Depends(auth)])
def files(gid: str, ext: str, download: int = 0):
    try:
        p = studio.output_path(gid, ext)
    except studio.StudioError as e:
        raise HTTPException(404, str(e))
    return FileResponse(p, media_type="audio/mpeg" if ext == "mp3" else "audio/wav",
                        filename=f"saudi-voice-{gid}.{ext}" if download else None)


@app.post("/api/voices", dependencies=[Depends(auth)])
def api_create_voice(voice_id: str = Form(...), name_ar: str = Form(...), gender: str = Form("male"),
                     description: str = Form(""), default_dialect: str = Form("najdi"), speed: float = Form(1.0),
                     base_on: str = Form(""), ref_text: str = Form(""), ref: UploadFile | None = File(None)):
    try:
        v = voices.create_voice(voice_id, name_ar, gender=gender, description=description,
                                default_dialect=default_dialect, speed=speed, base_on=base_on or None)
        if ref is not None and ref.filename:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / "ref_in"
                _save_upload(ref, tmp)
                voices.save_reference(voice_id, tmp, ref_text)
        v = voices.get_voice(voice_id)
        return {k: x for k, x in v.items() if not k.startswith("_")}
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/voices/{voice_id}/reference", dependencies=[Depends(auth)])
def api_set_ref(voice_id: str, ref_text: str = Form(...), ref: UploadFile = File(...)):
    try:
        voices.get_voice(voice_id)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "ref_in"
            _save_upload(ref, tmp)
            voices.save_reference(voice_id, tmp, ref_text)
        return {"ok": True}
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
