"""Saudi Voice Studio MCP server (official MCP Python SDK).

Run:
  stdio (Claude Code on a computer):   python -m svs.mcp_server
  HTTP  (Claude mobile / web via a free tunnel):  python -m svs.mcp_server --http
        -> MCP endpoint  /mcp   +  web UI  /   +  downloads /files/<id>.mp3  on ONE port.
"""
from __future__ import annotations

import base64
import binascii
import ipaddress
import socket
import sys
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import quote, urlparse

from mcp.server.mcpserver import MCPServer

from . import config, history, jobs, voices
from .engine import ENGINE

mcp = MCPServer(
    "saudi-voice-studio",
    instructions=(
        "استوديو الصوت السعودي: ولّد تعليقًا صوتيًا عربيًا/سعوديًا. استخدم generate_voiceover؛ "
        "إن لم تنتهِ المهمة خلال مهلة الانتظار استعلم بـ get_job حتى تصبح done ثم أعطِ المستخدم رابط التنزيل. "
        "لا تستنسخ صوتًا إلا بمقطع يملك المستخدم حق استخدامه (i_confirm_rights=true)."),
)


def _link(gid: str, ext: str, download: bool = True) -> str:
    q = []
    if config.TOKEN:
        q.append("token=" + quote(config.TOKEN))
    if download:
        q.append("download=1")
    return f"{config.PUBLIC_URL}/files/{gid}.{ext}" + ("?" + "&".join(q) if q else "")


def _job_view(job: dict) -> dict:
    out = {k: job[k] for k in ("id", "state", "progress", "message") if k in job}
    if job["state"] == "error":
        out["error"] = job.get("error")
    if job["state"] == "done":
        r = job["result"]
        out.update(
            mp3_url=_link(r["id"], "mp3"), wav_url=_link(r["id"], "wav"), stream_url=_link(r["id"], "mp3", False),
            seconds=r["seconds"], took_seconds=r["took"], device=r["device"], voice=r["voice"], dialect=r["dialect"],
            emotion=r["emotion"], reference_source=r["reference_source"], warnings=r.get("warnings", []),
            dialect_native=r["dialect_native"], dialect_note=r["dialect_note"])
    elif job["state"] == "running":
        out["hint"] = "ما زال التوليد جاريًا. استدعِ get_job بعد قليل (على المعالج يستغرق ~10× مدة الصوت)."
    return out


@mcp.tool()
def generate_voiceover(text: str, voice: str = "narrator_deep", dialect: str = "", emotion: str = "neutral",
                       speed: float = 0.0, wait_seconds: int = 45) -> dict:
    """ولّد تعليقًا صوتيًا. voice: معرّف من list_voices. dialect: najdi|hijazi|white|khaleeji (فارغ = افتراضي الشخصية).
    emotion: neutral|enthusiastic|calm|serious|warm|sad|dramatic. speed: 0 = افتراضي الشخصية، أو 0.6-1.4.
    يعيد روابط MP3/WAV عند الاكتمال، وإلا job id لاستعلام get_job."""
    if jobs.running_count() >= jobs.MAX_RUNNING:
        return {"state": "error", "error": "توجد مهام توليد قيد التنفيذ؛ انتظر حتى تنتهي ثم أعد المحاولة."}
    jid = jobs.start(text=text, voice_id=voice, dialect=dialect or None, emotion=emotion,
                     speed=speed or None, fmt="both")
    return _job_view(jobs.wait(jid, max(0, min(wait_seconds, 110))))


@mcp.tool()
def get_job(job_id: str, wait_seconds: int = 30) -> dict:
    """حالة مهمة توليد سابقة (وروابط التنزيل عند اكتمالها)."""
    if not jobs.get(job_id):
        return {"state": "error", "error": "مهمة غير موجودة (ربما أُعيد تشغيل الخادم)."}
    return _job_view(jobs.wait(job_id, max(0, min(wait_seconds, 110))))


@mcp.tool()
def list_voices() -> dict:
    """اعرض الأصوات واللهجات والمشاعر المتاحة وحالة المقطع المرجعي لكل صوت."""
    vs = [{k: v[k] for k in ("id", "name_ar", "gender", "description", "default_dialect", "speed",
                              "reference_source", "builtin") if k in v} for v in voices.list_voices()]
    return {"voices": vs, "dialects": {k: {"name_ar": d["name_ar"], "native": d["native"], "note": d["note"]}
                                      for k, d in voices.dialects().items()},
            "emotions": {k: e["name_ar"] for k, e in voices.emotions().items()},
            "reference_source_meaning": {"own": "مقطعك المصرّح به", "demo": "مقطع تجريبي مرفق بحزمة habibi-tts (لتجربة النظام فقط)"}}


@mcp.tool()
def create_voice(voice_id: str, name_ar: str, gender: str = "male", description: str = "",
                 default_dialect: str = "najdi", speed: float = 1.0, base_on: str = "",
                 pitch_semitones: float = 0.0, bass_db: float = 0.0, treble_db: float = 0.0,
                 reverb: float = 0.0) -> dict:
    """أنشئ شخصية صوتية جديدة (إعدادات إلقاء + معالجة FFmpeg). base_on: صوت موجود لنسخ إعداداته ومقطعه المرجعي.
    الصوت الفعلي يأتي من المقطع المرجعي: استخدم set_voice_reference أو clone_voice."""
    fx = {"pitch_st": pitch_semitones, "bass_db": bass_db, "treble_db": treble_db, "reverb": reverb, "compress": True}
    v = voices.create_voice(voice_id, name_ar, gender=gender, description=description,
                            default_dialect=default_dialect, speed=speed, fx=None if base_on and not any(
                                [pitch_semitones, bass_db, treble_db, reverb]) else fx, base_on=base_on or None)
    return {"created": v["id"], "reference_source": voices.reference_for(v, v["default_dialect"])[2]}


MAX_AUDIO = 25 * 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):  # a redirect could point at an internal address
        return None


def _public_https(url: str) -> None:
    """SSRF guard: https only, host must resolve ONLY to public addresses (no loopback/metadata/private)."""
    u = urlparse(url)
    if u.scheme != "https" or not u.hostname:
        raise ValueError("audio_url يجب أن يبدأ بـ https://")
    for info in socket.getaddrinfo(u.hostname, u.port or 443, proto=socket.IPPROTO_TCP):
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise ValueError("audio_url يشير إلى عنوان داخلي/غير عام — مرفوض.")


def _fetch_audio(audio_base64: str, audio_url: str, td: str) -> Path:
    dst = Path(td) / "ref_in"
    if audio_base64:
        payload = audio_base64.split(",")[-1]
        if len(payload) > MAX_AUDIO * 4 // 3 + 16:
            raise ValueError("المقطع كبير جدًا (الحد 25MB).")
        try:
            dst.write_bytes(base64.b64decode(payload, validate=True))
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"base64 غير صالح: {e}")
    elif audio_url:
        _public_https(audio_url)
        opener = urllib.request.build_opener(_NoRedirect)
        with opener.open(audio_url, timeout=30) as r:  # noqa: S310
            data = r.read(MAX_AUDIO + 1)
        if len(data) > MAX_AUDIO:
            raise ValueError("المقطع كبير جدًا (الحد 25MB).")
        dst.write_bytes(data)
    else:
        raise ValueError("أعطِ audio_base64 أو audio_url.")
    return dst


@mcp.tool()
def set_voice_reference(voice_id: str, transcript: str, i_confirm_rights: bool, audio_base64: str = "",
                        audio_url: str = "") -> dict:
    """اربط مقطعًا مرجعيًا (5-10 ثوانٍ، متحدث واحد، نظيف) بصوت موجود. transcript = النص المنطوق حرفيًا.
    i_confirm_rights يجب أن تكون true: أقرّ المستخدم بأنه يملك حق استخدام هذا الصوت أو لديه إذن صاحبه."""
    if not i_confirm_rights:
        return {"error": "رفضت العملية: لا يُستنسخ صوت دون إقرار صريح بالحق/الإذن (i_confirm_rights=true)."}
    voices.get_voice(voice_id)
    with tempfile.TemporaryDirectory() as td:
        voices.save_reference(voice_id, _fetch_audio(audio_base64, audio_url, td), transcript)
    return {"ok": True, "voice": voice_id}


@mcp.tool()
def clone_voice(voice_id: str, name_ar: str, transcript: str, i_confirm_rights: bool, audio_base64: str = "",
                audio_url: str = "", gender: str = "male", default_dialect: str = "najdi") -> dict:
    """استنسخ صوتًا مصرّحًا به من عينة قصيرة وأنشئ له شخصية جديدة في خطوة واحدة."""
    if not i_confirm_rights:
        return {"error": "رفضت العملية: لا يُستنسخ صوت دون إقرار صريح بالحق/الإذن (i_confirm_rights=true)."}
    voices.create_voice(voice_id, name_ar, gender=gender, default_dialect=default_dialect,
                        fx={"compress": True})
    try:
        with tempfile.TemporaryDirectory() as td:
            voices.save_reference(voice_id, _fetch_audio(audio_base64, audio_url, td), transcript)
    except Exception:
        voices.delete_voice(voice_id)
        raise
    return {"ok": True, "voice": voice_id, "next": f"generate_voiceover(voice='{voice_id}', text=...)"}


@mcp.tool()
def delete_voice(voice_id: str) -> dict:
    """احذف شخصية أنشأتها أنت (لا تُحذف الشخصيات الجاهزة)."""
    voices.delete_voice(voice_id)
    return {"ok": True}


@mcp.tool()
def get_download_links(generation_id: str) -> dict:
    """روابط تنزيل MP3 وWAV لتوليد سابق (من list_history)."""
    rec = history.get(generation_id)
    if not rec:
        return {"error": "معرّف غير موجود."}
    out = {"id": generation_id, "mp3_url": _link(generation_id, "mp3"), "wav_url": _link(generation_id, "wav")}
    if not (config.OUTPUTS / f"{generation_id}.mp3").is_file():
        out["warning"] = "الملف لم يعد موجودًا على الخادم (جلسة جديدة؟). أعد التوليد."
    return out


@mcp.tool()
def list_history(limit: int = 10) -> dict:
    """آخر عمليات التوليد."""
    return {"items": [{**r, "mp3_url": _link(r["id"], "mp3")} for r in history.recent(max(1, min(limit, 50)))]}


@mcp.tool()
def engine_status() -> dict:
    """فحص حالة المحرك: الجهاز (CPU/GPU)، النماذج المحمّلة، FFmpeg، الترخيص."""
    s = ENGINE.status()
    s["license_warning"] = "أوزان NAMAA وHabibi بترخيص CC-BY-NC-SA-4.0: للاستخدام غير التجاري فقط."
    s["public_url"] = config.PUBLIC_URL
    s["auth_required"] = bool(config.TOKEN)
    return s


def build_http_app():
    """One ASGI app: /mcp (MCP), plus the web UI/API/downloads mounted at /."""
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
    from starlette.routing import Mount

    from .web import app as web_app

    app = mcp.streamable_http_app(host="0.0.0.0" if config.HOST == "0.0.0.0" else config.HOST,
                                  transport_security=_transport_security())
    app.router.routes.append(Mount("/", app=web_app))

    class TokenGuard(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if config.TOKEN and request.url.path.startswith("/mcp"):
                import hmac
                given = request.headers.get("x-svs-token") or request.query_params.get("token") or ""
                bearer = request.headers.get("authorization", "")
                if bearer.lower().startswith("bearer "):
                    given = given or bearer[7:]
                if not hmac.compare_digest(given, config.TOKEN):
                    return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

    app.add_middleware(TokenGuard)
    return app


def _transport_security():
    from urllib.parse import urlparse

    from mcp.server.transport_security import TransportSecuritySettings
    host = urlparse(config.PUBLIC_URL).netloc
    return TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                     allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*", host, host.split(":")[0]])


def main() -> None:
    if "--http" in sys.argv:
        import uvicorn
        config.require_safe_bind()
        # access_log=False: the access log would print every request URL, including ?token=...
        uvicorn.run(build_http_app(), host=config.HOST, port=config.PORT, log_level="warning", access_log=False)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
