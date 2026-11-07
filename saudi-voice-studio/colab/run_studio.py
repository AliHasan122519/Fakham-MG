#@title ▶ تشغيل استوديو الصوت السعودي { display-mode: "form" }
#@markdown اضغط ▶ مرة واحدة وانتظر (أول مرة ~5–10 دقائق للتنزيل، وبعدها أسرع لأن النماذج على Drive).
#@markdown مجاني بالكامل — لا تُفعّل أي اشتراك ولا تُدخل بطاقة.
DOWNLOAD_SAUDI_MODEL = True  #@param {type:"boolean"}
#@markdown (النموذج الثاني للهجات الحجازية/البيضاء/الخليجية، ~1.3GB إضافية على Drive. النجدية لا تحتاجه.)

import base64, io, json, os, re, secrets, subprocess, sys, tarfile, threading, time, urllib.request
from pathlib import Path

PAYLOAD = "__PAYLOAD__"          # الكود مضمَّن هنا: لا حاجة لـ GitHub ولا لأوامر يدوية
PORT = 8765
IN_COLAB = "google.colab" in sys.modules
APP = Path(os.environ.get("SVS_APP_DIR", "/content/saudi-voice-studio"))


def say(msg):
    print(msg, flush=True)


def step(n, msg):
    say(f"\n[{n}/6] {msg}")


# 1) فك الكود
step(1, "تجهيز ملفات المشروع…")
APP.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as t:
    t.extractall(APP, filter="data") if sys.version_info >= (3, 12) else t.extractall(APP)

# 2) Google Drive (حفظ النماذج والبيانات)
step(2, "ربط Google Drive لحفظ النماذج (وافق على نافذة الإذن)…")
home = None
if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        home = Path("/content/drive/MyDrive/saudi-voice-studio")
    except Exception as e:
        say(f"⚠️ تعذّر ربط Drive ({e}). سأحفظ مؤقتًا في الجلسة؛ سيُعاد التنزيل في كل مرة.")
home = home or Path(os.environ.get("SVS_HOME", "/content/svs-data"))
home.mkdir(parents=True, exist_ok=True)
tok_file = home / ".token"
if not tok_file.exists():
    tok_file.write_text(secrets.token_urlsafe(24))
TOKEN = tok_file.read_text().strip()

# 3) المكتبات
step(3, "تثبيت المكتبات المجانية (pip + ffmpeg)…")
if os.environ.get("SVS_SKIP_INSTALL") != "1":
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
        subprocess.run("apt-get -qq update && apt-get -qq install -y ffmpeg", shell=True, check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "habibi-tts", "mcp>=2.3,<3", "fastapi", "uvicorn",
                    "python-multipart", "soundfile", "huggingface_hub", "scipy"], check=True)

# 4) الجهاز + النماذج
step(4, "فحص GPU وتنزيل النماذج إلى Drive…")
try:
    import torch
    GPU = torch.cuda.is_available()
    say(f"   الجهاز: {'GPU — ' + torch.cuda.get_device_name(0) if GPU else 'CPU فقط (بطيء ~10× مدة الصوت). لتجربة GPU: Runtime ← Change runtime type ← T4 GPU، ثم أعد التشغيل.'}")
except Exception:
    GPU = False
env = {**os.environ, "SVS_HOME": str(home), "SVS_TOKEN": TOKEN, "SVS_HOST": "127.0.0.1", "SVS_PORT": str(PORT)}
which = ["najdi"] + (["saudi"] if DOWNLOAD_SAUDI_MODEL else [])
subprocess.run([sys.executable, str(APP / "scripts/download_models.py"), *which], check=True, env=env, cwd=APP)

# 5) رابط HTTPS مجاني (Cloudflare Quick Tunnel — بلا حساب)
step(5, "فتح رابط HTTPS مجاني…")
public = None
cf = Path("/tmp/cloudflared")
tunnel_log = Path("/tmp/cloudflared.log")
try:
    if not cf.exists():
        urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", cf)
        cf.chmod(0o755)
    tunnel_log.write_text("")
    subprocess.Popen([str(cf), "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
                     stdout=open(tunnel_log, "a"), stderr=subprocess.STDOUT)
    for _ in range(45):
        # نتجاهل api.trycloudflare.com: هذا مضيف خدمة Cloudflare نفسها وليس رابطك
        m = [u for u in re.findall(r"https://([a-z0-9-]+)\.trycloudflare\.com", tunnel_log.read_text()) if u != "api"]
        if m:
            public = f"https://{m[0]}.trycloudflare.com"
            break
        time.sleep(1)
except Exception as e:
    say(f"⚠️ تعذّر فتح نفق Cloudflare: {e}")
env["SVS_PUBLIC_URL"] = public or f"http://127.0.0.1:{PORT}"

# 6) تشغيل الخادم (الواجهة + MCP + التنزيل على منفذ واحد)
step(6, "تشغيل الخادم…")
server_log = open("/tmp/svs_server.log", "w")
server = subprocess.Popen([sys.executable, "-m", "svs.mcp_server", "--http"], cwd=APP, env=env,
                          stdout=server_log, stderr=subprocess.STDOUT)
ok = False
for _ in range(60):
    try:
        ok = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=3)).get("ok")
        if ok:
            break
    except Exception:
        time.sleep(1)
if not ok:
    say("❌ لم يبدأ الخادم. آخر السجل:\n" + Path("/tmp/svs_server.log").read_text()[-1500:])
    raise SystemExit(1)

ui = f"{env['SVS_PUBLIC_URL']}/?token={TOKEN}"
mcp_url = f"{env['SVS_PUBLIC_URL']}/mcp?token={TOKEN}"
tunnel_ok = False
if public:
    try:   # تحقق أن الرابط العام يصل فعلًا
        urllib.request.urlopen(f"{public}/api/health", timeout=15)
        tunnel_ok = True
    except Exception as e:
        say(f"⚠️ النفق فُتح لكن لم أتمكن من الوصول إليه من داخل Colab ({e}). سأنتظر 20 ثانية وأعيد المحاولة…")
        time.sleep(20)
        try:
            urllib.request.urlopen(f"{public}/api/health", timeout=15)
            tunnel_ok = True
        except Exception:
            public = None  # لا نعرض رابطًا لا يعمل

say("\n" + "=" * 50)
if not public:
    say("⚠️ تعذّر الحصول على رابط عام يعمل (راجع /tmp/cloudflared.log). لن يعمل ربط Claude الآن.")
if public:
    say("✅ الخادم يعمل. اضغط مطوّلًا على الرابط لنسخه أو افتحه:")
    say(f"\n🎙️ رابط الواجهة (افتحه في متصفح الجوال):\n{ui}")
    say(f"\n🔌 رابط MCP (للصقه في موصل Claude — اختياري):\n{mcp_url}")
    say("\n🔒 الرابطان يحتويان رمزك السري — لا تشاركهما.")
else:
    say("✅ الخادم يعمل لكن بلا رابط عام (فشل النفق). استخدم رابط Colab الداخلي أدناه للواجهة فقط؛ ربط Claude غير متاح الآن.")
    if IN_COLAB:
        try:
            from google.colab import output
            output.serve_kernel_port_as_window(PORT, path=f"/?token={TOKEN}")
        except Exception as e:
            say(f"تعذّر فتح الرابط الداخلي: {e}")
say("=" * 50)
say("أبقِ هذه الصفحة مفتوحة وهذا الخلية تعمل (ما دامت تدور فالخادم حي). عند الانقطاع: اضغط ▶ من جديد.")
say(f"آخر تحديث: {time.strftime('%H:%M:%S')}")

while True:               # يُبقي الجلسة حية ويراقب الخادم
    if server.poll() is not None:
        say("❌ توقف الخادم:\n" + Path("/tmp/svs_server.log").read_text()[-1500:])
        break
    time.sleep(30)
