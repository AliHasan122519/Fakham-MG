#@title ▶ تشغيل استوديو الصوت السعودي { display-mode: "form" }
#@markdown اضغط ▶ مرة واحدة وانتظر (أول مرة ~5–10 دقائق للتنزيل، وبعدها أسرع لأن النماذج على Drive).
#@markdown مجاني بالكامل — لا تُفعّل أي اشتراك ولا تُدخل بطاقة.
DOWNLOAD_SAUDI_MODEL = True  #@param {type:"boolean"}
USE_GOOGLE_DRIVE = True  #@param {type:"boolean"}
#@markdown ⚠️ ربط Drive يمنح بيئة التشغيل وصولًا لكل ملفات Drive لديك (Colab لا يسمح بربط مجلد واحد فقط). الكود يكتب داخل `saudi-voice-studio/` فقط، لكن إن كنت لا تريد هذا الوصول أزل العلامة (ستُعاد تنزيلات النماذج ~2.6GB كل جلسة ولا تُحفظ الشخصيات).
ALLOW_UNVERIFIED_CLOUDFLARED = False  #@param {type:"boolean"}
#@markdown (اتركه معطّلًا: لا يُشغَّل cloudflared إلا إذا طابقت بصمته SHA-256 المثبّتة.)
#@markdown (النموذج الثاني للهجات الحجازية/البيضاء/الخليجية، ~1.3GB إضافية على Drive. النجدية لا تحتاجه.)

import base64, io, json, os, re, secrets, subprocess, sys, tarfile, threading, time, urllib.request
from pathlib import Path

PAYLOAD = "__PAYLOAD__"          # الكود مضمَّن هنا: لا حاجة لـ GitHub ولا لأوامر يدوية
PORT = 8765
# cloudflared: بصمة SHA-256 للإصدار الذي فُحص (linux-amd64). أي ملف آخر يُرفض.
CF_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
CF_SHA256 = "1f99fb518b83dec2ba8d870a7cc991d82a77e6196295a3d2754ba17ab9f91872"
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
    for ti in t.getmembers():   # لا مسارات مطلقة ولا ".." ولا روابط ولا أجهزة
        if ti.name.startswith("/") or ".." in Path(ti.name).parts or not (ti.isfile() or ti.isdir()):
            raise SystemExit(f"رُفض: عنصر غير آمن في الحزمة: {ti.name}")
    if hasattr(tarfile, "data_filter"):
        t.extractall(APP, filter="data")
    else:
        raise SystemExit("إصدار بايثون قديم: لا يدعم فك الضغط الآمن.")

# 2) Google Drive (حفظ النماذج والبيانات) — اختياري
step(2, "ربط Google Drive لحفظ النماذج (وافق على نافذة الإذن)…" if USE_GOOGLE_DRIVE else "بدون Drive: التخزين مؤقت في الجلسة.")
home = None
if IN_COLAB and USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        home = Path("/content/drive/MyDrive/saudi-voice-studio")
    except Exception as e:
        say(f"⚠️ تعذّر ربط Drive ({e}). سأحفظ مؤقتًا في الجلسة؛ سيُعاد التنزيل في كل مرة.")
home = home or Path(os.environ.get("SVS_HOME", "/content/svs-data"))
home.mkdir(parents=True, exist_ok=True)
# رمز سري جديد في كل تشغيل، ولا يُحفظ على Drive أبدًا (تسريب الرابط القديم لا يفيد أحدًا بعد انتهاء الجلسة)
TOKEN = secrets.token_urlsafe(32)

# 3) المكتبات
step(3, "تثبيت المكتبات المجانية (pip + ffmpeg)…")
if os.environ.get("SVS_SKIP_INSTALL") != "1":
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
        subprocess.run("apt-get -qq update && apt-get -qq install -y ffmpeg", shell=True, check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "habibi-tts==0.1.1", "mcp==2.3.0", "fastapi==0.162.2",
                    "uvicorn==0.54.0", "python-multipart==0.0.33", "soundfile==0.14.0", "huggingface_hub==1.33.0",
                    "scipy==1.18.1"], check=True)   # إصدارات مثبّتة هي التي اختُبرت

# 4) الجهاز + النماذج
step(4, "فحص GPU وتنزيل النماذج إلى Drive…")
try:
    import torch
    GPU = torch.cuda.is_available()
    say(f"   الجهاز: {'GPU — ' + torch.cuda.get_device_name(0) if GPU else 'CPU فقط (بطيء ~10× مدة الصوت). لتجربة GPU: Runtime ← Change runtime type ← T4 GPU، ثم أعد التشغيل.'}")
except Exception:
    GPU = False
# نمرّر للعمليات الفرعية متغيرات البيئة غير السرية فقط (لا مفاتيح/رموز/كلمات مرور من بيئتك)
_SECRETISH = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.I)
env = {k: v for k, v in os.environ.items() if not _SECRETISH.search(k)}
env.update({"SVS_HOME": str(home), "SVS_TOKEN": TOKEN, "SVS_HOST": "127.0.0.1", "SVS_PORT": str(PORT)})
which = ["najdi"] + (["saudi"] if DOWNLOAD_SAUDI_MODEL else [])
subprocess.run([sys.executable, str(APP / "scripts/download_models.py"), *which], check=True, env=env, cwd=APP)

# 5) رابط HTTPS مجاني (Cloudflare Quick Tunnel — بلا حساب)
step(5, "فتح رابط HTTPS مجاني…")
public = None
cf = Path("/tmp/cloudflared")
tunnel_log = Path("/tmp/cloudflared.log")
try:
    import hashlib
    def _sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if cf.exists() and _sha(cf) != CF_SHA256 and not ALLOW_UNVERIFIED_CLOUDFLARED:
        cf.unlink()
    if not cf.exists():
        urllib.request.urlretrieve(CF_URL, cf)
    if _sha(cf) != CF_SHA256 and not ALLOW_UNVERIFIED_CLOUDFLARED:
        cf.unlink()
        raise RuntimeError("بصمة cloudflared لا تطابق المثبّتة (ربما صدر إصدار جديد). رفضتُ تشغيله حفاظًا على الأمان. "
                           "إن كنت تثق بالمصدر فعّل ALLOW_UNVERIFIED_CLOUDFLARED.")
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
    say("\n🔒 الرابطان يحتويان رمزك السري ومن يملكهما يستطيع استخدام الاستوديو (وموارد Colab الخاصة بك).")
    say("   • لا تشاركهما، ولا تحفظ نسخة من الدفتر «مع المخرجات» (File ← Save a copy) ولا تشارك الدفتر وهو يعرضهما.")
    say("   • الرمز يتجدد كل تشغيل؛ اضغط «Disconnect and delete runtime» لإبطال الرابط فورًا.")
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
