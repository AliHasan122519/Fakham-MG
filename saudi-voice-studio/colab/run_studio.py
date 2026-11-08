#@title ▶ تشغيل استوديو الصوت السعودي { display-mode: "form" }
#@markdown اضغط ▶ مرة واحدة وانتظر. أول مرة 5–10 دقائق (تنزيل ~8GB مكتبات + نماذج) وسترى تقدّمًا وعدّادًا كل 15 ثانية — **لا تضغط إيقاف**. بعدها أسرع لأن النماذج على Drive.
#@markdown مجاني بالكامل — لا تُفعّل أي اشتراك ولا تُدخل بطاقة.
DOWNLOAD_SAUDI_MODEL = True  #@param {type:"boolean"}
USE_GOOGLE_DRIVE = True  #@param {type:"boolean"}
#@markdown ⚠️ ربط Drive يمنح بيئة التشغيل وصولًا لكل ملفات Drive لديك (Colab لا يسمح بربط مجلد واحد فقط). الكود يكتب داخل `saudi-voice-studio/` فقط، لكن إن كنت لا تريد هذا الوصول أزل العلامة (ستُعاد تنزيلات النماذج ~2.6GB كل جلسة ولا تُحفظ الشخصيات).
ALLOW_UNVERIFIED_CLOUDFLARED = False  #@param {type:"boolean"}
#@markdown (اتركه معطّلًا: لا يُشغَّل cloudflared إلا إذا طابقت بصمته SHA-256 المثبّتة.)
#@markdown (النموذج الثاني للهجات الحجازية/البيضاء/الخليجية، ~1.3GB إضافية على Drive. النجدية لا تحتاجه.)

import base64, hashlib, io, json, os, re, secrets, subprocess, sys, tarfile, threading, time, urllib.request
from pathlib import Path

PAYLOAD = "__PAYLOAD__"          # الكود مضمَّن هنا: لا حاجة لـ GitHub ولا لأوامر يدوية
PORT = 8765
# cloudflared: بصمة SHA-256 للإصدار الذي فُحص (linux-amd64). أي ملف آخر يُرفض.
CF_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
CF_SHA256 = "1f99fb518b83dec2ba8d870a7cc991d82a77e6196295a3d2754ba17ab9f91872"
# بيئة بايثون معزولة بإصدارات مثبّتة (هي التي اختُبرت). لا نلمس مكتبات Colab (torch/transformers…) إطلاقًا.
UV_PIN = "uv==0.13.7"
PINS = ["torch==2.8.0", "torchaudio==2.8.0", "habibi-tts==0.1.1", "mcp==2.3.0", "fastapi==0.162.2",
        "uvicorn==0.54.0", "python-multipart==0.0.33", "soundfile==0.14.0", "huggingface_hub==1.33.0"]
IN_COLAB = "google.colab" in sys.modules
APP = Path(os.environ.get("SVS_APP_DIR", "/content/saudi-voice-studio"))
VENV = Path(os.environ.get("SVS_VENV_DIR", "/content/svs-venv"))
VPY = VENV / "bin" / "python"
CHILDREN = []                    # كل العمليات الفرعية، لنُنهيها عند الإيقاف


def say(msg):
    print(msg, flush=True)


def step(n, msg):
    say(f"\n[{n}/6] {msg}")


def run(cmd, label, env=None, cwd=None):
    """يشغّل عملية فرعية ويعرض مخرجاتها **مباشرة** في الخلية.

    سبب الإصلاح: في Colab لا تظهر مخرجات العمليات الفرعية في الخلية إن لم تُلتقَط، فكانت خطوة pip/التنزيل
    (دقائق) تبدو متجمّدة والمستخدم يضغط إيقاف فيظهر KeyboardInterrupt داخل subprocess.waitpid.
    """
    say(f"   ▶ {label}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=cwd)
    CHILDREN.append(p)
    start, last_out, last_print, tail, buf = time.time(), [time.time()], 0.0, [], b""
    stop = threading.Event()

    def beat():                  # نبضة حياة كل 15 ثانية إن لم يصدر شيء
        while not stop.wait(5):
            if time.time() - last_out[0] >= 15:
                say(f"   ⏳ {label}… {int(time.time() - start)}ث — ما زال يعمل، لا تُوقفه")
                last_out[0] = time.time()

    threading.Thread(target=beat, daemon=True).start()
    try:
        while True:
            chunk = os.read(p.stdout.fileno(), 4096)
            if not chunk:
                break
            buf += chunk
            *lines, buf = re.split(rb"[\r\n]+", buf)
            for raw in lines:
                txt = raw.decode("utf-8", "replace").strip()
                if not txt:
                    continue
                tail = (tail + [txt])[-25:]
                if time.time() - last_print >= 1.0:        # لا نغرق الخلية بأشرطة التقدم
                    say("     " + txt[:200])
                    last_print, last_out[0] = time.time(), time.time()
        rc = p.wait()
    finally:
        stop.set()
    if rc != 0:
        raise RuntimeError(f"فشلت الخطوة «{label}» (رمز {rc}). آخر المخرجات:\n" + "\n".join(tail))
    say(f"   ✔ {label} ({int(time.time() - start)}ث)")


def shutdown():
    for c in CHILDREN:
        try:
            if c.poll() is None:
                c.terminate()
        except Exception:
            pass


def main():
    # 1) فك الكود
    step(1, "تجهيز ملفات المشروع…")
    APP.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as t:
        for ti in t.getmembers():   # لا مسارات مطلقة ولا ".." ولا روابط ولا أجهزة
            if ti.name.startswith("/") or ".." in Path(ti.name).parts or not (ti.isfile() or ti.isdir()):
                raise SystemExit(f"رُفض: عنصر غير آمن في الحزمة: {ti.name}")
        if not hasattr(tarfile, "data_filter"):
            raise SystemExit("إصدار بايثون قديم: لا يدعم فك الضغط الآمن.")
        t.extractall(APP, filter="data")

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
    # رمز سري جديد في كل تشغيل، ولا يُحفظ على Drive أبدًا
    token = secrets.token_urlsafe(32)

    # نمرّر للعمليات الفرعية متغيرات البيئة غير السرية فقط (لا مفاتيح/رموز/كلمات مرور من بيئتك)
    secretish = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.I)
    env = {k: v for k, v in os.environ.items() if not secretish.search(k)}
    env.update({"SVS_HOME": str(home), "SVS_TOKEN": token, "SVS_HOST": "127.0.0.1", "SVS_PORT": str(PORT),
                "PYTHONUNBUFFERED": "1", "UV_NO_PROGRESS": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})

    # 3) بيئة بايثون معزولة + المكتبات
    step(3, "تجهيز بيئة بايثون معزولة (أول مرة 3–8 دقائق؛ سترى التقدّم)…")
    stamp = VENV / (".ok-" + hashlib.sha256((UV_PIN + "|".join(PINS)).encode()).hexdigest()[:12])
    if not (VPY.exists() and stamp.exists()):
        if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
            run(["bash", "-c", "apt-get -qq update && apt-get -qq install -y ffmpeg"], "تثبيت ffmpeg", env=env)
        run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", UV_PIN], "تثبيت أداة التثبيت السريعة uv", env=env)
        uv = [sys.executable, "-m", "uv"]
        run(uv + ["venv", "--python", sys.executable, "--clear", str(VENV)], "إنشاء البيئة المعزولة", env=env)
        run(uv + ["pip", "install", "--python", str(VPY)] + PINS, "تنزيل وتثبيت المكتبات (torch + النموذج)", env=env)
        stamp.write_text("ok")
    else:
        say("   ✔ البيئة جاهزة من تشغيل سابق في هذه الجلسة.")

    # 4) الجهاز + النماذج
    step(4, "فحص GPU وتنزيل النماذج…")
    probe = subprocess.run([str(VPY), "-c", "import torch;print(('GPU — '+torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU')"],
                           capture_output=True, text=True, env=env).stdout.strip()
    say("   الجهاز: " + (probe if probe.startswith("GPU") else "CPU فقط (بطيء ~10× مدة الصوت). لتجربة GPU: Runtime ← Change runtime type ← T4 GPU ثم أعد التشغيل."))
    which = ["najdi"] + (["saudi"] if DOWNLOAD_SAUDI_MODEL else [])
    run([str(VPY), str(APP / "scripts/download_models.py"), *which], "تنزيل النماذج من Hugging Face (قد يستغرق دقائق)", env=env, cwd=APP)

    # 5) رابط HTTPS مجاني (Cloudflare Quick Tunnel — بلا حساب)
    step(5, "فتح رابط HTTPS مجاني…")
    public = None
    cf = Path("/tmp/cloudflared")
    tunnel_log = Path("/tmp/cloudflared.log")
    try:
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
        CHILDREN.append(subprocess.Popen([str(cf), "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
                                         stdout=open(tunnel_log, "a"), stderr=subprocess.STDOUT, env=env))
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
    server = subprocess.Popen([str(VPY), "-m", "svs.mcp_server", "--http"], cwd=APP, env=env,
                              stdout=open("/tmp/svs_server.log", "w"), stderr=subprocess.STDOUT)
    CHILDREN.append(server)
    ok = False
    for _ in range(90):
        try:
            ok = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=3)).get("ok")
            if ok:
                break
        except Exception:
            time.sleep(1)
        if server.poll() is not None:
            break
    if not ok:
        say("❌ لم يبدأ الخادم. آخر السجل:\n" + Path("/tmp/svs_server.log").read_text()[-1500:])
        shutdown()
        raise SystemExit(1)

    ui = f"{env['SVS_PUBLIC_URL']}/?token={token}"
    mcp_url = f"{env['SVS_PUBLIC_URL']}/mcp?token={token}"
    if public:
        try:   # تحقق أن الرابط العام يصل فعلًا
            urllib.request.urlopen(f"{public}/api/health", timeout=15)
        except Exception as e:
            say(f"⚠️ النفق فُتح لكن لم أتمكن من الوصول إليه من داخل Colab ({e}). سأنتظر 20 ثانية وأعيد المحاولة…")
            time.sleep(20)
            try:
                urllib.request.urlopen(f"{public}/api/health", timeout=15)
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
                output.serve_kernel_port_as_window(PORT, path=f"/?token={token}")
            except Exception as e:
                say(f"تعذّر فتح الرابط الداخلي: {e}")
    say("=" * 50)
    say("أبقِ هذه الصفحة مفتوحة وهذه الخلية تعمل (ما دامت تدور فالخادم حي). للإيقاف اضغط ■. عند الانقطاع: اضغط ▶ من جديد.")
    say(f"آخر تحديث: {time.strftime('%H:%M:%S')}")

    while True:               # يُبقي الجلسة حية ويراقب الخادم
        if server.poll() is not None:
            say("❌ توقف الخادم:\n" + Path("/tmp/svs_server.log").read_text()[-1500:])
            break
        time.sleep(30)


try:
    main()
except KeyboardInterrupt:
    shutdown()
    say("\n⏹ أُوقف التشغيل (ضغطتَ ■ أو قطع Colab الخلية). أُنهيتُ العمليات الفرعية. اضغط ▶ من جديد للمتابعة — "
        "ما اكتمل (البيئة والنماذج) لا يُعاد، لكن أكمل الانتظار في المرة القادمة ولا توقفه أثناء التنزيل.")
except Exception as e:
    shutdown()
    say(f"\n❌ {e}")
