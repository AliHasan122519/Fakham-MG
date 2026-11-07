import json
def code(s): return {"cell_type":"code","metadata":{},"source":s.strip("\n").splitlines(True),"outputs":[],"execution_count":None}
def md(s): return {"cell_type":"markdown","metadata":{},"source":s.strip("\n").splitlines(True)}
cells=[
md("# استوديو الصوت السعودي — Colab المجاني\n**مجاني 100%.** اختر `Runtime ← Change runtime type ← T4 GPU` إن توفر (غير مضمون). لا تُفعّل أي خطة مدفوعة.\n> هذا الدفتر لم يُختبر على Colab من بيئة التطوير؛ إن فشلت خطوة أخبرني بنص الخطأ."),
code('''
#@title 1) تثبيت (مرة لكل جلسة)
import os, subprocess
REPO = "https://github.com/alihasan122519/fakham-mg.git"   #@param {type:"string"}
BRANCH = "ccr-0cfe010a-fw0fdc"                            #@param {type:"string"}
# مستودع خاص؟ أضف سرًّا باسم GH_TOKEN من أيقونة المفتاح في Colab (fine-grained، قراءة فقط).
try:
    from google.colab import userdata; tok = userdata.get("GH_TOKEN")
    REPO = REPO.replace("https://", f"https://{tok}@")
except Exception: pass
if not os.path.exists("/content/fakham-mg"):
    subprocess.run(["git","clone","-b",BRANCH,REPO,"/content/fakham-mg"],check=True)
%cd /content/fakham-mg/saudi-voice-studio
!apt-get -qq install -y ffmpeg > /dev/null
!pip -q install habibi-tts "mcp>=2.3,<3" fastapi uvicorn python-multipart soundfile huggingface_hub scipy
'''),
code('''
#@title 2) حفظ النماذج والبيانات في Google Drive (لتستعيدها بعد توقف الجلسة)
from google.colab import drive; drive.mount("/content/drive")
import os; os.makedirs("/content/drive/MyDrive/saudi-voice-studio", exist_ok=True)
os.environ["SVS_HOME"] = "/content/drive/MyDrive/saudi-voice-studio"   # outputs, history, voices, models
!python scripts/download_models.py najdi saudi   # يتخطى ما هو محمّل
'''),
code('''
#@title 3) تشغيل الخادم + نفق Cloudflare المجاني (بلا حساب)
import os, re, secrets, subprocess, time, threading
os.environ["SVS_TOKEN"] = os.environ.get("SVS_TOKEN") or secrets.token_urlsafe(24)
os.environ["SVS_HOST"] = "127.0.0.1"
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
tun = subprocess.Popen(["cloudflared","tunnel","--url","http://127.0.0.1:8765","--no-autoupdate"],stderr=subprocess.PIPE,text=True)
url=None
for line in tun.stderr:
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line)
    if m: url=m.group(0); break
os.environ["SVS_PUBLIC_URL"] = url
print("الواجهة :", f"{url}/?token={os.environ['SVS_TOKEN']}")
print("رابط MCP:", f"{url}/mcp?token={os.environ['SVS_TOKEN']}")
threading.Thread(target=lambda: subprocess.run(["python","-m","svs.mcp_server","--http"]),daemon=True).start()
time.sleep(5)
import torch; print("GPU:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "— CPU فقط")
'''),
md("أبقِ هذه الصفحة مفتوحة. إن توقفت الجلسة: أعد تشغيل الخلايا 1→3 (النماذج على Drive، فلا إعادة تنزيل). الرابط يتغير في كل تشغيل؛ حدّث الموصل في Claude."),
]
nb={"cells":cells,"metadata":{"accelerator":"GPU","colab":{"provenance":[]},"kernelspec":{"name":"python3","display_name":"Python 3"}},"nbformat":4,"nbformat_minor":5}
json.dump(nb,open(__import__("os").path.join(__import__("os").path.dirname(__file__),"saudi_voice_studio.ipynb"),"w"),ensure_ascii=False,indent=1)
