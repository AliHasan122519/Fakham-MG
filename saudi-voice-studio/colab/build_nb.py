"""Builds colab/saudi_voice_studio.ipynb: one self-contained notebook (code embedded, no GitHub access needed)."""
import base64, io, json, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ["svs", "voices/presets.json", "voices/lexicon_default.json", "voices/semantic_prototypes.json", "voices/refs/.gitkeep", "web/index.html", "scripts/download_models.py", "scripts/benchmark.py", "scripts/stability.py", "scripts/session_test.py", "requirements.txt"]


def payload() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for item in INCLUDE:
            p = ROOT / item
            t.add(p, arcname=item, filter=lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti)
    return base64.b64encode(buf.getvalue()).decode()


def src(s: str):
    return s.splitlines(True)


run_code = (ROOT / "colab/run_studio.py").read_text(encoding="utf-8").replace("__PAYLOAD__", payload())
md = """# 🎙️ استوديو الصوت السعودي (مجاني 100%)

**من جوال Android فقط، بلا كمبيوتر وبلا تثبيت برامج:**

1. افتح هذا الدفتر في **Chrome** على جوالك (إن ظهرت صفحة مبسطة: قائمة ⋮ ← **موقع سطح المكتب / Desktop site**).
2. (اختياري، أسرع بكثير) القائمة: **Runtime ← Change runtime type ← T4 GPU**. إن لم يتوفر GPU اتركه CPU — يعمل لكن أبطأ.
3. اضغط زر ▶ في الخلية أدناه المسماة **تشغيل استوديو الصوت السعودي**، ووافق على إذن Google Drive.
4. **انتظر ولا تضغط ■ (إيقاف)**: أول مرة 5–10 دقائق (تنزيل ~8GB). سترى سطور تقدّم وعدّادًا كل 15 ثانية؛ الصمت المؤقت طبيعي.
5. بعدها يظهر **رابط الواجهة**: افتحه، اكتب النص، ولّد، استمع، نزّل MP3.

> ⚠️ مجاني: لا تُفعّل Colab Pro ولا أي اشتراك. الجلسة تنقطع بعد خمول/ساعات؛ اضغط ▶ من جديد وستعود الأمور سريعًا لأن النماذج على Drive.
> ⚠️ الأوزان بترخيص **غير تجاري** (CC-BY-NC-SA-4.0).
> ⚠️ هذا الدفتر لم يُختبر داخل Colab الحقيقي من بيئة التطوير (اختُبر منطقه محليًا). إن ظهر خطأ، انسخ نصه وأرسله."""
nb = {"cells": [
    {"cell_type": "markdown", "metadata": {}, "source": src(md)},
    {"cell_type": "code", "metadata": {"cellView": "form", "id": "run"}, "source": src(run_code), "outputs": [], "execution_count": None},
], "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "name": "Saudi Voice Studio"},
                "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 5}
(ROOT / "colab/saudi_voice_studio.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("notebook written:", (ROOT / "colab/saudi_voice_studio.ipynb").stat().st_size // 1024, "KB")
