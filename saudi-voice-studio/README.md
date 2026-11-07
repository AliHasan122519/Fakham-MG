# Saudi Voice Studio — استوديو الصوت السعودي

استوديو تعليق صوتي عربي/سعودي **مجاني بالكامل ومفتوح المصدر**: واجهة عربية للجوال + خادم MCP + مهارة Claude Code. المحرك: **NAMAA Saudi TTS V2** (نجدي) مع **Habibi‑TTS السعودي** لبقية اللهجات.

> 📱 من جوال Android فقط عبر Colab: [`START-HERE.md`](START-HERE.md) · 🚀 ابدأ من [`START-HERE.md`](START-HERE.md) · 📖 الدليل: [`docs/ARABIC-GUIDE.md`](docs/ARABIC-GUIDE.md)

## الحالة الفعلية (ما اختُبر وما لم يُختبر)

| البند | الحالة |
|---|---|
| توليد صوت NAMAA (نجدي) على CPU | ✅ مُختبر؛ التحقق بتحويل الصوت لنص (faster‑whisper) أعاد الجملة بدقة جيدة |
| لهجة «حجازية» عبر Habibi‑SAU | ✅ مُختبر مرة (الصوت أعاد النص حرفيًا) — جودة اللهجة نفسها لم أحكم عليها سمعيًا |
| خادم MCP (HTTP) + عميل MCP الرسمي محليًا: قائمة الأدوات، engine_status، generate_voiceover، التنزيل | ✅ مُختبر |
| الرمز السري: 401 دون رمز، 200 معه | ✅ مُختبر |
| الواجهة على عرض جوال (390px) | ✅ تحقق بلقطة شاشة (لا تمرير أفقي) |
| 9 اختبارات آلية (`pytest tests`) | ✅ تنجح |
| سرعة CPU | ~10× مدة الصوت (16 خطوة) — **بطيء**؛ GPU لم يُقَس |
| موصل Claude على الجوال + نفق Cloudflare | ⚠️ **لم يُختبر** (لا وصول لي إلى تطبيقك) |
| دفتر Colab (زر واحد، مكتفٍ بذاته): منطق الخلية محليًا | ✅ مُختبر (بدون Drive/نفق) |
| التشغيل داخل Colab الحقيقي، Drive، GPU، نفق Cloudflare | ⚠️ **لم يُختبر** |
| HF Spaces | ⚠️ **لم يُختبر** |
| أصوات الشخصيات الخمس | ⚠️ تتمايز بالإعدادات والمعالجة فقط؛ كلها تستخدم مقطعًا تجريبيًا واحدًا لكل لهجة حتى ترفع مقاطعك |
| المشاعر | ⚠️ تقريبية (سرعة/نبرة/وقفات) — لا مفتاح عاطفة حقيقي في النموذج |
| الترخيص | ⚠️ **غير تجاري** (CC‑BY‑NC‑SA‑4.0) — [التفاصيل](docs/ENGINES-AND-LICENSES.md) |

## التشغيل السريع (للمتقدمين؛ من الجوال استخدم Colab)
```bash
scripts/setup.sh && scripts/start.sh      # رابط الواجهة + رابط MCP
python -m pytest tests                    # اختبارات سريعة (بلا نماذج)
```

## البنية
```
svs/            text_ar.py  (تنظيف/أرقام/تقسيم)   engine.py (NAMAA/Habibi)   audio.py (FFmpeg)
                voices.py   studio.py   jobs.py   history.py   web.py (UI+API)   mcp_server.py (MCP)
voices/presets.json   الشخصيات واللهجات والمشاعر      web/index.html   الواجهة
colab/                دفتر Colab المجاني               scripts/         setup/start/install_skill
SKILL.md · START-HERE.md · docs/
```

## الأمان والخصوصية
- `SVS_TOKEN` يحمي الواجهة وMCP والتنزيل. لا تنشر الروابط.
- لا تُرفع الأوزان ولا `data/` ولا `voices/refs/*` إلى Git (`.gitignore`).
- استنساخ الصوت يشترط إقرار الحق/الإذن؛ النظام لا يستطيع التحقق منه.

## المتغيرات
`SVS_HOME` `SVS_MODEL_DIR` `SVS_DEVICE=auto|cpu|cuda` `SVS_NFE` `SVS_TOKEN` `SVS_PUBLIC_URL` `SVS_HOST` `SVS_PORT` `SVS_MAX_CHARS`

## الرخص
كود المشروع: MIT. الأوزان لها تراخيصها (انظر `docs/ENGINES-AND-LICENSES.md`).
