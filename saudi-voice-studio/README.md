# Saudi Voice Studio — استوديو الصوت السعودي

استوديو تعليق صوتي عربي/سعودي **مجاني بالكامل ومفتوح المصدر**: واجهة عربية للجوال + خادم MCP + مهارة Claude Code. المحرك: **NAMAA Saudi TTS V2** (نجدي) مع **Habibi‑TTS السعودي** لبقية اللهجات.

> 📱 من جوال Android فقط عبر Colab: [`START-HERE.md`](START-HERE.md) · 🚀 ابدأ من [`START-HERE.md`](START-HERE.md) · 📖 الدليل: [`docs/ARABIC-GUIDE.md`](docs/ARABIC-GUIDE.md)

## ما الجديد (v2: الأداء الذكي)
تحليل النص ← اكتشاف المشاعر لكل جملة ← **خطة إلقاء قابلة للتعديل** ← توليد جملة بجملة (ذاكرة مؤقتة) ← انتقالات طبيعية ← ماستر تكيّفي ← MP3 256kbps، مع قاموس نطق، مقارنة قبل/بعد، وإعادة توليد جملة واحدة. التفاصيل: [`docs/PERFORMANCE-ENGINE.md`](docs/PERFORMANCE-ENGINE.md) · الأرقام: [`docs/QUALITY-REPORT.md`](docs/QUALITY-REPORT.md).

## الحالة الفعلية (ما اختُبر وما لم يُختبر)
| البند | الحالة |
|---|---|
| توليد حقيقي بنموذج NAMAA/Habibi على CPU عبر خط الأداء الجديد، مع تفريغ صوتي للتحقق | ✅ |
| 5 شخصيات مولّدة فعليًا وقياس نغمتها وطيفها | ✅ (انظر التقرير) |
| خطة الأداء: اكتشاف المشاعر، وقفات/سرعة/جهارة/نبرة لكل جملة، تنويع ضد الرتابة | ✅ اختبارات آلية + توليد حقيقي |
| قاموس النطق، أرقام/وحدات، كلمات ملتبسة، محرر النص المنطوق | ✅ |
| قبل/بعد + مقاييس موضوعية، إعادة توليد جملة، مقارنة نسخ | ✅ |
| الواجهة (390px) بتشغيل آلي كامل على الخادم الحقيقي | ✅ |
| المحرك التعبيري الاختياري (Chatterbox) محليًا عبر العامل المنفصل | ✅ (فصحى) — ⚠️ لم يُختبر في Colab |
| 60+ اختبار آلي (`pytest tests`) بما فيها الأمان | ✅ |
| **الاستماع البشري/الطبيعية**: لا أستطيع الاستماع؛ لا أدّعي جودة بشرية | ⚠️ قيِّمها بأذنك |
| داخل Colab الحقيقي (Drive، GPU، نفق Cloudflare، موصل Claude) | ⚠️ **لم يُختبر** |
| سرعة GPU | ⚠️ لم تُقَس |
| العاطفة | تشكيل أداء، وليست تغيير جرس الصوت دون مقاطع عاطفية مرجعية |
| الأصوات الخمس | 3 مقاطع تجريبية + هوية/إيقاع/ماستر مختلف؛ ليست 5 ممثلين |
| الترخيص | ⚠️ **غير تجاري** لـ NAMAA/Habibi — [التفاصيل](docs/ENGINES-AND-LICENSES.md) |

## التشغيل السريع (للمتقدمين؛ من الجوال استخدم Colab)
```bash
scripts/setup.sh && scripts/start.sh      # رابط الواجهة + رابط MCP
python -m pytest tests                    # اختبارات سريعة (بلا نماذج)
```

## البنية
```
svs/            analysis.py (مشاعر/نوع النص)  pronounce.py (نطق/قاموس)  delivery.py (معاملات الإلقاء)  plan.py
                render.py (توليد/انتقالات/ماستر/نسخ)  engine.py (NAMAA/Habibi)  expressive*.py (اختياري)  audio.py (FFmpeg)
                voices.py  studio.py  jobs.py  history.py  web.py (UI+API)  mcp_server.py (MCP)
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
