# Fakham-MG — منظومة عروض وألعاب تفاعلية

كل الأدوات مجانية ومفتوحة المصدر، بدون أي API مدفوع.

| الأمر | الوظيفة |
|---|---|
| `npm run serve` | تشغيل الخادم على http://localhost:4173 |
| `npm test` | اختبارات Playwright لكل الأزرار والتفاعلات (5 اختبارات) |
| `npm run gen:pptx` | بناء `output/interactive-demo.pptx` (أزرار روابط داخلية + اختبار) |
| `npm run gen:sounds` | توليد المؤثرات الصوتية محلياً في `assets/sfx` |

- `presentations/demo` — عرض Reveal.js يعمل باللمس (RTL) مع GSAP وHowler
- `games/quiz` — لعبة تعليمية من 3 مراحل (نقاط، قلوب، فوز/خسارة)
- `scenes/room` — مشهد Three.js (PBR، ظلال ناعمة، تقريب/ابتعاد وجولة سينمائية)
- `scripts/make-pptx.mjs` — PowerPoint عبر pptxgenjs

## خوادم MCP (إعداد محلي في `claude mcp`)
```
claude mcp add playwright -- npx -y @playwright/mcp@latest
claude mcp add context7 -- npx -y @upstash/context7-mcp@latest
claude mcp add github -e GITHUB_PERSONAL_ACCESS_TOKEN=<token> -- docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN ghcr.io/github/github-mcp-server
```
خادم GitHub يتطلب Docker يعمل وتوكن شخصي.
