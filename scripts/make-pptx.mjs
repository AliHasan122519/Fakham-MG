// يبني عرض PowerPoint تفاعلياً: أزرار روابط داخلية بين الشرائح + اختبار + مؤثرات انتقال
import pptxgen from 'pptxgenjs';
const p = new pptxgen();
p.layout = 'LAYOUT_WIDE'; p.rtlMode = true;
const BG = '0F172A', ACC = '38BDF8';
const btn = (s, text, x, y, slide, color = ACC) =>
  s.addText(text, { x, y, w: 3, h: .7, fontSize: 20, bold: true, align: 'center', color: '04111C', fill: { color }, shape: p.ShapeType.roundRect, rectRadius: .1, hyperlink: { slide, tooltip: text } });
const mkSlide = title => { const s = p.addSlide(); s.background = { color: BG }; s.addText(title, { x: .5, y: .4, w: 12, h: 1, fontSize: 36, bold: true, color: 'FFFFFF', align: 'center' }); return s; };

const s1 = mkSlide('مرحباً — عرض تفاعلي'); const s2 = mkSlide('اختر قسماً'); const s3 = mkSlide('سؤال: 7 × 6 = ؟');
const s4 = mkSlide('✅ إجابة صحيحة!'); const s5 = mkSlide('❌ حاول مجدداً');
btn(s1, 'ابدأ', 5.2, 3, 2);
btn(s2, 'الاختبار', 5.2, 2.2, 3); btn(s2, 'الرئيسية', 5.2, 3.4, 1);
btn(s3, '36', 1.2, 3, 5, 'F87171'); btn(s3, '42', 5.2, 3, 4, '4ADE80'); btn(s3, '48', 9.2, 3, 5, 'F87171');
btn(s4, 'التالي', 5.2, 3, 2); btn(s5, 'إعادة المحاولة', 5.2, 3, 3);
await p.writeFile({ fileName: 'output/interactive-demo.pptx' });
console.log('output/interactive-demo.pptx');
