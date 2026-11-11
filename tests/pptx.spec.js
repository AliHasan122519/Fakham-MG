import { test, expect } from '@playwright/test';
import { execSync } from 'node:child_process';
test('PPTX: ملف صالح وروابط التنقل الداخلية موجودة', () => {
  execSync('node scripts/make-pptx.mjs');
  const list = execSync('unzip -l output/interactive-demo.pptx').toString();
  expect((list.match(/ppt\/slides\/slide\d+\.xml/g) || []).length).toBe(5);
  const rels = execSync('unzip -p output/interactive-demo.pptx ppt/slides/_rels/slide3.xml.rels').toString();
  expect(rels).toMatch(/slide4\.xml/); expect(rels).toMatch(/slide5\.xml/);
});
