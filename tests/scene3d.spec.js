import { test, expect } from '@playwright/test';
test('المشهد ثلاثي الأبعاد: الكاميرا والأزرار', async ({ page }) => {
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/scenes/room/'); await page.waitForFunction(() => window.__sceneReady);
  const r0 = await page.evaluate(() => window.__rig.r);
  await page.click('#zin'); await page.waitForTimeout(1200); const r1 = await page.evaluate(() => window.__rig.r);
  expect(r1).toBeLessThan(r0);
  await page.click('#zout'); await page.waitForTimeout(1200); expect(await page.evaluate(() => window.__rig.r)).toBeGreaterThan(r1);
  await page.click('#spin'); const a = await page.evaluate(() => window.__rig.a); await page.waitForTimeout(300);
  expect(await page.evaluate(() => window.__rig.a)).toBeGreaterThan(a);
  await page.click('#tour'); await page.waitForFunction(() => window.__tourDone, null, { timeout: 15000 });
  // التأكد من أن اللوحة ليست فارغة (رسم فعلي)
  const painted = await page.evaluate(() => { const c = document.querySelector('canvas'); const t = document.createElement('canvas'); t.width = 64; t.height = 64; const g = t.getContext('2d'); g.drawImage(c, 0, 0, 64, 64); const d = g.getImageData(0, 0, 64, 64).data; return new Set(Array.from(d).filter((_, i) => i % 4 < 3)).size; });
  expect(painted).toBeGreaterThan(10);
  expect(errors).toEqual([]);
});
