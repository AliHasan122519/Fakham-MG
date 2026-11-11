import { test, expect } from '@playwright/test';
test.use({ hasTouch: true, viewport: { width: 820, height: 1180 } });
test('العرض: جميع الأزرار والتنقل', async ({ page }) => {
  const errors = []; page.on('pageerror', e => errors.push(e.message)); page.on('console', m => m.type() === 'error' && errors.push(m.text()));
  await page.goto('/presentations/demo/'); await page.waitForFunction(() => window.__ready);
  await page.tap('#start'); await expect(page.locator('#s-counter')).toHaveClass(/present/);
  await page.tap('#inc'); await page.tap('#inc'); await expect(page.locator('#count')).toHaveText('2');
  await page.tap('#reset'); await expect(page.locator('#count')).toHaveText('0');
  await page.keyboard.press('ArrowLeft'); // RTL: التالي
  await page.tap('[data-a="36"]'); await expect(page.locator('#feedback')).toContainText('حاول');
  await page.tap('[data-a="42"]'); await expect(page.locator('#feedback')).toContainText('صحيحة');
  await page.keyboard.press('ArrowLeft'); await page.tap('#again');
  await expect(page.locator('#s-intro')).toHaveClass(/present/);
  const log = await page.evaluate(() => window.__sfxLog); expect(log).toEqual(expect.arrayContaining(['click', 'correct', 'wrong', 'whoosh']));
  await page.tap('#mute'); await expect(page.locator('#mute')).toHaveAttribute('aria-pressed', 'true');
  expect(errors).toEqual([]);
});
