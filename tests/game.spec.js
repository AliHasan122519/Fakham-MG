import { test, expect } from '@playwright/test';
const answers = { '2 + 3': '5', '4 + 4': '8', '3 × 4': '12', '5 × 5': '25', '12 ÷ 3': '4', '9 × 9': '81' };
async function solve(page, correct) {
  const q = (await page.textContent('#q')).replace(' = ؟', '');
  const target = correct ? answers[q] : (await page.locator('#answers button').allTextContents()).find(t => t !== answers[q]);
  await page.click(`#answers button[data-v="${target}"]`); await page.waitForTimeout(600);
}
test('اللعبة: الفوز عبر 3 مراحل', async ({ page }) => {
  await page.goto('/games/quiz/'); await page.click('#play');
  for (let i = 0; i < 6; i++) { if (i === 2) await expect(page.locator('#level')).toHaveText('المرحلة 2'); await solve(page, true); }
  await expect(page.locator('#result')).toContainText('فزت'); await expect(page.locator('#final')).toContainText('60');
  await page.click('#retry'); await expect(page.locator('#score')).toHaveText('النقاط: 0');
});
test('اللعبة: الخسارة بعد 3 أخطاء', async ({ page }) => {
  await page.goto('/games/quiz/'); await page.click('#play');
  for (let i = 0; i < 3; i++) await solve(page, false);
  await expect(page.locator('#result')).toContainText('انتهت');
});
