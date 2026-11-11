// يولّد مؤثرات صوتية WAV محلياً (مجاني، بدون أي خدمة خارجية)
import fs from 'node:fs';
const SR = 22050;
function wav(samples) {
  const b = Buffer.alloc(44 + samples.length * 2);
  b.write('RIFF', 0); b.writeUInt32LE(36 + samples.length * 2, 4); b.write('WAVEfmt ', 8);
  b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22); b.writeUInt32LE(SR, 24);
  b.writeUInt32LE(SR * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34); b.write('data', 36);
  b.writeUInt32LE(samples.length * 2, 40);
  samples.forEach((s, i) => b.writeInt16LE(Math.max(-1, Math.min(1, s)) * 32767 * 0.6 | 0, 44 + i * 2));
  return b;
}
function tone(notes, dur, { noise = 0, sweep = 0 } = {}) {
  const n = Math.floor(SR * dur), out = new Array(n).fill(0);
  notes.forEach((f, k) => {
    const start = Math.floor((n / notes.length) * k);
    for (let i = start; i < n; i++) {
      const t = (i - start) / SR, env = Math.exp(-t * 6);
      out[i] += env * Math.sin(2 * Math.PI * (f + sweep * t) * t) / notes.length * 1.6
              + noise * (Math.random() * 2 - 1) * env;
    }
  });
  return out;
}
const sounds = {
  click:   tone([880], 0.12),
  correct: tone([523, 659, 784, 1047], 0.6),
  wrong:   tone([220, 165], 0.45),
  whoosh:  tone([300], 0.5, { noise: 0.5, sweep: 900 }),
  levelup: tone([392, 523, 659, 784, 1047], 0.9),
};
fs.mkdirSync('assets/sfx', { recursive: true });
for (const [k, v] of Object.entries(sounds)) fs.writeFileSync(`assets/sfx/${k}.wav`, wav(v));
console.log('generated', Object.keys(sounds).join(', '));
