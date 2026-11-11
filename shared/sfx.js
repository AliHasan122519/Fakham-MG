// يتطلب تحميل howler.min.js قبله. window.SFX.play('click')
window.SFX = (() => {
  const names = ['click', 'correct', 'wrong', 'whoosh', 'levelup'];
  const h = {};
  names.forEach(n => (h[n] = new Howl({ src: [`/assets/sfx/${n}.wav`], volume: 0.7 })));
  return { play(n) { window.__sfxLog = (window.__sfxLog || []).concat(n); h[n] && h[n].play(); }, muted(m) { Howler.mute(m); } };
})();
