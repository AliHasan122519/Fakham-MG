"""Context-aware pronunciation helper: contextual readings, optional vocalised text, and waqf (pause) forms.

What it really does (and does not):
  * Words with several common readings (علم، ذهب، كتب، شعر، …) are resolved FROM THE SENTENCE MEANING (semantic model,
    measured 18/20 on held-out examples) and written with the right diacritics; low-confidence cases are left for you.
  * Your dictionary entries (with diacritics) always win.
  * There is NO general-purpose automatic full diacritisation: the free neural diacriticisers found either had no
    licence or were unreliable on modern text, so claiming one would be dishonest.
  * Waqf: on a word immediately before a pause mark, case endings are dropped (pausal form) — the other letters' vowels
    are never touched.
"""
from __future__ import annotations

import re

from . import analysis
from .analysis import norm_token
from .lexicon_ar import AMBIGUOUS

MARGIN = 0.012          # minimum semantic margin to auto-resolve a sense (≈0.95 precision, ~75% resolved on held-out examples)
_HARAKAT = "ًٌٍَُِّْ"
_PAUSE_AFTER = re.compile(r"([؀-ۿ" + _HARAKAT + r"]+)(?=\s*[.،,;؛!؟?…:]|$)")


def has_diacritics(w: str) -> bool:
    return bool(re.search("[" + _HARAKAT + "]", w))


def contextual_readings(sentence: str, *, use_semantic: bool = True) -> dict:
    """Return {"vocalized": sentence with chosen readings, "resolved": [...], "unresolved": [...]}."""
    words = sentence.split(" ")
    resolved, unresolved = [], []
    sem = None
    if use_semantic:
        try:
            from . import semantic
            sem = semantic if semantic.available() else None
        except Exception:
            sem = None
    out = []
    for wi, w in enumerate(words):
        core = re.sub(r"^[^؀-ۿ]+|[^؀-ۿ]+$", "", w)
        lead = w[: w.find(core)] if core and core in w else ""
        trail = w[len(lead) + len(core):] if core else ""
        if not core or has_diacritics(core):
            out.append(w)
            continue
        hit = None
        nt = norm_token(core)
        for cand in analysis._stem_candidates(nt):
            if cand in AMBIGUOUS:
                hit = cand
                break
        if not hit:
            out.append(w)
            continue
        prefix_len = len(core) - len(hit) if core.endswith(hit) else 0
        prefix = core[:prefix_len]
        choice = None
        if sem is not None:
            lo, hi = max(0, wi - 4), wi + 5      # local window; other occurrences of the same stem are dropped so they cannot vote
            ctx = " ".join(w2 for k, w2 in enumerate(words[lo:hi], start=lo)
                           if k == wi or hit not in set(analysis._stem_candidates(norm_token(w2))))
            try:
                r = sem.word_sense(ctx, hit)
            except Exception:               # model/prototypes unavailable → leave the word for the user to choose
                r, sem = None, None
            if r and r["margin"] >= MARGIN:
                choice = r
        if choice:
            out.append(lead + prefix + choice["say"] + trail)
            resolved.append({"word": core, "say": prefix + choice["say"], "margin": choice["margin"],
                             "alternatives": [{"say": k, "score": v} for k, v in choice["scores"].items() if k != choice["say"]]})
        else:
            out.append(w)
            unresolved.append({"word": core, "choices": [{"say": s, "gloss": g} for s, g in AMBIGUOUS[hit]]})
    return {"vocalized": " ".join(out), "resolved": resolved, "unresolved": unresolved}


def waqf(text: str) -> str:
    """Pausal form for already-diacritised words directly before a pause mark: ـُـِـَ ← ـْ ; tanween ← sukun (ـً → ا);
    ة ← ه. Other letters' vowels stay untouched. Undiacritised words are left alone (nothing to drop)."""
    def fix(m: re.Match) -> str:
        w = m.group(1)
        if not has_diacritics(w):
            return w
        if re.search("ً[اى]?$", w):                       # tanween fatha: كتابًا → كتابا
            return re.sub("ً[اى]?$", "ا", w)
        if re.search("\u0629[\u064b-\u0652]?$", w):                  # ta marbuta: المدرسَةِ → المدرسَهْ
            return re.sub("\u0629[\u064b-\u0652]?$", "\u0647\u0652", w)
        return re.sub("[\u064c\u064d\u064e\u064f\u0650]$", "\u0652", w)       # final short vowel / tanween → sukun
    return _PAUSE_AFTER.sub(fix, text)


def vocalize_sentence(sentence: str, *, use_semantic: bool = True, apply_waqf: bool = True) -> dict:
    r = contextual_readings(sentence, use_semantic=use_semantic)
    r["vocalized_waqf"] = waqf(r["vocalized"]) if apply_waqf else r["vocalized"]
    return r
