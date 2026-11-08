"""Arabic pronunciation layer: normalisation → editable dictionary → spoken text, plus ambiguity flags.

Honest scope: no automatic full diacritisation. Pronunciation control = (1) user-editable dictionary,
(2) per-word diacritic choices for the common ambiguous words, (3) a spoken-text editor in the UI.
The model reads the *spoken* text exactly as written, so what you type there is what it says.
"""
from __future__ import annotations

import json
import re

from . import config, text_ar
from .analysis import _stem_candidates, norm_token
from .lexicon_ar import AMBIGUOUS

_DEFAULT = config.ROOT / "voices" / "lexicon_default.json"
USER_LEXICON = config.DATA / "lexicon.json"
MAX_ENTRIES = 500
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]*")
_LETTERS = {"A": "إيه", "B": "بي", "C": "سي", "D": "دي", "E": "إي", "F": "إف", "G": "جي", "H": "إتش", "I": "آي", "J": "جي",
            "K": "كي", "L": "إل", "M": "إم", "N": "إن", "O": "أو", "P": "بي", "Q": "كيو", "R": "آر", "S": "إس", "T": "تي",
            "U": "يو", "V": "في", "W": "دبليو", "X": "إكس", "Y": "واي", "Z": "زد"}


def _load(path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def lexicon() -> dict:
    base = _load(_DEFAULT)
    user = _load(USER_LEXICON)
    entries = {e["word"]: e["say"] for e in base.get("entries", [])}
    entries.update({e["word"]: e["say"] for e in user.get("entries", []) if e.get("word") and e.get("say")})
    return {"entries": entries, "units": base.get("units", {}), "dialect_light": base.get("dialect_light", {}),
            "user_entries": user.get("entries", [])}


def save_user_entries(entries: list[dict]) -> int:
    clean = []
    for e in entries[:MAX_ENTRIES]:
        w, s = str(e.get("word", "")).strip(), str(e.get("say", "")).strip()
        if w and s and len(w) <= 60 and len(s) <= 200:
            clean.append({"word": w, "say": s})
    USER_LEXICON.parent.mkdir(parents=True, exist_ok=True)
    USER_LEXICON.write_text(json.dumps({"entries": clean}, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(clean)


def _apply_words(text: str, mapping: dict, changes: list, kind: str) -> str:
    for word in sorted(mapping, key=len, reverse=True):          # longest first
        say = mapping[word]
        pat = re.compile(r"(?<![\w؀-ۿ])" + re.escape(word) + r"(?![\w؀-ۿ])", re.I if word.isascii() else 0)
        if pat.search(text):
            text = pat.sub(say, text)
            changes.append({"from": word, "to": say, "kind": kind})
    return text


def to_spoken(text: str, dialect: str = "najdi", *, light_dialect: bool = False) -> dict:
    """Return {spoken, changes[], flags{ambiguous[], unknown_latin[]}} for one sentence."""
    lex = lexicon()
    changes: list[dict] = []
    t = text
    # units next to numbers: "5 كم" -> "5 كيلومتر" (before digits become words)
    for u, say in sorted(lex["units"].items(), key=lambda kv: -len(kv[0])):
        t2 = re.sub(r"(\d)\s*" + re.escape(u) + r"(?![\w؀-ۿ])", lambda m: f"{m.group(1)} {say}", t)
        if t2 != t:
            changes.append({"from": u, "to": say, "kind": "unit"})
            t = t2
    t = t.replace("%", " في المئة ")
    t = _apply_words(t, lex["entries"], changes, "dictionary")
    if light_dialect:
        t = _apply_words(t, lex["dialect_light"].get(dialect, {}), changes, "dialect")
    t = text_ar.normalize(t, numbers=True)
    unknown = []
    def _latin(m):
        w = m.group(0)
        if w.isupper() and 2 <= len(w) <= 5:                      # unknown acronym -> spell the letters
            changes.append({"from": w, "to": " ".join(_LETTERS.get(c, c) for c in w), "kind": "acronym"})
            return " ".join(_LETTERS.get(c, c) for c in w)
        unknown.append(w)
        return w
    t = _LATIN.sub(_latin, t)
    amb = []
    for w in t.split():
        if re.search("[\u064b-\u065f]", w):
            continue                                          # already diacritised by the user
        for cand in _stem_candidates(norm_token(w)):
            if cand in AMBIGUOUS:
                amb.append({"word": w, "stem": cand, "choices": [{"say": s_, "gloss": g} for s_, g in AMBIGUOUS[cand]]})
                break
    return {"spoken": re.sub(r"\s+", " ", t).strip(), "changes": changes,
            "flags": {"ambiguous": amb, "unknown_latin": unknown}}
