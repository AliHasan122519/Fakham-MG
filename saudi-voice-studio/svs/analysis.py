"""Text analysis: sentence/paragraph segmentation, genre + emotion detection (rule based, explainable)."""
from __future__ import annotations

import re
from collections import defaultdict

from . import lexicon_ar as L
from . import text_ar

_DIACRITICS = re.compile("[ً-ٰٟـ]")
_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ف", "ب", "ل", "ك")


def norm_token(w: str) -> str:
    w = _DIACRITICS.sub("", w)
    w = re.sub("[إأآٱ]", "ا", w)
    w = w.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return re.sub(r"[^\w؀-ۿ]", "", w)


def _stem_candidates(tok: str):
    yield tok
    for p in _PREFIXES:
        if tok.startswith(p) and len(tok) - len(p) >= 3:
            yield tok[len(p):]


def _build_index():
    idx = defaultdict(set)
    for emo, (_name, cues) in L.EMOTIONS.items():
        for c in cues.split():
            if "_" not in c:
                idx[c].add(emo)
    gidx = defaultdict(set)
    for g, (_n, cues) in L.GENRES.items():
        for c in cues.split():
            if "_" not in c:
                gidx[c].add(g)
    return idx, gidx


_EMO_IDX, _GENRE_IDX = _build_index()
# Weak cues: common words that only weakly signal an emotion; they need company before they win.
_WEAK = {"جميل", "رايع", "عرض", "الان", "قلب", "لن", "جديد", "ليل", "خبره", "جوده", "اليوم", "فرصه", "سلام", "تاريخ",
         "قوه", "ثقه", "اخبار", "اصحاب", "عيد", "نجاح", "ظل", "سر", "هدوء"}
_PHRASE_N = {e: [" ".join(norm_token(x) for x in p.split()) for p in ps] for e, ps in L.PHRASES.items()}
_GENRE_PHRASE = {g: [c.replace("_", " ") for c in cues.split() if "_" in c] for g, (_n, cues) in L.GENRES.items()}
_EMO_PHRASE = {e: [c.replace("_", " ") for c in cues.split() if "_" in c] for e, (_n, cues) in L.EMOTIONS.items()}


def _norm_text(s: str) -> str:
    return " ".join(t for t in (norm_token(w) for w in s.split()) if t)


def split_paragraphs_sentences(text: str) -> list[dict]:
    """Return [{text, paragraph (index), is_last_in_paragraph}] from raw user text (keeps paragraph breaks)."""
    out = []
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    for pi, para in enumerate(paras):
        sents = [s.strip() for s in re.split(r"(?<=[.!?؟؛…])\s+|(?<=[.!?؟])(?=[^\s.!?؟])", para) if s and s.strip()]
        for si, s in enumerate(sents):
            out.append({"text": s, "paragraph": pi, "first_in_paragraph": si == 0, "last_in_paragraph": si == len(sents) - 1})
    return out


def detect_genre(text: str) -> tuple[str, dict]:
    toks = [norm_token(w) for w in text.split()]
    ntext = " ".join(toks)
    score: dict[str, float] = defaultdict(float)
    for t in toks:
        for cand in _stem_candidates(t):
            for g in _GENRE_IDX.get(cand, ()):
                score[g] += 1.0
                break
    for g, ps in _GENRE_PHRASE.items():
        for p in ps:
            if p in ntext:
                score[g] += 1.5
    if re.search(r"[؟?]", text) and len(text.split()) < 14:
        score["conversation"] += 0.3
    if not score:
        return "narration", {}
    best = max(score, key=score.get)
    return (best if score[best] >= 1.0 else "narration"), dict(score)


def detect_emotion(sentence: str, *, genre: str = "narration", position: float = 0.5) -> dict:
    """Score a sentence. Returns {emotion, intensity (0..1), scores, cues, sentence_type}."""
    toks = [norm_token(w) for w in sentence.split()]
    ntext = " ".join(t for t in toks if t)
    scores: dict[str, float] = defaultdict(float)
    cues: dict[str, list[str]] = defaultdict(list)
    for orig, t in zip(sentence.split(), toks):
        for cand in _stem_candidates(t):
            hit = _EMO_IDX.get(cand)
            if hit:
                w = 0.5 if cand in _WEAK else 1.0
                for e in hit:
                    scores[e] += w
                    if orig not in cues[e]:
                        cues[e].append(orig)
                break
    for e, ps in {**{k: v for k, v in _PHRASE_N.items()}}.items():
        for p in ps + _EMO_PHRASE.get(e, []):
            if p and p in ntext:
                scores[e] += 1.5
                if p not in cues[e]:
                    cues[e].append(p)
    for e, w in L.GENRE_PRIOR.get(genre, {}).items():
        if e in L.EMOTIONS:
            scores[e] += w
    # punctuation / sentence type
    stype = "statement"
    if re.search(r"[!！]", sentence):
        stype = "exclamation"
        scores["excitement"] += 0.8
        scores["joy"] += 0.3
        scores["warning"] += 0.2
    if re.search(r"[؟?]", sentence):
        stype = "question"
        scores["suspense"] += 0.3
    if sentence.rstrip().endswith(("...", "…")):
        stype = "trailing"
        scores["suspense"] += 0.5
        scores["sadness"] += 0.3
    n_words = max(1, len(sentence.split()))
    if n_words <= 4:
        scores["confidence"] += 0.2   # short declaratives read as assertive
    if position >= 0.9 and genre in ("ad", "announcement"):
        scores["confidence"] += 0.6   # closing call-to-action
    if not scores:
        return {"emotion": "neutral", "intensity": 0.3, "scores": {}, "cues": {}, "sentence_type": stype}
    emo = max(scores, key=scores.get)
    top = scores[emo]
    second = sorted(scores.values())[-2] if len(scores) > 1 else 0.0
    intensity = max(0.25, min(1.0, 0.3 + 0.28 * top - 0.1 * second))
    if top < 0.6:
        emo, intensity = "neutral", 0.3
    return {"emotion": emo, "intensity": round(intensity, 2), "scores": {k: round(v, 2) for k, v in scores.items()},
            "cues": {k: v for k, v in cues.items() if k == emo}, "sentence_type": stype}


def analyze(text: str, *, use_semantic: bool = True) -> dict:
    """Full analysis of raw text → genre + per-sentence emotion proposals.

    With the semantic model available the decision is MEANING-based: 0.7 × embedding-similarity probability + 0.3 ×
    lexical evidence (+ neighbour smoothing for emotional continuity). Without it, the lexical method alone is used."""
    text = text_ar.normalize(text, numbers=False)
    sents = split_paragraphs_sentences(text)
    genre, gscores = detect_genre(text)
    n = len(sents)
    for i, s in enumerate(sents):
        s.update(detect_emotion(s["text"], genre=genre, position=(i / max(1, n - 1)) if n > 1 else 0.5))
        s["method"] = "lexical"
    method = "lexical"
    if use_semantic:
        try:
            from . import semantic
            if semantic.available():
                sem = semantic.emotion_scores([s["text"] for s in sents])
                for s, r in zip(sents, sem):
                    probs = r["probs"]
                    lex = s["scores"]
                    comb = {e: 0.7 * probs.get(e, 0.0) + 0.3 * min(1.0, lex.get(e, 0.0) / 2.5) for e in probs}
                    tot = sum(comb.values()) or 1.0
                    top = max(comb, key=comb.get)
                    conf = comb[top] / tot
                    if s["sentence_type"] == "exclamation" and top == "neutral":
                        top = "excitement" if probs.get("excitement", 0) >= probs.get("joy", 0) else "joy"
                    inten = max(0.25, min(1.0, 0.25 + 0.9 * conf))
                    if s["sentence_type"] == "exclamation":
                        inten = min(1.0, inten + 0.1)
                    if top == "neutral":
                        inten = 0.3
                    s.update(emotion=top, intensity=round(inten, 2), method="semantic+lexical",
                             probs={k: round(v, 3) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:3]},
                             nearest=r["nearest"].get(top), cues={top: s["cues"].get(top, [])} if s["cues"] else {})
                method = "semantic+lexical"
                gsem = semantic.genre_scores([s["text"] for s in sents])
                gbest = max(gsem, key=gsem.get)
                srt = sorted(gsem.values())
                if gscores.get(genre, 0) < 2.0 and srt[-1] - srt[-2] > 0.008:
                    genre = gbest
                gscores = {**gscores, **{"sem:" + k: round(v, 3) for k, v in gsem.items()}}
        except Exception:                      # model missing / offline: keep the lexical result
            method = "lexical"
    return {"genre": genre, "genre_name": L.GENRES.get(genre, ("سرد", ""))[0], "genre_scores": gscores,
            "style": L.GENRE_STYLE.get(genre, "narration"), "sentences": sents, "method": method}
