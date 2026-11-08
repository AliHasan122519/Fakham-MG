"""Meaning-based analysis (not keyword matching): sentence embeddings + editable prototype sentences.

Model: intfloat/multilingual-e5-small (MIT, 118M parameters, Arabic supported, runs on CPU in well under a second per
sentence). Everything degrades gracefully: if the model/transformers is unavailable, callers fall back to the lexical
analysis in analysis.py.

Three uses:
  1. emotion  — cosine similarity of each sentence to prototype sentences per emotion (+ context smoothing);
  2. genre    — document-level, same mechanism;
  3. word sense — pick the reading (with diacritics) of ambiguous words (علم/ذهب/كتب/شعر…) from the sentence context.
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np

from . import config

MODEL_ID = "intfloat/multilingual-e5-small"
PROTOS = config.ROOT / "voices" / "semantic_prototypes.json"
_LOCK = threading.Lock()
_STATE: dict = {}


def available() -> bool:
    if os.environ.get("SVS_SEMANTIC", "auto") == "off":
        return False
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _load():
    with _LOCK:
        if "model" not in _STATE:
            import torch
            from transformers import AutoModel, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(config.MODEL_DIR / "hf"))
            mdl = AutoModel.from_pretrained(MODEL_ID, cache_dir=str(config.MODEL_DIR / "hf")).eval()
            _STATE.update(tok=tok, model=mdl, torch=torch)
    return _STATE


def embed(texts: list[str], *, passage: bool = False) -> np.ndarray:
    st = _load()
    torch = st["torch"]
    prefix = "passage: " if passage else "query: "
    out = []
    for i in range(0, len(texts), 32):
        batch = st["tok"]([prefix + t for t in texts[i:i + 32]], padding=True, truncation=True, max_length=128,
                          return_tensors="pt")
        with torch.no_grad():
            h = st["model"](**batch).last_hidden_state
        m = batch["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1)
        out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(out)


def _protos() -> dict:
    if "protos" not in _STATE:
        d = json.loads(PROTOS.read_text(encoding="utf-8"))
        flat, idx = [], []
        for emo, ss in d["emotions"].items():
            for s in ss:
                flat.append(s)
                idx.append(emo)
        E = embed(flat)
        genres, gidx, gflat = {}, [], []
        for g, ss in d["genres"].items():
            for s in ss:
                gflat.append(s)
                gidx.append(g)
        G = embed(gflat)
        _STATE["protos"] = {"E": E, "emo_idx": idx, "emo_text": flat, "G": G, "g_idx": gidx, "senses": d["senses"]}
    return _STATE["protos"]


def emotion_scores(sentences: list[str], *, context: float = 0.15, temp: float = 0.05) -> list[dict]:
    """Per-sentence probability over emotions from meaning. `context` blends neighbours (emotional arc continuity)."""
    P = _protos()
    S = embed(sentences)
    emos = sorted(set(P["emo_idx"]))
    raw = np.zeros((len(sentences), len(emos)))
    nearest = [[None] * len(emos) for _ in sentences]
    for j, e in enumerate(emos):
        cols = [k for k, x in enumerate(P["emo_idx"]) if x == e]
        sims = S @ P["E"][cols].T                       # (n, protos)
        top = np.sort(sims, axis=1)[:, -3:].mean(axis=1)  # mean of top-3 neighbours
        raw[:, j] = top
        for i in range(len(sentences)):
            nearest[i][j] = P["emo_text"][cols[int(np.argmax(sims[i]))]]
    if len(sentences) > 1 and context > 0:
        sm = raw.copy()
        sm[1:] += context * raw[:-1]
        sm[:-1] += context * raw[1:]
        raw = sm / (1 + context * np.where((np.arange(len(sentences)) == 0) | (np.arange(len(sentences)) == len(sentences) - 1), 1, 2))[:, None]
    z = (raw - raw.max(axis=1, keepdims=True)) / temp
    prob = np.exp(z)
    prob /= prob.sum(axis=1, keepdims=True)
    return [{"probs": {e: float(prob[i, j]) for j, e in enumerate(emos)},
             "nearest": {e: nearest[i][j] for j, e in enumerate(emos)}} for i in range(len(sentences))]


def genre_scores(text_chunks: list[str]) -> dict[str, float]:
    P = _protos()
    v = embed([" ".join(text_chunks)[:900]])[0]
    sims = P["G"] @ v
    out: dict[str, float] = {}
    for g in set(P["g_idx"]):
        out[g] = float(np.mean([sims[k] for k, x in enumerate(P["g_idx"]) if x == g]))
    return out


def word_sense(sentence: str, stem: str) -> dict | None:
    """Choose the reading of an ambiguous word from sentence context. Returns {say, gloss?, margin, scores}."""
    P = _protos()
    senses = P["senses"].get(stem)
    if not senses:
        return None
    v = embed([sentence])[0]
    scores = {}
    for reading, ex in senses.items():
        E = embed(ex, passage=False)
        scores[reading] = float(np.sort(E @ v)[-2:].mean())
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 1.0
    return {"say": ranked[0][0], "margin": round(margin, 4), "scores": {k: round(x, 4) for k, x in scores.items()}}
