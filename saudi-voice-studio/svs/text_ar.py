"""معالجة النص العربي قبل التوليد: تنظيف، أرقام إلى كلمات، تقسيم إلى جمل.

No external dependencies. Conservative on purpose: we never rewrite words,
only normalise symbols/digits/whitespace so the character-level TTS model
sees clean input.
"""
from __future__ import annotations

import re

_TATWEEL = "ـ"
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_PUNCT_MAP = str.maketrans({"،": ",", "؛": ";", "؟": "?", "“": '"', "”": '"', "‘": "'", "’": "'", "…": "..."})
# Zero-width / bidi control characters that confuse tokenisers.
_INVISIBLE = re.compile("[​‌‍‎‏‪-‮⁦-⁩﻿]")

_ONES = ["", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة", "عشرة",
         "أحد عشر", "اثنا عشر", "ثلاثة عشر", "أربعة عشر", "خمسة عشر", "ستة عشر", "سبعة عشر",
         "ثمانية عشر", "تسعة عشر"]
_TENS = ["", "", "عشرون", "ثلاثون", "أربعون", "خمسون", "ستون", "سبعون", "ثمانون", "تسعون"]
_HUNDREDS = ["", "مائة", "مائتان", "ثلاثمائة", "أربعمائة", "خمسمائة", "ستمائة", "سبعمائة", "ثمانمائة", "تسعمائة"]
_SCALES = [(10**9, "مليار", "ملياران", "مليارات"), (10**6, "مليون", "مليونان", "ملايين"),
           (1000, "ألف", "ألفان", "آلاف")]


def _below_1000(n: int) -> str:
    parts = []
    h, r = divmod(n, 100)
    if h:
        parts.append(_HUNDREDS[h])
    if r:
        if r < 20:
            parts.append(_ONES[r])
        else:
            t, o = divmod(r, 10)
            parts.append((_ONES[o] + " و" if o else "") + _TENS[t])
    return " و".join(parts)


def number_to_words(n: int) -> str:
    """Cardinal number to Modern Standard Arabic words (masculine, simple form)."""
    if n == 0:
        return "صفر"
    if n < 0:
        return "سالب " + number_to_words(-n)
    parts = []
    for value, one, two, many in _SCALES:
        q, n = divmod(n, value)
        if not q:
            continue
        if q == 1:
            parts.append(one)
        elif q == 2:
            parts.append(two)
        elif q <= 10:
            parts.append(f"{_below_1000(q)} {many}")
        else:
            parts.append(f"{_below_1000(q)} {one}")
    if n:
        parts.append(_below_1000(n))
    return " و".join(parts)


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _replace_number(m: re.Match) -> str:
    s = m.group(0)
    try:
        if "." in s:
            whole, frac = s.split(".")
            return f"{number_to_words(int(whole))} فاصلة {' '.join(number_to_words(int(d)) for d in frac)}"
        return number_to_words(int(s))
    except (ValueError, RecursionError):
        return s


_SYMBOLS = {"%": " بالمائة ", "&": " و ", "+": " زائد ", "=": " يساوي ", "@": " آت ", "#": " "}


def normalize(text: str, numbers: bool = True) -> str:
    text = _INVISIBLE.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = text.translate(_ARABIC_INDIC)
    for k, v in _SYMBOLS.items():
        text = text.replace(k, v)
    if numbers:
        text = _NUM_RE.sub(_replace_number, text)
    text = text.translate(_PUNCT_MAP)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[\*\_\~\`\[\]\{\}\<\>\|\\/]+", " ", text)
    text = re.sub(r"([!?.,;:])\1{1,}", r"\1", text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


_SENT_SPLIT = re.compile(r"(?<=[.!?;؟\n])\s+")


def split_chunks(text: str, max_chars: int = 160) -> list[str]:
    """Split into sentence-aligned chunks no longer than ~max_chars."""
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        while len(s) > max_chars:  # hard-wrap very long sentences at a comma/space
            cut = max(s.rfind(",", 0, max_chars), s.rfind(" ", 0, max_chars))
            cut = cut if cut > 20 else max_chars
            piece, s = s[:cut].strip(" ,"), s[cut:].strip(" ,")
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(piece)
        if cur and len(cur) + 1 + len(s) > max_chars:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def has_arabic(text: str) -> bool:
    return bool(re.search(r"[؀-ۿ]", text))
