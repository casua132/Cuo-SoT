"""Extract the selected candidate identifier from a model response."""

from __future__ import annotations

import re

# (a), (b), ... with a few common bracketing variants.
_PARENTHESIZED_LETTER_RE = re.compile(r"[\(（]\s*([a-dA-D])\s*[\)）]")
# A bare letter appearing after an answer cue, e.g. "the answer is c".
_CUE_RE = re.compile(r"\b(?:answer|response|identifier|option|choice|selected)\b[^a-dA-D]*?[\(（]?\s*([a-dA-D])\b")
# Any single-letter token (last-resort fallback).
_BARE_LETTER_RE = re.compile(r"\b([a-dA-D])\b")


def normalize_answer(raw: str | None) -> str | None:
    """Normalize the CSV ``correct_answer`` field, e.g. ``'(c)'`` -> ``'c'``."""
    if not raw:
        return None
    s = str(raw).strip().strip("\"'` ")
    m = _PARENTHESIZED_LETTER_RE.search(s)
    if m:
        return m.group(1).lower()
    m = _BARE_LETTER_RE.search(s)
    return m.group(1).lower() if m else None


def extract_answer(response: str | None) -> str | None:
    """Extract the selected candidate identifier (``a``/``b``/``c``/``d``) from a model response.

    Resolution order:
      1. after an explicit ``identifier:``-style label (the cot output format);
      2. the first parenthesized option letter, e.g. ``(c)`` (the cot_opt output format);
      3. a bare letter after an answer cue;
      4. the last single-letter token (fallback).
    """
    if not response:
        return None
    lowered = response.lower().strip()

    label = re.search(r"identifier\s*[:\-]?\s*[\(（]?\s*([a-dA-D])", lowered)
    if label:
        return label.group(1).lower()

    m = _PARENTHESIZED_LETTER_RE.search(lowered)
    if m:
        return m.group(1).lower()

    m = _CUE_RE.search(lowered)
    if m:
        return m.group(1).lower()

    letters = _BARE_LETTER_RE.findall(lowered)
    if letters:
        return letters[-1].lower()
    return None
