"""Anchor location and diff generation. The model proposes; code edits.

The model outputs {anchor_text, replacement}; we find the anchor in the
clause. Exact match first, then fuzzy within the clause only (ratio >= 95),
otherwise the redline is downgraded to "needs manual placement" - never guess.
"""
from __future__ import annotations

import difflib
import re

from rapidfuzz import fuzz, process

FUZZY_CUTOFF = 95


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;:])\s+(?=[A-Z\"(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def locate_anchor(clause_text: str, anchor: str) -> tuple[int, int] | None:
    anchor = anchor.strip()
    if not anchor:
        return None
    i = clause_text.find(anchor)
    if i >= 0:
        return i, i + len(anchor)
    # whitespace-insensitive exact match (models often collapse line breaks)
    pattern = r"\s+".join(re.escape(tok) for tok in anchor.split())
    m = re.search(pattern, clause_text)
    if m:
        return m.start(), m.end()
    sents = split_sentences(clause_text)
    if not sents:
        return None
    best = process.extractOne(anchor, sents, scorer=fuzz.ratio, score_cutoff=FUZZY_CUTOFF)
    if best is None:
        return None
    s = clause_text.find(best[0])
    return (s, s + len(best[0])) if s >= 0 else None


def make_diff(clause_text: str, start: int, end: int, replacement: str, clause_id: str = "") -> str:
    after = clause_text[:start] + replacement + clause_text[end:]
    lines = difflib.unified_diff(
        clause_text.splitlines(),
        after.splitlines(),
        fromfile=f"clause {clause_id} (original)",
        tofile=f"clause {clause_id} (proposed)",
        lineterm="",
        n=1,
    )
    return "\n".join(lines)


def apply_replacement(clause_text: str, start: int, end: int, replacement: str) -> str:
    return clause_text[:start] + replacement + clause_text[end:]
