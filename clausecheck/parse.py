"""Turn a PDF / text file into one clean string.

The important design decision: pages are re-joined into a single text
*before* clause splitting, so a clause that spans a page break stays whole.
Splitting page-by-page was the "30% of clauses truncated" bug in the story.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


def extract_text(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return _extract_pdf(path)
    return _normalise(path.read_text(encoding="utf-8", errors="replace"))


def _extract_pdf(path: Path) -> str:
    import pdfplumber  # imported lazily: not needed for .txt input

    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    pages = _strip_repeated_lines(pages)
    joined = "\n".join(pages)
    return _normalise(joined)


def _strip_repeated_lines(pages: list[str], min_pages: int = 3, ratio: float = 0.5) -> list[str]:
    """Drop headers/footers: lines that repeat on most pages (page numbers, doc title)."""
    if len(pages) < min_pages:
        return pages
    counts = Counter()
    for p in pages:
        for line in {l.strip() for l in p.splitlines() if l.strip()}:
            counts[re.sub(r"\d+", "#", line)] += 1
    threshold = max(min_pages, int(len(pages) * ratio))
    repeated = {k for k, v in counts.items() if v >= threshold}
    out = []
    for p in pages:
        kept = [l for l in p.splitlines() if re.sub(r"\d+", "#", l.strip()) not in repeated]
        out.append("\n".join(kept))
    return out


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"
