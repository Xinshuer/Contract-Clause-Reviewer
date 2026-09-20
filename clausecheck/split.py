"""Deterministic clause splitting. No model involved - this is a regex.

Each numbered heading ("8.1 Cap on Liability") starts a clause. The clause
itself is the chunk (structure-aware chunking): it carries its id, a
breadcrumb path, and the ids of other clauses it cross-references, so the
reviewer can pull in "subject to Section 12" before judging.
"""
from __future__ import annotations

import re
from pathlib import Path

from .models import Clause, Contract
from .parse import extract_text

CLAUSE_HEAD = re.compile(r"^(?P<id>\d+(?:\.\d+)*)\.?\s+(?P<heading>[A-Z][^\n]{0,80})$", re.M)
XREF = re.compile(r"\b(?:Section|Clause|Article)s?\s+(\d+(?:\.\d+)*)", re.I)


def split_clauses(full_text: str) -> list[Clause]:
    heads = list(CLAUSE_HEAD.finditer(full_text))
    titles: dict[str, str] = {}
    out: list[Clause] = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(full_text)
        cid = h.group("id")
        heading = h.group("heading").strip()
        body = full_text[h.end():end].strip()
        titles[cid] = heading
        parent = cid.rsplit(".", 1)[0] if "." in cid else None
        path = f"{cid} {heading}"
        if parent and parent in titles:
            path = f"{parent} {titles[parent]} > {path}"
        refs = sorted({r for r in XREF.findall(body) if r != cid and r != parent}, key=_sort_key)
        out.append(
            Clause(
                clause_id=cid,
                heading=heading,
                text=body,
                section_path=path,
                refs=refs,
                char_start=h.end(),
                char_end=end,
            )
        )
    return out


def _sort_key(cid: str) -> tuple[int, ...]:
    return tuple(int(x) for x in cid.split("."))


def load_contract(path: str | Path, name: str | None = None) -> Contract:
    path = Path(path)
    text = extract_text(path)
    clauses = split_clauses(text)
    if not clauses:
        raise ValueError(
            f"No numbered clauses found in {path.name}. "
            "Clause Check expects headings like '8.1 Cap on Liability' on their own line."
        )
    return Contract(name=name or path.stem, text=text, clauses=clauses)


def contract_from_text(text: str, name: str = "contract") -> Contract:
    from .parse import _normalise

    text = _normalise(text)
    return Contract(name=name, text=text, clauses=split_clauses(text))
