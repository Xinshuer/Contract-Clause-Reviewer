"""The fixed workflow: parse -> split -> review each clause -> summarise.

Everything here is deterministic Python; the only model call is inside
review_clause(). The summary is code (sorting and counting), not a model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .config import Settings
from .models import ClauseReview, Contract, ReviewReport
from .playbook import Playbook
from .review import review_clause
from .split import load_contract
from .tracing import annotate, observe

RISK_ORDER = {"high": 0, "medium": 1, "low": 2}
VERDICT_ORDER = {"redline": 0, "flag": 1, "accept": 2}


@observe(name="review_contract")
def review_contract(
    source: str | Path | Contract,
    settings: Settings | None = None,
    playbook: Playbook | None = None,
    progress: Callable[[int, int, ClauseReview], None] | None = None,
) -> ReviewReport:
    settings = settings or Settings()
    playbook = playbook or Playbook.load(settings.playbook_path)
    contract = source if isinstance(source, Contract) else load_contract(source)
    clauses = contract.reviewable
    reviews: list[ClauseReview] = []
    for i, clause in enumerate(clauses, 1):
        review = review_clause(contract, clause, playbook, settings)
        reviews.append(review)
        if progress:
            progress(i, len(clauses), review)
    summary = summarise(reviews, settings)
    annotate(contract=contract.name, clauses=len(clauses), **{k: v for k, v in summary.items() if not isinstance(v, (list, dict))})
    return ReviewReport(contract_name=contract.name, model=settings.model, mode=settings.mode, summary=summary, reviews=reviews)


def summarise(reviews: list[ClauseReview], settings: Settings) -> dict:
    counts = {"accept": 0, "flag": 0, "redline": 0}
    for r in reviews:
        counts[r.verdict.verdict] += 1
    tokens = {
        "input": sum(r.trace.input_tokens for r in reviews),
        "output": sum(r.trace.output_tokens for r in reviews),
        "cache_read": sum(r.trace.cache_read_input_tokens for r in reviews),
        "cache_write": sum(r.trace.cache_creation_input_tokens for r in reviews),
    }
    p = settings.price()
    cost = (
        tokens["input"] * p["input"]
        + tokens["output"] * p["output"]
        + tokens["cache_read"] * p["cache_read"]
        + tokens["cache_write"] * p["cache_write"]
    ) / 1_000_000
    attention = sorted(
        (r for r in reviews if r.verdict.verdict != "accept"),
        key=lambda r: (RISK_ORDER[r.verdict.risk_level], VERDICT_ORDER[r.verdict.verdict], r.clause.clause_id),
    )
    return {
        "clauses_reviewed": len(reviews),
        "counts": counts,
        "needs_attention": [r.clause.clause_id for r in attention],
        "high_risk": [r.clause.clause_id for r in reviews if r.verdict.risk_level == "high"],
        "low_confidence": [r.clause.clause_id for r in reviews if r.verdict.confidence == "low"],
        "clauses_skipped_by_reader": round(counts["accept"] / max(len(reviews), 1), 2),
        "tokens": tokens,
        "estimated_cost_usd": round(cost, 4),
        "cache_hit_ratio": round(tokens["cache_read"] / max(tokens["input"] + tokens["cache_read"] + tokens["cache_write"], 1), 2),
        "total_latency_ms": sum(r.trace.latency_ms for r in reviews),
        "errors": [r.clause.clause_id for r in reviews if r.trace.error],
    }
