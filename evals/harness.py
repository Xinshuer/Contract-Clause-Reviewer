"""Shared eval harness: build a case's contract, run review_clause, check assertions.

The same review_clause() is exercised by pytest, by run_evals.py and by the
promptfoo provider, so "one piece of code, tested from both sides" holds.

Assertions are layered (cheap first):
  deterministic - parses into ClauseVerdict (implicit), verdict in allowed set,
                  topic / risk match, anchor located for redlines,
                  trajectory: required tool calls happened;
  statistical   - pass rate over N runs (run_evals.py / test_review.py);
  model-graded  - rationale quality, via promptfoo llm-rubric.
"""
from __future__ import annotations

import json
from pathlib import Path

from clausecheck.config import Settings
from clausecheck.models import Clause, ClauseReview, Contract
from clausecheck.playbook import Playbook
from clausecheck.review import review_clause
from clausecheck.split import split_clauses

CASES_PATH = Path(__file__).with_name("cases.json")


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]


def case_by_id(case_id: str) -> dict:
    return next(c for c in load_cases() if c["id"] == case_id)


def build_contract(case: dict) -> tuple[Contract, Clause]:
    """Assemble a minimal contract from the case clause + context, through the real splitter."""
    parts = [case["clause"], *case.get("context", [])]
    parts.sort(key=lambda c: tuple(int(x) for x in c["clause_id"].split(".")))
    text = "".join(f"{c['clause_id']} {c['heading']}\n{c['text']}\n\n" for c in parts)
    contract = Contract(name=f"case-{case['id']}", text=text, clauses=split_clauses(text))
    clause = contract.get(case["clause"]["clause_id"])
    assert clause is not None, f"splitter lost clause {case['clause']['clause_id']} in case {case['id']}"
    return contract, clause


def check(case: dict, review: ClauseReview) -> list[str]:
    """Return a list of failure reasons (empty = pass)."""
    e = case["expect"]
    v = review.verdict
    fails: list[str] = []
    if v.verdict not in e["verdict_in"]:
        fails.append(f"verdict {v.verdict} not in {e['verdict_in']}")
    if "topic" in e and v.topic != e["topic"]:
        fails.append(f"topic {v.topic} != {e['topic']}")
    if "risk_in" in e and v.risk_level not in e["risk_in"]:
        fails.append(f"risk {v.risk_level} not in {e['risk_in']}")
    if v.verdict == "redline":
        if review.redline is None or not review.redline.located:
            fails.append("redline verdict without a located anchor")
    for cid in e.get("must_consult", []):
        if cid not in v.consulted_clauses and f"get_clause:{cid}" not in review.trace.tool_calls:
            fails.append(f"did not consult clause {cid}")
    for prefix in e.get("must_call_prefix", []):
        if not any(t.startswith(prefix) for t in review.trace.tool_calls):
            fails.append(f"tool call starting with {prefix!r} missing (calls: {review.trace.tool_calls})")
    for s in e.get("rationale_must_not_contain", []):
        if s.lower() in v.rationale.lower():
            fails.append(f"rationale repeats injected text {s!r}")
    if review.trace.error:
        fails.append(f"trace error: {review.trace.error}")
    return fails


def run_case(case: dict, settings: Settings | None = None, playbook: Playbook | None = None) -> tuple[list[str], ClauseReview]:
    settings = settings or Settings()
    playbook = playbook or Playbook.load(settings.playbook_path)
    contract, clause = build_contract(case)
    review = review_clause(contract, clause, playbook, settings)
    return check(case, review), review


def high_risk_recall(results: dict[str, list[list[str]]], cases: list[dict]) -> float:
    """Share of must-flag / trap runs whose verdict assertion passed."""
    total = hit = 0
    for case in cases:
        if case["kind"] not in {"must_flag", "trap"}:
            continue
        for fails in results[case["id"]]:
            total += 1
            if not any(f.startswith("verdict") for f in fails):
                hit += 1
    return hit / total if total else 1.0
