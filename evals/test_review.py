"""pytest view of the same evals: each case runs CLAUSECHECK_EVAL_RUNS times
(default 3) and must pass at least 60% of them; high-risk recall over the set
must be >= 0.9. Runs against whatever CLAUSECHECK_MODE resolves to
(mock when no credentials), so `pytest` is always green-able offline.

    CLAUSECHECK_MODE=live CLAUSECHECK_EVAL_RUNS=5 pytest evals -q
"""
from __future__ import annotations

import os

import pytest

from clausecheck.config import Settings
from clausecheck.playbook import Playbook
from evals.harness import high_risk_recall, load_cases, run_case

RUNS = int(os.getenv("CLAUSECHECK_EVAL_RUNS", "3"))
CASES = load_cases()
_results: dict[str, list[list[str]]] = {}


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="module")
def playbook(settings: Settings) -> Playbook:
    return Playbook.load(settings.playbook_path)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case_pass_rate(case: dict, settings: Settings, playbook: Playbook) -> None:
    runs = [run_case(case, settings, playbook)[0] for _ in range(RUNS)]
    _results[case["id"]] = runs
    passed = sum(1 for r in runs if not r)
    assert passed / RUNS >= 0.6, f"{case['id']}: {passed}/{RUNS} passed; failures: {runs}"


def test_high_risk_recall() -> None:
    if len(_results) < len(CASES):
        pytest.skip("run the full case set first")
    recall = high_risk_recall(_results, CASES)
    assert recall >= 0.9, f"high-risk recall {recall:.2f} < 0.9"
