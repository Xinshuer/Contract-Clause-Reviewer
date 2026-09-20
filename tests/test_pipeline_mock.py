"""End-to-end run of the deterministic pipeline in mock mode (no API)."""
from pathlib import Path

from clausecheck.config import Settings
from clausecheck.models import ReviewReport
from clausecheck.pipeline import review_contract

ROOT = Path(__file__).resolve().parent.parent


def test_mock_review_of_sample_contract():
    report = review_contract(ROOT / "data" / "sample_contract.txt", Settings(mode="mock"))
    assert isinstance(report, ReviewReport)
    by_id = {r.clause.clause_id: r for r in report.reviews}
    # section titles are not reviewed
    assert "8" not in by_id and "8.1" in by_id
    # the cross-reference trap: 8.1 is only bad because of 12
    assert by_id["8.1"].verdict.verdict != "accept"
    assert "12" in by_id["8.1"].verdict.consulted_clauses
    # obvious ones
    assert by_id["4.2"].verdict.verdict == "redline" and by_id["4.2"].redline.located
    assert by_id["10.1"].verdict.topic == "data_processing"
    assert by_id["14"].verdict.verdict == "accept"
    assert by_id["2.2"].verdict.verdict == "accept"
    s = report.summary
    assert s["clauses_reviewed"] == len(report.reviews)
    assert set(s["counts"]) == {"accept", "flag", "redline"}
    assert s["estimated_cost_usd"] == 0
    # report round-trips through JSON
    ReviewReport.model_validate_json(report.model_dump_json())
