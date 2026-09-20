import json
from pathlib import Path

from clausecheck.playbook import Playbook
from clausecheck.split import load_contract
from clausecheck.tools import ReviewContext, tool_specs

ROOT = Path(__file__).resolve().parent.parent


def _ctx():
    contract = load_contract(ROOT / "data" / "sample_contract.txt")
    playbook = Playbook.load(ROOT / "data" / "playbook.json")
    return ReviewContext(contract, playbook, "8.1")


def test_playbook_loads_and_lookup_is_trimmed():
    pb = Playbook.load(ROOT / "data" / "playbook.json")
    assert len(pb.rules) >= 8
    res = pb.lookup("liability_cap")
    assert res["rules"] and "red_flags" not in res["rules"][0] and "keywords" not in res["rules"][0]
    assert pb.lookup("other")["rules"] == []


def test_tool_specs_are_generated_from_pydantic():
    specs = tool_specs()
    names = [s["name"] for s in specs]
    assert names == ["lookup_playbook", "get_clause", "propose_redline", "mark_for_review"]
    lookup = specs[0]
    assert "Do NOT" in lookup["description"]
    assert lookup["input_schema"]["properties"]["topic"]["enum"]


def test_invalid_args_are_returned_not_raised():
    ctx = _ctx()
    content, is_error = ctx.execute("lookup_playbook", {"topic": "HIGH"})
    assert is_error and content.startswith("INVALID_ARGS")
    content, is_error = ctx.execute("no_such_tool", {})
    assert is_error and "UNKNOWN_TOOL" in content


def test_get_clause_projection_and_not_found():
    ctx = _ctx()
    content, is_error = ctx.execute("get_clause", {"clause_id": "12"})
    data = json.loads(content)
    assert not is_error and data["clause_id"] == "12" and "shall not apply" in data["text"]
    assert ctx.consulted == ["12"] and ctx.tool_calls == ["get_clause:12"]
    content, is_error = ctx.execute("get_clause", {"clause_id": "42"})
    assert is_error and json.loads(content)["error"] == "NOT_FOUND"


def test_propose_redline_locates_or_reports():
    ctx = _ctx()
    ok, err = ctx.execute("propose_redline", {
        "clause_id": "8.1",
        "anchor_text": "Subject to Section 12, each party's aggregate liability",
        "replacement": "Each party's aggregate liability",
        "rationale": "Remove the one-sided carve-out reference.",
    })
    assert not err and json.loads(ok)["located"] is True
    assert ctx.proposals[0].diff.startswith("--- clause 8.1")
    bad, err = ctx.execute("propose_redline", {
        "clause_id": "8.1", "anchor_text": "text that is not in the clause at all", "replacement": "whatever text", "rationale": "because reasons here",
    })
    assert err and json.loads(bad)["error"] == "ANCHOR_NOT_FOUND"
    assert len(ctx.proposals) == 1 and ctx.proposals[0].located is False  # latest replaces
