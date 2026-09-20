"""LangGraph wrapper: the same pipeline as a checkpointed graph with a
human-in-the-loop interrupt before anything leaves the system.

Why a graph at all? The linear pipeline (pipeline.py) is enough to review.
The first thing that *needed* a framework was "pause for approval, resume
hours later, from another process": that is interrupt() + a checkpointer.

    load -> review -> summarise -> (redlines?) -> approve -> export
                                 \\-> export

`approve` calls interrupt(); the graph persists its state in SQLite and stops.
Resuming with Command(resume=True/False) continues from the same checkpoint.
The approved decision is executed by code - the model does not get to re-decide.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .config import Settings
from .models import ClauseReview, Contract, ReviewReport
from .pipeline import summarise
from .playbook import Playbook
from .redline import apply_replacement
from .review import review_clause
from .split import load_contract


class State(TypedDict, total=False):
    contract_path: str
    out_path: str
    contract: dict
    reviews: list[dict]
    summary: dict
    approved: bool | None
    exported: list[str]


def build_graph(settings: Settings | None = None, checkpointer=None):
    settings = settings or Settings()
    playbook = Playbook.load(settings.playbook_path)

    def load(state: State) -> dict:
        contract = load_contract(state["contract_path"])
        return {"contract": contract.model_dump()}

    def review(state: State) -> dict:
        contract = Contract.model_validate(state["contract"])
        reviews = [review_clause(contract, c, playbook, settings) for c in contract.reviewable]
        return {"reviews": [r.model_dump() for r in reviews]}

    def summarise_node(state: State) -> dict:
        reviews = [ClauseReview.model_validate(r) for r in state["reviews"]]
        return {"summary": summarise(reviews, settings)}

    def route(state: State) -> Literal["approve", "export"]:
        return "approve" if state["summary"]["counts"]["redline"] > 0 else "export"

    def approve(state: State) -> dict:
        pending = [
            {"clause_id": r["clause"]["clause_id"], "anchor": r["redline"]["matched_text"], "replacement": r["redline"]["replacement"], "rationale": r["verdict"]["rationale"]}
            for r in state["reviews"]
            if r["verdict"]["verdict"] == "redline" and r.get("redline") and r["redline"]["located"]
        ]
        decision = interrupt({"question": "Apply these redlines to the exported copy?", "redlines": pending})
        return {"approved": bool(decision)}

    def export(state: State) -> dict:
        contract = Contract.model_validate(state["contract"])
        reviews = [ClauseReview.model_validate(r) for r in state["reviews"]]
        report = ReviewReport(contract_name=contract.name, model=settings.model, mode=settings.mode, summary=state["summary"], reviews=reviews)
        out = Path(state.get("out_path") or f"{contract.name}.report.json")
        out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        written = [str(out)]
        if state.get("approved"):
            text = contract.text
            # apply from the end so earlier offsets stay valid
            for r in sorted((r for r in reviews if r.redline and r.redline.located and r.verdict.verdict == "redline"), key=lambda r: r.clause.char_start, reverse=True):
                # clause.text was .strip()ped; locate the anchor inside the untrimmed slice
                slice_ = text[r.clause.char_start:r.clause.char_end]
                off = slice_.find(r.redline.matched_text or "")
                if off >= 0:
                    s, e = r.clause.char_start + off, r.clause.char_start + off + len(r.redline.matched_text or "")
                    text = apply_replacement(text, s, e, r.redline.replacement)
            redlined = out.with_suffix(".redlined.txt")
            redlined.write_text(text, encoding="utf-8")
            written.append(str(redlined))
        return {"exported": written}

    g = StateGraph(State)
    g.add_node("load", load)
    g.add_node("review", review)
    g.add_node("summarise", summarise_node)
    g.add_node("approve", approve)
    g.add_node("export", export)
    g.add_edge(START, "load")
    g.add_edge("load", "review")
    g.add_edge("review", "summarise")
    g.add_conditional_edges("summarise", route)
    g.add_edge("approve", "export")
    g.add_edge("export", END)
    return g.compile(checkpointer=checkpointer)


def run_graph(contract_path: str, out_path: str, thread_id: str, settings: Settings | None = None, decide=None) -> dict:
    """Run to the interrupt, ask `decide(pending) -> bool`, resume. Returns final state."""
    settings = settings or Settings()
    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(settings.checkpoint_path)) as saver:
        app = build_graph(settings, checkpointer=saver)
        cfg = {"configurable": {"thread_id": thread_id}}
        result = app.invoke({"contract_path": contract_path, "out_path": out_path}, cfg)
        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            approved = decide(payload) if decide else False
            result = app.invoke(Command(resume=approved), cfg)
        return result


def pending_interrupt(thread_id: str, settings: Settings | None = None) -> dict | None:
    """Inspect a paused thread (e.g. from another process) without resuming it."""
    settings = settings or Settings()
    with SqliteSaver.from_conn_string(str(settings.checkpoint_path)) as saver:
        app = build_graph(settings, checkpointer=saver)
        snap = app.get_state({"configurable": {"thread_id": thread_id}})
        for task in snap.tasks:
            if task.interrupts:
                return task.interrupts[0].value
    return None


def resume(thread_id: str, approved: bool, settings: Settings | None = None) -> dict:
    settings = settings or Settings()
    with SqliteSaver.from_conn_string(str(settings.checkpoint_path)) as saver:
        app = build_graph(settings, checkpointer=saver)
        return app.invoke(Command(resume=approved), {"configurable": {"thread_id": thread_id}})


if __name__ == "__main__":  # quick smoke run
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_contract.txt"
    final = run_graph(path, "data/sample_contract.report.json", "demo-1", decide=lambda p: (print(json.dumps(p, indent=2)), True)[1])
    print(final.get("exported"))
