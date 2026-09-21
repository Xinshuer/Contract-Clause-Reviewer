"""Command line entry point.

    python -m clausecheck review data/sample_contract.txt
    python -m clausecheck review contract.pdf --out report.json --mode live
    python -m clausecheck review contract.pdf --graph --approve ask
    python -m clausecheck clauses data/sample_contract.txt      # just the split
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .models import ReviewReport
from .pipeline import review_contract
from .split import load_contract
from .tracing import flush


def _print_report(report: ReviewReport) -> None:
    print(f"\n{report.contract_name}  |  mode={report.mode}  model={report.model}")
    print("-" * 96)
    print(f"{'clause':<8}{'verdict':<9}{'risk':<8}{'topic':<20}{'rules':<14}{'tools'}")
    for r in report.reviews:
        v = r.verdict
        print(f"{v.clause_id:<8}{v.verdict:<9}{v.risk_level:<8}{v.topic:<20}{','.join(v.rule_ids) or '-':<14}{','.join(r.trace.tool_calls)}")
    s = report.summary
    print("-" * 96)
    print(f"counts={s['counts']}  needs_attention={s['needs_attention']}")
    print(f"tokens={s['tokens']}  cache_hit={s['cache_hit_ratio']}  est_cost=${s['estimated_cost_usd']}  latency={s['total_latency_ms']}ms")
    if s["errors"]:
        print(f"errors on clauses: {s['errors']}")


def cmd_review(args: argparse.Namespace) -> int:
    settings = Settings()
    if args.mode:
        settings = Settings(mode=args.mode)
    out = Path(args.out) if args.out else Path(args.path).with_suffix(".report.json")

    if args.graph:
        from .graph import run_graph

        def decide(payload: dict) -> bool:
            print("\nInterrupted before export. Pending redlines:")
            for p in payload["redlines"]:
                print(f"  [{p['clause_id']}] {p['anchor'][:90]!r} -> {p['replacement'][:90]!r}")
            if args.approve == "ask":
                return input("Apply to exported copy? [y/N] ").strip().lower() == "y"
            return args.approve == "yes"

        final = run_graph(args.path, str(out), args.thread, settings, decide)
        report = ReviewReport.model_validate_json(out.read_text(encoding="utf-8"))
        _print_report(report)
        print(f"approved={final.get('approved')}  exported={final.get('exported')}")
        flush()
        return 0

    def progress(i: int, n: int, review) -> None:
        print(f"[{i:>2}/{n}] {review.clause.clause_id:<6} {review.verdict.verdict:<8} {review.clause.heading[:50]}", flush=True)

    report = review_contract(args.path, settings, progress=progress)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    _print_report(report)
    print(f"\nreport written to {out}")
    flush()
    return 0


def cmd_clauses(args: argparse.Namespace) -> int:
    contract = load_contract(args.path)
    for c in contract.clauses:
        kind = "section" if c.is_section_title else "clause "
        print(f"{kind} {c.clause_id:<6} {c.heading[:45]:<45} refs={c.refs} chars={len(c.text)}")
    if args.json:
        Path(args.json).write_text(json.dumps([c.model_dump() for c in contract.clauses], indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles
    except Exception:
        pass
    p = argparse.ArgumentParser(prog="clausecheck", description="Contract Clause Reviewer")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("review", help="review every clause of a contract")
    r.add_argument("path")
    r.add_argument("--out", help="report JSON path")
    r.add_argument("--mode", choices=["auto", "live", "deepseek", "local", "mock"], help="override CLAUSECHECK_MODE")
    r.add_argument("--graph", action="store_true", help="run via the LangGraph workflow with an approval interrupt")
    r.add_argument("--approve", choices=["ask", "yes", "no"], default="ask", help="how to answer the interrupt (graph mode)")
    r.add_argument("--thread", default="cli-1", help="checkpoint thread id (graph mode)")
    r.set_defaults(fn=cmd_review)

    c = sub.add_parser("clauses", help="show how the contract is split")
    c.add_argument("path")
    c.add_argument("--json", help="write clauses.json")
    c.set_defaults(fn=cmd_clauses)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
