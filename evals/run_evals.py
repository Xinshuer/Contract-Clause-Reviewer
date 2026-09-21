"""Run every eval case N times and print a pass-rate table.

    python evals/run_evals.py --runs 5
    python evals/run_evals.py --runs 3 --mode mock
    python evals/run_evals.py --only cross-reference-trap

A single green run proves nothing for a stochastic system; the number that
matters is the pass rate per case, and pass^k for anything unattended.
Writes evals/results.json for later comparison (prompt v1 vs v2).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clausecheck.config import Settings  # noqa: E402
from clausecheck.playbook import Playbook  # noqa: E402
from clausecheck.tracing import flush  # noqa: E402
from evals.harness import high_risk_recall, load_cases, run_case  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--mode", choices=["auto", "live", "deepseek", "local", "mock"])
    ap.add_argument("--prompt", choices=["v1", "v2"], help="system prompt version (default: env / v2)")
    ap.add_argument("--only", help="run a single case id")
    ap.add_argument("--threshold", type=float, default=0.8, help="per-case pass rate required")
    ap.add_argument("--out", default=str(Path(__file__).with_name("results.json")))
    args = ap.parse_args()

    kw = {}
    if args.mode:
        kw["mode"] = args.mode
    if args.prompt:
        kw["prompt_version"] = args.prompt
    settings = Settings(**kw)
    playbook = Playbook.load(settings.playbook_path)
    cases = [c for c in load_cases() if not args.only or c["id"] == args.only]
    results: dict[str, list[list[str]]] = defaultdict(list)
    tokens = 0
    t0 = time.perf_counter()
    print(f"mode={settings.mode} model={settings.model} prompt={settings.prompt_version} runs={args.runs} cases={len(cases)}\n")
    for case in cases:
        for i in range(args.runs):
            fails, review = run_case(case, settings, playbook)
            results[case["id"]].append(fails)
            tokens += review.trace.total_tokens
            mark = "." if not fails else "F"
            print(f"{case['id']:<36} run {i + 1}: {mark} {('; '.join(fails)) if fails else review.verdict.verdict}", flush=True)

    print(f"\n{'case':<36}{'kind':<12}{'pass rate':<11}{'pass^k':<8}most common failure")
    print("-" * 100)
    worst = 0.0
    all_ok = True
    for case in cases:
        runs = results[case["id"]]
        passed = sum(1 for r in runs if not r)
        rate = passed / len(runs)
        pk = "yes" if passed == len(runs) else "no"
        common = ""
        if passed < len(runs):
            flat = [f for r in runs for f in r]
            common = max(set(flat), key=flat.count)
        if rate < args.threshold:
            all_ok = False
        worst = max(worst, 1 - rate)
        print(f"{case['id']:<36}{case['kind']:<12}{passed}/{len(runs):<9}{pk:<8}{common[:60]}")
    recall = high_risk_recall(results, cases)
    print("-" * 100)
    print(f"high-risk recall (must_flag + trap): {recall:.2f}   total tokens: {tokens}   wall: {time.perf_counter() - t0:.1f}s")

    Path(args.out).write_text(
        json.dumps(
            {
                "mode": settings.mode,
                "model": settings.model,
                "prompt_version": settings.prompt_version,
                "runs": args.runs,
                "high_risk_recall": recall,
                "cases": {cid: {"pass_rate": sum(1 for r in rs if not r) / len(rs), "failures": rs} for cid, rs in results.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    flush()
    print(f"results written to {args.out}")
    return 0 if all_ok and recall >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
