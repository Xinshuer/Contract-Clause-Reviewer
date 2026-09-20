"""promptfoo custom Python provider.

promptfoo passes each test's vars here; we run the *whole* review pipeline
for that case (not just a prompt) and return the verdict JSON as output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clausecheck.config import Settings  # noqa: E402
from clausecheck.playbook import Playbook  # noqa: E402
from evals.harness import build_contract, case_by_id, check  # noqa: E402
from clausecheck.review import review_clause  # noqa: E402

_SETTINGS = Settings()
_PLAYBOOK = Playbook.load(_SETTINGS.playbook_path)


def call_api(prompt: str, options: dict, context: dict) -> dict:
    case = case_by_id(context["vars"]["case_id"])
    contract, clause = build_contract(case)
    review = review_clause(contract, clause, _PLAYBOOK, _SETTINGS)
    fails = check(case, review)
    payload = review.verdict.model_dump()
    payload["_tool_calls"] = review.trace.tool_calls
    payload["_harness_failures"] = fails
    payload["_redline_located"] = bool(review.redline and review.redline.located)
    return {
        "output": json.dumps(payload, ensure_ascii=False),
        "tokenUsage": {
            "total": review.trace.total_tokens,
            "prompt": review.trace.input_tokens + review.trace.cache_read_input_tokens,
            "completion": review.trace.output_tokens,
        },
    }
