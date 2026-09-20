"""Python assertions for promptfoo (type: python, value: file://...:function)."""
from __future__ import annotations

import json


def harness_passes(output: str, context: dict) -> bool | dict:
    data = json.loads(output)
    fails = data.get("_harness_failures", [])
    return True if not fails else {"pass": False, "score": 0, "reason": "; ".join(fails)}


def verdict_not_accept(output: str, context: dict) -> bool:
    return json.loads(output)["verdict"] != "accept"


def verdict_accept(output: str, context: dict) -> bool:
    return json.loads(output)["verdict"] == "accept"


def redline_anchored(output: str, context: dict) -> bool | dict:
    data = json.loads(output)
    if data["verdict"] != "redline":
        return True
    return True if data.get("_redline_located") else {"pass": False, "score": 0, "reason": "redline without located anchor"}
