"""The playbook is the customer's negotiation position as data.

The model never invents the standard; it compares a clause against the
2-3 rules for its topic. Swap the JSON to serve a different customer -
no code or prompt change needed.
"""
from __future__ import annotations

import json
from pathlib import Path

from .models import PlaybookRule


class Playbook:
    def __init__(self, rules: list[PlaybookRule]):
        self.rules = rules
        self._by_topic: dict[str, list[PlaybookRule]] = {}
        for r in rules:
            self._by_topic.setdefault(r.topic, []).append(r)

    @classmethod
    def load(cls, path: str | Path) -> "Playbook":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([PlaybookRule(**r) for r in data["rules"]])

    @property
    def topics(self) -> list[str]:
        return sorted(self._by_topic)

    def for_topic(self, topic: str) -> list[PlaybookRule]:
        return list(self._by_topic.get(topic, []))

    def get(self, rule_id: str) -> PlaybookRule | None:
        return next((r for r in self.rules if r.id == rule_id), None)

    def lookup(self, topic: str) -> dict:
        """Tool-shaped result: trimmed projection, never the raw file."""
        rules = self.for_topic(topic)
        if not rules:
            return {"topic": topic, "rules": [], "note": f"No rules for this topic. Known topics: {self.topics}"}
        return {"topic": topic, "rules": [r.for_model() for r in rules]}
