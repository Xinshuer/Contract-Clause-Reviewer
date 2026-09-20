"""Pydantic models shared by the whole pipeline.

One definition is used in three places: the model-facing JSON schema
(structured output + tool input_schema), runtime validation, and the
report that the UI / API return. Keeping them identical is the point.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Topic = Literal[
    "payment",
    "auto_renewal",
    "ip_ownership",
    "confidentiality",
    "liability_cap",
    "indemnification",
    "data_processing",
    "termination",
    "unilateral_changes",
    "governing_law",
    "other",
]
Verdict = Literal["accept", "flag", "redline"]
Risk = Literal["low", "medium", "high"]
Confidence = Literal["low", "medium", "high"]


class Clause(BaseModel):
    clause_id: str
    heading: str
    text: str
    section_path: str = ""
    refs: list[str] = Field(default_factory=list, description="Cross-referenced clause ids found by the parser")
    char_start: int = 0
    char_end: int = 0

    @property
    def is_section_title(self) -> bool:
        return not self.text.strip()


class Contract(BaseModel):
    name: str
    text: str
    clauses: list[Clause]

    def get(self, clause_id: str) -> Clause | None:
        for c in self.clauses:
            if c.clause_id == clause_id:
                return c
        return None

    def full_text_of(self, clause_id: str) -> str | None:
        """Text of a clause; for a bare section title, the concatenated children."""
        c = self.get(clause_id)
        if c is None:
            return None
        if not c.is_section_title:
            return c.text
        children = [x for x in self.clauses if x.clause_id.startswith(clause_id + ".")]
        return "\n".join(f"{x.clause_id} {x.heading}\n{x.text}" for x in children)

    @property
    def reviewable(self) -> list[Clause]:
        return [c for c in self.clauses if not c.is_section_title]


class PlaybookRule(BaseModel):
    id: str
    topic: Topic
    title: str
    position: str
    fallback_positions: list[str] = Field(default_factory=list)
    never_accept: list[str] = Field(default_factory=list)
    severity: Risk = "medium"
    preferred_language: str = ""
    # Used only by the offline rule-based reviewer (mock mode) and for topic hints.
    keywords: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)

    def for_model(self) -> dict:
        """Projection sent to the model: no keywords / regexes, just the position."""
        return self.model_dump(exclude={"keywords", "red_flags"})


class ClauseVerdict(BaseModel):
    """The structured output the model must return for one clause."""

    clause_id: str
    topic: Topic
    verdict: Verdict = Field(description="accept = within playbook; flag = problem, needs a human; redline = replacement proposed")
    risk_level: Risk
    rule_ids: list[str] = Field(description="Playbook rule ids the verdict is based on; empty if none applied")
    rationale: str = Field(description="2-4 sentences. Quote the clause and name the specific risk. Neutral tone.")
    confidence: Confidence
    anchor_text: str | None = Field(default=None, description="Verbatim excerpt of the clause to replace (redline only)")
    replacement: str | None = Field(default=None, description="Proposed replacement text (redline only)")
    consulted_clauses: list[str] = Field(default_factory=list, description="Other clause ids you read via get_clause")


class RedlineProposal(BaseModel):
    clause_id: str
    anchor_text: str
    replacement: str
    rationale: str = ""
    located: bool = False
    start: int | None = None
    end: int | None = None
    matched_text: str | None = None
    diff: str = ""


class ReviewTrace(BaseModel):
    model: str = ""
    mode: str = ""
    prompt_version: str = ""
    steps: int = 0
    tool_calls: list[str] = Field(default_factory=list)
    model_verdict: str = Field(default="", description="verdict in the model's final JSON, before post-processing")
    marks: list[str] = Field(default_factory=list, description="reasons passed to mark_for_review")
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    latency_ms: int = 0
    stop_reason: str = ""
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )


class ClauseReview(BaseModel):
    clause: Clause
    verdict: ClauseVerdict
    redline: RedlineProposal | None = None
    trace: ReviewTrace = Field(default_factory=ReviewTrace)


class ReviewReport(BaseModel):
    contract_name: str
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    model: str
    mode: str
    summary: dict
    reviews: list[ClauseReview]


def strict_schema(model: type[BaseModel]) -> dict:
    """JSON schema for structured output: every property required, no extras."""
    schema = model.model_json_schema()

    def walk(node: dict) -> None:
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
            for sub in node["properties"].values():
                walk(sub)
        for key in ("items",):
            if isinstance(node.get(key), dict):
                walk(node[key])
        for key in ("anyOf", "oneOf", "allOf"):
            for sub in node.get(key, []) or []:
                walk(sub)
        for sub in (node.get("$defs") or {}).values():
            walk(sub)

    walk(schema)
    return schema
