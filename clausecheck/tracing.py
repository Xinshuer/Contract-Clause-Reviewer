"""Langfuse tracing that degrades to a no-op when keys are absent.

Usage:
    from clausecheck.tracing import observe, annotate, flush

    @observe(name="review_clause")
    def review_clause(...): ...
        annotate(clause_id=..., rules_used=[...])

Set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST to enable.
Langfuse's Python SDK is OpenTelemetry-based, so nested @observe calls become
child spans of the outer one automatically.
"""
from __future__ import annotations

import functools
import os
from typing import Any, Callable

ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

if ENABLED:
    from langfuse import get_client, observe  # type: ignore

    def annotate(**metadata: Any) -> None:
        try:
            get_client().update_current_span(metadata=metadata)
        except Exception:  # never let tracing break the pipeline
            pass

    def flush() -> None:
        try:
            get_client().flush()
        except Exception:
            pass

else:

    def observe(func: Callable | None = None, **_kwargs: Any):  # type: ignore[misc]
        """Identity decorator; supports both @observe and @observe(name=...)."""

        def wrap(f: Callable) -> Callable:
            @functools.wraps(f)
            def inner(*a: Any, **k: Any) -> Any:
                return f(*a, **k)

            return inner

        return wrap(func) if callable(func) else wrap

    def annotate(**_metadata: Any) -> None:
        return None

    def flush() -> None:
        return None
