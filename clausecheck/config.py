"""Runtime settings. Everything comes from environment variables / .env.

Mode:
  auto  - live if Anthropic credentials exist, else local if an OpenAI-compatible
          server answers at CLAUSECHECK_LOCAL_BASE_URL, else mock.
  live  - Anthropic API (claude-opus-5 by default).
  local - OpenAI-compatible local server (LM Studio, llama.cpp server, Ollama).
  mock  - never call a model. Deterministic rule-based reviewer; for tests/UI work.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "claude-opus-5"
MODES = ("auto", "live", "local", "mock")

# USD per million tokens (first-party Anthropic API, 2026-06 price list).
PRICES = {
    "claude-opus-5": {"input": 5.0, "output": 25.0, "cache_write": 6.25, "cache_read": 0.50},
    "claude-sonnet-5": {"input": 2.0, "output": 10.0, "cache_write": 2.50, "cache_read": 0.20},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_write": 1.25, "cache_read": 0.10},
}
FREE = {"input": 0.0, "output": 0.0, "cache_write": 0.0, "cache_read": 0.0}


def _has_anthropic_credentials() -> bool:
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return True
    profile_dir = Path.home() / ".config" / "anthropic"
    return profile_dir.exists() and any(profile_dir.iterdir())


def _local_server_answers(base_url: str) -> bool:
    try:
        import requests

        return requests.get(f"{base_url.rstrip('/')}/models", timeout=1.5).status_code == 200
    except Exception:
        return False


@dataclass
class Settings:
    model: str = field(default_factory=lambda: os.getenv("CLAUSECHECK_MODEL", DEFAULT_MODEL))
    mode: str = field(default_factory=lambda: os.getenv("CLAUSECHECK_MODE", "auto"))
    effort: str = field(default_factory=lambda: os.getenv("CLAUSECHECK_EFFORT", "medium"))
    prompt_version: str = field(default_factory=lambda: os.getenv("CLAUSECHECK_PROMPT_VERSION", "v3"))
    max_steps: int = field(default_factory=lambda: int(os.getenv("CLAUSECHECK_MAX_STEPS", "6")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("CLAUSECHECK_MAX_TOKENS", "8192")))
    # Anthropic server-side refusal fallback (beta). Off by default; see README.
    fallbacks: bool = field(default_factory=lambda: os.getenv("CLAUSECHECK_FALLBACKS", "0") == "1")
    # local OpenAI-compatible server
    local_base_url: str = field(default_factory=lambda: os.getenv("CLAUSECHECK_LOCAL_BASE_URL", "http://127.0.0.1:1234/v1"))
    local_model: str = field(default_factory=lambda: os.getenv("CLAUSECHECK_LOCAL_MODEL", "qwen38"))
    local_temperature: float = field(default_factory=lambda: float(os.getenv("CLAUSECHECK_LOCAL_TEMPERATURE", "0.3")))
    local_max_tokens: int = field(default_factory=lambda: int(os.getenv("CLAUSECHECK_LOCAL_MAX_TOKENS", "3000")))
    # Put the whole contract in the local model's system prompt (slow prefill, needs
    # a large context) instead of an index the model reads from via get_clause.
    local_full_contract: bool = field(default_factory=lambda: os.getenv("CLAUSECHECK_LOCAL_FULL_CONTRACT", "0") == "1")
    playbook_path: Path = field(default_factory=lambda: Path(os.getenv("CLAUSECHECK_PLAYBOOK", ROOT / "data" / "playbook.json")))
    checkpoint_path: Path = field(default_factory=lambda: Path(os.getenv("CLAUSECHECK_CHECKPOINTS", ROOT / "data" / "checkpoints.sqlite")))

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"CLAUSECHECK_MODE must be one of {MODES}, got {self.mode!r}")
        if self.prompt_version not in ("v1", "v2", "v3"):
            raise ValueError(f"CLAUSECHECK_PROMPT_VERSION must be v1, v2 or v3, got {self.prompt_version!r}")
        if self.mode == "auto":
            if _has_anthropic_credentials():
                self.mode = "live"
            elif _local_server_answers(self.local_base_url):
                self.mode = "local"
            else:
                self.mode = "mock"
        if self.mode == "local":
            self.model = self.local_model
        elif self.mode == "mock":
            self.model = "rule-based"

    @property
    def mock(self) -> bool:
        return self.mode == "mock"

    def price(self) -> dict:
        if self.mode != "live":
            return FREE
        return PRICES.get(self.model, PRICES[DEFAULT_MODEL])
