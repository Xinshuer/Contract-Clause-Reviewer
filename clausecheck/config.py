"""Runtime settings. Everything comes from environment variables / .env.

Mode:
  auto     - live if Anthropic credentials exist; else deepseek if DEEPSEEK_API_KEY
             is set; else local if an OpenAI-compatible server answers; else mock.
  live     - Anthropic API (claude-opus-5 by default).
  deepseek - DeepSeek's OpenAI-compatible API (deepseek-flash). Needs DEEPSEEK_API_KEY.
  local    - OpenAI-compatible local server (LM Studio, llama.cpp server, Ollama).
  mock     - never call a model. Deterministic rule-based reviewer; for tests/UI work.

deepseek and local share one backend (OpenAICompatBackend); the difference is
only base URL, model name, API key and which JSON mode the server supports.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "claude-opus-5"
MODES = ("auto", "live", "deepseek", "local", "mock")
COMPAT_MODES = ("deepseek", "local")

# USD per million tokens. Anthropic: first-party API, 2026-06 price list.
# DeepSeek: 2026-09 peak list price (off-peak is half; no cache-write premium, so
# cache_write = input); verify at https://api-docs.deepseek.com/quick_start/pricing
# before quoting a number. deepseek-chat / deepseek-reasoner were announced as discontinued
# from 2026-07-24; in 2026-09 they still answer as aliases of deepseek-flash but are no
# longer listed by /models, so don't rely on them.
PRICES = {
    "claude-opus-5": {"input": 5.0, "output": 25.0, "cache_write": 6.25, "cache_read": 0.50},
    "claude-sonnet-5": {"input": 2.0, "output": 10.0, "cache_write": 2.50, "cache_read": 0.20},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_write": 1.25, "cache_read": 0.10},
    "deepseek-flash": {"input": 0.30, "output": 1.20, "cache_write": 0.30, "cache_read": 0.006},
    "deepseek-v4-flash": {"input": 0.30, "output": 1.20, "cache_write": 0.30, "cache_read": 0.006},
    "deepseek-v4-pro": {"input": 1.32, "output": 3.96, "cache_write": 1.32, "cache_read": 0.044},
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
    prompt_version: str = field(default_factory=lambda: os.getenv("CLAUSECHECK_PROMPT_VERSION", "v2"))
    max_steps: int = field(default_factory=lambda: int(os.getenv("CLAUSECHECK_MAX_STEPS", "6")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("CLAUSECHECK_MAX_TOKENS", "8192")))
    # Anthropic server-side refusal fallback (beta). Off by default; see README.
    fallbacks: bool = field(default_factory=lambda: os.getenv("CLAUSECHECK_FALLBACKS", "0") == "1")

    # --- DeepSeek (OpenAI-compatible, hosted) ---------------------------------
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    deepseek_model: str = field(default_factory=lambda: os.getenv("CLAUSECHECK_DEEPSEEK_MODEL", "deepseek-flash"))

    # --- local OpenAI-compatible server ---------------------------------------
    local_base_url: str = field(default_factory=lambda: os.getenv("CLAUSECHECK_LOCAL_BASE_URL", "http://127.0.0.1:1234/v1"))
    local_model: str = field(default_factory=lambda: os.getenv("CLAUSECHECK_LOCAL_MODEL", "qwen38"))
    local_api_key: str = field(default_factory=lambda: os.getenv("CLAUSECHECK_LOCAL_API_KEY", ""))

    # --- shared by deepseek + local -------------------------------------------
    compat_temperature: float = field(default_factory=lambda: float(os.getenv("CLAUSECHECK_TEMPERATURE", os.getenv("CLAUSECHECK_LOCAL_TEMPERATURE", "0.3"))))
    compat_max_tokens: int = field(default_factory=lambda: int(os.getenv("CLAUSECHECK_COMPAT_MAX_TOKENS", os.getenv("CLAUSECHECK_LOCAL_MAX_TOKENS", "3000"))))
    # json_schema = grammar-constrained (LM Studio, llama.cpp); json_object = "valid JSON,
    # schema described in the prompt" (DeepSeek). Empty = pick per mode.
    json_mode: str = field(default_factory=lambda: os.getenv("CLAUSECHECK_JSON_MODE", ""))
    # Put the whole contract in the system prompt (needs a large context, slow prefill on
    # local models) instead of an index the model reads from via get_clause.
    full_contract: bool = field(default_factory=lambda: os.getenv("CLAUSECHECK_FULL_CONTRACT", os.getenv("CLAUSECHECK_LOCAL_FULL_CONTRACT", "")) == "1")

    playbook_path: Path = field(default_factory=lambda: Path(os.getenv("CLAUSECHECK_PLAYBOOK", ROOT / "data" / "playbook.json")))
    checkpoint_path: Path = field(default_factory=lambda: Path(os.getenv("CLAUSECHECK_CHECKPOINTS", ROOT / "data" / "checkpoints.sqlite")))

    # resolved for the OpenAI-compatible backend (set in __post_init__)
    compat_base_url: str = ""
    compat_api_key: str = ""
    compat_json_mode: str = ""

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"CLAUSECHECK_MODE must be one of {MODES}, got {self.mode!r}")
        if self.prompt_version not in ("v1", "v2", "v3"):
            raise ValueError(f"CLAUSECHECK_PROMPT_VERSION must be v1, v2 or v3, got {self.prompt_version!r}")
        if self.json_mode not in ("", "json_schema", "json_object"):
            raise ValueError(f"CLAUSECHECK_JSON_MODE must be json_schema or json_object, got {self.json_mode!r}")
        if self.mode == "auto":
            if _has_anthropic_credentials():
                self.mode = "live"
            elif self.deepseek_api_key:
                self.mode = "deepseek"
            elif _local_server_answers(self.local_base_url):
                self.mode = "local"
            else:
                self.mode = "mock"
        if self.mode == "deepseek":
            if not self.deepseek_api_key:
                raise ValueError("CLAUSECHECK_MODE=deepseek needs DEEPSEEK_API_KEY (put it in .env)")
            self.model = self.deepseek_model
            self.compat_base_url = self.deepseek_base_url
            self.compat_api_key = self.deepseek_api_key
            self.compat_json_mode = self.json_mode or "json_object"
            if os.getenv("CLAUSECHECK_FULL_CONTRACT") is None and os.getenv("CLAUSECHECK_LOCAL_FULL_CONTRACT") is None:
                self.full_contract = True  # hosted model: 64k+ context, cached prefix is cheap
        elif self.mode == "local":
            self.model = self.local_model
            self.compat_base_url = self.local_base_url
            self.compat_api_key = self.local_api_key
            self.compat_json_mode = self.json_mode or "json_schema"
        elif self.mode == "mock":
            self.model = "rule-based"

    @property
    def mock(self) -> bool:
        return self.mode == "mock"

    @property
    def compat(self) -> bool:
        return self.mode in COMPAT_MODES

    def price(self) -> dict:
        if self.mode == "local" or self.mode == "mock":
            return FREE
        return PRICES.get(self.model, PRICES[DEFAULT_MODEL] if self.mode == "live" else FREE)
