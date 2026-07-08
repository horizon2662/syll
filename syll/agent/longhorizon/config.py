"""Runner configuration (from environment)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunnerConfig:
    """Configuration for the end-to-end runner.

    Env vars:
      SYLL_MODEL     -- litellm model id, e.g. "glm-4.6", "anthropic/claude-sonnet-4-5"
      SYLL_API_KEY   -- API key (falls back to ZHIPUAI_API_KEY)
      SYLL_API_BASE  -- custom endpoint (optional; e.g. an Anthropic-compatible base)
    """

    model: str = "glm-4.6"
    api_key: str | None = None
    api_base: str | None = None
    workspace: Path = field(default_factory=lambda: Path.home() / ".syll")
    skill: str = "default"
    max_subagent_iterations: int = 12
    max_replans_per_step: int = 2
    compaction_threshold: int = 20  # notes lines before the orchestrator compacts
    # --- verifier-ceiling experiment knobs (defaults preserve original behavior) ---
    recovery_mode: str = "replan"  # replan | retry_same_node | retry_then_replan
    retry_context: str = "fresh"   # fresh | replay  (A1 context-sovereignty switch)
    max_retries_per_step: int = 3  # only used when recovery_mode includes retry
    context_window: int = 0  # input-token budget for the detector (0 = litellm lookup); SYLL_CONTEXT_WINDOW

    @classmethod
    def from_env(
        cls,
        skill: str = "default",
        workspace: str | Path | None = None,
    ) -> "RunnerConfig":
        # Prefer the Anthropic-compatible path (ANTHROPIC_AUTH_TOKEN) -- this is
        # what actually works for GLM/Zhipu's /api/anthropic endpoint (the
        # Claude Code harness authenticates the same way). Fall back to SYLL_*
        # / ZHIPUAI_* for OpenAI/litellm-style providers.
        return cls(
            model=os.environ.get("ANTHROPIC_MODEL")
            or os.environ.get("SYLL_MODEL", "glm-5.2"),
            api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or os.environ.get("SYLL_API_KEY")
            or os.environ.get("ZHIPUAI_API_KEY"),
            api_base=os.environ.get("ANTHROPIC_BASE_URL")
            or os.environ.get("SYLL_API_BASE") or None,
            workspace=Path(workspace) if workspace else Path.home() / ".syll",
            skill=skill,
            recovery_mode=os.environ.get("SYLL_RECOVERY_MODE", "replan"),
            retry_context=os.environ.get("SYLL_RETRY_CONTEXT", "fresh"),
            max_retries_per_step=int(os.environ.get("SYLL_MAX_RETRIES", "3")),
            context_window=int(os.environ.get("SYLL_CONTEXT_WINDOW", "0")),
        )

    @property
    def use_anthropic_sdk(self) -> bool:
        """True when the endpoint is Anthropic-compatible -> use the SDK
        provider (Bearer auth) instead of litellm (x-api-key)."""
        return bool(self.api_base and "/anthropic" in self.api_base)
