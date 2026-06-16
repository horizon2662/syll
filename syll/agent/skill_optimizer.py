"""SkillOptimizer — automatic skill/memory optimization engine.

Implements the SkillOpt ReflACT 6-stage pipeline adapted for Syll:
  1. Collect — gather conversation trajectories from sessions/events
  2. Reflect — LLM analysis of failure/success patterns, generate patches
  3. Aggregate — hierarchical merge of patches (failure-first)
  4. Select — rank and select top-N edits within budget
  5. Update — apply edits to skill/memory document
  6. Evaluate — validate candidate, accept/reject with gate control

Designed to run as a background asyncio task alongside `syll wake --skill-optimizer`.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from syll.agent.skill_edit import (
    Edit,
    GateAction,
    GateResult,
    Patch,
    apply_edit,
    apply_patch,
    validate_skill_document,
)
from syll.providers.base import LLMProvider

_PROMPTS_DIR = Path(__file__).parent / "prompts" / "skill_optimizer"
_OPTIMIZER_LOG_DIR = Path.home() / ".syll" / "optimizer"

# ── Prompt loading ────────────────────────────────────────────────────────


def _load_prompt(name: str) -> str:
    """Load an optimizer prompt from the prompts directory."""
    path = _PROMPTS_DIR / f"{name}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Optimizer prompt not found: {path}")


# ── Trajectory scoring ───────────────────────────────────────────────────


_POSITIVE_WORDS = {"谢谢", "好的", "好了", "完美", "perfect", "thanks", "great", "好"}
_NEGATIVE_WORDS = {"不对", "错了", "重试", "retry", "wrong", "error", "不对", "不行", "没成功"}
_ERROR_PATTERNS = [r"Error calling LLM:", r"Error:", r"Traceback", r"exception", r"failed"]


def score_trajectory(messages: list[dict]) -> tuple[float, str]:
    """Score a conversation trajectory. Returns (score 0-1, reason).

    0 = clear failure, 1 = clear success.
    """
    if not messages:
        return 0.0, "empty"

    has_error = False
    has_negative = False
    has_positive = False
    user_turns = 0
    assistant_turns = 0

    for msg in messages:
        role = msg.get("role", "")
        content = str(msg.get("content", ""))

        if role == "assistant":
            assistant_turns += 1
            for pat in _ERROR_PATTERNS:
                if re.search(pat, content, re.IGNORECASE):
                    has_error = True

        elif role == "user":
            user_turns += 1
            lower = content.lower().strip()
            for w in _POSITIVE_WORDS:
                if w in lower:
                    has_positive = True
            for w in _NEGATIVE_WORDS:
                if w in lower:
                    has_negative = True

    # Scoring logic
    if has_error:
        return 0.2, "llm_or_tool_error"
    if has_negative:
        return 0.3, "user_negative_feedback"
    if user_turns > 6:
        return 0.4, "many_turns"
    if has_positive:
        return 0.9, "user_positive_feedback"
    if assistant_turns > 0 and not has_error:
        return 0.7, "completed_no_error"
    return 0.5, "neutral"
