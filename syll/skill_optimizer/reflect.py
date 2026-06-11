"""LLM-based reflect engine for skill optimization.

Adapted from skillopt.gradient.reflect.  Uses Syll's own LLM provider
to analyze rollouts and generate skill edit patches.
"""
from __future__ import annotations

import json
import traceback
from typing import TYPE_CHECKING

from loguru import logger

from syll.skill_optimizer.types import RolloutResult

if TYPE_CHECKING:
    from syll.providers.base import LLMProvider


# ── Default analyst prompts ────────────────────────────────────────────────

ERROR_ANALYST_SYSTEM = """You are a skill optimization analyst. Your job is to analyze failed task
executions and propose edits to the skill document that would fix those failures.

The skill document is a markdown file that guides an AI agent's behavior.
You must propose specific, targeted edits that address the failure patterns
you observe in the trajectories.

Output a JSON object with this exact structure:
{
  "reasoning": "Brief analysis of why the failures occurred",
  "failure_summary": [
    {"failure_type": "...", "count": N, "description": "..."}
  ],
  "patch": {
    "edits": [
      {"op": "append|insert_after|replace|delete", "content": "...", "target": "..."}
    ]
  }
}

Rules:
- `append`: add new content at the end
- `insert_after`: insert content after the `target` text
- `replace`: replace the `target` text with `content`
- `delete`: remove the `target` text
- Each edit should be specific and actionable
- Focus on the most impactful fixes first
- Do NOT duplicate content that already exists in the skill"""

SUCCESS_ANALYST_SYSTEM = """You are a skill optimization analyst. Your job is to analyze successful task
executions and identify patterns that should be reinforced in the skill document.

Output a JSON object with this exact structure:
{
  "reasoning": "Brief analysis of why these succeeded and what patterns to reinforce",
  "patch": {
    "edits": [
      {"op": "append|insert_after|replace", "content": "...", "target": "..."}
    ]
  }
}

Rules:
- Focus on reinforcing successful strategies
- Only propose edits that add genuine value
- Avoid bloating the skill document unnecessarily"""


# ── Trajectory formatting ──────────────────────────────────────────────────

def fmt_trajectory(conversation: list[dict], max_chars: int = 4000) -> str:
    """Format a conversation into analyst-readable text."""
    lines: list[str] = []
    for item in conversation:
        if not isinstance(item, dict):
            lines.append(f"[agent] {str(item)[:500]}")
            continue
        role = item.get("role", "agent")
        content = str(item.get("content", ""))[:500]
        if content:
            lines.append(f"[{role}] {content}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


def _extract_json(response: str) -> dict | None:
    """Extract the first JSON object from a string."""
    import re
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try raw JSON parse
    response = response.strip()
    # Find first { and last }
    start = response.find("{")
    end = response.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
    return None


# ── Analyst runners ────────────────────────────────────────────────────────

async def run_error_analyst(
    provider: "LLMProvider",
    model: str,
    skill_content: str,
    results: list[RolloutResult | dict],
    edit_budget: int = 4,
) -> dict | None:
    """Analyze failed trajectories and propose edits.

    Returns a patch dict with ``source_type="failure"``, or None.
    """
    failures = []
    for r in results:
        r_dict = r.to_dict() if isinstance(r, RolloutResult) else r
        if not r_dict.get("hard") or float(r_dict.get("hard", 0)) < 1e-9:
            failures.append(r_dict)

    if not failures:
        return None

    # Format trajectories
    traj_parts: list[str] = []
    for i, item in enumerate(failures[:10], 1):
        task = item.get("task", "")
        resp = item.get("response", "")
        expected = item.get("expected", "")
        reason = item.get("fail_reason", "")
        conv = item.get("conversation", [])

        header = f"### Task {i} (id={item.get('id', '?')})\nTask: {task}\n"
        if reason:
            header += f"Fail reason: {reason}\n"
        if expected:
            header += f"Expected: {expected[:500]}\n"
        if resp:
            header += f"Agent response: {resp[:500]}\n"
        if conv:
            header += f"Conversation:\n{fmt_trajectory(conv)}\n"
        traj_parts.append(header)

    trajectories_text = "\n\n---\n\n".join(traj_parts)

    user_msg = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Edit Budget\nProduce at most {edit_budget} edits.\n\n"
        f"## Failed Trajectories ({len(failures)} total)\n{trajectories_text}"
    )

    try:
        response = await provider.chat(
            messages=[
                {"role": "system", "content": ERROR_ANALYST_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            model=model,
            max_tokens=4096,
            temperature=0.3,
        )
        result = _extract_json(response.content or "")
        if result and "patch" in result:
            result["source_type"] = "failure"
            return result
        logger.warning("Error analyst returned no valid JSON patch")
    except Exception:
        logger.error(f"Error analyst failed: {traceback.format_exc()}")
    return None


async def run_success_analyst(
    provider: "LLMProvider",
    model: str,
    skill_content: str,
    results: list[RolloutResult | dict],
    edit_budget: int = 4,
) -> dict | None:
    """Analyze successful trajectories and propose reinforcement edits."""
    successes = []
    for r in results:
        r_dict = r.to_dict() if isinstance(r, RolloutResult) else r
        if r_dict.get("hard") and float(r_dict.get("hard", 0)) > 0.5:
            successes.append(r_dict)

    if not successes:
        return None

    traj_parts: list[str] = []
    for i, item in enumerate(successes[:5], 1):
        task = item.get("task", "")
        resp = item.get("response", "")
        conv = item.get("conversation", [])

        header = f"### Task {i} (id={item.get('id', '?')})\nTask: {task}\n"
        if resp:
            header += f"Agent response: {resp[:300]}\n"
        if conv:
            header += f"Conversation:\n{fmt_trajectory(conv)}\n"
        traj_parts.append(header)

    trajectories_text = "\n\n---\n\n".join(traj_parts)

    user_msg = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Edit Budget\nProduce at most {edit_budget} edits.\n\n"
        f"## Successful Trajectories ({len(successes)} total)\n{trajectories_text}"
    )

    try:
        response = await provider.chat(
            messages=[
                {"role": "system", "content": SUCCESS_ANALYST_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            model=model,
            max_tokens=4096,
            temperature=0.3,
        )
        result = _extract_json(response.content or "")
        if result and "patch" in result:
            result["source_type"] = "success"
            return result
        logger.warning("Success analyst returned no valid JSON patch")
    except Exception:
        logger.error(f"Success analyst failed: {traceback.format_exc()}")
    return None


def _split_minibatches(items: list, batch_size: int) -> list[list]:
    """Split items into minibatches of at most *batch_size*."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


async def run_reflect(
    provider: "LLMProvider",
    model: str,
    skill_content: str,
    results: list[RolloutResult | dict],
    edit_budget: int = 4,
    failure_only: bool = False,
    minibatch_size: int = 8,
) -> list[dict]:
    """Full reflect stage: analyze failure + success trajectories.

    Splits failures/successes into minibatches and runs one analyst call
    per minibatch, then collects all patches.

    Returns a list of patch dicts (each with ``source_type``).
    """
    # Separate failure / success
    failures: list[dict] = []
    successes: list[dict] = []
    for r in results:
        r_dict = r.to_dict() if isinstance(r, RolloutResult) else r
        if not r_dict.get("hard") or float(r_dict.get("hard", 0)) < 1e-9:
            failures.append(r_dict)
        elif not failure_only:
            successes.append(r_dict)

    patches: list[dict] = []

    # Split failures into minibatches
    fail_batches = _split_minibatches(failures, minibatch_size)
    for batch in fail_batches:
        patch = await run_error_analyst(provider, model, skill_content, batch, edit_budget)
        if patch:
            patches.append(patch)

    # Split successes into minibatches
    if not failure_only:
        succ_batches = _split_minibatches(successes, minibatch_size)
        for batch in succ_batches:
            patch = await run_success_analyst(provider, model, skill_content, batch, edit_budget)
            if patch:
                patches.append(patch)

    return patches
