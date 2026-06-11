"""Syll-specific rollout: use AgentLoop to evaluate a skill on tasks.

This is the Syll adaptation of SkillOpt's environment-agnostic rollout.
Instead of benchmark environments, we use Syll's own AgentLoop to execute
tasks and evaluate results.
"""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from loguru import logger

from syll.skill_optimizer.types import RolloutResult

if TYPE_CHECKING:
    from pathlib import Path

    from syll.agent.loop import AgentLoop
    from syll.providers.base import LLMProvider


# ── Skill injection ────────────────────────────────────────────────────────

class SkillInjector:
    """Temporarily writes a skill to the workspace so AgentLoop picks it up.

    The AgentLoop's ContextBuilder reads skills from disk when building the
    system prompt.  During training the skill content changes in memory, so
    we must write the current/candidate skill to the workspace skill path
    **before** each rollout batch and restore the original afterwards.

    Usage::

        injector = SkillInjector(skill_path)
        injector.swap(candidate_skill)
        # ... run rollout ...
        injector.restore()
    """

    def __init__(self, skill_path: "Path") -> None:
        self.skill_path = skill_path
        self._original: str | None = None
        self._swapped = False

    def swap(self, new_content: str) -> None:
        """Save original and write new content to skill path."""
        if self.skill_path.exists():
            self._original = self.skill_path.read_text(encoding="utf-8")
        else:
            self._original = None
        self.skill_path.parent.mkdir(parents=True, exist_ok=True)
        self.skill_path.write_text(new_content, encoding="utf-8")
        self._swapped = True

    def restore(self) -> None:
        """Restore the original skill content."""
        if not self._swapped:
            return
        if self._original is not None:
            self.skill_path.write_text(self._original, encoding="utf-8")
        elif self.skill_path.exists():
            # Original didn't exist — remove the file we created
            try:
                self.skill_path.unlink()
            except OSError:
                pass
        self._swapped = False


# ── Task evaluation ────────────────────────────────────────────────────────

async def evaluate_task(
    agent_loop: "AgentLoop",
    task: str,
    expected: str = "",
    task_id: str = "",
) -> RolloutResult:
    """Execute a single task through the agent and evaluate the result.

    Parameters
    ----------
    agent_loop : AgentLoop
        The Syll agent loop to use for execution.
    task : str
        The task prompt to send to the agent.
    expected : str
        Optional expected answer for evaluation.
    task_id : str
        Optional task identifier.

    Returns
    -------
    RolloutResult
    """
    try:
        result = await agent_loop.process_direct(
            task,
            session_key=f"skill_opt:{task_id}",
        )
        response_text = result.text or ""

        # Simple evaluation: check if response matches expected answer
        hard = 0
        soft = 0.0
        fail_reason = ""

        if expected and expected.strip():
            expected_lower = expected.strip().lower()
            response_lower = response_text.strip().lower()

            # Exact match
            if expected_lower in response_lower:
                hard = 1
                soft = 1.0
            else:
                # Partial credit: word overlap
                expected_words = set(expected_lower.split())
                response_words = set(response_lower.split())
                if expected_words:
                    overlap = len(expected_words & response_words)
                    soft = overlap / len(expected_words)
                    hard = 0
                if soft < 0.1:
                    fail_reason = "no_match: response does not contain expected answer"
                else:
                    fail_reason = f"partial_match: {soft:.0%} word overlap"
        else:
            # No expected answer — check for non-empty response
            if response_text.strip():
                hard = 1
                soft = 1.0
            else:
                fail_reason = "empty_response"

        return RolloutResult(
            id=task_id,
            hard=hard,
            soft=soft,
            task=task,
            response=response_text[:2000],
            expected=expected[:500],
            fail_reason=fail_reason,
        )

    except Exception as e:
        logger.error(f"Task {task_id} rollout failed: {e}")
        return RolloutResult(
            id=task_id,
            hard=0,
            soft=0.0,
            task=task,
            fail_reason=f"error: {str(e)}",
        )


async def rollout_batch(
    agent_loop: "AgentLoop",
    tasks: list[dict],
    max_concurrent: int = 4,
) -> list[RolloutResult]:
    """Execute a batch of tasks and return results.

    Parameters
    ----------
    agent_loop : AgentLoop
        The Syll agent loop.
    tasks : list[dict]
        Task dicts with keys: ``id``, ``task``, ``expected`` (optional).
    max_concurrent : int
        Maximum concurrent task evaluations.

    Returns
    -------
    list[RolloutResult]
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _eval(task: dict) -> RolloutResult:
        async with semaphore:
            return await evaluate_task(
                agent_loop,
                task=task.get("task", ""),
                expected=task.get("expected", ""),
                task_id=task.get("id", ""),
            )

    coros = [_eval(t) for t in tasks]
    results = await asyncio.gather(*coros)
    return list(results)


def load_tasks(path: str | "Path") -> list[dict]:
    """Load tasks from a JSON file.

    Expected format:
    ```json
    [
      {"id": "1", "task": "Find the file report.docx", "expected": "/path/to/report.docx"},
      ...
    ]
    ```
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array of tasks, got {type(data).__name__}")
    # Validate required fields
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Task {i} must be a dict, got {type(item).__name__}")
        if "task" not in item:
            raise ValueError(f"Task {i} missing required 'task' field")
        if "id" not in item:
            item["id"] = str(i)
    return data
