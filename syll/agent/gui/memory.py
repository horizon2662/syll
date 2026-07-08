"""Unified GUI memory facade.

GUI-related memory is split across several scopes. This facade gives L1/L2/L3
a single entry point while keeping the underlying ``MemoryStore`` abstraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from syll.agent.aloha.act.enhanced.structured_memory import StructuredMemory
from syll.agent.gui_failure_ledger import GuiAttemptLedger
from syll.agent.longhorizon.skill_memory import SkillMemory


class GuiMemory:
    """Access point for all GUI-scoped memory in a workspace/session."""

    def __init__(self, workspace: Path, session_key: str | None = None):
        self.workspace = Path(workspace)
        self.session_key = session_key
        # Per-session/run execution history and failure ledger.
        self.execution_history = (
            StructuredMemory(self.workspace) if session_key else None
        )
        self.ledger = (
            GuiAttemptLedger(session_key) if session_key else None
        )

    # ------------------------------------------------------------------
    # Skill memory (per-skill procedural memory)
    # ------------------------------------------------------------------
    def skill(self, skill_name: str) -> SkillMemory:
        """Return the skill memory for ``skill_name``."""
        return SkillMemory(self.workspace, skill_name)

    # ------------------------------------------------------------------
    # Execution history
    # ------------------------------------------------------------------
    def record_step(
        self,
        index: int,
        action: str,
        expectation: str,
        verify_status: str,
        diagnosis: str = "",
        category: str = "",
    ) -> None:
        """Record one GUI step into the run execution history."""
        if self.execution_history is None:
            return
        self.execution_history.record_step(
            index=index,
            action=action,
            expectation=expectation,
            verify_status=verify_status,
            diagnosis=diagnosis,
            category=category,
        )

    def get_execution_context(self, max_recent_steps: int = 3) -> str:
        """Return compact execution context for the planner."""
        if self.execution_history is None:
            return ""
        return self.execution_history.get_execution_context(max_recent_steps)

    # ------------------------------------------------------------------
    # Failure ledger
    # ------------------------------------------------------------------
    def record_failure(
        self, *, instruction: str, kind: Any, reason: str, step: int = 0
    ) -> None:
        if self.ledger is None:
            return
        self.ledger.record_failure(
            instruction=instruction, kind=kind, reason=reason, step=step
        )

    def record_success(self, *, instruction: str) -> None:
        if self.ledger is None:
            return
        self.ledger.record_success(instruction=instruction)

    def is_locked(self, instruction: str) -> bool:
        if self.ledger is None:
            return False
        return self.ledger.is_locked(instruction)
