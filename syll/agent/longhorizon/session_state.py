"""Session-level checkpoint/resume (the long-horizon resume layer).

``HierarchicalPlanManager`` checkpoints the *plan* (milestones + steps).
This module checkpoints the *session*: which milestone/step the orchestrator
is on, which subagent runs are in flight, and the last compaction summary.
Together they let a multi-hour task resume cleanly after a crash or a
context compaction (Anthropic context engineering; orchestration survey
State unit, arXiv:2601.13671).

State is one JSON file per skill:
``workspace/aloha_skills/{skill}/session_state.json``
(plan state lives alongside in ``session_state.json`` via the plan manager;
this file adds the orchestrator cursor + run bookkeeping).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


class SessionState:
    """Orchestrator cursor + in-flight subagent bookkeeping."""

    def __init__(self, workspace: Path, skill: str):
        self.skill = skill
        self.dir = workspace / "aloha_skills" / skill
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "orchestrator_state.json"

    def checkpoint(
        self,
        *,
        current_milestone: int,
        current_step_index: int,
        running_runs: list[str],
        compaction_summary: str = "",
        extra: dict[str, Any] | None = None,
    ) -> Path:
        state = {
            "skill": self.skill,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "current_milestone": current_milestone,
            "current_step_index": current_step_index,
            "running_runs": running_runs,
            "compaction_summary": compaction_summary,
            "extra": extra or {},
        }
        self.file.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.file

    def resume(self) -> dict[str, Any] | None:
        if not self.file.exists():
            return None
        try:
            return json.loads(self.file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"session_state[{self.skill}] resume failed: {exc}")
            return None

    def clear(self) -> None:
        if self.file.exists():
            self.file.unlink()
