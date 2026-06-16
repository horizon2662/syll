"""Hierarchical plan manager: milestones -> steps, with checkpoint/resume.

Extends ``PlanManager`` (agent/aloha/act/enhanced/plan_manager.py), which
stores a FLAT list of steps. For long-horizon tasks a flat list becomes
unmanageable and offers no resume across crashes or context compaction.

Two additions:
- **Milestones** (hierarchical planning): a plan is a list of milestones,
  each with its own steps. The main agent plans milestones; a subagent
  expands a milestone into steps. (TaskWeave FPDA: propagate intent, then
  partition into tasks; Mobile-Agent-v3 Manager subgoal list.)
- **Checkpoint/resume**: the milestone + step statuses are persisted to
  ``session_state.json`` so a multi-hour task resumes from the last
  checkpoint. (Anthropic context engineering; orchestration survey State
  unit, arXiv:2601.13671.)

The parent's markdown ``execution_plan.md`` format is reused unchanged, so
the existing GUIExecuteSubAgent can still read it. Milestones are encoded as
``[MILESTONE i]`` step headers.

Does **not** modify the original ``plan_manager.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from loguru import logger

from syll.agent.aloha.act.enhanced.plan_manager import (
    ExecutionPlan,
    PlanManager,
    PlanStep,
)

MilestoneStatus = Literal["PENDING", "CURRENT", "DONE", "FAILED", "SKIPPED"]


@dataclass
class Milestone:
    """A high-level milestone: a title + its own ordered steps."""

    index: int  # 1-based
    title: str
    step_descriptions: list[str] = field(default_factory=list)
    status: MilestoneStatus = "PENDING"


class HierarchicalPlanManager(PlanManager):
    """Milestone -> step planning with checkpoint/resume.

    Usage::

        hpm = HierarchicalPlanManager(workspace, skill_name="deploy_app")
        plan = hpm.create_milestone_plan(
            task="Deploy the app to prod",
            milestones=[
                ("Run tests", ["pytest", "lint"]),
                ("Build image", ["docker build", "push to registry"]),
            ],
        )
        hpm.save_plan(plan)
        hpm.checkpoint(plan)
        # ... crash / context compaction ...
        resumed = hpm.resume()   # pick up where it left off
    """

    def __init__(self, workspace: Path, skill_name: str):
        super().__init__(workspace)
        self.skill_name = skill_name
        self.session_file = (
            workspace / "aloha_skills" / skill_name / "session_state.json"
        )

    # ------------------------------------------------------------------
    # milestone-structured plan
    # ------------------------------------------------------------------
    def create_milestone_plan(
        self,
        task: str,
        milestones: list[Milestone] | list[tuple[str, list[str]]],
    ) -> ExecutionPlan:
        """Build a plan from milestones. ``milestones`` may be either a list
        of ``Milestone`` or a list of ``(title, [step_desc, ...])`` tuples.

        Milestones are flattened into the parent's step list with an
        ``[MILESTONE i]`` header so the hierarchy is recoverable and the
        existing ``execution_plan.md`` readers keep working."""
        flat: list[str] = []
        for m_idx, m in enumerate(milestones, 1):
            if isinstance(m, Milestone):
                title, steps = m.title, m.step_descriptions
            else:
                title, steps = m
            flat.append(f"[MILESTONE {m_idx}] {title}")
            flat.extend(f"  - {s}" for s in steps)
        return self.create_plan(self.skill_name, task, flat)

    def current_milestone(self, plan: ExecutionPlan) -> int:
        """Return the 1-based index of the milestone the current step belongs
        to, or 0 if the plan is complete. A milestone is current while any of
        its steps are PENDING/CURRENT/FAILED."""
        active_idx = 0
        m_idx = 0
        for step in plan.steps:
            if step.description.lstrip().startswith("[MILESTONE"):
                m_idx += 1
            if step.status in ("PENDING", "CURRENT", "FAILED"):
                active_idx = m_idx
                break
        return active_idx

    # ------------------------------------------------------------------
    # checkpoint / resume (the long-horizon piece)
    # ------------------------------------------------------------------
    def checkpoint(self, plan: ExecutionPlan) -> Path:
        """Persist plan state so a crashed / compacted session can resume."""
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "skill": self.skill_name,
            "task": plan.task,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "steps": [
                {
                    "index": s.index,
                    "description": s.description,
                    "status": s.status,
                    "result": s.result,
                }
                for s in plan.steps
            ],
        }
        self.session_file.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.debug(f"checkpoint saved: {self.session_file}")
        return self.session_file

    def resume(self) -> ExecutionPlan | None:
        """Load the last checkpointed plan, or ``None`` if none exists."""
        if not self.session_file.exists():
            return None
        try:
            state = json.loads(self.session_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"resume failed for '{self.skill_name}': {exc}")
            return None
        plan = ExecutionPlan(
            task=state.get("task", ""),
            skill_name=self.skill_name,
            steps=[],
        )
        for sd in state.get("steps", []):
            plan.steps.append(
                PlanStep(
                    index=sd["index"],
                    description=sd["description"],
                    status=sd.get("status", "PENDING"),  # type: ignore[arg-type]
                    result=sd.get("result", ""),
                )
            )
        return plan

    # ------------------------------------------------------------------
    # diagnosis -> Align hook (TaskWeave FPDA)
    # ------------------------------------------------------------------
    def align(
        self,
        plan: ExecutionPlan,
        failed_step_index: int,
        diagnosis: str,
        *,
        new_steps: list[str] | None = None,
    ) -> ExecutionPlan:
        """Re-plan after a failure: mark the failed step, optionally splice in
        replacement steps, and re-checkpoint. This is the Align half of
        TaskWeave's FPDA -- the main agent consumes a subagent's ``diagnosis``
        and adjusts the plan rather than silently retrying."""
        for step in plan.steps:
            if step.index == failed_step_index:
                step.status = "FAILED"
                step.result = diagnosis
                break

        if new_steps:
            # Splice replacement steps right after the failed one.
            plan = self._splice_steps(plan, failed_step_index, new_steps)

        self.save_plan(plan)
        self.checkpoint(plan)
        return plan

    def _splice_steps(
        self, plan: ExecutionPlan, after_index: int, new_steps: list[str]
    ) -> ExecutionPlan:
        # ``after_index`` is the FAILED step's 1-based ``.index``. Convert it to
        # the step's 0-based list position so the [REPLAN] steps are spliced
        # immediately AFTER it (not after the wrong offset).
        pos = next(
            (i for i, s in enumerate(plan.steps) if s.index == after_index),
            len(plan.steps) - 1,
        )
        for offset, desc in enumerate(new_steps):
            plan.steps.insert(
                pos + 1 + offset,
                PlanStep(index=-(pos + offset), description=f"[REPLAN] {desc}"),
            )
        # Re-index sequentially (parent parsers expects 1-based order).
        for i, step in enumerate(plan.steps, 1):
            step.index = i
        return plan
