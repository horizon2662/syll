"""Plan persistence manager: save / update / read execution plans as files.

This allows a Plan Agent and an Execute Sub-Agent to share state via the
filesystem instead of through a shared context window.

Reference: MGA (WSDM'25) — structured memory for decoupling decisions
from historical inertia.

Does **not** modify the original ``TrajectoryManager``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from loguru import logger


StepStatus = Literal["PENDING", "CURRENT", "DONE", "FAILED", "SKIPPED"]


@dataclass
class PlanStep:
    """One step in an execution plan."""

    index: int  # 1-based
    description: str
    action: str = ""
    expectation: str = ""
    status: StepStatus = "PENDING"
    result: str = ""  # outcome or diagnosis


@dataclass
class ExecutionPlan:
    """A complete execution plan for a GUI task."""

    task: str
    skill_name: str
    steps: list[PlanStep] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


class PlanManager:
    """Manage execution plans as markdown files.

    File layout::

        workspace/aloha_skills/{skill_name}/execution_plan.md

    The markdown format is human-readable and easy for sub-agents to
    parse without a custom format.

    Usage::

        pm = PlanManager(workspace=Path("~/.syll/workspace"))
        plan = pm.create_plan("my_task", "Open Chrome and search X", steps)
        pm.save_plan(plan)

        # After executing step 1:
        pm.update_step(plan, step_index=1, status="DONE", result="OK")

        # Sub-agent reads current step:
        step = pm.get_current_step(plan)
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace

    # ------------------------------------------------------------------
    # Create / Save / Load
    # ------------------------------------------------------------------

    def create_plan(
        self, skill_name: str, task: str, step_descriptions: list[str]
    ) -> ExecutionPlan:
        """Create a new ExecutionPlan from a list of step descriptions."""
        steps = [
            PlanStep(index=i + 1, description=desc)
            for i, desc in enumerate(step_descriptions)
        ]
        if steps:
            steps[0].status = "CURRENT"

        now = datetime.now(timezone.utc).isoformat()
        return ExecutionPlan(
            task=task,
            skill_name=skill_name,
            steps=steps,
            created_at=now,
            updated_at=now,
        )

    def save_plan(self, plan: ExecutionPlan) -> Path:
        """Write the plan to a markdown file and return its path."""
        plan_path = self._plan_path(plan.skill_name)
        plan_path.parent.mkdir(parents=True, exist_ok=True)

        plan.updated_at = datetime.now(timezone.utc).isoformat()

        lines = [
            f"# Execution Plan: {plan.skill_name}",
            "",
            f"**Task**: {plan.task}",
            f"**Created**: {plan.created_at}",
            f"**Updated**: {plan.updated_at}",
            "",
            "## Steps",
            "",
        ]

        for step in plan.steps:
            status_icon = {
                "PENDING": "⬜",
                "CURRENT": "🔵",
                "DONE": "✅",
                "FAILED": "❌",
                "SKIPPED": "⏭️",
            }.get(step.status, "⬜")
            lines.append(f"### {status_icon} Step {step.index}: {step.description}")
            if step.action:
                lines.append(f"- **Action**: {step.action}")
            if step.expectation:
                lines.append(f"- **Expectation**: {step.expectation}")
            if step.result:
                lines.append(f"- **Result**: {step.result}")
            lines.append("")

        # Add summary footer
        done = sum(1 for s in plan.steps if s.status == "DONE")
        failed = sum(1 for s in plan.steps if s.status == "FAILED")
        total = len(plan.steps)
        lines.extend(
            [
                "---",
                "",
                f"**Progress**: {done}/{total} done, {failed} failed",
                "",
            ]
        )

        plan_path.write_text("\n".join(lines), encoding="utf-8")
        logger.debug(
            f"Plan saved to {plan_path} ({done}/{total} steps done)"
        )
        return plan_path

    def load_plan(self, skill_name: str) -> ExecutionPlan | None:
        """Load an execution plan from file."""
        plan_path = self._plan_path(skill_name)
        if not plan_path.exists():
            return None

        try:
            return self._parse_plan_file(plan_path, skill_name)
        except Exception as exc:
            logger.warning(f"Failed to parse plan for '{skill_name}': {exc}")
            return None

    # ------------------------------------------------------------------
    # Update / Query
    # ------------------------------------------------------------------

    def update_step(
        self,
        plan: ExecutionPlan,
        step_index: int,
        status: StepStatus,
        result: str = "",
    ) -> None:
        """Update the status of a specific step and auto-advance CURRENT."""
        for step in plan.steps:
            if step.index == step_index:
                step.status = status
                step.result = result
                break

        # Auto-advance: set next PENDING step to CURRENT
        if status in ("DONE", "FAILED", "SKIPPED"):
            for step in plan.steps:
                if step.status == "PENDING":
                    step.status = "CURRENT"
                    break

        plan.updated_at = datetime.now(timezone.utc).isoformat()

    def get_current_step(self, plan: ExecutionPlan) -> PlanStep | None:
        """Return the first step with status CURRENT or the first PENDING."""
        for step in plan.steps:
            if step.status == "CURRENT":
                return step
        for step in plan.steps:
            if step.status == "PENDING":
                return step
        return None

    def get_plan_summary(self, plan: ExecutionPlan) -> str:
        """Return a compact text summary for injection into a planner prompt.

        This replaces the full action_history with a compressed version
        that preserves progress awareness without the token cost.
        """
        done = [s for s in plan.steps if s.status == "DONE"]
        failed = [s for s in plan.steps if s.status == "FAILED"]
        current = self.get_current_step(plan)

        lines = [
            f"Plan progress: {len(done)}/{len(plan.steps)} steps completed.",
        ]
        if current:
            lines.append(f"Current step: {current.index}. {current.description}")
        if failed:
            fail_desc = ", ".join(f"Step {s.index}" for s in failed)
            lines.append(f"Failed steps: {fail_desc}")
        return " ".join(lines)

    def is_complete(self, plan: ExecutionPlan) -> bool:
        """True if all steps are DONE, FAILED, or SKIPPED."""
        return all(
            s.status in ("DONE", "FAILED", "SKIPPED") for s in plan.steps
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _plan_path(self, skill_name: str) -> Path:
        return (
            self.workspace
            / "aloha_skills"
            / skill_name
            / "execution_plan.md"
        )

    def _parse_plan_file(
        self, path: Path, skill_name: str
    ) -> ExecutionPlan:
        """Parse a plan markdown file back into an ExecutionPlan."""
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")

        task = ""
        created_at = ""
        updated_at = ""
        steps: list[PlanStep] = []
        current_step: PlanStep | None = None

        for line in lines:
            stripped = line.strip()

            # Extract task
            if stripped.startswith("**Task**"):
                task = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("**Created**"):
                created_at = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("**Updated**"):
                updated_at = stripped.split(":", 1)[1].strip()

            # Step header: "### ✅ Step 1: Description" etc.
            step_match = re.match(
                r"###\s+[^\s]+\s+Step\s+(\d+):\s+(.*)", stripped
            )
            if step_match:
                if current_step:
                    steps.append(current_step)
                idx = int(step_match.group(1))
                desc = step_match.group(2)
                status = self._icon_to_status(stripped)
                current_step = PlanStep(
                    index=idx, description=desc, status=status
                )
            elif current_step:
                if stripped.startswith("- **Action**"):
                    current_step.action = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("- **Expectation**"):
                    current_step.expectation = stripped.split(":", 1)[
                        1
                    ].strip()
                elif stripped.startswith("- **Result**"):
                    current_step.result = stripped.split(":", 1)[1].strip()

        if current_step:
            steps.append(current_step)

        return ExecutionPlan(
            task=task,
            skill_name=skill_name,
            steps=steps,
            created_at=created_at,
            updated_at=updated_at,
        )

    @staticmethod
    def _icon_to_status(line: str) -> StepStatus:
        if "✅" in line:
            return "DONE"
        if "❌" in line:
            return "FAILED"
        if "🔵" in line:
            return "CURRENT"
        if "⏭️" in line:
            return "SKIPPED"
        return "PENDING"
