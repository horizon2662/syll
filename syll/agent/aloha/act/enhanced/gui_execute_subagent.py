"""GUI Execute Sub-Agent: isolated agent that executes a single plan step.

Composes the existing ``AlohaPlanner``, ``AlohaExecutor``, and
``ActionVerifier`` to form a closed TVAE loop per step.  Each sub-agent
run has its own context window — the main agent's context is not polluted.

Reference: CoACT-1 (OSWorld SOTA 56.4%) — Orchestrator delegates to
a GUI Operator for execution.

Does **not** modify the original ``subagent.py``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from syll.agent.aloha.act.enhanced.action_verifier import (
    ActionVerifier,
    VerifyResult,
    VerifyStatus,
)
from syll.agent.aloha.act.enhanced.config import EnhancedConfig
from syll.agent.aloha.act.enhanced.plan_manager import (
    ExecutionPlan,
    PlanManager,
)

if TYPE_CHECKING:
    from syll.agent.aloha.act.executor import AlohaExecutor
    from syll.agent.aloha.act.planner import AlohaPlanner
    from syll.agent.aloha.act.verified_planner import VerifiedPlanner


@dataclass
class StepResult:
    """Structured result returned by a sub-agent after executing one step."""

    step_index: int
    action_success: bool  # executor did not throw
    verify_result: VerifyResult | None = None
    action: str = ""
    expectation: str = ""
    diagnosis: str = ""
    retries_used: int = 0


class GUIExecuteSubAgent:
    """Execute a single plan step with TVAE verification in an isolated context.

    Usage::

        sub = GUIExecuteSubAgent(
            planner=verified_planner,
            executor=aloha_executor,
            verifier=action_verifier,
            plan_manager=plan_manager,
            config=enhanced_config,
        )
        result = await sub.execute_step(plan, step_index=3)
        # result.verify_result tells you whether the action really worked
    """

    def __init__(
        self,
        planner: AlohaPlanner | VerifiedPlanner,
        executor: AlohaExecutor,
        verifier: ActionVerifier,
        plan_manager: PlanManager,
        config: EnhancedConfig | None = None,
        screenshot_fn=None,
    ):
        """
        Args:
            planner: The planner to use (VerifiedPlanner recommended).
            executor: The GUI executor (original AlohaExecutor is fine).
            verifier: Action verifier for TVAE checks.
            plan_manager: For reading/writing plan state.
            config: Feature flags and thresholds.
            screenshot_fn: Async callable that returns a base64 screenshot.
                If None, uses ``pyautogui.screenshot`` internally.
        """
        self.planner = planner
        self.executor = executor
        self.verifier = verifier
        self.plan_manager = plan_manager
        self.config = config or EnhancedConfig()
        self._screenshot_fn = screenshot_fn or self._default_screenshot

    async def execute_step(
        self,
        plan: ExecutionPlan,
        step_index: int,
        task: str = "",
        guidance_trajectory: str = "",
    ) -> StepResult:
        """Execute one plan step with verification and optional retry.

        The flow is:
        1. Read the step description from the plan.
        2. Take a *before* screenshot.
        3. Call the planner to decide the concrete action.
        4. Execute the action via the executor.
        5. Wait briefly, take an *after* screenshot.
        6. Run pixel-diff verification.
        7. If NO_CHANGE and retries remain → retry with recovery prompt.
        8. Return a structured ``StepResult``.
        """
        step = self.plan_manager.get_current_step(plan)
        if step is None:
            return StepResult(
                step_index=step_index,
                action_success=False,
                diagnosis="No pending steps found in plan.",
            )

        max_retries = self.config.max_consecutive_failures
        previous_verify: VerifyResult | None = None

        for attempt in range(max_retries + 1):
            # ---- 1. Before screenshot ----
            screenshot_before = await self._take_screenshot()

            # ---- 2. Plan ----
            plan_kwargs: dict = dict(
                task=task or step.description,
                guidance_trajectory=guidance_trajectory,
                screenshot_b64=screenshot_before,
            )

            # Inject previous verification failure if retrying
            if previous_verify is not None and hasattr(
                self.planner, "plan"
            ):
                try:
                    plan_kwargs["previous_verify_result"] = previous_verify
                except TypeError:
                    pass  # Original planner doesn't accept this kwarg

            plan_output = await self.planner.plan(**plan_kwargs)

            action_str = plan_output.get("Action", "")
            expectation = plan_output.get("Expectation", "")

            if not action_str or action_str.lower() == "none":
                return StepResult(
                    step_index=step_index,
                    action_success=False,
                    action=action_str,
                    expectation=expectation,
                    diagnosis="Planner returned null action (task complete or stuck).",
                    retries_used=attempt,
                )

            # ---- 3. Execute ----
            action_dict = self._parse_action(action_str)
            success, msg = await self.executor.execute(action_dict)

            if not success:
                previous_verify = VerifyResult(
                    status=VerifyStatus.NO_CHANGE,
                    confidence=1.0,
                    pixel_diff_score=0.0,
                    diagnosis=f"Executor error: {msg}",
                )
                continue

            # ---- 4. After screenshot (with delay) ----
            await asyncio.sleep(self.config.screenshot_delay_seconds)
            screenshot_after = await self._take_screenshot()

            # ---- 5. Verify ----
            verify = self.verifier.verify_pixel_diff(
                screenshot_before, screenshot_after
            )

            if verify.status == VerifyStatus.SUCCESS:
                self.plan_manager.update_step(
                    plan, step.index, "DONE", result="OK"
                )
                return StepResult(
                    step_index=step_index,
                    action_success=True,
                    verify_result=verify,
                    action=action_str,
                    expectation=expectation,
                    retries_used=attempt,
                )

            # NO_CHANGE or UNCERTAIN → retry
            previous_verify = verify
            logger.warning(
                f"Step {step_index} attempt {attempt + 1}: "
                f"{verify.status.value} — {verify.diagnosis}"
            )

        # All retries exhausted
        self.plan_manager.update_step(
            plan,
            step.index,
            "FAILED",
            result=previous_verify.diagnosis if previous_verify else "Max retries",
        )
        return StepResult(
            step_index=step_index,
            action_success=False,
            verify_result=previous_verify,
            action=plan_output.get("Action", ""),
            expectation=plan_output.get("Expectation", ""),
            diagnosis=(
                f"Failed after {max_retries} retries. "
                f"Last diagnosis: {previous_verify.diagnosis if previous_verify else 'N/A'}"
            ),
            retries_used=max_retries,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _default_screenshot() -> str:
        """Take a screenshot using pyautogui and return base64."""
        import base64
        import io

        import pyautogui
        from PIL import Image

        img = pyautogui.screenshot()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    async def _take_screenshot(self) -> str:
        """Take a screenshot, handling both sync and async callables."""
        import asyncio
        import inspect

        result = self._screenshot_fn()
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _parse_action(action_str: str) -> dict:
        """Try to parse an action string into a dict for the executor.

        The planner returns free-text actions; the executor expects a
        dict with ``action``, ``position``, ``value`` keys.  If parsing
        fails we return a minimal dict so the executor can report the
        issue.
        """
        import json
        import re

        # Already JSON?
        try:
            d = json.loads(action_str)
            if isinstance(d, dict):
                return d
        except (json.JSONDecodeError, ValueError):
            pass

        # Try to extract action type and coordinates from text like
        # "Click on the File menu at (500, 30)"
        click_match = re.search(
            r"click\s+.*?\(?\s*(\d+)\s*,\s*(\d+)\s*\)?",
            action_str,
            re.IGNORECASE,
        )
        if click_match:
            return {
                "action": "click",
                "position": [
                    int(click_match.group(1)),
                    int(click_match.group(2)),
                ],
                "value": "",
            }

        # Fallback: pass as a description for the executor to handle
        return {"action": "ERROR", "value": f"Unparseable action: {action_str}"}
