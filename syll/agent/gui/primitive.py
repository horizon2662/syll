"""L1 GUI primitive: one screenshot + one action + one verify.

This module is the canonical home for the single-step GUI pipeline
(capture → plan → ground → verify).  It is intentionally decoupled from
multi-step loop policy: callers (``EnhancedAlohaPlannerTool.execute``,
``GuiActionTool.execute``, or future L2/L3 planners) drive iteration and
retry while the primitive executes exactly one attempt.

The primitive receives a *driver* object that supplies tool-specific
operations (screenshot capture, actor calls, coordinate transform, audit
logging, monitor writes, etc.).  ``EnhancedAlohaPlannerTool`` and its
subclasses act as the driver today; in the future a dedicated primitive
driver can be plugged in without changing this module.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from syll.agent.aloha.act.enhanced.action_verifier import (
    ActionVerifier,
    VerifyResult,
    VerifyStatus,
)
from syll.agent.aloha.act.enhanced.step_context import (
    ExecuteContext,
    FailedAttempt,
    FailureCategory,
    StepContext,
)
from syll.agent.tools.base import ToolResult


class GuiPrimitive:
    """Stateless single-step GUI primitive.

    Wraps a driver and exposes exactly one capture→plan→ground→verify
    attempt per call.  All cross-step state lives in ``exec_ctx`` /
    ``step_ctx``; this class does not retain step history.
    """

    def __init__(self, driver: Any) -> None:
        self.driver = driver

    # ------------------------------------------------------------------
    # L1 primitive helpers (mirrored from EnhancedAlohaPlannerTool)
    # ------------------------------------------------------------------

    async def run_llm_verify(
        self, *, after_b64: str, expectation: str, verifier: ActionVerifier,
        actor_model: str, attempt: int
    ) -> tuple[VerifyResult | None, bool]:
        """LLM semantic verify. pixel-diff already confirmed the screen
        changed; this checks it changed INTO the expected state. Returns
        ``(result, should_retry)``; result is None when skipped/errored.
        """
        try:
            result = await verifier.verify_with_expectation(
                after_b64, expectation,
                model=actor_model,
            )
        except Exception as exc:
            logger.debug(f"  LLM verify skipped: {exc}")
            return None, False
        if result.is_no_change:
            logger.warning(
                f"  LLM verify NO_CHANGE (attempt {attempt + 1}): {result.diagnosis}"
            )
            return result, True
        logger.info(f"  LLM verify: {result.status.value}")
        return result, False

    async def capture_observation(
        self, exec_ctx: ExecuteContext, step_ctx: StepContext, shot_idx: int
    ) -> tuple[int, str | None]:
        """Capture before-screenshot + spatial analysis into step_ctx.

        Returns ``(new_shot_idx, error_msg)``; error_msg non-None → caller
        returns it (screenshot failed).
        """
        shot_idx += 1
        screenshot_path = await self.driver._take_screenshot(shot_idx)
        if not screenshot_path:
            return shot_idx, "Error: Failed to capture screenshot"
        exec_ctx.screenshots.append(screenshot_path)
        with open(screenshot_path, "rb") as f:
            step_ctx.screenshot_b64 = base64.b64encode(f.read()).decode()
        step_ctx.screenshot_path = screenshot_path
        if exec_ctx.spatial_analyzer:
            try:
                step_ctx.spatial_context = await exec_ctx.spatial_analyzer.analyze(
                    step_ctx.screenshot_b64
                )
            except Exception as exc:
                logger.debug(f"Spatial analysis skipped: {exc}")
        return shot_idx, None

    async def verify_step(
        self,
        exec_ctx: ExecuteContext,
        step_ctx: StepContext,
        *,
        shot_idx: int,
        plan_action: str,
        plan_output: dict,
        failed_attempts: list[FailedAttempt],
        last_verify_result: VerifyResult | None,
        actor_model: str,
    ) -> tuple[bool, int, VerifyResult | None, VerifyResult | None]:
        """TVAE verification of one executed action.

        Takes the after-screenshot, runs pixel-diff (+ LLM diagnose on
        NO_CHANGE + LLM verify on SUCCESS), records failed edges.

        Returns ``(should_retry, new_shot_idx, step_verify, last_verify_result)``.
        ``should_retry=True`` → caller continues the retry loop.
        """
        cfg = exec_ctx.cfg
        verifier = exec_ctx.verifier
        enable_llm_verify = getattr(cfg, "enable_llm_verify", False)
        screenshot_delay_seconds = getattr(cfg, "screenshot_delay_seconds", 0.5)

        await asyncio.sleep(screenshot_delay_seconds)
        shot_idx += 1
        after_path = await self.driver._take_screenshot(shot_idx)
        if not after_path:
            logger.warning("  After-screenshot capture failed, skipping verification")
            return False, shot_idx, step_ctx.step_verify, last_verify_result

        exec_ctx.screenshots.append(after_path)
        with open(after_path, "rb") as f:
            after_b64 = base64.b64encode(f.read()).decode()

        step_verify = verifier.verify_pixel_diff(step_ctx.screenshot_b64, after_b64)
        last_verify_result = step_verify

        if step_verify.is_no_change:
            if enable_llm_verify:
                diag = await verifier.diagnose_no_change(
                    step_ctx.screenshot_b64, after_b64,
                    action_desc=plan_action or "",
                    expectation=(plan_output.get("Expectation") or ""),
                    model=actor_model,
                )
                step_verify = diag
                last_verify_result = diag
            failed_attempts.append(FailedAttempt(
                action=plan_action or "",
                position=step_ctx.model_position,
                category=FailureCategory.from_string(step_verify.category or "UNKNOWN"),
                reason=step_verify.diagnosis or "",
            ))
            logger.warning(
                f"  Verify NO_CHANGE (attempt {step_ctx.attempt + 1}): "
                f"{step_verify.diagnosis}"
            )
            return True, shot_idx, step_verify, last_verify_result

        if enable_llm_verify:
            expectation = (plan_output.get("Expectation") or "").strip()
            if expectation:
                llm_result, retry = await self.run_llm_verify(
                    after_b64=after_b64, expectation=expectation,
                    verifier=verifier, actor_model=actor_model,
                    attempt=step_ctx.attempt,
                )
                if llm_result is not None:
                    step_verify = llm_result
                    last_verify_result = llm_result
                    if retry:
                        wrong_diag = await verifier.diagnose_no_change(
                            step_ctx.screenshot_b64, after_b64,
                            action_desc=plan_action or "",
                            expectation=expectation,
                            model=actor_model,
                            wrong_change=True,
                        )
                        step_verify = wrong_diag
                        last_verify_result = wrong_diag
                        failed_attempts.append(FailedAttempt(
                            action=plan_action or "",
                            position=step_ctx.model_position,
                            category=FailureCategory.from_string(wrong_diag.category or "UNKNOWN"),
                            reason=wrong_diag.diagnosis or "",
                        ))
                        logger.warning(
                            f"  LLM verify NO_CHANGE (attempt {step_ctx.attempt + 1}): "
                            f"{wrong_diag.diagnosis}"
                        )
                        return True, shot_idx, step_verify, last_verify_result

        logger.info(
            f"  Verify: {step_verify.status.value} "
            f"(diff={step_verify.pixel_diff_score:.4f})"
        )
        return False, shot_idx, step_verify, last_verify_result

    async def plan_one_step(
        self,
        exec_ctx: ExecuteContext,
        step_ctx: StepContext,
        *,
        guidance: str,
        skill: Any,
        failed_attempts: list[FailedAttempt],
        last_verify_result: VerifyResult | None,
        last_action_type: str,
    ) -> tuple[str, ToolResult | None]:
        """Plan one step + completion check.

        Builds planner input, calls VerifiedPlanner.plan, rewrites forbidden
        first-step actions, and checks completion.

        Returns ``(status, completion_result)``:
        - ``"proceed"`` — step_ctx.plan_output/plan_action set; continue to actor.
        - ``"done"`` / ``"error"`` — completion_result holds the ToolResult to return.
        """
        cfg = exec_ctx.cfg
        enable_prompt_delta = getattr(cfg, "enable_prompt_delta", False)

        # ---- Build planner history -------------------------------
        if exec_ctx.structured_memory:
            lookback = 5 if step_ctx.attempt > 0 else 3
            planner_history = [
                exec_ctx.structured_memory.get_execution_context(
                    max_recent_steps=lookback
                )
            ]
        else:
            planner_history = list(exec_ctx.action_history)

        # ---- Build plan kwargs -----------------------------------
        plan_kwargs: dict[str, Any] = dict(
            task=exec_ctx.instruction,
            guidance_trajectory=guidance,
            screenshot_b64=step_ctx.screenshot_b64,
            action_history=planner_history,
            failed_attempts=failed_attempts,
        )
        if last_verify_result is not None:
            plan_kwargs["previous_verify_result"] = last_verify_result
        if step_ctx.spatial_context:
            plan_kwargs["spatial_context"] = step_ctx.spatial_context
        if enable_prompt_delta and last_action_type:
            plan_kwargs["action_type_hint"] = last_action_type

        # ---- Plan ------------------------------------------------
        try:
            plan_output = await exec_ctx.planner.plan(**plan_kwargs)
        except Exception as exc:
            logger.error(f"VerifiedPlanner failed: {exc}")
            return "error", ToolResult(
                text=f"Planner error at step {step_ctx.step}: {exc}",
                media=self.driver._key_screenshots(exec_ctx.screenshots),
            )

        plan_action = plan_output.get("Action")
        plan_observation = plan_output.get("Observation", "")
        plan_reasoning = plan_output.get("Reasoning", "")
        current_step_num = plan_output.get("Current Step", step_ctx.step)

        plan_action = self.driver._rewrite_forbidden_first_step_action(
            plan_action, step_ctx.step, skill
        )

        logger.info(f"  Plan: step={current_step_num}, action={plan_action}")
        logger.debug(f"  Reasoning: {plan_reasoning}")

        # ---- Completion check ------------------------------------
        if plan_action is None or plan_action == "null" or plan_action == "":
            await self.driver._finalize_plan(
                exec_ctx.plan_manager, exec_ctx.plan,
                exec_ctx.structured_memory,
                current_step_num, exec_ctx.planner_model,
            )
            key_shots = self.driver._key_screenshots(exec_ctx.screenshots)
            summary = (
                exec_ctx.plan_manager.get_plan_summary(exec_ctx.plan)
                if exec_ctx.plan_manager and exec_ctx.plan
                else ""
            )
            self.driver._flush_skill_lessons(exec_ctx.structured_memory, exec_ctx.skill_name)
            return "done", ToolResult(
                text=(
                    f"GUI task completed via {self.driver._planner_label()}.\n\n"
                    f"Steps taken: {step_ctx.step}\n"
                    f"Final observation: {plan_observation}\n"
                    f"{f'Plan: {summary}' if summary else ''}\n\n"
                    f"Steps log:\n{json.dumps(exec_ctx.steps_log, indent=2)}"
                ),
                media=key_shots,
            )

        # ---- Proceed — store in step_ctx -------------------------
        step_ctx.plan_output = plan_output
        step_ctx.plan_action = plan_action
        return "proceed", None

    async def ground_action(
        self,
        exec_ctx: ExecuteContext,
        step_ctx: StepContext,
        *,
        os_name: str,
    ) -> tuple[str, ToolResult | None]:
        """Ground + execute one action.

        Calls actor, transforms coordinates, logs, applies intent/double-click
        upgrades, and executes via AlohaExecutor. Stores results in step_ctx.

        Returns ``(status, result)``:
        - ``"proceed"`` — executor succeeded; step_ctx.action_dict /
          model_position / executor_position / executor_result set.
        - ``"done"`` — actor reported completion; result holds the ToolResult.
        - ``"error"`` — actor raised; result holds the error ToolResult.
        - ``"retry"`` — executor failed; step_ctx.executor_result has the
          error message (caller sets last_verify_result and continues).
        """
        plan_action = step_ctx.plan_action

        # ---- Call actor ----
        try:
            action_dict, is_complete = await self.driver._call_actor(
                exec_ctx.mode, plan_action, step_ctx.screenshot_b64, os_name
            )
        except Exception as exc:
            logger.error(f"Actor failed: {exc}")
            return "error", ToolResult(
                text=f"Actor error at step {step_ctx.step}: {exc}",
                media=[step_ctx.screenshot_path],
            )

        if is_complete:
            key_shots = self.driver._key_screenshots(exec_ctx.screenshots)
            self.driver._flush_skill_lessons(exec_ctx.structured_memory, exec_ctx.skill_name)
            return "done", ToolResult(
                text=(
                    f"GUI task completed by actor.\n"
                    f"Steps taken: {step_ctx.step}\n\n"
                    f"Steps log:\n{json.dumps(exec_ctx.steps_log, indent=2)}"
                ),
                media=key_shots,
            )

        # ---- Coordinate transform ----
        model_position = None
        executor_position = None
        if "position" in action_dict:
            pos = action_dict["position"]
            if isinstance(pos, list) and len(pos) == 2:
                model_position = [int(pos[0]), int(pos[1])]
                x, y = self.driver._transform_coords(pos[0], pos[1], mode=exec_ctx.mode)
                executor_position = [x, y]
                action_dict["position"] = executor_position

        # ---- Log grounded action ----
        self.driver._log_action({
            "ts": time.time(),
            "step": step_ctx.step,
            "attempt": step_ctx.attempt,
            "instruction": exec_ctx.instruction,
            "plan_action": plan_action,
            "action": action_dict.get("action"),
            "model_position": model_position,
            "executor_position": executor_position,
            "mode": exec_ctx.mode,
            "coord_space": str(getattr(self.driver._config, "coord_space", "pixel")),
            "model_img_size": list(self.driver._model_img_size),
            "screenshot": step_ctx.screenshot_path,
            "click_count": action_dict.get("click_count"),
        })

        # ---- Intent + double-click upgrade ----
        if plan_action:
            action_dict["intent"] = plan_action
            action_dict["plan"] = plan_action
        action_dict["instruction"] = exec_ctx.instruction

        if action_dict.get("action") == "CLICK" and plan_action:
            normalized = plan_action.lower().replace("-", " ").replace("_", " ")
            if "double click" in normalized or "双击" in plan_action:
                action_dict["click_count"] = 2

        # ---- Store in step_ctx BEFORE executor (so retry has values) ----
        step_ctx.action_dict = action_dict
        step_ctx.model_position = model_position
        step_ctx.executor_position = executor_position

        # ---- Execute ----
        success, executor_result = await exec_ctx.executor.execute(action_dict)
        step_ctx.executor_result = executor_result

        if not success:
            return "retry", None

        return "proceed", None

    async def execute_single_step(
        self,
        exec_ctx: ExecuteContext,
        step_ctx: StepContext,
        *,
        guidance: str,
        skill: Any,
        failed_attempts: list[FailedAttempt],
        last_verify_result: VerifyResult | None,
        last_action_type: str,
        actor_model: str,
        os_name: str,
        shot_idx: int,
        instruction: str,
        max_steps: int,
        timings: dict[str, int] | None = None,
    ) -> tuple[str, ToolResult | None, VerifyResult | None, VerifyResult | None, int]:
        """Execute one attempt of a single GUI step (L1 primitive).

        This is the smallest unit of the enhanced pipeline: capture → plan →
        ground → verify. It is intentionally stateless with respect to cross-step
        history; ``exec_ctx``/``step_ctx`` carry all inputs/outputs.

        Returns:
            ``(status, result, step_verify, updated_last_verify_result, new_shot_idx)``
            where ``status`` is one of:
            - ``"proceed"``: step succeeded, caller should record and continue.
            - ``"done"``: planner/actor signaled task completion.
            - ``"error"``: unrecoverable error (screenshot/planner/actor failed).
            - ``"retry"``: this attempt failed but may be retried.
        """
        attempt = step_ctx.attempt
        if attempt > 0:
            logger.info(f"  Retry {attempt}")

        # ---- Capture observation (screenshot + spatial) --------
        _t = time.monotonic()
        new_shot_idx, cap_err = await self.capture_observation(exec_ctx, step_ctx, shot_idx)
        if timings is not None:
            timings["📸"] = int((time.monotonic() - _t) * 1000)
        if cap_err:
            self.driver._monitor_write(
                status="error", instruction=instruction,
                step=step_ctx.step, max_steps=max_steps, error=cap_err,
                timing=timings,
            )
            return "error", ToolResult(text=cap_err), None, last_verify_result, new_shot_idx

        # ---- Plan one step ------
        _t = time.monotonic()
        plan_status, completion_result = await self.plan_one_step(
            exec_ctx, step_ctx,
            guidance=guidance, skill=skill,
            failed_attempts=failed_attempts,
            last_verify_result=last_verify_result,
            last_action_type=last_action_type,
        )
        if timings is not None:
            timings["🧠"] = int((time.monotonic() - _t) * 1000)
        if plan_status != "proceed":
            self.driver._monitor_write(
                status="finished" if plan_status == "done" else "error",
                instruction=instruction, step=step_ctx.step, max_steps=max_steps,
                timing=timings,
            )
            return (
                ("done" if plan_status == "done" else "error"),
                completion_result,
                None,
                last_verify_result,
                new_shot_idx,
            )
        plan_action = step_ctx.plan_action

        self.driver._monitor_write(
            status="running", instruction=instruction,
            step=step_ctx.step, max_steps=max_steps,
            action=plan_action,
            thought=(step_ctx.plan_output or {}).get("Reasoning", ""),
            timing=timings,
        )

        # ---- Ground + execute action ----
        _t = time.monotonic()
        ground_status, ground_result = await self.ground_action(
            exec_ctx, step_ctx, os_name=os_name,
        )
        if timings is not None:
            timings["🎯"] = int((time.monotonic() - _t) * 1000)
        if ground_status in ("done", "error"):
            self.driver._monitor_write(
                status="finished" if ground_status == "done" else "error",
                instruction=instruction, step=step_ctx.step, max_steps=max_steps,
                timing=timings,
            )
            return (
                ("done" if ground_status == "done" else "error"),
                ground_result,
                None,
                last_verify_result,
                new_shot_idx,
            )
        if ground_status == "retry":
            updated_last_verify_result = VerifyResult(
                status=VerifyStatus.NO_CHANGE,
                confidence=1.0,
                pixel_diff_score=0.0,
                diagnosis=f"Executor error: {step_ctx.executor_result}",
            )
            logger.warning(
                f"  Executor failed (attempt {attempt + 1}): "
                f"{step_ctx.executor_result}"
            )
            return "retry", None, None, updated_last_verify_result, new_shot_idx

        # ---- TVAE verification ----
        step_verify: VerifyResult | None = None
        verifier = exec_ctx.verifier
        if verifier:
            _t = time.monotonic()
            should_retry, new_shot_idx, step_verify, updated_last_verify_result = (
                await self.verify_step(
                    exec_ctx, step_ctx,
                    shot_idx=new_shot_idx, plan_action=plan_action,
                    plan_output=step_ctx.plan_output, failed_attempts=failed_attempts,
                    last_verify_result=last_verify_result, actor_model=actor_model,
                )
            )
            if timings is not None:
                timings["✅"] = int((time.monotonic() - _t) * 1000)
            if should_retry:
                self.driver._monitor_write(
                    status="running", instruction=instruction,
                    step=step_ctx.step, max_steps=max_steps,
                    error=f"retry after verify",
                    timing=timings,
                )
                return "retry", None, step_verify, updated_last_verify_result, new_shot_idx

        return "proceed", None, step_verify, last_verify_result, new_shot_idx


@dataclass
class UITarsStepResult:
    """Result of one UI-TARS primitive step."""

    status: str  # "proceed" | "done" | "call_user" | "stuck" | "exec_fail" | "error"
    thought: str = ""
    action: str = ""
    summary: str = ""
    message: str = ""
    exec_success: bool = False
    exec_message: str = ""
    screenshot_path: str = ""


class UITarsPrimitive:
    """L1 single-step primitive for the UI-TARS single-model family.

    Maintains the multi-turn ``Conversation`` state across steps, but exposes
    exactly one screenshot → model → parse → execute attempt per ``step()``
    call.  Loop-level policy (ICL injection, ledger logging, event logging,
    max-steps termination) stays in the caller.
    """

    def __init__(
        self,
        driver: Any,
        conversations: list[Any] | None = None,
        max_repeat_actions: int = 3,
    ) -> None:
        self.driver = driver
        self.conversations = list(conversations or [])
        self.max_repeat_actions = max_repeat_actions
        self.recent_actions: list[str] = []

    def add_icl_context(self, turns: list[Any]) -> None:
        """Prepend in-context-learning turns to the conversation."""
        if turns:
            self.conversations.extend(turns)

    async def step(
        self,
        *,
        instruction: str,
        screenshot_b64: str,
        screenshot_path: str,
        img_size: tuple[int, int],
        step: int,
        max_steps: int,
    ) -> UITarsStepResult:
        """Run one UI-TARS step.

        Returns a :class:`UITarsStepResult` whose ``status`` tells the caller
        whether to continue, finish, or fail the overall task.
        """
        # ---- Append screenshot as a user turn ----
        # Import Conversation lazily to avoid a load-time circular dependency
        # between this module and syll.agent.tools.ui_tars.
        from syll.agent.tools.ui_tars import Conversation

        mime = self.driver._guess_image_mime(Path(screenshot_path))
        self.conversations.append(Conversation(
            role="user",
            screenshot_b64=screenshot_b64,
            screenshot_mime=mime,
            img_size=img_size,
        ))

        # ---- Call UI-TARS with retry ----
        response_text = await self.driver._call_uitars_with_retry(
            instruction, self.conversations
        )
        if not response_text:
            return UITarsStepResult(
                status="error",
                message="Error: UI-TARS API call failed after retries",
                screenshot_path=screenshot_path,
            )

        # ---- Parse response and append assistant turn ----
        thought, action_str = self.driver._parse_response(response_text)
        self.conversations.append(Conversation(
            role="assistant",
            text=response_text,
        ))

        # ---- Stuck detection ----
        self.recent_actions.append(action_str)
        if len(self.recent_actions) >= self.max_repeat_actions:
            last_n = self.recent_actions[-self.max_repeat_actions:]
            if all(a == last_n[0] for a in last_n):
                return UITarsStepResult(
                    status="stuck",
                    thought=thought,
                    action=action_str,
                    message=(
                        f"repeated {action_str} {self.max_repeat_actions} times"
                    ),
                    screenshot_path=screenshot_path,
                )

        # ---- Terminal actions ----
        finished_match = re.match(
            r"finished\((?:content=)?['\"]?(.+?)['\"]?\)", action_str
        )
        if finished_match:
            return UITarsStepResult(
                status="done",
                thought=thought,
                action=action_str,
                summary=finished_match.group(1),
                screenshot_path=screenshot_path,
            )

        call_user_match = re.match(
            r"call_user\((?:content=)?['\"]?(.+?)['\"]?\)", action_str
        )
        if call_user_match:
            return UITarsStepResult(
                status="call_user",
                thought=thought,
                action=action_str,
                message=call_user_match.group(1),
                screenshot_path=screenshot_path,
            )

        # ---- Execute action ----
        intent_text = "\n".join(part for part in (instruction, thought) if part)
        success, msg = await self.driver._execute_action_with_retry(
            action_str, intent_text=intent_text
        )
        if not success:
            return UITarsStepResult(
                status="exec_fail",
                thought=thought,
                action=action_str,
                message=msg,
                screenshot_path=screenshot_path,
            )

        return UITarsStepResult(
            status="proceed",
            thought=thought,
            action=action_str,
            exec_success=True,
            exec_message=msg,
            screenshot_path=screenshot_path,
        )
