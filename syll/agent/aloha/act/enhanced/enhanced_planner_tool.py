"""Enhanced Aloha Planner Tool: integrates Phase 1-4 enhanced modules.

Inherits from :class:`AlohaPlannerTool` and overrides ``execute()`` to
run the full enhanced pipeline when ``EnhancedConfig.ALL_ENABLED`` is
True.  Falls back to the parent's simple planner→actor→executor loop
otherwise.

Phases integrated:
- Phase 1: TVAE verification (VerifiedPlanner + ActionVerifier)
- Phase 2: Plan persistence (PlanManager) + Structured Memory
- Phase 4: Spatial analysis (SpatialAnalyzer) + Semantic trace generation

Does **not** modify the original ``AlohaPlannerTool``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import platform
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from syll.agent.aloha.act.enhanced.action_verifier import (
    ActionVerifier,
    VerifyResult,
    VerifyStatus,
)
from syll.agent.aloha.act.enhanced.config import EnhancedConfig
from syll.agent.aloha.act.enhanced.plan_manager import PlanManager
from syll.agent.aloha.act.enhanced.spatial_analyzer import SpatialAnalyzer
from syll.agent.aloha.act.enhanced.structured_memory import StructuredMemory
from syll.agent.aloha.act.enhanced.verified_planner import VerifiedPlanner
from syll.agent.tools.aloha_planner_tool import AlohaPlannerTool
from syll.agent.tools.base import ToolResult


class EnhancedAlohaPlannerTool(AlohaPlannerTool):
    """Enhanced GUI automation with TVAE verification, structured memory,
    and spatial analysis.  Drop-in replacement for ``AlohaPlannerTool``.

    When ``EnhancedConfig.ALL_ENABLED`` is False the tool behaves
    identically to its parent.
    """

    def __init__(
        self,
        gui_config: Any,
        aloha_skill_store: Any,
        syll_config: Any = None,
        enhanced_config: EnhancedConfig | None = None,
    ):
        super().__init__(gui_config, aloha_skill_store, syll_config)
        self._enhanced_config = enhanced_config or EnhancedConfig()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def execute(
        self,
        instruction: str,
        skill_name: str,
        max_steps: int | None = None,
        actor_mode: str | None = None,
        **kwargs: Any,
    ) -> str | ToolResult:
        """Execute a GUI task with the full enhanced pipeline."""

        # Fall back to original pipeline when disabled
        if not self._enhanced_config.ALL_ENABLED:
            return await super().execute(
                instruction, skill_name, max_steps, actor_mode, **kwargs
            )

        cfg = self._enhanced_config
        steps_limit = max_steps or self._config.max_steps
        os_name = platform.system()

        # ── Load skill & build guidance (same as parent) ───────────────
        skill = self._aloha_skill_store.load_skill(skill_name)
        if not skill:
            return ToolResult(text=f"Aloha skill '{skill_name}' not found")
        if not skill.trajectory and not skill.steps:
            return ToolResult(
                text=f"Skill '{skill_name}' has no trajectory or steps"
            )

        mode = self._resolve_actor_mode(skill, actor_mode)
        guidance = self._build_guidance(skill)

        # ── Resolve planner endpoint ───────────────────────────────────
        planner_model, planner_api_key, planner_api_base = (
            self._resolve_planner_endpoint()
        )

        # ── Phase 1: VerifiedPlanner + ActionVerifier ──────────────────
        planner = VerifiedPlanner(
            model=planner_model,
            os_name=os_name,
            api_key=planner_api_key,
            api_base=planner_api_base,
        )
        verifier = (
            ActionVerifier(pixel_diff_threshold=cfg.pixel_diff_threshold)
            if cfg.enable_tvae_verification
            else None
        )

        # ── Phase 2: Plan persistence + structured memory ─────────────
        workspace = self._resolve_workspace()

        plan_manager = (
            PlanManager(workspace=workspace)
            if cfg.enable_plan_persistence
            else None
        )
        structured_memory = (
            StructuredMemory(workspace=workspace)
            if cfg.enable_structured_memory
            else None
        )

        plan = None
        if plan_manager:
            step_descs = self._extract_step_descriptions(skill, guidance)
            plan = plan_manager.create_plan(skill_name, instruction, step_descs)
            plan_manager.save_plan(plan)

        # ── Phase 4: Spatial analyzer ─────────────────────────────────
        spatial_analyzer = (
            SpatialAnalyzer(
                model=planner_model,
                api_key=planner_api_key,
                api_base=planner_api_base,
            )
            if cfg.enable_spatial_context
            else None
        )

        # ── Executor ──────────────────────────────────────────────────
        from syll.agent.aloha.act.executor import AlohaExecutor

        executor = AlohaExecutor(self._config)

        # ── Main enhanced loop ────────────────────────────────────────
        screenshots: list[str] = []
        steps_log: list[dict] = []
        action_history: list[str] = []
        last_verify_result: VerifyResult | None = None
        last_action_type: str = ""
        _shot_idx = 0

        for step in range(1, steps_limit + 1):
            logger.info(f"[Enhanced] Step {step}/{steps_limit}: {instruction}")

            max_retries = (
                cfg.max_consecutive_failures if cfg.enable_tvae_verification else 0
            )
            step_succeeded = False
            step_verify: VerifyResult | None = None
            plan_output: dict = {}
            action_dict: dict = {}
            executor_result = ""
            model_position: list[int] | None = None
            executor_position: list[int] | None = None
            screenshot_path = ""
            attempt = 0

            # ── Inner TVAE retry loop ─────────────────────────────────
            for attempt in range(max_retries + 1):
                if attempt > 0:
                    logger.info(f"  Retry {attempt}/{max_retries}")

                # ---- Before screenshot --------------------------------
                _shot_idx += 1
                screenshot_path = await self._take_screenshot(_shot_idx)
                if not screenshot_path:
                    return ToolResult(
                        text="Error: Failed to capture screenshot"
                    )
                screenshots.append(screenshot_path)

                with open(screenshot_path, "rb") as f:
                    screenshot_b64 = base64.b64encode(f.read()).decode()

                # ---- Phase 4: Spatial analysis -------------------------
                spatial_context = ""
                if spatial_analyzer:
                    try:
                        spatial_context = await spatial_analyzer.analyze(
                            screenshot_b64
                        )
                    except Exception as exc:
                        logger.debug(f"Spatial analysis skipped: {exc}")

                # ---- Build planner input -------------------------------
                if structured_memory:
                    planner_history = [
                        structured_memory.get_execution_context(
                            max_recent_steps=3
                        )
                    ]
                else:
                    planner_history = list(action_history)

                # ---- Phase 1: Plan with VerifiedPlanner ----------------
                plan_kwargs: dict[str, Any] = dict(
                    task=instruction,
                    guidance_trajectory=guidance,
                    screenshot_b64=screenshot_b64,
                    action_history=planner_history,
                )
                if last_verify_result is not None:
                    plan_kwargs["previous_verify_result"] = last_verify_result
                if spatial_context:
                    plan_kwargs["spatial_context"] = spatial_context
                if cfg.enable_prompt_delta and last_action_type:
                    plan_kwargs["action_type_hint"] = last_action_type

                try:
                    plan_output = await planner.plan(**plan_kwargs)
                except Exception as exc:
                    logger.error(f"VerifiedPlanner failed: {exc}")
                    return ToolResult(
                        text=f"Planner error at step {step}: {exc}",
                        media=self._key_screenshots(screenshots),
                    )

                plan_action = plan_output.get("Action")
                plan_observation = plan_output.get("Observation", "")
                plan_reasoning = plan_output.get("Reasoning", "")
                current_step_num = plan_output.get("Current Step", step)

                plan_action = self._rewrite_forbidden_first_step_action(
                    plan_action, step, skill
                )

                logger.info(
                    f"  Plan: step={current_step_num}, action={plan_action}"
                )
                logger.debug(f"  Reasoning: {plan_reasoning}")

                # ---- Check completion ----------------------------------
                if (
                    plan_action is None
                    or plan_action == "null"
                    or plan_action == ""
                ):
                    await self._finalize_plan(
                        plan_manager, plan, structured_memory,
                        current_step_num, planner_model,
                    )
                    key_shots = self._key_screenshots(screenshots)
                    summary = (
                        plan_manager.get_plan_summary(plan)
                        if plan_manager and plan
                        else ""
                    )
                    return ToolResult(
                        text=(
                            f"GUI task completed via enhanced planner.\n\n"
                            f"Steps taken: {step}\n"
                            f"Final observation: {plan_observation}\n"
                            f"{f'Plan: {summary}' if summary else ''}\n\n"
                            f"Steps log:\n{json.dumps(steps_log, indent=2)}"
                        ),
                        media=key_shots,
                    )

                # ---- Call actor (reuse parent) -------------------------
                try:
                    action_dict, is_complete = await self._call_actor(
                        mode, plan_action, screenshot_b64, os_name
                    )
                except Exception as exc:
                    logger.error(f"Actor failed: {exc}")
                    return ToolResult(
                        text=f"Actor error at step {step}: {exc}",
                        media=[screenshot_path],
                    )

                if is_complete:
                    key_shots = self._key_screenshots(screenshots)
                    return ToolResult(
                        text=(
                            f"GUI task completed by actor.\n"
                            f"Steps taken: {step}\n\n"
                            f"Steps log:\n{json.dumps(steps_log, indent=2)}"
                        ),
                        media=key_shots,
                    )

                # ---- Coordinate transform (reuse parent) ----------------
                if "position" in action_dict:
                    pos = action_dict["position"]
                    if isinstance(pos, list) and len(pos) == 2:
                        model_position = [int(pos[0]), int(pos[1])]
                        x, y = self._transform_coords(
                            pos[0], pos[1], mode=mode
                        )
                        executor_position = [x, y]
                        action_dict["position"] = executor_position

                if plan_action:
                    action_dict["intent"] = plan_action
                    action_dict["plan"] = plan_action
                action_dict["instruction"] = instruction

                # Double-click upgrade
                if action_dict.get("action") == "CLICK" and plan_action:
                    normalized = (
                        plan_action.lower().replace("-", " ").replace("_", " ")
                    )
                    if "double click" in normalized or "双击" in plan_action:
                        action_dict["click_count"] = 2

                # ---- Execute action ------------------------------------
                success, executor_result = await executor.execute(action_dict)

                if not success:
                    last_verify_result = VerifyResult(
                        status=VerifyStatus.NO_CHANGE,
                        confidence=1.0,
                        pixel_diff_score=0.0,
                        diagnosis=f"Executor error: {executor_result}",
                    )
                    logger.warning(
                        f"  Executor failed (attempt {attempt + 1}): "
                        f"{executor_result}"
                    )
                    continue  # retry

                # ---- Phase 1: TVAE verification ------------------------
                if verifier:
                    await asyncio.sleep(cfg.screenshot_delay_seconds)
                    _shot_idx += 1
                    after_path = await self._take_screenshot(_shot_idx)
                    if after_path:
                        screenshots.append(after_path)
                        with open(after_path, "rb") as f:
                            after_b64 = base64.b64encode(f.read()).decode()

                        step_verify = verifier.verify_pixel_diff(
                            screenshot_b64, after_b64
                        )
                        last_verify_result = step_verify

                        if step_verify.is_no_change:
                            logger.warning(
                                f"  Verify NO_CHANGE (attempt {attempt + 1}): "
                                f"{step_verify.diagnosis}"
                            )
                            continue  # retry with recovery feedback

                        logger.info(
                            f"  Verify: {step_verify.status.value} "
                            f"(diff={step_verify.pixel_diff_score:.4f})"
                        )
                    else:
                        logger.warning(
                            "  After-screenshot capture failed, "
                            "skipping verification"
                        )

                # Step succeeded — exit retry loop
                step_succeeded = True
                break

            # ── Post-step recording ────────────────────────────────────
            verify_status = (
                step_verify.status.value
                if step_verify
                else ("SUCCESS" if step_succeeded else "NO_CHANGE")
            )
            diagnosis = (
                step_verify.diagnosis
                if step_verify
                else ("" if step_succeeded else executor_result)
            )

            # Append to raw history (always, for logging compatibility)
            action_history.append(
                self._format_action_history(
                    plan_output.get("Action"),
                    executor_result,
                    model_position,
                    executor_position,
                    action_dict.get("click_backend"),
                    action_dict.get("mac_accessibility"),
                    action_dict.get("event_style"),
                    action_dict.get("frontmost_app"),
                )
            )

            # Phase 2: Record in structured memory
            if structured_memory:
                structured_memory.record_step(
                    index=step,
                    action=plan_output.get("Action", ""),
                    expectation=plan_output.get("Expectation", ""),
                    verify_status=verify_status,
                    diagnosis=diagnosis,
                )

            # Phase 2: Update plan manager
            if plan_manager and plan:
                current_step_idx = plan_output.get("Current Step", step)
                plan_status = "DONE" if step_succeeded else "FAILED"
                plan_manager.update_step(
                    plan, current_step_idx, plan_status, result=diagnosis
                )
                plan_manager.save_plan(plan)

            # Phase 2: Periodic compression
            if structured_memory and step % 10 == 0:
                try:
                    await structured_memory.compress_history(
                        model=planner_model
                    )
                except Exception as exc:
                    logger.debug(f"Memory compression skipped: {exc}")

            # Classify action type for next prompt delta
            if cfg.enable_prompt_delta:
                last_action_type = _classify_action_type(action_dict)

            # Step log
            steps_log.append({
                "step": step,
                "plan": plan_output.get("Action"),
                "action": action_dict,
                "reasoning": plan_output.get("Reasoning", ""),
                "observation": plan_output.get("Observation", ""),
                "executor_result": executor_result,
                "verify_status": verify_status,
                "verify_diagnosis": diagnosis,
                "retries_used": attempt,
                "model_position": model_position,
                "executor_position": executor_position,
            })

            # Event log
            self._log_event(
                instruction, plan_output, executor_result,
                verify_status, step, screenshot_path, skill_name, mode,
            )

            if not step_succeeded:
                logger.warning(
                    f"Step {step} failed after all retries, continuing..."
                )

        # ── Phase 4: Generate semantic traces ─────────────────────────
        if cfg.enable_semantic_trace:
            await self._generate_semantic_traces(
                steps_log, planner_model, planner_api_key,
                planner_api_base, spatial_analyzer,
            )

        # ── Final result ──────────────────────────────────────────────
        key_shots = self._key_screenshots(screenshots)
        plan_summary = (
            plan_manager.get_plan_summary(plan)
            if plan_manager and plan
            else ""
        )
        return ToolResult(
            text=(
                f"Reached max steps ({steps_limit}). "
                f"Task may not be complete.\n\n"
                f"{f'Plan: {plan_summary}' if plan_summary else ''}\n\n"
                f"Steps log:\n{json.dumps(steps_log, indent=2)}"
            ),
            media=key_shots,
        )

    # ------------------------------------------------------------------
    # Helpers — skill loading & endpoint resolution
    # ------------------------------------------------------------------

    def _resolve_actor_mode(self, skill: Any, actor_mode: str | None) -> str:
        """Determine actor backend mode."""
        mode = actor_mode or getattr(skill.meta, "actor_mode", "")
        if not mode and self._syll_config:
            actor_model = self._syll_config.resolve_endpoint("actor").model.lower()
            if "claude" in actor_model or "anthropic" in actor_model:
                mode = "claude-cua"
        return (
            mode
            or getattr(self._config, "execution_mode", "ui-tars")
            or "ui-tars"
        )

    def _build_guidance(self, skill: Any) -> str:
        """Build guidance trajectory string from skill."""
        from syll.agent.aloha.act.trajectory_manager import TrajectoryManager

        traj_manager = TrajectoryManager()
        if skill.trajectory:
            guidance = traj_manager.get_trajectory_in_context(skill.trajectory)
        else:
            guidance_steps = []
            for s in skill.steps:
                desc = ""
                if s.trace:
                    desc = s.trace.action
                elif s.action.description:
                    desc = s.action.description
                else:
                    desc = s.action.type
                guidance_steps.append(f"Step [{s.index}]: {desc}")
            guidance = "\n".join(guidance_steps)
        return guidance or "(No guidance trajectory available)"

    def _resolve_planner_endpoint(
        self,
    ) -> tuple[str, str | None, str | None]:
        """Return (model, api_key, api_base) for the planner."""
        if self._syll_config:
            ep = self._syll_config.resolve_endpoint("planner")
            return ep.litellm_model, ep.api_key or None, ep.api_base
        return "gpt-4o", None, None

    def _resolve_workspace(self) -> Path:
        """Return the workspace path for persistent files."""
        workspace = Path(tempfile.gettempdir())
        if self._syll_config:
            workspace = (
                getattr(self._syll_config, "workspace_path", workspace)
                or workspace
            )
        return workspace

    # ------------------------------------------------------------------
    # Helpers — plan finalisation & event logging
    # ------------------------------------------------------------------

    @staticmethod
    async def _finalize_plan(
        plan_manager: PlanManager | None,
        plan: Any | None,
        structured_memory: StructuredMemory | None,
        current_step_num: int,
        model: str,
    ) -> None:
        """Mark plan complete and compress memory."""
        if plan_manager and plan:
            plan_manager.update_step(
                plan, current_step_num, "DONE", result="Task completed"
            )
            plan_manager.save_plan(plan)
        if structured_memory:
            try:
                await structured_memory.compress_history(model=model)
            except Exception as exc:
                logger.debug(f"Memory compression skipped: {exc}")

    def _log_event(
        self,
        instruction: str,
        plan_output: dict,
        executor_result: str,
        verify_status: str,
        step: int,
        screenshot_path: str,
        skill_name: str,
        mode: str,
    ) -> None:
        """Log a GUI action event to the event store."""
        if not self._event_store:
            return
        from syll.agent.events import Event, EventContent, EventSource

        self._event_store.log_event(
            Event(
                agent_type="gui_agent",
                event_type="action",
                source=EventSource(
                    platform="desktop", chat_id="gui", user_id="system"
                ),
                content=EventContent(
                    text=(
                        f"Instruction: {instruction}\n"
                        f"Plan: {plan_output.get('Action')}\n"
                        f"Reasoning: {plan_output.get('Reasoning', '')}\n"
                        f"Result: {executor_result}\n"
                        f"Verify: {verify_status}"
                    ),
                    media=[screenshot_path] if screenshot_path else [],
                    metadata={
                        "step": step,
                        "verify_status": verify_status,
                        "enhanced": True,
                        "skill_name": skill_name,
                        "actor_mode": mode,
                    },
                ),
            )
        )

    async def _generate_semantic_traces(
        self,
        steps_log: list[dict],
        planner_model: str,
        planner_api_key: str | None,
        planner_api_base: str | None,
        spatial_analyzer: SpatialAnalyzer | None,
    ) -> None:
        """Phase 4: generate enriched semantic traces for learning."""
        try:
            from syll.agent.aloha.act.enhanced.enhanced_trace_generator import (
                EnhancedTraceGenerator,
            )
            from syll.agent.aloha.learn.trace_generator import TraceGenerator

            base_gen = TraceGenerator(
                model=planner_model,
                api_key=planner_api_key,
                api_base=planner_api_base,
            )
            enhanced_gen = EnhancedTraceGenerator(
                base_generator=base_gen,
                spatial_analyzer=spatial_analyzer,
            )
            actions_data = [
                log_entry.get("action", {}) for log_entry in steps_log
            ]
            traces = await enhanced_gen.generate_trace(
                actions_with_screenshots=actions_data,
                screenshots_dir=str(self._screenshot_dir),
                overall_task="",
            )
            logger.info(f"Generated {len(traces)} enhanced traces")
        except Exception as exc:
            logger.debug(f"Enhanced trace generation skipped: {exc}")

    # ------------------------------------------------------------------
    # Helpers — step description extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_step_descriptions(
        skill: Any, guidance: str
    ) -> list[str]:
        """Extract step descriptions from a skill for PlanManager."""
        import re

        steps = getattr(skill, "steps", None) or []
        if steps:
            result = []
            for s in steps:
                if s.trace and s.trace.action:
                    result.append(s.trace.action)
                elif s.action and s.action.description:
                    result.append(s.action.description)
                elif s.action and s.action.type:
                    result.append(s.action.type)
                else:
                    result.append(f"Step {s.index}")
            return result

        # Parse from guidance text
        matches = re.findall(r"Step\s*\[\d+\]:\s*(.+)", guidance)
        return [m.strip() for m in matches] if matches else ["Complete the task"]


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------

_ACTION_TYPE_MAP = {
    "drag": "drag",
    "dbl": "click",
    "double": "click",
    "right_click": "click",
    "click": "click",
    "type": "type",
    "input": "type",
    "scroll": "scroll",
    "wheel": "scroll",
    "hotkey": "hotkey",
    "key": "hotkey",
}


def _classify_action_type(action_dict: dict) -> str:
    """Classify action type for prompt delta selection."""
    action = (action_dict.get("action") or "").lower()
    for key, label in _ACTION_TYPE_MAP.items():
        if key in action:
            return label
    return "other"
