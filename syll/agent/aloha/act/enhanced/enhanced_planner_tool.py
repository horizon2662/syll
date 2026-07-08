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
import time
import tempfile
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any

from loguru import logger

from syll.agent.gui.primitive import GuiPrimitive
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
from syll.agent.aloha.act.enhanced.step_context import (
    ExecuteContext,
    FailedAttempt,
    FailureCategory,
    StepContext,
)
from syll.agent.tools.aloha_planner_tool import AlohaPlannerTool
from syll.agent.tools.base import ToolResult
from syll.sandbox.environment import Environment, LocalEnvironment


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
        environment: Environment | None = None,
    ):
        super().__init__(gui_config, aloha_skill_store, syll_config, environment=environment)
        if enhanced_config is not None:
            self._enhanced_config = enhanced_config
        else:
            # Prefer the canonical schema config under tools.gui.agent, falling
            # back to the legacy standalone config.json section.
            agent_cfg = None
            if syll_config is not None:
                try:
                    agent_cfg = getattr(syll_config.tools.gui, "agent", None)
                except Exception:
                    pass
            if agent_cfg is not None:
                self._enhanced_config = EnhancedConfig.from_gui_agent_config(agent_cfg)
            else:
                self._enhanced_config = EnhancedConfig.from_config_file()
        # 0b: per-purpose LLMProviders so GUI model calls are observable
        # (usage → ContextMeter) instead of bare litellm.acompletion.
        # _context_meter is attached by the subagent when one is threaded
        # through; until then usage recording is a no-op.
        self._context_meter = None
        self._skill_memory = None
        self._actor_provider = None
        self._on_actor_usage = None
        self._monitor_launched = False  # GUI monitor overlay (best-effort)
        self._primitive = GuiPrimitive(self)

    # ------------------------------------------------------------------
    # 0b: purpose providers + usage recording
    # ------------------------------------------------------------------

    @staticmethod
    def _make_purpose_provider(api_key, api_base):
        """Build a LiteLLMProvider bound to a purpose endpoint (planner/actor)."""
        try:
            from syll.providers.litellm_provider import LiteLLMProvider
            return LiteLLMProvider(api_key=api_key, api_base=api_base)
        except Exception as exc:
            logger.debug(f"purpose provider build skipped: {exc}")
            return None

    def _on_usage_cb(self, phase: str):
        """Return an on_usage callback that records a GUI call's tokens,
        or None when no ContextMeter is attached (recording is then a no-op)."""
        meter = getattr(self, "_context_meter", None)
        if meter is None:
            return None

        def _cb(resp):
            try:
                u = getattr(resp, "usage", None) or {}
                if not u:
                    return
                meter.record(
                    prompt_tokens=int(u.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(u.get("completion_tokens", 0) or 0),
                    phase=phase,
                    extra={"component": "gui"},
                )
            except Exception:
                pass

        return _cb

    def _flush_skill_lessons(self, structured_memory, skill_name: str) -> None:
        """L2: lift failed-step diagnoses into SkillMemory so a later run of
        the same skill can avoid the same pitfalls — closes the gap where
        StructuredMemory step diagnoses never reached SKILL.md. Best-effort:
        never raises into the GUI loop.
        """
        sm = getattr(self, "_skill_memory", None)
        if sm is None or structured_memory is None:
            return
        try:
            pitfalls = [
                f"[{skill_name}] step {s.index} '{s.action}'"
                f"{f' [{s.category}]' if getattr(s, 'category', '') else ''}: {s.diagnosis}"
                for s in getattr(structured_memory, "_steps", [])
                if s.verify_status != "SUCCESS" and (s.diagnosis or "").strip()
            ]
            if pitfalls:
                sm.ingest(pitfalls, status_ok=False)
        except Exception as exc:
            logger.debug(f"skill lesson flush skipped: {exc}")

    async def _run_llm_verify(self, *, after_b64, expectation, verifier,
                              actor_model, attempt):
        """Delegate to the L1 GUI primitive."""
        return await self._primitive.run_llm_verify(
            after_b64=after_b64, expectation=expectation, verifier=verifier,
            actor_model=actor_model, attempt=attempt,
        )

    async def _record_step(self, exec_ctx: ExecuteContext, step_ctx: StepContext) -> str:
        """Post-step recording (phase-2 refactor): structured memory + plan
        manager + steps log + event log + action history. Pure move out of
        execute()'s post-step block; returns ``last_action_type`` for the
        next iteration's prompt delta.
        """
        cfg = exec_ctx.cfg
        step_verify = step_ctx.step_verify
        verify_status = (
            step_verify.status.value if step_verify
            else ("SUCCESS" if step_ctx.succeeded else "NO_CHANGE")
        )
        diagnosis = (
            step_verify.diagnosis if step_verify
            else ("" if step_ctx.succeeded else step_ctx.executor_result)
        )
        exec_ctx.action_history.append(
            self._format_action_history(
                step_ctx.plan_output.get("Action"),
                step_ctx.executor_result,
                step_ctx.model_position, step_ctx.executor_position,
                step_ctx.action_dict.get("click_backend"),
                step_ctx.action_dict.get("mac_accessibility"),
                step_ctx.action_dict.get("event_style"),
                step_ctx.action_dict.get("frontmost_app"),
            )
        )
        if exec_ctx.structured_memory:
            exec_ctx.structured_memory.record_step(
                index=step_ctx.step,
                action=step_ctx.plan_output.get("Action", ""),
                expectation=step_ctx.plan_output.get("Expectation", ""),
                verify_status=verify_status, diagnosis=diagnosis,
                category=(step_verify.category if step_verify else ""),
            )
        if exec_ctx.plan_manager and exec_ctx.plan:
            current_step_idx = step_ctx.plan_output.get("Current Step", step_ctx.step)
            plan_status = "DONE" if step_ctx.succeeded else "FAILED"
            exec_ctx.plan_manager.update_step(
                exec_ctx.plan, current_step_idx, plan_status, result=diagnosis
            )
            exec_ctx.plan_manager.save_plan(exec_ctx.plan)
        if exec_ctx.structured_memory and step_ctx.step % 10 == 0:
            try:
                await exec_ctx.structured_memory.compress_history(model=exec_ctx.planner_model)
            except Exception as exc:
                logger.debug(f"Memory compression skipped: {exc}")
        last_action_type = (
            _classify_action_type(step_ctx.action_dict) if cfg.enable_prompt_delta else ""
        )
        exec_ctx.steps_log.append({
            "step": step_ctx.step,
            "plan": step_ctx.plan_output.get("Action"),
            "action": step_ctx.action_dict,
            "reasoning": step_ctx.plan_output.get("Reasoning", ""),
            "observation": step_ctx.plan_output.get("Observation", ""),
            "executor_result": step_ctx.executor_result,
            "verify_status": verify_status, "verify_diagnosis": diagnosis,
            "retries_used": step_ctx.attempt,
            "model_position": step_ctx.model_position,
            "executor_position": step_ctx.executor_position,
        })
        self._log_event(
            exec_ctx.instruction, step_ctx.plan_output, step_ctx.executor_result,
            verify_status, step_ctx.step, step_ctx.screenshot_path,
            exec_ctx.skill_name, exec_ctx.mode,
        )
        if not step_ctx.succeeded:
            logger.warning(f"Step {step_ctx.step} failed after all retries, continuing...")
        return last_action_type

    async def _capture_observation(
        self, exec_ctx: ExecuteContext, step_ctx: StepContext, shot_idx: int
    ) -> tuple[int, str | None]:
        """Delegate to the L1 GUI primitive."""
        return await self._primitive.capture_observation(exec_ctx, step_ctx, shot_idx)

    async def _verify_step(
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
        """Delegate to the L1 GUI primitive."""
        return await self._primitive.verify_step(
            exec_ctx, step_ctx,
            shot_idx=shot_idx, plan_action=plan_action,
            plan_output=plan_output, failed_attempts=failed_attempts,
            last_verify_result=last_verify_result, actor_model=actor_model,
        )

    async def _plan_one_step(
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
        """Delegate to the L1 GUI primitive."""
        return await self._primitive.plan_one_step(
            exec_ctx, step_ctx,
            guidance=guidance, skill=skill,
            failed_attempts=failed_attempts,
            last_verify_result=last_verify_result,
            last_action_type=last_action_type,
        )

    async def _ground_action(
        self,
        exec_ctx: ExecuteContext,
        step_ctx: StepContext,
        *,
        os_name: str,
    ) -> tuple[str, ToolResult | None]:
        """Delegate to the L1 GUI primitive."""
        return await self._primitive.ground_action(exec_ctx, step_ctx, os_name=os_name)

    # ------------------------------------------------------------------
    # GUI monitor overlay (real-time progress; best-effort, never raises)
    # ------------------------------------------------------------------

    def _monitor_write(self, **kwargs: Any) -> None:
        """Write state to the GUI monitor overlay (optional, never blocks)."""
        try:
            from syll.desktop.gui_monitor import write_gui_state
            write_gui_state(**kwargs)
        except Exception:
            pass

    def _monitor_launch(self) -> None:
        """Launch the GUI monitor overlay (idempotent)."""
        if self._monitor_launched:
            return
        try:
            from syll.desktop.gui_monitor import launch_gui_monitor
            launch_gui_monitor()
            self._monitor_launched = True
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

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
        """Delegate to the L1 GUI primitive."""
        return await self._primitive.execute_single_step(
            exec_ctx, step_ctx,
            guidance=guidance, skill=skill,
            failed_attempts=failed_attempts,
            last_verify_result=last_verify_result,
            last_action_type=last_action_type,
            actor_model=actor_model, os_name=os_name,
            shot_idx=shot_idx, instruction=instruction,
            max_steps=max_steps, timings=timings,
        )

    async def execute(
        self,
        instruction: str,
        skill_name: str = "",
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

        # ── Load skill & build guidance ───────────────────────────────
        # skill_name is optional: a recorded skill gives the planner a
        # trajectory to follow; without one (ad-hoc task) the planner runs
        # generically from task+screenshot and the TVAE verifier still gates
        # every action. An explicit-but-missing skill_name still errors
        # (likely a typo), so the strict path is preserved.
        if skill_name:
            skill = self._aloha_skill_store.load_skill(skill_name)
            if not skill:
                return ToolResult(text=f"Aloha skill '{skill_name}' not found")
            if not skill.trajectory and not skill.steps:
                return ToolResult(
                    text=f"Skill '{skill_name}' has no trajectory or steps"
                )
            guidance = self._build_guidance(skill)
            step_descs = self._extract_step_descriptions(skill, guidance)
        else:
            skill = None
            skill_name = "ad-hoc"  # label for downstream plan/skill paths
            guidance = (
                "(No guidance trajectory available — plan generically from "
                "the task description and the current screenshot.)"
            )
            step_descs = []

        mode = self._resolve_actor_mode(skill, actor_mode)

        # ── Resolve planner + actor endpoints, build purpose providers ─
        # 0b: route GUI model calls through per-purpose LLMProviders so
        # usage is observable (ContextMeter) instead of bare litellm calls.
        planner_model, planner_api_key, planner_api_base = (
            self._resolve_planner_endpoint()
        )
        planner_provider = self._make_purpose_provider(
            planner_api_key, planner_api_base
        )
        actor_model, actor_api_key, actor_api_base = (
            self._resolve_actor_endpoint()
        )
        actor_provider = self._make_purpose_provider(
            actor_api_key, actor_api_base
        )
        self._actor_provider = actor_provider
        self._on_actor_usage = self._on_usage_cb("gui_actor")

        # ── Phase 1: VerifiedPlanner + ActionVerifier ──────────────────
        planner = VerifiedPlanner(
            model=planner_model,
            os_name=os_name,
            api_key=planner_api_key,
            api_base=planner_api_base,
            provider=planner_provider,
            on_usage=self._on_usage_cb("gui_planner"),
        )
        verifier = (
            ActionVerifier(
                pixel_diff_threshold=cfg.pixel_diff_threshold,
                provider=actor_provider,
                on_usage=self._on_usage_cb("gui_verify"),
            )
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
            StructuredMemory(
                workspace=workspace,
                provider=planner_provider,
                on_usage=self._on_usage_cb("gui_compress"),
            )
            if cfg.enable_structured_memory
            else None
        )

        plan = None
        if plan_manager:
            plan = plan_manager.create_plan(skill_name, instruction, step_descs)
            plan_manager.save_plan(plan)

        # ── Phase 4: Spatial analyzer ─────────────────────────────────
        # Uses the ACTOR model (vision-capable) rather than the planner
        # model, which may be text-only or have unreliable vision — the
        # analyzer's job is to READ the screenshot, so it needs working
        # image input. (actor endpoint + provider built above.)
        spatial_analyzer = (
            SpatialAnalyzer(
                model=actor_model,
                api_key=actor_api_key,
                api_base=actor_api_base,
                provider=actor_provider,
                on_usage=self._on_usage_cb("gui_spatial"),
            )
            if cfg.enable_spatial_context
            else None
        )

        # ── Executor ──────────────────────────────────────────────────
        from syll.agent.aloha.act.executor import AlohaExecutor

        executor = AlohaExecutor(self._config, environment=self._environment)

        # ── Main enhanced loop ────────────────────────────────────────
        screenshots: list[str] = []
        steps_log: list[dict] = []
        action_history: list[str] = []
        last_verify_result: VerifyResult | None = None
        last_action_type: str = ""
        shot_idx = 0

        # ExecuteContext packs the setup locals so phase-2 sub-methods take
        # (exec_ctx, step_ctx) instead of a long parameter list.
        exec_ctx = ExecuteContext(
            cfg=cfg, skill_name=skill_name, instruction=instruction, mode=mode,
            planner_model=planner_model, planner=planner, verifier=verifier,
            executor=executor, spatial_analyzer=spatial_analyzer,
            structured_memory=structured_memory, plan_manager=plan_manager, plan=plan,
            screenshots=screenshots, steps_log=steps_log, action_history=action_history,
        )

        # Launch the GUI monitor overlay so progress is visible while the
        # (long, multi-step) tool runs — mirrors UITarsTool's per-step writes.
        self._monitor_launch()
        self._monitor_write(
            status="running", instruction=instruction,
            step=0, max_steps=steps_limit,
        )

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
            attempt = 0
            # Per-step failure ledger (G2PO state-action graph, INFERENCE side):
            # each entry is a failed EDGE on this screen — {action, position,
            # category, reason}. Injected into the planner on retry so it picks
            # a NEW edge instead of repeating the last failed one.
            # NOTE: per-step only (resets each step). G2PO's cross-trajectory
            # state aggregation is a training-time construct; at inference we
            # fold same-step retries on the same screen into one node.
            failed_attempts: list[FailedAttempt] = []

            # step_ctx carries per-attempt intermediate state through the
            # phase-2 sub-methods (_capture_observation / ...).
            step_ctx = StepContext(step=step)

            # ── Inner TVAE retry loop ─────────────────────────────────
            # Delegated to _execute_single_attempt (L1 primitive) so the same
            # single-step logic can be reused by GuiActionTool without duplicating
            # the capture/plan/ground/verify sequence.
            step_succeeded = False
            step_verify = None
            for attempt in range(max_retries + 1):
                step_ctx.attempt = attempt
                status, attempt_result, step_verify, last_verify_result, shot_idx = (
                    await self.execute_single_step(
                        exec_ctx, step_ctx,
                        guidance=guidance, skill=skill,
                        failed_attempts=failed_attempts,
                        last_verify_result=last_verify_result,
                        last_action_type=last_action_type,
                        actor_model=actor_model, os_name=os_name,
                        shot_idx=shot_idx, instruction=instruction,
                        max_steps=steps_limit,
                    )
                )
                if status == "proceed":
                    step_succeeded = True
                    break
                if status in ("done", "error"):
                    return attempt_result
                # status == "retry" -> try again within max_retries

            # ── Post-step recording ────────────────────────────────────
            # step_ctx is already populated by the sub-methods above
            # (_plan_one_step set plan_output; _ground_action set action_dict /
            # positions / executor_result; _verify_step returned step_verify).
            step_ctx.step_verify = step_verify
            step_ctx.succeeded = step_succeeded
            last_action_type = await self._record_step(exec_ctx, step_ctx)

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
        self._monitor_write(
            status="finished", instruction=instruction,
            step=steps_limit, max_steps=steps_limit,
            error=f"reached max steps {steps_limit}",
        )
        self._flush_skill_lessons(structured_memory, skill_name)
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

    def _planner_label(self) -> str:
        """Label used in completion messages."""
        return "enhanced planner"

    def _resolve_actor_mode(self, skill: Any, actor_mode: str | None) -> str:
        """Determine actor backend mode — always UI-TARS."""
        return "ui-tars"

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
        raise RuntimeError(
            "Planner endpoint not configured — set models.planner in config.json"
        )

    def _resolve_actor_endpoint(
        self,
    ) -> tuple[str, str | None, str | None]:
        """Return (model, api_key, api_base) for the actor.

        Used by vision-dependent components (e.g. SpatialAnalyzer) that
        must actually *read* a screenshot. The actor is the vision-capable
        model, whereas the planner may be text-only or have unreliable
        vision — so perception-to-text should route through the actor.
        """
        if self._syll_config:
            ep = self._syll_config.resolve_endpoint("actor")
            return ep.litellm_model, ep.api_key or None, ep.api_base
        raise RuntimeError(
            "Actor endpoint not configured — set models.actor in config.json"
        )

    def _resolve_workspace(self) -> Path:
        """Return the workspace path for persistent files."""
        workspace = Path(tempfile.gettempdir())
        if self._syll_config:
            workspace = (
                getattr(self._syll_config, "workspace_path", workspace)
                or workspace
            )
        return workspace

    def _log_action(self, record: dict) -> None:
        """Append one grounded action to ``{workspace}/audit/actions.jsonl``.

        Captures the actor's raw model coords (qwen 0-1000) AND the executor
        screen-pixel coords per action, so a grounding failure is diagnosable
        downstream (Performance / Runs tab). ``_audit_workspace`` is set by
        UnifiedSubagentManager to the run workspace; falls back to
        _resolve_workspace() so the loop path still logs. Best-effort —
        logging must never raise into the GUI loop.
        """
        try:
            ws = getattr(self, "_audit_workspace", None) or self._resolve_workspace()
            audit = Path(ws) / "audit"
            audit.mkdir(parents=True, exist_ok=True)
            with open(audit / "actions.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.debug(f"action log skipped: {exc}")

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


class GuiActionTool(EnhancedAlohaPlannerTool):
    """EnhancedAlohaPlannerTool registered under the name ``gui_action``.

    Lets ad-hoc GUI tasks (no recorded skill) run through the full TVAE
    verifier pipeline (planner → actor → pixel-diff + LLM expectation verify)
    instead of the verifier-less UITarsTool. ``skill_name`` is optional —
    without it the planner runs generically (no trajectory guidance); an
    explicit skill_name still must resolve (typo guard).
    """

    @property
    def name(self) -> str:
        return "gui_action"

    @property
    def description(self) -> str:
        return (
            "Perform ONE verified GUI step and return what happened. Takes a "
            "screenshot, plans the next action, executes it, verifies it changed "
            "the screen (TVAE pixel-diff + LLM expectation check), then returns "
            "the screenshot + step detail. Call repeatedly — one step per call — "
            "so you can check the result and decide the next sub-goal between "
            "steps. Pass max_steps>1 only to chain steps autonomously. Optional "
            "skill_name follows a recorded trajectory."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "What GUI task to perform",
                },
                "skill_name": {
                    "type": "string",
                    "description": (
                        "Optional recorded Aloha skill to use as guidance "
                        "trajectory. Omit for ad-hoc tasks."
                    ),
                },
                "max_steps": {
                    "type": "integer",
                    "description": (
                        "Ignored — gui_action always does exactly ONE step per "
                        "call (plan -> actor -> verify). Call gui_action again "
                        "to do the next step. Kept for backward compatibility."
                    ),
                    "minimum": 1,
                    "maximum": 50,
                },
                "prior_failures": {
                    "type": "array",
                    "description": (
                        "Optional: pass the previous step's failure(s) back so "
                        "the planner avoids repeating them. Each entry's "
                        "category/reason come from the prior [STEP] result."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "COORD_OFF/ELEMENT_ABSENT/OCCLUDED/LOADING/NO_VISUAL_FEEDBACK/WORKFLOW_ORDER/UNKNOWN",
                            },
                            "reason": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["instruction"],
        }

    async def execute(
        self,
        instruction: str,
        skill_name: str = "",
        max_steps: int | None = None,
        prior_failures: list[dict] | None = None,
        **kwargs: Any,
    ) -> str | ToolResult:
        """ONE verified GUI step per call (no inner retry, no multi-step loop).

        plan → actor → execute → TVAE verify, then return the screenshot + a
        structured per-step result. The caller (outer agent) handles all
        iteration, retry, and reflection — call gui_action again for the next
        step. ``max_steps`` is ignored (always one step). ``prior_failures``
        lets the caller pass back the previous step's Category/Reason so the
        planner avoids repeating a known-bad action (structured, same injection
        path the inner loop used to use).
        """
        # Fall back to the parent multi-step pipeline when enhanced is disabled.
        if not self._enhanced_config.ALL_ENABLED:
            return await super().execute(
                instruction, skill_name=skill_name, max_steps=max_steps, **kwargs
            )

        cfg = self._enhanced_config
        os_name = platform.system()
        # spatial_analyzer is intentionally NOT built (see below) — its 800-
        # token UI description is too slow per step. On failure the lighter
        # diagnose_no_change classifies plan-vs-grounding instead. llm_verify
        # (the diagnose call) runs on the first NO_CHANGE — that IS the light
        # error classifier the design wants.

        # ── Lean setup ── no plan_manager / structured_memory / plan / trace
        # (they're per-call accumulators with zero cross-call value when each
        # call is one step; the outer agent + conversation history carry state).
        if skill_name:
            skill = self._aloha_skill_store.load_skill(skill_name)
            if not skill:
                return ToolResult(text=f"Aloha skill '{skill_name}' not found")
            if not skill.trajectory and not skill.steps:
                return ToolResult(text=f"Skill '{skill_name}' has no trajectory or steps")
            guidance = self._build_guidance(skill)
        else:
            skill = None
            skill_name = "ad-hoc"
            guidance = (
                "(No guidance trajectory available — plan generically from "
                "the task description and the current screenshot.)"
            )
        mode = self._resolve_actor_mode(skill, None)

        planner_model, planner_api_key, planner_api_base = (
            self._resolve_planner_endpoint()
        )
        planner_provider = self._make_purpose_provider(planner_api_key, planner_api_base)
        actor_model, actor_api_key, actor_api_base = self._resolve_actor_endpoint()
        actor_provider = self._make_purpose_provider(actor_api_key, actor_api_base)
        self._actor_provider = actor_provider
        self._on_actor_usage = self._on_usage_cb("gui_actor")

        planner = VerifiedPlanner(
            model=planner_model, os_name=os_name,
            api_key=planner_api_key, api_base=planner_api_base,
            provider=planner_provider, on_usage=self._on_usage_cb("gui_planner"),
        )
        verifier = (
            ActionVerifier(
                pixel_diff_threshold=cfg.pixel_diff_threshold,
                provider=actor_provider, on_usage=self._on_usage_cb("gui_verify"),
            ) if cfg.enable_tvae_verification else None
        )
        # spatial_analyzer dropped — its 800-token UI description is too slow
        # per step. The planner plans from screenshot + task (+ prior_failures
        # on retry); on failure, diagnose_no_change classifies the error
        # lightly (plan vs grounding) for a targeted retry.
        spatial_analyzer = None
        from syll.agent.aloha.act.executor import AlohaExecutor
        executor = AlohaExecutor(self._config)

        exec_ctx = ExecuteContext(
            cfg=cfg, skill_name=skill_name, instruction=instruction, mode=mode,
            planner_model=planner_model, planner=planner, verifier=verifier,
            executor=executor, spatial_analyzer=spatial_analyzer,
            structured_memory=None, plan_manager=None, plan=None,
            screenshots=[], steps_log=[], action_history=[],
        )
        step_ctx = StepContext(step=1, attempt=0)

        # prior_failures → FailedAttempt list (structured injection into planner)
        failed_attempts: list[FailedAttempt] = []
        for pf in (prior_failures or []):
            if isinstance(pf, dict):
                failed_attempts.append(FailedAttempt(
                    action="", position=None,
                    category=FailureCategory.from_string(str(pf.get("category", "UNKNOWN"))),
                    reason=str(pf.get("reason", "")),
                ))

        self._monitor_launch()
        self._monitor_write(status="running", instruction=instruction, step=0, max_steps=1)

        # ── Single step: delegated to _execute_single_attempt (L1 primitive) ──
        # Per-phase wall-time tracking is accumulated inside the primitive and
        # mirrored here for post-hoc analysis / monitor overlay.
        _timings: dict[str, int] = {}

        status, result, step_verify, _, shot_idx = await self.execute_single_step(
            exec_ctx, step_ctx,
            guidance=guidance, skill=skill,
            failed_attempts=failed_attempts,
            last_verify_result=None,
            last_action_type="",
            actor_model=actor_model,
            os_name=os_name,
            shot_idx=0,
            instruction=instruction,
            max_steps=1,
            timings=_timings,
        )

        for phase, ms in _timings.items():
            logger.info(f"[gui_action] {phase}: {ms}ms")

        if status == "done":
            return self._format_single_step_result(exec_ctx, step_ctx, outcome="DONE", step_verify=None)
        if status == "error":
            return result

        # status in ("proceed", "retry"). For a single-step tool there is no
        # inner retry: classify the observable outcome and return it.
        ground_status = "retry" if (status == "retry" and step_verify is None) else "proceed"
        step_ctx.step_verify = step_verify
        step_ctx.succeeded = (
            ground_status == "proceed" and (step_verify is None or step_verify.is_success)
        )
        await self._record_step(exec_ctx, step_ctx)  # event log + steps_log (audit)

        outcome = self._classify_outcome(ground_status=ground_status, step_verify=step_verify)
        _total_timing = " ".join(f"{k}{v/1000:.1f}s" for k, v in _timings.items())
        self._monitor_write(
            status="finished" if outcome == "SUCCESS" else "error",
            instruction=instruction, step=1, max_steps=1,
            error=None if outcome == "SUCCESS" else outcome,
            thought=f"⏱ {_total_timing}",
            timing=_timings,
        )
        return self._format_single_step_result(exec_ctx, step_ctx, outcome=outcome, step_verify=step_verify)

    # ------------------------------------------------------------------
    # per-step return helpers
    # ------------------------------------------------------------------

    def _classify_outcome(self, *, ground_status: str, step_verify) -> str:
        """Map the one attempt's outcomes to a single status word."""
        if ground_status == "retry":
            return "ERROR"
        if step_verify is None:
            return "UNCERTAIN"
        if step_verify.is_success:
            return "SUCCESS"
        if step_verify.is_no_change:
            return "NO_CHANGE"
        return "UNCERTAIN"

    # Plan-vs-grounding error buckets — each has a targeted retry strategy.
    # Replaces the slow spatial UI-description with a light classify that
    # tells the outer agent HOW to retry (re-ground vs re-plan).
    _ERROR_BUCKETS: dict[str, tuple[str, str]] = {
        # GROUNDING: the action was right but coords/execution missed → re-ground
        "COORD_OFF": (
            "GROUNDING",
            "Re-ground: re-call gui_action with the SAME goal — the actor will "
            "re-locate the target from a fresh screenshot (coords were off).",
        ),
        "OCCLUDED": (
            "GROUNDING",
            "Re-ground: the target was covered by a popup/overlay — dismiss it "
            "or re-call gui_action to re-ground once it's clear.",
        ),
        # PLAN: the chosen action/target was wrong → re-plan differently
        "ELEMENT_ABSENT": (
            "PLAN",
            "Re-plan: the target isn't on screen — scroll, navigate, or wait "
            "for it to load, then re-call gui_action.",
        ),
        "WORKFLOW_ORDER": (
            "PLAN",
            "Re-plan: wrong step order — reassess which action to do first.",
        ),
        "LOADING": (
            "PLAN",
            "Re-plan: the UI was mid-transition — wait, then re-call gui_action.",
        ),
        # UNCERTAIN
        "NO_VISUAL_FEEDBACK": (
            "UNCERTAIN",
            "The action may have worked with no visible change (e.g. background "
            "save) — verify by other means before retrying.",
        ),
        "UNKNOWN": (
            "UNCERTAIN",
            "Cause unclear — inspect the screenshot and decide whether to retry.",
        ),
    }

    @classmethod
    def _error_bucket(cls, category: str) -> tuple[str, str]:
        """(bucket, targeted-next-guidance) for a diagnose Category. Defaulted
        to UNCERTAIN for unknown categories."""
        return cls._ERROR_BUCKETS.get(
            (category or "").upper(), cls._ERROR_BUCKETS["UNKNOWN"]
        )

    def _format_single_step_result(
        self, exec_ctx, step_ctx, *, outcome: str, step_verify
    ) -> ToolResult:
        """Build the structured per-step return: a [STEP]/[DONE] tag + action +
        verify verdict + Category/Diagnosis at the top, plus before/after
        screenshots. Read by the outer agent to drive the next call."""
        plan_output = step_ctx.plan_output or {}
        tag = "DONE" if outcome == "DONE" else "STEP"
        lines = [f"[{tag}] gui_action: {outcome}", ""]

        if outcome == "DONE":
            lines.append("The planner/actor signaled the task is complete.")
            obs = plan_output.get("Observation", "")
            if obs:
                lines.append(f"Observation: {obs}")
            lines.append("")
            lines.append("Next: report success to the user — no further gui_action needed.")
        else:
            plan_action = step_ctx.plan_action or ""
            if plan_action:
                lines.append(f"Action: {plan_action}")
            reasoning = plan_output.get("Reasoning", "")
            if reasoning:
                lines.append(f"Reasoning: {reasoning}")
            if outcome == "ERROR":
                lines.append(f"Executor FAILED: {step_ctx.executor_result}")
            else:
                status_word = step_verify.status.value if step_verify else "UNCERTAIN"
                pd = f" (pixel_diff={step_verify.pixel_diff_score:.4f})" if step_verify else ""
                lines.append(f"Verify: {status_word}{pd}")
                cat = step_verify.category if step_verify else ""
                if cat:
                    bucket, _ = self._error_bucket(cat)
                    lines.append(f"  Error type: {bucket}  (Category: {cat})")
                diag = step_verify.diagnosis if step_verify else ""
                if diag:
                    lines.append(f"  Diagnosis: {diag}")
            mp = step_ctx.model_position
            ep = step_ctx.executor_position
            if mp is not None or ep is not None:
                lines.append(f"Position: model={mp} -> screen={ep}")
            lines.append("")
            if outcome == "SUCCESS":
                lines.append(
                    "Next: this step worked — call gui_action again for the NEXT "
                    "step toward the goal. Keep going (one step per call) until "
                    "you receive [DONE] or the overall task is genuinely complete. "
                    "Do NOT stop after a single success."
                )
            elif outcome == "NO_CHANGE":
                cat = step_verify.category if step_verify else ""
                _, guidance = self._error_bucket(cat)
                lines.append(f"Next: {guidance}")
            elif outcome == "UNCERTAIN":
                lines.append(
                    "Next: verification was inconclusive (likely a small but "
                    "real change — focus/select/value). Inspect the attached "
                    "screenshot and decide whether the action worked."
                )
            else:  # ERROR
                lines.append(
                    "Next: the executor could not perform the action. Check "
                    "the target and re-call gui_action."
                )

        media = self._key_screenshots(exec_ctx.screenshots)
        return ToolResult(text="\n".join(lines), media=media)
