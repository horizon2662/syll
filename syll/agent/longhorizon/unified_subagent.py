"""Unified subagent manager: one abstraction, three run modes, with fold.

Replaces the two incompatible subagent abstractions in the current codebase:
- ``SubagentManager`` (agent/subagent.py): fire-and-forget; pushes the full
  final string into a single system message; ``_failure_counts`` only
  suppresses repeated failures (no replan).
- ``GUIExecuteSubAgent`` (.../enhanced/gui_execute_subagent.py): a synchronous
  single-step executor, not a real isolated-context subagent.

This module unifies them behind ``SubagentContract`` + ``Blackboard`` +
``SkillMemory`` and adds:

1. **Fold** (Context-Folding, arXiv:2510.11967, ICML'26): a subagent runs in
   an isolated context and returns ONLY a condensed ``summary`` plus artifact
   references. Its intermediate steps never enter the main agent's context.
   (FoldAgent uses in-memory history copy for training; we use the filesystem
   blackboard so it survives cross-session / multi-process -- same principle.)
2. **Filesystem blackboard**: results persist as files; the main agent reads
   them on demand instead of receiving a fat message (Anthropic's "avoid the
   game of telephone"; orchestration survey State unit, arXiv:2601.13671).
3. **Diagnosis -> Align** (TaskWeave FPDA, arXiv:2606.01199): on failure the
   ``diagnosis`` is returned (NOT silently suppressed) so the main agent can
   replan.
4. **Notetaker gating** (Mobile-Agent-v3, arXiv:2602.16855): lessons are
   written to the skill's SKILL.md only when warranted.

Backward compatible: existing callers of ``spawn(task, label, ...)`` still work.
The extra contract fields are optional kwargs.

Does **not** modify the original ``subagent.py``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from syll.agent.tools.base import Tool
from syll.agent.tools.bundles import register_core_tools
from syll.agent.tools.registry import ToolRegistry
from syll.bus.events import InboundMessage
from syll.bus.queue import MessageBus
from syll.providers.base import LLMProvider
from syll.sandbox.environment import Environment, LocalEnvironment

from .blackboard import Blackboard
from .contract import SubagentContract, SubagentResult
from .skill_memory import SkillMemory

if TYPE_CHECKING:
    from syll.agent.mcp import MCPManager


class ReturnTool(Tool):
    """The fold primitive: the ONLY channel a subagent uses to hand back its result.

    Mirrors FoldAgent's ``return`` action. It carries a SUMMARY, not the full
    work -- large outputs must already be written to the blackboard. This is
    what keeps the main agent's context clean (the "fold").
    """

    def __init__(self, blackboard: Blackboard, done: "asyncio.Future[SubagentResult]"):
        self._bb = blackboard
        self._done = done

    @property
    def name(self) -> str:
        return "return"

    @property
    def description(self) -> str:
        return (
            "Finish your task and return a CONCISE summary to the main agent. "
            "Write any large output (code, reports, data) to files first; pass "
            "only file paths and a short summary here."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Concise result summary (<= ~2000 tokens).",
                },
                "status": {
                    "type": "string",
                    "enum": ["ok", "failed", "needs_input"],
                    "description": (
                        "ok=done; failed=could not complete (give diagnosis); "
                        "needs_input=blocked, must ask the main agent."
                    ),
                },
                "diagnosis": {
                    "type": "string",
                    "description": "If failed/needs_input, why. Feeds the main agent's replan.",
                },
                "artifacts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths of files you produced (relative to workspace).",
                },
                "lessons": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional reusable how-to / pitfall notes for this skill's memory.",
                },
            },
            "required": ["summary", "status"],
        }

    async def execute(
        self,
        summary: str,
        status: str = "ok",
        diagnosis: str = "",
        artifacts: list[str] | None = None,
        lessons: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        result = SubagentResult(
            run_id=self._bb.run_id,
            status=status,  # type: ignore[arg-type]
            summary=summary,
            artifacts=artifacts or [],
            diagnosis=diagnosis,
            lessons=lessons or [],
        )
        if not self._done.done():
            self._done.set_result(result)
        return "Returned to main agent."


class UnifiedSubagentManager:
    """Drop-in replacement for ``SubagentManager`` with fold + blackboard + skills."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
        mcp_manager: "MCPManager | None" = None,
        max_iterations: int = 15,
        gui_config: Any = None,
        syll_config: Any = None,
        event_store: Any = None,
        context_meter: Any = None,
        skill_memory: Any = None,
        environment: Environment | None = None,
    ):
        from syll.config.schema import ExecToolConfig

        self.environment = environment or LocalEnvironment(workspace_root=workspace)
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self.mcp_manager = mcp_manager
        self.max_iterations = max_iterations
        # GUI + app config so subagents can operate the desktop (reuses the
        # ghost's v2 GUI stack) and resolve model endpoints for planner/actor.
        self.gui_config = gui_config
        self.syll_config = syll_config
        self.event_store = event_store
        # 0b: ContextMeter threaded from the Runner so GUI model calls
        # (planner/actor/spatial/verify) record usage into this run's
        # context_curve.jsonl instead of staying invisible.
        self.context_meter = context_meter
        # L2: SkillMemory threaded from the Runner so failed GUI-step
        # diagnoses can be lifted into SKILL.md (lesson upflow).
        self.skill_memory = skill_memory
        self._running: dict[str, asyncio.Task[None]] = {}

    # ------------------------------------------------------------------
    # public API (signature compatible with SubagentManager.spawn)
    # ------------------------------------------------------------------
    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        **contract_kwargs: Any,
    ) -> str:
        """Spawn a subagent. Extra contract fields (objective, skill, mode,
        context_slice, dependencies, ...) are optional kwargs."""
        run_id = str(uuid.uuid4())[:8]
        display = label or (task[:30] + ("..." if len(task) > 30 else ""))
        contract = SubagentContract(task=task, **contract_kwargs)
        if not contract.skill and label:
            contract.skill = label

        bg = asyncio.create_task(
            self._run(run_id, contract, display, origin_channel, origin_chat_id)
        )
        self._running[run_id] = bg
        bg.add_done_callback(lambda _: self._running.pop(run_id, None))
        logger.info(f"Spawned subagent [{run_id}]: {display}")
        return (
            f"Subagent[{display}] started (id: {run_id}). "
            f"Artifacts -> workspace/agents/{run_id}/"
        )

    async def run_sync(
        self,
        task: str,
        label: str | None = None,
        **contract_kwargs: Any,
    ) -> SubagentResult:
        """Spawn a subagent and block until it finishes; return the folded result.

        This is the orchestrator-friendly entry point: the runner spawns a
        subagent per plan step and awaits its ``SubagentResult`` directly,
        instead of consuming a system message from the bus."""
        run_id = str(uuid.uuid4())[:8]
        display = label or (task[:30] + ("..." if len(task) > 30 else ""))
        contract = SubagentContract(task=task, **contract_kwargs)
        if not contract.skill and label:
            contract.skill = label

        bg = asyncio.create_task(
            self._run(run_id, contract, display, "cli", "direct")
        )
        self._running[run_id] = bg
        try:
            await bg
        except Exception as exc:
            logger.error(f"subagent[{run_id}] task crashed: {exc}")
        finally:
            self._running.pop(run_id, None)

        bb = Blackboard(self.workspace, run_id)
        d = bb.read_result() or {}
        return SubagentResult(
            run_id=run_id,
            status=d.get("status", "failed"),  # type: ignore[arg-type]
            summary=d.get("summary", ""),
            artifacts=d.get("artifacts", []),
            diagnosis=d.get("diagnosis", ""),
            lessons=d.get("lessons", []),
            iterations_used=d.get("iterations_used", 0),
            tokens_in=d.get("tokens_in", 0),
            tokens_out=d.get("tokens_out", 0),
            last_prompt_tokens=d.get("last_prompt_tokens", 0),
        )

    # ------------------------------------------------------------------
    # the run lifecycle: contract -> isolated branch -> fold -> announce
    # ------------------------------------------------------------------
    async def _run(
        self,
        run_id: str,
        contract: SubagentContract,
        label: str,
        origin_channel: str,
        origin_chat_id: str,
    ) -> None:
        bb = Blackboard(self.workspace, run_id)
        bb.write_contract(
            {
                "task": contract.task,
                "objective": contract.objective,
                "mode": contract.mode,
                "skill": contract.skill,
                "dependencies": contract.dependencies,
            }
        )

        loop = asyncio.get_running_loop()
        done: asyncio.Future[SubagentResult] = loop.create_future()
        ret = ReturnTool(bb, done)

        try:
            result = await self._run_isolated(run_id, contract, bb, ret, done)
        except Exception as exc:  # the branch crashed
            logger.error(f"subagent[{run_id}] crashed: {exc}")
            result = SubagentResult(
                run_id=run_id, status="failed", diagnosis=f"Crash: {exc}"
            )

        bb.write_result(self._result_to_dict(result))

        # Notetaker: gate skill-memory writes (Mobile-Agent-v3 rule)
        if contract.skill and result.lessons:
            SkillMemory(self.workspace, contract.skill).ingest(
                result.lessons, result.ok
            )

        await self._announce(run_id, label, result, origin_channel, origin_chat_id)

    async def _run_isolated(
        self,
        run_id: str,
        contract: SubagentContract,
        bb: Blackboard,
        ret: ReturnTool,
        done: "asyncio.Future[SubagentResult]",
    ) -> SubagentResult:
        """The branch: isolated context, runs until it calls ``return``, stops
        with a text answer, or the iteration budget is exhausted."""
        tools = self._build_tools()
        tools.register(ret)  # the fold primitive

        skill_ctx = ""
        if contract.skill:
            skill_ctx = SkillMemory(self.workspace, contract.skill).context_for_subagent()

        system_prompt = self._build_prompt(contract, skill_ctx)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": contract.render_prompt()},
        ]

        last_response = ""
        # Token accumulators for the runner's DecisionUnit (bubbled via the
        # fold). last_prompt_tokens ~= peak working context this subagent
        # reached, since context only grows inside the loop.
        tokens_in = tokens_out = 0
        last_prompt_tokens = 0

        def on_usage(resp, it):
            nonlocal tokens_in, tokens_out, last_prompt_tokens
            usage = resp.usage or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            tokens_in += prompt_tokens
            tokens_out += int(usage.get("completion_tokens", 0) or 0)
            last_prompt_tokens = prompt_tokens

        # Shared tool-calling primitive (L1). pre_iteration mirrors the old
        # `if done.done(): break` (ReturnTool completion); on_iteration_start
        # writes blackboard progress; on_usage accumulates the tokens bubbled
        # out via the fold. Message formatting is now unified (incl.
        # ToolResult.media + reasoning_content — previously dropped here).
        from syll.agent.loop_core import run_tool_loop

        loop_result = await run_tool_loop(
            self.provider,
            tools,
            messages,
            model=self.model,
            max_iterations=self.max_iterations,
            on_usage=on_usage,
            pre_iteration=lambda it: done.done(),
            on_iteration_start=lambda it: bb.write_progress(f"iteration {it}"),
        )
        completed_normally = loop_result.stop_reason == "no_tool_calls"
        last_response = (
            (loop_result.final_response.content or "") if loop_result.final_response else ""
        )
        iteration = loop_result.iterations

        if done.done():
            # Stamp the accumulated tokens onto the result ReturnTool built.
            return self._stamp_accounting(
                done.result(), iteration, tokens_in, tokens_out, last_prompt_tokens
            )

        if completed_normally:
            # Fold the model's final text answer as the summary (NOT a failure).
            return self._stamp_accounting(
                SubagentResult(
                    run_id=run_id,
                    status="ok",
                    summary=last_response[:2000] or "(completed with no text)",
                ),
                iteration, tokens_in, tokens_out, last_prompt_tokens,
            )

        # Truly exhausted the budget (kept calling tools, never finished).
        logger.warning(
            f"subagent[{run_id}] hit iteration budget ({self.max_iterations}) "
            "without finishing; folding last response as failure."
        )
        return self._stamp_accounting(
            SubagentResult(
                run_id=run_id,
                status="failed",
                summary=last_response[:2000] or "(no output)",
                diagnosis=(
                    f"Hit iteration budget ({self.max_iterations}) without finishing."
                ),
            ),
            iteration, tokens_in, tokens_out, last_prompt_tokens,
        )

    @staticmethod
    def _stamp_accounting(
        result: SubagentResult,
        iteration: int,
        tokens_in: int,
        tokens_out: int,
        last_prompt_tokens: int,
    ) -> SubagentResult:
        """Fold the accumulated iteration/token counters onto a SubagentResult.

        The three exit paths (ReturnTool completion / normal text answer /
        budget exhausted) all stamp the same four counters; this is the single
        place that mapping lives. Mutates and returns ``result``."""
        result.iterations_used = iteration
        result.tokens_in = tokens_in
        result.tokens_out = tokens_out
        result.last_prompt_tokens = last_prompt_tokens
        return result

    # ------------------------------------------------------------------
    # tool set + prompt
    # ------------------------------------------------------------------
    def _build_tools(self) -> ToolRegistry:
        tools = ToolRegistry()
        # Resolve the Brave key from syll_config when not passed explicitly
        # (the long-horizon runner constructs this manager without one, which
        # left web search keyless).
        brave = self.brave_api_key
        if not brave and self.syll_config is not None:
            try:
                brave = self.syll_config.tools.web.search.api_key or None
            except Exception:
                brave = None
        register_core_tools(
            tools,
            workspace=self.workspace,
            restrict_to_workspace=self.restrict_to_workspace,
            exec_config=self.exec_config,
            brave_api_key=brave,
            environment=self.environment,
        )

        # Video-learning: the runner can autonomously watch an online tutorial
        # (search -> download -> analyze frames -> SKILL.md) to acquire an
        # unfamiliar GUI procedure, then follow it via gui_action_planned. The
        # frame analyzer reuses the vision (actor) endpoint.
        try:
            from syll.agent.tools.video_learn import VideoLearnTool

            if self.syll_config is not None:
                ep = self.syll_config.resolve_endpoint("actor")
                vid_model = ep.litellm_model or self.model or ""
                vid_key, vid_base = ep.api_key, (ep.api_base or "")
            else:
                vid_model, vid_key, vid_base = (self.model or ""), "", ""
            tools.register(
                VideoLearnTool(
                    model=vid_model, api_key=vid_key, api_base=vid_base,
                    workspace=self.workspace, brave_api_key=brave or "",
                )
            )
        except Exception as exc:  # optional dep (yt-dlp) missing -> degrade
            logger.debug(f"video_learn tool not registered: {exc}")

        if self.mcp_manager is not None:
            for adapter in self.mcp_manager.iter_propagating_tools():
                if not tools.has(adapter.name):
                    tools.register(adapter)

        # GUI tools — reuse the ghost's v2 GUI stack verbatim (screenshot +
        # UI-TARS single-shot + Enhanced Planner/Actor TVAE loop). Gated on
        # gui_config.enabled, exactly like AgentLoop._register_default_tools.
        if self.gui_config is not None and getattr(self.gui_config, "enabled", False):
            from syll.agent.aloha_gui_skill import AlohaSkillStore
            from syll.agent.tools.screenshot import ScreenshotTool

            aloha_skill_store = AlohaSkillStore(self.workspace)
            tools.register(ScreenshotTool(environment=self.environment))

            # gui_action → Enhanced TVAE-verifier pipeline (skill optional),
            # matching AgentLoop — falls back to UITarsTool if Enhanced import
            # fails. Gets the same telemetry wiring as gui_action_planned.
            try:
                from syll.agent.aloha.act.enhanced.enhanced_planner_tool import (
                    GuiActionTool,
                )
                gui_action_tool = GuiActionTool(
                    self.gui_config, aloha_skill_store, syll_config=self.syll_config,
                    environment=self.environment,
                )
            except ImportError:
                from syll.agent.gui_skill import GUISkillStore
                from syll.agent.tools.ui_tars import UITarsTool

                gui_skill_store = GUISkillStore(self.workspace)
                gui_action_tool = UITarsTool(
                    self.gui_config,
                    gui_skill_store=gui_skill_store,
                    aloha_skill_store=aloha_skill_store,
                    syll_config=self.syll_config,
                    environment=self.environment,
                )
            if self.event_store is not None:
                gui_action_tool._event_store = self.event_store
            if self.context_meter is not None:
                gui_action_tool._context_meter = self.context_meter
            if self.skill_memory is not None:
                gui_action_tool._skill_memory = self.skill_memory
            gui_action_tool._audit_workspace = self.workspace
            tools.register(gui_action_tool)

            try:
                from syll.agent.aloha.act.enhanced.enhanced_planner_tool import (
                    EnhancedAlohaPlannerTool,
                )
                planner_tool = EnhancedAlohaPlannerTool(
                    self.gui_config, aloha_skill_store, syll_config=self.syll_config,
                    environment=self.environment,
                )
            except ImportError:
                from syll.agent.tools.aloha_planner_tool import AlohaPlannerTool
                planner_tool = AlohaPlannerTool(
                    self.gui_config, aloha_skill_store, syll_config=self.syll_config,
                    environment=self.environment,
                )
            if self.event_store is not None:
                planner_tool._event_store = self.event_store
            # 0b: thread the Runner's ContextMeter so GUI model calls
            # (planner/actor/spatial/verify) record usage into this run's
            # context_curve.jsonl instead of staying invisible.
            if self.context_meter is not None:
                planner_tool._context_meter = self.context_meter
            if self.skill_memory is not None:
                planner_tool._skill_memory = self.skill_memory
            # Give the planner tool the run workspace so it logs grounded
            # actions to {run_workspace}/audit/actions.jsonl (Performance/Runs).
            planner_tool._audit_workspace = self.workspace
            tools.register(planner_tool)

        return tools

    def _build_prompt(self, contract: SubagentContract, skill_ctx: str) -> str:
        import platform

        # GUI delegation: when GUI tools are available, the subagent must DRIVE
        # the UI through `gui_action` (the Enhanced verifier pipeline: vision
        # actor + TVAE per-step verify), NOT via shell/SendKeys or by eyeballing
        # a screenshot. Shell stays for non-GUI work only.
        gui_block = ""
        if self.gui_config is not None and getattr(self.gui_config, "enabled", False):
            gui_block = """
## GUI / desktop operations (IMPORTANT)
- To operate ANY graphical UI — open apps, click, type into windows, menus,
  draw shapes, navigate tabs — call `gui_action(instruction="<what>")`.
  It captures the screen, grounds via a VISION actor, performs the action, and
  verifies it (TVAE pixel-diff + expectation check). This is the ONLY correct
  way to drive the GUI. skill_name is optional (pass one only if you have a
  recorded skill to follow).
- DO NOT use exec/shell/PowerShell/SendKeys/osascript to manipulate windows.
- DO NOT take a screenshot and guess coordinates — you cannot see the image;
  the GUI tool sees it for you.
- Use exec/shell ONLY for non-GUI work (run scripts, install deps, file ops).
- If you do NOT know the exact UI steps for an unfamiliar app or task, FIRST
  call `video_learn(task="<app> <goal>")` to learn from an online tutorial; it
  writes a SKILL.md you can then follow via `gui_action(skill_name="<app>")`.
"""

        return f"""# Subagent (isolated context)

You operate with your OWN context window; the main agent never sees your
intermediate steps.

## Environment
Platform: {platform.system()} {platform.release()} ({platform.machine()}).
Use platform-appropriate commands only — never macOS-only commands (e.g.
`osascript`) on Windows, or Windows-only commands on macOS.

## Hard rules (do not violate)
1. DO THE WORK FIRST. Use tools to actually perform the task BEFORE calling
   `return`. Never report success for something you did not do -- claimed
   artifacts are VERIFIED to exist afterward, so a fabricated success is caught
   and treated as a failure.
2. Write files using ABSOLUTE paths under the workspace below. Relative paths
   resolve unpredictably; always prefix with the workspace path shown here.
3. Your `return.summary` must state what you ACTUALLY did (commands run, files
   read/written, GUI actions performed via gui_action), not a plausible
   guess. If you could not complete the task, return status="failed" + diagnosis.
4. Stay strictly within the assigned task -- do not branch or take side tasks.
{gui_block}
{skill_ctx}

## Workspace (write all files under here, using absolute paths)
{self.workspace}
"""

    # ------------------------------------------------------------------
    # folded announce: only summary + a reference, NEVER the full output
    # ------------------------------------------------------------------
    async def _announce(
        self,
        run_id: str,
        label: str,
        result: SubagentResult,
        origin_channel: str,
        origin_chat_id: str,
    ) -> None:
        if result.ok:
            art = (
                f" Artifacts: {', '.join(result.artifacts)}"
                if result.artifacts
                else ""
            )
            content = (
                f"subagent[{label}] DONE: {result.summary}{art}\n"
                f"(full result: workspace/agents/{run_id}/result.json)"
            )
        else:
            # Diagnosis is surfaced, not suppressed -- the main agent replans.
            content = (
                f"subagent[{label}] FAILED: {result.diagnosis or result.summary}\n"
                f"(diagnosis in workspace/agents/{run_id}/result.json -- replan accordingly.)"
            )

        await self.bus.publish_inbound(
            InboundMessage(
                channel="system",
                sender_id="subagent",
                chat_id=f"{origin_channel}:{origin_chat_id}",
                content=content,
            )
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _result_to_dict(r: SubagentResult) -> dict:
        return {
            "run_id": r.run_id,
            "status": r.status,
            "summary": r.summary,
            "artifacts": r.artifacts,
            "diagnosis": r.diagnosis,
            "lessons": r.lessons,
            "iterations_used": r.iterations_used,
            "tokens_in": r.tokens_in,
            "tokens_out": r.tokens_out,
            "last_prompt_tokens": r.last_prompt_tokens,
        }

    def get_running_count(self) -> int:
        return len(self._running)
