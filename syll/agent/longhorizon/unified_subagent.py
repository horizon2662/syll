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
from syll.agent.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool
from syll.agent.tools.registry import ToolRegistry
from syll.agent.tools.shell import ExecTool
from syll.agent.tools.web import WebFetchTool, WebSearchTool
from syll.bus.events import InboundMessage
from syll.bus.queue import MessageBus
from syll.providers.base import LLMProvider

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
    ):
        from syll.config.schema import ExecToolConfig

        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self.mcp_manager = mcp_manager
        self.max_iterations = max_iterations
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
        completed_normally = False
        iteration = 0
        while iteration < self.max_iterations:
            if done.done():  # branch already returned via ReturnTool
                break
            iteration += 1
            bb.write_progress(f"iteration {iteration}")

            response = await self.provider.chat(
                messages=messages,
                tools=tools.get_definitions(),
                model=self.model,
            )
            last_response = response.content or ""

            if not response.has_tool_calls:
                # Model gave a final text answer without calling return.
                # This is a NORMAL completion -> fold it as a success summary.
                completed_normally = True
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
            )
            for tc in response.tool_calls:
                out = await tools.execute(tc.name, tc.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": out,
                    }
                )

        if done.done():
            r = done.result()
            r.iterations_used = iteration
            return r

        if completed_normally:
            # Fold the model's final text answer as the summary (NOT a failure).
            return SubagentResult(
                run_id=run_id,
                status="ok",
                summary=last_response[:2000] or "(completed with no text)",
                iterations_used=iteration,
            )

        # Truly exhausted the budget (kept calling tools, never finished).
        logger.warning(
            f"subagent[{run_id}] hit iteration budget ({self.max_iterations}) "
            "without finishing; folding last response as failure."
        )
        return SubagentResult(
            run_id=run_id,
            status="failed",
            summary=last_response[:2000] or "(no output)",
            diagnosis=(
                f"Hit iteration budget ({self.max_iterations}) without finishing."
            ),
            iterations_used=iteration,
        )

    # ------------------------------------------------------------------
    # tool set + prompt
    # ------------------------------------------------------------------
    def _build_tools(self) -> ToolRegistry:
        tools = ToolRegistry()
        allowed = self.workspace if self.restrict_to_workspace else None
        tools.register(ReadFileTool(allowed_dir=allowed))
        tools.register(WriteFileTool(allowed_dir=allowed))
        tools.register(ListDirTool(allowed_dir=allowed))
        tools.register(
            ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
            )
        )
        tools.register(WebSearchTool(api_key=self.brave_api_key))
        tools.register(WebFetchTool())

        if self.mcp_manager is not None:
            for adapter in self.mcp_manager.iter_propagating_tools():
                if not tools.has(adapter.name):
                    tools.register(adapter)
        return tools

    def _build_prompt(self, contract: SubagentContract, skill_ctx: str) -> str:
        return f"""# Subagent (isolated context)

You operate with your OWN context window; the main agent never sees your
intermediate steps.

## Hard rules (do not violate)
1. DO THE WORK FIRST. Use tools to actually perform the task BEFORE calling
   `return`. Never report success for something you did not do -- claimed
   artifacts are VERIFIED to exist afterward, so a fabricated success is caught
   and treated as a failure.
2. Write files using ABSOLUTE paths under the workspace below. Relative paths
   resolve unpredictably; always prefix with the workspace path shown here.
3. Your `return.summary` must state what you ACTUALLY did (commands run, files
   read/written), not a plausible-sounding guess. If you could not complete the
   task, return status="failed" with a diagnosis -- do not pretend.
4. Stay strictly within the assigned task -- do not branch or take side tasks.

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
        }

    def get_running_count(self) -> int:
        return len(self._running)
