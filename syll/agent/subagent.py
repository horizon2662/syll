"""Subagent manager for background task execution."""

import asyncio
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from syll.agent.tools.bundles import register_core_tools
from syll.agent.tools.registry import ToolRegistry
from syll.bus.events import InboundMessage
from syll.bus.queue import MessageBus
from syll.providers.base import LLMProvider
from syll.sandbox.environment import Environment

if TYPE_CHECKING:
    from syll.agent.mcp import MCPManager


class SubagentManager:
    """
    Manages background subagent execution.

    Subagents are lightweight agent instances that run in the background
    to handle specific tasks. They share the same LLM provider but have
    isolated context and a focused system prompt.
    """

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
        environment: Environment | None = None,
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
        self.environment = environment
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._failure_counts: dict[str, int] = {}  # task label -> consecutive failures
        self._max_announce_retries = 1  # max retries for failed subagent announcements

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> str:
        """
        Spawn a subagent to execute a task in the background.

        Args:
            task: The task description for the subagent.
            label: Optional human-readable label for the task.
            origin_channel: The channel to announce results to.
            origin_chat_id: The chat ID to announce results to.

        Returns:
            Status message indicating the subagent was started.
        """
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")

        origin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
        }

        # Create background task
        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin)
        )
        self._running_tasks[task_id] = bg_task

        # Cleanup when done
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))

        logger.info(f"Spawned subagent [{task_id}]: {display_label}")
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info(f"Subagent [{task_id}] starting task: {label}")

        try:
            # Build subagent tools (no message tool, no spawn tool)
            tools = ToolRegistry()
            register_core_tools(
                tools,
                workspace=self.workspace,
                restrict_to_workspace=self.restrict_to_workspace,
                exec_config=self.exec_config,
                brave_api_key=self.brave_api_key,
                environment=self.environment,
            )

            # Phase 1c: propagate MCP tools whose servers opted into
            # `propagate_to_subagents`. Built fresh per spawn — no shared
            # mutable state, so MCP hot-reload naturally affects the next
            # subagent. Skip on collision with a builtin (defense in depth;
            # subagents have a smaller registry so collisions are unlikely).
            if self.mcp_manager is not None:
                propagated = 0
                for adapter in self.mcp_manager.iter_propagating_tools():
                    if tools.has(adapter.name):
                        logger.warning(
                            f"subagent[{task_id}] MCP tool {adapter.name!r} "
                            "collides with builtin; skipping"
                        )
                        continue
                    tools.register(adapter)
                    propagated += 1
                if propagated:
                    logger.info(
                        f"subagent[{task_id}] propagated {propagated} MCP tool(s)"
                    )

            # Build messages with subagent-specific prompt
            system_prompt = self._build_subagent_prompt(task)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            # Run agent loop (limited iterations) — shared primitive (L1).
            # Behaviour matches the previous inline loop: tool calls execute
            # until the model answers without one (or the budget is hit).
            # Message formatting is now unified (incl. ToolResult.media).
            from syll.agent.loop_core import run_tool_loop

            _loop = await run_tool_loop(
                self.provider,
                tools,
                messages,
                model=self.model,
                max_iterations=15,
            )
            # Extract content regardless of stop_reason — budget-exhaustion should
            # not discard the model's last response.  Matches loop.py _finalize_turn.
            last_content = (
                (_loop.final_response.content or "")
                if _loop.final_response
                else ""
            )
            if not last_content and _loop.final_response:
                rc = _loop.final_response.provider_extra.get("reasoning_content")
                if rc:
                    last_content = rc[:2000]
            if not last_content:
                last_content = "Task completed but no final response was generated."
            final_result = last_content

            logger.info(f"Subagent [{task_id}] completed successfully")
            await self._announce_result(task_id, label, task, final_result, origin, "ok")

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Subagent [{task_id}] failed: {e}")
            await self._announce_result(task_id, label, task, error_msg, origin, "error")

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus.

        Failure tracking: consecutive failures for the same label are counted.
        After ``_max_announce_retries`` failures, subsequent announcements are
        suppressed to prevent noise in the main conversation.
        """
        if status == "ok":
            self._failure_counts.pop(label, None)
            announce_content = f"✅ 后台任务「{label}」已完成：{result}"
        else:
            failures = self._failure_counts.get(label, 0) + 1
            self._failure_counts[label] = failures
            if failures > self._max_announce_retries:
                logger.warning(
                    f"Subagent [{task_id}] label '{label}' failed {failures} times, "
                    "suppressing further announcements"
                )
                return
            announce_content = f"❌ 后台任务「{label}」执行失败：{result}"

        # Inject as system message — main agent handles it without full loop
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.debug(f"Subagent [{task_id}] announced result to {origin['channel']}:{origin['chat_id']}")

    def _build_subagent_prompt(self, task: str) -> str:
        """Build a focused system prompt for the subagent."""
        return f"""# Subagent

You are a subagent spawned by the main agent to complete a specific task.

## Your Task
{task}

## Rules
1. Stay focused - complete only the assigned task, nothing else
2. Your final response will be reported back to the main agent
3. Do not initiate conversations or take on side tasks
4. Be concise but informative in your findings

## What You Can Do
- Read and write files in the workspace
- Execute shell commands
- Search the web and fetch web pages
- Complete the task thoroughly

## What You Cannot Do
- Send messages directly to users (no message tool available)
- Spawn other subagents
- Access the main agent's conversation history

## Workspace
Your workspace is at: {self.workspace}

When you have completed the task, provide a clear summary of your findings or actions."""

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
