"""Agent loop: the core processing engine."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from syll.agent.context import ContextBuilder
from syll.agent.events import Event, EventContent, EventSource, EventStore
from syll.agent.memory import GlobalMemoryStore, MemoryStore, migrate_workspace_memory_to_global
from syll.agent.context_compactor import ContextCompactor
from syll.agent.memory_flush import MemoryFlusher
from syll.agent.result import AgentResult
from syll.agent.subagent import SubagentManager
from syll.sandbox.environment import LocalEnvironment

if TYPE_CHECKING:
    from syll.agent.mcp import MCPManager
from syll.agent.tools.attach_file import AttachFileTool
from syll.agent.tools.bundles import register_core_tools
from syll.agent.tools.cron import CronTool
from syll.agent.tools.file_preview import FilePreviewTool
from syll.agent.tools.find_file import FindFileTool
from syll.agent.tools.message import MessageTool
from syll.agent.tools.registry import ToolRegistry
from syll.agent.tools.screenshot import ScreenshotTool
from syll.agent.tools.spawn import SpawnTool
from syll.agent.tools.video_learn import VideoLearnTool
from syll.bus.events import InboundMessage, OutboundMessage
from syll.bus.queue import MessageBus
from syll.providers.base import LLMProvider
from syll.session.manager import SessionManager


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        gui_config: "GuiConfig | None" = None,
        syll_config: "Config | None" = None,
        mcp_manager: "MCPManager | None" = None,
        global_memory_store: "MemoryStore | None" = None,
    ):
        from syll.config.schema import ExecToolConfig
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.gui_config = gui_config
        self.syll_config = syll_config
        self.mcp_manager = mcp_manager

        identity = syll_config.identity if syll_config else None
        global_memory = global_memory_store or GlobalMemoryStore()
        self.context = ContextBuilder(
            workspace, global_memory=global_memory, identity=identity
        )
        self.memory_flusher = MemoryFlusher(self.provider, self.context.memory, model=self.model)
        self.context_compactor = ContextCompactor(self.provider, self.model)

        # One-time migration of existing workspace MEMORY.md to the new global
        # user memory store. Safe to call on every startup: it only runs when
        # global MEMORY.md is missing and workspace memory has real content.
        try:
            migrate_workspace_memory_to_global(workspace)
        except Exception:
            pass
        self.sessions = SessionManager(workspace)
        self.environment = LocalEnvironment(workspace_root=workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            mcp_manager=mcp_manager,
            environment=self.environment,
        )
        self.event_store = EventStore(workspace.parent)

        # Phase 1c: track which tool names are owned by the MCP manager so
        # `reload_mcp_tools` can unregister exactly those (no prefix-strip
        # — see syll/agent/mcp.py docstring) without clobbering non-MCP
        # tools whose names happen to start with "mcp__".
        self._mcp_owned: set[str] = set()

        # Per-session ContextMeter cache. Keyed by session_key so the token
        # curve accumulates ACROSS turns (history grows across turns too —
        # that is part of what fills the context). One meter appends to one
        # {workspace}/audit/{session}/context_curve.jsonl.
        self._ctx_meters: dict = {}

        self._running = False
        self._register_default_tools()

    def get_context_meter(self, session_key: str):
        """Get or create the per-session ContextMeter.

        Shared by ``_process_message`` (non-streaming chat / process_direct)
        AND the WebSocket streaming path (``web/streaming.py``) so both append
        to ONE curve file per session. Returns None if creation fails.
        """
        meter = self._ctx_meters.get(session_key)
        if meter is not None:
            return meter
        try:
            from syll.agent.longhorizon.context_meter import (
                ContextMeter,
                resolve_context_window,
            )
            _cw = 0
            if self.syll_config:
                try:
                    _cw = self.syll_config.models.chat.context_window
                except Exception:
                    _cw = 0
            meter = ContextMeter(
                run_dir=self.workspace / "audit" / session_key.replace(":", "_"),
                run_id=session_key,
                budget_tokens=resolve_context_window(self.model, _cw),
            )
            self._ctx_meters[session_key] = meter
        except Exception as _e:
            logger.debug(f"context meter disabled for {session_key}: {_e}")
            meter = None
        return meter

    def reload_mcp_tools(self) -> int:
        """(Re)register MCP tools on `self.tools` from the manager.

        Idempotent. Unregisters exactly the names this loop previously
        owned (`self._mcp_owned`) — never prefix-strips on `mcp__*` because
        a non-MCP tool could be named that way too. Refuses to clobber a
        non-MCP tool whose name collides with an MCP adapter's; in that
        case the MCP tool is skipped with a warning.

        Call this:
          - At gateway boot, AFTER `MCPManager.start()` has connected.
          - On every `apply_server`/`remove_server` from the HTTP route
            (so the live tool set tracks config changes).

        Returns the number of MCP tools registered after the reload.
        """
        # Step 1: drop the old set.
        for owned_name in list(self._mcp_owned):
            self.tools.unregister(owned_name)
        self._mcp_owned.clear()

        if self.mcp_manager is None:
            return 0

        # Step 2: register every adapter from the manager.
        registered = 0
        for adapter in self.mcp_manager.iter_enabled_tools():
            n = adapter.name
            if self.tools.has(n):
                logger.warning(
                    f"reload_mcp_tools: name {n!r} collides with an existing "
                    "non-MCP tool; skipping the MCP adapter to avoid clobber"
                )
                continue
            self.tools.register(adapter)
            self._mcp_owned.add(n)
            registered += 1
        logger.info(f"reload_mcp_tools: {registered} MCP tool(s) registered")
        return registered

    def _register_default_tools(self) -> None:
        """Register the default tool set, grouped by category."""
        # File + shell + web tools (shared core). The loop additionally exposes
        # EditFileTool; subagents get read/write/list only.
        register_core_tools(
            self.tools,
            workspace=self.workspace,
            restrict_to_workspace=self.restrict_to_workspace,
            exec_config=self.exec_config,
            brave_api_key=self.brave_api_key,
            include_edit=True,
            environment=self.environment,
        )
        self._register_interactive_tools()
        self._register_video_tool()
        self._register_voice_tool()
        self._register_gui_tools()

    def _register_interactive_tools(self) -> None:
        """Message / spawn / cron / screenshot / file-discovery tools."""
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
        self.tools.register(ScreenshotTool(environment=self.environment))
        self.tools.register(FindFileTool())
        self.tools.register(FilePreviewTool())
        self.tools.register(AttachFileTool())

    def _register_video_tool(self) -> None:
        """Video-learning tool — registered only when yt-dlp is on PATH."""
        try:
            import shutil as _shutil
            if _shutil.which("yt-dlp"):
                vl_api_key = ""
                vl_api_base = ""
                if self.syll_config:
                    models_cfg = getattr(self.syll_config, "models", None)
                    if models_cfg:
                        chat_cfg = getattr(models_cfg, "chat", None)
                        if chat_cfg:
                            vl_api_key = getattr(chat_cfg, "api_key", "") or ""
                            vl_api_base = getattr(chat_cfg, "api_base", "") or ""
                self.tools.register(VideoLearnTool(
                    model=self.model,
                    api_key=vl_api_key,
                    api_base=vl_api_base,
                    workspace=self.workspace,
                    brave_api_key=self.brave_api_key or "",
                ))
                logger.info("VideoLearnTool registered (yt-dlp found)")
        except Exception as e:
            logger.debug(f"VideoLearnTool registration skipped: {e}")

    def _register_voice_tool(self) -> None:
        """Speak tool — only when TTS credentials are configured.

        Without this guard the tool would surface in the LLM prompt even when
        ``speak`` would immediately fail, wasting a turn."""
        if not (
            self.syll_config
            and getattr(self.syll_config, "voice", None)
            and self.syll_config.voice.enabled
            and self.syll_config.voice.tts.appid
            and self.syll_config.voice.tts.access_token
        ):
            return
        try:
            from syll.agent.tools.speak import SpeakTool
            from syll.providers.voice_volc import VolcengineTTSProvider
            tts_cfg = self.syll_config.voice.tts
            # Default resource_id is used ONLY as the fallback for
            # unknown voices; the builtin voice→resource_id catalog
            # covers both BigTTS 2.0 and Seed-TTS 2.0. We align the
            # fallback with the configured default_speaker so the
            # "empty resource_id + common voice" combination Just Works.
            tts = VolcengineTTSProvider(
                appid=tts_cfg.appid,
                access_token=tts_cfg.access_token,
                default_speaker=self.syll_config.voice.default_speaker,
                resource_id=tts_cfg.resource_id or "seed-tts-2.0",
                voice_resources=getattr(tts_cfg, "voice_resources", None) or None,
            )
            self.tools.register(SpeakTool(tts))
            logger.info("SpeakTool registered (voice.tts.provider=volcengine)")
        except Exception as e:
            logger.warning(f"SpeakTool registration failed: {e}")

    def _register_gui_tools(self) -> None:
        """Register the GUI tools, gated on gui_config.enabled.

        ``gui_action`` runs the Enhanced TVAE-verifier pipeline
        (``GuiActionTool``) so ad-hoc tasks get per-step verification; it falls
        back to ``UITarsTool`` only if the enhanced module is unavailable.
        ``gui_action_planned`` is the explicit trajectory-guided variant
        (skill_name required)."""
        if not (self.gui_config and self.gui_config.enabled):
            return
        from syll.agent.aloha_gui_skill import AlohaSkillStore

        aloha_skill_store = AlohaSkillStore(self.workspace)

        # gui_action → Enhanced verifier pipeline (skill optional). Per the
        # user directive, ALL gui_action calls go through gui_action_planned's
        # TVAE verifier, not the verifier-less UITarsTool.
        try:
            from syll.agent.aloha.act.enhanced.enhanced_planner_tool import (
                GuiActionTool,
            )

            gui_action_tool = GuiActionTool(
                self.gui_config, aloha_skill_store, syll_config=self.syll_config,
                environment=self.environment,
            )
            logger.info("Using GuiActionTool (Enhanced TVAE verifier) for gui_action")
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
            logger.warning("Enhanced unavailable; falling back to UITarsTool for gui_action")

        gui_action_tool._event_store = self.event_store
        self.tools.register(gui_action_tool)

        # gui_action_planned: explicit trajectory-guided variant (skill required)
        try:
            from syll.agent.aloha.act.enhanced.enhanced_planner_tool import (
                EnhancedAlohaPlannerTool,
            )

            planner_tool = EnhancedAlohaPlannerTool(
                self.gui_config,
                aloha_skill_store,
                syll_config=self.syll_config,
                environment=self.environment,
            )
            logger.info("Using EnhancedAlohaPlannerTool (Phase 1-4 enabled)")
        except ImportError:
            from syll.agent.tools.aloha_planner_tool import AlohaPlannerTool

            planner_tool = AlohaPlannerTool(
                self.gui_config,
                aloha_skill_store,
                syll_config=self.syll_config,
                environment=self.environment,
            )
            logger.info("Using original AlohaPlannerTool (enhanced unavailable)")

        planner_tool._event_store = self.event_store
        self.tools.register(planner_tool)

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")

        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )

                # Process it
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        *,
        prompt_content: str | None = None,
        language_hint_text: str | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message through the tool-calling loop.

        Returns the response message, or None if no response is needed.
        """
        if msg.channel == "system":
            return await self._process_system_message(msg)

        # /retry-gui: clear this session's GUI failure locks so the model can
        # retry gui_action after a (possibly transient) failure was recorded.
        if (msg.content or "").strip().lower() == "/retry-gui":
            from syll.agent.gui_failure_ledger import GuiAttemptLedger
            cleared = GuiAttemptLedger(msg.session_key).clear_all()
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=(
                    f"已清除本会话的 {cleared} 个 GUI 失败锁，可以重试 gui_action。"
                ),
            )

        routed = await self._maybe_route_gui_v3(msg, prompt_content)
        if routed is not None:
            return routed

        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}")
        session = self.sessions.get_or_create(msg.session_key)
        self._wire_tool_contexts(msg)

        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=prompt_content or msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
            language_hint_text=language_hint_text,
        )

        # Context-length detector for this session — shared with the streaming
        # path (web/streaming.py) via get_context_meter so both append to one
        # curve file.
        meter = self.get_context_meter(msg.session_key)

        # Compact context before sending it to the model. This applies a
        # three-tier strategy (micro-compaction -> LLM summary -> truncation)
        # when the estimated prompt tokens approach the configured context
        # window. Unknown budgets (0) are ignored.
        budget_tokens = getattr(meter, "budget_tokens", 0) or 0
        if budget_tokens:
            try:
                messages = await self.context_compactor.compact(messages, budget_tokens)
            except Exception as exc:
                logger.warning(f"Context compaction failed, using original context: {exc}")

        # Agent loop — shared primitive (L1). Usage capture + the GUI call
        # limiter are injected as callbacks; behaviour is identical to the
        # previous inline loop (and shares formatting with the subagent paths,
        # including ToolResult.media).
        from syll.agent.loop_core import run_tool_loop

        loop_result = await run_tool_loop(
            self.provider,
            self.tools,
            messages,
            model=self.model,
            max_iterations=self.max_iterations,
            max_tokens=self._resolve_max_tokens(),
            on_usage=self._make_usage_callback(meter),
            pre_execute=self._make_gui_limiter(),
        )
        return await self._finalize_turn(msg, session, loop_result)

    async def _maybe_route_gui_v3(
        self, msg: InboundMessage, prompt_content: str | None
    ) -> OutboundMessage | None:
        """Auto-route GUI/desktop tasks to the v3 (longhorizon) pipeline.

        Returns an OutboundMessage when the task was routed (fold + subagent +
        verification gate + checkpoint/replan instead of a single-shot GUI
        action), else None so the caller falls through to the normal flow.
        Gated on gui_config.enabled; any failure falls through so the ghost
        never breaks here."""
        if not (self.gui_config and getattr(self.gui_config, "enabled", False)):
            return None
        try:
            from syll.agent.longhorizon.router import is_gui_task, run_gui_via_v3

            task_text = (prompt_content or msg.content or "").strip()
            if task_text and await is_gui_task(self.provider, task_text, model=self.model):
                logger.info("Auto-routing GUI task -> v3 (longhorizon) pipeline")
                v3_workspace = self.workspace / "longhorizon_runs" / msg.session_key.replace(":", "_")
                v3_summary = await run_gui_via_v3(
                    task_text,
                    workspace=v3_workspace,
                    provider=self.provider,
                    model=self.model,
                )
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"[via v3 pipeline]\n{v3_summary}",
                )
        except Exception as e:
            logger.warning(f"v3 GUI routing failed, falling back to normal flow: {e}")
        return None

    def _wire_tool_contexts(self, msg: InboundMessage) -> None:
        """Give channel-aware tools the current channel/chat_id for replies."""
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(msg.channel, msg.chat_id)
        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(msg.channel, msg.chat_id)
        # UITarsTool (gui_action): attach the per-session GUI failure ledger so
        # genuine failures become revisable state instead of un-addressable
        # chat text. Duck-typed so the loop has no hard GUI-tool import.
        ui_tars_tool = self.tools.get("gui_action")
        if ui_tars_tool is not None and hasattr(ui_tars_tool, "set_session_context"):
            ui_tars_tool.set_session_context(msg.session_key)

    def _make_usage_callback(self, meter):
        """on_usage for run_tool_loop: record each model call's tokens to the
        session context meter (no-op when no meter is attached)."""
        def on_usage(resp, it):
            if meter is None:
                return
            try:
                usage = resp.usage or {}
                meter.record(
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    phase="orchestrator",
                    extra={
                        "iteration": it,
                        "had_tool_calls": resp.has_tool_calls,
                    },
                )
            except Exception as e:
                logger.debug(f"meter record skipped: {e}")
        return on_usage

    def _make_gui_limiter(self):
        """pre_execute for run_tool_loop: gate gui_action calls on the per-task
        failure ledger (cross-turn persistent lock, clearable via /retry-gui),
        then cap consecutive GUI calls within this turn. Returns a
        stop-instruction string when blocking, else None."""
        gui_consecutive = 0
        GUI_MAX_CONSECUTIVE = 2

        def pre_execute(tool_call):
            nonlocal gui_consecutive
            if tool_call.name in ("gui_action", "gui_action_planned"):
                # Ledger lock: a genuine GUI failure recorded for THIS task
                # blocks the call entirely (independent of the per-turn counter).
                ui_tars = self.tools.get("gui_action")
                ledger = getattr(ui_tars, "_gui_ledger", None) if ui_tars is not None else None
                if ledger is not None and tool_call.name == "gui_action":
                    instr = (tool_call.arguments or {}).get("instruction", "")
                    if instr and ledger.is_locked(instr):
                        status = ledger.lock_status(instr) or {}
                        logger.info(
                            f"GUI ledger lock active ({status.get('kind')}); "
                            f"blocking gui_action retry"
                        )
                        return (
                            f"此 GUI 任务此前因「{status.get('kind', '未知')}」失败并被锁定，"
                            "本次不执行。请告知用户失败原因，或建议其发送 /retry-gui "
                            "清除锁后重试。"
                        )
                gui_consecutive += 1
                # gui_action is single-step (outer agent drives iteration) — allow
                # many calls per turn for legitimate multi-step tasks. gui_action_planned
                # is autonomous multi-step — keep the tight anti-spam cap.
                cap = 15 if tool_call.name == "gui_action" else GUI_MAX_CONSECUTIVE
                if gui_consecutive > cap:
                    logger.warning(
                        f"GUI call limit reached ({gui_consecutive}/{cap} for "
                        f"{tool_call.name}), forcing stop to prevent infinite retries"
                    )
                    return (
                        "GUI 操作已达到最大调用次数。请直接告知用户操作失败，"
                        "不要再次调用 GUI 工具。建议用户手动操作或调整指令后重试。"
                    )
            else:
                gui_consecutive = 0
            return None
        return pre_execute

    def _resolve_max_tokens(self) -> int:
        """Resolve max_tokens for the main loop.

        Generous default (16384) so extended-thinking models have room for
        both thinking tokens and visible content.  Config can override via
        ``agents.defaults.max_tokens``.
        """
        if self.syll_config:
            try:
                return self.syll_config.agents.defaults.max_tokens
            except (AttributeError, TypeError):
                pass
        return 16384

    async def _finalize_turn(self, msg: InboundMessage, session, loop_result) -> OutboundMessage:
        """Persist the turn (session history + event + daily memory) and build
        the reply OutboundMessage."""
        collected_media = loop_result.media
        # Extract content regardless of stop_reason — budget-exhaustion should
        # not discard the model's last response.  Matches unified_subagent.py:357.
        last_content = (
            (loop_result.final_response.content or "")
            if loop_result.final_response
            else ""
        )
        # Also check reasoning_content (extended thinking models may put all
        # output in reasoning and leave content empty).
        if not last_content and loop_result.final_response:
            rc = loop_result.final_response.provider_extra.get("reasoning_content")
            if rc:
                last_content = rc[:2000]
        if not last_content:
            if loop_result.stop_reason == "budget":
                last_content = "⚠️ Agent reached iteration limit without a final answer."
            else:
                last_content = "⚠️ Agent completed but produced no text response."
        final_content = last_content

        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        event = Event(
            agent_type="im_agent",
            event_type="message",
            source=EventSource(
                platform=msg.channel,
                chat_id=msg.chat_id,
                user_id=msg.sender_id,
            ),
            content=EventContent(
                text=f"User: {msg.content}\nAssistant: {final_content}",
                media=collected_media,
                metadata={
                    "iterations": loop_result.iterations,
                    "session_key": msg.session_key,
                },
            ),
        )
        self.event_store.log_event(event)

        try:
            from datetime import datetime
            summary = f"- [{datetime.now().strftime('%H:%M')}] User: {msg.content[:100]}\n"
            self.context.memory.append_today(summary)
        except Exception as e:
            logger.debug(f"Failed to append daily memory: {e}")

        # Memory flush turn: promote reusable facts to long-term memory.
        try:
            await self.memory_flusher.flush(msg.content, final_content)
        except Exception as e:
            logger.debug(f"Memory flush turn failed: {e}")

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            media=collected_media,
        )

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).

        Unlike normal messages, system messages are handled directly without
        running a full agent loop. This prevents subagent failures from
        triggering retries or interfering with the user's active conversation.

        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")

        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id

        # Use the origin session for context
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)

        # Extract a concise summary directly from the announce content
        # instead of running a full agent loop (avoids retry cascades).
        content = msg.content.strip()

        # The announce format is: [Subagent 'label' status]
        # Task: ...
        # Result: ...
        # We just forward a clean summary to the user.
        summary = content
        # Strip the "Summarize this naturally..." instruction that's meant for LLM
        summary_line = summary.split("Summarize this naturally")[0].strip()
        if summary_line:
            summary = summary_line

        # Save to session history (mark as system message)
        session.add_message("user", f"[System: {msg.sender_id}] {content}")
        session.add_message("assistant", summary)
        self.sessions.save(session)

        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=summary,
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        *,
        language_hint_text: str | None = None,
        inject_skill_hints: bool = False,
    ) -> AgentResult:
        """
        Process a message directly (for CLI / REST / cron / ritual usage).

        Returns an ``AgentResult`` carrying both the final text and any
        media produced during the turn (e.g. ``speak`` TTS output).
        """
        prompt_content = content
        if inject_skill_hints:
            from syll.web.skill_router import inject_skill_hint

            prompt_content = inject_skill_hint(self, content)

        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )

        response = await self._process_message(
            msg,
            prompt_content=prompt_content,
            language_hint_text=language_hint_text or content,
        )
        if response is None:
            return AgentResult(text="")
        return AgentResult(
            text=response.content or "",
            media=list(response.media or []),
        )
