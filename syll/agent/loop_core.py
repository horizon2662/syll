"""Shared tool-calling loop primitive.

The same ``provider.chat → parse tool_calls → execute → append`` skeleton was
duplicated in three places (``AgentLoop._process_message``,
``SubagentManager._run_subagent``, ``UnifiedSubagentManager._run_isolated``).
This module factors it out so loop behaviour (usage capture, message
formatting) lives in one place. Differences between callers are injected via
optional callbacks.

Message formatting (``build_assistant_msg`` / ``build_tool_result_msg``) is
unified and mirrors ``ContextBuilder`` exactly — including multimodal
``ToolResult.media`` and ``reasoning_content`` — so the subagent paths no
longer silently drop them (this fixes the L2 ``UnifiedSubagentManager`` media
bug noted in the GUI↔loop investigation).
"""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from syll.agent.tools.base import ToolResult
from syll.providers.base import LLMResponse


@dataclass
class LoopResult:
    """Outcome of :func:`run_tool_loop`.

    Attributes:
        final_response: The last LLM response seen (None only if ``pre_iteration``
            broke before any model call).
        messages: The mutated message list (assistant + tool messages appended).
        iterations: Number of iterations actually executed.
        stop_reason: Why the loop ended — ``"no_tool_calls"`` (model gave a
            final answer), ``"pre_iteration"`` (a caller hook requested stop),
            or ``"budget"`` (iteration limit reached).
        media: Tool-result media paths collected along the way.
    """

    final_response: LLMResponse | None
    messages: list[dict[str, Any]]
    iterations: int
    stop_reason: str
    media: list[str] = field(default_factory=list)


def build_assistant_msg(
    content: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
    *,
    reasoning_content: str | None = None,
) -> dict[str, Any]:
    """Construct an assistant message dict (mirrors ContextBuilder)."""
    msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if reasoning_content:
        msg["reasoning_content"] = reasoning_content
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def build_tool_result_msg(
    tool_call_id: str,
    tool_name: str,
    result: str | ToolResult,
) -> dict[str, Any]:
    """Construct a tool-result message dict (mirrors ContextBuilder, incl. media)."""
    if isinstance(result, ToolResult) and result.media:
        content_parts: list[dict[str, Any]] = []
        for path in result.media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
        content_parts.append({"type": "text", "text": result.text})
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": content_parts,
        }
    text = result.text if isinstance(result, ToolResult) else result
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": text,
    }


async def run_tool_loop(
    provider: Any,
    tools: Any,
    messages: list[dict[str, Any]],
    *,
    model: str | None,
    max_iterations: int,
    max_tokens: int = 16384,
    on_usage: Callable[[LLMResponse, int], None] | None = None,
    should_stop: Callable[[LLMResponse], bool] | None = None,
    pre_iteration: Callable[[int], bool] | None = None,
    on_iteration_start: Callable[[int], None] | None = None,
    pre_execute: Callable[[Any], str | None] | None = None,
) -> LoopResult:
    """Run a tool-calling loop and return the outcome.

    Args:
        provider: LLMProvider with ``chat(messages, tools, model)``.
        tools: ToolRegistry with ``get_definitions()`` / ``execute(name, args)``.
        messages: Message list (mutated in place).
        model: Model id passed to ``provider.chat``.
        max_iterations: Iteration cap.
        max_tokens: Maximum output tokens per model call (default 16384 —
            generous for extended-thinking models).  Passed through to
            ``provider.chat()``.
        on_usage: Called with each ``(response, iteration)`` (e.g. ContextMeter).
        should_stop: Returns True to stop after a response. Defaults to
            "no tool calls" (the normal ReAct termination).
        pre_iteration: Called before each increment with the current count;
            returning True breaks the loop (used by the fold subagent's
            ``return``-tool check).
        on_iteration_start: Called after incrementing, before the model call
            (e.g. blackboard progress write).
        pre_execute: Called before each tool execution; if it returns a string,
            that string is appended as the tool result instead of executing
            (used by AgentLoop's GUI call limiter).
    """
    iteration = 0
    final_response: LLMResponse | None = None
    stop_reason = "budget"
    collected_media: list[str] = []
    _stop = should_stop or (lambda r: not r.has_tool_calls)

    while iteration < max_iterations:
        if pre_iteration is not None and pre_iteration(iteration):
            stop_reason = "pre_iteration"
            break
        iteration += 1
        if on_iteration_start is not None:
            on_iteration_start(iteration)

        response = await provider.chat(
            messages=messages,
            tools=tools.get_definitions(),
            model=model,
            max_tokens=max_tokens,
        )
        final_response = response

        if on_usage is not None:
            try:
                on_usage(response, iteration)
            except Exception:
                pass

        if _stop(response):
            stop_reason = "no_tool_calls"
            break

        # Has tool calls — record the assistant message, then execute each.
        tool_call_dicts = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in response.tool_calls
        ]
        messages.append(
            build_assistant_msg(
                response.content,
                tool_call_dicts,
                reasoning_content=response.provider_extra.get("reasoning_content"),
            )
        )

        for tool_call in response.tool_calls:
            if pre_execute is not None:
                injected = pre_execute(tool_call)
                if injected is not None:
                    messages.append(
                        build_tool_result_msg(tool_call.id, tool_call.name, injected)
                    )
                    continue
            result = await tools.execute(tool_call.name, tool_call.arguments)
            if isinstance(result, ToolResult) and result.media:
                collected_media.extend(result.media)
            messages.append(
                build_tool_result_msg(tool_call.id, tool_call.name, result)
            )

    return LoopResult(final_response, messages, iteration, stop_reason, collected_media)
