"""Anthropic-Messages provider (works with Anthropic-compatible endpoints
like Zhipu/GLM's ``/api/anthropic``).

Why this exists alongside ``LiteLLMProvider``:
- The GLM endpoint authenticates via ``ANTHROPIC_AUTH_TOKEN`` ->
  ``Authorization: Bearer`` (this is how the Claude Code harness itself runs
  on GLM). LiteLLM's anthropic provider sends ``x-api-key`` instead, which
  Zhipu's gateway rejects (401). The official ``anthropic`` SDK with
  ``auth_token=`` is the path that actually works.
- The subagent loop builds messages in **OpenAI format** (assistant
  ``tool_calls`` + ``role:tool`` results). The Anthropic Messages API uses a
  different block format (``tool_use`` / ``tool_result``). This provider
  converts both ways so the rest of the runner stays unchanged.

Implements the syll ``LLMProvider`` interface.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from syll.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class AnthropicMessagesProvider(LLMProvider):
    """LLMProvider backed by the Anthropic SDK (Bearer auth), with OpenAI<->Anthropic conversion."""

    def __init__(
        self,
        auth_token: str,
        base_url: str,
        default_model: str,
        api_key: str | None = None,
    ):
        super().__init__(api_key=api_key, api_base=base_url)
        self.auth_token = auth_token
        self.base_url = base_url
        self.default_model = default_model
        self._client = anthropic.AsyncAnthropic(auth_token=auth_token, base_url=base_url)

    def get_default_model(self) -> str:
        return self.default_model

    # ------------------------------------------------------------------
    # conversion: OpenAI messages/tools -> Anthropic
    # ------------------------------------------------------------------
    @staticmethod
    def _convert_messages(messages: list[dict]) -> tuple[str, list[dict]]:
        """Return (system_text, anthropic_messages).

        Handles OpenAI assistant ``tool_calls`` -> Anthropic ``tool_use``
        blocks, and OpenAI ``role:tool`` -> Anthropic ``tool_result`` blocks
        (grouped into user turns as Anthropic requires)."""
        system_parts: list[str] = []
        conv: list[dict] = []

        def new_user_with(blocks):
            conv.append({"role": "user", "content": blocks})

        for m in messages:
            role = m.get("role")
            if role == "system":
                if m.get("content"):
                    system_parts.append(str(m["content"]))
                continue

            if role == "tool":
                # OpenAI tool result -> Anthropic tool_result block in a user turn.
                block = {
                    "type": "tool_result",
                    "tool_use_id": str(m.get("tool_call_id") or ""),
                    "content": str(m.get("content") or ""),
                }
                # Merge into the previous user turn if it is a tool-result turn.
                if conv and conv[-1]["role"] == "user" and isinstance(
                    conv[-1]["content"], list
                ) and conv[-1]["content"] and all(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in conv[-1]["content"]
                ):
                    conv[-1]["content"].append(block)
                else:
                    new_user_with([block])
                continue

            if role == "assistant":
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": str(m["content"])})
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": str(tc.get("id") or ""),
                            "name": fn.get("name", ""),
                            "input": args,
                        }
                    )
                conv.append(
                    {"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]}
                )
                continue

            if role == "user":
                conv.append(
                    {"role": "user", "content": [{"type": "text", "text": str(m.get("content") or "")}]}
                )

        return "\n\n".join(system_parts), conv

    @staticmethod
    def _convert_tools(tools: list[dict] | None) -> list[dict] | None:
        if not tools:
            return None
        out = []
        for t in tools:
            fn = t.get("function", t) if isinstance(t, dict) else {}
            out.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
                }
            )
        return out

    # ------------------------------------------------------------------
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        model = model or self.default_model
        system_text, conv = self._convert_messages(messages)
        anth_tools = self._convert_tools(tools)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": conv,
            "temperature": temperature,
        }
        if system_text:
            kwargs["system"] = system_text
        if anth_tools:
            kwargs["tools"] = anth_tools

        try:
            resp = await self._client.messages.create(**kwargs)
        except Exception as exc:
            return LLMResponse(content=f"Error calling LLM: {exc}", finish_reason="error")

        content_text = ""
        tool_calls: list[ToolCallRequest] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                content_text += getattr(block, "text", "") or ""
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCallRequest(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=dict(getattr(block, "input", {}) or {}),
                    )
                )

        usage = {}
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": getattr(resp.usage, "input_tokens", 0),
                "completion_tokens": getattr(resp.usage, "output_tokens", 0),
            }

        return LLMResponse(
            # Preserve empty string — downstream callers now handle it correctly.
            # Previously `or None` converted "" → None, masking the fact that
            # the model returned no text blocks (only thinking / tool_use).
            content=content_text if content_text is not None else None,
            tool_calls=tool_calls,
            finish_reason=getattr(resp, "stop_reason", "stop") or "stop",
            usage=usage,
        )
