"""Tests for syll.agent.context_compactor."""

from __future__ import annotations

import pytest

from syll.agent.context_compactor import ContextCompactor


def _make_messages(n_tools: int = 5) -> list[dict]:
    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Do something with the GUI."},
    ]
    for i in range(n_tools):
        messages.append({
            "role": "assistant",
            "content": f"Step {i} plan",
            "tool_calls": [{"id": f"tc{i}", "function": {"name": "gui_action"}}],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": f"tc{i}",
            "name": "gui_action",
            "content": "x" * 4000,
        })
    return messages


@pytest.mark.asyncio
async def test_compact_returns_original_when_under_budget():
    compactor = ContextCompactor()
    messages = _make_messages(n_tools=2)
    budget = 1_000_000
    result = await compactor.compact(messages, budget)
    assert result == messages


@pytest.mark.asyncio
async def test_micro_compact_replaces_old_tool_results():
    compactor = ContextCompactor(micro_compact_keep_recent=1)
    messages = _make_messages(n_tools=3)
    budget = 6_000  # effective budget < original but > micro-compacted
    result = await compactor.compact(messages, budget)

    tool_msgs = [m for m in result if m.get("role") == "tool"]
    assert len(tool_msgs) == 3
    # Most recent tool result kept verbatim.
    assert tool_msgs[-1]["content"] == "x" * 4000
    # Older ones replaced with placeholder.
    assert "omitted" in tool_msgs[0]["content"]
    assert "omitted" in tool_msgs[1]["content"]


@pytest.mark.asyncio
async def test_truncate_drops_oldest_non_system_messages():
    compactor = ContextCompactor(micro_compact_keep_recent=0)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "last"},
    ]
    # Very small budget forces truncation.
    result = await compactor.compact(messages, budget_tokens=10)
    assert result[0]["role"] == "system"
    assert result[-1]["content"] == "last"
    # At least one intermediate message was dropped.
    assert len(result) < len(messages)


@pytest.mark.asyncio
async def test_llm_summarize_uses_provider(monkeypatch):
    class FakeProvider:
        def __init__(self):
            self.called = False

        async def chat(self, **kwargs):
            self.called = True
            class Resp:
                finish_reason = "stop"
                content = "Summary of older turns."
            return Resp()

    provider = FakeProvider()
    compactor = ContextCompactor(
        provider=provider, model="fake", micro_compact_keep_recent=6
    )
    messages = _make_messages(n_tools=6)
    result = await compactor.compact(messages, budget_tokens=6_000)

    assert provider.called
    assert any("Summary of older turns" in str(m.get("content", "")) for m in result)


@pytest.mark.asyncio
async def test_compact_unknown_budget_returns_original():
    compactor = ContextCompactor()
    messages = _make_messages()
    assert await compactor.compact(messages, 0) == messages
