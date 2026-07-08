"""Tests for MemoryFlusher."""

from pathlib import Path

import pytest

from syll.agent.memory import MemoryStore
from syll.agent.memory_flush import MemoryFlusher
from syll.providers.base import LLMResponse


class _FakeProvider:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        self.calls.append({"messages": messages, "model": model})
        text = self._responses.pop(0) if self._responses else "NONE"
        return LLMResponse(content=text)

    def get_default_model(self):
        return "stub"


@pytest.mark.anyio
async def test_flush_writes_extracted_facts(tmp_path: Path):
    store = MemoryStore(tmp_path, scope="global")
    provider = _FakeProvider(["User prefers Emacs.\nLikes dark mode."])
    flusher = MemoryFlusher(provider, store, model="stub")

    written = await flusher.flush("I use Emacs and dark mode", "ok")

    assert len(written) == 2
    assert "Emacs" in store.read_long_term()
    assert "dark mode" in store.read_long_term()


@pytest.mark.anyio
async def test_flush_skips_none_response(tmp_path: Path):
    store = MemoryStore(tmp_path, scope="global")
    provider = _FakeProvider(["NONE"])
    flusher = MemoryFlusher(provider, store, model="stub")

    written = await flusher.flush("hello", "hi")
    assert written == []
    assert store.read_long_term() == ""


@pytest.mark.anyio
async def test_flush_dedups_against_existing_memory(tmp_path: Path):
    store = MemoryStore(tmp_path, scope="global")
    store.write_long_term("# Long-term Memory\n\n- [2026-01-01] User prefers Emacs.\n")
    provider = _FakeProvider(["User prefers Emacs.\nUser likes tea."])
    flusher = MemoryFlusher(provider, store, model="stub")

    written = await flusher.flush("I still use Emacs and I like tea", "ok")

    assert len(written) == 1
    assert "tea" in store.read_long_term()
    assert store.read_long_term().count("Emacs") == 1


@pytest.mark.anyio
async def test_flush_drops_short_facts(tmp_path: Path):
    store = MemoryStore(tmp_path, scope="global")
    provider = _FakeProvider(["ok\nhi"])
    flusher = MemoryFlusher(provider, store, model="stub")

    written = await flusher.flush("hello", "hi")
    assert written == []


@pytest.mark.anyio
async def test_flush_ignores_empty_turn(tmp_path: Path):
    store = MemoryStore(tmp_path, scope="global")
    provider = _FakeProvider([])
    flusher = MemoryFlusher(provider, store, model="stub")

    written = await flusher.flush("", "")
    assert written == []
