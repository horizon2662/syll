"""Tests for MemoryStore and GlobalMemoryStore."""

from pathlib import Path

import pytest

from syll.agent.memory import GlobalMemoryStore, MemoryStore
from syll.utils.helpers import get_global_memory_path


def test_memory_store_uses_base_dir(tmp_path: Path):
    """MemoryStore should treat base_dir as the root containing memory/."""
    store = MemoryStore(tmp_path)
    assert store.base_dir == tmp_path
    assert store.memory_dir == tmp_path / "memory"
    assert store.memory_file == tmp_path / "memory" / "MEMORY.md"


def test_memory_store_round_trips_long_term(tmp_path: Path):
    store = MemoryStore(tmp_path)
    store.write_long_term("User prefers dark mode.")
    assert store.read_long_term() == "User prefers dark mode."


def test_memory_store_appends_today(tmp_path: Path):
    from syll.utils.helpers import today_date

    store = MemoryStore(tmp_path)
    store.append_today("- learned user's name is Alice")
    store.append_today("- user dislikes popups")

    today_file = store.get_today_file()
    text = today_file.read_text(encoding="utf-8")
    assert today_date() in text
    assert "Alice" in text
    assert "popups" in text


def test_memory_store_scope_label(tmp_path: Path):
    store = MemoryStore(tmp_path, scope="workspace")
    assert store.scope == "workspace"
    global_store = MemoryStore(tmp_path, scope="global")
    assert global_store.scope == "global"


def test_memory_store_lists_daily_files_newest_first(tmp_path: Path):
    from syll.utils.helpers import ensure_dir

    store = MemoryStore(tmp_path)
    ensure_dir(store.memory_dir)
    (store.memory_dir / "2026-01-01.md").write_text("old", encoding="utf-8")
    (store.memory_dir / "2026-07-07.md").write_text("new", encoding="utf-8")
    files = store.list_memory_files()
    assert [f.name for f in files] == ["2026-07-07.md", "2026-01-01.md"]


def test_global_memory_store_uses_default_path():
    """GlobalMemoryStore should default to ~/.syll/global_memory."""
    store = GlobalMemoryStore()
    assert store.base_dir == get_global_memory_path()
    assert store.scope == "global"


def test_global_memory_store_accepts_override(tmp_path: Path):
    store = GlobalMemoryStore(tmp_path)
    assert store.base_dir == tmp_path
    assert store.scope == "global"
