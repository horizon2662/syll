"""Tests for global user memory integration."""

from pathlib import Path

from syll.agent.context import ContextBuilder
from syll.agent.memory import (
    GlobalMemoryStore,
    MemoryStore,
    migrate_workspace_memory_to_global,
)
from syll.utils.helpers import ensure_dir


def test_context_builder_merges_global_and_workspace_memory(tmp_path: Path):
    """System prompt should contain both global and workspace memory sections."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    global_dir = tmp_path / "global"
    global_dir.mkdir()

    global_store = MemoryStore(global_dir, scope="global")
    global_store.write_long_term("User likes tea.")

    ws_store = MemoryStore(workspace, scope="workspace")
    ws_store.write_long_term("Project uses FastAPI.")

    ctx = ContextBuilder(workspace, global_memory=global_store)
    prompt = ctx.build_system_prompt()

    assert "## Global Memory" in prompt
    assert "User likes tea." in prompt
    assert "## Workspace Memory" in prompt
    assert "Project uses FastAPI." in prompt


def test_global_memory_store_defaults_to_user_dir(tmp_path: Path, monkeypatch):
    """GlobalMemoryStore uses ~/.syll/global_memory by default."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    store = GlobalMemoryStore()
    assert ".syll" in str(store.base_dir)
    assert "global_memory" in str(store.base_dir)


def test_migrate_workspace_memory_to_global_copies_content(tmp_path: Path):
    """Migration copies a non-template workspace MEMORY.md to global once."""
    workspace = tmp_path / "ws"
    ensure_dir(workspace / "memory")
    (workspace / "memory" / "MEMORY.md").write_text("User prefers Emacs.", encoding="utf-8")

    global_dir = tmp_path / "global"
    global_store = MemoryStore(global_dir, scope="global")

    result = migrate_workspace_memory_to_global(workspace, global_dir=global_dir)
    assert result is not None
    assert result.exists()
    assert result.read_text(encoding="utf-8") == "User prefers Emacs."

    # Second migration is a no-op because global MEMORY.md now exists.
    result2 = migrate_workspace_memory_to_global(workspace, global_dir=global_dir)
    assert result2 is None


def test_migrate_skips_template_workspace_memory(tmp_path: Path):
    """Migration does not copy the shipped template placeholder."""
    workspace = tmp_path / "ws"
    ensure_dir(workspace / "memory")
    (workspace / "memory" / "MEMORY.md").write_text(
        "# Long-term Memory\n\n(Important facts about the user)\n", encoding="utf-8"
    )

    result = migrate_workspace_memory_to_global(workspace)
    assert result is None


def test_migrate_no_op_when_workspace_memory_missing(tmp_path: Path):
    """Migration returns None if workspace has no MEMORY.md."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert migrate_workspace_memory_to_global(workspace) is None
