"""Memory system for persistent agent memory."""

import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from syll.utils.helpers import ensure_dir, today_date


class MemoryStore:
    """
    Memory system for the agent.

    Supports daily notes (memory/YYYY-MM-DD.md) and long-term memory (MEMORY.md).
    The store is rooted at ``base_dir`` so it can back either a workspace-local
    memory directory or a global user memory directory.
    """

    def __init__(
        self,
        base_dir: Path,
        scope: str = "workspace",
        *,
        subdir: str | None = None,
        key: str | None = None,
        memory_filename: str | None = None,
    ):
        """
        Args:
            base_dir: Root directory that contains the memory folder (or the
                parent of ``subdir``).
            scope: Human-readable scope label (e.g. ``workspace``, ``global``,
                ``skill``, ``gui_execution``).
            subdir: Optional subdirectory under ``base_dir`` to use instead of
                the default ``memory/``.  This lets one ``MemoryStore`` back
                skills, ledgers, or other scoped storage without forking the
                persistence interface.
            key: Optional further namespace under ``subdir`` (e.g. skill name).
            memory_filename: Optional filename for the long-term memory file.
                Defaults to ``MEMORY.md``.
        """
        self.base_dir = Path(base_dir)
        self.scope = scope
        # Backwards compatibility: the old constructor took ``workspace`` and
        # callers may still reference ``store.workspace``.
        self.workspace = self.base_dir
        if subdir:
            self.memory_dir = self.base_dir / subdir
        else:
            self.memory_dir = self.base_dir / "memory"
        if key:
            self.memory_dir = self.memory_dir / key
        ensure_dir(self.memory_dir)
        self.memory_file = self.memory_dir / (memory_filename or "MEMORY.md")

    def get_today_file(self) -> Path:
        """Get path to today's memory file."""
        return self.memory_dir / f"{today_date()}.md"

    def read_today(self) -> str:
        """Read today's memory notes."""
        today_file = self.get_today_file()
        if today_file.exists():
            return today_file.read_text(encoding="utf-8")
        return ""

    def append_today(self, content: str) -> None:
        """Append content to today's memory notes."""
        ensure_dir(self.memory_dir)
        today_file = self.get_today_file()

        if today_file.exists():
            existing = today_file.read_text(encoding="utf-8")
            content = existing + "\n" + content
        else:
            # Add header for new day
            header = f"# {today_date()}\n\n"
            content = header + content

        today_file.write_text(content, encoding="utf-8")

    def read_long_term(self) -> str:
        """Read long-term memory (MEMORY.md)."""
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        """Write to long-term memory (MEMORY.md)."""
        ensure_dir(self.memory_dir)
        self.memory_file.write_text(content, encoding="utf-8")

    def list_memory_files(self) -> list[Path]:
        """List all memory files sorted by date (newest first)."""
        if not self.memory_dir.exists():
            return []

        files = list(self.memory_dir.glob("????-??-??.md"))
        return sorted(files, reverse=True)

    def get_memory_context(self) -> str:
        """
        Get memory context for the agent.

        Returns:
            Formatted memory context including long-term and recent memories.
        """
        parts = []

        # Long-term memory
        long_term = self.read_long_term()
        if long_term:
            parts.append("## Long-term Memory\n" + long_term)

        # Today's notes
        today = self.read_today()
        if today:
            parts.append("## Today's Notes\n" + today)

        return "\n\n".join(parts) if parts else ""


class GlobalMemoryStore(MemoryStore):
    """Convenience alias for a user-scoped global memory store."""

    def __init__(self, base_dir: Path | None = None):
        from syll.utils.helpers import get_global_memory_path

        super().__init__(base_dir or get_global_memory_path(), scope="global")


def migrate_workspace_memory_to_global(
    workspace: Path, global_dir: Path | None = None
) -> Path | None:
    """One-shot migration: copy an existing workspace MEMORY.md to global memory.

    Copies only when:
      - the global MEMORY.md does not yet exist, and
      - the workspace MEMORY.md exists and contains non-template content.

    Args:
        workspace: Workspace path containing ``memory/MEMORY.md``.
        global_dir: Optional global memory root. Defaults to ``~/.syll/global_memory``.

    Returns:
        The global memory file path if migration ran, otherwise None.
    """
    from syll.utils.helpers import get_global_memory_path

    ws_memory = workspace / "memory" / "MEMORY.md"
    if not ws_memory.exists():
        return None

    ws_text = ws_memory.read_text(encoding="utf-8").strip()
    # Skip the shipped template placeholder.
    if not ws_text or "(Important facts about the user)" in ws_text:
        return None

    target_dir = global_dir or get_global_memory_path()
    global_memory_file = target_dir / "memory" / "MEMORY.md"
    if global_memory_file.exists():
        return None

    ensure_dir(global_memory_file.parent)
    global_memory_file.write_text(ws_text, encoding="utf-8")
    return global_memory_file


class JsonMemoryStore(MemoryStore):
    """JSON-backed memory store for small structured records.

    Uses the same scope/base_dir/key layout as :class:`MemoryStore`, but the
    long-term memory file is treated as a JSON document rather than markdown.
    """

    def read_json(self) -> Any | None:
        """Read and parse the JSON memory file, or ``None`` if absent."""
        if not self.memory_file.exists():
            return None
        try:
            return json.loads(self.memory_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Failed to read JSON memory {self.memory_file}: {exc}")
            return None

    def write_json(self, data: Any) -> None:
        """Atomically write ``data`` as JSON to the memory file."""
        ensure_dir(self.memory_dir)
        tmp = self.memory_file.with_suffix(self.memory_file.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.memory_file)
