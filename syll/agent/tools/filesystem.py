"""Filesystem tools for reading and writing files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

from syll.agent.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from syll.sandbox.environment import Environment


def _resolve_path(path: str, allowed_dir: str | Path | None) -> Path:
    """Resolve a path, optionally restricting it to a workspace."""
    file_path = Path(path).expanduser().resolve()
    if allowed_dir is not None:
        allowed = Path(allowed_dir).expanduser().resolve()
        try:
            file_path.relative_to(allowed)
        except ValueError:
            raise ValueError(
                f"Path '{path}' is outside allowed directory '{allowed}'"
            ) from None
    return file_path


class ReadFileTool(Tool):
    """Read the contents of a file."""

    def __init__(
        self,
        allowed_dir: str | Path | None = None,
        environment: "Environment | None" = None,
    ):
        self._allowed_dir = allowed_dir
        self._environment = environment

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the full contents of a file as UTF-8 text."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute path to the file to read.",
                }
            },
            "required": ["path"],
        }

    async def execute(self, path: str, **kwargs: Any) -> str:
        try:
            if self._environment is not None:
                return await self._environment.read_file(path)

            file_path = _resolve_path(path, self._allowed_dir)
            if not file_path.exists():
                return f"Error: File '{path}' not found."
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error: {e}"


class WriteFileTool(Tool):
    """Write text to a file, creating parent directories if necessary."""

    def __init__(
        self,
        allowed_dir: str | Path | None = None,
        environment: "Environment | None" = None,
    ):
        self._allowed_dir = allowed_dir
        self._environment = environment

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write UTF-8 text to a file, creating parent directories as needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write to the file.",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        try:
            if self._environment is not None:
                await self._environment.write_file(path, content)
                return f"File '{path}' written."

            file_path = _resolve_path(path, self._allowed_dir)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return f"File '{path}' written."
        except Exception as e:
            return f"Error: {e}"


class EditFileTool(Tool):
    """Replace an exact substring in a file."""

    def __init__(
        self,
        allowed_dir: str | Path | None = None,
        environment: "Environment | None" = None,
    ):
        self._allowed_dir = allowed_dir
        self._environment = environment

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Replace every occurrence of old_text with new_text in a file."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute path to the file to edit.",
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to replace.",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text.",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(
        self, path: str, old_text: str, new_text: str, **kwargs: Any
    ) -> str:
        try:
            if self._environment is not None:
                await self._environment.edit_file(path, old_text, new_text)
                return f"File '{path}' edited."

            file_path = _resolve_path(path, self._allowed_dir)
            if not file_path.exists():
                return f"Error: File '{path}' not found."
            text = file_path.read_text(encoding="utf-8")
            if old_text not in text:
                return "Error: old_text not found in file."
            file_path.write_text(text.replace(old_text, new_text), encoding="utf-8")
            return f"File '{path}' edited."
        except Exception as e:
            return f"Error: {e}"


class ListDirTool(Tool):
    """List the contents of a directory."""

    def __init__(
        self,
        allowed_dir: str | Path | None = None,
        environment: "Environment | None" = None,
    ):
        self._allowed_dir = allowed_dir
        self._environment = environment

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List the files and directories inside a directory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute path to the directory to list.",
                }
            },
            "required": ["path"],
        }

    async def execute(self, path: str, **kwargs: Any) -> str:
        try:
            if self._environment is not None:
                entries = await self._environment.list_dir(path)
                return "\n".join(entries) if entries else "(empty directory)"

            dir_path = _resolve_path(path, self._allowed_dir)
            if not dir_path.exists():
                return f"Error: Directory '{path}' not found."
            if not dir_path.is_dir():
                return f"Error: '{path}' is not a directory."
            entries = sorted(dir_path.iterdir())
            if not entries:
                return "(empty directory)"
            lines = []
            for entry in entries:
                label = f"{entry.name}/" if entry.is_dir() else entry.name
                lines.append(label)
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"
