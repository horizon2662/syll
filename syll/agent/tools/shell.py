"""Shell execution tool with basic safety guardrails."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from syll.agent.tools.base import Tool, ToolResult
from syll.config.schema import ExecToolConfig

if TYPE_CHECKING:
    from syll.sandbox.environment import Environment


@dataclass
class ExecToolConfig:
    """Configuration for ExecTool."""

    timeout: int = 60
    allowed_commands: list[str] | None = None


class ExecTool(Tool):
    """Execute a shell command and return stdout/stderr."""

    DANGEROUS_PATTERNS = {
        "rm -rf /",
        ":(){ :|:& };:",
        "dd if=/dev/zero of=/dev/sda",
        "> /dev/sda",
        "mkfs",
    }

    def __init__(
        self,
        working_dir: str | None = None,
        timeout: int = 60,
        restrict_to_workspace: bool = True,
        environment: "Environment | None" = None,
    ):
        self.working_dir = working_dir
        self.timeout = timeout
        self.restrict_to_workspace = restrict_to_workspace
        self._environment = environment

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command. Returns stdout, stderr, and exit code. "
            "Supports setting a working directory via working_dir."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute. Multiline commands are supported.",
                },
                "working_dir": {
                    "type": ["string", "null"],
                    "description": "Optional working directory for the command.",
                },
            },
            "required": ["command"],
        }

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Return error string if command violates safety rules, else None."""
        lower = command.lower()
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.lower() in lower:
                return f"Error: Command blocked by safety guard (contains '{pattern}')."

        if self.restrict_to_workspace and self.working_dir:
            resolved_cwd = os.path.abspath(cwd)
            resolved_workspace = os.path.abspath(self.working_dir)
            if resolved_cwd == resolved_workspace or resolved_cwd.startswith(
                resolved_workspace + os.sep
            ):
                return None
            return (
                f"Error: Working directory '{cwd}' is outside the allowed workspace "
                f"'{self.working_dir}'."
            )
        return None

    async def execute(
        self, command: str, working_dir: str | None = None, **kwargs: Any
    ) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()

        guard = self._guard_command(command, cwd)
        if guard:
            return guard

        if self._environment is not None:
            result = await self._environment.exec(command, cwd=cwd, timeout=self.timeout)
            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append(result.stderr)
            if result.returncode != 0:
                output_parts.append(f"Exit code: {result.returncode}")
            result_text = "\n".join(output_parts)
            if len(result_text) > 10000:
                result_text = result_text[:10000] + "\n... [truncated]"
            return result_text

        # Fallback for callers that have not wired an Environment yet.
        import asyncio

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            return "Error: Command timed out."
        except Exception as e:
            return f"Error executing command: {e}"

        output_parts = []
        decoded_stdout = stdout.decode(errors="replace")
        decoded_stderr = stderr.decode(errors="replace")
        if decoded_stdout:
            output_parts.append(decoded_stdout)
        if decoded_stderr:
            output_parts.append(decoded_stderr)
        if proc.returncode != 0:
            output_parts.append(f"Exit code: {proc.returncode}")
        result_text = "\n".join(output_parts)
        if len(result_text) > 10000:
            result_text = result_text[:10000] + "\n... [truncated]"
        return result_text
