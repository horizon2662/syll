"""Tool-bundle factories: the shared core tool set every agent variant registers.

The filesystem + shell + web core was open-coded in three places
(``AgentLoop._register_default_tools``, ``SubagentManager._run_subagent``,
``UnifiedSubagentManager._build_tools``). Each variant still composes its OWN
extras on top (message / spawn / cron / GUI / MCP / video-learn / ...); this
module is the single source of truth for the identical part.

Pattern: Factory Function. The three callers differ in workspace, brave key,
and whether ``EditFileTool`` is exposed, so there is no shared *configured*
instance to reuse — a function that registers the core onto the caller's own
``ToolRegistry`` is the minimal, lowest-risk abstraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from syll.agent.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from syll.agent.tools.registry import ToolRegistry
from syll.agent.tools.shell import ExecTool
from syll.agent.tools.web import WebFetchTool, WebSearchTool

if TYPE_CHECKING:
    from syll.config.schema import ExecToolConfig
    from syll.sandbox.environment import Environment


def register_core_tools(
    registry: ToolRegistry,
    *,
    workspace: Path,
    restrict_to_workspace: bool,
    exec_config: "ExecToolConfig",
    brave_api_key: str | None,
    include_edit: bool = False,
    environment: "Environment | None" = None,
) -> ToolRegistry:
    """Register the shared filesystem + shell + web tools onto ``registry``.

    The common core every agent loop / subagent needs:

    - ReadFileTool / WriteFileTool / ListDirTool (always)
    - EditFileTool only when ``include_edit=True`` (the main loop exposes it;
      subagents intentionally get read/write/list only)
    - ExecTool (shell) bound to the workspace
    - WebSearchTool / WebFetchTool

    Registration order matches the previous open-coded blocks exactly.
    Returns the same ``registry`` for chaining.
    """
    allowed_dir = workspace if restrict_to_workspace else None
    registry.register(ReadFileTool(allowed_dir=allowed_dir, environment=environment))
    registry.register(WriteFileTool(allowed_dir=allowed_dir, environment=environment))
    if include_edit:
        registry.register(EditFileTool(allowed_dir=allowed_dir, environment=environment))
    registry.register(ListDirTool(allowed_dir=allowed_dir, environment=environment))
    registry.register(
        ExecTool(
            working_dir=str(workspace),
            timeout=exec_config.timeout,
            restrict_to_workspace=restrict_to_workspace,
            environment=environment,
        )
    )
    registry.register(WebSearchTool(api_key=brave_api_key))
    registry.register(WebFetchTool())
    return registry
