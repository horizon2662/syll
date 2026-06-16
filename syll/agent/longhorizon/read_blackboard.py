"""Main-agent tools for the orchestrator.

These close the loops that the first skeleton left open:

- ``ReadBlackboardTool`` -- lets the main agent read a subagent run's
  progress / result / artifacts ON DEMAND (issue 1: "only terminal result").
  This is what makes fold safe: the main agent gets a small summary via the
  bus, then pulls detail only when it actually needs it (Anthropic just-in-time
  retrieval).
- ``ListRunsTool`` -- enumerate recent subagent runs.

Both follow the syll ``Tool`` base interface (``name``/``description``/
``parameters``/``execute``) so they register into a ``ToolRegistry`` like any
builtin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from syll.agent.tools.base import Tool


class ReadBlackboardTool(Tool):
    """Read a subagent run's blackboard: progress, result, or an artifact."""

    def __init__(self, workspace: Path):
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "read_subagent"

    @property
    def description(self) -> str:
        return (
            "Inspect a background subagent's work by run_id. Read its progress "
            "log, its folded result (summary/status/diagnosis/lessons), or a "
            "named artifact file it produced. Use this instead of re-running "
            "work when you need detail."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "The subagent run id (8 chars)."},
                "what": {
                    "type": "string",
                    "enum": ["progress", "result", "artifact"],
                    "description": "progress=heartbeat log; result=folded result.json; artifact=a file in artifacts/.",
                },
                "artifact": {
                    "type": "string",
                    "description": "Required when what='artifact': the artifact filename.",
                },
            },
            "required": ["run_id", "what"],
        }

    async def execute(self, run_id: str, what: str = "result", artifact: str = "", **kwargs: Any) -> str:
        base = self._workspace / "agents" / run_id
        if not base.exists():
            return f"run {run_id} not found"

        if what == "progress":
            p = base / "progress.md"
            return p.read_text(encoding="utf-8") if p.exists() else f"no progress for {run_id}"

        if what == "result":
            r = base / "result.json"
            if not r.exists():
                return f"no result yet for {run_id} (still running?)"
            try:
                data = json.loads(r.read_text(encoding="utf-8"))
            except Exception as exc:
                return f"result.json unreadable: {exc}"
            # Return a compact, main-agent-friendly view (not the raw artifacts).
            return json.dumps(
                {
                    "status": data.get("status"),
                    "summary": data.get("summary"),
                    "diagnosis": data.get("diagnosis"),
                    "artifacts": data.get("artifacts", []),
                    "lessons": data.get("lessons", []),
                },
                ensure_ascii=False,
                indent=2,
            )

        if what == "artifact":
            if not artifact:
                return "artifact filename required"
            ap = base / "artifacts" / artifact
            if not ap.exists():
                return f"artifact {artifact} not found in run {run_id}"
            text = ap.read_text(encoding="utf-8", errors="replace")
            return text if len(text) <= 8000 else text[:8000] + "\n...(truncated)"

        return f"unknown what={what}"


class ListRunsTool(Tool):
    """List recent subagent runs and their terminal status."""

    def __init__(self, workspace: Path):
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "list_subagents"

    @property
    def description(self) -> str:
        return "List recent background subagent runs with their status (ok/failed/running)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        base = self._workspace / "agents"
        if not base.exists():
            return "no subagent runs yet"
        rows = []
        for d in sorted(base.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            rp = d / "result.json"
            if rp.exists():
                try:
                    st = json.loads(rp.read_text(encoding="utf-8")).get("status", "?")
                except Exception:
                    st = "?"
            else:
                st = "running"
            rows.append(f"- {d.name}: {st}")
        return "\n".join(rows) if rows else "no subagent runs yet"
