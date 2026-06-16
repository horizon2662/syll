"""Global (main-agent) memory: the PROJECT.md layer.

The main orchestrator keeps a single ``workspace/PROJECT.md`` holding
cross-skill, whole-project facts: the top-level goal, hard constraints,
key decisions, and milestone outcomes. This is the ``app_id``/``org_id``
scope in Mem0's multi-scope model (arXiv:2504.19413) -- the layer above
per-skill procedural memory.

The main agent can *distill* a slice of this global memory into a skill's
``context_slice`` when it spawns a subagent, so a subagent gets relevant
global context without receiving the whole project history (Anthropic's
"avoid the game of telephone").

Does **not** modify the original ``memory.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


class GlobalMemory:
    """Main-agent global memory (PROJECT.md)."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.file = workspace / "PROJECT.md"

    def load(self) -> str:
        if self.file.exists():
            return self.file.read_text(encoding="utf-8").strip()
        return ""

    def init_goal(self, goal: str) -> None:
        """Seed PROJECT.md with the top-level goal if not present."""
        if self.load():
            return
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.file.write_text(
            f"# Project\n\n**Goal**: {goal}\n**Started**: {today}\n\n"
            "## Log\n",
            encoding="utf-8",
        )

    def log(self, note: str) -> None:
        """Append a timestamped log line (milestone outcome, decision, blocker)."""
        today = datetime.now(timezone.utc).isoformat(timespec="seconds")
        text = self.load()
        header = "" if text else "# Project\n\n## Log\n"
        sep = "" if (text and text.endswith("\n")) else "\n"
        self.file.write_text(
            f"{header}{text}{sep}- [{today}] {note}\n", encoding="utf-8"
        )

    def distill_for_skill(self, skill: str, query: str = "", max_chars: int = 2000) -> str:
        """Slice of global memory relevant to a skill, for the subagent context.

        Naive but sufficient: keep the Goal + recent log lines that mention the
        skill name or the query terms. Keeps the subagent's injected context
        small (no full project history)."""
        text = self.load()
        if not text:
            return ""
        lines = text.splitlines()
        out: list[str] = []
        # Always keep the Goal line.
        for ln in lines:
            if ln.strip().startswith("**Goal**") or ln.strip().startswith("#"):
                out.append(ln)
        terms = {w.lower() for w in (skill + " " + query).split() if len(w) > 2}
        for ln in lines:
            low = ln.lower()
            if any(t in low for t in terms) and ln not in out:
                out.append(ln)
        slice_ = "\n".join(out)
        if len(slice_) > max_chars:
            slice_ = slice_[:max_chars] + "\n...(truncated)"
        return f"(from PROJECT.md)\n{slice_}"
