"""Typed input/output contract between the main agent and a subagent.

Replaces the current fire-and-forget `_announce_result` pattern, which
pushes the subagent's *entire* final string into a single system message.

Why:
- Anthropic's multi-agent research team found that vague task descriptions
  cause subagents to "duplicate work, leave gaps", and that large outputs
  should go to the filesystem with only lightweight references passed back
  ("minimize the game of telephone").
- TaskWeave (arXiv:2606.01199) adds explicit *dependency queries* so a
  subagent declares what upstream results it needs before running.

Does **not** modify the original ``subagent.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ContractStatus = Literal["ok", "failed", "blocked", "needs_input"]
RunMode = Literal["background", "step", "skill"]


@dataclass
class SubagentContract:
    """Input contract: what the main agent hands to a subagent.

    Fields mirror what Anthropic's team found necessary for reliable
    delegation (objective / output format / tool guidance / boundaries),
    plus ``dependencies`` (TaskWeave dependency query Ψ) and
    ``context_slice`` (global memory distilled by the main agent).
    """

    task: str
    objective: str = ""
    output_format: str = ""  # e.g. "a JSON list of file paths"
    tools_hint: str = ""  # which tools/sources to prefer
    boundaries: str = ""  # what is OUT of scope
    context_slice: str = ""  # distilled from main agent's PROJECT memory
    dependencies: list[str] = field(default_factory=list)  # upstream refs to resolve first
    skill: str = ""  # if this run is a "skill", its name
    mode: RunMode = "background"

    def render_prompt(self) -> str:
        """Render the contract into a focused subagent prompt."""
        parts = [f"## Objective\n{self.objective or self.task}"]
        if self.output_format:
            parts.append(f"## Required Output\n{self.output_format}")
        if self.tools_hint:
            parts.append(f"## Tools / Sources\n{self.tools_hint}")
        if self.boundaries:
            parts.append(f"## Out of Scope\n{self.boundaries}")
        if self.context_slice:
            parts.append(f"## Context from Main Agent\n{self.context_slice}")
        if self.dependencies:
            parts.append(
                "## Dependencies (resolve these before starting)\n- "
                + "\n- ".join(self.dependencies)
            )
        parts.append(
            "## Return\nWhen done, call the `return` tool with a CONCISE "
            "summary (<= ~2000 tokens) and write any large output to your "
            "artifact directory. Do NOT paste large outputs into the return."
        )
        return "\n\n".join(parts)


@dataclass
class SubagentResult:
    """Output contract: the FOLDED return from a subagent.

    Mirrors Context-Folding (arXiv:2510.11967): the subagent's intermediate
    steps are discarded; only a condensed ``summary`` plus lightweight
    artifact *references* reach the main agent. ``diagnosis`` feeds the
    FPDA Align loop (TaskWeave). ``lessons`` are candidate skill-memory
    entries, gated by the Notetaker before writing.
    """

    run_id: str
    status: ContractStatus = "ok"
    summary: str = ""  # what the main agent sees (the "fold")
    artifacts: list[str] = field(default_factory=list)  # file paths, read on demand
    diagnosis: str = ""  # failure reason -> main agent replans
    lessons: list[str] = field(default_factory=list)  # candidate SKILL.md entries
    iterations_used: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ok"
