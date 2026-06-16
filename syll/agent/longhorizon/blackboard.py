"""Filesystem blackboard: the shared-state layer between agents.

Each subagent run gets its own directory under
``workspace/agents/{run_id}/`` holding ``contract.json``, ``progress.md``
(a heartbeat), and ``result.json``. The main agent reads these files
instead of receiving a fat system message.

This is the Syll adaptation of two ideas:
- Anthropic's "subagent output to a filesystem to minimize the game of
  telephone" -- large outputs persist independently, only references
  pass back.
- The orchestration survey's "state & knowledge management" unit
  (arXiv:2601.13671), which explicitly manages checkpoints, progress, and
  agent states as a first-class component.

Does **not** modify the original ``subagent.py``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


class Blackboard:
    """Per-run filesystem scratchpad shared between a subagent and the main agent."""

    def __init__(self, workspace: Path, run_id: str):
        self.run_id = run_id
        self.dir = workspace / "agents" / run_id
        self.artifacts_dir = self.dir / "artifacts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # contract (input)
    # ------------------------------------------------------------------
    @property
    def contract_path(self) -> Path:
        return self.dir / "contract.json"

    def write_contract(self, contract: dict) -> None:
        self.contract_path.write_text(
            json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # progress heartbeat (intermediate state, not just terminal)
    # ------------------------------------------------------------------
    @property
    def progress_path(self) -> Path:
        return self.dir / "progress.md"

    def write_progress(self, note: str) -> None:
        """Append a heartbeat line. Lets the main agent observe intermediate
        state without the subagent pushing messages -- fixes the
        'only terminal result' gap in the current ``_announce_result``."""
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"- [{ts}] {note}\n"
        if not self.progress_path.exists():
            self.progress_path.write_text(
                f"# Progress -- run {self.run_id}\n{line}", encoding="utf-8"
            )
        else:
            with self.progress_path.open("a", encoding="utf-8") as f:
                f.write(line)

    # ------------------------------------------------------------------
    # result (folded output)
    # ------------------------------------------------------------------
    @property
    def result_path(self) -> Path:
        return self.dir / "result.json"

    def write_result(self, result_dict: dict) -> None:
        self.result_path.write_text(
            json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def read_result(self) -> dict | None:
        if not self.result_path.exists():
            return None
        try:
            return json.loads(self.result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"blackboard[{self.run_id}] failed to read result: {exc}")
            return None

    def artifact_path(self, name: str) -> Path:
        """Path for the subagent to write a named artifact file."""
        return self.artifacts_dir / name
