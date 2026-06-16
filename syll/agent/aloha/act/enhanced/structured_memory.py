"""Structured memory: compress raw action history into semantic summaries.

Inherits from :class:`MemoryStore` so that all existing memory
functionality is preserved.  Adds trajectory compression inspired by
MGA (WSDM'25) — each step is represented by a 5-dimension abstraction
instead of raw action strings.

Does **not** modify the original ``memory.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from syll.agent.memory import MemoryStore


@dataclass
class StepRecord:
    """Lightweight record of one executed step."""

    index: int
    action: str
    expectation: str
    verify_status: str  # SUCCESS / NO_CHANGE / UNCERTAIN
    diagnosis: str = ""
    timestamp: str = ""


class StructuredMemory(MemoryStore):
    """Structured memory that compresses raw trajectories into summaries.

    Extends ``MemoryStore`` with:
    - Per-step recording of action + verification result.
    - Periodic compression into a semantic summary (via LLM or rules).
    - A compact ``get_execution_context()`` method that returns only the
      recent N raw steps plus a compressed summary of older steps.

    Usage::

        mem = StructuredMemory(workspace=Path("~/.syll/workspace"))
        mem.record_step(1, "Click File", "Menu opens", "SUCCESS")
        mem.record_step(2, "Click Save", "Dialog appears", "NO_CHANGE", "Coordinate off")

        context = mem.get_execution_context(max_recent_steps=3)
        # → inject into planner prompt instead of raw action_history
    """

    def __init__(self, workspace: Path):
        super().__init__(workspace)
        self._history_file = self.memory_dir / "execution_history.md"
        self._compressed_file = self.memory_dir / "execution_summary.md"
        self._steps: list[StepRecord] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        index: int,
        action: str,
        expectation: str,
        verify_status: str,
        diagnosis: str = "",
    ) -> None:
        """Append a step record to the in-memory list and persist it."""
        record = StepRecord(
            index=index,
            action=action,
            expectation=expectation,
            verify_status=verify_status,
            diagnosis=diagnosis,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._steps.append(record)
        self._append_to_history_file(record)

    # ------------------------------------------------------------------
    # Context retrieval (replaces raw action_history)
    # ------------------------------------------------------------------

    def get_execution_context(self, max_recent_steps: int = 3) -> str:
        """Return a compact context string for the planner.

        Format:
        1. Compressed summary of older steps (from file).
        2. Last N raw step records with verification results.
        3. Error pattern summary.

        This is designed to replace the ``action_history`` parameter in
        the planner call, reducing context length significantly.
        """
        parts: list[str] = []

        # 1. Compressed summary of older steps
        compressed = self._load_compressed_summary()
        if compressed:
            parts.append(f"## Previous Steps (summary)\n{compressed}")

        # 2. Recent N raw steps
        recent = self._steps[-max_recent_steps:]
        if recent:
            parts.append("## Recent Steps")
            for step in recent:
                icon = "✅" if step.verify_status == "SUCCESS" else "❌"
                line = (
                    f"{icon} Step {step.index}: {step.action} "
                    f"→ {step.verify_status}"
                )
                if step.diagnosis:
                    line += f" ({step.diagnosis})"
                parts.append(line)

        # 3. Error pattern summary
        errors = self._get_error_patterns()
        if errors:
            parts.append("## Error Patterns")
            parts.append(errors)

        return "\n\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Compression (LLM-based, optional)
    # ------------------------------------------------------------------

    async def compress_history(self, model: str = "gpt-4o") -> str:
        """Compress the full step history into a semantic summary via LLM.

        The summary follows MGA's 5-dimension abstraction:
        1. Interface state evolution (what changed on screen)
        2. Operation effect analysis (what each action accomplished)
        3. Behavioral pattern recognition (repeated actions, loops)
        4. Error identification (failed actions and reasons)
        5. State consistency verification (current state validity)

        The compressed result is cached to ``execution_summary.md``.
        """
        if not self._steps:
            return ""

        # Build a compact text of all steps for the LLM
        steps_text = "\n".join(
            f"Step {s.index}: {s.action} | Status: {s.verify_status} "
            f"| Diagnosis: {s.diagnosis}"
            for s in self._steps
        )

        prompt = (
            "Compress the following GUI action history into a concise "
            "semantic summary (max 300 words). Follow these 5 dimensions:\n"
            "1. What was accomplished (interface changes)\n"
            "2. Effect of each action\n"
            "3. Behavioral patterns (repeats, loops)\n"
            "4. Errors encountered and reasons\n"
            "5. Current state assessment\n\n"
            f"Action History:\n{steps_text}\n\n"
            "Summary:"
        )

        try:
            import litellm

            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0,
            )
            summary = response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning(f"Failed to compress history via LLM: {exc}")
            summary = self._rule_based_compression()

        # Cache the compressed summary
        self._compressed_file.write_text(summary, encoding="utf-8")
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_to_history_file(self, record: StepRecord) -> None:
        """Append one step record to the history file."""
        icon = "✅" if record.verify_status == "SUCCESS" else "❌"
        line = (
            f"{icon} [{record.timestamp}] Step {record.index}: "
            f"{record.action} → {record.verify_status}"
        )
        if record.diagnosis:
            line += f" | {record.diagnosis}"
        line += "\n"

        if not self._history_file.exists():
            header = f"# Execution History\n\n{line}"
            self._history_file.write_text(header, encoding="utf-8")
        else:
            with open(self._history_file, "a", encoding="utf-8") as f:
                f.write(line)

    def _load_compressed_summary(self) -> str:
        """Load the previously compressed summary from file."""
        if self._compressed_file.exists():
            return self._compressed_file.read_text(encoding="utf-8").strip()
        return ""

    def _get_error_patterns(self) -> str:
        """Identify recurring error patterns in the step history."""
        failed = [s for s in self._steps if s.verify_status == "NO_CHANGE"]
        if not failed:
            return ""

        # Group by similar diagnosis
        patterns: dict[str, list[int]] = {}
        for s in failed:
            key = s.diagnosis or "unknown"
            # Simplify the key
            if "coordinate" in key.lower() or "position" in key.lower():
                key = "coordinate_error"
            elif "not found" in key.lower() or "missing" in key.lower():
                key = "element_not_found"
            elif "timeout" in key.lower() or "loading" in key.lower():
                key = "timing_issue"
            patterns.setdefault(key, []).append(s.index)

        lines = []
        for pattern, indices in patterns.items():
            steps_str = ", ".join(f"Step {i}" for i in indices)
            count = len(indices)
            lines.append(
                f"- {pattern}: {count} occurrences ({steps_str})"
            )
        return "\n".join(lines)

    def _rule_based_compression(self) -> str:
        """Fallback compression without LLM: extract key facts."""
        total = len(self._steps)
        succeeded = sum(
            1 for s in self._steps if s.verify_status == "SUCCESS"
        )
        failed = sum(
            1 for s in self._steps if s.verify_status == "NO_CHANGE"
        )

        lines = [
            f"Executed {total} steps: {succeeded} succeeded, {failed} failed.",
        ]

        # Last 3 steps as one-liners
        for s in self._steps[-3:]:
            status = "✅" if s.verify_status == "SUCCESS" else "❌"
            lines.append(f"  {status} {s.action}")

        return "\n".join(lines)
