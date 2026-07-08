"""Verifier framework: state-grounded success checks for the sandbox.

Mirrors OpenComputer's per-app ``check-*`` verifier endpoints and ALE's
``evaluate()`` gate-and-score pattern. A :class:`Verifier` inspects live
environment state (files, DOM via CDP, the a11y tree, SQLite DBs) and returns
a hard ``passed`` gate plus a partial-credit ``score`` in [0, 1].

Design rules (OpenComputer: hard-coded verifiers align with human adjudication
94.1%; ALE rejects "does this look right?" at QC):
- programmatic checks first, LLM-judge only as a fallback;
- prefer deterministic checks over holistic model opinions;
- compose multi-check tasks with gate-and-score (a failed gate forces 0).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from syll.sandbox.environment import Environment


@dataclass
class VerifierResult:
    """Outcome of one verifier check.

    Attributes:
        passed: hard gate — did this criterion fully succeed?
        score: partial-credit in [0, 1] (ALE gate-and-score continuous metric).
        detail: human/agent-readable reason.
        evidence: structured proof (paths checked, values seen) for telemetry.
    """

    passed: bool
    score: float = 0.0
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def fail(detail: str, **evidence: Any) -> "VerifierResult":
        return VerifierResult(
            passed=False, score=0.0, detail=detail, evidence=evidence
        )

    @staticmethod
    def pass_(
        detail: str = "", score: float = 1.0, **evidence: Any
    ) -> "VerifierResult":
        return VerifierResult(
            passed=True, score=score, detail=detail, evidence=evidence
        )


class Verifier(ABC):
    """State-grounded success checker over a live :class:`Environment`."""

    name: str = "verifier"

    @abstractmethod
    async def check(self, env: Environment, spec: dict[str, Any]) -> VerifierResult:
        """Evaluate ``spec`` against ``env`` and return the result."""
