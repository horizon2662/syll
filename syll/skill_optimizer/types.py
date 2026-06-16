"""Core types for the Syll skill optimizer.

Adapted from skillopt.types — dataclass definitions for the 6-stage pipeline.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal


# ── Atomic types ────────────────────────────────────────────────────────────

EditOp = Literal["append", "insert_after", "replace", "delete"]


@dataclass
class Edit:
    """A single edit operation on a skill document."""

    op: EditOp
    content: str = ""
    target: str = ""
    support_count: int | None = None
    source_type: Literal["failure", "success"] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Edit:
        return cls(
            op=d.get("op", "append"),
            content=d.get("content", ""),
            target=d.get("target", ""),
            support_count=d.get("support_count"),
            source_type=d.get("source_type"),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"op": self.op, "content": self.content}
        if self.target:
            d["target"] = self.target
        if self.support_count is not None:
            d["support_count"] = self.support_count
        if self.source_type is not None:
            d["source_type"] = self.source_type
        return d


@dataclass
class Patch:
    """A set of edits with reasoning — output of Reflect/Aggregate/Select."""

    edits: list[Edit] = field(default_factory=list)
    reasoning: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Patch:
        edits_raw = d.get("edits", [])
        return cls(
            edits=[Edit.from_dict(e) if isinstance(e, dict) else e for e in edits_raw],
            reasoning=d.get("reasoning", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning": self.reasoning,
            "edits": [e.to_dict() if isinstance(e, Edit) else e for e in self.edits],
        }


# ── Stage ① ROLLOUT ────────────────────────────────────────────────────────

@dataclass
class RolloutResult:
    """Result of a single task rollout."""

    id: str
    hard: int          # 1 = success, 0 = failure
    soft: float        # partial credit score (0..1)
    task: str = ""     # original task prompt
    response: str = ""  # agent response
    expected: str = ""  # expected answer / ground truth
    conversation: list[dict[str, Any]] = field(default_factory=list)
    fail_reason: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> RolloutResult:
        return cls(
            id=str(d.get("id", "")),
            hard=int(d.get("hard", 0)),
            soft=float(d.get("soft", 0.0)),
            task=str(d.get("task", "")),
            response=str(d.get("response", "")),
            expected=str(d.get("expected", "")),
            conversation=d.get("conversation", []),
            fail_reason=str(d.get("fail_reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "hard": self.hard, "soft": self.soft}
        if self.task:
            d["task"] = self.task
        if self.response:
            d["response"] = self.response
        if self.expected:
            d["expected"] = self.expected
        if self.fail_reason:
            d["fail_reason"] = self.fail_reason
        return d


# ── Gate ────────────────────────────────────────────────────────────────────

GateAction = Literal["accept_new_best", "accept", "reject"]
GateMetric = Literal["hard", "soft", "mixed"]


@dataclass(frozen=True)
class GateResult:
    """Immutable outcome of the validation gate."""

    action: GateAction
    current_skill: str
    current_score: float
    best_skill: str
    best_score: float
    best_step: int


# ── Utilities ───────────────────────────────────────────────────────────────

def compute_score(results: list[RolloutResult | dict]) -> tuple[float, float]:
    """Compute hard and soft accuracy from rollout results."""
    if not results:
        return 0.0, 0.0

    def _hard(r: object) -> float:
        return float(r.hard if hasattr(r, "hard") else r.get("hard", 0))  # type: ignore[union-attr]

    def _soft(r: object) -> float:
        return float(r.soft if hasattr(r, "soft") else r.get("soft", 0.0))  # type: ignore[union-attr]

    hard = sum(_hard(r) for r in results) / len(results)
    soft = sum(_soft(r) for r in results) / len(results)
    return hard, soft


def skill_hash(content: str) -> str:
    """Return a short deterministic hash of skill content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]
