"""Value Objects for the enhanced GUI step/retry loop.

These replace the primitive-obsession (untyped dicts + loose strings) that the
retry machinery used before, so step state and failed attempts are passed as
typed records instead of ``{"action": ..., "category": ...}`` dicts —
eliminating the "what keys does this dict have?" class of adapter bugs (the
same class as the ``actor_api_key`` NameError and the ``category``-not-flowing
breaks caught in the adapter audit).

- ``FailedAttempt``: one failed edge on the retry state-action graph.
- ``StepContext``: the intermediate state of one step's retry loop
  (screenshot / plan / action / verify). Used by the phase-2 ``execute``
  refactor where sub-methods take a single ``ctx`` instead of an 8-arg list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureCategory(Enum):
    """Why a GUI action's verification returned NO_CHANGE.

    Type-safe replacement for the loose ``"COORD_OFF"`` strings that were
    scattered across action_verifier / FailedAttempt / _RETRY_STRATEGIES.
    Adding a category = add an enum value + a RetryStrategy hint.
    """

    COORD_OFF = "COORD_OFF"
    ELEMENT_ABSENT = "ELEMENT_ABSENT"
    OCCLUDED = "OCCLUDED"
    LOADING = "LOADING"
    NO_VISUAL_FEEDBACK = "NO_VISUAL_FEEDBACK"
    WORKFLOW_ORDER = "WORKFLOW_ORDER"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, raw: str) -> "FailureCategory":
        """Parse a category string (from VerifyResult.category or an LLM
        response) into the enum. Unknown / empty strings → UNKNOWN."""
        if not raw:
            return cls.UNKNOWN
        cleaned = raw.strip().upper().replace(" ", "_").replace("-", "_")
        try:
            return cls(cleaned)
        except ValueError:
            return cls.UNKNOWN


@dataclass
class FailedAttempt:
    """One failed retry edge on the current screen (G2PO state-action graph,
    inference side). Replaces the previous ``{action, position, category,
    reason}`` dict — now typed, so callers can't misspell a key.
    """

    action: str = ""
    position: list[int] | None = None
    category: FailureCategory = FailureCategory.UNKNOWN
    reason: str = ""


@dataclass
class StepContext:
    """Intermediate state of one step's retry loop (Value Object).

    Replaces the data-clump (``plan_output`` / ``action_dict`` /
    ``model_position`` / ``executor_position`` / ``step_verify`` /
    ``failed_attempts``) that ``execute`` threaded through its inner loop.
    Phase-2 sub-methods receive a single ``ctx`` instead of a long parameter
    list (Smell B4/B5: data clump + long param list).
    """

    step: int
    attempt: int = 0
    screenshot_b64: str = ""
    spatial_context: str = ""
    plan_output: dict = field(default_factory=dict)
    plan_action: str = ""
    action_dict: dict = field(default_factory=dict)
    model_position: list[int] | None = None
    executor_position: list[int] | None = None
    # VerifyResult | None — typed as Any to avoid a circular import with
    # action_verifier (which imports nothing from this module).
    step_verify: Any = None
    succeeded: bool = False
    executor_result: str = ""
    screenshot_path: str = ""


@dataclass
class ExecuteContext:
    """Per-execute services + accumulators (Value Object). Packs the ~15 setup
    locals so phase-2 sub-methods take ``(exec_ctx, step_ctx)`` instead of a
    long parameter list (Smell B5: long parameter list). Built once in
    ``execute()`` setup and threaded through ``_record_step`` /
    ``_capture_observation`` / ``_verify_step`` / ...
    """

    cfg: Any
    skill_name: str
    instruction: str
    mode: str
    planner_model: str
    planner: Any
    verifier: Any
    executor: Any
    spatial_analyzer: Any
    structured_memory: Any
    plan_manager: Any
    plan: Any
    screenshots: list = field(default_factory=list)
    steps_log: list = field(default_factory=list)
    action_history: list = field(default_factory=list)
