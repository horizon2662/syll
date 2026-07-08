"""Regression tests for Runner._first_replan_after (replan cursor math).

Sync tests — ``_first_replan_after`` is a staticmethod with no await, so no
anyio marker is needed (unlike the async GUI tests).
"""

from syll.agent.aloha.act.enhanced.plan_manager import ExecutionPlan, PlanStep
from syll.agent.longhorizon.runner import Runner


def _plan(*descs_with_status) -> ExecutionPlan:
    steps = []
    for i, (desc, status) in enumerate(descs_with_status, 1):
        step = PlanStep(index=i, description=desc)
        step.status = status
        steps.append(step)
    return ExecutionPlan(task="t", skill_name="s", steps=steps)


def test_first_replan_after_last_failed_step_exits_loop():
    """A FAILED last step with no spliced replan must advance to ``len(steps)``
    so ``while i < len(plan.steps)`` exits — NOT back to the FAILED step.

    Regression for the ``min(idx+1, len-1)`` clamp bug, which returned
    ``len-1`` (the failed step itself) and caused it to be re-attempted until
    the replan cap.
    """
    plan = _plan(("do a", "DONE"), ("do b", "FAILED"))
    assert Runner._first_replan_after(plan, "do b") == len(plan.steps)


def test_first_replan_after_middle_failed_step_advances_one():
    """Sanity: a middle FAILED step advances to the step right after it."""
    plan = _plan(("do a", "DONE"), ("do b", "FAILED"), ("do c", "PENDING"))
    assert Runner._first_replan_after(plan, "do b") == 2


def test_first_replan_after_unknown_returns_zero():
    """If no matching FAILED step is found, fall back to the start."""
    plan = _plan(("do a", "DONE"), ("do b", "DONE"))
    assert Runner._first_replan_after(plan, "missing") == 0
