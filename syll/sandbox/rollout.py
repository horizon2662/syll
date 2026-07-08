"""Rollout harness (ENPIRE R module): reset → run agent → verify → log, at scale.

A :class:`RolloutHarness` runs a suite of tasks against an agent. Each task
runs in its own :class:`~syll.sandbox.environment.Environment` (one per parallel
slot — a Docker container in production, a fresh workspace in tests):

    reset to baseline → materialize the start state → run the agent (timeout) →
    run the deterministic state-verifier → record {passed, score, agent_ok,
    tokens, wall}.

The agent reports its own ``ok`` (the noisy verifier); the state-verifier is the
deterministic oracle. Their cross-tabulation across the suite estimates β (false
-success) and γ (false-negative) of the agent's self-judgement — the same
{verdict × oracle} matrix runner.py measures in-process, lifted to the rollout
level. See plans/syll-syll-vivid-sifakis.md (Phase 1).

The harness is agent-agnostic: plug in any ``agent_runner`` callable. For the
Syll agent, that callable wraps ``UnifiedSubagentManager.run_sync`` against the
task's environment.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from syll.sandbox.environment import Environment
from syll.sandbox.verifier_synthesis import _maybe_reset
from syll.sandbox.verifiers import run_checks

# -----------------------------------------------------------------------
# Data shapes
# -----------------------------------------------------------------------

EnvironmentFactory = Callable[[], Environment]
"""Returns a fresh Environment per parallel slot (a container, VM, or workspace)."""


@dataclass
class Task:
    """One rollout task.

    Attributes:
        id: stable identifier (for the report + telemetry).
        instruction: the natural-language goal given to the agent.
        checks: state-verifier spec list (consumed by ``run_checks``).
        setup: optional async callable that materializes the start state in the
            env after reset (e.g. seeds input files).
        timeout_s: per-task wall-clock cap.
    """

    id: str
    instruction: str
    checks: list[dict[str, Any]] = field(default_factory=list)
    setup: Callable[[Environment], Awaitable[None]] | None = None
    timeout_s: float = 3600.0


@dataclass
class AgentRun:
    """What the agent reports back for one task.

    ``ok`` is the agent's *noisy* self-verdict (used to estimate β/γ against the
    deterministic oracle). ``tokens`` is the agent's LLM spend on this task.
    """

    ok: bool | None = None
    tokens: int = 0
    detail: str = ""


AgentRunner = Callable[[Environment, str], Awaitable[AgentRun]]
"""Drives ``env`` to attempt ``instruction``; returns the agent's run summary."""


@dataclass
class TaskResult:
    task_id: str
    passed: bool  # deterministic oracle (state-verifier)
    score: float
    agent_ok: bool | None  # noisy verdict (None if the agent didn't report)
    tokens: int
    wall_ms: int
    error: str | None = None
    check_details: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RolloutReport:
    results: list[TaskResult]

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return sum(1 for r in self.results if r.passed) / self.n if self.n else 0.0

    @property
    def mean_score(self) -> float:
        return sum(r.score for r in self.results) / self.n if self.n else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.results)

    @property
    def mean_wall_ms(self) -> float:
        return sum(r.wall_ms for r in self.results) / self.n if self.n else 0.0

    @property
    def time_to_success_ms(self) -> float:
        wins = [r.wall_ms for r in self.results if r.passed]
        return sum(wins) / len(wins) if wins else 0.0

    def confusion(self) -> dict[str, int]:
        """{agent_ok × passed} counts over tasks where the agent reported ok."""
        tp = fp = tn = fn = 0  # passed is the ground truth; agent_ok the noisy verdict
        for r in self.results:
            if r.agent_ok is None:
                continue
            if r.agent_ok and r.passed:
                tp += 1
            elif r.agent_ok and not r.passed:
                fp += 1  # false success — the β errors
            elif (not r.agent_ok) and r.passed:
                fn += 1  # false failure — the γ errors
            else:
                tn += 1
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    @property
    def beta(self) -> float | None:
        """False-success rate of the agent's self-judgement (β in the
        verifier-ceiling law). None if the agent never reported ok."""
        c = self.confusion()
        denom = c["tp"] + c["fp"]
        return c["fp"] / denom if denom else None

    @property
    def gamma(self) -> float | None:
        """False-failure rate (γ). None if the agent never reported not-ok."""
        c = self.confusion()
        denom = c["tn"] + c["fn"]
        return c["fn"] / denom if denom else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "pass_rate": self.pass_rate,
            "mean_score": self.mean_score,
            "total_tokens": self.total_tokens,
            "mean_wall_ms": self.mean_wall_ms,
            "time_to_success_ms": self.time_to_success_ms,
            "confusion": self.confusion(),
            "beta": self.beta,
            "gamma": self.gamma,
            "results": [
                {
                    "task_id": r.task_id,
                    "passed": r.passed,
                    "score": r.score,
                    "agent_ok": r.agent_ok,
                    "tokens": r.tokens,
                    "wall_ms": r.wall_ms,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


# -----------------------------------------------------------------------
# Harness
# -----------------------------------------------------------------------


class RolloutHarness:
    """Run a task suite with per-task reset → agent → verify → log.

    Args:
        env_factory: returns a fresh Environment per parallel slot.
        agent_runner: drives one task in one env.
        parallelism: hard cap (ENPIRE: token cost super-linear beyond ~4).
    """

    def __init__(
        self,
        env_factory: EnvironmentFactory,
        agent_runner: AgentRunner,
        parallelism: int = 4,
    ) -> None:
        self._env_factory = env_factory
        self._agent_runner = agent_runner
        self.parallelism = max(1, parallelism)

    async def run(self, tasks: list[Task]) -> RolloutReport:
        sem = asyncio.Semaphore(self.parallelism)

        async def one(task: Task) -> TaskResult:
            async with sem:
                env = self._env_factory()
                return await self._run_task(env, task)

        results = await asyncio.gather(*(one(t) for t in tasks))
        return RolloutReport(results=list(results))

    async def _run_task(self, env: Environment, task: Task) -> TaskResult:
        t0 = time.monotonic()
        agent_ok: bool | None = None
        tokens = 0
        error: str | None = None
        try:
            await _maybe_reset(env)
            if task.setup is not None:
                await task.setup(env)
            try:
                agent_run = await asyncio.wait_for(
                    self._agent_runner(env, task.instruction),
                    timeout=task.timeout_s,
                )
                agent_ok = agent_run.ok
                tokens = agent_run.tokens
            except asyncio.TimeoutError:
                error = "agent timeout"
            except Exception as exc:  # agent crash ≠ harness crash
                error = f"agent error: {exc!r}"
            verdict = await run_checks(env, task.checks)
            return TaskResult(
                task_id=task.id,
                passed=verdict.passed,
                score=verdict.score,
                agent_ok=agent_ok,
                tokens=tokens,
                wall_ms=int((time.monotonic() - t0) * 1000),
                error=error,
                check_details=verdict.evidence.get("checks", []),
            )
        except Exception as exc:  # reset/setup/verify infra failure
            return TaskResult(
                task_id=task.id,
                passed=False,
                score=0.0,
                agent_ok=None,
                tokens=tokens,
                wall_ms=int((time.monotonic() - t0) * 1000),
                error=f"harness error: {exc!r}",
            )


__all__ = [
    "Task",
    "AgentRun",
    "AgentRunner",
    "TaskResult",
    "RolloutReport",
    "RolloutHarness",
    "EnvironmentFactory",
]
