"""Tests for the rollout harness.

Uses a fake agent_runner + a fresh LocalEnvironment per task (one per parallel
slot). No Docker and no real LLM. Exercises the reset → run → verify → log
flow, the {agent_ok × passed} β/γ estimates, parallelism, and agent-error
isolation (an agent crash is recorded, not propagated).
"""

from __future__ import annotations

import tempfile

import pytest

from syll.sandbox.environment import LocalEnvironment
from syll.sandbox.rollout import (
    AgentRun,
    RolloutHarness,
    Task,
)


def _factory() -> LocalEnvironment:
    return LocalEnvironment(workspace_root=tempfile.mkdtemp())


async def _create_runner(env, instruction: str) -> AgentRun:
    """Fake agent: ``create:<path>:<content>`` writes the file and (over-
    confidently) reports ok=True; anything else reports ok=True without acting."""
    if instruction.startswith("create:"):
        _, path, content = instruction.split(":", 2)
        await env.write_file(path, content)
    return AgentRun(ok=True, tokens=100)


@pytest.mark.anyio
async def test_rollout_pass_rate_and_beta():
    tasks = [
        Task(
            id="t1",
            instruction="create:out/a.txt:hello",
            checks=[{"type": "file", "require_exists": ["out/a.txt"]}],
        ),
        Task(
            id="t2",
            instruction="do-nothing",
            checks=[{"type": "file", "require_exists": ["out/b.txt"]}],
        ),
    ]
    harness = RolloutHarness(_factory, _create_runner, parallelism=2)
    report = await harness.run(tasks)

    assert report.n == 2
    assert report.pass_rate == 0.5
    # Both agents claimed ok; t2 is a false success → β = 1/2.
    assert report.confusion() == {"tp": 1, "fp": 1, "tn": 0, "fn": 0}
    assert report.beta == 0.5
    assert report.total_tokens == 200
    assert report.time_to_success_ms > 0


@pytest.mark.anyio
async def test_rollout_gamma_when_agent_too_harsh():
    # Agent writes the file (task passes the oracle) but self-reports ok=False.
    async def harsh(env, instruction):
        _, path, content = instruction.split(":", 2)
        await env.write_file(path, content)
        return AgentRun(ok=False, tokens=80)

    tasks = [
        Task(
            id="g1",
            instruction="create:out/a.txt:x",
            checks=[{"type": "file", "require_exists": ["out/a.txt"]}],
        )
    ]
    report = await RolloutHarness(_factory, harsh).run(tasks)
    assert report.pass_rate == 1.0
    assert report.confusion() == {"tp": 0, "fp": 0, "tn": 0, "fn": 1}
    assert report.gamma == 1.0


@pytest.mark.anyio
async def test_rollout_parallel_completes_all():
    tasks = [
        Task(
            id=f"t{i}",
            instruction=f"create:out/{i}.txt:x",
            checks=[{"type": "file", "require_exists": [f"out/{i}.txt"]}],
        )
        for i in range(4)
    ]
    report = await RolloutHarness(_factory, _create_runner, parallelism=2).run(tasks)
    assert report.n == 4
    assert report.pass_rate == 1.0


@pytest.mark.anyio
async def test_rollout_agent_crash_is_recorded_not_raised():
    async def crashing(env, instruction):
        raise RuntimeError("agent blew up")

    tasks = [
        Task(
            id="c1",
            instruction="anything",
            checks=[{"type": "file", "require_exists": ["out/a.txt"]}],
        )
    ]
    report = await RolloutHarness(_factory, crashing).run(tasks)
    assert report.n == 1
    r = report.results[0]
    assert r.passed is False  # verifier sees no file
    assert r.error is not None and "agent blew up" in r.error


@pytest.mark.anyio
async def test_rollout_to_dict_shape():
    tasks = [
        Task(
            id="d1",
            instruction="create:out/a.txt:hello",
            checks=[{"type": "file", "require_exists": ["out/a.txt"]}],
        )
    ]
    report = await RolloutHarness(_factory, _create_runner).run(tasks)
    d = report.to_dict()
    assert set(d) >= {
        "n", "pass_rate", "mean_score", "total_tokens", "mean_wall_ms",
        "time_to_success_ms", "confusion", "beta", "gamma", "results",
    }
    assert d["results"][0]["task_id"] == "d1"
    assert d["results"][0]["passed"] is True
