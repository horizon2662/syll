"""Wiring tests for the Phase 3 evolution hook in Runner.

Constructs a real Runner (stub provider, tmp workspace) and verifies:
- the Evolver is wired when ``enable_evolution`` is on, absent when off;
- the running β estimate is computed from verdict × oracle counters;
- ``_maybe_evolve`` delegates to the Evolver with a file gate built from the
  step's artifacts, and skips cleanly when there is no evolver / no artifacts.

No real LLM — the Evolver is replaced with a recording stub.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from syll.agent.longhorizon.config import RunnerConfig
from syll.agent.longhorizon.evolution import EvolutionReport
from syll.agent.longhorizon.runner import Runner
from syll.providers.base import LLMResponse, LLMProvider


class _StubProvider(LLMProvider):
    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        return LLMResponse(content="")

    def get_default_model(self) -> str:
        return "stub"


class _StubEvolver:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def evolve(self, failure, beta=None):
        self.calls.append((failure, beta))
        return EvolutionReport(failure=failure, evolved=False, reason="stub")


def _make_runner(tmp_path: Path, enable_evolution: bool) -> Runner:
    cfg = RunnerConfig(model="stub", workspace=tmp_path, enable_evolution=enable_evolution)
    return Runner(cfg, _StubProvider())


def test_evolver_wired_when_enabled(tmp_path: Path):
    runner = _make_runner(tmp_path, enable_evolution=True)
    assert runner.evolver is not None


def test_evolver_absent_when_disabled(tmp_path: Path):
    runner = _make_runner(tmp_path, enable_evolution=False)
    assert runner.evolver is None


def test_evolution_beta_computation(tmp_path: Path):
    runner = _make_runner(tmp_path, enable_evolution=False)
    assert runner._evolution_beta is None  # no judgements yet
    runner._evolve_pass = 4
    runner._evolve_false_success = 1
    assert runner._evolution_beta == 0.25


@pytest.mark.anyio
async def test_maybe_evolve_skips_without_evolver(tmp_path: Path):
    runner = _make_runner(tmp_path, enable_evolution=False)
    step = SimpleNamespace(index=1, description="do thing")
    result = SimpleNamespace(artifacts=["out/a.txt"], diagnosis="d", summary="s")
    # No evolver → returns immediately, no error.
    await runner._maybe_evolve(plan=None, step=step, cur_milestone=1, result=result)


@pytest.mark.anyio
async def test_maybe_evolve_skips_when_no_artifacts(tmp_path: Path):
    runner = _make_runner(tmp_path, enable_evolution=False)
    stub = _StubEvolver()
    runner.evolver = stub
    step = SimpleNamespace(index=1, description="do thing")
    result = SimpleNamespace(artifacts=[], diagnosis="d", summary="s")
    await runner._maybe_evolve(plan=None, step=step, cur_milestone=1, result=result)
    assert stub.calls == []  # nothing to gate on → skip


@pytest.mark.anyio
async def test_maybe_evolve_delegates_with_artifacts(tmp_path: Path, monkeypatch):
    # The Docker pre-check must pass so the injected stub actually runs.
    monkeypatch.setattr(
        "syll.sandbox.backends.docker.docker_daemon_up", lambda **k: True
    )
    runner = _make_runner(tmp_path, enable_evolution=False)
    stub = _StubEvolver()
    runner.evolver = stub  # inject
    runner._evolve_pass = 2
    runner._evolve_false_success = 0  # β = 0.0
    step = SimpleNamespace(index=3, description="export sheet to out/r.csv")
    result = SimpleNamespace(
        artifacts=["out/r.csv"], diagnosis="clicked wrong menu", summary="tried X"
    )
    await runner._maybe_evolve(plan=None, step=step, cur_milestone=1, result=result)
    assert len(stub.calls) == 1
    failure, beta = stub.calls[0]
    assert failure.instruction == "export sheet to out/r.csv"
    assert failure.checks == [{"type": "file", "require_exists": ["out/r.csv"]}]
    assert beta == 0.0


@pytest.mark.anyio
async def test_maybe_evolve_skips_when_daemon_down(tmp_path: Path, monkeypatch):
    """Docker down → skip WITHOUT calling the evolver (don't burn an LLM call)."""
    monkeypatch.setattr(
        "syll.sandbox.backends.docker.docker_daemon_up", lambda **k: False
    )
    runner = _make_runner(tmp_path, enable_evolution=False)
    stub = _StubEvolver()
    runner.evolver = stub
    step = SimpleNamespace(index=1, description="do thing")
    result = SimpleNamespace(artifacts=["out/a.txt"], diagnosis="d", summary="s")
    await runner._maybe_evolve(plan=None, step=step, cur_milestone=1, result=result)
    assert stub.calls == []  # daemon down → never proposed


@pytest.mark.anyio
async def test_maybe_evolve_catches_evolve_error(tmp_path: Path, monkeypatch):
    """An evolve() crash must never propagate — the run continues regardless."""
    monkeypatch.setattr(
        "syll.sandbox.backends.docker.docker_daemon_up", lambda **k: True
    )

    class _Boom:
        async def evolve(self, failure, beta=None):
            raise RuntimeError("sandbox exploded")

    runner = _make_runner(tmp_path, enable_evolution=False)
    runner.evolver = _Boom()
    step = SimpleNamespace(index=1, description="do thing")
    result = SimpleNamespace(artifacts=["out/a.txt"], diagnosis="d", summary="s")
    # Must not raise.
    await runner._maybe_evolve(plan=None, step=step, cur_milestone=1, result=result)

