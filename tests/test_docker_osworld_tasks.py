"""OSWorld-style live Docker test (Phase 0 deep acceptance).

OSWorld's method, per task: (1) seed an isolated VM into a known initial state,
(2) revertToSnapshot before each run, (3) let the agent act, (4) score with a
task-specific state-grounded evaluator that checks the resulting environment
against a ground-truth reference (file existence / content match / app state).

This file runs that exact pattern through the Phase 0/1 machinery against a REAL
Docker engine: RolloutHarness (reset -> setup -> agent -> run_checks) driving a
DockerEnvironment, on a small suite of real shell/file tasks. Each task ships
its own setup + its own state-grounded evaluator (a run_checks spec). A scripted
agent stands in for the LLM. Task 4 is a NEGATIVE CONTROL — the agent produces
the WRONG output and the evaluator must catch it (the whole point of OSWorld
evaluation: it has to distinguish success from failure, not just "did it run").

Gated to skip without a running daemon + the built image.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from syll.sandbox.backends.docker import DockerEnvironment, docker_exe
from syll.sandbox.rollout import AgentRun, RolloutHarness, Task

IMAGE = "syll-sandbox-base:latest"
_DOCKER = docker_exe()


def _daemon_running() -> bool:
    if _DOCKER is None:
        return False
    try:
        return subprocess.run(
            [_DOCKER, "version", "--format", "{{.Server.Version}}"],
            capture_output=True, timeout=20,
        ).returncode == 0
    except Exception:
        return False


def _image_available() -> bool:
    if not _daemon_running():
        return False
    try:
        return subprocess.run(
            [_DOCKER, "image", "inspect", IMAGE], capture_output=True, timeout=20
        ).returncode == 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _image_available(),
    reason=f"needs a RUNNING docker daemon + image {IMAGE}",
)


def _h(b: str) -> str:
    return hashlib.sha256(b.encode()).hexdigest()


def _cleanup_sandbox_containers() -> None:
    """Remove leftover syll-sandbox-* containers from the run."""
    if _DOCKER is None:
        return
    try:
        ids = subprocess.run(
            [_DOCKER, "ps", "-aq", "--filter", "name=syll-sandbox-"],
            capture_output=True, timeout=20,
        ).stdout.decode().split()
        if ids:
            subprocess.run([_DOCKER, "rm", "-f", *ids], capture_output=True, timeout=30)
    except Exception:
        pass


# -----------------------------------------------------------------------
# Task suite: each = setup(seed) + instruction + solution(agent stand-in) +
# state-grounded evaluator (run_checks spec). Ground-truth uses sha256 exact
# match (OSWorld-style: the result must equal the reference, byte-for-byte).
# ---------------------------------------------------------------------------


def _build_tasks() -> list[SimpleNamespace]:
    async def setup_csv(e):
        await e.write_file("data.csv", "name,amount\nA,10\nB,20\nC,30\n")

    async def setup_rename(e):
        for i in range(1, 6):
            await e.write_file(f"dir/old_{i}.txt", f"content {i}\n")

    async def setup_conf(e):
        await e.write_file("app.conf", "[main]\ndebug=false\nlevel=info\n")

    async def setup_count(e):
        await e.write_file("rows.csv", "h\nr1\nr2\nr3\nr4\n")

    return [
        SimpleNamespace(
            id="csv-sum",
            instruction="Sum the amount column of data.csv and write just the number to out/total.txt",
            setup=setup_csv,
            solution=["mkdir -p out", "awk -F, 'NR>1{s+=$2} END{print s}' data.csv > out/total.txt"],
            checks=[
                {"type": "file", "require_exists": ["out/total.txt"]},
                {"type": "file", "sha256": {"out/total.txt": _h("60\n")}},
            ],
        ),
        SimpleNamespace(
            id="rename-pattern",
            instruction="Rename every dir/old_*.txt to dir/new_*.txt keeping the suffix number",
            setup=setup_rename,
            solution=['cd dir && for f in old_*.txt; do mv "$f" "new_${f#old_}"; done'],
            checks=[
                {"type": "file", "require_exists": [f"dir/new_{i}.txt" for i in range(1, 6)],
                 "require_absent": [f"dir/old_{i}.txt" for i in range(1, 6)]},
            ],
        ),
        SimpleNamespace(
            id="config-edit",
            instruction="In app.conf set debug to true, leaving every other line unchanged",
            setup=setup_conf,
            solution=["sed -i 's/^debug=false$/debug=true/' app.conf"],
            checks=[{"type": "file", "sha256": {"app.conf": _h("[main]\ndebug=true\nlevel=info\n")}}],
        ),
        SimpleNamespace(  # NEGATIVE CONTROL: agent writes the wrong number.
            id="row-count-wrong",
            instruction="Count data rows in rows.csv (exclude header) and write the number to out/count.txt",
            setup=setup_count,
            solution=["mkdir -p out", "echo 99 > out/count.txt"],  # wrong (correct = 4)
            checks=[
                {"type": "file", "require_exists": ["out/count.txt"]},
                {"type": "file", "sha256": {"out/count.txt": _h("4\n")}},
            ],
        ),
    ]


@pytest.mark.anyio
async def test_osworld_style_suite_on_docker():
    osw = _build_tasks()
    solutions = {t.instruction: t.solution for t in osw}

    async def scripted_agent(env, instruction):
        # Stand-in for the LLM: runs the task's solution shell commands.
        for cmd in solutions.get(instruction, []):
            await env.exec(cmd, timeout=120)
        return AgentRun(ok=True, tokens=0)  # agent CLAIMS success (noisy verdict)

    harness = RolloutHarness(
        env_factory=lambda: DockerEnvironment(image=IMAGE),
        agent_runner=scripted_agent,
        parallelism=2,
    )
    tasks = [
        Task(id=t.id, instruction=t.instruction, checks=t.checks, setup=t.setup)
        for t in osw
    ]
    try:
        report = await harness.run(tasks)
    finally:
        _cleanup_sandbox_containers()

    by_id = {r.task_id: r for r in report.results}

    # Positive tasks: the correct solution must PASS the state-grounded evaluator.
    assert by_id["csv-sum"].passed, by_id["csv-sum"].check_details
    assert by_id["rename-pattern"].passed, by_id["rename-pattern"].check_details
    assert by_id["config-edit"].passed, by_id["config-edit"].check_details
    # Negative control: the WRONG output must be CAUGHT (evaluator fails it),
    # even though the agent claimed success — this is the OSWorld point.
    assert not by_id["row-count-wrong"].passed
    assert by_id["row-count-wrong"].agent_ok is True  # agent was wrongly confident

    # Scorecard (OSWorld-style leaderboard row).
    assert report.pass_rate == 0.75
    # β = false-success rate of the agent's self-judgement: 1 wrong of 4 claimed = 0.25.
    assert report.beta == 0.25

    print("\n=== OSWorld-style scorecard (live Docker) ===")
    for r in report.results:
        flag = "PASS" if r.passed else "FAIL"
        print(f"  [{flag}] {r.task_id:18s} score={r.score:.2f}")
    print(f"  pass_rate={report.pass_rate:.2f}  β(false-success)={report.beta}")
