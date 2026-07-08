"""Tests for ``syll.web.runs_manager.RunsManager``.

Covers the blackboard-oriented dashboard surface after the legacy
``/runs/plan``, ``/runs/events`` and ``/runs/metrics`` endpoints were removed.
"""

import json

import pytest

from syll.web.runs_manager import RunsManager


@pytest.fixture
def mgr(tmp_path):
    return RunsManager(tmp_path)


def _write_jsonl(path, lines):
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def test_list_runs_discovers_blackboard(mgr, tmp_path):
    audit = tmp_path / "longhorizon_runs" / "gui_123" / "audit"
    audit.mkdir(parents=True)
    _write_jsonl(audit / "context_curve.jsonl", [
        {"prompt_tokens": 100, "completion_tokens": 10, "phase": "orchestrator"},
    ])
    agents = audit.parent / "agents" / "abc123"
    agents.mkdir(parents=True)
    (agents / "result.json").write_text(json.dumps({"status": "ok", "summary": "done"}), encoding="utf-8")

    runs = mgr.list_runs()
    assert len(runs) == 1
    run = runs[0]
    assert run["kind"] == "longhorizon"
    assert run["blackboard_rel"] == "longhorizon_runs/gui_123/agents/abc123/result.json"


def test_read_blackboard_returns_result_and_progress(mgr, tmp_path):
    audit = tmp_path / "longhorizon_runs" / "gui_123" / "audit"
    audit.mkdir(parents=True)
    _write_jsonl(audit / "context_curve.jsonl", [])
    agents = audit.parent / "agents" / "abc123"
    agents.mkdir(parents=True)
    (agents / "result.json").write_text(json.dumps({"status": "ok", "summary": "done"}), encoding="utf-8")
    (agents / "progress.md").write_text("# Progress\n- step 1\n", encoding="utf-8")

    bb = mgr.read_blackboard("longhorizon_runs/gui_123/audit")
    assert bb["result"]["status"] == "ok"
    assert "step 1" in bb["progress"]


def test_performance_summary_buckets_gui_phases(mgr, tmp_path):
    audit = tmp_path / "audit" / "session_1"
    audit.mkdir(parents=True)
    _write_jsonl(audit / "context_curve.jsonl", [
        {"prompt_tokens": 100, "completion_tokens": 10, "phase": "orchestrator"},
        {"prompt_tokens": 40, "completion_tokens": 5, "phase": "gui_planner"},
        {"prompt_tokens": 30, "completion_tokens": 4, "phase": "gui_actor"},
        {"prompt_tokens": 20, "completion_tokens": 3, "phase": "unknown"},
    ])

    perf = mgr.performance_summary()
    assert perf["by_phase"]["main"]["prompt_tokens"] == 140
    assert perf["by_phase"]["sub"]["prompt_tokens"] == 30
    assert perf["by_phase"]["other"]["prompt_tokens"] == 20
    assert perf["total"]["prompt_tokens"] == 190
