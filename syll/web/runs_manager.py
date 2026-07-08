"""Run-audit discovery + read for the dashboard.

Mirrors ``RecorderManager``'s role for the recorder: a stateless helper the
``/runs`` routes call. The key decoupling — the manager READS audit files from
disk, so it needs no live reference to the ContextMeter objects the runner/loop
hold. A run is any directory containing a ``context_curve.jsonl``.

Two audit roots under the workspace (matching where phase-1 writes):
  - loop sessions:      {WS}/audit/{session_key}/context_curve.jsonl
  - longhorizon runs:   {WS}/longhorizon_runs/{session}/audit/context_curve.jsonl

For longhorizon runs, the subagent blackboard lives at
{WS}/longhorizon_runs/{session}/agents/{run_id}/result.json (written by
``UnifiedSubagentManager``). The legacy plan.md / events.jsonl / metrics
pipeline has been deprecated in favour of this blackboard.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunsManager:
    """Discover run-audit dirs under a workspace and read their artifacts."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)

    # ------------------------------------------------------------------
    # path safety — all public reads resolve a workspace-relative path and
    # reject anything that escapes the workspace (local dashboard, but still).
    # ------------------------------------------------------------------
    def _safe(self, rel: str) -> Path:
        ws = self.workspace.resolve()
        p = (self.workspace / rel).resolve()
        if p != ws and ws not in p.parents:
            raise ValueError(f"path escapes workspace: {rel}")
        return p

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------
    def list_runs(self) -> list[dict[str, Any]]:
        """All known runs, newest first. Each row carries enough for the
        dashboard list view + the dir/file ids the detail endpoints need."""
        runs: list[dict[str, Any]] = []
        # loop sessions
        audit_root = self.workspace / "audit"
        if audit_root.is_dir():
            for d in audit_root.iterdir():
                if (d / "context_curve.jsonl").is_file():
                    runs.append(self._summarize(d, kind="session"))
        # longhorizon runs
        lh_root = self.workspace / "longhorizon_runs"
        if lh_root.is_dir():
            for session_dir in lh_root.iterdir():
                ad = session_dir / "audit"
                if (ad / "context_curve.jsonl").is_file():
                    runs.append(self._summarize(ad, kind="longhorizon"))
        runs.sort(key=lambda r: r.get("mtime", 0), reverse=True)
        return runs

    def _summarize(self, audit_dir: Path, kind: str) -> dict[str, Any]:
        curve = audit_dir / "context_curve.jsonl"
        points = self._read_jsonl(curve)
        last = points[-1] if points else {}
        peak = max((p.get("prompt_tokens", 0) for p in points), default=0)
        util = last.get("utilization")
        budget = last.get("budget_tokens")
        peak_util = (peak / budget) if budget else None
        remaining = (budget - peak) if budget else None
        # Verifier pass-rate over step points that carry a verdict. This is the
        # NOISY (self-reported) success rate — always available, no oracle needed.
        # The legacy oracle-corrected telemetry panel has been removed.
        verdicts = [p.get("verdict") for p in points if p.get("verdict") in ("PASS", "FAIL")]
        pass_n = sum(1 for v in verdicts if v == "PASS")
        fail_n = sum(1 for v in verdicts if v == "FAIL")
        success_rate = (pass_n / len(verdicts)) if verdicts else None
        blackboard_rel = self._blackboard_rel(audit_dir, kind)
        return {
            "id": audit_dir.relative_to(self.workspace).as_posix(),
            "kind": kind,
            "session": self._session_name(audit_dir, kind),
            "n_points": len(points),
            "last_prompt_tokens": last.get("prompt_tokens"),
            "peak_prompt_tokens": peak,
            "budget_tokens": budget,
            "utilization": util,
            "peak_utilization": peak_util,
            "remaining_tokens": remaining,
            "over_budget": bool(util is not None and util > 1.0),
            "success_rate": success_rate,
            "pass_count": pass_n,
            "fail_count": fail_n,
            "last_ts": last.get("ts"),
            "last_phase": last.get("phase"),
            "mtime": curve.stat().st_mtime if curve.is_file() else 0,
            "blackboard_rel": blackboard_rel,
        }

    def _session_name(self, audit_dir: Path, kind: str) -> str | None:
        if kind == "longhorizon":
            # {WS}/longhorizon_runs/{session}/audit
            return audit_dir.parent.name
        return audit_dir.name  # session dir name

    def _blackboard_rel(self, audit_dir: Path, kind: str) -> str | None:
        """Relative path (from workspace) of the blackboard result.json for this run, or None."""
        if kind != "longhorizon":
            return None
        # {WS}/longhorizon_runs/{session}/agents/{run_id}/result.json
        base = audit_dir.parent / "agents"
        if not base.is_dir():
            return None
        results = sorted(
            base.glob("*/result.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not results:
            return None
        return results[0].relative_to(self.workspace).as_posix()

    # ------------------------------------------------------------------
    # readers (detail endpoints)
    # ------------------------------------------------------------------
    def read_curve(self, rel_dir: str) -> list[dict[str, Any]]:
        return self._read_jsonl(self._safe(rel_dir) / "context_curve.jsonl")

    def status(self, rel_dir: str) -> dict[str, Any]:
        audit_dir = self._safe(rel_dir)
        kind = "longhorizon" if "longhorizon_runs" in audit_dir.parts else "session"
        return self._summarize(audit_dir, kind=kind)

    def read_blackboard(self, rel_dir: str) -> dict[str, Any]:
        """Read the subagent blackboard for a run.

        Returns the latest ``result.json`` + ``progress.md`` found under
        ``{run_workspace}/agents/{run_id}/``. Legacy plan.md / events.jsonl
        are no longer produced.
        """
        audit_dir = self._safe(rel_dir)
        run_ws = audit_dir.parent
        agents_dir = run_ws / "agents"
        out: dict[str, Any] = {"result": None, "progress": None}
        if not agents_dir.is_dir():
            return out
        results = sorted(
            agents_dir.glob("*/result.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not results:
            return out
        latest = results[0]
        try:
            out["result"] = json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            out["result"] = None
        progress = latest.parent / "progress.md"
        if progress.is_file():
            try:
                out["progress"] = progress.read_text(encoding="utf-8")
            except Exception:
                out["progress"] = None
        return out

    def read_actions(self, rel_dir: str) -> list[dict[str, Any]]:
        """Grounded-action rows for a run (from {audit_dir}/actions.jsonl).

        Each row: model_position (qwen 0-1000), executor_position (screen px),
        action type, screenshot, step/attempt — the grounding diagnostic feed.
        """
        return self._read_jsonl(self._safe(rel_dir) / "actions.jsonl")

    def recent_actions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Most recent grounded actions across ALL runs, newest last.

        Each is tagged with its run id + session so the Performance feed can
        show where qwen actually clicked (model 0-1000 → executor px).
        """
        out: list[dict[str, Any]] = []
        for r in self.list_runs():
            try:
                acts = self.read_actions(r["id"])
            except Exception:
                continue
            for a in acts:
                a = dict(a)
                a["run_id"] = r["id"]
                a["session"] = r.get("session")
                out.append(a)
        out.sort(key=lambda a: a.get("ts", 0))
        return out[-limit:]

    def performance_summary(self) -> dict[str, Any]:
        """Aggregate token usage across ALL runs, split main-agent vs subagent.

        Feeds the Performance tab. The split:
          main = phase 'orchestrator' (loop chat) or 'gui_planner' (GUI planning).
          sub  = phase 'step' (subagent attempts) or 'gui_actor' (GUI action grounding).
          other= any other phase bucket.
        prompt_tokens is the context-consumption metric (what fills the window);
        completion_tokens is output cost. Both summed.
        """
        runs = self.list_runs()
        buckets = {
            "main": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
            "sub": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
            "other": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
        }
        actions_by_type: dict[str, int] = {}
        per_run: list[dict[str, Any]] = []

        def _bucket_for(phase: str) -> str:
            if phase in ("orchestrator", "gui_planner"):
                return "main"
            if phase in ("step", "gui_actor"):
                return "sub"
            return "other"

        for r in runs:
            try:
                pts = self.read_curve(r["id"])
            except Exception:
                pts = []
            row = {
                "id": r["id"], "kind": r["kind"], "session": r.get("session"),
                "orchestrator": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
                "step": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
                "peak_prompt_tokens": r.get("peak_prompt_tokens", 0),
                "budget_tokens": r.get("budget_tokens"),
                "remaining_tokens": r.get("remaining_tokens"),
                "success_rate": r.get("success_rate"),
                "pass_count": r.get("pass_count", 0),
                "fail_count": r.get("fail_count", 0),
                "actions": 0,
                "actions_by_type": {},
            }
            for p in pts:
                phase = p.get("phase") or ""
                pt = int(p.get("prompt_tokens", 0) or 0)
                ct = int(p.get("completion_tokens", 0) or 0)
                key = _bucket_for(phase)
                buckets[key]["prompt_tokens"] += pt
                buckets[key]["completion_tokens"] += ct
                buckets[key]["calls"] += 1
                # Per-run aggregation mirrors the top-level buckets for display.
                if key == "main":
                    cell = row["orchestrator"]
                elif key == "sub":
                    cell = row["step"]
                else:
                    cell = None
                if cell is not None:
                    cell["prompt_tokens"] += pt
                    cell["completion_tokens"] += ct
                    cell["calls"] += 1
            # Grounded actions (model 0-1000 coords + executor screen px) —
            # the grounding-diagnostic feed. Count by action type.
            try:
                acts = self.read_actions(r["id"])
            except Exception:
                acts = []
            row["actions"] = len(acts)
            for a in acts:
                t = str(a.get("action") or "UNKNOWN")
                row["actions_by_type"][t] = row["actions_by_type"].get(t, 0) + 1
                actions_by_type[t] = actions_by_type.get(t, 0) + 1
            per_run.append(row)
        total = {
            k: sum(buckets[b][k] for b in buckets)
            for k in ("prompt_tokens", "completion_tokens", "calls")
        }
        # Context headroom + success, derived from per-run peaks/verdicts (each
        # `r` already carries these from _summarize). global_peak = the worst
        # context any single run reached -> how close we came to filling the window.
        global_peak = max((row["peak_prompt_tokens"] for row in per_run), default=0)
        budget = next((row["budget_tokens"] for row in per_run if row["budget_tokens"]), None)
        pass_total = sum(row["pass_count"] for row in per_run)
        fail_total = sum(row["fail_count"] for row in per_run)
        verdicts_total = pass_total + fail_total
        return {
            "total": total,
            "by_phase": {
                "main": {**buckets["main"], "label": "main agent (orchestrator + gui_planner)"},
                "sub": {**buckets["sub"], "label": "subagent (step + gui_actor)"},
                "other": {**buckets["other"], "label": "other"},
            },
            "context": {
                "peak_prompt_tokens": global_peak,
                "budget_tokens": budget,
                "remaining_at_peak": (budget - global_peak) if budget else None,
                "peak_utilization": (global_peak / budget) if budget else None,
            },
            "success": {
                "rate": (pass_total / verdicts_total) if verdicts_total else None,
                "pass": pass_total,
                "fail": fail_total,
                "label": "verifier pass-rate (self-reported)",
            },
            "actions": {
                "total": sum(actions_by_type.values()),
                "by_type": actions_by_type,
                "label": "grounded GUI actions (model 0-1000 → executor px); per-run detail at /runs/actions",
            },
            "per_run": per_run,
            "n_runs": len(runs),
        }

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue  # skip a half-flushed tail line
        return out
