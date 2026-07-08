"""Per-run context-length meter + audit dir.

The provider already returns ``usage.prompt_tokens`` on every call (see
``litellm_provider.py`` ~line 256-268) — until now it was discarded. This
module is the sink that captures it: one append-only JSONL line per LLM call
at ``{run_dir}/context_curve.jsonl``, plus a live ``status()`` snapshot
mirroring ``RecorderManager.get_status()`` so the dashboard can poll it the
same way it polls the recorder.

Single-writer-per-run_dir: concurrent subagents do NOT write here. They fold
their cumulative tokens into ``SubagentResult`` (see ``contract.py``) and the
runner writes ONE rolled-up point per step attempt. This keeps the curve file
race-free.

Phase 1: ``budget_tokens`` defaults to 0 (unknown). The raw ``prompt_tokens``
curve is still recorded — that is the detector signal. ``utilization`` and
``over_budget`` stay None/false until phase 1.5 wires the real model window.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock


def resolve_context_window(model: str, explicit: int = 0) -> int:
    """Best-effort input-token budget for the detector.

    Priority: an explicit config value > litellm's ``model_cost`` table > 0
    (unknown). The explicit value wins because models served through proxies
    (e.g. BigModel's /api/anthropic serving GLM) are NOT in litellm's table,
    so the lookup would silently return 0 — the user must set ``context_window``
    on the endpoint (ModelsConfig) or via ``SYLL_CONTEXT_WINDOW`` (RunnerConfig).
    """
    if explicit:
        return int(explicit)
    try:
        import litellm  # type: ignore

        return int(litellm.model_cost.get(model, {}).get("max_input_tokens", 0) or 0)
    except Exception:
        return 0


class ContextMeter:
    """Append-only token-usage curve for one run (or one chat session).

    Instantiate one per run. ``record()`` once per LLM call; ``status()``
    returns a live snapshot for the web dashboard; ``write_summary()`` freezes
    the final state at run end.
    """

    def __init__(self, run_dir: Path, run_id: str = "", budget_tokens: int = 0):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.curve_path = self.run_dir / "context_curve.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        # 0 = unknown; utilization/over_budget stay None/false until set.
        self.budget_tokens = max(0, int(budget_tokens or 0))
        self.run_id = run_id or self.run_dir.name
        self._lock = Lock()
        self._seq = 0
        self._peak = 0
        self._last = 0
        self._n_calls = 0

    def record(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int = 0,
        phase: str = "main",
        node_key: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        """Log one LLM call. Returns the point dict (handy for inline logging).

        Never raises into the caller's control flow — callers may wrap in
        try/except but the file write itself is the only fallible step and it
        is guarded by the lock; an OSError here would propagate, so the
        runner calls this inside its existing telemetry try/except.
        """
        self._seq += 1
        pt = max(0, int(prompt_tokens or 0))
        ct = max(0, int(completion_tokens or 0))
        point = {
            "seq": self._seq,
            "ts": time.time(),
            "run_id": self.run_id,
            "phase": phase,
            "node_key": node_key,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "budget_tokens": self.budget_tokens or None,
            "utilization": round(pt / self.budget_tokens, 4) if self.budget_tokens else None,
        }
        # extra values are merged flat (caller-supplied keys win).
        for k, v in (extra or {}).items():
            point[k] = v
        line = json.dumps(point, ensure_ascii=False)
        with self._lock:
            self._n_calls += 1
            self._peak = max(self._peak, pt)
            self._last = pt
            with open(self.curve_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return point

    def log_decision_unit(self, du: dict) -> None:
        """Append one DecisionUnit dict to ``{run_dir}/events.jsonl``.

        Co-locating the per-attempt telemetry with the context curve lets the
        metrics endpoint read a run's full record set without scanning the
        date-sharded EventStore. The EventStore write (via ``log_node``) still
        happens for the global dashboard — this is the run-local mirror.
        """
        line = json.dumps(du, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.run_dir / "events.jsonl", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def status(self) -> dict:
        """Live snapshot — shape mirrors RecorderManager.get_status().

        The dashboard polls this (version-bumped by seq) the same way the
        recorder UI polls recording status.
        """
        with self._lock:
            return {
                "run_id": self.run_id,
                "seq": self._seq,
                "n_calls": self._n_calls,
                "last_prompt_tokens": self._last,
                "peak_prompt_tokens": self._peak,
                "budget_tokens": self.budget_tokens or None,
                "utilization": (
                    round(self._last / self.budget_tokens, 4)
                    if self.budget_tokens else None
                ),
                "over_budget": bool(
                    self.budget_tokens and self._last > self.budget_tokens
                ),
                "curve_path": str(self.curve_path),
            }

    def write_summary(self, **fields) -> None:
        """Freeze final state at run end. Extra fields merged in (outcome, etc.)."""
        data = {"run_id": self.run_id, **self.status(), **fields}
        self.summary_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
