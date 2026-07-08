"""Telemetry + measurement for the verifier-ceiling law.

Standalone module — does **not** modify ``runner.py``, ``executor.py`` or
``events.py``. The runner imports ``log_node`` to emit one record per decision-unit
attempt; offline analysis calls the estimators to recover (p, beta, gamma, p_eff).

Design invariant (the anti-circularity rule): the verifier is the *device under
test*. ``p`` is read from the ORACLE; ``beta``/``gamma`` from the verifier-vs-oracle
confusion matrix. No stream measures itself. See ``paper-telemetry-spec.md``.

The estimator functions operate on plain dicts (``DecisionUnit.model_dump()`` or rows
parsed from ``events/*.jsonl``) so they are unit-testable without pydantic/loguru.
Validated against synthetic logs in ``~/.syll/theory_sim/fit_law.py``.
"""
from __future__ import annotations

import hashlib
import math
import uuid
from typing import Any, Iterable, Literal

try:
    from pydantic import BaseModel, Field
except Exception:  # estimators still work without pydantic
    BaseModel = object  # type: ignore

    def Field(default=None, **_):  # type: ignore
        return default


def node_key(level: str, milestone: int, step_index: int, desc: str) -> str:
    """Stable id for a node across retries (groups attempts of the same node)."""
    h = hashlib.sha1(desc.encode("utf-8")).hexdigest()[:8]
    return f"{level}:{milestone}.{step_index}:{h}"


def normalize_verdict(raw: str | None) -> str | None:
    """Map Syll's ActionVerifier statuses onto PASS/FAIL/UNCERTAIN."""
    if raw is None:
        return None
    r = str(raw).upper()
    if "SUCCESS" in r or r == "PASS":
        return "PASS"
    if "NO_CHANGE" in r or "UNCHANGED" in r or "FAIL" in r:
        return "FAIL"
    return "UNCERTAIN"


class DecisionUnit(BaseModel):  # type: ignore[misc]
    """One execution attempt of one decision unit (see spec §2)."""

    unit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    node_key: str = ""
    parent_key: str | None = None
    level: Literal["milestone", "step", "action"] = "step"
    seq: int = 0
    depth: int = 0
    attempt: int = 1
    is_retry: bool = False

    instruction: str = ""
    action: str | None = None
    skill_name: str | None = None
    executor_model: str = ""
    thinking: bool = False
    executor_tokens_in: int = 0
    executor_tokens_out: int = 0

    verifier_type: str = "none"
    verifier_verdict: str | None = None       # normalized PASS/FAIL/UNCERTAIN
    verifier_raw: str | None = None
    verifier_confidence: float | None = None
    verifier_score: float | None = None
    verifier_model: str | None = None
    verifier_tokens: int | None = None

    oracle_available: bool = False
    oracle_label: str | None = None           # "correct" / "wrong" / None
    oracle_type: str | None = None
    oracle_detail: str | None = None

    triggered_recovery: bool = False
    recovery_mode: str = "none"               # none/retry_fresh/retry_replay/replan
    terminal: bool = False
    terminal_outcome: str | None = None       # correct/silent_wrong/given_up
    diagnosis: str | None = None

    context_tokens_in: int = 0
    context_chars: int = 0
    memory_injected_chars: int = 0
    compaction_active: bool = False
    active_context_tokens: int = 0

    env_state_hash_before: str | None = None
    env_state_hash_after: str | None = None
    reversible: bool | None = None
    screenshot_before: str | None = None
    screenshot_after: str | None = None

    wall_ms: int = 0
    is_probe: bool = False


def log_node(store: Any, du: DecisionUnit, agent_type: str = "gui_agent") -> None:
    """Emit a DecisionUnit through Syll's existing EventStore (additive)."""
    from syll.agent.events import Event, EventContent, EventSource

    payload = du.model_dump() if hasattr(du, "model_dump") else dict(du.__dict__)
    media = [m for m in (du.screenshot_before, du.screenshot_after) if m]
    store.log_event(
        Event(
            agent_type=agent_type,
            event_type="action",
            source=EventSource(platform="telemetry", chat_id=du.run_id, user_id="instrument"),
            content=EventContent(text=du.instruction, media=media,
                                 metadata={"telemetry": payload}),
            tags=["telemetry", du.level],
        )
    )


# ----------------------------------------------------------------------
# Estimators (plain-dict in; validated in theory_sim/fit_law.py)
# ----------------------------------------------------------------------

def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a proportion. Returns (phat, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return p, center - half, center + half


def calibrate(rows: Iterable[dict], level: str = "step",
              verifier_type: str | None = None) -> dict:
    """Estimate beta (miss) and gamma (false-alarm) from the verifier-vs-oracle
    confusion matrix, over rows that have BOTH a verdict and an oracle label."""
    rows = [r for r in rows if r.get("level") == level and r.get("oracle_label")
            and r.get("verifier_verdict") in ("PASS", "FAIL")
            and (verifier_type is None or r.get("verifier_type") == verifier_type)]
    wrong = [r for r in rows if r["oracle_label"] == "wrong"]
    corr = [r for r in rows if r["oracle_label"] == "correct"]
    b_k = sum(r["verifier_verdict"] == "PASS" for r in wrong)
    g_k = sum(r["verifier_verdict"] == "FAIL" for r in corr)
    beta, b_lo, b_hi = _wilson(b_k, len(wrong))
    gamma, g_lo, g_hi = _wilson(g_k, len(corr))
    J = 1 - gamma - beta
    return dict(beta=beta, beta_ci=(b_lo, b_hi), gamma=gamma, gamma_ci=(g_lo, g_hi),
                youden_J=J, n_wrong=len(wrong), n_correct=len(corr),
                invertible=J > 0.05)


def p_from_oracle(rows: Iterable[dict], level: str = "step") -> float:
    fa = [r for r in rows if r.get("level") == level and r.get("attempt") == 1
          and r.get("oracle_label")]
    if not fa:
        return float("nan")
    return sum(r["oracle_label"] == "correct" for r in fa) / len(fa)


def p_rogan_gladen(rows: Iterable[dict], beta: float, gamma: float,
                   level: str = "step") -> dict:
    """Recover true p from the verifier ALONE (no oracle), de-biased with (beta,gamma).
    p = (A - beta) / (1 - gamma - beta), A = verifier accept-rate on first attempts."""
    fa = [r for r in rows if r.get("level") == level and r.get("attempt") == 1
          and r.get("verifier_verdict") in ("PASS", "FAIL")]
    if not fa:
        return dict(p=float("nan"), A=float("nan"), J=1 - gamma - beta, reliable=False)
    A = sum(r["verifier_verdict"] == "PASS" for r in fa) / len(fa)
    J = 1 - gamma - beta
    p = (A - beta) / J if abs(J) > 1e-9 else float("nan")
    return dict(p=p, A=A, J=J, reliable=J > 0.05)


def p_eff_observed(rows: Iterable[dict], level: str = "step") -> float:
    term = [r for r in rows if r.get("level") == level and r.get("terminal")]
    if not term:
        return float("nan")
    return sum(r["terminal_outcome"] == "correct" for r in term) / len(term)


def p_eff_theory(p: float, beta: float, gamma: float = 0.0) -> float:
    a, b = p * (1 - gamma), (1 - p) * beta
    return a / (a + b) if (a + b) else float("nan")


def n_eff(rows: Iterable[dict], run_id: str, level: str = "step") -> int:
    keys = {r["node_key"] for r in rows if r.get("run_id") == run_id
            and r.get("level") == level and r.get("terminal")}
    return len(keys)


def derive_terminal(rows: list[dict], level: str = "step") -> list[dict]:
    """Mark the last attempt per (run_id, node_key) terminal and set terminal_outcome
    from its (verdict, oracle_label). Needed for REAL emitted logs, where the runner
    emits one record per attempt without knowing in advance which is terminal.
    Mutates and returns rows. Idempotent for logs that already carry terminal info."""
    last: dict[tuple, dict] = {}
    for r in rows:
        if r.get("level") != level:
            continue
        k = (r.get("run_id"), r.get("node_key"))
        cur = last.get(k)
        if cur is None or r.get("attempt", 1) >= cur.get("attempt", 1):
            last[k] = r
    for r in rows:
        if r.get("level") == level:
            r["terminal"] = False
    for r in last.values():
        r["terminal"] = True
        v, o = r.get("verifier_verdict"), r.get("oracle_label")
        if v == "PASS":
            r["terminal_outcome"] = "silent_wrong" if o == "wrong" else "correct"
        else:
            r["terminal_outcome"] = "given_up"
    return rows


def fit_and_check(rows: list[dict], level: str = "step") -> dict:
    """Full pipeline: calibrate -> recover p two ways -> predict & compare the law."""
    rows = derive_terminal(list(rows), level)
    cal = calibrate(rows, level)
    p_o = p_from_oracle(rows, level)
    p_rg = p_rogan_gladen(rows, cal["beta"], cal["gamma"], level)
    pe_obs = p_eff_observed(rows, level)
    pe_thy = p_eff_theory(p_o, cal["beta"], cal["gamma"])
    H_pred = math.log(0.5) / math.log(pe_thy) if 0 < pe_thy < 1 else float("inf")
    return dict(calibration=cal, p_oracle=p_o, p_rogan_gladen=p_rg,
                p_eff_observed=pe_obs, p_eff_theory=pe_thy, H_half_pred=H_pred)
