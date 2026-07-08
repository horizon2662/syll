"""State-grounded verifiers (OpenComputer-style ``check-*`` over live state).

Registry + gate-and-score composition. Mirrors ALE's dominant composition
pattern: a failed gate forces score 0 regardless of partial progress on
continuous metrics.

Use::

    from syll.sandbox.verifiers import run_checks
    result = await run_checks(env, [
        {"type": "file", "require_exists": ["out/report.xlsx"]},
        {"type": "sqlite", "db_path": "out/state.db",
         "query": "SELECT value FROM kv WHERE k='done'", "equals": "1"},
    ])
    # result.passed, result.score, result.evidence["checks"]
"""

from __future__ import annotations

from typing import Any

from syll.sandbox.environment import Environment

from .a11y import A11yVerifier
from .base import Verifier, VerifierResult
from .cdp import CDPVerifier
from .file import FileVerifier
from .sqlite_verifier import SQLiteVerifier

_REGISTRY: dict[str, Verifier] = {}


def _register(v: Verifier) -> Verifier:
    _REGISTRY[v.name] = v
    return v


FILE = _register(FileVerifier())
SQLITE = _register(SQLiteVerifier())
CDP = _register(CDPVerifier())
A11Y = _register(A11yVerifier())


def get_verifier(name: str) -> Verifier:
    key = (name or "").lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown verifier {name!r}; known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


async def run_checks(
    env: Environment, checks: list[dict[str, Any]]
) -> VerifierResult:
    """Run a gate-and-score list of checks (ALE pattern).

    Each dict carries a ``type`` (verifier name); the rest is its spec. A
    failing gate (``passed=False`` on any check, including a verifier error)
    forces overall ``passed=False`` and score 0. Otherwise the overall score is
    the minimum across checks (every gate must clear the bar). Per-check
    evidence is aggregated under ``evidence["checks"]``.
    """
    if not checks:
        return VerifierResult.pass_("no checks", score=1.0)

    per_check: list[dict[str, Any]] = []
    passed_all = True
    min_score = 1.0
    for c in checks:
        ctype = c.get("type") or c.get("verifier") or ""
        spec = {k: v for k, v in c.items() if k not in ("type", "verifier")}
        try:
            res = await get_verifier(ctype).check(env, spec)
        except Exception as exc:  # a verifier error is a failed gate
            res = VerifierResult.fail(f"verifier {ctype!r} errored: {exc}")
        per_check.append(
            {
                "type": ctype,
                "passed": res.passed,
                "score": res.score,
                "detail": res.detail,
            }
        )
        if not res.passed:
            passed_all = False
        min_score = min(min_score, res.score)

    if not passed_all:
        return VerifierResult(
            passed=False,
            score=0.0,
            detail="one or more gates failed",
            evidence={"checks": per_check},
        )
    return VerifierResult(
        passed=True,
        score=min_score,
        detail="all gates passed",
        evidence={"checks": per_check},
    )


__all__ = [
    "Verifier",
    "VerifierResult",
    "FileVerifier",
    "SQLiteVerifier",
    "CDPVerifier",
    "A11yVerifier",
    "get_verifier",
    "run_checks",
]
