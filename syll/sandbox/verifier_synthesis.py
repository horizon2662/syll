"""Agent-synthesized verifiers (ENPIRE EN / Gym-Anything / ALE philosophy).

Instead of hand-coding a per-task success checker, a coding agent writes one
from a few success/failure demonstrations. This module asks an LLM to emit a
JSON array of check specs in the registry's format
(:mod:`syll.sandbox.verifiers`), then optionally smoke-tests the result against
the labeled demos (OpenComputer's self-evolving verification gate) before
returning it. The output is directly consumable by ``run_checks``.

The verifier is intentionally restricted to the deterministic registry (file /
sqlite / cdp / a11y) — no free-form LLM-judge — so a synthesized verifier, once
smoke-validated, can be re-run offline against any saved trajectory.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from syll.providers.base import LLMProvider
from syll.sandbox.environment import Environment
from syll.sandbox.verifiers import run_checks

# Menu presented to the synthesis model. Keep in sync with the registry.
VERIFIER_MENU = """\
You write a SUCCESS VERIFIER for a computer-use task. Emit a JSON array of check
objects — the task succeeds only if EVERY check passes (gate-and-score). Available
deterministic check types (use these, not free-form judging):

  {"type":"file","require_exists":["path",...],"require_absent":["path",...],
   "sha256":{"path":"<hex>"}}
  {"type":"sqlite","db_path":"path","query":"SELECT ... returning one scalar",
   "equals":"x"}            # or use "contains":"substring"
  {"type":"cdp","url_equals":"https://..."}    # or {"type":"cdp","url_contains":"frag"}
  {"type":"a11y","window_title_contains":"<title fragment>"}

Rules:
- Output ONLY a JSON array. No prose, no markdown fences.
- Prefer file / sqlite checks. Use cdp only for browser tasks, a11y only when a
  window title is the real signal.
- A check that the failure examples should FAIL must be specific enough to
  distinguish them from the success examples."""


Materializer = Callable[[Environment], Awaitable[None]]


@dataclass
class Demo:
    """One labeled example of a task outcome.

    Attributes:
        should_pass: whether this outcome is a success.
        describe: natural-language description of the env state (goes into the
            synthesis prompt — what files exist, their content, DB rows, etc.).
        materialize: optional async callable that recreates this state in an
            ``Environment`` for smoke validation. If None, the demo is prompt-only.
    """

    should_pass: bool
    describe: str
    materialize: Materializer | None = None


@dataclass
class SynthesisResult:
    """Outcome of :func:`synthesize_verifier`.

    Attributes:
        checks: the parsed, registry-valid check specs (empty on parse failure).
        raw: the raw model output (for debugging / telemetry).
        validated: True if :func:`validate_against_demos` was run.
        validation: per-demo agreement report (when validated).
    """

    checks: list[dict[str, Any]]
    raw: str
    validated: bool = False
    validation: dict[str, Any] = field(default_factory=dict)


def build_prompt(
    task: str,
    success_descriptions: list[str],
    failure_descriptions: list[str],
) -> str:
    succ = "\n".join(f"  + {d}" for d in success_descriptions) or "  (none)"
    fail = "\n".join(f"  - {d}" for d in failure_descriptions) or "  (none)"
    return (
        f"{VERIFIER_MENU}\n\n"
        f"TASK:\n{task}\n\n"
        f"SUCCESS EXAMPLES (your checks must PASS on these):\n{succ}\n\n"
        f"FAILURE EXAMPLES (your checks must FAIL on these):\n{fail}\n\n"
        f"JSON array of checks:"
    )


def parse_checks(text: str) -> list[dict[str, Any]]:
    """Robustly pull a JSON array of check specs out of ``text``.

    Accepts raw JSON, markdown-fenced JSON, or JSON trailing prose. Drops any
    parsed item that is not a dict with a known verifier ``type``.
    """
    if not text:
        return []
    # Strip markdown code fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fence.group(1) if fence else text
    # Prefer the outermost bracketed span.
    start = candidate.find("[")
    end = candidate.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    chunk = candidate[start : end + 1]
    try:
        items = json.loads(chunk)
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ctype = item.get("type") or item.get("verifier")
        if not isinstance(ctype, str):
            continue
        # Normalize to "type" and keep only registry-known kinds.
        normalized = dict(item)
        normalized["type"] = ctype.lower()
        try:
            from syll.sandbox.verifiers import get_verifier

            get_verifier(ctype)  # raises KeyError if unknown
        except KeyError:
            continue
        cleaned.append(normalized)
    return cleaned


async def synthesize_verifier(
    task: str,
    demos: list[Demo],
    provider: LLMProvider,
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> SynthesisResult:
    """Ask ``provider`` to write a verifier spec for ``task`` given ``demos``."""
    success = [d.describe for d in demos if d.should_pass]
    failure = [d.describe for d in demos if not d.should_pass]
    prompt = build_prompt(task, success, failure)
    resp = await provider.chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = resp.content or ""
    return SynthesisResult(checks=parse_checks(text), raw=text)


async def _maybe_reset(env: Environment) -> None:
    """Best-effort full reset before each demo.

    Real sandbox backends (Docker/VM) reset to baseline; ``LocalEnvironment``
    refuses (no snapshot), in which case we proceed and rely on the demo's own
    materializer to own the state — local tests use self-cleaning materializers.
    """
    try:
        await env.reset("full")
    except NotImplementedError:
        pass


async def validate_against_demos(
    checks: list[dict[str, Any]],
    env: Environment,
    demos: list[Demo],
) -> dict[str, Any]:
    """Smoke-test ``checks`` against each materializable demo.

    For every demo with a ``materialize`` callable: reset the env to baseline
    (best-effort), recreate the demo state, run ``checks`` via ``run_checks``,
    and record whether the verdict agreed with ``should_pass``. Returns a report
    with per-demo rows and an ``agreement`` fraction in [0, 1].
    """
    rows: list[dict[str, Any]] = []
    agreed = 0
    ran = 0
    for i, demo in enumerate(demos):
        if demo.materialize is None:
            continue
        ran += 1
        await _maybe_reset(env)
        await demo.materialize(env)
        res = await run_checks(env, checks)
        ok = res.passed == demo.should_pass
        agreed += int(ok)
        rows.append(
            {
                "index": i,
                "should_pass": demo.should_pass,
                "got_pass": res.passed,
                "agreed": ok,
                "detail": res.detail,
            }
        )
    return {
        "rows": rows,
        "ran": ran,
        "agreed": agreed,
        "agreement": (agreed / ran) if ran else 1.0,
    }


async def synthesize_and_validate(
    task: str,
    demos: list[Demo],
    env: Environment,
    provider: LLMProvider,
    model: str | None = None,
) -> SynthesisResult:
    """Synthesize, then smoke-validate against the materializable demos."""
    result = await synthesize_verifier(task, demos, provider, model=model)
    if not result.checks:
        result.validated = False
        result.validation = {"ran": 0, "agreed": 0, "agreement": 0.0, "note": "no checks parsed"}
        return result
    report = await validate_against_demos(result.checks, env, demos)
    result.validated = True
    result.validation = report
    return result


__all__ = [
    "Demo",
    "SynthesisResult",
    "VERIFIER_MENU",
    "build_prompt",
    "parse_checks",
    "synthesize_verifier",
    "validate_against_demos",
    "synthesize_and_validate",
]
