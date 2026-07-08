"""Online evolution loop (Phase 3): ENPIRE-E × Aspire.

Closes the self-improvement loop. When a task fails, a coding agent reads the
failure trace, proposes one or more code-as-policy skill candidates (Aspire
diagnose → patch), each is run through the re-execution gate in its own sandbox
(validate), and the ones that pass are installed into the global
:class:`~syll.agent.longhorizon.code_skill.CodeSkillLibrary` so they compound
for future tasks (distill).

Two guards keep this from backfiring:

1. **verifier-ceiling gate** (:meth:`Evolver.should_evolve`). A "validated"
   skill is only as trustworthy as the verifier. β is the verifier false-success
   rate; the effective skill quality is ``p_eff = p/(p+(1-p)β)`` and the
   horizon ``H ≈ ln2/[(1-p)β]`` (see ``paper-theory-verifier-ceiling``). When β
   is high, a passed gate is likely a false success, so evolving would install a
   broken skill. We therefore refuse to evolve unless β is below a threshold.
2. **Evolutionary search**. The proposer emits N diverse candidates; each is
   validated in parallel (Aspire). A single bad draft doesn't block a good one.

Phase 3 writes skills; it does NOT yet auto-trigger from the runner's failure
path by default — wire :meth:`Evolver.evolve` into the runner's step-failure
handler (after replan) once the evolution behaviour is reviewed. The evolution
*generation* of unverified code is meant to run against the Phase 0 Docker
sandbox (per the Phase 2 decision), while installed skills may run on the live
env like any validated skill.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from syll.agent.longhorizon.code_skill import CodeSkill, CodeSkillLibrary
from syll.providers.base import LLMProvider
from syll.sandbox.environment import Environment

EnvironmentFactory = Callable[[], Environment]


# -----------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class FailureCase:
    """A failed task offered to the evolver."""

    instruction: str
    diagnosis: str  # root cause (from GuiAttemptLedger / events.jsonl)
    checks: list[dict[str, Any]] = field(default_factory=list)  # verifier the skill must pass
    setup: Callable[[Environment], Awaitable[None]] | None = None  # materialize start state
    trace_summary: str = ""  # optional actions.jsonl excerpt for the proposer


@dataclass
class EvolutionReport:
    failure: FailureCase
    proposed: int = 0
    validated: list[CodeSkill] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    installed_names: list[str] = field(default_factory=list)
    evolved: bool = False
    reason: str = ""


# -----------------------------------------------------------------------
# Proposer prompt + parser
# ---------------------------------------------------------------------------

_ALLOWED_MODULES = ["asyncio", "json", "re", "math", "base64", "hashlib"]

_PROPOSER_HEADER = """\
You write code-as-policy skills that fix a FAILED computer-use task. Each skill
is Python source defining `async def run(env, **kw) -> dict` and drives the
`env` object (methods: exec, read_file/write_file/edit_file/list_dir, \
screenshot, click/type/scroll/... ).

Restricted namespace — you may NOT import anything. These modules are already
available as globals: asyncio, json, re, math, base64, hashlib. No `open`, no \
`os`, no `__import__`. For shell/file access use the `env` methods.

Emit each candidate skill as a fenced ```python block whose first two lines are:
    # name: <short-kebab-name>
    # desc: <one-line when-to-use>
then `async def run(env, **kw): ...`. Propose {N} DIVERSE candidates \
(different strategies). A skill succeeds only if it passes the checks below in a
clean re-run, so be specific and avoid side effects beyond the task."""


def build_proposal_prompt(failure: FailureCase, n: int) -> str:
    checks = "\n".join(f"  - {c}" for c in failure.checks) or "  (none — make the skill produce the task's deliverable)"
    trace = f"\n\nFAILURE TRACE (recent steps):\n{failure.trace_summary}" if failure.trace_summary else ""
    return (
        _PROPOSER_HEADER.replace("{N}", str(n))
        + f"\n\nTASK:\n{failure.instruction}\n"
        + f"\nDIAGNOSIS (why it failed):\n{failure.diagnosis}{trace}"
        + f"\n\nCHECKS the skill must pass (re-execution gate):\n{checks}\n"
    )


_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_candidates(text: str, fallback_desc: str = "") -> list[CodeSkill]:
    """Pull candidate skills out of the proposer's fenced python blocks."""
    out: list[CodeSkill] = []
    for i, m in enumerate(_BLOCK_RE.finditer(text or "")):
        body = m.group(1).strip()
        if "async def run" not in body:
            continue
        name, desc, code = _split_header(body, i, fallback_desc)
        out.append(CodeSkill(name=name, description=desc, code=code, checks=[]))
    return out


def _split_header(body: str, idx: int, fallback_desc: str) -> tuple[str, str, str]:
    lines = body.splitlines()
    name, desc = "", fallback_desc or ""
    code_start = 0
    for j, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("# name:"):
            name = s.split(":", 1)[1].strip()
        elif s.startswith("# desc:"):
            desc = s.split(":", 1)[1].strip()
        elif ln and not s.startswith("#"):
            code_start = j
            break
    else:
        code_start = len(lines)
    code = "\n".join(lines[code_start:]).strip() or body
    if not name:
        slug = re.sub(r"[^a-z0-9]+", "-", (fallback_desc or f"skill {idx}").lower()).strip("-")[:32] or f"skill-{idx}"
        name = f"{slug}-{idx}"
    return name, desc, code


# -----------------------------------------------------------------------
# Evolver
# ---------------------------------------------------------------------------


class Evolver:
    """Aspire-style online skill evolution, gated by the verifier-ceiling law.

    Args:
        library: the global CodeSkillLibrary validated skills compound into.
        env_factory: returns a fresh sandbox Environment per candidate
            (validation runs in parallel; each candidate gets its own env).
        provider / model: the coding agent that proposes patches.
        beta_threshold: refuse to evolve when the verifier's false-success rate
            β exceeds this (validated skills would be untrustworthy).
        variants: number of diverse candidates to propose + validate per failure.
        git_commit: if True, git-commit the library after a successful install
            (ENPIRE-fleet-style history; the library root must be / become a repo).
    """

    def __init__(
        self,
        library: CodeSkillLibrary,
        env_factory: EnvironmentFactory,
        provider: LLMProvider,
        model: str | None = None,
        beta_threshold: float = 0.3,
        variants: int = 3,
        temperature: float = 0.7,
        git_commit: bool = False,
    ) -> None:
        self.library = library
        self._env_factory = env_factory
        self._provider = provider
        self.model = model
        self.beta_threshold = beta_threshold
        self.variants = max(1, variants)
        self.temperature = temperature
        self.git_commit = git_commit

    # -- verifier-ceiling gate ------------------------------------------------

    def should_evolve(self, failure: FailureCase, beta: float | None) -> bool:
        """Only evolve when the verifier is trustworthy enough.

        β is the verifier's false-success rate. A validated skill's effective
        quality is ``p_eff = p/(p+(1-p)β)``; with β high, a passed gate is
        likely a false success, so installing a "validated" skill would pollute
        the library. We therefore refuse when β exceeds the threshold. β=None
        (no estimate yet) is allowed so the loop can bootstrap.
        """
        if beta is None:
            return True
        return beta <= self.beta_threshold

    # -- the loop -------------------------------------------------------------

    async def evolve(self, failure: FailureCase, beta: float | None = None) -> EvolutionReport:
        report = EvolutionReport(failure=failure)
        if not self.should_evolve(failure, beta):
            report.reason = (
                f"gated: β={beta} > threshold {self.beta_threshold} — verifier too "
                "unreliable to validate a new skill"
            )
            return report

        candidates = await self._propose(failure)
        report.proposed = len(candidates)
        if not candidates:
            report.reason = "no candidate skills parsed from proposer"
            return report

        results = await asyncio.gather(
            *(self._validate_one(c, failure) for c in candidates)
        )
        for skill, install in results:
            if install.get("installed"):
                report.validated.append(skill)
                report.installed_names.append(skill.name)
            else:
                report.rejected.append(
                    {
                        "name": skill.name,
                        "detail": (install.get("verdict") or {}).get("detail", ""),
                        "skill_error": (install.get("skill_result") or {}).get("error"),
                    }
                )
        report.evolved = bool(report.validated)
        if report.evolved and self.git_commit:
            _git_commit(self.library.root, f"evolve: {failure.instruction[:60]} ({len(report.validated)} skill(s))")
        report.reason = "evolved" if report.evolved else "all candidates rejected by the gate"
        return report

    async def _propose(self, failure: FailureCase) -> list[CodeSkill]:
        prompt = build_proposal_prompt(failure, self.variants)
        resp = await self._provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
        )
        return parse_candidates(resp.content or "", fallback_desc=failure.instruction)

    async def _validate_one(
        self, skill: CodeSkill, failure: FailureCase
    ) -> tuple[CodeSkill, dict[str, Any]]:
        env = self._env_factory()
        install = await self.library.install_validated(
            skill, env, setup=failure.setup, checks=failure.checks
        )
        return skill, install


# -----------------------------------------------------------------------
# Optional Git history (ENPIRE-fleet pattern)
# ---------------------------------------------------------------------------


def _git_commit(root: Path, message: str) -> None:
    """Best-effort git commit of the library; silent no-op if git/repo absent."""
    try:
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=False, capture_output=True)
        subprocess.run(
            ["git", "-C", str(root), "commit", "-m", message],
            check=False, capture_output=True,
        )
    except Exception:
        pass


__all__ = [
    "FailureCase",
    "EvolutionReport",
    "Evolver",
    "build_proposal_prompt",
    "parse_candidates",
]
