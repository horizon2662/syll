"""Code-as-policy skills (Aspire, arXiv:2607.00272) — validated executable
Environment drivers that compound.

A :class:`CodeSkill` is a small Python script that drives an
:class:`~syll.sandbox.environment.Environment` to accomplish a reusable sub-goal
(open a browser to a URL + dismiss a dialog, export a sheet to CSV, …). It
defines ``async def run(env, **kw) -> dict``. Two rules make it safe + compounding:

1. **Re-execution gate** — a skill only enters the library after running it
   against a state-verifier (reset → setup → run → ``run_checks``) and passing.
   Unvalidated repairs are never installed (Aspire).
2. **Compounding** — once validated, future agents retrieve it as in-context
   guidance and invoke it via the ``run_code_skill`` tool instead of re-deriving
   the pixel sequence. Task N reuses task 1's repair.

Execution is in a restricted namespace (curated builtins + a few stdlib modules;
no ``__import__`` / ``open`` / ``exec``), so skill source can't import arbitrary
code. Shell/file/GUI access still flows through the ``env`` argument by design —
that is the point of code-as-policy (a richer action surface than pixel clicks).

The library is **global** (``workspace/code_skills/<name>/``) so validated
repairs compound across tasks. JIT keyword-overlap retrieval (capped) keeps the
library useful as it grows — solving Aspire limitation #4 (stale/redundant
entries) the same way SkillMemory does for prose.
"""

from __future__ import annotations

import asyncio
import base64
import builtins as _builtins
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from syll.sandbox.environment import Environment
from syll.sandbox.verifier_synthesis import _maybe_reset
from syll.sandbox.verifiers import run_checks

# -----------------------------------------------------------------------
# Restricted execution namespace
# -----------------------------------------------------------------------

_DANGEROUS = {
    "__import__",
    "open",
    "exec",
    "eval",
    "compile",
    "globals",
    "locals",
    "breakpoint",
    "input",
    "memoryview",
}
_SAFE_BUILTINS = {
    k: v for k, v in vars(_builtins).items() if k not in _DANGEROUS
}

# Modules a skill may reasonably want without giving it arbitrary import power.
_SAFE_MODULES = {
    "asyncio": asyncio,
    "json": json,
    "re": re,
    "math": math,
    "base64": base64,
    "hashlib": hashlib,
}


def _load_run(code: str) -> Callable[..., Awaitable[Any]]:
    """Compile skill source in a restricted namespace and return its ``run``."""
    if "async def run" not in code:
        raise ValueError("code_skill must define `async def run(env, **kw) -> dict`")
    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, **_SAFE_MODULES}
    exec(code, namespace)  # noqa: S102 — restricted namespace, no __import__
    run = namespace.get("run")
    if not callable(run):
        raise ValueError("code_skill `run` is missing or not callable")
    return run


# -----------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------


@dataclass
class CodeSkill:
    """One validated (or candidate) code-as-policy skill."""

    name: str
    description: str  # when to use it (drives JIT retrieval)
    code: str  # `async def run(env, **kw) -> dict` source
    checks: list[dict[str, Any]] = field(default_factory=list)  # verifier it must pass
    validated: bool = False
    runs: int = 0
    successes: int = 0
    updated_at: str = ""

    def to_meta(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "checks": self.checks,
            "validated": self.validated,
            "runs": self.runs,
            "successes": self.successes,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_meta(name: str, meta: dict[str, Any], code: str) -> "CodeSkill":
        return CodeSkill(
            name=name,
            description=meta.get("description", ""),
            code=code,
            checks=meta.get("checks", []),
            validated=bool(meta.get("validated", False)),
            runs=int(meta.get("runs", 0)),
            successes=int(meta.get("successes", 0)),
            updated_at=meta.get("updated_at", ""),
        )


# -----------------------------------------------------------------------
# Execution + validation
# ---------------------------------------------------------------------------


async def execute_skill(skill: CodeSkill, env: Environment, **kwargs: Any) -> dict[str, Any]:
    """Run ``skill`` against ``env`` in the restricted namespace.

    Returns the skill's own dict (normalised to contain ``ok``); crashes become
    ``{"ok": False, "error": ...}``. Does NOT touch the verifier.
    """
    try:
        run = _load_run(skill.code)
    except Exception as exc:  # malformed skill source
        return {"ok": False, "error": f"load failed: {exc!r}"}
    try:
        result = await run(env, **kwargs)
    except Exception as exc:  # runtime crash
        return {"ok": False, "error": f"run failed: {exc!r}"}
    if isinstance(result, dict):
        result.setdefault("ok", True)
        return result
    return {"ok": bool(result), "result": result}


async def validate_skill(
    skill: CodeSkill,
    env: Environment,
    setup: Callable[[Environment], Awaitable[None]] | None = None,
    checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The re-execution gate. reset → setup → run skill → run_checks.

    Returns ``{ok, passed, skill_result, verdict}``. Does NOT install. The
    caller decides from ``passed`` whether to keep the skill.
    """
    await _maybe_reset(env)
    if setup is not None:
        await setup(env)
    skill_result = await execute_skill(skill, env)
    verdict = await run_checks(env, checks if checks is not None else skill.checks)
    return {
        "ok": bool(skill_result.get("ok")),
        "passed": bool(verdict.passed) and bool(skill_result.get("ok")),
        "skill_result": skill_result,
        "verdict": {
            "passed": verdict.passed,
            "score": verdict.score,
            "detail": verdict.detail,
        },
    }


# -----------------------------------------------------------------------
# Library (global)
# ---------------------------------------------------------------------------


class CodeSkillLibrary:
    """Global, on-disk library of code-as-policy skills.

    Stored under ``root/<name>/{skill.py, meta.json}`` where ``root`` is typically
    ``workspace/code_skills/``. Global (not per-skill) so validated repairs
    compound across tasks (Aspire).
    """

    def __init__(self, root: Path | str) -> None:
        # Deliberately do NOT mkdir here — the library root may live under a
        # workspace path that doesn't exist yet (e.g. test fixtures using
        # /tmp). Directories are created lazily on save().
        self.root = Path(root)

    def path_for(self, name: str) -> Path:
        return self.root / name

    def load(self, name: str) -> CodeSkill | None:
        d = self.path_for(name)
        meta_f = d / "meta.json"
        code_f = d / "skill.py"
        if not meta_f.exists() or not code_f.exists():
            return None
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        code = code_f.read_text(encoding="utf-8")
        return CodeSkill.from_meta(name, meta, code)

    def list(self) -> list[CodeSkill]:
        if not self.root.exists():
            return []
        out: list[CodeSkill] = []
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            skill = self.load(d.name)
            if skill is not None:
                out.append(skill)
        return out

    def save(self, skill: CodeSkill) -> Path:
        skill.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        d = self.path_for(skill.name)
        d.mkdir(parents=True, exist_ok=True)
        (d / "skill.py").write_text(skill.code, encoding="utf-8")
        (d / "meta.json").write_text(
            json.dumps(skill.to_meta(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return d

    def remove(self, name: str) -> bool:
        d = self.path_for(name)
        if not d.exists():
            return False
        # Remove only our own files; never recurse-blindly into unknown trees.
        for f in (d / "skill.py", d / "meta.json"):
            f.unlink(missing_ok=True)
        try:
            d.rmdir()
        except OSError:
            pass
        return True

    def get_relevant(self, query: str, max_chars: int = 2000) -> str:
        """JIT keyword-overlap retrieval over name + description, capped."""
        skills = [s for s in self.list() if s.validated]
        if not skills:
            return ""
        q_terms = {w for w in re.split(r"\W+", (query or "").lower()) if len(w) > 2}
        if not q_terms:
            ranked = skills
        else:
            def score(s: CodeSkill) -> int:
                text = f"{s.name} {s.description}".lower()
                return sum(1 for t in q_terms if t in text)

            ranked = sorted(skills, key=score, reverse=True)
        lines = [
            f"- {s.name}: {s.description} (runs={s.runs}, ok={s.successes})"
            for s in ranked
        ]
        out: list[str] = []
        used = 0
        for ln in lines:
            if used + len(ln) + 1 > max_chars:
                break
            out.append(ln)
            used += len(ln) + 1
        if not out:
            return ""
        header = f"(from code-skill library, {len(out)} of {len(skills)} validated skills)\n"
        return header + "\n".join(out)

    async def install_validated(
        self,
        skill: CodeSkill,
        env: Environment,
        setup: Callable[[Environment], Awaitable[None]] | None = None,
        checks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run the re-execution gate; install only if it passes.

        On pass: mark ``validated``, bump stats, save. On fail: do NOT save
        (unvalidated repairs never enter the library); return the report so the
        caller can revise.
        """
        report = await validate_skill(skill, env, setup=setup, checks=checks)
        if report["passed"]:
            skill.validated = True
            skill.runs += 1
            skill.successes += 1
            self.save(skill)
        report["installed"] = bool(report["passed"])
        return report


__all__ = [
    "CodeSkill",
    "CodeSkillLibrary",
    "execute_skill",
    "validate_skill",
]
