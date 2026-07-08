"""File-based verifier: existence, absence, and sha256 content checks.

Upgrades the runner's ``_verify_artifacts`` file-existence oracle
(runner.py ``_apply_oracle``) to a declarative spec. The deterministic
baseline verifier — no LLM involved.

Spec keys:
    require_exists:  list[str]              paths that must exist
    require_absent:  list[str]              paths that must NOT exist
    sha256:          dict[str, str]         path -> expected hex digest
"""

from __future__ import annotations

from typing import Any

from syll.sandbox.environment import Environment

from .base import Verifier, VerifierResult


class FileVerifier(Verifier):
    name = "file"

    async def check(self, env: Environment, spec: dict[str, Any]) -> VerifierResult:
        require_exists = spec.get("require_exists") or []
        require_absent = spec.get("require_absent") or []
        sha256 = spec.get("sha256") or {}

        missing: list[str] = []
        present_forbidden: list[str] = []
        hash_mismatch: list[str] = []

        for path in require_exists:
            if not await self._exists(env, path):
                missing.append(path)
        for path in require_absent:
            if await self._exists(env, path):
                present_forbidden.append(path)
        for path, expected in sha256.items():
            actual = await self._sha256(env, path)
            if actual is None:
                missing.append(path)
            elif actual.lower() != str(expected).lower():
                hash_mismatch.append(path)

        if missing or present_forbidden or hash_mismatch:
            parts = []
            if missing:
                parts.append(f"missing={missing}")
            if present_forbidden:
                parts.append(f"forbidden-but-present={present_forbidden}")
            if hash_mismatch:
                parts.append(f"hash-mismatch={hash_mismatch}")
            return VerifierResult.fail(
                "; ".join(parts),
                missing=missing,
                present_forbidden=present_forbidden,
                hash_mismatch=hash_mismatch,
            )

        n = len(require_exists) + len(require_absent) + len(sha256)
        return VerifierResult.pass_(f"all {n} file checks passed", checked=n)

    @staticmethod
    def _quote(path: str) -> str:
        # Double-quote wrapping: portable across the Linux sandbox (sh) and
        # Windows dev shells (cmd.exe, where single-quote wrapping silently
        # fails for ``python3 -c``). Embedded double quotes escaped as \" per
        # the CPython argv convention on both. Avoid ``$`` / backticks in specs
        # (sh would expand them inside double quotes).
        return '"' + path.replace('"', '\\"') + '"'

    async def _exists(self, env: Environment, path: str) -> bool:
        try:
            res = await env.exec(f"test -e {self._quote(path)} && echo y || echo n")
            return res.stdout.strip().endswith("y")
        except Exception:
            return False

    async def _sha256(self, env: Environment, path: str) -> str | None:
        # python3 stdlib one-liner. Single-line + double-quoted "rb" so the
        # script has NO inner single quotes — ``_quote`` then wraps it in a
        # clean shell single-quoted arg (portable across the Linux sandbox and
        # macOS/Windows dev shells; no sha256sum-vs-shasum dependency).
        script = "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())"
        try:
            res = await env.exec(
                f"python3 -c {self._quote(script)} {self._quote(path)}"
            )
            if res.returncode != 0:
                return None
            digest = res.stdout.strip()
            return digest[:64] or None
        except Exception:
            return None
