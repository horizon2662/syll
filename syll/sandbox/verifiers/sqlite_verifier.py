"""SQLite verifier: run a query and compare a scalar result.

Useful for app outputs that land in a profile/state DB (browser history,
LibreOffice recent-docs, app settings). Dependency-free: uses python3's
stdlib ``sqlite3`` over ``env.exec``.

Spec keys:
    db_path:  str                path to the .sqlite/.db file
    query:    str                SQL returning a single scalar value
    equals:   Any                expected scalar (string-compared)
    contains: str                alternative — result substring must contain this
"""

from __future__ import annotations

from typing import Any

from syll.sandbox.environment import Environment

from .base import Verifier, VerifierResult


class SQLiteVerifier(Verifier):
    name = "sqlite"

    @staticmethod
    def _quote(s: str) -> str:
        # See FileVerifier._quote: double-quote wrapping is portable across the
        # Linux sandbox (sh) and Windows dev shells (cmd.exe).
        return '"' + s.replace('"', '\\"') + '"'

    async def check(self, env: Environment, spec: dict[str, Any]) -> VerifierResult:
        db_path = spec.get("db_path")
        query = spec.get("query")
        if not db_path or not query:
            return VerifierResult.fail(
                "sqlite verifier requires 'db_path' and 'query'",
                spec=spec,
            )
        # Single-line script with NO inner single quotes (uses None + double
        # quotes) so ``_quote`` wraps it cleanly. Query/db errors surface as a
        # nonzero exit code we detect below.
        script = (
            "import sqlite3,sys;"
            "con=sqlite3.connect(sys.argv[1]);"
            "row=con.execute(sys.argv[2]).fetchone();"
            "con.close();"
            "print(None if (row is None or row[0] is None) else row[0])"
        )
        try:
            res = await env.exec(
                f"python3 -c {self._quote(script)} {self._quote(str(db_path))} {self._quote(query)}"
            )
        except Exception as exc:
            return VerifierResult.fail(f"sqlite exec failed: {exc}")
        if res.returncode != 0:
            return VerifierResult.fail(
                f"sqlite query errored: {res.stderr.strip()}",
                db_path=db_path,
            )
        actual = res.stdout.strip()
        if "equals" in spec:
            expected = str(spec["equals"])
            if actual == expected:
                return VerifierResult.pass_(
                    f"sqlite match: {actual!r}", actual=actual
                )
            return VerifierResult.fail(
                f"sqlite mismatch: got {actual!r}, want {expected!r}",
                actual=actual,
                expected=expected,
            )
        if "contains" in spec:
            needle = str(spec["contains"])
            if needle in actual:
                return VerifierResult.pass_(
                    f"sqlite contains {needle!r}: {actual!r}", actual=actual
                )
            return VerifierResult.fail(
                f"sqlite missing {needle!r} in {actual!r}",
                actual=actual,
                needle=needle,
            )
        # No comparator: treat a non-error query as a passed gate.
        return VerifierResult.pass_(
            f"sqlite query ran: {actual!r}", actual=actual
        )
