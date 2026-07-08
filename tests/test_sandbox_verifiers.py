"""Unit tests for the state-grounded verifier framework.

Exercises FileVerifier / SQLiteVerifier / run_checks against a LocalEnvironment
without a Docker container, plus the lifecycle contract (LocalEnvironment
refuses snapshot/reset; safety_check probes the workspace). These pin the
``passed``/``score`` gate-and-score behaviour that the rollout harness (Phase 1)
and the β-matrix oracle upgrade depend on.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from syll.sandbox.environment import LocalEnvironment
from syll.sandbox.verifiers import (
    A11yVerifier,
    CDPVerifier,
    FileVerifier,
    SQLiteVerifier,
    get_verifier,
    run_checks,
)


@pytest.fixture
def env(tmp_path: Path) -> LocalEnvironment:
    return LocalEnvironment(workspace_root=tmp_path)


@pytest.mark.anyio
async def test_file_verifier_exists_pass(env: LocalEnvironment):
    await env.write_file("out/x.txt", "hello")
    res = await FileVerifier().check(
        env, {"require_exists": ["out/x.txt"], "require_absent": ["out/y.txt"]}
    )
    assert res.passed is True
    assert res.score == 1.0


@pytest.mark.anyio
async def test_file_verifier_missing_fails(env: LocalEnvironment):
    res = await FileVerifier().check(env, {"require_exists": ["out/missing.txt"]})
    assert res.passed is False
    assert res.score == 0.0
    assert "out/missing.txt" in res.evidence["missing"]


@pytest.mark.anyio
async def test_file_verifier_absent_violation_fails(env: LocalEnvironment):
    await env.write_file("forbidden.txt", "x")
    res = await FileVerifier().check(env, {"require_absent": ["forbidden.txt"]})
    assert res.passed is False
    assert "forbidden.txt" in res.evidence["present_forbidden"]


@pytest.mark.anyio
async def test_file_verifier_sha256_pass_and_fail(env: LocalEnvironment):
    await env.write_file("a.txt", "hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    ok = await FileVerifier().check(env, {"sha256": {"a.txt": digest}})
    assert ok.passed is True
    bad = await FileVerifier().check(env, {"sha256": {"a.txt": "0" * 64}})
    assert bad.passed is False
    assert "a.txt" in bad.evidence["hash_mismatch"]


@pytest.mark.anyio
async def test_sqlite_verifier_equals_and_contains(env: LocalEnvironment):
    await env.write_file(
        "mk.py",
        "import sqlite3;c=sqlite3.connect('d.db');"
        "c.execute(\"CREATE TABLE kv(k,v)\");"
        "c.execute(\"INSERT INTO kv VALUES('done','1')\");c.commit()",
    )
    assert (await env.exec("python3 mk.py")).returncode == 0

    ok = await SQLiteVerifier().check(
        env,
        {"db_path": "d.db", "query": "SELECT v FROM kv WHERE k='done'", "equals": "1"},
    )
    assert ok.passed is True

    bad = await SQLiteVerifier().check(
        env,
        {"db_path": "d.db", "query": "SELECT v FROM kv WHERE k='done'", "equals": "2"},
    )
    assert bad.passed is False

    contains = await SQLiteVerifier().check(
        env,
        {"db_path": "d.db", "query": "SELECT v FROM kv WHERE k='done'", "contains": "1"},
    )
    assert contains.passed is True


@pytest.mark.anyio
async def test_sqlite_verifier_requires_db_and_query(env: LocalEnvironment):
    res = await SQLiteVerifier().check(env, {"db_path": "d.db"})
    assert res.passed is False
    assert "db_path" in res.detail and "query" in res.detail


@pytest.mark.anyio
async def test_cdp_verifier_unreachable_fails(env: LocalEnvironment):
    # No browser/CDP endpoint in the test env → must fail cleanly, not crash.
    res = await CDPVerifier().check(
        env, {"cdp_endpoint": "http://localhost:1", "url_contains": "x"}
    )
    assert res.passed is False


@pytest.mark.anyio
async def test_a11y_verifier_does_not_crash_without_xdotool(env: LocalEnvironment):
    res = await A11yVerifier().check(env, {"window_title_contains": "anything"})
    # On a dev box without xdotool this returns a clean fail; on the XFCE
    # sandbox it would actually probe the active window. Either way: no crash.
    assert isinstance(res.passed, bool)


@pytest.mark.anyio
async def test_run_checks_gate_and_score_pass(env: LocalEnvironment):
    await env.write_file("a.txt", "hello")
    res = await run_checks(
        env,
        [
            {"type": "file", "require_exists": ["a.txt"]},
        ],
    )
    assert res.passed is True
    assert res.score == 1.0
    assert len(res.evidence["checks"]) == 1


@pytest.mark.anyio
async def test_run_checks_failed_gate_forces_zero(env: LocalEnvironment):
    await env.write_file("a.txt", "hello")
    res = await run_checks(
        env,
        [
            {"type": "file", "require_exists": ["a.txt"]},
            {"type": "file", "require_exists": ["missing.txt"]},
        ],
    )
    assert res.passed is False
    assert res.score == 0.0


@pytest.mark.anyio
async def test_run_checks_unknown_verifier_is_failed_gate(env: LocalEnvironment):
    res = await run_checks(env, [{"type": "nope", "x": 1}])
    assert res.passed is False
    assert "errored" in res.evidence["checks"][0]["detail"]


def test_get_verifier_registry():
    assert get_verifier("file").name == "file"
    assert get_verifier("FILE").name == "file"
    with pytest.raises(KeyError):
        get_verifier("does-not-exist")


@pytest.mark.anyio
async def test_local_environment_lifecycle_refuses_reset(env: LocalEnvironment):
    # LocalEnvironment runs on the real machine — snapshot/reset must refuse.
    for call in ("checkpoint", "restore", "reset"):
        with pytest.raises(NotImplementedError):
            await getattr(env, call)("x") if call != "reset" else await env.reset()


@pytest.mark.anyio
async def test_local_environment_safety_check(env: LocalEnvironment):
    res = await env.safety_check()
    assert res.ok is True
