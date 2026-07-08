"""Tests for code-as-policy skills (Phase 2).

Covers the restricted-namespace execution (no __import__; provided modules
usable), the re-execution gate, the global library round-trip + JIT retrieval
(validated-only), install_validated's accept/reject behaviour, and the
run_code_skill tool's refusal of unvalidated skills. No Docker, no real LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from syll.agent.longhorizon.code_skill import (
    CodeSkill,
    CodeSkillLibrary,
    execute_skill,
    validate_skill,
)
from syll.agent.tools.code_skill_tool import RunCodeSkillTool
from syll.sandbox.environment import LocalEnvironment


@pytest.fixture
def env(tmp_path: Path) -> LocalEnvironment:
    return LocalEnvironment(workspace_root=tmp_path)


def _skill(name: str, body: str, checks: list[dict] | None = None) -> CodeSkill:
    code = "async def run(env, **kw):\n" + body
    return CodeSkill(name=name, description=name, code=code, checks=checks or [])


# -----------------------------------------------------------------------
# execute_skill — restricted namespace
# -----------------------------------------------------------------------


@pytest.mark.anyio
async def test_execute_runs_skill_that_writes_file(env: LocalEnvironment):
    skill = _skill(
        "seed", "    await env.write_file('out/a.txt', 'hello')\n    return {'ok': True}\n"
    )
    res = await execute_skill(skill, env)
    assert res["ok"] is True
    assert (await env.read_file("out/a.txt")) == "hello"


@pytest.mark.anyio
async def test_execute_crash_is_caught(env: LocalEnvironment):
    skill = _skill("boom", "    raise ValueError('nope')\n")
    res = await execute_skill(skill, env)
    assert res["ok"] is False
    assert "nope" in res["error"]


@pytest.mark.anyio
async def test_restricted_namespace_blocks_import(env: LocalEnvironment):
    # __import__ is removed → `import os` must fail at load time.
    skill = CodeSkill(
        name="evil",
        description="evil",
        code="import os\nasync def run(env, **kw):\n    os.system('echo pwned')\n    return {'ok': True}\n",
    )
    res = await execute_skill(skill, env)
    assert res["ok"] is False
    assert "load failed" in res["error"]


@pytest.mark.anyio
async def test_provided_module_is_usable_without_import(env: LocalEnvironment):
    skill = CodeSkill(
        name="jsonr",
        description="jsonr",
        code="async def run(env, **kw):\n    await env.write_file('out/j.txt', json.dumps({'a': 1}))\n    return {'ok': True}\n",
    )
    res = await execute_skill(skill, env)
    assert res["ok"] is True
    assert (await env.read_file("out/j.txt")) == '{"a": 1}'


# -----------------------------------------------------------------------
# validate_skill — the re-execution gate
# -----------------------------------------------------------------------


@pytest.mark.anyio
async def test_validate_passes_when_skill_meets_checks(env: LocalEnvironment):
    skill = _skill(
        "make",
        "    await env.write_file('out/done.txt', 'ok')\n    return {'ok': True}\n",
        checks=[{"type": "file", "require_exists": ["out/done.txt"]}],
    )

    async def setup(e):
        await e.exec("rm -rf out && mkdir -p out")

    report = await validate_skill(skill, env, setup=setup)
    assert report["passed"] is True


@pytest.mark.anyio
async def test_validate_fails_when_skill_does_not_meet_checks(env: LocalEnvironment):
    skill = _skill(
        "nope",
        "    return {'ok': True}\n",  # does nothing
        checks=[{"type": "file", "require_exists": ["out/missing.txt"]}],
    )

    async def setup(e):
        await e.exec("rm -rf out && mkdir -p out")

    report = await validate_skill(skill, env, setup=setup)
    assert report["passed"] is False


# -----------------------------------------------------------------------
# CodeSkillLibrary — global, JIT retrieval, install gate
# -----------------------------------------------------------------------


@pytest.fixture
def library(tmp_path: Path) -> CodeSkillLibrary:
    return CodeSkillLibrary(tmp_path / "code_skills")


@pytest.mark.anyio
async def test_library_save_load_roundtrip(library: CodeSkillLibrary):
    skill = _skill("s1", "    return {'ok': True}\n")
    skill.description = "open chrome to a url"
    skill.validated = True
    library.save(skill)
    loaded = library.load("s1")
    assert loaded is not None
    assert loaded.description == "open chrome to a url"
    assert loaded.validated is True
    assert loaded.code == skill.code
    assert [s.name for s in library.list()] == ["s1"]
    assert library.remove("s1") is True
    assert library.load("s1") is None


@pytest.mark.anyio
async def test_get_relevant_returns_only_validated(library: CodeSkillLibrary):
    good = _skill("export_csv", "    return {'ok': True}\n")
    good.description = "export the active sheet to CSV"
    good.validated = True
    bad = _skill("draft", "    return {'ok': True}\n")
    bad.description = "export sheet draft"
    bad.validated = False
    library.save(good)
    library.save(bad)
    snippet = library.get_relevant("export sheet CSV")
    assert "export_csv" in snippet
    assert "draft" not in snippet


@pytest.mark.anyio
async def test_install_validated_accepts_working_skill(library: CodeSkillLibrary, env: LocalEnvironment):
    skill = _skill(
        "installable",
        "    await env.write_file('out/done.txt', 'ok')\n    return {'ok': True}\n",
        checks=[{"type": "file", "require_exists": ["out/done.txt"]}],
    )

    async def setup(e):
        await e.exec("rm -rf out && mkdir -p out")

    report = await library.install_validated(skill, env, setup=setup)
    assert report["installed"] is True
    loaded = library.load("installable")
    assert loaded is not None and loaded.validated is True
    assert loaded.successes == 1


@pytest.mark.anyio
async def test_install_validated_rejects_broken_skill(library: CodeSkillLibrary, env: LocalEnvironment):
    skill = _skill(
        "broken",
        "    raise RuntimeError('cannot')\n",
        checks=[{"type": "file", "require_exists": ["out/x.txt"]}],
    )

    async def setup(e):
        await e.exec("rm -rf out && mkdir -p out")

    report = await library.install_validated(skill, env, setup=setup)
    assert report["installed"] is False
    assert library.load("broken") is None  # never entered the library


# -----------------------------------------------------------------------
# RunCodeSkillTool
# -----------------------------------------------------------------------


@pytest.mark.anyio
async def test_tool_runs_validated_skill(library: CodeSkillLibrary, env: LocalEnvironment):
    skill = _skill(
        "click_ok",
        "    await env.write_file('out/x.txt', 'done')\n    return {'ok': True, 'detail': 'wrote x'}\n",
    )
    skill.validated = True
    library.save(skill)
    tool = RunCodeSkillTool(library, environment=env)
    out = await tool.execute(name="click_ok")
    assert "wrote x" in out
    assert library.load("click_ok").runs == 1


@pytest.mark.anyio
async def test_tool_refuses_unvalidated_skill(library: CodeSkillLibrary, env: LocalEnvironment):
    skill = _skill("unval", "    return {'ok': True}\n")
    skill.validated = False
    library.save(skill)
    tool = RunCodeSkillTool(library, environment=env)
    out = await tool.execute(name="unval")
    assert "NOT validated" in out


@pytest.mark.anyio
async def test_tool_unknown_skill_lists_available(library: CodeSkillLibrary, env: LocalEnvironment):
    tool = RunCodeSkillTool(library, environment=env)
    out = await tool.execute(name="ghost")
    assert "no code_skill" in out and "available" in out
