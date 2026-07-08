"""Tests for the online evolution loop (Phase 3).

Uses a fake provider that returns canned candidate skill blocks and a fresh
LocalEnvironment per candidate. No Docker, no real LLM. Covers the
verifier-ceiling β gate, the candidate parser, and the
diagnose→patch→validate→distill loop (working / broken / mixed candidates /
gated).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from syll.agent.longhorizon.code_skill import CodeSkillLibrary
from syll.agent.longhorizon.evolution import (
    Evolver,
    FailureCase,
    build_proposal_prompt,
    parse_candidates,
)
from syll.providers.base import LLMProvider, LLMResponse
from syll.sandbox.environment import LocalEnvironment


class _FakeProvider(LLMProvider):
    def __init__(self, content: str) -> None:
        super().__init__()
        self._content = content
        self.calls = 0

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        self.calls += 1
        return LLMResponse(content=self._content)

    def get_default_model(self) -> str:
        return "fake"


@pytest.fixture
def library(tmp_path: Path) -> CodeSkillLibrary:
    return CodeSkillLibrary(tmp_path / "code_skills")


def _factory():
    return LocalEnvironment(workspace_root=tempfile.mkdtemp())


_GOOD = """\
```python
# name: make-done
# desc: writes the done marker file
async def run(env, **kw):
    await env.write_file("out/done.txt", "ok")
    return {"ok": True}
```
"""

_BROKEN = """\
```python
# name: broken
# desc: raises immediately
async def run(env, **kw):
    raise RuntimeError("cannot")
```
"""

_MIXED = _GOOD + "\n" + _BROKEN


def _failure() -> FailureCase:
    async def setup(e):
        await e.exec("rm -rf out && mkdir -p out")

    return FailureCase(
        instruction="write out/done.txt containing ok",
        diagnosis="agent clicked the wrong button; file never created",
        checks=[{"type": "file", "require_exists": ["out/done.txt"]}],
        setup=setup,
    )


# -----------------------------------------------------------------------
# β gate
# -----------------------------------------------------------------------


def test_should_evolve_bootstrap_allows_none_beta(library: CodeSkillLibrary):
    ev = Evolver(library, _factory, _FakeProvider(""), beta_threshold=0.3)
    assert ev.should_evolve(_failure(), beta=None) is True


def test_should_evolve_blocks_high_beta(library: CodeSkillLibrary):
    ev = Evolver(library, _factory, _FakeProvider(""), beta_threshold=0.3)
    assert ev.should_evolve(_failure(), beta=0.1) is True
    assert ev.should_evolve(_failure(), beta=0.5) is False


# -----------------------------------------------------------------------
# parser
# -----------------------------------------------------------------------


def test_parse_candidates_extracts_blocks():
    skills = parse_candidates(_MIXED, fallback_desc="task")
    assert len(skills) == 2
    assert skills[0].name == "make-done"
    assert "done marker" in skills[0].description
    assert "async def run" in skills[0].code
    assert skills[1].name == "broken"


def test_parse_candidates_empty_on_garbage():
    assert parse_candidates("nothing useful here") == []
    assert parse_candidates("") == []


def test_build_proposal_prompt_carries_task_and_checks():
    p = build_proposal_prompt(_failure(), 3)
    assert "write out/done.txt" in p
    assert "out/done.txt" in p  # a check
    assert "DIVERSE" in p  # the variants instruction


# -----------------------------------------------------------------------
# the loop
# -----------------------------------------------------------------------


@pytest.mark.anyio
async def test_evolve_installs_working_candidate(library: CodeSkillLibrary):
    ev = Evolver(library, _factory, _FakeProvider(_GOOD), variants=1)
    report = await ev.evolve(_failure(), beta=0.0)
    assert report.evolved is True
    assert "make-done" in report.installed_names
    assert library.load("make-done") is not None
    assert library.load("make-done").validated is True


@pytest.mark.anyio
async def test_evolve_rejects_broken_candidate(library: CodeSkillLibrary):
    ev = Evolver(library, _factory, _FakeProvider(_BROKEN), variants=1)
    report = await ev.evolve(_failure(), beta=0.0)
    assert report.evolved is False
    assert library.load("broken") is None  # never installed
    assert report.rejected and "cannot" in report.rejected[0]["skill_error"]


@pytest.mark.anyio
async def test_evolve_mixed_keeps_only_passing(library: CodeSkillLibrary):
    ev = Evolver(library, _factory, _FakeProvider(_MIXED), variants=2)
    report = await ev.evolve(_failure(), beta=0.0)
    assert report.evolved is True
    assert report.installed_names == ["make-done"]
    assert len(report.rejected) == 1
    assert library.load("make-done") is not None
    assert library.load("broken") is None


@pytest.mark.anyio
async def test_evolve_gated_does_not_call_provider(library: CodeSkillLibrary):
    provider = _FakeProvider(_GOOD)
    ev = Evolver(library, _factory, provider, beta_threshold=0.3)
    report = await ev.evolve(_failure(), beta=0.5)
    assert report.evolved is False
    assert "β=0.5" in report.reason or "gated" in report.reason
    assert provider.calls == 0  # gate short-circuits before proposing
    assert library.list() == []
