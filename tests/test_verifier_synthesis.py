"""Tests for agent-synthesized verifiers.

Covers the parser's robustness (raw JSON, fenced markdown, garbage, unknown
type filtering), the synthesis LLM round-trip with a fake provider, and the
demo smoke-validation agreement gate. No real LLM call is made.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from syll.providers.base import LLMProvider, LLMResponse
from syll.sandbox.environment import LocalEnvironment
from syll.sandbox.verifier_synthesis import (
    Demo,
    parse_checks,
    synthesize_and_validate,
    synthesize_verifier,
    validate_against_demos,
)


class _FakeProvider(LLMProvider):
    def __init__(self, content: str) -> None:
        super().__init__()
        self._content = content
        self.captured_prompt = ""

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):
        self.captured_prompt = messages[0]["content"]
        return LLMResponse(content=self._content)

    def get_default_model(self) -> str:
        return "fake"


@pytest.fixture
def env(tmp_path: Path) -> LocalEnvironment:
    return LocalEnvironment(workspace_root=tmp_path)


def test_parse_raw_json_array():
    checks = parse_checks('[{"type":"file","require_exists":["a.txt"]}]')
    assert checks == [{"type": "file", "require_exists": ["a.txt"]}]


def test_parse_markdown_fenced():
    checks = parse_checks(
        "Here you go:\n```json\n[{\"type\": \"file\", \"require_exists\": [\"a.txt\"]}]\n```\n"
    )
    assert checks and checks[0]["type"] == "file"


def test_parse_drops_unknown_types():
    checks = parse_checks(
        '[{"type":"file","require_exists":["a.txt"]},'
        '{"type":"wishful-thinking","hope":true},'
        '{"no_type":1}]'
    )
    assert len(checks) == 1
    assert checks[0]["type"] == "file"


def test_parse_garbage_returns_empty():
    assert parse_checks("") == []
    assert parse_checks("the task is impossible") == []
    assert parse_checks("not [valid json at all") == []


@pytest.mark.anyio
async def test_synthesize_roundtrip_with_fake_provider(env: LocalEnvironment):
    provider = _FakeProvider('[{"type":"file","require_exists":["out/done.txt"]}]')
    result = await synthesize_verifier(
        "create out/done.txt", [Demo(True, "done.txt exists")], provider
    )
    assert result.checks == [{"type": "file", "require_exists": ["out/done.txt"]}]
    assert "create out/done.txt" in provider.captured_prompt
    assert "done.txt exists" in provider.captured_prompt


@pytest.mark.anyio
async def test_synthesize_empty_on_garbage(env: LocalEnvironment):
    provider = _FakeProvider("I cannot help with that.")
    result = await synthesize_verifier("task", [Demo(True, "x")], provider)
    assert result.checks == []


@pytest.mark.anyio
async def test_validate_full_agreement(env: LocalEnvironment):
    async def success(e):
        await e.exec("rm -rf out && mkdir -p out")
        await e.write_file("out/done.txt", "ok")

    async def failure(e):
        await e.exec("rm -rf out && mkdir -p out")  # no done.txt

    demos = [
        Demo(True, "done.txt exists with ok", success),
        Demo(False, "done.txt missing", failure),
    ]
    checks = [{"type": "file", "require_exists": ["out/done.txt"]}]
    report = await validate_against_demos(checks, env, demos)
    assert report["ran"] == 2
    assert report["agreed"] == 2
    assert report["agreement"] == 1.0


@pytest.mark.anyio
async def test_validate_disagreement_when_check_too_weak(env: LocalEnvironment):
    # A check that passes on both demos agrees on success but disagrees on failure.
    async def success(e):
        await e.exec("rm -rf out && mkdir -p out")
        await e.write_file("out/done.txt", "ok")

    async def failure(e):
        await e.exec("rm -rf out && mkdir -p out")
        await e.write_file("out/done.txt", "WRONG")  # exists but wrong content

    demos = [Demo(True, "ok", success), Demo(False, "wrong content", failure)]
    # require_exists alone cannot distinguish these → disagrees on the failure demo.
    checks = [{"type": "file", "require_exists": ["out/done.txt"]}]
    report = await validate_against_demos(checks, env, demos)
    assert report["agreement"] == 0.5


@pytest.mark.anyio
async def test_synthesize_and_validate_end_to_end(env: LocalEnvironment):
    async def success(e):
        await e.exec("rm -rf out && mkdir -p out")
        await e.write_file("out/done.txt", "ok")

    async def failure(e):
        await e.exec("rm -rf out && mkdir -p out")

    provider = _FakeProvider('[{"type":"file","require_exists":["out/done.txt"]}]')
    result = await synthesize_and_validate(
        "create out/done.txt",
        [Demo(True, "done.txt exists", success), Demo(False, "empty", failure)],
        env,
        provider,
    )
    assert result.validated is True
    assert result.validation["agreement"] == 1.0
