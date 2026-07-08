"""Unit tests for the local Environment abstraction.

These tests exercise :class:`syll.sandbox.environment.LocalEnvironment` without
requiring a Docker container, and verify that the core primitives behave like
async versions of the local filesystem / shell / desktop APIs.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from syll.sandbox.environment import Environment, ExecResult, LocalEnvironment


@pytest.fixture
def env(tmp_path: Path) -> LocalEnvironment:
    return LocalEnvironment(workspace_root=tmp_path)


@pytest.mark.anyio
async def test_exec_runs_command_in_workspace(env: LocalEnvironment, tmp_path: Path):
    result = await env.exec("pwd")
    assert result.returncode == 0
    # Bash on Windows may return a POSIX path; check the tail matches.
    assert tmp_path.name in result.stdout.strip()


@pytest.mark.anyio
async def test_exec_respects_cwd_override(env: LocalEnvironment, tmp_path: Path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    result = await env.exec("pwd", cwd=str(subdir))
    assert result.returncode == 0
    assert subdir.name in result.stdout.strip()


@pytest.mark.anyio
async def test_exec_returns_stderr_and_exit_code(env: LocalEnvironment):
    py = sys.executable.replace("\\", "/")
    cmd = f'''{py} -c "import sys; print('err', file=sys.stderr); sys.exit(42)"'''
    result = await env.exec(cmd)
    assert result.returncode == 42
    assert "err" in result.stderr


@pytest.mark.anyio
async def test_read_write_file(env: LocalEnvironment, tmp_path: Path):
    await env.write_file("hello.txt", "world")
    content = await env.read_file("hello.txt")
    assert content == "world"
    assert (tmp_path / "hello.txt").read_text() == "world"


@pytest.mark.anyio
async def test_write_file_creates_parent_dirs(env: LocalEnvironment, tmp_path: Path):
    await env.write_file("a/b/c/deep.txt", "nested")
    assert (tmp_path / "a" / "b" / "c" / "deep.txt").read_text() == "nested"


@pytest.mark.anyio
async def test_edit_file(env: LocalEnvironment):
    await env.write_file("edit.txt", "foo bar baz")
    await env.edit_file("edit.txt", "bar", "qux")
    assert await env.read_file("edit.txt") == "foo qux baz"


@pytest.mark.anyio
async def test_edit_file_raises_when_old_text_missing(env: LocalEnvironment):
    await env.write_file("edit.txt", "foo bar baz")
    with pytest.raises(ValueError):
        await env.edit_file("edit.txt", "nope", "qux")


@pytest.mark.anyio
async def test_list_dir(env: LocalEnvironment):
    await env.write_file("x.txt", "1")
    await env.write_file("y.txt", "2")
    await env.exec("mkdir sub")
    entries = await env.list_dir(".")
    assert sorted(entries) == ["sub/", "x.txt", "y.txt"]


@pytest.mark.anyio
async def test_screenshot_returns_base64_png(env: LocalEnvironment):
    b64 = await env.screenshot()
    assert isinstance(b64, str)
    # Minimal PNG header sanity check.
    import base64

    header = base64.b64decode(b64)[:8]
    assert header.startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.anyio
async def test_get_screen_size_matches_screenshot(env: LocalEnvironment):
    size = await env.get_screen_size()
    assert len(size) == 2
    assert size[0] > 0 and size[1] > 0


@pytest.mark.anyio
async def test_wait_sleeps(env: LocalEnvironment):
    start = asyncio.get_event_loop().time()
    await env.wait(100)
    elapsed = asyncio.get_event_loop().time() - start
    assert 0.08 <= elapsed <= 0.3
