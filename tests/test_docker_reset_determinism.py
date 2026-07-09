"""Live reset-determinism tests for the Docker sandbox backend.

Skips automatically when Docker or the base image is absent, so this file is
safe to run in any environment. With Docker + `syll-sandbox-base:latest` it
verifies the two reset semantics the whole ENPIRE×Aspire loop depends on:

- ``reset("full")`` wipes mutations back to the clean baseline;
- ``checkpoint`` / ``reset("phase", ...)`` restores a predecessor state and
  drops post-checkpoint changes (ENPIRE phase-reset / MACU init_from).

Build the image first::

    docker build -t syll-sandbox-base:latest -f syll/sandbox/docker/Dockerfile syll/sandbox/docker
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from syll.sandbox.backends.docker import DockerEnvironment, docker_exe
from syll.sandbox.verifiers import FileVerifier

IMAGE = "syll-sandbox-base:latest"
_DOCKER = docker_exe()


def _daemon_running() -> bool:
    """True if the docker CLI is resolvable AND the engine responds."""
    if _DOCKER is None:
        return False
    try:
        return subprocess.run(
            [_DOCKER, "version", "--format", "{{.Server.Version}}"],
            capture_output=True, timeout=20,
        ).returncode == 0
    except Exception:
        return False


def _image_available() -> bool:
    if not _daemon_running():
        return False
    try:
        return subprocess.run(
            [_DOCKER, "image", "inspect", IMAGE], capture_output=True, timeout=20
        ).returncode == 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _image_available(),
    reason=(
        f"needs a RUNNING docker daemon + image {IMAGE}. Resolve docker "
        "(PATH or Docker Desktop default), start Docker Desktop, then build: "
        f"`docker build -t {IMAGE} -f syll/sandbox/docker/Dockerfile syll/sandbox/docker`"
    ),
)


def _rm_container(name: str) -> None:
    if _DOCKER is None:
        return
    subprocess.run([_DOCKER, "rm", "-f", name], capture_output=True)


@pytest.mark.anyio
async def test_full_reset_clears_mutations():
    env = DockerEnvironment(image=IMAGE, container="syll-test-full")
    _rm_container("syll-test-full")
    try:
        await env.reset("full")
        await env.write_file("out/marker.txt", "x")
        assert (await env.read_file("out/marker.txt")) == "x"

        await env.reset("full")
        res = await FileVerifier().check(
            env, {"require_exists": ["out/marker.txt"]}
        )
        assert res.passed is False  # mutation wiped by full reset
    finally:
        _rm_container("syll-test-full")


@pytest.mark.anyio
async def test_phase_reset_restores_checkpoint_and_drops_later_changes():
    env = DockerEnvironment(image=IMAGE, container="syll-test-phase")
    _rm_container("syll-test-phase")
    try:
        await env.reset("full")
        await env.write_file("out/a.txt", "baseline")
        cid = await env.checkpoint("phase1")

        await env.write_file("out/a.txt", "mutated")
        await env.write_file("out/extra.txt", "noise")

        await env.reset("phase", checkpoint_id=cid)
        assert (await env.read_file("out/a.txt")) == "baseline"
        assert (
            await FileVerifier().check(env, {"require_exists": ["out/extra.txt"]})
        ).passed is False  # post-checkpoint change dropped
    finally:
        _rm_container("syll-test-phase")


@pytest.mark.anyio
async def test_repeatable_start_state_across_runs():
    """Two fresh full-resets must yield the same start state (determinism)."""
    env = DockerEnvironment(image=IMAGE, container="syll-test-rep")
    _rm_container("syll-test-rep")
    try:
        await env.reset("full")
        listing_a = await env.list_dir("/root/workspace")
        # mutate, then reset and confirm the listing matches the original.
        await env.write_file("leftover.txt", "x")
        await env.reset("full")
        listing_b = await env.list_dir("/root/workspace")
        assert listing_a == listing_b
    finally:
        _rm_container("syll-test-rep")
