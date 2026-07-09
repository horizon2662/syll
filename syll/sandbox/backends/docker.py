"""Docker sandbox backend (ENPIRE EN reset + R rollout substrate).

Runs primitives inside a Docker desktop container via ``docker exec``.
Snapshot/reset is Docker-native and needs no extra filesystem plumbing:

- ``reset("full")``        — remove + recreate the container from the base
  image (a deterministic clean slate).
- ``checkpoint(tag)``      — ``docker commit`` the running container to a
  tagged image and return that tag as the checkpoint id.
- ``restore(checkpoint_id)`` / ``reset("phase", checkpoint_id)`` — recreate the
  container from the tagged image (ENPIRE phase-reset: restore a predecessor's
  terminal state, or fork it for a retry — MACU ``init_from`` / ``variant_of``).

Requires the ``docker`` CLI on PATH and the desktop image built (XFCE + the
target apps + xdotool + a screenshot tool such as ImageMagick ``import`` or
``scrot``). Lives unverified in environments without a Docker daemon; every
primitive raises ``RuntimeError`` if the container is not reachable.
"""

from __future__ import annotations

import asyncio
import base64
import os
import secrets
import shutil
from dataclasses import dataclass
from typing import Any

from syll.sandbox.environment import Environment, ExecResult, SafetyResult

# Docker Desktop's default install location (Windows) — used when `docker`
# isn't on the PATH the Python process inherited (common when launching from
# Git Bash, where Docker Desktop's bin is absent from the inherited PATH).
_DOCKER_DD_PATHS = (
    r"C:\Program Files\Docker\Docker\resources\bin\docker.exe",
    r"C:\Program Files\Docker\Docker\resources\bin\docker",
    "/usr/local/bin/docker",
    "/usr/bin/docker",
)


def docker_exe() -> str | None:
    """Resolve the docker CLI: PATH first, then Docker Desktop's default
    location. Returns ``None`` if no docker executable is found."""
    return (
        shutil.which("docker")
        or shutil.which("docker.exe")
        or next((p for p in _DOCKER_DD_PATHS if os.path.exists(p)), None)
    )


@dataclass
class _ContainerSpec:
    image: str
    name: str
    workspace_root: str = "/root/workspace"


def _shquote(s: str) -> str:
    """Single-quote a shell arg (double-quote wrapping is not safe for raw
    shell ``sh -c`` inside ``docker exec``)."""
    return "'" + s.replace("'", "'\\''") + "'"


class DockerEnvironment(Environment):
    """An :class:`Environment` backed by a Docker desktop container."""

    os_type = "linux"

    def __init__(
        self,
        image: str,
        container: str | None = None,
        workspace_root: str = "/root/workspace",
        run_args: str = "--shm-size 2g -e DISPLAY=:1",
    ) -> None:
        self._spec = _ContainerSpec(image=image, name=container or self._gen_name(), workspace_root=workspace_root)
        self._run_args = run_args
        self._started = False
        # Resolve the docker CLI once (PATH, else Docker Desktop default path).
        self._docker = docker_exe()

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------
    async def ensure_started(self) -> None:
        """Create + start the container if it isn't already running."""
        if self._docker is None:
            raise RuntimeError(
                "docker CLI not found on PATH or at Docker Desktop's default "
                "location (C:\\Program Files\\Docker\\Docker\\resources\\bin). "
                "Start Docker Desktop and/or add it to PATH."
            )
        if await self._container_running():
            self._started = True
            return
        await self._docker(
            "run", "-d", "--name", self._spec.name, *self._run_args.split(),
            self._spec.image, "tail", "-f", "/dev/null",
        )
        self._started = True

    async def checkpoint(self, tag: str) -> str:
        await self.ensure_started()
        image_tag = f"{self._spec.name}:{tag}"
        await self._docker("commit", self._spec.name, image_tag)
        return image_tag

    async def restore(self, checkpoint_id: str) -> None:
        await self._recreate_from_image(checkpoint_id)

    async def reset(
        self, mode: str = "full", checkpoint_id: str | None = None
    ) -> None:
        if mode == "full":
            await self._recreate_from_image(self._spec.image)
        elif mode in ("phase", "init_from", "variant_of"):
            if not checkpoint_id:
                raise ValueError(f"reset mode {mode!r} requires checkpoint_id")
            await self._recreate_from_image(checkpoint_id)
        else:
            raise ValueError(f"unknown reset mode {mode!r}")

    async def safety_check(self) -> SafetyResult:
        if not await self._container_running():
            return SafetyResult(ok=False, detail=f"container {self._spec.name} not running")
        return SafetyResult(ok=True)

    # ------------------------------------------------------------------
    # Shell / filesystem
    # ------------------------------------------------------------------
    async def exec(
        self, command: str, cwd: str | None = None, timeout: int = 60
    ) -> ExecResult:
        await self.ensure_started()
        target = cwd or self._spec.workspace_root
        inner = f"cd {_shquote(target)} 2>/dev/null; {command}"
        return await self._exec_sh(inner, timeout=timeout)

    async def read_file(self, path: str) -> str:
        res = await self._exec_sh(f"cat {_shquote(path)}")
        if res.returncode != 0:
            raise FileNotFoundError(f"read_file failed: {res.stderr.strip()}")
        return res.stdout

    async def write_file(self, path: str, content: str) -> None:
        parent = path.rsplit("/", 1)[0]
        if parent and parent != path:
            await self._exec_sh(f"mkdir -p {_shquote(parent)}")
        # Pipe content via stdin to avoid shell-quoting the body.
        await self._exec_sh(
            f"tee {_shquote(path)} >/dev/null", stdin=content
        )

    async def edit_file(self, path: str, old_text: str, new_text: str) -> None:
        text = await self.read_file(path)
        if old_text not in text:
            raise ValueError("old_text not found in file")
        await self.write_file(path, text.replace(old_text, new_text))

    async def list_dir(self, path: str) -> list[str]:
        res = await self._exec_sh(
            f"for e in {_shquote(path)}/* {_shquote(path)}/.*; do "
            f"[ -e \"$e\" ] || continue; "
            f"basename=$(basename \"$e\"); "
            f"[ -d \"$e\" ] && basename=\"$basename/\"; printf '%s\\n' \"$basename\"; done"
        )
        if res.returncode != 0:
            raise FileNotFoundError(f"list_dir failed: {res.stderr.strip()}")
        return sorted(line for line in res.stdout.splitlines() if line)

    # ------------------------------------------------------------------
    # Vision / GUI
    # ------------------------------------------------------------------
    async def screenshot(self) -> str:
        # Prefer ImageMagick `import`, fall back to `scrot`.
        script = (
            "import -window root /tmp/_s.png 2>/dev/null || scrot /tmp/_s.png 2>/dev/null; "
            "base64 -w0 /tmp/_s.png"
        )
        res = await self._exec_sh(script)
        if res.returncode != 0 or not res.stdout.strip():
            raise RuntimeError(f"screenshot failed (needs ImageMagick/scrot in image): {res.stderr.strip()}")
        return res.stdout.strip()

    async def get_screen_size(self) -> tuple[int, int]:
        res = await self._exec_sh("xdotool getdisplaygeometry 2>/dev/null || xdpyinfo 2>/dev/null | grep dimensions")
        out = res.stdout.strip().split()
        if out and out[0].isdigit():
            return int(out[0]), int(out[1]) if len(out) > 1 and out[1].isdigit() else 0
        return 0, 0

    # ------------------------------------------------------------------
    # Pointer / keyboard (xdotool)
    # ------------------------------------------------------------------
    async def click(self, x: int, y: int, button: str = "left") -> None:
        btn = {"left": "1", "middle": "2", "right": "3"}.get(button, "1")
        await self._exec_sh(f"xdotool mousemove {x} {y} click {btn}")

    async def double_click(self, x: int, y: int) -> None:
        await self._exec_sh(f"xdotool mousemove {x} {y} click --repeat 2 1")

    async def right_click(self, x: int, y: int) -> None:
        await self._exec_sh(f"xdotool mousemove {x} {y} click 3")

    async def move(self, x: int, y: int) -> None:
        await self._exec_sh(f"xdotool mousemove {x} {y}")

    async def scroll(self, x: int, y: int, scroll_x: int = 0, scroll_y: int = 0) -> None:
        btn = 5 if scroll_y > 0 else 4 if scroll_y < 0 else 6 if scroll_x > 0 else 7
        n = abs(scroll_y or scroll_x or 1)
        await self._exec_sh(f"xdotool mousemove {x} {y} click --repeat {n} {btn}")

    async def drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        await self._exec_sh(
            f"xdotool mousemove {start_x} {start_y} mousedown 1 "
            f"mousemove {end_x} {end_y} mouseup 1"
        )

    async def type(self, text: str) -> None:
        await self._exec_sh(f"xdotool type -- {_shquote(text)}")

    async def keypress(self, keys: list[str] | str) -> None:
        seq = "+".join(keys) if isinstance(keys, list) else keys
        await self._exec_sh(f"xdotool key {_shquote(seq)}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _gen_name() -> str:
        return "syll-sandbox-" + secrets.token_hex(6)

    async def _exec_sh(
        self, inner: str, *, timeout: int = 60, stdin: str | None = None
    ) -> ExecResult:
        await self.ensure_started()
        proc = await asyncio.create_subprocess_exec(
            self._docker, "exec", "-i", self._spec.name, "sh", "-c", inner,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        )
        try:
            data = await asyncio.wait_for(
                proc.communicate(stdin.encode() if stdin else None), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return ExecResult("", "timeout", 1)
        out, err = data
        return ExecResult(
            out.decode(errors="replace"),
            err.decode(errors="replace"),
            proc.returncode or 0,
        )

    async def _docker(self, *args: str) -> ExecResult:
        proc = await asyncio.create_subprocess_exec(
            self._docker, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        rc = proc.returncode or 0
        if rc != 0:
            raise RuntimeError(f"docker {args[0]} failed: {err.decode(errors='replace').strip()}")
        return ExecResult(out.decode(errors="replace"), err.decode(errors="replace"), rc)

    async def _container_running(self) -> bool:
        proc = await asyncio.create_subprocess_exec(
            self._docker, "ps", "--filter", f"name=^{self._spec.name}$", "--format", "{{.Names}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        return self._spec.name in out.decode().split()

    async def _recreate_from_image(self, image: str) -> None:
        # Remove the current container (ignore failure if absent) and start a
        # fresh one from `image`. This is the deterministic full-reset path.
        kill = await asyncio.create_subprocess_exec(
            self._docker, "rm", "-f", self._spec.name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await kill.communicate()
        await self._docker(
            "run", "-d", "--name", self._spec.name, *self._run_args.split(),
            image, "tail", "-f", "/dev/null",
        )
        self._started = True
