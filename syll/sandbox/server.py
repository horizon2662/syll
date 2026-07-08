"""FastAPI sandbox server: exposes an ``Environment`` backend over the HTTP wire
protocol defined in :mod:`syll.sandbox.protocol`.

One server manages one sandbox backend (a Docker container, VM, or in-process
``LocalEnvironment``). Routes mirror :class:`SandboxProtocol` plus the ENPIRE
lifecycle endpoints (``/start`` /restart /reset /checkpoint /restore /safety).
The server is backend-agnostic — give it any :class:`Environment` and it
forwards requests, so the same harness drives a local, Docker, or VM backend.

Run ad hoc::

    python -m syll.sandbox.server --backend local --workspace /tmp/sandbox
    python -m syll.sandbox.server --backend docker --image syll-desktop:latest
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from syll.sandbox.environment import Environment, LocalEnvironment

# -----------------------------------------------------------------------
# Request bodies (mirror syll.sandbox.protocol)
# -----------------------------------------------------------------------


class ExecBody(BaseModel):
    command: str
    cwd: str | None = None
    timeout: int = 60


class FileBody(BaseModel):
    action: str  # read_file | write_file | edit_file | list_dir
    path: str
    content: str | None = None
    old_text: str | None = None
    new_text: str | None = None


class PointerBody(BaseModel):
    action: str  # click | double_click | right_click | move | scroll | drag
    x: int | None = None
    y: int | None = None
    end_x: int | None = None
    end_y: int | None = None
    scroll_x: int = 0
    scroll_y: int = 0
    button: str = "left"


class KeyboardBody(BaseModel):
    action: str  # type | keypress
    text: str | None = None
    keys: list[str] | None = None


class WaitBody(BaseModel):
    ms: int = 1000


class ResetBody(BaseModel):
    mode: str = "full"
    checkpoint_id: str | None = None


class CheckpointBody(BaseModel):
    tag: str


class RestoreBody(BaseModel):
    checkpoint_id: str


# -----------------------------------------------------------------------
# App factory
# -----------------------------------------------------------------------


def create_sandbox_app(backend: Environment) -> FastAPI:
    """Build a FastAPI app that forwards requests to ``backend``."""
    app = FastAPI(title="Syll Sandbox", version="0.1.0")
    app.state.backend = backend

    @app.get("/sandbox/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "backend": type(backend).__name__}

    @app.post("/sandbox/exec")
    async def exec_(body: ExecBody) -> dict[str, Any]:
        res = await backend.exec(body.command, cwd=body.cwd, timeout=body.timeout)
        return {
            "stdout": res.stdout,
            "stderr": res.stderr,
            "returncode": res.returncode,
        }

    @app.post("/sandbox/file")
    async def file_(body: FileBody) -> dict[str, Any]:
        if body.action == "read_file":
            return {"content": await backend.read_file(body.path)}
        if body.action == "write_file":
            await backend.write_file(body.path, body.content or "")
            return {"ok": True}
        if body.action == "edit_file":
            await backend.edit_file(
                body.path, body.old_text or "", body.new_text or ""
            )
            return {"ok": True}
        if body.action == "list_dir":
            return {"entries": await backend.list_dir(body.path)}
        raise HTTPException(400, f"unknown file action {body.action!r}")

    @app.get("/sandbox/screenshot")
    async def screenshot() -> dict[str, str]:
        b64 = await backend.screenshot()
        return {"base64_png": b64}

    @app.get("/sandbox/screen_size")
    async def screen_size() -> dict[str, int]:
        w, h = await backend.get_screen_size()
        return {"width": w, "height": h}

    @app.post("/sandbox/pointer")
    async def pointer(body: PointerBody) -> dict[str, Any]:
        if body.action == "click":
            await backend.click(body.x or 0, body.y or 0, body.button)
        elif body.action == "double_click":
            await backend.double_click(body.x or 0, body.y or 0)
        elif body.action == "right_click":
            await backend.right_click(body.x or 0, body.y or 0)
        elif body.action == "move":
            await backend.move(body.x or 0, body.y or 0)
        elif body.action == "scroll":
            await backend.scroll(
                body.x or 0,
                body.y or 0,
                body.scroll_x,
                body.scroll_y,
            )
        elif body.action == "drag":
            await backend.drag(
                body.x or 0,
                body.y or 0,
                body.end_x or 0,
                body.end_y or 0,
            )
        else:
            raise HTTPException(400, f"unknown pointer action {body.action!r}")
        return {"ok": True}

    @app.post("/sandbox/keyboard")
    async def keyboard(body: KeyboardBody) -> dict[str, Any]:
        if body.action == "type":
            await backend.type(body.text or "")
        elif body.action == "keypress":
            await backend.keypress(body.keys if body.keys is not None else "")
        else:
            raise HTTPException(400, f"unknown keyboard action {body.action!r}")
        return {"ok": True}

    @app.post("/sandbox/wait")
    async def wait_(body: WaitBody) -> dict[str, Any]:
        await backend.wait(body.ms)
        return {"ok": True}

    # --- ENPIRE lifecycle ------------------------------------------------

    @app.post("/sandbox/start")
    async def start_() -> dict[str, Any]:
        # The backend is already attached; /start is a no-op marker kept for
        # ENPIRE endpoint parity (/start begins a rollout on real hardware).
        return {"ok": True, "backend": type(backend).__name__}

    @app.post("/sandbox/restart")
    async def restart_() -> dict[str, Any]:
        await backend.reset("full")
        return {"ok": True}

    @app.post("/sandbox/reset")
    async def reset_(body: ResetBody) -> dict[str, Any]:
        await backend.reset(body.mode, body.checkpoint_id)
        return {"ok": True, "mode": body.mode}

    @app.post("/sandbox/checkpoint")
    async def checkpoint_(body: CheckpointBody) -> dict[str, str]:
        cid = await backend.checkpoint(body.tag)
        return {"checkpoint_id": cid}

    @app.post("/sandbox/restore")
    async def restore_(body: RestoreBody) -> dict[str, Any]:
        await backend.restore(body.checkpoint_id)
        return {"ok": True}

    @app.get("/sandbox/safety")
    async def safety_() -> dict[str, Any]:
        res = await backend.safety_check()
        return {
            "ok": res.ok,
            "detail": res.detail,
            "reset_to": res.reset_to,
        }

    return app


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def _build_backend(args: argparse.Namespace) -> Environment:
    if args.backend == "local":
        return LocalEnvironment(workspace_root=args.workspace or ".")
    if args.backend == "docker":
        from syll.sandbox.backends.docker import DockerEnvironment

        return DockerEnvironment(image=args.image, container=args.container)
    raise SystemExit(f"unknown backend {args.backend!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Syll sandbox server")
    parser.add_argument("--backend", default="local", help="local | docker")
    parser.add_argument("--workspace", default=None, help="local backend workspace root")
    parser.add_argument("--image", default=None, help="docker image")
    parser.add_argument("--container", default=None, help="docker container name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8086)
    args = parser.parse_args()

    import uvicorn

    backend = _build_backend(args)
    app = create_sandbox_app(backend)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
