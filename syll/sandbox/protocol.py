"""Wire protocol for sandbox environments.

This module defines the HTTP/JSON schema a sandbox container can implement to
expose the same operations as :class:`syll.sandbox.environment.Environment`.
Keeping the protocol explicit and separate from both client and server makes
it easier to maintain compatibility when either side evolves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecRequest:
    """Request body for executing a shell command inside the sandbox."""

    command: str
    cwd: str | None = None
    timeout: int = 60


@dataclass
class FileRequest:
    """Generic file-operation request body."""

    path: str
    content: str | None = None
    old_text: str | None = None
    new_text: str | None = None


@dataclass
class ScreenshotResponse:
    """Response body returned by a screenshot request."""

    base64_png: str


@dataclass
class PointerRequest:
    """Pointer action request (click, move, scroll, drag)."""

    action: str  # click, double_click, right_click, move, scroll, drag
    x: int | None = None
    y: int | None = None
    end_x: int | None = None
    end_y: int | None = None
    scroll_x: int = 0
    scroll_y: int = 0
    button: str = "left"


@dataclass
class KeyboardRequest:
    """Keyboard action request."""

    action: str  # type, keypress
    text: str | None = None
    keys: list[str] = field(default_factory=list)


# Set of all sandbox action names. Useful for validating incoming requests.
SANDBOX_ACTIONS = {
    "exec",
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "screenshot",
    "click",
    "double_click",
    "right_click",
    "move",
    "scroll",
    "drag",
    "type",
    "keypress",
    "wait",
    "get_screen_size",
}


class SandboxProtocol:
    """Reference constants for the sandbox HTTP API.

    Not a server implementation — just routes and status codes shared by the
    client and any future container server.
    """

    BASE_PATH = "/sandbox"
    EXEC_ROUTE = f"{BASE_PATH}/exec"
    FILE_ROUTE = f"{BASE_PATH}/file"
    SCREENSHOT_ROUTE = f"{BASE_PATH}/screenshot"
    POINTER_ROUTE = f"{BASE_PATH}/pointer"
    KEYBOARD_ROUTE = f"{BASE_PATH}/keyboard"
    WAIT_ROUTE = f"{BASE_PATH}/wait"
    SCREEN_SIZE_ROUTE = f"{BASE_PATH}/screen_size"

    @staticmethod
    def encode_request_body(action: str, **params: Any) -> dict[str, Any]:
        """Return a JSON-serialisable request body for ``action``."""
        return {"action": action, **params}

    @staticmethod
    def decode_exec_response(body: dict[str, Any]) -> "ExecResult":
        """Convert a JSON exec response into an ``ExecResult``."""
        from .environment import ExecResult

        return ExecResult(
            stdout=body.get("stdout", ""),
            stderr=body.get("stderr", ""),
            returncode=body.get("returncode", 0),
        )
