"""Environment abstraction: the low-level interface tools use to touch the OS.

This is intentionally minimal — it covers only the primitives that core tools
need (filesystem, shell, screenshot, pointer/keyboard). Higher-level semantics
(e.g. file-size limits, truncation, allowed-directory checks) remain in the
tools themselves, so the sandbox does not have to duplicate Syll policy.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import platform
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class ExecResult:
    """Result of a shell command executed in an environment."""

    stdout: str
    stderr: str
    returncode: int


@dataclass
class SafetyResult:
    """Outcome of an environment safety check (ENPIRE EN hard constraints).

    A violation (``ok=False``) means the env has reached an out-of-bounds or
    dangerous state — a protected file deleted, a critical dialog dismissed, a
    forbidden URL opened — and forces immediate task failure + automated reset.
    ``reset_to`` optionally names the checkpoint to restore on reset.
    """

    ok: bool
    detail: str = ""
    reset_to: str | None = None


class Environment(ABC):
    """Abstract runtime environment for Syll tools."""

    os_type: str = "linux"
    workspace_root: str = ""

    # ------------------------------------------------------------------
    # Shell
    # ------------------------------------------------------------------
    @abstractmethod
    async def exec(
        self, command: str, cwd: str | None = None, timeout: int = 60
    ) -> ExecResult:
        """Run ``command`` and return stdout, stderr, and exit code."""

    # ------------------------------------------------------------------
    # Filesystem
    # ------------------------------------------------------------------
    @abstractmethod
    async def read_file(self, path: str) -> str:
        """Return UTF-8 text content of ``path``."""

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """Write ``content`` to ``path`` as UTF-8 text."""

    @abstractmethod
    async def edit_file(self, path: str, old_text: str, new_text: str) -> None:
        """Replace every occurrence of ``old_text`` with ``new_text``."""

    @abstractmethod
    async def list_dir(self, path: str) -> list[str]:
        """Return sorted directory entry names."""

    # ------------------------------------------------------------------
    # Vision / GUI
    # ------------------------------------------------------------------
    @abstractmethod
    async def screenshot(self) -> str:
        """Return a base64-encoded PNG screenshot."""

    @abstractmethod
    async def get_screen_size(self) -> tuple[int, int]:
        """Return (width, height) of the primary screen in logical pixels."""

    # ------------------------------------------------------------------
    # Pointer
    # ------------------------------------------------------------------
    @abstractmethod
    async def click(self, x: int, y: int, button: str = "left") -> None:
        """Click at screen coordinates (x, y)."""

    @abstractmethod
    async def double_click(self, x: int, y: int) -> None:
        """Double-click at screen coordinates (x, y)."""

    @abstractmethod
    async def right_click(self, x: int, y: int) -> None:
        """Right-click at screen coordinates (x, y)."""

    @abstractmethod
    async def move(self, x: int, y: int) -> None:
        """Move the pointer to screen coordinates (x, y)."""

    @abstractmethod
    async def scroll(
        self, x: int, y: int, scroll_x: int = 0, scroll_y: int = 0
    ) -> None:
        """Scroll at screen coordinates (x, y)."""

    @abstractmethod
    async def drag(
        self, start_x: int, start_y: int, end_x: int, end_y: int
    ) -> None:
        """Drag from ``start`` to ``end`` in screen coordinates."""

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------
    @abstractmethod
    async def type(self, text: str) -> None:
        """Type ``text`` as a sequence of key events."""

    @abstractmethod
    async def keypress(self, keys: list[str] | str) -> None:
        """Press a key or chord.

        ``keys`` may be a single key name or a list of keys to press in order.
        """

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    async def wait(self, ms: int = 1000) -> None:
        """Sleep for ``ms`` milliseconds."""
        await asyncio.sleep(ms / 1000.0)

    # ------------------------------------------------------------------
    # Sandbox lifecycle (ENPIRE EN: reset + verify)
    # ------------------------------------------------------------------
    # These turn an Environment into a repeatable experiment substrate.
    # ``checkpoint``/``restore`` implement ENPIRE's "reset to the onset of the
    # hardest phase" and MACU's init_from / variant_of semantics; ``reset`` is
    # the high-level entry point. Backends that cannot snapshot a real machine
    # (e.g. ``LocalEnvironment``) raise ``NotImplementedError`` — real reset
    # semantics live in the Docker/VM sandbox backends.
    @abstractmethod
    async def checkpoint(self, tag: str) -> str:
        """Capture current env state and return a checkpoint id.

        Used for ENPIRE phase-reset (snapshot at the onset of the hardest
        step) and MACU ``init_from`` / ``variant_of`` (inherit a predecessor's
        terminal state, or fork it for a structural retry).
        """

    @abstractmethod
    async def restore(self, checkpoint_id: str) -> None:
        """Restore env to the named checkpoint."""

    @abstractmethod
    async def reset(
        self, mode: str = "full", checkpoint_id: str | None = None
    ) -> None:
        """Reset the environment.

        mode:
          - ``"full"``: revert to the baseline start state.
          - ``"phase"``: restore a phase checkpoint (``checkpoint_id``) —
            ENPIRE's budget-focused "reset to hardest-phase onset".
          - ``"init_from"``: inherit a predecessor's terminal state (MACU);
            ``checkpoint_id`` names it.
          - ``"variant_of"``: fork a checkpoint for a structural retry.
        """

    @abstractmethod
    async def safety_check(self) -> SafetyResult:
        """Evaluate hard safety constraints; violations force a reset."""


class LocalEnvironment(Environment):
    """Environment implementation that delegates to the local machine."""

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        os_type: str | None = None,
        mac_click_style: str | None = None,
    ):
        self.workspace_root = Path(
            workspace_root or os.getcwd()
        ).expanduser().resolve()
        self.os_type = os_type or platform.system().lower()
        self.mac_click_style = mac_click_style or "auto"
        self._mouse_backend: Any | None = None
        self._pyautogui: Any | None = None

    def _resolve_path(self, path: str) -> Path:
        """Resolve ``path`` relative to ``workspace_root`` when not absolute."""
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.workspace_root / p
        return p.resolve()

    # ------------------------------------------------------------------
    # Shell
    # ------------------------------------------------------------------
    async def exec(
        self, command: str, cwd: str | None = None, timeout: int = 60
    ) -> ExecResult:
        cwd = cwd or str(self.workspace_root)
        logger.debug(f"[LocalEnvironment] exec: {command!r} in {cwd}")
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return ExecResult("", "timeout", 1)
        return ExecResult(
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
            proc.returncode or 0,
        )

    # ------------------------------------------------------------------
    # Filesystem
    # ------------------------------------------------------------------
    async def read_file(self, path: str) -> str:
        return self._resolve_path(path).read_text(encoding="utf-8")

    async def write_file(self, path: str, content: str) -> None:
        p = self._resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    async def edit_file(self, path: str, old_text: str, new_text: str) -> None:
        p = self._resolve_path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        text = p.read_text(encoding="utf-8")
        if old_text not in text:
            raise ValueError("old_text not found in file")
        p.write_text(text.replace(old_text, new_text), encoding="utf-8")

    async def list_dir(self, path: str) -> list[str]:
        p = self._resolve_path(path)
        if not p.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not p.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        return sorted(
            f"{entry.name}/" if entry.is_dir() else entry.name
            for entry in p.iterdir()
        )

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------
    async def screenshot(self) -> str:
        try:
            import mss
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(
                "Screenshot requires mss and Pillow. Install: pip install mss Pillow"
            ) from e

        with mss.mss() as sct:
            monitor = sct.monitors[0]
            shot = sct.grab(monitor)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            logical_w = monitor["width"]
            logical_h = monitor["height"]
            if img.width > logical_w or img.height > logical_h:
                img = img.resize((logical_w, logical_h), Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()

    async def get_screen_size(self) -> tuple[int, int]:
        try:
            import mss

            with mss.mss() as sct:
                monitor = sct.monitors[0]
                return monitor["width"], monitor["height"]
        except Exception:
            # Fallback to pyautogui for parity with the legacy screenshot path
            # and to keep tests that mock pyautogui.size working.
            pg = self._get_pyautogui()
            return pg.size()

    # ------------------------------------------------------------------
    # Pointer
    # ------------------------------------------------------------------
    async def click(self, x: int, y: int, button: str = "left") -> None:
        pg = self._get_pyautogui()
        if self._use_down_up():
            pg.mouseDown(x=x, y=y, button=button)
            await asyncio.sleep(0.03)
            pg.mouseUp(x=x, y=y, button=button)
        else:
            pg.click(x, y, button=button)

    async def double_click(self, x: int, y: int) -> None:
        pg = self._get_pyautogui()
        if self._use_down_up():
            await self.click(x, y)
            await asyncio.sleep(0.03)
            await self.click(x, y)
        else:
            pg.doubleClick(x, y)

    async def right_click(self, x: int, y: int) -> None:
        pg = self._get_pyautogui()
        if self._use_down_up():
            pg.mouseDown(x=x, y=y, button="right")
            await asyncio.sleep(0.03)
            pg.mouseUp(x=x, y=y, button="right")
        else:
            pg.rightClick(x, y)

    async def move(self, x: int, y: int) -> None:
        pg = self._get_pyautogui()
        pg.moveTo(x, y)

    async def scroll(
        self, x: int, y: int, scroll_x: int = 0, scroll_y: int = 0
    ) -> None:
        pg = self._get_pyautogui()
        pg.scroll(scroll_y, x=x, y=y)

    async def drag(
        self, start_x: int, start_y: int, end_x: int, end_y: int
    ) -> None:
        pg = self._get_pyautogui()
        pg.moveTo(start_x, start_y)
        pg.dragTo(end_x, end_y)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------
    async def type(self, text: str, interval: float = 0.05) -> None:
        """Type ``text``. For non-ASCII text, paste via the clipboard.

        ``pyautogui.write`` can only press characters present in the active
        keyboard layout, so CJK / emoji / other non-ASCII characters must be
        pasted (pyperclip + the platform paste shortcut).
        """
        pg = self._get_pyautogui()
        if text.isascii():
            pg.write(text, interval=interval)
            return
        try:
            import pyperclip

            pyperclip.copy(text)
            if self.os_type == "darwin":
                pg.hotkey("command", "v")
            else:
                pg.hotkey("ctrl", "v")
        except ImportError:
            # Fallback: press ASCII characters only.
            for ch in text:
                if ch.isascii():
                    pg.press(ch)

    async def keypress(self, keys: list[str] | str) -> None:
        pg = self._get_pyautogui()
        if isinstance(keys, str):
            # A single sequence like "command+shift+t".
            normalized = self._normalize_hotkey_sequence(keys)
            if len(normalized) >= 2:
                pg.hotkey(*normalized)
            else:
                pg.press(normalized[0] if normalized else keys)
        else:
            for k in keys:
                pg.press(k)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_pyautogui(self) -> Any:
        if self._pyautogui is None:
            import pyautogui

            self._pyautogui = pyautogui
        return self._pyautogui

    def _use_down_up(self) -> bool:
        """Return whether macOS clicks should use explicit mouseDown/mouseUp."""
        if self.os_type != "darwin":
            return False
        style = self.mac_click_style.lower()
        if style == "down_up":
            return True
        if style == "click":
            return False
        # auto: prefer down_up on macOS for better accessibility reliability.
        return True

    def _normalize_hotkey_sequence(self, sequence: str) -> list[str]:
        """Translate OS-agnostic hotkey names into pyautogui key names."""
        parts = [p.strip().lower() for p in sequence.replace("-", "+").split("+")]
        mod_map = {
            "command": "command" if self.os_type == "darwin" else "ctrl",
            "cmd": "command" if self.os_type == "darwin" else "ctrl",
            "win": "win",
            "super": "command" if self.os_type == "darwin" else "win",
            "option": "option" if self.os_type == "darwin" else "alt",
            "alt": "alt",
            "shift": "shift",
            "ctrl": "ctrl",
            "control": "ctrl",
        }
        return [mod_map.get(p, p) for p in parts]

    # ------------------------------------------------------------------
    # Sandbox lifecycle — LocalEnvironment runs on the real machine, so
    # snapshot / reset is refused (it would nuke the user's desktop). Real
    # reset semantics live in the Docker/VM backend
    # (``syll.sandbox.backends``). ``safety_check`` still probes the workspace.
    # ------------------------------------------------------------------
    async def checkpoint(self, tag: str) -> str:
        raise NotImplementedError(
            "LocalEnvironment cannot snapshot the real machine; use the "
            "Docker/VM sandbox backend (syll.sandbox.backends) for reset."
        )

    async def restore(self, checkpoint_id: str) -> None:
        raise NotImplementedError(
            "LocalEnvironment cannot restore the real machine; use the "
            "Docker/VM sandbox backend (syll.sandbox.backends) for reset."
        )

    async def reset(
        self, mode: str = "full", checkpoint_id: str | None = None
    ) -> None:
        raise NotImplementedError(
            "LocalEnvironment will not reset the real machine; use the "
            "Docker/VM sandbox backend (syll.sandbox.backends) for "
            "deterministic reset."
        )

    async def safety_check(self) -> SafetyResult:
        """Best-effort safety probe of the local workspace."""
        root = self.workspace_root
        if not root.exists() or not root.is_dir():
            return SafetyResult(ok=False, detail=f"workspace missing: {root}")
        if not os.access(root, os.W_OK):
            return SafetyResult(ok=False, detail=f"workspace not writable: {root}")
        return SafetyResult(ok=True)
