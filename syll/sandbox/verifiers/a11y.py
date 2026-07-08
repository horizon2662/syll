"""Accessibility-tree verifier for desktop GUI tasks.

Linux AT-SPI / Windows UIA / macOS AXUIElement let us read the live widget tree
and assert on window titles, focused controls, and element labels — a stronger
signal than pixel-diff for "did the right dialog open / field get filled".

Phase 0 scope: a single dependency-light check (active window title via
``xdotool`` when the XFCE sandbox exposes it). Full AT-SPI element queries need
``pyatspi`` (pygobject) wired into the Docker desktop image and are left as a
clearly-marked expansion point.

Spec keys:
    window_title_contains: str    active window title must contain this
    # (TODO, needs pyatspi in the desktop image):
    focused_role:   str           expected role of the focused control
    widget_label:   str           a widget with this accessible name must exist
"""

from __future__ import annotations

from typing import Any

from syll.sandbox.environment import Environment

from .base import Verifier, VerifierResult


class A11yVerifier(Verifier):
    name = "a11y"

    async def check(self, env: Environment, spec: dict[str, Any]) -> VerifierResult:
        if "window_title_contains" in spec:
            title = await self._active_window_title(env)
            if title is None:
                return VerifierResult.fail(
                    "could not read active window title (is xdotool installed "
                    "in the sandbox?)",
                    spec=spec,
                )
            needle = str(spec["window_title_contains"])
            if needle in title:
                return VerifierResult.pass_(
                    f"window title match: {title!r}", title=title
                )
            return VerifierResult.fail(
                f"window title {title!r} missing {needle!r}",
                title=title,
                needle=needle,
            )
        if "focused_role" in spec or "widget_label" in spec:
            return VerifierResult.fail(
                "AT-SPI element queries need pyatspi wired into the Docker "
                "desktop image (TODO)",
                spec=spec,
            )
        return VerifierResult.fail("a11y verifier needs a check key", spec=spec)

    @staticmethod
    async def _active_window_title(env: Environment) -> str | None:
        try:
            res = await env.exec(
                "xdotool getactivewindow getwindowname 2>/dev/null"
            )
        except Exception:
            return None
        if res.returncode != 0:
            return None
        title = res.stdout.strip()
        return title or None
