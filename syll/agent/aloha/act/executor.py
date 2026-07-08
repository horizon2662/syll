"""Executor: converts high-level actions into GUI operations.

Adapted from ShowUI-Aloha/Aloha_Act/ui_aloha/execute/executor/aloha_executor.py.
Uses a shared mouse backend layer so macOS can switch away from raw pyautogui clicks,
and now operates on an :class:`syll.sandbox.environment.Environment` so the same
execution code can target the local desktop or a future sandbox container.
"""

import asyncio
import platform
from typing import Any

from loguru import logger

from syll.agent.gui_click import (
    normalize_hotkey_sequence,
    open_desktop_app_with_shortcut,
    perform_click_sequence,
    perform_drag,
    perform_move,
    perform_press,
    perform_right_click,
    resolve_click_count,
    should_open_desktop_app_with_shortcut,
)
from syll.sandbox.environment import Environment, LocalEnvironment


class AlohaExecutor:
    """Executes parsed GUI actions via shared click backends."""

    SUPPORTED_ACTIONS = {
        "CLICK",
        "RIGHT_CLICK",
        "DOUBLE_CLICK",
        "TRIPLE_CLICK",
        "INPUT",
        "TYPE",
        "KEY",
        "HOTKEY",
        "DRAG",
        "SCROLL",
        "MOVE",
        "HOVER",
        "ENTER",
        "ESC",
        "ESCAPE",
        "PRESS",
        "WAIT",
        "PAUSE",
        "CONTINUE",
    }

    def __init__(
        self,
        gui_config: Any | None = None,
        environment: Environment | None = None,
    ):
        self._config = gui_config
        self._environment = environment or LocalEnvironment()

    async def execute(self, action_dict: dict) -> tuple[bool, str]:
        """Execute a single action dict from actor output."""
        if self._environment is None:
            return False, "Environment not configured"

        # Local-only pyautogui safety settings. Harmless when the environment is
        # a sandbox; the actual pointer operations go through ``self._environment``.
        try:
            import pyautogui

            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.3
        except Exception:
            pass

        action_name = str(action_dict.get("action", "")).upper()
        value = action_dict.get("value", "")
        position = action_dict.get("position")

        if action_name == "ERROR":
            return False, f"Actor returned error: {value}"

        if action_name not in self.SUPPORTED_ACTIONS:
            return False, f"Unsupported action: {action_name}"

        try:
            return await self._dispatch(action_name, position, value, action_dict)
        except Exception as exc:
            logger.error(f"Executor error: {exc}")
            return False, str(exc)

    async def _dispatch(
        self,
        action: str,
        position: list | None,
        value: str,
        raw: dict,
    ) -> tuple[bool, str]:
        env = self._environment
        if action == "CLICK":
            if not position:
                return False, "CLICK requires position"
            x, y = int(position[0]), int(position[1])
            if should_open_desktop_app_with_shortcut(action, raw):
                message = await open_desktop_app_with_shortcut(
                    env,
                    x,
                    y,
                    raw=raw,
                    config=self._config,
                )
                return True, message
            click_count = resolve_click_count(action, raw)
            message = await perform_click_sequence(
                env,
                x,
                y,
                click_count,
                raw=raw,
                config=self._config,
            )
            return True, message

        if action == "RIGHT_CLICK":
            if not position:
                return False, "RIGHT_CLICK requires position"
            message = await perform_right_click(
                env,
                int(position[0]),
                int(position[1]),
                raw=raw,
                config=self._config,
            )
            return True, message

        if action == "DOUBLE_CLICK":
            if not position:
                return False, "DOUBLE_CLICK requires position"
            x, y = int(position[0]), int(position[1])
            if should_open_desktop_app_with_shortcut(action, raw):
                message = await open_desktop_app_with_shortcut(
                    env,
                    x,
                    y,
                    raw=raw,
                    config=self._config,
                )
                return True, message
            message = await perform_click_sequence(
                env,
                x,
                y,
                2,
                raw=raw,
                config=self._config,
            )
            return True, message

        if action == "TRIPLE_CLICK":
            if not position:
                return False, "TRIPLE_CLICK requires position"
            x, y = int(position[0]), int(position[1])
            message = await perform_click_sequence(
                env,
                x,
                y,
                3,
                raw=raw,
                config=self._config,
            )
            return True, message

        if action in ("INPUT", "TYPE"):
            text = value or raw.get("text", "")
            if not text:
                return False, "TYPE requires value/text"
            await env.type(text)
            return True, f"Typed: {text[:50]}"

        if action in ("KEY", "HOTKEY"):
            keys = value if isinstance(value, list) else [value]
            pressed_sequences: list[str] = []
            for key in keys:
                normalized = normalize_hotkey_sequence(str(key))
                if not normalized:
                    continue
                # Pass a chord spec so the Environment can hotkey multi-modifier
                # sequences instead of pressing keys sequentially.
                spec = "+".join(normalized) if len(normalized) >= 2 else normalized[0]
                await env.keypress(spec)
                pressed_sequences.append(spec)
            return True, f"Pressed: {', '.join(pressed_sequences) or value}"

        if action == "ENTER":
            await env.keypress("enter")
            return True, "Pressed Enter"

        if action in ("ESC", "ESCAPE"):
            await env.keypress("escape")
            return True, "Pressed Escape"

        if action in ("MOVE", "HOVER"):
            if not position:
                return False, "MOVE requires position"
            message = await perform_move(
                env,
                int(position[0]),
                int(position[1]),
                raw=raw,
                config=self._config,
            )
            return True, message

        if action == "DRAG":
            start = None
            for key in ("from", "start", "value"):
                current = raw.get(key)
                if isinstance(current, (list, tuple)) and len(current) >= 2:
                    start = (int(current[0]), int(current[1]))
                    break
            end = None
            for key in ("to", "end", "position"):
                current = raw.get(key)
                if isinstance(current, (list, tuple)) and len(current) >= 2:
                    end = (int(current[0]), int(current[1]))
                    break
            if start and end:
                message = await perform_drag(
                    env,
                    start,
                    end,
                    raw=raw,
                    config=self._config,
                )
                return True, message
            if position:
                message = await perform_press(
                    env,
                    int(position[0]),
                    int(position[1]),
                    raw=raw,
                    config=self._config,
                    duration=0.1,
                )
                return True, message
            return False, "DRAG requires start and end coordinates"

        if action == "SCROLL":
            val = value
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                scroll_y = int(float(val[1]))
            elif val:
                scroll_y = int(float(val))
            else:
                scroll_y = -3

            x, y = 0, 0
            if position:
                x, y = int(position[0]), int(position[1])
            # Legacy parity: the original code called ``pyautogui.scroll(-scroll_y,
            # x=x, y=y)``; pass the negated value through the Environment.
            await env.scroll(x, y, scroll_x=0, scroll_y=-scroll_y)
            direction = "up" if scroll_y > 0 else "down"
            return True, f"Scrolled {direction} {abs(scroll_y)} at ({x}, {y})"

        if action == "PRESS":
            if not position:
                return False, "PRESS requires position"
            duration = float(raw.get("duration", raw.get("hold", 1.0)))
            message = await perform_press(
                env,
                int(position[0]),
                int(position[1]),
                raw=raw,
                config=self._config,
                duration=duration,
            )
            return True, message

        if action == "WAIT":
            seconds = 2.0
            if value:
                try:
                    seconds = float(value)
                except ValueError:
                    pass
            elif raw.get("ms"):
                seconds = float(raw["ms"]) / 1000
            await asyncio.sleep(seconds)
            return True, f"Waited {seconds}s"

        if action in ("PAUSE", "CONTINUE"):
            return True, f"{action} acknowledged"

        return False, f"Unhandled action: {action}"
