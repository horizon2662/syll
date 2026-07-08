"""UI-TARS GUI automation tool.

Uses a vision-language model (UI-TARS) to interpret screenshots and
execute GUI actions via shared click backends.

Architecture follows UI-TARS-desktop patterns:
- Multi-turn conversation: each screenshot+response appended as user/assistant turns
- Sliding window: only last N screenshots kept with images, older ones text-only
- Three-layer retry (screenshot / model / execute)
- Stuck detection: if same action repeated 3 times, auto call_user
- call_user action for human intervention requests
- Key screenshots collection (first + last)
- LiteLLM for unified provider support
- GUI execution monitor overlay for real-time progress visibility
"""

import base64
import json
import mimetypes
import re
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from syll.agent.events import Event, EventContent, EventSource, EventStore
from syll.agent.gui.primitive import UITarsPrimitive
from syll.agent.gui_click import (
    normalize_hotkey_sequence,
    open_desktop_app_with_shortcut,
    perform_click_sequence,
    perform_drag,
    perform_right_click,
    resolve_click_count,
    should_open_desktop_app_with_shortcut,
)
from syll.agent.tools.base import Tool, ToolResult
from syll.sandbox.environment import Environment, LocalEnvironment

UITARS_SYSTEM_PROMPT = """You are a GUI agent. You are given a screenshot of the current screen.
You need to help the user accomplish their task by performing actions on the screen.

## Output Format

1. Output your reasoning as: Thought: <your reasoning>
2. Output one action per step as: Action: <action>

## Action Space

click(start='(x, y)')
left_click(start='(x, y)')
right_click(start='(x, y)')
double_click(start='(x, y)')
drag(start='(x1, y1)', end='(x2, y2)')
type(content='text')
hotkey(key='ctrl+c')
scroll(start='(x, y)', direction='up|down', amount=3)
wait(seconds=2)
call_user(content='message')
finished(content='summary')

## Rules

1. Use the coordinate system from the screenshot (pixel coordinates)
2. Be precise with click targets
3. Wait after actions that trigger loading
4. If you are stuck, the action is not working, or you need human help, use call_user()
5. When the task is complete, use finished() with a summary
6. If you see a permission/authorization dialog, use call_user() to ask the user

## Coordinate Guidelines

- The screenshot has a known width and height (provided with each image)
- All coordinates must be in pixels within the image bounds
- For buttons/icons: aim at the CENTER of the element, not the edge
- For text fields: click in the middle of the text area
- For checkboxes/radio buttons: click the small square/circle, not the label
- When in doubt about exact position, aim for the visual center of the target element
"""

MAX_SCREENSHOT_HISTORY = 5  # sliding window: images for last N screenshots
MAX_REPEAT_ACTIONS = 3  # if same action repeated this many times, auto call_user
GUI_NO_RETRY_SUFFIX = (
    "\n\n[IMPORTANT: This GUI task failed and cannot be retried with gui_action. "
    "Report the failure to the user and suggest alternatives. "
    "Do NOT call gui_action or gui_action_planned again for this task.]"
)


class GuiFailureKind(Enum):
    """Why a UITarsTool.execute run ended in failure.

    Determines whether the outcome is retryable (infrastructure/transient) or
    a genuine GUI deadlock the same approach will not solve — which in turn
    decides whether the GUI failure lock is written (Stage 2 ledger)."""

    SCREENSHOT_FAIL = "screenshot_fail"     # screenshot capture failed (transient)
    MODEL_CALL_FAIL = "model_call_fail"     # UI-TARS model/API call failed (transient)
    STUCK = "stuck"                         # same action repeated MAX_REPEAT_ACTIONS (genuine)
    ACTION_EXEC_FAIL = "action_exec_fail"   # action execution failed (genuine)
    MAX_STEPS = "max_steps"                 # reached max steps without finishing (genuine)

    @property
    def retryable(self) -> bool:
        # Transient/infrastructure failures: the caller may retry on a later
        # turn once the underlying issue (API param, network, screen capture)
        # is resolved. Genuine GUI failures are not retryable same-way.
        return self in (GuiFailureKind.SCREENSHOT_FAIL, GuiFailureKind.MODEL_CALL_FAIL)


def _retry_suffix(kind: GuiFailureKind) -> str:
    """The model-facing suffix to append for a failure kind.

    Transient failures get no suffix (the model may retry gui_action later);
    genuine GUI deadlocks keep the do-not-retry instruction (until the
    GuiAttemptLedger lock is cleared)."""
    return "" if kind.retryable else GUI_NO_RETRY_SUFFIX


# UI-TARS-2 / Qwen-VL / Doubao emit these action-name variants; normalize them
# to the canonical names the executor branches expect.
_UITARS_ACTION_ALIASES = {
    "left_single": "click",
    "left_double": "double_click",
    "right_single": "right_click",
}

# Adaptive resolution tiers — sorted highest→lowest per aspect ratio.
# _compute_model_size picks the highest tier that fits the screen,
# preserving maximum detail for accurate grounding.
_RESOLUTION_TIERS: dict[str, list[tuple[int, int]]] = {
    # 16:9 (most common: 4K, QHD, FHD)
    "16:9": [
        (3840, 2160),  # 4K UHD
        (2560, 1440),  # QHD   ← sweet spot for 4K screens
        (1920, 1080),  # FHD
        (1600, 900),
        (1366, 768),
    ],
    # 16:10
    "16:10": [
        (2560, 1600),
        (1920, 1200),
        (1440, 900),
        (1280, 800),
    ],
    # 4:3
    "4:3": [
        (2048, 1536),
        (1600, 1200),
        (1024, 768),
    ],
    # 3:2 (Surface etc.)
    "3:2": [
        (2256, 1504),
        (1920, 1280),
        (1440, 960),
    ],
}

# Approximate ratio → tier key mapping (tolerance ±2%)
_ASPECT_RATIOS: list[tuple[float, str]] = [
    (16 / 9,  "16:9"),
    (16 / 10, "16:10"),
    (4 / 3,   "4:3"),
    (3 / 2,   "3:2"),
]

_CONFIG_FILE = Path.home() / ".syll" / "config.json"


@dataclass
class RetryConfig:
    """Retry limits for each phase."""

    screenshot: int = 2
    model: int = 2
    execute: int = 1


@dataclass
class Conversation:
    """A single turn in the UI-TARS conversation."""

    role: str  # "user" or "assistant"
    text: str | None = None
    screenshot_b64: str | None = None  # base64 encoded screenshot
    screenshot_mime: str = "image/png"
    is_icl: bool = False  # True for in-context learning examples
    img_size: tuple[int, int] | None = None  # (width, height) for dimension injection


class UITarsTool(Tool):
    """GUI automation tool powered by UI-TARS vision-language model."""

    def __init__(
        self,
        gui_config: Any,
        gui_skill_store: Any = None,
        aloha_skill_store: Any = None,
        syll_config: Any = None,
        environment: Environment | None = None,
    ):
        self._config = gui_config
        self._gui_skill_store = gui_skill_store
        self._aloha_skill_store = aloha_skill_store
        self._syll_config = syll_config
        self._environment = environment or LocalEnvironment()
        self._screenshot_dir = Path(tempfile.gettempdir()) / "syll_gui"
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._retry = RetryConfig()
        self._event_store: EventStore | None = None
        self._model_img_size: tuple[int, int] = (0, 0)  # set by _take_screenshot
        self._monitor_launched = False
        # LLMProvider cache: route actor calls through the shared provider so
        # usage lands in ContextMeter. Requires syll_config (no litellm fallback).
        self._actor_provider = None
        if self._syll_config:
            try:
                from syll.providers.litellm_provider import LiteLLMProvider
                _ep = self._syll_config.resolve_endpoint("actor")
                if _ep.api_key:
                    self._actor_provider = LiteLLMProvider(
                        api_key=_ep.api_key, api_base=_ep.api_base
                    )
            except Exception:
                pass
        # Per-session GUI failure ledger (attached by AgentLoop each message via
        # set_session_context, like MessageTool.set_context). None in tests / the
        # subagent path → all ledger calls are skipped and behavior is as before.
        self._gui_ledger: Any = None

    def set_session_context(self, session_key: str) -> None:
        """Attach a per-session GUI failure ledger.

        Called every message by AgentLoop._wire_tool_contexts so genuine GUI
        failures are recorded as addressable, clearable state — replacing the
        un-addressable GUI_NO_RETRY_SUFFIX chat text as the lock source of truth.
        """
        from syll.agent.gui_failure_ledger import GuiAttemptLedger
        self._gui_ledger = GuiAttemptLedger(session_key)

    @property
    def name(self) -> str:
        return "gui_action"

    @property
    def description(self) -> str:
        return (
            "Perform GUI actions on the desktop screen. Takes a screenshot, sends it to "
            "UI-TARS vision model for analysis, and executes the recommended action. "
            "Use this for tasks that require interacting with desktop applications."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "What GUI action to perform, e.g. 'click the Submit button' or 'open Chrome and navigate to google.com'",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum number of steps to execute (default: from config)",
                    "minimum": 1,
                    "maximum": 50,
                },
                "skill_name": {
                    "type": "string",
                    "description": "Name of a recorded GUI skill to inject as in-context learning examples",
                },
            },
            "required": ["instruction"],
        }

    # ── Monitor helpers ────────────────────────────────────────────────

    def _monitor_write(self, **kwargs: Any) -> None:
        """Write state to GUI monitor overlay."""
        try:
            from syll.desktop.gui_monitor import write_gui_state
            write_gui_state(**kwargs)
        except Exception:
            pass  # Monitor is optional — never block execution

    def _monitor_launch(self) -> None:
        """Launch GUI monitor overlay (idempotent)."""
        if self._monitor_launched:
            return
        try:
            from syll.desktop.gui_monitor import launch_gui_monitor
            launch_gui_monitor()
            self._monitor_launched = True
        except Exception:
            pass

    def _monitor_stop(self) -> None:
        """Stop GUI monitor overlay."""
        if not self._monitor_launched:
            return
        try:
            from syll.desktop.gui_monitor import stop_gui_monitor
            stop_gui_monitor()
        except Exception:
            pass
        self._monitor_launched = False

    # ── Main execution loop ────────────────────────────────────────────

    async def execute(
        self, instruction: str, max_steps: int | None = None,
        skill_name: str | None = None, **kwargs: Any,
    ) -> str | ToolResult:
        """Execute a GUI task using multi-turn screenshot -> UI-TARS -> action loop.

        Following UI-TARS-desktop pattern: each step's screenshot and model response
        are accumulated as conversation turns, so the model has full context.
        The per-step logic is delegated to :class:`UITarsPrimitive`.
        """
        steps = max_steps or self._config.max_steps
        screenshots: list[str] = []  # file paths for returning to user

        primitive = UITarsPrimitive(
            self,
            conversations=[],
            max_repeat_actions=MAX_REPEAT_ACTIONS,
        )

        # Launch GUI monitor overlay
        self._monitor_launch()
        self._monitor_write(
            status="running",
            instruction=instruction,
            step=0,
            max_steps=steps,
        )

        try:
            # Inject ICL examples from recorded GUI skill
            if skill_name:
                icl_turns = self._build_icl_context(skill_name)
                if icl_turns:
                    primitive.add_icl_context(icl_turns)
                    logger.info(f"Injected {len(icl_turns)} ICL turns from skill '{skill_name}'")

            for step in range(1, steps + 1):
                logger.info(f"GUI step {step}/{steps}: {instruction}")

                # Update monitor
                self._monitor_write(
                    status="running",
                    instruction=instruction,
                    step=step,
                    max_steps=steps,
                )

                # --- Screenshot with retry ---
                screenshot_path = await self._take_screenshot_with_retry(step)
                if not screenshot_path:
                    self._monitor_write(status="error", instruction=instruction,
                                         step=step, max_steps=steps,
                                         error="截图失败")
                    return ToolResult(
                        text="Error: Failed to capture screenshot after retries"
                             + _retry_suffix(GuiFailureKind.SCREENSHOT_FAIL),
                    )
                screenshots.append(screenshot_path)

                # Read screenshot as base64
                with open(screenshot_path, "rb") as f:
                    screenshot_b64 = base64.b64encode(f.read()).decode()

                # --- Single UI-TARS primitive step ---
                result = await primitive.step(
                    instruction=instruction,
                    screenshot_b64=screenshot_b64,
                    screenshot_path=screenshot_path,
                    img_size=self._model_img_size,
                    step=step,
                    max_steps=steps,
                )

                if result.status == "error":
                    self._monitor_write(status="error", instruction=instruction,
                                         step=step, max_steps=steps,
                                         error="模型调用失败")
                    return ToolResult(
                        text=result.message + _retry_suffix(GuiFailureKind.MODEL_CALL_FAIL),
                        media=[screenshot_path],
                    )

                if result.status == "stuck":
                    logger.warning(f"Stuck: same action repeated {MAX_REPEAT_ACTIONS} times")
                    self._monitor_write(status="error", instruction=instruction,
                                         step=step, max_steps=steps,
                                         action=result.action, error=f"重复操作 {MAX_REPEAT_ACTIONS} 次")
                    if self._gui_ledger:
                        self._gui_ledger.record_failure(
                            instruction=instruction, kind=GuiFailureKind.STUCK,
                            reason=result.message, step=step,
                        )
                    return ToolResult(
                        text=f"GUI agent appears stuck — repeated action '{result.action}' "
                             f"{MAX_REPEAT_ACTIONS} times. The action may not be working."
                             + _retry_suffix(GuiFailureKind.STUCK),
                        media=[screenshot_path],
                    )

                if result.status == "done":
                    summary = result.summary
                    key_shots = self._key_screenshots(screenshots)
                    self._monitor_write(status="finished", instruction=instruction,
                                         step=step, max_steps=steps, action=f"完成: {summary}")
                    if self._gui_ledger:
                        self._gui_ledger.record_success(instruction=instruction)
                    return ToolResult(
                        text=f"GUI task completed: {summary}\n\nSteps taken: {step}",
                        media=key_shots,
                    )

                if result.status == "call_user":
                    message = result.message
                    self._monitor_write(status="running", instruction=instruction,
                                         step=step, max_steps=steps,
                                         action=f"请求人工介入: {message}")
                    return ToolResult(
                        text=f"GUI agent requests human intervention: {message}",
                        media=[screenshot_path],
                    )

                if result.status == "exec_fail":
                    self._monitor_write(status="error", instruction=instruction,
                                         step=step, max_steps=steps,
                                         action=result.action, error=result.message)
                    if self._gui_ledger:
                        self._gui_ledger.record_failure(
                            instruction=instruction, kind=GuiFailureKind.ACTION_EXEC_FAIL,
                            reason=result.message, step=step,
                        )
                    return ToolResult(
                        text=f"Action failed at step {step}: {result.message}"
                             + _retry_suffix(GuiFailureKind.ACTION_EXEC_FAIL),
                        media=[screenshot_path],
                    )

                # proceed
                logger.info(f"  Thought: {result.thought}")
                logger.info(f"  Action: {result.action}")
                self._monitor_write(
                    status="running",
                    instruction=instruction,
                    step=step,
                    max_steps=steps,
                    action=result.action,
                    thought=result.thought,
                )

                # Log GUI action event
                if self._event_store:
                    event = Event(
                        agent_type="gui_agent",
                        event_type="action",
                        source=EventSource(platform="desktop", chat_id="gui", user_id="system"),
                        content=EventContent(
                            text=f"Instruction: {instruction}\nThought: {result.thought}\nAction: {result.action}",
                            media=[screenshot_path],
                            metadata={
                                "step": step,
                                "action": result.action,
                                "thought": result.thought,
                                "skill_name": skill_name,
                            },
                        ),
                    )
                    self._event_store.log_event(event)

            # Max steps reached
            key_shots = self._key_screenshots(screenshots)
            self._monitor_write(status="finished", instruction=instruction,
                                 step=steps, max_steps=steps,
                                 error=f"达到最大步数 {steps}")
            if self._gui_ledger:
                self._gui_ledger.record_failure(
                    instruction=instruction, kind=GuiFailureKind.MAX_STEPS,
                    reason=f"reached max steps ({steps}) without finishing", step=steps,
                )
            return ToolResult(
                text=f"GUI task reached max steps ({steps}) without completing. "
                     f"The task may not be complete — report this to the user."
                     + _retry_suffix(GuiFailureKind.MAX_STEPS),
                media=key_shots,
            )
        except Exception as e:
            self._monitor_write(status="error", instruction=instruction,
                                 step=0, max_steps=steps, error=str(e))
            raise
        finally:
            # Monitor stays for a few seconds to show final state, then auto-hides
            pass  # monitor auto-hides via QTimer.singleShot in gui_monitor.py

    def _key_screenshots(self, screenshots: list[str]) -> list[str]:
        """Return key screenshots: first + last (deduplicated)."""
        if not screenshots:
            return []
        if len(screenshots) == 1:
            return screenshots[:]
        return [screenshots[0], screenshots[-1]]

    # ----- Retry wrappers -----

    async def _take_screenshot_with_retry(self, step: int) -> str | None:
        for attempt in range(self._retry.screenshot + 1):
            result = await self._take_screenshot(step)
            if result:
                return result
            logger.warning(f"Screenshot attempt {attempt + 1} failed, retrying...")
        return None

    async def _call_uitars_with_retry(
        self, instruction: str, conversations: list[Conversation]
    ) -> str | None:
        for attempt in range(self._retry.model + 1):
            result = await self._call_uitars(instruction, conversations)
            if result:
                return result
            logger.warning(f"UI-TARS API attempt {attempt + 1} failed, retrying...")
        return None

    async def _execute_action_with_retry(
        self, action_str: str, intent_text: str = ""
    ) -> tuple[bool, str]:
        for attempt in range(self._retry.execute + 1):
            success, msg = await self._execute_action(action_str, intent_text=intent_text)
            if success:
                return success, msg
            logger.warning(f"Action attempt {attempt + 1} failed: {msg}")
        return False, msg  # type: ignore[possibly-undefined]

    # ----- Core methods -----

    async def _take_screenshot(self, step: int) -> str | None:
        """Capture the current screen, DPR-resize, then scale to model target resolution."""
        try:
            import mss
            from PIL import Image

            path = str(self._screenshot_dir / f"step_{step}.png")
            with mss.mss() as sct:
                selected = getattr(self._config, 'selected_screen', 0)
                monitor_idx = selected + 1  # mss: 0=all, 1=primary, 2=secondary...
                if monitor_idx >= len(sct.monitors):
                    monitor_idx = 1  # fallback to primary
                monitor = sct.monitors[monitor_idx]
                shot = sct.grab(monitor)
                img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

                # DPR-aware: resize Retina screenshots to logical size
                logical_w = monitor["width"]
                logical_h = monitor["height"]
                if img.width > logical_w or img.height > logical_h:
                    img = img.resize((logical_w, logical_h), Image.LANCZOS)

                # Adaptive resolution — picks the best tier for the screen
                model_w, model_h = self._compute_model_size(img.width, img.height)
                if model_w != img.width or model_h != img.height:
                    img = img.resize((model_w, model_h), Image.LANCZOS)
                self._model_img_size = (img.width, img.height)

                logger.debug(
                    f"Screenshot: screen={img.width}x{img.height} → model={self._model_img_size}"
                )
                img.save(path)
            return path
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    def _compute_model_size(self, w: int, h: int) -> tuple[int, int]:
        """Pick the best resolution tier for the current screen.

        Adaptive strategy:
          1. Read ``tools.gui.modelResolution`` from config:
             - ``"auto"`` (default) — pick highest tier ≤ screen size
             - ``"original"``       — no scaling, keep native resolution
             - ``"2160p"`` / ``"1440p"`` / ``"1080p"`` — force a specific cap
             - ``"WIDTHxHEIGHT"``   — exact custom target (e.g. ``"2560x1440"``)
          2. Match aspect ratio, then pick the highest fitting tier.
          3. If no tier fits, compute proportional downscale to ~3.7 Mpx.

        For a 4K screen (3840×2160) the auto result is **2560×1440** (QHD),
        preserving 44 % of pixels vs only 25 % at 1080p.
        """
        # --- Config override ---
        cfg_res = self._read_model_resolution_config()
        if cfg_res == "original":
            return w, h
        if isinstance(cfg_res, tuple):
            return cfg_res

        # --- Auto: pick highest matching tier ---
        ratio = w / h
        for target_ratio, tier_key in _ASPECT_RATIOS:
            if abs(ratio - target_ratio) < 0.03:
                for tw, th in _RESOLUTION_TIERS[tier_key]:
                    if tw <= w and th <= h:
                        return tw, th
                # Screen is smaller than all tiers → keep native
                return w, h

        # --- No matching aspect ratio → proportional scale ---
        target_pixels = 3_700_000  # ≈ 2560×1440
        if w * h <= target_pixels * 1.2:
            return w, h
        scale = (target_pixels / (w * h)) ** 0.5
        return int(w * scale), int(h * scale)

    @staticmethod
    def _read_model_resolution_config() -> str | tuple[int, int] | None:
        """Read ``tools.gui.modelResolution`` from config file.

        Returns:
            - ``None`` / ``"auto"`` → use adaptive tier logic
            - ``"original"`` → no scaling
            - ``"2160p"`` / ``"1440p"`` / ``"1080p"`` → capped resolution
            - ``(w, h)`` tuple → exact custom target
        """
        try:
            if not _CONFIG_FILE.exists():
                return None
            cfg = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            val = cfg.get("tools", {}).get("gui", {}).get("modelResolution", "auto")
        except Exception:
            return None

        if not val or val == "auto":
            return None

        if val == "original":
            return "original"

        # Named presets
        _PRESETS = {
            "2160p": (3840, 2160),
            "1440p": (2560, 1440),
            "1080p": (1920, 1080),
            "768p":  (1366, 768),
        }
        if val in _PRESETS:
            return _PRESETS[val]

        # Custom "WxH" format
        m = re.match(r"(\d+)\s*[x×]\s*(\d+)", val)
        if m:
            return int(m.group(1)), int(m.group(2))

        logger.warning(f"Unknown modelResolution value: {val!r}, falling back to auto")
        return None

    async def _call_uitars(
        self,
        instruction: str,
        conversations: list[Conversation],
    ) -> str | None:
        """Call UI-TARS via LiteLLM with multi-turn conversation history.

        Following UI-TARS-desktop pattern:
        - First message includes system prompt + instruction as user text
        - Each step adds: user (screenshot image + dimensions) → assistant (thought+action)
        - Sliding window: only last N screenshots include base64 images
        """
        try:
            messages: list[dict] = []

            # System message
            messages.append({"role": "system", "content": UITARS_SYSTEM_PROMPT})

            # First user message with instruction
            messages.append({"role": "user", "content": f"Task: {instruction}"})

            # Build conversation turns with sliding window for images
            # Only include base64 images for the last MAX_SCREENSHOT_HISTORY screenshots
            # ICL turns are never evicted from the window
            user_turns_with_images = [
                i for i, c in enumerate(conversations)
                if c.role == "user" and c.screenshot_b64 and not c.is_icl
            ]
            image_start_idx = max(0, len(user_turns_with_images) - MAX_SCREENSHOT_HISTORY)
            image_turn_indices = set(user_turns_with_images[image_start_idx:])

            for i, conv in enumerate(conversations):
                if conv.role == "user" and conv.screenshot_b64:
                    if i in image_turn_indices or conv.is_icl:
                        # Build content with image + dimension hint
                        content_parts: list[dict] = [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{conv.screenshot_mime};base64,{conv.screenshot_b64}"
                                },
                            }
                        ]
                        # Inject image dimensions so the model knows the coordinate space
                        if conv.img_size:
                            iw, ih = conv.img_size
                            content_parts.append({
                                "type": "text",
                                "text": f"[Screenshot size: {iw}x{ih} pixels. Coordinates must be within (0,0)-({iw},{ih}).]",
                            })
                        messages.append({
                            "role": "user",
                            "content": content_parts,
                        })
                    else:
                        # Older screenshot — text placeholder only
                        messages.append({
                            "role": "user",
                            "content": "[Screenshot taken]",
                        })
                elif conv.role == "assistant" and conv.text:
                    messages.append({
                        "role": "assistant",
                        "content": conv.text,
                    })

            if self._syll_config:
                model = self._syll_config.resolve_endpoint("actor").litellm_model
            else:
                model = "ui-tars"

            resp = await self._actor_provider.chat(
                messages=messages, model=model,
                max_tokens=1024, temperature=0.05,
            )
            return resp.content
        except Exception as e:
            logger.error(f"UI-TARS API call failed: {e}")
            return None

    def _parse_response(self, text: str) -> tuple[str, str]:
        """Parse UI-TARS response into (thought, action)."""
        thought = ""
        action = ""

        thought_match = re.search(r"Thought:\s*(.+?)(=Action:|$)", text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        action_match = re.search(r"Action:\s*(.+?)$", text, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()

        if not action:
            # Fallback: treat entire response as action
            action = text.strip()

        return thought, action

    def _build_icl_context(self, skill_name: str) -> list[Conversation]:
        """Build ICL (in-context learning) turns from a recorded GUI skill.

        Tries AlohaSkillStore first (richer trace data), then falls back to GUISkillStore.
        """
        # Try Aloha skill store first
        if self._aloha_skill_store:
            aloha_skill = self._aloha_skill_store.load_skill(skill_name)
            if aloha_skill and aloha_skill.steps:
                return self._build_aloha_icl(aloha_skill, skill_name)

        # Fall back to original GUISkillStore
        if not self._gui_skill_store:
            logger.warning("No GUI skill store available for ICL injection")
            return []

        skill = self._gui_skill_store.load_skill(skill_name)
        if not skill or not skill.steps:
            logger.warning(f"GUI skill '{skill_name}' not found or has no steps")
            return []

        turns: list[Conversation] = []
        for step in skill.steps:
            # Load keyframe screenshot as base64
            kf_path = self._gui_skill_store.get_keyframe_path(
                skill_name, step.screenshot_before
            )
            if not kf_path:
                logger.warning(
                    f"Keyframe {step.screenshot_before} not found for skill '{skill_name}'"
                )
                continue

            kf_data = kf_path.read_bytes()
            screenshot_b64 = base64.b64encode(kf_data).decode()
            mime = self._guess_image_mime(kf_path)

            # User turn with screenshot
            turns.append(Conversation(
                role="user",
                screenshot_b64=screenshot_b64,
                screenshot_mime=mime,
                is_icl=True,
            ))

            # Assistant turn — reconstruct action in UI-TARS format
            act = step.action
            thought_text = act.description or f"Execute {act.type} action"

            if act.type in ("click", "left_click", "right_click", "double_click") and act.coordinates:
                action_str = f"{act.type}(start='({act.coordinates[0]}, {act.coordinates[1]})')"
            elif act.type == "drag" and act.coordinates and act.end_coordinates:
                action_str = (
                    f"drag(start='({act.coordinates[0]}, {act.coordinates[1]})', "
                    f"end='({act.end_coordinates[0]}, {act.end_coordinates[1]})')"
                )
            elif act.type == "type" and act.content:
                action_str = f"type(content='{act.content}')"
            elif act.type == "hotkey" and act.content:
                action_str = f"hotkey(key='{act.content}')"
            elif act.type == "scroll":
                parts = []
                if act.coordinates:
                    parts.append(f"start='({act.coordinates[0]}, {act.coordinates[1]})'")
                if act.content:
                    parts.append(f"direction='{act.content}'")
                action_str = f"scroll({', '.join(parts)})" if parts else "scroll()"
            elif act.type == "wait":
                action_str = "wait(seconds=2)"
            else:
                action_str = f"{act.type}()"

            assistant_text = f"Thought: {thought_text}\nAction: {action_str}"
            turns.append(Conversation(
                role="assistant",
                text=assistant_text,
                is_icl=True,
            ))

        return turns

    def _build_aloha_icl(self, aloha_skill: Any, skill_name: str) -> list[Conversation]:
        """Build ICL turns from an Aloha recorded skill.

        Prefers cropped keyframes (tighter framing = better grounding example)
        and reconstructs click actions with the same open/focus→double-click
        upgrade the executor applies at runtime (``resolve_click_count``), so
        the in-context examples match what the model should actually emit.
        """
        turns: list[Conversation] = []
        for step in aloha_skill.steps:
            kf_path = self._aloha_keyframe_path(skill_name, step)
            if not kf_path:
                continue

            kf_data = kf_path.read_bytes()
            screenshot_b64 = base64.b64encode(kf_data).decode()
            mime = self._guess_image_mime(kf_path)

            turns.append(Conversation(
                role="user",
                screenshot_b64=screenshot_b64,
                screenshot_mime=mime,
                is_icl=True,
            ))

            thought, action_str = self._aloha_icl_assistant_text(step)
            assistant_text = ""
            if thought:
                assistant_text += f"Thought: {thought}\n"
            assistant_text += f"Action: {action_str}"

            turns.append(Conversation(
                role="assistant",
                text=assistant_text,
                is_icl=True,
            ))

        return turns

    def _aloha_keyframe_path(self, skill_name: str, step: Any):
        """Resolve the keyframe for an ICL step: prefer the crop, fall back to
        the full screenshot (e.g. when the crop wasn't recorded)."""
        crop = getattr(step, "screenshot_crop", "") or ""
        if crop:
            path = self._aloha_skill_store.get_keyframe_path(skill_name, crop)
            if path:
                return path
        return self._aloha_skill_store.get_keyframe_path(skill_name, step.screenshot)

    def _aloha_icl_assistant_text(self, step: Any) -> tuple[str, str]:
        """Build ``(thought, action_str)`` for one ICL assistant turn.

        ``thought`` comes from the trace (think/observation) when available,
        else the action description. ``action_str`` is the trace's action used
        verbatim when it already looks like an action call; otherwise it is
        reconstructed from coordinates with the open/focus→double-click upgrade
        so examples match runtime click behavior."""
        act = step.action
        trace = step.trace
        if trace and (trace.think or trace.observation):
            thought = trace.think or trace.observation
        else:
            thought = act.description or f"Execute {act.type}"

        trace_action = (trace.action or "") if trace else ""
        if "(" in trace_action:
            return thought, trace_action
        if act.coordinates:
            return thought, (
                f"{self._icl_action_type(act)}"
                f"(start='({act.coordinates[0]}, {act.coordinates[1]})')"
            )
        return thought, f"{act.type}()"

    def _icl_action_type(self, act: Any) -> str:
        """Action type for an ICL click example, upgrading to ``double_click``
        when the description matches gui_click's open/focus/launch intent
        patterns — consistency with runtime ``resolve_click_count``."""
        if act.type in ("click", "left_click"):
            try:
                if resolve_click_count("click", {"description": act.description or ""}) >= 2:
                    return "double_click"
            except Exception:
                pass
        return act.type

    @staticmethod
    def _guess_image_mime(path: Path) -> str:
        """Guess MIME type from file extension."""
        mime, _ = mimetypes.guess_type(str(path))
        return mime or "image/png"

    # ----- Action execution -----

    async def _execute_action(
        self, action_str: str, intent_text: str = ""
    ) -> tuple[bool, str]:
        """Parse and execute a GUI action string.

        Dispatches through the configured ``Environment``. Coordinates are
        scaled from model image space to screen space.
        """
        env = self._environment
        try:
            # Parse action name and argument string
            match = re.match(r"(\w+)\((.*)\)", action_str.strip(), re.DOTALL)
            if not match:
                return False, f"Invalid action format: {action_str}"

            action_name = match.group(1)
            args_str = match.group(2)
            args = self._parse_action_args(args_str)

            # Normalize UI-TARS-2 action-name aliases (left_single/left_double/
            # right_single → canonical names the branches below expect).
            action_name = _UITARS_ACTION_ALIASES.get(action_name, action_name)

            if action_name in ("click", "left_click", "double_click"):
                start = args.get("start")
                if not start:
                    return False, f"Missing start coordinates in: {action_str}"
                x, y = self._parse_coords(start)
                sx, sy = await self._scale_to_screen(x, y)

                # Check for macOS desktop app shortcut dispatch
                raw = {"intent": intent_text, "action_text": action_str}
                if should_open_desktop_app_with_shortcut(action_name, raw):
                    msg = await open_desktop_app_with_shortcut(
                        env, sx, sy, raw=raw, config=self._config,
                    )
                    return True, msg

                count = resolve_click_count(action_name, raw)
                msg = await perform_click_sequence(
                    env, sx, sy, count, raw=raw, config=self._config,
                )
                return True, msg

            elif action_name == "right_click":
                start = args.get("start")
                if not start:
                    return False, f"Missing start coordinates in: {action_str}"
                x, y = self._parse_coords(start)
                sx, sy = await self._scale_to_screen(x, y)
                raw = {"intent": intent_text, "action_text": action_str}
                msg = await perform_right_click(
                    env, sx, sy, raw=raw, config=self._config,
                )
                return True, msg

            elif action_name == "drag":
                start = args.get("start")
                end = args.get("end")
                if not start or not end:
                    return False, f"Missing coordinates in: {action_str}"
                sx, sy = self._parse_coords(start)
                ex, ey = self._parse_coords(end)
                sx, sy = await self._scale_to_screen(sx, sy)
                ex, ey = await self._scale_to_screen(ex, ey)
                raw = {"intent": intent_text, "action_text": action_str}
                msg = await perform_drag(
                    env, (sx, sy), (ex, ey), raw=raw, config=self._config,
                )
                return True, msg

            elif action_name == "type":
                content = args.get("content", "")
                await env.type(content)
                return True, f"Typed: {content[:50]}"

            elif action_name == "hotkey":
                key = args.get("key", "")
                await env.keypress(key)
                # Report the normalized keys so diagnostics match what was dispatched.
                normalized = normalize_hotkey_sequence(key, os_name=env.os_type)
                display = "+".join(normalized) if len(normalized) >= 2 else (normalized[0] if normalized else key)
                return True, f"Pressed: {display}"

            elif action_name == "scroll":
                start = args.get("start")
                direction = args.get("direction", "down")
                amount = int(args.get("amount", "3"))
                if start:
                    x, y = self._parse_coords(start)
                    sx, sy = await self._scale_to_screen(x, y)
                else:
                    try:
                        import pyautogui
                        sx, sy = pyautogui.position()
                    except Exception:
                        sx, sy = 0, 0
                scroll_amount = amount if direction == "down" else -amount
                await env.scroll(sx, sy, scroll_x=0, scroll_y=scroll_amount)
                return True, f"Scrolled {direction} by {amount}"

            elif action_name == "wait":
                seconds = float(args.get("seconds", "2"))
                await env.wait(int(seconds * 1000))
                return True, f"Waited {seconds}s"

            elif action_name in ("call_user", "finished"):
                # Handled in the main execute loop
                return True, action_name

            else:
                return False, f"Unknown action: {action_name}"

        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False, str(e)

    # ----- Argument parsing helpers -----

    @staticmethod
    def _parse_action_args(args_str: str) -> dict[str, str]:
        """Parse named arguments from an action string.

        Handles formats like: start='(960, 540)', content='hello', direction="down"
        """
        args: dict[str, str] = {}
        # Single-quoted values
        for m in re.finditer(r"(\w+)\s*=\s*'([^']*)'", args_str):
            args[m.group(1)] = m.group(2)
        # Double-quoted values
        for m in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', args_str):
            args[m.group(1)] = m.group(2)
        return args

    @staticmethod
    def _parse_coords(coord_str: str) -> tuple[int, int]:
        """Parse a coordinate string into (x, y).

        Accepts the formats vision models actually emit:
        - ``(960, 540)`` / ``[960, 540]`` / bare ``960, 540`` — comma-separated
        - ``<point>960 540</point>`` — UI-TARS / Qwen-VL / Doubao point tags,
          SPACE-separated (no comma)
        - ints or floats, optional sign."""
        s = str(coord_str)
        # <|box_start|>(X Y)<|box_end|> tag form — UI-TARS-2 / Doubao native
        m = re.search(
            r"<\|box_start\|>\s*\(?\s*(-?\d+(?:\.\d+)?)[,\s]+(-?\d+(?:\.\d+)?)\s*\)?\s*<\|box_end\|>",
            s, re.IGNORECASE,
        )
        if m:
            return int(float(m.group(1))), int(float(m.group(2)))
        # <point>X Y</point> tag form (space-separated) — Doubao/Qwen-VL emit this
        m = re.search(
            r"<point>\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*</point>",
            s, re.IGNORECASE,
        )
        if m:
            return int(float(m.group(1))), int(float(m.group(2)))
        # comma-separated (parens / brackets / bare)
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", s)
        if m:
            return int(float(m.group(1))), int(float(m.group(2)))
        raise ValueError(f"Invalid coordinates: {coord_str}")

    async def _scale_to_screen(self, x: int, y: int) -> tuple[int, int]:
        """Scale coordinates from model image space to screen space."""
        screen_w, screen_h = await self._environment.get_screen_size()
        # Models that emit 0-1000 normalized coords regardless of the image
        # size shown (e.g. qwen3-vl) are scaled by /1000; native UI-TARS uses
        # the screenshot's own dimensions.
        if str(getattr(self._config, "coord_space", "pixel")).lower() == "normalized":
            return int(x * screen_w / 1000), int(y * screen_h / 1000)
        model_w, model_h = self._model_img_size
        if model_w == 0 or model_h == 0:
            return x, y
        sx = int(x * screen_w / model_w)
        sy = int(y * screen_h / model_h)
        return sx, sy
