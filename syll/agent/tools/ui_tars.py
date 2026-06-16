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
from pathlib import Path
from typing import Any

from loguru import logger

from syll.agent.events import Event, EventContent, EventSource, EventStore
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
    ):
        self._config = gui_config
        self._gui_skill_store = gui_skill_store
        self._aloha_skill_store = aloha_skill_store
        self._syll_config = syll_config
        self._screenshot_dir = Path(tempfile.gettempdir()) / "syll_gui"
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._retry = RetryConfig()
        self._event_store: EventStore | None = None
        self._model_img_size: tuple[int, int] = (0, 0)  # set by _take_screenshot
        self._monitor_launched = False

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
        """
        steps = max_steps or self._config.max_steps
        conversations: list[Conversation] = []
        screenshots: list[str] = []  # file paths for returning to user
        recent_actions: list[str] = []  # for stuck detection

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
                    conversations.extend(icl_turns)
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
                             + GUI_NO_RETRY_SUFFIX,
                    )
                screenshots.append(screenshot_path)

                # Read screenshot as base64
                with open(screenshot_path, "rb") as f:
                    screenshot_b64 = base64.b64encode(f.read()).decode()

                # Add screenshot as user turn WITH image dimensions
                conversations.append(Conversation(
                    role="user",
                    screenshot_b64=screenshot_b64,
                    screenshot_mime=self._guess_image_mime(Path(screenshot_path)),
                    img_size=self._model_img_size,  # ← inject dimensions
                ))

                # --- Call UI-TARS with retry (multi-turn conversation) ---
                response_text = await self._call_uitars_with_retry(
                    instruction, conversations
                )
                if not response_text:
                    self._monitor_write(status="error", instruction=instruction,
                                         step=step, max_steps=steps,
                                         error="模型调用失败")
                    return ToolResult(
                        text="Error: UI-TARS API call failed after retries"
                             + GUI_NO_RETRY_SUFFIX,
                        media=[screenshot_path],
                    )

                # Parse response and add as assistant turn
                thought, action_str = self._parse_response(response_text)
                conversations.append(Conversation(
                    role="assistant",
                    text=response_text,
                ))
                logger.info(f"  Thought: {thought}")
                logger.info(f"  Action: {action_str}")

                # Update monitor with current action
                self._monitor_write(
                    status="running",
                    instruction=instruction,
                    step=step,
                    max_steps=steps,
                    action=action_str,
                    thought=thought,
                )

                # --- Stuck detection ---
                recent_actions.append(action_str)
                if len(recent_actions) >= MAX_REPEAT_ACTIONS:
                    last_n = recent_actions[-MAX_REPEAT_ACTIONS:]
                    if all(a == last_n[0] for a in last_n):
                        logger.warning(f"Stuck: same action repeated {MAX_REPEAT_ACTIONS} times")
                        self._monitor_write(status="error", instruction=instruction,
                                             step=step, max_steps=steps,
                                             action=action_str, error=f"重复操作 {MAX_REPEAT_ACTIONS} 次")
                        return ToolResult(
                            text=f"GUI agent appears stuck — repeated action '{action_str}' "
                                 f"{MAX_REPEAT_ACTIONS} times. The action may not be working."
                                 + GUI_NO_RETRY_SUFFIX,
                            media=[screenshot_path],
                        )

                # Check for finished
                finished_match = re.match(r"finished\((?:content=)?['\"]?(.+?)['\"]?\)", action_str)
                if finished_match:
                    summary = finished_match.group(1)
                    key_shots = self._key_screenshots(screenshots)
                    self._monitor_write(status="finished", instruction=instruction,
                                         step=step, max_steps=steps, action=f"完成: {summary}")
                    return ToolResult(
                        text=f"GUI task completed: {summary}\n\nSteps taken: {step}",
                        media=key_shots,
                    )

                # Check for call_user
                call_user_match = re.match(r"call_user\((?:content=)?['\"]?(.+?)['\"]?\)", action_str)
                if call_user_match:
                    message = call_user_match.group(1)
                    self._monitor_write(status="running", instruction=instruction,
                                         step=step, max_steps=steps,
                                         action=f"请求人工介入: {message}")
                    return ToolResult(
                        text=f"GUI agent requests human intervention: {message}",
                        media=[screenshot_path],
                    )

                # --- Execute action with retry ---
                intent_text = "\n".join(part for part in (instruction, thought) if part)
                success, msg = await self._execute_action_with_retry(action_str, intent_text=intent_text)
                if not success:
                    self._monitor_write(status="error", instruction=instruction,
                                         step=step, max_steps=steps,
                                         action=action_str, error=msg)
                    return ToolResult(
                        text=f"Action failed at step {step}: {msg}"
                             + GUI_NO_RETRY_SUFFIX,
                        media=[screenshot_path],
                    )

                # Log GUI action event
                if self._event_store:
                    event = Event(
                        agent_type="gui_agent",
                        event_type="action",
                        source=EventSource(platform="desktop", chat_id="gui", user_id="system"),
                        content=EventContent(
                            text=f"Instruction: {instruction}\nThought: {thought}\nAction: {action_str}",
                            media=[screenshot_path],
                            metadata={
                                "step": step,
                                "action": action_str,
                                "thought": thought,
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
            return ToolResult(
                text=f"GUI task reached max steps ({steps}) without completing. "
                     f"The task may not be complete — report this to the user."
                     + GUI_NO_RETRY_SUFFIX,
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
            import litellm

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

            # Determine model string from purpose-based config
            if self._syll_config:
                ep = self._syll_config.resolve_endpoint("actor")
                model = ep.litellm_model
                api_key = ep.api_key or None
                api_base = ep.api_base
            else:
                model = "ui-tars"
                api_key = None
                api_base = None

            response = await litellm.acompletion(
                model=model,
                messages=messages,
                api_key=api_key,
                api_base=api_base,
                max_tokens=1024,
                temperature=0.05,  # lower for more deterministic coordinate output
            )

            return response.choices[0].message.content
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

        Uses AlohaTrace data (observation/think/action/expectation) when available
        for richer in-context learning.
        """
        turns: list[Conversation] = []
        for step in aloha_skill.steps:
            kf_path = self._aloha_skill_store.get_keyframe_path(
                skill_name, step.screenshot
            )
            if not kf_path:
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

            # Build assistant turn — prefer trace data, fall back to action fields
            if step.trace:
                thought = step.trace.think or step.trace.observation or ""
                action_str = step.trace.action or ""
            else:
                act = step.action
                thought = act.description or f"Execute {act.type}"
                if act.coordinates:
                    action_str = f"{act.type}(start='({act.coordinates[0]}, {act.coordinates[1]})')"
                else:
                    action_str = f"{act.type}()"

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

        Dispatches to gui_click backends (pyautogui / pynput / quartz).
        Coordinates are scaled from model image space to screen space.
        """
        import pyautogui

        try:
            # Parse action name and argument string
            match = re.match(r"(\w+)\((.*)\)", action_str.strip(), re.DOTALL)
            if not match:
                return False, f"Invalid action format: {action_str}"

            action_name = match.group(1)
            args_str = match.group(2)
            args = self._parse_action_args(args_str)

            if action_name in ("click", "left_click", "double_click"):
                start = args.get("start")
                if not start:
                    return False, f"Missing start coordinates in: {action_str}"
                x, y = self._parse_coords(start)
                sx, sy = self._scale_to_screen(x, y)

                # Check for macOS desktop app shortcut dispatch
                raw = {"intent": intent_text, "action_text": action_str}
                if should_open_desktop_app_with_shortcut(action_name, raw):
                    msg = await open_desktop_app_with_shortcut(
                        pyautogui, sx, sy, raw=raw, config=self._config,
                    )
                    return True, msg

                count = resolve_click_count(action_name, raw)
                msg = await perform_click_sequence(
                    pyautogui, sx, sy, count, raw=raw, config=self._config,
                )
                return True, msg

            elif action_name == "right_click":
                start = args.get("start")
                if not start:
                    return False, f"Missing start coordinates in: {action_str}"
                x, y = self._parse_coords(start)
                sx, sy = self._scale_to_screen(x, y)
                raw = {"intent": intent_text, "action_text": action_str}
                msg = await perform_right_click(
                    pyautogui, sx, sy, raw=raw, config=self._config,
                )
                return True, msg

            elif action_name == "drag":
                start = args.get("start")
                end = args.get("end")
                if not start or not end:
                    return False, f"Missing coordinates in: {action_str}"
                sx, sy = self._parse_coords(start)
                ex, ey = self._parse_coords(end)
                sx, sy = self._scale_to_screen(sx, sy)
                ex, ey = self._scale_to_screen(ex, ey)
                raw = {"intent": intent_text, "action_text": action_str}
                msg = await perform_drag(
                    pyautogui, (sx, sy), (ex, ey), raw=raw, config=self._config,
                )
                return True, msg

            elif action_name == "type":
                content = args.get("content", "")
                pyautogui.write(content, interval=0.05)
                return True, f"Typed: {content[:50]}"

            elif action_name == "hotkey":
                key = args.get("key", "")
                keys = normalize_hotkey_sequence(key)
                pyautogui.hotkey(*keys)
                return True, f"Pressed hotkey: {key}"

            elif action_name == "scroll":
                start = args.get("start")
                direction = args.get("direction", "down")
                amount = int(args.get("amount", "3"))
                if start:
                    x, y = self._parse_coords(start)
                    sx, sy = self._scale_to_screen(x, y)
                else:
                    sx, sy = pyautogui.position()
                scroll_amount = amount if direction == "down" else -amount
                pyautogui.scroll(scroll_amount, x=sx, y=sy)
                return True, f"Scrolled {direction} by {amount}"

            elif action_name == "wait":
                seconds = float(args.get("seconds", "2"))
                import asyncio
                await asyncio.sleep(seconds)
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
        """Parse coordinate string like '(960, 540)' into (x, y)."""
        match = re.search(r"\((\d+)\s*,\s*(\d+)\)", coord_str)
        if match:
            return int(match.group(1)), int(match.group(2))
        raise ValueError(f"Invalid coordinates: {coord_str}")

    def _scale_to_screen(self, x: int, y: int) -> tuple[int, int]:
        """Scale coordinates from model image space to screen space."""
        model_w, model_h = self._model_img_size
        if model_w == 0 or model_h == 0:
            return x, y
        import pyautogui
        screen_w, screen_h = pyautogui.size()
        sx = int(x * screen_w / model_w)
        sy = int(y * screen_h / model_h)
        return sx, sy
