"""AlohaPlannerTool: GUI automation with Planner+Actor architecture.

Uses ShowUI-Aloha's planner-actor pattern:
1. Screenshot
2. Planner: trajectory guidance + screenshot + history → next action plan
3. Actor: plan + screenshot → concrete action (UI-TARS or Claude CUA)
4. Executor: shared click backend execution
5. Repeat until Planner sets Action=null or max_steps reached
"""

import base64
import json
import platform
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from loguru import logger

from syll.agent.aloha.act.enhanced.step_context import ExecuteContext, StepContext
from syll.agent.aloha.act.executor import AlohaExecutor
from syll.agent.aloha.act.planner import AlohaPlanner
from syll.agent.aloha.act.trajectory_manager import TrajectoryManager
from syll.agent.aloha_gui_skill import AlohaSkillStore
from syll.agent.events import Event, EventContent, EventSource, EventStore
from syll.agent.gui.primitive import GuiPrimitive
from syll.agent.tools.base import Tool, ToolResult
from syll.sandbox.environment import Environment, LocalEnvironment

# Coordinate extraction — handles every format vision models emit:
# (x,y) / [x,y] / bare x,y (comma) AND <point>x y</point> (space, UI-TARS-2 /
# Qwen-VL / Doubao). Floats and signs accepted; results coerced to int.
_COORD_PAIR_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")
_POINT_TAG_RE = re.compile(
    r"<point>\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*</point>", re.IGNORECASE
)
# UI-TARS-2 / Doubao box-token form: <|box_start|>(X Y)<|box_end|> — space-
# separated, parens optional. Doubao emits this for click/drag coordinates.
_BOX_TAG_RE = re.compile(
    r"<\|box_start\|>\s*\(?\s*(-?\d+(?:\.\d+)?)[,\s]+(-?\d+(?:\.\d+)?)\s*\)?\s*<\|box_end\|>",
    re.IGNORECASE,
)


def _extract_coord_pairs(action_str: str) -> list[list[int]]:
    """All (x, y) pairs in ``action_str`` (click→1, drag→2). Handles every
    coordinate format vision models emit: ``<|box_start|>(X Y)<|box_end|>``,
    ``<point>X Y</point>``, comma-separated ``(X,Y)``/``[X,Y]``/bare, AND
    0-1 float fractions (e.g. doubao's ``(0.908, 0.762)`` → auto-scaled to
    0-1000 so coordSpace="normalized" handles them).
    Most-specific (space-separated tag forms) checked first."""
    for pattern in (_BOX_TAG_RE, _POINT_TAG_RE, _COORD_PAIR_RE):
        raw = [
            (float(m.group(1)), float(m.group(2)))
            for m in pattern.finditer(action_str)
        ]
        if raw:
            # If ALL pairs are fractional values in [0, 1] (e.g. 0.908, 0.762),
            # the model is using 0-1 normalized coords. Scale to 0-1000 so the
            # existing coordSpace="normalized" transform handles them. Genuine
            # pixel/0-1000 integer coords are unaffected (they're > 1).
            if all(
                0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
                and (x != int(x) or y != int(y))  # has a fractional part
                for x, y in raw
            ):
                raw = [(x * 1000, y * 1000) for x, y in raw]
            return [[int(x), int(y)] for x, y in raw]
    return []


class AlohaPlannerTool(Tool):
    """GUI automation with demonstration-guided Planner+Actor architecture."""

    def __init__(
        self,
        gui_config: Any,
        aloha_skill_store: AlohaSkillStore,
        syll_config: Any = None,
        environment: Environment | None = None,
    ):
        self._config = gui_config
        self._aloha_skill_store = aloha_skill_store
        self._syll_config = syll_config
        self._environment = environment or LocalEnvironment()
        self._screenshot_dir = Path(tempfile.gettempdir()) / "syll_gui_planner"
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._event_store: EventStore | None = None
        self._screen_offset: tuple[int, int] = (0, 0)
        self._model_img_size: tuple[int, int] = (0, 0)
        self._primitive = GuiPrimitive(self)

    @property
    def name(self) -> str:
        return "gui_action_planned"

    @property
    def description(self) -> str:
        return (
            "Perform GUI actions using demonstration-guided planning. "
            "Uses a Planner+Actor architecture: the Planner follows a recorded "
            "trajectory to decide the next action, then the Actor executes it. "
            "Requires a recorded Aloha skill as guidance."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "What GUI task to perform",
                },
                "skill_name": {
                    "type": "string",
                    "description": "Name of an Aloha GUI skill to use as guidance trajectory",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum steps (default: from config)",
                    "minimum": 1,
                    "maximum": 50,
                },
                "actor_mode": {
                    "type": "string",
                    "description": "Actor backend (reserved, always ui-tars)",
                    "enum": ["ui-tars"],
                },
            },
            "required": ["instruction", "skill_name"],
        }

    async def execute(
        self,
        instruction: str,
        skill_name: str,
        max_steps: int | None = None,
        actor_mode: str | None = None,
        **kwargs: Any,
    ) -> str | ToolResult:
        """Execute a GUI task using Planner+Actor loop with trajectory guidance."""
        steps_limit = max_steps or self._config.max_steps
        os_name = platform.system()

        # Load skill and trajectory
        skill = self._aloha_skill_store.load_skill(skill_name)
        if not skill:
            return ToolResult(text=f"Aloha skill '{skill_name}' not found")

        if not skill.trajectory and not skill.steps:
            return ToolResult(text=f"Skill '{skill_name}' has no trajectory or steps")

        mode = "ui-tars"  # Always use UI-TARS as the sole actor backend

        # Build trajectory guidance
        traj_manager = TrajectoryManager()
        if skill.trajectory:
            guidance = traj_manager.get_trajectory_in_context(skill.trajectory)
        else:
            guidance_steps = []
            for s in skill.steps:
                desc = ""
                if s.trace:
                    desc = s.trace.action
                elif s.action.description:
                    desc = s.action.description
                else:
                    desc = s.action.type
                guidance_steps.append(f"Step [{s.index}]: {desc}")
            guidance = "\n".join(guidance_steps)

        if not guidance:
            guidance = "(No guidance trajectory available)"

        # Resolve planner endpoint via purpose-based config
        if self._syll_config:
            ep = self._syll_config.resolve_endpoint("planner")
            planner_model = ep.litellm_model
            planner_api_key = ep.api_key or None
            planner_api_base = ep.api_base
        else:
            planner_model = "gpt-4o"
            planner_api_key = None
            planner_api_base = None

        planner = AlohaPlanner(
            model=planner_model,
            os_name=os_name,
            api_key=planner_api_key,
            api_base=planner_api_base,
        )

        # Initialize executor
        executor = AlohaExecutor(self._config, environment=self._environment)

        # Minimal step config understood by the L1 primitive (no TVAE, no prompt
        # delta).  Values are getattr-safe in the primitive as well.
        step_cfg = SimpleNamespace(
            enable_llm_verify=False,
            enable_prompt_delta=False,
            screenshot_delay_seconds=getattr(self._config, "screenshot_delay_seconds", 0.5),
        )

        screenshots: list[str] = []
        action_history: list[str] = []
        steps_log: list[dict] = []

        exec_ctx = ExecuteContext(
            cfg=step_cfg,
            skill_name=skill_name,
            instruction=instruction,
            mode=mode,
            planner_model=planner_model,
            planner=planner,
            verifier=None,
            executor=executor,
            spatial_analyzer=None,
            structured_memory=None,
            plan_manager=None,
            plan=None,
            screenshots=screenshots,
            steps_log=steps_log,
            action_history=action_history,
        )

        shot_idx = 0
        last_action_type = ""

        for step in range(1, steps_limit + 1):
            logger.info(f"Planner step {step}/{steps_limit}: {instruction}")

            step_ctx = StepContext(step=step, attempt=0)
            status, result, _, _, shot_idx = await self._primitive.execute_single_step(
                exec_ctx, step_ctx,
                guidance=guidance, skill=skill,
                failed_attempts=[],
                last_verify_result=None,
                last_action_type=last_action_type,
                actor_model=planner_model,
                os_name=os_name,
                shot_idx=shot_idx,
                instruction=instruction,
                max_steps=steps_limit,
            )

            if status == "error":
                return result
            if status == "done":
                return result

            # proceed or retry: record step and continue.  AlohaPlannerTool does
            # not run an inner retry loop; it lets the planner adapt next step.
            plan_output = step_ctx.plan_output or {}
            plan_action = step_ctx.plan_action
            plan_reasoning = plan_output.get("Reasoning", "")
            plan_observation = plan_output.get("Observation", "")
            current_step = plan_output.get("Current Step", step)
            action_dict = step_ctx.action_dict or {}
            model_position = step_ctx.model_position
            executor_position = step_ctx.executor_position
            msg = step_ctx.executor_result
            click_backend = action_dict.get("click_backend")
            mac_accessibility = action_dict.get("mac_accessibility")
            event_style = action_dict.get("event_style")
            frontmost_app = action_dict.get("frontmost_app")

            action_history.append(
                self._format_action_history(
                    plan_action,
                    msg,
                    model_position,
                    executor_position,
                    click_backend,
                    mac_accessibility,
                    event_style,
                    frontmost_app,
                )
            )

            steps_log.append({
                "step": step,
                "plan": plan_action,
                "action": action_dict,
                "reasoning": plan_reasoning,
                "observation": plan_observation,
                "executor_result": msg,
                "model_position": model_position,
                "executor_position": executor_position,
                "click_backend": click_backend,
                "mac_accessibility": mac_accessibility,
                "event_style": event_style,
                "frontmost_app": frontmost_app,
            })

            if not msg or "failed" in msg.lower():
                logger.warning(f"Action failed at step {step}: {msg}")

            if self._event_store:
                event = Event(
                    agent_type="gui_agent",
                    event_type="action",
                    source=EventSource(platform="desktop", chat_id="gui", user_id="system"),
                    content=EventContent(
                        text=f"Instruction: {instruction}\n"
                             f"Plan: {plan_action}\n"
                             f"Reasoning: {plan_reasoning}\n"
                             f"Result: {msg}",
                        media=[step_ctx.screenshot_path] if step_ctx.screenshot_path else [],
                        metadata={
                            "step": step,
                            "plan_action": plan_action,
                            "current_step": current_step,
                            "actor_mode": mode,
                            "skill_name": skill_name,
                            "model_position": model_position,
                            "executor_position": executor_position,
                            "click_backend": click_backend,
                            "mac_accessibility": mac_accessibility,
                            "event_style": event_style,
                            "frontmost_app": frontmost_app,
                        },
                    ),
                )
                self._event_store.log_event(event)

        key_shots = self._key_screenshots(screenshots)
        return ToolResult(
            text=f"Reached max steps ({steps_limit}). Task may not be complete.\n\n"
                 f"Steps log:\n{json.dumps(steps_log, indent=2)}",
            media=key_shots,
        )

    _FIRST_STEP_HIDE_HOTKEY_RE = re.compile(
        r"\b(?:cmd|command)\s*(?:\+|-|\s)\s*h\b", re.IGNORECASE
    )

    @classmethod
    def _rewrite_forbidden_first_step_action(
        cls,
        plan_action: str | None,
        step: int,
        skill: Any,
    ) -> str | None:
        """Prevent step 1 from planning Cmd+H/Command+H for demo execution."""
        if step != 1 or not plan_action:
            return plan_action
        if not cls._FIRST_STEP_HIDE_HOTKEY_RE.search(plan_action):
            return plan_action
        replacement = cls._first_guidance_action(skill)
        return replacement or plan_action

    @staticmethod
    def _first_guidance_action(skill: Any) -> str | None:
        """Extract the first concrete action text from trajectory/steps."""
        trajectory = getattr(skill, "trajectory", None) or []
        if trajectory:
            first = trajectory[0]
            caption = first.get("caption", {}) if isinstance(first, dict) else {}
            action = caption.get("action")
            if action:
                return str(action)

        steps = getattr(skill, "steps", None) or []
        if not steps:
            return None
        first_step = steps[0]
        trace = getattr(first_step, "trace", None)
        if trace and getattr(trace, "action", None):
            return str(trace.action)
        action = getattr(first_step, "action", None)
        if action and getattr(action, "description", None):
            return str(action.description)
        if action and getattr(action, "type", None):
            return str(action.type)
        return None

    @staticmethod
    def _format_action_history(
        plan_action: str | None,
        executor_result: str,
        model_position: list[int] | None = None,
        executor_position: list[int] | None = None,
        click_backend: str | None = None,
        mac_accessibility: str | None = None,
        event_style: str | None = None,
        frontmost_app: str | None = None,
    ) -> str:
        """Format planner history with model/executor coordinates plus execution diagnostics."""
        parts: list[str] = []
        if plan_action:
            parts.append(plan_action)
        if model_position is not None:
            parts.append(f"model_position={model_position}")
        if executor_position is not None:
            parts.append(f"executor_position={executor_position}")
        if click_backend:
            parts.append(f"click_backend={click_backend}")
        if mac_accessibility:
            parts.append(f"mac_accessibility={mac_accessibility}")
        if event_style:
            parts.append(f"event_style={event_style}")
        if frontmost_app:
            parts.append(f"frontmost_app={frontmost_app}")
        prefix = " | ".join(parts) if parts else "action"
        return f"{prefix} → {executor_result}"

    def _transform_coords(self, x: int, y: int, mode: str = "ui-tars") -> tuple[int, int]:
        """Transform model coordinates through the unified pipeline (UI-TARS only)."""
        try:
            from syll.agent.tools.coordinate_transform import (
                ActorSpace,
                ActorType,
                CoordinateTransformService,
            )

            workspace = getattr(self._syll_config, 'workspace_path', None)
            if not workspace:
                # Fallback to manual offset
                return x + self._screen_offset[0], y + self._screen_offset[1]
            service = CoordinateTransformService(workspace / "coord_profiles")
            selected = getattr(self._config, 'selected_screen', 0)
            ctx = service.get_frame_context(selected)
            profile = service.load_profile(selected)

            # UI-TARS: pass model image dims for reverse-scaling
            mw, mh = self._model_img_size
            coord_space = str(getattr(self._config, "coord_space", "pixel")).lower()
            actor_type = (
                ActorType.NORMALIZED
                if coord_space == "normalized"
                else ActorType.UI_TARS
            )
            actor = ActorSpace(
                actor_type=actor_type, api_width=mw, api_height=mh
            )
            result = service.model_to_executor(x, y, ctx, actor, profile)
            return result.executor_x, result.executor_y
        except Exception as e:
            logger.debug(f"Coordinate transform fallback (offset only): {e}")
            return x + self._screen_offset[0], y + self._screen_offset[1]

    async def _call_actor(
        self, mode: str, plan_action: str, screenshot_b64: str, os_name: str
    ) -> tuple[dict, bool]:
        """Route to the UI-TARS actor backend."""
        return await self._call_uitars_actor(plan_action, screenshot_b64)

    async def _call_uitars_actor(
        self, plan_action: str, screenshot_b64: str
    ) -> tuple[dict, bool]:
        """Use UI-TARS as actor: send plan action + screenshot, get concrete action."""
        # Issue 8: Load system prompt from template file, with hardcoded fallback
        system_prompt = self._load_actor_system_prompt()

        content: list[dict] = [
            {"type": "text", "text": f"Planned action: {plan_action}\n\nExecute this action:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
        ]

        # Resolve actor endpoint via purpose-based config — NO local-model
        # fallback. If syll_config is missing, error clearly instead of
        # silently calling a local ui-tars server.
        if not self._syll_config:
            raise RuntimeError(
                "Actor endpoint not configured (syll_config missing). "
                "Set models.actor in config.json — all GUI calls must go through the API."
            )
        ep = self._syll_config.resolve_endpoint("actor")
        model = ep.litellm_model
        api_key = ep.api_key or None
        api_base = ep.api_base

        actor_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        # 0b: route through a shared LLMProvider when one is attached so the
        # call is observable; fall back to a bare litellm call otherwise.
        actor_provider = getattr(self, "_actor_provider", None)
        if actor_provider is not None:
            resp = await actor_provider.chat(
                messages=actor_messages,
                model=model,
                max_tokens=512,
                temperature=0.1,
            )
            _on_actor_usage = getattr(self, "_on_actor_usage", None)
            if _on_actor_usage is not None:
                try:
                    _on_actor_usage(resp)
                except Exception:
                    pass
            if resp.finish_reason == "error" or not resp.content:
                raise RuntimeError(resp.content or "LLM provider error")
            response_text = resp.content
        else:
            import litellm

            response = await litellm.acompletion(
                model=model,
                messages=actor_messages,
                api_key=api_key,
                api_base=api_base,
                max_tokens=512,
                temperature=0.1,
            )
            response_text = response.choices[0].message.content
        action_str = self._parse_uitars_action(response_text)

        # Convert UI-TARS action string to action dict
        action_dict = self._uitars_to_action_dict(action_str)
        is_complete = action_dict.get("action") == "FINISHED"

        return action_dict, is_complete

    @staticmethod
    def _load_actor_system_prompt() -> str:
        """Load UI-TARS actor system prompt from template, with hardcoded fallback."""
        template_path = (
            Path(__file__).parent.parent / "aloha/act/prompt_templates/actor/system_ui_tars.txt"
        )
        try:
            if template_path.exists():
                return template_path.read_text(encoding="utf-8")
        except Exception:
            pass
        # Hardcoded fallback
        return (
            "You are a GUI agent. Given a planned action and a screenshot, "
            "output the exact action to execute.\n\n"
            "## Output Format\nAction: <action>\n\n"
            "## Action Space\n"
            "click(start='(x, y)')\nleft_click(start='(x, y)')\n"
            "right_click(start='(x, y)')\ndouble_click(start='(x, y)')\n"
            "drag(start='(x1, y1)', end='(x2, y2)')\n"
            "type(content='text')\nhotkey(key='ctrl+c')\n"
            "scroll(start='(x, y)', direction='up|down', amount=3)\n"
            "wait(seconds=2)\nfinished(content='summary')\n"
        )

    @staticmethod
    def _parse_uitars_action(text: str) -> str:
        """Parse action from UI-TARS response."""
        import re

        m = re.search(r"Action:\s*(.+?)$", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    @staticmethod
    def _uitars_to_action_dict(action_str: str) -> dict:
        """Convert UI-TARS action string to executor action dict."""
        import re

        if action_str.startswith("finished"):
            return {"action": "FINISHED", "value": "", "position": [0, 0]}

        if action_str.startswith("call_user"):
            return {"action": "ERROR", "value": "Needs human intervention", "position": [0, 0]}

        # Parse coordinates
        coords = _extract_coord_pairs(action_str)

        if action_str.startswith(("click(", "left_click(", "left_single(")):
            if coords:
                return {"action": "CLICK", "value": "", "position": [int(coords[0][0]), int(coords[0][1])]}

        if action_str.startswith(("right_click(", "right_single(")):
            if coords:
                return {"action": "RIGHT_CLICK", "value": "", "position": [int(coords[0][0]), int(coords[0][1])]}

        if action_str.startswith(("double_click(", "left_double(")):
            if coords:
                return {"action": "DOUBLE_CLICK", "value": "", "position": [int(coords[0][0]), int(coords[0][1])]}

        if action_str.startswith("drag("):
            if len(coords) >= 2:
                return {
                    "action": "DRAG",
                    "value": [int(coords[0][0]), int(coords[0][1])],
                    "position": [int(coords[1][0]), int(coords[1][1])],
                }

        m = re.match(r"type\((?:content|text)=['\"](.+?)['\"]\)", action_str, re.DOTALL)
        if m:
            return {"action": "TYPE", "value": m.group(1), "position": [0, 0]}

        m = re.match(r"hotkey\((?:key=)?['\"](.+?)['\"]\)", action_str)
        if m:
            return {"action": "KEY", "value": m.group(1), "position": [0, 0]}

        if action_str.startswith("scroll("):
            direction_m = re.search(r"direction=['\"](\w+)['\"]", action_str)
            direction = direction_m.group(1) if direction_m else "down"
            amount_m = re.search(r"amount=(\d+)", action_str)
            amount = int(amount_m.group(1)) if amount_m else 3
            # Issue 2: Fix scroll direction — positive=down for pyautogui
            scroll_val = -amount if direction == "up" else amount
            pos = [int(coords[0][0]), int(coords[0][1])] if coords else [0, 0]
            return {"action": "SCROLL", "value": str(scroll_val), "position": pos}

        m = re.match(r"wait\((?:seconds=)?(\d+(?:\.\d+)?)?\)", action_str)
        if m:
            return {"action": "WAIT", "value": m.group(1) or "2", "position": [0, 0]}

        # Fallback: click
        if coords:
            return {"action": "CLICK", "value": "", "position": [int(coords[0][0]), int(coords[0][1])]}

        return {"action": "ERROR", "value": f"Cannot parse: {action_str}", "position": [0, 0]}

    async def _take_screenshot(self, step: int) -> str | None:
        """Capture the current screen, DPR-resize, then scale to model target."""
        try:
            import mss
            from PIL import Image

            path = str(self._screenshot_dir / f"planner_step_{step}.png")
            with mss.mss() as sct:
                selected = getattr(self._config, 'selected_screen', 0)
                monitor_idx = selected + 1  # mss: 0=all, 1=primary, 2=secondary...
                if monitor_idx >= len(sct.monitors):
                    monitor_idx = 1  # fallback to primary
                monitor = sct.monitors[monitor_idx]
                self._screen_offset = (monitor["left"], monitor["top"])

                shot = sct.grab(monitor)
                img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                logical_w = monitor["width"]
                logical_h = monitor["height"]
                if img.width > logical_w or img.height > logical_h:
                    img = img.resize((logical_w, logical_h), Image.LANCZOS)

                # Scale to WXGA/XGA/FWXGA target for model
                model_w, model_h = self._compute_model_size(img.width, img.height)
                if model_w < img.width:
                    img = img.resize((model_w, model_h), Image.LANCZOS)
                self._model_img_size = (img.width, img.height)

                img.save(path)
            return path
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    @staticmethod
    def _compute_model_size(w: int, h: int) -> tuple[int, int]:
        """Match aspect ratio to XGA/WXGA/FWXGA target."""
        from syll.agent.tools.coordinate_transform import SCALING_TARGETS

        ratio = w / h
        for tw, th in SCALING_TARGETS.values():
            if abs(tw / th - ratio) < 0.02 and tw < w:
                return tw, th
        return (1280, 800)  # fallback WXGA

    @staticmethod
    def _key_screenshots(screenshots: list[str]) -> list[str]:
        """Return key screenshots: first + last."""
        if not screenshots:
            return []
        if len(screenshots) == 1:
            return screenshots[:]
        return [screenshots[0], screenshots[-1]]

    def _planner_label(self) -> str:
        """Label used in completion messages."""
        return "planner"

    def _log_action(self, record: dict) -> None:
        """Best-effort audit log; no-op for the base planner tool."""
        pass

    async def _finalize_plan(
        self,
        plan_manager: Any | None,
        plan: Any | None,
        structured_memory: Any | None,
        current_step_num: int,
        model: str,
    ) -> None:
        """No-op finalization for the base planner tool."""
        pass

    def _flush_skill_lessons(self, structured_memory: Any, skill_name: str) -> None:
        """No-op skill-lesson flush for the base planner tool."""
        pass

    def _monitor_write(self, **kwargs: Any) -> None:
        """No-op monitor overlay for the base planner tool."""
        pass
