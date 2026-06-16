"""Enhanced trace generator: adds semantic descriptions and spatial context.

Composes (does NOT inherit from) the original :class:`TraceGenerator`.
The base generator produces Observation/Think/Action/Expectation traces;
this wrapper enriches them with:

1. **Semantic action descriptions** — "Click the File menu" instead of
   "click(500, 30)".
2. **Action type classification** — click / drag / type / scroll / hotkey.
3. **Spatial context** — a structured description of the UI layout.

Does **not** modify the original ``trace_generator.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

    from syll.agent.aloha.act.enhanced.spatial_analyzer import SpatialAnalyzer
    from syll.agent.aloha.learn.trace_generator import TraceGenerator


class EnhancedTraceGenerator:
    """Wrapper that enriches trace outputs with semantic information.

    Usage::

        base_gen = TraceGenerator(model="gpt-4o")
        spatial = SpatialAnalyzer()
        enhanced = EnhancedTraceGenerator(base_gen, spatial)

        traces = await enhanced.generate_trace(actions, screenshots_dir)
        # Each trace now includes 'semantic_action', 'action_type', 'spatial_context'
    """

    def __init__(
        self,
        base_generator: TraceGenerator,
        spatial_analyzer: SpatialAnalyzer | None = None,
    ):
        self.base_generator = base_generator
        self.spatial_analyzer = spatial_analyzer

    async def generate_trace(
        self,
        actions_with_screenshots: list[dict],
        screenshots_dir: str | Path,
        overall_task: str = "",
    ) -> list[dict]:
        """Generate enriched traces.

        Calls the base generator first, then augments each trace with
        semantic action descriptions, action type, and optional spatial
        context.
        """
        # Delegate to base generator
        base_traces = await self.base_generator.generate_trace(
            actions_with_screenshots=actions_with_screenshots,
            screenshots_dir=screenshots_dir,
            overall_task=overall_task,
        )

        # Enrich each trace
        for i, trace in enumerate(base_traces):
            caption = trace.get("caption", {})

            # 1. Semantic action description
            caption["semantic_action"] = self._to_semantic_action(
                caption.get("action", ""),
                self._get_raw_action(actions_with_screenshots, i),
            )

            # 2. Action type classification
            caption["action_type"] = self._classify_action_type(
                self._get_raw_action(actions_with_screenshots, i)
            )

            # 3. Spatial context (if analyzer available and screenshot exists)
            if self.spatial_analyzer:
                screenshot_b64 = self._get_screenshot_b64(
                    actions_with_screenshots, i
                )
                if screenshot_b64:
                    try:
                        caption["spatial_context"] = (
                            await self.spatial_analyzer.analyze(screenshot_b64)
                        )
                    except Exception as exc:
                        logger.debug(
                            f"Spatial analysis failed for step {i + 1}: {exc}"
                        )

            trace["caption"] = caption

        return base_traces

    # ------------------------------------------------------------------
    # Action enrichment helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_semantic_action(
        caption_action: str, raw_action: dict | None
    ) -> str:
        """Convert a coordinate-based action into a semantic description.

        If the caption already describes the action semantically (which
        the base TraceGenerator usually does), return it as-is.
        Otherwise, try to build one from the raw action data.
        """
        # The base generator already produces semantic captions via VLM
        # (e.g., "Click the File menu").  If it did its job, use it.
        if caption_action and not _looks_like_coordinates(caption_action):
            return caption_action

        # Fallback: build from raw action data
        if raw_action is None:
            return caption_action

        action_type = (raw_action.get("action") or "").lower()
        description = raw_action.get("description", "")

        if description:
            return description

        # Construct from action type and coordinates
        if "drag" in action_type:
            start = raw_action.get("coordinates", [])
            end = raw_action.get("end_coordinates", [])
            if start and end:
                return f"Drag from ({start[0]},{start[1]}) to ({end[0]},{end[1]})"
            return "Perform a drag action"

        if "click" in action_type:
            return "Click at the element shown in the screenshot"

        if "type" in action_type or "input" in action_type:
            text = raw_action.get("content", "")
            if text:
                return f"Type '{text}'"
            return "Type text into the focused field"

        if "scroll" in action_type:
            return "Scroll the page"

        if "hotkey" in action_type or "key" in action_type:
            keys = raw_action.get("content", "")
            if keys:
                return f"Press keyboard shortcut: {keys}"
            return "Press a keyboard shortcut"

        return caption_action or "Perform GUI action"

    @staticmethod
    def _classify_action_type(raw_action: dict | None) -> str:
        """Classify the action type for prompt delta selection.

        Returns one of: click, drag, type, scroll, hotkey, other.
        """
        if raw_action is None:
            return "other"

        action = (raw_action.get("action") or "").lower()

        if "drag" in action:
            return "drag"
        if "dbl" in action or "double" in action:
            return "click"
        if "right" in action and "click" in action:
            return "click"
        if "click" in action:
            return "click"
        if "type" in action or "input" in action:
            return "type"
        if "scroll" in action or "wheel" in action:
            return "scroll"
        if "hotkey" in action or "key" in action:
            return "hotkey"

        return "other"

    # ------------------------------------------------------------------
    # Screenshot access helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_raw_action(
        actions: list[dict], trace_index: int
    ) -> dict | None:
        """Get the raw action dict corresponding to a trace index."""
        # Skip non-action entries (CONFIG, Active Window headers)
        action_idx = 0
        for a in actions:
            act = (a.get("action") or "").strip()
            if act.upper() == "CONFIG" or act.startswith("Active Window"):
                continue
            if action_idx == trace_index:
                return a
            action_idx += 1
        return None

    @staticmethod
    def _get_screenshot_b64(
        actions: list[dict], trace_index: int
    ) -> str | None:
        """Get the base64-encoded screenshot for a trace index."""
        import base64

        raw = EnhancedTraceGenerator._get_raw_action(actions, trace_index)
        if raw is None:
            return None

        crop = raw.get("screenshot_crop", "")
        full = raw.get("screenshot_full", "")
        path_str = crop or full

        if not path_str:
            return None

        from pathlib import Path

        path = Path(path_str)
        if not path.exists():
            return None

        try:
            data = path.read_bytes()
            return base64.b64encode(data).decode("utf-8")
        except Exception:
            return None


def _looks_like_coordinates(text: str) -> bool:
    """Heuristic: does the text look like raw coordinates?"""
    import re

    # "(500, 30)" or "[500, 30]" or "x=500, y=30"
    return bool(re.search(r"[\(\[]?\d{2,4}\s*,\s*\d{2,4}[\)\]]?", text))
