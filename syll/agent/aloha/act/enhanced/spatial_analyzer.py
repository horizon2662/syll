"""Spatial semantic analyzer: extract structured UI descriptions from screenshots.

Reference: MGA (WSDM'25) — Task-Agnostic Spatial-Semantic Observer.

Provides a structured text description of the current interface layout
that can be injected into the planner's prompt, giving it a "map" of
the UI before it decides what to click.

Does **not** modify any existing code.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


class SpatialAnalyzer:
    """Extract a structured spatial-semantic description from a screenshot.

    The description covers:
    1. **Layout structure** — menu bar, toolbar, content area, status bar.
    2. **Interactive elements** — buttons, input fields, dropdowns, links.
    3. **Semantic roles** — what each element does.
    4. **Current state** — popups, loading indicators, error messages.

    This is fed to the planner as additional context so it can make
    better-informed decisions about where to click.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base

    async def analyze(self, screenshot_b64: str) -> str:
        """Analyze a screenshot and return a structured UI description.

        Uses an LLM to produce a compact (200-500 word) description
        of the interface layout, interactive elements, and current state.

        Args:
            screenshot_b64: Base64-encoded PNG screenshot.

        Returns:
            Structured text describing the UI layout.
        """
        prompt = (
            "Analyze this GUI screenshot and provide a structured description "
            "in the following format:\n\n"
            "## Layout\n"
            "Describe the major regions (menu bar, toolbar, sidebar, content "
            "area, status bar, etc.).\n\n"
            "## Interactive Elements\n"
            "List the visible interactive elements grouped by region. For each "
            "element give:\n"
            "- Type (button, input, dropdown, link, tab, checkbox, etc.)\n"
            "- Label or tooltip text (if visible)\n"
            "- Approximate position (top-left, center, bottom-right, etc.)\n\n"
            "## Current State\n"
            "- Any open dialogs, popups, or overlays?\n"
            "- Any loading indicators or error messages?\n"
            "- Which window/app is currently focused?\n\n"
            "Keep the total description under 400 words. Be precise about "
            "element labels — this will be used for GUI automation."
        )

        try:
            import litellm

            messages: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_b64}",
                            },
                        },
                    ],
                },
            ]

            kwargs: dict[str, Any] = dict(
                model=self.model,
                messages=messages,
                max_tokens=800,
                temperature=0,
            )
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["api_base"] = self.api_base

            response = await litellm.acompletion(**kwargs)
            return response.choices[0].message.content or ""

        except Exception as exc:
            logger.warning(f"Spatial analysis failed: {exc}")
            return f"(Spatial analysis unavailable: {exc})"
