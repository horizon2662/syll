"""Planner: plans the next concrete UI action using LLM with trajectory guidance.

Adapted from ShowUI-Aloha/Aloha_Act/ui_aloha/act/gui_agent/planner/ui_aloha_planner.py.
Uses litellm instead of direct OpenAI calls.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from loguru import logger

PROMPT_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"


class AlohaPlanner:
    """Plans the next UI action based on task, screenshot, trajectory guidance."""

    def __init__(
        self,
        model: str,
        os_name: str = "macOS",
        max_tokens: int = 1500,
        retry_attempts: int = 1,
        retry_delay_seconds: float = 1.0,
        api_key: str | None = None,
        api_base: str | None = None,
        provider: Any = None,
        on_usage: Any = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.os_name = os_name
        self.retry_attempts = retry_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.api_key = api_key
        self.api_base = api_base
        # 0b: when set, route model calls through a shared LLMProvider so
        # usage is observable (ContextMeter) instead of a bare litellm call.
        self.provider = provider
        self._on_usage = on_usage

        self._jinja_env = Environment(
            loader=FileSystemLoader(str(PROMPT_TEMPLATES_DIR)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    async def plan(
        self,
        task: str,
        guidance_trajectory: str = "",
        screenshot_b64: str = "",
        action_history: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Generate a plan for the next action.

        Args:
            task: The overall task description.
            guidance_trajectory: Formatted trajectory guidance string.
            screenshot_b64: Base64 encoded screenshot.
            action_history: List of previous action descriptions.

        Returns:
            Dict with Observation, Reasoning, Current Step, Action, Expectation.
        """
        system_prompt = self._get_system_prompt(guidance_trajectory)

        action_history = action_history or []
        action_history_str = ""
        if action_history:
            for i, action in enumerate(action_history):
                action_history_str += f"step {i + 1}: {action}\n"

        user_text = self._jinja_env.get_template("planner/user.txt").render(
            task=task,
            guidance_trajectory_example=guidance_trajectory,
            max_history_length=len(action_history),
            action_history_str=action_history_str,
        )

        content: list[dict] = [{"type": "text", "text": user_text}]
        if screenshot_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0,
        )
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        llm_response = await self._call_model_with_retry(kwargs)

        try:
            llm_response_json = self._extract_data(llm_response)
            parsed_dict = json.loads(llm_response_json)
            if not isinstance(parsed_dict, dict):
                raise ValueError("Not a JSON object")
        except Exception as e:
            logger.error(f"Failed to parse planner JSON: {e}")
            parsed_dict = {}

        current_step_raw = parsed_dict.get('Current Step in Guidance Trajectory')
        if isinstance(current_step_raw, str) and current_step_raw.strip():
            try:
                current_step_tuple = self._safer_parse_step_response(current_step_raw)
                parsed_dict['Current Step'] = current_step_tuple[0]
                parsed_dict['Current Step Explanation'] = current_step_tuple[1]
            except Exception:
                parsed_dict['Current Step'] = 1
                parsed_dict['Current Step Explanation'] = "Error parsing step"
        else:
            parsed_dict.setdefault('Current Step', 1)
            parsed_dict.setdefault('Current Step Explanation', "No step information")

        for field in ("Action", "Reasoning", "Observation", "Expectation"):
            parsed_dict.setdefault(field, "")

        return parsed_dict

    async def _call_model_with_retry(self, kwargs: dict) -> str:
        """Call the planner model and return its text, with a small retry budget.

        Routes through the injected ``provider`` (LLMProvider) when set — so
        the call is observable (usage → on_usage) — and falls back to a direct
        ``litellm.acompletion`` for backward compatibility.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.retry_attempts + 2):
            try:
                messages = kwargs["messages"]
                if attempt > 1:
                    messages = self._messages_with_json_reminder(messages)

                if self.provider is not None:
                    resp = await self.provider.chat(
                        messages=messages,
                        model=self.model,
                        max_tokens=kwargs.get("max_tokens", self.max_tokens),
                        temperature=kwargs.get("temperature", 0),
                    )
                    # provider.chat folds errors into finish_reason="error"
                    # rather than raising; surface them so the retry loop
                    # reacts, matching the legacy litellm behaviour.
                    if resp.finish_reason == "error" or not resp.content:
                        raise RuntimeError(resp.content or "LLM provider error")
                    if self._on_usage is not None:
                        try:
                            self._on_usage(resp)
                        except Exception:
                            pass
                    return resp.content

                import litellm

                request_kwargs = dict(kwargs)
                request_kwargs["messages"] = messages
                response = await litellm.acompletion(**request_kwargs)
                return response.choices[0].message.content
            except Exception as exc:  # pragma: no cover - provider-specific subclasses
                last_error = exc
                if attempt > self.retry_attempts:
                    break
                logger.warning(
                    f"Planner model attempt {attempt} failed: {exc}. Retrying..."
                )
                await asyncio.sleep(self.retry_delay_seconds)

        assert last_error is not None
        raise last_error

    def _get_system_prompt(self, guidance_trajectory: str = "") -> str:
        return self._jinja_env.get_template("planner/system.txt").render(
            os_name=self.os_name,
            guidance_trajectory_example=guidance_trajectory,
        )

    @staticmethod
    def _messages_with_json_reminder(messages: list[dict]) -> list[dict]:
        """Add a terse retry hint asking for raw JSON only."""
        copied = [dict(message) for message in messages]
        if copied and copied[0].get("role") == "system":
            copied[0]["content"] = (
                f"{copied[0]['content']}\n\nReturn ONLY one raw JSON object. "
                "Do not use markdown fences or any extra prose."
            )
        return copied

    @staticmethod
    def _extract_data(text: str) -> str:
        """Extract JSON block from markdown-wrapped response."""
        # Try ```json ... ``` blocks first
        m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Try raw JSON
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return m.group(0)
        return text

    @staticmethod
    def _safer_parse_step_response(s: str) -> tuple[int, str]:
        """Parse a string like '(4, explanation text)' into (int, str)."""
        s = s.strip()
        if s.startswith("(") and s.endswith(")"):
            content = s[1:-1]
        else:
            content = s

        match = re.match(r"\s*(\d+)\s*,\s*(.+)", content)
        if not match:
            raise ValueError(f"Could not parse step: {s}")

        return int(match.group(1)), match.group(2).strip().strip("'\"")
