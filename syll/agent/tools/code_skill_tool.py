"""``run_code_skill`` tool — invoke a validated code-as-policy skill.

Registered alongside ``gui_action`` so the agent can, per step, choose between
pixel actions and a reusable code-as-policy skill (Aspire). The tool refuses to
run unvalidated skills: only skills that passed the re-execution gate
(:func:`syll.sandbox... ` actually :func:`syll.agent.longhorizon.code_skill.validate_skill`)
are eligible, so the agent never auto-runs unverified generated code as a skill.
"""

from __future__ import annotations

from typing import Any

from syll.agent.longhorizon.code_skill import CodeSkillLibrary, execute_skill
from syll.agent.tools.base import Tool, ToolResult
from syll.sandbox.environment import Environment


class RunCodeSkillTool(Tool):
    """Run a validated code-as-policy skill against the attached environment."""

    def __init__(
        self,
        library: CodeSkillLibrary,
        environment: Environment | None = None,
        syll_config: Any | None = None,
    ) -> None:
        self._library = library
        self._env = environment
        self._syll_config = syll_config
        # Mirrors the auxiliary-dep attributes other GUI tools carry.
        self._event_store = None
        self._context_meter = None
        self._audit_workspace = None

    @property
    def name(self) -> str:
        return "run_code_skill"

    @property
    def description(self) -> str:
        return (
            "Run a validated code-as-policy skill by name. The skill drives the "
            "environment directly (shell/file/GUI), accomplishing a reusable "
            "sub-goal in one call. Only skills that passed the verifier gate are "
            "available. Use this when a known validated skill fits the step, "
            "instead of many fragile pixel clicks. Pass `kwargs` for any skill "
            "parameters."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the validated code-as-policy skill to run.",
                },
                "kwargs": {
                    "type": "object",
                    "description": "Optional parameters forwarded to the skill's run().",
                },
            },
            "required": ["name"],
        }

    async def execute(self, name: str = "", kwargs: dict[str, Any] | None = None, **_: Any) -> str | ToolResult:
        if self._env is None:
            return "run_code_skill has no environment attached"
        skill = self._library.load(name)
        if skill is None:
            available = [s.name for s in self._library.list()]
            return f"no code_skill named {name!r}; available: {available}"
        if not skill.validated:
            return (
                f"code_skill {name!r} is NOT validated — refusing to run. "
                "It must pass the re-execution gate first."
            )
        result = await execute_skill(skill, self._env, **(kwargs or {}))
        skill.runs += 1
        if result.get("ok"):
            skill.successes += 1
        self._library.save(skill)  # persist usage stats
        if result.get("ok"):
            return result.get("detail") or f"skill {name!r} ok"
        return f"skill {name!r} failed: {result.get('error') or result.get('detail', '')}"


__all__ = ["RunCodeSkillTool"]
