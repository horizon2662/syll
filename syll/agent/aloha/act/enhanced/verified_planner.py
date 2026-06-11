"""TVAE-enhanced planner: extends AlohaPlanner with a verification protocol.

Inherits from :class:`AlohaPlanner` and overrides ``plan()`` so that:

1. The system prompt includes a TVAE verification protocol.
2. When the previous action's verification result is *NO_CHANGE*, a
   diagnostic message is injected into ``action_history`` to force the
   planner to try a different approach.
3. Optional *prompt deltas* (action-type-specific hints) and *spatial
   context* (structured UI description) can be appended to the user
   prompt.

The original ``planner.py`` is **never** modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import ChoiceLoader, FileSystemLoader
from loguru import logger

from syll.agent.aloha.act.planner import AlohaPlanner

if TYPE_CHECKING:
    from syll.agent.aloha.act.enhanced.action_verifier import VerifyResult

_PROMPT_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"
_ORIGINAL_TEMPLATES_DIR = Path(__file__).parent.parent / "prompt_templates"


class VerifiedPlanner(AlohaPlanner):
    """TVAE-enhanced planner that adds verification awareness.

    Usage::

        # Drop-in replacement for AlohaPlanner
        planner = VerifiedPlanner(model="gpt-4o")

        # Normal call (backward compatible)
        result = await planner.plan(task="...", screenshot_b64="...")

        # With verification feedback
        result = await planner.plan(
            task="...",
            screenshot_b64="...",
            previous_verify_result=verify_result,
        )
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ChoiceLoader: try enhanced templates first, fall back to
        # original ``prompt_templates/`` so that ``planner/user.txt`` and
        # ``planner/special_action_note.txt`` are always available.
        self._jinja_env.loader = ChoiceLoader([
            FileSystemLoader(str(_PROMPT_TEMPLATES_DIR)),
            FileSystemLoader(str(_ORIGINAL_TEMPLATES_DIR)),
        ])

    async def plan(
        self,
        task: str,
        guidance_trajectory: str = "",
        screenshot_b64: str = "",
        action_history: list[str] | None = None,
        # === TVAE additions (all optional → backward compatible) ===
        previous_verify_result: VerifyResult | None = None,
        spatial_context: str = "",
        action_type_hint: str = "",
        prompt_delta: str = "",
    ) -> dict:
        """Enhanced plan with TVAE verification feedback.

        All new parameters are optional.  When they are not supplied the
        method behaves identically to the parent ``AlohaPlanner.plan()``.
        """
        action_history = list(action_history) if action_history else []

        # ---- Inject verification failure feedback ----
        if previous_verify_result is not None:
            action_history = self._inject_verification_feedback(
                action_history, previous_verify_result
            )

        # ---- Build extra context to inject via action_history ----
        # The parent's plan() renders the user template internally and
        # has no hook for extra text.  Appending to action_history is
        # the safest injection point — the template concatenates every
        # entry as "step N: …" which the LLM reads as additional
        # context.
        extra_parts: list[str] = []

        if prompt_delta:
            extra_parts.append(f"--- Action-Specific Hints ---\n{prompt_delta}")
        elif action_type_hint:
            delta = self._load_prompt_delta(action_type_hint)
            if delta:
                extra_parts.append(f"--- Action-Specific Hints ---\n{delta}")

        if spatial_context:
            extra_parts.append(f"--- Current Interface Structure ---\n{spatial_context}")

        if extra_parts:
            action_history.append("\n\n".join(extra_parts))

        # ---- Delegate to parent (unchanged logic) ----
        result = await super().plan(
            task=task,
            guidance_trajectory=guidance_trajectory,
            screenshot_b64=screenshot_b64,
            action_history=action_history,
        )

        return result

    def _get_system_prompt(self, guidance_trajectory: str = "") -> str:
        """Override to use the enhanced TVAE system prompt.

        Falls back to the parent's system prompt if our template is
        missing.
        """
        try:
            return self._jinja_env.get_template(
                "verified_planner/system.txt"
            ).render(
                os_name=self.os_name,
                guidance_trajectory_example=guidance_trajectory,
            )
        except Exception:
            # Graceful fallback to parent template
            logger.debug("Verified planner template not found, using parent")
            return super()._get_system_prompt(guidance_trajectory)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_verification_feedback(
        action_history: list[str],
        verify_result: VerifyResult,
    ) -> list[str]:
        """Append a structured diagnostic message based on verify result."""

        from syll.agent.aloha.act.enhanced.action_verifier import VerifyStatus

        if verify_result.status == VerifyStatus.NO_CHANGE:
            action_history.append(
                "⚠️ VERIFICATION FAILED: The previous action had NO EFFECT "
                "on the screen (screen unchanged). "
                f"Diagnosis: {verify_result.diagnosis or 'Unknown reason.'} "
                "You MUST try a DIFFERENT approach — do NOT repeat the "
                "same action. Consider: using keyboard shortcuts, scrolling "
                "first, clicking a nearby element, or waiting for the UI to load."
            )
        elif verify_result.status == VerifyStatus.UNCERTAIN:
            action_history.append(
                "⚠️ VERIFICATION UNCERTAIN: Could not confirm whether the "
                "previous action succeeded. Observe the current screenshot "
                "carefully and proceed accordingly."
            )
        # SUCCESS → no injection needed
        return action_history

    def _load_prompt_delta(self, action_type: str) -> str:
        """Load a prompt delta file for the given action type.

        Returns an empty string if the file does not exist.
        """
        delta_dir = Path(__file__).parent / "prompt_deltas"
        delta_path = delta_dir / f"{action_type.lower()}_delta.txt"
        if delta_path.exists():
            return delta_path.read_text(encoding="utf-8").strip()
        return ""

    async def recover(
        self,
        failed_plan: dict,
        verify_result: VerifyResult,
        task: str = "",
        screenshot_b64: str = "",
        guidance_trajectory: str = "",
        action_history: list[str] | None = None,
    ) -> dict:
        """Re-plan after a verification failure.

        Injects the failure context and calls ``plan()`` again with an
        explicit instruction to try something different.
        """
        action_history = list(action_history) if action_history else []
        action_history.append(
            f"RECOVERY MODE: The action '{failed_plan.get('Action', '?')}' "
            f"failed verification. Reason: {verify_result.diagnosis}. "
            "Generate a completely different action to achieve the same goal."
        )

        return await self.plan(
            task=task or "",
            guidance_trajectory=guidance_trajectory,
            screenshot_b64=screenshot_b64,
            action_history=action_history,
        )
