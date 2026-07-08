"""Action verifier: detect whether a GUI action changed the screen as expected.

Reference: VeriGUI (Baidu, 2026) — TVAE (Think-Verify-Action-Expectation) framework.

This is a standalone module — it does **not** modify ``executor.py``.
It is used by :class:`VerifiedPlanner` to close the verification loop
after every action.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    pass


class VerifyStatus(Enum):
    """Result of verifying a single GUI action."""

    SUCCESS = "SUCCESS"  # Screen changed (optionally matching expectation)
    NO_CHANGE = "NO_CHANGE"  # Screen unchanged → action likely failed
    UNCERTAIN = "UNCERTAIN"  # Cannot determine (e.g. tiny diff, loading)


@dataclass
class VerifyResult:
    """Structured outcome of an action verification step."""

    status: VerifyStatus
    confidence: float  # 0.0 – 1.0
    pixel_diff_score: float  # raw diff ratio [0, 1]
    diagnosis: str = ""
    # Failure category (set by diagnose_no_change; "" for pixel-diff/verify).
    # A FIELD rather than text packed into ``diagnosis``, so callers read it
    # structurally instead of parsing "[CATEGORY] reason" strings.
    category: str = ""

    @property
    def is_success(self) -> bool:
        return self.status == VerifyStatus.SUCCESS

    @property
    def is_no_change(self) -> bool:
        return self.status == VerifyStatus.NO_CHANGE


# Taxonomy of WHY a NO_CHANGE happened (SAFARI atomic-hypothesis verification +
# GUI-vs-CLI failure taxonomy). diagnose_no_change returns "[CATEGORY] reason"
# so the planner gets an actionable cause instead of the vague pixel-diff
# "likely had no effect" — different categories need different retry strategies.
NO_CHANGE_FAILURE_CATEGORIES = (
    "COORD_OFF",           # coordinate missed the target (grounding error)
    "ELEMENT_ABSENT",      # target not on screen (loading / scroll / tab)
    "OCCLUDED",            # target covered by popup/dialog/tooltip
    "LOADING",             # UI mid-transition (wait and retry)
    "NO_VISUAL_FEEDBACK",  # action likely worked, no visible change (verify otherwise)
    "WORKFLOW_ORDER",      # wrong step in the workflow (earlier/later action needed)
    "UNKNOWN",
)


class ActionVerifier:
    """Verify that a GUI action actually changed the screen.

    Two verification modes are offered:

    1. **Pixel diff** (fast, zero LLM cost): compare two screenshots pixel-
       by-pixel.  If the proportion of changed pixels is below a threshold,
       the action is considered a failure (NO_CHANGE).

    2. **LLM expectation check** (accurate, costs tokens): send the
       post-action screenshot and the planner's *Expectation* description
       to an LLM and ask whether they match.

    Typical usage::

        verifier = ActionVerifier()
        result = verifier.verify_pixel_diff(before_b64, after_b64)
        if result.is_no_change:
            # re-plan …
    """

    def __init__(self, pixel_diff_threshold: float = 0.005, provider=None, on_usage=None):
        self.pixel_diff_threshold = pixel_diff_threshold
        self.provider = provider
        self._on_usage = on_usage

    async def _call_vision_llm(
        self, messages: list[dict], *, model: str, max_tokens: int,
        temperature: float = 0,
    ) -> str | None:
        """One vision LLM call via the attached provider. Returns the text
        response, or None on error (caller picks the fallback VerifyResult).

        Consolidates the provider.chat + on_usage + finish_reason-check pattern
        previously duplicated between verify_with_expectation and
        diagnose_no_change. Provider-only (no litellm fallback) — consistent
        with the broader #5 收敛; a verifier always has a provider at 0b
        wire-up, so the litellm branch was dead code.
        """
        if self.provider is None:
            logger.warning("ActionVerifier has no provider; vision call skipped")
            return None
        try:
            resp = await self.provider.chat(
                messages=messages, model=model,
                max_tokens=max_tokens, temperature=temperature,
            )
            if self._on_usage is not None:
                try:
                    self._on_usage(resp)
                except Exception:
                    pass
            if resp.finish_reason == "error" or not resp.content:
                return None
            return (resp.content or "").strip()
        except Exception as exc:
            logger.warning(f"vision LLM call failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Pixel-diff verification
    # ------------------------------------------------------------------

    def verify_pixel_diff(
        self,
        screenshot_before_b64: str,
        screenshot_after_b64: str,
    ) -> VerifyResult:
        """Compare two base64-encoded screenshots via pixel difference.

        Uses SSIM-inspired structural comparison when Pillow is available,
        falling back to a simple per-pixel ratio otherwise.

        Returns:
            VerifyResult with status SUCCESS / NO_CHANGE.
        """
        try:
            img_before = self._decode_image(screenshot_before_b64)
            img_after = self._decode_image(screenshot_after_b64)
        except Exception as exc:
            logger.warning(f"Failed to decode screenshots for verification: {exc}")
            return VerifyResult(
                status=VerifyStatus.UNCERTAIN,
                confidence=0.0,
                pixel_diff_score=0.0,
                diagnosis=f"Screenshot decode error: {exc}",
            )

        # Ensure same dimensions — resize after to match if needed
        if img_before.size != img_after.size:
            img_after = img_after.resize(img_before.size)

        diff_score = self._compute_pixel_diff(img_before, img_after)

        # 3-way verdict. A correct focus / select / checkbox-toggle / value-
        # entered action can produce a TINY pixel change that the old binary
        # "< threshold = NO_CHANGE" misjudged as failure (triggering a retry
        # loop on a step that actually worked). Split into:
        #   - truly-zero (< floor)        → NO_CHANGE  (real failure)
        #   - small-but-nonzero (< thr)   → UNCERTAIN  (defer to outer agent's
        #                                              vision — it can see the
        #                                              focus/selection)
        #   - clear change (>= threshold) → SUCCESS
        floor = self.pixel_diff_threshold * 0.1
        if diff_score < floor:
            return VerifyResult(
                status=VerifyStatus.NO_CHANGE,
                confidence=1.0 - diff_score,
                pixel_diff_score=diff_score,
                diagnosis=(
                    f"Screen essentially unchanged after action "
                    f"(pixel diff {diff_score:.4f} < floor {floor:.4f}). "
                    "The action likely had no effect."
                ),
            )
        if diff_score < self.pixel_diff_threshold:
            return VerifyResult(
                status=VerifyStatus.UNCERTAIN,
                confidence=0.5,
                pixel_diff_score=diff_score,
                diagnosis=(
                    f"Small screen change (pixel diff {diff_score:.4f}, below "
                    f"threshold {self.pixel_diff_threshold:.4f} but above noise "
                    f"floor {floor:.4f}). The action may have worked (focus / "
                    f"select / value entered) — inspect the screenshot to confirm."
                ),
            )

        return VerifyResult(
            status=VerifyStatus.SUCCESS,
            confidence=min(diff_score, 1.0),
            pixel_diff_score=diff_score,
        )

    # ------------------------------------------------------------------
    # LLM expectation verification (optional, higher accuracy)
    # ------------------------------------------------------------------

    async def verify_with_expectation(
        self,
        screenshot_after_b64: str,
        expectation: str,
        model: str = "gpt-4o",
    ) -> VerifyResult:
        """Ask an LLM whether the current screenshot matches the *expectation*.

        This is the "V" (Verification) in TVAE: compare what the planner
        *expected* to happen against what the screenshot actually shows.

        Args:
            screenshot_after_b64: Base64-encoded screenshot taken after the
                action was executed.
            expectation: The ``Expectation`` field produced by the planner.
            model: LLM model to use for verification.
            api_key: Optional API key override.
            api_base: Optional API base override.

        Returns:
            VerifyResult with status SUCCESS / NO_CHANGE / UNCERTAIN.
        """
        if not expectation or not expectation.strip():
            return VerifyResult(
                status=VerifyStatus.UNCERTAIN,
                confidence=0.0,
                pixel_diff_score=0.0,
                diagnosis="Empty expectation — cannot verify.",
            )

        prompt = (
            "You are a GUI action verifier.  Compare the screenshot with the "
            "EXPECTED effect described below.  Reply with ONLY one word: "
            "SUCCESS, NO_CHANGE, or UNCERTAIN.\n\n"
            f"EXPECTED EFFECT:\n{expectation}\n\n"
            "Does the screenshot match this expectation?"
        )

        messages: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_after_b64}",
                        },
                    },
                ],
            },
        ]

        text = await self._call_vision_llm(messages, model=model, max_tokens=512)
        if text is None:
            return VerifyResult(
                status=VerifyStatus.UNCERTAIN,
                confidence=0.0,
                pixel_diff_score=0.0,
                diagnosis="LLM call error",
            )
        text = text.upper()

        if "SUCCESS" in text:
            return VerifyResult(
                status=VerifyStatus.SUCCESS,
                confidence=0.9,
                pixel_diff_score=0.0,
            )
        if "NO_CHANGE" in text or "UNCHANGED" in text or "FAIL" in text:
            return VerifyResult(
                status=VerifyStatus.NO_CHANGE,
                confidence=0.9,
                pixel_diff_score=0.0,
                diagnosis="LLM judged the screenshot does NOT match the expectation.",
            )

        return VerifyResult(
            status=VerifyStatus.UNCERTAIN,
            confidence=0.5,
            pixel_diff_score=0.0,
            diagnosis=f"LLM response was ambiguous: {text}",
        )

    async def diagnose_no_change(
        self,
        before_b64: str,
        after_b64: str,
        action_desc: str = "",
        expectation: str = "",
        *,
        model: str = "gpt-4o",
        wrong_change: bool = False,
    ) -> VerifyResult:
        """Diagnose WHY a verification failed — replaces the vague pixel-diff
        "likely had no effect" with an actionable category.

        Returns a NO_CHANGE VerifyResult whose ``diagnosis`` is the reason and
        ``category`` is the failure type. Different categories imply different
        retry strategies: COORD_OFF → re-ground via spatial context;
        ELEMENT_ABSENT → scroll/wait; OCCLUDED → dismiss popup;
        WORKFLOW_ORDER → the error is upstream, not in this action.

        ``wrong_change=False`` (default): pixel-diff said NO_CHANGE — the screen
        barely changed. Prompt says "screen changed almost nothing".
        ``wrong_change=True``: pixel-diff said SUCCESS but LLM verify said the
        screen changed INTO the wrong state. Prompt says "screen DID change but
        not into the expected state". The same 7 categories apply — a wrong
        click (COORD_OFF) produces a wrong-but-visible change, etc.
        """
        if wrong_change:
            situation = (
                "A GUI action ran and the screen DID change, but it did NOT "
                "change into the expected state — the action's intended effect "
                "was not achieved. "
            )
        else:
            situation = (
                "A GUI action ran but the screen changed almost nothing (pixel "
                "diff below threshold). "
            )
        prompt = (
            f"{situation}Compare the BEFORE and AFTER screenshots plus the "
            "action taken, then diagnose the MOST LIKELY reason. Reply with "
            "EXACTLY one line in this format:\n"
            "CATEGORY | one-sentence reason\n"
            "CATEGORY must be one of:\n"
            "- COORD_OFF: the coordinate missed the target element (it is elsewhere)\n"
            "- ELEMENT_ABSENT: the target is not on screen now (not loaded / needs scroll / different tab)\n"
            "- OCCLUDED: the target is covered by a popup/tooltip/dialog (dismiss first)\n"
            "- LOADING: the UI is mid-transition/loading (wait and retry)\n"
            "- NO_VISUAL_FEEDBACK: the action likely took effect but with no visible change\n"
            "- WORKFLOW_ORDER: wrong step in the workflow (an earlier/later action is needed first)\n"
            "- UNKNOWN: cannot determine\n\n"
            f"Action taken: {action_desc or '(unknown)'}\n"
            f"Expected effect: {expectation or '(none)'}"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "BEFORE screenshot:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{before_b64}"}},
                    {"type": "text", "text": "AFTER screenshot:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{after_b64}"}},
                ],
            }
        ]
        text = await self._call_vision_llm(messages, model=model, max_tokens=512)
        if text is None:
            return VerifyResult(
                VerifyStatus.NO_CHANGE, 0.0, 0.0,
                diagnosis="diagnose call error", category="UNKNOWN",
            )
        category, reason = "UNKNOWN", text[:200]
        if "|" in text:
            cat_part, _, rest = text.partition("|")
            cat = cat_part.strip().upper().replace(" ", "_").replace("-", "_")
            if cat in NO_CHANGE_FAILURE_CATEGORIES:
                category = cat
                reason = rest.strip()[:200]
        # category is a FIELD (not packed into diagnosis) — callers read it
        # structurally instead of parsing "[CATEGORY] reason".
        return VerifyResult(
            VerifyStatus.NO_CHANGE, 0.0, 0.0,
            diagnosis=reason, category=category,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_image(b64: str):
        """Decode a base64 image string into a Pillow Image."""
        from PIL import Image

        data = base64.b64decode(b64)
        return Image.open(io.BytesIO(data))

    @staticmethod
    def _compute_pixel_diff(img_before, img_after) -> float:
        """Compute a pixel-difference ratio between two same-sized images.

        Returns a float in [0, 1] where 0 = identical, 1 = completely
        different.

        Strategy: convert both to grayscale, compute absolute difference,
        count pixels above a per-pixel threshold, and return the ratio.
        """
        import numpy as np

        arr_before = np.asarray(img_before.convert("L"), dtype=np.float32)
        arr_after = np.asarray(img_after.convert("L"), dtype=np.float32)

        if arr_before.shape != arr_after.shape:
            return 1.0  # Completely different if shapes mismatch

        # Per-pixel absolute difference
        diff = np.abs(arr_before - arr_after)

        # Threshold: ignore tiny sub-pixel changes (< 10 gray levels)
        per_pixel_threshold = 10.0
        changed_pixels = np.sum(diff > per_pixel_threshold)
        total_pixels = diff.size

        if total_pixels == 0:
            return 0.0

        return float(changed_pixels) / float(total_pixels)
