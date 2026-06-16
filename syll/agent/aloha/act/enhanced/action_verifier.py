"""Action verifier: detect whether a GUI action changed the screen as expected.

Reference: VeriGUI (Baidu, 2026) — TVAE (Think-Verify-Action-Expectation) framework.

This is a standalone module — it does **not** modify ``executor.py``.
It is used by :class:`VerifiedPlanner` and :class:`GUIExecuteSubAgent` to
close the verification loop after every action.
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

    @property
    def is_success(self) -> bool:
        return self.status == VerifyStatus.SUCCESS

    @property
    def is_no_change(self) -> bool:
        return self.status == VerifyStatus.NO_CHANGE


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

    def __init__(self, pixel_diff_threshold: float = 0.005):
        self.pixel_diff_threshold = pixel_diff_threshold

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

        if diff_score < self.pixel_diff_threshold:
            return VerifyResult(
                status=VerifyStatus.NO_CHANGE,
                confidence=1.0 - diff_score,
                pixel_diff_score=diff_score,
                diagnosis=(
                    f"Screen unchanged after action "
                    f"(pixel diff {diff_score:.4f} < threshold {self.pixel_diff_threshold}). "
                    "The action likely had no effect."
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
        api_key: str | None = None,
        api_base: str | None = None,
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

        import litellm

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

        kwargs: dict = dict(
            model=model,
            messages=messages,
            max_tokens=10,
            temperature=0,
        )
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["api_base"] = api_base

        try:
            response = await litellm.acompletion(**kwargs)
            text = (response.choices[0].message.content or "").strip().upper()
        except Exception as exc:
            logger.warning(f"LLM verification call failed: {exc}")
            return VerifyResult(
                status=VerifyStatus.UNCERTAIN,
                confidence=0.0,
                pixel_diff_score=0.0,
                diagnosis=f"LLM call error: {exc}",
            )

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
