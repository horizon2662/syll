"""Tests for GuiActionTool — the single-step gui_action primitive.

Verifies the architecture change: gui_action does EXACTLY ONE plan→actor→verify
per call (no inner retry loop), returns a structured [STEP]/[DONE] result, and
passes prior_failures structurally into the planner. The outer agent handles
iteration/retry.
"""
from __future__ import annotations

import asyncio
import base64
import pathlib
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from syll.agent.aloha.act.enhanced.action_verifier import (
    ActionVerifier,
    VerifyResult,
    VerifyStatus,
)
from syll.agent.aloha.act.enhanced.config import EnhancedConfig
from syll.agent.aloha.act.enhanced.enhanced_planner_tool import GuiActionTool
from syll.agent.aloha.act.enhanced.spatial_analyzer import SpatialAnalyzer
from syll.agent.aloha.act.enhanced.step_context import FailureCategory
from syll.agent.aloha.act.enhanced.verified_planner import VerifiedPlanner
from syll.agent.aloha.act.executor import AlohaExecutor
from syll.agent.aloha_gui_skill import AlohaSkillStore
from syll.config.loader import load_config

# 1x1 transparent PNG — a real image file _capture_observation can read.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _make_tool(tmp, **cfg_overrides):
    """GuiActionTool with all external boundaries mocked (ad-hoc, no skill)."""
    cfg = load_config()
    tool = GuiActionTool(
        cfg.tools.gui, AlohaSkillStore(cfg.workspace_path), syll_config=cfg
    )
    kw = dict(
        ALL_ENABLED=True, enable_tvae_verification=True, enable_llm_verify=False,
        enable_prompt_delta=False, enable_plan_persistence=False,
        enable_structured_memory=False, enable_semantic_trace=False,
        enable_spatial_context=False, max_consecutive_failures=3,
    )
    kw.update(cfg_overrides)
    tool._enhanced_config = EnhancedConfig(**kw)
    png = pathlib.Path(tmp) / "s.png"
    png.write_bytes(_PNG)
    tool._take_screenshot = AsyncMock(return_value=str(png))
    tool._resolve_actor_mode = MagicMock(return_value="ui-tars")
    tool._resolve_planner_endpoint = MagicMock(return_value=("m", "k", "b"))
    tool._resolve_actor_endpoint = MagicMock(return_value=("m", "k", "b"))
    tool._make_purpose_provider = MagicMock(return_value=None)
    tool._rewrite_forbidden_first_step_action = MagicMock(side_effect=lambda a, s, k: a)
    tool._flush_skill_lessons = MagicMock()
    tool._key_screenshots = MagicMock(return_value=["before", "after"])
    tool._call_actor = AsyncMock(
        return_value=({"action": "CLICK", "position": [10, 20]}, False)
    )
    tool._transform_coords = MagicMock(return_value=(100, 200))
    tool._log_action = MagicMock()
    tool._format_action_history = MagicMock(return_value="fh")
    tool._log_event = MagicMock()
    return tool


_PLAN_CLICK = {
    "Action": "click Save", "Expectation": "dialog opens", "Reasoning": "r",
    "Observation": "o", "Current Step": 1,
}
_PLAN_DONE = {
    "Action": "", "Observation": "done", "Expectation": "",
    "Current Step": 1, "Reasoning": "r",
}


def test_single_step_success_one_attempt():
    """SUCCESS path: planner/actor called EXACTLY ONCE (no inner retry even
    though max_consecutive_failures=3)."""
    with tempfile.TemporaryDirectory() as td:
        tool = _make_tool(td)
        with patch.object(VerifiedPlanner, "plan", AsyncMock(return_value=_PLAN_CLICK)) as p, \
             patch.object(AlohaExecutor, "execute", AsyncMock(return_value=(True, "executed-ok"))), \
             patch.object(ActionVerifier, "verify_pixel_diff",
                          MagicMock(return_value=VerifyResult(VerifyStatus.SUCCESS, 0.9, 0.5))):
            result = asyncio.run(tool.execute("click save"))
        assert result.text.startswith("[STEP] gui_action: SUCCESS")
        assert "Action: click Save" in result.text
        assert "Verify: SUCCESS" in result.text
        assert p.call_count == 1  # NO inner retry
        assert tool._call_actor.await_count == 1


def test_no_retry_on_verify_no_change():
    """First-call NO_CHANGE: NO inner retry, planner once. diagnose RUNS (it IS
    the light plan/grounding classifier — runs on the first failure) and
    surfaces the Category + bucket so the outer agent gets a targeted retry hint."""
    with tempfile.TemporaryDirectory() as td:
        tool = _make_tool(td, enable_llm_verify=True)
        diag_mock = AsyncMock(return_value=VerifyResult(
            VerifyStatus.NO_CHANGE, 0.0, 0.0,
            diagnosis="Save at ~(620,540)", category="COORD_OFF"))
        with patch.object(VerifiedPlanner, "plan", AsyncMock(return_value=_PLAN_CLICK)) as p, \
             patch.object(AlohaExecutor, "execute", AsyncMock(return_value=(True, "ok"))), \
             patch.object(ActionVerifier, "verify_pixel_diff",
                          MagicMock(return_value=VerifyResult(VerifyStatus.NO_CHANGE, 1.0, 0.001))), \
             patch.object(ActionVerifier, "diagnose_no_change", diag_mock):
            result = asyncio.run(tool.execute("click save"))
        assert result.text.startswith("[STEP] gui_action: NO_CHANGE")
        assert "Category: COORD_OFF" in result.text
        assert "Error type: GROUNDING" in result.text  # COORD_OFF → GROUNDING
        assert diag_mock.await_count == 1
        assert p.call_count == 1  # no inner retry


def test_plan_error_buckets_to_replan():
    """ELEMENT_ABSENT → PLAN bucket (re-plan), not GROUNDING."""
    with tempfile.TemporaryDirectory() as td:
        tool = _make_tool(td, enable_llm_verify=True)
        diag_mock = AsyncMock(return_value=VerifyResult(
            VerifyStatus.NO_CHANGE, 0.0, 0.0,
            diagnosis="target not on screen", category="ELEMENT_ABSENT"))
        with patch.object(VerifiedPlanner, "plan", AsyncMock(return_value=_PLAN_CLICK)), \
             patch.object(AlohaExecutor, "execute", AsyncMock(return_value=(True, "ok"))), \
             patch.object(ActionVerifier, "verify_pixel_diff",
                          MagicMock(return_value=VerifyResult(VerifyStatus.NO_CHANGE, 1.0, 0.001))), \
             patch.object(ActionVerifier, "diagnose_no_change", diag_mock):
            result = asyncio.run(tool.execute("click save"))
        assert "Error type: PLAN" in result.text
        assert "Re-plan" in result.text


def test_retry_runs_diagnose_with_category():
    """On retry (prior_failures passed), llm_verify re-enables → diagnose runs
    and the Category is surfaced."""
    with tempfile.TemporaryDirectory() as td:
        tool = _make_tool(td, enable_llm_verify=True)
        diag_mock = AsyncMock(return_value=VerifyResult(
            VerifyStatus.NO_CHANGE, 0.0, 0.0,
            diagnosis="Save at ~(620,540)", category="COORD_OFF"))
        with patch.object(VerifiedPlanner, "plan", AsyncMock(return_value=_PLAN_CLICK)), \
             patch.object(AlohaExecutor, "execute", AsyncMock(return_value=(True, "ok"))), \
             patch.object(ActionVerifier, "verify_pixel_diff",
                          MagicMock(return_value=VerifyResult(VerifyStatus.NO_CHANGE, 1.0, 0.001))), \
             patch.object(ActionVerifier, "diagnose_no_change", diag_mock):
            result = asyncio.run(tool.execute(
                "click save",
                prior_failures=[{"category": "UNKNOWN", "reason": "prev NO_CHANGE"}],
            ))
        assert result.text.startswith("[STEP] gui_action: NO_CHANGE")
        assert "Category: COORD_OFF" in result.text  # diagnose ran on retry
        assert diag_mock.await_count == 1


def test_spatial_analyzer_never_runs():
    """spatial_analyzer is dropped (too slow) — analyze is NEVER called, on
    first call or retry."""
    with tempfile.TemporaryDirectory() as td:
        tool = _make_tool(td, enable_spatial_context=True, enable_tvae_verification=False)
        analyze_mock = AsyncMock(return_value="spatial desc")
        with patch.object(SpatialAnalyzer, "analyze", analyze_mock), \
             patch.object(VerifiedPlanner, "plan", AsyncMock(return_value=_PLAN_CLICK)), \
             patch.object(AlohaExecutor, "execute", AsyncMock(return_value=(True, "ok"))):
            asyncio.run(tool.execute("click save"))
            asyncio.run(tool.execute(
                "click save", prior_failures=[{"category": "UNKNOWN", "reason": "r"}]
            ))
        assert analyze_mock.await_count == 0


def test_done_when_planner_signals_completion():
    """Planner Action="" → [DONE]."""
    with tempfile.TemporaryDirectory() as td:
        tool = _make_tool(td)
        with patch.object(VerifiedPlanner, "plan", AsyncMock(return_value=_PLAN_DONE)):
            result = asyncio.run(tool.execute("do thing"))
        assert result.text.startswith("[DONE] gui_action: DONE")


def test_done_when_actor_signals_finished():
    """Actor returns is_complete=True → [DONE]."""
    with tempfile.TemporaryDirectory() as td:
        tool = _make_tool(td)
        tool._call_actor = AsyncMock(
            return_value=({"action": "FINISHED", "position": [0, 0]}, True)
        )
        with patch.object(VerifiedPlanner, "plan", AsyncMock(return_value=_PLAN_CLICK)):
            result = asyncio.run(tool.execute("do thing"))
        assert result.text.startswith("[DONE] gui_action: DONE")


def test_executor_failure_returns_error_no_retry():
    """Executor fails → [STEP] ERROR, planner once (no retry)."""
    with tempfile.TemporaryDirectory() as td:
        tool = _make_tool(td)
        with patch.object(VerifiedPlanner, "plan", AsyncMock(return_value=_PLAN_CLICK)) as p, \
             patch.object(AlohaExecutor, "execute",
                          AsyncMock(return_value=(False, "pyautogui click failed"))):
            result = asyncio.run(tool.execute("click save"))
        assert result.text.startswith("[STEP] gui_action: ERROR")
        assert "Executor FAILED:" in result.text
        assert p.call_count == 1


def test_max_steps_ignored():
    """max_steps=5 must still do exactly ONE step (single-step contract)."""
    with tempfile.TemporaryDirectory() as td:
        tool = _make_tool(td)
        with patch.object(VerifiedPlanner, "plan", AsyncMock(return_value=_PLAN_CLICK)) as p, \
             patch.object(AlohaExecutor, "execute", AsyncMock(return_value=(True, "ok"))), \
             patch.object(ActionVerifier, "verify_pixel_diff",
                          MagicMock(return_value=VerifyResult(VerifyStatus.SUCCESS, 0.9, 0.5))):
            asyncio.run(tool.execute("click save", max_steps=5))
        assert p.call_count == 1


def test_prior_failures_injected_into_planner():
    """prior_failures (structured) reach the planner as FailedAttempt entries
    — same injection path the inner loop used, just routed via the caller."""
    with tempfile.TemporaryDirectory() as td:
        tool = _make_tool(td)
        with patch.object(VerifiedPlanner, "plan", AsyncMock(return_value=_PLAN_CLICK)) as p, \
             patch.object(AlohaExecutor, "execute", AsyncMock(return_value=(True, "ok"))), \
             patch.object(ActionVerifier, "verify_pixel_diff",
                          MagicMock(return_value=VerifyResult(VerifyStatus.SUCCESS, 0.9, 0.5))):
            asyncio.run(tool.execute(
                "click save",
                prior_failures=[{"category": "COORD_OFF", "reason": "Save at ~(620,540)"}],
            ))
        fa = p.call_args.kwargs.get("failed_attempts")
        assert fa is not None and len(fa) == 1
        assert fa[0].category == FailureCategory.COORD_OFF
        assert "620" in fa[0].reason
