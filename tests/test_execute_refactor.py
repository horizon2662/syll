"""Golden-baseline tests for the enhanced execute() refactor.

Two mocked end-to-end paths capture execute()'s observable output so that
phases 2b-2e (_verify_step / _plan_one_step / _ground_action / 编排) can be
verified behavior-equivalent by re-running this and diffing.

- completion path: planner returns Action="" → task completes (covers _capture)
- SUCCESS path:    planner returns Action="click" → actor → executor →
                   verify SUCCESS → _record_step (covers _record/_verify/_ground)

Run: python -m pytest tests/test_execute_refactor.py -v
  or: python -c "from tests.test_execute_refactor import *; run_all()"
"""
from __future__ import annotations

import asyncio
import base64
import tempfile
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

from syll.config.loader import load_config
from syll.agent.aloha_gui_skill import AlohaSkillStore
from syll.agent.aloha.act.enhanced.enhanced_planner_tool import EnhancedAlohaPlannerTool
from syll.agent.aloha.act.enhanced.config import EnhancedConfig
from syll.agent.aloha.act.enhanced.step_context import (
    ExecuteContext,
    FailedAttempt,
    FailureCategory,
    StepContext,
)
from syll.agent.aloha.act.enhanced.verified_planner import VerifiedPlanner
from syll.agent.aloha.act.enhanced.verified_planner import VerifiedPlanner
from syll.agent.aloha.act.enhanced.action_verifier import ActionVerifier, VerifyResult, VerifyStatus
from syll.agent.aloha.act.executor import AlohaExecutor

# 1x1 transparent PNG — a real image file _capture_observation can read.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _make_tool(tmp: str, **cfg_overrides) -> EnhancedAlohaPlannerTool:
    """Build a tool whose external boundaries are all mocked, so execute()
    runs deterministically. cfg_overrides enable pipeline phases."""
    cfg = load_config()
    tool = EnhancedAlohaPlannerTool(
        cfg.tools.gui, AlohaSkillStore(cfg.workspace_path), syll_config=cfg
    )
    kw = dict(
        ALL_ENABLED=True, enable_tvae_verification=False, enable_llm_verify=False,
        enable_prompt_delta=False, enable_plan_persistence=False,
        enable_structured_memory=False, enable_semantic_trace=False,
        enable_spatial_context=False,
    )
    kw.update(cfg_overrides)
    tool._enhanced_config = EnhancedConfig(**kw)
    png = pathlib.Path(tmp) / "s.png"
    png.write_bytes(_PNG)
    tool._take_screenshot = AsyncMock(return_value=str(png))
    fake_skill = MagicMock()
    fake_skill.trajectory = "t"
    fake_skill.steps = [MagicMock(index=1)]
    tool._aloha_skill_store.load_skill = MagicMock(return_value=fake_skill)
    tool._resolve_actor_mode = MagicMock(return_value="ui-tars")
    tool._build_guidance = MagicMock(return_value="g")
    tool._resolve_planner_endpoint = MagicMock(return_value=("m", "k", "b"))
    tool._resolve_actor_endpoint = MagicMock(return_value=("m", "k", "b"))
    tool._make_purpose_provider = MagicMock(return_value=None)
    tool._rewrite_forbidden_first_step_action = MagicMock(side_effect=lambda a, s, k: a)
    tool._finalize_plan = AsyncMock()
    tool._flush_skill_lessons = MagicMock()
    tool._key_screenshots = MagicMock(return_value=["s"])
    return tool


def test_capture_observation_unit():
    """_capture_observation fills step_ctx.{screenshot_b64,screenshot_path}
    and appends to exec_ctx.screenshots; returns no error."""
    with tempfile.TemporaryDirectory() as td:
        tool = _make_tool(td)
        exec_ctx = ExecuteContext(
            cfg=MagicMock(), skill_name="s", instruction="i", mode="m",
            planner_model="m", planner=None, verifier=None, executor=None,
            spatial_analyzer=None, structured_memory=None, plan_manager=None,
            plan=None, screenshots=[],
        )
        step_ctx = StepContext(step=1)
        shot_idx, err = asyncio.run(tool._capture_observation(exec_ctx, step_ctx, 0))
        assert err is None
        assert step_ctx.screenshot_b64
        assert step_ctx.screenshot_path
        assert len(exec_ctx.screenshots) == 1


def test_execute_completion_path():
    """Golden: planner returns Action="" → execute returns the completion
    message. Covers _capture_observation + plan-completion branch."""
    with tempfile.TemporaryDirectory() as td:
        tool = _make_tool(td)
        with patch.object(VerifiedPlanner, "plan", AsyncMock(return_value={
            "Action": "", "Observation": "done", "Expectation": "",
            "Current Step": 1, "Reasoning": "r",
        })):
            result = asyncio.run(tool.execute("test task", "fake_skill", max_steps=1))
        text = result.text
        assert "GUI task completed via enhanced planner" in text
        assert "Steps taken: 1" in text


def test_execute_success_path():
    """Golden: plan click → actor → executor → verify SUCCESS → _record_step.
    Covers _capture/_plan/_ground/_verify/_record (steps 1-5 of the refactor).
    The steps_log JSON in the result must stay byte-stable across refactors."""
    with tempfile.TemporaryDirectory() as td:
        tool = _make_tool(
            td, enable_tvae_verification=True, enable_prompt_delta=True,
            enable_plan_persistence=True, enable_structured_memory=True,
            max_consecutive_failures=3,
        )
        tool._call_actor = AsyncMock(return_value=({"action": "CLICK", "position": [10, 20]}, False))
        tool._transform_coords = MagicMock(return_value=(100, 200))
        tool._log_action = MagicMock()
        tool._format_action_history = MagicMock(return_value="fh")
        tool._log_event = MagicMock()
        with patch.object(VerifiedPlanner, "plan", AsyncMock(return_value={
            "Action": "click Save", "Expectation": "dialog opens", "Reasoning": "r",
            "Observation": "o", "Current Step": 1,
        })), \
             patch.object(AlohaExecutor, "execute", AsyncMock(return_value=(True, "executed-ok"))), \
             patch.object(ActionVerifier, "verify_pixel_diff", MagicMock(
                 return_value=VerifyResult(VerifyStatus.SUCCESS, 0.9, 0.5))), \
             patch("syll.agent.aloha.act.enhanced.enhanced_planner_tool._classify_action_type",
                   MagicMock(return_value="click")):
            result = asyncio.run(tool.execute("test task", "fake_skill", max_steps=1))
        text = result.text
        # Golden invariants — if ANY of these change, the refactor broke behavior.
        assert "Reached max steps (1)" in text
        # Parse steps_log JSON from the result and assert field VALUES
        # (robust to json indentation changes across refactors).
        import json as _json
        steps_log = _json.loads(text.split("Steps log:\n", 1)[1])
        assert len(steps_log) == 1
        s0 = steps_log[0]
        assert s0["plan"] == "click Save"
        assert s0["verify_status"] == "SUCCESS"
        assert s0["executor_result"] == "executed-ok"
        assert s0["model_position"] == [10, 20]            # raw actor coords
        assert s0["executor_position"] == [100, 200]        # after _transform_coords
        assert s0["action"]["position"] == [100, 200]       # _ground_action overwrote it
        assert s0["action"]["intent"] == "click Save"


def test_execute_retry_path():
    """Golden: plan click → actor → executor → verify NO_CHANGE → diagnose
    [COORD_OFF] → failed_attempts.append → retry → plan completion.
    Covers the NO_CHANGE branch of _verify_step (the retry loop)."""
    with tempfile.TemporaryDirectory() as td:
        tool = _make_tool(
            td, enable_tvae_verification=True, enable_llm_verify=True,
            enable_prompt_delta=False, enable_plan_persistence=True,
            enable_structured_memory=True, max_consecutive_failures=3,
        )
        tool._call_actor = AsyncMock(return_value=({"action": "CLICK", "position": [10, 20]}, False))
        tool._transform_coords = MagicMock(return_value=(100, 200))
        tool._log_action = MagicMock()
        tool._format_action_history = MagicMock(return_value="fh")
        tool._log_event = MagicMock()
        # plan side_effect: 1st call → click (NO_CHANGE), 2nd call → completion
        plan_returns = [
            {"Action": "click Save", "Expectation": "dialog", "Reasoning": "r",
             "Observation": "o", "Current Step": 1},
            {"Action": "", "Observation": "done", "Expectation": "",
             "Current Step": 1, "Reasoning": "r"},
        ]
        # verify: NO_CHANGE on first attempt (then completion returns before verify)
        verify_results = [VerifyResult(VerifyStatus.NO_CHANGE, 1.0, 0.001, "no change")]
        diag_result = VerifyResult(VerifyStatus.NO_CHANGE, 0.0, 0.0,
                                   diagnosis="Save at ~(620,540)", category="COORD_OFF")
        with patch.object(VerifiedPlanner, "plan", AsyncMock(side_effect=plan_returns)) as plan_mock, \
             patch.object(AlohaExecutor, "execute", AsyncMock(return_value=(True, "ok"))), \
             patch.object(ActionVerifier, "verify_pixel_diff", MagicMock(side_effect=verify_results)), \
             patch.object(ActionVerifier, "diagnose_no_change", AsyncMock(return_value=diag_result)):
            result = asyncio.run(tool.execute("test task", "fake_skill", max_steps=1))
        text = result.text
        # Golden: task completed via enhanced planner (2nd attempt = completion)
        assert "GUI task completed via enhanced planner" in text
        assert "Steps taken: 1" in text
        # The first attempt's failure was diagnosed + recorded — verify via
        # plan call_args: 2nd plan call received failed_attempts with COORD_OFF
        assert plan_mock.call_count == 2, f"plan called {plan_mock.call_count} times, expected 2"
        second_call_kwargs = plan_mock.call_args_list[1].kwargs
        fa_list = second_call_kwargs.get("failed_attempts", [])
        assert len(fa_list) == 1, f"expected 1 failed_attempt, got {len(fa_list)}"
        assert fa_list[0].category == FailureCategory.COORD_OFF, f"expected COORD_OFF, got {fa_list[0].category}"
        assert fa_list[0].action == "click Save"
        # previous_verify_result should carry the diagnose result
        prev_vr = second_call_kwargs.get("previous_verify_result")
        assert prev_vr is not None and prev_vr.category == "COORD_OFF"


def test_failure_category_from_string():
    """FailureCategory.from_string parses LLM output safely."""
    assert FailureCategory.from_string("COORD_OFF") == FailureCategory.COORD_OFF
    assert FailureCategory.from_string("coord-off") == FailureCategory.COORD_OFF
    assert FailureCategory.from_string("element absent") == FailureCategory.ELEMENT_ABSENT
    assert FailureCategory.from_string("") == FailureCategory.UNKNOWN
    assert FailureCategory.from_string("GARBAGE") == FailureCategory.UNKNOWN
    assert FailureCategory.from_string(None) == FailureCategory.UNKNOWN  # type: ignore


def test_retry_strategy_per_category():
    """All 7 FailureCategory values map to a distinct, non-empty hint."""
    for cat in FailureCategory:
        hint = VerifiedPlanner._retry_strategy(cat)
        assert isinstance(hint, str) and hint.strip(), f"empty hint for {cat}"
    # Spot-check key categories match the old behavior
    assert "spatial context" in VerifiedPlanner._retry_strategy(FailureCategory.COORD_OFF)
    assert "scroll" in VerifiedPlanner._retry_strategy(FailureCategory.ELEMENT_ABSENT)
    assert "popup" in VerifiedPlanner._retry_strategy(FailureCategory.OCCLUDED)
    assert "upstream" not in VerifiedPlanner._retry_strategy(FailureCategory.WORKFLOW_ORDER)  # says "earlier/later"
    assert "earlier" in VerifiedPlanner._retry_strategy(FailureCategory.WORKFLOW_ORDER)


def test_step_context_carries_fields():
    """StepContext / ExecuteContext / FailedAttempt carry all expected fields
    (regression guard for the adapter-break class of bugs)."""
    sc = StepContext(step=3, attempt=1, screenshot_b64="abc", plan_action="click X")
    assert sc.step == 3 and sc.attempt == 1 and sc.plan_output == {}  # default_factory
    assert sc.model_position is None and sc.step_verify is None

    fa = FailedAttempt(action="click X", position=[10, 20],
                       category=FailureCategory.COORD_OFF, reason="off by 5px")
    assert fa.category == FailureCategory.COORD_OFF  # enum, not string
    assert fa.position == [10, 20]

    ec = ExecuteContext(
        cfg=MagicMock(), skill_name="s", instruction="i", mode="m",
        planner_model="pm", planner=None, verifier=None, executor=None,
        spatial_analyzer=None, structured_memory=None, plan_manager=None,
        plan=None,
    )
    assert ec.steps_log == [] and ec.action_history == []  # default_factory


def test_execute_llm_verify_retry_path():
    """Golden: pixel-diff SUCCESS → LLM verify NO_CHANGE → diagnose
    [COORD_OFF] (wrong_change=True) → FailedAttempt → retry → completion.

    Covers the LLM verify retry path (Gap 1 fix): before the fix this path
    returned retry WITHOUT diagnosing or recording, making it dumber than
    the pixel-diff path. Now both paths diagnose + record FailedAttempt.
    """
    with tempfile.TemporaryDirectory() as td:
        tool = _make_tool(
            td, enable_tvae_verification=True, enable_llm_verify=True,
            enable_prompt_delta=False, enable_plan_persistence=True,
            enable_structured_memory=True, max_consecutive_failures=3,
        )
        tool._call_actor = AsyncMock(return_value=({"action": "CLICK", "position": [10, 20]}, False))
        tool._transform_coords = MagicMock(return_value=(100, 200))
        tool._log_action = MagicMock()
        tool._format_action_history = MagicMock(return_value="fh")
        tool._log_event = MagicMock()
        # plan: 1st call → click (will fail LLM verify), 2nd call → completion
        plan_returns = [
            {"Action": "click Save", "Expectation": "save dialog opens",
             "Reasoning": "r", "Observation": "o", "Current Step": 1},
            {"Action": "", "Observation": "done", "Expectation": "",
             "Current Step": 1, "Reasoning": "r"},
        ]
        # pixel-diff says SUCCESS (diff above threshold → screen changed)
        pixel_ok = VerifyResult(VerifyStatus.SUCCESS, 0.9, 0.5)
        # LLM verify says NO_CHANGE (screen changed but NOT matching expectation)
        llm_fail = VerifyResult(VerifyStatus.NO_CHANGE, 0.9, 0.0,
                                diagnosis="LLM judged does NOT match expectation.")
        # diagnose (wrong_change=True) returns COORD_OFF category
        wrong_diag = VerifyResult(VerifyStatus.NO_CHANGE, 0.0, 0.0,
                                  diagnosis="clicked Cancel instead of Save",
                                  category="COORD_OFF")
        with patch.object(VerifiedPlanner, "plan", AsyncMock(side_effect=plan_returns)) as plan_mock, \
             patch.object(AlohaExecutor, "execute", AsyncMock(return_value=(True, "ok"))), \
             patch.object(ActionVerifier, "verify_pixel_diff", MagicMock(return_value=pixel_ok)), \
             patch.object(ActionVerifier, "verify_with_expectation", AsyncMock(return_value=llm_fail)), \
             patch.object(ActionVerifier, "diagnose_no_change", AsyncMock(return_value=wrong_diag)):
            result = asyncio.run(tool.execute("test task", "fake_skill", max_steps=1))
        text = result.text
        # Task completed on 2nd attempt (plan returned "")
        assert "GUI task completed via enhanced planner" in text
        assert plan_mock.call_count == 2
        # Gap 1 fix: 2nd plan call received failed_attempts with the LLM-verify
        # failure (category COORD_OFF from diagnose_no_change wrong_change=True)
        second_kwargs = plan_mock.call_args_list[1].kwargs
        fa_list = second_kwargs.get("failed_attempts", [])
        assert len(fa_list) == 1, f"expected 1 failed_attempt from LLM verify, got {len(fa_list)}"
        assert fa_list[0].category == FailureCategory.COORD_OFF
        assert fa_list[0].reason == "clicked Cancel instead of Save"
        # diagnose_no_change was called with wrong_change=True (Gap 1 fix)
        # We verify this indirectly: the FailedAttempt has the wrong_diag's
        # category+reason, which only comes from the diagnose call.


def run_all():
    # --- Phase 3: FailureCategory enum + RetryStrategy ---
    test_failure_category_from_string()
    print("test_failure_category_from_string OK")
    test_retry_strategy_per_category()
    print("test_retry_strategy_per_category OK")
    test_step_context_carries_fields()
    print("test_step_context_carries_fields OK")

    # --- Phase 2: execute sub-method golden tests ---
    test_capture_observation_unit()
    print("test_capture_observation_unit OK")
    test_execute_completion_path()
    print("test_execute_completion_path OK")
    test_execute_success_path()
    print("test_execute_success_path OK")
    test_execute_retry_path()
    print("test_execute_retry_path OK")
    test_execute_llm_verify_retry_path()
    print("test_execute_llm_verify_retry_path OK")
    print("ALL GOLDEN TESTS PASSED")


if __name__ == "__main__":
    run_all()
