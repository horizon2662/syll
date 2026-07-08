"""Stage 2b integration: the GUI limiter gate reads the failure ledger.

A genuine failure locks the task (the gate blocks gui_action); clear_all()
unlocks it (the gate allows again); a transient failure never locks.
"""

from types import SimpleNamespace

from syll.agent.gui_failure_ledger import GuiAttemptLedger
from syll.agent.loop import AgentLoop
from syll.agent.tools.registry import ToolRegistry
from syll.agent.tools.ui_tars import GuiFailureKind, UITarsTool


def _gate_with_ui_tars(tmp_path):
    """Register a UITarsTool (with an isolated ledger) and return (fake_loop, ui_tars)."""
    tools = ToolRegistry()
    ui_tars = UITarsTool(SimpleNamespace(max_steps=5, selected_screen=0))
    ui_tars._gui_ledger = GuiAttemptLedger("test:1", dir_root=tmp_path)
    tools.register(ui_tars)  # registered under ui_tars.name == "gui_action"
    return SimpleNamespace(tools=tools), ui_tars


def _gui_call(instruction: str):
    return SimpleNamespace(
        name="gui_action", arguments={"instruction": instruction}, id="t1"
    )


def test_limiter_allows_when_not_locked(tmp_path):
    fake_loop, _ = _gate_with_ui_tars(tmp_path)
    pre = AgentLoop._make_gui_limiter(fake_loop)
    assert pre(_gui_call("open the app")) is None


def test_limiter_blocks_after_genuine_failure(tmp_path):
    fake_loop, ui_tars = _gate_with_ui_tars(tmp_path)
    instr = "open the app"
    ui_tars._gui_ledger.record_failure(
        instruction=instr, kind=GuiFailureKind.STUCK, reason="repeated"
    )
    assert ui_tars._gui_ledger.is_locked(instr)
    pre = AgentLoop._make_gui_limiter(fake_loop)
    msg = pre(_gui_call(instr))
    assert msg is not None
    assert "锁定" in msg  # blocked by the ledger, not the per-turn counter


def test_limiter_allows_again_after_clear_all(tmp_path):
    fake_loop, ui_tars = _gate_with_ui_tars(tmp_path)
    instr = "open the app"
    ui_tars._gui_ledger.record_failure(
        instruction=instr, kind=GuiFailureKind.STUCK, reason="repeated"
    )
    assert ui_tars._gui_ledger.is_locked(instr)
    ui_tars._gui_ledger.clear_all()
    assert not ui_tars._gui_ledger.is_locked(instr)
    pre = AgentLoop._make_gui_limiter(fake_loop)
    assert pre(_gui_call(instr)) is None


def test_limiter_ignores_transient_failure(tmp_path):
    fake_loop, ui_tars = _gate_with_ui_tars(tmp_path)
    instr = "do thing"
    # a transient (infrastructure) failure is recorded but never locks
    ui_tars._gui_ledger.record_failure(
        instruction=instr, kind=GuiFailureKind.MODEL_CALL_FAIL, reason="api err"
    )
    assert not ui_tars._gui_ledger.is_locked(instr)
    pre = AgentLoop._make_gui_limiter(fake_loop)
    assert pre(_gui_call(instr)) is None
