"""Golden baseline tests for UITarsTool._execute_action dispatch + execute() loop.

These pin CURRENT behaviour so the stage-2 decomposition (dispatch table for
_execute_action, Template-Method split of execute) can be verified as
behaviour-preserving. They complement test_ui_tars_icl.py (which covers ICL
building, the mac-backend fallback, and hotkey normalisation).
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from syll.agent.tools.ui_tars import UITarsTool


def _tool(tmp_path) -> UITarsTool:
    config = SimpleNamespace(
        max_steps=5,
        selected_screen=0,
        click_backend="pyautogui",
        mac_click_style="click",
        preflight_permissions=True,
        coord_space="pixel",
    )
    return UITarsTool(config)


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ----- _execute_action dispatch (stage 2a target) ----------------------------


@pytest.mark.anyio
async def test_execute_action_dispatches_to_click_backends(tmp_path):
    """click / right_click / drag route to their perform_* backends."""
    tool = _tool(tmp_path)
    pyautogui = MagicMock()
    pyautogui.size.return_value = (1920, 1080)

    with patch.dict(sys.modules, {"pyautogui": pyautogui}), patch(
        "syll.agent.tools.ui_tars.perform_click_sequence",
        new=AsyncMock(return_value="clicked"),
    ) as pc, patch(
        "syll.agent.tools.ui_tars.perform_right_click",
        new=AsyncMock(return_value="rclicked"),
    ) as pr, patch(
        "syll.agent.tools.ui_tars.perform_drag",
        new=AsyncMock(return_value="dragged"),
    ) as pd, patch(
        "syll.agent.tools.ui_tars.should_open_desktop_app_with_shortcut",
        return_value=False,
    ), patch("syll.agent.tools.ui_tars.resolve_click_count", return_value=1), patch(
        "syll.agent.gui_click.asyncio.sleep", new=AsyncMock()
    ):
        ok, msg = await tool._execute_action("click(start='(10, 20)')", "click here")
        assert ok and msg == "clicked"
        pc.assert_awaited_once()

        ok, msg = await tool._execute_action("right_click(start='(10, 20)')", "rc")
        assert ok and msg == "rclicked"
        pr.assert_awaited_once()

        ok, msg = await tool._execute_action(
            "drag(start='(10, 20)', end='(30, 40)')", "drag it"
        )
        assert ok and msg == "dragged"
        pd.assert_awaited_once()


@pytest.mark.anyio
async def test_execute_action_drives_pyautogui_directly(tmp_path):
    """type / hotkey / scroll drive pyautogui directly (no perform_* backend)."""
    tool = _tool(tmp_path)
    pyautogui = MagicMock()
    pyautogui.size.return_value = (1920, 1080)

    with patch.dict(sys.modules, {"pyautogui": pyautogui}), patch(
        "syll.agent.gui_click.asyncio.sleep", new=AsyncMock()
    ):
        ok, msg = await tool._execute_action("type(content='hello')", "type text")
        assert ok and msg.startswith("Typed")
        pyautogui.write.assert_called_once_with("hello", interval=0.05)

        ok, msg = await tool._execute_action("hotkey(key='ctrl+c')", "copy")
        assert ok and msg == "Pressed: ctrl+c"
        pyautogui.hotkey.assert_called_once_with("ctrl", "c")

        ok, msg = await tool._execute_action(
            "scroll(start='(10, 20)', direction='down', amount=3)", "scroll"
        )
        assert ok and msg == "Scrolled down by 3"
        pyautogui.scroll.assert_called_once_with(3, x=10, y=20)


@pytest.mark.anyio
async def test_execute_action_finished_and_invalid(tmp_path):
    """finished() is a no-op pass-through; malformed input returns False."""
    tool = _tool(tmp_path)
    pyautogui = MagicMock()
    with patch.dict(sys.modules, {"pyautogui": pyautogui}):
        ok, msg = await tool._execute_action("finished()", "done")
        assert ok and msg == "finished"

        ok, msg = await tool._execute_action("not an action", "x")
        assert ok is False
        assert "Invalid action format" in msg


# ----- execute() loop (stage 2b target) --------------------------------------


def _patch_execute_internals(tool, shot_path, response_text):
    """Patch the screenshot/model/monitor dependencies of execute()."""
    return [
        patch.object(tool, "_take_screenshot_with_retry",
                     new=AsyncMock(return_value=str(shot_path))),
        patch.object(tool, "_call_uitars_with_retry",
                     new=AsyncMock(return_value=response_text)),
        patch.object(tool, "_monitor_launch"),
        patch.object(tool, "_monitor_write"),
    ]


@pytest.mark.anyio
async def test_execute_completes_on_finished(tmp_path):
    tool = _tool(tmp_path)
    shot = tmp_path / "step_1.png"
    shot.write_bytes(b"\x89PNG fake")

    patches = _patch_execute_internals(
        tool, shot, "Thought: all done\nAction: finished(content='all set')"
    )
    with patches[0], patches[1], patches[2], patches[3]:
        result = await tool.execute("do the thing", max_steps=3)

    assert "completed" in result.text
    assert "all set" in result.text


@pytest.mark.anyio
async def test_execute_detects_stuck_after_repeats(tmp_path):
    tool = _tool(tmp_path)
    shot = tmp_path / "step.png"
    shot.write_bytes(b"\x89PNG fake")

    patches = _patch_execute_internals(
        tool, shot, "Action: click(start='(1, 2)')"
    )
    with patches[0], patches[1], patches[2], patches[3], patch.object(
        tool, "_execute_action_with_retry", new=AsyncMock(return_value=(True, "ok"))
    ):
        result = await tool.execute("do the thing", max_steps=5)

    assert "stuck" in result.text.lower()
    assert "repeated" in result.text.lower()


@pytest.mark.anyio
async def test_execute_surfaces_call_user(tmp_path):
    tool = _tool(tmp_path)
    shot = tmp_path / "step.png"
    shot.write_bytes(b"\x89PNG fake")

    patches = _patch_execute_internals(
        tool, shot, "Action: call_user(content='need a human')"
    )
    with patches[0], patches[1], patches[2], patches[3]:
        result = await tool.execute("do the thing", max_steps=3)

    assert "human intervention" in result.text
    assert "need a human" in result.text


@pytest.mark.anyio
async def test_execute_action_types_non_ascii_via_clipboard(tmp_path):
    """Non-ASCII text is pasted via the clipboard — pyautogui.write can only
    press characters in the active keyboard layout, so CJK must be pasted."""
    tool = _tool(tmp_path)
    pyautogui = MagicMock()
    pyperclip = MagicMock()

    with patch.dict(sys.modules, {"pyautogui": pyautogui, "pyperclip": pyperclip}), patch(
        "syll.agent.gui_click.platform.system", return_value="Windows"
    ):
        ok, msg = await tool._execute_action("type(content='你好世界')", "type chinese")

    assert ok and msg.startswith("Typed")
    pyperclip.copy.assert_called_once_with("你好世界")
    pyautogui.hotkey.assert_called_once_with("ctrl", "v")
    pyautogui.write.assert_not_called()


def test_parse_coords_accepts_model_emit_variants():
    """_parse_coords must accept every coordinate format vision models emit —
    parenthesized ints, floats, brackets, bare, signed — not just (int, int).
    Regression for the 'Invalid coordinates' failures when e.g. Doubao emits
    floats or bracketed coords."""
    assert UITarsTool._parse_coords("(960, 540)") == (960, 540)
    assert UITarsTool._parse_coords("[960, 540]") == (960, 540)
    assert UITarsTool._parse_coords("960, 540") == (960, 540)
    assert UITarsTool._parse_coords("(960.0, 540.7)") == (960, 540)
    assert UITarsTool._parse_coords("start='(73, 294)'") == (73, 294)
    # <point>X Y</point> tag form (Doubao/Qwen-VL emit this — space-separated)
    assert UITarsTool._parse_coords("<point>846 445</point>") == (846, 445)
    assert UITarsTool._parse_coords("start='<point>100 200</point>'") == (100, 200)
    assert UITarsTool._parse_coords("<POINT>10 20</POINT>") == (10, 20)  # case-insensitive
    # <|box_start|>(X Y)<|box_end|> tag form (UI-TARS-2 / Doubao native)
    assert UITarsTool._parse_coords("<|box_start|>(542 520)<|box_end|>") == (542, 520)
    assert UITarsTool._parse_coords("start_box='<|box_start|>(542 520)<|box_end|>'") == (542, 520)


def test_parse_coords_rejects_truly_malformed():
    """No comma-separated number pair at all -> still raise (do not silently
    fabricate (0, 0))."""
    import pytest

    with pytest.raises(ValueError):
        UITarsTool._parse_coords("click the button")


