"""Regression: UI-TARS-2 / Doubao coordinate-format compatibility.

The actor emits coordinates in several formats; the parser must handle all of
them or the action silently fails as "Cannot parse". Pins box-token, point-tag,
and comma-separated forms for click + drag.
"""

from syll.agent.tools.aloha_planner_tool import (
    AlohaPlannerTool,
    _extract_coord_pairs,
)


# --- _extract_coord_pairs: all three formats ---

def test_box_token_format():
    """<|box_start|>(X Y)<|box_end|> — Doubao/UI-TARS-2 native form."""
    assert _extract_coord_pairs("click(start_box='<|box_start|>(542 520)<|box_end|>')") == [[542, 520]]


def test_point_tag_format():
    assert _extract_coord_pairs("click(start='<point>100 200</point>')") == [[100, 200]]


def test_comma_format():
    assert _extract_coord_pairs("click(start='(100, 200)')") == [[100, 200]]


def test_two_box_pairs_for_drag():
    """drag with start_box + end_box in box-token form → 2 pairs."""
    s = ("drag(start_box='<|box_start|>(10 20)<|box_end|>', "
         "end_box='<|box_start|>(30 40)<|box_end|>')")
    assert _extract_coord_pairs(s) == [[10, 20], [30, 40]]


def test_floats_truncated_to_int():
    assert _extract_coord_pairs("click(start='(100.7, 200.3)')") == [[100, 200]]


def test_no_coords_returns_empty():
    assert _extract_coord_pairs("type(content='hello')") == []


# --- _uitars_to_action_dict: end-to-end action parsing ---

def test_click_with_box_token():
    d = AlohaPlannerTool._uitars_to_action_dict(
        "click(start_box='<|box_start|>(542 520)<|box_end|>')"
    )
    assert d["action"] == "CLICK"
    assert d["position"] == [542, 520]


def test_drag_with_box_tokens():
    d = AlohaPlannerTool._uitars_to_action_dict(
        "drag(start_box='<|box_start|>(10 20)<|box_end|>', "
        "end_box='<|box_start|>(30 40)<|box_end|>')"
    )
    assert d["action"] == "DRAG"
    assert d["position"] == [30, 40]       # end position
    assert d["value"] == [10, 20]          # start position


def test_left_double_action_name():
    """UI-TARS-2 uses left_double for double-click."""
    d = AlohaPlannerTool._uitars_to_action_dict(
        "left_double(start_box='<|box_start|>(50 60)<|box_end|>')"
    )
    assert d["action"] == "DOUBLE_CLICK"
    assert d["position"] == [50, 60]


def test_comma_format_still_works():
    """Regression guard: the original comma format must not break."""
    d = AlohaPlannerTool._uitars_to_action_dict("click(start='(100, 200)')")
    assert d["action"] == "CLICK" and d["position"] == [100, 200]
