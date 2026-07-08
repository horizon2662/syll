"""Regression: UI-TARS-2 / Qwen-VL / Doubao action parsing.

The actor emits action-name variants (left_double / left_single / right_single)
and <point>x y</point> space-separated coords. Both parsers must normalize them
so a click actually lands instead of "Unknown action" / "Cannot parse".
"""

from syll.agent.tools.aloha_planner_tool import (
    AlohaPlannerTool,
    _extract_coord_pairs,
)


def test_extract_coord_pairs_handles_all_formats():
    assert _extract_coord_pairs("(10, 20)") == [[10, 20]]
    assert _extract_coord_pairs("[10, 20]") == [[10, 20]]
    assert _extract_coord_pairs("<point>846 445</point>") == [[846, 445]]
    assert _extract_coord_pairs("<POINT>1 2</POINT>") == [[1, 2]]  # case-insensitive
    assert _extract_coord_pairs("(960.0, 540.7)") == [[960, 540]]  # float truncation
    assert _extract_coord_pairs(
        "drag(start=<point>1 2</point>, end=<point>3 4</point>)"
    ) == [[1, 2], [3, 4]]
    assert _extract_coord_pairs("no numbers here") == []


def test_uitars_to_action_dict_alias_names_with_point_coords():
    """The Enhanced actor path: aliases map correctly + <point> coords parse."""
    parse = AlohaPlannerTool._uitars_to_action_dict

    r = parse("left_double(start=<point>846 445</point>)")
    assert r["action"] == "DOUBLE_CLICK"
    assert r["position"] == [846, 445]

    assert parse("left_single(start=<point>100 200</point>)")["action"] == "CLICK"

    r = parse("right_single(start='(30, 40)')")
    assert r["action"] == "RIGHT_CLICK"
    assert r["position"] == [30, 40]

    # canonical names still work
    assert parse("double_click(start=(1, 2))")["action"] == "DOUBLE_CLICK"
    assert parse("click(start=(1, 2))")["action"] == "CLICK"


def test_ui_tars_action_alias_map():
    """UITarsTool (fallback path) normalizes the same aliases."""
    from syll.agent.tools.ui_tars import _UITARS_ACTION_ALIASES

    assert _UITARS_ACTION_ALIASES["left_double"] == "double_click"
    assert _UITARS_ACTION_ALIASES["left_single"] == "click"
    assert _UITARS_ACTION_ALIASES["right_single"] == "right_click"
