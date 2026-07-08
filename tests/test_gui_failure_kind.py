"""Stage 1: GuiFailureKind classification + _retry_suffix.

Locks which failure causes are retryable (transient/infrastructure — no suffix,
model may retry) vs genuine GUI deadlocks (suffix kept until the ledger clears).
"""

from syll.agent.tools.ui_tars import GUI_NO_RETRY_SUFFIX, GuiFailureKind, _retry_suffix


def test_transient_kinds_are_retryable():
    assert GuiFailureKind.SCREENSHOT_FAIL.retryable is True
    assert GuiFailureKind.MODEL_CALL_FAIL.retryable is True


def test_genuine_kinds_are_not_retryable():
    assert GuiFailureKind.STUCK.retryable is False
    assert GuiFailureKind.ACTION_EXEC_FAIL.retryable is False
    assert GuiFailureKind.MAX_STEPS.retryable is False


def test_retry_suffix_transient_is_empty():
    # Transient failures must NOT emit the do-not-retry lock — the model may
    # retry once the underlying API/screen issue is resolved.
    assert _retry_suffix(GuiFailureKind.SCREENSHOT_FAIL) == ""
    assert _retry_suffix(GuiFailureKind.MODEL_CALL_FAIL) == ""


def test_retry_suffix_genuine_keeps_lock():
    assert _retry_suffix(GuiFailureKind.STUCK) == GUI_NO_RETRY_SUFFIX
    assert _retry_suffix(GuiFailureKind.ACTION_EXEC_FAIL) == GUI_NO_RETRY_SUFFIX
    assert _retry_suffix(GuiFailureKind.MAX_STEPS) == GUI_NO_RETRY_SUFFIX
