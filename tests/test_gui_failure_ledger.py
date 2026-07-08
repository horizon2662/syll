"""Stage 2a: GuiAttemptLedger — revisable per-session GUI failure state."""

from syll.agent.gui_failure_ledger import GuiAttemptLedger
from syll.agent.tools.ui_tars import GuiFailureKind


def _ledger(tmp_path):
    return GuiAttemptLedger("test:1", dir_root=tmp_path)


def test_transient_failure_does_not_lock(tmp_path):
    lg = _ledger(tmp_path)
    lg.record_failure(
        instruction="open app", kind=GuiFailureKind.MODEL_CALL_FAIL, reason="api err"
    )
    assert lg.is_locked("open app") is False
    st = lg.lock_status("open app")
    assert st is not None and st["retryable"] is True
    assert st["kind"] == "model_call_fail"


def test_genuine_failure_locks(tmp_path):
    lg = _ledger(tmp_path)
    lg.record_failure(instruction="open app", kind=GuiFailureKind.STUCK, reason="repeat")
    assert lg.is_locked("open app") is True
    assert lg.lock_status("open app")["attempts"] == 1


def test_latest_outcome_governs_lock_state(tmp_path):
    lg = _ledger(tmp_path)
    lg.record_failure(instruction="x", kind=GuiFailureKind.STUCK, reason="r")
    assert lg.is_locked("x")
    # a later transient outcome unlocks (latest governs), attempts accumulate
    lg.record_failure(instruction="x", kind=GuiFailureKind.SCREENSHOT_FAIL, reason="r2")
    assert lg.is_locked("x") is False
    assert lg.lock_status("x")["attempts"] == 2


def test_record_success_clears_entry(tmp_path):
    lg = _ledger(tmp_path)
    lg.record_failure(instruction="x", kind=GuiFailureKind.STUCK, reason="r")
    assert lg.is_locked("x")
    lg.record_success(instruction="x")
    assert lg.is_locked("x") is False
    assert lg.lock_status("x") is None


def test_clear_all_returns_count(tmp_path):
    lg = _ledger(tmp_path)
    lg.record_failure(instruction="a", kind=GuiFailureKind.STUCK, reason="r")
    lg.record_failure(instruction="b", kind=GuiFailureKind.MAX_STEPS, reason="r")
    assert lg.clear_all() == 2
    assert lg.is_locked("a") is False and lg.is_locked("b") is False


def test_clear_one_by_signature(tmp_path):
    lg = _ledger(tmp_path)
    lg.record_failure(instruction="a", kind=GuiFailureKind.STUCK, reason="r")
    sig = GuiAttemptLedger._task_sig("a")
    assert lg.clear(sig) is True
    assert lg.is_locked("a") is False
    assert lg.clear(sig) is False  # already gone


def test_task_sig_normalization_collision(tmp_path):
    # whitespace + case differences normalize to the SAME signature (so the
    # lock set under one phrasing is visible under the other)
    assert GuiAttemptLedger._task_sig("Click   Submit") == GuiAttemptLedger._task_sig("click submit")
    assert GuiAttemptLedger._task_sig("  Open App  ") == GuiAttemptLedger._task_sig("open app")
    lg = _ledger(tmp_path)
    lg.record_failure(instruction="Click   Submit", kind=GuiFailureKind.STUCK, reason="r")
    assert lg.is_locked("click submit") is True
    # boundary: punctuation is NOT normalized (a deliberate limit — only
    # whitespace/case collapse, so "submit." != "submit")
    assert GuiAttemptLedger._task_sig("submit.") != GuiAttemptLedger._task_sig("submit")


def test_atomic_write_leaves_no_tmp(tmp_path):
    lg = _ledger(tmp_path)
    lg.record_failure(instruction="x", kind=GuiFailureKind.STUCK, reason="r")
    assert lg.path.exists()
    assert list(lg.path.parent.glob("*.tmp")) == []


def test_corrupt_file_treated_as_empty(tmp_path):
    lg = _ledger(tmp_path)
    lg.path.parent.mkdir(parents=True, exist_ok=True)
    lg.path.write_text("not json {", encoding="utf-8")
    assert lg.is_locked("anything") is False  # no raise
    # fresh write after corruption still works
    lg.record_failure(instruction="x", kind=GuiFailureKind.STUCK, reason="r")
    assert lg.is_locked("x") is True
