"""Per-session GUI failure ledger — the revisable source of truth for the
"gui_action task is locked" gate.

Replaces the un-addressable ``GUI_NO_RETRY_SUFFIX`` chat text (which the model
reads next turn and refuses to retry) with a structured, per-task record that
can be inspected and explicitly cleared.

Semantics (per the approved design):
- Transient (infrastructure) failures are recorded for diagnostics but never
  lock — the model may retry once the underlying issue is resolved.
- Genuine GUI failures lock until ``clear``/``clear_all`` (no TTL — manual
  revision is the only unlock).
- A subsequent success removes the entry.
- The latest outcome governs the locked state; ``attempts`` accumulates.

The persistence layer is now :class:`JsonMemoryStore` (scope ``gui_ledger``)
so the ledger participates in the same storage abstraction as the rest of the
memory system, while keeping the file layout unchanged.
"""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from syll.agent.memory import JsonMemoryStore

_DEFAULT_ROOT = Path.home() / ".syll" / "gui_ledgers"

# Per-path locks so concurrent callers in the same process serialize writes to
# the same ledger file. Cross-process races converge on the next call (atomic
# os.replace guarantees the file is never half-written).
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _LOCKS_GUARD:
        lk = _LOCKS.get(key)
        if lk is None:
            lk = threading.Lock()
            _LOCKS[key] = lk
    return lk


def _safe_filename(name: str) -> str:
    """Make ``name`` safe as a filename (mirrors SessionManager's sanitizer)."""
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    return name


class GuiAttemptLedger:
    """Per-session ledger of GUI task outcomes, keyed by task signature."""

    def __init__(self, session_key: str, dir_root: Path | None = None):
        self._session_key = session_key
        root = Path(dir_root) if dir_root is not None else _DEFAULT_ROOT
        filename = f"{_safe_filename(session_key.replace(':', '_'))}.json"
        self._store = JsonMemoryStore(
            root,
            scope="gui_ledger",
            memory_filename=filename,
        )

    # ------------------------------------------------------------------
    # task signature
    # ------------------------------------------------------------------
    @staticmethod
    def _task_sig(instruction: str) -> str:
        """Normalize an instruction to a stable signature.

        Lowercase, collapse whitespace, truncate to 120 chars, then sha1 (8
        hex). Near-identical prompts collide by design ('click submit' and
        'Click Submit.' share a lock — errs toward safety)."""
        norm = " ".join((instruction or "").strip().lower().split())[:120]
        return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]

    @property
    def path(self) -> Path:
        """On-disk ledger file (exposed for tests / debugging)."""
        return self._store.memory_file

    # ------------------------------------------------------------------
    # load / save
    # ------------------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        data = self._store.read_json()
        if isinstance(data, dict) and isinstance(data.get("tasks"), dict):
            return data
        return {"version": 1, "session_key": self._session_key, "tasks": {}}

    def _save(self, data: dict[str, Any]) -> None:
        self._store.write_json(data)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def record_failure(
        self, *, instruction: str, kind: Any, reason: str, step: int = 0
    ) -> None:
        """Record a failed attempt. ``kind`` is a ``GuiFailureKind`` (duck-typed:
        needs ``.value`` and ``.retryable``). The latest outcome governs the
        locked state; ``attempts`` accumulates across retries."""
        sig = self._task_sig(instruction)
        retryable = bool(getattr(kind, "retryable", False))
        with _lock_for(self.path):
            data = self._load()
            prev = data["tasks"].get(sig, {})
            attempts = int(prev.get("attempts", 0)) + 1
            data["tasks"][sig] = {
                "instruction": instruction[:200],
                "kind": getattr(kind, "value", str(kind)),
                "retryable": retryable,
                "attempts": attempts,
                "last_failed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "step": step,
                "reason": (reason or "")[:300],
            }
            self._save(data)

    def record_success(self, *, instruction: str) -> None:
        """A successful attempt clears the entry (the task is no longer failed)."""
        sig = self._task_sig(instruction)
        with _lock_for(self.path):
            data = self._load()
            if sig in data["tasks"]:
                del data["tasks"][sig]
                self._save(data)

    def is_locked(self, instruction: str) -> bool:
        """True iff a non-retryable (genuine) failure entry exists for the task."""
        entry = self.lock_status(instruction)
        return bool(entry) and not entry.get("retryable", False)

    def lock_status(self, instruction: str) -> dict[str, Any] | None:
        """The task's failure entry, or None if it has none / was cleared."""
        sig = self._task_sig(instruction)
        with _lock_for(self.path):
            data = self._load()
        return data["tasks"].get(sig)

    def clear(self, task_sig: str) -> bool:
        """Remove one task entry by its signature (as returned by ``_task_sig``)."""
        with _lock_for(self.path):
            data = self._load()
            if task_sig in data["tasks"]:
                del data["tasks"][task_sig]
                self._save(data)
                return True
            return False

    def clear_all(self) -> int:
        """Remove every task entry for this session. Returns the count cleared."""
        with _lock_for(self.path):
            data = self._load()
            n = len(data["tasks"])
            if n:
                data["tasks"] = {}
                self._save(data)
            return n
