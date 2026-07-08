"""Unified memory interface (L2): one recall surface across the four memory
scopes Syll accumulated — chat / global / skill / gui_step.

Each adapter wraps an existing store **without modifying it** (respecting each
store's "does not modify memory.py" convention). ``MemoryHub.recall``
aggregates across scopes so a caller can ask "what do I know relevant to X"
without knowing which silo holds it — the read-side counterpart to the four
parallel ``MEMORY.md`` / ``PROJECT.md`` / ``SKILL.md`` / ``execution_history.md``
files that previously never shared an entry point.

Lives under ``longhorizon/`` next to ``global_memory`` / ``skill_memory`` (not
under ``agent/memory`` — that name is already taken by the ``memory.py``
module holding ``MemoryStore``, and Python cannot have both a module and a
package of the same name).

Write semantics differ too much between stores (``ingest(lessons, status)`` vs
``record_step(...)`` vs ``log(note)``) to collapse cleanly, so writes stay on
each store's native method; this module unifies the *read* path.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

ALL_SCOPES = ("user", "chat", "global", "skill", "gui_step")


@runtime_checkable
class Memory(Protocol):
    """Read-side interface every memory scope implements."""

    scope: str

    def recall(self, query: str = "", max_chars: int = 4000) -> str: ...


class UserMemoryAdapter:
    """Adapts the user-scoped global ``MemoryStore`` — scope: user."""

    scope = "user"

    def __init__(self, store: Any):
        self.store = store

    def recall(self, query: str = "", max_chars: int = 4000) -> str:
        try:
            text = self.store.get_memory_context() if hasattr(self.store, "get_memory_context") else ""
            return (text or "")[:max_chars]
        except Exception:
            return ""


class ChatMemoryAdapter:
    """Adapts ``MemoryStore`` (MEMORY.md + daily notes) — scope: chat."""

    scope = "chat"

    def __init__(self, store: Any):
        self.store = store

    def recall(self, query: str = "", max_chars: int = 4000) -> str:
        try:
            text = self.store.read_long_term() if hasattr(self.store, "read_long_term") else ""
            return (text or "")[:max_chars]
        except Exception:
            return ""


class GlobalMemoryAdapter:
    """Adapts ``GlobalMemory`` (PROJECT.md) — scope: global."""

    scope = "global"

    def __init__(self, store: Any):
        self.store = store

    def recall(self, query: str = "", max_chars: int = 4000) -> str:
        try:
            return (self.store.load() or "")[:max_chars]
        except Exception:
            return ""


class SkillMemoryAdapter:
    """Adapts ``SkillMemory`` (SKILL.md with JIT retrieval) — scope: skill."""

    scope = "skill"

    def __init__(self, store: Any):
        self.store = store

    def recall(self, query: str = "", max_chars: int = 4000) -> str:
        try:
            if hasattr(self.store, "get_relevant"):
                return self.store.get_relevant(query=query, max_chars=max_chars)
            return (self.store.load() or "")[:max_chars]
        except Exception:
            return ""


class GuiStepMemoryAdapter:
    """Adapts ``StructuredMemory`` (execution history) — scope: gui_step."""

    scope = "gui_step"

    def __init__(self, store: Any):
        self.store = store

    def recall(self, query: str = "", max_chars: int = 4000) -> str:
        try:
            return (self.store.get_execution_context() or "")[:max_chars]
        except Exception:
            return ""


class MemoryHub:
    """Aggregate recall across memory scopes.

    Example::

        hub = MemoryHub(
            chat=ChatMemoryAdapter(mem),
            skill=SkillMemoryAdapter(sm),
            global_=GlobalMemoryAdapter(gm),
        )
        ctx = hub.recall("open chrome", scopes=["skill", "global"])
    """

    def __init__(
        self,
        user: Any = None,
        chat: Any = None,
        global_: Any = None,
        skill: Any = None,
        gui_step: Any = None,
    ):
        self._stores = {
            "user": user,
            "chat": chat,
            "global": global_,
            "skill": skill,
            "gui_step": gui_step,
        }

    def recall(
        self,
        query: str = "",
        scopes=ALL_SCOPES,
        max_chars_per_scope: int = 4000,
    ) -> str:
        """Return concatenated recall from the requested scopes (empty ones skipped)."""
        parts: list[str] = []
        for scope in scopes:
            store = self._stores.get(scope)
            if store is None:
                continue
            try:
                text = store.recall(query=query, max_chars=max_chars_per_scope)
            except Exception:
                text = ""
            if text:
                parts.append(f"## [{scope} memory]\n{text}")
        return "\n\n".join(parts)
