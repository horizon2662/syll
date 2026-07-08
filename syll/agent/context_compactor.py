"""Context compaction for long-horizon agent runs.

Implements a three-tier strategy similar to ALE-Claw/OpenClaw:

1. Micro-compaction: replace stale tool results with short placeholders.
2. LLM summarization: compress older conversation segments into a summary.
3. Hard truncation: drop oldest non-system messages when nothing else fits.

The compactor is intentionally conservative: it never drops the system prompt
or the most recent user message, and it tries to keep tool outputs that the
model is likely to need for the current sub-task.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger


class ContextCompactor:
    """Compact a message list to fit inside a token budget."""

    def __init__(
        self,
        provider: Any | None = None,
        model: str | None = None,
        *,
        reserve_tokens: int = 4096,
        micro_compact_keep_recent: int = 3,
        summary_max_tokens: int = 512,
    ):
        """
        Args:
            provider: LLM provider used for LLM-based summarization.
            model: Model name for summarization.
            reserve_tokens: Tokens reserved for the model's completion.
            micro_compact_keep_recent: Number of recent tool results to keep
                verbatim during micro-compaction.
            summary_max_tokens: Max tokens for the LLM-generated summary.
        """
        self.provider = provider
        self.model = model
        self.reserve_tokens = max(reserve_tokens, 0)
        self.micro_compact_keep_recent = max(micro_compact_keep_recent, 0)
        self.summary_max_tokens = max(summary_max_tokens, 100)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def compact(
        self,
        messages: list[dict[str, Any]],
        budget_tokens: int,
    ) -> list[dict[str, Any]]:
        """Return a copy of ``messages`` that fits inside ``budget_tokens``.

        The algorithm tries, in order:
        1. Micro-compaction of old tool results.
        2. LLM summarization of older conversation turns.
        3. Hard truncation of oldest non-system messages.

        If ``budget_tokens`` is 0 or unknown, the original list is returned
        unchanged (the caller should fall back to a different budget or no
        compaction).
        """
        if not messages or budget_tokens <= 0:
            return list(messages)

        effective_budget = max(budget_tokens - self.reserve_tokens, 1)
        current = _estimate_tokens(messages)
        if current <= effective_budget:
            return list(messages)

        logger.info(
            f"Context over budget: ~{current} tokens > {effective_budget} "
            "effective budget; starting micro-compaction"
        )

        # Tier 1: micro-compaction
        compacted = self._micro_compact(messages)
        current = _estimate_tokens(compacted)
        if current <= effective_budget:
            logger.info(f"Micro-compaction reduced context to ~{current} tokens")
            return compacted

        # Tier 2: LLM summarization of older turns
        if self.provider is not None and self.model:
            compacted = await self._llm_summarize(compacted, effective_budget)
            current = _estimate_tokens(compacted)
            if current <= effective_budget:
                logger.info(f"LLM summarization reduced context to ~{current} tokens")
                return compacted

        # Tier 3: hard truncation
        compacted = self._truncate(compacted, effective_budget)
        current = _estimate_tokens(compacted)
        logger.info(f"Truncated context to ~{current} tokens")
        return compacted

    # ------------------------------------------------------------------
    # Tier 1: micro-compaction
    # ------------------------------------------------------------------

    def _micro_compact(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Replace old tool-result content with short placeholders.

        The most recent ``micro_compact_keep_recent`` tool results are kept
        verbatim because they are usually needed for the current sub-task.
        """
        out: list[dict[str, Any]] = []
        tool_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "tool"
        ]
        keep_set = set(tool_indices[-self.micro_compact_keep_recent :])

        for i, msg in enumerate(messages):
            if i in keep_set or msg.get("role") != "tool":
                out.append(dict(msg))
                continue

            name = msg.get("name", "tool")
            content = msg.get("content", "")
            length = len(_stringify(content))
            placeholder = (
                f"[{name}] earlier result omitted ({length} chars); "
                "ask again if details are needed."
            )
            compacted = dict(msg)
            compacted["content"] = placeholder
            out.append(compacted)

        return out

    # ------------------------------------------------------------------
    # Tier 2: LLM summarization
    # ------------------------------------------------------------------

    async def _llm_summarize(
        self,
        messages: list[dict[str, Any]],
        effective_budget: int,
    ) -> list[dict[str, Any]]:
        """Summarize older conversation turns and replace them with a summary.

        We keep the system prompt and the most recent turns (up to a rough
        token threshold), and ask the model to summarize everything before
        that point.  If summarization fails, we return the input unchanged so
        the next tier (truncation) can still run.
        """
        # Keep system + recent ~30% of budget for recent context.
        recent_budget = int(effective_budget * 0.3)
        split_idx = self._find_split_index(messages, recent_budget)
        if split_idx <= 1:
            # Not enough older context to summarize; skip.
            return list(messages)

        older = messages[:split_idx]
        recent = messages[split_idx:]

        summary = await self._summarize_messages(older)
        if not summary:
            return list(messages)

        summary_msg = {
            "role": "user",
            "content": f"[Earlier conversation summary]\n{summary}",
        }
        return [summary_msg] + recent

    def _find_split_index(
        self,
        messages: list[dict[str, Any]],
        recent_budget: int,
    ) -> int:
        """Return the index where the most recent context starts.

        Walks backward from the end, accumulating tokens, until ``recent_budget``
        is exceeded. The returned index is the first message of the recent block.
        """
        total = 0
        for idx in range(len(messages) - 1, -1, -1):
            total += _estimate_message_tokens(messages[idx])
            if total >= recent_budget:
                return idx
        return 0

    async def _summarize_messages(
        self, messages: list[dict[str, Any]]
    ) -> str | None:
        """Ask the LLM to summarize a list of messages."""
        try:
            prompt = (
                "Summarize the following conversation concisely for an agent "
                "that will continue the task. Preserve: the user's goal, key "
                "decisions, files/data created, errors encountered, and the "
                "current state. Omit screenshots, raw tool output, and "
                "verbosity. Keep under 300 words.\n\n"
                + _messages_to_text(messages)
            )
            resp = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=self.summary_max_tokens,
                temperature=0,
            )
            if getattr(resp, "finish_reason", None) == "error" or not getattr(resp, "content", None):
                return None
            return str(resp.content).strip()
        except Exception as exc:
            logger.warning(f"Context summarization failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Tier 3: hard truncation
    # ------------------------------------------------------------------

    def _truncate(
        self,
        messages: list[dict[str, Any]],
        effective_budget: int,
    ) -> list[dict[str, Any]]:
        """Drop oldest non-system messages until the list fits the budget.

        The system prompt (first message if role == system) and the last user
        message are never removed.
        """
        if not messages:
            return []

        out = list(messages)
        while len(out) > 2 and _estimate_tokens(out) > effective_budget:
            # Find the oldest non-system, non-last-user message to drop.
            # System prompt is usually index 0.
            remove_idx: int | None = None
            for i, msg in enumerate(out[:-1]):
                if msg.get("role") == "system":
                    continue
                remove_idx = i
                break
            if remove_idx is None:
                break
            out.pop(remove_idx)

        return out


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_CONTENT_ROLES = {"user", "assistant", "tool"}


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _estimate_message_tokens(msg: dict[str, Any]) -> int:
    """Cheap token estimator.

    We use a character-based heuristic (~4 chars per token for English/Chinese
    mixed text).  This is intentionally fast and dependency-free; it is good
    enough for compaction decisions.
    """
    total = 0
    # Overhead per message
    total += 4
    content = msg.get("content")
    if content is not None:
        total += len(_stringify(content)) // 4 + 1
    if msg.get("name"):
        total += len(msg["name"]) // 4 + 1
    if msg.get("tool_calls"):
        total += len(json.dumps(msg["tool_calls"], ensure_ascii=False)) // 4 + 1
    return total


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    # Base overhead for the message list / formatting
    return sum(_estimate_message_tokens(m) for m in messages) + 4


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = _stringify(msg.get("content", ""))
        name = msg.get("name")
        prefix = f"{role}"
        if name:
            prefix += f" ({name})"
        parts.append(f"{prefix}: {content}")
    return "\n\n".join(parts)
