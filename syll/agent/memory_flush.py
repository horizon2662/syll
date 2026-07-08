"""Memory flush turn: promote important facts from a single turn to long-term memory.

Inspired by ALE-Claw's memory flush before context compaction: instead of
blindly appending every turn to daily notes, a lightweight curator model looks
at the exchange and writes only the non-obvious, reusable facts to the global
``MEMORY.md``. The turn itself is not added to the chat transcript.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from syll.agent.memory import MemoryStore
from syll.providers.base import LLMProvider


class MemoryFlusher:
    """Extract and persist high-signal facts from one agent turn."""

    def __init__(
        self,
        provider: LLMProvider,
        memory_store: MemoryStore,
        model: str | None = None,
        max_facts: int = 3,
    ):
        self.provider = provider
        self.memory = memory_store
        self.model = model or provider.get_default_model()
        self.max_facts = max_facts

    _SYSTEM_PROMPT = """You are a memory curator for a personal AI assistant.
Your job is to decide what from this conversation turn is worth remembering long-term.

Rules:
- Save only non-obvious, reusable facts: user preferences, hidden requirements,
  failure patterns, important constraints, or the user's identity/purpose.
- Do NOT save raw requests, code listings, file paths, git history, or anything
  the user could easily repeat next time.
- If there is nothing worth remembering, reply with exactly: NONE
- Otherwise output one fact per line, plain text, no bullets, no numbering.
- Keep each fact to one sentence.
- Output at most {max_facts} facts.
"""

    async def flush(self, user_text: str, assistant_text: str) -> list[str]:
        """Extract facts from a turn and append new ones to long-term memory.

        Returns the list of facts actually written.
        """
        if not user_text and not assistant_text:
            return []

        facts = await self._extract_facts(user_text, assistant_text)
        if not facts:
            return []

        existing = self.memory.read_long_term().lower()
        today = datetime.now().strftime("%Y-%m-%d")
        written: list[str] = []
        lines: list[str] = []

        for fact in facts:
            fact = fact.strip()
            if not fact or fact.lower() == "none":
                continue
            if len(fact) < 12:
                continue
            if fact.lower() in existing:
                continue
            lines.append(f"- [{today}] {fact}")
            written.append(fact)
            existing += "\n" + fact.lower()

        if not lines:
            return []

        try:
            header = "# Long-term Memory\n\n" if not self.memory.read_long_term() else ""
            current = self.memory.read_long_term().rstrip()
            sep = "\n" if current else ""
            self.memory.write_long_term(current + sep + "\n".join(lines) + "\n")
            logger.debug(f"Memory flush wrote {len(lines)} fact(s)")
        except Exception as e:
            logger.warning(f"Memory flush failed to persist: {e}")
            return []

        return written

    async def _extract_facts(self, user_text: str, assistant_text: str) -> list[str]:
        """Ask the provider to extract memorable facts."""
        prompt = (
            f"User: {user_text[:2000]}\n\n"
            f"Assistant: {assistant_text[:2000]}\n\n"
            "What, if anything, is worth remembering?"
        )
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT.format(max_facts=self.max_facts)},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.provider.chat(
                messages=messages,
                tools=None,
                model=self.model,
                max_tokens=400,
                temperature=0,
            )
            text = (response.content or "").strip()
        except Exception as e:
            logger.warning(f"Memory flush extraction failed: {e}")
            return []

        if not text or text.upper() == "NONE":
            return []

        return [line.strip("-•* ") for line in text.splitlines() if line.strip()]
