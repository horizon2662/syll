"""Per-skill procedural memory with just-in-time retrieval.

Each skill owns ``workspace/skills/{skill}/``:
- ``system_prompt.md`` -- the skill's role definition
- ``SKILL.md`` -- procedural memory (how-to, pitfalls, success patterns)

Ideas (see ROADMAP.md):
- Procedural memory = the THIRD memory type (Mem0, arXiv:2504.19413).
- Notetaker gating (Mobile-Agent-v3, arXiv:2602.16855): writes only when
  warranted; failed guesses don't pollute.
- A-Mem (arXiv:2502.12110): dedup + date tags (linking/evolution lite).
- Just-in-time retrieval (Anthropic context engineering): when SKILL.md grows,
  ``get_relevant(query)`` returns only the highest-signal lines (keyword
  overlap + recency) instead of the whole file, so a mature skill doesn't
  blow up the subagent's context. A filesystem is competitive with vector
  stores (Letta: 74% LoCoMo), so we stay markdown -- no vector DB yet.

Does **not** modify the original ``memory.py``.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

_DATE_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2})\]")


class SkillMemory:
    """Procedural, per-skill memory with conditional writes + JIT retrieval."""

    def __init__(self, workspace: Path, skill: str):
        self.skill = skill
        self.dir = workspace / "skills" / skill
        self.dir.mkdir(parents=True, exist_ok=True)
        self.skill_file = self.dir / "SKILL.md"
        self.prompt_file = self.dir / "system_prompt.md"

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------
    def load(self) -> str:
        """Load the skill's procedural memory (empty if none yet)."""
        if self.skill_file.exists():
            return self.skill_file.read_text(encoding="utf-8").strip()
        return ""

    def load_prompt(self) -> str:
        if self.prompt_file.exists():
            return self.prompt_file.read_text(encoding="utf-8").strip()
        return ""

    def context_for_subagent(self) -> str:
        """Full skill memory to inject when it is still small."""
        text = self.load()
        if not text:
            return ""
        return f"(from skill '{self.skill}' memory)\n{text}"

    def get_relevant(self, query: str = "", max_chars: int = 4000) -> str:
        """Just-in-time retrieval: return the highest-signal slice of SKILL.md.

        Scoring per line: query keyword overlap + recency (newer dates rank
        higher). Header is always kept. Capped at ``max_chars`` so a mature
        skill cannot blow up the subagent's context.
        """
        text = self.load()
        if not text:
            return ""
        if len(text) <= max_chars:
            return text

        lines = text.splitlines()
        # Preserve a leading header block (lines starting with '#') verbatim.
        header: list[str] = []
        body_start = 0
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith("#") or not ln.strip():
                header.append(ln)
                body_start = i + 1
            else:
                break
        body = [ln for ln in lines[body_start:] if ln.strip()]

        q_terms = [w.lower() for w in re.split(r"\W+", query) if len(w) > 2]
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        def score(ln: str) -> float:
            s = 0.0
            low = ln.lower()
            for t in q_terms:
                if t in low:
                    s += 1.0
            m = _DATE_RE.search(ln)
            if m:
                # newer = higher; days-old penalty
                try:
                    d = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
                    age = (datetime.now(timezone.utc) - d).days
                    s += max(0.0, 3.0 - age / 30.0)
                except Exception:
                    s += 0.5
            return s

        body.sort(key=score, reverse=True)

        budget = max_chars - sum(len(l) + 1 for l in header)
        kept: list[str] = []
        for ln in body:
            if budget - (len(ln) + 1) < 0:
                break
            kept.append(ln)
            budget -= len(ln) + 1
        kept.sort(key=lambda ln: lines.index(ln) if ln in lines else 0)  # restore order

        out = "\n".join(header + kept).strip()
        return f"(from skill '{self.skill}' memory, {len(kept)} of {len(body)} notes)\n{out}"

    # ------------------------------------------------------------------
    # gated write (Mobile-Agent-v3 Notetaker rule + A-Mem evolution)
    # ------------------------------------------------------------------
    def ingest(
        self,
        lessons: list[str],
        status_ok: bool,
        *,
        min_len: int = 12,
    ) -> int:
        """Write candidate lessons into SKILL.md, but ONLY when warranted.

        Gating:
        - On success: append genuinely new how-to / pattern lessons.
        - On failure: append ONLY if the caller marks a lesson as a reusable
          pitfall; otherwise skip -- failed guesses must not pollute.
        - Dedup against existing content; each line tagged with date.

        Returns the number of lessons actually written.
        """
        existing = self.load()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        new_lines: list[str] = []
        for lesson in lessons:
            lesson = lesson.strip()
            if len(lesson) < min_len:
                continue
            if lesson in existing:  # dedup
                continue
            tag = "OK" if status_ok else "PITFALL"
            new_lines.append(f"- [{today}] {tag} {lesson}")

        if not new_lines:
            return 0

        header = "" if existing else (
            f"# Skill: {self.skill}\n\nProcedural memory (how-to + pitfalls).\n"
        )
        sep = "\n" if existing else ""
        self.skill_file.write_text(
            header + existing + sep + "\n".join(new_lines) + "\n",
            encoding="utf-8",
        )
        logger.info(f"skill[{self.skill}] wrote {len(new_lines)} lesson(s) to SKILL.md")
        return len(new_lines)
