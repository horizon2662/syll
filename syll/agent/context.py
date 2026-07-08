"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
from pathlib import Path
from typing import Any

from syll.agent.memory import GlobalMemoryStore, MemoryStore
from syll.agent.skills import SkillsLoader
from syll.agent.tools.base import ToolResult
from syll.config.schema import IdentityConfig
from syll.utils.language import build_turn_language_note


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.

    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]

    def __init__(
        self,
        workspace: Path,
        global_memory: MemoryStore | None = None,
        identity: IdentityConfig | None = None,
    ):
        self.workspace = workspace
        # ``self.memory`` is the user-scoped global memory store. All automatic
        # daily notes (``append_today``) go here so they survive workspace swaps.
        self.memory = global_memory or GlobalMemoryStore()
        # ``self.workspace_memory`` holds project/workspace-specific notes that
        # should not pollute the global user memory.
        self.workspace_memory = MemoryStore(workspace, scope="workspace")
        self.skills = SkillsLoader(workspace)
        self.identity = identity or IdentityConfig()
        # Lazily-built, reused across turns (avoids re-instantiating the skill
        # stores on every system-prompt build). Stores are stateless — they read
        # disk on each call — so a single cached instance is correct and current.
        self._gui_store = None
        self._aloha_store = None

    def _substitute_vars(self, text: str) -> str:
        """Replace {{ghost_name}} and {{user_name}} placeholders with config values.

        Uses plain str.replace rather than str.format so literal { and } inside
        skill worked examples (code blocks, JSON, tool call args) don't break.
        """
        text = text.replace("{{ghost_name}}", self.identity.ghost_name or "Syll")
        if self.identity.user_name:
            text = text.replace("{{user_name}}", self.identity.user_name)
        else:
            text = text.replace("{{user_name}}", "the user")
        return text

    def substitute(self, text: str) -> str:
        """Public wrapper over `_substitute_vars` for callers outside this class.

        Used by the cron ritual handler (`on_cron_job`) at fire time so profile
        renames propagate into already-installed ritual prompts without needing
        to re-install the jobs. Safe to call on any string.
        """
        return self._substitute_vars(text)

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """
        Build the system prompt from bootstrap files, memory, and skills.

        Args:
            skill_names: Optional list of skills to include.

        Returns:
            Complete system prompt.
        """
        parts = []

        # Core identity
        parts.append(self._get_identity())

        # Bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        # Lore fragments — story beats the agent may rarely surface as asides
        # when a conversational moment naturally rhymes with one. Loaded on
        # every turn so the LLM has them in context; rules for when/how to
        # surface them live at the top of fragments.md itself.
        fragments_path = self.workspace / "lore" / "fragments.md"
        if fragments_path.exists():
            try:
                fragments_content = fragments_path.read_text(encoding="utf-8")
                parts.append(f"# Lore Fragments\n\n{fragments_content}")
            except OSError:
                pass  # graceful degradation if read fails

        # Memory context — global user memory + workspace-local overlay.
        memory_parts: list[str] = []
        global_memory = self.memory.get_memory_context()
        if global_memory:
            memory_parts.append(f"## Global Memory\n{global_memory}")
        workspace_memory = self.workspace_memory.get_memory_context()
        if workspace_memory:
            memory_parts.append(f"## Workspace Memory\n{workspace_memory}")
        if memory_parts:
            parts.append("# Memory\n\n" + "\n\n".join(memory_parts))

        # Skills - progressive loading
        # 1. Always-loaded skills: include full content
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        # 2. Available skills: only show summary (agent uses read_file to load)
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")

        # GUI demonstration skills
        gui_skills_section = self._build_gui_skills_section()
        if gui_skills_section:
            parts.append(gui_skills_section)

        return self._substitute_vars("\n\n---\n\n".join(parts))

    def _get_identity(self) -> str:
        """Get the core identity section."""
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return f"""# {{{{ghost_name}}}}

You are {{{{ghost_name}}}}, a capable autonomous agent living in {{{{user_name}}}}'s computer.
You have access to tools for files, shell, web, messaging, GUI automation, and subagents.

## Current Time
{now}

## Runtime
{runtime}

## Workspace
{workspace_path}
- Memory: {workspace_path}/memory/MEMORY.md
- Daily notes: {workspace_path}/memory/YYYY-MM-DD.md
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

---

# Core Behavior Rules

## 1. Autonomous Execution

When given a task, **execute it fully end-to-end in one turn**. Do NOT stop after each step asking "shall I continue?" or "next?".

Rules:
- Multi-step tasks (download → process → save → report) must run ALL steps without pausing.
- When a skill or instruction describes a pipeline, run the entire pipeline automatically.
- Only pause for user confirmation when a step is genuinely destructive (deleting files, sending to external services, spending money).
- After completing the task, report the final result concisely.

## 2. Error Recovery

When a tool returns an error:
1. **Read the error message** — understand what went wrong before retrying.
2. **Fix and retry** — if the cause is obvious (wrong path, missing dependency, timeout, type error), fix it immediately and retry.
3. **Try alternative** — if retry fails the same way, try a different approach (different tool, different parameters, different code path).
4. **Report honestly** — only tell the user "it failed" after trying at least 2 approaches. Include the error message so the user can help.
5. **NEVER silently ignore an error** and pretend everything worked.

## 3. Progress Reporting

When executing multi-step tasks, briefly state what you are doing:
- "Downloading video..." → "Analyzing frames..." → "Generating skill..."
- Keep each progress line to one short sentence. Don't write paragraphs.
- The user sees your text output in real-time. Use this to keep them informed.

## 4. Tool Usage Best Practices

- **Prefer dedicated file tools** (read_file, write_file, edit_file) over shell commands (cat, sed) for file operations.
- **Read before editing** — always read a file before modifying it. If you haven't read it this turn, read it first.
- **Match surrounding code style** — observe naming, indentation, comment density, and idiom in the existing code. Write code that looks like it belongs.
- **Make independent tool calls in parallel** — if two tool calls don't depend on each other's results, make them in the same block.
- **Validate file existence before overwriting** — if a file might already exist and contain important data, read it first. If the content contradicts what you expected, surface that rather than silently overwriting.
- **Use absolute paths** for file operations to avoid ambiguity.

## 5. Code Editing Rules

- **Small, targeted edits** — prefer editing specific lines over rewriting entire files.
- **Verify after editing** — after making code changes, verify they work (run the code, check syntax, or at least re-read the edited section).
- **Don't duplicate existing code** — before writing new code, check if similar functionality already exists in the codebase.
- **Follow the user's language** — if they write in Chinese, respond in Chinese. If English, respond in English.
- **Reference file locations** — when discussing code, mention file paths and line numbers so the user can find them.

## 6. Memory & Knowledge Management

- **Save important findings** — if you discover something non-obvious (a bug, a config requirement, a hidden API), write it to memory.
- **Don't save obvious things** — code structure, git history, and file listings don't need to be memorized.
- **Update existing memories** — if a memory turns out to be wrong, update it rather than creating a duplicate.
- **When asked to remember something**, ask "what was non-obvious about it?" and save that insight, not the raw fact.

## 7. Communication Style

- **For direct questions and conversation**: respond with text directly. Do NOT call the `message` tool.
- **The `message` tool** is ONLY for sending to specific chat channels (WhatsApp, Telegram, etc.).
- **Be concise** — don't write 3 paragraphs when 3 lines suffice.
- **Be honest** — if tests fail, say so. If a step was skipped, say that. When something works, say it plainly without hedging.

## 8. Confirmation Policy

Ask before proceeding when:
- Deleting files or directories
- Overwriting files you didn't create
- Sending data to external services
- Executing commands that cost money or are hard to reverse

Do NOT ask for confirmation when:
- Reading files
- Running read-only commands
- Creating new files in your workspace
- The user explicitly said "just do it" or "go ahead"

## 9. Task Decomposition

For complex tasks:
1. **Plan briefly** — outline the steps before starting (2-3 sentences max).
2. **Execute in order** — work through the steps systematically.
3. **Verify at checkpoints** — after critical steps, verify the result before continuing.
4. **Report outcome** — at the end, clearly state what was done and what (if anything) failed.

## 10. Skill-Driven Behavior

When a skill (SKILL.md) is relevant to the current task:
- **Read the full SKILL.md** before acting — don't guess from the summary.
- **Follow the skill's workflow** — if it says "Step 1 → Step 2 → Step 3", do exactly that.
- **If the skill's instructions conflict with these rules**, follow the skill (it's task-specific).
- **After executing a skill**, note what worked and what didn't. If the skill needs improvement, suggest changes."""

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    def _gui_behavior_guidance_lines(self) -> list[str]:
        """Always-on guidance for the single-step gui_action tool."""
        return [
            "# GUI action — one verified step per call",
            "",
            "`gui_action` does EXACTLY ONE step per call: screenshot → plan → "
            "execute → verify (did the screen change?), then returns the result "
            "+ before/after screenshots. It does NOT retry internally and holds "
            "NO state between calls — every call is independent. YOU drive all "
            "iteration and reflection.",
            "",
            "YOU are the ORCHESTRATOR (the global planner) of the multi-step GUI "
            "task: keep the overall goal in mind, track what each step achieved, "
            "and call gui_action ONCE per step with a SPECIFIC instruction for "
            "that step. gui_action itself has NO memory of previous steps — so "
            "your instruction must carry the context (e.g. 'the File menu is "
            "already open, now click Save'). After each SUCCESS, immediately do "
            "the next step; do NOT stop until you get [DONE] or the task is "
            "genuinely complete. If you get stuck (2-3 NO_CHANGE on the same "
            "sub-goal), report progress + the blocker to the user.",
            "",
            "Read the first line of each result:",
            "- `[STEP] gui_action: SUCCESS` — action worked. Call gui_action "
            "again for the next sub-goal, or report completion. Do NOT pass "
            "prior_failures.",
            "- `[STEP] gui_action: UNCERTAIN` — small/ambiguous change (often a "
            "real focus/select/value-entered). LOOK at the attached screenshot: "
            "if it worked, continue (no prior_failures); if not, retry as NO_CHANGE.",
            "- `[STEP] gui_action: NO_CHANGE` — the action had no real effect. "
            "The result includes an Error type (GROUNDING or PLAN) + Category + "
            "a targeted Next hint. Retry the SAME sub-goal once or twice "
            "following that hint.",
            "- `[STEP] gui_action: ERROR` — executor failed. Check the target, retry.",
            "- `[DONE] gui_action: DONE` — task complete. Stop and report success.",
            "",
            "`prior_failures` lifecycle (IMPORTANT — the tool is stateless):",
            "- Pass `prior_failures=[{category, reason}]` ONLY when immediately "
            "re-attempting the SAME step that just returned NO_CHANGE — it tells "
            "the planner what to avoid this time.",
            "- Do NOT pass it on a fresh sub-goal, after a SUCCESS, or after "
            "moving on — stale failure context misleads the planner. Each new "
            "step starts clean.",
            "",
            "Error-type hints: GROUNDING (COORD_OFF/OCCLUDED — right action, "
            "missed coords → re-call with the same goal; the actor re-locates "
            "from the fresh screenshot); PLAN (ELEMENT_ABSENT/WORKFLOW_ORDER/"
            "LOADING — wrong target/order → scroll, navigate, wait, or pick a "
            "different action). Give up after 2-3 consecutive NO_CHANGE on the "
            "same sub-goal and report to the user.",
            "",
        ]

    def _build_gui_skills_section(self) -> str:
        """Build the GUI system-prompt section: always-on single-step behavior
        guidance, plus a list of recorded demonstration skills when any exist."""
        if self._gui_store is None:
            from syll.agent.gui_skill import GUISkillStore
            self._gui_store = GUISkillStore(self.workspace)
        if self._aloha_store is None:
            from syll.agent.aloha_gui_skill import AlohaSkillStore
            self._aloha_store = AlohaSkillStore(self.workspace)

        gui_skills = self._gui_store.list_gui_skills()
        try:
            aloha_skills = self._aloha_store.list_skills()
        except Exception:
            aloha_skills = []

        # Behavior guidance is always present (ad-hoc tasks need it even with
        # no recorded skills).
        lines: list[str] = self._gui_behavior_guidance_lines()

        skill_lines: list[str] = []
        if gui_skills:
            skill_lines += [
                "# GUI Demonstration Skills",
                "",
                "The following GUI skills have recorded demonstrations. When "
                "executing similar tasks, use "
                '`gui_action(instruction="...", skill_name="xxx")` to inject '
                "demonstration context:",
                "",
            ]
            for s in gui_skills:
                skill_lines.append(f"- **{s['name']}**: {s['description']} ({s['steps']} steps)")

        if aloha_skills:
            if not skill_lines:
                skill_lines += ["# GUI Demonstration Skills", ""]
            skill_lines += [
                "",
                "When the user asks you to perform any of these GUI tasks, "
                "you MUST use `gui_action_planned` with the matching skill_name. "
                "Do NOT use shell commands, exec, or other tools for these tasks.",
                "",
            ]
            for s in aloha_skills:
                desc = s.get('description') or s['name']
                aliases = s.get('aliases', [])
                alias_str = f" (also: {', '.join(aliases)})" if aliases else ""
                skill_lines.append(
                    f"- **{s['name']}**: {desc}{alias_str}\n"
                    f"  → `gui_action_planned(instruction=\"{desc}\", skill_name=\"{s['name']}\")`"
                )

        if skill_lines:
            lines += [""] + skill_lines
        return "\n".join(lines)

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        language_hint_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            skill_names: Optional skills to include.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.
            language_hint_text: Optional raw user text used only for reply-language alignment.

        Returns:
            List of messages including system prompt.
        """
        messages = []

        # System prompt
        system_prompt = self.build_system_prompt(skill_names)
        turn_language_note = build_turn_language_note(language_hint_text or current_message)
        if turn_language_note:
            system_prompt += f"\n\n## This Turn\n{turn_language_note}"
        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
        messages.append({"role": "system", "content": system_prompt})

        # History
        messages.extend(history)

        # Current message (with optional image attachments)
        user_content = self._build_user_content(current_message, media)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str | ToolResult
    ) -> list[dict[str, Any]]:
        """
        Add a tool result to the message list.

        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result (str or ToolResult with media).

        Returns:
            Updated message list.
        """
        if isinstance(result, ToolResult) and result.media:
            # Build multimodal content: images + text
            content_parts: list[dict[str, Any]] = []
            for path in result.media:
                p = Path(path)
                mime, _ = mimetypes.guess_type(path)
                if not p.is_file() or not mime or not mime.startswith("image/"):
                    continue
                b64 = base64.b64encode(p.read_bytes()).decode()
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"}
                })
            content_parts.append({"type": "text", "text": result.text})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": content_parts
            })
        else:
            text = result.text if isinstance(result, ToolResult) else result
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": text
            })
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        *,
        reasoning_content: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add an assistant message to the message list.

        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
            reasoning_content: Optional provider-specific reasoning text.

        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}

        if reasoning_content:
            msg["reasoning_content"] = reasoning_content

        if tool_calls:
            msg["tool_calls"] = tool_calls

        messages.append(msg)
        return messages
