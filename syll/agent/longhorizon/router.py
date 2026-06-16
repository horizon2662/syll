"""Auto-router: detect GUI/desktop tasks and hand them to the v3 (longhorizon) pipeline.

When a user task requires operating the graphical desktop (browsers, apps,
clicking/typing into windows, screenshots), route it to the longhorizon Runner
so it gets the v3 machinery — fold + subagent + verification gate + checkpoint/
replan + two-layer memory — instead of a single-shot GUI action.

Non-GUI tasks are left untouched (the normal ghost flow handles them).

The classifier is keyword-fast-pathed (obvious GUI signals skip the LLM call)
with a cheap LLM fallback for ambiguous cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from syll.providers.base import LLMProvider

# Strong GUI signals -> classify as GUI without an LLM call.
_GUI_KEYWORDS = (
    "open chrome", "open the browser", "open firefox", "open edge", "open safari",
    "on the screen", "on screen", "click ", "click on", "right-click", "double-click",
    "scroll down", "scroll up", "mouse", "cursor",
    "open the app", "open settings", "system settings", "control panel", "system tray",
    "taskbar", "dock", "menu bar",
    "take a screenshot", "screenshot of", "capture the screen",
    "minimize the window", "close the window", "open a new tab", "switch to the tab",
    "drag the", "drop the",
)


def _keyword_is_gui(message: str) -> bool:
    low = message.lower()
    return any(k in low for k in _GUI_KEYWORDS)


async def is_gui_task(
    provider: LLMProvider, message: str, model: str | None = None
) -> bool:
    """Classify whether a task needs graphical-desktop operation."""
    if _keyword_is_gui(message):
        return True
    # Cheap LLM fallback for ambiguous cases.
    try:
        resp = await provider.chat(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Does this user task require operating a graphical desktop "
                        "UI (opening desktop apps or browser windows, clicking, "
                        "typing into app windows, taking screenshots)? Tasks that "
                        "only need files, shell, code, or web search are NOT GUI. "
                        "Reply with ONE word: GUI or NOTGUI.\n\nTask: " + message
                    ),
                }
            ],
            model=model,
            max_tokens=8,
            temperature=0.0,
        )
        text = (resp.content or "").strip().lower()
        return text.startswith("gui")
    except Exception:
        return False


async def run_gui_via_v3(
    task: str,
    *,
    workspace: Path | str,
    provider: LLMProvider,
    model: str | None = None,
    skill: str = "gui",
) -> str:
    """Run a GUI task end-to-end through the v3 (longhorizon) pipeline.

    Reuses the caller's already-configured provider (the ghost's), so model/auth
    stay consistent. Returns a summary string the caller can reply with; the
    full audit trail lands in PROJECT.md / SKILL.md / agents/ under the workspace.
    """
    from syll.agent.longhorizon.config import RunnerConfig
    from syll.agent.longhorizon.runner import Runner

    cfg = RunnerConfig.from_env(skill=skill, workspace=workspace)
    if model:
        cfg.model = model
    runner = Runner(cfg, provider)
    await runner.run(task)
    summary = runner.global_mem.load()
    return summary or "(v3 pipeline finished; see PROJECT.md in the workspace)"
