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
    """Run a GUI task end-to-end through the v3 (longhorizon) subagent.

    Reuses the caller's already-configured provider (the ghost's), so model/auth
    stay consistent. Spawns one isolated subagent that drives ``gui_action`` /
    ``gui_action_planned`` repeatedly until the task is done, then folds its
    result back to the caller.
    """
    from syll.agent.events import EventStore
    from syll.agent.longhorizon.config import RunnerConfig
    from syll.agent.longhorizon.context_meter import ContextMeter, resolve_context_window
    from syll.agent.longhorizon.skill_memory import SkillMemory
    from syll.agent.longhorizon.unified_subagent import UnifiedSubagentManager
    from syll.bus.queue import MessageBus
    from syll.config.loader import load_config
    from syll.sandbox.environment import LocalEnvironment

    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)

    cfg = RunnerConfig.from_env(skill=skill, workspace=ws)
    if model:
        cfg.model = model

    try:
        syll_cfg = load_config()
    except Exception:
        syll_cfg = None

    gui_cfg = None
    if syll_cfg is not None:
        try:
            gui_cfg = syll_cfg.tools.gui
        except Exception:
            gui_cfg = None

    try:
        event_store = EventStore(ws.parent)
    except Exception:
        event_store = None

    meter = ContextMeter(
        run_dir=ws / "audit",
        run_id=ws.name,
        budget_tokens=resolve_context_window(cfg.model, cfg.context_window),
    )

    manager = UnifiedSubagentManager(
        provider=provider,
        workspace=ws,
        bus=MessageBus(),
        model=cfg.model,
        max_iterations=cfg.max_subagent_iterations,
        gui_config=gui_cfg,
        syll_config=syll_cfg,
        event_store=event_store,
        context_meter=meter,
        skill_memory=SkillMemory(ws, skill),
        environment=LocalEnvironment(workspace_root=ws),
    )

    result = await manager.run_sync(
        task=task,
        label="longhorizon",
        skill=skill,
        objective=task,
        mode="step",
    )

    if result.ok:
        return result.summary or "(v3 subagent completed with no summary)"
    return (
        f"(v3 subagent failed: {result.diagnosis or 'unknown failure'})"
        f"\n\nLast summary: {result.summary or '(none)'}"
    ).strip()
