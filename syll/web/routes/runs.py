"""Run-audit dashboard routes — mirrors ``routes/recorder.py``.

Surfaces per-run context-curve + status + blackboard that the agent loop and
``UnifiedSubagentManager`` write under ``{workspace}/audit/`` and
``{workspace}/longhorizon_runs/*/audit/``. The manager lives on
``app.state.runs_manager`` (see ``app.py``); it reads from disk so no live
ContextMeter handle is needed.

Legacy ``/runs/plan``, ``/runs/events`` and ``/runs/metrics`` endpoints have
been removed; the subagent blackboard at ``{run}/agents/{run_id}/result.json``
is now the single source of truth for run outcomes.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from syll.web.runs_manager import RunsManager

router = APIRouter(prefix="/runs", tags=["runs"])


def _get_manager(request: Request) -> RunsManager:
    mgr = getattr(request.app.state, "runs_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="runs manager not initialized")
    return mgr


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


@router.get("")
@router.get("/")
async def list_runs(request: Request):
    """All known runs, newest first."""
    return _get_manager(request).list_runs()


@router.get("/curve")
async def get_curve(
    request: Request,
    dir: str = Query(..., description="workspace-relative audit dir (run id)"),
):
    """Full context_curve.jsonl as a list of points."""
    try:
        return _get_manager(request).read_curve(dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/status")
async def get_status(
    request: Request,
    dir: str = Query(..., description="workspace-relative audit dir (run id)"),
):
    """Live snapshot of one run: last/peak tokens, utilization, blackboard path."""
    try:
        return _get_manager(request).status(dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/performance")
async def get_performance(request: Request):
    """Aggregate usage across all runs, split main-agent vs subagent.

    For the Performance tab: how much context the main agent (orchestrator)
    vs subagents consume, with per-run breakdown.
    """
    return _get_manager(request).performance_summary()


@router.get("/blackboard")
async def get_blackboard(
    request: Request,
    dir: str = Query(..., description="workspace-relative audit dir (run id)"),
):
    """Subagent blackboard for a run: result.json + progress.md.

    This replaces the legacy ``/runs/plan``, ``/runs/events`` and
    ``/runs/metrics`` endpoints. The blackboard is the single source of truth
    for a longhorizon run's outcome.
    """
    try:
        return _get_manager(request).read_blackboard(dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/stream")
async def stream_curve(
    request: Request,
    dir: str = Query(..., description="workspace-relative audit dir to tail"),
):
    """SSE: yield each new curve point as it's appended.

    Mirrors the recorder's live-status SSE — the dashboard polls this to draw
    the growing context curve without re-fetching the whole file.
    """
    mgr = _get_manager(request)

    async def event_stream():
        seen = 0
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    pts = mgr.read_curve(dir)
                except ValueError as exc:
                    yield _sse("error", {"detail": str(exc)})
                    break
                if len(pts) > seen:
                    for p in pts[seen:]:
                        yield _sse("point", p)
                    seen = len(pts)
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            return

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── GUI pipeline launcher (option C: a dedicated entry that runs the v3
#    longhorizon pipeline; normal chat is untouched). The pipeline writes the
#    context curve under {WS}/longhorizon_runs/{session}/audit/ and the subagent
#    blackboard under {WS}/longhorizon_runs/{session}/agents/{run_id}/, so the
#    Runs tab and /runs/blackboard pick it up automatically. This route is just
#    the trigger + a status feed.
# ─────────────────────────────────────────────────────────────────────────


class GuiRunRequest(BaseModel):
    task: str


def _gui_runs(request: Request) -> dict:
    """session -> asyncio.Task (the background runner). Lazily created."""
    d = getattr(request.app.state, "gui_runs", None)
    if d is None:
        d = {}
        request.app.state.gui_runs = d
    return d


@router.get("/actions")
async def get_actions(
    request: Request,
    dir: str = Query(..., description="workspace-relative audit dir (run id)"),
):
    """Grounded-action rows (model 0-1000 + executor screen px) for one run."""
    try:
        return _get_manager(request).read_actions(dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/actions/recent")
async def recent_actions(
    request: Request,
    limit: int = Query(20, ge=1, le=200, description="how many recent actions to return"),
):
    """Most recent grounded actions across ALL runs — the coord feed for the
    Performance tab (model 0-1000 → executor px, per action, with run id)."""
    return _get_manager(request).recent_actions(limit=limit)


@router.post("/gui/run")
async def run_gui_pipeline(request: Request, body: GuiRunRequest):
    """Launch a GUI task through the v3 (longhorizon) pipeline in the background.

    Returns immediately with the session + audit_dir. The pipeline reuses the
    agent loop's already-configured provider (model/auth stay consistent).
    Monitor via /runs/blackboard (subagent result), /runs/curve, /runs/gui/status,
    or the Runs tab.
    """
    agent_loop = getattr(request.app.state, "agent_loop", None)
    if agent_loop is None:
        raise HTTPException(status_code=503, detail="agent loop not available")
    task = (body.task or "").strip()
    if not task:
        raise HTTPException(status_code=400, detail="task is required")
    mgr = _get_manager(request)
    session = f"gui_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    run_ws = mgr.workspace / "longhorizon_runs" / session
    from syll.agent.longhorizon.router import run_gui_via_v3

    bg = asyncio.create_task(
        run_gui_via_v3(
            task,
            workspace=run_ws,
            provider=agent_loop.provider,
            model=agent_loop.model,
        )
    )
    _gui_runs(request)[session] = bg
    return {
        "session": session,
        "audit_dir": f"longhorizon_runs/{session}/audit",
        "task": task,
    }


@router.get("/gui/status")
async def gui_status(
    request: Request,
    session: str = Query(..., description="gui session id from /gui/run"),
):
    """Whether the background GUI run is finished (and its exception, if any)."""
    bg = _gui_runs(request).get(session)
    if bg is None:
        raise HTTPException(status_code=404, detail="unknown gui session")
    exc = bg.exception() if bg.done() else None
    cancelled = bg.cancelled()
    return {
        "session": session,
        "done": bg.done(),
        "cancelled": cancelled,
        "error": None if cancelled else (str(exc) if exc else None),
    }


@router.post("/gui/cancel")
async def cancel_gui_pipeline(
    request: Request,
    session: str = Query(..., description="gui session id to cancel"),
):
    """Cancel a background GUI run.

    ``bg.cancel()`` injects ``CancelledError`` at the pipeline's next await.
    Note: the legacy session-state resume path has been removed; cancellation
    stops the run but does not preserve an in-progress plan checkpoint.
    """
    bg = _gui_runs(request).get(session)
    if bg is None:
        raise HTTPException(status_code=404, detail="unknown gui session")
    if not bg.done():
        bg.cancel()
    return {"session": session, "cancelled": True}
