"""Memory API routes."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(tags=["memory"])


def _read_store(store: object, filename: str | None = None) -> dict:
    """Serialize a MemoryStore (or MemoryStore-like object)."""
    if filename:
        file_path = Path(store.memory_dir) / filename
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Memory file not found")
        return {
            "filename": filename,
            "content": file_path.read_text(encoding="utf-8"),
        }

    files = store.list_memory_files()
    return {
        "long_term": store.read_long_term(),
        "today": store.read_today(),
        "files": [{"name": f.name, "path": str(f)} for f in files],
    }


@router.get("/memory")
async def memory_overview(
    request: Request,
    scope: str = Query("merged", enum=["merged", "global", "workspace"]),
):
    """Return memory overview for the requested scope.

    - ``global``: user-scoped global memory.
    - ``workspace``: current workspace-local memory.
    - ``merged``: global as the baseline, workspace as overlay (default).
    """
    global_store = request.app.state.memory_store
    workspace_store = getattr(request.app.state, "workspace_memory_store", None)

    if scope == "global":
        return _read_store(global_store)

    if scope == "workspace":
        if workspace_store is None:
            raise HTTPException(status_code=404, detail="Workspace memory not available")
        return _read_store(workspace_store)

    # merged
    global_data = _read_store(global_store)
    merged = {
        "long_term": global_data["long_term"],
        "today": global_data["today"],
        "files": list(global_data["files"]),
    }

    if workspace_store is not None:
        ws_data = _read_store(workspace_store)
        if ws_data["long_term"]:
            merged["long_term"] = (
                merged["long_term"].rstrip() + "\n\n" + ws_data["long_term"]
            ).strip()
        if ws_data["today"]:
            merged["today"] = (
                merged["today"].rstrip() + "\n\n" + ws_data["today"]
            ).strip()
        # Merge file lists and dedupe by name, keeping newest first.
        seen = {f["name"] for f in merged["files"]}
        for f in ws_data["files"]:
            if f["name"] not in seen:
                merged["files"].append(f)
                seen.add(f["name"])
        merged["files"].sort(key=lambda x: x["name"], reverse=True)

    return merged


@router.get("/memory/{filename}")
async def read_memory_file(
    filename: str,
    request: Request,
    scope: str = Query("merged", enum=["merged", "global", "workspace"]),
):
    """Read a specific memory file from the requested scope."""
    global_store = request.app.state.memory_store
    workspace_store = getattr(request.app.state, "workspace_memory_store", None)

    if scope == "global":
        return _read_store(global_store, filename)

    if scope == "workspace":
        if workspace_store is None:
            raise HTTPException(status_code=404, detail="Workspace memory not available")
        return _read_store(workspace_store, filename)

    # merged — prefer workspace, fall back to global.
    if workspace_store is not None:
        try:
            return _read_store(workspace_store, filename)
        except HTTPException:
            pass
    return _read_store(global_store, filename)
