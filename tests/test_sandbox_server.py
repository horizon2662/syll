"""In-process tests for the FastAPI sandbox server.

Uses ``fastapi.testclient.TestClient`` against a ``LocalEnvironment`` backend so
the routes are exercised without a Docker daemon or a real HTTP socket. Pointer
and keyboard routes are intentionally not hit — they would drive the developer's
real desktop via pyautogui.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from syll.sandbox.environment import LocalEnvironment
from syll.sandbox.server import create_sandbox_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    backend = LocalEnvironment(workspace_root=tmp_path)
    return TestClient(create_sandbox_app(backend))


def test_health_reports_backend(client: TestClient):
    res = client.get("/sandbox/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert res.json()["backend"] == "LocalEnvironment"


def test_exec_runs_command(client: TestClient):
    res = client.post("/sandbox/exec", json={"command": "echo marker-123"})
    assert res.status_code == 200
    body = res.json()
    assert body["returncode"] == 0
    assert "marker-123" in body["stdout"]


def test_file_write_read_roundtrip(client: TestClient):
    w = client.post(
        "/sandbox/file",
        json={"action": "write_file", "path": "out/a.txt", "content": "hello"},
    )
    assert w.status_code == 200
    assert w.json() == {"ok": True}

    r = client.post(
        "/sandbox/file", json={"action": "read_file", "path": "out/a.txt"}
    )
    assert r.status_code == 200
    assert r.json()["content"] == "hello"

    ld = client.post(
        "/sandbox/file", json={"action": "list_dir", "path": "out"}
    )
    assert ld.status_code == 200
    assert ld.json()["entries"] == ["a.txt"]


def test_screenshot_returns_png(client: TestClient):
    res = client.get("/sandbox/screenshot")
    assert res.status_code == 200
    png = base64.b64decode(res.json()["base64_png"])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_screen_size_positive(client: TestClient):
    res = client.get("/sandbox/screen_size")
    assert res.status_code == 200
    body = res.json()
    assert body["width"] > 0 and body["height"] > 0


def test_unknown_file_action_is_400(client: TestClient):
    res = client.post(
        "/sandbox/file", json={"action": "nuke_disk", "path": "."}
    )
    assert res.status_code == 400


def test_local_restart_propagates_not_implemented(client: TestClient):
    # LocalEnvironment refuses reset; the server forwards to backend.reset,
    # which raises NotImplementedError. Starlette's TestClient re-raises server
    # exceptions by default, so we assert the raise (the Docker/VM backend is
    # what actually honours the lifecycle).
    with pytest.raises(NotImplementedError):
        client.post("/sandbox/restart")
