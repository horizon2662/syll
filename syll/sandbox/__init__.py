"""Sandbox environment abstraction layer for Syll.

Provides a unified ``Environment`` interface so that core tools (file system,
shell, screenshot, pointer/keyboard) can run either locally on the user's
machine or inside a future Docker sandbox container without changing tool
semantics.
"""

from .environment import Environment, ExecResult, LocalEnvironment, SafetyResult
from .protocol import (
    SANDBOX_ACTIONS,
    ExecRequest,
    FileRequest,
    SandboxProtocol,
    ScreenshotResponse,
)

__all__ = [
    "Environment",
    "ExecResult",
    "LocalEnvironment",
    "SafetyResult",
    "SandboxProtocol",
    "ExecRequest",
    "FileRequest",
    "ScreenshotResponse",
    "SANDBOX_ACTIONS",
]
