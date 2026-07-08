"""Data models for the GUI agent layer.

These are plain dataclasses so L1/L2/L3 can pass structured GUI state without
depending on the internal tool implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GuiStepStatus(str, Enum):
    """Outcome of a single GUI primitive step."""

    SUCCESS = "SUCCESS"
    NO_CHANGE = "NO_CHANGE"
    UNCERTAIN = "UNCERTAIN"
    ERROR = "ERROR"
    DONE = "DONE"


@dataclass
class GuiAction:
    """One GUI action to execute."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuiObservation:
    """Observation returned after a GUI action."""

    screenshot_path: str | None = None
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuiStepResult:
    """Result of one L1 primitive step."""

    status: GuiStepStatus
    action: GuiAction | None = None
    observation: GuiObservation | None = None
    reasoning: str = ""
    diagnosis: str = ""
    category: str = ""
    next_hint: str = ""
    media: list[str] = field(default_factory=list)
