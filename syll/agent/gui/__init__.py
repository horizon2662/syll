"""GUI agent layer (ALE-Claw-style separation).

This package hosts the three GUI layers:

- L1 ``primitive``: stateless single-step GUI actions (screenshot + one action).
- L2 ``planner``: skill/trajectory-aware planning that decides the next L1 call.
- L3: the main ``AgentLoop`` orchestrates iteration, memory, and recovery.

The current implementation is a thin facade over the existing proven tool code;
the boundary lets us migrate the internals incrementally without breaking
public tool names or tests.
"""

from syll.agent.gui.config import GuiAgentConfig, GuiConfig
from syll.agent.gui.memory import GuiMemory
from syll.agent.gui.primitive import GuiPrimitive, UITarsPrimitive
from syll.agent.gui.models import (
    GuiAction,
    GuiObservation,
    GuiStepResult,
    GuiStepStatus,
)

__all__ = [
    "GuiAction",
    "GuiAgentConfig",
    "GuiConfig",
    "GuiMemory",
    "GuiObservation",
    "GuiPrimitive",
    "GuiStepResult",
    "UITarsPrimitive",
    "GuiStepStatus",
]
