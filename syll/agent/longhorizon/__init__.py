"""Syll long-horizon refactor skeleton.

A self-contained, end-to-end-runnable implementation of the unified
main/sub-agent architecture described in ROADMAP.md. Imports the installed
``syll`` package's primitives (provider, tools, bus) but adds its own
orchestrator (``runner``) so the full loop can be exercised without
modifying site-packages.

Run::

    python -m syll.agent.longhorizon.runner "your task here"
"""

from .contract import SubagentContract, SubagentResult
from .blackboard import Blackboard
from .skill_memory import SkillMemory
from .global_memory import GlobalMemory
from .unified_subagent import UnifiedSubagentManager, ReturnTool
from .hierarchical_plan_manager import HierarchicalPlanManager, Milestone

__all__ = [
    "SubagentContract",
    "SubagentResult",
    "Blackboard",
    "SkillMemory",
    "GlobalMemory",
    "UnifiedSubagentManager",
    "ReturnTool",
    "HierarchicalPlanManager",
    "Milestone",
]
