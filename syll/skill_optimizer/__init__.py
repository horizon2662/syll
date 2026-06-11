"""SkillOpt-inspired skill optimizer for Syll.

Treats a Syll skill markdown document as a trainable parameter and
optimizes it through a 6-stage ReflACT pipeline adapted from Microsoft
SkillOpt (https://github.com/microsoft/SkillOpt):

  1. Rollout   — execute tasks with current skill via AgentLoop
  2. Reflect   — LLM analyzes trajectories, generates patches
  3. Aggregate — merge patches from multiple minibatches
  4. Select    — rank and select top edits within budget
  5. Update    — apply edits to the skill document
  6. Evaluate  — validate candidate on held-out set, accept/reject

Usage::

    from syll.skill_optimizer import SkillOptimizer
    from syll.skill_optimizer.rollout import load_tasks

    tasks = load_tasks("tasks.json")
    optimizer = SkillOptimizer(
        skill_name="file-retrieval",
        tasks=tasks,
        eval_tasks=[],
        provider=provider,
        model=model,
        agent_loop=agent_loop,
        workspace=workspace,
    )
    summary = await optimizer.train()
"""
from __future__ import annotations

from syll.skill_optimizer.trainer import SkillOptimizer

__all__ = ["SkillOptimizer"]
