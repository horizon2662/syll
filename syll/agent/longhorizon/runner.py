"""End-to-end orchestrator: ties every fix together into a runnable system.

This is the main agent. It uses ONE subagent abstraction
(``UnifiedSubagentManager``), drives a hierarchical plan, and closes every
loop the first skeleton left open:

- **Fold + blackboard read** (issue 1, 7): each step is a subagent that runs
  in an isolated context and folds back a summary + artifact refs. Detail is
  inspectable on demand (the main agent can ``read_subagent`` if needed).
- **Unified abstraction** (issue 2): no ``SubagentManager`` / no
  ``GUIExecuteSubAgent`` here -- just ``UnifiedSubagentManager``.
- **Diagnosis -> Align replan** (issue 3): a subagent failure returns a
  ``diagnosis``; the main brain proposes replacement steps and
  ``HierarchicalPlanManager.align()`` splices them in.
- **Two-layer memory** (issue 4): ``GlobalMemory`` (PROJECT.md, main-agent)
  + ``SkillMemory`` (per-skill procedural, gated writes). The main agent
  distills a global slice + a JIT skill slice into each subagent's context.
- **JIT retrieval + compaction** (issue 5): ``SkillMemory.get_relevant()``
  caps injected skill memory; the orchestrator compacts its own notes.
- **Checkpoint / resume** (issue 6): plan state + orchestrator cursor are
  checkpointed after every step; a crashed run resumes where it stopped.

Run::

    python -m syll.agent.longhorizon.runner "research X and write a report" --skill research
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from syll.bus.queue import MessageBus
from syll.providers.litellm_provider import LiteLLMProvider

from .config import RunnerConfig
from .contract import SubagentResult
from .global_memory import GlobalMemory
from .hierarchical_plan_manager import HierarchicalPlanManager
from .session_state import SessionState
from .skill_memory import SkillMemory
from .unified_subagent import UnifiedSubagentManager

_MILESTONE_RE = re.compile(r"\[MILESTONE\s+(\d+)\]\s*(.*)")


def _extract_json(text: str) -> Any:
    """Robustly pull a JSON array/object out of an LLM response."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


class Runner:
    """The main orchestrator agent."""

    def __init__(self, cfg: RunnerConfig, provider: LiteLLMProvider):
        self.cfg = cfg
        self.provider = provider
        ws = cfg.workspace
        ws.mkdir(parents=True, exist_ok=True)

        self.bus = MessageBus()

        # Load the syll Config (~/.syll/config.json) so subagents get the SAME
        # GUI stack + model endpoints the ghost uses — reusing v2's setup verbatim.
        from syll.config.loader import load_config

        try:
            syll_cfg = load_config()
        except Exception:
            syll_cfg = None
        gui_cfg = syll_cfg.tools.gui if syll_cfg is not None else None

        # Optional event store for the GUI tools (None is fine — they degrade).
        try:
            from syll.agent.events import EventStore

            event_store = EventStore(ws.parent)
        except Exception:
            event_store = None

        self.subagents = UnifiedSubagentManager(
            provider=provider,
            workspace=ws,
            bus=self.bus,
            model=cfg.model,
            max_iterations=cfg.max_subagent_iterations,
            gui_config=gui_cfg,
            syll_config=syll_cfg,
            event_store=event_store,
        )
        self.hpm = HierarchicalPlanManager(ws, cfg.skill)
        self.skill_mem = SkillMemory(ws, cfg.skill)
        self.global_mem = GlobalMemory(ws)
        self.session = SessionState(ws, cfg.skill)
        self._notes: list[str] = []  # orchestrator working memory (compacted)
        self._compaction_summary: str = ""

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    async def run(self, task: str) -> str:
        print(f"\n=== Runner[{self.cfg.skill}] model={self.cfg.model} ===")
        print(f"task: {task}\n")

        resumed = self.session.resume()
        plan = self.hpm.resume()
        if resumed and plan is not None:
            self._compaction_summary = resumed.get("compaction_summary", "")
            cur_m = resumed.get("current_milestone", 1)
            done = sum(1 for s in plan.steps if s.status == "DONE")
            print(f"[resume] continuing at milestone {cur_m} "
                  f"({len(plan.steps)} steps, {done} done)")
        else:
            self.global_mem.init_goal(task)
            milestones = await self._propose_milestones(task)
            if not milestones:
                print("[runner] could not produce a plan; aborting.")
                return ""
            plan = self.hpm.create_milestone_plan(task, milestones)
            self.hpm.save_plan(plan)
            self._checkpoint(plan, current_milestone=1, current_step_index=0)
            print(f"[plan] {len(milestones)} milestone(s)\n")

        await self._execute_plan(plan)
        self.global_mem.log("ALL MILESTONES COMPLETE")
        print("\n=== DONE ===")
        print(f"PROJECT.md : {self.cfg.workspace / 'PROJECT.md'}")
        print(f"SKILL.md   : {self.cfg.workspace / 'skills' / self.cfg.skill / 'SKILL.md'}")
        return self.global_mem.load()

    # ------------------------------------------------------------------
    # plan execution
    # ------------------------------------------------------------------
    async def _execute_plan(self, plan) -> None:
        i = 0
        cur_milestone = 1
        replan_counts: dict[str, int] = {}
        while i < len(plan.steps):
            step = plan.steps[i]

            m = _MILESTONE_RE.search(step.description)
            if m:
                cur_milestone = int(m.group(1))
                title = m.group(2).strip()
                self.global_mem.log(f"-> Milestone {cur_milestone}: {title}")
                print(f"\n--- Milestone {cur_milestone}: {title} ---")
                self._checkpoint(plan, cur_milestone, step.index)
                i += 1
                continue

            if step.status in ("DONE", "SKIPPED"):
                i += 1
                continue

            print(f"[step {step.index}] {step.description}")
            result = await self.subagents.run_sync(
                task=step.description,
                label=f"M{cur_milestone}.S{step.index}",
                skill=self.cfg.skill,
                objective=step.description,
                context_slice=self._build_context_slice(step.description),
            )

            # --- verification gate: catch false success ---
            # A subagent may claim status=ok + artifacts it never actually wrote.
            # Check the claimed files exist in the workspace; if not, downgrade
            # to a failure so the replan loop re-does the work for real.
            if result.ok and result.artifacts:
                ok_art, miss = self._verify_artifacts(result)
                if not ok_art:
                    print(f"  ! FALSE SUCCESS: claimed {result.artifacts} -> {miss}")
                    self.skill_mem.ingest(
                        [f"False success on '{step.description[:60]}': claimed "
                         f"{result.artifacts} but {miss}. Do the work for real and "
                         f"write under {self.cfg.workspace}."],
                        status_ok=False,
                    )
                    self.global_mem.log(
                        f"M{cur_milestone}.S{step.index} FALSE SUCCESS: {miss}"
                    )
                    result = SubagentResult(
                        run_id=result.run_id,
                        status="failed",
                        summary=result.summary,
                        diagnosis=(f"False success: claimed artifacts "
                                   f"{result.artifacts} not found ({miss}). Re-do "
                                   f"the work and actually create the files under "
                                   f"{self.cfg.workspace}."),
                    )

            if result.ok:
                self.hpm.update_step(plan, step.index, "DONE", result.summary[:200])
                if result.lessons:
                    self.skill_mem.ingest(result.lessons, status_ok=True)
                self.global_mem.log(
                    f"M{cur_milestone}.S{step.index} ok: {result.summary[:120]}"
                )
                self._notes.append(f"step {step.index} ok: {result.summary[:200]}")
                i += 1
            else:
                key = f"{step.index}:{step.description[:40]}"
                attempts = replan_counts.get(key, 0)
                if attempts >= self.cfg.max_replans_per_step:
                    print(f"  ! max replans reached for step {step.index}; "
                          f"marking FAILED and moving on")
                    self.hpm.update_step(plan, step.index, "FAILED", result.diagnosis[:200])
                    self.skill_mem.ingest(
                        [f"Step '{step.description[:60]}' repeatedly failed: "
                         f"{result.diagnosis[:120]}"],
                        status_ok=False,
                    )
                    self.global_mem.log(
                        f"M{cur_milestone}.S{step.index} GAVE UP: {result.diagnosis[:120]}"
                    )
                    i += 1
                else:
                    replan_counts[key] = attempts + 1
                    new_steps = await self._propose_replan(step.description, result.diagnosis)
                    print(f"  ! FAILED ({result.diagnosis[:80]}) "
                          f"-> replan {len(new_steps)} step(s)")
                    failed_desc = step.description
                    plan = self.hpm.align(
                        plan, step.index, result.diagnosis, new_steps=new_steps or None
                    )
                    self.global_mem.log(
                        f"M{cur_milestone}.S{step.index} failed "
                        f"-> replan attempt {attempts + 1}"
                    )
                    i = self._first_replan_after(plan, failed_desc)

            self._checkpoint(plan, cur_milestone, step.index)
            await self._compact_if_needed()

    # ------------------------------------------------------------------
    # main-brain LLM calls (decision points)
    # ------------------------------------------------------------------
    async def _propose_milestones(self, task: str) -> list[tuple[str, list[str]]]:
        prompt = (
            "You are a planner. Break the task into 2-4 milestones, each with "
            "1-3 steps. Each step MUST be a concrete, directly-executable action "
            "(e.g. 'run: ls <dir>', 'read file <path>', 'write <content> to "
            "<path>'), NOT a vague goal like 'navigate', 'analyze', or "
            "'understand'. Respond ONLY with JSON: an array of objects "
            "{\"title\": str, \"steps\": [str]}.\n\n"
            f"Skill memory so far:\n{self.skill_mem.get_relevant(task)[:1500] or '(none)'}\n\n"
            f"Task: {task}"
        )
        resp = await self.provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.cfg.model,
            max_tokens=1500,
            temperature=0.3,
        )
        data = _extract_json(resp.content or "")
        out: list[tuple[str, list[str]]] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    title = str(item.get("title", "milestone"))
                    steps = [str(s) for s in item.get("steps", []) if s]
                    if steps:
                        out.append((title, steps))
        return out

    async def _propose_replan(self, step_desc: str, diagnosis: str) -> list[str]:
        prompt = (
            "A plan step failed. Propose 1-3 REPLACEMENT steps that achieve the "
            "same goal while avoiding the diagnosed failure. Respond ONLY with "
            "JSON: {\"steps\": [str]}.\n\n"
            f"Failed step: {step_desc}\n"
            f"Diagnosis: {diagnosis}\n"
            f"Relevant skill memory:\n"
            f"{self.skill_mem.get_relevant(step_desc)[:1200] or '(none)'}"
        )
        resp = await self.provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.cfg.model,
            max_tokens=800,
            temperature=0.3,
        )
        data = _extract_json(resp.content or "")
        if isinstance(data, dict):
            steps = data.get("steps", [])
            return [str(s) for s in steps if s]
        if isinstance(data, list):
            return [str(s) for s in data if s]
        return []

    async def _compact_notes(self) -> str:
        """Anthropic-style compaction: summarize accumulated notes into one block."""
        if not self._notes:
            return self._compaction_summary
        prompt = (
            "Compress these step outcomes into a concise progress summary "
            "(<= 200 words). Keep decisions, what worked, and open issues.\n\n"
            + "\n".join(self._notes)
        )
        resp = await self.provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.cfg.model,
            max_tokens=500,
            temperature=0.2,
        )
        return (resp.content or "").strip()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _verify_artifacts(self, result: SubagentResult) -> tuple[bool, str]:
        """TVAE-style verification gate: artifacts a subagent claims to have
        produced must actually exist. Turns a hallucinated 'ok' (false success)
        into a detectable failure. Relative paths resolve against the workspace."""
        missing = []
        for art in result.artifacts:
            p = Path(art)
            if not p.is_absolute():
                p = self.cfg.workspace / art
            if not p.exists():
                missing.append(art)
        if missing:
            return False, f"missing in workspace: {missing}"
        return True, ""

    def _build_context_slice(self, step_desc: str) -> str:
        g = self.global_mem.distill_for_skill(self.cfg.skill, step_desc)
        s = self.skill_mem.get_relevant(step_desc)
        parts = [p for p in (g, s) if p]
        if self._compaction_summary:
            parts.append(f"(progress so far)\n{self._compaction_summary}")
        return "\n\n".join(parts)

    async def _compact_if_needed(self) -> None:
        """Anthropic-style compaction of the orchestrator's running notes."""
        if len(self._notes) >= self.cfg.compaction_threshold:
            self._compaction_summary = await self._compact_notes()
            self._notes = [self._compaction_summary]
            logger.info("orchestrator compacted its notes")

    def _checkpoint(self, plan, current_milestone: int, current_step_index: int) -> None:
        self.hpm.save_plan(plan)
        self.hpm.checkpoint(plan)
        self.session.checkpoint(
            current_milestone=current_milestone,
            current_step_index=current_step_index,
            running_runs=list(self.subagents._running.keys()),
            compaction_summary=self._compaction_summary,
        )

    @staticmethod
    def _first_replan_after(plan, failed_desc: str) -> int:
        """Index to continue at after a replan: the step right after the FAILED one.

        ``align()`` marks the failed step FAILED and splices [REPLAN] steps
        immediately after it, then re-indexes sequentially. We resume at the
        first step following the FAILED step."""
        for idx, step in enumerate(plan.steps):
            if step.description == failed_desc and step.status == "FAILED":
                return min(idx + 1, len(plan.steps) - 1)
        # Fallback: start over (should not happen).
        return 0


def _main() -> int:
    ap = argparse.ArgumentParser(description="Syll long-horizon runner (end-to-end)")
    ap.add_argument("task", help="The top-level task to accomplish")
    ap.add_argument("--skill", default="default", help="Skill name (per-skill memory scope)")
    ap.add_argument("--workspace", default=None, help="Workspace dir (default ~/.syll)")
    ap.add_argument("--model", default=None, help="Override SYLL_MODEL")
    args = ap.parse_args()

    cfg = RunnerConfig.from_env(skill=args.skill, workspace=args.workspace)
    if args.model:
        cfg.model = args.model

    if not cfg.api_key:
        print(
            "ERROR: no API key. Set SYLL_API_KEY (or ZHIPUAI_API_KEY) and optionally\n"
            "SYLL_MODEL / SYLL_API_BASE. Example:\n"
            "  export SYLL_API_KEY=...\n"
            "  export SYLL_MODEL=glm-4.6\n"
            "  python -m syll.agent.longhorizon.runner 'task' --skill research",
            file=sys.stderr,
        )
        return 2

    if cfg.use_anthropic_sdk:
        # GLM/Zhipu /api/anthropic: must auth via Bearer (ANTHROPIC_AUTH_TOKEN),
        # which litellm does not do. Use the Anthropic SDK provider instead.
        from .anthropic_provider import AnthropicMessagesProvider

        provider = AnthropicMessagesProvider(
            auth_token=cfg.api_key, base_url=cfg.api_base, default_model=cfg.model
        )
    else:
        provider = LiteLLMProvider(
            api_key=cfg.api_key, api_base=cfg.api_base, default_model=cfg.model
        )
    runner = Runner(cfg, provider)
    try:
        asyncio.run(runner.run(args.task))
    except KeyboardInterrupt:
        print("\n[interrupt] state checkpointed; re-run to resume.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
