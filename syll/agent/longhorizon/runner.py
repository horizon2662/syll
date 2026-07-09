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
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from loguru import logger

from syll.bus.queue import MessageBus
from syll.providers.litellm_provider import LiteLLMProvider
from syll.sandbox.environment import LocalEnvironment

from .config import RunnerConfig
from .contract import SubagentResult
from .telemetry import DecisionUnit, log_node, node_key as make_node_key
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
        self._gui_enabled = gui_cfg is not None and getattr(gui_cfg, "enabled", False)

        # Optional event store for the GUI tools (None is fine — they degrade).
        try:
            from syll.agent.events import EventStore

            event_store = EventStore(ws.parent)
        except Exception:
            event_store = None
        self.event_store = event_store
        self.run_id = str(uuid.uuid4())  # one telemetry episode id per process run
        self._seq = 0
        # Context-length detector: captures the token cost the provider already
        # returns (previously discarded). Writes {ws}/audit/context_curve.jsonl.
        # Phase 1.5: budget resolved from cfg.context_window (SYLL_CONTEXT_WINDOW)
        # > litellm table > 0. With a real budget, utilization/over_budget light up.
        try:
            from .context_meter import ContextMeter, resolve_context_window
            self.meter = ContextMeter(
                run_dir=ws / "audit",
                run_id=self.run_id,
                budget_tokens=resolve_context_window(
                    self.cfg.model, getattr(self.cfg, "context_window", 0)
                ),
            )
        except Exception as exc:
            logger.debug(f"context meter disabled: {exc}")
            self.meter = None

        self.environment = LocalEnvironment(workspace_root=ws)
        self.skill_mem = SkillMemory(ws, cfg.skill)
        self.subagents = UnifiedSubagentManager(
            provider=provider,
            workspace=ws,
            bus=self.bus,
            model=cfg.model,
            max_iterations=cfg.max_subagent_iterations,
            gui_config=gui_cfg,
            syll_config=syll_cfg,
            event_store=event_store,
            context_meter=self.meter,
            skill_memory=self.skill_mem,
            environment=self.environment,
        )
        self.hpm = HierarchicalPlanManager(ws, cfg.skill)
        self.global_mem = GlobalMemory(ws)
        self.session = SessionState(ws, cfg.skill)
        self._notes: list[str] = []  # orchestrator working memory (compacted)
        self._compaction_summary: str = ""
        # Phase 3: online evolution (default off). On step failure the Evolver
        # (diagnose -> patch -> validate -> distill) may learn a code-as-policy
        # skill for next time, gated by the verifier-ceiling β estimate. Uses the
        # same global code-skill library the agent reads (workspace/code_skills).
        # Generation runs against self.environment; for unverified-code isolation
        # prefer a Docker sandbox env_factory.
        self._evolve_pass = 0
        self._evolve_false_success = 0
        self.evolver = None
        if getattr(self.cfg, "enable_evolution", False):
            from .evolution import Evolver

            self.evolver = Evolver(
                self.subagents.code_skill_library,
                env_factory=lambda: LocalEnvironment(workspace_root=ws),
                provider=self.provider,
                model=cfg.model,
                beta_threshold=getattr(cfg, "evolution_beta_threshold", 0.3),
                variants=getattr(cfg, "evolution_variants", 3),
            )

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
            seq = self._next_seq()
            # retry-same-node loop (budget 0 under default recovery_mode="replan",
            # so this runs exactly once and control flow is unchanged).
            retry_budget = (
                self.cfg.max_retries_per_step
                if self.cfg.recovery_mode in ("retry_same_node", "retry_then_replan")
                else 0
            )
            result = None
            prior = None
            for attempt in range(1, retry_budget + 2):
                result = await self._attempt_step(
                    plan, step, cur_milestone, attempt, seq, prior_result=prior
                )
                if result.ok:
                    break
                prior = result  # fed back into context iff retry_context == "replay"
                if attempt <= retry_budget:
                    print(f"  ~ attempt {attempt} failed -> retry_same_node "
                          f"({self.cfg.retry_context}) [{attempt}/{retry_budget}]")

            if not result.ok:
                # Phase 3: fire-and-forget skill evolution on failure (default
                # off + β-gated inside; never blocks the replan/give-up flow).
                await self._maybe_evolve(plan, step, cur_milestone, result)

            if result.ok:
                self._record_step_success(plan, step, cur_milestone, result)
                i += 1
            elif self.cfg.recovery_mode in ("replan", "retry_then_replan"):
                key = f"{step.index}:{step.description[:40]}"
                attempts = replan_counts.get(key, 0)
                if attempts >= self.cfg.max_replans_per_step:
                    print(f"  ! max replans reached for step {step.index}; "
                          f"marking FAILED and moving on")
                    self._give_up_step(
                        plan, step, cur_milestone, result,
                        lesson=(f"Step '{step.description[:60]}' repeatedly failed: "
                                f"{result.diagnosis[:120]}"),
                        log_msg=f"GAVE UP: {result.diagnosis[:120]}",
                    )
                    i += 1
                else:
                    replan_counts[key] = attempts + 1
                    new_steps = await self._propose_replan(
                        step.description, result.diagnosis,
                        plan=plan, focus_step_index=step.index,
                    )
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
            else:
                # recovery_mode == "retry_same_node": retries exhausted -> give up
                print(f"  ! retries exhausted for step {step.index}; marking FAILED")
                self._give_up_step(
                    plan, step, cur_milestone, result,
                    lesson=(f"Step '{step.description[:60]}' failed after {retry_budget} "
                            f"retries: {result.diagnosis[:120]}"),
                    log_msg=f"GAVE UP after {retry_budget} retries",
                )
                i += 1

            self._checkpoint(plan, cur_milestone, step.index)
            await self._compact_if_needed()

    def _record_step_success(self, plan, step, cur_milestone, result) -> None:
        """Mark a step DONE, ingest its lessons, and log/record the outcome."""
        self.hpm.update_step(plan, step.index, "DONE", result.summary[:200])
        if result.lessons:
            self.skill_mem.ingest(result.lessons, status_ok=True)
        self.global_mem.log(
            f"M{cur_milestone}.S{step.index} ok: {result.summary[:120]}"
        )
        self._notes.append(f"step {step.index} ok: {result.summary[:200]}")

    @property
    def _evolution_beta(self) -> float | None:
        """Running false-success rate of the agent's self-judgement (β)."""
        if self._evolve_pass == 0:
            return None
        return self._evolve_false_success / self._evolve_pass

    async def _maybe_evolve(self, plan, step, cur_milestone, result) -> None:
        """Phase 3 hook: on step failure, try to learn a code-as-policy skill.

        Default-off (``cfg.enable_evolution``) and β-gated inside the Evolver.
        Builds a verifier gate from the step's expected artifacts; fire-and-forget
        — logs the report and never blocks the normal replan / give-up flow.
        """
        if self.evolver is None:
            return
        artifacts = list(getattr(result, "artifacts", []) or [])
        if not artifacts:
            logger.debug(f"[evolve] step {step.index}: no artifacts to gate on; skip")
            return
        from .evolution import FailureCase

        failure = FailureCase(
            instruction=step.description,
            diagnosis=(result.diagnosis or "")[:500],
            checks=[{"type": "file", "require_exists": artifacts}],
            trace_summary=(result.summary or "")[:400],
        )
        beta = self._evolution_beta
        report = await self.evolver.evolve(failure, beta=beta)
        if report.evolved:
            logger.info(
                f"[evolve] step {step.index}: installed {report.installed_names} "
                f"(β={beta})"
            )
            self.global_mem.log(
                f"M{cur_milestone}.S{step.index} evolved skills {report.installed_names}"
            )
        else:
            logger.info(f"[evolve] step {step.index}: {report.reason} (β={beta})")

    def _give_up_step(
        self, plan, step, cur_milestone, result, *, lesson: str, log_msg: str
    ) -> None:
        """Mark a step FAILED, ingest its pitfall as a skill lesson, log GAVE UP.

        Shared by the 'max replans reached' and 'retries exhausted' give-up
        paths (previously two near-identical open-coded blocks)."""
        self.hpm.update_step(plan, step.index, "FAILED", result.diagnosis[:200])
        self.skill_mem.ingest([lesson], status_ok=False)
        self.global_mem.log(f"M{cur_milestone}.S{step.index} {log_msg}")

    # ------------------------------------------------------------------
    # main-brain LLM calls (decision points)
    # ------------------------------------------------------------------
    async def _propose_milestones(self, task: str) -> list[tuple[str, list[str]]]:
        gui_clause = (
            "\n\nGUI RULE: if a step requires operating a graphical/desktop app "
            "(opening apps, clicking, typing into windows, drawing shapes, "
            "navigating menus/tabs), express it as ONE gui_action_planned step "
            "with a visual instruction (e.g. 'gui_action_planned: click the "
            "Shapes button on the Insert tab and draw a rectangle on the "
            "slide'). DO NOT script GUI work with shell/PowerShell/COM/SendKeys "
            "— that bypasses the vision actor and is forbidden. Reserve "
            "exec/shell for non-GUI work (scripts, deps, files)."
            if self._gui_enabled
            else ""
        )
        base_prompt = (
            "You are a planner. Break the task into 2-4 milestones, each with "
            "1-3 steps. Each step MUST be a concrete, directly-executable action "
            "(e.g. 'run: ls <dir>', 'read file <path>', 'write <content> to "
            "<path>', or for GUI: 'gui_action_planned: <visual goal>'), NOT a "
            "vague goal like 'navigate', 'analyze', or 'understand'. Respond "
            "ONLY with JSON: an array of objects {\"title\": str, \"steps\": [str]}."
            + gui_clause
            + f"\n\nSkill memory so far:\n{self.skill_mem.get_relevant(task)[:1500] or '(none)'}\n\n"
            f"Task: {task}"
        )
        prompt = base_prompt
        out: list[tuple[str, list[str]]] = []
        for _attempt in range(3):
            try:
                text = await self._orchestrator_llm(
                    prompt, label="plan", max_tokens=1500
                )
                data = _extract_json(text)
            except Exception:
                data = None
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        title = str(item.get("title", "milestone"))
                        steps = [str(s) for s in item.get("steps", []) if s]
                        if steps:
                            out.append((title, steps))
            if out:
                return out
            # Retry with a sterner "JSON only" instruction.
            prompt = base_prompt + (
                "\n\nIMPORTANT: reply with ONLY the JSON array — no prose, "
                "no code fence, no explanation."
            )
        # Fallback so a transient bad LLM response never aborts the whole run.
        logger.warning(
            "planner produced no usable plan after retries; "
            "falling back to task-as-single-milestone"
        )
        return [(task or "task", [task] if task else [])]

    async def _propose_replan(
        self, step_desc: str, diagnosis: str, *, plan=None, focus_step_index=None
    ) -> list[str]:
        progress = (
            self.hpm.render_progress(
                plan, focus_step_index=focus_step_index, last_diagnosis=diagnosis
            )
            if plan is not None
            else ""
        )
        prompt = (
            "A plan step failed. Propose 1-3 REPLACEMENT steps that achieve the "
            "same goal while avoiding the diagnosed failure. Respond ONLY with "
            "JSON: {\"steps\": [str]}.\n\n"
            f"Failed step: {step_desc}\n"
            f"Diagnosis: {diagnosis}\n"
        )
        if progress:
            # Plan-wide context so replan respects what's already done and what
            # depends on this step (long-horizon anti-disorientation).
            prompt += f"\n{progress}\n\n"
        prompt += (
            "Relevant skill memory:\n"
            f"{self.skill_mem.get_relevant(step_desc)[:1200] or '(none)'}"
        )
        text = await self._orchestrator_llm(prompt, label="replan", max_tokens=800)
        data = _extract_json(text)
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
        text = await self._orchestrator_llm(
            prompt, label="compact", max_tokens=500, temperature=0.2
        )
        return text.strip()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _meter_orchestrator(self, resp, label: str = "") -> None:
        """Record the orchestrator's OWN LLM cost (plan/replan/compact) so the
        Performance tab's main-vs-subagent split is accurate for longhorizon
        runs. Without this, longhorizon runs would look 100% subagent."""
        if self.meter is None or resp is None:
            return
        try:
            u = getattr(resp, "usage", None) or {}
            self.meter.record(
                prompt_tokens=int(u.get("prompt_tokens", 0) or 0),
                completion_tokens=int(u.get("completion_tokens", 0) or 0),
                phase="orchestrator",
                extra={"orchestrator": label},
            )
        except Exception:  # metering must never break a run
            pass

    async def _orchestrator_llm(
        self,
        prompt: str,
        *,
        label: str,
        max_tokens: int,
        temperature: float = 0.3,
    ) -> str:
        """One main-brain LLM call: send ``prompt`` as a single user turn,
        record its cost under ``label`` (plan/replan/compact), return the text.

        Consolidates the ``provider.chat`` + ``_meter_orchestrator`` preamble
        previously open-coded in _propose_milestones / _propose_replan /
        _compact_notes. Returns ``""`` when the provider yields no content,
        matching the previous ``resp.content or ""`` handling.
        """
        resp = await self.provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.cfg.model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._meter_orchestrator(resp, label)
        return (resp.content if resp else None) or ""

    async def _attempt_step(
        self, plan, step, cur_milestone: int, attempt: int, seq: int, prior_result=None
    ):
        """Run ONE attempt of a step and emit one DecisionUnit telemetry record.

        Streams kept independent (anti-circularity): the subagent's self-reported
        ``ok`` is the NOISY verifier verdict; ``_verify_artifacts`` (files exist on
        disk) is the DETERMINISTIC oracle. The verdict is captured BEFORE any
        oracle-driven downgrade so the {verdict x oracle} matrix measures beta.
        """
        ctx = self._build_context_with_replay(plan, step, prior_result)

        t0 = time.monotonic()
        result = await self.subagents.run_sync(
            task=step.description,
            label=f"M{cur_milestone}.S{step.index}#{attempt}",
            skill=self.cfg.skill,
            objective=step.description,
            context_slice=ctx,
        )
        wall_ms = int((time.monotonic() - t0) * 1000)

        verdict = "PASS" if result.ok else "FAIL"  # captured BEFORE the oracle gate
        oracle_label, oracle_detail, result = self._apply_oracle(step, cur_milestone, result)
        # Phase 3: feed the verifier-ceiling β estimate (verdict is the agent's
        # noisy self-judgement pre-oracle; oracle_label is the deterministic one).
        if verdict == "PASS":
            self._evolve_pass += 1
            if oracle_label == "wrong":
                self._evolve_false_success += 1

        self._emit_telemetry(
            step=step, cur_milestone=cur_milestone, attempt=attempt, seq=seq,
            result=result, verdict=verdict, oracle_label=oracle_label,
            oracle_detail=oracle_detail, context_slice=ctx, wall_ms=wall_ms,
        )
        return result

    def _build_context_with_replay(self, plan, step, prior_result) -> str:
        """Context slice for this attempt; re-injects the prior failed attempt
        when retry_context == 'replay' (A1 context-sovereignty violation, on
        purpose so a retry sees why it failed)."""
        ctx = self._build_context_slice(
            step.description,
            plan=plan,
            focus_step_index=step.index,
            last_diagnosis=(prior_result.diagnosis if prior_result else None),
        )
        if self.cfg.retry_context == "replay" and prior_result is not None:
            ctx = (
                f"{ctx}\n\n## Previous attempt (failed)\n"
                f"{(prior_result.summary or '')[:800]}\n"
                f"Diagnosis: {(prior_result.diagnosis or '')[:400]}"
            )
        return ctx

    def _apply_oracle(self, step, cur_milestone, result):
        """TVAE artifact-existence gate. Returns (oracle_label, oracle_detail, result).

        ``result`` is downgraded (false success -> failed) via copy-with-override
        when the oracle contradicts the verifier, preserving iterations/tokens/
        lessons/artifacts. NB: the caller must capture ``verdict`` from
        ``result.ok`` BEFORE calling this — it measures the noisy verifier,
        pre-oracle, for the {verdict x oracle} beta matrix."""
        if not result.artifacts:
            return None, None, result
        ok_art, miss = self._verify_artifacts(result)
        oracle_label = "correct" if ok_art else "wrong"
        oracle_detail = miss or "all claimed artifacts present"
        if result.ok and not ok_art:
            # ORACLE overrides verifier: false success -> downgrade for control flow.
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
            # ORACLE downgrade via copy-with-override: preserves
            # iterations_used / tokens / lessons / artifacts. The old
            # fresh-construction silently dropped them (bug R1).
            result = replace(
                result,
                status="failed",
                diagnosis=(f"False success: claimed artifacts "
                           f"{result.artifacts} not found ({miss}). Re-do the "
                           f"work and actually create the files under "
                           f"{self.cfg.workspace}."),
            )
        return oracle_label, oracle_detail, result

    def _emit_telemetry(
        self, *, step, cur_milestone, attempt, seq, result, verdict,
        oracle_label, oracle_detail, context_slice, wall_ms,
    ) -> None:
        """Emit one DecisionUnit + one meter curve point (additive; never raises)."""
        node_key = make_node_key("step", cur_milestone, step.index, step.description)

        # Meter curve point — independent of EventStore so the context detector
        # still records even if the event store import failed.
        if self.meter is not None:
            try:
                self.meter.record(
                    prompt_tokens=result.last_prompt_tokens,
                    completion_tokens=result.tokens_out,
                    phase="step",
                    node_key=node_key,
                    extra={
                        "attempt": attempt,
                        "verdict": verdict,
                        "oracle_label": oracle_label,
                        "seq": seq,
                        "milestone": cur_milestone,
                        "step_index": step.index,
                    },
                )
            except Exception as exc:  # meter must never break a run
                logger.debug(f"meter record skipped: {exc}")

        # Build the DecisionUnit once; mirror it run-locally (always, so
        # phase-3 metrics can read events.jsonl without the EventStore) and
        # log to the global EventStore when available.
        du = self._build_decision_unit(
            node_key=node_key, step=step, cur_milestone=cur_milestone,
            attempt=attempt, seq=seq, result=result, verdict=verdict,
            oracle_label=oracle_label, oracle_detail=oracle_detail,
            context_slice=context_slice, wall_ms=wall_ms,
        )
        if du is None:
            return
        self._emit_to_sinks(du)

    def _build_decision_unit(
        self, *, node_key, step, cur_milestone, attempt, seq, result, verdict,
        oracle_label, oracle_detail, context_slice, wall_ms,
    ):
        """Construct one DecisionUnit for this attempt, or None on error.

        The 25-field telemetry record (paper-telemetry-spec.md). Extracted from
        _emit_telemetry so the record shape has a single definition."""
        try:
            return DecisionUnit(
                run_id=self.run_id,
                node_key=node_key,
                parent_key=f"milestone:{cur_milestone}",
                level="step",
                seq=seq,
                depth=1,
                attempt=attempt,
                is_retry=attempt > 1,
                instruction=step.description,
                skill_name=self.cfg.skill,
                executor_model=self.cfg.model,
                verifier_type="llm_self_report",
                verifier_verdict=verdict,
                oracle_available=oracle_label is not None,
                oracle_label=oracle_label,
                oracle_type="artifact_exists" if oracle_label is not None else None,
                oracle_detail=oracle_detail,
                triggered_recovery=verdict == "FAIL",
                recovery_mode=(self.cfg.recovery_mode if attempt == 1
                               else f"retry_{self.cfg.retry_context}"),
                diagnosis=((result.diagnosis or "")[:300] or None),
                context_chars=len(context_slice or ""),
                memory_injected_chars=len(context_slice or ""),
                executor_tokens_in=result.tokens_in,
                executor_tokens_out=result.tokens_out,
                context_tokens_in=result.tokens_in,
                active_context_tokens=result.last_prompt_tokens,
                compaction_active=bool(self._compaction_summary),
                wall_ms=wall_ms,
            )
        except Exception as exc:
            logger.debug(f"telemetry du build skipped: {exc}")
            return None

    def _emit_to_sinks(self, du) -> None:
        """Mirror the DU run-locally (events.jsonl) and to the global EventStore.

        The two sinks are independent — either failing must not block the other
        or the run."""
        if self.meter is not None:
            try:
                self.meter.log_decision_unit(du.model_dump())
            except Exception as exc:  # local mirror must never break a run
                logger.debug(f"du local log skipped: {exc}")
        if self.event_store is not None:
            try:
                log_node(self.event_store, du)
            except Exception as exc:  # telemetry must never break a run
                logger.debug(f"telemetry emit skipped: {exc}")

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

    def _build_context_slice(
        self, step_desc: str, *, plan=None, focus_step_index=None, last_diagnosis=None
    ) -> str:
        g = self.global_mem.distill_for_skill(self.cfg.skill, step_desc)
        s = self.skill_mem.get_relevant(step_desc)
        parts = [p for p in (g, s) if p]
        if self._compaction_summary:
            parts.append(f"(progress so far)\n{self._compaction_summary}")
        # Long-horizon anti-disorientation: inject a compact plan-progress
        # snapshot so a retrying subagent sees the global thread + why its
        # previous attempt failed, not just the current step in isolation.
        if plan is not None:
            prog = self.hpm.render_progress(
                plan, focus_step_index=focus_step_index, last_diagnosis=last_diagnosis
            )
            if prog:
                parts.append(prog)
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
        first step following the FAILED step. Advancing to ``len(plan.steps)``
        (not ``len-1``) lets the loop exit when the failed step was the LAST
        step and replan produced no replacement steps — otherwise the cursor
        would point back at the FAILED step itself and re-attempt it."""
        for idx, step in enumerate(plan.steps):
            if step.description == failed_desc and step.status == "FAILED":
                return min(idx + 1, len(plan.steps))
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
