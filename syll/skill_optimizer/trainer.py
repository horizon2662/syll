"""Main skill optimizer — the 6-stage ReflACT training loop for Syll.

Adapted from skillopt.engine.trainer.  Orchestrates the full pipeline:
  1. Rollout   — execute tasks with current skill via AgentLoop
  2. Reflect   — LLM analyzes trajectories, generates patches
  3. Aggregate — merge patches
  4. Select    — rank and select top edits
  5. Update    — apply edits to skill document
  6. Evaluate  — validate candidate, accept/reject
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from syll.skill_optimizer.gate import evaluate_gate, select_gate_score
from syll.skill_optimizer.reflect import run_reflect
from syll.skill_optimizer.rollout import SkillInjector, rollout_batch
from syll.skill_optimizer.skill_ops import apply_patch_with_report
from syll.skill_optimizer.types import (
    GateMetric,
    RolloutResult,
    compute_score,
    skill_hash,
)

if TYPE_CHECKING:
    from syll.agent.loop import AgentLoop
    from syll.providers.base import LLMProvider


class SkillOptimizer:
    """Optimize a Syll skill document through iterative training.

    Parameters
    ----------
    skill_name : str
        Name of the skill to optimize (directory under workspace/skills/).
    tasks : list[dict]
        Training tasks. Each dict needs ``task`` and optionally ``expected``.
    eval_tasks : list[dict]
        Validation tasks for the gate evaluation.
    provider : LLMProvider
        Syll's LLM provider for reflect/optimizer calls.
    model : str
        Model name for optimizer calls.
    agent_loop : AgentLoop
        Syll agent loop for rollout execution.
    workspace : Path
        Path to the Syll workspace.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Tasks per rollout batch.
    edit_budget : int
        Max edits per step (textual learning rate).
    minibatch_size : int
        Tasks per reflect minibatch.
    gate_metric : str
        Gate comparison metric: ``hard``, ``soft``, or ``mixed``.
    failure_only : bool
        If True, skip success analyst.
    out_root : str
        Output directory for training artifacts.
    """

    def __init__(
        self,
        skill_name: str,
        tasks: list[dict],
        eval_tasks: list[dict],
        provider: "LLMProvider",
        model: str,
        agent_loop: "AgentLoop",
        workspace: Path,
        *,
        num_epochs: int = 3,
        batch_size: int = 10,
        edit_budget: int = 4,
        minibatch_size: int = 5,
        gate_metric: GateMetric = "hard",
        failure_only: bool = False,
        out_root: str = "",
        seed: int = 42,
        max_concurrent_rollouts: int = 2,
    ) -> None:
        self.skill_name = skill_name
        self.tasks = tasks
        self.eval_tasks = eval_tasks
        self.provider = provider
        self.model = model
        self.agent_loop = agent_loop
        self.workspace = workspace
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.edit_budget = edit_budget
        self.minibatch_size = minibatch_size
        self.gate_metric = gate_metric
        self.failure_only = failure_only
        self.seed = seed
        self.max_concurrent_rollouts = max_concurrent_rollouts

        if not out_root:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_root = str(workspace / "skill_optimizer" / f"{skill_name}_{ts}")
        self.out_root = os.path.abspath(out_root)

    # ── Skill I/O ──────────────────────────────────────────────────────────

    def _skill_path(self) -> Path:
        """Return the path to the current skill's SKILL.md."""
        # Check workspace skills first, then built-in
        ws_skill = self.workspace / "skills" / self.skill_name / "SKILL.md"
        if ws_skill.exists():
            return ws_skill
        from syll.agent.skills import BUILTIN_SKILLS_DIR
        builtin = BUILTIN_SKILLS_DIR / self.skill_name / "SKILL.md"
        if builtin.exists():
            return builtin
        return ws_skill  # Return workspace path even if missing

    def load_skill(self) -> str:
        """Load the current skill content."""
        path = self._skill_path()
        if path.exists():
            return path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Skill not found: {path}")

    def save_skill(self, content: str, path: str | Path | None = None) -> None:
        """Save skill content to disk."""
        target = Path(path) if path else self._skill_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    # ── Persistence ────────────────────────────────────────────────────────

    def _save_skill_snapshot(self, step: int, content: str) -> None:
        skills_dir = os.path.join(self.out_root, "skills")
        os.makedirs(skills_dir, exist_ok=True)
        with open(os.path.join(skills_dir, f"skill_v{step:04d}.md"), "w", encoding="utf-8") as f:
            f.write(content)

    def _save_history(self, history: list[dict]) -> None:
        with open(os.path.join(self.out_root, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def _save_runtime_state(self, state: dict) -> None:
        with open(os.path.join(self.out_root, "runtime_state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    # ── Aggregate ──────────────────────────────────────────────────────────

    def _aggregate_patches(
        self,
        patches: list[dict],
    ) -> dict:
        """Merge multiple patches into one (simple concat for now).

        In SkillOpt, this is a hierarchical LLM merge.  For Syll we use a
        simpler approach: concatenate all edits from all patches.
        """
        all_edits: list[dict] = []
        reasoning_parts: list[str] = []

        for p in patches:
            inner = p.get("patch", p)
            if not isinstance(inner, dict):
                continue
            edits = inner.get("edits", [])
            for e in edits:
                e.setdefault("source_type", p.get("source_type", "failure"))
            all_edits.extend(edits)
            if inner.get("reasoning"):
                reasoning_parts.append(inner["reasoning"])

        return {
            "edits": all_edits,
            "reasoning": "\n\n".join(reasoning_parts),
        }

    # ── Select ─────────────────────────────────────────────────────────────

    def _select_edits(self, merged: dict, budget: int) -> dict:
        """Select top edits within budget.

        Simple truncation for now.  SkillOpt uses LLM-based ranking;
        Syll can adopt that later.
        """
        edits = merged.get("edits", [])
        selected = edits[:budget] if budget else edits
        return {
            "edits": selected,
            "reasoning": merged.get("reasoning", ""),
        }

    # ── Rollout with skill injection ────────────────────────────────────────

    async def _rollout_with_skill(
        self,
        skill_content: str,
        tasks: list[dict],
    ) -> list[RolloutResult]:
        """Run rollout with a specific skill injected into the agent.

        Writes *skill_content* to the workspace skill path so AgentLoop's
        ContextBuilder picks it up, runs the rollout, then restores the
        original file.
        """
        skill_path = self._skill_path()
        injector = SkillInjector(skill_path)
        try:
            injector.swap(skill_content)
            return await rollout_batch(
                self.agent_loop, tasks, self.max_concurrent_rollouts,
            )
        finally:
            injector.restore()

    # ── Main training loop ─────────────────────────────────────────────────

    async def train(self) -> dict:
        """Execute the full skill optimization loop.

        Returns
        -------
        dict
            Training summary with scores, steps, and token usage.
        """
        os.makedirs(self.out_root, exist_ok=True)

        # Load initial skill
        current_skill = self.load_skill()
        best_skill = current_skill
        initial_skill = current_skill

        print(f"\n{'='*60}")
        print(f"  Syll Skill Optimizer — optimizing '{self.skill_name}'")
        print(f"{'='*60}")
        print(f"  skill:   {self._skill_path()}")
        print(f"  tasks:   {len(self.tasks)} train / {len(self.eval_tasks)} eval")
        print(f"  epochs:  {self.num_epochs}")
        print(f"  batch:   {self.batch_size}")
        print(f"  budget:  {self.edit_budget} edits/step")
        print(f"  gate:    {self.gate_metric}")
        print(f"  output:  {self.out_root}")
        print(f"{'='*60}\n")

        # Split tasks into train/eval
        rng = random.Random(self.seed)
        all_tasks = list(self.tasks)
        rng.shuffle(all_tasks)

        # If no explicit eval tasks, split 80/20
        if self.eval_tasks:
            train_tasks = all_tasks
            eval_tasks = self.eval_tasks
        else:
            split = max(1, int(len(all_tasks) * 0.8))
            train_tasks = all_tasks[:split]
            eval_tasks = all_tasks[split:]

        steps_per_epoch = max(1, math.ceil(len(train_tasks) / self.batch_size))
        total_steps = self.num_epochs * steps_per_epoch

        # ── Baseline evaluation ────────────────────────────────────────
        print(f"  [BASELINE] evaluating initial skill on {len(eval_tasks)} tasks...")
        baseline_results = await self._rollout_with_skill(initial_skill, eval_tasks)
        baseline_hard, baseline_soft = compute_score(baseline_results)
        current_score = select_gate_score(
            baseline_hard, baseline_soft, self.gate_metric,
        )
        best_score = current_score
        print(
            f"  [BASELINE] hard={baseline_hard:.4f} soft={baseline_soft:.4f} "
            f"gate[{self.gate_metric}]={current_score:.4f}"
        )

        # Save baseline
        self._save_skill_snapshot(0, initial_skill)
        history: list[dict] = []
        step_count = 0

        # Selection cache
        sel_cache: dict[str, tuple[float, float]] = {}

        t_start = time.time()

        # ── Training loop ──────────────────────────────────────────────
        for epoch in range(1, self.num_epochs + 1):
            epoch_tasks = list(train_tasks)
            rng.shuffle(epoch_tasks)

            print(f"\n  [EPOCH {epoch}/{self.num_epochs}] {len(epoch_tasks)} tasks, "
                  f"{steps_per_epoch} steps")

            for step_in_epoch in range(steps_per_epoch):
                step_count += 1
                step_t0 = time.time()
                step_dir = os.path.join(self.out_root, "steps", f"step_{step_count:04d}")
                os.makedirs(step_dir, exist_ok=True)

                # Get batch
                start_idx = step_in_epoch * self.batch_size
                batch = epoch_tasks[start_idx:start_idx + self.batch_size]

                print(f"\n  [STEP {step_count}/{total_steps}] "
                      f"epoch={epoch} batch={len(batch)} tasks")

                step_rec: dict = {
                    "step": step_count,
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "batch_size": len(batch),
                }

                # ① ROLLOUT — inject current skill into agent context
                t_phase = time.time()
                print(f"    [1/6 ROLLOUT] executing {len(batch)} tasks with current skill...")
                rollout_results = await self._rollout_with_skill(current_skill, batch)
                r_hard, r_soft = compute_score(rollout_results)
                step_rec["rollout_hard"] = round(r_hard, 4)
                step_rec["rollout_soft"] = round(r_soft, 4)
                step_rec["timing_rollout_s"] = round(time.time() - t_phase, 1)
                print(f"    [1/6 done] hard={r_hard:.4f} soft={r_soft:.4f}")

                # Save rollout results
                with open(os.path.join(step_dir, "rollout_results.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        [r.to_dict() if isinstance(r, RolloutResult) else r for r in rollout_results],
                        f, ensure_ascii=False, indent=2,
                    )

                # ② REFLECT
                t_phase = time.time()
                print(f"    [2/6 REFLECT] analyzing trajectories...")
                patches = await run_reflect(
                    self.provider,
                    self.model,
                    current_skill,
                    rollout_results,
                    edit_budget=self.edit_budget,
                    failure_only=self.failure_only,
                    minibatch_size=self.minibatch_size,
                )
                step_rec["n_patches"] = len(patches)
                step_rec["timing_reflect_s"] = round(time.time() - t_phase, 1)
                print(f"    [2/6 done] {len(patches)} patches generated")

                if not patches:
                    step_rec["action"] = "skip_no_patches"
                    history.append(step_rec)
                    self._save_history(history)
                    self._save_skill_snapshot(step_count, current_skill)
                    print("    [skip] no patches — skill unchanged")
                    continue

                # Save patches
                with open(os.path.join(step_dir, "patches.json"), "w", encoding="utf-8") as f:
                    json.dump(patches, f, ensure_ascii=False, indent=2)

                # ③ AGGREGATE
                t_phase = time.time()
                merged = self._aggregate_patches(patches)
                n_edits = len(merged.get("edits", []))
                step_rec["n_edits_merged"] = n_edits
                step_rec["timing_aggregate_s"] = round(time.time() - t_phase, 1)
                print(f"    [3/6 AGGREGATE] {n_edits} edits merged")

                with open(os.path.join(step_dir, "merged_patch.json"), "w", encoding="utf-8") as f:
                    json.dump(merged, f, ensure_ascii=False, indent=2)

                # ④ SELECT
                t_phase = time.time()
                ranked = self._select_edits(merged, self.edit_budget)
                n_selected = len(ranked.get("edits", []))
                step_rec["n_edits_selected"] = n_selected
                step_rec["timing_select_s"] = round(time.time() - t_phase, 1)
                print(f"    [4/6 SELECT] {n_edits} → {n_selected} edits (budget={self.edit_budget})")

                with open(os.path.join(step_dir, "ranked_edits.json"), "w", encoding="utf-8") as f:
                    json.dump(ranked, f, ensure_ascii=False, indent=2)

                # ⑤ UPDATE
                t_phase = time.time()
                candidate_skill, apply_report = apply_patch_with_report(current_skill, ranked)
                step_rec["timing_update_s"] = round(time.time() - t_phase, 1)
                print(
                    f"    [5/6 UPDATE] skill length: "
                    f"{len(current_skill)} → {len(candidate_skill)}"
                )

                # Save candidate
                with open(os.path.join(step_dir, "candidate_skill.md"), "w", encoding="utf-8") as f:
                    f.write(candidate_skill)
                if apply_report:
                    with open(os.path.join(step_dir, "edit_apply_report.json"), "w", encoding="utf-8") as f:
                        json.dump(apply_report, f, indent=2, ensure_ascii=False)

                # ⑥ EVALUATE — inject candidate skill into agent context
                t_phase = time.time()
                cand_hash = skill_hash(candidate_skill)
                if cand_hash in sel_cache:
                    cand_hard, cand_soft = sel_cache[cand_hash]
                    print(f"    [6/6 EVALUATE] cache hit: hard={cand_hard:.4f}")
                else:
                    print(f"    [6/6 EVALUATE] evaluating candidate on {len(eval_tasks)} tasks...")
                    eval_results = await self._rollout_with_skill(candidate_skill, eval_tasks)
                    cand_hard, cand_soft = compute_score(eval_results)
                    sel_cache[cand_hash] = (cand_hard, cand_soft)

                step_rec["selection_hard"] = cand_hard
                step_rec["selection_soft"] = cand_soft
                step_rec["timing_evaluate_s"] = round(time.time() - t_phase, 1)

                # Gate decision
                gate = evaluate_gate(
                    candidate_skill=candidate_skill,
                    cand_hard=cand_hard,
                    current_skill=current_skill,
                    current_score=current_score,
                    best_skill=best_skill,
                    best_score=best_score,
                    best_step=history[-1]["step"] if history else 0,
                    global_step=step_count,
                    cand_soft=cand_soft,
                    metric=self.gate_metric,
                )

                cand_gate_score = select_gate_score(
                    cand_hard, cand_soft, self.gate_metric,
                )

                current_skill = gate.current_skill
                current_score = gate.current_score
                best_skill = gate.best_skill
                best_score = gate.best_score
                best_step = gate.best_step

                step_rec["action"] = gate.action
                step_rec["candidate_gate_score"] = round(cand_gate_score, 4)
                step_rec["current_score"] = round(current_score, 4)
                step_rec["best_score"] = round(best_score, 4)
                step_rec["best_step"] = best_step

                if gate.action == "accept_new_best":
                    print(
                        f"    [6/6 EVALUATE] ✓ ACCEPT (new best) "
                        f"hard={cand_hard:.4f} > prev best"
                    )
                elif gate.action == "accept":
                    print(
                        f"    [6/6 EVALUATE] ✓ ACCEPT "
                        f"hard={cand_hard:.4f} > current"
                    )
                else:
                    print(
                        f"    [6/6 EVALUATE] ✗ REJECT "
                        f"hard={cand_hard:.4f} ≤ current={current_score:.4f}"
                    )

                # Save state
                step_rec["wall_time_s"] = round(time.time() - step_t0, 1)
                history.append(step_rec)
                self._save_history(history)
                self._save_skill_snapshot(step_count, current_skill)

                self._save_runtime_state({
                    "last_step": step_count,
                    "current_score": current_score,
                    "best_score": best_score,
                    "best_step": best_step,
                })

                timing = step_rec
                print(
                    f"  [STEP {step_count} done] action={gate.action} "
                    f"current={current_score:.4f} best={best_score:.4f} "
                    f"dt={timing.get('wall_time_s', 0)}s"
                )

        # ── Final save ─────────────────────────────────────────────────
        best_path = os.path.join(self.out_root, "best_skill.md")
        with open(best_path, "w", encoding="utf-8") as f:
            f.write(best_skill)

        total_wall = time.time() - t_start
        n_accept = sum(1 for h in history if "accept" in h.get("action", ""))
        n_reject = sum(1 for h in history if h.get("action") == "reject")
        n_skip = sum(1 for h in history if "skip" in h.get("action", ""))

        summary = {
            "skill_name": self.skill_name,
            "baseline_hard": baseline_hard,
            "baseline_soft": baseline_soft,
            "best_score": best_score,
            "best_step": best_step,
            "total_steps": len(history),
            "total_accepts": n_accept,
            "total_rejects": n_reject,
            "total_skips": n_skip,
            "total_wall_time_s": round(total_wall, 1),
            "improvement": best_score - select_gate_score(
                baseline_hard, baseline_soft, self.gate_metric,
            ),
        }
        with open(os.path.join(self.out_root, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"  Training Complete")
        print(f"{'='*60}")
        print(f"  steps:  {len(history)} (accept={n_accept} reject={n_reject} skip={n_skip})")
        print(f"  score:  baseline={select_gate_score(baseline_hard, baseline_soft, self.gate_metric):.4f} "
              f"→ best={best_score:.4f} (Δ={summary['improvement']:+.4f})")
        print(f"  time:   {total_wall:.0f}s")
        print(f"  best:   {best_path}")
        print(f"  output: {self.out_root}")
        print(f"{'='*60}\n")

        return summary
