# Skill Optimizer (SkillOpt Integration)

This module brings Microsoft's [SkillOpt](https://github.com/microsoft/SkillOpt) skill optimization loop to Syll. It treats a Syll skill markdown document as a trainable parameter and optimizes it through an iterative 6-stage pipeline:

1. **Rollout** — execute tasks with the current skill via Syll's AgentLoop
2. **Reflect** — an LLM analyst examines trajectories, generates edit patches
3. **Aggregate** — merge patches from multiple minibatches
4. **Select** — rank and select top edits within a budget (textual learning rate)
5. **Update** — apply edits to the skill document
6. **Evaluate** — validate the candidate on a held-out set; accept or reject

## Quick Start

### 1. Create a tasks file

```bash
syll skillopt create-tasks-template my-skill --count 10
```

This creates `~/.syll/workspace/skill_optimizer/my-skill/tasks.json`:

```json
[
  {"id": "1", "task": "Find report.docx on the desktop", "expected": "/Users/you/Desktop/report.docx"},
  {"id": "2", "task": "Find the latest PDF in Downloads", "expected": "/Users/you/Downloads/"},
  {"id": "3", "task": "Search for meeting notes from last week", "expected": ""}
]
```

Edit it with your actual tasks and expected answers.

### 2. Run optimization

```bash
syll skillopt train my-skill --epochs 3 --budget 4 --apply
```

- `--epochs 3`: train for 3 epochs
- `--budget 4`: max 4 edits per step
- `--apply`: apply the best skill back to the workspace after training

### 3. Evaluate without training

```bash
syll skillopt eval-skill my-skill
```

## Architecture

```
syll/skill_optimizer/
├── __init__.py       # Exports SkillOptimizer
├── types.py          # Core types: Edit, Patch, RolloutResult, GateResult
├── gate.py           # Validation gate: accept/reject decision
├── scoring.py        # (in types.py) compute_score, skill_hash
├── skill_ops.py      # Skill edit operations: apply_patch, apply_edit
├── reflect.py        # LLM-based analyst: error/success analysis → patches
├── rollout.py        # Syll AgentLoop rollout + task loading
└── trainer.py        # Main SkillOptimizer class (6-stage loop)
```

## Configuration

Add to `~/.syll/config.json`:

```json
{
  "skill_optimizer": {
    "enabled": true,
    "num_epochs": 3,
    "batch_size": 10,
    "edit_budget": 4,
    "minibatch_size": 5,
    "gate_metric": "hard",
    "failure_only": false,
    "max_concurrent_rollouts": 2,
    "seed": 42
  }
}
```

## CLI Commands

| Command | Description |
|---|---|
| `syll skillopt train <skill>` | Run the full training loop |
| `syll skillopt eval-skill <skill>` | Evaluate a skill on tasks |
| `syll skillopt list-tasks` | List available task files |
| `syll skillopt create-tasks-template <skill>` | Create a tasks.json template |

## How It Works

The optimizer follows SkillOpt's ReflACT protocol:

1. **Rollout**: Each task is sent to Syll's AgentLoop with the current skill loaded. The agent's response is compared to the expected answer (exact match → hard=1, word overlap → soft score).

2. **Reflect**: Failed and successful trajectories are analyzed by an LLM "analyst" which generates structured edit patches (`append`, `insert_after`, `replace`, `delete` operations on the skill markdown).

3. **Aggregate**: Patches from failure and success analysts are merged.

4. **Select**: The top N edits (within the `edit_budget`) are selected.

5. **Update**: Edits are applied to the skill document to produce a candidate.

6. **Evaluate**: The candidate is tested on the eval set. If it scores better than the current skill, the candidate is accepted; otherwise it's rejected.

This cycle repeats for each step within each epoch. The best skill across all steps is saved as `best_skill.md`.

## Output Structure

```
~/.syll/workspace/skill_optimizer/<skill_name>_<timestamp>/
├── best_skill.md           # Best validated skill document
├── summary.json            # Training summary
├── history.json            # Per-step history
├── runtime_state.json      # Resume checkpoint
├── skills/
│   ├── skill_v0000.md      # Initial skill
│   ├── skill_v0001.md      # After step 1
│   └── ...
└── steps/
    ├── step_0001/
    │   ├── rollout_results.json
    │   ├── patches.json
    │   ├── merged_patch.json
    │   ├── ranked_edits.json
    │   ├── candidate_skill.md
    │   └── edit_apply_report.json
    └── ...
```

## References

- **SkillOpt Paper**: [arXiv:2605.23904](https://arxiv.org/abs/2605.23904)
- **SkillOpt GitHub**: [microsoft/SkillOpt](https://github.com/microsoft/SkillOpt)
- **Syll Project**: [THU-SAGE/syll](https://github.com/THU-SAGE/syll)
