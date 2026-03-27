# Training Guide

Reference document for analyzing training runs and planning next steps.
Used together with the analysis prompt in `analyze_prompt.md`.

## Goal

Produce the strongest possible agent for a Chinese Checkers competition. Optimize for **winning**.

## Competition Constraints

- **Per-turn timeout:** 10 seconds
- **Total game time:** 60 seconds
- **~45 moves per game** → ~1.3s average per move budget
- **Must work with 2, 4, and 6 players**
- MCTS with 100 simulations ≈ 0.5-1.5s per move on CPU (much faster on GPU)
- Direct policy (no MCTS) ≈ <10ms per move
- GPU is likely available for competition play

Balance strength vs speed — GPU makes larger models and more MCTS sims viable.

## Training Phases

Suggested progression (use judgment — skip, repeat, or reorder as needed):

1. **Bootstrap** (no MCTS, `configs/fast_bootstrap.yaml`) — fast initial learning
2. **MCTS Light** (50 sims, `configs/mcts_light.yaml`) — learn with tree search
3. **MCTS Full** (100+ sims, `configs/default.yaml`) — strong play
4. **Multi-player** — retrain or fine-tune with `--players 4` and `--players 6`

## How to start training

```bash
./run_training.sh <config> --phase <phase> [--resume <checkpoint>]
```

Examples:
```bash
./run_training.sh configs/fast_bootstrap.yaml --phase bootstrap
./run_training.sh configs/mcts_light.yaml --phase mcts_light --resume checkpoints/bootstrap/<run>/model_best.pt
./run_training.sh configs/default.yaml --phase mcts_full --resume checkpoints/<run>/model_best.pt
```

## Available parameters to tune

All configurable in the YAML configs:

**Model:** `num_res_blocks`, `trunk_channels`, `policy_channels`, `value_hidden`
**Training:** `num_iterations`, `num_games_per_iteration`, `batch_size`, `epochs_per_iteration`, `learning_rate`, `weight_decay`, `buffer_capacity`, `temperature`
**MCTS:** `num_simulations`, `c_puct`, `temperature`
**Rewards:** `win_reward`, `loss_reward`, `distance_weight`, `pin_goal_weight`, `lagging_weight`, `home_exit_weight`, `mobility_weight`

## Key files

- `training_status.json` — current/last training state (status, phase, metrics, iteration)
- `logs/<run_name>/metrics.jsonl` — per-iteration metrics (losses, buffer size, timing)
- `logs/<run_name>/eval.jsonl` — evaluation results vs baselines
- `training_journal.md` — history of all runs, decisions, and results
- `checkpoints/<run_name>/` — saved model checkpoints

## Config editing rules

When changing a config:
- Copy the current config to `configs/archive/<config>_<timestamp>.yaml` before modifying
- Edit the config YAML directly
- Note ALL changes in the journal entry

## Journal entry format

```
## [YYYY-MM-DD HH:MM] Run Analysis

**Phase:** bootstrap / mcts_light / mcts_full / multi_player
**Run:** <run_name>
**Config:** <config file used>
**Device:** cpu / cuda
**Wall clock time:** <total training duration>

**Results:**
- Iterations completed: X / Y
- Final policy loss: X.XXXX (trend: decreasing/plateau/increasing)
- Final value loss: X.XXXX (trend: decreasing/plateau/increasing)
- Avg game length: X moves (X% hit 300-move max)
- Win rates: vs random X%, vs greedy X%, vs heuristic X%
- Avg scores: vs random X, vs greedy X, vs heuristic X
- Best checkpoint: <path> (score: X)

**Reward config used:**
- pin_goal_weight: X, distance_weight: X, lagging_weight: X, home_exit_weight: X

**Strengths observed:** <what the agent does well>
**Weaknesses observed:** <what the agent does poorly>

**Analysis:** <what went well, what didn't, key observations, comparison to previous runs>

**Decision:** <continue / advance phase / tune rewards / competition prep>
**Recommendation for next run:** <what to change and why>

**Command:** <exact command to start next run>

---
```
