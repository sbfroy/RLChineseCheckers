# Autonomous Training Rules

These rules guide the scheduled Claude Code agent that monitors and manages training.

## Check-in Procedure

Every time the agent runs, it must:

1. Read `training_status.json` to determine current state
2. Read `training_journal.md` to understand what has been tried before and what happened
3. Based on status, follow the appropriate section below
4. Append an entry to `training_journal.md` documenting what was observed and decided

## When status = "running"

Training is still in progress. Do a health check:

1. Read the latest lines of `logs/<run_name>/metrics.jsonl`
2. Look for obvious problems (NaN, loss exploding, etc.)
3. If something is critically wrong (NaN/Inf), kill the process and note it in the journal
4. Otherwise, note progress in journal and exit. Do NOT change anything while training is running.

## When status = "finished"

Training completed. Time to analyze and decide:

1. Read the full `logs/<run_name>/metrics.jsonl` for loss trends
2. Read `logs/<run_name>/eval.jsonl` for win rates (if available)
3. Read `training_journal.md` to see what was already tried in previous runs
4. Think about:
   - Are losses trending down or have they plateaued?
   - How are win rates vs baselines (random, greedy)?
   - What was tried before? Did previous adjustments help?
   - Is it time to move to the next phase, or does this phase need more work?
5. Decide what to do next. Use your judgment — you have access to all the configs and can tune any parameter.

## When status = "failed"

Training crashed:

1. Read the error and traceback from `training_status.json`
2. If you understand the issue and can fix it (OOM, config error) → fix and restart from last checkpoint
3. If it's an unknown bug → log it in journal, do NOT restart (wait for human)

## Goal

The goal is to produce the strongest possible agent for a competition. Optimize for **winning**.

## Competition Constraints

The agent must play within these limits:
- **Per-turn timeout:** 10 seconds
- **Total game time:** 60 seconds
- **~45 moves per game** → ~1.3s average per move budget
- **Must work with 2, 4, and 6 players**
- MCTS with 100 simulations ≈ 0.5-1.5s per move on CPU (much faster on GPU)
- Direct policy (no MCTS) ≈ <10ms per move
- GPU is likely available for competition play

Keep these in mind when tuning. Balance strength vs speed — but GPU makes larger models and more MCTS sims viable.

## Training Phases

A suggested progression (but use your judgment — skip, repeat, or reorder as needed):

1. **Bootstrap** (no MCTS, `configs/fast_bootstrap.yaml`) — fast initial learning
2. **MCTS Light** (50 sims, `configs/mcts_light.yaml`) — learn with tree search
3. **MCTS Full** (100+ sims, `configs/default.yaml`) — strong play
4. **Multi-player** — retrain or fine-tune with `--players 4` and `--players 6`

There is no fixed phase limit. Keep training and improving as long as progress is being made.

## When to Stop

Stop when you've exhausted reasonable approaches and improvements have plateaued:
- Win rates haven't improved across 3+ consecutive training runs despite trying different adjustments
- The model consistently beats all baselines and further gains are marginal
- You've tried tuning key parameters (learning rate, reward weights, MCTS sims, model size) without meaningful improvement

When stopping, write a final summary in the journal with:
- Best checkpoint path
- Best win rates achieved
- Recommended competition settings (MCTS sims, time limit, device)
- Any notes for the human

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

## Config Editing Rules

When changing a config:
- Copy the current config to `configs/archive/<config>_<timestamp>.yaml` before modifying
- Edit the config YAML directly
- Note ALL changes in the journal entry

## Journal Entry Format

```
## [YYYY-MM-DD HH:MM] Check-in

**Status:** running / finished / failed
**Phase:** bootstrap / mcts_light / mcts_full
**Run:** <run_name>

**Observations:**
- Iteration: X / Y
- Policy loss: X.XXXX (trend: decreasing/stable/increasing)
- Value loss: X.XXXX
- Win rates: vs random X%, vs greedy X% (if available)

**Decision:** <what was decided and why>

**Action taken:** <what was done — config change, restart, nothing>

---
```
