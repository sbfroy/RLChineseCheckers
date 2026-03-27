Analyze the latest training run and recommend next steps.

## Context you MUST understand

This is an RL agent for a Chinese Checkers **class competition**. The scoring formula is:
- `pin_goal_score`: up to 1000 pts (how many pins reached the goal zone) — **dominant factor**
- `distance_score`: up to 200 pts (how close remaining pins are to goal)
- `time_score`: up to 100 pts (faster = more points)
- `move_score`: ~1 pt per unused move

**Competitive targets:** avg score >800 vs greedy baseline (stretch: >1000). No stranded pieces. Inference <2s/move. Must work with 2, 4, and 6 players.

**Time constraints:** 10s/turn, 60s total game → ~1.3s/move budget.

**Training phases (in order):**
1. Bootstrap (no MCTS, `fast_bootstrap.yaml`) → basic forward motion
2. MCTS Light (50 sims, `mcts_light.yaml`) → strategic play
3. MCTS Full (100+ sims, `default.yaml`) → competitive play
4. Multi-player (4/6 players) → handle congestion
5. Reward tuning → fix specific weaknesses
6. Competition prep → best checkpoint, timing verification

## Data gathering

Do this:

1. Read `training_guide.md` for training phase definitions and tunable parameters
2. Read `training_status.json` to see what just finished (if it doesn't exist, check `logs/` for run directories and read the latest `run_metadata.json`)
3. Read the full run metadata at `logs/<run_name>/run_metadata.json` — this has the exact config, reward weights, MCTS sims, and device used
4. Read the full metrics log at `logs/<run_name>/metrics.jsonl` — look at loss trends, game lengths, max-moves percentage across ALL iterations
5. Read `logs/<run_name>/eval.jsonl` if it exists — check win rates, scores, and game lengths vs each baseline (random, greedy, heuristic)
6. Read `training_journal.md` to see what was tried in previous runs and what was learned

## Analysis

7. **Loss trends:** Are policy/value losses still decreasing? Plateaued for 20+ iterations = time to advance phase. Policy loss targets: ~4.0 (random) → 3.2-3.5 (bootstrap done) → <2.5 (MCTS training) → <2.0 (strong)
8. **Game quality:** What % of games hit 300 moves (max)? High % = agent not making enough progress. Avg game length should decrease as agent improves.
9. **Eval scores:** Score vs greedy is the key metric. Also check: does agent win any games outright? Does it beat random consistently (>90%)?
10. **Critical failure modes:**
    - Games hitting 300 moves → increase `distance_weight` and `pin_goal_weight`
    - Pieces stranded at back → increase `lagging_weight` (more negative)
    - Pieces cluster near goal but don't enter → increase `pin_goal_weight`
    - Agent worse than random → likely a bug, run tests
11. **Compare to previous runs:** What changed? Did it help? Was the change worth the training time?

## Decision and action

12. Decide what to do next: continue same phase, advance to next phase, tune parameters, or declare ready for competition
13. If config changes are needed, archive the old config first: `cp configs/<config>.yaml configs/archive/<config>_<date>.yaml`
14. Write a journal entry to `training_journal.md` using this format:

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
**Weaknesses observed:** <what the agent does poorly — stranded pieces? no forward progress? slow?>

**Analysis:** <what went well, what didn't, key observations, comparison to previous runs>

**Decision:** <continue / advance phase / tune rewards / competition prep>
**Recommendation for next run:** <what to change and why>

**Command:** <exact command to start next run>

---
```

15. Tell me the command to run
