# Training Journal

This file is maintained by the autonomous training agent.
Each entry records a check-in: what was observed, what was decided, and what action was taken.

---

## [2026-04-07 22:00] Run Analysis

**Phase:** bootstrap
**Run:** run_20260407_173030
**Config:** configs/fast_bootstrap.yaml
**Device:** cuda
**Wall clock time:** ~38 minutes (17:30 → 18:08)

**Results:**
- Iterations completed: 100 / 100
- Final policy loss: 3.9472 (trend: completely flat — no learning)
- Final value loss: 0.0023 (trend: decreased from 0.154, but meaningless — predicts ~0 for all positions)
- Avg game length: 300.0 moves (100% hit 300-move max across ALL iterations)
- Win rates: vs random 0%, vs greedy 0%, vs heuristic 0%
- Avg scores: vs random 217.8, vs greedy 206.0, vs heuristic 207.0
- Best checkpoint: NONE SAVED (checkpoint_dir not created)

**Reward config used:**
- pin_goal_weight: 0.3, distance_weight: 0.01, lagging_weight: -0.005, home_exit_weight: 0.05

**Strengths observed:** None — agent plays randomly

**Weaknesses observed:** Agent learned nothing. 100% of games hit max moves. Policy loss identical to random play (log(~50 legal moves) ≈ 3.9).

**Analysis:** The bootstrap phase with `mcts_simulations: 0` has a fundamental design flaw: without MCTS, the policy target is the network's own output (`self_play.py` line 90). The network trains to predict itself — a circular fixed point. The per-step reward shaping feeds the value head, but the value loss collapsing to ~0 just means it learned "all positions are equally worthless" (since all games are identical 300-move draws with random play). The policy head receives zero useful gradient because it's supervised against its own predictions.

Previous runs (March 27) appear to have had similar issues — old checkpoints at iter 3 and 5 in `checkpoints/` suggest earlier aborted attempts.

**Decision:** Skip bootstrap phase entirely. Go straight to MCTS Light (Phase 2). MCTS visit counts provide exploration-based policy targets that break the circular training loop. Even with a random starting policy, MCTS will find better-than-random moves through lookahead, giving the policy meaningful training signal.

**Recommendation for next run:**
- Use `configs/mcts_light.yaml` with MCTS 50 sims (no resume — start fresh since bootstrap produced nothing)
- Set device to cuda for speed
- Increase reward weights: `distance_weight: 0.05` (5x), `pin_goal_weight: 0.5` to create stronger shaping signal
- Keep eval_every: 20 for more frequent progress tracking
- Monitor first 10 iterations closely — if games still all hit 300 moves, the issue is deeper

**Command:**
```bash
./run_training.sh configs/mcts_light.yaml --phase mcts_light
```

---

