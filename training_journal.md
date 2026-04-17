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

## [2026-04-13 13:00] Run Analysis

**Phase:** mcts_light
**Run:** run_20260408_113340
**Config:** configs/mcts_light.yaml
**Device:** cuda
**Wall clock time:** ~48 hours (2026-04-08 11:33 → 2026-04-10 11:21)

**Results:**
- Iterations completed: 150 / 150
- Final policy loss: 3.6229 (trend: very slowly decreasing — 3.996 → 3.623 over 150 iters, ~0.37 total)
- Final value loss: 0.0036 (trend: collapsed to ~0 — predicting 0 for all positions)
- Avg game length: 300.0 moves (100% hit 300-move max across ALL 150 iterations — identical to bootstrap failure)
- Win rates: vs random 0%, vs greedy 0%, vs heuristic 0% (all 20/20 draws every eval)
- Avg scores: vs random 220.4, vs greedy 305.0, vs heuristic 197.0 (flat across all 7 evals — no improvement)
- Best checkpoint: model_best.pt at iter 20 (score 242.2 — first eval, never beaten)

**Reward config used:**
- pin_goal_weight: 0.5, distance_weight: 0.05, lagging_weight: -0.005, home_exit_weight: 0.05

**Strengths observed:** None — agent has not learned beyond random play.

**Weaknesses observed:** Identical pathology to bootstrap run. 100% max-moves, value head outputs ~0 everywhere, scores flat, no win/loss signal at all.

**Analysis:** Spent 48 GPU hours producing essentially zero learning. Hypothesis from previous journal entry (that MCTS would break the circular fixed-point) was wrong. The actual root cause is much deeper.

**ROOT CAUSE FOUND (diagnostic test):**
Ran `GreedyProgressAgent vs GreedyProgressAgent` and `HeuristicAgent vs HeuristicAgent` for 5 games each at `max_moves=300`:
- Greedy vs greedy: 0/5 wins, all hit max_moves, deterministic final scores 994 / 1097
- Heuristic vs heuristic: 0/5 wins, all hit max_moves, deterministic 1199 / 1097

Even strong rule-based agents **cannot win a single game** under the 300-move cap. Both get most pins to the goal (≈8-11 of ~10 pins) but strand 1-2 lagging pieces forever because no opponent can complete `_check_status` ("all pins in opposite zone"). Consequently:

1. **Self-play has produced ZERO terminal WIN signal across 50+ hours of training.** Every game ends in MAX_MOVES.
2. The terminal value reward path is `max_moves_reward (-0.3) + score_normalized*0.5` — for a typical 800/1300 score that's only ~0.0075. After per-step rewards and `max_abs` normalization in `self_play.py:155-159`, the terminal contribution becomes noise.
3. The value head correctly learned "all positions yield ~0" → value loss collapsed to 0.003.
4. The policy head trains on MCTS visit counts, but with 50 sims and ~50 legal moves per state, visit distributions are nearly uniform → near-zero gradient signal.
5. No bug in `_check_status`, `compute_scores`, or MCTS — the failure is **architectural**: the training loop assumes WIN events that never occur in this game/agent population.

**Comparison to previous run:** Same disease (100% max moves, flat scores, value head collapse). The bootstrap → MCTS-light transition did not help because both runs are starved of the same signal. The 5x distance_weight and pin_goal_weight bump from the previous journal entry was a band-aid that didn't address the missing terminal signal.

**Decision:** Halt long training runs. Fix the value-target signal before launching anything else. Specifically, the training loop in `training/self_play.py:144-159` must stop relying on `game.winner`/draw and instead use **score margin** as the terminal value target (`(my_score - avg_opp_score) / 1300`). This converts an unreachable binary outcome into a dense, gradient-rich, competition-aligned target that aligns directly with the scoring formula. Optionally, also bias self-play opponent slot toward a strong heuristic so the network gets contrasting signal.

**Recommendation for next run:**
Before any new training:
1. Patch `training/self_play.py` to compute value targets from score margin, not WIN/DRAW. Approx: `G = (scores[colour]["final_score"] - mean(opp scores)) / 1300.0` clamped to [-1, 1].
2. Optionally: in `evaluate.py`, the eval should still report wins/scores but recognize that "0% win rate" is meaningless under current game length cap — track score margin instead as the headline metric.
3. Diagnostic 10-iteration run with the patch to confirm value loss now grows (>0.05) and avg score against greedy improves above 305 within 10 iters.
4. Only after the diagnostic shows learning, launch a 100-iter MCTS-light training run.

**Command:** *No training command yet — awaiting user approval to patch `self_play.py`. After patching, the diagnostic command will be:*
```bash
./run_training.sh configs/mcts_light.yaml --phase mcts_light  # with num_iterations temporarily set to 10
```

---

## [2026-04-17 13:00] Run Analysis

**Phase:** mcts_light
**Run:** run_20260413_184930
**Config:** configs/mcts_light.yaml
**Device:** cuda
**Wall clock time:** ~48 hours (2026-04-13 18:49 → 2026-04-15 18:40)

**Results:**
- Iterations completed: 150 / 150
- Final policy loss: 3.6015 (trend: slowly decreasing — 3.977 → 3.602, total -0.375)
- Final value loss: 0.0038 (trend: collapsed from 0.133 to ~0.004 — same pathology as all previous runs)
- Avg game length: 300.0 moves (100% hit 300-move max across ALL 150 iterations)
- Win rates: vs random 0%, vs greedy 0%, vs heuristic 0% (all draws, every eval)
- Avg scores: vs random 201-223, vs greedy 196-205, vs heuristic 196-202 (flat, no improvement)
- Best checkpoint: model_best.pt at iter 20 (score 206.6 — first eval, never beaten)

**Reward config used:**
- pin_goal_weight: 0.5, distance_weight: 0.05, lagging_weight: -0.005, home_exit_weight: 0.05

**Strengths observed:** Policy loss decreased meaningfully (3.98 → 3.60), indicating MCTS does provide some signal. Run completed without crashes.

**Weaknesses observed:** Identical failure pattern to all previous runs: 100% max-moves games, value loss collapsed to ~0, eval scores flat at ~200, 0% win rate against everything.

**Analysis:**

The `use_score_terminal: true` fix from the previous entry was correctly implemented in `self_play.py:150-161` (score-margin terminal value), but it did NOT fix the problem. Root cause diagnosis:

1. **Self-play symmetry kills the signal.** Both players are the same model. Both score ~200. Score margin = (200-200)/1300 ≈ 0. Terminal value is still ~0, identical to before.
2. **max_abs normalization destroys cross-game variance.** Lines 175-177 normalize each game's returns independently, so even if individual games produce slightly different values, the normalization makes them all look the same to the value head.
3. **Vicious cycle confirmed:** Value head outputs ~0 everywhere → MCTS value estimates are useless → 50 sims over ~50 legal moves = ~1 visit per move → near-uniform visit distributions → weak policy targets → slow policy learning → still random play → value head stays useless.

The score-margin fix addressed the right conceptual problem (don't rely on WIN events) but missed the structural one: when both players are identical, the margin is always zero. Previous run (run_20260408_113340) had the exact same issue for the same reason.

**Three coordinated fixes applied:**

1. **Train against HeuristicAgent opponent** — breaks self-play symmetry. Heuristic scores ~1000-1100, RL agent ~200 initially. Different scores → different terminal values → cross-game variance → value head has something to learn.
2. **Absolute score terminal value** — `G = my_score / 1300` instead of `(my_score - opp) / 1300`. Always meaningful, independent of opponent.
3. **Removed max_abs normalization** — preserves cross-game variance in value targets. Reward weights sized so returns naturally fall in [-1, 1] (verified: std=0.24 in smoke test vs near-zero before).

Also reduced per-step reward weights to keep accumulated returns within [-1, 1] without normalization: distance_weight 0.01, pin_goal_weight 0.1, lagging_weight -0.001, home_exit_weight 0.02.

**Smoke test results:**
- VS opponent mode: 150 exp/game, value targets in [-0.72, 0.15], std=0.24
- Self-play mode: still works (backward compatible)
- All 49 existing tests pass

**Decision:** Run 50-iteration diagnostic with the three fixes to validate value head learning.

**Recommendation for next run:**
- Config: `configs/mcts_light.yaml` (updated with opponent: heuristic, reduced rewards, 50 iters)
- Fresh start (no resume — previous checkpoints learned nothing useful)
- Monitor: value_loss should NOT collapse to 0. Expect >0.01 sustained. Avg score vs greedy should improve above 305 (previous best) within 20 iters.
- If diagnostic succeeds: extend to 200 iterations with same config.
- If value loss still collapses: the issue is in the encoder/model architecture, not the training signal.

**Command:**
```bash
./run_training.sh configs/mcts_light.yaml --phase mcts_light
```

---

