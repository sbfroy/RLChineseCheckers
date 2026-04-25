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


## [2026-04-18 10:30] Run Analysis

**Phase:** mcts_light
**Run:** run_20260417_091431
**Config:** configs/mcts_light.yaml
**Device:** cuda
**Wall clock time:** ~13 hours (2026-04-17 09:14 → 2026-04-17 22:25)

**Results:**
- Iterations completed: 50 / 50
- Final policy loss: 3.4942 (trend: slow decrease 4.11 → 3.49 — mostly flat after iter 10)
- Final value loss: 0.0056 (trend: 0.155 → 0.005, stable low, not fully collapsed)
- Avg game length: 300.0 moves (100% hit 300-move max across all 50 iterations)
- Win rates: vs random 0%, vs greedy 0%, vs heuristic 0%
- Avg scores: vs random 201-238, vs greedy 195-197, vs heuristic 195-196 (flat across evals)
- Best checkpoint: N/A — `checkpoints/run_20260417_091431/` is empty on disk (likely cleaned between runs)

**Reward config used:**
- pin_goal_weight: 0.1, distance_weight: 0.01, lagging_weight: -0.001, home_exit_weight: 0.02

**Strengths observed:** Value loss didn't fully collapse this time (floors around 0.005), indicating the absolute-score terminal + no-normalization fix did create some target variance.

**Weaknesses observed:** Same fundamental pathology as all three prior runs — 100% max-moves, no pin_goal progress, flat scores.

**Analysis — SMOKING-GUN DIAGNOSTIC:**

Ran a live diagnostic on 2026-04-18 comparing untrained vs trained behavior:
- RandomAgent vs GreedyProgressAgent (baseline): random=226 pts, greedy=1057 pts (greedy finishes most pins when unobstructed)
- Fresh-init (untrained) PolicyValueNet + MCTS-20 vs Greedy: RL=196 pts in all 3 games (identical), Greedy=387 pts
- Trained model (from console.log eval) vs Greedy: 195-197 pts — **same as untrained**

**Training is producing zero net change in policy argmax.** Policy loss drops 4.11 → 3.49 only because the network is being pulled toward the near-uniform MCTS visit distribution (50 sims on ~50 legal moves ≈ 1 visit/action). The argmax remains stuck on whatever initialization produced. The RL net's initial argmax is a pin-clogging pattern so bad it drags even Greedy down from 1057 → 387.

**Secondary eval bug:** `ChineseCheckersAgent.select_action` uses `policy.argmax()` (deterministic), and greedy/heuristic are also deterministic. So "20 eval games" is really 1 game replayed 20x. The identical 195.0 numbers across 20 games are one sample, not a statistic.

**Root-cause synthesis (all three runs):**
1. Games never end in WIN (even greedy-vs-greedy hits max_moves), so there's no terminal WIN signal.
2. Absolute-score terminal has narrow positive dynamic range (0.15-0.25) — weak discrimination for the value head.
3. MCTS past depth 1 evaluates opponent-colour states, but the network only saw RL-colour states in training (only RL data is collected), so leaf values past the root are garbage.
4. Per-step reward shaping is too weak (distance_weight 0.01) — distance improvements barely register.
5. 50 MCTS sims over ~50 legal moves = near-uniform visit counts → weak policy targets.
6. Cycle: uniform policy → random-ish moves → no pin progress → tiny shaped rewards → no value variance → uniform MCTS → uniform policy.

Each previous fix (score margin → heuristic opponent → absolute score + no normalization) addressed a real issue but left this cold-start bootstrap failure unresolved. Pure RL-from-scratch on this game (unreachable WIN condition, long horizon, sparse reward) is too hard from random init.

**Decision:** HALT AlphaZero-style training. Pivot to **supervised imitation pre-training** as Phase 0.

**Recommendation for next run:**

**Phase 0 — Supervised imitation bootstrap (NEW, highest priority):**
1. Write `training/supervised_bootstrap.py`: generate ~2000 games of HeuristicAgent vs HeuristicAgent and HeuristicAgent vs RandomAgent, collecting (state, heuristic_action, final_score_of_that_colour) triples.
2. Policy head: cross-entropy on one-hot heuristic-action target.
3. Value head: MSE on `final_score / 1300`. Target variance is real because heuristic's final score varies with opponent.
4. Train 20-50 epochs, batch 128, lr 1e-3. Should take <30 min on CPU, ~5 min on GPU.
5. **Expected outcome:** agent scores 800-1000+ vs random, ~400-600 vs heuristic, *and actually moves pins forward*.

**Phase 0.5 — Fix eval methodology (cheap, do immediately):**
- In `agents/chinese_checkers_agent.py::select_action`: sample from `policy` with temperature (the existing `mcts.temperature` is already set), or fall back to argmax only when `eval_greedy=True`. Currently `policy.argmax()` makes all deterministic-opponent matches identical.
- Alternative: reduce `num_games` in eval from 20 to 5 for deterministic runs and add one stochastic opponent (e.g., RandomAgent) that provides real variance.

**Phase 1 (only after Phase 0 works) — AlphaZero refinement:**
- Resume from Phase 0 best checkpoint.
- MCTS 100 sims, heuristic opponent.
- Switch back to **score-margin terminal** (now that opponent scores differ from ours, margin has real variance).
- Scale per-step rewards 5-10x: distance_weight 0.05, pin_goal_weight 0.5.
- Monitor: value loss stays >0.02, avg score vs greedy climbs above Phase-0 baseline.

**Why pivot now:** ~100 GPU hours invested in AlphaZero-from-scratch with zero wins. Competition deadline approaching. Supervised imitation is the standard fix when RL-from-scratch fails due to exploration bottlenecks (AlphaGo used expert games; hard-exploration Atari uses human demos). Fast (<1 hour), reliable, and gives us a working agent immediately. We can then layer AZ on top from a strong starting point.

**Command:** *No training command yet — awaiting user approval to create `training/supervised_bootstrap.py` and fix the eval argmax bug. Once approved:*

```bash
python3.10 training/supervised_bootstrap.py --num-games 2000 --epochs 30 --device cuda
```

---

## [2026-04-18 13:00] Run Analysis

**Phase:** supervised_bootstrap (Phase 0)
**Run:** sup_20260418_090744
**Config:** command-line args to `training/supervised_bootstrap.py`
**Device:** cuda
**Wall clock time:** ~43 min (generation 263s + training 2318s + eval)

**Results:**
- Training: 30 epochs, 277,773 experiences from 2000 games
- Final train policy loss: 0.0441 (action accuracy 98.6%)
- Final val policy loss: 0.4830 (**val action accuracy 91.5%**) — plateaued around epoch 10-15, mild overfit afterwards
- Final train/val value loss: 0.0009 / 0.0011 (calibrated, narrow positive range)
- Avg game length: 300.0 moves (still hits max — heuristic itself can't fully win)

**Eval scores (vs each baseline, 5 games):**
- vs random: **agent 973.6** (opp 300.2) — was 201-238 in AZ runs
- vs greedy: **agent 1098.0** (opp 1098.0) — was 195-197; now ties greedy exactly
- vs heuristic: **agent 640.2** (opp 1097.0) — was 195-196; now weaker-but-real play

**Reward config used:** N/A (supervised learning — cross-entropy on teacher action, MSE on final_score/1300)

**Strengths observed:**
- Agent plays roughly at heuristic level — decisive pin-forward motion, no more clogging.
- 5x jump in score vs greedy (195 → 1098), 4x jump vs random (237 → 974).
- Value head learned a real distribution (std 0.051 in data, val loss 0.001).
- Training accuracy 98.6% shows the network has capacity to represent the heuristic policy.

**Weaknesses observed:**
- Weaker than heuristic itself (640 vs 1097). Expected: the imitation is <100% accurate, so a few suboptimal moves per game accumulate.
- No WINs (all games hit 300 moves). Structural — heuristic itself has this problem. Not blocking for the competition since scoring is what matters.
- Val policy loss climbed from 0.33 (epoch 3) → 0.48 (epoch 30). Overfitting. For a rerun, we'd early-stop around epoch 10.
- Value target range narrow [0.65, 0.85]. Fine for Phase 0, but will need widening for Phase 1 (otherwise value head becomes a near-constant predictor again).

**Analysis:**
Phase 0 landed. The bootstrap failure from AZ-from-scratch is solved — the network now has a sensible starting policy. Training loss curves are clean (smooth monotonic decrease, no instability). The gap between train and val accuracy (98.6% → 91.5%) is expected for imitation with 277k examples; it's not a blocker.

The agent is already competitive vs random/greedy baselines. The remaining gap is vs heuristic (640 vs 1097). Two paths forward:
1. **AZ refinement** (Phase 1) — use MCTS + self-play to improve past the teacher. This is the classic AlphaZero-style "learn beyond imitation" step.
2. **More supervised** — add more teachers (mix in greedy demos, use ε-heuristic for state diversity), train with early stopping. Could push val acc toward 95%+ but caps at "as good as the teachers".

**Decision:** Phase 1 — AZ refinement from the Phase 0 checkpoint. Goal: beat heuristic (>1097 vs heuristic's ~1097, i.e. agent becomes the stronger side).

**Recommendation for next run (Phase 1):**

Key config changes from previous AZ attempts:
1. **Resume** from `checkpoints/sup_20260418_090744/model_best.pt` (NOT fresh init).
2. **Switch to score-margin terminal** so the value head has real variance when RL plays at heuristic level vs heuristic (when both score ~1100, absolute-score terminal has no signal — we already proved this). Needs a `use_score_margin: true` flag added to RewardConfig + self_play.py.
3. **Scale per-step rewards back up** now that pins actually reach the goal: distance_weight 0.05, pin_goal_weight 0.5, lagging_weight -0.01, home_exit_weight 0.1.
4. **MCTS sims 100** (full strength, not light) — the network is now competent enough to exploit deeper search.
5. **Short run first** — 20-30 iters to confirm we don't regress from Phase 0 level. Then extend.
6. **Eval tip:** now that `select_action` samples when temperature > 0, the 5 eval games vs greedy/heuristic will actually vary.

**Command:** *Awaiting approval to add `use_score_margin` flag to rewards.py + self_play.py and create `configs/phase1_refine.yaml` resuming from Phase 0. Once ready:*

```bash
./run_training.sh configs/phase1_refine.yaml --phase mcts_full --resume checkpoints/sup_20260418_090744/model_best.pt
```

---

## [2026-04-20 19:50] Run Analysis

**Phase:** mcts_full (Phase 1 refinement)
**Run:** run_20260418_101611
**Config:** configs/phase1_refine.yaml
**Device:** cuda (Tesla V100-SXM3-32GB)
**Wall clock time:** ~15h 45min (2026-04-18 10:16 → 2026-04-19 02:01)

**Results:**
- Iterations completed: 30 / 30 (resumed from Phase 0 iter 30)
- Final policy loss: 2.1602 (trend: U-shape — fell 0.85 → 0.52 (iter 1-3), then **climbed monotonically to 2.16**)
- Final value loss: 0.0371 (trend: collapsed 2.68 → 0.04, value head learned to predict near-constant negative margin)
- Avg game length: 300.0 moves (100% hit max — unchanged from all prior runs)
- Win rates: vs random 0%, vs greedy 0%, vs heuristic 0% (all draws)
- Avg scores (final): vs random 441.2, vs greedy 213.2, vs heuristic 447.4 → avg 367.3
- Avg scores at iter 5 (best): vs random 952.4, vs greedy 1098.0, vs heuristic 540.4 → avg 863.6
- Best checkpoint: model_best.pt at iter 5 (score 863.6) — **then never beaten**

**Reward config used:**
- pin_goal_weight: 0.5, distance_weight: 0.05, lagging_weight: -0.01, home_exit_weight: 0.1
- use_score_margin: true, opponent: heuristic, MCTS sims: 100

**Strengths observed:**
- Iter 5 eval (avg 863.6) confirms the resume from Phase 0 worked — the loaded weights were intact and competitive.
- Self-play infrastructure stable for 15+ hours with no crashes.

**Weaknesses observed:**
- **Catastrophic regression**: avg score collapsed 863 → 300 by iter 10, recovered partially to 367 by iter 30 — net loss of ~60% from Phase 0 baseline.
- vs greedy collapsed worst (1098 → 213, an 80% loss). Agent went from tying greedy to scoring a third of greedy's pins.
- Policy loss climbing monotonically iter 5 → 30 (0.52 → 2.16) is the smoking gun for catastrophic forgetting of the supervised prior.
- vs heuristic eval still produces near-identical games (4 of 5 final-eval games scored 357 exactly) — the temperature-sampling fix from Phase 0 doesn't fully de-determinize MCTS-driven play when value head is saturated.

**Analysis — what went wrong:**

Phase 1 took a working 974/1098/640 agent and turned it into a 441/213/447 agent. The collapse pattern points to **catastrophic forgetting driven by toxic value signal**:

1. **Value head learned "I always lose"** (value loss 2.68 → 0.04). With `use_score_margin: true` and the agent always losing to HeuristicAgent (typical margin: -0.5 to -0.8), the variance in the value target was small enough that a near-constant predictor minimizes MSE. The value head provides no positional discrimination.
2. **MCTS visit counts went uniform.** Without value-head discrimination, 100 sims over ~50 legal moves trends toward uniform visits. Policy targets become low-information.
3. **Policy loss climbed (0.52 → 2.16)** because the network was being pulled away from its peaked Phase-0 distribution toward near-uniform MCTS targets — destroying the imitation prior.
4. **Learning rate too high for fine-tuning.** `lr=3e-4` with cosine schedule is fresh-init magnitude. Standard practice for fine-tuning is 10-30x lower (3e-5 to 1e-5). Combined with no regularization to the prior (no KL anchor), forgetting was inevitable.
5. **Heuristic-opponent mode is a dead-end at this skill level.** The Phase-0 agent loses ~640 vs ~1097 — too far behind for margin to vary meaningfully. We needed self-play (now feasible because Phase 0 broke symmetry already by being non-trivial).

The iteration-5 eval shows what should have happened all along: 863 avg, matching Phase 0. Continuing past that point actively destroyed the agent.

**Note on checkpoints:** Local `checkpoints/` only contains stale March files; the actual Phase 0 and Phase 1 `.pt` files live on the UiA school machine where training runs. Logs are synced to local; checkpoints are not. Phase 0 weights are still available for resume on the school side.

**Comparison to previous runs:** Unlike the three pre-Phase-0 runs that started from random init and stayed at ~200 score, this run started competent and got actively worse. That's a more useful failure mode — it shows the training loop *does* respond to gradients, just in the wrong direction. The fix is to constrain the gradient direction, not to find more signal.

**Decision:** Do NOT relaunch Phase 1 with the current config. The Phase 0 weights are still on the school machine; resume from those, with fundamentally different settings.

Phase 0 is already competitive: 974 vs random and 1098 vs greedy comfortably exceed the >800 vs greedy stretch goal. The 640 vs heuristic gap is the only real weakness, and it may not matter — heuristic is internal, not a competitor agent. The default plan should be **ship Phase 0**, with Phase 1 attempts treated as bonus upside.

**Recommendation for next run** — short, gated Phase 1 v2:
- Resume from Phase 0 best checkpoint (the same one used by run_20260418_101611).
- Edit `configs/phase1_refine.yaml` (after archiving the current copy):
  - `learning_rate: 0.00003` (10x lower — fine-tuning, not fresh init)
  - `opponent: self` (drop heuristic — agent always loses to it, value head can't discriminate)
  - `num_iterations: 10` (validation gate; if iter 5 eval drops below Phase 0 baseline of ~860, kill the run before wasting 15 hours)
- Optional but recommended: add a **KL anchor to a frozen Phase-0 prior** in the trainer's policy loss (`loss += β * KL(π_net || π_prior)` with β ~ 0.5–1.0). Without this, the policy will drift toward MCTS uniform again. Requires a code change to `training/trainer.py`.
- Keep score-margin terminal — it's the right choice for self-play (variance comes from the asymmetry the trainer introduces between current weights and replay-buffer states).

**Even simpler alternative (no code change):** declare Phase 0 the competition agent, spend remaining time on multi-player (4/6) testing, server-integration verification, and timing measurements (<2s/move on competition hardware). This is the pragmatic call given 115 GPU hours invested and a working agent in hand.

**Command** (Phase 1 v2 — gated 10-iter validation):

```bash
# On the school machine:
cp configs/phase1_refine.yaml configs/archive/phase1_refine_20260420.yaml
# Edit configs/phase1_refine.yaml: learning_rate: 0.00003, opponent: self, num_iterations: 10
./run_training.sh configs/phase1_refine.yaml --phase mcts_full --resume checkpoints/sup_20260418_090744/model_best.pt
```

If the iter-5 eval comes in below ~700 avg score, kill it and ship Phase 0 instead.

---

## [2026-04-21 09:00] Run Analysis

**Phase:** mcts_full (Phase 1 v2 — gated refinement)
**Run:** run_20260420_181217
**Config:** configs/phase1_refine.yaml (lr=3e-5, opponent=self, num_iterations=10)
**Device:** cuda (Tesla V100-SXM3-32GB)
**Wall clock time:** ~10h 55min (2026-04-20 18:12 → 2026-04-21 05:07)

**Results:**
- Iterations completed: 10 / 10 (resumed from Phase 0 iter 30)
- Final policy loss: 0.5675 (trend: stable ~0.63-0.73, gentle decrease — **no more blow-up** like v1)
- Final value loss: 2.8743 (trend: 2.75 → 2.87, stable high — **real signal from score-margin in self-play**; v1 had collapsed to 0.04)
- Avg game length: 300.0 moves (100% hit max)
- Win rates (final 10-game eval): vs random **10%** (1W/9D), vs greedy 0%, vs heuristic 0%
- Avg scores (iter 5 eval): vs random 738.4, vs greedy 1098.0, vs heuristic 408.6 → avg **748.3** (gate ≥700 met)
- Avg scores (iter 10 eval): vs random 867.6, vs greedy 1098.0, vs heuristic 510.4 → avg **825.3**
- Avg scores (final 10-game eval): vs random 952 (inc. one 1300!), vs greedy 1098, vs heuristic 483
- Best checkpoint: model_best.pt at iter 10 (score 825.3)

**Reward config used:**
- pin_goal_weight: 0.5, distance_weight: 0.05, lagging_weight: -0.01, home_exit_weight: 0.1
- use_score_margin: true, opponent: self (self-play), MCTS sims: 100, lr: 3e-5 cosine → 1e-5

**Strengths observed:**
- **Catastrophic forgetting fix worked**: policy loss stayed ~0.6 instead of climbing to 2.2 like v1. 10x-lower LR + self-play (instead of unwinnable-vs-heuristic) is the right recipe.
- **First-ever outright WIN** in training-log history: Game 9 vs random, 295 moves, agent 1300 / random 240. Previously every game across 115+ GPU hours ended in max_moves.
- **Monotonic improvement** iter 5 → iter 10 (+77 avg score). Still learning at end of run.
- Value loss ~2.87 sustained — self-play produces real margin variance, unlike heuristic-opponent mode where the agent always lost by a constant.
- Gate passed cleanly (748.3 ≥ 700).

**Weaknesses observed:**
- Still **below Phase 0 baseline** (825 vs 904 avg). vs random down 106, vs heuristic down 130. Ties greedy but no better.
- vs heuristic eval remains deterministic-looking (9 of 10 games scored 468-480) — temperature sampling alone doesn't fully de-determinize MCTS-driven play when value head dominates.

**Analysis:**
This is the first Phase 1 attempt that didn't destroy the model. The v1 diagnosis was correct: the 10x LR reduction + self-play opponent stopped the forgetting. Value loss staying high (2.87) confirms the score-margin signal is alive — in self-play the two copies diverge early, producing real per-game margin variance that wasn't available in the "always lose to heuristic" setup.

Net-net: we spent 11 hours to end up ~80 points below Phase 0. But the **trajectory** is positive (+77 in the last 5 iters), and we've cleared a wall we'd never cleared before (actual WIN). The cosine LR has wound down to 1e-5 — effectively no learning — so extending the current run would do little. A fresh cosine cycle from the v2 best, with 20 more iters, should push past Phase 0.

**Decision:** Continue — extend Phase 1 by 20 more iterations, resuming from v2 best, with a fresh LR schedule.

**Recommendation for next run:**
- Archive current config: `cp configs/phase1_refine.yaml configs/archive/phase1_refine_20260421.yaml`
- Edit `configs/phase1_refine.yaml`: `num_iterations: 20` (fresh cosine from 3e-5 → 1e-5). Everything else unchanged.
- Resume from `checkpoints/run_20260420_181217/model_best.pt` (the v2 best — 825.3 avg), NOT from Phase 0. This keeps the recovery progress.
- **Gate:** if iter 10 eval < 825 (v2 baseline), kill and ship Phase 0. If iter 10 ≥ 900 (match Phase 0), continue; if ≥ 950 (beat Phase 0), this becomes the competition model.
- Keep eval_every: 5 to catch regression early.
- **Fallback is still solid:** Phase 0 checkpoint (avg 904) remains the shipping candidate if this extension stalls.

**Command:**
```bash
# On the school machine:
cp configs/phase1_refine.yaml configs/archive/phase1_refine_20260421.yaml
# Edit configs/phase1_refine.yaml: num_iterations: 20
./run_training.sh configs/phase1_refine.yaml --phase mcts_full --resume checkpoints/run_20260420_181217/model_best.pt
```

---

## [2026-04-22 10:30] Run Analysis

**Phase:** mcts_full (Phase 1 v3 — extension from v2)
**Run:** run_20260421_070417
**Config:** configs/phase1_refine.yaml (num_iterations: 20, lr=3e-5 cosine → 1e-5, opponent: self)
**Device:** cuda
**Wall clock time:** ~21h 47min (2026-04-21 07:04 → 2026-04-22 04:52)

**Results:**
- Iterations completed: 20 / 20 (resumed from v2 best, `run_20260420_181217/model_best.pt`)
- Final policy loss: 0.6028 (trend: stable ~0.60-0.67 across all 20 iters, no blow-up)
- Final value loss: 2.2703 (trend: monotonic decrease 2.87 → 2.27 — real learning signal sustained)
- Avg game length: 300.0 moves (100% hit max, unchanged)
- Win rates (per-iter 5-game eval, iter 20): vs random 0%, vs greedy 0%, vs heuristic 0% (all draws)
- Avg scores per-iter 5-game eval: iter 5 → 847.5, iter 10 → 826.5, iter 15 → 840.4, iter 20 → 806.3 (noisy, essentially flat)
- **Final 10-game eval (iter 20)**: vs random 931, vs greedy 1098, vs heuristic 533 → avg **854.0**
- Best checkpoint: model_best.pt at iter 10 (score 826.5 on 5-game eval); iter 20 model is slightly below on 5-game but better on the 10-game final eval (854 vs 844 for v2)

**Reward config used:**
- pin_goal_weight: 0.5, distance_weight: 0.05, lagging_weight: -0.01, home_exit_weight: 0.1
- use_score_margin: true, opponent: self, MCTS sims: 100, lr: 3e-5 cosine → 1e-5

**Strengths observed:**
- **Stability held**: no catastrophic forgetting (unlike Phase 1 v1). Policy loss stayed ~0.6 throughout, confirming the v2 recipe generalizes.
- Value loss continued to decrease (2.87 → 2.27) — there IS still signal, the network is refining positional value estimates.
- Modest net gain on final 10-game eval: 844 (v2) → 854 (v3), +10 avg score. vs heuristic improved 483 → 533 (+50).
- vs greedy still saturated at 1098 (maxed out).

**Weaknesses observed:**
- **Still below Phase 0** (854 vs 904, delta = 50 pts). Two Phase 1 runs / ~33 GPU hours have narrowed the gap only slightly.
- Per-iter 5-game evals look like a decline (847 → 806). Real signal lost in small-sample noise — 10-game final eval gives the true picture.
- vs heuristic: Phase 1 v3 = 533 vs Phase 0 = 640. The weak spot got *worse* vs the hardest opponent.
- Diminishing returns: v2 → v3 gained only +10 avg over 22 hours.

**Analysis:**

Phase 1 v3 did what it was asked: it didn't crash, didn't regress catastrophically, and nudged the score up. But the gradient of improvement is now flat enough that another 20-iter extension would almost certainly land in the 860-870 range — still short of Phase 0's 904. The value head is still learning (loss dropping) but the policy isn't moving meaningfully (loss flat), which is the signature of a model that has settled into a local optimum that MCTS + self-play can't easily escape at this LR.

Meanwhile, **multi-player (4/6) is a hard competition requirement that has not been touched**: Phase 0 trained on 2-player games only (`supervised_bootstrap.py:86` hard-codes `num_players=2`), and all Phase 1 runs have also been 2-player. The competition requires the agent to play 2, 4, AND 6 players. Time invested in squeezing another 10-20 points out of 2-player is time not spent on an untested requirement.

**Comparison to previous runs:**
| Run | Final avg (10-game) | vs random | vs greedy | vs heuristic | Notes |
|---|---|---|---|---|---|
| Phase 0 | 904 | 974 | 1098 | 640 | supervised baseline, 2P |
| Phase 1 v1 | 367 | 441 | 213 | 447 | collapsed (forgetting) |
| Phase 1 v2 | 844 | 952 | 1098 | 483 | recovery, 10 iters |
| Phase 1 v3 (this) | 854 | 931 | 1098 | 533 | +10 over v2, still below Phase 0 |

Phase 0 is still our best 2-player model. Phase 1 is a valuable sanity check (we can train without collapse) but hasn't produced a superior checkpoint.

**Decision:** Stop 2-player refinement. Pivot to multi-player. Fine-tune Phase 0 with 4-player self-play (short 20-iter gated run) using the same conservative recipe that stabilized Phase 1 v2/v3 (lr=3e-5, self-play, score-margin). Phase 0 checkpoint is preserved — if 4-player fine-tune regresses, we ship Phase 0 for 2P and revisit 4/6P with a fresh strategy (e.g., a multi-player supervised bootstrap mirroring Phase 0).

**Recommendation for next run:**
- New config `configs/multiplayer_4p.yaml` (already created) — copy of phase1_refine with `num_players: 4` (both training and eval), same rewards, same lr schedule, 20 iters.
- Resume from Phase 0 best checkpoint: `checkpoints/sup_20260418_090744/model_best.pt` (NOT from Phase 1 v3 — Phase 0 is stronger on 2P and we want the strongest prior going into 4P).
- **Gate:** if iter 10 4-player eval avg < 700, kill the run and declare Phase 0 the 2P competition agent while we plan a separate 4/6P approach.
- Monitor: value loss should stay above 1.0 (4-player margin variance is larger than 2-player, so expect *more* value signal, not less). Policy loss should not climb above 1.5.

**Command:**
```bash
./run_training.sh configs/multiplayer_4p.yaml --phase mcts_full --resume checkpoints/sup_20260418_090744/model_best.pt
```

---

## [2026-04-22 15:00] Multiplayer validation — Phase 0 fails 4P/6P

**Phase:** validation (no training)
**Script:** validate_multiplayer.py
**Checkpoint under test:** checkpoints/sup_20260418_090744/model_best.pt (Phase 0)
**Device:** cuda | MCTS sims: 20 (the good setting — see note below)

**Results (1 game each vs GreedyProgressAgent baselines filling all other seats):**

| Players | RL pins in goal | RL score | Greedy baselines (pins in goal) | RL latency max |
|---|---|---|---|---|
| 2P | **8/10** | 1098 | blue=8/10 | 448ms (warmup), p95 74ms |
| 4P | **1/10** | 303 | blue=1, lawn_green=8, gray0=9 | 75ms |
| 6P | **0/10** | 195 | 7-8 each across 5 opponents | 87ms |

**Observations:**
- No crashes at any player count — the LocalGame / encoder / MCTS / ChineseCheckersAgent pipeline is polymorphic in num_players.
- Inference latency is excellent: max 87ms in 4P/6P, well under the 2000ms competition budget. That's 20x headroom.
- 2P result exactly matches Phase 0's eval score (1098), which validates both the agent AND the eval methodology for the first time end-to-end.
- **4P/6P are a total failure.** RL gets 0-1 pins in goal while greedy baselines get 7-9. Phase 0 has zero generalization to boards it never saw.
- Interesting 4P structural effect: red(RL)=303 AND blue(greedy)=385 both tank, while the diagonal pair (lawn_green=1097, gray0=1199) races through. RL's confused play clogs the N-S corridor shared with its direct-opposite colour, dragging blue down with it. The NE-SW diagonal stays clear.

**Separate diagnostic finding (from `play.py watch` sessions earlier today):**
- Phase 0 scores **1098 at MCTS 20** but **239 at MCTS 100** (raw visual game, not eval). Value head is near-constant (target range [0.65, 0.85] during training), so high MCTS sim counts over-weight the useless Q-values and degrade play. Eval during training used MCTS 20, which is why it reported correct numbers; Phase 1 self-play used MCTS 100, which explains why Phase 1 never broke past Phase 0. Captured in memory: `mcts_sim_count_pathology.md`.
- `--mcts-sims 0` is NOT "raw network" — it's uniform random play. `best_action()` uses visit counts, with 0 visits it falls through to uniform distribution. See `search/mcts.py:81-116`.

**Decision:** Multi-player supervised bootstrap is the next run. Repeat the Phase 0 recipe across 2/4/6P in a single training pass. Phase 1 AZ refinement is HALTED; `configs/multiplayer_4p.yaml` (AZ 4P fine-tune) is kept on disk as a fallback but NOT the recommended path — AZ with the current value head is not productive.

**Code changes made for this run:**
- `training/supervised_bootstrap.py`: `generate_games` now takes `num_players_list` and cycles player counts across games. Opponents filled by constructing a fresh instance for every non-teacher seat. New CLI flag `--num-players "2,4,6"`.
- `validate_multiplayer.py` (new): one game per player count, RL vs greedy baselines, reports pins/scores/latency and crash status.

**Recommendation for next run:**
- Run supervised bootstrap with 6000 games split evenly across 2/4/6P, 20 epochs (Phase 0 journal noted val loss plateau around epoch 10-15), cuda, eval at 2P only (default).
- After training: rerun `validate_multiplayer.py` on the new checkpoint. Success = RL pin counts ≥7/10 at 4P AND 6P, without 2P dropping below 7. Failure (e.g. 2P regresses because mixed data confuses the model) → train three separate models, one per player count.
- If success, endgame solver becomes the next work item (attacks the 8/10 → 10/10 stall that's still costing ~200 score per 2P game).

**Command:**
```bash
python3 training/supervised_bootstrap.py \
  --num-games 6000 \
  --epochs 20 \
  --device cuda \
  --num-players "2,4,6"
```

---

## [2026-04-22 15:50] Run Analysis

**Phase:** supervised_bootstrap (multi-player: 2/4/6P)
**Run:** sup_20260422_091725
**Config:** CLI args — num-games 6000, epochs 20, batch 256, lr 1e-3, num-players "2,4,6"
**Device:** cuda
**Wall clock time:** ~65 min (generation 814s + training 3075s)

**Results:**
- Iterations completed: 20 / 20 epochs
- 545,621 experiences from 6000 games (cycled across 2/4/6P)
- Final train policy loss: 0.2194 (trend: smooth monotonic decrease 1.00 → 0.22)
- Final val policy loss: 0.6235 (trend: bottomed ~0.549 at epoch 7-8, drifted back up to 0.62 by epoch 20 — mild overfit)
- Final train/val value loss: 0.00158 / 0.00161 (tight, narrow-range target — same pattern as Phase 0)
- Final train/val action accuracy: 92.8% / 85.0%
- Avg game length (2P eval): 300 moves (all draws — unchanged structural issue)

**2P eval scores (5 games, built-in eval uses 2P only):**
- vs random: **agent 1017** (opp 230) — was 974 in Phase 0 → **+43**
- vs greedy: **agent 548** (opp 1097) — was 1098 in Phase 0 → **−550 REGRESSION**
- vs heuristic: **agent 531** (opp 1098) — was 640 in Phase 0 → **−110 regression**

**4P / 6P eval:** NOT MEASURED — `evaluate_agent_quick` at `training/supervised_bootstrap.py:266` only runs 2P matches via `play_match`. Must run `validate_multiplayer.py` separately.

**Reward config used:** N/A (supervised — cross-entropy on teacher action, MSE on teacher's final score / 1300).

**Strengths observed:**
- Training pipeline extension to multi-player worked end-to-end. No crashes at any player count during generation. 6000 games × up-to-6 seats produced ~2x the Phase 0 experience count.
- Loss curves clean, smooth decrease. Val loss minimum at epoch 7-8 means the epoch-20 checkpoint is NOT at the sweet spot — a future run should early-stop around epoch 10.
- Value head target is coherent (narrow positive range, val loss 0.0016 — similar to Phase 0 behaviour).

**Weaknesses observed:**
- **2P regression vs greedy is severe** (1098 → 548). Shared network capacity split three ways; the 2P-specific heuristic pattern got diluted by 4P/6P teacher moves that differ structurally.
- Val accuracy 85% (vs Phase 0's 91.5%). Expected — 4/6P boards have ~50+ legal moves and richer opponent interactions; imitating a scalar heuristic across all three is harder than just 2P.
- Checkpoint saved at epoch 20 (final), not at val-loss minimum (~epoch 8). The script has no early-stopping or best-by-val-loss selection.

**Analysis:**

Two ways to read this run:

1. **Pessimistic:** We traded a working 2P agent (904 avg) for a weaker 2P agent (698 avg) — and we don't yet know if 4P/6P improved at all. If 4P/6P is still broken (1/10 pins, 0/10 pins like Phase 0), this run produced negative value.

2. **Optimistic:** The whole point of this run was 4P/6P coverage. 2P regression was expected (capacity split, teacher distribution mixing). If `validate_multiplayer.py` now shows the agent reaching ≥7/10 pins at 4P and 6P, this is a net win even with the 2P cost — because Phase 0 is useless at 4P/6P.

We **cannot decide between these two readings without the multiplayer validation.** The built-in 2P eval is insufficient. That's the gating measurement.

Per-iter training curves also signal that epoch 8-10 is the better checkpoint than epoch 20. If validation looks promising but marginal, we should redo training with a val-loss-based best-checkpoint selector (one-line change) before declaring this recipe final.

**Comparison to previous runs:**

| Checkpoint | Train data | 2P vs random | 2P vs greedy | 2P vs heur | 4P pins | 6P pins |
|---|---|---|---|---|---|---|
| Phase 0 (`sup_20260418_090744`) | 2000g @ 2P | 974 | 1098 | 640 | 1/10 | 0/10 |
| This run (`sup_20260422_091725`) | 6000g @ 2,4,6P | 1017 | 548 | 531 | **?** | **?** |

**Decision:** Do NOT train more yet. Run `validate_multiplayer.py` against the new checkpoint on the school machine to measure 4P/6P pin counts. Three possible outcomes drive three different next steps:

- **A. 4P ≥ 7 pins AND 6P ≥ 7 pins:** Multi-player bootstrap works. Accept the 2P regression as the price, or retrain with `model_best` selection (val-loss bottom) to recover some of it. This becomes the competition model.
- **B. 4P and/or 6P still < 3 pins:** Mixed-data hypothesis failed. Train three separate models (one per player count) from the same generator and select per game.
- **C. Partial improvement (e.g. 4P works, 6P doesn't):** Decide case-by-case — likely train a 6P-specialist on top of this checkpoint, keep this one for 4P, Phase 0 for 2P.

**Recommendation for next run:** Validate first, then decide. The validation command below tests all three player counts (2/4/6) against GreedyProgressAgent baselines at MCTS 20 (the known-good sim count per `mcts_sim_count_pathology` memory).

**Command:**
```bash
python3 validate_multiplayer.py \
  --checkpoint checkpoints/sup_20260422_091725/model_best.pt \
  --device cuda \
  --mcts-sims 20 \
  --players 2 4 6
```

---

## [2026-04-22 16:30] Multiplayer validation result + next run

**Validation of `sup_20260422_091725` (one game per player count, MCTS 20, greedy baselines):**

| Players | RL pins | RL score | Greedy baseline pins | vs Phase 0 |
|---|---|---|---|---|
| 2P | 3/10 | 568 | blue 8/10 | −5 (regression) |
| 4P | 8/10 | 1097 | blue 7, lawn_green 8, gray0 7 | +7 (fixed) |
| 6P | 5/10 | 778 | 6-8 across 5 opponents | +5 (partial fix) |

Latency: max 823ms (2P warmup), rest <100ms — comfortably under the 2000ms budget.

**Readout (per decision rule from previous entry):** Case C — partial. Mixed-data bootstrap fixed 4P outright, halved the 6P gap, and regressed 2P. Best per-board checkpoints going into competition:

- 2P → Phase 0 (`sup_20260418_090744/model_best.pt`) — 8/10 pins
- 4P → mixed run (`sup_20260422_091725/model_best.pt`) — 8/10 pins
- 6P → mixed run for now; ship a specialist if we can get ≥7/10

**Decision:** Train a **6P specialist**. Same supervised recipe, `--num-players "6"` only. This attacks the single remaining weak link without touching the two already-good checkpoints.

**Code change shipped with this run:** `training/supervised_bootstrap.py` now tracks the best `val_policy_loss` epoch and saves those weights as `model_best.pt`; the epoch-N weights go to `model_final.pt`. Previous runs (both Phase 0 and the mixed run) had val loss bottom near epoch 8 and drift upward afterward — we've been shipping overfit weights. The fix costs ~20 lines and applies to every future supervised run. Eval at end-of-training now runs against the best weights, not the final ones.

**Why 3000 games, 20 epochs:** At 6P the teacher makes ~50 moves per game (vs 150 at 2P), so 3000 games ≈ 150k teacher states — comparable to Phase 0's 277k after accounting for the higher branching factor and harder positions. 20 epochs is a generous ceiling now that best-by-val-loss selection will pick the right stopping point automatically.

**Gate after training:** rerun `validate_multiplayer.py --players 6 --checkpoint <new>/model_best.pt`. Success = ≥7/10 pins at 6P. Failure mode = marginal improvement to 6/10; still ship it as a slight win over 5/10.

**Command:**
```bash
python3 training/supervised_bootstrap.py \
  --num-games 3000 \
  --epochs 20 \
  --device cuda \
  --num-players "6"
```

---

## [2026-04-23 10:15] Run Analysis

**Phase:** supervised_bootstrap (6P specialist)
**Run:** sup_20260422_135523
**Config:** CLI args — num-games 3000, epochs 20, batch 256, lr 1e-3, num-players "6"
**Device:** cuda
**Wall clock time:** ~20 min (generation 374s + training 813s)

**Results:**
- Iterations completed: 20 / 20 epochs
- 143,323 experiences from 3000 games (6P only)
- Final train policy loss: 0.1334 (trend: smooth monotonic decrease 1.78 → 0.13, action accuracy 95.6%)
- Final val policy loss: 2.4218 (trend: **bottomed at 1.347 at epoch 3**, then climbed monotonically — severe overfitting after epoch 3)
- Final train/val value loss: 0.00359 / 0.00349 (narrow-range target, same pattern as all supervised runs)
- Final val action accuracy: 63.2% (peak 63.9% at epoch 6)
- Best checkpoint: `model_best.pt` = epoch 3 weights (val_policy_loss 1.347), per new best-by-val-loss selector shipped with previous run
- Avg game length (2P eval): 300.0 moves (all draws — expected, 6P-only training)

**2P eval scores (5 games, built-in — not the relevant metric for a 6P specialist):**
- vs random: agent 558 (opp 258) — Phase 0: 974, mixed: 1017
- vs greedy: agent 285 (opp 1077) — Phase 0: 1098, mixed: 548
- vs heuristic: agent 390 (opp 1036) — Phase 0: 640, mixed: 531

**4P / 6P eval:** NOT MEASURED. The built-in eval only runs 2P matches; the gating measurement for this run is `validate_multiplayer.py --players 6`, not yet run.

**Reward config used:** N/A (supervised — cross-entropy on teacher action, MSE on teacher's final score / 1300).

**Strengths observed:**
- Pipeline ran cleanly: 3000 6P games generated and trained without incident.
- Best-by-val-loss selector (shipped previous run) worked — `model_best.pt` is the epoch-3 weights (val loss 1.35) rather than the drifted epoch-20 weights (val loss 2.42). Without that change we'd be evaluating a substantially worse model.
- Value head behaviour matches Phase 0 / mixed (narrow, near-constant positive target).

**Weaknesses observed:**
- **Val policy loss floor much higher than any prior run** (1.35 vs Phase 0's 0.33, mixed's 0.55). Likely cause: 6P games have ~50-60 legal moves per state, so the teacher's chosen action is a lower-probability event under cross-entropy (branching factor dominates the loss floor).
- Overfitting starts immediately — val loss bottoms at epoch 3 and climbs every epoch after. Signals that 143k experiences is thin for the 6P distribution, or that 20 epochs is far too many at this data volume.
- 2P regression as expected (2P data not in training set), confirming a single-network 6P-only specialist cannot substitute for Phase 0 on 2P boards.

**Analysis:**

The run did exactly what was asked. The interesting signal is the val loss floor of 1.35 — it tells us the 6P imitation problem is genuinely harder than 2P, not just "needs more of the same". Even at capacity this model is only hitting 64% val action accuracy; compare mixed's 85% and Phase 0's 91.5%. That ceiling is the real question mark going into 6P validation.

**We cannot make a next-step decision until `validate_multiplayer.py --players 6` runs.** The gate from the previous entry is still in force: success = ≥7/10 pins at 6P. The built-in 2P eval is not informative here — it measures a board this model wasn't trained on.

Per-board shipping plan remains:
- 2P → Phase 0 (`sup_20260418_090744`) — 8/10 pins, validated
- 4P → mixed (`sup_20260422_091725`) — 8/10 pins, validated
- 6P → this run if ≥7 pins; mixed (5/10) as fallback

**Comparison to previous runs:**

| Run | Train data | Val policy loss min | Val action acc | 6P pins |
|---|---|---|---|---|
| Phase 0 | 2000g @ 2P | 0.33 @ ep 3 | 91.5% | 0/10 |
| Mixed | 6000g @ 2,4,6P | 0.55 @ ep 8 | 85.0% | 5/10 |
| This run | 3000g @ 6P | 1.35 @ ep 3 | 63.2% | **?** |

**Decision:** Do NOT train more yet. Run `validate_multiplayer.py --players 6` against the new checkpoint on the school machine. Three branches:

- **A. 6P ≥ 7 pins:** Ship per-board (Phase 0 for 2P, mixed for 4P, this for 6P). Declare training done; move to competition prep (timing verification, server integration, endgame solver for the 8/10 → 10/10 stall).
- **B. 6P = 6 pins:** Marginal win over mixed's 5/10 — ship it for 6P. Same next-step as A.
- **C. 6P ≤ 5 pins:** Specialist failed. Options then are (1) more 6P data (say 10k games, keep 20-epoch ceiling since best-by-val-loss will auto-stop around epoch 3-5), or (2) accept mixed's 5/10 for 6P and stop. Default to (2) given diminishing returns.

**Recommendation for next run:** Validate first. The command below tests 6P only against GreedyProgressAgent baselines at MCTS 20 (the known-good sim count per `mcts_sim_count_pathology` memory). 2P and 4P are not retested because this model wasn't trained for them — Phase 0 and mixed respectively remain the per-board picks for those.

**Command:**
```bash
python3 validate_multiplayer.py \
  --checkpoint checkpoints/sup_20260422_135523/model_best.pt \
  --device cuda \
  --mcts-sims 20 \
  --players 6
```

---

## [2026-04-23 10:30] Validation result — 6P specialist clears the gate

**Validation of `sup_20260422_135523/model_best.pt` (one 6P game, MCTS 20, greedy baselines):**

| Players | RL pins | RL score | Greedy baselines | vs prior checkpoints |
|---|---|---|---|---|
| 6P | **7/10** | 995.0 | 7, 8, 8, 8, 9 (5 greedy opp) | Phase 0: 0/10 → mixed: 5/10 → **specialist: 7/10** |

Latency: max 803ms (warmup), avg 80.3ms, p95 75.6ms — comfortably under the 2000ms competition budget.

**Readout:** Case A from the decision rule. 6P gate (≥7 pins) met exactly. The per-board shipping map is now fully validated:

| Board | Checkpoint | Pins | Score |
|---|---|---|---|
| 2P | `sup_20260418_090744/model_best.pt` (Phase 0) | 8/10 | 1098 |
| 4P | `sup_20260422_091725/model_best.pt` (mixed 2/4/6P) | 8/10 | 1097 |
| 6P | `sup_20260422_135523/model_best.pt` (6P specialist) | 7/10 | 995 |

**Decision:** Training is done. Pivot to competition integration. Two things shipping depends on and the current adapter gets wrong:

1. **Single-checkpoint dispatch** — `env/server_adapter.py` currently takes one `--checkpoint`. A competition submission must dispatch by player count.
2. **MCTS-sims default of 100** — per `mcts_sim_count_pathology`, Phase 0 scores 1098 at MCTS 20 but 239 at MCTS 100. Shipping with the default would actively destroy the 2P case.

**Code change shipped with this entry (`env/server_adapter.py`):**
- `CompetitionPlayer.__init__` now takes `checkpoints_by_players: Dict[int, str]` in addition to the single-checkpoint fallback.
- Agent construction is deferred to the first turn, at which point `turn_order` length determines the checkpoint loaded (one-time cost absorbed in the 10s turn budget).
- Default `--mcts-sims` dropped from 100 → **20** (the validated setting across all three checkpoints).
- New CLI flags: `--checkpoint-2p`, `--checkpoint-4p`, `--checkpoint-6p`, `--device`. Old `--checkpoint` still works as fallback.

**Recommendation for next run:** Smoke-test the modified adapter against the local game server. One window: start `game.py`. Other window: launch the adapter with all three checkpoints. Confirms (a) the adapter connects and plays cleanly end-to-end, (b) the right checkpoint gets loaded for the detected player count, and (c) per-move latency on the real RPC path stays under 2s.

If smoke passes, remaining work before competition is: (1) endgame solver to attack the 8/10 → 10/10 stall (~+200 score per 2P/4P game), (2) whatever final submission packaging the class requires.

**Command** (run from school machine; `game.py` from `multi system single machine minimal/` must already be running on 127.0.0.1:50555):

```bash
python3 env/server_adapter.py \
  --name RLAgent \
  --checkpoint-2p checkpoints/sup_20260418_090744/model_best.pt \
  --checkpoint-4p checkpoints/sup_20260422_091725/model_best.pt \
  --checkpoint-6p checkpoints/sup_20260422_135523/model_best.pt \
  --mcts-sims 20 \
  --device cuda
```

---

## [2026-04-23 14:30] Run Analysis

**Phase:** value_fix (Phase 1a — fix value head with frozen policy)
**Run:** run_20260423_103308
**Config:** configs/phase1_value_fix.yaml
**Device:** cuda
**Wall clock time:** ~4 hours (started 2026-04-23 10:33, still "running" per status — but no output)

**Results:**
- Iterations completed: 0 / 30
- No metrics.jsonl, no eval.jsonl, no console.log produced
- training_status.json shows pid 4027943, iteration 0, no metrics
- Run either crashed immediately after writing metadata, stalled, or was never actually started (metadata synced from a partial launch)

**Reward config used:**
- pin_goal_weight: 0.3, distance_weight: 0.01, lagging_weight: -0.005, home_exit_weight: 0.05
- use_score_terminal: true, use_score_margin: false, opponent: none (self-play)
- freeze_policy: true, MCTS sims: 20, lr: 3e-4

**Strengths observed:** None — no data produced.

**Weaknesses observed:** Even if the run had produced data, the design has a fatal flaw:

**DESIGN FLAW: self-play + absolute score terminal + frozen policy = near-constant value targets.**

Both players use the same frozen Phase 0 policy. Both score ~900-1100 in every game. Terminal value = `score / 1300` ≈ 0.7-0.85 for every game — the exact narrow range (std ≈ 0.051) that IS the MCTS sim count pathology. The value head would learn "all positions are worth ~0.75" — a near-constant — which is what it already does. This run cannot fix the problem it was designed to solve.

**Analysis:**

The value_fix concept is sound: freeze the policy (zero forgetting risk), train only the value head until it provides real positional discrimination, then unlock the policy with a working Q-signal. The flaw is in the opponent choice.

To learn value discrimination, the value head needs to see positions with **diverse outcomes** — some leading to 400 points, others to 1100. Self-play with a frozen good policy produces uniformly good outcomes for both sides, offering no contrast.

**Fix applied:** Added `opponent: mixed` support to the training pipeline. With frozen policy playing against a pool of {RandomAgent, GreedyProgressAgent, HeuristicAgent} sampled per iteration:
- vs random: agent scores ~974 → G ≈ 0.75
- vs greedy: agent scores ~1098 → G ≈ 0.84
- vs heuristic: agent scores ~640 → G ≈ 0.49

Value target range expands from [0.70, 0.85] (self-play) → [0.49, 0.84] (mixed). More importantly, positions reached when losing to heuristic look structurally different from positions when beating random — the value head can learn "blocked positions are bad, open paths are good."

**Code changes:**
- `train.py`: `opponent: mixed` creates `[RandomAgent(), GreedyProgressAgent(), HeuristicAgent()]`
- `training/trainer.py`: stores opponent pool, randomly samples one per iteration
- `configs/phase1_value_fix.yaml`: set `opponent: mixed`, updated header comment
- Config archived: `configs/archive/phase1_value_fix_20260423.yaml`
- All 49 tests pass

**Decision:** Kill the current stalled run (if still running). Relaunch value_fix with mixed opponents from Phase 0 checkpoint.

**Gate:** After 30 iterations, test at MCTS 50 and MCTS 100. If MCTS 50 scores **higher** than MCTS 20 (currently 1098), the value head fix worked — higher search depth helps instead of hurts. If MCTS 50 still scores ≤ MCTS 20, the value head is still near-constant; accept Phase 0 as-is and move to competition prep.

**Recommendation for next run:**
- Kill pid 4027943 on school machine if still alive: `kill 4027943`
- Resume from Phase 0 best checkpoint (confirmed working, 1098 at MCTS 20)
- 30 iterations × 60 games × mixed opponents → ~600 games per opponent type
- Monitor: value_loss should stay above 0.01 (not collapse to ~0.003 like all previous runs). If value_loss is sustained >0.05 at iter 10, that's strong evidence of real learning.

**Command:**
```bash
./run_training.sh configs/phase1_value_fix.yaml \
  --phase value_fix \
  --resume checkpoints/sup_20260418_090744/model_best.pt
```

---

## [2026-04-25 11:30] Run Analysis

**Phase:** value_fix (Phase 1a — fix value head, supposed to be with frozen policy)
**Run:** run_20260423_121821
**Config:** configs/phase1_value_fix.yaml
**Device:** cuda
**Wall clock time:** ~5h (12:18:34 → 17:18:01)

**Results:**
- Iterations completed: 30 / 30
- Final policy loss: 2.5183 (trend: **increasing** — 0.96 → 2.51, policy degrading throughout)
- Final value loss: 0.0939 (trend: decreasing 2.15 → ~0.09 then plateau, NO collapse for the first time ever)
- Avg game length: 300.0 moves (100% hit cap — same as every run, expected)
- Best checkpoint: `model_best.pt` saved at iter 10 with avg score 791.1 — never beaten in remaining 20 iters

**Per-eval score trajectory (avg across 3 baselines):**
| Iter | vs random | vs greedy | vs heuristic | avg |
|---|---|---|---|---|
| 5  | 282.7 | 269.6  | 249.5 | 267.3 |
| 10 | 802.2 | 1098.0 | 473.2 | **791.1** ← best |
| 15 | 513.8 | 429.5  | 514.9 | 486.1 |
| 20 | 697.8 | 1098.0 | 440.7 | 745.5 |
| 25 | 538.6 | 1098.0 | 524.2 | 720.3 |
| 30 | 436.0 | 1098.0 | 315.0 | 616.3 |

Phase 0 reference (best 2P checkpoint, MCTS 20): random 974, greedy 1098, heuristic 640, **avg ~904**.

**Reward config used:**
- pin_goal_weight: 0.3, distance_weight: 0.01, lagging_weight: -0.005, home_exit_weight: 0.05
- use_score_terminal: true, use_score_margin: false, opponent: mixed (random+greedy+heuristic)

**Strengths observed:**
- **Value head finally learning.** Loss settled at 0.09-0.14 across iters 5-30, never collapsed to ~0.003 like every previous run. Mixed-opponent terminal-score targets gave the value head genuine variance (G ranging ~0.49-0.84 depending on opponent), and it learned to discriminate.
- vs greedy mirror score stays pinned at 1098 — agent did not lose its ability to play greedy-tier moves.

**Weaknesses observed:**
- **Best checkpoint is strictly worse than Phase 0 across every opponent.** vs random 802 < 974, vs heuristic 473 < 640, vs greedy tied at 1098. So the run produced no useful checkpoint.
- **Policy actively degraded** during training: policy_loss climbed monotonically 0.96 → 2.51, scores dropped after the iter-10 peak. By iter 30, vs heuristic was 315 (half of Phase 0's 640).

**ROOT CAUSE — bug in `train.py:69`:**
```python
freeze_policy=args.freeze_policy if hasattr(args, 'freeze_policy') else tc.get("freeze_policy", False)
```
Argparse always sets `args.freeze_policy` (default False from `action="store_true"`), so `hasattr` is always True and the CLI default silently overrode the YAML's `freeze_policy: true`. `run_metadata.json` confirms the run actually trained with `freeze_policy: false` and `kl_anchor_weight: 0.0` — joint training, no anchor. The previous journal entry's plan ("freeze policy → only value head trains") never executed. What actually ran was vanilla RL with mixed opponents, no protection on the policy.

This explains both the surprise findings:
- Value head fix worked because mixed opponent diversity is what fixed it (independent of freezing).
- Policy degraded because nothing was holding it to Phase 0's distribution while MCTS visit counts pulled it elsewhere on each iteration.

**Comparison to previous runs:**

| Run | freeze_policy | opponent | Value loss floor | Best avg score | Notes |
|---|---|---|---|---|---|
| Phase 0 (sup_20260418_090744) | n/a (supervised) | n/a | 0.0035 (collapsed, low variance target) | ~904 | competition baseline |
| run_20260420_181217 (Phase 1 v1) | false | self-play | ~0.003 | regression | catastrophic forgetting |
| run_20260423_103308 (value_fix v2 first attempt) | (intended true) | mixed | — | — | crashed, no data |
| run_20260423_121821 (this) | **false (bug)** | mixed | **0.094** ✓ | 791.1 < Phase 0 | value learns, policy drifts |

**Decision:** Fix the bug. Re-run value_fix with the policy actually frozen this time. The hypothesis from the previous entry (freeze policy + mixed opponents → value head learns without policy regression) was never tested because of the CLI/YAML override bug. The mixed-opponent diversity demonstrably works (value loss 0.09); we just need to keep the policy still while the value head trains. With freeze_policy active, the only trainable parameters are the value head; policy can't drift, so by construction the new run can only equal or outperform Phase 0.

**Code change shipped with this entry:**
- `train.py:69` — `freeze_policy=args.freeze_policy or tc.get("freeze_policy", False)`. CLI flag and YAML can each enable; neither can disable when the other says yes. Verified the fix locally with a small argparse script: yaml=True/no-CLI → True; yaml=False/CLI=True → True; yaml=False/no-CLI → False.
- Belt-and-suspenders: command below also passes `--freeze-policy` on the CLI so the run is guaranteed to freeze regardless of which override path is taken.

**Gate (unchanged from previous entry):** After 30 iterations, eval the new `model_best.pt` at MCTS 50 and MCTS 100 against greedy. If MCTS 50 score > Phase 0's 1098 at MCTS 20, the value head fix translates to playing strength — ship the new checkpoint with MCTS 50 for 2P. If MCTS 50 ≤ MCTS 20, accept Phase 0 and pivot fully to endgame-solver work for the 9/10 → 10/10 stall.

**Recommendation for next run:** Resume from Phase 0 with `--freeze-policy` explicit. 30 iters × 60 games × mixed opponents on cuda → ~5h, same as before.

**Command:**
```bash
./run_training.sh configs/phase1_value_fix.yaml \
  --phase value_fix \
  --resume checkpoints/sup_20260418_090744/model_best.pt \
  --freeze-policy
```

---
