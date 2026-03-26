# 10. The Road Forward — What To Do and Why

## Where we are now

The complete training system is built and tested. The agent is **untrained** — it plays at random level. Everything from here is about training it to be strong.

This is the most important phase. The system is like a brain with the right structure but no knowledge. Training is how it acquires knowledge.

## The plan, step by step

### Step 1: Bootstrap training (first priority)

**What:** Train the network without MCTS, using only its own raw policy for self-play.

**Why:** MCTS is slow. With 100 simulations per move, a single game takes ~2 minutes. Without MCTS, a game takes ~2 seconds. You need volume early on — the network needs to see thousands of games to learn basic concepts like "move toward the goal." MCTS would make this 60x slower for marginal benefit at this stage.

**How:**
```bash
python3.10 train.py --iterations 100 --games 20 --mcts-sims 0 --no-eval
```

**Expected time:** ~30-40 minutes on CPU.

**What you should see:**
- Policy loss drops from ~4.0 to ~3.2-3.5 over 100 iterations
- Value loss drops from ~0.3 to ~0.05
- The network learns basic forward motion

**When to stop:** When policy loss stops decreasing for 20+ iterations. This means the network has learned everything it can from its own (weak) play, and needs a stronger teacher.

**What the agent learns in this phase:**
- "Moving toward the target zone is good"
- "Getting pins out of the home zone is good"
- "Having pieces far from the goal is bad"
- But NOT: "Setting up jump chains", "Coordinating pieces", "Planning ahead"

### Step 2: MCTS-enhanced training

**What:** Switch to MCTS self-play, where each move is chosen by search (50-100 simulations), not just the raw network.

**Why:** The MCTS search looks ahead. Even with a mediocre network, MCTS can find moves that the network alone would miss — like a jump chain that saves 3 moves, or a piece formation that creates future jump opportunities.

When the network trains on MCTS policies instead of its own raw policies, it's learning from a **stronger teacher**. The MCTS is always at least as strong as the network alone, usually much stronger.

**How:**
```bash
python3.10 train.py --iterations 200 --games 20 --mcts-sims 50 \
    --resume checkpoints/model_final.pt
```

**Expected time:** 6-12 hours on CPU (much faster on GPU). Each iteration takes 3-5 minutes because every move requires 50 MCTS simulations.

**What you should see:**
- Policy loss drops further, below 3.0 and toward 2.5
- Evaluation scores against baselines should improve
- The agent starts beating random consistently
- The agent may start competing with greedy

**What the agent learns in this phase:**
- Multi-move planning ("if I go here, I can jump next turn")
- Piece coordination ("keep pieces close enough to form jump bridges")
- Avoiding traps ("don't move there, it'll block my own future moves")

### Step 3: Evaluate and iterate

**What:** After MCTS training, evaluate against all baselines. Identify weaknesses.

**How:**
```python
# Quick evaluation script
python3.10 -c "
import sys; sys.path.insert(0, '.')
from models.policy_value_net import PolicyValueNet
from agents.chinese_checkers_agent import ChineseCheckersAgent
from training.evaluate import evaluate_agent
from training.trainer import Trainer

model = PolicyValueNet(num_res_blocks=6, trunk_channels=128)
Trainer.load_checkpoint('checkpoints/model_final.pt', model)
agent = ChineseCheckersAgent(model=model, mcts_simulations=100, temperature=0.1)
evaluate_agent(agent, num_games=20, verbose=True)
"
```

**What to look at:**
- Score vs greedy: Is it above 500? Above 800?
- Does the agent leave pieces stranded? (Watch per-game output)
- Does the agent win any games (all 10 pins in goal)?
- Average game length: still 300 means not enough progress

### Step 4: Tune reward shaping

**What:** Adjust the reward weights based on what you observed.

**Why:** The default reward weights are educated guesses. Actual training may reveal issues:

| Problem | Adjustment |
|---------|------------|
| Agent doesn't make forward progress | Increase `distance_weight` from 0.01 to 0.03 |
| Agent leaves pieces stranded at the back | Make `lagging_weight` more negative: -0.01 or -0.02 |
| Pieces cluster near goal but don't enter | Increase `pin_goal_weight` from 0.3 to 0.5 |
| Agent never exits home camp in early moves | Increase `home_exit_weight` from 0.05 to 0.1 |

Edit `configs/default.yaml` and re-run training (you can resume from checkpoint).

### Step 5: Deeper MCTS training

**What:** Increase MCTS simulations to 100-200 per move.

**Why:** More simulations = deeper search = stronger training targets. Diminishing returns, but each doubling of simulations typically gives a measurable improvement.

**How:**
```bash
python3.10 train.py --iterations 100 --games 15 --mcts-sims 200 \
    --resume checkpoints/model_final.pt
```

**Expected time:** 12-24+ hours on CPU. This is where GPU really helps.

**Trade-off:** Fewer games per iteration (because each game is slower), but each game produces higher-quality training data.

### Step 6: Multi-player training (if competition uses 4 or 6 players)

**What:** Train with 4 or 6 players instead of just 2.

**Why:** The dynamics change significantly with more players:
- More congestion in the center
- Other players' pieces are both obstacles and jump bridges
- Turn order means you wait longer between moves
- The value function needs to handle "how am I doing relative to 3-5 opponents"

**How:**
```bash
python3.10 train.py --iterations 100 --games 15 --mcts-sims 50 --players 4 \
    --resume checkpoints/model_final.pt
```

**When:** Only after the 2-player agent is strong. Starting with 2 players is faster and simpler.

### Step 7: Final competition preparation

**What:** Pick the best checkpoint, verify it works with the server, report timing.

**How:**
1. Evaluate all saved checkpoints against baselines
2. Pick the one with the highest average score
3. Test it with `server_adapter.py` against the actual game server
4. Record average time per move and total game time
5. Report timing to the professor

## The training timeline

Here's a realistic timeline:

| Phase | Duration (CPU) | Duration (GPU) | What you get |
|-------|---------------|----------------|--------------|
| Bootstrap (100 iter, no MCTS) | 30-40 min | 20-30 min | Basic forward motion |
| MCTS training (200 iter, 50 sims) | 6-12 hours | 2-4 hours | Strategic play, beats random |
| Deep MCTS (100 iter, 200 sims) | 12-24 hours | 4-8 hours | Competitive-level play |
| Multi-player adaptation | 6-12 hours | 2-4 hours | Generalizes to 4/6 players |
| Tuning and evaluation | 2-4 hours | 1-2 hours | Best checkpoint selected |

**Total: 1-3 days on CPU, or 8-18 hours on GPU.**

## Common problems and solutions

### "Policy loss isn't decreasing"

**Cause:** The network is either too small to learn, or the training data quality is too low.

**Fix:**
1. If using no MCTS: switch to MCTS training
2. If model is small (trunk_channels=32): increase to 128
3. If learning rate is too high: reduce from 1e-3 to 3e-4

### "Agent is worse than random"

**Cause:** Usually a bug, or training hasn't run long enough.

**Fix:**
1. Run `python3.10 -m pytest tests/` to verify everything works
2. Train for at least 50 iterations before evaluating
3. Check that action masking is working (illegal moves should have 0 probability)

### "Training is too slow"

**Cause:** MCTS simulations are the bottleneck.

**Fix:**
1. Reduce `mcts_simulations` from 100 to 30-50
2. Reduce `num_games_per_iteration` from 20 to 10
3. Use GPU (`--device cuda`)
4. Use the bootstrap strategy first (MCTS=0)

### "Agent leaves pieces behind"

**Cause:** The lagging-piece penalty isn't strong enough.

**Fix:** In `configs/default.yaml`:
```yaml
rewards:
  lagging_weight: -0.02  # was -0.005
```

### "All games hit 300 moves (max)"

**Cause:** Agent isn't making enough forward progress to win games.

**Fix:**
1. Increase `distance_weight` to encourage forward motion
2. Increase `pin_goal_weight` to encourage completing pieces
3. Make sure training has run long enough (100+ iterations with MCTS)

## GPU: how to make it work

Your system has PyTorch with CUDA support installed. If you have an NVIDIA GPU:

1. Install the NVIDIA CUDA driver for WSL2
2. Run training with `--device cuda`
3. The neural network forward passes (the bottleneck) will run on GPU
4. Game simulation stays on CPU (fine, it's not the bottleneck)

GPU provides 5-10x speedup on MCTS-heavy training because every simulation requires a neural network forward pass.

## What "strong enough to compete" looks like

For the competition, you don't necessarily need to win every game. The scoring formula means consistently getting pins to the goal zone (the 1000-point component) matters most.

**Competitive targets:**
- Average score > 800 against greedy baseline
- No games where pieces are stranded (lagging piece problem solved)
- Inference under 2 seconds per move (comfortable for time limits)
- Works with 2, 4, and 6 players

**Stretch goals:**
- Average score > 1000 against greedy
- Wins some games outright (all 10 pins in goal)
- Adapts strategy based on number of opponents
