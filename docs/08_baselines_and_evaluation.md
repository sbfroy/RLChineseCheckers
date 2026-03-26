# 8. Baseline Agents and Evaluation

## Why baselines matter

Without baselines, you have no idea if your trained agent is actually good. "Policy loss decreased" tells you the network is learning *something*, but not whether it's learning anything *useful*.

Baselines give you concrete performance targets:
- If you can't beat random, something is fundamentally broken
- If you can beat random but not greedy, the network has learned basics but not strategy
- If you beat greedy, the agent is genuinely learning to play

## The three baselines

### 1. Random Agent (`agents/random_agent.py`)

**Strategy:** Pick a uniformly random legal move every turn.

**Strength:** Terrible. Pieces wander aimlessly. Average score ~220 out of possible ~1300.

**Purpose:** The absolute minimum bar. Any trained agent that can't beat this is worse than random and has a bug or hasn't trained enough.

### 2. Greedy Progress Agent (`agents/heuristic_agent.py`)

**Strategy:** For every legal move, calculate how much closer it moves a piece to the goal zone. Pick the move with the largest distance reduction.

**Strength:** Moderate. Consistently makes forward progress. Average score ~1100. But it's shortsighted — it never sets up jump chains and tends to push whichever piece can advance the most *right now*, even if that leaves other pieces stranded.

**Purpose:** Tests whether the trained agent has learned to plan beyond single moves.

### 3. Heuristic Agent (`agents/heuristic_agent.py`)

**Strategy:** Like greedy, but adds two bonuses:
- **Lagging piece priority:** If the furthest-behind piece can move forward, it gets a 2x multiplier on its improvement score. This prevents stragglers.
- **Goal zone bonus:** Entering the goal zone gets a +5 bonus, encouraging pieces to actually complete rather than circle near the goal.

**Strength:** Slightly better than greedy in practice. Avoids the common failure mode of leaving 1-2 pieces stranded at the back while 8 pieces cluster near the goal.

**Purpose:** The strongest simple baseline. Beating this means the RL agent has learned strategies that aren't easily captured by hand-coded rules.

## Running evaluations

### During training (automatic)

The training loop automatically evaluates every `eval_every` iterations:

```
EVAL: {'random': '40%W, score=450', 'greedy': '0%W, score=280'}
```

### Manual evaluation

```python
from models.policy_value_net import PolicyValueNet
from agents.chinese_checkers_agent import ChineseCheckersAgent
from training.evaluate import evaluate_agent
from training.trainer import Trainer

# Load your trained model
model = PolicyValueNet(num_res_blocks=6, trunk_channels=128)
Trainer.load_checkpoint('checkpoints/model_final.pt', model)

# Create agent with MCTS for stronger play
agent = ChineseCheckersAgent(model=model, mcts_simulations=100, temperature=0.1)

# Evaluate against all baselines
evaluate_agent(agent, num_games=20, verbose=True)
```

### Baseline-vs-baseline benchmark

Useful for understanding the performance landscape:

```python
from training.evaluate import run_baseline_benchmark
run_baseline_benchmark(num_games=20)
```

Current results:
```
greedy vs random:   greedy wins ~10%, avg scores greedy=1128 vs random=249
heuristic vs random: heuristic wins ~0% but avg scores heuristic=1118 vs random=239
greedy vs heuristic: roughly equal, both ~1000 avg score
```

Note: "wins" are rare because getting all 10 pins to the goal before 300 moves is difficult. The important metric is the **competition score**, not just win/loss.

## What the scores mean

The competition scoring formula gives up to ~1300 points:
- **1000** for pins in goal (100 per pin)
- **200** for distance progress
- **100** for time (we get max since local games are fast)
- **~1** for move efficiency (negligible)

| Score range | What it means |
|-------------|---------------|
| ~200 | Barely moved. Random-level. |
| ~400-600 | Some progress. Pieces are moving forward but not reaching goal. |
| ~800-1000 | Good progress. Several pins near or in goal. |
| ~1000-1100 | Strong. Most pins very close to or in goal. |
| ~1200+ | Excellent. Nearly all pins in goal. Close to winning. |
| 1300 | Perfect game. All pins in goal, fast play. |

## Evaluation metrics to track over training

| Metric | What to watch for |
|--------|-------------------|
| **Win rate vs random** | Should reach >50% within 50 iterations |
| **Score vs random** | Should exceed 400 within 50 iterations |
| **Score vs greedy** | Should exceed 500 within 200 iterations (hard milestone) |
| **Win rate vs greedy** | Don't expect this quickly — greedy is a solid player |
| **Average game length** | If always 300 (max), agent isn't making enough progress |
| **Games where agent wins** | Any win = agent learned to complete all 10 pins |
