# 7. The Training Pipeline

## The AlphaZero training loop

Our training follows the AlphaZero pattern. Each **iteration** has three stages:

```
┌─────────────────────────────────────────────────────────┐
│                    ONE ITERATION                         │
│                                                          │
│  1. SELF-PLAY          2. STORE           3. TRAIN       │
│  ┌──────────┐     ┌──────────────┐    ┌──────────────┐  │
│  │ Agent vs  │────>│   Replay     │───>│  Update      │  │
│  │ Agent     │     │   Buffer     │    │  Network     │  │
│  │ (N games) │     │  (examples)  │    │  (gradient   │  │
│  └──────────┘     └──────────────┘    │   descent)   │  │
│                                        └──────────────┘  │
│                                                          │
│  Repeat for many iterations...                           │
└─────────────────────────────────────────────────────────┘
```

### Stage 1: Self-play (`training/self_play.py`)

The current network plays games against itself. For each game:

1. Create a `LocalGame`
2. For each turn:
   a. Encode the board state (spatial + scalars)
   b. Get the legal move mask
   c. Get a policy — either raw network output or MCTS search result
   d. Sample an action based on the policy (with temperature for exploration)
   e. Record the state, policy, and legal mask
   f. Compute the shaped reward for this step
   g. Execute the move
3. When the game ends, compute the final outcome (win/loss/draw)
4. Walk backward through all recorded states, computing **value targets** — discounted future rewards

Each game produces ~150 training examples (one per move) for each player. With 2 players and 20 games per iteration, that's ~6,000 new examples per iteration.

### Stage 2: Replay buffer (`training/replay_buffer.py`)

Training examples are stored in a **replay buffer** — a fixed-size queue (default 200,000 examples). When it's full, old examples are pushed out as new ones arrive.

Why a buffer?
- **Stability:** Training on only the most recent games would cause the network to oscillate wildly. The buffer mixes old and new data.
- **Efficiency:** Each experience is trained on multiple times before being discarded.
- **Diversity:** Old games provide variety, preventing overfitting to the current policy's style.

Each example contains:
```python
Experience(
    spatial,        # (10, 17, 17) board tensor
    scalars,        # (10,) scalar features
    legal_mask,     # (1210,) which actions were legal
    policy_target,  # (1210,) what MCTS/network said to do
    value_target,   # float in [-1, 1] — actual game outcome
)
```

### Stage 3: Training (`training/trainer.py`)

Sample random batches from the buffer and update the network:

1. Sample a batch of 128 examples
2. Run the network forward: `policy_logits, value_pred = model(spatial, scalars, mask)`
3. Compute two losses:
   - **Policy loss:** Cross-entropy between network's policy and the MCTS target
   - **Value loss:** Mean squared error between predicted and actual value
4. Total loss = policy_loss + value_loss
5. Backpropagate gradients through the network
6. Update weights with Adam optimizer
7. Repeat for several epochs (default 5 per iteration)

## The two losses explained

### Policy loss (cross-entropy)

The network outputs logits (raw scores) for each action. We want these to match the MCTS visit-count distribution. Cross-entropy measures how different two probability distributions are:

```
policy_loss = -sum(target_prob × log(predicted_prob))
```

If the network learns to match the MCTS policy exactly, this loss reaches zero. In practice, the MCTS knows better than the raw network (it looked ahead), so the network is always learning from a stronger teacher.

**What to watch:** Policy loss starts around 4.0 (essentially random) and should decrease. Below 3.0 means the network is learning the MCTS policy well. Below 2.0 would be excellent.

### Value loss (MSE)

The network predicts a position value in [-1, +1]. The target is the actual discounted game outcome. Mean squared error measures how far off the predictions are:

```
value_loss = mean((predicted_value - actual_value)^2)
```

**What to watch:** Starts around 0.3-0.5 (random predictions) and should decrease to 0.05-0.1 (reasonable predictions). If it stays high, the network isn't learning to evaluate positions.

## The reward system (`env/rewards.py`)

### Why not just win/loss?

In Chinese Checkers, games can last 100+ moves before anyone wins. If the only training signal is "you won" or "you lost" at the very end, the network gets almost no useful feedback during the game. This is called the **sparse reward problem**.

Reward shaping adds intermediate signals that guide learning:

### Step-by-step rewards

After each move, the agent receives a small reward based on:

| Signal | Weight | What it rewards |
|--------|--------|-----------------|
| **Pin entering goal zone** | +0.3 per pin | Getting pieces home — the main objective |
| **Distance reduction** | +0.01 per unit | Forward progress toward the goal |
| **Lagging piece penalty** | -0.005 × max_dist | Discourages leaving pieces behind |
| **Home camp exit** | +0.05 per pin | Getting pieces moving in early game |

### Terminal rewards

At game end:
| Outcome | Reward |
|---------|--------|
| **Win** | +1.0 |
| **Loss** | -1.0 |
| **Draw/timeout** | -0.3 to -0.5 |

### Computing value targets

The value target for each training example is the **discounted sum of future rewards** from that point to the end of the game:

```
G_t = r_t + γ × r_{t+1} + γ² × r_{t+2} + ... + γ^n × R_terminal
```

Where γ = 0.99 (discount factor). This means rewards in the near future matter more than distant rewards. The terminal reward (win/loss) propagates backward through the whole game, but earlier moves feel it less than later moves.

## Self-play details

### Temperature schedule

During self-play, move selection uses **temperature** to control exploration:

- **First 20 moves:** temperature = 1.0 (proportional sampling — lots of exploration)
- **After move 20:** temperature = 0.1 (near-greedy — mostly exploit what seems best)

This ensures early-game variety (the agent explores different openings) while converging on strong play in the mid/late game.

### With and without MCTS

**Without MCTS (bootstrap mode):**
- Policy comes directly from the network's output
- Very fast (~2 seconds per game)
- Quality limited by the network's current ability

**With MCTS:**
- Policy comes from MCTS search (much stronger)
- Much slower (~60-180 seconds per game depending on simulation count)
- Produces superior training targets

The recommended strategy: bootstrap without MCTS first to build basic understanding, then switch to MCTS training for refinement.

## Evaluation (`training/evaluate.py`)

### Baseline agents

Three baselines establish a performance ladder:

1. **Random agent:** Picks any legal move at random. Score ~220.
2. **Greedy agent:** Always picks the move that reduces total distance to goal the most. Score ~1100.
3. **Heuristic agent:** Like greedy but prioritizes lagging pieces and goal-zone entry. Score ~1100.

### How evaluation works

The trained agent plays N games against each baseline. We track:
- Win rate
- Average competition score
- Average game length

**Target milestones:**
- Beat random consistently → network learned basic forward motion
- Beat greedy → network learned lookahead strategies
- Beat heuristic → network learned sophisticated play

### When evaluation happens

By default, every 20 training iterations the agent is evaluated against random and greedy. This lets you track progress over time.

## Checkpointing

Every N iterations (default 10), the model weights, optimizer state, and training history are saved to disk:

```
checkpoints/
  model_iter_00010.pt
  model_iter_00020.pt
  ...
  model_final.pt
```

You can resume training from any checkpoint:
```bash
python3.10 train.py --resume checkpoints/model_iter_00050.pt
```

**Important:** Always keep the best-performing checkpoint, not just the latest one. Later training iterations aren't guaranteed to be stronger (the network can temporarily regress during exploration).
