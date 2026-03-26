# Training Guide

## Quick start

### Step 1: Bootstrap (no MCTS, fast)

Start here. This trains the network through pure self-play without search, building basic positional understanding. Each iteration takes ~18 seconds on CPU.

```bash
python3.10 train.py --config configs/fast_bootstrap.yaml
```

Or control directly:

```bash
python3.10 train.py --iterations 100 --games 20 --mcts-sims 0 --no-eval
```

**What to watch for:**
- `p_loss` (policy loss) should decrease from ~4.0 toward ~3.0 over 50-100 iterations
- `v_loss` (value loss) should decrease from ~0.3 toward ~0.05
- `buf` (buffer size) grows — more diverse training data is good
- If `p_loss` stops decreasing after 50+ iterations, the network is learning what it can from pure self-play. Time to add MCTS.

### Step 2: MCTS training (slower, stronger)

Once bootstrap is done, switch to MCTS self-play. This produces much stronger training targets because the search explores the game tree.

```bash
python3.10 train.py --config configs/default.yaml --resume checkpoints/bootstrap/model_final.pt
```

Or:

```bash
python3.10 train.py --iterations 200 --games 20 --mcts-sims 100 --resume checkpoints/bootstrap/model_final.pt
```

**What to watch for:**
- `sp` (self-play time) will be much higher — MCTS is compute-heavy
- `p_loss` should drop below bootstrap levels (below ~3.0)
- Evaluation scores against baselines should improve at `eval_every` intervals

### Step 3: Evaluate

Check how the trained agent performs against baselines:

```bash
python3.10 -c "
import sys; sys.path.insert(0, '.')
from models.policy_value_net import PolicyValueNet
from agents.chinese_checkers_agent import ChineseCheckersAgent
from training.evaluate import evaluate_agent
from training.trainer import Trainer

model = PolicyValueNet(num_res_blocks=6, trunk_channels=128)
Trainer.load_checkpoint('checkpoints/model_final.pt', model)
agent = ChineseCheckersAgent(model=model, mcts_simulations=50, temperature=0.1)
evaluate_agent(agent, num_games=20, verbose=True)
"
```

## GPU training

GPU is auto-detected. To force a specific device:

```bash
# Auto-detect (uses GPU if available)
python3.10 train.py --config configs/default.yaml

# Force GPU
python3.10 train.py --config configs/default.yaml --device cuda

# Force CPU
python3.10 train.py --device cpu
```

Or set in the YAML config:

```yaml
training:
  device: cuda   # or "cpu"
```

**Note:** This system has PyTorch 2.8.0+cu128 installed. If you have a CUDA GPU, it will be used automatically. The main speedup is in neural network forward passes during self-play and training. The game simulation itself (move generation, board logic) always runs on CPU.

## Config knobs

### Model architecture (`model:`)

| Knob | Default | Effect |
|------|---------|--------|
| `num_res_blocks` | 6 | More blocks = more capacity but slower. Try 4-10. |
| `trunk_channels` | 128 | Width of the network. 64 is fast for testing, 128-256 for production. |

**Tuning:** Start with `num_res_blocks=4, trunk_channels=64` for fast iteration. Scale up once the pipeline is stable.

### Training loop (`training:`)

| Knob | Default | Effect |
|------|---------|--------|
| `num_iterations` | 200 | Total training iterations. More = stronger. |
| `num_games_per_iteration` | 30 | Games per self-play batch. More = more data per iteration. |
| `num_players` | 2 | Start with 2. Add 4/6 player training later for generalization. |
| `batch_size` | 128 | Training batch size. Increase with GPU memory. |
| `epochs_per_iteration` | 5 | Training passes over buffer per iteration. 3-10 is typical. |
| `learning_rate` | 0.001 | Uses cosine annealing to `1e-5`. Lower if training is unstable. |
| `temperature` | 1.0 | Exploration in self-play. 1.0 is standard. |
| `buffer_capacity` | 200000 | Replay buffer size. Larger = more data diversity. |

**Key tradeoff:** `num_games_per_iteration` vs `mcts_simulations`. More games with less search per game can be better than fewer games with deep search, especially early in training.

### MCTS (`mcts:`)

| Knob | Default | Effect |
|------|---------|--------|
| `num_simulations` | 100 | Search depth per move. 0 = no MCTS. |
| `c_puct` | 1.5 | Exploration constant. Higher = more exploration. 1.0-2.5 typical. |
| `temperature` | 1.0 | Diversity of MCTS policy targets. 1.0 for training, 0.05 for competition. |

**Set to 0 for fast bootstrap.** Then increase to 50-200 for stronger training.

### Reward shaping (`rewards:`)

| Knob | Default | What it does |
|------|---------|--------------|
| `pin_goal_weight` | 0.3 | Bonus per pin entering goal zone |
| `distance_weight` | 0.01 | Reward per unit of distance reduced |
| `lagging_weight` | -0.005 | Penalty for max single-pin distance (anti-straggler) |
| `home_exit_weight` | 0.05 | Bonus per pin leaving home zone |
| `win_reward` | 1.0 | Terminal reward for winning |
| `loss_reward` | -1.0 | Terminal penalty for losing |

**Tuning priority:**
1. `distance_weight` — most important shaping signal. If agent isn't making progress, increase it.
2. `lagging_weight` — prevents stranded pieces. Make more negative if agent leaves pieces behind.
3. `pin_goal_weight` — big bonus for finishing. Should be the strongest per-move signal.

### Competition (`competition:`)

| Knob | Default | Effect |
|------|---------|--------|
| `mcts_simulations` | 200 | More search = stronger play. Limited by time budget. |
| `temperature` | 0.05 | Near-greedy. Tiny randomness prevents predictability. |
| `time_limit` | 2.0 | Seconds per move budget. Adjust when professor sets final limits. |

## What to watch during training

### Healthy training looks like:

```
Iter    1 | buf=   600 | sp=2.1s | p_loss=4.0341 | v_loss=0.1362
Iter   10 | buf=  6000 | sp=2.0s | p_loss=3.5000 | v_loss=0.0500
Iter   50 | buf= 30000 | sp=2.1s | p_loss=2.8000 | v_loss=0.0300
```

- **Policy loss declining** — the network is learning from its own play
- **Value loss declining** — the network is predicting outcomes better
- **Buffer growing** — accumulating diverse experience

### Warning signs:

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `p_loss` flat after 20+ iters | Network capacity too small or no MCTS | Increase `trunk_channels` or enable MCTS |
| `p_loss` oscillating wildly | Learning rate too high | Reduce `learning_rate` to 3e-4 |
| `v_loss` stuck > 0.5 | Value targets too noisy | Increase `num_games_per_iteration` |
| Very slow iterations | MCTS simulations too high | Reduce `mcts_simulations` or use time_limit |
| All games hit max_moves | Agent not making progress | Increase `distance_weight`, check reward shaping |

## Recommended training schedule

1. **Bootstrap** (no MCTS): 100 iterations, ~30 min on CPU
2. **MCTS training** (50 sims): 100 iterations, ~2-4 hours on CPU
3. **Deep MCTS training** (200 sims): 100 iterations, ~8-12 hours on CPU (much faster on GPU)
4. **Multi-player**: Re-train with `--players 4` or `--players 6` for generalization
5. **Final evaluation**: Run against all baselines to pick the best checkpoint

## Resuming training

```bash
python3.10 train.py --resume checkpoints/model_iter_00050.pt --iterations 100
```

This loads the model weights and continues training. The replay buffer starts fresh (older data ages out naturally).

## File structure

```
checkpoints/
  model_iter_00010.pt   # periodic checkpoints
  model_iter_00020.pt
  model_final.pt        # last checkpoint of the run
```

Each checkpoint contains: model weights, optimizer state, scheduler state, iteration number, training history.
