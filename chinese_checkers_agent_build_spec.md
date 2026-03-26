# Chinese Checkers RL Agent Build Specification for Claude Code

## Purpose

This document is a **full implementation brief** for building a strong Chinese Checkers agent for competition. The goal is not to produce a quick prototype, but to build a **high-quality, well-structured, well-tested, competition-oriented agent** that is grounded in sound Reinforcement Learning (RL) and game AI principles.

The agent should aim to be **practically dominant**, not merely academically interesting. It should combine:

- self-play Reinforcement Learning,
- a policy-value neural network,
- Monte Carlo Tree Search (MCTS),
- strong legal-action handling,
- reward shaping suited to Chinese Checkers,
- and a careful engineering process.

This specification is written for **Claude Code**. It should be treated as a directive for how to inspect the codebase, reason about the design, and implement the solution carefully.

---

## Very important meta-instruction

Do **not** rush into coding.

Before writing or changing anything substantial, first build a deep understanding of the current project. The existing files, architecture, game engine, assumptions, data structures, and interfaces matter. The implementation must fit the current repository cleanly.

The process should be:

1. **Understand the repository deeply**
2. **Map the current architecture and execution flow**
3. **Identify what already exists and what is missing**
4. **Design the agent so it integrates cleanly**
5. **Implement step by step**
6. **Test thoroughly**
7. **Refine weak points**

Do not take shortcuts. Think carefully before each architectural choice.

---

## Primary mission

Build a Chinese Checkers agent that is strong in practice and suitable for competition.

The preferred solution is a **hybrid search-learning agent**:

- **Policy-value network**
- **Self-play training**
- **Monte Carlo Tree Search for decision-making**
- **Action masking**
- **Reward shaping**
- **Curriculum or staged improvement if useful**

The system should be designed so that it can realistically outperform simpler baselines such as:

- random agents,
- greedy-progress agents,
- plain heuristic agents,
- and simpler RL agents without search.

---

## Core design philosophy

Chinese Checkers is difficult because:

- the action space can be large,
- jump chains make move generation nontrivial,
- rewards are delayed,
- the game can be long,
- and naive forward-progress heuristics are not enough.

A strong agent should therefore not only optimize short-term progress, but also:

- create future jump opportunities,
- avoid leaving pieces behind,
- maintain good structure,
- and make efficient long-term transitions toward the goal camp.

One especially important principle:

> Do not only optimize average progress. Also explicitly care about lagging pieces.

In Chinese Checkers, one or a few stranded pieces can lose the game. A strong agent should avoid this.

---

# Phase 1: Deep understanding of the current repository

## First objective

Before implementing the agent, inspect the repository thoroughly.

Claude Code should first produce an internal understanding of:

- the repository structure,
- the current entry points,
- how the game state is represented,
- how moves are generated,
- how agents currently interface with the environment,
- whether training infrastructure already exists,
- whether there are baseline agents,
- and what the competition constraints are.

## Known repository structure

The repository has been inspected. Here is the current state:

### Project layout
```
RLChineseCheckers/
├── single system/              # Local play (for testing/prototyping)
│   ├── checkers_board.py       # HexBoard class (121 cells, axial coords)
│   ├── checkers_pins.py        # Pin class (move generation, placement)
│   ├── checkers_gui.py         # Tkinter GUI
│   └── checkers_main.py        # Interactive CLI game loop
│
├── multi system single machine minimal/   # COMPETITION PLATFORM
│   ├── game.py                 # Server: game logic, scoring, timing, RPC
│   ├── player.py               # Client: connect, join, play (AGENT GOES HERE)
│   ├── checkers_board.py       # Same HexBoard class
│   ├── checkers_pins.py        # Same Pin class
│   └── checkers_gui.py         # Same GUI
│
├── chinese_checkers_agent_build_spec.md   # This file
├── README.md
├── TODO.md
└── papers.md
```

### Competition platform
- The **multi-system server** (`game.py`) is the competition platform.
- Your agent replaces the `PLAYING LOGIC` section in `player.py` (lines 153-183).
- The server handles all game state, move validation, scoring, and timing.
- Communication is via JSON-RPC over TCP socket on port 50555.

### Game engine details
- **Board**: hexagonal with R=4, 121 cells total (61 center hex + 60 colored triangle zones).
- **Coordinates**: axial `(q, r)` with `s = -q - r`.
- **Players**: 2 to 6, in color pairs (red↔blue, lawn green↔gray0, yellow↔purple).
- **Pieces**: 10 pins per player, starting in their colored triangle.
- **Legal moves**: single-step to adjacent empty cell OR multi-hop chains (DFS in `getPossibleMoves()`).
- **Win**: all 10 pins in opposite color zone. **Draw**: no legal moves.
- **Scoring**: 4-component formula (time, moves, pins-in-goal, distance). See Phase 7.

### Agent interface
- Agent receives state via `get_state` RPC → JSON with pin positions, scores, turn info.
- Agent gets legal moves via `get_legal_moves` RPC → `{pin_id: [to_index1, to_index2, ...]}`.
- Agent submits moves via `move` RPC → `{"op": "move", "pin_id": X, "to_index": Y}`.
- Actions are **full turns** — one `(pin_id, to_index)` pair per turn.

### What exists and what is missing
- **Exists**: complete game engine, server, client, scoring, logging, GUI.
- **Missing**: RL agent, training pipeline, baselines, tests, state encoder, environment wrapper.
- **No existing RL infrastructure** — everything must be built from scratch.

### Known constraints
- Turn timeout: currently 10s (will be finalized by professor after team feedback).
- Game time limit: currently 60s (will be finalized by professor after team feedback).
- The professor has asked teams to report average time per move and total game time.

## Deliverable from this phase

Before major implementation, Claude Code should form a clear plan based on the actual repository.

It should avoid creating redundant abstractions if the codebase already has good ones.

It should preserve compatibility with the existing architecture unless there is a strong reason not to.

---

# Phase 2: Choose the right action representation

This is one of the most important design choices.

Chinese Checkers actions can be represented in two broad ways:

## Confirmed: Full-turn action representation

The game engine uses **full-turn actions**. This is settled by the server API:

- `get_legal_moves` returns `{pin_id: [destination1, destination2, ...]}` per pin.
- A move is submitted as `(pin_id, to_index)` — one pair per turn.
- Jump chains are resolved internally by `getPossibleMoves()` — the agent only sees final destinations.

## Action format

Each action is a `(pin_id, to_index)` pair where:
- `pin_id` is 0-9 (each player has 10 pins),
- `to_index` is a board cell index (0-120, but only legal destinations are valid).

## Modeling options

Two ways to encode this for the neural network:

### Option A: Flat action space
Enumerate all `(pin_id, destination)` pairs as a single flat action space. The theoretical max is `10 × 121 = 1210` actions, but in practice most are illegal at any given state. Use **action masking** to restrict to legal moves.

### Option B: Hierarchical selection
Two-step: first select a pin (10 choices), then select a destination for that pin (variable count). This has a smaller branching factor per decision but adds complexity.

### Recommendation
**Prefer flat action space with masking** unless experimentation shows hierarchical is significantly better. Flat is simpler, well-supported by standard RL frameworks, and easier to integrate with MCTS.

---

# Phase 3: State representation

The agent should use a representation that captures both geometry and strategic structure.

## Board geometry

The board has **121 cells** in a hex layout:
- 61 center hexagonal cells (postype `'board'`)
- 60 colored triangle zones (10 cells per color, 6 colors)

Axial coordinates `(q, r)` range roughly from -8 to +8 but only 121 positions are valid. The board is **not rectangular** — it is a star-shaped hex grid.

## Multi-player consideration

The game supports **2 to 6 players**. The state encoding must handle a variable number of opponents. Two approaches:

### Approach A: Fixed 6-slot encoding
Always encode 6 color channels (one per possible player). Empty slots are all-zeros. This is simple and handles any player count.

### Approach B: Relative encoding
Encode from the agent's perspective: "my pieces", "my target", then opponents in turn-order. Pad unused opponent slots. This may generalize better.

**Recommendation**: Start with fixed 6-slot encoding for simplicity.

## Preferred representation
Use a **multi-channel tensor representation** of the board.

### Hex-to-tensor embedding
Since the board is not rectangular, embed the 121 hex cells into a padded 2D grid. One approach:
- Map axial `(q, r)` into a 17×17 grid (q and r each range roughly -8 to +8).
- Only 121 of the 289 grid cells are valid; mask the rest.
- Alternatively, use a flat vector of 121 cells (indexed consistently via `board.index_of`).

### Suggested channels

For a 6-player game with fixed slots:

1. My pieces (binary, 121 cells)
2. Opponent 1 pieces (by turn order)
3. Opponent 2 pieces
4. Opponent 3 pieces
5. Opponent 4 pieces
6. Opponent 5 pieces
7. My target camp cells
8. My home camp cells
9. Valid cell mask (which cells exist on the board)
10. Board-only cells vs colored zones

For a 2-player game, channels 3-6 would be all zeros.

## Additional engineered features
Also consider adding scalar features such as:
- total distance to goal (from scoring formula),
- maximum distance of any piece to goal (lagging piece indicator),
- number of pieces still in home camp,
- number of pieces already in target camp,
- count of currently available legal moves,
- move count so far (relevant for move_score Gaussian centered at 45),
- elapsed time (relevant for time_score),
- number of active players remaining,
- and turn number / position in turn order.

These can either:
- be concatenated into a later MLP layer,
- or be used for reward shaping / debugging only.

---

# Phase 4: Network architecture

## Recommended model
Build a **policy-value network** with two heads.

### Shared trunk
Use a shared feature extractor:
- preferably a CNN or residual CNN if the board can be spatially encoded,
- otherwise a graph-based or MLP-based architecture if the repository structure strongly favors that.

### Policy head
Outputs scores or probabilities over legal actions.

Important:
- the model does **not** need to score illegal actions if action masking is handled cleanly,
- but the implementation must make sure policy selection only uses legal actions.

### Value head
Outputs a scalar estimate of the position, such as:
- expected outcome in [-1, 1], or
- estimated win probability.

## Architectural preference order
1. Residual CNN on board tensor
2. Standard CNN + MLP heads
3. Strong MLP if the state is already feature-vector based and changing representation would be costly

## Engineering requirement
The architecture should be modular and easy to swap.

Create clear separation between:
- state encoding,
- model definition,
- action selection,
- training loop,
- and search.

---

# Phase 5: Action masking and legal move handling

This part must be done carefully.

A strong agent cannot rely on the neural network to learn legality by itself.

## Requirements
- Always generate legal actions from the environment or game engine.
- Always mask out illegal actions.
- The policy head should only be normalized over legal actions.
- MCTS expansion should only create legal children.

## Concrete API for legal moves

For **local training** (fast, no network):
```python
pin.getPossibleMoves()  # returns sorted list of legal destination indices
```

For **competition play** (via server):
```python
rpc({"op": "get_legal_moves", "game_id": ..., "player_id": ...})
# returns {"ok": True, "legal_moves": {pin_id: [dest1, dest2, ...], ...}}
```

Both return the same data — legal destination indices per pin.

## Preferred implementation approach
Use a move-indexing layer that maps between:
- `(pin_id, to_index)` pairs ↔ flat action indices.

The flat action space is `10 pins × 121 cells = 1210` possible actions. At any state, build a binary mask of size 1210 where `mask[pin_id * 121 + to_index] = 1` for each legal `(pin_id, to_index)`.

The policy head outputs logits over all 1210 actions, then apply the mask before softmax:
```python
masked_logits = logits.masked_fill(~legal_mask, -1e9)
probs = softmax(masked_logits)
```

Do not build a fragile action system that breaks whenever legal move counts vary.

---

# Phase 6: MCTS integration

This is likely the most important performance component.

## Recommended role of MCTS
Use MCTS to improve move selection at inference time and to produce stronger policy targets during self-play training.

## MCTS requirements
The MCTS implementation should:
- use policy priors from the policy head,
- use the value head for leaf evaluation,
- support legal-action expansion only,
- track visit counts,
- return action distributions for training,
- and return the best action for play.

## Important implementation details
Claude Code should think carefully about:
- UCB / PUCT formula choice,
- handling of terminal nodes,
- caching repeated evaluations,
- simulation budget per move,
- batching leaf evaluations if possible,
- and compatibility with competition time limits.

## Time budget constraints

**Time limits are not yet finalized.** The professor has stated:

> *"Change TURN_TIMEOUT_SEC and GAME_TIME_LIMIT_SEC accordingly for training. Please let me know your average time taken per move and average total time to finish a game. Once I receive responses from at least 5 different teams, I will base the final values of these parameters accordingly."*

Current defaults: `TURN_TIMEOUT_SEC = 10`, `GAME_TIME_LIMIT_SEC = 60`.

Design implications:
- MCTS simulation count **must be configurable** and adaptive to final limits.
- With 60s total and ~45 optimal moves, the average budget is ~1.3s/move.
- The 10s per-turn hard limit allows bursts for critical moves.
- Design MCTS to work well across a range: 0.5s to 5s per move.
- **Report timing stats to professor** as part of competition preparation.

## Practical guidance
Make the MCTS configurable:
- number of simulations (or time-based budget),
- exploration constant,
- temperature for early-game training,
- deterministic move choice for evaluation.

## Competition mode
In tournament play, use a more stable evaluation mode:
- low or zero temperature,
- time-aware simulation budget (use as much time as safely possible),
- no unnecessary randomness.

---

# Phase 7: Reward shaping

Do not begin with terminal rewards only unless there is a specific reason.

Chinese Checkers has delayed rewards, so reward shaping is useful.

## Competition scoring formula

The server uses a **4-component scoring formula** that determines competition ranking. The agent should be aware of and optimize for this:

```python
final_score = time_score + move_score + pin_goal_score + distance_score
```

### 1. Time score
```python
time_score = max(0.0, 100.0 - total_time_taken_sec)
```
Faster play earns more points. Maximum 100 points if instantaneous.

### 2. Move score (asymmetric Gaussian centered at 45 moves)
```python
sigma = 4 if move_count < 45 else 18
move_score = exp(-((move_count - 45)^2) / (2 * sigma^2))
```
Optimal at ~45 total moves. Heavily penalized below 45 (tight sigma=4), more forgiving above 45 (sigma=18). Maximum ~1.0 points.

### 3. Pin goal score
```python
pin_goal_score = pins_in_goal * 100.0
```
100 points per pin that reaches the opposite zone. Maximum 1000 points (all 10 pins).

### 4. Distance score
```python
distance_score = max(0.0, 200.0 - total_min_distance_to_goal)
```
Where `total_min_distance` is the sum of each non-goal pin's minimum hex distance to any target cell. Maximum 200 points if all pins are in goal.

### Score component weights (approximate)
- **Pin goal score dominates**: up to 1000 points — getting pins to goal is by far the most important.
- **Distance score**: up to 200 points — partial credit for progress even without winning.
- **Time score**: up to 100 points — speed is a tiebreaker, not the primary objective.
- **Move score**: up to ~1.0 points — essentially negligible, but avoid extreme move counts.

## Recommended reward design

The training reward should combine:

### 1. Terminal outcome (primary signal)
- Large positive reward for WIN (all pins in goal).
- Moderate negative reward for DRAW (no legal moves) or TIMEOUT.
- For non-terminal game end (time limit), use the normalized `final_score` as the terminal reward.

### 2. Per-move progress (shaped reward)
Use the change in the competition scoring components as step rewards:
```python
step_reward = α * Δpin_goal_score + β * Δdistance_score
```
Where Δ is the improvement from the previous state. This directly aligns training with the competition metric.

### 3. Lagging-piece penalty
Penalize positions where one or a few pieces remain far behind. This is a high-priority feature.

Suggestion: penalize the **max individual distance** to goal, not just the sum. This discourages leaving stragglers:
```python
lagging_penalty = -γ * max_single_pin_distance_to_goal
```

### 4. Home-camp exit reward
Reward efficient movement out of the starting camp in early game.

### 5. Jump-potential and anti-congestion signals
Reward positions that create future jump opportunities. Penalize formations that trap pieces or reduce mobility.

## Important caution
Reward shaping should help learning, not dominate the objective.

The implementation should make shaping weights (α, β, γ, etc.) configurable and easy to ablate.

A useful workflow is:
- start with moderate shaping aligned to the competition scoring,
- train a strong agent,
- then test reduced shaping to see whether the agent still performs well.

Since the competition scoring heavily weights `pin_goal_score` (1000 max) over everything else, the agent should primarily optimize for getting all pins to goal, with time and distance as secondary objectives.

---

# Phase 8: Distinctive competitive edge

The agent should not only be strong. It should also have a well-motivated edge.

## Recommended special idea
Use a strategic bias toward:

> **advancing lagging pieces and building future jump-chain potential**

This is both practical and distinctive.

Many weaker agents greedily push whichever piece can move furthest now. That often leaves a tail of slow pieces behind. A stronger agent should instead learn or be biased toward:

- improving the worst-positioned pieces,
- creating ladders / structures for future jumps,
- avoiding isolated rear pieces,
- and preserving long-term mobility.

## How to operationalize this
In the system, this can appear in several places:

- reward shaping,
- evaluation features,
- MCTS prior bias,
- training diagnostics,
- and ablation experiments.

This should be presented as an intentional strategy, not an accident.

---

# Phase 8.5: Multi-player handling

The game supports **2 to 6 players**. This has significant implications that cut across many phases.

## Why this matters

Most RL game research focuses on 2-player zero-sum games (e.g., Go, Chess). Chinese Checkers with 3-6 players is fundamentally different:

- **No zero-sum property**: one opponent losing does not directly help you.
- **Turn order matters**: with 6 players, you wait 5 turns between moves.
- **Shared obstacles**: other players' pieces are both obstacles and jump bridges for everyone.
- **Variable opponents**: the agent may face 1, 3, or 5 opponents depending on the game.
- **Alliance-free**: no formal cooperation, but blocking strategies exist.

## State encoding implications

The state encoder must handle variable player counts:
- Use fixed-size encoding (e.g., 6 color channels) with zeros for absent players.
- Include a "number of active players" feature.
- Encode relative turn position (how many turns until my next move).

## Value estimation implications

In a 2-player game, value is straightforward: expected win probability or score.

In a multi-player game, consider:
- **Option A**: Predict own `final_score` (not relative to opponents). This is simplest and directly matches the competition metric.
- **Option B**: Predict rank (1st, 2nd, ..., 6th). More informative but harder to train.
- **Recommendation**: Start with Option A — predict own normalized `final_score`.

## Training implications

- **Start training with 2 players** for simplicity and faster iteration.
- Once the 2-player agent is strong, introduce 4-player and 6-player games.
- The agent should generalize across player counts, not overfit to one configuration.
- Self-play with multiple copies of the same agent is a natural training setup for multi-player.

## Strategy implications

- Other players' pieces create jump opportunities — the agent should learn to exploit these.
- In multi-player games, the path through the center is more congested.
- Turn order position affects optimal strategy (earlier movers may have slight advantages).

---

# Phase 9: Training strategy

## Prerequisite: Local training environment

Training **must not** use the network server. The server (`game.py`) has RPC overhead, per-turn timeouts, and CLI interaction that make it unsuitable for fast self-play.

Instead, build a **local game environment** that:
- Uses `HexBoard` and `Pin` classes directly (from `checkers_board.py` and `checkers_pins.py`).
- Implements the same rules as the server: turn rotation, move validation, win/draw detection.
- Exposes a gym-like interface: `reset()`, `step(action)`, `get_legal_moves()`, `get_state()`.
- Runs entirely in-process with no network or timeouts.
- Supports 2-6 players (start with 2 for initial training).

The shared game logic (`checkers_board.py`, `checkers_pins.py`) is **identical** in both the single-system and multi-system directories. Use these directly.

The server's `Game.check_player_status()` and `Game.compute_scores()` methods contain the win/draw detection and scoring logic. Port these into the local environment wrapper.

## Stage 1: Build a strong baseline first
Before the full search-enhanced system, build a clean baseline that works end to end.

Recommended baseline:
- local game environment wrapper,
- legal action generation,
- masked policy network,
- simple value head,
- self-play,
- reward shaping (aligned with competition scoring),
- checkpointing,
- evaluation against random and greedy opponents.

Only after the pipeline works robustly should more advanced features be layered in.

## Stage 2: Warm start
If practical, initialize training with heuristic guidance.

Possible warm-start approaches:
- imitate a heuristic or shallow-search agent,
- pretrain value estimates from heuristic scores,
- or use simple self-play with hand-coded opponents first.

This can stabilize early learning.

## Stage 3: Self-play RL
Core self-play loop:
1. current agent plays games against itself or a pool,
2. store states, target policies, outcomes, and metadata,
3. train policy-value network,
4. periodically evaluate,
5. replace best model only when validated.

## Stage 4: League / opponent pool training
Do not only train against the newest model.

Maintain a pool such as:
- current best model,
- older model snapshots,
- greedy baseline,
- heuristic baseline,
- possibly noisy or style-diverse opponents.

This reduces overfitting and improves robustness.

## Stage 5: Curriculum if needed
If full-game training is unstable, use curriculum.

Examples:
- smaller board,
- fewer pieces,
- fewer players,
- shorter games,
- reduced simulation count at first.

Only introduce curriculum if it genuinely helps and integrates cleanly with the repo.

---

# Phase 10: Evaluation plan

A strong implementation must include rigorous evaluation.

## Minimum evaluation suite
Evaluate against at least:
- random agent,
- greedy-progress agent,
- heuristic agent,
- earlier versions of itself,
- MCTS-free variant,
- and, if possible, other provided competition agents.

## Metrics to track
Track:
- win rate,
- average game length,
- average progress per move,
- percentage of games where pieces remain stranded,
- policy entropy,
- value prediction calibration,
- and performance as first player vs later player if relevant.

## Critical ablations
Run ablations for:
- without MCTS,
- without lagging-piece penalty,
- without jump-potential shaping,
- without action masking improvements,
- without opponent-pool training.

This helps identify what truly matters.

---

# Phase 11: Code quality requirements

The code should be clean enough that another engineer can understand and extend it.

## Requirements
- Use clear file/module organization.
- Use type hints where appropriate.
- Use descriptive names.
- Avoid giant monolithic scripts.
- Separate training, inference, model, search, environment wrappers, and utilities.
- Add docstrings to important functions and classes.
- Add comments where logic is subtle.
- Avoid hard-coded magic numbers unless they are centralized in config.

## Configuration
Use config objects or config files for:
- model hyperparameters,
- reward weights,
- MCTS settings,
- training loop settings,
- evaluation settings,
- and file paths.

## Reproducibility
Include:
- deterministic seeding where practical,
- checkpoint saving,
- logging of hyperparameters,
- and consistent evaluation settings.

---

# Phase 12: Testing requirements

Testing is required.

## Must-test components
1. **Move generation**
   - legal moves are actually legal,
   - jump chains are complete and correct,
   - terminal states are detected correctly.

2. **Action mapping**
   - move-to-index and index-to-move mappings are correct,
   - masking never allows illegal action selection.

3. **State encoding**
   - encoding matches board state correctly,
   - player perspective is handled correctly.

4. **Model forward pass**
   - tensor shapes are correct,
   - masked policy behaves correctly,
   - value output range is sensible.

5. **MCTS**
   - only legal actions are expanded,
   - terminal states return correct values,
   - visit counts sum correctly,
   - selected move is legal.

6. **Training loop**
   - replay data is valid,
   - losses decrease at least on sanity checks,
   - checkpoints save/load correctly.

7. **End-to-end games**
   - agent can complete full games without crashing,
   - self-play does not deadlock,
   - tournament integration works.

## Strong recommendation
Write small but meaningful unit tests and at least one integration test.

---

# Phase 13: Suggested module structure

This is only a suggestion. Adapt to the repository if there is already a strong structure.

Possible module layout:

```text
agents/
  chinese_checkers_agent.py    # Main RL agent (policy-value + MCTS)
  heuristic_agent.py           # Greedy/heuristic baseline
  random_agent.py              # Random baseline

models/
  policy_value_net.py          # Neural network architecture
  encoders.py                  # Board state → tensor encoding

search/
  mcts.py                      # Monte Carlo Tree Search

training/
  self_play.py                 # Self-play game generation
  replay_buffer.py             # Experience storage
  trainer.py                   # Training loop
  evaluate.py                  # Evaluation against baselines

env/
  local_game.py                # Fast local env wrapping HexBoard + Pin (for training)
  server_adapter.py            # RPC client adapter (for competition play)
  action_mapping.py            # (pin_id, to_index) ↔ flat action index
  rewards.py                   # Reward shaping (aligned with competition scoring)

configs/
  default_agent.yaml
  training.yaml

tests/
  test_moves.py
  test_action_mapping.py
  test_encoding.py
  test_mcts.py
  test_training_smoke.py
```

Key integration points:
- `env/local_game.py` wraps `checkers_board.py` and `checkers_pins.py` for fast training without network overhead.
- `env/server_adapter.py` wraps the RPC calls from `player.py` for competition play.
- The agent should work with both environments via a common interface.

Do not force this exact structure if the repo already has a better one.

---

# Phase 14: Practical implementation order

Claude Code should implement in a controlled order.

## Step 1
Understand the repo thoroughly and identify integration points.

## Step 2
Implement or clean up state encoding and legal move/action mapping.

## Step 3
Implement a baseline policy-value model with masked legal-action selection.

## Step 4
Implement a basic self-play loop and training pipeline.

## Step 5
Add reward shaping and evaluation against simple baselines.

## Step 6
Integrate MCTS.

## Step 7
Add league training / opponent pool.

## Step 8
Tune and ablate.

## Step 9
Harden for competition inference.

---

# Phase 15: Competition readiness checklist

## Concrete competition interface

The agent must work within `player.py`'s framework:
- **Connect**: JSON-RPC over TCP socket to `127.0.0.1:50555`.
- **Join**: `{"op": "join", "player_name": "..."}` → receive `game_id`, `player_id`, `colour`.
- **Start**: `{"op": "start", "game_id": "...", "player_id": "..."}`.
- **Poll state**: `{"op": "get_state", "game_id": "..."}` → full game state JSON.
- **Get legal moves**: `{"op": "get_legal_moves", "game_id": "...", "player_id": "..."}`.
- **Submit move**: `{"op": "move", "game_id": "...", "player_id": "...", "pin_id": N, "to_index": N}`.

## Time constraints (TBD — design for flexibility)

Current defaults: `TURN_TIMEOUT_SEC = 10`, `GAME_TIME_LIMIT_SEC = 60`.

These **will be adjusted** by the professor after receiving timing reports from at least 5 teams. Design the agent to work well across a range of time budgets.

**Action item**: Report average time per move and average total game time to the professor.

## Checklist

Before finalizing, verify that:

- the agent always returns a legal move,
- inference fits within the per-turn timeout (currently 10s, expect changes),
- total game time stays within the game time limit (currently 60s, expect changes),
- the agent handles `turn_timeout_notice` gracefully (turn was skipped),
- the agent handles `FINISHED` status and reads final scores,
- checkpoints load correctly,
- evaluation mode is deterministic when desired,
- the system does not rely on unavailable files,
- all dependencies are available in the competition environment,
- the agent works correctly with 2, 4, or 6 players,
- and the final selected model was validated properly.

Also verify that the submission version is the strongest stable version, not merely the newest one.

---

# Phase 16: What not to do

Avoid the following mistakes:

- Do not jump into coding before understanding the repo.
- Do not build a brittle action-indexing system.
- Do not rely only on terminal rewards from the start.
- Do not skip evaluation against simple baselines.
- Do not assume a greedy forward-progress policy is strong enough.
- Do not ignore lagging pieces.
- Do not overcomplicate the architecture if the bottleneck is elsewhere.
- Do not break tournament compatibility for a theoretically nicer internal design.
- Do not leave the project in an untested state.
- Do not train on the network server — use a local environment wrapper for speed.
- Do not assume 2-player only — the competition may use 4 or 6 players.
- Do not ignore the competition scoring formula — optimize for `final_score`, not just win/loss.
- Do not hard-code time budgets — they will be adjusted by the professor.
- Do not forget to report timing stats to the professor for limit calibration.

---

# Final instruction to Claude Code

Your task is to behave like a careful senior engineer and research-minded builder.

First, gain a deep understanding of the current files and architecture. Trace how the existing system works. Identify constraints, interfaces, and the best integration strategy.

Then design and implement the strongest practical Chinese Checkers agent you can within the repository’s real constraints.

Favor correctness, robustness, legality, strong search integration, and high-quality engineering.

Think carefully before major design choices. Make deliberate decisions. Build step by step. Test thoroughly. Improve weak points. Optimize for actual competitive performance.

The final result should be a well-integrated, high-quality Chinese Checkers agent that has a serious chance of performing very strongly in competition.

