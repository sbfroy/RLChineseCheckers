# 3. Local Game Environment and Action Space

## Why we need a local game environment

Training a reinforcement learning agent requires playing **thousands of games**. The competition server (`game.py`) is designed for human-speed play with network communication, timeouts, and CLI interaction. It would be far too slow for training.

`env/local_game.py` solves this by wrapping the same game engine (`HexBoard` + `Pin`) in a fast, in-process interface. No network, no timeouts, no GUI — just pure game logic.

## How `local_game.py` works

### The interface

```python
from env.local_game import LocalGame

game = LocalGame(num_players=2)  # supports 2, 4, or 6
state = game.reset()             # start a new game, returns initial state

# Game loop
while not game.done:
    colour = game.current_colour()           # whose turn is it?
    legal = game.get_legal_moves(colour)     # {pin_id: [dest1, dest2, ...]}

    pin_id, to_index = pick_a_move(legal)    # your agent decides
    state, done, info = game.step(pin_id, to_index)  # execute the move

scores = game.compute_scores()               # competition scoring
```

### What `reset()` does

1. Creates a fresh `HexBoard`
2. Assigns colour pairs (red+blue for 2 players, add green+gray for 4, add yellow+purple for 6)
3. Places 10 pins per player in their home zones
4. Sets up turn order
5. Returns the initial state dictionary

### What `step(pin_id, to_index)` does

1. Validates the move is legal
2. Moves the pin (updates board occupancy)
3. Checks for win (all 10 pins in opposite zone)
4. Checks for max moves limit
5. Advances to next player's turn
6. Returns `(state_dict, done_bool, info_dict)`

### What `clone()` does

Creates a **deep copy** of the entire game state — new board, new pins, independent from the original. This is essential for MCTS, which needs to simulate future moves without affecting the real game.

### Distance computation

```python
game.compute_distance_info('red')
# Returns:
# {
#   "total_distance": 110,      # sum of all pin distances to goal
#   "max_distance": 13,         # furthest pin from goal (the "lagging" piece)
#   "pins_in_goal": 0,          # how many pins reached the target zone
#   "pins_in_home": 10,         # how many pins still in starting zone
#   "per_pin_distances": [...]   # individual distances
# }
```

This is used for reward shaping — the training signal that guides learning.

## The action space (`action_mapping.py`)

### The problem

A neural network needs a **fixed-size output**. But the number of legal moves varies wildly — sometimes a pin can reach 20 destinations via jump chains, sometimes it has no moves at all.

### The solution: flat action space with masking

Every possible move is a `(pin_id, to_index)` pair:
- `pin_id` ranges from 0 to 9 (10 pins)
- `to_index` ranges from 0 to 120 (121 cells)

So the **theoretical maximum** is 10 × 121 = **1210 possible actions**.

We assign each pair a flat index:
```
flat_index = pin_id × 121 + to_index
```

For example:
- Pin 0 moving to cell 0 = flat index 0
- Pin 0 moving to cell 120 = flat index 120
- Pin 5 moving to cell 60 = flat index 5×121 + 60 = 665
- Pin 9 moving to cell 120 = flat index 9×121 + 120 = 1209

### Action masking

At any given board state, most of these 1210 actions are **illegal**. Maybe only 14 are legal. We handle this with a **binary mask** — a vector of 1210 true/false values where `True` means "this action is legal right now."

```python
from env.action_mapping import build_legal_mask

legal_moves = game.get_legal_moves('red')  # {pin_id: [dest1, dest2, ...]}
mask = build_legal_mask(legal_moves)       # tensor of shape (1210,), dtype=bool
# mask[665] = True means pin 5 can move to cell 60
```

The neural network outputs scores for all 1210 actions, but before we pick one, we set all illegal actions to negative infinity:

```python
masked_logits = logits.masked_fill(~mask, -1e9)
probabilities = softmax(masked_logits)  # only legal actions get probability
```

This guarantees the agent **never makes an illegal move**, regardless of what the network outputs.

### Why this design?

- **Simple:** One flat vector, no hierarchical selection
- **Compatible with MCTS:** Each action maps cleanly to a tree branch
- **Safe:** Action masking is a hard constraint, not learned — the network can't accidentally pick an illegal move
- **Standard:** This is the same approach used in AlphaZero for chess/Go
