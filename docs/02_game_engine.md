# 2. The Game Engine

## How Chinese Checkers works in this codebase

The game engine is the professor's code in `multi system single machine minimal/`. We don't modify it — we build on top of it. Understanding it deeply is important because everything we build depends on its data structures.

## The Board (`checkers_board.py`)

### Shape and coordinates

The board is a **six-pointed star** made of hexagonal cells. It uses **axial coordinates** `(q, r)` where each cell has two numbers describing its position.

```
        b b b b           <-- blue's home (top)
       b b b · ·
      b b · · · ·
     b · · · · · ·
    · · · · · · · · ·     <-- center board
     · · · · · · · ·
      · · · · · · ·
       · · · · · ·
        · · · · ·
       · · · · · ·
      · · · · · · ·
     · · · · · · · ·
    · · · · · · · · ·     <-- center board
     r · · · · · · ·
      r r · · · · ·
       r r r · · ·
        r r r r           <-- red's home (bottom)
```

There are **121 cells** total:
- **61 center cells** (postype `'board'`) — the main playing area
- **60 coloured cells** (10 per colour, 6 colours) — the star points

The 6 colours come in opposing pairs:
- **red ↔ blue** (top-bottom)
- **lawn green ↔ gray0** (upper-right ↔ lower-left)
- **yellow ↔ purple** (upper-left ↔ lower-right)

### Cell indexing

Every cell gets a number from **0 to 120**. The cells are sorted by `(r, q)` — row first, then column. This means:
- **Blue's home** (top of the board) = indices **0-9**
- **Red's home** (bottom of the board) = indices **111-120**
- The center cells are in between

This indexing is stable — it never changes. When we say "pin is at index 65", that always means the same physical cell on the board.

### Key methods

```python
board = HexBoard()

board.cells[idx]           # Get the cell object at index idx
board.cells[idx].q         # Axial q coordinate
board.cells[idx].r         # Axial r coordinate
board.cells[idx].postype   # 'board', 'red', 'blue', etc.
board.cells[idx].occupied  # True if a piece is here

board.index_of[(q, r)]     # Get index from coordinates
board.axial_of_colour('red')  # List of indices for red's home zone
board.colour_opposites['red'] # Returns 'blue' (the target zone)
```

## The Pieces (`checkers_pins.py`)

### Pin objects

Each player has **10 pins** (pieces). A Pin knows:
- `pin.axialindex` — which cell it's on (0-120)
- `pin.id` — its number within the player's set (0-9)
- `pin.color` — which colour it belongs to
- `pin.board` — reference to the board

### How moves work

A pin can move in two ways:

**1. Single step:** Move to any adjacent empty cell. Each hex cell has up to 6 neighbours (the 6 directions in a hex grid).

**2. Jump chain:** Jump over an occupied adjacent cell to land on the empty cell beyond it. You can chain multiple jumps in one turn — as long as each individual hop is legal, you can keep going. This is how pieces travel long distances in a single turn.

```python
pin.getPossibleMoves()  # Returns sorted list of all legal destination indices
```

This method uses DFS (depth-first search) to find all reachable cells via jump chains. It returns every legal destination — both single steps and multi-hop endpoints.

**Important:** The method only checks if cells are occupied and on the board. It doesn't restrict which zones you can enter. A red piece CAN move through yellow's zone, for example.

### Moving a pin

```python
pin.placePin(new_index)  # Moves the pin, updates board occupancy
```

This updates `board.cells[old].occupied = False` and `board.cells[new].occupied = True`.

## The Competition Server (`game.py`)

### How a game runs

1. Server creates a game
2. Players connect via TCP and join with `{"op": "join"}`
3. Each player gets assigned a colour (in pairs)
4. Players send `{"op": "start"}` when ready
5. Server manages turn rotation
6. On your turn: call `get_legal_moves`, then `move`
7. Game ends when someone wins (all 10 pins in opposite zone) or time runs out

### The scoring formula

This is critical — the competition ranks teams by **score**, not just win/loss:

```
final_score = time_score + move_score + pin_goal_score + distance_score
```

| Component | Max points | Formula | What it means |
|-----------|-----------|---------|---------------|
| **pin_goal_score** | 1000 | pins_in_goal × 100 | **Most important.** 100 points per pin that reaches the opposite zone. |
| **distance_score** | 200 | 200 - total_distance_to_goal | Partial credit for progress even without winning. |
| **time_score** | 100 | 100 - seconds_taken | Faster play scores more. A tiebreaker. |
| **move_score** | ~1 | Gaussian centered at 45 moves | Negligible. Optimal at 45 total moves. |

**Key insight:** Getting pins to the goal zone is worth 5x more than distance progress. The agent should prioritize completing pieces over making partial progress with all of them.

### Time constraints

- **Per-turn timeout:** 10 seconds (professor may change)
- **Total game time:** 60 seconds (professor may change)
- If you exceed the per-turn timeout, your turn is skipped
- If total time runs out, the game ends and scores are computed

## How our system uses the game engine

We **don't** train on the server. The server has network overhead, timeouts, and CLI interaction that make it too slow for training. Instead:

- **For training:** We use `env/local_game.py`, which wraps `HexBoard` and `Pin` directly. Same rules, same moves, no network. A game runs in ~2 seconds instead of minutes.
- **For competition:** We use `env/server_adapter.py`, which connects to the server and translates between JSON-RPC and our agent's interface.

Both use the exact same underlying game engine code (`checkers_board.py` and `checkers_pins.py`), so moves that are legal in training are legal in competition.
