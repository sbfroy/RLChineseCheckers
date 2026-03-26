# 4. State Encoding

## The problem

A neural network doesn't understand game boards. It understands **numbers** — specifically, multi-dimensional arrays of floating point numbers (tensors). We need a way to convert a Chinese Checkers board position into a tensor that preserves the spatial relationships between cells.

## The approach: multi-channel 2D grid

### Mapping hex cells to a grid

The board has 121 cells with axial coordinates `(q, r)` where both range from -8 to +8. We embed these into a **17×17 grid** by shifting: `grid_row = r + 8`, `grid_col = q + 8`.

Only 121 of the 289 grid cells are valid (the rest are off the board). We include a "valid cell mask" channel so the network knows which cells exist.

Why a 2D grid? Because **convolutional neural networks (CNNs)** are designed to find spatial patterns in grids. By laying out the board as a grid, the CNN can learn patterns like "pieces near the center" or "clusters of pieces" just like it would learn edges in an image.

### The 10 channels

Each channel is a 17×17 grid where each cell is either 0 or 1. Think of it like 10 overlaid maps of the board, each showing different information:

| Channel | What it shows | Why it matters |
|---------|---------------|----------------|
| **0: My pieces** | 1 where my pins are, 0 elsewhere | The network needs to know what it controls |
| **1-5: Opponent pieces** | One channel per opponent, in turn order | Know where obstacles and jump bridges are |
| **6: My target zone** | 1 for the 10 cells I need to reach | The network needs to know the goal |
| **7: My home zone** | 1 for the 10 cells I started in | Helps learn "get out of home" patterns |
| **8: Valid cell mask** | 1 for all 121 real cells, 0 for padding | Tells the CNN which grid cells are real |
| **9: Board-only cells** | 1 for the 61 center cells, 0 for zones | Distinguishes center from star points |

### Perspective encoding

The encoding is always **from the agent's point of view**. If the agent plays red:
- Channel 0 = red's pieces
- Channel 1 = blue's pieces (the only opponent in a 2-player game)
- Channel 6 = blue's home zone (which is red's target)

If the agent plays blue:
- Channel 0 = blue's pieces
- Channel 1 = red's pieces
- Channel 6 = red's home zone (which is blue's target)

This means the network learns one general concept — "my pieces should move toward channel 6" — regardless of which colour it's actually playing. It doesn't need to learn separate strategies for red vs blue.

### Multi-player handling

In a 6-player game, channels 1-5 hold one opponent each, ordered by turn rotation starting from the next player after us. In a 2-player game, channels 2-5 are all zeros. The network learns to work with any number of opponents.

## Scalar features

Some important information doesn't fit naturally on the spatial grid. We encode 10 scalar (single-number) features:

| Index | Feature | Range | Why it helps |
|-------|---------|-------|--------------|
| 0 | Total distance to goal (normalized) | 0-1 | Overall progress indicator |
| 1 | Max single-pin distance (normalized) | 0-1 | **Lagging piece detector** |
| 2 | Fraction of pins still in home | 0-1 | Early-game indicator |
| 3 | Fraction of pins in goal | 0-1 | Late-game indicator |
| 4 | Number of legal actions (normalized) | 0-1 | Mobility indicator |
| 5 | Total game move count (normalized) | 0-1 | Game phase indicator |
| 6 | My own move count (normalized) | 0-1 | Move efficiency tracking |
| 7 | Number of active players (normalized) | 0-1 | Multi-player awareness |
| 8 | My position in turn order (normalized) | 0-1 | Turn advantage awareness |
| 9 | Distance from optimal 45 moves | 0-1 | Move score optimization |

These scalar features are concatenated with the CNN output inside the neural network, giving the decision-making layers access to both spatial patterns and global game state.

## How encoding is used in practice

```python
from models.encoders import BoardEncoder

encoder = BoardEncoder()  # create once, reuse

# From a LocalGame:
spatial, scalars = encoder.encode_from_game(game, 'red')
# spatial: tensor of shape (10, 17, 17)
# scalars: tensor of shape (10,)

# From raw data (for competition play):
spatial, scalars = encoder.encode(
    my_colour='red',
    pin_positions={'red': [111,112,...], 'blue': [0,1,...]},
    turn_order=['red', 'blue'],
    move_count=15,
    ...
)
```

The encoder object precomputes all the zone masks and coordinate mappings once at creation, so repeated encoding is fast.

## Why this specific design?

1. **CNN-compatible:** The 17×17 grid lets us use standard convolutional layers, which are proven effective for spatial pattern recognition.

2. **Perspective-relative:** By always encoding from "my" perspective, the network learns transferable knowledge — a good pattern for red is the same pattern for blue, just on a different part of the board.

3. **Information-complete:** Between the 10 spatial channels and 10 scalars, the network has access to everything it needs — piece positions, goals, game phase, and mobility.

4. **Efficient:** The encoding takes microseconds. It's never a bottleneck.
