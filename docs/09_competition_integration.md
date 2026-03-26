# 9. Competition Integration

## How the competition works

The professor runs a game server (`game.py`). Each team runs a client (`player.py`) that connects over TCP on port 50555. The communication is JSON-RPC — you send JSON messages and get JSON responses.

The game flow:
1. Server is started, admin creates a game
2. Players connect and join
3. When enough players join, they send "start"
4. Server manages turn rotation
5. On each turn, you request legal moves, think, and submit a move
6. Game ends on win, draw, or timeout

## What `env/server_adapter.py` provides

`CompetitionPlayer` is a drop-in replacement for the random logic in `player.py`. It:

1. Connects to the server and joins a game
2. Waits for the game to start
3. On each turn:
   - Gets the board state from the server
   - Gets legal moves from the server
   - Passes them to `ChineseCheckersAgent.select_action_from_server_state()`
   - Submits the chosen move
4. Prints timing stats and final scores

### Running it

```bash
# Start the server first (in one terminal):
cd "multi system single machine minimal"
python3.10 game.py
# Type "create" to create a game

# Run the RL agent (in another terminal):
python3.10 -c "
import sys; sys.path.insert(0, '.')
from env.server_adapter import CompetitionPlayer
player = CompetitionPlayer(
    player_name='MyRLAgent',
    checkpoint_path='checkpoints/model_final.pt',
    mcts_simulations=100,
    time_limit=2.0,  # seconds per move
)
player.run()
"
```

## How the agent picks moves in competition

`ChineseCheckersAgent.select_action_from_server_state()` is designed for server play:

1. Receives raw data from the server: pin positions, legal moves, turn order
2. Builds a legal action mask (1210 booleans)
3. Reconstructs a `LocalGame` from the server state using `LocalGame.from_server_state()`
4. Runs full MCTS search on the reconstructed game (same as training)
5. Returns the highest-visit-count action from the MCTS policy

If MCTS returns nothing (edge case), or if `use_mcts=False` is passed, it falls back to **direct network policy** — encoding the state and picking the highest-probability legal action from the raw network output. This fallback is fast (<10ms) but weaker than MCTS.

You can control this with the `use_mcts` parameter:
```python
# Full MCTS (default, stronger, ~0.5-1.5s per move on CPU)
agent.select_action_from_server_state(..., use_mcts=True)

# Direct policy only (fast fallback, <10ms, for time pressure)
agent.select_action_from_server_state(..., use_mcts=False)
```

## Time management

The time constraints are:
- **Per-turn timeout:** Currently 10s (professor may change)
- **Total game time:** Currently 60s (professor may change)

With ~45 optimal moves and 60s total, that's ~1.3s average per move. The neural network forward pass takes ~2-5ms on CPU, so direct-policy play is well within budget. If using MCTS with 100 simulations, expect ~0.5-1.5s per move on CPU.

The `time_limit` parameter on MCTS controls this:
```python
mcts = MCTS(time_limit=1.0)  # stop searching after 1 second
```

## Integrating with `player.py` directly

If you prefer to modify `player.py` directly instead of using `server_adapter.py`, replace the PLAYING LOGIC section (lines 153-183) with:

```python
# At the top of player.py, add:
import sys
sys.path.insert(0, '/path/to/RLChineseCheckers')
from agents.chinese_checkers_agent import ChineseCheckersAgent
agent = ChineseCheckersAgent(checkpoint_path='checkpoints/model_final.pt')

# Replace the PLAYING LOGIC section:
legal_moves_int = {int(k): v for k, v in legal_moves.items()}
movable = {pid: moves for pid, moves in legal_moves_int.items() if moves}
pid, to_index = agent.select_action_from_server_state(
    pin_positions=state.get("pins", {}),
    legal_moves=movable,
    my_colour=colour,
    turn_order=state.get("turn_order", []),
    move_count=state.get("move_count", 0),
)
```

## Pre-competition checklist

Before submitting:
- [ ] Agent always returns a legal move (guaranteed by action masking)
- [ ] Inference time per move is within budget (check timing output)
- [ ] Total game time stays within limit
- [ ] Agent handles `turn_timeout_notice` (turn was skipped — just continue)
- [ ] Agent handles `FINISHED` status (game ended)
- [ ] Checkpoint file is included with the submission
- [ ] Agent works with 2, 4, and 6 players
- [ ] The selected checkpoint is the best one, not just the latest one
- [ ] Report average time per move and total game time to the professor
