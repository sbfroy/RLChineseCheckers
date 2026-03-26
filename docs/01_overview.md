# 1. Project Overview

## What is this project?

This project builds an AI agent that plays Chinese Checkers. The agent is designed to compete against other teams' agents in a university competition. The competition runs on a server (`game.py`) that manages the game, and each team provides a client (`player.py`) that connects to it and makes moves.

The approach we're using is the same family of techniques behind AlphaGo and AlphaZero — a combination of **deep learning** and **tree search**. The idea is:

1. Train a neural network to evaluate board positions and suggest good moves
2. Use that neural network to guide a search algorithm (MCTS) that looks ahead in the game tree
3. The search algorithm produces better training data than the network alone
4. The network trains on that better data and improves
5. Repeat — each cycle makes both the network and search stronger

## Why this approach?

There are simpler approaches to playing Chinese Checkers:

- **Random agent**: Pick any legal move at random. Very weak.
- **Greedy agent**: Always move the piece that gets closest to the goal. Better, but shortsighted.
- **Heuristic agent**: Like greedy but also considers lagging pieces. Decent, but doesn't plan ahead.

The problem with all of these is they don't **think ahead**. Chinese Checkers has delayed rewards — a move might look bad now but set up a chain of jumps 3 turns later. Only an agent that can search the game tree and learn patterns will discover these strategies.

Our agent combines:
- A **neural network** that learns patterns from thousands of games
- **MCTS** (Monte Carlo Tree Search) that uses the network to search ahead
- **Self-play** where the agent plays against itself to generate training data

## What has been built?

The complete system, ready for training. Here's the file layout:

```
RLChineseCheckers/
├── single system/              # Original local game (not used by us)
├── multi system .../           # Competition server + client
│   ├── game.py                 # Server (professor's code, don't modify)
│   ├── player.py               # Client (our agent plugs in here)
│   ├── checkers_board.py       # Board representation (we reuse this)
│   └── checkers_pins.py        # Piece/move logic (we reuse this)
│
├── env/                        # Environment layer
│   ├── local_game.py           # Fast game simulator for training
│   ├── action_mapping.py       # Converts moves to/from neural net format
│   ├── rewards.py              # Reward shaping for training
│   └── server_adapter.py       # Connects our agent to competition server
│
├── models/                     # Neural network
│   ├── encoders.py             # Board state → tensor conversion
│   └── policy_value_net.py     # The neural network itself
│
├── search/                     # Search algorithm
│   └── mcts.py                 # Monte Carlo Tree Search
│
├── training/                   # Training pipeline
│   ├── self_play.py            # Generates games for training
│   ├── replay_buffer.py        # Stores training examples
│   ├── trainer.py              # The training loop
│   └── evaluate.py             # Tests agent against baselines
│
├── agents/                     # Agent implementations
│   ├── chinese_checkers_agent.py  # Our main RL agent
│   ├── random_agent.py         # Random baseline
│   └── heuristic_agent.py      # Greedy + heuristic baselines
│
├── configs/                    # Configuration files
│   ├── default.yaml            # Full training config
│   └── fast_bootstrap.yaml     # Fast initial training config
│
├── tests/                      # Test suite (48 tests)
├── train.py                    # Main training entry point
├── TRAINING.md                 # Training instructions
└── docs/                       # This documentation
```

## How do the pieces connect?

Here's the data flow at a high level:

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Local Game  │────>│ State Encoder │────>│ Neural Network│
│ (local_game) │     │  (encoders)   │     │(policy_value) │
└──────┬──────┘     └──────────────┘     └──────┬───────┘
       │                                        │
       │         ┌──────────┐                   │
       │<────────│   MCTS   │<──────────────────┘
       │         │ (search) │   uses network to
       │         └────┬─────┘   evaluate positions
       │              │
       │              v
       │    ┌──────────────────┐
       └───>│  Training Loop   │
            │ (self_play +     │
            │  trainer)        │
            └──────────────────┘
                    │
                    v
            Network gets better
                    │
                    v
              Repeat cycle
```

**During training:**
1. The agent plays games against itself using `local_game.py`
2. Each board state is converted to tensors by `encoders.py`
3. The neural network suggests moves and evaluates positions
4. MCTS uses the network to search deeper (optional but makes training stronger)
5. The game results become training data stored in the `replay_buffer`
6. The `trainer` updates the network weights to match the game outcomes
7. The improved network plays better games next iteration

**During competition:**
1. `server_adapter.py` connects to the game server
2. Receives board state via JSON-RPC
3. Encodes the state, runs the network (+ MCTS if time allows)
4. Sends the chosen move back to the server

## What state is the project in right now?

**Infrastructure: complete.** Every component is built and tested (48 tests, all passing). The training pipeline runs end-to-end.

**Training: barely started.** The model has only been trained for a few minutes as a smoke test. It's essentially untrained — about as strong as a random player. The entire value of this system comes from training it, which takes hours to days.

Think of it like building a car: the engine, chassis, wheels, and steering are all assembled and tested. But we haven't driven it yet. The training process is "driving" — it's where the agent actually learns to play well.
