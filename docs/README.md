# Documentation Index

Read these in order for a full understanding of the project.

| # | Document | What you'll learn |
|---|----------|-------------------|
| 1 | [Overview](01_overview.md) | What this project is, why this approach, what's been built, and how the pieces connect |
| 2 | [Game Engine](02_game_engine.md) | How the board, pieces, and moves work in the existing codebase |
| 3 | [Local Game and Actions](03_local_game_and_actions.md) | The training environment and how moves become numbers (action space) |
| 4 | [State Encoding](04_state_encoding.md) | How a board position becomes a tensor the neural network understands |
| 5 | [Neural Network](05_neural_network.md) | The policy-value network architecture and what each part does |
| 6 | [MCTS](06_mcts.md) | How Monte Carlo Tree Search works and why it's the key to strong play |
| 7 | [Training Pipeline](07_training_pipeline.md) | Self-play, replay buffer, losses, rewards — the full training loop |
| 8 | [Baselines and Evaluation](08_baselines_and_evaluation.md) | The three baseline agents and how to measure progress |
| 9 | [Competition Integration](09_competition_integration.md) | How to connect the trained agent to the game server |
| 10 | [Road Forward](10_road_forward.md) | Step-by-step plan for training, what to tune, what to watch for, timeline |

## Quick reference

| I want to... | Read... |
|--------------|---------|
| Understand what was built | Doc 1 |
| Understand the game rules and code | Doc 2 |
| Start training | Doc 10 (or `TRAINING.md` for just commands) |
| Understand why training isn't working | Doc 7 + Doc 10 (troubleshooting) |
| Tune the agent | Doc 10 (reward shaping section) |
| Connect to the competition server | Doc 9 |
| Understand what MCTS does | Doc 6 |
| Understand the neural network | Doc 5 |
