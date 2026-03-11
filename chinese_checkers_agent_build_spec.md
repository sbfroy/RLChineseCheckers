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

## Specific inspection tasks

Inspect and understand at least the following:

### 1. Project structure
- Identify top-level folders and their roles.
- Identify where the game logic lives.
- Identify where agents are implemented.
- Identify whether there is already an RL pipeline.
- Identify whether there are tests.
- Identify whether there are config files, scripts, notebooks, or experiment utilities.

### 2. Environment / game engine
Understand exactly:
- how the board is represented,
- how players are indexed,
- how legal moves are generated,
- how jump chains are handled,
- how terminal states are detected,
- how winners are defined,
- how turn progression works,
- whether the setup is 2-player or multi-player,
- and whether the competition uses standard or modified rules.

### 3. Agent interface
Find:
- the exact API an agent must implement,
- how an agent receives state information,
- what the expected action format is,
- whether actions are full turns or atomic steps,
- and how the tournament runner invokes agents.

### 4. Training / evaluation pipeline
Check whether the repository already includes:
- model training code,
- replay buffers,
- evaluation scripts,
- logging,
- checkpointing,
- or metrics dashboards.

### 5. Existing assumptions and constraints
Identify constraints such as:
- runtime budget per move,
- memory limits,
- file submission restrictions,
- package restrictions,
- required Python version,
- framework already used in the repo,
- and any competition-specific rules.

## Deliverable from this phase

Before major implementation, Claude Code should form a clear plan based on the actual repository.

It should avoid creating redundant abstractions if the codebase already has good ones.

It should preserve compatibility with the existing architecture unless there is a strong reason not to.

---

# Phase 2: Choose the right action representation

This is one of the most important design choices.

Chinese Checkers actions can be represented in two broad ways:

## Option A: Full-turn action representation
A single action represents one complete legal move for a turn, including an entire jump chain if applicable.

### Advantages
- simpler integration with a standard game engine,
- easier tournament compatibility,
- simpler interface between agent and environment.

### Disadvantages
- potentially very large action space,
- more difficult policy output if encoded as a fixed-size action space.

## Option B: Submove / branching representation
A turn is decomposed into smaller decisions:
- choose a piece,
- choose the next hop or step,
- continue while jumps remain,
- terminate turn when finished.

### Advantages
- smaller branching factor at each decision,
- easier to model complex jump chains,
- potentially better for RL.

### Disadvantages
- may require significant environment changes,
- may not match competition interface,
- more engineering complexity.

## Recommended default
Unless the current repository is already built around branching submoves, prefer:

> **Use full-turn legal actions externally**, while internally supporting efficient move generation and indexing.

This is usually the safest for competition integration.

However, Claude Code should inspect the current repo first. If the engine already has a clean submove abstraction, it may be worth using.

---

# Phase 3: State representation

The agent should use a representation that captures both geometry and strategic structure.

## Preferred representation
Use a **multi-channel tensor representation** of the board.

Possible channels include:

1. My pieces
2. Opponent pieces
3. Empty valid cells
4. My target camp cells
5. My home camp cells
6. Opponent target/home camps if useful
7. Current player indicator
8. Optional jump-availability indicators
9. Optional repetition or move-count features

If the board is not naturally rectangular, either:
- embed it into a padded rectangular tensor, or
- use an indexed graph-style representation if the repo already supports that better.

## Additional engineered features
Also consider adding scalar features such as:
- total distance to goal,
- maximum distance of any piece to goal,
- number of pieces still in home camp,
- number of pieces already in target camp,
- count of currently available jump moves,
- connectivity / clustering features,
- and turn number if relevant.

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

## Preferred implementation approach
Use a move-indexing layer that can:
- map legal move objects to action indices,
- map action indices back to move objects,
- and support fast lookup during training and inference.

If the number of possible full-turn actions is variable, handle that cleanly. For example:
- score candidate legal moves directly,
- or project legal moves into embeddings and rank them.

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

## Practical guidance
Make the MCTS configurable:
- number of simulations,
- exploration constant,
- temperature for early-game training,
- deterministic move choice for evaluation.

## Competition mode
In tournament play, use a more stable evaluation mode:
- low or zero temperature,
- enough simulations to be strong within time limits,
- no unnecessary randomness.

---

# Phase 7: Reward shaping

Do not begin with terminal rewards only unless there is a specific reason.

Chinese Checkers has delayed rewards, so reward shaping is useful.

## Recommended reward components
A good reward function can include:

### 1. Terminal outcome
- +1 for win
- -1 for loss
- optionally 0 for draw or unfinished outcome depending on rules

### 2. Progress toward target
Reward reductions in total distance from pieces to their target camp.

### 3. Lagging-piece penalty
Penalize positions where one or a few pieces remain far behind.

This is a high-priority feature.

### 4. Home-camp exit reward
Reward efficient movement out of the starting camp.

### 5. Target-camp occupation reward
Reward successfully placing pieces in the target camp, especially when placements are stable and useful.

### 6. Jump-potential reward
Reward positions that create future jump opportunities, especially for lagging pieces.

### 7. Anti-congestion / anti-isolation signals
Penalize formations that trap pieces or reduce mobility.

## Important caution
Reward shaping should help learning, not dominate the objective.

The implementation should make shaping weights configurable and easy to ablate.

A useful workflow is:
- start with moderate shaping,
- train a strong agent,
- then test reduced shaping to see whether the agent still performs well.

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

# Phase 9: Training strategy

## Stage 1: Build a strong baseline first
Before the full search-enhanced system, build a clean baseline that works end to end.

Recommended baseline:
- legal action generation,
- masked policy network,
- simple value head,
- self-play,
- reward shaping,
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
  chinese_checkers_agent.py
  heuristic_agent.py
  random_agent.py

models/
  policy_value_net.py
  encoders.py

search/
  mcts.py

training/
  self_play.py
  replay_buffer.py
  trainer.py
  evaluate.py

env/
  wrappers.py
  action_mapping.py
  rewards.py

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

Before finalizing, verify that:

- the agent always returns a legal move,
- inference is fast enough for competition limits,
- checkpoints load correctly,
- evaluation mode is deterministic when desired,
- the system does not rely on unavailable files,
- all dependencies are available in the competition environment,
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

---

# Final instruction to Claude Code

Your task is to behave like a careful senior engineer and research-minded builder.

First, gain a deep understanding of the current files and architecture. Trace how the existing system works. Identify constraints, interfaces, and the best integration strategy.

Then design and implement the strongest practical Chinese Checkers agent you can within the repository’s real constraints.

Favor correctness, robustness, legality, strong search integration, and high-quality engineering.

Think carefully before major design choices. Make deliberate decisions. Build step by step. Test thoroughly. Improve weak points. Optimize for actual competitive performance.

The final result should be a well-integrated, high-quality Chinese Checkers agent that has a serious chance of performing very strongly in competition.

