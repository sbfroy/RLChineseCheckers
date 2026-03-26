# 6. Monte Carlo Tree Search (MCTS)

## Why MCTS matters

MCTS is arguably the single most important component for competitive strength. Here's why:

The neural network on its own makes "gut feeling" decisions — it sees a position and immediately guesses what to do. This is fast but limited. It can't think ahead. It can't consider "if I move here, they'll move there, then I can jump..."

MCTS adds **lookahead**. It simulates many possible futures, guided by the neural network, and finds moves that are actually good — not just moves that look good at first glance.

In AlphaZero:
- The **network alone** plays at a strong amateur level
- The **network + MCTS** plays at superhuman level
- MCTS is what bridges that gap

## How MCTS works — step by step

MCTS builds a **tree** of game positions. Each node in the tree represents a board state. Each edge represents a move. The algorithm repeatedly:

1. **Select** — Walk down the tree, picking the most promising branch at each level
2. **Expand** — When you reach a leaf (unexplored position), add it to the tree
3. **Evaluate** — Use the neural network to score the new position
4. **Backpropagate** — Update all ancestors with the evaluation result

After many repetitions (simulations), the root node has strong statistics about which moves are good.

### Detailed walkthrough of one simulation

Imagine we're at move 10 of a game. Red wants to choose a move.

```
         ROOT (current position)
        /     |      \
    move A  move B  move C     (red's legal moves)
      |       |       |
     ...     ...     ...       (future positions)
```

**Step 1: Select**

Starting at the root, pick the child node with the best **PUCT score**:

```
score = Q(s,a) + c_puct × P(s,a) × sqrt(N_parent) / (1 + N(s,a))
```

Where:
- `Q(s,a)` = average value of all simulations that went through this node (exploitation)
- `P(s,a)` = the neural network's prior probability for this move (guidance)
- `N_parent` = how many times we've visited the parent
- `N(s,a)` = how many times we've visited this child
- `c_puct` = exploration constant (default 1.5)

This formula balances:
- **Exploitation:** Visit moves that have given good results so far (`Q` is high)
- **Exploration:** Also try moves that the network likes (`P` is high) but haven't been visited much (`N` is low)

Early on, when `N(s,a)` is 0 for all children, the formula picks based on the network's prior `P(s,a)`. As we visit more, `Q(s,a)` takes over — actual simulated results matter more than the network's initial guess.

Keep walking down the tree using this formula until you reach a node that hasn't been expanded yet.

**Step 2: Expand**

At the unexplored leaf node:
1. Get all legal moves from this position
2. Run the neural network on this position → get `(policy, value)`
3. Create child nodes for each legal move, storing the network's prior probability `P`

**Step 3: Evaluate**

The neural network already gave us a value estimate `v` in the previous step. This tells us "how good is this position?" from -1 (losing) to +1 (winning).

**Step 4: Backpropagate**

Walk back up the tree to the root, updating every node along the path:
- Increment visit count `N(s,a) += 1`
- Add the value to the running sum: `value_sum += v`
- The average `Q(s,a) = value_sum / N(s,a)` automatically updates

After doing this, the tree has slightly better statistics, and the next simulation will be slightly better informed.

### After all simulations

We've done (say) 100 simulations. The root node now has children with visit counts like:

```
move A: visited 45 times, avg value = 0.3
move B: visited 40 times, avg value = 0.2
move C: visited 15 times, avg value = -0.1
```

The final action is chosen based on **visit counts** (not values directly):
- **For training:** Sample proportionally to visit counts (with temperature). This adds exploration.
- **For competition:** Pick the most-visited move (greedy). This is the strongest play.

The visit-count distribution also becomes the **policy target** for training the network. It's like saying "the network should learn to output something close to what MCTS found after careful search."

## Why MCTS makes training stronger

Without MCTS, the training loop is:
1. Network plays against itself
2. Network trains on its own games

The problem: if the network is bad, it generates bad games, and trains on bad data. It's hard to improve from garbage.

With MCTS:
1. Network + search plays against itself → much stronger games
2. Network trains on search-improved policy targets

Even if the network is mediocre, MCTS adds enough lookahead to produce reasonable play. The training targets (MCTS policies) are always better than the raw network output, so the network always has something to improve toward.

## Our implementation (`search/mcts.py`)

### Key classes

**`MCTSNode`**: A node in the search tree.
- Stores: parent, action that led here, prior probability, visit count, value sum
- Key method: `ucb_score()` — computes the PUCT score for child selection
- Key method: `best_action(temperature)` — returns the chosen move after search

**`MCTS`**: The search algorithm.
- Configurable: number of simulations, time limit, exploration constant, temperature
- Key method: `search(game, colour, model, encoder)` — runs the full MCTS and returns a policy

### Important implementation detail: cloning

MCTS needs to simulate moves without affecting the real game. Every simulation starts by **cloning** the game state. This is why `LocalGame.clone()` exists — it deep-copies the board, pins, and all state so simulations are independent.

### Time-limited vs count-limited

```python
# Fixed number of simulations
mcts = MCTS(num_simulations=100)

# Time-limited (for competition)
mcts = MCTS(time_limit=2.0)  # stop after 2 seconds
```

In competition, you want to use as much of the time budget as safely possible. Time-limited mode automatically does as many simulations as fit within the budget.

### Temperature

Controls how "random" the action selection is:
- **temperature = 1.0**: Sample proportionally to visit counts. Used in training for exploration.
- **temperature = 0.1**: Nearly greedy — strongly prefer the most-visited move. Used in competition.
- **temperature = 0.0**: Pure greedy — always pick the most-visited move.

## Computational cost

MCTS is expensive. Each simulation requires:
1. Cloning the game state
2. Walking down the tree (fast)
3. One neural network forward pass (the bottleneck)
4. Backpropagation through the tree (fast)

With 100 simulations per move and ~150 moves per game, that's ~15,000 network evaluations per game. This is why:
- Training with MCTS is much slower than without
- GPU acceleration helps a lot (speeds up the network forward passes)
- The "bootstrap then MCTS" training strategy makes sense — learn the basics fast, then refine with search
