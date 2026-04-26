"""
Monte Carlo Tree Search (MCTS) with neural network guidance.

PUCT-based search using the policy-value network for:
- Prior probabilities from the policy head
- Leaf evaluation from the value head
- Legal-action-only expansion

Compatible with configurable simulation budgets and time limits.
"""

import math
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.local_game import LocalGame
from env.action_mapping import (
    ACTION_SPACE_SIZE, build_legal_mask, action_to_flat, flat_to_action
)
from models.encoders import BoardEncoder


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = [
        'parent', 'action', 'prior', 'colour',
        'children', 'visit_count', 'value_sum',
        'is_expanded', 'is_terminal',
    ]

    def __init__(
        self,
        parent: Optional['MCTSNode'] = None,
        action: int = -1,
        prior: float = 0.0,
        colour: str = "",
    ):
        self.parent = parent
        self.action = action       # flat action index that led to this node
        self.prior = prior         # P(s, a) from network
        self.colour = colour       # colour that played the action to reach this node

        self.children: Dict[int, 'MCTSNode'] = {}  # action -> child node
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.is_expanded: bool = False
        self.is_terminal: bool = False

    @property
    def q_value(self) -> float:
        """Mean action value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        """
        PUCT score: Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))
        """
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration

    def select_child(self, c_puct: float) -> 'MCTSNode':
        """Select the child with highest UCB score."""
        best_score = -float('inf')
        best_child = None
        for child in self.children.values():
            score = child.ucb_score(self.visit_count, c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def best_action(self, temperature: float = 0.0) -> Tuple[int, np.ndarray]:
        """
        Select the best action based on visit counts.

        Args:
            temperature: 0 = greedy (most visited), >0 = proportional

        Returns:
            (action_flat_idx, policy_distribution)
        """
        actions = list(self.children.keys())
        visits = np.array([self.children[a].visit_count for a in actions], dtype=np.float64)

        # Build full policy vector
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)

        if temperature < 0.01:
            # Greedy
            best_idx = np.argmax(visits)
            best_action = actions[best_idx]
            policy[best_action] = 1.0
            return best_action, policy
        else:
            # Temperature-scaled visit counts
            visits_temp = visits ** (1.0 / temperature)
            total = visits_temp.sum()
            if total == 0:
                probs = np.ones(len(actions)) / len(actions)
            else:
                probs = visits_temp / total

            for i, a in enumerate(actions):
                policy[a] = probs[i]

            chosen_idx = np.random.choice(len(actions), p=probs)
            return actions[chosen_idx], policy


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.

    Usage:
        mcts = MCTS(c_puct=1.5, num_simulations=100)
        policy = mcts.search(game, colour, model, encoder)
    """

    def __init__(
        self,
        c_puct: float = 1.5,
        num_simulations: int = 100,
        temperature: float = 1.0,
        time_limit: Optional[float] = None,
        device: str = "cpu",
        dirichlet_alpha: float = 0.0,
        root_noise_epsilon: float = 0.0,
    ):
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.time_limit = time_limit  # seconds, overrides num_simulations if set
        self.device = device
        # AlphaZero-style root exploration noise. Both must be > 0 to fire.
        # α controls noise sharpness (smaller = more concentrated); ε is the
        # mixing weight against the network prior.
        self.dirichlet_alpha = dirichlet_alpha
        self.root_noise_epsilon = root_noise_epsilon

    def search(
        self,
        game: LocalGame,
        colour: str,
        model,
        encoder: BoardEncoder,
        num_simulations: Optional[int] = None,
    ) -> np.ndarray:
        """
        Run MCTS from the current game state.

        Args:
            game: current game state (will not be modified)
            colour: the colour to play
            model: policy-value network
            encoder: board state encoder
            num_simulations: override default simulation count

        Returns:
            policy: (ACTION_SPACE_SIZE,) probability distribution from visit counts
        """
        n_sims = num_simulations or self.num_simulations
        root = MCTSNode(colour=colour)

        # Expand root
        self._expand(root, game, colour, model, encoder)

        if not root.children:
            # No legal moves
            return np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)

        # Root-only Dirichlet exploration noise. Mixing happens after expansion
        # so the noise rides on the legal-action subset only.
        if self.dirichlet_alpha > 0 and self.root_noise_epsilon > 0:
            actions = list(root.children.keys())
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
            eps = self.root_noise_epsilon
            for i, a in enumerate(actions):
                child = root.children[a]
                child.prior = (1.0 - eps) * child.prior + eps * float(noise[i])

        start_time = time.time()
        sims_done = 0

        while True:
            # Check termination
            if self.time_limit is not None:
                if time.time() - start_time >= self.time_limit:
                    break
            else:
                if sims_done >= n_sims:
                    break

            # 1. Select: traverse tree using UCB
            node = root
            sim_game = game.clone()
            search_path = [node]

            while node.is_expanded and not node.is_terminal and node.children:
                node = node.select_child(self.c_puct)
                search_path.append(node)

                # Apply the action to the simulation game
                if node.action >= 0:
                    pin_id, to_index = flat_to_action(node.action)
                    try:
                        _, done, info = sim_game.step(pin_id, to_index)
                        if done:
                            node.is_terminal = True
                    except (ValueError, RuntimeError):
                        node.is_terminal = True
                        break

            # 2. Expand and evaluate
            if not node.is_terminal and not node.is_expanded:
                current_colour = sim_game.current_colour() if not sim_game.done else colour
                value = self._expand(node, sim_game, current_colour, model, encoder)
            elif node.is_terminal:
                # Terminal value
                if sim_game.winner == colour:
                    value = 1.0
                elif sim_game.winner is not None:
                    value = -1.0
                else:
                    value = 0.0
            else:
                # Already expanded leaf with no children
                value = 0.0

            # 3. Backpropagate
            # Value is from the perspective of the colour that just played
            self._backpropagate(search_path, value, colour)

            sims_done += 1

        # Get policy from visit counts
        _, policy = root.best_action(temperature=self.temperature)
        return policy

    def _expand(
        self,
        node: MCTSNode,
        game: LocalGame,
        colour: str,
        model,
        encoder: BoardEncoder,
    ) -> float:
        """
        Expand a leaf node.

        Returns the value estimate for the position.
        """
        node.is_expanded = True

        if game.done:
            node.is_terminal = True
            if game.winner == colour:
                return 1.0
            elif game.winner is not None:
                return -1.0
            return 0.0

        current_colour = game.current_colour()

        # Get legal moves
        legal = game.get_legal_moves(current_colour)
        mask = build_legal_mask(legal)

        if not mask.any():
            node.is_terminal = True
            return 0.0

        # Neural network evaluation
        spatial, scalars = encoder.encode_from_game(game, current_colour)
        probs, value = model.predict(
            spatial.to(self.device),
            scalars.to(self.device),
            mask.to(self.device),
        )
        probs = probs.cpu().numpy()

        # Create children for legal actions
        for pid, dests in legal.items():
            for dest in dests:
                flat_idx = action_to_flat(pid, dest)
                child = MCTSNode(
                    parent=node,
                    action=flat_idx,
                    prior=probs[flat_idx],
                    colour=current_colour,
                )
                node.children[flat_idx] = child

        # Return value from current colour's perspective
        # If current colour == our colour, value is positive for good positions
        # If current colour != our colour, flip the sign
        return value if current_colour == colour else -value

    def _backpropagate(
        self,
        search_path: List[MCTSNode],
        value: float,
        root_colour: str,
    ):
        """Backpropagate value up the tree."""
        for node in reversed(search_path):
            node.visit_count += 1
            # Value is from root_colour's perspective
            # If the node's colour matches root_colour, add value directly
            # Otherwise, add negated value
            if node.colour == root_colour or node.colour == "":
                node.value_sum += value
            else:
                node.value_sum -= value
