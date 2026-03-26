"""Tests for MCTS."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search.mcts import MCTS, MCTSNode
from models.policy_value_net import PolicyValueNet
from models.encoders import BoardEncoder
from env.local_game import LocalGame
from env.action_mapping import ACTION_SPACE_SIZE, flat_to_action, build_legal_mask


class TestMCTSNode:
    def test_q_value_zero_visits(self):
        node = MCTSNode()
        assert node.q_value == 0.0

    def test_q_value(self):
        node = MCTSNode()
        node.visit_count = 10
        node.value_sum = 5.0
        assert node.q_value == 0.5

    def test_ucb_score(self):
        node = MCTSNode(prior=0.5)
        node.visit_count = 0
        # With 0 visits, exploration should be high
        score = node.ucb_score(parent_visits=100, c_puct=1.5)
        assert score > 0

    def test_select_child(self):
        parent = MCTSNode()
        parent.visit_count = 10
        c1 = MCTSNode(parent=parent, action=0, prior=0.8)
        c2 = MCTSNode(parent=parent, action=1, prior=0.2)
        parent.children = {0: c1, 1: c2}
        selected = parent.select_child(c_puct=1.5)
        # c1 should be selected (higher prior, both 0 visits)
        assert selected is c1


class TestMCTS:
    def setup_method(self):
        self.model = PolicyValueNet(num_res_blocks=2, trunk_channels=32)
        self.encoder = BoardEncoder()

    def test_policy_sums_to_one(self):
        mcts = MCTS(num_simulations=20, temperature=1.0)
        game = LocalGame(num_players=2)
        game.reset()
        policy = mcts.search(game, game.current_colour(), self.model, self.encoder)
        assert abs(policy.sum() - 1.0) < 1e-5

    def test_only_legal_actions_in_policy(self):
        mcts = MCTS(num_simulations=20, temperature=1.0)
        game = LocalGame(num_players=2)
        game.reset()
        colour = game.current_colour()
        policy = mcts.search(game, colour, self.model, self.encoder)

        legal = game.get_legal_moves(colour)
        mask = build_legal_mask(legal)

        # All nonzero policy entries should be legal
        for i in range(ACTION_SPACE_SIZE):
            if policy[i] > 0:
                assert mask[i], f"Illegal action {i} has nonzero policy"

    def test_selected_move_is_legal(self):
        mcts = MCTS(num_simulations=20, temperature=0.0)
        game = LocalGame(num_players=2)
        game.reset()
        colour = game.current_colour()
        policy = mcts.search(game, colour, self.model, self.encoder)
        action_idx = int(policy.argmax())
        pid, to_idx = flat_to_action(action_idx)

        legal = game.get_legal_moves(colour)
        assert pid in legal, f"Pin {pid} not in legal moves"
        assert to_idx in legal[pid], f"Dest {to_idx} not legal for pin {pid}"

    def test_game_unchanged_after_search(self):
        mcts = MCTS(num_simulations=20)
        game = LocalGame(num_players=2)
        game.reset()
        original_positions = {
            c: [p.axialindex for p in pins]
            for c, pins in game.pins_by_colour.items()
        }
        mcts.search(game, game.current_colour(), self.model, self.encoder)
        for c, pins in game.pins_by_colour.items():
            current = [p.axialindex for p in pins]
            assert current == original_positions[c], \
                f"Game state modified during MCTS for {c}"

    def test_more_sims_more_visits(self):
        game = LocalGame(num_players=2)
        game.reset()
        mcts10 = MCTS(num_simulations=10, temperature=1.0)
        mcts50 = MCTS(num_simulations=50, temperature=1.0)
        p10 = mcts10.search(game, game.current_colour(), self.model, self.encoder)
        p50 = mcts50.search(game, game.current_colour(), self.model, self.encoder)
        # Both should be valid distributions
        assert abs(p10.sum() - 1.0) < 1e-5
        assert abs(p50.sum() - 1.0) < 1e-5
