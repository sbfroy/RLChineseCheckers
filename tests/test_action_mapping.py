"""Tests for action mapping between (pin_id, to_index) and flat indices."""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.action_mapping import (
    action_to_flat, flat_to_action, build_legal_mask,
    legal_actions_from_mask, masked_softmax, ACTION_SPACE_SIZE,
    NUM_PINS, NUM_CELLS,
)
from env.local_game import LocalGame


class TestActionMapping:
    def test_round_trip(self):
        for pid in range(NUM_PINS):
            for cell in [0, 60, 120]:
                flat = action_to_flat(pid, cell)
                p, c = flat_to_action(flat)
                assert p == pid and c == cell

    def test_flat_range(self):
        assert action_to_flat(0, 0) == 0
        assert action_to_flat(9, 120) == ACTION_SPACE_SIZE - 1

    def test_action_space_size(self):
        assert ACTION_SPACE_SIZE == NUM_PINS * NUM_CELLS

    def test_legal_mask_from_game(self):
        game = LocalGame(num_players=2)
        game.reset()
        legal = game.get_legal_moves()
        mask = build_legal_mask(legal)

        assert mask.shape == (ACTION_SPACE_SIZE,)
        assert mask.dtype == torch.bool
        assert mask.any()

        # Verify mask matches legal moves
        for pid, dests in legal.items():
            for d in dests:
                flat = action_to_flat(pid, d)
                assert mask[flat], f"Legal move (pin={pid}, dest={d}) not in mask"

    def test_legal_mask_round_trip(self):
        game = LocalGame(num_players=2)
        game.reset()
        legal = game.get_legal_moves()
        mask = build_legal_mask(legal)
        actions = legal_actions_from_mask(mask)
        expected = game.get_all_legal_actions()
        assert set(actions) == set(expected)

    def test_masked_softmax(self):
        mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)
        mask[0] = True
        mask[100] = True
        mask[500] = True

        logits = torch.randn(ACTION_SPACE_SIZE)
        probs = masked_softmax(logits, mask)

        assert abs(probs.sum().item() - 1.0) < 1e-5
        assert (probs[~mask] < 1e-7).all()

    def test_masked_softmax_batch(self):
        B = 4
        mask = torch.zeros(B, ACTION_SPACE_SIZE, dtype=torch.bool)
        mask[:, 0] = True
        mask[:, 50] = True

        logits = torch.randn(B, ACTION_SPACE_SIZE)
        probs = masked_softmax(logits, mask)

        for b in range(B):
            assert abs(probs[b].sum().item() - 1.0) < 1e-5

    def test_mask_never_allows_illegal(self):
        """No illegal action should be selectable via the mask."""
        game = LocalGame(num_players=2)
        game.reset()
        legal = game.get_legal_moves()
        mask = build_legal_mask(legal)
        all_legal = game.get_all_legal_actions()
        legal_set = set(all_legal)

        for i in range(ACTION_SPACE_SIZE):
            if mask[i]:
                pid, dest = flat_to_action(i)
                assert (pid, dest) in legal_set
