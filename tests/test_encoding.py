"""Tests for state encoding."""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.encoders import (
    BoardEncoder, NUM_SPATIAL_CHANNELS, NUM_SCALAR_FEATURES, GRID_SIZE
)
from env.local_game import LocalGame


class TestBoardEncoder:
    def setup_method(self):
        self.encoder = BoardEncoder()
        self.game = LocalGame(num_players=2)
        self.game.reset()

    def test_output_shapes(self):
        spatial, scalars = self.encoder.encode_from_game(self.game, 'red')
        assert spatial.shape == (NUM_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE)
        assert scalars.shape == (NUM_SCALAR_FEATURES,)

    def test_my_pieces_channel(self):
        spatial, _ = self.encoder.encode_from_game(self.game, 'red')
        assert spatial[0].sum().item() == 10  # 10 pins

    def test_opponent_channel(self):
        spatial, _ = self.encoder.encode_from_game(self.game, 'red')
        assert spatial[1].sum().item() == 10  # opponent's 10 pins

    def test_valid_mask_channel(self):
        spatial, _ = self.encoder.encode_from_game(self.game, 'red')
        assert spatial[8].sum().item() == 121  # all valid cells

    def test_board_only_channel(self):
        spatial, _ = self.encoder.encode_from_game(self.game, 'red')
        assert spatial[9].sum().item() == 61  # 61 board-only cells

    def test_target_camp_channel(self):
        spatial, _ = self.encoder.encode_from_game(self.game, 'red')
        assert spatial[6].sum().item() == 10  # blue zone (red's target)

    def test_home_camp_channel(self):
        spatial, _ = self.encoder.encode_from_game(self.game, 'red')
        assert spatial[7].sum().item() == 10  # red zone (red's home)

    def test_perspective_correct(self):
        """Encoding for red vs blue should have my/opp channels swapped."""
        s_red, _ = self.encoder.encode_from_game(self.game, 'red')
        s_blue, _ = self.encoder.encode_from_game(self.game, 'blue')

        # Red's pieces (ch0 for red) should equal Blue's opponent (ch1 for blue)
        assert torch.allclose(s_red[0], s_blue[1])
        assert torch.allclose(s_red[1], s_blue[0])

    def test_6_player_encoding(self):
        game6 = LocalGame(num_players=6)
        game6.reset()
        spatial, scalars = self.encoder.encode_from_game(game6, game6.current_colour())
        # All 5 opponent channels should have 10 pieces each
        for ch in range(1, 6):
            assert spatial[ch].sum().item() == 10

    def test_scalars_range(self):
        _, scalars = self.encoder.encode_from_game(self.game, 'red')
        # All scalars should be normalized to [0, ~2] range
        assert scalars.min() >= 0.0
        assert scalars.max() <= 2.0

    def test_encoding_dtype(self):
        spatial, scalars = self.encoder.encode_from_game(self.game, 'red')
        assert spatial.dtype == torch.float32
        assert scalars.dtype == torch.float32
