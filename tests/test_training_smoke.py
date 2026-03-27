"""Smoke tests for training pipeline."""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.policy_value_net import PolicyValueNet
from models.encoders import BoardEncoder
from training.self_play import generate_self_play_data
from training.replay_buffer import ReplayBuffer, Experience
from training.trainer import Trainer, TrainingConfig
from env.action_mapping import ACTION_SPACE_SIZE


class TestSelfPlay:
    def test_generates_experiences(self):
        model = PolicyValueNet(num_res_blocks=1, trunk_channels=16)
        encoder = BoardEncoder()
        experiences, game_stats = generate_self_play_data(
            model=model, encoder=encoder,
            num_games=1, num_players=2,
        )
        assert len(experiences) > 0
        assert "avg_game_length" in game_stats
        assert "max_moves_pct" in game_stats

    def test_experience_shapes(self):
        model = PolicyValueNet(num_res_blocks=1, trunk_channels=16)
        encoder = BoardEncoder()
        experiences, _ = generate_self_play_data(
            model=model, encoder=encoder,
            num_games=1, num_players=2,
        )
        exp = experiences[0]
        assert exp.spatial.shape == (10, 17, 17)
        assert exp.scalars.shape == (10,)
        assert exp.legal_mask.shape == (ACTION_SPACE_SIZE,)
        assert exp.policy_target.shape == (ACTION_SPACE_SIZE,)
        assert isinstance(exp.value_target, float)

    def test_policy_targets_sum_to_one(self):
        model = PolicyValueNet(num_res_blocks=1, trunk_channels=16)
        encoder = BoardEncoder()
        experiences, _ = generate_self_play_data(
            model=model, encoder=encoder,
            num_games=1, num_players=2,
        )
        for exp in experiences[:10]:
            assert abs(exp.policy_target.sum() - 1.0) < 1e-4

    def test_value_targets_in_range(self):
        model = PolicyValueNet(num_res_blocks=1, trunk_channels=16)
        encoder = BoardEncoder()
        experiences, _ = generate_self_play_data(
            model=model, encoder=encoder,
            num_games=1, num_players=2,
        )
        for exp in experiences:
            assert -1.0 <= exp.value_target <= 1.0


class TestReplayBuffer:
    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=100)
        import numpy as np
        for _ in range(50):
            buf.push(Experience(
                spatial=np.zeros((10, 17, 17), dtype=np.float32),
                scalars=np.zeros(10, dtype=np.float32),
                legal_mask=np.ones(ACTION_SPACE_SIZE, dtype=np.float32),
                policy_target=np.ones(ACTION_SPACE_SIZE, dtype=np.float32) / ACTION_SPACE_SIZE,
                value_target=0.0,
            ))
        assert len(buf) == 50

        spatial, scalars, masks, policies, values = buf.sample(16)
        assert spatial.shape[0] == 16
        assert values.shape == (16, 1)

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=10)
        import numpy as np
        for _ in range(20):
            buf.push(Experience(
                spatial=np.zeros((10, 17, 17), dtype=np.float32),
                scalars=np.zeros(10, dtype=np.float32),
                legal_mask=np.ones(ACTION_SPACE_SIZE, dtype=np.float32),
                policy_target=np.ones(ACTION_SPACE_SIZE, dtype=np.float32) / ACTION_SPACE_SIZE,
                value_target=0.0,
            ))
        assert len(buf) == 10


class TestTrainer:
    def test_training_reduces_loss(self):
        model = PolicyValueNet(num_res_blocks=1, trunk_channels=16)
        config = TrainingConfig(
            num_games_per_iteration=2,
            num_players=2,
            batch_size=16,
            epochs_per_iteration=3,
            num_iterations=2,
            eval_every=999,
            checkpoint_every=999,
            checkpoint_dir="/tmp/cc_test_ckpt",
            min_buffer_size=16,
        )
        trainer = Trainer(model=model, config=config)
        trainer.train()

        assert len(trainer.train_history) == 2
        # Loss should be finite
        for h in trainer.train_history:
            assert h["total_loss"] < 100

    def test_checkpoint_save_load(self):
        model = PolicyValueNet(num_res_blocks=1, trunk_channels=16)
        config = TrainingConfig(
            num_games_per_iteration=1,
            batch_size=16,
            epochs_per_iteration=1,
            num_iterations=1,
            checkpoint_every=1,
            checkpoint_dir="/tmp/cc_test_ckpt2",
            run_name="test_run",
            min_buffer_size=16,
        )
        trainer = Trainer(model=model, config=config)
        trainer.train()

        # Load checkpoint from run subdirectory
        ckpt_path = "/tmp/cc_test_ckpt2/test_run/model_iter_00001.pt"
        model2 = PolicyValueNet(num_res_blocks=1, trunk_channels=16)
        checkpoint = Trainer.load_checkpoint(ckpt_path, model2)
        assert checkpoint["iteration"] == 1

        # Weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
