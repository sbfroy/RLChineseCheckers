"""
Self-play game generation for training data.

Plays games using the current model (with optional MCTS),
collects (state, policy, value) tuples for training.
"""

import sys
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.local_game import LocalGame
from env.action_mapping import (
    ACTION_SPACE_SIZE, build_legal_mask, action_to_flat, flat_to_action
)
from env.rewards import RewardShaper, RewardConfig
from models.encoders import BoardEncoder
from models.policy_value_net import PolicyValueNet
from training.replay_buffer import Experience, ReplayBuffer


class SelfPlayWorker:
    """
    Generates training data through self-play or vs external opponent.

    Each game produces experiences for the RL agent (all players in
    self-play mode, or only the RL agent when an opponent is set).
    Supports both pure-network play and MCTS-enhanced play.
    """

    def __init__(
        self,
        model: PolicyValueNet,
        encoder: BoardEncoder,
        reward_config: Optional[RewardConfig] = None,
        num_players: int = 2,
        temperature: float = 1.0,
        temperature_drop_move: int = 20,
        mcts=None,
        mcts_simulations: int = 50,
        device: str = "cpu",
        opponent=None,
    ):
        self.model = model
        self.encoder = encoder
        self.reward_shaper = RewardShaper(reward_config)
        self.num_players = num_players
        self.temperature = temperature
        self.temperature_drop_move = temperature_drop_move
        self.mcts = mcts
        self.mcts_simulations = mcts_simulations
        self.device = device
        self.opponent = opponent
        self._last_game_length = None
        self._last_game_hit_max = False

    def play_game(self) -> List[Experience]:
        """
        Play one complete game and return experiences.

        In self-play mode (no opponent), collects data for all players.
        With an opponent, collects data only for the RL agent.

        Returns list of Experience objects.
        """
        game = LocalGame(num_players=self.num_players)
        game.reset()

        # Determine which colour(s) the RL agent plays
        rl_colour = game.turn_order[0] if self.opponent is not None else None

        # Collect data per colour: list of (spatial, scalars, mask, policy, reward_so_far)
        game_data: Dict[str, List] = {c: [] for c in game.colours}
        step_rewards: Dict[str, List[float]] = {c: [] for c in game.colours}

        move_num = 0
        while not game.done:
            colour = game.current_colour()

            # Opponent's turn — use external agent, no data collection
            if self.opponent is not None and colour != rl_colour:
                try:
                    pin_id, to_index = self.opponent.select_action(game, colour)
                    state, done, info = game.step(pin_id, to_index)
                except Exception:
                    break
                move_num += 1
                continue

            # RL agent's turn — full pipeline
            # Encode state
            spatial, scalars = self.encoder.encode_from_game(game, colour)
            legal = game.get_legal_moves(colour)
            mask = build_legal_mask(legal)

            if not mask.any():
                break

            # Get policy (from MCTS or raw network)
            if self.mcts is not None:
                policy = self.mcts.search(game, colour, self.model, self.encoder,
                                          num_simulations=self.mcts_simulations)
            else:
                policy = self._get_network_policy(spatial, scalars, mask)

            # Select action with temperature
            temp = self.temperature if move_num < self.temperature_drop_move else 0.1
            action_idx = self._sample_action(policy, mask, temp)
            pin_id, to_index = flat_to_action(action_idx)

            # Compute distance info before move
            info_before = game.compute_distance_info(colour)

            # Execute move
            state, done, info = game.step(pin_id, to_index)

            # Compute distance info after move
            info_after = game.compute_distance_info(colour)

            # Compute shaped reward
            reward = self.reward_shaper.compute_reward(
                info_before, info_after, done,
                info.get("status", "CONTINUE"),
                info.get("scores"), colour
            )

            # Store experience data (value target filled in later)
            game_data[colour].append({
                "spatial": spatial.numpy(),
                "scalars": scalars.numpy(),
                "mask": mask.numpy(),
                "policy": policy,
            })
            step_rewards[colour].append(reward)

            move_num += 1

        # Track game stats
        self._last_game_length = move_num
        self._last_game_hit_max = (game.move_count >= game.max_moves)

        # Compute value targets: discounted sum of future rewards per player
        experiences = []
        gamma = 0.99

        # Absolute-score terminal value: uses the agent's own competition
        # score normalized by max possible score. Unlike score margin,
        # this always provides meaningful signal even in self-play, and
        # creates cross-game variance when playing against diverse opponents.
        final_scores = game.compute_scores() if game.status == "FINISHED" else None

        for colour in game.colours:
            # Skip opponent's data when using external opponent
            if self.opponent is not None and colour != rl_colour:
                continue

            data = game_data[colour]
            rewards = step_rewards[colour]

            if not data:
                continue

            # Compute returns (reverse cumulative discounted reward)
            returns = []
            G = 0.0

            # Terminal value: either score-margin (Phase 1, when config
            # enables it and the RL agent's score can differ meaningfully
            # from the opponent's) or absolute score (Phase 0-style).
            # Margin gives real variance when both sides play well; it
            # collapses to ~0 in symmetric self-play, so absolute-score
            # remains the default.
            if final_scores and colour in final_scores:
                norm = self.reward_shaper.config.score_normalization
                my_score = final_scores[colour]["final_score"]
                if self.reward_shaper.config.use_score_margin:
                    opp_scores = [
                        s["final_score"]
                        for c, s in final_scores.items() if c != colour
                    ]
                    opp_mean = (sum(opp_scores) / len(opp_scores)
                                if opp_scores else 0.0)
                    G = (my_score - opp_mean) / norm
                else:
                    G = my_score / norm
                G = max(-1.0, min(1.0, G))
            elif game.winner == colour:
                G = self.reward_shaper.config.win_reward
            elif game.winner is not None:
                G = self.reward_shaper.config.loss_reward
            elif game.status == "FINISHED":
                G = self.reward_shaper.config.draw_reward

            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            # No per-game normalization — we want cross-game variance
            # preserved so the value head learns consistent position
            # evaluations. Reward weights are sized so returns naturally
            # fall near [-1, 1]; clip catches outliers.

            for i, d in enumerate(data):
                experiences.append(Experience(
                    spatial=d["spatial"],
                    scalars=d["scalars"],
                    legal_mask=d["mask"],
                    policy_target=d["policy"],
                    value_target=np.clip(returns[i], -1.0, 1.0),
                ))

        return experiences

    def _get_network_policy(
        self, spatial: torch.Tensor, scalars: torch.Tensor, mask: torch.Tensor
    ) -> np.ndarray:
        """Get policy from raw network output."""
        probs, _ = self.model.predict(spatial.to(self.device), scalars.to(self.device), mask.to(self.device))
        return probs.cpu().numpy()

    def _sample_action(
        self, policy: np.ndarray, mask: torch.Tensor, temperature: float
    ) -> int:
        """Sample an action from the policy with temperature."""
        if temperature < 0.01:
            # Greedy
            legal_indices = mask.nonzero(as_tuple=False).squeeze(-1).numpy()
            best = legal_indices[policy[legal_indices].argmax()]
            return int(best)

        # Apply temperature
        legal_indices = mask.nonzero(as_tuple=False).squeeze(-1).numpy()
        probs = policy[legal_indices]

        # Temperature scaling on log-probs
        probs = np.maximum(probs, 1e-8)
        log_probs = np.log(probs) / temperature
        log_probs -= log_probs.max()
        probs = np.exp(log_probs)
        probs /= probs.sum()

        chosen = np.random.choice(legal_indices, p=probs)
        return int(chosen)


def generate_self_play_data(
    model: PolicyValueNet,
    encoder: BoardEncoder,
    num_games: int = 10,
    num_players: int = 2,
    temperature: float = 1.0,
    mcts=None,
    mcts_simulations: int = 50,
    reward_config: Optional[RewardConfig] = None,
    device: str = "cpu",
    opponent=None,
) -> Tuple[List[Experience], Dict]:
    """
    Generate training data from self-play or vs-opponent games.

    When opponent is set, the RL agent plays one colour and the opponent
    plays the other. Only the RL agent's experiences are collected.

    Returns (experiences, game_stats) where game_stats contains
    aggregate info about the games played.
    """
    worker = SelfPlayWorker(
        model=model,
        encoder=encoder,
        reward_config=reward_config,
        num_players=num_players,
        temperature=temperature,
        mcts=mcts,
        mcts_simulations=mcts_simulations,
        device=device,
        opponent=opponent,
    )

    all_experiences = []
    game_lengths = []
    max_moves_games = 0
    for i in range(num_games):
        exps = worker.play_game()
        all_experiences.extend(exps)
        if worker._last_game_length is not None:
            game_lengths.append(worker._last_game_length)
            if worker._last_game_hit_max:
                max_moves_games += 1

    game_stats = {
        "avg_game_length": round(sum(game_lengths) / max(len(game_lengths), 1), 1),
        "max_moves_pct": round(max_moves_games / max(num_games, 1), 3),
        "total_experiences": len(all_experiences),
    }

    return all_experiences, game_stats
