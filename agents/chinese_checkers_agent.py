"""
Main RL agent for Chinese Checkers competition.

Combines the policy-value network with MCTS for strong play.
Supports both local game interface and server RPC interface.
"""

import sys
import os
import torch
import numpy as np
from typing import Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.policy_value_net import PolicyValueNet
from models.encoders import BoardEncoder
from search.mcts import MCTS
from env.local_game import LocalGame
from env.action_mapping import build_legal_mask, flat_to_action, ACTION_SPACE_SIZE


class ChineseCheckersAgent:
    """
    The main RL agent for competition play.

    Modes:
    - Training: uses MCTS with exploration (temperature > 0)
    - Evaluation: uses MCTS with low temperature (near-greedy)
    - Competition: time-aware MCTS with deterministic play
    """

    def __init__(
        self,
        model: Optional[PolicyValueNet] = None,
        checkpoint_path: Optional[str] = None,
        num_res_blocks: int = 6,
        trunk_channels: int = 128,
        mcts_simulations: int = 100,
        c_puct: float = 1.5,
        temperature: float = 0.1,
        time_limit: Optional[float] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.encoder = BoardEncoder()

        # Load or create model
        if model is not None:
            self.model = model
        else:
            self.model = PolicyValueNet(
                num_res_blocks=num_res_blocks,
                trunk_channels=trunk_channels,
            )
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded checkpoint: {checkpoint_path}")

        self.model.to(device)
        self.model.eval()

        # MCTS
        self.mcts = MCTS(
            c_puct=c_puct,
            num_simulations=mcts_simulations,
            temperature=temperature,
            time_limit=time_limit,
            device=device,
        )

    def select_action(self, game: LocalGame, colour: str) -> Tuple[int, int]:
        """
        Select an action for local game play.

        Returns:
            (pin_id, to_index)
        """
        policy = self.mcts.search(game, colour, self.model, self.encoder)

        if policy.sum() == 0:
            # Fallback: random legal move
            actions = game.get_all_legal_actions(colour)
            if actions:
                return actions[0]
            raise RuntimeError(f"No legal actions for {colour}")

        action_idx = int(policy.argmax())
        return flat_to_action(action_idx)

    def select_action_from_server_state(
        self,
        pin_positions: dict,
        legal_moves: dict,
        my_colour: str,
        turn_order: list,
        move_count: int,
        my_move_count: int = 0,
        time_remaining: Optional[float] = None,
        use_mcts: bool = True,
    ) -> Tuple[int, int]:
        """
        Select an action from server state (for competition play).

        Reconstructs a LocalGame from the server data so MCTS can
        simulate future moves. Falls back to direct network policy
        if use_mcts=False.

        Returns:
            (pin_id, to_index)
        """
        # Build legal mask
        mask = build_legal_mask(legal_moves)

        if not mask.any():
            raise RuntimeError("No legal moves available")

        if use_mcts:
            # Reconstruct a LocalGame so MCTS can clone and simulate
            game = LocalGame.from_server_state(
                pin_positions=pin_positions,
                turn_order=turn_order,
                current_turn_colour=my_colour,
                move_count=move_count,
            )
            policy = self.mcts.search(game, my_colour, self.model, self.encoder)

            if policy.sum() > 0:
                action_idx = int(policy.argmax())
                return flat_to_action(action_idx)
            # Fall through to direct policy if MCTS returned nothing

        # Direct network policy (fast fallback)
        total_actions = sum(len(v) for v in legal_moves.values())
        spatial, scalars = self.encoder.encode(
            my_colour=my_colour,
            pin_positions=pin_positions,
            turn_order=turn_order,
            move_count=move_count,
            total_legal_actions=total_actions,
            num_active_players=len(turn_order),
            my_move_count=my_move_count,
        )

        probs, value = self.model.predict(
            spatial.to(self.device),
            scalars.to(self.device),
            mask.to(self.device),
        )

        probs = probs.cpu().numpy()
        action_idx = int(probs.argmax())
        return flat_to_action(action_idx)
