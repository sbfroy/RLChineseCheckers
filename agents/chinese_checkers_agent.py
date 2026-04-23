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
from env.colour_symmetry import (
    canonicalize_positions,
    canonicalize_legal_moves,
    canonicalize_turn_order,
    decanonicalize_to_index,
    decanonicalize_pin_id,
)


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

        If the MCTS temperature is >0, samples from the visit-count policy
        (gives real variance across games vs deterministic opponents).
        If temperature is ~0, takes the argmax (deterministic, for
        competition play).
        """
        policy = self.mcts.search(game, colour, self.model, self.encoder)

        if policy.sum() == 0:
            # Fallback: random legal move
            actions = game.get_all_legal_actions(colour)
            if actions:
                return actions[0]
            raise RuntimeError(f"No legal actions for {colour}")

        if self.mcts.temperature < 0.01:
            action_idx = int(policy.argmax())
        else:
            probs = policy / policy.sum()
            action_idx = int(np.random.choice(len(probs), p=probs))
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
        # Canonicalize: rotate the board so my_colour looks like red. Models
        # trained only on red/blue 2P games then see a familiar orientation
        # regardless of which seat the server assigned us. The chosen
        # to_index is rotated back to the original frame before returning.
        canon_positions = canonicalize_positions(pin_positions, my_colour)
        canon_legal_moves = canonicalize_legal_moves(legal_moves, my_colour)
        canon_turn_order = canonicalize_turn_order(turn_order, my_colour)
        canon_colour = 'red'

        # Build legal mask (in canonical frame)
        mask = build_legal_mask(canon_legal_moves)

        if not mask.any():
            raise RuntimeError("No legal moves available")

        if use_mcts:
            game = LocalGame.from_server_state(
                pin_positions=canon_positions,
                turn_order=canon_turn_order,
                current_turn_colour=canon_colour,
                move_count=move_count,
            )
            policy = self.mcts.search(game, canon_colour, self.model, self.encoder)

            if policy.sum() > 0:
                if self.mcts.temperature < 0.01:
                    action_idx = int(policy.argmax())
                else:
                    probs = policy / policy.sum()
                    action_idx = int(np.random.choice(len(probs), p=probs))
                canon_pin_id, canon_to_index = flat_to_action(action_idx)
                return (
                    decanonicalize_pin_id(canon_pin_id, my_colour),
                    decanonicalize_to_index(canon_to_index, my_colour),
                )
            # Fall through to direct policy if MCTS returned nothing

        # Direct network policy (fast fallback)
        total_actions = sum(len(v) for v in canon_legal_moves.values())
        spatial, scalars = self.encoder.encode(
            my_colour=canon_colour,
            pin_positions=canon_positions,
            turn_order=canon_turn_order,
            move_count=move_count,
            total_legal_actions=total_actions,
            num_active_players=len(canon_turn_order),
            my_move_count=my_move_count,
        )

        probs, value = self.model.predict(
            spatial.to(self.device),
            scalars.to(self.device),
            mask.to(self.device),
        )

        probs = probs.cpu().numpy()
        action_idx = int(probs.argmax())
        canon_pin_id, canon_to_index = flat_to_action(action_idx)
        return (
            decanonicalize_pin_id(canon_pin_id, my_colour),
            decanonicalize_to_index(canon_to_index, my_colour),
        )
