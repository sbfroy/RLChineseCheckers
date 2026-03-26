"""
Random baseline agent: picks a uniformly random legal action each turn.
"""

import random
from typing import Tuple
from env.local_game import LocalGame


class RandomAgent:
    """Plays uniformly random legal moves."""

    def select_action(self, game: LocalGame, colour: str) -> Tuple[int, int]:
        """
        Select a random legal action.

        Returns:
            (pin_id, to_index)
        """
        actions = game.get_all_legal_actions(colour)
        if not actions:
            raise RuntimeError(f"No legal actions for {colour}")
        return random.choice(actions)
