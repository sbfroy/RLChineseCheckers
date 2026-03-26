"""
Board state encoder: game state -> tensor representation for the neural network.

Encodes the 121-cell hex board into a 17x17 grid with multiple channels,
plus scalar features appended to a separate vector.

Grid mapping: axial (q, r) -> grid (q+8, r+8) in a 17x17 array.
Only 121 of the 289 grid cells are valid; the rest are masked to 0.
"""

import sys
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

_ENGINE_DIR = os.path.join(os.path.dirname(__file__), "..", "multi system single machine minimal")
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

from checkers_board import HexBoard

# Board constants
GRID_SIZE = 17  # q and r range from -8 to +8
NUM_CELLS = 121
COLOUR_OPPOSITES = {
    'red': 'blue', 'blue': 'red',
    'lawn green': 'gray0', 'gray0': 'lawn green',
    'yellow': 'purple', 'purple': 'yellow',
}
ALL_COLOURS = ['red', 'blue', 'lawn green', 'gray0', 'yellow', 'purple']

# Number of spatial channels
# 0: my pieces
# 1-5: opponent pieces (by turn order, padded with zeros if fewer)
# 6: my target camp
# 7: my home camp
# 8: valid cell mask
# 9: board-only cells (not in any colour zone)
NUM_SPATIAL_CHANNELS = 10

# Number of scalar features
NUM_SCALAR_FEATURES = 10


class BoardEncoder:
    """
    Encodes Chinese Checkers board state into tensors for the neural network.

    Produces:
    - spatial: (NUM_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE) float tensor
    - scalars: (NUM_SCALAR_FEATURES,) float tensor
    """

    def __init__(self):
        # Build coordinate mappings once using a reference board
        ref_board = HexBoard()

        # Map cell index -> (grid_row, grid_col)
        self.cell_to_grid: Dict[int, Tuple[int, int]] = {}
        self.valid_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        for idx, cell in enumerate(ref_board.cells):
            gr = cell.r + 8  # row in grid
            gc = cell.q + 8  # col in grid
            self.cell_to_grid[idx] = (gr, gc)
            self.valid_mask[gr, gc] = 1.0

        # Precompute zone masks
        self.zone_masks: Dict[str, np.ndarray] = {}
        for colour in ALL_COLOURS:
            mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            for idx in ref_board.axial_of_colour(colour):
                gr, gc = self.cell_to_grid[idx]
                mask[gr, gc] = 1.0
            self.zone_masks[colour] = mask

        # Board-only mask (cells that are 'board' type, not any colour zone)
        self.board_only_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for idx, cell in enumerate(ref_board.cells):
            if cell.postype == 'board':
                gr, gc = self.cell_to_grid[idx]
                self.board_only_mask[gr, gc] = 1.0

        # Store reference board for distance computations
        self._ref_board = ref_board

    def encode(
        self,
        my_colour: str,
        pin_positions: Dict[str, List[int]],
        turn_order: List[str],
        move_count: int,
        total_legal_actions: int = 0,
        num_active_players: int = 2,
        my_move_count: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the board state.

        Args:
            my_colour: the agent's colour
            pin_positions: {colour: [cell_idx, ...]} for all players
            turn_order: list of colours in turn order
            move_count: total moves so far
            total_legal_actions: number of legal actions for the agent
            num_active_players: number of players in the game
            my_move_count: agent's own move count

        Returns:
            spatial: (NUM_SPATIAL_CHANNELS, 17, 17)
            scalars: (NUM_SCALAR_FEATURES,)
        """
        spatial = np.zeros((NUM_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Channel 0: my pieces
        if my_colour in pin_positions:
            for idx in pin_positions[my_colour]:
                gr, gc = self.cell_to_grid[idx]
                spatial[0, gr, gc] = 1.0

        # Channels 1-5: opponents in turn order (relative to me)
        my_idx_in_order = turn_order.index(my_colour) if my_colour in turn_order else 0
        opp_channel = 1
        for offset in range(1, len(turn_order)):
            opp_idx = (my_idx_in_order + offset) % len(turn_order)
            opp_colour = turn_order[opp_idx]
            if opp_colour in pin_positions and opp_channel <= 5:
                for idx in pin_positions[opp_colour]:
                    gr, gc = self.cell_to_grid[idx]
                    spatial[opp_channel, gr, gc] = 1.0
                opp_channel += 1

        # Channel 6: my target camp
        target_colour = COLOUR_OPPOSITES[my_colour]
        spatial[6] = self.zone_masks[target_colour]

        # Channel 7: my home camp
        spatial[7] = self.zone_masks[my_colour]

        # Channel 8: valid cell mask
        spatial[8] = self.valid_mask

        # Channel 9: board-only cells
        spatial[9] = self.board_only_mask

        # Scalar features
        scalars = np.zeros(NUM_SCALAR_FEATURES, dtype=np.float32)

        # Compute distance info for my pieces
        if my_colour in pin_positions:
            target_cells = [self._ref_board.cells[i] for i in self._ref_board.axial_of_colour(target_colour)]
            dists = []
            pins_in_goal = 0
            pins_in_home = 0

            for idx in pin_positions[my_colour]:
                cell = self._ref_board.cells[idx]
                if cell.postype == target_colour:
                    pins_in_goal += 1
                    dists.append(0.0)
                else:
                    min_d = min(
                        max(abs(cell.q - t.q), abs(cell.r - t.r),
                            abs((-cell.q - cell.r) - (-t.q - t.r)))
                        for t in target_cells
                    )
                    dists.append(min_d)
                if cell.postype == my_colour:
                    pins_in_home += 1

            total_dist = sum(dists)
            max_dist = max(dists) if dists else 0

            scalars[0] = total_dist / 200.0             # normalized total distance
            scalars[1] = max_dist / 16.0                 # normalized max single-pin distance
            scalars[2] = pins_in_home / 10.0             # fraction still in home
            scalars[3] = pins_in_goal / 10.0             # fraction in goal
            scalars[4] = total_legal_actions / 100.0     # normalized legal action count
            scalars[5] = move_count / 300.0              # normalized total move count
            scalars[6] = my_move_count / 150.0           # normalized own move count
            scalars[7] = num_active_players / 6.0        # normalized player count
            # Turn position (0 = first, 1 = last)
            scalars[8] = my_idx_in_order / max(len(turn_order) - 1, 1)
            # Move score proxy (distance from optimal 45 moves)
            scalars[9] = abs(my_move_count - 45) / 45.0

        return torch.from_numpy(spatial), torch.from_numpy(scalars)

    def encode_from_game(self, game, colour: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience: encode directly from a LocalGame instance."""
        state = game.get_state()
        legal = game.get_legal_moves(colour)
        total_actions = sum(len(v) for v in legal.values())

        return self.encode(
            my_colour=colour,
            pin_positions=state["pins"],
            turn_order=state["turn_order"],
            move_count=state["move_count"],
            total_legal_actions=total_actions,
            num_active_players=len(state["turn_order"]),
            my_move_count=game.move_counts_by_colour.get(colour, 0),
        )
