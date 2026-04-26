"""
MaxDistance baseline agent (PunGrumpy-inspired play style).

Different from GreedyProgressAgent: instead of maximising distance reduction
to goal, this picks the move with the largest forward jump distance on the
board, with a tiebreak that prefers moving the most-backwards piece. Falls
back to sideways then backward when no forward move is available.

Why include it as a training opponent: produces a noticeably different state
distribution from our existing greedy/heuristic — chain-jumps tend to be
selected over single-step distance-reducing moves, which exposes the network
to more "leaping" board configurations than greedy alone produces.

Algorithmic credit: PunGrumpy/ai-chinese-checker (GPL-3.0). Reimplemented
natively in our engine; no PunGrumpy code is included or required at runtime.
"""

import sys
import os
import random
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.local_game import LocalGame, COLOUR_OPPOSITES, axial_dist


class MaxDistanceAgent:
    """Pick the move with the largest forward jump distance toward the goal."""

    def select_action(self, game: LocalGame, colour: str) -> Tuple[int, int]:
        legal = game.get_legal_moves(colour)
        if not legal:
            raise RuntimeError(f"No legal actions for {colour}")

        target_cells = game._target_cells[colour]
        opposite = COLOUR_OPPOSITES[colour]
        board = game.board

        forward, sideways, backward = [], [], []
        for pid, dests in legal.items():
            pin = game.pins_by_colour[colour][pid]
            cur_cell = board.cells[pin.axialindex]
            cur_dist = (
                0 if cur_cell.postype == opposite
                else min(axial_dist(cur_cell, t) for t in target_cells)
            )
            for dest in dests:
                dest_cell = board.cells[dest]
                dest_dist = (
                    0 if dest_cell.postype == opposite
                    else min(axial_dist(dest_cell, t) for t in target_cells)
                )
                # Forward = closer to goal, sideways = same distance, backward = further.
                # The pin's own start-distance is the tiebreak signal: we prefer
                # forward moves that originate from pieces that are themselves
                # furthest from the goal (matches PunGrumpy's "more backwards" rule).
                jump = cur_dist - dest_dist
                entry = (jump, cur_dist, pid, dest)
                if jump > 0:
                    forward.append(entry)
                elif jump == 0:
                    sideways.append(entry)
                else:
                    backward.append(entry)

        # Pick the bucket: forward > sideways > backward.
        bucket = forward or sideways or backward
        # Sort by primary key (jump distance, descending) then by lagging-piece
        # preference (start_dist, descending). Random tiebreak among equals so
        # opponent play has variance across data-generation games.
        bucket.sort(key=lambda e: (-e[0], -e[1]))
        top_jump, top_lag = bucket[0][0], bucket[0][1]
        winners = [e for e in bucket if e[0] == top_jump and e[1] == top_lag]
        choice = random.choice(winners)
        return (choice[2], choice[3])
