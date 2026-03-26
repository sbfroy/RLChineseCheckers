"""
Heuristic baseline agent for Chinese Checkers.

Strategies:
- GreedyProgressAgent: picks the move that reduces total distance to goal the most
- HeuristicAgent: combines distance progress with lagging-piece awareness
"""

import sys
import os
from typing import Tuple, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.local_game import LocalGame, COLOUR_OPPOSITES, axial_dist


class GreedyProgressAgent:
    """
    Greedy baseline: always picks the move that maximizes
    the reduction in total distance to the goal zone.
    """

    def select_action(self, game: LocalGame, colour: str) -> Tuple[int, int]:
        legal = game.get_legal_moves(colour)
        if not legal:
            raise RuntimeError(f"No legal actions for {colour}")

        opposite = COLOUR_OPPOSITES[colour]
        target_cells = game._target_cells[colour]
        board = game.board

        best_action = None
        best_improvement = -float('inf')

        for pid, dests in legal.items():
            pin = game.pins_by_colour[colour][pid]
            current_cell = board.cells[pin.axialindex]

            # Current distance to nearest target
            if current_cell.postype == opposite:
                current_dist = 0
            else:
                current_dist = min(axial_dist(current_cell, t) for t in target_cells)

            for dest in dests:
                dest_cell = board.cells[dest]
                if dest_cell.postype == opposite:
                    dest_dist = 0
                else:
                    dest_dist = min(axial_dist(dest_cell, t) for t in target_cells)

                improvement = current_dist - dest_dist
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_action = (pid, dest)

        return best_action


class HeuristicAgent:
    """
    Smarter heuristic: combines distance progress with lagging-piece priority.

    Scoring formula for each move:
    score = distance_improvement + lagging_bonus + goal_zone_bonus

    Prioritizes:
    1. Getting pins into the goal zone
    2. Moving the most-lagging pin forward
    3. General forward progress
    """

    def __init__(self, lagging_weight: float = 2.0, goal_bonus: float = 5.0):
        self.lagging_weight = lagging_weight
        self.goal_bonus = goal_bonus

    def select_action(self, game: LocalGame, colour: str) -> Tuple[int, int]:
        legal = game.get_legal_moves(colour)
        if not legal:
            raise RuntimeError(f"No legal actions for {colour}")

        opposite = COLOUR_OPPOSITES[colour]
        target_cells = game._target_cells[colour]
        board = game.board
        pins = game.pins_by_colour[colour]

        # Compute current distances for all pins
        pin_dists = []
        for p in pins:
            cell = board.cells[p.axialindex]
            if cell.postype == opposite:
                pin_dists.append(0)
            else:
                pin_dists.append(min(axial_dist(cell, t) for t in target_cells))

        max_dist = max(pin_dists) if pin_dists else 0

        best_action = None
        best_score = -float('inf')

        for pid, dests in legal.items():
            pin = pins[pid]
            current_cell = board.cells[pin.axialindex]
            current_dist = pin_dists[pid]

            for dest in dests:
                dest_cell = board.cells[dest]
                if dest_cell.postype == opposite:
                    dest_dist = 0
                else:
                    dest_dist = min(axial_dist(dest_cell, t) for t in target_cells)

                # Base: distance improvement
                improvement = current_dist - dest_dist

                # Lagging bonus: extra reward if this pin is the furthest behind
                lagging_bonus = 0.0
                if current_dist == max_dist and current_dist > 0:
                    lagging_bonus = self.lagging_weight * improvement

                # Goal zone bonus: extra reward for entering the goal zone
                goal_bonus = 0.0
                if dest_cell.postype == opposite and current_cell.postype != opposite:
                    goal_bonus = self.goal_bonus

                score = improvement + lagging_bonus + goal_bonus

                if score > best_score:
                    best_score = score
                    best_action = (pid, dest)

        return best_action
