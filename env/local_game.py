"""
Fast local game environment for training.

Wraps HexBoard and Pin directly — no network, no timeouts.
Gym-like interface: reset(), step(), get_legal_moves().
Supports 2, 4, or 6 players with self-play.
"""

import sys
import os
import math
import copy
from typing import Dict, List, Tuple, Optional, Any

# Add the game engine to path
_ENGINE_DIR = os.path.join(os.path.dirname(__file__), "..", "multi system single machine minimal")
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

from checkers_board import HexBoard, BoardPosition
from checkers_pins import Pin


# Colour pairs and ordering (same as game.py)
COLOUR_ORDER = ['red', 'lawn green', 'yellow', 'blue', 'gray0', 'purple']
COLOUR_OPPOSITES = {
    'red': 'blue', 'blue': 'red',
    'lawn green': 'gray0', 'gray0': 'lawn green',
    'yellow': 'purple', 'purple': 'yellow',
}
# For N-player games, assign colour pairs in this order
COLOUR_PAIRS = [('red', 'blue'), ('lawn green', 'gray0'), ('yellow', 'purple')]


def axial_dist(a: BoardPosition, b: BoardPosition) -> int:
    """Hex distance between two board positions."""
    dq = abs(a.q - b.q)
    dr = abs(a.r - b.r)
    ds = abs((-a.q - a.r) - (-b.q - b.r))
    return max(dq, dr, ds)


class LocalGame:
    """
    Fast local Chinese Checkers environment for training.

    Supports 2, 4, or 6 players. Each player has 10 pins.
    Tracks game state, legal moves, scoring, win/draw detection.
    """

    def __init__(self, num_players: int = 2, max_moves: int = 300):
        assert num_players in (2, 4, 6), "num_players must be 2, 4, or 6"
        self.num_players = num_players
        self.max_moves = max_moves

        # These get set in reset()
        self.board: Optional[HexBoard] = None
        self.colours: List[str] = []
        self.pins_by_colour: Dict[str, List[Pin]] = {}
        self.turn_order: List[str] = []
        self.current_turn_idx: int = 0
        self.move_count: int = 0
        self.move_counts_by_colour: Dict[str, int] = {}
        self.done: bool = False
        self.winner: Optional[str] = None
        self.status: str = "NOT_STARTED"

        # Precompute target cell indices per colour (set once per reset)
        self._target_indices: Dict[str, List[int]] = {}
        self._target_cells: Dict[str, List[BoardPosition]] = {}

    def reset(self) -> Dict[str, Any]:
        """Reset the game to initial state. Returns initial observation."""
        self.board = HexBoard()
        self.done = False
        self.winner = None
        self.move_count = 0
        self.status = "PLAYING"

        # Assign colours
        num_pairs = self.num_players // 2
        self.colours = []
        for i in range(num_pairs):
            c1, c2 = COLOUR_PAIRS[i]
            self.colours.extend([c1, c2])

        # Build turn order matching server logic
        first = self.colours[0]
        if first in COLOUR_ORDER:
            idx = COLOUR_ORDER.index(first)
            rotated = COLOUR_ORDER[idx:] + COLOUR_ORDER[:idx]
        else:
            rotated = COLOUR_ORDER[:]
        self.turn_order = [c for c in rotated if c in self.colours]
        self.current_turn_idx = 0

        # Initialize pins
        self.pins_by_colour = {}
        self.move_counts_by_colour = {}
        for colour in self.colours:
            idxs = self.board.axial_of_colour(colour)[:10]
            self.pins_by_colour[colour] = [
                Pin(self.board, idxs[i], id=i, color=colour)
                for i in range(len(idxs))
            ]
            self.move_counts_by_colour[colour] = 0

        # Precompute target zones
        self._target_indices = {}
        self._target_cells = {}
        for colour in self.colours:
            opp = COLOUR_OPPOSITES[colour]
            tidxs = self.board.axial_of_colour(opp)
            self._target_indices[colour] = tidxs
            self._target_cells[colour] = [self.board.cells[i] for i in tidxs]

        return self.get_state()

    def current_colour(self) -> str:
        """Return the colour whose turn it is."""
        return self.turn_order[self.current_turn_idx]

    def get_legal_moves(self, colour: Optional[str] = None) -> Dict[int, List[int]]:
        """
        Return legal moves for a colour: {pin_id: [dest_index, ...]}.
        Defaults to current turn colour.
        """
        if colour is None:
            colour = self.current_colour()
        pins = self.pins_by_colour[colour]
        legal = {}
        for pin in pins:
            moves = pin.getPossibleMoves()
            if moves:
                legal[pin.id] = moves
        return legal

    def get_all_legal_actions(self, colour: Optional[str] = None) -> List[Tuple[int, int]]:
        """Return flat list of (pin_id, to_index) legal actions."""
        legal = self.get_legal_moves(colour)
        actions = []
        for pid, dests in legal.items():
            for d in dests:
                actions.append((pid, d))
        return actions

    def step(self, pin_id: int, to_index: int) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        """
        Execute a move. Returns (state, done, info).

        info contains: colour, pin_id, from_index, to_index, status,
                       winner, scores (if done).
        """
        if self.done:
            raise RuntimeError("Game is already done. Call reset().")

        colour = self.current_colour()
        pins = self.pins_by_colour[colour]

        if not (0 <= pin_id < len(pins)):
            raise ValueError(f"Invalid pin_id {pin_id} for {colour}")

        pin = pins[pin_id]
        legal = pin.getPossibleMoves()
        if to_index not in legal:
            raise ValueError(f"Illegal move: pin {pin_id} to {to_index}. Legal: {legal}")

        from_index = pin.axialindex

        # Apply move (suppress placePin's print)
        self.board.cells[pin.axialindex].occupied = False
        pin.axialindex = to_index
        self.board.cells[to_index].occupied = True

        self.move_count += 1
        self.move_counts_by_colour[colour] += 1

        info = {
            "colour": colour,
            "pin_id": pin_id,
            "from_index": from_index,
            "to_index": to_index,
        }

        # Check win
        status = self._check_status(colour)
        if status == "WIN":
            self.done = True
            self.winner = colour
            self.status = "FINISHED"
            info["status"] = "WIN"
            info["winner"] = colour
            info["scores"] = self.compute_scores()
            return self.get_state(), True, info

        # Check draw (no legal moves for current player after advancing)
        # Also check max moves
        if self.move_count >= self.max_moves:
            self.done = True
            self.status = "FINISHED"
            info["status"] = "MAX_MOVES"
            info["scores"] = self.compute_scores()
            return self.get_state(), True, info

        # Advance turn
        self._advance_turn()

        # Check if next player has no moves (draw) — skip them
        skipped = 0
        while skipped < len(self.turn_order):
            next_colour = self.current_colour()
            if self.get_legal_moves(next_colour):
                break
            skipped += 1
            self._advance_turn()

        if skipped >= len(self.turn_order):
            # Nobody can move — game over
            self.done = True
            self.status = "FINISHED"
            info["status"] = "DRAW"
            info["scores"] = self.compute_scores()
            return self.get_state(), True, info

        info["status"] = "CONTINUE"
        return self.get_state(), False, info

    def _advance_turn(self):
        """Move to next player in turn order."""
        self.current_turn_idx = (self.current_turn_idx + 1) % len(self.turn_order)

    def _check_status(self, colour: str) -> str:
        """Check if a colour has won or drawn."""
        opposite = COLOUR_OPPOSITES[colour]
        pins = self.pins_by_colour[colour]

        # WIN: all pins in opposite zone
        if all(self.board.cells[p.axialindex].postype == opposite for p in pins):
            return "WIN"

        return "PLAYING"

    def get_state(self) -> Dict[str, Any]:
        """Return full game state as a dictionary."""
        return {
            "status": self.status,
            "current_colour": self.current_colour() if not self.done else None,
            "move_count": self.move_count,
            "turn_order": list(self.turn_order),
            "pins": {
                colour: [p.axialindex for p in pins]
                for colour, pins in self.pins_by_colour.items()
            },
            "winner": self.winner,
            "done": self.done,
        }

    def get_pin_positions(self, colour: str) -> List[int]:
        """Return list of cell indices for a colour's pins."""
        return [p.axialindex for p in self.pins_by_colour[colour]]

    def compute_scores(self) -> Dict[str, Dict[str, float]]:
        """Compute competition scoring for all players. Mirrors game.py logic."""
        scores = {}
        for colour in self.colours:
            pins = self.pins_by_colour[colour]
            opposite = COLOUR_OPPOSITES[colour]
            mc = self.move_counts_by_colour[colour]

            # Time score — not applicable in local training (set to max)
            time_score = 100.0

            # Move score — asymmetric Gaussian centered at 45
            if mc > 0:
                sigma = 4 if mc < 45 else 18
                move_score = math.exp(-((mc - 45) ** 2) / (2 * sigma ** 2))
            else:
                move_score = 0.0

            # Pins in goal
            pins_in_goal = sum(
                1 for p in pins
                if self.board.cells[p.axialindex].postype == opposite
            )
            pin_goal_score = pins_in_goal * 100.0

            # Distance score
            target_cells = self._target_cells[colour]
            total_dist = 0
            for p in pins:
                if self.board.cells[p.axialindex].postype != opposite:
                    best = min(axial_dist(self.board.cells[p.axialindex], tgt) for tgt in target_cells)
                    total_dist += best
            distance_score = max(0.0, 200.0 - total_dist) if mc > 0 else 0.0

            final_score = time_score + move_score + pin_goal_score + distance_score

            scores[colour] = {
                "final_score": final_score,
                "time_score": time_score,
                "move_score": move_score,
                "pin_goal_score": pin_goal_score,
                "distance_score": distance_score,
                "pins_in_goal": pins_in_goal,
                "total_distance": total_dist,
                "moves": mc,
            }
        return scores

    def compute_distance_info(self, colour: str) -> Dict[str, float]:
        """Compute distance-related metrics for a single colour."""
        pins = self.pins_by_colour[colour]
        opposite = COLOUR_OPPOSITES[colour]
        target_cells = self._target_cells[colour]

        dists = []
        pins_in_goal = 0
        pins_in_home = 0
        home_zone = colour

        for p in pins:
            cell = self.board.cells[p.axialindex]
            if cell.postype == opposite:
                pins_in_goal += 1
                dists.append(0.0)
            else:
                best = min(axial_dist(cell, tgt) for tgt in target_cells)
                dists.append(best)
            if cell.postype == home_zone:
                pins_in_home += 1

        return {
            "total_distance": sum(dists),
            "max_distance": max(dists) if dists else 0.0,
            "min_distance": min(dists) if dists else 0.0,
            "pins_in_goal": pins_in_goal,
            "pins_in_home": pins_in_home,
            "per_pin_distances": dists,
        }

    def clone(self) -> 'LocalGame':
        """Deep copy the game state for MCTS simulation."""
        new_game = LocalGame.__new__(LocalGame)
        new_game.num_players = self.num_players
        new_game.max_moves = self.max_moves
        new_game.done = self.done
        new_game.winner = self.winner
        new_game.move_count = self.move_count
        new_game.status = self.status
        new_game.current_turn_idx = self.current_turn_idx
        new_game.turn_order = list(self.turn_order)
        new_game.colours = list(self.colours)
        new_game.move_counts_by_colour = dict(self.move_counts_by_colour)

        # Deep copy the board
        new_game.board = HexBoard()
        # Reset all occupancy
        for cell in new_game.board.cells:
            cell.occupied = False

        # Deep copy pins and set occupancy
        new_game.pins_by_colour = {}
        for colour, pins in self.pins_by_colour.items():
            new_pins = []
            for p in pins:
                # Create pin without triggering occupancy in __init__
                new_pin = Pin.__new__(Pin)
                new_pin.board = new_game.board
                new_pin.axialindex = p.axialindex
                new_pin.color = p.color
                new_pin.id = p.id
                new_game.board.cells[p.axialindex].occupied = True
                new_pins.append(new_pin)
            new_game.pins_by_colour[colour] = new_pins

        # Recompute target caches
        new_game._target_indices = {}
        new_game._target_cells = {}
        for colour in new_game.colours:
            opp = COLOUR_OPPOSITES[colour]
            tidxs = new_game.board.axial_of_colour(opp)
            new_game._target_indices[colour] = tidxs
            new_game._target_cells[colour] = [new_game.board.cells[i] for i in tidxs]

        return new_game

    @classmethod
    def from_server_state(
        cls,
        pin_positions: Dict[str, List[int]],
        turn_order: List[str],
        current_turn_colour: str,
        move_count: int = 0,
    ) -> 'LocalGame':
        """
        Reconstruct a LocalGame from server state data.

        This allows MCTS to be used during competition play — we rebuild
        the board from the server's JSON state, then MCTS can clone and
        simulate from it.

        Args:
            pin_positions: {colour: [cell_idx, ...]} for all players
            turn_order: list of colours in turn order
            current_turn_colour: whose turn it is now
            move_count: total moves played so far
        """
        colours = list(pin_positions.keys())
        num_players = len(colours)

        game = cls.__new__(cls)
        game.num_players = num_players
        game.max_moves = 300
        game.done = False
        game.winner = None
        game.move_count = move_count
        game.status = "PLAYING"
        game.colours = colours
        game.turn_order = list(turn_order)

        # Set current turn index to match the server's current turn
        game.current_turn_idx = 0
        if current_turn_colour in turn_order:
            game.current_turn_idx = turn_order.index(current_turn_colour)

        # We don't know per-player move counts from server state,
        # so estimate evenly
        per_player = move_count // num_players
        game.move_counts_by_colour = {c: per_player for c in colours}

        # Build fresh board
        game.board = HexBoard()
        for cell in game.board.cells:
            cell.occupied = False

        # Place pins from server positions
        game.pins_by_colour = {}
        for colour, positions in pin_positions.items():
            pins = []
            for i, idx in enumerate(positions):
                pin = Pin.__new__(Pin)
                pin.board = game.board
                pin.axialindex = idx
                pin.color = colour
                pin.id = i
                game.board.cells[idx].occupied = True
                pins.append(pin)
            game.pins_by_colour[colour] = pins

        # Build target caches
        game._target_indices = {}
        game._target_cells = {}
        for colour in colours:
            opp = COLOUR_OPPOSITES[colour]
            tidxs = game.board.axial_of_colour(opp)
            game._target_indices[colour] = tidxs
            game._target_cells[colour] = [game.board.cells[i] for i in tidxs]

        return game
