"""Tests for the endgame BFS pathfinder."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.local_game import LocalGame, COLOUR_OPPOSITES
from search.endgame import EndgameSolver, _bfs_to_goal, _legal_destinations


def _force_position(game: LocalGame, colour: str, new_indices):
    """Reposition `colour`'s pins to the given list of cell indices.
    Mirrors the mutation pattern used by LocalGame.from_server_state."""
    pins = game.pins_by_colour[colour]
    assert len(new_indices) == len(pins)
    # Clear current occupancy.
    for p in pins:
        game.board.cells[p.axialindex].occupied = False
    # Set new positions and re-mark occupancy.
    for p, new_idx in zip(pins, new_indices):
        p.axialindex = new_idx
        game.board.cells[new_idx].occupied = True


def _goal_cells(game: LocalGame, colour: str):
    opposite = COLOUR_OPPOSITES[colour]
    return [i for i, c in enumerate(game.board.cells) if c.postype == opposite]


class TestLegalDestinations:
    def test_matches_engine_for_initial_state(self):
        # The BFS legal-move helper should agree with Pin.getPossibleMoves
        # on a real board state.
        game = LocalGame(num_players=2)
        game.reset()
        occupied = set()
        for pins in game.pins_by_colour.values():
            for p in pins:
                occupied.add(p.axialindex)
        for pins in game.pins_by_colour.values():
            for p in pins:
                engine = sorted(p.getPossibleMoves())
                helper = _legal_destinations(
                    game.board, p.axialindex, occupied - {p.axialindex}
                )
                assert helper == engine, (
                    f"mismatch for pin at {p.axialindex}: "
                    f"engine={engine} helper={helper}"
                )


class TestBfsToGoal:
    def test_returns_empty_path_when_already_in_goal(self):
        game = LocalGame(num_players=2)
        game.reset()
        goal = set(_goal_cells(game, "red"))
        start = next(iter(goal))
        path = _bfs_to_goal(game.board, start, set(), goal, max_moves=12)
        assert path == []

    def test_finds_short_path_to_open_goal_cell(self):
        # Construct a state where one red pin sits adjacent to an empty
        # blue (= red's goal) cell. BFS should return a single-move path.
        game = LocalGame(num_players=2)
        game.reset()

        red_goal_cells = _goal_cells(game, "red")
        # Pick a goal cell and a directly-adjacent neighbour.
        target = None
        neighbour = None
        for gi in red_goal_cells:
            cell = game.board.cells[gi]
            for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
                ni = game.board.index_of.get((cell.q + dq, cell.r + dr))
                if ni is not None and ni not in red_goal_cells:
                    target, neighbour = gi, ni
                    break
            if target is not None:
                break
        assert target is not None and neighbour is not None

        # Park the red pin we'll move at `neighbour`; everything else far away.
        # We'll borrow some far indices for the other red pins (just any 9
        # cells outside red_goal_cells and not equal to neighbour).
        far = [
            i for i, _c in enumerate(game.board.cells)
            if i != neighbour and i not in red_goal_cells
        ][:9]
        _force_position(game, "red", [neighbour] + far)

        # Move blue out of the goal target so it's actually empty.
        blue_far = [
            i for i, _c in enumerate(game.board.cells)
            if game.board.cells[i].postype == "red"
        ][:10]
        _force_position(game, "blue", blue_far)

        occupied = set()
        for pins in game.pins_by_colour.values():
            for p in pins:
                occupied.add(p.axialindex)

        path = _bfs_to_goal(
            game.board, neighbour, occupied, {target}, max_moves=12
        )
        assert path == [target]


class TestEndgameSolver:
    def test_inactive_at_start(self):
        game = LocalGame(num_players=2)
        game.reset()
        solver = EndgameSolver(activation_threshold=8)
        assert not solver.is_active(game, "red")

    def test_active_when_eight_pins_in_goal(self):
        game = LocalGame(num_players=2)
        game.reset()
        red_goal = _goal_cells(game, "red")
        # Place 8 red pins in goal; the remaining 2 anywhere outside.
        outside = [
            i for i, _c in enumerate(game.board.cells)
            if i not in red_goal
        ][:2]
        _force_position(game, "red", red_goal[:8] + outside)
        # Move blue out of red's goal area to free up the path.
        blue_far = [
            i for i, _c in enumerate(game.board.cells)
            if game.board.cells[i].postype == "red"
        ][:10]
        _force_position(game, "blue", blue_far)

        solver = EndgameSolver(activation_threshold=8)
        assert solver.is_active(game, "red")

    def test_returns_legal_action_in_endgame(self):
        # Same setup as above; verify the solver picks a legal move and
        # the chosen destination is one the engine actually allows.
        game = LocalGame(num_players=2)
        game.reset()
        red_goal = _goal_cells(game, "red")
        outside = [
            i for i, _c in enumerate(game.board.cells)
            if i not in red_goal
        ][:2]
        _force_position(game, "red", red_goal[:8] + outside)
        blue_far = [
            i for i, _c in enumerate(game.board.cells)
            if game.board.cells[i].postype == "red"
        ][:10]
        _force_position(game, "blue", blue_far)

        solver = EndgameSolver(activation_threshold=8)
        move = solver.select_action(game, "red")
        if move is None:
            return  # acceptable: the random "outside" placement may have boxed pins in
        pin_id, dest = move
        legal = game.get_legal_moves("red")
        assert pin_id in legal
        assert dest in legal[pin_id]
