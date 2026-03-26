"""Tests for move generation and game mechanics."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.local_game import LocalGame, COLOUR_OPPOSITES


class TestMoveGeneration:
    def test_initial_legal_moves_exist(self):
        game = LocalGame(num_players=2)
        game.reset()
        legal = game.get_legal_moves()
        assert len(legal) > 0, "Should have legal moves at start"

    def test_legal_moves_are_valid_indices(self):
        game = LocalGame(num_players=2)
        game.reset()
        legal = game.get_legal_moves()
        for pid, dests in legal.items():
            assert 0 <= pid < 10
            for d in dests:
                assert 0 <= d < 121

    def test_legal_moves_go_to_empty_cells(self):
        game = LocalGame(num_players=2)
        game.reset()
        legal = game.get_legal_moves()
        for pid, dests in legal.items():
            for d in dests:
                assert not game.board.cells[d].occupied, \
                    f"Dest {d} should be empty"

    def test_move_changes_position(self):
        game = LocalGame(num_players=2)
        game.reset()
        colour = game.current_colour()
        legal = game.get_legal_moves()
        pid = list(legal.keys())[0]
        dest = legal[pid][0]
        old_pos = game.pins_by_colour[colour][pid].axialindex
        game.step(pid, dest)
        new_pos = game.pins_by_colour[colour][pid].axialindex
        assert new_pos == dest
        assert new_pos != old_pos

    def test_illegal_move_raises(self):
        game = LocalGame(num_players=2)
        game.reset()
        with pytest.raises(ValueError):
            game.step(0, 60)  # Center cell, unlikely to be legal from start

    def test_move_after_done_raises(self):
        game = LocalGame(num_players=2, max_moves=2)
        game.reset()
        legal = game.get_legal_moves()
        pid = list(legal.keys())[0]
        game.step(pid, legal[pid][0])
        legal = game.get_legal_moves()
        pid = list(legal.keys())[0]
        game.step(pid, legal[pid][0])
        # Game should be done now (max_moves=2)
        with pytest.raises(RuntimeError):
            game.step(0, 0)


class TestGameMechanics:
    def test_turn_alternation_2p(self):
        game = LocalGame(num_players=2)
        game.reset()
        c1 = game.current_colour()
        legal = game.get_legal_moves()
        pid = list(legal.keys())[0]
        game.step(pid, legal[pid][0])
        c2 = game.current_colour()
        assert c1 != c2, "Turn should alternate"

    def test_turn_rotation_4p(self):
        game = LocalGame(num_players=4)
        game.reset()
        seen = []
        for _ in range(4):
            c = game.current_colour()
            seen.append(c)
            legal = game.get_legal_moves()
            pid = list(legal.keys())[0]
            game.step(pid, legal[pid][0])
        assert len(set(seen)) == 4, "Should cycle through 4 players"

    def test_max_moves_ends_game(self):
        game = LocalGame(num_players=2, max_moves=10)
        game.reset()
        for i in range(10):
            if game.done:
                break
            legal = game.get_legal_moves()
            pid = list(legal.keys())[0]
            game.step(pid, legal[pid][0])
        assert game.done

    def test_scoring_initial(self):
        game = LocalGame(num_players=2)
        game.reset()
        # Play one move to enable scoring
        legal = game.get_legal_moves()
        pid = list(legal.keys())[0]
        game.step(pid, legal[pid][0])
        scores = game.compute_scores()
        for colour in game.colours:
            s = scores[colour]
            assert "final_score" in s
            assert "pin_goal_score" in s
            assert "distance_score" in s
            assert s["pin_goal_score"] == 0.0  # No pins in goal at start

    def test_clone_independence(self):
        game = LocalGame(num_players=2)
        game.reset()
        clone = game.clone()

        # Play move on clone
        legal = clone.get_legal_moves()
        pid = list(legal.keys())[0]
        clone.step(pid, legal[pid][0])

        # Original should be unchanged
        assert game.move_count == 0
        assert clone.move_count == 1

    def test_distance_info(self):
        game = LocalGame(num_players=2)
        game.reset()
        for colour in game.colours:
            di = game.compute_distance_info(colour)
            assert di["total_distance"] > 0
            assert di["max_distance"] > 0
            assert di["pins_in_goal"] == 0
            assert di["pins_in_home"] == 10
            assert len(di["per_pin_distances"]) == 10

    def test_from_server_state(self):
        """Reconstruct a LocalGame from server-style state and verify it plays."""
        # Set up a normal game and play a few moves
        orig = LocalGame(num_players=2)
        orig.reset()
        for _ in range(4):
            colour = orig.current_colour()
            legal = orig.get_legal_moves(colour)
            pid = list(legal.keys())[0]
            orig.step(pid, legal[pid][0])

        # Extract server-style data
        pin_positions = {
            c: [p.axialindex for p in pins]
            for c, pins in orig.pins_by_colour.items()
        }
        turn_order = list(orig.turn_order)
        current = orig.current_colour()

        # Reconstruct
        rebuilt = LocalGame.from_server_state(
            pin_positions=pin_positions,
            turn_order=turn_order,
            current_turn_colour=current,
            move_count=orig.move_count,
        )

        # Check basic properties
        assert rebuilt.current_colour() == current
        assert rebuilt.move_count == orig.move_count
        assert rebuilt.num_players == orig.num_players
        assert not rebuilt.done

        # Check pin positions match
        for c in orig.colours:
            orig_pos = sorted(p.axialindex for p in orig.pins_by_colour[c])
            rebuilt_pos = sorted(p.axialindex for p in rebuilt.pins_by_colour[c])
            assert orig_pos == rebuilt_pos

        # Verify rebuilt game can generate legal moves and step
        legal = rebuilt.get_legal_moves()
        assert len(legal) > 0
        pid = list(legal.keys())[0]
        state, done, info = rebuilt.step(pid, legal[pid][0])
        assert info["status"] in ("CONTINUE", "WIN", "MAX_MOVES")
