"""
Diagnostic: does select_action_from_server_state produce sensible moves
at an initial 2P red-vs-blue state?

If YES, the oscillation bug lies in rotation (colour != red path).
If NO, the server-state path itself is broken even with identity rotation
— the 8/10-pins validation (which uses select_action, not
select_action_from_server_state) never exercised this code.

Usage:
    python3 diagnose_server_state_path.py \
        --checkpoint checkpoints/sup_20260418_090744/model_best.pt \
        --device cuda
"""

import argparse
import sys

from env.local_game import LocalGame
from agents.chinese_checkers_agent import ChineseCheckersAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mcts-sims", type=int, default=20)
    args = parser.parse_args()

    agent = ChineseCheckersAgent(
        checkpoint_path=args.checkpoint,
        mcts_simulations=args.mcts_sims,
        device=args.device,
    )

    game = LocalGame(num_players=2)
    game.reset()
    state = game.get_state()

    print(f"Initial state: turn_order={state['turn_order']}, current={state['current_colour']}, move_count={state['move_count']}")
    print(f"My (red) pins: {state['pins']['red'][:5]}...")
    print(f"Opp (blue) pins: {state['pins']['blue'][:5]}...")
    print()

    # Path A: local-game path (what validate_multiplayer.py uses)
    pin_a, to_a = agent.select_action(game, "red")
    print(f"select_action(game, 'red'):                      pin={pin_a} -> to={to_a}")

    # Reset MCTS root (both searches start fresh; not strictly necessary
    # but cleaner)
    legal_moves = game.get_legal_moves("red")
    # Path B: server-state path with IDENTITY rotation (my_colour='red')
    pin_b, to_b = agent.select_action_from_server_state(
        pin_positions=state["pins"],
        legal_moves=legal_moves,
        my_colour="red",
        turn_order=state["turn_order"],
        move_count=state["move_count"],
        my_move_count=0,
    )
    print(f"select_action_from_server_state (as red):        pin={pin_b} -> to={to_b}")

    print()
    # Distance from home to the chosen destination
    red_home_cells = set(game.board.axial_of_colour("red"))
    blue_home_cells = set(game.board.axial_of_colour("blue"))

    def describe(to_idx):
        if to_idx in red_home_cells:
            return "red home (no progress)"
        if to_idx in blue_home_cells:
            return "blue home (goal!)"
        return "open board"

    print(f"  A destination: {describe(to_a)}")
    print(f"  B destination: {describe(to_b)}")
    print()
    print(f"Paths {'MATCH' if (pin_a, to_a) == (pin_b, to_b) else 'DIFFER'}")

    if (pin_a, to_a) != (pin_b, to_b):
        print()
        print("  -> select_action_from_server_state is taking a different action")
        print("     than select_action on the IDENTICAL initial state (red identity).")
        print("     This means the bug is in the server-state reconstruction path,")
        print("     not in the colour rotation. Rotation is a red herring.")


if __name__ == "__main__":
    main()
