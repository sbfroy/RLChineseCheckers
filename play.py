#!/usr/bin/env python3.10
"""
Play or watch Chinese Checkers games.

Modes:
  watch    - Watch agent vs baseline (or agent vs agent)
  play     - Play interactively against the agent
  baseline - Watch two baselines play

Usage:
  python3.10 play.py watch                              # watch untrained agent vs greedy
  python3.10 play.py watch --checkpoint checkpoints/run_xxx/model_best.pt
  python3.10 play.py play                               # play against untrained agent
  python3.10 play.py play --checkpoint checkpoints/run_xxx/model_best.pt
  python3.10 play.py baseline                           # watch greedy vs random
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from env.local_game import LocalGame, COLOUR_OPPOSITES
from agents.chinese_checkers_agent import ChineseCheckersAgent
from agents.random_agent import RandomAgent
from agents.heuristic_agent import GreedyProgressAgent, HeuristicAgent


COLOUR_SHORT = {
    'red': 'R', 'blue': 'B', 'lawn green': 'G',
    'gray0': 'A', 'yellow': 'Y', 'purple': 'P',
}


def render_board(game, highlight_move=None):
    """Render the board as ASCII with pins shown as colour letters."""
    board = game.board

    # Build pin map: (q, r) -> display char
    pin_map = {}
    for colour, pins in game.pins_by_colour.items():
        ch = COLOUR_SHORT.get(colour, colour[0].upper())
        for p in pins:
            cell = board.cells[p.axialindex]
            pin_map[(cell.q, cell.r)] = ch

    # Highlight destination of last move
    highlight_pos = None
    if highlight_move:
        cell = board.cells[highlight_move]
        highlight_pos = (cell.q, cell.r)

    max_width = max(len(row) for row in board._rows)
    lines = []
    for row in board._rows:
        pad = " " * (max_width - len(row))
        parts = []
        for (q, r, t) in row:
            if (q, r) in pin_map:
                ch = pin_map[(q, r)]
                if highlight_pos and (q, r) == highlight_pos:
                    ch = f"[{ch}]"
                else:
                    ch = f" {ch} "
            elif t == 'board':
                ch = " . "
            else:
                ch = " . "
            parts.append(ch)
        lines.append(pad + "".join(parts))
    return "\n".join(lines)


def print_state(game, last_move_to=None, move_info=None):
    """Print the board and game info."""
    print("\033[2J\033[H", end="")  # clear screen
    print(render_board(game, highlight_move=last_move_to))
    print()

    # Colour legend
    legend = "  ".join(f"{COLOUR_SHORT[c]}={c}" for c in game.colours)
    print(f"Legend: {legend}")
    print(f"Move: {game.move_count}/{game.max_moves}", end="")
    if not game.done:
        print(f"  |  Turn: {game.current_colour()}")
    else:
        print(f"  |  Game over! Winner: {game.winner or 'none'}")

    if move_info:
        print(f"Last: {move_info}")

    # Progress summary
    for colour in game.colours:
        di = game.compute_distance_info(colour)
        bar_filled = di["pins_in_goal"]
        bar_empty = 10 - bar_filled
        bar = "#" * bar_filled + "-" * bar_empty
        print(
            f"  {COLOUR_SHORT[colour]} [{bar}] "
            f"{di['pins_in_goal']}/10 in goal, "
            f"dist={di['total_distance']:.0f}"
        )
    print()


def watch_game(agent_a, agent_b, name_a="Agent", name_b="Baseline",
               num_players=2, delay=0.5, max_moves=300):
    """Watch two agents play."""
    game = LocalGame(num_players=num_players, max_moves=max_moves)
    game.reset()

    agents = {game.colours[0]: (agent_a, name_a), game.colours[1]: (agent_b, name_b)}
    print_state(game)

    while not game.done:
        colour = game.current_colour()
        agent, name = agents[colour]

        # Get move
        start = time.time()
        if hasattr(agent, 'select_action'):
            pin_id, to_idx = agent.select_action(game, colour)
        else:
            # RandomAgent / heuristic
            actions = game.get_all_legal_actions(colour)
            pin_id, to_idx = agent.select_action(game, colour)

        elapsed = time.time() - start

        from_idx = game.pins_by_colour[colour][pin_id].axialindex
        state, done, info = game.step(pin_id, to_idx)

        move_str = (
            f"{name} ({COLOUR_SHORT[colour]}): "
            f"pin {pin_id} cell {from_idx}->{to_idx} ({elapsed:.2f}s)"
        )
        print_state(game, last_move_to=to_idx, move_info=move_str)

        if delay > 0 and not done:
            time.sleep(delay)

    # Final scores
    scores = game.compute_scores()
    print("=== FINAL SCORES ===")
    for colour, sc in scores.items():
        marker = ""
        print(
            f"  {COLOUR_SHORT[colour]} ({colour}): "
            f"{sc['final_score']:.1f} total  "
            f"[pins={sc['pin_goal_score']:.0f}, dist={sc['distance_score']:.0f}, "
            f"moves={sc['moves']}]"
        )


def interactive_game(agent, agent_name="Agent", num_players=2, max_moves=300):
    """Play interactively against the agent."""
    game = LocalGame(num_players=num_players, max_moves=max_moves)
    game.reset()

    human_colour = game.colours[0]
    agent_colour = game.colours[1]
    print(f"You are: {human_colour} ({COLOUR_SHORT[human_colour]})")
    print(f"Agent is: {agent_colour} ({COLOUR_SHORT[agent_colour]})")
    print(f"Your goal: move all pins to the {COLOUR_OPPOSITES[human_colour]} zone")
    print()
    input("Press Enter to start...")

    print_state(game)

    while not game.done:
        colour = game.current_colour()

        if colour == human_colour:
            # Human turn
            legal = game.get_legal_moves(colour)
            actions = []
            for pid, dests in sorted(legal.items()):
                for d in dests:
                    actions.append((pid, d))

            print(f"Your turn ({COLOUR_SHORT[colour]}). Legal moves:")
            for i, (pid, dest) in enumerate(actions):
                pin = game.pins_by_colour[colour][pid]
                from_cell = game.board.cells[pin.axialindex]
                to_cell = game.board.cells[dest]
                print(
                    f"  {i:3d}) pin {pid} "
                    f"({from_cell.q},{from_cell.r}) -> ({to_cell.q},{to_cell.r}) "
                    f"[idx {pin.axialindex}->{dest}]"
                )

            while True:
                try:
                    choice = input(f"\nPick move [0-{len(actions)-1}] (or 'q' to quit): ").strip()
                    if choice.lower() == 'q':
                        print("Quit.")
                        return
                    idx = int(choice)
                    if 0 <= idx < len(actions):
                        pin_id, to_idx = actions[idx]
                        break
                    print(f"Enter a number between 0 and {len(actions)-1}")
                except ValueError:
                    print("Enter a number or 'q'")

            from_idx = game.pins_by_colour[colour][pin_id].axialindex
            state, done, info = game.step(pin_id, to_idx)
            move_str = f"You ({COLOUR_SHORT[colour]}): pin {pin_id} {from_idx}->{to_idx}"

        else:
            # Agent turn
            start = time.time()
            pin_id, to_idx = agent.select_action(game, colour)
            elapsed = time.time() - start

            from_idx = game.pins_by_colour[colour][pin_id].axialindex
            state, done, info = game.step(pin_id, to_idx)
            move_str = f"{agent_name} ({COLOUR_SHORT[colour]}): pin {pin_id} {from_idx}->{to_idx} ({elapsed:.2f}s)"

        print_state(game, last_move_to=to_idx, move_info=move_str)

    # Final scores
    scores = game.compute_scores()
    print("=== FINAL SCORES ===")
    for colour, sc in scores.items():
        marker = " <<< YOU" if colour == human_colour else ""
        print(
            f"  {COLOUR_SHORT[colour]} ({colour}): "
            f"{sc['final_score']:.1f} total  "
            f"[pins={sc['pin_goal_score']:.0f}, dist={sc['distance_score']:.0f}, "
            f"moves={sc['moves']}]{marker}"
        )


def make_agent(args):
    """Build the RL agent from args."""
    return ChineseCheckersAgent(
        checkpoint_path=args.checkpoint,
        mcts_simulations=args.mcts_sims,
        temperature=0.1,
        device=args.device,
    )


def make_baseline(name):
    """Build a baseline agent by name."""
    if name == "random":
        return RandomAgent()
    elif name == "greedy":
        return GreedyProgressAgent()
    elif name == "heuristic":
        return HeuristicAgent()
    else:
        raise ValueError(f"Unknown baseline: {name}. Use: random, greedy, heuristic")


def main():
    parser = argparse.ArgumentParser(description="Play or watch Chinese Checkers")
    sub = parser.add_subparsers(dest="mode")

    # Watch mode
    wp = sub.add_parser("watch", help="Watch agent vs baseline")
    wp.add_argument("--checkpoint", default=None, help="Agent checkpoint path")
    wp.add_argument("--opponent", default="greedy", choices=["random", "greedy", "heuristic"])
    wp.add_argument("--mcts-sims", type=int, default=0, help="MCTS sims (0=network only)")
    wp.add_argument("--delay", type=float, default=0.3, help="Delay between moves (seconds)")
    wp.add_argument("--device", default="cpu")

    # Play mode
    pp = sub.add_parser("play", help="Play against the agent")
    pp.add_argument("--checkpoint", default=None, help="Agent checkpoint path")
    pp.add_argument("--mcts-sims", type=int, default=0, help="MCTS sims (0=network only)")
    pp.add_argument("--device", default="cpu")

    # Baseline mode
    bp = sub.add_parser("baseline", help="Watch two baselines play")
    bp.add_argument("--a", default="greedy", choices=["random", "greedy", "heuristic"])
    bp.add_argument("--b", default="random", choices=["random", "greedy", "heuristic"])
    bp.add_argument("--delay", type=float, default=0.2)

    args = parser.parse_args()

    if args.mode == "watch":
        agent = make_agent(args)
        baseline = make_baseline(args.opponent)
        watch_game(agent, baseline, name_a="RL Agent", name_b=args.opponent.title(),
                   delay=args.delay)

    elif args.mode == "play":
        agent = make_agent(args)
        interactive_game(agent, agent_name="RL Agent")

    elif args.mode == "baseline":
        a = make_baseline(args.a)
        b = make_baseline(args.b)
        watch_game(a, b, name_a=args.a.title(), name_b=args.b.title(),
                   delay=args.delay)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python3.10 play.py watch                    # watch untrained agent vs greedy")
        print("  python3.10 play.py play                     # play against untrained agent")
        print("  python3.10 play.py baseline                 # watch greedy vs random")
        print("  python3.10 play.py watch --checkpoint checkpoints/run_xxx/model_best.pt")


if __name__ == "__main__":
    main()
