#!/usr/bin/env python3
"""
Multiplayer validation for Phase 0.

Runs Phase 0 (RL agent as colour 0) against GreedyProgressAgent baselines
in 2P / 4P / 6P games. Confirms the agent (a) doesn't crash on non-2P
boards, (b) makes legal moves, (c) completes games. Measures per-move
inference latency so we can check the <2s/move competition constraint.

Uses MCTS 20 sims (the validated good setting — see memory note
"MCTS sim count pathology"). Higher sim counts make Phase 0 play badly.

Usage:
  python3 validate_multiplayer.py \
    --checkpoint checkpoints/sup_20260418_090744/model_best.pt \
    --device cuda
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from env.local_game import LocalGame
from agents.chinese_checkers_agent import ChineseCheckersAgent
from agents.heuristic_agent import GreedyProgressAgent


def run_game(agent, num_players, max_moves=300):
    """
    Play one game: RL agent as turn_order[0], Greedy baselines for all others.
    Returns dict with crash status, scores, pins-in-goal, RL latencies.
    """
    game = LocalGame(num_players=num_players, max_moves=max_moves)
    game.reset()

    rl_colour = game.turn_order[0]
    agents = {rl_colour: ("RL", agent)}
    for c in game.turn_order[1:]:
        agents[c] = (f"Greedy({c})", GreedyProgressAgent())

    rl_move_times = []
    crash = None

    while not game.done:
        colour = game.current_colour()
        name, a = agents[colour]
        t0 = time.time()
        try:
            pin_id, to_idx = a.select_action(game, colour)
            elapsed = time.time() - t0
            if colour == rl_colour:
                rl_move_times.append(elapsed)
            game.step(pin_id, to_idx)
        except Exception as e:
            crash = f"{type(e).__name__}: {e} (colour={colour}, move={game.move_count})"
            break

    scores = game.compute_scores()
    per_colour = {}
    for c in game.turn_order:
        s = scores.get(c, {})
        per_colour[c] = {
            "final_score": s.get("final_score", 0.0),
            "pins_in_goal": s.get("pins_in_goal", 0),
            "is_rl": c == rl_colour,
        }

    rl_lat = {}
    if rl_move_times:
        sorted_t = sorted(rl_move_times)
        rl_lat["moves"] = len(rl_move_times)
        rl_lat["avg_s"] = sum(rl_move_times) / len(rl_move_times)
        rl_lat["max_s"] = max(rl_move_times)
        rl_lat["p95_s"] = sorted_t[int(0.95 * len(sorted_t))]

    return {
        "num_players": num_players,
        "crash": crash,
        "move_count": game.move_count,
        "completed": game.done,
        "winner": game.winner,
        "rl_colour": rl_colour,
        "per_colour": per_colour,
        "rl_latency": rl_lat,
    }


def print_report(result):
    np_ = result["num_players"]
    print(f"\n{'=' * 60}")
    print(f" {np_}-PLAYER GAME  (RL plays as {result['rl_colour']})")
    print(f"{'=' * 60}")

    if result["crash"]:
        print(f"  !!! CRASH: {result['crash']}")
        return

    print(f"  Completed: {result['completed']}  moves: {result['move_count']}")
    print(f"  Winner: {result['winner'] or 'draw (hit max_moves)'}")
    print()
    print(f"  Per-colour final scores:")
    for c, info in result["per_colour"].items():
        marker = " <- RL" if info["is_rl"] else ""
        print(
            f"    {c:12s} score={info['final_score']:7.1f}  "
            f"pins_in_goal={info['pins_in_goal']:2d}/10{marker}"
        )

    lat = result["rl_latency"]
    if lat:
        print()
        print(f"  RL inference latency over {lat['moves']} moves:")
        print(f"    avg: {lat['avg_s']*1000:6.1f} ms")
        print(f"    p95: {lat['p95_s']*1000:6.1f} ms")
        print(f"    max: {lat['max_s']*1000:6.1f} ms")
        budget_ms = 2000
        print(
            f"    competition budget: {budget_ms} ms/move — "
            f"{'PASS' if lat['max_s']*1000 < budget_ms else 'FAIL'}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mcts-sims", type=int, default=20,
                        help="MCTS simulations per move (20 is the validated good setting)")
    parser.add_argument("--players", type=int, nargs="+", default=[2, 4, 6],
                        help="Player counts to test (e.g. --players 2 4 6)")
    parser.add_argument("--max-moves", type=int, default=300)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading agent: {args.checkpoint}")
    print(f"  device={args.device}, mcts_sims={args.mcts_sims}")
    agent = ChineseCheckersAgent(
        checkpoint_path=args.checkpoint,
        mcts_simulations=args.mcts_sims,
        temperature=0.1,
        device=args.device,
    )

    results = []
    for np_ in args.players:
        if np_ not in (2, 4, 6):
            print(f"Skipping invalid player count: {np_} (must be 2/4/6)")
            continue
        print(f"\nRunning {np_}-player game ...")
        t0 = time.time()
        r = run_game(agent, num_players=np_, max_moves=args.max_moves)
        r["wall_time_s"] = time.time() - t0
        results.append(r)
        print_report(r)
        print(f"  wall clock: {r['wall_time_s']:.1f}s")

    print(f"\n{'=' * 60}")
    print(" SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        rl = next(v for v in r["per_colour"].values() if v["is_rl"])
        status = "CRASH" if r["crash"] else ("OK" if r["completed"] else "HUNG")
        lat_str = ""
        if r["rl_latency"]:
            lat_str = f"  max {r['rl_latency']['max_s']*1000:.0f}ms"
        print(
            f"  {r['num_players']}P: {status:6s}  "
            f"RL score={rl['final_score']:7.1f}  "
            f"RL pins={rl['pins_in_goal']:2d}/10{lat_str}"
        )

    any_crash = any(r["crash"] for r in results)
    sys.exit(1 if any_crash else 0)


if __name__ == "__main__":
    main()
