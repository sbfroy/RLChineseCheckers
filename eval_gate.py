#!/usr/bin/env python3
"""Phase 1a gate eval.

Compares the Phase 1a value-fix checkpoint against the Phase 0 baseline at
several MCTS sim counts vs greedy and heuristic. Phase 1a passes the gate
iff its MCTS-50 score beats Phase 0's MCTS-20 score on the same opponent.

Usage:
  python3.10 eval_gate.py \\
    --phase0 checkpoints/sup_20260418_090744/model_best.pt \\
    --phase1a checkpoints/run_20260425_085340/model_best.pt \\
    --device cuda
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from agents.chinese_checkers_agent import ChineseCheckersAgent
from agents.heuristic_agent import GreedyProgressAgent, HeuristicAgent
from training.evaluate import play_match


def eval_one(ckpt, sims, opponent, num_games, device, use_endgame):
    agent = ChineseCheckersAgent(
        checkpoint_path=ckpt,
        mcts_simulations=sims,
        temperature=0.1,
        device=device,
        use_endgame=use_endgame,
    )
    res = play_match(
        agent_a=agent,
        agent_b=opponent,
        num_games=num_games,
        num_players=2,
        max_moves=300,
    )
    return res["avg_a_score"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase0", required=True)
    p.add_argument("--phase1a", required=True)
    p.add_argument("--sims", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--num-games", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--endgame-modes",
        nargs="+",
        default=["off", "on"],
        choices=["off", "on"],
        help="Run with endgame solver disabled, enabled, or both (default: both).",
    )
    args = p.parse_args()

    opponents = [
        ("greedy", GreedyProgressAgent()),
        ("heuristic", HeuristicAgent()),
    ]
    ckpts = [("phase0", args.phase0), ("phase1a", args.phase1a)]

    rows = []
    for endgame_mode in args.endgame_modes:
        use_endgame = endgame_mode == "on"
        for label, ckpt in ckpts:
            for sims in args.sims:
                scores = {}
                for op_name, op in opponents:
                    scores[op_name] = eval_one(
                        ckpt, sims, op, args.num_games, args.device, use_endgame
                    )
                rows.append((endgame_mode, label, sims, scores))

    print()
    print(f"{'endgame':<8} {'ckpt':<10} {'sims':<6} {'greedy':<10} {'heuristic':<10}")
    print("-" * 50)
    for endgame_mode, label, sims, scores in rows:
        print(
            f"{endgame_mode:<8} {label:<10} {sims:<6} "
            f"{scores['greedy']:<10.1f} {scores['heuristic']:<10.1f}"
        )
    print()
    print("Gate: endgame=on must beat endgame=off on greedy (target: > 1098).")


if __name__ == "__main__":
    main()
