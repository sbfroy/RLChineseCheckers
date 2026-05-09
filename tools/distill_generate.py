#!/usr/bin/env python3.10
"""
Self-distillation teacher data generation.

Plays games using the deployed agent (phase_0c_v1 + endgame solver) at a
high MCTS sim budget, recording the visit-count distribution at every
network-decided move. The resulting (state, soft_policy) tuples are the
training targets for distill_train.py.

This script is purely additive — it loads the existing teacher checkpoint
read-only and writes new data files. It does not modify any deployed code
path or overwrite any existing checkpoint.

Game mix:
  - self-play games: every seat uses the teacher at high sims. Produces
    decisive games in 4P/6P (both sides clear their goal triangles).
  - mixed games: one teacher seat, others filled with GreedyProgressAgent
    or HeuristicAgent. Matches the eval distribution (4p_vs_greedy etc.)
    so the student sees opponent styles it'll actually face.

Endgame solver moves are SKIPPED — the deployed agent uses the solver
deterministically, so the network's policy output is irrelevant during
endgame. Only midgame MCTS-policy moves are recorded.

Usage (school machine, in tmux):
  python3.10 tools/distill_generate.py \\
    --teacher checkpoints/phase_0c_v1/model_best.pt \\
    --output-dir data/distill_v1 \\
    --num-games 150 \\
    --mcts-sims 300 \\
    --device cuda
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.heuristic_agent import GreedyProgressAgent, HeuristicAgent
from env.action_mapping import ACTION_SPACE_SIZE, build_legal_mask, flat_to_action
from env.local_game import LocalGame
from models.encoders import BoardEncoder
from models.policy_value_net import PolicyValueNet
from search.endgame import EndgameSolver
from search.mcts import MCTS


def load_teacher(checkpoint_path: str, device: str) -> PolicyValueNet:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
    model = PolicyValueNet(num_res_blocks=6, trunk_channels=128)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def play_game(
    model: PolicyValueNet,
    encoder: BoardEncoder,
    mcts: MCTS,
    endgame: EndgameSolver,
    num_players: int,
    max_moves_per_player: int,
    device: str,
    mode: str,
    baseline_pool: list,
):
    """
    Play one game. Returns recorded experiences (only teacher moves) and
    a stats dict.

    mode == "self_play":  every colour uses the teacher.
    mode == "mixed":      colour 0 is teacher, others use a sampled baseline
                          (one baseline per game, applied to every non-teacher
                          seat — matches the train.py pattern).
    """
    max_moves = max_moves_per_player * num_players
    game = LocalGame(num_players=num_players, max_moves=max_moves)
    game.reset()

    teacher_colour = game.turn_order[0] if mode == "mixed" else None
    baseline = random.choice(baseline_pool) if mode == "mixed" else None

    records = []
    move_count = 0
    endgame_moves = 0
    teacher_moves = 0

    while not game.done:
        colour = game.current_colour()

        is_teacher_seat = (mode == "self_play") or (colour == teacher_colour)

        if not is_teacher_seat:
            try:
                pin_id, to_index = baseline.select_action(game, colour)
                game.step(pin_id, to_index)
            except Exception:
                break
            move_count += 1
            continue

        # Teacher seat. Endgame solver short-circuits MCTS.
        if endgame.is_active(game, colour):
            move = endgame.select_action(game, colour)
            if move is not None:
                pin_id, to_index = move
                game.step(pin_id, to_index)
                move_count += 1
                endgame_moves += 1
                continue

        # MCTS at high sims, T=1.0 returns visit distribution as soft target.
        spatial, scalars = encoder.encode_from_game(game, colour)
        legal = game.get_legal_moves(colour)
        mask = build_legal_mask(legal)

        if not mask.any():
            break

        policy = mcts.search(game, colour, model, encoder)
        if policy.sum() == 0:
            break

        # Record (state, soft policy target). Argmax for action — strongest
        # play, deterministic, matches deployment-temperature behaviour.
        records.append({
            "spatial": spatial.numpy().astype(np.float32),
            "scalars": scalars.numpy().astype(np.float32),
            "mask": mask.numpy().astype(bool),
            "policy_target": policy.astype(np.float32),
        })
        teacher_moves += 1

        action_idx = int(policy.argmax())
        pin_id, to_index = flat_to_action(action_idx)
        try:
            game.step(pin_id, to_index)
        except Exception:
            break
        move_count += 1

    finished = game.winner is not None
    hit_max = game.move_count >= game.max_moves

    return records, {
        "num_players": num_players,
        "mode": mode,
        "baseline": type(baseline).__name__ if baseline else None,
        "move_count": move_count,
        "teacher_moves": teacher_moves,
        "endgame_moves": endgame_moves,
        "finished_with_winner": finished,
        "hit_max_moves": hit_max,
    }


def main():
    parser = argparse.ArgumentParser(description="Distillation teacher data generator")
    parser.add_argument("--teacher", type=str, required=True,
                        help="Path to teacher checkpoint (read-only).")
    parser.add_argument("--output-dir", type=str, default="data/distill_v1")
    parser.add_argument("--num-games", type=int, default=150)
    parser.add_argument("--mcts-sims", type=int, default=300)
    parser.add_argument("--c-puct", type=float, default=1.0,
                        help="Match deployed agent (c=1.0) so the student "
                             "imitates a stronger version of the same prior.")
    parser.add_argument("--num-players-list", type=str, default="2,4,6")
    parser.add_argument("--self-play-frac", type=float, default=0.5,
                        help="Fraction of games where all seats are teacher. "
                             "Rest are 1-teacher vs sampled baseline.")
    parser.add_argument("--max-moves-per-player", type=int, default=200)
    parser.add_argument("--max-experiences", type=int, default=80000,
                        help="Stop generation early if dataset hits this size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    num_players_list = [int(x) for x in args.num_players_list.split(",")]

    print(f"Distillation data generator")
    print(f"  Teacher: {args.teacher}")
    print(f"  Output:  {args.output_dir}")
    print(f"  Games: {args.num_games} | sims: {args.mcts_sims} | c_puct: {args.c_puct}")
    print(f"  Players: {num_players_list} | self_play_frac: {args.self_play_frac}")
    print(f"  Device: {args.device}")

    model = load_teacher(args.teacher, args.device)
    encoder = BoardEncoder()
    mcts = MCTS(
        c_puct=args.c_puct,
        num_simulations=args.mcts_sims,
        temperature=1.0,
        device=args.device,
    )
    endgame = EndgameSolver(activation_threshold=8)
    baseline_pool = [GreedyProgressAgent(), HeuristicAgent()]

    all_records = []
    game_stats = []
    t0 = time.time()

    for g in range(args.num_games):
        # Round-robin player count, randomised mode.
        num_players = num_players_list[g % len(num_players_list)]
        mode = "self_play" if random.random() < args.self_play_frac else "mixed"

        records, stats = play_game(
            model=model,
            encoder=encoder,
            mcts=mcts,
            endgame=endgame,
            num_players=num_players,
            max_moves_per_player=args.max_moves_per_player,
            device=args.device,
            mode=mode,
            baseline_pool=baseline_pool,
        )
        all_records.extend(records)
        game_stats.append(stats)

        elapsed = time.time() - t0
        rate = (g + 1) / max(elapsed, 1.0)
        print(
            f"  game {g+1:4d}/{args.num_games} | "
            f"{num_players}P {mode:10s} | "
            f"moves={stats['move_count']:4d} | "
            f"teacher={stats['teacher_moves']:4d} "
            f"endgame={stats['endgame_moves']:3d} | "
            f"finished={stats['finished_with_winner']} | "
            f"buf={len(all_records):,} | "
            f"{elapsed:.0f}s ({rate:.2f} g/s)",
            flush=True,
        )

        if len(all_records) >= args.max_experiences:
            print(f"  Hit max_experiences cap ({args.max_experiences:,}), stopping.")
            break

    elapsed = time.time() - t0
    print(f"\nGenerated {len(all_records):,} teacher experiences in {elapsed:.0f}s")

    # Pack into numpy arrays.
    spatial = np.stack([r["spatial"] for r in all_records])
    scalars = np.stack([r["scalars"] for r in all_records])
    mask = np.stack([r["mask"] for r in all_records])
    policy_target = np.stack([r["policy_target"] for r in all_records])

    out_path = os.path.join(args.output_dir, "data.npz")
    np.savez_compressed(
        out_path,
        spatial=spatial,
        scalars=scalars,
        mask=mask,
        policy_target=policy_target,
    )

    finished = sum(1 for s in game_stats if s["finished_with_winner"])
    by_mode = {}
    by_np = {}
    for s in game_stats:
        by_mode.setdefault(s["mode"], 0)
        by_mode[s["mode"]] += 1
        by_np.setdefault(s["num_players"], 0)
        by_np[s["num_players"]] += 1

    metadata = {
        "teacher_checkpoint": args.teacher,
        "num_games_played": len(game_stats),
        "num_experiences": len(all_records),
        "mcts_sims": args.mcts_sims,
        "c_puct": args.c_puct,
        "self_play_frac": args.self_play_frac,
        "num_players_list": num_players_list,
        "max_moves_per_player": args.max_moves_per_player,
        "games_finished_with_winner": finished,
        "games_by_mode": by_mode,
        "games_by_num_players": by_np,
        "elapsed_seconds": round(elapsed, 1),
        "generated_at": datetime.now().isoformat(),
        "seed": args.seed,
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved {out_path}")
    print(f"Saved {meta_path}")
    print(f"  Finished games (clean winner): {finished}/{len(game_stats)}")
    print(f"  By mode: {by_mode}")
    print(f"  By player count: {by_np}")


if __name__ == "__main__":
    main()
