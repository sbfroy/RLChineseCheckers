#!/usr/bin/env python3.10
"""
Supervised imitation bootstrap for the Chinese Checkers policy-value network.

Phase 0 of training. Generates games between diverse agents and learns from
ALL competent agents' moves (not just one teacher). Each agent's moves are
labelled with that agent's own final competition score, so the network learns
both what good play looks like (policy) and what scores different positions
lead to (value).

Usage:
    python3.10 training/supervised_bootstrap.py \
      --num-games 50000 --epochs 150 --device cuda --run-name phase_0c_v1
"""

import argparse
import json
import os
import random as py_random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.chinese_checkers_agent import ChineseCheckersAgent
from agents.heuristic_agent import GreedyProgressAgent, HeuristicAgent
from agents.maxdistance_agent import MaxDistanceAgent
from agents.random_agent import RandomAgent
from env.action_mapping import action_to_flat, build_legal_mask
from env.local_game import LocalGame
from models.encoders import BoardEncoder
from models.policy_value_net import PolicyValueNet
from training.evaluate import play_match


class EpsilonHeuristicAgent:
    """HeuristicAgent with ε-greedy random-legal fallback (for diversity)."""

    def __init__(self, epsilon: float = 0.2):
        self.epsilon = epsilon
        self.inner = HeuristicAgent()

    def select_action(self, game, colour):
        if py_random.random() < self.epsilon:
            actions = game.get_all_legal_actions(colour)
            if actions:
                return py_random.choice(actions)
        return self.inner.select_action(game, colour)


RECORDABLE_AGENTS = [
    ("HeuristicAgent", HeuristicAgent),
    ("GreedyProgress", GreedyProgressAgent),
    ("MaxDistance", MaxDistanceAgent),
    ("EpsHeuristic_0.15", lambda: EpsilonHeuristicAgent(epsilon=0.15)),
]

NON_RECORDABLE_AGENTS = [
    ("Random", RandomAgent),
    ("EpsHeuristic_0.30", lambda: EpsilonHeuristicAgent(epsilon=0.30)),
]

ALL_AGENTS = RECORDABLE_AGENTS + NON_RECORDABLE_AGENTS
RECORDABLE_NAMES = {name for name, _ in RECORDABLE_AGENTS}


def generate_games(num_games: int, encoder: BoardEncoder,
                   score_norm: float = 1300.0,
                   num_players_list: list = None,
                   max_moves_per_player: int = 200,
                   max_experiences: int = 1_500_000,
                   verbose: bool = True) -> dict:
    """
    Play games between diverse agents. Moves from ALL competent agents are
    recorded as training data. Each agent's value target is its own final
    competition score / score_norm.

    Stops early if max_experiences is reached (for memory safety).
    """
    if num_players_list is None:
        num_players_list = [2, 4, 6]

    spatial_buf = []
    scalars_buf = []
    mask_buf = []
    action_buf = []
    value_buf = []

    seat_counts = {name: 0 for name, _ in ALL_AGENTS}
    recorded_counts = {name: 0 for name, _ in RECORDABLE_AGENTS}
    games_by_np = {np_: 0 for np_ in num_players_list}
    games_finished = 0
    total_generated = 0

    t0 = time.time()

    for g in range(num_games):
        if len(action_buf) >= max_experiences:
            if verbose:
                print(f"  Experience cap reached ({max_experiences:,}) at game {g}.")
            break

        num_players = num_players_list[g % len(num_players_list)]
        max_moves = max_moves_per_player * num_players
        game = LocalGame(num_players=num_players, max_moves=max_moves)
        game.reset()

        agents = {}
        agent_names = {}

        # First seat always gets a recordable agent (guarantees data per game).
        first_colour = game.turn_order[0]
        name, factory = py_random.choice(RECORDABLE_AGENTS)
        agents[first_colour] = factory()
        agent_names[first_colour] = name
        seat_counts[name] += 1

        # Remaining seats: sample from the full pool independently.
        for colour in game.turn_order[1:]:
            name, factory = py_random.choice(ALL_AGENTS)
            agents[colour] = factory()
            agent_names[colour] = name
            seat_counts[name] += 1

        # Track which colours we're recording (all recordable agents).
        record_colours = {
            c for c in game.turn_order if agent_names[c] in RECORDABLE_NAMES
        }

        # Per-colour buffers for this game: (spatial, scalars, mask, action)
        game_data = {c: [] for c in record_colours}

        while not game.done:
            colour = game.current_colour()
            recording = colour in record_colours

            if recording:
                spatial, scalars = encoder.encode_from_game(game, colour)
                legal = game.get_legal_moves(colour)
                mask = build_legal_mask(legal).numpy().astype(np.uint8)

            try:
                pin_id, to_index = agents[colour].select_action(game, colour)
            except Exception:
                break

            if recording:
                flat = action_to_flat(pin_id, to_index)
                game_data[colour].append((
                    spatial.numpy().astype(np.float32),
                    scalars.numpy().astype(np.float32),
                    mask,
                    flat,
                ))

            try:
                _, done, _ = game.step(pin_id, to_index)
            except Exception:
                break

        # Assign per-agent value targets from their own final scores.
        scores = game.compute_scores()
        hit_max = game.move_count >= max_moves
        if not hit_max:
            games_finished += 1
        games_by_np[num_players] = games_by_np.get(num_players, 0) + 1

        for colour in record_colours:
            moves = game_data[colour]
            if not moves:
                continue
            if colour in scores:
                v = scores[colour]["final_score"] / score_norm
            else:
                v = 0.0

            for sp, sc, mk, act in moves:
                spatial_buf.append(sp)
                scalars_buf.append(sc)
                mask_buf.append(mk)
                action_buf.append(act)
                value_buf.append(v)

            recorded_counts[agent_names[colour]] = (
                recorded_counts.get(agent_names[colour], 0) + len(moves)
            )

        total_generated = g + 1

        if verbose and (g + 1) % 500 == 0:
            dt = time.time() - t0
            mem_gb = len(action_buf) * 12822 / 1e9
            print(f"  {g+1}/{num_games} games | "
                  f"{len(action_buf):,} exps (~{mem_gb:.1f} GB) | "
                  f"{dt:.0f}s | "
                  f"finished: {games_finished}/{g+1}")

    dt = time.time() - t0
    if verbose:
        print(f"\n  Completed {total_generated:,} games in {dt:.0f}s")
        print(f"  Games that finished (not max_moves): "
              f"{games_finished}/{total_generated} "
              f"({100*games_finished/max(total_generated,1):.0f}%)")
        print(f"  Games by player count: {games_by_np}")
        print(f"\n  Seat distribution (total seats filled):")
        total_seats = sum(seat_counts.values()) or 1
        for name, n in sorted(seat_counts.items()):
            print(f"    {name:28s}  {n:7d}  ({100*n/total_seats:5.1f}%)")
        print(f"\n  Recorded experiences by agent type:")
        for name, n in sorted(recorded_counts.items()):
            print(f"    {name:28s}  {n:9,}")

    return {
        "spatial": np.stack(spatial_buf),
        "scalars": np.stack(scalars_buf),
        "mask": np.stack(mask_buf),
        "action": np.asarray(action_buf, dtype=np.int64),
        "value": np.asarray(value_buf, dtype=np.float32),
    }


def train_model(model, data, device, epochs, batch_size, lr, weight_decay,
                value_loss_weight: float = 1.0, val_frac: float = 0.05):
    """Train the policy-value net via supervised learning.

    Uses Subset to avoid copying the full dataset during train/val split,
    keeping peak memory close to 1× the dataset size.
    """
    model.to(device)

    N = len(data["action"])

    # Convert numpy → tensor once (shares memory via from_numpy, no copy).
    all_spatial = torch.from_numpy(data["spatial"])
    all_scalars = torch.from_numpy(data["scalars"])
    all_mask = torch.from_numpy(data["mask"]).bool()
    all_action = torch.from_numpy(data["action"])
    all_value = torch.from_numpy(data["value"]).unsqueeze(-1)

    # Free the numpy dict to reclaim memory.
    del data

    full_ds = TensorDataset(all_spatial, all_scalars, all_mask,
                            all_action, all_value)

    n_val = max(1, int(N * val_frac))
    perm = torch.randperm(N)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False,
                            num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    history = []
    best_val_pol = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_pol, total_val, n = 0.0, 0.0, 0
        correct, total = 0, 0
        t0 = time.time()

        for spatial, scalars, mask, action, value in train_loader:
            spatial = spatial.to(device)
            scalars = scalars.to(device)
            mask = mask.to(device)
            action = action.to(device)
            value = value.to(device)

            logits, v_pred = model(spatial, scalars, mask)
            pol_loss = F.cross_entropy(logits, action)
            val_loss = F.mse_loss(v_pred, value)
            loss = pol_loss + value_loss_weight * val_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bsz = action.size(0)
            total_pol += pol_loss.item() * bsz
            total_val += val_loss.item() * bsz
            n += bsz
            correct += (logits.argmax(dim=-1) == action).sum().item()
            total += bsz

        # Validation pass
        model.eval()
        v_pol, v_val, v_n, v_correct = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for vs_spatial, vs_scalars, vs_mask, vs_action, vs_value in val_loader:
                logits, v_pred = model(
                    vs_spatial.to(device),
                    vs_scalars.to(device),
                    vs_mask.to(device),
                )
                act = vs_action.to(device)
                val = vs_value.to(device)
                v_pol += F.cross_entropy(logits, act, reduction='sum').item()
                v_val += F.mse_loss(v_pred, val, reduction='sum').item()
                v_correct += (logits.argmax(dim=-1) == act).sum().item()
                v_n += act.size(0)

        entry = {
            "epoch": epoch,
            "policy_loss": total_pol / n,
            "value_loss": total_val / n,
            "action_accuracy": correct / total,
            "val_policy_loss": v_pol / max(v_n, 1),
            "val_value_loss": v_val / max(v_n, 1),
            "val_action_accuracy": v_correct / max(v_n, 1),
            "seconds": round(time.time() - t0, 1),
        }
        history.append(entry)
        improved = entry["val_policy_loss"] < best_val_pol
        if improved:
            best_val_pol = entry["val_policy_loss"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_epoch = epoch
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"pol={entry['policy_loss']:.4f} val_pol={entry['val_policy_loss']:.4f} | "
            f"v={entry['value_loss']:.4f} val_v={entry['val_value_loss']:.4f} | "
            f"acc={entry['action_accuracy']:.3f} val_acc={entry['val_action_accuracy']:.3f} | "
            f"{entry['seconds']:.1f}s"
            + ("  <- new best" if improved else "")
        )

    return history, best_state, best_epoch


def evaluate_agent_quick(model, device, num_games=5):
    agent = ChineseCheckersAgent(
        model=model, mcts_simulations=20, temperature=0.1, device=device,
    )
    results = {}
    for name, baseline in [("random", RandomAgent()),
                           ("greedy", GreedyProgressAgent()),
                           ("heuristic", HeuristicAgent())]:
        r = play_match(agent, baseline, num_games=num_games)
        results[name] = {
            "avg_score": round(r["avg_a_score"], 1),
            "opp_avg_score": round(r["avg_b_score"], 1),
            "avg_game_length": round(r["avg_game_length"], 1),
            "wins": r["a_wins"],
            "losses": r["b_wins"],
            "draws": r["draws"],
        }
        print(
            f"  vs {name}: avg_score={r['avg_a_score']:.1f} "
            f"(opp={r['avg_b_score']:.1f}) "
            f"W{r['a_wins']}/L{r['b_wins']}/D{r['draws']} "
            f"len={r['avg_game_length']:.0f}"
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Supervised imitation bootstrap")
    parser.add_argument("--num-games", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-res-blocks", type=int, default=6)
    parser.add_argument("--trunk-channels", type=int, default=128)
    parser.add_argument("--eval-games", type=int, default=5)
    parser.add_argument("--score-norm", type=float, default=1300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-moves-per-player", type=int, default=200,
                        help="Max moves per player per game. 200 = generous; "
                             "competition has no move limit.")
    parser.add_argument("--max-experiences", type=int, default=1_500_000,
                        help="Stop generating once this many experiences are "
                             "collected. ~12.5 KB per experience. "
                             "1.5M = ~19 GB RAM.")
    parser.add_argument(
        "--num-players",
        type=str,
        default="2,4,6",
        help="Comma-separated list of player counts to train on (e.g. '2,4,6'). "
             "Games cycle round-robin through the list, so each count gets equal "
             "coverage. Default '2,4,6' = balanced multi-player bootstrap.",
    )
    args = parser.parse_args()

    num_players_list = [int(x) for x in args.num_players.split(",") if x.strip()]
    for np_ in num_players_list:
        if np_ not in (2, 4, 6):
            parser.error(f"--num-players values must be 2, 4, or 6 (got {np_})")

    if args.device == "cpu" and torch.cuda.is_available():
        print(f"  Note: CUDA available ({torch.cuda.get_device_name(0)}); "
              f"pass --device cuda to use it.")

    if args.run_name is None:
        args.run_name = f"sup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log_dir = os.path.join("logs", args.run_name)
    ckpt_dir = os.path.join("checkpoints", args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    py_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    mem_est_gb = args.max_experiences * 12822 / 1e9
    print(f"=== Supervised bootstrap (Phase 0c) ===")
    print(f"  Run: {args.run_name}")
    print(f"  Games: {args.num_games} (will stop early at "
          f"{args.max_experiences:,} experiences)")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | "
          f"Device: {args.device}")
    print(f"  Player counts: {num_players_list}")
    print(f"  Max moves/player: {args.max_moves_per_player} "
          f"(2P={args.max_moves_per_player*2}, "
          f"4P={args.max_moves_per_player*4}, "
          f"6P={args.max_moves_per_player*6})")
    print(f"  Memory budget: ~{mem_est_gb:.1f} GB for data")
    print(f"  Recording moves from: "
          f"{', '.join(n for n, _ in RECORDABLE_AGENTS)}")

    encoder = BoardEncoder()

    print("\n=== Generating games ===")
    t0 = time.time()
    data = generate_games(
        args.num_games, encoder,
        score_norm=args.score_norm,
        num_players_list=num_players_list,
        max_moves_per_player=args.max_moves_per_player,
        max_experiences=args.max_experiences,
    )
    gen_time = time.time() - t0
    print(
        f"\nGenerated {len(data['action']):,} experiences in {gen_time:.0f}s"
    )
    print(
        f"  Value target: [{data['value'].min():.3f}, "
        f"{data['value'].max():.3f}] mean={data['value'].mean():.3f} "
        f"std={data['value'].std():.3f}"
    )
    total_bytes = sum(a.nbytes for a in data.values())
    print(f"  Data memory: {total_bytes / 1e9:.2f} GB")

    print("\n=== Building model ===")
    model = PolicyValueNet(
        num_res_blocks=args.num_res_blocks,
        trunk_channels=args.trunk_channels,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    print("\n=== Training ===")
    t0 = time.time()
    history, best_state, best_epoch = train_model(
        model, data,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        value_loss_weight=args.value_loss_weight,
    )
    train_time = time.time() - t0
    print(f"Trained {args.epochs} epochs in {train_time:.0f}s")
    if best_state is not None:
        print(f"  Best val_policy_loss at epoch {best_epoch} "
              f"(val_pol={history[best_epoch-1]['val_policy_loss']:.4f})")

    print("\n=== Saving checkpoints ===")
    final_path = os.path.join(ckpt_dir, "model_final.pt")
    torch.save({
        "iteration": args.epochs,
        "run_name": args.run_name,
        "model_state_dict": model.state_dict(),
        "history": history,
        "args": vars(args),
    }, final_path)
    print(f"  Saved final (epoch {args.epochs}): {final_path}")

    ckpt_path = os.path.join(ckpt_dir, "model_best.pt")
    save_state = best_state if best_state is not None else model.state_dict()
    torch.save({
        "iteration": best_epoch if best_state is not None else args.epochs,
        "run_name": args.run_name,
        "model_state_dict": save_state,
        "history": history,
        "args": vars(args),
    }, ckpt_path)
    print(f"  Saved best (epoch {best_epoch if best_state else args.epochs}): "
          f"{ckpt_path}")

    # Load best weights for eval.
    if best_state is not None:
        model.load_state_dict(best_state)

    if args.eval_games > 0:
        print(f"\n=== Quick 2P sanity eval ({args.eval_games} games per opponent) ===")
        print("  NOTE: 2P-only, doesn't reflect 4P/6P strength. Smoke check only.")
        eval_results = evaluate_agent_quick(
            model, args.device, num_games=args.eval_games,
        )
    else:
        print("\n=== Quick 2P sanity eval skipped (--eval-games 0) ===")
        eval_results = None

    print("\n=== GATE: run diagnose_play.py against this checkpoint ===")
    print("  python3 diagnose_play.py \\")
    print(f"    --checkpoint {ckpt_path} \\")
    print(f"    --device {args.device} \\")
    print("    --mcts-sims 20 \\")
    print("    --temperature 0.1 \\")
    print("    --matchups 2p_vs_random 2p_vs_greedy 2p_vs_heuristic "
          "2p_self_play 4p_vs_greedy 6p_vs_greedy \\")
    print("    --games-per-matchup 4")

    metadata = {
        "run_name": args.run_name,
        "phase": "supervised_bootstrap",
        "started_at": datetime.now().isoformat(),
        "args": vars(args),
        "num_experiences": int(len(data["action"])) if "action" in data else 0,
        "generation_seconds": round(gen_time, 1),
        "training_seconds": round(train_time, 1),
        "model_params": n_params,
        "final_metrics": history[-1] if history else None,
        "eval_results": eval_results,
    }
    with open(os.path.join(log_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(log_dir, "history.jsonl"), "w") as f:
        for h in history:
            f.write(json.dumps(h) + "\n")

    print(f"\nDone.")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Logs:       {log_dir}")


if __name__ == "__main__":
    main()
