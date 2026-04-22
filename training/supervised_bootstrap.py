#!/usr/bin/env python3.10
"""
Supervised imitation bootstrap for the Chinese Checkers policy-value network.

Phase 0 of training. Generates games with HeuristicAgent as the "teacher"
against a pool of opponents (Random, Greedy, stochastic-Heuristic) for
state diversity. Collects (state, teacher_action, final_score/norm) triples
and trains the network to imitate the teacher.

Produces a network that plays roughly at heuristic level, which is a strong
starting point for subsequent AlphaZero-style refinement.

Usage:
    python3.10 training/supervised_bootstrap.py --num-games 2000 --epochs 30 --device cuda
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
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.chinese_checkers_agent import ChineseCheckersAgent
from agents.heuristic_agent import GreedyProgressAgent, HeuristicAgent
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


def generate_games(num_games: int, encoder: BoardEncoder,
                   score_norm: float = 1300.0,
                   num_players_list: list = None,
                   verbose: bool = True) -> dict:
    """
    Play `num_games` games with HeuristicAgent as the teacher.

    The teacher alternates which board seat it plays (first vs last of
    turn_order) across games. The opponent class cycles through a pool
    for state diversity; every non-teacher seat in the game is filled
    with a fresh instance of that opponent class. Player count cycles
    through `num_players_list` (default [2, 4, 6]) so a single run
    produces data for all board sizes.

    Only teacher moves are recorded as training examples. Each example
    is labelled with the teacher's own final competition score at end
    of game.
    """
    if num_players_list is None:
        num_players_list = [2]

    teacher = HeuristicAgent()
    opponent_classes = [
        RandomAgent,
        GreedyProgressAgent,
        lambda: EpsilonHeuristicAgent(epsilon=0.25),
    ]

    spatial_buf = []
    scalars_buf = []
    mask_buf = []
    action_buf = []
    value_placeholder = []  # filled in at end of each game

    t0 = time.time()

    for g in range(num_games):
        opp_cls = opponent_classes[g % len(opponent_classes)]
        teacher_is_first = (g // len(opponent_classes)) % 2 == 0
        num_players = num_players_list[g % len(num_players_list)]

        game = LocalGame(num_players=num_players, max_moves=300)
        game.reset()

        teacher_colour = game.turn_order[0] if teacher_is_first else game.turn_order[-1]
        agents = {}
        for c in game.turn_order:
            if c == teacher_colour:
                agents[c] = teacher
            else:
                agents[c] = opp_cls()

        indices_this_game = []

        while not game.done:
            colour = game.current_colour()
            agent = agents[colour]
            is_teacher = colour == teacher_colour

            if is_teacher:
                spatial, scalars = encoder.encode_from_game(game, colour)
                legal = game.get_legal_moves(colour)
                mask = build_legal_mask(legal).numpy().astype(np.uint8)

            try:
                pin_id, to_index = agent.select_action(game, colour)
            except Exception:
                break

            if is_teacher:
                flat = action_to_flat(pin_id, to_index)
                spatial_buf.append(spatial.numpy().astype(np.float32))
                scalars_buf.append(scalars.numpy().astype(np.float32))
                mask_buf.append(mask)
                action_buf.append(flat)
                indices_this_game.append(len(action_buf) - 1)

            try:
                _, done, _ = game.step(pin_id, to_index)
            except Exception:
                break

        # Assign final value target to all teacher states in this game
        scores = game.compute_scores()
        if teacher_colour in scores:
            v = scores[teacher_colour]["final_score"] / score_norm
        else:
            v = 0.0
        for _ in indices_this_game:
            value_placeholder.append(v)

        if verbose and (g + 1) % 100 == 0:
            dt = time.time() - t0
            print(f"  Generated {g+1}/{num_games} games "
                  f"({len(action_buf):,} exps, {dt:.1f}s)")

    return {
        "spatial": np.stack(spatial_buf),
        "scalars": np.stack(scalars_buf),
        "mask": np.stack(mask_buf),
        "action": np.asarray(action_buf, dtype=np.int64),
        "value": np.asarray(value_placeholder, dtype=np.float32),
    }


def train_model(model, data, device, epochs, batch_size, lr, weight_decay,
                value_loss_weight: float = 1.0, val_frac: float = 0.05):
    """Train the policy-value net via supervised learning."""
    model.to(device)

    N = len(data["action"])
    n_val = max(1, int(N * val_frac))
    perm = np.random.permutation(N)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    def to_tensors(idx):
        return (
            torch.from_numpy(data["spatial"][idx]),
            torch.from_numpy(data["scalars"][idx]),
            torch.from_numpy(data["mask"][idx]).bool(),
            torch.from_numpy(data["action"][idx]),
            torch.from_numpy(data["value"][idx]).unsqueeze(-1),
        )

    train_tensors = to_tensors(train_idx)
    val_tensors = to_tensors(val_idx)
    train_ds = TensorDataset(*train_tensors)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False)

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
            vs_spatial, vs_scalars, vs_mask, vs_action, vs_value = val_tensors
            bsz = 512
            for i in range(0, len(vs_action), bsz):
                s = slice(i, i + bsz)
                logits, v_pred = model(
                    vs_spatial[s].to(device),
                    vs_scalars[s].to(device),
                    vs_mask[s].to(device),
                )
                act = vs_action[s].to(device)
                val = vs_value[s].to(device)
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
    parser.add_argument("--num-games", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
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
    parser.add_argument(
        "--num-players",
        type=str,
        default="2",
        help="Comma-separated list of player counts to train on (e.g. '2,4,6'). "
             "Games cycle through the list; use '2,4,6' for multiplayer bootstrap.",
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

    print(f"=== Supervised bootstrap ===")
    print(f"  Run: {args.run_name}")
    print(f"  Games: {args.num_games} | Epochs: {args.epochs} | "
          f"Batch: {args.batch_size} | Device: {args.device}")
    print(f"  Player counts (cycled across games): {num_players_list}")

    encoder = BoardEncoder()

    print("\n=== Generating games ===")
    t0 = time.time()
    data = generate_games(
        args.num_games, encoder,
        score_norm=args.score_norm,
        num_players_list=num_players_list,
    )
    gen_time = time.time() - t0
    print(
        f"Generated {len(data['action']):,} experiences in {gen_time:.1f}s"
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
    print(f"Trained {args.epochs} epochs in {train_time:.1f}s")
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

    # Load best weights into the live model so eval reflects what we shipped.
    if best_state is not None:
        model.load_state_dict(best_state)

    if args.eval_games > 0:
        print(f"\n=== Eval ({args.eval_games} games per opponent) ===")
        eval_results = evaluate_agent_quick(
            model, args.device, num_games=args.eval_games,
        )
    else:
        print("\n=== Eval skipped (--eval-games 0) ===")
        eval_results = None

    metadata = {
        "run_name": args.run_name,
        "phase": "supervised_bootstrap",
        "started_at": datetime.now().isoformat(),
        "args": vars(args),
        "num_experiences": int(len(data["action"])),
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
