#!/usr/bin/env python3.10
"""
Self-distillation student training.

Loads the teacher checkpoint as the starting point, freezes the value
head (the part that broke in v3-v8 RL attempts), and trains the policy
head via cross-entropy on soft targets — the visit-count distributions
recorded by tools/distill_generate.py.

Why freeze the value head:
  RL fine-tunes (v3-v8) all degraded the agent. The pattern of failure
  was always coupled to noisy value-head signal in multi-player games.
  Pure-supervised distillation on policy targets sidesteps that loop:
  the value head stays exactly as the deployed agent has it, and only
  the policy head moves toward the high-sim teacher's behaviour.

This script writes a NEW checkpoint at checkpoints/<run-name>/model_best.pt.
It does not overwrite phase_0c_v1, server_adapter.py, or anything in the
deployed call path.

Usage (school machine, in tmux):
  python3.10 tools/distill_train.py \\
    --teacher checkpoints/phase_0c_v1/model_best.pt \\
    --data data/distill_v1/data.npz \\
    --run-name phase_0c_v1_distilled \\
    --epochs 30 \\
    --batch-size 256 \\
    --lr 5e-5 \\
    --device cuda
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.policy_value_net import PolicyValueNet


def freeze_value_head(model: PolicyValueNet) -> int:
    """
    Freeze the value head so training only updates the trunk and policy
    head. Returns the number of frozen parameters for logging.
    """
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith("value_conv") or name.startswith("value_fc"):
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy with soft targets.

    F.cross_entropy expects integer class labels; here the targets are
    full probability distributions (visit-count distributions from MCTS),
    so we do it manually: -sum(target * log_softmax(logits)).

    Illegal actions are masked to -1e9 logits inside the model, so their
    softmax output is ~0 and contributes ~0 to the loss. Target prob on
    those is also 0 (visit count was 0).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def main():
    parser = argparse.ArgumentParser(description="Distillation student trainer")
    parser.add_argument("--teacher", type=str, required=True,
                        help="Teacher checkpoint — used as the student's "
                             "starting weights AND kept read-only.")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.npz from distill_generate.py")
    parser.add_argument("--run-name", type=str, default="phase_0c_v1_distilled")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--filter-unfinished", action="store_true",
                        help="Drop moves from games that hit max_moves and never "
                             "reached a clean winner. Removes stagnant-position "
                             "noise (the main cause of the distill_v1 6P "
                             "regression). Requires data generated with the "
                             "post-2026-05-10 generator (game_finished tag in npz).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Conservative LR — student starts from a strong "
                             "checkpoint, we want a small adjustment, not a "
                             "fresh fit.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--num-res-blocks", type=int, default=6)
    parser.add_argument("--trunk-channels", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Distillation student trainer")
    print(f"  Teacher (read-only): {args.teacher}")
    print(f"  Data: {args.data}")
    print(f"  Output: {out_dir}")
    print(f"  Epochs: {args.epochs} | batch: {args.batch_size} | lr: {args.lr}")
    print(f"  Device: {args.device}")

    # Load teacher weights as student starting point.
    if not os.path.exists(args.teacher):
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher}")
    model = PolicyValueNet(
        num_res_blocks=args.num_res_blocks,
        trunk_channels=args.trunk_channels,
    )
    ckpt = torch.load(args.teacher, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)

    # Freeze value head before building optimiser so it sees only trainable
    # params. Trunk + policy head still update; value head stays exactly as
    # phase_0c_v1 has it.
    frozen_count = freeze_value_head(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: total={total:,} trainable={trainable:,} frozen={frozen_count:,}")

    # Load distillation dataset.
    print(f"\nLoading {args.data} ...")
    npz = np.load(args.data)
    spatial = torch.from_numpy(npz["spatial"])
    scalars = torch.from_numpy(npz["scalars"])
    mask = torch.from_numpy(npz["mask"]).bool()
    policy_target = torch.from_numpy(npz["policy_target"])
    N = spatial.shape[0]
    print(f"  {N:,} experiences loaded")

    npz_keys = set(npz.files)
    has_finished = "game_finished" in npz_keys
    has_player_count = "player_count" in npz_keys
    player_count_np = npz["player_count"] if has_player_count else None

    if args.filter_unfinished:
        if not has_finished:
            raise ValueError(
                "--filter-unfinished requires data generated with the "
                "post-2026-05-10 generator. Re-run distill_generate.py to "
                "produce a dataset with the game_finished tag."
            )
        finished_np = npz["game_finished"].astype(bool)
        keep = torch.from_numpy(finished_np)
        spatial = spatial[keep]
        scalars = scalars[keep]
        mask = mask[keep]
        policy_target = policy_target[keep]
        if player_count_np is not None:
            player_count_np = player_count_np[finished_np]
        kept = int(keep.sum().item())
        print(f"  --filter-unfinished: keeping {kept:,} of {N:,} experiences "
              f"({kept / max(N, 1):.1%}) from games that finished with a winner")
        N = spatial.shape[0]

    if player_count_np is not None:
        unique, counts = np.unique(player_count_np, return_counts=True)
        breakdown = ", ".join(f"{int(p)}P={int(c):,}" for p, c in zip(unique, counts))
        print(f"  Per-player-count: {breakdown}")

    # Sanity: target distributions should be ~normalised (MCTS visit counts
    # were normalised by `MCTSNode.best_action` before being returned).
    sums = policy_target.sum(dim=-1)
    print(f"  policy_target sums: min={sums.min():.4f} "
          f"max={sums.max():.4f} mean={sums.mean():.4f}")

    full_ds = TensorDataset(spatial, scalars, mask, policy_target)
    n_val = max(1, int(N * args.val_frac))
    perm = torch.randperm(N)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    train_loader = DataLoader(
        Subset(full_ds, train_idx),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        Subset(full_ds, val_idx),
        batch_size=512, shuffle=False,
    )

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        # Top-1 agreement with the teacher's argmax move — interpretable
        # proxy for "is the student picking the same move."
        correct = 0
        t0 = time.time()
        for sp, sc, m, pt in train_loader:
            sp = sp.to(args.device)
            sc = sc.to(args.device)
            m = m.to(args.device)
            pt = pt.to(args.device)

            logits, _ = model(sp, sc, m)
            loss = soft_cross_entropy(logits, pt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()

            bsz = sp.size(0)
            total_loss += loss.item() * bsz
            n_seen += bsz
            student_top = logits.argmax(dim=-1)
            teacher_top = pt.argmax(dim=-1)
            correct += (student_top == teacher_top).sum().item()

        train_loss = total_loss / max(n_seen, 1)
        train_top1 = correct / max(n_seen, 1)

        # Validation pass.
        model.eval()
        v_total, v_n, v_correct = 0.0, 0, 0
        with torch.no_grad():
            for sp, sc, m, pt in val_loader:
                sp = sp.to(args.device)
                sc = sc.to(args.device)
                m = m.to(args.device)
                pt = pt.to(args.device)
                logits, _ = model(sp, sc, m)
                v_total += soft_cross_entropy(logits, pt).item() * sp.size(0)
                v_n += sp.size(0)
                v_correct += (logits.argmax(dim=-1) == pt.argmax(dim=-1)).sum().item()
        val_loss = v_total / max(v_n, 1)
        val_top1 = v_correct / max(v_n, 1)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_top1": round(train_top1, 4),
            "val_loss": round(val_loss, 5),
            "val_top1": round(val_top1, 4),
            "seconds": round(time.time() - t0, 1),
        }
        history.append(entry)
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={train_loss:.4f} top1={train_top1:.3f} | "
            f"val={val_loss:.4f} top1={val_top1:.3f} | "
            f"{entry['seconds']:.1f}s"
            + ("  <- new best" if improved else "")
        )

    elapsed = time.time() - t_start
    print(f"\nTraining done in {elapsed:.0f}s. Best epoch: {best_epoch} "
          f"(val_loss={best_val_loss:.4f})")

    # Save best checkpoint in the same format as train.py / supervised
    # bootstrap, so existing eval tooling can load it without changes.
    best_path = os.path.join(out_dir, "model_best.pt")
    torch.save({
        "model_state_dict": best_state,
        "epoch": best_epoch,
        "val_loss": best_val_loss,
        "config": {
            "num_res_blocks": args.num_res_blocks,
            "trunk_channels": args.trunk_channels,
        },
    }, best_path)

    final_path = os.path.join(out_dir, "model_final.pt")
    torch.save({
        "model_state_dict": {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()},
        "epoch": args.epochs,
        "val_loss": history[-1]["val_loss"],
        "config": {
            "num_res_blocks": args.num_res_blocks,
            "trunk_channels": args.trunk_channels,
        },
    }, final_path)

    log = {
        "run_name": args.run_name,
        "teacher_checkpoint": args.teacher,
        "data_path": args.data,
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": round(elapsed, 1),
        "history": history,
        "trained_at": datetime.now().isoformat(),
    }
    log_path = os.path.join(out_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nSaved {best_path}")
    print(f"Saved {final_path}")
    print(f"Saved {log_path}")
    print(
        "\nNext: run diagnose_play.py against the new checkpoint and compare "
        "gate metrics to phase_0c_v1. Adoption is a one-line change in "
        "env/server_adapter.py — only do that if the new checkpoint clearly "
        "beats the deployed model on multiple gates."
    )


if __name__ == "__main__":
    main()
