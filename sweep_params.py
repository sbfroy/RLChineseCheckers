#!/usr/bin/env python3
"""Competition parameter sweep for Phase 0c v1.

Runs sequential sensitivity analysis: sweep one parameter at a time while
holding others at baseline. Produces a summary table of pins-in-goal and
scores across matchups.

Usage:
    python3 sweep_params.py \
      --checkpoint checkpoints/phase_0c_v1/model_best.pt \
      --device cuda
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from itertools import product


BASELINE = {
    "mcts_sims": 100,
    "temperature": 0.1,
    "c_puct": 1.5,
    "endgame_threshold": 8,
    "dirichlet_alpha": 0.0,
    "root_noise_epsilon": 0.0,
}

SWEEPS = {
    "mcts_sims":          [50, 100, 200],
    "temperature":        [0.0, 0.1, 0.5],
    "c_puct":             [1.0, 1.5, 2.0, 3.0],
    "endgame_threshold":  [7, 8, 9],
    "noise":              [
        (0.0, 0.0),
        (0.3, 0.05),
        (0.3, 0.10),
        (0.5, 0.05),
    ],
    "comparison": "grid",
    "confirm":    "grid",
}

COMPARISON_GRID = [
    {"c_puct": 1.0, "temperature": 0.0},
    {"c_puct": 1.0, "temperature": 0.1},
    {"c_puct": 1.0, "temperature": 0.3},
    {"c_puct": 1.0, "temperature": 0.5},
    {"c_puct": 1.0, "temperature": 0.5, "dirichlet_alpha": 0.3, "root_noise_epsilon": 0.05},
    {"c_puct": 1.5, "temperature": 0.0},
    {"c_puct": 1.5, "temperature": 0.5},
]

CONFIRM_GRID = [
    {"c_puct": 1.0, "temperature": 0.3},
    {"c_puct": 1.0, "temperature": 0.0},
]

GRIDS = {
    "comparison": COMPARISON_GRID,
    "confirm":    CONFIRM_GRID,
}

MATCHUPS = [
    "2p_vs_heuristic",
    "2p_self_play",
    "4p_vs_greedy",
    "6p_vs_greedy",
]

GAMES_PER_MATCHUP = 5


def run_config(checkpoint, device, params, out_dir, matchups, games):
    """Run diagnose_play.py with given params, return parsed metadata."""
    cmd = [
        sys.executable, "diagnose_play.py",
        "--checkpoint", checkpoint,
        "--device", device,
        "--mcts-sims", str(params["mcts_sims"]),
        "--temperature", str(params["temperature"]),
        "--c-puct", str(params["c_puct"]),
        "--endgame-threshold", str(params["endgame_threshold"]),
        "--dirichlet-alpha", str(params["dirichlet_alpha"]),
        "--root-noise-epsilon", str(params["root_noise_epsilon"]),
        "--matchups", *matchups,
        "--games-per-matchup", str(games),
        "--out-dir", out_dir,
        "--no-baselines",
    ]
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd[2:])}")
    print(f"Output:  {out_dir}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    meta_path = os.path.join(out_dir, "run_metadata.json")
    if not os.path.exists(meta_path):
        print(f"WARNING: no metadata at {meta_path} (exit code {result.returncode})")
        return None, elapsed

    with open(meta_path) as f:
        metadata = json.load(f)
    return metadata, elapsed


def extract_scores(metadata):
    """Extract per-matchup pins and scores from metadata."""
    agg = metadata.get("aggregate", metadata.get("aggregate_metrics", {}))
    out = {}
    for matchup, m in agg.items():
        out[matchup] = {
            "pins": m.get("mean_rl_pins_in_goal"),
            "score": m.get("mean_rl_score"),
            "moves": m.get("mean_move_count"),
            "wins": m.get("rl_wins", 0),
            "games": m.get("completed", 0),
        }
    return out


def print_summary(results):
    """Print a formatted comparison table."""
    if not results:
        print("No results to display.")
        return

    seen = set()
    for _, scores, _ in results:
        seen.update(scores.keys())
    all_matchups = [m for m in MATCHUPS if m in seen]
    if not all_matchups:
        all_matchups = sorted(seen)
    header = f"  {'config':<45s}"
    for m in all_matchups:
        short = m.replace("_vs_", "/").replace("2p_self_play", "2p/self")
        header += f"  {short:>12s}"
    header += f"  {'total':>6s}  {'time':>6s}"
    print(f"\n{'='*len(header)}")
    print("PARAMETER SWEEP RESULTS (pins in goal)")
    print(f"{'='*len(header)}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, scores, elapsed in results:
        row = f"  {label:<45s}"
        total = 0
        for m in all_matchups:
            if m in scores and scores[m]["pins"] is not None:
                pins = scores[m]["pins"]
                total += pins
                row += f"  {pins:>12.1f}"
            else:
                row += f"  {'--':>12s}"
        row += f"  {total:>6.1f}"
        row += f"  {elapsed:>5.0f}s"
        print(row)

    print()

    print(f"{'='*len(header)}")
    print("PARAMETER SWEEP RESULTS (mean score)")
    print(f"{'='*len(header)}")
    print(header.replace("total", "t.scr"))
    print("  " + "-" * (len(header) - 2))

    for label, scores, elapsed in results:
        row = f"  {label:<45s}"
        total = 0
        for m in all_matchups:
            if m in scores and scores[m]["score"] is not None:
                score = scores[m]["score"]
                total += score
                row += f"  {score:>12.1f}"
            else:
                row += f"  {'--':>12s}"
        row += f"  {total:>6.0f}"
        row += f"  {elapsed:>5.0f}s"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Competition parameter sweep")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--games-per-matchup", type=int, default=GAMES_PER_MATCHUP)
    parser.add_argument("--matchups", nargs="+", default=MATCHUPS)
    parser.add_argument("--out-root", default=None,
                        help="root dir for sweep outputs (default: logs/sweep/<timestamp>)")
    parser.add_argument("--sweep", nargs="+", default=list(SWEEPS.keys()),
                        help="which parameters to sweep (default: all)")
    args = parser.parse_args()

    if args.out_root is None:
        args.out_root = os.path.join("logs", "sweep",
                                     datetime.now().strftime("%Y%m%d_%H%M%S"))

    os.makedirs(args.out_root, exist_ok=True)

    all_results = []
    config_idx = 0

    for sweep_param in args.sweep:
        if sweep_param not in SWEEPS:
            print(f"Unknown sweep parameter: {sweep_param}", file=sys.stderr)
            continue

        print(f"\n{'#'*70}")
        print(f"# SWEEPING: {sweep_param}")
        print(f"{'#'*70}")

        if SWEEPS[sweep_param] == "grid":
            grid = GRIDS[sweep_param]
            for overrides in grid:
                params = dict(BASELINE)
                params.update(overrides)
                parts = [f"{k}={v}" for k, v in overrides.items()]
                label = ", ".join(parts)

                safe_label = label.replace(" ", "_").replace(",", "").replace("=", "")
                out_dir = os.path.join(args.out_root, f"{config_idx:02d}_{safe_label}")
                config_idx += 1

                metadata, elapsed = run_config(
                    args.checkpoint, args.device, params,
                    out_dir, args.matchups, args.games_per_matchup,
                )

                if metadata is not None:
                    scores = extract_scores(metadata)
                    all_results.append((label, scores, elapsed))
                else:
                    all_results.append((label, {}, elapsed))
            continue

        values = SWEEPS[sweep_param]

        for val in values:
            params = dict(BASELINE)

            if sweep_param == "noise":
                alpha, epsilon = val
                params["dirichlet_alpha"] = alpha
                params["root_noise_epsilon"] = epsilon
                label = f"noise: alpha={alpha}, eps={epsilon}"
            else:
                params[sweep_param] = val
                label = f"{sweep_param}={val}"
                if params == BASELINE:
                    label += " (baseline)"

            safe_label = label.replace(" ", "_").replace(",", "").replace("=", "")
            out_dir = os.path.join(args.out_root, f"{config_idx:02d}_{safe_label}")
            config_idx += 1

            metadata, elapsed = run_config(
                args.checkpoint, args.device, params,
                out_dir, args.matchups, args.games_per_matchup,
            )

            if metadata is not None:
                scores = extract_scores(metadata)
                all_results.append((label, scores, elapsed))
            else:
                all_results.append((label, {}, elapsed))

    print_summary(all_results)

    summary_path = os.path.join(args.out_root, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump([
            {"label": label, "scores": scores, "elapsed": elapsed}
            for label, scores, elapsed in all_results
        ], f, indent=2)
    print(f"\nDetailed results saved to: {args.out_root}/")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
