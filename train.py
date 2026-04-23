#!/usr/bin/env python3.10
"""
Main training script for the Chinese Checkers RL agent.

Usage:
    python3.10 train.py                          # default config
    python3.10 train.py --config configs/default.yaml
    python3.10 train.py --iterations 50 --games 10 --mcts-sims 0
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
import yaml
import torch

sys.path.insert(0, os.path.dirname(__file__))

from models.policy_value_net import PolicyValueNet
from models.encoders import BoardEncoder
from training.trainer import Trainer, TrainingConfig
from training.evaluate import evaluate_agent, play_match
from search.mcts import MCTS
from env.rewards import RewardConfig
from agents.chinese_checkers_agent import ChineseCheckersAgent
from agents.random_agent import RandomAgent
from agents.heuristic_agent import GreedyProgressAgent, HeuristicAgent


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> PolicyValueNet:
    mc = cfg.get("model", {})
    return PolicyValueNet(
        num_res_blocks=mc.get("num_res_blocks", 6),
        trunk_channels=mc.get("trunk_channels", 128),
        policy_channels=mc.get("policy_channels", 32),
        value_hidden=mc.get("value_hidden", 128),
    )


def build_training_config(cfg: dict, args) -> TrainingConfig:
    tc = cfg.get("training", {})
    mc = cfg.get("mcts", {})
    return TrainingConfig(
        num_iterations=args.iterations or tc.get("num_iterations", 200),
        num_games_per_iteration=args.games or tc.get("num_games_per_iteration", 30),
        num_players=args.players or tc.get("num_players", 2),
        batch_size=tc.get("batch_size", 128),
        epochs_per_iteration=tc.get("epochs_per_iteration", 5),
        learning_rate=tc.get("learning_rate", 1e-3),
        weight_decay=tc.get("weight_decay", 1e-4),
        policy_loss_weight=tc.get("policy_loss_weight", 1.0),
        value_loss_weight=tc.get("value_loss_weight", 1.0),
        buffer_capacity=tc.get("buffer_capacity", 200_000),
        min_buffer_size=tc.get("min_buffer_size", 512),
        temperature=tc.get("temperature", 1.0),
        checkpoint_dir=tc.get("checkpoint_dir", "checkpoints"),
        checkpoint_every=tc.get("checkpoint_every", 10),
        eval_every=tc.get("eval_every", 20),
        device=args.device or tc.get("device", "cpu"),
        mcts_simulations=args.mcts_sims if args.mcts_sims is not None else mc.get("num_simulations", 100),
        freeze_policy=args.freeze_policy if hasattr(args, 'freeze_policy') else tc.get("freeze_policy", False),
        kl_anchor_weight=args.kl_weight if (hasattr(args, 'kl_weight') and args.kl_weight is not None) else tc.get("kl_anchor_weight", 0.0),
    )


def build_reward_config(cfg: dict) -> RewardConfig:
    rc = cfg.get("rewards", {})
    return RewardConfig(
        win_reward=rc.get("win_reward", 1.0),
        loss_reward=rc.get("loss_reward", -1.0),
        draw_reward=rc.get("draw_reward", -0.5),
        max_moves_reward=rc.get("max_moves_reward", -0.3),
        pin_goal_weight=rc.get("pin_goal_weight", 0.3),
        distance_weight=rc.get("distance_weight", 0.01),
        lagging_weight=rc.get("lagging_weight", -0.005),
        home_exit_weight=rc.get("home_exit_weight", 0.05),
        mobility_weight=rc.get("mobility_weight", 0.001),
        use_score_terminal=rc.get("use_score_terminal", True),
        score_normalization=rc.get("score_normalization", 1300.0),
        use_score_margin=rc.get("use_score_margin", False),
    )


def make_evaluator(model_factory_args, eval_cfg, device="cpu"):
    """Create an evaluation function for the trainer."""
    def evaluator(model):
        agent = ChineseCheckersAgent(model=model, mcts_simulations=20, temperature=0.1,
                                     device=device)
        results = {}
        total_score = 0
        num_baselines = 0
        for name, baseline in [("random", RandomAgent()), ("greedy", GreedyProgressAgent()),
                                ("heuristic", HeuristicAgent())]:
            num_games = eval_cfg.get("num_games", 10)
            r = play_match(agent, baseline, num_games=num_games)
            results[name] = {
                "win_rate": round(r["a_win_rate"], 3),
                "avg_score": round(r["avg_a_score"], 1),
                "avg_game_length": round(r["avg_game_length"], 1),
                "avg_game_time": round(r["avg_game_time"], 2),
                "wins": r["a_wins"],
                "losses": r["b_wins"],
                "draws": r["draws"],
                "num_games": num_games,
            }
            total_score += r["avg_a_score"]
            num_baselines += 1
        results["avg_score"] = round(total_score / max(num_baselines, 1), 1)
        return results
    return evaluator


def main():
    parser = argparse.ArgumentParser(description="Train Chinese Checkers RL agent")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--players", type=int, default=None)
    parser.add_argument("--mcts-sims", type=int, default=None,
                       help="MCTS simulations per move (0 = no MCTS)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint path")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Name for this run (default: auto-generated timestamp)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device: cpu or cuda (auto-detects if not set)")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--freeze-policy", action="store_true",
                       help="Freeze policy head, train only value head")
    parser.add_argument("--kl-anchor", type=str, default=None,
                       help="Checkpoint path for frozen KL anchor model (prevents policy drift)")
    parser.add_argument("--kl-weight", type=float, default=None,
                       help="KL divergence anchor weight (overrides config)")
    parser.add_argument("--phase", type=str, default="unknown",
                       choices=["bootstrap", "mcts_light", "mcts_full", "value_fix", "rl_refine", "unknown"],
                       help="Training phase name (for autonomous tracking)")
    args = parser.parse_args()

    # Auto-detect GPU if not specified
    if args.device is None and torch.cuda.is_available():
        args.device = "cuda"
        print(f"Auto-detected CUDA GPU: {torch.cuda.get_device_name(0)}")

    # Load config
    cfg = load_config(args.config) if os.path.exists(args.config) else {}

    # Build components
    model = build_model(cfg)
    train_cfg = build_training_config(cfg, args)
    if args.run_name:
        train_cfg.run_name = args.run_name
    reward_cfg = build_reward_config(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters, device: {train_cfg.device}")

    # Optional MCTS for self-play
    mcts = None
    if train_cfg.mcts_simulations > 0:
        mc = cfg.get("mcts", {})
        mcts = MCTS(
            c_puct=mc.get("c_puct", 1.5),
            num_simulations=train_cfg.mcts_simulations,
            temperature=mc.get("temperature", 1.0),
            device=train_cfg.device,
        )

    # Resume from checkpoint
    if args.resume:
        checkpoint = Trainer.load_checkpoint(args.resume, model)
        print(f"Resumed from iteration {checkpoint['iteration']}")

    # Load KL anchor model (frozen copy of prior policy)
    anchor_model = None
    if args.kl_anchor:
        anchor_model = build_model(cfg)
        anchor_ckpt = torch.load(args.kl_anchor, map_location="cpu", weights_only=False)
        anchor_model.load_state_dict(anchor_ckpt["model_state_dict"])
        print(f"  KL anchor loaded: {args.kl_anchor} (weight={train_cfg.kl_anchor_weight})")

    # Build opponent if configured
    opponent = None
    opp_name = cfg.get("training", {}).get("opponent")
    if opp_name:
        opp_map = {
            "heuristic": HeuristicAgent,
            "greedy": GreedyProgressAgent,
            "random": RandomAgent,
        }
        if opp_name == "mixed":
            opponent = [RandomAgent(), GreedyProgressAgent(), HeuristicAgent()]
            print(f"  Opponent: mixed pool ({len(opponent)} agents, sampled per iteration)")
        else:
            opp_cls = opp_map.get(opp_name)
            if opp_cls:
                opponent = opp_cls()
                print(f"  Opponent: {opp_name} ({opp_cls.__name__})")
            else:
                print(f"  WARNING: Unknown opponent '{opp_name}', using self-play")

    # Build trainer
    trainer = Trainer(
        model=model,
        config=train_cfg,
        reward_config=reward_cfg,
        mcts=mcts,
        opponent=opponent,
        anchor_model=anchor_model,
    )

    # Set phase and config info for status tracking
    trainer.phase = args.phase
    trainer.config_file = args.config
    trainer.save_run_metadata()

    # Build evaluator
    evaluator = None
    if not args.no_eval:
        evaluator = make_evaluator(cfg, cfg.get("evaluation", {}), device=train_cfg.device)

    # Train with status tracking
    try:
        trainer.train(evaluator=evaluator)

        # Final evaluation (skip if --no-eval)
        if not args.no_eval:
            print("\n=== Final Evaluation ===")
            agent = ChineseCheckersAgent(model=model, mcts_simulations=50, temperature=0.1,
                                         device=train_cfg.device)
            evaluate_agent(agent, num_games=10, verbose=True)

    except Exception as e:
        # Write failed status so the autonomous agent knows
        status_file = os.path.join(os.path.dirname(__file__), "training_status.json")
        with open(status_file, "w") as f:
            json.dump({
                "status": "failed",
                "phase": args.phase,
                "run_name": train_cfg.run_name,
                "config_file": args.config,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "failed_at": datetime.now().isoformat(),
                "current_iteration": trainer.iteration,
                "total_iterations": train_cfg.num_iterations,
            }, f, indent=2)
        print(f"\nTRAINING FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
