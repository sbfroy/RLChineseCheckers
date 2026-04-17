"""
Training loop for the policy-value network.

Trains on self-play data from the replay buffer.
Loss = policy_loss (cross-entropy with MCTS targets) + value_loss (MSE).
"""

import json
import os
import sys
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.policy_value_net import PolicyValueNet
from models.encoders import BoardEncoder
from training.replay_buffer import ReplayBuffer
from training.self_play import generate_self_play_data
from env.rewards import RewardConfig


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Self-play
    num_games_per_iteration: int = 20
    num_players: int = 2
    temperature: float = 1.0
    mcts_simulations: int = 50

    # Training
    batch_size: int = 128
    epochs_per_iteration: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    # Replay buffer
    buffer_capacity: int = 100_000
    min_buffer_size: int = 256

    # Training loop
    num_iterations: int = 100
    eval_every: int = 10
    checkpoint_every: int = 10
    checkpoint_dir: str = "checkpoints"
    run_name: str = ""  # Auto-generated if empty (e.g. "run_20260326_143022")

    # Device
    device: str = "cpu"


class Trainer:
    """
    AlphaZero-style training loop.

    Each iteration:
    1. Generate self-play data
    2. Add to replay buffer
    3. Train on random batches from buffer
    4. Periodically evaluate and checkpoint
    """

    def __init__(
        self,
        model: PolicyValueNet,
        config: Optional[TrainingConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        mcts=None,
        opponent=None,
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self.reward_config = reward_config or RewardConfig()
        self.mcts = mcts
        self.opponent = opponent
        self.encoder = BoardEncoder()
        self.buffer = ReplayBuffer(capacity=self.config.buffer_capacity)

        # Move model to device
        self.model.to(self.config.device)

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_iterations,
            eta_min=1e-5,
        )

        # Run name for checkpoint isolation
        if not self.config.run_name:
            self.config.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = os.path.join(self.config.checkpoint_dir, self.config.run_name)

        # Best model tracking
        self.best_eval_score = float("-inf")

        # Logging
        self.train_history = []
        self.iteration = 0

        # JSON logging directory
        self.log_dir = os.path.join("logs", self.config.run_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Status file path (project root)
        self.status_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "training_status.json"
        )

        # Phase tracking (set externally via train.py)
        self.phase = "unknown"
        self.config_file = ""

    def save_run_metadata(self):
        """Save full config and model info as run_metadata.json in the log dir.
        Call this after setting phase and config_file."""
        from dataclasses import asdict
        metadata = {
            "run_name": self.config.run_name,
            "started_at": datetime.now().isoformat(),
            "phase": self.phase,
            "config_file": self.config_file,
            "device": self.config.device,
            "training_config": asdict(self.config),
            "reward_config": asdict(self.reward_config),
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "mcts_enabled": self.mcts is not None,
            "mcts_simulations": self.config.mcts_simulations,
            "opponent": type(self.opponent).__name__ if self.opponent else None,
        }
        path = os.path.join(self.log_dir, "run_metadata.json")
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _write_status(self, **overrides):
        """Write/update training_status.json."""
        status = {
            "status": "running",
            "phase": self.phase,
            "run_name": self.config.run_name,
            "config_file": self.config_file,
            "started_at": self._started_at,
            "finished_at": None,
            "pid": os.getpid(),
            "current_iteration": self.iteration,
            "total_iterations": self.config.num_iterations,
            "latest_metrics": None,
            "latest_eval": None,
        }
        status.update(overrides)
        with open(self.status_file, "w") as f:
            json.dump(status, f, indent=2)

    def _log_metrics(self, iteration, metrics, sp_time, train_time, lr, new_exp,
                     game_stats=None):
        """Append one line to metrics.jsonl."""
        entry = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "buffer_size": len(self.buffer),
            "sp_time": round(sp_time, 2),
            "train_time": round(train_time, 2),
            "policy_loss": round(metrics["policy_loss"], 6),
            "value_loss": round(metrics["value_loss"], 6),
            "total_loss": round(metrics["total_loss"], 6),
            "lr": lr,
            "new_experiences": new_exp,
        }
        if game_stats:
            entry["avg_game_length"] = game_stats["avg_game_length"]
            entry["max_moves_pct"] = game_stats["max_moves_pct"]
        with open(os.path.join(self.log_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def _log_eval(self, iteration, eval_results):
        """Append one line to eval.jsonl."""
        entry = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "results": eval_results,
        }
        with open(os.path.join(self.log_dir, "eval.jsonl"), "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def train(self, evaluator=None):
        """Run the full training loop."""
        cfg = self.config
        os.makedirs(self.run_dir, exist_ok=True)
        self._started_at = datetime.now().isoformat()

        print(f"Starting training: {cfg.num_iterations} iterations")
        print(f"  Run: {cfg.run_name} -> {self.run_dir}")
        print(f"  Self-play: {cfg.num_games_per_iteration} games/iter, {cfg.num_players} players")
        print(f"  Training: batch={cfg.batch_size}, epochs={cfg.epochs_per_iteration}")
        print(f"  MCTS: {'enabled' if self.mcts else 'disabled'} ({cfg.mcts_simulations} sims)")
        print(f"  Logs: {self.log_dir}")
        print()

        self._write_status()

        latest_eval = None

        for iteration in range(1, cfg.num_iterations + 1):
            self.iteration = iteration
            iter_start = time.time()

            # 1. Self-play (or vs opponent)
            self.model.eval()
            experiences, game_stats = generate_self_play_data(
                model=self.model,
                encoder=self.encoder,
                num_games=cfg.num_games_per_iteration,
                num_players=cfg.num_players,
                temperature=cfg.temperature,
                mcts=self.mcts,
                mcts_simulations=cfg.mcts_simulations,
                reward_config=self.reward_config,
                device=cfg.device,
                opponent=self.opponent,
            )
            self.buffer.push_batch(experiences)
            sp_time = time.time() - iter_start

            # 2. Train
            train_start = time.time()
            if len(self.buffer) >= cfg.min_buffer_size:
                metrics = self._train_epoch()
            else:
                metrics = {"policy_loss": 0, "value_loss": 0, "total_loss": 0}
            train_time = time.time() - train_start

            self.scheduler.step()

            # Log
            lr = self.optimizer.param_groups[0]["lr"]
            total_time = time.time() - iter_start
            print(
                f"Iter {iteration:4d} | "
                f"buf={len(self.buffer):6d} | "
                f"sp={sp_time:.1f}s | "
                f"train={train_time:.1f}s | "
                f"p_loss={metrics['policy_loss']:.4f} | "
                f"v_loss={metrics['value_loss']:.4f} | "
                f"lr={lr:.2e} | "
                f"new_exp={len(experiences)}"
            )
            self.train_history.append(metrics)

            # JSON logging
            metrics_entry = self._log_metrics(
                iteration, metrics, sp_time, train_time, lr, len(experiences),
                game_stats=game_stats,
            )

            # 3. Evaluate
            if evaluator and iteration % cfg.eval_every == 0:
                eval_results = evaluator(self.model)
                print(f"  EVAL: {eval_results}")
                self._log_eval(iteration, eval_results)
                latest_eval = eval_results

                # Track best model by eval score
                eval_score = None
                if isinstance(eval_results, dict):
                    eval_score = eval_results.get("avg_score", eval_results.get("score"))
                if eval_score is not None and eval_score > self.best_eval_score:
                    self.best_eval_score = eval_score
                    self._save_checkpoint(iteration, tag="best")
                    print(f"  NEW BEST model (score={eval_score:.1f})")

            # Update status file
            self._write_status(
                current_iteration=iteration,
                latest_metrics={
                    "policy_loss": metrics_entry["policy_loss"],
                    "value_loss": metrics_entry["value_loss"],
                },
                latest_eval=latest_eval,
            )

            # 4. Checkpoint
            if iteration % cfg.checkpoint_every == 0:
                self._save_checkpoint(iteration)

        # Final checkpoint
        self._save_checkpoint(self.iteration, final=True)

        # Mark finished
        self._write_status(
            status="finished",
            current_iteration=self.iteration,
            finished_at=datetime.now().isoformat(),
            latest_metrics={
                "policy_loss": self.train_history[-1]["policy_loss"],
                "value_loss": self.train_history[-1]["value_loss"],
            },
            latest_eval=latest_eval,
        )

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch on replay buffer data."""
        cfg = self.config
        self.model.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for _ in range(cfg.epochs_per_iteration):
            spatial, scalars, masks, policy_targets, value_targets = self.buffer.sample(
                cfg.batch_size
            )

            spatial = spatial.to(cfg.device)
            scalars = scalars.to(cfg.device)
            masks = masks.to(cfg.device)
            policy_targets = policy_targets.to(cfg.device)
            value_targets = value_targets.to(cfg.device)

            # Forward pass
            policy_logits, value_pred = self.model(spatial, scalars, masks)

            # Policy loss: cross-entropy with MCTS policy targets
            log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_loss = -torch.sum(policy_targets * log_probs, dim=-1).mean()

            # Value loss: MSE
            value_loss = F.mse_loss(value_pred, value_targets)

            # Total loss
            loss = (cfg.policy_loss_weight * policy_loss +
                    cfg.value_loss_weight * value_loss)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        return {
            "policy_loss": total_policy_loss / max(num_batches, 1),
            "value_loss": total_value_loss / max(num_batches, 1),
            "total_loss": (total_policy_loss + total_value_loss) / max(num_batches, 1),
        }

    def _save_checkpoint(self, iteration: int, final: bool = False, tag: str = ""):
        """Save model checkpoint into the run directory."""
        if tag:
            name = tag
        elif final:
            name = "final"
        else:
            name = f"iter_{iteration:05d}"
        path = os.path.join(self.run_dir, f"model_{name}.pt")
        torch.save({
            "iteration": iteration,
            "run_name": self.config.run_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "buffer_size": len(self.buffer),
            "train_history": self.train_history,
        }, path)
        print(f"  Checkpoint saved: {path}")

    @staticmethod
    def load_checkpoint(
        path: str,
        model: PolicyValueNet,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> Dict:
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint
