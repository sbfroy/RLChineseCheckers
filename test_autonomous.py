#!/usr/bin/env python3.10
"""
Dry-run test for the autonomous training system.

Runs a tiny training (3 iterations, 2 games, no MCTS), then verifies:
1. training_status.json was created and shows "finished"
2. metrics.jsonl has 3 entries
3. Checkpoints were saved
4. Simulates an agent check-in: reads status + metrics, applies rules

Usage:
    python3.10 test_autonomous.py
"""

import json
import os
import shutil
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STATUS_FILE = os.path.join(PROJECT_ROOT, "training_status.json")
TEST_CONFIG_FILE = os.path.join(PROJECT_ROOT, "configs", "_test_autonomous.yaml")

# Test parameters
TEST_ITERATIONS = 3
TEST_GAMES = 1
TEST_PHASE = "bootstrap"
TEST_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "test_autonomous")


def print_section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}\n")


def cleanup():
    """Remove test artifacts."""
    # Remove test checkpoints
    if os.path.exists(TEST_CHECKPOINT_DIR):
        shutil.rmtree(TEST_CHECKPOINT_DIR)
    # Remove status file
    if os.path.exists(STATUS_FILE):
        os.remove(STATUS_FILE)
    # Remove test log dirs (find dirs starting with "run_" in logs/ that were just created)
    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    if os.path.exists(logs_dir):
        for entry in os.listdir(logs_dir):
            path = os.path.join(logs_dir, entry)
            # Only remove directories that are test runs (created in last 2 minutes)
            if os.path.isdir(path) and time.time() - os.path.getmtime(path) < 120:
                shutil.rmtree(path)
        # Remove log files created in last 2 minutes
        for entry in os.listdir(logs_dir):
            path = os.path.join(logs_dir, entry)
            if os.path.isfile(path) and time.time() - os.path.getmtime(path) < 120:
                os.remove(path)


def create_test_config():
    """Create a minimal config for fast testing."""
    import yaml
    config = {
        "model": {
            "num_res_blocks": 1,
            "trunk_channels": 16,
            "policy_channels": 4,
            "value_hidden": 16,
        },
        "training": {
            "num_iterations": TEST_ITERATIONS,
            "num_games_per_iteration": TEST_GAMES,
            "num_players": 2,
            "batch_size": 32,
            "epochs_per_iteration": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "buffer_capacity": 10000,
            "min_buffer_size": 16,
            "temperature": 1.0,
            "checkpoint_dir": "checkpoints/test_autonomous",
            "checkpoint_every": 2,
            "eval_every": 3,
            "device": "cpu",
        },
        "mcts": {
            "num_simulations": 0,
            "c_puct": 1.5,
            "temperature": 1.0,
        },
        "rewards": {
            "win_reward": 1.0,
            "loss_reward": -1.0,
            "distance_weight": 0.01,
        },
        "evaluation": {
            "num_games": 2,
        },
    }
    with open(TEST_CONFIG_FILE, "w") as f:
        yaml.safe_dump(config, f)
    return TEST_CONFIG_FILE


def run_training():
    """Run a tiny training to test the system."""
    print_section("Step 1: Running tiny training")

    config_path = create_test_config()

    cmd = [
        sys.executable, "train.py",
        "--config", config_path,
        "--mcts-sims", "0",
        "--phase", TEST_PHASE,
        "--no-eval",
        "--run-name", "test_autonomous_run",
    ]

    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    print("STDOUT (last 20 lines):")
    for line in result.stdout.strip().split("\n")[-20:]:
        print(f"  {line}")

    if result.returncode != 0:
        print(f"\nSTDERR:\n{result.stderr}")
        print("\nFAILED: Training exited with non-zero code")
        return False

    print("\nTraining completed successfully")
    return True


def check_status_file():
    """Verify training_status.json exists and is correct."""
    print_section("Step 2: Checking training_status.json")

    if not os.path.exists(STATUS_FILE):
        print("FAILED: training_status.json does not exist")
        return False

    with open(STATUS_FILE) as f:
        status = json.load(f)

    print(f"  status:            {status.get('status')}")
    print(f"  phase:             {status.get('phase')}")
    print(f"  run_name:          {status.get('run_name')}")
    print(f"  current_iteration: {status.get('current_iteration')}")
    print(f"  total_iterations:  {status.get('total_iterations')}")
    print(f"  latest_metrics:    {status.get('latest_metrics')}")
    print(f"  started_at:        {status.get('started_at')}")
    print(f"  finished_at:       {status.get('finished_at')}")

    checks = [
        ("status is 'finished'", status.get("status") == "finished"),
        ("phase is 'bootstrap'", status.get("phase") == TEST_PHASE),
        ("current_iteration == 3", status.get("current_iteration") == TEST_ITERATIONS),
        ("latest_metrics exists", status.get("latest_metrics") is not None),
        ("finished_at is set", status.get("finished_at") is not None),
        ("pid is set", status.get("pid") is not None),
    ]

    all_ok = True
    print()
    for name, passed in checks:
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {name}")
        if not passed:
            all_ok = False

    return all_ok


def check_metrics_log():
    """Verify metrics.jsonl was created with correct entries."""
    print_section("Step 3: Checking metrics.jsonl")

    log_dir = os.path.join(PROJECT_ROOT, "logs", "test_autonomous_run")
    metrics_file = os.path.join(log_dir, "metrics.jsonl")

    if not os.path.exists(metrics_file):
        print(f"FAILED: {metrics_file} does not exist")
        return False

    with open(metrics_file) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    print(f"  Found {len(lines)} metric entries")
    print()

    for entry in lines:
        print(
            f"  Iter {entry['iteration']:3d} | "
            f"p_loss={entry['policy_loss']:.4f} | "
            f"v_loss={entry['value_loss']:.4f} | "
            f"buf={entry['buffer_size']}"
        )

    checks = [
        (f"has {TEST_ITERATIONS} entries", len(lines) == TEST_ITERATIONS),
        ("all entries have iteration", all("iteration" in e for e in lines)),
        ("all entries have timestamp", all("timestamp" in e for e in lines)),
        ("all entries have policy_loss", all("policy_loss" in e for e in lines)),
        ("all entries have value_loss", all("value_loss" in e for e in lines)),
        ("iterations are sequential", [e["iteration"] for e in lines] == list(range(1, TEST_ITERATIONS + 1))),
    ]

    all_ok = True
    print()
    for name, passed in checks:
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {name}")
        if not passed:
            all_ok = False

    return all_ok


def check_checkpoints():
    """Verify checkpoints were saved."""
    print_section("Step 4: Checking checkpoints")

    # Checkpoints go to: checkpoints/test_autonomous/test_autonomous_run/
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", "test_autonomous", "test_autonomous_run")

    if not os.path.exists(checkpoint_dir):
        print(f"FAILED: Checkpoint dir does not exist: {checkpoint_dir}")
        return False

    files = os.listdir(checkpoint_dir)
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Files: {files}")

    checks = [
        ("final checkpoint exists", "model_final.pt" in files),
    ]

    all_ok = True
    print()
    for name, passed in checks:
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {name}")
        if not passed:
            all_ok = False

    return all_ok


def simulate_agent_checkin():
    """Simulate what the autonomous agent would do on a check-in."""
    print_section("Step 5: Simulating agent check-in")

    # 1. Read status
    with open(STATUS_FILE) as f:
        status = json.load(f)

    print(f"  Agent reads status: {status['status']}")
    print(f"  Phase: {status['phase']}")

    # 2. Read metrics
    log_dir = os.path.join(PROJECT_ROOT, "logs", "test_autonomous_run")
    metrics_file = os.path.join(log_dir, "metrics.jsonl")
    with open(metrics_file) as f:
        metrics = [json.loads(line) for line in f if line.strip()]

    last = metrics[-1]
    print(f"  Latest policy_loss: {last['policy_loss']:.4f}")
    print(f"  Latest value_loss:  {last['value_loss']:.4f}")

    # 3. Apply rules (simplified — in real run the agent reads autonomous_rules.md)
    print("\n  Applying decision rules...")

    # Since this is only 3 iterations, criteria won't be met — that's expected
    iterations_enough = status["current_iteration"] >= 80
    loss_low_enough = last["policy_loss"] < 3.5

    print(f"  Iterations >= 80?  {'YES' if iterations_enough else 'NO'} ({status['current_iteration']})")
    print(f"  Policy loss < 3.5? {'YES' if loss_low_enough else 'NO'} ({last['policy_loss']:.4f})")
    print(f"  (No eval data in this test run — skipping win rate check)")

    print(f"\n  Decision: Training finished but criteria not met (expected for 3-iter test).")
    print(f"  In a real run, the agent would continue training or adjust config.")

    # 4. Write test journal entry
    journal_path = os.path.join(PROJECT_ROOT, "training_journal.md")
    with open(journal_path, "a") as f:
        f.write(f"## [TEST] Simulated Check-in\n\n")
        f.write(f"**Status:** {status['status']}\n")
        f.write(f"**Phase:** {status['phase']}\n")
        f.write(f"**Run:** {status['run_name']}\n\n")
        f.write(f"**Observations:**\n")
        f.write(f"- Iteration: {status['current_iteration']} / {status['total_iterations']}\n")
        f.write(f"- Policy loss: {last['policy_loss']:.4f}\n")
        f.write(f"- Value loss: {last['value_loss']:.4f}\n\n")
        f.write(f"**Decision:** Test run — no action needed\n\n")
        f.write(f"**Action taken:** None (this was a test)\n\n")
        f.write(f"---\n\n")

    print(f"\n  Journal entry written to {journal_path}")
    return True


def main():
    print("=" * 50)
    print("  Autonomous Training System — Dry Run Test")
    print("=" * 50)

    results = {}

    # Pre-test cleanup (remove artifacts from previous failed runs)
    for path in [
        os.path.join(PROJECT_ROOT, "logs", "test_autonomous_run"),
        os.path.join(PROJECT_ROOT, "checkpoints", "test_autonomous"),
        STATUS_FILE,
    ]:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)

    # Run the test training
    results["training"] = run_training()
    if not results["training"]:
        print("\nTraining failed — cannot continue tests")
        cleanup()
        sys.exit(1)

    # Run all checks
    results["status_file"] = check_status_file()
    results["metrics_log"] = check_metrics_log()
    results["checkpoints"] = check_checkpoints()
    results["agent_sim"] = simulate_agent_checkin()

    # Summary
    print_section("Summary")

    all_passed = True
    for name, passed in results.items():
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  All tests PASSED! The autonomous system is ready.")
        print()
        print("  Next steps:")
        print("  1. Clean up test artifacts (see below)")
        print("  2. Start real training in tmux:")
        print("     tmux new -s training")
        print("     ./run_training.sh configs/fast_bootstrap.yaml --phase bootstrap")
        print("  3. Set up the scheduled agent")
    else:
        print("  Some tests FAILED. Fix the issues before proceeding.")

    # Ask about cleanup
    print()
    print("  Test artifacts created:")
    print(f"    - checkpoints/bootstrap/test_autonomous_run/")
    print(f"    - logs/test_autonomous_run/")
    print(f"    - training_status.json")
    print(f"    - training_journal.md (test entry appended)")

    # Auto-cleanup test artifacts
    test_checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", "test_autonomous")
    if os.path.exists(test_checkpoint_dir):
        shutil.rmtree(test_checkpoint_dir)
    if os.path.exists(TEST_CONFIG_FILE):
        os.remove(TEST_CONFIG_FILE)
    test_log_dir = os.path.join(PROJECT_ROOT, "logs", "test_autonomous_run")
    if os.path.exists(test_log_dir):
        shutil.rmtree(test_log_dir)
    if os.path.exists(STATUS_FILE):
        os.remove(STATUS_FILE)

    # Remove the test journal entry (rewrite without it)
    journal_path = os.path.join(PROJECT_ROOT, "training_journal.md")
    with open(journal_path, "w") as f:
        f.write("# Training Journal\n\n")
        f.write("This file is maintained by the autonomous training agent.\n")
        f.write("Each entry records a check-in: what was observed, what was decided, and what action was taken.\n\n")
        f.write("---\n\n")

    print("\n  Cleaned up test artifacts.")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
