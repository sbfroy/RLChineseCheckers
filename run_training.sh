#!/bin/bash
# =============================================================================
# run_training.sh — Wrapper for train.py with log capture
#
# Usage (inside tmux):
#   ./run_training.sh configs/fast_bootstrap.yaml --phase bootstrap
#   ./run_training.sh configs/mcts_light.yaml --phase mcts_light --resume checkpoints/bootstrap/run_.../model_best.pt
#   ./run_training.sh configs/default.yaml --phase mcts_full --resume checkpoints/run_.../model_best.pt
# =============================================================================

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.yaml> [--phase PHASE] [--resume CHECKPOINT] [extra args...]"
    echo ""
    echo "Phases: bootstrap, mcts_light, mcts_full"
    echo ""
    echo "Examples:"
    echo "  $0 configs/fast_bootstrap.yaml --phase bootstrap"
    echo "  $0 configs/mcts_light.yaml --phase mcts_light --resume checkpoints/bootstrap/run_.../model_best.pt"
    exit 1
fi

CONFIG="$1"
shift

# Create logs directory
mkdir -p logs

# Generate log filename from timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/train_${TIMESTAMP}.log"

echo "============================================="
echo "  Chinese Checkers RL Training"
echo "  Config: ${CONFIG}"
echo "  Log:    ${LOGFILE}"
echo "  Args:   $@"
echo "  Time:   $(date)"
echo "============================================="
echo ""

# Run training, pipe to both terminal and log file
python3.10 train.py --config "${CONFIG}" "$@" 2>&1 | tee "${LOGFILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Training completed successfully"
else
    echo "  Training FAILED (exit code: ${EXIT_CODE})"
fi
echo "  Log saved: ${LOGFILE}"
echo "  Time: $(date)"
echo "============================================="

exit $EXIT_CODE
