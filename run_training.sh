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

# Generate run name from timestamp (matches trainer's auto-naming)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="run_${TIMESTAMP}"

# Console log goes alongside the run's JSON logs in logs/<run_name>/
mkdir -p "logs/${RUN_NAME}"
LOGFILE="logs/${RUN_NAME}/console.log"

echo "============================================="
echo "  Chinese Checkers RL Training"
echo "  Config: ${CONFIG}"
echo "  Run:    ${RUN_NAME}"
echo "  Log:    ${LOGFILE}"
echo "  Args:   $@"
echo "  Time:   $(date)"
echo "============================================="
echo ""

# Run training, pipe to both terminal and log file
# Pass --run-name so trainer uses the same directory
python train.py --config "${CONFIG}" --run-name "${RUN_NAME}" "$@" 2>&1 | tee "${LOGFILE}"

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
