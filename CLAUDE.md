# Repo notes for Claude

## Training environment split

Training runs on UiA school computers (CUDA). The local repo only receives **logs**, not checkpoints.

- `logs/<run_name>/` (run_metadata.json, metrics.jsonl, eval.jsonl, console.log) → synced to local
- `checkpoints/<run_name>/*.pt` → **stays on the school machine**, not synced

The local `checkpoints/` directory contains only stale March files. Do **not** interpret a missing local `.pt` as "checkpoint lost" — verify with the user before claiming weights are gone. Resume paths like `--resume checkpoints/<run>/model_best.pt` are valid on the school machine even when they don't exist locally.

When `console.log` shows `Checkpoint saved: checkpoints/<run>/model_*.pt`, treat that as evidence the file exists on the school side.
