# Repo notes for Claude

## Training environment split

Training runs on UiA school computers (CUDA). The local repo only receives **logs**, not checkpoints.

- `logs/<run_name>/` (run_metadata.json, metrics.jsonl, eval.jsonl, console.log) → synced to local
- `checkpoints/<run_name>/*.pt` → **stays on the school machine**, not synced

The local `checkpoints/` directory contains only stale March files. Do **not** interpret a missing local `.pt` as "checkpoint lost" — verify with the user before claiming weights are gone. Resume paths like `--resume checkpoints/<run>/model_best.pt` are valid on the school machine even when they don't exist locally.

When `console.log` shows `Checkpoint saved: checkpoints/<run>/model_*.pt`, treat that as evidence the file exists on the school side.

## Hand-off rule: the user only runs the command

The user's only job when a run is recommended is to execute the final `./run_training.sh ...` command on the school machine. Do **not** ask them to:

- Edit any file (configs, code, docs)
- Run `cp` / archive commands
- Run `git` operations
- Change directories or set env vars
- Any other preparatory shell step

Claude does all of that directly via the Edit/Write/Bash tools in the local repo (configs and code are git-tracked; the school machine picks up changes via git pull). The assistant's final message should contain exactly one code block: the single command the user runs. Everything else — archiving the old config, editing YAML/code, updating the journal — must already be done by the time the user reads the response.

If a change can't be made from the local repo (e.g. something truly school-machine-only), say so explicitly and offer the single command that performs it, rather than handing the user a checklist.

## Command formatting: always use backslash line continuations

When giving the user a shell command, always split it across multiple lines using backslash (`\`) continuations, one flag per line. The user's terminal/paste path sometimes inserts newlines at arbitrary points inside a single-line command, which breaks flags like `--mcts-sims 20` off onto their own line and produces `command not found` errors. Backslash form is safe regardless of how the paste is wrapped — bash always reassembles it into one command.

Do this:

```bash
python3 play.py watch \
  --checkpoint checkpoints/sup_20260418_090744/model_best.pt \
  --opponent greedy \
  --mcts-sims 20 \
  --device cuda \
  --delay 0.1
```

Not this:

```bash
python3 play.py watch --checkpoint checkpoints/sup_20260418_090744/model_best.pt --opponent greedy --mcts-sims 20 --device cuda --delay 0.1
```

The rule applies to every user-runnable command: training launches, evals, diagnostics, anything pasted into the school machine's shell.
