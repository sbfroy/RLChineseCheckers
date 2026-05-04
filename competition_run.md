# Competition / test-run guide

How to connect the trained agent to the course's game server. Used for the
2026-05-04 test run and the 2026-05-22 tournament.

---

## What you have

- **Game server** (`game.py`) — at `multi system single machine minimal/game.py`
  in this repo. Exposes JSON-RPC on TCP port `50555`. For the tournament,
  the instructor hosts it on their machine; for a local self-test you can
  run it yourself.
- **Player client** (`env/server_adapter.py`) — your code. Connects to the
  server, joins as a player, and uses your trained checkpoint to pick moves.
- **Checkpoint** — `checkpoints/phase1_v6/model_best.pt` on the
  competition laptop. (The 2026-05-04 test run confirmed the agent must
  run from the student's own machine, not the school CUDA box.)

The competition parameters (`c_puct=1.0`, `temperature=0.3`, `mcts-sims=15`)
are baked in as defaults in `env/server_adapter.py`, so the launch command
only needs `--checkpoint`, `--device`, and (if remote) `--host`/`--port`.

`mcts-sims=15` was picked from the 2026-05-04 CPU sweep on the user's
laptop (i7-1165G7) after the 2026-05-04 test run revealed the tournament
must run on the student's own machine, not the school CUDA box. At
sims=15 a single move is ~1.0 s on CPU — well under the 10 s/turn cap and
fast enough to fit ~120-move 2P games inside the 60 s game wall-clock.
Higher sim counts maxed 2P quality but spilled the budget; lower counts
stalled games to max-moves. See `logs/sweep/20260504_194538/` for the
full pin/score table.

---

## Before the test run (do once)

On the competition laptop, pull the latest changes and confirm the file
exists:

```bash
git pull
ls -lh checkpoints/phase1_v6/model_best.pt
```

If the checkpoint is missing, the run won't start.

---

## Connecting to the game

### 1. If the game server runs on the same machine (default)

Two terminals on the laptop:

**Terminal 1 — start the course's game server:**

```bash
python3 "multi system single machine minimal/game.py"
```

Type `Create` + Enter to create a game.

**Terminal 2 — start your agent:**

```bash
venv/bin/python env/server_adapter.py \
  --name <your-team-name> \
  --checkpoint checkpoints/phase1_v6/model_best.pt
```

That's it. The agent joins, waits for the game to be `READY_TO_START`,
sends `start`, then plays its turns automatically. You should see log lines
like:

```
==== <your-team-name> ====
Joined game <id> as <colour>
=== GAME STARTED ===
Loading agent for 4P: checkpoint=checkpoints/phase1_v6/model_best.pt, mcts_sims=15, c_puct=1.0, temperature=0.3, root_noise=OFF
```

### 2. If the game server runs on a different machine (tournament)

Same launch command, plus `--host` and `--port`:

```bash
venv/bin/python env/server_adapter.py \
  --name AlphaZeroClue \
  --checkpoint checkpoints/phase1_v6/model_best.pt \
  --host 10.245.30.227 \
  --port 50555
```


The instructor will share the host/port for the test run.

---

## Smoke test before showing up to the test run

### Option A — full JSON-RPC self-test

Three terminals on the laptop:

**Terminal 1 — server:**

```bash
python3 "multi system single machine minimal/game.py"
```

Type `Create`, accept defaults until a game with 2 players is set up.

**Terminal 2 — your agent (player 1):**

```bash
venv/bin/python env/server_adapter.py \
  --name TestRL \
  --checkpoint checkpoints/phase1_v6/model_best.pt
```

**Terminal 3 — anything that joins as a second player.** Easiest is the
course-provided `player.py` controlled by hand, or a second instance of
your own agent under a different name:

```bash
venv/bin/python env/server_adapter.py \
  --name TestRL2 \
  --checkpoint checkpoints/phase1_v6/model_best.pt
```

What to verify:

- Both clients log `Joined game ...`, then `=== GAME STARTED ===`.
- Move latency is ~1.0 s after the first move (the first move loads the
  checkpoint and is slower). At sims=15 on CPU each move is roughly
  15 × 70 ms ≈ 1 s.
- If a move would repeat a recent one, you see a `[HEURISTIC]` tag in the
  log — that's the anti-oscillation fallback firing.
- Game finishes with `=== GAME FINISHED ===` and a score table.

If any of these fail, fix before tomorrow.

### Option B — local play simulator (no JSON-RPC)

This skips the JSON-RPC layer but exercises the trained agent end-to-end
against the bundled greedy/heuristic opponents. Useful as a quick health
check before the full self-test.

```bash
venv/bin/python play.py watch \
  --checkpoint checkpoints/phase1_v6/model_best.pt \
  --opponent greedy \
  --mcts-sims 15 \
  --device cpu \
  --delay 0.1
```

You'll see an ASCII board update each move. Expected: the agent moves
its pins steadily toward the opposite corner and finishes a 2P game in
~120 moves. If it stalls, oscillates, or fails to load the checkpoint,
fix before the real test run.

For multi-player practice (since the tournament uses 2/4/6 players):

```bash
venv/bin/python validate_multiplayer.py \
  --checkpoint checkpoints/phase1_v6/model_best.pt \
  --device cpu \
  --mcts-sims 15 \
  --players 2 4 6
```

---

## Useful flags (only if you need them)

All of these are optional — defaults are the tournament-locked config.

| flag | default | when to override |
|---|---|---|
| `--mcts-sims` | 15 | CPU sweep winner on the laptop. Don't raise on CPU — sims=20 was already slower and worse on 4P; sims=10 lost 2P pins. Only raise if running on CUDA (then 100 is the original CUDA-validated value). |
| `--c-puct` | 1.0 | Don't change — this is the sweep winner. |
| `--temperature` | 0.3 | `0.0` to play deterministically; `0.5` for more diversity. Sweep showed 0.3 is the sweet spot. |
| `--time-limit` | none | Soft per-move time limit in seconds. Leave unset unless required. |
| `--device` | `cpu` | Tournament runs on the laptop — keep `cpu`. `cuda` only if testing on the school machine. |
| `--dirichlet-alpha` / `--root-noise-epsilon` | 0 / 0 | Leave off. Only enable if opponents start memorizing openings. |
| `--checkpoint-2p` / `--checkpoint-4p` / `--checkpoint-6p` | none | Set per-player-count checkpoint overrides if you ever want different weights for different game sizes. Currently we use the same checkpoint everywhere. |

---

## What can go wrong

- **`connect-failed`** in the log — server isn't running, or wrong host/port.
- **`JOIN ERROR`** — server is full or already finished. Restart the server.
- **`No module named 'torch'`** — system `python3` doesn't have the deps.
  Use `venv/bin/python` (or `source venv/bin/activate` first).
- **Slow first move** — expected. The checkpoint loads on the first turn.
  If the *second* move is much slower than ~1 s, something is wrong
  (likely thread contention or a different model loaded).
- **Agent makes the same move repeatedly** — the anti-oscillation fallback
  should prevent this; if you see it, capture the log and report.

---

## Tournament-day checklist (2026-05-22)

1. `git pull` on the laptop — make sure code is current.
2. `ls checkpoints/phase1_v6/model_best.pt` — confirm checkpoint present.
3. Get host/port from the instructor.
4. Run the launch command with `--host` / `--port`.
5. Watch the log for `=== GAME STARTED ===`, then leave it alone.
