# Competition / test-run guide

How to connect the trained agent to the course's game server. Used for the
2026-05-04 test run and the 2026-05-22 tournament.

---

## What you have

- **Game server** (`game.py`) — **NOT in this repo.** Distributed by the
  course (Canvas / course Git repo / instructor). Exposes JSON-RPC on a
  TCP port (default `127.0.0.1:50555`). For the tournament, the instructor
  will host it; for a local self-test you need to drop their `game.py` into
  this directory.
- **Player client** (`env/server_adapter.py`) — your code. Connects to the
  server, joins as a player, and uses your trained checkpoint to pick moves.
- **Checkpoint** — `checkpoints/phase1_v6/model_best.pt` on the school
  machine.

If you don't have `game.py`, skip to
["Local-only smoke test (no game.py needed)"](#local-only-smoke-test-no-gamepy-needed)
below to at least confirm the agent plays correctly against the bundled
heuristic baseline.

The competition parameters (`c_puct=1.0`, `temperature=0.3`, `mcts-sims=100`)
are baked in as defaults in `env/server_adapter.py`, so the launch command
only needs `--checkpoint`, `--device`, and (if remote) `--host`/`--port`.

---

## Before the test run (do once)

On the school machine, pull the latest changes and confirm the file exists:

```bash
git pull
ls -lh checkpoints/phase1_v6/model_best.pt
```

If the checkpoint is missing, the run won't start.

---

## Connecting to the game

### 1. If the game server runs on the same machine (default)

Two terminals on the school machine:

**Terminal 1 — start the course's game server:**

```bash
python3 game.py
```

Type `Create` + Enter to create a game.

**Terminal 2 — start your agent:**

```bash
python3 env/server_adapter.py \
  --name <your-team-name> \
  --checkpoint checkpoints/phase1_v6/model_best.pt \
  --device cuda
```

That's it. The agent joins, waits for the game to be `READY_TO_START`,
sends `start`, then plays its turns automatically. You should see log lines
like:

```
==== <your-team-name> ====
Joined game <id> as <colour>
=== GAME STARTED ===
Loading agent for 4P: checkpoint=checkpoints/phase1_v6/model_best.pt, mcts_sims=100, c_puct=1.0, temperature=0.3, root_noise=OFF
```

### 2. If the game server runs on a different machine (tournament)

Same launch command, plus `--host` and `--port`:

```bash
python3 env/server_adapter.py \
  --name <your-team-name> \
  --checkpoint checkpoints/phase1_v6/model_best.pt \
  --device cuda \
  --host <server-ip> \
  --port <server-port>
```

The instructor will share the host/port for the test run.

---

## Smoke test before showing up to the test run

### Option A — full JSON-RPC self-test (requires `game.py`)

Drop the course's `game.py` into this directory, then use three terminals:

**Terminal 1 — server:**

```bash
python3 game.py
```

Type `Create`, accept defaults until a game with 2 players is set up.

**Terminal 2 — your agent (player 1):**

```bash
python3 env/server_adapter.py \
  --name TestRL \
  --checkpoint checkpoints/phase1_v6/model_best.pt \
  --device cuda
```

**Terminal 3 — anything that joins as a second player.** Easiest is the
course-provided `player.py` controlled by hand, or a second instance of
your own agent under a different name:

```bash
python3 env/server_adapter.py \
  --name TestRL2 \
  --checkpoint checkpoints/phase1_v6/model_best.pt \
  --device cuda
```

What to verify:

- Both clients log `Joined game ...`, then `=== GAME STARTED ===`.
- Move latency is ~0.06–0.16s after the first move (the first move loads
  the checkpoint and is slower).
- If a move would repeat a recent one, you see a `[HEURISTIC]` tag in the
  log — that's the anti-oscillation fallback firing.
- Game finishes with `=== GAME FINISHED ===` and a score table.

If any of these fail, fix before tomorrow.

### Local-only smoke test (no `game.py` needed)

This skips the JSON-RPC layer but exercises the trained agent end-to-end
against the bundled greedy/heuristic opponents. Good enough to confirm
the checkpoint loads, MCTS runs, and the agent plays reasonable moves.

```bash
python3 play.py watch \
  --checkpoint checkpoints/phase1_v6/model_best.pt \
  --opponent greedy \
  --mcts-sims 100 \
  --device cuda \
  --delay 0.1
```

You'll see an ASCII board update each move. Expected: the agent moves
its pins steadily toward the opposite corner and finishes a 2P game in
~120–180 moves. If it stalls, oscillates, or fails to load the
checkpoint, fix before the real test run.

For multi-player practice (since the tournament uses 2/4/6 players):

```bash
python3 validate_multiplayer.py \
  --checkpoint checkpoints/phase1_v6/model_best.pt \
  --device cuda \
  --mcts-sims 100 \
  --players 2 4 6
```

---

## Useful flags (only if you need them)

All of these are optional — defaults are the tournament-locked config.

| flag | default | when to override |
|---|---|---|
| `--mcts-sims` | 100 | Lower (e.g. 50) if the per-move time limit is tight; higher (200) only if time is generous and you want stronger play. |
| `--c-puct` | 1.0 | Don't change — this is the sweep winner. |
| `--temperature` | 0.3 | `0.0` to play deterministically; `0.5` for more diversity. Sweep showed 0.3 is the sweet spot. |
| `--time-limit` | none | Soft per-move time limit in seconds. Leave unset unless required. |
| `--device` | `cpu` | Use `cuda` on the school machine — much faster MCTS. |
| `--dirichlet-alpha` / `--root-noise-epsilon` | 0 / 0 | Leave off. Only enable if opponents start memorizing openings. |
| `--checkpoint-2p` / `--checkpoint-4p` / `--checkpoint-6p` | none | Set per-player-count checkpoint overrides if you ever want different weights for different game sizes. Currently we use the same checkpoint everywhere. |

---

## What can go wrong

- **`connect-failed`** in the log — server isn't running, or wrong host/port.
- **`JOIN ERROR`** — server is full or already finished. Restart the server.
- **`No module named 'torch'`** — wrong shell / virtualenv on the school
  machine. Activate the env that has the project deps.
- **Slow first move** — expected. The checkpoint loads on the first turn.
  If the *second* move is also slow, check `--device cuda` is set.
- **Agent makes the same move repeatedly** — the anti-oscillation fallback
  should prevent this; if you see it, capture the log and report.

---

## Tournament-day checklist (2026-05-22)

1. `git pull` on the school machine — make sure code is current.
2. `ls checkpoints/phase1_v6/model_best.pt` — confirm checkpoint present.
3. Get host/port from the instructor.
4. Run the launch command with `--host` / `--port`.
5. Watch the log for `=== GAME STARTED ===`, then leave it alone.
