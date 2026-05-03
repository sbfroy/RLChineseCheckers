# Competition / test-run guide

How to connect the trained agent to the course's game server. Used for the
2026-05-04 test run and the 2026-05-22 tournament.

---

## What you have

- **Game server** (`game.py`) — provided by the course. Exposes JSON-RPC on a
  TCP port (default `127.0.0.1:50555`). Started by the instructor or by you
  in a separate terminal for local self-tests.
- **Player client** (`env/server_adapter.py`) — your code. Connects to the
  server, joins as a player, and uses your trained checkpoint to pick moves.
- **Checkpoint** — `checkpoints/phase1_v6/model_best.pt` on the school
  machine.

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

Run a single 2P game against yourself to confirm everything works
end-to-end. Three terminals on the school machine:

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
