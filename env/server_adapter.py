"""
Server adapter for competition play.

Wraps the JSON-RPC client interface from player.py to work with
the ChineseCheckersAgent. Drop-in replacement for the random
playing logic in player.py.
"""

import os
import sys
import json
import socket
import time
from collections import deque
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_ENGINE_DIR = os.path.join(os.path.dirname(__file__), "..", "multi system single machine minimal")
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

from checkers_board import HexBoard
from agents.chinese_checkers_agent import ChineseCheckersAgent

HOST = "127.0.0.1"
PORT = 50555


def rpc(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send JSON to server and receive JSON reply."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10.0)
    try:
        s.connect((HOST, PORT))
    except Exception as e:
        return {"ok": False, "error": f"connect-failed: {e}"}

    s.sendall(json.dumps(payload).encode("utf-8"))
    chunks = []
    while True:
        chunk = s.recv(1_000_000)
        if not chunk:
            break
        chunks.append(chunk)
    data = b"".join(chunks)
    s.close()

    if not data:
        return {"ok": False, "error": "no-response"}

    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"bad-json: {e}"}


class CompetitionPlayer:
    """
    Competition player that uses the trained RL agent.

    Integrates with the game server via JSON-RPC.
    """

    def __init__(
        self,
        player_name: str = "RLAgent",
        checkpoint_path: Optional[str] = None,
        checkpoints_by_players: Optional[Dict[int, str]] = None,
        mcts_simulations: int = 15,
        c_puct: float = 1.0,
        temperature: float = 0.3,
        time_limit: Optional[float] = None,
        device: str = "cpu",
        dirichlet_alpha: float = 0.0,
        root_noise_epsilon: float = 0.0,
    ):
        self.player_name = player_name
        self.checkpoint_path = checkpoint_path
        self.checkpoints_by_players = checkpoints_by_players or {}
        self.mcts_simulations = mcts_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.time_limit = time_limit
        self.device = device
        self.dirichlet_alpha = dirichlet_alpha
        self.root_noise_epsilon = root_noise_epsilon
        self.agent: Optional[ChineseCheckersAgent] = None

        self.game_id = None
        self.player_id = None
        self.colour = None
        self.my_move_count = 0
        self.total_time = 0.0
        self._recent_moves: deque = deque(maxlen=20)
        self._board = HexBoard()

    def _select_checkpoint(self, num_players: int) -> Optional[str]:
        if num_players in self.checkpoints_by_players:
            return self.checkpoints_by_players[num_players]
        return self.checkpoint_path

    def _ensure_agent(self, num_players: int):
        if self.agent is not None:
            return
        ckpt = self._select_checkpoint(num_players)
        noise_str = (
            f"α={self.dirichlet_alpha}, ε={self.root_noise_epsilon}"
            if self.dirichlet_alpha > 0 and self.root_noise_epsilon > 0
            else "OFF"
        )
        print(
            f"Loading agent for {num_players}P: checkpoint={ckpt}, "
            f"mcts_sims={self.mcts_simulations}, c_puct={self.c_puct}, "
            f"temperature={self.temperature}, root_noise={noise_str}"
        )
        self.agent = ChineseCheckersAgent(
            checkpoint_path=ckpt,
            mcts_simulations=self.mcts_simulations,
            c_puct=self.c_puct,
            temperature=self.temperature,
            time_limit=self.time_limit,
            device=self.device,
            dirichlet_alpha=self.dirichlet_alpha,
            root_noise_epsilon=self.root_noise_epsilon,
        )

    def _warmup(self, state: Dict[str, Any]):
        """Run one forward pass through the network on the initial state.

        First inference after checkpoint load is slow (lazy init + weight
        paging). Doing it before the main loop keeps move 1 within the
        server's per-turn budget."""
        if self.agent is None:
            return
        try:
            import torch
            from env.action_mapping import ACTION_SPACE_SIZE
            pin_positions = state.get("pins", {})
            turn_order = state.get("turn_order", [])
            if not pin_positions or not turn_order:
                return
            spatial, scalars = self.agent.encoder.encode(
                my_colour=self.colour,
                pin_positions=pin_positions,
                turn_order=turn_order,
                move_count=0,
                total_legal_actions=10,
                num_active_players=len(turn_order),
                my_move_count=0,
            )
            mask = torch.ones(ACTION_SPACE_SIZE, dtype=torch.bool)
            self.agent.model.predict(
                spatial.to(self.device),
                scalars.to(self.device),
                mask.to(self.device),
            )
        except Exception as e:
            print(f"Warmup skipped: {e}")

    def _is_repeating(self, pin_id: int, to_index: int) -> bool:
        return (pin_id, to_index) in self._recent_moves

    @staticmethod
    def _hex_dist(a, b) -> int:
        dq = abs(a.q - b.q)
        dr = abs(a.r - b.r)
        ds = abs((-a.q - a.r) - (-b.q - b.r))
        return max(dq, dr, ds)

    def _heuristic_move(
        self,
        pin_positions: Dict[str, List[int]],
        legal_moves: Dict[int, List[int]],
    ) -> Tuple[int, int]:
        """Pick the legal move that most reduces distance to the nearest
        EMPTY goal cell. Prioritizes moving pins NOT yet in goal so we
        don't just shuffle already-placed pins around."""
        opposite = self._board.colour_opposites[self.colour]
        goal_indices = self._board.axial_of_colour(opposite)
        goal_set = set(goal_indices)
        my_pins = pin_positions[self.colour]

        all_occupied: set = set()
        for positions in pin_positions.values():
            all_occupied.update(positions)
        empty_goals = [g for g in goal_indices if g not in all_occupied]

        if not empty_goals:
            for pid, dests in legal_moves.items():
                for dest in dests:
                    if not self._is_repeating(pid, dest):
                        return (pid, dest)
            for pid, dests in legal_moves.items():
                if dests:
                    return (pid, dests[0])

        cells = self._board.cells

        not_in_goal = {}
        in_goal = {}
        for pid, dests in legal_moves.items():
            if my_pins[pid] in goal_set:
                in_goal[pid] = dests
            else:
                not_in_goal[pid] = dests

        best_move = self._best_gain(not_in_goal, my_pins, cells, empty_goals)
        if best_move is None:
            best_move = self._best_gain(in_goal, my_pins, cells, empty_goals)
        if best_move is None:
            for pid, dests in legal_moves.items():
                if dests:
                    return (pid, dests[0])
        return best_move

    def _best_gain(self, moves, my_pins, cells, empty_goals):
        best_move = None
        best_gain = -999
        for pid, dests in moves.items():
            cur_cell = cells[my_pins[pid]]
            cur_dist = min(self._hex_dist(cur_cell, cells[g]) for g in empty_goals)
            for dest in dests:
                if self._is_repeating(pid, dest):
                    continue
                dest_dist = min(self._hex_dist(cells[dest], cells[g]) for g in empty_goals)
                gain = cur_dist - dest_dist
                if gain > best_gain:
                    best_gain = gain
                    best_move = (pid, dest)
        return best_move

    def run(self):
        """Main game loop for competition play."""
        print(f"==== {self.player_name} ====")

        # JOIN
        r = rpc({"op": "join", "player_name": self.player_name})
        if not r.get("ok"):
            print("JOIN ERROR:", r.get("error"))
            return

        self.game_id = r["game_id"]
        self.player_id = r["player_id"]
        self.colour = r["colour"]
        print(f"Joined game {self.game_id} as {self.colour}")

        # Wait for game ready
        ready_state: Optional[Dict[str, Any]] = None
        while True:
            st = rpc({"op": "get_state", "game_id": self.game_id})
            status = st.get("state", {}).get("status", "")
            if status in ("READY_TO_START", "PLAYING"):
                ready_state = st["state"]
                break
            time.sleep(0.3)

        # Pre-load the agent BEFORE sending start. The server's per-turn
        # timer begins the moment the game enters PLAYING — if we wait
        # until then to load, ~5s of checkpoint paging + PyTorch lazy init
        # eats the first turn and we get skipped before move 1.
        ready_turn_order = (ready_state or {}).get("turn_order", [])
        if ready_turn_order:
            self._ensure_agent(num_players=len(ready_turn_order))
            self._warmup(ready_state)

        # Send start
        # input("Press Enter to send Start...")
        rpc({"op": "start", "game_id": self.game_id, "player_id": self.player_id})

        # Wait for PLAYING
        playing_state: Optional[Dict[str, Any]] = None
        while True:
            st = rpc({"op": "get_state", "game_id": self.game_id})
            if st.get("state", {}).get("status") == "PLAYING":
                playing_state = st["state"]
                break
            time.sleep(0.3)

        # Fallback: pre-load was skipped (no turn_order in ready state).
        # Catch up now so move 1 still hits a warm model.
        if self.agent is None and playing_state:
            play_turn_order = playing_state.get("turn_order", [])
            if play_turn_order:
                self._ensure_agent(num_players=len(play_turn_order))
                self._warmup(playing_state)

        print("=== GAME STARTED ===")

        # MAIN LOOP
        while True:
            st = rpc({"op": "get_state", "game_id": self.game_id})
            if not st.get("ok"):
                time.sleep(0.3)
                continue

            state = st["state"]

            if state.get("turn_timeout_notice"):
                print(f"TIMEOUT: {state['turn_timeout_notice']}")

            if state["status"] == "FINISHED":
                self._print_final_scores(state)
                break

            if state.get("current_turn_colour") != self.colour:
                time.sleep(0.2)
                continue

            # MY TURN
            move_start = time.time()

            # Get legal moves
            legal_req = rpc({
                "op": "get_legal_moves",
                "game_id": self.game_id,
                "player_id": self.player_id,
            })

            if not legal_req.get("ok"):
                time.sleep(0.3)
                continue

            legal_moves = legal_req.get("legal_moves", {})
            # Convert string keys to int
            legal_moves = {int(k): v for k, v in legal_moves.items()}

            movable = {pid: moves for pid, moves in legal_moves.items() if moves}
            if not movable:
                time.sleep(0.3)
                continue

            turn_order = state.get("turn_order", [])
            self._ensure_agent(num_players=len(turn_order))

            # Select action using trained agent
            pin_id, to_index = self.agent.select_action_from_server_state(
                pin_positions=state.get("pins", {}),
                legal_moves=movable,
                my_colour=self.colour,
                turn_order=state.get("turn_order", []),
                move_count=state.get("move_count", 0),
                my_move_count=self.my_move_count,
            )

            # Break oscillation: if RL picks a repeated move, override
            # (but never override endgame BFS — it's deterministic and always progresses)
            override = False
            if not getattr(self.agent, 'last_was_endgame', False) and self._is_repeating(pin_id, to_index):
                pin_id, to_index = self._heuristic_move(
                    state.get("pins", {}), movable
                )
                override = True

            self._recent_moves.append((pin_id, to_index))

            move_time = time.time() - move_start
            self.total_time += move_time
            self.my_move_count += 1

            # Submit move
            mv = rpc({
                "op": "move",
                "game_id": self.game_id,
                "player_id": self.player_id,
                "pin_id": pin_id,
                "to_index": to_index,
            })

            tag = " [HEURISTIC]" if override else ""
            if mv.get("ok"):
                print(
                    f"Move {self.my_move_count}: pin {pin_id} -> {to_index}{tag} "
                    f"({move_time:.2f}s, total={self.total_time:.1f}s)"
                )
                if mv.get("status") == "WIN":
                    print("WIN!")
            else:
                print(f"Move rejected: {mv.get('error')}")

            time.sleep(0.1)

    def _print_final_scores(self, state):
        """Print final game scores."""
        print("\n=== GAME FINISHED ===")
        print(f"Total time: {self.total_time:.1f}s over {self.my_move_count} moves")
        print(f"Avg time/move: {self.total_time/max(self.my_move_count,1):.2f}s")
        for pl in state.get("players", []):
            sc = pl.get("score")
            if sc:
                marker = " <<< ME" if pl["colour"] == self.colour else ""
                print(
                    f"  {pl['name']} ({pl['colour']}): "
                    f"{sc['final_score']:.1f} "
                    f"[pins={sc['pin_goal_score']:.0f}, "
                    f"dist={sc['distance_score']:.0f}, "
                    f"time={sc['time_score']:.0f}]{marker}"
                )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="RLAgent")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Fallback checkpoint used when no per-player checkpoint is set for the detected player count.",
    )
    parser.add_argument("--checkpoint-2p", default=None, help="Checkpoint to use when the joined game has 2 players.")
    parser.add_argument("--checkpoint-4p", default=None, help="Checkpoint to use when the joined game has 4 players.")
    parser.add_argument("--checkpoint-6p", default=None, help="Checkpoint to use when the joined game has 6 players.")
    parser.add_argument("--mcts-sims", type=int, default=15, help="MCTS simulations per move. CPU competition default — sweep_params on i7-1165G7 picked 15 (perfect 2P, best 4P). 100 was the CUDA default but is ~6.4s/move on CPU — too slow for the 60s game wall-clock.")
    parser.add_argument(
        "--c-puct", type=float, default=1.0,
        help="MCTS exploration constant. Competition default 1.0 (winner of "
             "phase1_v6 sweep — improves 6P play vs 1.5).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Action sampling temperature. Competition default 0.3 (winner of "
             "comparison grid — best 6P pins=8.6 with c_puct=1.0).",
    )
    parser.add_argument(
        "--time-limit", type=float, default=1.5,
        help="Hard wall-clock cap on MCTS per move (seconds). Default 1.5 — "
             "leaves ~0.5s headroom under a 2s/turn server budget for RPC "
             "overhead. When set, MCTS ignores --mcts-sims and runs as many "
             "simulations as fit in the budget. For tournaments with a 10s/turn "
             "budget pass --time-limit 9 (or similar) to use the full sims=15 "
             "locked config.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--host", default=HOST,
        help=f"Game server host (default: {HOST}). Set to the tournament "
             "server's address when connecting to a remote game.",
    )
    parser.add_argument(
        "--port", type=int, default=PORT,
        help=f"Game server port (default: {PORT}).",
    )
    parser.add_argument(
        "--dirichlet-alpha", type=float, default=0.0,
        help="MCTS root Dirichlet noise alpha. Default 0.0 (off) — temperature "
             "0.3 already provides decorrelation; sweep showed noise hurt "
             "slightly on top of it.",
    )
    parser.add_argument(
        "--root-noise-epsilon", type=float, default=0.0,
        help="MCTS root noise mixing weight. Default 0.0 (off). Set both "
             "alpha and epsilon > 0 to enable.",
    )
    args = parser.parse_args()

    HOST = args.host
    PORT = args.port

    ckpts_by_players: Dict[int, str] = {}
    if args.checkpoint_2p:
        ckpts_by_players[2] = args.checkpoint_2p
    if args.checkpoint_4p:
        ckpts_by_players[4] = args.checkpoint_4p
    if args.checkpoint_6p:
        ckpts_by_players[6] = args.checkpoint_6p

    player = CompetitionPlayer(
        player_name=args.name,
        checkpoint_path=args.checkpoint,
        checkpoints_by_players=ckpts_by_players,
        mcts_simulations=args.mcts_sims,
        c_puct=args.c_puct,
        temperature=args.temperature,
        time_limit=args.time_limit,
        device=args.device,
        dirichlet_alpha=args.dirichlet_alpha,
        root_noise_epsilon=args.root_noise_epsilon,
    )
    player.run()
