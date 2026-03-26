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
from typing import Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
    data = s.recv(1_000_000)
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
        mcts_simulations: int = 100,
        time_limit: Optional[float] = None,
        device: str = "cpu",
    ):
        self.player_name = player_name
        self.agent = ChineseCheckersAgent(
            checkpoint_path=checkpoint_path,
            mcts_simulations=mcts_simulations,
            time_limit=time_limit,
            device=device,
        )

        self.game_id = None
        self.player_id = None
        self.colour = None
        self.my_move_count = 0
        self.total_time = 0.0

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
        while True:
            st = rpc({"op": "get_state", "game_id": self.game_id})
            status = st.get("state", {}).get("status", "")
            if status in ("READY_TO_START", "PLAYING"):
                break
            time.sleep(0.3)

        # START
        rpc({"op": "start", "game_id": self.game_id, "player_id": self.player_id})

        # Wait for PLAYING
        while True:
            st = rpc({"op": "get_state", "game_id": self.game_id})
            if st.get("state", {}).get("status") == "PLAYING":
                break
            time.sleep(0.3)

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

            # Select action using trained agent
            pin_id, to_index = self.agent.select_action_from_server_state(
                pin_positions=state.get("pins", {}),
                legal_moves=movable,
                my_colour=self.colour,
                turn_order=state.get("turn_order", []),
                move_count=state.get("move_count", 0),
                my_move_count=self.my_move_count,
            )

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

            if mv.get("ok"):
                print(
                    f"Move {self.my_move_count}: pin {pin_id} -> {to_index} "
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
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mcts-sims", type=int, default=100)
    parser.add_argument("--time-limit", type=float, default=None)
    args = parser.parse_args()

    player = CompetitionPlayer(
        player_name=args.name,
        checkpoint_path=args.checkpoint,
        mcts_simulations=args.mcts_sims,
        time_limit=args.time_limit,
    )
    player.run()
