"""Endgame pathfinder for Chinese Checkers.

When most of our pins are already in goal, MCTS visits get noisy and the
policy prior offers no signal to drag specific lagging pins through the
narrow corridors that remain. This module replaces the network's choice
with a deterministic per-pin BFS: each BFS edge is one full move
(including chain-jumps, which the engine encodes as single legal
destinations from `Pin.getPossibleMoves`), so a geographically long path
can resolve in only a few turns.

Opponent pins are treated as static for the duration of one BFS, and we
re-plan every turn. That's strictly worse than full multi-pin
coordination, but multi-pin coordination is a Chinese-Checkers-puzzle-hard
search and re-planning closes most of the gap in practice.

When all empty goal cells are surrounded by our own pins (no path exists),
the solver performs a "gap shuffle": it temporarily moves an in-goal pin
out to create an opening for the straggler, then re-planning on subsequent
turns handles cleanup.
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from env.local_game import LocalGame, COLOUR_OPPOSITES


_HEX_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


def _legal_destinations(board, start_idx: int, occupied: Set[int]) -> List[int]:
    cell = board.cells[start_idx]
    q0, r0 = cell.q, cell.r
    index_of = board.index_of

    possible: Set[int] = set()
    for dq, dr in _HEX_DIRS:
        ni = index_of.get((q0 + dq, r0 + dr))
        if ni is not None and ni not in occupied:
            possible.add(ni)

    visited = {start_idx}
    stack = [start_idx]
    while stack:
        cur = stack.pop()
        ccell = board.cells[cur]
        cq, cr = ccell.q, ccell.r
        for dq, dr in _HEX_DIRS:
            adj = index_of.get((cq + dq, cr + dr))
            land = index_of.get((cq + 2 * dq, cr + 2 * dr))
            if adj is None or land is None:
                continue
            if adj in occupied and land not in occupied and land not in visited:
                possible.add(land)
                visited.add(land)
                stack.append(land)
    return sorted(possible)


def _bfs_to_goal(
    board,
    start_idx: int,
    occupied: Set[int],
    goal_targets: Set[int],
    max_moves: int,
) -> Optional[List[int]]:
    if start_idx in goal_targets:
        return []
    bfs_occupied = occupied - {start_idx}
    visited = {start_idx}
    queue = deque([(start_idx, [])])
    while queue:
        cell, path = queue.popleft()
        if len(path) >= max_moves:
            continue
        for dest in _legal_destinations(board, cell, bfs_occupied):
            if dest in visited:
                continue
            new_path = path + [dest]
            if dest in goal_targets:
                return new_path
            visited.add(dest)
            queue.append((dest, new_path))
    return None


class EndgameSolver:
    """Pathfinds lagging pins into the goal zone.

    Activation at 7 pins gives the BFS multiple empty goal cells to
    target, reducing blocked-goal situations.
    """

    def __init__(self, activation_threshold: int = 7, max_path_moves: int = 50):
        self.activation_threshold = activation_threshold
        self.max_path_moves = max_path_moves

    def is_active(self, game: LocalGame, colour: str) -> bool:
        opposite = COLOUR_OPPOSITES[colour]
        pins = game.pins_by_colour[colour]
        in_goal = sum(
            1 for p in pins if game.board.cells[p.axialindex].postype == opposite
        )
        return in_goal >= self.activation_threshold

    def select_action(
        self, game: LocalGame, colour: str
    ) -> Optional[Tuple[int, int]]:
        opposite = COLOUR_OPPOSITES[colour]
        board = game.board
        my_pins = game.pins_by_colour[colour]

        occupied: Set[int] = set()
        for pins in game.pins_by_colour.values():
            for p in pins:
                occupied.add(p.axialindex)

        goal_cells = {
            i for i, c in enumerate(board.cells) if c.postype == opposite
        }
        goal_targets = goal_cells - occupied
        if not goal_targets:
            return None

        lagging = [
            p for p in my_pins
            if board.cells[p.axialindex].postype != opposite
        ]
        if not lagging:
            return None

        idx_to_pid: Dict[int, int] = {p.axialindex: p.id for p in my_pins}

        best: Optional[Tuple[int, int, int]] = None
        for pin in lagging:
            path = _bfs_to_goal(
                board, pin.axialindex, occupied, goal_targets,
                self.max_path_moves,
            )
            if not path:
                continue
            score = (len(path), pin.id, path[0])
            if best is None or score < best:
                best = score

        if best is not None:
            _, pin_id, first_move = best
            return (pin_id, first_move)

        return self._gap_shuffle(
            board, occupied, goal_cells, goal_targets, lagging, idx_to_pid
        )

    def _gap_shuffle(
        self,
        board,
        occupied: Set[int],
        goal_cells: Set[int],
        goal_targets: Set[int],
        lagging,
        idx_to_pid: Dict[int, int],
    ) -> Optional[Tuple[int, int]]:
        """Move an in-goal pin out to create a path for a straggler.

        For each of our pins sitting in the goal zone, check whether
        stepping it to an empty neighbor would open a BFS path for any
        lagging pin. Pick the shuffle that yields the shortest path.
        """
        occupied_goal = goal_cells - goal_targets
        index_of = board.index_of
        best: Optional[Tuple[int, int, int]] = None  # (path_len, blocker_idx, step_out)

        for blocker_idx in occupied_goal:
            if blocker_idx not in idx_to_pid:
                continue

            cell = board.cells[blocker_idx]
            for dq, dr in _HEX_DIRS:
                ni = index_of.get((cell.q + dq, cell.r + dr))
                if ni is None or ni in occupied:
                    continue

                sim_occupied = (occupied - {blocker_idx}) | {ni}
                sim_targets = goal_cells - sim_occupied
                if not sim_targets:
                    continue

                for lag_pin in lagging:
                    path = _bfs_to_goal(
                        board, lag_pin.axialindex, sim_occupied,
                        sim_targets, self.max_path_moves,
                    )
                    if path is not None:
                        score = (len(path), blocker_idx, ni)
                        if best is None or score < best:
                            best = score
                        break

        if best is None:
            return None

        _, blocker_idx, step_out = best
        return (idx_to_pid[blocker_idx], step_out)
