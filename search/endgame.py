"""Endgame pathfinder for Chinese Checkers.

When most of our pins are already in goal, MCTS visits get noisy and the
policy prior offers no signal to drag specific lagging pins through the
narrow corridors that remain. This module replaces the network's choice
with a deterministic per-pin BFS.

When the empty goal cell is interior (surrounded by our pins, unreachable
from outside), the solver rearranges within the goal zone: slide an
in-goal pin into the empty interior cell, freeing a border position the
straggler can reach. Like a slide puzzle.
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

        # Try direct BFS for each lagging pin
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

        # No direct path — slide an in-goal pin deeper to free a border cell
        return self._slide_within_goal(
            board, occupied, goal_cells, goal_targets, lagging, idx_to_pid
        )

    def _slide_within_goal(
        self,
        board,
        occupied: Set[int],
        goal_cells: Set[int],
        goal_targets: Set[int],
        lagging,
        idx_to_pid: Dict[int, int],
    ) -> Optional[Tuple[int, int]]:
        """Slide an in-goal pin into the empty interior cell, freeing its
        old (border) position for the straggler."""
        occupied_goal = goal_cells - goal_targets
        best: Optional[Tuple[int, int, int]] = None

        for pin_idx in occupied_goal:
            if pin_idx not in idx_to_pid:
                continue

            dests = _legal_destinations(board, pin_idx, occupied - {pin_idx})
            for dest in dests:
                if dest not in goal_targets:
                    continue  # only slide within the goal zone

                # Simulate: pin slides from pin_idx to dest (both goal cells)
                sim_occupied = (occupied | {dest}) - {pin_idx}
                sim_targets = goal_cells - sim_occupied
                if not sim_targets:
                    continue

                for lag_pin in lagging:
                    path = _bfs_to_goal(
                        board, lag_pin.axialindex, sim_occupied,
                        sim_targets, self.max_path_moves,
                    )
                    if path is not None:
                        score = (len(path), pin_idx, dest)
                        if best is None or score < best:
                            best = score
                        break

        if best is None:
            return None

        _, pin_idx, dest = best
        return (idx_to_pid[pin_idx], dest)
