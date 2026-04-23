"""
Rotational symmetry for Chinese Checkers board.

The hex star has 6-fold rotational symmetry: rotating the board by 60°
cycles the six colour corners. This module canonicalizes any game state
so that the agent's colour always looks like 'red', allowing models
trained only on red/blue 2P games to play correctly as any colour.

Each permutation is built from axial-coordinate rotations (60° CCW:
(q, r) -> (q+r, -q)) and verified by checking that it maps the target
colour's home cells onto red's home cells.
"""

import os
import sys
from typing import Dict, List

_ENGINE_DIR = os.path.join(os.path.dirname(__file__), "..", "multi system single machine minimal")
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

from checkers_board import HexBoard


COLOUR_CW_CYCLE = ['red', 'gray0', 'yellow', 'blue', 'lawn green', 'purple']


def _rotate_60_ccw(q: int, r: int):
    return (q + r, -q)


def _build_cell_perm(board: HexBoard, k: int) -> Dict[int, int]:
    perm: Dict[int, int] = {}
    for idx, cell in enumerate(board.cells):
        q, r = cell.q, cell.r
        for _ in range(k):
            q, r = _rotate_60_ccw(q, r)
        perm[idx] = board.index_of[(q, r)]
    return perm


_board = HexBoard()

_CELL_PERM: Dict[str, Dict[int, int]] = {}
_CELL_PERM_INV: Dict[str, Dict[int, int]] = {}
_COLOUR_PERM: Dict[str, Dict[str, str]] = {}

for _k, _c in enumerate(COLOUR_CW_CYCLE):
    _perm = _build_cell_perm(_board, _k)
    _CELL_PERM[_c] = _perm
    _CELL_PERM_INV[_c] = {v: u for u, v in _perm.items()}
    _COLOUR_PERM[_c] = {
        other: COLOUR_CW_CYCLE[(j - _k) % 6]
        for j, other in enumerate(COLOUR_CW_CYCLE)
    }

# Verify each permutation maps colour c's home cells onto red's home cells.
_RED_HOME = {i for i, cell in enumerate(_board.cells) if cell.postype == 'red'}
for _c in COLOUR_CW_CYCLE:
    _c_home = {i for i, cell in enumerate(_board.cells) if cell.postype == _c}
    _rotated_home = {_CELL_PERM[_c][i] for i in _c_home}
    assert _rotated_home == _RED_HOME, (
        f"colour_symmetry: permutation for {_c} does not map its home onto red's "
        f"(got {len(_rotated_home & _RED_HOME)}/{len(_RED_HOME)} overlap)"
    )


def canonicalize_positions(
    pin_positions: Dict[str, List[int]],
    my_colour: str,
) -> Dict[str, List[int]]:
    cell_perm = _CELL_PERM[my_colour]
    colour_perm = _COLOUR_PERM[my_colour]
    out: Dict[str, List[int]] = {}
    for colour, indices in pin_positions.items():
        out[colour_perm[colour]] = [cell_perm[i] for i in indices]
    return out


def canonicalize_legal_moves(
    legal_moves: Dict[int, List[int]],
    my_colour: str,
) -> Dict[int, List[int]]:
    cell_perm = _CELL_PERM[my_colour]
    return {pid: [cell_perm[t] for t in tos] for pid, tos in legal_moves.items()}


def canonicalize_turn_order(turn_order: List[str], my_colour: str) -> List[str]:
    colour_perm = _COLOUR_PERM[my_colour]
    return [colour_perm[c] for c in turn_order]


def decanonicalize_to_index(to_index: int, my_colour: str) -> int:
    return _CELL_PERM_INV[my_colour][to_index]
