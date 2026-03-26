"""
Action mapping between (pin_id, to_index) pairs and flat action indices.

Flat action space: 10 pins x 121 cells = 1210 possible actions.
At any state, a binary mask of size 1210 indicates which actions are legal.
"""

import torch
from typing import Dict, List, Tuple

NUM_PINS = 10
NUM_CELLS = 121
ACTION_SPACE_SIZE = NUM_PINS * NUM_CELLS  # 1210


def action_to_flat(pin_id: int, to_index: int) -> int:
    """Convert (pin_id, to_index) to flat action index."""
    return pin_id * NUM_CELLS + to_index


def flat_to_action(flat_idx: int) -> Tuple[int, int]:
    """Convert flat action index to (pin_id, to_index)."""
    pin_id = flat_idx // NUM_CELLS
    to_index = flat_idx % NUM_CELLS
    return pin_id, to_index


def build_legal_mask(legal_moves: Dict[int, List[int]]) -> torch.Tensor:
    """
    Build a binary mask of size ACTION_SPACE_SIZE from legal moves dict.

    Args:
        legal_moves: {pin_id: [dest1, dest2, ...], ...}

    Returns:
        Boolean tensor of shape (ACTION_SPACE_SIZE,)
    """
    mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)
    for pin_id, dests in legal_moves.items():
        for d in dests:
            mask[pin_id * NUM_CELLS + d] = True
    return mask


def legal_actions_from_mask(mask: torch.Tensor) -> List[Tuple[int, int]]:
    """Convert a legal mask back to list of (pin_id, to_index) pairs."""
    indices = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(indices, int):
        indices = [indices]
    return [flat_to_action(i) for i in indices]


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply mask and softmax to logits.

    Args:
        logits: raw scores, shape (..., ACTION_SPACE_SIZE)
        mask: boolean mask, shape (..., ACTION_SPACE_SIZE)

    Returns:
        Probability distribution over legal actions, shape (..., ACTION_SPACE_SIZE)
    """
    masked_logits = logits.masked_fill(~mask, -1e9)
    return torch.softmax(masked_logits, dim=-1)
