"""
Policy-Value network for Chinese Checkers.

Architecture:
- Residual CNN trunk on the 17x17 spatial board representation
- Scalar features concatenated into the MLP heads
- Policy head: outputs logits over 1210 flat actions (with masking)
- Value head: outputs scalar position estimate in [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from models.encoders import NUM_SPATIAL_CHANNELS, NUM_SCALAR_FEATURES, GRID_SIZE
from env.action_mapping import ACTION_SPACE_SIZE


class ResBlock(nn.Module):
    """Residual block with two 3x3 convolutions and batch norm."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class PolicyValueNet(nn.Module):
    """
    Dual-headed policy-value network.

    Args:
        num_res_blocks: number of residual blocks in the trunk
        trunk_channels: number of channels in the trunk convolutions
        policy_channels: number of channels in the policy head conv
        value_hidden: hidden size for the value head MLP
    """

    def __init__(
        self,
        num_res_blocks: int = 6,
        trunk_channels: int = 128,
        policy_channels: int = 32,
        value_hidden: int = 128,
    ):
        super().__init__()

        # === Shared trunk ===
        # Initial convolution from input channels to trunk channels
        self.input_conv = nn.Sequential(
            nn.Conv2d(NUM_SPATIAL_CHANNELS, trunk_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(),
        )

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(trunk_channels) for _ in range(num_res_blocks)]
        )

        # Flatten size for trunk features
        trunk_flat_size = trunk_channels * GRID_SIZE * GRID_SIZE

        # === Policy head ===
        self.policy_conv = nn.Sequential(
            nn.Conv2d(trunk_channels, policy_channels, 1, bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(),
        )
        policy_flat_size = policy_channels * GRID_SIZE * GRID_SIZE
        self.policy_fc = nn.Sequential(
            nn.Linear(policy_flat_size + NUM_SCALAR_FEATURES, 512),
            nn.ReLU(),
            nn.Linear(512, ACTION_SPACE_SIZE),
        )

        # === Value head ===
        self.value_conv = nn.Sequential(
            nn.Conv2d(trunk_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        value_flat_size = GRID_SIZE * GRID_SIZE
        self.value_fc = nn.Sequential(
            nn.Linear(value_flat_size + NUM_SCALAR_FEATURES, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        spatial: torch.Tensor,
        scalars: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            spatial: (B, NUM_SPATIAL_CHANNELS, 17, 17)
            scalars: (B, NUM_SCALAR_FEATURES)
            legal_mask: (B, ACTION_SPACE_SIZE) boolean mask, optional

        Returns:
            policy_logits: (B, ACTION_SPACE_SIZE) — raw logits (masked if mask provided)
            value: (B, 1) — position estimate in [-1, 1]
        """
        # Trunk
        x = self.input_conv(spatial)
        x = self.res_blocks(x)

        # Policy head
        p = self.policy_conv(x)
        p = p.flatten(1)  # (B, policy_channels * 17 * 17)
        p = torch.cat([p, scalars], dim=1)
        policy_logits = self.policy_fc(p)

        # Apply legal mask if provided
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask, -1e9)

        # Value head
        v = self.value_conv(x)
        v = v.flatten(1)  # (B, 17*17)
        v = torch.cat([v, scalars], dim=1)
        value = self.value_fc(v)

        return policy_logits, value

    def predict(
        self,
        spatial: torch.Tensor,
        scalars: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Single-state prediction for inference.
        Returns (action_probs, value) with no gradient.

        Args:
            spatial: (NUM_SPATIAL_CHANNELS, 17, 17) — single state
            scalars: (NUM_SCALAR_FEATURES,) — single state
            legal_mask: (ACTION_SPACE_SIZE,) — single state

        Returns:
            probs: (ACTION_SPACE_SIZE,) — probability distribution
            value: scalar float
        """
        self.eval()
        with torch.no_grad():
            s = spatial.unsqueeze(0)
            sc = scalars.unsqueeze(0)
            m = legal_mask.unsqueeze(0)

            logits, v = self.forward(s, sc, m)
            probs = F.softmax(logits, dim=-1).squeeze(0)
            return probs, v.item()
