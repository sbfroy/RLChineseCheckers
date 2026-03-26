"""
Reward shaping for Chinese Checkers training.

Aligned with the competition scoring formula. Combines:
1. Terminal outcome reward (win/loss/draw)
2. Per-move progress (delta in distance and pins-in-goal)
3. Lagging-piece penalty (max single pin distance)
4. Home-camp exit reward
5. Mobility bonus (available legal actions)

All shaping weights are configurable for ablation.
"""

from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    """Configurable reward shaping weights."""
    # Terminal rewards
    win_reward: float = 1.0
    loss_reward: float = -1.0
    draw_reward: float = -0.5
    max_moves_reward: float = -0.3

    # Per-move shaping weights
    pin_goal_weight: float = 0.3        # reward per pin entering goal zone
    distance_weight: float = 0.01       # reward per unit of distance reduction
    lagging_weight: float = -0.005      # penalty for max pin distance
    home_exit_weight: float = 0.05      # reward per pin leaving home zone
    mobility_weight: float = 0.001      # reward for available actions

    # Score-based terminal (used when game ends by time/max moves)
    use_score_terminal: bool = True
    score_normalization: float = 1300.0  # max possible competition score


class RewardShaper:
    """
    Computes shaped rewards for training.

    Call compute_reward() after each step, passing before/after distance info.
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def compute_reward(
        self,
        info_before: Dict,
        info_after: Dict,
        done: bool,
        status: str,
        scores: Optional[Dict] = None,
        colour: Optional[str] = None,
    ) -> float:
        """
        Compute the shaped reward for a single step.

        Args:
            info_before: distance info dict before the move (from compute_distance_info)
            info_after: distance info dict after the move
            done: whether the game ended
            status: game status string ("WIN", "DRAW", "MAX_MOVES", "CONTINUE")
            scores: competition scores dict if game ended
            colour: agent's colour (for score lookup)

        Returns:
            float reward
        """
        cfg = self.config
        reward = 0.0

        if done:
            if status == "WIN":
                reward += cfg.win_reward
            elif status == "DRAW":
                reward += cfg.draw_reward
            elif status == "MAX_MOVES":
                reward += cfg.max_moves_reward
                # Add score-based bonus if available
                if cfg.use_score_terminal and scores and colour and colour in scores:
                    normalized_score = scores[colour]["final_score"] / cfg.score_normalization
                    reward += normalized_score * 0.5
            return reward

        # Per-move shaping
        # 1. Pin goal progress
        delta_pins_in_goal = info_after["pins_in_goal"] - info_before["pins_in_goal"]
        reward += cfg.pin_goal_weight * delta_pins_in_goal

        # 2. Distance reduction
        delta_dist = info_before["total_distance"] - info_after["total_distance"]
        reward += cfg.distance_weight * delta_dist

        # 3. Lagging piece penalty (penalize high max distance)
        reward += cfg.lagging_weight * info_after["max_distance"]

        # 4. Home camp exit
        delta_home = info_before["pins_in_home"] - info_after["pins_in_home"]
        reward += cfg.home_exit_weight * delta_home

        return reward
