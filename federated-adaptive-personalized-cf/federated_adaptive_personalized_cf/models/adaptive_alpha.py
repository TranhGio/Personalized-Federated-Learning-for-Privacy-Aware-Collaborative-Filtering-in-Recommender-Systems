"""Adaptive Alpha Computation for Personalized Federated Learning.

Alpha (α) controls the personalization level:
- α → 1: Fully personalized (use local user embedding)
- α → 0: Fully global (use global prototype)

The DataQuantity-based alpha uses a sigmoid function:
α = σ((n_interactions - threshold) * temperature)

Where:
- n_interactions: number of ratings for the user
- threshold: midpoint (e.g., 50 interactions)
- temperature: steepness of the sigmoid (e.g., 0.1)

Users with more interactions get higher alpha (more personalization)
because they have enough local data to learn good embeddings.
Users with fewer interactions rely more on the global prototype.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AlphaConfig:
    """Configuration for adaptive alpha computation.

    Attributes:
        min_alpha: Minimum personalization level (default: 0.1)
                   Even sparse users get some local influence
        max_alpha: Maximum personalization level (default: 0.95)
                   Even dense users get some global influence
        quantity_threshold: Interaction count at sigmoid midpoint (default: 50)
                           Users with ~50 interactions get α ≈ 0.5
        quantity_temperature: Sigmoid steepness (default: 0.1)
                             Higher = sharper transition
    """

    min_alpha: float = 0.1
    max_alpha: float = 0.95
    quantity_threshold: int = 50
    quantity_temperature: float = 0.1

    def __post_init__(self):
        """Validate configuration values."""
        if not 0 <= self.min_alpha < self.max_alpha <= 1:
            raise ValueError(
                f"Invalid alpha range: [{self.min_alpha}, {self.max_alpha}]. "
                "Must satisfy 0 <= min_alpha < max_alpha <= 1"
            )
        if self.quantity_threshold <= 0:
            raise ValueError(
                f"quantity_threshold must be positive, got {self.quantity_threshold}"
            )
        if self.quantity_temperature <= 0:
            raise ValueError(
                f"quantity_temperature must be positive, got {self.quantity_temperature}"
            )


class DataQuantityAlpha:
    """Compute adaptive alpha based on user interaction count.

    Alpha is computed using a sigmoid function that maps interaction
    count to a personalization level:

    α_raw = σ((n - threshold) * temperature)
          = 1 / (1 + exp(-(n - threshold) * temperature))

    The raw alpha is then clipped to [min_alpha, max_alpha].

    Example:
        With threshold=50, temperature=0.1:
        - n=10  → α ≈ 0.018 → clipped to 0.1 (min)
        - n=30  → α ≈ 0.12
        - n=50  → α = 0.5 (midpoint)
        - n=70  → α ≈ 0.88
        - n=100 → α ≈ 0.99 → clipped to 0.95 (max)
    """

    def __init__(self, config: Optional[AlphaConfig] = None):
        """Initialize with configuration.

        Args:
            config: AlphaConfig instance, uses defaults if None
        """
        self.config = config or AlphaConfig()

    def compute(self, n_interactions: int) -> float:
        """Compute alpha for a user based on their interaction count.

        Args:
            n_interactions: Number of ratings/interactions for the user

        Returns:
            Alpha value in [min_alpha, max_alpha]
        """
        threshold = self.config.quantity_threshold
        temp = self.config.quantity_temperature

        # Sigmoid function: σ(x) = 1 / (1 + exp(-x))
        x = (n_interactions - threshold) * temp
        alpha_raw = self._sigmoid(x)

        # Clip to valid range
        return np.clip(alpha_raw, self.config.min_alpha, self.config.max_alpha)

    def compute_from_stats(self, user_stats: Dict) -> float:
        """Compute alpha from user statistics dictionary.

        Args:
            user_stats: Dictionary containing 'n_interactions' key

        Returns:
            Alpha value in [min_alpha, max_alpha]
        """
        n_interactions = user_stats.get("n_interactions", 0)
        return self.compute(n_interactions)

    def _sigmoid(self, x: float) -> float:
        """Numerically stable sigmoid function.

        Args:
            x: Input value

        Returns:
            Sigmoid of x in (0, 1)
        """
        # Avoid overflow for large negative x
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1.0 + exp_x)

    def get_alpha_for_group(self, group: str) -> float:
        """Get typical alpha value for a user group.

        Useful for analysis and visualization.

        Args:
            group: One of 'sparse', 'medium', 'dense'

        Returns:
            Representative alpha for that group
        """
        group_midpoints = {
            "sparse": 15,  # midpoint of 0-30
            "medium": 65,  # midpoint of 30-100
            "dense": 150,  # representative for 100+
        }
        if group not in group_midpoints:
            raise ValueError(f"Unknown group: {group}. Use 'sparse', 'medium', or 'dense'")
        return self.compute(group_midpoints[group])

    def describe(self) -> str:
        """Return a human-readable description of the alpha computation."""
        return (
            f"DataQuantityAlpha(\n"
            f"  formula: α = clip(sigmoid((n - {self.config.quantity_threshold}) * "
            f"{self.config.quantity_temperature}), {self.config.min_alpha}, {self.config.max_alpha})\n"
            f"  sparse (15 interactions): α = {self.compute(15):.3f}\n"
            f"  medium (65 interactions): α = {self.compute(65):.3f}\n"
            f"  dense (150 interactions): α = {self.compute(150):.3f}\n"
            f")"
        )


def create_alpha_computer(config: Optional[AlphaConfig] = None) -> DataQuantityAlpha:
    """Factory function to create alpha computer.

    Args:
        config: AlphaConfig instance, uses defaults if None

    Returns:
        DataQuantityAlpha instance
    """
    return DataQuantityAlpha(config)
