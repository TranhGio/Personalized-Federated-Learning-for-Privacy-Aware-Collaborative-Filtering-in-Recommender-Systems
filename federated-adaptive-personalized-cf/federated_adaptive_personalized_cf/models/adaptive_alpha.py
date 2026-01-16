"""Adaptive Alpha Computation for Personalized Federated Learning.

Alpha (α) controls the personalization level:
- α → 1: Fully personalized (use local user embedding)
- α → 0: Fully global (use global prototype)

Two methods are supported:

1. DataQuantity-based alpha (single factor):
   α = σ((n_interactions - threshold) * temperature)

2. Multi-factor alpha (combines multiple user characteristics):
   α = w_q * f_quantity + w_d * f_diversity + w_c * f_coverage + w_s * f_consistency

Users with more interactions/diversity get higher alpha (more personalization)
because they have enough local data to learn good embeddings.
Users with fewer interactions rely more on the global prototype.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Union


@dataclass
class AlphaConfig:
    """Configuration for adaptive alpha computation.

    Attributes:
        method: Alpha computation method ("data_quantity" or "multi_factor")
        min_alpha: Minimum personalization level (default: 0.1)
        max_alpha: Maximum personalization level (default: 0.95)
        quantity_threshold: Interaction count at sigmoid midpoint
        quantity_temperature: Sigmoid steepness

        Multi-factor specific:
        factor_weights: Weights for each factor (must sum to 1.0)
        max_entropy: Maximum genre entropy for normalization
        coverage_threshold: Items count for full coverage credit
        max_rating_std: Maximum rating std for normalization
    """

    # Method selection
    method: str = "data_quantity"  # "data_quantity" or "multi_factor"

    # Common parameters
    min_alpha: float = 0.1
    max_alpha: float = 0.95
    quantity_threshold: int = 100
    quantity_temperature: float = 0.05

    # Multi-factor weights (must sum to 1.0)
    factor_weights: Dict[str, float] = field(default_factory=lambda: {
        'quantity': 0.40,
        'diversity': 0.25,
        'coverage': 0.20,
        'consistency': 0.15,
    })

    # Multi-factor normalization thresholds
    max_entropy: float = 3.0           # ~18 genres in MovieLens → log2(18) ≈ 4.17
    coverage_threshold: int = 100      # Items for full coverage credit
    max_rating_std: float = 1.5        # Typical max std for 1-5 ratings

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
        if self.method not in ("data_quantity", "multi_factor"):
            raise ValueError(
                f"Unknown method: {self.method}. Use 'data_quantity' or 'multi_factor'"
            )
        # Validate multi-factor weights sum to ~1.0
        if self.method == "multi_factor":
            weight_sum = sum(self.factor_weights.values())
            if not 0.99 <= weight_sum <= 1.01:
                raise ValueError(
                    f"factor_weights must sum to 1.0, got {weight_sum}"
                )


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


class DataQuantityAlpha:
    """Compute adaptive alpha based on user interaction count.

    Alpha is computed using a sigmoid function that maps interaction
    count to a personalization level:

    α_raw = σ((n - threshold) * temperature)
          = 1 / (1 + exp(-(n - threshold) * temperature))

    The raw alpha is then clipped to [min_alpha, max_alpha].

    Example:
        With threshold=100, temperature=0.05:
        - n=50  → α ≈ 0.076 → clipped to 0.1 (min)
        - n=80  → α ≈ 0.27
        - n=100 → α = 0.5 (midpoint)
        - n=120 → α ≈ 0.73
        - n=150 → α ≈ 0.92
    """

    def __init__(self, config: Optional[AlphaConfig] = None):
        """Initialize with configuration."""
        self.config = config or AlphaConfig()

    def compute(self, n_interactions: int) -> float:
        """Compute alpha for a user based on their interaction count."""
        threshold = self.config.quantity_threshold
        temp = self.config.quantity_temperature

        x = (n_interactions - threshold) * temp
        alpha_raw = _sigmoid(x)

        return float(np.clip(alpha_raw, self.config.min_alpha, self.config.max_alpha))

    def compute_from_stats(self, user_stats: Dict) -> float:
        """Compute alpha from user statistics dictionary."""
        n_interactions = user_stats.get("n_interactions", 0)
        return self.compute(n_interactions)

    def get_alpha_for_group(self, group: str) -> float:
        """Get typical alpha value for a user group."""
        group_midpoints = {
            "sparse": 15,
            "medium": 65,
            "dense": 150,
        }
        if group not in group_midpoints:
            raise ValueError(f"Unknown group: {group}. Use 'sparse', 'medium', or 'dense'")
        return self.compute(group_midpoints[group])

    def describe(self) -> str:
        """Return a human-readable description of the alpha computation."""
        return (
            f"DataQuantityAlpha(\n"
            f"  method: data_quantity\n"
            f"  formula: α = clip(sigmoid((n - {self.config.quantity_threshold}) * "
            f"{self.config.quantity_temperature}), {self.config.min_alpha}, {self.config.max_alpha})\n"
            f"  sparse (15 interactions): α = {self.compute(15):.3f}\n"
            f"  medium (65 interactions): α = {self.compute(65):.3f}\n"
            f"  dense (150 interactions): α = {self.compute(150):.3f}\n"
            f")"
        )


class MultiFactorAlpha:
    """Compute adaptive alpha based on multiple user characteristics.

    Factors considered:
    1. Quantity (f_q): Number of interactions → sigmoid normalized
       More data = better local model quality

    2. Diversity (f_d): Genre entropy → linear normalized
       Diverse preferences = needs personalization to capture breadth

    3. Coverage (f_c): Unique items rated → linear normalized
       Wide coverage = reliable local patterns

    4. Consistency (f_s): Rating stability → inverse of std
       Consistent ratings = stable preferences worth preserving

    Final alpha is a weighted combination:
    α = w_q * f_q + w_d * f_d + w_c * f_c + w_s * f_s
    """

    def __init__(self, config: Optional[AlphaConfig] = None):
        """Initialize with configuration."""
        self.config = config or AlphaConfig(method="multi_factor")
        self.weights = self.config.factor_weights

    def compute(self, n_interactions: int) -> float:
        """Fallback: compute alpha using only quantity factor."""
        return self._compute_quantity_factor(n_interactions)

    def compute_from_stats(self, user_stats: Dict) -> float:
        """Compute alpha from full user statistics.

        Args:
            user_stats: Dict with keys:
                - n_interactions: int
                - genre_entropy: float (optional)
                - n_unique_items: int (optional)
                - rating_std: float (optional)

        Returns:
            Alpha value in [min_alpha, max_alpha]
        """
        # Factor 1: Quantity (sigmoid normalized)
        n = user_stats.get('n_interactions', 0)
        f_quantity = self._compute_quantity_factor(n)

        # Factor 2: Diversity (genre entropy, linear normalized)
        entropy = user_stats.get('genre_entropy', self.config.max_entropy / 2)
        f_diversity = min(entropy / self.config.max_entropy, 1.0)

        # Factor 3: Coverage (unique items, linear normalized)
        n_unique = user_stats.get('n_unique_items', n)
        f_coverage = min(n_unique / self.config.coverage_threshold, 1.0)

        # Factor 4: Consistency (inverse of rating std)
        rating_std = user_stats.get('rating_std', self.config.max_rating_std / 2)
        f_consistency = 1.0 - min(rating_std / self.config.max_rating_std, 1.0)

        # Weighted combination
        alpha_raw = (
            self.weights['quantity'] * f_quantity +
            self.weights['diversity'] * f_diversity +
            self.weights['coverage'] * f_coverage +
            self.weights['consistency'] * f_consistency
        )

        return float(np.clip(alpha_raw, self.config.min_alpha, self.config.max_alpha))

    def compute_factors(self, user_stats: Dict) -> Dict[str, float]:
        """Compute individual factors for debugging/analysis.

        Returns dict with keys: quantity, diversity, coverage, consistency, alpha
        """
        n = user_stats.get('n_interactions', 0)
        f_quantity = self._compute_quantity_factor(n)

        entropy = user_stats.get('genre_entropy', self.config.max_entropy / 2)
        f_diversity = min(entropy / self.config.max_entropy, 1.0)

        n_unique = user_stats.get('n_unique_items', n)
        f_coverage = min(n_unique / self.config.coverage_threshold, 1.0)

        rating_std = user_stats.get('rating_std', self.config.max_rating_std / 2)
        f_consistency = 1.0 - min(rating_std / self.config.max_rating_std, 1.0)

        alpha = self.compute_from_stats(user_stats)

        return {
            'quantity': f_quantity,
            'diversity': f_diversity,
            'coverage': f_coverage,
            'consistency': f_consistency,
            'alpha': alpha,
        }

    def _compute_quantity_factor(self, n: int) -> float:
        """Sigmoid-normalized quantity factor."""
        x = (n - self.config.quantity_threshold) * self.config.quantity_temperature
        alpha_raw = _sigmoid(x)
        return float(np.clip(alpha_raw, self.config.min_alpha, self.config.max_alpha))

    def get_alpha_for_group(self, group: str) -> float:
        """Get typical alpha value for a user group (quantity-only fallback)."""
        group_midpoints = {
            "sparse": 15,
            "medium": 65,
            "dense": 150,
        }
        if group not in group_midpoints:
            raise ValueError(f"Unknown group: {group}. Use 'sparse', 'medium', or 'dense'")
        return self.compute(group_midpoints[group])

    def describe(self) -> str:
        """Return a human-readable description of the alpha computation."""
        return (
            f"MultiFactorAlpha(\n"
            f"  method: multi_factor\n"
            f"  formula: α = {self.weights['quantity']:.2f}*f_q + "
            f"{self.weights['diversity']:.2f}*f_d + "
            f"{self.weights['coverage']:.2f}*f_c + "
            f"{self.weights['consistency']:.2f}*f_s\n"
            f"  quantity: sigmoid((n - {self.config.quantity_threshold}) * {self.config.quantity_temperature})\n"
            f"  diversity: entropy / {self.config.max_entropy}\n"
            f"  coverage: n_unique / {self.config.coverage_threshold}\n"
            f"  consistency: 1 - (rating_std / {self.config.max_rating_std})\n"
            f"  range: [{self.config.min_alpha}, {self.config.max_alpha}]\n"
            f")"
        )


# Type alias for alpha computers
AlphaComputer = Union[DataQuantityAlpha, MultiFactorAlpha]


def create_alpha_computer(config: Optional[AlphaConfig] = None) -> AlphaComputer:
    """Factory function to create appropriate alpha computer.

    Args:
        config: AlphaConfig instance, uses defaults if None

    Returns:
        DataQuantityAlpha or MultiFactorAlpha based on config.method
    """
    if config is None:
        config = AlphaConfig()

    method = config.method.lower()

    if method == "data_quantity":
        return DataQuantityAlpha(config)
    elif method == "multi_factor":
        return MultiFactorAlpha(config)
    else:
        raise ValueError(f"Unknown alpha method: {method}")
