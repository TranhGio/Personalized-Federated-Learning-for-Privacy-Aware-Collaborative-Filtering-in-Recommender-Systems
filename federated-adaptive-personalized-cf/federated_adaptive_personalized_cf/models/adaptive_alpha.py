"""Adaptive Alpha Computation for Personalized Federated Learning.

Alpha (α) controls the personalization level:
- α → 1: Fully personalized (use local user embedding)
- α → 0: Fully global (use global prototype)

Three methods are supported:

1. DataQuantity-based alpha (single factor):
   α = σ((n_interactions - threshold) * temperature)

2. Multi-factor alpha (combines multiple user characteristics):
   α = w_q * f_quantity + w_d * f_diversity + w_c * f_coverage + w_s * f_consistency

3. Hierarchical Conditional alpha (addresses factor conflicts):
   - Stage 1: Groups correlated factors (quantity+coverage) and conflicting factors
     (diversity+consistency) using geometric/harmonic means
   - Stage 2: Applies conditional rules for user archetypes (sparse, niche, inconsistent)

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
        if self.method not in ("data_quantity", "multi_factor", "hierarchical_conditional"):
            raise ValueError(
                f"Unknown method: {self.method}. Use 'data_quantity', 'multi_factor', or 'hierarchical_conditional'"
            )
        # Validate multi-factor weights sum to ~1.0
        if self.method == "multi_factor":
            weight_sum = sum(self.factor_weights.values())
            if not 0.99 <= weight_sum <= 1.01:
                raise ValueError(
                    f"factor_weights must sum to 1.0, got {weight_sum}"
                )


@dataclass
class HierarchicalConditionalAlphaConfig:
    """Configuration for hierarchical conditional alpha computation.

    This method addresses conflicts in the multi-factor approach:
    1. Quantity-Coverage redundancy: Both highly correlated (0.8-1.0), combined via geometric mean
    2. Diversity-Consistency conflict: Negatively correlated (-0.3 to -0.5), combined via harmonic mean

    Hierarchical Aggregation:
        data_volume = sqrt(f_quantity * f_coverage)  # Geometric mean
        preference_quality = 2*f_d*f_s / (f_d + f_s)  # Harmonic mean
        base_alpha = w_dv * data_volume + w_pq * preference_quality

    Conditional Rules (domain-aware adjustments):
        1. Sparse users (< sparse_threshold): Apply penalty
        2. Niche specialists (low diversity, high quantity): Apply bonus
        3. Inconsistent raters (low consistency): Apply penalty
        4. Completionists (high coverage, low diversity): Apply bonus
    """

    # Base alpha bounds
    min_alpha: float = 0.1
    max_alpha: float = 0.95

    # Hierarchical weights (must sum to 1.0)
    data_volume_weight: float = 0.55
    preference_quality_weight: float = 0.45

    # Factor computation thresholds (reused from multi-factor)
    quantity_threshold: int = 100
    quantity_temperature: float = 0.05
    max_entropy: float = 3.0
    coverage_threshold: int = 100
    max_rating_std: float = 1.5

    # Conditional rule thresholds
    sparse_threshold: int = 20           # Users below this get penalty
    sparse_penalty_max: float = 0.5      # Maximum penalty factor (50% reduction)

    niche_diversity_threshold: float = 0.25   # f_diversity below this
    niche_quantity_threshold: float = 0.6     # f_quantity above this
    niche_bonus: float = 0.15                 # Alpha bonus for niche specialists

    inconsistent_threshold: float = 0.3       # f_consistency below this
    inconsistent_penalty: float = 0.3         # 30% penalty for inconsistent users

    completionist_coverage: float = 0.7       # f_coverage above this
    completionist_diversity: float = 0.3      # f_diversity below this
    completionist_bonus: float = 0.1          # Alpha bonus for completionists

    def __post_init__(self):
        """Validate configuration values."""
        if not 0 <= self.min_alpha < self.max_alpha <= 1:
            raise ValueError(
                f"Invalid alpha range: [{self.min_alpha}, {self.max_alpha}]. "
                "Must satisfy 0 <= min_alpha < max_alpha <= 1"
            )
        # Validate hierarchical weights sum to ~1.0
        weight_sum = self.data_volume_weight + self.preference_quality_weight
        if not 0.99 <= weight_sum <= 1.01:
            raise ValueError(
                f"Hierarchical weights must sum to 1.0, got {weight_sum}"
            )
        if self.sparse_threshold < 0:
            raise ValueError(f"sparse_threshold must be non-negative, got {self.sparse_threshold}")
        if not 0 <= self.sparse_penalty_max <= 1:
            raise ValueError(f"sparse_penalty_max must be in [0, 1], got {self.sparse_penalty_max}")


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


class HierarchicalConditionalAlpha:
    """Compute adaptive alpha using hierarchical aggregation with conditional rules.

    This method addresses two critical conflicts in the multi-factor approach:

    Conflict 1: Quantity-Coverage Redundancy
        - Both measure "amount of data" with correlation 0.8-1.0
        - Solution: Combine via geometric mean → data_volume = sqrt(f_q * f_c)

    Conflict 2: Diversity-Consistency Contradiction
        - Negatively correlated (-0.3 to -0.5)
        - Diverse users → variable ratings → low consistency
        - Consistent users → specialists → low diversity
        - Solution: Combine via harmonic mean → preference_quality = 2*f_d*f_s/(f_d+f_s)

    Stage 1 - Hierarchical Aggregation:
        data_volume = sqrt(f_quantity * f_coverage)
        preference_quality = 2 * f_diversity * f_consistency / (f_diversity + f_consistency)
        base_alpha = 0.55 * data_volume + 0.45 * preference_quality

    Stage 2 - Conditional Rules (domain-aware):
        Rule 1: Sparse users (< 20 interactions) → penalty
        Rule 2: Niche specialists (low diversity + high quantity) → bonus
        Rule 3: Inconsistent raters (low consistency) → penalty
        Rule 4: Completionists (high coverage + low diversity) → bonus

    Example alpha values:
        - Sparse new user (15 interactions): α ≈ 0.21
        - Dense niche specialist (250 interactions, horror only): α ≈ 0.95
        - Inconsistent casual user (80 interactions, high std): α ≈ 0.23
    """

    def __init__(self, config: Optional[HierarchicalConditionalAlphaConfig] = None):
        """Initialize with configuration."""
        self.config = config or HierarchicalConditionalAlphaConfig()

    def compute(self, n_interactions: int) -> float:
        """Fallback: compute alpha using only quantity factor (for compatibility)."""
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
        # Compute base factors (same as multi-factor)
        n = user_stats.get('n_interactions', 0)
        f_quantity = self._compute_quantity_factor(n)
        f_diversity = self._compute_diversity_factor(
            user_stats.get('genre_entropy', self.config.max_entropy / 2)
        )
        f_coverage = self._compute_coverage_factor(
            user_stats.get('n_unique_items', n)
        )
        f_consistency = self._compute_consistency_factor(
            user_stats.get('rating_std', self.config.max_rating_std / 2)
        )

        # ========================================
        # Stage 1: Hierarchical Aggregation
        # ========================================

        # Group 1: Data Volume (geometric mean of correlated factors)
        data_volume = np.sqrt(f_quantity * f_coverage)

        # Group 2: Preference Quality (harmonic mean of conflicting factors)
        if f_diversity + f_consistency > 0:
            preference_quality = 2 * f_diversity * f_consistency / (f_diversity + f_consistency)
        else:
            preference_quality = 0.0

        # Combine hierarchical groups
        base_alpha = (
            self.config.data_volume_weight * data_volume +
            self.config.preference_quality_weight * preference_quality
        )

        # ========================================
        # Stage 2: Conditional Rules
        # ========================================
        applied_rules = []

        # Rule 1: Sparse users (< sparse_threshold interactions)
        if n < self.config.sparse_threshold:
            penalty = self.config.sparse_penalty_max * (1 - n / self.config.sparse_threshold)
            base_alpha *= (1 - penalty)
            applied_rules.append('sparse')

        # Rule 2: Niche specialists (low diversity but high quantity)
        # Trust their niche expertise
        if (f_diversity < self.config.niche_diversity_threshold and
                f_quantity > self.config.niche_quantity_threshold):
            base_alpha = min(base_alpha + self.config.niche_bonus, 1.0)
            applied_rules.append('niche')

        # Rule 3: Inconsistent raters (high rating variance)
        # Reduce personalization because preferences are unreliable
        if f_consistency < self.config.inconsistent_threshold:
            base_alpha *= (1 - self.config.inconsistent_penalty)
            applied_rules.append('inconsistent')

        # Rule 4: Completionists (high coverage but low diversity)
        # Trust their item exploration within narrow genres
        if (f_coverage > self.config.completionist_coverage and
                f_diversity < self.config.completionist_diversity):
            base_alpha = min(base_alpha + self.config.completionist_bonus, 1.0)
            applied_rules.append('completionist')

        return float(np.clip(base_alpha, self.config.min_alpha, self.config.max_alpha))

    def compute_factors(self, user_stats: Dict) -> Dict[str, float]:
        """Compute all factors and hierarchical components for debugging/analysis.

        Returns dict with keys:
            - quantity, diversity, coverage, consistency (base factors)
            - data_volume, preference_quality (hierarchical groups)
            - base_alpha (before rules), alpha (final)
            - applied_rules (list of rule names that fired)
        """
        n = user_stats.get('n_interactions', 0)
        f_quantity = self._compute_quantity_factor(n)
        f_diversity = self._compute_diversity_factor(
            user_stats.get('genre_entropy', self.config.max_entropy / 2)
        )
        f_coverage = self._compute_coverage_factor(
            user_stats.get('n_unique_items', n)
        )
        f_consistency = self._compute_consistency_factor(
            user_stats.get('rating_std', self.config.max_rating_std / 2)
        )

        # Hierarchical components
        data_volume = np.sqrt(f_quantity * f_coverage)
        if f_diversity + f_consistency > 0:
            preference_quality = 2 * f_diversity * f_consistency / (f_diversity + f_consistency)
        else:
            preference_quality = 0.0

        base_alpha_before_rules = (
            self.config.data_volume_weight * data_volume +
            self.config.preference_quality_weight * preference_quality
        )

        # Track which rules apply
        applied_rules = []
        if n < self.config.sparse_threshold:
            applied_rules.append('sparse')
        if (f_diversity < self.config.niche_diversity_threshold and
                f_quantity > self.config.niche_quantity_threshold):
            applied_rules.append('niche')
        if f_consistency < self.config.inconsistent_threshold:
            applied_rules.append('inconsistent')
        if (f_coverage > self.config.completionist_coverage and
                f_diversity < self.config.completionist_diversity):
            applied_rules.append('completionist')

        alpha = self.compute_from_stats(user_stats)

        return {
            'quantity': f_quantity,
            'diversity': f_diversity,
            'coverage': f_coverage,
            'consistency': f_consistency,
            'data_volume': data_volume,
            'preference_quality': preference_quality,
            'base_alpha_before_rules': base_alpha_before_rules,
            'alpha': alpha,
            'applied_rules': applied_rules,
        }

    def _compute_quantity_factor(self, n: int) -> float:
        """Sigmoid-normalized quantity factor."""
        x = (n - self.config.quantity_threshold) * self.config.quantity_temperature
        return _sigmoid(x)

    def _compute_diversity_factor(self, genre_entropy: float) -> float:
        """Linear-normalized diversity factor based on genre entropy."""
        return min(genre_entropy / self.config.max_entropy, 1.0)

    def _compute_coverage_factor(self, n_unique: int) -> float:
        """Linear-normalized coverage factor based on unique items."""
        return min(n_unique / self.config.coverage_threshold, 1.0)

    def _compute_consistency_factor(self, rating_std: float) -> float:
        """Inverse-normalized consistency factor (lower std = higher consistency)."""
        return 1.0 - min(rating_std / self.config.max_rating_std, 1.0)

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
            f"HierarchicalConditionalAlpha(\n"
            f"  method: hierarchical_conditional\n"
            f"  Stage 1 - Hierarchical Aggregation:\n"
            f"    data_volume = sqrt(f_quantity * f_coverage)\n"
            f"    preference_quality = 2*f_d*f_s / (f_d + f_s)  [harmonic mean]\n"
            f"    base_alpha = {self.config.data_volume_weight:.2f}*data_volume + "
            f"{self.config.preference_quality_weight:.2f}*preference_quality\n"
            f"  Stage 2 - Conditional Rules:\n"
            f"    Rule 1: Sparse (n < {self.config.sparse_threshold}) → "
            f"penalty up to {self.config.sparse_penalty_max:.0%}\n"
            f"    Rule 2: Niche (f_d < {self.config.niche_diversity_threshold}, "
            f"f_q > {self.config.niche_quantity_threshold}) → bonus {self.config.niche_bonus:+.2f}\n"
            f"    Rule 3: Inconsistent (f_s < {self.config.inconsistent_threshold}) → "
            f"penalty {self.config.inconsistent_penalty:.0%}\n"
            f"    Rule 4: Completionist (f_c > {self.config.completionist_coverage}, "
            f"f_d < {self.config.completionist_diversity}) → bonus {self.config.completionist_bonus:+.2f}\n"
            f"  range: [{self.config.min_alpha}, {self.config.max_alpha}]\n"
            f")"
        )


# Type alias for alpha computers
AlphaComputer = Union[DataQuantityAlpha, MultiFactorAlpha, HierarchicalConditionalAlpha]


def create_alpha_computer(
    config: Optional[AlphaConfig] = None,
    hc_config: Optional[HierarchicalConditionalAlphaConfig] = None,
) -> AlphaComputer:
    """Factory function to create appropriate alpha computer.

    Args:
        config: AlphaConfig instance, uses defaults if None
        hc_config: HierarchicalConditionalAlphaConfig for hierarchical_conditional method
                   If None and method is hierarchical_conditional, uses defaults

    Returns:
        DataQuantityAlpha, MultiFactorAlpha, or HierarchicalConditionalAlpha based on config.method
    """
    if config is None:
        config = AlphaConfig()

    method = config.method.lower()

    if method == "data_quantity":
        return DataQuantityAlpha(config)
    elif method == "multi_factor":
        return MultiFactorAlpha(config)
    elif method == "hierarchical_conditional":
        # Use provided hc_config or create one with values from AlphaConfig
        if hc_config is None:
            hc_config = HierarchicalConditionalAlphaConfig(
                min_alpha=config.min_alpha,
                max_alpha=config.max_alpha,
                quantity_threshold=config.quantity_threshold,
                quantity_temperature=config.quantity_temperature,
                max_entropy=config.max_entropy,
                coverage_threshold=config.coverage_threshold,
                max_rating_std=config.max_rating_std,
            )
        return HierarchicalConditionalAlpha(hc_config)
    else:
        raise ValueError(f"Unknown alpha method: {method}")
