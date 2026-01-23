"""Alpha Distribution and Correlation Analysis.

Analyzes how adaptive alpha values are distributed across users
and their correlation with recommendation metrics.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class AlphaStatistics:
    """Statistics about alpha distribution."""
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile
    count: int


class AlphaAnalyzer:
    """Analyze alpha distribution and correlation with metrics.

    This class helps understand:
    1. How alpha values are distributed across users
    2. How alpha correlates with recommendation metrics
    3. Whether sparse users benefit from global prototype

    Example:
        analyzer = AlphaAnalyzer()
        analyzer.add_client_data(client_id=0, alpha=0.3, metrics={'ndcg@10': 0.15})
        analyzer.add_client_data(client_id=1, alpha=0.8, metrics={'ndcg@10': 0.25})
        stats = analyzer.compute_statistics()
        correlations = analyzer.compute_correlations()
    """

    def __init__(self):
        """Initialize analyzer with empty data."""
        self._alpha_values: List[float] = []
        self._metrics_by_alpha: Dict[str, List[Tuple[float, float]]] = {}
        self._client_data: Dict[int, Dict] = {}

    def add_client_data(
        self,
        client_id: int,
        alpha: float,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Add data from a client for analysis.

        Args:
            client_id: Client identifier
            alpha: Alpha value for the client
            metrics: Optional dict of metric name -> value
        """
        self._alpha_values.append(alpha)
        self._client_data[client_id] = {
            'alpha': alpha,
            'metrics': metrics or {}
        }

        # Track alpha-metric pairs for correlation
        if metrics:
            for metric_name, metric_value in metrics.items():
                if metric_name not in self._metrics_by_alpha:
                    self._metrics_by_alpha[metric_name] = []
                self._metrics_by_alpha[metric_name].append((alpha, metric_value))

    def add_user_alpha(
        self,
        user_id: int,
        alpha: float,
        n_interactions: Optional[int] = None
    ) -> None:
        """
        Add alpha value for a single user.

        Args:
            user_id: User identifier
            alpha: Alpha value
            n_interactions: Optional interaction count
        """
        self._alpha_values.append(alpha)

    def compute_statistics(self) -> AlphaStatistics:
        """
        Compute statistics about alpha distribution.

        Returns:
            AlphaStatistics dataclass with distribution info
        """
        if not self._alpha_values:
            return AlphaStatistics(
                mean=0.0, std=0.0, min=0.0, max=0.0,
                median=0.0, q25=0.0, q75=0.0, count=0
            )

        alphas = np.array(self._alpha_values)
        return AlphaStatistics(
            mean=float(np.mean(alphas)),
            std=float(np.std(alphas)),
            min=float(np.min(alphas)),
            max=float(np.max(alphas)),
            median=float(np.median(alphas)),
            q25=float(np.percentile(alphas, 25)),
            q75=float(np.percentile(alphas, 75)),
            count=len(alphas)
        )

    def compute_correlations(self) -> Dict[str, float]:
        """
        Compute Pearson correlation between alpha and each metric.

        Returns:
            Dict mapping metric name -> correlation coefficient
        """
        correlations = {}

        for metric_name, alpha_metric_pairs in self._metrics_by_alpha.items():
            if len(alpha_metric_pairs) < 3:
                # Need at least 3 points for meaningful correlation
                correlations[metric_name] = 0.0
                continue

            alphas = np.array([a for a, _ in alpha_metric_pairs])
            metrics = np.array([m for _, m in alpha_metric_pairs])

            # Check for zero variance
            if np.std(alphas) < 1e-10 or np.std(metrics) < 1e-10:
                correlations[metric_name] = 0.0
                continue

            correlation = np.corrcoef(alphas, metrics)[0, 1]
            correlations[metric_name] = float(correlation) if not np.isnan(correlation) else 0.0

        return correlations

    def group_by_alpha_range(
        self,
        ranges: Optional[List[Tuple[float, float]]] = None
    ) -> Dict[str, Dict]:
        """
        Group clients by alpha ranges and compute per-range metrics.

        Args:
            ranges: List of (min, max) alpha ranges. Defaults to:
                    [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]

        Returns:
            Dict mapping range label -> {count, mean_metrics}
        """
        if ranges is None:
            ranges = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]

        groups = {}
        for (low, high) in ranges:
            label = f"alpha_{low:.1f}_{high:.1f}"
            groups[label] = {
                'clients': [],
                'alphas': [],
                'metrics': {}
            }

        for client_id, data in self._client_data.items():
            alpha = data['alpha']
            for (low, high) in ranges:
                if low <= alpha < high or (high == 1.0 and alpha == 1.0):
                    label = f"alpha_{low:.1f}_{high:.1f}"
                    groups[label]['clients'].append(client_id)
                    groups[label]['alphas'].append(alpha)
                    for metric_name, metric_value in data['metrics'].items():
                        if metric_name not in groups[label]['metrics']:
                            groups[label]['metrics'][metric_name] = []
                        groups[label]['metrics'][metric_name].append(metric_value)
                    break

        # Compute summary statistics per group
        result = {}
        for label, group_data in groups.items():
            result[label] = {
                'count': len(group_data['clients']),
                'mean_alpha': float(np.mean(group_data['alphas'])) if group_data['alphas'] else 0.0,
                'mean_metrics': {
                    k: float(np.mean(v)) for k, v in group_data['metrics'].items()
                }
            }

        return result

    def get_summary(self) -> str:
        """
        Get a human-readable summary of alpha analysis.

        Returns:
            Formatted string for logging/display
        """
        stats = self.compute_statistics()
        correlations = self.compute_correlations()

        lines = [
            "Alpha Analysis Summary:",
            f"  Count: {stats.count}",
            f"  Mean: {stats.mean:.4f} (std: {stats.std:.4f})",
            f"  Range: [{stats.min:.4f}, {stats.max:.4f}]",
            f"  Quartiles: Q25={stats.q25:.4f}, Median={stats.median:.4f}, Q75={stats.q75:.4f}",
        ]

        if correlations:
            lines.append("  Correlations with metrics:")
            for metric_name, corr in sorted(correlations.items()):
                lines.append(f"    {metric_name}: {corr:.4f}")

        return '\n'.join(lines)

    def to_dict(self) -> Dict:
        """
        Export analysis results as a dictionary.

        Returns:
            Dict with statistics, correlations, and group analysis
        """
        stats = self.compute_statistics()
        return {
            'statistics': {
                'mean': stats.mean,
                'std': stats.std,
                'min': stats.min,
                'max': stats.max,
                'median': stats.median,
                'q25': stats.q25,
                'q75': stats.q75,
                'count': stats.count,
            },
            'correlations': self.compute_correlations(),
            'group_analysis': self.group_by_alpha_range(),
        }

    def reset(self) -> None:
        """Clear all stored data."""
        self._alpha_values = []
        self._metrics_by_alpha = {}
        self._client_data = {}


# =============================================================================
# Hierarchical Conditional Alpha Analysis Tools
# =============================================================================


@dataclass
class HierarchicalAlphaStatistics:
    """Statistics for hierarchical conditional alpha computation.

    Tracks both base factors and hierarchical components:
    - Base factors: quantity, diversity, coverage, consistency
    - Hierarchical groups: data_volume, preference_quality
    - Rule activations: sparse, niche, inconsistent, completionist
    """
    # Base factor statistics
    mean_quantity: float
    mean_diversity: float
    mean_coverage: float
    mean_consistency: float

    # Hierarchical group statistics
    mean_data_volume: float
    mean_preference_quality: float

    # Rule activation counts
    sparse_rule_count: int
    niche_rule_count: int
    inconsistent_rule_count: int
    completionist_rule_count: int
    total_users: int

    # Final alpha statistics
    mean_alpha: float
    std_alpha: float


class HierarchicalAlphaAnalyzer:
    """Analyzer specifically for HierarchicalConditionalAlpha method.

    Tracks:
    1. Individual factor contributions (quantity, diversity, coverage, consistency)
    2. Hierarchical group contributions (data_volume, preference_quality)
    3. Rule activation patterns (which rules fire most frequently)
    4. Comparison between multi-factor and hierarchical conditional methods

    Example:
        from federated_adaptive_personalized_cf.models import (
            HierarchicalConditionalAlpha, HierarchicalConditionalAlphaConfig
        )

        analyzer = HierarchicalAlphaAnalyzer()
        alpha_computer = HierarchicalConditionalAlpha()

        for user_id, user_stats in all_user_stats.items():
            factors = alpha_computer.compute_factors(user_stats)
            analyzer.add_user_factors(user_id, factors)

        stats = analyzer.compute_statistics()
        print(f"Data volume contribution: {stats.mean_data_volume:.4f}")
        print(f"Sparse rule activated for {stats.sparse_rule_count}/{stats.total_users} users")
    """

    def __init__(self):
        """Initialize analyzer with empty data."""
        self._user_factors: Dict[int, Dict] = {}

    def add_user_factors(self, user_id: int, factors: Dict) -> None:
        """
        Add computed factors for a user.

        Args:
            user_id: User identifier
            factors: Dict from HierarchicalConditionalAlpha.compute_factors()
                     containing quantity, diversity, coverage, consistency,
                     data_volume, preference_quality, applied_rules, alpha
        """
        self._user_factors[user_id] = factors

    def compute_statistics(self) -> HierarchicalAlphaStatistics:
        """
        Compute statistics across all tracked users.

        Returns:
            HierarchicalAlphaStatistics with factor and rule statistics
        """
        if not self._user_factors:
            return HierarchicalAlphaStatistics(
                mean_quantity=0.0, mean_diversity=0.0,
                mean_coverage=0.0, mean_consistency=0.0,
                mean_data_volume=0.0, mean_preference_quality=0.0,
                sparse_rule_count=0, niche_rule_count=0,
                inconsistent_rule_count=0, completionist_rule_count=0,
                total_users=0, mean_alpha=0.0, std_alpha=0.0
            )

        # Aggregate factor values
        quantities = []
        diversities = []
        coverages = []
        consistencies = []
        data_volumes = []
        preference_qualities = []
        alphas = []

        sparse_count = 0
        niche_count = 0
        inconsistent_count = 0
        completionist_count = 0

        for factors in self._user_factors.values():
            quantities.append(factors.get('quantity', 0))
            diversities.append(factors.get('diversity', 0))
            coverages.append(factors.get('coverage', 0))
            consistencies.append(factors.get('consistency', 0))
            data_volumes.append(factors.get('data_volume', 0))
            preference_qualities.append(factors.get('preference_quality', 0))
            alphas.append(factors.get('alpha', 0))

            applied_rules = factors.get('applied_rules', [])
            if 'sparse' in applied_rules:
                sparse_count += 1
            if 'niche' in applied_rules:
                niche_count += 1
            if 'inconsistent' in applied_rules:
                inconsistent_count += 1
            if 'completionist' in applied_rules:
                completionist_count += 1

        return HierarchicalAlphaStatistics(
            mean_quantity=float(np.mean(quantities)),
            mean_diversity=float(np.mean(diversities)),
            mean_coverage=float(np.mean(coverages)),
            mean_consistency=float(np.mean(consistencies)),
            mean_data_volume=float(np.mean(data_volumes)),
            mean_preference_quality=float(np.mean(preference_qualities)),
            sparse_rule_count=sparse_count,
            niche_rule_count=niche_count,
            inconsistent_rule_count=inconsistent_count,
            completionist_rule_count=completionist_count,
            total_users=len(self._user_factors),
            mean_alpha=float(np.mean(alphas)),
            std_alpha=float(np.std(alphas)),
        )

    def get_rule_activation_rates(self) -> Dict[str, float]:
        """
        Get the rate at which each rule is activated.

        Returns:
            Dict mapping rule name -> activation rate (0.0 to 1.0)
        """
        stats = self.compute_statistics()
        if stats.total_users == 0:
            return {'sparse': 0.0, 'niche': 0.0, 'inconsistent': 0.0, 'completionist': 0.0}

        return {
            'sparse': stats.sparse_rule_count / stats.total_users,
            'niche': stats.niche_rule_count / stats.total_users,
            'inconsistent': stats.inconsistent_rule_count / stats.total_users,
            'completionist': stats.completionist_rule_count / stats.total_users,
        }

    def get_hierarchical_contribution(self) -> Dict[str, float]:
        """
        Get the contribution of each hierarchical group to final alpha.

        Returns:
            Dict with data_volume and preference_quality contributions
        """
        stats = self.compute_statistics()
        return {
            'data_volume': stats.mean_data_volume,
            'preference_quality': stats.mean_preference_quality,
            'ratio': (stats.mean_data_volume / stats.mean_preference_quality
                     if stats.mean_preference_quality > 0 else float('inf')),
        }

    def get_factor_correlations(self) -> Dict[str, float]:
        """
        Compute correlations between base factors.

        Returns:
            Dict with pairwise correlation coefficients
        """
        if len(self._user_factors) < 3:
            return {}

        quantities = [f.get('quantity', 0) for f in self._user_factors.values()]
        diversities = [f.get('diversity', 0) for f in self._user_factors.values()]
        coverages = [f.get('coverage', 0) for f in self._user_factors.values()]
        consistencies = [f.get('consistency', 0) for f in self._user_factors.values()]

        def safe_corr(a, b):
            if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                return 0.0
            corr = np.corrcoef(a, b)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0

        return {
            'quantity_coverage': safe_corr(quantities, coverages),
            'quantity_diversity': safe_corr(quantities, diversities),
            'quantity_consistency': safe_corr(quantities, consistencies),
            'diversity_consistency': safe_corr(diversities, consistencies),
            'diversity_coverage': safe_corr(diversities, coverages),
            'coverage_consistency': safe_corr(coverages, consistencies),
        }

    def get_summary(self) -> str:
        """
        Get a human-readable summary of hierarchical alpha analysis.

        Returns:
            Formatted string for logging/display
        """
        stats = self.compute_statistics()
        rule_rates = self.get_rule_activation_rates()
        contrib = self.get_hierarchical_contribution()
        correlations = self.get_factor_correlations()

        lines = [
            "Hierarchical Conditional Alpha Analysis:",
            f"  Total users: {stats.total_users}",
            "",
            "  Base Factors (means):",
            f"    Quantity:    {stats.mean_quantity:.4f}",
            f"    Diversity:   {stats.mean_diversity:.4f}",
            f"    Coverage:    {stats.mean_coverage:.4f}",
            f"    Consistency: {stats.mean_consistency:.4f}",
            "",
            "  Hierarchical Groups:",
            f"    Data Volume:        {stats.mean_data_volume:.4f}",
            f"    Preference Quality: {stats.mean_preference_quality:.4f}",
            f"    DV/PQ Ratio:        {contrib['ratio']:.2f}",
            "",
            "  Rule Activation Rates:",
            f"    Sparse:       {rule_rates['sparse']*100:.1f}% ({stats.sparse_rule_count} users)",
            f"    Niche:        {rule_rates['niche']*100:.1f}% ({stats.niche_rule_count} users)",
            f"    Inconsistent: {rule_rates['inconsistent']*100:.1f}% ({stats.inconsistent_rule_count} users)",
            f"    Completionist:{rule_rates['completionist']*100:.1f}% ({stats.completionist_rule_count} users)",
            "",
            "  Final Alpha:",
            f"    Mean: {stats.mean_alpha:.4f} (std: {stats.std_alpha:.4f})",
        ]

        if correlations:
            lines.append("")
            lines.append("  Factor Correlations:")
            lines.append(f"    Quantity ↔ Coverage:    {correlations.get('quantity_coverage', 0):.3f}")
            lines.append(f"    Diversity ↔ Consistency:{correlations.get('diversity_consistency', 0):.3f}")

        return '\n'.join(lines)

    def reset(self) -> None:
        """Clear all stored data."""
        self._user_factors = {}


def compare_alpha_methods(
    user_stats_list: List[Dict],
    multi_factor_config,
    hierarchical_config,
) -> Dict[str, Dict]:
    """
    Compare alpha values between multi-factor and hierarchical conditional methods.

    Args:
        user_stats_list: List of user statistics dicts
        multi_factor_config: AlphaConfig with method="multi_factor"
        hierarchical_config: HierarchicalConditionalAlphaConfig

    Returns:
        Dict with comparison statistics:
        - 'multi_factor': {'mean', 'std', 'min', 'max'}
        - 'hierarchical': {'mean', 'std', 'min', 'max'}
        - 'difference': {'mean_diff', 'correlation', 'max_abs_diff'}
    """
    from federated_adaptive_personalized_cf.models import (
        MultiFactorAlpha, HierarchicalConditionalAlpha
    )

    mf_computer = MultiFactorAlpha(multi_factor_config)
    hc_computer = HierarchicalConditionalAlpha(hierarchical_config)

    mf_alphas = []
    hc_alphas = []

    for user_stats in user_stats_list:
        mf_alphas.append(mf_computer.compute_from_stats(user_stats))
        hc_alphas.append(hc_computer.compute_from_stats(user_stats))

    mf_arr = np.array(mf_alphas)
    hc_arr = np.array(hc_alphas)
    diff_arr = hc_arr - mf_arr

    # Compute correlation between methods
    if np.std(mf_arr) > 1e-10 and np.std(hc_arr) > 1e-10:
        corr = float(np.corrcoef(mf_arr, hc_arr)[0, 1])
    else:
        corr = 0.0

    return {
        'multi_factor': {
            'mean': float(np.mean(mf_arr)),
            'std': float(np.std(mf_arr)),
            'min': float(np.min(mf_arr)),
            'max': float(np.max(mf_arr)),
        },
        'hierarchical': {
            'mean': float(np.mean(hc_arr)),
            'std': float(np.std(hc_arr)),
            'min': float(np.min(hc_arr)),
            'max': float(np.max(hc_arr)),
        },
        'difference': {
            'mean_diff': float(np.mean(diff_arr)),
            'std_diff': float(np.std(diff_arr)),
            'max_abs_diff': float(np.max(np.abs(diff_arr))),
            'correlation': corr,
        },
    }
