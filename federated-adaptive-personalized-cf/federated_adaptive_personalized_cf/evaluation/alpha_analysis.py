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
