"""
Comprehensive Evaluator for Federated Recommendation
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from loguru import logger

from .metrics import compute_all_metrics, aggregate_metrics


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    k_values: List[int] = None
    user_groups: Dict[str, Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [5, 10, 20]
        
        if self.user_groups is None:
            self.user_groups = {
                "sparse": (0, 30),
                "medium": (30, 100),
                "dense": (100, 10000)
            }


class FederatedEvaluator:
    """
    Evaluator for Federated Recommendation System.
    
    Supports:
    - Overall metrics
    - Per-user-group metrics (sparse, medium, dense)
    - Alpha analysis
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.results_history = []
    
    def evaluate_user(self,
                      user_id: int,
                      recommendations: List[int],
                      ground_truth: set,
                      user_stats: Dict,
                      alpha: float) -> Dict:
        """Evaluate a single user"""
        metrics = compute_all_metrics(
            recommendations=recommendations,
            ground_truth=ground_truth,
            k_values=self.config.k_values
        )
        
        metrics["user_id"] = user_id
        metrics["alpha"] = alpha
        metrics["n_interactions"] = user_stats.get("n_interactions", 0)
        metrics["n_test_items"] = len(ground_truth)
        
        return metrics
    
    def evaluate_all_users(self,
                           user_recommendations: Dict[int, List[int]],
                           user_ground_truth: Dict[int, set],
                           user_stats: Dict[int, Dict],
                           user_alphas: Dict[int, float]) -> Dict:
        """
        Evaluate all users and compute aggregate metrics.
        
        Returns:
            Dict with:
            - overall: Overall aggregated metrics
            - by_group: Metrics per user group
            - alpha_analysis: Analysis of alpha values
            - per_user: Raw per-user metrics
        """
        per_user_results = []
        
        for user_id in user_recommendations.keys():
            if user_id not in user_ground_truth:
                continue
            
            metrics = self.evaluate_user(
                user_id=user_id,
                recommendations=user_recommendations[user_id],
                ground_truth=user_ground_truth[user_id],
                user_stats=user_stats.get(user_id, {}),
                alpha=user_alphas.get(user_id, 0.5)
            )
            per_user_results.append(metrics)
        
        if not per_user_results:
            logger.warning("No users to evaluate!")
            return {}
        
        # Overall metrics
        overall = aggregate_metrics(per_user_results)
        
        # Group users
        grouped_results = self._group_users(per_user_results)
        
        # Metrics by group
        by_group = {}
        for group_name, group_results in grouped_results.items():
            if group_results:
                by_group[group_name] = aggregate_metrics(group_results)
                by_group[group_name]["n_users"] = len(group_results)
                by_group[group_name]["avg_alpha"] = np.mean(
                    [r["alpha"] for r in group_results]
                )
        
        # Alpha analysis
        alpha_analysis = self._analyze_alpha(per_user_results)
        
        result = {
            "overall": overall,
            "by_group": by_group,
            "alpha_analysis": alpha_analysis,
            "per_user": per_user_results,
            "n_users_evaluated": len(per_user_results)
        }
        
        self.results_history.append(result)
        
        return result
    
    def _group_users(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Group results by user category"""
        grouped = defaultdict(list)
        
        for r in results:
            n = r.get("n_interactions", 0)
            
            for group_name, (low, high) in self.config.user_groups.items():
                if low <= n < high:
                    grouped[group_name].append(r)
                    break
        
        return grouped
    
    def _analyze_alpha(self, results: List[Dict]) -> Dict:
        """Analyze alpha distribution and correlation with metrics"""
        alphas = [r["alpha"] for r in results]
        ndcgs = [r["ndcg@10"] for r in results]
        n_interactions = [r["n_interactions"] for r in results]
        
        analysis = {
            "alpha_mean": float(np.mean(alphas)),
            "alpha_std": float(np.std(alphas)),
            "alpha_min": float(np.min(alphas)),
            "alpha_max": float(np.max(alphas)),
            "alpha_median": float(np.median(alphas)),
        }
        
        # Correlation between alpha and metrics
        if len(set(alphas)) > 1:  # Need variance for correlation
            analysis["alpha_ndcg_correlation"] = float(
                np.corrcoef(alphas, ndcgs)[0, 1]
            )
            analysis["alpha_interactions_correlation"] = float(
                np.corrcoef(alphas, n_interactions)[0, 1]
            )
        
        # Alpha distribution by quantile
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        for q in quantiles:
            analysis[f"alpha_q{int(q*100)}"] = float(np.quantile(alphas, q))
        
        return analysis
    
    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        
        # Overall
        overall = results.get("overall", {})
        logger.info("\nOverall Metrics:")
        for k in self.config.k_values:
            hr = overall.get(f"mean_hitrate@{k}", 0)
            ndcg = overall.get(f"mean_ndcg@{k}", 0)
            logger.info(f"  K={k}: HR={hr:.4f}, NDCG={ndcg:.4f}")
        
        # By group
        by_group = results.get("by_group", {})
        logger.info("\nMetrics by User Group:")
        for group_name, metrics in by_group.items():
            n = metrics.get("n_users", 0)
            hr = metrics.get("mean_hitrate@10", 0)
            ndcg = metrics.get("mean_ndcg@10", 0)
            avg_alpha = metrics.get("avg_alpha", 0)
            logger.info(
                f"  {group_name}: n={n}, HR@10={hr:.4f}, "
                f"NDCG@10={ndcg:.4f}, α={avg_alpha:.3f}"
            )
        
        # Alpha analysis
        alpha = results.get("alpha_analysis", {})
        logger.info("\nAlpha Analysis:")
        logger.info(
            f"  Mean: {alpha.get('alpha_mean', 0):.3f} ± "
            f"{alpha.get('alpha_std', 0):.3f}"
        )
        logger.info(
            f"  Range: [{alpha.get('alpha_min', 0):.3f}, "
            f"{alpha.get('alpha_max', 0):.3f}]"
        )
        if "alpha_ndcg_correlation" in alpha:
            logger.info(
                f"  Correlation with NDCG@10: "
                f"{alpha.get('alpha_ndcg_correlation', 0):.3f}"
            )
        
        logger.info("=" * 60)