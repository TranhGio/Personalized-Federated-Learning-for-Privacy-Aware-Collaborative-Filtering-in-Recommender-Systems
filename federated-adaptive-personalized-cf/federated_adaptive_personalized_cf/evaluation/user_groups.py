"""User Group Classification for Per-Group Metrics.

Classifies users into groups based on interaction count:
- Sparse: Few interactions, benefits from global prototype
- Medium: Moderate interactions, balanced personalization
- Dense: Many interactions, benefits from local personalization

These groups help analyze how adaptive personalization impacts
different user segments.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class UserGroupConfig:
    """Configuration for user group boundaries.

    Attributes:
        sparse: Tuple of (min, max) interaction count for sparse users
        medium: Tuple of (min, max) interaction count for medium users
        dense: Tuple of (min, max) interaction count for dense users
    """
    sparse: Tuple[int, int] = (0, 30)
    medium: Tuple[int, int] = (30, 100)
    dense: Tuple[int, int] = (100, 10000)

    def __post_init__(self):
        """Validate configuration."""
        if self.sparse[0] != 0:
            raise ValueError("Sparse group must start at 0")
        if self.sparse[1] != self.medium[0]:
            raise ValueError("Sparse max must equal medium min")
        if self.medium[1] != self.dense[0]:
            raise ValueError("Medium max must equal dense min")


def classify_user_group(
    n_interactions: int,
    config: Optional[UserGroupConfig] = None
) -> str:
    """
    Classify a user into a group based on interaction count.

    Args:
        n_interactions: Number of interactions for the user
        config: User group configuration (uses defaults if None)

    Returns:
        Group name: 'sparse', 'medium', or 'dense'
    """
    if config is None:
        config = UserGroupConfig()

    if n_interactions < config.sparse[1]:
        return 'sparse'
    elif n_interactions < config.medium[1]:
        return 'medium'
    else:
        return 'dense'


def classify_users_by_group(
    user_stats: Dict[int, Dict],
    config: Optional[UserGroupConfig] = None
) -> Dict[str, List[int]]:
    """
    Classify all users into groups.

    Args:
        user_stats: Dict mapping user_id -> user statistics dict
        config: User group configuration (uses defaults if None)

    Returns:
        Dict mapping group name -> list of user_ids
    """
    groups = {'sparse': [], 'medium': [], 'dense': []}

    for user_id, stats in user_stats.items():
        n_interactions = stats.get('n_interactions', 0)
        group = classify_user_group(n_interactions, config)
        groups[group].append(user_id)

    return groups


def get_group_statistics(
    user_stats: Dict[int, Dict],
    config: Optional[UserGroupConfig] = None
) -> Dict[str, Dict]:
    """
    Compute statistics for each user group.

    Args:
        user_stats: Dict mapping user_id -> user statistics dict
        config: User group configuration (uses defaults if None)

    Returns:
        Dict mapping group name -> group statistics dict
    """
    groups = classify_users_by_group(user_stats, config)

    statistics = {}
    for group_name, user_ids in groups.items():
        if not user_ids:
            statistics[group_name] = {
                'count': 0,
                'mean_interactions': 0.0,
                'std_interactions': 0.0,
                'min_interactions': 0,
                'max_interactions': 0,
            }
            continue

        interactions = [
            user_stats[uid].get('n_interactions', 0)
            for uid in user_ids
        ]

        statistics[group_name] = {
            'count': len(user_ids),
            'mean_interactions': float(np.mean(interactions)),
            'std_interactions': float(np.std(interactions)),
            'min_interactions': int(min(interactions)),
            'max_interactions': int(max(interactions)),
        }

    return statistics


def aggregate_metrics_by_group(
    user_metrics: Dict[int, Dict[str, float]],
    user_stats: Dict[int, Dict],
    config: Optional[UserGroupConfig] = None
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-user metrics by user group.

    Args:
        user_metrics: Dict mapping user_id -> metrics dict
        user_stats: Dict mapping user_id -> user statistics dict
        config: User group configuration (uses defaults if None)

    Returns:
        Dict mapping group name -> aggregated metrics dict
        Each metrics dict contains mean values for each metric
    """
    groups = classify_users_by_group(user_stats, config)

    group_metrics = {}
    for group_name, user_ids in groups.items():
        if not user_ids:
            group_metrics[group_name] = {}
            continue

        # Collect metrics for users in this group
        metrics_by_key = {}
        for uid in user_ids:
            if uid not in user_metrics:
                continue
            for key, value in user_metrics[uid].items():
                if key not in metrics_by_key:
                    metrics_by_key[key] = []
                metrics_by_key[key].append(value)

        # Compute mean for each metric
        group_metrics[group_name] = {
            key: float(np.mean(values))
            for key, values in metrics_by_key.items()
        }

    return group_metrics


def format_group_metrics(
    group_metrics: Dict[str, Dict[str, float]],
    metric_keys: Optional[List[str]] = None
) -> str:
    """
    Format group metrics as a human-readable string.

    Args:
        group_metrics: Dict mapping group name -> metrics dict
        metric_keys: List of metric keys to include (all if None)

    Returns:
        Formatted string for logging/display
    """
    lines = ["User Group Metrics:"]

    for group_name in ['sparse', 'medium', 'dense']:
        if group_name not in group_metrics:
            continue

        metrics = group_metrics[group_name]
        if not metrics:
            lines.append(f"  {group_name}: (no users)")
            continue

        keys = metric_keys if metric_keys else sorted(metrics.keys())
        metric_strs = [f"{k}={metrics[k]:.4f}" for k in keys if k in metrics]
        lines.append(f"  {group_name}: {', '.join(metric_strs)}")

    return '\n'.join(lines)
