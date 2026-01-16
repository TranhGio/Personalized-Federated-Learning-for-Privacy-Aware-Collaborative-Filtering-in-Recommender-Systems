"""Evaluation utilities for federated personalized collaborative filtering."""

from .user_groups import (
    UserGroupConfig,
    classify_user_group,
    classify_users_by_group,
    get_group_statistics,
    aggregate_metrics_by_group,
    format_group_metrics,
)

from .alpha_analysis import (
    AlphaStatistics,
    AlphaAnalyzer,
)

__all__ = [
    # User groups
    "UserGroupConfig",
    "classify_user_group",
    "classify_users_by_group",
    "get_group_statistics",
    "aggregate_metrics_by_group",
    "format_group_metrics",
    # Alpha analysis
    "AlphaStatistics",
    "AlphaAnalyzer",
]
