"""Split Learning Strategies for Federated Personalized Collaborative Filtering (Phase 3 Plan 01).

PersonalizedSplitFedAvg / PersonalizedSplitFedProx subclass Flower's FedAvg/FedProx
and override aggregate_evaluate to compute thesis metrics ONCE from summed sufficient
stats (sum(hit_count)/sum(evaluated_users)) instead of averaging per-client ratios.

aggregate_fit is INHERITED UNCHANGED from parent (D-23 split-learning invariant —
the client only sends GLOBAL params so FedAvg's weighted average of GLOBAL params is correct).

Parameter split vs Phase 2 baseline:
  - baseline: ALL params GLOBAL (aggregate_fit averages everything)
  - personalized: item_* GLOBAL, local_user_* LOCAL (D-03, aggregate_fit averages only GLOBAL)

Per-group sufficient stats live in FitMetricsContract (Phase 1 + Phase 2 Plan 01 D-22
extension). Each client's @app.evaluate() handler populates hit_count_{overall,sparse,
medium,dense}_at10, ndcg_sum_..._at10, and evaluated_users_{,sparse,medium,dense}.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from flwr.common import EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg as BaseFedAvg, FedProx as BaseFedProx


# D-03: flipped frozensets vs baseline (item_* GLOBAL, local_user_* LOCAL).
# PSN-06 single-row model contract: local_user_row + local_user_bias replace the
# old nn.Embedding(num_users, d) ghost table.
_GLOBAL_PARAM_KEYS = frozenset({
    "item_embeddings.weight",
    "item_bias.weight",
    "global_bias",
})
_LOCAL_PARAM_KEYS = frozenset({
    "local_user_row",
    "local_user_bias",
})

_SUFFICIENT_STAT_KEYS = (
    "hit_count_overall_at10", "ndcg_sum_overall_at10", "evaluated_users",
    "hit_count_sparse_at10", "ndcg_sum_sparse_at10", "evaluated_users_sparse",
    "hit_count_medium_at10", "ndcg_sum_medium_at10", "evaluated_users_medium",
    "hit_count_dense_at10",  "ndcg_sum_dense_at10",  "evaluated_users_dense",
)


def _sum_sufficient_stats(
    results: List[Tuple[ClientProxy, EvaluateRes]],
) -> Dict[str, Union[int, float]]:
    """Sum per-client sufficient-stat fields across all EvaluateRes responses.

    Each client returns the D-22 extended-contract fields via
    FitMetricsContract.to_dict() merged into EvaluateRes.metrics. Missing
    fields are treated as 0 (a client that reports nothing for a group
    contributes zero to that group).

    Parameters
    ----------
    results : list of (ClientProxy, EvaluateRes)
        Responses from Flower's evaluate phase.

    Returns
    -------
    Dict[str, int | float]
        Summed-across-clients totals for each stat.
    """
    totals: Dict[str, Union[int, float]] = {k: 0 for k in _SUFFICIENT_STAT_KEYS}
    for _proxy, eval_res in results:
        metrics = eval_res.metrics or {}
        for k in _SUFFICIENT_STAT_KEYS:
            v = metrics.get(k, 0) or 0
            totals[k] = totals[k] + v
    return totals


def _sufficient_stats_to_thesis_metrics(
    totals: Dict[str, Union[int, float]],
) -> Dict[str, Scalar]:
    """Convert summed sufficient stats into server-side ratio metrics.

    Computes overall and per-group ``sampled_hr@10`` / ``sampled_ndcg@10``
    as ``hit_count / evaluated_users``. Zero-division safe: a group with
    zero evaluated users gets 0.0 for both its HR and NDCG.

    Parameters
    ----------
    totals : Dict[str, int | float]
        Summed-across-clients sufficient stats from _sum_sufficient_stats.

    Returns
    -------
    Dict[str, Scalar]
        Thesis-table metrics dict with overall + per-group HR/NDCG plus
        per-group evaluated_users counts.
    """
    def _safe_ratio(num: Union[int, float], den: Union[int, float]) -> float:
        return float(num) / float(den) if den else 0.0

    metrics: Dict[str, Scalar] = {
        "sampled_hr@10": _safe_ratio(totals["hit_count_overall_at10"], totals["evaluated_users"]),
        "sampled_ndcg@10": _safe_ratio(totals["ndcg_sum_overall_at10"], totals["evaluated_users"]),
        "sampled_hr@10/sparse": _safe_ratio(totals["hit_count_sparse_at10"], totals["evaluated_users_sparse"]),
        "sampled_ndcg@10/sparse": _safe_ratio(totals["ndcg_sum_sparse_at10"], totals["evaluated_users_sparse"]),
        "sampled_hr@10/medium": _safe_ratio(totals["hit_count_medium_at10"], totals["evaluated_users_medium"]),
        "sampled_ndcg@10/medium": _safe_ratio(totals["ndcg_sum_medium_at10"], totals["evaluated_users_medium"]),
        "sampled_hr@10/dense": _safe_ratio(totals["hit_count_dense_at10"], totals["evaluated_users_dense"]),
        "sampled_ndcg@10/dense": _safe_ratio(totals["ndcg_sum_dense_at10"], totals["evaluated_users_dense"]),
        "evaluated_users": int(totals["evaluated_users"]),
        "evaluated_users_sparse": int(totals["evaluated_users_sparse"]),
        "evaluated_users_medium": int(totals["evaluated_users_medium"]),
        "evaluated_users_dense": int(totals["evaluated_users_dense"]),
    }
    return metrics


class PersonalizedSplitFedAvg(BaseFedAvg):
    """FedAvg variant for split learning with sufficient-stat aggregate_evaluate (D-20, PSN-04 server half).

    GLOBAL params: item_embeddings.weight, item_bias.weight, global_bias.
    LOCAL params: local_user_row, local_user_bias (on client only; never aggregated).
    aggregate_fit is inherited UNCHANGED — parent FedAvg averages the GLOBAL params the client sends.
    """

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Sum sufficient stats across clients, compute server-side ratios once.

        Parameters
        ----------
        server_round : int
            Current FL round number.
        results : list of (ClientProxy, EvaluateRes)
            Successful client evaluations.
        failures : list
            Failed/errored clients.

        Returns
        -------
        Tuple[Optional[float], Dict[str, Scalar]]
            ``(overall_loss, thesis_metrics_dict)``. ``overall_loss`` is
            the num-examples-weighted mean of per-client eval_loss (Flower
            convention). ``thesis_metrics_dict`` has the keys listed in
            :func:`_sufficient_stats_to_thesis_metrics`.
        """
        if not results:
            return None, {}
        totals = _sum_sufficient_stats(results)
        thesis_metrics = _sufficient_stats_to_thesis_metrics(totals)
        total_examples = sum(int(r.num_examples) for _, r in results) or 1
        loss = sum(float(r.loss) * int(r.num_examples) for _, r in results) / total_examples
        return float(loss), thesis_metrics


class PersonalizedSplitFedProx(BaseFedProx):
    """FedProx variant that reuses the sum-based aggregate_evaluate (D-20).

    aggregate_evaluate is an EXACT COPY of PersonalizedSplitFedAvg.aggregate_evaluate
    (not a super() call) to avoid diamond-inheritance with BaseFedProx; both use the
    module-level _sum_sufficient_stats + _sufficient_stats_to_thesis_metrics helpers.
    aggregate_fit is inherited from parent FedProx (proximal term is client-side;
    server aggregation is still FedAvg over GLOBAL params only).
    """

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Delegate to the same sufficient-stat aggregation as PersonalizedSplitFedAvg."""
        if not results:
            return None, {}
        totals = _sum_sufficient_stats(results)
        thesis_metrics = _sufficient_stats_to_thesis_metrics(totals)
        total_examples = sum(int(r.num_examples) for _, r in results) or 1
        loss = sum(float(r.loss) * int(r.num_examples) for _, r in results) / total_examples
        return float(loss), thesis_metrics


__all__ = [
    "PersonalizedSplitFedAvg",
    "PersonalizedSplitFedProx",
    "_GLOBAL_PARAM_KEYS",
    "_LOCAL_PARAM_KEYS",
]
