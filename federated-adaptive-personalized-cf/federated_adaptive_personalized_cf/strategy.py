"""Split Learning Strategies for Federated Adaptive Personalized Collaborative Filtering (Phase 4 Plan 01).

AdaptiveSplitFedAvg / AdaptiveSplitFedProx subclass Flower's FedAvg/FedProx and:
1. Override aggregate_evaluate to emit thesis metrics ONCE from summed sufficient stats
   (sum(hit_count)/sum(evaluated_users)) instead of averaging per-client ratios — mirrors
   Phase 3 PersonalizedSplitFedAvg (ADP-06 server half).
2. Override aggregate_fit to call super().aggregate_fit (weighted-average of GLOBAL params,
   D-23) AND then update the server EMA prototype via _aggregate_prototypes — preserves
   existing adaptive behavior. This DIVERGES from Phase 3 where aggregate_fit was pure-
   inherited; the adaptive module genuinely needs server-side prototype aggregation.
3. Hold a ``best_prototype`` snapshot field alongside ``_global_prototype`` (ADP-03, D-05);
   server_app.py (Plan 05) calls ``snapshot_best_prototype(round_num, embedding_dim)`` when
   current_ndcg > best_metric.

Parameter split
---------------
GLOBAL: item_embeddings.weight, item_bias.weight, global_bias
LOCAL (base): user_embeddings.weight, user_bias.weight
LOCAL (dynamic): _logit_alpha.weight, _item_perturbation.weight, personal_mlp.*,
                 fusion_gate / fusion_layer.weight + fusion_layer.bias
    — appended at runtime by DualPersonalizedBPRMF._LOCAL_PARAMS property based on
      enable_per_user_alpha / enable_item_perturbation / fusion_type flags. The
      frozenset ``LOCAL_PARAM_KEYS_BASE`` below is the BASE set only.
"""
from __future__ import annotations

from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg as BaseFedAvg, FedProx as BaseFedProx


# D-03 frozensets mirror Phase 3 with the BASE local set; runtime expansion owned by model
GLOBAL_PARAM_KEYS = frozenset({
    "item_embeddings.weight",
    "item_bias.weight",
    "global_bias",
})
LOCAL_PARAM_KEYS_BASE = frozenset({
    "user_embeddings.weight",
    "user_bias.weight",
})

# Key used for user prototype in metrics (Phase 3 + earlier adaptive code kept unchanged)
USER_PROTOTYPE_KEY = "user_prototype"

# 12 sufficient-stat keys (Phase 2 Plan 01 + Phase 3 Plan 01 contract)
_SUFFICIENT_STAT_KEYS = (
    "hit_count_overall_at10", "ndcg_sum_overall_at10", "evaluated_users",
    "hit_count_sparse_at10", "ndcg_sum_sparse_at10", "evaluated_users_sparse",
    "hit_count_medium_at10", "ndcg_sum_medium_at10", "evaluated_users_medium",
    "hit_count_dense_at10",  "ndcg_sum_dense_at10",  "evaluated_users_dense",
)


def _sum_sufficient_stats(
    results: List[Tuple[ClientProxy, EvaluateRes]],
) -> Dict[str, Union[int, float]]:
    """Sum each of the 12 sufficient-stat keys across EvaluateRes payloads.

    Missing fields are treated as 0 (a client that reports nothing for a
    group contributes zero to that group).

    Parameters
    ----------
    results : list of (ClientProxy, EvaluateRes)
        Successful client evaluations.

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
    """Derive thesis headline metrics from summed sufficient stats.

    Computes overall and per-group ``sampled_hr@10`` / ``sampled_ndcg@10``
    as ``hit_count / evaluated_users``. Zero-division safe: a group with
    zero evaluated users gets 0.0 for both its HR and NDCG.

    Parameters
    ----------
    totals : Dict[str, int | float]
        Summed-across-clients sufficient stats from
        :func:`_sum_sufficient_stats`.

    Returns
    -------
    Dict[str, Scalar]
        Thesis-table metrics dict with overall + per-group HR/NDCG plus
        per-group evaluated_users counts.
    """
    def _safe_ratio(num: Union[int, float], den: Union[int, float]) -> float:
        return float(num) / float(den) if den else 0.0

    return {
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


class AdaptiveSplitFedAvg(BaseFedAvg):
    """FedAvg variant for adaptive split learning with prototype EMA + best-round snapshot.

    aggregate_fit: ``super().aggregate_fit`` runs the weighted-average of GLOBAL params
                   (D-23) then ``_aggregate_prototypes`` updates ``self._global_prototype``
                   via EMA.
    aggregate_evaluate: sum-based sufficient-stat ratio (ADP-06).
    best_prototype: Optional[np.ndarray] — server-side snapshot field mirroring the
                    Phase 2 D-27 best_arrays pattern (ADP-03, D-05).
    """

    def __init__(
        self,
        fraction_fit: float = 1.0,
        prototype_momentum: float = 0.9,
        **kwargs,
    ):
        """Initialize AdaptiveSplitFedAvg.

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients sampled per round. Defaults to 1.0.
        prototype_momentum : float, optional
            EMA momentum for the server-side global user prototype.
            Defaults to 0.9 (half-life ~6 rounds).
        **kwargs :
            Additional kwargs forwarded to :class:`flwr.server.strategy.FedAvg`.
        """
        super().__init__(fraction_fit=fraction_fit, **kwargs)
        self.global_param_keys = GLOBAL_PARAM_KEYS
        self.local_param_keys_base = LOCAL_PARAM_KEYS_BASE
        self._is_split_learning = True
        self.prototype_momentum = prototype_momentum
        self._global_prototype: Optional[np.ndarray] = None
        # D-05: best-round snapshot field; server_app.py (Plan 05) calls snapshot_best_prototype.
        self.best_prototype: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return (
            f"AdaptiveSplitFedAvg(fraction_fit={self.fraction_fit}, "
            f"prototype_momentum={self.prototype_momentum})"
        )

    def get_global_prototype(self) -> Optional[np.ndarray]:
        """Return the current server-side global user prototype (EMA), or None if never aggregated."""
        return self._global_prototype

    def snapshot_best_prototype(self, round_num: int, embedding_dim: int) -> None:
        """D-05 best-round snapshot. Called by server_app.py when current_ndcg > best_metric.

        If ``_global_prototype`` is not None, copy it. Else (D-08 degenerate case: best
        round fires before any prototype was aggregated), snapshot ``np.zeros(embedding_dim)``
        and emit WARNING.

        Parameters
        ----------
        round_num : int
            Current FL round number (used only for the warning message).
        embedding_dim : int
            Dimension of the degenerate zero-vector fallback.
        """
        if self._global_prototype is not None:
            self.best_prototype = self._global_prototype.copy()
        else:
            self.best_prototype = np.zeros(int(embedding_dim), dtype=np.float32)
            log(
                WARNING,
                f"Prototype snapshot at best round R={round_num} is zero vector "
                f"— no prior prototype aggregation yet.",
            )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate GLOBAL params via super(), then update server EMA prototype.

        This override exists so prototype aggregation CONTINUES to run (existing adaptive
        behavior) — Phase 3's pure-inheritance approach is insufficient because the adaptive
        module has server-side state beyond just GLOBAL params.
        """
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)
        self._aggregate_prototypes(results)
        if self._global_prototype is not None:
            metrics["global_prototype_norm"] = float(np.linalg.norm(self._global_prototype))
        return aggregated_params, metrics

    def _aggregate_prototypes(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> None:
        """Weighted-mean client prototypes then EMA update on ``self._global_prototype``.

        Preserves existing behavior. Missing prototypes (client didn't contribute) are
        skipped; if no client contributed this round, the field stays unchanged.
        """
        prototypes_and_weights: List[Tuple[np.ndarray, int]] = []
        for _proxy, fit_res in results:
            metrics = fit_res.metrics or {}
            proto = metrics.get(USER_PROTOTYPE_KEY)
            if isinstance(proto, (list, tuple)):
                arr = np.asarray(proto, dtype=np.float32)
                prototypes_and_weights.append((arr, int(fit_res.num_examples)))
        if not prototypes_and_weights:
            return
        total_weight = sum(w for _, w in prototypes_and_weights)
        new_prototype = sum(p * w for p, w in prototypes_and_weights) / total_weight
        if self._global_prototype is None:
            self._global_prototype = new_prototype
        else:
            m = self.prototype_momentum
            self._global_prototype = m * self._global_prototype + (1.0 - m) * new_prototype

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Sum sufficient stats across clients; emit thesis metrics dict (ADP-06)."""
        if not results:
            return None, {}
        totals = _sum_sufficient_stats(results)
        thesis_metrics = _sufficient_stats_to_thesis_metrics(totals)
        total_examples = sum(int(r.num_examples) for _, r in results) or 1
        loss = sum(float(r.loss) * int(r.num_examples) for _, r in results) / total_examples
        return float(loss), thesis_metrics


class AdaptiveSplitFedProx(BaseFedProx):
    """FedProx variant that reuses the sum-based aggregate_evaluate + prototype EMA.

    aggregate_evaluate is an EXACT COPY of ``AdaptiveSplitFedAvg.aggregate_evaluate``
    (not a super() call) to avoid diamond-inheritance with BaseFedProx.
    aggregate_fit is overridden the same way as ``AdaptiveSplitFedAvg``.
    ``best_prototype`` field + ``snapshot_best_prototype`` helper are duplicated at
    the class level for the same reason.
    """

    def __init__(
        self,
        fraction_fit: float = 1.0,
        prototype_momentum: float = 0.9,
        proximal_mu: float = 0.01,
        **kwargs,
    ):
        """Initialize AdaptiveSplitFedProx.

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients sampled per round. Defaults to 1.0.
        prototype_momentum : float, optional
            EMA momentum for the server-side global user prototype.
            Defaults to 0.9.
        proximal_mu : float, optional
            FedProx proximal term coefficient (applied client-side on GLOBAL
            params only under split learning). Defaults to 0.01.
        **kwargs :
            Additional kwargs forwarded to :class:`flwr.server.strategy.FedProx`.
        """
        super().__init__(fraction_fit=fraction_fit, proximal_mu=proximal_mu, **kwargs)
        self.global_param_keys = GLOBAL_PARAM_KEYS
        self.local_param_keys_base = LOCAL_PARAM_KEYS_BASE
        self._is_split_learning = True
        self.prototype_momentum = prototype_momentum
        self._global_prototype: Optional[np.ndarray] = None
        self.best_prototype: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return (
            f"AdaptiveSplitFedProx(fraction_fit={self.fraction_fit}, "
            f"proximal_mu={self.proximal_mu}, "
            f"prototype_momentum={self.prototype_momentum})"
        )

    def get_global_prototype(self) -> Optional[np.ndarray]:
        """Return the current server-side global user prototype (EMA)."""
        return self._global_prototype

    def snapshot_best_prototype(self, round_num: int, embedding_dim: int) -> None:
        """D-05 best-round snapshot (FedProx branch; identical semantics to FedAvg)."""
        if self._global_prototype is not None:
            self.best_prototype = self._global_prototype.copy()
        else:
            self.best_prototype = np.zeros(int(embedding_dim), dtype=np.float32)
            log(
                WARNING,
                f"Prototype snapshot at best round R={round_num} is zero vector "
                f"— no prior prototype aggregation yet.",
            )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate GLOBAL params via super() FedProx, then update server EMA prototype."""
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)
        self._aggregate_prototypes(results)
        if self._global_prototype is not None:
            metrics["global_prototype_norm"] = float(np.linalg.norm(self._global_prototype))
        return aggregated_params, metrics

    def _aggregate_prototypes(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> None:
        """Weighted-mean + EMA update — exact copy of :meth:`AdaptiveSplitFedAvg._aggregate_prototypes`."""
        prototypes_and_weights: List[Tuple[np.ndarray, int]] = []
        for _proxy, fit_res in results:
            metrics = fit_res.metrics or {}
            proto = metrics.get(USER_PROTOTYPE_KEY)
            if isinstance(proto, (list, tuple)):
                arr = np.asarray(proto, dtype=np.float32)
                prototypes_and_weights.append((arr, int(fit_res.num_examples)))
        if not prototypes_and_weights:
            return
        total_weight = sum(w for _, w in prototypes_and_weights)
        new_prototype = sum(p * w for p, w in prototypes_and_weights) / total_weight
        if self._global_prototype is None:
            self._global_prototype = new_prototype
        else:
            m = self.prototype_momentum
            self._global_prototype = m * self._global_prototype + (1.0 - m) * new_prototype

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Delegate to the same sufficient-stat aggregation as :class:`AdaptiveSplitFedAvg`."""
        if not results:
            return None, {}
        totals = _sum_sufficient_stats(results)
        thesis_metrics = _sufficient_stats_to_thesis_metrics(totals)
        total_examples = sum(int(r.num_examples) for _, r in results) or 1
        loss = sum(float(r.loss) * int(r.num_examples) for _, r in results) / total_examples
        return float(loss), thesis_metrics


__all__ = [
    "AdaptiveSplitFedAvg",
    "AdaptiveSplitFedProx",
    "GLOBAL_PARAM_KEYS",
    "LOCAL_PARAM_KEYS_BASE",
    "USER_PROTOTYPE_KEY",
]
