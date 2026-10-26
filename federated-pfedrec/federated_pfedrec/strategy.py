"""PFedRec split-learning strategy with D-01 bias-GLOBAL classification.

Phase 5 changes vs prior SplitFedAvg:

- **D-01:** ``affine_output.bias`` moves from ``LOCAL_PARAM_KEYS`` to
  ``GLOBAL_PARAM_KEYS``. Source of truth:
  ``IJCAI-23-PFedRec/engine.py:143`` deletes ONLY ``affine_output.weight``
  from the per-user dict before aggregation; the bias stays in the dict
  passed to ``aggregate_clients_params`` and is averaged server-side.
  Closes CONCERNS divergence #9.
- **D-07:** ``SplitFedProx`` variant DROPPED. The IJCAI-23 reference uses
  FedAvg only; PFedRec's per-user score function does not benefit from a
  global proximal term (proximal scope would degenerate to a single tensor
  ``embedding_item.weight``).
- **D-12:** Class renamed to ``PFedRecSplitFedAvg`` to follow the
  cross-module module-prefixed convention (``BaselineFedAvg``,
  ``PersonalizedSplitFedAvg``, ``AdaptiveSplitFedAvg``).
- **D-24 / D-26:** ``aggregate_evaluate`` sums sufficient stats and divides
  ONCE at the end. With cross-device 1 user = 1 client this is
  mathematically uniform per-user and matches reference
  ``engine.py:81`` ``len(round_user_params)``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from flwr.common import EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg as BaseFedAvg


# D-01: bias is GLOBAL (engine.py:143 only deletes affine_output.weight).
GLOBAL_PARAM_KEYS = frozenset({
    "embedding_item.weight",
    "affine_output.bias",
})

# D-01: only affine_output.weight is LOCAL (per-user score function).
LOCAL_PARAM_KEYS = frozenset({
    "affine_output.weight",
})


# Module-level sufficient-stat helpers (Phase 3 PersonalizedSplitFedAvg
# pattern, frozensets flipped per D-01). These produce the same headline
# (sampled_hr@10, sampled_ndcg@10) regardless of FitRes.num_examples
# weighting because they sum sufficient stats and divide ONCE at the end.

_SUFFICIENT_STAT_KEYS: Tuple[str, ...] = (
    "hit_count_overall_at10", "ndcg_sum_overall_at10", "evaluated_users",
    "hit_count_sparse_at10", "ndcg_sum_sparse_at10", "evaluated_users_sparse",
    "hit_count_medium_at10", "ndcg_sum_medium_at10", "evaluated_users_medium",
    "hit_count_dense_at10",  "ndcg_sum_dense_at10",  "evaluated_users_dense",
)


def _sum_sufficient_stats(
    metrics_list: List[Dict[str, Union[int, float, None]]],
) -> Dict[str, float]:
    """Sum the 12 sufficient-stat fields across all client metric dicts.

    Parameters
    ----------
    metrics_list : List[Dict]
        Per-client metric dicts (each carrying the EvaluateMetricsContract
        keys defined in scripts/foundation/fedrec_foundation/fit_metrics.py).

    Returns
    -------
    Dict[str, float]
        Mapping ``field-name -> summed value across clients``. Missing
        fields are treated as 0.
    """
    sums: Dict[str, float] = {f: 0.0 for f in _SUFFICIENT_STAT_KEYS}
    for metrics in metrics_list:
        for field in _SUFFICIENT_STAT_KEYS:
            value = metrics.get(field, 0)
            if value is None:
                value = 0
            sums[field] += float(value)
    return sums


def _sufficient_stats_to_thesis_metrics(
    sums: Dict[str, float],
) -> Dict[str, Scalar]:
    """Convert summed sufficient stats to ratio-style thesis metrics.

    Parameters
    ----------
    sums : Dict[str, float]
        Output of :func:`_sum_sufficient_stats`.

    Returns
    -------
    Dict[str, Scalar]
        Keys: ``sampled_hr@10``, ``sampled_ndcg@10``, ``evaluated_users``
        plus per-group variants (``sparse`` / ``medium`` / ``dense``).
        Zero-safe — returns 0.0 on zero denominator.
    """

    def _ratio(num: float, den: float) -> float:
        return float(num) / float(den) if den else 0.0

    out: Dict[str, Scalar] = {
        "sampled_hr@10": _ratio(
            sums["hit_count_overall_at10"], sums["evaluated_users"]
        ),
        "sampled_ndcg@10": _ratio(
            sums["ndcg_sum_overall_at10"], sums["evaluated_users"]
        ),
        "evaluated_users": float(sums["evaluated_users"]),
    }
    for group in ("sparse", "medium", "dense"):
        out[f"sampled_hr@10/{group}"] = _ratio(
            sums[f"hit_count_{group}_at10"], sums[f"evaluated_users_{group}"]
        )
        out[f"sampled_ndcg@10/{group}"] = _ratio(
            sums[f"ndcg_sum_{group}_at10"], sums[f"evaluated_users_{group}"]
        )
        out[f"evaluated_users/{group}"] = float(sums[f"evaluated_users_{group}"])
    return out


class PFedRecSplitFedAvg(BaseFedAvg):
    """FedAvg variant for PFedRec with D-01 bias-GLOBAL + uniform weighting.

    Notes
    -----
    - ``aggregate_fit`` is INHERITED from ``BaseFedAvg`` unchanged. Phase 5
      ``server_app.py`` (Plan 04) sets ``FitRes.num_examples = 1`` per
      client (uniform weighting under D-24) so FedAvg's existing
      num_examples-weighted average is mathematically uniform — no
      ``aggregate_fit`` override needed.
    - ``aggregate_evaluate`` is OVERRIDDEN to sum sufficient stats and
      compute the ratio once at the end (BSL-06 / PSN-04 / ADP-06
      carry-forward).
    """

    def __init__(self, fraction_fit: float = 1.0, **kwargs):
        super().__init__(fraction_fit=fraction_fit, **kwargs)
        self.global_param_keys = GLOBAL_PARAM_KEYS
        self.local_param_keys = LOCAL_PARAM_KEYS
        self._is_split_learning = True

    def __repr__(self) -> str:
        return f"PFedRecSplitFedAvg(fraction_fit={self.fraction_fit})"

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Sum sufficient stats and compute the ratio once (D-24 / D-26).

        Parameters
        ----------
        server_round : int
            Current FL round number.
        results : list of (ClientProxy, EvaluateRes)
            Successful client evaluations. Each ``EvaluateRes.metrics`` is
            expected to carry the 12 sufficient-stat keys produced by
            ``EvaluateMetricsContract.to_dict()``.
        failures : list
            Failed clients (ignored — Flower already filtered).

        Returns
        -------
        Tuple[Optional[float], Dict[str, Scalar]]
            ``(eval_loss, thesis_metrics)``. ``eval_loss`` is the mean of
            per-client ``eval_loss`` diagnostics (or 0.0 when none was
            reported). ``thesis_metrics`` carries headline + per-group
            HR/NDCG plus evaluated-user counts.
        """
        if not results:
            return None, {}

        metric_dicts: List[Dict[str, Union[int, float, None]]] = []
        eval_losses: List[float] = []
        for _client_proxy, eval_res in results:
            metrics = (
                dict(eval_res.metrics)
                if hasattr(eval_res, "metrics") and eval_res.metrics is not None
                else {}
            )
            metric_dicts.append(metrics)
            loss_value = metrics.get("eval_loss")
            if loss_value is not None:
                eval_losses.append(float(loss_value))

        sums = _sum_sufficient_stats(metric_dicts)
        thesis = _sufficient_stats_to_thesis_metrics(sums)
        loss = (
            sum(eval_losses) / len(eval_losses) if eval_losses else 0.0
        )
        return float(loss), thesis


__all__ = [
    "PFedRecSplitFedAvg",
    "GLOBAL_PARAM_KEYS",
    "LOCAL_PARAM_KEYS",
    "_sum_sufficient_stats",
    "_sufficient_stats_to_thesis_metrics",
]
