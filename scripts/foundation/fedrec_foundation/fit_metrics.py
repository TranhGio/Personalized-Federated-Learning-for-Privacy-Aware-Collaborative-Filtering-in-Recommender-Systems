"""Client-side fit-metrics contract (FND-05 + Codex CR-4).

Every federated module's ``@app.train()`` handler MUST return a dict whose
keys include ``FIT_METRICS_REQUIRED_KEYS`` so the weight-policy abstraction
in ``weight_policy.py`` has the inputs it needs. Modules construct a
``FitMetricsContract``, call ``.to_dict()``, and merge with per-module metrics.

Codex CR-4 flagged that clients currently return only ``num-examples``; this
module is the fixed import surface Phases 2-5 populate so
``compute_aggregation_weight`` has real data to consume.

``from_dict`` uses an explicit ``try/except`` so a missing required field
surfaces as a clear ``ValueError`` (not a cryptic dataclass ``TypeError``).

Phase 2 extension: ``FitMetricsContract`` gains 12 per-group
(sparse/medium/dense) and overall sufficient-stat fields
(``hit_count_*``, ``ndcg_sum_*``, ``evaluated_users*``) — all OPTIONAL,
default None. A sibling ``EvaluateMetricsContract`` +
``EVAL_METRICS_REQUIRED_KEYS`` + ``validate_evaluate_metrics`` govern the
evaluate-side wire payload (D-21 strict-contract, D-22 per-group). Both
contracts are populated by per-module clients in Phase 2 Plan 03 and
aggregated server-side via module-specific ``BaselineFedAvg.aggregate_evaluate``
(Phase 2 Plan 01 Task 2). ``validate_fit_metrics`` continues to check only
the Phase 1 FIT required keys; ``validate_evaluate_metrics`` governs the
evaluate-side payload.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Dict, FrozenSet, Optional, Union

FIT_METRICS_REQUIRED_KEYS = ("train_loss", "num_positives", "num_training_examples")


@dataclass
class FitMetricsContract:
    """Minimum metrics a federated client MUST return from ``@app.train()``.

    Phases 2-5 populate this in each module's client_app.py train handler.
    Weight-policy resolution consumes ``num_positives`` /
    ``num_training_examples`` (see ``weight_policy.compute_aggregation_weight``).

    Phase 2 Plan 01 (D-22) extension: 12 OPTIONAL per-group + overall
    sufficient-stat fields. They are populated client-side by Phase 2
    Plan 03 and summed server-side by ``BaselineFedAvg.aggregate_evaluate``
    (BSL-06). All default ``None`` and are dropped by ``to_dict`` so the
    Phase 1 contract remains backward-compatible.

    Attributes
    ----------
    train_loss : float
        Final training loss (after last local epoch).
    num_positives : int
        Count of positive train samples for this client (for
        ``WeightPolicy.NUM_POSITIVES``).
    num_training_examples : int
        Total train sample count including negatives (for
        ``WeightPolicy.NUM_TRAINING_EXAMPLES``).
    round_num : Optional[int]
        Current round number (optional; some modules log it here, some in
        ``FitRes``).
    hit_count_overall_at10 : Optional[int]
        Sufficient stat: overall count of held-out hits in the top-10.
    ndcg_sum_overall_at10 : Optional[float]
        Sufficient stat: sum of per-user NDCG@10 values across this client.
    evaluated_users : Optional[int]
        Sufficient stat: count of users on this client with a valid eval.
    hit_count_sparse_at10, ndcg_sum_sparse_at10, evaluated_users_sparse : optional
    hit_count_medium_at10, ndcg_sum_medium_at10, evaluated_users_medium : optional
    hit_count_dense_at10, ndcg_sum_dense_at10, evaluated_users_dense : optional
        Per-group sufficient-stat fields (D-22). One group carries the
        non-zero values for a given client; the other two carry zeros.
    """

    train_loss: float
    num_positives: int
    num_training_examples: int
    # Per-module extension fields below (optional; modules add their own).
    round_num: Optional[int] = None
    # --- Phase 2 extension (D-22): per-group + overall sufficient stats for BSL-06 aggregation. ---
    # All optional to preserve the Phase 1 contract. Populated client-side in Phase 2 Plan 03.
    hit_count_overall_at10: Optional[int] = None
    ndcg_sum_overall_at10: Optional[float] = None
    evaluated_users: Optional[int] = None
    hit_count_sparse_at10: Optional[int] = None
    ndcg_sum_sparse_at10: Optional[float] = None
    evaluated_users_sparse: Optional[int] = None
    hit_count_medium_at10: Optional[int] = None
    ndcg_sum_medium_at10: Optional[float] = None
    evaluated_users_medium: Optional[int] = None
    hit_count_dense_at10: Optional[int] = None
    ndcg_sum_dense_at10: Optional[float] = None
    evaluated_users_dense: Optional[int] = None
    # --- G-03-01 extension: client echoes its partition_id ---
    # Optional int; server uses discovery-round responses to build a
    # partition_id -> node_id mapping so per-round client sampling can run
    # in stable partition-id space (0..N-1) instead of Flower's ephemeral
    # os.urandom-seeded node_id space. Omitting the field keeps the Phase 1
    # contract backwards-compatible.
    partition_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Return a dict suitable for a Flower ``FitRes.metrics`` merge.

        ``None`` values are dropped so downstream aggregators don't see null.

        Returns
        -------
        Dict[str, int | float]
            A dict with all non-None dataclass fields.
        """
        raw = asdict(self)
        return {k: v for k, v in raw.items() if v is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, Union[int, float]]) -> "FitMetricsContract":
        """Construct from a dict, ignoring unknown keys (forward-compat).

        Parameters
        ----------
        d : Dict[str, int | float]
            Flat dict containing at least every required dataclass field.
            Unknown keys are ignored so modules can safely add new metrics
            without breaking contract deserialization.

        Returns
        -------
        FitMetricsContract
            A populated instance.

        Raises
        ------
        ValueError
            If the input dict is missing a required dataclass field. The
            underlying dataclass ``TypeError`` is caught and re-raised as a
            clear ``ValueError`` so callers don't see cryptic
            ``__init__ missing N required positional arguments`` messages.
        """
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        try:
            return cls(**filtered)  # type: ignore[arg-type]
        except TypeError as e:
            raise ValueError(
                f"FitMetricsContract.from_dict missing required field: {e}"
            ) from e


def validate_fit_metrics(metrics: Dict[str, Union[int, float]]) -> None:
    """Raise ValueError if metrics dict is missing a required contract key.

    Checks every entry in ``FIT_METRICS_REQUIRED_KEYS`` for presence and
    int/float type. Intended as a cheap server-side guard before calling
    ``compute_aggregation_weight``.

    Parameters
    ----------
    metrics : Dict[str, int | float]
        The client's returned ``FitRes.metrics``.

    Raises
    ------
    ValueError
        If any required key is missing or is not an int/float.
    """
    for key in FIT_METRICS_REQUIRED_KEYS:
        if key not in metrics:
            raise ValueError(
                f"Client metrics missing required key {key!r}. "
                f"Expected at least {list(FIT_METRICS_REQUIRED_KEYS)}; "
                f"got {sorted(metrics.keys())}."
            )
        if not isinstance(metrics[key], (int, float)):
            raise ValueError(
                f"Client metrics[{key!r}] must be int|float, "
                f"got {type(metrics[key]).__name__}."
            )


# ======================================================================
# Phase 2 Plan 01 (D-21, D-22): EvaluateMetricsContract.
#
# Separate strict-contract dataclass for @app.evaluate() wire payloads.
# Required fields are the 3 canonical sufficient-stat keys that the server
# aggregator (BaselineFedAvg._sum_sufficient_stats) reads; optional
# diagnostic fields (eval_loss / sampled_hr_at10 / sampled_ndcg_at10) are
# cached client-side for logs only; optional per-group fields mirror
# FitMetricsContract's 9 per-group keys. validate_evaluate_metrics enforces
# D-21: no free-form extras on the evaluate wire.
# ======================================================================


EVAL_METRICS_REQUIRED_KEYS: FrozenSet[str] = frozenset({
    "hit_count_overall_at10",
    "ndcg_sum_overall_at10",
    "evaluated_users",
})


@dataclass
class EvaluateMetricsContract:
    """Strict-contract wire payload for Flower ``@app.evaluate()`` responses (D-21, D-22).

    Required fields MUST be present in any ``to_dict()`` output (these are
    the canonical sufficient-stat keys that :func:`_sum_sufficient_stats` reads
    server-side; aggregation is sum-based, NOT averaged per-client ratios):
      - ``hit_count_overall_at10`` : int — client's overall hit-count sufficient stat.
      - ``ndcg_sum_overall_at10`` : float — client's overall NDCG-sum sufficient stat.
      - ``evaluated_users`` : int — client's evaluated-user count (denominator
        for server-side ratio reconstruction).

    Optional diagnostic fields (cached client-side for per-round logs only; NOT
    consumed by the server aggregator, since the server re-computes the
    headline ratios from summed sufficient stats):
      - ``eval_loss`` : Optional[float] — per-client weighted average loss.
      - ``sampled_hr_at10`` : Optional[float] — client-local HR@10 ratio.
      - ``sampled_ndcg_at10`` : Optional[float] — client-local NDCG@10 ratio.

    Optional per-group fields (mirror FitMetricsContract's 9 per-group keys):
      - ``hit_count_{sparse,medium,dense}_at10`` : Optional[int]
      - ``ndcg_sum_{sparse,medium,dense}_at10`` : Optional[float]
      - ``evaluated_users_{sparse,medium,dense}`` : Optional[int]

    The server's ``BaselineFedAvg.aggregate_evaluate`` (Phase 2 Plan 01 Task 2)
    reads these sufficient stats and emits server-side ratios; client-local
    ratios (``sampled_hr_at10`` / ``sampled_ndcg_at10``) are informational
    only (they're what Flower shows in its per-round dump before our strategy
    override runs).

    Parameters
    ----------
    hit_count_overall_at10 : int
        Client's overall hit-count sufficient stat.
    ndcg_sum_overall_at10 : float
        Client's overall NDCG-sum sufficient stat.
    evaluated_users : int
        Client's evaluated-user count.
    eval_loss : Optional[float]
        Informational per-client weighted average loss.
    sampled_hr_at10 : Optional[float]
        Client-local HR@10 ratio.
    sampled_ndcg_at10 : Optional[float]
        Client-local NDCG@10 ratio.
    hit_count_sparse_at10, ndcg_sum_sparse_at10, evaluated_users_sparse : optional
    hit_count_medium_at10, ndcg_sum_medium_at10, evaluated_users_medium : optional
    hit_count_dense_at10, ndcg_sum_dense_at10, evaluated_users_dense : optional
        Per-group sufficient-stat fields (D-22). Optional for backwards
        compatibility; Plan 03 Task 2 populates all three groups
        (sparse/medium/dense), with the client's group receiving the
        non-zero values and the other two groups receiving zeros.

    Returns
    -------
    EvaluateMetricsContract
        Dataclass instance.
    """
    # --- Required fields (sufficient stats; unified naming with FitMetricsContract + strategy reader). ---
    hit_count_overall_at10: int
    ndcg_sum_overall_at10: float
    evaluated_users: int
    # --- Optional diagnostic fields (cached client-side; NOT consumed by server aggregator). ---
    eval_loss: Optional[float] = None
    sampled_hr_at10: Optional[float] = None
    sampled_ndcg_at10: Optional[float] = None
    # --- Optional per-group fields (D-22): same 3 groups as FitMetricsContract. ---
    hit_count_sparse_at10: Optional[int] = None
    ndcg_sum_sparse_at10: Optional[float] = None
    evaluated_users_sparse: Optional[int] = None
    hit_count_medium_at10: Optional[int] = None
    ndcg_sum_medium_at10: Optional[float] = None
    evaluated_users_medium: Optional[int] = None
    hit_count_dense_at10: Optional[int] = None
    ndcg_sum_dense_at10: Optional[float] = None
    evaluated_users_dense: Optional[int] = None
    # --- G-03-01 extension: client echoes its partition_id ---
    # Optional int; discovery-round responses populate ONLY this field so the
    # server can build partition_id -> node_id before the main training loop.
    # Normal evaluate responses also populate it for audit-trail purposes.
    partition_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Serialize to a JSON-ready dict; drops None-valued optional fields.

        Returns
        -------
        Dict[str, int | float]
            All required fields plus any non-None optional per-group fields.
        """
        result: Dict[str, Union[int, float]] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                continue
            result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, payload: Dict[str, Union[int, float]]) -> "EvaluateMetricsContract":
        """Construct from a dict; filters unknown keys; raises ValueError on missing required.

        Parameters
        ----------
        payload : Dict[str, int | float]
            Dict possibly containing extra keys (filtered) and missing optional
            keys (set to None). Required keys absent will raise TypeError ->
            wrapped as ValueError.

        Returns
        -------
        EvaluateMetricsContract

        Raises
        ------
        ValueError
            If required keys are missing or field types are wrong.
        """
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in payload.items() if k in known}
        try:
            return cls(**filtered)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"EvaluateMetricsContract.from_dict failed: {exc}") from exc


def validate_evaluate_metrics(payload: Dict[str, Union[int, float]]) -> None:
    """Assert the evaluate-wire payload satisfies the strict contract (D-21, D-22).

    Checks:
      1. All keys in :data:`EVAL_METRICS_REQUIRED_KEYS` are present.
      2. No free-form extras that are NOT known fields of
         :class:`EvaluateMetricsContract`.

    Parameters
    ----------
    payload : Dict[str, int | float]
        Flower EvaluateRes.metrics dict (or MetricRecord contents).

    Raises
    ------
    ValueError
        If the payload is missing required keys OR contains free-form extras.
    """
    missing = sorted(EVAL_METRICS_REQUIRED_KEYS - set(payload.keys()))
    if missing:
        raise ValueError(
            f"EvaluateMetricsContract missing required keys: {missing}. "
            f"Got payload keys: {sorted(payload.keys())}"
        )
    known = {f.name for f in fields(EvaluateMetricsContract)}
    extras = sorted(set(payload.keys()) - known)
    if extras:
        raise ValueError(
            f"EvaluateMetricsContract rejects free-form extras (D-21): {extras}. "
            f"Known contract fields: {sorted(known)}"
        )
