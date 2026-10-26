"""Tests for fedrec_foundation.weight_policy (FND-05) + fit_metrics (CR-4)."""
from __future__ import annotations

import pytest

from fedrec_foundation.weight_policy import WeightPolicy, compute_aggregation_weight
from fedrec_foundation.fit_metrics import (
    FitMetricsContract,
    FIT_METRICS_REQUIRED_KEYS,
    validate_fit_metrics,
)


def test_num_positives() -> None:
    """FND-05-a: num_positives policy weights clients by count of positive interactions."""
    assert compute_aggregation_weight({"num_positives": 10}, "num_positives") == 10.0
    assert compute_aggregation_weight({"num_positives": 0}, "num_positives") == 0.0


def test_num_training_examples() -> None:
    """FND-05-a: num_training_examples policy weights by total train sample count."""
    assert (
        compute_aggregation_weight({"num_training_examples": 123}, "num_training_examples")
        == 123.0
    )


def test_uniform_ignores_metrics() -> None:
    """FND-05-a: uniform policy returns 1.0 regardless of metrics."""
    assert compute_aggregation_weight({}, "uniform") == 1.0


def test_missing_key_raises() -> None:
    """FND-05-b: missing required metric key raises ValueError echoing the key."""
    with pytest.raises(ValueError, match="num_positives"):
        compute_aggregation_weight({"num_training_examples": 5}, "num_positives")


def test_unknown_policy_raises() -> None:
    """FND-05-b: an unknown policy string raises ValueError with the bad value echoed."""
    with pytest.raises(ValueError, match="Unknown"):
        compute_aggregation_weight({"num_positives": 1}, "made_up_policy")


def test_fit_metrics_contract() -> None:
    """FND-05-c + CR-4: client FitRes.metrics carries the documented policy keys.

    FitMetricsContract.to_dict() produces a dict carrying every required key
    for both num_positives and num_training_examples policies; the resulting
    dict passes validate_fit_metrics without error.
    """
    c = FitMetricsContract(
        train_loss=0.5, num_positives=20, num_training_examples=100, round_num=3
    )
    d = c.to_dict()
    assert d["train_loss"] == 0.5
    assert d["num_positives"] == 20
    assert d["num_training_examples"] == 100
    assert d["round_num"] == 3
    # validate_fit_metrics is happy with this dict.
    validate_fit_metrics(d)


def test_fit_metrics_contract_none_dropped() -> None:
    """CR-4: to_dict() drops None values so downstream aggregators don't see null."""
    c = FitMetricsContract(train_loss=0.1, num_positives=1, num_training_examples=2)
    d = c.to_dict()
    assert "round_num" not in d  # None dropped


def test_fit_metrics_contract_forward_compat() -> None:
    """CR-4: from_dict() ignores unknown keys so adding new metrics is safe."""
    d = {
        "train_loss": 0.1,
        "num_positives": 1,
        "num_training_examples": 2,
        "extra": "future_field",
    }
    c = FitMetricsContract.from_dict(d)
    assert c.train_loss == 0.1
    # Unknown keys ignored (don't crash).


def test_from_dict_missing_required_raises() -> None:
    """CR-4 polish: from_dict wraps dataclass TypeError in a clear ValueError.

    Missing-required-field calls must raise a ValueError whose message
    contains 'missing required field' — NOT a cryptic dataclass TypeError.
    """
    # Empty dict -> every required field missing.
    with pytest.raises(ValueError, match="missing required field"):
        FitMetricsContract.from_dict({})
    # Partial dict missing num_positives.
    with pytest.raises(ValueError, match="missing required field"):
        FitMetricsContract.from_dict({"train_loss": 0.1, "num_training_examples": 5})


def test_validate_fit_metrics_missing_raises() -> None:
    """CR-4: validate_fit_metrics raises ValueError when required key is absent."""
    with pytest.raises(ValueError, match="num_positives"):
        validate_fit_metrics({"train_loss": 0.1, "num_training_examples": 10})


def test_validate_fit_metrics_wrong_type_raises() -> None:
    """CR-4: validate_fit_metrics raises ValueError when required key has wrong type."""
    with pytest.raises(ValueError, match="int.*float"):
        validate_fit_metrics(
            {"train_loss": "oops", "num_positives": 1, "num_training_examples": 2}
        )


def test_required_keys_constant() -> None:
    """CR-4: FIT_METRICS_REQUIRED_KEYS is the exported, frozen required-key tuple."""
    assert FIT_METRICS_REQUIRED_KEYS == (
        "train_loss",
        "num_positives",
        "num_training_examples",
    )


# ======================================================================
# Phase 2 Plan 01 (D-22) extension: per-group + overall sufficient stats.
# FitMetricsContract gains 12 OPTIONAL fields populated client-side by
# Phase 2 Plan 03; BSL-06 sums them server-side via BaselineFedAvg.
# ======================================================================


def test_fit_metrics_per_group_fields() -> None:
    """D-22: FitMetricsContract carries per-group + overall sufficient stats."""
    contract = FitMetricsContract(
        train_loss=0.5, num_positives=30, num_training_examples=150, round_num=3,
        hit_count_overall_at10=24, ndcg_sum_overall_at10=12.5, evaluated_users=24,
        hit_count_sparse_at10=6, ndcg_sum_sparse_at10=2.0, evaluated_users_sparse=8,
        hit_count_medium_at10=10, ndcg_sum_medium_at10=5.0, evaluated_users_medium=10,
        hit_count_dense_at10=8, ndcg_sum_dense_at10=5.5, evaluated_users_dense=6,
    )
    d = contract.to_dict()
    for key in ["hit_count_overall_at10", "ndcg_sum_overall_at10", "evaluated_users",
                "hit_count_sparse_at10", "ndcg_sum_sparse_at10", "evaluated_users_sparse",
                "hit_count_medium_at10", "ndcg_sum_medium_at10", "evaluated_users_medium",
                "hit_count_dense_at10", "ndcg_sum_dense_at10", "evaluated_users_dense"]:
        assert key in d, f"missing per-group field {key}"


def test_fit_metrics_per_group_optional() -> None:
    """D-22: per-group fields default None and are DROPPED by to_dict (backward-compat)."""
    contract = FitMetricsContract(train_loss=0.5, num_positives=30, num_training_examples=150)
    d = contract.to_dict()
    assert "hit_count_sparse_at10" not in d
    assert "evaluated_users" not in d
    assert d == {"train_loss": 0.5, "num_positives": 30, "num_training_examples": 150}


def test_fit_metrics_forward_compat_with_per_group_extension() -> None:
    """D-22: forward-compat — unknown keys filtered, known per-group keys populated."""
    contract = FitMetricsContract.from_dict({
        "train_loss": 0.1, "num_positives": 2, "num_training_examples": 10,
        "hit_count_overall_at10": 1, "ndcg_sum_overall_at10": 0.63, "evaluated_users": 1,
        "alpha": 0.42,  # unknown — filtered
    })
    assert contract.hit_count_overall_at10 == 1
    assert contract.evaluated_users == 1
    assert contract.hit_count_sparse_at10 is None
