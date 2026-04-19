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
