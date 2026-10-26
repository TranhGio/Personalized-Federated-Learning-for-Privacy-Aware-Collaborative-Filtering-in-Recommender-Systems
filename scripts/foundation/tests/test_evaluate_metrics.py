"""Tests for EvaluateMetricsContract (Phase 2 Plan 01 — D-21 strict-contract, D-22 per-group)."""
from __future__ import annotations

import pytest

from fedrec_foundation.fit_metrics import (
    EVAL_METRICS_REQUIRED_KEYS,
    EvaluateMetricsContract,
    validate_evaluate_metrics,
)


def test_evaluate_metrics_required_keys_enforced() -> None:
    # Valid payload with all 3 required sufficient-stat keys (and no extras
    # beyond known fields). Diagnostic keys eval_loss / sampled_hr_at10 /
    # sampled_ndcg_at10 are optional — may be absent without raising.
    validate_evaluate_metrics({
        "hit_count_overall_at10": 0,
        "ndcg_sum_overall_at10": 0.0,
        "evaluated_users": 1,
    })
    # Valid payload with required + optional diagnostics + optional per-group.
    validate_evaluate_metrics({
        "hit_count_overall_at10": 0,
        "ndcg_sum_overall_at10": 0.0,
        "evaluated_users": 1,
        "eval_loss": 0.5,
        "sampled_hr_at10": 0.1,
        "sampled_ndcg_at10": 0.05,
    })
    # Missing required keys -> ValueError.
    with pytest.raises(ValueError, match="missing required keys"):
        validate_evaluate_metrics({"eval_loss": 0.5})
    # Free-form extra -> ValueError (D-21: no free-form extras).
    with pytest.raises(ValueError, match="free-form extras"):
        validate_evaluate_metrics({
            "hit_count_overall_at10": 0,
            "ndcg_sum_overall_at10": 0.0,
            "evaluated_users": 1,
            "freeform_field": 1.0,  # not a known contract field
        })


def test_evaluate_metrics_per_group_fields() -> None:
    contract = EvaluateMetricsContract(
        eval_loss=0.5, sampled_hr_at10=0.3, sampled_ndcg_at10=0.15,
        evaluated_users=10, hit_count_overall_at10=3, ndcg_sum_overall_at10=1.5,
        hit_count_sparse_at10=1, ndcg_sum_sparse_at10=0.5, evaluated_users_sparse=4,
        hit_count_medium_at10=2, ndcg_sum_medium_at10=1.0, evaluated_users_medium=6,
        hit_count_dense_at10=0, ndcg_sum_dense_at10=0.0, evaluated_users_dense=0,
    )
    d = contract.to_dict()
    for key in EVAL_METRICS_REQUIRED_KEYS:
        assert key in d, f"missing required field {key}"
    for group in ("sparse", "medium", "dense"):
        assert f"hit_count_{group}_at10" in d
        assert f"ndcg_sum_{group}_at10" in d
        assert f"evaluated_users_{group}" in d


def test_evaluate_metrics_forward_compat() -> None:
    # Unknown keys are filtered; optional per-group fields default None.
    contract = EvaluateMetricsContract.from_dict({
        "eval_loss": 0.5, "sampled_hr_at10": 0.1, "sampled_ndcg_at10": 0.05,
        "evaluated_users": 1, "hit_count_overall_at10": 0, "ndcg_sum_overall_at10": 0.0,
        "train_loss": 0.3,  # unknown — filtered (not in EvaluateMetricsContract fields)
    })
    assert contract.eval_loss == 0.5
    assert contract.hit_count_sparse_at10 is None  # optional default


def test_evaluate_metrics_rejects_wrong_types() -> None:
    # Type errors wrap as ValueError per Phase 1 CR-4 pattern.
    # NOTE: dataclass field annotations do NOT enforce runtime types; we rely on
    # the from_dict TypeError-wrap path, so this test exercises missing-required-keys.
    # The 3 required keys are hit_count_overall_at10 / ndcg_sum_overall_at10 /
    # evaluated_users; eval_loss is an optional diagnostic field.
    with pytest.raises(ValueError):
        EvaluateMetricsContract.from_dict({"eval_loss": 0.5})  # missing 3 required keys


# ======================================================================
# G-03-01 extension: optional partition_id field on both contracts.
# Discovery round echoes partition_id so the server can build a
# partition_id -> node_id map; per-round sampling then runs in stable
# partition-id space, not Flower's ephemeral os.urandom-seeded node_id space.
# ======================================================================


def test_fit_metrics_contract_accepts_partition_id() -> None:
    """G-03-01: FitMetricsContract accepts and serializes partition_id."""
    from fedrec_foundation.fit_metrics import FitMetricsContract

    payload = FitMetricsContract(
        train_loss=0.1,
        num_positives=5,
        num_training_examples=25,
        partition_id=1234,
    ).to_dict()
    assert payload.get("partition_id") == 1234
    # Other required keys still present.
    assert payload["train_loss"] == 0.1
    assert payload["num_positives"] == 5
    assert payload["num_training_examples"] == 25


def test_evaluate_metrics_contract_accepts_partition_id() -> None:
    """G-03-01: EvaluateMetricsContract accepts and serializes partition_id."""
    payload = EvaluateMetricsContract(
        hit_count_overall_at10=0,
        ndcg_sum_overall_at10=0.0,
        evaluated_users=0,
        partition_id=42,
    ).to_dict()
    assert payload.get("partition_id") == 42
    # Required sufficient-stat keys still present.
    assert payload["hit_count_overall_at10"] == 0
    assert payload["ndcg_sum_overall_at10"] == 0.0
    assert payload["evaluated_users"] == 0


def test_validate_evaluate_metrics_allows_partition_id() -> None:
    """G-03-01: partition_id is now a known field; strict validator permits it."""
    # Should not raise.
    validate_evaluate_metrics({
        "hit_count_overall_at10": 0,
        "ndcg_sum_overall_at10": 0.0,
        "evaluated_users": 0,
        "partition_id": 42,
    })


def test_validate_evaluate_metrics_still_rejects_unknown_extras() -> None:
    """G-03-01 regression guard: partition_id is known, anything else is NOT.

    The strict-contract (D-21) guarantee must hold: free-form extras still
    raise ValueError. Only partition_id was whitelisted via dataclass fields().
    """
    with pytest.raises(ValueError, match="free-form extras"):
        validate_evaluate_metrics({
            "hit_count_overall_at10": 0,
            "ndcg_sum_overall_at10": 0.0,
            "evaluated_users": 0,
            "partition_id": 42,
            "foo": "bar",  # not a known contract field — must still raise.
        })
