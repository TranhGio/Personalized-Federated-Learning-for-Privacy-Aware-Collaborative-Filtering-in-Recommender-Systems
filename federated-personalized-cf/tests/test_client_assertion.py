"""Client-app assertions + contract-payload tests (Phase 3 Plan 03).

Covers:
  - PSN-02: ``assert_benchmark_one_user_per_client`` raises
    ``AssertionError`` when ``num_users_in_client != 1`` under benchmark
    profile; a visible ``num_supernodes`` override bypasses the lock
    with a log line.
  - PSN-04 (client half) / BSL-07-style: ``get_primary_evaluator(mode)``
    resolves to ``"sampled_loo_99"`` for every recognized mode.
  - D-21 + G-03-01 carry-forward: ``FitMetricsContract`` and
    ``EvaluateMetricsContract`` carry optional ``partition_id`` in their
    ``to_dict()`` output + per-group sufficient-stat fields; free-form
    extras are rejected by ``validate_evaluate_metrics``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[2] / "data" / "derived" / "foundation_index.json"
    ).exists(),
    reason="foundation bundle not committed",
)


def test_benchmark_mode_asserts_one_user() -> None:
    """PSN-02: benchmark_cross_device with >1 user per client raises.

    Single-user partitions should pass through without raising;
    multi-user partitions should fail loud before any training happens.
    """
    from fedrec_foundation.mode import (
        assert_benchmark_one_user_per_client,
        resolve_mode_defaults,
    )

    profile = resolve_mode_defaults("benchmark_cross_device")
    with pytest.raises(AssertionError, match="exactly one user"):
        assert_benchmark_one_user_per_client(
            profile, num_users_in_client=3, overrides={}
        )
    # Single user — no raise.
    assert_benchmark_one_user_per_client(
        profile, num_users_in_client=1, overrides={}
    )


def test_benchmark_mode_skipped_with_override() -> None:
    """D-10: visible num-supernodes override bypasses the lock with a log line."""
    from fedrec_foundation.mode import (
        assert_benchmark_one_user_per_client,
        resolve_mode_defaults,
    )

    profile = resolve_mode_defaults("benchmark_cross_device")
    # 50 users in partition would normally raise, but the visible
    # override logs and returns instead of raising.
    assert_benchmark_one_user_per_client(
        profile,
        num_users_in_client=50,
        overrides={"num_supernodes": 10},
    )


def test_get_primary_evaluator_selects_sampled_loo_99() -> None:
    """PSN-04 client half / BSL-07-style: all three recognized modes route to sampled_loo_99."""
    from fedrec_foundation.evaluator import get_primary_evaluator

    for mode in ("benchmark_cross_device", "paper_compat_pfedrec", "cross_silo_legacy"):
        assert get_primary_evaluator(mode) == "sampled_loo_99"


def test_fit_metrics_contract_payload_shape_with_partition_id() -> None:
    """D-21 + G-03-01: FitMetricsContract per-group fields + optional partition_id in to_dict()."""
    from fedrec_foundation.fit_metrics import FitMetricsContract, validate_fit_metrics

    d = FitMetricsContract(
        train_loss=0.5,
        num_positives=1,
        num_training_examples=5,
        round_num=2,
        partition_id=42,
        hit_count_overall_at10=1,
        ndcg_sum_overall_at10=0.1,
        evaluated_users=1,
        hit_count_sparse_at10=1,
        ndcg_sum_sparse_at10=0.1,
        evaluated_users_sparse=1,
        hit_count_medium_at10=0,
        ndcg_sum_medium_at10=0.0,
        evaluated_users_medium=0,
        hit_count_dense_at10=0,
        ndcg_sum_dense_at10=0.0,
        evaluated_users_dense=0,
    ).to_dict()
    validate_fit_metrics(d)
    assert d.get("partition_id") == 42, "G-03-01: partition_id must appear in to_dict()"
    for key in (
        "train_loss",
        "num_positives",
        "num_training_examples",
        "partition_id",
        "hit_count_overall_at10",
        "evaluated_users_sparse",
        "evaluated_users_medium",
        "evaluated_users_dense",
    ):
        assert key in d, f"missing {key}"


def test_evaluate_metrics_contract_payload_shape_with_partition_id() -> None:
    """D-21 + G-03-01: EvaluateMetricsContract carries partition_id + rejects free-form extras.

    Plan 03 Task 2 populates ``partition_id=partition_id`` on every
    contract build — both the main @app.evaluate() reply AND the
    discover_only short-circuit. Verifying the shape here catches
    contract drift before wire transmission.
    """
    from fedrec_foundation.fit_metrics import (
        EvaluateMetricsContract,
        validate_evaluate_metrics,
    )

    d = EvaluateMetricsContract(
        hit_count_overall_at10=1,
        ndcg_sum_overall_at10=0.63,
        evaluated_users=1,
        eval_loss=0.0,
        sampled_hr_at10=1.0,
        sampled_ndcg_at10=0.63,
        partition_id=1234,
        hit_count_sparse_at10=1,
        ndcg_sum_sparse_at10=0.63,
        evaluated_users_sparse=1,
        hit_count_medium_at10=0,
        ndcg_sum_medium_at10=0.0,
        evaluated_users_medium=0,
        hit_count_dense_at10=0,
        ndcg_sum_dense_at10=0.0,
        evaluated_users_dense=0,
    ).to_dict()
    validate_evaluate_metrics(d)  # no free-form extras allowed
    assert d.get("partition_id") == 1234, "G-03-01: partition_id must appear in evaluate to_dict()"
    for key in (
        "eval_loss",
        "sampled_hr_at10",
        "sampled_ndcg_at10",
        "evaluated_users",
        "hit_count_overall_at10",
        "ndcg_sum_overall_at10",
        "partition_id",
        "hit_count_sparse_at10",
        "evaluated_users_medium",
        "evaluated_users_dense",
    ):
        assert key in d, f"missing {key}"
    # Negative guard: a payload with FitMetricsContract-style keys (or
    # any free-form key not defined on EvaluateMetricsContract) must
    # fail ``validate_evaluate_metrics`` — either missing required keys
    # or unknown free-form extras trigger a ValueError.
    with pytest.raises(ValueError, match="free-form extras|missing required"):
        validate_evaluate_metrics(
            {
                "train_loss": 0.3,
                "num_positives": 10,
                "num_training_examples": 50,
            }
        )
