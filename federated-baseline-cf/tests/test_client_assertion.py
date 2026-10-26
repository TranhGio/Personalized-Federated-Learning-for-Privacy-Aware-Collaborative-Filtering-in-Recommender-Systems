"""Client-app assertions + contract-payload tests (Phase 2 Plan 03).

Covers:
  - BSL-02: ``assert_benchmark_one_user_per_client`` raises
    ``AssertionError`` when ``num_users_in_client != 1`` under benchmark
    profile; a visible ``num_supernodes`` override bypasses the lock
    with a log line.
  - BSL-07: ``get_primary_evaluator(mode)`` resolves to
    ``"sampled_loo_99"`` for every recognized mode.
  - D-21 + D-22: both ``FitMetricsContract`` and
    ``EvaluateMetricsContract`` carry per-group sufficient-stat fields
    in their ``to_dict()`` output; free-form extras are rejected by
    ``validate_evaluate_metrics`` (iteration 1 WARNING 1 fix +
    iteration 2 unified-naming check).
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
    """BSL-02: benchmark_cross_device with >1 user per client raises.

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
    """BSL-07: all three recognized modes route to sampled_loo_99."""
    from fedrec_foundation.evaluator import get_primary_evaluator

    for mode in ("benchmark_cross_device", "paper_compat_pfedrec", "cross_silo_legacy"):
        assert get_primary_evaluator(mode) == "sampled_loo_99"


def test_fit_metrics_contract_payload_shape() -> None:
    """D-21 + D-22: FitMetricsContract per-group fields populated in to_dict()."""
    from fedrec_foundation.fit_metrics import FitMetricsContract, validate_fit_metrics

    d = FitMetricsContract(
        train_loss=0.3,
        num_positives=10,
        num_training_examples=50,
        round_num=2,
        hit_count_overall_at10=1,
        ndcg_sum_overall_at10=0.63,
        evaluated_users=1,
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
    validate_fit_metrics(d)
    for key in (
        "train_loss",
        "num_positives",
        "num_training_examples",
        "hit_count_overall_at10",
        "evaluated_users_sparse",
        "evaluated_users_medium",
        "evaluated_users_dense",
    ):
        assert key in d, f"missing {key}"


def test_evaluate_metrics_contract_payload_shape() -> None:
    """D-21 + D-22: EvaluateMetricsContract per-group fields populated + no extras.

    Iteration 1 WARNING 1 fix: client_app.py's @app.evaluate() handler
    builds its reply via ``EvaluateMetricsContract.to_dict()`` (NOT
    ``FitMetricsContract``). Verifying the shape here catches contract
    drift before wire transmission.
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
    for key in (
        "eval_loss",
        "sampled_hr_at10",
        "sampled_ndcg_at10",
        "evaluated_users",
        "hit_count_overall_at10",
        "ndcg_sum_overall_at10",
        "hit_count_sparse_at10",
        "evaluated_users_medium",
        "evaluated_users_dense",
    ):
        assert key in d, f"missing {key}"
    # Negative guard: a payload with FitMetricsContract-style keys must
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
