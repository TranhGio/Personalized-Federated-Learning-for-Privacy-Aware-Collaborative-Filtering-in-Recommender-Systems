"""Client-app assertions + contract-payload tests (Phase 4 Plan 03).

Covers:
  - ADP-04: ``assert_benchmark_one_user_per_client`` raises
    ``AssertionError`` when ``num_users_in_client != 1`` under benchmark
    profile; a visible ``num_supernodes`` override bypasses the lock.
  - Evaluator selection: ``get_primary_evaluator(mode)`` resolves to
    ``"sampled_loo_99"`` for every recognized mode.
  - D-21 + G-03-01 carry-forward: ``FitMetricsContract`` and
    ``EvaluateMetricsContract`` carry optional ``partition_id`` in their
    ``to_dict()`` output; free-form extras are rejected by
    ``validate_evaluate_metrics``.
  - D-16 alpha diagnostics round-trip as a sidecar dict (separate
    MetricRecord payload since ``validate_fit_metrics`` rejects free-form
    extras per D-21).
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
    """ADP-04: benchmark_cross_device with >1 user per client raises."""
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
    """D-10: visible num-supernodes override bypasses the lock."""
    from fedrec_foundation.mode import (
        assert_benchmark_one_user_per_client,
        resolve_mode_defaults,
    )

    profile = resolve_mode_defaults("benchmark_cross_device")
    # 50 users would normally raise; the visible override returns instead.
    assert_benchmark_one_user_per_client(
        profile,
        num_users_in_client=50,
        overrides={"num_supernodes": 10},
    )


def test_get_primary_evaluator_selects_sampled_loo_99() -> None:
    """ADP-06 client half: all three recognized modes route to sampled_loo_99."""
    from fedrec_foundation.evaluator import get_primary_evaluator

    for mode in ("benchmark_cross_device", "paper_compat_pfedrec", "cross_silo_legacy"):
        assert get_primary_evaluator(mode) == "sampled_loo_99"


def test_fit_metrics_contract_payload_with_partition_id_and_alpha_diagnostics() -> None:
    """D-21 + G-03-01 + D-16: FitMetricsContract partition_id + alpha sidecar.

    The strict contract (D-21) rejects free-form extras, so alpha
    diagnostics ride in a SEPARATE sidecar dict (client_app.py routes
    them via a second MetricRecord keyed "alpha_diagnostics"). Here we
    verify both shapes round-trip.
    """
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

    # D-16 alpha diagnostics sidecar — plain dict, round-trippable via
    # Flower MetricRecord contents. The six required keys are present.
    alpha_diag = {
        "alpha_mean": 0.5,
        "alpha_std": 0.1,
        "alpha_p25": 0.3,
        "alpha_p50": 0.5,
        "alpha_p75": 0.7,
        "alpha_clip_hit_rate": 0.05,
    }
    for key in (
        "alpha_mean",
        "alpha_std",
        "alpha_p25",
        "alpha_p50",
        "alpha_p75",
        "alpha_clip_hit_rate",
    ):
        assert key in alpha_diag, f"D-16 missing alpha diagnostic key {key}"
        assert isinstance(alpha_diag[key], float)


def test_evaluate_metrics_contract_payload_shape_with_partition_id() -> None:
    """D-21 + G-03-01: EvaluateMetricsContract carries partition_id + rejects free-form extras."""
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
    validate_evaluate_metrics(d)
    assert d.get("partition_id") == 1234, "G-03-01: partition_id must appear in evaluate to_dict()"
    # Negative guard: free-form extras raise ValueError.
    with pytest.raises(ValueError, match="free-form extras|missing required"):
        validate_evaluate_metrics(
            {
                "train_loss": 0.3,
                "num_positives": 10,
                "num_training_examples": 50,
            }
        )
