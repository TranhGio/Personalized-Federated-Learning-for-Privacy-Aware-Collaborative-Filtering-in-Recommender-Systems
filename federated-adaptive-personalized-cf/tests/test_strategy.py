"""Unit tests for AdaptiveSplitFedAvg/AdaptiveSplitFedProx (Phase 4 Plan 01).

Covers:
1-4. Sufficient-stat aggregate_evaluate sum + per-group ratios + zero-division + FedProx
     inheritance — clone of Phase 3 test_strategy.py with strategy class names substituted.
5.   aggregate_fit OVERRIDDEN: super().aggregate_fit runs first (D-23 preserved via super()
     call, not pure-inheritance identity), then _aggregate_prototypes updates the server EMA.
6.   best_prototype snapshot on live EMA — copy, not reference.
7.   best_prototype snapshot degenerate case (no prior aggregation) — zero vector + WARNING
     per D-08.
"""
from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pytest

from flwr.common import Code, FitRes, Parameters, Status
from flwr.server.strategy import FedAvg as BaseFedAvg

from federated_adaptive_personalized_cf.strategy import (
    AdaptiveSplitFedAvg,
    AdaptiveSplitFedProx,
    GLOBAL_PARAM_KEYS,
    LOCAL_PARAM_KEYS_BASE,
    USER_PROTOTYPE_KEY,
)


def test_aggregate_evaluate_sums_sufficient_stats(fake_evaluate_res, fake_client_proxy) -> None:
    strategy = AdaptiveSplitFedAvg(fraction_fit=0.1)
    results = [
        (fake_client_proxy, fake_evaluate_res(20, {
            "eval_loss": 0.5,
            "hit_count_overall_at10": 10, "ndcg_sum_overall_at10": 5.0, "evaluated_users": 20,
            "hit_count_sparse_at10": 0, "ndcg_sum_sparse_at10": 0.0, "evaluated_users_sparse": 0,
            "hit_count_medium_at10": 0, "ndcg_sum_medium_at10": 0.0, "evaluated_users_medium": 0,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.0, "evaluated_users_dense": 0,
        })),
        (fake_client_proxy, fake_evaluate_res(15, {
            "eval_loss": 0.6,
            "hit_count_overall_at10": 5, "ndcg_sum_overall_at10": 2.5, "evaluated_users": 15,
            "hit_count_sparse_at10": 0, "ndcg_sum_sparse_at10": 0.0, "evaluated_users_sparse": 0,
            "hit_count_medium_at10": 0, "ndcg_sum_medium_at10": 0.0, "evaluated_users_medium": 0,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.0, "evaluated_users_dense": 0,
        })),
        (fake_client_proxy, fake_evaluate_res(25, {
            "eval_loss": 0.4,
            "hit_count_overall_at10": 7, "ndcg_sum_overall_at10": 3.5, "evaluated_users": 25,
            "hit_count_sparse_at10": 0, "ndcg_sum_sparse_at10": 0.0, "evaluated_users_sparse": 0,
            "hit_count_medium_at10": 0, "ndcg_sum_medium_at10": 0.0, "evaluated_users_medium": 0,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.0, "evaluated_users_dense": 0,
        })),
    ]
    _loss, metrics = strategy.aggregate_evaluate(1, results, [])
    assert metrics["sampled_hr@10"] == pytest.approx(22.0 / 60.0)
    assert metrics["sampled_ndcg@10"] == pytest.approx(11.0 / 60.0)
    assert metrics["evaluated_users"] == 60


def test_aggregate_evaluate_per_group_ratios(fake_evaluate_res, fake_client_proxy) -> None:
    strategy = AdaptiveSplitFedAvg(fraction_fit=0.1)
    results = [
        (fake_client_proxy, fake_evaluate_res(23, {
            "eval_loss": 0.5,
            "hit_count_overall_at10": 12, "ndcg_sum_overall_at10": 6.0, "evaluated_users": 23,
            "hit_count_sparse_at10": 3, "ndcg_sum_sparse_at10": 1.0, "evaluated_users_sparse": 10,
            "hit_count_medium_at10": 4, "ndcg_sum_medium_at10": 2.0, "evaluated_users_medium": 8,
            "hit_count_dense_at10": 5, "ndcg_sum_dense_at10": 3.0, "evaluated_users_dense": 5,
        })),
        (fake_client_proxy, fake_evaluate_res(9, {
            "eval_loss": 0.5,
            "hit_count_overall_at10": 3, "ndcg_sum_overall_at10": 1.5, "evaluated_users": 9,
            "hit_count_sparse_at10": 1, "ndcg_sum_sparse_at10": 0.5, "evaluated_users_sparse": 5,
            "hit_count_medium_at10": 2, "ndcg_sum_medium_at10": 1.0, "evaluated_users_medium": 4,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.0, "evaluated_users_dense": 0,
        })),
    ]
    _loss, metrics = strategy.aggregate_evaluate(1, results, [])
    assert metrics["sampled_hr@10/sparse"] == pytest.approx(4.0 / 15.0)
    assert metrics["sampled_hr@10/medium"] == pytest.approx(6.0 / 12.0)
    assert metrics["sampled_hr@10/dense"] == pytest.approx(5.0 / 5.0)
    assert metrics["evaluated_users_sparse"] == 15
    assert metrics["evaluated_users_medium"] == 12
    assert metrics["evaluated_users_dense"] == 5


def test_aggregate_evaluate_zero_division_safe(fake_evaluate_res, fake_client_proxy) -> None:
    strategy = AdaptiveSplitFedAvg(fraction_fit=0.1)
    results = [
        (fake_client_proxy, fake_evaluate_res(5, {
            "eval_loss": 0.5,
            "hit_count_overall_at10": 2, "ndcg_sum_overall_at10": 1.0, "evaluated_users": 5,
            "hit_count_sparse_at10": 0, "ndcg_sum_sparse_at10": 0.0, "evaluated_users_sparse": 0,
            "hit_count_medium_at10": 2, "ndcg_sum_medium_at10": 1.0, "evaluated_users_medium": 5,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.0, "evaluated_users_dense": 0,
        })),
    ]
    _loss, metrics = strategy.aggregate_evaluate(1, results, [])
    assert metrics["sampled_hr@10/sparse"] == 0.0
    assert metrics["sampled_ndcg@10/sparse"] == 0.0
    assert metrics["sampled_hr@10/dense"] == 0.0


def test_adaptive_split_fedprox_inherits_aggregate_evaluate(
    fake_evaluate_res, fake_client_proxy
) -> None:
    strategy = AdaptiveSplitFedProx(fraction_fit=0.1, proximal_mu=0.01)
    results = [
        (fake_client_proxy, fake_evaluate_res(10, {
            "eval_loss": 0.5,
            "hit_count_overall_at10": 3, "ndcg_sum_overall_at10": 1.5, "evaluated_users": 10,
            "hit_count_sparse_at10": 0, "ndcg_sum_sparse_at10": 0.0, "evaluated_users_sparse": 0,
            "hit_count_medium_at10": 0, "ndcg_sum_medium_at10": 0.0, "evaluated_users_medium": 0,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.0, "evaluated_users_dense": 0,
        })),
    ]
    _loss, metrics = strategy.aggregate_evaluate(1, results, [])
    assert metrics["sampled_hr@10"] == pytest.approx(3.0 / 10.0)


def test_aggregate_fit_calls_super_then_prototypes(fake_client_proxy) -> None:
    """D-23 preserved in Phase 4 via super() call (not pure inheritance like Phase 3).

    aggregate_fit is OVERRIDDEN so prototype aggregation CONTINUES to run (existing
    adaptive behavior). The parent's weighted-average of GLOBAL params MUST still
    execute via super().aggregate_fit — verified by patching BaseFedAvg.aggregate_fit
    and asserting it was called.
    """
    strategy = AdaptiveSplitFedAvg(fraction_fit=1.0, prototype_momentum=0.9)
    fake_params = Parameters(tensors=[], tensor_type="numpy.ndarray")
    fit_res = FitRes(
        status=Status(Code.OK, "ok"),
        parameters=fake_params,
        num_examples=10,
        metrics={USER_PROTOTYPE_KEY: [1.0, 2.0, 3.0]},
    )
    with patch.object(
        BaseFedAvg, "aggregate_fit", return_value=(fake_params, {})
    ) as mock_super:
        strategy.aggregate_fit(
            server_round=1,
            results=[(fake_client_proxy, fit_res)],
            failures=[],
        )
        assert mock_super.called, "D-23 violated: super().aggregate_fit must run"
    # After aggregate_fit, prototype should be updated from the single client.
    assert strategy._global_prototype is not None
    assert np.allclose(strategy._global_prototype, np.array([1.0, 2.0, 3.0]))


def test_best_prototype_snapshot_at_best_round() -> None:
    """D-05 snapshot on live EMA — copy, not reference."""
    strategy = AdaptiveSplitFedAvg(fraction_fit=1.0)
    strategy._global_prototype = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    strategy.snapshot_best_prototype(round_num=5, embedding_dim=3)
    assert strategy.best_prototype is not None
    assert np.allclose(strategy.best_prototype, [1.0, 2.0, 3.0])
    # Copy, not reference — mutating _global_prototype must not touch best_prototype
    strategy._global_prototype[0] = 999.0
    assert np.allclose(strategy.best_prototype, [1.0, 2.0, 3.0])


def test_best_prototype_snapshot_degenerate_zero_vector(caplog) -> None:
    """D-08 degenerate case: best round before any prototype aggregated.

    Snapshot np.zeros(embedding_dim) + emit WARNING with substrings
    'Prototype snapshot at best round' AND 'zero vector'.
    """
    strategy = AdaptiveSplitFedAvg(fraction_fit=1.0)
    assert strategy._global_prototype is None
    with caplog.at_level(logging.WARNING):
        strategy.snapshot_best_prototype(round_num=1, embedding_dim=128)
    assert strategy.best_prototype is not None
    assert np.allclose(strategy.best_prototype, np.zeros(128))
    assert any(
        "Prototype snapshot at best round" in rec.getMessage()
        and "zero vector" in rec.getMessage()
        for rec in caplog.records
    ), f"Expected D-08 warning, got {caplog.records}"


def test_frozensets_match_contract() -> None:
    """GLOBAL_PARAM_KEYS and LOCAL_PARAM_KEYS_BASE are the declared split boundary."""
    assert GLOBAL_PARAM_KEYS == frozenset(
        {"item_embeddings.weight", "item_bias.weight", "global_bias"}
    )
    assert LOCAL_PARAM_KEYS_BASE == frozenset(
        {"user_embeddings.weight", "user_bias.weight"}
    )
