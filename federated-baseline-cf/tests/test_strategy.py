"""Unit tests for BaselineFedAvg/BaselineFedProx sufficient-stat aggregation (Phase 2 Plan 01)."""
from __future__ import annotations

import pytest

from federated_baseline_cf.strategy import BaselineFedAvg, BaselineFedProx


def test_aggregate_evaluate_sums_sufficient_stats(fake_evaluate_res, fake_client_proxy) -> None:
    strategy = BaselineFedAvg(fraction_fit=0.1)
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
    strategy = BaselineFedAvg(fraction_fit=0.1)
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
    strategy = BaselineFedAvg(fraction_fit=0.1)
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


def test_baseline_fedprox_inherits_aggregate_evaluate(fake_evaluate_res, fake_client_proxy) -> None:
    strategy = BaselineFedProx(fraction_fit=0.1, proximal_mu=0.01)
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


def test_aggregate_fit_inherited_unchanged() -> None:
    """aggregate_fit must NOT be overridden — baseline = all params global."""
    from flwr.server.strategy import FedAvg as _FedAvg
    assert BaselineFedAvg.aggregate_fit is _FedAvg.aggregate_fit, (
        "BaselineFedAvg MUST inherit aggregate_fit unchanged from FedAvg "
        "(D-23: baseline keeps all params global)"
    )
