"""Server integration tests (Phase 2 Plan 04).

Covers the three BSL requirements closed by Plan 04:

- **BSL-04**: ``server_rng(run_seed)``-driven per-round client selection is
  reproducible across processes; different seeds yield different sequences.
- **BSL-06**: ``BaselineFedAvg.aggregate_evaluate`` returns sum-based ratios,
  not mean-of-per-client-ratios (the silent sparse-user double-count bug).
- **BSL-08**: ``build_run_manifest`` integrates all four foundation
  fingerprints; D-15 double-write (embedded + sibling) roundtrips cleanly.

Tests are skipped if the foundation bundle (``data/derived/foundation_index.json``)
is absent so a minimal clone doesn't fail CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "derived" / "foundation_index.json").exists(),
    reason="foundation bundle not committed",
)


def test_server_rng_reproducible_per_round_selection() -> None:
    """BSL-04: server_rng(run_seed) produces identical client sequences across processes."""
    from fedrec_foundation.rng import server_rng
    # Two separate rng instances with the same seed -> same sequence.
    rng1 = server_rng(42)
    rng2 = server_rng(42)
    ids = list(range(6040))
    seq1 = [tuple(rng1.sample(sorted(ids), 50)) for _ in range(3)]
    seq2 = [tuple(rng2.sample(sorted(ids), 50)) for _ in range(3)]
    assert seq1 == seq2, "server_rng(42) must produce byte-identical sequences across instances"


def test_server_rng_different_seeds_different_selections() -> None:
    """BSL-04 negative guard: different run_seeds -> different sequences."""
    from fedrec_foundation.rng import server_rng
    rng1 = server_rng(42)
    rng2 = server_rng(43)
    ids = list(range(6040))
    s1 = rng1.sample(sorted(ids), 50)
    s2 = rng2.sample(sorted(ids), 50)
    assert s1 != s2, "Different seeds MUST yield different client-selection sequences"


def test_aggregate_evaluate_uses_sum_not_average() -> None:
    """BSL-06: BaselineFedAvg.aggregate_evaluate returns sum-based ratios, not mean-of-ratios."""
    from unittest.mock import MagicMock
    from flwr.common import EvaluateRes, Status, Code
    from federated_baseline_cf.strategy import BaselineFedAvg

    strategy = BaselineFedAvg(fraction_fit=0.1)
    proxy = MagicMock()
    proxy.cid = "c"

    # Client A: 1 hit on 1 user (HR=1.0), NDCG=1.0. Client B: 0 hits on 99 users (HR=0).
    # Per-client AVERAGE of ratios = (1.0 + 0.0) / 2 = 0.5  <-- WRONG
    # SUM-BASED ratio           = 1 / 100 = 0.01                          <-- CORRECT
    results = [
        (proxy, EvaluateRes(Status(Code.OK, "ok"), 0.5, 1, {
            "eval_loss": 0.5,
            "hit_count_overall_at10": 1, "ndcg_sum_overall_at10": 1.0, "evaluated_users": 1,
            "hit_count_sparse_at10": 0, "ndcg_sum_sparse_at10": 0.0, "evaluated_users_sparse": 0,
            "hit_count_medium_at10": 0, "ndcg_sum_medium_at10": 0.0, "evaluated_users_medium": 0,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.0, "evaluated_users_dense": 0,
        })),
        (proxy, EvaluateRes(Status(Code.OK, "ok"), 0.5, 99, {
            "eval_loss": 0.5,
            "hit_count_overall_at10": 0, "ndcg_sum_overall_at10": 0.0, "evaluated_users": 99,
            "hit_count_sparse_at10": 0, "ndcg_sum_sparse_at10": 0.0, "evaluated_users_sparse": 0,
            "hit_count_medium_at10": 0, "ndcg_sum_medium_at10": 0.0, "evaluated_users_medium": 0,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.0, "evaluated_users_dense": 0,
        })),
    ]
    _loss, metrics = strategy.aggregate_evaluate(1, results, [])
    assert metrics["sampled_hr@10"] == pytest.approx(1.0 / 100.0, abs=1e-9), (
        f"BSL-06: sum-based ratio must be 1/100=0.01, got {metrics['sampled_hr@10']}"
    )
    # Sanity: mean-of-ratios would be 0.5; we MUST not be 0.5.
    assert metrics["sampled_hr@10"] < 0.5, "BSL-06: aggregation is NOT averaging per-client ratios"


def test_build_run_manifest_integrates_foundation_index() -> None:
    """BSL-08: build_run_manifest integrates all four foundation fingerprints."""
    from fedrec_foundation.bundle import verify_bundle
    from fedrec_foundation.mode import resolve_mode_defaults
    from fedrec_foundation.manifest import build_run_manifest, generate_run_id
    from fedrec_foundation.paths import data_derived
    from fedrec_foundation.split import load_split_manifest

    derived = data_derived()
    idx = verify_bundle(derived)
    split = load_split_manifest(derived / "split_manifest.json")
    profile = resolve_mode_defaults("benchmark_cross_device")

    manifest = build_run_manifest(
        run_id=generate_run_id(),
        mode_profile=profile,
        run_seed=42,
        mapping_sha256=idx.mapping_sha256,
        split_hash=idx.split_hash,
        exclusion_sha256=idx.exclusion_sha256,
        foundation_contract_sha256=idx.foundation_contract_sha256,
        raw_data_hash=split.raw_data_hash,
        builder_version=split.builder_version,
        overrides={"lr": 0.005},
        module="baseline",
    )
    # All 4 IMP-2 fingerprints present.
    assert manifest.mapping_sha256 == idx.mapping_sha256
    assert manifest.split_hash == idx.split_hash
    assert manifest.exclusion_sha256 == idx.exclusion_sha256
    assert manifest.foundation_contract_sha256 == idx.foundation_contract_sha256
    assert manifest.raw_data_hash == split.raw_data_hash
    # Mode profile propagated.
    assert manifest.mode == "benchmark_cross_device"
    assert manifest.num_supernodes == 6040
    assert manifest.weight_policy == "num_positives"
    assert manifest.primary_evaluator == "sampled_loo_99"
    # Overrides captured.
    assert manifest.overrides == {"lr": 0.005}
    assert manifest.module == "baseline"


def test_embed_and_sibling_double_write_roundtrip(tmp_path) -> None:
    """BSL-08 + D-15: double-write (embedded + sibling) roundtrips to JSON cleanly."""
    from fedrec_foundation.bundle import verify_bundle
    from fedrec_foundation.mode import resolve_mode_defaults
    from fedrec_foundation.manifest import (
        build_run_manifest, generate_run_id, embed_manifest_in_result, write_manifest_sibling,
    )
    from fedrec_foundation.paths import data_derived
    from fedrec_foundation.split import load_split_manifest

    derived = data_derived()
    idx = verify_bundle(derived)
    split = load_split_manifest(derived / "split_manifest.json")
    profile = resolve_mode_defaults("benchmark_cross_device")
    manifest = build_run_manifest(
        run_id=generate_run_id(), mode_profile=profile, run_seed=42,
        mapping_sha256=idx.mapping_sha256, split_hash=idx.split_hash,
        exclusion_sha256=idx.exclusion_sha256,
        foundation_contract_sha256=idx.foundation_contract_sha256,
        raw_data_hash=split.raw_data_hash, builder_version=split.builder_version,
        overrides={}, module="baseline",
    )
    result = {"training_rounds": 10, "final_metrics": {"sampled_ndcg@10": 0.25}}
    embed_manifest_in_result(manifest, result)  # mutates
    result_path = tmp_path / f"{manifest.run_id}_results.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4, default=str)
    sibling = write_manifest_sibling(manifest, result_path)
    # Both artifacts exist + contain the foundation_contract_sha256.
    assert result_path.exists()
    assert sibling.exists()
    with open(result_path) as f:
        roundtrip = json.load(f)
    assert roundtrip["_manifest"]["foundation_contract_sha256"] == idx.foundation_contract_sha256
    with open(sibling) as f:
        sibling_json = json.load(f)
    assert sibling_json["foundation_contract_sha256"] == idx.foundation_contract_sha256
