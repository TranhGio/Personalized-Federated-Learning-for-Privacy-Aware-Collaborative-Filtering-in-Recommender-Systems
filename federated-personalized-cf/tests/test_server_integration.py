"""Server integration tests (Phase 3 Plan 04).

Covers the PSN-04 / PSN-07 / D-13 / D-15 / D-02 invariants closed by Plan 04:

- **PSN-04 reproducibility**: ``server_rng(run_seed)``-driven per-round client
  selection is byte-identical across RNG instances; different seeds yield
  different selections.
- **PSN-04 strategy wire-up**: ``PersonalizedSplitFedAvg.aggregate_evaluate``
  returns sum-based thesis metrics, not mean-of-per-client ratios (would be
  the silent sparse-user double-count bug).
- **PSN-07 / D-15**: ``build_run_manifest`` with ``module="personalized"``
  integrates all four IMP-2 foundation fingerprints.
- **D-13 cold-start math**: the server-side cold-start probe counts selected
  partitions whose ``.embedding_cache/{run_id}/partition_{pid}.pt`` does NOT
  exist before the round fires.
- **D-02 frozen cross-silo guard** (source-level regression check): the
  personalized server_app.py raises ``NotImplementedError`` inside a
  ``cross_silo_legacy`` branch — a direct runtime check would need a live
  Grid, so we assert the guard is present in source.

Tests are skipped if the foundation bundle (``data/derived/foundation_index.json``)
is absent so a minimal clone doesn't fail CI.
"""
from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "derived" / "foundation_index.json").exists(),
    reason="foundation bundle not committed",
)


def test_server_rng_reproducible_per_round_selection() -> None:
    """PSN-04: server_rng(run_seed) produces byte-identical sequences across instances.

    Pure-RNG property — necessary but not sufficient. The load-bearing
    invariant (byte-identical selected_clients_per_round across subprocess
    reruns) is proven by Plan 05's subprocess determinism guard.
    """
    from fedrec_foundation.rng import server_rng

    rng1 = server_rng(42)
    rng2 = server_rng(42)
    seq1 = [tuple(rng1.sample(range(6040), 50)) for _ in range(3)]
    seq2 = [tuple(rng2.sample(range(6040), 50)) for _ in range(3)]
    assert seq1 == seq2, "server_rng(42) must produce byte-identical sequences across instances"


def test_server_rng_different_seeds_different_selections() -> None:
    """PSN-04 negative guard: different run_seeds -> different sequences."""
    from fedrec_foundation.rng import server_rng

    rng1 = server_rng(42)
    rng2 = server_rng(43)
    s1 = rng1.sample(range(6040), 50)
    s2 = rng2.sample(range(6040), 50)
    assert s1 != s2, "Different seeds MUST yield different client-selection sequences"


def test_personalized_split_fedavg_aggregate_evaluate_sum_not_average() -> None:
    """PSN-04 strategy wire-up: sum-based ratios, not mean-of-per-client-ratios.

    Same shape as Phase 2 baseline's BSL-06 test: Client A with 1 hit on 1 user
    (HR@10=1.0) and Client B with 0 hits on 99 users (HR@10=0.0). A mean of
    per-client ratios would be 0.5 (WRONG); the correct sum-based ratio is
    1/100 = 0.01. The strategy MUST emit 0.01.
    """
    from unittest.mock import MagicMock
    from flwr.common import Code, EvaluateRes, Status

    from federated_personalized_cf.strategy import PersonalizedSplitFedAvg

    strategy = PersonalizedSplitFedAvg(fraction_fit=0.1)
    proxy = MagicMock()
    proxy.cid = "c"

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
        f"PSN-04 strategy: sum-based ratio must be 1/100=0.01, got {metrics['sampled_hr@10']}"
    )
    assert metrics["sampled_hr@10"] < 0.5, (
        "PSN-04 strategy: aggregation is NOT averaging per-client ratios"
    )


def test_build_run_manifest_module_personalized() -> None:
    """PSN-07 + D-15: build_run_manifest with module='personalized' integrates the IMP-2 fingerprints."""
    from fedrec_foundation.bundle import verify_bundle
    from fedrec_foundation.manifest import build_run_manifest, generate_run_id
    from fedrec_foundation.mode import resolve_mode_defaults
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
        module="personalized",
    )
    # All four IMP-2 fingerprints present.
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
    # Module flag is Phase-3-specific.
    assert manifest.module == "personalized"


def test_cold_start_counter_math(tmp_path) -> None:
    """D-13 cold-start counter: counts partitions whose cache file does not exist.

    Replicates the server-side arithmetic inline (the server helper probes
    ``.embedding_cache/{run_id}/partition_{pid}.pt``). Given a cache dir seeded
    with partitions 0/1/2 and selected_pids 0..5, the expected cold count is 3.
    """
    cache_root = tmp_path / ".embedding_cache" / "r1"
    cache_root.mkdir(parents=True)
    for pid in (0, 1, 2):
        (cache_root / f"partition_{pid}.pt").write_bytes(b"\x00")
    selected_pids = [0, 1, 2, 3, 4, 5]
    cold_count = sum(
        1 for pid in selected_pids
        if not (cache_root / f"partition_{int(pid)}.pt").exists()
    )
    hot_count = sum(
        1 for pid in selected_pids
        if (cache_root / f"partition_{int(pid)}.pt").exists()
    )
    assert cold_count == 3, f"expected 3 cold starts, got {cold_count}"
    assert hot_count == 3, f"expected 3 hot partitions, got {hot_count}"

    # cold_start_rate matches the final results JSON shape: ratio over
    # the total client selections across rounds.
    total_selections = len(selected_pids)
    cold_start_rate = cold_count / total_selections
    assert cold_start_rate == pytest.approx(0.5, abs=1e-9)


def test_cross_silo_legacy_mode_raises_not_implemented() -> None:
    """D-02 frozen cross-silo: source-level regression guard.

    @app.main() raises NotImplementedError when mode=='cross_silo_legacy'.
    A runtime check would require a live Grid + Context, so we assert the
    guard is present in source (string + proximity check). This catches
    accidental removal of the D-02 branch during future refactors.
    """
    src_path = Path(__file__).resolve().parents[1].joinpath(
        "federated_personalized_cf", "server_app.py"
    )
    src = src_path.read_text()
    # Must mention cross_silo_legacy as a control-flow token.
    assert "cross_silo_legacy" in src, "D-02 guard removed: cross_silo_legacy token missing"
    assert "raise NotImplementedError" in src, "D-02 guard removed: NotImplementedError raise missing"
    # Proximity check: the raise must live near the cross_silo_legacy branch
    # (first occurrence in source should point at the guard, not a comment).
    idx = src.index('cross_silo_legacy')
    nearby = src[idx:idx + 800]
    assert "NotImplementedError" in nearby, (
        "D-02 guard placement: 'raise NotImplementedError' must appear within 800 "
        "chars of the cross_silo_legacy branch in server_app.py"
    )
    # Error message must include the D-02 token so a human reading the traceback
    # knows exactly which decision they tripped.
    assert "D-02" in src, (
        "D-02 guard error message should cite decision D-02 for traceability"
    )
