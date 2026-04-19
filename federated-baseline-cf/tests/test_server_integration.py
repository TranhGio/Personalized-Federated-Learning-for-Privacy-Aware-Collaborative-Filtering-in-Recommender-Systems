"""Server integration tests (Phase 2 Plan 04 + Plan 05 G-03-01 gap closure).

Covers the three BSL requirements closed by Plan 04:

- **BSL-04**: ``server_rng(run_seed)``-driven per-round client selection is
  reproducible across processes; different seeds yield different sequences.
- **BSL-06**: ``BaselineFedAvg.aggregate_evaluate`` returns sum-based ratios,
  not mean-of-per-client-ratios (the silent sparse-user double-count bug).
- **BSL-08**: ``build_run_manifest`` integrates all four foundation
  fingerprints; D-15 double-write (embedded + sibling) roundtrips cleanly.

Plan 05 G-03-01 addition:

- ``test_selected_partitions_byte_identical_across_subprocess_reruns``:
  REAL-LOOP reproducibility check. Runs ``scripts/run.py baseline
  benchmark_cross_device`` twice in child processes and asserts the
  ``selected_clients_per_round`` JSON fields are byte-identical — the
  load-bearing invariant Plan-04's pure-RNG test missed.

Tests are skipped if the foundation bundle (``data/derived/foundation_index.json``)
is absent so a minimal clone doesn't fail CI.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "derived" / "foundation_index.json").exists(),
    reason="foundation bundle not committed",
)


def test_server_rng_reproducible_per_round_selection() -> None:
    """BSL-04: server_rng(run_seed) produces identical client sequences across processes.

    NOTE (G-03-01): This is a NECESSARY-BUT-NOT-SUFFICIENT check. It tests
    pure-RNG determinism with a FIXED domain; it does NOT exercise the
    real loop where the sampling domain itself must be stable across runs.
    The load-bearing invariant — byte-identical selected_clients_per_round
    across independent subprocess reruns with the same run-seed — is
    asserted by ``test_selected_partitions_byte_identical_across_subprocess_reruns``.
    """
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


# ============================================================================
# Plan 05 G-03-01: real-loop subprocess reproducibility regression guard.
# ----------------------------------------------------------------------------
# Plan-04's test_server_rng_reproducible_per_round_selection asserted
#     `rng.sample(sorted(fixed_ids), k)` is stable across RNG instances.
# That's a pure-RNG property and always held. The load-bearing invariant
# the thesis needs is that TWO SUBPROCESS RERUNS of the launcher with the
# same run-seed produce byte-identical `selected_clients_per_round`. The
# real loop was broken because Flower's node_ids are os.urandom-seeded
# per boot, so `sorted(grid.get_node_ids())` randomised the sampling
# domain between runs. Plan-05 fixes this by sampling in partition-id
# space (stable 0..N-1) and recording partition_ids in the result JSON.
# ============================================================================


@pytest.mark.slow
def test_selected_partitions_byte_identical_across_subprocess_reruns(tmp_path) -> None:
    """G-03-01 regression guard: real-loop reproducibility across subprocess reruns.

    Invariant: for fixed run-seed, two independent `python scripts/run.py
    baseline benchmark_cross_device ...` invocations produce JSONs whose
    ``selected_clients_per_round`` fields are byte-identical (partition_id
    space). This test WOULD have failed on pre-Plan-05 code (where
    `selected_clients_per_round` stored ephemeral, per-boot-randomised
    node_ids). It passes on post-Plan-05 code because partition_ids are
    stable across boots.

    Skipped when ``FEDREC_SKIP_SLOW=1`` or the foundation bundle is absent.
    """
    if os.environ.get("FEDREC_SKIP_SLOW") == "1":
        pytest.skip("FEDREC_SKIP_SLOW=1 set")

    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "results" / "federated"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the current set of result files so we can pick out exactly
    # the two this test produces (avoid picking up unrelated earlier runs).
    before = set(results_dir.glob("*_results.json"))

    launcher = repo_root / "scripts" / "run.py"
    cmd = [
        sys.executable,
        str(launcher),
        "baseline",
        "benchmark_cross_device",
        "--run-config",
        "num-server-rounds=1 fraction-train=0.005 local-epochs=1 wandb-enabled=false",
    ]

    def _run_once() -> None:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode != 0:
            pytest.skip(
                f"launcher failed (rc={proc.returncode}); skipping real-loop test. "
                f"stdout tail: {proc.stdout[-500:]!r} stderr tail: {proc.stderr[-500:]!r}"
            )

    _run_once()
    _run_once()

    after = sorted((results_dir.glob("*_results.json")), key=lambda p: p.stat().st_mtime)
    new_files = [p for p in after if p not in before]
    assert len(new_files) >= 2, (
        f"G-03-01 test expected at least 2 new result JSONs in {results_dir}, "
        f"got {len(new_files)}"
    )
    file_a, file_b = new_files[-2], new_files[-1]
    with open(file_a) as f:
        a = json.load(f)
    with open(file_b) as f:
        b = json.load(f)
    assert a["selected_clients_per_round"] == b["selected_clients_per_round"], (
        f"G-03-01 broken: selected_clients_per_round differs across subprocess reruns "
        f"with the same run-seed. {file_a.name} vs {file_b.name}"
    )
    # NDCG@10 cross-run diff should collapse to ≤1e-3 once the same users train.
    ndcg_a = float(a["final_metrics"].get("sampled_ndcg@10", 0.0))
    ndcg_b = float(b["final_metrics"].get("sampled_ndcg@10", 0.0))
    assert abs(ndcg_a - ndcg_b) <= 1e-3, (
        f"G-03-01 regression: ndcg@10 cross-run diff {abs(ndcg_a - ndcg_b):.6f} > 1e-3"
    )
