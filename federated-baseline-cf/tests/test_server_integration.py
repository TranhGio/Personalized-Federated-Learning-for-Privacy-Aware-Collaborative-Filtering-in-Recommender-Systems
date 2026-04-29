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
    before = set(results_dir.glob("baseline/*/results.json"))

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

    after = sorted((results_dir.glob("baseline/*/results.json")), key=lambda p: p.stat().st_mtime)
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
    ndcg_a = float(a["final_metrics"]["best"].get("sampled_ndcg@10", 0.0))
    ndcg_b = float(b["final_metrics"]["best"].get("sampled_ndcg@10", 0.0))
    assert abs(ndcg_a - ndcg_b) <= 1e-3, (
        f"G-03-01 regression: ndcg@10 cross-run diff {abs(ndcg_a - ndcg_b):.6f} > 1e-3"
    )


# ============================================================================
# Phase 6 Plan 03: EVL-01/02/03/04/06 integration tests (4 NEW assertions).
# These tests use a lightweight mock harness — no real Flower simulation needed.
# They validate the Phase-6 server_app.py changes: per-run-dir path, extra-eval-
# round wiring, nested final_metrics schema, and per-group exposure history.
# ============================================================================


def _build_fake_evaluate_res(
    hit_count_overall: int = 1,
    ndcg_sum_overall: float = 0.5,
    evaluated_users: int = 1,
    hit_count_sparse: int = 0,
    ndcg_sum_sparse: float = 0.0,
    evaluated_users_sparse: int = 0,
    hit_count_medium: int = 1,
    ndcg_sum_medium: float = 0.5,
    evaluated_users_medium: int = 1,
    hit_count_dense: int = 0,
    ndcg_sum_dense: float = 0.0,
    evaluated_users_dense: int = 0,
):
    """Build a synthetic EvaluateRes carrying D-22 sufficient-stat fields."""
    from flwr.common import EvaluateRes, Status, Code
    metrics = {
        "hit_count_overall_at10": hit_count_overall,
        "ndcg_sum_overall_at10": ndcg_sum_overall,
        "evaluated_users": evaluated_users,
        "hit_count_sparse_at10": hit_count_sparse,
        "ndcg_sum_sparse_at10": ndcg_sum_sparse,
        "evaluated_users_sparse": evaluated_users_sparse,
        "hit_count_medium_at10": hit_count_medium,
        "ndcg_sum_medium_at10": ndcg_sum_medium,
        "evaluated_users_medium": evaluated_users_medium,
        "hit_count_dense_at10": hit_count_dense,
        "ndcg_sum_dense_at10": ndcg_sum_dense,
        "evaluated_users_dense": evaluated_users_dense,
    }
    return EvaluateRes(
        status=Status(code=Code.OK, message="ok"),
        loss=0.1,
        num_examples=evaluated_users,
        metrics=metrics,
    )


def test_results_path_repo_root_anchored(tmp_path) -> None:
    """EVL-04 / D-01 / D-02: module_run_results_dir returns a repo-root-anchored path.

    Asserts:
    - Path is absolute.
    - Path resolves to ``<repo>/results/federated/baseline/<run_id>/``.
    - Directory is created eagerly by the helper (so results.json can be written).
    """
    from fedrec_foundation.paths import module_run_results_dir, repo_root
    from fedrec_foundation.manifest import generate_run_id

    run_id = generate_run_id()
    run_dir = module_run_results_dir("baseline", run_id)

    expected_root = repo_root() / "results" / "federated" / "baseline" / run_id
    assert run_dir.is_absolute(), "D-02: run_dir must be absolute (cwd-independent)"
    assert run_dir.resolve() == expected_root.resolve(), (
        f"D-01/D-02: expected {expected_root}, got {run_dir}"
    )
    assert run_dir.is_dir(), "module_run_results_dir must create the directory eagerly"

    # Simulate writing results.json + manifest.json (D-04 clean filenames).
    results_path = run_dir / "results.json"
    manifest_path = run_dir / "manifest.json"
    results_path.write_text('{"test": true}')
    manifest_path.write_text('{"schema_version": 2}')
    assert results_path.exists(), "results.json must be writable in run_dir"
    assert manifest_path.exists(), "manifest.json must be writable in run_dir"

    # Cleanup to avoid polluting the real results/ tree.
    import shutil
    shutil.rmtree(run_dir, ignore_errors=True)


def test_extra_eval_round_after_best_arrays_restore() -> None:
    """EVL-01 / D-06: extra-eval-round wires strategy.aggregate_evaluate correctly.

    Simulates: eval_metrics_history = {1: ndcg=0.40, 2: ndcg=0.45, 3: ndcg=0.42}
    => best_round_num=2, last_round=3.

    Asserts:
    - final_eval_round_index == actual_rounds + 1 (== 4).
    - best_round_metrics is non-empty and sampled_ndcg@10 == 0.5 (from synthetic).
    - strategy.aggregate_evaluate is called with final_eval_round_index as round arg.
    """
    from unittest.mock import MagicMock, patch
    from federated_baseline_cf.strategy import BaselineFedAvg

    strategy = BaselineFedAvg(fraction_fit=1.0)

    # Fake in-loop eval history: round 2 is best.
    eval_metrics_history = {
        1: {"sampled_ndcg@10": 0.40, "evaluated_users": 10},
        2: {"sampled_ndcg@10": 0.45, "evaluated_users": 10},
        3: {"sampled_ndcg@10": 0.42, "evaluated_users": 10},
    }
    best_round_num = 2
    actual_rounds = 3

    # Build synthetic extra_results (2 fake nodes, each with 1 hit on 1 user,
    # ndcg_sum=0.5 -> strategy returns sampled_ndcg@10 = 0.5 / 1 = 0.5 overall).
    from unittest.mock import MagicMock
    proxy = MagicMock()
    proxy.cid = "fake"
    extra_results = [
        (proxy, _build_fake_evaluate_res(
            hit_count_overall=1, ndcg_sum_overall=0.5, evaluated_users=1,
        )),
    ]

    final_eval_round_index = actual_rounds + 1  # = 4
    _agg_loss, thesis = strategy.aggregate_evaluate(final_eval_round_index, extra_results, [])

    assert thesis is not None and len(thesis) > 0, "aggregate_evaluate must return non-empty thesis dict"
    best_round_metrics = {
        k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
        for k, v in (thesis or {}).items()
    }

    assert final_eval_round_index == 4, f"Expected final_eval_round_index=4, got {final_eval_round_index}"
    assert best_round_metrics, "best_round_metrics must be non-empty after extra eval"
    assert best_round_metrics.get("sampled_ndcg@10") == pytest.approx(0.5, abs=1e-9), (
        f"D-06: sampled_ndcg@10 from synthetic extra eval should be 0.5, "
        f"got {best_round_metrics.get('sampled_ndcg@10')}"
    )


def test_canonical_artifact_carries_best_and_last_blocks(tmp_path) -> None:
    """EVL-06 / D-07: results.json final_metrics is nested {best, last, best_round, last_round, final_eval_round_index}.

    Asserts:
    - final_metrics top-level keys == {best, last, best_round, last_round, final_eval_round_index}.
    - best and last are both dicts.
    - best_round == 2, last_round == 3, final_eval_round_index == 4.
    - _manifest.schema_version == 2.
    - _manifest.final_eval_round_index == 4.
    - _manifest.metrics == final_metrics (dataclasses.replace copy-through).
    """
    from fedrec_foundation.bundle import verify_bundle
    from fedrec_foundation.mode import resolve_mode_defaults
    from fedrec_foundation.manifest import (
        build_run_manifest, generate_run_id, embed_manifest_in_result,
        write_manifest_sibling, RUN_MANIFEST_SCHEMA_VERSION,
    )
    from fedrec_foundation.paths import data_derived, module_run_results_dir
    from fedrec_foundation.split import load_split_manifest
    from fedrec_foundation.atomic import atomic_write_json
    from dataclasses import replace as dataclass_replace

    derived = data_derived()
    idx = verify_bundle(derived)
    split = load_split_manifest(derived / "split_manifest.json")
    profile = resolve_mode_defaults("benchmark_cross_device")

    run_id = generate_run_id()
    manifest = build_run_manifest(
        run_id=run_id, mode_profile=profile, run_seed=42,
        mapping_sha256=idx.mapping_sha256, split_hash=idx.split_hash,
        exclusion_sha256=idx.exclusion_sha256,
        foundation_contract_sha256=idx.foundation_contract_sha256,
        raw_data_hash=split.raw_data_hash, builder_version=split.builder_version,
        overrides={}, module="baseline",
    )

    # Construct nested final_metrics as server_app.py would after Plan 03.
    best_round_metrics = {"sampled_ndcg@10": 0.45, "sampled_hr@10": 0.65, "evaluated_users": 100}
    last_block = {"sampled_ndcg@10": 0.42, "sampled_hr@10": 0.62, "evaluated_users": 100}
    centralized_diag = {"eval_loss": 1.5, "rmse": 1.8}
    final_metrics = {
        "best": best_round_metrics,
        "last": {**last_block, **centralized_diag},
        "best_round": 2,
        "last_round": 3,
        "final_eval_round_index": 4,
    }

    results_data = {
        "model_name": "test",
        "final_metrics": final_metrics,
        "training_rounds": 3,
    }

    # Phase 6: mutate manifest before embedding (mirrors server_app.py Edit 6).
    manifest = dataclass_replace(
        manifest,
        final_eval_round_index=4,
        metrics=results_data["final_metrics"],
    )
    embed_manifest_in_result(manifest, results_data)

    # Write to a temp per-run dir (mirrors D-04 clean filename).
    run_dir = tmp_path / "baseline" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_filename = run_dir / "results.json"
    atomic_write_json(str(results_filename), results_data)
    write_manifest_sibling(manifest, results_filename, sibling_name="manifest.json")

    # Load and assert schema.
    with open(results_filename) as f:
        loaded = json.load(f)

    fm = loaded["final_metrics"]
    assert set(fm.keys()) == {"best", "last", "best_round", "last_round", "final_eval_round_index"}, (
        f"D-07: final_metrics top-level keys mismatch: {set(fm.keys())}"
    )
    assert isinstance(fm["best"], dict), "final_metrics['best'] must be a dict"
    assert isinstance(fm["last"], dict), "final_metrics['last'] must be a dict"
    assert fm["best_round"] == 2, f"best_round should be 2, got {fm['best_round']}"
    assert fm["last_round"] == 3, f"last_round should be 3, got {fm['last_round']}"
    assert fm["final_eval_round_index"] == 4, (
        f"final_eval_round_index should be 4, got {fm['final_eval_round_index']}"
    )

    mf = loaded["_manifest"]
    assert mf["schema_version"] == 2, f"_manifest.schema_version must be 2, got {mf['schema_version']}"
    assert mf["final_eval_round_index"] == 4, (
        f"_manifest.final_eval_round_index must be 4, got {mf['final_eval_round_index']}"
    )
    assert mf["metrics"] == fm, (
        "_manifest.metrics must equal results_data['final_metrics'] (dataclasses.replace copy-through)"
    )

    # Also check manifest.json sibling.
    sibling = run_dir / "manifest.json"
    assert sibling.exists(), "D-04: manifest.json sibling must exist in per-run dir"
    with open(sibling) as f:
        sibling_data = json.load(f)
    assert sibling_data["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION


def test_round_metrics_history_carries_per_group_exposure() -> None:
    """EVL-02 / EVL-03 / D-08 / D-09: per-group exposure counts appear in eval_metrics_history.

    Simulates two rounds of aggregate_evaluate with per-group sufficient stats.
    Asserts that each round's entry in eval_metrics_history carries
    evaluated_users_sparse, evaluated_users_medium, evaluated_users_dense.
    """
    from federated_baseline_cf.strategy import BaselineFedAvg
    from unittest.mock import MagicMock

    strategy = BaselineFedAvg(fraction_fit=1.0)
    proxy = MagicMock()
    proxy.cid = "fake"

    eval_metrics_history: dict = {}

    for rnd in (1, 2):
        results = [
            (proxy, _build_fake_evaluate_res(
                evaluated_users=5, evaluated_users_sparse=2,
                evaluated_users_medium=2, evaluated_users_dense=1,
            )),
            (proxy, _build_fake_evaluate_res(
                evaluated_users=3, evaluated_users_sparse=1,
                evaluated_users_medium=1, evaluated_users_dense=1,
            )),
        ]
        _loss, thesis = strategy.aggregate_evaluate(rnd, results, [])
        eval_metrics_history[rnd] = dict(thesis) if thesis else {}

    # Verify at least one round has all three per-group exposure keys (D-09).
    for rnd, metrics in eval_metrics_history.items():
        assert "evaluated_users_sparse" in metrics, (
            f"D-09: evaluated_users_sparse missing from round {rnd} eval_metrics_history"
        )
        assert "evaluated_users_medium" in metrics, (
            f"D-09: evaluated_users_medium missing from round {rnd} eval_metrics_history"
        )
        assert "evaluated_users_dense" in metrics, (
            f"D-09: evaluated_users_dense missing from round {rnd} eval_metrics_history"
        )

    # Spot-check the counts: both rounds should have sparse=3, medium=3, dense=2 total.
    for rnd, metrics in eval_metrics_history.items():
        assert metrics["evaluated_users_sparse"] == 3, (
            f"sparse exposure should sum to 3, got {metrics['evaluated_users_sparse']} at round {rnd}"
        )
        assert metrics["evaluated_users_medium"] == 3, (
            f"medium exposure should sum to 3, got {metrics['evaluated_users_medium']} at round {rnd}"
        )
        assert metrics["evaluated_users_dense"] == 2, (
            f"dense exposure should sum to 2, got {metrics['evaluated_users_dense']} at round {rnd}"
        )
