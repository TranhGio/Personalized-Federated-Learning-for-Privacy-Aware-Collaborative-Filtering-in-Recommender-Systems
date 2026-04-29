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


# =============================================================================
# Phase 6 Plan 04: D-02/D-06/D-07 + EVL-01/02/03/04/06 integration tests
# =============================================================================

def _server_app_src() -> str:
    """Return the text of federated_personalized_cf/server_app.py."""
    src_path = Path(__file__).resolve().parents[1].joinpath(
        "federated_personalized_cf", "server_app.py"
    )
    return src_path.read_text()


def test_results_path_repo_root_anchored() -> None:
    """D-02: server_app.py imports module_run_results_dir and never uses Path('../results/federated').

    Source-level assertions (acceptance criteria 1, 4, 5):
    - ``from fedrec_foundation.paths import module_run_results_dir`` present (grep count == 1).
    - ``Path('../results/federated')`` literal absent (D-02 hard cutover).
    - ``module_run_results_dir(_MODULE, run_id)`` call present.
    Also validates the resolver itself:
    - ``module_run_results_dir("personalized", "test_run_abc")`` resolves to
      ``<repo>/results/federated/personalized/test_run_abc/`` (D-01 layout).
    - Returned path is absolute (D-02 anchoring).
    - manifest.json write via write_manifest_sibling with sibling_name='manifest.json'
      lands in the same per-run directory (D-04).
    """
    from fedrec_foundation.paths import module_run_results_dir, repo_root

    src = _server_app_src()

    # Source-level checks.
    assert src.count("from fedrec_foundation.paths import module_run_results_dir") >= 1, (
        "D-02: 'from fedrec_foundation.paths import module_run_results_dir' not found in server_app.py"
    )
    assert "Path('../results/federated')" not in src, (
        "D-02 hard cutover: Path('../results/federated') literal must be gone from server_app.py"
    )
    assert "Path(\"../results/federated\")" not in src, (
        "D-02 hard cutover: Path('../results/federated') literal must be gone from server_app.py"
    )
    assert "module_run_results_dir(_MODULE, run_id)" in src, (
        "D-02: 'module_run_results_dir(_MODULE, run_id)' call missing from server_app.py"
    )
    assert '_MODULE: str = "personalized"' in src, (
        "D-02: '_MODULE: str = \"personalized\"' local constant missing from server_app.py"
    )

    # Functional check: resolver returns the correct repo-root-anchored path.
    run_dir = module_run_results_dir("personalized", "test_run_phase6_plan04")
    repo = repo_root()
    expected = repo / "results" / "federated" / "personalized" / "test_run_phase6_plan04"
    assert run_dir == expected, (
        f"module_run_results_dir layout mismatch: got {run_dir}, expected {expected}"
    )
    assert run_dir.is_absolute(), "D-02: module_run_results_dir must return an absolute path"
    assert run_dir.is_dir(), "module_run_results_dir must create the directory eagerly"

    # D-04 manifest sibling check: write_manifest_sibling with sibling_name='manifest.json'.
    src_check = _server_app_src()
    assert 'sibling_name="manifest.json"' in src_check or "sibling_name='manifest.json'" in src_check, (
        "D-04: sibling_name='manifest.json' kwarg missing from server_app.py write_manifest_sibling call"
    )
    # Also check that the atomic_write_json import is present (replaces json.dump).
    assert "from fedrec_foundation.atomic import atomic_write_json" in src_check, (
        "MINOR: atomic_write_json import missing from server_app.py"
    )
    assert "json.dump(results_data" not in src_check, (
        "MINOR: legacy json.dump(results_data, ...) must be replaced by atomic_write_json"
    )


def test_extra_eval_round_replaces_history_lookup() -> None:
    """D-06: extra-eval-round wiring replaces the silent eval_metrics_history lookup.

    Source-level checks:
    - The D-06-forbidden ``eval_metrics_history.get(final_round_for_metrics`` literal
      must be absent (the bug has been removed).
    - ``final_eval_round_index`` must appear at least 5 times (assignment + multiple uses).
    - ``best_round_metrics`` must appear at least 4 times.
    - ``strategy.aggregate_evaluate(final_eval_round_index`` call must be present
      (the extra-eval-round calls the strategy aggregator).
    - The Pitfall-9 guard ``max(eval_metrics_history.keys())`` must be present.
    - The mode conditional ``if mode in ("benchmark_cross_device", "thesis_crossdevice_main", "paper_compat_pfedrec")``
      must be present (Pitfall 8 cross-silo coexistence + Phase 7 D-04 thesis-mode gate).
    Also checks that the D-06 extra-eval-round is ordered AFTER arrays = best_arrays.
    """
    src = _server_app_src()

    # D-06 forbidden lookup must be gone.
    assert "eval_metrics_history.get(final_round_for_metrics" not in src, (
        "D-06 BUG STILL PRESENT: 'eval_metrics_history.get(final_round_for_metrics' "
        "must be removed from server_app.py"
    )

    # Extra-eval-round artifacts.
    assert src.count("final_eval_round_index") >= 5, (
        f"'final_eval_round_index' appears {src.count('final_eval_round_index')} times, "
        "expected >= 5"
    )
    assert src.count("best_round_metrics") >= 4, (
        f"'best_round_metrics' appears {src.count('best_round_metrics')} times, "
        "expected >= 4"
    )
    assert "strategy.aggregate_evaluate(final_eval_round_index" in src, (
        "D-06: 'strategy.aggregate_evaluate(final_eval_round_index' call missing — "
        "extra-eval-round must invoke strategy.aggregate_evaluate"
    )

    # Pitfall 9: last_round from max key.
    assert "max(eval_metrics_history.keys())" in src, (
        "Pitfall 9: 'max(eval_metrics_history.keys())' missing — "
        "last_round must derive from max key, not actual_rounds"
    )

    # Pitfall 8 + Phase 7 D-04: cross-silo mode branch carries thesis_crossdevice_main.
    assert 'if mode in ("benchmark_cross_device", "thesis_crossdevice_main", "paper_compat_pfedrec")' in src, (
        "Phase 7 D-04: thesis_crossdevice_main mode joins the per-run-dir gate"
    )

    # D-06 ordering: best_arrays restore must precede extra-eval-round.
    idx_best = src.find("arrays = best_arrays")
    idx_extra = src.find("strategy.aggregate_evaluate(final_eval_round_index")
    assert idx_best >= 0, "'arrays = best_arrays' not found — D-27 restore block missing"
    assert idx_extra >= 0, "strategy.aggregate_evaluate(final_eval_round_index) not found"
    assert idx_extra > idx_best, (
        "D-06 ordering violation: extra-eval-round must appear AFTER arrays = best_arrays"
    )


def test_canonical_artifact_carries_best_and_last_blocks() -> None:
    """D-07: final_metrics is nested {best, last, best_round, last_round, final_eval_round_index}.

    Source-level checks:
    - ``final_metrics = {`` block with ``"best":`` and ``"last":`` keys.
    - W&B summary uses ``best/`` and ``last/`` namespaces, not ``final/``.
    - dataclass_replace import for post-build manifest mutation.
    - Manifest is mutated via dataclass_replace AFTER final_metrics is assigned.
    """
    src = _server_app_src()

    # Nested schema must be present.
    assert '"best"' in src or "'best'" in src, (
        "D-07: 'best' key missing from final_metrics dict in server_app.py"
    )
    assert '"last"' in src or "'last'" in src, (
        "D-07: 'last' key missing from final_metrics dict in server_app.py"
    )
    assert '"best_round"' in src or "'best_round'" in src, (
        "D-07: 'best_round' key missing from final_metrics dict in server_app.py"
    )
    assert '"last_round"' in src or "'last_round'" in src, (
        "D-07: 'last_round' key missing from final_metrics dict in server_app.py"
    )
    assert '"final_eval_round_index"' in src or "'final_eval_round_index'" in src, (
        "D-07: 'final_eval_round_index' key missing from final_metrics dict in server_app.py"
    )

    # W&B namespaces: final/* must be gone; best/* and last/* must be present.
    assert 'wandb.run.summary[f"final/' not in src, (
        "W&B migration: wandb.run.summary[f\"final/...\"] still present in server_app.py"
    )
    assert "wandb.run.summary[f\"best/" in src or "wandb.run.summary[f'best/" in src, (
        "W&B migration: wandb.run.summary[f\"best/...\"] missing from server_app.py"
    )
    assert "wandb.run.summary[f\"last/" in src or "wandb.run.summary[f'last/" in src, (
        "W&B migration: wandb.run.summary[f\"last/...\"] missing from server_app.py"
    )

    # dataclass_replace must be imported.
    assert "from dataclasses import replace as dataclass_replace" in src, (
        "dataclass_replace import missing from server_app.py"
    )

    # Edit-order check: final_metrics block before dataclass_replace call.
    idx_final = src.find("final_metrics = {")
    idx_replace = src.find("dataclass_replace(manifest")
    assert idx_final >= 0, "'final_metrics = {' block not found"
    assert idx_replace >= 0, "'dataclass_replace(manifest' not found"
    assert idx_replace > idx_final, (
        "Edit-order invariant violated: final_metrics block must appear before dataclass_replace"
    )

    # np.float64 coercion present (plan-checker MAJOR requirement).
    assert "float(v) if isinstance(v, (int, float))" in src, (
        "MAJOR: np.float64 JSON-safe coercion 'float(v) if isinstance(v, (int, float))' "
        "missing from server_app.py (path-(b) at best_round_metrics assignment site)"
    )

    # schema_version == 2 in manifest.
    from fedrec_foundation.manifest import RUN_MANIFEST_SCHEMA_VERSION
    assert RUN_MANIFEST_SCHEMA_VERSION == 2, (
        f"Expected RUN_MANIFEST_SCHEMA_VERSION == 2, got {RUN_MANIFEST_SCHEMA_VERSION}"
    )


def test_round_metrics_history_carries_per_group_exposure() -> None:
    """D-09: strategy.aggregate_evaluate emits per-group evaluated_users counts.

    Source-level check that the strategy's aggregate_evaluate produces
    evaluated_users_sparse, evaluated_users_medium, evaluated_users_dense keys.
    Uses a live strategy call (unit-test compatible; no Grid needed).
    """
    from unittest.mock import MagicMock
    from flwr.common import Code, EvaluateRes, Status

    from federated_personalized_cf.strategy import PersonalizedSplitFedAvg

    strategy = PersonalizedSplitFedAvg(fraction_fit=0.1)
    proxy = MagicMock()
    proxy.cid = "c"

    # Construct results that populate per-group stats.
    results = [
        (proxy, EvaluateRes(Status(Code.OK, "ok"), 0.5, 10, {
            "eval_loss": 0.5,
            "hit_count_overall_at10": 3, "ndcg_sum_overall_at10": 0.5, "evaluated_users": 10,
            "hit_count_sparse_at10": 1, "ndcg_sum_sparse_at10": 0.1, "evaluated_users_sparse": 3,
            "hit_count_medium_at10": 2, "ndcg_sum_medium_at10": 0.3, "evaluated_users_medium": 5,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.1, "evaluated_users_dense": 2,
        })),
        (proxy, EvaluateRes(Status(Code.OK, "ok"), 0.5, 5, {
            "eval_loss": 0.5,
            "hit_count_overall_at10": 1, "ndcg_sum_overall_at10": 0.2, "evaluated_users": 5,
            "hit_count_sparse_at10": 0, "ndcg_sum_sparse_at10": 0.0, "evaluated_users_sparse": 1,
            "hit_count_medium_at10": 1, "ndcg_sum_medium_at10": 0.2, "evaluated_users_medium": 3,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.0, "evaluated_users_dense": 1,
        })),
    ]
    _loss, metrics = strategy.aggregate_evaluate(1, results, [])

    # D-09: full required key set must be present (plan-07 strengthening:
    # canonical required_keys = {evaluated_users, evaluated_users_sparse,
    # evaluated_users_medium, evaluated_users_dense}).
    required_keys = {
        "evaluated_users",
        "evaluated_users_sparse",
        "evaluated_users_medium",
        "evaluated_users_dense",
    }
    missing = required_keys - set(metrics.keys())
    assert not missing, (
        f"D-09 regression: strategy.aggregate_evaluate output missing keys: {missing}. "
        f"Available: {sorted(metrics.keys())[:20]}"
    )

    # Verify the summed values are correct (not averaged per-client).
    assert metrics["evaluated_users_sparse"] == 4, (
        f"D-09: evaluated_users_sparse should be 3+1=4, got {metrics['evaluated_users_sparse']}"
    )
    assert metrics["evaluated_users_medium"] == 8, (
        f"D-09: evaluated_users_medium should be 5+3=8, got {metrics['evaluated_users_medium']}"
    )
    assert metrics["evaluated_users_dense"] == 3, (
        f"D-09: evaluated_users_dense should be 2+1=3, got {metrics['evaluated_users_dense']}"
    )

    # Source-level check: server_app stores thesis_metrics dict (which carries all
    # per-group keys) into eval_metrics_history. Verify the storage line is present.
    src = _server_app_src()
    # eval_metrics_history[round_num] = dict(thesis_metrics) stores all per-group keys
    # without explicitly naming them; the strategy output carries evaluated_users_sparse etc.
    assert "eval_metrics_history[round_num] = dict(thesis_metrics)" in src, (
        "D-09: eval_metrics_history storage line 'eval_metrics_history[round_num] = dict(thesis_metrics)' "
        "missing from server_app.py — per-group exposure counts won't be persisted per round"
    )
