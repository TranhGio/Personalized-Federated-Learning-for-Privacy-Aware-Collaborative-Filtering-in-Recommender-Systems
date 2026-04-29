"""Server integration tests (Phase 4 Plan 05).

Covers the ADP-03 / ADP-06 / ADP-08 / D-02 / D-05 / D-06 / D-07 / D-13 / D-15 /
D-16 invariants closed by Plan 05:

- **ADP-06 reproducibility**: ``server_rng(run_seed)``-driven per-round client
  selection is byte-identical across RNG instances; different seeds yield
  different selections.
- **ADP-06 strategy wire-up**: ``AdaptiveSplitFedAvg.aggregate_evaluate``
  returns sum-based thesis metrics, not mean-of-per-client ratios (would be
  the silent sparse-user double-count bug).
- **ADP-08 / D-15 / D-06**: ``build_run_manifest`` with ``module="adaptive"``
  integrates all four IMP-2 foundation fingerprints AND the embedded ``_manifest``
  dict is mutable post-embed so D-06 can inject ``best_prototype`` afterwards.
- **D-13 cold-start math**: the server-side cold-start probe counts selected
  partitions whose ``.embedding_cache/{run_id}/partition_{pid}.pt`` does NOT
  exist before the round fires.
- **D-02 frozen cross-silo guard** (source-level regression check): the
  adaptive server_app.py raises ``NotImplementedError`` inside a
  ``cross_silo_legacy`` branch — a direct runtime check would need a live
  Grid, so we assert the guard is present in source.
- **D-05 best_prototype snapshot sequence**: ``snapshot_best_prototype`` is
  called inside the best-metric branch (proximity guard).
- **D-07 best_prototype restore sequence**: ``strategy._global_prototype =
  strategy.best_prototype`` is restored AFTER ``arrays = best_arrays``
  (proximity guard).

Tests are skipped if the foundation bundle (``data/derived/foundation_index.json``)
is absent so a minimal clone doesn't fail CI.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FOUNDATION_INDEX = _REPO_ROOT / "data" / "derived" / "foundation_index.json"

pytestmark = pytest.mark.skipif(
    not _FOUNDATION_INDEX.exists(),
    reason="foundation bundle not committed (data/derived/foundation_index.json missing)",
)


# =============================================================================
# ADP-06: server_rng reproducibility
# =============================================================================
def test_server_rng_reproducible_per_round_selection() -> None:
    """ADP-06: server_rng(run_seed) produces byte-identical sequences across instances.

    Pure-RNG property — necessary but not sufficient. The load-bearing
    invariant (byte-identical selected_clients_per_round across subprocess
    reruns) is proven by Plan 06's subprocess determinism guard.
    """
    from fedrec_foundation.rng import server_rng

    rng1 = server_rng(42)
    rng2 = server_rng(42)
    seq1 = [tuple(rng1.sample(range(6040), 50)) for _ in range(3)]
    seq2 = [tuple(rng2.sample(range(6040), 50)) for _ in range(3)]
    assert seq1 == seq2, "server_rng(42) must produce byte-identical sequences across instances"


def test_server_rng_different_seeds_different_selections() -> None:
    """ADP-06 negative guard: different run_seeds -> different sequences."""
    from fedrec_foundation.rng import server_rng

    rng1 = server_rng(42)
    rng2 = server_rng(43)
    s1 = rng1.sample(range(6040), 50)
    s2 = rng2.sample(range(6040), 50)
    assert s1 != s2, "Different seeds MUST yield different client-selection sequences"


# =============================================================================
# ADP-06: AdaptiveSplitFedAvg sum aggregation sanity
# =============================================================================
def test_adaptive_split_fedavg_aggregate_evaluate_sum_not_average(fake_client_proxy) -> None:
    """ADP-06 strategy wire-up: sum-based ratios, not mean-of-per-client-ratios.

    Same shape as Phase 3 personalized PSN-04 test: Client A with 1 hit on 1
    user (HR@10=1.0) and Client B with 0 hits on 99 users (HR@10=0.0). A mean
    of per-client ratios would be 0.5 (WRONG); the correct sum-based ratio is
    1/100 = 0.01. The strategy MUST emit 0.01.
    """
    from flwr.common import Code, EvaluateRes, Status

    from federated_adaptive_personalized_cf.strategy import AdaptiveSplitFedAvg

    strategy = AdaptiveSplitFedAvg(fraction_fit=0.1)
    proxy = fake_client_proxy

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
    _loss, metrics = strategy.aggregate_evaluate(server_round=1, results=results, failures=[])
    assert metrics["sampled_hr@10"] == pytest.approx(1.0 / 100.0, abs=1e-9), (
        f"ADP-06 strategy: sum-based ratio must be 1/100=0.01, got {metrics['sampled_hr@10']}"
    )
    assert metrics["sampled_hr@10"] < 0.5, (
        "ADP-06 strategy: aggregation is NOT averaging per-client ratios"
    )


# =============================================================================
# ADP-08 + D-15 + D-06: build_run_manifest + _manifest dict extensibility
# =============================================================================
def test_build_run_manifest_module_adaptive_with_best_prototype() -> None:
    """ADP-08 + D-15: build_run_manifest with module='adaptive' integrates the IMP-2
    fingerprints AND the embedded _manifest dict is mutable post-embed (D-06)."""
    from fedrec_foundation.bundle import verify_bundle
    from fedrec_foundation.manifest import (
        build_run_manifest,
        embed_manifest_in_result,
        generate_run_id,
    )
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
        module="adaptive",
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
    # Module flag is Phase-4-specific.
    assert manifest.module == "adaptive"

    # D-06: prove the _manifest dict is mutable post-embed (Research §Pattern 2:
    # post-hoc mutation is safe — the dict is held by reference inside results_data).
    results_data = embed_manifest_in_result(manifest, {})
    _manifest_dict = results_data["_manifest"]
    for fingerprint in ("mapping_sha256", "split_hash", "exclusion_sha256", "foundation_contract_sha256"):
        assert fingerprint in _manifest_dict, (
            f"IMP-2 fingerprint {fingerprint} missing from embedded _manifest"
        )
    # Mutate post-embed: best_prototype injection should stick.
    _manifest_dict["best_prototype"] = [1.0] * 128
    assert results_data["_manifest"]["best_prototype"] == [1.0] * 128, (
        "D-06 violated: _manifest dict is NOT mutable post-embed; "
        "best_prototype injection cannot persist"
    )


# =============================================================================
# D-13: cold-start counter arithmetic
# =============================================================================
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

    # cold_start_rate matches the final results JSON shape: ratio over the
    # total client selections across rounds.
    total_selections = len(selected_pids)
    cold_start_rate = cold_count / total_selections
    assert cold_start_rate == pytest.approx(0.5, abs=1e-9)


# =============================================================================
# D-02: cross-silo source-level regression guard
# =============================================================================
def test_cross_silo_legacy_mode_raises_not_implemented() -> None:
    """D-02 frozen cross-silo: source-level regression guard.

    @app.main() raises NotImplementedError when mode=='cross_silo_legacy'.
    A runtime check would require a live Grid + Context, so we assert the
    guard is present in source (string + proximity check). This catches
    accidental removal of the D-02 branch during future refactors.
    """
    src_path = _REPO_ROOT / "federated-adaptive-personalized-cf" / \
        "federated_adaptive_personalized_cf" / "server_app.py"
    src = src_path.read_text()
    # Must mention cross_silo_legacy as a control-flow token.
    assert "cross_silo_legacy" in src, "D-02 guard removed: cross_silo_legacy token missing"
    assert "raise NotImplementedError" in src, "D-02 guard removed: NotImplementedError raise missing"
    # Proximity check: the raise must live near the cross_silo_legacy branch
    # (first occurrence in source should point at the guard, not a comment).
    idx = src.index("cross_silo_legacy")
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
# D-05 + D-07: best_prototype snapshot + restore sequence (source-level guards)
# =============================================================================
def test_snapshot_best_prototype_called_inside_best_metric_branch() -> None:
    """D-05 + D-07: best_prototype snapshot fires in the best-metric branch
    AND best_prototype restore follows the arrays restore.

    Full end-to-end integration requires a live Grid we cannot instantiate
    cheaply, so we assert proximity in source order:
      - ``snapshot_best_prototype`` appears AFTER ``best_metric = current_ndcg``
        (D-05 — fires at the SAME moment as best_arrays snapshot).
      - ``strategy._global_prototype = strategy.best_prototype`` appears AFTER
        ``arrays = best_arrays`` (D-07 — restore both pieces of state).
    """
    src_path = _REPO_ROOT / "federated-adaptive-personalized-cf" / \
        "federated_adaptive_personalized_cf" / "server_app.py"
    src = src_path.read_text()

    # D-05: snapshot_best_prototype must appear in the source...
    assert "snapshot_best_prototype" in src, (
        "D-05 violated: server_app.py does not call strategy.snapshot_best_prototype"
    )
    # ...and it must appear AFTER the best_metric = current_ndcg assignment (proximity).
    best_metric_assign_idx = src.find("best_metric = current_ndcg")
    assert best_metric_assign_idx != -1, (
        "best_metric = current_ndcg line missing — D-27 best-round tracking regressed"
    )
    snapshot_idx = src.find("snapshot_best_prototype(", best_metric_assign_idx)
    assert snapshot_idx != -1, (
        "D-05 violated: snapshot_best_prototype() call missing AFTER "
        "best_metric = current_ndcg assignment"
    )
    assert snapshot_idx > best_metric_assign_idx, (
        "D-05 violated: snapshot_best_prototype must fire inside the best-metric branch, "
        "AFTER best_metric assignment"
    )

    # D-07 restore sequence: strategy._global_prototype = strategy.best_prototype
    # appears AFTER the best-round-restore "arrays = best_arrays" line.
    # NOTE: use rfind for the prototype restore because the docstring at the
    # top of the module ALSO mentions ``strategy._global_prototype = strategy.best_prototype``
    # as documentation; the actual code statement is the LAST occurrence in the file.
    arrays_restore_idx = src.find("arrays = best_arrays")
    assert arrays_restore_idx != -1, (
        "arrays = best_arrays line missing — D-27 best-round restore regressed"
    )
    proto_restore_idx = src.rfind("strategy._global_prototype = strategy.best_prototype")
    assert proto_restore_idx != -1, (
        "D-07 violated: prototype restore missing from server_app.py"
    )
    assert proto_restore_idx > arrays_restore_idx, (
        "D-07 violated: prototype restore code statement must follow "
        "`arrays = best_arrays` in source order"
    )


# =============================================================================
# UAT GAP-04-01: Sibling RecordDict extraction (D-05/D-06/D-16 runtime fix)
# =============================================================================
# Surfaced by results/federated/adaptive/20260427-132620-eb2d19_results.json:
# best_prototype was [0.0]*128 + alpha_diagnostics_history was missing because
# server_app.py:670 only extracted the strict "metrics" MetricRecord and dropped
# the sibling user_prototype + alpha_diagnostics records that the client emits
# as separate top-level RecordDict records (per client_app.py:741, 747 — the
# strict FitMetricsContract D-21 forbids inline free-form extras).
#
# These tests pin the post-fix contract: the helper merges siblings into
# metrics_dict so strategy._aggregate_prototypes (strategy.py:228) and the D-16
# alpha aggregator (server_app.py:725) read them via fit_res.metrics.

def test_extract_sibling_records_user_prototype() -> None:
    """D-05/D-06: ``user_prototype`` sibling -> ``metrics_dict[USER_PROTOTYPE_KEY]`` list.

    Constructs a RecordDict in the exact shape ``client_app.py`` emits at line 741
    and asserts the helper unwraps the inner List[float] into ``metrics_dict``
    where ``AdaptiveSplitFedAvg._aggregate_prototypes`` can read it.
    """
    from flwr.common.record import MetricRecord, RecordDict

    from federated_adaptive_personalized_cf.server_app import _extract_sibling_records
    from federated_adaptive_personalized_cf.strategy import USER_PROTOTYPE_KEY

    proto_payload = [0.1, 0.2, 0.3, 0.4, 0.5]
    record_dict = RecordDict({
        "metrics": MetricRecord({
            "hit_count_overall_at10": 1,
            "evaluated_users": 1,
        }),
        USER_PROTOTYPE_KEY: MetricRecord({USER_PROTOTYPE_KEY: proto_payload}),
    })
    metrics_dict = {"hit_count_overall_at10": 1, "evaluated_users": 1}

    result = _extract_sibling_records(record_dict, metrics_dict)

    assert USER_PROTOTYPE_KEY in result, (
        "GAP-04-01 regressed: user_prototype sibling not merged into metrics_dict"
    )
    assert list(result[USER_PROTOTYPE_KEY]) == proto_payload, (
        f"user_prototype payload corrupted: expected {proto_payload}, "
        f"got {result[USER_PROTOTYPE_KEY]}"
    )
    # Mutates in place + returns same reference
    assert result is metrics_dict
    # Original metrics keys preserved
    assert result["hit_count_overall_at10"] == 1
    assert result["evaluated_users"] == 1


def test_extract_sibling_records_alpha_diagnostics() -> None:
    """D-16: ``alpha_diagnostics`` sibling -> ``metrics_dict["alpha_diagnostics"]`` dict.

    Pins the contract that the D-16 server-side aggregator at server_app.py:725
    reads ``fit_res.metrics["alpha_diagnostics"]`` (a Dict[str, float] with the
    6 scalar fields) populated by this helper from the sibling MetricRecord
    emitted by client_app.py:747.
    """
    from flwr.common.record import MetricRecord, RecordDict

    from federated_adaptive_personalized_cf.server_app import _extract_sibling_records

    alpha_payload = {
        "alpha_mean": 0.55,
        "alpha_std": 0.12,
        "alpha_p25": 0.42,
        "alpha_p50": 0.55,
        "alpha_p75": 0.68,
        "alpha_clip_hit_rate": 0.05,
    }
    record_dict = RecordDict({
        "metrics": MetricRecord({}),
        "alpha_diagnostics": MetricRecord(alpha_payload),
    })
    metrics_dict: dict = {}

    _extract_sibling_records(record_dict, metrics_dict)

    assert "alpha_diagnostics" in metrics_dict, (
        "GAP-04-01 regressed: alpha_diagnostics sibling not merged into metrics_dict"
    )
    assert metrics_dict["alpha_diagnostics"] == alpha_payload, (
        f"alpha_diagnostics payload corrupted: expected {alpha_payload}, "
        f"got {metrics_dict['alpha_diagnostics']}"
    )


def test_extract_sibling_records_no_siblings_no_op() -> None:
    """No siblings -> ``metrics_dict`` unchanged (defensive).

    Required because not every client emits both siblings (e.g., the
    ``compute_user_prototype`` hook returns None for a model without a personal
    user row, and ``alpha_diagnostics`` is None when ``enable_per_user_alpha=false``).
    The server must still build a valid fit_res in those cases without injecting
    spurious ``user_prototype: None`` or ``alpha_diagnostics: {}`` entries.
    """
    from flwr.common.record import MetricRecord, RecordDict

    from federated_adaptive_personalized_cf.server_app import _extract_sibling_records
    from federated_adaptive_personalized_cf.strategy import USER_PROTOTYPE_KEY

    record_dict = RecordDict({"metrics": MetricRecord({"foo": 1})})
    metrics_dict = {"foo": 1}

    _extract_sibling_records(record_dict, metrics_dict)

    assert USER_PROTOTYPE_KEY not in metrics_dict, (
        "Spurious user_prototype injected when sibling absent"
    )
    assert "alpha_diagnostics" not in metrics_dict, (
        "Spurious alpha_diagnostics injected when sibling absent"
    )
    assert metrics_dict == {"foo": 1}, (
        f"metrics_dict corrupted by no-op path: {metrics_dict}"
    )


# =============================================================================
# Phase 6 Plan 05: EVL-01/02/03/04/06 + Pitfall-4 regression guards
# =============================================================================


# -----------------------------------------------------------------------------
# Test 1 (6-05-01): results path is repo-root anchored via module_run_results_dir
# -----------------------------------------------------------------------------
def test_results_path_repo_root_anchored() -> None:
    """EVL-04 D-02: module_run_results_dir resolves to <repo>/results/federated/adaptive/<run_id>/.

    Verifies server_app.py imports module_run_results_dir (not a bare Path) and
    that the source contains the module_run_results_dir call-site for cross-device modes.
    """
    src_path = _REPO_ROOT / "federated-adaptive-personalized-cf" / \
        "federated_adaptive_personalized_cf" / "server_app.py"
    src = src_path.read_text()

    # D-02: old module-relative path must be gone
    assert 'Path("../results/federated/adaptive")' not in src, (
        "D-02 VIOLATED: old module-relative path still present in server_app.py; "
        "must be replaced by module_run_results_dir(_MODULE, run_id)"
    )
    # Foundation helper must be imported
    assert "from fedrec_foundation.paths import module_run_results_dir" in src, (
        "EVL-04 VIOLATED: module_run_results_dir not imported in server_app.py"
    )
    # Call site must be present for cross-device mode
    assert "module_run_results_dir(_MODULE, run_id)" in src, (
        "EVL-04 D-02 VIOLATED: module_run_results_dir(_MODULE, run_id) call site missing"
    )
    # _MODULE local constant
    assert '_MODULE: str = "adaptive"' in src, (
        "EVL-04 VIOLATED: _MODULE constant missing from server_app.py"
    )
    # D-04 clean filename: sibling_name kwarg must be present
    assert 'sibling_name="manifest.json"' in src, (
        "D-04 VIOLATED: sibling_name='manifest.json' call missing from server_app.py"
    )
    # atomic_write_json must replace json.dump
    assert "atomic_write_json" in src, (
        "MINOR VIOLATED: atomic_write_json not imported/used in server_app.py"
    )
    assert "json.dump(results_data" not in src, (
        "MINOR VIOLATED: legacy json.dump(results_data ...) still present; must use atomic_write_json"
    )


# -----------------------------------------------------------------------------
# Test 2 (6-05-01): extra-eval-round block is present and replaces D-06 forbidden lookup
# -----------------------------------------------------------------------------
def test_extra_eval_round_replaces_forbidden_history_lookup() -> None:
    """EVL-01 D-06: extra-eval-round block wired; D-06 forbidden lookup removed.

    Checks source-level invariants without running a live Grid:
    1. 'final_metrics = dict(eval_metrics_history.get(final_round_for_metrics' is absent.
    2. 'final_eval_round_index' is present (the D-06 broadcast block token).
    3. 'strategy.aggregate_evaluate(final_eval_round_index' appears (broadcast result feeds best).
    4. nested final_metrics schema: 'best_round_metrics or last_block' pattern.
    5. max(eval_metrics_history.keys()) present (Pitfall 9 last-round guard).
    """
    src_path = _REPO_ROOT / "federated-adaptive-personalized-cf" / \
        "federated_adaptive_personalized_cf" / "server_app.py"
    src = src_path.read_text()

    # D-06 forbidden lookup must be absent
    assert "final_metrics = dict(eval_metrics_history.get(final_round_for_metrics" not in src, (
        "D-06 BUG: forbidden eval_metrics_history history lookup still present"
    )
    # D-06 broadcast block tokens
    assert "final_eval_round_index" in src, (
        "EVL-01 D-06 VIOLATED: final_eval_round_index token missing — extra-eval-round not wired"
    )
    # Nested schema: best block derives from broadcast result
    assert '"best"' in src or "'best'" in src, (
        "EVL-06 D-07 VIOLATED: nested best block missing from final_metrics construction"
    )
    assert '"last"' in src or "'last'" in src, (
        "EVL-06 D-07 VIOLATED: nested last block missing from final_metrics construction"
    )
    # Pitfall 9 last-round guard
    assert "max(eval_metrics_history.keys())" in src, (
        "Pitfall 9 VIOLATED: max(eval_metrics_history.keys()) guard missing"
    )
    # dataclasses.replace import and call-site for manifest mutation
    assert "from dataclasses import replace as dataclass_replace" in src, (
        "VIOLATED: dataclass_replace import missing from server_app.py"
    )
    assert "dataclass_replace" in src, (
        "VIOLATED: dataclass_replace call site missing from server_app.py"
    )


# -----------------------------------------------------------------------------
# Test 3 (6-05-01): canonical artifact schema — best/last blocks + schema_version=2
# -----------------------------------------------------------------------------
def test_canonical_artifact_carries_best_and_last_blocks() -> None:
    """EVL-01 EVL-06 D-07: source carries nested final_metrics + manifest dataclass_replace.

    Also verifies Phase-4 D-06 best_prototype post-embed mutation is preserved verbatim
    (results_data['_manifest']['best_prototype'] still present in source).
    """
    src_path = _REPO_ROOT / "federated-adaptive-personalized-cf" / \
        "federated_adaptive_personalized_cf" / "server_app.py"
    src = src_path.read_text()

    # Manifest dataclass_replace must appear AFTER final_metrics block (edit-order invariant).
    # Note: final_metrics uses a type annotation: "final_metrics: Dict[str, Any] = {"
    # and the call site is "manifest = dataclass_replace(" (not "dataclass_replace(manifest").
    idx_final = src.find("final_metrics")
    idx_replace = src.find("manifest = dataclass_replace(")
    assert idx_final >= 0, (
        "final_metrics block missing — nested schema not implemented"
    )
    assert idx_replace > 0, (
        "dataclass_replace manifest mutation missing from server_app.py"
    )
    assert idx_replace > idx_final, (
        "Edit-order invariant VIOLATED: dataclass_replace must appear AFTER final_metrics block"
    )
    # Phase-4 D-06 post-embed mutation preserved verbatim
    assert "results_data[\"_manifest\"][\"best_prototype\"]" in src, (
        "Phase-4 D-06 VIOLATED: best_prototype post-embed mutation removed from server_app.py"
    )
    # best/* and last/* W&B summary namespaces
    assert 'wandb.run.summary[f"best/' in src, (
        'EVL-06 VIOLATED: best/* W&B summary namespace missing from server_app.py'
    )
    assert 'wandb.run.summary[f"last/' in src, (
        'EVL-06 VIOLATED: last/* W&B summary namespace missing from server_app.py'
    )
    # final/* must be gone (for thesis metrics; alpha/* and prototype/* are preserved)
    assert 'wandb.run.summary[f"final/' not in src, (
        'EVL-06 VIOLATED: legacy final/* W&B summary namespace still present'
    )


# -----------------------------------------------------------------------------
# Test 4 (6-05-01): per-group exposure history present in strategy output
# -----------------------------------------------------------------------------
def test_round_metrics_history_carries_per_group_exposure() -> None:
    """EVL-02 EVL-03 D-08 D-09: strategy.aggregate_evaluate emits evaluated_users per group.

    Creates a minimal fixture with known per-group sufficient stats; asserts the
    strategy emits evaluated_users_sparse / _medium / _dense alongside overall
    evaluated_users. These are D-09 exposure counts Phase 6 surfaces in the canonical block.
    """
    from flwr.common import Code, EvaluateRes, Status

    from federated_adaptive_personalized_cf.strategy import AdaptiveSplitFedAvg

    strategy = AdaptiveSplitFedAvg(fraction_fit=0.1)
    from unittest.mock import MagicMock
    proxy = MagicMock()
    proxy.cid = "test"

    results = [
        (proxy, EvaluateRes(Status(Code.OK, "ok"), 0.0, 5, {
            "eval_loss": 0.0,
            "hit_count_overall_at10": 2, "ndcg_sum_overall_at10": 1.5, "evaluated_users": 5,
            "hit_count_sparse_at10": 1, "ndcg_sum_sparse_at10": 0.5, "evaluated_users_sparse": 2,
            "hit_count_medium_at10": 1, "ndcg_sum_medium_at10": 0.7, "evaluated_users_medium": 2,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.3, "evaluated_users_dense": 1,
        })),
    ]
    _loss, metrics = strategy.aggregate_evaluate(server_round=1, results=results, failures=[])
    assert "evaluated_users" in metrics, "EVL-03 D-09: evaluated_users missing from strategy output"
    assert "evaluated_users_sparse" in metrics, "EVL-02 D-08: evaluated_users_sparse missing"
    assert "evaluated_users_medium" in metrics, "EVL-02 D-08: evaluated_users_medium missing"
    assert "evaluated_users_dense" in metrics, "EVL-02 D-08: evaluated_users_dense missing"
    assert int(metrics["evaluated_users"]) == 5
    assert int(metrics["evaluated_users_sparse"]) == 2
    assert int(metrics["evaluated_users_medium"]) == 2
    assert int(metrics["evaluated_users_dense"]) == 1


# -----------------------------------------------------------------------------
# Test 5 (6-05-02): PITFALL 4 REGRESSION GUARD — extra-eval-round attaches best_prototype
# -----------------------------------------------------------------------------
def test_extra_eval_broadcasts_best_prototype() -> None:
    """Pitfall-4: extra-eval ConfigRecord MUST carry strategy._global_prototype.tolist().

    Source-level invariant: verifies that the extra-eval-round message construction
    block attaches global_prototype to the eval ConfigRecord, mirroring the in-loop
    eval-config build site at server_app.py lines 814-815.

    Without this attachment, every client falls back to a zero or stale prototype
    during the canonical eval, and the best_* block reports lower NDCG@10 than
    the in-loop best_round_num round did (the warning sign in RESEARCH §Pitfall 4).
    """
    src_path = _REPO_ROOT / "federated-adaptive-personalized-cf" / \
        "federated_adaptive_personalized_cf" / "server_app.py"
    src = src_path.read_text()

    # The attachment must be present in the extra-eval block
    assert 'extra_eval_config_dict["global_prototype"]' in src, (
        "Pitfall-4 VIOLATED: extra_eval_config_dict['global_prototype'] assignment missing. "
        "The extra-eval-round ConfigRecord does NOT attach the restored best_prototype. "
        "Every client will fall back to a zero/stale prototype during canonical eval, "
        "causing best_* block NDCG@10 to be lower than the in-loop best_round did."
    )
    # The value must come from final_global_prototype (the restored prototype from D-07)
    assert "final_global_prototype.tolist()" in src, (
        "Pitfall-4 VIOLATED: final_global_prototype.tolist() not used in extra-eval ConfigRecord. "
        "Must mirror in-loop eval-config build site at server_app.py:814-815."
    )
    # The prototype attachment must be inside a None-guard (same as in-loop site)
    # Verify that 'if final_global_prototype is not None:' appears near the assignment
    attach_idx = src.find('extra_eval_config_dict["global_prototype"]')
    assert attach_idx >= 0, "Pitfall-4: attachment site not found (previous assertion should have caught this)"
    # Look back 300 chars for the None-guard
    nearby = src[max(0, attach_idx - 300):attach_idx + 200]
    assert "final_global_prototype is not None" in nearby, (
        "Pitfall-4: extra-eval ConfigRecord prototype attachment lacks None-guard. "
        "Must mirror the in-loop None check at server_app.py:814."
    )
    # The comment explaining Pitfall-4 closure should be present (regression documentation)
    assert "PITFALL 4" in src or "Pitfall 4" in src, (
        "Pitfall-4: No Pitfall 4 comment found — future maintainers won't know why this matters"
    )


def _server_app_src() -> str:
    """Read this module's server_app.py source for static wiring assertions."""
    src_path = Path(__file__).resolve().parents[1] / "federated_adaptive_personalized_cf" / "server_app.py"
    return src_path.read_text(encoding="utf-8")


def test_thesis_label_in_manifest() -> None:
    """Phase 7 D-22 + Pitfall 2: server_app reads thesis-run-label / ablation-dimension /
    ablation-value from context.run_config and mutates the manifest via dataclass_replace
    BEFORE embed_manifest_in_result so results.json's _manifest carries the thesis-tagging fields.

    This is a STATIC source-level wiring test — it does NOT spawn a Flower run.
    The integration loop is exercised by Plan 05's smoke-run gate.
    """
    src = _server_app_src()
    # The 3 thesis kwargs MUST appear inside the dataclass_replace(manifest, ...) call.
    assert 'thesis_run_label=str(context.run_config.get("thesis-run-label"' in src, (
        "Phase 7 D-22: server_app must read thesis-run-label from run_config and pass to dataclass_replace"
    )
    assert 'ablation_dimension=str(context.run_config.get("ablation-dimension"' in src, (
        "Phase 7 D-22: server_app must read ablation-dimension from run_config and pass to dataclass_replace"
    )
    assert 'ablation_value=str(context.run_config.get("ablation-value"' in src, (
        "Phase 7 D-22: server_app must read ablation-value from run_config and pass to dataclass_replace"
    )
    # Phase-6 final_eval_round_index + metrics kwargs must coexist (regression guard for Phase 6).
    assert "final_eval_round_index=final_eval_round_index" in src, (
        "Phase 6 D-07: final_eval_round_index kwarg MUST coexist with Phase 7 thesis kwargs"
    )
    assert 'metrics=results_data["final_metrics"]' in src, (
        "Phase 6 D-07: metrics kwarg MUST coexist with Phase 7 thesis kwargs"
    )
    # Mutation MUST execute BEFORE embed_manifest_in_result.
    idx_thesis_kwarg = src.find('thesis_run_label=str(context.run_config.get')
    idx_embed = src.find("embed_manifest_in_result(manifest, results_data)")
    assert idx_thesis_kwarg != -1, "Could not locate the thesis_run_label kwarg in server_app.py"
    assert idx_embed != -1, "Could not locate embed_manifest_in_result(manifest, results_data) call site"
    assert idx_thesis_kwarg < idx_embed, (
        "Phase 7 D-22 + Pitfall 2 invariant: dataclass_replace(manifest, ...thesis fields...) "
        "MUST execute BEFORE embed_manifest_in_result so the embedded _manifest dict "
        "carries the thesis-tagging fields. Found embed_manifest_in_result first — order is wrong."
    )
    # Pitfall 3 site #1 + #2: BOTH mode tuples include thesis_crossdevice_main.
    assert src.count('"thesis_crossdevice_main"') >= 2, (
        "Phase 7 Pitfall 3: BOTH `mode in (...)` tuples in this server_app must include "
        "'thesis_crossdevice_main' (W&B project gate + results-path gate)"
    )
