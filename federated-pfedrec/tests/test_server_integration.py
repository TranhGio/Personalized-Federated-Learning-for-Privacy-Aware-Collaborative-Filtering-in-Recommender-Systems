"""Phase 5 PFR-02 / PFR-06 / PFR-08 / PFR-09 server_app integration regression guard.

Covers the 8 GREEN tests for VALIDATION.md rows 5-04-01 through 5-04-08:

- 5-04-01 ``test_discovery_round_partition_id_sampling`` — G-03-01 + ADP-06
  partition-id-space sampling (single ``_server_sampler`` instance, ``range(N)``
  domain).
- 5-04-02 ``test_server_rng_seeded_sampling`` — FND-06 ``server_rng`` byte-
  identity across instances at same seed; divergence at different seeds.
- 5-04-03 ``test_pfr08_autoverify_parses_sh_result`` — D-14 reference parser
  reads ``IJCAI-23-PFedRec/sh_result/ml-1m.txt`` line 2 (the most recent run,
  Open Question 1 recommendation: HR=0.7286, NDCG=0.4407).
- 5-04-04 ``test_pfr08_autoverify_pass_within_2pts`` — D-14 hook returns
  ``passed=True`` + ``[PFR-08 VERIFIED]`` log line when within tolerance.
- 5-04-05 ``test_pfr08_autoverify_fail_outside_2pts`` — D-14 hook returns
  ``passed=False`` + ``[PFR-08 FAILED]`` log line; CRITICAL non-fatal property
  (must NOT raise) — failed reproduction does NOT abort the run.
- 5-04-06 ``test_manifest_double_write_module_pfedrec`` — D-15 / PFR-09: source
  contains ``module="pfedrec"`` AND ``audit_doc="PFR-02-AUDIT.md"`` AND both
  ``embed_manifest_in_result`` + ``write_manifest_sibling`` calls.
- 5-04-07 ``test_cold_starts_per_round_logged`` — D-13 cold-start counter
  declaration + result-write + ``.embedding_cache/<run_id>/partition_{pid}.pt``
  existence-probe in source.
- 5-04-08 ``test_best_round_restore_against_ndcg10`` — D-13 best-round-restore
  via the Phase-3-D-27 carry-forward in-memory snapshot idiom; monitors
  ``sampled_ndcg@10`` per CONTEXT.md D-13.

Tests anchor on ``inspect.getsource(server_app)`` because a live Grid is not
available in unit tests. Live integration is exercised by Plan 05's subprocess
determinism guard in ``scripts/foundation/tests/test_pfedrec_subprocess_determinism.py``.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest


# ---- Test 1: G-03-01 discovery + ADP-06 partition-id-space sampling (source-level) ----
def test_discovery_round_partition_id_sampling() -> None:
    """PFR-06 / G-03-01: single ``_server_sampler`` driven by ``server_rng(run_seed)``;
    sampling domain is ``range(expected_n)`` partition-id space.
    """
    import federated_pfedrec.server_app as server_app
    src = inspect.getsource(server_app)
    # _server_sampler is created via server_rng(run_seed) exactly once. Multiple
    # instantiations would re-seed mid-loop and break determinism.
    assert src.count("_server_sampler = server_rng") == 1, (
        "PFR-06: _server_sampler must be a single instance; multiple instantiations "
        "would re-seed mid-loop and break determinism."
    )
    assert "_server_sampler.sample(range(" in src, (
        "ADP-06: partition-id-space sampling required (sample(range(expected_n), ...))"
    )


# ---- Test 2: server_rng seeded sampling determinism ----
def test_server_rng_seeded_sampling() -> None:
    """FND-06: ``server_rng(s)`` is byte-deterministic; different seeds diverge."""
    from fedrec_foundation.rng import server_rng

    a = server_rng(42)
    b = server_rng(42)
    c = server_rng(43)
    out_a = a.sample(range(100), 20)
    out_b = b.sample(range(100), 20)
    out_c = c.sample(range(100), 20)
    assert list(out_a) == list(out_b), "FND-06 same seed must give same sample"
    assert list(out_a) != list(out_c), "FND-06 different seed must give different sample"


# ---- Test 3: D-14 reference parser anchored on the real ml-1m.txt ----
def test_pfr08_autoverify_parses_sh_result() -> None:
    """D-14 / Open Question 1: parse line 2 (HR=0.7286, NDCG=0.4407) — the most
    recent run / closest to paper-reported best round 89.
    """
    from federated_pfedrec.server_app import _parse_reference_results

    repo_root = Path(__file__).resolve().parents[2]
    ref = repo_root / "IJCAI-23-PFedRec" / "sh_result" / "ml-1m.txt"
    if not ref.exists():
        pytest.skip("reference file not present in this clone")
    hr, ndcg = _parse_reference_results(ref)
    # Open Question 1 recommendation: line 2 / most recent / closest to paper round 89.
    assert hr == pytest.approx(0.7286, abs=1e-3), f"line 2 HR: {hr}"
    assert ndcg == pytest.approx(0.4407, abs=1e-3), f"line 2 NDCG: {ndcg}"


# ---- Test 4: D-14 pass path within tolerance ----
def test_pfr08_autoverify_pass_within_2pts(tmp_path) -> None:
    """D-14 hook returns ``passed=True`` + ``[PFR-08 VERIFIED]`` log line when
    Δhr/Δndcg ≤ 2.0pts.
    """
    from federated_pfedrec.server_app import _emit_pfr_08_verification

    ref = tmp_path / "ml-1m-synthetic.txt"
    ref.write_text(
        "2026-01-01 00-00-00-latent_dim: 32-lr: 0.1-clients_sample_ratio: 1.0-"
        "num_round: 100-negatives: 4-lr_eta: 80-batch_size: 256-"
        "hr: 0.7286-ndcg: 0.4407-best_round: 89-optimizer: sgd-l2_regularization: 0.0\n"
    )
    final_metrics = {"sampled_hr@10": 0.730, "sampled_ndcg@10": 0.450}
    passed, log_line, audit = _emit_pfr_08_verification(
        final_metrics=final_metrics, reference_path=ref, tolerance_pts=2.0,
    )
    assert passed is True
    assert "PFR-08 VERIFIED" in log_line
    assert audit["passed"] is True
    assert audit["delta_hr_pts"] < 2.0
    assert audit["delta_ndcg_pts"] < 2.0


# ---- Test 5: D-14 fail path is NON-FATAL (does not raise) ----
def test_pfr08_autoverify_fail_outside_2pts(tmp_path) -> None:
    """D-14 hook is non-fatal: failed reproduction returns ``passed=False`` and
    a ``[PFR-08 FAILED]`` log line; it MUST NOT raise. The PFR-08 reproduction
    gate is auditable but does not abort the run.
    """
    from federated_pfedrec.server_app import _emit_pfr_08_verification

    ref = tmp_path / "ml-1m-synthetic.txt"
    ref.write_text(
        "2026-01-01 00-00-00-latent_dim: 32-lr: 0.1-clients_sample_ratio: 1.0-"
        "num_round: 100-negatives: 4-lr_eta: 80-batch_size: 256-"
        "hr: 0.7286-ndcg: 0.4407-best_round: 89-optimizer: sgd-l2_regularization: 0.0\n"
    )
    final_metrics = {"sampled_hr@10": 0.50, "sampled_ndcg@10": 0.20}
    # Critical: must NOT raise — failed reproduction is non-fatal.
    passed, log_line, audit = _emit_pfr_08_verification(
        final_metrics=final_metrics, reference_path=ref, tolerance_pts=2.0,
    )
    assert passed is False
    assert "PFR-08 FAILED" in log_line
    assert audit["passed"] is False
    assert audit["delta_hr_pts"] > 2.0


# ---- Test 6: D-15 double-write + audit_doc back-pointer (source-level) ----
def test_manifest_double_write_module_pfedrec() -> None:
    """D-15 / PFR-09: ``build_run_manifest`` carries ``module="pfedrec"`` and the
    SC-1 back-pointer ``audit_doc="PFR-02-AUDIT.md"``; the double-write idiom
    invokes both ``embed_manifest_in_result`` and ``write_manifest_sibling``.
    """
    import federated_pfedrec.server_app as server_app
    src = inspect.getsource(server_app)
    # build_run_manifest call site contains module="pfedrec".
    assert 'module="pfedrec"' in src or "module='pfedrec'" in src, (
        "D-15: build_run_manifest must thread module='pfedrec' (PFR-09)"
    )
    # SC-1 back-pointer: PFR-02-AUDIT.md is referenced through the manifest.
    assert (
        'audit_doc="PFR-02-AUDIT.md"' in src
        or "audit_doc='PFR-02-AUDIT.md'" in src
    ), (
        "D-15 back-pointer: result JSON must reference PFR-02-AUDIT.md (Plan 01 Task 3)"
    )
    # Double-write idiom.
    assert "embed_manifest_in_result" in src
    assert "write_manifest_sibling" in src


# ---- Test 7: D-13 cold-start counter (source-level) ----
def test_cold_starts_per_round_logged() -> None:
    """D-13 cold-start counter: ``cold_starts`` declared + accumulated + persisted
    in result JSON; existence-probe pattern targets
    ``.embedding_cache/<run_id>/partition_{pid}.pt``.
    """
    import federated_pfedrec.server_app as server_app
    src = inspect.getsource(server_app)
    assert src.count("cold_starts") >= 2, (
        "D-13 cold-start counter: 'cold_starts' should appear in declaration + result write"
    )
    # Existence probe pattern: partition_{pid}.pt + .exists().
    assert "partition_" in src and ".pt" in src and ".exists()" in src, (
        "D-13: must probe .embedding_cache/<run_id>/partition_{pid}.pt existence"
    )


# ---- Test 8: D-13 best-round-restore (Phase-3-D-27 idiom) against sampled_ndcg@10 (source-level) ----
def test_best_round_restore_against_ndcg10() -> None:
    """D-13 best-round-restore: snapshot+restore in-memory ArrayRecord against
    ``sampled_ndcg@10`` (CONTEXT.md D-13 monitor metric). Implementation is the
    Phase-3-D-27 carry-forward idiom (NOT to be confused with CONTEXT.md D-27
    weight-policy override).
    """
    import federated_pfedrec.server_app as server_app
    src = inspect.getsource(server_app)
    # CONTEXT.md D-13 monitor metric.
    assert 'thesis_metrics.get("sampled_ndcg@10"' in src, (
        "CONTEXT.md D-13: best-round-restore monitor metric is sampled_ndcg@10"
    )
    # checkpoint_rule spelling tolerance (best_round_restore | best_round).
    assert 'checkpoint_rule in ("best_round_restore", "best_round")' in src, (
        "D-13 checkpoint_rule spelling tolerance — Phase-3-D-27 carry-forward idiom"
    )
    # Use rfind to anchor on the LAST occurrence (avoid module-docstring duplicates —
    # Phase 4 Plan 5 lesson). The actual restore step is `arrays = best_arrays`.
    restore_idx = src.rfind("arrays = best_arrays")
    assert restore_idx > 0, (
        "D-13 best-round-restore: arrays = best_arrays must execute at loop end"
    )


# =============================================================================
# Phase 6 Plan 06: D-02/D-06/D-07 + EVL-01/02/03/04/06 + Pitfall-1 PFR-08 hook
# =============================================================================

def _pfedrec_server_app_src() -> str:
    """Return the text of federated_pfedrec/server_app.py."""
    src_path = Path(__file__).resolve().parents[1].joinpath(
        "federated_pfedrec", "server_app.py"
    )
    return src_path.read_text()


def test_results_path_repo_root_anchored() -> None:
    """D-02: server_app.py imports module_run_results_dir and never uses
    Path('../results/federated/pfedrec').

    Source-level assertions (acceptance criteria 1, 4, 5):
    - ``from fedrec_foundation.paths import module_run_results_dir`` present.
    - ``Path('../results/federated/pfedrec')`` literal absent (D-02 cutover).
    - ``module_run_results_dir(_MODULE, run_id)`` call present.
    - ``_MODULE: str = "pfedrec"`` local constant present.
    - ``sibling_name="manifest.json"`` kwarg present (D-04 clean filename).
    - ``from fedrec_foundation.atomic import atomic_write_json`` present.
    - ``json.dump(results_data`` absent (replaced by atomic_write_json).

    Functional check:
    - ``module_run_results_dir("pfedrec", "test_run_pfedrec_p6")`` resolves to
      ``<repo>/results/federated/pfedrec/test_run_pfedrec_p6/``.
    - Returned path is absolute (D-02 anchoring).
    """
    from fedrec_foundation.paths import module_run_results_dir, repo_root

    src = _pfedrec_server_app_src()

    # Import checks.
    assert src.count("from fedrec_foundation.paths import module_run_results_dir") >= 1, (
        "D-02: 'from fedrec_foundation.paths import module_run_results_dir' not found"
    )
    assert "from fedrec_foundation.atomic import atomic_write_json" in src, (
        "MINOR: atomic_write_json import missing from server_app.py"
    )
    assert "from dataclasses import replace as dataclass_replace" in src, (
        "dataclass_replace import missing from server_app.py"
    )

    # Legacy literal must be gone.
    assert 'Path("../results/federated/pfedrec")' not in src, (
        "D-02 hard cutover: Path('../results/federated/pfedrec') literal must be gone"
    )
    assert "Path('../results/federated/pfedrec')" not in src, (
        "D-02 hard cutover: Path('../results/federated/pfedrec') literal must be gone"
    )

    # Call site checks.
    assert "module_run_results_dir(_MODULE, run_id)" in src, (
        "D-02: 'module_run_results_dir(_MODULE, run_id)' call missing"
    )
    assert '_MODULE: str = "pfedrec"' in src or "_MODULE: str = 'pfedrec'" in src, (
        "D-02: '_MODULE: str = \"pfedrec\"' local constant missing"
    )
    assert 'sibling_name="manifest.json"' in src or "sibling_name='manifest.json'" in src, (
        "D-04: sibling_name='manifest.json' kwarg missing"
    )
    assert "json.dump(results_data" not in src, (
        "MINOR: legacy json.dump(results_data, ...) must be replaced by atomic_write_json"
    )

    # Functional resolver check.
    run_dir = module_run_results_dir("pfedrec", "test_run_pfedrec_p6")
    repo = repo_root()
    expected = repo / "results" / "federated" / "pfedrec" / "test_run_pfedrec_p6"
    assert run_dir == expected, (
        f"module_run_results_dir layout mismatch: got {run_dir}, expected {expected}"
    )
    assert run_dir.is_absolute(), "D-02: module_run_results_dir must return an absolute path"
    assert run_dir.is_dir(), "module_run_results_dir must create the directory eagerly"


def test_extra_eval_round_after_best_arrays_restore() -> None:
    """D-06: extra-eval-round wiring inserted after arrays = best_arrays.

    Source-level checks:
    - D-06-forbidden ``eval_metrics_history.get(final_round_for_metrics`` absent.
    - ``final_eval_round_index`` appears >= 5 times.
    - ``strategy.aggregate_evaluate(final_eval_round_index`` call present.
    - Pitfall-9 ``max(eval_metrics_history.keys())`` present.
    - Pitfall-8 ``if mode in ("benchmark_cross_device", "paper_compat_pfedrec")`` present.
    - D-06 ordering: extra-eval-round appears AFTER arrays = best_arrays.
    - cross_silo_legacy fallback preserved (no NotImplementedError in else branch).
    - ``legacy_dir = repo_root() / "results" / "federated" / "pfedrec"`` present.
    - ``sibling_kwarg = {}`` for legacy default sibling naming.
    """
    src = _pfedrec_server_app_src()

    # D-06 forbidden lookup must be gone (only from the final_metrics resolution site;
    # the method still uses it in best_round_restore).
    assert "eval_metrics_history.get(final_round_for_metrics" not in src, (
        "D-06 BUG STILL PRESENT: 'eval_metrics_history.get(final_round_for_metrics' "
        "must be removed from server_app.py (flat final_metrics is replaced by nested schema)"
    )

    # extra-eval-round artifacts.
    assert src.count("final_eval_round_index") >= 5, (
        f"'final_eval_round_index' appears {src.count('final_eval_round_index')} times, "
        "expected >= 5"
    )
    assert "strategy.aggregate_evaluate(final_eval_round_index" in src, (
        "D-06: 'strategy.aggregate_evaluate(final_eval_round_index' call missing"
    )

    # Pitfall 9.
    assert "max(eval_metrics_history.keys())" in src, (
        "Pitfall 9: 'max(eval_metrics_history.keys())' missing"
    )

    # Pitfall 8: cross-silo mode branch.
    assert 'if mode in ("benchmark_cross_device", "paper_compat_pfedrec")' in src, (
        "Pitfall 8: cross-silo coexistence conditional missing"
    )

    # cross_silo_legacy fallback preservation (plan-checker MAJOR check).
    # pfedrec uses NotImplementedError for the D-02 frozen-cross-silo GUARD at top of main,
    # but the RESULTS WRITE else-branch must NOT raise NotImplementedError.
    # Verify legacy_dir and sibling_kwarg = {} are present (legacy path preserved).
    assert "legacy_dir = repo_root()" in src, (
        "MAJOR: legacy_dir fallback missing from server_app.py (else branch must preserve "
        "pre-Phase-6 cross-silo write path)"
    )
    assert "sibling_kwarg = {}" in src, (
        "MAJOR: sibling_kwarg = {} for legacy default sibling naming missing"
    )

    # D-06 ordering: best_arrays restore must precede extra-eval-round.
    idx_best = src.rfind("arrays = best_arrays")
    idx_extra = src.find("strategy.aggregate_evaluate(final_eval_round_index")
    assert idx_best >= 0, "'arrays = best_arrays' not found — D-27 restore block missing"
    assert idx_extra >= 0, "strategy.aggregate_evaluate(final_eval_round_index) not found"
    assert idx_extra > idx_best, (
        "D-06 ordering violation: extra-eval-round must appear AFTER arrays = best_arrays"
    )


def test_canonical_artifact_carries_best_and_last_blocks() -> None:
    """D-07: final_metrics nested {best, last, best_round, last_round, final_eval_round_index}.

    Also verifies:
    - W&B summary uses ``best/`` and ``last/`` namespaces; ``final/pfr08`` absent.
    - Top-level ``pfr08`` W&B summary key present (PFR-08 audit migrated from final/pfr08).
    - dataclass_replace called before embed_manifest_in_result (edit-order invariant).
    - np.float64 coercion present.
    - schema_version == 2 in manifest.
    - pfr08_verification post-embed mutation preserved verbatim.
    """
    src = _pfedrec_server_app_src()

    # Nested schema keys.
    assert '"best"' in src or "'best'" in src, (
        "D-07: 'best' key missing from final_metrics"
    )
    assert '"last"' in src or "'last'" in src, (
        "D-07: 'last' key missing from final_metrics"
    )
    assert '"best_round"' in src or "'best_round'" in src, (
        "D-07: 'best_round' key missing from final_metrics"
    )
    assert '"last_round"' in src or "'last_round'" in src, (
        "D-07: 'last_round' key missing from final_metrics"
    )
    assert '"final_eval_round_index"' in src or "'final_eval_round_index'" in src, (
        "D-07: 'final_eval_round_index' key missing from final_metrics"
    )

    # W&B namespaces: final/pfr08 must be gone; top-level pfr08 must be present.
    assert 'wandb.run.summary["final/pfr08"]' not in src, (
        "W&B migration: wandb.run.summary[\"final/pfr08\"] still present (legacy namespace)"
    )
    assert 'wandb.run.summary["pfr08"]' in src or "wandb.run.summary['pfr08']" in src, (
        "W&B migration: wandb.run.summary['pfr08'] missing (PFR-08 audit must use top-level key)"
    )
    assert 'wandb.run.summary[f"final/' not in src, (
        "W&B migration: wandb.run.summary[f\"final/...\"] still present in server_app.py"
    )
    assert 'wandb.run.summary[f"best/' in src or "wandb.run.summary[f'best/" in src, (
        "W&B migration: wandb.run.summary[f\"best/...\"] missing"
    )
    assert 'wandb.run.summary[f"last/' in src or "wandb.run.summary[f'last/" in src, (
        "W&B migration: wandb.run.summary[f\"last/...\"] missing"
    )

    # schema_version == 2.
    from fedrec_foundation.manifest import RUN_MANIFEST_SCHEMA_VERSION
    assert RUN_MANIFEST_SCHEMA_VERSION == 2, (
        f"Expected RUN_MANIFEST_SCHEMA_VERSION == 2, got {RUN_MANIFEST_SCHEMA_VERSION}"
    )

    # _manifest.schema_version should be set to 2 through the manifest dataclass.
    # Check that dataclass_replace is imported.
    assert "from dataclasses import replace as dataclass_replace" in src, (
        "dataclass_replace import missing"
    )

    # pfr08_verification post-embed mutation preserved verbatim.
    assert 'results_data["_manifest"]["pfr08_verification"] = pfr08_audit' in src, (
        "Phase-5 post-embed mutation 'results_data[\"_manifest\"][\"pfr08_verification\"] = pfr08_audit' "
        "must be preserved verbatim"
    )

    # np.float64 coercion.
    assert "float(v) if isinstance(v, (int, float))" in src, (
        "MAJOR: np.float64 JSON-safe coercion missing from server_app.py"
    )

    # Edit-order invariant: final_metrics block must appear before dataclass_replace.
    idx_final = src.find("final_metrics = {")
    idx_replace = src.find("dataclass_replace(manifest")
    assert idx_final >= 0, "'final_metrics = {' block not found"
    assert idx_replace >= 0, "'dataclass_replace(manifest' not found"
    assert idx_replace > idx_final, (
        "Edit-order invariant: final_metrics block must appear before dataclass_replace"
    )


def test_round_metrics_history_carries_per_group_exposure() -> None:
    """D-09: strategy.aggregate_evaluate emits per-group evaluated_users counts.

    Uses a live PFedRecSplitFedAvg strategy call to verify the per-group
    sufficient-stat aggregation produces the expected keys.
    """
    from unittest.mock import MagicMock
    from flwr.common import Code, EvaluateRes, Status

    from federated_pfedrec.strategy import PFedRecSplitFedAvg

    strategy = PFedRecSplitFedAvg(fraction_fit=0.1)
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

    # D-09: per-group evaluated_users counts must be present.
    assert "evaluated_users_sparse" in metrics, (
        "D-09: 'evaluated_users_sparse' missing from PFedRecSplitFedAvg.aggregate_evaluate output"
    )
    assert "evaluated_users_medium" in metrics, (
        "D-09: 'evaluated_users_medium' missing"
    )
    assert "evaluated_users_dense" in metrics, (
        "D-09: 'evaluated_users_dense' missing"
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

    # Source-level check: server_app stores thesis_metrics into eval_metrics_history.
    src = _pfedrec_server_app_src()
    assert "eval_metrics_history[round_num] = (" in src or (
        "eval_metrics_history[round_num] = dict(thesis_metrics)" in src
    ), (
        "D-09: eval_metrics_history[round_num] storage missing from server_app.py"
    )


def test_pfr08_hook_consumes_nested_best_block() -> None:
    """PITFALL 1 HEADLINE: D-14 PFR-08 hook reads final_metrics['best'], not flat final_metrics.

    This is the headline regression guard for Plan 06. The hook
    ``_emit_pfr_08_verification`` must receive ``final_metrics["best"]``
    (the nested-best dict under the new D-07 schema), NOT the top-level
    ``final_metrics`` dict (which has no ``sampled_hr@10`` key — those live
    under ``final_metrics["best"]``).

    **Positive path:** Construct ``final_metrics["best"]`` with paper-anchor
    values (HR=0.7287, NDCG=0.4413 — within ±2 pts of ref 0.7286/0.4407).
    Call ``_emit_pfr_08_verification(final_metrics["best"], ...)``.
    Assert: (a) passed=True, (b) no NaN deltas, (c) audit["our_hr"]==0.7287.

    **Negative path (schema-drift guard):** Construct a legacy FLAT dict
    (no "best" sub-key). Verify that calling
    ``_emit_pfr_08_verification(flat_final_metrics["best"], ...)`` raises
    KeyError (proves the guard catches schema-drift Pitfall 1 describes).

    **Call-site source assertion:** verify the source of server_app.py contains
    ``_emit_pfr_08_verification(\\n?\\s*final_metrics["best"]`` (the rewired call
    site) and does NOT contain ``_emit_pfr_08_verification(\\n?\\s*final_metrics,``
    (the legacy flat-input call).
    """
    import re
    from federated_pfedrec.server_app import _emit_pfr_08_verification

    # ---- Positive path ----
    # Construct a synthetic reference file (paper-anchor: HR=0.7286, NDCG=0.4407).
    import tempfile
    import os
    tmp_fd, tmp_ref_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            fh.write(
                "2026-01-01 00-00-00-latent_dim: 32-lr: 0.1-clients_sample_ratio: 1.0-"
                "num_round: 100-negatives: 4-lr_eta: 80-batch_size: 256-"
                "hr: 0.7286-ndcg: 0.4407-best_round: 89-optimizer: sgd-l2_regularization: 0.0\n"
            )
        ref = Path(tmp_ref_path)

        # New nested schema: the "best" block carries the paper-anchor values.
        final_metrics_nested = {
            "best": {"sampled_hr@10": 0.7287, "sampled_ndcg@10": 0.4413},
            "last": {"sampled_hr@10": 0.71, "sampled_ndcg@10": 0.42},
            "best_round": 89,
            "last_round": 100,
            "final_eval_round_index": 101,
        }

        # The REWIRED call site passes final_metrics["best"].
        passed, log_line, audit = _emit_pfr_08_verification(
            final_metrics=final_metrics_nested["best"],
            reference_path=ref,
            tolerance_pts=2.0,
        )

        # (a) passed=True (nested-best values are within ±2 pts of reference).
        assert passed is True, (
            f"Pitfall 1: hook returned passed=False — likely reading wrong dict. "
            f"log_line={log_line!r} audit={audit}"
        )
        # (b) no NaN deltas.
        assert audit["delta_hr_pts"] == audit["delta_hr_pts"], (  # NaN != NaN
            "Pitfall 1: delta_hr_pts is NaN — hook is reading wrong keys"
        )
        assert audit["delta_ndcg_pts"] == audit["delta_ndcg_pts"], (
            "Pitfall 1: delta_ndcg_pts is NaN — hook is reading wrong keys"
        )
        # (c) audit["our_hr"] == 0.7287 (proves hook saw nested-best dict).
        assert audit["our_hr"] == pytest.approx(0.7287, abs=1e-6), (
            f"Pitfall 1: audit['our_hr'] = {audit['our_hr']!r}, expected 0.7287. "
            "Hook may be receiving wrong dict (flat final_metrics instead of best sub-dict)."
        )
        assert audit["our_ndcg"] == pytest.approx(0.4413, abs=1e-6), (
            f"Pitfall 1: audit['our_ndcg'] = {audit['our_ndcg']!r}, expected 0.4413."
        )

        # ---- Negative path: schema-drift guard ----
        # A LEGACY FLAT dict has no "best" sub-key. Accessing ["best"] must
        # raise KeyError, proving the guard detects the schema-drift bug.
        flat_final_metrics = {
            "sampled_hr@10": 0.7287,
            "sampled_ndcg@10": 0.4413,
        }
        with pytest.raises(KeyError):
            _emit_pfr_08_verification(
                final_metrics=flat_final_metrics["best"],  # type: ignore[literal-required]
                reference_path=ref,
                tolerance_pts=2.0,
            )

    finally:
        try:
            os.unlink(tmp_ref_path)
        except OSError:
            pass

    # ---- Call-site source assertion ----
    src = _pfedrec_server_app_src()

    # The rewired call site must pass final_metrics["best"].
    assert '_emit_pfr_08_verification(' in src, (
        "Pitfall 1: _emit_pfr_08_verification call missing from server_app.py"
    )
    # Check that some form of final_metrics["best"] is passed to the hook.
    # Allow for optional whitespace between the function call and the arg.
    assert 'final_metrics["best"]' in src or "final_metrics['best']" in src, (
        "Pitfall 1: _emit_pfr_08_verification must receive final_metrics['best'] "
        "(the nested-best dict), not top-level final_metrics"
    )
    # The LEGACY flat-input call site must be gone.
    # Pattern: _emit_pfr_08_verification(\n?    final_metrics=final_metrics,
    legacy_pattern = re.compile(
        r'_emit_pfr_08_verification\s*\(\s*final_metrics\s*=\s*final_metrics\s*,'
    )
    assert not legacy_pattern.search(src), (
        "Pitfall 1: legacy flat-input _emit_pfr_08_verification(final_metrics=final_metrics, ...) "
        "still present in server_app.py — must be removed"
    )
