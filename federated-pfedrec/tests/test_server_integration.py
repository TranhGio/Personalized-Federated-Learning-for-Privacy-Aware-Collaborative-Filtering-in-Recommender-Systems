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
