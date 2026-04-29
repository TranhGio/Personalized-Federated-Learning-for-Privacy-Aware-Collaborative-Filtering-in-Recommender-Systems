---
phase: 6
slug: evaluation-reporting-harness
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-29
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (already installed via `[project.optional-dependencies] dev` in each module's pyproject.toml) |
| **Config file** | `scripts/foundation/pyproject.toml` (foundation tests) + per-module `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `cd scripts/foundation && pytest tests/test_paths.py tests/test_manifest.py -x -q` |
| **Full suite command** | `cd scripts/foundation && pytest tests/ -q && cd ../../federated-baseline-cf && pytest tests/ -q && cd ../federated-personalized-cf && pytest tests/ -q && cd ../federated-adaptive-personalized-cf && pytest tests/ -q && cd ../federated-pfedrec && pytest tests/ -q` |
| **Estimated runtime** | Quick: ~5s · Foundation suite: ~30s · Full (excludes @pytest.mark.slow subprocess guards): ~2 min · With slow gates: ~10–15 min |

---

## Sampling Rate

- **After every task commit:** Run `cd scripts/foundation && pytest tests/ -q -m "not slow"` (foundation unit tests, ~30s)
- **After every plan wave:** Run module-specific `pytest tests/ -q -m "not slow"` for the modules touched in that wave
- **Before `/gsd:verify-work`:** Full suite must be green INCLUDING `@pytest.mark.slow` subprocess determinism guards (`FEDREC_SKIP_SLOW=0 pytest tests/ -q`)
- **Max feedback latency:** 30 seconds for unit tests, 15 minutes for full suite with slow gates

---

## Per-Task Verification Map

> Task IDs follow Phase-N-Plan-Task convention. EVL requirements come from `.planning/REQUIREMENTS.md` §EVL-01..EVL-06. Test surfaces enumerate the 8 RESEARCH.md Validation Architecture items.

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 6-01-01 | 01 (foundation paths) | 1 | EVL-04, D-02 | unit | `pytest scripts/foundation/tests/test_paths.py::test_module_run_results_dir_repo_root_anchored -x` | ❌ W0 | ⬜ pending |
| 6-01-02 | 01 (foundation paths) | 1 | EVL-04, D-01 | unit | `pytest scripts/foundation/tests/test_paths.py::test_module_run_results_dir_layout -x` | ❌ W0 | ⬜ pending |
| 6-01-03 | 01 (foundation paths) | 1 | EVL-04 | unit | `pytest scripts/foundation/tests/test_paths.py::test_module_run_results_dir_whitelist -x` | ❌ W0 | ⬜ pending |
| 6-02-01 | 02 (manifest schema v2) | 1 | EVL-01, EVL-06 | unit | `pytest scripts/foundation/tests/test_manifest.py::test_run_manifest_schema_version_2 -x` | ❌ W0 | ⬜ pending |
| 6-02-02 | 02 (manifest schema v2) | 1 | EVL-06 | unit | `pytest scripts/foundation/tests/test_manifest.py::test_run_manifest_backward_compat_v1 -x` | ❌ W0 | ⬜ pending |
| 6-02-03 | 02 (manifest schema v2) | 1 | EVL-01 | unit | `pytest scripts/foundation/tests/test_manifest.py::test_run_manifest_carries_final_eval_round_index -x` | ❌ W0 | ⬜ pending |
| 6-03-01 | 03 (baseline server_app) | 2 | EVL-01, EVL-04, D-02, D-06, D-07 | integration | `pytest federated-baseline-cf/tests/test_server_integration.py::test_results_path_repo_root_anchored -x` | ❌ W0 | ⬜ pending |
| 6-03-02 | 03 (baseline server_app) | 2 | EVL-01, D-06 | integration | `pytest federated-baseline-cf/tests/test_server_integration.py::test_extra_eval_round_after_best_arrays_restore -x` | ❌ W0 | ⬜ pending |
| 6-03-03 | 03 (baseline server_app) | 2 | EVL-01, EVL-06, D-07 | integration | `pytest federated-baseline-cf/tests/test_server_integration.py::test_canonical_artifact_carries_best_and_last_blocks -x` | ❌ W0 | ⬜ pending |
| 6-03-04 | 03 (baseline server_app) | 2 | EVL-04 (folded todo) | regression | `pytest federated-baseline-cf/tests/test_server_integration.py::test_selected_partitions_byte_identical_across_subprocess_reruns -x -m slow` | ✅ (existing — re-enabled) | ⬜ pending |
| 6-04-01 | 04 (personalized server_app) | 2 | EVL-01, EVL-04, D-02, D-06, D-07 | integration | `pytest federated-personalized-cf/tests/test_server_integration.py -k "results_path or extra_eval or best_last_blocks" -x` | ❌ W0 | ⬜ pending |
| 6-04-02 | 04 (personalized server_app) | 2 | EVL-04 (path migration) | regression | `pytest scripts/foundation/tests/test_personalized_subprocess_determinism.py -x -m slow` | ✅ (existing — path probe update) | ⬜ pending |
| 6-05-01 | 05 (adaptive server_app) | 2 | EVL-01, EVL-04, D-02, D-06, D-07, prototype restore | integration | `pytest federated-adaptive-personalized-cf/tests/test_server_integration.py -k "results_path or extra_eval or best_last_blocks or prototype_attached" -x` | ❌ W0 | ⬜ pending |
| 6-05-02 | 05 (adaptive server_app) | 2 | EVL-01 (Pitfall 4 — prototype must be on broadcast eval ConfigRecord) | integration | `pytest federated-adaptive-personalized-cf/tests/test_server_integration.py::test_extra_eval_broadcasts_best_prototype -x` | ❌ W0 | ⬜ pending |
| 6-05-03 | 05 (adaptive server_app) | 2 | EVL-04 (path migration) | regression | `pytest scripts/foundation/tests/test_adaptive_subprocess_determinism.py -x -m slow` | ✅ (existing — path probe update) | ⬜ pending |
| 6-06-01 | 06 (pfedrec server_app + PFR-08 hook) | 3 | EVL-01, EVL-04, D-02, D-06, D-07, D-14 hook | integration | `pytest federated-pfedrec/tests/test_server_integration.py -k "results_path or extra_eval or pfr08_consumes_best" -x` | ❌ W0 | ⬜ pending |
| 6-06-02 | 06 (pfedrec server_app) | 3 | EVL-01 (Pitfall 1 — PFR-08 must read final_metrics["best"]) | integration | `pytest federated-pfedrec/tests/test_server_integration.py::test_pfr08_hook_consumes_nested_best_block -x` | ❌ W0 | ⬜ pending |
| 6-06-03 | 06 (pfedrec server_app) | 3 | EVL-04 (path migration) | regression | `pytest scripts/foundation/tests/test_pfedrec_subprocess_determinism.py -x -m slow` | ✅ (existing — path probe update) | ⬜ pending |
| 6-07-01 | 07 (W&B summary keys + sweep migration) | 3 | EVL-05, EVL-06, Pitfall 7 | unit + grep | `pytest federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py::test_summary_keys_use_best_last_namespace -x && grep -q "name: best/sampled_ndcg@10" federated-adaptive-personalized-cf/sweep.yaml` | ❌ W0 | ⬜ pending |
| 6-07-02 | 07 (per-round exposure history) | 3 | EVL-03, D-09 | integration | `pytest federated-baseline-cf/tests/test_server_integration.py::test_round_metrics_history_carries_per_group_exposure -x` (one assertion per module) | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

> NEW test files / fixtures that must exist before Phase 6 execution begins. Creating them is itself task 6-00-01.

- [ ] `scripts/foundation/tests/test_paths.py` — NEW. Tests for `module_run_results_dir(module, run_id)` helper. Covers D-01 layout, D-02 repo-root anchoring, whitelist validation. (~3 tests, ~30 LOC)
- [ ] `scripts/foundation/tests/test_manifest.py` — EXTENDED. Add tests for `RunManifest` schema v2 (`final_eval_round_index`, `metrics` fields), backward-compat with v1 fixtures. (~3 new tests added to existing file)
- [ ] `federated-baseline-cf/tests/test_server_integration.py` — EXTENDED. Add ~4 NEW assertions (path, extra-eval, best/last blocks, per-group exposure history). Re-enable the slow `test_selected_partitions_byte_identical_across_subprocess_reruns` after D-02 lands.
- [ ] `federated-personalized-cf/tests/test_server_integration.py` — EXTENDED. ~4 NEW assertions (same shape as baseline).
- [ ] `federated-adaptive-personalized-cf/tests/test_server_integration.py` — EXTENDED. ~5 NEW assertions (baseline shape + best_prototype broadcast attached to extra-eval ConfigRecord per Pitfall 4).
- [ ] `federated-pfedrec/tests/test_server_integration.py` — EXTENDED. ~4 NEW assertions (baseline shape + PFR-08 hook reads `final_metrics["best"]` per Pitfall 1).
- [ ] `federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py` — NEW. Tests for `best/`, `last/` namespaced summary keys + sweep.yaml `metric.name` migration. (~2 tests)
- [ ] Update path probes in 4 existing `scripts/foundation/tests/test_*_subprocess_determinism.py` files: `_RESULTS_DIR` glob now reads `<repo_root>/results/federated/<module>/<run_id>/results.json` instead of legacy flat `_*_results.json`. (NO new files; in-place edits to existing tests.)

*Pytest dev dep already declared in all 4 module pyproject.toml files (Phase 2 BSL contract); no new framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| W&B project routing observed in dashboard | EVL-05, D-05 | wandb.init() side-effect not interceptable in unit test without wandb mock infra; covered by existing W&B integration in Phases 2-5 | After Phase 6 lands, run `flwr run federated-baseline-cf/ --run-config "num-server-rounds=2 mode=benchmark_cross_device wandb-enabled=true"` and confirm in W&B UI that the run appears under `<entity>/federated-cf-cross-device` (not legacy `<entity>/federated-baseline-cf`). Repeat for personalized, adaptive, pfedrec. |
| Cross-silo legacy results path coexistence (D-03) | EVL-04 | Filesystem state across runs; integration test asserts no clobber but visual confirmation is the spec | Pre-Phase-6: `ls results/federated/` shows existing flat `*_results.json` files. Post-Phase-6: same files remain unchanged + new `<module>/<run_id>/` dirs appear. `git status` shows only new files, zero deletions/modifications. |
| Best-round restore correctness on a real 100-round PFedRec paper_compat run | EVL-01, PFR-08 | Reproducibility within ±2 points of paper requires the full convergence; unit test only verifies the restore mechanism mechanics, not numerical correctness on real data | Post-Phase-6 run: `python scripts/run.py pfedrec paper_compat_pfedrec --run-config "run-seed=42"`. Verify `results/federated/pfedrec/<run_id>/manifest.json` shows `final_metrics.best.sampled_hr@10` within 0.729±0.02 and `final_metrics.best.sampled_ndcg@10` within 0.441±0.02; `pfr08_verification.passed=true`. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s for unit tests, < 15min for full suite with slow gates
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
