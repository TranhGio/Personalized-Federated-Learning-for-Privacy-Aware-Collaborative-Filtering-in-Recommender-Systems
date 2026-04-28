---
phase: 5
slug: pfedrec-migration-reproduction
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-28
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `federated-pfedrec/pyproject.toml` (`[project.optional-dependencies] dev = ['pytest>=7.0']` — Wave 0 installs) |
| **Quick run command** | `cd federated-pfedrec && pytest tests/ -x --no-header -q` |
| **Full suite command** | `pytest scripts/foundation/tests/ federated-pfedrec/tests/ -v` |
| **Estimated runtime** | ~30 seconds (excludes `@pytest.mark.slow` subprocess regression) |

---

## Sampling Rate

- **After every task commit:** Run `cd federated-pfedrec && pytest tests/ -x --no-header -q`
- **After every plan wave:** Run `pytest scripts/foundation/tests/ federated-pfedrec/tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green AND `@pytest.mark.slow` subprocess regression must pass at least once
- **Max feedback latency:** 30 seconds (quick), 90 seconds (full + slow gate)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 5-01-01 | 01 | 1 | PFR-02 (D-01 bias-GLOBAL) | unit | `pytest federated-pfedrec/tests/test_strategy.py::test_global_param_keys_includes_bias -x` | ❌ W0 | ⬜ pending |
| 5-01-02 | 01 | 1 | PFR-02 (D-01 bias-GLOBAL) | unit | `pytest federated-pfedrec/tests/test_strategy.py::test_local_param_keys_excludes_bias -x` | ❌ W0 | ⬜ pending |
| 5-01-03 | 01 | 1 | PFR-02 / D-12 | unit | `pytest federated-pfedrec/tests/test_strategy.py::test_strategy_class_renamed_pfedrecsplitfedavg -x` | ❌ W0 | ⬜ pending |
| 5-01-04 | 01 | 1 | D-21 / PFR-03 | unit | `pytest federated-pfedrec/tests/test_pfedrec_mlp.py::test_set_local_parameters_strict_true_hard_fails -x` | ❌ W0 | ⬜ pending |
| 5-01-05 | 01 | 1 | D-20 / D-01 | unit | `pytest federated-pfedrec/tests/test_pfedrec_mlp.py::test_local_params_tuple_only_affine_weight -x` | ❌ W0 | ⬜ pending |
| 5-01-06 | 01 | 1 | PFR-02 sufficient-stat | unit | `pytest federated-pfedrec/tests/test_strategy.py::test_aggregate_evaluate_sufficient_stat_uniform -x` | ❌ W0 | ⬜ pending |
| 5-02-01 | 02 | 1 | PFR-01 | unit | `pytest federated-pfedrec/tests/test_pyproject.py::test_num_supernodes_6040 -x` | ❌ W0 | ⬜ pending |
| 5-02-02 | 02 | 1 | PFR-01 | unit | `pytest federated-pfedrec/tests/test_pyproject.py::test_partition_mode_natural -x` | ❌ W0 | ⬜ pending |
| 5-02-03 | 02 | 1 | D-09 / D-02 NotImpl | unit | `pytest federated-pfedrec/tests/test_dataset.py::test_load_partition_data_raises_on_non_natural -x` | ❌ W0 | ⬜ pending |
| 5-02-04 | 02 | 1 | D-09 / D-02 NotImpl | unit | `pytest federated-pfedrec/tests/test_dataset.py::test_load_full_data_raises_on_non_natural -x` | ❌ W0 | ⬜ pending |
| 5-02-05 | 02 | 1 | D-25 (foundation) | unit | `pytest scripts/foundation/tests/test_mode.py::test_paper_compat_pfedrec_weight_policy_uniform -x` | ❌ W0 | ⬜ pending |
| 5-03-01 | 03 | 2 | PFR-05 | unit | `pytest federated-pfedrec/tests/test_client_app.py::test_benchmark_one_user_per_client_assert -x` | ❌ W0 | ⬜ pending |
| 5-03-02 | 03 | 2 | PFR-04 / FND-03 | unit | `pytest federated-pfedrec/tests/test_task.py::test_train_negs_exclude_held_out_test_positive -x` | ❌ W0 | ⬜ pending |
| 5-03-03 | 03 | 2 | PFR-07 / D-02 | unit | `pytest federated-pfedrec/tests/test_task.py::test_train_negs_resampled_every_round -x` | ❌ W0 | ⬜ pending |
| 5-03-04 | 03 | 2 | PFR-06 / FND-06 | unit | `pytest federated-pfedrec/tests/test_task.py::test_eval_neg_rng_factory_used -x` | ❌ W0 | ⬜ pending |
| 5-03-05 | 03 | 2 | D-04 | unit | `pytest federated-pfedrec/tests/test_task.py::test_eval_bce_over_positives_plus_99_negs -x` | ❌ W0 | ⬜ pending |
| 5-03-06 | 03 | 2 | D-16 / D-17 / PFR-03 | unit | `pytest federated-pfedrec/tests/test_cache.py::test_partition_pid_pt_layout -x` | ❌ W0 | ⬜ pending |
| 5-03-07 | 03 | 2 | D-17 schema_v3 | unit | `pytest federated-pfedrec/tests/test_cache.py::test_manifest_schema_v3_fields -x` | ❌ W0 | ⬜ pending |
| 5-03-08 | 03 | 2 | D-17 sentinel | unit | `pytest federated-pfedrec/tests/test_cache.py::test_bias_classification_sentinel_global -x` | ❌ W0 | ⬜ pending |
| 5-03-09 | 03 | 2 | D-22 cold round | unit | `pytest federated-pfedrec/tests/test_client_app.py::test_cold_round_probe_then_load -x` | ❌ W0 | ⬜ pending |
| 5-03-10 | 03 | 2 | D-19 | unit | `pytest federated-pfedrec/tests/test_pfedrec_mlp.py::test_kaiming_default_init_paper_faithful -x` | ❌ W0 | ⬜ pending |
| 5-04-01 | 04 | 3 | PFR-06 / G-03-01 | unit | `pytest federated-pfedrec/tests/test_server_integration.py::test_discovery_round_partition_id_sampling -x` | ❌ W0 | ⬜ pending |
| 5-04-02 | 04 | 3 | PFR-06 seeded | unit | `pytest federated-pfedrec/tests/test_server_integration.py::test_server_rng_seeded_sampling -x` | ❌ W0 | ⬜ pending |
| 5-04-03 | 04 | 3 | D-14 | unit | `pytest federated-pfedrec/tests/test_server_integration.py::test_pfr08_autoverify_parses_sh_result -x` | ❌ W0 | ⬜ pending |
| 5-04-04 | 04 | 3 | D-14 | unit | `pytest federated-pfedrec/tests/test_server_integration.py::test_pfr08_autoverify_pass_within_2pts -x` | ❌ W0 | ⬜ pending |
| 5-04-05 | 04 | 3 | D-14 | unit | `pytest federated-pfedrec/tests/test_server_integration.py::test_pfr08_autoverify_fail_outside_2pts -x` | ❌ W0 | ⬜ pending |
| 5-04-06 | 04 | 3 | D-15 / PFR-09 | unit | `pytest federated-pfedrec/tests/test_server_integration.py::test_manifest_double_write_module_pfedrec -x` | ❌ W0 | ⬜ pending |
| 5-04-07 | 04 | 3 | D-13 cold counter | unit | `pytest federated-pfedrec/tests/test_server_integration.py::test_cold_starts_per_round_logged -x` | ❌ W0 | ⬜ pending |
| 5-04-08 | 04 | 3 | D-27 best-round | unit | `pytest federated-pfedrec/tests/test_server_integration.py::test_best_round_restore_against_ndcg10 -x` | ❌ W0 | ⬜ pending |
| 5-05-01 | 05 | 3 | PFR-06 determinism | integration (slow) | `pytest scripts/foundation/tests/test_pfedrec_subprocess_determinism.py -m slow -v` | ❌ W0 | ⬜ pending |
| 5-05-02 | 05 | 3 | D-16 / D-17 byte-identity | integration (slow) | `pytest scripts/foundation/tests/test_pfedrec_subprocess_determinism.py::test_partition_pt_byte_identical -m slow -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `federated-pfedrec/tests/test_strategy.py` — strategy class rename + GLOBAL/LOCAL frozenset coverage (PFR-02 D-01, D-12)
- [ ] `federated-pfedrec/tests/test_pfedrec_mlp.py` — model shape (D-20), strict-load (D-21), Kaiming init (D-19)
- [ ] `federated-pfedrec/tests/test_pyproject.py` — cross-device defaults regression (PFR-01)
- [ ] `federated-pfedrec/tests/test_dataset.py` — D-09 NotImplementedError at both entry points
- [ ] `federated-pfedrec/tests/test_task.py` — exclusion threading (PFR-04), per-round neg resampling (PFR-07/D-02), eval BCE scope (D-04), FND-06 RNG wiring
- [ ] `federated-pfedrec/tests/test_cache.py` — partition_{pid}.pt layout (D-16), manifest_v3 fields (D-17), bias_classification sentinel (D-17)
- [ ] `federated-pfedrec/tests/test_client_app.py` — one-user assertion (PFR-05), cold-round probe (D-22)
- [ ] `federated-pfedrec/tests/test_server_integration.py` — discovery round + partition-id sampling (PFR-06/G-03-01), seeded sampling (PFR-06), D-14 PFR-08 auto-verify (3 tests: parse, pass, fail), D-15 manifest double-write (PFR-09), D-13 cold-start counter, D-27 best-round restore
- [ ] `federated-pfedrec/tests/conftest.py` — shared fixtures (foundation bundle, run_seed=42, tmp_path-redirected `_CACHE_BASE_DIR`)
- [ ] `scripts/foundation/tests/test_mode.py` — extend with `test_paper_compat_pfedrec_weight_policy_uniform` (D-25 regression guard)
- [ ] `scripts/foundation/tests/test_pfedrec_subprocess_determinism.py` — new file; mirrors Phase 3/4 subprocess byte-identity guard adapted for schema_v3 cache + selected_clients_per_round
- [ ] `federated-pfedrec/pyproject.toml` add `[project.optional-dependencies] dev = ['pytest>=7.0']` (mirror Phase 3 Plan 02 / Phase 4 Plan 02)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| End-to-end PFR-08 reproduction within ±2 points | PFR-08 | Full 100-round federated run with 6040 supernodes; ~3 hours wallclock on RTX 5090; not feasible as a unit test | `cd federated-pfedrec && python ../scripts/run.py pfedrec paper_compat_pfedrec`. Watch stdout for `[PFR-08 VERIFIED]` (D-14 auto-verify) at run end. Cross-check `results/federated/<run_id>_results.json` `final_metrics.sampled_hr@10` and `sampled_ndcg@10` against `IJCAI-23-PFedRec/sh_result/ml-1m.txt` line 2 (HR=0.7286, NDCG=0.4407). |
| W&B run lands in `federated-cf-cross-device` project | D-10 | Requires live W&B credentials; not run in CI | `wandb login`; trigger a 3-round smoke run (`flwr run . --run-config "num-server-rounds=3"`); confirm the run appears under the W&B `federated-cf-cross-device` project alongside Phase 2/3/4 cross-device runs. |
| `[MODE OVERRIDE]` log line on CLI override (D-11) | D-11 | Requires live `flwr run` invocation | `flwr run . --run-config "lr=0.05"`; grep stdout for `[MODE OVERRIDE] lr: mode=paper_compat_pfedrec default=0.1 user-override=0.05`. |
| `clean_cache.py --keep N` works on schema_v3 caches | D-23 | Requires multiple historic `.embedding_cache/{run_id}/` dirs to exercise the keep-N policy | After 3+ runs accumulate caches: `python scripts/clean_cache.py --keep 1`; verify only the newest run's cache remains. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s (quick) / 90s (full + slow gate)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
