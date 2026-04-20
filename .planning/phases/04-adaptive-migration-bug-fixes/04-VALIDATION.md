---
phase: 04
slug: adaptive-migration-bug-fixes
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-20
---

# Phase 04 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from RESEARCH.md §Validation Architecture. Planner fills the Per-Task Verification Map during plan creation.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest>=7.0` (dev extra; mirror Phase 2 Plan 02 + Phase 3 Plan 02) |
| **Config file** | none — per-module `[tool.pytest.ini_options]` if needed; Phase 2+3 precedent uses default discovery |
| **Quick run command** | `pytest federated-adaptive-personalized-cf/tests/ -v -x` |
| **Full suite command** | `pytest federated-adaptive-personalized-cf/tests/ scripts/foundation/tests/ -v` |
| **Estimated runtime** | ~8-12 seconds (unit) + ~3-4 minutes if `FEDREC_SKIP_SLOW=0` (subprocess regression) |

---

## Sampling Rate

- **After every task commit:** Run `pytest federated-adaptive-personalized-cf/tests/ -v -x`
- **After every plan wave:** Run `pytest federated-adaptive-personalized-cf/tests/ scripts/foundation/tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green + grep-strip regression pass
- **Max feedback latency:** ~12 seconds (quick run)

---

## Per-Task Verification Map

*Planner populates during plan creation. Each task gets a row.*

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Phase Requirements → Test Map (from RESEARCH.md)

| Req ID | Behavior | Test Type | Test File (Wave 0) |
|--------|----------|-----------|---------------------|
| ADP-01 | pyproject cross-device defaults + schema v2 keys + `[dev]` extra | grep regression | `tests/test_pyproject_shape.py` |
| ADP-02 | `enable_*` BEFORE `load` → cached `_logit_alpha` + `_item_perturbation` restored (not re-init) | unit | `tests/test_dual_model.py::test_enable_before_load_restores_cached_alpha` |
| ADP-03 | `AdaptiveSplitFedAvg.best_prototype` snapshot when `current_ndcg > best_metric`; `_global_prototype = best_prototype` before final broadcast | unit + integration | `tests/test_strategy.py::test_best_prototype_snapshot_at_best_round` + `tests/test_server_integration.py::test_d07_best_prototype_restored_before_final_broadcast` |
| ADP-04 | `assert_benchmark_one_user_per_client` raises on >1 user, passes on =1 | unit | `tests/test_client_assertion.py::test_benchmark_mode_asserts_one_user` |
| ADP-05 | ExclusionTable.for_user merged into user_rated_items; train negatives exclude held-out test positive | unit | `tests/test_task_rng.py::test_train_negatives_exclude_test_positive` |
| ADP-06 | server_rng byte-identical + stdlib random eradicated + schema v2 cache contract | unit + grep + integration | `tests/test_task_rng.py::test_random_seed_calls_stripped` + `tests/test_server_integration.py::test_server_rng_reproducible_per_round_selection` + `tests/test_embedding_cache_manifest_v2.py` |
| ADP-07 | Alpha factory returns values in `[min_alpha, max_alpha]` for edge-case inputs; each HC rule branch fires | unit | `tests/test_alpha_factory.py` |
| ADP-08 | `_manifest.module == "adaptive"` + 4 IMP-2 fingerprints + `best_prototype: List[float]` in result JSON | unit + integration | `tests/test_server_integration.py::test_build_run_manifest_module_adaptive_with_best_prototype` |

---

## Wave 0 Requirements

- [ ] `federated-adaptive-personalized-cf/tests/__init__.py` — package marker
- [ ] `federated-adaptive-personalized-cf/tests/conftest.py` — fake_evaluate_res + fake_client_proxy (copy from Phase 3 `federated-personalized-cf/tests/conftest.py`)
- [ ] `federated-adaptive-personalized-cf/tests/test_strategy.py` — `AdaptiveSplitFedAvg` sufficient-stat + best_prototype snapshot + aggregate_fit override (ADP-03 unit)
- [ ] `federated-adaptive-personalized-cf/tests/test_dual_model.py` — enable-before-load cache restore (ADP-02)
- [ ] `federated-adaptive-personalized-cf/tests/test_task_rng.py` — BSL-05-style cross-file strip + FND-03 exclusion + cold-round α=0 (ADP-05 + ADP-06 RNG half)
- [ ] `federated-adaptive-personalized-cf/tests/test_client_assertion.py` — one-user assert + FitMetricsContract + partition_id echo (ADP-04 + ADP-06 client half)
- [ ] `federated-adaptive-personalized-cf/tests/test_embedding_cache_manifest_v2.py` — schema_version=2 + 12 signature fields + hard-fail delta (ADP-06 cache half)
- [ ] `federated-adaptive-personalized-cf/tests/test_alpha_factory.py` — ADP-07 crafted-input edge cases
- [ ] `federated-adaptive-personalized-cf/tests/test_server_integration.py` — server_rng + best_prototype snapshot + D-07 restored broadcast + cold-start counter + D-15 manifest (ADP-03 integration + ADP-06 server half + ADP-08)
- [ ] `federated-adaptive-personalized-cf/pyproject.toml` `[project.optional-dependencies] dev = ["pytest>=7.0"]` — mirror Phase 2 Plan 02 + Phase 3 Plan 02
- [ ] Framework install documented in `docs/setup.md`: `pip install -e "federated-adaptive-personalized-cf[dev]"`

**Foundation bundle guard:** tests use `pytestmark = pytest.mark.skipif(not (data_derived() / "foundation_index.json").exists(), reason="foundation bundle not committed")` — consistent with Phase 2+3 precedent. Bundle is committed per Phase 1 Plan 02 D-04.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| End-to-end `flwr run .` inside `federated-adaptive-personalized-cf/` completes successfully with `mode=benchmark_cross_device` on ML-1M | ADP-01..08 integration | Requires 6040-client Flower simulation with real ML-1M dataset; ~minutes of runtime; can't be pinned as a unit test without the foundation bundle and ml-1m data being present in CI | Run: `cd federated-adaptive-personalized-cf && flwr run . --run-config "num-server-rounds=1 fraction-train=0.005 local-epochs=1 wandb-enabled=false"` — expect: (a) 6040 supernodes spawn, (b) discovery round completes, (c) main-loop round 1 completes, (d) `results/federated/adaptive/<run_id>_results.json` + `<run_id>-manifest.json` both written, (e) `_manifest.module == "adaptive"`, (f) `_manifest.best_prototype` is a `List[float]` of length `embedding_dim` |
| Alpha clip-floor diagnostics show sparse-user clip-hit-rate in W&B | D-16 | Manual visual inspection of W&B run charts | Run any benchmark_cross_device run ≥5 rounds; check W&B run → `round/alpha_clip_hit_rate` chart exists + non-zero for sparse-heavy rounds |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (10 files + 1 pyproject edit)
- [ ] No watch-mode flags
- [ ] Feedback latency < 12s (quick run)
- [ ] `nyquist_compliant: true` set in frontmatter after planner populates Per-Task Verification Map

**Approval:** pending (awaiting planner output + Per-Task Verification Map fill)
