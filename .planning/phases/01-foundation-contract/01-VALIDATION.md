---
phase: 1
slug: foundation-contract
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-19
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest` (NOT currently installed — Wave 0 gap) |
| **Config file** | `scripts/foundation/pyproject.toml` will declare `[tool.pytest.ini_options] testpaths = ["tests"]` |
| **Quick run command** | `cd scripts/foundation && pytest -x tests/` |
| **Full suite command** | `cd scripts/foundation && pytest tests/ -v` |
| **Estimated runtime** | < 10 s quick, < 30 s full (RNG cross-process subprocess test dominates) |

---

## Sampling Rate

- **After every task commit:** Run `cd scripts/foundation && pytest -x tests/` (quick, fail-fast)
- **After every plan wave:** Run `cd scripts/foundation && pytest tests/ -v` (full suite)
- **Before `/gsd:verify-work`:** Full suite green AND `python -m fedrec_foundation.build` runs clean AND `for mod in federated-*-cf; do (cd $mod && python -c "import fedrec_foundation"); done` succeeds
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

Task IDs will be finalized when `gsd-planner` writes PLAN.md files. The mapping below is the Req-ID → test contract that every plan task must ultimately satisfy.

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| TBD (FND-01-a) | TBD | 1 | FND-01 | unit | `pytest tests/test_mapping.py -x` | ❌ W0 | ⬜ pending |
| TBD (FND-01-b) | TBD | 1 | FND-01 | unit | `pytest tests/test_mapping.py::test_sort_order -x` | ❌ W0 | ⬜ pending |
| TBD (FND-01-c) | TBD | 1 | FND-01 + CR-1 | unit | `pytest tests/test_mapping.py::test_item_mapping_from_ratings_only -x` | ❌ W0 | ⬜ pending |
| TBD (FND-02-a) | TBD | 1 | FND-02 | unit | `pytest tests/test_split.py::test_hash_deterministic -x` | ❌ W0 | ⬜ pending |
| TBD (FND-02-b) | TBD | 1 | FND-02 | unit | `pytest tests/test_split.py::test_timestamp_tiebreak -x` | ❌ W0 | ⬜ pending |
| TBD (FND-02-c) | TBD | 1 | FND-02 + D-04 | unit | `pytest tests/test_split.py::test_split_lock_refuses_overwrite -x` | ❌ W0 | ⬜ pending |
| TBD (FND-02-d) | TBD | 1 | FND-02 + CR-5 | unit | `pytest tests/test_split.py::test_train_only_user_stats -x` | ❌ W0 | ⬜ pending |
| TBD (FND-03-a) | TBD | 1 | FND-03 | unit | `pytest tests/test_exclusion.py::test_includes_test_item -x` | ❌ W0 | ⬜ pending |
| TBD (FND-03-b) | TBD | 1 | FND-03 + D-05 | unit | `pytest tests/test_exclusion.py::test_safe_load -x` | ❌ W0 | ⬜ pending |
| TBD (FND-03-c) | TBD | 1 | FND-03 + IMP-3 | unit | `pytest tests/test_exclusion.py::test_indptr_layout -x` | ❌ W0 | ⬜ pending |
| TBD (FND-04-a) | TBD | 1 | FND-04 | unit | `pytest tests/test_evaluator.py -x` | ❌ W0 | ⬜ pending |
| TBD (FND-05-a) | TBD | 1 | FND-05 | unit | `pytest tests/test_weight_policy.py -x` | ❌ W0 | ⬜ pending |
| TBD (FND-05-b) | TBD | 1 | FND-05 | unit | `pytest tests/test_weight_policy.py::test_unknown_policy_raises -x` | ❌ W0 | ⬜ pending |
| TBD (FND-05-c) | TBD | 1 | FND-05 + CR-4 | unit | `pytest tests/test_weight_policy.py::test_fit_metrics_contract -x` | ❌ W0 | ⬜ pending |
| TBD (FND-06-a) | TBD | 1 | FND-06 + CR-3 | integration | `pytest tests/test_rng.py::test_derive_rng_stable_across_processes -x` | ❌ W0 | ⬜ pending |
| TBD (FND-06-b) | TBD | 1 | FND-06 | unit | `pytest tests/test_rng.py::test_tuple_uniqueness -x` | ❌ W0 | ⬜ pending |
| TBD (FND-06-c) | TBD | 1 | FND-06 + CR-3 | unit | `pytest tests/test_rng.py::test_all_three_rng_factories -x` | ❌ W0 | ⬜ pending |
| TBD (FND-06-d) | TBD | 1 | FND-06 + CR-3 | unit | `pytest tests/test_rng.py::test_torch_generator_reproducible -x` | ❌ W0 | ⬜ pending |
| TBD (FND-06-e) | TBD | 1 | FND-06 | unit | `pytest tests/test_rng.py::test_sample_reproducible -x` | ❌ W0 | ⬜ pending |
| TBD (FND-07-a) | TBD | 1 | FND-07 | unit | `pytest tests/test_manifest.py::test_all_fields_populated -x` | ❌ W0 | ⬜ pending |
| TBD (FND-07-b) | TBD | 1 | FND-07 + D-15 | unit | `pytest tests/test_manifest.py::test_both_writes -x` | ❌ W0 | ⬜ pending |
| TBD (FND-07-c) | TBD | 1 | FND-07 + IMP-2 | unit | `pytest tests/test_manifest.py::test_composite_foundation_hash -x` | ❌ W0 | ⬜ pending |
| TBD (mode-a) | TBD | 1 | D-06..D-11 | unit | `pytest tests/test_mode.py::test_override_logging -x` | ❌ W0 | ⬜ pending |
| TBD (mode-b) | TBD | 1 | D-11 + CR-2 | unit | `pytest tests/test_mode.py::test_assertion_flags -x` | ❌ W0 | ⬜ pending |
| TBD (mode-c) | TBD | 1 | D-06 + CR-2 | integration | `pytest tests/test_launcher.py::test_launcher_sets_num_supernodes -x` | ❌ W0 | ⬜ pending |
| TBD (bundle-a) | TBD | 1 | D-04 + N-3 | integration | `pytest tests/test_integration.py::test_build_idempotent -x` | ❌ W0 | ⬜ pending |
| TBD (bundle-b) | TBD | 1 | N-3 | integration | `pytest tests/test_integration.py::test_bundle_atomic_publication -x` | ❌ W0 | ⬜ pending |
| TBD (import-a) | TBD | 2 | IMP-1 | smoke | `for mod in federated-*-cf; do (cd $mod && python -c "import fedrec_foundation"); done` | ❌ W0 | ⬜ pending |
| TBD (build-e2e) | TBD | 2 | end-to-end | integration | `pytest tests/test_integration.py::test_build_creates_all_artifacts -x` | ❌ W0 | ⬜ pending |
| TBD (empirical-a) | TBD | 2 | Codex anchor | integration | `pytest tests/test_integration.py::test_ml1m_counts_6040_3706 -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Wave 0 is the test-infrastructure bootstrapping phase. All plan Wave 1+ tasks depend on these artifacts existing.

- [ ] `scripts/foundation/pyproject.toml` — hatchling build, `[project]` with name `fedrec-foundation`, version `0.1.0`, `[tool.pytest.ini_options]` pointing at `tests/`
- [ ] `scripts/foundation/fedrec_foundation/__init__.py` — `__version__ = "0.1.0"` only
- [ ] `scripts/foundation/tests/__init__.py` — empty
- [ ] `scripts/foundation/tests/conftest.py` — shared fixtures (small synthetic `ratings_df`, `movies_df`; temp-dir fixtures for file-IO tests; `monkeypatch PYTHONHASHSEED` fixture)
- [ ] `scripts/foundation/tests/test_mapping.py` — FND-01 + CR-1 behaviors (stub tests)
- [ ] `scripts/foundation/tests/test_split.py` — FND-02 + CR-5 + D-04 behaviors
- [ ] `scripts/foundation/tests/test_exclusion.py` — FND-03 + IMP-3 behaviors
- [ ] `scripts/foundation/tests/test_evaluator.py` — FND-04 behaviors
- [ ] `scripts/foundation/tests/test_weight_policy.py` — FND-05 + CR-4 behaviors
- [ ] `scripts/foundation/tests/test_rng.py` — FND-06 + CR-3 behaviors (incl. cross-process subprocess test with varying `PYTHONHASHSEED`)
- [ ] `scripts/foundation/tests/test_mode.py` — D-06..D-11 + CR-2 behaviors
- [ ] `scripts/foundation/tests/test_manifest.py` — FND-07 + IMP-2 behaviors
- [ ] `scripts/foundation/tests/test_launcher.py` — launcher determinism / num-supernodes enforcement (CR-2)
- [ ] `scripts/foundation/tests/test_integration.py` — end-to-end build + idempotence + empirical ML-1M anchors
- [ ] Framework install: `pip install pytest` into the project's active env; add `pytest = ">=7.0"` to `scripts/foundation/pyproject.toml` dev dependencies
- [ ] Dev-env install order doc: `docs/setup.md` or equivalent — `pip install -e scripts/foundation/ && for m in federated-*-cf; do pip install -e "$m"; done` (install foundation BEFORE the four modules)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `data/derived/*` files commit cleanly to git without LFS | D-01 | git tree inspection is a one-time check; automating it for every run is overkill | After first build: `git add data/derived/ && git status --short && git diff --cached --stat` — confirm `mapping.json` < 200 KB, `split_manifest.json` < 500 KB, `exclusion_items.npz` < 3 MB, each file tracked by git (no LFS marker) |
| `pyproject.toml` local-path dependency syntax resolves against current Flower template (IMP-1 open question) | IMP-1 | Exact PEP 621 / PEP 660 syntax compatibility varies with Flower's hatchling version pins; must be verified on real `pip install -e .` | In a clean virtualenv: `pip install -e scripts/foundation/ && pip install -e federated-baseline-cf/ && python -c "import fedrec_foundation; from federated_baseline_cf import dataset; print(fedrec_foundation.__version__)"` — confirm both imports succeed |
| W&B logs carry `_manifest` without duplication of fields | FND-07 + D-15 | Requires live W&B integration; flaky in unit tests | After a single round run in each module: open W&B UI, verify the run's Config tab shows the manifest fields under `_manifest.*` namespace and NOT duplicated in the top-level Config |

---

## Codex-Added Validation Requirements (supplementary)

These are integrated into the per-task map above but called out explicitly for Nyquist coverage:

1. **Empirical ML-1M anchor** (`test_ml1m_counts_6040_3706`): built mapping on this repo's `data/ml-1m/` yields exactly 6040 users and 3706 items.
2. **Cross-process RNG determinism** (`test_derive_rng_stable_across_processes`): two subprocess runs with different `PYTHONHASHSEED` values produce byte-identical RNG output streams.
3. **Four-layer seeded reproducibility** (`test_all_three_rng_factories` + `test_torch_generator_reproducible`): same `run_seed` reproduces Python client selection, NumPy eval negatives, Torch model init, and DataLoader iteration order as four separate assertions.
4. **Train-only user stats** (`test_train_only_user_stats`): for any user whose LOO test item is X, `train_user_stats[user]` treats the user as if interaction with X never happened; `full_user_stats` (if present at all) lives in a separately-keyed section with a different semantic label.
5. **Composite foundation fingerprint** (`test_composite_foundation_hash`): `foundation_contract_sha256` changes if ANY of {mapping.json bytes, split_manifest.json bytes, exclusion_items.npz bytes} is mutated by a single byte; three separate one-byte-mutation assertions.
6. **Launcher / app-level mode agreement** (`test_launcher_sets_num_supernodes` + `test_assertion_flags`): `benchmark_cross_device` launcher invocation passes `num-supernodes=6040` to `flwr run`; `cross_silo_legacy` passes `num-supernodes=5`; mismatched launch (app-mode != launcher-num-supernodes) triggers the startup assertion.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (14 new files enumerated)
- [ ] No watch-mode flags
- [ ] Feedback latency < 30 s
- [ ] `nyquist_compliant: true` set in frontmatter once planner wires actual task IDs

**Approval:** pending
