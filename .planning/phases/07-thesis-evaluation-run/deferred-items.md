# Phase 7 Deferred Items

Tracks out-of-scope discoveries surfaced during Phase 7 plan execution.

## Found during Plan 07-01 (foundation extensions)

### Slow subprocess determinism tests fail under live `flwr run`
- **Files:** `scripts/foundation/tests/test_adaptive_determinism.py::test_adaptive_determinism_subprocess_byte_identical`, `scripts/foundation/tests/test_personalized_determinism.py::test_personalized_determinism_subprocess_byte_identical`
- **Symptom:** `AssertionError: No result JSON found after launcher run_id=adp_det_a` (and `psn_det_a`).
- **Root cause:** The `@pytest.mark.slow` test invokes `scripts/run.py` end-to-end, which forks `flwr run` and expects a `results.json` to materialize. The simulation never produces the artifact in this environment (likely OOM, GPU-resource contention, or a config-drift in the per-module pyproject defaults predating Phase 7).
- **Why deferred:** Neither failing test references any symbol that Plan 07-01 touched (`thesis_crossdevice_main`, `atomic_write_text`, `RUN_MANIFEST_SCHEMA_VERSION`). The 4 source files (`mode.py`, `manifest.py`, `atomic.py`, `scripts/run.py`) are unrelated to the failure mode. Confirmed via `grep -L thesis_crossdevice_main\|atomic_write_text\|RUN_MANIFEST_SCHEMA_VERSION` over both test files: zero matches.
- **Where it surfaced:** Background full-suite run during Plan 07-01 verification (`pytest scripts/foundation/tests/ -ra` with no `-m "not slow"` filter; ran for ~10m40s).
- **What still passes:** The fast-suite (107 passed, 4 deselected) and the 9 newly-added Plan 07-01 tests are all GREEN; the PFedRec subprocess slow test was skipped cleanly (`pfr08_verification absent on both runs (smoke config too small to trigger the D-14 auto-verify hook)`); the baseline subprocess determinism test passed.
- **Recommended next action:** Re-run with `FEDREC_SKIP_SLOW=1` removed and inspect adaptive/personalized `results/federated/<module>/` directories for stale lock files / cache pollution, or rebuild the foundation bundle and clean `.embedding_cache/`. Track as a Phase 7 Plan 02+ smoke-run gate when the per-server-app `thesis_crossdevice_main` mode tuples land — the integration tests in Plan 02 should catch any genuine regression upstream of the slow guards.

## Found during Plan 07-02 (server_app + manifest wiring)

### Slow `test_selected_partitions_byte_identical_across_subprocess_reruns` fails in baseline
- **File:** `federated-baseline-cf/tests/test_server_integration.py::test_selected_partitions_byte_identical_across_subprocess_reruns`
- **Symptom:** `AssertionError: No result JSON found after launcher run_id=...` when running `pytest tests/ -ra -q` (without `-m "not slow"` filter).
- **Root cause:** Same family as the Plan 07-01 deferred adaptive/personalized subprocess slow tests. The `@pytest.mark.slow` test invokes `scripts/run.py` end-to-end and expects a `results.json` to materialize from a live `flwr run`; the simulation does not produce the artifact in this environment.
- **Why deferred:** Identical scope-out reasoning as Plan 07-01 — the test does NOT reference any symbol Plan 07-02 touched (`thesis_crossdevice_main`, `thesis_run_label`, `ablation_dimension`, `ablation_value`, the manifest-mutation patch, or the pyproject thesis keys). Confirmed via `grep -L thesis_crossdevice_main\|thesis_run_label\|ablation_dimension` over the failing test file: zero matches.
- **What still passes:** All 4 module fast suites GREEN under `-m "not slow"` (baseline 27, personalized 39, adaptive 74, pfedrec 42 = 182 passed). 4/4 new `test_thesis_label_in_manifest` tests GREEN. 3/3 stale `schema_version=2` literal regressions auto-fixed (Rule 1) and now GREEN.
- **Recommended next action:** Plan 07-05 smoke-run gate (`python scripts/run.py adaptive thesis_crossdevice_main --run-config "thesis-run-label=main run-seed=42 num-server-rounds=2 fraction-train=0.001 wandb-enabled=false"`) should produce a `results.json` end-to-end; if it does, the slow subprocess tests should also pass. Phase 7 Plan 02 does not unblock or fix the `flwr run` simulation pipeline.
