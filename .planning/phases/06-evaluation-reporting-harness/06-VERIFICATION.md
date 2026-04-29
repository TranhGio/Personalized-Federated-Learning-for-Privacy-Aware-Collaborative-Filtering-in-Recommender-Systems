---
phase: 06-evaluation-reporting-harness
verified: 2026-04-29T09:30:00Z
status: human_needed
score: 4/4 success criteria verified (automated checks), 6/6 EVL requirements covered
re_verification: false
human_verification:
  - test: "Run one full cross-device baseline run and inspect W&B dashboard"
    expected: "Run appears under entity/federated-cf-cross-device (not legacy federated-baseline-cf or federated-cf project); results/federated/baseline/<run_id>/results.json + manifest.json appear on disk; legacy flat results/federated/*_results.json files are untouched."
    why_human: "wandb.init() side-effect is not interceptable in unit tests without wandb mock infra. Filesystem coexistence requires a real run."
  - test: "Run full 100-round PFedRec paper_compat run and inspect canonical artifact"
    expected: "results/federated/pfedrec/<run_id>/manifest.json shows final_metrics.best.sampled_hr@10 within 0.729 +/- 0.02 and final_metrics.best.sampled_ndcg@10 within 0.441 +/- 0.02; manifest shows pfr08_verification.passed=true; best_round >= 1; final_eval_round_index >= 2 (extra eval round ran after restore)."
    why_human: "Reproducing within +/-2 points of PFR-08 reference requires the full convergence; unit tests verify restore mechanics and hook wiring, not numerical correctness on real data."
  - test: "Confirm cross-silo legacy result files are untouched after a cross-device run"
    expected: "git status shows only new directories under results/federated/<module>/<run_id>/; pre-Phase-6 flat *_results.json files are unmodified (git shows no diffs to them)."
    why_human: "Filesystem state validation across prior vs new runs; integration test asserts no-clobber at code level but visual git-status confirmation is the spec."
---

# Phase 6: Evaluation & Reporting Harness Verification Report

**Phase Goal:** Every module emits best-round metrics from a restored best-round checkpoint, per-user-group (sparse/medium/dense) HR@10 and NDCG@10 as first-class fields, sampling-exposure support counts, and writes results plus a protocol fingerprint manifest to a cross-device-scoped location and W&B project.
**Verified:** 2026-04-29T09:30:00Z
**Status:** human_needed (all automated checks pass; W&B routing, path coexistence, and PFR-08 numerical reproduction require a live run)
**Re-verification:** No -- initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Best-round restore produces canonical `best_*` from a fresh post-restore eval, not stale history | VERIFIED | D-06 extra-eval-round block wired in all 4 `server_app.py` files; `final_eval_round_index` computed as `actual_rounds + 1`; `best_round_metrics` populated from fresh `strategy.aggregate_evaluate(final_eval_round_index, ...)` call |
| 2 | Per-user-group (sparse/medium/dense) HR@10 and NDCG@10 are first-class fields plus per-group exposure counts | VERIFIED | All 4 `strategy.py` files emit `sampled_hr@10/sparse`, `sampled_ndcg@10/sparse`, `evaluated_users_sparse` (baseline/personalized/adaptive use underscore; pfedrec uses slash) via `_sufficient_stats_to_thesis_metrics`; flow through nested `final_metrics["best"]` |
| 3 | Cross-device results written under `results/federated/<module>/<run_id>/` with manifest; legacy flat paths untouched by cross-device runs | VERIFIED (automated) / HUMAN (coexistence) | `module_run_results_dir(_MODULE, run_id)` called in all 4 `server_app.py` files for `benchmark_cross_device` / `paper_compat_pfedrec` modes; legacy `repo_root() / "results" / "federated" / ...` flat path preserved in else-branch; zero occurrences of forbidden `Path("../results/federated")` across all 4 files |
| 4 | All four modules log to dedicated cross-device W&B project; canonical `best_*` fields, `last_*` as diagnostic only; `final/*` removed | VERIFIED (automated) / HUMAN (W&B UI) | `wandb.run.summary[f"best/{key}"]` and `wandb.run.summary[f"last/{key}"]` in all 4 `server_app.py`; zero `wandb.run.summary[f"final/{key}"]` or `wandb.run.summary["final/..."]` occurrences; all 4 route to `federated-cf-cross-device` project in benchmark modes |

**Score:** 4/4 truths verified (3 fully automated; 1 partially human for live W&B routing and path coexistence)

---

## Per Success-Criterion Verdict (ROADMAP.md SC 1-4)

### SC-1: Best-round restore and canonical artifact

**Verdict: VERIFIED**

Evidence:
- D-06 extra-eval-round block present in all 4 `server_app.py` files (lines baseline:639-698, personalized:789-842, adaptive:974-1037, pfedrec:914-968).
- Each block fires only when `checkpoint_rule in ('best_round_restore', 'best_round')` and `best_round_num > 0`.
- Block broadcasts evaluate to all `partition_to_node_id.values()` nodes (no sampling -- reproducibility over latency per D-06 decision).
- Result becomes `best_round_metrics` (coerced from np.float64 to Python float at assignment).
- Adaptive additionally restores `strategy._global_prototype = strategy.best_prototype` before the broadcast (Pitfall-4 closure: restored prototype attached to every extra-eval ConfigRecord).
- PFedRec PFR-08 hook rewired to `_emit_pfr_08_verification(final_metrics["best"], ...)` (Pitfall-1 closure).
- `final_eval_round_index` carried in manifest via `dataclass_replace(manifest, final_eval_round_index=..., metrics=...)` BEFORE `embed_manifest_in_result` in all 4 modules.
- `RunManifest.final_eval_round_index: int = 0` field present in `scripts/foundation/fedrec_foundation/manifest.py` with sentinel documented.
- Tests: `test_extra_eval_round_after_best_arrays_restore` (baseline), `test_extra_eval_round_replaces_history_lookup` (personalized), `test_extra_eval_round_replaces_forbidden_history_lookup` (adaptive), `test_extra_eval_round_after_best_arrays_restore` (pfedrec) -- all green.

### SC-2: Per-user-group metrics as first-class fields with exposure counts

**Verdict: VERIFIED**

Evidence:
- Baseline strategy (`strategy.py` line 93-101): emits `sampled_hr@10/sparse`, `sampled_ndcg@10/sparse`, `sampled_hr@10/medium`, `sampled_ndcg@10/medium`, `sampled_hr@10/dense`, `sampled_ndcg@10/dense`, `evaluated_users_sparse`, `evaluated_users_medium`, `evaluated_users_dense`.
- PFedRec strategy (`strategy.py` lines 115-121): same per-group keys emitted via `_sufficient_stats_to_thesis_metrics`, with slash-delimiter form (`evaluated_users/sparse` etc.) -- documented deviation from underscore form.
- All per-group metrics flow through `final_metrics["best"]` and `final_metrics["last"]` blocks in all 4 `server_app.py` files.
- W&B logging: `wandb.run.summary[f"best/{key}"]` loop in all 4 modules covers per-group keys from `final_metrics["best"]`.
- D-09 per-round exposure: `eval_metrics_history[round_num]` stores the full `thesis_metrics` dict (including per-group counts) in all 4 modules.
- Tests: `test_round_metrics_history_carries_per_group_exposure` in all 4 modules -- all green; strengthened by Plan 07 to require `{evaluated_users, evaluated_users_sparse, evaluated_users_medium, evaluated_users_dense}` (underscore modules) or `{evaluated_users, evaluated_users/sparse, ...}` (PFedRec slash form).

### SC-3: Cross-device result path isolation and legacy coexistence

**Verdict: VERIFIED (automated) / HUMAN (live coexistence)**

Evidence:
- `module_run_results_dir` helper at `scripts/foundation/fedrec_foundation/paths.py:59-102` enforces `<repo>/results/federated/<module>/<run_id>/` layout.
- `_ALLOWED_MODULES = frozenset({"baseline", "personalized", "adaptive", "pfedrec"})` whitelist raises ValueError on typo.
- All 4 `server_app.py` files: `run_dir = module_run_results_dir(_MODULE, run_id)` for `benchmark_cross_device` / `paper_compat_pfedrec` modes.
- Legacy else-branch in all 4 files: `legacy_dir = repo_root() / "results" / "federated"` (flat layout; not module-relative; no `Path("../results/federated")`).
- `atomic_write_json` replaces `json.dump` at both paths.
- `sibling_name="manifest.json"` used in cross-device branch; default `<run_id>-manifest.json` preserved in legacy branch.
- Zero `Path("../results/federated")` occurrences across all 4 server_app files.
- Tests: `test_results_path_repo_root_anchored` in all 4 modules (source-level assertions); `test_baseline_subprocess_determinism.py`, `test_personalized_determinism.py`, `test_adaptive_determinism.py`, `test_pfedrec_subprocess_determinism.py` path probes updated to `<module>/*/results.json` layout.
- Human: whether legacy flat `results/federated/*_results.json` are untouched after a cross-device run requires a live run.

### SC-4: Dedicated W&B project, `best_*` canonical, `last_*` diagnostic

**Verdict: VERIFIED (automated) / HUMAN (W&B UI)**

Evidence:
- All 4 `server_app.py` files: `default_project = "federated-cf-cross-device"` for benchmark/cross-device modes (baseline:293, personalized:380, adaptive:488, pfedrec:512).
- Per-module override surface preserved: `wandb_project_cfg = str(context.run_config.get("wandb-project", "")).strip()`.
- W&B summary: `wandb.run.summary[f"best/{key}"]` and `wandb.run.summary[f"last/{key}"]` in all 4 `server_app.py`; `final/*` namespace fully removed from all 4.
- `sweep.yaml` line 18: `name: best/sampled_ndcg@10` (Pitfall-7 closure; previously `final/sampled_ndcg@10`).
- Tests: `test_wandb_summary_keys.py` (5 items): `test_sweep_yaml_metric_name_uses_best_namespace` uses `yaml.safe_load` structured parse; `test_summary_keys_use_best_last_namespace` parametrized over 4 server_apps -- all green.
- Human: actual W&B project routing requires confirming in the dashboard that the run appears under `<entity>/federated-cf-cross-device`.

---

## EVL Requirement Coverage Matrix

| Requirement | Description | Code Evidence | Tests | Status |
|-------------|-------------|---------------|-------|--------|
| EVL-01 | Best-round restore: `best_*` from post-restore broadcast | D-06 extra-eval-round in all 4 server_apps; `final_eval_round_index` in manifest via `dataclass_replace`; PFR-08 hook consumes `final_metrics["best"]` | 4x `test_extra_eval_round_*` + 1x `test_pfr08_hook_consumes_nested_best_block` | VERIFIED |
| EVL-02 | Per-user-group NDCG@10 and HR@10 as first-class fields | Per-group keys in all 4 strategy `aggregate_evaluate` outputs; flow to `final_metrics["best"]`; W&B `best/<group-metric>` logging | 4x `test_round_metrics_history_carries_per_group_exposure` | VERIFIED |
| EVL-03 | Per-group sampling-exposure counts per round and in canonical block | `evaluated_users_sparse/medium/dense` (or slash variants for pfedrec) in `_sufficient_stats_to_thesis_metrics`; stored in `eval_metrics_history[round_num]` and in `final_metrics["best"]` | 4x `test_round_metrics_history_carries_per_group_exposure` (strengthened by Plan 07) | VERIFIED |
| EVL-04 | Results under `results/federated/<module>/<run_id>/`; legacy locations untouched | `module_run_results_dir` in paths.py; `_ALLOWED_MODULES` whitelist; all 4 server_apps use it; legacy else-branch preserved; 0 `Path("../results/federated")` occurrences | `test_paths.py` (3 tests, 13 items); 4x `test_results_path_repo_root_anchored`; 4x determinism test path-probe updates | VERIFIED (automated) / HUMAN (coexistence) |
| EVL-05 | Cross-device W&B runs to new dedicated project (`federated-cf-cross-device`) | All 4 server_apps route to `federated-cf-cross-device` in benchmark mode; `wandb-project` override surface preserved | `test_summary_keys_use_best_last_namespace` (parametrized; verifies no `final/*`); sweep.yaml `name: best/...` structured parse | VERIFIED (automated) / HUMAN (W&B UI) |
| EVL-06 | `best_*` canonical; `last_*` diagnostic; `best_round` in filename/manifest | All 4 server_apps: `final_metrics = {"best": ..., "last": ..., "best_round": ..., "last_round": ..., "final_eval_round_index": ...}`; manifest carries `RunManifest.metrics` and `RunManifest.final_eval_round_index`; W&B `best/*` + `last/*`; no `final/*` | `test_run_manifest_schema_version_2`, `test_run_manifest_carries_final_eval_round_index`, `test_write_manifest_sibling_custom_name`; 4x `test_canonical_artifact_carries_best_and_last_blocks` | VERIFIED |

---

## Per-Plan Must-Haves Verdict

### Plan 01: Foundation paths helper

| Must-Have | Status | Evidence |
|-----------|--------|----------|
| `module_run_results_dir(module, run_id)` returns `<repo>/results/federated/<module>/<run_id>/` as absolute Path | VERIFIED | `paths.py:100` -- `out = repo_root() / "results" / "federated" / module / run_id` |
| Helper creates the directory (parents=True, exist_ok=True) | VERIFIED | `paths.py:101` -- `out.mkdir(parents=True, exist_ok=True)` |
| Raises ValueError on module name outside allowed set (Pitfall-6 typo guard) | VERIFIED | `paths.py:95-98` -- whitelist check + ValueError |
| Works under cwd != repo_root (Flower subprocess chdir) | VERIFIED | `test_paths.py::test_module_run_results_dir_repo_root_anchored` uses `monkeypatch.chdir(tmp_path)`; passes |
| Key links: `module_run_results_dir` -> `repo_root()` via `repo_root() / "results" / "federated"` | VERIFIED | `paths.py:100` verbatim |
| Key links: `_ALLOWED_MODULES = frozenset({"baseline", "personalized", "adaptive", "pfedrec"})` | VERIFIED | `paths.py:56` verbatim |

### Plan 02: Manifest schema v2

| Must-Have | Status | Evidence |
|-----------|--------|----------|
| `RUN_MANIFEST_SCHEMA_VERSION == 2` | VERIFIED | `manifest.py:29` -- `RUN_MANIFEST_SCHEMA_VERSION: int = 2` |
| `RunManifest.final_eval_round_index: int = 0` | VERIFIED | `manifest.py:87` -- present with sentinel docstring |
| `RunManifest.metrics: Dict[str, Any] = field(default_factory=dict)` | VERIFIED | `manifest.py:94` -- present with docstring |
| Backward-compat: existing v1 fixtures construct without TypeError | VERIFIED | `test_manifest.py::test_run_manifest_backward_compat_v1` -- passes |
| `write_manifest_sibling` accepts optional `sibling_name` kwarg | VERIFIED | `manifest.py:225` -- `sibling_name: Optional[str] = None` |

### Plan 03: Baseline server_app rewire

| Must-Have | Status | Evidence |
|-----------|--------|----------|
| Baseline writes to `<repo>/results/federated/baseline/<run_id>/` via `module_run_results_dir` | VERIFIED | `server_app.py:46,904` -- import + call with `_MODULE="baseline"` |
| Post-restore extra-eval-round broadcast runs after `arrays = best_arrays` | VERIFIED | `server_app.py:639-698` -- D-06 block |
| `final_metrics` nested `{best, last, best_round, last_round, final_eval_round_index}` | VERIFIED | `server_app.py:783-801` -- nested schema with Pitfall-9 max(keys()) |
| Cross-silo legacy path preserved (D-03/Pitfall-8) | VERIFIED | `server_app.py:909-913` -- else-branch with `repo_root() / "results" / "federated"` |
| W&B uses `best/*` + `last/*`; no `final/*` | VERIFIED | `server_app.py:807-825`; zero `final/*` occurrences |
| Manifest carries `final_eval_round_index` + `metrics` via `dataclass_replace` | VERIFIED | `server_app.py:889-892` -- `dataclass_replace(manifest, final_eval_round_index=..., metrics=...)` BEFORE `embed_manifest_in_result` |

### Plan 04: Personalized server_app rewire

| Must-Have | Status | Evidence |
|-----------|--------|----------|
| Personalized writes to `<repo>/results/federated/personalized/<run_id>/` | VERIFIED | `server_app.py:82,989` -- import + call with `_MODULE="personalized"` |
| D-06-forbidden `eval_metrics_history[best_round_num]` lookup removed | VERIFIED | Plan 04 SUMMARY documents removal; `test_extra_eval_round_replaces_history_lookup` pins this |
| Extra-eval-round result populates `best_round_metrics` | VERIFIED | `server_app.py:831-842` |
| Manifest mutation before embed | VERIFIED | `server_app.py:975-976` -- `dataclass_replace(manifest,` BEFORE `embed_manifest_in_result` |

### Plan 05: Adaptive server_app rewire

| Must-Have | Status | Evidence |
|-----------|--------|----------|
| Adaptive writes to `<repo>/results/federated/adaptive/<run_id>/` | VERIFIED | `server_app.py:88,1293` -- import + call with `_MODULE="adaptive"` |
| Pitfall-4 closure: extra-eval ConfigRecord attaches restored best_prototype | VERIFIED | `server_app.py:960-996` -- `strategy._global_prototype = strategy.best_prototype` then `final_global_prototype = strategy.get_global_prototype()` then `extra_eval_config_dict["global_prototype"] = final_global_prototype.tolist()` |
| Phase-4 best_prototype post-embed mutation preserved | VERIFIED | `server_app.py:1278-` -- preserved verbatim per SUMMARY |
| Nested final_metrics with Pitfall-9 and np.float64 coercion | VERIFIED | `server_app.py:1031-1033` |
| 5 new integration tests including `test_extra_eval_broadcasts_best_prototype` | VERIFIED | `test_server_integration.py:434,477,522,566,605` -- 5 test functions present; all green |

### Plan 06: PFedRec server_app rewire

| Must-Have | Status | Evidence |
|-----------|--------|----------|
| PFedRec writes to `<repo>/results/federated/pfedrec/<run_id>/` | VERIFIED | `server_app.py:111,1157` -- import + call with `_MODULE="pfedrec"` |
| Pitfall-1 closure: PFR-08 hook receives `final_metrics["best"]` | VERIFIED | `server_app.py:1141-1142` -- `_emit_pfr_08_verification(final_metrics["best"], ...)` |
| Extra-eval-round fires after `arrays = best_arrays` | VERIFIED | `server_app.py:914-968` -- D-06 block |
| D-07 nested final_metrics with Pitfall-9 | VERIFIED | `server_app.py:975-990` -- `max(eval_metrics_history.keys())` |
| `test_pfr08_hook_consumes_nested_best_block` present (headline regression guard) | VERIFIED | `test_server_integration.py:502` -- 3-part test (positive, negative KeyError, source assertion) |

### Plan 07: Cross-cutting W&B namespace and D-09 guards

| Must-Have | Status | Evidence |
|-----------|--------|----------|
| `sweep.yaml metric.name: best/sampled_ndcg@10` (Pitfall-7) | VERIFIED | `sweep.yaml:18` -- `name: best/sampled_ndcg@10`; 0 occurrences of `name: final/sampled_ndcg@10` |
| `test_wandb_summary_keys.py` with YAML structured parse (not substring grep) | VERIFIED | `test_wandb_summary_keys.py:32,70` -- 2 test functions present; `yaml.safe_load` used; 5 test items all green |
| D-09 per-round exposure guard strengthened to full 4-key `required_keys` set in all 4 modules | VERIFIED | `test_server_integration.py` in all 4 modules: `test_round_metrics_history_carries_per_group_exposure` passes with `evaluated_users` + 3 per-group keys |

---

## Pitfall Closure Verification

| Pitfall | Description | Closure Evidence |
|---------|-------------|-----------------|
| Pitfall-1 | PFR-08 hook reads flat `final_metrics` (NaN deltas under D-07 nested schema) | `pfedrec/server_app.py:1141-1142` -- `_emit_pfr_08_verification(final_metrics["best"], ...)`; `test_pfr08_hook_consumes_nested_best_block` with positive + KeyError negative paths |
| Pitfall-4 | Adaptive extra-eval broadcast uses stale/zero prototype if best_prototype not attached to ConfigRecord | `adaptive/server_app.py:960-996` -- restore `strategy._global_prototype = strategy.best_prototype` then attach to `extra_eval_config_dict["global_prototype"]`; `test_extra_eval_broadcasts_best_prototype` |
| Pitfall-6 | Module string typo silently writes to wrong results dir (e.g., `"basline"`) | `paths.py:95-98` -- ValueError check against `_ALLOWED_MODULES`; `test_module_run_results_dir_whitelist` with 8 typo variants |
| Pitfall-7 | `sweep.yaml metric.name: final/sampled_ndcg@10` -- W&B agent reads a key that no longer exists post-Plans-03-06 migration | `sweep.yaml:18` -- `name: best/sampled_ndcg@10`; `test_sweep_yaml_metric_name_uses_best_namespace` uses `yaml.safe_load` structured parse |
| Pitfall-8 | Cross-silo mode overwrites legacy flat files with per-run-dir layout | Mode branch in all 4 server_apps: `if mode in ("benchmark_cross_device", "paper_compat_pfedrec")` for new layout; else legacy flat preserved |
| Pitfall-9 | `last_round = actual_rounds` breaks under early stopping (actual_rounds may not equal last eval round) | `max(eval_metrics_history.keys())` in all 4 server_apps (lines baseline:787, personalized:856, adaptive:1048, pfedrec:979) |
| Pitfall-10 | Centralized eval feeds `best_*` block instead of federated eval (centralized eval runs only in baseline) | Plan 03 SUMMARY: centralized eval feeds `final_metrics["last"]` diagnostics ONLY; `best_*` comes exclusively from D-06 federated extra-eval-round |

---

## Required Artifacts Verification

| Artifact | Status | Key Evidence |
|----------|--------|--------------|
| `scripts/foundation/fedrec_foundation/paths.py` | VERIFIED | `module_run_results_dir` at line 59; `_ALLOWED_MODULES` at line 56 |
| `scripts/foundation/fedrec_foundation/manifest.py` | VERIFIED | `RUN_MANIFEST_SCHEMA_VERSION = 2` at line 29; `final_eval_round_index: int = 0` at line 87; `metrics: Dict[str, Any] = field(default_factory=dict)` at line 94; `sibling_name: Optional[str] = None` at line 225 |
| `scripts/foundation/tests/test_paths.py` | VERIFIED | 3 test functions (13 items): `test_module_run_results_dir_repo_root_anchored`, `test_module_run_results_dir_layout` (4 parametrize), `test_module_run_results_dir_whitelist` (8 parametrize) |
| `scripts/foundation/tests/test_manifest.py` | VERIFIED | 5 new tests appended: `test_run_manifest_schema_version_2`, `test_run_manifest_backward_compat_v1`, `test_run_manifest_carries_final_eval_round_index`, `test_write_manifest_sibling_default_filename`, `test_write_manifest_sibling_custom_name` |
| `scripts/foundation/tests/test_baseline_subprocess_determinism.py` | VERIFIED | File exists; `test_selected_partitions_byte_identical_across_subprocess_reruns` present; path probe uses `baseline/*/results.json` per-run-dir layout |
| `federated-baseline-cf/federated_baseline_cf/server_app.py` | VERIFIED | `from fedrec_foundation.paths import ... module_run_results_dir` (line 46); `_MODULE = "baseline"` (line 210); D-06 block (lines 639-698); nested `final_metrics` (lines 783-801); `best/*` + `last/*` W&B (lines 807-825); `dataclass_replace(manifest, ...)` (line 889) |
| `federated-personalized-cf/federated_personalized_cf/server_app.py` | VERIFIED | Same pattern; `_MODULE = "personalized"` (line 329); D-06 extra-eval-round (lines 789-842); nested final_metrics (lines 862-870) |
| `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` | VERIFIED | `_MODULE = "adaptive"` (line 424); D-06 block with Pitfall-4 prototype attach (lines 974-1037); nested final_metrics (lines 1055-1059); `dataclass_replace(manifest, ...)` (line 1269) |
| `federated-pfedrec/federated_pfedrec/server_app.py` | VERIFIED | `_MODULE = "pfedrec"` (line 405); D-06 block (lines 914-968); nested final_metrics (lines 975-990); Pitfall-1 `_emit_pfr_08_verification(final_metrics["best"], ...)` (line 1141); `dataclass_replace(manifest, ...)` (line 1110) |
| `federated-adaptive-personalized-cf/sweep.yaml` | VERIFIED | Line 18: `name: best/sampled_ndcg@10` |
| `federated-adaptive-personalized-cf/tests/test_wandb_summary_keys.py` | VERIFIED | 2 test functions; `yaml.safe_load` structured parse; 5 test items green |

---

## Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| `baseline/server_app.py::run_dir` | `paths.py::module_run_results_dir` | `module_run_results_dir(_MODULE, run_id)` | VERIFIED -- line 904 |
| `personalized/server_app.py::run_dir` | `paths.py::module_run_results_dir` | `module_run_results_dir(_MODULE, run_id)` | VERIFIED -- line 989 |
| `adaptive/server_app.py::run_dir` | `paths.py::module_run_results_dir` | `module_run_results_dir(_MODULE, run_id)` | VERIFIED -- line 1293 |
| `pfedrec/server_app.py::run_dir` | `paths.py::module_run_results_dir` | `module_run_results_dir(_MODULE, run_id)` | VERIFIED -- line 1157 |
| `baseline/server_app.py::extra_eval` | `strategy.aggregate_evaluate` | `strategy.aggregate_evaluate(final_eval_round_index, extra_results, [])` | VERIFIED -- line 681 |
| `personalized/server_app.py::extra_eval` | `strategy.aggregate_evaluate` | `strategy.aggregate_evaluate(final_eval_round_index, extra_results, [])` | VERIFIED -- line 831 |
| `adaptive/server_app.py::extra_eval` | `strategy.aggregate_evaluate` | `strategy.aggregate_evaluate(final_eval_round_index, extra_results, [])` | VERIFIED -- line 1025 |
| `pfedrec/server_app.py::extra_eval` | `strategy.aggregate_evaluate` | `strategy.aggregate_evaluate(final_eval_round_index, extra_results, [])` | VERIFIED -- line 958 |
| `pfedrec/server_app.py::pfr08 hook` | `final_metrics["best"]` | `_emit_pfr_08_verification(final_metrics["best"], ...)` | VERIFIED -- line 1142 |
| `adaptive/server_app.py::extra_eval_config` | `strategy.best_prototype` | `final_global_prototype = strategy.get_global_prototype()` after `strategy._global_prototype = strategy.best_prototype` | VERIFIED -- lines 960-996 |
| all 4 `server_app.py::manifest` | `manifest.py::RunManifest.metrics` | `dataclass_replace(manifest, final_eval_round_index=N, metrics=results_data["final_metrics"])` BEFORE `embed_manifest_in_result` | VERIFIED -- confirmed in all 4 files |
| `sweep.yaml::metric.name` | `server_app.py::wandb.run.summary["best/sampled_ndcg@10"]` | `name: best/sampled_ndcg@10` | VERIFIED -- sweep.yaml line 18; summary loop present in server_app.py |

---

## Test Suite Counts (Independent Confirmation)

| Suite | Passing | Deselected (slow) | Command |
|-------|---------|-------------------|---------|
| Foundation | 100 | 4 | `cd scripts/foundation && pytest tests/ -q -m "not slow"` |
| Baseline | 26 | 1 | `cd federated-baseline-cf && pytest tests/ -q -m "not slow"` |
| Personalized | 38 | 0 | `cd federated-personalized-cf && pytest tests/ -q -m "not slow"` |
| Adaptive | 73 | 0 | `cd federated-adaptive-personalized-cf && pytest tests/ -q -m "not slow"` |
| PFedRec | 41 | 0 | `cd federated-pfedrec && pytest tests/ -q -m "not slow"` |
| **Total** | **278** | **5** | |

Zero regressions across all suites. Counts match SUMMARY.md claims exactly.

---

## Anti-Patterns Found

No blockers or warnings identified. The code does not contain:
- Placeholder/stub implementations in the Phase 6 code paths (all extra-eval-round, path, and manifest blocks are fully connected).
- Hardcoded empty return values in the critical paths.
- Forbidden `Path("../results/federated")` patterns (confirmed zero occurrences across all 4 `server_app.py` files).
- Legacy `wandb.run.summary[f"final/{key}"]` patterns (confirmed zero occurrences).
- Old `name: final/sampled_ndcg@10` in sweep.yaml (confirmed zero occurrences).

**One notable structural observation** (not a blocker): The PFedRec strategy emits per-group exposure counts with slash-delimiter (`evaluated_users/sparse`) while the other three modules use underscore (`evaluated_users_sparse`). This deviation is fully documented in Plan 06 SUMMARY and Plan 07 SUMMARY. Tests in `test_server_integration.py` for PFedRec correctly assert the slash-delimiter form. For Phase 7 thesis table generation, any cross-module aggregation script that reads per-group exposure counts will need to handle both formats.

---

## Human Verification Required

### 1. W&B project routing (EVL-05)

**Test:** After Phase 6 lands, run `cd federated-baseline-cf && flwr run . --run-config "num-server-rounds=2 mode=benchmark_cross_device wandb-enabled=true"`. Repeat with a short run for personalized, adaptive, and pfedrec (2-round runs).
**Expected:** Each run appears in the W&B UI under `<entity>/federated-cf-cross-device` project (not legacy per-module projects). Per-run group correctly shows `best/*` and `last/*` summary keys.
**Why human:** `wandb.init()` and `wandb.run.summary` side-effects are not interceptable in unit tests without a wandb mock infrastructure. The routing logic is code-verified but live dashboard confirmation is required.

### 2. Cross-silo legacy path coexistence (EVL-04 D-03)

**Test:** Confirm that `ls results/federated/` before and after a cross-device run shows: (a) pre-existing `*_results.json` flat files are unmodified, and (b) new `<module>/<run_id>/results.json` + `manifest.json` appear in the per-run-dir layout. `git status` should show only new files (untracked), zero modifications to pre-existing result files.
**Expected:** D-03 coexistence -- new and old result files coexist without collision; `sibling_name="manifest.json"` goes into the per-run dir; legacy `<run_id>-manifest.json` in the flat dir for any pre-Phase-6 runs.
**Why human:** Filesystem state validation across prior vs new runs requires a live run. Integration tests assert no-clobber at code level but visual `git status` confirmation is the human spec for this constraint.

### 3. Best-round restore correctness on full PFedRec paper_compat run (EVL-01, PFR-08)

**Test:** Run `python scripts/run.py pfedrec paper_compat_pfedrec --run-config "run-seed=42"` (100 rounds). Inspect `results/federated/pfedrec/<run_id>/manifest.json`.
**Expected:** `final_metrics.best.sampled_hr@10` within 0.729 +/- 0.02; `final_metrics.best.sampled_ndcg@10` within 0.441 +/- 0.02; `pfr08_verification.passed = true`; `final_eval_round_index >= 2` (extra eval round ran after best-round restore); `best_round >= 1`.
**Why human:** Reproducing within +/-2 points of the IJCAI-23 PFR-08 reference requires the full 100-round convergence. Unit tests verify the restore mechanism and hook wiring; numerical correctness on real data is a live-run verification.

---

## Gaps Summary

No gaps identified. All automated must-haves are verified at both the artifact level (files exist with the required contents) and the wiring level (imports connected, data flows complete). The three human-verification items are observational (W&B routing, filesystem coexistence, numerical reproduction) -- they do not indicate any code deficiency but require a live run to confirm.

One cross-phase note for Phase 7 planning: the PFedRec slash-delimiter deviation (`evaluated_users/sparse` vs `evaluated_users_sparse`) means any Phase 7 thesis table aggregation script that reads per-group exposure counts across all four modules must normalize these two key formats. This is a data consumer concern, not a Phase 6 deficiency.

---

_Verified: 2026-04-29T09:30:00Z_
_Verifier: Claude (gsd-verifier), claude-sonnet-4-6_
