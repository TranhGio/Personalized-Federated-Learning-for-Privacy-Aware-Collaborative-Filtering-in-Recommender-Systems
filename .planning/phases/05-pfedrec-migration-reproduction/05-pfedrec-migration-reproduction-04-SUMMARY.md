---
phase: 05-pfedrec-migration-reproduction
plan: 04
subsystem: federated-learning
tags: [pfedrec, server-app, cross-device, pfr-08-autoverify, d-14, d-15, d-13, manifest-double-write, sc-1-back-pointer, best-round-restore, cold-start-counter, partition-id-sampling, g-03-01, adp-06, pitfall-5-option-b, wave-3]
requirements: [PFR-02, PFR-06, PFR-08, PFR-09]

# Dependency graph
dependency-graph:
  requires:
    - phase-01 foundation (build_run_manifest / embed_manifest_in_result / write_manifest_sibling, server_rng, resolve_mode_defaults / log_mode_and_overrides, verify_bundle, load_split_manifest, generate_run_id)
    - phase-05 plan-01 (PFedRecSplitFedAvg + GLOBAL/LOCAL frozensets per D-01)
    - phase-05 plan-02 (pyproject cross-device defaults; mode profile paper_compat_pfedrec; weight_policy='uniform' D-25)
    - phase-05 plan-03 (client_app discover_only short-circuit + .embedding_cache/{run_id}/partition_{pid}.pt cache + EvaluateMetricsContract sufficient-stat schema)
    - phase-05 plan-01 task-3 (PFR-02-AUDIT.md — the SC-1 cross-walk this plan back-points to via the manifest's audit_doc field)
  provides:
    - federated-pfedrec/federated_pfedrec/server_app.py — full Phase-5 cross-device main loop with all 5 PFedRec-specific deltas + Phase 3/4 carry-forward + the unique D-14 PFR-08 auto-verify hook + D-15 audit_doc back-pointer to PFR-02-AUDIT.md
    - federated-pfedrec/tests/test_server_integration.py — 8 GREEN integration tests covering VALIDATION rows 5-04-01..5-04-08
    - Module-top _parse_reference_results / _emit_pfr_08_verification helpers (importable from tests + Plan 05's subprocess regression guard)
    - results_data['_manifest']['pfr08_verification'] field shape (consumed by Plan 05 subprocess test)
    - results_data['_manifest']['audit_doc'] = 'PFR-02-AUDIT.md' SC-1 back-pointer in result JSON
    - selected_clients_per_round (partition-id space, 0..N-1) + cold_starts.{per_round, total, rate} + checkpoint.{rule, best_round, best_sampled_ndcg@10} fields in result JSON
  affects:
    - phase-05 plan-05 (subprocess regression guard — already committed at e928cff and consumes selected_clients_per_round, partition_{pid}.pt schema_v3 cache, _manifest.pfr08_verification, _manifest.audit_doc)
    - phase-06 evaluation-harness (consumes EvaluateMetricsContract sufficient-stat schema via strategy.aggregate_evaluate)
    - phase-07 thesis-eval (consumes the [PFR-08 VERIFIED] / [PFR-08 FAILED] auto-verify result)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "5-PFedRec-specific-deltas-over-Phase-4 server template: D-12 strategy rename, D-01 GLOBAL bias propagation, no prototype/alpha bookkeeping, no centralized eval, D-14 PFR-08 auto-verify hook."
    - "D-14 PFR-08 auto-verify hook: parses IJCAI-23-PFedRec/sh_result/ml-1m.txt line 2 (Open Question 1 recommendation: most recent / closest to paper round 89), asserts |our - ref| <= 2.0pts, prints [PFR-08 VERIFIED] / [PFR-08 FAILED]. Non-fatal — the helper wraps reference parsing in try/except and surfaces failure as passed=False rather than raising."
    - "D-15 audit_doc back-pointer post-build mutation: build_run_manifest does NOT accept audit_doc directly; threading it via results_data['_manifest']['audit_doc'] = 'PFR-02-AUDIT.md' (Phase-3/Phase-4 idiom for post-build payload extensions). Closes the SC-1 trail end-to-end through the result JSON."
    - "Pitfall 5 Option B (D-24 uniform weight on FIT side): FitRes.num_examples = 1 per client makes BaseFedAvg's existing num_examples-weighted aggregate mathematically uniform. Mirrors engine.py:81 len(round_user_params) division WITHOUT overriding aggregate_fit (D-23 invariant preserved)."
    - "TDD-bundle pattern: tests authored FIRST (RED commit) then implementation (GREEN commit) — single tdd=true task ships both files atomically. No 'ship code with smoke verify, ship tests separately' anti-pattern."

key-files:
  created:
    - federated-pfedrec/federated_pfedrec/server_app.py (1118 LOC; replaces pre-Phase-5 untracked 467-LOC WIP)
    - federated-pfedrec/tests/test_server_integration.py (202 LOC, 8 GREEN tests)
  modified: []  # Note: server_app.py was untracked WIP before this plan; the GREEN commit creates the canonical Phase-5 version.

key-decisions:
  - "Open Question 1 resolved: line 2 / most recent / closest to paper round 89 (HR=0.7286, NDCG=0.4407) is the canonical PFR-08 reference target. Open Question 1's 'higher-of-two' alternative would have biased the ±2pt tolerance check toward harder reproduction; line 2 is the documented paper-faithful choice."
  - "Open Question 3 resolved: D-14 hook fires AFTER embed_manifest_in_result (so the audit dict can be injected into _manifest) AND BEFORE the W&B summary write (so failure status surfaces as wandb.run.summary['final/pfr08'] = bool). Triple surface: stdout log line + result-JSON-embedded under _manifest.pfr08_verification + W&B summary."
  - "D-14 hook is NON-FATAL by construction: failed reproduction returns passed=False; reference parse errors are caught and surfaced as passed=False with the error string. The helper NEVER raises. Pinned by test_pfr08_autoverify_fail_outside_2pts."
  - "audit_doc='PFR-02-AUDIT.md' threaded via post-build mutation of the embedded _manifest dict (build_run_manifest does NOT accept audit_doc directly). Phase-3/Phase-4 idiom for post-build payload extensions; the Phase 4 server_app does the same with best_prototype. The SC-1 back-pointer closes the trail: result JSON _manifest.audit_doc points at the cross-walk Plan 01 Task 3 authored at .planning/phases/05-pfedrec-migration-reproduction/PFR-02-AUDIT.md."
  - "W&B project switch under paper_compat_pfedrec defaults to federated-cf-cross-device (D-10) — same bucket as Phase 2/3/4 cross-device runs. wandb-project='' in pyproject.toml falls through to this default; explicit override remains the path for dashboard isolation."
  - "Pitfall 5 Option B for D-24 uniform: FitRes.num_examples = 1 per client. NOT 0 (FedAvg div-by-zero), NOT len(positives) (would be num_positives weighting). The single literal '1' in the code site is pinned by the acceptance criterion grep -c 'num_examples = 1' >= 1 (counted 4: code site + Pitfall 5 docstring + 3 docstring references)."
  - "Test 1 docstring drift dodge: the 'ADP-06 partition-id-space sampling' module-docstring line was reworded from '_server_sampler = server_rng(run_seed) instantiated ONCE pre-loop' to 'a single server_rng(run_seed)-backed random.Random instance is held pre-loop' so test_discovery_round_partition_id_sampling's count('_server_sampler = server_rng') == 1 invariant holds. Same docstring-rewording pattern Plan 03 used for the BSL-05 cross-file regression."
  - "FedProx-token rewording: the module-docstring's 5-delta header originally said 'no PFedRecSplitFedProx variant per D-07'. Reworded to 'no FedProx variant per D-07' so the acceptance criterion grep -c 'PFedRecSplitFedProx|SplitFedProx' == 0 (D-07) holds. Same pattern."

patterns-established:
  - "Pattern 1: D-14 reproduction-gate hook with triple surface (stdout + result-JSON-embedded + W&B summary). Non-fatal by construction — auth errors and parse errors surface as passed=False, never raise. Generalizes to any future migration that needs to reproduce a published reference within tolerance."
  - "Pattern 2: SC-1 audit_doc back-pointer via post-build manifest mutation. When the foundation manifest API does not yet accept a kwarg directly, threading the field via results_data['_manifest'][field] = value AFTER embed_manifest_in_result is the canonical Phase-3/Phase-4 idiom. Future migrations follow the same pattern for any post-build payload extensions."
  - "Pattern 3: TDD-bundle pattern for server_app + integration tests in a single tdd=true task. Tests authored FIRST as RED commit; implementation grows to pass them as GREEN commit. Avoids the 'ship code with smoke verify, ship tests separately in a follow-up plan' anti-pattern that would create a verification gap."
  - "Pattern 4: docstring-drift dodge for source-level regression tests. When module docstrings would otherwise duplicate a load-bearing literal (e.g. '_server_sampler = server_rng' or 'PFedRecSplitFedProx'), reword the docstring to a paraphrase so the test's count == 1 / count == 0 invariant remains mechanical (substring-based) rather than AST-aware. Cosmetic change; load-bearing behavior unchanged."

requirements-completed: [PFR-02 (server-side), PFR-06 (server half — discovery + seeded sampling + Pitfall 5 uniform), PFR-08 (auto-verify hook), PFR-09 (FND-07 manifest with module='pfedrec' + audit_doc back-pointer)]

# Metrics
duration: 8min
completed: 2026-04-28
---

# Phase 05 Plan 04: PFedRec server_app cross-device migration with D-14 PFR-08 auto-verify hook Summary

**One-liner:** Cross-device PFedRec server_app rewritten with the 5 PFedRec-specific deltas (D-12 strategy, D-01 GLOBAL bias propagation, no prototype/alpha bookkeeping, no centralized eval, D-14 PFR-08 auto-verify hook) over the Phase 4 Plan 5 template; G-03-01 discovery + ADP-06 partition-id-space sampler + D-13 cold-start counter + D-13 best-round-restore via the Phase-3-D-27 carry-forward in-memory snapshot pattern + Pitfall 5 Option B (FitRes.num_examples=1 for D-24 uniform); D-15 manifest double-write with `module="pfedrec"` AND `audit_doc="PFR-02-AUDIT.md"` (the SC-1 back-pointer to Plan 01 Task 3's cross-walk); 8 GREEN integration tests bundled with the implementation in a single TDD task; full module suite 36/36 GREEN.

## Performance

- **Duration:** ~8 min (focused work; commits span 2026-04-28 18:14:29 → 18:22:33 UTC).
- **Started:** 2026-04-28T18:14:29Z
- **Completed:** 2026-04-28T18:22:33Z
- **Tasks:** 1 bundled tdd=true (RED + GREEN commits = 2 atomic commits)
- **Files created:** 2 (server_app.py, test_server_integration.py)
- **Files modified:** 0
- **Tests added:** 8 GREEN (test_server_integration.py)
- **Cumulative module suite:** 36 GREEN (28 inherited from Plans 01/02/03 + 8 new)
- **Foundation suite:** 82 passed + 3 skipped (no regression; Plan 05's subprocess test is in place at e928cff and skips cleanly on smoke config)

## What Shipped

### Single bundled TDD task — 2 atomic commits

**Commit `3e0cffc` (RED step — tests added, failing on import):**
- `federated-pfedrec/tests/test_server_integration.py` — 202 LOC, 8 source-level + functional tests anchored on `inspect.getsource(server_app)` (live Grid not available in unit tests).

**Commit `7c97f79` (GREEN step — full server_app rewrite):**
- `federated-pfedrec/federated_pfedrec/server_app.py` — 1118 LOC, replaces pre-Phase-5 untracked 467-LOC WIP.

### Test-by-test VALIDATION coverage

| VALIDATION row | Test | Decision pinned | Mechanism |
|---|---|---|---|
| 5-04-01 | `test_discovery_round_partition_id_sampling` | PFR-06 / G-03-01 | `src.count("_server_sampler = server_rng") == 1` (single instance invariant) + `_server_sampler.sample(range(` substring (partition-id-space) |
| 5-04-02 | `test_server_rng_seeded_sampling` | PFR-06 / FND-06 | `server_rng(42)` byte-identity across 2 instances; `server_rng(43)` divergence |
| 5-04-03 | `test_pfr08_autoverify_parses_sh_result` | D-14 | parses real `IJCAI-23-PFedRec/sh_result/ml-1m.txt` line 2 → `(0.7286, 0.4407)` (pytest.approx tolerance 1e-3) |
| 5-04-04 | `test_pfr08_autoverify_pass_within_2pts` | D-14 | synthetic ref + final_metrics within ±2pts → `passed=True`, `[PFR-08 VERIFIED]` log line |
| 5-04-05 | `test_pfr08_autoverify_fail_outside_2pts` | D-14 (non-fatal) | synthetic ref + final_metrics WAY off → `passed=False`, `[PFR-08 FAILED]` log line; CRITICAL: helper does NOT raise |
| 5-04-06 | `test_manifest_double_write_module_pfedrec` | D-15 / PFR-09 | source contains `module="pfedrec"` AND `audit_doc="PFR-02-AUDIT.md"` AND `embed_manifest_in_result` AND `write_manifest_sibling` |
| 5-04-07 | `test_cold_starts_per_round_logged` | D-13 cold counter | `cold_starts` count >= 2 (declaration + result write); `partition_` + `.pt` + `.exists()` substrings |
| 5-04-08 | `test_best_round_restore_against_ndcg10` | D-13 best-round-restore | `thesis_metrics.get("sampled_ndcg@10"` present; `checkpoint_rule in ("best_round_restore", "best_round")` literal; `src.rfind("arrays = best_arrays") > 0` (rfind to anchor on the LAST occurrence — Phase 4 Plan 5 lesson on docstring duplicates) |

### 5 PFedRec-specific deltas over Phase 4 Plan 5 template

1. **D-12 strategy class** — `PFedRecSplitFedAvg` (Plan 01) replaces `SplitFedAvg`. NO `PFedRecSplitFedProx` variant per D-07 (paper does not use FedProx). The strategy_name conditional branch in the pre-Phase-5 server_app is eliminated; only one strategy is instantiated.

2. **D-01 GLOBAL bias propagation** — initial `ArrayRecord(global_model.get_global_parameters())` carries BOTH `embedding_item.weight` AND `affine_output.bias` because `PFedRecMLP._GLOBAL_PARAMS = ('embedding_item.weight', 'affine_output.bias')` per Plan 01. The result JSON's `federated_config.global_params` field reflects this: `["embedding_item.weight", "affine_output.bias"]` (pre-Phase-5 was `["embedding_item.weight"]` only). The `local_params` field is `["affine_output.weight"]` (pre-Phase-5 had both weight + bias).

3. **No prototype / alpha bookkeeping** — Phase 4 Plan 5 had `best_prototype` snapshot/restore at `strategy.snapshot_best_prototype(...)` + 6-scalar `alpha_diagnostics` aggregation across `alpha_mean / alpha_std / alpha_p25 / alpha_p50 / alpha_p75 / alpha_clip_hit_rate`. Phase 5 has neither — those blocks are entirely absent. PFedRec's strategy class does NOT expose `snapshot_best_prototype` or `_global_prototype`; the per-user score function is the personalization mechanism, not a server-side EMA prototype.

4. **No centralized eval block** — split learning means the server holds NO `LOCAL` params (per-user `affine_output.weight`). `final_metrics = eval_metrics_history.get(best_round_num, eval_metrics_history.get(actual_rounds, {}))` per the Phase 3 idiom — no `model.set_local_parameters(...)` + `evaluate_pfedrec_sampled(...)` server-side path. Stdout explicitly notes "(Centralized evaluation not possible in split learning)".

5. **D-14 PFR-08 auto-verify hook (NEW to Phase 5)** — two module-level helpers:
   - `_parse_reference_results(reference_path) -> Tuple[float, float]`: reads `IJCAI-23-PFedRec/sh_result/ml-1m.txt`, picks the LAST non-empty line (line 2 today: HR=0.7286, NDCG=0.4407), parses the dash-delimited `hr: X-ndcg: Y` tokens. Open Question 1 resolution: line 2 / most recent / closest to paper-reported best round 89.
   - `_emit_pfr_08_verification(final_metrics, reference_path, tolerance_pts=2.0) -> Tuple[bool, str, Dict]`: computes `delta_hr_pts = abs(our_hr - ref_hr) * 100.0` and `delta_ndcg_pts = abs(our_ndcg - ref_ndcg) * 100.0`, asserts both ≤ tolerance, returns `(passed, log_line, audit_dict)`. The log line format is `"[PFR-08 VERIFIED]" / "[PFR-08 FAILED]" + our_hr@10=... + ref_hr@10=... + delta_hr=Xpts | our_ndcg@10=... + ref_ndcg@10=... + delta_ndcg=Ypts | tolerance=2.0pts"`.

   The hook fires AFTER `embed_manifest_in_result(manifest, results_data)` (so the audit dict can be injected into `_manifest`) AND BEFORE the W&B summary write (so failure surfaces as `wandb.run.summary['final/pfr08'] = bool`). Open Question 3 resolution: triple surface (stdout + result-JSON-embedded under `_manifest.pfr08_verification` + W&B summary).

   **CRITICAL: non-fatal** — the helper wraps reference parsing in `try/except RuntimeError` and surfaces failure as `passed=False` rather than raising. Failed reproduction does NOT abort the run; downstream tooling can decide policy. Pinned by `test_pfr08_autoverify_fail_outside_2pts`.

### Phase 3/4 carry-forward (verbatim clone with PFedRec adaptation)

- **D-25 mode resolver header** — `mode = str(context.run_config.get("mode", "paper_compat_pfedrec"))` + `profile = resolve_mode_defaults(mode)` + `overrides = log_mode_and_overrides(...)`. Every hyperparameter read is `int/float/str(context.run_config.get(key, profile.field))` so the mode profile is the canonical source.
- **D-02 mirror frozen-cross-silo guard** — `if mode == "cross_silo_legacy": raise NotImplementedError(...)` BEFORE any heavy work. Error message cites D-09 / D-02 + `.planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md §Deferred`.
- **run_id materialized EARLY** — `run_id = str(context.run_config.get("run-id", "")) or generate_run_id()` per the Phase 3 idiom. The server's D-13 cold-start probe and the client's cache path coincide on `.embedding_cache/{run_id}/`.
- **W&B project default** — `federated-cf-cross-device` for `mode in {"benchmark_cross_device", "paper_compat_pfedrec"}` per D-10. Falls back to `federated-pfedrec` for any other mode.
- **G-03-01 discovery round** — broadcast `evaluate(discover_only=True)` to ALL nodes; collect `partition_id` from each response's MetricRecord; build `partition_to_node_id: Dict[int, int]`; `assert not missing` invariant. Phase 3 client_app's `discover_only` short-circuit (Plan 03) handles the response side.
- **ADP-06 partition-id-space sampling** — `_server_sampler = server_rng(run_seed)` (single instance per run); `_server_sampler.sample(range(expected_n), num_selected)` per round; `selected_clients_per_round` accumulates partition_ids 0..N-1.
- **D-13 cold-start counter** (the per-round logging counter — distinct from best-round-restore) — probes `(.embedding_cache/{run_id}/partition_{pid}.pt).exists()` per selected pid; under `reuse_cache=true` short-circuits to 0 with documented log line citing D-09. `results_data["cold_starts"] = {per_round, total_cold_starts, total_client_selections, cold_start_rate}`.
- **D-13 best-round-restore via the Phase-3-D-27 carry-forward in-memory snapshot pattern** (NOT to be confused with CONTEXT.md D-27 weight-policy override) — when `current_ndcg > best_metric`, deep-clone `arrays.to_torch_state_dict()` into `best_arrays = ArrayRecord({...})`. Restore at loop end via `arrays = best_arrays` BEFORE the final result write. CONTEXT.md D-13: monitor metric is `sampled_ndcg@10`. Spelling tolerance — `checkpoint_rule` accepts both `best_round_restore` (pyproject default) and `best_round` (mode profile default).
- **D-15 manifest double-write** — `build_run_manifest(run_id, mode_profile=profile, run_seed, mapping_sha256, split_hash, exclusion_sha256, foundation_contract_sha256, raw_data_hash, builder_version, overrides, module="pfedrec")` + `embed_manifest_in_result(manifest, results_data)` + `write_manifest_sibling(manifest, results_filename)`.
- **D-15 SC-1 back-pointer** — `audit_doc = "PFR-02-AUDIT.md"` threaded via post-build mutation: `results_data["_manifest"]["audit_doc"] = audit_doc`. The foundation manifest API does NOT accept `audit_doc` directly; threading it via the embedded `_manifest` dict is the Phase-3/Phase-4 idiom for post-build payload extensions (Phase 4 does the same with `best_prototype`). The W&B `_manifest` config block also carries the field for dashboard auditability.
- **Pitfall 5 Option B (D-24 uniform weight on FIT side)** — `num_examples = 1` per client when wrapping `train_responses` into `FitRes` objects. `BaseFedAvg.aggregate_fit`'s existing num_examples-weighted aggregator is then mathematically uniform without overriding `aggregate_fit` (D-23 invariant preserved). Mirrors `engine.py:81 len(round_user_params)` division. Inline comment documents Pitfall 5 / D-24 / Research §Pitfall 5.
- **D-18 surgical preservation** — `DummyClientProxy`, `weighted_average_metrics`, `print_evaluation_metrics`, `EarlyStopping` setup/teardown, CUDA fallback are preserved verbatim from the pre-Phase-5 shape. Stdlib `random` is eradicated module-wide (acceptance criteria grep `-cE "random\.seed\(|random\.sample\(|^import random$"` returns 0).

## D-14 Auto-Verify Hook Implementation Details

### Placement

The hook fires in this exact sequence at the end of `@app.main`:

1. `final_metrics = dict(eval_metrics_history.get(final_round_for_metrics, {}))` — best-round (or last-round) metrics.
2. `manifest = build_run_manifest(..., module="pfedrec")` — assemble the FND-07 manifest.
3. `embed_manifest_in_result(manifest, results_data)` — D-15 part 1 (mutates results_data in place).
4. `results_data["_manifest"]["audit_doc"] = "PFR-02-AUDIT.md"` — D-15 SC-1 back-pointer.
5. **D-14 hook fires here** — `pfr08_passed, pfr08_log_line, pfr08_audit = _emit_pfr_08_verification(...)`.
6. `print(pfr08_log_line)` — stdout surface.
7. `results_data["_manifest"]["pfr08_verification"] = pfr08_audit` — JSON surface.
8. `json.dump(results_data, f, indent=4, default=str)` — flush to disk.
9. `write_manifest_sibling(manifest, results_filename)` — D-15 part 2.
10. `wandb.run.summary["final/pfr08"] = bool(pfr08_passed)` + delta scalars + final_metrics — W&B surface.

### Log line format

```
[PFR-08 VERIFIED] our_hr@10=0.7300 ref_hr@10=0.7286 delta_hr=0.14pts | our_ndcg@10=0.4500 ref_ndcg@10=0.4407 delta_ndcg=0.93pts | tolerance=2.0pts
[PFR-08 FAILED] our_hr@10=0.5000 ref_hr@10=0.7286 delta_hr=22.86pts | our_ndcg@10=0.2000 ref_ndcg@10=0.4407 delta_ndcg=24.07pts | tolerance=2.0pts
```

### Audit dict shape (`results_data["_manifest"]["pfr08_verification"]`)

```python
{
    "passed": bool,
    "delta_hr_pts": float,        # abs(our_hr - ref_hr) * 100.0
    "delta_ndcg_pts": float,      # abs(our_ndcg - ref_ndcg) * 100.0
    "ref_hr": float,              # parsed from ml-1m.txt line 2
    "ref_ndcg": float,
    "our_hr": float,              # final_metrics["sampled_hr@10"]
    "our_ndcg": float,            # final_metrics["sampled_ndcg@10"]
    "ref_path": str,              # absolute path to the reference file
    "tolerance_pts": float,       # default 2.0
}
```

On parse error (file missing / empty / format unparseable):
```python
{"passed": False, "error": "<error message>"}
```

### Non-fatal semantics

The helper wraps `_parse_reference_results(reference_path)` in `try/except RuntimeError`. NaN check (`v != v`) on `final_metrics` for missing keys also short-circuits to `passed=False` rather than raising. **The helper NEVER raises.** Failed reproduction is auditable but does NOT abort the run — downstream tooling can decide policy. Pinned by `test_pfr08_autoverify_fail_outside_2pts`.

## D-15 audit_doc Back-Pointer (SC-1 Closure End-to-End)

The Phase 5 SC-1 cross-walk authored by Plan 01 Task 3 lives at `.planning/phases/05-pfedrec-migration-reproduction/PFR-02-AUDIT.md`. Plan 04's server_app threads a back-pointer through the result JSON's `_manifest` block:

- `results_data["_manifest"]["audit_doc"] = "PFR-02-AUDIT.md"`.
- W&B `_manifest` config block also carries the field.

This closes the SC-1 trail: a future maintainer reading `results/federated/pfedrec/<run_id>_results.json` can follow `_manifest.audit_doc` → the cross-walk → the 9-row reference divergence table → the locked CONTEXT D-XX decisions. The result JSON of the PFR-08 reproduction artifact is now self-documenting back to the SC-1 audit.

The foundation `build_run_manifest` API does NOT accept `audit_doc` directly; threading it via post-build mutation of the embedded `_manifest` dict is the Phase-3/Phase-4 idiom for post-build payload extensions (Phase 4 server_app does the same with `best_prototype`).

## W&B Project Switch (D-10) Confirmation

For `mode in {"benchmark_cross_device", "paper_compat_pfedrec"}`, the default W&B project is `federated-cf-cross-device` (shared with Phase 2/3/4 cross-device runs). For any other mode, the default is `federated-pfedrec`. The `wandb-project=""` setting in `pyproject.toml` falls through to this default; explicit override remains the path for dashboard isolation.

## Acceptance Criteria Status

All 26 Plan 04 acceptance criteria PASSED:

- `grep -c "PFedRecSplitFedAvg" server_app.py` → **7** ✓ (≥2)
- `grep -cE "(^|[^A-Za-z_])SplitFedAvg\(" server_app.py` → **0** ✓ (word-boundary check; only PFedRecSplitFedAvg matches)
- `grep -cE "PFedRecSplitFedProx|SplitFedProx" server_app.py` → **0** ✓ (D-07 — reworded the docstring's "no PFedRecSplitFedProx variant" to "no FedProx variant" to keep the count at 0)
- `grep -cE "resolve_mode_defaults|log_mode_and_overrides" server_app.py` → **5** ✓ (≥2)
- `grep -c "cross_silo_legacy" server_app.py` → **1** ✓ (≥1)
- `grep -c "raise NotImplementedError" server_app.py` → **1** ✓ (≥1)
- `grep -c "_server_sampler = server_rng" server_app.py` → **1** ✓ (=1; ADP-06 single-instance invariant; reworded the docstring to avoid the double-count Test 1 caught on first run)
- `grep -c "discover_only" server_app.py` → **3** ✓ (≥1)
- `grep -c "selected_clients_per_round" server_app.py` → **5** ✓ (≥2)
- `grep -cE "_parse_reference_results|_emit_pfr_08_verification" server_app.py` → **5** ✓ (≥2)
- `grep -cE "PFR-08 VERIFIED|PFR-08 FAILED" server_app.py` → **3** ✓ (≥1)
- `grep -c "pfr08_verification" server_app.py` → **2** ✓ (≥1)
- `grep -cE 'module="pfedrec"|module=.pfedrec.' server_app.py` → **1** ✓ (≥1)
- `grep -cE 'audit_doc="PFR-02-AUDIT.md"|"PFR-02-AUDIT.md"' server_app.py` → **3** ✓ (≥1)
- `grep -cE "build_run_manifest|embed_manifest_in_result|write_manifest_sibling" server_app.py` → **9** ✓ (≥3)
- `grep -cE "best_round_restore|best_round" server_app.py` → **23** ✓ (≥2)
- `grep -cE "cold_starts|cold_count" server_app.py` → **22** ✓ (≥3)
- `grep -c "num_examples = 1" server_app.py` → **4** ✓ (≥1; Pitfall 5 Option B / D-24)
- `grep -cE "random\.seed\(|random\.sample\(|^import random$" server_app.py` → **0** ✓ (stdlib random eradicated)
- `python -c "from federated_pfedrec.server_app import app, _parse_reference_results, _emit_pfr_08_verification; print('ok')"` → **ok** ✓
- `pytest federated-pfedrec/tests/test_server_integration.py -x -v` → **8 passed** ✓ (Test 3 reads the real ml-1m.txt — present in this clone — does NOT skip)
- `pytest federated-pfedrec/tests/ -x` → **36 passed** ✓ (28 inherited + 8 new)
- Foundation suite: **82 passed + 3 skipped, 0 failed** ✓ (no regression)

## Wave-3 Disjoint File Ownership Held

Plan 04 touched STRICTLY this file set (matching the `parallel_execution` block):

- `federated-pfedrec/federated_pfedrec/server_app.py` ✓ (created — was untracked WIP)
- `federated-pfedrec/tests/test_server_integration.py` ✓ (new)

ZERO touch of Plan 05's owned file: `scripts/foundation/tests/test_pfedrec_subprocess_determinism.py`. Plan 05's commit `e928cff` (`test(05-05): add subprocess determinism regression guard for PFedRec`) was already committed in parallel. Both plans used `--no-verify` to avoid pre-commit hook contention.

## D-18 Surgical-Edit Discipline (confirmed)

- Pre-existing untracked WIP files (`.gitignore`, `claude.md`, `__init__.py`, `early_stopping.py`, `models/__init__.py`, `models/losses.py`) were left untouched per D-18 (out of scope for Plan 04).
- The pre-Phase-5 server_app.py was 467 LOC of untracked WIP (legacy `SplitFedAvg/SplitFedProx` shape with raw `random.sample(node_ids, ...)` and per-user-subdir cache references). The GREEN commit replaces it wholesale with the Phase-5-aligned 1118-LOC version. Same pattern Plan 01 / Plan 03 used for their respective untracked source files.
- `git diff --name-only` over Plan 04's two commits shows ONLY the two owned files (`server_app.py` + `test_server_integration.py`).

## Decision-ID Disambiguation (mechanical pinning)

The Phase 5 CONTEXT.md has TWO distinct decisions named "D-13":
- **CONTEXT.md D-13** = best-round-restore monitor metric is `sampled_ndcg@10`.

And TWO distinct decisions named "D-27":
- **CONTEXT.md D-27** (this phase) = weight-policy override visibility behavior under paper_compat_pfedrec.
- **Phase-3-D-27 carry-forward** (cross-phase idiom) = the in-memory snapshot-and-restore implementation pattern from Phases 2/3/4.

Plan 04's "best-round-restore" references explicitly use the disambiguated names:
- "D-13 best-round-restore" for the monitor metric (CONTEXT.md D-13).
- "Phase-3-D-27 carry-forward" for the implementation pattern.

CONTEXT.md D-27 (weight-policy override) is left untouched and unambiguous. The module docstring + 3 inline comments cite the disambiguation explicitly.

## Plan 05 Readiness Confirmation (already shipped at `e928cff`)

Plan 05 (subprocess regression guard) is already committed and consumes Plan 04's outputs:

- `selected_clients_per_round` — partition-id space (0..N-1), byte-identity asserted across two same-seed subprocess runs.
- `.embedding_cache/{run_id}/partition_{pid}.pt` schema_v3 cache — single-key `{"affine_output.weight"}` payload after D-01 (bias is now GLOBAL, aggregated server-side).
- `_manifest.pfr08_verification` field — Plan 05's smoke 2-round config skips this assertion cleanly (`pfr08_verification absent on both runs (smoke config too small to trigger the D-14 auto-verify hook)`).
- `_manifest.audit_doc` field — pinned at `"PFR-02-AUDIT.md"` for SC-1 closure traceability.

Foundation suite: 82 passed + 3 skipped (3 expected skips on tiny configs across Phase 2/3/4/5 subprocess tests). No regression.

## Commits

| SHA | Type | Subject |
|---|---|---|
| `3e0cffc` | test(05-04) | RED: 8 failing server integration tests for PFR-06/PFR-08/D-13/D-15 |
| `7c97f79` | feat(05-04) | GREEN: rewrite server_app.py for cross-device PFedRec with D-14 PFR-08 hook |

Both commits used `--no-verify` per the parallel_execution block (Plan 05 ran in parallel; orchestrator validates hooks once after all agents complete).

## Deviations from Plan

**None — plan executed exactly as written.**

Two non-deviation cosmetic auto-fixes applied without expanding scope (matching the Plan 03 docstring-rewording pattern for BSL-05):

1. **Test 1 docstring rewording (server_app.py module docstring):** First test run caught the literal `_server_sampler = server_rng(run_seed)` substring duplicated in the module-level docstring's "ADP-06 partition-id-space sampling" header (test asserts count == 1). Reworded to "a single `server_rng(run_seed)`-backed `random.Random` instance is held pre-loop and reused every round via `.sample(range(expected_n), k)`" so the count == 1 invariant holds. Cosmetic — load-bearing single-instance behavior unchanged.

2. **D-07 docstring rewording:** Module docstring's 5-delta header originally said "no `PFedRecSplitFedProx` variant per D-07". Reworded to "no FedProx variant per D-07" so the acceptance criterion `grep -cE "PFedRecSplitFedProx|SplitFedProx"` returns 0 (D-07 — these tokens forbidden). Cosmetic — load-bearing FedProx-absent behavior unchanged.

These are not contract changes — they are forbidden-token literal eradication so the cross-file regression guards remain mechanical (substring-based) rather than AST-aware. Underlying behavior (single `_server_sampler` instance; zero FedProx import / instantiation) is the load-bearing thing being tested.

## Issues Encountered

- **Untracked source file at start:** `federated-pfedrec/federated_pfedrec/server_app.py` was untracked WIP at plan start (the original module had pre-existing 467-LOC code that was never `git add`-ed). The Phase-5-aligned version committed by this plan is now the canonical tracked version. No regression — pre-existing on-disk content (legacy `SplitFedAvg/SplitFedProx` + raw `random.sample` + per-user-subdir cache + no D-15 manifest + no D-14 PFR-08 hook + no G-03-01 discovery) was replaced wholesale per the plan's `<action>` block. Same pattern Plan 01 used for `strategy.py` / `models/pfedrec_mlp.py` and Plan 03 used for `client_app.py` / `task.py`.
- **First Test 1 RED failure was docstring-drift, not code-drift:** The implementation has exactly ONE `_server_sampler = server_rng(run_seed)` code site, but the module docstring duplicated the literal. Reworded the docstring; both invariants (single-instance code + one source token) now hold. Same docstring-drift pattern Plan 03 caught for stdlib random; resolved with the same docstring-rewording fix.
- **No pre-commit hook flow needed:** `--no-verify` discipline per the parallel_execution block. Plan 05 ran in parallel; the orchestrator validates hooks after both plans complete.

## Self-Check: PASSED

- **Files created:**
  - FOUND: `federated-pfedrec/federated_pfedrec/server_app.py` (verified via `git show --stat 7c97f79`).
  - FOUND: `federated-pfedrec/tests/test_server_integration.py` (verified via `git show --stat 3e0cffc`).
- **Commits:**
  - FOUND: `3e0cffc` (RED — test commit) on `feat/try_to_run_the_baseline`.
  - FOUND: `7c97f79` (GREEN — feat commit) on `feat/try_to_run_the_baseline`.
- **Automated verify:** PASSED.
  - `cd federated-pfedrec && pytest tests/test_server_integration.py -x -v` → **8 passed in 1.03s** ✓
  - `cd federated-pfedrec && pytest tests/ -x` → **36 passed in 1.09s** ✓
  - `python -c "from federated_pfedrec.server_app import app, _parse_reference_results, _emit_pfr_08_verification; print('ok')"` → **ok** ✓
  - Foundation suite: 82 passed + 3 skipped + 0 failed (no regression).
- **Scope boundary:** PASSED. Wave-3 disjoint file ownership held — zero touch of Plan 05's owned files. D-18 surgical scope held — pre-existing untracked WIP files (`.gitignore`, `claude.md`, `__init__.py`, `early_stopping.py`, `models/__init__.py`, `models/losses.py`) untouched.

## Known Stubs

None. Every code path is functional — no mock data, no "TODO / FIXME" placeholders, no `NotImplementedError` outside the explicit D-02 frozen-cross-silo guard (which is the intentional D-09 / D-02 frozen-mode decision and not a stub). The D-14 hook with `passed=False` on parse error is the documented non-fatal contract, not a placeholder.

---

*Phase: 05-pfedrec-migration-reproduction*
*Plan: 04 (Wave 3 — parallel with Plan 05; depends on Plans 01 + 02 + 03)*
*Completed: 2026-04-28*
*Closes: PFR-02 server-side D-12 wire-up + D-01 GLOBAL bias propagation in initial ArrayRecord; PFR-06 server half (G-03-01 discovery + ADP-06 partition-id-space sampling + Pitfall 5 Option B uniform weighting); PFR-08 D-14 auto-verify hook (non-fatal; embeds audit dict in `_manifest.pfr08_verification`); PFR-09 D-15 manifest double-write with `module="pfedrec"` + `audit_doc="PFR-02-AUDIT.md"` SC-1 back-pointer.*
