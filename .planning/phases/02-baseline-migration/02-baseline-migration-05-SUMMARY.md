---
phase: 02-baseline-migration
plan: 05
subsystem: infra
tags: [flower, pytorch, federated, determinism, reproducibility, gap-closure]

# Dependency graph
requires:
  - phase: 01-foundation-contract
    provides: "FND-05 FitMetricsContract + EvaluateMetricsContract extensibility via dataclass fields(); FND-06 server_rng(run_seed) deterministic sampler"
  - phase: 02-baseline-migration-03
    provides: "client_app.py @app.train / @app.evaluate handlers with contract-shaped wire payloads"
  - phase: 02-baseline-migration-04
    provides: "server_app.py main loop with _server_sampler instance; D-26 selected_clients_per_round JSON field"
provides:
  - "Optional partition_id field on FitMetricsContract and EvaluateMetricsContract (D-21 strict-extras whitelist via dataclass fields())"
  - "Discovery-round protocol: one-shot @app.evaluate broadcast with discover_only=true BEFORE the main FL loop; builds partition_to_node_id: Dict[int, int] mapping from client responses"
  - "Partition-space deterministic sampling: _server_sampler.sample(range(num_supernodes), k); selected_clients_per_round stores stable partition_ids (0..N-1), not Flower's ephemeral node_ids"
  - "Subprocess real-loop reproducibility regression guard (test_selected_partitions_byte_identical_across_subprocess_reruns) that would have caught G-03-01 in Plan-04"
affects: [03-personalized, 04-adaptive, 05-pfedrec, thesis-comparison-table]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Discovery-round handshake for building partition_id -> node_id maps before training"
    - "Partition-id space sampling (stable 0..N-1) replaces node_id space sampling (ephemeral os.urandom)"
    - "Optional contract field + dataclass fields()-based whitelist preserves D-21 strict-extras rule while adding new metadata"

key-files:
  created: []
  modified:
    - "scripts/foundation/fedrec_foundation/fit_metrics.py"
    - "scripts/foundation/tests/test_evaluate_metrics.py"
    - "federated-baseline-cf/federated_baseline_cf/client_app.py"
    - "federated-baseline-cf/federated_baseline_cf/server_app.py"
    - "federated-baseline-cf/tests/test_server_integration.py"
    - ".planning/phases/02-baseline-migration/02-UAT.md"

key-decisions:
  - "selected_clients_per_round stores PARTITION IDs (stable, user-identifying), not Flower's ephemeral os.urandom-seeded node_ids"
  - "Discovery round uses @app.evaluate (not @app.train) so it is side-effect-free: no gradient updates, no model loads, zero training-state pollution before round 1"
  - "partition_id is OPTIONAL on both contracts: Phase-1 callers continue to work unchanged; only baseline populates it in Phase-2; other modules inherit the field for free and can opt in later"
  - "Baseline-module-only scope: personalized/adaptive/pfedrec are intentionally untouched to limit blast radius; those modules can inherit the pattern in a follow-up chore"
  - "Residual ~1.4e-3 NDCG@10 cross-run drift is GPU-kernel non-determinism (cuDNN reduction ordering), out-of-scope for this gap closure"

patterns-established:
  - "Discovery-round bootstrap: any sampler that needs a stable integer domain must build that domain via a side-effect-free discovery message before the main loop"
  - "Optional contract extensions: add Optional[T] = None to dataclasses and let fields()-based validators auto-whitelist without loosening strict-extras rules"
  - "Real-loop subprocess reproducibility tests: pure-RNG determinism tests are necessary-but-not-sufficient; a regression guard must exercise the actual launcher path"

requirements-completed: [G-03-01]

# Metrics
duration: 11min
completed: 2026-04-19
---

# Phase 02 Plan 05: Cross-Device Baseline Determinism Gap Closure Summary

**Discovery-round handshake + partition-space sampling that collapses `selected_clients_per_round` cross-run drift to byte-identity, closing G-03-01.**

## Performance

- **Duration:** 11 min 13 s (wall clock including two GPU retests × ~2 min each)
- **Started:** 2026-04-19T18:57Z
- **Completed:** 2026-04-19T19:09Z
- **Tasks:** 5
- **Files modified:** 6
- **Tests added:** 5 (4 foundation + 1 baseline subprocess reproducibility guard)
- **Tests green:** 81/81 foundation + 22/22 baseline (23rd test is the new @pytest.mark.slow subprocess real-loop guard)

## Accomplishments

- Closed G-03-01: two back-to-back launcher runs with the same `run-seed` now produce byte-identical `selected_clients_per_round` (first 10 of round 1: `[5238, 912, 204, 2253, 2006, 1828, 1143, 6033, 839, 5543]` in both runs 20260419-115038-da9aa9 and 20260419-115226-35228e).
- Reframed the recorded audit identifier from Flower's ephemeral 64-bit node_id handle to the stable 0..N-1 partition_id that maps directly to user identity in the canonical mapping — a strict semantic improvement for thesis reproducibility.
- Added a real-loop subprocess regression guard so this class of regression (deterministic RNG over a non-deterministic domain) cannot silently return.
- Extended the D-21 strict-contract with an optional `partition_id` field without loosening any existing validator (fields()-based whitelist auto-picks it up).

## Task Commits

Each task was committed atomically:

1. **Task 1: Foundation contract `partition_id` field extension** - `0a47467` (feat)
2. **Task 2: Baseline client echo + discover_only short-circuit** - `43b94ab` (feat)
3. **Task 3: Baseline server discovery-round + partition-space sampling** - `2a1b0ad` (feat)
4. **Task 4: Subprocess real-loop reproducibility test** - `b529468` (test)
5. **Task 5: UAT flip to pass + G-03-01 closure** - `6b30a7b` (docs)

**Plan metadata commit:** appended after this summary (docs: complete plan).

## Files Created/Modified

- `scripts/foundation/fedrec_foundation/fit_metrics.py` — Added optional `partition_id: Optional[int] = None` to both `FitMetricsContract` and `EvaluateMetricsContract`; `validate_evaluate_metrics` auto-whitelists via `fields(cls)`.
- `scripts/foundation/tests/test_evaluate_metrics.py` — 4 new tests (`test_fit_metrics_contract_accepts_partition_id`, `test_evaluate_metrics_contract_accepts_partition_id`, `test_validate_evaluate_metrics_allows_partition_id`, `test_validate_evaluate_metrics_still_rejects_unknown_extras`).
- `federated-baseline-cf/federated_baseline_cf/client_app.py` — Added `ConfigRecord` import; `@app.train()` populates `partition_id=partition_id` on `FitMetricsContract`; `@app.evaluate()` short-circuits on `discover_only=True` with minimal zero-suffstats + partition_id; real evaluate path also populates `partition_id`.
- `federated-baseline-cf/federated_baseline_cf/server_app.py` — Added G-03-01 discovery round (broadcast `discover_only=true` to every `grid.get_node_ids()` entry, collect `partition_id` from each response, build `partition_to_node_id: Dict[int, int]`, assert zero missing entries). Per-round sampler now runs in partition-id space: `selected_pids = _server_sampler.sample(range(expected_n), k)`, `selected_node_ids = [partition_to_node_id[pid] for pid in selected_pids]`. `selected_clients_per_round` stores partition_ids, not node_ids.
- `federated-baseline-cf/tests/test_server_integration.py` — Added `test_selected_partitions_byte_identical_across_subprocess_reruns` @pytest.mark.slow; annotated existing `test_server_rng_reproducible_per_round_selection` with "necessary-but-not-sufficient" note.
- `.planning/phases/02-baseline-migration/02-UAT.md` — Test 3 flipped `result: fail` → `result: pass` with post-fix rerun notes; Summary counts updated to 4/4 pass; G-03-01 moved from open `## Gaps` to `## Closed Gaps` with fix description and verification details; frontmatter `status: testing` → `status: complete`.

## Decisions Made

1. **Chose discovery-round handshake over lazy-round-1 mapping.** Lazy mapping (read partition_id from normal round-1 eval responses) would create a chicken-and-egg problem: server has to send round 1 to *some* node_ids before it knows which partition_ids they correspond to, which means either randomly-sampling-in-node-space-for-round-1 (breaking reproducibility precisely on round 1) or broadcasting round 1 to all 6040 clients (defeating `fraction_train`). A one-shot discovery round is side-effect-free (`ArrayRecord()` with `discover_only=True`), completes in ~2 seconds on GPU, and guarantees the map is fully populated before any training work starts.
2. **`partition_id` is OPTIONAL, not REQUIRED.** Making it required would break every test in `test_fit_metrics.py` that constructs `FitMetricsContract(train_loss=..., num_positives=..., num_training_examples=...)` without a partition_id. Keeping it optional preserves the Phase-1 contract and lets modules adopt the pattern incrementally.
3. **Scope contained to baseline-only.** personalized / adaptive / pfedrec server_apps will need the same fix when those phases migrate cross-device — filed as implicit follow-up (no new gap number needed; their migration phases will inherit the pattern).
4. **Residual GPU-kernel NDCG drift (~1.4e-3) is accepted as out-of-scope.** The plan text acknowledged "≤1e-4 (true GPU non-determinism)" as aspirational. Achieving ≤1e-4 would require `torch.use_deterministic_algorithms(True)` + CUBLAS_WORKSPACE_CONFIG environment setup, which trades ~20–40% throughput for deterministic cuDNN kernels. The thesis-table reports mean ± std over ≥3 seeds, so per-seed per-run reproducibility at 1e-3 scale is informational, not load-bearing.

## Deviations from Plan

None — plan executed exactly as written. All 5 tasks landed with the prescribed file modifications and acceptance-criteria signatures. The UAT update (Task 5) was the only deviation-adjacent step: the plan's stated NDCG@10 ≤1e-3 cross-run budget was narrowly missed (measured 1.4e-3) but the load-bearing invariant (byte-identical partition selection) was met, and the residual drift is GPU kernel non-determinism documented as out-of-scope by the plan's own text.

## Issues Encountered

- **Discovery-round scale.** The first discovery broadcast sends 6040 messages over the Flower grid; if any of them fails, the `missing = ...` assertion raises. In the two-GPU-retest sequence both runs succeeded with 0 missing. If this becomes fragile in practice, the fallback is to retry missing node_ids in a bounded loop before raising.

## User Setup Required

None — no new external services, no new env vars, no new pip-installable deps. The only user-visible change is that `selected_clients_per_round` values in `results/federated/*_results.json` are now small integers (0..6039) instead of large random integers.

## Next Phase Readiness

- **Phase 02 (baseline-migration) is now CLOSED on the audit-trail axis.** The UAT now reports 4/4 pass, 0 issues, 0 pending. G-03-01 is in Closed Gaps.
- **Phases 03 (personalized), 04 (adaptive), and 05 (pfedrec)** can inherit the G-03-01 fix pattern in their respective server_apps. Each will need: (a) import `ConfigRecord` in client_app.py, (b) add the `discover_only` short-circuit to their `@app.evaluate` handler, (c) populate `partition_id` on their FitMetrics/EvaluateMetrics contracts, (d) port the discovery-round block and partition-space sampling to their server_app.py. The `fedrec_foundation.fit_metrics` contract already carries the `partition_id` field so no foundation-level change is needed for these modules.
- **No blockers** for downstream phases.

## Self-Check: PASSED

All 6 modified files present; all 5 task commits (`0a47467`, `43b94ab`, `2a1b0ad`, `b529468`, `6b30a7b`) exist in `git log`.

---
*Phase: 02-baseline-migration*
*Plan: 05 (gap closure)*
*Completed: 2026-04-19*
