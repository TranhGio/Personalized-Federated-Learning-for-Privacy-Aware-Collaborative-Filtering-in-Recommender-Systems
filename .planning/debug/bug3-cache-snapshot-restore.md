---
status: investigating
trigger: "Path B fix for Bug 3 — snapshot all client cache files at every best-round update so end-of-run best_round_restore restores both global params AND matching local-state cache"
created: 2026-05-06
updated: 2026-05-07
---

## Current Focus

hypothesis: Path B alone is insufficient — rolling back the cache freezes EVERY user's local state at R83 vintage, but their local was last trained against R{LTR(U)} globals which differ from R83 globals. **Average staleness (R_eval - LTR) is identical between runs (8.84 vs 8.89 rounds), so Path B's snapshot did NOT reduce desynchronization on average.** Sparse users hit hardest by full-pop drop (49% NDCG drop) despite GROWING in-sample at R83 (0.245 → 0.259) — confirms synchronization, not undertraining, dominates. Diagnosis: the cache rollback is directionally correct for ~81% of users but the user/item embedding pair STILL desync because local was trained against historical globals, not R83 globals.

test: implement Alt-A (end-of-training calibration pass — train every partition for 1 local epoch against the restored R83 globals) on top of Path B. This synchronizes every user's local state to R83 globals at eval time.

expecting: full-pop NDCG@10 to track in-sample R83 within 30% (target ≥ 0.20). If Alt-A produces ≥ 0.20, thesis is recoverable; if < 0.18, escalate to decision checkpoint with Alt-B/C/D/E options.

next_action: human-verify checkpoint — user re-runs the 100-round adaptive thesis cell with `final-calibration-enabled=true`.

## Symptoms

expected: full-pop NDCG@10 in `final_metrics.best` should track in-sample peak NDCG@10 within ~30%. Adaptive's R83 in-sample 0.2510 should produce a full-pop ≥ 0.20.
actual: 100-round cold run `20260505-141804-c3bc5d` shows in-sample R83 NDCG@10 = 0.2510, full-pop restored R94 NDCG@10 = 0.0831 (3.02× gap). Per-group all lose to baseline by ≥2×.
errors: none. Restoration step succeeds; produces wrong number because user-embedding cache files are from round-of-last-sample (median ~R50, range R1-R93) while item embeddings get rolled back to R83.
reproduction: inspect `results/federated/adaptive/20260505-141804-c3bc5d/results.json` — final_metrics.best.sampled_ndcg@10 = 0.0831 vs eval_metrics_history R83 = 0.2510.
started: Bug 3 deferred from `resolved-adaptive-cache-prototype-collapse.md`; warm-start sidestepped it. Cold-start 100-round confirmation re-exposed it.

## Eliminated

(symptoms_prefilled — Bug 3 root cause was already established in `resolved-adaptive-cache-prototype-collapse.md`; no further investigation needed)

## Evidence

- timestamp: 2026-05-06
  checked: server_app.py round-loop — confirmed the existing `[CHECKPOINT] New best ...` block at the end of the eval phase is the right hook point (both train and eval `grid.send_and_receive` have returned, `aggregate_evaluate` ran, all clients finished writing their round-N `partition_{pid}.pt` files; `cache_root` is already computed at top of main).
  found: snapshot fires AFTER `best_arrays` is captured AND `snapshot_best_prototype()` is called, so all three pieces of best-round state (global params, prototype, client cache) are captured atomically together.
  implication: Path B is structurally clean; no refactor of the round-loop control flow needed.

- timestamp: 2026-05-06
  checked: client_app.py:198-273 — cache layout is `<module>/.embedding_cache/{run_id}/partition_{pid}.pt + manifest.json`, written via tempfile + os.replace.
  found: under `reuse_cache=False` (the standard run_id-scoped regime that Path B targets), every client writes into the same `cache_root` the server probes for D-13. No sig_<hash> divergence to handle.
  implication: snapshot/restore can copy the entire dir contents (filtering existing snapshot subdirs) without touching client-side code.

- timestamp: 2026-05-06
  checked: pytest tests/ — 86/86 pass after the change (74 baseline + 9 unit tests for the helper + 3 source-integration guards).
  found: snapshot/restore/cleanup contract verified end-to-end at the unit level; integration guards verify the snapshot/restore calls are wired into the right branches in source order.
  implication: code change is safe to commit; only thing left is the empirical verification on the 100-round thesis cell.

- timestamp: 2026-05-07
  checked: human-verify run `20260506-074753-bc134c` (Path B active) vs prior cold-run `20260505-141804-c3bc5d` (no Path B). Compared `final_metrics.best`, `eval_metrics_history`, `selected_clients_per_round`, train/eval losses across both runs.
  found:
    - Path B fired correctly (snapshot + restore + cleanup all executed; `[D-06.5]` markers in run completion).
    - Headline: full-pop NDCG@10 dropped 0.0831 → 0.0563 (-32%); per-group NDCG dropped 49% (sparse), 40% (medium), 17% (dense) — sparse users hit hardest.
    - In-sample R83 sparse NDCG IMPROVED slightly between runs (0.2447 → 0.2589). Sparse-user training is fine — H-3 (undertraining dominates) is FALSIFIED.
    - Determinism check: `selected_clients_per_round` is byte-identical for R1, R2, R3, R83, R85. **Sampling is deterministic.** But R1 train_loss differs by 2e-5 (0.7051 vs 0.7053) and R1 per-group HR@10/sparse differs by +4.5pp. R1 fires BEFORE any Path B code, so the nondeterminism is **pre-existing**, not introduced by Path B. Likely source: `_aggregate_prototypes` at strategy.py:236 computes `sum(p * w for p, w in prototypes_and_weights) / total_weight` — order-dependent FP arithmetic across Ray actor return order, accumulating across 80+ rounds. **Confirmed by source diff**: between commits `0fdc77d` and `b871525`, only `cache_snapshot.py` (new), `server_app.py` (Path B integration), and `tests/test_cache_snapshot.py` changed; `strategy.py`, `task.py`, `client_app.py` unchanged. So the determinism drift is from Ray scheduling, not Path B.
  implication:
    - Path B's mechanical execution is correct; the spec was directionally wrong. The cache rollback freezes every user's local at R83 vintage, but each user's R83-vintage local was trained against their LTR(U)<R83 historical globals, not R83 globals. The user/item embedding pair STILL desync — Path B just changes which historical generation the local belongs to.
    - The in-sample/full-pop gap is fundamentally a synchronization gap, not an undertraining gap. Need to align local to R83 globals AT eval time, not freeze it at R83 vintage.

- timestamp: 2026-05-07
  checked: LTR (last-trained-round) distribution analysis from `selected_clients_per_round` × 6040 users.
  found:
    - Both runs sampled all 6040 users at least once (no completely-cold users at eval time).
    - NEW run LTR distribution at R83 (Path B snapshot moment): R71-R83 = 60.8%, R51-R70 = 19.5%, R1-R50 = 0.6%. Snapshot captures relatively recent local state for most users.
    - NEW run had 1151 users sampled in R84-R85 (post-best). Path B regresses these by replacing fresh local with R83 vintage. The other 4889 users' cache was unchanged (they weren't sampled after R83).
    - **Average staleness identical**: OLD live-cache (R93 - LTR) = 8.89 rounds vs NEW snapshot (R83 - LTR_at_R83) = 8.84 rounds. Path B did not reduce average desynchronization.
  implication: confirms Alt-A is the right move. Calibration pass = 1 epoch on every user against R83 globals = staleness goes to 0 for everyone.

- timestamp: 2026-05-07
  checked: server_app.py round-loop, message construction, ConfigRecord shape; client_app.py @app.train() flow at lines 525-768.
  found:
    - Calibration broadcast can re-use the existing `message_type="train"` flow with a tweaked `ConfigRecord({lr, proximal_mu=0.0, round_num, run_id, reuse_cache, local_epochs_override=N, global_prototype})`.
    - Client step 9b (line 707) saves cache after every train, so a calibration train DOES update the cache — exactly what Alt-A needs.
    - Returned client params can be DISCARDED — we do NOT want to update server-side R83 globals (that would defeat best_round_restore).
    - Cost: 6040 clients × 1 local epoch ≈ same compute as one full in-loop training round at fraction_train=1.0. Wall-clock estimate: ~12 min for calibration + existing ~30-50 min D-06 eval.
  implication: Alt-A is mechanically simple — one new gated block in server_app.py and a one-line override read in client_app.py.

## Resolution

root_cause: end-of-run `best_round_restore` restored ONLY the in-memory GLOBAL params (`arrays = best_arrays` + `strategy._global_prototype = strategy.best_prototype`); each client's on-disk cache (`.embedding_cache/{run_id}/partition_{pid}.pt`) was left at "whatever round that user was last sampled." Under cross-device with N=6040 / fraction_train=0.1 / 100 rounds, ~90% of users had their LOCAL state trained against historical globals incompatible with the rolled-back R83 globals, producing the 3.02× full-pop / in-sample NDCG@10 gap (0.0831 vs 0.2510) in run `20260505-141804-c3bc5d`.

**Path B alone is insufficient** — it simply changes WHICH historical-globals generation the local belongs to (R83-vintage instead of LTR-vintage), but the user/item embedding spaces still desync because local was trained against the GLOBALS-AT-LTR, not R83 globals. Empirical evidence (run `20260506-074753-bc134c`): full-pop NDCG@10 = 0.0563 (worse, not better, than no fix); average staleness 8.84 rounds (same as no-fix's 8.89).

The synchronization fix must align local TO R83 globals at eval time, not just freeze it at R83 vintage. Alt-A does this via a 1-epoch calibration training pass against the restored R83 globals.

fix (Path B):
  1. New helper module `federated_adaptive_personalized_cf/cache_snapshot.py` with three functions:
     - `snapshot_cache(cache_root, round_num)` — copies live cache into a sibling `_best_snapshot_round_{N}/` (single rolling snapshot, removes any prior). Uses `cp -a --reflink=auto` for near-instant CoW copies on ext4/xfs/btrfs; falls back to `shutil.copytree`. Atomic via tmp dir + `os.rename`. Excludes any pre-existing snapshot dirs from the copy so we never nest snapshots.
     - `restore_cache(cache_root, round_num)` — copies the round-N snapshot's files back over the live cache (atomic via per-file `.restoretmp` + `os.replace`). Returns False if no matching snapshot exists.
     - `cleanup_snapshots(cache_root)` — drops all snapshot subdirs at end of run; live cache untouched.
  2. `server_app.py` integration:
     - Snapshot trigger: inside the `[CHECKPOINT] New best sampled_ndcg@10=...` block, AFTER `best_arrays` is captured AND `snapshot_best_prototype()` is called. Guarded by `if checkpoint_rule == "best_round_restore"` so `best_round` and `last_round` runs incur no disk overhead.
     - Restore trigger: inside the existing `best_round_restore` end-of-run block, AFTER `arrays = best_arrays` AND `strategy._global_prototype = strategy.best_prototype`, BEFORE the D-06 extra-eval-round broadcast. Logs a WARNING if the snapshot is missing.
     - Cleanup: at end of `main()` after `wandb.finish()`. Guarded by the same `checkpoint_rule == "best_round_restore"` condition.
  3. Telemetry: `[D-06.5] Snapshotted client cache at round {N} -> _best_snapshot_round_{N}/ (size: {GB} GB, took {sec}s)` and `[D-06.5] Restored client cache from snapshot at round {N} (best_round_restore active)`.

verification:
  - 86/86 tests pass (74 baseline + 9 unit + 3 source-integration).
  - Unit tests cover: missing-cache no-op, single rolling snapshot semantics, snapshot-of-snapshot exclusion, restore overwrites drifted live cache, restore-without-snapshot returns False, cleanup leaves live cache intact, full lifecycle drift→snapshot→drift→restore→cleanup.
  - Source-integration guards: `snapshot_cache(cache_root` appears AFTER `best_metric = current_ndcg` AND inside a `checkpoint_rule == "best_round_restore"` guard; `restore_cache(cache_root` appears AFTER `arrays = best_arrays` AND BEFORE `[D-06] Broadcasting extra eval round`; `cleanup_snapshots(cache_root)` appears in `main()`.
  - End-to-end empirical verification deferred to user's 100-round thesis re-run (see CHECKPOINT block below).

files_changed:
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/cache_snapshot.py  (new helper module — Path B, commit 4d6bf62)
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py  (Path B 3 integration points, commit 4d6bf62; Alt-A calibration block + run-config reads, this commit)
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py  (Alt-A: honor `local_epochs_override` from msg_config, this commit)
  - federated-adaptive-personalized-cf/pyproject.toml  (Alt-A: `final-calibration-enabled` + `final-calibration-epochs` keys, this commit)
  - federated-adaptive-personalized-cf/tests/test_cache_snapshot.py  (Path B unit tests, commit 4d6bf62)
  - federated-adaptive-personalized-cf/tests/test_pyproject_shape.py  (Alt-A: defaults-off guard, this commit)
  - federated-adaptive-personalized-cf/tests/test_server_integration.py  (Alt-A: source-order guard, this commit)
  - federated-adaptive-personalized-cf/tests/test_client_assertion.py  (Alt-A: client honor-override guard, this commit)

## Alt-A — End-of-training calibration pass

Why Alt-A (vs Alt-B/C/D/E):
- **Alt-A (calibration pass) — chosen.** Cheapest experiment that directly tests the synchronization hypothesis. Adds ~12 min wall-clock; no architectural change; fully gated behind a default-off flag so non-thesis runs are unchanged. If Alt-A produces NDCG@10 ≥ 0.20 on full-pop, thesis is recoverable.
- Alt-B (last_round) — abandons best-round semantics; reports in-sample 604-client biased number as headline. Cheap but methodologically weaker for the thesis.
- Alt-C (coverage-aware sampling) — invasive; touches strategy.py / server-side sampler. ~100 LOC. Out of scope without empirical justification (the user explicitly said "don't go beyond Alt-A unless evidence demands").
- Alt-D (raise fraction-train) — purely a config change; would help but doesn't address the synchronization gap directly. Bumps wall-clock proportionally (fraction_train=0.5 → 5x rounds).
- Alt-E (Path B + Alt-A) — what we have. Path B stays per user instruction; Alt-A is added on top. Path B is now belt-and-suspenders: it preserves R83 vintage as a fallback if calibration fails to flip the flag.

Implementation (commits this session):
1. New run-config keys (default off):
   - `final-calibration-enabled` = false
   - `final-calibration-epochs` = 1
2. Server-side: after `restore_cache(...)` (D-06.5) and BEFORE the D-06 extra-eval-round broadcast, IF `final_calibration_enabled and checkpoint_rule in ("best_round_restore", "best_round") and best_round_num > 0`, broadcast a `train` message to ALL `partition_to_node_id.values()` (all 6040 nodes) with msg_config:
   - `lr` (same as in-loop training)
   - `proximal_mu = 0.0` (no proximal term — we are NOT trying to constrain to globals; we are aligning local to them)
   - `round_num = actual_rounds + 1` (calibration happens at the slot the D-06 eval will use)
   - `run_id`, `reuse_cache` (mirror in-loop training)
   - `local_epochs_override = final_calibration_epochs` (per-message override)
   - `global_prototype` (the restored best-round prototype, mirrors in-loop training and D-06 eval)
   Returned client params are intentionally DISCARDED — we do NOT call `aggregate_fit` or update server-side `arrays`. Cache files updated by client step 9b.
3. Client-side: `local_epochs` read at client_app.py:676 now honors `msg_config.get("local_epochs_override", context.run_config["local-epochs"])`. Default behavior unchanged when override absent.
4. Telemetry: `[D-06.7] Broadcasting end-of-training calibration pass to all {N} partitions ...` and `[D-06.7] Calibration pass complete: {success}/{total} clients succeeded ({failed} errors).`

Tests added:
- `test_pyproject_shape.test_final_calibration_keys_default_off` — defaults stay off so existing thesis cells are unchanged.
- `test_server_integration.test_final_calibration_pass_wired_between_restore_and_d06_eval` — source-order guard: calibration must sit between `restore_cache` and the D-06 eval broadcast, gated by `final_calibration_enabled`, using `message_type="train"` and `local_epochs_override`.
- `test_client_assertion.test_client_train_honors_local_epochs_override` — client_app honors per-message override; falls back to context.run_config when absent.

89/89 tests pass (was 86 + 3 Alt-A guards).

## Smoke probe note

Constraint asked for a 2-round smoke at `num-supernodes ≤ 50`. The thesis_crossdevice_main mode hard-asserts `len(grid.get_node_ids()) == 6040` at server_app.py:582 (G-03-01 invariant — no override mechanism). A smaller smoke is therefore not possible inside this mode; running with 20 nodes confirmed the assertion fires correctly (no syntax/wiring failure in calibration code; mode-asserts catches the supernode mismatch as designed). Verification falls back to the source-level guards added above.

## Awaiting user

Run the full 100-round adaptive thesis cell with the calibration flag on:

```bash
RAY_memory_usage_threshold=0.97 \
python scripts/run.py adaptive thesis_crossdevice_main \
  --run-config "final-calibration-enabled=true wandb-enabled=false"
```

Expected wall-clock delta vs prior run: +12 min (calibration broadcast), no other changes. Headline number to inspect: `final_metrics.best.sampled_ndcg@10` from `results/federated/adaptive/<new_run_id>/results.json` and per-group `sampled_ndcg@10/{sparse,medium,dense}`. Compare against the 0.0831 (no fix) and 0.0563 (Path B-only) baselines plus the 0.2013 baseline-method floor.
