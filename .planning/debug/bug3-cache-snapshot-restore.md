---
status: awaiting_human_verify
trigger: "Path B fix for Bug 3 — snapshot all client cache files at every best-round update so end-of-run best_round_restore restores both global params AND matching local-state cache"
created: 2026-05-06
updated: 2026-05-06
---

## Current Focus

hypothesis: Path B implemented — snapshot in `[CHECKPOINT] New best` branch (only under `best_round_restore`), restore in best_round_restore branch BEFORE the D-06 extra-eval-round, cleanup at end of main(). Helper module is self-contained (no torch/flwr imports), unit-tested.
test: 86/86 pytest including 12 new tests (9 unit + 3 source-level integration guards).
expecting: when user re-runs the 100-round adaptive thesis cell, `final_metrics.best.sampled_ndcg@10` should track the in-sample peak round NDCG within ~30% (was 3.02× off).
next_action: human-verify checkpoint — user runs the 100-round cold thesis cell.

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

## Resolution

root_cause: end-of-run `best_round_restore` restored ONLY the in-memory GLOBAL params (`arrays = best_arrays` + `strategy._global_prototype = strategy.best_prototype`); each client's on-disk cache (`.embedding_cache/{run_id}/partition_{pid}.pt`) was left at "whatever round that user was last sampled." Under cross-device with N=6040 / fraction_train=0.1 / 100 rounds, ~90% of users had their LOCAL state from rounds incompatible with the rolled-back GLOBAL state, producing the 3.02× full-pop / in-sample NDCG@10 gap (0.0831 vs 0.2510) in run `20260505-141804-c3bc5d`.

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
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/cache_snapshot.py  (new helper module)
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py  (3 integration points)
  - federated-adaptive-personalized-cf/tests/test_cache_snapshot.py  (new tests)
