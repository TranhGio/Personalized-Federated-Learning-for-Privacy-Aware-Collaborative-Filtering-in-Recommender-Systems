---
status: diagnosed
phase: 04-adaptive-migration-bug-fixes
source: [04-adaptive-migration-bug-fixes-VERIFICATION.md]
started: 2026-04-27T00:00:00Z
updated: 2026-04-27T20:30:00Z
---

## Current Test

number: 2
name: Round-to-round alpha drift (deferred — Test 1 surfaced runtime gaps requiring fix first)
expected: |
  Re-run after gap-closure fix lands; inspect cached _logit_alpha.weight values
  to confirm enable-before-load is effective at runtime.
awaiting: gap closure

## Tests

### 1. End-to-end cross-device flwr run at N=6040
expected: |
  Run `flwr run .` (or `flwr run . --federation local-sim-gpu`) inside
  federated-adaptive-personalized-cf/ with the default config
  (mode=benchmark_cross_device, partition-mode=natural, num-supernodes=6040,
  num-server-rounds=2 or higher, fraction-train=0.01 for a fast smoke test).
  Observe:
  - Process starts 6040 virtual clients without error
  - Discovery round completes (partition_to_node_id built for all 6040)
  - Round 1 + Round 2 both emit per-round sampled_ndcg@10 / sampled_hr@10
  - No benchmark AssertionError ("expected 1 user per client")
  - W&B run logs to project federated-cf-cross-device
  - Result JSON written under results/federated/adaptive/{run_id}_results.json
    with _manifest.module='adaptive' + selected_clients_per_round + best_prototype
result: issue
reported: |
  Run completed end-to-end and produced
  results/federated/adaptive/20260427-132620-eb2d19_results.json with the right
  shape: 6040 supernodes confirmed, partition_mode=natural, mode=benchmark_cross_device,
  module='adaptive', selected_clients_per_round populated with real partition IDs
  (5238/912/204..., 5066/2962/4729...), 2 rounds of metrics emitted, all 4 IMP-2
  fingerprints in manifest, cold_starts counter populated (60+60=120, rate=1.0
  expected for non-overlapping rounds).

  HOWEVER — two D-marked features silently no-op at runtime:

  GAP 1: best_prototype is all zeros (length 128, all entries 0.0)
    - global_prototype_norm: 0.0
    - D-08 fallback fired in snapshot_best_prototype because _global_prototype
      was None at best-round capture (round 1)

  GAP 2: alpha_diagnostics_history is MISSING from result JSON entirely
    - eval_metrics_history rounds 1+2 contain no alpha/* keys
    - train_metrics_history rounds 1+2 only have num_positives/num_training_examples/
      partition_id/round_num/train_loss — no alpha_diagnostics nor user_prototype
severity: major
root_cause: |
  Client-server RecordDict-record contract mismatch.
  - client_app.py:741,747 sends user_prototype + alpha_diagnostics as TOP-LEVEL
    sibling RecordDict records (separate from the strict-contract "metrics" key,
    intentional per D-21).
  - server_app.py:670 only extracts response.content.get("metrics", ...) when
    building fit_results. Sibling records are dropped.
  - Downstream: strategy._aggregate_prototypes reads fit_res.metrics
    (strategy.py:228) — empty → _global_prototype stays None → D-08 fallback
    on snapshot. Same for D-16 alpha aggregator at server_app.py:725.

fix_surface: |
  server_app.py around line 670 must also extract sibling records:

      proto_record = response.content.get(USER_PROTOTYPE_KEY)
      if proto_record is not None:
          proto_dict = dict(proto_record)
          if USER_PROTOTYPE_KEY in proto_dict:
              metrics_dict[USER_PROTOTYPE_KEY] = list(proto_dict[USER_PROTOTYPE_KEY])

      alpha_record = response.content.get("alpha_diagnostics")
      if alpha_record is not None:
          metrics_dict["alpha_diagnostics"] = dict(alpha_record)

  Plus 2 regression tests:
    - test_server_unwraps_user_prototype_sibling_record
    - test_server_unwraps_alpha_diagnostics_sibling_record

verification_implication: |
  ADP-03 + D-16 are NOT actually satisfied at runtime despite source-level grep
  matches and "Complete" status in REQUIREMENTS.md. Recommend reverting
  ADP-03 status to Pending until fix lands and a follow-up run shows
  best_prototype.tolist() != [0]*128 and alpha_diagnostics_history populated.

### 2. Round-to-round alpha drift (ADP-02 runtime proof)
expected: |
  After Test 1 finishes (or with `enable-per-user-alpha=true
  enable-item-perturbation=true num-server-rounds=3`), inspect the cached
  per-partition state:
  ```python
  import torch
  s1 = torch.load('.embedding_cache/{run_id}/partition_0.pt', map_location='cpu', weights_only=True)
  print(s1['_logit_alpha.weight'])  # should NOT be all sigmoid(0.5)=0.62
  print(s1['_item_perturbation.weight'].abs().max())  # should be > 0
  ```
  After two consecutive rounds the values should differ from the heuristic
  initialization, proving the enable-before-load fix is effective at runtime
  (not just at the source-ordering level which is already pinned by unit tests).
result: blocked
reason: |
  Test 2 requires Test 1 to produce non-zero artifacts. Test 1 surfaced the
  RecordDict-sibling-record server-side drop bug (see Test 1 root_cause); fix
  that first, then re-run Test 1 + run Test 2 in the same retest cycle.

## Summary

total: 2
passed: 0
issues: 1
pending: 0
skipped: 0
blocked: 1

## Gaps

### GAP-04-01: Server-side sibling RecordDict records dropped (D-05/D-06/D-16)
status: failed
must_have: 3 (best-round prototype EMA restore) AND alpha_diagnostics_history (D-16)
requirement_ids: [ADP-03]  # D-16 is a Phase-4 design decision, not a numbered requirement
severity: major
evidence: |
  Live run results/federated/adaptive/20260427-132620-eb2d19_results.json:
    - _manifest.best_prototype: [0.0] * 128 (D-08 fallback fired)
    - global_prototype_norm: 0.0
    - alpha_diagnostics_history: NOT PRESENT
    - eval_metrics_history rounds 1+2: no alpha/* keys
fix_files:
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py
  - federated-adaptive-personalized-cf/tests/test_server_integration.py
fix_summary: |
  Around server_app.py:670, after extracting "metrics" MetricRecord, ALSO extract
  the sibling RecordDict records "user_prototype" and "alpha_diagnostics" and
  merge them into metrics_dict so strategy._aggregate_prototypes and the D-16
  server-side aggregator (server_app.py:725) can read them via fit_res.metrics.

  Add 2 regression tests:
    - test_server_unwraps_user_prototype_sibling_record
    - test_server_unwraps_alpha_diagnostics_sibling_record
  Both build a fake response Message with the same RecordDict shape the client
  emits (lines 741, 747 of client_app.py) and assert fit_res.metrics carries
  user_prototype + alpha_diagnostics after the unwrap path.
plan_for: 04.1 (gap-closure phase) OR inline hot-fix
