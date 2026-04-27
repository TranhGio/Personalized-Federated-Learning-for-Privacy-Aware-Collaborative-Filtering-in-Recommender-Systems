---
status: partial
phase: 04-adaptive-migration-bug-fixes
source: [04-adaptive-migration-bug-fixes-VERIFICATION.md]
started: 2026-04-27T00:00:00Z
updated: 2026-04-27T00:00:00Z
---

## Current Test

number: 1
name: End-to-end cross-device flwr run at N=6040
expected: |
  6040 supernodes spawn under natural partitioning; per-round sampled_ndcg@10
  + sampled_hr@10 logged; no benchmark AssertionError; no CUDA OOM; result JSON
  written with _manifest.module='adaptive' + best_prototype field.
awaiting: user response

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
result: pending

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
result: pending

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
