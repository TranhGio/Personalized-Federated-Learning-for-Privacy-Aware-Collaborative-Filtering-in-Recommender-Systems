---
status: testing
phase: 02-baseline-migration
source:
  - 02-baseline-migration-01-SUMMARY.md
  - 02-baseline-migration-02-SUMMARY.md
  - 02-baseline-migration-03-SUMMARY.md
  - 02-baseline-migration-04-SUMMARY.md
  - 02-baseline-migration-VERIFICATION.md (human_verification items)
started: 2026-04-19T15:45:00Z
updated: 2026-04-19T16:10:00Z
---

## Current Test

number: 2
name: End-to-end via `scripts/run.py` launcher (GPU)
expected: |
  Run the canonical cross-device launcher via scripts/run.py, which exercises a
  different code path than `flwr run .` — the launcher applies the mode resolver
  at federation level, generates the run_id, and writes the sibling manifest.
  This confirms the Phase 2 D-25 "launcher is canonical entry point" contract.
awaiting: user response

## GPU Tuning Notes (applied before Test 1)

The `local-sim-gpu` federation block in `federated-baseline-cf/pyproject.toml`
was updated for aggressive cross-device parallelism:

| Setting | Before | After | Rationale |
|---------|--------|-------|-----------|
| `num-gpus` per client | 0.2 (5 concurrent) | **0.05 (20 concurrent)** | BPR-MF is tiny (~15-50MB VRAM/client); consumer GPU ≥8GB VRAM fits 20+ easily |
| `num-cpus` per client | 12 (CPU-starved at N>1) | **1** | One thread per client for dataloader/neg sampling; avoids CPU-count starvation |

With `fraction-train=0.01` (60 clients/round) and 20 concurrent, each round is
only ~3 serial GPU batches → very fast on any modern CUDA card.

If you have multiple GPUs, Flower auto-spans them (no config change needed).
For a larger model (NCF, dim=256), bump `num-gpus` back up to 0.1-0.2.

## Tests

### 1. GPU Smoke Test — fast path, `local-sim-gpu` federation
expected: |
  Run this from the repo root:

  ```bash
  cd federated-baseline-cf
  flwr run . local-sim-gpu \
      --run-config "num-server-rounds=2 fraction-train=0.01 model-type=bpr local-epochs=1"
  ```

  What's happening:
  - Uses the GPU federation (`local-sim-gpu`) → clients allocated 0.05 GPU each
  - 2 rounds × ~60 clients/round × 20 concurrent = ~6 GPU batches total
  - `model-type=bpr` exercises the thesis-relevant ranking loss path
  - `local-epochs=1` keeps each client's local training short

  Expected observations:
  - Flower spawns 6040 supernodes (visible in startup log)
  - `partition-mode=natural` default visible in the run config dump
  - GPU activity shows in `nvidia-smi` during training (run in a second terminal: `watch -n 1 nvidia-smi`)
  - Round 1 + Round 2 both emit `sampled_ndcg@10`, `sampled_hr@10`, and per-group
    sparse/medium/dense sums
  - No `AssertionError` from the one-user-per-client benchmark assertion
  - No CUDA OOM, no crash
  - Total wall time: ≤ 3 minutes on a modern GPU

  Troubleshooting:
  - If CUDA OOM: bump `num-gpus` to 0.1 in pyproject.toml (halves concurrency, doubles per-client memory budget)
  - If runs on CPU instead of GPU: check `nvidia-smi`; make sure PyTorch sees CUDA (`python -c "import torch; print(torch.cuda.is_available())"`)
  - If hang at "spawning 6040 supernodes": Flower needs a moment to register all actors; give it 30-60s before suspecting a hang
result: pass
notes: |
  Passed on second attempt with `mode=benchmark_cross_device` added to --run-config.
  First attempt (run_id 20260419-085653-63e04c) used pyproject default mode
  (cross_silo_legacy) and produced a misleading manifest; second attempt
  (run_id 20260419-090228-08262a) cleanly reports num_supernodes=6040,
  partition_mode=natural, mode=benchmark_cross_device. Training signal: best-round
  restore picked round 2 (sampled_ndcg@10=0.079 > round 1's 0.025); full-eval
  sampled_ndcg@10=0.064 vs random-baseline ~0.10-NDCG-territory → plausible for
  2-round BPR smoke. Per-group sufficient-stat accounting checks out
  (sparse+medium+dense sums to total evaluated_users per round).

### 2. End-to-end via `scripts/run.py` launcher (GPU)
expected: |
  Chore commit 848529e landed two launcher fixes:
  1. federated-baseline-cf pyproject default federation flipped to local-sim-gpu
  2. scripts/run.py: --federation CLI arg added; hardcoded "local-simulation" removed

  Launcher now uses the module's pyproject default when --federation is omitted.
  For baseline, that default is now local-sim-gpu → GPU by default.

  Run (no --federation needed):

  ```bash
  python scripts/run.py baseline benchmark_cross_device \
      --run-config "num-server-rounds=2 fraction-train=0.01 model-type=bpr"
  ```

  (Opt into CPU explicitly with `--federation local-simulation` if you want to
  sanity-check the fallback path — not required for pass.)

  Expected artifacts:
  - Fresh `results/federated/{run_id}_results.json` + sibling `-manifest.json`
  - Manifest top-level matches Test 1 Run 2 shape: `mode=benchmark_cross_device`,
    `num_supernodes=6040`, `partition_mode=natural`, `weight_policy=num_positives`
  - `_manifest.overrides` captures runtime deltas (fraction_train=0.01, etc.)
  - 2-round training completes on GPU (visible in `nvidia-smi`)
  - `checkpoint.rule=best_round_restore`, `best_round` ∈ {1, 2}
  - `selected_clients_per_round` is a list-of-lists (2 × 60 clients)
  - `sampled_hr@10` / `sampled_ndcg@10` appear as headline metrics

  What this confirms beyond Test 1:
  - scripts/run.py subprocess path works end-to-end (D-25 "canonical launcher")
  - Launcher injects `mode=benchmark_cross_device` automatically from positional arg
    (no need to repeat it in --run-config)
result: [pending]

### 3. Determinism — same seed, byte-identical selections (GPU)
expected: |
  Re-run the Test 2 command with the same seed — two runs back-to-back:

  ```bash
  python scripts/run.py baseline benchmark_cross_device \
      --federation local-sim-gpu \
      --run-config "num-server-rounds=2 fraction-train=0.01 model-type=bpr run-seed=42"
  # capture run_id_1 from console output

  python scripts/run.py baseline benchmark_cross_device \
      --federation local-sim-gpu \
      --run-config "num-server-rounds=2 fraction-train=0.01 model-type=bpr run-seed=42"
  # capture run_id_2
  ```

  Compare selected clients across both runs:

  ```bash
  python -c "
  import json, glob
  files = sorted(glob.glob('results/federated/*_results.json'))[-2:]
  a, b = [json.load(open(f)) for f in files]
  print('files:', files)
  print('selected_clients match:', a['selected_clients_per_round'] == b['selected_clients_per_round'])
  print('ndcg@10 close:', abs(a['final_metrics']['sampled_ndcg@10'] - b['final_metrics']['sampled_ndcg@10']) < 1e-9)
  "
  ```

  Expected: both booleans `True`. `selected_clients_per_round` must be byte-identical
  (server_rng determinism); `ndcg@10` should also match to within FP tolerance since
  eval negatives are seeded per (user, round).

  GPU non-determinism note: if `ndcg@10` differs slightly (~1e-4), that's expected
  from CUDA kernel non-determinism — the CLIENT SELECTION match is the actual
  thesis-relevant invariant.
result: [pending]

### 4. W&B project is `federated-cf-cross-device`
expected: |
  After any run with `wandb-enabled=true`, open https://wandb.ai/ and confirm the
  run appears under project `federated-cf-cross-device` (NOT the old cross-silo
  project name). Each round logs `round/selected_clients`, plus overall + per-group
  `sampled_hr@10` / `sampled_ndcg@10` sums.

  Skip this test if W&B is offline or disabled locally.
result: [pending]

## Summary

total: 4
passed: 1
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps

[none yet]
