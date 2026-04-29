---
status: partial
phase: 06-evaluation-reporting-harness
source: [06-VERIFICATION.md]
started: 2026-04-29T07:30:00Z
updated: 2026-04-29T07:30:00Z
---

## Current Test

[awaiting human testing — all 3 items require live federated runs and/or W&B dashboard access]

## Tests

### 1. W&B project routing — dashboard confirmation

expected: After running any cross-device flwr run for the four modules (baseline, personalized, adaptive, pfedrec), the runs appear in the `federated-cf-cross-device` W&B project (separate from the existing cross-silo project) and DO NOT contaminate the legacy cross-silo project. Per-run page shows nested summary keys under `best/*` and `last/*` namespaces; the legacy `final/*` namespace is fully absent.
result: [pending]

### 2. Cross-silo path isolation — filesystem coexistence

expected: After running a cross-device flwr run for any module, `git status` (or a `find results/federated -maxdepth 1 -type f -newer <pre-run-marker>`) shows that legacy flat-layout cross-silo files under `results/federated/<run_id>_results.json` and `results/federated/<run_id>-manifest.json` are untouched. New cross-device artifacts are written ONLY under `results/federated/<module>/<run_id>/results.json` + `results/federated/<module>/<run_id>/manifest.json`.
result: [pending]

### 3. PFedRec paper_compat reproduction — full 100-round run

expected: A full `flwr run .` on `federated-pfedrec` with the paper_compat preset (100 rounds, latent-dim 32, dual LR, BCE loss) produces a canonical `manifest.json` with:
  - `best.sampled_hr@10` within `0.729 ± 0.02` (target: 0.709 to 0.749)
  - `best.sampled_ndcg@10` within `0.441 ± 0.02` (target: 0.421 to 0.461)
  - `pfr08_verification.passed = true` (D-14 PFR-08 hook)
  - `final_eval_round_index` matches the manifest-encoded `best_round`
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
