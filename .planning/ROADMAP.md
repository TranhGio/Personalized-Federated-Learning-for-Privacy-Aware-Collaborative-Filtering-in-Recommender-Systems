# Roadmap: Federated Movie Recommendation — Cross-Device Migration & Thesis Evaluation

**Created:** 2026-04-19
**Granularity:** standard
**Phases:** 7
**Coverage:** 52/52 v1 requirements mapped
**Milestone framing:** M1 = Phases 1–6 (cross-device migration + PFedRec reproduction + eval harness); M2 = Phase 7 (thesis evaluation under the standardized protocol).

## Core Value

Under a correct cross-device protocol (1 user = 1 client, N=6040), the adaptive/hierarchical-conditional method beats all three baselines on NDCG@10 — including on sparse users — while the Flower PFedRec reproduces the published IJCAI-23 reference within ±2 points.

## Phases

- [x] **Phase 1: Foundation Contract** — Shared cross-device protocol (ID mapping, LOO manifest, exclusion set, primary evaluator, weight policy, seeding, run manifest) consumed by all four modules. (completed 2026-04-19)
- [x] **Phase 2: Baseline Migration** — `federated-baseline-cf` moved to cross-device with seeded sampling, sufficient-stat metrics, and protocol fingerprint. (completed 2026-04-19)
- [x] **Phase 3: Personalized Migration** — `federated-personalized-cf` (split-learning) moved to cross-device with run-namespaced cache and one-user client semantics. (completed 2026-04-20)
- [x] **Phase 4: Adaptive Migration & Bug Fixes** — `federated-adaptive-personalized-cf` (thesis module) moved to cross-device; per-user alpha / item perturbation / prototype accumulation bugs fixed. (completed 2026-04-28)
- [ ] **Phase 5: PFedRec Migration & Reproduction** — `federated-pfedrec` re-audited against IJCAI-23 reference, migrated, and reproduces published HR@10 / NDCG@10 within ±2 points.
- [ ] **Phase 6: Evaluation & Reporting Harness** — Best-round restore, per-user-group metrics, protocol fingerprint manifests, dedicated cross-device W&B project.
- [ ] **Phase 7: Thesis Evaluation Run** — Standardized cross-device comparison + ablations; adaptive beats all baselines on overall and sparse-user NDCG@10.

## Phase Details

### Phase 1: Foundation Contract
**Goal**: A single shared cross-device protocol contract — canonical ID mapping, deterministic LOO split, exclusion set, primary evaluator choice, weight policy, seeding discipline, and run manifest — exists on disk and is ready to be imported by every downstream module.
**Depends on**: Nothing (foundation).
**Requirements**: FND-01, FND-02, FND-03, FND-04, FND-05, FND-06, FND-07
**Success Criteria** (what must be TRUE):
  1. A single canonical `raw_user_id → user_idx` / `raw_item_id → item_idx` artifact exists under `.planning/artifacts/` (or equivalent) and any module that imports it observes the same indices for the same raw IDs.
  2. Running the split builder twice produces the same `split_hash` and the same held-out test item per user (deterministic tiebreaking), and `exclude_items[user]` always contains that held-out item.
  3. A config flag picks ONE primary evaluator (`sampled_loo_99`) and ONE aggregation `weight-policy` per module, both of which appear in the run manifest alongside partition mode, num-supernodes, fractions, seeds, and negative counts.
  4. Setting a single run seed produces identical server-selected client sequences and identical evaluator negative samples across two back-to-back runs of the same config; no evaluator internally calls `random.seed(...)` or `np.random.seed(...)`.
**Plans**: 6 plans
  - [x] 01-foundation-contract-01-PLAN.md — Wave-0 package scaffolding + 14 test stubs (gates Plans 02-05)
  - [x] 01-foundation-contract-02-PLAN.md — FND-01 mapping + FND-02 split manifest + FND-03 exclusion set + atomic bundle publication
  - [x] 01-foundation-contract-03-PLAN.md — FND-04 primary evaluator + FND-05 weight-policy + FitMetricsContract (CR-4)
  - [x] 01-foundation-contract-04-PLAN.md — FND-06 three RNG factories (CR-3) + FND-07 run manifest (IMP-2 composite hash)
  - [x] 01-foundation-contract-05-PLAN.md — Mode resolver (D-06..D-11) + scripts/run.py launcher (CR-2)
  - [x] 01-foundation-contract-06-PLAN.md — Add fedrec-foundation as local-path dep to all 4 modules + cross-module smoke test

### Phase 2: Baseline Migration
**Goal**: `federated-baseline-cf` runs as a correct cross-device benchmark — 6040 clients, one user per client in benchmark mode, seeded sampling, sufficient-statistic metrics, test-positive excluded from training negatives, and protocol fingerprint logged.
**Depends on**: Phase 1.
**Requirements**: BSL-01, BSL-02, BSL-03, BSL-04, BSL-05, BSL-06, BSL-07, BSL-08
**Success Criteria** (what must be TRUE):
  1. `flwr run .` inside `federated-baseline-cf/` spawns 6040 supernodes under `partition-mode = "natural"` by default, and the per-round client-loader for each selected node contains exactly one raw user (benchmark assertion passes).
  2. With a fixed run seed, two back-to-back runs select the same client IDs per round and log the same selected-client list, and the sampled evaluator produces the same 99 negatives per (user, round) without reseeding globals.
  3. Running one round with a user whose held-out test item is known shows that test item never appears among the sampled training negatives for that user.
  4. The result artifact for one run contains a protocol fingerprint (partition mode, num-supernodes, fractions, weight policy, primary evaluator, seeds, checkpoint rule) and reports headline NDCG@10 / HR@10 computed once at the server from summed `hit_count@10`, `ndcg_sum@10`, and `evaluated_users` — not from averaged per-client metrics.
**Plans**: 5/5 plans complete
  - [x] 02-baseline-migration-01-PLAN.md — Extend FitMetricsContract (D-22) + BaselineFedAvg/FedProx subclass (D-20) — wave 1 gate for BSL-06
  - [x] 02-baseline-migration-02-PLAN.md — Rip-and-replace dataset.py with foundation adapter (D-17) + pyproject.toml cross-device defaults (BSL-01)
  - [x] 02-baseline-migration-03-PLAN.md — client_app.py one-user assert + FitMetricsContract payload + task.py RNG/exclusion/gradient-mask (BSL-02, BSL-03, BSL-05, BSL-07, D-21, D-22, D-24)
  - [x] 02-baseline-migration-04-PLAN.md — server_app.py mode resolver + seeded sampling + BaselineFedAvg wiring + manifest double-write + best-round restore (BSL-04, BSL-06, BSL-08, D-25, D-26, D-27)
  - [x] 02-baseline-migration-05-PLAN.md — scripts/run.py launcher polish + subprocess determinism regression guard (selected_clients_per_round byte-identity)

### Phase 3: Personalized Migration
**Goal**: `federated-personalized-cf` runs as a correct cross-device split-learning benchmark — 6040 clients, one local user per client, run-namespaced embedding cache, local user row collapsed to a single-user representation, sufficient-stat metrics, and protocol fingerprint logged.
**Depends on**: Phase 1.
**Requirements**: PSN-01, PSN-02, PSN-03, PSN-04, PSN-05, PSN-06, PSN-07
**Success Criteria** (what must be TRUE):
  1. `flwr run .` inside `federated-personalized-cf/` spawns 6040 supernodes under natural partitioning by default, and each client in benchmark mode asserts exactly one local user and fails loudly otherwise.
  2. Two runs with different `embedding-dim` or `split-hash` never reuse each other's cache: the `.embedding_cache/` path for a run is scoped to `run_id/method/num_users/num_items/dim/split_hash`, and a cache load with any mismatched signature field hard-fails instead of partially loading.
  3. A client's local user-state footprint at training time is a single-user row (or keyed lookup), not a `num_users × d` ghost table, and after a round only GLOBAL params (item embeddings, item bias, global bias) are returned to the server while the local user row and bias stay on disk.
  4. Training negatives for a client never include that user's held-out test item, and the result artifact carries the Phase-1 protocol fingerprint with headline metrics computed at the server from sufficient statistics.
**Plans**: 5/5 plans complete
  - [x] 03-personalized-migration-01-PLAN.md — PersonalizedSplitFedAvg/FedProx strategy (D-20, D-23) + BPRMF/BasicMF single-row refactor (D-01, D-03) — Wave 1 parallel
  - [x] 03-personalized-migration-02-PLAN.md — pyproject cross-device defaults (PSN-01) + dataset.py foundation adapter (D-17) + D-02 NotImplementedError — Wave 1 parallel
  - [x] 03-personalized-migration-03-PLAN.md — client_app.py + task.py contract wire + D-04..D-10 manifest-sidecar cache (PSN-02, PSN-03, PSN-05, PSN-06) — Wave 2
  - [x] 03-personalized-migration-04-PLAN.md — server_app.py main loop + discovery round + partition-id sampling + D-13 cold-start + D-15 manifest (PSN-04, PSN-07) — Wave 3
  - [x] 03-personalized-migration-05-PLAN.md — scripts/clean_cache.py + subprocess determinism regression guard (PSN-04/05) — Wave 3

### Phase 4: Adaptive Migration & Bug Fixes
**Goal**: `federated-adaptive-personalized-cf` (thesis contribution) runs as a correct cross-device benchmark AND its per-user learned alpha, item perturbation, and server prototype EMA actually accumulate / restore correctly across rounds.
**Depends on**: Phase 1.
**Requirements**: ADP-01, ADP-02, ADP-03, ADP-04, ADP-05, ADP-06, ADP-07, ADP-08
**Success Criteria** (what must be TRUE):
  1. `flwr run .` inside `federated-adaptive-personalized-cf/` spawns 6040 supernodes under natural partitioning by default, each client asserts exactly one local user in benchmark mode, and training negatives for a user never include that user's held-out test item.
  2. With `enable-per-user-alpha=true` and `enable-item-perturbation=true`, the cached `_logit_alpha.weight` and `item_perturbation` tensors from round N are demonstrably loaded at the start of round N+1 (not re-initialized from the heuristic), e.g. alpha values drift continuously across rounds rather than snapping back to the heuristic each round.
  3. When early stopping restores the best round, the server prototype EMA (`p_global`) restored for the final evaluation equals the EMA at that best round — not the last-round EMA.
  4. The hierarchical-conditional / multi-factor / data-quantity alpha factory produces values in `[0.1, 0.95]` for the documented edge-case user-stats inputs (unit test), and the module logs the Phase-1 protocol fingerprint with server-side sampling seeded and evaluator RNG fixed.
**Plans**: 6 plans
  - [x] 04-adaptive-migration-bug-fixes-01-PLAN.md — AdaptiveSplitFedAvg/FedProx strategy with best_prototype snapshot (ADP-03, ADP-06, D-05) + DualPersonalizedBPRMF enable-before-load fingerprint tests (ADP-02) — Wave 1 parallel
  - [x] 04-adaptive-migration-bug-fixes-02-PLAN.md — pyproject cross-device defaults + schema-v2 signature keys (ADP-01) + dataset.py foundation adapter (D-17) + D-02 NotImplementedError at both entry points — Wave 1 parallel
  - [x] 04-adaptive-migration-bug-fixes-03-PLAN.md — task.py FND-06/FND-03/D-13/D-14 cold-round + D-24 ghost-table isolation (ADP-05, ADP-06 RNG) + client_app.py ADP-02 enable-before-load ordering fix + schema_v2 manifest-sidecar cache + alpha diagnostics (ADP-02, ADP-04, ADP-06 client) — Wave 2
  - [x] 04-adaptive-migration-bug-fixes-04-PLAN.md — ADP-07 alpha factory clip-bounds + HC rule-branch coverage tests against unmodified adaptive_alpha.py — Wave 2 parallel
  - [x] 04-adaptive-migration-bug-fixes-05-PLAN.md — server_app.py AdaptiveSplitFedAvg wire-up + D-05/D-06/D-07 best_prototype snapshot/embed/restore + D-13/D-15/D-16/D-27 + D-02 guard (ADP-03, ADP-06, ADP-08) — Wave 3
  - [x] 04-adaptive-migration-bug-fixes-06-PLAN.md — Subprocess determinism regression guard for schema-v2 cache including _logit_alpha + _item_perturbation + best_prototype byte-identity (ADP-06) — Wave 3

### Phase 5: PFedRec Migration & Reproduction
**Goal**: `federated-pfedrec` is re-audited from scratch against `IJCAI-23-PFedRec/`, divergences are resolved with explicit keep-flower / align-to-reference decisions, the module runs cross-device, and it reproduces the published IJCAI-23 numbers within ±2 points on HR@10 and NDCG@10 under `paper_compat_pfedrec` mode.
**Depends on**: Phase 1. Internal sequence: PFR-02 (reference audit) precedes the coding work.
**Requirements**: PFR-01, PFR-02, PFR-03, PFR-04, PFR-05, PFR-06, PFR-07, PFR-08, PFR-09
**Success Criteria** (what must be TRUE):
  1. A diff table comparing Flower PFedRec to `IJCAI-23-PFedRec/` across aggregation policy, `affine_output.weight` vs `.bias` scope, participation fraction, eval protocol, train negative handling, and early-stopping rule exists in the repository with a keep-flower or align-to-reference decision and rationale for every row.
  2. `flwr run .` inside `federated-pfedrec/` spawns 6040 supernodes under natural partitioning by default, the client path collapses to the single-user branch in benchmark mode (no inner loop over `user_test_items.keys()`), and each user's `(affine_output.weight, affine_output.bias)` is persisted/restored as one atomic per-user artifact keyed by stable `user_idx`; cache loads hard-fail on schema or shape mismatch.
  3. Training negatives are re-sampled every round (not cached across rounds), and a unit test asserts that a user's held-out test positive never appears in that user's sampled training negatives.
  4. Under `paper_compat_pfedrec` mode (dim=32, SGD lr=0.1, BCE, 1 local epoch, 4 training negatives, 100 rounds), the final best-round result artifact reports HR@10 and NDCG@10 within ±2 absolute points of the IJCAI-23 reference (HR@10 ≈ 0.729, NDCG@10 ≈ 0.441), and the Phase-1 protocol fingerprint is attached.
**Plans**: 5 plans
  - [x] 05-pfedrec-migration-reproduction-01-PLAN.md — Strategy + Model: PFedRecSplitFedAvg (D-12) + D-01 bias-GLOBAL flip + D-21 strict=True + D-07 drop FedProx (PFR-02 + PFR-03 partial) — Wave 1 parallel
  - [x] 05-pfedrec-migration-reproduction-02-PLAN.md — pyproject cross-device defaults (PFR-01) + dataset.py foundation adapter + D-09 NotImplementedError + foundation mode.py D-25 (PFR-01, PFR-02 D-25, PFR-09 partial) — Wave 1 parallel
  - [x] 05-pfedrec-migration-reproduction-03-PLAN.md — client_app + task: PFR-05 single-user collapse + FND-03 exclusion + FND-06 RNG + D-04 eval BCE + D-22 cold-round + manifest-sidecar schema_v3 with bias_classification sentinel (PFR-02, PFR-03, PFR-04, PFR-05, PFR-06 client, PFR-07) — Wave 2
  - [ ] 05-pfedrec-migration-reproduction-04-PLAN.md — server_app: G-03-01 discovery + ADP-06 sampler + PFedRecSplitFedAvg wire-up + D-13 cold-start + D-14 PFR-08 auto-verify hook + D-15 manifest module=pfedrec + D-27 best-round-restore (PFR-02, PFR-06 server, PFR-08, PFR-09) — Wave 3 parallel
  - [ ] 05-pfedrec-migration-reproduction-05-PLAN.md — Subprocess determinism regression guard (PFR-06) — Wave 3 parallel

### Phase 6: Evaluation & Reporting Harness
**Goal**: Every module emits best-round metrics from a restored best-round checkpoint, per-user-group (sparse/medium/dense) HR@10 and NDCG@10 as first-class fields, sampling-exposure support counts, and writes results plus a protocol fingerprint manifest to a cross-device-scoped location and W&B project.
**Depends on**: Phases 2, 3, 4, 5 (all per-module migrations).
**Requirements**: EVL-01, EVL-02, EVL-03, EVL-04, EVL-05, EVL-06
**Success Criteria** (what must be TRUE):
  1. For any module, running with early stopping enabled produces a canonical result artifact whose headline `ndcg@10` / `hr@10` come from ONE final evaluation after restoring the best-round global + local + strategy state — not from the last-round in-memory arrays — and the filename / manifest encodes `best_round`.
  2. Every module's result artifact and W&B run contains `ndcg@10/sparse`, `ndcg@10/medium`, `ndcg@10/dense` (plus HR@10 variants) as first-class fields, together with per-group sampling-exposure counts so a reader can interpret per-group metrics with the right variance lens.
  3. Cross-device results are written under `results/federated/<module>/<run_id>/` with the full protocol fingerprint manifest, and the legacy cross-silo result locations under `results/federated/` are not touched or overwritten by any cross-device run.
  4. All four modules log their cross-device runs to a new, dedicated W&B project (separate from the existing cross-silo project), and the canonical reported field is `best_*` with `last_*` preserved only as a diagnostic.
**Plans**: 5 plans
  - [x] 05-pfedrec-migration-reproduction-01-PLAN.md — Strategy + Model: PFedRecSplitFedAvg (D-12) + D-01 bias-GLOBAL flip + D-21 strict=True + D-07 drop FedProx (PFR-02 + PFR-03 partial) — Wave 1 parallel
  - [x] 05-pfedrec-migration-reproduction-02-PLAN.md — pyproject cross-device defaults (PFR-01) + dataset.py foundation adapter + D-09 NotImplementedError + foundation mode.py D-25 (PFR-01, PFR-02 D-25, PFR-09 partial) — Wave 1 parallel
  - [ ] 05-pfedrec-migration-reproduction-03-PLAN.md — client_app + task: PFR-05 single-user collapse + FND-03 exclusion + FND-06 RNG + D-04 eval BCE + D-22 cold-round + manifest-sidecar schema_v3 with bias_classification sentinel (PFR-02, PFR-03, PFR-04, PFR-05, PFR-06 client, PFR-07) — Wave 2
  - [ ] 05-pfedrec-migration-reproduction-04-PLAN.md — server_app: G-03-01 discovery + ADP-06 sampler + PFedRecSplitFedAvg wire-up + D-13 cold-start + D-14 PFR-08 auto-verify hook + D-15 manifest module=pfedrec + D-27 best-round-restore (PFR-02, PFR-06 server, PFR-08, PFR-09) — Wave 3 parallel
  - [ ] 05-pfedrec-migration-reproduction-05-PLAN.md — Subprocess determinism regression guard (PFR-06) — Wave 3 parallel

### Phase 7: Thesis Evaluation Run
**Goal**: Under ONE standardized cross-device comparison config shared by all four modules, the adaptive method beats all three baselines on overall NDCG@10 and on sparse-user NDCG@10; ablations across alpha methods, per-user alpha, item perturbation, contrastive λ, and fusion type are produced with per-group breakdowns, and everything is exported as the thesis tables.
**Depends on**: Phase 6 (and transitively all migrations).
**Requirements**: THS-01, THS-02, THS-03, THS-04, THS-05, THS-06, THS-07
**Success Criteria** (what must be TRUE):
  1. A single `thesis_crossdevice_main` config (dim, optimizer, training negatives, local epochs, rounds, eval protocol, weight policy, fractions, seeds) exists and is invoked unchanged by all four modules; deviations from per-module historical configs are documented in the run notes.
  2. The exported main comparison table at `results/federated/_thesis/` reports mean ± std over ≥3 seeds for HR@10 and NDCG@10 overall and per user group (sparse / medium / dense) for all four modules, and the adaptive module (`model-type=dual alpha-method=hierarchical_conditional`) wins on OVERALL NDCG@10 against baseline, personalized, and PFedRec.
  3. The same comparison on the SPARSE user slice shows the adaptive module winning on sparse NDCG@10 against all three baselines (primary thesis claim).
  4. An ablation table exists for the adaptive module covering {hierarchical_conditional, multi_factor, data_quantity} × {per-user alpha on/off} × {item perturbation on/off} × {contrastive λ ∈ 0, 0.1} × {fusion ∈ add, gate, concat}, each row reporting per-user-group metrics so the "where does the win come from" question is answerable directly from the artifact.
**Plans**: 5 plans
  - [x] 05-pfedrec-migration-reproduction-01-PLAN.md — Strategy + Model: PFedRecSplitFedAvg (D-12) + D-01 bias-GLOBAL flip + D-21 strict=True + D-07 drop FedProx (PFR-02 + PFR-03 partial) — Wave 1 parallel
  - [x] 05-pfedrec-migration-reproduction-02-PLAN.md — pyproject cross-device defaults (PFR-01) + dataset.py foundation adapter + D-09 NotImplementedError + foundation mode.py D-25 (PFR-01, PFR-02 D-25, PFR-09 partial) — Wave 1 parallel
  - [ ] 05-pfedrec-migration-reproduction-03-PLAN.md — client_app + task: PFR-05 single-user collapse + FND-03 exclusion + FND-06 RNG + D-04 eval BCE + D-22 cold-round + manifest-sidecar schema_v3 with bias_classification sentinel (PFR-02, PFR-03, PFR-04, PFR-05, PFR-06 client, PFR-07) — Wave 2
  - [ ] 05-pfedrec-migration-reproduction-04-PLAN.md — server_app: G-03-01 discovery + ADP-06 sampler + PFedRecSplitFedAvg wire-up + D-13 cold-start + D-14 PFR-08 auto-verify hook + D-15 manifest module=pfedrec + D-27 best-round-restore (PFR-02, PFR-06 server, PFR-08, PFR-09) — Wave 3 parallel
  - [ ] 05-pfedrec-migration-reproduction-05-PLAN.md — Subprocess determinism regression guard (PFR-06) — Wave 3 parallel

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation Contract | 6/6 | Complete | 2026-04-19 |
| 2. Baseline Migration | 5/5 | Complete | 2026-04-19 |
| 3. Personalized Migration | 5/5 | Complete | 2026-04-20 |
| 4. Adaptive Migration & Bug Fixes | 6/6 | Complete | 2026-04-28 |
| 5. PFedRec Migration & Reproduction | 0/5 | In progress | - |
| 6. Evaluation & Reporting Harness | 0/0 | Not started | - |
| 7. Thesis Evaluation Run | 0/0 | Not started | - |

## Dependency Graph

```
Phase 1 (FND) ──┬──► Phase 2 (BSL) ──┐
                ├──► Phase 3 (PSN) ──┤
                ├──► Phase 4 (ADP) ──┼──► Phase 6 (EVL) ──► Phase 7 (THS)
                └──► Phase 5 (PFR) ──┘
```

Phases 2, 3, 4, 5 are parallelizable after Phase 1 lands. Within Phase 5, PFR-02 (reference audit) gates the rest of the phase and cannot parallelize with the PFR coding work.

## Coverage

| Category | Count | Phase |
|----------|-------|-------|
| FND-01 – FND-07 | 7 | Phase 1 |
| BSL-01 – BSL-08 | 8 | Phase 2 |
| PSN-01 – PSN-07 | 7 | Phase 3 |
| ADP-01 – ADP-08 | 8 | Phase 4 |
| PFR-01 – PFR-09 | 9 | Phase 5 |
| EVL-01 – EVL-06 | 6 | Phase 6 |
| THS-01 – THS-07 | 7 | Phase 7 |
| **Total** | **52** | — |

All 52 v1 requirements mapped. No orphans. No duplicates.

---
*Roadmap created: 2026-04-19*
*Last updated: 2026-04-28 — Phase 4 verified passed after GAP-04-01 hot-fix (commit a03f7bf); 4/7 phases Complete; Phase 5 (PFedRec migration) is next.*
