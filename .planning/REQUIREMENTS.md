# Requirements: Federated Movie Recommendation — Cross-Device Migration & Thesis Evaluation

**Defined:** 2026-04-19
**Core Value:** Under a correct cross-device protocol (1 user = 1 client, N=6040), the adaptive/hierarchical-conditional method beats all three baselines on NDCG@10 — including on sparse users — while the Flower PFedRec reproduces the published IJCAI-23 reference within ±2 points.

## v1 Requirements

Requirements for this planning cycle (migration milestone + thesis-evaluation milestone). Each maps to exactly one roadmap phase.

### Foundation (FND)

Shared protocol contract consumed by all four modules. Everything downstream depends on this.

- [x] **FND-01**: A single canonical `raw_user_id → user_idx` and `raw_item_id → item_idx` mapping artifact is persisted under `.planning/artifacts/` (or equivalent) and imported by every module's `dataset.py`, `client_app.py`, and evaluator.
- [x] **FND-02**: A deterministic leave-one-out split manifest is produced once (stable-sort by `(user_id, timestamp, movie_id)`), persisted with a `split_hash`, and loaded by every module.
- [x] **FND-03**: A per-user exclusion set `exclude_items[user] = train_pos ∪ test_pos` is exposed by the shared data layer and consumed wherever training negatives are sampled.
- [x] **FND-04**: ONE primary evaluation protocol (`sampled_loo_99` = leave-one-out + 99 negatives, NCF) is declared the thesis-table protocol; `allrank_*` is kept as a namespaced secondary.
- [x] **FND-05**: An explicit aggregation `weight-policy` config is introduced (`uniform` / `num_positives` / `num_training_examples`); each module picks one by default and logs it.
- [x] **FND-06**: Run-scoped seeding: Python/NumPy/Torch seeded once per run; a server RNG is derived for client sampling; per-user RNG streams are derived from `(run_seed, user_id, round, purpose)`. No global `random.seed(...)` or `np.random.seed(...)` inside evaluators or round loops.
- [x] **FND-07**: A run manifest (`protocol fingerprint`) is saved with every result artifact: partition mode, `num-supernodes`, client fractions, weight policy, eval protocol, negative counts, seeds, checkpoint rule.

### Baseline Migration (BSL)

Migrate `federated-baseline-cf` to cross-device.

- [x] **BSL-01**: `federated-baseline-cf/pyproject.toml` defaults to `num-supernodes = 6040` and `partition-mode = "natural"`; cross-silo remains as an explicit opt-in via config override.
- [x] **BSL-02**: `client_app.py` asserts exactly one local user per client in benchmark mode.
- [x] **BSL-03**: Training negative sampling uses the FND-03 exclusion set so the held-out test item is NEVER drawn as a training negative.
- [x] **BSL-04**: Server-side `random.sample(node_ids, ...)` is replaced with a seeded RNG derived from the run seed (FND-06); selected client IDs are logged per round.
- [x] **BSL-05**: Sampled evaluator no longer calls `random.seed(seed)`; it accepts a `random.Random` instance seeded from FND-06.
- [x] **BSL-06**: Clients return sufficient statistics (`hit_count@10`, `ndcg_sum@10`, `evaluated_users`) instead of pre-averaged per-client metrics; server computes the final ratio once.
- [x] **BSL-07**: Module-level evaluator path uses only the FND-04 primary protocol for the thesis table; any secondary `allrank_*` call is explicitly namespaced.
- [x] **BSL-08**: Module logs the FND-07 protocol fingerprint alongside results.

### PFedRec Migration & Reproduction (PFR)

Migrate `federated-pfedrec`, close the IJCAI-23 reference gap, and reproduce published numbers.

- [x] **PFR-01**: `federated-pfedrec/pyproject.toml` defaults to `num-supernodes = 6040` and `partition-mode = "natural"`.
- [x] **PFR-02**: Re-audit Flower-vs-`IJCAI-23-PFedRec/` divergence from scratch; produce a diff table covering aggregation policy, `affine_output.weight` vs `.bias` scope, participation fraction, eval protocol, train negative handling, and early-stopping rule. Decide each divergence as `keep-flower` / `align-to-reference` and log the rationale.
- [x] **PFR-03**: PFedRec per-user head (`affine_output.weight` + `affine_output.bias`) is saved and loaded as one atomic artifact per user keyed by stable `user_idx`; cache hard-fails on shape or schema mismatch.
- [x] **PFR-04**: Training negatives exclude the held-out test positive (FND-03); unit test asserts this.
- [x] **PFR-05**: Client-side partition-scope loop over `user_test_items.keys()` is collapsed to a single-user path in benchmark mode.
- [x] **PFR-06**: Server-side sampling, evaluator RNG, sufficient-statistic aggregation match the Foundation contract (FND-04, FND-05, FND-06).
- [x] **PFR-07**: Training negatives are re-sampled every round (not cached across rounds).
- [x] **PFR-08**: Flower PFedRec reproduces the IJCAI-23 reference on ML-1M: HR@10 and NDCG@10 within ±2 points of paper numbers (HR@10 ≈ 0.729, NDCG@10 ≈ 0.441 at round 89), under `paper_compat_pfedrec` mode (dim=32, SGD lr=0.1, BCE, 1 local epoch, 4 training negatives, 100 rounds).
- [x] **PFR-09**: Module logs the FND-07 protocol fingerprint alongside results.

### Personalized Migration (PSN)

Migrate `federated-personalized-cf` (split learning with local user embeddings).

- [x] **PSN-01**: `federated-personalized-cf/pyproject.toml` defaults to `num-supernodes = 6040` and `partition-mode = "natural"`.
- [x] **PSN-02**: Benchmark-mode one-user assertion in `client_app.py`.
- [x] **PSN-03**: Training negatives exclude the held-out test positive (FND-03).
- [x] **PSN-04**: Server-side sampling seeded; evaluator RNG fixed; sufficient-stat metrics (FND-04, FND-05, FND-06).
- [x] **PSN-05**: `.embedding_cache/` path includes run-id + method + num_users + num_items + dim + split_hash; loads hard-fail on mismatch.
- [x] **PSN-06**: Local user-embedding row collapses from shape `(num_users, d)` to a single local row or key lookup, avoiding the global-table ghost.
- [x] **PSN-07**: Module logs FND-07 protocol fingerprint.

### Adaptive Migration & Bug Fixes (ADP)

Migrate `federated-adaptive-personalized-cf` (thesis contribution) and fix the documented per-user-alpha / item-perturbation / prototype bugs.

- [x] **ADP-01**: `federated-adaptive-personalized-cf/pyproject.toml` defaults to `num-supernodes = 6040` and `partition-mode = "natural"`.
- [x] **ADP-02**: `enable_per_user_alpha()` and `enable_item_perturbation()` are called BEFORE `load_local_user_embeddings()` so `_logit_alpha.weight` and `item_perturbation` are in `_LOCAL_PARAMS` at load time; cached per-round values are restored instead of re-initialized.
- [x] **ADP-03**: Server-side prototype EMA (`p_global`) is saved as part of the best-round checkpoint and restored at final evaluation time.
- [x] **ADP-04**: Benchmark-mode one-user assertion in `client_app.py`.
- [x] **ADP-05**: Training negatives exclude the held-out test positive (FND-03).
- [x] **ADP-06**: Server-side sampling seeded; evaluator RNG fixed; sufficient-stat metrics; run-scoped cache (FND-04, FND-05, FND-06 + PSN-05 pattern).
- [x] **ADP-07**: Hierarchical-conditional / multi-factor / data-quantity alpha factory works unchanged in the cross-device setting; a unit test asserts alpha values fall in `[0.1, 0.95]` for edge-case user-stats inputs.
- [x] **ADP-08**: Module logs FND-07 protocol fingerprint.

### Evaluation & Reporting Harness (EVL)

Unified evaluation layer, best-round restore, per-group reporting.

- [x] **EVL-01**: Best-round restore: for every module, `best_*` metrics and the corresponding global + local parameter state are saved; after the last round, the best-round state is restored and ONE final evaluation is written as the canonical result artifact.
- [x] **EVL-02**: Per-user-group (sparse 0–30 / medium 30–100 / dense 100+) NDCG@10 and HR@10 are emitted as first-class fields (`ndcg@10/sparse`, `ndcg@10/medium`, `ndcg@10/dense`, plus HR@10 variants) by every module.
- [x] **EVL-03**: Per-user and per-group sampling-exposure counts are logged each round; reports surface support counts so per-group metrics can be read with the right variance lens.
- [x] **EVL-04**: Results are written to `results/federated/<module>/<run_id>/` with the FND-07 manifest; legacy cross-silo result locations stay untouched.
- [x] **EVL-05**: All cross-device W&B runs log to a NEW W&B project (named `<ENTITY>/thesis-crossdevice-*` or similar), separate from the existing cross-silo project.
- [x] **EVL-06**: Canonical reporting uses `best_*` metrics; `last_*` is kept as a diagnostic field only; result filenames encode `best_round`.

### Thesis Evaluation Run (THS)

Run the standardized cross-device comparison under one shared config and produce the thesis comparison.

- [x] **THS-01**: Define ONE standardized cross-device comparison config shared by all four modules (dim, optimizer, training negatives, local epochs, rounds, eval protocol, weight policy); document deviations from per-module historical configs.
- [x] **THS-02**: Run all four modules under the standardized config, multiple seeds (≥3); produce one comparison table with mean ± std on HR@10, NDCG@10 overall and per user group (sparse/medium/dense).
- [ ] **THS-03**: Adaptive module (`model-type=dual alpha-method=hierarchical_conditional`) beats all three baselines on OVERALL NDCG@10 under the standardized config.
- [ ] **THS-04**: Adaptive module beats all three baselines on SPARSE-user NDCG@10 (thesis claim is strongest here).
- [x] **THS-05**: Ablations: hierarchical-conditional vs multi-factor vs data-quantity alpha; per-user learned alpha on/off; item perturbation on/off; contrastive λ ∈ {0, 0.1}; fusion ∈ {add, gate, concat}.
- [ ] **THS-06**: Ablations also report per-user-group metrics so the "where does the win come from" question is answerable directly from the artifacts.
- [ ] **THS-07**: Thesis comparison table + ablation tables + sparse-user slice are exported as markdown to `results/federated/_thesis/`.

## v2 Requirements

Deferred beyond this thesis cycle.

### Differential Privacy (DP)

- **DP-01**: Add DP-SGD noise to the training loop; quantify the utility-privacy tradeoff.
- **DP-02**: Privacy accounting per round (Rényi DP) exported to results.

### Shared Library Refactor (REF)

- **REF-01**: Extract `dataset.py` / `early_stopping.py` / base model classes into a `fedrec_common/` package; eliminate the 4× byte-duplicated files.
- **REF-02**: Unit-test suite over `fedrec_common/` covering FND-01–07 contracts.

### Extended Dataset Coverage (EXT)

- **EXT-01**: Replicate the thesis comparison on ML-10M (or ML-20M) to show the adaptive method's win generalizes.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Differential privacy in this thesis cycle | Soft deadline; utility under cross-device is the primary novelty. DP is v2. |
| Rewriting centralized baselines (SVD, centralized NCF) | Already correct; re-running under LOO+99neg adds noise without changing the relative comparison. |
| Datasets other than MovieLens 1M | Reproducibility vs FedRec literature is fixed to ML-1M. Generalization is v2. |
| Production hardening (retries, monitoring, deploy infra) | Research codebase; correctness for reported metrics is the priority. |
| Extracting `fedrec_common/` during this cycle | Refactor risks invalidating the codebase map and bug audit mid-experiment; deferred to v2. |
| Cross-silo runs as a primary reported result | Appendix only; not in the main thesis table. |
| Full-rank evaluation as a primary metric | `sampled_loo_99` is the primary; full-rank stays as a namespaced secondary. |
| Real-edge-device deployment of Flower clients | Flower simulation is the target; real-edge is out of scope for a thesis. |
| Changing Flower or PyTorch major versions | Locked by existing code; no stack swap. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FND-01 | Phase 1: Foundation Contract | Complete |
| FND-02 | Phase 1: Foundation Contract | Complete |
| FND-03 | Phase 1: Foundation Contract | Complete |
| FND-04 | Phase 1: Foundation Contract | Complete |
| FND-05 | Phase 1: Foundation Contract | Complete |
| FND-06 | Phase 1: Foundation Contract | Complete |
| FND-07 | Phase 1: Foundation Contract | Complete |
| BSL-01 | Phase 2: Baseline Migration | Complete |
| BSL-02 | Phase 2: Baseline Migration | Complete |
| BSL-03 | Phase 2: Baseline Migration | Complete |
| BSL-04 | Phase 2: Baseline Migration | Complete |
| BSL-05 | Phase 2: Baseline Migration | Complete |
| BSL-06 | Phase 2: Baseline Migration | Complete |
| BSL-07 | Phase 2: Baseline Migration | Complete |
| BSL-08 | Phase 2: Baseline Migration | Complete |
| PSN-01 | Phase 3: Personalized Migration | Complete |
| PSN-02 | Phase 3: Personalized Migration | Complete |
| PSN-03 | Phase 3: Personalized Migration | Complete |
| PSN-04 | Phase 3: Personalized Migration | Complete |
| PSN-05 | Phase 3: Personalized Migration | Complete |
| PSN-06 | Phase 3: Personalized Migration | Complete |
| PSN-07 | Phase 3: Personalized Migration | Complete |
| ADP-01 | Phase 4: Adaptive Migration & Bug Fixes | Complete |
| ADP-02 | Phase 4: Adaptive Migration & Bug Fixes | Complete |
| ADP-03 | Phase 4: Adaptive Migration & Bug Fixes | Complete |
| ADP-04 | Phase 4: Adaptive Migration & Bug Fixes | Complete |
| ADP-05 | Phase 4: Adaptive Migration & Bug Fixes | Complete |
| ADP-06 | Phase 4: Adaptive Migration & Bug Fixes | Complete |
| ADP-07 | Phase 4: Adaptive Migration & Bug Fixes | Complete |
| ADP-08 | Phase 4: Adaptive Migration & Bug Fixes | Complete |
| PFR-01 | Phase 5: PFedRec Migration & Reproduction | Complete |
| PFR-02 | Phase 5: PFedRec Migration & Reproduction | Complete |
| PFR-03 | Phase 5: PFedRec Migration & Reproduction | Complete |
| PFR-04 | Phase 5: PFedRec Migration & Reproduction | Complete |
| PFR-05 | Phase 5: PFedRec Migration & Reproduction | Complete |
| PFR-06 | Phase 5: PFedRec Migration & Reproduction | Complete |
| PFR-07 | Phase 5: PFedRec Migration & Reproduction | Complete |
| PFR-08 | Phase 5: PFedRec Migration & Reproduction | Complete |
| PFR-09 | Phase 5: PFedRec Migration & Reproduction | Complete |
| EVL-01 | Phase 6: Evaluation & Reporting Harness | Complete |
| EVL-02 | Phase 6: Evaluation & Reporting Harness | Complete |
| EVL-03 | Phase 6: Evaluation & Reporting Harness | Complete |
| EVL-04 | Phase 6: Evaluation & Reporting Harness | Complete |
| EVL-05 | Phase 6: Evaluation & Reporting Harness | Complete |
| EVL-06 | Phase 6: Evaluation & Reporting Harness | Complete |
| THS-01 | Phase 7: Thesis Evaluation Run | Complete |
| THS-02 | Phase 7: Thesis Evaluation Run | Complete |
| THS-03 | Phase 7: Thesis Evaluation Run | Pending |
| THS-04 | Phase 7: Thesis Evaluation Run | Pending |
| THS-05 | Phase 7: Thesis Evaluation Run | Complete |
| THS-06 | Phase 7: Thesis Evaluation Run | Pending |
| THS-07 | Phase 7: Thesis Evaluation Run | Pending |

**Coverage:**
- v1 requirements: 52 total
- Mapped to phases: 52 ✓
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-19*
*Last updated: 2026-04-19 after roadmap creation (traceability populated)*
