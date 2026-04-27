# Federated Movie Recommendation — Cross-Device Migration & Thesis Evaluation

## What This Is

Master's thesis project on **Personalized Federated Learning for Privacy-Aware Collaborative Filtering**, implemented on MovieLens 1M with Flower (flwr) + PyTorch. The repository contains four parallel federated implementations — a lower-bound baseline, the PFedRec (IJCAI-23) calibration baseline, a split-learning personalized baseline, and the thesis contribution (adaptive/hierarchical-conditional alpha with dual-level personalization).

This planning cycle migrates the entire comparative study from its current cross-silo setup (`num-supernodes=5`) to the methodologically defensible cross-device setup used by every published FedRec paper (**1 user = 1 client, N=6040**), then re-runs the thesis evaluation under that corrected protocol.

## Core Value

Under a correct cross-device protocol (1 user = 1 client, N=6040), the adaptive/hierarchical-conditional method must beat all three baselines on NDCG@10 — including on sparse users — while PFedRec reproduces the published reference (HR@10 ≈ 0.70, NDCG@10 ≈ 0.38, within ±2 points).

If the adaptive method does not win under the corrected protocol, the thesis contribution has to be rethought. Methodological correctness is non-negotiable.

## Requirements

### Validated

<!-- Already shipped in the current codebase. Verified by the codebase map at .planning/codebase/. -->

- ✓ Four-module Flower codebase: `federated-baseline-cf`, `federated-pfedrec`, `federated-personalized-cf`, `federated-adaptive-personalized-cf` — existing
- ✓ MovieLens 1M loader with user/item index mapping, leave-one-out splitter — existing
- ✓ Dirichlet and natural per-user partitioners — existing (cross-silo default)
- ✓ BPR-MF / Basic-MF / PFedRec-MLP / DualPersonalizedBPRMF model implementations — existing
- ✓ Split-learning client/strategy plumbing (`get_/set_global_parameters`, `get_/set_local_parameters`, `.embedding_cache/`) — existing
- ✓ W&B logging, leave-one-out + 99-negatives evaluation (`evaluate_ranking_sampled`) — existing
- ✓ Adaptive alpha factory (data_quantity / multi_factor / hierarchical_conditional) + per-user learned alpha / item perturbation / contrastive loss — existing
- ✓ Per-user-group bucketing (sparse/medium/dense) + alpha analysis utilities — existing
- ✓ Centralized baselines (SVD notebook, centralized NCF) — existing; out of scope to modify
- ✓ Codebase map (`.planning/codebase/*.md`) documenting current architecture and known concerns — existing
- ✓ Shared cross-device foundation contract (`fedrec-foundation` package) — validated in Phase 1: canonical ID mapping (6040 users × 3706 items), deterministic LOO split manifest, per-user exclusion set, primary evaluator selector, weight policy, three sha256-namespaced RNG factories, run manifest with composite `foundation_contract_sha256`, federation-level launcher. On disk under `scripts/foundation/` and `data/derived/`; wired as local-path dep into all 4 federated modules.
- ✓ `federated-baseline-cf` migrated to cross-device — validated in Phase 2: `num-supernodes = 6040` + `partition-mode = "natural"` defaults, one-user-per-client benchmark assertion on train/evaluate, FND-06 RNG factories wired into DataLoader + negative sampling, test positive excluded from training negatives, `BaselineFedAvg`/`BaselineFedProx` compute server-side NDCG@10/HR@10 (overall + per-group) via summed sufficient-stats, best-round restore + D-15 double-write manifest. 22/22 baseline + 77/77 foundation tests GREEN. BSL-01..08 satisfied.
- ✓ `federated-personalized-cf` migrated to cross-device split-learning — validated in Phase 3: `num-supernodes = 6040` + `partition-mode = "natural"` defaults, one-user-per-client benchmark assertion, `PersonalizedSplitFedAvg`/`FedProx` with sufficient-stat `aggregate_evaluate` + `aggregate_fit` inherited unchanged (D-23), BPRMF/BasicMF collapsed to single-row `nn.Parameter` (no more `num_users × d` ghost table; disk payload per client ~516 B vs ~3 MB), run-namespaced manifest-sidecar embedding cache with hard-fail signature mismatch, FND-06 RNG factories wired into DataLoader + negative sampling, held-out test item excluded from training negatives via FND-03 ExclusionTable, discovery-round + partition-id-space seeded sampling, D-27 best-round restore + D-15 double-write manifest (`module="personalized"`), D-13 cold-start counter. 34/34 personalized + 81/81 foundation tests GREEN. PSN-01..07 satisfied.
- ✓ `federated-adaptive-personalized-cf` migrated to cross-device + adaptive bug-fix surface — validated in Phase 4 (verification re-passed 2026-04-28 after GAP-04-01 hot-fix): `num-supernodes = 6040` + `partition-mode = "natural"` defaults, mode default flipped to `benchmark_cross_device`, one-user-per-client benchmark assertion in train+evaluate, `AdaptiveSplitFedAvg`/`FedProx` with `aggregate_fit` overridden to run prototype EMA + sufficient-stat `aggregate_evaluate`, ADP-02 enable-before-load ordering fix (`_apply_enable_before_load` runs BEFORE `_load_local_user_state` so `_logit_alpha.weight` + `_item_perturbation.weight` are loaded into `_LOCAL_PARAMS`), schema_v2 cache including `_logit_alpha.weight` + `_item_perturbation.weight` + `personal_mlp.*` + `fusion_layer.*` keys, FND-03 exclusion folded into both training negative sampling AND eval negative pool, FND-06 RNG factories wired throughout, D-05 best_prototype snapshot at best-metric fire + D-07 paired restore before final broadcast + D-06 best_prototype embedded in result `_manifest`, D-13 cold-start counter, D-15 manifest double-write with `module="adaptive"`, D-16 alpha diagnostics aggregate (alpha_mean/std/p25/p50/p75/clip_hit_rate weighted-averaged across contributing clients per round), D-02 NotImplementedError frozen-cross-silo guard, GAP-04-01 RecordDict-sibling-record extraction (`_extract_sibling_records` helper merges `user_prototype` + `alpha_diagnostics` siblings into `metrics_dict` so the prototype EMA + alpha aggregator actually populate at runtime), subprocess determinism regression guard for schema_v2 cache + best_prototype byte-identity. 63/63 adaptive + 81/81 foundation + 22/22 baseline + 34/34 personalized = 200 GREEN tests (no cross-phase regression). Live runtime evidence in result `20260427-165100-e8a31d_results.json`: best_prototype non-zero (norm=0.000232), alpha_diagnostics_history populated for rounds 1+2. ADP-01..08 satisfied.

### Active

<!-- Hypotheses driving this planning cycle. -->

**Cross-device migration (Milestone 1):**

- [ ] All four Flower modules run with `num-supernodes = 6040` and `partition-mode = "natural"` (1 user = 1 client) as the primary configuration — `federated-baseline-cf` DONE in Phase 2, `federated-personalized-cf` DONE in Phase 3, `federated-adaptive-personalized-cf` DONE in Phase 4 (remaining: Phase 5 PFedRec)
- [ ] Per-round client sampling fraction C is treated as a hyperparameter and swept; defaults chosen per module match the published FedRec protocol the module calibrates against
- [ ] A single standardized evaluation harness (leave-one-out + 99 negatives, NCF protocol) is used across all four modules so cross-module comparisons are apples-to-apples
- [ ] Per-user-group metrics (sparse 0–30 / medium 30–100 / dense 100+) reported for every run
- [ ] Known `federated-pfedrec` bugs (tracked in `.planning/codebase/CONCERNS.md`) are re-discovered from the IJCAI-23 reference and fixed as part of the migration — we do NOT trust the prior note list
- [ ] Test-positive-leaks-into-training-negatives bug fixed in all four modules (`user_rated_items` must include the held-out test item) — DONE for `federated-baseline-cf` in Phase 2, DONE for `federated-personalized-cf` in Phase 3, DONE for `federated-adaptive-personalized-cf` in Phase 4 (ADP-05 — task.py:499-503 + 1068-1100); remaining: Phase 5 PFedRec
- [x] Per-user learned alpha and item perturbation actually accumulate across rounds (fix `enable_per_user_alpha` / `enable_item_perturbation` ordering vs `load_local_user_embeddings`) — DONE in Phase 4 (ADP-02): `_apply_enable_before_load` fires BEFORE `_load_local_user_state` at client_app.py:591-630; schema_v2 cache round-trips `_logit_alpha.weight` + `_item_perturbation.weight`; live runtime evidence in result 20260427-165100-e8a31d (alpha_diagnostics_history populated)
- [ ] Early stopping checkpoints best-round parameters instead of reporting last-round metrics
- [ ] Server-level seed set so per-round client selection and evaluation negative sampling are reproducible — DONE for `federated-baseline-cf` in Phase 2, DONE for `federated-personalized-cf` in Phase 3, DONE for `federated-adaptive-personalized-cf` in Phase 4 (ADP-06 + subprocess determinism guard); remaining: Phase 5 PFedRec
- [ ] Embedding cache is experiment-scoped (doesn't silently contaminate across runs with different hyperparameters) — DONE in Phase 4 for adaptive (run-namespaced + 12-field schema_v2 signature with hard-fail mismatch); remaining: Phase 5 PFedRec

**Reproduction (Milestone 1):**

- [ ] Flower `federated-pfedrec` reproduces IJCAI-23 PFedRec on ML-1M within ±2 points on HR@10 and NDCG@10 at 100 rounds, dim=32, SGD(lr=0.1), BCE, 1 local epoch, 4 training negatives

**Thesis evaluation (Milestone 2):**

- [ ] All four modules evaluated under one standardized cross-device config (dim, optimizer, negatives, local epochs, rounds, eval protocol) for the main comparison table
- [ ] Adaptive method (`model-type=dual alpha-method=hierarchical_conditional`) beats all three baselines on overall NDCG@10
- [ ] Adaptive method beats all three baselines on **sparse-user** NDCG@10 (thesis claim is strongest here)
- [ ] Ablations for each adaptive ingredient: hierarchical-conditional vs multi-factor vs data-quantity alpha; per-user learned alpha on/off; item perturbation on/off; contrastive λ on/off; dual-level fusion (add/gate/concat)
- [ ] Results exported to `results/federated/` with full experiment metadata and logged to a dedicated **cross-device** W&B project (separate from existing cross-silo runs)

### Out of Scope

- **Differential privacy / privacy quantification** — Thesis focuses on utility under cross-device. DP is future work; explicitly excluded from success criteria.
- **Rewriting centralized baselines (SVD, centralized NCF)** — Already correct; kept as-is. We do NOT re-evaluate them under LOO+99neg because re-running under leave-one-out adds noise without changing the relative comparison.
- **Datasets other than MovieLens 1M** — Scope is fixed to ML-1M for reproducibility vs the FedRec literature. Generalization to other datasets is future work.
- **Production hardening** (retries, monitoring, deploy infra) — This is a research codebase; correctness for reported metrics is the priority.
- **Refactoring the four-way code duplication into `fedrec_common/`** — Valuable but orthogonal; deferred until after the main experiments land. If done early it risks invalidating the codebase map and the bug audit.
- **Cross-silo results as a reported result line** — Kept in the appendix at most; not a main comparison target.

## Context

- **Brownfield project.** The repository already contains four working Flower FL modules, a codebase map (`.planning/codebase/`), a CLAUDE.md with module-level architecture notes, and digested paper summaries (`Papers/digested/`). A prior migration plan exists at `docs/superpowers/plans/2026-04-04-cross-device-migration.md` but has not been executed.
- **Known concerns already cataloged.** `.planning/codebase/CONCERNS.md` enumerates the cross-silo methodological problem, six PFedRec reference-divergence points, nine individual PFedRec bugs, the test-positive-leak pattern across all four modules, and the per-user-learned-alpha accumulation bug. This milestone treats that document as a hypothesis list — each item is re-verified against the IJCAI-23 reference before being declared a bug.
- **Paper knowledge base** at `Papers/digested/` covers the direct prior art (PFedRec, FedPer, FedProx, NCF, SCAFFOLD, adaptive FedOpt, personalized FedRec survey, device-recommendation survey, LightFR, LightGCN, FedCA, DualPersonalized). Read these before inventing approaches.
- **Compute profile.** Local research machine (has at least one recent NVIDIA GPU; the `get_device()` helper has an RTX 5090 / old-PyTorch compatibility fallback). A full 6040-client run with C=0.01 is routine; C=0.1 is feasible but slower.
- **Reference implementation preserved.** `IJCAI-23-PFedRec/` is unmodified upstream PFedRec code used as a calibration oracle for the Flower re-implementation.

## Constraints

- **Tech stack**: Flower (flwr) ≥ 1.22.0, PyTorch ≥ 2.7.1, Python ≥ 3.9 — Fixed by existing code; changing is out of scope for this cycle.
- **Dataset**: MovieLens 1M only (6,040 users / 3,706 items / ~1M ratings) — Thesis scope; generalization deferred.
- **Evaluation protocol**: Leave-one-out + 99 negative samples (NCF protocol), NDCG@10 as the primary metric — Convention in the FedRec literature; required for apples-to-apples comparison.
- **Timeline**: Soft thesis deadline — Prioritize reproduction + thesis-contribution evaluation over nice-to-haves like shared-code refactoring.
- **Hardware**: Single-machine Flower simulation with 6,040 virtual clients — Blocks any design that assumes real distributed edge devices.
- **Backwards compatibility**: Cross-silo configs must continue to run (as an explicit opt-in) so existing W&B runs remain reproducible and appendix results can be regenerated if needed — We override defaults, we do not delete the code paths.
- **Tracking**: A new W&B project is used for cross-device runs to keep the run list clean and to avoid accidentally mixing cross-silo and cross-device numbers in comparison plots.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Migrate to cross-device (1 user = 1 client, N=6040) for all four modules | Every published FedRec paper uses cross-device; cross-silo with num-supernodes=5 is not defensible at thesis review | — Pending |
| Keep all four module directories; do not collapse into shared `fedrec_common/` yet | Refactor risks invalidating the codebase map and bug audit mid-experiment | — Pending |
| Re-discover PFedRec bugs from the IJCAI-23 reference instead of trusting the prior note list | Prior notes may be stale or partial; IJCAI reference is ground truth | — Pending |
| Per-round sampling fraction C treated as a hyperparameter (swept), not fixed | Trade-off between rounds-to-converge and compute-per-round is module-dependent; let the data decide | — Pending |
| New W&B project for cross-device runs | Keeps cross-silo historical runs untouched; avoids mixing incomparable numbers in dashboards | — Pending |
| Per-user-group breakdown (sparse/medium/dense) is a first-class reported metric | Thesis claim about the adaptive method is strongest on sparse users; overall NDCG alone can hide the effect | — Pending |
| Centralized baselines (SVD, centralized NCF) remain as-is | They are already correct; re-running under LOO+99neg adds noise without changing the story | — Pending |
| DP / privacy quantification deferred to future work | Scope containment for a soft-deadline thesis; utility under cross-device is the primary novelty | — Pending |
| Research agents route through the Codex MCP server | User preference: Codex MCP is already configured in this session and gives better research outputs than web search alone | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-28 after Phase 4 (Adaptive Migration & Bug Fixes) completion — ADP-01..08 shipped, `federated-adaptive-personalized-cf` now cross-device with all thesis-contribution machinery active by default: hierarchical-conditional alpha + per-user learned alpha + item perturbation + dual-level personalization + global prototype EMA + best-round prototype snapshot/restore + alpha diagnostics aggregation. GAP-04-01 (server-side RecordDict-sibling drop) discovered + hot-fixed during UAT (commit a03f7bf). 200 GREEN tests across all 4 modules with no cross-phase regression. Live runtime evidence in `results/federated/adaptive/20260427-165100-e8a31d_results.json`. Phase 5 (PFedRec migration & reproduction) is next — only remaining migration before the evaluation harness (Phase 6) and thesis evaluation run (Phase 7).*
