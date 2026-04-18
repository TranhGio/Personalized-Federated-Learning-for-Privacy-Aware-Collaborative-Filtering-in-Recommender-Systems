# Research Summary — Cross-Device FedRec Migration

**Research date:** 2026-04-19
**Research tool:** Codex MCP (single session, reasoning over the FedRec literature)
**Scope:** Brownfield thesis codebase — four parallel Flower FL modules (baseline, PFedRec re-implementation, split-learning personalized, adaptive/hierarchical-conditional) on MovieLens 1M, migrating from cross-silo (`num-supernodes=5`) to cross-device (`num-supernodes=6040`, 1 user = 1 client).

## Key Findings

### Stack

No stack changes. Flower ≥ 1.22 + PyTorch ≥ 2.7 + ML-1M is already the standard profile for cross-device FedRec work (PFedRec IJCAI-23, FedRAP ICLR-24, CoFedRec WWW-24, GPFedRec KDD-24, P²FedRec 2025 all run similar stacks). The migration is a configuration and semantics change, not a stack swap. See `STACK.md`.

### Table Stakes for "correct cross-device"

The minimum bar the four modules must meet after migration (from `FEATURES.md`):

- **Client universe**: `num-supernodes = 6040`, natural per-user partitioning the default, benchmark-mode asserts exactly one user per client.
- **Identity**: one canonical `raw_user_id → user_idx` mapping persisted and consumed by every module. Caches keyed by canonical `user_idx`, not partition order.
- **Splits**: deterministic LOO split with stable-sort tiebreaking; persisted split manifest with `split_hash`. Train-negative sampling excludes the held-out test positive.
- **Participation**: `C < 1` default (sweep-tunable); seeded server-side client sampling; sampled clients materialized on demand, not eagerly.
- **Aggregation**: explicit `weight-policy` config (`uniform` / `num_positives` / `num_training_examples`); default `num_positives` for BPR modules. Metric aggregation uses sufficient statistics, not pre-averaged per-client metrics.
- **Evaluation**: ONE primary protocol (`sampled_loo_99`) for the thesis table; `allrank_*` kept as a namespaced secondary. Metric names encode the protocol.
- **State**: run-namespaced `.embedding_cache/` with signature-checked loads that hard-fail on `num_users` / dim / schema mismatch. Per-user RNG streams `(run_seed, user_id, round, purpose)`.
- **Reporting**: best-round restore (global + local + strategy state); canonical fields are `best_*`, not `last_*`.

### Watch Out For (distilled pitfalls)

From `PITFALLS.md`, the top-priority items for this codebase:

1. **Test positive leaked into training negatives** — present in ALL FOUR modules today. Pure methodology bug.
2. **Unseeded server-side client sampling** — present in all four `server_app.py`. Blocks reproducibility.
3. **Global RNG reseeding inside evaluators** — makes per-round eval deterministic-in-a-bad-way; noted in baseline, personalized, adaptive.
4. **Shared `.embedding_cache/` across experiments** — silent contamination risk on every hyperparameter change.
5. **Adaptive module's per-user learned alpha doesn't accumulate across rounds** — load-order bug between `enable_per_user_alpha` and `load_local_user_embeddings`.
6. **PFedRec `affine_output.bias` handling diverges from reference** — reference aggregates it globally; Flower version keeps it local. Must be an explicit decision, not an oversight.
7. **Eager client instantiation** — Flower simulation must handle 6040 virtual clients with only sampled ones active.
8. **Best-round checkpoint not restored** — early stopping logs `best_round` but final evaluation uses last-round arrays.
9. **Full-rank vs sampled evaluation being treated as comparable** — the thesis table must use ONE protocol.
10. **User-ID mapping drift across modules** — one canonical mapping must be enforced.

### Architectural Deltas

From `ARCHITECTURE.md`:

- **Foundation layer first** (canonical ID mapping, split manifest, exclusion set, primary-evaluator choice, aggregation-weight policy) — everything downstream depends on this.
- **Per-module migration** can then parallelize across the four modules (num-supernodes=6040, natural partitioning default, run-namespaced cache, per-user RNG, sufficient-statistic metrics).
- **PFedRec reproduction** is a sequence dependency: cannot claim reproduction before the foundation and per-module steps land.
- **Adaptive-method bugs** can be fixed in parallel with per-module migration because they are self-contained in the adaptive module.
- **Results harness + thesis comparison** are sequenced last, after all four modules are migrated and stabilized.

### Build Order Implication for the Roadmap

Six phase clusters naturally fall out (exact breakdown is the roadmapper's job):

1. Foundation / shared protocol contract
2. Per-module cross-device migration (parallelizable across four modules)
3. PFedRec bug fixes + reference-alignment audit
4. Adaptive-module bug fixes (load order, prototype restore, per-user alpha accumulation)
5. Evaluation harness + per-group reporting + best-round restore
6. Thesis comparison runs + ablations under the unified protocol

## References — Paper Citations

| Tag | Paper | Link |
|-----|-------|------|
| FA17 | McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017 | https://proceedings.mlr.press/v54/mcmahan17a.html |
| NCF17 | He et al., "Neural Collaborative Filtering," WWW 2017 | https://doi.org/10.1145/3038912.3052569 |
| FP19 | Arivazhagan et al., "Federated Learning with Personalization Layers," arXiv 2019 | https://arxiv.org/abs/1912.00818 |
| FX20 | Li et al., "Federated Optimization in Heterogeneous Networks (FedProx)," MLSys 2020 | https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html |
| SC20 | Karimireddy et al., "SCAFFOLD," ICML 2020 | https://proceedings.mlr.press/v119/karimireddy20a.html |
| MR20 | Lin et al., "Meta Matrix Factorization for Federated Rating Predictions," SIGIR 2020 | https://doi.org/10.1145/3397271.3401081 |
| FE20 | Lin et al., "FedRec: Federated Recommendation With Explicit Feedback," IEEE IS 2020 | https://doi.org/10.1109/MIS.2020.3017205 |
| FR21 | Singhal et al., "Federated Reconstruction: Partially Local Federated Learning," NeurIPS 2021 | https://proceedings.neurips.cc/paper_files/paper/2021/hash/5d44a2b0d85aa1a4dd3f218be6422c66-Abstract.html |
| FO21 | Reddi et al., "Adaptive Federated Optimization," ICLR 2021 | https://openreview.net/forum?id=LkFG3lB13U5 |
| FN22 | Perifanis & Efraimidis, "Federated Neural Collaborative Filtering," KBS 2022 | https://doi.org/10.1016/j.knosys.2022.108441 |
| FG22 | Wu et al., "A federated graph neural network framework for privacy-preserving personalization," Nature Communications 2022 | https://doi.org/10.1038/s41467-022-30714-9 |
| LF22 | Zhang et al., "LightFR," TOIS 2022/2023 | https://doi.org/10.1145/3578361 |
| PR23 | Zhang et al., "Dual Personalization on Federated Recommendation (PFedRec)," IJCAI 2023 | https://www.ijcai.org/proceedings/2023/507 |
| RA24 | Li et al., "Federated Recommendation with Additive Personalization (FedRAP)," ICLR 2024 | https://openreview.net/forum?id=xkXdE81mOK |
| CF24 | He et al., "Co-clustering for Federated Recommender System," WWW 2024 | https://openreview.net/forum?id=aAcScFLHzF |
| CA24 | Zhang et al., "Beyond Similarity: Personalized Federated Recommendation with Composite Aggregation (FedCA)," arXiv 2024 | https://doi.org/10.48550/arXiv.2406.03933 |
| GP24 | Zhang et al., "GPFedRec: Graph-Guided Personalization for Federated Recommendation," KDD 2024 | https://doi.org/10.1145/3637528.3671702 |
| P225 | Hu et al., "P²FedRec," Proc. ACM Manag. Data 2025 | https://doi.org/10.1145/3769811 |
| SV25 | Zhang et al., "Personalized Recommendation Models in Federated Settings: A Survey," 2025 | https://arxiv.org/abs/2504.07101 |
| FL | Flower scaling docs and advanced example | https://flower.ai/docs/framework/ |

## Outputs

- `.planning/research/STACK.md` — pointer to the existing codebase stack map plus cross-device confirmation
- `.planning/research/ARCHITECTURE.md` — migration deltas and build-order implications
- `.planning/research/FEATURES.md` — table stakes / differentiating / anti-features, with phase tags
- `.planning/research/PITFALLS.md` — 26 pitfalls with warning signs, prevention, phase tags, and direct repo citations
- `.planning/research/SUMMARY.md` — this file
