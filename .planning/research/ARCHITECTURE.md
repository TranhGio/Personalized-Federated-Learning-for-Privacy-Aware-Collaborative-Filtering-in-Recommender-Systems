# Architecture Research

**Brownfield note:** The existing per-module architecture is already fully mapped at `.planning/codebase/ARCHITECTURE.md`. This document captures only the architectural *deltas* required by the cross-device migration, plus build-order implications for the roadmap.

## Current Architecture (summary)

Four parallel Python packages share the five-file skeleton `dataset.py / task.py / client_app.py / server_app.py / models/` plus (for three of them) `strategy.py`. The primary architectural axis is the **personalization boundary** — which tensors are global (aggregated) vs local (cached on client):

| Module | Global | Local |
|--------|--------|-------|
| `federated-baseline-cf` | everything | (none) |
| `federated-pfedrec` | `embedding_item.weight` | `affine_output.weight`, `affine_output.bias` (per-user) |
| `federated-personalized-cf` | item embeddings + biases + global bias | user embeddings, user bias |
| `federated-adaptive-personalized-cf` | same as personalized + server prototype EMA | same as personalized + `personal_mlp.*`, `fusion_gate/layer`, `logit_alpha` (per-user), `item_perturbation` |

See `.planning/codebase/ARCHITECTURE.md` (§Personalization Boundary Matrix) for the authoritative table.

## Migration Deltas

### Data Layer (`dataset.py`)

- **Switch to natural partitioning as default.** `partition-mode = "natural"` exists in all four modules. Make it the default; the Dirichlet partitioner stays for the explicit cross-silo ablation.
- **Canonical user-index mapping.** One `raw_user_id → user_idx` artifact persisted to disk and imported by every module's `dataset.py`, `client_app.py`, and evaluator. Today each module re-derives its own mapping in `create_global_mappings()`.
- **Split manifest.** Persist the leave-one-out split deterministically (stable sort on `(user_id, timestamp, movie_id)`) and save a `split_hash` so caches and resumes can detect divergence.
- **Exclusion set for negative sampling.** Add `exclude_items[user] = train_pos ∪ test_pos` in one place in `dataset.py`; consume it in every `task.py`.

### Orchestration Layer (`server_app.py`)

- **`num-supernodes = 6040`** in every `pyproject.toml`; current default is `5`.
- **Seeded server-side sampling.** Replace `random.sample(node_ids, num_selected)` with a `Random(run_seed + round_idx)`-derived call. Persist selected client IDs per round for replayability.
- **Sample fraction as a hyperparameter.** Keep `fraction-train` configurable; sweep. Default per-module set to match the paper the module calibrates against.
- **Aggregation weight policy made explicit.** Add a `weight-policy` config: `uniform` / `num_positives` / `num_training_examples`. Strategy reads it from config; FedRec-standard is `num_positives` for BPR modules.
- **Sufficient-statistic metric aggregation.** Clients return `hit_count@k`, `ndcg_sum@k`, `evaluated_users`, not per-client averages. Server computes the final ratio once.
- **Early-stopping checkpoint restore.** Save `best_arrays` and (for split modules) `best_local_state_hashes` in memory; after the loop, restore and run ONE final evaluation that goes into the result artifact.

### Client Layer (`client_app.py`)

- **Benchmark-mode one-user assertion.** In benchmark mode assert exactly one raw user per client loader; fail loudly if a multi-user partition slips through.
- **Collapse loops.** PFedRec's client currently loops over `user_test_items.keys()` inside a partition — collapse to the single-user path when `num-supernodes == num_users`.
- **Run-namespaced cache.** `.embedding_cache/` path includes `run_id / method / num_users / num_items / dim / split_hash`. Cache loads hard-fail on shape or schema mismatch.
- **Per-user RNG streams.** Derive from `(run_seed, user_id, round, purpose)`. Today clients default to `seed=42` everywhere.
- **PFedRec per-user atomic head.** `(affine_output.weight, affine_output.bias)` saved as one `.pt` file per user. Strategy's `LOCAL_PARAM_KEYS` must include both.
- **Per-user alpha / item perturbation load-order fix.** Call `enable_per_user_alpha()` / `enable_item_perturbation()` BEFORE `load_local_user_embeddings()` so the local params exist in `_LOCAL_PARAMS` at load time (bug documented in CONCERNS.md).

### Evaluation Layer (`task.py`)

- **Pick ONE primary evaluator.** `evaluate_ranking_sampled` (LOO + 99 negatives, NCF protocol) is the primary. `evaluate_ranking` (all-items) stays as a secondary, namespaced `allrank_*` metric.
- **No global RNG reseeding.** Evaluators accept a `random.Random` instance; no `random.seed(seed)` at the top of `evaluate_ranking_sampled`.
- **Per-user-group metrics emitted as first-class fields.** Sparse/medium/dense NDCG@10 become `ndcg@10/sparse`, `ndcg@10/medium`, `ndcg@10/dense`; reporting scripts surface them in every comparison.

### Strategy Layer (`strategy.py`, three modules)

- **`GLOBAL_PARAM_KEYS` / `LOCAL_PARAM_KEYS` reviewed per module.** Especially `affine_output.bias` in PFedRec (currently LOCAL but reference aggregates it — decision point documented in CONCERNS.md).
- **Prototype aggregation stays intact** in the adaptive module but is logged per-round.

## Build Order Implications

Dependencies between the deltas shape the phase order:

1. **Foundation** (must come first): canonical ID mapping, split manifest, exclusion set, primary-evaluator decision. Everything downstream depends on these.
2. **Per-module migration** (parallelizable across the four modules once foundation lands): num-supernodes=6040, natural partitioning default, per-user RNG, run-namespaced cache, sufficient-statistic metrics.
3. **PFedRec reproduction** (depends on 1 + 2 + PFedRec bug fixes): must close the gap to IJCAI-23 reference numbers.
4. **Adaptive-method bug fixes** (depends on 1 + 2): per-user alpha load-order, item-perturbation load-order, prototype restore on checkpoint.
5. **Evaluation + results harness** (depends on 1 + 2): unified W&B logging, per-group reporting, best-round restore.
6. **Thesis comparison runs** (depends on 1–5): standardized config, adaptive vs three baselines, sparse-user slice.

Per-module work in step 2 is independent and fits the "parallel plans" execution preference.

## Data Flow (unchanged)

Still: ML-1M CSV → user/item indices → LOO split → per-user DataLoader → per-round train/eval → W&B. The per-client counts and sampling rules are what change.

## References

- `.planning/codebase/ARCHITECTURE.md` — current architecture detail (authoritative)
- `.planning/codebase/CONCERNS.md` — known bugs/gaps informing the deltas
- Codex research (2026-04-19) — cross-device protocol features and build order
- `docs/superpowers/plans/2026-04-04-cross-device-migration.md` — earlier draft plan (superseded by this cycle but worth cross-checking)
