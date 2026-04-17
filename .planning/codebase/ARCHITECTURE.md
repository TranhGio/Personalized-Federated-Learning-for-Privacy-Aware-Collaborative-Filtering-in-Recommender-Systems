# Architecture

**Analysis Date:** 2026-04-17

## Pattern Overview

**Overall:** Flower (flwr) federated learning pub-sub with custom aggregation strategies, implemented four times in parallel as a research progression (baseline → PFedRec → personalized → adaptive). Each implementation is an independent Python package with a shared five-file skeleton: `dataset.py`, `task.py`, `client_app.py`, `server_app.py`, `models/`. The adaptive and PFedRec variants additionally include a `strategy.py` with custom split-aware strategies, and the adaptive variant adds an `evaluation/` subpackage.

**Key Characteristics:**

- **Flower Grid messaging API** — `grid.send_and_receive(messages)` drives a synchronous round-by-round loop in each `server_app.py`. Clients respond to `@app.train()` and `@app.evaluate()` hooks via `flwr.clientapp.ClientApp`.
- **Parallel comparative design** — Four sibling directories (`federated-baseline-cf/`, `federated-pfedrec/`, `federated-personalized-cf/`, `federated-adaptive-personalized-cf/`) implement the same MovieLens 1M collaborative-filtering task with varying global/local parameter boundaries. The **personalization boundary is the primary architectural differentiator**, not the framework code.
- **Client-side local caching** — All personalized variants persist LOCAL parameters to disk (`.embedding_cache/partition_{id}/`) between rounds, so the same partition resumes with its private state even though Flower simulation may recycle client processes.
- **BPR-MF is the canonical model** — All three BPR-capable modules share a nearly identical `BPRMF` class (file: `models/bpr_mf.py`); the personalized and adaptive variants extend it with `get_global_parameters()` / `set_global_parameters()` / `get_local_parameters()` / `set_local_parameters()` methods that classify weights as `_GLOBAL_PARAMS` or `_LOCAL_PARAMS`.
- **PFedRec deviates from MF** — `federated-pfedrec/` uses `PFedRecMLP` (file: `federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py`) which has NO user embedding; personalization is a per-user Linear `affine_output` layer cached at `.embedding_cache/partition_{id}/user_{uid}/affine_output.pt`.
- **Reference implementation preserved** — `IJCAI-23-PFedRec/` contains the unmodified upstream PFedRec code (`train.py`, `engine.py`, `mlp.py`, `data.py`) used as a calibration reference for `federated-pfedrec/`.

## Layers

**`dataset.py` (data layer):**
- Purpose: Download ML-1M, build `user2idx`/`item2idx` mappings, partition into clients, build `MovieLensDataset` + `DataLoader` per partition.
- Location: `federated-baseline-cf/federated_baseline_cf/dataset.py`, `federated-pfedrec/federated_pfedrec/dataset.py`, `federated-personalized-cf/federated_personalized_cf/dataset.py`, `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py`
- Contains: `load_movielens_1m()`, `dirichlet_partition_users()`, `natural_partition_users()` (cross-device 1-user-per-client), `create_leave_one_out_split()`, `create_global_mappings()`, `load_partition_data()`, `load_full_data()`
- Module-level `_partition_cache` keyed by `f"{num_partitions}_{alpha}_{split_mode}_{partition_mode}"` prevents re-partitioning per client in single-process simulation.
- The adaptive variant additionally computes `user_stats` (n_interactions, genre_entropy, n_unique_items, rating_std) used for alpha.
- Depends on: `torch.utils.data.Dataset`, `pandas`, `numpy`, raw CSVs in `data/ml-1m/`
- Used by: `task.py` via `load_partition_data()` and `load_full_data()`

**`task.py` (training/evaluation layer):**
- Purpose: Model factory + train/test/eval_ranking functions. Module-level `_dataset_cache` remembers `num_users` / `num_items` after the first `load_data` call.
- Location: `federated-baseline-cf/federated_baseline_cf/task.py`, `federated-pfedrec/federated_pfedrec/task.py`, `federated-personalized-cf/federated_personalized_cf/task.py`, `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py`
- Contains: `load_data()`, `get_model()`, `train()` (dispatch on `model-type`), `train_basic_mf()`, `train_bpr_mf()`, `test()`, `evaluate_ranking()` (all-items ranking), `evaluate_ranking_sampled()` (leave-one-out + 99 negatives, NCF protocol).
- PFedRec variant instead has `prepare_user_train_data()`, `train_pfedrec_single_user()`, `evaluate_pfedrec_sampled()`, `test_pfedrec()` — trained per-user inside a partition.
- Adaptive variant adds `compute_client_alpha()`, `compute_per_user_alpha()`, `get_user_stats()`.
- Depends on: `models/`, `dataset.py`
- Used by: `client_app.py` (both train and evaluate), `server_app.py` (centralized eval in baseline only)

**`client_app.py` (Flower client):**
- Purpose: Implements `@app.train()` and `@app.evaluate()` message handlers. Drives the per-round split-learning protocol on each client.
- Location: one per module, e.g., `federated-personalized-cf/federated_personalized_cf/client_app.py`
- Responsibilities diverge per module:
  - **baseline**: receive full state_dict → train → return full state_dict. No caching.
  - **pfedrec**: receive global `embedding_item.weight` → loop over users in partition, each load/save `affine_output.pt` → average item embeddings across users → return averaged item embedding.
  - **personalized**: receive global params → load user embeddings from `.embedding_cache/` → train → save user embeddings → return only global params (~44% communication reduction).
  - **adaptive**: same as personalized plus: compute client alpha (or per-user alpha tensor), call `model.set_alpha()` and `model.set_global_prototype()`, receive `global_prototype` list in config, attach `user_prototype` list to return metrics.
- All variants include a `get_device()` helper with CUDA compatibility fallback (handles RTX 5090 vs old PyTorch).
- All personalized variants define disk caching helpers: `get_cache_dir()`, `save_local_user_embeddings()` / `save_user_local_params()`, `load_local_user_embeddings()` / `load_user_local_params()`, `clear_embedding_cache()`.

**`server_app.py` (Flower server orchestrator):**
- Purpose: Owns the federated training loop, manages wandb, writes results JSON, runs early stopping, handles centralized evaluation (baseline only).
- Location: one per module, e.g., `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py`
- Uses `flwr.serverapp.Grid` to create/send/receive messages. Defines a `DummyClientProxy` shim (`ClientProxy` subclass returning `None` from all methods) so it can feed `FitRes` tuples into Flower's strategy aggregation functions.
- Round structure (consistent across all four modules):
  1. `grid.get_node_ids()` → `random.sample(node_ids, num_selected)` using `fraction-train`
  2. Construct `Message` with `ArrayRecord` of current global params + `ConfigRecord` of {lr, proximal_mu, [global_prototype]}
  3. `grid.send_and_receive(train_messages)` → parse responses into `FitRes`
  4. `strategy.aggregate_fit(server_round, results, failures=[])` → update `arrays`
  5. Build eval messages, `grid.send_and_receive(eval_messages)`, `weighted_average_metrics()` helper
  6. wandb log, early-stopping check
- Writes final JSON to `../results/federated/` (baseline), `../results/federated/personalized/`, `../results/federated/pfedrec/`, `../results/federated/` depending on module.

**`strategy.py` (custom aggregation, personalized/pfedrec/adaptive only):**
- Purpose: Subclass `flwr.server.strategy.FedAvg` / `FedProx` to encode which param keys are global.
- Files: `federated-pfedrec/federated_pfedrec/strategy.py`, `federated-personalized-cf/federated_personalized_cf/strategy.py`, `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py`
- All three define module-level frozensets `GLOBAL_PARAM_KEYS` and `LOCAL_PARAM_KEYS` (contents differ per module — see boundary table below).
- The `SplitFedAvg` / `SplitFedProx` classes are thin wrappers; **the split actually happens in `client_app.py`** which only sends global params. They exist mainly to carry `_is_split_learning = True` and (for adaptive) to implement prototype aggregation.
- The adaptive variant's `SplitFedAvg.aggregate_fit()` overrides the base, calling `super().aggregate_fit()` then `self._aggregate_prototypes(results)` — it extracts `user_prototype` lists from client metrics, does weighted mean, and applies EMA: `p_global = momentum * p_old + (1 - momentum) * p_new`.
- Baseline module uses `FedAvg` / `FedProx` directly from `flwr.server.strategy` (no custom strategy file).

**`models/` (model definitions):**
- Purpose: `nn.Module` subclasses plus loss functions. Split-aware models expose `_GLOBAL_PARAMS` / `_LOCAL_PARAMS` tuples and matching `get_/set_global_parameters()` / `get_/set_local_parameters()` methods.
- Layout per module:
  - `federated-baseline-cf/federated_baseline_cf/models/`: `basic_mf.py`, `bpr_mf.py`, `losses.py`
  - `federated-pfedrec/federated_pfedrec/models/`: `pfedrec_mlp.py`, `losses.py`
  - `federated-personalized-cf/federated_personalized_cf/models/`: same as baseline but with split methods on each class
  - `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/`: adds `dual_personalized_bpr_mf.py` (DualPersonalizedBPRMF — thesis novel model) and `adaptive_alpha.py` (alpha factory)

**`evaluation/` (adaptive module only):**
- Location: `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/evaluation/`
- Files: `alpha_analysis.py` (`AlphaAnalyzer`, `AlphaStatistics`), `user_groups.py` (`UserGroupConfig` with sparse/medium/dense buckets, `classify_user_group()`, `aggregate_metrics_by_group()`).

## Personalization Boundary Matrix (the core architectural axis)

| Module | `GLOBAL_PARAM_KEYS` (shared) | `LOCAL_PARAM_KEYS` (private, cached) | Comm savings |
|--------|------------------------------|--------------------------------------|--------------|
| `federated-baseline-cf` | **all** state_dict keys (user + item embeddings, all biases) | (none) | baseline ~874K params |
| `federated-pfedrec` | `embedding_item.weight` | `affine_output.weight`, `affine_output.bias` (per-user, one file per user) | ~118K params/round |
| `federated-personalized-cf` | `item_embeddings.weight`, `item_bias.weight`, `global_bias` | `user_embeddings.weight`, `user_bias.weight` | ~485K params/round (-44%) |
| `federated-adaptive-personalized-cf` | same as personalized + prototype EMA state on server | same as personalized + `personal_mlp.*`, `fusion_gate/layer`, `logit_alpha` (per-user), `item_perturbation` | ~38% transmitted |

Frozenset definitions live in `federated-personalized-cf/federated_personalized_cf/strategy.py` lines 15-24, `federated-pfedrec/federated_pfedrec/strategy.py` lines 15-22, `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` lines 34-43.

## Data Flow

**ML-1M → Dirichlet partition → per-client DataLoader → round-based train/eval → W&B logging:**

1. **Download & parse**: `download_movielens_1m()` fetches the zip if `data/ml-1m/` missing; `load_movielens_1m()` reads `ratings.dat`, `movies.dat`, `users.dat` with `pd.read_csv(sep="::", engine="python")`.
2. **Global mappings**: `create_global_mappings(ratings_df)` → `(user2idx, idx2user, item2idx, idx2item)` over the union of user_ids and movie_ids (6040 users, 3706 movies).
3. **Partitioning**: branch on `partition-mode`:
   - `"dirichlet"`: `compute_user_genre_distribution()` builds a per-user genre proportion matrix; `np.random.dirichlet([alpha]*num_genres, num_clients)` samples client target distributions; each user is greedily assigned to the client minimizing KL divergence. Cross-silo style, typically `num-supernodes=5`.
   - `"natural"`: 1 user = 1 client via `natural_partition_users()`. Cross-device setup (recommended per Papers/digested/yin_2025_devicers_survey.md).
4. **Leave-one-out split**: `create_leave_one_out_split()` sorts each user's ratings by timestamp and holds out the last interaction as test (NCF protocol — He et al., WWW 2017).
5. **DataLoader**: `MovieLensDataset(ratings_df, user2idx, item2idx)` wraps tensors; `DataLoader(batch_size=..., shuffle=True)` returns dict `{"user", "item", "rating"}` per batch.
6. **Server loop** in `server_app.main()`: for `round_num in 1..num_server_rounds`:
   - Sample clients → send `Message(arrays=ArrayRecord, config=ConfigRecord({lr, proximal_mu, [global_prototype]}))`.
   - Each client executes its `@app.train()` hook: load global params from message, load local params from cache (if any), train for `local-epochs`, save local params, return global params (+ optional `user_prototype` list in metrics).
   - Server parses responses → `FitRes` list → `strategy.aggregate_fit()` → new global `arrays`.
   - Eval messages sent with updated arrays → `weighted_average_metrics()` over client responses.
   - `wandb.log({"train/...": ..., "eval/...": ...}, step=round_num)` if `wandb-enabled`.
   - `early_stopper.step(round_num, metrics)` → may `break`.
7. **Final JSON**: `json.dump(results_data, f, indent=4)` to `results/federated/{...}_results.json` with full `federated_config`, `early_stopping` summary, `final_metrics`, `timestamp`.

**State Management:**

- **Server state**: strategy object holds aggregation state (`self._global_prototype` EMA in adaptive module); `arrays: ArrayRecord` is rebuilt each round by mapping `parameters_to_ndarrays(aggregated_params)` back onto `list(arrays.to_torch_state_dict().keys())`.
- **Client state**: `_device_cache`, `_partition_cache`, `_dataset_cache`, `_user_stats_cache`, `_item_popularity_cache` — module-level globals that survive across same-process Flower simulation clients. Disk-side state: `.embedding_cache/partition_{id}/user_embeddings.pt` (or per-user `affine_output.pt` for PFedRec) loaded/saved via atomic temp-file-plus-rename pattern.

## Key Abstractions

**Split learning global/local parameter split:**
- Each `nn.Module` subclass in the personalized/adaptive/pfedrec variants owns class-level tuples `_GLOBAL_PARAMS_*` and `_LOCAL_PARAMS_*` (e.g., `federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py:41-46`).
- Instance properties `_GLOBAL_PARAMS` / `_LOCAL_PARAMS` are resolved in `__init__` based on `use_bias` flag.
- Methods `get_global_parameters()`, `set_global_parameters(dict)`, `get_local_parameters()`, `set_local_parameters(dict, strict=False)`, `get_global_parameter_names()` are a uniform interface (see `federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py:342-466`). `set_local_parameters` has shape-mismatch handling: when saved tensor has fewer users than current, it partially loads `current_state[name][:num_saved] = saved_tensor`.
- `SplitFedAvg`/`SplitFedProx` subclasses in `strategy.py` are thin: the actual split happens in `client_app.py` where `model.get_global_parameters()` returns only the keys the server expects, and `ArrayRecord(global_params)` is sent back.

**Adaptive alpha (`federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/adaptive_alpha.py`):**
- Factory `create_alpha_computer(config: AlphaConfig, hc_config: HierarchicalConditionalAlphaConfig)` returns one of three classes:
  - `DataQuantityAlpha`: single-factor sigmoid on interaction count.
  - `MultiFactorAlpha`: `0.40*f_quantity + 0.25*f_diversity + 0.20*f_coverage + 0.15*f_consistency` (weights from AlphaConfig).
  - `HierarchicalConditionalAlpha` (default): Stage 1 — `data_volume = sqrt(f_q*f_c)` (geometric mean, resolves 0.8-1.0 correlation), `preference_quality = harmonic(f_d, f_s)` (resolves -0.3 to -0.5 conflict), `base_alpha = 0.55*data_volume + 0.45*preference_quality`. Stage 2 — 4 conditional rules for sparse/niche/inconsistent/completionist archetypes.
- Per-client alpha clipped to `[min_alpha=0.1, max_alpha=0.95]`.
- `compute_client_alpha()` in `task.py` takes aggregated `user_stats` and returns a scalar; `compute_per_user_alpha()` returns a per-user tensor (used when `enable-per-user-alpha=true`).

**Dual-level personalization (`models/dual_personalized_bpr_mf.py`):**
- `DualPersonalizedBPRMF` holds both a standard MF pathway and a local `PersonalMLP` pathway.
- Level 1 (statistical): `p_effective = alpha * p_local + (1-alpha) * p_global` (called via `model.set_alpha(...)` and `model.set_global_prototype(...)` before `forward()`).
- Level 2 (neural): `PersonalMLP` scores element-wise product `p_effective ⊙ q_item`.
- Fusion (`fusion_type` = `add` | `gate` | `concat`): `add` = simple sum, `gate` = learnable sigmoid gate, `concat` = `Linear([score_cf; score_mlp])`.
- `PersonalMLP` weights (`personal_mlp.*`) and `fusion_gate/layer` are LOCAL — never transmitted.

**Global prototype (EMA):**
- Server state in `SplitFedAvg._global_prototype` (numpy array).
- Each client computes `model.compute_user_prototype()` after training and serializes as a list into `FitRes.metrics[USER_PROTOTYPE_KEY]` (key = `"user_prototype"`, defined `strategy.py:46`).
- Server's `_aggregate_prototypes(results)`: `new_prototype = Σ(prototype * num_examples) / total_weight`, then `p_global = momentum * p_old + (1 - momentum) * new_prototype` with `momentum=0.9`. Missing on first round → `p_global = new_prototype` directly.
- Next round, server attaches `train_config_dict["global_prototype"] = global_prototype.tolist()` to the message; client converts back to tensor via `torch.tensor(global_prototype_list, dtype=torch.float32)` and calls `model.set_global_prototype(tensor)`.

**Per-user learned alpha (opt-in, `enable-per-user-alpha=true`):**
- Stored as `logit_alpha` Embedding (LOCAL) inside `DualPersonalizedBPRMF`.
- Initialized from heuristic via `torch.logit(heuristic_alpha)`, refined by BPR gradient descent. Cached/restored through `get_local_parameters()`.

**Dual-side item perturbation (opt-in, `enable-item-perturbation=true`):**
- `_item_perturbation = nn.Embedding(num_items, embedding_dim)` zero-initialized.
- `q_effective[i] = q_global[i] + perturbation[i]`, L2-regularized in loss: `reg * ||perturbation||^2`.
- LOCAL — never sent to server.

**Contrastive local-global alignment (opt-in, `contrastive-lambda > 0`):**
- `InfoNCEContrastiveLoss` (in `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/losses.py`).
- `L_total = L_BPR + lambda * L_contrastive + reg * ||perturbation||^2`.
- Positive pair: `(p_local[u], p_effective[u])`; negatives: other users in batch.

## Entry Points

**Per-module `flwr run .` (primary entrypoint):**
- Configured in each `pyproject.toml` under `[tool.flwr.app.components]`:
  - `federated-baseline-cf/pyproject.toml:35-36`: `serverapp = "federated_baseline_cf.server_app:app"`, `clientapp = "federated_baseline_cf.client_app:app"`
  - `federated-pfedrec/pyproject.toml:33-34`: same pattern with `federated_pfedrec`
  - `federated-personalized-cf/pyproject.toml:35-36`: same with `federated_personalized_cf`
  - `federated-adaptive-personalized-cf/pyproject.toml:35-36`: same with `federated_adaptive_personalized_cf`
- Federation defined in `[tool.flwr.federations.local-simulation]` (or `local-sim-gpu`) with `options.num-supernodes = 5`.
- Runtime overrides: `flwr run . --run-config "key=value other-key=value"` reads into `context.run_config`.
- `@app.main()` in `server_app.py` is the server entrypoint; `@app.train()` and `@app.evaluate()` in `client_app.py` are the client entrypoints.

**Centralized baselines (alternate entrypoints):**
- `centralize_baseline_ncf.py` at project root — Keras/TensorFlow NCF baseline that writes to `results/centralized/ncf_baseline_results.json`.
- `centralized_baseline_svd.ipynb` — Surprise-SVD notebook writing to `results/centralized/svd_baseline_results.json`.

**Auxiliary scripts:**
- `scripts/run_all_baselines.sh`, `scripts/run_baseline_sweep_loo.sh`, `scripts/compare_all_results.py` — orchestrate multiple `flwr run` invocations.
- Per-module `scripts/run_fedprox_sweep.sh` and `scripts/analyze_sweep_results.py`.
- `federated-adaptive-personalized-cf/scripts/run_wandb_sweep.py` — W&B sweep agent wired via `sweep.yaml` (Bayesian + Hyperband).
- Per-module `visualize_partitions.py`, `test_dataset.py`, `test_models.py` — standalone validation scripts at each module root.

**Reference implementation:** `IJCAI-23-PFedRec/train.py` — upstream PFedRec paper code, kept unmodified for calibration against `federated-pfedrec/`.

## Error Handling

**Strategy:** Best-effort per-client; the server tolerates individual client failures and continues the round.

**Patterns:**
- `client_app.py`: every `@app.train()` / `@app.evaluate()` handler wraps the full train/eval path; exceptions propagate as Flower `Message.has_error() == True`.
- `server_app.py`: for each response, `if response.has_error(): print(warning); continue` (see `federated-baseline-cf/federated_baseline_cf/server_app.py:325-327`). Failed clients are excluded from `fit_results` and metrics aggregation.
- CUDA fallback: `get_device()` in every `client_app.py` tests `torch.zeros(1).cuda()` and falls back to CPU on `RuntimeError` (handles RTX 5090 + old PyTorch mismatch).
- Atomic cache saves: `save_local_user_embeddings()` uses `tempfile.mkstemp` + `torch.save` + `os.replace` pattern; on exception, temp file is `os.unlink`-ed.
- Shape mismatch in local params: `set_local_parameters(..., strict=False)` does partial load when saved tensor is smaller than current (see `federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py:436-455`).
- Early stopping shielding: `if early_stopper is not None and round_eval_metrics:` — never triggers on empty eval rounds.

## Cross-Cutting Concerns

**Logging:**
- Per-round stdout with `print(f"{'='*50}\nRound {round_num}/{num_rounds}\n{'='*50}")` headers.
- W&B integrated in each `server_app.py`: `wandb.init(project=..., entity=..., config=...)` at start, `wandb.log(round_metrics, step=round_num)` each round, `wandb.run.summary[f"final/{key}"] = value` at end.
- wandb artefacts in `federated-<module>/wandb/` per module; most recent in `latest-run` symlink.

**Validation:**
- `AlphaConfig.__post_init__()` validates `0 <= min_alpha < max_alpha <= 1`, positive thresholds, method in allowed set, and (for multi-factor) `factor_weights` summing to 1.0 ± 0.01.
- `UserGroupConfig.__post_init__()` validates adjacency `sparse[1] == medium[0]` and `medium[1] == dense[0]`.

**Authentication:** None — simulation-only. `[tool.flwr.federations.remote-federation]` entry in each pyproject.toml shows the TLS pattern but `insecure = true` is the default.

**Evaluation protocol (cross-cutting convention):**
- Primary metric: `sampled_ndcg@10` (NCF protocol — 1 positive + 99 negatives).
- Secondary: `hit_rate@{5,10,20}`, `mrr`, `coverage@K`, `novelty@K`.
- Rating prediction (`rmse`, `mae`) reported but not optimized under BPR.
- Results JSON structure: `{model_name, dataset, federated_config, early_stopping, timestamp, final_metrics, training_rounds}`.

---

*Architecture analysis: 2026-04-17*
