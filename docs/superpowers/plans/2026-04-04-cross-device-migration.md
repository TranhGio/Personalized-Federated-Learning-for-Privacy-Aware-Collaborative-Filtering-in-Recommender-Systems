# Cross-Silo → Cross-Device Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch all 4 federated implementations from cross-silo (5 clients, ~1200 users each) to cross-device (6040 clients, 1 user = 1 client) to match every published federated recommendation baseline.

**Architecture:** Add a `partition-mode` config (`"dirichlet"` or `"natural"`) to each module. In `"natural"` mode, `partition_id == user_idx` (1:1 mapping). Each client trains on exactly one user's data. User embeddings are cached per-user (single row, not full table). Server randomly samples `fraction-train` clients per round.

**Tech Stack:** Flower (flwr) 1.22+, PyTorch 2.7+, MovieLens 1M (6040 users, 3706 items)

---

## File Structure

**Files modified per module** (4 modules × ~5 files each):

| Module | dataset.py | client_app.py | server_app.py | pyproject.toml | task.py |
|--------|-----------|---------------|---------------|----------------|---------|
| federated-baseline-cf | Add natural partitioning | Minor | Random sampling | Config | No change |
| federated-personalized-cf | Add natural partitioning | Per-user cache | Random sampling | Config | No change |
| federated-pfedrec | Add natural partitioning | Simplify loop | Random sampling | Config | No change |
| federated-adaptive-personalized-cf | Add natural partitioning | Per-user cache + alpha | Random sampling | Config | No change |

**Key insight:** `task.py` (training/eval logic) needs NO changes in any module. With 1 user per partition, existing multi-user code naturally operates on a single user's data.

---

### Task 1: Add Natural Partitioning to Baseline dataset.py

**Files:**
- Modify: `federated-baseline-cf/federated_baseline_cf/dataset.py`

This establishes the pattern all other modules will copy.

- [ ] **Step 1: Add `natural_partition_users()` function**

Add after the existing `dirichlet_partition_users()` function (~line 283):

```python
def natural_partition_users(
    ratings_df: pd.DataFrame,
    user2idx: Dict[int, int],
) -> Dict[int, pd.DataFrame]:
    """Cross-device partitioning: 1 user = 1 client.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Full ratings with 'user_id' column.
    user2idx : Dict[int, int]
        Mapping from raw user_id to contiguous index (0..N-1).

    Returns
    -------
    Dict[int, pd.DataFrame]
        {user_idx: DataFrame of that user's ratings}
    """
    partitions: Dict[int, pd.DataFrame] = {}
    for user_id, user_idx in user2idx.items():
        partitions[user_idx] = ratings_df[ratings_df["user_id"] == user_id].copy()
    return partitions
```

- [ ] **Step 2: Modify `load_partition_data()` to accept `partition_mode`**

Add `partition_mode: str = "dirichlet"` parameter. In the cache-miss branch, choose partitioning strategy:

```python
def load_partition_data(
    partition_id: int,
    num_partitions: int,
    alpha: float = 0.5,
    test_ratio: float = 0.2,
    batch_size: int = 32,
    data_dir: Optional[str] = None,
    split_mode: str = "leave-one-out",
    partition_mode: str = "dirichlet",  # NEW: "dirichlet" or "natural"
):
```

Inside the cache-miss branch, replace the partitioning call:

```python
    # Change cache key to include partition_mode
    cache_key = f"{num_partitions}_{alpha}_{split_mode}_{partition_mode}"
    
    if cache_key not in _dataset_cache:
        ratings_df, movies_df, users_df = load_movielens_1m(data_dir)
        user2idx, item2idx = create_global_mappings(ratings_df)
        num_users = len(user2idx)
        num_items = len(item2idx)

        if partition_mode == "natural":
            partitions = natural_partition_users(ratings_df, user2idx)
        else:
            partitions = dirichlet_partition_users(
                ratings_df, movies_df, num_partitions, alpha
            )

        _dataset_cache[cache_key] = {
            "partitions": partitions,
            "user2idx": user2idx,
            "item2idx": item2idx,
            "num_users": num_users,
            "num_items": num_items,
            "movies_df": movies_df,
        }
```

- [ ] **Step 3: Verify existing tests pass**

Run: `cd federated-baseline-cf && python test_dataset.py`

Expected: All tests pass (no behavior change when `partition_mode="dirichlet"`)

- [ ] **Step 4: Commit**

```bash
git add federated-baseline-cf/federated_baseline_cf/dataset.py
git commit -m "feat(baseline): add natural partitioning mode for cross-device FL"
```

---

### Task 2: Wire `partition-mode` Through Baseline Client & Config

**Files:**
- Modify: `federated-baseline-cf/pyproject.toml`
- Modify: `federated-baseline-cf/federated_baseline_cf/client_app.py`
- Modify: `federated-baseline-cf/federated_baseline_cf/task.py` (just the `load_data` wrapper)

- [ ] **Step 1: Add config to pyproject.toml**

In `[tool.flwr.app.config]` section, add:

```toml
partition-mode = "natural"       # "dirichlet" or "natural" (1 user = 1 client)
```

Change existing defaults for cross-device:

```toml
num-server-rounds = 100
local-epochs = 1
embedding-dim = 32
fraction-train = 0.1
```

In `[tool.flwr.federations.local-simulation.options]`:

```toml
num-supernodes = 6040
```

- [ ] **Step 2: Pass `partition-mode` through task.py `load_data()`**

In `task.py`, modify the `load_data()` wrapper function to accept and pass `partition_mode`:

```python
def load_data(
    partition_id: int,
    num_partitions: int,
    alpha: float = 0.5,
    batch_size: int = 256,
    split_mode: str = "leave-one-out",
    partition_mode: str = "dirichlet",  # NEW
):
    trainloader, testloader, num_users, num_items, user2idx, item2idx = (
        load_partition_data(
            partition_id=partition_id,
            num_partitions=num_partitions,
            alpha=alpha,
            batch_size=batch_size,
            split_mode=split_mode,
            partition_mode=partition_mode,  # NEW
        )
    )
    # ... rest unchanged
```

- [ ] **Step 3: Read `partition-mode` in client_app.py**

In the train/evaluate functions where `load_data()` is called:

```python
partition_mode = context.run_config.get("partition-mode", "dirichlet")
trainloader, testloader = load_data(
    partition_id=partition_id,
    num_partitions=num_partitions,
    alpha=alpha,
    split_mode=split_mode,
    partition_mode=partition_mode,  # NEW
)
```

- [ ] **Step 4: Commit**

```bash
git add federated-baseline-cf/
git commit -m "feat(baseline): wire partition-mode config for cross-device support"
```

---

### Task 3: Add Random Client Sampling to Baseline Server

**Files:**
- Modify: `federated-baseline-cf/federated_baseline_cf/server_app.py`

Currently the server takes the first N node IDs. For cross-device with 6040 clients and fraction_train=0.1, we need random sampling.

- [ ] **Step 1: Add random sampling to training round**

Find the client selection code (where `node_ids[:num_selected]` is used) and replace with:

```python
import random as stdlib_random

# Select clients for training
all_node_ids = sorted(grid.get_node_ids())
num_selected = max(1, int(len(all_node_ids) * fraction_train))
selected_node_ids = stdlib_random.sample(all_node_ids, num_selected)
```

- [ ] **Step 2: Evaluate on ALL clients (not just trained ones)**

For evaluation rounds, use all clients (or a separate eval fraction):

```python
# For evaluation: use all clients or a separate eval sample
eval_node_ids = all_node_ids  # Evaluate all for accurate metrics
```

Note: With 6040 clients, evaluating all every round is expensive. Consider evaluating a sample or evaluating less frequently. Add `eval-every-n-rounds` config:

```python
eval_every = int(context.run_config.get("eval-every-n-rounds", 1))
if server_round % eval_every == 0 or server_round == num_rounds:
    # run evaluation
```

- [ ] **Step 3: Commit**

```bash
git add federated-baseline-cf/federated_baseline_cf/server_app.py
git commit -m "feat(baseline): random client sampling + configurable eval frequency"
```

---

### Task 4: Replicate Dataset Changes to Personalized-CF

**Files:**
- Modify: `federated-personalized-cf/federated_personalized_cf/dataset.py`
- Modify: `federated-personalized-cf/federated_personalized_cf/task.py`
- Modify: `federated-personalized-cf/federated_personalized_cf/client_app.py`
- Modify: `federated-personalized-cf/federated_personalized_cf/server_app.py`
- Modify: `federated-personalized-cf/pyproject.toml`

- [ ] **Step 1: Add `natural_partition_users()` to dataset.py**

Copy the exact same function from Task 1 Step 1 into `federated_personalized_cf/dataset.py`.

Modify `load_partition_data()` with the same `partition_mode` parameter and logic from Task 1 Step 2.

- [ ] **Step 2: Wire `partition-mode` through task.py and client_app.py**

Same pattern as Task 2: add parameter to `load_data()`, read from `context.run_config`.

- [ ] **Step 3: Optimize per-user embedding cache for cross-device**

Current cache saves the FULL `user_embeddings.weight` tensor (6040 × 128 = 3.1MB per client). For 6040 clients that's ~18GB of cache. Instead, save only the relevant user's row.

In `client_app.py`, modify `save_local_user_embeddings()`:

```python
def save_local_user_embeddings(
    model, partition_id: int, user_idx: int, round_num: int = 0
):
    """Save only this user's embedding row (cross-device efficient)."""
    cache_dir = get_cache_dir(partition_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "user_embedding.pt"

    state = {
        "user_embedding": model.user_embeddings.weight[user_idx].detach().cpu().clone(),
        "user_bias": model.user_bias.weight[user_idx].detach().cpu().clone(),
        "_round": round_num,
        "_user_idx": user_idx,
    }

    tmp_path = cache_path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, cache_path)
```

Modify `load_local_user_embeddings()`:

```python
def load_local_user_embeddings(
    model, partition_id: int, user_idx: int, device: torch.device
) -> bool:
    """Load this user's embedding row from cache."""
    cache_path = get_cache_dir(partition_id) / "user_embedding.pt"
    if not cache_path.exists():
        return False

    state = torch.load(cache_path, map_location=device, weights_only=True)
    with torch.no_grad():
        model.user_embeddings.weight[user_idx] = state["user_embedding"].to(device)
        model.user_bias.weight[user_idx] = state["user_bias"].to(device)
    return True
```

In the train function, determine `user_idx` from `partition_id`:

```python
partition_mode = context.run_config.get("partition-mode", "dirichlet")
if partition_mode == "natural":
    # In cross-device mode, partition_id IS the user index
    user_idx = partition_id
    save_local_user_embeddings(model, partition_id, user_idx, round_num)
    # ... and for loading:
    load_local_user_embeddings(model, partition_id, user_idx, device)
else:
    # Cross-silo: save full local parameters (existing behavior)
    save_local_parameters_full(model, partition_id, round_num)
```

- [ ] **Step 4: Update pyproject.toml**

```toml
[tool.flwr.app.config]
partition-mode = "natural"
num-server-rounds = 100
local-epochs = 1
embedding-dim = 32
fraction-train = 0.1

[tool.flwr.federations.local-simulation.options]
num-supernodes = 6040
```

- [ ] **Step 5: Add random client sampling to server_app.py**

Same pattern as Task 3.

- [ ] **Step 6: Commit**

```bash
git add federated-personalized-cf/
git commit -m "feat(personalized): cross-device mode with per-user embedding cache"
```

---

### Task 5: Migrate PFedRec (Minimal Changes)

**Files:**
- Modify: `federated-pfedrec/federated_pfedrec/dataset.py`
- Modify: `federated-pfedrec/federated_pfedrec/task.py` (load_data wrapper only)
- Modify: `federated-pfedrec/federated_pfedrec/client_app.py`
- Modify: `federated-pfedrec/federated_pfedrec/server_app.py`
- Modify: `federated-pfedrec/pyproject.toml`

PFedRec is already designed for per-user training. The inner loop in `client_app.py` iterates `for user_idx, (user_items, user_ratings) in user_train_data.items()`. With 1 user per partition, this loop runs exactly once.

- [ ] **Step 1: Add natural partitioning to dataset.py**

Same `natural_partition_users()` function and `load_partition_data()` modification as Task 1.

- [ ] **Step 2: Wire config through task.py and client_app.py**

Same pattern: add `partition_mode` parameter, read from run_config.

- [ ] **Step 3: Verify per-user cache works with natural partitioning**

PFedRec already caches per-user: `.embedding_cache/partition_{id}/user_{user_idx}/affine_output.pt`

In cross-device mode, each partition has exactly 1 user, so cache structure becomes:
```
.embedding_cache/partition_0/user_0/affine_output.pt
.embedding_cache/partition_1/user_1/affine_output.pt
...
.embedding_cache/partition_6039/user_6039/affine_output.pt
```

This works without code changes. Verify that `user_idx` is consistent between `prepare_user_train_data()` and the cache functions.

- [ ] **Step 4: Update pyproject.toml**

```toml
[tool.flwr.app.config]
partition-mode = "natural"
num-server-rounds = 100
local-epochs = 1
latent-dim = 32
lr = 0.1
lr-eta = 80
fraction-train = 0.1

[tool.flwr.federations.local-simulation.options]
num-supernodes = 6040
```

- [ ] **Step 5: Add random client sampling to server_app.py**

Same pattern as Task 3.

- [ ] **Step 6: Commit**

```bash
git add federated-pfedrec/
git commit -m "feat(pfedrec): cross-device mode with natural partitioning"
```

---

### Task 6: Migrate Adaptive-Personalized-CF

**Files:**
- Modify: `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py`
- Modify: `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py`
- Modify: `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py`
- Modify: `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py`
- Modify: `federated-adaptive-personalized-cf/pyproject.toml`

- [ ] **Step 1: Add natural partitioning to dataset.py**

Same function. Also ensure `compute_user_stats()` works for single-user partitions — it already operates per-user, so a partition with 1 user returns stats for that 1 user.

- [ ] **Step 2: Wire config through task.py and client_app.py**

Same pattern as other modules.

- [ ] **Step 3: Add per-user embedding cache (same as Task 4 Step 3)**

Copy the `save_local_user_embeddings()` and `load_local_user_embeddings()` pattern from Personalized-CF.

Additionally, for the adaptive module's extra local state:
- `logit_alpha` (per-user learned alpha) — already per-user, works naturally
- `item_perturbation` — local parameter, cached with model

- [ ] **Step 4: Simplify alpha computation for single-user clients**

In cross-device mode, `compute_client_alpha()` receives stats for 1 user. The weighted average of 1 user's stats IS that user's stats. No code change needed — it already works correctly with 1 user.

`compute_per_user_alpha()` also works: it returns `{user_id: alpha}` for a single user.

Verify this by checking that `user_stats` dict has exactly 1 entry in cross-device mode.

- [ ] **Step 5: Update pyproject.toml**

```toml
[tool.flwr.app.config]
partition-mode = "natural"
num-server-rounds = 100
local-epochs = 1
embedding-dim = 32
fraction-train = 0.1
# Keep thesis-specific configs:
model-type = "dual"
fusion-type = "concat"
mlp-hidden-dims = "128,64,32"   # Scaled down for dim=32
alpha-method = "hierarchical_conditional"
strategy = "fedprox"
proximal-mu = 0.01

[tool.flwr.federations.local-simulation.options]
num-supernodes = 6040
```

- [ ] **Step 6: Add random client sampling to server_app.py**

Same pattern as Task 3.

- [ ] **Step 7: Commit**

```bash
git add federated-adaptive-personalized-cf/
git commit -m "feat(adaptive): cross-device mode with per-user cache and alpha"
```

---

### Task 7: Validation — Run PFedRec Cross-Device First

PFedRec is the best validation target because we have ground truth from the reference implementation (HR@10=72.86%, NDCG@10=44.07%).

**Files:** None (execution only)

- [ ] **Step 1: Clear PFedRec caches**

```bash
cd federated-pfedrec && rm -rf .embedding_cache/
```

- [ ] **Step 2: Run Flower PFedRec in cross-device mode**

```bash
cd federated-pfedrec && flwr run . --run-config \
  "partition-mode=natural latent-dim=32 lr=0.1 lr-eta=80 num-server-rounds=100 local-epochs=1 fraction-train=1.0 num-negatives=4"
```

Note: Start with `fraction-train=1.0` (all 6040 clients per round) to match the reference exactly. This will be slow but provides ground truth.

- [ ] **Step 3: Compare against reference result**

Expected: HR@10 in range 65-73%, NDCG@10 in range 40-45%.

If significantly lower than reference, investigate:
- Aggregation weighting (reference uses uniform average, Flower may use weighted)
- User index consistency between partitioning and cache
- Item embedding LR correctness

- [ ] **Step 4: Run with fraction-train=0.1**

```bash
flwr run . --run-config \
  "partition-mode=natural latent-dim=32 lr=0.1 lr-eta=80 num-server-rounds=100 local-epochs=1 fraction-train=0.1 num-negatives=4"
```

This selects ~604 clients per round (matching PFedRec paper's `clients_sample_ratio` use case).

- [ ] **Step 5: Commit results**

Save results JSON and commit.

---

### Task 8: Validation — Run All Methods Under Standard Config

**Files:** None (execution only)

Standard comparison config (cross-device, same for all methods):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| partition-mode | natural | Cross-device |
| num-supernodes | 6040 | 1 user = 1 client |
| embedding-dim | 32 | Match published baselines |
| num-server-rounds | 100 | Match PFedRec paper |
| local-epochs | 1 | Cross-device standard |
| fraction-train | 0.1 | Sample 604 clients/round |
| eval negatives | 99 | NCF protocol |
| early-stopping | patience=10 | Prevent overfitting |

- [ ] **Step 1: Run Baseline**

```bash
cd federated-baseline-cf && rm -rf .embedding_cache/ && flwr run . --run-config \
  "partition-mode=natural embedding-dim=32 num-server-rounds=100 local-epochs=1 fraction-train=0.1 model-type=bpr early-stopping-enabled=true early-stopping-patience=10"
```

- [ ] **Step 2: Run Personalized**

```bash
cd federated-personalized-cf && rm -rf .embedding_cache/ && flwr run . --run-config \
  "partition-mode=natural embedding-dim=32 num-server-rounds=100 local-epochs=1 fraction-train=0.1 model-type=bpr early-stopping-enabled=true early-stopping-patience=10"
```

- [ ] **Step 3: Run PFedRec**

```bash
cd federated-pfedrec && rm -rf .embedding_cache/ && flwr run . --run-config \
  "partition-mode=natural latent-dim=32 num-server-rounds=100 local-epochs=1 fraction-train=0.1 lr=0.1 lr-eta=80 early-stopping-enabled=true early-stopping-patience=10"
```

- [ ] **Step 4: Run Adaptive (Thesis)**

```bash
cd federated-adaptive-personalized-cf && rm -rf .embedding_cache/ && flwr run . --run-config \
  "partition-mode=natural embedding-dim=32 num-server-rounds=100 local-epochs=1 fraction-train=0.1 model-type=dual fusion-type=concat mlp-hidden-dims=128,64,32 strategy=fedprox proximal-mu=0.01 alpha-method=hierarchical_conditional early-stopping-enabled=true early-stopping-patience=10"
```

- [ ] **Step 5: Compare all results**

Create comparison table:

| Method | What's Local | HR@10 | NDCG@10 | MRR |
|--------|-------------|-------|---------|-----|
| FedAvg Baseline | Nothing | ? | ? | ? |
| Split MF | User embedding | ? | ? | ? |
| PFedRec | Per-user affine | ? | ? | ? |
| Dual-Level (Thesis) | User emb + MLP | ? | ? | ? |
| PFedRec Reference (paper) | Per-user affine | 73.26 | 44.36 | — |

---

## Notes on Centralized Baselines

The centralized baselines (`centralized_baseline_svd.ipynb`, `centralize_baseline_ncf.py`) do NOT need cross-device changes — they don't use federation. They serve as upper-bound reference points. Re-run them at `embedding-dim=32` for a fair capacity comparison.

## Performance Considerations

- **Memory:** 6040 partitions cached in `_dataset_cache` ≈ same memory as original DataFrame (~80MB). Fine.
- **Disk cache:** Per-user embedding = 32 floats × 4 bytes = 128 bytes per user. 6040 users ≈ 1MB total. Negligible.
- **Simulation speed:** With `fraction-train=0.1`, ~604 clients train per round. Each has ~165 ratings, 1 local epoch. Expect ~2-5 min per round depending on parallelism.
- **GPU utilization:** Low per-client. Set Flower backend resources to maximize client parallelism: `num-cpus=2, num-gpus=0.1` per client.

## Backward Compatibility

The `partition-mode` config defaults to `"dirichlet"` in code, but the pyproject.toml defaults change to `"natural"`. To run cross-silo experiments, override at command line:

```bash
flwr run . --run-config "partition-mode=dirichlet" \
  --override federations.local-simulation.options.num-supernodes=5
```
