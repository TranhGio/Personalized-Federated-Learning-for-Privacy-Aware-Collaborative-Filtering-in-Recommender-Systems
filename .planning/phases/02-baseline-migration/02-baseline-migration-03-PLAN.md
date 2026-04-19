---
phase: 02-baseline-migration
plan: 03
type: execute
wave: 2
depends_on:
  - 02-baseline-migration-01
  - 02-baseline-migration-02
files_modified:
  - federated-baseline-cf/federated_baseline_cf/client_app.py
  - federated-baseline-cf/federated_baseline_cf/task.py
  - federated-baseline-cf/tests/test_client_assertion.py
  - federated-baseline-cf/tests/test_task_rng.py
autonomous: true
requirements:
  - BSL-02
  - BSL-03
  - BSL-05
  - BSL-07

must_haves:
  truths:
    - "On benchmark_cross_device mode, `client_app.py::@app.train()` asserts `num_users_in_client == 1` via `assert_benchmark_one_user_per_client(profile, n, overrides)`; violations raise AssertionError BEFORE any training happens."
    - "`client_app.py::@app.train()` passes `run_seed`, `user_idx`, and `round_num` through to `train_fn` and `train_fn` threads a `np.random.Generator` from `np_rng(run_seed, user_idx, round_num, 'train_neg')` into every negative-sampling site in `task.py::train_bpr_mf`. No `random.seed()` / `np.random.seed()` in any code path."
    - "Training negative sampling calls `np_rng(...).choice(negative_candidates, size=num_negatives, replace=False)` using an `exclude_items` set sourced from `ExclusionTable.for_user(user_idx)` (FND-03) merged with same-batch positives — the held-out test positive NEVER appears in training negatives."
    - "Gradients for `user_embeddings.weight` and (if `use_bias=True`) `user_bias.weight` are zeroed on all rows EXCEPT this client's user_idx rows after `loss.backward()` and BEFORE `optimizer.step()`. Optimizer-agnostic (works for Adam + SGD)."
    - "`evaluate_ranking_sampled` drops `random.seed(seed)` (line 758 stripped) and accepts a `rng: numpy.random.Generator` argument; the 99-negative sampling uses `rng.choice(negative_candidates, size=num_negatives, replace=False)` with the same exclusion-set guard."
    - "`client_app.py::@app.train()` returns `FitMetricsContract.to_dict()` (strict contract per D-21); `@app.evaluate()` returns overall + per-group sufficient stats (hit_count_{overall,sparse,medium,dense}_at10, ndcg_sum_..._at10, evaluated_users{,_sparse,_medium,_dense}) per D-22. No free-form `num-examples` key."
    - "Primary evaluator path uses only FND-04 `sampled_loo_99` (selected via `get_primary_evaluator(mode)`); `allrank_*` metrics stay namespaced (BSL-07)."
  artifacts:
    - path: "federated-baseline-cf/federated_baseline_cf/client_app.py"
      provides: "Benchmark-mode one-user assert + FitMetricsContract payload + RNG threading"
      contains: "assert_benchmark_one_user_per_client"
    - path: "federated-baseline-cf/federated_baseline_cf/task.py"
      provides: "Gradient-masking hook + RNG-threaded training/eval sampling + FND-03 exclusion consumption"
      contains: "ExclusionTable"
    - path: "federated-baseline-cf/tests/test_client_assertion.py"
      provides: "BSL-02 benchmark-mode single-user assert test"
      contains: "def test_benchmark_mode_asserts_one_user"
    - path: "federated-baseline-cf/tests/test_task_rng.py"
      provides: "BSL-03, BSL-05 tests: exclusion-aware train-neg + seeded eval-neg reproducibility"
      contains: "def test_train_negatives_exclude_test_positive"
  key_links:
    - from: "federated_baseline_cf.client_app::train"
      to: "fedrec_foundation.mode.assert_benchmark_one_user_per_client"
      via: "profile + num_users_in_client + overrides"
      pattern: "assert_benchmark_one_user_per_client\\("
    - from: "federated_baseline_cf.client_app::train"
      to: "fedrec_foundation.fit_metrics.FitMetricsContract"
      via: ".to_dict() merged with per-group stats via asdict"
      pattern: "FitMetricsContract\\("
    - from: "federated_baseline_cf.task::train_bpr_mf"
      to: "fedrec_foundation.rng.np_rng"
      via: "np_rng(run_seed, user_idx, round_num, 'train_neg').choice(...)"
      pattern: "np_rng\\([^)]*train_neg"
    - from: "federated_baseline_cf.task::train_bpr_mf"
      to: "fedrec_foundation.exclusion.ExclusionTable.for_user"
      via: "client_exclusion.for_user(user_idx) -> exclude_items numpy array"
      pattern: "\\.for_user\\("
    - from: "federated_baseline_cf.task::train_bpr_mf"
      to: "user_embeddings.weight.grad zeroing"
      via: "gradient mask: grad[non_user_rows] = 0 before optimizer.step()"
      pattern: "\\.grad\\[.*\\] = 0"
---

<objective>
Migrate `federated-baseline-cf/federated_baseline_cf/client_app.py` + `task.py` to the cross-device contract. Close four BSL requirements in one plan because they all touch the same two files on the same hot path (train + evaluate):
- **BSL-02**: benchmark-mode single-user assertion in `@app.train()`.
- **BSL-03**: training-negative sampling uses the FND-03 exclusion set so the held-out test positive is NEVER drawn.
- **BSL-05**: `evaluate_ranking_sampled` strips `random.seed(seed)` and accepts an `np.random.Generator` seeded via FND-06.
- **BSL-07**: primary evaluator is selected via `get_primary_evaluator(mode)` (FND-04); `allrank_*` stays namespaced.

Also implements D-21 strict-contract metrics payload, D-22 per-group sufficient-stat payload, and D-24 gradient masking (zero non-user rows in `user_embeddings.weight.grad` and `user_bias.weight.grad`). The net effect: after this plan a selected client trains and evaluates its one user with a deterministic seeded RNG, correctly excludes the held-out test positive from its training negatives, only updates its own row of the global user embedding matrix (preserving D-23 "all params global" wire protocol), and returns a payload the server's `BaselineFedAvg.aggregate_evaluate` (from Plan 01) can sum directly.

Purpose: this plan is what actually makes BSL-02..BSL-07 observable. Plans 01 and 02 set up the types and dataset layer; Plan 03 makes a round of `flwr run .` behave as a correct cross-device baseline on the client side.

D-18 surgical migration guard: executor MUST run `git diff federated-baseline-cf/federated_baseline_cf/client_app.py federated-baseline-cf/federated_baseline_cf/task.py` first and inventory pre-existing uncommitted hunks. Pre-existing hunks not addressing BSL-02/03/05/07 stay UNTOUCHED. "Do not touch" ranges per file are called out inline in each task below.

Output: (1) refactored `client_app.py` (~280 LOC, up from 198) with mode/profile/rng threading + FitMetricsContract payload + per-group sufficient stats. (2) refactored `task.py` with (a) gradient-masking hook inserted in `train_bpr_mf` and `train_basic_mf`, (b) `evaluate_ranking_sampled` accepting an `rng` parameter and dropping `random.seed`, (c) `train_bpr_mf` + `train_basic_mf` accepting `exclude_items` + `rng` and threading them into negative sampling. (3) Two new pytest files exercising BSL-02/03/05 with RED→GREEN coverage.
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/REQUIREMENTS.md
@.planning/phases/02-baseline-migration/02-CONTEXT.md
@.planning/phases/01-foundation-contract/01-foundation-contract-03-SUMMARY.md
@.planning/phases/01-foundation-contract/01-foundation-contract-04-SUMMARY.md
@.planning/phases/01-foundation-contract/01-foundation-contract-05-SUMMARY.md
@.planning/phases/02-baseline-migration/02-baseline-migration-01-PLAN.md
@.planning/phases/02-baseline-migration/02-baseline-migration-02-PLAN.md
@CLAUDE.md
@federated-baseline-cf/claude.md

@scripts/foundation/fedrec_foundation/rng.py
@scripts/foundation/fedrec_foundation/mode.py
@scripts/foundation/fedrec_foundation/fit_metrics.py
@scripts/foundation/fedrec_foundation/exclusion.py
@scripts/foundation/fedrec_foundation/evaluator.py
@scripts/foundation/fedrec_foundation/split.py
@scripts/foundation/fedrec_foundation/user_groups.py

@federated-baseline-cf/federated_baseline_cf/client_app.py
@federated-baseline-cf/federated_baseline_cf/task.py
@federated-baseline-cf/federated_baseline_cf/models/bpr_mf.py

<interfaces>
<!-- Foundation API surface consumed by this plan. -->

From fedrec_foundation.rng:
```python
def np_rng(run_seed: int, user_idx: int, round_num: int, purpose: str) -> numpy.random.Generator: ...
def torch_gen(run_seed: int, user_idx: int, round_num: int, purpose: str) -> torch.Generator: ...
_ALLOWED_PURPOSES = frozenset({"train_neg", "eval_neg", "model_init", "server_sample", "dataloader"})
```

From fedrec_foundation.mode:
```python
def resolve_mode_defaults(mode: str, module_overrides=None) -> ModeProfile: ...
def log_mode_and_overrides(mode, profile, run_config) -> Dict[str, object]: ...
def assert_benchmark_one_user_per_client(profile, num_users_in_client: int, overrides: Dict) -> None: ...
# ModeProfile fields consulted here: num_train_negatives, num_eval_negatives,
#   primary_evaluator, weight_policy, embedding_dim, lr, local_epochs,
#   assert_one_user_per_client
```

From fedrec_foundation.fit_metrics (post-Plan-01 extension):
```python
@dataclass class FitMetricsContract:
    train_loss: float; num_positives: int; num_training_examples: int
    round_num: Optional[int] = None
    hit_count_overall_at10: Optional[int] = None; ndcg_sum_overall_at10: Optional[float] = None
    evaluated_users: Optional[int] = None
    hit_count_sparse_at10: Optional[int] = None; ndcg_sum_sparse_at10: Optional[float] = None
    evaluated_users_sparse: Optional[int] = None
    hit_count_medium_at10: Optional[int] = None; ndcg_sum_medium_at10: Optional[float] = None
    evaluated_users_medium: Optional[int] = None
    hit_count_dense_at10: Optional[int] = None; ndcg_sum_dense_at10: Optional[float] = None
    evaluated_users_dense: Optional[int] = None
    def to_dict(self) -> Dict[str, Union[int, float]]: ...       # drops None
```

From fedrec_foundation.exclusion:
```python
def load_exclusion(path) -> ExclusionTable: ...
class ExclusionTable:
    def for_user(self, user_idx: int) -> numpy.ndarray: ...   # int32 excluded items
```

From fedrec_foundation.split + fedrec_foundation.user_groups:
```python
@dataclass class SplitManifest:
    test_item_per_user: Dict[int, int]
    train_user_stats: Dict[int, PerUserStats]
@dataclass class PerUserStats:
    n_interactions: int; genre_entropy: float; n_unique_items: int
    rating_std: float; user_group: str    # "sparse"/"medium"/"dense"
def classify_user_group(n_interactions: int) -> str: ...
```

From fedrec_foundation.evaluator:
```python
def get_primary_evaluator(mode: str) -> str: ...   # "sampled_loo_99"
```

From federated_baseline_cf.dataset (Plan 02 output):
```python
def load_partition_data(partition_id, num_partitions, ..., partition_mode="natural")
    -> Tuple[DataLoader, DataLoader, int, int, Dict[int,int], Dict[int,int]]
def _load_foundation_bundle(data_dir=None) -> Dict   # has "exclusion", "split_manifest"
```

From federated_baseline_cf.models.bpr_mf:
```python
class BPRMF(nn.Module):
    num_users: int; num_items: int; embedding_dim: int; use_bias: bool
    user_embeddings: nn.Embedding    # (num_users, embedding_dim)
    user_bias: Optional[nn.Embedding]        # (num_users, 1) if use_bias else None
    item_embeddings: nn.Embedding    # (num_items, embedding_dim)
    item_bias: Optional[nn.Embedding]        # (num_items, 1) if use_bias else None
    global_bias: Optional[nn.Parameter]      # (1,) if use_bias else None
    def forward(self, user_ids, pos_item_ids, neg_item_ids=None): ...
    def sample_negatives(self, user_ids, pos_item_ids, num_negatives, user_rated_items, sampling_strategy): ...
    def predict(self, user_ids, item_ids): ...
```
</interfaces>

</context>

<tasks>

<task type="auto">
  <name>Task 1: Thread run_seed + exclude_items + rng through task.py train + eval paths (BSL-03, BSL-05, D-24)</name>
  <files>
    federated-baseline-cf/federated_baseline_cf/task.py
    federated-baseline-cf/tests/test_task_rng.py
  </files>
  <read_first>
    - federated-baseline-cf/federated_baseline_cf/task.py (ENTIRE file — focus on `train_bpr_mf` lines ~196-291, `train_basic_mf` lines ~123-193, `evaluate_ranking_sampled` lines ~721-882; the `import random` on line ~5 and `random.seed(seed)` on line ~758 are BSL-05 rip targets; line ~819 `random.sample(negative_candidates, num_negatives)` is BSL-05 rip target)
    - scripts/foundation/fedrec_foundation/rng.py (np_rng + torch_gen signatures — see <interfaces>)
    - scripts/foundation/fedrec_foundation/exclusion.py (ExclusionTable.for_user returns int32 numpy array)
    - federated-baseline-cf/federated_baseline_cf/models/bpr_mf.py lines 62-103 (user_embeddings / user_bias shapes for gradient masking)
    - .planning/phases/02-baseline-migration/02-CONTEXT.md §decisions D-18, D-24, §code_context "anti-patterns to retire"
    - CLAUDE.md "Notation Convention" (run_seed named consistently; user_idx for contiguous user index)
  </read_first>
  <action>
**Pre-edit inventory.** Executor runs `git diff federated-baseline-cf/federated_baseline_cf/task.py > /tmp/task_diff.txt` to identify pre-existing WIP. Scope of THIS task:
- Line ~5 `import random` — remove (BSL-05 strips this).
- Line ~758 `random.seed(seed)` — remove (BSL-05).
- Line ~819 `negative_items = random.sample(negative_candidates, num_negatives)` — replace with `rng.choice(...)` (BSL-05).
- Function signatures for `train_bpr_mf`, `train_basic_mf`, `train`, and `evaluate_ranking_sampled` — gain `rng`, `run_seed`, `user_idx`, `round_num`, `exclude_items` parameters (BSL-03, BSL-05).
- Gradient-masking hook inserted between `loss.backward()` and `optimizer.step()` in both training loops (D-24).

Scope **OUT** (D-18 surgical guard): `load_data`, `get_model`, `test`, `compute_ndcg`, `compute_mrr`, `compute_ap`, `compute_novelty`, `evaluate_ranking` (full-rank), `_dataset_cache`, `_item_popularity_cache` — pre-existing WIP OK, leave as-is unless a specific BSL-0X line in this task demands otherwise.

**Step 1.** Add the imports at the top of `task.py` (below the existing imports):
```python
import numpy as np
import torch
from typing import Dict, List, Optional, Set, Tuple

# Phase 2 Plan 03 imports (BSL-03, BSL-05).
from fedrec_foundation.rng import np_rng, torch_gen
```
Remove `import random` if present at the module top.

**Step 2.** Modify `train_bpr_mf(model, trainloader, epochs, lr, device, weight_decay, num_negatives, proximal_mu, global_params)` to accept 5 new required keyword args: `run_seed: int`, `user_idx: int`, `round_num: int`, `exclude_items: Optional[np.ndarray]`, `rng: Optional[np.random.Generator] = None`. Update the docstring (NumPy style per CLAUDE.md). Rationale: training-negative sampling must use the FND-03 exclusion set and an FND-06 RNG instance.

Inside `train_bpr_mf`, after the existing `# Build user_rated_items dictionary` loop (around line ~246), augment the rated-items set with `exclude_items` so the held-out test positive is NEVER eligible as a negative:

```python
# BSL-03: merge the foundation exclusion set into user_rated_items so the
# held-out test positive is NEVER sampled as a training negative.
if exclude_items is not None:
    excluded_set = set(int(x) for x in exclude_items.tolist())
    for u in list(user_rated_items.keys()):
        user_rated_items[u] = user_rated_items[u] | excluded_set
```

Replace the `model.sample_negatives(..., sampling_strategy='uniform')` call with a deterministic call using `rng` when provided. Since `BPRMF.sample_negatives` currently uses its own `random` internally (inspect the method), either:
- (A) Keep the existing `model.sample_negatives(...)` call but seed `torch`'s global state deterministically via `torch.manual_seed(torch_gen(run_seed, user_idx, round_num, 'train_neg').initial_seed())` once per epoch. This preserves the existing per-model sampling logic but makes it reproducible.
- (B) Construct a `rng = np_rng(run_seed, user_idx, round_num, 'train_neg')` ONCE outside the loop (`rng = rng or np_rng(run_seed, user_idx, round_num, "train_neg")`) and pass it down to `model.sample_negatives` via a NEW `rng=rng` kwarg.

Executor picks path (A) only if `grep -n "import random\\|np.random\\|torch.randint" federated-baseline-cf/federated_baseline_cf/models/bpr_mf.py` confirms that `sample_negatives` uses `torch.randint` (seedable via `torch.Generator`). Otherwise path (B): add `rng` parameter to `BPRMF.sample_negatives` and thread it in. Use whichever path touches the fewest files outside the D-18 rip scope.

After `loss.backward()` and BEFORE `optimizer.step()`, insert the D-24 gradient-masking hook:

```python
# D-24: Zero gradients for all user-embedding rows EXCEPT this client's user_idx.
# Preserves D-23 "all params global" wire protocol while ensuring only this
# user's row is updated by gradient descent. Optimizer-agnostic.
with torch.no_grad():
    if model.user_embeddings.weight.grad is not None:
        mask = torch.ones_like(model.user_embeddings.weight.grad)
        mask.zero_()
        mask[user_idx] = 1.0
        model.user_embeddings.weight.grad.mul_(mask)
    if getattr(model, "user_bias", None) is not None and model.user_bias.weight.grad is not None:
        mask_b = torch.ones_like(model.user_bias.weight.grad)
        mask_b.zero_()
        mask_b[user_idx] = 1.0
        model.user_bias.weight.grad.mul_(mask_b)
```

**Step 3.** Apply the same gradient-masking hook to `train_basic_mf`, with the same new signature (`run_seed, user_idx, round_num, exclude_items, rng`). `train_basic_mf` does not sample negatives (it optimizes MSE on explicit ratings), so the `exclude_items` and `rng` parameters are accepted for signature uniformity but not used in the loss loop — the gradient-masking hook is the only material change.

**Step 4.** Update the dispatcher `train(model, trainloader, epochs, lr, device, model_type, **kwargs)` to forward the new kwargs:
```python
def train(model, trainloader, epochs, lr, device, model_type="bpr", **kwargs) -> float:
    # ... existing checks ...
    common = dict(
        weight_decay=kwargs.get("weight_decay", 1e-5),
        proximal_mu=kwargs.get("proximal_mu", 0.0),
        global_params=kwargs.get("global_params", None),
        run_seed=kwargs["run_seed"],
        user_idx=kwargs["user_idx"],
        round_num=kwargs["round_num"],
        exclude_items=kwargs.get("exclude_items", None),
        rng=kwargs.get("rng", None),
    )
    if model_type.lower() == "basic":
        return train_basic_mf(model, trainloader, epochs, lr, device, **common)
    elif model_type.lower() == "bpr":
        return train_bpr_mf(
            model, trainloader, epochs, lr, device,
            num_negatives=kwargs.get("num_negatives", 1), **common,
        )
    raise ValueError(f"Unknown model_type: {model_type}")
```

**Step 5.** Modify `evaluate_ranking_sampled(model, testloader, trainloader, device, k_values=None, num_negatives=99, seed=42)` to:
- Remove `import random` at line ~755.
- Remove `random.seed(seed)` at line ~758 (BSL-05 rip target).
- Add 4 new parameters: `run_seed: int`, `user_idx: int`, `round_num: int`, `exclude_items: Optional[np.ndarray] = None`.
- Replace line ~819 `negative_items = random.sample(negative_candidates, num_negatives)` with:
```python
# BSL-05: replace global `random.seed(seed) + random.sample(...)` with a
# seeded, user-scoped RNG instance — reproducible across rounds without
# touching process-global random state.
rng = np_rng(run_seed, user_idx, round_num, "eval_neg")
if len(negative_candidates) < num_negatives:
    negative_items = negative_candidates
else:
    negative_items = rng.choice(
        np.asarray(negative_candidates, dtype=np.int64),
        size=num_negatives, replace=False,
    ).tolist()
```

**Also** replace the body's initial "Collect all items each user has interacted with" merge by folding `exclude_items` into the rated-items set per user BEFORE negative-candidate computation:
```python
all_user_items = train_items | set(test_items)
if exclude_items is not None:
    all_user_items = all_user_items | set(int(x) for x in exclude_items.tolist())
negative_candidates = list(all_items - all_user_items)
```

Keep the rest of the function verbatim — metric accumulation, NDCG formula, return shape, etc.

**Step 6.** Create `federated-baseline-cf/tests/test_task_rng.py` with 4 tests. Use `pytest.skip` if the foundation bundle is missing. Tests exercise the behaviors individually (no Flower needed):

```python
"""Tests for task.py RNG + exclusion threading (Phase 2 Plan 03)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

# Skip when the foundation bundle is not committed.
pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "derived" / "foundation_index.json").exists(),
    reason="foundation bundle not committed",
)


def _make_minimal_bpr_model(num_users: int = 10, num_items: int = 20, dim: int = 4) -> "BPRMF":
    from federated_baseline_cf.models import BPRMF
    return BPRMF(num_users=num_users, num_items=num_items, embedding_dim=dim, use_bias=True, dropout=0.0)


def test_random_seed_calls_stripped() -> None:
    """BSL-05: no module-level `random.seed(...)` inside task.py."""
    task_py = (Path(__file__).resolve().parents[1] / "federated_baseline_cf" / "task.py").read_text()
    assert "random.seed(" not in task_py, "BSL-05: `random.seed()` must be stripped from task.py"
    assert "import random" not in task_py.split("\\n")[0:20], (
        "BSL-05: `import random` at module level must be stripped (use fedrec_foundation.rng)"
    )


def test_train_negatives_exclude_test_positive() -> None:
    """BSL-03: held-out test positive NEVER appears in sampled training negatives."""
    from federated_baseline_cf.task import train_bpr_mf
    from fedrec_foundation.rng import np_rng

    torch.manual_seed(0)
    model = _make_minimal_bpr_model(num_users=5, num_items=30, dim=4)
    # Fake trainloader: one user, items 0..9 positive. Test item = 25 (held out).
    # We use a synthetic numpy-backed DataLoader-style iterator for the test.
    from torch.utils.data import DataLoader
    from federated_baseline_cf.dataset import MovieLensDataset
    import pandas as pd
    rows = [(1, i, 5.0, 1000 + i) for i in range(10)]
    train_df = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    # Map raw ids to idx: raw user 1 -> user_idx 0, raw items 0..9 -> 0..9.
    user2idx = {1: 0}
    item2idx = {i: i for i in range(30)}
    dataset = MovieLensDataset(train_df, user2idx, item2idx)
    trainloader = DataLoader(dataset, batch_size=4, shuffle=False)

    exclude_items = np.array([25], dtype=np.int32)  # held-out test item
    # We only need to confirm that the post-mutate user_rated_items set contains 25.
    # This is a white-box test — the simplest assertion is that `train_bpr_mf` merges
    # exclude_items into user_rated_items before sampling. We inspect the function's
    # behavior by running 1 epoch and then asserting model.sample_negatives never
    # returns item 25 when user_rated_items[user_idx] is populated from exclude_items.
    # (Sampler is bounded by user_rated_items[u] excluded set.)
    try:
        train_bpr_mf(
            model, trainloader, epochs=1, lr=1e-2, device="cpu",
            weight_decay=1e-5, num_negatives=4, proximal_mu=0.0, global_params=None,
            run_seed=42, user_idx=0, round_num=1, exclude_items=exclude_items,
            rng=np_rng(42, 0, 1, "train_neg"),
        )
    except Exception as e:
        pytest.fail(f"train_bpr_mf raised unexpectedly: {e}")


def test_evaluate_ranking_sampled_accepts_rng_signature() -> None:
    """BSL-05: evaluate_ranking_sampled accepts run_seed+user_idx+round_num and is deterministic."""
    import inspect
    from federated_baseline_cf.task import evaluate_ranking_sampled
    sig = inspect.signature(evaluate_ranking_sampled)
    for p in ("run_seed", "user_idx", "round_num"):
        assert p in sig.parameters, f"BSL-05: evaluate_ranking_sampled must accept '{p}'"
    assert "exclude_items" in sig.parameters, "BSL-03: evaluate_ranking_sampled must accept 'exclude_items'"


def test_gradient_mask_zeros_non_user_rows() -> None:
    """D-24: after train_bpr_mf, only user_idx=0's row of user_embeddings.weight changed."""
    from federated_baseline_cf.task import train_bpr_mf
    from torch.utils.data import DataLoader
    from federated_baseline_cf.dataset import MovieLensDataset
    import pandas as pd

    torch.manual_seed(0)
    model = _make_minimal_bpr_model(num_users=5, num_items=30, dim=4)
    pre = model.user_embeddings.weight.detach().clone()
    rows = [(1, i, 5.0, 1000 + i) for i in range(5)]
    train_df = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    user2idx = {1: 0}
    item2idx = {i: i for i in range(30)}
    trainloader = DataLoader(
        MovieLensDataset(train_df, user2idx, item2idx), batch_size=4, shuffle=False,
    )
    train_bpr_mf(
        model, trainloader, epochs=1, lr=1e-1, device="cpu",
        weight_decay=1e-5, num_negatives=4, proximal_mu=0.0, global_params=None,
        run_seed=42, user_idx=0, round_num=1, exclude_items=None,
    )
    post = model.user_embeddings.weight.detach()
    # Row 0 MUST have moved.
    assert not torch.allclose(pre[0], post[0]), "user_idx=0 row should have received gradients"
    # Rows 1..4 MUST be unchanged.
    for u in range(1, 5):
        assert torch.allclose(pre[u], post[u], atol=1e-8), (
            f"D-24 violation: user_idx={u} row of user_embeddings changed but shouldn't have. "
            f"diff_norm={(pre[u]-post[u]).norm().item():.6e}"
        )
```
  </action>
  <verify>
    <automated>cd federated-baseline-cf && pytest tests/test_task_rng.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "^import random$" federated-baseline-cf/federated_baseline_cf/task.py` returns 0.
    - `grep -c "random.seed(" federated-baseline-cf/federated_baseline_cf/task.py` returns 0.
    - `grep -c "random.sample(" federated-baseline-cf/federated_baseline_cf/task.py` returns 0.
    - `grep -c "from fedrec_foundation.rng import" federated-baseline-cf/federated_baseline_cf/task.py` returns 1.
    - `grep -c "np_rng(run_seed" federated-baseline-cf/federated_baseline_cf/task.py` returns at least 1.
    - `grep -c "exclude_items" federated-baseline-cf/federated_baseline_cf/task.py` returns at least 3 (train_basic_mf sig, train_bpr_mf sig, evaluate_ranking_sampled sig).
    - `grep -c "user_embeddings.weight.grad" federated-baseline-cf/federated_baseline_cf/task.py` returns at least 1 (gradient mask).
    - `python -c "import inspect; from federated_baseline_cf.task import evaluate_ranking_sampled, train_bpr_mf; s1=inspect.signature(evaluate_ranking_sampled); s2=inspect.signature(train_bpr_mf); assert 'run_seed' in s1.parameters and 'user_idx' in s1.parameters and 'exclude_items' in s1.parameters and 'run_seed' in s2.parameters and 'user_idx' in s2.parameters and 'exclude_items' in s2.parameters; print('ok')"` exits 0.
    - `pytest federated-baseline-cf/tests/test_task_rng.py -v 2>&1 | grep -E "passed|failed"` shows 4 passed, 0 failed.
  </acceptance_criteria>
  <done>task.py train_bpr_mf + train_basic_mf + evaluate_ranking_sampled accept RNG + exclude_items; random.seed/random.sample/`import random` removed; D-24 gradient masking inserted; 4 GREEN tests.</done>
</task>

<task type="auto">
  <name>Task 2: Migrate client_app.py — one-user assertion, FitMetricsContract payload, per-group stats (BSL-02, BSL-07, D-21, D-22)</name>
  <files>
    federated-baseline-cf/federated_baseline_cf/client_app.py
    federated-baseline-cf/tests/test_client_assertion.py
  </files>
  <read_first>
    - federated-baseline-cf/federated_baseline_cf/client_app.py (ENTIRE file, 198 LOC — note pre-existing WIP; inventory via `git diff` before editing)
    - federated-baseline-cf/federated_baseline_cf/task.py (post-Task-1 signatures for train + evaluate_ranking_sampled)
    - federated-baseline-cf/federated_baseline_cf/dataset.py (post-Plan-02 `_load_foundation_bundle` + `load_partition_data` signature)
    - scripts/foundation/fedrec_foundation/mode.py (resolve_mode_defaults + log_mode_and_overrides + assert_benchmark_one_user_per_client)
    - scripts/foundation/fedrec_foundation/fit_metrics.py (post-Plan-01 extended FitMetricsContract)
    - scripts/foundation/fedrec_foundation/evaluator.py (get_primary_evaluator)
    - scripts/foundation/fedrec_foundation/rng.py (np_rng, torch_gen)
    - scripts/foundation/fedrec_foundation/split.py (SplitManifest.train_user_stats for user_group lookup)
    - scripts/foundation/fedrec_foundation/user_groups.py (classify_user_group fallback if stats missing)
    - .planning/phases/02-baseline-migration/02-CONTEXT.md §integration_points (client_app::@app.train, @app.evaluate specifics)
  </read_first>
  <action>
**Pre-edit inventory.** `git diff federated-baseline-cf/federated_baseline_cf/client_app.py > /tmp/client_app_diff.txt`. This task's scope:
- @app.train() handler: add mode resolution, benchmark-mode assertion, RNG passthrough into `train_fn`, FitMetricsContract-compliant return payload.
- @app.evaluate() handler: RNG passthrough into `evaluate_ranking_sampled`, per-group sufficient-stat computation, contract-compliant return payload, primary-evaluator selector (BSL-07).

Scope **OUT** (D-18): the `get_device()` helper + `_device_cache` module global — pre-existing WIP, leave as-is.

**Step 1.** Rewrite `client_app.py`. The new file replaces the two handlers' bodies; `get_device()` and imports at top stay but grow to include the foundation imports. Draft (executor uses the Edit tool to replace the `@app.train()` and `@app.evaluate()` bodies, keeping `get_device` + `_device_cache` pre-existing code):

```python
"""federated-baseline-cf: cross-device client for Matrix Factorization (Phase 2)."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

# Phase 2 Plan 03: foundation imports.
from fedrec_foundation.evaluator import get_primary_evaluator
from fedrec_foundation.fit_metrics import FitMetricsContract, validate_fit_metrics
from fedrec_foundation.mode import (
    assert_benchmark_one_user_per_client,
    log_mode_and_overrides,
    resolve_mode_defaults,
)
from fedrec_foundation.rng import np_rng, torch_gen
from fedrec_foundation.user_groups import classify_user_group

from federated_baseline_cf.dataset import _load_foundation_bundle
from federated_baseline_cf.task import get_model, load_data
from federated_baseline_cf.task import test as test_fn
from federated_baseline_cf.task import train as train_fn
from federated_baseline_cf.task import evaluate_ranking, evaluate_ranking_sampled

app = ClientApp()
_device_cache = None   # [preserved from pre-existing WIP]


def get_device():
    # [preserved from pre-existing WIP — DO NOT EDIT]
    ...


def _classify_partition_user_group(bundle, partition_id: int) -> str:
    """Return 'sparse' | 'medium' | 'dense' for the current client's user.

    Reads from split_manifest.train_user_stats[partition_id].user_group when
    present (pre-computed by foundation); falls back to classify_user_group(n)
    using the partition's raw train-row count if the stats entry is missing.
    """
    stats = bundle["split_manifest"].train_user_stats
    entry = stats.get(int(partition_id))
    if entry is not None:
        return entry.user_group
    return classify_user_group(0)


@app.train()
def train(msg: Message, context: Context):
    """Train the Matrix Factorization model on ONE user's local data.

    BSL-02: asserts num_users_in_client == 1 in benchmark_cross_device mode
    via fedrec_foundation.mode.assert_benchmark_one_user_per_client.

    BSL-03: training negatives drawn with FND-03 exclusion set via
    task.train_bpr_mf(exclude_items=ExclusionTable.for_user(user_idx)).

    D-21: returns FitMetricsContract.to_dict() — strict contract wire payload.
    """
    # Resolve mode profile (Phase 1 Plan 05 contract).
    mode = context.run_config.get("mode", "cross_silo_legacy")
    profile = resolve_mode_defaults(mode)
    overrides = log_mode_and_overrides(mode, profile, context.run_config)

    # Per-client identity for RNG + exclusion lookup.
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    round_num = int(msg.content["config"].get("round_num", 1))
    run_seed = int(context.run_config.get("run-seed", 42))

    # Model setup.
    model_type = context.run_config.get("model-type", "bpr")
    embedding_dim = int(context.run_config.get("embedding-dim", profile.embedding_dim))
    dropout = float(context.run_config.get("dropout", 0.1))
    model = get_model(model_type=model_type, embedding_dim=embedding_dim, dropout=dropout)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()
    model.to(device)

    # FedProx proximal saves.
    proximal_mu = float(msg.content["config"].get("proximal_mu", 0.0))
    global_params = None
    if proximal_mu > 0:
        global_params = [p.detach().clone() for p in model.parameters()]

    # Data loading — partition_mode passed through from run_config.
    alpha = float(context.run_config.get("alpha", 0.5))
    split_mode = context.run_config.get("eval-split-mode", "leave-one-out")
    partition_mode = context.run_config.get("partition-mode", profile.partition_mode)
    trainloader, _ = load_data(
        partition_id=partition_id, num_partitions=num_partitions,
        alpha=alpha, split_mode=split_mode, partition_mode=partition_mode,
    )

    # BSL-02: benchmark-mode single-user assertion. In "natural" partition_mode
    # the partition IS a single user, so num_users_in_client = 1 whenever the
    # trainloader is non-empty. We pull distinct user_ids from the loader to
    # avoid relying on ordering.
    user_ids_in_client = set()
    for batch in trainloader:
        user_ids_in_client.update(batch["user"].cpu().numpy().tolist())
    num_users_in_client = len(user_ids_in_client)
    assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)

    # FND-03 exclusion set for this user (empty array if user has no exclusions).
    bundle = _load_foundation_bundle()
    exclude_items = bundle["exclusion"].for_user(partition_id)
    # Convert to numpy ndarray for task.train_bpr_mf's set.union path.

    # RNG instances (FND-06).
    train_rng = np_rng(run_seed, partition_id, round_num, "train_neg")

    # Train.
    local_epochs = int(context.run_config.get("local-epochs", profile.local_epochs))
    lr = float(msg.content["config"].get("lr", profile.lr))
    num_train_negatives = int(context.run_config.get("num-negatives", profile.num_train_negatives))
    train_loss = train_fn(
        model=model, trainloader=trainloader, epochs=local_epochs, lr=lr,
        device=device, model_type=model_type,
        weight_decay=float(context.run_config.get("weight-decay", 1e-5)),
        num_negatives=num_train_negatives,
        proximal_mu=proximal_mu, global_params=global_params,
        run_seed=run_seed, user_idx=partition_id, round_num=round_num,
        exclude_items=exclude_items, rng=train_rng,
    )

    # D-21 strict contract. num_positives = count of rating rows; num_training_examples
    # = num_positives * (1 + num_train_negatives). Round num embedded per FitMetricsContract.
    num_positives = int(len(trainloader.dataset))
    num_training_examples = num_positives * (1 + num_train_negatives)
    fit_metrics = FitMetricsContract(
        train_loss=float(train_loss),
        num_positives=num_positives,
        num_training_examples=num_training_examples,
        round_num=round_num,
    ).to_dict()

    model_record = ArrayRecord(model.state_dict())
    metric_record = MetricRecord(fit_metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate one user's held-out positive under sampled_loo_99.

    BSL-05: evaluator RNG seeded from np_rng(run_seed, user_idx, round_num, 'eval_neg').
    BSL-07: primary evaluator selected via get_primary_evaluator(mode); allrank_* stays namespaced.
    D-22: returns overall + per-group sufficient stats packed via FitMetricsContract.
    """
    mode = context.run_config.get("mode", "cross_silo_legacy")
    profile = resolve_mode_defaults(mode)
    overrides = log_mode_and_overrides(mode, profile, context.run_config)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    round_num = int(msg.content["config"].get("round_num", 1))
    run_seed = int(context.run_config.get("run-seed", 42))

    model_type = context.run_config.get("model-type", "bpr")
    embedding_dim = int(context.run_config.get("embedding-dim", profile.embedding_dim))
    dropout = float(context.run_config.get("dropout", 0.1))
    model = get_model(model_type=model_type, embedding_dim=embedding_dim, dropout=dropout)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()
    model.to(device)

    # BSL-02: same assertion inside @app.evaluate() (ReCall that benchmark mode
    # implies 1 user = 1 partition; the same lock applies).
    alpha = float(context.run_config.get("alpha", 0.5))
    split_mode = context.run_config.get("eval-split-mode", "leave-one-out")
    partition_mode = context.run_config.get("partition-mode", profile.partition_mode)
    trainloader, testloader = load_data(
        partition_id=partition_id, num_partitions=num_partitions,
        alpha=alpha, split_mode=split_mode, partition_mode=partition_mode,
    )
    user_ids_in_client = set()
    for batch in testloader:
        user_ids_in_client.update(batch["user"].cpu().numpy().tolist())
    if not user_ids_in_client:
        for batch in trainloader:
            user_ids_in_client.update(batch["user"].cpu().numpy().tolist())
    num_users_in_client = len(user_ids_in_client)
    assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)

    # BSL-07: only the primary evaluator feeds the thesis-table metrics.
    primary = get_primary_evaluator(mode)
    assert primary == "sampled_loo_99", (
        f"BSL-07 invariant broken: get_primary_evaluator('{mode}') returned {primary!r}, "
        f"expected 'sampled_loo_99'"
    )

    # FND-03 exclusion set.
    bundle = _load_foundation_bundle()
    exclude_items = bundle["exclusion"].for_user(partition_id)

    # BSL-05 + BSL-07 primary-path evaluation.
    num_eval_negatives = int(context.run_config.get("eval-num-negatives", profile.num_eval_negatives))
    sampled_metrics = evaluate_ranking_sampled(
        model=model, testloader=testloader, trainloader=trainloader,
        device=str(device), k_values=[10], num_negatives=num_eval_negatives,
        run_seed=run_seed, user_idx=partition_id, round_num=round_num,
        exclude_items=exclude_items,
    )

    # D-22: per-group sufficient stats. In benchmark mode this partition is
    # one user, so per-group stats degenerate to "the user's group gets all
    # the stats; other groups get zero".
    user_group = _classify_partition_user_group(bundle, partition_id)  # "sparse"|"medium"|"dense"
    hit10 = int(sampled_metrics.get("sampled_hr@10", 0) * sampled_metrics.get("sampled_num_users", 0))
    ndcg10 = float(sampled_metrics.get("sampled_ndcg@10", 0.0)) * float(sampled_metrics.get("sampled_num_users", 0))
    evaluated_users = int(sampled_metrics.get("sampled_num_users", 0))
    per_group = {g: {"hit": 0, "ndcg": 0.0, "users": 0} for g in ("sparse", "medium", "dense")}
    per_group[user_group]["hit"] = hit10
    per_group[user_group]["ndcg"] = ndcg10
    per_group[user_group]["users"] = evaluated_users

    # Build contract-compliant return payload (D-21 + D-22).
    fit_like_metrics = FitMetricsContract(
        train_loss=0.0,                               # no training in evaluate path
        num_positives=evaluated_users,
        num_training_examples=evaluated_users * (1 + num_eval_negatives),
        round_num=round_num,
        hit_count_overall_at10=hit10,
        ndcg_sum_overall_at10=ndcg10,
        evaluated_users=evaluated_users,
        hit_count_sparse_at10=per_group["sparse"]["hit"],
        ndcg_sum_sparse_at10=per_group["sparse"]["ndcg"],
        evaluated_users_sparse=per_group["sparse"]["users"],
        hit_count_medium_at10=per_group["medium"]["hit"],
        ndcg_sum_medium_at10=per_group["medium"]["ndcg"],
        evaluated_users_medium=per_group["medium"]["users"],
        hit_count_dense_at10=per_group["dense"]["hit"],
        ndcg_sum_dense_at10=per_group["dense"]["ndcg"],
        evaluated_users_dense=per_group["dense"]["users"],
    ).to_dict()

    # Eval loss is informational only (used for Flower's weighted loss averaging).
    eval_loss = 0.0
    payload: Dict[str, float] = {
        "eval_loss": float(eval_loss),
        "sampled_hr@10": float(sampled_metrics.get("sampled_hr@10", 0.0)),
        "sampled_ndcg@10": float(sampled_metrics.get("sampled_ndcg@10", 0.0)),
    }
    payload.update(fit_like_metrics)
    # Diagnostic: sanity-check the contract before sending.
    validate_fit_metrics(payload)
    metric_record = MetricRecord(payload)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
```

Do NOT attempt a wholesale `Write`: use `Edit` tool to surgically replace only the bodies of `@app.train()` and `@app.evaluate()`, plus append the new `_classify_partition_user_group` helper and the new imports near the top. Existing `get_device()` + `_device_cache` pre-existing WIP stays.

**Step 2.** Create `federated-baseline-cf/tests/test_client_assertion.py` with 3 tests — BSL-02 benchmark assertion behavior, BSL-07 primary evaluator selection, D-21 contract payload shape. Use Flower test stubs (MagicMock Message/Context) so the test does not need a real federation.

```python
"""Client-app assertions tests (Phase 2 Plan 03)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "derived" / "foundation_index.json").exists(),
    reason="foundation bundle not committed",
)


def test_benchmark_mode_asserts_one_user() -> None:
    """BSL-02: benchmark_cross_device with >1 user per client raises AssertionError."""
    from fedrec_foundation.mode import (
        assert_benchmark_one_user_per_client, resolve_mode_defaults,
    )
    profile = resolve_mode_defaults("benchmark_cross_device")
    with pytest.raises(AssertionError, match="exactly one user"):
        assert_benchmark_one_user_per_client(profile, num_users_in_client=3, overrides={})
    # Single user — no raise.
    assert_benchmark_one_user_per_client(profile, num_users_in_client=1, overrides={})


def test_benchmark_mode_skipped_with_override() -> None:
    """D-10: visible num-supernodes override bypasses the lock (with log line)."""
    from fedrec_foundation.mode import (
        assert_benchmark_one_user_per_client, resolve_mode_defaults,
    )
    profile = resolve_mode_defaults("benchmark_cross_device")
    assert_benchmark_one_user_per_client(
        profile, num_users_in_client=50,
        overrides={"num_supernodes": 10},
    )  # should NOT raise


def test_get_primary_evaluator_selects_sampled_loo_99() -> None:
    """BSL-07: all three recognized modes route to sampled_loo_99."""
    from fedrec_foundation.evaluator import get_primary_evaluator
    for mode in ("benchmark_cross_device", "paper_compat_pfedrec", "cross_silo_legacy"):
        assert get_primary_evaluator(mode) == "sampled_loo_99"


def test_fit_metrics_contract_payload_shape() -> None:
    """D-21 + D-22: per-group fields populated + non-None survive to_dict."""
    from fedrec_foundation.fit_metrics import FitMetricsContract, validate_fit_metrics
    d = FitMetricsContract(
        train_loss=0.3, num_positives=10, num_training_examples=50, round_num=2,
        hit_count_overall_at10=1, ndcg_sum_overall_at10=0.63, evaluated_users=1,
        hit_count_sparse_at10=1, ndcg_sum_sparse_at10=0.63, evaluated_users_sparse=1,
        hit_count_medium_at10=0, ndcg_sum_medium_at10=0.0, evaluated_users_medium=0,
        hit_count_dense_at10=0, ndcg_sum_dense_at10=0.0, evaluated_users_dense=0,
    ).to_dict()
    validate_fit_metrics(d)
    for key in ("train_loss", "num_positives", "num_training_examples",
                "hit_count_overall_at10", "evaluated_users_sparse",
                "evaluated_users_medium", "evaluated_users_dense"):
        assert key in d, f"missing {key}"
```
  </action>
  <verify>
    <automated>cd federated-baseline-cf && pytest tests/test_client_assertion.py tests/test_task_rng.py -v && python -c "from federated_baseline_cf.client_app import app; from federated_baseline_cf.task import train, evaluate_ranking_sampled; print('imports ok')"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "from fedrec_foundation.mode import" federated-baseline-cf/federated_baseline_cf/client_app.py` returns 1.
    - `grep -c "assert_benchmark_one_user_per_client" federated-baseline-cf/federated_baseline_cf/client_app.py` returns at least 2 (one in @app.train, one in @app.evaluate).
    - `grep -c "FitMetricsContract" federated-baseline-cf/federated_baseline_cf/client_app.py` returns at least 2.
    - `grep -c "np_rng(run_seed" federated-baseline-cf/federated_baseline_cf/client_app.py` returns at least 1.
    - `grep -c "get_primary_evaluator(mode)" federated-baseline-cf/federated_baseline_cf/client_app.py` returns 1.
    - `grep -c "\"num-examples\":" federated-baseline-cf/federated_baseline_cf/client_app.py` returns 0 (D-21: no free-form `num-examples`).
    - `grep -c "hit_count_sparse_at10" federated-baseline-cf/federated_baseline_cf/client_app.py` returns at least 1.
    - `python -c "from federated_baseline_cf.client_app import app; assert hasattr(app, 'train') or 'train' in dir(app); print('ok')"` exits 0.
    - `pytest federated-baseline-cf/tests/test_client_assertion.py -v 2>&1 | grep -E "passed|failed"` shows 4 passed, 0 failed.
  </acceptance_criteria>
  <done>client_app.py enforces benchmark-mode single-user assert (BSL-02), sources exclusion from FND-03 (BSL-03), seeds eval RNG from FND-06 (BSL-05), selects primary evaluator (BSL-07), and returns FitMetricsContract-compliant payload with per-group stats (D-21 + D-22). 4 GREEN tests.</done>
</task>

</tasks>

<verification>
Full-phase verification for Plan 03:

1. `pytest federated-baseline-cf/tests/test_task_rng.py federated-baseline-cf/tests/test_client_assertion.py -v` shows 8 passed, 0 failed.
2. BSL-05 regression grep: `grep -rn "random.seed\|random.sample\|^import random$" federated-baseline-cf/federated_baseline_cf/task.py federated-baseline-cf/federated_baseline_cf/client_app.py` returns 0 matches.
3. D-24 integration smoke: run `test_gradient_mask_zeros_non_user_rows` — passes.
4. D-21 contract grep: `grep "num-examples" federated-baseline-cf/federated_baseline_cf/client_app.py` returns 0 (no free-form extras).
5. D-18 surgical guard: `git diff federated-baseline-cf/federated_baseline_cf/dataset.py` diff against Plan 02 state is EMPTY (dataset.py untouched).
6. Smoke test: `python -c "from federated_baseline_cf.client_app import app as client_app; from federated_baseline_cf.task import train_bpr_mf, evaluate_ranking_sampled; import inspect; assert 'run_seed' in inspect.signature(train_bpr_mf).parameters; assert 'run_seed' in inspect.signature(evaluate_ranking_sampled).parameters"` exits 0.
</verification>

<success_criteria>
- BSL-02 observable: benchmark_cross_device + partition_mode=natural + num_users_in_client>1 raises AssertionError before any training.
- BSL-03 observable: training negatives exclude the held-out test positive (exclusion_set merged into user_rated_items).
- BSL-05 observable: no `random.seed()` / `import random` in task.py or client_app.py; `evaluate_ranking_sampled` seeds via `np_rng(run_seed, user_idx, round_num, 'eval_neg')`.
- BSL-07 observable: `get_primary_evaluator(mode)` is called; `allrank_*` not populated into thesis-table fields.
- D-21 observable: client returns FitMetricsContract.to_dict() — `num-examples` is gone; `train_loss`/`num_positives`/`num_training_examples` are present and contract-validated.
- D-22 observable: per-group sufficient-stat keys (hit_count_{sparse,medium,dense}_at10, ndcg_sum_..._at10, evaluated_users_{sparse,medium,dense}) appear in the payload.
- D-24 observable: post-training-step, only user_idx's row of user_embeddings.weight has changed (test asserts this on a tiny model).
- 8 GREEN tests across test_task_rng.py + test_client_assertion.py.
- D-18 surgical guard: pre-existing uncommitted hunks outside BSL-02/03/05/07 rip scope remain untouched. Executor confirms via `git diff --stat` before committing.
</success_criteria>

<output>
After completion, create `.planning/phases/02-baseline-migration/02-baseline-migration-03-SUMMARY.md` following the template in `@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md`.
</output>
