---
phase: 05-pfedrec-migration-reproduction
plan: 03
type: execute
wave: 2
depends_on: ["05-pfedrec-migration-reproduction-01", "05-pfedrec-migration-reproduction-02"]
files_modified:
  - federated-pfedrec/federated_pfedrec/client_app.py
  - federated-pfedrec/federated_pfedrec/task.py
  - federated-pfedrec/tests/conftest.py
  - federated-pfedrec/tests/test_client_app.py
  - federated-pfedrec/tests/test_task.py
  - federated-pfedrec/tests/test_cache.py
autonomous: true
requirements: [PFR-02, PFR-03, PFR-04, PFR-05, PFR-06, PFR-07]
must_haves:
  truths:
    - "Client-side per-partition single-user collapse: PFR-05 — `for user_idx in user_test_items.keys()` loop replaced by single-user direct call (one user per partition in benchmark mode)"
    - "FND-03 ExclusionTable.for_user(user_idx) is threaded into the training-negative pool — held-out test positive is NEVER drawn as a training negative (PFR-04)"
    - "FND-06 RNG factories (np_rng) are wired per round, per user, per purpose ('train_neg', 'eval_neg') — PFR-07 closes (training negatives re-sampled every round); BSL-05-style cross-file regression: zero `random.seed(`, zero `random.sample(`, zero module-level `import random` in BOTH task.py and client_app.py"
    - "Eval-time BCE loss is computed over positives + 99 negatives (D-04; matches engine.py:195-196)"
    - "D-22 cold-round probe-then-load: `if cache_path.exists(): load(); else: cold_round=True` — D-19 PyTorch Linear default init preserved on cold start"
    - "Manifest-sidecar cache schema_version=3 with 9 fields including bias_classification='global' sentinel (D-17); atomic_write_json + tempfile prefix='partition_tmp_'"
    - "Cache .pt payload contains EXACTLY 1 key (`affine_output.weight`) shape `(1, latent_dim)` per D-20; D-21 strict shape guard fires on BOTH save AND load"
    - "torch.load uses weights_only=True (Pitfall 6 close)"
    - "@app.evaluate discover_only=True short-circuit returns zero-suffstats EvaluateMetricsContract with partition_id only — no model load, no data load (G-03-01)"
  artifacts:
    - path: "federated-pfedrec/federated_pfedrec/client_app.py"
      provides: "Cross-device client with one-user collapse + FND-03 exclusion + FND-06 RNG + manifest-sidecar cache schema_v3 + discover_only short-circuit + D-21 strict load"
      contains: "assert_benchmark_one_user_per_client"
    - path: "federated-pfedrec/federated_pfedrec/task.py"
      provides: "FND-06 RNG factories + FND-03 exclusion in negative pool + D-04 eval BCE over 99 negs"
      contains: "from fedrec_foundation.rng import np_rng"
    - path: "federated-pfedrec/tests/test_cache.py"
      provides: "D-16 partition_{pid}.pt layout + D-17 schema_v3 manifest + D-21 strict-load + bias_classification='global' sentinel"
    - path: "federated-pfedrec/tests/test_task.py"
      provides: "FND-03 exclusion + FND-06 per-round RNG + D-04 eval BCE coverage tests"
    - path: "federated-pfedrec/tests/test_client_app.py"
      provides: "PFR-05 one-user assertion + D-22 cold-round probe + discover_only short-circuit"
  key_links:
    - from: "client_app.py @app.train (single-user path)"
      to: "task.train_pfedrec_single_user(... exclude_items=exclusion_table.for_user(user_idx) ..., rng=np_rng(run_seed, user_idx, round_num, 'train_neg'))"
      via: "ExclusionTable + np_rng wired per round"
      pattern: "exclude_items=.*for_user|np_rng\\(.*train_neg"
    - from: "client_app.py manifest-sidecar"
      to: ".embedding_cache/{run_id}/manifest.json (schema_version=3, bias_classification='global' sentinel)"
      via: "atomic_write_json + 9-field signature"
      pattern: "schema_version.*3|bias_classification.*global"
    - from: "client_app.py @app.evaluate discover_only short-circuit"
      to: "EvaluateMetricsContract(partition_id=partition_id, hit_count_overall_at10=0, ...)"
      via: "context.run_config['discover_only'] check FIRST"
      pattern: "discover_only"
---

<objective>
Client-side full migration: client_app.py one-user collapse + manifest-sidecar cache schema_v3 + task.py FND-06/FND-03/D-04 wire (Wave-2 single-plan).

Purpose:
  - PFR-05: Collapse `for user_idx in user_train_data.keys()` loop in client_app.py to a single-user direct path. In cross-device, partition_id == user_idx; benchmark-mode one-user assertion locks this.
  - PFR-04: Thread FND-03 `ExclusionTable.for_user(user_idx)` into BOTH training-negative pool (in `prepare_user_train_data` + `train_pfedrec_single_user`) AND eval-negative pool (in `evaluate_pfedrec_sampled`). Held-out test positive provably never drawn.
  - PFR-06 (client half) + PFR-07: Replace `task.py:130 rng = random.Random(seed)` with `np_rng(run_seed, user_idx, round_num, "train_neg")` per round per user. Training negatives re-sampled every round (closes CONCERNS bug #5).
  - PFR-02 D-04: Extend eval-time BCE computation to include positives + 99 negatives (matches `engine.py:195-196`).
  - PFR-03 / D-16 / D-17 / D-21 / D-22: Replace per-user-subdir cache with single-file-per-partition manifest-sidecar layout. Schema_version=3 with 9 fields including `bias_classification='global'` sentinel. Cold-round probe-then-load. D-21 strict-load default with rm -rf hint.
  - G-03-01 carry-forward: `@app.evaluate` discover_only=True short-circuit returns zero-suffstats EvaluateMetricsContract with partition_id only.
  - Pitfall 6: switch `torch.load(weights_only=True)`.

Output:
  - 2 modified files: client_app.py + task.py.
  - 4 new test files: conftest.py + test_client_app.py + test_task.py + test_cache.py — total ~14 GREEN tests covering all VALIDATION.md per-task verification rows for Plan 03 (5-03-01 through 5-03-10).
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/PROJECT.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/ROADMAP.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/STATE.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-RESEARCH.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-VALIDATION.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-01-PLAN.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-02-PLAN.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/03-personalized-migration/03-personalized-migration-03-SUMMARY.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/codebase/CONCERNS.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/IJCAI-23-PFedRec/engine.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec/federated_pfedrec/client_app.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec/federated_pfedrec/task.py

<interfaces>
<!-- Foundation contract surfaces this plan consumes -->
```python
# fedrec_foundation.rng (FND-06, Plan 1 Plan 04)
from fedrec_foundation.rng import np_rng, torch_gen, py_rng
# np_rng(run_seed: int, user_idx: int, round_num: int, purpose: str) -> np.random.Generator

# fedrec_foundation.exclusion (FND-03, Plan 1 Plan 02)
from fedrec_foundation.exclusion import ExclusionTable, load_exclusion
# ExclusionTable.for_user(user_idx: int) -> Set[int]   (positives ∪ test held-out)

# fedrec_foundation.fit_metrics (Plan 1 Plan 03 + Phase 2 Plan 01)
from fedrec_foundation.fit_metrics import (
    EvaluateMetricsContract, FitMetricsContract,
    validate_evaluate_metrics, validate_fit_metrics,
)
# EvaluateMetricsContract has: hit_count_overall_at10, ndcg_sum_overall_at10,
# evaluated_users + 9 per-group fields + optional eval_loss/sampled_hr_at10/
# sampled_ndcg_at10 + optional partition_id

# fedrec_foundation.mode
from fedrec_foundation.mode import (
    assert_benchmark_one_user_per_client, log_mode_and_overrides,
    resolve_mode_defaults,
)

# fedrec_foundation.evaluator (Plan 1 Plan 03)
from fedrec_foundation.evaluator import get_primary_evaluator
# get_primary_evaluator(mode: str) -> Literal["sampled_loo_99"]

# fedrec_foundation.atomic
from fedrec_foundation.atomic import atomic_write_json

# fedrec_foundation.evaluation user groups (Plan 1 Plan 02)
from fedrec_foundation.user_groups import classify_user_group
```

```python
# Phase 3 client_app cache helper signatures (clone with schema_v3 9-field signature)
def _signature_fields(*, run_id, method, num_users, num_items, latent_dim,
                      split_hash, loss="bce", num_train_negatives=4,
                      bias_classification="global") -> Dict:
    return {
        "schema_version": 3,
        "run_id": run_id,
        "method": method,                     # "pfedrec"
        "num_users": num_users,
        "num_items": num_items,
        "latent_dim": latent_dim,
        "split_hash": split_hash,
        "loss": loss,                         # "bce"
        "num_train_negatives": num_train_negatives,  # 4 (paper-compat)
        "bias_classification": bias_classification,  # "global" (D-17 sentinel)
    }


def _cache_dir_for_run(*, run_id: str, reuse_cache: bool, signature: Dict) -> Path: ...

def _save_local_user_state(*, partition_id, state_dict, run_id, reuse_cache, signature) -> None:
    """D-21: assert state_dict.keys() == {'affine_output.weight'};
    assert tensor shape == (1, signature['latent_dim']); atomic .pt write
    via tempfile prefix='partition_tmp_' + os.replace; manifest.json sidecar
    written first via atomic_write_json."""
    ...

def _load_local_user_state(*, partition_id, run_id, reuse_cache, signature) -> Optional[Dict]:
    """Returns None on cache miss (cold round per D-22). Raises RuntimeError
    with per-field delta + literal `Run: rm -rf .embedding_cache/{run_id}/`
    hint on signature mismatch (D-17). Asserts shape (1, latent_dim) AFTER
    torch.load (D-21)."""
    ...
```

```python
# IJCAI-23-PFedRec/engine.py:195-196 (D-04 source-of-truth — eval BCE over positives + 99 negs)
test_score = user_model(test_item)
negative_score = user_model(negative_item)
ratings_pred = torch.cat((test_score, negative_score))  # 100 items
loss = self.crit(ratings_pred.view(-1), ratings)         # ratings = [1,0,...,0]
all_loss[user] = loss.item()
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Rewrite task.py with FND-06 RNG factories + FND-03 exclusion-in-train-neg-pool + D-04 eval BCE over positives + 99 negs; ship test_task.py with 4 GREEN tests</name>
  <files>federated-pfedrec/federated_pfedrec/task.py, federated-pfedrec/tests/test_task.py, federated-pfedrec/tests/conftest.py</files>
  <read_first>
    - federated-pfedrec/federated_pfedrec/task.py — current state (line 130 `rng = random.Random(seed)`; line 134-166 prepare_user_train_data; line 195-220 train_pfedrec_single_user; line 432 evaluate_pfedrec_sampled / test_pfedrec)
    - federated-pfedrec/federated_pfedrec/dataset.py — post-Plan-02 placeholder state. Plan 02 leaves `load_partition_data` and `load_full_data` as the D-09 NotImplementedError guard plus a trailing `raise NotImplementedError("Plan 03 implements the foundation-adapter body")` placeholder. Plan 03 fills these bodies (build (trainloader, testloader, num_users, num_items) for the natural cross-device path from `bundle.split_manifest` + `bundle.exclusion`). DO NOT re-introduce the Plan-02 D-09 guard prose here; preserve it verbatim.
    - federated-personalized-cf/federated_personalized_cf/task.py — Phase 3 task.py reference (FND-06 + FND-03 wiring + _sample_negatives_seeded helper pattern)
    - IJCAI-23-PFedRec/engine.py — lines 84-146 (per-user round update + dual-LR), lines 149-212 (fed_evaluate with BCE over 99 negs at 195-196)
    - .planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md §decisions D-02, D-03, D-04, D-19, D-22
    - .planning/phases/05-pfedrec-migration-reproduction/05-RESEARCH.md §Pattern 4 + Pitfall 3 (DO NOT change `lr * num_items * lr_eta = 29,648` — intentional dual-LR)
    - scripts/foundation/fedrec_foundation/rng.py — np_rng signature
    - scripts/foundation/fedrec_foundation/exclusion.py — ExclusionTable.for_user signature
  </read_first>
  <behavior>
    - Test 1 (test_train_negs_exclude_held_out_test_positive): import `_sample_train_negatives_seeded` (or whatever helper task.py exposes); build `user_rated_items={1, 2, 3}`, `exclude_items={5, 6}` (representing held-out test positive = 5); call helper with `rng=np.random.default_rng(42)`, `num_negatives=4`, `num_items=100`; assert returned negatives are disjoint from BOTH `user_rated_items` AND `exclude_items` — `assert {5, 6}.isdisjoint(set(returned_negs))`. PFR-04 / FND-03.
    - Test 2 (test_train_negs_resampled_every_round): call `np_rng(run_seed=42, user_idx=0, round_num=1, "train_neg")` and `np_rng(run_seed=42, user_idx=0, round_num=2, "train_neg")` — assert the two `Generator.choice(num_items, 4)` outputs DIFFER (PFR-07 — different rounds → different negatives). Also assert `np_rng(run_seed=42, user_idx=0, round_num=1, "train_neg")` called twice produces IDENTICAL output (FND-06 determinism).
    - Test 3 (test_eval_neg_rng_factory_used): introspect `inspect.signature(evaluate_pfedrec_sampled)` (or whatever the eval function is named in task.py); assert it accepts kwargs `run_seed`, `user_idx`, `round_num`, `exclude_items`; assert that reading the source via `inspect.getsource(evaluate_pfedrec_sampled)` shows it uses `np_rng(...)` and does NOT contain `random.seed(`, `random.sample(`, or module-level `import random` (BSL-05-style cross-file regression — Phase 3 idiom).
    - Test 4 (test_eval_bce_over_positives_plus_99_negs): build a tiny PFedRecMLP, build a synthetic `(test_item=42, negative_items=[5,6,7,...])` (1 positive + 99 negs); call the eval function; assert it returns an `eval_loss` field whose computation uses 100 items (not 1). The simplest acceptance is to assert `inspect.getsource(...)` of the eval function contains the substring `torch.cat((test_score, negative_score))` OR `torch.cat([test_score, negative_score])` (D-04 source of truth from engine.py:195-196 — alignment to reference).
  </behavior>
  <action>

**Task 1.1 — Modify `federated-pfedrec/federated_pfedrec/task.py` (rip-and-replace stdlib random + thread FND-06 / FND-03 / D-04):**

1. Add module-top imports:
```python
from typing import Optional, Set
import numpy as np

from fedrec_foundation.rng import np_rng, torch_gen
```

2. ELIMINATE (delete or rewrite):
   - Any module-level `import random` line.
   - The `rng = random.Random(seed)` line at line 130 (current).
   - All `random.seed(...)` calls.
   - All `random.sample(...)` calls.

3. Replace `prepare_user_train_data` (currently builds `user_positives` from trainloader only) with:

```python
def prepare_user_train_data(
    user_idx: int,
    user_train_items: List[int],
    *,
    num_items: int,
    num_negatives: int,
    run_seed: int,
    round_num: int,
    exclude_items: Optional[Set[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (item_ids, ratings, user_ids) batch for one user via per-round
    re-sampled FND-06 RNG (PFR-07). Held-out test positive excluded via
    FND-03 ExclusionTable (PFR-04).

    Parameters
    ----------
    user_idx : int
    user_train_items : List[int]
        Items the user has positively interacted with (LOO-train positives).
    num_items : int
    num_negatives : int
        Number of negatives per positive (4 in paper-compat).
    run_seed, round_num : int
        Per-round RNG keys (FND-06).
    exclude_items : Optional[Set[int]]
        Result of `ExclusionTable.for_user(user_idx)`; merged into the
        no-go set so the held-out test positive is never drawn (PFR-04).
    rng : Optional[np.random.Generator]
        Pre-seeded generator from `np_rng(run_seed, user_idx, round_num,
        "train_neg")`. If None, the helper builds one from the (run_seed,
        user_idx, round_num) tuple itself (caller-convenience).
    """
    if rng is None:
        rng = np_rng(run_seed, user_idx, round_num, "train_neg")

    user_rated: Set[int] = set(user_train_items)
    if exclude_items is not None:
        user_rated |= set(exclude_items)

    negatives = _sample_train_negatives_seeded(
        user_rated_items=user_rated,
        num_items=num_items,
        num_negatives=num_negatives * len(user_train_items),
        rng=rng,
    )
    # ... (build the (item_ids, ratings, user_ids) BCE tuple — 1 positive + N negs each)
    # See Phase 3 task.py shape; PFedRec rating is binary (1 / 0).
```

4. Add private module-level helper `_sample_train_negatives_seeded` (Phase 3 idiom, simplified to flat-set rejection-uniform sampler):

```python
def _sample_train_negatives_seeded(
    user_rated_items: Set[int],
    num_items: int,
    num_negatives: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rejection-uniform sampler: returns int32 array of length num_negatives
    with no element in user_rated_items.

    Distribution-equivalent to the reference's `random.sample(...)`-based
    sampler at engine.py-via-data.py:80, but seeded from FND-06 np_rng
    instead of stdlib random.

    Parameters
    ----------
    user_rated_items : Set[int]
        Union of positives + held-out test (FND-03 exclusion threaded by caller).
    num_items : int
    num_negatives : int
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        int32 array of size (num_negatives,).
    """
    out = np.empty(num_negatives, dtype=np.int64)
    filled = 0
    while filled < num_negatives:
        # Sample a batch; reject hits.
        batch = rng.integers(0, num_items, size=2 * (num_negatives - filled))
        for v in batch:
            if int(v) not in user_rated_items:
                out[filled] = int(v)
                filled += 1
                if filled == num_negatives:
                    break
    return out
```

5. Update `train_pfedrec_single_user` signature to accept `run_seed`, `user_idx`, `round_num`, `exclude_items` kwargs and forward them to `prepare_user_train_data`. CRITICAL — DO NOT MODIFY the dual-LR optimizer code (`lr * num_items * lr_eta`) per Pitfall 3. Add a comment near that line:
```python
# DO NOT change — matches reference IJCAI-23-PFedRec/engine.py:117-119
# (effective item LR = lr * num_items * lr_eta).
optimizer_i = torch.optim.SGD(
    model.embedding_item.parameters(),
    lr=lr * num_items * lr_eta,
    weight_decay=l2_regularization,
)
```

6. Update `evaluate_pfedrec_sampled` (or the equivalent eval function) to:
   - Add 4 new kwargs: `run_seed`, `user_idx`, `round_num`, `exclude_items`.
   - Replace `random.seed(seed)` / `random.sample(...)` with `np_rng(run_seed, user_idx, round_num, "eval_neg").choice(...)`.
   - Fold `exclude_items` into the negative-candidate pool BEFORE sampling 99 negs (PFR-04 on eval side too).
   - **D-04: BCE-over-99-negs**: when computing the eval-time loss, use:
     ```python
     test_score = model(torch.tensor([test_item], dtype=torch.long, device=device))
     negative_score = model(torch.tensor(negative_items, dtype=torch.long, device=device))
     ratings_pred = torch.cat((test_score, negative_score))  # 100 items
     ratings_true = torch.zeros(100, dtype=torch.float32, device=device)
     ratings_true[0] = 1.0
     eval_loss = nn.BCELoss()(ratings_pred.view(-1), ratings_true)
     ```
     This matches `engine.py:195-196` exactly.
   - The HR@10 / NDCG@10 computation remains over the same 100-item candidate pool (unchanged).

7. Update `train` / `test` dispatcher (if present in task.py) to forward all 4 new kwargs.

8. Re-word any docstring or comment that previously contained `random.seed(`, `random.sample(`, or `import random` to natural-language prose (Phase 3 precedent — the BSL-05 cross-file regression test is a plain regex, not an AST check).

**Task 1.2 — Create `federated-pfedrec/tests/test_task.py`:**

```python
"""Phase 5 PFR-04 + PFR-06 + PFR-07 + D-04 client task regression guard."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest


def test_train_negs_exclude_held_out_test_positive() -> None:
    """PFR-04 / FND-03: held-out test positive is NEVER drawn as a training negative."""
    from federated_pfedrec.task import _sample_train_negatives_seeded

    user_rated = {1, 2, 3}
    exclude = {5, 6}  # represents held-out test positive(s)
    no_go = user_rated | exclude

    rng = np.random.default_rng(42)
    negs = _sample_train_negatives_seeded(
        user_rated_items=no_go,
        num_items=100,
        num_negatives=20,
        rng=rng,
    )
    neg_set = set(int(x) for x in negs.tolist())
    assert neg_set.isdisjoint(no_go), (
        f"PFR-04 violated: train negs intersect held-out: {neg_set & no_go}"
    )


def test_train_negs_resampled_every_round() -> None:
    """PFR-07 / D-02: per-round RNG produces different negatives across rounds; same key produces same output."""
    from fedrec_foundation.rng import np_rng

    rng_r1_a = np_rng(run_seed=42, user_idx=0, round_num=1, purpose="train_neg")
    rng_r1_b = np_rng(run_seed=42, user_idx=0, round_num=1, purpose="train_neg")
    rng_r2 = np_rng(run_seed=42, user_idx=0, round_num=2, purpose="train_neg")

    out_r1_a = rng_r1_a.integers(0, 1000, size=20).tolist()
    out_r1_b = rng_r1_b.integers(0, 1000, size=20).tolist()
    out_r2 = rng_r2.integers(0, 1000, size=20).tolist()

    assert out_r1_a == out_r1_b, "FND-06 same key must give same output"
    assert out_r1_a != out_r2, "PFR-07: round 1 vs round 2 must differ"


def test_eval_neg_rng_factory_used() -> None:
    """PFR-06 / FND-06: eval function uses np_rng (not stdlib random); BSL-05 cross-file regression."""
    import federated_pfedrec.task as task_mod

    # Read source of the eval function (function name may vary; pick one of the candidates)
    eval_fn = None
    for name in ("evaluate_pfedrec_sampled", "evaluate_ranking_sampled", "evaluate_pfedrec"):
        if hasattr(task_mod, name):
            eval_fn = getattr(task_mod, name)
            break
    assert eval_fn is not None, "task.py must export a sampled eval function"

    sig = inspect.signature(eval_fn)
    assert "run_seed" in sig.parameters
    assert "user_idx" in sig.parameters
    assert "round_num" in sig.parameters
    assert "exclude_items" in sig.parameters

    src_eval = inspect.getsource(eval_fn)
    assert "np_rng" in src_eval, "PFR-06: eval must use FND-06 np_rng factory"
    assert "random.seed(" not in src_eval, "BSL-05: stdlib random.seed must be eradicated"
    assert "random.sample(" not in src_eval, "BSL-05: stdlib random.sample must be eradicated"

    # Cross-file regression: scan task.py and client_app.py
    task_path = Path(task_mod.__file__)
    client_path = task_path.parent / "client_app.py"
    for path in (task_path, client_path):
        text = path.read_text()
        # Module-level `import random` (allow `import random_xyz` etc. by checking lone `import random` lines).
        assert "\nimport random\n" not in text and not text.startswith("import random\n"), (
            f"BSL-05: module-level `import random` must be removed from {path.name}"
        )
        assert "random.seed(" not in text, f"BSL-05: random.seed( must be removed from {path.name}"
        assert "random.sample(" not in text, f"BSL-05: random.sample( must be removed from {path.name}"


def test_eval_bce_over_positives_plus_99_negs() -> None:
    """D-04: eval BCE loss is computed over (positive + 99 negatives) — matches engine.py:195-196."""
    import federated_pfedrec.task as task_mod

    eval_fn = None
    for name in ("evaluate_pfedrec_sampled", "evaluate_ranking_sampled", "evaluate_pfedrec"):
        if hasattr(task_mod, name):
            eval_fn = getattr(task_mod, name)
            break
    assert eval_fn is not None

    src = inspect.getsource(eval_fn)
    # Reference engine.py:195-196 idiom — torch.cat over (test_score, negative_score)
    assert (
        "torch.cat((test_score, negative_score))" in src
        or "torch.cat([test_score, negative_score])" in src
        or ("ratings_pred" in src and "torch.cat" in src and "negative_score" in src)
    ), "D-04: eval-time BCE must concatenate positive + 99 negs (engine.py:195-196)"
```

**Task 1.3 — Create `federated-pfedrec/tests/conftest.py`** (shared fixtures for Wave-2 tests):

```python
"""Phase 5 test conftest: foundation bundle skip + tmp_path-redirected cache root."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLE_PATH = _REPO_ROOT / "data" / "derived" / "foundation_index.json"


def pytest_collection_modifyitems(config, items):
    """Skip tests requiring the foundation bundle if it's not committed (e.g. minimal clones)."""
    if _BUNDLE_PATH.exists():
        return
    skip_marker = pytest.mark.skip(reason="foundation bundle not committed (skip Phase-5 integration tests)")
    for item in items:
        if "foundation_bundle_required" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture
def run_seed() -> int:
    return 42
```

Verify: `cd federated-pfedrec && pytest tests/test_task.py -x -v` → 4 GREEN.
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec && pytest tests/test_task.py -x -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "from fedrec_foundation.rng import" federated-pfedrec/federated_pfedrec/task.py` returns 1
    - `grep -c "np_rng(" federated-pfedrec/federated_pfedrec/task.py` returns at least 2 (train_neg + eval_neg purposes)
    - `grep -cE "random\\.seed\\(|random\\.sample\\(|^import random$" federated-pfedrec/federated_pfedrec/task.py` returns 0
    - `grep -c "exclude_items" federated-pfedrec/federated_pfedrec/task.py` returns at least 3 (param in train + eval + helper, threaded through)
    - `grep -c "_sample_train_negatives_seeded" federated-pfedrec/federated_pfedrec/task.py` returns at least 2 (def + call)
    - `grep -cE "torch.cat\\(\\(test_score, negative_score\\)\\)|torch.cat\\(\\[test_score, negative_score\\]\\)" federated-pfedrec/federated_pfedrec/task.py` returns at least 1 (D-04 — engine.py:195-196 idiom)
    - `grep -c "lr \\* num_items \\* lr_eta" federated-pfedrec/federated_pfedrec/task.py` returns at least 1 (Pitfall 3 — paper-faithful dual LR preserved)
    - `pytest federated-pfedrec/tests/test_task.py -x -v` exits 0 with 4 tests passed
  </acceptance_criteria>
  <done>
    - task.py: FND-06 RNG factories wired (D-02 / PFR-07); FND-03 exclusion threaded into train + eval negative pools (PFR-04); D-04 eval BCE over positives + 99 negs (matches engine.py:195-196); stdlib random eradicated; dual LR preserved (Pitfall 3)
    - 4 GREEN tests covering PFR-04, PFR-07, PFR-06, D-04
    - conftest.py shared fixture infrastructure ready for Tasks 2 + 3
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Rewrite client_app.py with one-user collapse + manifest-sidecar cache schema_v3 + cold-round probe + discover_only short-circuit; ship test_cache.py + test_client_app.py with 9 GREEN tests</name>
  <files>federated-pfedrec/federated_pfedrec/client_app.py, federated-pfedrec/tests/test_cache.py, federated-pfedrec/tests/test_client_app.py</files>
  <read_first>
    - federated-pfedrec/federated_pfedrec/client_app.py — current state (per-user-subdir cache; loop over user_test_items.keys())
    - federated-personalized-cf/federated_personalized_cf/client_app.py — Phase 3 manifest-sidecar cache reference (D-04..D-10 idiom; clone with schema_v3 9-field signature)
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py — Phase 4 schema_v2 idiom (clone with schema_v3 + bias_classification)
    - .planning/phases/03-personalized-migration/03-personalized-migration-03-SUMMARY.md — Phase 3 client_app cache + discover_only + Rule-1 fix (tempfile prefix='partition_tmp_')
    - .planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md §decisions D-13, D-16, D-17, D-19, D-20, D-21, D-22
    - .planning/phases/05-pfedrec-migration-reproduction/05-RESEARCH.md §Pattern 3 + Pattern 4 + Pitfall 6
    - federated-pfedrec/federated_pfedrec/task.py (post-Task 1) — for the new train/eval signatures
    - scripts/foundation/fedrec_foundation/atomic.py — atomic_write_json signature
    - scripts/foundation/fedrec_foundation/mode.py — assert_benchmark_one_user_per_client signature
    - scripts/foundation/fedrec_foundation/fit_metrics.py — EvaluateMetricsContract / FitMetricsContract field shapes
  </read_first>
  <behavior>
    - Test 1 (test_partition_pid_pt_layout): in tmp_path-redirected `_CACHE_BASE_DIR`, call `_save_local_user_state(partition_id=42, state_dict={'affine_output.weight': torch.randn(1, 32)}, run_id='r1', reuse_cache=False, signature=signature(latent_dim=32))`; assert file exists at `<tmp>/r1/partition_42.pt` (D-16 layout — single file per partition, no `user_<uid>` subdir).
    - Test 2 (test_manifest_schema_v3_fields): after the same save, assert `<tmp>/r1/manifest.json` exists; load JSON; assert it has all 10 expected keys: `schema_version=3`, `run_id`, `method='pfedrec'`, `num_users`, `num_items`, `latent_dim`, `split_hash`, `loss='bce'`, `num_train_negatives`, `bias_classification`.
    - Test 3 (test_bias_classification_sentinel_global): assert `manifest['bias_classification'] == 'global'` (D-17 sentinel — catches future regression that reverts D-01).
    - Test 4 (test_strict_load_shape_mismatch_raises): seed cache with `latent_dim=32`; build new signature with `latent_dim=64`; call `_load_local_user_state(partition_id=42, run_id='r1', reuse_cache=False, signature=new_sig)`; assert `RuntimeError` whose message contains `'latent_dim'`, `'rm -rf'`, AND the `r1` run_id path.
    - Test 5 (test_reuse_cache_sig_path): call `_cache_dir_for_run(reuse_cache=True, ...)` twice with different `run_id` but otherwise-identical signature; assert paths COLLIDE on `sig_<16-hex-chars>` directory (D-18 reuse pattern).
    - Test 6 (test_save_payload_shape_guard): hand state_dict with extra key `extra_param` (`{'affine_output.weight': torch.randn(1, 32), 'extra_param': torch.zeros(5)}`); assert `_save_local_user_state` raises `AssertionError` with `'D-21'` in message BEFORE any disk write (`<tmp>` remains empty).
    - Test 7 (test_benchmark_one_user_per_client_assert): build a synthetic context where partition contains 1 user — assertion passes. Build another where partition contains 5 users (and `num-supernodes` is NOT in overrides) — `assertion fails with AssertionError`. PFR-05.
    - Test 8 (test_cold_round_probe_then_load): setup: empty `tmp_path/r1/` cache dir. Call `_load_local_user_state(partition_id=0, run_id='r1', reuse_cache=False, signature=signature)` — assert returns `None` (cache miss, cold round per D-22). Then save state, call again — assert returns the state dict.
    - Test 9 (test_discover_only_short_circuit): build a Message with `config={'discover_only': True}` and `partition_id=42` from context.node_config; invoke `@app.evaluate` handler — assert it returns an EvaluateMetricsContract dict with `partition_id=42`, `hit_count_overall_at10=0`, `evaluated_users=0`; assert NO model load and NO bundle load occurred (test using a monkeypatched `_load_foundation_bundle` that records calls — should record 0 calls when discover_only).
    - Test 10 (test_torch_load_weights_only_true): grep `_load_local_user_state` source for `weights_only=True`; assert at least 1 occurrence (Pitfall 6 — PyTorch 2.6+ safe default).
  </behavior>
  <action>

NOTE — SC-2 RECONCILIATION (the cache-layout task surfaces this explicitly):

ROADMAP §Phase 5 SC-2 phrase *"each user's `(affine_output.weight, affine_output.bias)` is persisted/restored as one atomic per-user artifact keyed by stable `user_idx`"* is reconciled with CONTEXT.md D-01 as: weight is the per-user disk cache payload (single key `affine_output.weight` shape `(1, latent_dim)` written atomically to `partition_{pid}.pt`); bias is the per-user state aggregated atomically server-side per `IJCAI-23-PFedRec/engine.py:143` (server pulls every user's `affine_output.bias` into `aggregate_clients_params` once per round and broadcasts the aggregated value back). The atomicity contract is preserved (per-round, per-user) but the bias channel moves from disk to server-side aggregation per the IJCAI-23 reference. The gsd-verifier MUST accept this reconciliation when evaluating SC-2; PFR-02-AUDIT.md (Plan 01 Task 3) carries the human-readable cross-walk and the explicit closure note.

Rip-and-replace `federated-pfedrec/federated_pfedrec/client_app.py` cloning Phase 3 client_app.py shape with Phase-5-specific deltas:

**Top-of-file imports:**
```python
import json
import os
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from flwr.app import (
    ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict,
)
from flwr.clientapp import ClientApp

from fedrec_foundation.atomic import atomic_write_json
from fedrec_foundation.evaluator import get_primary_evaluator
from fedrec_foundation.fit_metrics import (
    EvaluateMetricsContract, FitMetricsContract,
    validate_evaluate_metrics, validate_fit_metrics,
)
from fedrec_foundation.mode import (
    assert_benchmark_one_user_per_client, log_mode_and_overrides, resolve_mode_defaults,
)
from fedrec_foundation.rng import np_rng

from federated_pfedrec.dataset import _load_foundation_bundle, load_partition_data
from federated_pfedrec.models.pfedrec_mlp import PFedRecMLP
from federated_pfedrec.task import (
    train_pfedrec_single_user, evaluate_pfedrec_sampled,
)

app = ClientApp()

# Module-level constants (test-time monkeypatchable)
_MODULE_DIR = Path(__file__).parent
_CACHE_BASE_DIR = _MODULE_DIR.parent / ".embedding_cache"
_device_cache: Optional[torch.device] = None
```

**Module-level cache helpers (clone Phase 3 with schema_v3 9-field signature):**

```python
def _signature_fields(
    *,
    run_id: str,
    method: str = "pfedrec",
    num_users: int,
    num_items: int,
    latent_dim: int,
    split_hash: str,
    loss: str = "bce",
    num_train_negatives: int = 4,
    bias_classification: str = "global",
) -> Dict:
    """D-17 schema_v3 signature for PFedRec cache. 10 fields including
    schema_version and the bias_classification='global' D-01 sentinel.

    SC-2/D-01 reconciliation note: per-user disk payload carries only
    `affine_output.weight`; the SC-2 phrase about (.weight, .bias)
    per-user atomicity is preserved end-to-end because the bias channel
    is aggregated atomically server-side per engine.py:143. See
    PFR-02-AUDIT.md (Plan 01 Task 3) for the human-readable cross-walk.
    """
    return {
        "schema_version": 3,
        "run_id": run_id,
        "method": method,
        "num_users": num_users,
        "num_items": num_items,
        "latent_dim": latent_dim,
        "split_hash": split_hash,
        "loss": loss,
        "num_train_negatives": num_train_negatives,
        "bias_classification": bias_classification,
    }


def _cache_dir_for_run(*, run_id: str, reuse_cache: bool, signature: Dict) -> Path:
    """D-18: under reuse_cache=True, dir is `sig_<sha256[:16]>` (run_id-agnostic)."""
    if reuse_cache:
        sig_keys = sorted(set(signature.keys()) - {"run_id", "schema_version"})
        sig_str = json.dumps({k: signature[k] for k in sig_keys}, sort_keys=True)
        sig_hash = hashlib.sha256(sig_str.encode()).hexdigest()[:16]
        return _CACHE_BASE_DIR / f"sig_{sig_hash}"
    return _CACHE_BASE_DIR / run_id


def _save_local_user_state(
    *,
    partition_id: int,
    state_dict: Dict[str, torch.Tensor],
    run_id: str,
    reuse_cache: bool,
    signature: Dict,
) -> None:
    """Atomic per-user save with D-21 shape guard BEFORE disk write.

    Payload shape: exactly 1 key (`affine_output.weight`) with shape
    `(1, signature['latent_dim'])` (D-20 native PyTorch shape). The
    SC-2/D-01 reconciliation: bias is aggregated atomically server-side
    (engine.py:143), not persisted to per-user disk; D-15 result-JSON +
    sibling manifest carry the atomicity end-to-end across runs.
    """
    cache_dir = _cache_dir_for_run(run_id=run_id, reuse_cache=reuse_cache, signature=signature)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # D-21 shape guard BEFORE disk write
    assert set(state_dict.keys()) == {"affine_output.weight"}, (
        f"D-21 expected single-key payload {{'affine_output.weight'}}, "
        f"got {set(state_dict.keys())}"
    )
    expected_shape = (1, signature["latent_dim"])
    actual_shape = tuple(state_dict["affine_output.weight"].shape)
    assert actual_shape == expected_shape, (
        f"D-21 expected shape {expected_shape}, got {actual_shape}"
    )

    # Manifest sidecar first (D-17)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        atomic_write_json(str(manifest_path), signature)

    # Atomic .pt write (Phase 3 Rule 1: tempfile prefix MUST NOT start with `.`)
    pt_path = cache_dir / f"partition_{partition_id}.pt"
    fd, tmp = tempfile.mkstemp(dir=str(cache_dir), prefix="partition_tmp_", suffix=".pt")
    os.close(fd)
    try:
        torch.save(state_dict, tmp)
        os.replace(tmp, str(pt_path))
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _load_local_user_state(
    *,
    partition_id: int,
    run_id: str,
    reuse_cache: bool,
    signature: Dict,
) -> Optional[Dict[str, torch.Tensor]]:
    """D-22 probe-then-load. Returns None on cache miss (cold round).
    Raises RuntimeError on signature mismatch (D-17) or shape mismatch (D-21).
    Pitfall 6: torch.load uses weights_only=True.
    """
    cache_dir = _cache_dir_for_run(run_id=run_id, reuse_cache=reuse_cache, signature=signature)
    pt_path = cache_dir / f"partition_{partition_id}.pt"
    manifest_path = cache_dir / "manifest.json"

    if not pt_path.exists():
        return None  # D-22 cold round

    # D-17 manifest signature check
    with open(manifest_path) as f:
        on_disk = json.load(f)
    diffs = [
        (k, on_disk.get(k), signature[k])
        for k in signature
        if on_disk.get(k) != signature[k]
    ]
    if diffs:
        msg = "; ".join(
            f"{k}: on-disk={od!r} vs live={live!r}" for k, od, live in diffs
        )
        raise RuntimeError(
            f"D-17 manifest mismatch for {pt_path}: {msg}. "
            f"Run: rm -rf {cache_dir}/"
        )

    state = torch.load(pt_path, weights_only=True, map_location="cpu")

    # D-21 shape guard AFTER load
    assert set(state.keys()) == {"affine_output.weight"}, (
        f"D-21 expected single-key payload, got {set(state.keys())}"
    )
    expected_shape = (1, signature["latent_dim"])
    actual_shape = tuple(state["affine_output.weight"].shape)
    assert actual_shape == expected_shape, (
        f"D-21 expected shape {expected_shape}, got {actual_shape}"
    )
    return state


def _get_device() -> torch.device:
    """CUDA fallback (preserve from D-18 surgical scope)."""
    global _device_cache
    if _device_cache is not None:
        return _device_cache
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1).cuda()
            _device_cache = torch.device("cuda")
        except RuntimeError:
            _device_cache = torch.device("cpu")
    else:
        _device_cache = torch.device("cpu")
    return _device_cache
```

**`@app.train` body — single-user collapse path (PFR-05):**

```python
@app.train()
def train(message: Message, context: Context) -> Message:
    """Cross-device PFedRec @app.train: single-user path.

    PFR-05: client partition contains exactly one user (benchmark mode).
    PFR-04: training negatives exclude held-out test positive (FND-03).
    PFR-07: training negatives re-sampled per round via FND-06 RNG.
    """
    # 1. Mode resolve
    mode = str(context.run_config.get("mode", "paper_compat_pfedrec"))
    profile = resolve_mode_defaults(mode)
    overrides = log_mode_and_overrides(mode, profile, context.run_config)

    # 2. Per-client identity
    partition_id = int(context.node_config["partition-id"])
    user_idx = partition_id  # cross-device: 1 user = 1 partition
    round_num = int(message.content["config"]["round_num"])
    run_seed = int(context.run_config.get("run-seed", profile_default_or(profile, "run_seed", 42)))
    run_id = str(message.content["config"]["run_id"])
    reuse_cache = bool(message.content["config"].get("reuse_cache", False))

    # 3. Load foundation bundle + build signature
    bundle = _load_foundation_bundle()
    latent_dim = int(context.run_config.get("latent-dim", profile.embedding_dim))
    num_train_negatives = int(context.run_config.get("num-negatives", profile.num_train_negatives))
    signature = _signature_fields(
        run_id=run_id,
        num_users=bundle.mapping.num_users,
        num_items=bundle.mapping.num_items,
        latent_dim=latent_dim,
        split_hash=bundle.split_hash,
        num_train_negatives=num_train_negatives,
    )

    # 4. Build model + load global params
    device = _get_device()
    model = PFedRecMLP(num_items=bundle.mapping.num_items, latent_dim=latent_dim).to(device)
    global_state = {
        k: torch.from_numpy(v) for k, v in message.content["arrays"].numpy_state_dict().items()
    }
    model.set_global_parameters(global_state)  # adds affine_output.bias under D-01

    # 5. D-22 probe-then-load (cold round preserves Kaiming default — D-19)
    cache_path = _cache_dir_for_run(run_id=run_id, reuse_cache=reuse_cache, signature=signature) \
        / f"partition_{partition_id}.pt"
    cold_round = not cache_path.exists()
    if not cold_round:
        local_state = _load_local_user_state(
            partition_id=partition_id, run_id=run_id, reuse_cache=reuse_cache, signature=signature,
        )
        if local_state is not None:
            model.set_local_parameters(local_state, strict=True, run_id=run_id)  # D-21

    # 6. Load partition data + one-user assertion (PFR-05)
    user_train_data, user_test_items, num_users, num_items = load_partition_data(
        partition_id=partition_id, num_partitions=profile.num_supernodes,
        partition_mode="natural", batch_size=int(context.run_config.get("batch-size", 256)),
        num_negatives=num_train_negatives,
    )
    assert_benchmark_one_user_per_client(profile, len(user_train_data), overrides)

    # 7. Single-user collapse: NO loop over user_train_data.keys()
    user_train_items, user_train_ratings = next(iter(user_train_data.items()))[1]
    exclude_items = bundle.exclusion.for_user(user_idx)

    # 8. Train with FND-06 RNG + FND-03 exclusion (PFR-04 + PFR-07)
    train_pfedrec_single_user(
        model=model,
        user_idx=user_idx,
        user_train_items=user_train_items,
        user_train_ratings=user_train_ratings,
        num_items=num_items,
        local_epochs=int(context.run_config.get("local-epochs", profile.local_epochs)),
        batch_size=int(context.run_config.get("batch-size", 256)),
        lr=float(context.run_config.get("lr", profile.lr)),
        lr_eta=int(context.run_config.get("lr-eta", 80)),
        num_train_negatives=num_train_negatives,
        l2_regularization=float(context.run_config.get("l2-regularization", 0.0)),
        device=device,
        run_seed=run_seed,
        round_num=round_num,
        exclude_items=exclude_items,
    )

    # 9. Persist single-user single-file (D-16)
    local_payload = {
        "affine_output.weight": model.affine_output.weight.data.cpu().clone()
    }
    _save_local_user_state(
        partition_id=partition_id, state_dict=local_payload,
        run_id=run_id, reuse_cache=reuse_cache, signature=signature,
    )

    # 10. Build wire payload — return GLOBAL params (item embeddings + affine_output.bias)
    global_params = model.get_global_parameters()
    arrays_out = ArrayRecord({k: v.numpy() for k, v in global_params.items()})

    fit_metrics = FitMetricsContract(
        num_positives=len(user_train_items),
        num_training_examples=len(user_train_items) * (1 + num_train_negatives),
        partition_id=partition_id,
        round_num=round_num,
    )
    metrics_dict = fit_metrics.to_dict()
    validate_fit_metrics(metrics_dict)

    return Message(
        content=RecordDict({"arrays": arrays_out, "metrics": MetricRecord(metrics_dict)}),
        reply_to=message,
    )
```

**`@app.evaluate` body — discover_only short-circuit FIRST + single-user eval path:**

```python
@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    """Cross-device PFedRec @app.evaluate with G-03-01 discover_only short-circuit."""
    # G-03-01: discover_only=True short-circuits BEFORE any heavy work
    config = message.content.get("config", {})
    if config.get("discover_only", False):
        partition_id = int(context.node_config["partition-id"])
        contract = EvaluateMetricsContract(
            hit_count_overall_at10=0,
            ndcg_sum_overall_at10=0.0,
            evaluated_users=0,
            partition_id=partition_id,
        )
        metrics_dict = contract.to_dict()
        validate_evaluate_metrics(metrics_dict)
        return Message(
            content=RecordDict({"metrics": MetricRecord(metrics_dict)}),
            reply_to=message,
        )

    # Normal evaluate path (mode resolve + bundle + signature + model load + eval)
    mode = str(context.run_config.get("mode", "paper_compat_pfedrec"))
    profile = resolve_mode_defaults(mode)
    overrides = log_mode_and_overrides(mode, profile, context.run_config)
    primary = get_primary_evaluator(mode)
    assert primary == "sampled_loo_99", f"PFR-06: expected sampled_loo_99, got {primary!r}"

    # ... (same as @app.train for partition_id, run_seed, round_num, bundle load, signature)
    # ... assert_benchmark_one_user_per_client(profile, len(user_test_items), overrides)
    # ... call evaluate_pfedrec_sampled(model, user_idx=partition_id, run_seed=run_seed,
    #     round_num=round_num, exclude_items=bundle.exclusion.for_user(partition_id))
    # ... build EvaluateMetricsContract with all sufficient stats + partition_id + per-group routing
    # ... validate + return
```

(The full @app.evaluate body mirrors Phase 3 client_app.py @app.evaluate. Per-group routing via `_classify_partition_user_group(bundle, partition_id)` reading `bundle.split_manifest.train_user_stats[pid].user_group` follows Phase 3 exactly.)

**Module helper** `_classify_partition_user_group`: clone Phase 3 verbatim (reads `split_manifest.train_user_stats[pid].user_group`; falls back to `classify_user_group(0)` on elided users).

**Test files — `federated-pfedrec/tests/test_cache.py` (5 tests covering D-16/D-17/D-21/D-22 + bias_classification):**

(Sketch — executor writes the full body following Test 1-6 + Test 8 + Test 10 from the behavior block above.)

**Test file — `federated-pfedrec/tests/test_client_app.py` (3 tests covering PFR-05 + discover_only short-circuit + cold-round probe):**

(Sketch — executor writes the full body following Test 7 + Test 8 + Test 9 from the behavior block above. Use `monkeypatch.setattr(client_app, "_CACHE_BASE_DIR", tmp_path)` to redirect the cache root; use mock `Context` and `Message` builders.)

Verify: `cd federated-pfedrec && pytest tests/test_cache.py tests/test_client_app.py tests/test_task.py -x -v` → all GREEN (≥13 tests). Full module suite: `cd federated-pfedrec && pytest tests/ -x` → all GREEN (15+ tests including Plan 01/02 tests).
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec && pytest tests/test_cache.py tests/test_client_app.py -x -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "from fedrec_foundation.rng import np_rng" federated-pfedrec/federated_pfedrec/client_app.py` returns at least 1
    - `grep -c "from fedrec_foundation.atomic import atomic_write_json" federated-pfedrec/federated_pfedrec/client_app.py` returns at least 1
    - `grep -c "from fedrec_foundation.mode import" federated-pfedrec/federated_pfedrec/client_app.py` returns at least 1
    - `grep -c "from fedrec_foundation.fit_metrics import" federated-pfedrec/federated_pfedrec/client_app.py` returns at least 1
    - `grep -c "assert_benchmark_one_user_per_client" federated-pfedrec/federated_pfedrec/client_app.py` returns at least 1 (PFR-05)
    - `grep -c "discover_only" federated-pfedrec/federated_pfedrec/client_app.py` returns at least 1 (G-03-01)
    - `grep -c "for user_idx in user_train_data" federated-pfedrec/federated_pfedrec/client_app.py` returns 0 (PFR-05 — loop collapsed)
    - `grep -c "schema_version.*3\\|schema_version=3\\|\"schema_version\": 3" federated-pfedrec/federated_pfedrec/client_app.py` returns at least 1 (D-17)
    - `grep -c "bias_classification" federated-pfedrec/federated_pfedrec/client_app.py` returns at least 2 (sentinel field declaration)
    - `grep -c "weights_only=True" federated-pfedrec/federated_pfedrec/client_app.py` returns at least 1 (Pitfall 6)
    - `grep -c "partition_tmp_" federated-pfedrec/federated_pfedrec/client_app.py` returns at least 1 (Phase 3 Rule 1 — tempfile prefix)
    - `grep -cE "random\\.seed\\(|random\\.sample\\(|^import random$" federated-pfedrec/federated_pfedrec/client_app.py` returns 0
    - `grep -c "EvaluateMetricsContract" federated-pfedrec/federated_pfedrec/client_app.py` returns at least 2 (discover_only short-circuit + normal eval)
    - `grep -c "validate_evaluate_metrics\\|validate_fit_metrics" federated-pfedrec/federated_pfedrec/client_app.py` returns at least 2 (D-21 strict-contract validation on both sides)
    - `pytest federated-pfedrec/tests/test_cache.py federated-pfedrec/tests/test_client_app.py -x -v` exits 0 with ≥9 tests passed
    - `pytest federated-pfedrec/tests/ -x` exits 0 with all Plan-01/02/03 tests passed (≥15 cumulative)
  </acceptance_criteria>
  <done>
    - client_app.py: PFR-05 single-user collapse (no loop over user_train_data.keys()); FND-03 ExclusionTable threaded; FND-06 RNG via task.py wiring; D-22 cold-round probe; D-21 strict-load with rm -rf hint; D-16 single-file-per-partition layout; D-17 schema_v3 manifest with bias_classification='global' sentinel; G-03-01 discover_only short-circuit; Pitfall 6 closed (weights_only=True)
    - 5 cache tests + 3 client_app tests + Pitfall 6 test = 9 GREEN tests across test_cache.py + test_client_app.py
    - Full module suite passes (Plan 01 + Plan 02 + Plan 03 = ≥17 GREEN)
  </done>
</task>

</tasks>

<verification>
- Module test suite: `cd federated-pfedrec && pytest tests/ -x -v` → all GREEN (≥17 tests across test_strategy / test_pfedrec_mlp / test_pyproject / test_dataset / test_task / test_cache / test_client_app)
- BSL-05 cross-file regression: `grep -rnE "random\.seed\(|random\.sample\(|^import random$" federated-pfedrec/federated_pfedrec/{task,client_app}.py` → 0 matches
- D-04 trace: `grep -n "torch.cat" federated-pfedrec/federated_pfedrec/task.py` → at least 1 hit at the eval-time concatenation site
- VALIDATION.md per-task verification map for Plan 03 (5-03-01 through 5-03-10) — all 10 rows have a corresponding test method
- D-18 surgical: `git diff --name-only` shows ONLY the 6 files in `files_modified`
</verification>

<success_criteria>
- task.py: FND-06 RNG factories wired (D-02 / PFR-07); FND-03 exclusion threaded (PFR-04); D-04 eval BCE over positives + 99 negs; dual LR preserved; stdlib random eradicated
- client_app.py: PFR-05 single-user collapse; D-22 probe-then-load; D-21 strict; D-16 / D-17 manifest-sidecar with bias_classification='global' sentinel; G-03-01 discover_only short-circuit; weights_only=True; Phase 3 Rule 1 prefix='partition_tmp_'; explicit SC-2/D-01 reconciliation note in the cache-layout signature_fields docstring
- 4 new test files (conftest.py + test_task.py + test_cache.py + test_client_app.py)
- ≥13 new GREEN tests in this plan; cumulative module suite ≥17 GREEN with Plans 01/02
- Wave-2 single-plan ownership held: pyproject.toml, dataset.py (other than fills of the Plan-02 NotImplementedError placeholders), models/, strategy.py, server_app.py UNTOUCHED (those are owned by Plans 01/02/04)
</success_criteria>

<output>
After completion, create `.planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-03-SUMMARY.md` covering:
- PFR-04 (FND-03 exclusion in train + eval neg pools), PFR-05 (single-user collapse), PFR-06 (FND-06 RNG client half), PFR-07 (per-round resampling), D-04 (eval BCE over 99 negs), D-22 (probe-then-load), D-17 (schema_v3 + bias_classification), Pitfall 6 (weights_only=True)
- Test counts mapped to VALIDATION.md per-task rows 5-03-01..5-03-10
- Confirmation Wave-2 single-plan ownership held
- Plan 04 readiness: server_app.py can now consume the EvaluateMetricsContract surface + discover_only short-circuit + the .embedding_cache/{run_id}/partition_{pid}.pt cache path
- SC-2/D-01 reconciliation surfaced in the signature_fields docstring + the cache-layout task `<action>` block; PFR-02-AUDIT.md (Plan 01) carries the human-readable cross-walk
</output>
</output>
