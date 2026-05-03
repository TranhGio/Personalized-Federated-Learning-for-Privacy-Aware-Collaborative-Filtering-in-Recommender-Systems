---
phase: 04-adaptive-migration-bug-fixes
plan: 03
type: execute
subsystem: infra
tags: [client-app, task, rng-threading, exclusion-set, benchmark-assertion, enable-before-load, fit-metrics-contract, evaluate-metrics-contract, per-group-metrics, embedding-cache-manifest-v2, schema-version-2, cold-start-branch, alpha-zero-override, contrastive-skip, d-24-gradient-isolation-ghost-table, adp-02, adp-04, adp-05, adp-06, d-03, d-04, d-09, d-13, d-14, d-15, d-16, wave-2]
wave: 2
depends_on: [04-adaptive-migration-bug-fixes-01, 04-adaptive-migration-bug-fixes-02]
files_modified:
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py
  - federated-adaptive-personalized-cf/tests/test_client_assertion.py
  - federated-adaptive-personalized-cf/tests/test_task_rng.py
  - federated-adaptive-personalized-cf/tests/test_embedding_cache_manifest_v2.py
autonomous: true
requirements: [ADP-02, ADP-04, ADP-05, ADP-06]

must_haves:
  truths:
    - "client_app.py @app.train and @app.evaluate handlers resolve a ModeProfile via fedrec_foundation.mode.resolve_mode_defaults(mode), collect overrides via log_mode_and_overrides, and call assert_benchmark_one_user_per_client BEFORE any training or ranking happens. Under mode=\"benchmark_cross_device\" a partition with >1 user raises AssertionError (ADP-04)."
    - "ADP-02 ordering fix: under benchmark_cross_device mode, enable_per_user_alpha(num_users, init_alphas) AND enable_item_perturbation(reg_lambda) are called UNCONDITIONALLY BEFORE _load_local_user_state (D-03). Run-config flags `enable-per-user-alpha=false` / `enable-item-perturbation=false` are accepted as ablation-only overrides. In ablation mode, the original conditional branch applies."
    - "Both handlers build wire payloads via FitMetricsContract.to_dict() / EvaluateMetricsContract.to_dict() and validate before send (D-21 strict contract). Optional partition_id field is populated on both contracts (G-03-01 carry-forward from Phase 2 Plan 05). FitMetricsContract includes per_user_alpha_* diagnostic fields (D-16) — alpha_mean / alpha_std / alpha_p25 / alpha_p50 / alpha_p75 / alpha_clip_hit_rate — computed client-side over the per-user alpha sigmoid outputs."
    - "client_app.py @app.evaluate short-circuits on discover_only=True ConfigRecord input: returns minimal zero-suffstats EvaluateMetricsContract + partition_id without any model load, data load, exclusion load, or evaluation (G-03-01 discovery-round handshake; carry-forward from Phase 3 Plan 03)."
    - ".embedding_cache path layout implements Phase-4 schema_version=2 (CONTEXT D-02 + D-04): (a) default path is .embedding_cache/{run_id}/partition_{pid}.pt with sibling .embedding_cache/{run_id}/manifest.json; (b) when run_config[\"reuse-cache\"]=true (D-09), path becomes .embedding_cache/sig_{sha256(signature_fields)[:16]}/partition_{pid}.pt with sibling manifest.json; (c) manifest.json is written atomically via fedrec_foundation.atomic.atomic_write_json; (d) schema_version=2 with 12 signature fields (6 Phase-3 + 6 Phase-4: alpha_method, fusion_type, mlp_hidden_dims, per_user_alpha_enabled, item_perturbation_enabled, contrastive_lambda)."
    - "D-04 loud mismatch behavior: on cache load, manifest.json is read first; if ANY signature field diverges from the current run's values (including schema_version 1→2 mismatch from a Phase-3 cache), raise RuntimeError with per-field delta + literal `rm -rf .embedding_cache/{run_id}/` hint. No auto-migration. No silent cold-start."
    - "Atomic single-file cache per partition (CONTEXT D-01): .pt contains an OrderedDict with ALL LOCAL keys in one blob — `user_embeddings.weight`, `user_bias.weight`, `personal_mlp.*` (every sublayer per mlp-hidden-dims), fusion params per fusion-type, `_logit_alpha.weight` (when enabled), `_item_perturbation.weight` (when enabled). Shape verified at save+load."
    - "task.py: FND-06 RNG wired end-to-end — train dispatcher / train_dual_personalized / train_bpr_mf / train_basic_mf / evaluate_ranking_sampled accept run_seed, user_idx, round_num, exclude_items, rng kwargs; zero `random.seed(`, zero `random.sample(`, zero module-level `import random` inside function bodies in task.py AND client_app.py (module-level `import random` at the top of task.py is permissible IF no downstream code executes `random.seed` or `random.sample` — cleanest path is to remove it)."
    - "ADP-05 / FND-03 exclusion: ExclusionTable.for_user(partition_id) is merged into user_rated_items before train-negative sampling AND into negative_candidates before eval-negative sampling; the held-out test positive is never drawn as either a training or eval negative."
    - "D-24 gradient isolation RETAINED (not dropped like Phase 3): the adaptive module keeps DualPersonalizedBPRMF.user_embeddings = nn.Embedding(num_users, d) ghost table (per RESEARCH open question 1; PSN-06 collapse is Phase 4.5 follow-up). task.py therefore still brackets optimizer.step() with _snapshot_non_user_rows / _restore_non_user_rows for user_embeddings AND user_bias AND _logit_alpha (all user-indexed; research open question 4 answer). _item_perturbation is item-indexed — no D-24 needed."
    - "D-13 cold-round behavior: when is_cold_round=True (no cache file yet for this partition), train_dual_personalized sets model.set_alpha(0.0) for this training pass (p_effective = p_global, α=0), sets contrastive_lambda_eff = 0.0 (D-14), runs the BPR + item_perturbation regularization loss only, then restores saved_alpha in a try/finally. Next round (cache exists) the normal alpha blend + contrastive resume."
    - "D-16 alpha diagnostics logged per-round: FitMetricsContract carries alpha_mean / alpha_std / alpha_p25 / alpha_p50 / alpha_p75 / alpha_clip_hit_rate fields (optional). client_app.py computes these from the per-user alpha sigmoid outputs AFTER training completes (so the diagnostics reflect the refined-alpha distribution). When enable-per-user-alpha=false, these fields are omitted or set to None."
    - "13+ new pytest tests land across 3 files: test_client_assertion.py (5 tests — one-user assert, override bypass, evaluator selection, FitMetrics payload with partition_id + alpha diagnostics, EvaluateMetrics payload with partition_id), test_task_rng.py (4 tests — stdlib-random strip, exclusion in training negatives, RNG signature guard, cold-round alpha=0 override), test_embedding_cache_manifest_v2.py (5 tests — schema_version=2 + 12 fields, loud-mismatch RuntimeError with rm -rf hint, reuse-cache sig path, extended LOCAL key payload shape, Phase-3 v1 manifest rejection)."
  artifacts:
    - path: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py"
      provides: "Cross-device @app.train + @app.evaluate with mode resolver, benchmark one-user assert, ADP-02 enable-before-load ordering fix, D-04 schema_version=2 cache-manifest layout, strict contract payloads with partition_id + D-16 alpha diagnostics, discover_only short-circuit, D-13/D-14 cold-round kwarg"
    - path: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py"
      provides: "train dispatcher + train_dual_personalized / train_bpr_mf / train_basic_mf / evaluate_ranking_sampled extended with 5 FND-06 kwargs + is_cold_round kwarg; exclusion-aware negative sampling; _sample_negatives_seeded helper; D-24 snapshot/restore around optimizer.step; stdlib random eradicated"
    - path: "federated-adaptive-personalized-cf/tests/test_client_assertion.py"
      provides: "5 GREEN tests: benchmark one-user assert, override bypass, primary evaluator selection, FitMetrics payload shape with partition_id + alpha diagnostics, EvaluateMetrics payload shape with partition_id"
    - path: "federated-adaptive-personalized-cf/tests/test_task_rng.py"
      provides: "4 GREEN tests: stdlib-random strip regression (task.py + client_app.py body), train negatives exclude test positive, evaluate_ranking_sampled accepts FND-06 signature, cold-round alpha=0 override verified via model.set_alpha spy"
    - path: "federated-adaptive-personalized-cf/tests/test_embedding_cache_manifest_v2.py"
      provides: "5 GREEN tests: v2 manifest sidecar written+loaded, v2 loud-mismatch RuntimeError with rm -rf hint, reuse-cache sig_{hash} path, extended LOCAL key payload shape (base + MLP + fusion + logit_alpha + item_perturbation), schema_version 1→2 rejection"
  key_links:
    - from: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py"
      to: "fedrec_foundation.mode.assert_benchmark_one_user_per_client"
      via: "called before any training; raises AssertionError if num_users_in_partition > 1 under benchmark mode"
      pattern: "assert_benchmark_one_user_per_client"
    - from: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py"
      to: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/dual_personalized_bpr_mf.py"
      via: "enable_per_user_alpha + enable_item_perturbation called BEFORE _load_local_user_state — the ADP-02 fix"
      pattern: "enable_per_user_alpha.*\\n.*enable_item_perturbation.*\\n.*_load_local_user_state"
    - from: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py"
      to: "fedrec_foundation.fit_metrics.FitMetricsContract"
      via: "payload built + validated + sent back in both train and evaluate handlers with partition_id + D-16 alpha diagnostics"
      pattern: "FitMetricsContract\\(|EvaluateMetricsContract\\("
    - from: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py"
      to: "fedrec_foundation.rng.np_rng"
      via: "per-call generator derived from (run_seed, user_idx, round_num, purpose) tuple"
      pattern: "np_rng\\("
---

<objective>
Wire client_app.py + task.py into the Phase-4 cross-device contract, mirroring Phase 3 Plan 03 surgical migration with 5 adaptive-specific deltas:

1. **ADP-02 enable-before-load fix** — In benchmark_cross_device mode, enable_per_user_alpha + enable_item_perturbation are called UNCONDITIONALLY BEFORE _load_local_user_state so the cached _logit_alpha.weight and _item_perturbation.weight tensors are in _LOCAL_PARAMS at load time. This is the primary bug fix of Phase 4.

2. **Schema_version=2 manifest-sidecar cache** — Extend Phase-3 v1 (6 fields: run_id, method, num_users, num_items, dim, split_hash) to v2 (12 fields: v1 + alpha_method, fusion_type, mlp_hidden_dims, per_user_alpha_enabled, item_perturbation_enabled, contrastive_lambda). Any config change that alters cached tensor shape OR semantics hard-fails on load (D-04).

3. **D-13/D-14 cold-round branch** — Thread is_cold_round from the cache-exists probe into train_dual_personalized. When True: force α=0 for this round (prototype-only blend) and skip contrastive loss; restore alpha at end of training pass.

4. **D-24 gradient isolation RETAINED** — Unlike Phase 3 (where PSN-06 collapsed to single-row eliminating the ghost-table problem), Phase 4 keeps the num_users × d user_embeddings table. task.py snapshots non-user rows of user_embeddings + user_bias + _logit_alpha around optimizer.step() and restores them. _item_perturbation is NOT snapshotted (per-item not per-user — item index space is legitimately full-table updated every batch).

5. **D-16 alpha diagnostics** — After training completes, compute per-user alpha sigmoid outputs and populate FitMetricsContract's alpha_mean / alpha_std / alpha_p25 / alpha_p50 / alpha_p75 / alpha_clip_hit_rate fields.

Plus: benchmark one-user assertion (ADP-04), FND-03 exclusion threading (ADP-05), FND-06 RNG wiring (ADP-06 RNG half), strict FitMetrics/EvaluateMetrics payloads with optional partition_id (G-03-01 carry-forward), discover_only short-circuit.

Purpose: Closes ADP-02 (bug fix — primary Phase 4 objective), ADP-04, ADP-05, and the client half of ADP-06.

Output:
- federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py (migrated)
- federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py (RNG + exclusion + cold-round branch + D-24 helpers + training dispatcher)
- 3 new test files with ~14 GREEN tests
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/04-adaptive-migration-bug-fixes/04-CONTEXT.md
@.planning/phases/04-adaptive-migration-bug-fixes/04-RESEARCH.md
@.planning/phases/02-baseline-migration/02-baseline-migration-03-SUMMARY.md
@.planning/phases/03-personalized-migration/03-personalized-migration-03-SUMMARY.md
@.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-01-PLAN.md
@.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-02-PLAN.md

<interfaces>
<!-- Phase 1 foundation surfaces consumed by this plan -->
```python
# fedrec_foundation.mode
def resolve_mode_defaults(mode: str) -> ModeProfile: ...
def log_mode_and_overrides(mode: str, profile: ModeProfile, run_config: Dict) -> Dict[str, Any]: ...
def assert_benchmark_one_user_per_client(profile: ModeProfile, num_users_in_client: int, overrides: Dict[str, Any]) -> None: ...

# fedrec_foundation.rng
def np_rng(run_seed: int, user_idx: int, round_num: int, purpose: str) -> np.random.Generator: ...
def torch_gen(run_seed: int, user_idx: int, round_num: int, purpose: str) -> torch.Generator: ...

# fedrec_foundation.fit_metrics (post Phase 2 Plan 05)
@dataclass
class FitMetricsContract:
    train_loss: float
    num_positives: int
    num_training_examples: int
    round_num: Optional[int] = None
    partition_id: Optional[int] = None                 # G-03-01
    hit_count_overall_at10: Optional[int] = None
    ndcg_sum_overall_at10: Optional[float] = None
    evaluated_users: Optional[int] = None
    # ... 9 more per-group fields ...
    # Phase 4 Plan 03 may need to extend via `extra_diagnostics: Dict[str,float]`
    # OR validate_fit_metrics may need to accept a whitelist of `alpha_*` keys.
    # If the contract currently rejects free-form keys, client_app.py routes alpha diagnostics
    # via a SEPARATE MetricRecord keyed "alpha_diagnostics" — confirm in the foundation source.
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FitMetricsContract": ...

def validate_fit_metrics(payload: Dict[str, Any]) -> None: ...

@dataclass
class EvaluateMetricsContract:
    hit_count_overall_at10: int
    ndcg_sum_overall_at10: float
    evaluated_users: int
    eval_loss: Optional[float] = None
    sampled_hr_at10: Optional[float] = None
    sampled_ndcg_at10: Optional[float] = None
    partition_id: Optional[int] = None
    # 9 per-group fields
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvaluateMetricsContract": ...

def validate_evaluate_metrics(payload: Dict[str, Any]) -> None: ...

# fedrec_foundation.atomic
def atomic_write_json(path: Path, data: Dict[str, Any]) -> None: ...

# fedrec_foundation.evaluator
def get_primary_evaluator(mode: str) -> str: ...   # returns "sampled_loo_99" for cross-device

# fedrec_foundation.user_groups
def classify_user_group(n_interactions: int) -> str: ...   # "sparse"/"medium"/"dense"
```

<!-- Phase 4 Plan 01 AdaptiveSplitFedAvg + strategy constants -->
```python
# federated_adaptive_personalized_cf.strategy (post Plan 01)
GLOBAL_PARAM_KEYS = frozenset({'item_embeddings.weight', 'item_bias.weight', 'global_bias'})
LOCAL_PARAM_KEYS_BASE = frozenset({'user_embeddings.weight', 'user_bias.weight'})
USER_PROTOTYPE_KEY = 'user_prototype'
```

<!-- DualPersonalizedBPRMF contract (UNCHANGED by Phase 4) -->
```python
class DualPersonalizedBPRMF(nn.Module):
    num_users: int                      # 6040
    num_items: int                      # 3706
    user_embeddings: nn.Embedding       # (num_users, embedding_dim) — ghost table
    user_bias: Optional[nn.Embedding]   # (num_users, 1)
    item_embeddings: nn.Embedding       # (num_items, embedding_dim) — GLOBAL
    item_bias: Optional[nn.Embedding]   # (num_items, 1) — GLOBAL
    global_bias: Optional[nn.Parameter] # scalar — GLOBAL
    personal_mlp: nn.Sequential
    fusion_gate: Optional[nn.Parameter] # fusion_type == 'gate'
    fusion_layer: Optional[nn.Linear]   # fusion_type == 'concat'
    _logit_alpha: Optional[nn.Embedding]      # (num_users, 1) after enable_per_user_alpha
    _item_perturbation: Optional[nn.Embedding] # (num_items, embedding_dim) after enable_item_perturbation

    @property
    def _LOCAL_PARAMS(self) -> tuple: ...     # dynamic based on flags + fusion_type

    def enable_per_user_alpha(self, num_users: int, init_alphas: Dict[int, float]) -> None: ...
    def enable_item_perturbation(self, reg_lambda: float) -> None: ...
    def set_alpha(self, alpha: float) -> None: ...          # sets the scalar client alpha (fallback path)
    def get_alpha(self) -> float: ...                        # returns current scalar alpha
    def set_global_prototype(self, proto: torch.Tensor) -> None: ...
    def get_global_parameters(self) -> OrderedDict: ...
    def set_global_parameters(self, state: Dict[str, torch.Tensor]) -> None: ...
    def get_local_parameters(self) -> OrderedDict: ...
    def set_local_parameters(self, state: Dict[str, torch.Tensor], strict: bool = False) -> Tuple[List[str], List[str]]: ...
    def compute_user_prototype(self) -> torch.Tensor: ...
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: task.py migration — FND-06 RNG + FND-03 exclusion + _sample_negatives_seeded + D-13/D-14 cold-round branch + D-24 ghost-table gradient isolation for user_embeddings + user_bias + _logit_alpha (ADP-05, ADP-06 RNG half)</name>
  <files>federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py, federated-adaptive-personalized-cf/tests/test_task_rng.py</files>
  <read_first>
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py (ENTIRE FILE — inventory `import random`, `random.seed(seed)` at line 952-953, `random.sample(` at line 1012, the existing train dispatcher, train_dual_personalized, train_bpr_mf, train_basic_mf, evaluate_ranking_sampled signatures + compute_client_alpha / compute_per_user_alpha / get_user_stats helpers — those latter three are USED by client_app.py and must stay callable)
    - federated-baseline-cf/federated_baseline_cf/task.py (post-Plan-03 TEMPLATE — the _sample_negatives_seeded helper + _snapshot_non_user_rows / _restore_non_user_rows D-24 helpers + the 5-kwarg extension pattern)
    - federated-baseline-cf/tests/test_task_rng.py (4-test TEMPLATE; Phase 2 version covered D-24 gradient masking — that test SHOULD port to Phase 4 against the GHOST TABLE user_embeddings since the adaptive module does NOT collapse)
    - federated-personalized-cf/federated_personalized_cf/task.py (Phase 3 version — no D-24 masking because single-row model ate the problem; Phase 4 is DIFFERENT — D-24 IS needed)
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/dual_personalized_bpr_mf.py (observe: user_embeddings is nn.Embedding(num_users, d), user_bias is nn.Embedding(num_users, 1), _logit_alpha is nn.Embedding(num_users, 1) — all three user-indexed; _item_perturbation is nn.Embedding(num_items, d) — item-indexed)
    - .planning/phases/04-adaptive-migration-bug-fixes/04-RESEARCH.md §Pitfall 3/4 + §Open Question 4 (D-24 gradient isolation applies to user_embeddings + user_bias + _logit_alpha; NOT to _item_perturbation) + §Pattern 3 (cold-round branch exact code shape) + §"Runtime State Inventory" (user must `rm -rf federated-adaptive-personalized-cf/.embedding_cache/` before first Phase-4 run — document in plan)
  </read_first>
  <behavior>
    Tests to write FIRST (RED), then implement. 4 tests:

    1. test_random_seed_calls_stripped: reads BOTH federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py AND federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py source files; asserts no regex matches for `^import random$` OR `random\.seed\(` OR `random\.sample\(` anywhere in either file's body (Phase-2 cross-file guard pattern).

    2. test_train_negatives_exclude_test_positive: build a minimal DualPersonalizedBPRMF(num_users=2, num_items=10, embedding_dim=4, mlp_hidden_dims=[8], fusion_type='add'); construct a 1-user trainloader with positive items [0, 1, 2]; call train dispatcher with exclude_items={5}, item_id=5 intentionally the held-out test item; assert no crash; assert _sample_negatives_seeded does NOT return 5 by probing with a fixed rng (construct rng = np_rng(42, 0, 1, 'train_neg'); call _sample_negatives_seeded(user_rated={0,1,2,5}, num_items=10, num_negatives=3, rng=rng) 20 times; assert 5 never appears).

    3. test_evaluate_ranking_sampled_accepts_rng_signature: introspects inspect.signature(evaluate_ranking_sampled); asserts parameter names include `run_seed`, `user_idx`, `round_num`, `exclude_items`.

    4. test_cold_round_sets_alpha_zero_and_skips_contrastive: mock model.set_alpha with a spy (use unittest.mock); build a tiny trainloader; call train_dual_personalized(model, trainloader, epochs=1, lr=0.01, device='cpu', run_seed=42, user_idx=0, round_num=1, exclude_items=set(), rng=np_rng(42, 0, 1, 'train_neg'), is_cold_round=True, contrastive_lambda=0.1, contrastive_tau=0.1, ...); assert model.set_alpha was called with 0.0 during the cold round AND was called with the saved_alpha value at end (restoration); assert contrastive loss was not contributed to (probe via a pytest capsys or a module-level counter exposed for testing).
  </behavior>
  <action>
    Step 1 — Add imports to task.py top; remove `import random` if it's at module top:
    ```python
    from typing import Dict, Iterable, List, Optional, Set, Tuple

    import numpy as np
    import torch

    from fedrec_foundation.rng import np_rng, torch_gen
    ```
    If any comments/docstrings mention the stdlib `random` API, re-word to avoid the literal substrings `random.seed(`, `random.sample(`, `import random` (the acceptance grep is plain regex — docstring mentions are false positives).

    Step 2 — Add 4 new module-level private helpers (clone from baseline, adapt for ghost-table).

    2a. _sample_negatives_seeded (identical to Phase 3 Plan 03 Task 1 Step 2):
    ```python
    def _sample_negatives_seeded(
        user_rated_items: Set[int],
        num_items: int,
        num_negatives: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Rejection-sampled negatives from an np.random.Generator instance.

        Deterministic under a given rng (FND-06). Used to replace model.sample_negatives
        process-global np.random.randint calls (ADP-06).
        """
        out: List[int] = []
        pool = num_items
        max_tries = num_negatives * 64 + 16
        while len(out) < num_negatives and max_tries > 0:
            cand = int(rng.integers(0, pool))
            if cand not in user_rated_items:
                out.append(cand)
            max_tries -= 1
        return np.asarray(out, dtype=np.int64)
    ```

    2b. D-24 gradient isolation helpers (CLONE Phase-2 Plan 03 pattern; applies to user_embeddings + user_bias + _logit_alpha). Because the adaptive model keeps the ghost table, the Phase-2 approach (snapshot/restore bracketing optimizer.step) is the correct mitigation:
    ```python
    # D-24 gradient isolation — protects non-user rows of the ghost-table LOCAL params
    # from Adam's weight-decay + momentum drift. Applies to:
    #   - user_embeddings (num_users, d)   — ghost table, user-indexed
    #   - user_bias       (num_users, 1)   — ghost table, user-indexed
    #   - _logit_alpha    (num_users, 1)   — ghost table, user-indexed (when enable_per_user_alpha)
    # Does NOT apply to _item_perturbation (num_items, d) — item-indexed, legitimate full-
    # table updates each batch.
    _D24_PROTECTED_EMBEDDINGS = (
        "user_embeddings",
        "user_bias",
        "_logit_alpha",
    )


    def _snapshot_non_user_rows(model, user_idx: int) -> Dict[str, torch.Tensor]:
        """Return a cloned copy of every D-24-protected embedding weight, with the
        user_idx row replaced by NaN so `_restore_non_user_rows` never overwrites the
        legitimate post-step update on that row.
        """
        snapshots: Dict[str, torch.Tensor] = {}
        for name in _D24_PROTECTED_EMBEDDINGS:
            module = getattr(model, name, None)
            if module is None:
                continue
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            snap = weight.detach().clone()
            snap[int(user_idx)] = float("nan")
            snapshots[name] = snap
        return snapshots


    def _restore_non_user_rows(model, snapshots: Dict[str, torch.Tensor]) -> None:
        """Copy every non-NaN row of each snapshot back into the model weight. The NaN
        row at user_idx is left untouched so the legitimate gradient update on that row
        survives.
        """
        with torch.no_grad():
            for name, snap in snapshots.items():
                module = getattr(model, name, None)
                if module is None:
                    continue
                weight = getattr(module, "weight", None)
                if weight is None:
                    continue
                mask = ~torch.isnan(snap).any(dim=tuple(range(1, snap.dim())))
                weight.data[mask] = snap[mask]
    ```

    2c. Cold-round helpers live INSIDE train_dual_personalized (see Step 3) — no additional module-level helper needed.

    Step 3 — Extend `train_dual_personalized` (the adaptive-specific training fn) with FND-06 kwargs + is_cold_round + D-24 bracketing. Exact shape:
    ```python
    def train_dual_personalized(
        model,
        trainloader,
        *,
        epochs: int,
        lr: float,
        device,
        num_negatives: int = 1,
        weight_decay: float = 1e-5,
        proximal_mu: float = 0.0,
        global_params: Optional[List[torch.Tensor]] = None,
        global_param_names: Optional[List[str]] = None,
        contrastive_lambda: float = 0.0,
        contrastive_tau: float = 0.1,
        # ==== FND-06 + FND-03 kwargs (NEW; backward-compatible with defaults) ====
        run_seed: Optional[int] = None,
        user_idx: Optional[int] = None,
        round_num: Optional[int] = None,
        exclude_items: Optional[Iterable[int]] = None,
        rng: Optional[np.random.Generator] = None,
        # ==== D-13/D-14 cold-round override (NEW) ====
        is_cold_round: bool = False,
        **existing_kwargs,
    ) -> float:
        """..."""
        if rng is None and run_seed is not None and user_idx is not None:
            rng = np_rng(int(run_seed), int(user_idx), int(round_num or 0), "train_neg")

        # D-13 + D-14: cold-round overrides
        saved_alpha: Optional[float] = None
        contrastive_lambda_eff = contrastive_lambda
        if is_cold_round:
            if hasattr(model, "get_alpha") and hasattr(model, "set_alpha"):
                saved_alpha = float(model.get_alpha())
                model.set_alpha(0.0)  # D-13: prototype-only blend
            contrastive_lambda_eff = 0.0  # D-14: skip contrastive on cold round

        try:
            # ==== existing train_dual_personalized body below (preserved D-18) ====
            # 1. Build user_rated_items from trainloader (single-user under cross-device).
            # 2. Fold exclude_items into user_rated_items (FND-03 / ADP-05):
            user_rated: Set[int] = set()
            for batch in trainloader:
                items = batch["item"].numpy()
                user_rated.update(int(i) for i in items)
            if exclude_items is not None:
                user_rated |= set(int(x) for x in exclude_items)

            # 3. Set up optimizer. Keep existing Adam(weight_decay=...) construction.
            #    FedProx proximal term applies ONLY to global_params (D-18 architectural).

            # 4. Training loop. Per batch:
            #    a. Snapshot non-user rows of D-24-protected embeddings.
            snapshots = _snapshot_non_user_rows(model, int(user_idx)) if user_idx is not None else {}
            #    b. Forward + backward + optimizer.step().
            #    c. Restore non-user rows of D-24-protected embeddings.
            # _restore_non_user_rows(model, snapshots)
            #    Replace the old `negatives = model.sample_negatives(user_rated, num_negatives)`
            #    (uses process-global np.random) with _sample_negatives_seeded when rng provided:
            #    if rng is not None:
            #        neg_ids = _sample_negatives_seeded(user_rated, model.num_items, num_negatives, rng)
            #    else:
            #        neg_ids = model.sample_negatives(user_rated, num_negatives)  # backward compat
            #    ... use contrastive_lambda_eff (not contrastive_lambda) when computing loss ...
            pass  # placeholder — executor fills in the body based on existing train_dual_personalized

        finally:
            # Restore alpha so subsequent evaluation sees the original value (D-13 cleanup)
            if is_cold_round and saved_alpha is not None and hasattr(model, "set_alpha"):
                model.set_alpha(saved_alpha)

        return train_loss
    ```

    The executor fills in the body of train_dual_personalized by PRESERVING the existing arithmetic (BPR loss + contrastive + item_perturbation regularization + FedProx proximal) and adding: (a) the rng-based negative sampling replacement, (b) the snapshot/restore bracket around optimizer.step, (c) the is_cold_round / contrastive_lambda_eff substitution, (d) the exclude_items fold into user_rated, (e) the try/finally for set_alpha restoration.

    Step 4 — Extend `train_bpr_mf` and `train_basic_mf` with the same 5 FND-06 kwargs + is_cold_round (though is_cold_round is a no-op for the non-dual pathways since they don't consume alpha); add the snapshot/restore bracket around optimizer.step; replace model.sample_negatives with _sample_negatives_seeded when rng provided; fold exclude_items into user_rated.

    Step 5 — Extend `evaluate_ranking_sampled` with 4 new keyword-only parameters (mirrors Phase 3 Plan 03 Task 1 Step 5):
    ```python
    def evaluate_ranking_sampled(
        model, testloader, trainloader,
        *,
        device,
        k_values: Optional[List[int]] = None,
        num_negatives: int = 99,
        seed: int = 42,   # LEGACY — now IGNORED; documented in docstring
        # FND-06 kwargs (NEW)
        run_seed: Optional[int] = None,
        user_idx: Optional[int] = None,
        round_num: Optional[int] = None,
        exclude_items: Optional[Iterable[int]] = None,
        **existing_kwargs,
    ) -> Dict[str, float]:
        """..."""
    ```
    STRIP the `import random` + `random.seed(seed)` at lines 952-953 + `random.sample(...)` at ~line 1012. Replace with:
    ```python
    rng = None
    if run_seed is not None and user_idx is not None:
        rng = np_rng(int(run_seed), int(user_idx), int(round_num or 0), "eval_neg")
    # ... when choosing negatives ...
    if rng is not None:
        negative_items = rng.choice(negative_candidates, size=num_negatives, replace=False).tolist()
    else:
        # deterministic fallback if caller did not provide run_seed
        gen = np.random.default_rng(int(seed))
        negative_items = gen.choice(negative_candidates, size=num_negatives, replace=False).tolist()
    # Fold exclude_items into negative_candidates pool construction:
    candidate_pool = set(range(num_total_items)) - user_rated_items - set(exclude_items or [])
    ```
    Docstring MUST explicitly note that the legacy `seed: int = 42` parameter is IGNORED when run_seed is provided.

    Step 6 — Update the `train` dispatcher to forward the 6 new kwargs (5 FND-06 + 1 is_cold_round) to train_dual_personalized / train_bpr_mf / train_basic_mf. Preserve the existing dispatch-on-model-type logic.

    Step 7 — Preserve verbatim (D-18): `load_data`, `get_model`, `test`, `compute_client_alpha`, `compute_per_user_alpha`, `get_user_stats`, any module-level caches.

    Step 8 — Create federated-adaptive-personalized-cf/tests/test_task_rng.py with the 4 tests listed in <behavior>. For test_random_seed_calls_stripped:
    ```python
    import re
    from pathlib import Path
    import pytest

    _MODULE_DIR = Path(__file__).resolve().parents[1] / "federated_adaptive_personalized_cf"
    _TASK_PY = _MODULE_DIR / "task.py"
    _CLIENT_APP_PY = _MODULE_DIR / "client_app.py"

    @pytest.mark.parametrize("src_path", [_TASK_PY, _CLIENT_APP_PY])
    def test_random_seed_calls_stripped(src_path):
        src = src_path.read_text()
        # Strip ALL comments (lines whose stripped content starts with "#") and docstrings
        # before regex matching, so false-positives in documentation are ignored.
        # Simple approach: check non-comment non-docstring regions only.
        import ast
        tree = ast.parse(src)
        offending = re.findall(r"^\s*import random\s*$|random\.seed\(|random\.sample\(", src, re.MULTILINE)
        # Filter out matches inside docstrings / comments — ast-based strip:
        for m in list(offending):
            # Remove this match from src once per occurrence, then re-check the
            # stripped-of-docstrings version. For simplicity, assert the count in source
            # (comment false-positives are acceptable per BSL-05 precedent).
            pass
        assert len(offending) == 0, (
            f"{src_path.name} still contains stdlib random calls: {offending}. "
            f"Use fedrec_foundation.rng.np_rng(run_seed, user_idx, round_num, purpose) instead."
        )
    ```

    For test_cold_round_sets_alpha_zero_and_skips_contrastive:
    ```python
    import numpy as np
    import torch
    from unittest.mock import MagicMock
    from fedrec_foundation.rng import np_rng
    from federated_adaptive_personalized_cf.models.dual_personalized_bpr_mf import DualPersonalizedBPRMF

    def test_cold_round_sets_alpha_zero_and_skips_contrastive():
        from federated_adaptive_personalized_cf.task import train_dual_personalized
        # Minimal DualPersonalizedBPRMF + minimal trainloader — just enough for 1 epoch
        model = DualPersonalizedBPRMF(num_users=2, num_items=10, embedding_dim=4,
                                       mlp_hidden_dims=[8], fusion_type="add", use_bias=True)
        # Manually set an alpha that differs from 0 so we can see the override.
        model.set_alpha(0.7)
        original_set_alpha = model.set_alpha
        spy = MagicMock(side_effect=original_set_alpha)
        model.set_alpha = spy

        # Fake 1-user trainloader: 3 positives
        class FakeDS:
            def __init__(self):
                self.data = [{"user": torch.tensor([0,0,0]), "item": torch.tensor([0,1,2]), "rating": torch.tensor([1.0,1.0,1.0])}]
            def __iter__(self): return iter(self.data)
        trainloader = FakeDS()

        rng = np_rng(42, 0, 1, "train_neg")
        train_dual_personalized(
            model, trainloader, epochs=1, lr=0.01, device="cpu",
            run_seed=42, user_idx=0, round_num=1,
            exclude_items=set(), rng=rng,
            is_cold_round=True, contrastive_lambda=0.1, contrastive_tau=0.1,
        )
        calls = [c.args[0] for c in spy.call_args_list if c.args]
        # The very first set_alpha call inside train_dual_personalized must be 0.0 (D-13).
        assert any(abs(a - 0.0) < 1e-9 for a in calls), (
            f"D-13 violated: train_dual_personalized did not call set_alpha(0.0) on cold round. Calls: {calls}"
        )
        # The final set_alpha call must restore the saved alpha (0.7).
        assert abs(calls[-1] - 0.7) < 1e-9, (
            f"D-13 cleanup violated: saved alpha not restored. Last call: {calls[-1]}"
        )
    ```

    Step 9 — Verify: `cd federated-adaptive-personalized-cf && pytest tests/test_task_rng.py -v` → 4 passed. Full suite: `pytest tests/ -v` → ~14 passed (Plan 01=10 + Plan 02=9 + Plan 03 Task 1=4 = 23 after all commits of 03 Task 1, but Task 2 lands additional tests).

    Step 10 — Commit (--no-verify; Wave-2 parallel rule):
    ```
    git add federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py \
            federated-adaptive-personalized-cf/tests/test_task_rng.py
    git commit --no-verify -m "feat(04-03): task.py FND-06 RNG + FND-03 exclusion + D-13/D-14 cold-round + D-24 ghost-table isolation (ADP-05, ADP-06)"
    ```
  </action>
  <acceptance_criteria>
    - `grep -cE "^import random$" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py` returns 0
    - `grep -cE "random\\.seed\\(" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py` returns 0
    - `grep -cE "random\\.sample\\(" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py` returns 0
    - `grep -c "from fedrec_foundation.rng import" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py` returns 1
    - `grep -c "def _sample_negatives_seeded" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py` returns 1
    - `grep -c "def _snapshot_non_user_rows" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py` returns 1
    - `grep -c "def _restore_non_user_rows" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py` returns 1
    - `grep -c "_D24_PROTECTED_EMBEDDINGS" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py` returns at least 2 (tuple declaration + helper body reference)
    - `grep -cE '"_logit_alpha"|"user_embeddings"|"user_bias"' federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py` returns at least 3 (inside _D24_PROTECTED_EMBEDDINGS tuple)
    - `grep -c '"_item_perturbation"' federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py` returns 0 OR the string only appears inside a D-24 negative-filter comment (item-indexed, explicitly excluded from D-24 per Research open question 4)
    - `grep -cE "is_cold_round" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py` returns at least 4 (param + D-13 branch + D-14 branch + restoration finally)
    - `grep -cE "run_seed" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py` returns at least 4 (train dispatcher + train_dual_personalized + train_bpr_mf + evaluate_ranking_sampled)
    - `grep -cE "exclude_items" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py` returns at least 4
    - `cd federated-adaptive-personalized-cf && pytest tests/test_task_rng.py -v` exits 0 with "4 passed"
    - `python -c "import inspect; from federated_adaptive_personalized_cf.task import evaluate_ranking_sampled; p = inspect.signature(evaluate_ranking_sampled).parameters; assert 'run_seed' in p and 'user_idx' in p and 'round_num' in p and 'exclude_items' in p; print('ok')"` prints `ok`
    - `python -c "import inspect; from federated_adaptive_personalized_cf.task import train_dual_personalized; p = inspect.signature(train_dual_personalized).parameters; assert 'run_seed' in p and 'user_idx' in p and 'round_num' in p and 'exclude_items' in p and 'rng' in p and 'is_cold_round' in p; print('ok')"` prints `ok`
    - `git diff --stat federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/ federated-adaptive-personalized-cf/pyproject.toml` returns empty after commit (D-18 scope; Plans 01/02/05 own those)
  </acceptance_criteria>
  <done>task.py threads FND-06 RNG + FND-03 exclusion-set into all training fns and the sampled evaluator; _sample_negatives_seeded helper replaces process-global np.random; D-24 ghost-table gradient isolation (user_embeddings + user_bias + _logit_alpha only) restored via _snapshot_non_user_rows/_restore_non_user_rows around optimizer.step; D-13/D-14 is_cold_round threads into train_dual_personalized with alpha=0 override + contrastive skip + try/finally restoration; stdlib random eradicated from task.py; 4 GREEN tests guard the contract.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: client_app.py migration — mode resolver + ADP-02 enable-before-load ordering fix + one-user assert + discover_only + schema_v2 manifest-sidecar cache + strict contracts with alpha diagnostics + cold-round signal (ADP-02, ADP-04, ADP-06 client half)</name>
  <files>federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py, federated-adaptive-personalized-cf/tests/test_client_assertion.py, federated-adaptive-personalized-cf/tests/test_embedding_cache_manifest_v2.py</files>
  <read_first>
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py (ENTIRE FILE — preserve pre-existing WIP per D-18; lines ~230-363 contain the bug-site: construct model → set_global_parameters → load_local_user_embeddings → enable_per_user_alpha → enable_item_perturbation — the rip target is REORDERING these calls: enable_* BEFORE _load_local_user_state, not after)
    - federated-personalized-cf/federated_personalized_cf/client_app.py (CANONICAL Phase-3 TEMPLATE — mode resolver, one-user assert, strict FitMetrics/EvaluateMetrics payloads, per-group sufficient-stat routing, discover_only short-circuit, schema_v1 manifest-sidecar _signature_fields / _cache_dir_for_run / _save_local_user_state / _load_local_user_state helpers — Phase 4 extends to schema_version=2)
    - federated-personalized-cf/tests/test_client_assertion.py (5-test TEMPLATE for one-user assert + FitMetrics payload — adapt for D-16 alpha diagnostics)
    - federated-personalized-cf/tests/test_embedding_cache_manifest.py (4-test TEMPLATE for manifest sidecar — Phase 4 extends with v2 signature field count + v1→v2 rejection test)
    - scripts/foundation/fedrec_foundation/mode.py (assert_benchmark_one_user_per_client signature)
    - scripts/foundation/fedrec_foundation/atomic.py (atomic_write_json signature)
    - scripts/foundation/fedrec_foundation/fit_metrics.py (FitMetricsContract fields — check if alpha_mean/alpha_std/alpha_clip_hit_rate are accepted or if they require `extra_diagnostics` dict sidecar; Phase 2 Plan 01 locked strict contract (D-21), so extra keys are REJECTED by validate_fit_metrics. Phase 4 routes alpha diagnostics via a SEPARATE `MetricRecord({"alpha_diagnostics": {...}})` or via a sibling dict in the train-response content; confirm by reading foundation source before implementing)
    - scripts/foundation/fedrec_foundation/user_groups.py (classify_user_group)
    - .planning/phases/04-adaptive-migration-bug-fixes/04-CONTEXT.md §D-01..D-04 + §D-16 (manifest schema + alpha diagnostic fields)
    - .planning/phases/04-adaptive-migration-bug-fixes/04-RESEARCH.md §"Pattern 1: Enable-Before-Load Ordering" (lines ~226-280 — the exact reordered sequence) + §"Pattern 4: Schema-Version=2 Manifest" (lines ~378-396) + §Pitfall 1/5 (bug symptoms)
    - .planning/phases/03-personalized-migration/03-personalized-migration-03-SUMMARY.md (G-03-01 discover_only + partition_id pattern; _signature_fields extension recipe)
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/evaluation/alpha_analysis.py (exposes AlphaAnalyzer — Research §Pattern 5 recommends refactoring a `compute_scalar_summary(per_user_alphas) -> Dict[str, float]` method so aggregate_evaluate can call per-round; if that method does not yet exist, Task 2 either exposes it OR client_app.py computes the 6 alpha diagnostic scalars inline after training)
  </read_first>
  <behavior>
    Tests to write FIRST (RED), then implement. 5 client_assertion tests + 5 manifest_v2 tests.

    test_client_assertion.py:
    1. test_benchmark_mode_asserts_one_user: `assert_benchmark_one_user_per_client(profile, 3, {})` raises AssertionError containing "one user"; `(profile, 1, {})` returns without raising.
    2. test_benchmark_mode_override_bypass: `(profile, 50, {"num_supernodes": 10})` returns without raising (D-10 Phase-1 override path).
    3. test_get_primary_evaluator_selects_sampled_loo_99: assert `get_primary_evaluator("benchmark_cross_device") == "sampled_loo_99"`.
    4. test_fit_metrics_contract_payload_with_partition_id_and_alpha_diagnostics: builds FitMetricsContract(...) with partition_id=42 + the per-group fields; to_dict() contains 'partition_id' == 42. Separately, confirm the alpha diagnostics path: build a MetricRecord (or equivalent Flower content-dict) that carries {"alpha_diagnostics": {"alpha_mean": 0.5, "alpha_std": 0.1, "alpha_p25": 0.3, "alpha_p50": 0.5, "alpha_p75": 0.7, "alpha_clip_hit_rate": 0.05}} and round-trip via dict.
    5. test_evaluate_metrics_contract_payload_shape_with_partition_id: EvaluateMetricsContract(..., partition_id=1234).to_dict() contains 'partition_id'; validate_evaluate_metrics(payload) passes; payload with an unknown key fails with ValueError (D-21 strict contract).

    test_embedding_cache_manifest_v2.py:
    1. test_manifest_v2_sidecar_written_and_loaded: call helper `_save_local_user_state(partition_id=0, state_dict, run_id="r1", num_users=6040, num_items=3706, dim=64, split_hash="abc123", alpha_method="hierarchical_conditional", fusion_type="concat", mlp_hidden_dims="512,256,128", per_user_alpha_enabled=True, item_perturbation_enabled=True, contrastive_lambda=0.1)` into tmp_path; verify `.embedding_cache/r1/manifest.json` exists with schema_version=2 and all 12 signature fields; verify `.embedding_cache/r1/partition_0.pt` exists and contains the LOCAL key set (base + MLP + fusion + logit_alpha + item_perturbation).
    2. test_manifest_v2_mismatch_raises_runtime_error: seed cache with alpha_method="hierarchical_conditional"; attempt load with alpha_method="multi_factor"; assert RuntimeError raised; assert error message contains "alpha_method" AND "rm -rf .embedding_cache/" with the specific run_id path.
    3. test_reuse_cache_sig_path_v2: call save/load with reuse_cache=True; assert path resolves to `.embedding_cache/sig_<16-hex-chars>/`; two runs with identical Phase-4 signature hash to same sig path.
    4. test_extended_local_key_payload_shape: save a state_dict containing user_embeddings.weight (6040, 64) + user_bias.weight (6040, 1) + personal_mlp.*.weight/bias + fusion_layer.weight + fusion_layer.bias + _logit_alpha.weight (6040, 1) + _item_perturbation.weight (3706, 64); load it back; assert all keys round-trip byte-identical (torch.equal).
    5. test_schema_v1_manifest_raises_when_loading_under_v2: manually write a manifest.json with schema_version=1 and only 6 fields; attempt load under v2; assert RuntimeError raised with "schema_version" in error message AND "rm -rf" hint.
  </behavior>
  <action>
    Step 1 — Add imports to client_app.py top (preserve existing imports — ADD to them):
    ```python
    import hashlib
    import json
    import os
    import tempfile
    from collections import OrderedDict
    from pathlib import Path
    from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

    import numpy as np
    import torch
    from flwr.common import ArrayRecord, ConfigRecord, Message, MessageType, MetricRecord, RecordDict

    from fedrec_foundation.atomic import atomic_write_json
    from fedrec_foundation.evaluator import get_primary_evaluator
    from fedrec_foundation.fit_metrics import (
        EvaluateMetricsContract, FitMetricsContract,
        validate_evaluate_metrics, validate_fit_metrics,
    )
    from fedrec_foundation.mode import (
        assert_benchmark_one_user_per_client,
        log_mode_and_overrides,
        resolve_mode_defaults,
    )
    from fedrec_foundation.rng import np_rng
    from fedrec_foundation.user_groups import classify_user_group
    ```
    REMOVE any top-level `import random` (grep + strip).

    Step 2 — Add module-level helpers. Phase-4 schema_version=2 signature + cache helpers:

    2a. `_signature_fields_v2(...)` — 12 fields. Use keyword-only args for all 12:
    ```python
    def _signature_fields_v2(
        *,
        run_id: str,
        method: str,
        num_users: int,
        num_items: int,
        dim: int,
        split_hash: str,
        alpha_method: str,
        fusion_type: str,
        mlp_hidden_dims: str,          # joined string e.g. "512,256,128"
        per_user_alpha_enabled: bool,
        item_perturbation_enabled: bool,
        contrastive_lambda: float,
    ) -> Dict[str, Any]:
        """D-02 Phase-4 schema_version=2 signature (12 fields: 6 Phase-3 + 6 Phase-4)."""
        return {
            "schema_version": 2,
            "run_id": str(run_id),
            "method": str(method),             # e.g. "dual"
            "num_users": int(num_users),
            "num_items": int(num_items),
            "dim": int(dim),
            "split_hash": str(split_hash),
            "alpha_method": str(alpha_method),
            "fusion_type": str(fusion_type),
            "mlp_hidden_dims": str(mlp_hidden_dims),
            "per_user_alpha_enabled": bool(per_user_alpha_enabled),
            "item_perturbation_enabled": bool(item_perturbation_enabled),
            "contrastive_lambda": float(contrastive_lambda),
        }
    ```

    2b. `_cache_dir_for_run(*, run_id, reuse_cache, signature) -> Path`. Mirror Phase 3 Plan 03 verbatim:
    ```python
    def _cache_dir_for_run(*, run_id: str, reuse_cache: bool, signature: Dict[str, Any]) -> Path:
        base = Path(".embedding_cache")
        if not reuse_cache:
            return base / str(run_id)
        # Exclude run_id from the content hash so D-09 reuse works across runs.
        payload = json.dumps({k: v for k, v in signature.items() if k != "run_id"},
                             sort_keys=True).encode("utf-8")
        sig_hex = hashlib.sha256(payload).hexdigest()[:16]
        return base / f"sig_{sig_hex}"
    ```

    2c. `_save_local_user_state(*, partition_id, state_dict, run_id, reuse_cache, signature)` — extends Phase 3 with extended LOCAL key set shape guard:
    ```python
    def _save_local_user_state(
        *,
        partition_id: int,
        state_dict: Dict[str, torch.Tensor],
        run_id: str,
        reuse_cache: bool,
        signature: Dict[str, Any],
    ) -> None:
        """Atomic single-file save of ALL LOCAL params (base + MLP + fusion + optional
        _logit_alpha + optional _item_perturbation) in one .pt blob (CONTEXT D-01).
        """
        cache_dir = _cache_dir_for_run(run_id=run_id, reuse_cache=reuse_cache, signature=signature)
        cache_dir.mkdir(parents=True, exist_ok=True)
        pt_path = cache_dir / f"partition_{int(partition_id)}.pt"
        # Shape guard: LOCAL key set MUST include user_embeddings.weight + user_bias.weight at
        # minimum; _logit_alpha.weight present iff signature["per_user_alpha_enabled"]; etc.
        required = {"user_embeddings.weight"}
        if signature.get("per_user_alpha_enabled"):
            required.add("_logit_alpha.weight")
        if signature.get("item_perturbation_enabled"):
            required.add("_item_perturbation.weight")
        missing = required - set(state_dict.keys())
        assert not missing, (
            f"D-01/D-10 violated: LOCAL state is missing required keys {missing}. "
            f"Got: {sorted(state_dict.keys())}"
        )
        # Atomic write via tempfile + os.replace. Use 'partition_tmp_' prefix per Phase-3
        # Rule-1 auto-fix — torch.save's PyTorchFileWriter rejects dot-prefixed tempfile names.
        fd, tmp = tempfile.mkstemp(prefix="partition_tmp_", suffix=".pt", dir=str(cache_dir))
        os.close(fd)
        try:
            torch.save(OrderedDict(state_dict), tmp)
            os.replace(tmp, pt_path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        atomic_write_json(cache_dir / "manifest.json", signature)
    ```

    2d. `_load_local_user_state(*, partition_id, run_id, reuse_cache, signature)` — D-04 loud mismatch with rm -rf hint:
    ```python
    def _load_local_user_state(
        *,
        partition_id: int,
        run_id: str,
        reuse_cache: bool,
        signature: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        cache_dir = _cache_dir_for_run(run_id=run_id, reuse_cache=reuse_cache, signature=signature)
        pt_path = cache_dir / f"partition_{int(partition_id)}.pt"
        manifest_path = cache_dir / "manifest.json"
        if not pt_path.exists() or not manifest_path.exists():
            return None  # cold start
        with open(manifest_path, "r") as f:
            cached = json.load(f)
        # D-04: compare every signature field. schema_version mismatch (1 vs 2) triggers loud fail.
        all_keys = (
            "schema_version", "run_id", "method", "num_users", "num_items", "dim", "split_hash",
            "alpha_method", "fusion_type", "mlp_hidden_dims",
            "per_user_alpha_enabled", "item_perturbation_enabled", "contrastive_lambda",
        )
        deltas: List[str] = []
        for key in all_keys:
            if reuse_cache and key == "run_id":
                continue
            if cached.get(key) != signature.get(key):
                deltas.append(f"  {key}: cached={cached.get(key)!r}, current={signature.get(key)!r}")
        if deltas:
            raise RuntimeError(
                "Embedding-cache signature mismatch (D-04, schema_version=2):\n"
                + "\n".join(deltas)
                + f"\nRun: rm -rf {cache_dir}/"
            )
        state = torch.load(pt_path, map_location="cpu", weights_only=True)
        return state
    ```

    2e. `_classify_partition_user_group(bundle, partition_id)` — same as Phase 3 Plan 03.

    2f. `_compute_alpha_diagnostics(model) -> Dict[str, float]` — D-16 6-scalar summary:
    ```python
    def _compute_alpha_diagnostics(model) -> Optional[Dict[str, float]]:
        """Compute 6 scalar alpha diagnostics over per-user alpha sigmoid outputs.

        Returns None when per-user alpha is not enabled on the model.
        Keys: alpha_mean, alpha_std, alpha_p25, alpha_p50, alpha_p75, alpha_clip_hit_rate.
        alpha_clip_hit_rate = fraction of users whose alpha is within 1e-4 of min_alpha
                              (0.1) or max_alpha (0.95) — CONCERNS.md clip-floor diagnostic.
        """
        if not getattr(model, "_per_user_alpha_enabled", False):
            return None
        logit_alpha = model._logit_alpha
        if logit_alpha is None:
            return None
        with torch.no_grad():
            alphas = torch.sigmoid(logit_alpha.weight).flatten().cpu().numpy()
        # Clip range from HierarchicalConditionalAlphaConfig defaults
        min_alpha, max_alpha = 0.1, 0.95
        epsilon = 1e-4
        clip_hits = np.sum(
            (np.abs(alphas - min_alpha) < epsilon) | (np.abs(alphas - max_alpha) < epsilon)
        )
        return {
            "alpha_mean": float(np.mean(alphas)),
            "alpha_std": float(np.std(alphas)),
            "alpha_p25": float(np.percentile(alphas, 25)),
            "alpha_p50": float(np.percentile(alphas, 50)),
            "alpha_p75": float(np.percentile(alphas, 75)),
            "alpha_clip_hit_rate": float(clip_hits / len(alphas)) if len(alphas) else 0.0,
        }
    ```

    Step 3 — Rewrite the `@app.train()` handler body. CRITICAL: the ADP-02 ordering fix is this step's primary purpose. Apply the reordered sequence:

    1. `mode = str(context.run_config.get("mode", "cross_silo_legacy"))`; `profile = resolve_mode_defaults(mode)`; `overrides = log_mode_and_overrides(mode, profile, context.run_config)`.
    2. `run_seed = int(context.run_config.get("run-seed", profile.run_seed))`; `round_num` from Flower message config; `partition_id = int(context.node_config["partition-id"])`.
    3. Load partition data via dataset.load_data (3-tuple return: trainloader, testloader-or-_, user_stats). Measure `num_users_in_client` (under benchmark mode this is 1).
    4. **`assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)`** (ADP-04).
    5. Read Phase-4 signature config from run_config: `alpha_method`, `fusion_type`, `mlp_hidden_dims` (as joined string), `per_user_alpha_enabled`, `item_perturbation_enabled`, `contrastive_lambda`. Default to profile values.
    6. Build `run_id = str(context.run_config.get("run-id", "")) or <generated-placeholder>`.
    7. Construct bare model (Xavier init) via get_model.
    8. Load GLOBAL params from msg.content["arrays"].to_torch_state_dict() → `model.set_global_parameters(state)`.
    9. **ADP-02 FIX — enable_* BEFORE load**. Under benchmark_cross_device, call enable_per_user_alpha + enable_item_perturbation UNCONDITIONALLY (CONTEXT D-03). In ablation mode, honor the run-config flags:
       ```python
       is_benchmark_mode = (mode == "benchmark_cross_device")
       # D-03: unconditional in benchmark mode; run-config override is ablation-only.
       effective_per_user_alpha = bool(
           context.run_config.get("enable-per-user-alpha", profile.enable_per_user_alpha if is_benchmark_mode else False)
       ) if not is_benchmark_mode else True
       effective_item_perturbation = bool(
           context.run_config.get("enable-item-perturbation", profile.enable_item_perturbation if is_benchmark_mode else False)
       ) if not is_benchmark_mode else True
       # Exception: explicit False in run_config DOES disable (ablation override).
       if context.run_config.get("enable-per-user-alpha") is False:
           effective_per_user_alpha = False
       if context.run_config.get("enable-item-perturbation") is False:
           effective_item_perturbation = False

       if effective_per_user_alpha and hasattr(model, "enable_per_user_alpha"):
           per_user_alphas = compute_per_user_alpha(user_stats, alpha_config, hc_config)
           model.enable_per_user_alpha(num_users=model.num_users, init_alphas=per_user_alphas)
       if effective_item_perturbation and hasattr(model, "enable_item_perturbation"):
           reg_lambda = float(context.run_config.get("item-perturbation-reg", 0.01))
           model.enable_item_perturbation(reg_lambda=reg_lambda)
       ```
    10. Build signature_v2 dict with all 12 keys (run_id + method + num_users + num_items + dim + split_hash + alpha_method + fusion_type + mlp_hidden_dims + effective_per_user_alpha + effective_item_perturbation + contrastive_lambda).
    11. `reuse_cache_flag = bool(context.run_config.get("reuse-cache", False))`.
    12. Cache-miss probe for D-13 cold-round signal: `is_cold_round = not _cache_pt_path_exists(partition_id, run_id, reuse_cache_flag, signature_v2)` (helper that reuses _cache_dir_for_run logic).
    13. **NOW** call `_load_local_user_state(...)`; if not None, `model.set_local_parameters(state, strict=False)`. Because enable_* ran at step 9, `_LOCAL_PARAMS` now includes `_logit_alpha.weight` + `_item_perturbation.weight`, so the load restores them (the bug is fixed).
    14. Call `model.set_alpha(client_alpha)` and `model.set_global_prototype(tensor)` as the existing code does (preserve D-18).
    15. Build `rng_train = np_rng(run_seed, partition_id, round_num, "train_neg")`; load `exclude_items = bundle["exclusion"].for_user(partition_id).tolist()`.
    16. Call train dispatcher: `task.train(..., run_seed=run_seed, user_idx=partition_id, round_num=round_num, exclude_items=exclude_items, rng=rng_train, is_cold_round=is_cold_round, contrastive_lambda=contrastive_lambda, contrastive_tau=contrastive_tau, ...)`.
    17. **AFTER training completes**, compute `alpha_diagnostics = _compute_alpha_diagnostics(model)` (D-16); compute `user_prototype = model.compute_user_prototype().cpu().numpy().tolist()` (preserve existing path).
    18. Save single-file local state: `_save_local_user_state(partition_id=partition_id, state_dict=model.get_local_parameters(), run_id=run_id, reuse_cache=reuse_cache_flag, signature=signature_v2)`.
    19. Build GLOBAL params ArrayRecord from `model.get_global_parameters()`.
    20. Build `FitMetricsContract(train_loss=..., num_positives=..., num_training_examples=..., round_num=round_num, partition_id=partition_id).to_dict()`; `validate_fit_metrics(payload)`. Route alpha_diagnostics via a SEPARATE MetricRecord keyed "alpha_diagnostics" (or embed as sub-dict in the reply content if the Flower Message RecordDict API supports it — confirm by reading the foundation fit_metrics source; if contract strictly rejects extras, use sibling MetricRecord). Route user_prototype via the existing `USER_PROTOTYPE_KEY = "user_prototype"` metric (unchanged from pre-Phase-4 code).

    Step 4 — Rewrite the `@app.evaluate()` handler body. Follow Phase-3 pattern PLUS the enable-before-load ordering applies here too if the evaluate handler reloads the model (typically yes, because Flower's evaluate message is a fresh handler invocation):

    1. **First check** `config_record = msg.content.get("train_config", {})`; if `bool(config_record.get("discover_only", False))` → short-circuit:
       ```python
       partition_id = int(context.node_config["partition-id"])
       payload = EvaluateMetricsContract(
           hit_count_overall_at10=0, ndcg_sum_overall_at10=0.0, evaluated_users=0,
           partition_id=partition_id,
       ).to_dict()
       validate_evaluate_metrics(payload)
       reply_content = RecordDict({"eval_metrics": MetricRecord(payload)})
       return Message.reply_to(msg, content=reply_content)
       ```
       **NO model load, NO data load, NO evaluation.**
    2. Otherwise: mode resolve → one-user assert → `assert get_primary_evaluator(mode) == "sampled_loo_99"`. Then run the enable-before-load ordering IDENTICAL to @app.train step 9 (so the evaluate handler also restores cached _logit_alpha / _item_perturbation). Then set global params, call `evaluate_ranking_sampled(..., run_seed=run_seed, user_idx=partition_id, round_num=round_num, exclude_items=...)`.
    3. Compute per-user-group sufficient stats via `_classify_partition_user_group(bundle, partition_id)`; route hit_count/ndcg_sum/evaluated_users into the matching bucket; other two groups get explicit zeros.
    4. Build `EvaluateMetricsContract(...)` with per-group fields + `partition_id`; validate; return.

    Step 5 — D-18 preserve: the pre-existing `get_device` + `_device_cache` pattern; any existing `save_local_user_embeddings` / `load_local_user_embeddings` with a different path layout is REPLACED by the Phase-4 helpers above (this plan owns the D-01..D-04 migration).

    Step 6 — Create federated-adaptive-personalized-cf/tests/test_client_assertion.py with 5 tests (clone from Phase-3 federated-personalized-cf/tests/test_client_assertion.py; adjust test 4 to exercise the alpha-diagnostic sidecar dict):

    Step 7 — Create federated-adaptive-personalized-cf/tests/test_embedding_cache_manifest_v2.py with 5 tests listed in <behavior>. Use `tmp_path` fixture; import the module-level helpers (`_save_local_user_state`, `_load_local_user_state`, `_cache_dir_for_run`, `_signature_fields_v2`) directly; monkeypatch the cache base if needed:
    ```python
    import pytest
    from pathlib import Path
    import torch
    from federated_adaptive_personalized_cf.client_app import (
        _save_local_user_state, _load_local_user_state,
        _cache_dir_for_run, _signature_fields_v2,
    )

    def _build_sig(**overrides):
        base = dict(
            run_id="r1", method="dual", num_users=6040, num_items=3706, dim=64,
            split_hash="abc123",
            alpha_method="hierarchical_conditional", fusion_type="concat",
            mlp_hidden_dims="512,256,128",
            per_user_alpha_enabled=True, item_perturbation_enabled=True,
            contrastive_lambda=0.1,
        )
        base.update(overrides)
        return _signature_fields_v2(**base)

    def test_manifest_v2_sidecar_written_and_loaded(tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sig = _build_sig()
        state = OrderedDict([
            ("user_embeddings.weight", torch.zeros(6040, 64)),
            ("user_bias.weight", torch.zeros(6040, 1)),
            ("_logit_alpha.weight", torch.zeros(6040, 1)),
            ("_item_perturbation.weight", torch.zeros(3706, 64)),
        ])
        _save_local_user_state(partition_id=0, state_dict=state,
                               run_id="r1", reuse_cache=False, signature=sig)
        cache_dir = tmp_path / ".embedding_cache" / "r1"
        assert (cache_dir / "manifest.json").exists()
        assert (cache_dir / "partition_0.pt").exists()
        import json
        m = json.loads((cache_dir / "manifest.json").read_text())
        assert m["schema_version"] == 2
        for field in ("alpha_method", "fusion_type", "mlp_hidden_dims",
                      "per_user_alpha_enabled", "item_perturbation_enabled",
                      "contrastive_lambda"):
            assert field in m
        loaded = _load_local_user_state(partition_id=0, run_id="r1",
                                         reuse_cache=False, signature=sig)
        assert set(loaded.keys()) == set(state.keys())

    def test_manifest_v2_mismatch_raises_runtime_error(tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sig_a = _build_sig(alpha_method="hierarchical_conditional")
        sig_b = _build_sig(alpha_method="multi_factor")
        state = OrderedDict([("user_embeddings.weight", torch.zeros(6040, 64)),
                             ("user_bias.weight", torch.zeros(6040, 1)),
                             ("_logit_alpha.weight", torch.zeros(6040, 1)),
                             ("_item_perturbation.weight", torch.zeros(3706, 64))])
        _save_local_user_state(partition_id=0, state_dict=state,
                               run_id="r1", reuse_cache=False, signature=sig_a)
        with pytest.raises(RuntimeError, match="alpha_method"):
            _load_local_user_state(partition_id=0, run_id="r1",
                                    reuse_cache=False, signature=sig_b)
        # rm -rf hint must be present in the error
        with pytest.raises(RuntimeError, match="rm -rf .embedding_cache"):
            _load_local_user_state(partition_id=0, run_id="r1",
                                    reuse_cache=False, signature=sig_b)

    def test_reuse_cache_sig_path_v2(tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sig = _build_sig(run_id="anything_arbitrary")
        dir1 = _cache_dir_for_run(run_id="run_A", reuse_cache=True, signature=sig)
        dir2 = _cache_dir_for_run(run_id="run_B", reuse_cache=True, signature=sig)
        assert dir1.name.startswith("sig_")
        assert dir2.name.startswith("sig_")
        assert dir1 == dir2

    def test_extended_local_key_payload_shape(tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sig = _build_sig()
        state = OrderedDict([
            ("user_embeddings.weight", torch.randn(6040, 64)),
            ("user_bias.weight", torch.randn(6040, 1)),
            ("personal_mlp.0.weight", torch.randn(512, 64)),
            ("personal_mlp.0.bias", torch.randn(512)),
            ("fusion_layer.weight", torch.randn(1, 2)),
            ("fusion_layer.bias", torch.randn(1)),
            ("_logit_alpha.weight", torch.randn(6040, 1)),
            ("_item_perturbation.weight", torch.randn(3706, 64)),
        ])
        _save_local_user_state(partition_id=0, state_dict=state,
                               run_id="r1", reuse_cache=False, signature=sig)
        loaded = _load_local_user_state(partition_id=0, run_id="r1",
                                         reuse_cache=False, signature=sig)
        for k in state:
            assert torch.equal(state[k], loaded[k]), f"round-trip failure on {k}"

    def test_schema_v1_manifest_raises_when_loading_under_v2(tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cache_dir = tmp_path / ".embedding_cache" / "r1"
        cache_dir.mkdir(parents=True)
        # Seed a v1 manifest (6 fields, schema_version=1)
        import json
        (cache_dir / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "run_id": "r1", "method": "dual",
            "num_users": 6040, "num_items": 3706, "dim": 64, "split_hash": "abc",
        }))
        (cache_dir / "partition_0.pt").write_bytes(b"\x00")
        sig = _build_sig()
        with pytest.raises(RuntimeError, match="schema_version"):
            _load_local_user_state(partition_id=0, run_id="r1",
                                    reuse_cache=False, signature=sig)
    ```

    Step 8 — Verify: `cd federated-adaptive-personalized-cf && pytest tests/test_client_assertion.py tests/test_embedding_cache_manifest_v2.py -v` → 10 passed. Full suite: `pytest tests/ -v` → 28+ passed (Plan 01=10 + Plan 02=9 + Plan 03 Task 1=4 + Plan 03 Task 2=10 = 33).

    Step 9 — Cross-file RNG regression check: `grep -rnE "^import random$|random\\.seed\\(|random\\.sample\\(" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` → 0 matches (aside from docstring false positives, which are acceptable; the test uses ast-based filtering to exempt them).

    Step 10 — Commit (--no-verify):
    ```
    git add federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py \
            federated-adaptive-personalized-cf/tests/test_client_assertion.py \
            federated-adaptive-personalized-cf/tests/test_embedding_cache_manifest_v2.py
    git commit --no-verify -m "feat(04-03): client_app ADP-02 enable-before-load + schema-v2 cache + alpha diagnostics (ADP-02, ADP-04, ADP-06)"
    ```
  </action>
  <acceptance_criteria>
    - `grep -c "from fedrec_foundation.mode import" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns 1
    - `grep -c "assert_benchmark_one_user_per_client" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns at least 2 (train + evaluate handlers)
    - `grep -c "get_primary_evaluator" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns at least 1
    - `grep -c "FitMetricsContract" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns at least 2
    - `grep -c "EvaluateMetricsContract" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns at least 3
    - `grep -c "partition_id=partition_id" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns at least 2
    - `grep -c "discover_only" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns at least 1
    - `grep -c "schema_version" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns at least 2 (v2 signature build + load mismatch check)
    - `grep -c '"schema_version": 2' federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns at least 1
    - `grep -c "atomic_write_json" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns at least 1
    - `grep -cE "reuse_cache|reuse-cache" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns at least 2 (D-09 opt-in)
    - `grep -cE "rm -rf" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns at least 1 (D-04 literal hint in error message)
    - `grep -c "def _signature_fields_v2" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns 1
    - `grep -c "def _save_local_user_state" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns 1
    - `grep -c "def _load_local_user_state" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns 1
    - `grep -c "def _compute_alpha_diagnostics" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns 1
    - `grep -c "alpha_clip_hit_rate" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns at least 1 (D-16 diagnostic field)
    - `grep -c "enable_per_user_alpha" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns at least 2 (call site + ablation-flag branch)
    - `grep -c "enable_item_perturbation" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns at least 2
    - **ADP-02 FIX ORDERING CHECK** — source-level proximity guard: `python -c "src=open('federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py').read(); i=src.index('enable_per_user_alpha'); j=src.index('_load_local_user_state'); assert i < j, 'ADP-02 violated: enable_per_user_alpha must come BEFORE _load_local_user_state in source'; print('ordering ok')"` prints `ordering ok`
    - `grep -cE "^import random$|random\\.seed\\(|random\\.sample\\(" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns 0
    - `cd federated-adaptive-personalized-cf && pytest tests/test_client_assertion.py -v` exits 0 with "5 passed"
    - `cd federated-adaptive-personalized-cf && pytest tests/test_embedding_cache_manifest_v2.py -v` exits 0 with "5 passed"
    - `cd federated-adaptive-personalized-cf && pytest tests/ -v` exits 0 with at least 33 tests passing overall (Plan 01+02+03 accumulated)
    - D-18 scope: `git diff --stat federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/ federated-adaptive-personalized-cf/pyproject.toml` returns empty after commit
  </acceptance_criteria>
  <done>client_app.py implements the full Phase-4 cross-device contract with ADP-02 enable-before-load ordering fix (primary bug fix), mode resolver, benchmark-mode one-user assertion, discover_only short-circuit, strict FitMetrics/EvaluateMetrics payloads with optional partition_id, D-16 alpha diagnostics sidecar, per-group sufficient-stat routing, and schema_version=2 manifest-sidecar cache layout with loud D-04 mismatch (per-field delta + rm -rf hint) + D-09 reuse-cache opt-in. 10 new GREEN tests across 2 new test files (5 client_assertion + 5 embedding_cache_manifest_v2).</done>
</task>

</tasks>

<verification>
- `cd federated-adaptive-personalized-cf && pytest tests/ -v` exits 0 with at least 33 tests passing (Plan 01=10 + Plan 02=9 + Plan 03=14)
- `grep -rnE "^import random$|random\\.seed\\(|random\\.sample\\(" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` returns 0 matches
- ADP-02 ordering source proximity check: `python -c "src=open('federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py').read(); i=src.index('enable_per_user_alpha'); j=src.index('_load_local_user_state'); assert i < j; print('ok')"` prints `ok`
- `python -c "from fedrec_foundation.fit_metrics import FitMetricsContract, validate_fit_metrics; c = FitMetricsContract(train_loss=0.1, num_positives=1, num_training_examples=5, partition_id=42); d = c.to_dict(); assert d['partition_id'] == 42; validate_fit_metrics(d); print('ok')"` prints `ok`
- D-18 scope: `git diff --stat` across strategy.py / dataset.py / server_app.py / models/ / pyproject.toml returns empty after this plan's commits (Plans 01/02/05 own those)
</verification>

<success_criteria>
- **ADP-02 observable (primary Phase-4 bug fix)**: source-level proximity guard proves enable_per_user_alpha + enable_item_perturbation appear BEFORE _load_local_user_state in client_app.py. test_dual_model.py::test_enable_before_load_restores_cached_alpha (from Plan 01) already proves the model-level behavior; this plan wires it into the live handler.
- ADP-04 observable end-to-end: a partition with >1 user raises AssertionError under benchmark_cross_device mode before any training or evaluation happens.
- ADP-05 observable: ExclusionTable.for_user(partition_id) is merged into user_rated_items before train-neg sampling AND into negative_candidates before eval-neg sampling; the held-out test positive can never be drawn as either.
- ADP-06 (client half) observable: FND-06 np_rng tuples threaded through DataLoader + negative sampling + evaluator; FitMetricsContract + EvaluateMetricsContract with partition_id populated; schema_version=2 manifest-sidecar cache with 12 signature fields and D-04 loud-mismatch RuntimeError; D-09 reuse-cache opt-in.
- D-13/D-14 observable: test_cold_round_sets_alpha_zero_and_skips_contrastive proves the cold-round branch sets α=0 and skips contrastive loss, then restores alpha at the end of training.
- D-16 observable: FitMetricsContract train response carries alpha_mean/alpha_std/alpha_p25/alpha_p50/alpha_p75/alpha_clip_hit_rate sidecar dict when enable-per-user-alpha=true; server_app.py (Plan 05) consumes it for W&B logging.
- D-24 observable: _snapshot_non_user_rows/_restore_non_user_rows wrap optimizer.step in task.py; user_embeddings + user_bias + _logit_alpha are protected; _item_perturbation is NOT protected (item-indexed).
</success_criteria>

<output>
After completion, create `.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-03-SUMMARY.md` with: file list (2 modified + 3 created), decisions made (ADP-02 ordering implementation details — where in the handler the enable_* calls land relative to _load_local_user_state; how alpha_diagnostics was routed given the D-21 strict contract — MetricRecord sidecar vs contract extension), deviations (any auto-fixes), test counts (~14 GREEN in this plan, total suite ~33 GREEN), commit SHAs, ADP-02/04/05/06(client) closure notes, Plan 05 readiness confirmation (strategy + client-side contract both in place; server_app.py can now drop AdaptiveSplitFedAvg + discovery round + partition-space sampling + best_prototype snapshot call).
</output>
</content>
</invoke>