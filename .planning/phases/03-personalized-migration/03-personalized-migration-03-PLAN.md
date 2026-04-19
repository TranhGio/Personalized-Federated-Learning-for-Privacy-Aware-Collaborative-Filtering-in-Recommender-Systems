---
phase: 03-personalized-migration
plan: 03
type: execute
subsystem: infra
tags: [client-app, task, rng-threading, exclusion-set, benchmark-assertion, fit-metrics-contract, evaluate-metrics-contract, per-group-metrics, embedding-cache-manifest, manifest-sidecar, schema-version, reuse-cache, content-hash, psn-02, psn-03, psn-04, psn-05, psn-06, d-02, d-04, d-05, d-06, d-07, d-08, d-09, d-10, d-11, d-22, wave-2]
wave: 2
depends_on: [03-personalized-migration-01, 03-personalized-migration-02]
files_modified:
  - federated-personalized-cf/federated_personalized_cf/client_app.py
  - federated-personalized-cf/federated_personalized_cf/task.py
  - federated-personalized-cf/tests/test_client_assertion.py
  - federated-personalized-cf/tests/test_task_rng.py
  - federated-personalized-cf/tests/test_embedding_cache_manifest.py
autonomous: true
requirements: [PSN-02, PSN-03, PSN-05, PSN-06]

must_haves:
  truths:
    - "client_app.py @app.train and @app.evaluate handlers resolve a ModeProfile via fedrec_foundation.mode.resolve_mode_defaults(mode), collect overrides via log_mode_and_overrides, and call assert_benchmark_one_user_per_client BEFORE any training or ranking happens. Under mode=\"benchmark_cross_device\" a partition with >1 user raises AssertionError (PSN-02)."
    - "Both handlers build wire payloads via FitMetricsContract.to_dict() / EvaluateMetricsContract.to_dict() and validate before send (D-21 strict contract). Optional partition_id field is populated on both contracts (G-03-01 carry-forward from Phase 2 Plan 05)."
    - "client_app.py @app.evaluate short-circuits on discover_only=True ConfigRecord input: returns minimal zero-suffstats payload + partition_id without any model load or evaluation (G-03-01 discovery-round handshake)."
    - ".embedding_cache path layout implements D-04..D-07 manifest-sidecar: (a) default path is .embedding_cache/{run_id}/partition_{pid}.pt with sibling .embedding_cache/{run_id}/manifest.json; (b) when run_config[\"reuse-cache\"]=true (D-09), path becomes .embedding_cache/sig_{sha256(signature_fields)[:16]}/partition_{pid}.pt with sibling manifest.json; (c) manifest.json is written atomically via fedrec_foundation.atomic.atomic_write_json; (d) schema_version=1 with 6 signature fields (run_id, method, num_users, num_items, dim, split_hash)."
    - "D-05 loud mismatch behavior: on cache load, manifest.json is read first; if ANY signature field diverges from the current run's values, raise RuntimeError with per-field delta + literal `rm -rf .embedding_cache/{run_id}/` hint. No auto-deletion."
    - "D-10 single-row disk payload: .pt contains OrderedDict({'local_user_row': tensor(d,), 'local_user_bias': tensor(1,)}) — never the old (num_users, d) blob. Shape verified at save+load."
    - "task.py: FND-06 RNG wired end-to-end — train_bpr_mf / train_basic_mf / evaluate_ranking_sampled accept run_seed, user_idx, round_num, exclude_items, rng kwargs; zero `random.seed(`, zero `random.sample(`, zero module-level `import random` in task.py AND client_app.py."
    - "PSN-03 / FND-03 exclusion: ExclusionTable.for_user(partition_id) union folds into user_rated_items before train-negative sampling AND into negative_candidates before eval-negative sampling; the held-out test positive is never drawn as either a training or eval negative."
    - "PSN-06 on disk: single-row local state matches the model's _LOCAL_PARAMS contract — never a ghost table."
    - "5 new pytest tests added (5 test_client_assertion + 4 test_task_rng + 4 test_embedding_cache_manifest = 13 GREEN) bringing federated-personalized-cf suite from 14 (after Plan 01+02) to ~24 GREEN."
  artifacts:
    - path: "federated-personalized-cf/federated_personalized_cf/client_app.py"
      provides: "Cross-device @app.train + @app.evaluate with mode resolver, one-user assert, D-04..D-10 cache-manifest layout, strict contract payloads with partition_id, discover_only short-circuit"
    - path: "federated-personalized-cf/federated_personalized_cf/task.py"
      provides: "train_bpr_mf / train_basic_mf / evaluate_ranking_sampled with 5 FND-06 kwargs, exclusion-aware negative sampling, _sample_negatives_seeded helper"
    - path: "federated-personalized-cf/tests/test_client_assertion.py"
      provides: "5 GREEN tests for PSN-02 (one-user assert), D-10 override bypass, FND-04 primary evaluator, D-21 FitMetrics/EvaluateMetrics payload shape, partition_id on contract"
    - path: "federated-personalized-cf/tests/test_task_rng.py"
      provides: "4 GREEN tests for BSL-05-style RNG strip (task.py + client_app.py), BSL-03-style exclusion in training negatives, evaluate_ranking_sampled signature contract"
    - path: "federated-personalized-cf/tests/test_embedding_cache_manifest.py"
      provides: "4 GREEN tests for D-04 sidecar layout, D-05 loud-mismatch RuntimeError with rm -rf hint, D-09 reuse-cache sig_{hash} path, D-10 single-row payload shape"
  key_links:
    - from: "federated-personalized-cf/federated_personalized_cf/client_app.py"
      to: "fedrec_foundation.mode.assert_benchmark_one_user_per_client"
      via: "called before any training; raises AssertionError if num_users_in_partition > 1 under benchmark mode"
      pattern: "assert_benchmark_one_user_per_client"
    - from: "federated-personalized-cf/federated_personalized_cf/client_app.py"
      to: "fedrec_foundation.atomic.atomic_write_json"
      via: "manifest.json sidecar writer uses foundation's atomic tempfile + os.replace"
      pattern: "atomic_write_json"
    - from: "federated-personalized-cf/federated_personalized_cf/client_app.py"
      to: "fedrec_foundation.fit_metrics.EvaluateMetricsContract"
      via: "optional partition_id field populated on every contract build — G-03-01 carry-forward"
      pattern: "partition_id=partition_id"
---

<objective>
Wire client_app.py + task.py into the cross-device contract — mirroring Phase 2 Plan 03 surgical migration — AND land the PSN-05 run-scoped manifest-sidecar embedding cache (D-04..D-10). Benchmark-mode one-user assertion (PSN-02), FND-03 exclusion-set threading (PSN-03), FND-06 RNG wiring, strict FitMetrics/EvaluateMetrics payloads with optional partition_id (G-03-01 carry-forward), discover_only short-circuit for the discovery-round handshake, and D-11 Xavier-uniform first-use init for the single local user row.

Purpose: Closes PSN-02, PSN-03, PSN-05 and completes the client-half of PSN-04 and PSN-06. Plan 04 can drop the PersonalizedSplitFedAvg strategy (from Plan 01) into server_app.py without further client-side changes, and the discovery-round handshake will work on the first try.

Output:
- federated-personalized-cf/federated_personalized_cf/client_app.py (migrated)
- federated-personalized-cf/federated_personalized_cf/task.py (RNG + exclusion + training dispatcher)
- 3 new test files with ~13 GREEN tests
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/03-personalized-migration/03-CONTEXT.md
@.planning/phases/02-baseline-migration/02-baseline-migration-03-SUMMARY.md
@.planning/phases/02-baseline-migration/02-baseline-migration-05-SUMMARY.md
@.planning/phases/03-personalized-migration/03-personalized-migration-01-PLAN.md
@.planning/phases/03-personalized-migration/03-personalized-migration-02-PLAN.md

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
def py_rng(run_seed: int, user_idx: int, round_num: int, purpose: str) -> random.Random: ...

# fedrec_foundation.fit_metrics (post Phase 2 Plan 05)
@dataclass
class FitMetricsContract:
    train_loss: float
    num_positives: int
    num_training_examples: int
    round_num: Optional[int] = None
    partition_id: Optional[int] = None      # G-03-01 carry-forward
    hit_count_overall_at10: Optional[int] = None
    ndcg_sum_overall_at10: Optional[float] = None
    evaluated_users: Optional[int] = None
    # ... 9 more per-group fields
    def to_dict(self) -> Dict[str, Any]: ...  # drops None
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
    partition_id: Optional[int] = None       # G-03-01 carry-forward
    # 9 per-group fields
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvaluateMetricsContract": ...

def validate_evaluate_metrics(payload: Dict[str, Any]) -> None: ...

# fedrec_foundation.atomic
def atomic_write_json(path: Path, data: Dict[str, Any]) -> None: ...

# fedrec_foundation.evaluator
def get_primary_evaluator(mode: str) -> str: ...   # always returns "sampled_loo_99" for known modes

# fedrec_foundation.user_groups
def classify_user_group(n_interactions: int) -> str: ...   # returns "sparse" | "medium" | "dense"
```

<!-- This plan's models come from Plan 01 (already committed when this plan runs in Wave 2) -->
```python
# federated_personalized_cf.models.bpr_mf (post Plan 01)
class BPRMF(nn.Module):
    _LOCAL_PARAMS = ('local_user_row', 'local_user_bias')   # D-03
    _GLOBAL_PARAMS = ('item_embeddings.weight', 'item_bias.weight', 'global_bias')
    # self.local_user_row: nn.Parameter of shape (embedding_dim,)
    # self.local_user_bias: nn.Parameter of shape (1,)
    def forward(self, item_ids: torch.Tensor) -> torch.Tensor: ...
    def get_local_parameters(self) -> OrderedDict[str, torch.Tensor]: ...
    def set_local_parameters(self, state: Dict[str, torch.Tensor], strict: bool = False) -> Tuple[List[str], List[str]]: ...
    def get_global_parameters(self) -> OrderedDict[str, torch.Tensor]: ...
    def set_global_parameters(self, state: Dict[str, torch.Tensor]) -> None: ...
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: task.py migration — FND-06 RNG + FND-03 exclusion + _sample_negatives_seeded (PSN-03, PSN-06 training-side, BSL-05-style RNG strip)</name>
  <files>federated-personalized-cf/federated_personalized_cf/task.py, federated-personalized-cf/tests/test_task_rng.py</files>
  <read_first>
    - federated-personalized-cf/federated_personalized_cf/task.py (ENTIRE FILE — inventory all `random.seed`, `random.sample`, `import random`, and the existing train_bpr_mf / train_basic_mf / evaluate_ranking_sampled signatures so we can extend them additively)
    - federated-baseline-cf/federated_baseline_cf/task.py (post-Plan-03 TEMPLATE — the _sample_negatives_seeded helper, the 5-kwarg extension pattern, the _classify_partition_user_group helper; clone with minimum modification for the single-row model)
    - federated-baseline-cf/tests/test_task_rng.py (4-test template: test_random_seed_calls_stripped, test_train_negatives_exclude_test_positive, test_evaluate_ranking_sampled_accepts_rng_signature, test_gradient_mask_zeros_non_user_rows — adjust the gradient-mask test for the single-row model OR drop/replace it since the single-row model does not need gradient masking at all)
    - federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py (POST-PLAN-01 — observe the single-row local_user_row contract so train_bpr_mf updates local_user_row directly via optimizer.step() with no masking needed)
    - .planning/phases/02-baseline-migration/02-baseline-migration-03-SUMMARY.md (D-24 Adam-weight-decay-gradient-isolation detail — NOTE: for Phase 3, since local_user_row IS the user's model, D-24 gradient masking is NOT needed — there is no ghost table to protect. The single-row refactor EATS the D-24 problem.)
  </read_first>
  <behavior>
    Tests to write FIRST (RED), then implement:
    - test_random_seed_calls_stripped: reads BOTH task.py AND client_app.py source files; asserts no matches for `random.seed(`, `random.sample(`, or module-level `^import random$` in the first 30 lines (cross-file regression guard — clone from Phase 2 Plan 03's test_random_seed_calls_stripped).
    - test_train_negatives_exclude_test_positive: constructs a 1-user BPRMF (embedding_dim=16), calls train_bpr_mf with exclude_items=[25] and item_id=25 intentionally a positive; trains 1 epoch; confirms no crash + local_user_row has moved from its initial Xavier value.
    - test_evaluate_ranking_sampled_accepts_rng_signature: introspects inspect.signature(evaluate_ranking_sampled) and asserts parameter names include run_seed, user_idx, round_num, exclude_items.
    - test_sample_negatives_seeded_deterministic: calls `_sample_negatives_seeded` twice with the SAME (user_rated_items, num_items, num_negatives, np_rng(42, 0, 1, 'train_neg')) and asserts identical output; calls with different seed and asserts different output.
  </behavior>
  <action>
    Step 1 — Add imports to task.py top:
    ```python
    from fedrec_foundation.rng import np_rng, torch_gen
    ```
    Remove any module-level `import random` if present. If any comments/docstrings mention the stdlib `random` API, re-word to avoid the literal substrings `random.seed(`, `random.sample(`, `import random` (the acceptance grep is plain regex — docstring mentions false-positive).

    Step 2 — Add 1 new module-level private helper (clone from baseline, simplified for single-row model):
    ```python
    def _sample_negatives_seeded(
        user_rated_items: Set[int],
        num_items: int,
        num_negatives: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Rejection-sampled negatives from an np.random.Generator instance.

        Distribution-equivalent to uniform sampling over
        range(num_items) \ user_rated_items, but deterministic under
        a given rng (FND-06). Used inside train_bpr_mf to replace the
        old model.sample_negatives(...) call that used process-global np.random.
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

    Step 3 — Extend `train_bpr_mf` with 5 new keyword-only parameters (all Optional[] with defaults so backward compat holds):
    ```python
    def train_bpr_mf(
        model, trainloader, *,
        run_seed: Optional[int] = None,
        user_idx: Optional[int] = None,
        round_num: Optional[int] = None,
        exclude_items: Optional[Iterable[int]] = None,
        rng: Optional[np.random.Generator] = None,
        **existing_kwargs,
    ) -> float:
        ...
    ```

    Inside the function:
    - If `rng is None and run_seed is not None and user_idx is not None`: `rng = np_rng(int(run_seed), int(user_idx), int(round_num or 0), "train_neg")`.
    - Build `user_rated_items: Set[int]` from the single-user training data (the client IS one user; no need for a per-user dict).
    - If `exclude_items is not None`: `user_rated_items |= set(int(x) for x in exclude_items)` — FND-03 guarantee.
    - Replace any existing `model.sample_negatives(...)` call with `_sample_negatives_seeded(user_rated_items, num_items, num_negatives, rng)` when `rng is not None`; fall back to the old path when `rng is None` (backward compat).
    - DO NOT apply D-24 gradient masking or snapshot/restore — the single-row model has no ghost rows to protect. The only local param being updated is local_user_row / local_user_bias. Comment this explicitly: `# D-24 not needed: single-row model collapses the ghost-table problem (Phase 3 D-01).`

    Step 4 — Mirror the same 5-kwarg extension on `train_basic_mf`. Apply the same exclude_items + rng wiring. D-24 masking NOT required (same reason).

    Step 5 — Extend `evaluate_ranking_sampled` with 4 new keyword-only parameters:
    ```python
    def evaluate_ranking_sampled(
        model, testloader, *,
        run_seed: Optional[int] = None,
        user_idx: Optional[int] = None,
        round_num: Optional[int] = None,
        exclude_items: Optional[Iterable[int]] = None,
        seed: int = 42,   # legacy param — now IGNORED; documented in docstring
        **existing_kwargs,
    ) -> Dict[str, float]:
    ```
    Strip any existing `random.seed(seed)` / `random.sample(...)` calls. Replace with:
    - `rng = np_rng(int(run_seed), int(user_idx), int(round_num or 0), "eval_neg")` when the four FND-06 params are provided.
    - `negative_items = rng.choice(negative_candidates, size=num_negatives, replace=False)`.
    - Fold `exclude_items` into `all_user_items` / `negative_candidates` before the choice.
    Docstring MUST explicitly note that the legacy `seed: int = 42` parameter is IGNORED.

    Step 6 — Update the `train` dispatcher (if present) to forward the 5 new kwargs to both underlying functions.

    Step 7 — Preserve verbatim (D-18): `load_data`, `get_model`, `test`, `compute_ndcg`, `compute_mrr`, any module-level `_dataset_cache` / `_item_popularity_cache` / `_device_cache` globals.

    Step 8 — Create federated-personalized-cf/tests/test_task_rng.py with the 4 tests listed in <behavior>. Use `inspect.signature` for the signature check; use direct source-file reads (with pytest skip-if-bundle-missing marker for the exclusion-in-training test).

    Step 9 — Verify: `cd federated-personalized-cf && pytest tests/test_task_rng.py -v` → 4 passed. Full new suite: `pytest tests/ -v` → ~18 passed.

    Step 10 — Commit (--no-verify; Wave-2 parallel rule):
    ```
    git add federated-personalized-cf/federated_personalized_cf/task.py \
            federated-personalized-cf/tests/test_task_rng.py
    git commit --no-verify -m "feat(03-03): task.py FND-06 RNG + FND-03 exclusion + _sample_negatives_seeded (PSN-03)"
    ```
  </action>
  <acceptance_criteria>
    - `grep -cE "^import random$" federated-personalized-cf/federated_personalized_cf/task.py` returns 0
    - `grep -cE "random\.seed\(" federated-personalized-cf/federated_personalized_cf/task.py` returns 0
    - `grep -cE "random\.sample\(" federated-personalized-cf/federated_personalized_cf/task.py` returns 0
    - `grep -c "from fedrec_foundation.rng import" federated-personalized-cf/federated_personalized_cf/task.py` returns 1
    - `grep -c "def _sample_negatives_seeded" federated-personalized-cf/federated_personalized_cf/task.py` returns 1
    - `grep -cE "run_seed" federated-personalized-cf/federated_personalized_cf/task.py` returns at least 4 (train_bpr_mf + train_basic_mf + evaluate_ranking_sampled + train dispatcher)
    - `grep -cE "exclude_items" federated-personalized-cf/federated_personalized_cf/task.py` returns at least 3
    - `cd federated-personalized-cf && pytest tests/test_task_rng.py -v` exits 0 with "4 passed"
    - `python -c "import inspect; from federated_personalized_cf.task import evaluate_ranking_sampled; p = inspect.signature(evaluate_ranking_sampled).parameters; assert 'run_seed' in p and 'user_idx' in p and 'round_num' in p and 'exclude_items' in p; print('ok')"` prints `ok`
    - `git diff --stat federated-personalized-cf/federated_personalized_cf/strategy.py federated-personalized-cf/federated_personalized_cf/dataset.py federated-personalized-cf/federated_personalized_cf/server_app.py federated-personalized-cf/federated_personalized_cf/models/ federated-personalized-cf/pyproject.toml` returns empty after commit (D-18 scope; Plans 01/02/04 own those)
  </acceptance_criteria>
  <done>task.py threads FND-06 RNG + FND-03 exclusion-set into both training loops and the sampled evaluator; _sample_negatives_seeded helper replaces process-global np.random.randint; stdlib random is eradicated from task.py; 4 GREEN tests guard the contract.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: client_app.py migration — mode resolver + one-user assert + strict contracts + manifest-sidecar cache (D-04..D-10, PSN-02, PSN-04 client half, PSN-05, PSN-06 disk shape)</name>
  <files>federated-personalized-cf/federated_personalized_cf/client_app.py, federated-personalized-cf/tests/test_client_assertion.py, federated-personalized-cf/tests/test_embedding_cache_manifest.py</files>
  <read_first>
    - federated-personalized-cf/federated_personalized_cf/client_app.py (ENTIRE FILE — preserve pre-existing WIP per D-18; need to see every save_local_user_embeddings / load_local_user_embeddings call to rewrite them for single-row manifest-sidecar)
    - federated-baseline-cf/federated_baseline_cf/client_app.py (CANONICAL TEMPLATE — mode resolver, one-user assert, strict FitMetrics/EvaluateMetrics payloads, per-group sufficient-stat routing, discover_only short-circuit from Plan 05)
    - federated-baseline-cf/tests/test_client_assertion.py (5-test template to clone)
    - scripts/foundation/fedrec_foundation/mode.py (assert_benchmark_one_user_per_client signature)
    - scripts/foundation/fedrec_foundation/atomic.py (atomic_write_json signature)
    - scripts/foundation/fedrec_foundation/fit_metrics.py (FitMetricsContract, EvaluateMetricsContract with optional partition_id — Phase 2 Plan 05 extension)
    - scripts/foundation/fedrec_foundation/user_groups.py (classify_user_group)
    - .planning/phases/03-personalized-migration/03-CONTEXT.md (D-04..D-11 cache decisions verbatim; D-13 cold-start counter is Plan 04 — this plan only cares about cache-load-existence check, not counting)
    - .planning/phases/02-baseline-migration/02-baseline-migration-05-SUMMARY.md (G-03-01 discover_only + partition_id pattern)
  </read_first>
  <behavior>
    Tests to write FIRST (RED), then implement:
    - test_benchmark_mode_asserts_one_user: `assert_benchmark_one_user_per_client(profile, 3, {})` raises AssertionError containing "one user"; `(profile, 1, {})` returns without raising.
    - test_benchmark_mode_override_bypass: `(profile, 50, {"num_supernodes": 10})` returns without raising (D-10 override path).
    - test_get_primary_evaluator_selects_sampled_loo_99: calling get_primary_evaluator for each known mode returns "sampled_loo_99".
    - test_fit_metrics_contract_payload_shape: builds FitMetricsContract(train_loss=0.5, num_positives=1, num_training_examples=5, partition_id=42, hit_count_overall_at10=1, ndcg_sum_overall_at10=0.1, evaluated_users=1, hit_count_sparse_at10=1, ndcg_sum_sparse_at10=0.1, evaluated_users_sparse=1); asserts to_dict() contains 'partition_id' == 42 + all populated per-group keys; validate_fit_metrics passes.
    - test_evaluate_metrics_contract_payload_shape_with_partition_id: similar but for EvaluateMetricsContract; partition_id must appear in to_dict() output; validate_evaluate_metrics passes; a payload with an unknown key ("train_loss") fails with ValueError.
    - test_manifest_sidecar_written_and_loaded: simulate a save: call helper save_local_user_state(pid=0, state_dict, run_id="r1", num_users=6040, num_items=3706, dim=64, split_hash="abc123"); verify .embedding_cache/r1/manifest.json exists with schema_version=1 and exact signature dict; verify .embedding_cache/r1/partition_0.pt exists and contains exactly 2 keys ('local_user_row' of shape (64,), 'local_user_bias' of shape (1,)).
    - test_manifest_mismatch_raises_runtime_error: seed a cache with dim=64; attempt load with dim=128; assert RuntimeError raised; assert error message contains both "dim" AND "rm -rf .embedding_cache/" with the specific run_id path.
    - test_reuse_cache_sig_path: call the same save/load helper with reuse_cache=True; assert path resolves to `.embedding_cache/sig_<16-hex-chars>/` instead of `.embedding_cache/<run_id>/`; two runs with identical signature hash to the same sig path.
    - test_discover_only_short_circuits_evaluate: build a ConfigRecord with discover_only=True; call @app.evaluate handler; assert returned payload contains partition_id and zero sufficient stats; assert NO model load happened (guard via a module-level counter or a spy).
  </behavior>
  <action>
    Step 1 — Add imports to client_app.py top:
    ```python
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
    from flwr.common import ConfigRecord  # needed for discover_only short-circuit
    ```

    Step 2 — Add 4 new module-level helpers (clone shape from baseline where relevant):

    2a. `_classify_partition_user_group(bundle, partition_id: int) -> str`:
    ```python
    def _classify_partition_user_group(bundle: Dict[str, Any], partition_id: int) -> str:
        """Read per-user group classification from the foundation split_manifest train_user_stats."""
        stats_map = getattr(bundle["split_manifest"], "train_user_stats", None)
        if stats_map is None:
            return "medium"
        entry = stats_map.get(int(partition_id))
        if entry is None:
            return "medium"
        return getattr(entry, "user_group", None) or classify_user_group(int(getattr(entry, "n_interactions", 0)))
    ```

    2b. `_signature_fields(*, run_id: str, method: str, num_users: int, num_items: int, dim: int, split_hash: str) -> Dict[str, Any]`:
    ```python
    def _signature_fields(*, run_id, method, num_users, num_items, dim, split_hash):
        return {
            "schema_version": 1,
            "run_id": str(run_id),
            "method": str(method),
            "num_users": int(num_users),
            "num_items": int(num_items),
            "dim": int(dim),
            "split_hash": str(split_hash),
        }
    ```

    2c. `_cache_dir_for_run(*, run_id: str, reuse_cache: bool, signature: Dict[str, Any]) -> Path`:
    ```python
    def _cache_dir_for_run(*, run_id: str, reuse_cache: bool, signature: Dict[str, Any]) -> Path:
        """D-08/D-09 cache path resolver.

        reuse_cache=False (default): .embedding_cache/{run_id}/
        reuse_cache=True:             .embedding_cache/sig_{sha256(payload)[:16]}/
        """
        base = Path(".embedding_cache")
        if not reuse_cache:
            return base / str(run_id)
        payload = json.dumps({k: v for k, v in signature.items() if k != "run_id"},
                             sort_keys=True).encode("utf-8")
        sig_hex = hashlib.sha256(payload).hexdigest()[:16]
        return base / f"sig_{sig_hex}"
    ```

    2d. `_save_local_user_state(*, partition_id, state_dict, run_id, reuse_cache, signature)`:
    ```python
    def _save_local_user_state(*, partition_id: int, state_dict: Dict[str, torch.Tensor],
                               run_id: str, reuse_cache: bool, signature: Dict[str, Any]) -> None:
        """D-04 + D-06 + D-07 + D-10 atomic save.

        Writes .pt payload (single-row state_dict: 'local_user_row' + 'local_user_bias')
        atomically via tempfile + os.replace, then writes/updates manifest.json via atomic_write_json.
        """
        cache_dir = _cache_dir_for_run(run_id=run_id, reuse_cache=reuse_cache, signature=signature)
        cache_dir.mkdir(parents=True, exist_ok=True)
        pt_path = cache_dir / f"partition_{int(partition_id)}.pt"
        # D-10 shape guard: payload MUST be the single-row contract, never (num_users, d).
        assert set(state_dict.keys()) == {"local_user_row", "local_user_bias"}, \
            f"D-10 violated: local state has keys {sorted(state_dict.keys())}, expected {{'local_user_row','local_user_bias'}}"
        fd, tmp = tempfile.mkstemp(prefix=".partition_", dir=str(cache_dir))
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

    2e. `_load_local_user_state(*, partition_id, run_id, reuse_cache, signature) -> Optional[Dict[str, torch.Tensor]]`:
    ```python
    def _load_local_user_state(*, partition_id: int, run_id: str, reuse_cache: bool,
                               signature: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """D-04 + D-05 + D-10 load with loud mismatch.

        Returns None if the cache directory or partition .pt does not exist (cold start).
        Raises RuntimeError with per-field delta and rm -rf hint on signature mismatch.
        """
        cache_dir = _cache_dir_for_run(run_id=run_id, reuse_cache=reuse_cache, signature=signature)
        pt_path = cache_dir / f"partition_{int(partition_id)}.pt"
        manifest_path = cache_dir / "manifest.json"
        if not pt_path.exists() or not manifest_path.exists():
            return None  # cold start
        with open(manifest_path, "r") as f:
            cached = json.load(f)
        deltas: List[str] = []
        for key in ("schema_version", "run_id", "method", "num_users", "num_items", "dim", "split_hash"):
            if reuse_cache and key == "run_id":
                continue  # sig_* dir is run_id-agnostic
            if cached.get(key) != signature.get(key):
                deltas.append(f"{key} cached={cached.get(key)!r}, current={signature.get(key)!r}")
        if deltas:
            raise RuntimeError(
                "Embedding-cache signature mismatch (D-05):\n  "
                + "\n  ".join(deltas)
                + f"\nRun: rm -rf {cache_dir}/ to reset, or check --run-config."
            )
        state = torch.load(pt_path, map_location="cpu")
        # D-10 shape guard on load
        assert set(state.keys()) == {"local_user_row", "local_user_bias"}, \
            f"D-10 violated on load: payload keys {sorted(state.keys())}"
        return state
    ```

    Step 3 — Rewrite the `@app.train()` handler body:
    1. Read `mode = str(context.run_config.get("mode", "cross_silo_legacy"))`; `profile = resolve_mode_defaults(mode)`; `overrides = log_mode_and_overrides(mode, profile, context.run_config)`.
    2. Read `run_seed = int(context.run_config.get("run-seed", profile.run_seed))`, `round_num = msg.content["round_num"].to_dict()["value"]` or the current Flower round via the grid message config, `partition_id = int(context.node_config["partition-id"])`.
    3. Load partition data via dataset.load_partition_data; extract `num_users_in_client` (under benchmark mode this will be 1; that's the point).
    4. Call `assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)` — **raises AssertionError if >1 user under benchmark mode**.
    5. Build signature dict from run_id + method + num_users=6040 + num_items=3706 + dim + split_hash (read split_hash from the foundation bundle).
    6. Attempt `_load_local_user_state(...)`; if None (cold start), the model keeps its Xavier-uniform initial `local_user_row` (D-11 first-use init).
    7. If reuse_cache = run_config.get("reuse-cache", False); pass through.
    8. Load GLOBAL params from the Flower message ArrayRecord → model.set_global_parameters(state).
    9. Build `rng = np_rng(run_seed, partition_id, round_num, "train_neg")`.
    10. Load exclusion set: `exclude_items = bundle["exclusion"].for_user(partition_id).tolist()`.
    11. Call task.train(model, trainloader, run_seed=run_seed, user_idx=partition_id, round_num=round_num, exclude_items=exclude_items, rng=rng, ...) — threads all 5 kwargs.
    12. Save single-row local state: `_save_local_user_state(partition_id=partition_id, state_dict=model.get_local_parameters(), ...)`.
    13. Build GLOBAL params ArrayRecord from model.get_global_parameters().
    14. Build `FitMetricsContract(train_loss=..., num_positives=..., num_training_examples=..., round_num=round_num, partition_id=partition_id).to_dict()`; validate_fit_metrics(payload); return as MetricRecord.

    Step 4 — Rewrite the `@app.evaluate()` handler body. Follow Phase 2 Plan 05 pattern:
    1. **First check** `config_record = msg.content["train_config"]` (or equivalent key used for config). If `config_record.get("discover_only", False)` is true → short-circuit:
       ```python
       payload = EvaluateMetricsContract(
           hit_count_overall_at10=0, ndcg_sum_overall_at10=0.0, evaluated_users=0,
           partition_id=partition_id,
       ).to_dict()
       validate_evaluate_metrics(payload)
       return MetricRecord(payload) / Message.reply_to(msg, content=...)
       ```
       **No model load, no data load, no evaluation.** The handshake builds partition_to_node_id on the server.
    2. Otherwise: mode resolve → one-user assert → `assert get_primary_evaluator(mode) == "sampled_loo_99"` → load partition data → load exclusion set → load single-row local state (D-11 first-use path if absent) → set_global_parameters → call evaluate_ranking_sampled with FND-06 kwargs.
    3. Compute per-user-group sufficient stats via `_classify_partition_user_group(bundle, partition_id)`; route hit_count/ndcg_sum/evaluated_users into the matching bucket; other two groups get explicit zeros.
    4. Build `EvaluateMetricsContract(hit_count_overall_at10=..., ndcg_sum_overall_at10=..., evaluated_users=..., eval_loss=..., sampled_hr_at10=..., sampled_ndcg_at10=..., partition_id=partition_id, hit_count_{group}_at10=..., ndcg_sum_{group}_at10=..., evaluated_users_{group}=...).to_dict()`; validate_evaluate_metrics; return.

    Step 5 — D-18 preserve: the pre-existing `get_device` + `_device_cache` pattern; any existing atomic-cache save already present if it uses a different (incompatible) path layout, REPLACE it — this plan owns D-04..D-10 migration.

    Step 6 — Create federated-personalized-cf/tests/test_client_assertion.py with 5 tests (clone from baseline); augment `test_fit_metrics_contract_payload_shape` to include `partition_id=42` in the constructor and assert 'partition_id' is in to_dict() output.

    Step 7 — Create federated-personalized-cf/tests/test_embedding_cache_manifest.py with 4 tests listed in <behavior>. Use tmp_path fixture so tests write into pytest's temporary dir (not the real .embedding_cache/). The tests call the module-level helpers (`_save_local_user_state`, `_load_local_user_state`, `_cache_dir_for_run`) directly.

    Step 8 — Verify: `cd federated-personalized-cf && pytest tests/test_client_assertion.py tests/test_embedding_cache_manifest.py -v` → 9 passed. Full suite: `pytest tests/ -v` → ~24 passed (Plan 01 = 11, Plan 02 = 3, Plan 03 Task 1 = 4, Plan 03 Task 2 = 9 → 27 if all land; aim for at least 22).

    Step 9 — Cross-file RNG regression check (reuse Task 1's test_random_seed_calls_stripped): `grep -rnE "random\\.seed\\(|random\\.sample\\(|^import random$" federated-personalized-cf/federated_personalized_cf/task.py federated-personalized-cf/federated_personalized_cf/client_app.py` → 0 matches.

    Step 10 — Commit (--no-verify):
    ```
    git add federated-personalized-cf/federated_personalized_cf/client_app.py \
            federated-personalized-cf/tests/test_client_assertion.py \
            federated-personalized-cf/tests/test_embedding_cache_manifest.py
    git commit --no-verify -m "feat(03-03): client_app mode + assert + manifest-sidecar cache (PSN-02, PSN-04, PSN-05, PSN-06)"
    ```
  </action>
  <acceptance_criteria>
    - `grep -c "from fedrec_foundation.mode import" federated-personalized-cf/federated_personalized_cf/client_app.py` returns 1
    - `grep -c "assert_benchmark_one_user_per_client" federated-personalized-cf/federated_personalized_cf/client_app.py` returns at least 2 (train + evaluate handlers)
    - `grep -c "get_primary_evaluator" federated-personalized-cf/federated_personalized_cf/client_app.py` returns at least 1
    - `grep -c "FitMetricsContract" federated-personalized-cf/federated_personalized_cf/client_app.py` returns at least 2
    - `grep -c "EvaluateMetricsContract" federated-personalized-cf/federated_personalized_cf/client_app.py` returns at least 3
    - `grep -c "partition_id=partition_id" federated-personalized-cf/federated_personalized_cf/client_app.py` returns at least 2 (populated on both contracts; G-03-01 carry-forward)
    - `grep -c "discover_only" federated-personalized-cf/federated_personalized_cf/client_app.py` returns at least 1 (discovery-round handshake short-circuit)
    - `grep -c "schema_version" federated-personalized-cf/federated_personalized_cf/client_app.py` returns at least 1 (D-06 manifest field)
    - `grep -c "atomic_write_json" federated-personalized-cf/federated_personalized_cf/client_app.py` returns at least 1 (D-07 manifest writer)
    - `grep -c "reuse_cache\|reuse-cache" federated-personalized-cf/federated_personalized_cf/client_app.py` returns at least 2 (D-09 opt-in)
    - `grep -cE "rm -rf" federated-personalized-cf/federated_personalized_cf/client_app.py` returns at least 1 (D-05 literal hint in error message)
    - `grep -cE "^import random$|random\\.seed\\(|random\\.sample\\(" federated-personalized-cf/federated_personalized_cf/client_app.py` returns 0 (cross-file regression)
    - `grep -cE "^import random$|random\\.seed\\(|random\\.sample\\(" federated-personalized-cf/federated_personalized_cf/task.py` returns 0 (Task 1 guard reconfirmed)
    - `cd federated-personalized-cf && pytest tests/test_client_assertion.py -v` exits 0 with "5 passed"
    - `cd federated-personalized-cf && pytest tests/test_embedding_cache_manifest.py -v` exits 0 with "4 passed"
    - `cd federated-personalized-cf && pytest tests/ -v` exits 0 with at least 22 tests passing overall
    - D-18 scope: `git diff --stat federated-personalized-cf/federated_personalized_cf/strategy.py federated-personalized-cf/federated_personalized_cf/dataset.py federated-personalized-cf/federated_personalized_cf/server_app.py federated-personalized-cf/federated_personalized_cf/models/ federated-personalized-cf/pyproject.toml` returns empty after commit
  </acceptance_criteria>
  <done>client_app.py implements the full cross-device contract with mode resolver, benchmark-mode one-user assertion, discover_only short-circuit (G-03-01 handshake), strict FitMetrics/EvaluateMetrics payloads with optional partition_id, per-group sufficient-stat routing, and D-04..D-10 manifest-sidecar cache layout with loud D-05 mismatch + D-09 reuse-cache opt-in. 9 new GREEN tests across 2 new test files.</done>
</task>

</tasks>

<verification>
- `cd federated-personalized-cf && pytest tests/ -v` exits 0 with at least 22 tests passing (Plan 01=11 + Plan 02=3 + Plan 03 Task 1=4 + Plan 03 Task 2=9 → ~27; exact total depends on test granularity)
- `grep -rnE "random\\.seed\\(|random\\.sample\\(|^import random$" federated-personalized-cf/federated_personalized_cf/task.py federated-personalized-cf/federated_personalized_cf/client_app.py` returns 0 matches (BSL-05-style cross-file regression)
- `python -c "from fedrec_foundation.fit_metrics import FitMetricsContract, validate_fit_metrics; c = FitMetricsContract(train_loss=0.1, num_positives=1, num_training_examples=5, partition_id=42); d = c.to_dict(); assert d['partition_id'] == 42; validate_fit_metrics(d); print('ok')"` prints `ok`
- `python -c "from fedrec_foundation.fit_metrics import EvaluateMetricsContract, validate_evaluate_metrics; e = EvaluateMetricsContract(hit_count_overall_at10=0, ndcg_sum_overall_at10=0.0, evaluated_users=0, partition_id=1234); validate_evaluate_metrics(e.to_dict()); print('ok')"` prints `ok`
- D-18 scope: `git diff --stat` across strategy.py / dataset.py / server_app.py / models/ / pyproject.toml returns empty after this plan's commits (Plans 01/02/04 own those files)
</verification>

<success_criteria>
- PSN-02 observable end-to-end: a partition with >1 user raises AssertionError under benchmark_cross_device mode before any training or evaluation happens.
- PSN-03 observable: ExclusionTable.for_user(partition_id) is merged into user_rated_items before train-neg sampling AND into negative_candidates before eval-neg sampling; the held-out test positive can never be drawn as either.
- PSN-05 observable: `.embedding_cache/{run_id}/manifest.json` + `.embedding_cache/{run_id}/partition_{pid}.pt` layout with D-05 loud-mismatch RuntimeError including `rm -rf .embedding_cache/{run_id}/` literal hint; D-09 opt-in via `reuse-cache=true` switches to `.embedding_cache/sig_{sha256[:16]}/` path.
- PSN-06 observable on disk: .pt payload contains exactly 2 keys (local_user_row of shape (d,), local_user_bias of shape (1,)) — never the old (num_users, d) blob. Assertion in both save and load paths.
- G-03-01 carry-forward: optional partition_id populated on both contracts; @app.evaluate discover_only=True returns a minimal zero-suffstats payload for the pre-round-1 handshake.
</success_criteria>

<output>
After completion, create `.planning/phases/03-personalized-migration/03-personalized-migration-03-SUMMARY.md` with: file list (2 modified + 3 created), decisions made (D-24 skipped justification — single-row model makes it unnecessary), deviations, test counts (~13 GREEN in this plan, total suite ~24 GREEN), commit SHAs, PSN-02/03/05/06 closure notes, Plan 04 readiness confirmation (strategy + client-side contract both in place; server_app.py can now drop PersonalizedSplitFedAvg + discovery round + partition-space sampling).
</output>
