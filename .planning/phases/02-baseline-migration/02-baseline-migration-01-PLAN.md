---
phase: 02-baseline-migration
plan: 01
type: tdd
wave: 1
depends_on: []
files_modified:
  - scripts/foundation/fedrec_foundation/fit_metrics.py
  - scripts/foundation/tests/test_weight_policy.py
  - scripts/foundation/tests/test_evaluate_metrics.py
  - federated-baseline-cf/federated_baseline_cf/strategy.py
  - federated-baseline-cf/tests/__init__.py
  - federated-baseline-cf/tests/conftest.py
  - federated-baseline-cf/tests/test_strategy.py
autonomous: true
requirements:
  - BSL-06

must_haves:
  truths:
    - "FitMetricsContract accepts per-group sufficient-stat keys (hit_count_sparse@10, evaluated_users_sparse, ndcg_sum_sparse@10, hit_count_medium@10, evaluated_users_medium, ndcg_sum_medium@10, hit_count_dense@10, evaluated_users_dense, ndcg_sum_dense@10, hit_count@10, evaluated_users, ndcg_sum@10) as dataclass fields; validate_fit_metrics still checks original required keys plus the new per-group keys."
    - "EvaluateMetricsContract exists as a separate dataclass using the SAME `_at10` (no underscore before 10) naming convention as FitMetricsContract and the strategy `_sum_sufficient_stats` reader. Required fields (hit_count_overall_at10, ndcg_sum_overall_at10, evaluated_users) are what the server aggregator consumes; 9 optional per-group keys (hit_count_{sparse,medium,dense}_at10, ndcg_sum_{sparse,medium,dense}_at10, evaluated_users_{sparse,medium,dense}) mirror FitMetricsContract; 3 optional diagnostic keys (eval_loss, sampled_hr_at10, sampled_ndcg_at10) are cached client-side for logs only and are NOT consumed for aggregation (server re-computes ratios from summed sufficient stats). EVAL_METRICS_REQUIRED_KEYS frozenset + validate_evaluate_metrics(payload) are exported. D-21/D-22 strict-contract guarantee extends to the evaluate wire payload (Plan 03 Task 2 consumes it). Iteration 2 fix: field names now match the strategy reader exactly — previous `_at_10`-vs-`_at10` drift would have silently zeroed all evaluate-side aggregation in production."
    - "BaselineFedAvg(FedAvg) and BaselineFedProx(FedProx) subclasses exist in federated_baseline_cf/strategy.py; each exposes aggregate_evaluate(server_round, results, failures) that sums sufficient stats across clients and returns (overall_loss, {'sampled_hr@10': hit_count/evaluated_users, 'sampled_ndcg@10': ndcg_sum/evaluated_users, 'sampled_hr@10/sparse': ..., 'sampled_ndcg@10/sparse': ..., 'sampled_hr@10/medium': ..., 'sampled_ndcg@10/medium': ..., 'sampled_hr@10/dense': ..., 'sampled_ndcg@10/dense': ..., 'evaluated_users': ..., 'evaluated_users_sparse': ..., 'evaluated_users_medium': ..., 'evaluated_users_dense': ...})."
    - "Aggregation does NOT average per-client ratios — it returns sum(hit_count) / sum(evaluated_users). The strategy's aggregate_fit() method is INHERITED unchanged from flwr.server.strategy.FedAvg / FedProx."
    - "pytest federated-baseline-cf/tests/test_strategy.py -v exits 0 with at least 3 GREEN tests."
  artifacts:
    - path: "scripts/foundation/fedrec_foundation/fit_metrics.py"
      provides: "Extended FitMetricsContract with per-group sufficient-stat fields + EvaluateMetricsContract"
      contains: "hit_count_sparse"
    - path: "scripts/foundation/tests/test_evaluate_metrics.py"
      provides: "Unit tests for EvaluateMetricsContract + validate_evaluate_metrics"
      contains: "def test_evaluate_metrics_required_keys_enforced"
    - path: "federated-baseline-cf/federated_baseline_cf/strategy.py"
      provides: "BaselineFedAvg + BaselineFedProx with sufficient-stat aggregate_evaluate"
      contains: "class BaselineFedAvg"
    - path: "federated-baseline-cf/tests/test_strategy.py"
      provides: "Unit tests proving sufficient-stat aggregation returns sum-based ratios"
      contains: "def test_aggregate_evaluate_sums_sufficient_stats"
    - path: "federated-baseline-cf/tests/conftest.py"
      provides: "pytest config + test fixtures"
      contains: "def fake_evaluate_res"
  key_links:
    - from: "federated_baseline_cf.strategy.BaselineFedAvg.aggregate_evaluate"
      to: "fedrec_foundation.fit_metrics.validate_fit_metrics"
      via: "server-side validation before sufficient-stat aggregation"
      pattern: "validate_fit_metrics"
    - from: "federated_baseline_cf.strategy.BaselineFedAvg.aggregate_evaluate"
      to: "FitMetricsContract per-group fields"
      via: "sum sufficient stats, divide by evaluated_users"
      pattern: "hit_count_sparse|evaluated_users_sparse|ndcg_sum_sparse"
    - from: "federated_baseline_cf.client_app::@app.evaluate (Plan 03 consumer)"
      to: "fedrec_foundation.fit_metrics.EvaluateMetricsContract"
      via: "strict contract + validate_evaluate_metrics on the evaluate-side wire payload"
      pattern: "EvaluateMetricsContract\\(|validate_evaluate_metrics\\("
---

<objective>
Ship the contract extension and strategy scaffold that BSL-06 (sufficient-stat aggregation, D-20, D-21, D-22) depends on. This plan is a blocking Wave 1 gate for Plans 03 and 04.

Purpose: Under D-22 the `FitMetricsContract` is the single source of truth for everything that crosses the client/server boundary on the FIT side; per-group sufficient stats must be first-class fields of the contract, not free-form metrics extras. On the EVALUATE side (Plan 03 Task 2 consumer) the same strict-contract discipline is required, so this plan introduces a sibling `EvaluateMetricsContract` with matching per-group fields plus a `validate_evaluate_metrics(payload)` checker — otherwise D-21's "no free-form extras" guarantee is unenforceable because Plan 03's evaluate payload carries extra keys (`eval_loss`, `sampled_hr@10`, `sampled_ndcg@10`) that `validate_fit_metrics` does not check. Per D-20, baseline aggregation lives in a `BaselineFedAvg(FedAvg)` subclass that overrides `aggregate_evaluate` to compute headline metrics ONCE from summed sufficient statistics (server-side ratio), rather than averaging per-client ratios (which silently double-counts sparse users and produces mis-weighted thesis-table numbers). Plan 04 instantiates this strategy; Plan 03 populates these contract fields client-side; extending both contracts in Wave 1 unblocks both.

Output: (1) Extended `FitMetricsContract` with 12 new per-group + overall sufficient-stat fields — backwards-compatible (all new fields are `Optional` with `None` default). (2) New `EvaluateMetricsContract` with 3 required fields (`hit_count_overall_at10`, `ndcg_sum_overall_at10`, `evaluated_users`) plus 9 optional per-group fields (`hit_count_{sparse,medium,dense}_at10`, `ndcg_sum_{sparse,medium,dense}_at10`, `evaluated_users_{sparse,medium,dense}`) and 3 optional diagnostic fields (`eval_loss`, `sampled_hr_at10`, `sampled_ndcg_at10` — diagnostic caches only; NOT consumed for aggregation, since the server re-computes the headline ratios from summed sufficient stats via `_sum_sufficient_stats` + `_sufficient_stats_to_thesis_metrics`). **Unified naming convention (iteration 2 fix): EvaluateMetricsContract field names now match FitMetricsContract AND the strategy's `_sum_sufficient_stats` reader — no more `_at_10`-with-underscore vs `_at10`-without-underscore drift. Previous iteration used two distinct suffixes which silently zeroed all evaluate-side metrics in production (tests bypassed it by constructing strategy-shape dicts directly).** `EVAL_METRICS_REQUIRED_KEYS` frozenset; `validate_evaluate_metrics(payload)` function. (3) New `federated_baseline_cf/strategy.py` with `BaselineFedAvg(FedAvg)` and `BaselineFedProx(FedProx)` that override `aggregate_evaluate` to emit the correct server-side ratio. (4) pytest test harness in `federated-baseline-cf/tests/` mirroring Phase 1 layout (conftest.py fixtures + per-module tests). (5) New `scripts/foundation/tests/test_evaluate_metrics.py` covering the new contract. (pyproject.toml dev-dep declaration is OWNED by Plan 02 Task 1 — this plan does NOT touch pyproject.toml to avoid a Wave-1 write race with Plan 02.)
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
@CLAUDE.md
@federated-baseline-cf/claude.md

@scripts/foundation/fedrec_foundation/fit_metrics.py
@scripts/foundation/fedrec_foundation/weight_policy.py
@scripts/foundation/tests/test_weight_policy.py
@scripts/foundation/tests/conftest.py

<interfaces>
<!-- Current FitMetricsContract (Phase 1 Plan 03). The extension below ADDS optional per-group + overall sufficient-stat fields to FitMetricsContract AND adds a sibling EvaluateMetricsContract for the evaluate-side wire payload. -->

From scripts/foundation/fedrec_foundation/fit_metrics.py:
```python
FIT_METRICS_REQUIRED_KEYS = ("train_loss", "num_positives", "num_training_examples")

@dataclass
class FitMetricsContract:
    train_loss: float
    num_positives: int
    num_training_examples: int
    round_num: Optional[int] = None

    def to_dict(self) -> Dict[str, Union[int, float]]: ...   # drops None
    @classmethod
    def from_dict(cls, d) -> "FitMetricsContract": ...       # wraps TypeError as ValueError
```

From scripts/foundation/fedrec_foundation/weight_policy.py:
```python
class WeightPolicy(str, Enum):
    UNIFORM = "uniform"
    NUM_POSITIVES = "num_positives"
    NUM_TRAINING_EXAMPLES = "num_training_examples"

def compute_aggregation_weight(client_metrics, policy) -> float: ...
```

From flwr.server.strategy:
```python
class FedAvg:
    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures) -> Tuple[Optional[float], Dict[str, Scalar]]: ...
class FedProx(FedAvg):
    def __init__(self, *, fraction_fit=1.0, proximal_mu=0.0, ...): ...
```

From fedrec_foundation.user_groups:
```python
USER_GROUP_BOUNDARIES = (30, 100)   # sparse < 30 <= medium < 100 <= dense
BUCKET_SEMANTICS = "half_open"
def classify_user_group(n_interactions: int) -> str: ...   # "sparse" | "medium" | "dense"
```
</interfaces>

</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Extend FitMetricsContract with per-group sufficient-stat fields + add EvaluateMetricsContract (D-21, D-22)</name>
  <files>
    scripts/foundation/fedrec_foundation/fit_metrics.py
    scripts/foundation/tests/test_weight_policy.py
    scripts/foundation/tests/test_evaluate_metrics.py
  </files>
  <read_first>
    - scripts/foundation/fedrec_foundation/fit_metrics.py (ENTIRE file — current dataclass shape, from_dict filter behavior, validate_fit_metrics type check)
    - scripts/foundation/tests/test_weight_policy.py (ENTIRE file — existing 12 tests; DO NOT remove or re-label any)
    - scripts/foundation/tests/conftest.py (pytest config — foundation tests run with `cd scripts/foundation && pytest`)
    - scripts/foundation/fedrec_foundation/user_groups.py (user_group strings "sparse"/"medium"/"dense")
    - .planning/phases/02-baseline-migration/02-CONTEXT.md §decisions (D-21 strict-contract wire payload, D-22 per-group sufficient stats)
    - CLAUDE.md §"Code Standards" (typing pre-3.10: `typing.Dict`, `typing.Optional`, `typing.Union`; NumPy-style docstrings)
  </read_first>
  <behavior>
    - Test 1 (RED→GREEN, test_fit_metrics_per_group_fields, in test_weight_policy.py): FitMetricsContract(train_loss=0.5, num_positives=30, num_training_examples=150, round_num=3, hit_count_overall_at10=24, ndcg_sum_overall_at10=12.5, evaluated_users=24, hit_count_sparse_at10=6, ndcg_sum_sparse_at10=2.0, evaluated_users_sparse=8, hit_count_medium_at10=10, ndcg_sum_medium_at10=5.0, evaluated_users_medium=10, hit_count_dense_at10=8, ndcg_sum_dense_at10=5.5, evaluated_users_dense=6).to_dict() returns a dict with all 15 keys present and None-dropped.
    - Test 2 (RED→GREEN, test_fit_metrics_per_group_optional, in test_weight_policy.py): FitMetricsContract(train_loss=0.5, num_positives=30, num_training_examples=150).to_dict() STILL works (does not raise), and the 12 new per-group fields default to None and are DROPPED by to_dict — i.e., backward-compatible.
    - Test 3 (RED→GREEN, test_fit_metrics_forward_compat_with_per_group_extension, in test_weight_policy.py): from_dict({"train_loss": 0.1, "num_positives": 2, "num_training_examples": 10, "hit_count_overall_at10": 1, "ndcg_sum_overall_at10": 0.63, "evaluated_users": 1, "alpha": 0.42}) succeeds (unknown "alpha" is filtered; known per-group "hit_count_overall_at10" is populated; missing per-group fields remain None).
    - Existing 12 tests in test_weight_policy.py remain GREEN with zero edits.
    - Test 4 (RED→GREEN, test_evaluate_metrics_required_keys_enforced, in test_evaluate_metrics.py): validate_evaluate_metrics({"eval_loss": 0.5, "sampled_hr_at10": 0.1, "sampled_ndcg_at10": 0.05, "evaluated_users": 1, "hit_count_overall_at10": 0, "ndcg_sum_overall_at10": 0.0}) succeeds; validate_evaluate_metrics({"eval_loss": 0.5}) raises ValueError with `missing required keys` message.
    - Test 5 (RED→GREEN, test_evaluate_metrics_per_group_fields, in test_evaluate_metrics.py): EvaluateMetricsContract(eval_loss=0.5, sampled_hr_at10=0.3, sampled_ndcg_at10=0.15, evaluated_users=10, hit_count_overall_at10=3, ndcg_sum_overall_at10=1.5, hit_count_sparse_at10=1, ndcg_sum_sparse_at10=0.5, evaluated_users_sparse=4, hit_count_medium_at10=2, ndcg_sum_medium_at10=1.0, evaluated_users_medium=6, hit_count_dense_at10=0, ndcg_sum_dense_at10=0.0, evaluated_users_dense=0).to_dict() returns a dict with all 15 keys present and None-dropped. All 6 required keys present even when optional per-group keys are absent.
    - Test 6 (RED→GREEN, test_evaluate_metrics_forward_compat, in test_evaluate_metrics.py): EvaluateMetricsContract.from_dict({"eval_loss": 0.5, "sampled_hr_at10": 0.1, "sampled_ndcg_at10": 0.05, "evaluated_users": 1, "hit_count_overall_at10": 0, "ndcg_sum_overall_at10": 0.0, "train_loss": 0.3}) succeeds; unknown key "train_loss" is filtered; optional per-group fields default to None.
    - Test 7 (RED→GREEN, test_evaluate_metrics_rejects_wrong_types, in test_evaluate_metrics.py): EvaluateMetricsContract(eval_loss="not-a-float", ...) raises ValueError (wraps TypeError per Phase 1 CR-4 pattern).
  </behavior>
  <action>
Extend `scripts/foundation/fedrec_foundation/fit_metrics.py` by adding 12 optional per-group and overall sufficient-stat fields to the `FitMetricsContract` dataclass, keeping all existing fields and behavior identical. Do NOT add the new keys to `FIT_METRICS_REQUIRED_KEYS`. Do NOT change `validate_fit_metrics` semantics — it validates only the original three required keys (train_loss, num_positives, num_training_examples).

In the same file, ADD a sibling `EvaluateMetricsContract` dataclass + `EVAL_METRICS_REQUIRED_KEYS` frozenset + `validate_evaluate_metrics(payload)` function. This is a separate public API from FitMetricsContract. The evaluate-side payload (constructed in Plan 03 Task 2) must validate via `validate_evaluate_metrics(payload)` — NOT `validate_fit_metrics(payload)` — which is why a sibling contract is required (D-21: "No free-form extras in the metrics dict" — the evaluate payload carries `eval_loss`, `sampled_hr_at10`, `sampled_ndcg_at10` that are NOT in FIT_METRICS_REQUIRED_KEYS).

Use TDD:
1. First run `pytest scripts/foundation/tests/test_weight_policy.py -v` to confirm 12 tests GREEN baseline.
2. Add the 3 FitMetricsContract extension tests (test_fit_metrics_per_group_fields, test_fit_metrics_per_group_optional, test_fit_metrics_forward_compat_with_per_group_extension) to `scripts/foundation/tests/test_weight_policy.py`, confirm they FAIL, then extend the dataclass to make them GREEN.
3. Create `scripts/foundation/tests/test_evaluate_metrics.py` with the 4 EvaluateMetricsContract tests, confirm they FAIL (ImportError — the contract does not exist yet), then add the EvaluateMetricsContract + EVAL_METRICS_REQUIRED_KEYS + validate_evaluate_metrics to make them GREEN.

**Do NOT modify `federated-baseline-cf/pyproject.toml` in this task.** Plan 02 Task 1 exclusively owns pyproject.toml modifications — including adding the `[project.optional-dependencies] dev = ["pytest>=7.0"]` section — so that Wave 1 (Plans 01 + 02 running in parallel) has a single writer on every file. See CONTEXT.md §decisions D-18 (surgical migration) and the revision iteration 1 blocker fix.

Exact field additions to `FitMetricsContract` (all `Optional[Union[int, float]] = None`; name them with `_at10` suffix instead of `@10` because `@` breaks Python identifier rules):

```python
from dataclasses import dataclass, field, fields
from typing import Dict, FrozenSet, Optional, Union

FIT_METRICS_REQUIRED_KEYS = ("train_loss", "num_positives", "num_training_examples")  # UNCHANGED

@dataclass
class FitMetricsContract:
    train_loss: float
    num_positives: int
    num_training_examples: int
    # Per-module extension fields below (optional; modules add their own).
    round_num: Optional[int] = None
    # --- Phase 2 extension (D-22): per-group + overall sufficient stats for BSL-06 aggregation. ---
    # All optional to preserve the Phase 1 contract. Populated client-side in Phase 2 Plan 03.
    hit_count_overall_at10: Optional[int] = None
    ndcg_sum_overall_at10: Optional[float] = None
    evaluated_users: Optional[int] = None
    hit_count_sparse_at10: Optional[int] = None
    ndcg_sum_sparse_at10: Optional[float] = None
    evaluated_users_sparse: Optional[int] = None
    hit_count_medium_at10: Optional[int] = None
    ndcg_sum_medium_at10: Optional[float] = None
    evaluated_users_medium: Optional[int] = None
    hit_count_dense_at10: Optional[int] = None
    ndcg_sum_dense_at10: Optional[float] = None
    evaluated_users_dense: Optional[int] = None
```

Exact new additions for `EvaluateMetricsContract` (naming: **`_at10` — no underscore between `at` and `10` — unified with FitMetricsContract AND with the strategy's `_sum_sufficient_stats` key tuple** in Task 2 of this plan; iteration 2 fix removes the previous `_at_10`-vs-`_at10` drift that silently zeroed all evaluate-side aggregation in production):

```python
EVAL_METRICS_REQUIRED_KEYS: FrozenSet[str] = frozenset({
    "hit_count_overall_at10",
    "ndcg_sum_overall_at10",
    "evaluated_users",
})


@dataclass
class EvaluateMetricsContract:
    """Strict-contract wire payload for Flower @app.evaluate() responses (D-21, D-22).

    Required fields MUST be present in any ``to_dict()`` output (these are
    the canonical sufficient-stat keys that :func:`_sum_sufficient_stats` reads
    server-side; aggregation is sum-based, NOT averaged per-client ratios):
      - ``hit_count_overall_at10`` : int — client's overall hit-count sufficient stat.
      - ``ndcg_sum_overall_at10`` : float — client's overall NDCG-sum sufficient stat.
      - ``evaluated_users`` : int — client's evaluated-user count (denominator
        for server-side ratio reconstruction).

    Optional diagnostic fields (cached client-side for per-round logs only; NOT
    consumed by the server aggregator, since the server re-computes the
    headline ratios from summed sufficient stats):
      - ``eval_loss`` : Optional[float] — per-client weighted average loss.
      - ``sampled_hr_at10`` : Optional[float] — client-local HR@10 ratio.
      - ``sampled_ndcg_at10`` : Optional[float] — client-local NDCG@10 ratio.

    Optional per-group fields (mirror FitMetricsContract's 6 per-group keys):
      - ``hit_count_{sparse,medium,dense}_at10`` : Optional[int]
      - ``ndcg_sum_{sparse,medium,dense}_at10`` : Optional[float]
      - ``evaluated_users_{sparse,medium,dense}`` : Optional[int]

    The server's ``BaselineFedAvg.aggregate_evaluate`` (Phase 2 Plan 01 Task 2)
    reads these sufficient stats and emits server-side ratios; client-local
    ratios (``sampled_hr_at10`` / ``sampled_ndcg_at10``) are informational
    only (they're what Flower shows in its per-round dump before our strategy
    override runs).

    Parameters
    ----------
    eval_loss : float
        Informational per-client weighted average loss.
    sampled_hr_at10 : float
        Client-local HR@10 ratio.
    sampled_ndcg_at10 : float
        Client-local NDCG@10 ratio.
    evaluated_users : int
        Client's evaluated-user count.
    hit_count_overall_at10 : int
        Client's overall hit-count sufficient stat.
    ndcg_sum_overall_at10 : float
        Client's overall NDCG-sum sufficient stat.
    hit_count_sparse_at10, ndcg_sum_sparse_at10, evaluated_users_sparse : optional
    hit_count_medium_at10, ndcg_sum_medium_at10, evaluated_users_medium : optional
    hit_count_dense_at10, ndcg_sum_dense_at10, evaluated_users_dense : optional
        Per-group sufficient-stat fields (D-22). Optional for backwards
        compatibility; Plan 03 Task 2 populates all three groups
        (sparse/medium/dense), with the client's group receiving the
        non-zero values and the other two groups receiving zeros.

    Returns
    -------
    EvaluateMetricsContract
        Dataclass instance.
    """
    # --- Required fields (sufficient stats; unified naming with FitMetricsContract + strategy reader). ---
    hit_count_overall_at10: int
    ndcg_sum_overall_at10: float
    evaluated_users: int
    # --- Optional diagnostic fields (cached client-side; NOT consumed by server aggregator). ---
    eval_loss: Optional[float] = None
    sampled_hr_at10: Optional[float] = None
    sampled_ndcg_at10: Optional[float] = None
    # --- Optional per-group fields (D-22): same 6 groups as FitMetricsContract. ---
    hit_count_sparse_at10: Optional[int] = None
    ndcg_sum_sparse_at10: Optional[float] = None
    evaluated_users_sparse: Optional[int] = None
    hit_count_medium_at10: Optional[int] = None
    ndcg_sum_medium_at10: Optional[float] = None
    evaluated_users_medium: Optional[int] = None
    hit_count_dense_at10: Optional[int] = None
    ndcg_sum_dense_at10: Optional[float] = None
    evaluated_users_dense: Optional[int] = None

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Serialize to a JSON-ready dict; drops None-valued optional fields.

        Returns
        -------
        Dict[str, int | float]
            All required fields plus any non-None optional per-group fields.
        """
        result: Dict[str, Union[int, float]] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                continue
            result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, payload: Dict[str, Union[int, float]]) -> "EvaluateMetricsContract":
        """Construct from a dict; filters unknown keys; raises ValueError on type errors.

        Parameters
        ----------
        payload : Dict[str, int | float]
            Dict possibly containing extra keys (filtered) and missing optional
            keys (set to None). Required keys absent will raise TypeError ->
            wrapped as ValueError.

        Returns
        -------
        EvaluateMetricsContract

        Raises
        ------
        ValueError
            If required keys are missing or field types are wrong.
        """
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in payload.items() if k in known}
        try:
            return cls(**filtered)
        except TypeError as exc:
            raise ValueError(f"EvaluateMetricsContract.from_dict failed: {exc}") from exc


def validate_evaluate_metrics(payload: Dict[str, Union[int, float]]) -> None:
    """Assert the evaluate-wire payload satisfies the strict contract (D-21, D-22).

    Checks:
      1. All keys in :data:`EVAL_METRICS_REQUIRED_KEYS` are present.
      2. No free-form extras that are NOT known fields of
         :class:`EvaluateMetricsContract`.

    Parameters
    ----------
    payload : Dict[str, int | float]
        Flower EvaluateRes.metrics dict (or MetricRecord contents).

    Raises
    ------
    ValueError
        If the payload is missing required keys OR contains free-form extras.
    """
    missing = sorted(EVAL_METRICS_REQUIRED_KEYS - set(payload.keys()))
    if missing:
        raise ValueError(
            f"EvaluateMetricsContract missing required keys: {missing}. "
            f"Got payload keys: {sorted(payload.keys())}"
        )
    known = {f.name for f in fields(EvaluateMetricsContract)}
    extras = sorted(set(payload.keys()) - known)
    if extras:
        raise ValueError(
            f"EvaluateMetricsContract rejects free-form extras (D-21): {extras}. "
            f"Known contract fields: {sorted(known)}"
        )
```

Update the module docstring with a single paragraph noting: "Phase 2 extension: FitMetricsContract gains 12 per-group (sparse/medium/dense) and overall sufficient-stat fields (hit_count_*, ndcg_sum_*, evaluated_users*) — all OPTIONAL, default None. A sibling `EvaluateMetricsContract` + `EVAL_METRICS_REQUIRED_KEYS` + `validate_evaluate_metrics` govern the evaluate-side wire payload (D-21 strict-contract, D-22 per-group). Both contracts are populated by per-module clients in Phase 2 Plan 03 and aggregated server-side via module-specific `BaselineFedAvg.aggregate_evaluate` (Phase 2 Plan 01 Task 2). `validate_fit_metrics` continues to check only the Phase 1 FIT required keys; `validate_evaluate_metrics` governs the evaluate-side payload."

Also append the 3 FitMetricsContract tests to `scripts/foundation/tests/test_weight_policy.py`. Do NOT modify any of the existing 12 tests. Use exact test names:

```python
def test_fit_metrics_per_group_fields() -> None:
    contract = FitMetricsContract(
        train_loss=0.5, num_positives=30, num_training_examples=150, round_num=3,
        hit_count_overall_at10=24, ndcg_sum_overall_at10=12.5, evaluated_users=24,
        hit_count_sparse_at10=6, ndcg_sum_sparse_at10=2.0, evaluated_users_sparse=8,
        hit_count_medium_at10=10, ndcg_sum_medium_at10=5.0, evaluated_users_medium=10,
        hit_count_dense_at10=8, ndcg_sum_dense_at10=5.5, evaluated_users_dense=6,
    )
    d = contract.to_dict()
    for key in ["hit_count_overall_at10", "ndcg_sum_overall_at10", "evaluated_users",
                "hit_count_sparse_at10", "ndcg_sum_sparse_at10", "evaluated_users_sparse",
                "hit_count_medium_at10", "ndcg_sum_medium_at10", "evaluated_users_medium",
                "hit_count_dense_at10", "ndcg_sum_dense_at10", "evaluated_users_dense"]:
        assert key in d, f"missing per-group field {key}"

def test_fit_metrics_per_group_optional() -> None:
    # Backwards-compat: per-group fields default None and are DROPPED by to_dict.
    contract = FitMetricsContract(train_loss=0.5, num_positives=30, num_training_examples=150)
    d = contract.to_dict()
    assert "hit_count_sparse_at10" not in d
    assert "evaluated_users" not in d
    assert d == {"train_loss": 0.5, "num_positives": 30, "num_training_examples": 150}

def test_fit_metrics_forward_compat_with_per_group_extension() -> None:
    # Forward-compat: unknown keys filtered, known per-group keys populated.
    contract = FitMetricsContract.from_dict({
        "train_loss": 0.1, "num_positives": 2, "num_training_examples": 10,
        "hit_count_overall_at10": 1, "ndcg_sum_overall_at10": 0.63, "evaluated_users": 1,
        "alpha": 0.42,  # unknown — filtered
    })
    assert contract.hit_count_overall_at10 == 1
    assert contract.evaluated_users == 1
    assert contract.hit_count_sparse_at10 is None
```

Create `scripts/foundation/tests/test_evaluate_metrics.py` with the 4 EvaluateMetricsContract tests. The file follows the same pytest conventions as `test_weight_policy.py` (imports at top, no skipif gates — these tests do not require foundation bundle artifacts). Exact test body:

```python
"""Tests for EvaluateMetricsContract (Phase 2 Plan 01 — D-21 strict-contract, D-22 per-group)."""
from __future__ import annotations

import pytest

from fedrec_foundation.fit_metrics import (
    EVAL_METRICS_REQUIRED_KEYS,
    EvaluateMetricsContract,
    validate_evaluate_metrics,
)


def test_evaluate_metrics_required_keys_enforced() -> None:
    # Valid payload with all 3 required sufficient-stat keys (and no extras
    # beyond known fields). Diagnostic keys eval_loss / sampled_hr_at10 /
    # sampled_ndcg_at10 are optional — may be absent without raising.
    validate_evaluate_metrics({
        "hit_count_overall_at10": 0,
        "ndcg_sum_overall_at10": 0.0,
        "evaluated_users": 1,
    })
    # Valid payload with required + optional diagnostics + optional per-group.
    validate_evaluate_metrics({
        "hit_count_overall_at10": 0,
        "ndcg_sum_overall_at10": 0.0,
        "evaluated_users": 1,
        "eval_loss": 0.5,
        "sampled_hr_at10": 0.1,
        "sampled_ndcg_at10": 0.05,
    })
    # Missing required keys -> ValueError.
    with pytest.raises(ValueError, match="missing required keys"):
        validate_evaluate_metrics({"eval_loss": 0.5})
    # Free-form extra -> ValueError (D-21: no free-form extras).
    with pytest.raises(ValueError, match="free-form extras"):
        validate_evaluate_metrics({
            "hit_count_overall_at10": 0,
            "ndcg_sum_overall_at10": 0.0,
            "evaluated_users": 1,
            "freeform_field": 1.0,  # not a known contract field
        })


def test_evaluate_metrics_per_group_fields() -> None:
    contract = EvaluateMetricsContract(
        eval_loss=0.5, sampled_hr_at10=0.3, sampled_ndcg_at10=0.15,
        evaluated_users=10, hit_count_overall_at10=3, ndcg_sum_overall_at10=1.5,
        hit_count_sparse_at10=1, ndcg_sum_sparse_at10=0.5, evaluated_users_sparse=4,
        hit_count_medium_at10=2, ndcg_sum_medium_at10=1.0, evaluated_users_medium=6,
        hit_count_dense_at10=0, ndcg_sum_dense_at10=0.0, evaluated_users_dense=0,
    )
    d = contract.to_dict()
    for key in EVAL_METRICS_REQUIRED_KEYS:
        assert key in d, f"missing required field {key}"
    for group in ("sparse", "medium", "dense"):
        assert f"hit_count_{group}_at10" in d
        assert f"ndcg_sum_{group}_at10" in d
        assert f"evaluated_users_{group}" in d


def test_evaluate_metrics_forward_compat() -> None:
    # Unknown keys are filtered; optional per-group fields default None.
    contract = EvaluateMetricsContract.from_dict({
        "eval_loss": 0.5, "sampled_hr_at10": 0.1, "sampled_ndcg_at10": 0.05,
        "evaluated_users": 1, "hit_count_overall_at10": 0, "ndcg_sum_overall_at10": 0.0,
        "train_loss": 0.3,  # unknown — filtered (not in EvaluateMetricsContract fields)
    })
    assert contract.eval_loss == 0.5
    assert contract.hit_count_sparse_at10 is None  # optional default


def test_evaluate_metrics_rejects_wrong_types() -> None:
    # Type errors wrap as ValueError per Phase 1 CR-4 pattern.
    # NOTE: dataclass field annotations do NOT enforce runtime types; we rely on
    # the from_dict TypeError-wrap path, so this test exercises missing-required-keys.
    # The 3 required keys are hit_count_overall_at10 / ndcg_sum_overall_at10 /
    # evaluated_users; eval_loss is an optional diagnostic field.
    with pytest.raises(ValueError):
        EvaluateMetricsContract.from_dict({"eval_loss": 0.5})  # missing 3 required keys
```
  </action>
  <verify>
    <automated>cd scripts/foundation && pytest tests/test_weight_policy.py tests/test_evaluate_metrics.py -v && python -c "from fedrec_foundation.fit_metrics import FitMetricsContract, EvaluateMetricsContract, FIT_METRICS_REQUIRED_KEYS, EVAL_METRICS_REQUIRED_KEYS, validate_evaluate_metrics; c = FitMetricsContract(train_loss=0.1, num_positives=2, num_training_examples=10, hit_count_overall_at10=1, evaluated_users=1); assert c.hit_count_overall_at10 == 1 and FIT_METRICS_REQUIRED_KEYS == ('train_loss', 'num_positives', 'num_training_examples'); e = EvaluateMetricsContract(eval_loss=0.5, sampled_hr_at10=0.1, sampled_ndcg_at10=0.05, evaluated_users=1, hit_count_overall_at10=0, ndcg_sum_overall_at10=0.0); validate_evaluate_metrics(e.to_dict()); print('ok')"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "hit_count_overall_at10\|evaluated_users_sparse\|evaluated_users_dense" scripts/foundation/fedrec_foundation/fit_metrics.py` returns at least 3 matches.
    - `grep -c "FIT_METRICS_REQUIRED_KEYS = (\"train_loss\", \"num_positives\", \"num_training_examples\")" scripts/foundation/fedrec_foundation/fit_metrics.py` returns 1 (UNCHANGED).
    - `grep -c "^EVAL_METRICS_REQUIRED_KEYS" scripts/foundation/fedrec_foundation/fit_metrics.py` returns 1.
    - `grep -c "^class EvaluateMetricsContract" scripts/foundation/fedrec_foundation/fit_metrics.py` returns 1.
    - `grep -c "^def validate_evaluate_metrics" scripts/foundation/fedrec_foundation/fit_metrics.py` returns 1.
    - `pytest scripts/foundation/tests/test_weight_policy.py -v 2>&1 | grep -E "passed|failed"` shows 15 passed, 0 failed (12 original + 3 new).
    - `pytest scripts/foundation/tests/test_evaluate_metrics.py -v 2>&1 | grep -E "passed|failed"` shows 4 passed, 0 failed.
    - `python -c "from fedrec_foundation.fit_metrics import FitMetricsContract; FitMetricsContract(train_loss=0.5, num_positives=5, num_training_examples=25)"` exits 0 (backward compat).
    - `python -c "from fedrec_foundation.fit_metrics import validate_fit_metrics; validate_fit_metrics({'train_loss': 0.1, 'num_positives': 1, 'num_training_examples': 5})"` exits 0 (Phase 1 contract preserved).
    - `python -c "from fedrec_foundation.fit_metrics import EvaluateMetricsContract, validate_evaluate_metrics; e = EvaluateMetricsContract(eval_loss=0.5, sampled_hr_at10=0.1, sampled_ndcg_at10=0.05, evaluated_users=1, hit_count_overall_at10=0, ndcg_sum_overall_at10=0.0); validate_evaluate_metrics(e.to_dict())"` exits 0.
  </acceptance_criteria>
  <done>FitMetricsContract has 12 new optional sufficient-stat fields; EvaluateMetricsContract + EVAL_METRICS_REQUIRED_KEYS + validate_evaluate_metrics exist; 15 + 4 = 19 tests GREEN across test_weight_policy.py + test_evaluate_metrics.py; Phase 1 contract semantics preserved.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Create BaselineFedAvg + BaselineFedProx strategy subclasses (D-20)</name>
  <files>
    federated-baseline-cf/federated_baseline_cf/strategy.py
    federated-baseline-cf/tests/__init__.py
    federated-baseline-cf/tests/conftest.py
    federated-baseline-cf/tests/test_strategy.py
  </files>
  <read_first>
    - federated-baseline-cf/federated_baseline_cf/server_app.py (lines 1-30 for current strategy imports; lines 260-280 where FedAvg/FedProx instantiated — do NOT edit in this task, only understand)
    - federated-baseline-cf/pyproject.toml (CURRENT state — this task does NOT modify pyproject.toml; Plan 02 Task 1 owns it per BLOCKER 1 fix from iteration 1)
    - scripts/foundation/tests/conftest.py (pytest fixture patterns to mirror)
    - scripts/foundation/fedrec_foundation/fit_metrics.py (post-Task-1 extended contract — your strategy will sum these fields)
    - .planning/phases/02-baseline-migration/02-CONTEXT.md §decisions (D-20 sufficient-stat aggregation)
    - CLAUDE.md §"Code Standards" (typing pre-3.10, NumPy docstrings)
  </read_first>
  <behavior>
    - Test 1 (RED→GREEN, test_aggregate_evaluate_sums_sufficient_stats): Given 3 fake EvaluateRes results with overall sufficient stats (hit_count_overall_at10=10, ndcg_sum_overall_at10=5.0, evaluated_users=20), (hit_count_overall_at10=5, ndcg_sum_overall_at10=2.5, evaluated_users=15), (hit_count_overall_at10=7, ndcg_sum_overall_at10=3.5, evaluated_users=25), BaselineFedAvg(fraction_fit=0.1).aggregate_evaluate(1, results, []) returns (loss, metrics) where metrics["sampled_hr@10"] == (10+5+7)/(20+15+25) == 22/60 ≈ 0.3667, and metrics["sampled_ndcg@10"] == (5.0+2.5+3.5)/60 == 11.0/60 ≈ 0.1833. Aggregation is a server-side SUM, not an average of per-client ratios.
    - Test 2 (RED→GREEN, test_aggregate_evaluate_per_group_ratios): Given 2 fake EvaluateRes results with per-group stats (sparse: hit=3,ndcg=1.0,users=10; medium: hit=4,ndcg=2.0,users=8; dense: hit=5,ndcg=3.0,users=5) and (sparse: hit=1,ndcg=0.5,users=5; medium: hit=2,ndcg=1.0,users=4; dense: hit=0,ndcg=0.0,users=0), aggregated metrics include "sampled_hr@10/sparse" == 4/15, "sampled_hr@10/medium" == 6/12, "sampled_hr@10/dense" == 5/5, and "evaluated_users_sparse" == 15.
    - Test 3 (RED→GREEN, test_aggregate_evaluate_zero_division_safe): Given results where sum(evaluated_users_sparse) == 0, aggregate_evaluate returns 0.0 for "sampled_hr@10/sparse" and "sampled_ndcg@10/sparse" (no ZeroDivisionError).
    - Test 4 (RED→GREEN, test_baseline_fedprox_inherits_aggregate_evaluate): BaselineFedProx(fraction_fit=0.1, proximal_mu=0.01).aggregate_evaluate(...) behaves identically to BaselineFedAvg because it inherits the override.
    - Test 5 (RED→GREEN, test_aggregate_fit_inherited_unchanged): BaselineFedAvg.aggregate_fit is the same method object as flwr.server.strategy.FedAvg.aggregate_fit (or calls super()). The override is ONLY on aggregate_evaluate.
  </behavior>
  <action>
**Do NOT modify `federated-baseline-cf/pyproject.toml` in this task.** Plan 02 Task 1 owns the pyproject.toml — it adds the `[project.optional-dependencies] dev = ["pytest>=7.0"]` section there. This avoids the Wave 1 race where Plans 01 and 02 both write to pyproject.toml in parallel (iteration 1 BLOCKER 1 fix).

Create `federated-baseline-cf/tests/__init__.py` as an empty file (makes the tests directory a package so pytest picks up `conftest.py`).

Create `federated-baseline-cf/tests/conftest.py` with shared fixtures for strategy tests:

```python
"""Shared fixtures for federated-baseline-cf tests (Phase 2)."""
from __future__ import annotations

from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from flwr.common import EvaluateRes, Status, Code


def _make_eval_res(num_examples: int, metrics: Dict[str, float]) -> EvaluateRes:
    """Construct a Flower EvaluateRes with given num_examples + metrics."""
    return EvaluateRes(
        status=Status(code=Code.OK, message="ok"),
        loss=float(metrics.get("eval_loss", 0.0)),
        num_examples=int(num_examples),
        metrics=dict(metrics),
    )


@pytest.fixture
def fake_evaluate_res():
    """Factory fixture returning EvaluateRes builders for strategy tests."""
    return _make_eval_res


@pytest.fixture
def fake_client_proxy():
    """Minimal MagicMock ClientProxy so strategy.aggregate_evaluate can index into results."""
    proxy = MagicMock()
    proxy.cid = "test_client"
    return proxy
```

Create `federated-baseline-cf/federated_baseline_cf/strategy.py` with these two classes (type hints pre-3.10, NumPy-style docstrings per CLAUDE.md):

```python
"""Custom FedAvg/FedProx strategies for federated-baseline-cf (D-20).

BaselineFedAvg overrides aggregate_evaluate to emit thesis-table metrics
computed ONCE from summed sufficient statistics across clients, rather
than averaging per-client ratios. Per BSL-06 (the core sufficient-stat
requirement) and D-20 (custom strategy placement).

aggregate_fit is INHERITED UNCHANGED — baseline's "all params global"
invariant means no special aggregation beyond FedAvg/FedProx.

Per-group sufficient stats live in FitMetricsContract (Phase 1 + Phase 2
Plan 01 D-22 extension). Each client's @app.evaluate() handler populates
hit_count_{overall,sparse,medium,dense}_at10, ndcg_sum_..._at10, and
evaluated_users_{,sparse,medium,dense} — validate_fit_metrics continues
to check only the Phase 1 required keys; validate_evaluate_metrics (D-21)
governs the evaluate-side wire payload.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from flwr.common import EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg as BaseFedAvg, FedProx as BaseFedProx


def _sum_sufficient_stats(
    results: List[Tuple[ClientProxy, EvaluateRes]],
) -> Dict[str, Union[int, float]]:
    """Sum per-client sufficient-stat fields across all EvaluateRes responses.

    Each client returns the D-22 extended-contract fields via
    FitMetricsContract.to_dict() merged into EvaluateRes.metrics. Missing
    fields are treated as 0 (a client that reports nothing for a group
    contributes zero to that group).

    Parameters
    ----------
    results : list of (ClientProxy, EvaluateRes)
        Responses from Flower's evaluate phase.

    Returns
    -------
    Dict[str, int | float]
        Summed-across-clients totals for each stat.
    """
    stat_keys = (
        "hit_count_overall_at10", "ndcg_sum_overall_at10", "evaluated_users",
        "hit_count_sparse_at10", "ndcg_sum_sparse_at10", "evaluated_users_sparse",
        "hit_count_medium_at10", "ndcg_sum_medium_at10", "evaluated_users_medium",
        "hit_count_dense_at10", "ndcg_sum_dense_at10", "evaluated_users_dense",
    )
    totals: Dict[str, Union[int, float]] = {k: 0 for k in stat_keys}
    for _, eval_res in results:
        m = eval_res.metrics or {}
        for k in stat_keys:
            v = m.get(k, 0) or 0
            totals[k] = totals[k] + v
    return totals


def _sufficient_stats_to_thesis_metrics(
    totals: Dict[str, Union[int, float]],
) -> Dict[str, Scalar]:
    """Convert summed sufficient stats into server-side ratio metrics.

    Computes overall and per-group ``sampled_hr@10`` / ``sampled_ndcg@10``
    as ``hit_count / evaluated_users``. Zero-divison safe: a group with
    zero evaluated users gets 0.0 for both its HR and NDCG.

    Parameters
    ----------
    totals : Dict[str, int | float]
        Summed-across-clients sufficient stats from _sum_sufficient_stats.

    Returns
    -------
    Dict[str, Scalar]
        Thesis-table metrics dict with keys:
        - ``sampled_hr@10``, ``sampled_ndcg@10``
        - ``sampled_hr@10/sparse``, ``sampled_ndcg@10/sparse``
        - ``sampled_hr@10/medium``, ``sampled_ndcg@10/medium``
        - ``sampled_hr@10/dense``, ``sampled_ndcg@10/dense``
        - ``evaluated_users``, ``evaluated_users_sparse``,
          ``evaluated_users_medium``, ``evaluated_users_dense``
    """
    def _safe_ratio(num: Union[int, float], den: Union[int, float]) -> float:
        return float(num) / float(den) if den else 0.0

    metrics: Dict[str, Scalar] = {
        "sampled_hr@10": _safe_ratio(totals["hit_count_overall_at10"], totals["evaluated_users"]),
        "sampled_ndcg@10": _safe_ratio(totals["ndcg_sum_overall_at10"], totals["evaluated_users"]),
        "sampled_hr@10/sparse": _safe_ratio(totals["hit_count_sparse_at10"], totals["evaluated_users_sparse"]),
        "sampled_ndcg@10/sparse": _safe_ratio(totals["ndcg_sum_sparse_at10"], totals["evaluated_users_sparse"]),
        "sampled_hr@10/medium": _safe_ratio(totals["hit_count_medium_at10"], totals["evaluated_users_medium"]),
        "sampled_ndcg@10/medium": _safe_ratio(totals["ndcg_sum_medium_at10"], totals["evaluated_users_medium"]),
        "sampled_hr@10/dense": _safe_ratio(totals["hit_count_dense_at10"], totals["evaluated_users_dense"]),
        "sampled_ndcg@10/dense": _safe_ratio(totals["ndcg_sum_dense_at10"], totals["evaluated_users_dense"]),
        "evaluated_users": int(totals["evaluated_users"]),
        "evaluated_users_sparse": int(totals["evaluated_users_sparse"]),
        "evaluated_users_medium": int(totals["evaluated_users_medium"]),
        "evaluated_users_dense": int(totals["evaluated_users_dense"]),
    }
    return metrics


class BaselineFedAvg(BaseFedAvg):
    """FedAvg variant with sufficient-stat aggregate_evaluate (D-20, BSL-06).

    aggregate_fit is INHERITED UNCHANGED — baseline's all-params-global
    invariant is satisfied by parent FedAvg. Only aggregate_evaluate is
    overridden so headline ``sampled_hr@10`` / ``sampled_ndcg@10`` are
    computed ONCE from summed sufficient stats (not averaged per-client
    ratios, which silently double-counts sparse users).
    """

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Sum sufficient stats across clients, compute server-side ratios once.

        Parameters
        ----------
        server_round : int
            Current FL round number.
        results : list of (ClientProxy, EvaluateRes)
            Successful client evaluations.
        failures : list
            Failed/errored clients.

        Returns
        -------
        Tuple[Optional[float], Dict[str, Scalar]]
            ``(overall_loss, thesis_metrics_dict)``. ``overall_loss`` is
            the num-examples-weighted mean of per-client eval_loss
            (Flower convention). ``thesis_metrics_dict`` has the keys
            listed in :func:`_sufficient_stats_to_thesis_metrics`.
        """
        if not results:
            return None, {}
        totals = _sum_sufficient_stats(results)
        thesis_metrics = _sufficient_stats_to_thesis_metrics(totals)
        # Loss aggregation: num-examples-weighted mean (Flower convention).
        total_examples = sum(int(r.num_examples) for _, r in results) or 1
        loss = sum(float(r.loss) * int(r.num_examples) for _, r in results) / total_examples
        return float(loss), thesis_metrics


class BaselineFedProx(BaseFedProx):
    """FedProx variant that reuses BaselineFedAvg's aggregate_evaluate (D-20).

    Mirrors BaselineFedAvg's override exactly — inheritance via the same
    _sum_sufficient_stats / _sufficient_stats_to_thesis_metrics helpers.
    aggregate_fit is inherited from parent FedProx (proximal term is
    client-side, aggregation is still FedAvg).
    """

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Delegate to the same sufficient-stat aggregation as BaselineFedAvg."""
        if not results:
            return None, {}
        totals = _sum_sufficient_stats(results)
        thesis_metrics = _sufficient_stats_to_thesis_metrics(totals)
        total_examples = sum(int(r.num_examples) for _, r in results) or 1
        loss = sum(float(r.loss) * int(r.num_examples) for _, r in results) / total_examples
        return float(loss), thesis_metrics


__all__ = ["BaselineFedAvg", "BaselineFedProx"]
```

Then create `federated-baseline-cf/tests/test_strategy.py` with the 5 tests specified in <behavior>. Use the exact test names. All should be RED on first run (NotImplementedError-equivalent — strategy.py does not exist yet); after implementing strategy.py they must be GREEN. Run `pytest federated-baseline-cf/tests/test_strategy.py -v` to confirm.

```python
"""Unit tests for BaselineFedAvg/BaselineFedProx sufficient-stat aggregation (Phase 2 Plan 01)."""
from __future__ import annotations

import pytest

from federated_baseline_cf.strategy import BaselineFedAvg, BaselineFedProx


def test_aggregate_evaluate_sums_sufficient_stats(fake_evaluate_res, fake_client_proxy) -> None:
    strategy = BaselineFedAvg(fraction_fit=0.1)
    results = [
        (fake_client_proxy, fake_evaluate_res(20, {
            "eval_loss": 0.5,
            "hit_count_overall_at10": 10, "ndcg_sum_overall_at10": 5.0, "evaluated_users": 20,
            "hit_count_sparse_at10": 0, "ndcg_sum_sparse_at10": 0.0, "evaluated_users_sparse": 0,
            "hit_count_medium_at10": 0, "ndcg_sum_medium_at10": 0.0, "evaluated_users_medium": 0,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.0, "evaluated_users_dense": 0,
        })),
        (fake_client_proxy, fake_evaluate_res(15, {
            "eval_loss": 0.6,
            "hit_count_overall_at10": 5, "ndcg_sum_overall_at10": 2.5, "evaluated_users": 15,
            "hit_count_sparse_at10": 0, "ndcg_sum_sparse_at10": 0.0, "evaluated_users_sparse": 0,
            "hit_count_medium_at10": 0, "ndcg_sum_medium_at10": 0.0, "evaluated_users_medium": 0,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.0, "evaluated_users_dense": 0,
        })),
        (fake_client_proxy, fake_evaluate_res(25, {
            "eval_loss": 0.4,
            "hit_count_overall_at10": 7, "ndcg_sum_overall_at10": 3.5, "evaluated_users": 25,
            "hit_count_sparse_at10": 0, "ndcg_sum_sparse_at10": 0.0, "evaluated_users_sparse": 0,
            "hit_count_medium_at10": 0, "ndcg_sum_medium_at10": 0.0, "evaluated_users_medium": 0,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.0, "evaluated_users_dense": 0,
        })),
    ]
    _loss, metrics = strategy.aggregate_evaluate(1, results, [])
    assert metrics["sampled_hr@10"] == pytest.approx(22.0 / 60.0)
    assert metrics["sampled_ndcg@10"] == pytest.approx(11.0 / 60.0)
    assert metrics["evaluated_users"] == 60


def test_aggregate_evaluate_per_group_ratios(fake_evaluate_res, fake_client_proxy) -> None:
    strategy = BaselineFedAvg(fraction_fit=0.1)
    results = [
        (fake_client_proxy, fake_evaluate_res(23, {
            "eval_loss": 0.5,
            "hit_count_overall_at10": 12, "ndcg_sum_overall_at10": 6.0, "evaluated_users": 23,
            "hit_count_sparse_at10": 3, "ndcg_sum_sparse_at10": 1.0, "evaluated_users_sparse": 10,
            "hit_count_medium_at10": 4, "ndcg_sum_medium_at10": 2.0, "evaluated_users_medium": 8,
            "hit_count_dense_at10": 5, "ndcg_sum_dense_at10": 3.0, "evaluated_users_dense": 5,
        })),
        (fake_client_proxy, fake_evaluate_res(9, {
            "eval_loss": 0.5,
            "hit_count_overall_at10": 3, "ndcg_sum_overall_at10": 1.5, "evaluated_users": 9,
            "hit_count_sparse_at10": 1, "ndcg_sum_sparse_at10": 0.5, "evaluated_users_sparse": 5,
            "hit_count_medium_at10": 2, "ndcg_sum_medium_at10": 1.0, "evaluated_users_medium": 4,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.0, "evaluated_users_dense": 0,
        })),
    ]
    _loss, metrics = strategy.aggregate_evaluate(1, results, [])
    assert metrics["sampled_hr@10/sparse"] == pytest.approx(4.0 / 15.0)
    assert metrics["sampled_hr@10/medium"] == pytest.approx(6.0 / 12.0)
    assert metrics["sampled_hr@10/dense"] == pytest.approx(5.0 / 5.0)
    assert metrics["evaluated_users_sparse"] == 15
    assert metrics["evaluated_users_medium"] == 12
    assert metrics["evaluated_users_dense"] == 5


def test_aggregate_evaluate_zero_division_safe(fake_evaluate_res, fake_client_proxy) -> None:
    strategy = BaselineFedAvg(fraction_fit=0.1)
    results = [
        (fake_client_proxy, fake_evaluate_res(5, {
            "eval_loss": 0.5,
            "hit_count_overall_at10": 2, "ndcg_sum_overall_at10": 1.0, "evaluated_users": 5,
            "hit_count_sparse_at10": 0, "ndcg_sum_sparse_at10": 0.0, "evaluated_users_sparse": 0,
            "hit_count_medium_at10": 2, "ndcg_sum_medium_at10": 1.0, "evaluated_users_medium": 5,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.0, "evaluated_users_dense": 0,
        })),
    ]
    _loss, metrics = strategy.aggregate_evaluate(1, results, [])
    assert metrics["sampled_hr@10/sparse"] == 0.0
    assert metrics["sampled_ndcg@10/sparse"] == 0.0
    assert metrics["sampled_hr@10/dense"] == 0.0


def test_baseline_fedprox_inherits_aggregate_evaluate(fake_evaluate_res, fake_client_proxy) -> None:
    strategy = BaselineFedProx(fraction_fit=0.1, proximal_mu=0.01)
    results = [
        (fake_client_proxy, fake_evaluate_res(10, {
            "eval_loss": 0.5,
            "hit_count_overall_at10": 3, "ndcg_sum_overall_at10": 1.5, "evaluated_users": 10,
            "hit_count_sparse_at10": 0, "ndcg_sum_sparse_at10": 0.0, "evaluated_users_sparse": 0,
            "hit_count_medium_at10": 0, "ndcg_sum_medium_at10": 0.0, "evaluated_users_medium": 0,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.0, "evaluated_users_dense": 0,
        })),
    ]
    _loss, metrics = strategy.aggregate_evaluate(1, results, [])
    assert metrics["sampled_hr@10"] == pytest.approx(3.0 / 10.0)


def test_aggregate_fit_inherited_unchanged() -> None:
    """aggregate_fit must NOT be overridden — baseline = all params global."""
    from flwr.server.strategy import FedAvg as _FedAvg
    assert BaselineFedAvg.aggregate_fit is _FedAvg.aggregate_fit, (
        "BaselineFedAvg MUST inherit aggregate_fit unchanged from FedAvg "
        "(D-23: baseline keeps all params global)"
    )
```
  </action>
  <verify>
    <automated>pip install -e federated-baseline-cf/ && cd federated-baseline-cf && pytest tests/test_strategy.py -v</automated>
  </verify>
  <acceptance_criteria>
    - File exists: `federated-baseline-cf/federated_baseline_cf/strategy.py` (confirm via `ls`).
    - `grep -n "class BaselineFedAvg(BaseFedAvg)" federated-baseline-cf/federated_baseline_cf/strategy.py` returns 1 match.
    - `grep -n "class BaselineFedProx(BaseFedProx)" federated-baseline-cf/federated_baseline_cf/strategy.py` returns 1 match.
    - `grep -n "def aggregate_evaluate" federated-baseline-cf/federated_baseline_cf/strategy.py` returns 2 matches (one per subclass).
    - `grep -c "def aggregate_fit" federated-baseline-cf/federated_baseline_cf/strategy.py` returns 0 (inheritance preserved).
    - `python -c "from federated_baseline_cf.strategy import BaselineFedAvg, BaselineFedProx; assert callable(BaselineFedAvg.aggregate_evaluate) and callable(BaselineFedProx.aggregate_evaluate); print('ok')"` exits 0.
    - `pytest federated-baseline-cf/tests/test_strategy.py -v 2>&1 | grep -E "passed|failed"` shows 5 passed, 0 failed.
    - `federated-baseline-cf/tests/conftest.py` exists with `fake_evaluate_res` and `fake_client_proxy` fixtures.
    - `git diff --name-only federated-baseline-cf/pyproject.toml | wc -l` returns 0 for THIS task (pyproject.toml is owned by Plan 02 Task 1).
  </acceptance_criteria>
  <done>BaselineFedAvg + BaselineFedProx ship with 5 GREEN tests; aggregate_fit inheritance preserved; pyproject.toml untouched (Plan 02 owns dev-dep declaration).</done>
</task>

</tasks>

<verification>
Full-phase verification for Plan 01:

1. `pytest scripts/foundation/tests/test_weight_policy.py -v` — shows 15 passed (12 Phase 1 + 3 Phase 2).
2. `pytest scripts/foundation/tests/test_evaluate_metrics.py -v` — shows 4 passed.
3. `pytest federated-baseline-cf/tests/test_strategy.py -v` — shows 5 passed.
4. Contract backward compat: `python -c "from fedrec_foundation.fit_metrics import FitMetricsContract, validate_fit_metrics, FIT_METRICS_REQUIRED_KEYS; assert FIT_METRICS_REQUIRED_KEYS == ('train_loss','num_positives','num_training_examples'); c = FitMetricsContract(train_loss=0.1, num_positives=1, num_training_examples=5); assert c.hit_count_sparse_at10 is None; validate_fit_metrics(c.to_dict()); print('ok')"` exits 0.
5. EvaluateMetricsContract smoke: `python -c "from fedrec_foundation.fit_metrics import EvaluateMetricsContract, validate_evaluate_metrics, EVAL_METRICS_REQUIRED_KEYS; e = EvaluateMetricsContract(eval_loss=0.5, sampled_hr_at10=0.1, sampled_ndcg_at10=0.05, evaluated_users=1, hit_count_overall_at10=0, ndcg_sum_overall_at10=0.0); validate_evaluate_metrics(e.to_dict()); print('ok')"` exits 0.
6. Strategy smoke test: `python -c "from federated_baseline_cf.strategy import BaselineFedAvg, BaselineFedProx; assert BaselineFedAvg.aggregate_fit.__qualname__ == 'FedAvg.aggregate_fit'; print('ok')"` exits 0 (inheritance check at runtime).
7. Wave-1 write-race invariant: `git diff --name-only federated-baseline-cf/pyproject.toml` shows NO diff attributable to Plan 01 (Plan 02 Task 1 owns pyproject.toml).
</verification>

<success_criteria>
- FitMetricsContract has 12 new optional per-group/overall sufficient-stat fields; `FIT_METRICS_REQUIRED_KEYS` unchanged; `validate_fit_metrics` semantics unchanged (Phase 1 contract intact).
- EvaluateMetricsContract + `EVAL_METRICS_REQUIRED_KEYS` (3 required sufficient-stat keys: `hit_count_overall_at10`, `ndcg_sum_overall_at10`, `evaluated_users` — unified with FitMetricsContract + strategy `_sum_sufficient_stats` reader) + 3 optional diagnostic fields (`eval_loss`, `sampled_hr_at10`, `sampled_ndcg_at10` — cached client-side; NOT consumed for aggregation) + 9 optional per-group fields + `validate_evaluate_metrics(payload)` enforce D-21 strict-contract (rejects free-form extras) + D-22 per-group fields. Iteration 2 unified-naming fix: field names now match the strategy reader exactly, preventing the silent-zero-metrics bug from iteration 1 (dict keys on the wire must match dict keys read by `_sum_sufficient_stats`).
- `BaselineFedAvg(FedAvg)` and `BaselineFedProx(FedProx)` exist; each overrides `aggregate_evaluate` to compute `sampled_hr@10` / `sampled_ndcg@10` (overall + per-group) as sum-based ratios.
- `aggregate_fit` is inherited unchanged (D-23 preserved).
- 15 + 4 + 5 = 24 GREEN tests across `test_weight_policy.py` + `test_evaluate_metrics.py` + `test_strategy.py`.
- pyproject.toml UNTOUCHED by this plan (Plan 02 Task 1 owns dev-dep declaration — resolves iteration 1 Wave-1 write race).
- Pre-existing uncommitted hunks in `federated-baseline-cf/federated_baseline_cf/{dataset,client_app,server_app,task}.py` are UNTOUCHED (D-18 surgical migration). Executor runs `git diff federated-baseline-cf/` before committing to verify no drift.
</success_criteria>

<output>
After completion, create `.planning/phases/02-baseline-migration/02-baseline-migration-01-SUMMARY.md` following the template in `@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md`.
</output>
</content>
</invoke>