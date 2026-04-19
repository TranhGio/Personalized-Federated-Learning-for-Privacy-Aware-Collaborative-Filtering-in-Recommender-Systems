---
phase: 02-baseline-migration
plan: 01
type: tdd
wave: 1
depends_on: []
files_modified:
  - scripts/foundation/fedrec_foundation/fit_metrics.py
  - scripts/foundation/tests/test_weight_policy.py
  - federated-baseline-cf/federated_baseline_cf/strategy.py
  - federated-baseline-cf/tests/__init__.py
  - federated-baseline-cf/tests/conftest.py
  - federated-baseline-cf/tests/test_strategy.py
  - federated-baseline-cf/pyproject.toml
autonomous: true
requirements:
  - BSL-06

must_haves:
  truths:
    - "FitMetricsContract accepts per-group sufficient-stat keys (hit_count_sparse@10, evaluated_users_sparse, ndcg_sum_sparse@10, hit_count_medium@10, evaluated_users_medium, ndcg_sum_medium@10, hit_count_dense@10, evaluated_users_dense, ndcg_sum_dense@10, hit_count@10, evaluated_users, ndcg_sum@10) as dataclass fields; validate_fit_metrics still checks original required keys plus the new per-group keys."
    - "BaselineFedAvg(FedAvg) and BaselineFedProx(FedProx) subclasses exist in federated_baseline_cf/strategy.py; each exposes aggregate_evaluate(server_round, results, failures) that sums sufficient stats across clients and returns (overall_loss, {'sampled_hr@10': hit_count/evaluated_users, 'sampled_ndcg@10': ndcg_sum/evaluated_users, 'sampled_hr@10/sparse': ..., 'sampled_ndcg@10/sparse': ..., 'sampled_hr@10/medium': ..., 'sampled_ndcg@10/medium': ..., 'sampled_hr@10/dense': ..., 'sampled_ndcg@10/dense': ..., 'evaluated_users': ..., 'evaluated_users_sparse': ..., 'evaluated_users_medium': ..., 'evaluated_users_dense': ...})."
    - "Aggregation does NOT average per-client ratios — it returns sum(hit_count) / sum(evaluated_users). The strategy's aggregate_fit() method is INHERITED unchanged from flwr.server.strategy.FedAvg / FedProx."
    - "pytest federated-baseline-cf/tests/test_strategy.py -v exits 0 with at least 3 GREEN tests."
  artifacts:
    - path: "scripts/foundation/fedrec_foundation/fit_metrics.py"
      provides: "Extended FitMetricsContract with per-group sufficient-stat fields"
      contains: "hit_count_sparse"
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
---

<objective>
Ship the contract extension and strategy scaffold that BSL-06 (sufficient-stat aggregation, D-20, D-21, D-22) depends on. This plan is a blocking Wave 1 gate for Plans 03 and 04.

Purpose: Under D-22 the `FitMetricsContract` is the single source of truth for everything that crosses the client/server boundary. Per-group sufficient stats must be first-class fields of the contract, not free-form metrics extras. Per D-20, baseline aggregation lives in a `BaselineFedAvg(FedAvg)` subclass that overrides `aggregate_evaluate` to compute headline metrics ONCE from summed sufficient statistics (server-side ratio), rather than averaging per-client ratios (which silently double-counts sparse users and produces mis-weighted thesis-table numbers). Plan 04 instantiates this strategy; Plan 03 populates these contract fields client-side; extending the contract in Wave 1 unblocks both.

Output: (1) Extended `FitMetricsContract` with 12 new per-group + overall sufficient-stat fields — backwards-compatible (all new fields are `Optional` with `None` default). (2) New `federated_baseline_cf/strategy.py` with `BaselineFedAvg(FedAvg)` and `BaselineFedProx(FedProx)` that override `aggregate_evaluate` to emit the correct server-side ratio. (3) pytest test harness in `federated-baseline-cf/tests/` mirroring Phase 1 layout (conftest.py fixtures + per-module tests). (4) pyproject.toml `pytest` dev dep declared under `[project.optional-dependencies]`.
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
<!-- Current FitMetricsContract (Phase 1 Plan 03). The extension below ADDS optional per-group + overall sufficient-stat fields. -->

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
  <name>Task 1: Extend FitMetricsContract with per-group sufficient-stat fields (D-22)</name>
  <files>
    scripts/foundation/fedrec_foundation/fit_metrics.py
    scripts/foundation/tests/test_weight_policy.py
  </files>
  <read_first>
    - scripts/foundation/fedrec_foundation/fit_metrics.py (ENTIRE file — current dataclass shape, from_dict filter behavior, validate_fit_metrics type check)
    - scripts/foundation/tests/test_weight_policy.py (ENTIRE file — existing 12 tests; DO NOT remove or re-label any)
    - scripts/foundation/fedrec_foundation/user_groups.py (user_group strings "sparse"/"medium"/"dense")
    - .planning/phases/02-baseline-migration/02-CONTEXT.md §decisions (D-22 per-group sufficient stats)
    - CLAUDE.md §"Code Standards" (typing pre-3.10: `typing.Dict`, `typing.Optional`, `typing.Union`; NumPy-style docstrings)
  </read_first>
  <behavior>
    - Test 1 (RED→GREEN, test_fit_metrics_per_group_fields): FitMetricsContract(train_loss=0.5, num_positives=30, num_training_examples=150, round_num=3, hit_count_overall_at10=24, ndcg_sum_overall_at10=12.5, evaluated_users=24, hit_count_sparse_at10=6, ndcg_sum_sparse_at10=2.0, evaluated_users_sparse=8, hit_count_medium_at10=10, ndcg_sum_medium_at10=5.0, evaluated_users_medium=10, hit_count_dense_at10=8, ndcg_sum_dense_at10=5.5, evaluated_users_dense=6).to_dict() returns a dict with all 15 keys present and None-dropped.
    - Test 2 (RED→GREEN, test_fit_metrics_per_group_optional): FitMetricsContract(train_loss=0.5, num_positives=30, num_training_examples=150).to_dict() STILL works (does not raise), and the 12 new per-group fields default to None and are DROPPED by to_dict — i.e., backward-compatible.
    - Test 3 (RED→GREEN, test_fit_metrics_forward_compat_with_per_group_extension): from_dict({"train_loss": 0.1, "num_positives": 2, "num_training_examples": 10, "hit_count_overall_at10": 1, "ndcg_sum_overall_at10": 0.63, "evaluated_users": 1, "alpha": 0.42}) succeeds (unknown "alpha" is filtered; known per-group "hit_count_overall_at10" is populated; missing per-group fields remain None).
    - Existing 12 tests in test_weight_policy.py remain GREEN with zero edits.
  </behavior>
  <action>
Extend `scripts/foundation/fedrec_foundation/fit_metrics.py` by adding 12 optional per-group and overall sufficient-stat fields to the `FitMetricsContract` dataclass, keeping all existing fields and behavior identical. Do NOT add the new keys to `FIT_METRICS_REQUIRED_KEYS`. Do NOT change `validate_fit_metrics` semantics — it validates only the original three required keys (train_loss, num_positives, num_training_examples).

Use TDD: first run `pytest scripts/foundation/tests/test_weight_policy.py -v` to confirm 12 tests GREEN baseline. Then add the 3 new tests (test_fit_metrics_per_group_fields, test_fit_metrics_per_group_optional, test_fit_metrics_forward_compat_with_per_group_extension) to `scripts/foundation/tests/test_weight_policy.py`, confirm they FAIL (ModuleAttributeError / TypeError on unexpected kwargs), then extend the dataclass to make them GREEN.

Exact field additions to `FitMetricsContract` (all `Optional[Union[int, float]] = None`; name them with `_at10` suffix instead of `@10` because `@` breaks Python identifier rules):

```python
from typing import Dict, Optional, Union

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

Update the module docstring with a single paragraph noting: "Phase 2 extension: per-group (sparse/medium/dense) and overall sufficient-stat fields (hit_count_*, ndcg_sum_*, evaluated_users*) are OPTIONAL and default to None. They are populated by per-module clients in Phase 2 Plan 03 and aggregated server-side via module-specific `BaselineFedAvg.aggregate_evaluate` (Phase 2 Plan 01). `validate_fit_metrics` continues to check only the Phase 1 required keys."

Also append the 3 new tests to `scripts/foundation/tests/test_weight_policy.py`. Do NOT modify any of the existing 12 tests. Use exact test names:

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
  </action>
  <verify>
    <automated>cd scripts/foundation && pytest tests/test_weight_policy.py -v && python -c "from fedrec_foundation.fit_metrics import FitMetricsContract, FIT_METRICS_REQUIRED_KEYS; c = FitMetricsContract(train_loss=0.1, num_positives=2, num_training_examples=10, hit_count_overall_at10=1, evaluated_users=1); assert c.hit_count_overall_at10 == 1 and FIT_METRICS_REQUIRED_KEYS == ('train_loss', 'num_positives', 'num_training_examples'); print('ok')"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "hit_count_overall_at10\|evaluated_users_sparse\|evaluated_users_dense" scripts/foundation/fedrec_foundation/fit_metrics.py` returns at least 3 matches.
    - `grep -c "FIT_METRICS_REQUIRED_KEYS = (\"train_loss\", \"num_positives\", \"num_training_examples\")" scripts/foundation/fedrec_foundation/fit_metrics.py` returns 1 (UNCHANGED).
    - `pytest scripts/foundation/tests/test_weight_policy.py -v 2>&1 | grep -E "passed|failed"` shows 15 passed, 0 failed (12 original + 3 new).
    - `python -c "from fedrec_foundation.fit_metrics import FitMetricsContract; FitMetricsContract(train_loss=0.5, num_positives=5, num_training_examples=25)"` exits 0 (backward compat).
    - `python -c "from fedrec_foundation.fit_metrics import validate_fit_metrics; validate_fit_metrics({'train_loss': 0.1, 'num_positives': 1, 'num_training_examples': 5})"` exits 0 (Phase 1 contract preserved).
  </acceptance_criteria>
  <done>FitMetricsContract has 12 new optional sufficient-stat fields; 15/15 tests GREEN; Phase 1 contract semantics preserved.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Create BaselineFedAvg + BaselineFedProx strategy subclasses (D-20)</name>
  <files>
    federated-baseline-cf/federated_baseline_cf/strategy.py
    federated-baseline-cf/tests/__init__.py
    federated-baseline-cf/tests/conftest.py
    federated-baseline-cf/tests/test_strategy.py
    federated-baseline-cf/pyproject.toml
  </files>
  <read_first>
    - federated-baseline-cf/federated_baseline_cf/server_app.py (lines 1-30 for current strategy imports; lines 260-280 where FedAvg/FedProx instantiated — do NOT edit in this task, only understand)
    - federated-baseline-cf/pyproject.toml (project.dependencies list)
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
First, add `pytest` as a dev dependency to `federated-baseline-cf/pyproject.toml` under an `[project.optional-dependencies]` section. Only add if it does not already exist. Do NOT re-order or modify existing `[project] dependencies`. Insert immediately after the `[project]` block:

```toml
[project.optional-dependencies]
dev = ["pytest>=7.0"]
```

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
to check only the Phase 1 required keys.
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
    - `grep "pytest" federated-baseline-cf/pyproject.toml` shows at least 1 match.
  </acceptance_criteria>
  <done>BaselineFedAvg + BaselineFedProx ship with 5 GREEN tests; aggregate_fit inheritance preserved; pytest dev dep declared.</done>
</task>

</tasks>

<verification>
Full-phase verification for Plan 01:

1. `pytest scripts/foundation/tests/test_weight_policy.py -v` — shows 15 passed (12 Phase 1 + 3 Phase 2).
2. `pytest federated-baseline-cf/tests/test_strategy.py -v` — shows 5 passed.
3. Contract backward compat: `python -c "from fedrec_foundation.fit_metrics import FitMetricsContract, validate_fit_metrics, FIT_METRICS_REQUIRED_KEYS; assert FIT_METRICS_REQUIRED_KEYS == ('train_loss','num_positives','num_training_examples'); c = FitMetricsContract(train_loss=0.1, num_positives=1, num_training_examples=5); assert c.hit_count_sparse_at10 is None; validate_fit_metrics(c.to_dict()); print('ok')"` exits 0.
4. Strategy smoke test: `python -c "from federated_baseline_cf.strategy import BaselineFedAvg, BaselineFedProx; assert BaselineFedAvg.aggregate_fit.__qualname__ == 'FedAvg.aggregate_fit'; print('ok')"` exits 0 (inheritance check at runtime).
</verification>

<success_criteria>
- FitMetricsContract has 12 new optional per-group/overall sufficient-stat fields; `FIT_METRICS_REQUIRED_KEYS` unchanged; `validate_fit_metrics` semantics unchanged (Phase 1 contract intact).
- `BaselineFedAvg(FedAvg)` and `BaselineFedProx(FedProx)` exist; each overrides `aggregate_evaluate` to compute `sampled_hr@10` / `sampled_ndcg@10` (overall + per-group) as sum-based ratios.
- `aggregate_fit` is inherited unchanged (D-23 preserved).
- 15 + 5 = 20 GREEN tests across `scripts/foundation/tests/test_weight_policy.py` + `federated-baseline-cf/tests/test_strategy.py`.
- pytest dev dep declared in `federated-baseline-cf/pyproject.toml`.
- Pre-existing uncommitted hunks in `federated-baseline-cf/federated_baseline_cf/{dataset,client_app,server_app,task}.py` are UNTOUCHED (D-18 surgical migration). Executor runs `git diff federated-baseline-cf/` before committing to verify no drift.
</success_criteria>

<output>
After completion, create `.planning/phases/02-baseline-migration/02-baseline-migration-01-SUMMARY.md` following the template in `@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md`.
</output>
