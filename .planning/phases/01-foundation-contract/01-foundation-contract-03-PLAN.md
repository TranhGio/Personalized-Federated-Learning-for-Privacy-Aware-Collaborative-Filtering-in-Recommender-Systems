---
phase: 01-foundation-contract
plan: 03
type: execute
wave: 2
depends_on: [01-foundation-contract-01]
files_modified:
  - scripts/foundation/fedrec_foundation/evaluator.py
  - scripts/foundation/fedrec_foundation/weight_policy.py
  - scripts/foundation/fedrec_foundation/fit_metrics.py
  - scripts/foundation/tests/test_evaluator.py
  - scripts/foundation/tests/test_weight_policy.py
autonomous: true
requirements: [FND-04, FND-05]
must_haves:
  truths:
    - "Downstream code can import `EvalProtocol.SAMPLED_LOO_99.value` as the primary evaluator string — ONE authoritative constant, not scattered string literals."
    - "Downstream code can import `WeightPolicy` enum and `compute_aggregation_weight(metrics, policy)` and get one consistent float back for `uniform` / `num_positives` / `num_training_examples`."
    - "Every downstream client module has a `FitMetricsContract` dataclass it MUST populate in its `@app.train()` return value (CR-4), covering `train_loss`, `num_positives`, `num_training_examples`, plus optional per-module extensions."
    - "`compute_aggregation_weight({'num_positives': 10}, 'num_positives') == 10.0` and unknown policies raise ValueError."
  artifacts:
    - path: "scripts/foundation/fedrec_foundation/evaluator.py"
      provides: "EvalProtocol enum + get_primary_evaluator(mode) string resolver"
      exports: ["EvalProtocol", "get_primary_evaluator"]
    - path: "scripts/foundation/fedrec_foundation/weight_policy.py"
      provides: "WeightPolicy enum + compute_aggregation_weight(metrics, policy) -> float"
      exports: ["WeightPolicy", "compute_aggregation_weight"]
    - path: "scripts/foundation/fedrec_foundation/fit_metrics.py"
      provides: "FitMetricsContract dataclass — the client-side fit-metrics contract (CR-4)"
      exports: ["FitMetricsContract", "FIT_METRICS_REQUIRED_KEYS", "validate_fit_metrics"]
  key_links:
    - from: "scripts/foundation/fedrec_foundation/weight_policy.py::compute_aggregation_weight"
      to: "FitMetricsContract keys: num_positives, num_training_examples"
      via: "CR-4 contract: policies consume what clients are required to emit"
      pattern: "num_positives"
    - from: "scripts/foundation/fedrec_foundation/fit_metrics.py"
      to: "@dataclass FitMetricsContract"
      via: "codebase dataclass-first convention"
      pattern: "@dataclass"
---

<objective>
Implement FND-04 (primary evaluator selector) and FND-05 (aggregation weight policy abstraction + client-side FitMetricsContract dataclass per Codex CR-4). These three tiny modules are the contracts Phases 2–5 will implement against.

Purpose: Without `FitMetricsContract` the weight-policy enum is non-functional — CR-4 flagged that clients currently return only `num-examples` and `num_positives` / `num_training_examples` aren't emitted anywhere. Phase 1 MUST provide the dataclass + the validator so Phases 2–5 have a fixed import surface to populate.

Output: Three fully-implemented Python modules with passing tests that Plans 02 and 06 depend on (via integration tests).
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/01-foundation-contract/01-CONTEXT.md
@.planning/phases/01-foundation-contract/01-RESEARCH.md
@.planning/phases/01-foundation-contract/01-VALIDATION.md
@CLAUDE.md
@.planning/codebase/CONVENTIONS.md

<interfaces>
From scripts/foundation/fedrec_foundation/evaluator.py:
```python
from enum import Enum

class EvalProtocol(str, Enum):
    SAMPLED_LOO_99 = "sampled_loo_99"   # D-12 primary
    ALLRANK = "allrank"                  # D-12 secondary, namespaced

def get_primary_evaluator(mode: str) -> str: ...   # returns SAMPLED_LOO_99.value for all three modes
```

From scripts/foundation/fedrec_foundation/weight_policy.py:
```python
from enum import Enum
from typing import Dict, Union

class WeightPolicy(str, Enum):
    UNIFORM = "uniform"
    NUM_POSITIVES = "num_positives"
    NUM_TRAINING_EXAMPLES = "num_training_examples"

def compute_aggregation_weight(
    client_metrics: Dict[str, Union[int, float]],
    policy: str,
) -> float: ...
```

From scripts/foundation/fedrec_foundation/fit_metrics.py (CR-4 new):
```python
from dataclasses import dataclass, asdict
from typing import Dict, Union, Optional

FIT_METRICS_REQUIRED_KEYS = ("train_loss", "num_positives", "num_training_examples")

@dataclass
class FitMetricsContract:
    train_loss: float
    num_positives: int              # count of positive train samples for this user / client
    num_training_examples: int      # total train sample count (pos + neg) for this user / client
    # Per-module extension fields below (optional; modules add their own).
    round_num: Optional[int] = None

    def to_dict(self) -> Dict[str, Union[int, float]]: ...   # returns FitRes-ready dict
    @classmethod
    def from_dict(cls, d: Dict[str, Union[int, float]]) -> "FitMetricsContract": ...

def validate_fit_metrics(metrics: Dict[str, Union[int, float]]) -> None:
    """Raises ValueError if any FIT_METRICS_REQUIRED_KEYS is missing or of wrong type."""
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Implement FND-04 evaluator selector + flip test_evaluator green</name>
  <files>
    scripts/foundation/fedrec_foundation/evaluator.py
    scripts/foundation/tests/test_evaluator.py
  </files>
  <read_first>
    - .planning/phases/01-foundation-contract/01-CONTEXT.md D-12 (primary evaluator locked to sampled_loo_99)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"Pattern 4: Primary Evaluator Selector (FND-04)" (implementation lines 667-684)
    - CLAUDE.md (typing, docstrings)
  </read_first>
  <behavior>
    - `EvalProtocol` is a `str, Enum` with values `SAMPLED_LOO_99` = `"sampled_loo_99"` (D-12 primary) and `ALLRANK` = `"allrank"` (D-12 secondary, namespaced).
    - `get_primary_evaluator(mode: str) -> str` returns `EvalProtocol.SAMPLED_LOO_99.value` for every known mode. Unknown mode raises `ValueError`.
    - test_evaluator.py flips green: one test that asserts all three modes return `"sampled_loo_99"`.
  </behavior>
  <action>
Create `scripts/foundation/fedrec_foundation/evaluator.py` by copying the research Pattern 4 code verbatim (lines 667-684) and adding a NumPy-style docstring + module header comment. Keep the `ALLRANK` enum entry but do NOT let `get_primary_evaluator` return it — it exists only so downstream code can prefix secondary metrics with `EvalProtocol.ALLRANK.value`.

Add an explicit mode-whitelist for safety:
```python
_KNOWN_MODES = frozenset({"benchmark_cross_device", "paper_compat_pfedrec", "cross_silo_legacy"})

def get_primary_evaluator(mode: str) -> str:
    if mode not in _KNOWN_MODES:
        raise ValueError(f"Unknown mode {mode!r}. Expected one of {sorted(_KNOWN_MODES)}.")
    return EvalProtocol.SAMPLED_LOO_99.value
```

Flip `tests/test_evaluator.py` to GREEN:
```python
"""Tests for fedrec_foundation.evaluator (FND-04)."""
from __future__ import annotations

import pytest

from fedrec_foundation.evaluator import EvalProtocol, get_primary_evaluator


def test_primary_evaluator_all_modes() -> None:
    for mode in ("benchmark_cross_device", "paper_compat_pfedrec", "cross_silo_legacy"):
        assert get_primary_evaluator(mode) == "sampled_loo_99"
        assert get_primary_evaluator(mode) == EvalProtocol.SAMPLED_LOO_99.value


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        get_primary_evaluator("not_a_mode")


def test_allrank_is_namespaced() -> None:
    # ALLRANK exists but is not returned by get_primary_evaluator; it's only
    # useful as a metric prefix for namespaced secondary metrics.
    assert EvalProtocol.ALLRANK.value == "allrank"
```
  </action>
  <verify>
    <automated>cd scripts/foundation &amp;&amp; pytest tests/test_evaluator.py -v</automated>
  </verify>
  <acceptance_criteria>
    - File `scripts/foundation/fedrec_foundation/evaluator.py` defines `EvalProtocol`, `get_primary_evaluator`.
    - `grep "SAMPLED_LOO_99" scripts/foundation/fedrec_foundation/evaluator.py` matches.
    - `cd scripts/foundation && pytest tests/test_evaluator.py -v` prints 3 passed.
  </acceptance_criteria>
  <done>FND-04 implemented; primary evaluator string is a single constant.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Implement FND-05 weight-policy + FitMetricsContract (CR-4) + flip test_weight_policy green</name>
  <files>
    scripts/foundation/fedrec_foundation/weight_policy.py
    scripts/foundation/fedrec_foundation/fit_metrics.py
    scripts/foundation/tests/test_weight_policy.py
  </files>
  <read_first>
    - .planning/phases/01-foundation-contract/01-CONTEXT.md (Claude's Discretion: baseline/personalized/adaptive default to num_positives)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md CODEX PEER REVIEW §CR-4 (FitMetricsContract LOCKED)
    - .planning/phases/01-foundation-contract/01-RESEARCH.md §"Pattern 5: Weight-Policy Abstraction (FND-05)" (lines 691-738)
    - federated-baseline-cf/federated_baseline_cf/client_app.py (existing `return {"num-examples": len(trainloader.dataset), ...}` — what Phases 2-5 will extend to populate FitMetricsContract)
    - federated-pfedrec/federated_pfedrec/client_app.py (same pattern; per CR-4 clients only emit num-examples today)
    - CLAUDE.md (dataclass-first config, typing.Dict, NumPy-style docstrings)
  </read_first>
  <behavior>
    - `WeightPolicy` enum with values `UNIFORM`, `NUM_POSITIVES`, `NUM_TRAINING_EXAMPLES`.
    - `compute_aggregation_weight(metrics, policy)`:
      - `UNIFORM` returns `1.0`.
      - `NUM_POSITIVES` returns `float(metrics["num_positives"])`; raises `ValueError` with message including "num_positives" if key missing.
      - `NUM_TRAINING_EXAMPLES` returns `float(metrics["num_training_examples"])`; raises `ValueError` if key missing.
      - Unknown policy raises `ValueError` with message "Unknown weight policy: {policy}".
    - `FitMetricsContract` dataclass with fields `train_loss: float`, `num_positives: int`, `num_training_examples: int`, `round_num: Optional[int] = None` (extensible — downstream modules add their own fields in subclasses). `to_dict()` returns `asdict(self)`; `from_dict(d)` picks known keys only (ignores extras so downstream extensions don't break).
    - `FIT_METRICS_REQUIRED_KEYS = ("train_loss", "num_positives", "num_training_examples")` — exported constant.
    - `validate_fit_metrics(metrics)` checks every required key is present and the right type (int/float); raises `ValueError` otherwise.
    - Tests flip green: `test_num_positives`, `test_unknown_policy_raises`, `test_fit_metrics_contract`.
  </behavior>
  <action>
Create `scripts/foundation/fedrec_foundation/weight_policy.py` by copying research Pattern 5 (lines 691-731) verbatim. Use `typing.Dict`, `typing.Union`. Add NumPy-style docstrings.

Create `scripts/foundation/fedrec_foundation/fit_metrics.py`:
```python
"""Client-side fit-metrics contract (FND-05 + Codex CR-4).

Every federated module's @app.train() handler MUST return a dict whose
keys include FIT_METRICS_REQUIRED_KEYS, so the weight-policy abstraction
in weight_policy.py has the inputs it needs. Modules construct a
FitMetricsContract, call .to_dict(), and merge with per-module metrics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional, Union

FIT_METRICS_REQUIRED_KEYS = ("train_loss", "num_positives", "num_training_examples")


@dataclass
class FitMetricsContract:
    """Minimum metrics a federated client MUST return from @app.train().

    Phases 2-5 populate this in each module's client_app.py train handler.
    Weight-policy resolution consumes num_positives / num_training_examples.

    Attributes
    ----------
    train_loss : float
        Final training loss (after last local epoch).
    num_positives : int
        Count of positive train samples for this client (for WeightPolicy.NUM_POSITIVES).
    num_training_examples : int
        Total train sample count including negatives (for WeightPolicy.NUM_TRAINING_EXAMPLES).
    round_num : Optional[int]
        Current round number (optional; some modules log it here, some in FitRes).
    """
    train_loss: float
    num_positives: int
    num_training_examples: int
    round_num: Optional[int] = None

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Return a dict suitable for a Flower FitRes.metrics merge.

        None values are dropped so downstream aggregators don't see null.
        """
        raw = asdict(self)
        return {k: v for k, v in raw.items() if v is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, Union[int, float]]) -> "FitMetricsContract":
        """Construct from a dict, ignoring unknown keys (forward-compat)."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)  # type: ignore[arg-type]


def validate_fit_metrics(metrics: Dict[str, Union[int, float]]) -> None:
    """Raise ValueError if metrics dict is missing a required FitMetricsContract key.

    Parameters
    ----------
    metrics : Dict[str, int | float]
        The client's returned FitRes.metrics.

    Raises
    ------
    ValueError
        If any required key is missing or is not an int/float.
    """
    for key in FIT_METRICS_REQUIRED_KEYS:
        if key not in metrics:
            raise ValueError(
                f"Client metrics missing required key {key!r}. "
                f"Expected at least {list(FIT_METRICS_REQUIRED_KEYS)}; "
                f"got {sorted(metrics.keys())}."
            )
        if not isinstance(metrics[key], (int, float)):
            raise ValueError(
                f"Client metrics[{key!r}] must be int|float, "
                f"got {type(metrics[key]).__name__}."
            )
```

Flip `tests/test_weight_policy.py` to GREEN:
```python
"""Tests for fedrec_foundation.weight_policy (FND-05) + fit_metrics (CR-4)."""
from __future__ import annotations

import pytest

from fedrec_foundation.weight_policy import WeightPolicy, compute_aggregation_weight
from fedrec_foundation.fit_metrics import (
    FitMetricsContract, FIT_METRICS_REQUIRED_KEYS, validate_fit_metrics,
)


def test_num_positives() -> None:
    assert compute_aggregation_weight({"num_positives": 10}, "num_positives") == 10.0
    assert compute_aggregation_weight({"num_positives": 0}, "num_positives") == 0.0


def test_num_training_examples() -> None:
    assert compute_aggregation_weight({"num_training_examples": 123}, "num_training_examples") == 123.0


def test_uniform_ignores_metrics() -> None:
    assert compute_aggregation_weight({}, "uniform") == 1.0


def test_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="num_positives"):
        compute_aggregation_weight({"num_training_examples": 5}, "num_positives")


def test_unknown_policy_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        compute_aggregation_weight({"num_positives": 1}, "made_up_policy")


def test_fit_metrics_contract() -> None:
    c = FitMetricsContract(train_loss=0.5, num_positives=20, num_training_examples=100, round_num=3)
    d = c.to_dict()
    assert d["train_loss"] == 0.5
    assert d["num_positives"] == 20
    assert d["num_training_examples"] == 100
    assert d["round_num"] == 3
    # validate_fit_metrics is happy with this dict.
    validate_fit_metrics(d)


def test_fit_metrics_contract_none_dropped() -> None:
    c = FitMetricsContract(train_loss=0.1, num_positives=1, num_training_examples=2)
    d = c.to_dict()
    assert "round_num" not in d  # None dropped


def test_fit_metrics_contract_forward_compat() -> None:
    d = {"train_loss": 0.1, "num_positives": 1, "num_training_examples": 2, "extra": "future_field"}
    c = FitMetricsContract.from_dict(d)
    assert c.train_loss == 0.1
    # Unknown keys ignored (don't crash).


def test_validate_fit_metrics_missing_raises() -> None:
    with pytest.raises(ValueError, match="num_positives"):
        validate_fit_metrics({"train_loss": 0.1, "num_training_examples": 10})


def test_validate_fit_metrics_wrong_type_raises() -> None:
    with pytest.raises(ValueError, match="int.*float"):
        validate_fit_metrics(
            {"train_loss": "oops", "num_positives": 1, "num_training_examples": 2}
        )


def test_required_keys_constant() -> None:
    assert FIT_METRICS_REQUIRED_KEYS == ("train_loss", "num_positives", "num_training_examples")
```

Also update Plan 01's test stub `tests/test_weight_policy.py` if any remnant remains — the module-level `pytest.mark.skip` is removed.
  </action>
  <verify>
    <automated>cd scripts/foundation &amp;&amp; pytest tests/test_weight_policy.py -v</automated>
  </verify>
  <acceptance_criteria>
    - File `scripts/foundation/fedrec_foundation/weight_policy.py` defines `WeightPolicy` enum with three members and `compute_aggregation_weight`.
    - File `scripts/foundation/fedrec_foundation/fit_metrics.py` defines `FitMetricsContract`, `FIT_METRICS_REQUIRED_KEYS`, `validate_fit_metrics`.
    - `grep "num_positives" scripts/foundation/fedrec_foundation/fit_metrics.py` matches.
    - `grep "@dataclass" scripts/foundation/fedrec_foundation/fit_metrics.py` matches.
    - `cd scripts/foundation && pytest tests/test_weight_policy.py -v` prints at least 10 passed (tests listed in action).
    - `python -c "from fedrec_foundation.fit_metrics import FitMetricsContract, validate_fit_metrics; c=FitMetricsContract(train_loss=0.1, num_positives=5, num_training_examples=10); validate_fit_metrics(c.to_dict())"` succeeds.
  </acceptance_criteria>
  <done>FND-05 implemented; weight-policy enum + aggregation weight function work; FitMetricsContract dataclass locks the client-side fit-metrics contract per CR-4.</done>
</task>

</tasks>

<verification>
- `cd scripts/foundation && pytest tests/test_evaluator.py tests/test_weight_policy.py -v` — all tests pass.
- `python -c "from fedrec_foundation.evaluator import get_primary_evaluator; from fedrec_foundation.weight_policy import compute_aggregation_weight; from fedrec_foundation.fit_metrics import FitMetricsContract; print(get_primary_evaluator('benchmark_cross_device')); print(compute_aggregation_weight({'num_positives': 5}, 'num_positives'))"` prints `sampled_loo_99` then `5.0`.
</verification>

<success_criteria>
- FND-04: `EvalProtocol.SAMPLED_LOO_99.value == "sampled_loo_99"` is the single authoritative primary-evaluator constant.
- FND-05: `compute_aggregation_weight(metrics, policy)` resolves to the right float per policy; unknown policy raises `ValueError`.
- CR-4: `FitMetricsContract` dataclass with `num_positives` + `num_training_examples` fields is available as the client-side fit-metrics contract Phases 2-5 populate.
</success_criteria>

<output>
After completion, create `.planning/phases/01-foundation-contract/01-foundation-contract-03-SUMMARY.md` — document the three modules' public APIs, note that Phases 2-5 must import `FitMetricsContract` and `validate_fit_metrics` in each client_app.py's `@app.train()` return path (this is the cross-phase contract).
</output>
