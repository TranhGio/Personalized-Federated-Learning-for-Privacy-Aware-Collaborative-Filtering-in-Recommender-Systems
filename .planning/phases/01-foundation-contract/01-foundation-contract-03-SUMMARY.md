---
phase: 01-foundation-contract
plan: 03
subsystem: infra
tags: [enum, dataclass, str-enum, aggregation-weight, evaluator-selector, fit-metrics-contract, cr-4, fnd-04, fnd-05, wave-2, tdd]

requires:
  - phase: 01-foundation-contract
    provides: "Installable fedrec-foundation package + skip-stub tests at scripts/foundation/tests/test_evaluator.py and test_weight_policy.py"
provides:
  - "fedrec_foundation.evaluator: EvalProtocol(str, Enum) with SAMPLED_LOO_99 + ALLRANK, plus get_primary_evaluator(mode) that whitelists {benchmark_cross_device, paper_compat_pfedrec, cross_silo_legacy} (FND-04)"
  - "fedrec_foundation.weight_policy: WeightPolicy(str, Enum) with UNIFORM / NUM_POSITIVES / NUM_TRAINING_EXAMPLES, plus compute_aggregation_weight(metrics, policy) -> float (FND-05)"
  - "fedrec_foundation.fit_metrics: @dataclass FitMetricsContract (train_loss, num_positives, num_training_examples, round_num optional), FIT_METRICS_REQUIRED_KEYS tuple, validate_fit_metrics(d), FitMetricsContract.from_dict(d) with TypeError -> ValueError wrapping (CR-4)"
  - "15 GREEN tests across test_evaluator.py (3) + test_weight_policy.py (12). Module-level pytestmark skip REMOVED from both files."
affects: [02-foundation-contract, 04-foundation-contract, 05-foundation-contract, 06-foundation-contract, 02-baseline, 03-pfedrec, 04-personalized, 05-adaptive]

tech-stack:
  added: []
  patterns:
    - "str-Enum pattern for config-level constants: value is a string matching the public constant; avoids scattered literals across modules."
    - "Dataclass contract + from_dict forward-compat: from_dict filters unknown keys so downstream modules can add new metrics without breaking deserialization."
    - "TypeError-to-ValueError wrapping: dataclass __init__ TypeErrors are re-raised as ValueError('...missing required field: ...') so callers see a clear error surface."
    - "Closed-enum whitelist for mode strings: factory-style resolver raises ValueError on typos; matches CONVENTIONS.md error handling rule."

key-files:
  created:
    - "scripts/foundation/fedrec_foundation/evaluator.py"
    - "scripts/foundation/fedrec_foundation/weight_policy.py"
    - "scripts/foundation/fedrec_foundation/fit_metrics.py"
  modified:
    - "scripts/foundation/tests/test_evaluator.py (un-skipped, 3 tests)"
    - "scripts/foundation/tests/test_weight_policy.py (un-skipped + expanded, 12 tests)"

key-decisions:
  - "Mode-whitelist on get_primary_evaluator rather than silent fallback — typos in config must fail loud, matching CONVENTIONS.md's 'factory functions raise ValueError on unknown enum strings' rule."
  - "FitMetricsContract.from_dict wraps dataclass TypeError as ValueError with 'missing required field' prefix — Codex CR-4 demanded a clear error surface; callers now get a message with the field name instead of a cryptic 'FitMetricsContract.__init__() missing N required positional arguments' TypeError."
  - "to_dict() drops None values so optional fields like round_num don't pollute FitRes.metrics with nulls downstream aggregators would need special-cased."
  - "Kept ALLRANK in the enum even though get_primary_evaluator never returns it — downstream phases use EvalProtocol.ALLRANK.value as a metric-name PREFIX (e.g., allrank_ndcg@10) so secondary metrics are namespaced and never accidentally mixed into thesis tables."
  - "validate_fit_metrics performs type check (int|float) on each required key — prevents a broken client emitting metrics={'train_loss': 'oops', ...} from silently contaminating weight-policy arithmetic later in the round."

patterns-established:
  - "Config-level constant + resolver: Phase 1 ships the CONSTANT string plus a resolver function; per-module wiring is Phase 2-5 work. Downstream modules import EvalProtocol.SAMPLED_LOO_99.value (never the bare string literal)."
  - "Weight-policy contract with explicit error messages that echo both the expected keys and the observed keys — makes debugging a missing-metric report easy when a client accidentally drops num_positives from its FitRes."
  - "Dataclass-backed contract with @dataclass fields as the single source of truth for required keys — FIT_METRICS_REQUIRED_KEYS tuple is exported for explicit use but is documentation, not a second source of truth (the real required set is fields(FitMetricsContract) with default==MISSING)."

requirements-completed: [FND-04, FND-05]

duration: "3 min"
completed: "2026-04-19"
---

# Phase 01 Plan 03: FND-04 Primary Evaluator Selector + FND-05 Weight Policy + CR-4 FitMetricsContract Summary

**Three tiny foundation modules — EvalProtocol / WeightPolicy enums plus a FitMetricsContract dataclass with TypeError-to-ValueError wrapping — that Phases 2-5 must import in every `@app.train()` return path to lock the cross-phase fit-metrics contract.**

## Performance

- **Duration:** ~3 min (154 seconds)
- **Started:** 2026-04-19T03:10:06Z
- **Completed:** 2026-04-19T03:12:40Z
- **Tasks:** 2 (both TDD, completed autonomously)
- **Files created:** 3 (evaluator.py, weight_policy.py, fit_metrics.py)
- **Files modified:** 2 (test_evaluator.py, test_weight_policy.py — un-skipped + expanded)

## Accomplishments

- **FND-04 shipped.** `EvalProtocol.SAMPLED_LOO_99.value` (`"sampled_loo_99"`) is now the single authoritative primary-evaluator string, with `get_primary_evaluator(mode)` as the resolver. Phases 2-5 no longer have to re-declare this literal.
- **FND-05 shipped.** `WeightPolicy` enum + `compute_aggregation_weight(metrics, policy)` resolve UNIFORM/NUM_POSITIVES/NUM_TRAINING_EXAMPLES to a weight float with clear ValueErrors on unknown policy or missing required key.
- **CR-4 satisfied.** `FitMetricsContract` is the fixed import surface for the client-side fit-metrics contract; `from_dict({})` now raises `ValueError: FitMetricsContract.from_dict missing required field: ...` instead of a cryptic `TypeError` — exactly the polish Codex CR-4 demanded.
- **15 tests GREEN** (up from 5 skipped). Module-level `pytestmark = pytest.mark.skip(...)` removed from both test files; the phase's "-ra" pytest flag will no longer surface these as pending.
- **Zero regressions.** `pytest tests/test_evaluator.py tests/test_weight_policy.py -v` prints 15 passed, 0 failed.

## Task Commits

Each task used a strict TDD RED -> GREEN cycle; all commits use `--no-verify` per the parallel-execution rules (Wave 2 orchestrator validates hooks after all siblings complete).

1. **Task 1 RED: add failing tests for FND-04 evaluator selector** - `37dcb6a` (test)
2. **Task 1 GREEN: implement FND-04 primary evaluator selector** - `bd30e44` (feat)
3. **Task 2 RED: add failing tests for FND-05 weight policy + CR-4 FitMetricsContract** - `aafe435` (test)
4. **Task 2 GREEN: implement FND-05 weight policy + CR-4 FitMetricsContract** - `07f18ca` (feat)

_Plan metadata commit (SUMMARY.md + STATE.md + ROADMAP.md + REQUIREMENTS.md) is appended separately after the sibling Wave 2 agents finish._

## Files Created/Modified

### New modules

- `scripts/foundation/fedrec_foundation/evaluator.py` - `EvalProtocol(str, Enum)` with `SAMPLED_LOO_99` and `ALLRANK`; `_KNOWN_MODES` frozenset (`benchmark_cross_device`, `paper_compat_pfedrec`, `cross_silo_legacy`); `get_primary_evaluator(mode)` whitelists, raises `ValueError("Unknown mode ...")` on typos, returns `SAMPLED_LOO_99.value` for every recognized mode.
- `scripts/foundation/fedrec_foundation/weight_policy.py` - `WeightPolicy(str, Enum)` with three members; `compute_aggregation_weight(client_metrics, policy)` dispatches UNIFORM -> 1.0, NUM_POSITIVES -> `float(metrics["num_positives"])`, NUM_TRAINING_EXAMPLES -> `float(metrics["num_training_examples"])`. Unknown policy -> `ValueError("Unknown weight policy: ...")`. Missing required key -> `ValueError` echoing both the expected key and `sorted(metrics.keys())`.
- `scripts/foundation/fedrec_foundation/fit_metrics.py` - `@dataclass FitMetricsContract` with `train_loss: float`, `num_positives: int`, `num_training_examples: int`, optional `round_num: Optional[int] = None`; `to_dict()` returns `asdict(self)` with `None` values filtered out; `from_dict(d)` filters unknown keys (forward-compat) then wraps `TypeError` from `cls(**filtered)` as `ValueError("FitMetricsContract.from_dict missing required field: ...") from e`; `FIT_METRICS_REQUIRED_KEYS` tuple exported; `validate_fit_metrics(d)` verifies presence + `int|float` type for every required key.

### Tests flipped GREEN

- `scripts/foundation/tests/test_evaluator.py` - Module-level `pytestmark.skip` deleted. 3 tests: `test_primary_evaluator_all_modes` (all three modes -> `"sampled_loo_99"`), `test_unknown_mode_raises` (ValueError on typo), `test_allrank_is_namespaced` (`EvalProtocol.ALLRANK.value == "allrank"`).
- `scripts/foundation/tests/test_weight_policy.py` - Module-level `pytestmark.skip` deleted. 12 tests: original 4 (`test_num_positives`, `test_unknown_policy_raises`, `test_fit_metrics_contract`, `test_from_dict_missing_required_raises`) plus 8 additions covering `test_num_training_examples`, `test_uniform_ignores_metrics`, `test_missing_key_raises`, `test_fit_metrics_contract_none_dropped`, `test_fit_metrics_contract_forward_compat`, `test_validate_fit_metrics_missing_raises`, `test_validate_fit_metrics_wrong_type_raises`, `test_required_keys_constant`.

## Cross-Phase Contract — IMPORTANT for Phases 2-5

**Every module's `client_app.py::@app.train()` return path MUST use `FitMetricsContract.to_dict()` to build its returned metrics dict.** Raw dict literals that emit only `num-examples` (as Codex CR-4 flagged) will break the weight-policy abstraction downstream.

Recommended usage in every Phase 2-5 client:

```python
from fedrec_foundation.fit_metrics import FitMetricsContract
from fedrec_foundation.weight_policy import compute_aggregation_weight  # server-side

# In client_app.py @app.train():
metrics = FitMetricsContract(
    train_loss=final_loss,
    num_positives=n_pos,
    num_training_examples=n_pos + n_neg,
    round_num=current_round,
).to_dict()
# merge with per-module metrics (e.g., user_prototype, alpha, grad_norm) then return.
```

And every Phase 2-5 server's strategy aggregator should call `validate_fit_metrics(fit_res.metrics)` before calling `compute_aggregation_weight(...)`; a broken client that drops required keys will then raise a clear ValueError (`"Client metrics missing required key 'num_positives'..."`) instead of silently contaminating the aggregation weight arithmetic.

## Decisions Made

- **Closed-enum whitelist on `get_primary_evaluator`** — typos in run config must fail loud (ValueError), not silently default. Matches CONVENTIONS.md: "Raise ValueError from factory functions on unknown enum strings". Example existing pattern: `create_alpha_computer`.
- **`to_dict()` drops None values** — optional `round_num=None` would serialize as a JSON null, which downstream Flower aggregators would need to special-case. Dropping it is cheaper and safer.
- **`from_dict` is forward-compatible** — unknown keys are filtered, not rejected. Lets Phase 2-5 modules emit per-module extension metrics (e.g., `alpha`, `contrastive_loss`, `item_perturbation_norm`) alongside the contract fields without breaking `FitMetricsContract.from_dict`.
- **`from_dict` wraps TypeError as ValueError with "missing required field" prefix** — Codex CR-4 polish; the underlying dataclass TypeError is caught and re-raised with a clearer message containing the field name. See `test_from_dict_missing_required_raises` for both empty-dict and partial-dict coverage.
- **`validate_fit_metrics` type-checks every required key (int|float)** — cheaply prevents a broken client returning `metrics={"train_loss": "nan"}` from silently contaminating weight-policy arithmetic later in the round.
- **`ALLRANK` retained in the enum even though `get_primary_evaluator` never returns it** — downstream phases use `EvalProtocol.ALLRANK.value` as a metric-name PREFIX (e.g., `allrank_ndcg@10`) so secondary metrics are namespaced and never accidentally mixed into thesis tables (D-12).

## Deviations from Plan

None - plan executed exactly as written.

Both tasks followed their `<action>` blocks verbatim, including the TDD RED -> GREEN sequence, the `try/except TypeError -> ValueError` wrapping pattern in `from_dict`, and the 12 tests enumerated in Task 2's `<action>` block. Typing matches CONVENTIONS.md (`typing.Dict`, `typing.Union`, `typing.Optional` — NOT PEP 604). No auto-fixes applied; no blocking issues encountered.

## Issues Encountered

None. Every automated verify command passed on first run:

- Task 1 RED: `pytest tests/test_evaluator.py` correctly failed with `ModuleNotFoundError: No module named 'fedrec_foundation.evaluator'`.
- Task 1 GREEN: `pytest tests/test_evaluator.py -v` prints 3 passed in 0.01s.
- Task 2 RED: `pytest tests/test_weight_policy.py` correctly failed with `ModuleNotFoundError: No module named 'fedrec_foundation.weight_policy'`.
- Task 2 GREEN: `pytest tests/test_weight_policy.py -v` prints 12 passed in 0.01s.
- Smoke test: `python -c "from fedrec_foundation.evaluator import get_primary_evaluator; from fedrec_foundation.weight_policy import compute_aggregation_weight; from fedrec_foundation.fit_metrics import FitMetricsContract; print(get_primary_evaluator('benchmark_cross_device')); print(compute_aggregation_weight({'num_positives': 5}, 'num_positives'))"` prints `sampled_loo_99` then `5.0`.
- CR-4 error-handling smoke test: `python -c "from fedrec_foundation.fit_metrics import FitMetricsContract; FitMetricsContract.from_dict({})"` fails with `ValueError: FitMetricsContract.from_dict missing required field: FitMetricsContract.__init__() missing 3 required positional arguments: 'train_loss', 'num_positives', and 'num_training_examples'` — exactly the clear ValueError CR-4 demanded.

Collection errors in sibling-plan test files (`test_split.py`, `test_exclusion.py`, `test_mode.py`) during full-suite `pytest tests/ -v` are **outside my territory** — Plans 02 and 05 are creating those modules in parallel and will be green before the orchestrator's post-wave hook validation. Per the parallel_execution rules, I do NOT edit those files.

## User Setup Required

None - no external service configuration required. All three modules are pure-Python with no new third-party dependencies (numpy/pandas already declared in Plan 01's pyproject.toml; nothing new added in Plan 03).

## Next Phase Readiness

**Ready for Plans 02, 04, 05, 06** (sibling Wave 2 + subsequent waves) — the FND-04 / FND-05 / CR-4 contracts are now importable from `fedrec_foundation.evaluator`, `fedrec_foundation.weight_policy`, `fedrec_foundation.fit_metrics`.

**Ready for Phases 2-5** — every module's `client_app.py::@app.train()` handler now has a fixed import surface for fit-metrics:

- Import: `from fedrec_foundation.fit_metrics import FitMetricsContract`
- Populate per-client: `FitMetricsContract(train_loss=..., num_positives=..., num_training_examples=..., round_num=...)`
- Serialize: `.to_dict()` into the FitRes.metrics dict
- Server-side validation: `validate_fit_metrics(fit_res.metrics)` before `compute_aggregation_weight(fit_res.metrics, policy_from_run_config)`

**No blockers.** No architectural decisions deferred. No follow-up work required for Plan 03 scope.

## Self-Check: PASSED

- **Files created:**
  - FOUND: `scripts/foundation/fedrec_foundation/evaluator.py`
  - FOUND: `scripts/foundation/fedrec_foundation/weight_policy.py`
  - FOUND: `scripts/foundation/fedrec_foundation/fit_metrics.py`
- **Files modified:**
  - FOUND (un-skipped + 3 tests): `scripts/foundation/tests/test_evaluator.py`
  - FOUND (un-skipped + 12 tests): `scripts/foundation/tests/test_weight_policy.py`
- **Commits:**
  - FOUND: `37dcb6a` (Task 1 RED), `bd30e44` (Task 1 GREEN), `aafe435` (Task 2 RED), `07f18ca` (Task 2 GREEN)
  - Verified via `git log --oneline -5` on `feat/try_to_run_the_baseline`.
- **Automated verify:** PASSED. `pytest tests/test_evaluator.py tests/test_weight_policy.py -v` -> 15 passed in 0.01s. Smoke tests print `sampled_loo_99` + `5.0`. `from_dict({})` raises `ValueError("...missing required field: ...")` (NOT a TypeError).
- **Acceptance criteria matrix:**
  - `grep "SAMPLED_LOO_99" scripts/foundation/fedrec_foundation/evaluator.py` — 5 matches (import+enum+docstring+return) ✓
  - `grep "num_positives" scripts/foundation/fedrec_foundation/fit_metrics.py` — matches ✓
  - `grep "@dataclass" scripts/foundation/fedrec_foundation/fit_metrics.py` — matches ✓
  - `grep "missing required field" scripts/foundation/fedrec_foundation/fit_metrics.py` — matches ✓

---

*Phase: 01-foundation-contract*
*Plan: 03*
*Completed: 2026-04-19*
