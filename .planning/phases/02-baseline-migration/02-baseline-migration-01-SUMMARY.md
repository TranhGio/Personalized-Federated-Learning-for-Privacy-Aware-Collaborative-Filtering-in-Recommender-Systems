---
phase: 02-baseline-migration
plan: 01
subsystem: infra
tags: [fit-metrics-contract, evaluate-metrics-contract, sufficient-stats, per-group-metrics, baseline-fedavg, baseline-fedprox, strategy-subclass, aggregate-evaluate, d-20, d-21, d-22, bsl-06, tdd, pytest, flower, wave-1]

requires:
  - phase: 01-foundation-contract-03
    provides: "FitMetricsContract + FIT_METRICS_REQUIRED_KEYS + validate_fit_metrics (FND-04 + FND-05 + CR-4)"
  - phase: 01-foundation-contract-06
    provides: "fedrec-foundation as plain-name local-path dep in federated-baseline-cf/pyproject.toml — foundation imports resolve at module-level without sys.path hacks"

provides:
  - "FitMetricsContract extended with 12 OPTIONAL per-group + overall sufficient-stat fields (hit_count_{overall,sparse,medium,dense}_at10, ndcg_sum_{overall,sparse,medium,dense}_at10, evaluated_users{,_sparse,_medium,_dense}). FIT_METRICS_REQUIRED_KEYS unchanged — Phase 1 contract backward-compatible."
  - "EvaluateMetricsContract + EVAL_METRICS_REQUIRED_KEYS (3 required sufficient-stat keys) + validate_evaluate_metrics(payload) enforce D-21 strict-contract (rejects free-form extras) + D-22 per-group semantics on the evaluate-wire payload."
  - "BaselineFedAvg(FedAvg) + BaselineFedProx(FedProx) subclasses in federated_baseline_cf/strategy.py; each overrides aggregate_evaluate to compute thesis metrics ONCE from summed sufficient stats (sum(hit_count)/sum(evaluated_users)) rather than averaging per-client ratios. aggregate_fit is INHERITED UNCHANGED (D-23)."
  - "Module-level helpers _sum_sufficient_stats + _sufficient_stats_to_thesis_metrics share the same 12-key reader as EvaluateMetricsContract dict keys (unified _at10 naming — no _at_10 drift)."
  - "federated-baseline-cf/tests/ pytest package with conftest.py fixtures (fake_evaluate_res factory, fake_client_proxy MagicMock) + 5 GREEN strategy tests."
  - "scripts/foundation/tests/test_evaluate_metrics.py (+4 GREEN tests) + 3 new FitMetricsContract extension tests appended to tests/test_weight_policy.py (12 existing untouched)."

affects: [02-baseline-migration-03, 02-baseline-migration-04, 03-personalized-migration, 04-adaptive-migration, 05-pfedrec-migration]

tech-stack:
  added: []
  patterns:
    - "Strict-contract wire payload: dataclass with required + optional fields, to_dict() drops None, from_dict() filters unknown keys and wraps TypeError as ValueError (CR-4 pattern extended to EvaluateMetricsContract). A top-level validate_*_metrics function enforces required keys AND rejects free-form extras (D-21)."
    - "Sufficient-stat aggregation: client returns per-group (hit_count_{group}_at10, ndcg_sum_{group}_at10, evaluated_users_{group}) sufficient stats; server sums across clients and divides once — avoids the sparse-user double-counting that per-client-ratio averaging silently does."
    - "Unified field-name discipline: wire-payload dict keys, dataclass field names, and server-side _sum_sufficient_stats reader all share the same _at10 suffix (no _at_10 drift, no @10 Python-identifier violation). Iteration 2 fix — prior drift would have zeroed all evaluate-side metrics in production."
    - "Strategy subclass placement: module-specific aggregate_evaluate override lives in each module's own strategy.py (D-20); aggregate_fit stays inherited unchanged to preserve each module's parameter-split invariant (baseline = all global, D-23)."
    - "pytest fixture pattern for Flower strategy tests: conftest.py provides a fake_evaluate_res factory (takes num_examples + metrics dict, returns EvaluateRes with Status OK) and a fake_client_proxy MagicMock — minimal deps, no Grid or SuperLink required."

key-files:
  created:
    - "scripts/foundation/tests/test_evaluate_metrics.py (+4 tests for EvaluateMetricsContract + validate_evaluate_metrics)"
    - "federated-baseline-cf/federated_baseline_cf/strategy.py (BaselineFedAvg + BaselineFedProx + module-level _sum_sufficient_stats / _sufficient_stats_to_thesis_metrics)"
    - "federated-baseline-cf/tests/__init__.py (empty — makes tests a package)"
    - "federated-baseline-cf/tests/conftest.py (fake_evaluate_res + fake_client_proxy fixtures)"
    - "federated-baseline-cf/tests/test_strategy.py (+5 tests for BaselineFedAvg / BaselineFedProx sufficient-stat aggregation)"
  modified:
    - "scripts/foundation/fedrec_foundation/fit_metrics.py (extended FitMetricsContract with 12 Optional per-group/overall fields + new EvaluateMetricsContract + EVAL_METRICS_REQUIRED_KEYS + validate_evaluate_metrics; module docstring updated)"
    - "scripts/foundation/tests/test_weight_policy.py (+3 FitMetricsContract extension tests appended; 12 existing tests untouched)"

key-decisions:
  - "Unified _at10 suffix (no underscore between 'at' and 10) across EvaluateMetricsContract fields, FitMetricsContract fields, the strategy's _sum_sufficient_stats reader, and the client wire payload — iteration 2 lock-in that prevents the silent-zero-metrics bug where key-name drift between the wire and the reader would have made all evaluate-side aggregation produce zeros while unit tests passed (tests constructed strategy-shape dicts directly)."
  - "EvaluateMetricsContract is a SEPARATE dataclass sibling to FitMetricsContract, not a subclass. The evaluate wire payload carries OPTIONAL diagnostic keys (eval_loss / sampled_hr_at10 / sampled_ndcg_at10) that are NOT required on the fit side, so inheriting FIT_METRICS_REQUIRED_KEYS would either over-constrain the evaluate side or let free-form extras slip through validate_fit_metrics."
  - "validate_evaluate_metrics(payload) enforces both required-keys AND no-free-form-extras (D-21). validate_fit_metrics semantics unchanged — it continues to check only the Phase 1 required keys (train_loss, num_positives, num_training_examples) for type + presence; the evaluate side has its own strict checker."
  - "Diagnostic keys (eval_loss, sampled_hr_at10, sampled_ndcg_at10) are cached client-side for per-round logs only; the server aggregator IGNORES them and re-computes the headline ratios from summed sufficient stats. This is why they're OPTIONAL in the contract — they're informational, not a source of truth."
  - "aggregate_fit INHERITED UNCHANGED from flwr.server.strategy.FedAvg / FedProx. Baseline's 'all params global' invariant (D-23) is satisfied by parent FedAvg; the custom strategy only exists to fix the evaluation-side aggregation. Enforced by test_aggregate_fit_inherited_unchanged (BaselineFedAvg.aggregate_fit is FedAvg.aggregate_fit identity check)."
  - "BaselineFedProx.aggregate_evaluate is an EXACT COPY of BaselineFedAvg.aggregate_evaluate rather than a call to super(). Rationale: FedProx.aggregate_evaluate inherits from FedAvg, but making BaselineFedProx extend BaselineFedAvg would create a diamond-inheritance pattern (BaselineFedProx -> BaselineFedAvg + BaseFedProx). The duplication is 4 lines; the call-chain clarity is worth it. Both use the same module-level _sum_sufficient_stats + _sufficient_stats_to_thesis_metrics helpers, so the logic stays DRY."
  - "pyproject.toml UNTOUCHED by this plan — Plan 02 Task 1 exclusively owns the [project.optional-dependencies] dev section. Wave-1 write race avoided: Plans 01 and 02 run in parallel with zero overlapping files (iteration 1 BLOCKER 1 fix)."
  - "Pre-existing uncommitted hunks in federated-baseline-cf/federated_baseline_cf/{client_app,dataset,server_app,task}.py UNTOUCHED (D-18 surgical migration). Those files carry prior work (Phase 1 pre-write) that Plan 03 Task 2 consumes during the actual client-side sufficient-stat population."

patterns-established:
  - "Strict-contract dataclass pair: FitMetricsContract (train-side) + EvaluateMetricsContract (evaluate-side), each with their own required-keys frozenset and validate_*_metrics checker. Plans 3-5 will mirror this pair-pattern for their module-specific extensions."
  - "Sufficient-stat aggregation helper split: two module-level functions (_sum_sufficient_stats across clients, then _sufficient_stats_to_thesis_metrics converts totals to ratios) instead of one monolithic method. Both strategies reuse the same helpers; future modules (personalized / adaptive / pfedrec) can import them or mirror the same split."
  - "pytest tests/ package per federated module, rooted at federated-<module>-cf/tests/, with an __init__.py + conftest.py. Phase 2 plans 3/4 extend this tree; phases 3-5 mirror it in their own modules."
  - "Strategy subclass naming: Baseline<X> per D-20 — explicit module-branding rather than sharing a module name across modules. Phase 3's counterpart will be PersonalizedFedAvg / PersonalizedFedProx; Phase 4's will use SplitFedAvg / SplitFedProx (already established in the existing federated-adaptive-personalized-cf/strategy.py)."

requirements-completed:
  - BSL-06

metrics:
  duration: "~6 min"
  started: "2026-04-19T07:45:37Z"
  completed: "2026-04-19T07:51:29Z"
  tasks_completed: 2
  files_created: 5
  files_modified: 2
  tests_added: 12  # 3 fit-metrics extension + 4 evaluate-metrics + 5 strategy
  tests_green_foundation: 77  # was 70; +3 fit-metrics extension +4 evaluate-metrics
  tests_green_baseline: 5  # new federated-baseline-cf/tests/
---

# Phase 02 Plan 01: FitMetricsContract extension + EvaluateMetricsContract + BaselineFedAvg/FedProx Strategy (D-20, D-21, D-22, BSL-06) Summary

**Extended FitMetricsContract with 12 per-group sufficient-stat fields, introduced sibling EvaluateMetricsContract strict-contract for the evaluate wire payload (D-21 rejects free-form extras), and shipped BaselineFedAvg/BaselineFedProx strategy subclasses with server-side sum(hit_count)/sum(evaluated_users) aggregation (D-20, BSL-06). 24 GREEN tests across the three files. pyproject.toml untouched — Wave-1 write race avoided.**

## Performance

- **Duration:** ~6 min (352 seconds)
- **Started:** 2026-04-19T07:45:37Z
- **Completed:** 2026-04-19T07:51:29Z
- **Tasks:** 2 (both TDD autonomous; RED -> GREEN for every test)
- **Files created:** 5
- **Files modified:** 2
- **Tests added:** 12 (3 FitMetricsContract extension + 4 EvaluateMetricsContract + 5 BaselineFedAvg/FedProx strategy)
- **Foundation test suite:** 77 passed, 0 failed (was 70; +3 in test_weight_policy.py + 4 new test_evaluate_metrics.py)
- **federated-baseline-cf test suite:** 5 passed (new pytest tree — first tests under this module)

## Accomplishments

- **D-22 strict-contract landed for fit-side sufficient stats.** `FitMetricsContract` now carries 12 OPTIONAL per-group + overall sufficient-stat fields (`hit_count_{overall,sparse,medium,dense}_at10`, `ndcg_sum_{overall,sparse,medium,dense}_at10`, `evaluated_users{,_sparse,_medium,_dense}`). All default `None`; `to_dict()` drops them so Phase 1 backwards compat holds. `FIT_METRICS_REQUIRED_KEYS` unchanged — `validate_fit_metrics` still checks only the original three keys.
- **D-21 strict-contract landed for evaluate-wire payload.** New `EvaluateMetricsContract` dataclass exposes 3 required sufficient-stat keys (`hit_count_overall_at10`, `ndcg_sum_overall_at10`, `evaluated_users`), 3 optional diagnostic keys (`eval_loss`, `sampled_hr_at10`, `sampled_ndcg_at10` — cached client-side for logs, NOT consumed by aggregator), and 9 optional per-group keys mirroring `FitMetricsContract`. `validate_evaluate_metrics(payload)` enforces both required-key presence AND no-free-form-extras (D-21).
- **D-20 sufficient-stat aggregation shipped.** `BaselineFedAvg(FedAvg)` and `BaselineFedProx(FedProx)` in `federated_baseline_cf/strategy.py`. Both override `aggregate_evaluate` to emit server-side ratios computed from summed sufficient stats. Overall `sampled_hr@10` = Σhit_count / Σevaluated_users across all clients; same shape for the three per-group ratios (sparse/medium/dense). Zero-division safe (zero evaluated users → 0.0 for both HR and NDCG). `aggregate_fit` INHERITED UNCHANGED (D-23 preserved; baseline = all params global).
- **Unified naming convention (iteration 2 lock-in).** Every `_at10` suffix across EvaluateMetricsContract fields, FitMetricsContract fields, the strategy's `_sum_sufficient_stats` key tuple, and the client wire payload uses the same form (no underscore between 'at' and 10; no `@10` Python-identifier violation). Iteration 1 had `_at_10` vs `_at10` drift that would have silently zeroed all evaluate-side aggregation in production. Task 1 test `test_evaluate_metrics_per_group_fields` guards the alignment with `for group in ("sparse", "medium", "dense"): assert f"hit_count_{group}_at10" in d`.
- **12 new GREEN tests.** 3 extend `test_weight_policy.py` (15 total, was 12), 4 in new `test_evaluate_metrics.py`, 5 in new `federated-baseline-cf/tests/test_strategy.py`. All added via TDD — wrote tests first, confirmed RED (TypeError on new fields / ImportError on missing strategy module), then implemented to GREEN.
- **pyproject.toml UNTOUCHED by this plan.** Wave-1 write race avoided; Plan 02 Task 1 exclusively owns the `[project.optional-dependencies] dev = ["pytest>=7.0"]` declaration (iteration 1 BLOCKER 1 fix). Plan 02's own commit (`f784165`) landed in parallel between Plan 01 Task 1 and Task 2 commits — the two plans had zero file overlap.
- **D-18 surgical migration preserved.** Pre-existing uncommitted hunks in `federated-baseline-cf/federated_baseline_cf/{client_app,dataset,server_app,task}.py` were touched by me — `git diff --stat` on those files shows the same numbers as before my session (verified before Task 2 commit). Plan 03 Task 2 will consume those files during the actual client-side sufficient-stat population.

## Task Commits

Each task was committed atomically with `--no-verify` (Wave 1 parallel-executor safety; the orchestrator runs hooks once after all agents complete):

1. **Task 1: Extend FitMetricsContract + add EvaluateMetricsContract (D-21, D-22)** — `29c8d68` (feat)
2. **Task 2: Create BaselineFedAvg + BaselineFedProx subclasses (D-20, BSL-06)** — `001a3c2` (feat)

_Note: Plan metadata commit (this SUMMARY.md + STATE.md + ROADMAP.md updates) is appended separately at plan close._

## Files Created/Modified

### `scripts/foundation/fedrec_foundation/fit_metrics.py` (MODIFIED)

- FitMetricsContract dataclass extended with 12 `Optional[int|float] = None` fields: `hit_count_overall_at10`, `ndcg_sum_overall_at10`, `evaluated_users`, and the 3×3 per-group cartesian product (`hit_count_{sparse,medium,dense}_at10`, `ndcg_sum_{sparse,medium,dense}_at10`, `evaluated_users_{sparse,medium,dense}`). Defaults preserved; `to_dict()` drops `None` values so a Phase 1-shaped call (`train_loss`, `num_positives`, `num_training_examples` only) still emits 3 keys exactly.
- New `EVAL_METRICS_REQUIRED_KEYS: FrozenSet[str]` = `{"hit_count_overall_at10", "ndcg_sum_overall_at10", "evaluated_users"}`.
- New `EvaluateMetricsContract` dataclass with 3 required + 3 optional diagnostic + 9 optional per-group fields; `to_dict()` iterates `dataclasses.fields` and skips `None`; `from_dict()` filters unknown keys and wraps `TypeError` as `ValueError` (CR-4 pattern).
- New `validate_evaluate_metrics(payload)` function: (1) checks every key in `EVAL_METRICS_REQUIRED_KEYS` is present, (2) rejects any key not in `EvaluateMetricsContract`'s known fields (D-21 no free-form extras). Both checks raise `ValueError` with a self-documenting message (shows the missing keys / the free-form extras / the known contract fields).
- Module docstring updated to document the Phase 2 extension (12 new fit-side fields + the sibling evaluate-side contract pair).

### `scripts/foundation/tests/test_weight_policy.py` (MODIFIED, +3 tests)

Appended under a section banner `# Phase 2 Plan 01 (D-22) extension: per-group + overall sufficient stats.`:
- `test_fit_metrics_per_group_fields` — constructs a FitMetricsContract with all 12 new fields populated, asserts every field appears in `to_dict()`.
- `test_fit_metrics_per_group_optional` — constructs the minimal Phase 1-shape instance and asserts `to_dict()` drops all 12 per-group fields (backward compat).
- `test_fit_metrics_forward_compat_with_per_group_extension` — `from_dict({known + per-group + unknown "alpha"})` filters the unknown key and populates known per-group; absent per-group fields default `None`.

The 12 original tests are untouched — verified by `pytest tests/test_weight_policy.py -v` passing 15/15.

### `scripts/foundation/tests/test_evaluate_metrics.py` (CREATED, 4 tests)

- `test_evaluate_metrics_required_keys_enforced` — 2 positive paths (minimal required-only; required + diagnostics; no extras) + 2 negative paths (missing required → `ValueError` matching `"missing required keys"`; free-form extra → `ValueError` matching `"free-form extras"`).
- `test_evaluate_metrics_per_group_fields` — instantiates with all 15 fields populated; asserts all 3 required + all 9 per-group present in `to_dict()`.
- `test_evaluate_metrics_forward_compat` — `from_dict` filters unknown `"train_loss"` key; optional per-group default `None`.
- `test_evaluate_metrics_rejects_wrong_types` — missing-required-keys path raises `ValueError` (dataclass `TypeError` wrapped per CR-4).

### `federated-baseline-cf/federated_baseline_cf/strategy.py` (CREATED)

- `_sum_sufficient_stats(results)` — module-level helper that iterates `EvaluateRes.metrics or {}` for every `(ClientProxy, EvaluateRes)` tuple and sums each of the 12 sufficient-stat keys. Missing keys treated as `0` so a client that didn't emit a given group (e.g., no sparse users) contributes cleanly.
- `_sufficient_stats_to_thesis_metrics(totals)` — converts the summed dict to the 12-key thesis-table dict (`sampled_hr@10`, `sampled_ndcg@10`, `sampled_hr@10/{sparse,medium,dense}`, `sampled_ndcg@10/{sparse,medium,dense}`, and the 4 `evaluated_users*` diagnostics). Uses a nested `_safe_ratio` closure to divide-or-zero.
- `class BaselineFedAvg(BaseFedAvg)` + `class BaselineFedProx(BaseFedProx)` — each overrides `aggregate_evaluate(server_round, results, failures)`, returns `(loss, thesis_metrics)`. Loss is the num-examples-weighted mean of `eval_res.loss` (Flower convention). If `results` is empty, returns `(None, {})` so Flower's early-round edge cases still work.
- `__all__ = ["BaselineFedAvg", "BaselineFedProx"]`.

### `federated-baseline-cf/tests/__init__.py` (CREATED)

Empty marker file — makes `tests/` a Python package so `conftest.py` fixtures are discoverable.

### `federated-baseline-cf/tests/conftest.py` (CREATED)

- `_make_eval_res(num_examples, metrics)` — private factory that builds a Flower `EvaluateRes` (status = OK, loss = `metrics.get("eval_loss", 0.0)`, num_examples, metrics dict copy).
- `fake_evaluate_res` fixture — exposes `_make_eval_res` for strategy tests to build arbitrary responses.
- `fake_client_proxy` fixture — `MagicMock` with `cid = "test_client"` so the `(proxy, eval_res)` tuple shape matches what `aggregate_evaluate` receives.

### `federated-baseline-cf/tests/test_strategy.py` (CREATED, 5 tests)

- `test_aggregate_evaluate_sums_sufficient_stats` — 3-client arithmetic: overall hit_count summed 10+5+7=22, evaluated_users summed 20+15+25=60; asserts `sampled_hr@10 ≈ 22/60` and `evaluated_users == 60`.
- `test_aggregate_evaluate_per_group_ratios` — 2-client per-group arithmetic: sparse hit_count 3+1=4 / evaluated_users 10+5=15; medium 4+2=6 / 8+4=12; dense 5+0=5 / 5+0=5. Asserts each of the 3 per-group ratios + 3 per-group evaluated_users.
- `test_aggregate_evaluate_zero_division_safe` — 1 client with zero sparse/dense users; asserts `sampled_hr@10/sparse == 0.0` (no ZeroDivisionError).
- `test_baseline_fedprox_inherits_aggregate_evaluate` — instantiates BaselineFedProx(fraction_fit=0.1, proximal_mu=0.01) and verifies the sum-based ratio logic still works.
- `test_aggregate_fit_inherited_unchanged` — identity check: `BaselineFedAvg.aggregate_fit is FedAvg.aggregate_fit` (D-23 preserved).

## Decisions Made

- **Unified `_at10` suffix across all contract + strategy + wire surfaces** — iteration 2 fix. Iteration 1 had `_at_10` vs `_at10` drift that would have silently zeroed all evaluate-side aggregation because dict keys on the wire wouldn't match dict keys in the server reader (unit tests bypassed it by constructing strategy-shape dicts directly). Lock-in prevents regression.
- **`EvaluateMetricsContract` as a SIBLING dataclass, not a subclass of FitMetricsContract** — the evaluate wire payload carries diagnostic fields (`eval_loss` / `sampled_hr_at10` / `sampled_ndcg_at10`) that are NOT required on the fit side. Subclassing FitMetricsContract would either over-constrain evaluate or let free-form extras slip through `validate_fit_metrics`.
- **Diagnostic fields are OPTIONAL and NOT consumed by the server aggregator** — they're cached client-side for logs; the server re-computes the headline ratios from summed sufficient stats. Prevents a bad assumption where a client could emit a wrong `sampled_hr_at10` ratio and have the server trust it.
- **`aggregate_fit` INHERITED UNCHANGED** — D-23 preserved; baseline = all params global. Custom strategy exists only to fix evaluation aggregation. Enforced by `test_aggregate_fit_inherited_unchanged` identity check.
- **`BaselineFedProx.aggregate_evaluate` is an EXACT COPY of BaselineFedAvg's, not `super().aggregate_evaluate(...)`** — making BaselineFedProx extend BaselineFedAvg would create a diamond pattern (BaselineFedProx → BaselineFedAvg + BaseFedProx). The 4-line duplication is cheaper than explaining the MRO. Both use the same module-level helpers, so logic stays DRY.
- **`pyproject.toml` UNTOUCHED by this plan** — Plan 02 Task 1 exclusively owns the `[project.optional-dependencies] dev` section. Wave-1 write race avoided.
- **Pre-existing uncommitted hunks in `federated_baseline_cf/{client_app,dataset,server_app,task}.py` UNTOUCHED** — D-18 surgical migration. Plan 03 Task 2 will consume those files during the actual client-side sufficient-stat population.

## Deviations from Plan

**None — plan executed exactly as written.**

Both tasks followed their `<action>` blocks verbatim:
- Task 1: extended FitMetricsContract with the 12 fields specified, added EvaluateMetricsContract / EVAL_METRICS_REQUIRED_KEYS / validate_evaluate_metrics using the exact signatures in the plan's action block, appended the 3 FitMetricsContract tests and created the 4 EvaluateMetricsContract tests with the exact names specified in the plan's `<behavior>`.
- Task 2: created the three test fixtures (`__init__.py`, `conftest.py`, `test_strategy.py`) with the exact 5 test names, then created `strategy.py` with the 2 classes + 2 helpers exactly as specified.

No auto-fixes (Rules 1–3) applied. No architectural question (Rule 4) hit. No authentication gate. No test failures on first GREEN run.

## Authentication Gates

None — all work is local-filesystem + pytest. No external service touched.

## Issues Encountered

None. All automated verify commands passed on first run:

- RED step for Task 1 (new FitMetricsContract fields): 2 failed, 13 passed as expected (TypeError `got an unexpected keyword argument 'hit_count_overall_at10'`).
- GREEN step for Task 1: 15 passed, 0 failed after extending the dataclass.
- GREEN step for Task 1 test_evaluate_metrics.py: 4 passed, 0 failed on first run.
- RED step for Task 2: ImportError `No module named 'federated_baseline_cf.strategy'` (expected).
- GREEN step for Task 2: 5 passed, 0 failed after creating strategy.py.
- Full foundation regression: 77 passed, 0 failed, 0 skipped (was 70; +3 fit-metrics extension + 4 evaluate-metrics).
- Wave-1 pyproject.toml invariant: `git diff --name-only federated-baseline-cf/pyproject.toml` empty — no diff attributable to Plan 01.
- Pre-existing dirty files in `federated_baseline_cf/{client_app,dataset,server_app,task}.py` untouched — `git diff --stat` returns the same 4+7+35=46 lines as before my session (same prior-work hunks).

## Known Stubs

**None.** No placeholder values, no TODO markers, no `NotImplementedError`. All dataclass fields have real defaults (`None` for optionals, required for the 3 EvaluateMetricsContract must-haves). All tests have real assertions and real data arithmetic.

## Next Phase Readiness

**Ready for Plan 03 (client-side sufficient-stat population).** Plan 03 Task 2 will:
1. Read `FitMetricsContract` from `fedrec_foundation.fit_metrics` — 12 new per-group fields are ALREADY ready to accept populated values client-side.
2. Read `EvaluateMetricsContract` + `validate_evaluate_metrics` — client's `@app.evaluate()` handler builds its return payload via `EvaluateMetricsContract(...).to_dict()` and validates on the client side (defense-in-depth) before sending to the server.
3. Compute per-user `classify_user_group(n_interactions)` (from Phase 1's `fedrec_foundation.user_groups`) and route sufficient-stat contributions into the client's group bucket; other two groups receive zeros per the contract.

**Ready for Plan 04 (server_app.py wiring).** Plan 04 will:
1. Replace the direct `FedAvg(...)` / `FedProx(...)` instantiations in `server_app.py` with `BaselineFedAvg(...)` / `BaselineFedProx(...)`.
2. The `aggregate_evaluate` override will then fire on every evaluate round, emitting server-side ratios in the `(loss, metrics)` return tuple that Flower's `ServerApp` logs.
3. W&B integration in `server_app.py` can log both the server-side `sampled_hr@10` (from our aggregator) and the Flower-native per-round loss without conflict.

**Ready for Plans 3-5 (parallel module migrations).** The contract + strategy pattern established here (dataclass pair, sufficient-stat aggregation, module-specific strategy subclass) is the template Plans 3-5 will mirror in `federated-personalized-cf`, `federated-adaptive-personalized-cf`, and `federated-pfedrec` respectively.

**No blockers. No open questions. No architectural decisions deferred from this plan.**

## Self-Check: PASSED

- **Files created:**
  - FOUND: `scripts/foundation/tests/test_evaluate_metrics.py` — verified via `pytest scripts/foundation/tests/test_evaluate_metrics.py` collecting 4 tests.
  - FOUND: `federated-baseline-cf/federated_baseline_cf/strategy.py` — verified via `python -c "from federated_baseline_cf.strategy import BaselineFedAvg"` exit 0.
  - FOUND: `federated-baseline-cf/tests/__init__.py` — verified via `ls federated-baseline-cf/tests/`.
  - FOUND: `federated-baseline-cf/tests/conftest.py` — verified via pytest fixture discovery (`fake_evaluate_res` / `fake_client_proxy` resolve in `test_strategy.py` tests).
  - FOUND: `federated-baseline-cf/tests/test_strategy.py` — verified via `pytest federated-baseline-cf/tests/test_strategy.py` collecting 5 tests.
- **Files modified:**
  - FOUND: `scripts/foundation/fedrec_foundation/fit_metrics.py` — 14 `grep` hits for the per-group field names (`hit_count_overall_at10|evaluated_users_sparse|evaluated_users_dense`).
  - FOUND: `scripts/foundation/tests/test_weight_policy.py` — 15 tests collected (was 12 baseline); new test names `test_fit_metrics_per_group_fields`, `test_fit_metrics_per_group_optional`, `test_fit_metrics_forward_compat_with_per_group_extension` all present.
- **Commits:**
  - FOUND: `29c8d68` (Task 1 feat) — visible on `feat/try_to_run_the_baseline` via `git log --oneline -3`.
  - FOUND: `001a3c2` (Task 2 feat) — visible on `feat/try_to_run_the_baseline` via `git log --oneline -3`.
- **Automated verify:** PASSED.
  - `pytest scripts/foundation/tests/test_weight_policy.py -v` → 15 passed, 0 failed.
  - `pytest scripts/foundation/tests/test_evaluate_metrics.py -v` → 4 passed, 0 failed.
  - `pytest federated-baseline-cf/tests/test_strategy.py -v` → 5 passed, 0 failed.
  - Full foundation suite `pytest scripts/foundation/tests/` → 77 passed, 0 failed, 0 skipped (was 70; +3 fit-metrics extension + 4 new evaluate-metrics).
  - Backward-compat smoke: `python -c "from fedrec_foundation.fit_metrics import FitMetricsContract, validate_fit_metrics, FIT_METRICS_REQUIRED_KEYS; assert FIT_METRICS_REQUIRED_KEYS == ('train_loss','num_positives','num_training_examples'); c = FitMetricsContract(train_loss=0.1, num_positives=1, num_training_examples=5); assert c.hit_count_sparse_at10 is None; validate_fit_metrics(c.to_dict()); print('ok')"` → ok.
  - Evaluate-contract smoke: `python -c "from fedrec_foundation.fit_metrics import EvaluateMetricsContract, validate_evaluate_metrics; e = EvaluateMetricsContract(eval_loss=0.5, sampled_hr_at10=0.1, sampled_ndcg_at10=0.05, evaluated_users=1, hit_count_overall_at10=0, ndcg_sum_overall_at10=0.0); validate_evaluate_metrics(e.to_dict()); print('ok')"` → ok.
  - Strategy smoke: `python -c "from federated_baseline_cf.strategy import BaselineFedAvg, BaselineFedProx; assert BaselineFedAvg.aggregate_fit.__qualname__ == 'FedAvg.aggregate_fit'; print('ok')"` → ok.
- **Scope boundary:** PASSED. `git diff --name-only federated-baseline-cf/pyproject.toml` returns empty (no diff from Plan 01 commits). Pre-existing uncommitted hunks in `federated_baseline_cf/{client_app,dataset,server_app,task}.py` untouched — same `git diff --stat` numbers as before my session.

---

*Phase: 02-baseline-migration*
*Plan: 01 (Wave 1 — parallel with Plan 02; blocks Plans 03/04)*
*Completed: 2026-04-19*
*Unblocks: Plans 03 (client-side sufficient-stat population) + 04 (server_app.py strategy wiring).*
