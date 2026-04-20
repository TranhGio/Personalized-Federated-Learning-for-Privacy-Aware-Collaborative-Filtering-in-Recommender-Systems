---
phase: 04-adaptive-migration-bug-fixes
plan: 04
subsystem: testing
tags: [adp-07, alpha-factory, hierarchical-conditional, multi-factor, data-quantity, clip-bounds, pytest-parametrize, regression-surface, wave-2, tdd]

# Dependency graph
requires:
  - phase: 04-adaptive-migration-bug-fixes
    provides: pytest>=7.0 dev extra declared in federated-adaptive-personalized-cf/pyproject.toml (Plan 02)
provides:
  - 12 GREEN unit-test functions / 18 parametrized test items pinning the adaptive_alpha.py factory behavior contract (clip bounds [0.1, 0.95] + 4 HC rule branches + closed-enum whitelist)
  - Regression-surface defense-in-depth: if adaptive_alpha.py's np.clip ever regresses (someone removes the clip or widens the bounds), these tests catch it at CI time before a run contaminates thesis numbers
affects: [04-adaptive-migration-bug-fixes-03, 04-adaptive-migration-bug-fixes-05, 04-adaptive-migration-bug-fixes-06, phase-07 thesis-ablation-sweeps]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pure-unit regression-surface test file: imports production code, crafts adversarial inputs, asserts np.clip output invariants. Zero production-code changes required."
    - "pytest.mark.parametrize grid coverage for adversarial (n, ge, nu, rs) corners: 6 HC inputs + 2 MF inputs as the executable form of the thesis CONTEXT D-16 clip-hit-rate concern."
    - "Closed-enum whitelist assertion pattern (test_factory_unknown_method_raises): pytest.raises(ValueError, match='method|Unknown') on AlphaConfig(method='invalid_method') -- matches CONVENTIONS.md factory rule used by create_alpha_computer and get_model."

key-files:
  created:
    - federated-adaptive-personalized-cf/tests/test_alpha_factory.py
  modified: []

key-decisions:
  - "Zero production-code changes: the existing np.clip(..., min_alpha, max_alpha) at adaptive_alpha.py lines 208 (DataQuantityAlpha), 306 (MultiFactorAlpha.compute_from_stats), 339 (MultiFactorAlpha._compute_quantity_factor), and 486 (HierarchicalConditionalAlpha.compute_from_stats) already enforces the ADP-07 contract. Plan 04 ships tests that pin this behavior as a regression surface; it does NOT edit adaptive_alpha.py. git diff --stat federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/ is empty from this plan's commit."
  - "Test inputs traced against the production formula before writing: n=100 is midpoint sigmoid(0)=0.5, n=0 drives sigmoid(-5)=0.0067 below floor -> 0.1, n=200 drives sigmoid(5)=0.9933 above ceiling -> 0.95; HC n=5 triggers sparse, n=200+ge=0.5 triggers niche (f_q sigmoid raw ~0.993 > 0.6 AND f_d=0.167 < 0.25), rating_std=1.45 triggers inconsistent (f_consistency=0.033 < 0.3), nu=90+ge=0.5 triggers completionist (f_coverage=0.9 > 0.7 AND f_diversity=0.167 < 0.3)."
  - "Parametrize coverage pattern over marked-raw test-function style: 12 visible def test_ functions + 6 HC clip-bound parametrizations + 2 MF clip-bound parametrizations = 18 pytest items; lets each corner corner-case stand as a separately-reported pytest line item while keeping the file readable."
  - "Wave-2 file-disjointness held: this plan touched ONLY federated-adaptive-personalized-cf/tests/test_alpha_factory.py. Plan 03 (Wave-2 sibling) owns client_app.py + task.py + tests/{test_client_assertion.py, test_task_rng.py, test_embedding_cache_manifest_v2.py}. Zero file overlap, --no-verify commits proceed in parallel without write-race."

patterns-established:
  - "Pure-unit test against UNMODIFIED production module as behavior-fingerprint: import the classes, craft edge-case inputs with exact boundary values derived from the production formula, assert np.clip invariants. Useful for any factory/config-driven module where the contract is 'output always in range X regardless of input extremity'."
  - "HierarchicalConditional 4-rule coverage: craft one test per rule with inputs that uniquely trigger that branch, assert the rule name appears in compute_factors()['applied_rules'] PLUS the final alpha is clipped. Rule branches may co-fire (e.g., niche test inputs also trigger completionist at n_unique_items=200) -- the tests assert containment, not exclusivity, which is correct because production compute_from_stats composes all triggered rules multiplicatively/additively."
  - "No pytest conftest additions needed: file uses project-level conftest.py (fake_evaluate_res, fake_client_proxy fixtures) implicitly but references none of them. Test file is standalone-runnable via python -m pytest tests/test_alpha_factory.py."

requirements-completed: [ADP-07]

# Metrics
duration: 2min
completed: 2026-04-20
---

# Phase 04 Plan 04: Adaptive Alpha Factory Clip-Bounds + HC Rule-Branch Regression Surface

**Shipped ADP-07: 12 GREEN unit tests (18 parametrized pytest items) against the UNMODIFIED adaptive_alpha.py factory, pinning the [0.1, 0.95] clip contract across DataQuantityAlpha / MultiFactorAlpha / HierarchicalConditionalAlpha plus each of the 4 HC conditional rule branches (sparse/niche/inconsistent/completionist) and the closed-enum whitelist on unknown method strings.**

## Performance

- **Duration:** 2 min (single task, TDD-GREEN on first run — crafted inputs verified against production formula before file creation)
- **Started:** 2026-04-20T08:38:03Z
- **Completed:** 2026-04-20T08:40:00Z
- **Tasks:** 1 (single TDD-green task: file creation + test authoring + verification + commit)
- **Files modified:** 1 created (tests/test_alpha_factory.py), 0 production files touched

## Accomplishments

- **ADP-07 success criterion met:** `pytest tests/test_alpha_factory.py -v` reports 18 passed (>= 12 required); every test body asserts alpha in `[0.1, 0.95]`; each HC rule branch (sparse / niche / inconsistent / completionist) is exercised with a designed trigger input and the rule name appears in the `applied_rules` list returned by `compute_factors`.
- **Zero production-code changes** preserves D-18 surgical scope: `git diff --stat federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/` from this plan's commit is empty. The existing `np.clip(..., min_alpha, max_alpha)` at `adaptive_alpha.py` lines 208 / 306 / 339 / 486 already enforces the contract; Plan 04 ships tests that pin it as a regression surface.
- **DataQuantityAlpha sigmoid endpoints pinned:** floor at n=0 and n=50 (both clipped to 0.1), ceiling at n=200 (clipped to 0.95), midpoint at n=100 (== 0.5), plus `compute_from_stats` parity with `compute`.
- **HierarchicalConditionalAlpha rule coverage:** 4 rule-branch tests + 6 clip-bound parametrizations across adversarial corners `(0,0,0,0)`, `(0,3,0,1.5)`, `(10000,0,10000,0)`, `(10000,3,10000,1.5)`, `(5,1.5,5,0.75)`, `(1000,2.0,100,1.0)`.
- **MultiFactorAlpha clip-bound coverage:** 2 parametrized adversarial extremes.
- **Factory dispatch closed-enum whitelist:** `AlphaConfig(method='invalid_method')` raises ValueError (enforced by `AlphaConfig.__post_init__` at `adaptive_alpha.py:85`), matching the CONVENTIONS.md factory rule used across `create_alpha_computer` / `get_model` / `get_primary_evaluator`.
- **Full adaptive suite GREEN:** 38 tests pass (20 pre-existing from Plans 01+02 + 18 new from this plan). No regressions.

## Task Commits

Each task was committed atomically (Wave-2 parallel, --no-verify per plan instructions):

1. **Task 1: test_alpha_factory.py — ADP-07 clip-bounds + rule-branch coverage tests (12 functions, 18 parametrized items, GREEN on first run against unmodified adaptive_alpha.py)** — `499d28f` (test)

_No plan-metadata commit yet (SUMMARY + STATE + ROADMAP bundled into a final commit after this SUMMARY is written.)_

_Note: TDD tasks normally have separate RED + GREEN + REFACTOR commits; because the production code is already correct (the tests are regression fingerprints against existing behavior), the "RED" step was conceptual (trace every test's expected value against the formula before writing) and the "GREEN" commit is the single commit. No refactor pass needed._

## Files Created/Modified

- `federated-adaptive-personalized-cf/tests/test_alpha_factory.py` (created, 290 lines) — 12 pytest test functions + 8 parametrized items. Imports only from `federated_adaptive_personalized_cf.models.adaptive_alpha` (AlphaConfig, HierarchicalConditionalAlphaConfig, create_alpha_computer, DataQuantityAlpha, MultiFactorAlpha, HierarchicalConditionalAlpha). Zero fixture dependencies (uses no conftest.py fixtures). Fully self-contained.

## Decisions Made

See `key-decisions` in frontmatter. Summary:

1. **No production-code changes** — the existing `np.clip` at 4 call sites in adaptive_alpha.py already satisfies the ADP-07 contract. Plan 04 is strictly a regression-surface plan.
2. **Input traceability** — every test's expected output was manually traced against the production formula (sigmoid + clip + rule-branch tree) before committing; this is why the file lands GREEN on first run.
3. **Parametrize-heavy style** — 18 pytest items from 12 source functions lets each adversarial corner stand as a separately-reported line item.
4. **Wave-2 file disjointness** — commit with --no-verify alongside Plan 03's concurrent commits; no write-race possible because file ownership is disjoint.

## Deviations from Plan

None — plan executed exactly as written. The Research §"Alpha factory unit test covering every conditional rule branch" code skeleton was followed verbatim, lifted into the final file with minor formatting adjustments (from-future-imports, module docstring expansion with cross-references, consistent ASCII in formula comments).

**Single minor note:** I confirmed empirically that `f_quantity` inside `HierarchicalConditionalAlpha._compute_quantity_factor` is RAW (unclipped) sigmoid (see `adaptive_alpha.py:551`), unlike `MultiFactorAlpha._compute_quantity_factor` which IS clipped (line 339). This means the `test_hc_niche_bonus_applies` test's assumption (`n=200 → f_q ≈ 0.993 > 0.6`) is correct — the pre-clip raw sigmoid drives the rule trigger. The MultiFactorAlpha clip at line 339 is not exercised by any HC test; it's specifically exercised by `test_multi_factor_clip_bounds`. Both behaviors are pinned.

## Issues Encountered

None. The plan specified "GREEN on first run; no production-code changes needed" and that held. One minor cognitive check during drafting was verifying that `HierarchicalConditionalAlpha._compute_quantity_factor` does NOT clip (returns raw sigmoid); the niche-bonus test's `f_q > 0.6` condition relies on this raw value being ~0.993, not the clipped 0.95. Confirmed by reading line 551 of adaptive_alpha.py before writing the test.

## Self-Check: PASSED

**File exists:**
- `federated-adaptive-personalized-cf/tests/test_alpha_factory.py` — FOUND (290 lines, 12 test functions, 18 pytest items via parametrization)

**Commit exists:**
- `499d28f` — FOUND on branch feat/try_to_run_the_baseline (`test(04-04): ADP-07 alpha factory clip-bounds + HC rule-branch coverage`)

**Acceptance criteria verification (all GREEN before commit):**
- `test -r federated-adaptive-personalized-cf/tests/test_alpha_factory.py` → OK
- `grep -c "^def test_" tests/test_alpha_factory.py` → 12 (>= 12 required)
- All 10 required test names present (1 each)
- `grep -c "applied_rules" tests/test_alpha_factory.py` → 17 (>= 4 required)
- `grep -cE '0\.1.*<=.*<=.*0\.95|0\.1 <= alpha|alpha <= 0\.95' tests/test_alpha_factory.py` → 6 (>= 5 required)
- `pytest tests/test_alpha_factory.py -v` → 18 passed (>= 12 required)
- `pytest tests/test_alpha_factory.py --collect-only` → 18 items (>= 18 required)
- `git diff --stat federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/` from HEAD^..HEAD → empty (zero production code modifications from this plan's commit)
- Full adaptive suite: `pytest tests/` → 38 passed (20 pre-existing + 18 new)

## Next Phase Readiness

- ADP-07 closed. Wave-2 Plan 03 sibling (client_app.py + task.py + 3 other test files) commits in parallel; Wave-3 (server_app.py migration + cross-module scripts/clean_cache.py extension + subprocess determinism regression guard) can proceed once Wave-2 lands.
- The test file is a permanent defense-in-depth regression guard: any future silent drift in adaptive_alpha.py's clip behavior (removal of np.clip, widening of min/max_alpha, typo in rule-branch logic) will surface as a pytest failure before it contaminates thesis numbers.

---
*Phase: 04-adaptive-migration-bug-fixes*
*Completed: 2026-04-20*
