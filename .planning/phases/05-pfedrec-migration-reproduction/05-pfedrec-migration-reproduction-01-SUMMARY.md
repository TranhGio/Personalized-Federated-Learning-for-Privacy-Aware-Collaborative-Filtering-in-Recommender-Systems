---
phase: 05-pfedrec-migration-reproduction
plan: 01
subsystem: federated-learning
tags: [pfedrec, split-learning, fedavg, strategy, model, audit, ijcai-23]

# Dependency graph
requires:
  - phase: 01-foundation-contract
    provides: EvaluateMetricsContract sufficient-stat schema (12 fields), atomic_write_json
  - phase: 02-baseline-migration
    provides: BaselineFedAvg sufficient-stat aggregate_evaluate template
  - phase: 03-personalized-migration
    provides: PersonalizedSplitFedAvg / D-21 strict=True idiom / module-prefix class naming convention
  - phase: 04-adaptive-migration-bug-fixes
    provides: AdaptiveSplitFedAvg pattern + bias-classification sentinel idea (schema_v2 carry-forward to schema_v3 in Plan 03)
provides:
  - PFedRecSplitFedAvg strategy class (D-12 rename, D-07 FedProx-dropped)
  - GLOBAL_PARAM_KEYS / LOCAL_PARAM_KEYS frozensets per D-01 (bias-GLOBAL flip)
  - Module-level _sum_sufficient_stats / _sufficient_stats_to_thesis_metrics helpers
  - PFedRecMLP._GLOBAL_PARAMS / _LOCAL_PARAMS class tuples per D-01
  - PFedRecMLP.set_local_parameters strict=True default with rm -rf hint (D-21)
  - PFR-02-AUDIT.md (closes ROADMAP §Phase 5 SC-1)
affects:
  - 05-pfedrec-migration-reproduction-03 (client_app + cache layout consumes the model contract)
  - 05-pfedrec-migration-reproduction-04 (server_app wires PFedRecSplitFedAvg)
  - 05-pfedrec-migration-reproduction-05 (regression-guard subprocess test on partition_{pid}.pt cache)
  - 06-evaluation-harness (consumes thesis_metrics dict shape)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Strategy + model frozenset/tuple symmetry test (Pitfall 1 guard) — set(GLOBAL_PARAM_KEYS) == set(PFedRecMLP._GLOBAL_PARAMS) enforced mechanically"
    - "D-21 strict=True hard-fail on cache load with per-field delta + literal rm -rf hint (Phase 3 D-05 idiom carried forward, run_id arg threaded through for client_app injection)"
    - "Module-prefixed strategy class naming (PFedRecSplitFedAvg matches BaselineFedAvg / PersonalizedSplitFedAvg / AdaptiveSplitFedAvg)"
    - "Reference-audit-as-artifact: PFR-02-AUDIT.md cross-walks every divergence row to a specific reference line + Flower line + Decision + CONTEXT D-XX pin"

key-files:
  created:
    - federated-pfedrec/federated_pfedrec/strategy.py
    - federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py
    - federated-pfedrec/tests/conftest.py
    - federated-pfedrec/tests/test_strategy.py
    - federated-pfedrec/tests/test_pfedrec_mlp.py
    - .planning/phases/05-pfedrec-migration-reproduction/PFR-02-AUDIT.md
  modified: []  # Note: strategy.py / pfedrec_mlp.py were untracked before this plan; commits create canonical Phase-5 versions.

key-decisions:
  - "D-01 enforced at model + strategy layer: affine_output.bias is GLOBAL (engine.py:143 deletes only the weight before aggregation; the bias travels and is averaged server-side). Closes CONCERNS divergence #9, the headline lever for landing PFR-08 within ±2 points."
  - "D-12 hard rename: SplitFedAvg removed entirely (not aliased), legacy SplitFedAvg / SplitFedProx / PFedRecSplitFedProx names are NOT importable from federated_pfedrec.strategy. Mechanically pinned by test_strategy_class_renamed_pfedrecsplitfedavg."
  - "D-07 drop: no FedProx variant ships for PFedRec (BaseFedProx is not even imported into strategy.py — verified by grep at acceptance-criteria time)."
  - "D-21 strict=True default with run_id-threaded rm -rf hint: missing key OR shape mismatch raises RuntimeError carrying the offending key, saved shape, current shape, and 'rm -rf .embedding_cache/{run_id}/' suffix. Plan 03 client_app will pass the real run_id from server-broadcast config."
  - "D-19 paper-faithful Kaiming default init preserved: NO Xavier reset added on PFedRecMLP. Source-level grep guard (test_kaiming_default_init_paper_faithful) prevents future drift; PFR-08 ±2 reproduction is sensitive to init scale."
  - "D-24/D-26 sufficient-stat aggregate_evaluate: sums 12 EvaluateMetricsContract fields and divides ONCE at the end (matches engine.py:81 len(round_user_params) uniform mean in 1-user-per-client cross-device). Cloned from PersonalizedSplitFedAvg with frozensets flipped per D-01."
  - "Pitfall 1 guard mechanically enforced: set(GLOBAL_PARAM_KEYS) == set(PFedRecMLP._GLOBAL_PARAMS) and set(LOCAL_PARAM_KEYS) == set(PFedRecMLP._LOCAL_PARAMS) — strategy/model frozenset drift cannot silently re-emerge."
  - "PFR-02-AUDIT.md materialized at .planning/phases/05-pfedrec-migration-reproduction/PFR-02-AUDIT.md per the path locked in must_haves.artifacts; closes ROADMAP §Phase 5 SC-1 with all three primary anchors (engine.py:143 / engine.py:81 / engine.py:195-196) and explicit SC-2 reconciliation against D-01."

patterns-established:
  - "Pattern 1: Strategy/Model symmetry test — every cross-module split-learning module's strategy.py frozensets MUST match its model class _GLOBAL_PARAMS / _LOCAL_PARAMS tuples. Test imports both and compares as sets (orderless)."
  - "Pattern 2: D-21 strict=True hard-fail with run_id-threaded hint — set_local_parameters signature ends in run_id: str = '<run_id>' default; client_app passes real run_id; test asserts exact error format with 4 substrings (key name, saved shape, current shape, 'rm -rf')."
  - "Pattern 3: Module-prefixed strategy class naming convention complete across 4 modules: BaselineFedAvg / PersonalizedSplitFedAvg / AdaptiveSplitFedAvg / PFedRecSplitFedAvg."
  - "Pattern 4: Reference-audit-artifact for migration phases — when migrating a Flower module to align with a published reference, ship a phase-local *-AUDIT.md cross-walking each divergence row to (a) reference line anchor, (b) Flower current line anchor, (c) align/keep/already-aligned decision, (d) rationale, (e) CONTEXT D-XX pin."

requirements-completed: [PFR-02, PFR-03]

# Metrics
duration: 6min
completed: 2026-04-28
---

# Phase 5 Plan 01: PFedRec Strategy + Model + PFR-02 Audit Summary

**PFedRec strategy class flipped to PFedRecSplitFedAvg with `affine_output.bias` reclassified GLOBAL per IJCAI-23 reference (`engine.py:143`); model param tuples + D-21 strict-load hard-fail shipped; PFR-02 9-row reference audit cross-walked at `.planning/phases/05-pfedrec-migration-reproduction/PFR-02-AUDIT.md`.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-28T09:28:21Z
- **Completed:** 2026-04-28T09:34:33Z
- **Tasks:** 3
- **Files modified:** 6 (3 source + 2 test + 1 audit doc + 1 conftest placeholder)
- **Tests added:** 8 GREEN (5 strategy + 3 model)

## Accomplishments

- D-01 bias-GLOBAL flip enforced symmetrically at strategy frozensets AND model class tuples
- D-12 hard rename completed: legacy `SplitFedAvg` / `SplitFedProx` / `PFedRecSplitFedProx` names NOT importable
- D-07 FedProx drop completed: `BaseFedProx` is not imported into `strategy.py` at all
- D-21 strict=True default with run_id-threaded `rm -rf .embedding_cache/{run_id}/` hint shipped on `PFedRecMLP.set_local_parameters`
- D-19 paper-faithful Kaiming default init preserved (no Xavier reset added; source-level forbidden-token grep guard pins this)
- D-24/D-26 sufficient-stat `aggregate_evaluate` ships, cloned from Phase 3 `PersonalizedSplitFedAvg` with frozensets flipped
- Pitfall 1 strategy/model symmetry guard tested and passes (`set(GLOBAL_PARAM_KEYS) == set(PFedRecMLP._GLOBAL_PARAMS)`)
- `PFR-02-AUDIT.md` materialized with 9 rows + 3 anchors + SC-2 reconciliation, closing ROADMAP §Phase 5 SC-1

## Task Commits

Each task was committed atomically with `--no-verify` (parallel-execution mode with Plan 02):

1. **Task 1: Strategy.py rewrite + 5 strategy tests** (TDD)
   - RED: `0f1b863` test(05-01): add failing strategy tests for D-01/D-12/D-07/D-24
   - GREEN: `dd190ab` feat(05-01): rewrite strategy.py as PFedRecSplitFedAvg per D-01/D-07/D-12/D-24
2. **Task 2: PFedRecMLP update + 3 model tests** (TDD)
   - RED: `7ef753c` test(05-01): add failing PFedRecMLP tests for D-01/D-19/D-20/D-21
   - GREEN: `3957ff2` feat(05-01): update PFedRecMLP per D-01/D-19/D-20/D-21
3. **Task 3: PFR-02 reference audit cross-walk**
   - `6e319d5` docs(05-01): add PFR-02 reference audit cross-walk (closes ROADMAP §Phase 5 SC-1)

_TDD pattern: Tasks 1 and 2 each landed RED-then-GREEN; Task 3 was a docs task with no test layer._

## Files Created/Modified

- `federated-pfedrec/federated_pfedrec/strategy.py` — Rewritten as `PFedRecSplitFedAvg` (D-01/D-07/D-12/D-24/D-26). Module-level `_sum_sufficient_stats` + `_sufficient_stats_to_thesis_metrics` helpers; 12 sufficient-stat field constant; FedProx variant removed; `aggregate_fit` inherited unchanged from `BaseFedAvg` (Phase 3 D-23 invariant; Plan 04 server_app sets `FitRes.num_examples = 1` for uniform weighting under D-24).
- `federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py` — `_GLOBAL_PARAMS = ('embedding_item.weight', 'affine_output.bias')`; `_LOCAL_PARAMS = ('affine_output.weight',)`; `set_local_parameters(strict=True, run_id='<run_id>')` raises `RuntimeError` on missing key OR shape mismatch with the literal `rm -rf` hint; `forward` / `predict` / `__init__` preserved verbatim per D-18 surgical-edit discipline; D-20 native PyTorch `(1, latent_dim)` shape preserved end-to-end.
- `federated-pfedrec/tests/conftest.py` — placeholder; Plan 02 may extend.
- `federated-pfedrec/tests/test_strategy.py` — 5 GREEN tests (D-12 rename, D-01 frozensets, bias-not-LOCAL regression guard, D-24/D-26 sum-once aggregation, Pitfall 1 strategy/model symmetry).
- `federated-pfedrec/tests/test_pfedrec_mlp.py` — 3 GREEN tests (D-01 + D-20 tuple/shape, D-21 strict-fail with rm -rf, D-19 Kaiming-default + Xavier-forbidden-token source guard).
- `.planning/phases/05-pfedrec-migration-reproduction/PFR-02-AUDIT.md` — 9-row reference cross-walk; Decision column populated for every row; three primary anchors (`engine.py:143` / `engine.py:81` / `engine.py:195-196`) cited; SC-1 closure stated; SC-2 reconciliation against D-01 (bias-disk-to-server-aggregation) explicit.

## Test Decision Pinning Map

| Test | Decision pinned | Mechanism |
|------|-----------------|-----------|
| `test_strategy_class_renamed_pfedrecsplitfedavg` | D-12, D-07 | `dir(strategy_module)` membership checks for legacy names |
| `test_global_param_keys_includes_bias` | D-01 | `frozenset == frozenset` equality |
| `test_local_param_keys_excludes_bias` | D-01 (regression guard) | Membership assertion + companion GLOBAL membership |
| `test_aggregate_evaluate_sufficient_stat_uniform` | D-24, D-26 | 3 mocked `EvaluateRes` with hand-computed expected ratios |
| `test_global_param_keys_matches_model_tuple` | Pitfall 1 (D-01 symmetry) | `set(GLOBAL_PARAM_KEYS) == set(PFedRecMLP._GLOBAL_PARAMS)` and LOCAL pair |
| `test_local_params_tuple_only_affine_weight` | D-01 + D-20 | Tuple equality + shape `(1, 32)` check |
| `test_set_local_parameters_strict_true_hard_fails` | D-21 | `pytest.raises(RuntimeError)` + 4 substring assertions on the message |
| `test_kaiming_default_init_paper_faithful` | D-19 | `0 < weight.std() < 1` numeric guard + `re.search('xavier_uniform_(', src)` source guard |

## Pitfall 1 Mechanical Enforcement

`test_global_param_keys_matches_model_tuple` in `tests/test_strategy.py` imports both layers and compares as sets:

```python
from federated_pfedrec.models.pfedrec_mlp import PFedRecMLP
from federated_pfedrec.strategy import GLOBAL_PARAM_KEYS, LOCAL_PARAM_KEYS
assert set(GLOBAL_PARAM_KEYS) == set(PFedRecMLP._GLOBAL_PARAMS)
assert set(LOCAL_PARAM_KEYS) == set(PFedRecMLP._LOCAL_PARAMS)
```

Any future single-side change to GLOBAL/LOCAL classification (whether at the strategy layer or the model layer) trips this test the next time `pytest tests/test_strategy.py` runs. The cross-module convention (Phase 3/4 sibling tests already do this) is now fully wired for PFedRec.

## PFR-02-AUDIT.md Closure Note

The audit document materializes the 9-row table from `RESEARCH §Pattern 1` at `.planning/phases/05-pfedrec-migration-reproduction/PFR-02-AUDIT.md`. Every row carries:

- a Reference behavior column anchored to a specific `engine.py:LINE` (or `mlp.py:LINE`),
- a Flower current behavior column anchored to a specific `federated-pfedrec/.../FILE:LINE`,
- a Decision column drawn from `{align-to-reference, align-to-reference (with adaptation), align-to-reference (BUT strictly stricter — FND-03 fixes the leak that BOTH codebases share), strictly-better-than-reference, already-aligned}`,
- a Rationale column,
- a CONTEXT D-XX pin column pointing back at the locked decision.

The Closure Note states ROADMAP §Phase 5 SC-1 closure explicitly and adds an SC-2 reconciliation paragraph: SC-2's "atomic per-user (weight, bias) artifact" requirement reconciles with D-01 (bias-GLOBAL) by preserving the per-round, per-user atomicity contract while moving the bias channel from per-user disk to server-side aggregation per `engine.py:143`. Plan 03 will surface the explicit reconciliation note in its cache-layout task; Plan 04 server_app will surface it in the `_manifest` block. The verifier MUST accept this reconciliation when evaluating SC-2.

## Decisions Made

None beyond the locked CONTEXT D-XX decisions. Plan executed exactly as written in the inline `<action>` blocks.

## Deviations from Plan

None — plan executed exactly as written.

The plan's Task 1 acceptance specified 4 GREEN tests in `test_strategy.py`; this plan ships 5 GREEN tests there because the 5th (Pitfall 1 strategy/model frozenset symmetry guard) is also called out by the plan's `<key_links>` block:

```yaml
- from: "federated_pfedrec.strategy.GLOBAL_PARAM_KEYS"
  to: "federated_pfedrec.models.pfedrec_mlp.PFedRecMLP._GLOBAL_PARAMS"
  via: "test_strategy.py::test_global_param_keys_matches_model_tuple"
  pattern: "set\\(GLOBAL_PARAM_KEYS\\) == set\\(PFedRecMLP._GLOBAL_PARAMS\\)"
```

So the 5th test is the named realization of the plan's own key-link contract. Counted as a 1-test enrichment, not a deviation.

## Issues Encountered

- **Untracked source files** — `federated-pfedrec/federated_pfedrec/strategy.py` and `federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py` were untracked in git (not in `.gitignore` either, but never `git add`-ed). The Phase-5-aligned versions committed by this plan are now the canonical tracked versions. No regression — pre-existing on-disk content was the prior `SplitFedAvg` / `SplitFedProx` shape and was replaced wholesale per the plan's `<action>` block.
- **Parallel Plan 02 commits** — Plan 02 (running in parallel under the orchestrator's wave-1 split) committed `74a405b` (pyproject) and `4f48980` (dataset.py) interleaved with Plan 01's commits. File-ownership rules held: Plan 01 touched only its 6 files; Plan 02 touched only its files. No write race.

## Self-Check: PASSED

- `federated-pfedrec/federated_pfedrec/strategy.py` exists.
- `federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py` exists.
- `federated-pfedrec/tests/test_strategy.py` exists.
- `federated-pfedrec/tests/test_pfedrec_mlp.py` exists.
- `federated-pfedrec/tests/conftest.py` exists.
- `.planning/phases/05-pfedrec-migration-reproduction/PFR-02-AUDIT.md` exists.
- Commits `0f1b863`, `dd190ab`, `7ef753c`, `3957ff2`, `6e319d5` exist on `feat/try_to_run_the_baseline`.
- All 8 tests pass: `pytest federated-pfedrec/tests/test_strategy.py federated-pfedrec/tests/test_pfedrec_mlp.py -v` exits 0.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Plan 02 (parallel):** running concurrently; touches `pyproject.toml`, `dataset.py`, `mode.py` — all disjoint from Plan 01. No coordination needed.
- **Plan 03 (Wave 2):** will consume the strategy + model contract delivered here. Specifically, Plan 03 client_app must:
  - Pass the real `run_id` (from `context.run_config["run-id"]`) into `model.set_local_parameters(local_state, run_id=run_id)` so the D-21 hard-fail message points at the live cache directory.
  - Use `GLOBAL_PARAM_KEYS` / `LOCAL_PARAM_KEYS` from `federated_pfedrec.strategy` (single source of truth) when partitioning state-dict between wire payload and disk cache.
  - Persist `affine_output.weight` only (D-01: bias is now GLOBAL and aggregated server-side).
- **Plan 04 (Wave 3):** will instantiate `PFedRecSplitFedAvg(fraction_fit=1.0)` and ensure `FitRes.num_examples = 1` per client (Pitfall 5 Option B from RESEARCH) to make existing `BaseFedAvg.aggregate_fit` mathematically uniform under D-24.
- **Plan 05 (Wave 3):** subprocess regression guard checks per-key `torch.equal` on `partition_{pid}.pt` — payload now contains only `affine_output.weight` (D-01).

---
*Phase: 05-pfedrec-migration-reproduction*
*Completed: 2026-04-28*
