---
phase: 03-personalized-migration
plan: 01
subsystem: infra
tags: [strategy, personalized-split-fedavg, personalized-split-fedprox, single-row-model, bpr-mf, basic-mf, sufficient-stats, d-01, d-03, d-20, d-23, psn-06, tdd, wave-1]

# Dependency graph
requires:
  - phase: 01-foundation-contract
    provides: EvaluateMetricsContract 12 sufficient-stat keys (hit_count_*/ndcg_sum_*/evaluated_users_*) that the aggregate_evaluate override reads
provides:
  - PersonalizedSplitFedAvg + PersonalizedSplitFedProx with sufficient-stat aggregate_evaluate override (aggregate_fit inherited unchanged per D-23)
  - Module-level _GLOBAL_PARAM_KEYS = {item_embeddings.weight, item_bias.weight, global_bias} + _LOCAL_PARAM_KEYS = {local_user_row, local_user_bias} frozensets (D-03 flipped split)
  - BPRMF + BasicMF refactored to single-row contract — local_user_row nn.Parameter(shape=(d,)) + local_user_bias nn.Parameter(shape=(1,)) replace the num_users×d ghost table (D-01)
  - forward() drops user_ids — the client IS one user (D-02)
  - get/set_local_parameters contract: 2-key OrderedDict({'local_user_row', 'local_user_bias'}) — disk payload per client drops from ~3 MB to ~516 B
  - federated-personalized-cf/tests/ pytest package (conftest fixtures + test_strategy.py + test_single_row_model.py)
affects: [03-plan-03-client-app, 03-plan-04-server-app, 04-adaptive-migration, 05-pfedrec-migration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "sufficient-stat server-side aggregation (sum(hit)/sum(evaluated) over per-group totals) — cloned from Phase 2 baseline with flipped GLOBAL/LOCAL split"
    - "D-23 split-learning invariant: aggregate_fit inherited UNCHANGED from parent FedAvg/FedProx; only aggregate_evaluate is overridden"
    - "D-01 single-row model contract: per-client nn.Parameter(d,) instead of nn.Embedding(num_users, d) ghost table"

key-files:
  created:
    - federated-personalized-cf/tests/__init__.py
    - federated-personalized-cf/tests/conftest.py
    - federated-personalized-cf/tests/test_strategy.py
    - federated-personalized-cf/tests/test_single_row_model.py
  modified:
    - federated-personalized-cf/federated_personalized_cf/strategy.py
    - federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py
    - federated-personalized-cf/federated_personalized_cf/models/basic_mf.py

key-decisions:
  - "D-18 rip authorized for strategy.py: the pre-existing SplitFedAvg/SplitFedProx/extract_global_params/extract_local_params helpers in strategy.py ARE the aggregator under refactor, so rip-and-replace is in scope. Unrelated WIP in models/ (local_user_row init, etc.) was preserved verbatim."
  - "D-23 invariant preserved by identity check (PersonalizedSplitFedAvg.aggregate_fit is BaseFedAvg.aggregate_fit) — the parent's weighted average of GLOBAL params is correct for split learning because the client only sends GLOBAL params."
  - "PersonalizedSplitFedProx.aggregate_evaluate is an EXACT COPY (not super() call) of PersonalizedSplitFedAvg.aggregate_evaluate — mirrors Phase 2 baseline's diamond-inheritance avoidance; helpers are module-level so duplication is 4 lines."
  - "local_user_bias uses nn.Parameter when use_bias=True; nn.register_buffer with persistent=False when use_bias=False (BPRMF only; BasicMF always has bias). Keeps get_local_parameters key set consistent across branches while preventing buffer leakage to state_dict serialization."
  - "num_users constructor arg retained for API compat but NOT stored as self.num_users — the client is one user; a model-side user table would contradict D-01."

patterns-established:
  - "Test parity with Phase 2 baseline: conftest.py fixtures (fake_evaluate_res + fake_client_proxy) and test_strategy.py test names copied verbatim; only the strategy class names and frozenset contents differ. Future modules (adaptive, pfedrec) follow this same shape."
  - "Single-row roundtrip test: construct model → save local params → mutate local_user_row in the dict → set_local_parameters → assert restore. This is the minimum-viable cache-correctness guard before Plan 03 wires disk persistence."
  - "No-ghost-table structural test: open models/bpr_mf.py (and basic_mf.py) as text and assert 'nn.Embedding(num_users' not in src AND 'self.user_embeddings' not in src. Cheap regression guard against accidental reintroduction of the ghost table."

requirements-completed: [PSN-06]

# Metrics
duration: 4min
completed: 2026-04-20
---

# Phase 03 Plan 01: Personalized Split Strategies + Single-Row Model Refactor Summary

**PersonalizedSplitFedAvg/FedProx with sufficient-stat aggregate_evaluate + BPRMF/BasicMF collapsed to single nn.Parameter per client (no more num_users×d ghost table).**

## Performance

- **Duration:** ~4 min (between two task commits; does not include test iteration)
- **Started:** 2026-04-19T23:11:50+07:00 (Task 1 commit timestamp)
- **Completed:** 2026-04-19T23:15:49+07:00 (Task 2 commit timestamp)
- **Tasks:** 2 (both TDD)
- **Files modified:** 7 (3 source + 4 test)

## Accomplishments

- `PersonalizedSplitFedAvg` + `PersonalizedSplitFedProx` shipped in `federated-personalized-cf/federated_personalized_cf/strategy.py` with sufficient-stat `aggregate_evaluate` override (sum(hit_count)/sum(evaluated_users) over overall + sparse/medium/dense groups) — mirrors Phase 2 baseline with the GLOBAL/LOCAL frozensets flipped (item_* GLOBAL, local_user_* LOCAL).
- `aggregate_fit` inherited unchanged from parent FedAvg/FedProx (D-23 split-learning invariant) — verified by identity check `PersonalizedSplitFedAvg.aggregate_fit is BaseFedAvg.aggregate_fit`.
- `BPRMF` and `BasicMF` refactored: `nn.Embedding(num_users, d)` ghost table → `nn.Parameter(shape=(embedding_dim,))` single row + `nn.Parameter(shape=(1,))` scalar bias. `forward()`/`forward_item_only()`/`_compute_score()` dropped the `user_ids` argument — the client is one user.
- `get_local_parameters()` / `set_local_parameters()` now operate on a 2-key `OrderedDict({'local_user_row', 'local_user_bias'})`. Per-client on-disk payload size drops from ~3 MB (6040×128×4B) to ~516 B (128+1 floats) — the primary PSN-06 win.
- New pytest package at `federated-personalized-cf/tests/` with 12 GREEN tests (5 strategy + 7 single-row model) guarding the new contract.

## Task Commits

Each task was committed atomically:

1. **Task 1: PersonalizedSplitFedAvg + PersonalizedSplitFedProx strategy subclasses** — `858915d` (feat)
2. **Task 2: BPRMF + BasicMF single-row refactor** — `fabc7eb` (feat)

## Files Created/Modified

- `federated-personalized-cf/federated_personalized_cf/strategy.py` — rip-and-replace: sufficient-stat aggregator + flipped GLOBAL/LOCAL frozensets
- `federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py` — D-01 single-row refactor; `_LOCAL_PARAMS_WITH_BIAS = ('local_user_row', 'local_user_bias')`
- `federated-personalized-cf/federated_personalized_cf/models/basic_mf.py` — D-01 single-row refactor (no `use_bias` toggle; always has bias)
- `federated-personalized-cf/tests/__init__.py` — package marker
- `federated-personalized-cf/tests/conftest.py` — `fake_evaluate_res` + `fake_client_proxy` fixtures (copied verbatim from baseline)
- `federated-personalized-cf/tests/test_strategy.py` — 5 GREEN tests (sufficient-stat sums, per-group ratios, zero-division safety, FedProx inheritance, `aggregate_fit` identity check)
- `federated-personalized-cf/tests/test_single_row_model.py` — 7 GREEN tests (BPRMF/BasicMF shape, local-params contract, global-params contract, no-ghost-table structural check, roundtrip)

## Decisions Made

See the `key-decisions` block in frontmatter. Highlights:

- **D-18 rip authorized on strategy.py only.** The pre-existing `SplitFedAvg/SplitFedProx/extract_global_params/extract_local_params` in `strategy.py` were the aggregator under refactor, so rip-and-replace is in scope. Unrelated uncommitted WIP in `models/bpr_mf.py` and `basic_mf.py` (init paths, local_user_row scaffolding) was preserved and extended, not overwritten.
- **D-23 preserved via identity check.** `PersonalizedSplitFedAvg.aggregate_fit is BaseFedAvg.aggregate_fit` returns True; the parent's weighted-average-of-GLOBAL-params is correct because the client only sends GLOBAL params. Same for FedProx.
- **FedProx aggregate_evaluate duplicates FedAvg's body** (4 lines; helpers are module-level) instead of using `super()` — avoids diamond inheritance with BaseFedProx and mirrors the Phase 2 baseline pattern exactly.
- **`local_user_bias` buffer branch (BPRMF use_bias=False)** uses `persistent=False` so it never leaks into `state_dict()` serialization while still exposing a stable `self.local_user_bias` attribute to keep the 2-key contract uniform.
- **`num_users` retained as constructor arg, dropped as attribute.** API-compatibility one-liner for any remaining caller; `self.num_users` is gone because a per-client model has no user table.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Minor Spec Drift] BasicMF has no `use_bias` toggle — always has bias**

- **Found during:** Task 2 (single-row refactor)
- **Issue:** The plan's `test_basic_mf_single_row_shape` test assumed a `BasicMF(..., use_bias=True)` constructor mirroring BPRMF. `BasicMF.__init__` in this codebase only takes `(num_users, num_items, embedding_dim, dropout)` — bias is always present.
- **Fix:** Dropped the `use_bias` keyword from BasicMF tests. The single-row contract is still enforced: `local_user_row` is always a `nn.Parameter(shape=(embedding_dim,))` and `local_user_bias` is always `nn.Parameter(shape=(1,))`. No BasicMF `use_bias=False` branch exists, so no asymmetric test is needed.
- **Files modified:** `federated-personalized-cf/tests/test_single_row_model.py` (adapted BasicMF test to the actual constructor signature)
- **Verification:** `pytest tests/test_single_row_model.py::test_basic_mf_single_row_shape -v` passes.
- **Committed in:** `fabc7eb` (Task 2 commit)

**2. [Rule 1 — Minor Spec Drift] 7 single-row model tests shipped vs 6 planned**

- **Found during:** Task 2 test authoring
- **Issue:** Plan listed 6 model tests; writing them, the no-ghost-table structural check was needed separately for BPRMF and BasicMF (they are independent files with independent regression surfaces).
- **Fix:** Added `test_basic_mf_no_ghost_table` alongside `test_bpr_mf_no_ghost_table`. Both assert `'nn.Embedding(num_users' not in src` AND `'self.user_embeddings' not in src` on their respective module source files.
- **Impact:** Positive — extra regression guard; combined suite is 12 GREEN instead of planned 11.
- **Committed in:** `fabc7eb` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both minor spec-drift, Rule 1)
**Impact on plan:** No scope creep. Deviations strengthen the test surface without touching acceptance criteria.

## Issues Encountered

None. TDD RED → GREEN on both tasks; no test flakes; no unexpected interactions with pre-existing WIP.

## User Setup Required

None.

## Next Phase Readiness

- **Plan 03 (Wave 2: client_app + task.py contract wire + D-04..D-10 manifest-sidecar cache)** is now unblocked. It consumes:
  - `PersonalizedSplitFedAvg.aggregate_evaluate` keys (the 12 sufficient-stat fields) as the client-side `FitRes.metrics` payload contract.
  - The 2-key `get_local_parameters()` / `set_local_parameters()` contract for the disk cache (payload size ~516 B/user, not 3 MB/user).
  - `forward()` without `user_ids` for the per-user training loop.
- **Plan 04 (Wave 3: server_app.py)** can drop `PersonalizedSplitFedAvg/FedProx` directly into the Flower server loop without further model changes.
- Pre-existing uncommitted WIP in `federated-personalized-cf/client_app.py`, `server_app.py`, `task.py` is untouched (D-18 scope preserved; Plans 03/04 own those files).

---
*Phase: 03-personalized-migration, Plan 01*
*Completed: 2026-04-20*
