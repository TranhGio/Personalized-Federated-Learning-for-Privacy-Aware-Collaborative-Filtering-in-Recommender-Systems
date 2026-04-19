---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
last_updated: "2026-04-19T03:07:16.707Z"
progress:
  total_phases: 7
  completed_phases: 0
  total_plans: 6
  completed_plans: 1
---

# STATE: Federated Movie Recommendation — Cross-Device Migration & Thesis Evaluation

**Last updated:** 2026-04-19 after roadmap creation

## Project Reference

**Core value:** Under a correct cross-device protocol (1 user = 1 client, N=6040), the adaptive/hierarchical-conditional method beats all three baselines on NDCG@10 — including on sparse users — while the Flower PFedRec reproduces the IJCAI-23 reference within ±2 points.

**Current focus:** Phase 01 — foundation-contract

**Branch:** `feat/try_to_run_the_baseline` (existing; thesis work continues on this branch until milestone boundary is reached).

## Current Position

Phase: 01 (foundation-contract) — EXECUTING
Plan: 2 of 6

## Performance Metrics

Populated as phases complete. Primary thesis metric: `sampled_ndcg@10` (leave-one-out + 99 negatives).

| Module | Protocol | NDCG@10 | HR@10 | Sparse NDCG@10 | Notes |
|--------|----------|---------|-------|-----------------|-------|
| baseline | — | — | — | — | cross-device run pending |
| personalized | — | — | — | — | cross-device run pending |
| adaptive | — | — | — | — | cross-device run pending |
| pfedrec (paper_compat) | — | — | — | — | target: HR@10 ≈ 0.729 ± 2pts, NDCG@10 ≈ 0.441 ± 2pts |
| Phase 01-foundation-contract P01 | 5min | 2 tasks | 19 files |

## Accumulated Context

### Decisions

- **Migrate to cross-device** (1 user = 1 client, N=6040) for all four modules; cross-silo kept as explicit opt-in only.
- **Do NOT extract `fedrec_common/`** during this cycle — refactor risks invalidating the codebase map and bug audit mid-experiment.
- **Re-audit PFedRec bugs from scratch** against `IJCAI-23-PFedRec/` (don't trust the prior note list at face value — validate each against the reference).
- **Per-round sampling fraction `C`** is a swept hyperparameter, not fixed; defaults per module match the paper each module calibrates against.
- **New W&B project** for cross-device runs — keep cross-silo dashboards untouched.
- **Per-user-group (sparse/medium/dense) metrics** are first-class reported fields, not an afterthought.
- **Centralized baselines (SVD, NCF)** remain as-is — not re-evaluated under LOO+99neg.
- **DP / privacy quantification** deferred to v2.
- **Primary evaluator:** `sampled_loo_99` (NCF protocol). `allrank_*` is a namespaced secondary, never mixed into thesis tables.
- **Canonical reported metric:** `best_*` (best-round restored), not `last_*`.
- [Phase 01-foundation-contract]: Foundation package lives at scripts/foundation/ (not inside any federated-*-cf/ module and not at repo root) — neutral shared location avoids namespace collision and duplication while respecting PROJECT.md decision to defer fedrec_common/ extraction.
- [Phase 01-foundation-contract]: Plan 01 uses skip-stub TDD handoff: downstream plans un-skip by deleting pytestmark and replacing NotImplementedError bodies — enumerates all 31 expected FND-01..07 tests at pytest --collect-only while keeping every run green (2 passed, 31 skipped, 0 failed).
- [Phase 01-foundation-contract]: compute_raw_data_hash concatenation order is LOCKED to ratings.dat || movies.dat || users.dat — every FND-02/FND-07 downstream fingerprint depends on it; any future change invalidates committed split manifests and run manifests.

### Todos

*(Rolling list of actionable items surfaced during work, carried across sessions.)*

- None yet — phase planning will populate this as concrete tasks materialize.

### Blockers

- None.

### Open Questions

- What exact value of `fraction-train` should each module default to after migration? (To be decided during Phase 2–5 planning; treated as a swept hyperparameter.)
- Should the canonical ID-mapping / split-manifest artifact live under `.planning/artifacts/` or under a new `data/artifacts/` path? (Decided during Phase 1 planning.)

## Session Continuity

**Last session summary (2026-04-19):** Initialized planning — drafted PROJECT.md, REQUIREMENTS.md (52 v1 items across FND / BSL / PFR / PSN / ADP / EVL / THS), research outputs (SUMMARY, ARCHITECTURE, FEATURES, PITFALLS), and the codebase map (ARCHITECTURE, CONCERNS). Roadmap drafted as 7 phases honoring the dependency structure: FND blocks everything → {BSL, PSN, ADP, PFR} parallelizable → EVL cross-cutting → THS last.

**Next session entry point:** Run `/gsd:plan-phase 1` to decompose the Foundation Contract phase into executable plans (canonical ID mapping, deterministic split manifest, exclusion set, primary-evaluator declaration, weight policy, run-scoped seeding, run manifest schema).

**Key files to reread on session resume:**

- `.planning/ROADMAP.md` — phase structure and success criteria
- `.planning/REQUIREMENTS.md` — traceability table
- `.planning/research/ARCHITECTURE.md` — migration deltas and build-order implications
- `.planning/codebase/CONCERNS.md` — known bugs to re-verify during migration

---
*State initialized: 2026-04-19 alongside roadmap creation.*
