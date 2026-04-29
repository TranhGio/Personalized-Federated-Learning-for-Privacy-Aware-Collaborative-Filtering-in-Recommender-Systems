---
phase: 07-thesis-evaluation-run
plan: 05
subsystem: thesis-runbook
tags: [runbook, uat, checkpoint-pause, manual-execution-gate, operational-doc]

# Dependency graph
requires:
  - phase: 07-thesis-evaluation-run-01
    provides: _THESIS_CROSSDEVICE_MAIN ModeProfile + RunManifest schema v3 thesis-tagging fields
  - phase: 07-thesis-evaluation-run-02
    provides: 4 server_app.py mode-tuple gates + manifest-mutation thesis kwargs + pyproject default sentinels
  - phase: 07-thesis-evaluation-run-03
    provides: scripts/thesis/run_thesis_sweep.py orchestrator with 12-cell main + 21-cell ablation matrices
  - phase: 07-thesis-evaluation-run-04
    provides: scripts/thesis/aggregate_results.py with D-20 hard-fail, D-11 win-bolding, 6-file output
provides:
  - "07-thesis-evaluation-run-05-RUNBOOK.md — operator-facing 5-gate execution guide (A pre-flight, B main 19.5hr, C ablation 31.5hr, D pre-aggregation, E aggregation+verification)"
  - "07-thesis-evaluation-run-05-UAT.md — gate-by-gate pass/fail checklist with paste-target tables for main_comparison + sparse_slice and THS-01..THS-07 final-closure block"
  - "Checkpoint pause at Task 2 (Gate A) — Tasks 2-5 require ~50hr of human-supervised GPU time and return control to the orchestrator awaiting operator action"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Operational runbook pattern: Gates as numbered sub-sections with explicit Bash commands + Expected output blocks; a Closing Checklist at the bottom for sign-off; both success and failure paths (Outcome A / B-1 / B-2) documented inline"
    - "UAT-as-checkbox-template: parallel structure to runbook (one ⬜ status column per check), paste-target code blocks where the operator drops the actual table contents, dedicated D-12 contingency section that's N/A on Outcome A and a full decision tree on Outcome B"
    - "Checkpoint pause at human-only gates: Task 1 produces the docs autonomously; Tasks 2-5 are human-time and explicitly NOT auto-approved (overriding auto-mode default) because the underlying work is ~50hr of GPU wallclock that cannot be skipped or simulated"

key-files:
  created:
    - ".planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-RUNBOOK.md"
    - ".planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-UAT.md"
  modified: []

key-decisions:
  - "Auto-mode default for checkpoint:human-verify (auto-approve) is OVERRIDDEN at Task 2 because Tasks 2-5 represent ~50hr of real GPU time, not visual verification of already-completed work. Auto-approving would record false PASS on un-executed work and break THS-03/THS-04 reliability."
  - "RUNBOOK + UAT written as STATIC operational documents (not generated at execution time). They reference the existing CLI signatures from Plans 03 + 04 verbatim — no new code is shipped by Plan 05; all execution machinery was built in prior plans."
  - "Gate A's D-20 demo (A.4) explicitly tests the aggregator's hard-fail behavior on a partial 1-of-33 fixture — this validates the safety net BEFORE committing to the long-running Gates B + C. If A.4 reports a different missing-cell count or doesn't exit 1, debug the aggregator before touching the main matrix."
  - "D-12 contingency documented inline in BOTH the runbook (Gate E.5) AND the UAT (E.5 checkbox section) so a thesis-failing run has an explicit decision path without requiring a follow-up planning session."

patterns-established:
  - "Pattern: Long-running experiment pause via checkpoint — when a plan's tasks include >1hr of execution that cannot be auto-approved (GPU runs, real-world data collection, etc.), Task 1 ships the docs/runbook autonomously and Tasks 2..N are checkpoint:human-verify gates that return structured pause messages to the orchestrator."
  - "Pattern: Runbook companion with UAT — for any plan involving manual operator gates, ship two parallel documents: a procedural RUNBOOK.md (here are the commands) and a status UAT.md (here is where you record the outcome). The UAT serves as the audit trail Phase 7 SUMMARY references for thesis-claim attribution."
  - "Pattern: Outcome-tree documentation — document Outcome A (success), B-1 (recovery via ablation), and B-2 (replan) explicitly inline so the operator never has to make a decision the planner didn't anticipate. PROJECT.md core value is referenced from B-2 as the trigger for thesis-level replanning."

requirements-completed: []
# Note: requirements THS-02, THS-03, THS-04, THS-05, THS-06, THS-07 are
# CONDITIONALLY tagged in PLAN.md frontmatter but cannot be marked complete
# until Tasks 2-5 (the actual ~50hr matrix execution + aggregation) finish.
# Task 1 (this commit) ships the operational scaffolding only.

# Metrics
duration: ~2min (Task 1 only — Tasks 2-5 are human-time, ~50hr)
completed: 2026-04-29 (Task 1; Tasks 2-5 PAUSED at checkpoint)
---

# Phase 7 Plan 05: Operational Runbook + UAT Summary (Task 1 of 5 — CHECKPOINT PAUSED)

**Operational documentation shipped (RUNBOOK + UAT) covering the 5-gate execution path (A pre-flight 1.5hr → B main matrix 19.5hr → C ablation matrix 31.5hr → D pre-aggregation count → E aggregation + thesis-claim verification). Plan 05 is intentionally PAUSED at Task 2 (Gate A) — Tasks 2-5 require ~50hr of human-supervised GPU wallclock that cannot be auto-approved and must be executed by the operator following the runbook.**

## Performance

- **Duration:** ~2 min for Task 1 (autonomous doc-writing); Tasks 2-5 pending operator execution at ~50hr wallclock
- **Started:** 2026-04-29T14:38:00Z
- **Task 1 completed:** 2026-04-29T14:40:10Z (CHECKPOINT PAUSE at Task 2)
- **Tasks (Plan 05 total):** 5 (1 autonomous + 4 checkpoint:human-verify)
- **Tasks completed:** 1 of 5 (Task 1 only — RUNBOOK + UAT docs)
- **Files created:** 2 (RUNBOOK + UAT)

## Accomplishments

### Task 1 — Operational documents shipped

- **`07-thesis-evaluation-run-05-RUNBOOK.md`** (~138 lines, ~5KB): Step-by-step execution guide with 5 numbered gates:
  - **Gate A** (~1.5 hr): Pre-flight smoke — single-cell run + manifest-fields verification + idempotency check + D-20 hard-fail demo (4 sub-steps).
  - **Gate B** (~19.5 hr): Main matrix execution — `nohup ... --phase=main` + progress monitoring + end-of-run cell count + failed-cell handling (4 sub-steps).
  - **Gate C** (~31.5 hr): Ablation matrix execution — same shape as Gate B with `--phase=ablation` (4 sub-steps).
  - **Gate D**: Pre-aggregation count — bash one-liner that asserts `main=12 ablation=21 total=33` BEFORE invoking the aggregator (D-20 safety net).
  - **Gate E**: Aggregation + thesis-claim verification — `python scripts/thesis/aggregate_results.py` + visual inspection of `main_comparison.md` / `sparse_slice.md` / `ablations.md` + D-12 contingency tree (Outcome A / B-1 / B-2) + PFedRec calibration check (6 sub-steps).
  - Closing checklist at bottom for operator sign-off.
- **`07-thesis-evaluation-run-05-UAT.md`** (~135 lines, ~5KB): Gate-by-gate pass/fail tracking template:
  - Per-gate status tables with `⬜ PASS / ⬜ FAIL` checkboxes for each sub-step.
  - Per-cell tables for Gate B (4 modules × 3 seeds) and Gate C (7 ablations × 3 seeds) with `⬜` per intersection.
  - Paste-target code blocks under E.2 + E.3 where the operator drops actual `main_comparison.md` and `sparse_slice.md` table contents.
  - Per-claim THS-03 / THS-04 win-check arithmetic (D-11) with fillable mean ± std fields.
  - D-12 contingency decision tree (Outcome A / B-1 / B-2 mutually-exclusive checkboxes).
  - Final-closure table mapping THS-01..THS-07 to per-requirement Phase 7 status.

### Task 1 — Verification

Task 1's `<verify>` block invariants confirmed:
- Both RUNBOOK + UAT files exist on disk under `.planning/phases/07-thesis-evaluation-run/`.
- RUNBOOK contains `"Gate A — Pre-flight smoke"` and `"Gate E — Aggregation"` headers.
- UAT contains `"THS-04 win check"` (D-11 thesis-claim arithmetic block).
- Verification command exit 0; printed `Runbook + UAT documents created`.

## Task Commits

1. **Task 1: RUNBOOK + UAT** — `663d398` (docs)
   - `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-RUNBOOK.md` (NEW)
   - `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-UAT.md` (NEW)
   - Atomic commit with `--no-verify` per parallel-executor protocol.

**Plan metadata commit:** Will be created with this SUMMARY.md + STATE.md + ROADMAP.md updates.

## Tasks 2-5 — CHECKPOINT PAUSED (awaiting operator)

| Task | Gate | Type | Wallclock | Status |
|------|------|------|-----------|--------|
| Task 2 | Gate A — Pre-flight smoke + D-20 demo | checkpoint:human-verify | ~1.5 hr GPU | ⏸ PAUSED |
| Task 3 | Gate B — Main matrix (12 cells) | checkpoint:human-verify | ~19.5 hr GPU | ⏸ PAUSED |
| Task 4 | Gate C — Ablation matrix (21 cells) | checkpoint:human-verify | ~31.5 hr GPU | ⏸ PAUSED |
| Task 5 | Gates D + E — Aggregation + thesis-claim verification | checkpoint:human-verify | minutes (post-matrix) | ⏸ PAUSED |

**Why paused:** Tasks 2-5 represent ~50 hours of real GPU wallclock executing the canonical thesis matrix on real hardware. These gates cannot be auto-approved because the underlying work IS the canonical thesis claim — auto-approving would falsely mark un-executed work as PASSED and break the THS-03 / THS-04 attribution chain.

**Resume protocol:** When the operator finishes a gate and updates the corresponding UAT section, resume Plan 05 by re-running `/gsd:execute-phase 07` (or whatever the current GSD harness convention is). The continuation agent will see Task 1 + the relevant Gate's commit hashes in the prompt's `<completed_tasks>` block and resume from the next pending checkpoint task.

**Per-gate resume signals:**
- Gate A → `"Gate A passed"` with smoke-run `run_id`.
- Gate B → `"Gate B passed"` with final main-cell count (12).
- Gate C → `"Gate C passed"` with final ablation-cell count (21).
- Gate E → `"Phase 7 complete (Outcome A)"` / `"complete-with-caveat (Outcome B-1)"` / `"failed (Outcome B-2)"`.

## Files Created/Modified

### Created (Task 1)
- `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-RUNBOOK.md` — 5-gate operational runbook
- `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-UAT.md` — gate-by-gate UAT checklist

### Modified
- None during Task 1 (per `<scope>` boundary — Plan 05 ships docs, not code).

## Decisions Made

1. **Auto-mode override at Task 2:** Auto mode is active per session config, but the user's explicit prompt instruction at the orchestrator-spawn level overrides the default `checkpoint:human-verify` auto-approve behavior because these specific gates represent real GPU time, not visual verification. Auto-approving Tasks 2-5 would falsify the thesis-claim attribution chain (THS-03 / THS-04 require evidence of actual matrix execution, not a checkbox). The override is per-plan, not global.
2. **Static documents, not generated:** Both RUNBOOK + UAT are written as fixed markdown content per the plan's `<action>` block (verbatim "EXACT content" specification). They do NOT pull from any runtime state, do NOT include timestamps, and do NOT include W&B URLs (the `<your-entity>` placeholder is preserved so the operator substitutes their own at run time). This matches the Phase 7 D-21 W&B-routing pattern (operator owns the entity string).
3. **Closing checklist + final-closure table separation:** RUNBOOK ends with a 7-item operator sign-off list (per-gate boolean); UAT ends with a 7-row THS-XX status table (per-requirement boolean). Both are needed: the runbook closes the procedural loop (did you run all the gates?), the UAT closes the requirements loop (did each thesis claim pass?).
4. **D-12 contingency in BOTH docs:** Documented inline at Gate E.5 (RUNBOOK) and as the dedicated E.5 checkbox section (UAT) so a thesis-failing run has an explicit decision path without requiring a follow-up planning session. PROJECT.md "Core Value" is referenced from Outcome B-2 as the trigger for thesis-level replanning via `/gsd:plan-phase`.

## Deviations from Plan

None — Task 1 executed exactly as written. The `<action>` block's "EXACT content" specification was followed verbatim for both RUNBOOK and UAT. Tasks 2-5 are paused at the planned checkpoint:human-verify boundaries.

## Issues Encountered

None during Task 1. Tasks 2-5 will encounter their own issues at execution time (CUDA OOM, foundation-bundle issues, etc.) — the runbook documents the standard recovery path (`--retry-failed`) for each.

## User Setup Required

For Tasks 2-5 (per RUNBOOK Prerequisites section):

- **W&B login active:** `wandb login` (or `WANDB_API_KEY` env var) BEFORE Gate A.
- **Foundation bundle on disk:** `data/derived/foundation_index.json` must exist (Phase 1 deliverable; verifiable via `pytest scripts/foundation/tests/ -ra`).
- **Plans 01-04 merged:** All foundation extensions, server_app wiring, orchestrator, and aggregator must be on the current branch.

No new external service configuration introduced by Plan 05.

## Self-Check: PASSED

Verified before STATE update:

- `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-RUNBOOK.md` — exists (FOUND)
- `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-UAT.md` — exists (FOUND)
- RUNBOOK contains "Gate A — Pre-flight smoke" header (FOUND via grep)
- RUNBOOK contains "Gate E — Aggregation" header (FOUND via grep)
- UAT contains "THS-04 win check" (FOUND via grep)
- Task 1 commit `663d398` (docs) — present in `git log` (FOUND)
- Task 1 verification command exit 0 with stdout `Runbook + UAT documents created` (FOUND)

## Known Stubs

None. The RUNBOOK and UAT are documentation deliverables for the operator; their content is the deliverable, not data wiring. The paste-target code blocks (`[paste main_comparison.md table contents here]` etc.) are intentional placeholders the operator fills at Gate E execution time — they are NOT data stubs in the code-wiring sense.

## Next Phase Readiness

- **Tasks 2-5 (checkpoint pause):** Operator follows the runbook gate by gate. After each gate, fills the corresponding UAT section. After Gate E completes, returns the resume signal so the continuation agent can write the final SUMMARY.md amendment + close Phase 7 STATE.md with the thesis-claim outcome.
- **Phase 7 closure:** Will happen after Task 5 (Gate E) reports an outcome. The continuation agent's responsibilities will include:
  - Updating this SUMMARY.md to mark Tasks 2-5 as completed (with observed wallclock vs estimate, failure counts, and the Gate-E table contents pasted in).
  - Updating the `requirements-completed:` frontmatter field (currently empty pending Tasks 2-5) with `[THS-02, THS-03, THS-04, THS-05, THS-06, THS-07]`.
  - Marking the relevant THS requirements complete via `gsd-tools requirements mark-complete` based on the Gate-E outcome (full set on Outcome A; subset on Outcome B-1 with documented variant; none on Outcome B-2 with replan trigger).
  - Updating `.planning/STATE.md` with the canonical thesis-claim outcome and per-module mean ± std on `final_metrics["best"]["sampled_ndcg@10"]`.

---
*Phase: 07-thesis-evaluation-run*
*Plan: 05*
*Task 1 completed: 2026-04-29 — RUNBOOK + UAT shipped*
*Tasks 2-5: PAUSED at checkpoint:human-verify (~50hr GPU wallclock pending)*
