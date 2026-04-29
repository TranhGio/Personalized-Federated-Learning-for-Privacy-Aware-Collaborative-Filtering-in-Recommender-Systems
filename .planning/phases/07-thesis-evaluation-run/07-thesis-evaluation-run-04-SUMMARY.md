---
phase: 07-thesis-evaluation-run
plan: 04
subsystem: result-aggregation
tags: [aggregator, markdown, csv, atomic-write, d-11-win-criterion, d-20-hard-fail, thesis-tables]

# Dependency graph
requires:
  - phase: 07-thesis-evaluation-run-01
    provides: atomic_write_text helper, _THESIS_CROSSDEVICE_MAIN ModeProfile, RunManifest schema v3 with thesis_run_label / ablation_dimension / ablation_value fields
  - phase: 07-thesis-evaluation-run-02
    provides: per-module server_app + pyproject mutation that populates thesis_run_label / ablation_dimension / ablation_value in the embedded _manifest of every results.json
  - phase: 06-evaluation-reporting-harness
    provides: nested final_metrics={best,last,best_round,last_round,final_eval_round_index} schema with sampled_ndcg@10 + per-group variants under final_metrics['best']
provides:
  - "scripts/thesis/aggregate_results.py: standalone aggregator reading per-run results.json, filtering by _manifest.thesis_run_label != '' AND run_seed in {42, 1337, 2026}, computing mean+/-std across seeds, emitting 6 markdown+CSV files under results/federated/_thesis/"
  - "D-20 hard-fail-on-missing-cells: aggregator imports build_main_matrix + build_ablation_matrix from orchestrator (single source of truth for expected-cell set), prints sorted (module, label, seed) tuples to stderr and exits 1 when cells are absent"
  - "D-11 win criterion (strict >, non-overlapping ±1σ intervals): adaptive must beat baseline + personalized only; PFedRec excluded from win comparison per D-05; HR cells are informational"
  - "D-24 cell format `0.4123 ± 0.0089` (4 decimal places, em-dash on missing data); winners bolded as `**0.4123 ± 0.0089**`"
  - "scripts/foundation/tests/test_thesis_aggregator.py: 15 tests covering all 11 VALIDATION.md rows (7-04-01..7-04-11) + 4 bonus regression tests"
affects: [07-thesis-evaluation-run-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Aggregator-as-pure-function: glob -> filter -> aggregate -> render -> atomic write; idempotent; reads orchestrator's matrix for expected-cell set"
    - "Slash-uniform HR/NDCG keys + delimiter-tolerant evaluated_users extraction (Pitfall 1 mitigation)"
    - "Sparse-evaluable-only filter: drop seeds with evaluated_users{_,/}sparse == 0 from sparse aggregation; emit n_seeds_with_sparse=K/3 footnote"
    - "Population std (np.std(arr, ddof=0)) for thesis-reporting convention"
    - "PFedRec drift detection: mean HR/NDCG outside ±2pts of IJCAI-23 reference triggers inline markdown note (no halt)"

key-files:
  created:
    - "scripts/thesis/aggregate_results.py"
    - "scripts/foundation/tests/test_thesis_aggregator.py"
    - ".planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-04-SUMMARY.md"
  modified: []

key-decisions:
  - "Aggregator imports build_main_matrix + build_ablation_matrix from scripts.thesis.run_thesis_sweep (Plan 03) — single source of truth for the D-20 expected-cell set; eliminates drift between orchestrator and aggregator."
  - "extract_evaluated_users handles BOTH delimiter conventions ('/' for pfedrec, '_' for others) via two-key probe; HR/NDCG metric keys use slash UNIFORMLY across all 4 modules per Pitfall 1."
  - "Population std (ddof=0) chosen over sample std (ddof=1) — matches the de facto thesis-reporting convention with N=3 seeds where the difference is small but the choice must be deterministic."
  - "Sparse-evaluable-only filter is OPT-IN via aggregate_by_seed parameter; main_comparison.csv overall NDCG/HR keep all 3 seeds; sparse columns drop the zero-evaluable seed and surface n_seeds_with_sparse=K/3 footnote in markdown."
  - "Aggregator is read-only relative to results/; --check-only mode verifies expected-cell-set without writing any files (pre-aggregation gate for Plan 05)."

patterns-established:
  - "Pattern: D-20 hard-fail surface — compute expected = expected_main_cells() | expected_ablation_cells(); observed = {(r.module, r.thesis_run_label, r.run_seed) for r in records}; missing = sorted(expected - observed); if missing: print sorted tuples and SystemExit(1)."
  - "Pattern: Atomic markdown/CSV write — atomic_write_text(str(path), render_*(...)) wraps tempfile + os.replace; 6 files emitted in one transaction-like block; no .tmp-* leftovers visible after run."
  - "Pattern: Bold-the-winner without HR pollution — in render_main_md, only NDCG cells are bolded (HR cells are informational); pfedrec rows are pre-excluded from the comparable set for win evaluation; PFedRec footnote marker added separately."
  - "Pattern: Synthetic-fixture testing — _make_results_data + _write_synthetic_run + _seed_full_main_matrix + _seed_full_ablation_matrix factory functions in test file produce on-disk results-root layout matching the Phase-6 D-07 schema; tests use tmp_path; no live flwr runs."

requirements-completed: [THS-03, THS-04, THS-05, THS-06, THS-07]

# Metrics
duration: ~15min
completed: 2026-04-29
---

# Phase 7 Plan 04: Thesis Result Aggregator Summary

**Read-only Python aggregator that globs `results/federated/<module>/*/results.json`, filters by `_manifest.thesis_run_label` + canonical seeds, computes mean ± std across seeds with D-11 win-bolding + D-20 missing-cell hard-fail, and emits 6 atomic markdown/CSV files (main_comparison + ablations + sparse_slice) under `results/federated/_thesis/`.**

## Performance

- **Duration:** ~15 min (Task 1 source 8 min + Task 2 tests 5 min + verification 1 min + summary 1 min)
- **Started:** 2026-04-29T21:27:00Z
- **Completed:** 2026-04-29T21:42:00Z
- **Tasks:** 2
- **Files modified:** 2 (1 source + 1 test) plus 1 SUMMARY.md

## Accomplishments

- **Aggregator (D-19):** Single-file Python script (`scripts/thesis/aggregate_results.py`, 726 lines, executable) implementing the full pipeline: glob → filter → extract → aggregate → check → render → atomic write. Public API surface (8 symbols): `collect_thesis_results`, `extract_metric`, `extract_evaluated_users`, `aggregate_by_seed`, `find_missing_cells`, `expected_main_cells`, `expected_ablation_cells`, `fmt_cell`, `is_winner`, `run_aggregator`, `main`, `ThesisResult` (dataclass).
- **D-20 hard-fail-on-missing-cells closure:** Aggregator imports `build_main_matrix` + `build_ablation_matrix` from Plan 03's orchestrator → 12 main + 21 ablation = 33 expected cells. Empty-results-root smoke run prints all 33 missing tuples (sorted alphabetically by module/label/seed) and exits with code 1.
- **D-11 win criterion closure:** `is_winner(my_mean, my_std, others)` implements `my_lower > other_mean + other_std` STRICTLY (not >=); empty-others returns False (cannot win against nothing); applied per-metric to baseline/personalized/adaptive only with PFedRec excluded per D-05.
- **D-24 cell format closure:** `fmt_cell(0.4123, 0.0089) == "0.4123 ± 0.0089"`; missing data → em-dash sentinel `"—"`; pinned by `test_cell_format`.
- **D-08 PFedRec footnote closure:** `_PFEDREC_FOOTNOTE` constant matches the verbatim CONTEXT.md text; emitted at the bottom of `main_comparison.md` and `sparse_slice.md`; PFedRec cells get `†` marker; PFedRec excluded from win comparison.
- **D-15 sparse-slice closure:** Dedicated `sparse_slice.{md,csv}` with all 4 main modules + every ablation cell; partial-seed footnote `n_seeds_with_sparse=K/3` emitted when 0 < K < 3.
- **D-17 6-file output closure:** All 6 files written via `atomic_write_text` from Plan 01 — main_comparison.{md,csv}, ablations.{md,csv}, sparse_slice.{md,csv}.
- **Test coverage:** 15 tests covering all 11 VALIDATION.md aggregator rows + 4 bonus regression tests; full test suite passes in ~0.07s; foundation suite went 124 → 139 fast tests passing (no regression).

## Task Commits

Each task was committed atomically with `--no-verify` per parallel-executor protocol:

1. **Task 1: Aggregator implementation** — `3fd8741` (feat)
   - scripts/thesis/aggregate_results.py: 726 lines, executable; 8 public functions + 1 dataclass; reads orchestrator constants for expected-cell set
2. **Task 2: Aggregator unit tests** — `9b61207` (test)
   - scripts/foundation/tests/test_thesis_aggregator.py: 481 lines; 15 tests covering all 11 VALIDATION rows plus 4 bonus regressions

**Plan metadata commit:** Will be created with this SUMMARY.md + STATE.md + ROADMAP.md + REQUIREMENTS.md updates.

_Note: Task 1 = `feat`, Task 2 = `test`. Both used `--no-verify` per `<parallel_execution>` protocol (Wave 3 parallel with Plan 03)._

## Files Created/Modified

### Source

- `scripts/thesis/aggregate_results.py` (NEW, 726 lines, mode 755) — Phase 7 thesis result aggregator. Imports `THESIS_SEEDS`, `ThesisCell`, `build_main_matrix`, `build_ablation_matrix` from Plan 03's orchestrator (single source of truth for expected-cell set). Imports `atomic_write_text` from Plan 01's foundation extension. Exports 11 module-level public symbols + `ThesisResult` dataclass. Sub-pipeline functions are prefixed `_` (private).

### Tests

- `scripts/foundation/tests/test_thesis_aggregator.py` (NEW, 481 lines) — 15 tests covering all 11 VALIDATION.md rows (7-04-01..7-04-11) + 4 bonus regression tests. Synthetic-fixture helpers: `_make_results_data`, `_write_synthetic_run`, `_seed_full_main_matrix`, `_seed_full_ablation_matrix` produce on-disk results-root layouts matching the Phase-6 D-07 schema. Tests use `tmp_path`; no live flwr runs.

## Test Coverage Map (VALIDATION.md rows 7-04-01..7-04-11)

| VALIDATION row | Test name | Status |
|---|---|---|
| 7-04-01 (extract overall NDCG@10) | `test_extract_overall_ndcg10` | PASSED |
| 7-04-02 (uniform slash for HR/NDCG) | `test_extract_sparse_ndcg10_uniform_slash` | PASSED |
| 7-04-03 (D-11 positive) | `test_d11_win_criterion` | PASSED |
| 7-04-04 (D-11 negative / overlap) | `test_d11_overlap_no_winner` | PASSED |
| 7-04-05 (sparse partial-seed Pitfall 10) | `test_sparse_partial_seeds` | PASSED |
| 7-04-06 (ablation label grouping) | `test_ablation_label_grouping` | PASSED |
| 7-04-07 (D-20 missing-cell hard-fail) | `test_d20_hard_fail_missing` | PASSED |
| 7-04-08 (CSV per-group columns) | `test_csv_per_group_columns` | PASSED |
| 7-04-09 (6 output files) | `test_six_output_files` | PASSED |
| 7-04-10 (atomic write — no .tmp-*) | `test_atomic_write_no_tmp` | PASSED |
| 7-04-11 (D-24 cell format) | `test_cell_format` | PASSED |

Bonus regression tests (4 additional):

| Test name | Coverage |
|---|---|
| `test_extract_evaluated_users_handles_both_delimiters` | Pitfall 1: pfedrec slash + non-pfedrec underscore both accepted by `extract_evaluated_users` |
| `test_winner_bolded_in_main_md` | D-11 + Discretion: rendered `main_comparison.md` contains `**...**` bolded cells |
| `test_collect_filters_legacy_phase6_manifests` | Pitfall 7: legacy schema_version=2 manifests with no `thesis_run_label` are filtered out by aggregator |
| `test_check_only_does_not_write_files` | `--check-only` flag does not create `_thesis_out/` (pre-aggregation gate behavior) |

## Public API Surface

| Symbol | Kind | Purpose |
|---|---|---|
| `ThesisResult` | dataclass | One results.json record matched to a thesis cell |
| `collect_thesis_results(results_root) -> List[ThesisResult]` | function | Glob results.json files; filter to thesis-tagged runs |
| `extract_metric(data, metric_key) -> Optional[float]` | function | Read `final_metrics['best'][metric_key]` |
| `extract_evaluated_users(data, group) -> int` | function | Pitfall 1: handles both `/` and `_` delimiters |
| `aggregate_by_seed(records, metric_key, sparse_evaluable_only=False) -> Dict[(module,label), (mean,std,n)]` | function | Group + compute mean/std/n |
| `expected_main_cells() -> Set[Tuple[str,str,int]]` | function | D-20 expected cell set: 12 tuples |
| `expected_ablation_cells() -> Set[Tuple[str,str,int]]` | function | D-20 expected cell set: 21 tuples |
| `find_missing_cells(records, expected) -> List[Tuple]` | function | D-20 hard-fail surface; sorted output |
| `fmt_cell(mean, std) -> str` | function | D-24 format: `"0.4123 ± 0.0089"` or `"—"` |
| `is_winner(my_mean, my_std, others) -> bool` | function | D-11 strict-> win criterion |
| `run_aggregator(results_root, output_dir, check_only=False) -> int` | function | Top-level orchestration |
| `main(argv) -> int` | function | CLI entry point with --results-root / --output-dir / --check-only |

## `--check-only` Flag Behavior

- **Default mode:** `run_aggregator(..., check_only=False)` performs glob → filter → expected-cell check → render → atomic write. On D-20 missing cells: SystemExit(1). On success: writes 6 files, returns 0.
- **--check-only mode:** `run_aggregator(..., check_only=True)` performs glob → filter → expected-cell check → return 0 BEFORE rendering. On D-20 missing cells: SystemExit(1) (same behavior). On success: prints `"[INFO] --check-only: N records present, expected set complete. No files written."` and returns 0. The output directory is NEVER created or touched in this mode (pinned by `test_check_only_does_not_write_files`).
- **Use case:** Pre-aggregation gate for Plan 05 manual runbook — runners can verify all 33 cells are on disk before kicking off the full aggregation.

## D-08 PFedRec Footnote (verbatim)

```
† PFedRec (paper-faithful) — `dim=32, SGD lr=0.1, BCE, fraction-train=1.0; matches IJCAI-23 reference within ±2 points`. Not counted toward "adaptive beats baselines" claim per Phase 7 D-05.
```

This text is stored verbatim in the `_PFEDREC_FOOTNOTE` module-level constant in `aggregate_results.py` and emitted as the bottom line of both `main_comparison.md` and `sparse_slice.md`.

## Decisions Made

1. **Population std (ddof=0) over sample std (ddof=1):** Matches the de facto thesis-reporting convention. Plan-text guidance: "Use `np.std(arr, ddof=0)` (population std) — matches the de facto thesis reporting convention." With N=3 seeds, the difference between ddof=0 and ddof=1 is `sqrt(3/2) ≈ 1.22x` — non-trivial but consistent across all rows so winner ranking is preserved.
2. **`--check-only` returns 0 on success without writing files:** Useful as a pre-aggregation gate (Plan 05 runbook can verify all cells exist before triggering the long-running render pipeline). Pinned by `test_check_only_does_not_write_files`.
3. **HR cells NOT bolded:** Per Plan's `<action>` block "HR cells are informational — bolding is NDCG-only per Open Question 3 in RESEARCH.md." Implemented via `if metric.startswith("ndcg10")` gate around the bold-the-winner logic.
4. **Sparse-evaluable-only is opt-in:** `aggregate_by_seed` accepts `sparse_evaluable_only: bool = False`; non-sparse callers (overall NDCG/HR) keep all 3 seeds; sparse-column callers (sparse NDCG/HR) drop seeds with `evaluated_users{_,/}sparse == 0` and the markdown surface adds `n_seeds_with_sparse=K/3` footnote when K < 3.
5. **PFedRec drift is a soft warning, not a halt:** `_check_pfedrec_drift` returns a markdown string (or None) inserted at the top of `main_comparison.md`; aggregator never raises on drift. Per Plan: "PFedRec drift is reported as an inline markdown note, not a hard fail."

## Deviations from Plan

None - plan executed exactly as written. The `<action>` block specified "Create `scripts/thesis/aggregate_results.py` with EXACT content" and that content was written verbatim. The 11 plan-required tests + 4 bonus tests in the test file match the action-block content exactly.

## Issues Encountered

None.

## Self-Check: PASSED

Verified before STATE update:

- `scripts/thesis/aggregate_results.py` — exists, executable (mode 755), 726 lines (FOUND)
- All 8+ public symbols importable: `collect_thesis_results`, `extract_metric`, `extract_evaluated_users`, `aggregate_by_seed`, `find_missing_cells`, `expected_main_cells`, `expected_ablation_cells`, `fmt_cell`, `is_winner`, `run_aggregator`, `main`, `ThesisResult` (FOUND)
- `expected_main_cells()` returns 12 tuples; `expected_ablation_cells()` returns 21 tuples (FOUND via API smoke check)
- `fmt_cell(0.4123, 0.0089) == "0.4123 ± 0.0089"` per D-24 (FOUND via API smoke check)
- `is_winner` correctly identifies non-overlapping vs overlapping intervals per D-11 (FOUND)
- Empty results-root + `--check-only` prints `Missing 33 cells` and exits 1 (D-20 hard-fail) (FOUND)
- `scripts/foundation/tests/test_thesis_aggregator.py` — exists, 481 lines (FOUND)
- All 15 tests GREEN: `pytest scripts/foundation/tests/test_thesis_aggregator.py -x -v` reports `15 passed in 0.07s` (FOUND)
- Foundation suite (fast) GREEN: `pytest scripts/foundation/tests/ -m "not slow" -q` reports `139 passed, 4 deselected, 0 failures` (FOUND — 124 pre-Plan-04 + 15 new = 139)
- Commit `3fd8741` (feat: aggregator) — present in `git log` (FOUND)
- Commit `9b61207` (test: aggregator tests) — present in `git log` (FOUND)
- `_PFEDREC_FOOTNOTE` constant matches D-08 text verbatim (FOUND via grep + plan reference comparison)

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Plan 05 ready:** Manual runbook can document the canonical incantations:
  - `python scripts/thesis/run_thesis_sweep.py --phase=main` (~19.5hr; produces 12 main results.json files)
  - `python scripts/thesis/run_thesis_sweep.py --phase=ablation` (~31.5hr; produces 21 ablation results.json files)
  - `python scripts/thesis/aggregate_results.py --check-only` (pre-aggregation gate; 0 = all 33 cells present)
  - `python scripts/thesis/aggregate_results.py` (renders 6 thesis files under `results/federated/_thesis/`)
- **Plan 03 contract honored:** Aggregator imports `THESIS_SEEDS`, `ThesisCell`, `build_main_matrix`, `build_ablation_matrix` from `scripts.thesis.run_thesis_sweep` — Plan 03's public API. The wave-validation step verifies both plans' files coexist cleanly.
- **D-20 hard-fail tested end-to-end:** A 32/33 fixture (drop one cell) correctly fails with `Missing 1 cells` listing the exact tuple — `test_d20_hard_fail_missing` pins this end-to-end. The full 33-cell fixture passes through to render all 6 files.

No blockers. Plan 04's deliverable is the read-only consumption side of the Phase-7 thesis pipeline; Plan 05's manual runbook + smoke run is the next step.

---
*Phase: 07-thesis-evaluation-run*
*Plan: 04*
*Completed: 2026-04-29*
