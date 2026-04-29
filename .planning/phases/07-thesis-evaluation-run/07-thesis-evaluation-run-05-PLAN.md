---
phase: 07-thesis-evaluation-run
plan: 05
type: execute
wave: 4
depends_on:
  - 07-thesis-evaluation-run-01-PLAN.md
  - 07-thesis-evaluation-run-02-PLAN.md
  - 07-thesis-evaluation-run-03-PLAN.md
  - 07-thesis-evaluation-run-04-PLAN.md
files_modified:
  - .planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-RUNBOOK.md
  - .planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-UAT.md
autonomous: false
requirements:
  - THS-02
  - THS-03
  - THS-04
  - THS-05
  - THS-06
  - THS-07
user_setup: []

must_haves:
  truths:
    - "A 1-cell smoke run completes via `python scripts/thesis/run_thesis_sweep.py --module=adaptive --seed=42 --phase=main` and lands a single results.json + manifest.json on disk with thesis_run_label='main'"
    - "Re-running the same single-cell command immediately after the first reports `[SKIP]` (idempotent skip-on-existing)"
    - "Pre-aggregation aggregator hard-fails with `Missing 32 cells` after the smoke run (D-20 demonstration)"
    - "Full main matrix completes (12 results.json files: 3 baseline + 3 personalized + 3 adaptive at thesis_crossdevice_main + 3 pfedrec at paper_compat_pfedrec)"
    - "Full ablation matrix completes (21 results.json files)"
    - "Aggregator with full coverage (33 cells) emits 6 output files under results/federated/_thesis/"
    - "main_comparison.md table renders 4 module rows (baseline / personalized / adaptive / pfedrec); adaptive row's NDCG cells are bolded (D-11) OR the table documents the negative result with ablation recovery analysis (D-12)"
    - "sparse_slice.md exists with the THS-04 thesis-claim view"
  artifacts:
    - path: "results/federated/_thesis/main_comparison.md"
      provides: "THS-07 main thesis comparison table — 4 rows, mean+/-std cells, adaptive winner bolded if D-11 satisfied"
      min_lines: 5
    - path: "results/federated/_thesis/main_comparison.csv"
      provides: "THS-07 CSV companion"
      min_lines: 5
    - path: "results/federated/_thesis/ablations.md"
      provides: "THS-05/THS-06 ablation table — 8 rows (1 reference + 7 ablation cells), overall + sparse NDCG/HR"
      min_lines: 10
    - path: "results/federated/_thesis/ablations.csv"
      provides: "THS-05/THS-06 CSV companion"
      min_lines: 10
    - path: "results/federated/_thesis/sparse_slice.md"
      provides: "THS-04 thesis-claim sparse-user slice — every main + ablation row, NDCG@10 sparse"
      min_lines: 5
    - path: "results/federated/_thesis/sparse_slice.csv"
      provides: "THS-04 CSV companion"
      min_lines: 5
  key_links:
    - from: "scripts/thesis/run_thesis_sweep.py main matrix execution"
      to: "results/federated/<module>/<run_id>/results.json (12 files)"
      via: "subprocess.run(scripts/run.py ...) per cell; manifest.thesis_run_label='main' propagated"
      pattern: "thesis_run_label.*main"
    - from: "scripts/thesis/aggregate_results.py post-sweep run"
      to: "results/federated/_thesis/{main_comparison,ablations,sparse_slice}.{md,csv}"
      via: "atomic_write_text per file after collect+aggregate+render"
      pattern: "atomic_write_text"
---

<objective>
Operational runbook + UAT for the actual ~50hr thesis-evaluation matrix execution. This plan does NOT write production code — Plans 01-04 do. This plan documents the human-in-the-loop checkpoints, validates the full pipeline end-to-end on real hardware, and closes Phase 7 with the thesis tables on disk.

**Why a separate plan?** The orchestrator + aggregator are pure code (Plans 03 + 04). The actual experiment runs take ~50 hours of GPU wallclock and produce real metric values that drive thesis claims. Wrapping the run-and-verify cycle in checkpoint:human-verify tasks gives:
1. A pre-flight gate (Task 1: smoke run → confirm 1 cell completes correctly → confirm aggregator's D-20 hard-fail behavior on partial state).
2. The main-matrix execution gate (Task 2: ~19.5 hr; user kicks off + monitors progress).
3. The ablation-matrix execution gate (Task 3: ~31.5 hr; user kicks off + monitors).
4. The aggregation + thesis-claim verification gate (Task 4: aggregator run + visual review of D-11 win/no-win + drift detection).

Per CONTEXT D-12: if adaptive does NOT win the main comparison, the ablation table becomes a "recovery run" looking for ANY adaptive variant that beats baseline+personalized. The runbook documents both the success path AND the negative-result path explicitly.
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/07-thesis-evaluation-run/07-CONTEXT.md
@.planning/phases/07-thesis-evaluation-run/07-RESEARCH.md
@.planning/phases/07-thesis-evaluation-run/07-VALIDATION.md
@.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-01-PLAN.md
@.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-02-PLAN.md
@.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-03-PLAN.md
@.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-04-PLAN.md

<interfaces>
From scripts/thesis/run_thesis_sweep.py (Plan 03):
```bash
# CLI:
python scripts/thesis/run_thesis_sweep.py --phase={main,ablation,all} [--dry-run] [--retry-failed] [--module=<one>] [--seed=<int>] [--results-root=<path>]
```

From scripts/thesis/aggregate_results.py (Plan 04):
```bash
# CLI:
python scripts/thesis/aggregate_results.py [--results-root=<path>] [--output-dir=<path>] [--check-only]
```

Wallclock estimates (CONTEXT D-09 + D-14):
- Single baseline / personalized run at thesis_crossdevice_main: ~1 hr (RTX 5090, 100 rounds, 0.1 fraction-train, dim=64, BPR-MF / Adam)
- Single adaptive run at thesis_crossdevice_main: ~1.5 hr (extra dual-level personalization overhead)
- Single pfedrec run at paper_compat_pfedrec: ~3 hr (fraction-train=1.0; 100% client participation per round)
- Main matrix total: 3*1 + 3*1 + 3*1.5 + 3*3 = 3 + 3 + 4.5 + 9 = ~19.5 hr
- Ablation matrix total: 21 * 1.5 = ~31.5 hr (all adaptive cells)
- Grand total: ~51 hr

Pre-aggregation gate (CONTEXT D-20 + VALIDATION):
- All 33 thesis-tagged manifests on disk before running aggregator
- Verifiable via: `find results/federated -name manifest.json -exec grep -l 'thesis_run_label' {} \; | wc -l` returns 33

Smoke-run command:
```bash
python scripts/thesis/run_thesis_sweep.py --module=adaptive --seed=42 --phase=main
```
Expected: 1 cell runs (~1.5 hr); results.json lands at `results/federated/adaptive/<run_id>/results.json`.

Re-run idempotency check:
```bash
python scripts/thesis/run_thesis_sweep.py --module=adaptive --seed=42 --phase=main
```
Second invocation: prints `[SKIP] cell 1/1: ('adaptive', 'main', 42, 'none', '')` and exits 0 immediately.
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Write the operational runbook + UAT documents</name>
  <read_first>
    - .planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-01-PLAN.md
    - .planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-02-PLAN.md
    - .planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-03-PLAN.md
    - .planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-04-PLAN.md
    - .planning/phases/07-thesis-evaluation-run/07-VALIDATION.md "Manual-Only Verifications"
    - .planning/phases/07-thesis-evaluation-run/07-CONTEXT.md sections D-12 and D-23 (failure handling)
  </read_first>
  <behavior>
    - Two new markdown documents under `.planning/phases/07-thesis-evaluation-run/`:
      1. `07-thesis-evaluation-run-05-RUNBOOK.md` — the step-by-step execution guide for the user.
      2. `07-thesis-evaluation-run-05-UAT.md` — a structured user-acceptance-test checklist that captures pass/fail per gate.
    - The runbook is referenceable during the 50hr execution; the UAT is filled in by the user as gates pass.
    - Both documents reference the existing CLI signatures from Plans 03 + 04 — no new code is needed.
  </behavior>
  <action>
**Step 1 — Create `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-RUNBOOK.md`** with EXACT content:

```markdown
# Phase 7 — Thesis Evaluation Runbook

**Purpose:** Step-by-step guide for executing the thesis-evaluation sweep on real hardware.
**Estimated wallclock:** ~50 hours (19.5 hr main + 31.5 hr ablation).
**Prerequisites:**
- Plans 01..04 are complete and merged. (`pytest scripts/foundation/tests/ -ra` reports all green.)
- The foundation bundle is on disk: `data/derived/foundation_index.json` exists.
- W&B login is active: `wandb login` (or `WANDB_API_KEY` env var) before kicking off any cell.

***

## Gate A — Pre-flight smoke (~1.5 hr)

Goal: confirm a single cell runs end-to-end and produces a thesis-tagged manifest on disk.

### A.1 — Single-cell smoke run
```bash
cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
python scripts/thesis/run_thesis_sweep.py --module=adaptive --seed=42 --phase=main
```
Expected:
- One `[RUN] cell 1/1: ('adaptive', 'main', 42, 'none', '')` line at start.
- ~1.5 hours of flwr-run output, with W&B run name `thesis-main-adaptive-seed42` visible at https://wandb.ai/<your-entity>/federated-cf-cross-device.
- One `[SUMMARY] completed=1 failed=0 skipped=0` line at end; exit code 0.
- One new directory: `results/federated/adaptive/<run_id>/` containing `results.json` + `manifest.json`.

### A.2 — Verify manifest carries thesis fields
```bash
find results/federated/adaptive -name manifest.json -newer .planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-04-PLAN.md \
  | xargs -I{} python -c "import json; m = json.load(open('{}')); print('thesis_run_label =', m.get('thesis_run_label'), '| run_seed =', m.get('run_seed'))"
```
Expected:
- `thesis_run_label = main | run_seed = 42`

### A.3 — Re-run idempotency check
```bash
python scripts/thesis/run_thesis_sweep.py --module=adaptive --seed=42 --phase=main
```
Expected:
- `[SKIP] cell 1/1: ('adaptive', 'main', 42, 'none', '')` — already on disk.
- `[SUMMARY] completed=0 failed=0 skipped=1`; exit code 0.
- No new directory created.

### A.4 — D-20 hard-fail demonstration
```bash
python scripts/thesis/aggregate_results.py --check-only 2>&1 | head -20
```
Expected:
- `[D-20 HARD-FAIL] Missing 32 cells:` followed by sorted list of missing tuples.
- Exit code 1 (proves the safety net works).
- No files written under `results/federated/_thesis/`.

If ALL of A.1..A.4 pass, proceed to Gate B. If any fail, stop here and debug.

***

## Gate B — Main matrix execution (~19.5 hr)

Goal: complete all 12 main-comparison cells.

### B.1 — Kick off main matrix
```bash
nohup python scripts/thesis/run_thesis_sweep.py --phase=main > /tmp/thesis_main.log 2>&1 &
echo "Sweep PID: $!"
```
Expected:
- 11 more cells run (cell 1/12 was already done in Gate A → skipped).
- Each cell takes ~1-3 hr; total wallclock for the remaining 11 cells ≈ ~18 hr.
- W&B dashboard at https://wandb.ai/<your-entity>/federated-cf-cross-device shows runs grouped by `thesis/run_label=main`.

### B.2 — Monitor progress
```bash
# In another terminal:
watch -n 60 'cat results/federated/_thesis/_progress.json 2>/dev/null | python -m json.tool'
```
Expected: `_progress.json` updates after every cell completion with `{"completed": N, "failed": M, "remaining": K, "last_cell": [...], "elapsed_sec": ...}`.

### B.3 — End-of-run cell count
```bash
find results/federated -path '*/manifest.json' -exec grep -l '"thesis_run_label": "main"' {} \; | wc -l
```
Expected: `12` (4 modules × 3 seeds).

### B.4 — Failed-cell handling
If `failed_cells.json` exists at end of B.1:
```bash
cat results/federated/_thesis/failed_cells.json | python -m json.tool
# Inspect stderr_excerpt fields — common causes: CUDA OOM, foundation bundle missing.
python scripts/thesis/run_thesis_sweep.py --retry-failed --phase=main
```
The `--retry-failed` flag re-runs only cells whose results.json is still missing on disk (D-23 + D-31).

If failures persist after one retry, STOP and investigate (do not enter Gate C until the 12 main cells are present on disk).

***

## Gate C — Ablation matrix execution (~31.5 hr)

Goal: complete all 21 ablation cells.

### C.1 — Kick off ablation matrix
```bash
nohup python scripts/thesis/run_thesis_sweep.py --phase=ablation > /tmp/thesis_ablation.log 2>&1 &
echo "Sweep PID: $!"
```
Expected:
- 21 cells (7 ablation knobs × 3 seeds), all `module=adaptive` at `thesis_crossdevice_main`.
- ~1.5 hr per cell × 21 = ~31.5 hr.

### C.2 — Monitor progress
Same as B.2.

### C.3 — End-of-run cell count
```bash
find results/federated/adaptive -path '*/manifest.json' -exec grep -l '"thesis_run_label": "ablation_' {} \; | wc -l
```
Expected: `21`.

### C.4 — Failed-cell handling
Same as B.4.

***

## Gate D — Pre-aggregation gate

Goal: confirm 33 thesis-tagged manifests on disk BEFORE running the aggregator (no partial table emission).

```bash
# Count main + ablation manifests.
MAIN=$(find results/federated -path '*/manifest.json' -exec grep -l '"thesis_run_label": "main"' {} \; | wc -l)
ABL=$(find results/federated -path '*/manifest.json' -exec grep -l '"thesis_run_label": "ablation_' {} \; | wc -l)
TOTAL=$((MAIN + ABL))
echo "main=$MAIN ablation=$ABL total=$TOTAL"
```
Expected: `main=12 ablation=21 total=33`.

If `total < 33`, run `--check-only` to see the explicit missing list, then return to Gate B/C as appropriate.

***

## Gate E — Aggregation + thesis-claim verification

Goal: emit the 6 thesis output files; visually inspect the main comparison + sparse slice.

### E.1 — Run the aggregator
```bash
python scripts/thesis/aggregate_results.py
```
Expected:
- `[INFO] Collected 33 thesis-tagged result records.`
- `[OK] 6 output files written to <repo>/results/federated/_thesis`.
- Exit code 0.

### E.2 — Inspect main_comparison.md
```bash
cat results/federated/_thesis/main_comparison.md
```
Expected:
- 4 rows (baseline, personalized, adaptive, pfedrec).
- Cells in `0.4123 ± 0.0089` format (4 decimal places).
- PFedRec row carries `†` footnote markers; footnote text matches D-08.
- **Outcome A (success — adaptive wins per D-11)**: adaptive's NDCG@10 cells are wrapped in `**bold**` formatting. Sparse NDCG@10 cell is bolded too — primary thesis claim (THS-04) confirmed.
- **Outcome B (failure — adaptive does NOT win)**: NO cells are bolded under D-11. Document the negative result; proceed to E.5 (D-12 contingency).

### E.3 — Inspect sparse_slice.md (THS-04 thesis-claim view)
```bash
cat results/federated/_thesis/sparse_slice.md
```
Expected:
- One row per main module + one row per ablation cell.
- Sparse NDCG@10 winner bolded among comparable main rows.
- `n_seeds_with_sparse=K/3` footnotes ONLY on rows where any seed had zero sparse evaluations (Pitfall 10 — should be rare).

### E.4 — Inspect ablations.md (THS-05/THS-06)
```bash
cat results/federated/_thesis/ablations.md
```
Expected:
- 8 rows: 1 reference (main config) + 7 ablation cells.
- Columns: Cell label, Overall NDCG@10, Overall HR@10, Sparse NDCG@10, Sparse HR@10.
- Medium/dense omitted (D-15) — but available in ablations.csv if needed.

### E.5 — D-12 contingency (only if Outcome B at E.2)
If the main comparison shows adaptive losing or tying baseline/personalized on overall NDCG@10:

1. Inspect `ablations.md` row by row. Look for the cell whose NDCG@10 mean exceeds adaptive-main's NDCG@10 mean by more than σ.
2. Re-render an "augmented main comparison" mentally: replace adaptive-main with the winning ablation cell. Does THAT row beat baseline + personalized under D-11?
3. **Outcome B-1 (recovery success)**: An ablation cell wins. Document in `07-thesis-evaluation-run-05-UAT.md` (Task 1's other output) which knob configuration is the actual thesis claim. Phase 7 closes with the contribution restated around that variant.
4. **Outcome B-2 (recovery failure)**: NO ablation cell beats baseline + personalized under D-11. This is the "thesis contribution must be rethought" path per PROJECT.md core value. Phase 7 closes with the negative result documented; trigger a thesis-level replan via `/gsd:plan-phase` of a new milestone.

### E.6 — PFedRec calibration check
```bash
grep -A1 "PFedRec reproduction drifted" results/federated/_thesis/main_comparison.md
```
Expected (success): no output (drift note absent → reproduction within ±2 points).
Expected (drift): the markdown body contains a "**PFedRec reproduction drifted from IJCAI-23 reference**" line. This is informational only (PFedRec is a calibration reference, not a thesis claim — D-05). Investigate before reporting, but do NOT block Phase 7 closure.

***

## Closing checklist
- [ ] Gate A passed (smoke + idempotency + D-20 demo).
- [ ] Gate B completed (12 main manifests on disk).
- [ ] Gate C completed (21 ablation manifests on disk).
- [ ] Gate D verified (find...wc returns 33).
- [ ] Gate E completed (6 output files; main_comparison.md inspected; thesis claim status determined).
- [ ] UAT document (`07-thesis-evaluation-run-05-UAT.md`) filled in with PASS/FAIL per gate.
- [ ] Phase 7 STATE.md updated with the thesis claim outcome (Outcome A / B-1 / B-2).
```

**Step 2 — Create `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-UAT.md`** with EXACT content:

```markdown
# Phase 7 — User Acceptance Test (UAT)

**Purpose:** Track gate-by-gate pass/fail and the thesis-claim outcome.
**Filled in by:** the user, during/after the ~50hr matrix execution.
**Reference:** `07-thesis-evaluation-run-05-RUNBOOK.md` for the step-by-step procedure.

***

## Gate A — Pre-flight smoke

| Check | Expected | Observed | Status |
|-------|----------|----------|--------|
| A.1 single-cell run completes | exit 0; 1 results.json on disk | | ⬜ |
| A.2 manifest carries thesis fields | `thesis_run_label = main, run_seed = 42` | | ⬜ |
| A.3 re-run reports SKIP | `completed=0 failed=0 skipped=1` | | ⬜ |
| A.4 D-20 hard-fail demo | exit 1; `Missing 32 cells:` visible | | ⬜ |

**Gate A overall:** ⬜ PASS / ⬜ FAIL
**Notes:**

***

## Gate B — Main matrix execution

| Module | Seed=42 | Seed=1337 | Seed=2026 | Failed cells |
|--------|---------|-----------|-----------|--------------|
| baseline      | ⬜ | ⬜ | ⬜ | |
| personalized  | ⬜ | ⬜ | ⬜ | |
| adaptive      | ⬜ | ⬜ | ⬜ | |
| pfedrec       | ⬜ | ⬜ | ⬜ | |

**Total main manifests on disk:** ___ (expected: 12)
**Wallclock elapsed:** ___ hr
**Gate B overall:** ⬜ PASS / ⬜ FAIL
**Notes:**

***

## Gate C — Ablation matrix execution

| Ablation cell | Seed=42 | Seed=1337 | Seed=2026 |
|---------------|---------|-----------|-----------|
| `alpha_method=multi_factor`   | ⬜ | ⬜ | ⬜ |
| `alpha_method=data_quantity`  | ⬜ | ⬜ | ⬜ |
| `per_user_alpha=true`         | ⬜ | ⬜ | ⬜ |
| `item_perturbation=true`      | ⬜ | ⬜ | ⬜ |
| `contrastive_lambda=0.1`      | ⬜ | ⬜ | ⬜ |
| `fusion_type=add`             | ⬜ | ⬜ | ⬜ |
| `fusion_type=gate`            | ⬜ | ⬜ | ⬜ |

**Total ablation manifests on disk:** ___ (expected: 21)
**Wallclock elapsed:** ___ hr
**Gate C overall:** ⬜ PASS / ⬜ FAIL
**Notes:**

***

## Gate D — Pre-aggregation gate

| Check | Expected | Observed | Status |
|-------|----------|----------|--------|
| Main manifests | 12 | | ⬜ |
| Ablation manifests | 21 | | ⬜ |
| Total | 33 | | ⬜ |

**Gate D overall:** ⬜ PASS / ⬜ FAIL

***

## Gate E — Aggregation + thesis-claim verification

### E.1 — Aggregator run
**Exit code:** ___ (expected: 0)
**Files emitted:** ⬜ main_comparison.md ⬜ main_comparison.csv ⬜ ablations.md ⬜ ablations.csv ⬜ sparse_slice.md ⬜ sparse_slice.csv

### E.2 — Main comparison inspection (the headline result)

Paste the table from `main_comparison.md` here:

```
[paste main_comparison.md table contents here]
```

**Adaptive overall NDCG@10:** ___ ± ___
**Personalized overall NDCG@10:** ___ ± ___
**Baseline overall NDCG@10:** ___ ± ___
**PFedRec overall NDCG@10:** ___ ± ___ (calibration; expected: 0.441 ± 0.02)

**THS-03 win check (D-11):**
- Adaptive (mean - std) = ___ — ___ = ___
- Baseline (mean + std) = ___ + ___ = ___
- Personalized (mean + std) = ___ + ___ = ___
- Adaptive lower bound > BOTH baseline and personalized upper bounds? ⬜ YES (THS-03 PASS) / ⬜ NO (THS-03 FAIL)

### E.3 — Sparse-slice inspection (THS-04, the thesis claim's strongest form)

Paste the table from `sparse_slice.md` here:

```
[paste sparse_slice.md table contents here]
```

**THS-04 win check (D-11 on sparse NDCG@10):**
- Adaptive sparse (mean - std) = ___
- Baseline sparse (mean + std) = ___
- Personalized sparse (mean + std) = ___
- Adaptive lower bound > BOTH? ⬜ YES (THS-04 PASS) / ⬜ NO (THS-04 FAIL)

### E.4 — Ablation inspection

**Best-performing ablation cell on overall NDCG@10:** ___ (mean ± std)
**Best-performing ablation cell on sparse NDCG@10:** ___ (mean ± std)
**Does any ablation cell beat both baseline and personalized under D-11?** ⬜ YES / ⬜ NO

### E.5 — D-12 contingency (only if THS-03 or THS-04 FAILED)

⬜ N/A (Outcome A — main comparison won)
⬜ Outcome B-1 (recovery): cell `___` wins under D-11; restate thesis claim around this variant.
⬜ Outcome B-2 (failure): no cell wins under D-11; trigger thesis-level replan per PROJECT.md core value.

### E.6 — PFedRec calibration

**PFedRec mean HR@10:** ___ (target: 0.729 ± 0.02; range: 0.709 .. 0.749)
**PFedRec mean NDCG@10:** ___ (target: 0.441 ± 0.02; range: 0.421 .. 0.461)
**Drift note in markdown?** ⬜ YES (investigate; non-blocking) / ⬜ NO (within tolerance — PFR-08 reproduces)

***

## Final closure

| Requirement | Phase 7 status |
|-------------|----------------|
| THS-01 standardized config defined | ⬜ |
| THS-02 multi-seed comparison table emitted | ⬜ |
| THS-03 adaptive wins overall NDCG@10 | ⬜ PASS / ⬜ FAIL / ⬜ Recovery via ablation |
| THS-04 adaptive wins sparse NDCG@10 | ⬜ PASS / ⬜ FAIL / ⬜ Recovery via ablation |
| THS-05 ablations executed | ⬜ |
| THS-06 ablations report per-group metrics | ⬜ |
| THS-07 thesis tables exported to `_thesis/` | ⬜ |

**Overall Phase 7 status:** ⬜ COMPLETE (Outcome A) / ⬜ COMPLETE-WITH-CAVEAT (Outcome B-1) / ⬜ FAILED (Outcome B-2)

**Date completed:** ___
**STATE.md updated?** ⬜ YES
```
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system && test -f .planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-RUNBOOK.md && test -f .planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-UAT.md && grep -q "Gate A — Pre-flight smoke" .planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-RUNBOOK.md && grep -q "Gate E — Aggregation" .planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-RUNBOOK.md && grep -q "THS-04 win check" .planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-UAT.md && echo "Runbook + UAT documents created"</automated>
  </verify>
  <done>
    - `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-RUNBOOK.md` exists, references Gates A through E.
    - `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-UAT.md` exists, contains the gate-by-gate checklist with paste-target tables.
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 2: Pre-flight smoke run + D-20 demo (Gate A) — ~1.5 hr</name>
  <what-built>
    Plans 01-04 produced the orchestrator + aggregator + foundation extensions. Before committing to ~50hr of GPU time, the runbook's Gate A validates that:
    1. A single thesis cell runs end-to-end and produces a thesis-tagged manifest.
    2. Re-running the same cell hits the skip-on-existing path.
    3. The aggregator's D-20 hard-fail correctly reports `Missing 32 cells` against the partial state.
  </what-built>
  <how-to-verify>
    Follow `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-RUNBOOK.md` Gate A, sub-steps A.1 through A.4:

    1. Run: `python scripts/thesis/run_thesis_sweep.py --module=adaptive --seed=42 --phase=main`
       - Expected: ~1.5 hr execution; exit code 0; one new `results/federated/adaptive/<run_id>/` directory.
    2. Verify manifest: `find results/federated/adaptive -name manifest.json -newer .planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-04-PLAN.md | xargs -I{} python -c "import json; m = json.load(open('{}')); print('thesis_run_label =', m.get('thesis_run_label'), '| run_seed =', m.get('run_seed'))"`
       - Expected: `thesis_run_label = main | run_seed = 42`.
    3. Re-run idempotency: same command; expect `[SKIP]` line and `completed=0 failed=0 skipped=1`.
    4. D-20 demo: `python scripts/thesis/aggregate_results.py --check-only` — expect exit 1 + `Missing 32 cells:` visible in stderr.

    Fill in the Gate A section of `07-thesis-evaluation-run-05-UAT.md` with PASS/FAIL per row. If any row fails, debug before proceeding to Task 3.
  </how-to-verify>
  <resume-signal>Type "Gate A passed" with the run_id of the smoke run, or describe the failure and which sub-step did not pass.</resume-signal>
  <action>See &lt;how-to-verify&gt; above. Checkpoint task — operator follows the runbook steps; no automated action.</action>
  <verify>
    <automated>echo "checkpoint:human-verify task — verification is human-in-the-loop per resume-signal"; true</automated>
  </verify>
  <done>Operator types the resume-signal phrase confirming the gate passed.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 3: Main matrix execution (Gate B) — ~19.5 hr</name>
  <what-built>
    With Gate A passed, the runbook's Gate B kicks off all 12 main-comparison cells. The orchestrator's idempotent skip-on-existing means cell 1/12 (the smoke run from Gate A) is skipped, leaving 11 cells × ~1.7 hr average ≈ ~19 hr of actual execution.
  </what-built>
  <how-to-verify>
    Follow `07-thesis-evaluation-run-05-RUNBOOK.md` Gate B, sub-steps B.1 through B.4:

    1. Kick off: `nohup python scripts/thesis/run_thesis_sweep.py --phase=main > /tmp/thesis_main.log 2>&1 &`
    2. Monitor `_progress.json` over the next ~19 hours.
    3. Final cell count: `find results/federated -path '*/manifest.json' -exec grep -l '"thesis_run_label": "main"' {} \; | wc -l` returns `12`.
    4. If failures: inspect `failed_cells.json`, identify the cause (most common: CUDA OOM, foundation bundle missing). Run `python scripts/thesis/run_thesis_sweep.py --retry-failed --phase=main`. Repeat at most once. If failures persist past one retry, STOP and investigate before Gate C.

    Fill in the Gate B section of `07-thesis-evaluation-run-05-UAT.md` with one checkmark per (module, seed) tuple.
  </how-to-verify>
  <resume-signal>Type "Gate B passed" with the final main-cell count (12), or describe any cells that failed and why.</resume-signal>
  <action>See &lt;how-to-verify&gt; above. Checkpoint task — operator follows the runbook steps; no automated action.</action>
  <verify>
    <automated>echo "checkpoint:human-verify task — verification is human-in-the-loop per resume-signal"; true</automated>
  </verify>
  <done>Operator types the resume-signal phrase confirming the gate passed.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 4: Ablation matrix execution (Gate C) — ~31.5 hr</name>
  <what-built>
    With Gate B passed (12 main cells on disk), the runbook's Gate C kicks off all 21 ablation cells. All 21 are `module=adaptive` at `thesis_crossdevice_main` per D-13; each runs ~1.5 hr; total ~31.5 hr.
  </what-built>
  <how-to-verify>
    Follow `07-thesis-evaluation-run-05-RUNBOOK.md` Gate C, sub-steps C.1 through C.4:

    1. Kick off: `nohup python scripts/thesis/run_thesis_sweep.py --phase=ablation > /tmp/thesis_ablation.log 2>&1 &`
    2. Monitor `_progress.json` over the next ~31.5 hours.
    3. Final cell count: `find results/federated/adaptive -path '*/manifest.json' -exec grep -l '"thesis_run_label": "ablation_' {} \; | wc -l` returns `21`.
    4. Failure handling: same as Gate B (retry once, stop if persistent).

    Fill in the Gate C section of `07-thesis-evaluation-run-05-UAT.md` with one checkmark per (ablation_cell, seed).
  </how-to-verify>
  <resume-signal>Type "Gate C passed" with the final ablation-cell count (21), or describe any cells that failed and why.</resume-signal>
  <action>See &lt;how-to-verify&gt; above. Checkpoint task — operator follows the runbook steps; no automated action.</action>
  <verify>
    <automated>echo "checkpoint:human-verify task — verification is human-in-the-loop per resume-signal"; true</automated>
  </verify>
  <done>Operator types the resume-signal phrase confirming the gate passed.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 5: Aggregation + thesis-claim verification (Gates D + E)</name>
  <what-built>
    With 33 thesis-tagged manifests on disk (12 main + 21 ablation), the aggregator emits the 6 thesis output files under `results/federated/_thesis/`. Visual inspection of `main_comparison.md` and `sparse_slice.md` determines the THS-03 / THS-04 thesis-claim outcome.
  </what-built>
  <how-to-verify>
    Follow `07-thesis-evaluation-run-05-RUNBOOK.md` Gate D + Gate E:

    1. **Gate D — pre-aggregation count**: `MAIN=$(find results/federated -path '*/manifest.json' -exec grep -l '"thesis_run_label": "main"' {} \; | wc -l); ABL=$(find results/federated -path '*/manifest.json' -exec grep -l '"thesis_run_label": "ablation_' {} \; | wc -l); echo "main=$MAIN abl=$ABL"` — expect `main=12 abl=21`.
    2. **Gate E.1 — aggregator run**: `python scripts/thesis/aggregate_results.py` — expect exit 0 + 6 files written.
    3. **Gate E.2 — main_comparison inspection**: `cat results/federated/_thesis/main_comparison.md`. Determine outcome:
       - **Outcome A**: adaptive's NDCG@10 cells are `**bolded**` per D-11 → THS-03 PASS.
       - **Outcome B**: no bolded NDCG@10 cells → THS-03 FAIL → proceed to E.5.
    4. **Gate E.3 — sparse_slice inspection**: `cat results/federated/_thesis/sparse_slice.md` and check whether adaptive's sparse NDCG@10 row is bolded → THS-04 PASS or FAIL.
    5. **Gate E.4 — ablations inspection**: `cat results/federated/_thesis/ablations.md`. Identify the best ablation cell on (a) overall NDCG@10 and (b) sparse NDCG@10.
    6. **Gate E.5 — D-12 contingency** (only if THS-03 or THS-04 FAILED): mentally substitute the best ablation cell for adaptive-main; does THAT row beat baseline + personalized under D-11? If yes (Outcome B-1), restate thesis claim around the winning ablation cell. If no (Outcome B-2), trigger thesis-level replan per PROJECT.md.
    7. **Gate E.6 — PFedRec calibration**: `grep -A1 "PFedRec reproduction drifted" results/federated/_thesis/main_comparison.md`. Empty output = PFR-08 reproduces within ±2 points.

    Fill in Gate D + Gate E sections of `07-thesis-evaluation-run-05-UAT.md`. Update STATE.md with the final thesis-claim outcome (A / B-1 / B-2).
  </how-to-verify>
  <resume-signal>Type "Phase 7 complete (Outcome A)" with the bolded-cell observations, OR "Phase 7 complete-with-caveat (Outcome B-1)" identifying the winning ablation cell, OR "Phase 7 failed (Outcome B-2)" indicating thesis-level replan is needed.</resume-signal>
  <action>See &lt;how-to-verify&gt; above. Checkpoint task — operator follows the runbook steps; no automated action.</action>
  <verify>
    <automated>echo "checkpoint:human-verify task — verification is human-in-the-loop per resume-signal"; true</automated>
  </verify>
  <done>Operator types the resume-signal phrase confirming the gate passed.</done>
</task>

</tasks>

<verification>
- Gate A demonstrates the full pipeline on a single cell + idempotency + D-20 safety net.
- Gates B + C produce all 33 thesis-tagged manifests on disk.
- Gate D verifies pre-aggregation count.
- Gate E emits the 6 output files and triggers the thesis-claim outcome determination.
- The UAT document captures pass/fail per gate; STATE.md captures the final outcome.
</verification>

<success_criteria>
- [ ] `07-thesis-evaluation-run-05-RUNBOOK.md` exists with Gates A-E documented.
- [ ] `07-thesis-evaluation-run-05-UAT.md` exists with the gate-by-gate checklist.
- [ ] Gate A passes (smoke + idempotency + D-20).
- [ ] Gate B passes (12 main manifests).
- [ ] Gate C passes (21 ablation manifests).
- [ ] Gate D passes (33 total).
- [ ] Gate E completes (6 output files + thesis-claim outcome determined).
- [ ] STATE.md updated with the Phase 7 outcome.
- [ ] All 7 THS-01..THS-07 requirements have a documented status in the UAT.
</success_criteria>

<output>
After completion, create `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-SUMMARY.md` documenting:
- Per-gate observed wallclock vs estimated.
- Per-gate failure count and recovery actions taken.
- The final main_comparison.md table contents (paste-block).
- The final sparse_slice.md table contents.
- The thesis-claim outcome (A / B-1 / B-2) with rationale.
- PFedRec calibration drift observation.
- A pointer to `07-thesis-evaluation-run-05-UAT.md` for the full audit trail.
- STATE.md update status.
</output>
</content>
</invoke>
