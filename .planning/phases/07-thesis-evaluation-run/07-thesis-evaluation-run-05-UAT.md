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
