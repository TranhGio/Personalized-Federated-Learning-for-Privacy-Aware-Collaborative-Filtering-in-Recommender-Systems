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
