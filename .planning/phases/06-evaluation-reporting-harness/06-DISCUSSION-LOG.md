# Phase 6: Evaluation & Reporting Harness - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-29
**Phase:** 06-evaluation-reporting-harness
**Areas discussed:** Results path schema, W&B project naming, Best-round final-eval semantics, Per-user-group reporting depth
**Folded todos:** phase2-baseline-determinism-path-bug.md

---

## Area Selection (entry gate)

| Option | Description | Selected |
|--------|-------------|----------|
| Results path schema | EVL-04: `results/federated/<module>/<run_id>/` layout, repo-root anchoring, filename schema, legacy artifact handling. Resolves Phase 2 path-bug todo. | ✓ |
| W&B project naming | EVL-05: keep current default vs rename to `thesis-crossdevice-*`. Per-mode routing? | ✓ |
| Best-round final-eval semantics | EVL-01 + EVL-06: re-run eval after restore vs trust snapshot; `best_*` only or both `best_*`+`last_*`; filename encoding. | ✓ |
| Per-group reporting depth | EVL-02 + EVL-03: pfedrec parity audit, sampling-exposure history vs final-only, per-group breakdowns of extra fields. | ✓ |

**User selected all four areas.**

| Option | Description | Selected |
|--------|-------------|----------|
| Fold into Phase 6 | Treat the baseline path-bug todo as part of EVL-04 cleanup — same root cause, same fix. | ✓ |
| Keep deferred | Leave the todo pending; address separately later. | |

---

## Results path schema (EVL-04)

### Q1: How should result artifacts be organized on disk?

| Option | Description | Selected |
|--------|-------------|----------|
| Per-run directory | `results/federated/<module>/<run_id>/` containing results.json + manifest.json. One folder per run; matches REQUIREMENTS.md EVL-04 wording exactly; scales to ablation sweeps. | ✓ |
| Flat with run_id in filename | `results/federated/<module>/<run_id>_results.json` + `<run_id>_manifest.json`. Simpler today (matches current pattern) but clutters fast at 100+ runs. | |
| Flat with date prefix | `results/federated/<module>/<YYYYMMDD>/<run_id>_results.json`. Date-based archival. Heavier than EVL-04 wording demands. | |

**User's choice:** Per-run directory.

### Q2: Where should the results root live?

| Option | Description | Selected |
|--------|-------------|----------|
| Repo-root anchored | `<repo>/results/federated/<module>/<run_id>/` resolved via foundation helper. Fixes Phase 2 path-bug todo; survives running flwr from any CWD. | ✓ |
| Env-var overridable | Default = repo-root anchored, but honor `$FEDREC_RESULTS_DIR` if set. Useful for sweeps writing to /scratch/. Maps cleanly to existing FEDREC_CACHE_ROOT pattern. | |
| Module-relative | Keep current `../results/federated/...`. Backwards-compat for old scripts but does NOT fix the Phase 2 path-bug todo. | |

**User's choice:** Repo-root anchored.

### Q3: What should happen to existing cross-silo / pre-Phase-6 result artifacts?

| Option | Description | Selected |
|--------|-------------|----------|
| Leave untouched | Don't move anything. New Phase-6 runs go to the new layout; old cross-silo runs remain reproducible. Matches PROJECT.md "cross-silo configs must continue to run" constraint. | ✓ |
| Archive then re-layout | Move existing flat files into `results/federated/_legacy/`. Cleaner long-term but adds migration risk. | |
| Delete legacy | Wipe pre-Phase-6 artifacts. Loses appendix reproducibility. | |

**User's choice:** Leave untouched.

### Q4: Filename schema inside the per-run directory?

| Option | Description | Selected |
|--------|-------------|----------|
| Clean filenames | `results.json` + `manifest.json`. Directory IS the run identifier; no redundant prefix. | ✓ |
| Run-id prefixed | `<run_id>_results.json` + `<run_id>_manifest.json`. Redundant with directory name but survives accidental flatten/copy. | |
| Encode best_round | `<run_id>_best_round_<N>_results.json` per REQUIREMENTS.md EVL-06 hint — but information is duplicated inside manifest.json which already carries best_round. | |

**User's choice:** Clean filenames.

---

## W&B project naming (EVL-05)

### Q1: What W&B project should cross-device runs log to?

| Option | Description | Selected |
|--------|-------------|----------|
| Keep current default | Continue with `federated-cf-cross-device` for all 4 modules. Already wired; zero churn. | ✓ |
| Rename to thesis-crossdevice | Change to `thesis-crossdevice` (or `thesis-crossdevice-<entity>`) per REQUIREMENTS.md hint. Tighter naming but means existing cross-device runs sit under the old project. | |
| Per-mode split | paper_compat_pfedrec → `thesis-crossdevice-pfedrec-repro`, benchmark_cross_device → `thesis-crossdevice-benchmark`. Dashboard separation but more wiring. | |

**User's choice:** Keep current default.

---

## Best-round final-eval semantics (EVL-01, EVL-06)

### Q1: After restoring best-round arrays, should we re-run one final evaluation pass?

| Option | Description | Selected |
|--------|-------------|----------|
| Re-run final eval | Broadcast one extra `@app.evaluate` round and emit those numbers as `best_*`. Cleanest semantics — canonical artifact reflects exactly the restored state. Costs one extra round (~10–60s on 6040 clients). | ✓ |
| Trust snapshot metrics | Cache `best_metric` + per-group sufficient stats AT the moment best_arrays is snapshotted; emit those as canonical without an extra eval pass. Faster but couples result correctness to in-loop sufficient-stat capture. | |

**User's choice:** Re-run final eval.

### Q2: Should the canonical result artifact also carry `last_*` fields, or only `best_*`?

| Option | Description | Selected |
|--------|-------------|----------|
| Keep both | `best_*` is canonical; `last_*` preserved as a diagnostic block. Matches REQUIREMENTS.md EVL-06 wording. Useful for spotting overfitting/late-round drift. | ✓ |
| best_* only | Drop `last_*` from canonical artifact; reduce ambiguity. W&B per-round history still has the full curve. | |

**User's choice:** Keep both.

---

## Per-user-group reporting depth (EVL-02, EVL-03)

### Q1: How deep should per-user-group reporting go?

| Option | Description | Selected |
|--------|-------------|----------|
| HR@10 + NDCG@10 per group | Just the two ranking metrics per (sparse/medium/dense) — EVL-02 minimum. Already plumbed in 4/4 strategy.py. | ✓ |
| Add per-group eval_loss | Also emit `eval_loss/sparse|medium|dense`. Useful for diagnosing why sparse users underperform. Small extra plumbing in client wire. | |
| Add adaptive-only alpha breakdown | Adaptive module also emits `alpha_mean/sparse|medium|dense` and clip_hit_rate per group. Helps thesis answer "where does the adaptive win come from" directly from artifacts. | |

**User's choice:** HR@10 + NDCG@10 per group.

### Q2: How often should sampling-exposure counts be reported?

| Option | Description | Selected |
|--------|-------------|----------|
| Per-round + final | Emit `evaluated_users_{overall,sparse,medium,dense}` every round AND in canonical block. Lets reader plot exposure-vs-round curves to spot sparse-group starvation under low C. | ✓ |
| Final-only | Only report exposure counts for the best-round / canonical evaluation. Simpler artifact but loses history. | |

**User's choice:** Per-round + final.

---

## Claude's Discretion

User explicitly deferred to Claude on these implementation details (captured in CONTEXT.md `<decisions>` "Claude's Discretion" subsection):

- Foundation helper naming for repo-root resolution.
- Internal wiring of the "extra final eval round" inside each module's server_app.py.
- Manifest schema evolution (new fields like `last_round_metrics`, `final_eval_round_index`); manifest schema-version field bump if needed.

## Deferred Ideas

Items raised during discussion but explicitly out of scope for Phase 6 (captured in CONTEXT.md `<deferred>` section):

- Per-group eval_loss / per-group alpha breakdown — belongs in Phase 7 ablations.
- W&B project rename to `thesis-crossdevice-*` — could revisit at thesis-paper time.
- Per-mode W&B project split — single project + run tags is sufficient for now.
- Migrating legacy flat result files into `_legacy/` subtree — coexistence is acceptable.
- Encoding `best_round` in filename — rejected in favor of clean filenames + manifest-internal field.
