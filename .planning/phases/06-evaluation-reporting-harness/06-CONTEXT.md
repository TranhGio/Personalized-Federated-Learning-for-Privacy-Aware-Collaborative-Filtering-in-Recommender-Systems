# Phase 6: Evaluation & Reporting Harness - Context

**Gathered:** 2026-04-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Unify the result-emission contract across all four federated modules so every cross-device run produces a canonical artifact reporting best-round metrics from a restored best-round checkpoint, per-user-group (sparse/medium/dense) HR@10 and NDCG@10 as first-class fields, per-round sampling-exposure counts, and writes to a standardized repo-root-anchored path with the FND-07 protocol fingerprint manifest. All cross-device W&B runs log to a dedicated project separate from the existing cross-silo project.

**Server-side / strategy-side / manifest-side only.** Client_app changes already shipped in Phases 2–5. New ML behavior, new metrics, or new client_app plumbing belong in other phases.

</domain>

<decisions>
## Implementation Decisions

### Results path schema (EVL-04)
- **D-01:** Per-run directory layout: `results/federated/<module>/<run_id>/` containing `results.json` + `manifest.json` (and optional sidecars like `alpha_diagnostics.json` for adaptive). One directory per run; the directory IS the run identifier.
- **D-02:** Results root is **repo-root anchored**: `<repo>/results/federated/...` resolved via the foundation package (walk-up from `scripts/foundation/` or equivalent helper). Server_app must NOT use module-relative `../results/federated/`. Resolves the folded `phase2-baseline-determinism-path-bug.md` todo — baseline test path assertion will pass after this lands.
- **D-03:** Existing pre-Phase-6 result artifacts under `results/federated/` are **left untouched**. Cross-silo reproducibility from PROJECT.md constraint is preserved. New Phase-6 runs go to the new `<module>/<run_id>/` layout; legacy flat files coexist.
- **D-04:** Clean filenames inside the per-run directory: `results.json` + `manifest.json` (no run_id prefix, no best_round_<N> infix). best_round and run_id live INSIDE the manifest, not in the filename.

### W&B project routing (EVL-05)
- **D-05:** Keep current default project `federated-cf-cross-device` for all four modules in cross-device modes. Already wired in every `server_app.py`; zero churn. Existing `wandb-project` config override surface preserved (modules read it before falling back to default). Mode-conditional routing (paper_compat_pfedrec → same project) stays as-is.

### Best-round restore semantics (EVL-01, EVL-06)
- **D-06:** After restoring best-round arrays (and best_prototype for adaptive), **broadcast one extra `@app.evaluate` round** and emit those numbers as the canonical `best_*` block. Canonical artifact reflects exactly the restored state, not stale in-loop sufficient-stats from when best_metric was first hit. Costs one extra evaluation round per run (~10–60s on 6040 clients depending on C) — acceptable for correctness.
- **D-07:** Both `best_*` and `last_*` blocks live in the canonical artifact. `best_*` is the canonical reported metric (Phase 1 decision, carried forward). `last_*` is preserved as a diagnostic field for spotting overfitting / late-round drift. W&B per-round history continues to carry the full curve under `round_metrics_history`.

### Per-user-group reporting (EVL-02, EVL-03)
- **D-08:** Per-group fields are **HR@10 and NDCG@10 only**, for `sparse` (0–30 interactions), `medium` (30–100), `dense` (100+). No per-group `eval_loss`, no per-group `alpha_*` (deferred to a future ablation phase or as adaptive-internal logging). Already plumbed in all four `strategy.py` files via summed sufficient stats — Phase 6 just standardizes the artifact emission path.
- **D-09:** Sampling-exposure counts (`evaluated_users_{overall,sparse,medium,dense}`) reported **per-round AND in the canonical block**. Per-round history goes into W&B `round_metrics_history` and the result JSON's `round_metrics` array. Final block carries the cumulative exposure used for the canonical `best_*` evaluation.

### Folded Todos
- **`phase2-baseline-determinism-path-bug.md`** (medium priority, surfaced 2026-04-20 from Phase 3 regression gate) — Resolved by D-02. The baseline test `test_selected_partitions_byte_identical_across_subprocess_reruns` asserts `repo_root/results/federated/`; current baseline `server_app.py` writes to `<repo_root>/../results/federated/`. After D-02 ships (foundation-helper repo-root resolution applied to all four `server_app.py` files), the test path expectation aligns with the write path. Phase 6 planning should include a task to verify the @pytest.mark.slow gate actually exercises after the path fix.

### Claude's Discretion
- Exact name of the foundation helper that resolves repo root (e.g., `fedrec_foundation.paths.repo_root_results_dir()` vs reusing existing `_REPO_ROOT` constant). Planner picks naming to match Phase 1 conventions.
- Internal wiring of the "extra final eval round" in each module's `server_app.py` main loop (after the main FL loop exits, before W&B summary write, before manifest double-write). Standard pattern preferred but per-module adaptation acceptable if the module has structural differences (e.g., pfedrec's PFR-08 auto-verify hook ordering).
- Schema of `manifest.json` evolution (adding new fields like `last_round_metrics`, `final_eval_round_index`) — must remain backward-readable but Phase 6 may bump a manifest schema version field.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` §EVL-01..EVL-06 — Six requirements this phase delivers.
- `.planning/ROADMAP.md` §"Phase 6: Evaluation & Reporting Harness" — Goal, dependencies (Phases 2–5), success criteria.
- `.planning/PROJECT.md` §Active "Reproduction" + "Thesis evaluation" — How EVL outputs feed Phase 7's thesis comparison.

### Foundation contract (consumed, not modified)
- `scripts/foundation/fedrec_foundation/manifest.py` — `RunManifest`, `embed_manifest_in_result`, `write_manifest_sibling` (D-15 double-write helpers; Phase 6 extends without breaking schema).
- `scripts/foundation/fedrec_foundation/mode.py` — Mode resolver. Phase 6 reads `ModeProfile.checkpoint_rule` for restore semantics.
- `scripts/foundation/fedrec_foundation/evaluator.py` — Primary evaluator selector (FND-04). Per-group sufficient stats already flow through this.

### Existing per-module emission code (to be unified)
- `federated-baseline-cf/federated_baseline_cf/server_app.py` lines 786–800 — `results_dir = Path("../results/federated")` (current bug surface; D-02 target).
- `federated-baseline-cf/federated_baseline_cf/strategy.py` lines 49–102 — Per-group sufficient-stat summation pattern (already correct; reused in Phases 3/4/5).
- `federated-personalized-cf/federated_personalized_cf/server_app.py` lines 263–400, 895–910 — W&B init + results write.
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` lines 440–510, 1183–1200 — W&B init + results write + alpha diagnostics.
- `federated-pfedrec/federated_pfedrec/server_app.py` lines 476–520, 1073–1090 — W&B init + results write + PFR-08 auto-verify hook (D-14, must remain functional after Phase 6 path changes).

### Closed-out phase summaries (for prior decisions)
- `.planning/phases/02-baseline-migration/02-baseline-migration-04-SUMMARY.md` — D-27 best-round restore pattern (in-memory snapshot, server-side).
- `.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-05-SUMMARY.md` — D-05/D-06/D-07 best_prototype snapshot/embed/restore (extends D-27 with strategy-state restore).
- `.planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-04-SUMMARY.md` — D-13/D-27 carry-forward + D-14 PFR-08 hook ordering.

### Folded-todo file
- `.planning/todos/pending/phase2-baseline-determinism-path-bug.md` — Folded into Phase 6 scope; will be removed when this CONTEXT.md commits.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Per-group sufficient-stat aggregator pattern** in `strategy.py` (all 4 modules): `_sum_sufficient_stats` + `_sufficient_stats_to_thesis_metrics` — already emits `sampled_hr@10/sparse|medium|dense` and `evaluated_users_*`. Phase 6 leverages this; no new aggregation logic needed.
- **D-15 manifest double-write** (`embed_manifest_in_result` + `write_manifest_sibling` in `scripts/foundation/`): atomic dual-surface manifest emission. Phase 6 keeps this pattern; only the path it writes to changes.
- **D-27 in-memory best-round snapshot** (`best_arrays = ArrayRecord(...)` inside `current_ndcg > best_metric` branch in baseline/personalized/adaptive `server_app.py`): per-module pattern Phase 6 builds on by adding the post-loop "one extra eval round" after the existing arrays = best_arrays restore.
- **W&B project resolver** (every `server_app.py` lines ~285–310): `default_project = "federated-cf-cross-device"` if mode in cross-device modes else module-historical project; honors `wandb-project` config override. D-05 keeps this verbatim.

### Established Patterns
- **Foundation walk-up for repo-root resolution**: `scripts/foundation/` is the canonical anchor for cross-module path resolution (used by `data/derived/`, `_REPO_ROOT` constants in foundation helpers). D-02 reuses this anchor — no new walk-up logic needed.
- **Per-module strategy.py owns metric aggregation; server_app.py owns I/O & manifest emission**: Phase 6 changes are concentrated in `server_app.py` (path, extra eval round, last_*/best_* artifact structure) with zero changes to `strategy.py` aggregation logic.
- **Mode-conditional behavior** via `ModeProfile` from `scripts/foundation/fedrec_foundation/mode.py`: Phase 6 reads `checkpoint_rule` (e.g., `"best_round_restore"`) to gate D-06's extra-eval-round logic. Modes that don't restore (e.g., debug modes) get last_* only.

### Integration Points
- **Phase 7 (Thesis Evaluation Run)** consumes the standardized artifacts produced by Phase 6: THS-02..THS-07 read `best_*` from `results/federated/<module>/<run_id>/results.json` per-seed and aggregate. Phase 6 must finalize the schema before Phase 7 sweeps begin.
- **Per-module subprocess determinism guards** (`scripts/foundation/tests/test_*_subprocess_determinism.py`, all 4 modules): These tests probe `results/federated/...` for byte-identity. After D-02 lands, the path probes update to the new layout; the byte-identity invariant remains.

</code_context>

<specifics>
## Specific Ideas

- "Repo-root anchored, fix Phase 2 path bug at the same time" — user explicitly chose to fold the existing todo rather than defer it. The fix is a one-call helper applied to four `server_app.py` files; tests follow.
- "Re-run final eval, even if it costs an extra round" — user prioritized correctness semantics over latency. The canonical `best_*` block in the artifact MUST come from a fresh evaluation under the restored state, not from cached per-round sufficient stats.
- "Keep `last_*` as a diagnostic" — explicit from REQUIREMENTS.md EVL-06; user reaffirmed.
- "No per-group eval_loss, no per-group alpha breakdown" — user kept Phase 6 narrow. Adaptive thesis claims live in Phase 7 ablations; Phase 6 only delivers the harness EVL-02/03 require.

</specifics>

<deferred>
## Deferred Ideas

- **Per-group eval_loss / per-group alpha breakdown** — Useful for thesis "where does adaptive's win come from" analysis but explicitly out of scope for Phase 6's harness work. Belongs in Phase 7 (Thesis Evaluation Run) ablations or as adaptive-module-internal logging.
- **W&B project rename to `thesis-crossdevice-*`** — REQUIREMENTS.md hints at this naming; user chose to keep `federated-cf-cross-device` for zero churn. Could revisit at thesis-paper time if reviewer feedback demands it.
- **Per-mode W&B project split** (e.g., paper_compat_pfedrec → its own project) — Considered, deferred. Single project + run tags is sufficient for now.
- **Migrating legacy flat result files into `_legacy/` subtree** — Deferred. Cross-silo reproducibility constraint says don't touch them; coexistence with new layout is acceptable.
- **Encoding `best_round` in filename** — Considered (REQUIREMENTS.md EVL-06 hint), rejected by D-04 in favor of clean filenames + manifest-internal field. Could be reversed if downstream tooling needs filename-only signal.

</deferred>

---

*Phase: 06-evaluation-reporting-harness*
*Context gathered: 2026-04-29*
