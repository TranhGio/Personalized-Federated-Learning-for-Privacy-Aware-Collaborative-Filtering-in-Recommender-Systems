---
phase: 7
slug: thesis-evaluation-run
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-29
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7+ (already installed via `scripts/foundation/[project.optional-dependencies] dev`) |
| **Config file** | `scripts/foundation/pyproject.toml` `[tool.pytest.ini_options]` (`testpaths=["tests"]`, `addopts="-ra"`) |
| **Quick run command** | `pytest scripts/foundation/tests/test_thesis_orchestrator.py scripts/foundation/tests/test_thesis_aggregator.py scripts/foundation/tests/test_mode.py scripts/foundation/tests/test_manifest.py -x -v` |
| **Full suite command** | `cd scripts/foundation && pytest -ra` (runs all 100+ existing foundation tests + new thesis tests) |
| **Estimated runtime** | ~30 seconds (quick) / ~3 minutes (full foundation suite) |

---

## Sampling Rate

- **After every task commit:** Run `pytest scripts/foundation/tests/test_thesis_orchestrator.py scripts/foundation/tests/test_thesis_aggregator.py -x -v` (~30s)
- **After every plan wave:** Run `cd scripts/foundation && pytest -ra` plus per-module `pytest federated-<module>-cf/tests/ -ra`
- **Before `/gsd:verify-work`:** Full suite must be green AND a 1-cell smoke run + skip-on-rerun + aggregator hard-fail demo must succeed BEFORE running the full ~50hr matrix
- **Pre-aggregation gate:** All 33 thesis-tagged manifests on disk (verifiable via `find results/federated -name manifest.json -exec grep -l 'thesis_run_label' {} \; | wc -l → 33`) before running aggregator
- **Max feedback latency:** ~30 seconds (quick) — well under the 60s Nyquist target

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 7-01-01 | 01 | 1 | THS-01 / mode | unit | `pytest scripts/foundation/tests/test_mode.py::test_thesis_crossdevice_main_profile -v` | ❌ W0 | ⬜ pending |
| 7-01-02 | 01 | 1 | THS-01 / launcher | smoke | `pytest scripts/foundation/tests/test_launcher.py::test_thesis_mode_dry_run -v` | ❌ W0 | ⬜ pending |
| 7-01-03 | 01 | 1 | THS-22 / manifest schema | unit | `pytest scripts/foundation/tests/test_manifest.py::test_run_manifest_schema_version_3 -v` | ❌ W0 | ⬜ pending |
| 7-01-04 | 01 | 1 | THS-22 / manifest backcompat | unit | `pytest scripts/foundation/tests/test_manifest.py::test_run_manifest_backward_compat_v2 -v` | ❌ W0 | ⬜ pending |
| 7-01-05 | 01 | 1 | THS-07 / atomic_write_text | unit | `pytest scripts/foundation/tests/test_atomic.py::test_atomic_write_text -v` | ❌ W0 | ⬜ pending |
| 7-02-01 | 02 | 2 | THS-22 / server_app baseline | integration | `pytest federated-baseline-cf/tests/test_server_integration.py::test_thesis_label_in_manifest -v` | ❌ W0 | ⬜ pending |
| 7-02-02 | 02 | 2 | THS-22 / server_app personalized | integration | `pytest federated-personalized-cf/tests/test_server_integration.py::test_thesis_label_in_manifest -v` | ❌ W0 | ⬜ pending |
| 7-02-03 | 02 | 2 | THS-22 / server_app adaptive | integration | `pytest federated-adaptive-personalized-cf/tests/test_server_integration.py::test_thesis_label_in_manifest -v` | ❌ W0 | ⬜ pending |
| 7-02-04 | 02 | 2 | THS-22 / server_app pfedrec | integration | `pytest federated-pfedrec/tests/test_server_integration.py::test_thesis_label_in_manifest -v` | ❌ W0 | ⬜ pending |
| 7-03-01 | 03 | 3 | THS-02 / matrix size | unit | `pytest scripts/foundation/tests/test_thesis_orchestrator.py::test_main_matrix_size -v` | ❌ W0 | ⬜ pending |
| 7-03-02 | 03 | 3 | THS-05 / ablation matrix | unit | `pytest scripts/foundation/tests/test_thesis_orchestrator.py::test_ablation_matrix_size -v` | ❌ W0 | ⬜ pending |
| 7-03-03 | 03 | 3 | THS-02 / skip-on-existing | unit | `pytest scripts/foundation/tests/test_thesis_orchestrator.py::test_skip_on_existing_full_tuple -v` | ❌ W0 | ⬜ pending |
| 7-03-04 | 03 | 3 | THS-02 / TOML quoting | unit | `pytest scripts/foundation/tests/test_thesis_orchestrator.py::test_run_config_quoting -v` | ❌ W0 | ⬜ pending |
| 7-03-05 | 03 | 3 | THS-02 / dry-run | smoke | `pytest scripts/foundation/tests/test_thesis_orchestrator.py::test_dry_run_no_subprocess -v` | ❌ W0 | ⬜ pending |
| 7-04-01 | 04 | 3 | THS-03 / overall extract | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_extract_overall_ndcg10 -v` | ❌ W0 | ⬜ pending |
| 7-04-02 | 04 | 3 | THS-04 / sparse extract | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_extract_sparse_ndcg10_uniform_slash -v` | ❌ W0 | ⬜ pending |
| 7-04-03 | 04 | 3 | THS-03 / D-11 win | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_d11_win_criterion -v` | ❌ W0 | ⬜ pending |
| 7-04-04 | 04 | 3 | THS-03 / D-11 no-win | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_d11_overlap_no_winner -v` | ❌ W0 | ⬜ pending |
| 7-04-05 | 04 | 3 | THS-04 / sparse footnote | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_sparse_partial_seeds -v` | ❌ W0 | ⬜ pending |
| 7-04-06 | 04 | 3 | THS-05 / ablation grouping | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_ablation_label_grouping -v` | ❌ W0 | ⬜ pending |
| 7-04-07 | 04 | 3 | THS-06 / D-20 hard-fail | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_d20_hard_fail_missing -v` | ❌ W0 | ⬜ pending |
| 7-04-08 | 04 | 3 | THS-06 / per-group columns | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_csv_per_group_columns -v` | ❌ W0 | ⬜ pending |
| 7-04-09 | 04 | 3 | THS-07 / six output files | smoke | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_six_output_files -v` | ❌ W0 | ⬜ pending |
| 7-04-10 | 04 | 3 | THS-07 / atomic write | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_atomic_write_no_tmp -v` | ❌ W0 | ⬜ pending |
| 7-04-11 | 04 | 3 | THS-07 / cell format | unit | `pytest scripts/foundation/tests/test_thesis_aggregator.py::test_cell_format -v` | ❌ W0 | ⬜ pending |
| 7-05-01 | 05 | 4 | THS-02..04 / main runs (manual) | manual | Run `python scripts/thesis/run_thesis_sweep.py --phase=main` (~19.5 hr) | ❌ W0 | ⬜ pending |
| 7-05-02 | 05 | 4 | THS-05..06 / ablations (manual) | manual | Run `python scripts/thesis/run_thesis_sweep.py --phase=ablation` (~31.5 hr) | ❌ W0 | ⬜ pending |
| 7-05-03 | 05 | 4 | THS-07 / aggregate (manual) | manual | Run `python scripts/thesis/aggregate_results.py` (verifies 6 files written, hard-fails on missing) | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `scripts/foundation/fedrec_foundation/atomic.py` — extend with `atomic_write_text` companion to `atomic_write_json`
- [ ] `scripts/foundation/fedrec_foundation/mode.py` — add `_THESIS_CROSSDEVICE_MAIN = ModeProfile(...)` + register in `_REGISTRY`
- [ ] `scripts/foundation/fedrec_foundation/manifest.py` — bump `RUN_MANIFEST_SCHEMA_VERSION` 2→3 + add `thesis_run_label`, `ablation_dimension`, `ablation_value` fields with safe defaults
- [ ] `scripts/run.py` — add `"thesis_crossdevice_main": 6040` to `MODE_NUM_SUPERNODES`
- [ ] All 4 `server_app.py` files — add `"thesis_crossdevice_main"` to BOTH `mode in (...)` tuples (W&B project gate + path gate) and read `thesis-run-label`/`ablation-dimension`/`ablation-value` from `run_config`, mutate manifest via `dataclasses.replace` BEFORE `embed_manifest_in_result`
- [ ] `scripts/foundation/tests/test_atomic.py` — add `test_atomic_write_text` covering content correctness + no `.tmp-*` leftovers
- [ ] `scripts/foundation/tests/test_mode.py` — extend with `test_thesis_crossdevice_main_profile` + extend existing `test_all_three_modes_registered` → `test_all_four_modes_registered`
- [ ] `scripts/foundation/tests/test_manifest.py` — add `test_run_manifest_schema_version_3`, `test_run_manifest_backward_compat_v2`, `test_run_manifest_carries_thesis_fields`
- [ ] `scripts/foundation/tests/test_launcher.py` — add `test_thesis_mode_dry_run` covering `scripts/run.py adaptive thesis_crossdevice_main --dry-run`
- [ ] Per-module `tests/test_server_integration.py` — add `test_thesis_label_in_manifest` (4 modules × ~10 lines each)
- [ ] `scripts/thesis/__init__.py` — empty file (per D-18 directory layout)
- [ ] `scripts/thesis/run_thesis_sweep.py` — NEW orchestrator (matrix-driven `flwr run` launcher with skip-on-existing, fail-and-log, --retry-failed)
- [ ] `scripts/thesis/aggregate_results.py` — NEW aggregator (per-run results.json reader; produces 6 output files; D-20 hard-fail on missing cells)
- [ ] `scripts/foundation/tests/test_thesis_orchestrator.py` — NEW (matrix builders + skip + dry-run + run-config quoting)
- [ ] `scripts/foundation/tests/test_thesis_aggregator.py` — NEW (result extraction + win criterion + sparse handling + missing-cell hard-fail + 6-file emission + atomic write + cell format)
- [ ] No framework install needed — pytest 7+ already in `scripts/foundation/pyproject.toml [project.optional-dependencies] dev`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Full ~19.5hr main matrix actually runs and produces 12 baseline+personalized+adaptive results.json + 9 PFedRec results.json | THS-02, THS-03, THS-04 | Wallclock-prohibitive in CI (~19.5 hr on RTX 5090) | `python scripts/thesis/run_thesis_sweep.py --phase=main`. Verify `find results/federated -name manifest.json -newer .planning/phases/07-thesis-evaluation-run/07-PLAN.md \| xargs grep -l '"thesis_run_label": "main"' \| wc -l` returns 21. |
| Full ~31.5hr ablation matrix runs and produces 21 results.json | THS-05, THS-06 | Wallclock-prohibitive in CI | `python scripts/thesis/run_thesis_sweep.py --phase=ablation`. Same pattern with `"thesis_run_label": "ablation_*"`. |
| `_thesis/main_comparison.{md,csv}` shows adaptive winning per D-11 (or, if not, ablation tables tell the story per D-12) | THS-03, THS-04, THS-07 | Visual / interpretive judgment of the thesis-claim outcome | `cat results/federated/_thesis/main_comparison.md` and verify adaptive's row is bolded under D-11 win criterion. |
| W&B dashboard shows runs grouped by `thesis/run_label` summary field | D-21 (operational) | Requires logged-in W&B session | Open the `federated-cf-cross-device` project on wandb.ai and group by `thesis/run_label`. |
| `--retry-failed` flag re-runs only cells whose `results.json` is still missing | D-23, D-31 | Requires actual flwr-run failure to surface | Force-fail one cell (e.g., kill mid-run), then `python scripts/thesis/run_thesis_sweep.py --retry-failed` — verify only the missing cell re-runs. |
| Smoke run + skip-on-rerun + aggregator hard-fail demo | THS-02, THS-06 | Bridges automated and manual verification before committing to ~50hr | (1) `python scripts/thesis/run_thesis_sweep.py --module=adaptive --seed=42 --thesis-run-label=main` (~1.5hr), (2) re-run same command (must skip), (3) `python scripts/thesis/aggregate_results.py` (must hard-fail with "Missing 32 cells: [list]"). |

---

## Edge Cases (covered in automated tests)

- **Missing seed in some module** (D-20 hard-fail surface): `test_d20_hard_fail_missing` — synthetic fixture has only `{42, 1337}` for adaptive; aggregator fails listing `(adaptive, main, 2026)` missing.
- **Schema-v2 manifest** (legacy Phase 6 run still on disk): `test_run_manifest_backward_compat_v2` — pre-v3 manifests load with default `thesis_run_label=""`; aggregator filter excludes them via `dict.get` default.
- **Empty / corrupt results.json** (mid-write crash): aggregator catches `json.JSONDecodeError` per file and continues — no propagation. Covered by an aggregator-level smoke test.
- **One seed has zero sparse evaluations** (Pitfall 10): `test_sparse_partial_seeds` — sparse-slice rendering emits `n_seeds_with_sparse=2/3` footnote when `evaluated_users_sparse=0` for one seed.
- **PFedRec mode collision**: `test_ablation_label_grouping` (and friends) — `paper_compat_pfedrec` runs with `thesis_run_label="main"` are correctly counted in main_comparison (filter is on label, not on mode).
- **Concurrent-write race**: orchestrator default is serial-within-module + serial-between-module (CONTEXT.md operational default), so single-writer is enforced; no automated race test required.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or are listed under Manual-Only with documented justification
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify (verified by manual scan of the per-task table after Plan 01..05 are written)
- [ ] Wave 0 covers all MISSING references (5 source files extended + 5 test files added/extended + 4 per-module integration tests)
- [ ] No watch-mode flags (pytest invoked with `-x -v` only)
- [ ] Feedback latency < 60s (quick run ≈ 30s)
- [ ] `nyquist_compliant: true` set in frontmatter (after Plan 01..05 verify the per-task map matches actual task IDs)

**Approval:** pending
