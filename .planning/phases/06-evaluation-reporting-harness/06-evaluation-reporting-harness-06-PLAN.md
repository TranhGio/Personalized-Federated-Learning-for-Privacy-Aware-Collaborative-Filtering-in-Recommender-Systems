---
phase: 06-evaluation-reporting-harness
plan: 06
type: execute
wave: 3
depends_on:
  - 06-evaluation-reporting-harness-01
  - 06-evaluation-reporting-harness-02
files_modified:
  - federated-pfedrec/federated_pfedrec/server_app.py
  - federated-pfedrec/tests/test_server_integration.py
  - scripts/foundation/tests/test_pfedrec_subprocess_determinism.py
autonomous: true
requirements: [EVL-01, EVL-02, EVL-03, EVL-04, EVL-06]
must_haves:
  truths:
    - "PFedRec server_app writes results.json + manifest.json to <repo>/results/federated/pfedrec/<run_id>/ via module_run_results_dir for cross-device/paper_compat modes (D-01 + D-02)"
    - "When checkpoint_rule in {best_round_restore, best_round} and best_round_num > 0, a post-restore broadcast eval round runs against ALL nodes in partition_to_node_id (D-06)"
    - "Pitfall 1 closure: D-14 PFR-08 auto-verify hook (server_app.py:1062) reads final_metrics['best'] under the new nested schema (NOT the flat top-level keys, which would silently return None and mark the run as PFR-08 FAILED)"
    - "PFR-08 hook ORDER preserved: fires AFTER embed_manifest_in_result + Phase-6 dataclasses.replace, BEFORE the W&B summary write — same order as Phase 5 Plan 04"
    - "final_metrics is now nested {best, last, best_round, last_round, final_eval_round_index}; last derives from max-key of eval_metrics_history (Pitfall 9)"
    - "W&B run.summary uses best/* and last/* namespaces; final/pfr08* surface key MIGRATED to best/pfr08* + last/pfr08*-equivalent (or kept as standalone wandb.run.summary['pfr08'] etc — see action for exact decision); existing _manifest['pfr08_verification'] post-embed mutation preserved verbatim"
    - "Manifest carries final_eval_round_index + nested metrics block via dataclasses.replace post-build mutation"
    - "Test path probe in scripts/foundation/tests/test_pfedrec_subprocess_determinism.py updated from flat *_results.json to per-run-dir */results.json glob; existing _manifest.pfr08_verification byte-identity invariant preserved"
  artifacts:
    - path: "federated-pfedrec/federated_pfedrec/server_app.py"
      provides: "PFedRec server_app with extra-eval-round + per-run-dir + nested final_metrics + best/last W&B namespaces + REWIRED PFR-08 hook reading final_metrics['best']"
      contains: "from fedrec_foundation.paths import module_run_results_dir"
    - path: "federated-pfedrec/tests/test_server_integration.py"
      provides: "5 NEW assertions: results path, extra-eval-round wired, best/last block schema, per-group exposure history, Pitfall-1 PFR-08 hook reads final_metrics['best']"
      contains: "def test_pfr08_hook_consumes_nested_best_block"
    - path: "scripts/foundation/tests/test_pfedrec_subprocess_determinism.py"
      provides: "Updated _RESULTS_DIR probe matches Phase 6 per-run-dir layout; existing _manifest.pfr08_verification byte-identity invariant preserved"
      contains: "_RESULTS_DIR / \"pfedrec\""
  key_links:
    - from: "federated-pfedrec/federated_pfedrec/server_app.py::_emit_pfr_08_verification"
      to: "federated-pfedrec/federated_pfedrec/server_app.py::final_metrics['best']"
      via: "Hook input changed from `final_metrics` (flat dict) to `final_metrics['best']` (nested best block) per Pitfall 1"
      pattern: "_emit_pfr_08_verification\\(\\s*final_metrics\\[.best.\\]"
    - from: "federated-pfedrec/federated_pfedrec/server_app.py::run_dir"
      to: "scripts/foundation/fedrec_foundation/paths.py::module_run_results_dir"
      via: "run_dir = module_run_results_dir(_MODULE='pfedrec', run_id=run_id)"
      pattern: "module_run_results_dir\\(_MODULE, run_id\\)|module_run_results_dir\\(.pfedrec.,"
---

<objective>
Wire the Plan 01+02 foundation primitives into `federated-pfedrec/federated_pfedrec/server_app.py`, AND rewire the D-14 PFR-08 auto-verify hook to consume `final_metrics["best"]` under the new nested schema (Pitfall 1 — the most fragile rewiring in the phase). Closes EVL-01/02/03/04/06 for the pfedrec module.

Purpose:
  - Replace `Path("../results/federated/pfedrec")` (server_app.py:1073) with `module_run_results_dir("pfedrec", run_id)` for cross-device/paper_compat modes (D-02). Legacy paths preserved (Pitfall 8).
  - Insert the D-06 extra-eval-round block AFTER `arrays = best_arrays` (server_app.py:901) and BEFORE the existing `final_metrics = ...` resolution (server_app.py:924-925). The block REPLACES the silent `eval_metrics_history[best_round_num]` lookup pattern.
  - Restructure `final_metrics` from flat (current line 924+) to nested `{best, last, best_round, last_round, final_eval_round_index}` per D-07.
  - **Pitfall 1 closure (HEADLINE for pfedrec):** the D-14 PFR-08 auto-verify hook at server_app.py:1062 currently reads `final_metrics["sampled_hr@10"]` and `final_metrics["sampled_ndcg@10"]` (flat keys via `_emit_pfr_08_verification(final_metrics, ...)`). Under the new nested schema those keys live at `final_metrics["best"]["sampled_hr@10"]`. The call site MUST change to `_emit_pfr_08_verification(final_metrics["best"], ...)`. Without this, the hook returns NaN deltas and stamps `[PFR-08 FAILED]` for purely structural reasons. The hook ORDER (after `embed_manifest_in_result`, before W&B summary) is unchanged — only the input dict changes.
  - Mutate manifest via `dataclasses.replace(manifest, ...)` between `build_run_manifest` and `embed_manifest_in_result`. The Phase-5 post-embed mutation `results_data["_manifest"]["pfr08_verification"] = pfr08_audit` (line 1069) is PRESERVED verbatim.
  - Migrate W&B summary keys for thesis metrics from `final/*` (line 1100) to `best/*` and `last/*`. The PFR-08-specific summary keys (`final/pfr08`, `final/pfr08_delta_hr_pts`, `final/pfr08_delta_ndcg_pts` at lines 1091-1095) MIGRATE to top-level `pfr08`, `pfr08_delta_hr_pts`, `pfr08_delta_ndcg_pts` (no namespace prefix — the audit dict is its own surface, independent of the best/last split).
  - Update path probe in `scripts/foundation/tests/test_pfedrec_subprocess_determinism.py` from flat to per-run-dir glob; preserve `_manifest.pfr08_verification` byte-identity invariant.

Output:
  - `federated-pfedrec/federated_pfedrec/server_app.py` modified.
  - `federated-pfedrec/tests/test_server_integration.py` extended: 5 NEW assertions including the headline Pitfall-1 hook test.
  - `scripts/foundation/tests/test_pfedrec_subprocess_determinism.py` updated: path probe migrated.
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/PROJECT.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/ROADMAP.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/STATE.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/06-evaluation-reporting-harness/06-CONTEXT.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/06-evaluation-reporting-harness/06-RESEARCH.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/06-evaluation-reporting-harness/06-VALIDATION.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec/federated_pfedrec/server_app.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec/federated_pfedrec/strategy.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec/tests/test_server_integration.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/scripts/foundation/tests/test_pfedrec_subprocess_determinism.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-04-SUMMARY.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-05-SUMMARY.md

<interfaces>
<!-- Wave 1 deps (Plans 01 + 02) -->
```python
from fedrec_foundation.paths import module_run_results_dir, repo_root
from fedrec_foundation.atomic import atomic_write_json
from fedrec_foundation.manifest import (
    build_run_manifest,
    embed_manifest_in_result,
    write_manifest_sibling,  # accepts sibling_name kwarg
)
from dataclasses import replace as dataclass_replace
```

<!-- PFR-08 auto-verify hook signature (server_app.py:285-357) -->
```python
def _emit_pfr_08_verification(
    final_metrics: Dict[str, float],   # <-- CURRENTLY FLAT; AFTER PHASE 6 must receive final_metrics["best"]
    reference_path: Path,
    tolerance_pts: float = 2.0,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Reads final_metrics["sampled_hr@10"] and final_metrics["sampled_ndcg@10"]."""
    ...
    our_hr = final_metrics.get("sampled_hr@10", float("nan"))
    our_ndcg = final_metrics.get("sampled_ndcg@10", float("nan"))
    ...
```

<!-- Existing pfedrec server_app.py drop sites (RESEARCH §Pattern 2 module table row 4) -->
```python
# Line 568: partition_to_node_id (G-03-01)
# Line 608+838: best_arrays / best_arrays = ArrayRecord({...})
# Line 901: arrays = best_arrays  (D-13 best-round restore — Phase 5 Plan 04)
# Line 924-929: final_metrics: Dict[str, Any] = dict(eval_metrics_history.get(...))   # D-06 BUG
# Line 1026-1041: build_run_manifest + embed_manifest_in_result
# Line 1062-1067: D-14 PFR-08 hook fires (HOOK ORDER ANCHOR — must NOT move)
#   pfr08_passed, pfr08_log_line, pfr08_audit = _emit_pfr_08_verification(
#       final_metrics, reference_path=..., tolerance_pts=2.0,
#   )                                  # ^^^ INPUT MUST CHANGE TO final_metrics["best"]
# Line 1069: results_data["_manifest"]["pfr08_verification"] = pfr08_audit   # PRESERVE VERBATIM
# Line 1073-1080: results_dir = Path("../results/federated/pfedrec") + write_manifest_sibling
# Line 1091-1095: wandb.run.summary["final/pfr08"] etc.   # MIGRATE
# Line 1100: wandb.run.summary[f"final/{key}"]   # MIGRATE for thesis metrics
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Wire module_run_results_dir + extra-eval-round + nested final_metrics + best/last W&B namespaces into pfedrec server_app.py; REWIRE D-14 PFR-08 hook to consume final_metrics['best'] (Pitfall 1); ship 5 NEW integration tests including the headline hook regression guard; update test_pfedrec_subprocess_determinism.py path probe</name>
  <files>federated-pfedrec/federated_pfedrec/server_app.py, federated-pfedrec/tests/test_server_integration.py, scripts/foundation/tests/test_pfedrec_subprocess_determinism.py</files>
  <read_first>
    - federated-pfedrec/federated_pfedrec/server_app.py — current state (CRITICAL line refs: 100-103 manifest imports, 285-357 _emit_pfr_08_verification function — DO NOT change its signature, ONLY the call site, 568 partition_to_node_id, 608+838+901 best_arrays, 924-929 final_metrics flat — D-06 BUG, 1026-1041 build_run_manifest + embed_manifest_in_result, 1055-1069 D-14 PFR-08 hook fires + post-embed mutation, 1073-1080 results_dir + write_manifest_sibling, 1091-1100 W&B summary final/* + pfr08 keys)
    - federated-pfedrec/federated_pfedrec/strategy.py — PFedRecSplitFedAvg.aggregate_evaluate (DO NOT TOUCH)
    - federated-pfedrec/tests/test_server_integration.py — current shape (extend, do not rewrite)
    - scripts/foundation/tests/test_pfedrec_subprocess_determinism.py — current `_RESULTS_DIR` glob + `_manifest.pfr08_verification` byte-identity invariant (preserve invariant; update probe)
    - .planning/phases/06-evaluation-reporting-harness/06-CONTEXT.md §decisions D-01..D-09
    - .planning/phases/06-evaluation-reporting-harness/06-RESEARCH.md §Pattern 2 pfedrec drop site row + §Common Pitfalls Pitfall 1 (D-14 hook stale read) — HEADLINE for this plan
    - .planning/phases/06-evaluation-reporting-harness/06-VALIDATION.md §Per-Task Verification Map rows 6-06-01, 6-06-02, 6-06-03
    - .planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-04-SUMMARY.md — D-14 hook ordering documentation
  </read_first>
  <behavior>
    - Test 1 (test_results_path_repo_root_anchored): Mock-mode test with `mode="paper_compat_pfedrec"`; assert `module_run_results_dir("pfedrec", run_id)` resolves to `<repo>/results/federated/pfedrec/<run_id>/`; assert `results.json` and `manifest.json` exist there (D-04).
    - Test 2 (test_extra_eval_round_after_best_arrays_restore): Build `eval_metrics_history` with rounds {1: ndcg=0.30, 89: ndcg=0.44, 100: ndcg=0.41} so `best_round_num=89` (matching paper anchor). Mock `grid.send_and_receive`; assert `final_eval_round_index == 101`; assert `strategy.aggregate_evaluate` called with `final_eval_round_index`.
    - Test 3 (test_canonical_artifact_carries_best_and_last_blocks): Same fixture; assert nested final_metrics schema; assert `_manifest["schema_version"] == 2`; assert `_manifest["final_eval_round_index"] == 101`; assert `_manifest["metrics"]` mirrors `final_metrics`.
    - Test 4 (test_round_metrics_history_carries_per_group_exposure): Same fixture; assert `evaluated_users_sparse|medium|dense` present in at least one round entry of `results["eval_metrics_history"]`.
    - Test 5 (test_pfr08_hook_consumes_nested_best_block) — **PITFALL 1 HEADLINE REGRESSION GUARD**: Construct fake `final_metrics` in the new nested schema with `final_metrics["best"] = {"sampled_hr@10": 0.7287, "sampled_ndcg@10": 0.4413}` (paper anchor values, within ±2 of 0.729/0.441). Patch the call site so `_emit_pfr_08_verification` is invoked with `final_metrics["best"]` (NOT the top-level `final_metrics`). Assert: (a) the hook returns `passed=True` (the nested-best values match paper within tolerance), (b) the hook does NOT return NaN deltas, (c) `pfr08_audit["our_hr"] == 0.7287` and `pfr08_audit["our_ndcg"] == 0.4413` (proving the hook saw the nested-best dict, not the top-level final_metrics where those keys are absent). Then construct a NEGATIVE-PATH fixture: pass the legacy flat dict (no `"best"` sub-key); call `_emit_pfr_08_verification(final_metrics_flat["best"])` — KeyError must surface OR (if the test guards against KeyError) the hook returns `passed=False` with `delta_hr=nan`. The negative path proves the regression guard catches the schema-drift bug Pitfall 1 describes.
  </behavior>
  <action>
**Edit 1: Add foundation imports.** Locate manifest imports around line 100-103. Add:

```python
from fedrec_foundation.paths import module_run_results_dir, repo_root
from fedrec_foundation.atomic import atomic_write_json
from dataclasses import replace as dataclass_replace
```

**Edit 2: Extract module-name local constant.** Inside `@app.main()`:

```python
_MODULE: str = "pfedrec"   # cross-references: build_run_manifest, module_run_results_dir
```

**Edit 3: Insert the D-06 extra-eval-round block.** AFTER line 901 (`arrays = best_arrays`) and BEFORE the current `final_metrics` resolution (line 924). The block should mirror the in-loop eval-config build site (verify by reading the existing in-loop eval-config builder; for pfedrec it is in the FL loop body — find the `eval_messages.append(grid.create_message(...))` block and copy its eval_config shape verbatim). Insert:

```python
    # =========================================================================
    # D-06: extra eval round on the restored best-round state. All nodes
    # broadcast (no sampling). Result becomes the canonical `final_metrics["best"]`.
    # =========================================================================
    final_eval_round_index: int = 0
    best_round_metrics: Dict[str, Any] = {}

    if checkpoint_rule in ("best_round_restore", "best_round") and best_round_num > 0:
        final_eval_round_index = actual_rounds + 1
        print(
            f"\n[D-06] Broadcasting extra eval round {final_eval_round_index} "
            f"on restored best-round state (best_round={best_round_num}, "
            f"target nodes={len(partition_to_node_id)})..."
        )

        eval_node_ids = sorted(partition_to_node_id.values())
        extra_eval_messages = []
        for nid in eval_node_ids:
            eval_config = ConfigRecord({"lr": lr})
            content = RecordDict({"arrays": arrays, "config": eval_config})
            extra_eval_messages.append(grid.create_message(
                content=content,
                message_type="evaluate",
                dst_node_id=nid,
                group_id=f"final_eval_round_{final_eval_round_index}",
            ))
        extra_eval_responses = list(grid.send_and_receive(extra_eval_messages))

        extra_results: List[Tuple[ClientProxy, EvaluateRes]] = []
        for response in extra_eval_responses:
            if response.has_error():
                continue
            m = dict(response.content.get("metrics", MetricRecord()))
            num_examples = int(
                m.get("num_training_examples", m.get("evaluated_users", m.get("num-examples", 1)))
            )
            extra_results.append((
                DummyClientProxy(str(response.metadata.src_node_id)),
                EvaluateRes(
                    status=Status(code=Code.OK, message="ok"),
                    loss=float(m.get("eval_loss", 0.0)),
                    num_examples=num_examples,
                    metrics=m,
                ),
            ))
        if extra_results:
            _agg_loss, thesis = strategy.aggregate_evaluate(
                final_eval_round_index, extra_results, []
            )
            best_round_metrics = dict(thesis) if thesis else {}
            print(
                f"[D-06] Extra eval complete. Canonical best/sampled_ndcg@10="
                f"{best_round_metrics.get('sampled_ndcg@10')} "
                f"best/sampled_hr@10={best_round_metrics.get('sampled_hr@10')}"
            )
        else:
            print("[D-06] WARNING: no extra-eval responses; best block falls back to in-loop value.")
```

**Edit 4: Restructure final_metrics.** REPLACE the existing flat resolution at lines 924-929 (currently `final_metrics: Dict[str, Any] = dict(eval_metrics_history.get(final_round_for_metrics, {}))` or similar — read the file to confirm exact shape). New body:

```python
    # =========================================================================
    # D-07: nested final_metrics. `best` from D-06 extra-eval-round; `last`
    # from max-key of eval_metrics_history (Pitfall 9).
    # =========================================================================
    if eval_metrics_history:
        last_round = max(eval_metrics_history.keys())
        last_block = dict(eval_metrics_history[last_round])
    else:
        last_round = 0
        last_block = {}

    final_metrics = {
        "best": best_round_metrics or last_block,
        "last": last_block,
        "best_round": best_round_num if best_round_num > 0 else last_round,
        "last_round": last_round,
        "final_eval_round_index": final_eval_round_index,
    }
```

**Edit 5: Mutate manifest with new fields.** AFTER `manifest = build_run_manifest(...)` (~line 1026) and BEFORE `embed_manifest_in_result(manifest, results_data)` (~line 1041), insert:

```python
    manifest = dataclass_replace(
        manifest,
        final_eval_round_index=final_eval_round_index,
        metrics=results_data["final_metrics"],
    )
```

**Edit 6 — PITFALL 1 HEADLINE: REWIRE the D-14 PFR-08 hook input.** Locate the hook call site at lines 1062-1067:

```python
    pfr08_passed, pfr08_log_line, pfr08_audit = _emit_pfr_08_verification(
        final_metrics,                      # <-- OLD: flat dict
        reference_path=...,
        tolerance_pts=2.0,
    )
```

Change ONLY the first arg to `final_metrics["best"]`:

```python
    # Pitfall 1 closure: under the new D-07 nested schema, sampled_hr@10 and
    # sampled_ndcg@10 live at final_metrics["best"][...], NOT at
    # final_metrics[...]. Passing final_metrics directly would make the hook
    # read None for both keys and stamp PFR-08 FAILED with NaN deltas.
    pfr08_passed, pfr08_log_line, pfr08_audit = _emit_pfr_08_verification(
        final_metrics["best"],
        reference_path=...,
        tolerance_pts=2.0,
    )
```

**HOOK ORDER UNCHANGED**: the hook still fires AFTER `embed_manifest_in_result(manifest, results_data)` (line 1041) and BEFORE the W&B summary write (line 1085+). Only the input dict changes. The `_emit_pfr_08_verification` function signature itself is NOT modified.

**Edit 7: PRESERVE existing pfr08_verification post-embed mutation.** Line 1069 (`results_data["_manifest"]["pfr08_verification"] = pfr08_audit`) MUST remain verbatim. The audit dict is independent of the Phase-6 schema-v2 metrics field.

**Edit 8: Replace results-dir + filename + manifest-sibling.** REPLACE lines 1073-1080. New body:

```python
    if mode in ("benchmark_cross_device", "paper_compat_pfedrec"):
        run_dir = module_run_results_dir(_MODULE, run_id)
        results_filename = run_dir / "results.json"
        sibling_kwarg = {"sibling_name": "manifest.json"}
    else:  # cross_silo_legacy
        legacy_dir = repo_root() / "results" / "federated" / "pfedrec"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        results_filename = legacy_dir / f"{run_id}_results.json"
        sibling_kwarg = {}

    atomic_write_json(str(results_filename), results_data)
    sibling_path = write_manifest_sibling(manifest, results_filename, **sibling_kwarg)
    print(f"Results saved to: {results_filename.resolve()}")
    print(f"Manifest sibling: {sibling_path.resolve()}")
```

**Edit 9: Migrate W&B summary keys.** REPLACE the legacy block at lines 1091-1100. The PFR-08 audit summary keys MIGRATE from `final/pfr08*` to `pfr08*` (no namespace prefix — the audit dict is its own surface). Thesis metrics MIGRATE from `final/*` to `best/*` + `last/*`:

```python
        # PFR-08 audit surface (independent of best/last namespacing — top-level keys)
        wandb.run.summary["pfr08"] = bool(pfr08_passed)
        wandb.run.summary["pfr08_delta_hr_pts"] = float(pfr08_audit.get("delta_hr_pts", float("nan")))
        wandb.run.summary["pfr08_delta_ndcg_pts"] = float(pfr08_audit.get("delta_ndcg_pts", float("nan")))

        # Thesis metrics (D-07 best/* + last/* namespaces; final/* deprecated)
        for key, value in final_metrics["best"].items():
            if isinstance(value, (int, float)):
                wandb.run.summary[f"best/{key}"] = value
        for key, value in final_metrics["last"].items():
            if isinstance(value, (int, float)):
                wandb.run.summary[f"last/{key}"] = value
        wandb.run.summary["best_round"] = final_metrics["best_round"]
        wandb.run.summary["last_round"] = final_metrics["last_round"]
        wandb.run.summary["final_eval_round_index"] = final_metrics["final_eval_round_index"]
```

**Edit 10: Update test_pfedrec_subprocess_determinism.py path probe.** Locate the `_RESULTS_DIR` constant + glob pattern. Update the glob from the legacy flat layout (likely `_RESULTS_DIR.glob("pfedrec/*_results.json")` or `_RESULTS_DIR.rglob("*pfedrec*_results.json")`) to the new per-run-dir layout: `_RESULTS_DIR.glob("pfedrec/*/results.json")` (or, if `_RESULTS_DIR` is already pfedrec-scoped, `_RESULTS_DIR.glob("*/results.json")`). PRESERVE the existing `_manifest.pfr08_verification` byte-identity invariant.

**Edit 11: Extend test_server_integration.py with 5 NEW tests.** Read the existing file shape; reuse mocking style. Add Tests 1-5 from the behavior block. Test 5 (`test_pfr08_hook_consumes_nested_best_block`) is the headline — explicitly construct the new nested schema, pass `final_metrics["best"]` to `_emit_pfr_08_verification`, assert the hook reads the nested-best values correctly.

**Verify by running:**

```bash
cd federated-pfedrec && pytest tests/test_server_integration.py -x -v -k "results_path or extra_eval or pfr08_consumes_best or best_last_blocks or per_group_exposure"
cd scripts/foundation && pytest tests/test_pfedrec_subprocess_determinism.py -x -v -m slow
```
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec && pytest tests/test_server_integration.py -x -v -k "results_path or extra_eval or pfr08_consumes_best or best_last_blocks or per_group_exposure"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "from fedrec_foundation.paths import module_run_results_dir" federated-pfedrec/federated_pfedrec/server_app.py` returns 1
    - `grep -c "from dataclasses import replace as dataclass_replace" federated-pfedrec/federated_pfedrec/server_app.py` returns 1
    - `grep -c "_MODULE: str = .pfedrec." federated-pfedrec/federated_pfedrec/server_app.py` returns 1
    - `grep -c "Path(.\\.\\./results/federated/pfedrec.)" federated-pfedrec/federated_pfedrec/server_app.py` returns 0 (D-02 cutover)
    - `grep -c "module_run_results_dir(_MODULE, run_id)" federated-pfedrec/federated_pfedrec/server_app.py` returns 1
    - `grep -c "final_eval_round_index" federated-pfedrec/federated_pfedrec/server_app.py` returns at least 5
    - `grep -c "_emit_pfr_08_verification(\\s*final_metrics\\[.best.\\]" federated-pfedrec/federated_pfedrec/server_app.py` returns 1 (Pitfall 1 closure — hook input changed to nested-best)
    - `grep -E "_emit_pfr_08_verification\\(\\s*final_metrics," federated-pfedrec/federated_pfedrec/server_app.py` returns 0 (legacy flat-input call site removed)
    - `grep -c "wandb.run.summary\\[.pfr08.\\]" federated-pfedrec/federated_pfedrec/server_app.py` returns 1 (PFR-08 audit migrated from final/pfr08 to top-level pfr08)
    - `grep -c "wandb.run.summary\\[.final/pfr08.\\]" federated-pfedrec/federated_pfedrec/server_app.py` returns 0 (legacy namespace removed)
    - `grep -c "wandb.run.summary\\[f.final/" federated-pfedrec/federated_pfedrec/server_app.py` returns 0 (thesis-metric namespace removed)
    - `grep -c "wandb.run.summary\\[f.best/" federated-pfedrec/federated_pfedrec/server_app.py` returns 1
    - `grep -c "wandb.run.summary\\[f.last/" federated-pfedrec/federated_pfedrec/server_app.py` returns 1
    - `grep -c "results_data\\[._manifest.\\]\\[.pfr08_verification.\\]" federated-pfedrec/federated_pfedrec/server_app.py` returns at least 1 (Phase-5 post-embed mutation preserved)
    - `grep -c "max(eval_metrics_history.keys())" federated-pfedrec/federated_pfedrec/server_app.py` returns 1 (Pitfall 9)
    - `grep -c "if mode in (.benchmark_cross_device., .paper_compat_pfedrec.)" federated-pfedrec/federated_pfedrec/server_app.py` returns 1 (Pitfall 8)
    - `grep -c "sibling_name=.manifest.json." federated-pfedrec/federated_pfedrec/server_app.py` returns 1
    - `grep -c "atomic_write_json" federated-pfedrec/federated_pfedrec/server_app.py` returns at least 1
    - `grep -E "pfedrec/\\*/results\\.json|.\\*/results\\.json" scripts/foundation/tests/test_pfedrec_subprocess_determinism.py` returns at least 1 line (path probe migrated)
    - `cd federated-pfedrec && pytest tests/test_server_integration.py -x -v -k "results_path or extra_eval or pfr08_consumes_best or best_last_blocks or per_group_exposure"` exits 0 with all 5 NEW tests passing
    - `cd federated-pfedrec && pytest tests/ -q -m "not slow"` exits 0 (no regressions; PFR-08 hook still functional)
  </acceptance_criteria>
  <done>
    - server_app.py: extra-eval-round inserted after best-arrays restore (line 901); flat final_metrics restructured to nested schema; D-14 PFR-08 hook input REWIRED from `final_metrics` to `final_metrics["best"]` (Pitfall 1 closure) — HOOK ORDER preserved (after embed_manifest_in_result + Phase-6 dataclasses.replace, before W&B summary); manifest mutated via dataclasses.replace BEFORE embed_manifest_in_result; existing pfr08_verification post-embed mutation preserved verbatim; W&B summary final/* -> best/* + last/* (thesis metrics); final/pfr08* -> top-level pfr08* (PFR-08 audit surface); results path resolves via module_run_results_dir for cross-device/paper_compat; cross-silo legacy preserved (D-03); atomic_write_json replaces json.dump
    - test_server_integration.py: 5 NEW tests pinning EVL-01/02/03/04/06 + the headline Pitfall-1 PFR-08-hook-consumes-nested-best regression guard
    - test_pfedrec_subprocess_determinism.py: _RESULTS_DIR glob updated to per-run-dir layout; existing pfr08_verification byte-identity invariant preserved
    - Existing pfedrec tests remain GREEN
  </done>
</task>

</tasks>

<verification>
- Imports clean: `python -c "from federated_pfedrec import server_app; print('ok')"` exits 0
- Pitfall 1 mechanically closed: `grep -c "_emit_pfr_08_verification(\\s*final_metrics\\[.best.\\]" federated-pfedrec/federated_pfedrec/server_app.py` returns 1
- Legacy hook call removed: `grep -E "_emit_pfr_08_verification\\(\\s*final_metrics," federated-pfedrec/federated_pfedrec/server_app.py` returns 0
- pfr08_verification audit dict surface preserved: `grep -c "results_data\\[._manifest.\\]\\[.pfr08_verification.\\]" federated-pfedrec/federated_pfedrec/server_app.py` returns at least 1
- 5 NEW integration tests pass
- Path probe migrated: `grep -E "pfedrec/.+/results\\.json" scripts/foundation/tests/test_pfedrec_subprocess_determinism.py` matches at least 1 line
- D-18 surgical scope: `git diff --stat` shows ONLY changes to server_app.py + test_server_integration.py + test_pfedrec_subprocess_determinism.py
- Wave-3 file-disjointness held: this plan modifies ONLY federated-pfedrec/ + ONE foundation test file; Plan 07 (cross-cutting) is concurrent but touches sweep.yaml + ONE test file per module under different names
</verification>

<success_criteria>
- federated-pfedrec/federated_pfedrec/server_app.py:
  - Extra-eval-round block (D-06) wired AFTER best-arrays restore (line 901) and BEFORE final_metrics resolution
  - Nested final_metrics block (D-07) replaces flat dict; D-06-forbidden lookup removed
  - **D-14 PFR-08 hook REWIRED to consume `final_metrics["best"]` (Pitfall 1 closure — the hook still reads `sampled_hr@10` / `sampled_ndcg@10` keys, but from the nested-best dict). Hook ORDER unchanged: AFTER embed_manifest_in_result + Phase-6 dataclasses.replace, BEFORE W&B summary write**
  - module_run_results_dir replaces Path("../results/federated/pfedrec") for cross-device/paper_compat; legacy cross-silo path preserved (D-03 + Pitfall 8)
  - W&B summary uses best/* + last/* for thesis metrics; PFR-08 audit migrates `final/pfr08*` -> top-level `pfr08*` (independent surface)
  - Manifest carries final_eval_round_index + metrics via dataclasses.replace BEFORE embed_manifest_in_result; existing pfr08_verification post-embed mutation preserved verbatim
  - D-04 clean filename via sibling_name="manifest.json"
  - atomic_write_json replaces json.dump
- 5 NEW integration tests in test_server_integration.py covering EVL-01/02/03/04/06 + the headline Pitfall-1 PFR-08 hook regression guard
- Updated path probe in scripts/foundation/tests/test_pfedrec_subprocess_determinism.py preserves existing _manifest.pfr08_verification byte-identity invariant
- Pitfall 9 closure: last_round = max(eval_metrics_history.keys())
- Files outside listed `files_modified` remain byte-identical to pre-Plan-06 state (D-18 surgical)
</success_criteria>

<output>
After completion, create `.planning/phases/06-evaluation-reporting-harness/06-evaluation-reporting-harness-06-SUMMARY.md` covering:
- Path migration (D-02)
- Extra-eval-round wiring (D-06)
- Nested final_metrics schema (D-07)
- **HEADLINE: Pitfall 1 closure — D-14 PFR-08 hook input rewired from `final_metrics` (flat) to `final_metrics["best"]` (nested-best); hook ORDER preserved verbatim**
- W&B summary key migration (best/* + last/* for thesis metrics; final/pfr08* -> top-level pfr08*)
- Phase-5 pfr08_verification post-embed mutation explicitly preserved
- Path probe update in test_pfedrec_subprocess_determinism.py (preserves pfr08_verification byte-identity invariant)
- 5 NEW integration tests (the Pitfall-1 hook-consumes-nested-best test is the headline)
</output>
</content>
</invoke>