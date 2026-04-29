---
phase: 06-evaluation-reporting-harness
plan: 04
type: execute
wave: 2
depends_on:
  - 06-evaluation-reporting-harness-01
  - 06-evaluation-reporting-harness-02
files_modified:
  - federated-personalized-cf/federated_personalized_cf/server_app.py
  - federated-personalized-cf/tests/test_server_integration.py
  - scripts/foundation/tests/test_personalized_determinism.py
autonomous: true
requirements: [EVL-01, EVL-02, EVL-03, EVL-04, EVL-06]
must_haves:
  truths:
    - "Personalized server_app writes results.json + manifest.json to <repo>/results/federated/personalized/<run_id>/ via module_run_results_dir for cross-device modes (D-01 + D-02)"
    - "When checkpoint_rule in {best_round_restore, best_round} and best_round_num > 0, a post-restore broadcast eval round runs against ALL nodes in partition_to_node_id (D-06)"
    - "Replaces the silent eval_metrics_history[best_round_num] lookup at server_app.py:796 with the broadcast result — closing the bug D-06 forbids"
    - "final_metrics is now nested {best, last, best_round, last_round, final_eval_round_index}; last derives from max-key of eval_metrics_history (Pitfall 9)"
    - "Cross-silo (mode != benchmark_cross_device) writes legacy flat <repo>/results/federated/<run_id>_results.json (D-03 + Pitfall 8)"
    - "W&B run.summary uses best/* and last/* namespaces; final/* removed"
    - "Manifest carries final_eval_round_index + nested metrics block via dataclasses.replace post-build mutation"
    - "Test path probe in scripts/foundation/tests/test_personalized_determinism.py updated from flat *_results.json to per-run */results.json glob"
  artifacts:
    - path: "federated-personalized-cf/federated_personalized_cf/server_app.py"
      provides: "Personalized server_app with extra-eval-round + per-run-dir + nested final_metrics + best/last W&B namespaces"
      contains: "from fedrec_foundation.paths import module_run_results_dir"
    - path: "federated-personalized-cf/tests/test_server_integration.py"
      provides: "4 NEW assertions: results path repo-root anchored, extra-eval-round wired (replaces line-796 lookup), best/last block schema, per-round exposure history"
      contains: "def test_results_path_repo_root_anchored"
    - path: "scripts/foundation/tests/test_personalized_determinism.py"
      provides: "Updated _RESULTS_DIR probe matches Phase 6 per-run-dir layout"
      contains: "_RESULTS_DIR / \"personalized\""
  key_links:
    - from: "federated-personalized-cf/federated_personalized_cf/server_app.py::run_dir"
      to: "scripts/foundation/fedrec_foundation/paths.py::module_run_results_dir"
      via: "run_dir = module_run_results_dir(_MODULE='personalized', run_id=run_id)"
      pattern: "module_run_results_dir\\(_MODULE, run_id\\)|module_run_results_dir\\(.personalized.,"
    - from: "federated-personalized-cf/federated_personalized_cf/server_app.py::extra_eval_round"
      to: "federated-personalized-cf/federated_personalized_cf/strategy.py::PersonalizedSplitFedAvg.aggregate_evaluate"
      via: "strategy.aggregate_evaluate(final_eval_round_index, extra_results, [])"
      pattern: "strategy\\.aggregate_evaluate\\(final_eval_round_index"
---

<objective>
Wire the Plan 01+02 foundation primitives into `federated-personalized-cf/federated_personalized_cf/server_app.py`. This closes EVL-01/02/03/04/06 for the personalized module AND replaces the canonical-bug pattern at server_app.py:796 (`final_metrics = dict(eval_metrics_history.get(final_round_for_metrics, {}))`) — exactly what D-06 forbids.

Purpose:
  - Replace `Path("../results/federated")` (server_app.py:898) with `module_run_results_dir("personalized", run_id)` for cross-device modes (D-02). Cross-silo legacy paths preserved via mode branch (Pitfall 8).
  - Insert the D-06 extra-eval-round block AFTER `arrays = best_arrays` (server_app.py:775) and BEFORE the existing "Using federated evaluation metrics..." print (server_app.py:783). The extra-eval-round REPLACES the silent `eval_metrics_history[final_round_for_metrics]` lookup at line 796 — the bug D-06 was explicitly created to close.
  - Restructure `final_metrics` from flat dict (line 796) to nested `{best, last, best_round, last_round, final_eval_round_index}` per D-07.
  - Migrate W&B summary keys from `final/*` (line 814) to `best/*` and `last/*`.
  - Mutate manifest via `dataclasses.replace(manifest, final_eval_round_index=N, metrics=...)` between `build_run_manifest` and `embed_manifest_in_result`.
  - Use `write_manifest_sibling(..., sibling_name="manifest.json")` for cross-device per D-04.
  - Update path probe in existing `scripts/foundation/tests/test_personalized_determinism.py` from flat `*_results.json` to per-run-dir `*/results.json` glob.

Output:
  - `federated-personalized-cf/federated_personalized_cf/server_app.py` modified (extra-eval-round + nested final_metrics + per-run-dir + W&B namespace migration).
  - `federated-personalized-cf/tests/test_server_integration.py` extended: 4 NEW assertions.
  - `scripts/foundation/tests/test_personalized_determinism.py` updated: `_RESULTS_DIR.glob` probe migrated to new layout.
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
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-personalized-cf/federated_personalized_cf/server_app.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-personalized-cf/federated_personalized_cf/strategy.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-personalized-cf/tests/test_server_integration.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/scripts/foundation/tests/test_personalized_determinism.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/03-personalized-migration/03-personalized-migration-04-SUMMARY.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/03-personalized-migration/03-personalized-migration-05-SUMMARY.md

<interfaces>
<!-- Wave 1 deps (Plans 01 + 02) -->
```python
from fedrec_foundation.paths import module_run_results_dir, repo_root
from fedrec_foundation.atomic import atomic_write_json
from fedrec_foundation.manifest import (
    build_run_manifest,
    embed_manifest_in_result,
    write_manifest_sibling,  # accepts sibling_name kwarg per Plan 02
)
from dataclasses import replace as dataclass_replace
```

<!-- Existing personalized server_app.py drop sites -->
```python
# Line 435: partition_to_node_id: Dict[int, int] = {}  (G-03-01 discovery; reuse)
# Line 492: best_arrays = arrays  (D-27 fallback)
# Line 719: best_arrays = ArrayRecord({...})  (D-27 snapshot inside checkpoint branch)
# Line 770-775: D-27 best-arrays restore (`arrays = best_arrays`)
# Line 783: "Using federated evaluation metrics..." print  (DROP SITE for extra-eval-round; insert BEFORE this print)
# Line 786-799: final_round_for_metrics + final_metrics = dict(eval_metrics_history.get(...))  (REPLACE — D-06 bug)
# Line 814: wandb.run.summary[f"final/{key}"]  (MIGRATE -> best/* and last/*)
# Line 851: results_data["final_metrics"] = final_metrics  (now points to nested dict)
# Line 878-890: build_run_manifest(...)  (extend with dataclass_replace)
# Line 893: embed_manifest_in_result(manifest, results_data)  (must come AFTER replace)
# Line 898-906: results_dir = Path("../results/federated") + write_manifest_sibling  (REPLACE)
```

<!-- Existing personalized strategy.py — DO NOT MODIFY -->
```python
# federated-personalized-cf/federated_personalized_cf/strategy.py
# Already provides PersonalizedSplitFedAvg.aggregate_evaluate (sufficient-stat aggregator)
# emitting per-group sampled_hr@10 / sampled_ndcg@10 / evaluated_users keys.
# Phase 6 reuses this verbatim — no strategy.py touches.
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Wire module_run_results_dir + extra-eval-round + nested final_metrics + best/last W&B namespaces into federated-personalized-cf/federated_personalized_cf/server_app.py; ship 4 NEW integration tests in test_server_integration.py; update test_personalized_determinism.py path probe</name>
  <files>federated-personalized-cf/federated_personalized_cf/server_app.py, federated-personalized-cf/tests/test_server_integration.py, scripts/foundation/tests/test_personalized_determinism.py</files>
  <read_first>
    - federated-personalized-cf/federated_personalized_cf/server_app.py — current state (CRITICAL line refs: 72-75 imports, 435 partition_to_node_id, 492+719+770-775 best_arrays, 783 federated-eval print, 786-799 final_round_for_metrics + flat final_metrics — THE BUG, 805-814 W&B final/* loop, 878-890 build_run_manifest, 893 embed_manifest_in_result, 898-906 results_dir + write_manifest_sibling)
    - federated-personalized-cf/federated_personalized_cf/strategy.py — PersonalizedSplitFedAvg.aggregate_evaluate (sufficient-stat path the extra-eval-round invokes; DO NOT TOUCH)
    - federated-personalized-cf/tests/test_server_integration.py — current shape (extend, do not rewrite)
    - scripts/foundation/tests/test_personalized_determinism.py — current `_RESULTS_DIR` glob and shape (update path probe in-place; preserve other test logic verbatim)
    - .planning/phases/06-evaluation-reporting-harness/06-CONTEXT.md §decisions D-01..D-09
    - .planning/phases/06-evaluation-reporting-harness/06-RESEARCH.md §Pattern 2 (extra-eval-round) + §Pattern 3 (nested final_metrics) + §Code Examples Example 3-4-5 + §Common Pitfalls Pitfall 8 + Pitfall 9
    - .planning/phases/06-evaluation-reporting-harness/06-VALIDATION.md §Per-Task Verification Map rows 6-04-01 + 6-04-02
  </read_first>
  <behavior>
    - Test 1 (test_results_path_repo_root_anchored): Mock-mode test that exercises the result-write path with `mode="benchmark_cross_device"`; assert `module_run_results_dir("personalized", run_id)` resolves to `<repo>/results/federated/personalized/<run_id>/`; assert `results.json` exists at `run_dir / "results.json"`; assert `manifest.json` exists at `run_dir / "manifest.json"` (D-04).
    - Test 2 (test_extra_eval_round_replaces_history_lookup): Build `eval_metrics_history` with rounds {1: ndcg=0.30, 2: ndcg=0.42, 3: ndcg=0.40} so `best_round_num=2` and `last_round=3`. Mock `grid.send_and_receive` for the extra-eval-round (~2 fake nodes). Assert that AFTER the extra-eval-round, `final_metrics["best"]["sampled_ndcg@10"]` equals the broadcast aggregated value (NOT `eval_metrics_history[2]["sampled_ndcg@10"]`). Assert `final_eval_round_index == 4`. Crucially: assert that `eval_metrics_history.get(2, {}).get("sampled_ndcg@10")` and `final_metrics["best"]["sampled_ndcg@10"]` are different in the test fixture (proves the broadcast result is what landed in `best`, not the cached round-2 value).
    - Test 3 (test_canonical_artifact_carries_best_and_last_blocks): Same fixture; load `results.json`; assert `set(results["final_metrics"].keys()) == {"best", "last", "best_round", "last_round", "final_eval_round_index"}`; assert `results["_manifest"]["schema_version"] == 2`; assert `results["_manifest"]["final_eval_round_index"] == 4`; assert `results["_manifest"]["metrics"] == results["final_metrics"]`.
    - Test 4 (test_round_metrics_history_carries_per_group_exposure): Same fixture; assert at least one round in `results["eval_metrics_history"]` carries all three of `evaluated_users_sparse`, `evaluated_users_medium`, `evaluated_users_dense` (D-09 per-round counts).
    - Test 5 (existing slow guard, in scripts/foundation/tests/test_personalized_determinism.py): UPDATE the `_RESULTS_DIR.glob(...)` probe pattern to look for `<repo>/results/federated/personalized/*/results.json` (the new per-run-dir layout) instead of the old flat `<repo>/results/federated/personalized/*_results.json` pattern. The byte-identity invariants on `selected_clients_per_round` and `partition_*.pt` cache content REMAIN unchanged.
  </behavior>
  <action>
**Edit 1: Add foundation imports.** Locate the existing manifest import block around server_app.py:72-75. Add:

```python
from fedrec_foundation.paths import module_run_results_dir, repo_root
from fedrec_foundation.atomic import atomic_write_json
from dataclasses import replace as dataclass_replace
```

**Edit 2: Extract module-name local constant.** Inside `@app.main()` near the top (after `run_id` is materialized), add:

```python
_MODULE: str = "personalized"   # cross-references: build_run_manifest, module_run_results_dir
```

**Edit 3: Insert the D-06 extra-eval-round block.** AFTER line 775 (`arrays = best_arrays`) AND BEFORE line 779 (`# FEDERATED-EVAL-ONLY final metrics` comment / line 783 federated-eval print). The block REPLACES the line-796 silent `eval_metrics_history[final_round_for_metrics]` lookup. Insert:

```python
    # =========================================================================
    # D-06: extra eval round on the restored best-round state. All nodes
    # broadcast (no sampling — reproducibility > latency). Result becomes the
    # canonical `final_metrics["best"]` block, REPLACING the line-796 silent
    # eval_metrics_history[best_round_num] lookup that D-06 forbids.
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
                f"{best_round_metrics.get('sampled_ndcg@10')}"
            )
        else:
            print("[D-06] WARNING: no extra-eval responses; best block falls back to in-loop value.")
```

The block uses `Dict`, `Any`, `List`, `Tuple`, `ClientProxy`, `EvaluateRes`, `Status`, `Code`, `MetricRecord`, `RecordDict`, `ConfigRecord`, `DummyClientProxy` — verify all are already imported at the top of server_app.py for the in-loop eval; add any missing.

**Edit 4: Restructure final_metrics.** REPLACE lines 786-799 (current `final_round_for_metrics` + `final_metrics = dict(eval_metrics_history.get(final_round_for_metrics, {}))` block — the D-06 bug). New body:

```python
    # =========================================================================
    # FEDERATED-EVAL-ONLY final metrics (split learning cannot run centralized
    # eval — the server never sees the LOCAL user rows).
    # =========================================================================
    print("\n📊 Building canonical final_metrics block (D-07 nested schema)...")

    # Pitfall 9: last_round derives from max-key of eval_metrics_history (NOT
    # actual_rounds), guarding against early-stopping edge cases.
    if eval_metrics_history:
        last_round = max(eval_metrics_history.keys())
        last_block = dict(eval_metrics_history[last_round])
    else:
        last_round = 0
        last_block = {}

    # D-07: nested {best, last, best_round, last_round, final_eval_round_index}.
    # `best` comes from the D-06 extra-eval-round if checkpoint_rule restored;
    # otherwise collapses to last (cross-silo last_round modes).
    final_metrics = {
        "best": best_round_metrics or last_block,  # collapse for last_round modes
        "last": last_block,
        "best_round": best_round_num if best_round_num > 0 else last_round,
        "last_round": last_round,
        "final_eval_round_index": final_eval_round_index,
    }

    print_evaluation_metrics(
        final_metrics["best_round"],
        final_metrics["best"],
        context,
    )
```

**Edit 5: Migrate W&B summary keys.** REPLACE the loops at lines 805-814. New body:

```python
    if wandb_enabled and wandb_run is not None:
        final_log = {"round": actual_rounds + 1}
        for key, value in final_metrics["best"].items():
            if isinstance(value, (int, float)):
                final_log[f"final_eval/best/{key}"] = value
        wandb.log(final_log, step=actual_rounds + 1)

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

**Edit 6: Mutate manifest with new fields.** AFTER `manifest = build_run_manifest(...)` (~line 878-890) and BEFORE `embed_manifest_in_result(manifest, results_data)` (~line 893), insert:

```python
    manifest = dataclass_replace(
        manifest,
        final_eval_round_index=final_eval_round_index,
        metrics=results_data["final_metrics"],
    )
```

**Edit 7: Replace results-dir + filename + manifest-sibling.** REPLACE lines 898-906. New body with mode-conditional cross-silo coexistence (Pitfall 8):

```python
    # =========================================================================
    # Phase 6 D-01/D-02: per-module per-run directory layout for cross-device.
    # Cross-silo legacy mode keeps the flat <run_id>_results.json layout (D-03).
    # =========================================================================
    if mode in ("benchmark_cross_device", "paper_compat_pfedrec"):
        run_dir = module_run_results_dir(_MODULE, run_id)
        results_filename = run_dir / "results.json"  # D-04 clean filename
        sibling_kwarg = {"sibling_name": "manifest.json"}
    else:  # cross_silo_legacy — preserved per D-03
        legacy_dir = repo_root() / "results" / "federated"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        results_filename = legacy_dir / f"{run_id}_results.json"
        sibling_kwarg = {}

    atomic_write_json(str(results_filename), results_data)
    sibling_path = write_manifest_sibling(manifest, results_filename, **sibling_kwarg)
    print(f"Results saved to: {results_filename.resolve()}")
    print(f"Manifest sibling: {sibling_path.resolve()}")
```

The legacy `with open(results_filename, 'w') as f: json.dump(...)` is replaced by `atomic_write_json` per RESEARCH §Anti-Patterns.

**Edit 8: Update test_personalized_determinism.py path probe.** Locate the `_RESULTS_DIR` constant (likely around lines 1-50). Update the glob pattern from the legacy flat layout to the new per-run-dir layout. Find the existing definition (something like `_RESULTS_DIR = _REPO_ROOT / "results" / "federated"`) and the glob (something like `_RESULTS_DIR.glob("*_results.json")` or `_RESULTS_DIR.rglob("*personalized*results.json")`). REPLACE the glob with `_RESULTS_DIR.glob("personalized/*/results.json")` (or, if `_RESULTS_DIR` is already personalized-scoped via a Phase 3 update, use `_RESULTS_DIR.glob("*/results.json")`).

The other invariants (selected_clients_per_round byte-identity, partition_*.pt torch.equal) MUST remain untouched — only the path probe pattern changes.

**Edit 9: Extend test_server_integration.py with 4 NEW tests.** Read the existing file shape; reuse its mocking style (likely `unittest.mock.patch` + fake `Grid` instances). Add Tests 1-4 from the behavior block.

**Verify by running:**

```bash
cd federated-personalized-cf && pytest tests/test_server_integration.py -x -v -k "results_path or extra_eval or best_last_blocks or per_group_exposure"
cd scripts/foundation && pytest tests/test_personalized_determinism.py -x -v -m slow
```

Integration tests MUST pass. Slow test passes once the subprocess actually runs (manual gate post-CI).
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-personalized-cf && pytest tests/test_server_integration.py -x -v -k "results_path or extra_eval or best_last_blocks or per_group_exposure"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "from fedrec_foundation.paths import module_run_results_dir" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1
    - `grep -c "from dataclasses import replace as dataclass_replace" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1
    - `grep -c "_MODULE: str = .personalized." federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1
    - `grep -c "Path(.\\.\\./results/federated.)" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 0 (D-02 hard cutover)
    - `grep -c "module_run_results_dir(_MODULE, run_id)" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1
    - `grep -c "final_eval_round_index" federated-personalized-cf/federated_personalized_cf/server_app.py` returns at least 5
    - `grep -c "best_round_metrics" federated-personalized-cf/federated_personalized_cf/server_app.py` returns at least 4
    - `grep -c "wandb.run.summary\\[f.final/" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 0
    - `grep -c "wandb.run.summary\\[f.best/" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1
    - `grep -c "wandb.run.summary\\[f.last/" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1
    - `grep -c "max(eval_metrics_history.keys())" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1 (Pitfall 9)
    - `grep -c "if mode in (.benchmark_cross_device., .paper_compat_pfedrec.)" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1 (Pitfall 8)
    - `grep -c "sibling_name=.manifest.json." federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1
    - `grep -c "atomic_write_json" federated-personalized-cf/federated_personalized_cf/server_app.py` returns at least 1
    - `grep -c "final_metrics = dict(eval_metrics_history.get(final_round_for_metrics" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 0 (the D-06-forbidden lookup is removed)
    - `grep -E "personalized/\\*/results\\.json|.\\*/results\\.json" scripts/foundation/tests/test_personalized_determinism.py` returns at least 1 line (path probe migrated)
    - `cd federated-personalized-cf && pytest tests/test_server_integration.py -x -v -k "results_path or extra_eval or best_last_blocks or per_group_exposure"` exits 0
    - `cd federated-personalized-cf && pytest tests/ -q -m "not slow"` exits 0 (no regressions)
  </acceptance_criteria>
  <done>
    - server_app.py: extra-eval-round block inserted after best-arrays restore (line 775); flat final_metrics replaced with nested {best, last, best_round, last_round, final_eval_round_index} — D-06 forbidden lookup at line 796 removed; W&B summary final/* -> best/* + last/*; manifest mutated via dataclasses.replace; results path resolves via module_run_results_dir for cross-device; cross-silo legacy path preserved (D-03 + Pitfall 8); atomic_write_json replaces json.dump
    - test_server_integration.py: 4 NEW tests pinning EVL-01/02/03/04/06 (results path, extra-eval-round REPLACES history lookup, best/last block schema, per-group exposure history)
    - test_personalized_determinism.py: _RESULTS_DIR glob pattern updated to per-run-dir layout (`personalized/*/results.json` or equivalent); other invariants unchanged
    - Existing personalized tests remain GREEN; no regressions
  </done>
</task>

</tasks>

<verification>
- Imports clean: `python -c "from federated_personalized_cf import server_app; print('ok')"` exits 0
- Path migration: `grep -c "Path(.\\.\\./results/federated.)" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 0
- D-06-forbidden lookup removed: `grep -c "final_metrics = dict(eval_metrics_history.get(final_round_for_metrics" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 0
- 4 NEW integration tests pass
- Path probe migrated: `grep -E "personalized/.+/results\\.json" scripts/foundation/tests/test_personalized_determinism.py` matches at least 1 line
- D-18 surgical scope: `git diff --stat` shows ONLY changes to server_app.py + test_server_integration.py + test_personalized_determinism.py
- Wave-2 file-disjointness held: this plan modifies ONLY federated-personalized-cf/ + ONE foundation test file; Plans 03 (baseline) and 05 (adaptive) own their dirs
</verification>

<success_criteria>
- federated-personalized-cf/federated_personalized_cf/server_app.py: extra-eval-round block (D-06) wired against `partition_to_node_id.values()`; flat final_metrics replaced by nested schema (D-07); D-06-forbidden eval_metrics_history lookup at line 796 explicitly removed; module_run_results_dir replaces Path("../results/federated"); cross-silo legacy path preserved (D-03 + Pitfall 8); W&B summary uses best/* and last/*; manifest carries final_eval_round_index + metrics; D-04 clean filename via sibling_name="manifest.json"; atomic_write_json replaces json.dump
- 4 NEW integration tests in test_server_integration.py
- Updated path probe in scripts/foundation/tests/test_personalized_determinism.py preserves byte-identity invariants while finding files in new layout
- Pitfall 9 closure: last_round = max(eval_metrics_history.keys())
- Files outside listed `files_modified` remain byte-identical to pre-Plan-04 state (D-18 surgical)
</success_criteria>

<output>
After completion, create `.planning/phases/06-evaluation-reporting-harness/06-evaluation-reporting-harness-04-SUMMARY.md` covering:
- Path migration to per-run-dir layout (D-02)
- Extra-eval-round wiring (D-06) — explicit replacement of the line-796 silent eval_metrics_history lookup
- Nested final_metrics schema (D-07)
- W&B summary key migration (best/* + last/*; final/* removed)
- Cross-silo coexistence (D-03 + Pitfall 8)
- Path probe update in scripts/foundation/tests/test_personalized_determinism.py
- 4 NEW integration tests
</output>
</content>
</invoke>