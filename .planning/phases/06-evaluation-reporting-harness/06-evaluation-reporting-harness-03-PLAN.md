---
phase: 06-evaluation-reporting-harness
plan: 03
type: execute
wave: 2
depends_on:
  - 06-evaluation-reporting-harness-01
  - 06-evaluation-reporting-harness-02
files_modified:
  - federated-baseline-cf/federated_baseline_cf/server_app.py
  - federated-baseline-cf/tests/test_server_integration.py
  - scripts/foundation/tests/test_baseline_subprocess_determinism.py
autonomous: true
requirements: [EVL-01, EVL-02, EVL-03, EVL-04, EVL-06]
must_haves:
  truths:
    - "Baseline server_app writes results.json + manifest.json to <repo>/results/federated/baseline/<run_id>/ via module_run_results_dir (D-01 + D-02 — closes folded phase2-baseline-determinism-path-bug.md)"
    - "When checkpoint_rule in {best_round_restore, best_round} and best_round_num > 0, a post-restore broadcast eval round runs against ALL nodes in partition_to_node_id (D-06 — no sampling)"
    - "final_metrics is now nested {best, last, best_round, last_round, final_eval_round_index} with `best` populated from the extra-eval-round and `last` from max-key of eval_metrics_history (Pitfall 9)"
    - "Cross-silo (mode != benchmark_cross_device, checkpoint_rule == last_round) writes legacy flat <repo>/results/federated/<run_id>_results.json layout — D-03 coexistence (Pitfall 8)"
    - "W&B run.summary uses best/* and last/* namespaces (final/* removed); per-round eval_metrics_history continues to log under eval/* prefix"
    - "Manifest carries final_eval_round_index (>=1 for best_round_restore, 0 for last_round) AND nested metrics block via dataclasses.replace post-build mutation"
    - "D-04 sibling filename: write_manifest_sibling(..., sibling_name='manifest.json') for cross-device runs; legacy <run_id>-manifest.json kept for cross-silo"
  artifacts:
    - path: "federated-baseline-cf/federated_baseline_cf/server_app.py"
      provides: "Baseline server_app with extra-eval-round + per-run-dir + nested final_metrics + best/last W&B namespaces"
      contains: "from fedrec_foundation.paths import module_run_results_dir"
    - path: "federated-baseline-cf/tests/test_server_integration.py"
      provides: "4 NEW assertions: results path repo-root anchored, extra-eval-round wired, best/last block schema, per-round exposure history"
      contains: "def test_results_path_repo_root_anchored"
    - path: "scripts/foundation/tests/test_baseline_subprocess_determinism.py"
      provides: "Re-enabled phase2-path-bug regression guard (selected_clients_per_round byte-identity across subprocess reruns)"
      contains: "test_selected_partitions_byte_identical_across_subprocess_reruns"
  key_links:
    - from: "federated-baseline-cf/federated_baseline_cf/server_app.py::run_dir"
      to: "scripts/foundation/fedrec_foundation/paths.py::module_run_results_dir"
      via: "run_dir = module_run_results_dir(_MODULE='baseline', run_id=run_id)"
      pattern: "module_run_results_dir\\(_MODULE, run_id\\)|module_run_results_dir\\(.baseline.,"
    - from: "federated-baseline-cf/federated_baseline_cf/server_app.py::extra_eval_round"
      to: "federated-baseline-cf/federated_baseline_cf/strategy.py::aggregate_evaluate"
      via: "strategy.aggregate_evaluate(final_eval_round_index, extra_results, [])"
      pattern: "strategy\\.aggregate_evaluate\\(final_eval_round_index"
    - from: "federated-baseline-cf/federated_baseline_cf/server_app.py::manifest"
      to: "scripts/foundation/fedrec_foundation/manifest.py::RunManifest.metrics field (schema v2)"
      via: "manifest = replace(manifest, final_eval_round_index=N, metrics=results_data['final_metrics'])"
      pattern: "replace\\(manifest, final_eval_round_index="
---

<objective>
Wire the foundation primitives from Plans 01+02 into `federated-baseline-cf/federated_baseline_cf/server_app.py`. This closes EVL-01 / EVL-02 / EVL-03 / EVL-04 / EVL-06 for the baseline module AND resolves the folded `phase2-baseline-determinism-path-bug.md` todo (the baseline subprocess determinism regression guard re-enables once the path is repo-root anchored).

Purpose:
  - Replace `Path("../results/federated")` (server_app.py:788) with `module_run_results_dir("baseline", run_id)` per D-02 — closes Pitfall 2 (the folded path bug). Cross-silo legacy paths preserved via mode branch (Pitfall 8).
  - Insert the D-06 extra-eval-round block after `arrays = best_arrays` (server_app.py:626) and BEFORE the existing centralized eval (server_app.py:633). The federated extra-eval populates `best_round_metrics`; the existing centralized eval (RMSE/MAE/ranking_metrics) continues to compute and feeds the `last`-block diagnostics per Pitfall 10.
  - Restructure `final_metrics` from flat dict (line 703) to nested `{best, last, best_round, last_round, final_eval_round_index}` per D-07.
  - Migrate W&B summary keys from `final/*` (line 723) to `best/*` and `last/*` namespaces.
  - Mutate manifest via `dataclasses.replace(manifest, final_eval_round_index=N, metrics=...)` after `build_run_manifest` and BEFORE `embed_manifest_in_result`.
  - Use `write_manifest_sibling(..., sibling_name="manifest.json")` for cross-device per D-04 clean filename.
  - Re-enable the baseline subprocess determinism regression guard at `scripts/foundation/tests/test_baseline_subprocess_determinism.py` — once D-02 lands, the test's `_RESULTS_DIR` glob finds the new per-run dir layout.

Output:
  - `federated-baseline-cf/federated_baseline_cf/server_app.py` modified: Path("../results/federated") replaced; extra-eval-round block inserted; final_metrics nested; W&B summary keys migrated.
  - `federated-baseline-cf/tests/test_server_integration.py` extended: 4 NEW assertions pinning EVL-01/02/03/04/06 invariants.
  - `scripts/foundation/tests/test_baseline_subprocess_determinism.py` (NEW): re-enabled regression guard for the folded path-bug todo.
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
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-baseline-cf/federated_baseline_cf/server_app.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-baseline-cf/federated_baseline_cf/strategy.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-baseline-cf/tests/test_server_integration.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/scripts/foundation/tests/test_personalized_determinism.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/02-baseline-migration/02-baseline-migration-04-SUMMARY.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/02-baseline-migration/02-baseline-migration-05-SUMMARY.md

<interfaces>
<!-- Foundation helper from Plan 01 (this plan's Wave 1 dep) -->
```python
from fedrec_foundation.paths import module_run_results_dir
# module_run_results_dir(module: str, run_id: str) -> Path
# Whitelist: {"baseline", "personalized", "adaptive", "pfedrec"}
# Returns <repo>/results/federated/<module>/<run_id>/ (created if missing)
```

<!-- Manifest schema v2 from Plan 02 (this plan's Wave 1 dep) -->
```python
from fedrec_foundation.manifest import (
    build_run_manifest,
    embed_manifest_in_result,
    write_manifest_sibling,  # now accepts sibling_name kwarg
    RUN_MANIFEST_SCHEMA_VERSION,  # == 2
)
# RunManifest dataclass now carries:
#   final_eval_round_index: int = 0   (sentinel: 0 = no extra eval ran)
#   metrics: Dict[str, Any] = field(default_factory=dict)
# Use dataclasses.replace(manifest, ...) for post-build mutation.
```

<!-- Existing baseline server_app.py drop sites (FROM RESEARCH §Pattern 2 + §Code Examples Example 3-4-5) -->
```python
# Line 626: arrays = best_arrays  (D-27 best-round restore — keep verbatim)
# Lines 632-700: centralized eval block (RMSE/MAE/ranking_metrics — KEEP, feeds `last` diagnostics)
# Line 703: final_metrics = {...}  (REPLACE — flat -> nested {best, last, ...})
# Line 723: wandb.run.summary[f"final/{key}"]  (MIGRATE -> best/* and last/*)
# Line 788: results_dir = Path("../results/federated")  (REPLACE -> module_run_results_dir)
# Line 792: results_filename = results_dir / f"{run_id}_results.json"  (CHANGE for cross-device -> results.json)
# Line 797: write_manifest_sibling(manifest, results_filename)  (ADD sibling_name="manifest.json" for cross-device)
```

<!-- Existing baseline strategy.py — DO NOT MODIFY in this plan -->
```python
# federated-baseline-cf/federated_baseline_cf/strategy.py
# Already emits sufficient stats via _sum_sufficient_stats + _sufficient_stats_to_thesis_metrics
# Per-group keys: sampled_hr@10/{sparse,medium,dense}, sampled_ndcg@10/{sparse,medium,dense},
#                 evaluated_users_{sparse,medium,dense}
# strategy.aggregate_evaluate(round_num, results, failures) returns (loss, thesis_dict)
# Phase 6 reuses this verbatim — no strategy.py touches.
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Wire module_run_results_dir + extra-eval-round + nested final_metrics + best/last W&B namespaces into federated-baseline-cf/federated_baseline_cf/server_app.py; ship 4 NEW integration tests in test_server_integration.py</name>
  <files>federated-baseline-cf/federated_baseline_cf/server_app.py, federated-baseline-cf/tests/test_server_integration.py</files>
  <read_first>
    - federated-baseline-cf/federated_baseline_cf/server_app.py — current state (CRITICAL line refs: 36-39 imports, 346 partition_to_node_id, 400+574+626 best_arrays, 500-547 in-loop eval pattern, 569 D-27 checkpoint_rule branch, 632-700 centralized eval, 703 final_metrics flat, 715-723 W&B final/* loop, 788-797 results_dir + write_manifest_sibling)
    - federated-baseline-cf/federated_baseline_cf/strategy.py — _sum_sufficient_stats / _sufficient_stats_to_thesis_metrics / aggregate_evaluate (sufficient-stat path that the extra-eval-round invokes; DO NOT TOUCH this file in this plan)
    - federated-baseline-cf/tests/test_server_integration.py — current shape (extend, do not rewrite)
    - scripts/foundation/fedrec_foundation/paths.py — module_run_results_dir (Plan 01 output)
    - scripts/foundation/fedrec_foundation/manifest.py — schema v2 RunManifest (Plan 02 output)
    - .planning/phases/06-evaluation-reporting-harness/06-CONTEXT.md §decisions D-01, D-02, D-03, D-04, D-06, D-07, D-08, D-09
    - .planning/phases/06-evaluation-reporting-harness/06-RESEARCH.md §Pattern 1 (paths) + §Pattern 2 (extra-eval-round) + §Pattern 3 (nested final_metrics) + §Code Examples Example 3 (per-run dir) + Example 4 (extra-eval-round) + Example 5 (W&B summary) + §Common Pitfalls Pitfall 8 (cross-silo coexistence) + Pitfall 9 (last_round max-key) + Pitfall 10 (centralized eval feeds last)
  </read_first>
  <behavior>
    - Test 1 (test_results_path_repo_root_anchored): Mock-mode test (or simulate via the existing test_server_integration.py harness — read the file to see what mocking style it uses; baseline already has `test_server_integration.py`). Construct a fake server_app run that exercises the result-write path; assert that the path returned by `module_run_results_dir("baseline", run_id)` resolves to `<repo>/results/federated/baseline/<run_id>/` (call `repo_root()` from `fedrec_foundation.paths` for ground truth); assert `results.json` exists at `run_dir / "results.json"`; assert `manifest.json` exists at `run_dir / "manifest.json"` (D-04 clean filenames).
    - Test 2 (test_extra_eval_round_after_best_arrays_restore): Build a fake `eval_metrics_history` dict with rounds {1: ndcg=0.40, 2: ndcg=0.45, 3: ndcg=0.42} so `best_round_num=2` and `last_round=3`. Mock `grid.send_and_receive` to return synthetic EvaluateRes per node carrying `hit_count_overall_at10=1`, `ndcg_sum_overall_at10=0.5`, `evaluated_users=1`, etc. (~2 fake nodes is enough). Run the post-loop block; assert `final_eval_round_index == actual_rounds + 1` (i.e. 4 in this fixture); assert `best_round_metrics` is non-empty AND `best_round_metrics.get("sampled_ndcg@10") == 0.5` (from the synthetic input); assert the call site explicitly invokes `strategy.aggregate_evaluate(final_eval_round_index, extra_results, [])`.
    - Test 3 (test_canonical_artifact_carries_best_and_last_blocks): Run the same fake harness; load the written `results.json`; assert `results["final_metrics"]` is a dict with keys `{"best", "last", "best_round", "last_round", "final_eval_round_index"}` (set equality on the top-level keys). Assert `results["final_metrics"]["best"]` and `["last"]` are both dicts. Assert `results["final_metrics"]["best_round"] == 2` and `["last_round"] == 3` and `["final_eval_round_index"] == 4`. Assert `results["_manifest"]["schema_version"] == 2`. Assert `results["_manifest"]["final_eval_round_index"] == 4`. Assert `results["_manifest"]["metrics"] == results["final_metrics"]` (dataclasses.replace post-build mutation copy-through).
    - Test 4 (test_round_metrics_history_carries_per_group_exposure): Construct fake `eval_metrics_history` rounds where each carries `evaluated_users_sparse`, `evaluated_users_medium`, `evaluated_users_dense` keys (D-09 per-round exposure counts). Assert that the round-by-round `wandb.log` would emit `eval/evaluated_users_sparse`, `eval/evaluated_users_medium`, `eval/evaluated_users_dense` keys per round (test by reading the eval_metrics_history dict written into `results.json` — assert at least one round has all three keys).
  </behavior>
  <action>
Edit `federated-baseline-cf/federated_baseline_cf/server_app.py` with surgical edits. **PRESERVE the existing centralized eval block (lines 632-700) verbatim** — its outputs (`rmse`, `mae`, `ranking_metrics`, `sampled_metrics`) feed the `last`-block diagnostics per Pitfall 10. Phase 6 only changes how results are PACKAGED, not how they are COMPUTED.

**Edit 1: Add foundation imports.** Locate the existing `from fedrec_foundation.manifest import (...)` block around line 35-40. Add a new import line:

```python
from fedrec_foundation.paths import module_run_results_dir
from dataclasses import replace as dataclass_replace
```

**Edit 2: Extract module-name local constant.** Inside `@app.main()` near the top of the function (right after `run_id = generate_run_id()`), add:

```python
_MODULE: str = "baseline"   # cross-references: build_run_manifest, module_run_results_dir, default W&B project switch
```

**Edit 3: Insert the D-06 extra-eval-round block.** AFTER line 626 (`arrays = best_arrays`) AND BEFORE line 630 (`# CENTRALIZED EVALUATION...` block). The centralized eval (rating prediction RMSE/MAE) STAYS — it feeds the `last`-block diagnostics per Pitfall 10. Insert this block:

```python
    # =========================================================================
    # D-06: extra eval round on the restored best-round state.
    # All 6040 nodes (no sampling — reproducibility > latency per CONTEXT.md).
    # Result becomes the canonical `final_metrics["best"]` block.
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
            # MAJOR fix (plan-checker iteration 1): coerce np.float64 values to
            # Python floats at the assignment site so downstream dataclass_replace
            # (Edit 6) + atomic_write_json (Edit 7) never see np.float64. Without
            # this, json.dumps raises TypeError: Object of type float64 is not
            # JSON serializable on the manifest's metrics field.
            best_round_metrics = {
                k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                for k, v in (thesis or {}).items()
            }
            print(
                f"[D-06] Extra eval complete. Canonical best/sampled_ndcg@10="
                f"{best_round_metrics.get('sampled_ndcg@10')}"
            )
        else:
            print("[D-06] WARNING: no extra-eval responses; best_round_metrics will fall back to in-loop value.")
```

The above block uses `Dict`, `Any`, `List`, `Tuple`, `ClientProxy`, `EvaluateRes`, `Status`, `Code`, `MetricRecord`, `RecordDict`, `ConfigRecord`, `DummyClientProxy` — these are already imported at the top of server_app.py for the in-loop eval. Confirm by reading lines 1-50 before editing; if any are missing, add them to existing imports.

**Edit 4: Restructure final_metrics from flat to nested.** REPLACE the existing block at lines 703-708 (currently `final_metrics = {"eval_loss": ..., **rating_metrics, **ranking_metrics, **sampled_metrics}`) with the nested layout. The existing centralized eval outputs feed `last`-block diagnostics per Pitfall 10:

```python
    # Diagnostic: the existing centralized eval results (rating prediction +
    # ranking metrics) become the `last` block per Pitfall 10. They are NOT
    # the canonical thesis metric — those come from the federated extra-eval-
    # round per D-06.
    centralized_diag = {
        "eval_loss": float(eval_loss),
        **rating_metrics,
        **ranking_metrics,
        **sampled_metrics,
    }

    # D-07: nested {best, last, best_round, last_round, final_eval_round_index}.
    # Pitfall 9: last_round derives from max-key of eval_metrics_history (NOT
    # actual_rounds), guarding against early-stopping edge cases.
    if eval_metrics_history:
        last_round = max(eval_metrics_history.keys())
        last_block_federated = dict(eval_metrics_history[last_round])
    else:
        last_round = 0
        last_block_federated = {}

    # The `last` block carries BOTH the federated last-round sufficient-stat
    # metrics (per-group HR/NDCG, exposure counts) AND the centralized rating-
    # prediction diagnostics. The `best` block is federated-only.
    final_metrics = {
        "best": best_round_metrics or last_block_federated,  # collapse to last for last_round modes
        "last": {**last_block_federated, **centralized_diag},
        "best_round": best_round_num if best_round_num > 0 else last_round,
        "last_round": last_round,
        "final_eval_round_index": final_eval_round_index,
    }

    print_evaluation_metrics(actual_rounds, final_metrics["best"], context)
```

**Edit 5: Migrate W&B summary keys.** REPLACE the loop at lines 715-723 (currently `for key, value in final_metrics.items(): if isinstance(value, (int, float)): wandb.run.summary[f"final/{key}"] = value`). New body:

```python
    # Log final metrics to wandb under best/* and last/* namespaces (D-07).
    # The legacy final/* namespace is deprecated; sweep.yaml metric.name MUST
    # migrate to best/sampled_ndcg@10 (Pitfall 7 — see Plan 07).
    if wandb_enabled:
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

**Edit 6: Mutate manifest with new fields.** AFTER `manifest = build_run_manifest(...)` (~line 761) and BEFORE `embed_manifest_in_result(manifest, results_data)` (~line 784), insert:

```python
    # Phase 6: post-build mutation to populate schema-v2 fields (final_eval_round_index, metrics).
    manifest = dataclass_replace(
        manifest,
        final_eval_round_index=final_eval_round_index,
        metrics=results_data["final_metrics"],
    )
```

**Edit 7: Replace results-dir + filename + manifest-sibling.** REPLACE lines 786-797 (currently `results_dir = Path("../results/federated")` ... `sibling_path = write_manifest_sibling(manifest, results_filename)`). New body with mode-conditional branching for D-03 cross-silo coexistence (Pitfall 8):

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
        sibling_kwarg = {}  # default <run_id>-manifest.json

    atomic_write_json(str(results_filename), results_data)
    sibling_path = write_manifest_sibling(manifest, results_filename, **sibling_kwarg)
    print(f"Results saved to: {results_filename.resolve()}")
    print(f"Manifest sibling: {sibling_path.resolve()}")
```

The above uses `repo_root` and `atomic_write_json` from `fedrec_foundation.paths` and `fedrec_foundation.atomic`. Add these imports at the top of server_app.py:

```python
from fedrec_foundation.paths import module_run_results_dir, repo_root
from fedrec_foundation.atomic import atomic_write_json
```

The legacy `with open(results_filename, 'w') as f: json.dump(results_data, f, indent=4, default=str)` is removed in favor of `atomic_write_json` (per RESEARCH §Anti-Patterns: "Don't write the result JSON via `json.dump` directly").

**Edit 8: Re-enable the baseline subprocess determinism regression guard.** Create `scripts/foundation/tests/test_baseline_subprocess_determinism.py` modeled after the existing `scripts/foundation/tests/test_personalized_determinism.py`. The test runs `scripts/run.py baseline benchmark_cross_device` twice with the same seed, then asserts byte-identity of `selected_clients_per_round` AND that the result file lands at `<repo>/results/federated/baseline/<run_id>/results.json`. Mark with `@pytest.mark.slow` and the `FEDREC_SKIP_SLOW=1` escape hatch idiom (read `test_personalized_determinism.py` for the exact convention).

The test MUST contain:
- An import `from fedrec_foundation.paths import repo_root`
- A `_RESULTS_DIR = repo_root() / "results" / "federated" / "baseline"` glob root
- A `_RESULTS_DIR.glob("*/results.json")` probe (NOT the legacy `*_results.json` flat pattern — Phase 6 layout per D-01)
- A test function `test_selected_partitions_byte_identical_across_subprocess_reruns` that runs two same-seed subprocess invocations of `python scripts/run.py baseline benchmark_cross_device --run-config "run-seed=42 num-server-rounds=2 fraction-train=0.001"` and asserts `result_a["selected_clients_per_round"] == result_b["selected_clients_per_round"]` (Pitfall 5 — extra-eval-round does NOT sample, so byte-identity invariant survives)
- A coverage guard: `pytest.skip` if no result files are found (cold-run sanity)

**Edit 9: Extend test_server_integration.py with 4 NEW tests** (Test 1-4 from the behavior block above). Read the existing test_server_integration.py to identify its test fixture style (likely uses `unittest.mock.patch` or builds fake `Grid`/`Message` objects). Reuse that style verbatim — do NOT introduce new mocking infrastructure.

**Edit 10 (BLOCKER fix from plan-checker iteration 1): Migrate the in-tree slow test in `federated-baseline-cf/tests/test_server_integration.py`.**

The existing slow test `test_selected_partitions_byte_identical_across_subprocess_reruns` (lines 205-276) was authored against the pre-Phase-6 schema/path conventions and silently breaks once Plans 02+03 land. Two surfaces must change:

1. **Path probe (line 227 + 256)**: `results_dir.glob("*_results.json")` finds files at the legacy flat layout `<repo>/results/federated/<run_id>_results.json`. Phase 6 cross-device runs land at `<repo>/results/federated/baseline/<run_id>/results.json`. Replace BOTH glob calls with `results_dir.glob("baseline/*/results.json")` (the new per-run-dir layout — no other glob pattern would catch only Phase-6 baseline writes).
2. **Schema probe (lines 272-273)**: `a["final_metrics"].get("sampled_ndcg@10")` reads from the OLD flat schema. After Plan 03 Edit 4 lands, `sampled_ndcg@10` lives at `final_metrics["best"]["sampled_ndcg@10"]`. Replace BOTH lookups (`ndcg_a` and `ndcg_b`) with `a["final_metrics"]["best"].get("sampled_ndcg@10", 0.0)` and the same for `b`.

Concrete edits to apply (find-and-replace):

```python
# Line 227 (current):
    before = set(results_dir.glob("*_results.json"))
# Replace with:
    before = set(results_dir.glob("baseline/*/results.json"))

# Line 256 (current):
    after = sorted((results_dir.glob("*_results.json")), key=lambda p: p.stat().st_mtime)
# Replace with:
    after = sorted((results_dir.glob("baseline/*/results.json")), key=lambda p: p.stat().st_mtime)

# Lines 272-273 (current):
    ndcg_a = float(a["final_metrics"].get("sampled_ndcg@10", 0.0))
    ndcg_b = float(b["final_metrics"].get("sampled_ndcg@10", 0.0))
# Replace with:
    ndcg_a = float(a["final_metrics"]["best"].get("sampled_ndcg@10", 0.0))
    ndcg_b = float(b["final_metrics"]["best"].get("sampled_ndcg@10", 0.0))
```

The rest of the test body (subprocess invocation, byte-identity assertion on `selected_clients_per_round`) is preserved verbatim — those invariants survive Phase 6 unchanged. The `results_dir = repo_root / "results" / "federated"` line at 222 stays as the glob root; only the glob pattern changes (so the path stays anchored at `<repo>/results/federated/`, with the per-module subdir baked into the glob).

**Verify by running:**

```bash
cd federated-baseline-cf && pytest tests/test_server_integration.py -x -v -k "results_path or extra_eval or best_last_blocks or per_group_exposure"
cd scripts/foundation && pytest tests/test_baseline_subprocess_determinism.py -x -v -m slow
```

The integration tests MUST pass. The slow test will pass once the subprocess actually runs (may require manual execution after CI gate).
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-baseline-cf && pytest tests/test_server_integration.py -x -v -k "results_path or extra_eval or best_last_blocks or per_group_exposure"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "from fedrec_foundation.paths import module_run_results_dir" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 1
    - `grep -c "from dataclasses import replace as dataclass_replace" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 1
    - `grep -c "_MODULE: str = .baseline." federated-baseline-cf/federated_baseline_cf/server_app.py` returns 1
    - `grep -c "Path(.\\.\\./results/federated.)" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 0 (D-02 — old path holdover removed)
    - `grep -c "module_run_results_dir(_MODULE, run_id)" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 1
    - `grep -c "final_eval_round_index" federated-baseline-cf/federated_baseline_cf/server_app.py` returns at least 5 (init + assignment + manifest mutation + final_metrics dict + W&B summary)
    - `grep -c "best_round_metrics" federated-baseline-cf/federated_baseline_cf/server_app.py` returns at least 4 (init + populate + collapse-to-last + final_metrics)
    - `grep -c "wandb.run.summary\\[f.final/" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 0 (legacy namespace removed)
    - `grep -c "wandb.run.summary\\[f.best/" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 1
    - `grep -c "wandb.run.summary\\[f.last/" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 1
    - `grep -c "sibling_name=.manifest.json." federated-baseline-cf/federated_baseline_cf/server_app.py` returns 1
    - `grep -c "atomic_write_json" federated-baseline-cf/federated_baseline_cf/server_app.py` returns at least 1
    - `grep -c "max(eval_metrics_history.keys())" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 1 (Pitfall 9 closure)
    - `grep -c "if mode in (.benchmark_cross_device., .paper_compat_pfedrec.)" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 1 (Pitfall 8 cross-silo branch)
    - `test -f scripts/foundation/tests/test_baseline_subprocess_determinism.py` exits 0
    - `grep -c "test_selected_partitions_byte_identical_across_subprocess_reruns" scripts/foundation/tests/test_baseline_subprocess_determinism.py` returns 1
    - `grep -c "@pytest.mark.slow" scripts/foundation/tests/test_baseline_subprocess_determinism.py` returns at least 1
    - `cd federated-baseline-cf && pytest tests/test_server_integration.py -x -v -k "results_path or extra_eval or best_last_blocks or per_group_exposure"` exits 0 with all 4 NEW tests passing
    - `cd federated-baseline-cf && pytest tests/ -q -m "not slow"` exits 0 (no regressions in existing tests)
    - **BLOCKER (in-tree slow test migration, plan-checker iteration 1):** `grep -c 'final_metrics\["best"\]' federated-baseline-cf/tests/test_server_integration.py` returns at least 1 (the two `ndcg_a`/`ndcg_b` lookups now read from the nested-best block)
    - **BLOCKER (in-tree slow test migration, plan-checker iteration 1):** `grep -c 'glob("\*_results.json")' federated-baseline-cf/tests/test_server_integration.py` returns 0 (the two legacy flat-layout globs at lines 227+256 are removed)
    - **BLOCKER (in-tree slow test migration, plan-checker iteration 1):** `grep -c 'glob("baseline/\*/results.json")' federated-baseline-cf/tests/test_server_integration.py` returns 2 (both globs migrated to the new per-run-dir layout)
    - **MAJOR (np.float64 JSON-serialization, plan-checker iteration 1):** Plan adopts path (b): `final_metrics["best"]` and `final_metrics["last"]` blocks build their numeric values via explicit `float(...)` cast at the assignment site, so passing them through `dataclass_replace(manifest, metrics=results_data["final_metrics"])` followed by `atomic_write_json` cannot raise `TypeError: Object of type float64 is not JSON serializable`. Concretely: in Edit 3 (extra-eval-round block), wrap the `best_round_metrics = dict(thesis) if thesis else {}` line so that numeric values get coerced — change to `best_round_metrics = {k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v for k, v in (thesis or {}).items()}`. Acceptance: `grep -c 'float(v) if isinstance(v, (int, float))' federated-baseline-cf/federated_baseline_cf/server_app.py` returns at least 1
    - **MINOR (legacy json.dump removal, plan-checker iteration 1):** `grep -c "json.dump(results_data" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 0 (atomic_write_json fully replaces the legacy json.dump path)
    - **MINOR (edit-order ambiguity, plan-checker iteration 1):** `python -c "src=open('federated-baseline-cf/federated_baseline_cf/server_app.py').read(); idx_final=src.find('final_metrics = {'); idx_replace=src.find('dataclass_replace(manifest'); assert idx_final >= 0 and idx_replace > idx_final, f'final_metrics block must appear before dataclass_replace, got idx_final={idx_final} idx_replace={idx_replace}'"` exits 0 (proves Edit 4 lands BEFORE Edit 6 in source order — guards against an executor swapping their order and silently producing empty `metrics` field on the manifest)
    - **MINOR (print_evaluation_metrics call-site uniqueness, plan-checker iteration 1):** Verified via re-reading `federated-baseline-cf/federated_baseline_cf/server_app.py:632-700`: the existing centralized eval block does NOT call `print_evaluation_metrics` (the only legacy call lives at line 711 inside the flat-final_metrics block being REPLACED by Edit 4). After Edit 4, exactly one `print_evaluation_metrics(actual_rounds, final_metrics["best"], context)` call remains. Acceptance: `grep -c "print_evaluation_metrics(" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 1 (single call site after edits — no double-print)
  </acceptance_criteria>
  <done>
    - server_app.py: extra-eval-round block inserted between best-arrays restore (line 626) and centralized eval (line 632); flat final_metrics restructured to nested {best, last, best_round, last_round, final_eval_round_index}; W&B summary keys migrated final/* -> best/* + last/*; manifest mutated via dataclasses.replace; results path resolves via module_run_results_dir for cross-device; cross-silo legacy path preserved (D-03 + Pitfall 8); atomic_write_json replaces json.dump
    - test_server_integration.py: 4 NEW tests pin EVL-01/02/03/04/06 (results path, extra-eval-round wired, best/last block schema, per-group exposure history)
    - test_baseline_subprocess_determinism.py: NEW @pytest.mark.slow regression guard re-enables the folded phase2 path-bug todo (Pitfall 2 closure)
    - **BLOCKER closure (plan-checker iteration 1):** in-tree slow test `test_selected_partitions_byte_identical_across_subprocess_reruns` (federated-baseline-cf/tests/test_server_integration.py:205-276) migrated to the new per-run-dir glob (`baseline/*/results.json`) and the nested-best schema lookup (`final_metrics["best"]["sampled_ndcg@10"]`). Test now finds Phase-6 outputs and reads the canonical thesis metric correctly.
    - **MAJOR closure (np.float64 JSON-serialization, plan-checker iteration 1):** path (b) chosen — Edit 3 coerces `best_round_metrics` numeric values to Python primitives at the assignment site so downstream `dataclass_replace` + `atomic_write_json` never see `np.float64`. Documented in SUMMARY.
    - Existing baseline tests remain GREEN; no regressions
  </done>
</task>

</tasks>

<verification>
- Imports clean: `python -c "from federated_baseline_cf.server_app import app; print('ok')"` exits 0 (or, since this is a Flower @app, `python -c "from federated_baseline_cf import server_app; print('ok')"`)
- Path migration: `grep -c "Path(.\\.\\./results/federated.)" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 0 (D-02 hard cutover)
- Cross-silo coexistence: `grep -c "if mode in (.benchmark_cross_device., .paper_compat_pfedrec.)" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 1 (Pitfall 8 mode branch)
- 4 NEW integration tests pass: `cd federated-baseline-cf && pytest tests/test_server_integration.py -x -v -k "results_path or extra_eval or best_last_blocks or per_group_exposure"` exits 0
- Existing baseline tests stay green: `cd federated-baseline-cf && pytest tests/ -q -m "not slow"` exits 0
- D-18 surgical scope: `git diff --stat` shows ONLY changes to server_app.py + test_server_integration.py + new test_baseline_subprocess_determinism.py; pyproject.toml / dataset.py / client_app.py / task.py / strategy.py / models/ UNTOUCHED
- Wave-2 file-disjointness held: this plan modifies ONLY federated-baseline-cf/ + a new foundation test file; Plans 04 (personalized) and 05 (adaptive) own their respective module dirs
</verification>

<success_criteria>
- federated-baseline-cf/federated_baseline_cf/server_app.py: extra-eval-round block (D-06) wired against `partition_to_node_id.values()` with NO sampling; nested final_metrics block (D-07) replaces flat dict; module_run_results_dir replaces Path("../results/federated"); cross-silo legacy path preserved via mode branch (D-03 + Pitfall 8); W&B summary uses best/* and last/* namespaces; manifest carries final_eval_round_index + metrics via dataclasses.replace; D-04 clean sibling filename via sibling_name="manifest.json"; atomic_write_json replaces json.dump
- 4 NEW integration tests in test_server_integration.py covering EVL-01/02/03/04/06
- NEW @pytest.mark.slow regression guard at scripts/foundation/tests/test_baseline_subprocess_determinism.py re-enables the folded phase2-baseline-determinism-path-bug.md todo
- Existing baseline test suite remains GREEN
- Pitfall 9 closure: last_round = max(eval_metrics_history.keys()) (not actual_rounds)
- Pitfall 10 closure: centralized eval outputs (RMSE/MAE/ranking_metrics) feed final_metrics["last"] diagnostics; final_metrics["best"] comes ONLY from the federated extra-eval-round
- Files outside listed `files_modified` remain byte-identical to pre-Plan-03 state (D-18 surgical scope)
</success_criteria>

<output>
After completion, create `.planning/phases/06-evaluation-reporting-harness/06-evaluation-reporting-harness-03-SUMMARY.md` covering:
- Path migration (D-02) + closure of folded phase2-baseline-determinism-path-bug.md
- Extra-eval-round wiring (D-06) including the explicit "no sampling, broadcast to all nodes" choice from Pitfall 5
- Nested final_metrics schema (D-07) and Pitfall 9/10 closures
- W&B summary key migration (best/* + last/*; final/* removed)
- Cross-silo coexistence (D-03 + Pitfall 8 mode branch)
- 4 NEW integration tests + 1 NEW subprocess determinism guard
</output>
</content>
</invoke>