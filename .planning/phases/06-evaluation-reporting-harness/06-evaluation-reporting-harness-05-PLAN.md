---
phase: 06-evaluation-reporting-harness
plan: 05
type: execute
wave: 2
depends_on:
  - 06-evaluation-reporting-harness-01
  - 06-evaluation-reporting-harness-02
files_modified:
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py
  - federated-adaptive-personalized-cf/tests/test_server_integration.py
  - scripts/foundation/tests/test_adaptive_determinism.py
autonomous: true
requirements: [EVL-01, EVL-02, EVL-03, EVL-04, EVL-06]
must_haves:
  truths:
    - "Adaptive server_app writes results.json + manifest.json to <repo>/results/federated/adaptive/<run_id>/ via module_run_results_dir for cross-device modes (D-01 + D-02); legacy cross-silo path preserved (D-03)"
    - "When checkpoint_rule in {best_round_restore, best_round} and best_round_num > 0, a post-restore broadcast eval round runs against ALL nodes in partition_to_node_id (D-06)"
    - "Pitfall 4 closure: the extra-eval-round eval ConfigRecord ATTACHES strategy._global_prototype (already restored to best_prototype per D-07) so clients see the restored prototype during the canonical eval"
    - "Replaces the silent eval_metrics_history[best_round_num] lookup at server_app.py:978 with the broadcast result"
    - "final_metrics is now nested {best, last, best_round, last_round, final_eval_round_index}; last derives from max-key of eval_metrics_history (Pitfall 9)"
    - "W&B run.summary uses best/* and last/* namespaces; existing alpha/* and prototype/* summary keys preserved verbatim (no churn for adaptive-specific diagnostic surfaces)"
    - "Manifest carries final_eval_round_index + nested metrics block via dataclasses.replace; existing _manifest['best_prototype'] post-build mutation preserved verbatim"
    - "alpha_diagnostics_history keeps writing to results_data unchanged (Phase 4 D-16 contract preserved)"
    - "Test path probe in scripts/foundation/tests/test_adaptive_determinism.py updated from flat *_results.json to per-run-dir */results.json glob; existing _manifest.best_prototype byte-identity invariant preserved"
  artifacts:
    - path: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py"
      provides: "Adaptive server_app with extra-eval-round attaching best_prototype to ConfigRecord (Pitfall 4) + per-run-dir + nested final_metrics + best/last W&B namespaces"
      contains: "from fedrec_foundation.paths import module_run_results_dir"
    - path: "federated-adaptive-personalized-cf/tests/test_server_integration.py"
      provides: "5 NEW assertions: results path, extra-eval-round wired (Pitfall 4 prototype attached), best/last block schema, per-group exposure history, prototype-on-broadcast-config invariant"
      contains: "def test_extra_eval_broadcasts_best_prototype"
    - path: "scripts/foundation/tests/test_adaptive_determinism.py"
      provides: "Updated _RESULTS_DIR probe matches Phase 6 per-run-dir layout; existing best_prototype byte-identity invariant preserved"
      contains: "_RESULTS_DIR / \"adaptive\""
  key_links:
    - from: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py::extra_eval_messages"
      to: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py::strategy._global_prototype"
      via: "eval_config['global_prototype'] = strategy._global_prototype.tolist() (Pitfall 4)"
      pattern: "eval_config\\[.global_prototype.\\] = "
    - from: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py::manifest"
      to: "scripts/foundation/fedrec_foundation/manifest.py::RunManifest.metrics"
      via: "manifest = replace(manifest, final_eval_round_index=N, metrics=results_data['final_metrics'])"
      pattern: "replace\\(manifest, final_eval_round_index="
    - from: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py::best_prototype"
      to: "results_data['_manifest']['best_prototype']"
      via: "Phase-4 D-06 post-embed mutation preserved verbatim — Phase 6 layers on top, does NOT replace"
      pattern: "results_data\\[._manifest.\\]\\[.best_prototype.\\]"
---

<objective>
Wire the Plan 01+02 foundation primitives into `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py`. This closes EVL-01/02/03/04/06 for the adaptive module while preserving the Phase-4 D-05/D-06/D-07 best_prototype snapshot/restore/embed plumbing AND closing Pitfall 4 (the extra-eval-round broadcast must carry the restored best_prototype on its eval ConfigRecord).

Purpose:
  - Replace `Path("../results/federated/adaptive")` (server_app.py:1183) with `module_run_results_dir("adaptive", run_id)` for cross-device modes (D-02). Legacy cross-silo path preserved (Pitfall 8).
  - Insert the D-06 extra-eval-round block AFTER `strategy._global_prototype = strategy.best_prototype` (server_app.py:956 — the Phase-4 D-07 prototype restore) AND BEFORE the federated-eval print (server_app.py:965). The block REPLACES the line-978 silent `eval_metrics_history[final_round_for_metrics]` lookup.
  - **Pitfall 4 closure:** the extra-eval-round eval ConfigRecord MUST attach `strategy._global_prototype.tolist()` under the key `"global_prototype"`, mirroring the in-loop eval-config build site at server_app.py:814-815 (`eval_config_dict["global_prototype"] = global_prototype.tolist()`). Without this, every client falls back to a zero or stale prototype during the canonical eval and the `best_*` block reports lower NDCG@10 than the in-loop best_round_num round did (the warning sign described in RESEARCH §Pitfall 4).
  - Restructure `final_metrics` from flat (line 978) to nested `{best, last, best_round, last_round, final_eval_round_index}` per D-07.
  - Migrate W&B summary keys from `final/*` (line 996) to `best/*` and `last/*`. PRESERVE existing `alpha/*` and `prototype/*` summary keys verbatim (these are adaptive-specific diagnostic surfaces).
  - Mutate manifest via `dataclasses.replace(manifest, final_eval_round_index=N, metrics=...)` between `build_run_manifest` and `embed_manifest_in_result`. The Phase-4 post-embed mutation `results_data['_manifest']['best_prototype'] = ...` (line 1176-1179) is PRESERVED verbatim — it lives layered on top.
  - Update path probe in `scripts/foundation/tests/test_adaptive_determinism.py` from flat to per-run-dir glob; preserve existing `_manifest.best_prototype` byte-identity invariant.

Output:
  - `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` modified.
  - `federated-adaptive-personalized-cf/tests/test_server_integration.py` extended: 5 NEW assertions including Pitfall-4 prototype-on-broadcast-config test.
  - `scripts/foundation/tests/test_adaptive_determinism.py` updated: path probe migrated.
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
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-adaptive-personalized-cf/tests/test_server_integration.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/scripts/foundation/tests/test_adaptive_determinism.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-05-SUMMARY.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-06-SUMMARY.md

<interfaces>
<!-- Wave 1 deps (Plans 01 + 02) -->
```python
from fedrec_foundation.paths import module_run_results_dir, repo_root
from fedrec_foundation.atomic import atomic_write_json
from fedrec_foundation.manifest import (
    build_run_manifest,
    embed_manifest_in_result,
    write_manifest_sibling,
)
from dataclasses import replace as dataclass_replace
```

<!-- Existing adaptive server_app.py drop sites -->
```python
# Line 568: partition_to_node_id (G-03-01)
# Line 608+896: best_arrays / best_arrays = ArrayRecord({...})
# Line 814-815: in-loop eval-config build site:
#   if global_prototype is not None:
#       eval_config_dict["global_prototype"] = global_prototype.tolist()
# Line 943-957: D-07 best_prototype restore block; specifically line 956:
#   strategy._global_prototype = strategy.best_prototype
# Line 978: final_metrics = dict(eval_metrics_history.get(final_round_for_metrics, {}))   # D-06 BUG TARGET
# Line 996: wandb.run.summary[f"final/{key}"]   # MIGRATE
# Lines 1030-1043: alpha/* + prototype/* W&B summary surfaces (PRESERVE VERBATIM)
# Line 1156-1170: build_run_manifest(...) (extend with dataclass_replace)
# Lines 1173-1179: D-06 post-embed mutation:
#   results_data["_manifest"]["best_prototype"] = [...]   # PRESERVE VERBATIM
# Line 1183: results_dir = Path("../results/federated/adaptive")   # REPLACE
# Line 1191: write_manifest_sibling(manifest, results_filename)    # ADD sibling_name
```

<!-- Adaptive strategy.py — DO NOT MODIFY -->
```python
# AdaptiveSplitFedAvg.aggregate_evaluate already emits sufficient stats with
# per-group sampled_hr@10 / sampled_ndcg@10 / evaluated_users keys.
# strategy.best_prototype + strategy._global_prototype are already in scope.
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Wire module_run_results_dir + Pitfall-4-aware extra-eval-round + nested final_metrics + best/last W&B namespaces into adaptive server_app.py; ship 5 NEW integration tests including the Pitfall-4 prototype-on-broadcast-config invariant; update test_adaptive_determinism.py path probe</name>
  <files>federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py, federated-adaptive-personalized-cf/tests/test_server_integration.py, scripts/foundation/tests/test_adaptive_determinism.py</files>
  <read_first>
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py — current state (CRITICAL line refs: 78 manifest imports, 568 partition_to_node_id, 608+896 best_arrays, 805-815 in-loop eval-config with global_prototype attachment, 943-957 D-07 prototype restore, 978 D-06 BUG, 996 W&B final/* loop, 1030-1043 alpha/* + prototype/* summary surfaces — PRESERVE, 1156-1170 build_run_manifest, 1173-1179 best_prototype post-embed mutation — PRESERVE, 1183 results_dir, 1191 write_manifest_sibling)
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py — AdaptiveSplitFedAvg.aggregate_evaluate + best_prototype + _global_prototype (DO NOT TOUCH)
    - federated-adaptive-personalized-cf/tests/test_server_integration.py — current shape (extend, do not rewrite)
    - scripts/foundation/tests/test_adaptive_determinism.py — current `_RESULTS_DIR` glob + best_prototype byte-identity invariant (update probe in-place)
    - .planning/phases/06-evaluation-reporting-harness/06-CONTEXT.md §decisions D-01..D-09
    - .planning/phases/06-evaluation-reporting-harness/06-RESEARCH.md §Pattern 2 (extra-eval-round) including the ADAPTIVE-ONLY ADDITION comment + §Common Pitfalls Pitfall 4 (best_prototype on broadcast ConfigRecord)
    - .planning/phases/06-evaluation-reporting-harness/06-VALIDATION.md §Per-Task Verification Map rows 6-05-01, 6-05-02, 6-05-03
    - .planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-05-SUMMARY.md — D-05/D-06/D-07 prototype snapshot/embed/restore patterns
  </read_first>
  <behavior>
    - Test 1 (test_results_path_repo_root_anchored): Mock-mode test with `mode="benchmark_cross_device"`; assert `module_run_results_dir("adaptive", run_id)` resolves to `<repo>/results/federated/adaptive/<run_id>/`; assert `results.json` and `manifest.json` exist there (D-04).
    - Test 2 (test_extra_eval_round_after_best_arrays_restore): Construct fake `eval_metrics_history` with rounds {1: ndcg=0.32, 2: ndcg=0.46, 3: ndcg=0.43} so `best_round_num=2`. Mock `grid.send_and_receive`; assert `final_eval_round_index == actual_rounds + 1` AND `strategy.aggregate_evaluate(final_eval_round_index, extra_results, [])` was invoked AND `final_metrics["best"]["sampled_ndcg@10"]` is the broadcast result (NOT `eval_metrics_history[2]["sampled_ndcg@10"]`).
    - Test 3 (test_canonical_artifact_carries_best_and_last_blocks): Same fixture; load `results.json`; assert nested final_metrics schema; assert `_manifest["schema_version"] == 2`; assert `_manifest["final_eval_round_index"] >= 1`; assert `_manifest["metrics"] == final_metrics`; **PRESERVATION CHECK:** assert `_manifest["best_prototype"]` is still present (Phase 4 D-06 post-embed mutation preserved by Phase 6).
    - Test 4 (test_round_metrics_history_carries_per_group_exposure): Same as baseline/personalized; assert at least one round in `results["eval_metrics_history"]` carries `evaluated_users_sparse|medium|dense`.
    - Test 5 (test_extra_eval_broadcasts_best_prototype) — **PITFALL 4 REGRESSION GUARD**: This is the headline adaptive-specific test. Construct a fixture where `strategy.best_prototype` is a known non-None numpy array (e.g., `np.ones(embedding_dim) * 0.5`). After `strategy._global_prototype = strategy.best_prototype` (D-07 restore) runs, intercept the extra-eval-round message construction; for at least one of the constructed `extra_eval_messages`, extract its `eval_config` ConfigRecord; assert the key `"global_prototype"` is present in the ConfigRecord; assert `eval_config["global_prototype"]` equals `strategy.best_prototype.tolist()` element-wise. The test fails if the extra-eval-round forgets to attach the restored prototype (the exact warning sign described in RESEARCH §Pitfall 4).
  </behavior>
  <action>
**Edit 1: Add foundation imports.** Locate the manifest imports around line 78. Add:

```python
from fedrec_foundation.paths import module_run_results_dir, repo_root
from fedrec_foundation.atomic import atomic_write_json
from dataclasses import replace as dataclass_replace
```

**Edit 2: Extract module-name local constant.** Inside `@app.main()`, after `run_id` is materialized:

```python
_MODULE: str = "adaptive"   # cross-references: build_run_manifest, module_run_results_dir
```

**Edit 3: Insert the D-06 extra-eval-round block — WITH PITFALL 4 PROTOTYPE ATTACHMENT.** AFTER line 956 (`strategy._global_prototype = strategy.best_prototype` — the D-07 prototype restore) AND BEFORE the federated-eval print at line 965. The block MUST mirror the in-loop eval-config build site at server_app.py:814-815 (`eval_config_dict["global_prototype"] = global_prototype.tolist()`) so clients receiving the extra-eval messages see the RESTORED prototype.

```python
    # =========================================================================
    # D-06: extra eval round on the restored best-round state.
    # PITFALL 4 closure: eval ConfigRecord ATTACHES the restored best_prototype
    # so clients see the same prototype that produced best_round_num's metrics.
    # Without this attach, every client falls back to a zero/stale prototype
    # during the canonical eval and the best_* block reports lower NDCG than
    # the in-loop best round did (warning sign in RESEARCH §Pitfall 4).
    # =========================================================================
    final_eval_round_index: int = 0
    best_round_metrics: Dict[str, Any] = {}

    if checkpoint_rule in ("best_round_restore", "best_round") and best_round_num > 0:
        final_eval_round_index = actual_rounds + 1
        # Use the RESTORED prototype (D-07) — strategy._global_prototype was just
        # assigned to strategy.best_prototype at line 956.
        final_global_prototype = strategy.get_global_prototype()
        print(
            f"\n[D-06] Broadcasting extra eval round {final_eval_round_index} "
            f"on restored best-round state (best_round={best_round_num}, "
            f"target nodes={len(partition_to_node_id)}, "
            f"prototype_attached={final_global_prototype is not None})..."
        )

        eval_node_ids = sorted(partition_to_node_id.values())
        extra_eval_messages = []
        for nid in eval_node_ids:
            extra_eval_config_dict = {"lr": lr}
            # PITFALL 4: attach the restored prototype, mirroring in-loop eval
            # ConfigRecord construction at server_app.py:814-815.
            if final_global_prototype is not None:
                extra_eval_config_dict["global_prototype"] = final_global_prototype.tolist()
            eval_config = ConfigRecord(extra_eval_config_dict)
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

**Edit 4: Restructure final_metrics.** REPLACE lines 977-981 (current `final_metrics = dict(eval_metrics_history.get(final_round_for_metrics, {}))` block — the D-06 bug). New body:

```python
    # =========================================================================
    # D-07: nested final_metrics schema. `best` from D-06 extra-eval-round;
    # `last` from max-key of eval_metrics_history (Pitfall 9 — guards against
    # early-stopping edge cases).
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

    print_evaluation_metrics(
        final_metrics["best_round"],
        final_metrics["best"],
        context,
    )
```

**Edit 5: Migrate W&B summary keys.** REPLACE the loop around line 996 (`for key, value in final_metrics.items(): wandb.run.summary[f"final/{key}"]`). The existing `early_stopping/*`, `alpha/*`, `prototype/*`, `training/*` summary surfaces (lines 999-1043) MUST be preserved verbatim — they are adaptive-specific diagnostics independent of the best/last namespacing. New body for the final_metrics-specific block ONLY:

```python
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

The existing `wandb.log(final_log, step=actual_rounds + 1)` for per-round shipping should switch its key prefix from `final/` to `final_eval/best/`:

```python
        final_log = {"round": actual_rounds + 1}
        for key, value in final_metrics["best"].items():
            if isinstance(value, (int, float)):
                final_log[f"final_eval/best/{key}"] = value
        wandb.log(final_log, step=actual_rounds + 1)
```

**Edit 6: Mutate manifest with new fields — BEFORE existing best_prototype post-embed mutation.** AFTER `manifest = build_run_manifest(...)` (~line 1156-1170) and BEFORE `embed_manifest_in_result(manifest, results_data)` (~line 1173), insert:

```python
    manifest = dataclass_replace(
        manifest,
        final_eval_round_index=final_eval_round_index,
        metrics=results_data["final_metrics"],
    )
```

**CRITICAL: PRESERVE the existing best_prototype post-embed mutation block at lines 1173-1179 verbatim.** This Phase-4 D-06 surface is independent of the Phase-6 schema-v2 metrics field. Do NOT touch:

```python
    # Phase 4 D-06 — DO NOT TOUCH (preserved verbatim by Phase 6):
    # results_data["_manifest"]["best_prototype"] = [...]
```

**Edit 7: Replace results-dir + filename + manifest-sibling.** REPLACE lines 1183-1192 (currently `results_dir = Path("../results/federated/adaptive")` + `with open(...)` + `write_manifest_sibling`). New body:

```python
    if mode in ("benchmark_cross_device", "paper_compat_pfedrec"):
        run_dir = module_run_results_dir(_MODULE, run_id)
        results_filename = run_dir / "results.json"
        sibling_kwarg = {"sibling_name": "manifest.json"}
    else:  # cross_silo_legacy — preserved per D-03
        legacy_dir = repo_root() / "results" / "federated" / "adaptive"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        results_filename = legacy_dir / f"{run_id}_results.json"
        sibling_kwarg = {}

    atomic_write_json(str(results_filename), results_data)
    sibling_path = write_manifest_sibling(manifest, results_filename, **sibling_kwarg)
    print(f"Results saved to: {results_filename.resolve()}")
    print(f"Manifest sibling: {sibling_path.resolve()}")
```

**Edit 8: Update test_adaptive_determinism.py path probe.** Locate `_RESULTS_DIR` and the glob pattern. Update the glob from the legacy flat layout (likely `_RESULTS_DIR.glob("adaptive/*_results.json")` or `_RESULTS_DIR.rglob("*adaptive*_results.json")`) to the new per-run-dir layout: `_RESULTS_DIR.glob("adaptive/*/results.json")` (or, if `_RESULTS_DIR` is already adaptive-scoped, `_RESULTS_DIR.glob("*/results.json")`). PRESERVE the existing `_manifest.best_prototype` byte-identity invariant — only the path probe changes.

**Edit 9: Extend test_server_integration.py with 5 NEW tests.** Read the existing file shape; reuse its mocking style. Add Tests 1-5 from the behavior block above. Test 5 (`test_extra_eval_broadcasts_best_prototype`) is the headline regression guard — it MUST intercept message construction (e.g., monkeypatch `grid.create_message` to capture `content` arg) and assert `eval_config["global_prototype"]` is present and equals `strategy.best_prototype.tolist()`.

**Verify by running:**

```bash
cd federated-adaptive-personalized-cf && pytest tests/test_server_integration.py -x -v -k "results_path or extra_eval or best_last_blocks or per_group_exposure or prototype_attached"
cd scripts/foundation && pytest tests/test_adaptive_determinism.py -x -v -m slow
```
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-adaptive-personalized-cf && pytest tests/test_server_integration.py -x -v -k "results_path or extra_eval or best_last_blocks or per_group_exposure or prototype_attached"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "from fedrec_foundation.paths import module_run_results_dir" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1
    - `grep -c "from dataclasses import replace as dataclass_replace" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1
    - `grep -c "_MODULE: str = .adaptive." federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1
    - `grep -c "Path(.\\.\\./results/federated/adaptive.)" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 0 (D-02 cutover)
    - `grep -c "module_run_results_dir(_MODULE, run_id)" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1
    - `grep -c "extra_eval_config_dict\\[.global_prototype.\\]" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1 (Pitfall 4 — prototype attached to extra-eval ConfigRecord)
    - `grep -c "final_eval_round_index" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 5
    - `grep -c "wandb.run.summary\\[f.final/" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 0 (legacy namespace removed for final_metrics; alpha/* + prototype/* preserved)
    - `grep -c "wandb.run.summary\\[f.best/" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1
    - `grep -c "wandb.run.summary\\[f.last/" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1
    - `grep -c "wandb.run.summary\\[.alpha/" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 4 (alpha/* surfaces preserved)
    - `grep -c "wandb.run.summary\\[.prototype/" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 1 (prototype/final_norm preserved)
    - `grep -c "results_data\\[._manifest.\\]\\[.best_prototype.\\]" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 2 (Phase-4 post-embed mutation preserved verbatim)
    - `grep -c "max(eval_metrics_history.keys())" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1 (Pitfall 9)
    - `grep -c "if mode in (.benchmark_cross_device., .paper_compat_pfedrec.)" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1 (Pitfall 8)
    - `grep -c "sibling_name=.manifest.json." federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1
    - `grep -c "atomic_write_json" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 1
    - `grep -c "final_metrics = dict(eval_metrics_history.get(final_round_for_metrics" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 0 (D-06 forbidden lookup removed)
    - `grep -E "adaptive/\\*/results\\.json|.\\*/results\\.json" scripts/foundation/tests/test_adaptive_determinism.py` returns at least 1 line (path probe migrated)
    - `cd federated-adaptive-personalized-cf && pytest tests/test_server_integration.py -x -v -k "results_path or extra_eval or best_last_blocks or per_group_exposure or prototype_attached"` exits 0 with all 5 NEW tests passing
    - `cd federated-adaptive-personalized-cf && pytest tests/ -q -m "not slow"` exits 0 (no regressions)
  </acceptance_criteria>
  <done>
    - server_app.py: extra-eval-round block inserted AFTER D-07 prototype restore (line 956); ConfigRecord ATTACHES restored prototype (Pitfall 4 closure); flat final_metrics restructured to nested {best, last, best_round, last_round, final_eval_round_index}; D-06 forbidden lookup at line 978 removed; W&B summary final/* -> best/* + last/* (alpha/* and prototype/* surfaces preserved); manifest mutated via dataclasses.replace BEFORE existing best_prototype post-embed mutation; results path resolves via module_run_results_dir for cross-device; cross-silo legacy preserved (D-03 + Pitfall 8); atomic_write_json replaces json.dump
    - test_server_integration.py: 5 NEW tests pinning EVL-01/02/03/04/06 + Pitfall 4 prototype-attached invariant
    - test_adaptive_determinism.py: _RESULTS_DIR glob updated to per-run-dir layout; existing best_prototype byte-identity invariant preserved
    - Existing adaptive tests remain GREEN
  </done>
</task>

</tasks>

<verification>
- Imports clean: `python -c "from federated_adaptive_personalized_cf import server_app; print('ok')"` exits 0
- Path migration: `grep -c "Path(.\\.\\./results/federated/adaptive.)" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 0
- Pitfall 4 closure mechanically present: `grep -c "extra_eval_config_dict\\[.global_prototype.\\] = final_global_prototype.tolist()" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1
- D-06 forbidden lookup removed: `grep -c "final_metrics = dict(eval_metrics_history.get(final_round_for_metrics" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 0
- Phase-4 best_prototype post-embed mutation preserved: `grep -c "results_data\\[._manifest.\\]\\[.best_prototype.\\]" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 2
- 5 NEW integration tests pass
- D-18 surgical scope: `git diff --stat` shows ONLY changes to server_app.py + test_server_integration.py + test_adaptive_determinism.py
- Wave-2 file-disjointness held: this plan modifies ONLY federated-adaptive-personalized-cf/ + ONE foundation test file
</verification>

<success_criteria>
- federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py:
  - Extra-eval-round block (D-06) wired AFTER D-07 best_prototype restore (line 956), with the eval ConfigRecord attaching `strategy._global_prototype.tolist()` (Pitfall 4 closure mirroring in-loop site at line 814-815)
  - Nested final_metrics block (D-07) replaces flat dict; D-06-forbidden eval_metrics_history lookup at line 978 removed
  - module_run_results_dir replaces Path("../results/federated/adaptive") for cross-device modes; legacy cross-silo path preserved (D-03 + Pitfall 8)
  - W&B summary uses best/* + last/* (final/* removed for thesis metrics); alpha/* + prototype/* + early_stopping/* + training/* surfaces preserved verbatim
  - Manifest carries final_eval_round_index + metrics via dataclasses.replace (BEFORE existing best_prototype post-embed mutation, which is preserved verbatim)
  - D-04 clean filename via sibling_name="manifest.json"
  - atomic_write_json replaces json.dump
- 5 NEW integration tests in test_server_integration.py covering EVL-01/02/03/04/06 and Pitfall-4 prototype-attached invariant
- Updated path probe in scripts/foundation/tests/test_adaptive_determinism.py preserves existing _manifest.best_prototype byte-identity invariant while finding files in new layout
- Pitfall 9 closure: last_round = max(eval_metrics_history.keys())
- Files outside listed `files_modified` remain byte-identical to pre-Plan-05 state (D-18 surgical)
</success_criteria>

<output>
After completion, create `.planning/phases/06-evaluation-reporting-harness/06-evaluation-reporting-harness-05-SUMMARY.md` covering:
- Path migration (D-02)
- Extra-eval-round wiring with Pitfall-4 prototype-attached invariant — explicit mirror of in-loop eval-config build site at server_app.py:814-815
- Nested final_metrics schema (D-07)
- W&B summary key migration (best/* + last/* for final_metrics; alpha/* + prototype/* preserved)
- Phase-4 best_prototype post-embed mutation explicitly preserved (layered on top of Phase-6 schema-v2 metrics field)
- Path probe update in test_adaptive_determinism.py (preserves best_prototype byte-identity invariant)
- 5 NEW integration tests (the Pitfall-4 test is the headline)
</output>
</content>
</invoke>