---
phase: 05-pfedrec-migration-reproduction
plan: 04
type: execute
wave: 3
depends_on: ["05-pfedrec-migration-reproduction-03"]
files_modified:
  - federated-pfedrec/federated_pfedrec/server_app.py
  - federated-pfedrec/tests/test_server_integration.py
autonomous: true
requirements: [PFR-02, PFR-06, PFR-08, PFR-09]
must_haves:
  truths:
    - "Server boots cross-device with PFedRecSplitFedAvg strategy (Plan 01) wired in (D-12); FedProx variant absent (D-07)"
    - "Initial ArrayRecord includes BOTH 'embedding_item.weight' AND 'affine_output.bias' as GLOBAL params (D-01 propagation)"
    - "G-03-01 discovery round + ADP-06 partition-id-space _server_sampler = server_rng(run_seed) drives selected_clients_per_round in stable partition-id space (PFR-06)"
    - "D-25 mode resolver header at @app.main entry; D-02-mirror NotImplementedError for mode='cross_silo_legacy' BEFORE any heavy work"
    - "D-13 cold-start counter probes .embedding_cache/{run_id}/partition_{pid}.pt before each round (Phase 3 carry-forward)"
    - "D-14 PFR-08 auto-verify hook reads IJCAI-23-PFedRec/sh_result/ml-1m.txt (line 2 = HR=0.7286, NDCG=0.4407 — most recent / closest to paper round 89), parses tokens, asserts |our - reference| ≤ 2.0 absolute points; prints [PFR-08 VERIFIED] / [PFR-08 FAILED] AND embeds results_data['_manifest']['pfr08_verification'] AND wandb.run.summary['final/pfr08']; non-fatal — failed reproduction does NOT abort"
    - "D-15 manifest double-write with module='pfedrec' (PFR-09); audit_doc='PFR-02-AUDIT.md' embedded in the manifest dict so result JSONs reference the SC-1 cross-walk"
    - "D-13 best-round-restore against sampled_ndcg@10 (CONTEXT.md D-13 metric choice; implemented via the Phase-3-D-27 carry-forward in-memory snapshot pattern); FitRes.num_examples = 1 per client (D-24 uniform via Pitfall 5 Option B)"
    - "Auto-verify hook fires AFTER embed_manifest_in_result and BEFORE W&B summary write"
  artifacts:
    - path: "federated-pfedrec/federated_pfedrec/server_app.py"
      provides: "Cross-device PFedRec main loop with D-14 PFR-08 auto-verify hook + D-15 manifest double-write + Phase-3-template carry-forward (D-13 best-round-restore via the Phase-3-D-27 idiom)"
      contains: "[PFR-08 VERIFIED]"
    - path: "federated-pfedrec/tests/test_server_integration.py"
      provides: "8 GREEN integration tests covering G-03-01 / PFR-06 / D-14 (3 sub-tests) / D-15 / D-13 cold-start / D-13 best-round-restore"
  key_links:
    - from: "server_app.py @app.main initial ArrayRecord"
      to: "model.get_global_parameters() returns 2 keys per D-01"
      via: "PFedRecMLP._GLOBAL_PARAMS = ('embedding_item.weight', 'affine_output.bias')"
      pattern: "ArrayRecord\\(.*get_global_parameters\\(\\)"
    - from: "server_app.py D-14 hook"
      to: "IJCAI-23-PFedRec/sh_result/ml-1m.txt line 2 (HR=0.7286, NDCG=0.4407)"
      via: "_parse_reference_results helper + _emit_pfr_08_verification"
      pattern: "PFR-08 VERIFIED|PFR-08 FAILED"
    - from: "server_app.py D-15 manifest"
      to: "results_data['_manifest']['module']='pfedrec' + audit_doc='PFR-02-AUDIT.md' + sibling <run_id>-manifest.json"
      via: "build_run_manifest + embed_manifest_in_result + write_manifest_sibling"
      pattern: "module=\"pfedrec\"|module='pfedrec'"
---

<objective>
Server-side full migration: clone Phase 4 Plan 5 server_app.py shape with 5 PFedRec-specific deltas including the unique D-14 PFR-08 auto-verify hook (Wave-3 parallel with Plan 05).

Purpose:
  - PFR-02 (server-side D-12 strategy + D-01 GLOBAL bias propagation): wire `PFedRecSplitFedAvg` (Plan 01) and ensure initial `ArrayRecord` includes both keys.
  - PFR-06 server half: G-03-01 discovery round + ADP-06 partition-id-space `_server_sampler = server_rng(run_seed)` (Phase 4 carry-forward); FitRes.num_examples = 1 per client (Pitfall 5 Option B for D-24 uniform weight).
  - PFR-08 (the headline reproduction gate): D-14 auto-verify hook reads `IJCAI-23-PFedRec/sh_result/ml-1m.txt` line 2 (HR=0.7286, NDCG=0.4407 — Open Question 1 recommendation: most recent / closest to paper-reported round 89), asserts `|our - ref| ≤ 2.0pts` (multiplying ratios by 100 for readable log lines), prints `[PFR-08 VERIFIED]` / `[PFR-08 FAILED]`, embeds verification dict in `results_data['_manifest']['pfr08_verification']`, logs `wandb.run.summary['final/pfr08'] = bool`. NON-FATAL — failed reproduction does NOT abort the run.
  - PFR-09: D-15 double-write manifest with `module="pfedrec"` (Phase 3 / Phase 4 idiom). The manifest dict additionally carries `audit_doc='PFR-02-AUDIT.md'` so the result JSON includes a back-pointer to the SC-1 cross-walk authored by Plan 01 Task 3.

Output:
  - 1 modified file (server_app.py — full cross-device main loop) shipped in a single `tdd="true"` task with bundled tests.
  - 1 new test file (test_server_integration.py — 8 GREEN integration tests) bundled with the same task to avoid the "ship code with smoke verify, ship tests separately" anti-pattern.

DECISION-ID DISAMBIGUATION (cross-file legend):
  - "CONTEXT.md D-13" (this phase) = best-round-restore monitor metric is `sampled_ndcg@10`.
  - "Phase-3-D-27 carry-forward" (cross-phase idiom) = the in-memory snapshot-and-restore implementation pattern that Phases 2/3/4 used.
  - "CONTEXT.md D-27" (this phase) = weight-policy override behavior under paper_compat_pfedrec (NOT best-round-restore). Mentions of "D-27" in this plan that refer to the snapshot pattern have been renamed to "D-13 best-round-restore" or "Phase-3-D-27 carry-forward" to avoid collision with CONTEXT.md D-27.
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/PROJECT.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/ROADMAP.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/STATE.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-RESEARCH.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-VALIDATION.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-01-PLAN.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-02-PLAN.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-03-PLAN.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/02-baseline-migration/02-baseline-migration-04-SUMMARY.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/02-baseline-migration/02-baseline-migration-05-SUMMARY.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/03-personalized-migration/03-personalized-migration-04-SUMMARY.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-05-SUMMARY.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/IJCAI-23-PFedRec/sh_result/ml-1m.txt
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/IJCAI-23-PFedRec/engine.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec/federated_pfedrec/server_app.py

<interfaces>
<!-- Phase 4 Plan 5 server_app shape (clone with 5 PFedRec-specific deltas; the canonical 6-delta-over-Phase-3 template) -->
<!-- Source: federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py -->

```python
# Module-top imports the executor consumes
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict, Message
from flwr.serverapp import Grid, ServerApp
from flwr.common import FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy

from fedrec_foundation.bundle import verify_bundle
from fedrec_foundation.manifest import (
    build_run_manifest, embed_manifest_in_result, write_manifest_sibling, generate_run_id,
)
from fedrec_foundation.mode import resolve_mode_defaults, log_mode_and_overrides
from fedrec_foundation.paths import data_derived
from fedrec_foundation.rng import server_rng
from fedrec_foundation.split import load_split_manifest

from federated_pfedrec.strategy import PFedRecSplitFedAvg
from federated_pfedrec.models.pfedrec_mlp import PFedRecMLP
```

```python
# D-14 PFR-08 auto-verify helpers (Phase 5 NEW; mirror RESEARCH §Code Examples)
def _parse_reference_results(reference_path: Path) -> Tuple[float, float]:
    """Parse IJCAI-23-PFedRec/sh_result/ml-1m.txt; pick LAST line (line 2 today: HR=0.7286, NDCG=0.4407)."""
    if not reference_path.exists():
        raise RuntimeError(f"PFR-08 reference file not found: {reference_path}")
    with open(reference_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise RuntimeError(f"PFR-08 reference file is empty: {reference_path}")
    target = lines[-1]
    tokens = target.split("-")
    hr, ndcg = None, None
    for token in tokens:
        if token.lstrip().startswith("hr:"):
            hr = float(token.split(":")[1].strip())
        elif token.lstrip().startswith("ndcg:"):
            ndcg = float(token.split(":")[1].strip())
    if hr is None or ndcg is None:
        raise RuntimeError(f"PFR-08 reference parse failed: {target!r}")
    return hr, ndcg


def _emit_pfr_08_verification(
    final_metrics: Dict[str, float],
    reference_path: Path,
    tolerance_pts: float = 2.0,
) -> Tuple[bool, str, Dict]:
    """Returns (passed, log_line, audit_dict). Non-fatal — failed reproduction
    does NOT raise. Multiplies HR/NDCG ratios by 100 for log readability."""
    try:
        ref_hr, ref_ndcg = _parse_reference_results(reference_path)
    except RuntimeError as e:
        return False, f"[PFR-08 FAILED: {e}]", {"passed": False, "error": str(e)}

    our_hr = final_metrics.get("sampled_hr@10", float("nan"))
    our_ndcg = final_metrics.get("sampled_ndcg@10", float("nan"))
    if any(v != v for v in (our_hr, our_ndcg)):  # NaN check
        return False, f"[PFR-08 FAILED: missing metric our_hr={our_hr} our_ndcg={our_ndcg}]", \
            {"passed": False, "error": "missing metric"}

    delta_hr_pts = abs(our_hr - ref_hr) * 100.0
    delta_ndcg_pts = abs(our_ndcg - ref_ndcg) * 100.0
    passed = delta_hr_pts <= tolerance_pts and delta_ndcg_pts <= tolerance_pts
    tag = "VERIFIED" if passed else "FAILED"
    log_line = (
        f"[PFR-08 {tag}] our_hr@10={our_hr:.4f} ref_hr@10={ref_hr:.4f} "
        f"Δhr={delta_hr_pts:.2f}pts | "
        f"our_ndcg@10={our_ndcg:.4f} ref_ndcg@10={ref_ndcg:.4f} "
        f"Δndcg={delta_ndcg_pts:.2f}pts | tolerance={tolerance_pts:.1f}pts"
    )
    audit = {
        "passed": passed,
        "delta_hr_pts": delta_hr_pts,
        "delta_ndcg_pts": delta_ndcg_pts,
        "ref_hr": ref_hr,
        "ref_ndcg": ref_ndcg,
        "our_hr": our_hr,
        "our_ndcg": our_ndcg,
        "ref_path": str(reference_path),
        "tolerance_pts": tolerance_pts,
    }
    return passed, log_line, audit
```

```python
# Phase 3 Plan 4 / Phase 4 Plan 5 main loop shape (clone)
@app.main()
def main(grid: Grid, context: Context):
    # 1. Mode resolve + D-02 guard + D-25 hyperparam reads
    mode = str(context.run_config.get("mode", "paper_compat_pfedrec"))
    profile = resolve_mode_defaults(mode)
    overrides = log_mode_and_overrides(mode, profile, context.run_config)
    if mode == "cross_silo_legacy":
        raise NotImplementedError(
            "PFedRec cross-silo path is FROZEN per Phase 5 D-09. "
            "See .planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md §Deferred."
        )

    num_rounds = int(context.run_config.get("num-server-rounds", profile.num_server_rounds))
    fraction_train = float(context.run_config.get("fraction-train", profile.fraction_train))
    run_seed = int(context.run_config.get("run-seed", 42))
    weight_policy = str(context.run_config.get("weight-policy", profile.weight_policy))
    checkpoint_rule = str(context.run_config.get("checkpoint-rule", profile.checkpoint_rule))
    reuse_cache = bool(context.run_config.get("reuse-cache", False))
    latent_dim = int(context.run_config.get("latent-dim", profile.embedding_dim))
    lr = float(context.run_config.get("lr", profile.lr))
    lr_eta = int(context.run_config.get("lr-eta", 80))
    local_epochs = int(context.run_config.get("local-epochs", profile.local_epochs))
    batch_size = int(context.run_config.get("batch-size", 256))
    num_train_negatives = int(context.run_config.get("num-negatives", profile.num_train_negatives))

    # 2. run_id materialized EARLY (Phase 3 idiom)
    run_id = str(context.run_config.get("run-id") or generate_run_id())

    # 3. W&B init (federated-cf-cross-device default for paper_compat_pfedrec — D-10)
    default_project = "federated-cf-cross-device" if mode in ("benchmark_cross_device", "paper_compat_pfedrec") else "federated-cf"
    wandb_project = context.run_config.get("wandb-project") or default_project
    wandb_enabled = bool(context.run_config.get("wandb-enabled", True))
    if wandb_enabled:
        wandb.init(project=wandb_project, config={"run_id": run_id, "mode": mode, **overrides})

    # 4. Foundation bundle (for FND-07 fingerprints)
    derived = data_derived()
    foundation_index = verify_bundle(derived)
    split_manifest = load_split_manifest(derived / "split_manifest.json")

    # 5. Initial arrays (split learning: GLOBAL only, but Phase 5 GLOBAL has 2 keys per D-01)
    bundle_num_users = ...  # from foundation_index or split_manifest
    bundle_num_items = ...
    global_model = PFedRecMLP(num_items=bundle_num_items, latent_dim=latent_dim)
    arrays = ArrayRecord(global_model.get_global_parameters())  # dict with 2 keys per D-01

    # 6. Strategy wire-up (Plan 01)
    strategy = PFedRecSplitFedAvg(fraction_fit=fraction_train)

    # 7. G-03-01 discovery round (Phase 2/3/4 carry-forward)
    all_node_ids = sorted(grid.get_node_ids())
    expected_n = profile.num_supernodes
    discovery_messages = [
        Message(content=RecordDict({"config": ConfigRecord({"discover_only": True})}),
                message_type="evaluate", dst_node_id=node_id)
        for node_id in all_node_ids
    ]
    discovery_responses = list(grid.send_and_receive(discovery_messages))
    partition_to_node_id: Dict[int, int] = {}
    for response in discovery_responses:
        if response.has_error():
            continue
        m = response.content["metrics"]
        pid = int(m["partition_id"])
        partition_to_node_id[pid] = response.metadata.src_node_id
    missing = sorted(set(range(expected_n)) - set(partition_to_node_id.keys()))
    assert not missing, f"Discovery failed for partitions {missing[:5]}..."

    # 8. _server_sampler (single instance, partition-id space)
    _server_sampler = server_rng(run_seed)

    # 9. FL loop with D-13 cold-start counter + D-13 best-round-restore (Phase-3-D-27 idiom)
    selected_clients_per_round: List[List[int]] = []
    cold_starts_per_round: List[int] = []
    eval_metrics_history: Dict[int, Dict[str, float]] = {}
    best_metric = float("-inf")
    best_round_num = 0
    best_arrays = arrays

    for round_num in range(1, num_rounds + 1):
        # Sample partitions
        num_selected = max(1, int(expected_n * fraction_train))
        selected_pids = _server_sampler.sample(range(expected_n), num_selected)
        selected_node_ids = [partition_to_node_id[pid] for pid in selected_pids]
        selected_clients_per_round.append([int(p) for p in selected_pids])

        # D-13 cold-start probe (counter — distinct from best-round-restore)
        cache_root = Path(".embedding_cache") / run_id
        if reuse_cache:
            cold_count = 0
            print(f"[D-13] reuse_cache=true; cold_count short-circuited to 0 per D-09")
        else:
            cold_count = sum(
                1 for pid in selected_pids
                if not (cache_root / f"partition_{pid}.pt").exists()
            )
        cold_starts_per_round.append(cold_count)

        # Train
        train_config = ConfigRecord({
            "lr": lr, "lr_eta": lr_eta, "round_num": round_num,
            "run_id": run_id, "reuse_cache": reuse_cache,
        })
        train_messages = [
            Message(
                content=RecordDict({"arrays": arrays, "config": train_config}),
                message_type="train", dst_node_id=node_id,
            )
            for node_id in selected_node_ids
        ]
        train_responses = list(grid.send_and_receive(train_messages))

        # D-24 uniform weight via Pitfall 5 Option B: FitRes.num_examples = 1
        fit_results = []
        for response in train_responses:
            if response.has_error():
                continue
            metrics = dict(response.content["metrics"])
            arr_record = response.content["arrays"]
            num_examples = 1  # D-24 uniform under paper_compat_pfedrec
            fit_results.append((DummyClientProxy("c"), FitRes(
                status=..., parameters=ndarrays_to_parameters([...]),
                num_examples=num_examples, metrics=metrics,
            )))
        aggregated_params, agg_metrics = strategy.aggregate_fit(round_num, fit_results, [])
        # apply aggregated_params to arrays for next round

        # Evaluate
        eval_config = ConfigRecord({
            "round_num": round_num, "run_id": run_id, "reuse_cache": reuse_cache,
        })
        eval_messages = [
            Message(
                content=RecordDict({"arrays": arrays, "config": eval_config}),
                message_type="evaluate", dst_node_id=node_id,
            )
            for node_id in selected_node_ids
        ]
        eval_responses = list(grid.send_and_receive(eval_messages))
        eval_results = [(DummyClientProxy("c"), EvaluateRes(
            status=..., loss=0.0, num_examples=1,
            metrics=dict(response.content["metrics"]),
        )) for response in eval_responses if not response.has_error()]
        loss, thesis_metrics = strategy.aggregate_evaluate(round_num, eval_results, [])
        eval_metrics_history[round_num] = dict(thesis_metrics)

        # D-13 best-round-restore snapshot (Phase-3-D-27 carry-forward — the in-memory
        # snapshot-and-restore implementation pattern from Phases 2/3/4 — NOT to be
        # confused with CONTEXT.md D-27 which is the weight-policy override behavior).
        # Monitor metric is sampled_ndcg@10 per CONTEXT.md D-13.
        current_ndcg = thesis_metrics.get("sampled_ndcg@10", float("-inf"))
        if checkpoint_rule in ("best_round_restore", "best_round") and current_ndcg > best_metric:
            best_metric = current_ndcg
            best_round_num = round_num
            best_arrays = ArrayRecord({
                k: v.detach().clone() for k, v in arrays.to_torch_state_dict().items()
            })

        # W&B per-round logs
        if wandb_enabled:
            wandb.log({"round": round_num, **thesis_metrics, "round/cold_starts": cold_count}, step=round_num)

    # 10. Restore best-round arrays (D-13 best-round-restore via the Phase-3-D-27 idiom)
    if checkpoint_rule in ("best_round_restore", "best_round") and best_round_num > 0:
        arrays = best_arrays

    # 11. Final metrics + result JSON
    final_metrics = eval_metrics_history.get(best_round_num, eval_metrics_history.get(num_rounds, {}))
    results_data = {
        "model_name": "PFedRecMLP",
        "dataset": "ml-1m",
        "federated_config": {...},
        "final_metrics": final_metrics,
        "training_rounds": eval_metrics_history,
        "selected_clients_per_round": selected_clients_per_round,
        "cold_starts": {
            "per_round": cold_starts_per_round,
            "total_cold_starts": sum(cold_starts_per_round),
            "total_client_selections": sum(len(s) for s in selected_clients_per_round),
            "cold_start_rate": (sum(cold_starts_per_round) / max(1, sum(len(s) for s in selected_clients_per_round))),
        },
        "best_round": best_round_num,
    }

    # 12. D-15 manifest double-write + D-14 PFR-08 auto-verify
    manifest = build_run_manifest(
        run_id=run_id, mode_profile=profile, run_seed=run_seed,
        foundation_index=foundation_index, overrides=overrides,
        module="pfedrec",
        audit_doc="PFR-02-AUDIT.md",   # D-15 back-pointer to the SC-1 cross-walk (Plan 01 Task 3)
    )
    embed_manifest_in_result(manifest, results_data)  # mutates results_data in place

    # D-14 hook fires AFTER embed_manifest_in_result and BEFORE W&B summary write
    reference_path = Path(__file__).resolve().parents[2] / "IJCAI-23-PFedRec" / "sh_result" / "ml-1m.txt"
    pfr08_passed, pfr08_log_line, pfr08_audit = _emit_pfr_08_verification(
        final_metrics=final_metrics, reference_path=reference_path, tolerance_pts=2.0,
    )
    print(pfr08_log_line)
    results_data["_manifest"]["pfr08_verification"] = pfr08_audit  # post-embed mutation pattern

    results_filename = f"results/federated/{run_id}_results.json"
    Path(results_filename).parent.mkdir(parents=True, exist_ok=True)
    with open(results_filename, "w") as f:
        json.dump(results_data, f, indent=2)
    write_manifest_sibling(manifest, results_filename)

    # D-14 W&B summary push (non-fatal)
    if wandb_enabled:
        wandb.run.summary["final/pfr08"] = bool(pfr08_passed)
        wandb.run.summary["final/pfr08_delta_hr_pts"] = float(pfr08_audit.get("delta_hr_pts", float("nan")))
        wandb.run.summary["final/pfr08_delta_ndcg_pts"] = float(pfr08_audit.get("delta_ndcg_pts", float("nan")))
        for k, v in final_metrics.items():
            wandb.run.summary[f"final/{k}"] = v
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Ship server_app.py (full Phase-5 cross-device main loop with D-14 PFR-08 auto-verify hook) + test_server_integration.py (8 GREEN integration tests) bundled together — single TDD task, no smoke-only verify gap</name>
  <files>federated-pfedrec/federated_pfedrec/server_app.py, federated-pfedrec/tests/test_server_integration.py</files>
  <read_first>
    - federated-pfedrec/federated_pfedrec/server_app.py — current 587-LOC state to replace
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py — Phase 4 Plan 5 server_app reference (clone the shape; remove prototype/alpha-diagnostics; add PFR-08 hook)
    - federated-personalized-cf/federated_personalized_cf/server_app.py — Phase 3 Plan 4 server_app reference (D-13 cold-start counter + Phase-3-D-27 best-round in-memory snapshot pattern)
    - federated-baseline-cf/federated_baseline_cf/server_app.py — Phase 2 Plan 4/5 reference (G-03-01 discovery round)
    - federated-pfedrec/federated_pfedrec/strategy.py — Plan 01 PFedRecSplitFedAvg + GLOBAL_PARAM_KEYS
    - federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py — Plan 01 _GLOBAL_PARAMS
    - federated-personalized-cf/tests/test_server_integration.py — Phase 3 6-test pattern reference
    - federated-adaptive-personalized-cf/tests/test_server_integration.py — Phase 4 7-test pattern reference (Pattern 3: source-level proximity tests with rfind() for docstring duplicates)
    - .planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md §decisions D-02, D-09, D-10, D-13 (best-round-restore monitor metric), D-14, D-15, D-24, D-26, D-27 (the CONTEXT-local D-27 = weight-policy override; NOT the carry-forward best-round idiom)
    - .planning/phases/05-pfedrec-migration-reproduction/05-RESEARCH.md §Pattern 2 (5-delta-over-Phase-4 server template) + §Open Questions 1-4
    - .planning/phases/05-pfedrec-migration-reproduction/05-VALIDATION.md §Per-Task Verification Map rows 5-04-01 through 5-04-08 (8 tests)
    - .planning/phases/02-baseline-migration/02-baseline-migration-04-SUMMARY.md
    - .planning/phases/03-personalized-migration/03-personalized-migration-04-SUMMARY.md
    - .planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-05-SUMMARY.md
    - .planning/phases/05-pfedrec-migration-reproduction/PFR-02-AUDIT.md (Plan 01 Task 3) — the SC-1 cross-walk this server_app references via D-15 manifest's audit_doc field
    - IJCAI-23-PFedRec/sh_result/ml-1m.txt — auto-verify target file format (2 lines, dash-separated tokens)
    - scripts/foundation/fedrec_foundation/manifest.py — build_run_manifest signature + RunManifest fields (confirm `module` kwarg + ability to thread an `audit_doc` field; if the foundation API does not yet accept `audit_doc` directly, place it under `manifest.overrides` or in a post-build mutation `manifest['audit_doc'] = 'PFR-02-AUDIT.md'` keyed access — match Phase-3/4 idiom)
    - scripts/foundation/fedrec_foundation/rng.py — server_rng signature + .sample(range, k) API
  </read_first>
  <behavior>
    Bundled with Task 1 implementation are 8 GREEN tests (VALIDATION.md rows 5-04-01..5-04-08). Tests are AUTHORED FIRST per `tdd=true`; implementation follows. Tests assert at the source level (Phase 3/4 idiom — live Grid not available in unit tests).

    - Test 1 (test_discovery_round_partition_id_sampling): assert `_server_sampler` is created via `server_rng(run_seed)` exactly once in source (single instance); assert `_server_sampler.sample(range(expected_n), ...)` is called inside the FL loop (substring match).
    - Test 2 (test_server_rng_seeded_sampling): import `server_rng` from `fedrec_foundation.rng`; build two instances `server_rng(42)` and `server_rng(42)`; sample range(100) k=20 from each; assert byte-identical lists. Build `server_rng(43)`; sample; assert different list.
    - Test 3 (test_pfr08_autoverify_parses_sh_result): import `_parse_reference_results` from `federated_pfedrec.server_app`; call with the actual `IJCAI-23-PFedRec/sh_result/ml-1m.txt` path; assert returns `(0.7286423841059603, 0.4407401988138434)` (line 2 — most recent — Open Question 1 recommendation). Allow `pytest.approx(0.7286, abs=1e-3)` tolerance for float robustness. Skip cleanly if reference file not present in the clone.
    - Test 4 (test_pfr08_autoverify_pass_within_2pts): import `_emit_pfr_08_verification`; call with `final_metrics={"sampled_hr@10": 0.730, "sampled_ndcg@10": 0.450}` (within +0.0014 / +0.0093 of reference) and a writable temp reference file containing one synthetic line `"2026-01-01-...-hr: 0.7286-ndcg: 0.4407-..."`; assert returns `(passed=True, log_line containing "[PFR-08 VERIFIED]", audit_dict with passed=True, delta_hr_pts<2.0, delta_ndcg_pts<2.0)`.
    - Test 5 (test_pfr08_autoverify_fail_outside_2pts): same call shape but `final_metrics={"sampled_hr@10": 0.50, "sampled_ndcg@10": 0.20}` (way off); assert returns `(passed=False, log_line containing "[PFR-08 FAILED]", audit_dict with passed=False, delta_hr_pts>2.0)`. Critical: the function MUST NOT raise — failed reproduction is non-fatal.
    - Test 6 (test_manifest_double_write_module_pfedrec): grep server_app.py source for `build_run_manifest(...)` invocation; assert the source contains `module="pfedrec"` (exact substring) within ~200 chars of the `build_run_manifest` call site (source-level proximity check). Also assert source contains `audit_doc="PFR-02-AUDIT.md"` AND both `embed_manifest_in_result` AND `write_manifest_sibling` calls (D-15 double-write + SC-1 back-pointer).
    - Test 7 (test_cold_starts_per_round_logged): grep server_app.py source for the literal substring `"cold_starts"` (the result-JSON key); assert at least 2 occurrences (declaration + result write). Assert `_REPO_ROOT/.embedding_cache/<run_id>/partition_{pid}.pt` substring or equivalent existence-probe pattern (`partition_{pid}.pt` + `.exists()`) appears in source.
    - Test 8 (test_best_round_restore_against_ndcg10): grep server_app.py source for `current_ndcg = thesis_metrics.get("sampled_ndcg@10"`; assert match (CONTEXT.md D-13 monitor metric). Assert source contains `if checkpoint_rule in ("best_round_restore", "best_round")` (D-13 best-round-restore + spelling tolerance via the Phase-3-D-27 carry-forward idiom). Assert source contains `arrays = best_arrays` (the actual restore step). Use `src.rfind(...)` for the restore string if proximity-aware (Phase 4 Plan 5 lesson — module docstring may duplicate the literal).
  </behavior>
  <action>

This is ONE bundled TDD task that ships:
  - `federated-pfedrec/federated_pfedrec/server_app.py` — full cross-device main loop with all Plan-01-04 carry-forward + the unique D-14 PFR-08 auto-verify hook.
  - `federated-pfedrec/tests/test_server_integration.py` — 8 source-level + functional tests covering VALIDATION rows 5-04-01..5-04-08.

The two artifacts are bundled into a single `tdd="true"` task to match the file-count footprint of Plan 03 (which also bundles tests with implementation) and to avoid the anti-pattern where Task 1 ships server_app.py with only a smoke-import verify and Task 2 backfills the tests. The tests are authored FIRST and the implementation grows to satisfy them (RED→GREEN per `tdd=true`).

**Step 1 — Author the 8 tests first** (tests RED until server_app.py exists). Test file lives at `federated-pfedrec/tests/test_server_integration.py`.

Tests 1, 6, 7, 8 are SOURCE-LEVEL via `inspect.getsource(server_app_module)` (they read the new server_app.py source as text). Test 2 imports `server_rng` and exercises it directly. Test 3 reads the actual `IJCAI-23-PFedRec/sh_result/ml-1m.txt` (skips cleanly if absent). Tests 4 + 5 use `tmp_path` to seed a synthetic reference file.

```python
"""Phase 5 PFR-02 / PFR-06 / PFR-08 / PFR-09 server_app integration regression guard."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest


# ---- Test 1: G-03-01 discovery + ADP-06 partition-id-space sampling (source-level) ----
def test_discovery_round_partition_id_sampling() -> None:
    import federated_pfedrec.server_app as server_app
    src = inspect.getsource(server_app)
    # _server_sampler is created via server_rng(run_seed) exactly once
    assert src.count("_server_sampler = server_rng") == 1, (
        "PFR-06: _server_sampler must be a single instance; multiple instantiations "
        "would re-seed mid-loop and break determinism."
    )
    assert "_server_sampler.sample(range(" in src, (
        "ADP-06: partition-id-space sampling required (sample(range(expected_n), ...))"
    )


# ---- Test 2: server_rng seeded sampling determinism ----
def test_server_rng_seeded_sampling() -> None:
    from fedrec_foundation.rng import server_rng

    a = server_rng(42); b = server_rng(42); c = server_rng(43)
    out_a = a.sample(range(100), 20)
    out_b = b.sample(range(100), 20)
    out_c = c.sample(range(100), 20)
    assert list(out_a) == list(out_b), "FND-06 same seed must give same sample"
    assert list(out_a) != list(out_c), "FND-06 different seed must give different sample"


# ---- Test 3: D-14 reference parser anchored on the real ml-1m.txt ----
def test_pfr08_autoverify_parses_sh_result() -> None:
    from federated_pfedrec.server_app import _parse_reference_results

    repo_root = Path(__file__).resolve().parents[2]
    ref = repo_root / "IJCAI-23-PFedRec" / "sh_result" / "ml-1m.txt"
    if not ref.exists():
        pytest.skip("reference file not present in this clone")
    hr, ndcg = _parse_reference_results(ref)
    # Open Question 1 recommendation: line 2 / most recent / closest to paper round 89
    assert hr == pytest.approx(0.7286, abs=1e-3), f"line 2 HR: {hr}"
    assert ndcg == pytest.approx(0.4407, abs=1e-3), f"line 2 NDCG: {ndcg}"


# ---- Test 4: D-14 pass path within tolerance ----
def test_pfr08_autoverify_pass_within_2pts(tmp_path) -> None:
    from federated_pfedrec.server_app import _emit_pfr_08_verification

    ref = tmp_path / "ml-1m-synthetic.txt"
    ref.write_text(
        "2026-01-01 00-00-00-latent_dim: 32-lr: 0.1-clients_sample_ratio: 1.0-"
        "num_round: 100-negatives: 4-lr_eta: 80-batch_size: 256-"
        "hr: 0.7286-ndcg: 0.4407-best_round: 89-optimizer: sgd-l2_regularization: 0.0\n"
    )
    final_metrics = {"sampled_hr@10": 0.730, "sampled_ndcg@10": 0.450}
    passed, log_line, audit = _emit_pfr_08_verification(
        final_metrics=final_metrics, reference_path=ref, tolerance_pts=2.0,
    )
    assert passed is True
    assert "PFR-08 VERIFIED" in log_line
    assert audit["passed"] is True
    assert audit["delta_hr_pts"] < 2.0
    assert audit["delta_ndcg_pts"] < 2.0


# ---- Test 5: D-14 fail path is NON-FATAL (does not raise) ----
def test_pfr08_autoverify_fail_outside_2pts(tmp_path) -> None:
    from federated_pfedrec.server_app import _emit_pfr_08_verification

    ref = tmp_path / "ml-1m-synthetic.txt"
    ref.write_text(
        "2026-01-01 00-00-00-latent_dim: 32-lr: 0.1-clients_sample_ratio: 1.0-"
        "num_round: 100-negatives: 4-lr_eta: 80-batch_size: 256-"
        "hr: 0.7286-ndcg: 0.4407-best_round: 89-optimizer: sgd-l2_regularization: 0.0\n"
    )
    final_metrics = {"sampled_hr@10": 0.50, "sampled_ndcg@10": 0.20}
    # Critical: must NOT raise — failed reproduction is non-fatal.
    passed, log_line, audit = _emit_pfr_08_verification(
        final_metrics=final_metrics, reference_path=ref, tolerance_pts=2.0,
    )
    assert passed is False
    assert "PFR-08 FAILED" in log_line
    assert audit["passed"] is False
    assert audit["delta_hr_pts"] > 2.0


# ---- Test 6: D-15 double-write + audit_doc back-pointer (source-level) ----
def test_manifest_double_write_module_pfedrec() -> None:
    import federated_pfedrec.server_app as server_app
    src = inspect.getsource(server_app)
    # build_run_manifest call site contains module="pfedrec"
    assert "module=\"pfedrec\"" in src or "module='pfedrec'" in src, (
        "D-15: build_run_manifest must thread module='pfedrec' (PFR-09)"
    )
    # SC-1 back-pointer
    assert "audit_doc=\"PFR-02-AUDIT.md\"" in src or "audit_doc='PFR-02-AUDIT.md'" in src, (
        "D-15 back-pointer: result JSON must reference PFR-02-AUDIT.md (Plan 01 Task 3)"
    )
    # Double-write idiom
    assert "embed_manifest_in_result" in src
    assert "write_manifest_sibling" in src


# ---- Test 7: D-13 cold-start counter (source-level) ----
def test_cold_starts_per_round_logged() -> None:
    import federated_pfedrec.server_app as server_app
    src = inspect.getsource(server_app)
    assert src.count("cold_starts") >= 2, (
        "D-13 cold-start counter: 'cold_starts' should appear in declaration + result write"
    )
    # existence probe pattern
    assert "partition_" in src and ".pt" in src and ".exists()" in src, (
        "D-13: must probe .embedding_cache/<run_id>/partition_{pid}.pt existence"
    )


# ---- Test 8: D-13 best-round-restore (Phase-3-D-27 idiom) against sampled_ndcg@10 (source-level) ----
def test_best_round_restore_against_ndcg10() -> None:
    import federated_pfedrec.server_app as server_app
    src = inspect.getsource(server_app)
    # CONTEXT.md D-13 monitor metric
    assert 'thesis_metrics.get("sampled_ndcg@10"' in src, (
        "CONTEXT.md D-13: best-round-restore monitor metric is sampled_ndcg@10"
    )
    # checkpoint_rule spelling tolerance (best_round_restore | best_round)
    assert 'checkpoint_rule in ("best_round_restore", "best_round")' in src, (
        "D-13 checkpoint_rule spelling tolerance — Phase-3-D-27 carry-forward idiom"
    )
    # Use rfind to anchor on the LAST occurrence (avoid module-docstring duplicates — Phase 4 Plan 5 lesson)
    restore_idx = src.rfind("arrays = best_arrays")
    assert restore_idx > 0, "D-13 best-round-restore: arrays = best_arrays must execute at loop end"
```

**Step 2 — Implement `federated-pfedrec/federated_pfedrec/server_app.py`** to satisfy the 8 tests above.

Rip-and-replace the current 587-LOC server_app.py cloning the Phase 4 Plan 5 server template with the 5 PFedRec-specific deltas listed in RESEARCH §Pattern 2:

**Delta 1: Strategy class** = `PFedRecSplitFedAvg` (no FedProx variant per D-07; no `if strategy_name == "fedprox"` branch).

**Delta 2: Initial arrays** = `ArrayRecord(global_model.get_global_parameters())` where `PFedRecMLP._GLOBAL_PARAMS = ('embedding_item.weight', 'affine_output.bias')` per D-01. Both keys must appear in the initial broadcast.

**Delta 3: NO prototype / alpha bookkeeping** — Phase 4 Plan 5 had `best_prototype` snapshot/restore + 6-scalar `alpha_diagnostics` aggregation. Phase 5 has neither. Skip those blocks entirely.

**Delta 4: NO centralized eval block** — split learning (server doesn't hold LOCAL params). `final_metrics = eval_metrics_history[best_round_num]` per Phase 3 idiom.

**Delta 5: D-14 PFR-08 auto-verify hook** — NEW to Phase 5. Implement the two helpers `_parse_reference_results` and `_emit_pfr_08_verification` exactly as shown in the `<interfaces>` block above (top-of-module helpers, importable from tests).

The hook fires AFTER `embed_manifest_in_result(manifest, results_data)` and BEFORE the W&B summary write (RESEARCH Open Question 3 recommendation). Hook is non-fatal — failed reproduction does NOT raise.

**Pitfall 5 Option B (D-24 uniform weight on FIT side):** When wrapping `train_responses` into `FitRes` objects for `strategy.aggregate_fit(...)`, set `num_examples = 1` for each (NOT the per-user training-sample count). FedAvg's existing num_examples-weighted aggregator is then mathematically uniform. Add an inline comment naming Pitfall 5 / D-24:

```python
# Pitfall 5 Option B: weight_policy="uniform" under paper_compat_pfedrec means
# every client contributes weight=1 (mirrors engine.py:81 len(round_user_params)
# division). Setting FitRes.num_examples = 1 makes FedAvg's existing
# num_examples-weighted aggregate mathematically uniform without overriding
# aggregate_fit. See RESEARCH §Pitfall 5.
num_examples = 1  # D-24 uniform under paper_compat_pfedrec
```

**Phase 3/4 carry-forward (verbatim clone):**
- D-25 mode resolver header (resolve_mode_defaults + log_mode_and_overrides) at @app.main entry.
- D-02 mirror: `if mode == "cross_silo_legacy": raise NotImplementedError(...)` BEFORE any heavy work, error message cites `.planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md §Deferred`.
- run_id materialized EARLY (`generate_run_id()` when no override).
- W&B project default: `federated-cf-cross-device` for `mode in {"benchmark_cross_device", "paper_compat_pfedrec"}`; `federated-cf` otherwise.
- G-03-01 discovery round: broadcast `discover_only=True` ConfigRecord; build `partition_to_node_id` from responses.
- ADP-06 partition-id-space sampling: single `_server_sampler = server_rng(run_seed)` instance pre-loop; `_server_sampler.sample(range(expected_n), k)` per round; `selected_clients_per_round` stores partition_ids 0..N-1.
- D-13 cold-start COUNTER (the per-round logging counter — NOT the best-round restore): probe `.embedding_cache/{run_id}/partition_{pid}.pt` existence per selected pid; `results_data["cold_starts"] = {per_round, total, rate}`; under `reuse_cache=true` short-circuit to 0.
- D-13 best-round-restore (the in-memory snapshot via the Phase-3-D-27 carry-forward idiom — NOT to be confused with CONTEXT.md D-27 weight-policy override): `if current_ndcg > best_metric:` snapshot ArrayRecord with deep-clone; restore at loop end before final result write; `checkpoint_rule` accepts both 'best_round_restore' and 'best_round' spellings; monitor metric is `sampled_ndcg@10` per CONTEXT.md D-13.
- D-15 manifest double-write: `build_run_manifest(..., module="pfedrec", audit_doc="PFR-02-AUDIT.md")` + `embed_manifest_in_result` + `write_manifest_sibling`. The `audit_doc="PFR-02-AUDIT.md"` field is a back-pointer to the SC-1 cross-walk authored by Plan 01 Task 3; if `build_run_manifest` does not yet accept the kwarg directly, threading it via `manifest.overrides['audit_doc'] = 'PFR-02-AUDIT.md'` (or a post-build dict mutation `manifest['audit_doc'] = 'PFR-02-AUDIT.md'`) is acceptable provided the resulting result JSON's `_manifest` block contains the key.
- DummyClientProxy (D-18 surgical: preserve from current server_app.py if present; else copy Phase 3/4 idiom).
- stdlib random eradicated module-wide.

**File structure** — Place the two D-14 helpers (`_parse_reference_results`, `_emit_pfr_08_verification`) at MODULE LEVEL above the `@app.main` definition. Keep them importable from tests.

**No deviation from Plans 01-03 contracts:**
- Initial ArrayRecord MUST include both `embedding_item.weight` AND `affine_output.bias` (test enforces).
- PFedRecSplitFedAvg (NOT SplitFedAvg, NOT PFedRecSplitFedProx) is the only strategy class instantiated.
- ConfigRecord broadcast to clients MUST include `run_id` + `reuse_cache` keys (Plan 03 client_app.py reads these).

**Step 3 — Run the bundled test suite GREEN.**

```
cd federated-pfedrec && pytest tests/test_server_integration.py -x -v
```

All 8 tests must pass (Test 3 may skip if the reference file is absent). The smoke import is implicitly exercised by Tests 1, 6, 7, 8 (all import server_app).
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec && pytest tests/test_server_integration.py -x -v</automated>
  </verify>
  <acceptance_criteria>
    - File exists at `federated-pfedrec/tests/test_server_integration.py`
    - File contains 8 test functions matching VALIDATION.md row IDs 5-04-01 through 5-04-08
    - `grep -c "PFedRecSplitFedAvg" federated-pfedrec/federated_pfedrec/server_app.py` returns at least 2 (import + instantiation)
    - `grep -cE "(^|[^A-Za-z_])SplitFedAvg\\(" federated-pfedrec/federated_pfedrec/server_app.py` (word-boundary check) — only `PFedRecSplitFedAvg` matches; standalone `SplitFedAvg` does not appear
    - `grep -c "PFedRecSplitFedProx\|SplitFedProx" federated-pfedrec/federated_pfedrec/server_app.py` returns 0 (D-07)
    - `grep -c "resolve_mode_defaults\|log_mode_and_overrides" federated-pfedrec/federated_pfedrec/server_app.py` returns at least 2 (D-25 header)
    - `grep -c "cross_silo_legacy" federated-pfedrec/federated_pfedrec/server_app.py` returns at least 1 (D-02 mirror guard)
    - `grep -c "raise NotImplementedError" federated-pfedrec/federated_pfedrec/server_app.py` returns at least 1
    - `grep -c "_server_sampler = server_rng" federated-pfedrec/federated_pfedrec/server_app.py` returns 1 (ADP-06)
    - `grep -c "discover_only" federated-pfedrec/federated_pfedrec/server_app.py` returns at least 1 (G-03-01)
    - `grep -c "selected_clients_per_round" federated-pfedrec/federated_pfedrec/server_app.py` returns at least 2 (declare + accumulate + result write)
    - `grep -c "_parse_reference_results\|_emit_pfr_08_verification" federated-pfedrec/federated_pfedrec/server_app.py` returns at least 2 (def each)
    - `grep -c "PFR-08 VERIFIED\|PFR-08 FAILED" federated-pfedrec/federated_pfedrec/server_app.py` returns at least 1 (D-14 log lines)
    - `grep -c "pfr08_verification" federated-pfedrec/federated_pfedrec/server_app.py` returns at least 1 (D-14 manifest field)
    - `grep -c 'module="pfedrec"\|module=.pfedrec.' federated-pfedrec/federated_pfedrec/server_app.py` returns at least 1 (D-15 PFR-09)
    - `grep -c 'audit_doc="PFR-02-AUDIT.md"\|audit_doc=.PFR-02-AUDIT.md.' federated-pfedrec/federated_pfedrec/server_app.py` returns at least 1 (D-15 back-pointer to SC-1 cross-walk)
    - `grep -c "build_run_manifest\|embed_manifest_in_result\|write_manifest_sibling" federated-pfedrec/federated_pfedrec/server_app.py` returns at least 3 (D-15 double-write)
    - `grep -c "best_round_restore\|best_round" federated-pfedrec/federated_pfedrec/server_app.py` returns at least 2 (D-13 best-round-restore via Phase-3-D-27 idiom + spelling tolerance)
    - `grep -c "cold_starts\|cold_count" federated-pfedrec/federated_pfedrec/server_app.py` returns at least 3 (D-13 cold-start counter)
    - `grep -c "num_examples = 1" federated-pfedrec/federated_pfedrec/server_app.py` returns at least 1 (Pitfall 5 Option B / D-24)
    - `grep -cE "random\\.seed\\(|random\\.sample\\(|^import random$" federated-pfedrec/federated_pfedrec/server_app.py` returns 0 (stdlib random eradicated)
    - `python -c "from federated_pfedrec.server_app import app, _parse_reference_results, _emit_pfr_08_verification; print('ok')"` prints `ok`
    - `pytest federated-pfedrec/tests/test_server_integration.py -x -v` exits 0 with exactly 8 tests passed (or 7 passed + 1 skipped if reference file genuinely missing — Test 3 has a `pytest.skip` guard)
    - Cumulative module suite: `pytest federated-pfedrec/tests/ -x` exits 0 with all Plan-01/02/03/04 tests passing (≥25 cumulative)
  </acceptance_criteria>
  <done>
    - server_app.py: full cross-device migration with all 5 PFedRec-specific deltas + Phase 3/4 carry-forward (D-13 cold-start counter; D-13 best-round-restore via the Phase-3-D-27 in-memory snapshot idiom; D-15 double-write with audit_doc back-pointer to PFR-02-AUDIT.md)
    - PFR-02 server-side D-12 wire-up + D-01 GLOBAL bias propagation in initial ArrayRecord
    - PFR-06: G-03-01 discovery + ADP-06 partition-id-space sampling + Pitfall 5 Option B uniform weighting
    - PFR-08: D-14 auto-verify hook fires post-embed pre-W&B-summary; non-fatal; embeds audit dict in `_manifest.pfr08_verification`
    - PFR-09: D-15 double-write manifest with `module="pfedrec"` + `audit_doc="PFR-02-AUDIT.md"`
    - 8 GREEN integration tests covering VALIDATION.md per-task verification map for Plan 04 (5-04-01..5-04-08)
    - Smoke import GREEN; tests author-first per `tdd=true`; no Task-1-ships-code-with-smoke-verify-Task-2-backfills-tests anti-pattern
  </done>
</task>

</tasks>

<verification>
- Module test suite: `cd federated-pfedrec && pytest tests/ -x -v` → all GREEN (≥25 cumulative across Plans 01/02/03/04)
- Smoke import: `python -c "from federated_pfedrec.server_app import app, _parse_reference_results, _emit_pfr_08_verification; print('ok')"` → prints `ok`
- D-18 surgical: `git diff --name-only` shows ONLY server_app.py + test_server_integration.py (Wave-3 file ownership disjoint with Plan 05's scripts/foundation/tests/test_pfedrec_subprocess_determinism.py)
- BSL-05 cross-file regression in server_app.py: `grep -rnE "random\.seed\(|random\.sample\(|^import random$" federated-pfedrec/federated_pfedrec/server_app.py` → 0
- D-15 back-pointer: result JSON's `_manifest.audit_doc` field equals `"PFR-02-AUDIT.md"` (PFR-08 reproduction artifact references the SC-1 cross-walk)
</verification>

<success_criteria>
- server_app.py: full Phase-5 cross-device main loop with all 5 PFedRec-specific deltas (D-12 strategy, D-01 GLOBAL bias, no prototype/alpha, no centralized eval, D-14 auto-verify) + Phase 3/4 carry-forward (D-25 mode resolver, D-02 mirror, G-03-01 discovery, ADP-06 sampler, D-13 cold-start counter, D-13 best-round-restore via the Phase-3-D-27 in-memory snapshot idiom, D-15 double-write with audit_doc='PFR-02-AUDIT.md', Pitfall 5 Option B for D-24 uniform)
- 8 GREEN tests (test_server_integration.py) cover VALIDATION rows 5-04-01..08 — bundled with the implementation in a single TDD task (no smoke-verify gap)
- D-14 hook is non-fatal (test 5 verifies failure path doesn't raise)
- D-15 manifest carries audit_doc back-pointer to PFR-02-AUDIT.md (Plan 01 Task 3) — closes the SC-1 trail end-to-end through the result JSON
- Cumulative module test suite ≥25 GREEN
- Wave-3 file-ownership disjoint with Plan 05
- Decision-ID disambiguation: every "best-round-restore" reference is renamed to "D-13 best-round-restore" or "Phase-3-D-27 carry-forward" — CONTEXT.md D-27 (weight-policy override) is left untouched and unambiguous
</success_criteria>

<output>
After completion, create `.planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-04-SUMMARY.md` covering:
- 5 PFedRec-specific deltas + Phase 3/4 carry-forward (with explicit naming: D-13 best-round-restore via Phase-3-D-27 idiom; CONTEXT.md D-27 weight-policy override is a separate concern)
- D-14 auto-verify hook implementation (placement, log format, audit dict shape, non-fatal semantics)
- D-15 audit_doc back-pointer to PFR-02-AUDIT.md (closes SC-1 trail end-to-end through result JSON)
- W&B project switch to federated-cf-cross-device for paper_compat_pfedrec (D-10)
- Open Question 1 resolution (line 2 / most-recent / closest-to-paper-round-89 chosen)
- Open Question 3 resolution (BOTH stdout + result-JSON-embedded + W&B-summary; non-fatal)
- Test counts mapped to VALIDATION rows 5-04-01..08 (bundled with implementation per `tdd=true`)
- Plan 05 readiness: subprocess regression guard can consume selected_clients_per_round + .embedding_cache schema_v3 cache + _manifest.pfr08_verification + _manifest.audit_doc fields
</output>
</output>
