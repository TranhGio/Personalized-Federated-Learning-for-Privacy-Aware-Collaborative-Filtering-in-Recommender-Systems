---
phase: 03-personalized-migration
plan: 04
type: execute
subsystem: infra
tags: [server-app, mode-resolver, seeded-sampling, personalized-split-fedavg, personalized-split-fedprox, run-manifest, best-round-restore, discovery-round, partition-id-space, cold-start-counter, cross-device, psn-04, psn-07, d-02, d-13, d-15, d-18, d-25, d-26, d-27, wave-3]
wave: 3
depends_on: [03-personalized-migration-01, 03-personalized-migration-02, 03-personalized-migration-03]
files_modified:
  - federated-personalized-cf/federated_personalized_cf/server_app.py
  - federated-personalized-cf/tests/test_server_integration.py
autonomous: true
requirements: [PSN-04, PSN-07]

must_haves:
  truths:
    - "server_app.py @app.main() resolves a ModeProfile via fedrec_foundation.mode.resolve_mode_defaults(mode) at startup; every hyperparameter is read as `int(context.run_config.get(key, profile.field))` so the profile is canonical and pyproject values are the override surface (D-25)."
    - "_server_sampler = server_rng(run_seed) is instantiated ONCE before the FL loop; discovery round broadcasts @app.evaluate with discover_only=true to every grid.get_node_ids() entry BEFORE round 1 to build partition_to_node_id: Dict[int, int]; _server_sampler.sample(range(num_supernodes), k) runs in partition-id space; selected_clients_per_round stores stable partition_ids 0..N-1 (G-03-01 pattern from Phase 2 Plan 05)."
    - "strategy = PersonalizedSplitFedAvg(...) or PersonalizedSplitFedProx(...) replaces the raw FedAvg/FedProx instantiation; strategy.aggregate_evaluate is called per round with wrapped EvaluateRes tuples and populates eval_metrics_history[round_num] with sum-based thesis metrics."
    - "D-27 in-memory best-round restore: best_metric / best_round_num / best_arrays tracked inside the FL loop on checkpoint_rule ∈ ('best_round_restore', 'best_round'); at training end, arrays = best_arrays is set BEFORE centralized evaluation."
    - "D-15 double-write manifest: build_run_manifest called once with module=\"personalized\"; embed_manifest_in_result injects _manifest into results_data; write_manifest_sibling writes <run_id>-manifest.json beside the result JSON."
    - "D-13 cold-start counter: pre-round check of `.embedding_cache/{run_id}/partition_{pid}.pt` existence (or sig_{hash}/partition_{pid}.pt under reuse-cache=true) for each selected partition_id → cold_starts_this_round: int; accumulated as total_cold_starts; logged per-round to W&B as round/cold_starts; reported in final results JSON as {total_cold_starts, cold_start_rate}."
    - "Default W&B project federated-cf-cross-device for benchmark_cross_device mode (PROJECT.md constraint); legacy cross_silo_legacy stays on federated-cf; explicit run_config['wandb-project'] still wins."
    - "D-02 benchmark-mode guard: if `mode == \"cross_silo_legacy\"` AND the resolved num-supernodes is 6040 → raise NotImplementedError pointing at pre-Phase-3 commit (this module's cross-silo is frozen per D-02); alternatively, server_app detects the impossible combination (mode says cross-silo-legacy but natural partitioning is active with N=6040) and fails loud at startup BEFORE the FL loop starts."
    - "5 GREEN integration tests covering PSN-04 reproducibility (byte-identical selections across seeds), strategy wire-up (sum-not-average sanity), D-15 double-write roundtrip, D-13 cold-start counter math, D-02 NotImplementedError for cross-silo mode."
  artifacts:
    - path: "federated-personalized-cf/federated_personalized_cf/server_app.py"
      provides: "Cross-device server main loop with mode resolver, discovery round, seeded partition-id-space sampling, PersonalizedSplitFedAvg wire-up, D-13 cold-start counter, D-15 manifest double-write, D-27 best-round restore"
    - path: "federated-personalized-cf/tests/test_server_integration.py"
      provides: "5 GREEN integration tests (server_rng reproducibility + distinguishability, strategy sum aggregation, D-15 double-write roundtrip, D-13 cold-start counting, D-02 cross-silo NotImplementedError)"
  key_links:
    - from: "federated-personalized-cf/federated_personalized_cf/server_app.py"
      to: "federated-personalized-cf/federated_personalized_cf/strategy.py::PersonalizedSplitFedAvg"
      via: "strategy instantiation with fraction_fit + PersonalizedSplitFedAvg constructor; strategy.aggregate_evaluate called per round"
      pattern: "PersonalizedSplitFedAvg|PersonalizedSplitFedProx"
    - from: "federated-personalized-cf/federated_personalized_cf/server_app.py"
      to: "fedrec_foundation.manifest.build_run_manifest"
      via: "called once with module=\"personalized\" + overrides + foundation fingerprints at result write time"
      pattern: "build_run_manifest"
    - from: "federated-personalized-cf/federated_personalized_cf/server_app.py"
      to: ".embedding_cache/{run_id}/partition_{pid}.pt"
      via: "D-13 cold-start check: Path exists() BEFORE the round sends train message, accumulated as cold_starts_this_round"
      pattern: "cold_start"
---

<objective>
Migrate federated-personalized-cf/federated_personalized_cf/server_app.py to the cross-device contract mirroring Phase 2 Plans 04 + 05: (1) mode resolver at startup (D-25), (2) one-shot discovery round broadcast + partition_to_node_id build BEFORE the main loop (G-03-01), (3) seeded sampling in partition-id space via _server_sampler = server_rng(run_seed) (PSN-04 server side), (4) PersonalizedSplitFedAvg / PersonalizedSplitFedProx wire-up from Plan 01 (PSN-04), (5) D-27 in-memory best-round restore, (6) D-15 double-write manifest with module="personalized", (7) D-13 cold-start counter (Phase-3-unique — tracks per-round count of newly-activated partitions), (8) D-02 NotImplementedError for cross-silo mode.

Purpose: Closes PSN-04 and PSN-07. After this plan, `python scripts/run.py personalized benchmark_cross_device` produces a reproducible cross-device run end-to-end with a protocol-fingerprinted result artifact.

Output:
- federated-personalized-cf/federated_personalized_cf/server_app.py (migrated; D-18 surgical edits preserve pre-existing WIP for DummyClientProxy / weighted_average_metrics / print_evaluation_metrics / centralized-eval path)
- federated-personalized-cf/tests/test_server_integration.py (5 GREEN tests)
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/03-personalized-migration/03-CONTEXT.md
@.planning/phases/02-baseline-migration/02-baseline-migration-04-SUMMARY.md
@.planning/phases/02-baseline-migration/02-baseline-migration-05-SUMMARY.md
@.planning/phases/03-personalized-migration/03-personalized-migration-01-PLAN.md
@.planning/phases/03-personalized-migration/03-personalized-migration-03-PLAN.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: server_app.py migration — mode resolver + discovery round + partition-id sampling + PersonalizedSplitFedAvg + D-13 cold-start counter + D-15 manifest + D-27 best-round + D-02 guard (PSN-04, PSN-07)</name>
  <files>federated-personalized-cf/federated_personalized_cf/server_app.py</files>
  <read_first>
    - federated-personalized-cf/federated_personalized_cf/server_app.py (ENTIRE FILE — need to inventory DummyClientProxy, weighted_average_metrics, early-stopping setup, centralized eval block, wandb init, existing random.sample(node_ids,...) call at or near line 297, existing FedAvg/FedProx instantiation)
    - federated-baseline-cf/federated_baseline_cf/server_app.py (CANONICAL TEMPLATE — post-Plan-04 + post-Plan-05 shape: mode resolver block, W&B project switch, strategy instantiation, discovery round broadcast, partition_to_node_id build, _server_sampler.sample(range(N), k), selected_clients_per_round = partition_ids, D-27 best-round, D-15 double-write)
    - scripts/foundation/fedrec_foundation/manifest.py (build_run_manifest signature; embed_manifest_in_result / write_manifest_sibling / generate_run_id)
    - scripts/foundation/fedrec_foundation/mode.py (ModeProfile + resolve_mode_defaults + log_mode_and_overrides)
    - scripts/foundation/fedrec_foundation/rng.py (server_rng factory — the sampler interface `.sample(domain, k)`)
    - scripts/foundation/fedrec_foundation/bundle.py (verify_bundle, load_split_manifest)
    - scripts/foundation/fedrec_foundation/paths.py (data_derived())
    - federated-personalized-cf/federated_personalized_cf/strategy.py (POST-PLAN-01 — observe the PersonalizedSplitFedAvg constructor signature is identical to flower.FedAvg; PersonalizedSplitFedProx takes proximal_mu)
    - .planning/phases/02-baseline-migration/02-baseline-migration-05-SUMMARY.md (G-03-01 discovery round step-by-step + partition_to_node_id build + zero-missing assertion)
    - .planning/phases/03-personalized-migration/03-CONTEXT.md (D-13 cold-start counter exact fields; D-02 frozen cross-silo)
  </read_first>
  <action>
    Step 1 — Inventory pre-existing WIP with `git diff federated-personalized-cf/federated_personalized_cf/server_app.py`. Record which code blocks are D-18 pre-existing and must be preserved verbatim. Typical preserves: DummyClientProxy class, weighted_average_metrics helper (for RMSE/MAE — retain for D-18 scope-out), print_evaluation_metrics, early-stopping setup/teardown, CUDA device fallback, load_full_data call for centralized eval, final wandb.run.summary logging.

    Step 2 — Add/update imports:
    ```python
    from flwr.common import (
        ArrayRecord, ConfigRecord, Code, EvaluateRes, Status,
        MetricRecord, Message,
    )
    from federated_personalized_cf.strategy import (
        PersonalizedSplitFedAvg, PersonalizedSplitFedProx,
    )
    from fedrec_foundation.bundle import verify_bundle
    from fedrec_foundation.manifest import (
        build_run_manifest, embed_manifest_in_result,
        generate_run_id, write_manifest_sibling,
    )
    from fedrec_foundation.mode import log_mode_and_overrides, resolve_mode_defaults
    from fedrec_foundation.paths import data_derived
    from fedrec_foundation.rng import server_rng
    from fedrec_foundation.split import load_split_manifest
    ```
    REMOVE `import random` from the module top if present.

    Step 3 — Insert the mode resolver block near the top of @app.main() (before any hyperparameter reads):
    ```python
    # ==== D-25 mode resolver ====
    mode = str(context.run_config.get("mode", "cross_silo_legacy"))
    run_seed = int(context.run_config.get("run-seed", 42))
    profile = resolve_mode_defaults(mode)
    overrides = log_mode_and_overrides(mode, profile, context.run_config)
    # ==== D-02 cross-silo guard ====
    # Personalized cross-device migration removed multi-user-per-client support.
    # cross_silo_legacy mode for this module is frozen; check out a pre-Phase-3 commit.
    if mode == "cross_silo_legacy":
        raise NotImplementedError(
            "Personalized cross-device migration removed multi-user-per-client support per D-02. "
            "Check out a pre-Phase-3 commit (see .planning/phases/03-personalized-migration/03-CONTEXT.md §Deferred) "
            "to reproduce legacy cross-silo numbers."
        )
    ```

    Step 4 — Convert every hyperparameter read to the D-25 shape:
    ```python
    num_rounds = int(context.run_config.get("num-server-rounds", profile.num_server_rounds))
    fraction_train = float(context.run_config.get("fraction-train", profile.fraction_train))
    embedding_dim = int(context.run_config.get("embedding-dim", profile.embedding_dim))
    lr = float(context.run_config.get("lr", profile.lr))
    weight_policy = str(context.run_config.get("weight-policy", profile.weight_policy))
    checkpoint_rule = str(context.run_config.get("checkpoint-rule", getattr(profile, "checkpoint_rule", "best_round_restore")))
    reuse_cache_flag = bool(context.run_config.get("reuse-cache", False))  # D-09
    ```

    Step 5 — W&B project switch (PROJECT.md constraint):
    ```python
    default_project = "federated-cf-cross-device" if mode in ("benchmark_cross_device", "paper_compat_pfedrec") else "federated-cf"
    wandb_project = str(context.run_config.get("wandb-project", default_project))
    wandb_config = {
        "mode": mode, "run_seed": run_seed, "weight_policy": weight_policy,
        "partition_mode": str(context.run_config.get("partition-mode", "natural")),
        "checkpoint_rule": checkpoint_rule,
        "reuse_cache": reuse_cache_flag,
        # ... preserve existing wandb_config contents
    }
    ```

    Step 6 — Strategy instantiation: replace any existing `FedAvg(...)` / `FedProx(...)` / `SplitFedAvg(...)` / `SplitFedProx(...)` with:
    ```python
    strategy_name = str(context.run_config.get("strategy", "fedavg")).lower()
    if strategy_name == "fedprox":
        proximal_mu = float(context.run_config.get("proximal-mu", 0.01))
        strategy = PersonalizedSplitFedProx(
            fraction_fit=fraction_train,
            fraction_evaluate=fraction_train,
            proximal_mu=proximal_mu,
        )
    else:
        strategy = PersonalizedSplitFedAvg(
            fraction_fit=fraction_train,
            fraction_evaluate=fraction_train,
        )
    ```

    Step 7 — Discovery round (G-03-01 pattern from Phase 2 Plan 05). Insert BEFORE the main FL loop:
    ```python
    # ==== G-03-01 discovery round: build partition_id -> node_id map ====
    expected_num_supernodes = int(context.run_config.get("num-supernodes", 6040))
    all_node_ids = list(grid.get_node_ids())
    assert len(all_node_ids) == expected_num_supernodes, \
        f"Discovery pre-check: got {len(all_node_ids)} supernodes, expected {expected_num_supernodes}"
    discovery_config = ConfigRecord({"discover_only": True})
    discovery_messages = [
        Message(
            content=RecordDict({"train_config": discovery_config}),  # adapt key name to match client_app contract
            message_type=MessageType.EVALUATE,
            dst_node_id=nid,
            group_id=str(0),  # pre-round-1 marker
        )
        for nid in all_node_ids
    ]
    partition_to_node_id: Dict[int, int] = {}
    for response in grid.send_and_receive(discovery_messages):
        if response.has_error():
            continue
        eval_metrics = response.content["eval_metrics"].to_dict() if "eval_metrics" in response.content else {}
        pid = eval_metrics.get("partition_id")
        if pid is not None:
            partition_to_node_id[int(pid)] = int(response.metadata.src_node_id)
    missing = set(range(expected_num_supernodes)) - set(partition_to_node_id.keys())
    assert not missing, f"Discovery round missed partitions: {sorted(list(missing))[:10]}{'...' if len(missing) > 10 else ''}"
    ```

    Step 8 — Pre-loop state init:
    ```python
    _server_sampler = server_rng(run_seed)
    selected_clients_per_round: List[List[int]] = []
    best_metric: float = float("-inf")
    best_round_num: int = 0
    best_arrays = arrays  # fallback if no round improves
    total_cold_starts: int = 0
    cold_starts_per_round: List[int] = []
    ```

    Step 9 — Per-round client sampling (partition-id space):
    ```python
    num_selected = max(1, int(round(expected_num_supernodes * fraction_train)))
    selected_pids = list(_server_sampler.sample(range(expected_num_supernodes), num_selected))
    selected_clients_per_round.append([int(p) for p in selected_pids])
    selected_node_ids = [partition_to_node_id[pid] for pid in selected_pids]
    # ==== D-13 cold-start counter: check each selected partition's cache path BEFORE sending the train message ====
    run_id_placeholder = str(context.run_config.get("run-id", "")) or generate_run_id()  # materialize early so cache path resolves the same in client
    # Path layout mirrors client_app._cache_dir_for_run
    if reuse_cache_flag:
        # For reuse-cache=true the sig is shared across runs; existence counts as NOT-cold regardless of run_id.
        cold_count = 0
        for pid in selected_pids:
            # The signature for existence probe matches client-side: method,num_users,num_items,dim,split_hash.
            # Pre-compute the sig path once per round.
            ...
    else:
        cold_count = sum(
            1 for pid in selected_pids
            if not Path(".embedding_cache") / run_id_placeholder / f"partition_{int(pid)}.pt"
        )
    # Simpler + correct: just use Path.exists()
    cache_root = Path(".embedding_cache") / run_id_placeholder
    cold_count = sum(1 for pid in selected_pids if not (cache_root / f"partition_{int(pid)}.pt").exists())
    cold_starts_per_round.append(cold_count)
    total_cold_starts += cold_count
    # W&B: log per-round cold starts
    if wandb_run is not None:
        wandb_run.log({
            "round/selected_clients": [int(p) for p in selected_pids],
            "round/cold_starts": cold_count,
        }, step=round_num)
    ```

    Step 10 — Evaluation aggregation. Wrap each eval response into `EvaluateRes(status=Status(Code.OK, "ok"), loss=metrics_dict.get("eval_loss", 0.0), num_examples=num_examples, metrics=metrics_dict)` where num_examples falls back through `num_training_examples → evaluated_users → num-examples → 1` (Phase 2 Plan 04 pattern). Then:
    ```python
    strategy_loss, thesis_metrics = strategy.aggregate_evaluate(round_num, eval_results, [])
    eval_metrics_history[round_num] = dict(thesis_metrics)
    # Preserve RMSE/MAE via legacy weighted_average_metrics fallback (D-18 scope-out)
    rating_agg = weighted_average_metrics(round_eval_metrics)
    for key in ("rmse", "mae", "eval_loss"):
        if key in rating_agg and key not in eval_metrics_history[round_num]:
            eval_metrics_history[round_num][key] = rating_agg[key]
    # D-27 best-round tracking
    current_ndcg = thesis_metrics.get("sampled_ndcg@10", 0.0) if thesis_metrics else 0.0
    if thesis_metrics and current_ndcg > best_metric:
        best_metric = current_ndcg
        best_round_num = round_num
        best_arrays = ArrayRecord({k: v.detach().clone() for k, v in arrays.to_torch_state_dict().items()})
    ```

    Step 11 — Best-round restore BEFORE centralized evaluation:
    ```python
    if checkpoint_rule in ("best_round_restore", "best_round") and best_round_num > 0:
        arrays = best_arrays
    ```

    Step 12 — Manifest + D-15 double-write at result-write time:
    ```python
    run_id = run_id_placeholder if run_id_placeholder else generate_run_id()
    index = verify_bundle(data_derived())
    split_manifest = load_split_manifest(data_derived() / "split_manifest.json")
    manifest = build_run_manifest(
        run_id=run_id,
        module="personalized",      # <-- Phase-3-specific
        mode_profile=profile,
        overrides=overrides,
        foundation_index=index,
        split_manifest=split_manifest,
        checkpoint_rule=checkpoint_rule,
    )
    results_data["federated_config"].update({
        "mode": mode, "run_seed": run_seed, "weight_policy": weight_policy,
        "checkpoint_rule": checkpoint_rule, "reuse_cache": reuse_cache_flag,
    })
    results_data["selected_clients_per_round"] = selected_clients_per_round
    results_data["checkpoint"] = {
        "rule": checkpoint_rule,
        "best_round": best_round_num,
        "best_sampled_ndcg@10": best_metric if best_metric > float("-inf") else None,
    }
    # D-13 cold-start fields in result JSON
    total_selections = sum(len(r) for r in selected_clients_per_round)
    results_data["cold_starts"] = {
        "per_round": cold_starts_per_round,
        "total_cold_starts": total_cold_starts,
        "total_client_selections": total_selections,
        "cold_start_rate": (total_cold_starts / total_selections) if total_selections else 0.0,
    }
    results_data = embed_manifest_in_result(manifest, results_data)
    results_filename = f"{run_id}_results.json"
    # ... existing result write code (preserve)
    write_manifest_sibling(manifest, results_filename)
    if wandb_run is not None:
        wandb_run.config.update({"_manifest": {
            "run_id": run_id, "mode": mode,
            "num_supernodes": expected_num_supernodes,
            "foundation_contract_sha256": index.foundation_contract_sha256,
            "split_hash": split_manifest.split_hash,
            "run_seed": run_seed, "checkpoint_rule": checkpoint_rule,
        }})
        wandb_run.summary["total_cold_starts"] = total_cold_starts
        wandb_run.summary["cold_start_rate"] = results_data["cold_starts"]["cold_start_rate"]
    ```

    Step 13 — Preserve verbatim (D-18): DummyClientProxy, weighted_average_metrics, print_evaluation_metrics, early-stopping setup/teardown, CUDA device fallback, get_model / load_full_data wiring, centralized eval block, final `wandb.run.summary` logging. Do NOT touch these code regions — the plan's rip targets are strictly the rip-target lines above.

    Step 14 — Commit (--no-verify):
    ```
    git add federated-personalized-cf/federated_personalized_cf/server_app.py
    git commit --no-verify -m "feat(03-04): server_app cross-device migration + D-13 cold-start + D-02 guard (PSN-04, PSN-07)"
    ```
  </action>
  <acceptance_criteria>
    - `grep -c "from federated_personalized_cf.strategy import" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1
    - `grep -c "PersonalizedSplitFedAvg(" federated-personalized-cf/federated_personalized_cf/server_app.py` returns at least 1
    - `grep -c "PersonalizedSplitFedProx(" federated-personalized-cf/federated_personalized_cf/server_app.py` returns at least 1
    - `grep -c "resolve_mode_defaults(" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1
    - `grep -c "log_mode_and_overrides(" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1
    - `grep -c "server_rng(run_seed)" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1
    - `grep -c "build_run_manifest(" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1
    - `grep -c 'module="personalized"' federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1
    - `grep -c "embed_manifest_in_result(" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1
    - `grep -c "write_manifest_sibling(" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1
    - `grep -c "discover_only" federated-personalized-cf/federated_personalized_cf/server_app.py` returns at least 2 (ConfigRecord build + partition_to_node_id logic)
    - `grep -c "partition_to_node_id" federated-personalized-cf/federated_personalized_cf/server_app.py` returns at least 2
    - `grep -c "selected_clients_per_round" federated-personalized-cf/federated_personalized_cf/server_app.py` returns at least 3 (init + append + result JSON field)
    - `grep -c "cold_starts_per_round\|total_cold_starts\|cold_start" federated-personalized-cf/federated_personalized_cf/server_app.py` returns at least 4 (D-13)
    - `grep -cE "best_round_restore|best_metric|best_round_num|best_arrays" federated-personalized-cf/federated_personalized_cf/server_app.py` returns at least 5 (D-27 tracking)
    - `grep -cE "^import random$|random\\.sample\\(" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 0
    - `grep -c "NotImplementedError" federated-personalized-cf/federated_personalized_cf/server_app.py` returns at least 1 (D-02 cross-silo guard)
    - `grep -c "cross_silo_legacy" federated-personalized-cf/federated_personalized_cf/server_app.py` returns at least 1 (D-02 branch)
    - `grep -c "federated-cf-cross-device" federated-personalized-cf/federated_personalized_cf/server_app.py` returns 1 (W&B project switch)
    - `python -c "import ast; ast.parse(open('federated-personalized-cf/federated_personalized_cf/server_app.py').read()); print('syntax ok')"` prints `syntax ok`
    - `python -c "from federated_personalized_cf.server_app import app; print('import ok')"` prints `import ok`
    - D-18 scope: `git diff --stat federated-personalized-cf/federated_personalized_cf/client_app.py federated-personalized-cf/federated_personalized_cf/task.py federated-personalized-cf/federated_personalized_cf/dataset.py federated-personalized-cf/federated_personalized_cf/strategy.py federated-personalized-cf/federated_personalized_cf/models/ federated-personalized-cf/pyproject.toml` returns empty after this commit
  </acceptance_criteria>
  <done>server_app.py implements the mode-first cross-device bootstrap; discovery round + partition-id-space sampling live pre-loop; PersonalizedSplitFedAvg/FedProx drive aggregate_evaluate; D-13 cold-start counter tallied per round and in final results; D-15 double-write manifest written with module="personalized"; D-27 best-round restore wired; D-02 cross-silo guard raises NotImplementedError at startup.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: server_app integration tests — PSN-04 reproducibility, strategy wire-up, D-15 double-write, D-13 cold-start math, D-02 NotImplementedError</name>
  <files>federated-personalized-cf/tests/test_server_integration.py</files>
  <read_first>
    - federated-baseline-cf/tests/test_server_integration.py (5-test TEMPLATE; adapt strategy class names)
    - federated-personalized-cf/federated_personalized_cf/server_app.py (POST-TASK-1 — observe where the D-13 cold-start counter is built; the integration test constructs a fake cache dir and asserts the counter math)
    - scripts/foundation/fedrec_foundation/rng.py (server_rng interface)
    - scripts/foundation/fedrec_foundation/manifest.py (manifest dataclass shape — what fields the roundtrip test asserts are present)
    - scripts/foundation/fedrec_foundation/mode.py (ModeProfile dataclass — fields asserted in manifest test)
  </read_first>
  <behavior>
    Tests to write (GREEN on first run):
    - test_server_rng_reproducible_per_round_selection: Two `server_rng(42)` instances produce byte-identical 3-round composite sequences (rng.sample(range(6040), 50) x3). Same as baseline.
    - test_server_rng_different_seeds_different_selections: `server_rng(42)` vs `server_rng(43)` give different selections (negative guard).
    - test_personalized_split_fedavg_aggregate_evaluate_sum_not_average: Instantiate PersonalizedSplitFedAvg(fraction_fit=0.1); call aggregate_evaluate with 2 synthetic clients (1-hit-on-1-user vs 0-hits-on-99-users via fake_evaluate_res fixture); assert sampled_hr@10 ≈ 1/100 = 0.01, NOT 0.5.
    - test_build_run_manifest_module_personalized: Call build_run_manifest with module="personalized" + resolve_mode_defaults("benchmark_cross_device") + real foundation index + split_manifest; assert manifest.module == "personalized" and all 4 IMP-2 fingerprints (mapping_sha256, split_hash, exclusion_sha256, foundation_contract_sha256) are present.
    - test_cold_start_counter_math (uses tmp_path): Create .embedding_cache/r1/ dir with partition_{0,1,2}.pt files but NOT partition_{3,4,5}.pt; call the module-level cold-start helper (or replicate its logic inline) with selected_pids = [0, 1, 2, 3, 4, 5] and run_id="r1"; assert cold_count == 3.
    - test_cross_silo_legacy_mode_raises_not_implemented (optional, tricky because @app.main() needs a Grid — can be skipped or replaced with an import-time sentinel check): read server_app.py source; assert it contains `raise NotImplementedError` inside a `if mode == "cross_silo_legacy":` branch (string assertion on the source is sufficient as a regression guard).
  </behavior>
  <action>
    Step 1 — Create federated-personalized-cf/tests/test_server_integration.py (MIRROR federated-baseline-cf/tests/test_server_integration.py with strategy class names substituted). Apply `pytestmark = pytest.mark.skipif(not (<repo>/data/derived/foundation_index.json).exists(), reason="foundation bundle not committed")` at the top.

    Step 2 — Write 5-6 tests listed in <behavior>. For test_cold_start_counter_math, use tmp_path fixture + monkeypatch Path(".embedding_cache") to resolve under tmp_path (or write the test to accept a cache_root argument to a helper extracted from server_app.py):
    ```python
    def test_cold_start_counter_math(tmp_path):
        cache_root = tmp_path / ".embedding_cache" / "r1"
        cache_root.mkdir(parents=True)
        for pid in (0, 1, 2):
            (cache_root / f"partition_{pid}.pt").write_bytes(b"\x00")
        selected_pids = [0, 1, 2, 3, 4, 5]
        cold_count = sum(1 for pid in selected_pids if not (cache_root / f"partition_{int(pid)}.pt").exists())
        assert cold_count == 3, f"expected 3 cold starts, got {cold_count}"
        assert sum(1 for pid in selected_pids if (cache_root / f"partition_{int(pid)}.pt").exists()) == 3
    ```

    Step 3 — For test_cross_silo_legacy_mode_raises_not_implemented: source-level regression guard (no grid needed):
    ```python
    def test_cross_silo_legacy_mode_raises_not_implemented():
        src = Path(__file__).resolve().parents[1].joinpath(
            "federated_personalized_cf", "server_app.py"
        ).read_text()
        assert 'cross_silo_legacy' in src
        assert "raise NotImplementedError" in src
        # Confirm the raise is inside a cross_silo_legacy branch (proximity guard)
        cross_silo_idx = src.index('cross_silo_legacy')
        nearby = src[cross_silo_idx:cross_silo_idx + 400]
        assert "NotImplementedError" in nearby, "D-02 guard must raise NotImplementedError close to the cross_silo_legacy check"
    ```

    Step 4 — Verify: `cd federated-personalized-cf && pytest tests/test_server_integration.py -v` → 5 or 6 passed. Full suite: `pytest tests/ -v` → ~28-29 passed.

    Step 5 — Commit (--no-verify):
    ```
    git add federated-personalized-cf/tests/test_server_integration.py
    git commit --no-verify -m "test(03-04): server integration tests — PSN-04 reproducibility + D-15 + D-13 + D-02 guard"
    ```
  </action>
  <acceptance_criteria>
    - `cd federated-personalized-cf && pytest tests/test_server_integration.py -v` exits 0 with at least "5 passed" (6 if the D-02 source-level guard counts separately)
    - `cd federated-personalized-cf && pytest tests/ -v` exits 0 with at least 28 tests passing overall
    - `grep -c "test_server_rng_reproducible_per_round_selection" federated-personalized-cf/tests/test_server_integration.py` returns 1
    - `grep -c "test_personalized_split_fedavg_aggregate_evaluate_sum_not_average" federated-personalized-cf/tests/test_server_integration.py` returns 1
    - `grep -c "test_build_run_manifest_module_personalized" federated-personalized-cf/tests/test_server_integration.py` returns 1
    - `grep -c "test_cold_start_counter_math" federated-personalized-cf/tests/test_server_integration.py` returns 1
    - `grep -c "test_cross_silo_legacy_mode_raises_not_implemented" federated-personalized-cf/tests/test_server_integration.py` returns 1
    - D-18 scope: only tests/test_server_integration.py modified by this task
  </acceptance_criteria>
  <done>5-6 GREEN integration tests prove: (a) PSN-04 seeded sampling reproducibility, (b) PersonalizedSplitFedAvg sum-not-average aggregation, (c) D-15 manifest integration with module="personalized", (d) D-13 cold-start counter arithmetic, (e) D-02 cross-silo NotImplementedError source-level guard.</done>
</task>

</tasks>

<verification>
- `cd federated-personalized-cf && pytest tests/ -v` exits 0 with at least 28 tests passing (accumulated from Plans 01/02/03/04)
- `python -c "import ast; ast.parse(open('federated-personalized-cf/federated_personalized_cf/server_app.py').read()); print('syntax ok')"` prints `syntax ok`
- `python -c "from federated_personalized_cf.server_app import app; print('import ok')"` prints `import ok`
- `python scripts/run.py --dry-run personalized benchmark_cross_device 2>&1 | grep -c "num-supernodes=6040"` returns at least 1
- `grep -cE "^import random$|random\\.sample\\(|random\\.seed\\(" federated-personalized-cf/federated_personalized_cf/server_app.py federated-personalized-cf/federated_personalized_cf/client_app.py federated-personalized-cf/federated_personalized_cf/task.py` returns 0 (module-wide stdlib random eradication)
- D-18 scope: `git diff --stat federated-personalized-cf/federated_personalized_cf/strategy.py federated-personalized-cf/federated_personalized_cf/dataset.py federated-personalized-cf/federated_personalized_cf/client_app.py federated-personalized-cf/federated_personalized_cf/task.py federated-personalized-cf/federated_personalized_cf/models/ federated-personalized-cf/pyproject.toml` returns empty after this plan's 2 commits (Plans 01/02/03 own those)
</verification>

<success_criteria>
- PSN-04 observable: `python scripts/run.py personalized benchmark_cross_device` samples 6040 supernodes via `_server_sampler.sample(range(6040), k)` in deterministic partition-id space; two runs with the same run-seed produce byte-identical selected_clients_per_round (proven via the Wave-3 subprocess determinism test in Plan 05).
- PSN-07 observable: Result JSON gains a top-level `_manifest` key with module="personalized" + all Phase-1 fingerprints; sibling `{run_id}-manifest.json` file exists beside the result JSON.
- D-13 cold-start counter is first-class: `results_data["cold_starts"] = {per_round, total_cold_starts, total_client_selections, cold_start_rate}`; W&B logs `round/cold_starts` per round and `total_cold_starts` / `cold_start_rate` to summary.
- D-02 frozen cross-silo: `mode="cross_silo_legacy"` at startup raises NotImplementedError with explicit reference to pre-Phase-3 commits; the guard fires BEFORE any training or data load.
- D-18 surgical discipline: pre-existing WIP in DummyClientProxy / weighted_average_metrics / print_evaluation_metrics / centralized-eval block is preserved verbatim; only the plan's rip targets are modified.
</success_criteria>

<output>
After completion, create `.planning/phases/03-personalized-migration/03-personalized-migration-04-SUMMARY.md` with: file list (1 modified + 1 created), decisions made (D-02 guard placement, W&B project default, D-13 counter algorithm), deviations (any auto-fixes logged), test counts (~5-6 new → ~28 total suite), commit SHAs, PSN-04 + PSN-07 closure notes, Plan 05 (subprocess determinism regression guard) readiness confirmation.
</output>
