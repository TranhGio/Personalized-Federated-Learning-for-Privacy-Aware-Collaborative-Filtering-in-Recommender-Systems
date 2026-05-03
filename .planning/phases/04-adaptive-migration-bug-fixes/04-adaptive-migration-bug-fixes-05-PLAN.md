---
phase: 04-adaptive-migration-bug-fixes
plan: 05
type: execute
subsystem: infra
tags: [server-app, mode-resolver, seeded-sampling, adaptive-split-fedavg, adaptive-split-fedprox, run-manifest, best-round-restore, best-prototype-snapshot, best-prototype-restore, discovery-round, partition-id-space, cold-start-counter, alpha-diagnostics, cross-device, adp-03, adp-06, adp-08, d-02, d-05, d-06, d-07, d-13, d-15, d-16, d-18, d-25, d-26, d-27, wave-3]
wave: 3
depends_on: [04-adaptive-migration-bug-fixes-01, 04-adaptive-migration-bug-fixes-02, 04-adaptive-migration-bug-fixes-03]
files_modified:
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py
  - federated-adaptive-personalized-cf/tests/test_server_integration.py
autonomous: true
requirements: [ADP-03, ADP-06, ADP-08]

must_haves:
  truths:
    - "server_app.py @app.main() resolves a ModeProfile via fedrec_foundation.mode.resolve_mode_defaults(mode) at startup; every hyperparameter read is `int(context.run_config.get(key, profile.field))` / `float(...)` / `str(...)` so the profile is canonical and pyproject values are the override surface (D-25)."
    - "`_server_sampler = server_rng(run_seed)` is instantiated ONCE before the FL loop; discovery round broadcasts @app.evaluate with discover_only=true to every grid.get_node_ids() entry BEFORE round 1 to build partition_to_node_id: Dict[int, int]; _server_sampler.sample(range(num_supernodes), k) runs in partition-id space; selected_clients_per_round stores stable partition_ids 0..N-1 (G-03-01 carry-forward)."
    - "`strategy = AdaptiveSplitFedAvg(...)` or `AdaptiveSplitFedProx(...)` (from Plan 01) replaces the existing raw SplitFedAvg / FedAvg instantiation; strategy.aggregate_evaluate is called per round with wrapped EvaluateRes tuples and populates eval_metrics_history[round_num] with sum-based thesis metrics; strategy.aggregate_fit is the override that runs super().aggregate_fit THEN updates _global_prototype."
    - "D-27 in-memory best-round restore (ADP-03 in-memory half): best_metric / best_round_num / best_arrays tracked inside the FL loop on checkpoint_rule ∈ ('best_round_restore', 'best_round'); at the SAME moment best_arrays is captured, call `strategy.snapshot_best_prototype(round_num=round_num, embedding_dim=embedding_dim)` so strategy.best_prototype mirrors best_arrays (D-05)."
    - "D-07 best-prototype-restore before final broadcast: at end-of-training BEFORE the final centralized-eval/broadcast, set `arrays = best_arrays` AND `strategy._global_prototype = strategy.best_prototype` (when both are non-None). Clients receiving the final train_config_dict['global_prototype'] see the RESTORED prototype — not the last-round drift."
    - "D-15 double-write manifest: build_run_manifest called once with module=\"adaptive\"; embed_manifest_in_result injects _manifest into results_data; write_manifest_sibling writes <run_id>-manifest.json beside the result JSON."
    - "D-06 best_prototype embedded in result JSON: after embed_manifest_in_result runs, explicitly mutate `results_data['_manifest']['best_prototype']` = `strategy.best_prototype.tolist()` when non-None, else `None`. Payload is ~4 KB at dim=128 (negligible). ADP-08 'full protocol fingerprint' literally includes the best-round prototype for post-hoc verification."
    - "D-13 cold-start counter (carry-forward from Phase 3): pre-round check of `.embedding_cache/{run_id}/partition_{pid}.pt` existence (or sig_{hash}/partition_{pid}.pt under reuse-cache=true) for each selected partition_id → cold_starts_this_round; accumulated as total_cold_starts; logged per-round to W&B as round/cold_starts; reported in final results JSON as {per_round, total, rate}."
    - "D-16 alpha diagnostics per-round: when any selected client returned alpha_diagnostics in its FitMetricsContract sidecar (see Plan 03), the server weighted-averages alpha_mean/alpha_std/alpha_p25/alpha_p50/alpha_p75/alpha_clip_hit_rate by num_examples across contributing clients and writes the aggregate into eval_metrics_history[round_num] + `wandb_run.log({'round/alpha_*': ...}, step=round_num)`."
    - "Default W&B project federated-cf-cross-device for benchmark_cross_device mode (PROJECT.md constraint); legacy cross_silo_legacy stays on the existing project; explicit run_config['wandb-project'] still wins."
    - "D-02 benchmark-mode guard: if `mode == \"cross_silo_legacy\"` at startup → raise NotImplementedError pointing at pre-Phase-4 commit (adaptive cross-silo is frozen per D-02; mirrors Phase 3 Plan 04)."
    - "6 GREEN integration tests covering ADP-06 reproducibility (server_rng byte-identity), AdaptiveSplitFedAvg wire-up (sum-not-average), D-15 double-write roundtrip with module='adaptive', D-05+D-06 best_prototype snapshot + embedded-in-manifest, D-13 cold-start counter math, D-02 NotImplementedError source-level guard."
  artifacts:
    - path: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py"
      provides: "Cross-device server main loop with mode resolver, discovery round, seeded partition-id-space sampling, AdaptiveSplitFedAvg wire-up, D-05 best_prototype snapshot, D-07 best_prototype restore before final broadcast, D-06 best_prototype embedded in result JSON, D-13 cold-start counter, D-15 manifest double-write, D-16 alpha diagnostics aggregate, D-27 best-round restore, D-02 guard"
    - path: "federated-adaptive-personalized-cf/tests/test_server_integration.py"
      provides: "6 GREEN integration tests: server_rng byte-identity, AdaptiveSplitFedAvg sum aggregation, D-15 + module='adaptive' + best_prototype embedded, D-13 cold-start counter math, D-02 source-level NotImplementedError guard, D-07 snapshot_best_prototype sequence"
  key_links:
    - from: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py"
      to: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py::AdaptiveSplitFedAvg"
      via: "strategy instantiation with fraction_fit + prototype_momentum + AdaptiveSplitFedAvg constructor; strategy.aggregate_evaluate called per round; strategy.snapshot_best_prototype called at best-metric fire"
      pattern: "AdaptiveSplitFedAvg|AdaptiveSplitFedProx|snapshot_best_prototype"
    - from: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py"
      to: "fedrec_foundation.manifest.build_run_manifest"
      via: "called once with module=\"adaptive\" + overrides + foundation fingerprints at result write time"
      pattern: "build_run_manifest"
    - from: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py"
      to: ".embedding_cache/{run_id}/partition_{pid}.pt"
      via: "D-13 cold-start check: Path exists() BEFORE the round sends train message, accumulated as cold_starts_this_round"
      pattern: "cold_start"
    - from: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py"
      to: "results/federated/adaptive/<run_id>_results.json + <run_id>-manifest.json"
      via: "D-15 double-write with module='adaptive' and best_prototype embedded in _manifest dict (D-06)"
      pattern: "embed_manifest_in_result|write_manifest_sibling|best_prototype"
---

<objective>
Migrate federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py to the Phase-4 cross-device contract, cloning Phase 3 Plan 04 verbatim with 4 adaptive-specific additions:

1. **Strategy swap** — Replace existing `SplitFedAvg` / `SplitFedProx` (from the pre-Phase-4 strategy.py) with `AdaptiveSplitFedAvg` / `AdaptiveSplitFedProx` (from Plan 01's strategy.py). The new classes are drop-in replacements with the same constructor surface PLUS a `best_prototype` field and `snapshot_best_prototype` method.

2. **D-05 best_prototype snapshot** — At the same moment D-27 snapshots `best_arrays` (when current_ndcg > best_metric), ALSO call `strategy.snapshot_best_prototype(round_num, embedding_dim)` so server-side state is symmetrized.

3. **D-07 best_prototype restore** — At end-of-training BEFORE the final centralized-eval/broadcast, in addition to `arrays = best_arrays`, ALSO set `strategy._global_prototype = strategy.best_prototype`. The final broadcast's train_config_dict['global_prototype'] then carries the RESTORED prototype — not the last-round drift.

4. **D-06 best_prototype embedded in result JSON** — After `embed_manifest_in_result`, explicitly assign `results_data['_manifest']['best_prototype'] = strategy.best_prototype.tolist() if not None else None`. This makes ADP-08's "full protocol fingerprint" literal about which prototype was reported.

Plus the standard Phase-3-Plan-04 carry-forward: mode resolver (D-25), discovery round + partition_to_node_id build (G-03-01), seeded partition-id sampling (ADP-06 server half), D-27 best-round restore, D-15 double-write with module='adaptive', D-13 cold-start counter, D-02 NotImplementedError for cross-silo, W&B project switch to federated-cf-cross-device.

Plus one Phase-4-unique addition: **D-16 alpha diagnostics aggregate** — server weighted-averages the alpha_mean/alpha_std/alpha_p25/alpha_p50/alpha_p75/alpha_clip_hit_rate sidecar dicts (populated client-side by Plan 03) across contributing clients and logs to W&B + eval_metrics_history[round_num].

Purpose: Closes ADP-03 (in-memory best-round restore for prototype — primary Phase 4 ADP-03 delta over Phase 3), ADP-06 (server half: seeded sampling + sufficient-stat aggregator + run-scoped cache), and ADP-08 (protocol fingerprint + best_prototype embedded). After this plan, `python scripts/run.py adaptive benchmark_cross_device` produces a reproducible cross-device adaptive run end-to-end with a protocol-fingerprinted result artifact that literally includes the best-round prototype.

Output:
- federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py (migrated; D-18 surgical edits preserve pre-existing WIP for DummyClientProxy / weighted_average_metrics / print_evaluation_metrics / centralized-eval path / AlphaAnalyzer integration if used by server)
- federated-adaptive-personalized-cf/tests/test_server_integration.py (new — 6 GREEN tests)
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/04-adaptive-migration-bug-fixes/04-CONTEXT.md
@.planning/phases/04-adaptive-migration-bug-fixes/04-RESEARCH.md
@.planning/phases/03-personalized-migration/03-personalized-migration-04-SUMMARY.md
@.planning/phases/02-baseline-migration/02-baseline-migration-04-SUMMARY.md
@.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-01-PLAN.md
@.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-03-PLAN.md

<interfaces>
<!-- Phase 4 Plan 01 AdaptiveSplitFedAvg/Prox contract -->
```python
# federated_adaptive_personalized_cf.strategy (post Plan 01)
class AdaptiveSplitFedAvg(BaseFedAvg):
    def __init__(self, fraction_fit: float = 1.0, prototype_momentum: float = 0.9, **kwargs): ...
    # Inherits: aggregate_fit, aggregate_evaluate, get_global_prototype, _aggregate_prototypes
    self.best_prototype: Optional[np.ndarray]   # Plan 01 field (ADP-03 D-05)
    def snapshot_best_prototype(self, round_num: int, embedding_dim: int) -> None: ...
    # Snapshots self._global_prototype; falls back to np.zeros(embedding_dim) + WARNING (D-08)

class AdaptiveSplitFedProx(BaseFedProx):
    def __init__(self, fraction_fit: float = 1.0, prototype_momentum: float = 0.9,
                 proximal_mu: float = 0.01, **kwargs): ...
    # Same contract as AdaptiveSplitFedAvg
```

<!-- Phase 1 foundation APIs (unchanged) -->
```python
# fedrec_foundation.mode
def resolve_mode_defaults(mode: str) -> ModeProfile: ...
def log_mode_and_overrides(mode: str, profile: ModeProfile, run_config: Dict) -> Dict[str, Any]: ...

# fedrec_foundation.rng
def server_rng(run_seed: int) -> random.Random: ...

# fedrec_foundation.manifest
def build_run_manifest(*, run_id, module, mode_profile, overrides, foundation_index,
                       split_manifest, checkpoint_rule) -> RunManifest: ...
def embed_manifest_in_result(manifest: RunManifest, results_data: Dict) -> Dict: ...
def write_manifest_sibling(manifest: RunManifest, results_filename: str) -> None: ...
def generate_run_id() -> str: ...

# fedrec_foundation.bundle
def verify_bundle(data_dir: Path) -> FoundationIndex: ...

# fedrec_foundation.split
def load_split_manifest(path: Path) -> SplitManifest: ...

# fedrec_foundation.paths
def data_derived() -> Path: ...
```

<!-- Client payload shape (post Plan 03) that server-side aggregator consumes -->
```python
# FitMetricsContract sidecar alpha_diagnostics (D-16): a separate dict in the Message content
# (not inside FitMetricsContract itself — D-21 contract rejects free-form extras). Exact key
# name: "alpha_diagnostics". Dict has 6 floats: alpha_mean, alpha_std, alpha_p25, alpha_p50,
# alpha_p75, alpha_clip_hit_rate. Empty/missing when enable-per-user-alpha=false.
```

<!-- Phase 3 server_app.py canonical template -->
<!-- See federated-personalized-cf/federated_personalized_cf/server_app.py for the
     Phase-3 implementation pattern that this plan mirrors verbatim with the 4
     Phase-4 additions + 2 Phase-4 unique additions (alpha diagnostics + prototype snapshot). -->
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: server_app.py migration — mode resolver + discovery round + partition-id sampling + AdaptiveSplitFedAvg wire-up + D-05/D-07 best_prototype snapshot+restore + D-06 embedded in manifest + D-13 cold-start counter + D-15 manifest + D-16 alpha diagnostics + D-27 best-round + D-02 guard (ADP-03, ADP-06, ADP-08)</name>
  <files>federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py</files>
  <read_first>
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py (ENTIRE FILE — inventory DummyClientProxy, weighted_average_metrics, early-stopping setup/teardown, centralized eval block, wandb init, existing random.sample(node_ids,...) call site if any, existing SplitFedAvg/SplitFedProx/FedAvg/FedProx instantiation, AlphaAnalyzer integration if present)
    - federated-personalized-cf/federated_personalized_cf/server_app.py (CANONICAL Phase-3 TEMPLATE — post-Plan-04 shape: mode resolver block, W&B project switch, strategy instantiation, discovery round broadcast, partition_to_node_id build, _server_sampler.sample(range(N), k), selected_clients_per_round = partition_ids, D-27 best-round, D-15 double-write, D-13 cold-start counter)
    - federated-personalized-cf/tests/test_server_integration.py (TEMPLATE — adapt 5 test names + substitute strategy class names; add the 6th Phase-4-specific test for D-05/D-06 best_prototype)
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py (POST-PLAN-01 — AdaptiveSplitFedAvg + AdaptiveSplitFedProx + snapshot_best_prototype interface)
    - scripts/foundation/fedrec_foundation/manifest.py (build_run_manifest signature; embed_manifest_in_result mutation pattern; the _manifest dict IS extensible — Research §Code Examples confirms post-hoc mutation is safe)
    - scripts/foundation/fedrec_foundation/mode.py (ModeProfile + resolve_mode_defaults + log_mode_and_overrides)
    - scripts/foundation/fedrec_foundation/rng.py (server_rng factory — sampler interface `.sample(domain, k)`)
    - scripts/foundation/fedrec_foundation/bundle.py (verify_bundle, load_split_manifest, FoundationIndex)
    - scripts/foundation/fedrec_foundation/paths.py (data_derived())
    - .planning/phases/04-adaptive-migration-bug-fixes/04-CONTEXT.md §D-05, D-06, D-07 (best_prototype snapshot at best round; embed in manifest; restore before final broadcast)
    - .planning/phases/04-adaptive-migration-bug-fixes/04-RESEARCH.md §"Pattern 2: Best-Round Prototype Snapshot" (lines ~282-342 — ready-to-paste code)
    - .planning/phases/03-personalized-migration/03-personalized-migration-04-SUMMARY.md (G-03-01 discovery round step-by-step; D-13 cold-start counter; D-02 cross-silo guard; D-25 hyperparameter shape)
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/evaluation/alpha_analysis.py (if the existing server integrates AlphaAnalyzer, preserve its call per D-18; Research §Pattern 5 recommends adding a `compute_scalar_summary` method for per-round D-16 if not already exposed — but client-side already computes the 6 scalars per Plan 03, so the server just aggregates them)
  </read_first>
  <action>
    Step 1 — Inventory pre-existing WIP with `git diff federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py`. Record which code blocks are D-18 pre-existing and must be preserved verbatim. Typical preserves: DummyClientProxy class, weighted_average_metrics helper (for RMSE/MAE — retain for D-18 scope-out), print_evaluation_metrics, early-stopping setup/teardown, CUDA device fallback, load_full_data call for centralized eval, final wandb.run.summary logging, AlphaAnalyzer integration (if present).

    Step 2 — Add/update imports:
    ```python
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Tuple

    import numpy as np

    from flwr.common import (
        ArrayRecord, Code, ConfigRecord, EvaluateRes, Message, MessageType,
        MetricRecord, RecordDict, Status,
    )
    from flwr.server.client_proxy import ClientProxy

    from federated_adaptive_personalized_cf.strategy import (
        AdaptiveSplitFedAvg, AdaptiveSplitFedProx, USER_PROTOTYPE_KEY,
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
    REMOVE any top-level `import random` (grep + strip).

    Step 3 — Insert the D-25 mode resolver block near the top of @app.main() (before any hyperparameter reads):
    ```python
    # ==== D-25 mode resolver ====
    mode = str(context.run_config.get("mode", "cross_silo_legacy"))
    run_seed = int(context.run_config.get("run-seed", 42))
    profile = resolve_mode_defaults(mode)
    overrides = log_mode_and_overrides(mode, profile, context.run_config)

    # ==== D-02 cross-silo guard ====
    # Adaptive cross-device migration froze cross-silo mode per D-02; pre-Phase-4 commits
    # are the reproduction oracle.
    if mode == "cross_silo_legacy":
        raise NotImplementedError(
            "Adaptive cross-device migration removed multi-user-per-client support per D-02. "
            "Check out a pre-Phase-4 commit (see .planning/phases/04-adaptive-migration-bug-fixes/04-CONTEXT.md §Deferred) "
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
    checkpoint_rule = str(context.run_config.get(
        "checkpoint-rule",
        getattr(profile, "checkpoint_rule", "best_round_restore"),
    ))
    reuse_cache_flag = bool(context.run_config.get("reuse-cache", False))
    prototype_momentum = float(context.run_config.get("prototype-momentum", 0.9))
    ```
    Preserve all pre-existing hyperparameter reads (e.g., mlp_hidden_dims, fusion_type, alpha_method) but route them through the same D-25 pattern.

    Step 5 — W&B project switch (PROJECT.md constraint, mirrors Phase 3):
    ```python
    default_project = "federated-cf-cross-device" if mode == "benchmark_cross_device" else "federated-adaptive-personalized-cf"
    wandb_project = str(context.run_config.get("wandb-project", default_project))
    wandb_config = {
        "mode": mode, "run_seed": run_seed, "weight_policy": weight_policy,
        "partition_mode": str(context.run_config.get("partition-mode", "natural")),
        "checkpoint_rule": checkpoint_rule, "reuse_cache": reuse_cache_flag,
        # ... preserve existing wandb_config contents (mlp-hidden-dims, fusion-type,
        # alpha-method, enable-per-user-alpha, enable-item-perturbation, etc.)
    }
    ```

    Step 6 — Strategy instantiation. Replace the existing `SplitFedAvg(...)` / `SplitFedProx(...)` / `FedAvg(...)` / `FedProx(...)` with:
    ```python
    strategy_name = str(context.run_config.get("strategy", "fedprox")).lower()
    if strategy_name == "fedprox":
        proximal_mu = float(context.run_config.get("proximal-mu", 0.01))
        strategy = AdaptiveSplitFedProx(
            fraction_fit=fraction_train,
            fraction_evaluate=fraction_train,
            prototype_momentum=prototype_momentum,
            proximal_mu=proximal_mu,
        )
    else:
        strategy = AdaptiveSplitFedAvg(
            fraction_fit=fraction_train,
            fraction_evaluate=fraction_train,
            prototype_momentum=prototype_momentum,
        )
    ```

    Step 7 — Discovery round (G-03-01). Insert BEFORE the main FL loop. Use Phase 3 pattern verbatim; the ConfigRecord dispatch key must match the client-side `msg.content.get("train_config", {})` probe from Plan 03:
    ```python
    # ==== G-03-01 discovery round: build partition_id -> node_id map ====
    expected_num_supernodes = int(context.run_config.get("num-supernodes", 6040))
    all_node_ids = list(grid.get_node_ids())
    assert len(all_node_ids) == expected_num_supernodes, \
        f"Discovery pre-check: got {len(all_node_ids)} supernodes, expected {expected_num_supernodes}"
    discovery_config = ConfigRecord({"discover_only": True})
    discovery_messages = [
        Message(
            content=RecordDict({"train_config": discovery_config}),
            message_type=MessageType.EVALUATE,
            dst_node_id=nid,
            group_id=str(0),
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
    assert not missing, (
        f"Discovery round missed partitions: {sorted(list(missing))[:10]}"
        f"{'...' if len(missing) > 10 else ''}"
    )
    ```

    Step 8 — Pre-loop state init (Phase 4 adds best_metric / best_round_num / best_arrays tracking mirroring Phase 3, AND pre-computes run_id early so client+server cache paths coincide):
    ```python
    run_id = str(context.run_config.get("run-id", "")) or generate_run_id()
    _server_sampler = server_rng(run_seed)
    selected_clients_per_round: List[List[int]] = []
    best_metric: float = float("-inf")
    best_round_num: int = 0
    best_arrays = arrays  # fallback if no round improves
    total_cold_starts: int = 0
    cold_starts_per_round: List[int] = []
    eval_metrics_history: Dict[int, Dict[str, Any]] = {}
    alpha_diagnostics_history: Dict[int, Dict[str, float]] = {}
    ```

    Step 9 — Per-round client sampling (partition-id space) + D-13 cold-start counter:
    ```python
    num_selected = max(1, int(round(expected_num_supernodes * fraction_train)))
    selected_pids = list(_server_sampler.sample(range(expected_num_supernodes), num_selected))
    selected_clients_per_round.append([int(p) for p in selected_pids])
    selected_node_ids = [partition_to_node_id[pid] for pid in selected_pids]

    # ==== D-13 cold-start counter (mirrors Phase 3 Plan 04) ====
    if reuse_cache_flag:
        # For reuse-cache=true, cache dir is sig_{hash}/. The server cannot construct the
        # hash without knowing the full 12-field v2 signature; skip the per-round cold count
        # and emit total_cold_starts=0 with a documented caveat in the result JSON.
        cold_count = 0
    else:
        cache_root = Path(".embedding_cache") / run_id
        cold_count = sum(
            1 for pid in selected_pids
            if not (cache_root / f"partition_{int(pid)}.pt").exists()
        )
    cold_starts_per_round.append(cold_count)
    total_cold_starts += cold_count
    if wandb_run is not None:
        wandb_run.log({
            "round/selected_clients": [int(p) for p in selected_pids],
            "round/cold_starts": cold_count,
        }, step=round_num)
    ```

    Step 10 — Train message build + broadcast. Preserve existing train-message assembly logic. Ensure the config broadcast to clients includes `global_prototype = strategy.get_global_prototype().tolist()` (if non-None) so clients receive the EMA on round N. The train_config dict build lives in the existing server code — do not rip it; just ensure it reads from `strategy.get_global_prototype()` (accessor from Plan 01) instead of the old `strategy._global_prototype` attribute.

    Step 11 — Evaluation aggregation. Wrap each eval response into `EvaluateRes(status=Status(Code.OK, "ok"), loss=metrics_dict.get("eval_loss", 0.0), num_examples=num_examples, metrics=metrics_dict)` where num_examples falls back through `num_training_examples → evaluated_users → num-examples → 1` (Phase 2 Plan 04 pattern). Then:
    ```python
    strategy_loss, thesis_metrics = strategy.aggregate_evaluate(round_num, eval_results, [])
    eval_metrics_history[round_num] = dict(thesis_metrics or {})
    # Preserve RMSE/MAE via legacy weighted_average_metrics fallback (D-18 scope-out)
    rating_agg = weighted_average_metrics(round_eval_metrics)
    for key in ("rmse", "mae", "eval_loss"):
        if key in rating_agg and key not in eval_metrics_history[round_num]:
            eval_metrics_history[round_num][key] = rating_agg[key]

    # ==== D-16 alpha diagnostics aggregate (Phase-4 unique) ====
    # Client-side FitRes sidecar carries alpha_diagnostics dict. Server weighted-averages
    # the 6 scalar fields across contributing clients by num_examples.
    alpha_contributions: List[Tuple[Dict[str, float], int]] = []
    for _proxy, fit_res in fit_results:
        metrics = fit_res.metrics or {}
        ad = metrics.get("alpha_diagnostics")
        if isinstance(ad, dict) and ad:
            alpha_contributions.append((ad, int(fit_res.num_examples)))
    if alpha_contributions:
        total_w = sum(w for _, w in alpha_contributions)
        agg: Dict[str, float] = {}
        for key in ("alpha_mean", "alpha_std", "alpha_p25", "alpha_p50", "alpha_p75", "alpha_clip_hit_rate"):
            agg[key] = sum(ad.get(key, 0.0) * w for ad, w in alpha_contributions) / total_w
        alpha_diagnostics_history[round_num] = agg
        eval_metrics_history[round_num].update(
            {f"alpha/{k}": v for k, v in agg.items()}
        )
        if wandb_run is not None:
            wandb_run.log({f"round/alpha/{k}": v for k, v in agg.items()}, step=round_num)

    # ==== D-27 best-round tracking + D-05 best_prototype snapshot ====
    current_ndcg = float(thesis_metrics.get("sampled_ndcg@10", 0.0)) if thesis_metrics else 0.0
    if thesis_metrics and current_ndcg > best_metric:
        best_metric = current_ndcg
        best_round_num = round_num
        best_arrays = ArrayRecord(
            {k: v.detach().clone() for k, v in arrays.to_torch_state_dict().items()}
        )
        # D-05: snapshot prototype at the same moment
        strategy.snapshot_best_prototype(round_num=round_num, embedding_dim=embedding_dim)
        print(f"  [CHECKPOINT] New best sampled_ndcg@10={best_metric:.4f} at round {best_round_num}")
    ```

    Step 12 — D-27 + D-07 best-round restore BEFORE the centralized/final broadcast:
    ```python
    if checkpoint_rule in ("best_round_restore", "best_round") and best_round_num > 0:
        arrays = best_arrays
        # D-07: restore the best-round prototype BEFORE the next broadcast so clients see
        # the prototype that corresponds to best_arrays (not the last-round drift).
        if strategy.best_prototype is not None:
            strategy._global_prototype = strategy.best_prototype
    ```

    Step 13 — Manifest + D-15 double-write + D-06 best_prototype embedded at result-write time:
    ```python
    index = verify_bundle(data_derived())
    split_manifest = load_split_manifest(data_derived() / "split_manifest.json")
    manifest = build_run_manifest(
        run_id=run_id,
        module="adaptive",
        mode_profile=profile,
        overrides=overrides,
        foundation_index=index,
        split_manifest=split_manifest,
        checkpoint_rule=checkpoint_rule,
    )

    results_data["federated_config"].update({
        "mode": mode, "run_seed": run_seed, "weight_policy": weight_policy,
        "checkpoint_rule": checkpoint_rule, "reuse_cache": reuse_cache_flag,
        "prototype_momentum": prototype_momentum,
    })
    results_data["selected_clients_per_round"] = selected_clients_per_round
    results_data["checkpoint"] = {
        "rule": checkpoint_rule,
        "best_round": best_round_num,
        "best_sampled_ndcg@10": best_metric if best_metric > float("-inf") else None,
    }
    # D-13 cold-start fields
    total_selections = sum(len(r) for r in selected_clients_per_round)
    results_data["cold_starts"] = {
        "per_round": cold_starts_per_round,
        "total_cold_starts": total_cold_starts,
        "total_client_selections": total_selections,
        "cold_start_rate": (total_cold_starts / total_selections) if total_selections else 0.0,
    }
    # D-16 alpha diagnostics history
    if alpha_diagnostics_history:
        results_data["alpha_diagnostics_history"] = {
            int(r): {k: float(v) for k, v in d.items()}
            for r, d in alpha_diagnostics_history.items()
        }

    # D-15 double-write with module="adaptive"
    results_data = embed_manifest_in_result(manifest, results_data)

    # D-06: embed best_prototype in the _manifest dict AFTER embed_manifest_in_result mutates it.
    if strategy.best_prototype is not None:
        results_data["_manifest"]["best_prototype"] = [float(x) for x in strategy.best_prototype.tolist()]
    else:
        results_data["_manifest"]["best_prototype"] = None

    results_filename = f"{run_id}_results.json"
    # ... existing result-write code (preserve verbatim) ...
    write_manifest_sibling(manifest, results_filename)

    if wandb_run is not None:
        wandb_run.config.update({
            "_manifest": {
                "run_id": run_id, "mode": mode,
                "num_supernodes": expected_num_supernodes,
                "foundation_contract_sha256": index.foundation_contract_sha256,
                "split_hash": split_manifest.split_hash,
                "run_seed": run_seed, "checkpoint_rule": checkpoint_rule,
            }
        })
        wandb_run.summary["total_cold_starts"] = total_cold_starts
        wandb_run.summary["cold_start_rate"] = results_data["cold_starts"]["cold_start_rate"]
        if strategy.best_prototype is not None:
            wandb_run.summary["best_prototype_norm"] = float(np.linalg.norm(strategy.best_prototype))
    ```

    Step 14 — D-18 preserve verbatim: DummyClientProxy, weighted_average_metrics, print_evaluation_metrics, early-stopping setup/teardown, CUDA device fallback, get_model / load_full_data wiring, centralized eval block (if adaptive uses one — split learning typically can't, but preserve whatever the pre-Phase-4 code did), final `wandb.run.summary` logging, any AlphaAnalyzer integration. Do NOT touch these code regions.

    Step 15 — Commit (--no-verify):
    ```
    git add federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py
    git commit --no-verify -m "feat(04-05): server_app cross-device + AdaptiveSplitFedAvg + D-05/D-06/D-07 best_prototype + D-13 + D-15 + D-16 (ADP-03, ADP-06, ADP-08)"
    ```
  </action>
  <acceptance_criteria>
    - `grep -c "from federated_adaptive_personalized_cf.strategy import" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1
    - `grep -c "AdaptiveSplitFedAvg(" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 1
    - `grep -c "AdaptiveSplitFedProx(" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 1
    - `grep -c "SplitFedAvg(" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 0 (old constructor eradicated — verify the renamed class is referenced, not the old one)
    - `grep -c "resolve_mode_defaults(" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1
    - `grep -c "log_mode_and_overrides(" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1
    - `grep -c "server_rng(run_seed)" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1
    - `grep -c "build_run_manifest(" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1
    - `grep -c 'module="adaptive"' federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1
    - `grep -c "embed_manifest_in_result(" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1
    - `grep -c "write_manifest_sibling(" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1
    - `grep -c "discover_only" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 2
    - `grep -c "partition_to_node_id" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 2
    - `grep -c "selected_clients_per_round" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 3
    - `grep -cE "cold_starts_per_round|total_cold_starts|cold_start" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 4
    - `grep -cE "best_round_restore|best_metric|best_round_num|best_arrays" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 5
    - `grep -c "snapshot_best_prototype" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 1 (D-05 snapshot call at best-metric fire)
    - `grep -c "strategy._global_prototype = strategy.best_prototype" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1 (D-07 restore before final broadcast)
    - `grep -c "best_prototype" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 4 (snapshot call + restore + embedded in manifest + W&B summary)
    - `grep -c 'results_data\\["_manifest"\\]\\["best_prototype"\\]' federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 1 (D-06 embedded)
    - `grep -cE "alpha_diagnostics_history|alpha_diagnostics" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 3 (D-16 aggregate + history + W&B log)
    - `grep -c "alpha_clip_hit_rate" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 1 (D-16 diagnostic key list)
    - `grep -cE "^import random$|random\\.sample\\(|random\\.seed\\(" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 0
    - `grep -c "NotImplementedError" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 1 (D-02 cross-silo guard)
    - `grep -c "cross_silo_legacy" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns at least 1
    - `grep -c "federated-cf-cross-device" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` returns 1 (W&B project switch)
    - `python -c "import ast; ast.parse(open('federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py').read()); print('syntax ok')"` prints `syntax ok`
    - `python -c "from federated_adaptive_personalized_cf.server_app import app; print('import ok')"` prints `import ok`
    - D-18 scope: `git diff --stat federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/ federated-adaptive-personalized-cf/pyproject.toml` returns empty after this commit
  </acceptance_criteria>
  <done>server_app.py implements the Phase-4 mode-first cross-device bootstrap: discovery round + partition-id-space sampling live pre-loop; AdaptiveSplitFedAvg/FedProx drive aggregate_evaluate AND aggregate_fit (with prototype EMA); D-05 best_prototype snapshot fires at the same moment as D-27 best_arrays snapshot; D-07 restores both state pieces before the final broadcast; D-06 embeds best_prototype in the result JSON's _manifest; D-13 cold-start counter tallied per round and in final results; D-15 double-write manifest with module="adaptive"; D-16 alpha diagnostics aggregated and logged per-round + history; D-02 cross-silo guard raises NotImplementedError at startup.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: server_app integration tests — ADP-06 reproducibility, AdaptiveSplitFedAvg wire-up, D-15 + module='adaptive' + best_prototype embedded, D-05 snapshot sequence, D-13 cold-start math, D-02 NotImplementedError source guard</name>
  <files>federated-adaptive-personalized-cf/tests/test_server_integration.py</files>
  <read_first>
    - federated-personalized-cf/tests/test_server_integration.py (5-test TEMPLATE; Phase 4 extends with best_prototype snapshot + module='adaptive' + best_prototype embedded in _manifest)
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py (POST-TASK-1 — observe where the D-13 cold-start counter is built + where snapshot_best_prototype is called + where _manifest is mutated to embed best_prototype)
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py (POST-PLAN-01 — AdaptiveSplitFedAvg interface)
    - scripts/foundation/fedrec_foundation/rng.py (server_rng interface)
    - scripts/foundation/fedrec_foundation/manifest.py (manifest dataclass shape — what fields the roundtrip test asserts are present; confirm `module` is a RunManifest field)
    - scripts/foundation/fedrec_foundation/mode.py (ModeProfile dataclass — fields asserted in manifest test)
  </read_first>
  <behavior>
    6 tests (GREEN on first run; tests use tmp_path / pytest fixtures / source-level source assertions where a Grid-dependent path is untestable without a real Flower simulation):

    1. test_server_rng_reproducible_per_round_selection: Two `server_rng(42)` instances produce byte-identical 3-round composite sequences (`rng.sample(range(6040), 50)` × 3). Same-seed reproducibility.

    2. test_server_rng_different_seeds_different_selections: `server_rng(42)` vs `server_rng(43)` give different selections (negative-guard distinguishability).

    3. test_adaptive_split_fedavg_aggregate_evaluate_sum_not_average: Instantiate AdaptiveSplitFedAvg(fraction_fit=0.1); call aggregate_evaluate with 2 synthetic clients (1-hit-on-1-user vs 0-hits-on-99-users via fake_evaluate_res fixture); assert sampled_hr@10 ≈ 1/100 = 0.01, NOT 0.5 (per-client ratio double-average).

    4. test_build_run_manifest_module_adaptive_with_best_prototype (integration + source-level): Call build_run_manifest with module="adaptive" + resolve_mode_defaults("benchmark_cross_device") + real foundation index + split_manifest; assert manifest.module == "adaptive" and all 4 IMP-2 fingerprints (mapping_sha256, split_hash, exclusion_sha256, foundation_contract_sha256) present. Then SIMULATE the server-side mutation: `results_data = embed_manifest_in_result(manifest, {}); results_data["_manifest"]["best_prototype"] = [1.0] * 128; assert results_data["_manifest"]["best_prototype"] == [1.0] * 128` — this proves the _manifest dict is mutable post-embed (as Research §Pattern 2 asserts).

    5. test_cold_start_counter_math (tmp_path): Create .embedding_cache/r1/ dir with partition_{0,1,2}.pt files but NOT partition_{3,4,5}.pt; simulate the server-side cold-count helper: `cold_count = sum(1 for pid in [0,1,2,3,4,5] if not (cache_root / f"partition_{pid}.pt").exists())`; assert cold_count == 3.

    6. test_cross_silo_legacy_mode_raises_not_implemented (source-level regression): read server_app.py; assert contains `cross_silo_legacy` AND `raise NotImplementedError` in close proximity (within 400 chars); source-level regression guard.

    7. test_snapshot_best_prototype_called_inside_best_metric_branch (source-level regression): read server_app.py; assert the literal string `snapshot_best_prototype` appears AFTER the `best_metric = current_ndcg` line (proves D-05 snapshot fires at the same moment as D-27 best_arrays snapshot); source-level regression guard (full end-to-end integration requires a Grid which we can't instantiate cheaply).
  </behavior>
  <action>
    Step 1 — Create federated-adaptive-personalized-cf/tests/test_server_integration.py (MIRROR federated-personalized-cf/tests/test_server_integration.py with strategy class names substituted and the 2 Phase-4-specific tests added). Apply `pytestmark = pytest.mark.skipif(not (<repo>/data/derived/foundation_index.json).exists(), reason="foundation bundle not committed")` at the top:

    ```python
    """ADP-03 + ADP-06 + ADP-08 integration tests — server-side reproducibility, strategy wire-up,
    D-15 double-write with module='adaptive', D-05 best_prototype snapshot sequence, D-13 cold-start
    counter, D-02 cross-silo guard.

    Mirrors Phase 3 Plan 04 Task 2 with 2 Phase-4-specific tests (best_prototype sequence + module guard).
    """
    from __future__ import annotations

    import json
    import os
    from pathlib import Path

    import numpy as np
    import pytest

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _FOUNDATION_INDEX = _REPO_ROOT / "data" / "derived" / "foundation_index.json"
    pytestmark = pytest.mark.skipif(
        not _FOUNDATION_INDEX.exists(),
        reason="foundation bundle not committed (data/derived/foundation_index.json missing)",
    )

    from fedrec_foundation.rng import server_rng


    # =============================================================================
    # ADP-06: server_rng reproducibility
    # =============================================================================
    def test_server_rng_reproducible_per_round_selection():
        rng_a = server_rng(42)
        rng_b = server_rng(42)
        seq_a = [list(rng_a.sample(range(6040), 50)) for _ in range(3)]
        seq_b = [list(rng_b.sample(range(6040), 50)) for _ in range(3)]
        assert seq_a == seq_b, "server_rng(42) not reproducible across instances"


    def test_server_rng_different_seeds_different_selections():
        rng_a = server_rng(42)
        rng_b = server_rng(43)
        seq_a = list(rng_a.sample(range(6040), 50))
        seq_b = list(rng_b.sample(range(6040), 50))
        assert seq_a != seq_b, "server_rng(42) and server_rng(43) produced identical selections (RNG broken)"


    # =============================================================================
    # ADP-06: AdaptiveSplitFedAvg sum aggregation sanity
    # =============================================================================
    def test_adaptive_split_fedavg_aggregate_evaluate_sum_not_average(fake_client_proxy):
        """Given 2 clients (hit=1,eval=1 and hit=0,eval=99), HR@10 must be 1/100 = 0.01,
        not (1.0 + 0.0)/2 = 0.5 (per-client ratio average)."""
        from federated_adaptive_personalized_cf.strategy import AdaptiveSplitFedAvg
        from flwr.common import EvaluateRes, Code, Status

        strategy = AdaptiveSplitFedAvg(fraction_fit=0.1)
        r1 = EvaluateRes(status=Status(Code.OK, "ok"), loss=0.0, num_examples=1,
                          metrics={"hit_count_overall_at10": 1, "ndcg_sum_overall_at10": 1.0,
                                   "evaluated_users": 1})
        r2 = EvaluateRes(status=Status(Code.OK, "ok"), loss=0.0, num_examples=99,
                          metrics={"hit_count_overall_at10": 0, "ndcg_sum_overall_at10": 0.0,
                                   "evaluated_users": 99})
        results = [(fake_client_proxy(0), r1), (fake_client_proxy(1), r2)]
        _, thesis_metrics = strategy.aggregate_evaluate(server_round=1, results=results, failures=[])
        assert abs(thesis_metrics["sampled_hr@10"] - (1.0 / 100.0)) < 1e-9, (
            f"Sum-not-average regressed: got {thesis_metrics['sampled_hr@10']}, expected 0.01"
        )


    # =============================================================================
    # ADP-08: build_run_manifest with module="adaptive" + _manifest extensibility
    # =============================================================================
    def test_build_run_manifest_module_adaptive_with_best_prototype(tmp_path):
        from fedrec_foundation.bundle import verify_bundle
        from fedrec_foundation.manifest import (
            build_run_manifest, embed_manifest_in_result, generate_run_id,
        )
        from fedrec_foundation.mode import resolve_mode_defaults
        from fedrec_foundation.paths import data_derived
        from fedrec_foundation.split import load_split_manifest

        run_id = generate_run_id()
        profile = resolve_mode_defaults("benchmark_cross_device")
        index = verify_bundle(data_derived())
        split_mf = load_split_manifest(data_derived() / "split_manifest.json")

        manifest = build_run_manifest(
            run_id=run_id, module="adaptive", mode_profile=profile,
            overrides={}, foundation_index=index, split_manifest=split_mf,
            checkpoint_rule="best_round_restore",
        )
        assert getattr(manifest, "module", None) == "adaptive", \
            f"Expected manifest.module == 'adaptive', got {getattr(manifest, 'module', None)}"
        # All 4 IMP-2 fingerprints present (adapt to actual field names)
        results_data = embed_manifest_in_result(manifest, {})
        _manifest_dict = results_data["_manifest"]
        for fingerprint in ("mapping_sha256", "split_hash", "exclusion_sha256", "foundation_contract_sha256"):
            assert fingerprint in _manifest_dict, \
                f"IMP-2 fingerprint {fingerprint} missing from _manifest"
        # D-06: _manifest dict is mutable post-embed — proves best_prototype can be injected
        _manifest_dict["best_prototype"] = [1.0] * 128
        assert results_data["_manifest"]["best_prototype"] == [1.0] * 128


    # =============================================================================
    # D-13: cold-start counter arithmetic
    # =============================================================================
    def test_cold_start_counter_math(tmp_path):
        cache_root = tmp_path / ".embedding_cache" / "r1"
        cache_root.mkdir(parents=True)
        for pid in (0, 1, 2):
            (cache_root / f"partition_{pid}.pt").write_bytes(b"\x00")
        selected_pids = [0, 1, 2, 3, 4, 5]
        cold_count = sum(
            1 for pid in selected_pids
            if not (cache_root / f"partition_{int(pid)}.pt").exists()
        )
        assert cold_count == 3, f"expected 3 cold starts, got {cold_count}"


    # =============================================================================
    # D-02: cross-silo source-level regression guard
    # =============================================================================
    def test_cross_silo_legacy_mode_raises_not_implemented():
        src_path = _REPO_ROOT / "federated-adaptive-personalized-cf" / \
                   "federated_adaptive_personalized_cf" / "server_app.py"
        src = src_path.read_text()
        assert "cross_silo_legacy" in src
        assert "raise NotImplementedError" in src
        # Proximity guard: the raise must be near the cross_silo_legacy branch
        cross_silo_idx = src.index("cross_silo_legacy")
        nearby = src[cross_silo_idx:cross_silo_idx + 500]
        assert "NotImplementedError" in nearby, \
            "D-02 guard must raise NotImplementedError in the cross_silo_legacy branch"


    # =============================================================================
    # D-05: snapshot_best_prototype called inside the best-metric branch
    # =============================================================================
    def test_snapshot_best_prototype_called_inside_best_metric_branch():
        src_path = _REPO_ROOT / "federated-adaptive-personalized-cf" / \
                   "federated_adaptive_personalized_cf" / "server_app.py"
        src = src_path.read_text()
        # snapshot_best_prototype must appear in the source...
        assert "snapshot_best_prototype" in src, \
            "D-05 violated: server_app.py does not call strategy.snapshot_best_prototype"
        # ...and it must appear AFTER the best_metric = current_ndcg assignment (proximity).
        best_metric_assign_idx = src.find("best_metric = current_ndcg")
        assert best_metric_assign_idx != -1, \
            "best_metric = current_ndcg line missing — D-27 best-round tracking regressed"
        snapshot_idx = src.find("snapshot_best_prototype")
        assert snapshot_idx > best_metric_assign_idx, (
            "D-05 violated: snapshot_best_prototype must fire inside the best-metric branch, "
            "AFTER best_metric assignment"
        )
        # D-07 restore sequence: strategy._global_prototype = strategy.best_prototype appears
        # AFTER the best-round-restore "arrays = best_arrays" line.
        arrays_restore_idx = src.find("arrays = best_arrays")
        proto_restore_idx = src.find("strategy._global_prototype = strategy.best_prototype")
        assert proto_restore_idx != -1, "D-07 violated: prototype restore missing from server_app.py"
        assert proto_restore_idx > arrays_restore_idx, (
            "D-07 violated: prototype restore must follow `arrays = best_arrays` in source order"
        )
    ```

    Step 2 — Verify: `cd federated-adaptive-personalized-cf && pytest tests/test_server_integration.py -v` → 7 passed. Full suite: `pytest tests/ -v` → ~45 passed (Plan 01=10 + Plan 02=9 + Plan 03=14 + Plan 04=12+ + Plan 05=7 ≈ 52 total).

    Step 3 — Commit (--no-verify):
    ```
    git add federated-adaptive-personalized-cf/tests/test_server_integration.py
    git commit --no-verify -m "test(04-05): server integration — ADP-06 + AdaptiveSplitFedAvg + D-05/D-06/D-07 best_prototype + D-13 + D-02"
    ```
  </action>
  <acceptance_criteria>
    - `cd federated-adaptive-personalized-cf && pytest tests/test_server_integration.py -v` exits 0 with at least "7 passed"
    - `cd federated-adaptive-personalized-cf && pytest tests/ -v` exits 0 with at least 45 tests passing overall (accumulated Plans 01-05)
    - `grep -c "test_server_rng_reproducible_per_round_selection" federated-adaptive-personalized-cf/tests/test_server_integration.py` returns 1
    - `grep -c "test_server_rng_different_seeds_different_selections" federated-adaptive-personalized-cf/tests/test_server_integration.py` returns 1
    - `grep -c "test_adaptive_split_fedavg_aggregate_evaluate_sum_not_average" federated-adaptive-personalized-cf/tests/test_server_integration.py` returns 1
    - `grep -c "test_build_run_manifest_module_adaptive_with_best_prototype" federated-adaptive-personalized-cf/tests/test_server_integration.py` returns 1
    - `grep -c "test_cold_start_counter_math" federated-adaptive-personalized-cf/tests/test_server_integration.py` returns 1
    - `grep -c "test_cross_silo_legacy_mode_raises_not_implemented" federated-adaptive-personalized-cf/tests/test_server_integration.py` returns 1
    - `grep -c "test_snapshot_best_prototype_called_inside_best_metric_branch" federated-adaptive-personalized-cf/tests/test_server_integration.py` returns 1
    - D-18 scope: only tests/test_server_integration.py modified by this task
  </acceptance_criteria>
  <done>7 GREEN integration tests prove: (a) ADP-06 server_rng reproducibility + distinguishability, (b) AdaptiveSplitFedAvg sum-not-average aggregation, (c) ADP-08 build_run_manifest with module="adaptive" + 4 IMP-2 fingerprints + D-06 _manifest dict mutability post-embed, (d) D-13 cold-start counter arithmetic, (e) D-02 cross-silo NotImplementedError source-level guard, (f) D-05 snapshot_best_prototype fires in best-metric branch (proximity guard), (g) D-07 prototype restore follows arrays restore (proximity guard).</done>
</task>

</tasks>

<verification>
- `cd federated-adaptive-personalized-cf && pytest tests/ -v` exits 0 with at least 45 tests passing (accumulated from Plans 01-05)
- `python -c "import ast; ast.parse(open('federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py').read()); print('syntax ok')"` prints `syntax ok`
- `python -c "from federated_adaptive_personalized_cf.server_app import app; print('import ok')"` prints `import ok`
- `python scripts/run.py --dry-run adaptive benchmark_cross_device 2>&1 | grep -c "num-supernodes=6040"` returns at least 1
- `grep -cE "^import random$|random\\.sample\\(|random\\.seed\\(" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py` returns 0 (module-wide stdlib random eradication)
- D-18 scope: `git diff --stat federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/ federated-adaptive-personalized-cf/pyproject.toml` returns empty after this plan's 2 commits (Plans 01/02/03 own those)
</verification>

<success_criteria>
- ADP-03 observable end-to-end: `python scripts/run.py adaptive benchmark_cross_device` produces a result JSON whose `_manifest.best_prototype` is a `List[float]` of length `embedding_dim` (D-06); server_app.py's `strategy._global_prototype = strategy.best_prototype` runs before the final broadcast (D-07); snapshot_best_prototype is called inside the best-metric branch (D-05).
- ADP-06 observable: `python scripts/run.py adaptive benchmark_cross_device` samples 6040 supernodes via `_server_sampler.sample(range(6040), k)` in deterministic partition-id space; two runs with the same run-seed produce byte-identical selected_clients_per_round (proven by Plan 06 subprocess test).
- ADP-08 observable: Result JSON gains a top-level `_manifest` key with module="adaptive" + all Phase-1 fingerprints + best_prototype field; sibling `{run_id}-manifest.json` file exists beside the result JSON.
- D-13 cold-start counter is first-class: `results_data["cold_starts"] = {per_round, total_cold_starts, total_client_selections, cold_start_rate}`; W&B logs `round/cold_starts` per round.
- D-16 alpha diagnostics observable: `results_data["alpha_diagnostics_history"] = {round: {alpha_mean/std/p25/p50/p75/clip_hit_rate}}` populated when enable-per-user-alpha=true; W&B logs `round/alpha/*` per round.
- D-02 frozen cross-silo: `mode="cross_silo_legacy"` at startup raises NotImplementedError with explicit reference to pre-Phase-4 commits; fires BEFORE any training or data load.
- D-18 surgical discipline: pre-existing WIP in DummyClientProxy / weighted_average_metrics / print_evaluation_metrics / centralized-eval block is preserved verbatim.
</success_criteria>

<output>
After completion, create `.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-05-SUMMARY.md` with: file list (1 modified + 1 created), decisions made (D-07 restore placement — BEFORE final broadcast vs AFTER; alpha diagnostics aggregation weighting — num_examples vs hit_count; W&B project default choice; alpha_diagnostics_history shape in result JSON; any reuse_cache=true cold-start caveat), deviations (any auto-fixes logged), test counts (~7 new → ~52 total suite), commit SHAs, ADP-03 + ADP-06 + ADP-08 closure notes, Plan 06 (subprocess determinism regression guard) readiness confirmation.
</output>
</content>
</invoke>