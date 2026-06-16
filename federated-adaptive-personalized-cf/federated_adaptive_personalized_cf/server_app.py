"""federated-adaptive-personalized-cf: Adaptive Split Learning Server (Phase 4 Plan 05).

Cross-device migration with 4 adaptive-specific additions over Phase 3:

1. **AdaptiveSplitFedAvg / AdaptiveSplitFedProx** — replaces the old SplitFedAvg;
   includes prototype EMA override in aggregate_fit + sum-based aggregate_evaluate.

2. **D-05 best_prototype snapshot** — ``strategy.snapshot_best_prototype(round_num, embedding_dim)``
   fires at the SAME moment as D-27 ``best_arrays`` snapshot (when current_ndcg > best_metric).

3. **D-07 best_prototype restore** — before the final broadcast / result-write,
   ``strategy._global_prototype = strategy.best_prototype`` so clients receiving
   the final ``global_prototype`` see the RESTORED prototype, not last-round drift.

4. **D-06 best_prototype embedded in manifest** — after ``embed_manifest_in_result``,
   ``results_data['_manifest']['best_prototype'] = strategy.best_prototype.tolist()`` (or None).

Plus standard Phase-3-Plan-04 carry-forward:
- **D-25 mode resolver** — ``resolve_mode_defaults(mode)`` at entry; all hyperparameters
  read as ``int/float/str(context.run_config.get(key, profile.field))``.
- **D-02 cross-silo guard** — adaptive cross-silo support removed per D-02; any
  ``mode="cross_silo_legacy"`` invocation raises ``NotImplementedError`` at startup BEFORE
  any training or data load.
- **G-03-01 discovery round** — one-shot ``evaluate(discover_only=true)`` broadcast to ALL
  nodes BEFORE the main loop builds ``partition_to_node_id: Dict[int, int]`` for stable
  partition-id-space sampling.
- **ADP-06 / PSN-04 seeded sampling** — ``_server_sampler = server_rng(run_seed)``
  instantiated ONCE pre-loop; ``_server_sampler.sample(range(N), k)`` per round.
- **D-27 best-round restore** — in-memory; no disk writes.
- **D-15 double-write manifest** with ``module="adaptive"``.
- **D-13 cold-start counter** — probed per round, accumulated, reported in result JSON.
- **D-16 alpha diagnostics aggregate** — server weighted-averages alpha_mean/std/p25/p50/p75/
  alpha_clip_hit_rate from client FitRes sidecars (populated by Plan 03 when
  ``enable-per-user-alpha=true``).
- W&B project switch to ``federated-cf-cross-device`` for benchmark mode.

D-18 surgical discipline:
  DummyClientProxy, weighted_average_metrics, print_evaluation_metrics,
  EarlyStopping setup/teardown, CUDA device fallback, AlphaAnalyzer integration,
  and final wandb.run.summary logging are preserved verbatim from the pre-Phase-4 code.

NOTE: Centralized evaluation is NOT possible in split learning (server only has GLOBAL
params). Final headline metrics come from strategy-aggregated federated eval.
"""

import torch
import json
import wandb
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from flwr.common import (
    Code,
    EvaluateRes,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.common.record import ArrayRecord, ConfigRecord, MetricRecord, RecordDict
from flwr.common.context import Context
from flwr.serverapp import Grid, ServerApp

from federated_adaptive_personalized_cf.task import get_model
from federated_adaptive_personalized_cf.strategy import (
    AdaptiveSplitFedAvg,
    AdaptiveSplitFedProx,
    USER_PROTOTYPE_KEY,
)
from federated_adaptive_personalized_cf.cache_snapshot import (
    cleanup_snapshots,
    restore_cache,
    snapshot_cache,
)
from federated_adaptive_personalized_cf.evaluation import AlphaAnalyzer
from federated_adaptive_personalized_cf.early_stopping import EarlyStopping

# Phase 4 Plan 05: foundation imports (ADP-06, D-13, D-15, D-25, D-27, G-03-01).
from fedrec_foundation.bundle import verify_bundle
from fedrec_foundation.manifest import (
    build_run_manifest,
    embed_manifest_in_result,
    generate_run_id,
    write_manifest_sibling,
)
from fedrec_foundation.mode import (
    log_mode_and_overrides,
    resolve_mode_defaults,
)
from fedrec_foundation.paths import data_derived
from fedrec_foundation.paths import module_run_results_dir
from fedrec_foundation.rng import server_rng
from fedrec_foundation.split import load_split_manifest
# Phase 6 Plan 05: repo-root-anchored path + atomic write + dataclasses.replace.
from fedrec_foundation.atomic import atomic_write_json
from dataclasses import replace as dataclass_replace

# Create ServerApp
app = ServerApp()


class DummyClientProxy(ClientProxy):
    """Minimal ClientProxy for strategy compatibility."""

    def __init__(self, cid: str):
        super().__init__(cid)

    def get_properties(self, ins, timeout, group_id):
        return None

    def get_parameters(self, ins, timeout, group_id):
        return None

    def fit(self, ins, timeout, group_id):
        return None

    def evaluate(self, ins, timeout, group_id):
        return None

    def reconnect(self, ins, timeout, group_id):
        return None


def weighted_average_metrics(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """
    Aggregate evaluation metrics from multiple clients using weighted average.

    NOTE: This function is available for custom metric aggregation but is not
    currently used for thesis-table metrics. Flower's new ServerApp API handles
    metric aggregation automatically based on num-examples.

    This function aggregates both rating prediction metrics (RMSE, MAE) and
    ranking metrics (Hit Rate, Precision, Recall, NDCG, MRR) across clients.

    Parameters
    ----------
    metrics : List[Tuple[int, Dict[str, float]]]
        List of (num_examples, metrics_dict) tuples from each client.

    Returns
    -------
    Dict[str, float]
        Dictionary of aggregated metrics.
    """
    # Calculate total number of examples
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {}

    # Aggregate metrics using weighted average
    aggregated = {}

    # Get all metric keys from first client (assumes all clients report same metrics)
    if metrics:
        metric_keys = metrics[0][1].keys()

        for key in metric_keys:
            if key == "num-examples":
                continue

            # Only aggregate numeric values (skip lists, dicts, etc.)
            first_value = metrics[0][1].get(key)
            if not isinstance(first_value, (int, float)):
                continue

            # Weighted average: sum(metric * num_examples) / total_examples
            weighted_sum = sum(
                metrics_dict.get(key, 0.0) * num_examples
                for num_examples, metrics_dict in metrics
                if isinstance(metrics_dict.get(key, 0.0), (int, float))
            )
            aggregated[key] = weighted_sum / total_examples

    return aggregated


def print_evaluation_metrics(round_num: int, metrics: Dict[str, float], context: Context):
    """
    Pretty print evaluation metrics for a federated round.

    Parameters
    ----------
    round_num : int
        Current federated learning round.
    metrics : Dict[str, float]
        Aggregated metrics dictionary.
    context : Context
        Flower context with configuration.
    """
    print(f"\n{'='*70}")
    print(f"Evaluation Results - Round {round_num}")
    print(f"{'='*70}")

    # Rating prediction metrics
    if "rmse" in metrics or "mae" in metrics:
        print("\n📊 Rating Prediction Metrics:")
        if "eval_loss" in metrics:
            print(f"  Loss:      {metrics['eval_loss']:.4f}")
        if "rmse" in metrics:
            print(f"  RMSE:      {metrics['rmse']:.4f}")
        if "mae" in metrics:
            print(f"  MAE:       {metrics['mae']:.4f}")

    # Ranking metrics
    enable_ranking = context.run_config.get("enable-ranking-eval", True)
    if enable_ranking:
        # Parse K values from comma-separated string
        k_values_str = context.run_config.get("ranking-k-values", "5,10,20")
        k_values = [int(k.strip()) for k in k_values_str.split(",")]

        # Check if we have any ranking metrics
        has_ranking = any(f"hit_rate@{k}" in metrics for k in k_values)

        if has_ranking:
            print("\n🎯 Ranking Metrics:")

            # MRR (not K-dependent)
            if "mrr" in metrics:
                print(f"  MRR:       {metrics['mrr']:.4f}")

            # Metrics for each K value
            for k in sorted(k_values):
                print(f"\n  @ K={k}:")
                if f"hit_rate@{k}" in metrics:
                    print(f"    Hit Rate:   {metrics[f'hit_rate@{k}']:.4f}")
                if f"precision@{k}" in metrics:
                    print(f"    Precision:  {metrics[f'precision@{k}']:.4f}")
                if f"recall@{k}" in metrics:
                    print(f"    Recall:     {metrics[f'recall@{k}']:.4f}")
                if f"f1@{k}" in metrics:
                    print(f"    F1:         {metrics[f'f1@{k}']:.4f}")
                if f"ndcg@{k}" in metrics:
                    print(f"    NDCG:       {metrics[f'ndcg@{k}']:.4f}")
                if f"map@{k}" in metrics:
                    print(f"    MAP:        {metrics[f'map@{k}']:.4f}")

            # Diversity/Popularity metrics (only for first K value to avoid repetition)
            k = sorted(k_values)[0]
            has_diversity = any(f"{m}@{k}" in metrics for m in ['coverage', 'novelty'])
            if has_diversity:
                print("\n📈 Diversity/Popularity Metrics:")
                for k in sorted(k_values):
                    print(f"\n  @ K={k}:")
                    if f"coverage@{k}" in metrics:
                        print(f"    Coverage:   {metrics[f'coverage@{k}']:.4f}")
                    if f"novelty@{k}" in metrics:
                        print(f"    Novelty:    {metrics[f'novelty@{k}']:.4f}")

        # Sampled ranking metrics (leave-one-out with N negatives)
        # For fair comparison with published baselines (NCF, FedMF, PFedRec)
        has_sampled = any(f"sampled_hr@{k}" in metrics for k in k_values)
        if has_sampled:
            num_neg = int(metrics.get('sampled_num_negatives', 99))
            print(f"\n🔬 Sampled Ranking Metrics (leave-one-out + {num_neg} negatives):")
            print("  (For fair comparison with NCF, FedMF, PFedRec baselines)")

            if "sampled_mrr" in metrics:
                print(f"\n  MRR:       {metrics['sampled_mrr']:.4f}")

            for k in sorted(k_values):
                print(f"\n  @ K={k}:")
                if f"sampled_hr@{k}" in metrics:
                    print(f"    Hit Rate:   {metrics[f'sampled_hr@{k}']:.4f}")
                if f"sampled_ndcg@{k}" in metrics:
                    print(f"    NDCG:       {metrics[f'sampled_ndcg@{k}']:.4f}")

    print(f"\n{'='*70}\n")


def _cold_start_cache_root(run_id: str, reuse_cache: bool) -> Path:
    """Resolve the cache dir the D-13 cold-start probe should check.

    Under D-09 reuse-cache=true the client-side cache path includes a sig_<hash>
    prefix that the server cannot construct without client-side signature fields.
    In that regime the counter is short-circuited to zero and the log line names
    D-09 explicitly. For the standard run_id-scoped cache, the dir is the
    module-anchored ``federated-adaptive-personalized-cf/.embedding_cache/{run_id}``.

    The probe path MUST be resolved the same way the client resolves it
    (``client_app._CACHE_BASE_DIR = Path(__file__).resolve().parent.parent /
    ".embedding_cache"``); using a relative ``Path(".embedding_cache")`` here
    silently misses the cache because the server CWD is the repo root, not the
    module dir, so the probe always reports cold_start_rate=1.0 even when
    every client is reading from the cache successfully.

    Parameters
    ----------
    run_id : str
        Run identifier passed from context or generated at startup.
    reuse_cache : bool
        Whether D-09 reuse-cache mode is active (short-circuits counter to 0).

    Returns
    -------
    Path
        ``<module_dir>/../.embedding_cache/{run_id}`` for the standard case.
    """
    return Path(__file__).resolve().parent.parent / ".embedding_cache" / run_id


def _extract_sibling_records(
    record_dict: RecordDict,
    metrics_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge top-level RecordDict sibling records into ``metrics_dict`` (D-05/D-06/D-16).

    The Phase-4 client-server contract puts ``user_prototype`` and ``alpha_diagnostics``
    as TOP-LEVEL sibling records in the response RecordDict (separate from the strict
    ``metrics`` key) per D-21 — the FitMetricsContract validator rejects free-form
    inline extras, so the client routes these payloads as their own MetricRecords
    (see ``client_app.py`` lines 741 + 747). Downstream consumers
    (``AdaptiveSplitFedAvg._aggregate_prototypes`` for the prototype EMA, the D-16
    alpha-diagnostics aggregator inside ``main()``) both read from
    ``fit_res.metrics``; without this merge the siblings are silently dropped,
    surfacing at runtime as ``best_prototype = [0.0] * embedding_dim`` (D-08
    fallback) and ``alpha_diagnostics_history`` missing from the result JSON.

    Closes UAT GAP-04-01 surfaced by ``20260427-132620-eb2d19_results.json``.

    Parameters
    ----------
    record_dict : RecordDict
        Full ``response.content`` from a train Message reply.
    metrics_dict : Dict[str, Any]
        Mutable dict already populated from the strict ``metrics`` MetricRecord.
        Mutated in place.

    Returns
    -------
    Dict[str, Any]
        Same ``metrics_dict`` reference with ``user_prototype`` (List[float]) and
        ``alpha_diagnostics`` (Dict[str, float]) merged in when their sibling
        records are present.
    """
    proto_record = record_dict.get(USER_PROTOTYPE_KEY)
    if proto_record is not None:
        proto_dict = dict(proto_record)
        proto_payload = proto_dict.get(USER_PROTOTYPE_KEY)
        if isinstance(proto_payload, (list, tuple)):
            metrics_dict[USER_PROTOTYPE_KEY] = list(proto_payload)

    alpha_record = record_dict.get("alpha_diagnostics")
    if alpha_record is not None:
        alpha_dict = dict(alpha_record)
        if alpha_dict:
            metrics_dict["alpha_diagnostics"] = alpha_dict

    return metrics_dict


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp (Phase 4 Plan 05).

    Phase-4 adaptive cross-device migration; see module docstring for the full
    decision list.
    """

    # =========================================================================
    # D-25 mode resolver + D-19 visible overrides. Profile is canonical; every
    # hyperparameter read is ``int(context.run_config.get(key, profile.field))``.
    # =========================================================================
    mode = str(context.run_config.get("mode", "cross_silo_legacy"))
    run_seed = int(context.run_config.get("run-seed", 42))
    profile = resolve_mode_defaults(mode)
    print(
        f"\n[MODE] Resolved profile mode={profile.mode!r} "
        f"num_supernodes={profile.num_supernodes} "
        f"weight_policy={profile.weight_policy!r} "
        f"primary_evaluator={profile.primary_evaluator!r}"
    )
    overrides = log_mode_and_overrides(mode, profile, context.run_config)
    if overrides:
        print(
            f"  OVERRIDE: {len(overrides)} key(s) diverge from mode default. "
            f"Run is NOT comparable to benchmark thesis table."
        )

    # =========================================================================
    # D-02 frozen cross-silo guard. Adaptive cross-device migration removed
    # multi-user-per-client support per D-02; cross-silo numbers live in
    # pre-Phase-4 git history and are not re-derived. The guard fires BEFORE
    # any model load or training so a wrong-mode invocation fails loud immediately.
    # =========================================================================
    if mode == "cross_silo_legacy":
        raise NotImplementedError(
            "Adaptive cross-device migration removed multi-user-per-client support "
            "per D-02. Check out a pre-Phase-4 commit (see "
            ".planning/phases/04-adaptive-migration-bug-fixes/04-CONTEXT.md §Deferred) "
            "to reproduce legacy cross-silo numbers."
        )

    # =========================================================================
    # D-25 hyperparameters — profile is source of truth; run_config overrides win.
    # =========================================================================
    num_rounds: int = int(context.run_config.get("num-server-rounds", profile.num_server_rounds))
    fraction_train: float = float(context.run_config.get("fraction-train", profile.fraction_train))
    # BUG 4 NOTE (2026-06-01): ModeProfile.fraction_eval (=1.0) is advisory-only and was
    # silently ignored — the strategy's evaluate fraction was hardwired to fraction_train.
    # Expose it as an OVERRIDE-ONLY knob defaulting to fraction_train so existing runs (and
    # best_round selection, which keys off the per-round trained-subset metric) are unchanged.
    # The diagnostic discriminator uses `diagnostic-fullpop-eval`, NOT this knob.
    # Sentinel -1.0 (pyproject default) => mirror fraction_train (no behavior change).
    _fe_raw: float = float(context.run_config.get("fraction-eval", -1.0))
    fraction_eval: float = fraction_train if _fe_raw < 0 else _fe_raw
    lr: float = float(context.run_config.get("lr", profile.lr))
    model_type: str = str(context.run_config.get("model-type", "bpr"))
    embedding_dim: int = int(context.run_config.get("embedding-dim", profile.embedding_dim))
    dropout: float = float(context.run_config.get("dropout", 0.1))

    strategy_name: str = str(context.run_config.get("strategy", "fedprox")).lower()
    proximal_mu: float = float(context.run_config.get("proximal-mu", 0.01))
    prototype_momentum: float = float(context.run_config.get("prototype-momentum", 0.9))

    weight_policy: str = str(context.run_config.get("weight-policy", profile.weight_policy))
    checkpoint_rule: str = str(
        context.run_config.get(
            "checkpoint-rule",
            getattr(profile, "checkpoint_rule", "best_round_restore"),
        )
    )
    reuse_cache_flag: bool = bool(context.run_config.get("reuse-cache", False))  # D-09

    # Bug 3 / Alt-A: end-of-training calibration pass (D-06.7).
    # Runs ONE extra training pass to ALL partitions after best_round_restore
    # (and Path B cache snapshot restore) so every client's local state aligns
    # with the restored best-round GLOBAL params before D-06 full-pop eval.
    # Default off — opt in for full thesis runs.
    final_calibration_enabled: bool = bool(
        context.run_config.get("final-calibration-enabled", False)
    )
    final_calibration_epochs: int = int(
        context.run_config.get("final-calibration-epochs", 1)
    )

    # Get adaptive alpha configuration
    alpha_min = float(context.run_config.get("alpha-min", 0.1))
    alpha_max = float(context.run_config.get("alpha-max", 0.95))
    alpha_quantity_threshold = context.run_config.get("alpha-quantity-threshold", 50)
    alpha_quantity_temperature = context.run_config.get("alpha-quantity-temperature", 0.1)

    # Dual model specific configuration (Level 2: PersonalMLP)
    mlp_hidden_dims = None
    fusion_type = "add"
    if model_type == "dual":
        mlp_dims_str = str(context.run_config.get("mlp-hidden-dims", "512,256,128"))
        mlp_hidden_dims = [int(d.strip()) for d in mlp_dims_str.split(",")]
        fusion_type = str(context.run_config.get("fusion-type", "concat"))

    # Materialize the run_id early so the D-13 cold-start probe resolves to
    # the same cache dir the client will write into this round.
    run_id = str(context.run_config.get("run-id", "")) or generate_run_id()
    _MODULE: str = "adaptive"   # cross-references: build_run_manifest, module_run_results_dir

    # Early stopping configuration
    early_stopping_enabled = context.run_config.get("early-stopping-enabled", False)
    early_stopping_patience = context.run_config.get("early-stopping-patience", 10)
    early_stopping_metric = context.run_config.get("early-stopping-metric", "sampled_ndcg@10")
    early_stopping_mode = context.run_config.get("early-stopping-mode", "max")
    early_stopping_min_delta = context.run_config.get("early-stopping-min-delta", 0.001)

    early_stopper = None
    if early_stopping_enabled:
        early_stopper = EarlyStopping(
            patience=early_stopping_patience,
            metric_name=early_stopping_metric,
            mode=early_stopping_mode,
            min_delta=early_stopping_min_delta,
            verbose=True,
        )
        print(f"  Early stopping: Enabled (patience={early_stopping_patience}, metric={early_stopping_metric})")

    # =========================================================================
    # Initialize Weights & Biases (PROJECT.md constraint: cross-device runs
    # go to a dedicated W&B project, benchmark mode -> federated-cf-cross-device).
    # =========================================================================
    wandb_enabled = context.run_config.get("wandb-enabled", False)
    wandb_run = None
    if wandb_enabled:
        wandb_config = {
            "run_id": run_id,
            "num_rounds": num_rounds,
            "fraction_train": fraction_train,
            "local_epochs": context.run_config.get("local-epochs", 12),
            "strategy": strategy_name,
            "proximal_mu": proximal_mu,
            "model_type": model_type,
            "embedding_dim": embedding_dim,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": context.run_config.get("weight-decay", 1e-5),
            "dirichlet_alpha": context.run_config.get("alpha", 0.5),
            # Adaptive personalization config
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "alpha_quantity_threshold": alpha_quantity_threshold,
            "alpha_quantity_temperature": alpha_quantity_temperature,
            "prototype_momentum": prototype_momentum,
            # Next-Gen Personalization Techniques
            "enable_per_user_alpha": context.run_config.get("enable-per-user-alpha", False),
            "enable_item_perturbation": context.run_config.get("enable-item-perturbation", False),
            "item_perturbation_reg": context.run_config.get("item-perturbation-reg", 0.01),
            "contrastive_lambda": context.run_config.get("contrastive-lambda", 0.0),
            "contrastive_tau": context.run_config.get("contrastive-tau", 0.1),
            # Early stopping config
            "early_stopping_enabled": early_stopping_enabled,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_metric": early_stopping_metric,
            # D-25 contract keys
            "mode": mode,
            "run_seed": run_seed,
            "weight_policy": weight_policy,
            "partition_mode": str(context.run_config.get("partition-mode", profile.partition_mode)),
            "checkpoint_rule": checkpoint_rule,
            "reuse_cache": reuse_cache_flag,
        }
        # Phase 7 D-04: thesis_crossdevice_main joins the cross-device W&B project gate.
        default_project = (
            "federated-cf-cross-device"
            if mode in ("benchmark_cross_device", "thesis_crossdevice_main", "paper_compat_pfedrec")
            else "federated-adaptive-personalized-cf"
        )
        wandb_project_cfg = str(context.run_config.get("wandb-project", "")).strip()
        wandb_project = wandb_project_cfg if wandb_project_cfg else default_project
        wandb_entity = context.run_config.get("wandb-entity", "")
        wandb_run_name = context.run_config.get("wandb-run-name", "")
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity if wandb_entity else None,
            name=wandb_run_name if wandb_run_name else None,
            config=wandb_config,
        )
        print("  Weights & Biases: Enabled")

    # Load global Matrix Factorization model (split-learning: only GLOBAL
    # params are kept on the server — item embeddings, item bias, global bias).
    print(f"\nInitializing {model_type.upper()} Matrix Factorization model...")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Dropout: {dropout}")
    if model_type == "dual":
        print(f"  MLP hidden dims: {mlp_hidden_dims}")
        print(f"  Fusion type: {fusion_type}")

    global_model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
        mlp_hidden_dims=mlp_hidden_dims,
        fusion_type=fusion_type,
    )

    num_params = sum(p.numel() for p in global_model.parameters())
    print(f"  Total parameters (local + global union): {num_params:,}")

    # Split-learning: only send GLOBAL params to clients.
    arrays = ArrayRecord(global_model.get_global_parameters())

    # =========================================================================
    # Phase-4: AdaptiveSplitFedAvg / AdaptiveSplitFedProx (from Plan 01).
    # Replaces the old SplitFedAvg / SplitFedProx / FedAvg / FedProx.
    # These new classes add: prototype EMA (aggregate_fit override), sum-based
    # aggregate_evaluate, and best_prototype snapshot field (D-05).
    # =========================================================================
    if strategy_name == "fedprox":
        strategy = AdaptiveSplitFedProx(
            fraction_fit=fraction_train,
            fraction_evaluate=fraction_eval,
            prototype_momentum=prototype_momentum,
            proximal_mu=proximal_mu,
        )
        print(f"  Strategy: AdaptiveSplitFedProx (proximal_mu={proximal_mu}, prototype_momentum={prototype_momentum})")
    else:
        strategy = AdaptiveSplitFedAvg(
            fraction_fit=fraction_train,
            fraction_evaluate=fraction_eval,
            prototype_momentum=prototype_momentum,
        )
        print(f"  Strategy: AdaptiveSplitFedAvg (prototype_momentum={prototype_momentum})")

    # =========================================================================
    # G-03-01: discovery round. Build partition_id -> node_id mapping BEFORE
    # the main loop so per-round sampling runs in stable partition-id space
    # (0..N-1) instead of Flower's os.urandom-seeded ephemeral node_id space.
    # Plan 03 already wired the client-side discover_only short-circuit.
    # =========================================================================
    all_node_ids = list(grid.get_node_ids())
    expected_n = int(profile.num_supernodes)
    assert len(all_node_ids) == expected_n, (
        f"G-03-01 invariant: grid.get_node_ids() returned {len(all_node_ids)} "
        f"node_ids, expected num_supernodes={expected_n} from profile {profile.mode!r}."
    )
    print(f"\n[G-03-01] Running discovery round over {expected_n} supernodes...")
    discovery_config = ConfigRecord({"discover_only": True})
    discovery_messages = [
        grid.create_message(
            content=RecordDict({"arrays": ArrayRecord(), "config": discovery_config}),
            message_type="evaluate",
            dst_node_id=nid,
            group_id="discovery",
        )
        for nid in all_node_ids
    ]
    discovery_responses = list(grid.send_and_receive(discovery_messages))
    partition_to_node_id: Dict[int, int] = {}
    for r in discovery_responses:
        if r.has_error():
            continue
        m = dict(r.content.get("metrics", MetricRecord()))
        pid = m.get("partition_id")
        if pid is None:
            continue
        partition_to_node_id[int(pid)] = int(r.metadata.src_node_id)
    missing = sorted(set(range(expected_n)) - set(partition_to_node_id.keys()))
    assert not missing, (
        f"G-03-01 invariant: discovery round did not collect partition_ids "
        f"for {len(missing)} nodes (first 5 missing: {missing[:5]}). "
        f"Cannot proceed — partition-space sampling would KeyError."
    )
    print(f"[G-03-01] Discovery complete: {len(partition_to_node_id)} partition -> node_id entries.")

    # Start federated learning
    print(f"\nStarting Federated Learning with {num_rounds} rounds...")
    print(f"  Clients per round: {fraction_train * 100:.0f}%")
    print(f"  Ranking evaluation: {'Enabled' if context.run_config.get('enable-ranking-eval', True) else 'Disabled'}")
    if context.run_config.get('enable-ranking-eval', True):
        k_values_str = context.run_config.get('ranking-k-values', "5,10,20")
        print(f"  K values: {k_values_str}")

    # =========================================================================
    # FEDERATED TRAINING LOOP using Grid's message-passing API
    # =========================================================================
    train_metrics_history: Dict[int, Dict] = {}
    eval_metrics_history: Dict[int, Dict[str, Any]] = {}
    per_client_metrics_history: Dict[int, List[Tuple[str, Dict]]] = {}  # For AlphaAnalyzer

    # ADP-06 / PSN-04: seeded RNG for per-round client selection — one instance
    # for the whole run, so the sequence across rounds is stable for a given run_seed.
    _server_sampler = server_rng(run_seed)
    selected_clients_per_round: List[List[int]] = []  # D-26 persisted in result JSON

    # D-27 best-round tracking (in-memory; no disk writes).
    best_metric: float = float("-inf")
    best_round_num: int = 0
    best_arrays = arrays  # fallback if no eval round improves

    # D-13 cold-start counter. Tracked per round AND accumulated.
    total_cold_starts: int = 0
    cold_starts_per_round: List[int] = []
    cache_root = _cold_start_cache_root(run_id, reuse_cache_flag)

    # D-16 alpha diagnostics history (populated when enable-per-user-alpha=true).
    alpha_diagnostics_history: Dict[int, Dict[str, float]] = {}

    # Bug 2 discriminator (gated by diagnostic-fullpop-eval). Each round, after the
    # normal trained-subset eval, ALSO evaluate users trained >=1 time but NOT this
    # round, bucketed by rounds-since-last-trained, to separate fixable staleness
    # (NDCG decays with age) from structural failure (NDCG flat-low even at age 1-2).
    # Never-trained (cold-init) users are excluded so the cold-init confound is gone.
    # This emits diag/* telemetry ONLY; it does not affect best_round selection.
    diagnostic_fullpop_eval: bool = bool(context.run_config.get("diagnostic-fullpop-eval", False))
    last_trained_round: Dict[int, int] = {}  # pid -> last round it was in the train set
    node_to_pid: Dict[int, int] = {v: k for k, v in partition_to_node_id.items()}
    _DIAG_AGE_BUCKETS = (("1_2", 1, 2), ("3_10", 3, 10), ("11_30", 11, 30), ("31plus", 31, 10**9))

    # Track the last executed round so post-loop bookkeeping (early stop) can
    # report the correct final round.
    round_num = 0

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*50}")
        print(f"Round {round_num}/{num_rounds}")
        print(f"{'='*50}")

        # Build train config with global prototype (if available)
        # Pass run-id + round metadata so clients resolve the same cache path
        # the D-13 counter is probing at the server.
        global_prototype = strategy.get_global_prototype()
        train_config_dict: Dict[str, Any] = {
            "lr": lr,
            "proximal_mu": proximal_mu,
            "round_num": int(round_num),
            "run_id": str(run_id),
            "reuse_cache": bool(reuse_cache_flag),
        }
        # Add global prototype to config if available (EMA updated by aggregate_fit)
        if global_prototype is not None:
            train_config_dict["global_prototype"] = global_prototype.tolist()

        train_config = ConfigRecord(train_config_dict)

        # =====================================================================
        # G-03-01: sample in partition-id space (stable 0..N-1), translate to
        # node_ids for message addressing. Deterministic across runs for a
        # given run_seed because the sampling DOMAIN is now seed-independent.
        # =====================================================================
        num_selected = max(1, int(expected_n * fraction_train))
        selected_pids: List[int] = _server_sampler.sample(range(expected_n), num_selected)
        selected_node_ids = [partition_to_node_id[pid] for pid in selected_pids]

        selected_clients_per_round.append([int(pid) for pid in selected_pids])

        # Bug 2 discriminator bookkeeping: record this round as each selected
        # user's most-recent training round (age = round_num - last_trained_round).
        if diagnostic_fullpop_eval:
            for pid in selected_pids:
                last_trained_round[int(pid)] = round_num

        # =====================================================================
        # D-13 cold-start counter. Probe the cache BEFORE the train message
        # is sent: counts partitions for whom no local-state .pt file exists
        # yet. Under D-09 reuse-cache=true the server cannot cheaply resolve
        # the sig_<hash> path — the expected regime is "all hot" — so the
        # counter is short-circuited to 0 and the log line names D-09 explicitly.
        # =====================================================================
        if reuse_cache_flag:
            cold_count = 0
        else:
            cold_count = sum(
                1 for pid in selected_pids
                if not (cache_root / f"partition_{int(pid)}.pt").exists()
            )
        cold_starts_per_round.append(int(cold_count))
        total_cold_starts += int(cold_count)
        if reuse_cache_flag:
            print(
                f"  [D-13] reuse-cache=true (D-09) — server-side cold-start counter "
                f"short-circuited to 0; client logs will show hit/miss per partition"
            )
        else:
            print(f"  [D-13] cold_starts={cold_count}/{num_selected} this round")

        if wandb_enabled and wandb_run is not None:
            wandb.log(
                {
                    "round/selected_clients": [int(pid) for pid in selected_pids],
                    "round/cold_starts": int(cold_count),
                },
                step=round_num,
            )

        print(f"  Selected {num_selected}/{expected_n} clients for training")

        # =====================================================================
        # TRAINING PHASE
        # =====================================================================
        train_messages = []
        for node_id in selected_node_ids:
            content = RecordDict({
                "arrays": arrays,
                "config": train_config,
            })
            msg = grid.create_message(
                content=content,
                message_type="train",
                dst_node_id=node_id,
                group_id=f"train_round_{round_num}",
            )
            train_messages.append(msg)

        train_responses = list(grid.send_and_receive(train_messages))

        # Parse training responses into FitRes format for strategy
        fit_results = []
        round_train_metrics = []
        per_client_metrics_history[round_num] = []  # Initialize per-client list for this round

        for response in train_responses:
            if response.has_error():
                print(f"  Warning: Client {response.metadata.src_node_id} returned error")
                continue

            resp_arrays = response.content.get("arrays", ArrayRecord())
            resp_metrics = response.content.get("metrics", MetricRecord())

            metrics_dict = dict(resp_metrics) if resp_metrics else {}
            # D-05/D-06/D-16: merge sibling RecordDict records (user_prototype +
            # alpha_diagnostics) into metrics_dict so strategy._aggregate_prototypes
            # and the alpha-diagnostics aggregator can read them via fit_res.metrics.
            # Without this, the siblings emitted by client_app.py:741, 747 are
            # silently dropped (UAT GAP-04-01).
            _extract_sibling_records(response.content, metrics_dict)
            num_examples = int(metrics_dict.get(
                "num_training_examples",
                metrics_dict.get("num-examples", 1),
            ))

            # Create FitRes for strategy aggregation
            parameters = ndarrays_to_parameters(list(resp_arrays.to_torch_state_dict().values()))
            fit_res = FitRes(
                status=None,
                parameters=parameters,
                num_examples=num_examples,
                metrics=metrics_dict,
            )

            client_id = str(response.metadata.src_node_id)
            client_proxy = DummyClientProxy(client_id)
            fit_results.append((client_proxy, fit_res))

            round_train_metrics.append((num_examples, metrics_dict))

            # Store per-client metrics for AlphaAnalyzer (before aggregation)
            per_client_metrics_history[round_num].append((client_id, metrics_dict))

        # Aggregate training results using strategy.
        # AdaptiveSplitFedAvg.aggregate_fit: super().aggregate_fit (weighted-average
        # of GLOBAL params) then _aggregate_prototypes updates EMA prototype.
        if fit_results:
            aggregated_params, agg_metrics = strategy.aggregate_fit(
                server_round=round_num,
                results=fit_results,
                failures=[],
            )

            if aggregated_params is not None:
                param_ndarrays = parameters_to_ndarrays(aggregated_params)
                param_keys = list(arrays.to_torch_state_dict().keys())
                new_state_dict = {k: torch.from_numpy(v) for k, v in zip(param_keys, param_ndarrays)}
                arrays = ArrayRecord(new_state_dict)

            train_metrics_history[round_num] = weighted_average_metrics(round_train_metrics)
            train_metrics_history[round_num].update(agg_metrics)

            train_loss = train_metrics_history[round_num].get('train_loss', 'N/A')
            if isinstance(train_loss, (int, float)):
                print(f"  Training loss: {train_loss:.4f}")

            # ==================================================================
            # D-16 alpha diagnostics aggregate (Phase-4 unique). Client FitRes
            # sidecar carries alpha_diagnostics when enable-per-user-alpha=true.
            # Server weighted-averages 6 scalar fields across contributing clients
            # by num_examples. Logged per-round to W&B + eval_metrics_history.
            # ==================================================================
            alpha_contributions: List[Tuple[Dict[str, float], int]] = []
            for _proxy, fit_res in fit_results:
                fit_metrics = fit_res.metrics or {}
                ad = fit_metrics.get("alpha_diagnostics")
                if isinstance(ad, dict) and ad:
                    alpha_contributions.append((ad, int(fit_res.num_examples)))
            if alpha_contributions:
                total_w = sum(w for _, w in alpha_contributions)
                alpha_agg: Dict[str, float] = {}
                for key in ("alpha_mean", "alpha_std", "alpha_p25", "alpha_p50", "alpha_p75", "alpha_clip_hit_rate"):
                    alpha_agg[key] = sum(
                        ad.get(key, 0.0) * w for ad, w in alpha_contributions
                    ) / total_w
                alpha_diagnostics_history[round_num] = alpha_agg
                if wandb_enabled and wandb_run is not None:
                    wandb.log(
                        {f"round/alpha/{k}": v for k, v in alpha_agg.items()},
                        step=round_num,
                    )

        # =====================================================================
        # EVALUATION PHASE
        # =====================================================================
        eval_messages = []
        # Use updated global_prototype after aggregate_fit ran the EMA
        global_prototype = strategy.get_global_prototype()
        for node_id in selected_node_ids:
            eval_config_dict: Dict[str, Any] = {
                "lr": lr,
                "round_num": int(round_num),
                "run_id": str(run_id),
                "reuse_cache": bool(reuse_cache_flag),
            }
            if global_prototype is not None:
                eval_config_dict["global_prototype"] = global_prototype.tolist()
            eval_config = ConfigRecord(eval_config_dict)
            content = RecordDict({
                "arrays": arrays,
                "config": eval_config,
            })
            msg = grid.create_message(
                content=content,
                message_type="evaluate",
                dst_node_id=node_id,
                group_id=f"eval_round_{round_num}",
            )
            eval_messages.append(msg)

        eval_responses = list(grid.send_and_receive(eval_messages))

        # =====================================================================
        # Wrap each eval response into EvaluateRes and let the strategy emit
        # thesis metrics from SUMMED sufficient stats (ADP-06 server half).
        # =====================================================================
        eval_results: List[Tuple[ClientProxy, EvaluateRes]] = []
        round_eval_metrics = []  # retained for RMSE/MAE rating-path fallback (D-18)
        for response in eval_responses:
            if response.has_error():
                continue

            resp_metrics = response.content.get("metrics", MetricRecord())
            metrics_dict = dict(resp_metrics) if resp_metrics else {}
            num_examples = int(metrics_dict.get(
                "num_training_examples",
                metrics_dict.get(
                    "evaluated_users",
                    metrics_dict.get("num-examples", 1),
                ),
            ))
            eval_res = EvaluateRes(
                status=Status(code=Code.OK, message="ok"),
                loss=float(metrics_dict.get("eval_loss", 0.0)),
                num_examples=num_examples,
                metrics=metrics_dict,
            )
            client_id = str(response.metadata.src_node_id)
            proxy = DummyClientProxy(client_id)
            eval_results.append((proxy, eval_res))
            round_eval_metrics.append((num_examples, metrics_dict))

        if eval_results:
            _agg_loss, thesis_metrics = strategy.aggregate_evaluate(round_num, eval_results, [])
            eval_metrics_history[round_num] = dict(thesis_metrics) if thesis_metrics else {}

            # Preserve RMSE/MAE via the legacy per-client-ratio path — D-18 scope-out.
            rating_agg = weighted_average_metrics(round_eval_metrics)
            for rk in ("rmse", "mae", "eval_loss"):
                if rk in rating_agg and rk not in eval_metrics_history[round_num]:
                    eval_metrics_history[round_num][rk] = rating_agg[rk]

            # Merge per-round alpha diagnostics into eval_metrics_history (D-16)
            if round_num in alpha_diagnostics_history:
                for k, v in alpha_diagnostics_history[round_num].items():
                    eval_metrics_history[round_num][f"alpha/{k}"] = v

            rmse = eval_metrics_history[round_num].get('rmse', 'N/A')
            ndcg10 = eval_metrics_history[round_num].get('sampled_ndcg@10', 'N/A')
            hr10 = eval_metrics_history[round_num].get('sampled_hr@10', 'N/A')
            rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
            ndcg10_str = f"{ndcg10:.4f}" if isinstance(ndcg10, (int, float)) else str(ndcg10)
            hr10_str = f"{hr10:.4f}" if isinstance(hr10, (int, float)) else str(hr10)
            print(f"  RMSE: {rmse_str}")
            print(f"  Sampled HR@10: {hr10_str}")
            print(f"  Sampled NDCG@10: {ndcg10_str}")

            # ==================================================================
            # D-27 best-round tracking + D-05 best_prototype snapshot.
            # At the SAME moment best_arrays is captured, snapshot_best_prototype
            # so server-side state is symmetrized.
            # ==================================================================
            if checkpoint_rule in ("best_round_restore", "best_round") and thesis_metrics:
                current_ndcg = float(thesis_metrics.get("sampled_ndcg@10", 0.0))
                if round_num == 1 or current_ndcg > best_metric:
                    best_metric = current_ndcg
                    best_round_num = round_num
                    best_arrays = ArrayRecord({
                        k: v.detach().clone() for k, v in arrays.to_torch_state_dict().items()
                    })
                    # D-05: snapshot prototype at the same moment as best_arrays
                    strategy.snapshot_best_prototype(round_num=round_num, embedding_dim=embedding_dim)
                    print(f"  [CHECKPOINT] New best sampled_ndcg@10={best_metric:.4f} at round {best_round_num}")
                    # =================================================================
                    # D-06.5 (Bug 3 / Path B): snapshot every client's local-state cache
                    # alongside the GLOBAL params. ONLY under best_round_restore — for
                    # best_round and last_round the snapshot is unused and would just
                    # burn ~24 GB of disk per ML-1M run. Hook timing: train + eval
                    # send_and_receive have BOTH returned and aggregate_evaluate ran,
                    # so every client has finished writing its round-N partition_*.pt
                    # file. No race with in-flight client writes.
                    # =================================================================
                    if checkpoint_rule == "best_round_restore":
                        snapshot_cache(cache_root, round_num=best_round_num)

        # =====================================================================
        # Bug 2 discriminator (gated): full-pop eval over users trained >=1 time
        # but NOT this round, bucketed by staleness age. Emits diag/* telemetry
        # only — does NOT feed best_round. Strided (rounds 1-3 + every 5th) to
        # bound cost; aggregate_evaluate is side-effect-free so per-bucket calls
        # are safe. Read: NDCG decaying with age => fixable staleness; NDCG flat-
        # low even at age 1-2 => structural/co-adaptation.
        # =====================================================================
        if diagnostic_fullpop_eval and (round_num <= 3 or round_num % 5 == 0):
            selected_set = set(int(p) for p in selected_pids)
            diag_pids = [pid for pid in last_trained_round if pid not in selected_set]
            if diag_pids:
                _gp = strategy.get_global_prototype()
                _diag_cfg_base: Dict[str, Any] = {
                    "lr": lr, "round_num": int(round_num), "run_id": str(run_id),
                    "reuse_cache": bool(reuse_cache_flag),
                }
                if _gp is not None:
                    _diag_cfg_base["global_prototype"] = _gp.tolist()
                diag_msgs = []
                for pid in diag_pids:
                    nid = partition_to_node_id.get(int(pid))
                    if nid is None:
                        continue
                    diag_msgs.append(grid.create_message(
                        content=RecordDict({"arrays": arrays, "config": ConfigRecord(dict(_diag_cfg_base))}),
                        message_type="evaluate", dst_node_id=nid,
                        group_id=f"diag_eval_round_{round_num}",
                    ))
                diag_responses = list(grid.send_and_receive(diag_msgs))
                bucket_results: Dict[str, List[Tuple[ClientProxy, EvaluateRes]]] = {
                    b[0]: [] for b in _DIAG_AGE_BUCKETS
                }
                all_diag_results: List[Tuple[ClientProxy, EvaluateRes]] = []
                for response in diag_responses:
                    if response.has_error():
                        continue
                    _rm = response.content.get("metrics", MetricRecord())
                    _md = dict(_rm) if _rm else {}
                    _nx = int(_md.get("evaluated_users", _md.get("num_training_examples", 1)))
                    _er = EvaluateRes(
                        status=Status(code=Code.OK, message="ok"),
                        loss=float(_md.get("eval_loss", 0.0)), num_examples=_nx, metrics=_md,
                    )
                    _proxy = DummyClientProxy(str(response.metadata.src_node_id))
                    all_diag_results.append((_proxy, _er))
                    _pid = node_to_pid.get(int(response.metadata.src_node_id))
                    if _pid is not None and _pid in last_trained_round:
                        _age = round_num - last_trained_round[_pid]
                        for _bn, _lo, _hi in _DIAG_AGE_BUCKETS:
                            if _lo <= _age <= _hi:
                                bucket_results[_bn].append((_proxy, _er))
                                break
                if all_diag_results:
                    _l, diag_metrics = strategy.aggregate_evaluate(round_num, all_diag_results, [])
                    eval_metrics_history.setdefault(round_num, {})
                    fp_ndcg = float(diag_metrics.get("sampled_ndcg@10", 0.0))
                    eval_metrics_history[round_num]["diag/fullpop_ndcg@10"] = fp_ndcg
                    eval_metrics_history[round_num]["diag/fullpop_hr@10"] = float(diag_metrics.get("sampled_hr@10", 0.0))
                    eval_metrics_history[round_num]["diag/fullpop_n"] = len(all_diag_results)
                    _parts = [f"fullpop(n={len(all_diag_results)}) ndcg={fp_ndcg:.4f}"]
                    for _bn, _lo, _hi in _DIAG_AGE_BUCKETS:
                        _br = bucket_results[_bn]
                        if _br:
                            _bl, _bm = strategy.aggregate_evaluate(round_num, _br, [])
                            _bndcg = float(_bm.get("sampled_ndcg@10", 0.0))
                            eval_metrics_history[round_num][f"diag/age_{_bn}_ndcg@10"] = _bndcg
                            eval_metrics_history[round_num][f"diag/age_{_bn}_n"] = len(_br)
                            _parts.append(f"age{_bn}(n={len(_br)})={_bndcg:.4f}")
                    print(f"  [DIAG] " + "  ".join(_parts))

        # Log to wandb
        if wandb_enabled and wandb_run is not None:
            round_log = {"round": round_num}
            for key, value in train_metrics_history.get(round_num, {}).items():
                if isinstance(value, (int, float)):
                    round_log[f"train/{key}"] = value
            for key, value in eval_metrics_history.get(round_num, {}).items():
                if isinstance(value, (int, float)):
                    round_log[f"eval/{key}"] = value
            wandb.log(round_log, step=round_num)

        # Check early stopping
        if early_stopper is not None and round_eval_metrics:
            current_eval_metrics = eval_metrics_history.get(round_num, {})
            if early_stopper.step(round_num, current_eval_metrics):
                print(f"\n  Training stopped early at round {round_num}")
                if wandb_enabled and wandb_run is not None:
                    wandb.log({
                        "early_stopped": True,
                        "early_stopped_round": round_num,
                        "best_round": early_stopper.best_round,
                        f"best_{early_stopping_metric}": early_stopper.best_metric,
                    }, step=round_num)
                break

    # Determine actual rounds completed
    actual_rounds = round_num if early_stopper and early_stopper.state.should_stop else num_rounds

    # Print training complete message
    print("\n" + "=" * 70)
    print("FEDERATED TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total rounds completed: {actual_rounds}/{num_rounds}")
    if early_stopper and early_stopper.state.should_stop:
        print(f"Early stopping: Triggered at round {actual_rounds}")
        print(f"Best {early_stopping_metric}: {early_stopper.best_metric:.4f} at round {early_stopper.best_round}")
    print("=" * 70)

    # =========================================================================
    # D-27: restore best-round global params for the manifest artifact.
    # D-07: ALSO restore strategy._global_prototype = strategy.best_prototype
    # BEFORE the final broadcast so clients receiving the final global_prototype
    # see the RESTORED prototype, not last-round drift.
    # =========================================================================
    # Persistence (re-eval enablement, 2026-06-12): keep the LAST-round globals
    # before the restore overwrites `arrays`; both vintages saved at results write.
    last_arrays = arrays
    if checkpoint_rule in ("best_round_restore", "best_round") and best_round_num > 0:
        print(
            f"\n[CHECKPOINT] Restoring global params snapshot from best round {best_round_num} "
            f"(sampled_ndcg@10={best_metric:.4f})"
        )
        arrays = best_arrays
        # D-07: restore the best-round prototype BEFORE the next broadcast so
        # clients see the prototype that corresponds to best_arrays (not last-round drift).
        if strategy.best_prototype is not None:
            strategy._global_prototype = strategy.best_prototype
            print(f"  [D-07] Restored best_prototype (norm={float(np.linalg.norm(strategy.best_prototype)):.4f})")
        # =====================================================================
        # D-06.5 (Bug 3 / Path B): restore the per-client local-state cache to
        # match the round of best_arrays. Without this, each client's cache is
        # at whatever round it was last sampled (median ~best_round/k under
        # cross-device with N=6040 / fraction_train=0.1) — user/item embeddings
        # then come from incompatible generations of training, producing the
        # 3.02× full-pop / in-sample NDCG@10 gap observed in the 100-round cold
        # thesis run (20260505-141804-c3bc5d). Restoring the cache here, BEFORE
        # the D-06 extra-eval-round broadcasts, gives every client a coherent
        # (global, local) state pair to evaluate on.
        # =====================================================================
        if checkpoint_rule == "best_round_restore":
            restored = restore_cache(cache_root, round_num=best_round_num)
            if not restored:
                print(
                    f"  [D-06.5] WARNING: no client cache snapshot found for round "
                    f"{best_round_num}; D-06 full-pop eval will use de-synchronized "
                    f"local state (Bug 3 unfixed for this run)."
                )
    else:
        print(f"\n[CHECKPOINT] checkpoint_rule={checkpoint_rule!r}: keeping last-round params")

    # =========================================================================
    # D-06.7 (Bug 3 / Alt-A): end-of-training calibration pass.
    #
    # Even after Path B's cache snapshot/restore, ~19% of users with LTR > best_round
    # (those sampled in rounds AFTER the best round, before training stopped) get
    # rolled BACK by Path B — their local state regresses to an older vintage.
    # Path B alone produced WORSE full-pop NDCG@10 than no fix in run
    # 20260506-074753-bc134c (0.0563 vs 0.0831 prior). Diagnosis: the snapshot
    # mechanically captured R{best_round} cache state as designed, but for users
    # sampled after best_round their LIVE cache had been MORE RECENTLY trained;
    # restoring the snapshot is a regression for them.
    #
    # Alt-A fixes this by training every partition for ONE local epoch against
    # the restored best-round GLOBAL params. This brings every user's local state
    # into proper alignment with the rolled-back globals BEFORE the D-06 full-pop
    # eval, eliminating the user/item-embedding desynchronization that drove the
    # original 3.02x full-pop / in-sample gap and Path B's 4.33x regression.
    #
    # Returned client params are intentionally DISCARDED — we are NOT updating
    # the server's restored globals (that would defeat best_round_restore). This
    # pass is purely a client-side cache-update.
    # =========================================================================
    if (
        final_calibration_enabled
        and checkpoint_rule in ("best_round_restore", "best_round")
        and best_round_num > 0
    ):
        calib_round_index = actual_rounds + 1
        calib_global_prototype = strategy.get_global_prototype()
        print(
            f"\n[D-06.7] Broadcasting end-of-training calibration pass to all "
            f"{len(partition_to_node_id)} partitions "
            f"(epochs={final_calibration_epochs}, "
            f"prototype_attached={calib_global_prototype is not None})..."
        )

        calib_node_ids = sorted(partition_to_node_id.values())
        calib_messages = []
        for nid in calib_node_ids:
            calib_config_dict: Dict[str, Any] = {
                "lr": lr,
                # No proximal term — we are NOT trying to constrain to the
                # current globals; we are aligning local state to them.
                "proximal_mu": 0.0,
                "round_num": int(calib_round_index),
                "run_id": str(run_id),
                "reuse_cache": bool(reuse_cache_flag),
                # Override client-side `local-epochs` for this calibration pass.
                # Client reads `local_epochs_override` from msg_config in
                # @app.train(); falls back to context.run_config["local-epochs"]
                # when absent so normal training rounds are unaffected.
                "local_epochs_override": int(final_calibration_epochs),
            }
            if calib_global_prototype is not None:
                calib_config_dict["global_prototype"] = calib_global_prototype.tolist()
            calib_config = ConfigRecord(calib_config_dict)
            content = RecordDict({"arrays": arrays, "config": calib_config})
            calib_messages.append(grid.create_message(
                content=content,
                message_type="train",
                dst_node_id=nid,
                group_id=f"calibration_round_{calib_round_index}",
            ))
        calib_responses = list(grid.send_and_receive(calib_messages))

        # Count successes; do NOT aggregate or update server-side params.
        calib_success = sum(1 for r in calib_responses if not r.has_error())
        calib_failed = len(calib_responses) - calib_success
        print(
            f"[D-06.7] Calibration pass complete: {calib_success}/{len(calib_messages)} "
            f"clients succeeded ({calib_failed} errors). "
            f"Server-side globals UNCHANGED — calibration is client-cache-only."
        )

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
        # assigned to strategy.best_prototype at the restore block above.
        final_global_prototype = strategy.get_global_prototype()
        print(
            f"\n[D-06] Broadcasting extra eval round {final_eval_round_index} "
            f"on restored best-round state (best_round={best_round_num}, "
            f"target nodes={len(partition_to_node_id)}, "
            f"prototype_attached={final_global_prototype is not None})..."
        )

        # Fail-closed D-06 guard (post-Job-1 review, 2026-06-12): probe every
        # partition's cached local state (user emb + PersonalMLP/fusion heads)
        # BEFORE broadcasting — a missing file means that user would be scored
        # COLD (the 0.0554-crater bug class; the dual's deep local head is the
        # whole scorer). Count stamped into the best block; strict-d06-cache=true
        # aborts. _cold_start_cache_root is already module-anchored here.
        d06_cache_misses = -1
        if not reuse_cache_flag:
            _probe_root = _cold_start_cache_root(run_id, reuse_cache_flag)
            _missing_pids = [
                pid for pid in sorted(partition_to_node_id.keys())
                if not (_probe_root / f"partition_{pid}.pt").exists()
            ]
            d06_cache_misses = len(_missing_pids)
            if d06_cache_misses:
                print(
                    f"  [D-06 GUARD] {d06_cache_misses}/{len(partition_to_node_id)} partitions "
                    f"have NO cached local state under {_probe_root} "
                    f"(first missing: {_missing_pids[:5]}) — these users would evaluate COLD."
                )
                if bool(context.run_config.get("strict-d06-cache", False)):
                    raise RuntimeError(
                        f"strict-d06-cache: aborting D-06 eval — "
                        f"{d06_cache_misses} partitions lack warm local state"
                    )
            else:
                print(f"  [D-06 GUARD] cache probe OK: all {len(partition_to_node_id)} partitions warm")
        else:
            print("  [D-06 GUARD] skipped (reuse-cache=true: sig-hash dir not resolvable server-side)")

        eval_node_ids = sorted(partition_to_node_id.values())
        extra_eval_messages = []
        for nid in eval_node_ids:
            # BUG FIX (run-id audit, mirrors pfedrec 01d8b72): the D-06 full-pop
            # eval MUST stamp run_id/reuse_cache so the client loads each user's
            # cached LOCAL state (user_embeddings, PersonalMLP, fusion,
            # logit_alpha) from .embedding_cache/{run_id}/ — matching the
            # in-loop eval config above. Without run_id the client fell back to
            # run_id="default" (nonexistent dir) and scored every user with
            # COLD local state — for the dual model the deep local head is the
            # whole scorer, so the D-06 best block cratered to ~0.05 even when
            # the D-06.7 calibration had just written warm heads to {run_id}/.
            extra_eval_config_dict: Dict[str, Any] = {
                "lr": lr,
                "round_num": int(final_eval_round_index),
                "run_id": str(run_id),
                "reuse_cache": bool(reuse_cache_flag),
            }
            # PITFALL 4: attach the restored prototype, mirroring in-loop eval
            # ConfigRecord construction at server_app.py lines 814-815.
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
            # MAJOR fix (plan-checker iteration 1, np.float64 JSON-serialization):
            # coerce numeric values to Python floats at assignment so downstream
            # dataclass_replace + atomic_write_json never raise TypeError on np.float64.
            best_round_metrics = {
                k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                for k, v in (thesis or {}).items()
            }
            # D-06 guard provenance: -1 = not probed (reuse-cache); 0 = all warm;
            # >0 = that many users evaluated with cold local state (suspect run).
            best_round_metrics["d06_cache_misses"] = int(d06_cache_misses)
            print(
                f"[D-06] Extra eval complete. Canonical best/sampled_ndcg@10="
                f"{best_round_metrics.get('sampled_ndcg@10')}"
            )
        else:
            print("[D-06] WARNING: no extra-eval responses; best block falls back to in-loop value.")

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

    final_metrics: Dict[str, Any] = {
        "best": best_round_metrics or last_block,
        "last": last_block,
        "best_round": best_round_num if best_round_num > 0 else last_round,
        "last_round": last_round,
        "final_eval_round_index": final_eval_round_index,
    }

    # =========================================================================
    # FEDERATED-EVAL-ONLY final metrics (split learning cannot run centralized
    # eval — the server never sees the LOCAL user rows).
    # =========================================================================
    print("\n  Using federated evaluation metrics...")
    print("  (Centralized evaluation not possible in split learning)")

    # Print evaluation results
    print_evaluation_metrics(final_metrics["best_round"], final_metrics["best"], context)

    # Log final metrics to wandb
    if wandb_enabled and wandb_run is not None:
        # Phase 6 Plan 05: ship best block to W&B under final_eval/best/ namespace.
        final_log = {"round": actual_rounds + 1}
        for key, value in final_metrics["best"].items():
            if isinstance(value, (int, float)):
                final_log[f"final_eval/best/{key}"] = value
        wandb.log(final_log, step=actual_rounds + 1)

        # W&B run.summary: best/* and last/* namespaces (EVL-06).
        # alpha/* + prototype/* + early_stopping/* + training/* surfaces preserved verbatim below.
        for key, value in final_metrics["best"].items():
            if isinstance(value, (int, float)):
                wandb.run.summary[f"best/{key}"] = value
        for key, value in final_metrics["last"].items():
            if isinstance(value, (int, float)):
                wandb.run.summary[f"last/{key}"] = value
        wandb.run.summary["best_round"] = final_metrics["best_round"]
        wandb.run.summary["last_round"] = final_metrics["last_round"]
        wandb.run.summary["final_eval_round_index"] = final_metrics["final_eval_round_index"]

        if early_stopper:
            wandb.run.summary["early_stopping/enabled"] = early_stopping_enabled
            wandb.run.summary["early_stopping/stopped_early"] = early_stopper.state.should_stop
            wandb.run.summary["early_stopping/best_round"] = early_stopper.best_round
            wandb.run.summary[f"early_stopping/best_{early_stopping_metric}"] = early_stopper.best_metric
            wandb.run.summary["training/actual_rounds"] = actual_rounds

    # =========================================================================
    # AlphaAnalyzer integration (D-18 preserve verbatim from pre-Phase-4 code).
    # Uses per_client_metrics_history for per-client alpha correlation analysis.
    # =========================================================================
    alpha_analyzer = AlphaAnalyzer()
    alpha_entry_id = 0
    for rnd, client_metrics_list in per_client_metrics_history.items():
        for client_id, metrics_dict in client_metrics_list:
            if "client_alpha" in metrics_dict:
                alpha_analyzer.add_client_data(
                    client_id=alpha_entry_id,  # Unique integer ID for each client-round pair
                    alpha=metrics_dict["client_alpha"],
                    metrics={k: v for k, v in metrics_dict.items() if k not in ["client_alpha", "num-examples"] and isinstance(v, (int, float))}
                )
                alpha_entry_id += 1

    # Log alpha statistics
    alpha_stats = alpha_analyzer.compute_statistics()
    if alpha_stats.count > 0:
        print("\n  Adaptive Alpha Analysis:")
        print(f"  Mean alpha: {alpha_stats.mean:.4f} (std: {alpha_stats.std:.4f})")
        print(f"  Range: [{alpha_stats.min:.4f}, {alpha_stats.max:.4f}]")
        print(f"  Quartiles: Q25={alpha_stats.q25:.4f}, Median={alpha_stats.median:.4f}, Q75={alpha_stats.q75:.4f}")

        if wandb_enabled and wandb_run is not None:
            wandb.run.summary["alpha/mean"] = alpha_stats.mean
            wandb.run.summary["alpha/std"] = alpha_stats.std
            wandb.run.summary["alpha/min"] = alpha_stats.min
            wandb.run.summary["alpha/max"] = alpha_stats.max

    # Get global prototype info from strategy
    final_prototype = strategy.get_global_prototype()
    prototype_norm = float(np.linalg.norm(final_prototype)) if final_prototype is not None else None

    if prototype_norm is not None:
        print(f"\n  Global Prototype:")
        print(f"  Final norm: {prototype_norm:.4f}")
        if wandb_enabled and wandb_run is not None:
            wandb.run.summary["prototype/final_norm"] = prototype_norm

    # =========================================================================
    # Results JSON — Phase 4 additions: D-25 contract keys in federated_config,
    # D-26 selected_clients_per_round, D-13 cold_starts block,
    # D-27 checkpoint block, D-16 alpha_diagnostics_history, D-15 embedded _manifest.
    # =========================================================================
    total_selections = sum(len(r) for r in selected_clients_per_round)
    cold_start_rate = (total_cold_starts / total_selections) if total_selections else 0.0

    # Determine local params based on model type
    local_params_list = ["user_embeddings.weight", "user_bias.weight"]
    if model_type == "dual":
        if context.run_config.get("enable-per-user-alpha", False):
            local_params_list.append("_logit_alpha.weight")
        if context.run_config.get("enable-item-perturbation", False):
            local_params_list.append("_item_perturbation.weight")
        local_params_list.append("personal_mlp.*")
        if fusion_type == "gate":
            local_params_list.append("fusion_gate")
        elif fusion_type == "concat":
            local_params_list.extend(["fusion_layer.weight", "fusion_layer.bias"])

    # Prepare early stopping summary for results
    early_stopping_summary = None
    if early_stopper:
        early_stopping_summary = early_stopper.get_summary()

    results_data: Dict[str, Any] = {
        "model_name": f"{model_type.upper()}_MF_Personalized_Split_{strategy_name.upper()}_Adaptive",
        "dataset": "ml-1m",
        "architecture": "split_learning_adaptive" if model_type != "dual" else "dual_level_personalization",
        "federated_config": {
            "num_rounds": num_rounds,
            "actual_rounds": actual_rounds,
            "num_clients": len(list(grid.get_node_ids())),
            "fraction_train": fraction_train,
            "strategy": strategy_name,
            "proximal_mu": proximal_mu,
            "model_type": model_type,
            "embedding_dim": embedding_dim,
            "dropout": dropout,
            "learning_rate": lr,
            "split_learning": True,
            "global_params": ["item_embeddings.weight", "item_bias.weight", "global_bias"],
            "local_params": local_params_list,
            # Dual model specific config
            "mlp_hidden_dims": mlp_hidden_dims if model_type == "dual" else None,
            "fusion_type": fusion_type if model_type == "dual" else None,
            # D-25 contract keys
            "mode": mode,
            "run_seed": run_seed,
            "weight_policy": weight_policy,
            "checkpoint_rule": checkpoint_rule,
            "reuse_cache": reuse_cache_flag,
            "prototype_momentum": prototype_momentum,
        },
        "adaptive_config": {
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "quantity_threshold": alpha_quantity_threshold,
            "quantity_temperature": alpha_quantity_temperature,
            "prototype_momentum": prototype_momentum,
            "enable_per_user_alpha": context.run_config.get("enable-per-user-alpha", False),
            "enable_item_perturbation": context.run_config.get("enable-item-perturbation", False),
            "item_perturbation_reg": context.run_config.get("item-perturbation-reg", 0.01),
            "contrastive_lambda": context.run_config.get("contrastive-lambda", 0.0),
            "contrastive_tau": context.run_config.get("contrastive-tau", 0.1),
        },
        "early_stopping": early_stopping_summary,
        "alpha_analysis": {
            "mean": alpha_stats.mean,
            "std": alpha_stats.std,
            "min": alpha_stats.min,
            "max": alpha_stats.max,
            "median": alpha_stats.median,
        } if alpha_stats.count > 0 else None,
        "global_prototype_norm": prototype_norm,
        "timestamp": datetime.now().isoformat(),
        "final_metrics": final_metrics,
        "training_rounds": actual_rounds,
        "eval_metrics_history": eval_metrics_history,
        "train_metrics_history": train_metrics_history,
        # D-26: partition-id space (stable 0..N-1), not ephemeral node_ids.
        "selected_clients_per_round": selected_clients_per_round,
        # D-27 in-memory best-round checkpoint block.
        "checkpoint": {
            "rule": checkpoint_rule,
            "best_round": best_round_num,
            "best_sampled_ndcg@10": best_metric if best_metric != float("-inf") else None,
        },
        # D-13 cold-start accounting.
        "cold_starts": {
            "per_round": cold_starts_per_round,
            "total_cold_starts": total_cold_starts,
            "total_client_selections": total_selections,
            "cold_start_rate": cold_start_rate,
        },
    }

    # D-16 alpha diagnostics history
    if alpha_diagnostics_history:
        results_data["alpha_diagnostics_history"] = {
            int(r): {k: float(v) for k, v in d.items()}
            for r, d in alpha_diagnostics_history.items()
        }

    # =========================================================================
    # PSN-07 / ADP-08: protocol fingerprint manifest (FND-07 + D-15 double-write).
    # module="adaptive" (Phase-4-specific).
    # =========================================================================
    foundation_idx = verify_bundle(data_derived())
    split_mf = load_split_manifest(data_derived() / "split_manifest.json")
    manifest = build_run_manifest(
        run_id=run_id,
        mode_profile=profile,
        run_seed=run_seed,
        mapping_sha256=foundation_idx.mapping_sha256,
        split_hash=foundation_idx.split_hash,
        exclusion_sha256=foundation_idx.exclusion_sha256,
        foundation_contract_sha256=foundation_idx.foundation_contract_sha256,
        raw_data_hash=split_mf.raw_data_hash,
        builder_version=split_mf.builder_version,
        overrides=overrides,
        module="adaptive",
    )

    # Phase 6 Plan 05 Edit 6: mutate manifest with final_eval_round_index + metrics
    # BEFORE embed_manifest_in_result so the embedded _manifest dict carries schema-v2 fields.
    # Phase 7 D-22: thesis-tagging fields read from run_config; sentinels for non-thesis runs.
    # The Phase-4 best_prototype post-embed mutation below is PRESERVED verbatim — it layers
    # on top of this Phase-6 schema-v2 metrics field + Phase-7 thesis fields (not a replacement).
    manifest = dataclass_replace(
        manifest,
        final_eval_round_index=final_eval_round_index,
        metrics=results_data["final_metrics"],
        thesis_run_label=str(context.run_config.get("thesis-run-label", "")),
        ablation_dimension=str(context.run_config.get("ablation-dimension", "none")),
        ablation_value=str(context.run_config.get("ablation-value", "")),
    )

    # D-15 part 1: embed manifest INTO the result JSON.
    embed_manifest_in_result(manifest, results_data)

    # Phase 4 D-06 — DO NOT TOUCH (preserved verbatim by Phase 6):
    # D-06: embed best_prototype in the _manifest dict AFTER embed_manifest_in_result
    # mutates results_data. The _manifest dict is extensible — Research §Pattern 2
    # confirms post-hoc mutation is safe (dict is held by reference).
    if strategy.best_prototype is not None:
        results_data["_manifest"]["best_prototype"] = [float(x) for x in strategy.best_prototype.tolist()]
    else:
        results_data["_manifest"]["best_prototype"] = None

    # Phase 6 Plan 05 Edit 7: repo-root-anchored per-run dir (D-02) + atomic write.
    # benchmark_cross_device, thesis_crossdevice_main, and paper_compat_pfedrec: per-run-dir (D-01) + clean filename (D-04).
    # Phase 7 D-04: thesis_crossdevice_main joins the per-run-dir gate.
    # Note: the D-02 guard above raises NotImplementedError for cross_silo_legacy
    # before reaching here, so the else-branch is a safety net for unknown future modes.
    print("\nSaving evaluation results...")
    if mode in ("benchmark_cross_device", "thesis_crossdevice_main", "paper_compat_pfedrec"):
        run_dir = module_run_results_dir(_MODULE, run_id)
        results_filename = run_dir / "results.json"
        atomic_write_json(str(results_filename), results_data)
        # D-04: clean per-run-dir filename via sibling_name="manifest.json".
        sibling_path = write_manifest_sibling(manifest, results_filename, sibling_name="manifest.json")
        # Persistence (re-eval enablement, 2026-06-12): save restored-BEST +
        # LAST-round GLOBAL params (+ the restored prototype) so finished runs
        # stay re-evaluable offline (p_u/PersonalMLP variant tests).
        try:
            torch.save(arrays.to_torch_state_dict(), run_dir / "global_state_best.pt")
            torch.save(last_arrays.to_torch_state_dict(), run_dir / "global_state_last.pt")
            _proto = strategy.get_global_prototype()
            if _proto is not None:
                np.save(str(run_dir / "best_prototype.npy"), _proto)
            print(f"  Global state saved: {run_dir}/global_state_{{best,last}}.pt")
        except Exception as _pe:  # noqa: BLE001
            print(f"[WARN] global-state persistence failed (non-fatal): {_pe}")
    else:
        # Fallback for any non-cross-device mode that does NOT raise at D-02 guard.
        # Uses repo_root() anchor not module-relative path. D-03 coexistence preserved.
        from fedrec_foundation.paths import repo_root as _repo_root
        _legacy_dir = _repo_root() / "results" / "federated" / "adaptive"
        _legacy_dir.mkdir(parents=True, exist_ok=True)
        results_filename = _legacy_dir / f"{run_id}_results.json"
        atomic_write_json(str(results_filename), results_data)
        sibling_path = write_manifest_sibling(manifest, results_filename)
    print(f"Results saved to: {results_filename.resolve()}")
    print(f"Manifest sibling: {sibling_path.resolve()}")

    # W&B: attach manifest fingerprints + D-13 summary to the run's config
    # so dashboards can filter/audit by contract hashes.
    if wandb_enabled and wandb_run is not None:
        wandb.config.update({
            "_manifest": {
                "run_id": manifest.run_id,
                "mode": manifest.mode,
                "num_supernodes": manifest.num_supernodes,
                "foundation_contract_sha256": manifest.foundation_contract_sha256,
                "split_hash": manifest.split_hash,
                "run_seed": manifest.run_seed,
                "checkpoint_rule": manifest.checkpoint_rule,
            }
        })
        wandb.run.summary["total_cold_starts"] = int(total_cold_starts)
        wandb.run.summary["cold_start_rate"] = float(cold_start_rate)
        if strategy.best_prototype is not None:
            wandb.run.summary["best_prototype_norm"] = float(np.linalg.norm(strategy.best_prototype))

    # Finish wandb run
    if wandb_enabled and wandb_run is not None:
        wandb.finish()
        print("  Weights & Biases run completed")

    # =========================================================================
    # D-06.5 (Bug 3 / Path B) cleanup: drop the snapshot dir under the live
    # cache_root to free ~24 GB. The live partition_*.pt files (post-restore
    # under best_round_restore) are LEFT INTACT so future warm-start workflows
    # can clone them via reflink. No-op when checkpoint_rule != best_round_restore.
    # =========================================================================
    if checkpoint_rule == "best_round_restore":
        cleanup_snapshots(cache_root)
