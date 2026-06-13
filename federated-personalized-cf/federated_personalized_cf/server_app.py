"""federated-personalized-cf: Split Learning for Personalized Collaborative Filtering (Phase 3 Plan 04).

Cross-device migration mirrors Phase 2 Plans 04 + 05:

- **D-25 mode resolver** at @app.main() entry; every hyperparameter is read as
  ``int(context.run_config.get(key, profile.field))`` so pyproject values are
  only the override surface.
- **D-02 frozen cross-silo**: ``mode="cross_silo_legacy"`` raises
  ``NotImplementedError`` at startup BEFORE any training or data load — Phase 3
  removes multi-user-per-client support (see CONTEXT.md §Deferred).
- **G-03-01 discovery round**: one-shot ``@app.evaluate(discover_only=true)``
  broadcast BEFORE the main loop to build ``partition_to_node_id: Dict[int, int]``
  so per-round client sampling can work in stable partition-id space (0..N-1)
  instead of Flower's ephemeral node_ids.
- **PSN-04 seeded sampling**: ``_server_sampler = server_rng(run_seed)``
  instantiated ONCE pre-loop; ``_server_sampler.sample(range(N), k)`` per round
  yields a byte-identical partition-id sequence for a given run_seed.
- **PSN-04 strategy wire-up**: ``PersonalizedSplitFedAvg`` / ``PersonalizedSplitFedProx``
  from Plan 01 replaces raw ``FedAvg`` / ``FedProx``; ``strategy.aggregate_evaluate``
  returns thesis metrics from summed sufficient stats.
- **D-27 best-round restore** (in-memory; no disk writes) tracked during the
  FL loop; restored before result-JSON write so the reported best metric is
  canonical per STATE.md ``best_*`` convention.
- **D-15 double-write manifest** with ``module="personalized"``: embedded
  ``_manifest`` key in the result JSON + sibling ``<run_id>-manifest.json``.
- **D-13 cold-start counter (Phase-3-unique)**: per-round count of selected
  partitions whose local-state cache (``.embedding_cache/{run_id}/partition_{pid}.pt``
  or ``.embedding_cache/sig_<hash>/partition_{pid}.pt`` under D-09 reuse-cache=true)
  did NOT exist BEFORE this round. Accumulated as ``total_cold_starts`` and
  reported as ``cold_start_rate`` in the final results JSON + W&B summary.

NOTE: Centralized evaluation is NOT possible in split learning (the server
only holds GLOBAL params — item embeddings, item bias, global bias). Final
headline metrics come from the strategy-aggregated federated eval path.
D-27 best-round restore is still valuable for the ArrayRecord snapshot that
gets written into the manifest artifact.

D-18 surgical edit discipline: DummyClientProxy, weighted_average_metrics,
print_evaluation_metrics, EarlyStopping setup/teardown, and the W&B init
block are preserved — only the @app.main() body is rip-and-replaced.
"""

import torch
import json
import wandb
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
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

from federated_personalized_cf.task import get_model
from federated_personalized_cf.strategy import (
    PersonalizedSplitFedAvg,
    PersonalizedSplitFedProx,
)
from federated_personalized_cf.early_stopping import EarlyStopping

# Phase 3 Plan 04: foundation imports (PSN-04, PSN-07, D-13, D-15, D-25, D-27).
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
from fedrec_foundation.paths import module_run_results_dir, repo_root
from fedrec_foundation.atomic import atomic_write_json
from fedrec_foundation.rng import server_rng
from fedrec_foundation.split import load_split_manifest
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

    Args:
        metrics: List of (num_examples, metrics_dict) tuples from each client

    Returns:
        Dictionary of aggregated metrics
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {}

    aggregated = {}

    if metrics:
        metric_keys = metrics[0][1].keys()

        for key in metric_keys:
            if key == "num-examples":
                continue

            # Only aggregate numeric values
            first_value = metrics[0][1].get(key)
            if not isinstance(first_value, (int, float)):
                continue

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

    Args:
        round_num: Current federated learning round
        metrics: Aggregated metrics dictionary
        context: Flower context with configuration
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
        k_values_str = context.run_config.get("ranking-k-values", "5,10,20")
        k_values = [int(k.strip()) for k in k_values_str.split(",")]

        has_ranking = any(f"hit_rate@{k}" in metrics for k in k_values)

        if has_ranking:
            print("\n🎯 Ranking Metrics:")

            if "mrr" in metrics:
                print(f"  MRR:       {metrics['mrr']:.4f}")

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

            # Diversity/Popularity metrics
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

    Mirrors ``client_app._cache_dir_for_run`` in shape but does NOT need to
    know the signature fields: at the server, for the D-13 counter, we only
    need to know whether a ``partition_{pid}.pt`` file exists before the
    round fires. Under D-09 reuse-cache=true the path becomes run-agnostic;
    the server cannot construct the sig_<hash> dir without access to the
    client-side signature fields, so under reuse-cache=true the counter is
    short-circuited to zero (the cache is expected to hit on every client —
    it's the whole point of D-09) and the caller logs that fact.

    Parameters
    ----------
    run_id : str
    reuse_cache : bool

    Returns
    -------
    Path
        ``Path(".embedding_cache") / run_id`` (for default D-08 behaviour).
    """
    # CWD fix (post-Job-1 review, 2026-06-12): the server process CWD is the
    # repo root (scripts/run.py), but clients write the cache MODULE-LOCAL
    # (Ray actors run in the app dir) — a bare relative path made the D-13
    # probe miss every file and report cold_start_rate=1.0 spuriously.
    # Anchor to the module root: <repo>/federated-personalized-cf/.
    return Path(__file__).resolve().parents[1] / ".embedding_cache" / run_id


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp (Phase 3 Plan 04).

    Phase-3 cross-device migration; see module docstring for the full
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
            f"⚠ OVERRIDE: {len(overrides)} key(s) diverge from mode default. "
            f"Run is NOT comparable to benchmark thesis table."
        )

    # =========================================================================
    # D-02 frozen-cross-silo guard. Personalized cross-device migration removed
    # multi-user-per-client support; cross-silo numbers live in pre-Phase-3
    # git history and are not re-derived. The guard fires BEFORE any model
    # load or training so a wrong-mode invocation fails loud immediately.
    # =========================================================================
    if mode == "cross_silo_legacy":
        raise NotImplementedError(
            "Personalized cross-device migration removed multi-user-per-client support "
            "per D-02. Check out a pre-Phase-3 commit (see "
            ".planning/phases/03-personalized-migration/03-CONTEXT.md §Deferred) "
            "to reproduce legacy cross-silo numbers."
        )

    # =========================================================================
    # D-25 hyperparameters — profile is source of truth; run_config overrides win.
    # =========================================================================
    num_rounds: int = int(context.run_config.get("num-server-rounds", profile.num_server_rounds))
    fraction_train: float = float(context.run_config.get("fraction-train", profile.fraction_train))
    lr: float = float(context.run_config.get("lr", profile.lr))
    model_type: str = str(context.run_config.get("model-type", "bpr"))
    embedding_dim: int = int(context.run_config.get("embedding-dim", profile.embedding_dim))
    dropout: float = float(context.run_config.get("dropout", 0.1))

    strategy_name: str = str(context.run_config.get("strategy", "fedavg")).lower()
    proximal_mu: float = float(context.run_config.get("proximal-mu", 0.0))

    weight_policy: str = str(context.run_config.get("weight-policy", profile.weight_policy))
    checkpoint_rule: str = str(
        context.run_config.get(
            "checkpoint-rule",
            getattr(profile, "checkpoint_rule", "best_round_restore"),
        )
    )
    reuse_cache_flag: bool = bool(context.run_config.get("reuse-cache", False))  # D-09

    # D-06.7 calibration knobs (port from pfedrec 01d8b72) — parsed HERE so a
    # malformed --run-config value fails at startup, not after a 30h run.
    # Consumed by the D-06.7 block before the D-06 eval.
    final_calibration_enabled: bool = bool(
        context.run_config.get("final-calibration-enabled", False)
    )
    final_calibration_epochs: int = int(
        context.run_config.get("final-calibration-epochs", 1)
    )

    # Materialize the run_id early so the D-13 cold-start probe resolves to
    # the same cache dir the client will write into this round.
    run_id = str(context.run_config.get("run-id", "")) or generate_run_id()
    _MODULE: str = "personalized"   # cross-references: build_run_manifest, module_run_results_dir

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
    # go to a dedicated W&B project).
    # =========================================================================
    wandb_enabled = context.run_config.get("wandb-enabled", False)
    wandb_run = None
    if wandb_enabled:
        wandb_config = {
            "run_id": run_id,
            "num_rounds": num_rounds,
            "fraction_train": fraction_train,
            "local_epochs": context.run_config.get("local-epochs", 5),
            "strategy": strategy_name,
            "proximal_mu": proximal_mu,
            "model_type": model_type,
            "embedding_dim": embedding_dim,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": context.run_config.get("weight-decay", 1e-5),
            "alpha": context.run_config.get("alpha", 0.5),
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
            else "federated-cf"
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

    global_model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )

    num_params = sum(p.numel() for p in global_model.parameters())
    print(f"  Total parameters (local + global union): {num_params:,}")

    # Split-learning: only send GLOBAL params to clients.
    arrays = ArrayRecord(global_model.get_global_parameters())

    # =========================================================================
    # G-03-01: discovery round. Build partition_id -> node_id mapping BEFORE
    # the main loop so per-round sampling runs in stable partition-id space
    # (0..N-1) instead of Flower's os.urandom-seeded ephemeral node_id space.
    # Clones the Phase 2 Plan 05 pattern verbatim — Plan 03 already wired the
    # client-side discover_only short-circuit.
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

    # =========================================================================
    # PSN-04: strategy wire-up. PersonalizedSplitFedAvg / PersonalizedSplitFedProx
    # override aggregate_evaluate to emit thesis metrics from SUMMED sufficient
    # stats (not averaged per-client ratios); aggregate_fit is inherited unchanged
    # from parent (D-23 split-learning invariant — client only sends GLOBAL params).
    # =========================================================================
    if strategy_name == "fedprox":
        strategy = PersonalizedSplitFedProx(
            fraction_fit=fraction_train,
            proximal_mu=proximal_mu,
        )
        print(f"  Strategy: PersonalizedSplitFedProx (proximal_mu={proximal_mu})")
    else:
        strategy = PersonalizedSplitFedAvg(
            fraction_fit=fraction_train,
        )
        print(f"  Strategy: PersonalizedSplitFedAvg")

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
    eval_metrics_history: Dict[int, Dict] = {}

    # PSN-04: seeded RNG for per-round client selection — one instance for the
    # whole run, so the sequence across rounds is stable for a given run_seed.
    _server_sampler = server_rng(run_seed)
    selected_clients_per_round: List[List[int]] = []  # D-26 persisted in result JSON

    # D-27 best-round tracking (in-memory; no disk writes).
    best_metric: float = float("-inf")
    best_round_num: int = 0
    best_arrays = arrays  # fallback if no eval round improves

    # D-13 cold-start counter (Phase-3-unique). Tracked per round AND accumulated.
    total_cold_starts: int = 0
    cold_starts_per_round: List[int] = []
    cache_root = _cold_start_cache_root(run_id, reuse_cache_flag)

    # Track the last executed round so post-loop bookkeeping (early stop) can
    # report the correct final round.
    round_num = 0

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*50}")
        print(f"Round {round_num}/{num_rounds}")
        print(f"{'='*50}")

        # Pass run-id + round metadata so clients resolve the same cache path
        # the D-13 counter is probing at the server.
        train_config = ConfigRecord({
            "lr": lr,
            "proximal_mu": proximal_mu,
            "round_num": int(round_num),
            "run_id": str(run_id),
            "reuse_cache": bool(reuse_cache_flag),
        })

        # =====================================================================
        # G-03-01: sample in partition-id space (stable 0..N-1), translate to
        # node_ids for message addressing. Deterministic across runs for a
        # given run_seed because the sampling DOMAIN is now seed-independent.
        # =====================================================================
        num_selected = max(1, int(expected_n * fraction_train))
        selected_pids: List[int] = _server_sampler.sample(range(expected_n), num_selected)
        selected_node_ids = [partition_to_node_id[pid] for pid in selected_pids]

        selected_clients_per_round.append([int(pid) for pid in selected_pids])

        # =====================================================================
        # D-13 cold-start counter (Phase-3-unique). Probe the cache BEFORE
        # the train message is sent: counts partitions for whom no local-state
        # .pt file exists yet. Under D-09 reuse-cache=true the server cannot
        # cheaply resolve the sig_<hash> path without the client-side signature
        # (split_hash / dim / method / run_id / num_users / num_items) — the
        # expected regime is "all hot because sig matches", so the counter is
        # short-circuited to 0 and the log line names D-09 explicitly.
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

        # Parse training responses and aggregate.
        fit_results = []
        round_train_metrics = []

        for response in train_responses:
            if response.has_error():
                print(f"  Warning: Client {response.metadata.src_node_id} returned error")
                continue

            resp_arrays = response.content.get("arrays", ArrayRecord())
            resp_metrics = response.content.get("metrics", MetricRecord())

            metrics_dict = dict(resp_metrics) if resp_metrics else {}
            num_examples = int(metrics_dict.get(
                "num_training_examples",
                metrics_dict.get("num-examples", 1),
            ))

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

        # Aggregate training results using strategy (inherited FedAvg.aggregate_fit
        # under D-23 — averages only GLOBAL params because the client only sent them).
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

            train_loss = train_metrics_history[round_num].get('train_loss', 'N/A')
            if isinstance(train_loss, (int, float)):
                print(f"  Training loss: {train_loss:.4f}")

        # =====================================================================
        # EVALUATION PHASE
        # =====================================================================
        eval_messages = []
        for node_id in selected_node_ids:
            eval_config = ConfigRecord({
                "lr": lr,
                "round_num": int(round_num),
                "run_id": str(run_id),
                "reuse_cache": bool(reuse_cache_flag),
            })
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
        # PSN-04: wrap each eval response into EvaluateRes and let the strategy
        # emit thesis metrics from SUMMED sufficient stats.
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

            rmse = eval_metrics_history[round_num].get('rmse', 'N/A')
            ndcg10 = eval_metrics_history[round_num].get('sampled_ndcg@10', 'N/A')
            hr10 = eval_metrics_history[round_num].get('sampled_hr@10', 'N/A')
            rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
            ndcg10_str = f"{ndcg10:.4f}" if isinstance(ndcg10, (int, float)) else str(ndcg10)
            hr10_str = f"{hr10:.4f}" if isinstance(hr10, (int, float)) else str(hr10)
            print(f"  RMSE: {rmse_str}")
            print(f"  Sampled HR@10: {hr10_str}")
            print(f"  Sampled NDCG@10: {ndcg10_str}")

            # D-27 best-round tracking (in-memory; no disk writes).
            if checkpoint_rule in ("best_round_restore", "best_round") and thesis_metrics:
                current_ndcg = float(thesis_metrics.get("sampled_ndcg@10", 0.0))
                if round_num == 1 or current_ndcg > best_metric:
                    best_metric = current_ndcg
                    best_round_num = round_num
                    best_arrays = ArrayRecord({
                        k: v.detach().clone() for k, v in arrays.to_torch_state_dict().items()
                    })
                    print(f"  [CHECKPOINT] New best sampled_ndcg@10={best_metric:.4f} at round {best_round_num}")

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
                print(f"\n⏹ Training stopped early at round {round_num}")
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
    # Centralized evaluation is NOT possible in split learning (the server
    # only has GLOBAL params), so the "restore" here is the ArrayRecord
    # snapshot that goes into the manifest + any downstream loader that
    # needs the best-round item embeddings. Final reported headline metrics
    # come from eval_metrics_history[best_round_num] (federated aggregation).
    # =========================================================================
    # Persistence (re-eval enablement, 2026-06-12): keep a handle on the
    # LAST-round globals before the best-round restore overwrites `arrays`,
    # so both vintages can be saved into the run dir at the results write.
    last_arrays = arrays
    if checkpoint_rule in ("best_round_restore", "best_round") and best_round_num > 0:
        print(
            f"\n[CHECKPOINT] Restoring global params snapshot from best round {best_round_num} "
            f"(sampled_ndcg@10={best_metric:.4f})"
        )
        arrays = best_arrays
    else:
        print(f"\n[CHECKPOINT] checkpoint_rule={checkpoint_rule!r}: keeping last-round params")

    # =========================================================================
    # D-06.7 (calibration-uniformity port from pfedrec 01d8b72): optional
    # end-of-training calibration pass. When `final-calibration-enabled=true`,
    # after best-round restore and BEFORE the D-06 full-pop eval, broadcast
    # `final-calibration-epochs` local epoch(s) of training to ALL partitions
    # against the restored best-round globals, so each user re-saves LOCAL
    # user_embeddings/user_bias aligned to the item embeddings D-06 evaluates
    # with (users not sampled in the best round otherwise carry a stale
    # vintage). Returned params are DISCARDED — server globals stay restored.
    # proximal_mu=0: we are aligning locals to the restored globals, not
    # constraining the globals. NOTE: unlike pfedrec there is no freeze-items
    # variant — the BPR-MF train path has a single optimizer, and the client's
    # locally-drifted global copy is discarded; the shallow user-embedding
    # local state realigns in 1 epoch. Default false → zero effect on existing
    # runs. Required so the final-matrix protocol can enable calibration
    # uniformly across all four modules.
    # =========================================================================
    if (
        final_calibration_enabled
        and checkpoint_rule in ("best_round_restore", "best_round")
        and best_round_num > 0
    ):
        calib_round_index = actual_rounds + 1
        print(
            f"\n[D-06.7] Broadcasting end-of-training calibration pass to all "
            f"{len(partition_to_node_id)} partitions "
            f"(epochs={final_calibration_epochs})..."
        )
        calib_node_ids = sorted(partition_to_node_id.values())
        calib_messages = []
        for nid in calib_node_ids:
            calib_config = ConfigRecord({
                "lr": lr,
                "proximal_mu": 0.0,
                "round_num": int(calib_round_index),
                "run_id": str(run_id),
                "reuse_cache": bool(reuse_cache_flag),
                "local_epochs_override": int(final_calibration_epochs),
            })
            content = RecordDict({"arrays": arrays, "config": calib_config})
            calib_messages.append(grid.create_message(
                content=content,
                message_type="train",
                dst_node_id=nid,
                group_id=f"calibration_round_{calib_round_index}",
            ))
        calib_responses = list(grid.send_and_receive(calib_messages))
        calib_success = sum(1 for r in calib_responses if not r.has_error())
        calib_failed = len(calib_responses) - calib_success
        print(
            f"[D-06.7] Calibration pass complete: {calib_success}/{len(calib_messages)} "
            f"clients succeeded ({calib_failed} errors). "
            f"Server-side globals UNCHANGED — calibration is client-cache-only."
        )

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

        # Fail-closed D-06 guard (post-Job-1 review, 2026-06-12): probe every
        # partition's cached local state BEFORE broadcasting. A missing file
        # means that user would be scored with COLD local state — the silent
        # bug class behind the 0.0711/0.0554 craters. The miss count is
        # stamped into the best block (d06_cache_misses); claim runs set
        # strict-d06-cache=true to ABORT instead of reporting a cold number.
        # reuse-cache=true uses a sig-hash dir the server cannot resolve —
        # guard skipped with a notice (probe count = -1).
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
            # cached LOCAL state (user_embeddings/user_bias) from
            # .embedding_cache/{run_id}/ — matching the in-loop eval config
            # above. Without run_id the client fell back to run_id="default"
            # (nonexistent dir) and scored every user with COLD Xavier-init
            # local rows, understating the canonical best block.
            eval_config = ConfigRecord({
                "lr": lr,
                "round_num": int(final_eval_round_index),
                "run_id": str(run_id),
                "reuse_cache": bool(reuse_cache_flag),
            })
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
            _agg_loss, thesis = strategy.aggregate_evaluate(final_eval_round_index, extra_results, [])
            # MAJOR fix (plan-checker iteration 1, np.float64 JSON-serialization):
            # coerce numeric values to Python floats at assignment so downstream
            # dataclass_replace + atomic_write_json never raise TypeError on
            # np.float64. Path (b) from checker spec.
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

    # Log final metrics to wandb
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

    # =========================================================================
    # Results JSON — Phase 3 additions: D-25 contract keys in federated_config,
    # D-26 selected_clients_per_round (partition-id space), D-13 cold_starts
    # block, D-27 checkpoint block, D-15 embedded _manifest.
    # =========================================================================
    total_selections = sum(len(r) for r in selected_clients_per_round)
    cold_start_rate = (total_cold_starts / total_selections) if total_selections else 0.0

    results_data = {
        "model_name": f"{model_type.upper()}_MF_Personalized_Split_{strategy_name.upper()}",
        "dataset": "ml-1m",
        "architecture": "split_learning",
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
            "local_params": ["local_user_row", "local_user_bias"],
            # D-25 contract keys
            "mode": mode,
            "run_seed": run_seed,
            "weight_policy": weight_policy,
            "checkpoint_rule": checkpoint_rule,
            "reuse_cache": reuse_cache_flag,
        },
        "early_stopping": early_stopper.get_summary() if early_stopper else None,
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
        # D-13 cold-start accounting (Phase-3-unique).
        "cold_starts": {
            "per_round": cold_starts_per_round,
            "total_cold_starts": total_cold_starts,
            "total_client_selections": total_selections,
            "cold_start_rate": cold_start_rate,
        },
    }

    # =========================================================================
    # PSN-07: protocol fingerprint manifest (FND-07 + D-15 double-write).
    # module="personalized" (Phase-3-specific).
    # =========================================================================
    foundation_idx = verify_bundle(data_derived())
    split_manifest = load_split_manifest(data_derived() / "split_manifest.json")
    manifest = build_run_manifest(
        run_id=run_id,
        mode_profile=profile,
        run_seed=run_seed,
        mapping_sha256=foundation_idx.mapping_sha256,
        split_hash=foundation_idx.split_hash,
        exclusion_sha256=foundation_idx.exclusion_sha256,
        foundation_contract_sha256=foundation_idx.foundation_contract_sha256,
        raw_data_hash=split_manifest.raw_data_hash,
        builder_version=split_manifest.builder_version,
        overrides=overrides,
        module="personalized",
    )

    # Phase 6 D-06/D-07: mutate manifest with final_eval_round_index + metrics
    # AFTER final_metrics is assigned and BEFORE embed_manifest_in_result.
    # Phase 7 D-22: thesis-tagging fields read from run_config; sentinels for non-thesis runs.
    manifest = dataclass_replace(manifest,
        final_eval_round_index=final_eval_round_index,
        metrics=results_data["final_metrics"],
        thesis_run_label=str(context.run_config.get("thesis-run-label", "")),
        ablation_dimension=str(context.run_config.get("ablation-dimension", "none")),
        ablation_value=str(context.run_config.get("ablation-value", "")),
    )

    # D-15: embed manifest INTO the result JSON (double-write part 1).
    embed_manifest_in_result(manifest, results_data)

    # =========================================================================
    # Phase 6 D-01/D-02: per-module per-run directory layout for cross-device.
    # Phase 7 D-04: thesis_crossdevice_main joins the per-run-dir gate.
    # Cross-silo legacy mode keeps the flat <run_id>_results.json layout (D-03).
    # =========================================================================
    print("\nSaving evaluation results...")
    if mode in ("benchmark_cross_device", "thesis_crossdevice_main", "paper_compat_pfedrec"):
        run_dir = module_run_results_dir(_MODULE, run_id)
        results_filename = run_dir / "results.json"  # D-04 clean filename
        atomic_write_json(str(results_filename), results_data)
        sibling_path = write_manifest_sibling(manifest, results_filename, sibling_name="manifest.json")
        # Persistence (re-eval enablement, 2026-06-12): save the restored-BEST
        # and LAST-round GLOBAL params so finished runs stay re-evaluable
        # offline (scripts/thesis/reeval.py p_u-variant tests) — previously the
        # globals died with the process and no post-hoc analysis was possible.
        # Must never kill a finished 30h run, hence the broad guard.
        try:
            torch.save(arrays.to_torch_state_dict(), run_dir / "global_state_best.pt")
            torch.save(last_arrays.to_torch_state_dict(), run_dir / "global_state_last.pt")
            print(f"  Global state saved: {run_dir}/global_state_{{best,last}}.pt")
        except Exception as _pe:  # noqa: BLE001
            print(f"[WARN] global-state persistence failed (non-fatal): {_pe}")
    else:  # cross_silo_legacy — preserved per D-03
        legacy_dir = repo_root() / "results" / "federated"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        results_filename = legacy_dir / f"{run_id}_results.json"
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

    # Finish wandb run
    if wandb_enabled and wandb_run is not None:
        wandb.finish()
        print("  Weights & Biases run completed")
