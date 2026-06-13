"""federated-baseline-cf: A Flower / PyTorch app for Matrix Factorization.

Uses Grid message-passing API for federated orchestration with support
for early stopping and sampled evaluation metrics.
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
from flwr.server.strategy import FedAvg, FedProx

from federated_baseline_cf.task import (
    get_model, test, evaluate_ranking, evaluate_ranking_sampled,
)
from federated_baseline_cf.dataset import load_full_data
from federated_baseline_cf.early_stopping import EarlyStopping

# Phase 2 Plan 04: foundation + strategy imports (BSL-04, BSL-06, BSL-08, D-25, D-26, D-27).
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
from fedrec_foundation.atomic import atomic_write_json
from fedrec_foundation.paths import data_derived, module_run_results_dir, repo_root
from fedrec_foundation.rng import server_rng
from dataclasses import replace as dataclass_replace
from fedrec_foundation.split import load_split_manifest

from federated_baseline_cf.strategy import BaselineFedAvg, BaselineFedProx

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


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # =========================================================================
    # Phase 2: mode resolver owns canonical hyperparams; pyproject values are
    # fallback only (D-25). Overrides visible per D-19, captured in manifest.overrides.
    # =========================================================================
    _MODULE: str = "baseline"   # cross-references: build_run_manifest, module_run_results_dir, default W&B project switch

    mode = str(context.run_config.get("mode", "cross_silo_legacy"))
    profile = resolve_mode_defaults(mode)
    print(
        f"\n[MODE] Resolved profile mode={profile.mode!r} "
        f"num_supernodes={profile.num_supernodes} "
        f"weight_policy={profile.weight_policy!r} "
        f"primary_evaluator={profile.primary_evaluator!r}"
    )
    overrides = log_mode_and_overrides(mode, profile, context.run_config)
    if overrides:
        # D-19 loud warning per key — already printed inside log_mode_and_overrides; add a SUMMARY line.
        print(
            f"⚠ OVERRIDE: {len(overrides)} key(s) diverge from mode default. "
            f"Run is NOT comparable to benchmark thesis table."
        )

    run_seed = int(context.run_config.get("run-seed", 42))

    # Read run config: profile is the source of truth; context.run_config overrides win (D-25).
    num_rounds: int = int(context.run_config.get("num-server-rounds", profile.num_server_rounds))
    fraction_train: float = float(context.run_config.get("fraction-train", profile.fraction_train))
    lr: float = float(context.run_config.get("lr", profile.lr))
    model_type: str = context.run_config.get("model-type", "bpr")
    embedding_dim: int = int(context.run_config.get("embedding-dim", profile.embedding_dim))
    dropout: float = float(context.run_config.get("dropout", 0.1))

    # FedProx configuration
    strategy_name: str = str(context.run_config.get("strategy", "fedavg")).lower()
    proximal_mu: float = float(context.run_config.get("proximal-mu", 0.0))

    # D-25 additional Phase-2 contract keys (fallback to profile where profile has them).
    weight_policy: str = str(context.run_config.get("weight-policy", profile.weight_policy))
    checkpoint_rule: str = str(context.run_config.get("checkpoint-rule", profile.checkpoint_rule))

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

    # run_id materialized EARLY so W&B config carries it (joinability with the
    # results/manifest artifacts; run-id audit follow-up). The BSL-08 manifest
    # block below reuses this same value.
    run_id = generate_run_id()

    # Initialize Weights & Biases if enabled
    wandb_enabled = context.run_config.get("wandb-enabled", False)
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
        }
        # D-25: expose mode + run_seed + contract keys so W&B dashboards can filter by mode.
        wandb_config.update({
            "mode": mode,
            "run_seed": run_seed,
            "weight_policy": weight_policy,
            "partition_mode": profile.partition_mode,
            "checkpoint_rule": checkpoint_rule,
        })
        # Cross-device runs go to a dedicated W&B project per PROJECT.md constraint; legacy stays on "federated-cf".
        # Phase 7 D-04: thesis_crossdevice_main joins the cross-device W&B project gate.
        default_project = (
            "federated-cf-cross-device"
            if mode in ("benchmark_cross_device", "thesis_crossdevice_main", "paper_compat_pfedrec")
            else "federated-cf"
        )
        # Empty-string in pyproject means "use mode-based default"; explicit non-empty wins.
        wandb_project_cfg = str(context.run_config.get("wandb-project", "")).strip()
        wandb_project = wandb_project_cfg if wandb_project_cfg else default_project
        wandb_entity = context.run_config.get("wandb-entity", "")
        wandb_run_name = context.run_config.get("wandb-run-name", "")
        wandb.init(
            project=wandb_project,
            entity=wandb_entity if wandb_entity else None,
            name=wandb_run_name if wandb_run_name else None,
            config=wandb_config,
        )
        print("  Weights & Biases: Enabled")

    # Load global Matrix Factorization model
    print(f"\nInitializing {model_type.upper()} Matrix Factorization model...")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Dropout: {dropout}")

    global_model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )

    num_params = sum(p.numel() for p in global_model.parameters())
    print(f"  Total parameters: {num_params:,}")

    arrays = ArrayRecord(global_model.state_dict())

    # =========================================================================
    # G-03-01: discovery round. Build partition_id -> node_id mapping BEFORE the
    # main loop so per-round sampling can work in stable partition-id space.
    # Flower's node_ids are os.urandom-seeded per boot; partition_ids are the
    # stable 0..N-1 identity we actually want to record for thesis reproducibility.
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

    # BSL-06: BaselineFedAvg / BaselineFedProx overrides aggregate_evaluate to
    # emit headline metrics from SUMMED sufficient stats (not averaged per-client ratios).
    if strategy_name == "fedprox":
        strategy = BaselineFedProx(
            fraction_fit=fraction_train,
            proximal_mu=proximal_mu,
        )
        print(f"  Strategy: BaselineFedProx (proximal_mu={proximal_mu})")
    else:
        strategy = BaselineFedAvg(
            fraction_fit=fraction_train,
        )
        print(f"  Strategy: BaselineFedAvg")

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

    # BSL-04: seeded RNG for per-round client selection — deterministic across processes.
    # Single instance at loop-start => sequence across rounds is stable for a given run_seed.
    _server_sampler = server_rng(run_seed)
    selected_clients_per_round: List[List[int]] = []  # D-26: persisted in result JSON + W&B

    # D-27: best-round tracking (in-memory; no disk writes). At training end,
    # restore the best-round state_dict before running the final centralized evaluation.
    best_metric: float = float("-inf")
    best_round_num: int = 0
    best_arrays = arrays  # fallback if no eval round improves

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*50}")
        print(f"Round {round_num}/{num_rounds}")
        print(f"{'='*50}")

        train_config = ConfigRecord({"lr": lr, "proximal_mu": proximal_mu})

        # G-03-01: sample in partition-id space (stable 0..N-1), translate to
        # node_ids for message addressing. Deterministic across runs for a
        # given run_seed because the sampling DOMAIN is now seed-independent.
        # Plan-04's `sorted(grid.get_node_ids())` domain was non-deterministic
        # (Flower's node_ids are os.urandom-seeded per boot), so the sampler
        # ran deterministically over a randomised domain — see 02-UAT.md G-03-01.
        num_selected = max(1, int(expected_n * fraction_train))
        selected_pids: List[int] = _server_sampler.sample(range(expected_n), num_selected)
        selected_node_ids = [partition_to_node_id[pid] for pid in selected_pids]

        # D-26: persist + log PARTITION IDs (user-identifying, stable) — not node_ids.
        selected_clients_per_round.append([int(pid) for pid in selected_pids])
        if wandb_enabled:
            wandb.log({"round/selected_clients": [int(pid) for pid in selected_pids]}, step=round_num)

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

        # Parse training responses and aggregate
        fit_results = []
        round_train_metrics = []

        for response in train_responses:
            if response.has_error():
                print(f"  Warning: Client {response.metadata.src_node_id} returned error")
                continue

            resp_arrays = response.content.get("arrays", ArrayRecord())
            resp_metrics = response.content.get("metrics", MetricRecord())

            metrics_dict = dict(resp_metrics) if resp_metrics else {}
            num_examples = int(metrics_dict.get("num-examples", 1))

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

        # Aggregate training results using strategy
        if fit_results:
            aggregated_params, agg_metrics = strategy.aggregate_fit(
                server_round=round_num,
                results=fit_results,
                failures=[],
            )

            # Update global parameters for next round
            if aggregated_params is not None:
                param_ndarrays = parameters_to_ndarrays(aggregated_params)
                param_keys = list(arrays.to_torch_state_dict().keys())
                new_state_dict = {k: torch.from_numpy(v) for k, v in zip(param_keys, param_ndarrays)}
                arrays = ArrayRecord(new_state_dict)

            # Aggregate training metrics
            train_metrics_history[round_num] = weighted_average_metrics(round_train_metrics)

            train_loss = train_metrics_history[round_num].get('train_loss', 'N/A')
            if isinstance(train_loss, (int, float)):
                print(f"  Training loss: {train_loss:.4f}")

        # =====================================================================
        # EVALUATION PHASE
        # =====================================================================
        eval_messages = []
        for node_id in selected_node_ids:
            eval_config = ConfigRecord({"lr": lr})
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
        # EVALUATION AGGREGATION (BSL-06 via BaselineFedAvg.aggregate_evaluate)
        # Wrap each Flower eval response into an EvaluateRes and let the
        # strategy emit server-side ratios from SUMMED sufficient stats.
        # =====================================================================
        eval_results: List[Tuple[ClientProxy, EvaluateRes]] = []
        round_eval_metrics = []  # retained for RMSE/MAE rating-path fallback (D-18)
        for response in eval_responses:
            if response.has_error():
                continue

            resp_metrics = response.content.get("metrics", MetricRecord())
            metrics_dict = dict(resp_metrics) if resp_metrics else {}
            num_examples = int(metrics_dict.get("num_training_examples",
                                                metrics_dict.get("evaluated_users",
                                                                 metrics_dict.get("num-examples", 1))))
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

        # BSL-06: strategy computes sum-based headline ratios + per-group ratios.
        if eval_results:
            _agg_loss, thesis_metrics = strategy.aggregate_evaluate(round_num, eval_results, [])
            eval_metrics_history[round_num] = dict(thesis_metrics) if thesis_metrics else {}

            # Preserve rating metrics (RMSE/MAE) via the legacy per-client-ratio path — D-18 scope-out.
            # These keys aren't sufficient-stat aggregated; strategy.aggregate_evaluate returns only
            # the thesis-table metrics. Merge RMSE/MAE in without clobbering the sufficient-stat ratios.
            rating_agg = weighted_average_metrics(round_eval_metrics)
            for rk in ("rmse", "mae", "eval_loss"):
                if rk in rating_agg and rk not in eval_metrics_history[round_num]:
                    eval_metrics_history[round_num][rk] = rating_agg[rk]

            # Print key metrics
            rmse = eval_metrics_history[round_num].get('rmse', 'N/A')
            ndcg10 = eval_metrics_history[round_num].get('sampled_ndcg@10', 'N/A')
            hr10 = eval_metrics_history[round_num].get('sampled_hr@10', 'N/A')
            rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
            ndcg10_str = f"{ndcg10:.4f}" if isinstance(ndcg10, (int, float)) else str(ndcg10)
            hr10_str = f"{hr10:.4f}" if isinstance(hr10, (int, float)) else str(hr10)
            print(f"  RMSE: {rmse_str}")
            print(f"  Sampled HR@10: {hr10_str}")
            print(f"  Sampled NDCG@10: {ndcg10_str}")

            # D-27: track best-round global params (in-memory). No disk writes.
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
        if wandb_enabled:
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
                if wandb_enabled:
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
    print("\n" + "="*70)
    print("FEDERATED TRAINING COMPLETE")
    print("="*70)
    print(f"Total rounds completed: {actual_rounds}/{num_rounds}")
    if early_stopper and early_stopper.state.should_stop:
        print(f"Early stopping: Triggered at round {actual_rounds}")
        print(f"Best {early_stopping_metric}: {early_stopper.best_metric:.4f} at round {early_stopper.best_round}")
    print("="*70)

    # =========================================================================
    # D-27: Restore best-round global params before running the final centralized eval.
    # Canonical reported metric is best_* per STATE.md; last-round is not comparable.
    # =========================================================================
    # Persistence (re-eval enablement, 2026-06-12): keep the LAST-round globals
    # before the restore overwrites `arrays`; both vintages saved at results write.
    last_arrays = arrays
    if checkpoint_rule in ("best_round_restore", "best_round") and best_round_num > 0:
        print(
            f"\n[CHECKPOINT] Restoring global params from best round {best_round_num} "
            f"(sampled_ndcg@10={best_metric:.4f}) before centralized evaluation"
        )
        arrays = best_arrays
    else:
        print(f"\n[CHECKPOINT] checkpoint_rule={checkpoint_rule!r}: keeping last-round params")

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

    # =========================================================================
    # CENTRALIZED EVALUATION: Run evaluation on server with final model
    # =========================================================================
    print("\n📊 Running centralized evaluation with final model...")

    # Load final model weights
    final_model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )
    final_state_dict = arrays.to_torch_state_dict()
    final_model.load_state_dict(final_state_dict)

    # Auto-detect device
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            device = torch.device("cuda:0")
            print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e:
            print(f"  CUDA available but not compatible: {e}")
            print(f"  Falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print(f"  Using CPU")
    final_model.to(device)

    # Load full test data for evaluation
    split_mode = context.run_config.get("eval-split-mode", "leave-one-out")
    trainloader, testloader, _, _, _, _ = load_full_data(
        test_ratio=0.2,
        batch_size=256,
        split_mode=split_mode,
    )

    # Compute rating prediction metrics (RMSE, MAE)
    print("  Computing rating prediction metrics...")
    eval_loss, rating_metrics = test(
        model=final_model,
        testloader=testloader,
        device=str(device),
        model_type=model_type,
    )

    # Compute ranking metrics
    print("  Computing ranking metrics...")
    k_values_str = context.run_config.get("ranking-k-values", "5,10,20")
    k_values = [int(k.strip()) for k in k_values_str.split(",")]

    ranking_metrics = evaluate_ranking(
        model=final_model,
        testloader=testloader,
        device=str(device),
        k_values=k_values,
        trainloader=trainloader,
    )

    # Compute sampled ranking metrics
    print("  Computing sampled ranking metrics...")
    num_negatives = context.run_config.get("eval-num-negatives", 99)
    sampled_metrics = evaluate_ranking_sampled(
        model=final_model,
        testloader=testloader,
        trainloader=trainloader,
        device=str(device),
        k_values=k_values,
        num_negatives=num_negatives,
    )

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

    # Print evaluation results
    print_evaluation_metrics(actual_rounds, final_metrics["best"], context)

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

    # Create results JSON
    results_data = {
        "model_name": f"{model_type.upper()}_MF_Federated_{strategy_name.upper()}",
        "dataset": "ml-1m",
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
            "mode": mode,
            "run_seed": run_seed,
            "weight_policy": weight_policy,
            "checkpoint_rule": checkpoint_rule,
        },
        "early_stopping": early_stopper.get_summary() if early_stopper else None,
        "timestamp": datetime.now().isoformat(),
        "final_metrics": final_metrics,
        "training_rounds": actual_rounds,
        "eval_metrics_history": eval_metrics_history,
        "train_metrics_history": train_metrics_history,
    }

    # =========================================================================
    # BSL-08: protocol fingerprint manifest (FND-07 + D-15 double-write).
    # run_id was materialized early (before W&B init) — reuse it here.
    # =========================================================================
    # Verify the bundle ONCE; raises if tampered. Reads fingerprints from foundation_index.json.
    foundation_idx = verify_bundle(data_derived())
    # raw_data_hash + builder_version live on the SplitManifest (single source of truth per IMP-2).
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
        module="baseline",
    )

    # D-26: selected_clients_per_round is a first-class field in the JSON.
    results_data["selected_clients_per_round"] = selected_clients_per_round
    results_data["checkpoint"] = {
        "rule": checkpoint_rule,
        "best_round": best_round_num,
        "best_sampled_ndcg@10": best_metric if best_metric != float("-inf") else None,
    }

    # Phase 6: post-build mutation to populate schema-v2 fields (final_eval_round_index, metrics).
    # Phase 7 D-22: thesis-tagging fields read from run_config; sentinels for non-thesis runs.
    # IMPORTANT: must happen BEFORE embed_manifest_in_result so the _manifest key in
    # results_data carries the Phase-6 + Phase-7 fields.
    # Acceptance: idx_final must appear before idx_replace in source order (plan-checker iter 1).
    manifest = dataclass_replace(
        manifest,
        final_eval_round_index=final_eval_round_index,
        metrics=results_data["final_metrics"],
        thesis_run_label=str(context.run_config.get("thesis-run-label", "")),
        ablation_dimension=str(context.run_config.get("ablation-dimension", "none")),
        ablation_value=str(context.run_config.get("ablation-value", "")),
    )

    # D-15: double-write (embedded in result JSON + sibling file).
    embed_manifest_in_result(manifest, results_data)  # mutates in place

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
        # Persistence (re-eval enablement, 2026-06-12): save restored-BEST +
        # LAST-round GLOBAL params so finished runs stay re-evaluable offline.
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
        sibling_path = write_manifest_sibling(manifest, results_filename)  # default <run_id>-manifest.json
    print(f"Results saved to: {results_filename.resolve()}")
    print(f"Manifest sibling: {sibling_path.resolve()}")

    # W&B: attach manifest fingerprints to the run's config so dashboards can filter/audit.
    if wandb_enabled:
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

    # Finish wandb run
    if wandb_enabled:
        wandb.finish()
        print("  Weights & Biases run completed")
