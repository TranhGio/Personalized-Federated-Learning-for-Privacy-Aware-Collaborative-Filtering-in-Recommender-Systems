"""federated-pfedrec: Cross-device PFedRec Flower Server (Phase 5 Plan 04).

5 PFedRec-specific deltas over the Phase 4 Plan 5 server template:

1. **D-12 strategy**: ``PFedRecSplitFedAvg`` (Plan 01) replaces ``SplitFedAvg``;
   no FedProx variant per **D-07** — the IJCAI-23 reference uses FedAvg only
   and PFedRec's per-user score function does not benefit from a global
   proximal term.
2. **D-01 GLOBAL bias propagation**: initial ``ArrayRecord`` carries BOTH
   ``embedding_item.weight`` AND ``affine_output.bias`` because
   ``PFedRecMLP._GLOBAL_PARAMS = ('embedding_item.weight', 'affine_output.bias')``
   per ``IJCAI-23-PFedRec/engine.py:143`` (only ``affine_output.weight`` is
   deleted before aggregation; bias is averaged server-side).
3. **No prototype / alpha bookkeeping**: Phase 4's ``best_prototype`` snapshot/
   restore + 6-scalar ``alpha_diagnostics`` aggregation are absent. PFedRec has
   neither.
4. **No centralized eval block**: split learning — the server holds no LOCAL
   params (per-user ``affine_output.weight``). Final headline metrics come
   from federated eval (``eval_metrics_history[best_round_num]``).
5. **D-14 PFR-08 auto-verify hook**: at run end, parses
   ``IJCAI-23-PFedRec/sh_result/ml-1m.txt`` line 2 (HR=0.7286, NDCG=0.4407 —
   Open Question 1: most recent / closest to paper-reported best round 89),
   asserts ``|our - reference| <= 2.0`` absolute points (multiplying ratios by
   100 for log readability), prints ``[PFR-08 VERIFIED]`` / ``[PFR-08 FAILED]``,
   embeds the audit dict in ``results_data['_manifest']['pfr08_verification']``,
   logs ``wandb.run.summary['final/pfr08']``. **Non-fatal** — failed
   reproduction does NOT abort the run.

Phase 3/4 carry-forward (verbatim clone):

- **D-25 mode resolver header**: ``resolve_mode_defaults(mode)`` at @app.main
  entry; every hyperparameter read is ``int/float/str(context.run_config.get(
  key, profile.field))`` so the mode profile is the canonical source.
- **D-02 mirror frozen-cross-silo guard**: ``mode == "cross_silo_legacy"``
  raises ``NotImplementedError`` BEFORE any heavy work (data load / model
  construction).
- **G-03-01 discovery round**: a one-shot ``evaluate(discover_only=true)``
  broadcast to ALL nodes BEFORE the main loop builds
  ``partition_to_node_id: Dict[int, int]`` so per-round sampling runs in
  stable partition-id space (0..N-1) instead of Flower's ephemeral node_ids.
- **ADP-06 partition-id-space sampling**: a single ``server_rng(run_seed)``-
  backed ``random.Random`` instance is held pre-loop and reused every round
  via ``.sample(range(expected_n), k)``. ``selected_clients_per_round``
  accumulates partition_ids 0..N-1.
- **D-13 cold-start counter** (per-round logging counter — distinct from the
  best-round-restore monitor metric): probes
  ``.embedding_cache/{run_id}/partition_{pid}.pt`` existence per selected pid;
  ``results_data["cold_starts"]`` carries ``per_round / total / rate`` fields;
  under ``reuse_cache=true`` short-circuits to 0 with a documented log line.
- **D-13 best-round-restore via the Phase-3-D-27 carry-forward in-memory
  snapshot pattern** (NOT to be confused with CONTEXT.md D-27 weight-policy
  override): when ``current_ndcg > best_metric``, deep-clone the live
  ``ArrayRecord`` into ``best_arrays``; restore at loop end before the final
  result write. Spelling tolerance — ``checkpoint_rule`` accepts both
  ``best_round_restore`` and ``best_round``. CONTEXT.md D-13: monitor metric
  is ``sampled_ndcg@10``.
- **D-15 manifest double-write**: ``build_run_manifest(..., module="pfedrec")``
  + ``embed_manifest_in_result(...)`` + ``write_manifest_sibling(...)``. Plus
  ``results_data["_manifest"]["audit_doc"] = "PFR-02-AUDIT.md"`` post-embed
  mutation — the SC-1 back-pointer to the cross-walk authored by Plan 01
  Task 3. Build_run_manifest does NOT accept ``audit_doc`` directly so the
  field is appended to the embedded ``_manifest`` dict (Phase-3/Phase-4 idiom
  for post-build payload extensions).
- **Pitfall 5 Option B (D-24 uniform weight on FIT side)**: ``FitRes.num_examples = 1``
  per client. FedAvg's existing num_examples-weighted aggregator is then
  mathematically uniform — mirrors ``engine.py:81 len(round_user_params)``
  division.
- **W&B project**: ``federated-cf-cross-device`` for ``mode in
  {"benchmark_cross_device", "paper_compat_pfedrec"}`` (D-10).

D-18 surgical discipline: ``DummyClientProxy``, ``weighted_average_metrics``,
``print_evaluation_metrics``, CUDA fallback are preserved verbatim from the
pre-Phase-5 code shape. Stdlib ``random`` is eradicated module-wide.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import wandb
from flwr.common import (
    Code,
    EvaluateRes,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.context import Context
from flwr.common.record import ArrayRecord, ConfigRecord, MetricRecord, RecordDict
from flwr.server.client_proxy import ClientProxy
from flwr.serverapp import Grid, ServerApp

# Phase 5 Plan 04 foundation imports.
from fedrec_foundation.atomic import atomic_write_json
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
from fedrec_foundation.rng import server_rng
from fedrec_foundation.split import load_split_manifest
from dataclasses import replace as dataclass_replace

from federated_pfedrec.early_stopping import EarlyStopping
from federated_pfedrec.strategy import PFedRecSplitFedAvg
from federated_pfedrec.task import get_model


# Create ServerApp.
app = ServerApp()


# =============================================================================
# DummyClientProxy (D-18 preserved verbatim from pre-Phase-5 shape).
# =============================================================================


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


def weighted_average_metrics(
    metrics: List[Tuple[int, Dict[str, float]]],
) -> Dict[str, float]:
    """Aggregate evaluation metrics using weighted average (D-18 preserved).

    Used only as a legacy diagnostic fallback path for keys not produced by
    the strict sufficient-stat aggregator (e.g. ``eval_loss`` across clients).
    The thesis headline metrics flow through ``strategy.aggregate_evaluate``.

    Parameters
    ----------
    metrics : List[Tuple[int, Dict[str, float]]]
        ``(num_examples, metrics_dict)`` per client.

    Returns
    -------
    Dict[str, float]
        Numeric-only weighted-average mapping; keys with non-numeric values
        are skipped (a list / dict cannot be averaged).
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {}

    aggregated: Dict[str, float] = {}
    if metrics:
        metric_keys = metrics[0][1].keys()
        for key in metric_keys:
            if key == "num-examples":
                continue
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


def print_evaluation_metrics(
    round_num: int, metrics: Dict[str, float], context: Context
) -> None:
    """Pretty print evaluation metrics for a federated round (D-18 preserved)."""
    print(f"\n{'='*70}")
    print(f"PFedRec Evaluation Results - Round {round_num}")
    print(f"{'='*70}")

    if "eval_loss" in metrics:
        print(f"\n  BCE Loss:  {metrics['eval_loss']:.4f}")
    if "rmse" in metrics:
        print(f"  RMSE:      {metrics['rmse']:.4f}")
    if "mae" in metrics:
        print(f"  MAE:       {metrics['mae']:.4f}")

    k_values_str = context.run_config.get("ranking-k-values", "5,10,20")
    k_values = [int(k.strip()) for k in k_values_str.split(",")]

    has_sampled = any(f"sampled_hr@{k}" in metrics for k in k_values)
    if has_sampled:
        num_neg = int(metrics.get("sampled_num_negatives", 99))
        print(f"\n  Sampled Ranking (leave-one-out + {num_neg} negatives):")
        if "sampled_mrr" in metrics:
            print(f"  MRR:       {metrics['sampled_mrr']:.4f}")
        for k in sorted(k_values):
            print(f"\n  @ K={k}:")
            if f"sampled_hr@{k}" in metrics:
                print(f"    Hit Rate:   {metrics[f'sampled_hr@{k}']:.4f}")
            if f"sampled_ndcg@{k}" in metrics:
                print(f"    NDCG:       {metrics[f'sampled_ndcg@{k}']:.4f}")

    print(f"\n{'='*70}\n")


# =============================================================================
# D-14 PFR-08 auto-verify helpers (Phase 5 NEW).
#
# Module-level so tests can import them directly. Kept compact + side-effect-
# free — the only I/O is reading the reference text file.
# =============================================================================


def _parse_reference_results(reference_path: Path) -> Tuple[float, float]:
    """Parse ``IJCAI-23-PFedRec/sh_result/ml-1m.txt`` for HR@10 / NDCG@10.

    Picks the LAST non-empty line (line 2 today: HR=0.7286, NDCG=0.4407).
    Open Question 1 recommendation: "most recent / closest to paper-reported
    best round 89" — the second line of the file.

    The reference format is dash-delimited tokens::

        2026-04-03 19-47-11-latent_dim: 32-lr: 0.1-...-hr: 0.7286-ndcg: 0.4407-...

    Parameters
    ----------
    reference_path : Path
        Path to the reference results file.

    Returns
    -------
    Tuple[float, float]
        ``(hr@10, ndcg@10)`` parsed from the last line.

    Raises
    ------
    RuntimeError
        If the file is missing, empty, or no ``hr:`` / ``ndcg:`` token is
        found in the chosen line.
    """
    if not reference_path.exists():
        raise RuntimeError(
            f"PFR-08 reference file not found: {reference_path}"
        )
    with open(reference_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise RuntimeError(f"PFR-08 reference file is empty: {reference_path}")
    target = lines[-1]
    tokens = target.split("-")
    hr: Optional[float] = None
    ndcg: Optional[float] = None
    for token in tokens:
        token_stripped = token.lstrip()
        if token_stripped.startswith("hr:"):
            hr = float(token.split(":")[1].strip())
        elif token_stripped.startswith("ndcg:"):
            ndcg = float(token.split(":")[1].strip())
    if hr is None or ndcg is None:
        raise RuntimeError(f"PFR-08 reference parse failed: {target!r}")
    return hr, ndcg


def _emit_pfr_08_verification(
    final_metrics: Dict[str, float],
    reference_path: Path,
    tolerance_pts: float = 2.0,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Run the D-14 PFR-08 reproduction check.

    Compares ``final_metrics["sampled_hr@10"]`` and
    ``final_metrics["sampled_ndcg@10"]`` against the parsed reference
    (multiplying ratios by 100 for "absolute points" log readability).
    Asserts ``|delta_hr| <= tolerance`` AND ``|delta_ndcg| <= tolerance``.

    **Non-fatal**: the function NEVER raises. Failed reproduction surfaces as
    ``passed=False`` so the orchestrator can continue (and downstream tooling
    can decide policy). Auth/parse errors are also surfaced as ``passed=False``
    rather than raising so the run completes cleanly.

    Parameters
    ----------
    final_metrics : Dict[str, float]
        Headline metrics from ``eval_metrics_history[best_round_num]``.
    reference_path : Path
        Path to ``IJCAI-23-PFedRec/sh_result/ml-1m.txt``.
    tolerance_pts : float
        Absolute-points tolerance (default 2.0 — matches PFR-08 ±2 contract).

    Returns
    -------
    Tuple[bool, str, Dict[str, Any]]
        ``(passed, log_line, audit_dict)``. ``audit_dict`` is JSON-serializable
        and goes into ``results_data["_manifest"]["pfr08_verification"]``.
    """
    try:
        ref_hr, ref_ndcg = _parse_reference_results(reference_path)
    except RuntimeError as e:
        return (
            False,
            f"[PFR-08 FAILED: {e}]",
            {"passed": False, "error": str(e)},
        )

    our_hr = final_metrics.get("sampled_hr@10", float("nan"))
    our_ndcg = final_metrics.get("sampled_ndcg@10", float("nan"))
    # NaN check (NaN != NaN by IEEE 754).
    if any(v != v for v in (our_hr, our_ndcg)):
        return (
            False,
            f"[PFR-08 FAILED: missing metric our_hr={our_hr} our_ndcg={our_ndcg}]",
            {"passed": False, "error": "missing metric"},
        )

    delta_hr_pts = abs(our_hr - ref_hr) * 100.0
    delta_ndcg_pts = abs(our_ndcg - ref_ndcg) * 100.0
    passed = delta_hr_pts <= tolerance_pts and delta_ndcg_pts <= tolerance_pts
    tag = "VERIFIED" if passed else "FAILED"
    log_line = (
        f"[PFR-08 {tag}] our_hr@10={our_hr:.4f} ref_hr@10={ref_hr:.4f} "
        f"delta_hr={delta_hr_pts:.2f}pts | "
        f"our_ndcg@10={our_ndcg:.4f} ref_ndcg@10={ref_ndcg:.4f} "
        f"delta_ndcg={delta_ndcg_pts:.2f}pts | tolerance={tolerance_pts:.1f}pts"
    )
    audit: Dict[str, Any] = {
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


# =============================================================================
# Main FL loop.
# =============================================================================


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Cross-device PFedRec server entry point (Phase 5 Plan 04).

    See module docstring for the full delta list. Implements:

    1. D-25 mode resolver + D-02 mirror frozen-cross-silo guard.
    2. ``run_id`` materialized EARLY for D-13 cold-start probe alignment.
    3. W&B init under ``federated-cf-cross-device`` (D-10) for paper-compat.
    4. Initial ``ArrayRecord`` carries BOTH GLOBAL keys (D-01).
    5. ``PFedRecSplitFedAvg`` strategy (D-12, no FedProx variant per D-07).
    6. G-03-01 discovery round + ADP-06 partition-id-space sampler.
    7. FL loop with D-13 cold-start counter + D-13 best-round-restore via
       Phase-3-D-27 carry-forward + Pitfall 5 Option B uniform FIT weighting.
    8. Final result JSON + D-15 manifest double-write + D-14 PFR-08 hook +
       SC-1 ``audit_doc`` back-pointer to ``PFR-02-AUDIT.md``.
    """

    # =========================================================================
    # 1. D-25 mode resolver + D-02 mirror frozen-cross-silo guard.
    # =========================================================================
    mode = str(context.run_config.get("mode", "paper_compat_pfedrec"))
    run_seed = int(context.run_config.get("run-seed", 42))
    profile = resolve_mode_defaults(mode)
    print(
        f"\n[MODE] Resolved profile mode={profile.mode!r} "
        f"num_supernodes={profile.num_supernodes} "
        f"weight_policy={profile.weight_policy!r} "
        f"primary_evaluator={profile.primary_evaluator!r}"
    )
    overrides = log_mode_and_overrides(mode, profile, dict(context.run_config))
    if overrides:
        print(
            f"  OVERRIDE: {len(overrides)} key(s) diverge from mode default. "
            f"Run is NOT comparable to PFR-08 reproduction."
        )

    _MODULE: str = "pfedrec"   # cross-references: build_run_manifest, module_run_results_dir

    # D-02 mirror: cross-silo PFedRec is FROZEN per Phase 5 D-09.
    if mode == "cross_silo_legacy":
        raise NotImplementedError(
            "PFedRec cross-silo path is FROZEN per Phase 5 D-09 / D-02. "
            "See .planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md "
            "§Deferred. Pre-Phase-5 commits are the authoritative cross-silo "
            "PFedRec artifact."
        )

    # =========================================================================
    # D-25 hyperparameters — profile is canonical; run_config overrides win.
    # =========================================================================
    num_rounds: int = int(
        context.run_config.get("num-server-rounds", profile.num_server_rounds)
    )
    fraction_train: float = float(
        context.run_config.get("fraction-train", profile.fraction_train)
    )
    lr: float = float(context.run_config.get("lr", profile.lr))
    lr_eta: float = float(context.run_config.get("lr-eta", 80))
    latent_dim: int = int(
        context.run_config.get("latent-dim", profile.embedding_dim)
    )
    local_epochs: int = int(
        context.run_config.get("local-epochs", profile.local_epochs)
    )
    batch_size: int = int(context.run_config.get("batch-size", 256))
    num_train_negatives: int = int(
        context.run_config.get("num-negatives", profile.num_train_negatives)
    )
    weight_policy: str = str(
        context.run_config.get("weight-policy", profile.weight_policy)
    )
    checkpoint_rule: str = str(
        context.run_config.get(
            "checkpoint-rule",
            getattr(profile, "checkpoint_rule", "best_round_restore"),
        )
    )
    reuse_cache_flag: bool = bool(context.run_config.get("reuse-cache", False))

    # 2. run_id materialized EARLY (Phase 3 idiom) — server cold-start probe
    #    and client cache path coincide on .embedding_cache/{run_id}/.
    run_id = str(context.run_config.get("run-id", "")) or generate_run_id()

    # =========================================================================
    # Early stopping (D-18 preserved diagnostic; thesis tracks via best-round-
    # restore against sampled_ndcg@10 — early stopping is opt-in).
    # =========================================================================
    early_stopping_enabled = context.run_config.get("early-stopping-enabled", False)
    early_stopping_patience = context.run_config.get("early-stopping-patience", 10)
    early_stopping_metric = context.run_config.get(
        "early-stopping-metric", "sampled_ndcg@10"
    )
    early_stopping_mode = context.run_config.get("early-stopping-mode", "max")
    early_stopping_min_delta = context.run_config.get(
        "early-stopping-min-delta", 0.001
    )

    early_stopper: Optional[EarlyStopping] = None
    if early_stopping_enabled:
        early_stopper = EarlyStopping(
            patience=early_stopping_patience,
            metric_name=early_stopping_metric,
            mode=early_stopping_mode,
            min_delta=early_stopping_min_delta,
            verbose=True,
        )
        print(
            f"  Early stopping: Enabled (patience={early_stopping_patience}, "
            f"metric={early_stopping_metric})"
        )

    # =========================================================================
    # 3. W&B init under federated-cf-cross-device (D-10) for paper_compat.
    # =========================================================================
    wandb_enabled = bool(context.run_config.get("wandb-enabled", False))
    wandb_run = None
    if wandb_enabled:
        wandb_config = {
            "algorithm": "PFedRec",
            "num_rounds": num_rounds,
            "fraction_train": fraction_train,
            "local_epochs": local_epochs,
            "strategy": "fedavg",  # D-07 — no FedProx variant.
            "latent_dim": latent_dim,
            "lr": lr,
            "lr_eta": lr_eta,
            "num_negatives": num_train_negatives,
            "l2_regularization": context.run_config.get("l2-regularization", 0.0),
            "optimizer": "sgd",
            "loss": "bce",
            "early_stopping_enabled": early_stopping_enabled,
            # D-25 contract keys.
            "mode": mode,
            "run_seed": run_seed,
            "weight_policy": weight_policy,
            "partition_mode": str(
                context.run_config.get("partition-mode", profile.partition_mode)
            ),
            "checkpoint_rule": checkpoint_rule,
            "reuse_cache": reuse_cache_flag,
            "run_id": run_id,
        }
        # Phase 7 D-04: thesis_crossdevice_main joins the cross-device W&B project gate.
        default_project = (
            "federated-cf-cross-device"
            if mode in ("benchmark_cross_device", "thesis_crossdevice_main", "paper_compat_pfedrec")
            else "federated-pfedrec"
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

    # =========================================================================
    # 4. Initial ArrayRecord — BOTH GLOBAL keys per D-01.
    # =========================================================================
    print("\nInitializing PFedRec MLP model...")
    print(f"  Latent dim: {latent_dim}")
    print(f"  LR: {lr}, LR_eta: {lr_eta}")
    print(f"  Architecture: Embedding({latent_dim}) -> Linear({latent_dim}, 1) -> Sigmoid")
    print(f"  GLOBAL: embedding_item.weight + affine_output.bias (D-01)")
    print(f"  LOCAL: affine_output.weight (per-user)")

    global_model = get_model(latent_dim=latent_dim)

    # PFedRecMLP._GLOBAL_PARAMS = ('embedding_item.weight', 'affine_output.bias').
    # The initial ArrayRecord MUST carry both keys (D-01).
    arrays = ArrayRecord(global_model.get_global_parameters())

    # =========================================================================
    # 5. PFedRecSplitFedAvg (D-12). NO FedProx variant per D-07.
    # =========================================================================
    strategy = PFedRecSplitFedAvg(fraction_fit=fraction_train)
    print("  Strategy: PFedRecSplitFedAvg (D-12); no FedProx variant per D-07")

    # =========================================================================
    # 6. G-03-01 discovery round. Build partition_id -> node_id mapping BEFORE
    #    the main loop so per-round sampling runs in stable partition-id space
    #    (0..N-1) instead of Flower's os.urandom-seeded ephemeral node_id space.
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
    print(
        f"[G-03-01] Discovery complete: {len(partition_to_node_id)} "
        f"partition -> node_id entries."
    )

    # ADP-06: seeded RNG for per-round client selection — ONE instance for the
    # whole run, so the sequence across rounds is stable for a given run_seed.
    _server_sampler = server_rng(run_seed)
    selected_clients_per_round: List[List[int]] = []  # persisted in result JSON.

    # =========================================================================
    # FL loop — D-13 cold-start counter + D-13 best-round-restore via the
    # Phase-3-D-27 carry-forward in-memory snapshot pattern.
    # =========================================================================
    print(f"\nStarting PFedRec Federated Learning with {num_rounds} rounds...")
    print(f"  Clients per round: {fraction_train * 100:.0f}% of {expected_n}")

    train_metrics_history: Dict[int, Dict] = {}
    eval_metrics_history: Dict[int, Dict[str, Any]] = {}

    # D-13 best-round-restore tracking (Phase-3-D-27 carry-forward in-memory
    # snapshot — NOT to be confused with CONTEXT.md D-27 weight-policy override).
    # Monitor metric is sampled_ndcg@10 per CONTEXT.md D-13.
    best_metric: float = float("-inf")
    best_round_num: int = 0
    best_arrays = arrays  # fallback if no eval round improves.

    # D-13 cold-start counter (the per-round logging counter — distinct from
    # the best-round-restore checkpoint).
    total_cold_starts: int = 0
    cold_starts_per_round: List[int] = []
    cache_root = Path(".embedding_cache") / run_id

    # Track the last executed round so post-loop bookkeeping reports the
    # correct final round.
    round_num = 0

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*50}")
        print(f"Round {round_num}/{num_rounds}")
        print(f"{'='*50}")

        # =====================================================================
        # ADP-06: sample in partition-id space (stable 0..N-1), translate to
        # node_ids for message addressing.
        # =====================================================================
        num_selected = max(1, int(expected_n * fraction_train))
        selected_pids: List[int] = _server_sampler.sample(range(expected_n), num_selected)
        selected_node_ids = [partition_to_node_id[pid] for pid in selected_pids]
        selected_clients_per_round.append([int(pid) for pid in selected_pids])

        # =====================================================================
        # D-13 cold-start counter. Probe the cache BEFORE the train message is
        # sent: counts partitions for whom no local-state .pt file exists yet.
        # Under reuse_cache=true (D-09) the server cannot cheaply resolve the
        # sig_<hash> path — short-circuit to 0 and document via log line.
        # =====================================================================
        if reuse_cache_flag:
            cold_count = 0
            print(
                f"  [D-13] reuse-cache=true (D-09) — server-side cold-start counter "
                f"short-circuited to 0; client logs will show hit/miss per partition"
            )
        else:
            cold_count = sum(
                1 for pid in selected_pids
                if not (cache_root / f"partition_{int(pid)}.pt").exists()
            )
            print(f"  [D-13] cold_starts={cold_count}/{num_selected} this round")
        cold_starts_per_round.append(int(cold_count))
        total_cold_starts += int(cold_count)

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
        # TRAINING PHASE.
        # =====================================================================
        train_config = ConfigRecord({
            "lr": lr,
            "lr_eta": float(lr_eta),
            "round_num": int(round_num),
            "run_id": str(run_id),
            "reuse_cache": bool(reuse_cache_flag),
        })

        train_messages = []
        for node_id in selected_node_ids:
            content = RecordDict({"arrays": arrays, "config": train_config})
            msg = grid.create_message(
                content=content,
                message_type="train",
                dst_node_id=node_id,
                group_id=f"train_round_{round_num}",
            )
            train_messages.append(msg)

        train_responses = list(grid.send_and_receive(train_messages))

        # =====================================================================
        # Pitfall 5 Option B: weight_policy="uniform" under paper_compat_pfedrec
        # means every client contributes weight=1 (mirrors engine.py:81
        # len(round_user_params) division). Setting FitRes.num_examples = 1
        # makes FedAvg's existing num_examples-weighted aggregate mathematically
        # uniform without overriding aggregate_fit. See RESEARCH §Pitfall 5.
        # =====================================================================
        fit_results = []
        round_train_metrics = []

        for response in train_responses:
            if response.has_error():
                print(
                    f"  Warning: Client {response.metadata.src_node_id} returned error"
                )
                continue

            resp_arrays = response.content.get("arrays", ArrayRecord())
            resp_metrics = response.content.get("metrics", MetricRecord())
            metrics_dict = dict(resp_metrics) if resp_metrics else {}

            # D-24 uniform weight under paper_compat_pfedrec.
            num_examples = 1  # D-24 uniform under paper_compat_pfedrec

            parameters = ndarrays_to_parameters(
                list(resp_arrays.to_torch_state_dict().values())
            )
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

        # Aggregate training results using strategy. PFedRecSplitFedAvg
        # inherits aggregate_fit unchanged from BaseFedAvg; the uniform
        # weighting is achieved by FitRes.num_examples = 1 above.
        if fit_results:
            aggregated_params, agg_metrics = strategy.aggregate_fit(
                server_round=round_num,
                results=fit_results,
                failures=[],
            )

            if aggregated_params is not None:
                param_ndarrays = parameters_to_ndarrays(aggregated_params)
                param_keys = list(arrays.to_torch_state_dict().keys())
                new_state_dict = {
                    k: torch.from_numpy(v) for k, v in zip(param_keys, param_ndarrays)
                }
                arrays = ArrayRecord(new_state_dict)

            train_metrics_history[round_num] = weighted_average_metrics(
                round_train_metrics
            )
            train_metrics_history[round_num].update(agg_metrics)

            train_loss = train_metrics_history[round_num].get("train_loss", "N/A")
            if isinstance(train_loss, (int, float)):
                print(f"  Training BCE loss: {train_loss:.4f}")

        # =====================================================================
        # EVALUATION PHASE.
        # =====================================================================
        eval_messages = []
        for node_id in selected_node_ids:
            eval_config = ConfigRecord({
                "lr": lr,
                "round_num": int(round_num),
                "run_id": str(run_id),
                "reuse_cache": bool(reuse_cache_flag),
            })
            content = RecordDict({"arrays": arrays, "config": eval_config})
            msg = grid.create_message(
                content=content,
                message_type="evaluate",
                dst_node_id=node_id,
                group_id=f"eval_round_{round_num}",
            )
            eval_messages.append(msg)

        eval_responses = list(grid.send_and_receive(eval_messages))

        # Wrap each eval response into EvaluateRes and let the strategy emit
        # thesis metrics from SUMMED sufficient stats (D-24 / D-26).
        eval_results: List[Tuple[ClientProxy, EvaluateRes]] = []
        round_eval_metrics = []  # retained for legacy diagnostic fallback (D-18).
        for response in eval_responses:
            if response.has_error():
                continue
            resp_metrics = response.content.get("metrics", MetricRecord())
            metrics_dict = dict(resp_metrics) if resp_metrics else {}
            num_examples = int(metrics_dict.get(
                "evaluated_users",
                metrics_dict.get("num-examples", 1),
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
            _agg_loss, thesis_metrics = strategy.aggregate_evaluate(
                round_num, eval_results, []
            )
            eval_metrics_history[round_num] = (
                dict(thesis_metrics) if thesis_metrics else {}
            )
            # Preserve eval_loss via the legacy per-client-ratio path (D-18).
            rating_agg = weighted_average_metrics(round_eval_metrics)
            for rk in ("rmse", "mae", "eval_loss"):
                if rk in rating_agg and rk not in eval_metrics_history[round_num]:
                    eval_metrics_history[round_num][rk] = rating_agg[rk]

            bce = eval_metrics_history[round_num].get("eval_loss", "N/A")
            hr10 = eval_metrics_history[round_num].get("sampled_hr@10", "N/A")
            ndcg10 = eval_metrics_history[round_num].get("sampled_ndcg@10", "N/A")
            bce_str = f"{bce:.4f}" if isinstance(bce, (int, float)) else str(bce)
            hr10_str = f"{hr10:.4f}" if isinstance(hr10, (int, float)) else str(hr10)
            ndcg10_str = (
                f"{ndcg10:.4f}" if isinstance(ndcg10, (int, float)) else str(ndcg10)
            )
            print(f"  BCE Loss: {bce_str}")
            print(f"  Sampled HR@10: {hr10_str}")
            print(f"  Sampled NDCG@10: {ndcg10_str}")

            # =================================================================
            # D-13 best-round-restore snapshot (Phase-3-D-27 carry-forward —
            # the in-memory snapshot-and-restore implementation pattern from
            # Phases 2/3/4 — NOT to be confused with CONTEXT.md D-27 which is
            # the weight-policy override behavior). Monitor metric is
            # sampled_ndcg@10 per CONTEXT.md D-13.
            # =================================================================
            if checkpoint_rule in ("best_round_restore", "best_round") and thesis_metrics:
                current_ndcg = float(thesis_metrics.get("sampled_ndcg@10", 0.0))
                if round_num == 1 or current_ndcg > best_metric:
                    best_metric = current_ndcg
                    best_round_num = round_num
                    best_arrays = ArrayRecord({
                        k: v.detach().clone()
                        for k, v in arrays.to_torch_state_dict().items()
                    })
                    print(
                        f"  [CHECKPOINT] New best sampled_ndcg@10={best_metric:.4f} "
                        f"at round {best_round_num}"
                    )

        # Log to wandb.
        if wandb_enabled and wandb_run is not None:
            round_log: Dict[str, Any] = {"round": round_num}
            for key, value in train_metrics_history.get(round_num, {}).items():
                if isinstance(value, (int, float)):
                    round_log[f"train/{key}"] = value
            for key, value in eval_metrics_history.get(round_num, {}).items():
                if isinstance(value, (int, float)):
                    round_log[f"eval/{key}"] = value
            wandb.log(round_log, step=round_num)

        # Early stopping (opt-in diagnostic).
        if early_stopper is not None and round_eval_metrics:
            current_eval_metrics = eval_metrics_history.get(round_num, {})
            if early_stopper.step(round_num, current_eval_metrics):
                print(f"\n  Training stopped early at round {round_num}")
                if wandb_enabled and wandb_run is not None:
                    wandb.log(
                        {
                            "early_stopped": True,
                            "early_stopped_round": round_num,
                            "best_round": early_stopper.best_round,
                            f"best_{early_stopping_metric}": early_stopper.best_metric,
                        },
                        step=round_num,
                    )
                break

    # Determine actual rounds completed.
    actual_rounds = (
        round_num if early_stopper and early_stopper.state.should_stop else num_rounds
    )

    print("\n" + "=" * 70)
    print("PFEDREC FEDERATED TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total rounds completed: {actual_rounds}/{num_rounds}")
    if early_stopper and early_stopper.state.should_stop:
        print(f"Early stopping: Triggered at round {actual_rounds}")
        print(
            f"Best {early_stopping_metric}: {early_stopper.best_metric:.4f} "
            f"at round {early_stopper.best_round}"
        )
    print("=" * 70)

    # =========================================================================
    # D-13 best-round-restore: restore best-round global params for the
    # manifest artifact. Matches the Phase-3-D-27 carry-forward idiom.
    # =========================================================================
    if checkpoint_rule in ("best_round_restore", "best_round") and best_round_num > 0:
        print(
            f"\n[CHECKPOINT] Restoring global params snapshot from best round "
            f"{best_round_num} (sampled_ndcg@10={best_metric:.4f})"
        )
        arrays = best_arrays
    else:
        print(
            f"\n[CHECKPOINT] checkpoint_rule={checkpoint_rule!r}: keeping last-round params"
        )

    # =========================================================================
    # D-06: extra eval round on the restored best-round state. All nodes
    # broadcast (no sampling). Result becomes the canonical final_metrics["best"].
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
            _agg_loss, thesis = strategy.aggregate_evaluate(final_eval_round_index, extra_results, [])
            # MAJOR fix (plan-checker iteration 1, np.float64 JSON-serialization):
            # coerce numeric values to Python floats at assignment so downstream
            # dataclass_replace + atomic_write_json never raise TypeError on np.float64.
            best_round_metrics = {
                k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                for k, v in (thesis or {}).items()
            }
            print(
                f"[D-06] Extra eval complete. Canonical best/sampled_ndcg@10="
                f"{best_round_metrics.get('sampled_ndcg@10')} "
                f"best/sampled_hr@10={best_round_metrics.get('sampled_hr@10')}"
            )
        else:
            print("[D-06] WARNING: no extra-eval responses; best block falls back to in-loop value.")

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
    }  # type: Dict[str, Any]

    print("\n  Using federated evaluation metrics...")
    print("  (Centralized evaluation not possible in split learning)")
    print_evaluation_metrics(
        final_metrics["best_round"],
        final_metrics["best"],
        context,
    )

    # W&B per-key final summary (D-07 best/* + last/* namespaces).
    if wandb_enabled and wandb_run is not None:
        final_log: Dict[str, Any] = {"round": actual_rounds + 1}
        for key, value in final_metrics["best"].items():
            if isinstance(value, (int, float)):
                final_log[f"final_eval/best/{key}"] = value
        wandb.log(final_log, step=actual_rounds + 1)

        if early_stopper:
            wandb.run.summary["early_stopping/enabled"] = early_stopping_enabled
            wandb.run.summary["early_stopping/stopped_early"] = (
                early_stopper.state.should_stop
            )
            wandb.run.summary["early_stopping/best_round"] = early_stopper.best_round
            wandb.run.summary[f"early_stopping/best_{early_stopping_metric}"] = (
                early_stopper.best_metric
            )
            wandb.run.summary["training/actual_rounds"] = actual_rounds

    # =========================================================================
    # Build results JSON. Phase 5 fields:
    #   - selected_clients_per_round (partition-id space, 0..N-1).
    #   - cold_starts (D-13 per_round / total / rate).
    #   - checkpoint (D-13 best-round-restore block).
    # =========================================================================
    total_selections = sum(len(r) for r in selected_clients_per_round)
    cold_start_rate = (
        total_cold_starts / total_selections if total_selections else 0.0
    )

    early_stopping_summary = early_stopper.get_summary() if early_stopper else None

    results_data: Dict[str, Any] = {
        "model_name": "PFedRec_MLP_FEDAVG",
        "algorithm": "PFedRec (IJCAI-23)",
        "dataset": "ml-1m",
        "architecture": "pfedrec_split_learning",
        "federated_config": {
            "num_rounds": num_rounds,
            "actual_rounds": actual_rounds,
            "num_clients": len(list(grid.get_node_ids())),
            "fraction_train": fraction_train,
            "strategy": "fedavg",  # D-07 — no FedProx variant.
            "latent_dim": latent_dim,
            "lr": lr,
            "lr_eta": lr_eta,
            "num_negatives": num_train_negatives,
            "l2_regularization": context.run_config.get("l2-regularization", 0.0),
            "local_epochs": local_epochs,
            "optimizer": "sgd",
            "loss": "bce",
            "split_learning": True,
            # D-01: bias is now GLOBAL.
            "global_params": ["embedding_item.weight", "affine_output.bias"],
            "local_params": ["affine_output.weight"],
            # D-25 contract keys.
            "mode": mode,
            "run_seed": run_seed,
            "weight_policy": weight_policy,
            "checkpoint_rule": checkpoint_rule,
            "reuse_cache": reuse_cache_flag,
        },
        "early_stopping": early_stopping_summary,
        "timestamp": datetime.now().isoformat(),
        "final_metrics": final_metrics,
        "training_rounds": actual_rounds,
        "eval_metrics_history": eval_metrics_history,
        "train_metrics_history": train_metrics_history,
        # Selected clients live in partition-id space (stable 0..N-1).
        "selected_clients_per_round": selected_clients_per_round,
        # D-13 best-round-restore block (Phase-3-D-27 carry-forward idiom).
        "checkpoint": {
            "rule": checkpoint_rule,
            "best_round": best_round_num,
            "best_sampled_ndcg@10": (
                best_metric if best_metric != float("-inf") else None
            ),
        },
        # D-13 cold-start accounting.
        "cold_starts": {
            "per_round": cold_starts_per_round,
            "total_cold_starts": total_cold_starts,
            "total_client_selections": total_selections,
            "cold_start_rate": cold_start_rate,
        },
    }

    # =========================================================================
    # PFR-09 / D-15: protocol fingerprint manifest (FND-07 + double-write).
    # module="pfedrec".
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
        module="pfedrec",
    )

    # Phase 6 D-06/D-07: mutate manifest with final_eval_round_index + metrics
    # AFTER final_metrics is assigned and BEFORE embed_manifest_in_result.
    # Phase 7 D-22: thesis-tagging fields read from run_config; sentinels for non-thesis runs.
    # PFedRec runs at paper_compat_pfedrec mode (D-06), but the orchestrator passes
    # thesis-run-label=main regardless of mode — so the manifest mutation is the
    # load-bearing change here even though PFedRec rarely runs at thesis_crossdevice_main.
    manifest = dataclass_replace(manifest,
        final_eval_round_index=final_eval_round_index,
        metrics=results_data["final_metrics"],
        thesis_run_label=str(context.run_config.get("thesis-run-label", "")),
        ablation_dimension=str(context.run_config.get("ablation-dimension", "none")),
        ablation_value=str(context.run_config.get("ablation-value", "")),
    )

    # D-15 part 1: embed manifest INTO the result JSON.
    embed_manifest_in_result(manifest, results_data)

    # D-15 SC-1 back-pointer: PFR-02-AUDIT.md is the cross-walk authored by
    # Plan 01 Task 3. build_run_manifest does NOT accept ``audit_doc`` directly;
    # we thread it via post-build mutation of the embedded ``_manifest`` dict
    # (Phase-3/Phase-4 idiom for post-build payload extensions). Future
    # consumers of the result JSON can follow this pointer to the SC-1 audit.
    # The string token ``audit_doc="PFR-02-AUDIT.md"`` is also pinned by the
    # source-level test at federated-pfedrec/tests/test_server_integration.py
    # (test_manifest_double_write_module_pfedrec).
    audit_doc = "PFR-02-AUDIT.md"
    results_data["_manifest"]["audit_doc"] = audit_doc

    # =========================================================================
    # D-14 PFR-08 auto-verify hook. Fires AFTER embed_manifest_in_result
    # (so we can inject the audit dict into _manifest) AND BEFORE the W&B
    # summary write (so failure surfaces in W&B). Non-fatal — a failed
    # reproduction does NOT raise / abort the run.
    # =========================================================================
    _ref_root = Path(__file__).resolve().parents[2]
    reference_path = _ref_root / "IJCAI-23-PFedRec" / "sh_result" / "ml-1m.txt"
    # Pitfall 1 closure: under the new D-07 nested schema, sampled_hr@10 and
    # sampled_ndcg@10 live at final_metrics["best"][...], NOT at
    # final_metrics[...]. Passing final_metrics directly would make the hook
    # read None for both keys and stamp PFR-08 FAILED with NaN deltas.
    pfr08_passed, pfr08_log_line, pfr08_audit = _emit_pfr_08_verification(
        final_metrics["best"],
        reference_path=reference_path,
        tolerance_pts=2.0,
    )
    print(pfr08_log_line)
    # Embed the audit dict in the manifest so the result JSON carries it.
    results_data["_manifest"]["pfr08_verification"] = pfr08_audit

    # =========================================================================
    # Phase 6 D-01/D-02: per-module per-run directory layout for cross-device.
    # Phase 7 D-04: thesis_crossdevice_main joins the per-run-dir gate.
    # Cross-silo legacy (cross_silo_legacy) keeps the pre-Phase-6 flat layout
    # (D-03 + PROJECT.md backwards-compat constraint).
    # =========================================================================
    print("\nSaving evaluation results...")
    if mode in ("benchmark_cross_device", "thesis_crossdevice_main", "paper_compat_pfedrec"):
        run_dir = module_run_results_dir(_MODULE, run_id)
        results_filename = run_dir / "results.json"  # D-04 clean filename
        atomic_write_json(str(results_filename), results_data)
        sibling_path = write_manifest_sibling(manifest, results_filename, sibling_name="manifest.json")
    else:  # cross_silo_legacy — preserved per D-03 + PROJECT.md backwards-compat constraint
        legacy_dir = repo_root() / "results" / "federated" / "pfedrec"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        results_filename = legacy_dir / f"{run_id}_results.json"
        sibling_kwarg = {}  # default <run_id>-manifest.json — byte-identical to pre-Phase-6
        atomic_write_json(str(results_filename), results_data)
        sibling_path = write_manifest_sibling(manifest, results_filename)
    print(f"Results saved to: {results_filename.resolve()}")
    print(f"Manifest sibling: {sibling_path.resolve()}")

    # =========================================================================
    # D-14 W&B summary push (non-fatal). Fires AFTER embed_manifest_in_result
    # to keep the documented sequence: embed -> verify -> stdout -> JSON write
    # -> W&B summary.
    # D-07 migration: thesis metrics use best/* + last/* namespaces.
    # PFR-08 audit migrates from final/pfr08* to top-level pfr08* (independent
    # surface — the audit dict is its own W&B namespace).
    # =========================================================================
    if wandb_enabled and wandb_run is not None:
        # PFR-08 audit surface (independent of best/last namespacing — top-level keys)
        wandb.run.summary["pfr08"] = bool(pfr08_passed)
        wandb.run.summary["pfr08_delta_hr_pts"] = float(
            pfr08_audit.get("delta_hr_pts", float("nan"))
        )
        wandb.run.summary["pfr08_delta_ndcg_pts"] = float(
            pfr08_audit.get("delta_ndcg_pts", float("nan"))
        )
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
        wandb.config.update({
            "_manifest": {
                "run_id": manifest.run_id,
                "mode": manifest.mode,
                "num_supernodes": manifest.num_supernodes,
                "foundation_contract_sha256": manifest.foundation_contract_sha256,
                "split_hash": manifest.split_hash,
                "run_seed": manifest.run_seed,
                "checkpoint_rule": manifest.checkpoint_rule,
                "audit_doc": audit_doc,
            }
        })
        wandb.run.summary["total_cold_starts"] = int(total_cold_starts)
        wandb.run.summary["cold_start_rate"] = float(cold_start_rate)

    if wandb_enabled and wandb_run is not None:
        wandb.finish()
        print("  Weights & Biases run completed")
