"""federated-baseline-cf: cross-device Flower client for Matrix Factorization (Phase 2).

Phase 2 Plan 03 migrates the @app.train() and @app.evaluate() handlers to:
  - Resolve a ModeProfile (benchmark_cross_device / paper_compat_pfedrec /
    cross_silo_legacy) via fedrec_foundation.mode.resolve_mode_defaults.
  - Assert 1-user-per-client under benchmark mode (BSL-02).
  - Thread FND-06 RNG seeds (run_seed, user_idx, round_num) into task.py
    training and evaluation (BSL-05).
  - Source the per-user exclusion set from FND-03
    (ExclusionTable.for_user(user_idx)) so the held-out test positive is
    NEVER drawn as a train or eval negative (BSL-03).
  - Return FitMetricsContract.to_dict() (validated) from @app.train() and
    EvaluateMetricsContract.to_dict() (validated) from @app.evaluate()
    — strict contract wire payloads (D-21) with per-group sufficient
    stats (D-22) for the server's BaselineFedAvg.aggregate_evaluate
    (BSL-06) to sum directly.
  - Select the primary evaluator via get_primary_evaluator(mode) (BSL-07).

Pre-existing WIP from earlier uncommitted hunks — the ``get_device`` helper
and ``_device_cache`` module global — is preserved verbatim (D-18 surgical
edit discipline).
"""
from __future__ import annotations

from typing import Dict

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

# Phase 2 Plan 03 foundation imports (D-21/D-22/BSL-02/BSL-03/BSL-05/BSL-07).
from fedrec_foundation.evaluator import get_primary_evaluator
from fedrec_foundation.fit_metrics import (
    EvaluateMetricsContract,
    FitMetricsContract,
    validate_evaluate_metrics,
    validate_fit_metrics,
)
from fedrec_foundation.mode import (
    assert_benchmark_one_user_per_client,
    log_mode_and_overrides,
    resolve_mode_defaults,
)
from fedrec_foundation.rng import np_rng
from fedrec_foundation.user_groups import classify_user_group

from federated_baseline_cf.dataset import _load_foundation_bundle
from federated_baseline_cf.task import get_model, load_data
from federated_baseline_cf.task import test as test_fn
from federated_baseline_cf.task import train as train_fn
from federated_baseline_cf.task import evaluate_ranking, evaluate_ranking_sampled

# Flower ClientApp
app = ClientApp()

# Cache for device detection (avoid repeated CUDA tests)
_device_cache = None


def get_device():
    """Get device with safe CUDA detection (handles incompatible GPU architectures)."""
    global _device_cache
    if _device_cache is not None:
        return _device_cache

    if torch.cuda.is_available():
        try:
            # Test if CUDA actually works by creating a small tensor
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            _device_cache = torch.device("cuda:0")
        except RuntimeError:
            # CUDA available but not compatible (e.g., RTX 5090 with old PyTorch)
            _device_cache = torch.device("cpu")
    else:
        _device_cache = torch.device("cpu")

    return _device_cache


def _classify_partition_user_group(bundle: Dict, partition_id: int) -> str:
    """Return the ``"sparse" | "medium" | "dense"`` label for this client's user.

    Reads from ``bundle["split_manifest"].train_user_stats[partition_id]
    .user_group`` when present (pre-computed by the foundation builder on
    TRAIN-only rows per CR-5). Falls back to
    ``classify_user_group(n_interactions=0)`` (i.e. ``"sparse"``) if the
    user_idx is not present — which only happens when the user has fewer
    than 2 ratings and the foundation split elided them.

    Parameters
    ----------
    bundle : Dict
        Output of ``_load_foundation_bundle()`` from ``dataset.py``.
    partition_id : int
        The client's user_idx (under ``partition_mode="natural"``).

    Returns
    -------
    str
        One of ``"sparse"`` / ``"medium"`` / ``"dense"``.
    """
    stats = bundle["split_manifest"].train_user_stats
    entry = stats.get(int(partition_id))
    if entry is not None:
        return entry.user_group
    return classify_user_group(0)


@app.train()
def train(msg: Message, context: Context):
    """Train the Matrix Factorization model on ONE user's local data.

    Phase 2 Plan 03 contract hooks:

    - **BSL-02**: asserts ``num_users_in_client == 1`` under benchmark
      mode via ``assert_benchmark_one_user_per_client(profile, n,
      overrides)`` BEFORE any training happens.
    - **BSL-03**: training negatives drawn from the FND-03 exclusion set
      so the held-out test positive NEVER becomes a training negative.
    - **BSL-05**: RNG seeded via
      ``np_rng(run_seed, user_idx, round_num, "train_neg")``; no
      process-global RNG seeding.
    - **D-21**: return payload is
      ``FitMetricsContract.to_dict()``; ``validate_fit_metrics`` runs
      as defense-in-depth before the reply is sent.
    """
    # Resolve mode profile (Phase 1 Plan 05 contract).
    mode = context.run_config.get("mode", "cross_silo_legacy")
    profile = resolve_mode_defaults(mode)
    overrides = log_mode_and_overrides(mode, profile, context.run_config)

    # Per-client identity for RNG + exclusion lookup.
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    round_num = int(msg.content["config"].get("round_num", 1))
    run_seed = int(context.run_config.get("run-seed", 42))

    # Model setup.
    model_type = context.run_config.get("model-type", "bpr")
    embedding_dim = int(context.run_config.get("embedding-dim", profile.embedding_dim))
    dropout = float(context.run_config.get("dropout", 0.1))
    model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()
    model.to(device)

    # === FedProx: Save global parameters BEFORE training ===
    # Get proximal_mu from config (0.0 means standard FedAvg behavior)
    proximal_mu = float(msg.content["config"].get("proximal_mu", 0.0))

    # Save global parameters for proximal term (only if proximal_mu > 0)
    global_params = None
    if proximal_mu > 0:
        global_params = [p.detach().clone() for p in model.parameters()]
    # === End FedProx modification ===

    # Load the data — partition_mode resolved from profile default then overridden.
    alpha = float(context.run_config.get("alpha", 0.5))
    split_mode = context.run_config.get("eval-split-mode", "leave-one-out")
    partition_mode = context.run_config.get("partition-mode", profile.partition_mode)
    trainloader, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        alpha=alpha,
        split_mode=split_mode,
        partition_mode=partition_mode,
    )

    # BSL-02: benchmark-mode single-user assertion. Under
    # ``partition_mode="natural"`` the partition IS a single user, so
    # ``num_users_in_client == 1`` whenever the trainloader is non-empty.
    # We iterate distinct user ids from the loader so the check doesn't
    # depend on ordering.
    user_ids_in_client = set()
    for batch in trainloader:
        user_ids_in_client.update(batch["user"].cpu().numpy().tolist())
    num_users_in_client = len(user_ids_in_client)
    assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)

    # FND-03 exclusion set for this user (empty array if user has no
    # exclusions — e.g. single-interaction users that the foundation
    # split elided).
    bundle = _load_foundation_bundle()
    exclude_items = bundle["exclusion"].for_user(partition_id)

    # FND-06 RNG instance for training-negative sampling.
    train_rng = np_rng(run_seed, partition_id, round_num, "train_neg")

    # Call the training function — thread run_seed / user_idx / round_num
    # / exclude_items / rng into task.train.
    local_epochs = int(context.run_config.get("local-epochs", profile.local_epochs))
    lr = float(msg.content["config"].get("lr", profile.lr))
    num_train_negatives = int(context.run_config.get("num-negatives", profile.num_train_negatives))
    train_loss = train_fn(
        model=model,
        trainloader=trainloader,
        epochs=local_epochs,
        lr=lr,
        device=device,
        model_type=model_type,
        weight_decay=float(context.run_config.get("weight-decay", 1e-5)),
        num_negatives=num_train_negatives,
        # FedProx parameters
        proximal_mu=proximal_mu,
        global_params=global_params,
        # Phase 2 Plan 03 contract (BSL-03, BSL-05, D-24).
        run_seed=run_seed,
        user_idx=partition_id,
        round_num=round_num,
        exclude_items=exclude_items,
        rng=train_rng,
    )

    # D-21 strict-contract wire payload. num_positives = count of rating
    # rows; num_training_examples = positives + sampled negatives per
    # positive per local epoch. round_num embedded for downstream logs.
    num_positives = int(len(trainloader.dataset))
    num_training_examples = int(num_positives * (1 + max(num_train_negatives, 0)))
    fit_metrics = FitMetricsContract(
        train_loss=float(train_loss),
        num_positives=num_positives,
        num_training_examples=num_training_examples,
        round_num=round_num,
    ).to_dict()
    # Defense-in-depth: validate before sending to catch contract drift.
    validate_fit_metrics(fit_metrics)

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metric_record = MetricRecord(fit_metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate this client's one user on the held-out positive.

    Phase 2 Plan 03 contract hooks:

    - **BSL-02**: asserts ``num_users_in_client == 1`` under benchmark
      mode (same lock as @app.train).
    - **BSL-03**: evaluator excludes the FND-03 exclusion set from the
      sampled-negative pool.
    - **BSL-05**: evaluator RNG seeded via
      ``np_rng(run_seed, user_idx, round_num, "eval_neg")``.
    - **BSL-07**: primary evaluator selected via
      ``get_primary_evaluator(mode)``. ``allrank_*`` metrics (when
      ``enable-ranking-eval=true``) stay namespaced and are NOT consumed
      by the server's thesis-table metrics.
    - **D-21**: return payload is
      ``EvaluateMetricsContract.to_dict()`` (not
      ``FitMetricsContract``) validated via
      ``validate_evaluate_metrics`` before send — rejects free-form
      extras so the ``BaselineFedAvg.aggregate_evaluate`` reader only
      sees known contract fields.
    - **D-22**: per-group sufficient-stat fields
      (``hit_count_{sparse,medium,dense}_at10``,
      ``ndcg_sum_{sparse,medium,dense}_at10``,
      ``evaluated_users_{sparse,medium,dense}``) travel in the payload;
      the client's user-group receives the non-zero values; the other
      two groups receive zeros.
    """
    mode = context.run_config.get("mode", "cross_silo_legacy")
    profile = resolve_mode_defaults(mode)
    overrides = log_mode_and_overrides(mode, profile, context.run_config)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    round_num = int(msg.content["config"].get("round_num", 1))
    run_seed = int(context.run_config.get("run-seed", 42))

    # Get model configuration
    model_type = context.run_config.get("model-type", "bpr")
    embedding_dim = int(context.run_config.get("embedding-dim", profile.embedding_dim))
    dropout = float(context.run_config.get("dropout", 0.1))

    # Load the model and initialize it with the received weights
    model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Use safe device detection (handles incompatible GPU architectures)
    device = get_device()
    model.to(device)

    # Load the data (both train and test for item popularity computation).
    alpha = float(context.run_config.get("alpha", 0.5))
    split_mode = context.run_config.get("eval-split-mode", "leave-one-out")
    partition_mode = context.run_config.get("partition-mode", profile.partition_mode)
    trainloader, testloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        alpha=alpha,
        split_mode=split_mode,
        partition_mode=partition_mode,
    )

    # BSL-02: same one-user lock as @app.train (benchmark mode implies
    # 1 user per partition). Pull distinct user ids from the testloader
    # first; fall back to trainloader if the held-out test set is empty
    # (single-interaction users elided by foundation split).
    user_ids_in_client = set()
    for batch in testloader:
        user_ids_in_client.update(batch["user"].cpu().numpy().tolist())
    if not user_ids_in_client:
        for batch in trainloader:
            user_ids_in_client.update(batch["user"].cpu().numpy().tolist())
    num_users_in_client = len(user_ids_in_client)
    assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)

    # BSL-07: only the primary evaluator feeds thesis-table metrics.
    primary = get_primary_evaluator(mode)
    assert primary == "sampled_loo_99", (
        f"BSL-07 invariant broken: get_primary_evaluator('{mode}') returned "
        f"{primary!r}, expected 'sampled_loo_99'"
    )

    # FND-03 exclusion set for this user.
    bundle = _load_foundation_bundle()
    exclude_items = bundle["exclusion"].for_user(partition_id)

    # Call the rating-prediction test (for RMSE / MAE diagnostics only —
    # NOT consumed by the thesis-table aggregator).
    eval_loss, _rating_metrics = test_fn(
        model=model,
        testloader=testloader,
        device=device,
        model_type=model_type,
    )

    # BSL-05 + BSL-07 primary-path evaluation.
    num_eval_negatives = int(
        context.run_config.get("eval-num-negatives", profile.num_eval_negatives)
    )
    sampled_metrics = evaluate_ranking_sampled(
        model=model,
        testloader=testloader,
        trainloader=trainloader,
        device=str(device),
        k_values=[10],
        num_negatives=num_eval_negatives,
        run_seed=run_seed,
        user_idx=partition_id,
        round_num=round_num,
        exclude_items=exclude_items,
    )

    # Optional all-items ranking — runs only when explicitly enabled and
    # stays NAMESPACED as ``allrank_*`` so the thesis-table aggregator
    # (BSL-06) does not mix it with the primary ``sampled_*`` fields
    # (BSL-07). The returned dict is cached client-side for logs only;
    # the strict-contract wire payload does NOT carry these keys.
    enable_ranking_eval = bool(context.run_config.get("enable-ranking-eval", False))
    if enable_ranking_eval:
        k_values_str = context.run_config.get("ranking-k-values", "5,10,20")
        k_values = [int(k.strip()) for k in k_values_str.split(",")]
        # Side-effect: populate item-popularity cache for potential
        # server-side logging. The return value is intentionally dropped
        # to avoid leaking ``allrank_*`` keys into the wire payload.
        _ = evaluate_ranking(
            model=model,
            testloader=testloader,
            device=str(device),
            k_values=k_values,
            trainloader=trainloader,
        )

    # D-22: per-group sufficient-stat routing. In benchmark mode the
    # partition is one user, so the stats for that user's group carry
    # the non-zero values; the other two groups carry zeros.
    user_group = _classify_partition_user_group(bundle, partition_id)
    sampled_num_users = int(sampled_metrics.get("sampled_num_users", 0))
    # Reconstruct sufficient stats from the ratio * denominator product.
    hr10_ratio = float(sampled_metrics.get("sampled_hr@10", 0.0))
    ndcg10_ratio = float(sampled_metrics.get("sampled_ndcg@10", 0.0))
    hit10 = int(round(hr10_ratio * sampled_num_users))
    ndcg10_sum = float(ndcg10_ratio * sampled_num_users)
    evaluated_users = sampled_num_users

    per_group = {g: {"hit": 0, "ndcg": 0.0, "users": 0} for g in ("sparse", "medium", "dense")}
    per_group[user_group]["hit"] = hit10
    per_group[user_group]["ndcg"] = ndcg10_sum
    per_group[user_group]["users"] = evaluated_users

    # D-21 + D-22 strict-contract wire payload.
    # Required (sufficient stats consumed by server aggregator):
    #   hit_count_overall_at10, ndcg_sum_overall_at10, evaluated_users.
    # Optional diagnostic caches (NOT consumed by aggregator — server
    #   re-computes from summed sufficient stats):
    #   eval_loss, sampled_hr_at10, sampled_ndcg_at10.
    # Optional per-group sufficient-stat fields (D-22): 9 keys, one
    #   group gets real numbers; the others zero.
    eval_payload = EvaluateMetricsContract(
        hit_count_overall_at10=hit10,
        ndcg_sum_overall_at10=ndcg10_sum,
        evaluated_users=evaluated_users,
        eval_loss=float(eval_loss),
        sampled_hr_at10=hr10_ratio,
        sampled_ndcg_at10=ndcg10_ratio,
        hit_count_sparse_at10=per_group["sparse"]["hit"],
        ndcg_sum_sparse_at10=per_group["sparse"]["ndcg"],
        evaluated_users_sparse=per_group["sparse"]["users"],
        hit_count_medium_at10=per_group["medium"]["hit"],
        ndcg_sum_medium_at10=per_group["medium"]["ndcg"],
        evaluated_users_medium=per_group["medium"]["users"],
        hit_count_dense_at10=per_group["dense"]["hit"],
        ndcg_sum_dense_at10=per_group["dense"]["ndcg"],
        evaluated_users_dense=per_group["dense"]["users"],
    ).to_dict()
    # Defense-in-depth: reject free-form extras before sending (D-21).
    validate_evaluate_metrics(eval_payload)

    metric_record = MetricRecord(eval_payload)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
