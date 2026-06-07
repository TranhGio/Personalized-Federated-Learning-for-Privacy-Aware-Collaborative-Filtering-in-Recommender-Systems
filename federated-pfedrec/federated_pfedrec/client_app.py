"""federated-pfedrec: cross-device PFedRec Flower client (Phase 5 Plan 03).

Phase 5 Plan 03 migrates ``@app.train()`` / ``@app.evaluate()`` to the cross-
device PFedRec contract:

- **PFR-05** single-user collapse: in benchmark mode, the client partition
  contains exactly one user (``partition_id == user_idx``). The legacy
  per-user-key iteration is GONE. Both handlers call
  ``assert_benchmark_one_user_per_client(profile, num_users, overrides)``
  before any heavy work.
- **PFR-04 / FND-03**: ``ExclusionTable.for_user(user_idx)`` is threaded into
  ``prepare_user_train_data`` (training-negative pool) AND
  ``evaluate_pfedrec_sampled`` (eval-negative pool). The held-out test
  positive is provably never drawn.
- **PFR-06 / PFR-07 / FND-06**: every stochastic step routes through
  ``np_rng(run_seed, user_idx, round_num, "<purpose>")``. Stdlib ``random``
  is NOT imported anywhere in this module (BSL-05 cross-file regression).
- **D-16 / D-17 / D-21 / D-22 manifest-sidecar cache**: per-partition
  ``partition_{pid}.pt`` (single-file layout) plus ``manifest.json``
  schema_v3 sidecar with the ``bias_classification='global'`` D-01 sentinel.
  ``D-22`` cold-round probe-then-load. ``D-21`` strict shape guard fires on
  BOTH save AND load. ``Pitfall 6`` ``torch.load(weights_only=True)``.
- **G-03-01** discover_only short-circuit: ``@app.evaluate`` checks
  ``msg.content['config']['discover_only']`` FIRST and returns a zero-
  suffstats ``EvaluateMetricsContract`` payload (with ``partition_id``)
  before any model / bundle / data load.
- **D-04** eval-time BCE loss over (positive + 99 negatives) — the contract
  hook is the kwargs threaded into ``evaluate_pfedrec_sampled``; the BCE
  computation lives in ``task.py``.

SC-2 / D-01 reconciliation note (also surfaced in ``_signature_fields``
docstring): per-user disk payload carries ONLY ``affine_output.weight``;
SC-2's "atomic per-user (weight, bias) artifact" requirement is preserved
end-to-end because the bias channel is aggregated atomically server-side
per ``IJCAI-23-PFedRec/engine.py:143``. ``PFR-02-AUDIT.md`` (Plan 01 Task 3)
carries the human-readable cross-walk.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp

from fedrec_foundation.atomic import atomic_write_json
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
from fedrec_foundation.user_groups import classify_user_group

from federated_pfedrec.dataset import _load_foundation_bundle
from federated_pfedrec.task import (
    evaluate_pfedrec_sampled,
    get_model,
    load_data,
    prepare_user_train_data,
    train_pfedrec_single_user,
)


# Flower ClientApp.
app = ClientApp()

# Module directory for cache path resolution + tests.
_MODULE_DIR = Path(__file__).parent

# D-16 cache base dir — module-level for test-time monkeypatching. Production
# cache lives under ``{module_root}/.embedding_cache``.
_CACHE_BASE_DIR: Path = _MODULE_DIR.parent / ".embedding_cache"

# Cache for device detection (avoid repeated CUDA tests).
_device_cache: Optional[torch.device] = None


# =============================================================================
# Device detection (D-18 preserved from pre-existing WIP).
# =============================================================================


def get_device() -> torch.device:
    """Get device with safe CUDA detection (handles incompatible GPU architectures)."""
    global _device_cache
    if _device_cache is not None:
        return _device_cache

    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            _device_cache = torch.device("cuda:0")
        except RuntimeError:
            # CUDA available but not compatible (e.g. RTX 5090 + old PyTorch).
            _device_cache = torch.device("cpu")
    else:
        _device_cache = torch.device("cpu")
    return _device_cache


# =============================================================================
# D-16..D-22 manifest-sidecar embedding cache helpers (PFedRec schema_v3).
# =============================================================================


def _signature_fields(
    *,
    run_id: str,
    method: str = "pfedrec",
    num_users: int,
    num_items: int,
    latent_dim: int,
    split_hash: str,
    loss: str = "bce",
    num_train_negatives: int = 4,
    bias_classification: str = "global",
) -> Dict[str, Any]:
    """Build the schema_v3 10-field PFedRec cache signature.

    SC-2 / D-01 reconciliation note: the per-user disk payload carries ONLY
    ``affine_output.weight``. SC-2's ``(affine_output.weight,
    affine_output.bias)`` atomic per-user artifact phrase is reconciled with
    D-01 by aggregating the bias channel atomically server-side per
    ``IJCAI-23-PFedRec/engine.py:143``. The ``bias_classification='global'``
    field is a D-17 sentinel that catches any future regression that reverts
    D-01 (cache load hard-fails on signature mismatch).

    Parameters
    ----------
    run_id : str
        Flower run identifier.
    method : str
        ``"pfedrec"`` literal — distinguishes the cache from sibling modules.
    num_users : int
        Catalog user-population size (6040 for ML-1M).
    num_items : int
        Catalog item-population size (3706 for ML-1M).
    latent_dim : int
        Embedding dimensionality — must match runtime model.
    split_hash : str
        ``SplitManifest.split_hash`` — guards against silent split drift.
    loss : str
        ``"bce"`` — catches future regressions that change the loss family.
    num_train_negatives : int
        Negatives per positive at training time (paper default: 4).
    bias_classification : str
        ``"global"`` D-01 sentinel.

    Returns
    -------
    Dict[str, Any]
        10-field signature dict.
    """
    return {
        "schema_version": 3,
        "run_id": str(run_id),
        "method": str(method),
        "num_users": int(num_users),
        "num_items": int(num_items),
        "latent_dim": int(latent_dim),
        "split_hash": str(split_hash),
        "loss": str(loss),
        "num_train_negatives": int(num_train_negatives),
        "bias_classification": str(bias_classification),
    }


def _cache_dir_for_run(
    *,
    run_id: str,
    reuse_cache: bool,
    signature: Dict[str, Any],
) -> Path:
    """D-18 cache path resolver.

    ``reuse_cache=False`` (default): ``{base}/{run_id}/``.

    ``reuse_cache=True``: ``{base}/sig_<sha256[:16]>/`` — two runs with
    identical signature fields (ignoring ``run_id``) share the same cache.

    Parameters
    ----------
    run_id : str
        Used only under ``reuse_cache=False``.
    reuse_cache : bool
        Opt-in reuse flag (D-18).
    signature : Dict[str, Any]
        Signature dict from ``_signature_fields``.

    Returns
    -------
    pathlib.Path
        Resolved cache directory (NOT created here).
    """
    base = Path(_CACHE_BASE_DIR)
    if not reuse_cache:
        return base / str(run_id)
    payload = json.dumps(
        {k: v for k, v in signature.items() if k not in ("run_id", "schema_version")},
        sort_keys=True,
    ).encode("utf-8")
    sig_hex = hashlib.sha256(payload).hexdigest()[:16]
    return base / f"sig_{sig_hex}"


def _save_local_user_state(
    *,
    partition_id: int,
    state_dict: Dict[str, torch.Tensor],
    run_id: str,
    reuse_cache: bool,
    signature: Dict[str, Any],
) -> None:
    """D-16 + D-17 + D-21 atomic save of the per-partition local state.

    D-21 shape guard fires BEFORE any disk write: payload MUST have exactly
    one key (``affine_output.weight``) with shape ``(1, latent_dim)``. The
    SC-2 / D-01 reconciliation: the bias channel lives in the server-side
    aggregation (per ``engine.py:143``), not on per-user disk.

    Parameters
    ----------
    partition_id : int
        Client's partition index (== ``user_idx`` under cross-device).
    state_dict : Dict[str, torch.Tensor]
        Must have keys ``{'affine_output.weight'}`` exactly (D-21).
    run_id : str
        Flower run identifier.
    reuse_cache : bool
        D-18 opt-in flag.
    signature : Dict[str, Any]
        Output of ``_signature_fields``; written verbatim to ``manifest.json``.

    Raises
    ------
    AssertionError
        When ``state_dict`` violates the D-21 single-key contract.
    """
    # D-21 shape guard BEFORE any disk write. Two layers — key-set and tensor shape.
    assert set(state_dict.keys()) == {"affine_output.weight"}, (
        f"D-21 expected single-key payload {{'affine_output.weight'}}, "
        f"got {sorted(state_dict.keys())}"
    )
    expected_shape = (1, int(signature["latent_dim"]))
    actual_shape = tuple(state_dict["affine_output.weight"].shape)
    assert actual_shape == expected_shape, (
        f"D-21 expected shape {expected_shape}, got {actual_shape}"
    )

    cache_dir = _cache_dir_for_run(
        run_id=run_id, reuse_cache=reuse_cache, signature=signature
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    # D-17 manifest sidecar (atomic JSON write).
    atomic_write_json(str(cache_dir / "manifest.json"), signature)

    # D-16 atomic .pt write — Phase 3 Rule-1: tempfile prefix MUST NOT start
    # with '.' (PyTorchFileWriter rejects it).
    pt_path = cache_dir / f"partition_{int(partition_id)}.pt"
    fd, tmp = tempfile.mkstemp(dir=str(cache_dir), prefix="partition_tmp_", suffix=".pt")
    os.close(fd)
    try:
        torch.save(OrderedDict(state_dict), tmp)
        os.replace(tmp, str(pt_path))
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _load_local_user_state(
    *,
    partition_id: int,
    run_id: str,
    reuse_cache: bool,
    signature: Dict[str, Any],
) -> Optional[Dict[str, torch.Tensor]]:
    """D-22 cold-round probe-then-load with D-21 strict shape guard + Pitfall 6.

    Returns ``None`` on cache miss (D-22 cold round). Raises ``RuntimeError``
    with per-field delta + literal ``rm -rf .embedding_cache/{run_id}/`` hint
    on signature mismatch (D-17). Raises ``AssertionError`` on D-21 shape
    mismatch AFTER load. ``torch.load`` uses ``weights_only=True`` (Pitfall 6).

    Parameters
    ----------
    partition_id : int
        Client's partition index.
    run_id : str
        Current run identifier.
    reuse_cache : bool
        D-18 opt-in flag.
    signature : Dict[str, Any]
        Current run's signature dict.

    Returns
    -------
    Optional[Dict[str, torch.Tensor]]
        Single-key state dict on hit; ``None`` on cold start.

    Raises
    ------
    RuntimeError
        On any signature-field mismatch (D-17).
    AssertionError
        On D-21 shape contract violation in the loaded payload.
    """
    cache_dir = _cache_dir_for_run(
        run_id=run_id, reuse_cache=reuse_cache, signature=signature
    )
    pt_path = cache_dir / f"partition_{int(partition_id)}.pt"
    manifest_path = cache_dir / "manifest.json"

    # D-22 cold-round probe. Probe the .pt path explicitly so the manifest
    # alone never causes a load.
    if not pt_path.exists() or not manifest_path.exists():
        return None

    # D-17 manifest signature comparison.
    with open(manifest_path) as f:
        on_disk = json.load(f)
    deltas: List[str] = []
    for key in signature:
        if reuse_cache and key == "run_id":
            continue
        if on_disk.get(key) != signature[key]:
            deltas.append(
                f"{key}: cached={on_disk.get(key)!r} vs current={signature[key]!r}"
            )
    if deltas:
        raise RuntimeError(
            "D-17 PFedRec cache signature mismatch:\n  "
            + "\n  ".join(deltas)
            + f"\nRun: rm -rf {cache_dir}/ to reset, "
            f"or check --run-config / run_id={run_id!r} for drifted keys."
        )

    # Pitfall 6: weights_only=True (PyTorch 2.6+ safe default).
    state = torch.load(str(pt_path), map_location="cpu", weights_only=True)

    # D-21 shape guard AFTER load.
    assert set(state.keys()) == {"affine_output.weight"}, (
        f"D-21 expected single-key payload, got {sorted(state.keys())}"
    )
    expected_shape = (1, int(signature["latent_dim"]))
    actual_shape = tuple(state["affine_output.weight"].shape)
    assert actual_shape == expected_shape, (
        f"D-21 expected shape {expected_shape}, got {actual_shape}"
    )
    return state


def _classify_partition_user_group(bundle: Any, partition_id: int) -> str:
    """Return the ``"sparse" | "medium" | "dense"`` label for this client's user.

    Reads from ``bundle.split_manifest.train_user_stats[partition_id].user_group``
    when present (pre-computed by the foundation builder on TRAIN-only rows
    per CR-5). Falls back to ``classify_user_group(0)`` (i.e. ``"sparse"``)
    for users elided from the foundation split.

    Parameters
    ----------
    bundle : _FoundationBundle
        Output of ``_load_foundation_bundle()``.
    partition_id : int
        Client's user_idx (under ``partition_mode='natural'``).

    Returns
    -------
    str
        One of ``"sparse"`` / ``"medium"`` / ``"dense"``.
    """
    stats_map = getattr(bundle.split_manifest, "train_user_stats", None)
    if stats_map is None:
        return classify_user_group(0)
    entry = stats_map.get(int(partition_id))
    if entry is None:
        return classify_user_group(0)
    group = getattr(entry, "user_group", None)
    if group is not None:
        return group
    return classify_user_group(int(getattr(entry, "n_interactions", 0)))


# =============================================================================
# Training Function (PFR-05 single-user collapse + PFR-04 + PFR-07 + D-22).
# =============================================================================


@app.train()
def train(msg: Message, context: Context):
    """Train PFedRec on ONE user's local data (PFR-05 single-user collapse).

    Cross-device PFedRec flow:
      1. Resolve mode profile + log overrides (D-25).
      2. Identify partition_id == user_idx (cross-device).
      3. Load foundation bundle + build schema_v3 cache signature.
      4. Construct PFedRecMLP (Kaiming default per D-19) + load GLOBAL params
         (item embedding + ``affine_output.bias`` per D-01).
      5. D-22 probe-then-load LOCAL ``affine_output.weight`` from cache.
      6. Load partition data (cross-device: 1 user); PFR-05 assertion.
      7. Build per-user (items, ratings) BCE batch via prepare_user_train_data
         (FND-03 exclusion + FND-06 RNG threaded in).
      8. Train via train_pfedrec_single_user (Pitfall 3 dual-LR preserved).
      9. D-16 atomic save of LOCAL ``affine_output.weight``.
      10. Build wire payload — return GLOBAL params (item embedding +
          ``affine_output.bias``) + D-21 strict-contract FitMetricsContract.
    """
    # 1. Resolve mode profile + log overrides.
    mode = str(context.run_config.get("mode", "paper_compat_pfedrec"))
    profile = resolve_mode_defaults(mode)
    overrides = log_mode_and_overrides(mode, profile, dict(context.run_config))

    # 2. Per-client identity.
    partition_id = int(context.node_config["partition-id"])
    user_idx = partition_id  # cross-device: 1 user = 1 partition.
    num_partitions = int(context.node_config["num-partitions"])

    msg_config = msg.content.get("config") or ConfigRecord()
    round_num = int(msg_config.get("round_num", 1))
    run_seed = int(context.run_config.get("run-seed", 42))
    reuse_cache = bool(context.run_config.get("reuse-cache", False))
    # Server stamps run_id into msg_config (server_app.py); fall back to
    # run_config for backward compatibility, then "default" as last resort.
    run_id = str(
        msg_config.get(
            "run_id",
            context.run_config.get("run-id", context.run_config.get("run_id", "default")),
        )
    )

    # Hyperparams via D-25 mode resolver canonical-source pattern.
    latent_dim = int(context.run_config.get("latent-dim", profile.embedding_dim))
    num_train_negatives = int(
        context.run_config.get("num-negatives", profile.num_train_negatives)
    )
    batch_size = int(context.run_config.get("batch-size", 256))
    # C3 / D-06.7: the server's end-of-training calibration pass stamps
    # `local_epochs_override` into msg_config to force a fixed number of local
    # epochs for that pass. Normal training rounds never set it, so this falls
    # back to the run-config / mode-profile default and is unaffected.
    local_epochs = int(
        msg_config.get(
            "local_epochs_override",
            context.run_config.get("local-epochs", profile.local_epochs),
        )
    )
    lr = float(msg_config.get("lr", profile.lr))
    # C3 / D-06.7: read lr_eta from msg_config first so the calibration pass can
    # set lr_eta=0 (freeze the global item embedding; affine-only calibration).
    # Falls back to run-config for normal training rounds (which don't stamp it).
    lr_eta = float(msg_config.get("lr_eta", context.run_config.get("lr-eta", 80)))
    l2_reg = float(context.run_config.get("l2-regularization", 0.0))
    proximal_mu = float(msg_config.get("proximal_mu", 0.0))

    # 3. Load foundation bundle + build cache signature.
    bundle = _load_foundation_bundle()
    signature = _signature_fields(
        run_id=run_id,
        num_users=int(bundle.mapping.num_users),
        num_items=int(bundle.mapping.num_items),
        latent_dim=latent_dim,
        split_hash=str(bundle.split_hash),
        num_train_negatives=num_train_negatives,
    )

    # 4. Construct model + load GLOBAL params (item embedding + affine_output.bias).
    device = get_device()
    model = get_model(num_items=int(bundle.mapping.num_items), latent_dim=latent_dim)
    model.to(device)
    global_state = msg.content["arrays"].to_torch_state_dict()
    model.set_global_parameters(global_state)

    # 5. D-22 cold-round probe-then-load. Probe before load so the cold-round
    # path keeps the model's PyTorch nn.Linear default init (D-19).
    pt_path = _cache_dir_for_run(
        run_id=run_id, reuse_cache=reuse_cache, signature=signature
    ) / f"partition_{partition_id}.pt"
    cold_round = not pt_path.exists()
    if not cold_round:
        local_state = _load_local_user_state(
            partition_id=partition_id,
            run_id=run_id,
            reuse_cache=reuse_cache,
            signature=signature,
        )
        if local_state is not None:
            model.set_local_parameters(local_state, strict=True, run_id=run_id)
            print(f"  Client {partition_id}: loaded cached local state")
        else:
            cold_round = True
    if cold_round:
        print(f"  Client {partition_id}: cold start — using Kaiming default (D-19)")

    # 6. Load partition data + PFR-05 single-user assertion.
    trainloader, _testloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        partition_mode="natural",
    )
    user_ids_in_client = set()
    user_train_items: List[int] = []
    for batch in trainloader:
        users = batch["user"].cpu().numpy().tolist()
        items = batch["item"].cpu().numpy().tolist()
        user_ids_in_client.update(int(u) for u in users)
        user_train_items.extend(int(i) for i in items)
    num_users_in_client = len(user_ids_in_client)
    assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)

    # 7. Build BCE batch with FND-03 exclusion + FND-06 RNG (PFR-04 + PFR-07).
    exclude_items = bundle.exclusion.for_user(user_idx)
    items_list, ratings_list = prepare_user_train_data(
        user_idx=user_idx,
        user_train_items=user_train_items,
        num_items=int(bundle.mapping.num_items),
        num_negatives=num_train_negatives,
        run_seed=run_seed,
        round_num=round_num,
        exclude_items=exclude_items,
    )

    # 8. Train (Pitfall 3 dual-LR preserved inside train_pfedrec_single_user).
    train_loss = 0.0
    if items_list:
        train_loss = train_pfedrec_single_user(
            model=model,
            user_items=items_list,
            user_ratings=ratings_list,
            lr=lr,
            lr_eta=lr_eta,
            num_items=int(bundle.mapping.num_items),
            local_epochs=local_epochs,
            batch_size=batch_size,
            device=device,
            l2_regularization=l2_reg,
            proximal_mu=proximal_mu,
            global_item_embedding=None,
            run_seed=run_seed,
            user_idx=user_idx,
            round_num=round_num,
            exclude_items=exclude_items,
        )

    # 9. D-16 atomic save of LOCAL params (single-key payload per D-21).
    local_payload = {
        "affine_output.weight": model.affine_output.weight.data.detach().cpu().clone()
    }
    _save_local_user_state(
        partition_id=partition_id,
        state_dict=local_payload,
        run_id=run_id,
        reuse_cache=reuse_cache,
        signature=signature,
    )

    # 10. Build wire payload — return GLOBAL params (item embedding + bias).
    global_params_out = model.get_global_parameters()
    model_record = ArrayRecord(global_params_out)

    num_positives = int(len(user_train_items))
    num_training_examples = int(num_positives * (1 + num_train_negatives))
    fit_metrics = FitMetricsContract(
        train_loss=float(train_loss),
        num_positives=num_positives,
        num_training_examples=num_training_examples,
        round_num=round_num,
        partition_id=partition_id,
    ).to_dict()
    validate_fit_metrics(fit_metrics)

    metric_record = MetricRecord(fit_metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


# =============================================================================
# Evaluation Function (G-03-01 discover_only short-circuit + PFR-04 + PFR-05 + D-04).
# =============================================================================


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate this client's one user via sampled LOO + 99 negatives.

    Phase 5 Plan 03 contract:

    - **G-03-01 discover_only short-circuit**: FIRST check
      ``msg.content['config']['discover_only']``. If True, build a zero-
      suffstats ``EvaluateMetricsContract`` (with ``partition_id``) and
      return immediately — no model load, no bundle load, no data load.
    - **PFR-05** benchmark-mode single-user assertion.
    - **PFR-04 / FND-03**: ``ExclusionTable.for_user(user_idx)`` is folded
      into the eval-negative pool so the held-out test positive is never
      drawn as a sampled negative.
    - **PFR-06 / FND-06**: eval-negative draws via
      ``np_rng(run_seed, user_idx, round_num, "eval_neg")`` (in ``task.py``).
    - **D-04** BCE scope: per-user ``eval_loss`` over (positive + 99 negs)
      computed inside ``evaluate_pfedrec_sampled``; surfaced as the optional
      ``eval_loss`` field of the EvaluateMetricsContract.
    - **D-21** strict-contract wire payload (``validate_evaluate_metrics``).
    - **D-22** per-group sufficient-stat routing.
    """
    # G-03-01 discover_only short-circuit — FIRST check, NO heavy work.
    partition_id = int(context.node_config["partition-id"])
    config = msg.content.get("config") or ConfigRecord()
    if bool(config.get("discover_only", False)):
        payload = EvaluateMetricsContract(
            hit_count_overall_at10=0,
            ndcg_sum_overall_at10=0.0,
            evaluated_users=0,
            partition_id=partition_id,
        ).to_dict()
        validate_evaluate_metrics(payload)
        content = RecordDict({"metrics": MetricRecord(payload)})
        return Message(content=content, reply_to=msg)

    # 1. Resolve mode profile + log overrides.
    mode = str(context.run_config.get("mode", "paper_compat_pfedrec"))
    profile = resolve_mode_defaults(mode)
    overrides = log_mode_and_overrides(mode, profile, dict(context.run_config))
    primary = get_primary_evaluator(mode)
    assert primary == "sampled_loo_99", (
        f"PFR-06 invariant broken: get_primary_evaluator({mode!r}) returned "
        f"{primary!r}, expected 'sampled_loo_99'"
    )

    # 2. Per-client identity.
    user_idx = partition_id
    num_partitions = int(context.node_config["num-partitions"])
    round_num = int(config.get("round_num", 1))
    run_seed = int(context.run_config.get("run-seed", 42))
    reuse_cache = bool(context.run_config.get("reuse-cache", False))
    # Server stamps run_id into msg config (server_app.py); fall back to
    # run_config for backward compatibility, then "default" as last resort.
    run_id = str(
        config.get(
            "run_id",
            context.run_config.get("run-id", context.run_config.get("run_id", "default")),
        )
    )

    latent_dim = int(context.run_config.get("latent-dim", profile.embedding_dim))
    num_train_negatives = int(
        context.run_config.get("num-negatives", profile.num_train_negatives)
    )
    batch_size = int(context.run_config.get("batch-size", 256))
    num_eval_negatives = int(
        context.run_config.get("eval-num-negatives", profile.num_eval_negatives)
    )

    # 3. Foundation bundle + signature.
    bundle = _load_foundation_bundle()
    signature = _signature_fields(
        run_id=run_id,
        num_users=int(bundle.mapping.num_users),
        num_items=int(bundle.mapping.num_items),
        latent_dim=latent_dim,
        split_hash=str(bundle.split_hash),
        num_train_negatives=num_train_negatives,
    )

    # 4. Construct model + load GLOBAL params.
    device = get_device()
    model = get_model(num_items=int(bundle.mapping.num_items), latent_dim=latent_dim)
    model.to(device)
    global_state = msg.content["arrays"].to_torch_state_dict()
    model.set_global_parameters(global_state)

    # 5. D-22 probe-then-load LOCAL state.
    pt_path = _cache_dir_for_run(
        run_id=run_id, reuse_cache=reuse_cache, signature=signature
    ) / f"partition_{partition_id}.pt"
    if pt_path.exists():
        local_state = _load_local_user_state(
            partition_id=partition_id,
            run_id=run_id,
            reuse_cache=reuse_cache,
            signature=signature,
        )
        if local_state is not None:
            model.set_local_parameters(local_state, strict=True, run_id=run_id)

    # 6. Load partition data + PFR-05 single-user assertion.
    trainloader, testloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        partition_mode="natural",
    )
    user_ids_in_client: set = set()
    user_test_items: List[int] = []
    user_train_items: set = set()
    for batch in testloader:
        users = batch["user"].cpu().numpy().tolist()
        items = batch["item"].cpu().numpy().tolist()
        user_ids_in_client.update(int(u) for u in users)
        user_test_items.extend(int(i) for i in items)
    if not user_ids_in_client:
        for batch in trainloader:
            user_ids_in_client.update(int(u) for u in batch["user"].cpu().numpy().tolist())
    for batch in trainloader:
        items = batch["item"].cpu().numpy().tolist()
        user_train_items.update(int(i) for i in items)
    num_users_in_client = len(user_ids_in_client)
    assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)

    # 7. PFR-04 + PFR-06 + D-04 sampled eval (in task.py).
    exclude_items = bundle.exclusion.for_user(user_idx)
    sampled = evaluate_pfedrec_sampled(
        model=model,
        test_items=user_test_items,
        train_items_set=user_train_items,
        num_items=int(bundle.mapping.num_items),
        device=device,
        k_values=[10],
        num_negatives=num_eval_negatives,
        run_seed=run_seed,
        user_idx=user_idx,
        round_num=round_num,
        exclude_items=exclude_items,
    )

    # 8. D-22 per-group routing + D-21 strict-contract wire payload.
    user_group = _classify_partition_user_group(bundle, partition_id)
    sampled_num_users = int(sampled.get("sampled_num_users", 0))
    hr10_ratio = float(sampled.get("sampled_hr@10", 0.0))
    ndcg10_ratio = float(sampled.get("sampled_ndcg@10", 0.0))
    eval_loss = float(sampled.get("eval_loss", 0.0))
    hit10 = int(round(hr10_ratio * sampled_num_users))
    ndcg10_sum = float(ndcg10_ratio * sampled_num_users)
    evaluated_users = sampled_num_users

    per_group: Dict[str, Dict[str, float]] = {
        g: {"hit": 0, "ndcg": 0.0, "users": 0}
        for g in ("sparse", "medium", "dense")
    }
    per_group[user_group]["hit"] = hit10
    per_group[user_group]["ndcg"] = ndcg10_sum
    per_group[user_group]["users"] = evaluated_users

    eval_payload = EvaluateMetricsContract(
        hit_count_overall_at10=hit10,
        ndcg_sum_overall_at10=ndcg10_sum,
        evaluated_users=evaluated_users,
        eval_loss=eval_loss,
        sampled_hr_at10=hr10_ratio,
        sampled_ndcg_at10=ndcg10_ratio,
        hit_count_sparse_at10=int(per_group["sparse"]["hit"]),
        ndcg_sum_sparse_at10=float(per_group["sparse"]["ndcg"]),
        evaluated_users_sparse=int(per_group["sparse"]["users"]),
        hit_count_medium_at10=int(per_group["medium"]["hit"]),
        ndcg_sum_medium_at10=float(per_group["medium"]["ndcg"]),
        evaluated_users_medium=int(per_group["medium"]["users"]),
        hit_count_dense_at10=int(per_group["dense"]["hit"]),
        ndcg_sum_dense_at10=float(per_group["dense"]["ndcg"]),
        evaluated_users_dense=int(per_group["dense"]["users"]),
        partition_id=partition_id,
    ).to_dict()
    validate_evaluate_metrics(eval_payload)

    metric_record = MetricRecord(eval_payload)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
