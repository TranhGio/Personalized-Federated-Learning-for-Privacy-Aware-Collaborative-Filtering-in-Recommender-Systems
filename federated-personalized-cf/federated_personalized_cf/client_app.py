"""federated-personalized-cf: cross-device Flower client (Phase 3 Plan 03).

Phase 3 Plan 03 migrates the ``@app.train()`` / ``@app.evaluate()`` handlers to
the split-learning + single-row contract:

- Resolve a ModeProfile via
  ``fedrec_foundation.mode.resolve_mode_defaults(mode)``; collect visible
  overrides via ``log_mode_and_overrides``; assert 1-user-per-client under
  benchmark mode (PSN-02).
- Thread FND-06 RNG seeds (``run_seed`` / ``user_idx`` / ``round_num``) into
  ``task.train`` / ``task.evaluate_ranking_sampled`` (PSN-03 / BSL-05-style).
- Source the per-user exclusion set from FND-03
  (``ExclusionTable.for_user(user_idx)``) so the held-out test positive is
  NEVER drawn as a train or eval negative (PSN-03).
- Persist the LOCAL single-row user state (``local_user_row`` +
  ``local_user_bias``) to the D-04..D-10 manifest-sidecar embedding cache
  (PSN-05, PSN-06 disk shape):
    * ``.embedding_cache/{run_id}/manifest.json`` (6-field signature,
      ``schema_version=1``) written atomically via
      ``fedrec_foundation.atomic.atomic_write_json``.
    * ``.embedding_cache/{run_id}/partition_{pid}.pt`` (2-key
      ``OrderedDict({'local_user_row', 'local_user_bias'})``) written
      atomically via ``tempfile.mkstemp`` + ``os.replace``.
    * Signature mismatch on load raises ``RuntimeError`` with per-field
      delta + literal ``rm -rf`` hint (D-05 loud-mismatch).
    * ``reuse-cache=true`` (D-09) switches the path to
      ``.embedding_cache/sig_{sha256(fields)[:16]}/`` (run_id-agnostic).
- Build strict-contract wire payloads (D-21) via ``FitMetricsContract`` /
  ``EvaluateMetricsContract`` with the optional ``partition_id`` field
  populated (G-03-01 carry-forward). ``discover_only=True`` on an
  ``@app.evaluate()`` ConfigRecord short-circuits to a zero-suffstats
  payload without any model / data load (discovery-round handshake).

Pre-existing WIP from earlier uncommitted hunks — the ``get_device`` helper
and ``_device_cache`` module global — is preserved verbatim (D-18 surgical
edit discipline).
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
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

# Phase 3 Plan 03 foundation imports (D-04..D-10, PSN-02..06).
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
from fedrec_foundation.rng import np_rng
from fedrec_foundation.user_groups import classify_user_group

from federated_personalized_cf.dataset import _load_foundation_bundle
from federated_personalized_cf.task import get_model, load_data
from federated_personalized_cf.task import test as test_fn
from federated_personalized_cf.task import train as train_fn
from federated_personalized_cf.task import evaluate_ranking, evaluate_ranking_sampled


# Flower ClientApp
app = ClientApp()

# Cache for device detection (avoid repeated CUDA tests). D-18 preserved WIP.
_device_cache = None

# Module directory (kept for back-compat with the legacy path helpers).
_MODULE_DIR = Path(__file__).parent

# D-04 cache base dir — ``_save_local_user_state`` / ``_load_local_user_state``
# write into ``{_CACHE_BASE_DIR}/{run_id}/partition_{pid}.pt`` (or into
# ``{_CACHE_BASE_DIR}/sig_<hash>/partition_{pid}.pt`` when reuse_cache=True).
# Module-level for test-time monkeypatching.
_CACHE_BASE_DIR = _MODULE_DIR.parent / ".embedding_cache"


# =============================================================================
# Device detection (D-18 preserved from pre-existing WIP).
# =============================================================================

def get_device():
    """Get device with safe CUDA detection (handles incompatible GPU architectures)."""
    global _device_cache
    if _device_cache is not None:
        return _device_cache

    if torch.cuda.is_available():
        try:
            # Test if CUDA actually works by creating a small tensor.
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            _device_cache = torch.device("cuda:0")
        except RuntimeError:
            # CUDA available but not compatible (e.g., RTX 5090 with old PyTorch).
            _device_cache = torch.device("cpu")
    else:
        _device_cache = torch.device("cpu")

    return _device_cache


# =============================================================================
# D-04..D-10 manifest-sidecar embedding cache helpers.
# =============================================================================


def _signature_fields(
    *,
    run_id: str,
    method: str,
    num_users: int,
    num_items: int,
    dim: int,
    split_hash: str,
) -> Dict[str, Any]:
    """Build the 6-field (+ schema_version) signature dict for D-04 manifest.json.

    Parameters
    ----------
    run_id : str
        Flower run identifier (used to namespace the cache directory).
    method : str
        ``"bpr"`` or ``"basic"`` — the client-side model family.
    num_users : int
        Catalog user-population size (6040 for ML-1M).
    num_items : int
        Catalog item-population size (3706 for ML-1M).
    dim : int
        Embedding dimensionality — must match the runtime model.
    split_hash : str
        ``fedrec_foundation.split.SplitManifest.split_hash`` — guards against
        silent split drift between runs.

    Returns
    -------
    Dict[str, Any]
        Signature dict ready for ``atomic_write_json`` and for the D-05
        per-field mismatch comparison at load time.
    """
    return {
        "schema_version": 1,
        "run_id": str(run_id),
        "method": str(method),
        "num_users": int(num_users),
        "num_items": int(num_items),
        "dim": int(dim),
        "split_hash": str(split_hash),
    }


def _cache_dir_for_run(
    *,
    run_id: str,
    reuse_cache: bool,
    signature: Dict[str, Any],
) -> Path:
    """D-08/D-09 cache path resolver.

    ``reuse_cache=False`` (default): ``{_CACHE_BASE_DIR}/{run_id}/``.

    ``reuse_cache=True``: ``{_CACHE_BASE_DIR}/sig_<16-hex-chars>/`` — two
    runs with identical signature fields (ignoring ``run_id``) share the
    same cache dir silently per D-09.

    Parameters
    ----------
    run_id : str
        Used only under ``reuse_cache=False``.
    reuse_cache : bool
        Opt-in reuse flag (D-09).
    signature : Dict[str, Any]
        Output of ``_signature_fields``.

    Returns
    -------
    pathlib.Path
        Absolute (or ``_CACHE_BASE_DIR``-relative) cache directory. NOT
        created by this call — callers are expected to mkdir as needed.
    """
    base = Path(_CACHE_BASE_DIR)
    if not reuse_cache:
        return base / str(run_id)
    # D-09: drop run_id from the hash so runs with identical fields collide.
    payload = json.dumps(
        {k: v for k, v in signature.items() if k != "run_id"},
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
    """D-04 + D-06 + D-07 + D-10 atomic save of the single-row local state.

    Writes the ``.pt`` payload (single-row ``OrderedDict({'local_user_row',
    'local_user_bias'})``) atomically via ``tempfile.mkstemp`` +
    ``os.replace``, then writes/updates ``manifest.json`` via
    ``atomic_write_json``. A D-10 shape guard rejects any non-single-row
    state dict BEFORE any disk write happens.

    Parameters
    ----------
    partition_id : int
        Client's partition index (``user_idx`` under ``partition_mode="natural"``).
    state_dict : Dict[str, torch.Tensor]
        Must have keys ``{'local_user_row', 'local_user_bias'}`` exactly.
    run_id : str
        Flower run identifier.
    reuse_cache : bool
        D-09 opt-in flag.
    signature : Dict[str, Any]
        Output of ``_signature_fields``; also written verbatim to
        ``manifest.json``.

    Raises
    ------
    AssertionError
        When ``state_dict`` keys differ from the D-10 single-row contract.
    """
    cache_dir = _cache_dir_for_run(run_id=run_id, reuse_cache=reuse_cache, signature=signature)
    # D-10 shape guard: payload MUST be the single-row contract.
    assert set(state_dict.keys()) == {"local_user_row", "local_user_bias"}, (
        f"D-10 violated: local state has keys {sorted(state_dict.keys())}, "
        f"expected {{'local_user_row', 'local_user_bias'}}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    pt_path = cache_dir / f"partition_{int(partition_id)}.pt"

    # D-07 atomic write: tempfile + os.replace. The prefix must NOT start
    # with '.' because torch.save routes through a PyTorchFileWriter that
    # rejects filenames beginning with a dot (see torch/serialization.py).
    fd, tmp = tempfile.mkstemp(prefix="partition_tmp_", suffix=".pt", dir=str(cache_dir))
    os.close(fd)
    try:
        torch.save(OrderedDict(state_dict), tmp)
        os.replace(tmp, str(pt_path))
    except Exception:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
        raise

    # D-06 + D-07: manifest.json sidecar written atomically via foundation helper.
    atomic_write_json(str(cache_dir / "manifest.json"), signature)


def _load_local_user_state(
    *,
    partition_id: int,
    run_id: str,
    reuse_cache: bool,
    signature: Dict[str, Any],
) -> Optional[Dict[str, torch.Tensor]]:
    """D-04 + D-05 + D-10 load of the single-row local state with loud mismatch.

    Returns ``None`` if the cache directory or partition ``.pt`` does not
    exist (legitimate cold start — caller keeps the model's Xavier-uniform
    init per D-11).

    Raises ``RuntimeError`` with per-field delta + literal ``rm -rf`` hint
    when any field of the on-disk ``manifest.json`` diverges from the
    current run's signature (D-05).

    Parameters
    ----------
    partition_id : int
        Client's partition index.
    run_id : str
        Current run identifier.
    reuse_cache : bool
        D-09 opt-in flag.
    signature : Dict[str, Any]
        Current run's signature dict (``_signature_fields`` output).

    Returns
    -------
    Optional[Dict[str, torch.Tensor]]
        2-key state dict on hit; ``None`` on cold start.

    Raises
    ------
    RuntimeError
        On any signature-field mismatch against the on-disk manifest.
    AssertionError
        If the loaded payload does not satisfy the D-10 shape contract.
    """
    cache_dir = _cache_dir_for_run(run_id=run_id, reuse_cache=reuse_cache, signature=signature)
    pt_path = cache_dir / f"partition_{int(partition_id)}.pt"
    manifest_path = cache_dir / "manifest.json"
    if not pt_path.exists() or not manifest_path.exists():
        return None  # cold start

    with open(manifest_path, "r") as f:
        cached = json.load(f)

    # D-05: compare every signature field and raise with a per-field delta
    # on mismatch. Under reuse_cache, run_id is allowed to diverge because
    # the sig_<hash> dir is run_id-agnostic by construction.
    deltas: List[str] = []
    for key in ("schema_version", "run_id", "method", "num_users", "num_items", "dim", "split_hash"):
        if reuse_cache and key == "run_id":
            continue
        if cached.get(key) != signature.get(key):
            deltas.append(
                f"{key}: cached={cached.get(key)!r}, current={signature.get(key)!r}"
            )
    if deltas:
        raise RuntimeError(
            "Embedding-cache signature mismatch (D-05):\n  "
            + "\n  ".join(deltas)
            + f"\nRun: rm -rf {cache_dir}/ to reset, "
            f"or check --run-config for drifted keys."
        )

    state = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    # D-10 shape guard on load.
    assert set(state.keys()) == {"local_user_row", "local_user_bias"}, (
        f"D-10 violated on load: payload keys {sorted(state.keys())}"
    )
    return state


def _classify_partition_user_group(bundle: Dict[str, Any], partition_id: int) -> str:
    """Return the ``"sparse" | "medium" | "dense"`` label for this client's user.

    Reads from ``bundle["split_manifest"].train_user_stats[partition_id]
    .user_group`` when present (pre-computed by the foundation builder on
    TRAIN-only rows per CR-5). Falls back to
    ``classify_user_group(n_interactions=0)`` (i.e. ``"sparse"``) if the
    user_idx is not present — which only happens when the user has fewer
    than 2 ratings and the foundation split elided them.

    Parameters
    ----------
    bundle : Dict[str, Any]
        Output of ``_load_foundation_bundle()`` from ``dataset.py``.
    partition_id : int
        The client's user_idx (under ``partition_mode="natural"``).

    Returns
    -------
    str
        One of ``"sparse"`` / ``"medium"`` / ``"dense"``.
    """
    stats_map = getattr(bundle["split_manifest"], "train_user_stats", None)
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
# Training Function (Split Architecture, D-04..D-10 cache layout).
# =============================================================================

@app.train()
def train(msg: Message, context: Context):
    """Train the split-learning MF model on ONE user's local data (PSN-02..06).

    Split Learning Flow:
      1. Create model with Xavier-uniform init (D-11 first-use state).
      2. Load GLOBAL params from the server message (item embeddings + biases).
      3. Load LOCAL params from the D-04..D-10 manifest-sidecar cache
         (cold start -> keep the Xavier init; stale cache -> D-05
         RuntimeError with per-field delta).
      4. Train on local data (FND-06 RNG + FND-03 exclusion threaded in).
      5. Save LOCAL params to the manifest-sidecar cache (D-10 single-row
         shape guard applies; manifest.json refreshed atomically).
      6. Return ONLY GLOBAL params in the ArrayRecord; wire payload is a
         strict-contract ``FitMetricsContract.to_dict()`` validated via
         ``validate_fit_metrics`` before send (D-21) with optional
         ``partition_id`` populated (G-03-01 carry-forward).

    PSN-02 one-user assertion fires BEFORE any training under benchmark mode.
    """
    # Per-client identity pulled from Flower's node_config.
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    # Resolve the mode profile + log any visible overrides (D-06..D-11 + CR-2).
    mode = str(context.run_config.get("mode", "cross_silo_legacy"))
    profile = resolve_mode_defaults(mode)
    overrides = log_mode_and_overrides(mode, profile, dict(context.run_config))

    # Per-round metadata from the message config + run config.
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

    # Model configuration.
    model_type = str(context.run_config.get("model-type", "bpr"))
    embedding_dim = int(context.run_config.get("embedding-dim", profile.embedding_dim))
    dropout = float(context.run_config.get("dropout", 0.1))

    # Step 1: construct model (Xavier-uniform init per D-11 first-use).
    model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )

    # Step 2: load GLOBAL parameters from the server message.
    global_state = msg.content["arrays"].to_torch_state_dict()
    model.set_global_parameters(global_state)

    # Load the foundation bundle once so we can build the cache signature
    # (split_hash) and look up the per-user exclusion set + group label.
    bundle = _load_foundation_bundle()
    split_hash = str(getattr(bundle["split_manifest"], "split_hash", ""))
    num_users = int(getattr(bundle["mapping"], "num_users", 6040))
    num_items = int(getattr(bundle["mapping"], "num_items", 3706))

    signature = _signature_fields(
        run_id=run_id,
        method=model_type,
        num_users=num_users,
        num_items=num_items,
        dim=embedding_dim,
        split_hash=split_hash,
    )

    # Step 3: load LOCAL parameters from the D-04..D-10 cache if present.
    # On cold start (_load_local_user_state returns None) the model keeps
    # its Xavier-uniform init per D-11; no warm-start from server side.
    local_state = _load_local_user_state(
        partition_id=partition_id,
        run_id=run_id,
        reuse_cache=reuse_cache,
        signature=signature,
    )
    if local_state is not None:
        loaded, _missing = model.set_local_parameters(local_state, strict=False)
        print(f"  Client {partition_id}: loaded cached local state ({loaded})")
    else:
        print(f"  Client {partition_id}: cold start — using Xavier init (D-11)")

    device = get_device()
    model.to(device)

    # === FedProx: Save ONLY global parameters for proximal term (split-learning invariant). ===
    proximal_mu = float(msg_config.get("proximal_mu", 0.0))
    global_params_for_prox = None
    global_param_names = None
    if proximal_mu > 0:
        global_param_names = model.get_global_parameter_names()
        global_params_for_prox = []
        for name, p in model.named_parameters():
            if name in set(global_param_names):
                global_params_for_prox.append(p.detach().clone())

    # Load the partition's train data (natural partition = single user).
    alpha = float(context.run_config.get("alpha", 0.5))
    split_mode = str(context.run_config.get("eval-split-mode", "leave-one-out"))
    partition_mode = str(context.run_config.get("partition-mode", profile.partition_mode))
    trainloader, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        alpha=alpha,
        split_mode=split_mode,
        partition_mode=partition_mode,
    )

    # PSN-02: benchmark-mode single-user assertion. Under
    # ``partition_mode="natural"`` the partition IS a single user, so
    # ``num_users_in_client == 1`` whenever the trainloader is non-empty.
    user_ids_in_client = set()
    for batch in trainloader:
        user_ids_in_client.update(batch["user"].cpu().numpy().tolist())
    num_users_in_client = len(user_ids_in_client)
    assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)

    # FND-03 exclusion for this user (empty array if the user's row was elided).
    exclude_items = bundle["exclusion"].for_user(partition_id)

    # FND-06 RNG instance for training-negative sampling (BSL-05-style).
    train_rng = np_rng(run_seed, partition_id, round_num, "train_neg")

    # Step 4: Train the model — thread run_seed / user_idx / round_num /
    # exclude_items / rng into task.train so every stochastic step routes
    # through the seeded RNG (FND-06) and excludes the held-out positive.
    local_epochs = int(context.run_config.get("local-epochs", profile.local_epochs))
    lr = float(msg_config.get("lr", profile.lr))
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
        # Split-learning FedProx: proximal term applied ONLY to GLOBAL params.
        proximal_mu=proximal_mu,
        global_params=global_params_for_prox,
        global_param_names=global_param_names,
        # Phase 3 Plan 03 contract (PSN-03 / BSL-05-style).
        run_seed=run_seed,
        user_idx=partition_id,
        round_num=round_num,
        exclude_items=exclude_items,
        rng=train_rng,
    )

    # Step 5: save LOCAL parameters to the manifest-sidecar cache.
    # Model.get_local_parameters returns the 2-key OrderedDict; the save
    # helper applies the D-10 shape guard.
    local_params_out = model.get_local_parameters()
    _save_local_user_state(
        partition_id=partition_id,
        state_dict=dict(local_params_out),  # drop OrderedDict -> dict for the guard
        run_id=run_id,
        reuse_cache=reuse_cache,
        signature=signature,
    )

    # Step 6: wire payload is GLOBAL params only + strict-contract metrics.
    global_params_out = model.get_global_parameters()
    model_record = ArrayRecord(global_params_out)

    # D-21 strict-contract wire payload. num_positives = count of rating
    # rows; num_training_examples = positives + sampled negatives per
    # positive per local epoch. G-03-01: echo partition_id.
    num_positives = int(len(trainloader.dataset))
    num_training_examples = int(num_positives * (1 + max(num_train_negatives, 0)))
    fit_metrics = FitMetricsContract(
        train_loss=float(train_loss),
        num_positives=num_positives,
        num_training_examples=num_training_examples,
        round_num=round_num,
        partition_id=partition_id,
    ).to_dict()
    # Defense-in-depth: validate before sending to catch contract drift.
    validate_fit_metrics(fit_metrics)

    metric_record = MetricRecord(fit_metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


# =============================================================================
# Evaluation Function (Split Architecture, G-03-01 discovery handshake).
# =============================================================================

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate this client's one user on the held-out positive.

    Phase 3 Plan 03 contract hooks:

    - **G-03-01 handshake**: FIRST check ``msg.content["config"]`` for
      ``discover_only=True``. If true, short-circuit with a zero-
      suffstats ``EvaluateMetricsContract`` payload (partition_id
      echoed) and return immediately — no model load, no data load,
      no evaluation. The server uses this to build
      ``partition_id -> node_id`` before round 1 so the per-round
      sampler can work in partition-id space.
    - **PSN-02**: asserts ``num_users_in_client == 1`` under benchmark
      mode BEFORE any evaluation happens.
    - **PSN-03**: evaluator excludes the FND-03 exclusion set from the
      sampled-negative pool so the held-out test positive is never drawn.
    - **PSN-04 / BSL-05-style**: evaluator RNG seeded via
      ``np_rng(run_seed, user_idx, round_num, "eval_neg")``.
    - **PSN-04 / BSL-07-style**: primary evaluator selected via
      ``get_primary_evaluator(mode)``. ``allrank_*`` metrics (when
      ``enable-ranking-eval=true``) stay namespaced and are NOT consumed
      by the server's thesis-table metrics.
    - **D-21**: return payload is ``EvaluateMetricsContract.to_dict()``
      validated via ``validate_evaluate_metrics`` before send — rejects
      free-form extras so the server aggregator only sees known fields.
      G-03-01: ``partition_id`` populated on every contract build.
    - **D-22**: per-group (sparse/medium/dense) sufficient-stat fields
      populated; the client's user-group receives the non-zero values,
      the other two groups receive zeros.
    - **D-04..D-10**: local user state loaded from the manifest-sidecar
      cache (cold start => Xavier init carried from ``get_model``,
      matches D-11).
    """
    # G-03-01 discovery-round short-circuit: server uses this to build
    # partition_id -> node_id mapping before round 1 so the per-round
    # sampler can work in partition-id space (stable 0..N-1) instead of
    # Flower's ephemeral node_id space (os.urandom, not seedable).
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

    # Resolve the mode profile + log any visible overrides.
    mode = str(context.run_config.get("mode", "cross_silo_legacy"))
    profile = resolve_mode_defaults(mode)
    overrides = log_mode_and_overrides(mode, profile, dict(context.run_config))

    # Per-round metadata.
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

    # Model configuration.
    model_type = str(context.run_config.get("model-type", "bpr"))
    embedding_dim = int(context.run_config.get("embedding-dim", profile.embedding_dim))
    dropout = float(context.run_config.get("dropout", 0.1))

    # Step 1: construct model with Xavier init (D-11 first-use state).
    model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )

    # Step 2: load GLOBAL parameters from the server message.
    global_state = msg.content["arrays"].to_torch_state_dict()
    model.set_global_parameters(global_state)

    # Foundation bundle for cache signature + exclusion + user-group lookup.
    bundle = _load_foundation_bundle()
    split_hash = str(getattr(bundle["split_manifest"], "split_hash", ""))
    num_users = int(getattr(bundle["mapping"], "num_users", 6040))
    num_items = int(getattr(bundle["mapping"], "num_items", 3706))

    signature = _signature_fields(
        run_id=run_id,
        method=model_type,
        num_users=num_users,
        num_items=num_items,
        dim=embedding_dim,
        split_hash=split_hash,
    )

    # Step 3: load LOCAL parameters from the cache if present. D-05 loud
    # mismatch fires before we touch the dataset.
    local_state = _load_local_user_state(
        partition_id=partition_id,
        run_id=run_id,
        reuse_cache=reuse_cache,
        signature=signature,
    )
    if local_state is not None:
        model.set_local_parameters(local_state, strict=False)
    # (Cold start => evaluation happens against the Xavier-init local row;
    # this should only occur before the first fit round completes.)

    device = get_device()
    model.to(device)

    # Load the partition data (train + test for the evaluator; train needed
    # by evaluate_ranking_sampled to exclude observed items from the
    # negative-candidate pool).
    alpha = float(context.run_config.get("alpha", 0.5))
    split_mode = str(context.run_config.get("eval-split-mode", "leave-one-out"))
    partition_mode = str(context.run_config.get("partition-mode", profile.partition_mode))
    trainloader, testloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        alpha=alpha,
        split_mode=split_mode,
        partition_mode=partition_mode,
    )

    # PSN-02: same one-user lock as @app.train. Pull distinct user ids from
    # the testloader first; fall back to trainloader if the held-out test
    # set is empty (single-interaction users elided by foundation split).
    user_ids_in_client = set()
    for batch in testloader:
        user_ids_in_client.update(batch["user"].cpu().numpy().tolist())
    if not user_ids_in_client:
        for batch in trainloader:
            user_ids_in_client.update(batch["user"].cpu().numpy().tolist())
    num_users_in_client = len(user_ids_in_client)
    assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)

    # PSN-04 / BSL-07-style: only the primary evaluator feeds thesis-table metrics.
    primary = get_primary_evaluator(mode)
    assert primary == "sampled_loo_99", (
        f"BSL-07 invariant broken: get_primary_evaluator({mode!r}) returned "
        f"{primary!r}, expected 'sampled_loo_99'"
    )

    # FND-03 exclusion for this user.
    exclude_items = bundle["exclusion"].for_user(partition_id)

    # Rating-prediction diagnostics (RMSE / MAE) — NOT consumed by the
    # thesis-table aggregator; cached as optional fields in the evaluate
    # contract for per-round logging only.
    eval_loss, _rating_metrics = test_fn(
        model=model,
        testloader=testloader,
        device=str(device),
        model_type=model_type,
    )

    # PSN-03 / BSL-05-style primary-path evaluation.
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
    # does not mix it with the primary ``sampled_*`` fields. The return
    # value is intentionally dropped; the side effect is populating the
    # item-popularity cache for potential server-side logging.
    enable_ranking_eval = bool(context.run_config.get("enable-ranking-eval", False))
    if enable_ranking_eval:
        k_values_str = str(context.run_config.get("ranking-k-values", "5,10,20"))
        k_values = [int(k.strip()) for k in k_values_str.split(",")]
        _ = evaluate_ranking(
            model=model,
            testloader=testloader,
            device=str(device),
            k_values=k_values,
            trainloader=trainloader,
        )

    # D-22: per-group sufficient-stat routing. In benchmark mode the
    # partition is one user, so the stats for that user's group carry the
    # non-zero values; the other two groups carry zeros.
    user_group = _classify_partition_user_group(bundle, partition_id)
    sampled_num_users = int(sampled_metrics.get("sampled_num_users", 0))
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
    # G-03-01: partition_id echoed for audit-trail consistency.
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
        partition_id=partition_id,
    ).to_dict()
    # Defense-in-depth: reject free-form extras before sending (D-21).
    validate_evaluate_metrics(eval_payload)

    metric_record = MetricRecord(eval_payload)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
