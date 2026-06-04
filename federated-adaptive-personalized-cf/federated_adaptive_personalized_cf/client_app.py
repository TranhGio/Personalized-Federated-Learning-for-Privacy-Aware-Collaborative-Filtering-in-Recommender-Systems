"""federated-adaptive-personalized-cf: cross-device Flower client (Phase 4 Plan 03).

Phase 4 Plan 03 migrates the ``@app.train()`` / ``@app.evaluate()`` handlers
to the split-learning + adaptive-personalization contract with the ADP-02
enable-before-load ordering fix as the primary change:

- **ADP-02** (primary bug fix): under ``mode="benchmark_cross_device"`` the
  calls to ``enable_per_user_alpha`` AND ``enable_item_perturbation`` are
  made UNCONDITIONALLY and BEFORE ``_load_local_user_state`` so the
  cached ``_logit_alpha.weight`` + ``_item_perturbation.weight`` tensors
  are in ``_LOCAL_PARAMS`` at load time. Pre-Phase-4 code called
  ``enable_*`` AFTER the load, which silently re-initialized those
  LOCAL tensors from the heuristic every round (CONCERNS.md §enable-after-load).
- **ADP-04**: benchmark-mode single-user assertion fires BEFORE any
  training or evaluation.
- **ADP-05 + ADP-06**: FND-06 RNG + FND-03 exclusion threaded through
  ``task.train`` / ``task.evaluate_ranking_sampled``.
- **D-01..D-04 schema_version=2 manifest-sidecar cache**: single atomic
  ``.pt`` blob per partition with ALL Phase-4 LOCAL keys (base +
  PersonalMLP + fusion + logit_alpha + item_perturbation); sibling
  ``manifest.json`` carries the 12-field fingerprint (6 Phase-3 + 6
  Phase-4: alpha_method, fusion_type, mlp_hidden_dims,
  per_user_alpha_enabled, item_perturbation_enabled, contrastive_lambda).
  Any mismatch on load raises RuntimeError with the per-field delta and
  a literal ``rm -rf`` hint; Phase-3 v1 caches are also rejected.
- **D-13 + D-14**: cold-round signal passed to
  ``task.train_dual_personalized`` so α=0 + contrastive-skip applies on
  the first round for any partition.
- **D-16 alpha diagnostics**: after training, 6 scalar diagnostics
  (``alpha_mean``, ``alpha_std``, ``alpha_p25``, ``alpha_p50``,
  ``alpha_p75``, ``alpha_clip_hit_rate``) are sent back in a sibling
  ``MetricRecord`` keyed ``"alpha_diagnostics"`` because
  ``FitMetricsContract`` is strict (D-21) and rejects free-form extras.
- **D-21 + G-03-01**: strict-contract wire payloads with
  ``partition_id`` populated; discover_only short-circuit mirrors Phase
  3 for the server-side partition-id sampler bootstrap.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
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
from fedrec_foundation.rng import np_rng
from fedrec_foundation.user_groups import classify_user_group

from federated_adaptive_personalized_cf.dataset import _load_foundation_bundle
from federated_adaptive_personalized_cf.models import (
    AlphaConfig,
    HierarchicalConditionalAlphaConfig,
)
from federated_adaptive_personalized_cf.strategy import USER_PROTOTYPE_KEY
from federated_adaptive_personalized_cf.task import (
    compute_client_alpha,
    compute_per_user_alpha,
    get_model,
    load_data,
)
from federated_adaptive_personalized_cf.task import evaluate_ranking, evaluate_ranking_sampled
from federated_adaptive_personalized_cf.task import test as test_fn
from federated_adaptive_personalized_cf.task import train as train_fn


app = ClientApp()

# Cache for device detection (avoid repeated CUDA tests). D-18 preserved WIP.
_device_cache = None

# Module directory (kept for back-compat with legacy path helpers).
_MODULE_DIR = Path(__file__).parent

# D-01 cache base dir — the Phase-4 schema_version=2 manifest-sidecar helpers
# write into ``{_CACHE_BASE_DIR}/{run_id}/partition_{pid}.pt`` (default) or
# ``{_CACHE_BASE_DIR}/sig_<hash>/partition_{pid}.pt`` under ``reuse_cache=True``.
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
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            _device_cache = torch.device("cuda:0")
        except RuntimeError:
            _device_cache = torch.device("cpu")
    else:
        _device_cache = torch.device("cpu")

    return _device_cache


# =============================================================================
# D-01..D-04 schema_version=2 manifest-sidecar embedding cache helpers.
# =============================================================================


def _signature_fields_v2(
    *,
    run_id: str,
    method: str,
    num_users: int,
    num_items: int,
    dim: int,
    split_hash: str,
    alpha_method: str,
    fusion_type: str,
    mlp_hidden_dims: str,
    per_user_alpha_enabled: bool,
    item_perturbation_enabled: bool,
    contrastive_lambda: float,
) -> Dict[str, Any]:
    """Build the Phase-4 schema_version=2 signature (12 fields: 6 Phase-3 + 6 Phase-4).

    Parameters
    ----------
    run_id : str
        Flower run identifier (used to namespace the cache directory).
    method : str
        Model family name — typically ``"dual"`` for the adaptive module.
    num_users : int
        Catalog user-population size (6040 for ML-1M).
    num_items : int
        Catalog item-population size (3706 for ML-1M).
    dim : int
        Embedding dimensionality.
    split_hash : str
        ``fedrec_foundation.split.SplitManifest.split_hash``.
    alpha_method : str
        ``"hierarchical_conditional"`` / ``"multi_factor"`` / ``"data_quantity"``.
    fusion_type : str
        ``"add"`` / ``"gate"`` / ``"concat"``.
    mlp_hidden_dims : str
        Comma-joined string of hidden-layer dims (e.g. ``"512,256,128"``).
    per_user_alpha_enabled : bool
        Whether ``_logit_alpha.weight`` is part of the LOCAL key set.
    item_perturbation_enabled : bool
        Whether ``_item_perturbation.weight`` is part of the LOCAL key set.
    contrastive_lambda : float
        Weight on the InfoNCE auxiliary loss.

    Returns
    -------
    Dict[str, Any]
        Signature dict ready for ``atomic_write_json`` and the D-04 load
        comparison.
    """
    return {
        "schema_version": 2,
        "run_id": str(run_id),
        "method": str(method),
        "num_users": int(num_users),
        "num_items": int(num_items),
        "dim": int(dim),
        "split_hash": str(split_hash),
        "alpha_method": str(alpha_method),
        "fusion_type": str(fusion_type),
        "mlp_hidden_dims": str(mlp_hidden_dims),
        "per_user_alpha_enabled": bool(per_user_alpha_enabled),
        "item_perturbation_enabled": bool(item_perturbation_enabled),
        "contrastive_lambda": float(contrastive_lambda),
    }


def _cache_dir_for_run(
    *,
    run_id: str,
    reuse_cache: bool,
    signature: Dict[str, Any],
) -> Path:
    """D-01 / D-09 cache path resolver.

    ``reuse_cache=False`` (default): ``{_CACHE_BASE_DIR}/{run_id}/``.

    ``reuse_cache=True``: ``{_CACHE_BASE_DIR}/sig_<16-hex-chars>/``. Two
    runs with identical signature fields (ignoring ``run_id``) share the
    same cache dir silently.
    """
    base = Path(_CACHE_BASE_DIR)
    if not reuse_cache:
        return base / str(run_id)
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
    """D-01 + D-02 atomic save of the Phase-4 extended LOCAL state.

    Writes the ``.pt`` payload (all LOCAL keys in one blob: base +
    PersonalMLP + fusion + optional ``_logit_alpha`` + optional
    ``_item_perturbation``) atomically via ``tempfile.mkstemp`` +
    ``os.replace``, then writes/updates ``manifest.json`` via
    ``atomic_write_json``. A shape guard rejects any state dict missing
    the required keys BEFORE any disk write happens.
    """
    cache_dir = _cache_dir_for_run(run_id=run_id, reuse_cache=reuse_cache, signature=signature)
    # Shape guard: required LOCAL keys at minimum are user_embeddings.weight
    # (always) + _logit_alpha.weight (iff per_user_alpha_enabled) +
    # _item_perturbation.weight (iff item_perturbation_enabled).
    required = {"user_embeddings.weight"}
    if signature.get("per_user_alpha_enabled"):
        required.add("_logit_alpha.weight")
    if signature.get("item_perturbation_enabled"):
        required.add("_item_perturbation.weight")
    missing = required - set(state_dict.keys())
    assert not missing, (
        f"D-01/D-03 violated: LOCAL state missing required keys {missing}. "
        f"Got: {sorted(state_dict.keys())}"
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    pt_path = cache_dir / f"partition_{int(partition_id)}.pt"

    # Atomic write via tempfile + os.replace. Prefix MUST NOT start with
    # '.' — torch.save's PyTorchFileWriter rejects dot-prefixed temp names.
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

    atomic_write_json(str(cache_dir / "manifest.json"), signature)


def _load_local_user_state(
    *,
    partition_id: int,
    run_id: str,
    reuse_cache: bool,
    signature: Dict[str, Any],
) -> Optional[Dict[str, torch.Tensor]]:
    """D-04 schema_version=2 load with loud mismatch.

    Returns ``None`` on cold start (cache dir or partition .pt missing).
    Raises ``RuntimeError`` with per-field delta + literal
    ``rm -rf {cache_dir}/`` hint when any signature field (including
    ``schema_version``) diverges from the on-disk manifest. Phase-3
    schema_version=1 caches trigger the same loud failure — no
    auto-migration, no silent cold-start.
    """
    cache_dir = _cache_dir_for_run(run_id=run_id, reuse_cache=reuse_cache, signature=signature)
    pt_path = cache_dir / f"partition_{int(partition_id)}.pt"
    manifest_path = cache_dir / "manifest.json"
    if not pt_path.exists() or not manifest_path.exists():
        return None  # cold start

    with open(manifest_path, "r") as f:
        cached = json.load(f)

    all_keys = (
        "schema_version",
        "run_id",
        "method",
        "num_users",
        "num_items",
        "dim",
        "split_hash",
        "alpha_method",
        "fusion_type",
        "mlp_hidden_dims",
        "per_user_alpha_enabled",
        "item_perturbation_enabled",
        "contrastive_lambda",
    )
    deltas: List[str] = []
    for key in all_keys:
        if reuse_cache and key == "run_id":
            continue
        if cached.get(key) != signature.get(key):
            deltas.append(
                f"  {key}: cached={cached.get(key)!r}, current={signature.get(key)!r}"
            )
    if deltas:
        raise RuntimeError(
            "Embedding-cache signature mismatch (D-04, schema_version=2):\n"
            + "\n".join(deltas)
            + f"\nRun: rm -rf {cache_dir}/ to reset, "
            f"or check --run-config for drifted keys."
        )

    state = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    return state


def _classify_partition_user_group(bundle: Dict[str, Any], partition_id: int) -> str:
    """Return the ``"sparse" | "medium" | "dense"`` label for this client's user."""
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


def _compute_alpha_diagnostics(model) -> Optional[Dict[str, float]]:
    """D-16: compute 6 scalar diagnostics over per-user alpha sigmoid outputs.

    Returns ``None`` when per-user alpha is not enabled on the model.
    Keys: ``alpha_mean``, ``alpha_std``, ``alpha_p25``, ``alpha_p50``,
    ``alpha_p75``, ``alpha_clip_hit_rate``. The clip-hit-rate is the
    fraction of users whose refined alpha is within 1e-4 of the clip
    floor (0.1) or ceiling (0.95) — CONCERNS.md clip-floor diagnostic.
    """
    if not getattr(model, "_per_user_alpha_enabled", False):
        return None
    logit_alpha = getattr(model, "_logit_alpha", None)
    if logit_alpha is None:
        return None
    with torch.no_grad():
        alphas = torch.sigmoid(logit_alpha.weight).flatten().cpu().numpy()
    if len(alphas) == 0:
        return None
    min_alpha, max_alpha = 0.1, 0.95
    epsilon = 1e-4
    clip_hits = int(
        np.sum(
            (np.abs(alphas - min_alpha) < epsilon)
            | (np.abs(alphas - max_alpha) < epsilon)
        )
    )
    return {
        "alpha_mean": float(np.mean(alphas)),
        "alpha_std": float(np.std(alphas)),
        "alpha_p25": float(np.percentile(alphas, 25)),
        "alpha_p50": float(np.percentile(alphas, 50)),
        "alpha_p75": float(np.percentile(alphas, 75)),
        "alpha_clip_hit_rate": float(clip_hits / len(alphas)),
    }


def _build_alpha_configs(context: Context):
    """Factor out the AlphaConfig + HC-config construction shared by both handlers."""
    alpha_method = context.run_config.get("alpha-method", "hierarchical_conditional")

    factor_weights = {
        "quantity": context.run_config.get("alpha-weight-quantity", 0.40),
        "diversity": context.run_config.get("alpha-weight-diversity", 0.25),
        "coverage": context.run_config.get("alpha-weight-coverage", 0.20),
        "consistency": context.run_config.get("alpha-weight-consistency", 0.15),
    }

    alpha_config = AlphaConfig(
        method=alpha_method,
        min_alpha=context.run_config.get("alpha-min", 0.1),
        max_alpha=context.run_config.get("alpha-max", 0.95),
        quantity_threshold=context.run_config.get("alpha-quantity-threshold", 100),
        quantity_temperature=context.run_config.get("alpha-quantity-temperature", 0.05),
        factor_weights=factor_weights,
        max_entropy=context.run_config.get("alpha-max-entropy", 3.0),
        coverage_threshold=context.run_config.get("alpha-coverage-threshold", 100),
        max_rating_std=context.run_config.get("alpha-max-rating-std", 1.5),
    )

    hc_config = None
    if alpha_method == "hierarchical_conditional":
        hc_config = HierarchicalConditionalAlphaConfig(
            min_alpha=context.run_config.get("alpha-min", 0.1),
            max_alpha=context.run_config.get("alpha-max", 0.95),
            data_volume_weight=context.run_config.get("alpha-hc-data-volume-weight", 0.55),
            preference_quality_weight=context.run_config.get("alpha-hc-preference-weight", 0.45),
            quantity_threshold=context.run_config.get("alpha-quantity-threshold", 100),
            quantity_temperature=context.run_config.get("alpha-quantity-temperature", 0.05),
            max_entropy=context.run_config.get("alpha-max-entropy", 3.0),
            coverage_threshold=context.run_config.get("alpha-coverage-threshold", 100),
            max_rating_std=context.run_config.get("alpha-max-rating-std", 1.5),
            sparse_threshold=context.run_config.get("alpha-hc-sparse-threshold", 20),
            sparse_penalty_max=context.run_config.get("alpha-hc-sparse-penalty-max", 0.5),
            niche_diversity_threshold=context.run_config.get("alpha-hc-niche-diversity-threshold", 0.25),
            niche_quantity_threshold=context.run_config.get("alpha-hc-niche-quantity-threshold", 0.6),
            niche_bonus=context.run_config.get("alpha-hc-niche-bonus", 0.15),
            inconsistent_threshold=context.run_config.get("alpha-hc-inconsistent-threshold", 0.3),
            inconsistent_penalty=context.run_config.get("alpha-hc-inconsistent-penalty", 0.3),
            completionist_coverage=context.run_config.get("alpha-hc-completionist-coverage", 0.7),
            completionist_diversity=context.run_config.get("alpha-hc-completionist-diversity", 0.3),
            completionist_bonus=context.run_config.get("alpha-hc-completionist-bonus", 0.1),
        )

    return alpha_method, alpha_config, hc_config


def _resolve_enable_flags(
    context: Context, is_benchmark_mode: bool
) -> Tuple[bool, bool]:
    """Resolve effective enable flags honoring D-03 (unconditional in benchmark mode).

    Under benchmark mode, per-user alpha defaults ON regardless of its absence
    in ``run_config``. Run-config values are consulted as **ablation-only
    overrides** — an explicit ``enable-per-user-alpha=false`` in
    ``--run-config`` DOES disable the feature for a specific sweep cell.
    Outside benchmark mode, the run-config value is honored verbatim with the
    pre-Phase-4 default of ``False``.

    BUG 1 FIX (2026-06-01): ``item-perturbation`` now defaults OFF even in
    benchmark mode. Empirically (factorial decomposition 2026-05-29..06-01)
    every cross-device run with item-perturbation ON collapses in-loop NDCG@10
    to ~0.07 while every IP-off run learns to ~0.24. Root cause: the
    ``(num_items, dim)`` zero-init perturbation is LOCAL (never aggregated);
    under cross-device each client rates ~1-2% of items, so the vast majority
    of perturbation rows stay zero and the mixed zero/trained rows become
    incoherent noise that overwhelms the global item signal. The technique is
    fundamentally mismatched to cross-device. It remains available as an
    explicit ablation via ``--run-config "enable-item-perturbation=true"`` so
    the failure mode can still be reproduced for the thesis ablation table.
    """
    raw_per_user = context.run_config.get("enable-per-user-alpha")
    raw_item_perturb = context.run_config.get("enable-item-perturbation")

    if is_benchmark_mode:
        # D-03: per-user-alpha unconditionally ON unless ablation flips OFF.
        effective_per_user_alpha = True if raw_per_user is None else bool(raw_per_user)
        # BUG 1 FIX: item-perturbation defaults OFF (was ON). Only an explicit
        # enable-item-perturbation=true turns it on (ablation reproduction).
        effective_item_perturbation = bool(raw_item_perturb) if raw_item_perturb is not None else False
    else:
        effective_per_user_alpha = bool(raw_per_user) if raw_per_user is not None else False
        effective_item_perturbation = bool(raw_item_perturb) if raw_item_perturb is not None else False

    return effective_per_user_alpha, effective_item_perturbation


def _apply_enable_before_load(
    model,
    *,
    effective_per_user_alpha: bool,
    effective_item_perturbation: bool,
    per_user_alphas: Dict[int, float],
    item_perturb_reg: float,
) -> None:
    """ADP-02 enable-before-load ordering fix — primary Phase-4 bug fix.

    This helper is called BEFORE ``_load_local_user_state`` so the
    ``_logit_alpha.weight`` and ``_item_perturbation.weight`` tensors
    appear in ``model._LOCAL_PARAMS`` at load time. Under the
    pre-Phase-4 code these calls ran AFTER the load, silently
    re-initializing the LOCAL tensors from the heuristic every round
    (CONCERNS.md §enable-after-load).
    """
    if effective_per_user_alpha and hasattr(model, "enable_per_user_alpha"):
        # enable_per_user_alpha requires init_alphas dict keyed by user_id;
        # compute_per_user_alpha produces exactly that shape.
        model.enable_per_user_alpha(
            num_users=model.num_users,
            init_alphas=per_user_alphas,
        )

    if effective_item_perturbation and hasattr(model, "enable_item_perturbation"):
        model.enable_item_perturbation(reg_lambda=item_perturb_reg)


# =============================================================================
# Training Function (Split Architecture, ADP-02 enable-before-load, D-13 cold-round).
# =============================================================================


@app.train()
def train(msg: Message, context: Context):
    """Train the adaptive MF model on ONE user's local data (Phase 4 cross-device).

    Split Learning Flow (ADP-02 ordering):
      1. Construct model with Xavier init (via get_model).
      2. Load GLOBAL params from the server message.
      3. Load partition data + compute user stats / alphas.
      4. ADP-04: assert one user per client under benchmark mode.
      5. **ADP-02**: call ``enable_per_user_alpha`` +
         ``enable_item_perturbation`` UNCONDITIONALLY before the cache
         load so the Phase-4 LOCAL keys are in ``_LOCAL_PARAMS``.
      6. Load LOCAL params from the schema_version=2 manifest-sidecar
         cache. On mismatch, D-04 raises RuntimeError with rm -rf hint.
      7. Set adaptive alpha + global prototype (legacy path for the
         scalar-alpha fallback; per-user alpha is refined by gradients).
      8. Train with FND-06 RNG + FND-03 exclusion + D-13/D-14
         cold-round signal.
      9. Compute D-16 alpha diagnostics, persist LOCAL state, emit
         strict FitMetricsContract + sidecar alpha_diagnostics +
         user_prototype metrics.
    """
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    # Mode resolver (D-06..D-11 + CR-2).
    mode = str(context.run_config.get("mode", "cross_silo_legacy"))
    profile = resolve_mode_defaults(mode)
    overrides = log_mode_and_overrides(mode, profile, dict(context.run_config))
    is_benchmark_mode = (mode == "benchmark_cross_device")

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
    model_type = str(context.run_config.get("model-type", "dual"))
    embedding_dim = int(context.run_config.get("embedding-dim", 128))
    dropout = float(context.run_config.get("dropout", 0.1))

    mlp_hidden_dims: Optional[List[int]] = None
    fusion_type = str(context.run_config.get("fusion-type", "concat"))
    mlp_dims_str = str(context.run_config.get("mlp-hidden-dims", "512,256,128"))
    if model_type == "dual":
        mlp_hidden_dims = [int(d.strip()) for d in mlp_dims_str.split(",")]

    contrastive_lambda = float(context.run_config.get("contrastive-lambda", 0.1))
    contrastive_tau = float(context.run_config.get("contrastive-tau", 0.1))

    # --- Step 1: construct model with Xavier-uniform init ---
    model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
        mlp_hidden_dims=mlp_hidden_dims,
        fusion_type=fusion_type,
    )

    # --- Step 2: load GLOBAL parameters from the server message ---
    global_state = msg.content["arrays"].to_torch_state_dict()
    model.set_global_parameters(global_state)

    # Load the foundation bundle for cache signature + exclusion.
    bundle = _load_foundation_bundle()
    split_hash = str(getattr(bundle["split_manifest"], "split_hash", ""))
    num_users = int(getattr(bundle["mapping"], "num_users", 6040))
    num_items = int(getattr(bundle["mapping"], "num_items", 3706))

    # --- Step 3: load partition data + compute alphas ---
    dirichlet_alpha = float(context.run_config.get("alpha", 0.5))
    split_mode = str(context.run_config.get("eval-split-mode", "leave-one-out"))
    partition_mode = str(context.run_config.get("partition-mode", "natural"))
    trainloader, _testloader, user_stats = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        alpha=dirichlet_alpha,
        compute_stats=True,
        split_mode=split_mode,
        partition_mode=partition_mode,
    )

    # --- Step 4: ADP-04 one-user assertion ---
    user_ids_in_client = set()
    for batch in trainloader:
        user_ids_in_client.update(batch["user"].cpu().numpy().tolist())
    num_users_in_client = len(user_ids_in_client)
    assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)

    alpha_method, alpha_config, hc_config = _build_alpha_configs(context)
    client_alpha = compute_client_alpha(user_stats, alpha_config, hc_config)

    # --- Step 5: ADP-02 enable-before-load (the primary Phase-4 fix) ---
    effective_per_user_alpha, effective_item_perturbation = _resolve_enable_flags(
        context, is_benchmark_mode
    )
    per_user_alphas = compute_per_user_alpha(user_stats, alpha_config, hc_config) if effective_per_user_alpha else {}
    item_perturb_reg = float(context.run_config.get("item-perturbation-reg", 0.01))
    _apply_enable_before_load(
        model,
        effective_per_user_alpha=effective_per_user_alpha,
        effective_item_perturbation=effective_item_perturbation,
        per_user_alphas=per_user_alphas,
        item_perturb_reg=item_perturb_reg,
    )

    # Build the schema_version=2 signature now that the enable flags are
    # resolved — the cached tensor keyspace depends on them.
    signature = _signature_fields_v2(
        run_id=run_id,
        method=model_type,
        num_users=num_users,
        num_items=num_items,
        dim=embedding_dim,
        split_hash=split_hash,
        alpha_method=alpha_method,
        fusion_type=fusion_type,
        mlp_hidden_dims=mlp_dims_str,
        per_user_alpha_enabled=effective_per_user_alpha,
        item_perturbation_enabled=effective_item_perturbation,
        contrastive_lambda=contrastive_lambda,
    )

    # D-15: cold-round signal derived from cache-exists probe.
    cache_dir = _cache_dir_for_run(
        run_id=run_id, reuse_cache=reuse_cache, signature=signature
    )
    pt_path = cache_dir / f"partition_{int(partition_id)}.pt"
    is_cold_round = not pt_path.exists()

    # --- Step 6: load LOCAL params from the Phase-4 cache ---
    local_state = _load_local_user_state(
        partition_id=partition_id,
        run_id=run_id,
        reuse_cache=reuse_cache,
        signature=signature,
    )
    if local_state is not None:
        loaded, _missing = model.set_local_parameters(local_state, strict=False)
        print(f"  Client {partition_id}: loaded Phase-4 cached state ({len(loaded)} keys)")
    else:
        print(f"  Client {partition_id}: cold start — using initialized LOCAL state")

    device = get_device()
    model.to(device)

    # --- Step 7: set adaptive alpha + global prototype (scalar fallback path) ---
    if hasattr(model, "set_alpha"):
        model.set_alpha(client_alpha)

    global_prototype_list = msg.content.get("config", {}).get("global_prototype", None)
    if global_prototype_list is not None and hasattr(model, "set_global_prototype"):
        global_prototype = torch.tensor(global_prototype_list, dtype=torch.float32)
        model.set_global_prototype(global_prototype)

    # --- FedProx: save ONLY global params for proximal term ---
    proximal_mu = float(msg_config.get("proximal_mu", 0.0))
    global_params_for_prox: Optional[List[torch.Tensor]] = None
    global_param_names: Optional[List[str]] = None
    if proximal_mu > 0:
        global_param_names = model.get_global_parameter_names()
        global_params_for_prox = []
        for name, p in model.named_parameters():
            if name in set(global_param_names):
                global_params_for_prox.append(p.detach().clone())

    # --- Step 8: train with FND-06 RNG + FND-03 exclusion + D-13/D-14 cold-round ---
    exclude_items = bundle["exclusion"].for_user(partition_id)
    train_rng = np_rng(run_seed, partition_id, round_num, "train_neg")

    # Honor per-message local_epochs_override (D-06.7 / Bug 3 Alt-A calibration
    # pass). Server can shorten the local-epochs budget for the end-of-training
    # calibration broadcast without changing the global `local-epochs` config.
    # Falls back to context.run_config when absent so normal training rounds
    # are unaffected.
    local_epochs = int(
        msg_config.get(
            "local_epochs_override",
            context.run_config.get("local-epochs", 10),
        )
    )
    lr = float(msg_config.get("lr", 0.001))
    num_train_negatives = int(context.run_config.get("num-negatives", 1))

    train_loss = train_fn(
        model=model,
        trainloader=trainloader,
        epochs=local_epochs,
        lr=lr,
        device=device,
        model_type=model_type,
        weight_decay=float(context.run_config.get("weight-decay", 1e-5)),
        num_negatives=num_train_negatives,
        proximal_mu=proximal_mu,
        global_params=global_params_for_prox,
        global_param_names=global_param_names,
        contrastive_lambda=contrastive_lambda,
        contrastive_tau=contrastive_tau,
        run_seed=run_seed,
        user_idx=partition_id,
        round_num=round_num,
        exclude_items=exclude_items,
        rng=train_rng,
        is_cold_round=is_cold_round,
    )

    # --- Step 9a: D-16 alpha diagnostics (computed after training) ---
    alpha_diagnostics = _compute_alpha_diagnostics(model)

    # --- Step 9b: save LOCAL state with the Phase-4 extended key set ---
    local_params_out = model.get_local_parameters()
    _save_local_user_state(
        partition_id=partition_id,
        state_dict=dict(local_params_out),
        run_id=run_id,
        reuse_cache=reuse_cache,
        signature=signature,
    )

    # --- Step 9c: user prototype + GLOBAL params out ---
    # Cross-device protocol: 1 partition = 1 user_id, so partition_id IS the
    # user index in the user_embeddings table. Pass it explicitly so the
    # prototype is the trained row, not mean-over-all-rows (which collapses
    # to ~zero-mean Xavier noise under cross-device — see debug session
    # adaptive-cache-prototype-collapse.md, Bug #2).
    user_prototype: Optional[List[float]] = None
    if hasattr(model, "compute_user_prototype"):
        user_prototype = (
            model.compute_user_prototype(user_id=partition_id)
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

    global_params_out = model.get_global_parameters()
    model_record = ArrayRecord(global_params_out)

    # --- Step 9d: strict FitMetricsContract payload (D-21 + G-03-01) ---
    num_positives = int(len(trainloader.dataset))
    num_training_examples = int(num_positives * (1 + max(num_train_negatives, 0)))
    fit_metrics = FitMetricsContract(
        train_loss=float(train_loss),
        num_positives=num_positives,
        num_training_examples=num_training_examples,
        round_num=round_num,
        partition_id=partition_id,
    ).to_dict()
    validate_fit_metrics(fit_metrics)

    # user_prototype is a list of floats — route through the legacy
    # USER_PROTOTYPE_KEY metric. Since FitMetricsContract is strict
    # (D-21) and rejects free-form extras, the prototype + alpha
    # diagnostics ride in SEPARATE MetricRecords.
    metrics_record = MetricRecord(fit_metrics)
    record_dict_content: Dict[str, Any] = {
        "arrays": model_record,
        "metrics": metrics_record,
    }
    # Route user_prototype as its own record keyed USER_PROTOTYPE_KEY
    # so the existing server-side aggregator code path can read it
    # without confusing the strict fit-metrics validator.
    if user_prototype is not None:
        record_dict_content[USER_PROTOTYPE_KEY] = MetricRecord(
            {USER_PROTOTYPE_KEY: user_prototype}
        )

    # D-16: alpha_diagnostics sidecar (strict contract forbids inline).
    if alpha_diagnostics is not None:
        record_dict_content["alpha_diagnostics"] = MetricRecord(alpha_diagnostics)

    content = RecordDict(record_dict_content)
    return Message(content=content, reply_to=msg)


# =============================================================================
# Evaluation Function (Split Architecture, discover_only + ADP-02 ordering).
# =============================================================================


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate this client's one user on the held-out positive.

    Flow:
      - G-03-01 discover_only short-circuit: return a minimal zero
        EvaluateMetricsContract + partition_id without loading data or
        running the model. Used by the server to build
        partition_id -> node_id before round 1.
      - Otherwise: apply the ADP-02 enable-before-load ordering
        IDENTICAL to @app.train (so cached _logit_alpha /
        _item_perturbation tensors are restored before evaluation).
      - Run evaluate_ranking_sampled with FND-06 RNG + FND-03 exclusion
        (ADP-05 + ADP-06).
      - Emit strict EvaluateMetricsContract with per-group
        sufficient-stat routing (D-22) + partition_id (G-03-01).
    """
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

    mode = str(context.run_config.get("mode", "cross_silo_legacy"))
    profile = resolve_mode_defaults(mode)
    overrides = log_mode_and_overrides(mode, profile, dict(context.run_config))
    is_benchmark_mode = (mode == "benchmark_cross_device")

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

    model_type = str(context.run_config.get("model-type", "dual"))
    embedding_dim = int(context.run_config.get("embedding-dim", 128))
    dropout = float(context.run_config.get("dropout", 0.1))

    mlp_hidden_dims: Optional[List[int]] = None
    fusion_type = str(context.run_config.get("fusion-type", "concat"))
    mlp_dims_str = str(context.run_config.get("mlp-hidden-dims", "512,256,128"))
    if model_type == "dual":
        mlp_hidden_dims = [int(d.strip()) for d in mlp_dims_str.split(",")]

    contrastive_lambda = float(context.run_config.get("contrastive-lambda", 0.1))

    # --- Step 1: construct model ---
    model = get_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        dropout=dropout,
        mlp_hidden_dims=mlp_hidden_dims,
        fusion_type=fusion_type,
    )

    # --- Step 2: load GLOBAL parameters from the server message ---
    global_state = msg.content["arrays"].to_torch_state_dict()
    model.set_global_parameters(global_state)

    bundle = _load_foundation_bundle()
    split_hash = str(getattr(bundle["split_manifest"], "split_hash", ""))
    num_users = int(getattr(bundle["mapping"], "num_users", 6040))
    num_items = int(getattr(bundle["mapping"], "num_items", 3706))

    # --- Step 3: load partition data ---
    dirichlet_alpha = float(context.run_config.get("alpha", 0.5))
    split_mode = str(context.run_config.get("eval-split-mode", "leave-one-out"))
    partition_mode = str(context.run_config.get("partition-mode", "natural"))
    trainloader, testloader, user_stats = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        alpha=dirichlet_alpha,
        compute_stats=True,
        split_mode=split_mode,
        partition_mode=partition_mode,
    )

    # --- Step 4: one-user assertion (ADP-04) ---
    user_ids_in_client = set()
    for batch in testloader:
        user_ids_in_client.update(batch["user"].cpu().numpy().tolist())
    if not user_ids_in_client:
        for batch in trainloader:
            user_ids_in_client.update(batch["user"].cpu().numpy().tolist())
    num_users_in_client = len(user_ids_in_client)
    assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)

    # Primary evaluator guard (ADP-06 client half).
    primary = get_primary_evaluator(mode)
    assert primary == "sampled_loo_99", (
        f"Primary-evaluator invariant broken: get_primary_evaluator({mode!r}) "
        f"returned {primary!r}, expected 'sampled_loo_99'"
    )

    # --- Step 5: ADP-02 enable-before-load ordering (mirror @app.train) ---
    alpha_method, alpha_config, hc_config = _build_alpha_configs(context)
    client_alpha = compute_client_alpha(user_stats, alpha_config, hc_config)

    effective_per_user_alpha, effective_item_perturbation = _resolve_enable_flags(
        context, is_benchmark_mode
    )
    per_user_alphas = compute_per_user_alpha(user_stats, alpha_config, hc_config) if effective_per_user_alpha else {}
    item_perturb_reg = float(context.run_config.get("item-perturbation-reg", 0.01))
    _apply_enable_before_load(
        model,
        effective_per_user_alpha=effective_per_user_alpha,
        effective_item_perturbation=effective_item_perturbation,
        per_user_alphas=per_user_alphas,
        item_perturb_reg=item_perturb_reg,
    )

    signature = _signature_fields_v2(
        run_id=run_id,
        method=model_type,
        num_users=num_users,
        num_items=num_items,
        dim=embedding_dim,
        split_hash=split_hash,
        alpha_method=alpha_method,
        fusion_type=fusion_type,
        mlp_hidden_dims=mlp_dims_str,
        per_user_alpha_enabled=effective_per_user_alpha,
        item_perturbation_enabled=effective_item_perturbation,
        contrastive_lambda=contrastive_lambda,
    )

    # --- Step 6: load LOCAL params from the Phase-4 cache ---
    local_state = _load_local_user_state(
        partition_id=partition_id,
        run_id=run_id,
        reuse_cache=reuse_cache,
        signature=signature,
    )
    if local_state is not None:
        model.set_local_parameters(local_state, strict=False)

    device = get_device()
    model.to(device)

    # --- Step 7: set alpha + global prototype ---
    if hasattr(model, "set_alpha"):
        model.set_alpha(client_alpha)

    global_prototype_list = msg.content.get("config", {}).get("global_prototype", None)
    if global_prototype_list is not None and hasattr(model, "set_global_prototype"):
        global_prototype = torch.tensor(global_prototype_list, dtype=torch.float32)
        model.set_global_prototype(global_prototype)

    exclude_items = bundle["exclusion"].for_user(partition_id)

    # Rating-prediction diagnostics (RMSE / MAE) — NOT consumed by the
    # thesis-table aggregator; cached optional fields on the contract.
    eval_loss, _rating_metrics = test_fn(
        model=model,
        testloader=testloader,
        device=str(device),
        model_type=model_type,
    )

    # --- Step 8: primary-path evaluator (FND-06 RNG + FND-03 exclusion) ---
    num_eval_negatives = int(context.run_config.get("eval-num-negatives", 99))
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

    # Optional all-items ranking — stays NAMESPACED (allrank_*) and NOT
    # consumed by the aggregator. Only runs when explicitly enabled.
    enable_ranking_eval = bool(context.run_config.get("enable-ranking-eval", False))
    if enable_ranking_eval:
        k_values_str = str(context.run_config.get("ranking-k-values", "10"))
        k_values = [int(k.strip()) for k in k_values_str.split(",")]
        _ = evaluate_ranking(
            model=model,
            testloader=testloader,
            device=str(device),
            k_values=k_values,
            trainloader=trainloader,
        )

    # --- Step 9: D-22 per-group sufficient-stat routing ---
    user_group = _classify_partition_user_group(bundle, partition_id)
    sampled_num_users = int(sampled_metrics.get("sampled_num_users", 0))
    hr10_ratio = float(sampled_metrics.get("sampled_hr@10", 0.0))
    ndcg10_ratio = float(sampled_metrics.get("sampled_ndcg@10", 0.0))
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
        eval_loss=float(eval_loss),
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

    content = RecordDict({"metrics": MetricRecord(eval_payload)})
    return Message(content=content, reply_to=msg)
