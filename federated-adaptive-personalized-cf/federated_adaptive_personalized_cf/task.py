"""Matrix Factorization training and evaluation for MovieLens 1M (Phase 4 Plan 03).

Adaptive-module migration hooks (ADP-02, ADP-05, ADP-06, D-13, D-14, D-24):

- FND-06 RNG threaded end-to-end: ``train`` / ``train_dual_personalized`` /
  ``train_bpr_mf`` / ``train_basic_mf`` / ``evaluate_ranking_sampled`` all
  accept ``run_seed`` / ``user_idx`` / ``round_num`` / ``exclude_items`` /
  ``rng`` keyword-only params.
- FND-03 exclusion folded into negative-candidate pools on both the train
  and eval sides so the held-out test positive is NEVER drawn (ADP-05).
- Stdlib-random seeding / sampling / module-level import are all stripped
  (cross-file regression test in ``tests/test_task_rng.py``).
- ``_sample_negatives_seeded`` replaces ``BPRMF.sample_negatives(...)``
  (which uses process-global ``np.random``) with a rejection sampler
  drawn from an ``np.random.Generator`` instance so draws are
  deterministic across rounds without touching the stdlib RNG.
- D-13 + D-14 cold-round branch: when ``is_cold_round=True``,
  ``train_dual_personalized`` forces ``model.set_alpha(0.0)`` (prototype-
  only blend) and zeros the contrastive-loss coefficient for that pass;
  restores the saved alpha in a try/finally. Directly benefits sparse
  users whose first round for any partition starts from Xavier-noisy
  ``p_local``.
- D-24 ghost-table gradient isolation: the adaptive module KEEPS the
  ``num_users * d`` ``user_embeddings`` table (PSN-06 single-row collapse
  is deferred to Phase 4.5). ``_snapshot_non_user_rows`` /
  ``_restore_non_user_rows`` bracket ``optimizer.step()`` so Adam's
  weight-decay + momentum don't drift the non-active rows of the three
  user-indexed LOCAL params: ``user_embeddings`` + ``user_bias`` +
  ``_logit_alpha``. ``_item_perturbation`` is item-indexed and therefore
  legitimately full-table updated every batch — NOT protected.
"""

from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.optim as optim

# Phase 4 Plan 03 (ADP-06 / BSL-05-style): foundation RNG factories.
# Do NOT use Python's stdlib RNG here — task.py + client_app.py are ADP-06
# strip targets. All stochastic operations route through np_rng() /
# torch_gen() instances seeded via the four-tier SHA-256 derivation in
# fedrec_foundation.rng (see FND-06 + CR-3).
from fedrec_foundation.rng import np_rng, torch_gen  # noqa: F401 (torch_gen kept for parity)

from federated_adaptive_personalized_cf.dataset import load_partition_data
from federated_adaptive_personalized_cf.models import (
    AlphaConfig,
    BasicMF,
    BPRMF,
    BPRLoss,
    DataQuantityAlpha,
    DualPersonalizedBPRMF,
    HierarchicalConditionalAlphaConfig,
    MSELoss,
    create_alpha_computer,
)


# Global cache for dataset metadata
_dataset_cache: Dict[str, object] = {}

# Global cache for item popularity (computed from training data)
_item_popularity_cache: Dict[int, float] = {}

# Global cache for user statistics
_user_stats_cache: Dict[int, Dict] = {}


# ============================================================================
# D-24 gradient isolation — protects non-active rows of the ghost-table LOCAL
# params from Adam's weight-decay + momentum drift.
#
# Applies to (all user-indexed):
#   - user_embeddings (num_users, d)   — ghost table
#   - user_bias       (num_users, 1)   — ghost table
#   - _logit_alpha    (num_users, 1)   — ghost table (when enable_per_user_alpha)
#
# Does NOT apply to _item_perturbation (num_items, d) — item-indexed; training
# legitimately updates all items (or the sampled subset) every batch.
# ============================================================================

_D24_PROTECTED_EMBEDDINGS: Tuple[str, ...] = (
    "user_embeddings",
    "user_bias",
    "_logit_alpha",
)


def _snapshot_non_user_rows(model, user_idx: Optional[int]) -> Dict[str, torch.Tensor]:
    """Return a cloned copy of every D-24-protected embedding weight.

    The ``user_idx`` row is replaced by NaN inside the snapshot so
    :func:`_restore_non_user_rows` never overwrites the legitimate
    post-step update on the active row.

    Parameters
    ----------
    model : nn.Module
        The DualPersonalizedBPRMF (or BPRMF) instance.
    user_idx : Optional[int]
        Index of the "active" user whose row should survive the restore.
        When ``None`` (legacy callers), no snapshot is produced.

    Returns
    -------
    Dict[str, torch.Tensor]
        Mapping from protected-embedding attr name to its snapshot tensor.
    """
    snapshots: Dict[str, torch.Tensor] = {}
    if user_idx is None:
        return snapshots
    for name in _D24_PROTECTED_EMBEDDINGS:
        module = getattr(model, name, None)
        if module is None:
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        snap = weight.detach().clone()
        # Mark active row as NaN so restore skips it.
        snap[int(user_idx)] = float("nan")
        snapshots[name] = snap
    return snapshots


def _restore_non_user_rows(model, snapshots: Dict[str, torch.Tensor]) -> None:
    """Copy every non-NaN row of each snapshot back into the model weight.

    The NaN row(s) at ``user_idx`` are left untouched so the legitimate
    gradient update on that row survives.

    Parameters
    ----------
    model : nn.Module
        The DualPersonalizedBPRMF (or BPRMF) instance.
    snapshots : Dict[str, torch.Tensor]
        Output of :func:`_snapshot_non_user_rows`.
    """
    if not snapshots:
        return
    with torch.no_grad():
        for name, snap in snapshots.items():
            module = getattr(model, name, None)
            if module is None:
                continue
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            # Build a row-level mask: True where NO element of the row is NaN.
            if snap.dim() > 1:
                mask = ~torch.isnan(snap).any(dim=tuple(range(1, snap.dim())))
            else:
                mask = ~torch.isnan(snap)
            # Copy protected rows back (moving to the weight's device to be
            # safe when the model sits on a GPU).
            restore = snap[mask].to(weight.data.device)
            weight.data[mask] = restore


# ============================================================================
# Seeded negative-sampling helper (ADP-06).
# ============================================================================


def _sample_negatives_seeded(
    user_rated_items: Set[int],
    num_items: int,
    num_negatives: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rejection-sampled negatives from an ``np.random.Generator`` (ADP-06).

    Distribution-equivalent to uniform sampling over
    ``range(num_items) \\ user_rated_items`` but deterministic under a
    given ``rng`` instance (FND-06). The FND-03 exclusion set is expected
    to already be merged into ``user_rated_items`` by the caller.

    Parameters
    ----------
    user_rated_items : Set[int]
        Flat set of item indices to reject from the draw (observed
        positives + FND-03 exclusion union).
    num_items : int
        Catalog size; items drawn from ``[0, num_items)``.
    num_negatives : int
        Number of negatives to return.
    rng : numpy.random.Generator
        FND-06 RNG instance.

    Returns
    -------
    numpy.ndarray
        Array of ``num_negatives`` item indices (dtype ``int64``). May be
        short if rejection runs out of tries on a pathologically tight
        rated-set.
    """
    out: List[int] = []
    pool = int(num_items)
    max_tries = int(num_negatives) * 64 + 16
    while len(out) < int(num_negatives) and max_tries > 0:
        cand = int(rng.integers(0, pool))
        if cand not in user_rated_items:
            out.append(cand)
        max_tries -= 1
    return np.asarray(out, dtype=np.int64)


# ============================================================================
# Data + model factories (preserved verbatim — D-18 surgical-edit discipline).
# ============================================================================


def load_data(
    partition_id: int,
    num_partitions: int,
    alpha: float = 0.5,
    test_ratio: float = 0.2,
    batch_size: int = 256,
    data_dir: str = None,
    compute_stats: bool = True,
    split_mode: str = "leave-one-out",
    partition_mode: str = "dirichlet",
):
    """Load MovieLens 1M data for a specific partition.

    Same 7-tuple return shape as Phase 3 (including ``user_stats``) for
    backwards compatibility with the adaptive client handlers and
    ``compute_client_alpha`` / ``compute_per_user_alpha``.
    """
    trainloader, testloader, num_users, num_items, user2idx, item2idx, user_stats = load_partition_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        alpha=alpha,
        test_ratio=test_ratio,
        batch_size=batch_size,
        data_dir=data_dir,
        compute_stats=compute_stats,
        split_mode=split_mode,
        partition_mode=partition_mode,
    )

    if user_stats is None:
        user_stats = {}

    _dataset_cache['num_users'] = num_users
    _dataset_cache['num_items'] = num_items
    _dataset_cache['user2idx'] = user2idx
    _dataset_cache['item2idx'] = item2idx

    _user_stats_cache.update(user_stats)

    if compute_stats:
        return trainloader, testloader, user_stats
    return trainloader, testloader


def get_user_stats() -> Dict[int, Dict]:
    """Get cached user statistics for alpha computation."""
    return _user_stats_cache.copy()


def compute_client_alpha(
    user_stats: Dict[int, Dict],
    alpha_config: Optional[AlphaConfig] = None,
    hc_config: Optional[HierarchicalConditionalAlphaConfig] = None,
) -> float:
    """Compute aggregate alpha for a client based on user statistics."""
    if not user_stats:
        return 1.0

    alpha_computer = create_alpha_computer(alpha_config, hc_config)

    total_interactions = 0
    weighted_alpha_sum = 0.0

    for user_id, stats in user_stats.items():
        n_interactions = stats.get('n_interactions', 0)
        if n_interactions > 0:
            user_alpha = alpha_computer.compute_from_stats(stats)
            weighted_alpha_sum += user_alpha * n_interactions
            total_interactions += n_interactions

    if total_interactions == 0:
        return 1.0

    return weighted_alpha_sum / total_interactions


def compute_per_user_alpha(
    user_stats: Dict[int, Dict],
    alpha_config: Optional[AlphaConfig] = None,
    hc_config: Optional[HierarchicalConditionalAlphaConfig] = None,
) -> Dict[int, float]:
    """Compute alpha for each user based on their statistics."""
    alpha_computer = create_alpha_computer(alpha_config, hc_config)

    user_alphas: Dict[int, float] = {}
    for user_id, stats in user_stats.items():
        user_alphas[user_id] = alpha_computer.compute_from_stats(stats)

    return user_alphas


def get_model(
    model_type: str = "bpr",
    num_users: Optional[int] = None,
    num_items: Optional[int] = None,
    embedding_dim: int = 64,
    dropout: float = 0.1,
    mlp_hidden_dims: Optional[List[int]] = None,
    fusion_type: str = "add",
):
    """Create a Matrix Factorization model."""
    if num_users is None:
        num_users = _dataset_cache.get('num_users', 6040)
    if num_items is None:
        num_items = _dataset_cache.get('num_items', 3706)

    if model_type.lower() == "basic":
        model = BasicMF(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
    elif model_type.lower() == "bpr":
        model = BPRMF(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            dropout=dropout,
            use_bias=True,
        )
    elif model_type.lower() == "dual":
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [embedding_dim, embedding_dim // 2]
        model = DualPersonalizedBPRMF(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout,
            use_bias=True,
            fusion_type=fusion_type,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'basic', 'bpr', or 'dual'.")

    return model


# ============================================================================
# Training functions (FND-06 RNG + FND-03 exclusion + D-24 gradient isolation).
# ============================================================================


def train_basic_mf(
    model: BasicMF,
    trainloader,
    epochs: int,
    lr: float,
    device: str,
    weight_decay: float = 1e-5,
    proximal_mu: float = 0.0,
    global_params: list = None,
    global_param_names: list = None,
    *,
    run_seed: Optional[int] = None,
    user_idx: Optional[int] = None,
    round_num: Optional[int] = None,
    exclude_items: Optional[Iterable[int]] = None,
    rng: Optional[np.random.Generator] = None,
    is_cold_round: bool = False,
) -> float:
    """Train BasicMF model with MSE loss.

    BasicMF has no negative sampling and no alpha blend, so the FND-06 /
    FND-03 / D-13 / D-14 kwargs are accepted for signature parity and are
    otherwise unused (``del`` below).
    """
    del exclude_items, rng, run_seed, user_idx, round_num, is_cold_round

    model.to(device)
    model.train()

    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in trainloader:
            user_ids = batch['user'].to(device)
            item_ids = batch['item'].to(device)
            ratings = batch['rating'].to(device)

            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)

            if proximal_mu > 0 and global_params is not None:
                proximal_term = 0.0
                if global_param_names is not None:
                    global_param_set = set(global_param_names)
                    idx = 0
                    for name, local_w in model.named_parameters():
                        if name in global_param_set:
                            proximal_term += (local_w - global_params[idx].to(device)).norm(2) ** 2
                            idx += 1
                else:
                    for local_w, global_w in zip(model.parameters(), global_params):
                        proximal_term += (local_w - global_w.to(device)).norm(2) ** 2
                loss = loss + (proximal_mu / 2) * proximal_term

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        total_loss += epoch_loss

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def _collect_user_rated_items(trainloader) -> Set[int]:
    """Build a flat item-set from all batches in a trainloader."""
    items: Set[int] = set()
    for batch in trainloader:
        for i in batch['item'].numpy():
            items.add(int(i))
    return items


def train_bpr_mf(
    model: BPRMF,
    trainloader,
    epochs: int,
    lr: float,
    device: str,
    weight_decay: float = 1e-5,
    num_negatives: int = 1,
    proximal_mu: float = 0.0,
    global_params: list = None,
    global_param_names: list = None,
    contrastive_lambda: float = 0.0,
    contrastive_tau: float = 0.1,
    *,
    run_seed: Optional[int] = None,
    user_idx: Optional[int] = None,
    round_num: Optional[int] = None,
    exclude_items: Optional[Iterable[int]] = None,
    rng: Optional[np.random.Generator] = None,
    is_cold_round: bool = False,
) -> float:
    """Train BPRMF model with BPR loss (Phase 4 cross-device contract).

    Threads ADP-05 (FND-03 exclusion), ADP-06 (FND-06 RNG), D-13/D-14
    (cold-round alpha/contrastive override — no-op for plain BPRMF), and
    D-24 (gradient isolation for the ghost-table ``user_embeddings`` and
    ``user_bias`` LOCAL params).
    """
    model.to(device)
    model.train()

    criterion = BPRLoss()

    # Contrastive loss (only if enabled). DualPersonalizedBPRMF supports
    # ``get_local_embedding`` / ``get_effective_embedding``; plain BPRMF
    # does not, so the loss routes a no-op in that case.
    contrastive_loss_fn = None
    if contrastive_lambda > 0 and not is_cold_round:
        from federated_adaptive_personalized_cf.models.losses import InfoNCEContrastiveLoss
        contrastive_loss_fn = InfoNCEContrastiveLoss(temperature=contrastive_tau)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # BSL-05-style: seeded RNG (never touches np.random process-global state).
    if rng is None and run_seed is not None and user_idx is not None:
        rng = np_rng(int(run_seed), int(user_idx), int(round_num or 0), "train_neg")

    # Build user_rated_items dictionary for negative sampling.
    # Pre-Phase-4 BPRMF expected a Dict[int, Set[int]] keyed by user; under
    # cross-device each batch contains exactly one user, so we also need
    # the legacy dict form for the fallback sample_negatives path.
    user_rated_items: Dict[int, Set[int]] = {}
    for batch in trainloader:
        users = batch['user'].numpy()
        items = batch['item'].numpy()
        for u, i in zip(users, items):
            user_rated_items.setdefault(int(u), set()).add(int(i))

    # ADP-05: fold FND-03 exclusion into the rated-set for every user.
    if exclude_items is not None:
        excluded = {int(x) for x in np.asarray(list(exclude_items)).tolist()}
        for u in list(user_rated_items.keys()):
            user_rated_items[u] |= excluded
        # Ensure the active user's bucket exists even when the trainloader
        # was empty (cold rounds on users with no ratings).
        if user_idx is not None and int(user_idx) not in user_rated_items:
            user_rated_items[int(user_idx)] = set(excluded)

    num_items = int(getattr(model, "num_items", 0))

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in trainloader:
            user_ids = batch['user'].to(device)
            pos_item_ids = batch['item'].to(device)
            batch_size = int(pos_item_ids.shape[0])

            # ADP-06: seeded negative sampling replaces the process-global
            # ``model.sample_negatives`` path. Fall back to the model's
            # built-in sampler only when the FND-06 rng was not supplied.
            if rng is not None:
                # One negative per positive (num_negatives == 1) or a
                # (batch_size, num_negatives) matrix (num_negatives > 1).
                if num_negatives == 1:
                    # Merge every user's rated items into a single set —
                    # under cross-device the batch is one user anyway so
                    # the set stays small.
                    union_rated = set().union(*user_rated_items.values()) if user_rated_items else set()
                    draws = _sample_negatives_seeded(union_rated, num_items, batch_size, rng)
                    neg_item_ids = torch.from_numpy(draws).to(device)
                else:
                    rows = []
                    for _ in range(batch_size):
                        union_rated = set().union(*user_rated_items.values()) if user_rated_items else set()
                        rows.append(_sample_negatives_seeded(union_rated, num_items, num_negatives, rng))
                    neg_item_ids = torch.from_numpy(np.stack(rows)).to(device)
            else:
                neg_item_ids = model.sample_negatives(
                    user_ids,
                    pos_item_ids,
                    num_negatives=num_negatives,
                    user_rated_items=user_rated_items,
                    sampling_strategy='uniform',
                )

            pos_scores, neg_scores = model(user_ids, pos_item_ids, neg_item_ids)
            loss = criterion(pos_scores, neg_scores)

            # Contrastive local-global alignment (D-14: skipped on cold rounds).
            if contrastive_loss_fn is not None and hasattr(model, 'get_local_embedding'):
                unique_users = torch.unique(user_ids)
                local_emb = model.get_local_embedding(unique_users)
                effective_emb = model.get_effective_embedding(unique_users)
                cl_loss = contrastive_loss_fn(local_emb, effective_emb)
                loss = loss + contrastive_lambda * cl_loss

            # Item perturbation L2 regularization.
            if hasattr(model, 'get_item_perturbation_reg_loss'):
                loss = loss + model.get_item_perturbation_reg_loss()

            # FedProx proximal term — ONLY on GLOBAL params (split learning).
            if proximal_mu > 0 and global_params is not None:
                proximal_term = 0.0
                if global_param_names is not None:
                    global_param_set = set(global_param_names)
                    idx = 0
                    for name, local_w in model.named_parameters():
                        if name in global_param_set:
                            proximal_term += (local_w - global_params[idx].to(device)).norm(2) ** 2
                            idx += 1
                else:
                    for local_w, global_w in zip(model.parameters(), global_params):
                        proximal_term += (local_w - global_w.to(device)).norm(2) ** 2
                loss = loss + (proximal_mu / 2) * proximal_term

            # D-24 gradient isolation: snapshot -> step -> restore non-user rows
            # of the three user-indexed LOCAL embeddings. No-op when user_idx
            # is None (legacy callers) or the snapshot dict is empty.
            snapshots = _snapshot_non_user_rows(model, user_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _restore_non_user_rows(model, snapshots)

            epoch_loss += loss.item()
            num_batches += 1

        total_loss += epoch_loss

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train_dual_personalized(
    model: DualPersonalizedBPRMF,
    trainloader,
    epochs: int,
    lr: float,
    device: str,
    weight_decay: float = 1e-5,
    num_negatives: int = 1,
    proximal_mu: float = 0.0,
    global_params: list = None,
    global_param_names: list = None,
    contrastive_lambda: float = 0.0,
    contrastive_tau: float = 0.1,
    *,
    run_seed: Optional[int] = None,
    user_idx: Optional[int] = None,
    round_num: Optional[int] = None,
    exclude_items: Optional[Iterable[int]] = None,
    rng: Optional[np.random.Generator] = None,
    is_cold_round: bool = False,
) -> float:
    """Train DualPersonalizedBPRMF with the Phase 4 cross-device contract.

    Under the hood this is :func:`train_bpr_mf` plus the D-13/D-14
    cold-round bracket:

    - **D-13** (cold round, ``is_cold_round=True``): save the current
      ``model.get_alpha()``, call ``model.set_alpha(0.0)`` so
      ``p_effective = p_global`` for this training pass; restore the
      saved alpha in the try/finally cleanup.
    - **D-14** (cold round): zero the contrastive-loss coefficient for
      this round. ``L = L_BPR + reg * ||item_perturbation||^2`` only on
      cold rounds — positive pair ``(Xavier_noise, p_global)`` is either
      a noise anchor or trivial.
    - **D-24**: inherited from ``train_bpr_mf`` — snapshot/restore
      brackets ``optimizer.step()`` to isolate non-active rows of the
      ghost-table LOCAL params.

    Parameters mirror :func:`train_bpr_mf` exactly.
    """
    # ============================================================
    # D-13 + D-14: cold-round overrides. Save + override BEFORE any
    # training; the try/finally cleanup restores the saved alpha even
    # if training raises.
    # ============================================================
    saved_alpha: Optional[float] = None
    contrastive_lambda_eff = contrastive_lambda
    if is_cold_round:
        if hasattr(model, "get_alpha") and hasattr(model, "set_alpha"):
            saved_alpha = float(model.get_alpha())
            model.set_alpha(0.0)  # D-13: prototype-only blend
        contrastive_lambda_eff = 0.0  # D-14: skip contrastive on cold round

    try:
        return train_bpr_mf(
            model,
            trainloader,
            epochs,
            lr,
            device,
            weight_decay=weight_decay,
            num_negatives=num_negatives,
            proximal_mu=proximal_mu,
            global_params=global_params,
            global_param_names=global_param_names,
            contrastive_lambda=contrastive_lambda_eff,
            contrastive_tau=contrastive_tau,
            run_seed=run_seed,
            user_idx=user_idx,
            round_num=round_num,
            exclude_items=exclude_items,
            rng=rng,
            is_cold_round=is_cold_round,
        )
    finally:
        # D-13 cleanup: restore the saved alpha so subsequent evaluation
        # sees the original value. Guard on hasattr so we never AttributeError
        # if the caller passed a plain BPRMF by mistake.
        if is_cold_round and saved_alpha is not None and hasattr(model, "set_alpha"):
            model.set_alpha(saved_alpha)


def train(
    model,
    trainloader,
    epochs: int,
    lr: float,
    device: str,
    model_type: str = "bpr",
    **kwargs,
) -> float:
    """Unified training dispatcher (Phase 4 cross-device contract).

    Threads five FND-06/FND-03 kwargs (run_seed, user_idx, round_num,
    exclude_items, rng) and one D-13/D-14 kwarg (is_cold_round) into the
    underlying trainer. The dual-model pathway goes through
    :func:`train_dual_personalized` so the cold-round bracket is honored.
    """
    common = dict(
        weight_decay=kwargs.get('weight_decay', 1e-5),
        proximal_mu=kwargs.get('proximal_mu', 0.0),
        global_params=kwargs.get('global_params', None),
        global_param_names=kwargs.get('global_param_names', None),
        run_seed=kwargs.get('run_seed', None),
        user_idx=kwargs.get('user_idx', None),
        round_num=kwargs.get('round_num', None),
        exclude_items=kwargs.get('exclude_items', None),
        rng=kwargs.get('rng', None),
        is_cold_round=kwargs.get('is_cold_round', False),
    )
    if model_type.lower() == "basic":
        return train_basic_mf(
            model,
            trainloader,
            epochs,
            lr,
            device,
            **common,
        )
    elif model_type.lower() == "bpr":
        return train_bpr_mf(
            model,
            trainloader,
            epochs,
            lr,
            device,
            num_negatives=kwargs.get('num_negatives', 1),
            contrastive_lambda=kwargs.get('contrastive_lambda', 0.0),
            contrastive_tau=kwargs.get('contrastive_tau', 0.1),
            **common,
        )
    elif model_type.lower() == "dual":
        return train_dual_personalized(
            model,
            trainloader,
            epochs,
            lr,
            device,
            num_negatives=kwargs.get('num_negatives', 1),
            contrastive_lambda=kwargs.get('contrastive_lambda', 0.0),
            contrastive_tau=kwargs.get('contrastive_tau', 0.1),
            **common,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def test(
    model,
    testloader,
    device: str,
    model_type: str = "bpr",
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on test set (rating-prediction diagnostics)."""
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_squared_error = 0.0
    total_absolute_error = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in testloader:
            user_ids = batch['user'].to(device)
            item_ids = batch['item'].to(device)
            ratings = batch['rating'].to(device)

            if model_type.lower() == "basic":
                predictions = model(user_ids, item_ids)
                predictions = torch.clamp(predictions, min=1.0, max=5.0)
            elif model_type.lower() in ("bpr", "dual"):
                predictions = model(user_ids, item_ids, neg_item_ids=None)
                predictions = torch.clamp(predictions, min=1.0, max=5.0)

            squared_errors = (predictions - ratings) ** 2
            absolute_errors = torch.abs(predictions - ratings)

            total_squared_error += squared_errors.sum().item()
            total_absolute_error += absolute_errors.sum().item()
            num_samples += len(ratings)

            if model_type.lower() == "basic":
                criterion = MSELoss()
                loss = criterion(predictions, ratings)
            else:
                mse = squared_errors.mean()
                loss = mse

            total_loss += loss.item() * len(ratings)

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    rmse = np.sqrt(total_squared_error / num_samples) if num_samples > 0 else 0.0
    mae = total_absolute_error / num_samples if num_samples > 0 else 0.0

    metrics = {
        'rmse': rmse,
        'mae': mae,
    }

    return avg_loss, metrics


# ============================================================================
# Ranking-metric helpers (preserved verbatim from pre-Phase-4 file).
# ============================================================================


def compute_ndcg(ranked_items, relevant_items, k):
    """Compute Normalized Discounted Cumulative Gain (NDCG) at K."""
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)

    num_relevant = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_mrr(ranked_items, relevant_items):
    """Compute Mean Reciprocal Rank (MRR) for a single user."""
    for i, item in enumerate(ranked_items):
        if item in relevant_items:
            return 1.0 / (i + 1)
    return 0.0


def compute_ap(ranked_items, relevant_items, k: int) -> float:
    """Compute Average Precision at K for a single user."""
    if not relevant_items:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for i, item in enumerate(ranked_items[:k]):
        if item in relevant_items:
            hits += 1
            precision_sum += hits / (i + 1)

    return precision_sum / min(k, len(relevant_items))


def compute_novelty(
    ranked_items,
    item_popularity: Dict[int, float],
    k: int,
) -> float:
    """Compute Novelty at K for a single user's recommendations."""
    if len(ranked_items) == 0:
        return 0.0

    novelties = []
    for item in ranked_items[:k]:
        pop = item_popularity.get(item, 1e-10)
        novelties.append(-np.log2(max(pop, 1e-10)))

    return float(np.mean(novelties)) if novelties else 0.0


def evaluate_ranking(
    model,
    testloader,
    device: str,
    k_values: list = None,
    item_popularity: Optional[Dict[int, float]] = None,
    trainloader=None,
) -> Dict[str, float]:
    """Comprehensive all-items ranking evaluation (diagnostic only)."""
    if k_values is None:
        k_values = [5, 10, 20]

    model.to(device)
    model.eval()

    user_test_items: Dict[int, Set[int]] = {}
    for batch in testloader:
        users = batch['user'].numpy()
        items = batch['item'].numpy()
        for u, i in zip(users, items):
            if u not in user_test_items:
                user_test_items[u] = set()
            user_test_items[u].add(i)

    if item_popularity is None:
        item_popularity = {}
        if trainloader is not None:
            item_counts = Counter()
            total_interactions = 0
            for batch in trainloader:
                items = batch['item'].numpy()
                item_counts.update(items)
                total_interactions += len(items)
            if total_interactions > 0:
                for item_id, count in item_counts.items():
                    item_popularity[int(item_id)] = count / total_interactions
        _item_popularity_cache.update(item_popularity)

    num_total_items = model.num_items if hasattr(model, 'num_items') else _dataset_cache.get('num_items', 3706)

    metrics_per_k = {k: {
        'hits': 0,
        'precisions': [],
        'recalls': [],
        'f1s': [],
        'ndcgs': [],
        'aps': [],
        'novelties': [],
        'recommended_items': set(),
    } for k in k_values}

    mrr_scores = []
    num_users = 0
    max_k = max(k_values)

    with torch.no_grad():
        for user_id in user_test_items.keys():
            test_items = user_test_items[user_id]

            top_items, _ = model.recommend(user_id, top_k=max_k, exclude_items=None)

            mrr = compute_mrr(top_items, test_items)
            mrr_scores.append(mrr)

            for k in k_values:
                top_k_items = top_items[:k]

                hits_for_user = len(set(top_k_items) & test_items)
                if hits_for_user > 0:
                    metrics_per_k[k]['hits'] += 1

                precision = hits_for_user / k if k > 0 else 0
                recall = hits_for_user / len(test_items) if len(test_items) > 0 else 0

                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0

                ndcg = compute_ndcg(top_k_items, test_items, k)
                ap = compute_ap(top_k_items, test_items, k)
                novelty = compute_novelty(top_k_items, item_popularity, k)

                metrics_per_k[k]['recommended_items'].update(top_k_items)

                metrics_per_k[k]['precisions'].append(precision)
                metrics_per_k[k]['recalls'].append(recall)
                metrics_per_k[k]['f1s'].append(f1)
                metrics_per_k[k]['ndcgs'].append(ndcg)
                metrics_per_k[k]['aps'].append(ap)
                metrics_per_k[k]['novelties'].append(novelty)

            num_users += 1

    results: Dict[str, float] = {}

    for k in k_values:
        results[f'hit_rate@{k}'] = metrics_per_k[k]['hits'] / num_users if num_users > 0 else 0.0
        results[f'accuracy@{k}'] = results[f'hit_rate@{k}']
        results[f'precision@{k}'] = float(np.mean(metrics_per_k[k]['precisions'])) if metrics_per_k[k]['precisions'] else 0.0
        results[f'recall@{k}'] = float(np.mean(metrics_per_k[k]['recalls'])) if metrics_per_k[k]['recalls'] else 0.0
        results[f'f1@{k}'] = float(np.mean(metrics_per_k[k]['f1s'])) if metrics_per_k[k]['f1s'] else 0.0
        results[f'ndcg@{k}'] = float(np.mean(metrics_per_k[k]['ndcgs'])) if metrics_per_k[k]['ndcgs'] else 0.0
        results[f'map@{k}'] = float(np.mean(metrics_per_k[k]['aps'])) if metrics_per_k[k]['aps'] else 0.0
        results[f'coverage@{k}'] = len(metrics_per_k[k]['recommended_items']) / num_total_items if num_total_items > 0 else 0.0
        results[f'novelty@{k}'] = float(np.mean(metrics_per_k[k]['novelties'])) if metrics_per_k[k]['novelties'] else 0.0

    results['mrr'] = float(np.mean(mrr_scores)) if mrr_scores else 0.0

    return results


def evaluate_ranking_sampled(
    model,
    testloader,
    trainloader,
    device: str,
    k_values: Optional[List[int]] = None,
    num_negatives: int = 99,
    seed: int = 42,
    *,
    run_seed: Optional[int] = None,
    user_idx: Optional[int] = None,
    round_num: Optional[int] = None,
    exclude_items: Optional[Iterable[int]] = None,
) -> Dict[str, float]:
    """Ranking evaluation with leave-one-out and sampled negatives (ADP-05 + ADP-06).

    Phase 4 Plan 03 strips the process-global stdlib-random seeding that
    the pre-Phase-4 code used (the old ``seed``-call at line 952-953 and
    the stdlib ``sample``-call at line 1012 — both against the
    ``random`` module). The new contract derives its seed from the
    (run_seed, user_idx, round_num, "eval_neg") tuple per FND-06; the
    legacy ``seed`` argument is IGNORED when ``run_seed`` is provided.

    Parameters
    ----------
    model : nn.Module
        Model instance (BasicMF, BPRMF, or DualPersonalizedBPRMF).
    testloader : DataLoader
        Test data loader.
    trainloader : DataLoader
        Training data loader (to build the observed-items exclusion pool).
    device : str
        'cuda' or 'cpu'.
    k_values : Optional[List[int]]
        K values to evaluate (default: ``[5, 10, 20]``).
    num_negatives : int
        Number of negatives per positive (default: 99 per NCF protocol).
    seed : int
        Legacy argument kept for backwards compatibility; IGNORED. The
        new contract derives its seed from ``(run_seed, user_idx,
        round_num, "eval_neg")`` via :func:`np_rng` (FND-06 / ADP-06).
    run_seed : Optional[int]
        FND-06 root seed. Typically ``context.run_config["run-seed"]``.
    user_idx : Optional[int]
        Client's partition_id / user_idx (drives per-user RNG namespace).
    round_num : Optional[int]
        Current FL round number.
    exclude_items : Optional[Iterable[int]]
        FND-03 exclusion set. Merged into the rated-items union so the
        held-out test positive never appears as a sampled negative.

    Returns
    -------
    Dict[str, float]
        Sampled ranking metrics with keys like ``sampled_hr@10``,
        ``sampled_ndcg@10``, ``sampled_mrr``, ``sampled_num_negatives``,
        ``sampled_num_users``.
    """
    # Legacy `seed` kwarg is intentionally unused — BSL-05-style forbids
    # touching the stdlib RNG. Preserve the parameter for signature
    # backwards-compat so old callers do not break.
    del seed

    if k_values is None:
        k_values = [5, 10, 20]

    model.to(device)
    model.eval()

    user_train_items: Dict[int, Set[int]] = {}
    for batch in trainloader:
        users = batch['user'].numpy()
        items = batch['item'].numpy()
        for u, i in zip(users, items):
            key = int(u)
            if key not in user_train_items:
                user_train_items[key] = set()
            user_train_items[key].add(int(i))

    user_test_items: Dict[int, List[int]] = {}
    for batch in testloader:
        users = batch['user'].numpy()
        items = batch['item'].numpy()
        for u, i in zip(users, items):
            key = int(u)
            if key not in user_test_items:
                user_test_items[key] = []
            user_test_items[key].append(int(i))

    num_total_items = model.num_items if hasattr(model, 'num_items') else _dataset_cache.get('num_items', 3706)
    all_items = set(range(num_total_items))

    # ADP-05: fold the foundation exclusion set into the rated-items pool.
    excluded_set: Set[int] = set()
    if exclude_items is not None:
        excluded_set = {int(x) for x in np.asarray(list(exclude_items)).tolist()}

    # ADP-06: seeded RNG. Fall back to a deterministic default_rng when
    # the FND-06 inputs are absent — never touch the stdlib RNG.
    if run_seed is not None and user_idx is not None:
        rng = np_rng(int(run_seed), int(user_idx), int(round_num or 0), "eval_neg")
    else:
        rng = np.random.default_rng(0)

    metrics_per_k = {k: {
        'hits': 0,
        'ndcgs': [],
    } for k in k_values}

    mrr_scores: List[float] = []
    num_users = 0

    with torch.no_grad():
        for user_id in sorted(user_test_items.keys()):
            test_items = user_test_items[user_id]
            train_items = user_train_items.get(user_id, set())

            if len(test_items) == 0:
                continue

            positive_item = int(test_items[0])

            # Build the negative-candidate pool. ADP-05 merges the FND-03
            # exclusion set so the held-out test positive never appears.
            all_user_items = train_items | set(test_items) | excluded_set
            negative_candidates = list(all_items - all_user_items)

            if len(negative_candidates) < num_negatives:
                negative_items = negative_candidates
            else:
                # ADP-06: seeded ``rng.choice`` replaces the old stdlib
                # ``random.sample`` call (which was tied to a process-
                # global seed state).
                chosen = rng.choice(
                    np.asarray(negative_candidates, dtype=np.int64),
                    size=num_negatives,
                    replace=False,
                )
                negative_items = [int(x) for x in chosen.tolist()]

            candidate_items = [positive_item] + [int(x) for x in negative_items]

            user_tensor = torch.tensor([user_id] * len(candidate_items), dtype=torch.long).to(device)
            item_tensor = torch.tensor(candidate_items, dtype=torch.long).to(device)

            candidate_scores = model.predict(user_tensor, item_tensor)

            scores = [
                (item_id, candidate_scores[i].item())
                for i, item_id in enumerate(candidate_items)
            ]
            scores.sort(key=lambda x: x[1], reverse=True)
            ranked_items = [item_id for item_id, _ in scores]

            try:
                positive_rank = ranked_items.index(positive_item) + 1
            except ValueError:
                positive_rank = len(ranked_items) + 1

            mrr = 1.0 / positive_rank
            mrr_scores.append(mrr)

            for k in k_values:
                top_k_items = ranked_items[:k]

                if positive_item in top_k_items:
                    metrics_per_k[k]['hits'] += 1

                if positive_item in top_k_items:
                    pos_in_topk = top_k_items.index(positive_item)
                    ndcg = 1.0 / np.log2(pos_in_topk + 2)
                else:
                    ndcg = 0.0
                metrics_per_k[k]['ndcgs'].append(ndcg)

            num_users += 1

    results: Dict[str, float] = {}

    for k in k_values:
        results[f'sampled_hr@{k}'] = metrics_per_k[k]['hits'] / num_users if num_users > 0 else 0.0
        results[f'sampled_ndcg@{k}'] = float(np.mean(metrics_per_k[k]['ndcgs'])) if metrics_per_k[k]['ndcgs'] else 0.0

    results['sampled_mrr'] = float(np.mean(mrr_scores)) if mrr_scores else 0.0
    results['sampled_num_negatives'] = num_negatives
    results['sampled_num_users'] = num_users

    return results
