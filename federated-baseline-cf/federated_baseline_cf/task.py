"""Matrix Factorization training and evaluation for MovieLens 1M."""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter

from federated_baseline_cf.dataset import load_partition_data
from federated_baseline_cf.models import BasicMF, BPRMF, MSELoss, BPRLoss

# Phase 2 Plan 03 imports (BSL-03, BSL-05): foundation RNG factories.
# Do NOT use Python's stdlib `random` here — task.py + client_app.py are
# BSL-05 rip targets. All stochastic operations route through np_rng() /
# torch_gen() instances seeded via the four-tier SHA-256 derivation in
# fedrec_foundation.rng (see FND-06 + CR-3).
from fedrec_foundation.rng import np_rng, torch_gen


# Global cache for dataset metadata
_dataset_cache = {}

# Global cache for item popularity (computed from training data)
_item_popularity_cache = {}


def load_data(
    partition_id: int,
    num_partitions: int,
    alpha: float = 0.5,
    test_ratio: float = 0.2,
    batch_size: int = 256,
    data_dir: str = None,
    split_mode: str = "leave-one-out",
    partition_mode: str = "dirichlet",
):
    """
    Load MovieLens 1M data for a specific partition.

    Parameters
    ----------
    partition_id : int
        ID of this client partition.
    num_partitions : int
        Total number of client partitions.
    alpha : float
        Dirichlet concentration parameter (only used when partition_mode="dirichlet").
    test_ratio : float
        Ratio of test data (only used when split_mode="random").
    batch_size : int
        Batch size for DataLoader.
    data_dir : str, optional
        Directory for data storage (defaults to project root data/).
    split_mode : str
        "leave-one-out" (NCF protocol) or "random" (legacy).
    partition_mode : str
        "dirichlet" (cross-silo) or "natural" (cross-device, 1 user = 1 client).

    Returns
    -------
    Tuple of (trainloader, testloader)
    """
    trainloader, testloader, num_users, num_items, user2idx, item2idx = load_partition_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        alpha=alpha,
        test_ratio=test_ratio,
        batch_size=batch_size,
        data_dir=data_dir,
        split_mode=split_mode,
        partition_mode=partition_mode,
    )

    # Cache metadata for model initialization
    _dataset_cache['num_users'] = num_users
    _dataset_cache['num_items'] = num_items
    _dataset_cache['user2idx'] = user2idx
    _dataset_cache['item2idx'] = item2idx

    return trainloader, testloader


def get_model(
    model_type: str = "bpr",
    num_users: int = None,
    num_items: int = None,
    embedding_dim: int = 64,
    dropout: float = 0.1,
):
    """
    Create a Matrix Factorization model.

    Args:
        model_type: "basic" for BasicMF (MSE), "bpr" for BPRMF
        num_users: Number of users (if None, uses cached value)
        num_items: Number of items (if None, uses cached value)
        embedding_dim: Embedding dimensionality (default: 64)
        dropout: Dropout rate (default: 0.1)

    Returns:
        Model instance (BasicMF or BPRMF)
    """
    # Use cached values if not provided
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
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'basic' or 'bpr'.")

    return model


def train_basic_mf(
    model: BasicMF,
    trainloader,
    epochs: int,
    lr: float,
    device: str,
    weight_decay: float = 1e-5,
    proximal_mu: float = 0.0,
    global_params: list = None,
    *,
    run_seed: int = 42,
    user_idx: int = 0,
    round_num: int = 1,
    exclude_items: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Train BasicMF model with MSE loss.

    When proximal_mu > 0 and global_params is provided, adds FedProx proximal
    term: ``loss = MSE_loss + (proximal_mu / 2) * ||w - w_global||^2``.

    Phase 2 Plan 03 (D-24) gradient-masking hook: after ``loss.backward()`` and
    before ``optimizer.step()``, zeros the gradients of ``user_embeddings.weight``
    and (if ``use_bias=True``) ``user_bias.weight`` on ALL rows except the
    client's own ``user_idx``. This preserves the D-23 "all params global"
    wire protocol while ensuring only the client's own user-row receives
    gradient updates.

    The ``exclude_items`` and ``rng`` parameters are accepted for signature
    uniformity with :func:`train_bpr_mf` but are not used here — BasicMF
    optimizes MSE on explicit ratings and does no negative sampling.

    Parameters
    ----------
    model : BasicMF
        BasicMF model instance.
    trainloader : DataLoader
        Training data loader.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    device : str
        Device to train on ('cuda' or 'cpu').
    weight_decay : float
        L2 regularization strength.
    proximal_mu : float
        FedProx proximal term coefficient (0.0 = standard training).
    global_params : list, optional
        List of global model parameters (required if proximal_mu > 0).
    run_seed : int
        Root seed for the run (FND-06).
    user_idx : int
        Global user index of this client (0..num_users-1). Drives D-24
        gradient masking and is part of the RNG namespace.
    round_num : int
        Current FL round number (0-indexed).
    exclude_items : Optional[numpy.ndarray]
        FND-03 exclusion set (unused by BasicMF — kept for signature parity).
    rng : Optional[numpy.random.Generator]
        Unused by BasicMF — kept for signature parity with train_bpr_mf.

    Returns
    -------
    float
        Average training loss.
    """
    # exclude_items / rng are unused here but referenced so linters don't
    # complain about unused args.
    del exclude_items, rng

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

            # Forward pass
            predictions = model(user_ids, item_ids)

            # Compute base loss (MSE)
            loss = criterion(predictions, ratings)

            # FedProx: Add proximal term if enabled
            if proximal_mu > 0 and global_params is not None:
                proximal_term = 0.0
                for local_w, global_w in zip(model.parameters(), global_params):
                    proximal_term += (local_w - global_w.to(device)).norm(2) ** 2
                loss = loss + (proximal_mu / 2) * proximal_term

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # D-24: zero gradients for rows other than this client's user_idx
            # (first line of defense against cross-row leakage).
            _apply_user_row_grad_mask(model, user_idx)

            # D-24: snapshot non-user rows of user_embeddings / user_bias so
            # any Adam weight-decay + momentum leak after ``optimizer.step()``
            # is reverted. Optimizer-agnostic (works for Adam + SGD).
            _user_row_snapshot = _snapshot_non_user_rows(model, user_idx)
            optimizer.step()
            _restore_non_user_rows(model, _user_row_snapshot, user_idx)

            epoch_loss += loss.item()
            num_batches += 1

        total_loss += epoch_loss

    # run_seed / round_num are part of the contract surface even when
    # unused; reference them so static analysis doesn't warn.
    _ = (run_seed, round_num)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def _apply_user_row_grad_mask(model: torch.nn.Module, user_idx: int) -> None:
    """D-24: zero gradients for all user-embedding rows except this client's.

    First line of defense: zero every non-user-idx row of
    ``user_embeddings.weight.grad`` (and ``user_bias.weight.grad`` when
    present). This stops the raw gradient contribution from flowing to
    other users' rows.

    NOTE: this alone is NOT sufficient under optimizers that decouple
    weight updates from the raw gradient — e.g. Adam's weight-decay term
    applies an L2 penalty regardless of gradient, and Adam's momentum may
    carry a non-zero step even when the current grad is zero. The
    training loop wraps ``optimizer.step()`` with
    :func:`_snapshot_non_user_rows` / :func:`_restore_non_user_rows` to
    cover those cases (Rule 1: auto-fix bugs — empirical regression
    caught during Task 1 TDD, non-zero Adam updates leaked into
    ``user_embeddings`` row 1 despite the gradient mask).

    Parameters
    ----------
    model : torch.nn.Module
        Model with a ``user_embeddings`` attribute (nn.Embedding). May also
        expose ``user_bias`` (nn.Embedding) if ``use_bias`` is True.
    user_idx : int
        Row index (0..num_users-1) that is allowed to receive gradient updates.
    """
    with torch.no_grad():
        ue = getattr(model, "user_embeddings", None)
        if ue is not None and ue.weight.grad is not None:
            mask = torch.zeros_like(ue.weight.grad)
            mask[user_idx] = 1.0
            ue.weight.grad.mul_(mask)
        ub = getattr(model, "user_bias", None)
        if ub is not None and getattr(ub, "weight", None) is not None and ub.weight.grad is not None:
            mask_b = torch.zeros_like(ub.weight.grad)
            mask_b[user_idx] = 1.0
            ub.weight.grad.mul_(mask_b)


def _snapshot_non_user_rows(
    model: torch.nn.Module, user_idx: int
) -> Dict[str, torch.Tensor]:
    """Save (cloned) non-user rows of ``user_embeddings`` / ``user_bias``.

    Paired with :func:`_restore_non_user_rows` to bracket
    ``optimizer.step()`` and hold non-user rows byte-identical across the
    step. Needed because a gradient-only mask is defeated by Adam's
    weight-decay + momentum (see :func:`_apply_user_row_grad_mask` note).

    The snapshot uses ``.detach().clone()`` so the stored tensor does not
    track autograd and survives the in-place update during
    ``optimizer.step()``.

    Parameters
    ----------
    model : torch.nn.Module
        Model with ``user_embeddings`` (and optionally ``user_bias``).
    user_idx : int
        The single row that is allowed to move in this step.

    Returns
    -------
    Dict[str, torch.Tensor]
        Keyed by ``"user_embeddings"`` / ``"user_bias"`` when present. The
        matching row ``user_idx`` is marked NaN so a miswiring that
        restores the user row too would be caught in tests. Callers
        should treat the dict as opaque and only pass it to
        :func:`_restore_non_user_rows`.
    """
    snapshots: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        ue = getattr(model, "user_embeddings", None)
        if ue is not None:
            snap = ue.weight.detach().clone()
            # Mark the user-idx row with NaN so restore does NOT write it back.
            snap[user_idx].fill_(float("nan"))
            snapshots["user_embeddings"] = snap
        ub = getattr(model, "user_bias", None)
        if ub is not None and getattr(ub, "weight", None) is not None:
            snap_b = ub.weight.detach().clone()
            snap_b[user_idx].fill_(float("nan"))
            snapshots["user_bias"] = snap_b
    return snapshots


def _restore_non_user_rows(
    model: torch.nn.Module,
    snapshots: Dict[str, torch.Tensor],
    user_idx: int,
) -> None:
    """Write the non-user rows back from the pre-step snapshot.

    Skips the user-idx row (marked NaN by :func:`_snapshot_non_user_rows`)
    so the optimizer's update to that row is preserved verbatim.

    Parameters
    ----------
    model : torch.nn.Module
    snapshots : Dict[str, torch.Tensor]
        Output of :func:`_snapshot_non_user_rows`.
    user_idx : int
        The row that is allowed to move in this step.
    """
    with torch.no_grad():
        ue = getattr(model, "user_embeddings", None)
        if ue is not None and "user_embeddings" in snapshots:
            snap = snapshots["user_embeddings"]
            # Build a mask that is True everywhere except the user-idx row.
            not_user = torch.ones(snap.shape[0], dtype=torch.bool, device=snap.device)
            not_user[user_idx] = False
            ue.weight.data[not_user] = snap[not_user]
        ub = getattr(model, "user_bias", None)
        if ub is not None and "user_bias" in snapshots:
            snap_b = snapshots["user_bias"]
            not_user_b = torch.ones(snap_b.shape[0], dtype=torch.bool, device=snap_b.device)
            not_user_b[user_idx] = False
            ub.weight.data[not_user_b] = snap_b[not_user_b]


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
    *,
    run_seed: int = 42,
    user_idx: int = 0,
    round_num: int = 1,
    exclude_items: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Train BPRMF model with BPR loss.

    When proximal_mu > 0 and global_params is provided, adds FedProx proximal
    term: ``loss = BPR_loss + (proximal_mu / 2) * ||w - w_global||^2``.

    Phase 2 Plan 03 adds three cross-device contract hooks:

    - **BSL-03**: ``exclude_items`` (FND-03 exclusion set from
      ``ExclusionTable.for_user(user_idx)``) is merged into
      ``user_rated_items`` so the held-out test positive is NEVER drawn as
      a training negative.
    - **BSL-05**: ``rng`` (FND-06 ``np_rng(run_seed, user_idx, round_num,
      "train_neg")`` instance) replaces global numpy-random state for
      negative sampling. The global numpy-random state is NEVER seeded
      (no touching of process-global RNGs); a fresh
      ``numpy.random.Generator`` drives every negative draw.
    - **D-24**: gradient masking hook zeros rows of
      ``user_embeddings.weight.grad`` (and ``user_bias.weight.grad`` when
      present) for every user OTHER than this client's ``user_idx``. The
      wire protocol (D-23) still carries all user rows, but only one row
      moves per step.

    Critical for SOTA performance (RecSys 2024):
        - Proper negative sampling
        - Correct loss implementation
        - Appropriate regularization

    Parameters
    ----------
    model : BPRMF
        BPRMF model instance.
    trainloader : DataLoader
        Training data loader.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    device : str
        Device to train on.
    weight_decay : float
        L2 regularization strength.
    num_negatives : int
        Number of negative samples per positive.
    proximal_mu : float
        FedProx proximal term coefficient (0.0 = standard training).
    global_params : list, optional
        List of global model parameters (required if proximal_mu > 0).
    run_seed : int
        Root seed for the run (FND-06).
    user_idx : int
        Global user index of this client (0..num_users-1). Drives D-24
        gradient masking and is part of the RNG namespace.
    round_num : int
        Current FL round number (0-indexed).
    exclude_items : Optional[numpy.ndarray]
        FND-03 exclusion set (int32 array of item indices) for this user.
        Merged into ``user_rated_items`` so the held-out test positive is
        never drawn as a training negative (BSL-03).
    rng : Optional[numpy.random.Generator]
        FND-06 RNG instance. If None, one is constructed via
        ``np_rng(run_seed, user_idx, round_num, "train_neg")``.

    Returns
    -------
    float
        Average training loss.
    """
    model.to(device)
    model.train()

    criterion = BPRLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # BSL-05: seeded RNG (never touches np.random process-global state).
    if rng is None:
        rng = np_rng(run_seed, user_idx, round_num, "train_neg")

    # Build user_rated_items dictionary for negative sampling
    user_rated_items: Dict[int, Set[int]] = {}
    for batch in trainloader:
        users = batch['user'].numpy()
        items = batch['item'].numpy()
        for u, i in zip(users, items):
            key = int(u)
            if key not in user_rated_items:
                user_rated_items[key] = set()
            user_rated_items[key].add(int(i))

    # BSL-03: merge the foundation exclusion set into user_rated_items so the
    # held-out test positive is NEVER sampled as a training negative.
    if exclude_items is not None and len(exclude_items) > 0:
        excluded_set: Set[int] = set(int(x) for x in np.asarray(exclude_items).tolist())
        # Apply to every known user (in cross-device this is just user_idx).
        for u in list(user_rated_items.keys()):
            user_rated_items[u] = user_rated_items[u] | excluded_set
        # Also seed the current user's set even if it had no rows in the
        # batch loop above (edge case: empty loader, still need the key).
        if int(user_idx) not in user_rated_items:
            user_rated_items[int(user_idx)] = set(excluded_set)

    num_items = int(getattr(model, "num_items", 0))

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in trainloader:
            user_ids = batch['user'].to(device)
            pos_item_ids = batch['item'].to(device)

            # BSL-05 negative sampling: replace stochastic ``model.sample_negatives``
            # (which uses process-global ``np.random``) with a seeded
            # ``numpy.random.Generator`` that draws candidates and rejects
            # anything in ``user_rated_items`` / ``exclude_items``. Equivalent
            # distribution to uniform sampling from the unrated-item set.
            neg_item_ids = _sample_negatives_seeded(
                user_ids=user_ids.cpu().numpy(),
                num_items=num_items,
                num_negatives=num_negatives,
                user_rated_items=user_rated_items,
                rng=rng,
                device=device,
            )

            # Forward pass
            pos_scores, neg_scores = model(user_ids, pos_item_ids, neg_item_ids)

            # Compute BPR loss
            loss = criterion(pos_scores, neg_scores)

            # FedProx: Add proximal term if enabled
            if proximal_mu > 0 and global_params is not None:
                proximal_term = 0.0
                for local_w, global_w in zip(model.parameters(), global_params):
                    proximal_term += (local_w - global_w.to(device)).norm(2) ** 2
                loss = loss + (proximal_mu / 2) * proximal_term

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # D-24: zero gradients for every user-embedding row except this
            # client's ``user_idx`` (first line of defense; does not cover
            # Adam weight-decay + momentum leaking to other rows).
            _apply_user_row_grad_mask(model, user_idx)

            # D-24: snapshot + restore non-user rows around ``optimizer.step()``
            # so Adam's weight-decay + momentum cannot move rows whose
            # gradients we just zeroed. Preserves the D-23 all-params-global
            # wire protocol while ensuring only one row moves per step.
            _user_row_snapshot = _snapshot_non_user_rows(model, user_idx)
            optimizer.step()
            _restore_non_user_rows(model, _user_row_snapshot, user_idx)

            epoch_loss += loss.item()
            num_batches += 1

        total_loss += epoch_loss

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def _sample_negatives_seeded(
    user_ids: np.ndarray,
    num_items: int,
    num_negatives: int,
    user_rated_items: Dict[int, Set[int]],
    rng: np.random.Generator,
    device,
) -> torch.Tensor:
    """Deterministic negative sampling using an ``np.random.Generator`` (BSL-05).

    Equivalent in distribution to ``BPRMF.sample_negatives(..., 'uniform')``
    but sources randomness from the provided ``rng`` so the draws are
    reproducible across rounds without touching ``np.random`` global state.

    Parameters
    ----------
    user_ids : numpy.ndarray
        Batch user indices, shape ``(batch_size,)``.
    num_items : int
        Size of the item catalog.
    num_negatives : int
        Number of negatives to draw per positive.
    user_rated_items : Dict[int, Set[int]]
        Excluded items per user (rated + foundation exclusion-set).
    rng : numpy.random.Generator
        FND-06 RNG instance.
    device : torch.device
        Target device for the returned tensor.

    Returns
    -------
    torch.Tensor
        ``shape=(batch_size,)`` for ``num_negatives == 1`` or
        ``shape=(batch_size, num_negatives)`` otherwise.
    """
    batch_size = int(user_ids.shape[0])

    if num_negatives == 1:
        out = np.empty(batch_size, dtype=np.int64)
        for b in range(batch_size):
            rated = user_rated_items.get(int(user_ids[b]), set())
            # Rejection sample; draws from the uniform item distribution.
            while True:
                cand = int(rng.integers(0, num_items))
                if cand not in rated:
                    out[b] = cand
                    break
        return torch.from_numpy(out).to(device)

    out_multi = np.empty((batch_size, num_negatives), dtype=np.int64)
    for b in range(batch_size):
        rated = user_rated_items.get(int(user_ids[b]), set())
        drawn: Set[int] = set()
        i = 0
        while i < num_negatives:
            cand = int(rng.integers(0, num_items))
            if cand in rated or cand in drawn:
                continue
            out_multi[b, i] = cand
            drawn.add(cand)
            i += 1
    return torch.from_numpy(out_multi).to(device)


def train(
    model,
    trainloader,
    epochs: int,
    lr: float,
    device: str,
    model_type: str = "bpr",
    **kwargs
) -> float:
    """Unified training function for both model types.

    Phase 2 Plan 03 threads five cross-device contract kwargs into the
    underlying ``train_basic_mf`` / ``train_bpr_mf`` calls:

    - ``run_seed`` : FND-06 root seed.
    - ``user_idx`` : client's user index (drives RNG namespace + D-24 mask).
    - ``round_num`` : current FL round.
    - ``exclude_items`` : FND-03 exclusion set (``np.ndarray`` or None).
    - ``rng`` : optional pre-constructed ``np.random.Generator``.

    Parameters
    ----------
    model : BasicMF or BPRMF
        Model instance.
    trainloader : DataLoader
        Training data loader.
    epochs : int
        Number of epochs.
    lr : float
        Learning rate.
    device : str
        Device ('cuda' or 'cpu').
    model_type : str
        "basic" or "bpr".
    **kwargs
        Additional arguments:

        - ``weight_decay`` : L2 regularization strength.
        - ``num_negatives`` : Number of negative samples (BPR only).
        - ``proximal_mu`` : FedProx proximal term coefficient.
        - ``global_params`` : Global model parameters for FedProx.
        - ``run_seed`` : FND-06 root seed (required for determinism).
        - ``user_idx`` : client's user index (required for D-24 mask + RNG).
        - ``round_num`` : current FL round (required for RNG namespace).
        - ``exclude_items`` : FND-03 exclusion set (optional).
        - ``rng`` : pre-constructed ``np.random.Generator`` (optional).

    Returns
    -------
    float
        Average training loss.
    """
    common = dict(
        weight_decay=kwargs.get('weight_decay', 1e-5),
        proximal_mu=kwargs.get('proximal_mu', 0.0),
        global_params=kwargs.get('global_params', None),
        run_seed=int(kwargs.get('run_seed', 42)),
        user_idx=int(kwargs.get('user_idx', 0)),
        round_num=int(kwargs.get('round_num', 1)),
        exclude_items=kwargs.get('exclude_items', None),
        rng=kwargs.get('rng', None),
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
    """
    Evaluate model on test set.

    Computes:
        - Loss (MSE or BPR depending on model type)
        - RMSE (Root Mean Squared Error)
        - MAE (Mean Absolute Error)

    Args:
        model: Model instance
        testloader: Test data loader
        device: Device
        model_type: "basic" or "bpr"

    Returns:
        Tuple of (loss, metrics_dict)
        where metrics_dict contains {'rmse': float, 'mae': float}
    """
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

            # Get predictions/scores
            if model_type.lower() == "basic":
                predictions = model(user_ids, item_ids)
                # Clamp to valid rating range [1, 5]
                predictions = torch.clamp(predictions, min=1.0, max=5.0)
            elif model_type.lower() == "bpr":
                # For BPR, get scores (not clamped)
                predictions = model(user_ids, item_ids, neg_item_ids=None)
                # For evaluation, can clamp to rating range
                predictions = torch.clamp(predictions, min=1.0, max=5.0)

            # Compute errors
            squared_errors = (predictions - ratings) ** 2
            absolute_errors = torch.abs(predictions - ratings)

            total_squared_error += squared_errors.sum().item()
            total_absolute_error += absolute_errors.sum().item()
            num_samples += len(ratings)

            # Compute loss based on model type
            if model_type.lower() == "basic":
                criterion = MSELoss()
                loss = criterion(predictions, ratings)
            else:
                # For BPR, use MSE for evaluation (common practice)
                mse = squared_errors.mean()
                loss = mse

            total_loss += loss.item() * len(ratings)

    # Compute metrics
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    rmse = np.sqrt(total_squared_error / num_samples) if num_samples > 0 else 0.0
    mae = total_absolute_error / num_samples if num_samples > 0 else 0.0

    metrics = {
        'rmse': rmse,
        'mae': mae,
    }

    return avg_loss, metrics


def compute_ndcg(ranked_items, relevant_items, k):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at K.

    NDCG measures ranking quality with position discounting.
    Score = DCG / IDCG where:
    - DCG = sum(rel_i / log2(i+1)) for i in top-K
    - IDCG = ideal DCG (perfect ranking)

    Args:
        ranked_items: List of recommended item IDs (in rank order)
        relevant_items: Set of relevant (ground truth) item IDs
        k: Cutoff position

    Returns:
        NDCG@K score (0 to 1, higher is better)
    """
    # DCG calculation
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in relevant_items:
            # Relevance = 1 (binary relevance)
            # Position discount: 1/log2(rank+1), where rank starts at 1
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0

    # IDCG calculation (ideal ranking - all relevant items first)
    num_relevant = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))

    # Normalize
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_mrr(ranked_items, relevant_items):
    """
    Compute Mean Reciprocal Rank (MRR) for a single user.

    MRR = 1 / rank_of_first_relevant_item

    Args:
        ranked_items: List of recommended item IDs (in rank order)
        relevant_items: Set of relevant (ground truth) item IDs

    Returns:
        Reciprocal rank (1/rank if hit found, 0 otherwise)
    """
    for i, item in enumerate(ranked_items):
        if item in relevant_items:
            return 1.0 / (i + 1)  # i+1 because rank starts at 1
    return 0.0


def compute_ap(ranked_items, relevant_items, k: int) -> float:
    """
    Compute Average Precision at K for a single user.

    AP@K = (1/min(K, |relevant|)) * sum(P(i) * rel(i)) for i in 1..K
    where P(i) is precision at position i, and rel(i) is 1 if item at i is relevant.

    Args:
        ranked_items: List of recommended item IDs (in rank order)
        relevant_items: Set of relevant (ground truth) item IDs
        k: Cutoff position

    Returns:
        Average Precision at K (0 to 1, higher is better)
    """
    if not relevant_items:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for i, item in enumerate(ranked_items[:k]):
        if item in relevant_items:
            hits += 1
            # Precision at this position
            precision_sum += hits / (i + 1)

    # Normalize by minimum of K and number of relevant items
    return precision_sum / min(k, len(relevant_items))


def compute_novelty(
    ranked_items,
    item_popularity: Dict[int, float],
    k: int,
) -> float:
    """
    Compute Novelty at K for a single user's recommendations.

    Novelty = average of -log2(popularity) for recommended items.
    Higher novelty means recommending less popular (more surprising) items.

    Args:
        ranked_items: List of recommended item IDs (in rank order)
        item_popularity: Dict mapping item_id -> popularity (0 to 1)
        k: Cutoff position

    Returns:
        Average novelty score (higher = more novel/surprising)
    """
    if len(ranked_items) == 0:
        return 0.0

    novelties = []
    for item in ranked_items[:k]:
        pop = item_popularity.get(item, 1e-10)  # Avoid log(0)
        # Self-information: -log2(p) where p is popularity
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
    """
    Comprehensive ranking evaluation with multiple metrics.

    Computes for each K in k_values:
        - Hit Rate@K: Fraction of users with at least one hit in top-K
        - Precision@K: Average fraction of relevant items in top-K
        - Recall@K: Average fraction of relevant items retrieved
        - F1@K: Harmonic mean of Precision and Recall
        - NDCG@K: Normalized Discounted Cumulative Gain (ranking quality)
        - MAP@K: Mean Average Precision at K
        - Coverage@K: Fraction of catalog items appearing in recommendations
        - Novelty@K: Average inverse popularity of recommended items
        - MRR: Mean Reciprocal Rank (position of first relevant item)
        - Accuracy@K: Same as Hit Rate (binary hit/miss)

    Args:
        model: Model instance (BasicMF or BPRMF)
        testloader: Test data loader
        device: Device ('cuda' or 'cpu')
        k_values: List of K values to evaluate (default: [5, 10, 20])
        item_popularity: Dict mapping item_id -> popularity (0 to 1).
            If None and trainloader provided, computed from trainloader.
        trainloader: Training data loader (used to compute item_popularity if not provided)

    Returns:
        Dictionary of ranking metrics with keys like:
        - 'hit_rate@5', 'precision@10', 'ndcg@20', 'mrr', etc.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    model.to(device)
    model.eval()

    # Collect test interactions per user
    user_test_items = {}
    for batch in testloader:
        users = batch['user'].numpy()
        items = batch['item'].numpy()
        for u, i in zip(users, items):
            if u not in user_test_items:
                user_test_items[u] = set()
            user_test_items[u].add(i)

    # Compute item popularity from training data if not provided
    if item_popularity is None:
        item_popularity = {}
        if trainloader is not None:
            item_counts = Counter()
            total_interactions = 0
            for batch in trainloader:
                items = batch['item'].numpy()
                item_counts.update(items)
                total_interactions += len(items)
            # Normalize to get popularity (fraction of interactions)
            if total_interactions > 0:
                for item_id, count in item_counts.items():
                    item_popularity[item_id] = count / total_interactions
        # Cache for future use
        _item_popularity_cache.update(item_popularity)

    # Get number of items for coverage calculation
    num_total_items = model.num_items if hasattr(model, 'num_items') else _dataset_cache.get('num_items', 3706)

    # Initialize metric accumulators for each K
    metrics_per_k = {k: {
        'hits': 0,
        'precisions': [],
        'recalls': [],
        'f1s': [],
        'ndcgs': [],
        'aps': [],  # Average Precision scores
        'novelties': [],
        'recommended_items': set(),  # For coverage
    } for k in k_values}

    mrr_scores = []
    num_users = 0
    max_k = max(k_values)

    with torch.no_grad():
        for user_id in user_test_items.keys():
            # Get test items for this user
            test_items = user_test_items[user_id]

            # Get top-MAX_K recommendations (we'll slice for different K)
            top_items, _ = model.recommend(user_id, top_k=max_k, exclude_items=None)

            # Compute MRR (only once per user, independent of K)
            mrr = compute_mrr(top_items, test_items)
            mrr_scores.append(mrr)

            # Compute metrics for each K value
            for k in k_values:
                top_k_items = top_items[:k]

                # Compute hits
                hits_for_user = len(set(top_k_items) & test_items)
                if hits_for_user > 0:
                    metrics_per_k[k]['hits'] += 1

                # Compute precision and recall
                precision = hits_for_user / k if k > 0 else 0
                recall = hits_for_user / len(test_items) if len(test_items) > 0 else 0

                # Compute F1@K (harmonic mean of precision and recall)
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0

                # Compute NDCG@K
                ndcg = compute_ndcg(top_k_items, test_items, k)

                # Compute AP@K (Average Precision)
                ap = compute_ap(top_k_items, test_items, k)

                # Compute Novelty@K
                novelty = compute_novelty(top_k_items, item_popularity, k)

                # Track recommended items for coverage
                metrics_per_k[k]['recommended_items'].update(top_k_items)

                metrics_per_k[k]['precisions'].append(precision)
                metrics_per_k[k]['recalls'].append(recall)
                metrics_per_k[k]['f1s'].append(f1)
                metrics_per_k[k]['ndcgs'].append(ndcg)
                metrics_per_k[k]['aps'].append(ap)
                metrics_per_k[k]['novelties'].append(novelty)

            num_users += 1

    # Aggregate metrics
    results = {}

    for k in k_values:
        # Hit Rate@K (also called Accuracy@K in some literature)
        results[f'hit_rate@{k}'] = metrics_per_k[k]['hits'] / num_users if num_users > 0 else 0.0
        results[f'accuracy@{k}'] = results[f'hit_rate@{k}']  # Same metric, different name

        # Precision@K
        results[f'precision@{k}'] = float(np.mean(metrics_per_k[k]['precisions'])) if metrics_per_k[k]['precisions'] else 0.0

        # Recall@K
        results[f'recall@{k}'] = float(np.mean(metrics_per_k[k]['recalls'])) if metrics_per_k[k]['recalls'] else 0.0

        # F1@K
        results[f'f1@{k}'] = float(np.mean(metrics_per_k[k]['f1s'])) if metrics_per_k[k]['f1s'] else 0.0

        # NDCG@K
        results[f'ndcg@{k}'] = float(np.mean(metrics_per_k[k]['ndcgs'])) if metrics_per_k[k]['ndcgs'] else 0.0

        # MAP@K (Mean Average Precision)
        results[f'map@{k}'] = float(np.mean(metrics_per_k[k]['aps'])) if metrics_per_k[k]['aps'] else 0.0

        # Coverage@K (fraction of catalog items recommended)
        results[f'coverage@{k}'] = len(metrics_per_k[k]['recommended_items']) / num_total_items if num_total_items > 0 else 0.0

        # Novelty@K (average inverse popularity)
        results[f'novelty@{k}'] = float(np.mean(metrics_per_k[k]['novelties'])) if metrics_per_k[k]['novelties'] else 0.0

    # MRR (not K-dependent)
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
    run_seed: int = 42,
    user_idx: int = 0,
    round_num: int = 1,
    exclude_items: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Ranking evaluation with leave-one-out and negative sampling.

    This follows the evaluation protocol used in NCF, FedMF, PFedRec papers:
    - For each user, take ONE positive test item
    - Sample N random negatives (items user hasn't interacted with)
    - Rank the 1 positive among (1 + N) candidates
    - Compute HR@K, NDCG@K on this smaller candidate pool

    When used with leave-one-out data split, each user has exactly 1 test
    item (their last interaction by timestamp), matching the NCF protocol
    exactly.

    Phase 2 Plan 03 (BSL-03, BSL-05) wiring:

    - Global Python-stdlib RNG seeding + sampling calls are REMOVED.
      Negative sampling sources randomness from a per-round
      ``np.random.Generator`` built via
      ``np_rng(run_seed, user_idx, round_num, "eval_neg")`` (FND-06).
    - If ``exclude_items`` is provided (FND-03), it is merged into the
      per-user rated-items set before building the negative-candidate
      pool. Under benchmark mode each client holds exactly one user so
      ``user_idx`` identifies that user; the exclusion set contains the
      union of train positives and the held-out test positive.

    Parameters
    ----------
    model : nn.Module
        Model instance (BasicMF or BPRMF).
    testloader : DataLoader
        Test data loader.
    trainloader : DataLoader
        Training data loader (to get user's train items).
    device : str
        Device ('cuda' or 'cpu').
    k_values : Optional[List[int]]
        List of K values to evaluate (default: [5, 10, 20]).
    num_negatives : int
        Number of negative samples per positive (default: 99).
    seed : int
        Legacy argument kept for backwards compatibility; IGNORED. The
        new contract derives its seed from (run_seed, user_idx, round_num,
        "eval_neg") per FND-06 / BSL-05.
    run_seed : int
        FND-06 root seed. Typically ``context.run_config["run-seed"]``.
    user_idx : int
        Client's user index (drives the per-user RNG namespace).
    round_num : int
        Current FL round number.
    exclude_items : Optional[numpy.ndarray]
        FND-03 exclusion set (int32 array of item indices). Merged into
        rated-items so the held-out test positive never appears as a
        sampled negative in eval.

    Returns
    -------
    Dict[str, float]
        Dictionary of sampled ranking metrics with keys like:
        - 'sampled_hr@10', 'sampled_ndcg@10', etc.
    """
    # ``seed`` argument is intentionally ignored — kept for backwards
    # compatibility with any pre-Phase-2 caller. BSL-05 forbids seeding
    # the Python stdlib RNG; seeds derive from (run_seed, user_idx,
    # round_num, "eval_neg") via np_rng.
    del seed

    if k_values is None:
        k_values = [5, 10, 20]

    model.to(device)
    model.eval()

    # Collect all items each user has interacted with (train + test)
    user_train_items: Dict[int, Set[int]] = {}
    for batch in trainloader:
        users = batch['user'].numpy()
        items = batch['item'].numpy()
        for u, i in zip(users, items):
            key = int(u)
            if key not in user_train_items:
                user_train_items[key] = set()
            user_train_items[key].add(int(i))

    # Collect test items per user
    # With leave-one-out split, each user has exactly 1 test item
    user_test_items: Dict[int, List[int]] = {}
    for batch in testloader:
        users = batch['user'].numpy()
        items = batch['item'].numpy()
        for u, i in zip(users, items):
            key = int(u)
            if key not in user_test_items:
                user_test_items[key] = []
            user_test_items[key].append(int(i))

    # Get total number of items
    num_total_items = model.num_items if hasattr(model, 'num_items') else _dataset_cache.get('num_items', 3706)
    all_items = set(range(num_total_items))

    # BSL-03: fold the foundation exclusion set into the rated-items pool
    # so the held-out test positive never appears among sampled negatives.
    excluded_set: Set[int] = set()
    if exclude_items is not None and len(exclude_items) > 0:
        excluded_set = set(int(x) for x in np.asarray(exclude_items).tolist())

    # BSL-05: seeded RNG — one instance per-user, namespaced by user_idx
    # so concurrent clients don't share a stream.
    rng = np_rng(run_seed, user_idx, round_num, "eval_neg")

    # Initialize metric accumulators for each K
    metrics_per_k = {k: {
        'hits': 0,
        'ndcgs': [],
    } for k in k_values}

    mrr_scores = []
    num_users = 0

    with torch.no_grad():
        for user_id in sorted(user_test_items.keys()):
            test_items = user_test_items[user_id]
            train_items = user_train_items.get(user_id, set())

            if len(test_items) == 0:
                continue

            # Use the single held-out item (leave-one-out gives exactly 1)
            positive_item = test_items[0]

            # Sample negative items (items user hasn't interacted with).
            # BSL-03: union includes the foundation exclusion-set.
            all_user_items = train_items | set(test_items) | excluded_set
            negative_candidates = list(all_items - all_user_items)

            if len(negative_candidates) < num_negatives:
                # Not enough negatives, use all available
                negative_items = negative_candidates
            else:
                # BSL-05: replace global stdlib uniform sampling with the
                # seeded ``rng.choice(...)`` — reproducible across rounds
                # without touching the stdlib RNG process-global state.
                chosen = rng.choice(
                    np.asarray(negative_candidates, dtype=np.int64),
                    size=num_negatives,
                    replace=False,
                )
                negative_items = [int(x) for x in chosen.tolist()]

            # Candidate pool: 1 positive + N negatives
            candidate_items = [int(positive_item)] + [int(x) for x in negative_items]

            # Get scores for all candidates (batch processing for efficiency)
            user_tensor = torch.tensor([user_id] * len(candidate_items), dtype=torch.long).to(device)
            item_tensor = torch.tensor(candidate_items, dtype=torch.long).to(device)

            # BSL-EVAL-LEAK FIX (debug session: baseline-eval-leakage):
            # Use raw forward() scores for ranking — NOT model.predict().
            # BasicMF.predict() clamps outputs to [1.0, 5.0] for rating-prediction
            # (RMSE/MAE), which collapses every candidate to the same value when
            # the model's pre-clamp scores fall below 1.0 (the typical regime
            # under Xavier init or early training). Tied scores then sort by
            # Python's stable list.sort, leaving the positive (always at input
            # index 0) at rank 1 → degenerate HR@10 = NDCG@10 = MRR = 1.0.
            # BPRMF.forward signature differs (takes neg_item_ids); BasicMF.forward
            # takes only (user, item). Class name dispatch is the simplest
            # signature-safe way to call forward() raw.
            if type(model).__name__ == "BPRMF":
                candidate_scores = model(user_tensor, item_tensor, neg_item_ids=None)
            else:
                candidate_scores = model(user_tensor, item_tensor)

            # Create (item_id, score) pairs
            scores = [(item_id, candidate_scores[i].item()) for i, item_id in enumerate(candidate_items)]

            # Sort by score (descending)
            scores.sort(key=lambda x: x[1], reverse=True)
            ranked_items = [item_id for item_id, _ in scores]

            # Find rank of positive item
            try:
                positive_rank = ranked_items.index(positive_item) + 1  # 1-indexed
            except ValueError:
                positive_rank = len(ranked_items) + 1

            # Compute MRR
            mrr = 1.0 / positive_rank
            mrr_scores.append(mrr)

            # Compute metrics for each K
            for k in k_values:
                top_k_items = ranked_items[:k]

                # Hit@K: is positive item in top-K?
                if positive_item in top_k_items:
                    metrics_per_k[k]['hits'] += 1

                # NDCG@K: with single relevant item
                if positive_item in top_k_items:
                    pos_in_topk = top_k_items.index(positive_item)
                    ndcg = 1.0 / np.log2(pos_in_topk + 2)  # +2 because index is 0-based
                else:
                    ndcg = 0.0
                metrics_per_k[k]['ndcgs'].append(ndcg)

            num_users += 1

    # Aggregate metrics with 'sampled_' prefix
    results = {}

    for k in k_values:
        # Hit Rate@K (sampled)
        results[f'sampled_hr@{k}'] = metrics_per_k[k]['hits'] / num_users if num_users > 0 else 0.0

        # NDCG@K (sampled)
        results[f'sampled_ndcg@{k}'] = float(np.mean(metrics_per_k[k]['ndcgs'])) if metrics_per_k[k]['ndcgs'] else 0.0

    # MRR (sampled)
    results['sampled_mrr'] = float(np.mean(mrr_scores)) if mrr_scores else 0.0

    # Add metadata
    results['sampled_num_negatives'] = num_negatives
    results['sampled_num_users'] = num_users

    return results
