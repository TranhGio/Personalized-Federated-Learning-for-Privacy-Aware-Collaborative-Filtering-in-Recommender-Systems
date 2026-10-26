"""Matrix Factorization training and evaluation for MovieLens 1M (single-row per-client contract).

Phase 3 Plan 03 (PSN-03, PSN-06 training-side, BSL-05-style RNG strip):

- FND-06 RNG threaded end-to-end: ``train_bpr_mf`` / ``train_basic_mf`` /
  ``evaluate_ranking_sampled`` all accept ``run_seed`` / ``user_idx`` /
  ``round_num`` / ``exclude_items`` / ``rng`` keyword-only params.
- FND-03 exclusion folded into negative-candidate pools on both the
  train and eval sides so the held-out test positive is NEVER drawn as
  either a training or an eval negative (PSN-03).
- Stdlib-random seeding / sampling / module-level import are all
  stripped from this file (and from ``client_app.py``; cross-file
  regression test in ``tests/test_task_rng.py``).
- ``_sample_negatives_seeded`` replaces ``BPRMF.sample_negatives(...)``
  (which uses process-global ``np.random``) with a flat-set rejection
  sampler drawn from an ``np.random.Generator`` instance so draws are
  deterministic across rounds without touching the stdlib RNG.

D-24 gradient masking is NOT needed under the Phase 3 single-row
contract (D-01). The only LOCAL tensors are ``local_user_row`` (shape
``(d,)``) and ``local_user_bias`` (shape ``(1,)``) — the client IS one
user, and there is no ghost table to protect from cross-row Adam
weight-decay + momentum leakage. The single-row refactor collapses the
Phase 2 D-24 problem.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Iterable, List, Optional, Set, Tuple
from collections import Counter

from federated_personalized_cf.dataset import load_partition_data
from federated_personalized_cf.models import BasicMF, BPRMF, MSELoss, BPRLoss

# Phase 3 Plan 03 (PSN-03, BSL-05-style): foundation RNG factories.
# Do NOT use Python's stdlib RNG here — task.py + client_app.py are PSN-03
# strip targets. All stochastic operations route through np_rng() /
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
        "dirichlet" (cross-silo, raises NotImplementedError per D-02) or
        "natural" (cross-device, 1 user = 1 client).

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


def _sample_negatives_seeded(
    user_rated_items: Set[int],
    num_items: int,
    num_negatives: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rejection-sampled negatives from an ``np.random.Generator`` (PSN-03 + BSL-05).

    Distribution-equivalent to uniform sampling over
    ``range(num_items) \\ user_rated_items`` but deterministic under a
    given ``rng`` instance (FND-06). Used inside ``train_bpr_mf`` to
    replace the old ``model.sample_negatives(...)`` call that used
    process-global ``np.random``.

    Under the Phase 3 single-row contract, the client IS one user, so
    ``user_rated_items`` is a FLAT set of item indices (not a dict keyed
    by user_id). The FND-03 exclusion set is expected to already be
    merged into this set by the caller (``train_bpr_mf``).

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
        rated-set; caller should size the catalog so that is rare.
    """
    out: List[int] = []
    pool = int(num_items)
    # Generous try budget so extremely dense users (rated ≈ catalog) don't
    # stall the loop. 64x base + 16 is plenty for ML-1M (3706 items with
    # most users holding <500 positives).
    max_tries = int(num_negatives) * 64 + 16
    while len(out) < int(num_negatives) and max_tries > 0:
        cand = int(rng.integers(0, pool))
        if cand not in user_rated_items:
            out.append(cand)
        max_tries -= 1
    return np.asarray(out, dtype=np.int64)


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
) -> float:
    """Train BasicMF model with MSE loss (single-row per-client contract).

    When ``proximal_mu > 0`` and ``global_params`` is provided, adds the
    FedProx proximal term: ``loss = MSE_loss + (proximal_mu / 2) * ||w -
    w_global||^2``. For split learning, the proximal term is only applied
    to GLOBAL parameters (item embeddings + item bias + global bias),
    never to the LOCAL user row.

    Phase 3 Plan 03 (PSN-03, BSL-05) wiring: ``exclude_items`` and
    ``rng`` kwargs are accepted for signature uniformity with
    :func:`train_bpr_mf` but are unused here — ``BasicMF`` optimizes MSE
    on explicit ratings and does no negative sampling.

    D-24 not needed: single-row model collapses the ghost-table problem
    (Phase 3 D-01). Only ``local_user_row`` and ``local_user_bias``
    receive gradients on the local side; no row masking or snapshot/restore
    bracket is required.

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
        List of global model parameter tensors (required if ``proximal_mu > 0``).
    global_param_names : list, optional
        List of global parameter names for split-learning FedProx. If
        provided, the proximal term only applies to these parameters.
    run_seed : Optional[int]
        FND-06 root seed. Unused in MSE training; carried for signature
        parity with ``train_bpr_mf``.
    user_idx : Optional[int]
        Client's partition_id / user_idx. Unused in MSE training.
    round_num : Optional[int]
        Current FL round (0-indexed). Unused in MSE training.
    exclude_items : Optional[Iterable[int]]
        FND-03 exclusion set. Unused (BasicMF has no negative sampling).
    rng : Optional[numpy.random.Generator]
        Unused.

    Returns
    -------
    float
        Average training loss.
    """
    # exclude_items / rng / run_seed / user_idx / round_num are accepted
    # for signature parity with train_bpr_mf but not consumed here.
    del exclude_items, rng, run_seed, user_idx, round_num

    model.to(device)
    model.train()

    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in trainloader:
            item_ids = batch['item'].to(device)
            ratings = batch['rating'].to(device)

            # Forward pass (single-row model — no user_ids argument)
            predictions = model(item_ids)

            # Compute base loss (MSE)
            loss = criterion(predictions, ratings)

            # FedProx: Add proximal term if enabled (only for global params
            # in split learning; the local user row is NEVER regularized).
            if proximal_mu > 0 and global_params is not None:
                proximal_term = 0.0
                if global_param_names is not None:
                    # Split learning: only apply to global parameters.
                    global_param_set = set(global_param_names)
                    idx = 0
                    for name, local_w in model.named_parameters():
                        if name in global_param_set:
                            proximal_term += (local_w - global_params[idx].to(device)).norm(2) ** 2
                            idx += 1
                else:
                    # Standard FedProx: apply to all parameters.
                    for local_w, global_w in zip(model.parameters(), global_params):
                        proximal_term += (local_w - global_w.to(device)).norm(2) ** 2
                loss = loss + (proximal_mu / 2) * proximal_term

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        total_loss += epoch_loss

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


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
    *,
    run_seed: Optional[int] = None,
    user_idx: Optional[int] = None,
    round_num: Optional[int] = None,
    exclude_items: Optional[Iterable[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Train BPRMF model with BPR loss (single-row per-client contract).

    Phase 3 Plan 03 adds three cross-device contract hooks:

    - **PSN-03**: ``exclude_items`` (FND-03 exclusion set from
      ``ExclusionTable.for_user(partition_id)``) is merged into
      ``user_rated_items`` so the held-out test positive is NEVER drawn
      as a training negative.
    - **BSL-05-style**: ``rng`` (FND-06 ``np_rng(run_seed, user_idx,
      round_num, "train_neg")`` instance) replaces global numpy-random
      state for negative sampling. The global numpy-random state is
      NEVER seeded (no touching of process-global RNGs); a fresh
      ``numpy.random.Generator`` drives every negative draw.
    - **D-24 not needed**: single-row model collapses the ghost-table
      problem (D-01). Only ``local_user_row`` / ``local_user_bias`` are
      LOCAL parameters; GLOBAL ``item_embeddings`` / ``item_bias`` /
      ``global_bias`` are legitimately updated on the client and then
      aggregated server-side. No row-mask or snapshot/restore bracket
      is required.

    When ``proximal_mu > 0`` the proximal term is applied ONLY to GLOBAL
    parameters — the local user row is not regularized.

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
        List of global model parameter tensors (required if ``proximal_mu > 0``).
    global_param_names : list, optional
        List of global parameter names for split-learning FedProx.
    run_seed : Optional[int]
        FND-06 root seed for the run. Used to construct the per-round
        ``np.random.Generator`` when ``rng`` is None.
    user_idx : Optional[int]
        Client's partition_id / user_idx (0..num_users-1). Part of the
        RNG namespace.
    round_num : Optional[int]
        Current FL round number.
    exclude_items : Optional[Iterable[int]]
        FND-03 exclusion set. Merged into ``user_rated_items`` so the
        held-out test positive is never drawn as a training negative.
    rng : Optional[numpy.random.Generator]
        FND-06 RNG instance. If None AND ``run_seed`` / ``user_idx`` are
        provided, one is constructed via
        ``np_rng(run_seed, user_idx, round_num or 0, "train_neg")``.

    Returns
    -------
    float
        Average training loss.
    """
    model.to(device)
    model.train()

    criterion = BPRLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # BSL-05-style: seeded RNG (never touches np.random process-global state).
    if rng is None and run_seed is not None and user_idx is not None:
        rng = np_rng(int(run_seed), int(user_idx), int(round_num or 0), "train_neg")

    # Build the single-user rated-items set. Under Phase 3 the client IS
    # one user, so user_rated_items is a flat Set[int] (not a dict).
    user_rated_items: Set[int] = set()
    for batch in trainloader:
        items = batch['item'].numpy()
        for i in items:
            user_rated_items.add(int(i))

    # PSN-03: merge the FND-03 exclusion set so the held-out test positive
    # is NEVER sampled as a training negative.
    if exclude_items is not None:
        excluded = [int(x) for x in np.asarray(list(exclude_items)).tolist()]
        user_rated_items |= set(excluded)

    num_items = int(getattr(model, "num_items", 0))

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in trainloader:
            pos_item_ids = batch['item'].to(device)
            batch_size = int(pos_item_ids.shape[0])

            # BSL-05-style negative sampling: replace stochastic
            # ``model.sample_negatives(...)`` (which uses process-global
            # ``np.random``) with a seeded ``numpy.random.Generator``
            # that draws candidates from the unrated-item set. When the
            # FND-06 rng is not supplied, fall back to the model's
            # original sampler for backwards compatibility with any
            # legacy caller that still exists.
            if rng is not None:
                if num_negatives == 1:
                    draws = _sample_negatives_seeded(
                        user_rated_items, num_items, batch_size, rng
                    )
                    neg_item_ids = torch.from_numpy(draws).to(device)
                else:
                    rows = []
                    for _ in range(batch_size):
                        rows.append(_sample_negatives_seeded(
                            user_rated_items, num_items, num_negatives, rng
                        ))
                    neg_item_ids = torch.from_numpy(np.stack(rows)).to(device)
            else:
                # Legacy backwards-compat path: ``model.sample_negatives``
                # uses process-global ``np.random`` and does not satisfy
                # FND-06 / PSN-03 — this branch exists only for callers
                # from before the contract wire-up.
                neg_item_ids = model.sample_negatives(
                    pos_item_ids,
                    num_negatives=num_negatives,
                    user_rated_items=user_rated_items,
                    sampling_strategy='uniform',
                )

            # Forward pass (single-row model — no user_ids argument).
            pos_scores, neg_scores = model(pos_item_ids, neg_item_ids)

            # Compute BPR loss
            loss = criterion(pos_scores, neg_scores)

            # FedProx: Add proximal term if enabled (only for global params
            # in split learning; the local user row is NEVER regularized).
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

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        total_loss += epoch_loss

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


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

    Phase 3 Plan 03 threads five cross-device contract kwargs into the
    underlying ``train_basic_mf`` / ``train_bpr_mf`` calls:

    - ``run_seed`` : FND-06 root seed.
    - ``user_idx`` : client's partition_id / user_idx (drives RNG namespace).
    - ``round_num`` : current FL round.
    - ``exclude_items`` : FND-03 exclusion set (``Iterable[int]`` or None).
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
        - ``global_param_names`` : Global parameter names for split-learning FedProx.
        - ``run_seed`` : FND-06 root seed (required for determinism).
        - ``user_idx`` : client's partition_id (required for RNG namespace).
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
        global_param_names=kwargs.get('global_param_names', None),
        run_seed=kwargs.get('run_seed', None),
        user_idx=kwargs.get('user_idx', None),
        round_num=kwargs.get('round_num', None),
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
    Evaluate model on test set (single-row contract — no user_ids in forward).

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
            item_ids = batch['item'].to(device)
            ratings = batch['rating'].to(device)

            # Get predictions/scores (single-row model — no user_ids)
            if model_type.lower() == "basic":
                predictions = model(item_ids)
                # Clamp to valid rating range [1, 5]
                predictions = torch.clamp(predictions, min=1.0, max=5.0)
            elif model_type.lower() == "bpr":
                # For BPR, positive-only score path (neg_item_ids=None)
                predictions = model(item_ids, neg_item_ids=None)
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
    Comprehensive all-items ranking evaluation (single-row contract).

    NOTE: Under the Phase 3 single-row contract, the client IS one user.
    The returned ``allrank_*`` metrics are aggregated across the client's
    single user (trivially one row for HR/NDCG means). This function is
    kept for cache-population side effects (item popularity) and
    diagnostic logging; it is NOT the primary evaluator (BSL-07 /
    ``fedrec_foundation.evaluator.get_primary_evaluator`` returns
    ``"sampled_loo_99"``, which is computed by ``evaluate_ranking_sampled``).

    Args:
        model: Model instance (single-row BasicMF or BPRMF)
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

    # Collect test interactions (the client is one user, but we still
    # bucket by user id for symmetry with the ghost-table evaluator).
    user_test_items: Dict[int, Set[int]] = {}
    for batch in testloader:
        users = batch['user'].numpy()
        items = batch['item'].numpy()
        for u, i in zip(users, items):
            key = int(u)
            if key not in user_test_items:
                user_test_items[key] = set()
            user_test_items[key].add(int(i))

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
                    item_popularity[int(item_id)] = count / total_interactions
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

            # Single-row model.recommend takes only top_k (no user_id)
            top_items, _ = model.recommend(top_k=max_k, exclude_items=None)
            top_items = [int(x) for x in top_items]

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
    run_seed: Optional[int] = None,
    user_idx: Optional[int] = None,
    round_num: Optional[int] = None,
    exclude_items: Optional[Iterable[int]] = None,
) -> Dict[str, float]:
    """Ranking evaluation with leave-one-out and negative sampling (single-row contract).

    Follows the evaluation protocol used in NCF, FedMF, PFedRec papers:
    - For the single client user, take ONE positive test item.
    - Sample N random negatives (items user hasn't interacted with).
    - Rank the 1 positive among (1 + N) candidates.
    - Compute HR@K, NDCG@K on the (1 + N)-candidate pool.

    Phase 3 Plan 03 (PSN-03 + BSL-05-style) wiring:

    - Global Python-stdlib RNG seeding + sampling calls are REMOVED.
      Negative sampling sources randomness from a per-round
      ``np.random.Generator`` built via
      ``np_rng(run_seed, user_idx, round_num, "eval_neg")`` (FND-06).
    - If ``exclude_items`` is provided (FND-03), it is merged into the
      per-user rated-items set before the negative-candidate pool is
      built. Under benchmark mode each client holds exactly one user so
      ``user_idx`` identifies that user; the exclusion set contains the
      union of train positives and the held-out test positive.

    Parameters
    ----------
    model : nn.Module
        Single-row model instance (BasicMF or BPRMF) — ``predict(item_ids)``
        returns scores for the client's ONE user against the candidate items.
    testloader : DataLoader
        Test data loader.
    trainloader : DataLoader
        Training data loader (to get user's train items).
    device : str
        Device ('cuda' or 'cpu').
    k_values : Optional[List[int]]
        List of K values to evaluate (default: ``[5, 10, 20]``).
    num_negatives : int
        Number of negative samples per positive (default: 99 per NCF protocol).
    seed : int
        Legacy argument kept for backwards compatibility; IGNORED. The
        new contract derives its seed from ``(run_seed, user_idx,
        round_num, "eval_neg")`` per FND-06 / BSL-05.
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
        Dictionary of sampled ranking metrics with keys like:
        ``sampled_hr@10``, ``sampled_ndcg@10``, ``sampled_mrr``,
        ``sampled_num_negatives``, ``sampled_num_users``.
    """
    # ``seed`` argument is intentionally ignored — kept for backwards
    # compatibility with any pre-Phase-3 caller. BSL-05-style forbids
    # seeding the Python stdlib RNG; seeds derive from (run_seed,
    # user_idx, round_num, "eval_neg") via np_rng.
    del seed

    if k_values is None:
        k_values = [5, 10, 20]

    model.to(device)
    model.eval()

    # Collect all items each user has interacted with (train + test).
    # Under Phase 3 single-row the client has ONE user; we still bucket
    # by user id for symmetry with the ghost-table evaluator.
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

    # Get total number of items
    num_total_items = model.num_items if hasattr(model, 'num_items') else _dataset_cache.get('num_items', 3706)
    all_items = set(range(num_total_items))

    # PSN-03: fold the foundation exclusion set into the rated-items pool
    # so the held-out test positive never appears among sampled negatives.
    excluded_set: Set[int] = set()
    if exclude_items is not None:
        excluded_set = set(int(x) for x in np.asarray(list(exclude_items)).tolist())

    # BSL-05-style: seeded RNG — one instance per-user, namespaced by
    # user_idx so concurrent clients don't share a stream. When the
    # FND-06 inputs are missing, fall back to a deterministic generator
    # so we never touch the stdlib RNG.
    if run_seed is not None and user_idx is not None:
        rng = np_rng(int(run_seed), int(user_idx), int(round_num or 0), "eval_neg")
    else:
        rng = np.random.default_rng(0)

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
            positive_item = int(test_items[0])

            # Sample negative items (items user hasn't interacted with).
            # PSN-03: union includes the foundation exclusion-set.
            all_user_items = train_items | set(test_items) | excluded_set
            negative_candidates = list(all_items - all_user_items)

            if len(negative_candidates) < num_negatives:
                # Not enough negatives, use all available
                negative_items = negative_candidates
            else:
                # BSL-05-style: replace global stdlib uniform sampling
                # with the seeded ``rng.choice(...)`` — reproducible
                # across rounds without touching the stdlib RNG.
                chosen = rng.choice(
                    np.asarray(negative_candidates, dtype=np.int64),
                    size=num_negatives,
                    replace=False,
                )
                negative_items = [int(x) for x in chosen.tolist()]

            # Candidate pool: 1 positive + N negatives
            candidate_items = [positive_item] + [int(x) for x in negative_items]

            # Score candidates against the single-client user
            # (single-row model.predict takes only item_ids).
            item_tensor = torch.tensor(candidate_items, dtype=torch.long).to(device)
            candidate_scores = model.predict(item_tensor)

            # Create (item_id, score) pairs
            scores = [
                (item_id, candidate_scores[i].item())
                for i, item_id in enumerate(candidate_items)
            ]

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
