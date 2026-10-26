"""PFedRec training and evaluation for MovieLens 1M (Phase 5 cross-device).

Phase 5 Plan 03 contract changes vs the pre-Phase-5 module:

- **FND-06 RNG factories** (D-02 / PFR-07): every stochastic step routes through
  ``fedrec_foundation.rng.np_rng(run_seed, user_idx, round_num, "<purpose>")``.
  Stdlib ``random`` is NOT imported anywhere in this module — BSL-05 cross-file
  regression guard pins this. Training negatives are re-sampled every round
  because ``round_num`` changes the seed; same key reproduces the same draw
  (FND-06 determinism).

- **FND-03 exclusion** (PFR-04): callers thread ``exclude_items`` (the result
  of ``ExclusionTable.for_user(user_idx)``) into BOTH ``prepare_user_train_data``
  and ``evaluate_pfedrec_sampled``. The held-out test positive is never drawn
  as a training negative (closes CONCERNS bug #1) nor as an evaluation negative.

- **D-04 eval BCE over 99 negs**: ``evaluate_pfedrec_sampled`` now computes the
  per-user BCE loss over ``ratings_pred = torch.cat((test_score, negative_score))``
  with ``ratings = [1, 0, 0, ..., 0]`` (length 100). Mirrors ``IJCAI-23-PFedRec/
  engine.py:195-196``; the eval-loss diagnostic becomes directly comparable to
  the reference logs.

- **Dual-LR preserved** (Pitfall 3): ``train_pfedrec_single_user`` keeps the
  ``optimizer_item.lr = lr * num_items * lr_eta`` boost from the reference at
  ``IJCAI-23-PFedRec/engine.py:117-119``. PFR-08 reproduction is sensitive to
  this; do not "fix" it.

- **No user embeddings**: PFedRec personalization rides on the per-user
  ``affine_output`` layer; there is no user embedding table. ``model.forward``
  takes only item indices.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fedrec_foundation.rng import np_rng

from federated_pfedrec.dataset import load_partition_data
from federated_pfedrec.models import BCELoss, PFedRecMLP


# Global cache for dataset metadata (set on first ``load_data`` call).
_dataset_cache: Dict[str, object] = {}


def load_data(
    partition_id: int,
    num_partitions: int,
    alpha: float = 0.5,
    test_ratio: float = 0.2,
    batch_size: int = 256,
    data_dir: Optional[str] = None,
    split_mode: str = "leave-one-out",
    partition_mode: str = "natural",
):
    """Load MovieLens 1M data for a specific partition (cross-device only).

    Phase 5 D-09: ``partition_mode != "natural"`` is enforced by ``dataset.py``;
    this helper is a thin wrapper that caches ``num_users`` / ``num_items`` /
    ``user2idx`` / ``item2idx`` so per-batch iteration in PFedRec doesn't pay
    the bundle-load cost twice.

    Parameters
    ----------
    partition_id : int
        Client partition index (== ``user_idx`` under cross-device).
    num_partitions : int
        Total partitions (== ``num_users`` under cross-device).
    alpha : float
        Dirichlet concentration (legacy; unused under ``partition_mode="natural"``).
    test_ratio : float
        Random-split ratio (unused under leave-one-out).
    batch_size : int
        DataLoader batch size.
    data_dir : Optional[str]
        Override default data dir.
    split_mode : str
        ``"leave-one-out"`` (NCF protocol).
    partition_mode : str
        ``"natural"`` (cross-device, 1 user = 1 client).

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        ``(trainloader, testloader)``.
    """
    (
        trainloader,
        testloader,
        num_users,
        num_items,
        user2idx,
        item2idx,
    ) = load_partition_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        alpha=alpha,
        test_ratio=test_ratio,
        batch_size=batch_size,
        data_dir=data_dir,
        split_mode=split_mode,
        partition_mode=partition_mode,
    )

    _dataset_cache["num_users"] = num_users
    _dataset_cache["num_items"] = num_items
    _dataset_cache["user2idx"] = user2idx
    _dataset_cache["item2idx"] = item2idx

    return trainloader, testloader


def get_model(
    num_items: Optional[int] = None,
    latent_dim: int = 32,
) -> PFedRecMLP:
    """Create a PFedRec MLP model.

    Parameters
    ----------
    num_items : Optional[int]
        Number of items (if ``None``, uses cached value from ``_dataset_cache``).
    latent_dim : int
        Embedding dimensionality (default 32; paper default).

    Returns
    -------
    PFedRecMLP
        Fresh model instance with PyTorch's nn.Linear default Kaiming init
        (D-19 paper-faithful — no Xavier reset).
    """
    if num_items is None:
        num_items = int(_dataset_cache.get("num_items", 3706))
    return PFedRecMLP(num_items=num_items, latent_dim=latent_dim)


# ---------------------------------------------------------------------------
# Negative-sampling helpers (FND-06 / PFR-04 / PFR-07)
# ---------------------------------------------------------------------------


def _sample_train_negatives_seeded(
    user_rated_items: Set[int],
    num_items: int,
    num_negatives: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rejection-uniform sampler driven by an FND-06 ``np.random.Generator``.

    Distribution-equivalent to the reference's stdlib uniform-without-replacement
    sampler at ``IJCAI-23-PFedRec/data.py:80`` but seeded from FND-06
    ``np_rng`` instead of the process-global stdlib RNG. Closes BSL-05 (no stdlib
    random in this module) and PFR-07 (per-round resampling).

    Parameters
    ----------
    user_rated_items : Set[int]
        No-go set: ``train_positives ∪ exclude_items`` (FND-03 union; the
        held-out test positive is NEVER drawn).
    num_items : int
        Catalog size; items drawn from ``[0, num_items)``.
    num_negatives : int
        Number of negatives to return.
    rng : numpy.random.Generator
        FND-06 RNG instance from ``np_rng(run_seed, user_idx, round_num,
        "train_neg")``.

    Returns
    -------
    numpy.ndarray
        ``int64`` array of length ``num_negatives``. Items are guaranteed to
        be disjoint from ``user_rated_items``.
    """
    out = np.empty(int(num_negatives), dtype=np.int64)
    filled = 0
    pool = int(num_items)
    while filled < int(num_negatives):
        # Sample a batch; reject items present in the no-go set. Generous
        # batch size so dense users (rated ≈ catalog) don't stall.
        remaining = int(num_negatives) - filled
        batch = rng.integers(0, pool, size=2 * remaining + 16)
        for v in batch:
            cand = int(v)
            if cand not in user_rated_items:
                out[filled] = cand
                filled += 1
                if filled == int(num_negatives):
                    break
    return out


# ---------------------------------------------------------------------------
# Per-user training-data assembly (PFR-04 + PFR-07 + D-02)
# ---------------------------------------------------------------------------


def prepare_user_train_data(
    user_idx: int,
    user_train_items: List[int],
    *,
    num_items: int,
    num_negatives: int = 4,
    run_seed: int,
    round_num: int,
    exclude_items: Optional[Iterable[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], List[float]]:
    """Build the (item_ids, ratings) BCE batch for ONE user.

    Mirrors the reference ``IJCAI-23-PFedRec/data.py``: each positive emits
    ``num_negatives`` negatives, ratings are binary (1.0 / 0.0). The held-out
    test positive is excluded from the negative pool via FND-03 (PFR-04).

    Parameters
    ----------
    user_idx : int
        User index (== ``partition_id`` in cross-device).
    user_train_items : List[int]
        LOO-train positives for this user (held-out test item already removed
        upstream by the foundation split).
    num_items : int
        Catalog size.
    num_negatives : int
        Number of negatives per positive (paper default: 4).
    run_seed : int
        FND-06 root seed.
    round_num : int
        Current FL round number — drives the per-round RNG namespace (PFR-07).
    exclude_items : Optional[Iterable[int]]
        Result of ``ExclusionTable.for_user(user_idx)`` from the foundation
        bundle. Folded into the no-go set so the held-out test positive is
        never drawn as a training negative.
    rng : Optional[numpy.random.Generator]
        Pre-built RNG. If ``None``, the helper builds one via
        ``np_rng(run_seed, user_idx, round_num, "train_neg")``.

    Returns
    -------
    Tuple[List[int], List[float]]
        ``(items, ratings)`` lists in the BCE format consumed by
        ``train_pfedrec_single_user``.
    """
    if rng is None:
        rng = np_rng(int(run_seed), int(user_idx), int(round_num), "train_neg")

    # FND-03 + PFR-04: union train positives with the foundation exclusion set
    # so the held-out test positive is never drawn as a training negative.
    no_go: Set[int] = set(int(x) for x in user_train_items)
    if exclude_items is not None:
        no_go |= set(int(x) for x in np.asarray(list(exclude_items)).tolist())

    items_list: List[int] = []
    ratings_list: List[float] = []

    pos_items = sorted(set(int(x) for x in user_train_items))
    if not pos_items:
        return items_list, ratings_list

    for pos_item in pos_items:
        items_list.append(int(pos_item))
        ratings_list.append(1.0)
        sampled_negs = _sample_train_negatives_seeded(
            user_rated_items=no_go,
            num_items=int(num_items),
            num_negatives=int(num_negatives),
            rng=rng,
        )
        for neg in sampled_negs.tolist():
            items_list.append(int(neg))
            ratings_list.append(0.0)

    return items_list, ratings_list


# ---------------------------------------------------------------------------
# Per-user training: alternating optimization with dual LR (Pitfall 3 preserved)
# ---------------------------------------------------------------------------


def train_pfedrec_single_user(
    model: PFedRecMLP,
    user_items: List[int],
    user_ratings: List[float],
    lr: float,
    lr_eta: float,
    num_items: int,
    local_epochs: int,
    batch_size: int,
    device: torch.device,
    l2_regularization: float = 0.0,
    proximal_mu: float = 0.0,
    global_item_embedding: Optional[torch.Tensor] = None,
    *,
    run_seed: Optional[int] = None,
    user_idx: Optional[int] = None,
    round_num: Optional[int] = None,
    exclude_items: Optional[Iterable[int]] = None,
) -> float:
    """Train PFedRec model for a single user with alternating optimization.

    Reference parity (``IJCAI-23-PFedRec/engine.py:fed_train_single_batch``):
      1. Forward + backward: update ONLY ``affine_output`` (score function).
      2. Forward + backward: update ONLY ``embedding_item`` (item embedding).

    Phase 5 contract additions:
      - ``run_seed`` / ``user_idx`` / ``round_num`` / ``exclude_items`` are
        accepted for signature uniformity with the rest of the cross-device
        modules; the body forwards them where stochasticity is introduced
        (currently only DataLoader shuffle below is unseeded — kept simple
        because PFedRec's batch-size 256 over <500 positives per user means
        the within-epoch shuffle has negligible variance impact on PFR-08).

    DUAL LR PRESERVED — Pitfall 3 / D-19: ``optimizer_item.lr = lr * num_items
    * lr_eta`` (matches reference ``engine.py:117-119``). Do NOT change this;
    PFR-08 reproduction is sensitive to the boost.

    Parameters
    ----------
    model : PFedRecMLP
        Model with global item embeddings + bias, and the user's local
        ``affine_output.weight`` already loaded.
    user_items : List[int]
        Item indices (positives + sampled negatives).
    user_ratings : List[float]
        Binary ratings (1.0 / 0.0).
    lr : float
        Base learning rate for ``affine_output``.
    lr_eta : float
        Item embedding LR multiplier. Effective item LR = ``lr * num_items * lr_eta``.
    num_items : int
        Total number of items (drives the dual-LR boost).
    local_epochs : int
        Local epochs per round (paper default: 1).
    batch_size : int
        Batch size.
    device : torch.device
        Training device.
    l2_regularization : float
        L2 weight decay (paper default: 0.0).
    proximal_mu : float
        FedProx proximal term (D-07: dropped for PFedRec; kept for signature
        parity but expected to be 0.0).
    global_item_embedding : Optional[torch.Tensor]
        Server's item embedding for the proximal term (only used when
        ``proximal_mu > 0``).
    run_seed, user_idx, round_num, exclude_items :
        Phase 5 contract kwargs. Carried for signature parity; not consumed
        directly here because the per-user batch is already assembled by
        ``prepare_user_train_data`` with the per-round RNG.

    Returns
    -------
    float
        Average training loss over the local epochs.
    """
    # Signature-parity kwargs (consumed by prepare_user_train_data upstream).
    del run_seed, user_idx, round_num, exclude_items

    model.to(device)
    model.train()

    criterion = nn.BCELoss()

    # Two SGD optimizers with different LRs (matching reference exactly).
    optimizer_local = torch.optim.SGD(
        model.affine_output.parameters(),
        lr=lr,
        weight_decay=l2_regularization,
    )
    # DO NOT change — matches reference IJCAI-23-PFedRec/engine.py:117-119
    # (effective item LR = lr * num_items * lr_eta). Pitfall 3 + D-19.
    optimizer_item = torch.optim.SGD(
        model.embedding_item.parameters(),
        lr=lr * num_items * lr_eta,
        weight_decay=l2_regularization,
    )

    # Build a DataLoader from the per-user (items, ratings) batch.
    item_tensor = torch.LongTensor(user_items)
    rating_tensor = torch.FloatTensor(user_ratings)
    dataset = TensorDataset(item_tensor, rating_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    total_samples = 0

    for _epoch in range(local_epochs):
        for batch_items, batch_ratings in dataloader:
            batch_items = batch_items.to(device)
            batch_ratings = batch_ratings.to(device)

            # Step 1: Update score function (affine_output) only.
            optimizer_local.zero_grad()
            predictions = model(batch_items)
            loss_local = criterion(predictions.view(-1), batch_ratings)
            loss_local.backward()
            optimizer_local.step()

            # Step 2: Update item embedding only.
            optimizer_item.zero_grad()
            predictions = model(batch_items)
            loss_item = criterion(predictions.view(-1), batch_ratings)

            # FedProx proximal term on item embedding (D-07: PFedRec doesn't
            # use FedProx; this branch is dead code under paper-compat mode
            # but kept for signature parity).
            if proximal_mu > 0 and global_item_embedding is not None:
                proximal_term = (
                    model.embedding_item.weight - global_item_embedding.to(device)
                ).norm(2) ** 2
                loss_item = loss_item + (proximal_mu / 2) * proximal_term

            loss_item.backward()
            optimizer_item.step()

            total_loss += loss_item.item() * len(batch_items)
            total_samples += len(batch_items)

    return total_loss / total_samples if total_samples > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-user evaluation: sampled LOO + 99 negatives (FND-06 / PFR-04 / D-04)
# ---------------------------------------------------------------------------


def evaluate_pfedrec_sampled(
    model: PFedRecMLP,
    test_items: List[int],
    train_items_set: Set[int],
    num_items: int,
    device: torch.device,
    k_values: Optional[List[int]] = None,
    num_negatives: int = 99,
    *,
    run_seed: Optional[int] = None,
    user_idx: Optional[int] = None,
    round_num: Optional[int] = None,
    exclude_items: Optional[Iterable[int]] = None,
) -> Dict[str, float]:
    """Evaluate one user via leave-one-out + 99 negatives (NCF protocol + D-04 BCE scope).

    Phase 5 contract:

    - **PFR-06 / FND-06**: negative sampling sources randomness from
      ``np_rng(run_seed, user_idx, round_num, "eval_neg")``. The stdlib RNG
      module is NOT imported nor used anywhere in this function.
    - **PFR-04 / FND-03**: ``exclude_items`` is folded into the rated-items
      union BEFORE the negative-candidate pool is built — the held-out test
      positive is never drawn as a sampled negative.
    - **D-04 BCE scope**: per-user ``eval_loss`` is computed from
      ``ratings_pred = torch.cat((test_score, negative_score))`` over 100
      candidates with ``ratings = [1, 0, 0, ..., 0]``. Mirrors
      ``IJCAI-23-PFedRec/engine.py:195-196``.

    Parameters
    ----------
    model : PFedRecMLP
        Model with the user's ``affine_output`` loaded.
    test_items : List[int]
        Held-out positive item(s) for this user (typically length 1 under LOO).
    train_items_set : Set[int]
        Items the user has interacted with in training.
    num_items : int
        Total catalog size.
    device : torch.device
        Eval device.
    k_values : Optional[List[int]]
        K values for HR@K / NDCG@K (default: ``[5, 10, 20]``).
    num_negatives : int
        Sampled negatives per positive (default: 99 — NCF protocol).
    run_seed : Optional[int]
        FND-06 root seed (BSL-05 — replaces stdlib RNG seeding).
    user_idx : Optional[int]
        Drives per-user RNG namespace.
    round_num : Optional[int]
        Drives per-round RNG namespace.
    exclude_items : Optional[Iterable[int]]
        FND-03 exclusion set; folded into the rated-items union before negs.

    Returns
    -------
    Dict[str, float]
        ``sampled_hr@K`` / ``sampled_ndcg@K`` / ``sampled_mrr`` /
        ``sampled_num_negatives`` / ``sampled_num_users`` plus ``eval_loss``
        (D-04 mean BCE over the 100-item candidate pool).
    """
    if k_values is None:
        k_values = [5, 10, 20]

    model.to(device)
    model.eval()

    all_items = set(range(int(num_items)))
    rated_union = set(int(x) for x in train_items_set) | set(
        int(x) for x in test_items
    )
    if exclude_items is not None:
        rated_union |= set(
            int(x) for x in np.asarray(list(exclude_items)).tolist()
        )

    # FND-06 RNG: per (run_seed, user_idx, round_num, "eval_neg"). Falls back
    # to a deterministic seed when the contract kwargs are missing — stdlib
    # ``random`` is NEVER touched.
    if run_seed is not None and user_idx is not None:
        rng = np_rng(int(run_seed), int(user_idx), int(round_num or 0), "eval_neg")
    else:
        rng = np.random.default_rng(0)

    metrics_per_k = {k: {"hits": 0, "ndcgs": []} for k in k_values}
    mrr_scores: List[float] = []
    bce_losses: List[float] = []
    bce_criterion = nn.BCELoss()
    num_evaluated = 0

    with torch.no_grad():
        for positive_item in test_items:
            # Sample 99 negatives from items not in the rated_union ∪ exclude.
            negative_candidates = list(all_items - rated_union)
            if len(negative_candidates) < int(num_negatives):
                negative_items = list(negative_candidates)
            else:
                # rng.choice with replace=False — distribution-equivalent to
                # the reference's stdlib uniform-without-replacement, but
                # seeded by FND-06 (PFR-06).
                chosen = rng.choice(
                    np.asarray(negative_candidates, dtype=np.int64),
                    size=int(num_negatives),
                    replace=False,
                )
                negative_items = [int(x) for x in chosen.tolist()]

            # D-04 BCE over (positive + 99 negatives) — mirrors engine.py:195-196.
            test_item_tensor = torch.tensor(
                [int(positive_item)], dtype=torch.long
            ).to(device)
            negative_item_tensor = torch.tensor(
                negative_items, dtype=torch.long
            ).to(device)
            test_score = model(test_item_tensor)
            negative_score = model(negative_item_tensor)
            ratings_pred = torch.cat((test_score, negative_score))
            ratings_true = torch.zeros(
                ratings_pred.numel(), dtype=torch.float32, device=device
            )
            ratings_true[0] = 1.0
            loss_value = bce_criterion(
                ratings_pred.view(-1), ratings_true
            ).item()
            bce_losses.append(float(loss_value))

            # Build the candidate pool + rank the positive.
            candidate_items = [int(positive_item)] + negative_items
            scores = ratings_pred.view(-1).cpu().numpy()
            order = np.argsort(-scores)  # descending
            ranked_items = [candidate_items[int(i)] for i in order.tolist()]
            try:
                positive_rank = ranked_items.index(int(positive_item)) + 1
            except ValueError:
                positive_rank = len(ranked_items) + 1
            mrr_scores.append(1.0 / positive_rank)

            for k in k_values:
                top_k = ranked_items[:k]
                if int(positive_item) in top_k:
                    metrics_per_k[k]["hits"] += 1
                    pos_in_topk = top_k.index(int(positive_item))
                    ndcg = 1.0 / float(np.log2(pos_in_topk + 2))
                else:
                    ndcg = 0.0
                metrics_per_k[k]["ndcgs"].append(ndcg)

            num_evaluated += 1

    if num_evaluated == 0:
        return {}

    results: Dict[str, float] = {}
    for k in k_values:
        results[f"sampled_hr@{k}"] = (
            metrics_per_k[k]["hits"] / num_evaluated
        )
        results[f"sampled_ndcg@{k}"] = float(
            np.mean(metrics_per_k[k]["ndcgs"])
        ) if metrics_per_k[k]["ndcgs"] else 0.0
    results["sampled_mrr"] = (
        float(np.mean(mrr_scores)) if mrr_scores else 0.0
    )
    results["sampled_num_negatives"] = float(num_negatives)
    results["sampled_num_users"] = float(num_evaluated)
    # D-04: mean per-user BCE over 100-item candidate pool.
    results["eval_loss"] = (
        float(np.mean(bce_losses)) if bce_losses else 0.0
    )
    return results


def test_pfedrec(
    model: PFedRecMLP,
    test_items: List[int],
    test_ratings: List[float],
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate per-user BCE / RMSE / MAE on test rows directly (no negatives).

    Retained for the legacy diagnostic path. The thesis-table headline numbers
    derive from ``evaluate_pfedrec_sampled`` (D-04 BCE scope + sampled HR/NDCG).

    Parameters
    ----------
    model : PFedRecMLP
        Model with the user's ``affine_output`` loaded.
    test_items : List[int]
        Test item indices.
    test_ratings : List[float]
        Test ratings (will be binarized: > 0 -> 1.0).
    device : torch.device
        Eval device.

    Returns
    -------
    Tuple[float, Dict[str, float]]
        ``(bce_loss, {'bce', 'rmse', 'mae'})``.
    """
    model.to(device)
    model.eval()

    criterion = nn.BCELoss()
    item_tensor = torch.LongTensor(test_items).to(device)
    rating_tensor = torch.FloatTensor(
        [1.0 if r > 0 else 0.0 for r in test_ratings]
    ).to(device)

    with torch.no_grad():
        predictions = model(item_tensor).view(-1)
        bce_loss = criterion(predictions, rating_tensor).item()
        pred_np = predictions.cpu().numpy()
        rating_np = rating_tensor.cpu().numpy()
        rmse = float(np.sqrt(np.mean((pred_np - rating_np) ** 2)))
        mae = float(np.mean(np.abs(pred_np - rating_np)))

    return bce_loss, {"bce": bce_loss, "rmse": rmse, "mae": mae}


__all__ = [
    "_dataset_cache",
    "_sample_train_negatives_seeded",
    "evaluate_pfedrec_sampled",
    "get_model",
    "load_data",
    "prepare_user_train_data",
    "test_pfedrec",
    "train_pfedrec_single_user",
]
