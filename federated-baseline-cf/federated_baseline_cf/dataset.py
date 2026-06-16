"""MovieLens 1M Dataset Loading — Phase 2 thin adapter over fedrec_foundation.

Post-Phase-2 this module is responsible only for:
  1. Raw data download + parse (``download_movielens_1m``, ``load_movielens_1m``).
  2. Building the natural cross-device partitioning (1 user = 1 client) from the
     canonical ``user2idx`` exposed by ``fedrec_foundation.mapping``
     (``natural_partition_users``).
  3. Wrapping the above into PyTorch DataLoaders keyed by (partition_id,
     num_partitions) and returning the tuple shape consumed by
     ``client_app.py`` / ``task.py`` (``load_partition_data``,
     ``load_full_data``).

Mapping / split / exclusion-set construction is DELEGATED to the Phase 1
foundation bundle at ``data/derived/`` (committed, hash-locked). Callers of
``load_partition_data`` observe the same ``user2idx`` / ``item2idx`` / held-out
test items as every other federated module -- there is now a single source
of truth for the cross-device protocol.

Per D-17: ``create_global_mappings``, ``create_leave_one_out_split``,
``compute_user_genre_distribution``, ``dirichlet_partition_users``,
``create_train_test_split`` are REMOVED. The corresponding foundation loaders
(``fedrec_foundation.mapping.load_mapping``, ``.split.load_split_manifest``,
``.exclusion.load_exclusion``) are the replacements.

Per D-18: ``MovieLensDataset``, ``download_movielens_1m``, ``load_movielens_1m``,
and ``natural_partition_users`` retain their pre-existing WIP state; only
the D-17 rip targets and ``load_partition_data`` / ``load_full_data`` bodies
change in this plan.
"""

import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fedrec_foundation.bundle import verify_bundle
from fedrec_foundation.exclusion import ExclusionTable, load_exclusion
from fedrec_foundation.mapping import CanonicalMapping, load_mapping
from fedrec_foundation.paths import data_derived
from fedrec_foundation.split import SplitManifest, load_split_manifest

# Default data directory: relative to project root (../../data from this module)
_MODULE_DIR = Path(__file__).parent
_DEFAULT_DATA_DIR = _MODULE_DIR.parent.parent / "data"


# Module-level in-memory cache: avoids re-reading the bundle each client.
# Keyed by the foundation_contract_sha256 so a bundle rebuild invalidates
# the cache automatically.
_foundation_cache: Dict[str, Dict] = {}


class MovieLensDataset(Dataset):
    """PyTorch Dataset for MovieLens ratings."""

    def __init__(self, ratings_df: pd.DataFrame, user2idx: Dict, item2idx: Dict):
        """
        Initialize MovieLens Dataset.

        Args:
            ratings_df: DataFrame with columns [user_id, movie_id, rating, timestamp]
            user2idx: Mapping from user_id to index
            item2idx: Mapping from movie_id to index
        """
        self.ratings = ratings_df
        self.user2idx = user2idx
        self.item2idx = item2idx

        # Convert to indexed format
        self.users = torch.LongTensor(
            [user2idx[uid] for uid in ratings_df["user_id"].values]
        )
        self.items = torch.LongTensor(
            [item2idx[mid] for mid in ratings_df["movie_id"].values]
        )
        self.ratings_tensor = torch.FloatTensor(ratings_df["rating"].values)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            "user": self.users[idx],
            "item": self.items[idx],
            "rating": self.ratings_tensor[idx],
        }


def download_movielens_1m(data_dir: Optional[str] = None) -> str:
    """
    Download MovieLens 1M dataset.

    Args:
        data_dir: Directory to save the dataset (defaults to project root data/)

    Returns:
        Path to the extracted dataset directory
    """
    if data_dir is None:
        data_dir = str(_DEFAULT_DATA_DIR)
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    ml_dir = data_path / "ml-1m"
    if ml_dir.exists():
        print(f"MovieLens 1M already exists at {ml_dir}")
        return str(ml_dir)

    # Download dataset
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = data_path / "ml-1m.zip"

    print(f"Downloading MovieLens 1M from {url}...")
    urlretrieve(url, zip_path)

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    # Clean up zip file
    zip_path.unlink()
    print(f"MovieLens 1M downloaded and extracted to {ml_dir}")

    return str(ml_dir)


def load_movielens_1m(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens 1M dataset.

    Args:
        data_dir: Directory containing the ml-1m folder (defaults to project root data/)

    Returns:
        Tuple of (ratings_df, movies_df, users_df)
    """
    if data_dir is None:
        data_dir = str(_DEFAULT_DATA_DIR)
    ml_dir = Path(data_dir) / "ml-1m"

    # Load ratings
    ratings = pd.read_csv(
        ml_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )

    # Load movies
    movies = pd.read_csv(
        ml_dir / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )

    # Load users
    users = pd.read_csv(
        ml_dir / "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )

    print(f"Loaded {len(ratings)} ratings from {len(users)} users on {len(movies)} movies")
    return ratings, movies, users


def natural_partition_users(
    ratings_df: pd.DataFrame,
    user2idx: Dict[int, int],
) -> Dict[int, pd.DataFrame]:
    """
    Cross-device partitioning: 1 user = 1 client.

    Each user becomes a separate partition, matching the standard
    federated recommendation setup used by PFedRec, FedMF, FedNCF, etc.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Full ratings with 'user_id' column.
    user2idx : Dict[int, int]
        Mapping from raw user_id to contiguous index (0..N-1).

    Returns
    -------
    Dict[int, pd.DataFrame]
        {user_idx: DataFrame of that user's ratings}
    """
    print(f"Natural partitioning: {len(user2idx)} users → {len(user2idx)} clients (1:1)")
    partitions: Dict[int, pd.DataFrame] = {}
    # Group by user_id for efficient partitioning
    grouped = ratings_df.groupby("user_id")
    for user_id, user_idx in user2idx.items():
        if user_id in grouped.groups:
            partitions[user_idx] = grouped.get_group(user_id).copy()
        else:
            partitions[user_idx] = pd.DataFrame(columns=ratings_df.columns)
    print(
        f"Natural partitioning complete: "
        f"min={min(len(p) for p in partitions.values())} ratings, "
        f"max={max(len(p) for p in partitions.values())} ratings, "
        f"mean={np.mean([len(p) for p in partitions.values()]):.1f}"
    )
    return partitions


# --- Phase 2 Plan 02: foundation-backed adapters ---


def _load_foundation_bundle(data_dir: Optional[str] = None) -> Dict:
    """Load mapping / split / exclusion from the committed ``data/derived/`` bundle.

    Calls ``verify_bundle`` first -- a tampered or incomplete bundle raises
    ``RuntimeError`` at load time (fail-loud per N-3).

    Parameters
    ----------
    data_dir : Optional[str]
        If provided, overrides the default ``<repo>/data/`` location. Uses
        ``fedrec_foundation.paths.data_derived()`` as the canonical default.

    Returns
    -------
    Dict
        A dict with keys:
        - ``mapping`` (CanonicalMapping)
        - ``split_manifest`` (SplitManifest)
        - ``exclusion`` (ExclusionTable) -- DO NOT close; module-level cache.
        - ``foundation_contract_sha256`` (str)
    """
    if data_dir is not None:
        derived = Path(data_dir).resolve() / "derived"
    else:
        derived = data_derived()

    idx = verify_bundle(derived)  # raises on mismatch/missing
    contract_key = idx.foundation_contract_sha256
    if contract_key in _foundation_cache:
        return _foundation_cache[contract_key]

    bundle = {
        "mapping": load_mapping(str(derived / "mapping.json")),
        "split_manifest": load_split_manifest(derived / "split_manifest.json"),
        "exclusion": load_exclusion(derived / "exclusion_items.npz"),
        "foundation_contract_sha256": contract_key,
    }
    _foundation_cache[contract_key] = bundle
    return bundle


def load_partition_data(
    partition_id: int,
    num_partitions: int,
    alpha: float = 0.5,
    test_ratio: float = 0.2,
    batch_size: int = 32,
    data_dir: Optional[str] = None,
    split_mode: str = "leave-one-out",
    partition_mode: str = "natural",
):
    """Load one client's partition backed by the foundation bundle.

    Post-Phase-2 callers (``client_app.py::@app.train()``, ``@app.evaluate()``,
    ``task.py::load_data``) pass ``partition_id`` and expect the same 6-tuple
    as before. The ``alpha`` / ``test_ratio`` / ``split_mode`` parameters
    are retained for backwards compatibility with cross-silo legacy
    (``partition_mode="dirichlet"``) but under the benchmark path
    (``partition_mode="natural"``) they are unused -- the bundle's
    deterministic LOO split is the authoritative split.

    Parameters
    ----------
    partition_id : int
        Client's partition index. Under ``partition_mode="natural"`` this
        is the ``user_idx`` in ``[0, num_users)``.
    num_partitions : int
        Total partitions. Under ``partition_mode="natural"`` this should
        equal ``bundle["mapping"].num_users``.
    alpha : float
        Dirichlet concentration (unused under natural partitioning).
    test_ratio : float
        Random-split ratio (unused under LOO split mode).
    batch_size : int
        DataLoader batch size.
    data_dir : Optional[str]
        Override the default ``<repo>/data/`` location.
    split_mode : str
        ``"leave-one-out"`` or ``"random"``. Under the foundation path we
        only support ``"leave-one-out"`` -- passing ``"random"`` raises
        ``ValueError``.
    partition_mode : str
        ``"natural"`` (cross-device) or ``"dirichlet"`` (cross-silo legacy).

    Returns
    -------
    Tuple[DataLoader, DataLoader, int, int, Dict[int,int], Dict[int,int]]
        ``(trainloader, testloader, num_users, num_items, user2idx, item2idx)``.
    """
    from torch.utils.data import DataLoader

    if split_mode != "leave-one-out":
        raise ValueError(
            f"split_mode={split_mode!r} not supported post-Phase-2 foundation migration; "
            f"use 'leave-one-out' (NCF protocol) or run with partition_mode='dirichlet' "
            f"(cross-silo legacy path still uses random split)."
        )

    bundle = _load_foundation_bundle(data_dir)
    mapping: CanonicalMapping = bundle["mapping"]
    split: SplitManifest = bundle["split_manifest"]
    user2idx = mapping.user2idx
    item2idx = mapping.item2idx
    num_users = mapping.num_users
    num_items = mapping.num_items

    if partition_mode == "natural":
        # Cross-device: 1 user = 1 client.
        # Download + parse raw ratings once (cached inside download_movielens_1m).
        download_movielens_1m(data_dir)
        ratings_df, _, _ = load_movielens_1m(data_dir)
        # Partition by user_idx so partition_id == user_idx.
        partitions = natural_partition_users(ratings_df, user2idx)
        if partition_id not in partitions:
            raise ValueError(
                f"partition_id={partition_id} not in natural partition keyspace "
                f"[0, {num_users}); did num-supernodes match num_users at federation init?"
            )
        client_ratings = partitions[partition_id].copy()
        client_ratings["user_idx"] = client_ratings["user_id"].map(user2idx).astype(int)
        client_ratings["item_idx"] = client_ratings["movie_id"].map(item2idx).astype(int)

        # Build train + test split using the foundation's test_item_per_user map.
        test_item = split.test_item_per_user.get(int(partition_id))
        if test_item is not None:
            test_mask = client_ratings["item_idx"] == int(test_item)
            test_df = client_ratings[test_mask].copy()
            train_df = client_ratings[~test_mask].copy()
        else:
            # User has < 2 interactions -- no held-out test item.
            test_df = client_ratings.iloc[0:0].copy()
            train_df = client_ratings.copy()
    elif partition_mode == "dirichlet":
        # Cross-silo legacy: delegate to the surviving dirichlet path only
        # if data_dir-dependent raw data is available. This branch is a
        # thin shim that preserves the pre-Phase-2 behavior for appendix
        # runs; production thesis runs use natural.
        raise NotImplementedError(
            "Cross-silo legacy (partition_mode='dirichlet') requires the "
            "pre-Phase-2 dirichlet_partition_users implementation. Per D-17 that "
            "helper is removed; re-add only if explicit cross-silo legacy runs "
            "are required (see .planning/phases/02-baseline-migration/02-CONTEXT.md "
            "§Deferred -- cross_silo_legacy regression tests)."
        )
    else:
        raise ValueError(f"Unknown partition_mode={partition_mode!r}")

    # Build PyTorch Datasets + DataLoaders (shuffle generator threaded by the
    # client in Plan 03 via torch_gen(run_seed, user_idx, round_num, 'dataloader')).
    train_dataset = MovieLensDataset(train_df, user2idx, item2idx)
    test_dataset = MovieLensDataset(test_df, user2idx, item2idx)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader, num_users, num_items, user2idx, item2idx


def load_full_data(
    test_ratio: float = 0.2,
    batch_size: int = 256,
    data_dir: Optional[str] = None,
    split_mode: str = "leave-one-out",
):
    """Load the full (non-partitioned) dataset for server-side centralized evaluation.

    Same return shape as :func:`load_partition_data` but across ALL users.
    Consumed by ``server_app.py`` for the end-of-training centralized eval
    (Plan 04 keeps this call path intact).

    Parameters
    ----------
    test_ratio : float
        Unused under ``split_mode="leave-one-out"`` (kept for API parity).
    batch_size : int
        DataLoader batch size.
    data_dir : Optional[str]
        Override default data path.
    split_mode : str
        ``"leave-one-out"``; ``"random"`` raises ``ValueError``.

    Returns
    -------
    Tuple[DataLoader, DataLoader, int, int, Dict[int,int], Dict[int,int]]
        ``(trainloader, testloader, num_users, num_items, user2idx, item2idx)``
        -- identical shape to ``load_partition_data``.
    """
    from torch.utils.data import DataLoader

    if split_mode != "leave-one-out":
        raise ValueError(
            f"split_mode={split_mode!r} not supported post-Phase-2 foundation "
            f"migration; use 'leave-one-out'."
        )

    bundle = _load_foundation_bundle(data_dir)
    mapping: CanonicalMapping = bundle["mapping"]
    split: SplitManifest = bundle["split_manifest"]
    user2idx = mapping.user2idx
    item2idx = mapping.item2idx
    num_users = mapping.num_users
    num_items = mapping.num_items

    download_movielens_1m(data_dir)
    ratings_df, _, _ = load_movielens_1m(data_dir)
    ratings_df["user_idx"] = ratings_df["user_id"].map(user2idx).astype(int)
    ratings_df["item_idx"] = ratings_df["movie_id"].map(item2idx).astype(int)

    # Build test mask: one row per user matching test_item_per_user.
    # Users without a held-out test item (single-interaction users) map to NaN
    # which never equals an item_idx, so they stay entirely in train.
    test_item_series = ratings_df["user_idx"].map(split.test_item_per_user)
    test_mask = ratings_df["item_idx"] == test_item_series
    test_df = ratings_df[test_mask].copy()
    train_df = ratings_df[~test_mask].copy()

    train_dataset = MovieLensDataset(train_df, user2idx, item2idx)
    test_dataset = MovieLensDataset(test_df, user2idx, item2idx)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader, num_users, num_items, user2idx, item2idx
