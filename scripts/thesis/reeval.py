#!/usr/bin/env python
"""Offline re-evaluation of a finished federated run WITHOUT flwr.

Loads saved GLOBAL params (``global_state_best.pt`` / ``global_state_last.pt``
written by the server-side persistence patch) plus per-user LOCAL state from
``.embedding_cache/<run_id>/partition_{pid}.pt`` and replays the exact
sampled LOO-99 evaluation over all users — byte-paired with the run's own
D-06 full-population eval by stamping ``round_num = final_eval_round_index``
into the FND-06 seeded negative sampler.

Local-state VARIANTS isolate how much the per-user local state contributes
to ranking quality:

- ``cached``     : each user's own trained local row (FAIL CLOSED on misses).
- ``random``     : fresh Xavier-uniform local row, seeded by (run_seed, pid)
                   — the model's D-11 cold-start init.
- ``shuffled``   : seeded permutation — user i is scored with user perm(i)'s
                   cached local state (fail closed on misses).
- ``zero``       : zero tensors of the right shapes.
- ``calibrated`` : cached locals + ONE local epoch of training against the
                   loaded globals (D-06.7 calibration replay; globals are
                   snapshot-restored after the epoch so eval runs against
                   the exact restored global state, matching the federation).

Aggregation mirrors ``strategy.aggregate_evaluate``: per-user sufficient
stats (hit count, NDCG sum, evaluated users) summed across users, ratios
computed ONCE server-side — overall plus sparse/medium/dense groups via the
foundation ``train_user_stats`` group labels.

Usage
-----
    python scripts/thesis/reeval.py --self-test
    python scripts/thesis/reeval.py \\
        --run-dir results/federated/personalized/20260610-064423-f18e64 \\
        --globals best --module personalized --local-variant cached zero

NOTE: ``global_state_{best,last}.pt`` are produced by a persistence patch
added to every server_app at run end. Runs finished before that patch do
NOT have them — re-run (or back-fill) before pointing this tool at them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Import path bootstrap (mirrors scripts/thesis/aggregate_results.py).
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]
_FOUNDATION_PKG = _REPO_ROOT / "scripts" / "foundation"
_PERSONALIZED_PKG = _REPO_ROOT / "federated-personalized-cf"
for _p in (str(_FOUNDATION_PKG), str(_PERSONALIZED_PKG), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fedrec_foundation.rng import np_rng, torch_gen  # noqa: E402
from fedrec_foundation.user_groups import classify_user_group  # noqa: E402


USER_GROUPS = ("sparse", "medium", "dense")
VARIANTS = ("cached", "random", "shuffled", "zero", "calibrated")
PROGRESS_EVERY = 500


# ===========================================================================
# Run-config resolution (results.json + manifest.json provenance)
# ===========================================================================


@dataclass
class ReevalConfig:
    """Resolved configuration for one re-evaluation pass.

    All values are read from the run's ``results.json`` (``_manifest`` +
    ``federated_config`` blocks) so the replay matches the run's own D-06
    eval — never from live pyproject defaults.

    Attributes
    ----------
    run_dir : Path
        ``results/federated/<module>/<run_id>/`` directory.
    run_id : str
        ``_manifest.run_id`` — namespaces the embedding cache.
    run_seed : int
        FND-06 root seed (``_manifest.run_seed``).
    eval_round_num : int
        ``final_metrics.final_eval_round_index`` — passed as ``round_num``
        into the seeded eval-negative sampler so negatives bit-pair with the
        run's own D-06 full-pop eval.
    model_type : str
        ``"bpr"`` or ``"basic"`` (``federated_config.model_type``).
    embedding_dim : int
        Latent dimensionality (``federated_config.embedding_dim``).
    dropout : float
        Model dropout (``federated_config.dropout``).
    lr : float
        Client learning rate — used only by the ``calibrated`` variant.
    weight_decay : float
        L2 strength for calibration (not serialized in results.json; the
        client default ``1e-5`` from pyproject is assumed).
    num_train_negatives : int
        BPR training negatives per positive (calibration replay).
    num_eval_negatives : int
        Sampled eval negatives (99 = NCF protocol).
    reuse_cache : bool
        D-09 flag — switches the cache dir to ``sig_<hash>``.
    split_hash : str
        Foundation split hash, used for the D-05 cache signature check.
    batch_size : int
        Calibration DataLoader batch size (client-side default 256).
    calib_epochs : int
        Local epochs for the ``calibrated`` variant (D-06.7 default 1).
    """

    run_dir: Path
    run_id: str
    run_seed: int
    eval_round_num: int
    model_type: str
    embedding_dim: int
    dropout: float
    lr: float
    weight_decay: float
    num_train_negatives: int
    num_eval_negatives: int
    reuse_cache: bool
    split_hash: str
    batch_size: int = 256
    calib_epochs: int = 1
    source_config: Dict[str, Any] = field(default_factory=dict)
    manifest_excerpt: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.model_type not in ("bpr", "basic"):
            raise ValueError(f"Unsupported model_type {self.model_type!r} (expected 'bpr'/'basic')")
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if self.num_eval_negatives <= 0:
            raise ValueError(f"num_eval_negatives must be positive, got {self.num_eval_negatives}")
        if self.eval_round_num <= 0:
            raise ValueError(
                f"eval_round_num must be positive, got {self.eval_round_num} — "
                f"the run never produced a D-06 final eval round?"
            )


def resolve_config(run_dir: Path, calib_epochs: int = 1) -> ReevalConfig:
    """Build a :class:`ReevalConfig` from ``<run_dir>/results.json``.

    Parameters
    ----------
    run_dir : Path
        Run directory containing ``results.json`` (and optionally a
        standalone ``manifest.json`` used as a fallback for ``_manifest``).
    calib_epochs : int
        Local epochs for the ``calibrated`` variant.

    Returns
    -------
    ReevalConfig
        Fully resolved replay configuration.

    Raises
    ------
    FileNotFoundError
        If ``results.json`` is missing.
    ValueError
        If mandatory provenance fields cannot be resolved.
    """
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found in {run_dir}")
    with open(results_path, "r") as f:
        results = json.load(f)

    manifest = results.get("_manifest") or {}
    if not manifest:
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
    if not manifest:
        raise ValueError(f"No _manifest block in {results_path} and no manifest.json sidecar")

    fed_cfg = results.get("federated_config") or {}

    # round_num for the seeded eval negatives: prefer the nested
    # final_metrics block, fall back to the manifest, then to rounds+1.
    final_metrics = results.get("final_metrics") or {}
    eval_round = int(
        final_metrics.get(
            "final_eval_round_index",
            manifest.get("final_eval_round_index", 0),
        )
        or 0
    )
    if eval_round <= 0:
        fallback = int(results.get("training_rounds", 0)) + 1
        print(
            f"[reeval] WARNING: final_eval_round_index missing/zero; falling back to "
            f"training_rounds+1 = {fallback}. Negatives may NOT bit-pair with the run's D-06 eval."
        )
        eval_round = fallback

    overrides = manifest.get("overrides") or {}
    lr = float(fed_cfg.get("learning_rate", overrides.get("lr", 0.001)))

    return ReevalConfig(
        run_dir=run_dir,
        run_id=str(manifest.get("run_id", run_dir.name)),
        run_seed=int(manifest.get("run_seed", fed_cfg.get("run_seed", 42))),
        eval_round_num=eval_round,
        model_type=str(fed_cfg.get("model_type", "bpr")),
        embedding_dim=int(fed_cfg.get("embedding_dim", 64)),
        dropout=float(fed_cfg.get("dropout", 0.1)),
        lr=lr,
        weight_decay=1e-5,
        num_train_negatives=int(manifest.get("num_train_negatives", 4)),
        num_eval_negatives=int(manifest.get("num_eval_negatives", 99)),
        reuse_cache=bool(fed_cfg.get("reuse_cache", False)),
        split_hash=str(manifest.get("split_hash", "")),
        calib_epochs=int(calib_epochs),
        source_config=dict(fed_cfg),
        manifest_excerpt={
            k: manifest.get(k)
            for k in (
                "run_id", "mode", "module", "run_seed", "split_hash",
                "final_eval_round_index", "num_eval_negatives",
                "num_train_negatives", "fraction_train", "git_commit",
            )
        },
    )


# ===========================================================================
# Per-user dataset snapshot (loaded ONCE, not per user)
# ===========================================================================


@dataclass
class UserEvalData:
    """One user's frozen train/test slice for the LOO-99 replay.

    Attributes
    ----------
    pid : int
        Partition id == canonical ``user_idx`` (natural partitioning).
    train_items : numpy.ndarray
        Item indices of the user's TRAIN rows (int64).
    train_ratings : numpy.ndarray
        Ratings aligned with ``train_items`` (float32).
    test_item : Optional[int]
        Held-out LOO positive (None when the foundation split elided it).
    test_rating : float
        Rating of the held-out row (diagnostic only).
    exclude_items : numpy.ndarray
        FND-03 exclusion set (``train positives ∪ {test positive}``).
    user_group : str
        ``"sparse" | "medium" | "dense"`` from foundation train_user_stats.
    """

    pid: int
    train_items: np.ndarray
    train_ratings: np.ndarray
    test_item: Optional[int]
    test_rating: float
    exclude_items: np.ndarray
    user_group: str


@dataclass
class RunData:
    """Dataset snapshot shared across all users / variants.

    Attributes
    ----------
    num_users : int
        Catalog user count (6040 for ML-1M).
    num_items : int
        Catalog item count (3706 for ML-1M).
    split_hash : str
        Foundation split hash (for the cache-signature check).
    users : Dict[int, UserEvalData]
        Per-pid eval slices, keyed by partition id.
    """

    num_users: int
    num_items: int
    split_hash: str
    users: Dict[int, UserEvalData]


def load_personalized_run_data(limit_users: Optional[int] = None) -> RunData:
    """Load the foundation bundle + raw ratings ONCE and slice per user.

    Replicates ``dataset.load_partition_data`` semantics exactly (test mask
    by ``item_idx == split.test_item_per_user[pid]``) without rebuilding the
    DataFrame per partition.

    Parameters
    ----------
    limit_users : Optional[int]
        Evaluate only pids ``[0, limit_users)`` (debug).

    Returns
    -------
    RunData
        Frozen per-user slices for every pid in range.
    """
    from federated_personalized_cf.dataset import (
        _load_foundation_bundle,
        download_movielens_1m,
        load_movielens_1m,
    )

    bundle = _load_foundation_bundle()
    mapping = bundle["mapping"]
    split = bundle["split_manifest"]
    exclusion = bundle["exclusion"]

    download_movielens_1m()
    ratings_df, _, _ = load_movielens_1m()
    ratings_df = ratings_df.copy()
    ratings_df["user_idx"] = ratings_df["user_id"].map(mapping.user2idx).astype(int)
    ratings_df["item_idx"] = ratings_df["movie_id"].map(mapping.item2idx).astype(int)

    num_users = int(mapping.num_users)
    num_items = int(mapping.num_items)
    n = num_users if limit_users is None else min(int(limit_users), num_users)

    stats_map = getattr(split, "train_user_stats", None)

    def _group_for(pid: int) -> str:
        # Mirrors client_app._classify_partition_user_group exactly.
        if stats_map is None:
            return classify_user_group(0)
        entry = stats_map.get(int(pid))
        if entry is None:
            return classify_user_group(0)
        group = getattr(entry, "user_group", None)
        if group is not None:
            return group
        return classify_user_group(int(getattr(entry, "n_interactions", 0)))

    grouped = ratings_df.groupby("user_idx")
    users: Dict[int, UserEvalData] = {}
    for pid in range(n):
        if pid in grouped.groups:
            rows = grouped.get_group(pid)
            items = rows["item_idx"].to_numpy(dtype=np.int64)
            ratings = rows["rating"].to_numpy(dtype=np.float32)
        else:
            items = np.empty(0, dtype=np.int64)
            ratings = np.empty(0, dtype=np.float32)

        test_item = split.test_item_per_user.get(int(pid))
        if test_item is not None:
            mask = items == int(test_item)
            if mask.any():
                test_rating = float(ratings[mask][0])
                train_items = items[~mask]
                train_ratings = ratings[~mask]
                resolved_test: Optional[int] = int(test_item)
            else:
                # Manifest names a test item the user's rows don't contain —
                # mirror load_partition_data: everything stays in train.
                test_rating = 0.0
                train_items, train_ratings = items, ratings
                resolved_test = None
        else:
            test_rating = 0.0
            train_items, train_ratings = items, ratings
            resolved_test = None

        users[pid] = UserEvalData(
            pid=pid,
            train_items=train_items,
            train_ratings=train_ratings,
            test_item=resolved_test,
            test_rating=test_rating,
            exclude_items=np.asarray(exclusion.for_user(pid)),
            user_group=_group_for(pid),
        )

    return RunData(
        num_users=num_users,
        num_items=num_items,
        split_hash=str(getattr(split, "split_hash", "")),
        users=users,
    )


def _make_batches(
    pid: int,
    items: np.ndarray,
    ratings: np.ndarray,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Build a list of batch dicts duck-typing a DataLoader over one user's rows.

    ``task.evaluate_ranking_sampled`` / ``task.train`` only iterate the
    loader and index ``batch['user'] / batch['item'] / batch['rating']`` —
    a list of dict-of-tensors is drop-in and avoids 6040 DataLoader builds.

    Parameters
    ----------
    pid : int
        Partition id, stamped into the ``user`` column.
    items : numpy.ndarray
        Item indices.
    ratings : numpy.ndarray
        Ratings aligned with ``items``.
    batch_size : int
        Rows per batch.
    generator : Optional[torch.Generator]
        When given, rows are shuffled with this seeded generator (CR-3
        spirit) — used for the calibration epoch only.

    Returns
    -------
    List[Dict[str, torch.Tensor]]
        Batches in iteration order.
    """
    n = len(items)
    if n == 0:
        return []
    item_t = torch.as_tensor(items, dtype=torch.long)
    rating_t = torch.as_tensor(ratings, dtype=torch.float32)
    if generator is not None:
        perm = torch.randperm(n, generator=generator)
        item_t = item_t[perm]
        rating_t = rating_t[perm]
    batches: List[Dict[str, torch.Tensor]] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        size = end - start
        batches.append({
            "user": torch.full((size,), int(pid), dtype=torch.long),
            "item": item_t[start:end],
            "rating": rating_t[start:end],
        })
    return batches


# ===========================================================================
# Module adapters (personalized fully implemented; adaptive pluggable)
# ===========================================================================


class ModuleAdapter:
    """Per-module plug point: model construction, local-state handling, calibration.

    Implementations must keep the eval path byte-identical to the module's
    own ``@app.evaluate`` handler (same evaluator function, same RNG
    namespace inputs, same exclusion set).
    """

    module_name: str = "base"

    def build_model(self, run_data: RunData) -> torch.nn.Module:
        """Construct the module's real model class."""
        raise NotImplementedError

    def set_globals(self, model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
        """Load GLOBAL params into the model (local tensors preserved)."""
        raise NotImplementedError

    def set_local(self, model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
        """Load LOCAL params into the model."""
        raise NotImplementedError

    def cached_local_path(self, pid: int) -> Path:
        """Path of the user's cached local-state ``.pt`` payload."""
        raise NotImplementedError

    def load_cached_local(self, pid: int) -> Dict[str, torch.Tensor]:
        """Load + shape-check the user's cached local state."""
        raise NotImplementedError

    def random_local(self, pid: int, run_seed: int) -> Dict[str, torch.Tensor]:
        """Fresh default-init local state, seeded by (run_seed, pid)."""
        raise NotImplementedError

    def zero_local(self) -> Dict[str, torch.Tensor]:
        """Zero tensors of the local-state shapes."""
        raise NotImplementedError

    def calibrate(
        self,
        model: torch.nn.Module,
        user: UserEvalData,
        cfg: ReevalConfig,
        device: str,
    ) -> None:
        """Run the D-06.7 calibration epoch(s) on this user's train rows."""
        raise NotImplementedError

    def evaluate_user(
        self,
        model: torch.nn.Module,
        user: UserEvalData,
        cfg: ReevalConfig,
        device: str,
    ) -> Dict[str, float]:
        """Run the module's primary sampled LOO-99 evaluator for one user."""
        raise NotImplementedError


class PersonalizedAdapter(ModuleAdapter):
    """Adapter for ``federated-personalized-cf`` (split-learning BPR-MF/BasicMF).

    LOCAL payload (single-row contract, D-10):
    ``OrderedDict({'local_user_row': (d,), 'local_user_bias': (1,)})`` at
    ``.embedding_cache/<run_id>/partition_{pid}.pt`` with a ``manifest.json``
    signature sidecar (D-04/D-05).

    GLOBAL state-dict keys: ``item_embeddings.weight``, ``item_bias.weight``,
    ``global_bias`` (the model's ``get_global_parameters()`` names — exactly
    what ``arrays.to_torch_state_dict()`` serializes).
    """

    module_name = "personalized"

    def __init__(self, cfg: ReevalConfig, cache_base_dir: Optional[Path] = None):
        self.cfg = cfg
        self.cache_base_dir = (
            Path(cache_base_dir)
            if cache_base_dir is not None
            else _REPO_ROOT / "federated-personalized-cf" / ".embedding_cache"
        )
        self._cache_dir: Optional[Path] = None
        self._run_data: Optional[RunData] = None

    # ---- data ------------------------------------------------------------

    def load_run_data(self, limit_users: Optional[int]) -> RunData:
        """Load (once) and memoize the per-user dataset snapshot."""
        if self._run_data is None:
            self._run_data = load_personalized_run_data(limit_users)
        return self._run_data

    def inject_run_data(self, run_data: RunData) -> None:
        """Inject a synthetic snapshot (self-test path; skips foundation load)."""
        self._run_data = run_data

    # ---- cache -----------------------------------------------------------

    def _signature(self, run_data: RunData) -> Dict[str, Any]:
        """The 6-field (+schema_version) D-04 signature this run should match."""
        return {
            "schema_version": 1,
            "run_id": str(self.cfg.run_id),
            "method": str(self.cfg.model_type),
            "num_users": int(run_data.num_users),
            "num_items": int(run_data.num_items),
            "dim": int(self.cfg.embedding_dim),
            "split_hash": str(self.cfg.split_hash),
        }

    def resolve_cache_dir(self, run_data: RunData) -> Path:
        """Resolve the run's cache dir (D-08 run_id dir or D-09 sig_<hash> dir)."""
        if self._cache_dir is not None:
            return self._cache_dir
        if not self.cfg.reuse_cache:
            self._cache_dir = self.cache_base_dir / self.cfg.run_id
        else:
            signature = self._signature(run_data)
            payload = json.dumps(
                {k: v for k, v in signature.items() if k != "run_id"},
                sort_keys=True,
            ).encode("utf-8")
            self._cache_dir = self.cache_base_dir / f"sig_{hashlib.sha256(payload).hexdigest()[:16]}"
        return self._cache_dir

    def verify_cache_signature(self, run_data: RunData) -> None:
        """D-05-style loud signature check of the cache's ``manifest.json``.

        Raises
        ------
        RuntimeError
            On any field mismatch (or a missing manifest sidecar).
        """
        cache_dir = self.resolve_cache_dir(run_data)
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                f"Embedding-cache manifest missing: {manifest_path} — was the run's "
                f"per-user cache deleted (rm -rf .embedding_cache)? Variants needing "
                f"cached locals cannot proceed."
            )
        with open(manifest_path, "r") as f:
            cached = json.load(f)
        signature = self._signature(run_data)
        deltas: List[str] = []
        for key in ("schema_version", "run_id", "method", "num_users", "num_items", "dim", "split_hash"):
            if self.cfg.reuse_cache and key == "run_id":
                continue
            if cached.get(key) != signature.get(key):
                deltas.append(f"{key}: cached={cached.get(key)!r}, expected={signature.get(key)!r}")
        if deltas:
            raise RuntimeError(
                "Embedding-cache signature mismatch (D-05):\n  " + "\n  ".join(deltas)
            )

    def cached_local_path(self, pid: int) -> Path:
        assert self._cache_dir is not None, "resolve_cache_dir must run first"
        return self._cache_dir / f"partition_{int(pid)}.pt"

    def load_cached_local(self, pid: int) -> Dict[str, torch.Tensor]:
        state = torch.load(str(self.cached_local_path(pid)), map_location="cpu", weights_only=False)
        assert set(state.keys()) == {"local_user_row", "local_user_bias"}, (
            f"D-10 violated on load: payload keys {sorted(state.keys())} for pid={pid}"
        )
        return state

    # ---- model -----------------------------------------------------------

    def build_model(self, run_data: RunData) -> torch.nn.Module:
        from federated_personalized_cf.task import get_model

        return get_model(
            model_type=self.cfg.model_type,
            num_users=run_data.num_users,
            num_items=run_data.num_items,
            embedding_dim=self.cfg.embedding_dim,
            dropout=self.cfg.dropout,
        )

    def set_globals(self, model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
        model.set_global_parameters(state)

    def set_local(self, model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
        model.set_local_parameters(state, strict=False)

    # ---- local-state variants ---------------------------------------------

    def random_local(self, pid: int, run_seed: int) -> Dict[str, torch.Tensor]:
        """Xavier-uniform local row (model's D-11 init) seeded by (run_seed, pid).

        The model inits ``local_user_row`` via ``xavier_uniform_`` on a
        ``(1, d)`` view: fan_in=d, fan_out=1, so bound = sqrt(6/(d+1)).
        Drawn from ``torch_gen(run_seed, pid, -1, "model_init")`` for
        cross-process reproducibility. ``local_user_bias`` is zero-init,
        matching ``BPRMF._init_weights``.
        """
        d = int(self.cfg.embedding_dim)
        bound = math.sqrt(6.0 / (d + 1))
        g = torch_gen(int(run_seed), int(pid), -1, "model_init")
        row = torch.empty(d).uniform_(-bound, bound, generator=g)
        return OrderedDict({
            "local_user_row": row,
            "local_user_bias": torch.zeros(1),
        })

    def zero_local(self) -> Dict[str, torch.Tensor]:
        return OrderedDict({
            "local_user_row": torch.zeros(int(self.cfg.embedding_dim)),
            "local_user_bias": torch.zeros(1),
        })

    # ---- calibration -------------------------------------------------------

    def calibrate(
        self,
        model: torch.nn.Module,
        user: UserEvalData,
        cfg: ReevalConfig,
        device: str,
    ) -> None:
        """Replay the D-06.7 calibration train call for one user.

        Mirrors the server's calibration ConfigRecord: ``lr=run lr``,
        ``proximal_mu=0.0``, ``round_num = final_eval_round_index`` (the
        server uses ``actual_rounds + 1`` for both calibration and the D-06
        eval), ``local_epochs_override = calib_epochs``. Globals are NOT
        frozen during the epoch (single-optimizer train path, matching the
        client); the caller restores the global snapshot afterwards, which
        reproduces the federation discarding the client's drifted globals.
        """
        from federated_personalized_cf.task import train as train_fn

        round_num = cfg.eval_round_num
        gen = torch_gen(cfg.run_seed, user.pid, round_num, "dataloader")
        batches = _make_batches(
            user.pid, user.train_items, user.train_ratings, cfg.batch_size, generator=gen
        )
        if not batches:
            return
        train_fn(
            model=model,
            trainloader=batches,
            epochs=cfg.calib_epochs,
            lr=cfg.lr,
            device=device,
            model_type=cfg.model_type,
            weight_decay=cfg.weight_decay,
            num_negatives=cfg.num_train_negatives,
            proximal_mu=0.0,
            run_seed=cfg.run_seed,
            user_idx=user.pid,
            round_num=round_num,
            exclude_items=user.exclude_items,
            rng=np_rng(cfg.run_seed, user.pid, round_num, "train_neg"),
        )

    # ---- evaluation ---------------------------------------------------------

    def evaluate_user(
        self,
        model: torch.nn.Module,
        user: UserEvalData,
        cfg: ReevalConfig,
        device: str,
    ) -> Dict[str, float]:
        """Byte-identical replay of the client's ``@app.evaluate`` primary path.

        Calls the REAL ``task.evaluate_ranking_sampled`` with
        ``k_values=[10]``, ``num_negatives=cfg.num_eval_negatives``,
        ``run_seed/user_idx/round_num`` matching the D-06 broadcast and the
        FND-03 exclusion set — so the sampled negatives bit-pair with the
        run's own final eval.
        """
        from federated_personalized_cf.task import evaluate_ranking_sampled

        trainloader = _make_batches(user.pid, user.train_items, user.train_ratings, cfg.batch_size)
        if user.test_item is not None:
            testloader = _make_batches(
                user.pid,
                np.asarray([user.test_item], dtype=np.int64),
                np.asarray([user.test_rating], dtype=np.float32),
                cfg.batch_size,
            )
        else:
            testloader = []
        return evaluate_ranking_sampled(
            model=model,
            testloader=testloader,
            trainloader=trainloader,
            device=device,
            k_values=[10],
            num_negatives=cfg.num_eval_negatives,
            run_seed=cfg.run_seed,
            user_idx=user.pid,
            round_num=cfg.eval_round_num,
            exclude_items=user.exclude_items,
        )


def get_adapter(module: str, cfg: ReevalConfig, cache_base_dir: Optional[Path] = None) -> ModuleAdapter:
    """Adapter factory — the per-module plug point.

    Parameters
    ----------
    module : str
        ``"personalized"`` (implemented) or ``"adaptive"`` (planned).
    cfg : ReevalConfig
        Resolved run configuration.
    cache_base_dir : Optional[Path]
        Override the module's ``.embedding_cache`` location (self-test).

    Returns
    -------
    ModuleAdapter

    Raises
    ------
    NotImplementedError
        For ``module="adaptive"`` — the adapter contract is in place but
        the dual-model specifics are not implemented yet.
    """
    if module == "personalized":
        return PersonalizedAdapter(cfg, cache_base_dir=cache_base_dir)
    if module == "adaptive":
        raise NotImplementedError(
            "module='adaptive' re-evaluation is not implemented yet. To add it, write an "
            "AdaptiveAdapter(ModuleAdapter) mirroring PersonalizedAdapter but with: "
            "(1) DualPersonalizedBPRMF construction (model-type=dual, fusion-type, MLP dims "
            "from the run's federated_config); (2) the dual LOCAL payload (user_embeddings, "
            "user_bias, personal_mlp.*, fusion gate/layer, logit_alpha, item_perturbation) "
            "loaded from federated-adaptive-personalized-cf/.embedding_cache/<run_id>/; "
            "(3) set_alpha() + set_global_prototype() called before every forward "
            "(the prototype EMA must also be persisted by the server patch); "
            "(4) the adaptive task.evaluate_ranking_sampled call signature. "
            "Until then, run with --module personalized."
        )
    raise ValueError(f"Unknown module {module!r}")


# ===========================================================================
# Sufficient-stat aggregation (mirrors strategy.aggregate_evaluate)
# ===========================================================================


class SuffStats:
    """Micro-average accumulator with the strategy's exact sufficient-stat math.

    Per user the client reports ``hit = round(hr@10 * n)``,
    ``ndcg_sum = ndcg@10 * n``, ``evaluated_users = n`` (n is 0 or 1 under
    the single-row contract); the server sums and computes ratios ONCE.
    """

    def __init__(self):
        self.totals: Dict[str, float] = {}
        for g in ("overall",) + USER_GROUPS:
            self.totals[f"hit_{g}"] = 0
            self.totals[f"ndcg_{g}"] = 0.0
            self.totals[f"mrr_{g}"] = 0.0
            self.totals[f"users_{g}"] = 0

    def add_user(self, sampled_metrics: Dict[str, float], user_group: str) -> None:
        """Fold one user's evaluator output into the running totals.

        Parameters
        ----------
        sampled_metrics : Dict[str, float]
            Output of ``evaluate_ranking_sampled`` (k_values=[10]).
        user_group : str
            ``"sparse" | "medium" | "dense"``.
        """
        n = int(sampled_metrics.get("sampled_num_users", 0))
        if n <= 0:
            return
        hr = float(sampled_metrics.get("sampled_hr@10", 0.0))
        ndcg = float(sampled_metrics.get("sampled_ndcg@10", 0.0))
        mrr = float(sampled_metrics.get("sampled_mrr", 0.0))
        hit = int(round(hr * n))
        ndcg_sum = ndcg * n
        mrr_sum = mrr * n
        for g in ("overall", user_group):
            self.totals[f"hit_{g}"] += hit
            self.totals[f"ndcg_{g}"] += ndcg_sum
            self.totals[f"mrr_{g}"] += mrr_sum
            self.totals[f"users_{g}"] += n

    def to_metrics(self) -> Dict[str, float]:
        """Compute the thesis-table ratio metrics (zero-division safe)."""

        def _ratio(num: float, den: float) -> float:
            return float(num) / float(den) if den else 0.0

        t = self.totals
        out: Dict[str, float] = {
            "sampled_hr@10": _ratio(t["hit_overall"], t["users_overall"]),
            "sampled_ndcg@10": _ratio(t["ndcg_overall"], t["users_overall"]),
            "sampled_mrr": _ratio(t["mrr_overall"], t["users_overall"]),
            "evaluated_users": int(t["users_overall"]),
        }
        for g in USER_GROUPS:
            out[f"sampled_hr@10/{g}"] = _ratio(t[f"hit_{g}"], t[f"users_{g}"])
            out[f"sampled_ndcg@10/{g}"] = _ratio(t[f"ndcg_{g}"], t[f"users_{g}"])
            out[f"sampled_mrr/{g}"] = _ratio(t[f"mrr_{g}"], t[f"users_{g}"])
            out[f"evaluated_users_{g}"] = int(t[f"users_{g}"])
        return out


# ===========================================================================
# Core re-evaluation loop
# ===========================================================================


def _sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file (provenance for the JSON dump)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_globals(run_dir: Path, which: str) -> Tuple[Dict[str, torch.Tensor], Path]:
    """Load ``global_state_<which>.pt`` from the run dir.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    which : str
        ``"best"`` or ``"last"``.

    Returns
    -------
    Tuple[Dict[str, torch.Tensor], Path]
        The torch state dict (GLOBAL param names) and its path.

    Raises
    ------
    FileNotFoundError
        With a pointer to the persistence patch when the file is absent.
    """
    path = run_dir / f"global_state_{which}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. This file is written by the server-side persistence "
            f"patch (torch.save(arrays.to_torch_state_dict(), ...) at run end). Runs "
            f"finished before that patch do not have it — re-run or back-fill the "
            f"global state before re-evaluating."
        )
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{path} did not contain a non-empty state dict")
    return state, path


def _resolve_local_state(
    adapter: ModuleAdapter,
    variant: str,
    pid: int,
    cfg: ReevalConfig,
    shuffle_map: Optional[Dict[int, int]],
) -> Dict[str, torch.Tensor]:
    """Produce the variant-specific local state for one user."""
    if variant in ("cached", "calibrated"):
        return adapter.load_cached_local(pid)
    if variant == "shuffled":
        assert shuffle_map is not None
        return adapter.load_cached_local(shuffle_map[pid])
    if variant == "random":
        return adapter.random_local(pid, cfg.run_seed)
    if variant == "zero":
        return adapter.zero_local()
    raise ValueError(f"Unknown variant {variant!r}")


def run_reeval(
    adapter: ModuleAdapter,
    cfg: ReevalConfig,
    run_data: RunData,
    global_state: Dict[str, torch.Tensor],
    variant: str,
    device: str = "cpu",
) -> Dict[str, float]:
    """Re-evaluate every user under one local-state variant.

    Parameters
    ----------
    adapter : ModuleAdapter
        Module plug point (model + cache + evaluator).
    cfg : ReevalConfig
        Resolved run configuration.
    run_data : RunData
        Per-user dataset snapshot (loaded once).
    global_state : Dict[str, torch.Tensor]
        Saved GLOBAL params (state-dict names).
    variant : str
        One of :data:`VARIANTS`.
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    Dict[str, float]
        Aggregated thesis-table metrics (overall + per-group).
    """
    pids = sorted(run_data.users.keys())

    # Fail-closed pre-scan for variants that consume the cache.
    shuffle_map: Optional[Dict[int, int]] = None
    if variant in ("cached", "calibrated", "shuffled"):
        adapter.verify_cache_signature(run_data)
        if variant == "shuffled":
            perm = np_rng(cfg.run_seed, -1, -1, "server_sample").permutation(len(pids))
            shuffle_map = {pids[i]: pids[int(perm[i])] for i in range(len(pids))}
            needed = sorted(set(shuffle_map.values()))
        else:
            needed = pids
        missing = [p for p in needed if not adapter.cached_local_path(p).exists()]
        if missing:
            shown = ", ".join(str(p) for p in missing[:20])
            more = "" if len(missing) <= 20 else f" (+{len(missing) - 20} more)"
            raise RuntimeError(
                f"variant={variant!r} FAIL CLOSED: {len(missing)} missing cache payload(s) "
                f"under {adapter.resolve_cache_dir(run_data)}: pids [{shown}]{more}"
            )

    model = adapter.build_model(run_data)
    adapter.set_globals(model, global_state)
    model.to(device)
    # CPU clones for the post-calibration restore (calibrated trains globals
    # locally like the real client; the federation discards them).
    global_snapshot = OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in global_state.items()
    )

    stats = SuffStats()
    t0 = time.time()
    for i, pid in enumerate(pids):
        user = run_data.users[pid]
        local_state = _resolve_local_state(adapter, variant, pid, cfg, shuffle_map)
        adapter.set_local(model, local_state)

        if variant == "calibrated":
            adapter.calibrate(model, user, cfg, device)
            # Restore the exact saved globals before scoring (mirrors the
            # D-06 broadcast evaluating against the server-restored arrays).
            adapter.set_globals(model, global_snapshot)
            model.to(device)

        sampled = adapter.evaluate_user(model, user, cfg, device)
        stats.add_user(sampled, user.user_group)

        if (i + 1) % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            partial = stats.to_metrics()
            print(
                f"[reeval:{variant}] {i + 1}/{len(pids)} users "
                f"({rate:.0f} u/s) | running ndcg@10={partial['sampled_ndcg@10']:.4f} "
                f"hr@10={partial['sampled_hr@10']:.4f}"
            )

    metrics = stats.to_metrics()
    print(
        f"[reeval:{variant}] done: {len(pids)} pids in {time.time() - t0:.1f}s "
        f"({metrics['evaluated_users']} evaluated)"
    )
    return metrics


# ===========================================================================
# Output: pretty table + JSON dump
# ===========================================================================


def print_table(variant: str, globals_kind: str, cfg: ReevalConfig, metrics: Dict[str, float]) -> None:
    """Print the per-variant results table."""
    width = 72
    print("=" * width)
    print(
        f" reeval | module run: {cfg.run_id} | globals={globals_kind} | "
        f"variant={variant}"
    )
    print(
        f" eval round_num={cfg.eval_round_num} | seed={cfg.run_seed} | "
        f"negatives={cfg.num_eval_negatives}"
    )
    print("-" * width)
    header = f" {'group':<10}{'users':>8}{'hr@10':>12}{'ndcg@10':>12}{'mrr':>12}"
    print(header)
    print("-" * width)
    rows = [("overall", "", "")] + [(g, f"/{g}", f"_{g}") for g in USER_GROUPS]
    for label, ratio_suffix, count_suffix in rows:
        users = metrics.get(f"evaluated_users{count_suffix}", 0)
        hr = metrics.get(f"sampled_hr@10{ratio_suffix}", 0.0)
        ndcg = metrics.get(f"sampled_ndcg@10{ratio_suffix}", 0.0)
        mrr = metrics.get(f"sampled_mrr{ratio_suffix}", 0.0)
        print(f" {label:<10}{users:>8}{hr:>12.4f}{ndcg:>12.4f}{mrr:>12.4f}")
    print("=" * width)


def dump_json(
    cfg: ReevalConfig,
    module: str,
    variant: str,
    globals_kind: str,
    globals_path: Path,
    metrics: Dict[str, float],
    device: str,
    limit_users: Optional[int],
    cache_dir: Optional[Path],
) -> Path:
    """Write ``<run-dir>/reeval_<variant>_<globals>.json`` with full provenance."""
    out_path = cfg.run_dir / f"reeval_{variant}_{globals_kind}.json"
    payload = {
        "tool": "scripts/thesis/reeval.py",
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv,
        "module": module,
        "variant": variant,
        "globals": globals_kind,
        "globals_file": str(globals_path),
        "globals_sha256": _sha256_file(globals_path),
        "device": device,
        "limit_users": limit_users,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "run_id": cfg.run_id,
        "run_seed": cfg.run_seed,
        "eval_round_num": cfg.eval_round_num,
        "num_eval_negatives": cfg.num_eval_negatives,
        "num_train_negatives": cfg.num_train_negatives,
        "calib_epochs": cfg.calib_epochs if variant == "calibrated" else None,
        "lr": cfg.lr,
        "model_type": cfg.model_type,
        "embedding_dim": cfg.embedding_dim,
        "metrics": metrics,
        "source_federated_config": cfg.source_config,
        "source_manifest_excerpt": cfg.manifest_excerpt,
    }
    tmp_fd, tmp_name = tempfile.mkstemp(prefix="reeval_tmp_", suffix=".json", dir=str(cfg.run_dir))
    os.close(tmp_fd)
    try:
        with open(tmp_name, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp_name, str(out_path))
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    print(f"[reeval] wrote {out_path}")
    return out_path


# ===========================================================================
# Self-test (synthetic end-to-end, no real data / no flwr / CPU only)
# ===========================================================================


def self_test() -> int:
    """End-to-end plumbing validation on a tiny synthetic setup.

    Builds the REAL ``BPRMF`` model class (20 users, 50 items, dim 8), saves
    fake globals as ``global_state_{best,last}.pt``, fabricates per-user
    cache payloads in the exact D-04/D-10 format, fabricates ``results.json``
    provenance, then runs every variant end-to-end on CPU.

    Cached rows are constructed maximally aligned with each user's held-out
    positive (``row = 100 * q_test`` over unit-norm item embeddings), so
    ``cached`` must clearly beat ``zero`` and ``random`` — validating that
    the variants actually swap the local state.

    Returns
    -------
    int
        0 on success (asserts otherwise).
    """
    print("[self-test] building synthetic run (20 users, 50 items, dim 8)...")
    from federated_personalized_cf.task import get_model

    num_users, num_items, dim = 20, 50, 8
    run_seed, eval_round = 123, 7
    run_id = "selftest-000000-feed42"

    with tempfile.TemporaryDirectory(prefix="reeval_selftest_") as tmp:
        tmp_path = Path(tmp)
        run_dir = tmp_path / "run"
        cache_base = tmp_path / "embedding_cache"
        run_dir.mkdir()
        (cache_base / run_id).mkdir(parents=True)

        # --- real model class with controlled global state ---------------
        model = get_model(
            model_type="bpr", num_users=num_users, num_items=num_items,
            embedding_dim=dim, dropout=0.1,
        )
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            emb = torch.randn(num_items, dim, generator=g)
            emb = emb / emb.norm(dim=1, keepdim=True)  # unit rows: q_i·q_i = 1 >= q_i·q_j
            model.item_embeddings.weight.copy_(emb)
            model.item_bias.weight.copy_(torch.randn(num_items, 1, generator=g) * 0.01)
            model.global_bias.zero_()
        global_state = model.get_global_parameters()
        torch.save(global_state, run_dir / "global_state_best.pt")
        torch.save(global_state, run_dir / "global_state_last.pt")

        # --- fabricated results.json provenance ---------------------------
        results = {
            "_manifest": {
                "run_id": run_id,
                "mode": "thesis_crossdevice_main",
                "module": "personalized",
                "run_seed": run_seed,
                "split_hash": "selftest-split-hash",
                "final_eval_round_index": eval_round,
                "num_eval_negatives": 20,
                "num_train_negatives": 2,
                "fraction_train": 0.1,
                "git_commit": "selftest",
            },
            "federated_config": {
                "model_type": "bpr",
                "embedding_dim": dim,
                "dropout": 0.1,
                "learning_rate": 0.05,
                "reuse_cache": False,
                "num_clients": num_users,
            },
            "final_metrics": {"final_eval_round_index": eval_round},
            "training_rounds": 6,
        }
        with open(run_dir / "results.json", "w") as f:
            json.dump(results, f)

        cfg = resolve_config(run_dir)
        assert cfg.eval_round_num == eval_round and cfg.run_seed == run_seed

        # --- synthetic per-user data + REAL-format cache payloads ---------
        data_rng = np.random.default_rng(99)
        users: Dict[int, UserEvalData] = {}
        groups = ["sparse", "medium", "dense"]
        for pid in range(num_users):
            perm = data_rng.permutation(num_items)
            train_items = np.asarray(sorted(int(x) for x in perm[:8]), dtype=np.int64)
            test_item = int(perm[8])
            exclude = np.asarray(
                sorted(set(train_items.tolist()) | {test_item}), dtype=np.int32
            )
            users[pid] = UserEvalData(
                pid=pid,
                train_items=train_items,
                train_ratings=data_rng.integers(1, 6, size=8).astype(np.float32),
                test_item=test_item,
                test_rating=5.0,
                exclude_items=exclude,
                user_group=groups[pid % 3],
            )
            # Cache payload in the exact D-04/D-10 single-row format,
            # adversarially aligned with the held-out positive.
            payload = OrderedDict({
                "local_user_row": (100.0 * emb[test_item]).clone(),
                "local_user_bias": torch.zeros(1),
            })
            torch.save(payload, cache_base / run_id / f"partition_{pid}.pt")
        signature = {
            "schema_version": 1,
            "run_id": run_id,
            "method": "bpr",
            "num_users": num_users,
            "num_items": num_items,
            "dim": dim,
            "split_hash": "selftest-split-hash",
        }
        with open(cache_base / run_id / "manifest.json", "w") as f:
            json.dump(signature, f)

        run_data = RunData(
            num_users=num_users, num_items=num_items,
            split_hash="selftest-split-hash", users=users,
        )

        # --- run every variant end-to-end ---------------------------------
        adapter = get_adapter("personalized", cfg, cache_base_dir=cache_base)
        adapter.inject_run_data(run_data)
        loaded_globals, globals_path = _load_globals(run_dir, "best")

        all_metrics: Dict[str, Dict[str, float]] = {}
        for variant in VARIANTS:
            metrics = run_reeval(adapter, cfg, run_data, loaded_globals, variant, device="cpu")
            print_table(variant, "best", cfg, metrics)
            out = dump_json(
                cfg, "personalized", variant, "best", globals_path, metrics,
                device="cpu", limit_users=None,
                cache_dir=adapter.resolve_cache_dir(run_data),
            )
            assert out.exists(), f"JSON dump missing for {variant}"
            all_metrics[variant] = metrics

        # --- assertions ----------------------------------------------------
        for variant, m in all_metrics.items():
            for key in ("sampled_ndcg@10", "sampled_hr@10", "sampled_mrr"):
                assert math.isfinite(m[key]), f"{variant}:{key} not finite: {m[key]}"
            assert m["evaluated_users"] == num_users, (
                f"{variant}: evaluated {m['evaluated_users']}/{num_users}"
            )
            for grp in USER_GROUPS:
                assert math.isfinite(m[f"sampled_ndcg@10/{grp}"]), f"{variant}:{grp} not finite"

        cached_ndcg = all_metrics["cached"]["sampled_ndcg@10"]
        zero_ndcg = all_metrics["zero"]["sampled_ndcg@10"]
        random_ndcg = all_metrics["random"]["sampled_ndcg@10"]
        assert cached_ndcg > zero_ndcg, (
            f"'zero' should differ from (be below) 'cached': cached={cached_ndcg:.4f} "
            f"zero={zero_ndcg:.4f}"
        )
        assert cached_ndcg > random_ndcg, (
            f"'random' should differ from (be below) 'cached': cached={cached_ndcg:.4f} "
            f"random={random_ndcg:.4f}"
        )
        assert abs(cached_ndcg - zero_ndcg) > 1e-9 and abs(cached_ndcg - random_ndcg) > 1e-9

        print("\n[self-test] summary (overall sampled_ndcg@10):")
        for variant in VARIANTS:
            print(f"  {variant:<11} {all_metrics[variant]['sampled_ndcg@10']:.4f}")
        print("[self-test] PASS — all 5 variants finite; zero/random differ from cached.")
    return 0


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse CLI."""
    parser = argparse.ArgumentParser(
        prog="reeval.py",
        description=(
            "Offline re-evaluation of a finished federated run (no flwr): load saved "
            "global params + per-user local cache, replay the sampled LOO-99 eval over "
            "all users with local-state variants."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="results/federated/<module>/<run_id>/ directory (reads results.json)",
    )
    parser.add_argument(
        "--globals", choices=("best", "last"), default="best",
        help="which saved global state to load (global_state_<which>.pt)",
    )
    parser.add_argument(
        "--module", choices=("personalized", "adaptive"), default="personalized",
        help="federated module the run belongs to",
    )
    parser.add_argument(
        "--local-variant", nargs="+", choices=VARIANTS, default=["cached"],
        help="local-state variant(s) to evaluate",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu",
        help="torch device (keep cpu while a GPU job is running)",
    )
    parser.add_argument(
        "--limit-users", type=int, default=None,
        help="evaluate only pids [0, N) (debug)",
    )
    parser.add_argument(
        "--calib-epochs", type=int, default=1,
        help="local epochs for the 'calibrated' variant (D-06.7 default)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="override the module's .embedding_cache base directory",
    )
    parser.add_argument(
        "--no-json", action="store_true",
        help="skip writing reeval_<variant>_<globals>.json into the run dir",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="run the synthetic end-to-end self-test and exit",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : Optional[List[str]]
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code.
    """
    args = build_parser().parse_args(argv)

    if args.self_test:
        return self_test()

    if args.run_dir is None:
        print("error: --run-dir is required (or use --self-test)", file=sys.stderr)
        return 2
    run_dir = args.run_dir.resolve()

    cfg = resolve_config(run_dir, calib_epochs=args.calib_epochs)
    print(
        f"[reeval] run {cfg.run_id} | module={args.module} | globals={args.globals} | "
        f"eval round_num={cfg.eval_round_num} | seed={cfg.run_seed}"
    )

    adapter = get_adapter(args.module, cfg, cache_base_dir=args.cache_dir)
    global_state, globals_path = _load_globals(run_dir, args.globals)
    print(f"[reeval] loaded globals: {globals_path} ({sorted(global_state.keys())})")

    print("[reeval] loading dataset snapshot (foundation bundle + ratings, once)...")
    run_data = adapter.load_run_data(args.limit_users)
    print(
        f"[reeval] snapshot ready: {len(run_data.users)} pids "
        f"({run_data.num_users} users x {run_data.num_items} items)"
    )

    summary: Dict[str, Dict[str, float]] = {}
    for variant in args.local_variant:
        metrics = run_reeval(adapter, cfg, run_data, global_state, variant, device=args.device)
        print_table(variant, args.globals, cfg, metrics)
        if not args.no_json:
            cache_dir = (
                adapter.resolve_cache_dir(run_data)
                if isinstance(adapter, PersonalizedAdapter)
                else None
            )
            dump_json(
                cfg, args.module, variant, args.globals, globals_path, metrics,
                device=args.device, limit_users=args.limit_users, cache_dir=cache_dir,
            )
        summary[variant] = metrics

    if len(summary) > 1:
        print("\n[reeval] cross-variant summary (overall):")
        print(f" {'variant':<12}{'hr@10':>10}{'ndcg@10':>10}{'ndcg sparse':>13}{'medium':>10}{'dense':>10}")
        for variant, m in summary.items():
            print(
                f" {variant:<12}{m['sampled_hr@10']:>10.4f}{m['sampled_ndcg@10']:>10.4f}"
                f"{m['sampled_ndcg@10/sparse']:>13.4f}{m['sampled_ndcg@10/medium']:>10.4f}"
                f"{m['sampled_ndcg@10/dense']:>10.4f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
