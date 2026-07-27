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

The pfedrec module additionally supports the Stage-A head-blend suite
(``global_head*``, ``alpha_grid``, ``alpha_oracle_group``,
``alpha_oracle_user``, ``alpha_policy_hc``) plus an EXPLICIT trust gate: the
``cached`` variant is compared field-by-field against the run's own D-06
``final_metrics.best`` block (|delta| < 1e-6, exact user counts) and the tool
aborts (exit 3) on mismatch — no variant number is trustworthy unless the
replay reproduces the run's own eval.

Usage
-----
    python scripts/thesis/reeval.py --self-test
    python scripts/thesis/reeval.py \\
        --run-dir results/federated/personalized/20260610-064423-f18e64 \\
        --globals best --module personalized --local-variant cached zero
    python scripts/thesis/reeval.py \\
        --run-dir results/federated/pfedrec/20260726-083121-cf72e0 \\
        --module pfedrec --globals best --local-variant cached \\
        global_head alpha_grid alpha_oracle_group alpha_oracle_user alpha_policy_hc

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
_PFEDREC_PKG = _REPO_ROOT / "federated-pfedrec"
_ADAPTIVE_PKG = _REPO_ROOT / "federated-adaptive-personalized-cf"
for _p in (
    str(_FOUNDATION_PKG),
    str(_PERSONALIZED_PKG),
    str(_PFEDREC_PKG),
    str(_ADAPTIVE_PKG),
    str(_REPO_ROOT / "scripts"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fedrec_foundation.rng import np_rng, torch_gen  # noqa: E402
from fedrec_foundation.user_groups import classify_user_group  # noqa: E402


USER_GROUPS = ("sparse", "medium", "dense")
VARIANTS = ("cached", "random", "shuffled", "zero", "calibrated")
# pfedrec adds the head-blend suite; "zero" is EXCLUDED by design (degenerate:
# all candidates score sigmoid(bias) -> stable argsort ranks the held-out
# positive first -> NDCG=1.0 artifact).
PFEDREC_VARIANTS = (
    "cached", "random", "shuffled", "calibrated",
    "global_head", "global_head_weighted", "global_head_cold",
    "alpha_grid", "alpha_oracle_group", "alpha_oracle_user", "alpha_policy_hc",
)
# Pre-registered alpha grid — NOT a CLI knob (Stage A pre-registration).
ALPHA_GRID = tuple(round(0.05 * i, 2) for i in range(21))
# Buckets with fewer fit-users than this fall back to the global alpha*.
MIN_BUCKET_USERS = 30
# Trust gate: |reeval - run D-06 best| must be below this on every ratio.
TRUST_GATE_TOL = 1e-6
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
    module: str = "personalized"
    lr_eta: float = 0.0
    loss: str = "bce"
    l2_regularization: float = 0.0
    source_config: Dict[str, Any] = field(default_factory=dict)
    manifest_excerpt: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.model_type not in ("bpr", "basic", "pfedrec"):
            raise ValueError(
                f"Unsupported model_type {self.model_type!r} (expected 'bpr'/'basic'/'pfedrec')"
            )
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if self.num_eval_negatives <= 0:
            raise ValueError(f"num_eval_negatives must be positive, got {self.num_eval_negatives}")
        if self.eval_round_num <= 0:
            raise ValueError(
                f"eval_round_num must be positive, got {self.eval_round_num} — "
                f"the run never produced a D-06 final eval round?"
            )


def resolve_config(run_dir: Path, calib_epochs: int = 1, module: str = "personalized") -> ReevalConfig:
    """Build a :class:`ReevalConfig` from ``<run_dir>/results.json``.

    Parameters
    ----------
    run_dir : Path
        Run directory containing ``results.json`` (and optionally a
        standalone ``manifest.json`` used as a fallback for ``_manifest``).
    calib_epochs : int
        Local epochs for the ``calibrated`` variant.
    module : str
        ``"personalized"`` (MF-family ``federated_config`` schema) or
        ``"pfedrec"`` (``latent_dim``/``lr``/``lr_eta``/``loss`` schema).
        Cross-checked against ``_manifest.module`` when present.

    Returns
    -------
    ReevalConfig
        Fully resolved replay configuration.

    Raises
    ------
    FileNotFoundError
        If ``results.json`` is missing.
    ValueError
        If mandatory provenance fields cannot be resolved, or the run dir
        belongs to a different module than requested.
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

    # Module cross-check: a pfedrec run replayed with the personalized
    # config schema silently resolves embedding_dim=64 / lr=0.001 — wrong
    # model shape and wrong calibration. Fail loudly instead.
    manifest_module = str(manifest.get("module") or "")
    if manifest_module and manifest_module != module:
        raise ValueError(
            f"run dir {run_dir} belongs to module {manifest_module!r} "
            f"but --module {module!r} was requested"
        )

    overrides = manifest.get("overrides") or {}
    manifest_excerpt = {
        k: manifest.get(k)
        for k in (
            "run_id", "mode", "module", "run_seed", "split_hash",
            "final_eval_round_index", "num_eval_negatives",
            "num_train_negatives", "fraction_train", "git_commit",
        )
    }
    common = dict(
        run_dir=run_dir,
        run_id=str(manifest.get("run_id", run_dir.name)),
        run_seed=int(manifest.get("run_seed", fed_cfg.get("run_seed", 42))),
        eval_round_num=eval_round,
        num_train_negatives=int(manifest.get("num_train_negatives", 4)),
        num_eval_negatives=int(manifest.get("num_eval_negatives", 99)),
        reuse_cache=bool(fed_cfg.get("reuse_cache", False)),
        split_hash=str(manifest.get("split_hash", "")),
        calib_epochs=int(calib_epochs),
        module=module,
        source_config=dict(fed_cfg),
        manifest_excerpt=manifest_excerpt,
    )

    if module == "pfedrec":
        # PFedRec's federated_config records latent_dim / lr / lr_eta / loss.
        # REQUIRE them — silently defaulting would rebuild the model at the
        # wrong dim (cache-shape mismatch) or calibrate at the wrong lr.
        missing = [k for k in ("latent_dim", "lr", "lr_eta") if k not in fed_cfg]
        if missing:
            raise ValueError(
                f"pfedrec federated_config is missing required field(s) {missing} "
                f"in {results_path} — refusing to guess (dim/lr defaults would be wrong)"
            )
        return ReevalConfig(
            model_type="pfedrec",
            embedding_dim=int(fed_cfg["latent_dim"]),
            dropout=0.0,
            lr=float(fed_cfg["lr"]),
            weight_decay=0.0,
            lr_eta=float(fed_cfg["lr_eta"]),
            loss=str(fed_cfg.get("loss", "bce")),
            l2_regularization=float(fed_cfg.get("l2_regularization", 0.0)),
            **common,
        )

    lr = float(fed_cfg.get("learning_rate", overrides.get("lr", 0.001)))
    return ReevalConfig(
        model_type=str(fed_cfg.get("model_type", "bpr")),
        embedding_dim=int(fed_cfg.get("embedding_dim", 64)),
        dropout=float(fed_cfg.get("dropout", 0.1)),
        lr=lr,
        weight_decay=1e-5,
        **common,
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
    user_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)


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

    def _stats_for(pid: int) -> Dict[str, Any]:
        # The four alpha inputs + group, straight from the foundation split
        # manifest (TRAIN-rows-only stats, CR-5). Missing entry -> zeros.
        entry = stats_map.get(int(pid)) if stats_map is not None else None
        return {
            "n_interactions": int(getattr(entry, "n_interactions", 0) or 0),
            "genre_entropy": float(getattr(entry, "genre_entropy", 0.0) or 0.0),
            "n_unique_items": int(getattr(entry, "n_unique_items", 0) or 0),
            "rating_std": float(getattr(entry, "rating_std", 0.0) or 0.0),
            "user_group": _group_for(pid),
        }

    grouped = ratings_df.groupby("user_idx")
    users: Dict[int, UserEvalData] = {}
    user_stats: Dict[int, Dict[str, Any]] = {}
    for pid in range(n):
        user_stats[pid] = _stats_for(pid)
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
        user_stats=user_stats,
    )


def load_staleness_map(run_dir: Path) -> Tuple[Dict[int, int], int]:
    """Per-user rounds-since-last-participation at the restored best round.

    Reads ``results.json``'s ``selected_clients_per_round`` (list of per-round
    partition-id lists; index r-1 = round r) and ``checkpoint.best_round``.
    ``staleness[pid] = best_round - max{r <= best_round : pid selected in r}``.
    Users never selected before best_round have NO entry — callers should
    default missing pids to the returned ``best_round`` sentinel.

    NOTE: for runs with ``final-calibration-enabled=true`` every user's head
    was re-trained AFTER best-round restore, so at D-06 eval time effective
    head staleness is ~0 for everyone — this map is an ANALYSIS covariate
    (pre-calibration staleness), not a live cache property.

    Returns ``({}, 0)`` when the fields are absent (older runs).
    """
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return {}, 0
    with open(results_path, "r") as f:
        results = json.load(f)
    selected = results.get("selected_clients_per_round") or []
    ckpt = results.get("checkpoint") or {}
    best_round = int(ckpt.get("best_round") or len(selected))
    if not selected or best_round <= 0:
        return {}, 0
    last_seen: Dict[int, int] = {}
    for r_index, round_pids in enumerate(selected[:best_round], start=1):
        for pid in round_pids:
            last_seen[int(pid)] = r_index
    return {pid: best_round - r for pid, r in last_seen.items()}, best_round


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
    _run_data: Optional[RunData] = None

    # ---- data (shared: the foundation snapshot is module-agnostic) --------

    def load_run_data(self, limit_users: Optional[int]) -> RunData:
        """Load (once) and memoize the per-user dataset snapshot."""
        if self._run_data is None:
            self._run_data = load_personalized_run_data(limit_users)
        return self._run_data

    def inject_run_data(self, run_data: RunData) -> None:
        """Inject a synthetic snapshot (self-test path; skips foundation load)."""
        self._run_data = run_data

    # ---- cache -------------------------------------------------------------

    def resolve_cache_dir(self, run_data: RunData) -> Path:
        """Resolve the run's per-user local-state cache directory."""
        raise NotImplementedError

    def verify_cache_signature(self, run_data: RunData) -> None:
        """Loud signature check of the cache's ``manifest.json``."""
        raise NotImplementedError

    # ---- model -------------------------------------------------------------

    def build_model(self, run_data: RunData) -> torch.nn.Module:
        """Construct the module's real model class."""
        raise NotImplementedError

    def blend_local(
        self,
        local_state: Dict[str, torch.Tensor],
        w_global: torch.Tensor,
        alpha: float,
    ) -> Dict[str, torch.Tensor]:
        """Blend the user's local state with a global head: alpha*local + (1-alpha)*global."""
        raise NotImplementedError(
            f"module {self.module_name!r} does not support head blending"
        )

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


class PfedrecAdapter(ModuleAdapter):
    """Adapter for ``federated-pfedrec`` (per-user linear head, IJCAI-23).

    LOCAL payload (single-key contract, D-21):
    ``OrderedDict({'affine_output.weight': (1, latent_dim)})`` at
    ``.embedding_cache/<run_id>/partition_{pid}.pt`` with a schema_v3
    ``manifest.json`` signature sidecar (10 fields).

    GLOBAL state-dict keys: ``embedding_item.weight``, ``affine_output.bias``
    (D-01: the bias is GLOBAL). ``set_global_parameters`` iterates only the
    model's ``_GLOBAL_PARAMS`` tuple, so extra keys in a loaded state dict
    (e.g. a Stage-B ``global_head``) are ignored by construction.

    Head blending: ``blend_local`` writes ``alpha*w_local + (1-alpha)*w_global``
    into ``affine_output.weight`` — the untouched model then computes exactly
    ``sigmoid((alpha*w_l + (1-alpha)*w_g) . q_i + b)``.
    """

    module_name = "pfedrec"

    def __init__(self, cfg: ReevalConfig, cache_base_dir: Optional[Path] = None):
        self.cfg = cfg
        self.cache_base_dir = (
            Path(cache_base_dir)
            if cache_base_dir is not None
            else _REPO_ROOT / "federated-pfedrec" / ".embedding_cache"
        )
        self._cache_dir: Optional[Path] = None
        self._run_data: Optional[RunData] = None

    # ---- cache -----------------------------------------------------------

    def _signature(self, run_data: RunData) -> Dict[str, Any]:
        """The 10-field schema_v3 signature (mirrors client_app._signature_fields)."""
        return {
            "schema_version": 3,
            "run_id": str(self.cfg.run_id),
            "method": "pfedrec",
            "num_users": int(run_data.num_users),
            "num_items": int(run_data.num_items),
            "latent_dim": int(self.cfg.embedding_dim),
            "split_hash": str(self.cfg.split_hash),
            "loss": str(self.cfg.loss),
            "num_train_negatives": int(self.cfg.num_train_negatives),
            "bias_classification": "global",
        }

    def resolve_cache_dir(self, run_data: RunData) -> Path:
        """Mirror client_app._cache_dir_for_run (D-18).

        ``reuse_cache=True`` hashes the signature EXCLUDING both ``run_id``
        AND ``schema_version`` — this differs from the personalized rule
        (which drops only ``run_id``); do not unify them.
        """
        if self._cache_dir is not None:
            return self._cache_dir
        if not self.cfg.reuse_cache:
            self._cache_dir = self.cache_base_dir / self.cfg.run_id
        else:
            signature = self._signature(run_data)
            payload = json.dumps(
                {k: v for k, v in signature.items() if k not in ("run_id", "schema_version")},
                sort_keys=True,
            ).encode("utf-8")
            self._cache_dir = self.cache_base_dir / f"sig_{hashlib.sha256(payload).hexdigest()[:16]}"
        return self._cache_dir

    def verify_cache_signature(self, run_data: RunData) -> None:
        """D-17-style loud check of all 10 schema_v3 signature fields."""
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
        for key in signature:
            if self.cfg.reuse_cache and key == "run_id":
                continue
            if cached.get(key) != signature.get(key):
                deltas.append(f"{key}: cached={cached.get(key)!r}, expected={signature.get(key)!r}")
        if deltas:
            raise RuntimeError(
                "Embedding-cache signature mismatch (D-17):\n  " + "\n  ".join(deltas)
            )

    def cached_local_path(self, pid: int) -> Path:
        assert self._cache_dir is not None, "resolve_cache_dir must run first"
        return self._cache_dir / f"partition_{int(pid)}.pt"

    def load_cached_local(self, pid: int) -> Dict[str, torch.Tensor]:
        # weights_only=True matches the pfedrec client convention (Pitfall 6).
        state = torch.load(str(self.cached_local_path(pid)), map_location="cpu", weights_only=True)
        assert set(state.keys()) == {"affine_output.weight"}, (
            f"D-21 violated on load: payload keys {sorted(state.keys())} for pid={pid}"
        )
        d = int(self.cfg.embedding_dim)
        shape = tuple(state["affine_output.weight"].shape)
        assert shape == (1, d), f"head shape {shape} != (1, {d}) for pid={pid}"
        return state

    # ---- model -----------------------------------------------------------

    def build_model(self, run_data: RunData) -> torch.nn.Module:
        from federated_pfedrec.task import get_model

        return get_model(
            num_items=run_data.num_items,
            latent_dim=self.cfg.embedding_dim,
        )

    def set_globals(self, model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
        model.set_global_parameters(state)

    def set_local(self, model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
        model.set_local_parameters(state, strict=True, run_id=self.cfg.run_id)

    # ---- local-state variants ---------------------------------------------

    def random_local(self, pid: int, run_seed: int) -> Dict[str, torch.Tensor]:
        """Kaiming-default head (D-19 cold-start init), NOT Xavier.

        ``nn.Linear`` default init on a ``(1, d)`` weight is
        ``kaiming_uniform_(a=sqrt(5))`` => bound ``1/sqrt(d)``. Drawn from
        ``torch_gen(run_seed, pid, -1, "model_init")`` for reproducibility.
        """
        d = int(self.cfg.embedding_dim)
        bound = 1.0 / math.sqrt(d)
        g = torch_gen(int(run_seed), int(pid), -1, "model_init")
        w = torch.empty(1, d).uniform_(-bound, bound, generator=g)
        return OrderedDict({"affine_output.weight": w})

    def zero_local(self) -> Dict[str, torch.Tensor]:
        raise RuntimeError(
            "variant 'zero' is degenerate for pfedrec: every candidate scores "
            "sigmoid(bias), the stable argsort ranks the held-out positive "
            "(candidate index 0) first -> NDCG@10 = 1.0 artifact. Use 'random' "
            "or 'global_head_cold' as the floor instead."
        )

    def blend_local(
        self,
        local_state: Dict[str, torch.Tensor],
        w_global: torch.Tensor,
        alpha: float,
    ) -> Dict[str, torch.Tensor]:
        w_local = local_state["affine_output.weight"]
        a = float(alpha)
        blended = a * w_local + (1.0 - a) * w_global.to(dtype=w_local.dtype)
        return OrderedDict({"affine_output.weight": blended})

    # ---- calibration -------------------------------------------------------

    def calibrate(
        self,
        model: torch.nn.Module,
        user: UserEvalData,
        cfg: ReevalConfig,
        device: str,
    ) -> None:
        """Replay the D-06.7 calibration pass for one user.

        Mirrors the server's calibration ConfigRecord: ``lr=run lr``,
        ``lr_eta=0.0`` (final-calibration-freeze-items default — item table
        frozen), ``proximal_mu=0.0``, ``round_num = final_eval_round_index``
        (server uses actual_rounds+1 for calibration AND D-06),
        ``local_epochs_override = calib_epochs``.

        Caveats: (1) on runs that already ran D-06.7 (e.g. cf72e0) the cached
        heads are ALREADY post-calibration — this variant is a SECOND
        calibration epoch, diagnostic only; (2) train_pfedrec_single_user's
        DataLoader shuffle is unseeded (upstream-documented as negligible),
        so this variant is not bit-reproducible across processes.
        """
        from federated_pfedrec.task import prepare_user_train_data, train_pfedrec_single_user

        round_num = cfg.eval_round_num
        items, ratings = prepare_user_train_data(
            user_idx=user.pid,
            user_train_items=[int(x) for x in user.train_items.tolist()],
            num_items=int(self._run_data.num_items) if self._run_data else 0,
            num_negatives=cfg.num_train_negatives,
            run_seed=cfg.run_seed,
            round_num=round_num,
            exclude_items=user.exclude_items,
        )
        if not items:
            return
        train_pfedrec_single_user(
            model,
            items,
            ratings,
            lr=cfg.lr,
            lr_eta=0.0,
            num_items=int(self._run_data.num_items) if self._run_data else 0,
            local_epochs=cfg.calib_epochs,
            batch_size=cfg.batch_size,
            device=torch.device(device),
            l2_regularization=cfg.l2_regularization,
            proximal_mu=0.0,
            run_seed=cfg.run_seed,
            user_idx=user.pid,
            round_num=round_num,
            exclude_items=user.exclude_items,
        )

    # ---- evaluation ---------------------------------------------------------

    def evaluate_user(
        self,
        model: torch.nn.Module,
        user: UserEvalData,
        cfg: ReevalConfig,
        device: str,
    ) -> Dict[str, float]:
        """Byte-paired replay of the client's ``@app.evaluate`` scoring step.

        Calls the module's OWN ``evaluate_pfedrec_sampled`` — candidate-pool
        ordering and the ``np_rng(..., "eval_neg")`` draw only bit-pair with
        the run's D-06 eval when the module's own function runs.
        """
        from federated_pfedrec.task import evaluate_pfedrec_sampled

        assert self._run_data is not None, "run data must be loaded/injected first"
        test_items = [int(user.test_item)] if user.test_item is not None else []
        return evaluate_pfedrec_sampled(
            model,
            test_items,
            {int(x) for x in user.train_items.tolist()},
            self._run_data.num_items,
            torch.device(device),
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
        ``"personalized"`` / ``"pfedrec"`` (implemented) or ``"adaptive"``
        (planned).
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
    if module == "pfedrec":
        return PfedrecAdapter(cfg, cache_base_dir=cache_base_dir)
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
    *,
    alpha_for_user: Optional[Any] = None,
    w_global: Optional[torch.Tensor] = None,
    staleness_map: Optional[Dict[int, int]] = None,
    label: Optional[str] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
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
        One of :data:`VARIANTS` — the LOCAL-STATE SOURCING rule. Head-blend
        passes use ``variant="cached"`` plus ``alpha_for_user``/``w_global``.
    device : str
        ``"cpu"`` or ``"cuda"``.
    alpha_for_user : Optional[Callable[[int], float]]
        When given, each user's local state is replaced by
        ``adapter.blend_local(local, w_global, alpha_for_user(pid))`` before
        scoring (``w_global`` required).
    w_global : Optional[torch.Tensor]
        Global head tensor for the blend.
    staleness_map : Optional[Dict[int, int]]
        Per-pid rounds-since-last-participation (analysis covariate for the
        per-user rows; missing pids are recorded as None).
    label : Optional[str]
        Display label for progress lines (defaults to ``variant``).

    Returns
    -------
    Tuple[Dict[str, float], List[Dict[str, Any]]]
        Aggregated thesis-table metrics (overall + per-group) and the
        per-user rows (one dict per pid, in pid order).
    """
    if alpha_for_user is not None and w_global is None:
        raise ValueError("alpha_for_user requires w_global")
    display = label or variant
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
    per_user_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, pid in enumerate(pids):
        user = run_data.users[pid]
        local_state = _resolve_local_state(adapter, variant, pid, cfg, shuffle_map)

        alpha_used: Optional[float] = None
        if alpha_for_user is not None:
            alpha_used = float(alpha_for_user(pid))
            local_state = adapter.blend_local(local_state, w_global, alpha_used)
        elif variant == "cached":
            alpha_used = 1.0  # provenance: pure-local scoring
        adapter.set_local(model, local_state)

        if variant == "calibrated":
            adapter.calibrate(model, user, cfg, device)
            # Restore the exact saved globals before scoring (mirrors the
            # D-06 broadcast evaluating against the server-restored arrays).
            adapter.set_globals(model, global_snapshot)
            model.to(device)

        sampled = adapter.evaluate_user(model, user, cfg, device)
        stats.add_user(sampled, user.user_group)

        u_stats = run_data.user_stats.get(pid, {})
        per_user_rows.append({
            "pid": pid,
            "user_group": user.user_group,
            "n_interactions": int(u_stats.get("n_interactions", len(user.train_items))),
            "staleness_at_best": (
                staleness_map.get(pid) if staleness_map is not None else None
            ),
            "alpha_used": alpha_used,
            "sampled_ndcg@10": float(sampled.get("sampled_ndcg@10", 0.0)),
            "sampled_hr@10": float(sampled.get("sampled_hr@10", 0.0)),
            "sampled_mrr": float(sampled.get("sampled_mrr", 0.0)),
            "eval_loss": float(sampled.get("eval_loss", float("nan"))),
            "evaluated": int(sampled.get("sampled_num_users", 0)),
        })

        if (i + 1) % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            partial = stats.to_metrics()
            print(
                f"[reeval:{display}] {i + 1}/{len(pids)} users "
                f"({rate:.0f} u/s) | running ndcg@10={partial['sampled_ndcg@10']:.4f} "
                f"hr@10={partial['sampled_hr@10']:.4f}"
            )

    metrics = stats.to_metrics()
    print(
        f"[reeval:{display}] done: {len(pids)} pids in {time.time() - t0:.1f}s "
        f"({metrics['evaluated_users']} evaluated)"
    )
    return metrics, per_user_rows


# ===========================================================================
# Stage A machinery: trust gate, global heads, alpha suite (pfedrec)
# ===========================================================================


def trust_gate(
    reeval_metrics: Dict[str, float],
    best_block: Dict[str, Any],
    *,
    limit_users: Optional[int],
    globals_kind: str,
    eval_round_num: int,
    final_eval_round_index: Optional[int],
) -> Dict[str, Any]:
    """Compare the ``cached`` replay against the run's own D-06 best block.

    The sidecar-level "valid" label certifies the CODE had the fix; only a
    bit-match here proves the OFFLINE REPLAY reproduces the run's D-06 eval.
    No downstream variant number is trustworthy unless this gate PASSes.

    Key mapping: reeval emits underscore-form group counts
    (``evaluated_users_sparse``); the run block uses slash-form
    (``evaluated_users/sparse``). ``sampled_mrr`` / ``eval_loss`` are absent
    from the run block and are skipped.

    Returns the verdict dict (embedded into the JSON dump); status is one of
    ``PASS`` / ``FAIL`` / ``SKIPPED``.
    """
    preconditions: Dict[str, Any] = {
        "limit_users_unset": limit_users is None,
        "globals_is_best": globals_kind == "best",
        "d06_cache_misses": best_block.get("d06_cache_misses"),
        "round_num_matches": (
            final_eval_round_index is None or int(final_eval_round_index) == int(eval_round_num)
        ),
    }
    if limit_users is not None or globals_kind != "best":
        return {
            "status": "SKIPPED",
            "reason": "trust gate only defined for full-population runs against globals=best",
            "preconditions": preconditions,
        }

    deltas: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []
    max_abs_delta = 0.0

    ratio_keys = ["sampled_hr@10", "sampled_ndcg@10"] + [
        f"{m}/{g}" for m in ("sampled_hr@10", "sampled_ndcg@10") for g in USER_GROUPS
    ]
    for key in ratio_keys:
        ours = reeval_metrics.get(key)
        theirs = best_block.get(key)
        if theirs is None:
            deltas[key] = {"status": "not_compared", "reeval": ours, "run": None}
            continue
        delta = abs(float(ours) - float(theirs))
        max_abs_delta = max(max_abs_delta, delta)
        ok = delta < TRUST_GATE_TOL
        deltas[key] = {"reeval": float(ours), "run": float(theirs), "delta": delta, "ok": ok}
        if not ok:
            failures.append(key)

    count_pairs = [("evaluated_users", "evaluated_users")] + [
        (f"evaluated_users_{g}", f"evaluated_users/{g}") for g in USER_GROUPS
    ]
    for ours_key, theirs_key in count_pairs:
        ours = int(round(float(reeval_metrics.get(ours_key, -1))))
        theirs_raw = best_block.get(theirs_key)
        if theirs_raw is None:
            deltas[theirs_key] = {"status": "not_compared", "reeval": ours, "run": None}
            continue
        theirs = int(round(float(theirs_raw)))
        ok = ours == theirs
        deltas[theirs_key] = {"reeval": ours, "run": theirs, "ok": ok}
        if not ok:
            failures.append(theirs_key)

    status = "PASS" if not failures else "FAIL"
    return {
        "status": status,
        "tolerance": TRUST_GATE_TOL,
        "max_abs_delta": max_abs_delta,
        "failed_fields": failures,
        "deltas": deltas,
        "preconditions": preconditions,
        "not_compared": ["sampled_mrr", "eval_loss"],
    }


def compute_global_head(
    adapter: "PfedrecAdapter",
    run_data: RunData,
    mode: str,
    run_seed: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Derive a population head ``w_global`` — no such artifact exists in a
    vanilla PFedRec run (the server aggregates only item table + bias).

    Modes
    -----
    - ``uniform``  : plain mean of all cached heads — CANONICAL for the alpha
      blends (matches the run's ``weight_policy: uniform`` / D-24 semantics).
    - ``weighted`` : mean weighted by ``n_interactions`` (what a
      num-examples FedAvg over heads would produce) — reported alternative.
    - ``cold``     : one Kaiming-default draw — sanity floor only.
    """
    d = int(adapter.cfg.embedding_dim)
    if mode == "cold":
        bound = 1.0 / math.sqrt(d)
        g = torch_gen(int(run_seed), -1, -1, "model_init")
        w = torch.empty(1, d).uniform_(-bound, bound, generator=g)
        n_heads = 0
    else:
        acc = torch.zeros(1, d, dtype=torch.float64)
        weight_sum = 0.0
        n_heads = 0
        for pid in sorted(run_data.users.keys()):
            head = adapter.load_cached_local(pid)["affine_output.weight"].to(torch.float64)
            if mode == "weighted":
                wgt = float(run_data.user_stats.get(pid, {}).get("n_interactions", 0)) or 1.0
            else:
                wgt = 1.0
            acc += wgt * head
            weight_sum += wgt
            n_heads += 1
        if weight_sum <= 0:
            raise RuntimeError(f"global head mode={mode!r}: no heads accumulated")
        w = (acc / weight_sum).to(torch.float32)
    spec = {
        "mode": mode,
        "n_heads": n_heads,
        "sha256": hashlib.sha256(w.numpy().tobytes()).hexdigest(),
        "norm": float(w.norm()),
    }
    return w, spec


_PER_USER_FIELDS = (
    "pid", "user_group", "n_interactions", "staleness_at_best", "alpha_used",
    "sampled_ndcg@10", "sampled_hr@10", "eval_loss", "evaluated",
)


def write_per_user_csv(path: Path, rows: List[Dict[str, Any]]) -> Path:
    """Write the standard per-user rows CSV (thesis queue #4 artifact)."""
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_PER_USER_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[reeval] wrote {path} ({len(rows)} rows)")
    return path


def write_grid_per_user_csv(
    path: Path,
    matrix: Dict[float, Dict[int, Dict[str, Any]]],
    run_data: RunData,
    staleness_map: Optional[Dict[int, int]],
) -> Path:
    """Wide per-user CSV over the whole alpha grid (ndcg/hr/loss per alpha)."""
    import csv

    alphas = sorted(matrix.keys())
    pids = sorted(next(iter(matrix.values())).keys()) if matrix else []
    header = ["pid", "user_group", "n_interactions", "staleness_at_best"]
    for a in alphas:
        header.append(f"ndcg@a{a:.2f}")
    for a in alphas:
        header.append(f"hr@a{a:.2f}")
    for a in alphas:
        header.append(f"loss@a{a:.2f}")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for pid in pids:
            stats = run_data.user_stats.get(pid, {})
            row: List[Any] = [
                pid,
                run_data.users[pid].user_group,
                int(stats.get("n_interactions", 0)),
                staleness_map.get(pid) if staleness_map is not None else None,
            ]
            row += [matrix[a][pid]["sampled_ndcg@10"] for a in alphas]
            row += [matrix[a][pid]["sampled_hr@10"] for a in alphas]
            row += [matrix[a][pid]["eval_loss"] for a in alphas]
            writer.writerow(row)
    print(f"[reeval] wrote {path} ({len(pids)} rows x {len(header)} cols)")
    return path


def _aggregate_assignment(
    matrix: Dict[float, Dict[int, Dict[str, Any]]],
    alpha_by_pid: Dict[int, float],
    run_data: RunData,
) -> Dict[str, float]:
    """SuffStats aggregation of per-user grid entries under an alpha assignment."""
    stats = SuffStats()
    for pid, a in alpha_by_pid.items():
        r = matrix[a][pid]
        stats.add_user(
            {
                "sampled_num_users": r["evaluated"],
                "sampled_hr@10": r["sampled_hr@10"],
                "sampled_ndcg@10": r["sampled_ndcg@10"],
                "sampled_mrr": r["sampled_mrr"],
            },
            run_data.users[pid].user_group,
        )
    return stats.to_metrics()


def _alpha_star(
    matrix: Dict[float, Dict[int, Dict[str, Any]]],
    pids: List[int],
) -> float:
    """Best single alpha on a pid subset by micro-averaged NDCG@10.

    Iterates the grid ascending with ``>=`` so ties break toward the LARGEST
    alpha (toward pure-local — conservative for the oracle histograms).
    """
    best_a, best_v = ALPHA_GRID[0], -1.0
    for a in sorted(matrix.keys()):
        num = 0.0
        den = 0
        for pid in pids:
            r = matrix[a][pid]
            n = int(r["evaluated"])
            if n > 0:
                num += r["sampled_ndcg@10"] * n
                den += n
        v = (num / den) if den else 0.0
        if v >= best_v:
            best_v, best_a = v, a
    return best_a


def _bucket_families(
    run_data: RunData,
    pids: List[int],
    staleness_map: Optional[Dict[int, int]],
) -> Dict[str, Dict[int, str]]:
    """The three pre-registered bucket families for the group oracle."""
    families: Dict[str, Dict[int, str]] = {}

    families["user_group"] = {pid: run_data.users[pid].user_group for pid in pids}

    n_values = np.asarray(
        [int(run_data.user_stats.get(pid, {}).get("n_interactions", 0)) for pid in pids],
        dtype=np.float64,
    )
    if len(n_values) >= 10:
        edges = np.percentile(n_values, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        families["n_interactions_decile"] = {
            pid: f"decile_{int(np.searchsorted(edges, n, side='right'))}"
            for pid, n in zip(pids, n_values)
        }

    if staleness_map:
        stale_edges = ((0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, None))

        def _stale_label(s: Optional[int]) -> str:
            if s is None:
                return "unknown"
            for lo, hi in stale_edges:
                if hi is None:
                    if s >= lo:
                        return f"{lo}plus"
                elif lo <= s <= hi:
                    return f"{lo}" if lo == hi else f"{lo}-{hi}"
            return "unknown"

        families["staleness"] = {pid: _stale_label(staleness_map.get(pid)) for pid in pids}

    return families


def oracle_group(
    matrix: Dict[float, Dict[int, Dict[str, Any]]],
    run_data: RunData,
    staleness_map: Optional[Dict[int, int]],
) -> Dict[str, Any]:
    """Per-bucket oracle alpha — in-sample AND cross-fitted (the gate quantity).

    Cross-fit: alpha* per bucket is chosen on even pids and applied to odd
    pids, and vice versa — removing the only overfitting objection to the
    gate at ~zero cost. Buckets with < MIN_BUCKET_USERS fit-users fall back
    to the fit-half's global alpha*.
    """
    pids = sorted(next(iter(matrix.values())).keys())
    families = _bucket_families(run_data, pids, staleness_map)

    out: Dict[str, Any] = {"families": {}, "min_bucket_users": MIN_BUCKET_USERS}
    best_family = None
    best_crossfit_ndcg = -1.0

    for family, bucket_of in families.items():
        buckets = sorted(set(bucket_of.values()))
        members: Dict[str, List[int]] = {b: [] for b in buckets}
        for pid in pids:
            members[bucket_of[pid]].append(pid)

        # ---- in-sample ----------------------------------------------------
        global_star = _alpha_star(matrix, pids)
        in_sample_star: Dict[str, float] = {}
        for b in buckets:
            in_sample_star[b] = (
                _alpha_star(matrix, members[b])
                if len(members[b]) >= MIN_BUCKET_USERS
                else global_star
            )
        in_sample_assign = {pid: in_sample_star[bucket_of[pid]] for pid in pids}
        in_sample_metrics = _aggregate_assignment(matrix, in_sample_assign, run_data)

        # ---- cross-fitted ---------------------------------------------------
        halves = {
            "even": [p for p in pids if p % 2 == 0],
            "odd": [p for p in pids if p % 2 == 1],
        }
        crossfit_assign: Dict[int, float] = {}
        half_stars: Dict[str, Dict[str, float]] = {}
        for fit_name, apply_name in (("even", "odd"), ("odd", "even")):
            fit_pids = halves[fit_name]
            fit_global_star = _alpha_star(matrix, fit_pids)
            stars: Dict[str, float] = {}
            for b in buckets:
                fit_members = [p for p in members[b] if p in set(fit_pids)]
                stars[b] = (
                    _alpha_star(matrix, fit_members)
                    if len(fit_members) >= MIN_BUCKET_USERS
                    else fit_global_star
                )
            half_stars[fit_name] = stars
            for pid in halves[apply_name]:
                crossfit_assign[pid] = stars[bucket_of[pid]]
        crossfit_metrics = _aggregate_assignment(matrix, crossfit_assign, run_data)

        bucket_table = [
            {
                "bucket": b,
                "n_users": len(members[b]),
                "alpha_star_in_sample": in_sample_star[b],
                "alpha_star_fit_even": half_stars["even"][b],
                "alpha_star_fit_odd": half_stars["odd"][b],
            }
            for b in buckets
        ]
        out["families"][family] = {
            "buckets": bucket_table,
            "alpha_star_global": global_star,
            "in_sample": in_sample_metrics,
            "crossfit": crossfit_metrics,
        }
        if float(crossfit_metrics["sampled_ndcg@10"]) > best_crossfit_ndcg:
            best_crossfit_ndcg = float(crossfit_metrics["sampled_ndcg@10"])
            best_family = family

    out["best_family"] = best_family
    out["best_crossfit_ndcg"] = best_crossfit_ndcg
    return out


def oracle_user(
    matrix: Dict[float, Dict[int, Dict[str, Any]]],
    run_data: RunData,
) -> Tuple[Dict[str, float], Dict[int, float], Dict[str, int]]:
    """Per-user oracle alpha — LOOSE UPPER BOUND ONLY (fit on the single
    held-out item => selection on noise). Report, never gate on it, except
    as a refutation bound. Ties break toward the largest alpha.
    """
    pids = sorted(next(iter(matrix.values())).keys())
    alphas = sorted(matrix.keys())
    assign: Dict[int, float] = {}
    for pid in pids:
        best_a, best_v = alphas[0], -1.0
        for a in alphas:
            v = matrix[a][pid]["sampled_ndcg@10"]
            if v >= best_v:
                best_v, best_a = v, a
        assign[pid] = best_a
    metrics = _aggregate_assignment(matrix, assign, run_data)
    histogram: Dict[str, int] = {}
    for a in alphas:
        histogram[f"{a:.2f}"] = sum(1 for v in assign.values() if v == a)
    return metrics, assign, histogram


def stage_a_verdict(
    gate_ndcg: float,
    user_oracle_ndcg: float,
    reference_ndcg: float,
    reference_groups: Dict[str, float],
    crossfit_groups: Dict[str, float],
    alpha_star_global: float,
    bucket_stars: List[float],
) -> Dict[str, Any]:
    """Pre-registered Stage-A decision bands (see plan; cross-seed sigma~0.005).

    GO      : delta_gate >= +0.010 overall, OR >= +0.015 on one user group
              with overall delta_gate >= 0.
    MARGINAL: +0.005 <= delta_gate < +0.010.
    NO-GO   : delta_gate < +0.005; HARD NO-GO regardless when the per-user
              oracle can't clear +0.010 or the grid optimum is pure-local
              (alpha*=1.0 and every bucket >= 0.95).
    """
    delta_gate = float(gate_ndcg) - float(reference_ndcg)
    delta_user = float(user_oracle_ndcg) - float(reference_ndcg)
    group_deltas = {
        g: float(crossfit_groups.get(g, 0.0)) - float(reference_groups.get(g, 0.0))
        for g in USER_GROUPS
    }

    hard_nogo_reasons: List[str] = []
    if delta_user < 0.010:
        hard_nogo_reasons.append(
            f"per-user oracle delta {delta_user:+.4f} < +0.010 — even cherry-picked "
            f"alpha can't clear 2 sigma; no honest policy ever will"
        )
    if alpha_star_global >= 1.0 and bucket_stars and all(a >= 0.95 for a in bucket_stars):
        hard_nogo_reasons.append(
            "grid optimum is pure-local (alpha*=1.0) and every bucket selects alpha >= 0.95"
        )

    if hard_nogo_reasons:
        band = "NO-GO"
    elif delta_gate >= 0.010 or (delta_gate >= 0.0 and any(d >= 0.015 for d in group_deltas.values())):
        band = "GO"
    elif delta_gate >= 0.005:
        band = "MARGINAL"
    else:
        band = "NO-GO"

    return {
        "band": band,
        "delta_gate": delta_gate,
        "delta_user_oracle": delta_user,
        "group_deltas": group_deltas,
        "reference_ndcg": float(reference_ndcg),
        "hard_nogo_reasons": hard_nogo_reasons,
        "thresholds": {
            "go_overall": 0.010,
            "go_single_group": 0.015,
            "marginal_low": 0.005,
            "hard_user_oracle": 0.010,
        },
    }


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
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``<run-dir>/reeval_<variant>_<globals>.json`` with full provenance.

    ``schema_version`` 2 (additive over 1): optional keys ``trust_gate``,
    ``alpha`` / ``alpha_policy``, ``w_global_spec``, ``metrics_by_alpha``,
    ``bucket_tables``, ``verdict``, ``per_user_csv`` may appear via ``extra``.
    """
    out_path = cfg.run_dir / f"reeval_{variant}_{globals_kind}.json"
    payload = {
        "tool": "scripts/thesis/reeval.py",
        "schema_version": 2,
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
    if extra:
        payload.update(extra)
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
# pfedrec suite driver (Stage A: trust gate -> baselines -> alpha suite)
# ===========================================================================


_GLOBAL_HEAD_MODES = {
    "global_head": "uniform",
    "global_head_weighted": "weighted",
    "global_head_cold": "cold",
}


def _order_pfedrec_variants(requested: List[str]) -> List[str]:
    """Canonical execution order: cached (trust gate) first, grid before oracles."""
    ordered = list(dict.fromkeys(requested))  # dedupe, keep user order
    oracles = {"alpha_oracle_group", "alpha_oracle_user"}
    if oracles & set(ordered) and "alpha_grid" not in ordered:
        first_oracle = min(ordered.index(v) for v in ordered if v in oracles)
        ordered.insert(first_oracle, "alpha_grid")
        print("[reeval] auto-inserted 'alpha_grid' (required by the oracle variants)")
    if "alpha_grid" in ordered:
        # grid must precede any oracle
        grid_i = ordered.index("alpha_grid")
        for v in list(oracles & set(ordered)):
            if ordered.index(v) < grid_i:
                ordered.remove(v)
                ordered.append(v)
    if "cached" in ordered:
        ordered.remove("cached")
        ordered.insert(0, "cached")
    return ordered


def run_pfedrec_suite(
    args: argparse.Namespace,
    cfg: ReevalConfig,
    adapter: "PfedrecAdapter",
    run_data: RunData,
    global_state: Dict[str, torch.Tensor],
    globals_path: Path,
) -> int:
    """Execute the requested pfedrec variants with trust gate + artifacts.

    Returns a process exit code; raises ``SystemExit(3)`` on trust-gate FAIL
    so no downstream variant number can be mistaken for a verified one.
    """
    if "zero" in args.local_variant:
        print(
            "error: variant 'zero' is degenerate for pfedrec (all candidates score "
            "sigmoid(bias) -> NDCG=1.0 artifact); use 'random'/'global_head_cold'.",
            file=sys.stderr,
        )
        return 2
    unknown = [v for v in args.local_variant if v not in PFEDREC_VARIANTS]
    if unknown:
        print(f"error: variant(s) {unknown} not supported for module=pfedrec", file=sys.stderr)
        return 2

    ordered = _order_pfedrec_variants(list(args.local_variant))

    # Run-block reference (ground truth for the trust gate + gate deltas).
    with open(cfg.run_dir / "results.json", "r") as f:
        _results = json.load(f)
    best_block: Dict[str, Any] = (_results.get("final_metrics") or {}).get("best") or {}
    final_eval_round_index = (_results.get("final_metrics") or {}).get("final_eval_round_index")
    reference_ndcg = best_block.get("sampled_ndcg@10")
    reference_groups = {
        g: float(best_block.get(f"sampled_ndcg@10/{g}", 0.0)) for g in USER_GROUPS
    }

    raw_stale, best_round = load_staleness_map(cfg.run_dir)
    staleness_map: Optional[Dict[int, int]] = None
    if best_round > 0:
        staleness_map = {
            pid: raw_stale.get(pid, best_round) for pid in run_data.users
        }

    heads: Dict[str, Tuple[torch.Tensor, Dict[str, Any]]] = {}

    def _head(mode: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if mode not in heads:
            print(f"[reeval] building global head (mode={mode})...")
            heads[mode] = compute_global_head(adapter, run_data, mode, cfg.run_seed)
        return heads[mode]

    def _emit(
        variant: str,
        metrics: Dict[str, float],
        rows: Optional[List[Dict[str, Any]]],
        extra: Dict[str, Any],
    ) -> None:
        print_table(variant, args.globals, cfg, metrics)
        csv_path: Optional[Path] = None
        if rows is not None and not args.no_per_user:
            csv_path = write_per_user_csv(
                cfg.run_dir / f"reeval_{variant}_{args.globals}_per_user.csv", rows
            )
        if not args.no_json:
            payload_extra = dict(extra)
            if csv_path is not None:
                payload_extra["per_user_csv"] = str(csv_path)
            dump_json(
                cfg, "pfedrec", variant, args.globals, globals_path, metrics,
                device=args.device, limit_users=args.limit_users,
                cache_dir=adapter.resolve_cache_dir(run_data), extra=payload_extra,
            )

    grid_matrix: Optional[Dict[float, Dict[int, Dict[str, Any]]]] = None
    grid_full_metrics: Dict[float, Dict[str, float]] = {}
    summary: Dict[str, Dict[str, float]] = {}

    for variant in ordered:
        if variant in ("cached", "random", "shuffled", "calibrated"):
            metrics, rows = run_reeval(
                adapter, cfg, run_data, global_state, variant,
                device=args.device, staleness_map=staleness_map,
            )
            extra: Dict[str, Any] = {}
            if variant == "cached" and not args.skip_trust_gate:
                verdict = trust_gate(
                    metrics, best_block,
                    limit_users=args.limit_users, globals_kind=args.globals,
                    eval_round_num=cfg.eval_round_num,
                    final_eval_round_index=final_eval_round_index,
                )
                extra["trust_gate"] = verdict
                print(
                    f"[reeval] TRUST GATE: {verdict['status']} "
                    f"(max|delta|={verdict.get('max_abs_delta', float('nan')):.2e}, "
                    f"tol={TRUST_GATE_TOL:.0e})"
                )
                if verdict["status"] == "FAIL":
                    for fld in verdict["failed_fields"]:
                        d = verdict["deltas"][fld]
                        print(f"  FAIL {fld}: reeval={d.get('reeval')} run={d.get('run')}")
                    _emit(variant, metrics, rows, extra)
                    print(
                        "[reeval] trust gate FAILED — the offline replay does not "
                        "reproduce the run's D-06 eval; NO variant number is "
                        "trustworthy. Aborting.", file=sys.stderr,
                    )
                    raise SystemExit(3)
            _emit(variant, metrics, rows, extra)
            summary[variant] = metrics

        elif variant in _GLOBAL_HEAD_MODES:
            mode = _GLOBAL_HEAD_MODES[variant]
            w, spec = _head(mode)
            metrics, rows = run_reeval(
                adapter, cfg, run_data, global_state, "cached",
                device=args.device, staleness_map=staleness_map,
                alpha_for_user=lambda pid: 0.0, w_global=w, label=variant,
            )
            _emit(variant, metrics, rows, {"alpha": 0.0, "w_global_spec": spec})
            summary[variant] = metrics

        elif variant == "alpha_grid":
            w, spec = _head("uniform")
            grid_matrix = {}
            metrics_by_alpha: Dict[str, Dict[str, float]] = {}
            for a in ALPHA_GRID:
                m, rows = run_reeval(
                    adapter, cfg, run_data, global_state, "cached",
                    device=args.device, staleness_map=staleness_map,
                    alpha_for_user=lambda pid, _a=a: _a, w_global=w,
                    label=f"alpha={a:.2f}",
                )
                grid_matrix[a] = {r["pid"]: r for r in rows}
                grid_full_metrics[a] = m
                metrics_by_alpha[f"{a:.2f}"] = {
                    k: m[k]
                    for k in m
                    if k.startswith("sampled_ndcg@10") or k.startswith("sampled_hr@10")
                }
            pids_all = sorted(run_data.users.keys())
            star = _alpha_star(grid_matrix, pids_all)
            metrics = grid_full_metrics[star]
            csv_path = None
            if not args.no_per_user:
                csv_path = write_grid_per_user_csv(
                    cfg.run_dir / f"reeval_alpha_grid_{args.globals}_per_user.csv",
                    grid_matrix, run_data, staleness_map,
                )
            extra = {
                "alpha_grid": [float(a) for a in ALPHA_GRID],
                "alpha_star_global": float(star),
                "metrics_by_alpha": metrics_by_alpha,
                "w_global_spec": spec,
            }
            if csv_path is not None:
                extra["per_user_csv"] = str(csv_path)
            print_table(f"alpha_grid (alpha*={star:.2f})", args.globals, cfg, metrics)
            if not args.no_json:
                dump_json(
                    cfg, "pfedrec", variant, args.globals, globals_path, metrics,
                    device=args.device, limit_users=args.limit_users,
                    cache_dir=adapter.resolve_cache_dir(run_data), extra=extra,
                )
            summary[f"alpha_grid@{star:.2f}"] = metrics

        elif variant == "alpha_oracle_group":
            assert grid_matrix is not None, "grid must run before oracles (driver bug)"
            result = oracle_group(grid_matrix, run_data, staleness_map)
            best_family = result["best_family"]
            metrics = result["families"][best_family]["crossfit"]
            # The per-user oracle is needed for the hard NO-GO check even when
            # not requested — it is free post-processing.
            user_metrics, _assign, _hist = oracle_user(grid_matrix, run_data)
            verdict: Optional[Dict[str, Any]] = None
            if reference_ndcg is not None and args.limit_users is None:
                pids_all = sorted(run_data.users.keys())
                bucket_stars = [
                    float(b["alpha_star_in_sample"])
                    for fam in result["families"].values()
                    for b in fam["buckets"]
                ]
                verdict = stage_a_verdict(
                    gate_ndcg=float(metrics["sampled_ndcg@10"]),
                    user_oracle_ndcg=float(user_metrics["sampled_ndcg@10"]),
                    reference_ndcg=float(reference_ndcg),
                    reference_groups=reference_groups,
                    crossfit_groups={
                        g: float(metrics.get(f"sampled_ndcg@10/{g}", 0.0)) for g in USER_GROUPS
                    },
                    alpha_star_global=float(_alpha_star(grid_matrix, pids_all)),
                    bucket_stars=bucket_stars,
                )
                print(
                    f"[reeval] STAGE-A GATE: {verdict['band']} | "
                    f"delta_gate={verdict['delta_gate']:+.4f} "
                    f"(oracle_group crossfit {metrics['sampled_ndcg@10']:.4f} vs run "
                    f"{float(reference_ndcg):.4f}) | delta_user_oracle="
                    f"{verdict['delta_user_oracle']:+.4f}"
                )
                for reason in verdict["hard_nogo_reasons"]:
                    print(f"  HARD NO-GO: {reason}")
            print_table(f"oracle_group[{best_family}] crossfit", args.globals, cfg, metrics)
            if not args.no_json:
                dump_json(
                    cfg, "pfedrec", variant, args.globals, globals_path, metrics,
                    device=args.device, limit_users=args.limit_users,
                    cache_dir=adapter.resolve_cache_dir(run_data),
                    extra={
                        "bucket_tables": result,
                        "gate_quantity": "crossfit best-family overall sampled_ndcg@10",
                        "verdict": verdict,
                        "user_oracle_metrics_for_verdict": user_metrics,
                    },
                )
            summary[f"oracle_group[{best_family}]"] = metrics

        elif variant == "alpha_oracle_user":
            assert grid_matrix is not None, "grid must run before oracles (driver bug)"
            metrics, assign, histogram = oracle_user(grid_matrix, run_data)
            print_table("oracle_user (UPPER BOUND ONLY)", args.globals, cfg, metrics)
            if not args.no_json:
                dump_json(
                    cfg, "pfedrec", variant, args.globals, globals_path, metrics,
                    device=args.device, limit_users=args.limit_users,
                    cache_dir=adapter.resolve_cache_dir(run_data),
                    extra={
                        "upper_bound_only": True,
                        "alpha_histogram": histogram,
                        "note": (
                            "per-user alpha fit on the single held-out item — "
                            "selection on noise; NEVER a gate quantity"
                        ),
                    },
                )
            summary["oracle_user"] = metrics

        elif variant == "alpha_policy_hc":
            from federated_adaptive_personalized_cf.models.adaptive_alpha import (  # noqa: E402
                AlphaConfig,
                HierarchicalConditionalAlphaConfig,
                create_alpha_computer,
            )

            computer = create_alpha_computer(
                AlphaConfig(method="hierarchical_conditional"),
                hc_config=HierarchicalConditionalAlphaConfig(),
            )

            def _hc_alpha(pid: int) -> float:
                return float(computer.compute_from_stats(run_data.user_stats.get(pid, {})))

            w, spec = _head("uniform")
            metrics, rows = run_reeval(
                adapter, cfg, run_data, global_state, "cached",
                device=args.device, staleness_map=staleness_map,
                alpha_for_user=_hc_alpha, w_global=w, label=variant,
            )
            alphas_used = [r["alpha_used"] for r in rows if r["alpha_used"] is not None]
            _emit(variant, metrics, rows, {
                "alpha_policy": "hierarchical_conditional (default config)",
                "w_global_spec": spec,
                "alpha_summary": {
                    "mean": float(np.mean(alphas_used)) if alphas_used else None,
                    "min": float(np.min(alphas_used)) if alphas_used else None,
                    "max": float(np.max(alphas_used)) if alphas_used else None,
                },
            })
            summary[variant] = metrics

        else:  # pragma: no cover — guarded by the choices validation above
            raise ValueError(f"unhandled pfedrec variant {variant!r}")

    if len(summary) > 1:
        print("\n[reeval] cross-variant summary (overall):")
        print(f" {'variant':<26}{'hr@10':>10}{'ndcg@10':>10}{'ndcg sparse':>13}{'medium':>10}{'dense':>10}")
        for name, m in summary.items():
            print(
                f" {name:<26}{m['sampled_hr@10']:>10.4f}{m['sampled_ndcg@10']:>10.4f}"
                f"{m['sampled_ndcg@10/sparse']:>13.4f}{m['sampled_ndcg@10/medium']:>10.4f}"
                f"{m['sampled_ndcg@10/dense']:>10.4f}"
            )
    return 0


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
            metrics, _rows = run_reeval(adapter, cfg, run_data, loaded_globals, variant, device="cpu")
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
        print("[self-test] PASS (personalized) — all 5 variants finite; zero/random differ from cached.")

    _self_test_pfedrec()
    return 0


def _self_test_pfedrec() -> None:
    """Synthetic end-to-end validation of the pfedrec adapter + alpha suite."""
    print("\n[self-test:pfedrec] building synthetic run (20 users, 50 items, dim 8)...")
    from federated_pfedrec.task import get_model

    num_users, num_items, dim = 20, 50, 8
    run_seed, eval_round, best_round = 123, 7, 5
    run_id = "selftest-pfedrec-feed42"

    with tempfile.TemporaryDirectory(prefix="reeval_selftest_pfr_") as tmp:
        tmp_path = Path(tmp)
        run_dir = tmp_path / "run"
        cache_base = tmp_path / "embedding_cache"
        run_dir.mkdir()
        (cache_base / run_id).mkdir(parents=True)

        # --- real PFedRecMLP with controlled globals -----------------------
        model = get_model(num_items=num_items, latent_dim=dim)
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            emb = torch.randn(num_items, dim, generator=g)
            emb = emb / emb.norm(dim=1, keepdim=True)  # unit rows
            model.embedding_item.weight.copy_(emb)
            model.affine_output.bias.zero_()
        global_state = model.get_global_parameters()
        torch.save(global_state, run_dir / "global_state_best.pt")

        # --- fabricated pfedrec-shaped results.json ------------------------
        selected = [[pid for pid in range(num_users) if (pid + r) % 3 == 0] for r in range(6)]
        results = {
            "_manifest": {
                "run_id": run_id,
                "mode": "paper_compat_pfedrec",
                "module": "pfedrec",
                "run_seed": run_seed,
                "split_hash": "selftest-split-hash",
                "final_eval_round_index": eval_round,
                "num_eval_negatives": 20,
                "num_train_negatives": 2,
                "fraction_train": 0.1,
                "git_commit": "selftest",
            },
            "federated_config": {
                "latent_dim": dim,
                "lr": 0.1,
                "lr_eta": 80.0,
                "loss": "bce",
                "l2_regularization": 0.0,
                "reuse_cache": False,
                "num_clients": num_users,
            },
            "final_metrics": {"final_eval_round_index": eval_round},
            "training_rounds": 6,
            "checkpoint": {"best_round": best_round, "rule": "best_round_restore"},
            "selected_clients_per_round": selected,
        }
        with open(run_dir / "results.json", "w") as f:
            json.dump(results, f)

        # --- config resolution: pfedrec schema + module cross-check --------
        cfg = resolve_config(run_dir, module="pfedrec")
        assert cfg.model_type == "pfedrec" and cfg.embedding_dim == dim
        assert cfg.lr == 0.1 and cfg.lr_eta == 80.0 and cfg.loss == "bce"
        try:
            resolve_config(run_dir, module="personalized")
            raise AssertionError("module mismatch not rejected")
        except ValueError:
            pass

        # --- staleness map ---------------------------------------------------
        stale, br = load_staleness_map(run_dir)
        assert br == best_round and all(0 <= v <= best_round for v in stale.values())

        # --- synthetic users + REAL-format cache payloads -------------------
        data_rng = np.random.default_rng(99)
        users: Dict[int, UserEvalData] = {}
        user_stats: Dict[int, Dict[str, Any]] = {}
        groups = ["sparse", "medium", "dense"]
        for pid in range(num_users):
            perm = data_rng.permutation(num_items)
            train_items = np.asarray(sorted(int(x) for x in perm[:8]), dtype=np.int64)
            test_item = int(perm[8])
            exclude = np.asarray(sorted(set(train_items.tolist()) | {test_item}), dtype=np.int32)
            users[pid] = UserEvalData(
                pid=pid,
                train_items=train_items,
                train_ratings=np.ones(8, dtype=np.float32),
                test_item=test_item,
                test_rating=1.0,
                exclude_items=exclude,
                user_group=groups[pid % 3],
            )
            user_stats[pid] = {
                "n_interactions": 8 + pid,
                "genre_entropy": 1.0,
                "n_unique_items": 8,
                "rating_std": 0.5,
                "user_group": groups[pid % 3],
            }
            # Head maximally aligned with the held-out positive (unit-norm
            # embeddings => positive strictly outranks every negative).
            payload = OrderedDict({"affine_output.weight": (100.0 * emb[test_item]).reshape(1, dim).clone()})
            torch.save(payload, cache_base / run_id / f"partition_{pid}.pt")
        signature = {
            "schema_version": 3,
            "run_id": run_id,
            "method": "pfedrec",
            "num_users": num_users,
            "num_items": num_items,
            "latent_dim": dim,
            "split_hash": "selftest-split-hash",
            "loss": "bce",
            "num_train_negatives": 2,
            "bias_classification": "global",
        }
        with open(cache_base / run_id / "manifest.json", "w") as f:
            json.dump(signature, f)

        run_data = RunData(
            num_users=num_users, num_items=num_items,
            split_hash="selftest-split-hash", users=users, user_stats=user_stats,
        )
        adapter = get_adapter("pfedrec", cfg, cache_base_dir=cache_base)
        adapter.inject_run_data(run_data)
        loaded_globals, _globals_path = _load_globals(run_dir, "best")
        stale_full = {pid: stale.get(pid, br) for pid in users}

        # --- cached + trust gate ---------------------------------------------
        cached_m, cached_rows = run_reeval(
            adapter, cfg, run_data, loaded_globals, "cached",
            device="cpu", staleness_map=stale_full,
        )
        assert cached_m["evaluated_users"] == num_users
        assert cached_m["sampled_ndcg@10"] > 0.99, "aligned heads must rank the positive first"
        best_block: Dict[str, Any] = {
            "sampled_hr@10": cached_m["sampled_hr@10"],
            "sampled_ndcg@10": cached_m["sampled_ndcg@10"],
            "evaluated_users": float(num_users),
            "d06_cache_misses": 0,
        }
        for grp in USER_GROUPS:
            best_block[f"sampled_hr@10/{grp}"] = cached_m[f"sampled_hr@10/{grp}"]
            best_block[f"sampled_ndcg@10/{grp}"] = cached_m[f"sampled_ndcg@10/{grp}"]
            best_block[f"evaluated_users/{grp}"] = float(cached_m[f"evaluated_users_{grp}"])
        verdict = trust_gate(
            cached_m, best_block, limit_users=None, globals_kind="best",
            eval_round_num=eval_round, final_eval_round_index=eval_round,
        )
        assert verdict["status"] == "PASS", f"trust gate should PASS: {verdict}"
        perturbed = dict(best_block)
        perturbed["sampled_ndcg@10"] = float(best_block["sampled_ndcg@10"]) + 1e-3
        verdict_bad = trust_gate(
            cached_m, perturbed, limit_users=None, globals_kind="best",
            eval_round_num=eval_round, final_eval_round_index=eval_round,
        )
        assert verdict_bad["status"] == "FAIL", "perturbed gate must FAIL"

        # --- per-user CSV ------------------------------------------------------
        csv_path = write_per_user_csv(tmp_path / "per_user.csv", cached_rows)
        with open(csv_path) as f:
            lines = f.read().strip().splitlines()
        assert lines[0] == ",".join(_PER_USER_FIELDS)
        assert len(lines) == num_users + 1

        # --- blend identities ---------------------------------------------------
        w_uniform, spec = compute_global_head(adapter, run_data, "uniform", run_seed)
        assert spec["n_heads"] == num_users
        blend1_m, _ = run_reeval(
            adapter, cfg, run_data, loaded_globals, "cached", device="cpu",
            alpha_for_user=lambda p: 1.0, w_global=w_uniform, label="alpha=1.00",
        )
        assert abs(blend1_m["sampled_ndcg@10"] - cached_m["sampled_ndcg@10"]) < 1e-12, (
            "alpha=1.0 blend must equal cached exactly"
        )
        blend0_m, _ = run_reeval(
            adapter, cfg, run_data, loaded_globals, "cached", device="cpu",
            alpha_for_user=lambda p: 0.0, w_global=w_uniform, label="alpha=0.00",
        )
        assert cached_m["sampled_ndcg@10"] > blend0_m["sampled_ndcg@10"], (
            "diluted population head must underperform aligned per-user heads"
        )

        # --- grid + oracles -----------------------------------------------------
        matrix: Dict[float, Dict[int, Dict[str, Any]]] = {}
        grid_best = -1.0
        for a in ALPHA_GRID:
            m, rows = run_reeval(
                adapter, cfg, run_data, loaded_globals, "cached", device="cpu",
                alpha_for_user=lambda p, _a=a: _a, w_global=w_uniform, label=f"a={a:.2f}",
            )
            assert math.isfinite(m["sampled_ndcg@10"])
            matrix[a] = {r["pid"]: r for r in rows}
            grid_best = max(grid_best, m["sampled_ndcg@10"])
        og = oracle_group(matrix, run_data, stale_full)
        ou_m, _assign, hist = oracle_user(matrix, run_data)
        best_fam = og["best_family"]
        in_sample_best = max(
            fam["in_sample"]["sampled_ndcg@10"] for fam in og["families"].values()
        )
        assert ou_m["sampled_ndcg@10"] >= grid_best - 1e-9, "user oracle >= best single alpha"
        assert in_sample_best >= grid_best - 1e-9, "in-sample group oracle >= best single alpha"
        assert sum(hist.values()) == num_users
        assert best_fam in og["families"]

        # --- HC policy alphas within clip bounds ---------------------------------
        from federated_adaptive_personalized_cf.models.adaptive_alpha import (
            AlphaConfig,
            HierarchicalConditionalAlphaConfig,
            create_alpha_computer,
        )
        computer = create_alpha_computer(
            AlphaConfig(method="hierarchical_conditional"),
            hc_config=HierarchicalConditionalAlphaConfig(),
        )
        for pid in range(num_users):
            a = computer.compute_from_stats(user_stats[pid])
            assert 0.1 - 1e-9 <= a <= 0.95 + 1e-9, f"HC alpha {a} out of clip bounds"

        # --- zero variant hard-rejected -------------------------------------------
        try:
            adapter.zero_local()
            raise AssertionError("zero_local must raise for pfedrec")
        except RuntimeError:
            pass

        # --- random head: Kaiming bound + determinism -----------------------------
        r1 = adapter.random_local(3, run_seed)["affine_output.weight"]
        r2 = adapter.random_local(3, run_seed)["affine_output.weight"]
        assert torch.equal(r1, r2), "seeded random head must be deterministic"
        assert r1.abs().max().item() <= 1.0 / math.sqrt(dim) + 1e-9

        print("[self-test:pfedrec] PASS — adapter, trust gate, blends, grid, oracles, policy.")


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
        "--module", choices=("personalized", "adaptive", "pfedrec"), default="personalized",
        help="federated module the run belongs to",
    )
    parser.add_argument(
        "--local-variant", nargs="+",
        choices=sorted(set(VARIANTS) | set(PFEDREC_VARIANTS)), default=["cached"],
        help=(
            "local-state variant(s) to evaluate; the alpha/global_head suite is "
            "pfedrec-only, 'zero' is personalized-only (degenerate for pfedrec)"
        ),
    )
    parser.add_argument(
        "--no-per-user", action="store_true",
        help="skip writing per-user CSVs (pfedrec suite writes them by default)",
    )
    parser.add_argument(
        "--skip-trust-gate", action="store_true",
        help="do NOT compare the cached variant against the run's D-06 best block (escape hatch)",
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

    cfg = resolve_config(run_dir, calib_epochs=args.calib_epochs, module=args.module)
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

    if args.module == "pfedrec":
        return run_pfedrec_suite(args, cfg, adapter, run_data, global_state, globals_path)

    pfedrec_only = set(PFEDREC_VARIANTS) - set(VARIANTS)
    bad = [v for v in args.local_variant if v in pfedrec_only]
    if bad:
        print(f"error: variant(s) {bad} require --module pfedrec", file=sys.stderr)
        return 2

    summary: Dict[str, Dict[str, float]] = {}
    for variant in args.local_variant:
        metrics, _rows = run_reeval(adapter, cfg, run_data, global_state, variant, device=args.device)
        print_table(variant, args.globals, cfg, metrics)
        if not args.no_json:
            try:
                cache_dir: Optional[Path] = adapter.resolve_cache_dir(run_data)
            except NotImplementedError:
                cache_dir = None
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
