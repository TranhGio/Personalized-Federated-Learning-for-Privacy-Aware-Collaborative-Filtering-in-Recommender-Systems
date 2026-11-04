#!/usr/bin/env python
"""Per-user paired statistics + GO/NO-GO verdict for the Q_u gate (Việc 2).

Implements README §6–§7.1 of
``results/federated/pfedrec/20260726-083121-cf72e0/gate_qu/README.md`` (v1.1).
Needs only the two committed CSVs (no cache, no ``.pt``):

- ``--grid``     long per-user CSV from ``reeval.py --local-variant qu_grid``
- ``--baseline`` ``reeval_cached_best_per_user.csv`` of the same run
- ``--grid-json`` (recommended) ``reeval_qu_grid_best.json`` — the binding
  NDCG@10 of each cell is read from its ``metrics_by_cell`` (sufficient-stat
  aggregation, README §5), not recomputed from the CSV.

For every cell with α < 1 (and per user group): d_u = cell − baseline,
Wilcoxon signed-rank one-sided (H1: cell > baseline, ``zero_method="wilcox"``,
``method="approx"``) — the binding test — plus the two-sided p, the Pratt
one-sided p, Holm over the α < 1 cells, rank-biserial r, mean Δ with a paired
bootstrap 95% CI (10 000 resamples, seed 42). The test measures variation
BETWEEN USERS, not across training seeds.

``--contrast A B`` (repeatable) adds a paired cell-vs-cell contrast A − B on
the same users (H1: A > B), used by the PFedRec post-tuning re-score
(``<run>/posttune/README.md`` §6, leaky vs clean pool). The GO/NO-GO
``verdict`` is still emitted but is only meaningful for the gate.

Usage
-----
    python scripts/thesis/gate_stats.py \\
        --grid <gate_qu>/reeval_qu_grid_best_per_user.csv \\
        --baseline <run>/reeval_cached_best_per_user.csv \\
        --grid-json <gate_qu>/reeval_qu_grid_best.json \\
        --out <gate_qu>/stats_wilcoxon.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

USER_GROUPS = ("sparse", "medium", "dense")
# Pre-registered decision constants (README §7.1) — NOT CLI knobs.
GO_NDCG_THRESHOLD = 0.3547
P_THRESHOLD = 0.05
WINNER_TIE_MARGIN = 0.001
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CHUNK = 500


@dataclass
class GateStatsConfig:
    """Inputs of one gate-statistics pass.

    Attributes
    ----------
    grid_csv : Path
        Long per-user CSV (15 columns) of the Q_u variant.
    baseline_csv : Path
        Cached per-user CSV of the source run (the 0.3447 reference).
    out_path : Path
        Destination ``stats_wilcoxon.json``.
    grid_json : Optional[Path]
        ``reeval_qu_*.json`` with ``metrics_by_cell`` (binding NDCG means).
    contrasts : List[Tuple[str, str]]
        Paired cell-vs-cell contrasts ``(A, B)`` tested as ``A > B`` on the
        same users (posttune README §6, e.g. ``("a0.00_E1_leaky",
        "a0.00_E1_clean")``). Cell names are checked against the CSV when
        the stats are computed. Empty by default.
    """

    grid_csv: Path
    baseline_csv: Path
    out_path: Path
    grid_json: Optional[Path] = None
    contrasts: List[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        norm: List[Tuple[str, str]] = []
        for pair in self.contrasts:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError(f"contrast must be two distinct cell names, got {pair!r}")
            norm.append((str(pair[0]), str(pair[1])))
        self.contrasts = norm
        for name in ("grid_csv", "baseline_csv"):
            p = Path(getattr(self, name))
            if not p.is_file():
                raise FileNotFoundError(f"{name}: {p} does not exist")
            setattr(self, name, p)
        if self.grid_json is not None:
            self.grid_json = Path(self.grid_json)
            if not self.grid_json.is_file():
                raise FileNotFoundError(f"grid_json: {self.grid_json} does not exist")
        self.out_path = Path(self.out_path)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def cell_key(alpha: float, epochs: int, pool: str, suffix_pool: bool) -> str:
    """Same cell id as ``reeval.qu_cell_key`` (kept local: no torch import)."""
    key = f"a{float(alpha):.2f}_E{int(epochs)}"
    return f"{key}_{pool}" if suffix_pool else key


def load_baseline(path: Path) -> Dict[int, Dict[str, Any]]:
    """``pid -> {ndcg, hr, group}`` from a cached per-user CSV."""
    out: Dict[int, Dict[str, Any]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["pid"])] = {
                "ndcg": float(row["sampled_ndcg@10"]),
                "hr": float(row["sampled_hr@10"]),
                "group": row["user_group"],
            }
    return out


def load_grid(path: Path) -> Tuple[Dict[str, Dict[int, Dict[str, float]]], Dict[str, Dict[str, Any]]]:
    """Group the long CSV by cell.

    Returns
    -------
    Tuple[Dict[str, Dict[int, Dict[str, float]]], Dict[str, Dict[str, Any]]]
        ``cell -> pid -> {ndcg, hr}`` and ``cell -> {alpha, epochs, pool}``.
    """
    raw: List[Dict[str, str]] = []
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    pools = {r["pool"] for r in raw}
    suffix = len(pools) > 1
    cells: Dict[str, Dict[int, Dict[str, float]]] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for r in raw:
        a, e, pool = float(r["alpha_table"]), int(r["epochs"]), r["pool"]
        key = cell_key(a, e, pool, suffix)
        meta.setdefault(key, {"alpha_table": a, "epochs": e, "pool": pool})
        pid = int(r["pid"])
        if pid in cells.setdefault(key, {}):
            raise ValueError(f"duplicate pid {pid} in cell {key}")
        cells[key][pid] = {"ndcg": float(r["sampled_ndcg@10"]), "hr": float(r["sampled_hr@10"])}
    return cells, meta


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def signed_rank_sums(d: np.ndarray) -> Tuple[float, float, int]:
    """W+ and W- over non-zero differences (average ranks of |d|), and n_nonzero."""
    from scipy.stats import rankdata

    nz = d[d != 0]
    if nz.size == 0:
        return 0.0, 0.0, 0
    ranks = rankdata(np.abs(nz))
    return float(ranks[nz > 0].sum()), float(ranks[nz < 0].sum()), int(nz.size)


def wilcoxon_block(d: np.ndarray) -> Dict[str, Any]:
    """README §6 Wilcoxon bundle for one difference vector."""
    from scipy.stats import wilcoxon

    w_plus, w_minus, n_nonzero = signed_rank_sums(d)
    out: Dict[str, Any] = {
        "n": int(d.size),
        "n_nonzero": n_nonzero,
        "sd_d": float(np.std(d, ddof=1)) if d.size > 1 else None,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "rank_biserial_r": (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) > 0 else None,
        "p_one_sided": None,
        "p_two_sided": None,
        "p_pratt_one_sided": None,
    }
    if n_nonzero == 0:
        return out
    out["p_one_sided"] = float(
        wilcoxon(d, alternative="greater", zero_method="wilcox", method="approx").pvalue
    )
    out["p_two_sided"] = float(
        wilcoxon(d, alternative="two-sided", zero_method="wilcox", method="approx").pvalue
    )
    out["p_pratt_one_sided"] = float(
        wilcoxon(d, alternative="greater", zero_method="pratt", method="approx").pvalue
    )
    return out


def paired_bootstrap_ci(
    d_ndcg: np.ndarray,
    d_hr: np.ndarray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Mean Δ and 95% percentile CI, same user resamples for NDCG and HR.

    The RNG is created INSIDE the function (project seeding rule) so every
    call is reproducible regardless of the caller.
    """
    n = int(d_ndcg.size)
    out: Dict[str, Any] = {"mean_delta_ndcg": float(d_ndcg.mean()), "mean_delta_hr": float(d_hr.mean())}
    if n == 0:
        out.update({"ci95_ndcg": None, "ci95_hr": None})
        return out
    rng = np.random.default_rng(seed)
    means_n = np.empty(resamples)
    means_h = np.empty(resamples)
    done = 0
    while done < resamples:
        k = min(BOOTSTRAP_CHUNK, resamples - done)
        idx = rng.integers(0, n, size=(k, n))
        means_n[done:done + k] = d_ndcg[idx].mean(axis=1)
        means_h[done:done + k] = d_hr[idx].mean(axis=1)
        done += k
    out["ci95_ndcg"] = [float(np.percentile(means_n, 2.5)), float(np.percentile(means_n, 97.5))]
    out["ci95_hr"] = [float(np.percentile(means_h, 2.5)), float(np.percentile(means_h, 97.5))]
    return out


def holm_adjust(pvalues: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Holm step-down adjusted p-values (None entries are skipped)."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    m = len(idx)
    adjusted: List[Optional[float]] = [None] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: pvalues[j])):
        val = min(1.0, (m - rank) * float(pvalues[i]))
        running = max(running, val)
        adjusted[i] = running
    return adjusted


def select_winner(candidates: List[Dict[str, Any]]) -> Optional[str]:
    """README §7.1 point 3: highest NDCG; ties (< 0.001) -> larger α, then smaller E."""
    if not candidates:
        return None
    best = max(c["ndcg"] for c in candidates)
    tied = [c for c in candidates if best - c["ndcg"] < WINNER_TIE_MARGIN]
    tied.sort(key=lambda c: (-c["alpha_table"], c["epochs"]))
    return tied[0]["cell"]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def contrast_cells(
    cells: Dict[str, Dict[int, Dict[str, float]]],
    baseline: Dict[int, Dict[str, Any]],
    cell_a: str,
    cell_b: str,
) -> Dict[str, Any]:
    """Paired contrast ``cell_a − cell_b`` per user (H1: A > B), overall and per group.

    Parameters
    ----------
    cells : Dict[str, Dict[int, Dict[str, float]]]
        ``load_grid`` output (``cell -> pid -> {ndcg, hr}``).
    baseline : Dict[int, Dict[str, Any]]
        ``load_baseline`` output; only its ``group`` field is used.
    cell_a, cell_b : str
        Cell names as they appear in the CSV-derived keys.

    Returns
    -------
    Dict[str, Any]
        ``{"cell_a", "cell_b", "mean_ndcg_a", "mean_ndcg_b", "by_scope"}`` where each
        scope holds the Wilcoxon bundle and bootstrap CI of ``wilcoxon_block`` /
        ``paired_bootstrap_ci``.

    Raises
    ------
    KeyError
        If a cell is absent from the CSV.
    ValueError
        If the two cells do not cover the same users.
    """
    for name in (cell_a, cell_b):
        if name not in cells:
            raise KeyError(f"contrast cell {name!r} not in grid CSV; have {sorted(cells)}")
    pids_a, pids_b = set(cells[cell_a]), set(cells[cell_b])
    if pids_a != pids_b:
        raise ValueError(
            f"contrast {cell_a} vs {cell_b}: user sets differ ({len(pids_a ^ pids_b)} pid(s) not shared)"
        )
    pids = sorted(pids_a)
    groups = np.asarray([baseline[p]["group"] for p in pids])
    d_ndcg = np.asarray([cells[cell_a][p]["ndcg"] - cells[cell_b][p]["ndcg"] for p in pids])
    d_hr = np.asarray([cells[cell_a][p]["hr"] - cells[cell_b][p]["hr"] for p in pids])
    out: Dict[str, Any] = {
        "cell_a": cell_a,
        "cell_b": cell_b,
        "hypothesis": "one-sided, A > B, zero_method=wilcox, approx",
        "mean_ndcg_a": float(np.mean([cells[cell_a][p]["ndcg"] for p in pids])),
        "mean_ndcg_b": float(np.mean([cells[cell_b][p]["ndcg"] for p in pids])),
        "by_scope": {},
    }
    scopes: Dict[str, np.ndarray] = {"overall": np.ones(len(pids), dtype=bool)}
    scopes.update({g: groups == g for g in USER_GROUPS})
    for scope, mask in scopes.items():
        block: Dict[str, Any] = {"n": int(mask.sum())}
        block.update(wilcoxon_block(d_ndcg[mask]))
        block.update(paired_bootstrap_ci(d_ndcg[mask], d_hr[mask]))
        out["by_scope"][scope] = block
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_gate_stats(cfg: GateStatsConfig) -> Dict[str, Any]:
    """Build the full ``stats_wilcoxon.json`` payload (see module docstring)."""
    import scipy

    baseline = load_baseline(cfg.baseline_csv)
    cells, meta = load_grid(cfg.grid_csv)
    binding: Dict[str, Dict[str, float]] = {}
    if cfg.grid_json is not None:
        binding = json.loads(cfg.grid_json.read_text()).get("metrics_by_cell") or {}

    per_cell: Dict[str, Any] = {}
    for key in sorted(cells):
        rows = cells[key]
        missing = sorted(set(rows) - set(baseline))
        if missing:
            raise ValueError(f"cell {key}: {len(missing)} pid(s) absent from the baseline CSV")
        pids = sorted(rows)
        groups = np.asarray([baseline[p]["group"] for p in pids])
        d_ndcg = np.asarray([rows[p]["ndcg"] - baseline[p]["ndcg"] for p in pids])
        d_hr = np.asarray([rows[p]["hr"] - baseline[p]["hr"] for p in pids])
        ndcg_csv = float(np.mean([rows[p]["ndcg"] for p in pids]))
        entry: Dict[str, Any] = dict(meta[key])
        entry["ndcg_binding"] = float(binding[key]["sampled_ndcg@10"]) if key in binding else ndcg_csv
        entry["ndcg_source"] = "grid_json.metrics_by_cell" if key in binding else "csv_mean"
        entry["hr"] = float(binding[key]["sampled_hr@10"]) if key in binding else float(np.mean([rows[p]["hr"] for p in pids]))
        tested = meta[key]["alpha_table"] < 1.0
        entry["tested"] = tested
        scopes: Dict[str, np.ndarray] = {"overall": np.ones(len(pids), dtype=bool)}
        scopes.update({g: groups == g for g in USER_GROUPS})
        entry["by_scope"] = {}
        for scope, mask in scopes.items():
            block: Dict[str, Any] = {"n": int(mask.sum())}
            if tested:
                block.update(wilcoxon_block(d_ndcg[mask]))
                block.update(paired_bootstrap_ci(d_ndcg[mask], d_hr[mask]))
            else:
                block["n_nonzero"] = int(np.count_nonzero(d_ndcg[mask]) + np.count_nonzero(d_hr[mask]))
            entry["by_scope"][scope] = block
        per_cell[key] = entry

    tested_keys = [k for k in sorted(per_cell) if per_cell[k]["tested"]]
    holm = holm_adjust([per_cell[k]["by_scope"]["overall"]["p_one_sided"] for k in tested_keys])
    for k, ph in zip(tested_keys, holm):
        per_cell[k]["by_scope"]["overall"]["p_holm_one_sided"] = ph

    def _passes(k: str, p_field: str) -> bool:
        p = per_cell[k]["by_scope"]["overall"].get(p_field)
        return per_cell[k]["ndcg_binding"] >= GO_NDCG_THRESHOLD and p is not None and p < P_THRESHOLD

    def _verdict(p_field: str) -> Dict[str, Any]:
        passing = [k for k in tested_keys if _passes(k, p_field)]
        winner = select_winner([
            {"cell": k, "ndcg": per_cell[k]["ndcg_binding"], "alpha_table": per_cell[k]["alpha_table"],
             "epochs": per_cell[k]["epochs"]} for k in passing
        ])
        return {"decision": "GO" if passing else "NO-GO", "passing_cells": passing, "winner": winner}

    raw = _verdict("p_one_sided")
    holm_v = _verdict("p_holm_one_sided")
    reproduction = {k: per_cell[k]["by_scope"]["overall"]["n_nonzero"] for k in per_cell if not per_cell[k]["tested"]}
    contrasts = {
        f"{a}__minus__{b}": contrast_cells(cells, baseline, a, b) for a, b in cfg.contrasts
    }
    return {
        "tool": "scripts/thesis/gate_stats.py",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv,
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "inputs_sha256": {
            "grid_csv": _sha256(cfg.grid_csv),
            "baseline_csv": _sha256(cfg.baseline_csv),
            "grid_json": _sha256(cfg.grid_json) if cfg.grid_json is not None else None,
        },
        "rule": {
            "condition_1": f"overall NDCG@10 (ndcg_binding) >= {GO_NDCG_THRESHOLD}",
            "condition_2": f"Wilcoxon one-sided (cell > baseline, zero_method=wilcox, approx) raw p < {P_THRESHOLD}",
            "winner": f"highest NDCG; ties < {WINNER_TIE_MARGIN} -> larger alpha, then smaller E",
            "note": "tests measure between-user variation, not seed variation",
        },
        "cells": per_cell,
        "contrasts": contrasts,
        "verdict": raw,
        "verdict_if_holm": holm_v,
        "holm_flips": raw["decision"] != holm_v["decision"] or raw["winner"] != holm_v["winner"],
        "reproduction_cells_nonzero": reproduction,
        "reproduction_ok": all(v == 0 for v in reproduction.values()),
    }


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Atomic JSON write (tempfile in the same dir + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="gate_stats_tmp_", suffix=".json", dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, str(path))
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point; prints a compact table and writes ``--out``."""
    parser = argparse.ArgumentParser(prog="gate_stats.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid", type=Path, required=True, help="long per-user CSV of qu_grid/qu_diag")
    parser.add_argument("--baseline", type=Path, required=True, help="reeval_cached_best_per_user.csv")
    parser.add_argument("--grid-json", type=Path, default=None, help="reeval_qu_grid_best.json (binding NDCG)")
    parser.add_argument("--out", type=Path, required=True, help="destination stats_wilcoxon.json")
    parser.add_argument(
        "--contrast", nargs=2, action="append", default=[], metavar=("CELL_A", "CELL_B"),
        help="paired contrast A > B on the same users (repeatable), e.g. a0.00_E1_leaky a0.00_E1_clean",
    )
    args = parser.parse_args(argv)
    cfg = GateStatsConfig(args.grid, args.baseline, args.out, args.grid_json, contrasts=args.contrast)
    payload = compute_gate_stats(cfg)
    write_json_atomic(cfg.out_path, payload)

    print(f"{'cell':<18}{'ndcg':>10}{'Δndcg':>10}{'n_nz':>7}{'p1':>11}{'p2':>11}{'pHolm':>11}{'r_rb':>8}")
    for k, c in payload["cells"].items():
        o = c["by_scope"]["overall"]
        if not c["tested"]:
            print(f"{k:<18}{c['ndcg_binding']:>10.5f}{'—':>10}{o['n_nonzero']:>7}   (reproduction cell)")
            continue
        fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else "—"  # noqa: E731
        print(
            f"{k:<18}{c['ndcg_binding']:>10.5f}{o['mean_delta_ndcg']:>+10.5f}{o['n_nonzero']:>7}"
            f"{fmt(o['p_one_sided']):>11}{fmt(o['p_two_sided']):>11}{fmt(o.get('p_holm_one_sided')):>11}"
            f"{fmt(o['rank_biserial_r']):>8}"
        )
    for name, c in payload["contrasts"].items():
        o = c["by_scope"]["overall"]
        p1 = f"{o['p_one_sided']:.3g}" if isinstance(o["p_one_sided"], float) else "—"
        print(
            f"contrast {c['cell_a']} − {c['cell_b']}: Δndcg {o['mean_delta_ndcg']:+.5f} "
            f"CI95 [{o['ci95_ndcg'][0]:+.5f}, {o['ci95_ndcg'][1]:+.5f}] p1 {p1} n_nz {o['n_nonzero']}"
        )
    v = payload["verdict"]
    print(f"\nverdict: {v['decision']} | passing={v['passing_cells']} | winner={v['winner']} "
          f"| holm_flips={payload['holm_flips']} | reproduction_ok={payload['reproduction_ok']}")
    print(f"wrote {cfg.out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
