"""Tests for scripts/thesis/gate_stats.py (Q_u gate statistics, README §6–§7.1)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

pytest.importorskip("scipy")

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (
    str(_REPO_ROOT / "scripts"),
    str(_REPO_ROOT / "scripts" / "foundation"),
    str(_REPO_ROOT / "federated-pfedrec"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from thesis import gate_stats  # noqa: E402
from thesis.gate_stats import (  # noqa: E402
    GateStatsConfig,
    compute_gate_stats,
    holm_adjust,
    select_winner,
    signed_rank_sums,
)

NDCG_LEVELS = [0.0] + [1.0 / np.log2(r + 1) for r in range(1, 11)]
GROUPS = ("sparse", "medium", "dense")


def _write_baseline(path: Path, ndcg: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "user_group", "n_interactions", "staleness_at_best", "alpha_used",
                    "sampled_ndcg@10", "sampled_hr@10", "eval_loss", "evaluated"])
        for pid, v in enumerate(ndcg):
            w.writerow([pid, GROUPS[pid % 3], 50, 3, 1.0, repr(float(v)), 1.0 if v > 0 else 0.0, 0.1, 1])


def _write_grid(path: Path, cells: Dict[str, np.ndarray]) -> None:
    fields = ["pid", "user_group", "n_interactions", "staleness_at_best", "alpha_table", "epochs", "pool",
              "n_touched_rows", "n_touched_candidates", "test_row_touched", "mean_row_shift",
              "sampled_ndcg@10", "sampled_hr@10", "sampled_mrr", "eval_loss"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for key, ndcg in cells.items():
            a, e = float(key[1:5]), int(key.split("_E")[1])
            for pid, v in enumerate(ndcg):
                w.writerow([pid, GROUPS[pid % 3], 50, 3, a, e, "clean", 10, 1, 0, 0.1,
                            repr(float(v)), 1.0 if v > 0 else 0.0, 0.2, 0.1])


def _planted(n: int = 600, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(NDCG_LEVELS, size=n, p=[0.4] + [0.06] * 10)


def test_holm_matches_hand_computation():
    adj = holm_adjust([0.01, 0.04, 0.03, None])
    assert adj[3] is None
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)


def test_signed_rank_sums_and_zero_dropping():
    w_plus, w_minus, n = signed_rank_sums(np.array([0.0, 1.0, -2.0, 3.0, 0.0]))
    assert n == 3 and w_plus == 1 + 3 and w_minus == 2


def test_winner_rule_prefers_larger_alpha_then_smaller_epochs_on_ties():
    cands = [
        {"cell": "a0.30_E1", "ndcg": 0.3600, "alpha_table": 0.3, "epochs": 1},
        {"cell": "a0.70_E3", "ndcg": 0.3595, "alpha_table": 0.7, "epochs": 3},
        {"cell": "a0.70_E1", "ndcg": 0.3592, "alpha_table": 0.7, "epochs": 1},
        {"cell": "a0.50_E1", "ndcg": 0.3500, "alpha_table": 0.5, "epochs": 1},
    ]
    assert select_winner(cands) == "a0.70_E1"
    assert select_winner([]) is None


def _run(tmp_path: Path, base: np.ndarray, cells: Dict[str, np.ndarray]) -> Dict:
    _write_baseline(tmp_path / "base.csv", base)
    _write_grid(tmp_path / "grid.csv", cells)
    cfg = GateStatsConfig(tmp_path / "grid.csv", tmp_path / "base.csv", tmp_path / "out.json")
    return compute_gate_stats(cfg)


def test_go_when_a_cell_clears_both_conditions(tmp_path):
    base = _planted()
    better = base.copy()
    # Promote 25% of users one NDCG level -> clear +0.010 and a tiny p.
    rng = np.random.default_rng(1)
    for i in rng.choice(len(base), size=len(base) // 4, replace=False):
        lvl = NDCG_LEVELS.index(base[i]) if base[i] in NDCG_LEVELS else 0
        better[i] = 1.0 if lvl in (0, 1) else NDCG_LEVELS[lvl - 1]
    shift = 0.3547 - base.mean()
    assert better.mean() - base.mean() > max(shift, 0.0)
    payload = _run(tmp_path, base, {"a1.00_E1": base, "a0.30_E1": better, "a0.70_E1": base})
    assert better.mean() >= 0.3547  # planted cell clears the absolute threshold
    assert payload["verdict"]["decision"] == "GO"
    assert payload["verdict"]["winner"] == "a0.30_E1"
    assert payload["holm_flips"] is False
    o = payload["cells"]["a0.30_E1"]["by_scope"]["overall"]
    assert o["p_one_sided"] < 1e-6 and o["rank_biserial_r"] == 1.0
    lo, hi = o["ci95_ndcg"]
    assert lo < o["mean_delta_ndcg"] < hi
    assert payload["reproduction_ok"] is True
    assert payload["cells"]["a0.70_E1"]["by_scope"]["overall"]["p_one_sided"] is None


def test_no_go_when_cell_is_worse_even_if_two_sided_significant(tmp_path):
    base = _planted(seed=3)
    base = np.where(base == 0.0, 1.0, base)  # high baseline so condition 1 holds in level terms
    worse = np.where(np.arange(len(base)) % 2 == 0, 0.0, base)
    payload = _run(tmp_path, base, {"a1.00_E1": base, "a0.30_E1": worse})
    o = payload["cells"]["a0.30_E1"]["by_scope"]["overall"]
    assert o["p_two_sided"] < 0.05 and o["p_one_sided"] > 0.5
    assert payload["verdict"]["decision"] == "NO-GO"


def test_reproduction_cell_mismatch_is_flagged(tmp_path):
    base = _planted(seed=4)
    off = base.copy()
    off[0] = 1.0 if base[0] != 1.0 else 0.0
    payload = _run(tmp_path, base, {"a1.00_E1": off})
    assert payload["reproduction_ok"] is False


def _write_diag_grid(path: Path, cells: Dict[str, np.ndarray]) -> None:
    """Long CSV for a qu_diag-style grid: keys like ``a0.00_E1_leaky``."""
    fields = ["pid", "user_group", "n_interactions", "staleness_at_best", "alpha_table", "epochs", "pool",
              "n_touched_rows", "n_touched_candidates", "test_row_touched", "mean_row_shift",
              "sampled_ndcg@10", "sampled_hr@10", "sampled_mrr", "eval_loss"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for key, ndcg in cells.items():
            a, rest = float(key[1:5]), key.split("_E")[1]
            e, pool = int(rest.split("_")[0]), rest.split("_")[1]
            for pid, v in enumerate(ndcg):
                w.writerow([pid, GROUPS[pid % 3], 50, 3, a, e, pool, 10, 1, 0, 0.1,
                            repr(float(v)), 1.0 if v > 0 else 0.0, 0.2, 0.1])


def test_contrast_leaky_vs_clean(tmp_path):
    base = _planted(seed=7)
    clean = base.copy()
    leaky = base.copy()
    rng = np.random.default_rng(8)
    for i in rng.choice(len(base), size=len(base) // 3, replace=False):
        lvl = NDCG_LEVELS.index(base[i]) if base[i] in NDCG_LEVELS else 0
        leaky[i] = 1.0 if lvl in (0, 1) else NDCG_LEVELS[lvl - 1]
    _write_baseline(tmp_path / "base.csv", base)
    _write_diag_grid(tmp_path / "grid.csv", {"a0.00_E1_clean": clean, "a0.00_E1_leaky": leaky})
    cfg = GateStatsConfig(
        tmp_path / "grid.csv", tmp_path / "base.csv", tmp_path / "out.json",
        contrasts=[("a0.00_E1_leaky", "a0.00_E1_clean")],
    )
    payload = compute_gate_stats(cfg)
    assert set(payload["cells"]) == {"a0.00_E1_clean", "a0.00_E1_leaky"}
    c = payload["contrasts"]["a0.00_E1_leaky__minus__a0.00_E1_clean"]
    o = c["by_scope"]["overall"]
    assert c["mean_ndcg_a"] > c["mean_ndcg_b"]
    assert o["mean_delta_ndcg"] == pytest.approx(leaky.mean() - clean.mean())
    assert o["p_one_sided"] < 1e-6 and o["rank_biserial_r"] == 1.0
    assert set(c["by_scope"]) == {"overall", "sparse", "medium", "dense"}
    # Reverse direction is not significant one-sided.
    cfg_rev = GateStatsConfig(
        tmp_path / "grid.csv", tmp_path / "base.csv", tmp_path / "out2.json",
        contrasts=[("a0.00_E1_clean", "a0.00_E1_leaky")],
    )
    rev = compute_gate_stats(cfg_rev)["contrasts"]["a0.00_E1_clean__minus__a0.00_E1_leaky"]
    assert rev["by_scope"]["overall"]["p_one_sided"] > 0.5
    # Unknown cell -> KeyError; identical cells -> ValueError at config time.
    with pytest.raises(KeyError):
        compute_gate_stats(GateStatsConfig(tmp_path / "grid.csv", tmp_path / "base.csv", tmp_path / "o.json",
                                           contrasts=[("a0.00_E3_leaky", "a0.00_E1_clean")]))
    with pytest.raises(ValueError):
        GateStatsConfig(tmp_path / "grid.csv", tmp_path / "base.csv", tmp_path / "o.json",
                        contrasts=[("a0.00_E1_clean", "a0.00_E1_clean")])


def test_contrast_cli_flag(tmp_path):
    base = _planted(seed=9)
    _write_baseline(tmp_path / "base.csv", base)
    _write_diag_grid(tmp_path / "grid.csv", {"a0.00_E1_clean": base, "a0.00_E1_leaky": base})
    out = tmp_path / "stats.json"
    rc = gate_stats.main([
        "--grid", str(tmp_path / "grid.csv"), "--baseline", str(tmp_path / "base.csv"), "--out", str(out),
        "--contrast", "a0.00_E1_leaky", "a0.00_E1_clean",
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    c = payload["contrasts"]["a0.00_E1_leaky__minus__a0.00_E1_clean"]
    assert c["by_scope"]["overall"]["n_nonzero"] == 0 and c["by_scope"]["overall"]["p_one_sided"] is None


def test_end_to_end_with_qu_grid_output(tmp_path):
    """Pipe the synthetic qu_grid run (test_reeval_qu fixture) through gate_stats."""
    import argparse

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_reeval_qu import EVAL_ROUND, NUM_USERS, _build_run  # noqa: E402
    from thesis import reeval  # noqa: E402

    cfg_r, adapter, run_data, global_state, run_dir = _build_run(tmp_path)
    base_m, base_rows = reeval.run_reeval(adapter, cfg_r, run_data, global_state, "cached")
    base_csv = reeval.write_per_user_csv(tmp_path / "base.csv", base_rows)
    best_block = {"sampled_hr@10": base_m["sampled_hr@10"], "sampled_ndcg@10": base_m["sampled_ndcg@10"],
                  "evaluated_users": float(NUM_USERS), "d06_cache_misses": 0}
    for g in GROUPS:
        best_block[f"sampled_hr@10/{g}"] = base_m[f"sampled_hr@10/{g}"]
        best_block[f"sampled_ndcg@10/{g}"] = base_m[f"sampled_ndcg@10/{g}"]
        best_block[f"evaluated_users/{g}"] = float(base_m[f"evaluated_users_{g}"])
    out_dir = tmp_path / "gate_qu"
    args = argparse.Namespace(out_dir=out_dir, device="cpu", skip_trust_gate=False, limit_users=None,
                              globals="best", no_json=False, no_per_user=False)
    reeval._run_qu_suite_variant("qu_grid", args, cfg_r, adapter, run_data, global_state,
                                 run_dir / "global_state_best.pt", best_block=best_block,
                                 final_eval_round_index=EVAL_ROUND, staleness_map=None)
    grid_csv = out_dir / "reeval_qu_grid_best_per_user.csv"
    grid_json = out_dir / "reeval_qu_grid_best.json"
    cfg = GateStatsConfig(grid_csv, base_csv, tmp_path / "stats.json", grid_json)
    payload = compute_gate_stats(cfg)
    assert payload["reproduction_ok"] is True
    assert set(payload["cells"]) == set(json.loads(grid_json.read_text())["metrics_by_cell"])
    for c in payload["cells"].values():
        assert c["ndcg_source"] == "grid_json.metrics_by_cell"
    gate_stats.write_json_atomic(cfg.out_path, payload)
    assert json.loads(cfg.out_path.read_text())["verdict"]["decision"] in ("GO", "NO-GO")
