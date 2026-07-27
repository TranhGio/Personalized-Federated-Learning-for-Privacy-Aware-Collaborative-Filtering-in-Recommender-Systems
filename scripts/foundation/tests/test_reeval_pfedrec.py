"""Tests for the pfedrec adapter + alpha suite in scripts/thesis/reeval.py (Stage A).

Complements the tool's built-in ``--self-test`` (synthetic end-to-end) with
unit-level pins: config resolution hard-failures, signature/cache-dir parity
with the live client_app, trust-gate field mapping, blend math, oracle
cross-fitting on a planted matrix, and the global-head builders.
"""
from __future__ import annotations

import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

# Bootstrap import path for the reeval tool + the pfedrec package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (
    str(_REPO_ROOT / "scripts"),
    str(_REPO_ROOT / "scripts" / "foundation"),
    str(_REPO_ROOT / "federated-pfedrec"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from thesis import reeval  # noqa: E402
from thesis.reeval import (  # noqa: E402
    ALPHA_GRID,
    MIN_BUCKET_USERS,
    PfedrecAdapter,
    ReevalConfig,
    RunData,
    UserEvalData,
    compute_global_head,
    oracle_group,
    oracle_user,
    resolve_config,
    stage_a_verdict,
    trust_gate,
    write_per_user_csv,
)

USER_GROUPS = ("sparse", "medium", "dense")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _pfedrec_results_payload(**overrides: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "_manifest": {
            "run_id": "unittest-run",
            "mode": "paper_compat_pfedrec",
            "module": "pfedrec",
            "run_seed": 42,
            "split_hash": "unittest-split",
            "final_eval_round_index": 101,
            "num_eval_negatives": 99,
            "num_train_negatives": 4,
            "fraction_train": 0.1,
            "git_commit": "unittest",
        },
        "federated_config": {
            "latent_dim": 32,
            "lr": 0.1,
            "lr_eta": 80.0,
            "loss": "bce",
            "l2_regularization": 0.0,
            "reuse_cache": False,
        },
        "final_metrics": {"final_eval_round_index": 101},
        "training_rounds": 100,
    }
    payload.update(overrides)
    return payload


@pytest.fixture()
def pfedrec_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "results.json", "w") as f:
        json.dump(_pfedrec_results_payload(), f)
    return run_dir


def _mk_cfg(run_dir: Path, dim: int = 32) -> ReevalConfig:
    return ReevalConfig(
        run_dir=run_dir,
        run_id="unittest-run",
        run_seed=42,
        eval_round_num=101,
        model_type="pfedrec",
        embedding_dim=dim,
        dropout=0.0,
        lr=0.1,
        weight_decay=0.0,
        num_train_negatives=4,
        num_eval_negatives=99,
        reuse_cache=False,
        split_hash="unittest-split",
        module="pfedrec",
        lr_eta=80.0,
        loss="bce",
    )


def _mk_run_data(num_users: int = 6, num_items: int = 40, dim: int = 32) -> RunData:
    rng = np.random.default_rng(1)
    users: Dict[int, UserEvalData] = {}
    stats: Dict[int, Dict[str, Any]] = {}
    for pid in range(num_users):
        perm = rng.permutation(num_items)
        train = np.asarray(sorted(int(x) for x in perm[:5]), dtype=np.int64)
        users[pid] = UserEvalData(
            pid=pid,
            train_items=train,
            train_ratings=np.ones(5, dtype=np.float32),
            test_item=int(perm[5]),
            test_rating=1.0,
            exclude_items=np.asarray(sorted(set(train.tolist()) | {int(perm[5])}), dtype=np.int32),
            user_group=USER_GROUPS[pid % 3],
        )
        stats[pid] = {
            "n_interactions": 5 + pid,
            "genre_entropy": 1.0,
            "n_unique_items": 5,
            "rating_std": 0.5,
            "user_group": USER_GROUPS[pid % 3],
        }
    return RunData(
        num_users=num_users, num_items=num_items,
        split_hash="unittest-split", users=users, user_stats=stats,
    )


# ---------------------------------------------------------------------------
# 1-2. Config resolution
# ---------------------------------------------------------------------------


def test_resolve_config_pfedrec_fields(pfedrec_run_dir: Path) -> None:
    cfg = resolve_config(pfedrec_run_dir, module="pfedrec")
    assert cfg.model_type == "pfedrec"
    assert cfg.embedding_dim == 32          # from latent_dim, NOT the 64 default
    assert cfg.lr == 0.1                     # from lr, NOT the 0.001 default
    assert cfg.lr_eta == 80.0
    assert cfg.loss == "bce"
    assert cfg.eval_round_num == 101


@pytest.mark.parametrize("missing", ["latent_dim", "lr", "lr_eta"])
def test_resolve_config_pfedrec_requires_fields(tmp_path: Path, missing: str) -> None:
    payload = _pfedrec_results_payload()
    del payload["federated_config"][missing]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "results.json", "w") as f:
        json.dump(payload, f)
    with pytest.raises(ValueError, match=missing):
        resolve_config(run_dir, module="pfedrec")


def test_module_mismatch_rejected(pfedrec_run_dir: Path) -> None:
    with pytest.raises(ValueError, match="belongs to module"):
        resolve_config(pfedrec_run_dir, module="personalized")


# ---------------------------------------------------------------------------
# 3. Signature + cache-dir parity with the live client_app
# ---------------------------------------------------------------------------


def test_signature_matches_client_app(pfedrec_run_dir: Path) -> None:
    from federated_pfedrec import client_app as ca

    cfg = _mk_cfg(pfedrec_run_dir)
    run_data = _mk_run_data(num_users=6040, num_items=3706)
    adapter = PfedrecAdapter(cfg, cache_base_dir=pfedrec_run_dir / "cache")
    ours = adapter._signature(run_data)
    theirs = ca._signature_fields(
        run_id=cfg.run_id,
        num_users=run_data.num_users,
        num_items=run_data.num_items,
        latent_dim=cfg.embedding_dim,
        split_hash=cfg.split_hash,
        loss=cfg.loss,
        num_train_negatives=cfg.num_train_negatives,
    )
    assert ours == theirs, f"adapter signature drifted from client_app: {ours} != {theirs}"


def test_reuse_cache_dir_rule_matches_client_app(pfedrec_run_dir: Path) -> None:
    """The sig_<hash> rule drops BOTH run_id AND schema_version (pfedrec-only rule)."""
    from federated_pfedrec import client_app as ca

    cfg = _mk_cfg(pfedrec_run_dir)
    cfg.reuse_cache = True
    run_data = _mk_run_data(num_users=6040, num_items=3706)
    adapter = PfedrecAdapter(cfg, cache_base_dir=Path("/base"))
    ours = adapter.resolve_cache_dir(run_data)
    theirs = ca._cache_dir_for_run(
        run_id=cfg.run_id, reuse_cache=True, signature=adapter._signature(run_data)
    )
    assert ours.name == theirs.name, f"sig-dir rule drifted: {ours.name} != {theirs.name}"


# ---------------------------------------------------------------------------
# 4. Blend math
# ---------------------------------------------------------------------------


def test_blend_math(pfedrec_run_dir: Path) -> None:
    cfg = _mk_cfg(pfedrec_run_dir, dim=4)
    adapter = PfedrecAdapter(cfg)
    w_local = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    w_global = torch.tensor([[0.0, 0.0, 10.0, -10.0]])
    local = OrderedDict({"affine_output.weight": w_local})

    out = adapter.blend_local(local, w_global, 0.3)["affine_output.weight"]
    expected = 0.3 * w_local + 0.7 * w_global
    assert torch.allclose(out, expected)

    assert torch.equal(adapter.blend_local(local, w_global, 1.0)["affine_output.weight"], w_local)
    assert torch.equal(adapter.blend_local(local, w_global, 0.0)["affine_output.weight"], w_global)


# ---------------------------------------------------------------------------
# 5. Zero variant hard-rejected; Kaiming random head
# ---------------------------------------------------------------------------


def test_zero_variant_rejected_for_pfedrec(pfedrec_run_dir: Path) -> None:
    adapter = PfedrecAdapter(_mk_cfg(pfedrec_run_dir))
    with pytest.raises(RuntimeError, match="degenerate"):
        adapter.zero_local()


def test_random_local_kaiming_bound_and_determinism(pfedrec_run_dir: Path) -> None:
    cfg = _mk_cfg(pfedrec_run_dir, dim=32)
    adapter = PfedrecAdapter(cfg)
    a = adapter.random_local(7, 42)["affine_output.weight"]
    b = adapter.random_local(7, 42)["affine_output.weight"]
    c = adapter.random_local(8, 42)["affine_output.weight"]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.shape == (1, 32)
    assert a.abs().max().item() <= 1.0 / math.sqrt(32) + 1e-9  # Kaiming, NOT Xavier


# ---------------------------------------------------------------------------
# 6. Trust gate mapping + tolerance + preconditions
# ---------------------------------------------------------------------------


def _gate_metrics() -> Dict[str, float]:
    m: Dict[str, float] = {
        "sampled_hr@10": 0.61, "sampled_ndcg@10": 0.3447, "sampled_mrr": 0.28,
        "evaluated_users": 6040,
    }
    for g, n in zip(USER_GROUPS, (809, 2322, 2909)):
        m[f"sampled_hr@10/{g}"] = 0.6
        m[f"sampled_ndcg@10/{g}"] = 0.34
        m[f"evaluated_users_{g}"] = n
    return m


def _gate_block() -> Dict[str, Any]:
    b: Dict[str, Any] = {
        "sampled_hr@10": 0.61, "sampled_ndcg@10": 0.3447,
        "evaluated_users": 6040.0, "d06_cache_misses": 0,
    }
    for g, n in zip(USER_GROUPS, (809, 2322, 2909)):
        b[f"sampled_hr@10/{g}"] = 0.6
        b[f"sampled_ndcg@10/{g}"] = 0.34
        b[f"evaluated_users/{g}"] = float(n)  # slash-form, float — the run's format
    return b


def test_trust_gate_pass_and_key_mapping() -> None:
    verdict = trust_gate(
        _gate_metrics(), _gate_block(),
        limit_users=None, globals_kind="best",
        eval_round_num=101, final_eval_round_index=101,
    )
    assert verdict["status"] == "PASS"
    assert verdict["deltas"]["evaluated_users/sparse"]["ok"] is True


def test_trust_gate_fails_on_ratio_delta() -> None:
    block = _gate_block()
    block["sampled_ndcg@10"] += 1e-5  # above the 1e-6 tolerance
    verdict = trust_gate(
        _gate_metrics(), block,
        limit_users=None, globals_kind="best",
        eval_round_num=101, final_eval_round_index=101,
    )
    assert verdict["status"] == "FAIL"
    assert "sampled_ndcg@10" in verdict["failed_fields"]


def test_trust_gate_fails_on_count_off_by_one() -> None:
    block = _gate_block()
    block["evaluated_users/dense"] = 2908.0
    verdict = trust_gate(
        _gate_metrics(), block,
        limit_users=None, globals_kind="best",
        eval_round_num=101, final_eval_round_index=101,
    )
    assert verdict["status"] == "FAIL"


def test_trust_gate_refuses_limit_users() -> None:
    verdict = trust_gate(
        _gate_metrics(), _gate_block(),
        limit_users=100, globals_kind="best",
        eval_round_num=101, final_eval_round_index=101,
    )
    assert verdict["status"] == "SKIPPED"


# ---------------------------------------------------------------------------
# 7. Oracle cross-fitting on a planted matrix
# ---------------------------------------------------------------------------


def _planted_matrix(run_data: RunData, good_alpha_by_group: Dict[str, float]):
    """Every user's NDCG peaks at their group's planted alpha (value 1.0, else 0.2)."""
    matrix: Dict[float, Dict[int, Dict[str, Any]]] = {}
    for a in ALPHA_GRID:
        matrix[a] = {}
        for pid, user in run_data.users.items():
            peak = good_alpha_by_group[user.user_group]
            val = 1.0 if abs(a - peak) < 1e-9 else 0.2
            matrix[a][pid] = {
                "pid": pid, "evaluated": 1,
                "sampled_ndcg@10": val, "sampled_hr@10": val, "sampled_mrr": val,
                "eval_loss": 0.5,
            }
    return matrix


def test_oracle_crossfit_recovers_planted_structure() -> None:
    # 90 users -> each group has 30 members (= MIN_BUCKET_USERS) in-sample,
    # 15 per half -> below the threshold, so crossfit falls back to the
    # fit-half's global alpha*; in-sample must recover the planted alphas.
    run_data = _mk_run_data(num_users=90)
    planted = {"sparse": 0.2, "medium": 0.5, "dense": 0.9}
    matrix = _planted_matrix(run_data, planted)
    out = oracle_group(matrix, run_data, staleness_map=None)

    fam = out["families"]["user_group"]
    stars = {row["bucket"]: row["alpha_star_in_sample"] for row in fam["buckets"]}
    assert stars == pytest.approx(planted)
    assert fam["in_sample"]["sampled_ndcg@10"] == pytest.approx(1.0)
    # crossfit <= in-sample by construction on a planted matrix
    assert fam["crossfit"]["sampled_ndcg@10"] <= fam["in_sample"]["sampled_ndcg@10"] + 1e-12
    assert out["best_family"] in out["families"]


def test_oracle_user_upper_bound_and_tiebreak() -> None:
    run_data = _mk_run_data(num_users=6)
    # Flat matrix: every alpha identical -> tie-break must pick the LARGEST alpha.
    matrix = {
        a: {
            pid: {
                "pid": pid, "evaluated": 1, "sampled_ndcg@10": 0.4,
                "sampled_hr@10": 0.4, "sampled_mrr": 0.4, "eval_loss": 0.1,
            }
            for pid in run_data.users
        }
        for a in ALPHA_GRID
    }
    metrics, assign, hist = oracle_user(matrix, run_data)
    assert metrics["sampled_ndcg@10"] == pytest.approx(0.4)
    assert all(a == 1.0 for a in assign.values()), "ties must break toward pure-local"
    assert hist["1.00"] == len(run_data.users)


# ---------------------------------------------------------------------------
# 8. Stage-A verdict bands
# ---------------------------------------------------------------------------


def test_stage_a_verdict_bands() -> None:
    ref_groups = {g: 0.34 for g in USER_GROUPS}

    def _v(gate: float, user: float, star: float = 0.5, buckets=(0.5,)):
        return stage_a_verdict(
            gate_ndcg=0.3447 + gate, user_oracle_ndcg=0.3447 + user,
            reference_ndcg=0.3447, reference_groups=ref_groups,
            crossfit_groups=ref_groups, alpha_star_global=star,
            bucket_stars=list(buckets),
        )

    assert _v(0.012, 0.03)["band"] == "GO"
    assert _v(0.007, 0.03)["band"] == "MARGINAL"
    assert _v(0.002, 0.03)["band"] == "NO-GO"
    # hard NO-GO: user oracle can't clear +0.010 even though gate looks GO
    hard = _v(0.012, 0.005)
    assert hard["band"] == "NO-GO" and hard["hard_nogo_reasons"]
    # hard NO-GO: grid optimum is pure-local everywhere
    pure = _v(0.012, 0.03, star=1.0, buckets=(0.95, 1.0))
    assert pure["band"] == "NO-GO" and pure["hard_nogo_reasons"]


# ---------------------------------------------------------------------------
# 9. Per-user CSV schema
# ---------------------------------------------------------------------------


def test_per_user_csv_schema(tmp_path: Path) -> None:
    rows: List[Dict[str, Any]] = [{
        "pid": 0, "user_group": "sparse", "n_interactions": 12,
        "staleness_at_best": 3, "alpha_used": 0.7,
        "sampled_ndcg@10": 0.5, "sampled_hr@10": 1.0, "sampled_mrr": 0.5,
        "eval_loss": 0.61, "evaluated": 1,
    }]
    path = write_per_user_csv(tmp_path / "x.csv", rows)
    lines = path.read_text().strip().splitlines()
    assert lines[0] == (
        "pid,user_group,n_interactions,staleness_at_best,alpha_used,"
        "sampled_ndcg@10,sampled_hr@10,eval_loss,evaluated"
    )
    assert len(lines) == 2


# ---------------------------------------------------------------------------
# 10. Global-head builders
# ---------------------------------------------------------------------------


def test_global_head_means(tmp_path: Path, pfedrec_run_dir: Path) -> None:
    cfg = _mk_cfg(pfedrec_run_dir, dim=2)
    run_data = _mk_run_data(num_users=3, num_items=10, dim=2)
    # n_interactions: pid0=5, pid1=6, pid2=7 (from _mk_run_data: 5+pid)
    cache = tmp_path / "cache" / cfg.run_id
    cache.mkdir(parents=True)
    heads = [
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[1.0, 1.0]]),
    ]
    for pid, h in enumerate(heads):
        torch.save(OrderedDict({"affine_output.weight": h}), cache / f"partition_{pid}.pt")
    adapter = PfedrecAdapter(cfg, cache_base_dir=tmp_path / "cache")
    adapter.resolve_cache_dir(run_data)

    w_u, spec_u = compute_global_head(adapter, run_data, "uniform", 42)
    assert torch.allclose(w_u, torch.tensor([[2.0 / 3.0, 2.0 / 3.0]]))
    assert spec_u["n_heads"] == 3

    w_w, _ = compute_global_head(adapter, run_data, "weighted", 42)
    expected = (5 * heads[0] + 6 * heads[1] + 7 * heads[2]) / 18.0
    assert torch.allclose(w_w, expected)

    w_c1, _ = compute_global_head(adapter, run_data, "cold", 42)
    w_c2, _ = compute_global_head(adapter, run_data, "cold", 42)
    assert torch.equal(w_c1, w_c2)
    assert w_c1.abs().max().item() <= 1.0 / math.sqrt(2) + 1e-9
