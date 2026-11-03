"""Tests for the Q_u gate variants (``qu_grid`` / ``qu_diag``) in scripts/thesis/reeval.py.

Pins the pre-registered guarantees of
``results/federated/pfedrec/20260726-083121-cf72e0/gate_qu/README.md`` (v1.1)
on a small synthetic PFedRec run (CPU, no flwr, no real data):

- the masked blend leaves untouched rows bit-identical for every alpha, while
  a naive all-row blend does not (the float32 1-ulp argument in §3 step 4);
- the one-epoch fine-tune freezes the head, moves only rows seen in a batch,
  and is bit-reproducible for a fixed generator;
- the clean negative pool can draw the test item, the leaky pool never does;
- alpha = 1.0 cells reproduce the cached replay per user, and all §7.2 trust
  checks pass end-to-end; artifacts land in ``--out-dir`` with the 15-column CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

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
    QU_EPOCH_SNAPSHOTS,
    QU_GRID_ALPHAS,
    RunData,
    UserEvalData,
    _QU_PER_USER_FIELDS,
    blend_item_table,
    finetune_item_table_one_epoch,
    get_adapter,
    qu_cell_key,
    resolve_config,
    run_qu_variant,
    run_reeval,
    write_per_user_csv,
    write_qu_per_user_csv,
)
from fedrec_foundation.rng import torch_gen  # noqa: E402

NUM_USERS, NUM_ITEMS, DIM, N_TRAIN = 24, 60, 8, 10
RUN_SEED, EVAL_ROUND = 42, 9
RUN_ID = "unittest-qu"
GROUPS = ("sparse", "medium", "dense")


def _build_run(tmp_path: Path) -> Tuple[Any, Any, RunData, Dict[str, torch.Tensor], Path]:
    """Synthetic PFedRec run dir + head cache + RunData (mirrors the tool's self-test)."""
    from federated_pfedrec.task import get_model

    run_dir = tmp_path / "run"
    cache_base = tmp_path / "embedding_cache"
    run_dir.mkdir()
    (cache_base / RUN_ID).mkdir(parents=True)

    model = get_model(num_items=NUM_ITEMS, latent_dim=DIM)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        model.embedding_item.weight.copy_(0.3 * torch.randn(NUM_ITEMS, DIM, generator=g))
        model.affine_output.bias.fill_(-0.2)
    global_state = model.get_global_parameters()
    torch.save(global_state, run_dir / "global_state_best.pt")

    results = {
        "_manifest": {
            "run_id": RUN_ID, "mode": "paper_compat_pfedrec", "module": "pfedrec",
            "run_seed": RUN_SEED, "split_hash": "unittest-split",
            "final_eval_round_index": EVAL_ROUND, "num_eval_negatives": 20,
            "num_train_negatives": 4, "fraction_train": 0.1, "git_commit": "unittest",
        },
        "federated_config": {
            "latent_dim": DIM, "lr": 0.1, "lr_eta": 80.0, "loss": "bce",
            "l2_regularization": 0.0, "reuse_cache": False, "num_clients": NUM_USERS,
        },
        "final_metrics": {"final_eval_round_index": EVAL_ROUND},
        "training_rounds": 8,
        "checkpoint": {"best_round": 6, "rule": "best_round_restore"},
        "selected_clients_per_round": [[p for p in range(NUM_USERS) if (p + r) % 3 == 0] for r in range(8)],
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f)

    data_rng = np.random.default_rng(5)
    users: Dict[int, UserEvalData] = {}
    stats: Dict[int, Dict[str, Any]] = {}
    for pid in range(NUM_USERS):
        perm = data_rng.permutation(NUM_ITEMS)
        train = np.asarray(sorted(int(x) for x in perm[:N_TRAIN]), dtype=np.int64)
        test = int(perm[N_TRAIN])
        users[pid] = UserEvalData(
            pid=pid, train_items=train, train_ratings=np.ones(N_TRAIN, dtype=np.float32),
            test_item=test, test_rating=1.0,
            exclude_items=np.asarray(sorted(set(train.tolist()) | {test}), dtype=np.int32),
            user_group=GROUPS[pid % 3],
        )
        stats[pid] = {"n_interactions": N_TRAIN, "user_group": GROUPS[pid % 3]}
        head = torch.from_numpy(data_rng.normal(size=(1, DIM)).astype(np.float32))
        torch.save(OrderedDict({"affine_output.weight": head}), cache_base / RUN_ID / f"partition_{pid}.pt")
    with open(cache_base / RUN_ID / "manifest.json", "w") as f:
        json.dump({
            "schema_version": 3, "run_id": RUN_ID, "method": "pfedrec",
            "num_users": NUM_USERS, "num_items": NUM_ITEMS, "latent_dim": DIM,
            "split_hash": "unittest-split", "loss": "bce", "num_train_negatives": 4,
            "bias_classification": "global",
        }, f)

    cfg = resolve_config(run_dir, module="pfedrec")
    run_data = RunData(num_users=NUM_USERS, num_items=NUM_ITEMS, split_hash="unittest-split",
                       users=users, user_stats=stats)
    adapter = get_adapter("pfedrec", cfg, cache_base_dir=cache_base)
    adapter.inject_run_data(run_data)
    adapter.resolve_cache_dir(run_data)
    return cfg, adapter, run_data, global_state, run_dir


@pytest.fixture()
def synth(tmp_path: Path):
    return _build_run(tmp_path)


# ---------------------------------------------------------------------------
# Unit pins
# ---------------------------------------------------------------------------


def test_masked_blend_keeps_untouched_rows_bit_exact():
    g = torch.Generator().manual_seed(3)
    q = torch.randn(200, 16, generator=g)
    q_user = q.clone()
    touched = torch.tensor([1, 5, 9, 40], dtype=torch.long)
    q_user[touched] += torch.randn(len(touched), 16, generator=g)
    untouched = torch.ones(200, dtype=torch.bool)
    untouched[touched] = False
    for a in QU_GRID_ALPHAS + (0.0,):
        out = blend_item_table(q, q_user, touched, a)
        assert torch.equal(out[untouched], q[untouched]), a
    expected = 0.3 * q[touched] + 0.7 * q_user[touched]
    assert torch.equal(blend_item_table(q, q_user, touched, 0.3)[touched], expected)
    assert torch.equal(blend_item_table(q, q_user, touched, 1.0), q)


def test_naive_all_row_blend_is_not_bit_exact():
    """The README §3 justification: a*q + (1-a)*q != q in float32 for a=0.3/0.7."""
    q = torch.randn(3706, 32, generator=torch.Generator().manual_seed(0))
    for a in (0.3, 0.7):
        assert not torch.equal(a * q + (1.0 - a) * q, q)


def test_finetune_freezes_head_and_moves_only_batch_rows(synth):
    cfg, adapter, run_data, global_state, _ = synth
    from federated_pfedrec.task import prepare_user_train_data

    user = run_data.users[0]
    model = adapter.build_model(run_data)
    adapter.set_globals(model, global_state)
    adapter.set_local(model, adapter.load_cached_local(0))
    head_before = model.affine_output.weight.detach().clone()
    bias_before = model.affine_output.bias.detach().clone()
    q_before = model.embedding_item.weight.detach().clone()
    items, ratings = prepare_user_train_data(
        user_idx=0, user_train_items=user.train_items.tolist(), num_items=NUM_ITEMS,
        num_negatives=4, run_seed=RUN_SEED, round_num=201, exclude_items=None,
    )
    finetune_item_table_one_epoch(model, items, ratings, 0.1 * NUM_ITEMS * 80, 256,
                                  torch_gen(RUN_SEED, 0, 201, "dataloader"))
    assert torch.equal(model.affine_output.weight, head_before)
    assert torch.equal(model.affine_output.bias, bias_before)
    q_after = model.embedding_item.weight.detach()
    seen = sorted(set(items))
    unseen = [i for i in range(NUM_ITEMS) if i not in set(seen)]
    assert torch.equal(q_after[unseen], q_before[unseen])
    assert not torch.equal(q_after[seen], q_before[seen])


def test_finetune_is_deterministic_for_fixed_generator(synth):
    cfg, adapter, run_data, global_state, _ = synth
    from federated_pfedrec.task import prepare_user_train_data

    user = run_data.users[3]
    tables = []
    for _ in range(2):
        model = adapter.build_model(run_data)
        adapter.set_globals(model, global_state)
        adapter.set_local(model, adapter.load_cached_local(3))
        items, ratings = prepare_user_train_data(
            user_idx=3, user_train_items=user.train_items.tolist(), num_items=NUM_ITEMS,
            num_negatives=4, run_seed=RUN_SEED, round_num=201, exclude_items=None,
        )
        finetune_item_table_one_epoch(model, items, ratings, 48.0, 8, torch_gen(RUN_SEED, 3, 201, "dataloader"))
        tables.append(model.embedding_item.weight.detach().clone())
    assert torch.equal(tables[0], tables[1])


def test_clean_pool_can_draw_test_item_leaky_never(synth):
    _, _, run_data, _, _ = synth
    from federated_pfedrec.task import prepare_user_train_data

    hits = {"clean": 0, "leaky": 0}
    for pid, user in run_data.users.items():
        for e in (1, 2, 3):
            for pool, exclude in (("clean", None), ("leaky", user.exclude_items)):
                items, ratings = prepare_user_train_data(
                    user_idx=pid, user_train_items=user.train_items.tolist(), num_items=NUM_ITEMS,
                    num_negatives=4, run_seed=RUN_SEED, round_num=200 + e, exclude_items=exclude,
                )
                negs = [i for i, r in zip(items, ratings) if r == 0.0]
                assert not set(negs) & set(user.train_items.tolist())
                hits[pool] += sum(1 for i in negs if i == user.test_item)
    assert hits["clean"] > 0
    assert hits["leaky"] == 0


def test_cell_keys():
    assert qu_cell_key(0.3, 1, "clean", False) == "a0.30_E1"
    assert qu_cell_key(1.0, 3, "clean", False) == "a1.00_E3"
    assert qu_cell_key(0.0, 3, "leaky", True) == "a0.00_E3_leaky"


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_qu_grid_end_to_end_trust_checks_pass(synth, tmp_path):
    cfg, adapter, run_data, global_state, _ = synth
    base_m, base_rows = run_reeval(adapter, cfg, run_data, global_state, "cached")
    base_csv = write_per_user_csv(tmp_path / "base.csv", base_rows)
    metrics, rows, checks = run_qu_variant(
        "qu_grid", adapter, cfg, run_data, global_state,
        baseline_rows=base_rows, baseline_csv=base_csv,
    )
    assert checks["status"] == "PASS", checks["failures"]
    expected_keys = {qu_cell_key(a, e, "clean", False) for e in QU_EPOCH_SNAPSHOTS for a in QU_GRID_ALPHAS}
    assert set(metrics) == expected_keys
    for e in QU_EPOCH_SNAPSHOTS:
        m1 = metrics[qu_cell_key(1.0, e, "clean", False)]
        for k in ("sampled_ndcg@10", "sampled_hr@10", "sampled_mrr"):
            assert m1[k] == base_m[k], (e, k)
    assert len(rows) == NUM_USERS * len(expected_keys)
    assert checks["n_hit"]["clean"] > 0
    assert checks["candidate_list_errors"] == 0
    # Personalisation actually changes something in at least one non-trivial cell.
    assert any(
        metrics[qu_cell_key(a, e, "clean", False)]["sampled_ndcg@10"] != base_m["sampled_ndcg@10"]
        for e in QU_EPOCH_SNAPSHOTS for a in QU_GRID_ALPHAS if a < 1.0
    )
    # Untouched-candidate users have d_u == 0 in every cell (checked inside, re-pinned here).
    base_by_pid = {r["pid"]: r["sampled_ndcg@10"] for r in base_rows}
    for r in rows:
        if r["n_touched_candidates"] == 0:
            assert r["sampled_ndcg@10"] == base_by_pid[r["pid"]]
        assert 0 <= r["n_touched_candidates"] <= 21
        assert r["test_row_touched"] in (0, 1)

    path = write_qu_per_user_csv(tmp_path / "qu.csv", rows)
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        n = sum(1 for _ in reader)
    assert tuple(header) == _QU_PER_USER_FIELDS and len(header) == 15
    assert n == len(rows)


def test_qu_diag_pools(synth):
    cfg, adapter, run_data, global_state, _ = synth
    _, base_rows = run_reeval(adapter, cfg, run_data, global_state, "cached")
    metrics, rows, checks = run_qu_variant(
        "qu_diag", adapter, cfg, run_data, global_state, baseline_rows=base_rows,
    )
    assert checks["status"] == "PASS", checks["failures"]
    assert set(metrics) == {"a0.00_E1_clean", "a0.00_E3_clean", "a0.00_E1_leaky", "a0.00_E3_leaky"}
    assert checks["n_hit"]["leaky"] == 0 and checks["n_hit"]["clean"] > 0
    assert {r["pool"] for r in rows} == {"clean", "leaky"}


def test_suite_writes_to_out_dir_and_leaves_run_dir_untouched(synth, tmp_path):
    cfg, adapter, run_data, global_state, run_dir = synth
    base_m, _ = run_reeval(adapter, cfg, run_data, global_state, "cached")
    best_block: Dict[str, Any] = {
        "sampled_hr@10": base_m["sampled_hr@10"], "sampled_ndcg@10": base_m["sampled_ndcg@10"],
        "evaluated_users": float(NUM_USERS), "d06_cache_misses": 0,
    }
    for g in GROUPS:
        best_block[f"sampled_hr@10/{g}"] = base_m[f"sampled_hr@10/{g}"]
        best_block[f"sampled_ndcg@10/{g}"] = base_m[f"sampled_ndcg@10/{g}"]
        best_block[f"evaluated_users/{g}"] = float(base_m[f"evaluated_users_{g}"])
    before = sorted(p.name for p in run_dir.iterdir())
    out_dir = tmp_path / "gate_qu"
    args = argparse.Namespace(
        out_dir=out_dir, device="cpu", skip_trust_gate=False, limit_users=None,
        globals="best", no_json=False, no_per_user=False,
    )
    metrics = reeval._run_qu_suite_variant(
        "qu_grid", args, cfg, adapter, run_data, global_state, run_dir / "global_state_best.pt",
        best_block=best_block, final_eval_round_index=EVAL_ROUND, staleness_map=None,
    )
    assert sorted(p.name for p in run_dir.iterdir()) == before
    payload = json.loads((out_dir / "reeval_qu_grid_best.json").read_text())
    assert payload["trust_gate"]["status"] == "PASS"
    assert payload["trust_checks"]["status"] == "PASS"
    assert payload["metrics_source"] == "a1.00_E1"
    assert payload["metrics"] == metrics["a1.00_E1"]
    assert set(payload["metrics_by_cell"]) == set(metrics)
    assert (out_dir / "reeval_qu_grid_best_per_user.csv").exists()


def test_suite_aborts_when_trust_gate_fails(synth, tmp_path):
    cfg, adapter, run_data, global_state, run_dir = synth
    base_m, _ = run_reeval(adapter, cfg, run_data, global_state, "cached")
    bad_block = {"sampled_ndcg@10": base_m["sampled_ndcg@10"] + 1e-3, "d06_cache_misses": 0}
    args = argparse.Namespace(
        out_dir=tmp_path / "gate_qu", device="cpu", skip_trust_gate=False, limit_users=None,
        globals="best", no_json=False, no_per_user=False,
    )
    with pytest.raises(SystemExit) as exc:
        reeval._run_qu_suite_variant(
            "qu_grid", args, cfg, adapter, run_data, global_state, run_dir / "global_state_best.pt",
            best_block=bad_block, final_eval_round_index=EVAL_ROUND, staleness_map=None,
        )
    assert exc.value.code == 3
    assert not (tmp_path / "gate_qu" / "reeval_qu_grid_best_per_user.csv").exists()


def test_dispatch_through_run_pfedrec_suite(synth, tmp_path):
    """qu_grid is reachable from the real suite driver (CLI path) and writes to --out-dir."""
    cfg, adapter, run_data, global_state, run_dir = synth
    out_dir = tmp_path / "gate_qu_cli"
    args = reeval.build_parser().parse_args([
        "--run-dir", str(run_dir), "--module", "pfedrec", "--local-variant", "qu_grid",
        "--skip-trust-gate", "--out-dir", str(out_dir),
    ])
    rc = reeval.run_pfedrec_suite(args, cfg, adapter, run_data, global_state, run_dir / "global_state_best.pt")
    assert rc == 0
    payload = json.loads((out_dir / "reeval_qu_grid_best.json").read_text())
    assert payload["trust_gate"]["status"] == "SKIPPED"
    assert payload["trust_checks"]["status"] == "PASS"
    assert payload["lr_item"] == pytest.approx(0.1 * NUM_ITEMS * 80.0)
    assert not list(run_dir.glob("reeval_*"))
