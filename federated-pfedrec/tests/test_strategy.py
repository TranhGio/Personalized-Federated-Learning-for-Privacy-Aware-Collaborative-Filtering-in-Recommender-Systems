"""Phase 5 Plan 01 strategy tests.

Pins:
- D-12 rename ``SplitFedAvg`` -> ``PFedRecSplitFedAvg``.
- D-07 drop ``SplitFedProx`` (no FedProx variant for PFedRec).
- D-01 frozenset symmetry: ``GLOBAL_PARAM_KEYS = {'embedding_item.weight',
  'affine_output.bias'}`` (bias-GLOBAL); ``LOCAL_PARAM_KEYS =
  {'affine_output.weight'}``.
- D-24 / D-26 sufficient-stat ``aggregate_evaluate`` — sums sufficient stats and
  divides ONCE at the end (matches ``IJCAI-23-PFedRec/engine.py:81``
  ``len(round_user_params)`` uniform mean).
"""
from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Test 1: D-12 rename + D-07 FedProx drop.
# ---------------------------------------------------------------------------


def test_strategy_class_renamed_pfedrecsplitfedavg() -> None:
    """``PFedRecSplitFedAvg`` is the new public class name; legacy names removed."""
    from federated_pfedrec import strategy as strategy_module
    from federated_pfedrec.strategy import PFedRecSplitFedAvg

    # The new D-12 class exists.
    assert PFedRecSplitFedAvg is not None
    assert PFedRecSplitFedAvg.__name__ == "PFedRecSplitFedAvg"

    # Old names are gone (D-12 strict rename + D-07 FedProx drop).
    module_attrs = dir(strategy_module)
    assert "SplitFedAvg" not in module_attrs, (
        "D-12: legacy 'SplitFedAvg' name MUST be removed in favor of "
        "'PFedRecSplitFedAvg'"
    )
    assert "SplitFedProx" not in module_attrs, (
        "D-07: 'SplitFedProx' MUST be removed (PFedRec uses FedAvg only)"
    )
    assert "PFedRecSplitFedProx" not in module_attrs, (
        "D-07: no FedProx variant ships for PFedRec"
    )


# ---------------------------------------------------------------------------
# Test 2: D-01 frozenset symmetry.
# ---------------------------------------------------------------------------


def test_global_param_keys_includes_bias() -> None:
    """``affine_output.bias`` lives in GLOBAL_PARAM_KEYS (D-01: bias is GLOBAL)."""
    from federated_pfedrec.strategy import GLOBAL_PARAM_KEYS, LOCAL_PARAM_KEYS

    assert GLOBAL_PARAM_KEYS == frozenset(
        {"embedding_item.weight", "affine_output.bias"}
    ), (
        "D-01 source-of-truth (engine.py:143 deletes only affine_output.weight): "
        "GLOBAL_PARAM_KEYS must hold both 'embedding_item.weight' and "
        "'affine_output.bias'."
    )
    assert LOCAL_PARAM_KEYS == frozenset({"affine_output.weight"}), (
        "D-01: only 'affine_output.weight' is per-user LOCAL."
    )
    # Disjointness — every key is classified exactly once.
    assert GLOBAL_PARAM_KEYS.isdisjoint(LOCAL_PARAM_KEYS)


# ---------------------------------------------------------------------------
# Test 3: bias is NOT in LOCAL_PARAM_KEYS.
# ---------------------------------------------------------------------------


def test_local_param_keys_excludes_bias() -> None:
    """Defense-in-depth: ``affine_output.bias`` MUST NOT appear in LOCAL_PARAM_KEYS."""
    from federated_pfedrec.strategy import GLOBAL_PARAM_KEYS, LOCAL_PARAM_KEYS

    assert "affine_output.bias" not in LOCAL_PARAM_KEYS, (
        "D-01 regression guard: bias was previously LOCAL (CONCERNS divergence #9). "
        "Phase 5 D-01 moves it to GLOBAL — do not regress."
    )
    assert "affine_output.bias" in GLOBAL_PARAM_KEYS


# ---------------------------------------------------------------------------
# Test 4: D-24 / D-26 sufficient-stat aggregate_evaluate.
# ---------------------------------------------------------------------------


def _build_eval_res(
    hit_count_overall: int,
    ndcg_sum_overall: float,
    evaluated_users: int,
    eval_loss: float,
) -> MagicMock:
    """Construct a mocked ``EvaluateRes``-like object with a ``metrics`` dict."""
    eval_res = MagicMock()
    eval_res.metrics = {
        "hit_count_overall_at10": hit_count_overall,
        "ndcg_sum_overall_at10": ndcg_sum_overall,
        "evaluated_users": evaluated_users,
        "eval_loss": eval_loss,
    }
    eval_res.num_examples = evaluated_users
    eval_res.loss = eval_loss
    return eval_res


def test_aggregate_evaluate_sufficient_stat_uniform() -> None:
    """``aggregate_evaluate`` sums sufficient stats and divides once (D-24, D-26).

    Reference parity: ``IJCAI-23-PFedRec/engine.py:81`` divides aggregated
    sums by ``len(round_user_params)`` — uniform per-user weight. With 1
    user = 1 client cross-device, summing per-client sufficient stats and
    dividing ONCE by total ``evaluated_users`` is mathematically uniform.
    """
    from federated_pfedrec.strategy import PFedRecSplitFedAvg

    results: List[MagicMock] = [
        (MagicMock(), _build_eval_res(1, 0.6, 1, 0.5)),
        (MagicMock(), _build_eval_res(0, 0.0, 1, 0.7)),
        (MagicMock(), _build_eval_res(1, 0.4, 1, 0.3)),
    ]

    strategy = PFedRecSplitFedAvg()
    loss, thesis = strategy.aggregate_evaluate(
        server_round=1, results=results, failures=[]
    )

    # 2 hits / 3 users = 0.6666...
    assert thesis["sampled_hr@10"] == pytest.approx(2.0 / 3.0)
    # 1.0 NDCG sum / 3 users = 0.3333...
    assert thesis["sampled_ndcg@10"] == pytest.approx(1.0 / 3.0)
    # 3 evaluated users surfaced.
    assert thesis["evaluated_users"] == pytest.approx(3.0)
    # Loss = mean of the three eval_loss values = (0.5 + 0.7 + 0.3) / 3 = 0.5.
    assert loss == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Pitfall 1 guard — strategy frozensets agree with model param tuples.
# ---------------------------------------------------------------------------


def test_global_param_keys_matches_model_tuple() -> None:
    """``set(GLOBAL_PARAM_KEYS) == set(PFedRecMLP._GLOBAL_PARAMS)`` (Pitfall 1).

    Mechanically prevents drift between the strategy-side wire contract and
    the model-side state-dict-classification tuple.
    """
    from federated_pfedrec.models.pfedrec_mlp import PFedRecMLP
    from federated_pfedrec.strategy import GLOBAL_PARAM_KEYS, LOCAL_PARAM_KEYS

    assert set(GLOBAL_PARAM_KEYS) == set(PFedRecMLP._GLOBAL_PARAMS)
    assert set(LOCAL_PARAM_KEYS) == set(PFedRecMLP._LOCAL_PARAMS)
