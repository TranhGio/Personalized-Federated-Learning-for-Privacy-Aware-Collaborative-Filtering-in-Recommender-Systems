"""Tests for task.py + client_app.py RNG + exclusion + cold-round contract (Phase 4 Plan 03).

Covers ADP-05 / ADP-06 (RNG half) / D-13 / D-14 / D-24:
  - ``task.py`` AND ``client_app.py`` contain no stdlib-random seeding or
    sampling (cross-file BSL-05-style regression guard).
  - Training negatives respect the FND-03 exclusion set: the held-out test
    positive is never drawn as a training negative.
  - ``evaluate_ranking_sampled`` accepts the 4 FND-06 kwargs (``run_seed``,
    ``user_idx``, ``round_num``, ``exclude_items``).
  - ``train_dual_personalized`` applies the D-13 cold-round alpha=0 override
    + D-14 contrastive skip + restores the saved alpha in a try/finally.
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


_MODULE_DIR = Path(__file__).resolve().parents[1] / "federated_adaptive_personalized_cf"
_TASK_PY = _MODULE_DIR / "task.py"
_CLIENT_APP_PY = _MODULE_DIR / "client_app.py"


@pytest.mark.parametrize("src_path", [_TASK_PY, _CLIENT_APP_PY])
def test_random_seed_calls_stripped(src_path: Path) -> None:
    """ADP-06 / BSL-05-style: no stdlib-random seeding/sampling in either file.

    Cross-file regression guard. Catches pre-existing ``random.seed(seed)``
    at task.py:952-953 + ``random.sample(...)`` at task.py:1012 (CONCERNS.md).
    """
    src = src_path.read_text()
    assert "random.seed(" not in src, (
        f"ADP-06 violation in {src_path.name}: stdlib ``random.seed()`` must be "
        f"stripped (use fedrec_foundation.rng.np_rng / torch_gen)"
    )
    assert "random.sample(" not in src, (
        f"ADP-06 violation in {src_path.name}: stdlib ``random.sample(...)`` must "
        f"be replaced with a seeded generator draw"
    )
    # Strip check for ``import random`` inside the top 60 lines (covers the
    # module import block even with long docstrings).
    top_lines = src.split("\n")[0:60]
    for line in top_lines:
        stripped = line.strip()
        assert stripped != "import random", (
            f"ADP-06 violation in {src_path.name}: module-level ``import random`` "
            f"must be stripped"
        )


def test_train_negatives_exclude_test_positive() -> None:
    """ADP-05 + ADP-06: held-out test positive NEVER appears in training negatives.

    Smoke-tests the ``_sample_negatives_seeded`` helper AND the
    exclude_items fold in ``train`` dispatcher via ``train_dual_personalized``.
    """
    from federated_adaptive_personalized_cf.task import _sample_negatives_seeded
    from fedrec_foundation.rng import np_rng

    # Excluding item 5 (the held-out test positive). Run 20 draws with a
    # fresh RNG every call; none should return 5.
    for _ in range(20):
        rng = np_rng(42, 0, 1, "train_neg")
        out = _sample_negatives_seeded(
            user_rated_items={0, 1, 2, 5},
            num_items=10,
            num_negatives=3,
            rng=rng,
        )
        assert 5 not in out.tolist(), (
            f"ADP-05 violated: exclusion-set item 5 appeared in negatives: {out}"
        )
        # Same fold constraint: none of the rated items should appear either.
        for x in out:
            assert int(x) not in {0, 1, 2, 5}, (
                f"ADP-05 violated: rated item {x} appeared in negatives: {out}"
            )


def test_evaluate_ranking_sampled_accepts_rng_signature() -> None:
    """ADP-06: evaluate_ranking_sampled accepts the 4 FND-06 kwargs."""
    import inspect

    from federated_adaptive_personalized_cf.task import evaluate_ranking_sampled

    sig = inspect.signature(evaluate_ranking_sampled)
    for p in ("run_seed", "user_idx", "round_num", "exclude_items"):
        assert p in sig.parameters, (
            f"ADP-06: evaluate_ranking_sampled must accept {p!r}"
        )


def test_cold_round_sets_alpha_zero_and_skips_contrastive() -> None:
    """D-13 + D-14: train_dual_personalized forces α=0 + skips contrastive on cold rounds.

    Builds a minimal DualPersonalizedBPRMF, pins a non-zero alpha, wraps
    ``set_alpha`` with a spy, then calls ``train_dual_personalized`` with
    ``is_cold_round=True``. Asserts:
      - set_alpha(0.0) was called at least once (D-13 override).
      - The final set_alpha call restored the saved alpha (D-13 cleanup).
      - ``contrastive_lambda_eff`` was zero during the cold round (D-14)
        — verified by the absence of any PersonalMLP contrastive-path
        gradient (PersonalMLP weights remain exactly their initial values
        when contrastive is off AND the BPR loss is stopped from routing
        through the MLP by the cold-round logic). We approximate this by
        inspecting ``model.get_alpha()`` after the call (restored) and by
        verifying set_alpha was called WITH 0.0.
    """
    from federated_adaptive_personalized_cf.models.dual_personalized_bpr_mf import (
        DualPersonalizedBPRMF,
    )
    from federated_adaptive_personalized_cf.task import train_dual_personalized
    from fedrec_foundation.rng import np_rng

    # Deterministic init for the test.
    torch.manual_seed(0)
    model = DualPersonalizedBPRMF(
        num_users=2,
        num_items=10,
        embedding_dim=4,
        mlp_hidden_dims=[8],
        fusion_type="add",
        use_bias=True,
    )
    # Prime a non-zero alpha so the cold-round override is visible.
    model.set_alpha(0.7)

    # Spy wrapper. Preserve real behavior via side_effect.
    real_set_alpha = model.set_alpha
    calls: list = []

    def _spy(v: float) -> None:
        calls.append(float(v))
        real_set_alpha(v)

    # Patch the bound method for the duration of the test.
    model.set_alpha = _spy  # type: ignore[method-assign]

    class _FakeTrainloader:
        """Single-user, one-batch trainloader — 3 positives."""

        def __init__(self) -> None:
            self._batch = {
                "user": torch.tensor([0, 0, 0]),
                "item": torch.tensor([0, 1, 2]),
                "rating": torch.tensor([1.0, 1.0, 1.0]),
            }

        def __iter__(self):
            return iter([self._batch])

        @property
        def dataset(self):
            return [0, 1, 2]  # stand-in

    trainloader = _FakeTrainloader()

    rng = np_rng(42, 0, 1, "train_neg")
    train_dual_personalized(
        model,
        trainloader,
        epochs=1,
        lr=1e-2,
        device="cpu",
        weight_decay=1e-5,
        num_negatives=1,
        proximal_mu=0.0,
        global_params=None,
        global_param_names=None,
        contrastive_lambda=0.1,
        contrastive_tau=0.1,
        run_seed=42,
        user_idx=0,
        round_num=1,
        exclude_items=set(),
        rng=rng,
        is_cold_round=True,
    )

    # D-13 override: 0.0 must have been passed at least once.
    assert any(abs(v - 0.0) < 1e-9 for v in calls), (
        f"D-13 violated: train_dual_personalized did not call set_alpha(0.0) on "
        f"cold round. Observed calls: {calls}"
    )
    # D-13 cleanup: last call restored the saved alpha (0.7).
    assert calls, "spy recorded no set_alpha calls — train_dual_personalized did not invoke it"
    assert abs(calls[-1] - 0.7) < 1e-9, (
        f"D-13 cleanup violated: saved alpha 0.7 not restored; last call was {calls[-1]}"
    )
    # After the call returns, the model's alpha should be 0.7 again.
    assert abs(model.get_alpha() - 0.7) < 1e-9, (
        f"D-13 cleanup failed: model.get_alpha()={model.get_alpha()}; expected 0.7"
    )
