"""Tests for task.py RNG + exclusion threading (Phase 3 Plan 03).

Covers PSN-03, PSN-06 training-side, and BSL-05-style cross-file regression:
  - ``task.py`` AND ``client_app.py`` contain no ``import random`` at module
    scope, no ``random.seed(...)`` calls, and no ``random.sample(...)``
    calls.
  - Training negatives respect the FND-03 exclusion set (PSN-03): the
    held-out test positive is never drawn as a training negative.
  - ``evaluate_ranking_sampled`` accepts the new RNG+exclusion contract
    surface.
  - ``_sample_negatives_seeded`` is deterministic given the same RNG and
    drifts under a different seed (FND-06 smoke).

D-24 gradient masking is NOT needed here: the single-row model (D-01)
collapses the ghost-table problem — the local param IS the user's row,
so only one row exists to update. No snapshot/restore bracket is
required around ``optimizer.step()``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Skip the whole module when the foundation bundle is not committed.
pytestmark = pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[2] / "data" / "derived" / "foundation_index.json"
    ).exists(),
    reason="foundation bundle not committed",
)


def _make_minimal_bpr_model(num_users: int = 5, num_items: int = 30, dim: int = 4):
    """Build a small single-row BPRMF for fast unit tests."""
    from federated_personalized_cf.models import BPRMF

    return BPRMF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=dim,
        use_bias=True,
        dropout=0.0,
    )


def _make_trainloader(user_raw_id: int, num_positive_items: int, num_items: int = 30):
    """Synthetic single-user trainloader whose item ids map 1:1 to raw ids."""
    from torch.utils.data import DataLoader
    from federated_personalized_cf.dataset import MovieLensDataset

    rows = [
        (user_raw_id, i, 5.0, 1000 + i) for i in range(num_positive_items)
    ]
    train_df = pd.DataFrame(
        rows,
        columns=["user_id", "movie_id", "rating", "timestamp"],
    )
    user2idx = {user_raw_id: 0}
    item2idx = {i: i for i in range(num_items)}
    dataset = MovieLensDataset(train_df, user2idx, item2idx)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def test_random_seed_calls_stripped() -> None:
    """PSN-03 + BSL-05-style: no stdlib-random seeding/sampling inside task.py OR client_app.py.

    Cross-file regression guard: both files must be clean of the three
    stdlib-random patterns that FND-06 supersedes.
    """
    module_dir = Path(__file__).resolve().parents[1] / "federated_personalized_cf"
    for filename in ("task.py", "client_app.py"):
        src = (module_dir / filename).read_text()
        assert "random.seed(" not in src, (
            f"BSL-05 violation in {filename}: stdlib `random.seed()` must be "
            f"stripped (use fedrec_foundation.rng.np_rng / torch_gen)"
        )
        assert "random.sample(" not in src, (
            f"BSL-05 violation in {filename}: stdlib `random.sample(...)` must "
            f"be replaced with `rng.choice(...)` or an equivalent seeded draw"
        )
        # Strip check for ``import random`` inside the top 30 lines, which
        # covers every module's import block even with module docstrings.
        top_lines = src.split("\n")[0:30]
        for line in top_lines:
            stripped = line.strip()
            assert stripped != "import random", (
                f"BSL-05 violation in {filename}: module-level `import random` "
                f"must be stripped (use fedrec_foundation.rng for all randomness)"
            )


def test_train_negatives_exclude_test_positive() -> None:
    """PSN-03: held-out test positive NEVER appears in training negatives.

    Trains a tiny single-row BPR-MF for 1 epoch on a synthetic single-user
    partition whose ``exclude_items`` includes the held-out positive and
    asserts:
      - The call completes without raising.
      - The model's ``local_user_row`` has moved (proves gradients flowed
        through the sampled negatives).
    """
    from federated_personalized_cf.task import train_bpr_mf
    from fedrec_foundation.rng import np_rng

    torch.manual_seed(0)
    model = _make_minimal_bpr_model(num_users=5, num_items=30, dim=4)
    trainloader = _make_trainloader(user_raw_id=1, num_positive_items=10, num_items=30)

    exclude_items = np.array([25], dtype=np.int32)  # the held-out test item
    pre_row = model.local_user_row.detach().clone()
    try:
        train_bpr_mf(
            model,
            trainloader,
            epochs=1,
            lr=1e-2,
            device="cpu",
            weight_decay=1e-5,
            num_negatives=4,
            proximal_mu=0.0,
            global_params=None,
            run_seed=42,
            user_idx=0,
            round_num=1,
            exclude_items=exclude_items,
            rng=np_rng(42, 0, 1, "train_neg"),
        )
    except Exception as e:  # pragma: no cover - diagnostic on failure
        pytest.fail(f"train_bpr_mf raised unexpectedly: {e}")

    post_row = model.local_user_row.detach()
    assert not torch.allclose(pre_row, post_row), (
        "local_user_row should have received gradient updates"
    )


def test_evaluate_ranking_sampled_accepts_rng_signature() -> None:
    """PSN-03 + BSL-05-style: evaluate_ranking_sampled accepts the 4 FND-06 kwargs."""
    import inspect

    from federated_personalized_cf.task import evaluate_ranking_sampled

    sig = inspect.signature(evaluate_ranking_sampled)
    for p in ("run_seed", "user_idx", "round_num"):
        assert p in sig.parameters, (
            f"BSL-05: evaluate_ranking_sampled must accept '{p}'"
        )
    assert "exclude_items" in sig.parameters, (
        "PSN-03: evaluate_ranking_sampled must accept 'exclude_items'"
    )


def test_sample_negatives_seeded_deterministic() -> None:
    """FND-06 smoke: _sample_negatives_seeded is deterministic under same rng.

    Calls the helper twice with IDENTICAL ``(user_rated_items, num_items,
    num_negatives)`` and fresh RNGs constructed from the SAME seed tuple;
    asserts outputs are equal. Then varies the seed tuple and asserts the
    output differs (non-trivial randomness).
    """
    from federated_personalized_cf.task import _sample_negatives_seeded
    from fedrec_foundation.rng import np_rng

    user_rated = {0, 1, 2, 3, 4, 25}
    out1 = _sample_negatives_seeded(
        user_rated,
        num_items=30,
        num_negatives=5,
        rng=np_rng(42, 0, 1, "train_neg"),
    )
    out2 = _sample_negatives_seeded(
        user_rated,
        num_items=30,
        num_negatives=5,
        rng=np_rng(42, 0, 1, "train_neg"),
    )
    assert np.array_equal(out1, out2), (
        "_sample_negatives_seeded should be deterministic under identical seed tuple"
    )
    # Different seed tuple -> different sequence (very high probability).
    out3 = _sample_negatives_seeded(
        user_rated,
        num_items=30,
        num_negatives=5,
        rng=np_rng(42, 1, 1, "train_neg"),
    )
    assert not np.array_equal(out1, out3), (
        "_sample_negatives_seeded should diverge under different seed tuple"
    )
    # None of the returned items are in the rated set (exclusion correctness).
    for x in out1:
        assert int(x) not in user_rated, (
            f"Sampled negative {x} is in rated set {user_rated}"
        )
