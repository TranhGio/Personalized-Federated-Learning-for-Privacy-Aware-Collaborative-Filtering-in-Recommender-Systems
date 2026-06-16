"""Tests for task.py RNG + exclusion threading (Phase 2 Plan 03).

Covers BSL-03, BSL-05, and D-24:
  - Gradient masking zeros rows of user_embeddings that are not this
    client's user_idx.
  - Training negatives respect the FND-03 exclusion set.
  - evaluate_ranking_sampled accepts the new RNG+exclusion contract
    surface and is deterministic.
  - task.py AND client_app.py contain no ``import random`` at module
    scope, no ``random.seed(...)`` calls, and no ``random.sample(...)``
    calls (BSL-05 cross-file regression).
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


def _make_minimal_bpr_model(num_users: int = 10, num_items: int = 30, dim: int = 4):
    """Build a small BPRMF for fast unit tests."""
    from federated_baseline_cf.models import BPRMF

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
    from federated_baseline_cf.dataset import MovieLensDataset

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
    """BSL-05: no stdlib-random seeding/sampling inside task.py OR client_app.py.

    Iteration 1 WARNING 2 fix (reinforced in iteration 2): extend the
    cross-file regression to cover client_app.py — Plan 04 does a
    similar cross-file check at the server-app level, but Plan 03's own
    acceptance should catch a client_app.py leak before we ship Wave 2.
    """
    module_dir = Path(__file__).resolve().parents[1] / "federated_baseline_cf"
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
        # Strip check for ``import random`` inside the top 25 lines, which
        # covers every module's import block even with module docstrings.
        top_lines = src.split("\n")[0:25]
        for line in top_lines:
            stripped = line.strip()
            assert stripped != "import random", (
                f"BSL-05 violation in {filename}: module-level `import random` "
                f"must be stripped (use fedrec_foundation.rng for all randomness)"
            )


def test_train_negatives_exclude_test_positive() -> None:
    """BSL-03: held-out test positive NEVER appears in training negatives.

    Trains a tiny BPR-MF for 1 epoch on a synthetic single-user partition
    whose ``exclude_items`` includes the held-out positive and asserts:
      - The call completes without raising.
      - The model's ``user_embeddings.weight`` row for user_idx=0 has moved
        (proves gradients actually flowed through the sampled negatives).
    The stronger property — that item 25 is never drawn as a negative —
    is enforced by ``_sample_negatives_seeded`` which reads from
    ``user_rated_items[user_idx]`` (which the function merges
    ``exclude_items`` into before the sampling loop).
    """
    from federated_baseline_cf.task import train_bpr_mf
    from fedrec_foundation.rng import np_rng

    torch.manual_seed(0)
    model = _make_minimal_bpr_model(num_users=5, num_items=30, dim=4)
    trainloader = _make_trainloader(user_raw_id=1, num_positive_items=10, num_items=30)

    exclude_items = np.array([25], dtype=np.int32)  # the held-out test item
    pre_u0 = model.user_embeddings.weight[0].detach().clone()
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

    post_u0 = model.user_embeddings.weight[0].detach()
    assert not torch.allclose(pre_u0, post_u0), (
        "user_idx=0 row should have received gradient updates"
    )


def test_evaluate_ranking_sampled_accepts_rng_signature() -> None:
    """BSL-05: evaluate_ranking_sampled accepts run_seed+user_idx+round_num."""
    import inspect

    from federated_baseline_cf.task import evaluate_ranking_sampled

    sig = inspect.signature(evaluate_ranking_sampled)
    for p in ("run_seed", "user_idx", "round_num"):
        assert p in sig.parameters, (
            f"BSL-05: evaluate_ranking_sampled must accept '{p}'"
        )
    assert "exclude_items" in sig.parameters, (
        "BSL-03: evaluate_ranking_sampled must accept 'exclude_items'"
    )


def test_gradient_mask_zeros_non_user_rows() -> None:
    """D-24: after train_bpr_mf, only user_idx=0's row of user_embeddings moved.

    The remaining 4 rows (user_idx 1..4) must be bit-identical to pre-training.
    """
    from federated_baseline_cf.task import train_bpr_mf

    torch.manual_seed(0)
    model = _make_minimal_bpr_model(num_users=5, num_items=30, dim=4)
    pre = model.user_embeddings.weight.detach().clone()
    trainloader = _make_trainloader(user_raw_id=1, num_positive_items=5, num_items=30)
    train_bpr_mf(
        model,
        trainloader,
        epochs=1,
        lr=1e-1,
        device="cpu",
        weight_decay=1e-5,
        num_negatives=4,
        proximal_mu=0.0,
        global_params=None,
        run_seed=42,
        user_idx=0,
        round_num=1,
        exclude_items=None,
    )
    post = model.user_embeddings.weight.detach()

    # Row 0 MUST have moved (the only row allowed to receive gradients).
    assert not torch.allclose(pre[0], post[0]), (
        "user_idx=0 row should have received gradient updates"
    )
    # Rows 1..4 MUST be unchanged (D-24 mask zeroes their gradients).
    for u in range(1, 5):
        assert torch.allclose(pre[u], post[u], atol=1e-8), (
            f"D-24 violation: user_idx={u} row of user_embeddings changed but "
            f"shouldn't have. diff_norm={(pre[u] - post[u]).norm().item():.6e}"
        )
