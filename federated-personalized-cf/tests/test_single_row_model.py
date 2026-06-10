"""Unit tests for BPRMF + BasicMF single-row refactor (Phase 3 Plan 01, D-01, D-03, PSN-06).

Asserts the D-01 ghost-table removal: nn.Embedding(num_users, d) is replaced by
local_user_row nn.Parameter (shape=(d,)) + local_user_bias nn.Parameter (shape=(1,)).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import pytest

from federated_personalized_cf.models.bpr_mf import BPRMF
from federated_personalized_cf.models.basic_mf import BasicMF


def test_bpr_mf_single_row_shape() -> None:
    """BPRMF must have local_user_row (embedding_dim,) and local_user_bias (1,) — no ghost table."""
    model = BPRMF(num_users=6040, num_items=3706, embedding_dim=64, use_bias=True)
    assert model.local_user_row.shape == torch.Size([64])
    assert model.local_user_bias.shape in (torch.Size([]), torch.Size([1]))
    assert not hasattr(model, "user_embeddings"), (
        "D-01 violated: nn.Embedding(num_users, d) ghost table still present as attribute"
    )
    # Old user_bias attribute must be gone (replaced by local_user_bias)
    user_bias_attr = getattr(model, "user_bias", None)
    assert user_bias_attr is None or not isinstance(user_bias_attr, nn.Embedding), (
        "D-01 violated: old user_bias nn.Embedding(num_users, 1) still present"
    )


def test_bpr_mf_local_params_contract() -> None:
    """get_local_parameters() must return exactly {local_user_row, local_user_bias}."""
    model = BPRMF(num_users=6040, num_items=3706, embedding_dim=64, use_bias=True)
    local = model.get_local_parameters()
    assert set(local.keys()) == {"local_user_row", "local_user_bias"}
    assert local["local_user_row"].shape == torch.Size([64])
    assert local["local_user_bias"].shape in (torch.Size([]), torch.Size([1]))


def test_bpr_mf_global_params_contract() -> None:
    """get_global_parameters() returns full 3-key set when use_bias=True, 1-key when False."""
    model_with_bias = BPRMF(num_users=6040, num_items=3706, embedding_dim=64, use_bias=True)
    assert set(model_with_bias.get_global_parameters().keys()) == {
        "item_embeddings.weight", "item_bias.weight", "global_bias",
    }
    model_no_bias = BPRMF(num_users=6040, num_items=3706, embedding_dim=64, use_bias=False)
    assert set(model_no_bias.get_global_parameters().keys()) == {"item_embeddings.weight"}


def test_bpr_mf_no_ghost_table() -> None:
    """D-01: source file must not contain nn.Embedding(num_users, ...) or self.user_embeddings."""
    src_path = Path(__file__).resolve().parents[1] / "federated_personalized_cf" / "models" / "bpr_mf.py"
    src = src_path.read_text()
    assert "nn.Embedding(num_users" not in src, "D-01 violated: found ghost user table in bpr_mf.py"
    assert "self.user_embeddings" not in src, "D-01 violated: attribute user_embeddings still present"


def test_basic_mf_single_row_shape() -> None:
    """BasicMF mirrors BPRMF's single-row contract."""
    model = BasicMF(num_users=6040, num_items=3706, embedding_dim=64)
    assert model.local_user_row.shape == torch.Size([64])
    assert model.local_user_bias.shape in (torch.Size([]), torch.Size([1]))
    assert not hasattr(model, "user_embeddings"), (
        "D-01 violated: nn.Embedding(num_users, d) ghost table still present in BasicMF"
    )


def test_basic_mf_no_ghost_table() -> None:
    """D-01: basic_mf.py source must not contain the ghost table either."""
    src_path = Path(__file__).resolve().parents[1] / "federated_personalized_cf" / "models" / "basic_mf.py"
    src = src_path.read_text()
    assert "nn.Embedding(num_users" not in src, "D-01 violated: found ghost user table in basic_mf.py"
    assert "self.user_embeddings" not in src, "D-01 violated: user_embeddings still in basic_mf.py"


def test_set_local_parameters_single_row_roundtrip() -> None:
    """save -> modify -> restore roundtrip preserves local_user_row values."""
    model = BPRMF(num_users=6040, num_items=3706, embedding_dim=64, use_bias=True)
    saved = model.get_local_parameters()
    saved_row = saved["local_user_row"].clone()

    # Mutate in-place on the live model
    with torch.no_grad():
        model.local_user_row.fill_(0.0)
    assert torch.all(model.local_user_row == 0.0)

    loaded, missing = model.set_local_parameters(saved, strict=False)
    assert "local_user_row" in loaded
    assert "local_user_bias" in loaded
    assert missing == []
    assert torch.allclose(model.local_user_row, saved_row)
