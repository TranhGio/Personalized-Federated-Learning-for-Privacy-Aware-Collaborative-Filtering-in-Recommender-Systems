"""Phase 5 Plan 01 PFedRecMLP tests.

Pins:
- D-01: ``_GLOBAL_PARAMS = ('embedding_item.weight', 'affine_output.bias')`` and
  ``_LOCAL_PARAMS = ('affine_output.weight',)`` (bias-GLOBAL flip).
- D-19: First-round init = PyTorch nn.Linear default (Kaiming-uniform); no
  Xavier reset (paper-faithful per ``IJCAI-23-PFedRec/mlp.py`` defaults).
- D-20: Persisted ``affine_output.weight`` shape is native PyTorch
  ``(1, latent_dim)`` — no shape collapse.
- D-21: ``set_local_parameters(strict=True)`` is the new default; raises
  ``RuntimeError`` on shape mismatch with per-field delta + literal
  ``rm -rf .embedding_cache/{run_id}/`` hint.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# Test 1: D-01 + D-20 param tuples and native shape.
# ---------------------------------------------------------------------------


def test_local_params_tuple_only_affine_weight() -> None:
    """``_LOCAL_PARAMS`` is exactly ``('affine_output.weight',)`` (D-01)."""
    from federated_pfedrec.models.pfedrec_mlp import PFedRecMLP

    assert PFedRecMLP._GLOBAL_PARAMS == (
        "embedding_item.weight",
        "affine_output.bias",
    ), (
        "D-01 source-of-truth (engine.py:143): bias-GLOBAL — "
        "_GLOBAL_PARAMS must hold both 'embedding_item.weight' and "
        "'affine_output.bias'."
    )
    assert PFedRecMLP._LOCAL_PARAMS == ("affine_output.weight",), (
        "D-01: only 'affine_output.weight' is per-user LOCAL."
    )
    assert "affine_output.bias" not in PFedRecMLP._LOCAL_PARAMS

    # D-20: native PyTorch shape (1, latent_dim) preserved end-to-end.
    model = PFedRecMLP(num_items=100, latent_dim=32)
    local_state = model.get_local_parameters()
    assert tuple(local_state.keys()) == ("affine_output.weight",)
    assert tuple(local_state["affine_output.weight"].shape) == (1, 32)


# ---------------------------------------------------------------------------
# Test 2: D-21 strict=True default — hard-fails with rm -rf hint.
# ---------------------------------------------------------------------------


def test_set_local_parameters_strict_true_hard_fails() -> None:
    """``set_local_parameters`` with default strict=True raises on shape mismatch.

    The ``RuntimeError`` message MUST surface (a) the offending param name,
    (b) the saved shape, (c) the current model shape, and (d) the literal
    ``rm -rf`` hint that lets the operator clear the bogus cache.
    """
    from federated_pfedrec.models.pfedrec_mlp import PFedRecMLP

    model = PFedRecMLP(num_items=100, latent_dim=32)

    # Shape (1, 16) vs model's (1, 32) — bogus latent_dim.
    bad_state = OrderedDict(
        {"affine_output.weight": torch.randn(1, 16)}
    )

    with pytest.raises(RuntimeError) as excinfo:
        model.set_local_parameters(bad_state)  # default strict=True

    msg = str(excinfo.value)
    assert "affine_output.weight" in msg, "RuntimeError must name the offending key"
    assert "(1, 16)" in msg, "RuntimeError must surface the saved shape"
    assert "(1, 32)" in msg, "RuntimeError must surface the current shape"
    assert "rm -rf" in msg, (
        "D-21 mandates a literal 'rm -rf .embedding_cache/{run_id}/' hint."
    )


# ---------------------------------------------------------------------------
# Test 3: D-19 — Kaiming default init (no Xavier reset).
# ---------------------------------------------------------------------------


def test_kaiming_default_init_paper_faithful() -> None:
    """``affine_output.weight`` uses PyTorch's nn.Linear default (Kaiming-uniform).

    D-19: paper faithfulness wins — ``IJCAI-23-PFedRec/mlp.py`` does not
    apply Xavier; PFedRec is sensitive to init scale (RecSys 2024). Cross-
    module Xavier resets in BPRMF / BasicMF / DualPersonalizedBPRMF are
    intentionally NOT mirrored here.
    """
    from federated_pfedrec.models import pfedrec_mlp as pfedrec_mlp_mod
    from federated_pfedrec.models.pfedrec_mlp import PFedRecMLP

    model = PFedRecMLP(num_items=100, latent_dim=32)
    weight = model.affine_output.weight

    # Non-zero (some init was applied — Kaiming-uniform default).
    assert weight.abs().sum().item() > 0.0, "Init should not produce all-zero"

    # Loose Kaiming-range guard — PyTorch's nn.Linear default produces
    # std ≈ sqrt(1/in_features) ≈ 1/sqrt(32) ≈ 0.177. Xavier-uniform would
    # produce std ≈ sqrt(2/(in+out)) ≈ 0.247 (close enough that we use a
    # generous band rather than a tight match), but the FORBIDDEN-TOKEN
    # source check below is the hard guarantee.
    assert 0.0 < weight.std().item() < 1.0, (
        "Init magnitude outside Kaiming sanity range — verify nn.Linear "
        "default behavior is preserved."
    )

    # Hard guarantee — no Xavier reset has been added to the module source.
    src_path = Path(pfedrec_mlp_mod.__file__)
    src = src_path.read_text(encoding="utf-8")
    assert not re.search(r"xavier_uniform_\s*\(", src), (
        "D-19 forbids Xavier init reset on PFedRec model. Found "
        "'xavier_uniform_(' in pfedrec_mlp.py — remove it."
    )
    assert not re.search(r"xavier_normal_\s*\(", src), (
        "D-19 forbids Xavier init reset on PFedRec model. Found "
        "'xavier_normal_(' in pfedrec_mlp.py — remove it."
    )
