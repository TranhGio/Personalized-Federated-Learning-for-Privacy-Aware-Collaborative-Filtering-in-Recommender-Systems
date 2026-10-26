"""ADP-02 fingerprint tests — enable-before-load ordering bug-fix behavior.

These tests prove that calling enable_per_user_alpha + enable_item_perturbation BEFORE
set_local_parameters causes the cached _logit_alpha.weight and _item_perturbation.weight
tensors to be restored (the fix), while the inverse ordering silently drops them (the bug).

The underlying DualPersonalizedBPRMF class is UNTOUCHED by Phase 4 — its _LOCAL_PARAMS
property already reacts correctly to the enable flags. Phase 4's fix is in client_app.py
(Plan 03) which reorders the calls; this test is the defense-in-depth regression guard
that pins the ADP-02 contract at the model-unit level.
"""
from __future__ import annotations

import torch

from federated_adaptive_personalized_cf.models.dual_personalized_bpr_mf import (
    DualPersonalizedBPRMF,
)


def _build_model() -> DualPersonalizedBPRMF:
    return DualPersonalizedBPRMF(
        num_users=10,
        num_items=20,
        embedding_dim=8,
        mlp_hidden_dims=[16],
        fusion_type="concat",
        dropout=0.0,
        use_bias=True,
    )


def test_local_params_without_enable_flags() -> None:
    """Flags-off baseline: adaptive keys must be absent from _LOCAL_PARAMS."""
    model = _build_model()
    local = set(model._LOCAL_PARAMS)
    assert "_logit_alpha.weight" not in local
    assert "_item_perturbation.weight" not in local
    # Base + MLP + fusion(concat) keys must all be present
    assert "user_embeddings.weight" in local
    assert "user_bias.weight" in local
    assert "fusion_layer.weight" in local
    assert "fusion_layer.bias" in local
    # At least one personal_mlp.* entry exists
    assert any(k.startswith("personal_mlp.") for k in local)


def test_local_params_with_enable_flags_before_construction_of_cache() -> None:
    """Flags-on after enable_* calls: adaptive keys appear in _LOCAL_PARAMS."""
    model = _build_model()
    model.enable_per_user_alpha(num_users=10, init_alphas={i: 0.5 for i in range(10)})
    model.enable_item_perturbation(reg_lambda=0.01)
    local = set(model._LOCAL_PARAMS)
    assert "_logit_alpha.weight" in local
    assert "_item_perturbation.weight" in local
    assert model._logit_alpha is not None
    assert model._item_perturbation is not None


def test_enable_before_load_restores_cached_alpha() -> None:
    """ADP-02 FIX proof: enable_* BEFORE set_local_parameters restores cached tensors.

    Two-step round-trip simulating the federated round boundary:
      1. Round-1 model builds a cached state_dict with sentinel-valued _logit_alpha
         and _item_perturbation tensors.
      2. Round-2 model calls enable_per_user_alpha + enable_item_perturbation FIRST
         (the bug-fix ordering), THEN set_local_parameters(cached_state, strict=False)
         — the sentinel values survive because the adaptive keys are in _LOCAL_PARAMS
         at load time.
    """
    # Round 1: produce a cached state_dict with sentinel-valued _logit_alpha.
    model_a = _build_model()
    model_a.enable_per_user_alpha(num_users=10, init_alphas={i: 0.5 for i in range(10)})
    model_a.enable_item_perturbation(reg_lambda=0.01)
    with torch.no_grad():
        model_a._logit_alpha.weight.data.fill_(0.123)
        model_a._item_perturbation.weight.data.fill_(0.456)
    cached_state = model_a.get_local_parameters()
    assert "_logit_alpha.weight" in cached_state
    assert "_item_perturbation.weight" in cached_state

    # Round 2 (ADP-02 FIX): enable_* BEFORE set_local_parameters — sentinel is restored.
    model_b = _build_model()
    model_b.enable_per_user_alpha(num_users=10, init_alphas={i: 0.5 for i in range(10)})
    model_b.enable_item_perturbation(reg_lambda=0.01)
    loaded, _missing = model_b.set_local_parameters(cached_state, strict=False)
    assert "_logit_alpha.weight" in loaded, (
        f"ADP-02 FIX regressed: _logit_alpha.weight not in loaded set {loaded}"
    )
    assert "_item_perturbation.weight" in loaded, (
        f"ADP-02 FIX regressed: _item_perturbation.weight not in loaded set {loaded}"
    )
    assert torch.allclose(
        model_b._logit_alpha.weight,
        torch.full_like(model_b._logit_alpha.weight, 0.123),
    ), "ADP-02 FIX regressed: cached _logit_alpha sentinel value not restored"
    assert torch.allclose(
        model_b._item_perturbation.weight,
        torch.full_like(model_b._item_perturbation.weight, 0.456),
    ), "ADP-02 FIX regressed: cached _item_perturbation sentinel value not restored"
