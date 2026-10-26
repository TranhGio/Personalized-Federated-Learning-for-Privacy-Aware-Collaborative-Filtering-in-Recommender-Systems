"""ADP-07 unit tests: alpha factory clip-bounds + hierarchical conditional rule coverage.

All tests run against the UNMODIFIED federated_adaptive_personalized_cf.models.adaptive_alpha
module (Phase 4 requires no production code changes here -- the existing np.clip inside
compute_from_stats at adaptive_alpha.py lines 208/306/339/486 already enforces the contract).

Surface pinned by this file:
    1. DataQuantityAlpha sigmoid endpoints clip correctly at floor (0.1) and ceiling (0.95),
       with midpoint sanity at n==quantity_threshold.
    2. Each of HierarchicalConditionalAlpha's 4 conditional rule branches fires on its
       designed trigger input and appears in the applied_rules list returned by compute_factors:
       (a) sparse-penalty, (b) niche-bonus, (c) inconsistent-penalty, (d) completionist-bonus.
    3. HierarchicalConditionalAlpha clip bounds hold across 6 adversarial (n, ge, nu, rs) inputs.
    4. MultiFactorAlpha clip bounds hold across 2 adversarial inputs.
    5. Factory dispatch is a closed-enum whitelist: unknown method string raises ValueError
       (enforced in AlphaConfig.__post_init__ at line ~85; matches CONVENTIONS.md factory rule
       used by create_alpha_computer and get_model elsewhere).

Reference: .planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-04-PLAN.md (ADP-07)
"""
from __future__ import annotations

import pytest

from federated_adaptive_personalized_cf.models.adaptive_alpha import (
    AlphaConfig,
    HierarchicalConditionalAlphaConfig,
    create_alpha_computer,
    DataQuantityAlpha,
    MultiFactorAlpha,
    HierarchicalConditionalAlpha,
)


# =============================================================================
# DataQuantityAlpha -- endpoint clips + midpoint sanity
# =============================================================================
def test_data_quantity_min_clip_at_very_sparse():
    """n=0 and n=50 both drive raw sigmoid below min_alpha=0.1 -> must be clipped to 0.1."""
    config = AlphaConfig(
        method="data_quantity",
        min_alpha=0.1,
        max_alpha=0.95,
        quantity_threshold=100,
        quantity_temperature=0.05,
    )
    computer = DataQuantityAlpha(config)
    # sigmoid((0 - 100) * 0.05) = sigmoid(-5) ~= 0.0067 -> clipped to 0.1
    assert computer.compute(0) == pytest.approx(0.1, abs=1e-6)
    # sigmoid((50 - 100) * 0.05) = sigmoid(-2.5) ~= 0.0759 -> still clipped to 0.1
    assert computer.compute(50) == pytest.approx(0.1, abs=1e-6)


def test_data_quantity_max_clip_at_dense():
    """n=200 drives raw sigmoid above max_alpha=0.95 -> must be clipped to 0.95."""
    config = AlphaConfig(
        method="data_quantity",
        min_alpha=0.1,
        max_alpha=0.95,
        quantity_threshold=100,
        quantity_temperature=0.05,
    )
    computer = DataQuantityAlpha(config)
    # sigmoid((200 - 100) * 0.05) = sigmoid(5) ~= 0.9933 -> clipped to 0.95
    assert computer.compute(200) == pytest.approx(0.95, abs=1e-6)


def test_data_quantity_midpoint():
    """n=100 (== quantity_threshold) -> sigmoid(0) == 0.5 (inside clip range, not clipped)."""
    config = AlphaConfig(
        method="data_quantity",
        min_alpha=0.1,
        max_alpha=0.95,
        quantity_threshold=100,
        quantity_temperature=0.05,
    )
    computer = DataQuantityAlpha(config)
    assert computer.compute(100) == pytest.approx(0.5, abs=1e-3)


def test_data_quantity_compute_from_stats_equals_compute():
    """compute_from_stats({'n_interactions': N, ...}) must delegate to compute(N) exactly."""
    config = AlphaConfig(
        method="data_quantity",
        min_alpha=0.1,
        max_alpha=0.95,
        quantity_threshold=100,
        quantity_temperature=0.05,
    )
    computer = DataQuantityAlpha(config)
    from_stats = computer.compute_from_stats({
        "n_interactions": 100,
        "genre_entropy": 1.5,
        "n_unique_items": 50,
        "rating_std": 0.75,
    })
    direct = computer.compute(100)
    assert from_stats == pytest.approx(direct, abs=1e-6)


# =============================================================================
# HierarchicalConditionalAlpha -- each conditional rule fires on designed input
# =============================================================================
def test_hc_sparse_penalty_applies():
    """n=5 < sparse_threshold=20 -> 'sparse' appears in applied_rules; alpha in [0.1, 0.95]."""
    config = HierarchicalConditionalAlphaConfig(
        sparse_threshold=20,
        sparse_penalty_max=0.5,
    )
    computer = HierarchicalConditionalAlpha(config)
    factors = computer.compute_factors({
        "n_interactions": 5,
        "genre_entropy": 1.5,
        "n_unique_items": 5,
        "rating_std": 0.75,
    })
    assert "sparse" in factors["applied_rules"], (
        f"Expected 'sparse' in applied_rules for n=5 (< sparse_threshold=20). "
        f"Got: {factors['applied_rules']}"
    )
    assert 0.1 <= factors["alpha"] <= 0.95


def test_hc_niche_bonus_applies():
    """Low diversity + high quantity -> 'niche' appears in applied_rules; alpha in [0.1, 0.95].

    High quantity: n=200 -> f_quantity = sigmoid(5) ~= 0.993 > 0.6 (niche_quantity_threshold).
    Low diversity: genre_entropy=0.5 -> f_diversity = 0.5 / 3.0 ~= 0.167 < 0.25
    (niche_diversity_threshold).
    """
    config = HierarchicalConditionalAlphaConfig(
        niche_diversity_threshold=0.25,
        niche_quantity_threshold=0.6,
        niche_bonus=0.15,
        max_entropy=3.0,
    )
    computer = HierarchicalConditionalAlpha(config)
    factors = computer.compute_factors({
        "n_interactions": 200,
        "genre_entropy": 0.5,
        "n_unique_items": 200,
        "rating_std": 0.75,
    })
    assert "niche" in factors["applied_rules"], (
        f"Expected 'niche' in applied_rules for n=200 + low diversity. "
        f"Got: {factors['applied_rules']}"
    )
    assert 0.1 <= factors["alpha"] <= 0.95


def test_hc_inconsistent_penalty_applies():
    """High rating_std -> 'inconsistent' appears in applied_rules; alpha in [0.1, 0.95].

    rating_std=1.45 -> f_consistency = 1 - min(1.45/1.5, 1.0) ~= 0.033 < 0.3
    (inconsistent_threshold).
    """
    config = HierarchicalConditionalAlphaConfig(
        inconsistent_threshold=0.3,
        inconsistent_penalty=0.3,
        max_rating_std=1.5,
    )
    computer = HierarchicalConditionalAlpha(config)
    factors = computer.compute_factors({
        "n_interactions": 100,
        "genre_entropy": 2.0,
        "n_unique_items": 100,
        "rating_std": 1.45,
    })
    assert "inconsistent" in factors["applied_rules"], (
        f"Expected 'inconsistent' in applied_rules for rating_std=1.45. "
        f"Got: {factors['applied_rules']}"
    )
    assert 0.1 <= factors["alpha"] <= 0.95


def test_hc_completionist_bonus_applies():
    """High coverage + low diversity -> 'completionist' appears in applied_rules; alpha in [0.1, 0.95].

    n_unique=90 -> f_coverage = 90/100 = 0.9 > 0.7 (completionist_coverage).
    genre_entropy=0.5 -> f_diversity = 0.5/3.0 ~= 0.167 < 0.3 (completionist_diversity).
    """
    config = HierarchicalConditionalAlphaConfig(
        completionist_coverage=0.7,
        completionist_diversity=0.3,
        completionist_bonus=0.1,
        coverage_threshold=100,
        max_entropy=3.0,
    )
    computer = HierarchicalConditionalAlpha(config)
    factors = computer.compute_factors({
        "n_interactions": 90,
        "genre_entropy": 0.5,
        "n_unique_items": 90,
        "rating_std": 0.75,
    })
    assert "completionist" in factors["applied_rules"], (
        f"Expected 'completionist' in applied_rules for coverage=0.9 + low diversity. "
        f"Got: {factors['applied_rules']}"
    )
    assert 0.1 <= factors["alpha"] <= 0.95


@pytest.mark.parametrize("n,ge,nu,rs", [
    (0, 0.0, 0, 0.0),
    (0, 3.0, 0, 1.5),
    (10000, 0.0, 10000, 0.0),
    (10000, 3.0, 10000, 1.5),
    (5, 1.5, 5, 0.75),
    (1000, 2.0, 100, 1.0),
])
def test_hc_min_max_clip_bounds(n, ge, nu, rs):
    """Every HC input must produce alpha in [min_alpha, max_alpha] regardless of extremity.

    Covers adversarial corners: all-zero, all-max, midpoint, etc. Guards against
    a future regression that removes the final np.clip(..., min_alpha, max_alpha).
    """
    config = HierarchicalConditionalAlphaConfig(min_alpha=0.1, max_alpha=0.95)
    computer = HierarchicalConditionalAlpha(config)
    alpha = computer.compute_from_stats({
        "n_interactions": n,
        "genre_entropy": ge,
        "n_unique_items": nu,
        "rating_std": rs,
    })
    assert 0.1 <= alpha <= 0.95, (
        f"HC alpha {alpha} out of [0.1, 0.95] bounds for n={n}, ge={ge}, nu={nu}, rs={rs}"
    )


# =============================================================================
# MultiFactorAlpha -- clip bounds on adversarial extremes
# =============================================================================
@pytest.mark.parametrize("n,ge,nu,rs", [
    (0, 0.0, 0, 0.0),
    (10000, 3.0, 10000, 0.0),
])
def test_multi_factor_clip_bounds(n, ge, nu, rs):
    """MultiFactorAlpha output must stay in [min_alpha, max_alpha] on adversarial inputs."""
    config = AlphaConfig(
        method="multi_factor",
        min_alpha=0.1,
        max_alpha=0.95,
        factor_weights={
            "quantity": 0.4,
            "diversity": 0.25,
            "coverage": 0.2,
            "consistency": 0.15,
        },
    )
    computer = MultiFactorAlpha(config)
    alpha = computer.compute_from_stats({
        "n_interactions": n,
        "genre_entropy": ge,
        "n_unique_items": nu,
        "rating_std": rs,
    })
    assert 0.1 <= alpha <= 0.95


# =============================================================================
# Factory dispatch -- closed-enum whitelist
# =============================================================================
def test_factory_returns_correct_computer_class():
    """create_alpha_computer dispatches on config.method to the right class."""
    assert isinstance(
        create_alpha_computer(AlphaConfig(method="data_quantity")),
        DataQuantityAlpha,
    )
    assert isinstance(
        create_alpha_computer(AlphaConfig(method="multi_factor")),
        MultiFactorAlpha,
    )
    assert isinstance(
        create_alpha_computer(
            AlphaConfig(method="hierarchical_conditional"),
            hc_config=HierarchicalConditionalAlphaConfig(),
        ),
        HierarchicalConditionalAlpha,
    )


def test_factory_unknown_method_raises():
    """Unknown method strings must fail loud at AlphaConfig construction time.

    AlphaConfig.__post_init__ at adaptive_alpha.py:85 enforces a method whitelist
    ({'data_quantity', 'multi_factor', 'hierarchical_conditional'}). Matches the
    CONVENTIONS.md closed-enum factory rule used across the codebase.
    """
    with pytest.raises(ValueError, match="method|Unknown"):
        AlphaConfig(method="invalid_method")
