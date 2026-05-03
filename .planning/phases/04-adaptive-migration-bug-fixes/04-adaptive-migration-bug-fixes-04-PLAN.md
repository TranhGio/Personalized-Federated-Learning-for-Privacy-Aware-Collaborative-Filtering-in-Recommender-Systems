---
phase: 04-adaptive-migration-bug-fixes
plan: 04
type: execute
subsystem: infra
tags: [alpha-factory, hierarchical-conditional, multi-factor, data-quantity, clip-bounds, edge-case-inputs, adp-07, d-16-diagnostic-rule-coverage, tdd, wave-2]
wave: 2
depends_on: [04-adaptive-migration-bug-fixes-02]
files_modified:
  - federated-adaptive-personalized-cf/tests/test_alpha_factory.py
autonomous: true
requirements: [ADP-07]

must_haves:
  truths:
    - "Every branch of create_alpha_computer (DataQuantityAlpha, MultiFactorAlpha, HierarchicalConditionalAlpha) returns alpha values strictly in [0.1, 0.95] for a crafted grid of edge-case user-stats inputs covering the adversarial extremes: (n=0, ge=0, nu=0, rs=0), (n=10000, ge=3.0, nu=10000, rs=1.5), (n=5, ge=1.5, nu=5, rs=0.75), (n=1000, ge=2.0, nu=100, rs=1.0)."
    - "Each of the 4 HierarchicalConditionalAlpha conditional rule branches fires on its designed trigger input and appears in the `applied_rules` list returned by compute_factors: (a) sparse-penalty at n < sparse_threshold=20, (b) niche-bonus at low diversity + high quantity, (c) inconsistent-penalty at low consistency (high rating_std), (d) completionist-bonus at high coverage + low diversity."
    - "DataQuantityAlpha's sigmoid endpoints clip correctly: n=0 → 0.1 (floor), n=200 → 0.95 (ceiling), n=100 (midpoint) → 0.5 ± eps with default quantity_threshold=100, quantity_temperature=0.05."
    - "MultiFactorAlpha clip-range test covers both floor and ceiling corner cases."
    - "The factory function itself round-trips each method string and raises ValueError on unknown method strings (closed-enum whitelist, CONVENTIONS.md factory pattern)."
    - "test_alpha_factory.py ships 10+ GREEN unit tests against the UNMODIFIED adaptive_alpha.py class (no production-code changes required by this plan — the tests are behavior fingerprints)."
  artifacts:
    - path: "federated-adaptive-personalized-cf/tests/test_alpha_factory.py"
      provides: "10+ GREEN unit tests pinning alpha factory output bounds + hierarchical conditional rule branch coverage for ADP-07"
  key_links:
    - from: "federated-adaptive-personalized-cf/tests/test_alpha_factory.py"
      to: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/adaptive_alpha.py"
      via: "imports AlphaConfig, HierarchicalConditionalAlphaConfig, create_alpha_computer, DataQuantityAlpha, MultiFactorAlpha, HierarchicalConditionalAlpha and exercises compute / compute_factors / compute_from_stats"
      pattern: "create_alpha_computer|HierarchicalConditionalAlpha|DataQuantityAlpha|MultiFactorAlpha"
---

<objective>
Ship the ADP-07 regression surface — 10+ GREEN unit tests against the UNMODIFIED federated_adaptive_personalized_cf/models/adaptive_alpha.py factory. Crafted inputs cover:

1. Clip-range bounds [min_alpha=0.1, max_alpha=0.95] across all 3 alpha classes on adversarial edge-case inputs.
2. DataQuantityAlpha sigmoid endpoint behavior (floor at n=0, ceiling at n=200, midpoint at n=100).
3. Each of HierarchicalConditionalAlpha's 4 conditional rule branches (sparse, niche, inconsistent, completionist) fires on its designed trigger input and appears in applied_rules.
4. Factory function dispatch is a closed-enum whitelist — unknown method string raises ValueError.

Purpose: Closes ADP-07. This plan runs in parallel with Plan 03 in Wave 2 — it does NOT depend on Plans 01 or 03 because adaptive_alpha.py is UNMODIFIED by Phase 4 (the factory already clips to [0.1, 0.95] inside compute_from_stats per CONCERNS.md line refs). Depends only on Plan 02's pytest dev-dep declaration (pyproject.toml) so `pip install -e .[dev]` works.

The plan owns ONE new test file; no production code changes.

Output:
- federated-adaptive-personalized-cf/tests/test_alpha_factory.py (new — 10+ GREEN tests)
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/04-adaptive-migration-bug-fixes/04-CONTEXT.md
@.planning/phases/04-adaptive-migration-bug-fixes/04-RESEARCH.md
@.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-02-PLAN.md

<interfaces>
<!-- federated_adaptive_personalized_cf.models.adaptive_alpha (UNMODIFIED by Phase 4) -->
```python
@dataclass
class AlphaConfig:
    method: str                 # "data_quantity" | "multi_factor" | "hierarchical_conditional"
    min_alpha: float = 0.1
    max_alpha: float = 0.95
    quantity_threshold: int = 100
    quantity_temperature: float = 0.05
    factor_weights: Dict[str, float] = field(default_factory=lambda: {
        "quantity": 0.40, "diversity": 0.25, "coverage": 0.20, "consistency": 0.15,
    })
    max_entropy: float = 3.0
    coverage_threshold: int = 100
    max_rating_std: float = 1.5

    def __post_init__(self):
        # method whitelist; factor_weights sum == 1.0 ± 0.01; etc.
        ...


@dataclass
class HierarchicalConditionalAlphaConfig:
    min_alpha: float = 0.1
    max_alpha: float = 0.95
    data_volume_weight: float = 0.55
    preference_quality_weight: float = 0.45
    quantity_threshold: int = 100
    quantity_temperature: float = 0.05
    max_entropy: float = 3.0
    coverage_threshold: int = 100
    max_rating_std: float = 1.5
    sparse_threshold: int = 20
    sparse_penalty_max: float = 0.5
    niche_diversity_threshold: float = 0.25
    niche_quantity_threshold: float = 0.6
    niche_bonus: float = 0.15
    inconsistent_threshold: float = 0.3
    inconsistent_penalty: float = 0.3
    completionist_coverage: float = 0.7
    completionist_diversity: float = 0.3
    completionist_bonus: float = 0.1


class DataQuantityAlpha:
    def __init__(self, config: AlphaConfig): ...
    def compute(self, n_interactions: int) -> float: ...
    def compute_from_stats(self, user_stats: Dict[str, Any]) -> float: ...


class MultiFactorAlpha:
    def __init__(self, config: AlphaConfig): ...
    def compute_from_stats(self, user_stats: Dict[str, Any]) -> float: ...
    # returns scalar alpha in [min_alpha, max_alpha]


class HierarchicalConditionalAlpha:
    def __init__(self, config: HierarchicalConditionalAlphaConfig): ...
    def compute_factors(self, user_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Returns dict with keys: f_quantity, f_diversity, f_coverage, f_consistency,
        data_volume, preference_quality, base_alpha, applied_rules (List[str]), alpha.
        applied_rules contains "sparse" / "niche" / "inconsistent" / "completionist"
        depending on which conditional rule branches fired.
        """
    def compute_from_stats(self, user_stats: Dict[str, Any]) -> float:
        """Returns final alpha clipped to [min_alpha, max_alpha]."""


AlphaComputer = Union[DataQuantityAlpha, MultiFactorAlpha, HierarchicalConditionalAlpha]


def create_alpha_computer(
    config: AlphaConfig,
    hc_config: Optional[HierarchicalConditionalAlphaConfig] = None,
) -> AlphaComputer:
    """Factory. Dispatches on config.method. Raises ValueError on unknown method (closed enum)."""
```

<!-- The crafted-input grid and test-body skeletons from Research §Code Examples / §Alpha factory unit test (lines ~770-898) -->
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: test_alpha_factory.py — ADP-07 clip-bounds + rule-branch coverage tests (10+ GREEN against unmodified adaptive_alpha.py)</name>
  <files>federated-adaptive-personalized-cf/tests/test_alpha_factory.py</files>
  <read_first>
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/adaptive_alpha.py (ENTIRE FILE — lines 98-170 for HierarchicalConditionalAlphaConfig defaults; lines 208, 306, 339, 486 for the np.clip(..., min_alpha, max_alpha) call sites; lines 380-550 for compute_factors + applied_rules construction; UNCHANGED by Phase 4)
    - .planning/phases/04-adaptive-migration-bug-fixes/04-RESEARCH.md §"Alpha factory unit test covering every conditional rule branch (ADP-07)" (lines ~770-898 — ready-to-paste test skeletons for each rule)
    - .planning/phases/04-adaptive-migration-bug-fixes/04-CONTEXT.md §ADP-07 (requirement text: "alpha values fall in [0.1, 0.95] for edge-case user-stats inputs")
  </read_first>
  <behavior>
    10+ tests (GREEN on first run; no production-code changes needed since the factory already clips):

    1. test_data_quantity_min_clip_at_very_sparse: n=0 → alpha ≈ 0.1 (floor clip); n=50 → alpha ≈ 0.1 (still clipped because sigmoid((50-100)*0.05) ≈ 0.0759 < 0.1).
    2. test_data_quantity_max_clip_at_dense: n=200 → alpha ≈ 0.95 (ceiling clip because sigmoid((200-100)*0.05) ≈ 0.9933 > 0.95).
    3. test_data_quantity_midpoint: n=100 → alpha ≈ 0.5 (sigmoid(0) == 0.5, falls within [0.1, 0.95]).
    4. test_hc_sparse_penalty_applies: n=5 → applied_rules includes "sparse"; alpha in [0.1, 0.95].
    5. test_hc_niche_bonus_applies: n=200, genre_entropy=0.5 → applied_rules includes "niche"; alpha in [0.1, 0.95].
    6. test_hc_inconsistent_penalty_applies: n=100, rating_std=1.45 → applied_rules includes "inconsistent"; alpha in [0.1, 0.95].
    7. test_hc_completionist_bonus_applies: n=90, n_unique_items=90, genre_entropy=0.5 → applied_rules includes "completionist"; alpha in [0.1, 0.95].
    8. test_hc_min_max_clip_bounds: parametrized over 6 adversarial inputs ((0,0,0,0), (0,3,0,1.5), (10000,0,10000,0), (10000,3,10000,1.5), (5,1.5,5,0.75), (1000,2.0,100,1.0)) — every output falls in [0.1, 0.95].
    9. test_multi_factor_clip_bounds: 2 adversarial inputs ((0,0,0,0), (10000,3,10000,0)) — every output falls in [0.1, 0.95].
    10. test_factory_returns_correct_computer_class: create_alpha_computer(AlphaConfig(method="data_quantity")) is DataQuantityAlpha; same for multi_factor → MultiFactorAlpha; hierarchical_conditional → HierarchicalConditionalAlpha (requires hc_config kwarg).
    11. test_factory_unknown_method_raises: AlphaConfig(method="invalid_method") raises ValueError (enforced by AlphaConfig.__post_init__ method whitelist).
    12. test_data_quantity_compute_from_stats_equals_compute: DataQuantityAlpha.compute_from_stats({"n_interactions": 100, ...}) equals .compute(100) for the same integer — cross-check that compute_from_stats delegates.
  </behavior>
  <action>
    Step 1 — Create federated-adaptive-personalized-cf/tests/test_alpha_factory.py using the Research §"Alpha factory unit test" skeleton verbatim, extended to 12 tests:

    ```python
    """ADP-07 unit tests: alpha factory clip-bounds + hierarchical conditional rule coverage.

    All tests run against the UNMODIFIED federated_adaptive_personalized_cf.models.adaptive_alpha
    module (Phase 4 requires no production code changes here — the existing np.clip inside
    compute_from_stats at adaptive_alpha.py lines 208/306/339/486 already enforces the contract).
    """
    import pytest

    from federated_adaptive_personalized_cf.models.adaptive_alpha import (
        AlphaConfig, HierarchicalConditionalAlphaConfig,
        create_alpha_computer,
        DataQuantityAlpha, MultiFactorAlpha, HierarchicalConditionalAlpha,
    )


    # =============================================================================
    # DataQuantityAlpha — endpoint clips + midpoint sanity
    # =============================================================================
    def test_data_quantity_min_clip_at_very_sparse():
        config = AlphaConfig(method="data_quantity", min_alpha=0.1, max_alpha=0.95,
                             quantity_threshold=100, quantity_temperature=0.05)
        computer = DataQuantityAlpha(config)
        # sigmoid((0 - 100) * 0.05) = sigmoid(-5) ≈ 0.0067 → clipped to 0.1
        assert computer.compute(0) == pytest.approx(0.1, abs=1e-6)
        # sigmoid((50 - 100) * 0.05) = sigmoid(-2.5) ≈ 0.0759 → still clipped to 0.1
        assert computer.compute(50) == pytest.approx(0.1, abs=1e-6)


    def test_data_quantity_max_clip_at_dense():
        config = AlphaConfig(method="data_quantity", min_alpha=0.1, max_alpha=0.95,
                             quantity_threshold=100, quantity_temperature=0.05)
        computer = DataQuantityAlpha(config)
        # sigmoid((200 - 100) * 0.05) = sigmoid(5) ≈ 0.9933 → clipped to 0.95
        assert computer.compute(200) == pytest.approx(0.95, abs=1e-6)


    def test_data_quantity_midpoint():
        config = AlphaConfig(method="data_quantity", min_alpha=0.1, max_alpha=0.95,
                             quantity_threshold=100, quantity_temperature=0.05)
        computer = DataQuantityAlpha(config)
        # sigmoid(0) == 0.5 — within [0.1, 0.95] so not clipped
        assert computer.compute(100) == pytest.approx(0.5, abs=1e-3)


    def test_data_quantity_compute_from_stats_equals_compute():
        config = AlphaConfig(method="data_quantity", min_alpha=0.1, max_alpha=0.95,
                             quantity_threshold=100, quantity_temperature=0.05)
        computer = DataQuantityAlpha(config)
        from_stats = computer.compute_from_stats({
            "n_interactions": 100, "genre_entropy": 1.5,
            "n_unique_items": 50, "rating_std": 0.75,
        })
        direct = computer.compute(100)
        assert from_stats == pytest.approx(direct, abs=1e-6)


    # =============================================================================
    # HierarchicalConditionalAlpha — each conditional rule fires on designed input
    # =============================================================================
    def test_hc_sparse_penalty_applies():
        config = HierarchicalConditionalAlphaConfig(sparse_threshold=20, sparse_penalty_max=0.5)
        computer = HierarchicalConditionalAlpha(config)
        factors = computer.compute_factors({"n_interactions": 5, "genre_entropy": 1.5,
                                            "n_unique_items": 5, "rating_std": 0.75})
        assert "sparse" in factors["applied_rules"], (
            f"Expected 'sparse' in applied_rules for n=5 (< sparse_threshold=20). "
            f"Got: {factors['applied_rules']}"
        )
        assert 0.1 <= factors["alpha"] <= 0.95


    def test_hc_niche_bonus_applies():
        config = HierarchicalConditionalAlphaConfig(
            niche_diversity_threshold=0.25,
            niche_quantity_threshold=0.6,
            niche_bonus=0.15, max_entropy=3.0,
        )
        computer = HierarchicalConditionalAlpha(config)
        # High quantity: n=200 → f_quantity ≈ 0.99 > 0.6 (niche_quantity_threshold).
        # Low diversity: genre_entropy=0.5 → f_diversity = 0.5/3.0 ≈ 0.17 < 0.25 (niche_diversity_threshold).
        factors = computer.compute_factors({"n_interactions": 200, "genre_entropy": 0.5,
                                            "n_unique_items": 200, "rating_std": 0.75})
        assert "niche" in factors["applied_rules"], (
            f"Expected 'niche' in applied_rules for n=200 + low diversity. "
            f"Got: {factors['applied_rules']}"
        )
        assert 0.1 <= factors["alpha"] <= 0.95


    def test_hc_inconsistent_penalty_applies():
        config = HierarchicalConditionalAlphaConfig(
            inconsistent_threshold=0.3, inconsistent_penalty=0.3, max_rating_std=1.5,
        )
        computer = HierarchicalConditionalAlpha(config)
        # High rating_std=1.45 → f_consistency = 1 - 1.45/1.5 ≈ 0.033 < 0.3.
        factors = computer.compute_factors({"n_interactions": 100, "genre_entropy": 2.0,
                                            "n_unique_items": 100, "rating_std": 1.45})
        assert "inconsistent" in factors["applied_rules"], (
            f"Expected 'inconsistent' in applied_rules for rating_std=1.45. "
            f"Got: {factors['applied_rules']}"
        )
        assert 0.1 <= factors["alpha"] <= 0.95


    def test_hc_completionist_bonus_applies():
        config = HierarchicalConditionalAlphaConfig(
            completionist_coverage=0.7, completionist_diversity=0.3,
            completionist_bonus=0.1,
            coverage_threshold=100, max_entropy=3.0,
        )
        computer = HierarchicalConditionalAlpha(config)
        # n_unique=90 → f_coverage = 0.9 > 0.7 (completionist_coverage threshold).
        # genre_entropy=0.5 → f_diversity ≈ 0.17 < 0.3 (completionist_diversity threshold).
        factors = computer.compute_factors({"n_interactions": 90, "genre_entropy": 0.5,
                                            "n_unique_items": 90, "rating_std": 0.75})
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
        """Every HC input must produce alpha in [min_alpha, max_alpha] regardless of input extremity."""
        config = HierarchicalConditionalAlphaConfig(min_alpha=0.1, max_alpha=0.95)
        computer = HierarchicalConditionalAlpha(config)
        alpha = computer.compute_from_stats({
            "n_interactions": n, "genre_entropy": ge,
            "n_unique_items": nu, "rating_std": rs,
        })
        assert 0.1 <= alpha <= 0.95, (
            f"HC alpha {alpha} out of [0.1, 0.95] bounds for n={n}, ge={ge}, nu={nu}, rs={rs}"
        )


    # =============================================================================
    # MultiFactorAlpha — clip bounds on adversarial extremes
    # =============================================================================
    @pytest.mark.parametrize("n,ge,nu,rs", [
        (0, 0.0, 0, 0.0),
        (10000, 3.0, 10000, 0.0),
    ])
    def test_multi_factor_clip_bounds(n, ge, nu, rs):
        config = AlphaConfig(
            method="multi_factor", min_alpha=0.1, max_alpha=0.95,
            factor_weights={"quantity": 0.4, "diversity": 0.25,
                            "coverage": 0.2, "consistency": 0.15},
        )
        computer = MultiFactorAlpha(config)
        alpha = computer.compute_from_stats({
            "n_interactions": n, "genre_entropy": ge,
            "n_unique_items": nu, "rating_std": rs,
        })
        assert 0.1 <= alpha <= 0.95


    # =============================================================================
    # Factory dispatch
    # =============================================================================
    def test_factory_returns_correct_computer_class():
        assert isinstance(create_alpha_computer(AlphaConfig(method="data_quantity")),
                          DataQuantityAlpha)
        assert isinstance(create_alpha_computer(AlphaConfig(method="multi_factor")),
                          MultiFactorAlpha)
        assert isinstance(
            create_alpha_computer(
                AlphaConfig(method="hierarchical_conditional"),
                hc_config=HierarchicalConditionalAlphaConfig(),
            ),
            HierarchicalConditionalAlpha,
        )


    def test_factory_unknown_method_raises():
        # AlphaConfig.__post_init__ enforces a method whitelist (CONVENTIONS.md factory rule).
        with pytest.raises(ValueError, match="method|Unknown"):
            AlphaConfig(method="invalid_method")
    ```

    Step 2 — Verify: `cd federated-adaptive-personalized-cf && pytest tests/test_alpha_factory.py -v` → 12+ passed (counting parametrizations: 3 DQ + 1 equality + 4 HC rule + 6 HC clip params + 2 MF clip params + 2 factory = 18 test items). Full suite: `pytest tests/ -v` → 38+ passed (accumulated from Plans 01+02+03 = ~33, plus this plan's ~18).

    Step 3 — Commit (--no-verify; Wave-2 parallel rule; this plan's file set is disjoint from Plan 03's):
    ```
    git add federated-adaptive-personalized-cf/tests/test_alpha_factory.py
    git commit --no-verify -m "test(04-04): ADP-07 alpha factory clip-bounds + HC rule-branch coverage"
    ```
  </action>
  <acceptance_criteria>
    - `test -r federated-adaptive-personalized-cf/tests/test_alpha_factory.py` succeeds
    - `grep -c "^def test_" federated-adaptive-personalized-cf/tests/test_alpha_factory.py` returns at least 12 (12 test functions before pytest.mark.parametrize expansion)
    - `grep -c "test_data_quantity_min_clip_at_very_sparse" federated-adaptive-personalized-cf/tests/test_alpha_factory.py` returns 1
    - `grep -c "test_data_quantity_max_clip_at_dense" federated-adaptive-personalized-cf/tests/test_alpha_factory.py` returns 1
    - `grep -c "test_hc_sparse_penalty_applies" federated-adaptive-personalized-cf/tests/test_alpha_factory.py` returns 1
    - `grep -c "test_hc_niche_bonus_applies" federated-adaptive-personalized-cf/tests/test_alpha_factory.py` returns 1
    - `grep -c "test_hc_inconsistent_penalty_applies" federated-adaptive-personalized-cf/tests/test_alpha_factory.py` returns 1
    - `grep -c "test_hc_completionist_bonus_applies" federated-adaptive-personalized-cf/tests/test_alpha_factory.py` returns 1
    - `grep -c "test_hc_min_max_clip_bounds" federated-adaptive-personalized-cf/tests/test_alpha_factory.py` returns 1
    - `grep -c "test_multi_factor_clip_bounds" federated-adaptive-personalized-cf/tests/test_alpha_factory.py` returns 1
    - `grep -c "test_factory_returns_correct_computer_class" federated-adaptive-personalized-cf/tests/test_alpha_factory.py` returns 1
    - `grep -c "test_factory_unknown_method_raises" federated-adaptive-personalized-cf/tests/test_alpha_factory.py` returns 1
    - `grep -c "applied_rules" federated-adaptive-personalized-cf/tests/test_alpha_factory.py` returns at least 4 (one per HC rule-branch test, asserting the applied_rules list contents)
    - `grep -cE '0\\.1.*<=.*<=.*0\\.95|0\\.1 <= alpha|alpha <= 0\\.95' federated-adaptive-personalized-cf/tests/test_alpha_factory.py` returns at least 5 (clip-range assertions repeated across tests)
    - `cd federated-adaptive-personalized-cf && pytest tests/test_alpha_factory.py -v` exits 0 with at least "12 passed" reported (may report more due to parametrize expansion)
    - `cd federated-adaptive-personalized-cf && pytest tests/test_alpha_factory.py --collect-only 2>&1 | grep -cE "test_"` returns at least 18 (12 functions + 6 HC parametrize + 2 MF parametrize)
    - `git diff --stat federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/` returns empty after commit (this plan touches ZERO production code — only a test file)
  </acceptance_criteria>
  <done>ADP-07 regression surface shipped: 12 test functions (18 test items with parametrizations) covering (a) DataQuantityAlpha floor/ceiling/midpoint sigmoid endpoints, (b) each of 4 HierarchicalConditionalAlpha rule branches (sparse, niche, inconsistent, completionist) fires on designed inputs and `applied_rules` contains the rule name, (c) HC clip bounds hold under 6 adversarial (n, ge, nu, rs) inputs, (d) MultiFactorAlpha clip bounds hold under 2 adversarial inputs, (e) factory dispatch returns correct class for each valid method, (f) invalid method string raises ValueError. Zero production code changes.</done>
</task>

</tasks>

<verification>
- `cd federated-adaptive-personalized-cf && pytest tests/test_alpha_factory.py -v` exits 0 with at least "12 passed" (possibly more due to parametrizations)
- `cd federated-adaptive-personalized-cf && pytest tests/ -v` exits 0 with at least 38 tests passing (accumulated from Plans 01+02+03 — runs against whatever subset of those plans have already landed by Wave-2 sync)
- D-18 scope: `git diff --stat federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/` returns empty after this plan's commit (zero production code modifications)
- Wave-2 file ownership disjointness: this plan touches ONLY tests/test_alpha_factory.py. Plan 03 (Wave 2 sibling) touches client_app.py + task.py + 3 OTHER test files (test_client_assertion.py + test_task_rng.py + test_embedding_cache_manifest_v2.py). Zero file overlap.
</verification>

<success_criteria>
- ADP-07 observable: `pytest tests/test_alpha_factory.py` reports ≥12 GREEN unit tests against the UNMODIFIED adaptive_alpha.py factory; every test body asserts alpha ∈ [0.1, 0.95]; each HC rule branch is exercised with a designed trigger input and the rule name appears in applied_rules; factory dispatch is proven to be a closed-enum whitelist.
- No regression risk on production code: this plan touches ZERO source files under federated_adaptive_personalized_cf/; if adaptive_alpha.py's clip behavior ever silently regresses (someone removes the np.clip), these tests will catch it.
- Wave-2 parallel safety: file ownership disjoint from Plan 03; both plans commit concurrently without write-race.
</success_criteria>

<output>
After completion, create `.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-04-SUMMARY.md` with: file list (1 created: tests/test_alpha_factory.py), decisions made (none — pure regression-surface plan; the tests mirror Research §Code Examples verbatim), deviations (any: e.g., if compute_factors uses different key names than "applied_rules" the test asserts on — the executor must read adaptive_alpha.py and adapt), test counts (12 functions, ~18 items), commit SHA, ADP-07 closure note.
</output>
</content>
</invoke>