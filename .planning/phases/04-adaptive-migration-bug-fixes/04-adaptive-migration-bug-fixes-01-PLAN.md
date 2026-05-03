---
phase: 04-adaptive-migration-bug-fixes
plan: 01
type: execute
subsystem: infra
tags: [strategy, adaptive-split-fedavg, adaptive-split-fedprox, sufficient-stats, best-prototype-snapshot, prototype-ema, adp-03, adp-06, d-05, d-20, d-23, tdd, wave-1]
wave: 1
depends_on: []
files_modified:
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py
  - federated-adaptive-personalized-cf/tests/__init__.py
  - federated-adaptive-personalized-cf/tests/conftest.py
  - federated-adaptive-personalized-cf/tests/test_strategy.py
  - federated-adaptive-personalized-cf/tests/test_dual_model.py
autonomous: true
requirements: [ADP-03, ADP-06]

must_haves:
  truths:
    - "AdaptiveSplitFedAvg(BaseFedAvg) and AdaptiveSplitFedProx(BaseFedProx) subclasses exist in federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py; they own a prototype EMA (`_global_prototype`) + best-prototype snapshot (`best_prototype`) field AND override aggregate_evaluate to emit thesis metrics from summed sufficient stats (sum(hit_count)/sum(evaluated_users)) — Phase 3 PersonalizedSplitFedAvg clone with prototype extension."
    - "Module-level frozensets GLOBAL_PARAM_KEYS = {'item_embeddings.weight', 'item_bias.weight', 'global_bias'} and LOCAL_PARAM_KEYS_BASE = {'user_embeddings.weight', 'user_bias.weight'} are declared in strategy.py; LOCAL_PARAM_KEYS_BASE is the BASE set — the model's dynamic `_LOCAL_PARAMS` property at runtime appends `_logit_alpha.weight`, `_item_perturbation.weight`, `personal_mlp.*`, and fusion params as the enable flags dictate."
    - "aggregate_fit is OVERRIDDEN (not inherited): `super().aggregate_fit` runs the weighted-average of GLOBAL params then `self._aggregate_prototypes(results)` updates the server EMA. This differs from Phase 3's Plan 01 where aggregate_fit was pure-inherited — the adaptive module genuinely needs to aggregate prototypes (existing behavior, preserved)."
    - "`self.best_prototype: Optional[np.ndarray]` is initialized to `None` in `__init__` (D-05); `snapshot_best_prototype(round_num)` helper writes `self.best_prototype = self._global_prototype.copy()` if a prototype exists, else `np.zeros(embedding_dim, dtype=np.float32)` + emits WARNING per D-08."
    - "DualPersonalizedBPRMF enable-before-load contract is fingerprinted by test_dual_model.py: constructing a model with `enable_per_user_alpha(num_users=6040, init_alphas=...)` + `enable_item_perturbation(reg_lambda=0.01)` BEFORE set_local_parameters adds '_logit_alpha.weight' and '_item_perturbation.weight' to `_LOCAL_PARAMS`, and set_local_parameters(state, strict=False) restores those cached tensors (bug-fix ADP-02 behavior proof)."
    - "federated-adaptive-personalized-cf/tests/ pytest package exists with pytest fixtures (fake_evaluate_res, fake_client_proxy) and 10+ GREEN tests covering strategy sufficient-stat aggregation (5 tests) + best_prototype snapshot (2 tests) + dual_model enable-before-load contract (3 tests)."
  artifacts:
    - path: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py"
      provides: "AdaptiveSplitFedAvg + AdaptiveSplitFedProx with sufficient-stat aggregate_evaluate, prototype EMA aggregate_fit override, best_prototype snapshot field, GLOBAL_PARAM_KEYS + LOCAL_PARAM_KEYS_BASE frozensets, _sum_sufficient_stats + _sufficient_stats_to_thesis_metrics module-level helpers"
      contains: "class AdaptiveSplitFedAvg, class AdaptiveSplitFedProx, self.best_prototype, def _aggregate_prototypes, def snapshot_best_prototype"
    - path: "federated-adaptive-personalized-cf/tests/test_strategy.py"
      provides: "7 GREEN tests: 5 sufficient-stat clones from Phase 3 (sum aggregation, per-group ratios, zero-division, FedProx inherit, aggregate_fit override-includes-super) + 2 best_prototype snapshot tests (snapshot-on-best + degenerate-zero-vector-with-warning)"
    - path: "federated-adaptive-personalized-cf/tests/test_dual_model.py"
      provides: "3 GREEN tests: enable-before-load restores cached _logit_alpha/_item_perturbation, _LOCAL_PARAMS contains adaptive keys when flags on, set_local_parameters round-trip on extended key set"
  key_links:
    - from: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py"
      to: "fedrec_foundation.fit_metrics.EvaluateMetricsContract"
      via: "_sum_sufficient_stats reads the 12 _at10 / evaluated_users keys emitted client-side"
      pattern: "hit_count_overall_at10|ndcg_sum_overall_at10|evaluated_users"
    - from: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py"
      to: "federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/dual_personalized_bpr_mf.py"
      via: "LOCAL_PARAM_KEYS_BASE is a strict subset of model._LOCAL_PARAMS; the dynamic runtime expansion is the model's responsibility"
      pattern: "'user_embeddings.weight'"
---

<objective>
Lay the Phase 4 foundation in parallel with Plan 02: ship AdaptiveSplitFedAvg / AdaptiveSplitFedProx sufficient-stat aggregator with prototype-EMA aggregate_fit override AND best_prototype snapshot field (ADP-03 server half), closely mirroring Phase 3 PersonalizedSplitFedAvg with three deltas — (1) aggregate_fit is OVERRIDDEN to keep prototype aggregation, (2) a new `self.best_prototype: Optional[np.ndarray]` field tracked alongside the existing `_global_prototype`, (3) a `snapshot_best_prototype(round_num)` helper that server_app.py (Plan 05) will call at end-of-round when current_ndcg > best_metric.

Also fingerprints the DualPersonalizedBPRMF enable-before-load contract with a 3-test integration against the existing model class — proving the ADP-02 bug fix is meaningful BEFORE Plan 03 wires it into client_app.py.

Adds a pytest tests/ package (first use in adaptive module) with 10+ GREEN tests.

Purpose: Closes the strategy half of ADP-06 (sufficient-stat metrics) and lays ADP-03 strategy scaffolding so Plan 05 can drop the new strategy into server_app.py without touching models. Preserves the split-learning D-23 invariant (aggregate_fit parent's weighted average of GLOBAL params still runs) exactly as Phase 3 did — the only addition is the prototype branch AFTER super().aggregate_fit.

Output:
- federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py (rewritten)
- federated-adaptive-personalized-cf/tests/__init__.py (new — package marker)
- federated-adaptive-personalized-cf/tests/conftest.py (new — fake_evaluate_res + fake_client_proxy fixtures)
- federated-adaptive-personalized-cf/tests/test_strategy.py (new — 7 tests)
- federated-adaptive-personalized-cf/tests/test_dual_model.py (new — 3 tests)
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@.planning/phases/04-adaptive-migration-bug-fixes/04-CONTEXT.md
@.planning/phases/04-adaptive-migration-bug-fixes/04-RESEARCH.md
@.planning/phases/03-personalized-migration/03-personalized-migration-01-SUMMARY.md

<interfaces>
<!-- Phase 3 PersonalizedSplitFedAvg template (federated-personalized-cf/federated_personalized_cf/strategy.py, already shipped) -->
```python
# The Phase-3 sufficient-stat helpers + strategy subclasses to clone; Phase 4 adds prototype branch.
_SUFFICIENT_STAT_KEYS = (
    "hit_count_overall_at10", "ndcg_sum_overall_at10", "evaluated_users",
    "hit_count_sparse_at10", "ndcg_sum_sparse_at10", "evaluated_users_sparse",
    "hit_count_medium_at10", "ndcg_sum_medium_at10", "evaluated_users_medium",
    "hit_count_dense_at10",  "ndcg_sum_dense_at10",  "evaluated_users_dense",
)

def _sum_sufficient_stats(results): ...
def _sufficient_stats_to_thesis_metrics(totals): ...

class PersonalizedSplitFedAvg(BaseFedAvg):
    # aggregate_fit INHERITED UNCHANGED (D-23)
    def aggregate_evaluate(self, server_round, results, failures): ...  # sum-based ratio
```

<!-- Existing adaptive DualPersonalizedBPRMF._LOCAL_PARAMS property (models/dual_personalized_bpr_mf.py lines 542-569, UNCHANGED by Phase 4) -->
```python
class DualPersonalizedBPRMF(nn.Module):
    _LOCAL_PARAMS_BASE = ('user_embeddings.weight', 'user_bias.weight')
    _GLOBAL_PARAMS_WITH_BIAS = ('item_embeddings.weight', 'item_bias.weight', 'global_bias')

    @property
    def _LOCAL_PARAMS(self) -> tuple:
        base = list(self._LOCAL_PARAMS_BASE) if self.use_bias else ['user_embeddings.weight']
        if self._per_user_alpha_enabled and self._logit_alpha is not None:
            base.append('_logit_alpha.weight')
        if self._item_perturbation_enabled and self._item_perturbation is not None:
            base.append('_item_perturbation.weight')
        mlp_params = [name for name, _ in self.personal_mlp.named_parameters()]
        base.extend([f'personal_mlp.{name}' for name in mlp_params])
        if self.fusion_type == "gate":
            base.append('fusion_gate')
        elif self.fusion_type == "concat":
            base.extend(['fusion_layer.weight', 'fusion_layer.bias'])
        return tuple(base)

    def enable_per_user_alpha(self, num_users: int, init_alphas: Dict[int, float]) -> None: ...
    def enable_item_perturbation(self, reg_lambda: float) -> None: ...
    def get_local_parameters(self) -> OrderedDict: ...
    def set_local_parameters(self, state_dict: Dict[str, torch.Tensor], strict: bool = False) -> Tuple[List[str], List[str]]: ...
```

<!-- fedrec_foundation.fit_metrics.EvaluateMetricsContract (shipped Phase 1 Plan 03 + Phase 2 Plan 05) -->
```python
@dataclass
class EvaluateMetricsContract:
    hit_count_overall_at10: int
    ndcg_sum_overall_at10: float
    evaluated_users: int
    eval_loss: Optional[float] = None
    sampled_hr_at10: Optional[float] = None
    sampled_ndcg_at10: Optional[float] = None
    partition_id: Optional[int] = None
    hit_count_sparse_at10: Optional[int] = None
    # ... 8 more per-group fields
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvaluateMetricsContract": ...
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: AdaptiveSplitFedAvg + AdaptiveSplitFedProx strategy subclasses with sufficient-stat aggregate_evaluate + prototype EMA aggregate_fit override + best_prototype snapshot (ADP-03, ADP-06, D-05, D-20, tests/ package scaffolding)</name>
  <files>federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py, federated-adaptive-personalized-cf/tests/__init__.py, federated-adaptive-personalized-cf/tests/conftest.py, federated-adaptive-personalized-cf/tests/test_strategy.py</files>
  <read_first>
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py (ENTIRE FILE — preserves existing SplitFedAvg / SplitFedProx / USER_PROTOTYPE_KEY / _aggregate_prototypes logic; the rip target is renaming the classes to AdaptiveSplitFedAvg/AdaptiveSplitFedProx AND adding the sufficient-stat aggregate_evaluate override AND the best_prototype field + snapshot helper)
    - federated-personalized-cf/federated_personalized_cf/strategy.py (CANONICAL TEMPLATE — the post-Phase-3 sufficient-stat helpers and aggregate_evaluate shape to clone)
    - federated-personalized-cf/tests/conftest.py (fake_evaluate_res + fake_client_proxy fixtures — copy EXACTLY, only swap module-docstring "personalized" → "adaptive")
    - federated-personalized-cf/tests/test_strategy.py (5-test template — clone with strategy class names substituted, add 2 new tests for prototype EMA + best_prototype snapshot)
    - scripts/foundation/fedrec_foundation/fit_metrics.py (EvaluateMetricsContract + 12 sufficient-stat key list)
    - .planning/phases/04-adaptive-migration-bug-fixes/04-RESEARCH.md §"Adaptive strategy override with best_prototype snapshot" (lines ~672-728 — the exact AdaptiveSplitFedAvg skeleton with aggregate_fit override including super() call)
    - .planning/phases/04-adaptive-migration-bug-fixes/04-CONTEXT.md §"ADP-03 server prototype EMA best-round restore" (D-05 decision text)
    - .planning/phases/03-personalized-migration/03-personalized-migration-01-SUMMARY.md (D-23 identity-check pattern Phase 3 used; Phase 4 diverges — aggregate_fit IS overridden here because of prototype aggregation)
  </read_first>
  <behavior>
    Tests to write FIRST (RED), then implement. Test file has 7 tests:

    1. test_aggregate_evaluate_sums_sufficient_stats: 3 clients with overall hit_count=(10, 5, 7), evaluated_users=(20, 15, 25) → AdaptiveSplitFedAvg.aggregate_evaluate returns sampled_hr@10 ≈ 22/60 ≈ 0.3667, evaluated_users == 60.

    2. test_aggregate_evaluate_per_group_ratios: 2 clients with per-group sparse/medium/dense hit + evaluated_users → asserts 3 per-group ratios + 3 per-group evaluated_users match arithmetic.

    3. test_aggregate_evaluate_zero_division_safe: 1 client with evaluated_users_sparse=0 → sampled_hr@10/sparse == 0.0 (no ZeroDivisionError).

    4. test_adaptive_split_fedprox_inherits_aggregate_evaluate: instantiate AdaptiveSplitFedProx(fraction_fit=0.1, proximal_mu=0.01) and verify sum-based ratio logic still works.

    5. test_aggregate_fit_calls_super_then_prototypes: use `unittest.mock.patch.object(BaseFedAvg, 'aggregate_fit')` to spy on super().aggregate_fit; instantiate AdaptiveSplitFedAvg; call aggregate_fit(1, results=[], failures=[]); assert super was called AND self._global_prototype was (attempted) updated. (Verifies D-23 modification — parent still runs BUT prototype aggregation follows.)

    6. test_best_prototype_snapshot_at_best_round: instantiate AdaptiveSplitFedAvg; manually set `strategy._global_prototype = np.array([1.0, 2.0, 3.0], dtype=np.float32)`; call `strategy.snapshot_best_prototype(round_num=5, embedding_dim=3)`; assert `strategy.best_prototype is not None` and `np.allclose(strategy.best_prototype, [1.0, 2.0, 3.0])`; assert `strategy.best_prototype is not strategy._global_prototype` (copy, not reference).

    7. test_best_prototype_snapshot_degenerate_zero_vector: instantiate fresh strategy (so `_global_prototype` is None — no prior aggregation); call `strategy.snapshot_best_prototype(round_num=1, embedding_dim=128)` and capture WARNING via caplog; assert `np.allclose(strategy.best_prototype, np.zeros(128))` AND assert at least one WARNING-level log message contains the substrings "Prototype snapshot at best round" AND "zero vector".
  </behavior>
  <action>
    Step 1 — Rip-and-replace federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py with this skeleton (clone Phase 3 structure + extend with prototype + best_prototype). Use exactly this code:

    ```python
    """Split Learning Strategies for Federated Adaptive Personalized Collaborative Filtering (Phase 4 Plan 01).

    AdaptiveSplitFedAvg / AdaptiveSplitFedProx subclass Flower's FedAvg/FedProx and:
    1. Override aggregate_evaluate to emit thesis metrics ONCE from summed sufficient stats
       (sum(hit_count)/sum(evaluated_users)) instead of averaging per-client ratios — mirrors
       Phase 3 PersonalizedSplitFedAvg (ADP-06 server half).
    2. Override aggregate_fit to call super().aggregate_fit (weighted-average of GLOBAL params,
       D-23) AND then update the server EMA prototype via _aggregate_prototypes — preserves
       existing adaptive behavior. This DIVERGES from Phase 3 where aggregate_fit was pure-
       inherited; the adaptive module genuinely needs server-side prototype aggregation.
    3. Hold a `best_prototype` snapshot field alongside `_global_prototype` (ADP-03, D-05);
       server_app.py (Plan 05) calls snapshot_best_prototype(round_num, embedding_dim) when
       current_ndcg > best_metric.

    Parameter split:
      GLOBAL: item_embeddings.weight, item_bias.weight, global_bias
      LOCAL (base): user_embeddings.weight, user_bias.weight
      LOCAL (dynamic): _logit_alpha.weight, _item_perturbation.weight, personal_mlp.*,
                       fusion_gate / fusion_layer.weight + fusion_layer.bias
        — appended at runtime by DualPersonalizedBPRMF._LOCAL_PARAMS property based on
          enable_per_user_alpha / enable_item_perturbation / fusion_type flags. The
          frozenset LOCAL_PARAM_KEYS_BASE below is the BASE set only.
    """
    from logging import WARNING
    from typing import Dict, List, Optional, Tuple, Union

    import numpy as np
    from flwr.common import EvaluateRes, FitRes, Parameters, Scalar
    from flwr.common.logger import log
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import FedAvg as BaseFedAvg, FedProx as BaseFedProx

    # D-03 frozensets mirror Phase 3 with the BASE local set; runtime expansion owned by model
    GLOBAL_PARAM_KEYS = frozenset({
        "item_embeddings.weight",
        "item_bias.weight",
        "global_bias",
    })
    LOCAL_PARAM_KEYS_BASE = frozenset({
        "user_embeddings.weight",
        "user_bias.weight",
    })

    # Key used for user prototype in metrics (Phase 3 + earlier adaptive code kept unchanged)
    USER_PROTOTYPE_KEY = "user_prototype"

    # 12 sufficient-stat keys (Phase 2 Plan 01 + Phase 3 Plan 01 contract)
    _SUFFICIENT_STAT_KEYS = (
        "hit_count_overall_at10", "ndcg_sum_overall_at10", "evaluated_users",
        "hit_count_sparse_at10", "ndcg_sum_sparse_at10", "evaluated_users_sparse",
        "hit_count_medium_at10", "ndcg_sum_medium_at10", "evaluated_users_medium",
        "hit_count_dense_at10",  "ndcg_sum_dense_at10",  "evaluated_users_dense",
    )


    def _sum_sufficient_stats(
        results: List[Tuple[ClientProxy, EvaluateRes]],
    ) -> Dict[str, float]:
        """Sum each of the 12 sufficient-stat keys across EvaluateRes payloads."""
        totals: Dict[str, float] = {k: 0 for k in _SUFFICIENT_STAT_KEYS}
        for _proxy, eval_res in results:
            metrics = eval_res.metrics or {}
            for k in _SUFFICIENT_STAT_KEYS:
                totals[k] += metrics.get(k, 0) or 0
        return totals


    def _sufficient_stats_to_thesis_metrics(
        totals: Dict[str, float],
    ) -> Dict[str, float]:
        """Derive thesis headline metrics from summed sufficient stats."""
        def _safe_ratio(num, den):
            return (num / den) if den else 0.0

        return {
            "sampled_hr@10":   _safe_ratio(totals["hit_count_overall_at10"], totals["evaluated_users"]),
            "sampled_ndcg@10": _safe_ratio(totals["ndcg_sum_overall_at10"], totals["evaluated_users"]),
            "sampled_hr@10/sparse":   _safe_ratio(totals["hit_count_sparse_at10"],  totals["evaluated_users_sparse"]),
            "sampled_ndcg@10/sparse": _safe_ratio(totals["ndcg_sum_sparse_at10"],  totals["evaluated_users_sparse"]),
            "sampled_hr@10/medium":   _safe_ratio(totals["hit_count_medium_at10"], totals["evaluated_users_medium"]),
            "sampled_ndcg@10/medium": _safe_ratio(totals["ndcg_sum_medium_at10"], totals["evaluated_users_medium"]),
            "sampled_hr@10/dense":    _safe_ratio(totals["hit_count_dense_at10"],  totals["evaluated_users_dense"]),
            "sampled_ndcg@10/dense":  _safe_ratio(totals["ndcg_sum_dense_at10"],  totals["evaluated_users_dense"]),
            "evaluated_users":        totals["evaluated_users"],
            "evaluated_users_sparse": totals["evaluated_users_sparse"],
            "evaluated_users_medium": totals["evaluated_users_medium"],
            "evaluated_users_dense":  totals["evaluated_users_dense"],
        }


    class AdaptiveSplitFedAvg(BaseFedAvg):
        """FedAvg variant for adaptive split learning with prototype EMA + best-round snapshot.

        aggregate_fit: super() runs the weighted-average of GLOBAL params (D-23) then
                       _aggregate_prototypes updates self._global_prototype via EMA.
        aggregate_evaluate: sum-based sufficient-stat ratio (ADP-06).
        best_prototype: Optional[np.ndarray] — server-side snapshot field mirroring the
                        Phase 2 D-27 best_arrays pattern (ADP-03, D-05).
        """

        def __init__(
            self,
            fraction_fit: float = 1.0,
            prototype_momentum: float = 0.9,
            **kwargs,
        ):
            super().__init__(fraction_fit=fraction_fit, **kwargs)
            self.global_param_keys = GLOBAL_PARAM_KEYS
            self.local_param_keys_base = LOCAL_PARAM_KEYS_BASE
            self._is_split_learning = True
            self.prototype_momentum = prototype_momentum
            self._global_prototype: Optional[np.ndarray] = None
            # D-05: best-round snapshot field; server_app.py (Plan 05) calls snapshot_best_prototype.
            self.best_prototype: Optional[np.ndarray] = None

        def __repr__(self) -> str:
            return (
                f"AdaptiveSplitFedAvg(fraction_fit={self.fraction_fit}, "
                f"prototype_momentum={self.prototype_momentum})"
            )

        def get_global_prototype(self) -> Optional[np.ndarray]:
            """Return the current server-side global user prototype (EMA), or None if never aggregated."""
            return self._global_prototype

        def snapshot_best_prototype(self, round_num: int, embedding_dim: int) -> None:
            """D-05 best-round snapshot. Called by server_app.py when current_ndcg > best_metric.

            If `_global_prototype` is not None, copy it. Else (D-08 degenerate case: best
            round fires before any prototype was aggregated), snapshot np.zeros(embedding_dim)
            and emit WARNING.
            """
            if self._global_prototype is not None:
                self.best_prototype = self._global_prototype.copy()
            else:
                self.best_prototype = np.zeros(int(embedding_dim), dtype=np.float32)
                log(
                    WARNING,
                    f"Prototype snapshot at best round R={round_num} is zero vector "
                    f"— no prior prototype aggregation yet.",
                )

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate GLOBAL params via super(), then update server EMA prototype.

            This override exists so prototype aggregation CONTINUES to run (existing adaptive
            behavior) — Phase 3's pure-inheritance approach is insufficient because the adaptive
            module has server-side state beyond just GLOBAL params.
            """
            aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)
            self._aggregate_prototypes(results)
            if self._global_prototype is not None:
                metrics["global_prototype_norm"] = float(np.linalg.norm(self._global_prototype))
            return aggregated_params, metrics

        def _aggregate_prototypes(
            self,
            results: List[Tuple[ClientProxy, FitRes]],
        ) -> None:
            """Weighted-mean client prototypes then EMA update on self._global_prototype.

            Preserves existing behavior. Missing prototypes (client didn't contribute) are
            skipped; if no client contributed this round, the field stays unchanged.
            """
            prototypes_and_weights: List[Tuple[np.ndarray, int]] = []
            for _proxy, fit_res in results:
                metrics = fit_res.metrics or {}
                proto = metrics.get(USER_PROTOTYPE_KEY)
                if isinstance(proto, (list, tuple)):
                    arr = np.asarray(proto, dtype=np.float32)
                    prototypes_and_weights.append((arr, int(fit_res.num_examples)))
            if not prototypes_and_weights:
                return
            total_weight = sum(w for _, w in prototypes_and_weights)
            new_prototype = sum(p * w for p, w in prototypes_and_weights) / total_weight
            if self._global_prototype is None:
                self._global_prototype = new_prototype
            else:
                m = self.prototype_momentum
                self._global_prototype = m * self._global_prototype + (1.0 - m) * new_prototype

        def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
            """Sum sufficient stats across clients; emit thesis metrics dict (ADP-06)."""
            if not results:
                return (None, {})
            loss_num = sum(r.loss * r.num_examples for _, r in results)
            loss_den = sum(r.num_examples for _, r in results) or 1
            totals = _sum_sufficient_stats(results)
            return (loss_num / loss_den, _sufficient_stats_to_thesis_metrics(totals))


    class AdaptiveSplitFedProx(BaseFedProx):
        """FedProx variant that reuses the sum-based aggregate_evaluate + prototype EMA.

        aggregate_evaluate is an EXACT COPY of AdaptiveSplitFedAvg.aggregate_evaluate
        (not super() call) to avoid diamond-inheritance with BaseFedProx.
        aggregate_fit is overridden the same way as AdaptiveSplitFedAvg.
        best_prototype field + snapshot_best_prototype helper are duplicated at the class level.
        """

        def __init__(
            self,
            fraction_fit: float = 1.0,
            prototype_momentum: float = 0.9,
            proximal_mu: float = 0.01,
            **kwargs,
        ):
            super().__init__(fraction_fit=fraction_fit, proximal_mu=proximal_mu, **kwargs)
            self.global_param_keys = GLOBAL_PARAM_KEYS
            self.local_param_keys_base = LOCAL_PARAM_KEYS_BASE
            self._is_split_learning = True
            self.prototype_momentum = prototype_momentum
            self._global_prototype: Optional[np.ndarray] = None
            self.best_prototype: Optional[np.ndarray] = None

        def __repr__(self) -> str:
            return (
                f"AdaptiveSplitFedProx(fraction_fit={self.fraction_fit}, "
                f"proximal_mu={self.proximal_mu}, "
                f"prototype_momentum={self.prototype_momentum})"
            )

        def get_global_prototype(self) -> Optional[np.ndarray]:
            return self._global_prototype

        def snapshot_best_prototype(self, round_num: int, embedding_dim: int) -> None:
            if self._global_prototype is not None:
                self.best_prototype = self._global_prototype.copy()
            else:
                self.best_prototype = np.zeros(int(embedding_dim), dtype=np.float32)
                log(
                    WARNING,
                    f"Prototype snapshot at best round R={round_num} is zero vector "
                    f"— no prior prototype aggregation yet.",
                )

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)
            self._aggregate_prototypes(results)
            if self._global_prototype is not None:
                metrics["global_prototype_norm"] = float(np.linalg.norm(self._global_prototype))
            return aggregated_params, metrics

        def _aggregate_prototypes(
            self,
            results: List[Tuple[ClientProxy, FitRes]],
        ) -> None:
            prototypes_and_weights: List[Tuple[np.ndarray, int]] = []
            for _proxy, fit_res in results:
                metrics = fit_res.metrics or {}
                proto = metrics.get(USER_PROTOTYPE_KEY)
                if isinstance(proto, (list, tuple)):
                    arr = np.asarray(proto, dtype=np.float32)
                    prototypes_and_weights.append((arr, int(fit_res.num_examples)))
            if not prototypes_and_weights:
                return
            total_weight = sum(w for _, w in prototypes_and_weights)
            new_prototype = sum(p * w for p, w in prototypes_and_weights) / total_weight
            if self._global_prototype is None:
                self._global_prototype = new_prototype
            else:
                m = self.prototype_momentum
                self._global_prototype = m * self._global_prototype + (1.0 - m) * new_prototype

        def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
            if not results:
                return (None, {})
            loss_num = sum(r.loss * r.num_examples for _, r in results)
            loss_den = sum(r.num_examples for _, r in results) or 1
            totals = _sum_sufficient_stats(results)
            return (loss_num / loss_den, _sufficient_stats_to_thesis_metrics(totals))


    __all__ = [
        "AdaptiveSplitFedAvg",
        "AdaptiveSplitFedProx",
        "GLOBAL_PARAM_KEYS",
        "LOCAL_PARAM_KEYS_BASE",
        "USER_PROTOTYPE_KEY",
    ]
    ```

    Step 2 — Surgical preservation (D-18 discipline): the OLD strategy.py defines `SplitFedAvg`, `SplitFedProx`, `USER_PROTOTYPE_KEY`, `GLOBAL_PARAM_KEYS`, `LOCAL_PARAM_KEYS` module-level names. The rip-and-replace COMPLETELY REPLACES those class names with `AdaptiveSplitFedAvg` / `AdaptiveSplitFedProx` (renamed), preserves `USER_PROTOTYPE_KEY` and `GLOBAL_PARAM_KEYS` verbatim at the module level, and CHANGES `LOCAL_PARAM_KEYS` to `LOCAL_PARAM_KEYS_BASE`. Any pre-existing uncommitted WIP unrelated to these symbols (e.g., import reorganization) is dropped — this is our helper file, rip-and-replace authorized.

    Step 3 — Create federated-adaptive-personalized-cf/tests/__init__.py as an empty file (marker — makes tests/ a package).

    Step 4 — Create federated-adaptive-personalized-cf/tests/conftest.py. COPY VERBATIM from federated-personalized-cf/tests/conftest.py (the existing Phase 3 file with fake_evaluate_res + fake_client_proxy fixtures). Change ONLY the module docstring header to reference "adaptive" instead of "personalized". Do not add new fixtures — Task 2 and downstream plans add fixtures as needed.

    Step 5 — Create federated-adaptive-personalized-cf/tests/test_strategy.py. Clone Phase 3 federated-personalized-cf/tests/test_strategy.py with class-name substitutions (`PersonalizedSplitFedAvg` → `AdaptiveSplitFedAvg`, `PersonalizedSplitFedProx` → `AdaptiveSplitFedProx`) AND add 3 new tests (5, 6, 7 in behavior list) specific to Phase 4. For test 5 (aggregate_fit calls super):

    ```python
    def test_aggregate_fit_calls_super_then_prototypes(fake_client_proxy, monkeypatch):
        from unittest.mock import patch, MagicMock
        from flwr.server.strategy import FedAvg as BaseFedAvg
        from federated_adaptive_personalized_cf.strategy import AdaptiveSplitFedAvg, USER_PROTOTYPE_KEY
        from flwr.common import FitRes, Status, Code, Parameters

        strategy = AdaptiveSplitFedAvg(fraction_fit=1.0, prototype_momentum=0.9)
        fake_params = Parameters(tensors=[], tensor_type="numpy.ndarray")
        with patch.object(BaseFedAvg, "aggregate_fit", return_value=(fake_params, {})) as mock_super:
            # Pass a single fake result that carries a user_prototype list
            fit_res = FitRes(
                status=Status(Code.OK, "ok"),
                parameters=fake_params,
                num_examples=10,
                metrics={USER_PROTOTYPE_KEY: [1.0, 2.0, 3.0]},
            )
            result = strategy.aggregate_fit(server_round=1, results=[(fake_client_proxy(0), fit_res)], failures=[])
            assert mock_super.called, "D-23 violated: super().aggregate_fit must run"
        # After aggregate_fit, prototype should be updated from the single client.
        import numpy as np
        assert strategy._global_prototype is not None
        assert np.allclose(strategy._global_prototype, np.array([1.0, 2.0, 3.0]))
    ```

    For tests 6 and 7 (best_prototype snapshot):
    ```python
    def test_best_prototype_snapshot_at_best_round():
        import numpy as np
        from federated_adaptive_personalized_cf.strategy import AdaptiveSplitFedAvg
        strategy = AdaptiveSplitFedAvg(fraction_fit=1.0)
        strategy._global_prototype = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        strategy.snapshot_best_prototype(round_num=5, embedding_dim=3)
        assert strategy.best_prototype is not None
        assert np.allclose(strategy.best_prototype, [1.0, 2.0, 3.0])
        # Copy, not reference — mutating _global_prototype must not touch best_prototype
        strategy._global_prototype[0] = 999.0
        assert np.allclose(strategy.best_prototype, [1.0, 2.0, 3.0])

    def test_best_prototype_snapshot_degenerate_zero_vector(caplog):
        import logging
        import numpy as np
        from federated_adaptive_personalized_cf.strategy import AdaptiveSplitFedAvg
        strategy = AdaptiveSplitFedAvg(fraction_fit=1.0)
        assert strategy._global_prototype is None
        with caplog.at_level(logging.WARNING):
            strategy.snapshot_best_prototype(round_num=1, embedding_dim=128)
        assert strategy.best_prototype is not None
        assert np.allclose(strategy.best_prototype, np.zeros(128))
        assert any("Prototype snapshot at best round" in rec.getMessage()
                   and "zero vector" in rec.getMessage()
                   for rec in caplog.records), f"Expected D-08 warning, got {caplog.records}"
    ```

    Step 6 — Verify: `cd federated-adaptive-personalized-cf && pip install -e .[dev] && pytest tests/test_strategy.py -v` → 7 passed. (Plan 02 adds pytest dev-dep; Plan 01 runs after Plan 02 in Wave 1 if pytest isn't yet installed — but the two plans commit in parallel. The acceptance runs `pip install pytest>=7.0` as a fallback.)

    Step 7 — Commit (--no-verify per Wave-1 parallel rule; Plan 02 owns pyproject.toml, so we cannot touch it):
    ```
    git add federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py \
            federated-adaptive-personalized-cf/tests/__init__.py \
            federated-adaptive-personalized-cf/tests/conftest.py \
            federated-adaptive-personalized-cf/tests/test_strategy.py
    git commit --no-verify -m "feat(04-01): AdaptiveSplitFedAvg + AdaptiveSplitFedProx + best_prototype snapshot (ADP-03, ADP-06, D-05)"
    ```
  </action>
  <acceptance_criteria>
    - `grep -c "^class AdaptiveSplitFedAvg" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` returns 1
    - `grep -c "^class AdaptiveSplitFedProx" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` returns 1
    - `grep -c "class SplitFedAvg" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` returns 0 (old class renamed)
    - `grep -c "GLOBAL_PARAM_KEYS = frozenset" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` returns 1
    - `grep -c "LOCAL_PARAM_KEYS_BASE = frozenset" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` returns 1
    - `grep -c "'user_embeddings.weight'" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` returns at least 1 (inside LOCAL_PARAM_KEYS_BASE — ghost-table acknowledged at strategy level; per RESEARCH open question 1, adaptive keeps the num_users × d embedding)
    - `grep -c "'item_embeddings.weight'" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` returns at least 1 (inside GLOBAL_PARAM_KEYS)
    - `grep -c "def _sum_sufficient_stats" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` returns 1
    - `grep -c "def _sufficient_stats_to_thesis_metrics" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` returns 1
    - `grep -c "def aggregate_fit" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` returns 2 (overridden on BOTH FedAvg and FedProx — diverges from Phase 3 where aggregate_fit was inherited)
    - `grep -c "def aggregate_evaluate" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` returns 2 (one per subclass)
    - `grep -c "self.best_prototype" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` returns at least 4 (init to None on both subclasses + mutation in snapshot_best_prototype on both)
    - `grep -c "def snapshot_best_prototype" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` returns 2 (duplicated across subclasses to avoid diamond-inheritance with BaseFedProx)
    - `grep -cE "no prior prototype aggregation yet" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` returns at least 2 (D-08 warning message duplicated across subclasses)
    - `grep -c "super().aggregate_fit" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` returns 2 (one per subclass)
    - `cd federated-adaptive-personalized-cf && pip install --quiet "pytest>=7.0" 2>/dev/null; pytest tests/test_strategy.py -v` exits 0 with "7 passed"
    - `python -c "from federated_adaptive_personalized_cf.strategy import AdaptiveSplitFedAvg, AdaptiveSplitFedProx, GLOBAL_PARAM_KEYS, LOCAL_PARAM_KEYS_BASE, USER_PROTOTYPE_KEY; assert GLOBAL_PARAM_KEYS == frozenset({'item_embeddings.weight', 'item_bias.weight', 'global_bias'}); assert LOCAL_PARAM_KEYS_BASE == frozenset({'user_embeddings.weight', 'user_bias.weight'}); s = AdaptiveSplitFedAvg(fraction_fit=1.0); assert s.best_prototype is None; print('ok')"` prints `ok`
    - `git diff --stat federated-adaptive-personalized-cf/pyproject.toml federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/` returns empty after commit (Wave-1 write-race avoidance; Plans 02/03/04/05 own those)
  </acceptance_criteria>
  <done>AdaptiveSplitFedAvg + AdaptiveSplitFedProx shipped with sufficient-stat aggregate_evaluate + prototype EMA aggregate_fit override + best_prototype snapshot field + snapshot_best_prototype helper with D-08 zero-vector degenerate handling; GLOBAL_PARAM_KEYS + LOCAL_PARAM_KEYS_BASE frozensets declare the flipped split (item_* GLOBAL, user_* LOCAL base); 7 GREEN strategy tests; D-05 + D-23 + D-20 + ADP-03 strategy scaffolding + ADP-06 aggregator ready for Plan 05 server wire-up.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: DualPersonalizedBPRMF enable-before-load fingerprint tests (ADP-02 bug-fix behavior proof against UNMODIFIED model)</name>
  <files>federated-adaptive-personalized-cf/tests/test_dual_model.py</files>
  <read_first>
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/dual_personalized_bpr_mf.py (lines 95-175 for __init__ ghost-table construction; lines 400-570 for enable_per_user_alpha/enable_item_perturbation + _LOCAL_PARAMS property + get/set_local_parameters; UNMODIFIED by Phase 4)
    - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/adaptive_alpha.py (AlphaConfig + create_alpha_computer — tests need to construct a minimal alpha_computer to feed compute_per_user_alpha)
    - .planning/phases/04-adaptive-migration-bug-fixes/04-RESEARCH.md §"Pattern 1: Enable-Before-Load Ordering" + §"Pitfall 1" (lines ~226-280, 484-498 — the exact ordering sequence tests must prove is now correct)
    - .planning/phases/04-adaptive-migration-bug-fixes/04-CONTEXT.md §D-03 (enable_* unconditional in benchmark_cross_device)
  </read_first>
  <behavior>
    Tests to write (GREEN on first run — the tests drive a fresh model + enable flags + set_local_parameters sequence and assert cached tensors are restored). 3 tests:

    1. test_local_params_without_enable_flags: construct DualPersonalizedBPRMF(num_users=10, num_items=20, embedding_dim=8, mlp_hidden_dims=[16], fusion_type="concat") with no enable_* calls; assert `'_logit_alpha.weight' not in model._LOCAL_PARAMS` and `'_item_perturbation.weight' not in model._LOCAL_PARAMS`; assert set(model._LOCAL_PARAMS) ⊇ {'user_embeddings.weight', 'user_bias.weight', 'fusion_layer.weight', 'fusion_layer.bias'}.

    2. test_local_params_with_enable_flags_before_construction_of_cache: construct the same model; call `model.enable_per_user_alpha(num_users=10, init_alphas={i: 0.5 for i in range(10)})` AND `model.enable_item_perturbation(reg_lambda=0.01)`; assert `'_logit_alpha.weight' in model._LOCAL_PARAMS` AND `'_item_perturbation.weight' in model._LOCAL_PARAMS`; assert `model._logit_alpha is not None` AND `model._item_perturbation is not None`.

    3. test_enable_before_load_restores_cached_alpha: two-step round-trip proving the ADP-02 bug fix:
       - Round-1 simulation: construct model_a, enable_per_user_alpha + enable_item_perturbation, MUTATE model_a._logit_alpha.weight.data to a known sentinel value like `torch.full_like(..., 0.123)`, grab `state = model_a.get_local_parameters()`.
       - Round-2 simulation: construct fresh model_b, call enable_per_user_alpha + enable_item_perturbation BEFORE set_local_parameters (the bug-fix ordering), call `loaded, missing = model_b.set_local_parameters(state, strict=False)`; assert `'_logit_alpha.weight' in loaded` AND `'_item_perturbation.weight' in loaded` AND `torch.allclose(model_b._logit_alpha.weight, torch.full_like(model_b._logit_alpha.weight, 0.123))` — the sentinel survived the round-trip because the keys were in _LOCAL_PARAMS at load time.
       - Round-3 simulation (negative / regression): construct fresh model_c, call set_local_parameters(state, strict=False) BEFORE enable_per_user_alpha; assert `'_logit_alpha.weight' NOT in loaded` (the bug manifests: load silently skipped the adaptive keys); this is documented as the bug being fixed.
  </behavior>
  <action>
    Step 1 — Create federated-adaptive-personalized-cf/tests/test_dual_model.py. Header:

    ```python
    """ADP-02 fingerprint tests — enable-before-load ordering bug-fix behavior.

    These tests prove that calling enable_per_user_alpha + enable_item_perturbation BEFORE
    set_local_parameters causes the cached _logit_alpha.weight and _item_perturbation.weight
    tensors to be restored (the fix), while the inverse ordering silently drops them (the bug).

    The underlying DualPersonalizedBPRMF class is UNTOUCHED by Phase 4 — its _LOCAL_PARAMS
    property already reacts correctly to the enable flags. Phase 4's fix is in client_app.py
    (Plan 03) which reorders the calls; this test is the defense-in-depth regression guard.
    """
    import pytest
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


    def test_local_params_without_enable_flags():
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


    def test_local_params_with_enable_flags_before_construction_of_cache():
        model = _build_model()
        model.enable_per_user_alpha(num_users=10, init_alphas={i: 0.5 for i in range(10)})
        model.enable_item_perturbation(reg_lambda=0.01)
        local = set(model._LOCAL_PARAMS)
        assert "_logit_alpha.weight" in local
        assert "_item_perturbation.weight" in local
        assert model._logit_alpha is not None
        assert model._item_perturbation is not None


    def test_enable_before_load_restores_cached_alpha():
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
        loaded, missing = model_b.set_local_parameters(cached_state, strict=False)
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
    ```

    Step 2 — Verify: `cd federated-adaptive-personalized-cf && pytest tests/test_dual_model.py -v` → 3 passed. Also confirm the combined tests/ directory suite: `pytest tests/ -v` → 10 passed (Task 1 = 7 strategy tests + Task 2 = 3 model tests).

    Step 3 — Commit (--no-verify per Wave-1 parallel rule):
    ```
    git add federated-adaptive-personalized-cf/tests/test_dual_model.py
    git commit --no-verify -m "test(04-01): enable-before-load ADP-02 fingerprint tests"
    ```
  </action>
  <acceptance_criteria>
    - `test -r federated-adaptive-personalized-cf/tests/test_dual_model.py` succeeds
    - `grep -c "^def test_local_params_without_enable_flags" federated-adaptive-personalized-cf/tests/test_dual_model.py` returns 1
    - `grep -c "^def test_local_params_with_enable_flags_before_construction_of_cache" federated-adaptive-personalized-cf/tests/test_dual_model.py` returns 1
    - `grep -c "^def test_enable_before_load_restores_cached_alpha" federated-adaptive-personalized-cf/tests/test_dual_model.py` returns 1
    - `grep -c "ADP-02" federated-adaptive-personalized-cf/tests/test_dual_model.py` returns at least 2 (the bug ID appears in assertion messages and module docstring)
    - `grep -c "from federated_adaptive_personalized_cf.models.dual_personalized_bpr_mf import" federated-adaptive-personalized-cf/tests/test_dual_model.py` returns 1
    - `cd federated-adaptive-personalized-cf && pytest tests/test_dual_model.py -v` exits 0 with "3 passed"
    - `cd federated-adaptive-personalized-cf && pytest tests/ -v` exits 0 with "10 passed" (combined Task 1 + Task 2 suite)
    - `git diff --stat federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/` returns empty after commit (model + strategy untouched by Task 2)
  </acceptance_criteria>
  <done>ADP-02 enable-before-load ordering is fingerprinted by 3 GREEN tests against the UNTOUCHED DualPersonalizedBPRMF class: (a) flags-off baseline, (b) flags-on adds _logit_alpha.weight + _item_perturbation.weight to _LOCAL_PARAMS, (c) round-trip save→load-with-correct-ordering restores cached sentinel tensor values. This is the behavior proof Plan 03 will target when it reorders the calls in client_app.py.</done>
</task>

</tasks>

<verification>
- `cd federated-adaptive-personalized-cf && pytest tests/ -v` exits 0 with "10 passed" (7 strategy + 3 model)
- `python -c "from federated_adaptive_personalized_cf.strategy import AdaptiveSplitFedAvg, AdaptiveSplitFedProx, GLOBAL_PARAM_KEYS, LOCAL_PARAM_KEYS_BASE; s = AdaptiveSplitFedAvg(fraction_fit=1.0, prototype_momentum=0.9); assert s.best_prototype is None; assert s._global_prototype is None; p = AdaptiveSplitFedProx(fraction_fit=1.0, proximal_mu=0.01); assert p.best_prototype is None; print('ok')"` prints `ok`
- `python -c "from federated_adaptive_personalized_cf.models.dual_personalized_bpr_mf import DualPersonalizedBPRMF; m = DualPersonalizedBPRMF(num_users=10, num_items=20, embedding_dim=8, mlp_hidden_dims=[16], fusion_type='concat'); m.enable_per_user_alpha(num_users=10, init_alphas={i: 0.5 for i in range(10)}); m.enable_item_perturbation(reg_lambda=0.01); assert '_logit_alpha.weight' in m._LOCAL_PARAMS and '_item_perturbation.weight' in m._LOCAL_PARAMS; print('ok')"` prints `ok`
- `git log --oneline -3` shows the 2 task commits (Task 1 `feat(04-01): AdaptiveSplit...`, Task 2 `test(04-01): enable-before-load...`)
- `git diff --stat federated-adaptive-personalized-cf/pyproject.toml federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/` returns empty (Wave-1 write-race safety; Plans 02/03/04/05 own those)
</verification>

<success_criteria>
- ADP-03 strategy scaffolding observable: AdaptiveSplitFedAvg / AdaptiveSplitFedProx carry a `best_prototype: Optional[np.ndarray]` field + `snapshot_best_prototype(round_num, embedding_dim)` helper; D-08 degenerate case (best round before any prototype aggregated) emits a WARNING and snapshots `np.zeros(embedding_dim)`.
- ADP-06 server-half observable at the strategy layer: sufficient-stat aggregate_evaluate (12-key sum) ships mirroring Phase 3; aggregate_fit is OVERRIDDEN to call super() + _aggregate_prototypes (not pure-inherited like Phase 3 — the adaptive module has extra server state to maintain).
- ADP-02 bug-fix behavior proof: 3 GREEN tests against the UNTOUCHED DualPersonalizedBPRMF class demonstrate enable_per_user_alpha + enable_item_perturbation BEFORE set_local_parameters restores cached tensors; Plan 03 will target this contract when it reorders client_app.py.
- tests/ package exists at federated-adaptive-personalized-cf/tests/ (package with __init__.py + conftest.py + test_strategy.py + test_dual_model.py) and reports 10 GREEN tests.
- Wave-1 write-race prevented: pyproject.toml, dataset.py, client_app.py, server_app.py, task.py, models/ are untouched by Plan 01 (Plans 02/03/04/05 own them).
</success_criteria>

<output>
After completion, create `.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-01-SUMMARY.md` using the standard summary template with: file list (5 files: strategy.py + 4 test files), decisions made (aggregate_fit override justification — adaptive module needs server-side prototype EMA beyond just GLOBAL params; best_prototype snapshot placed on strategy object per research Pattern 2), deviations (if any), test counts (10 GREEN), commit SHAs, next-plan readiness (Plan 03 depends on this strategy + test_dual_model behavior proof; Plan 05 depends on this strategy for wire-up + snapshot_best_prototype call site).
</output>
</content>
</invoke>