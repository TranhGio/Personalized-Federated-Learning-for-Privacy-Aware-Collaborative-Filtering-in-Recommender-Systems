---
phase: 05-pfedrec-migration-reproduction
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - federated-pfedrec/federated_pfedrec/strategy.py
  - federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py
  - federated-pfedrec/tests/test_strategy.py
  - federated-pfedrec/tests/test_pfedrec_mlp.py
autonomous: true
requirements: [PFR-02, PFR-03]
must_haves:
  truths:
    - "PFedRecMLP._GLOBAL_PARAMS contains both 'embedding_item.weight' and 'affine_output.bias' (D-01)"
    - "PFedRecMLP._LOCAL_PARAMS contains exactly ['affine_output.weight'] (D-01)"
    - "Strategy class is named PFedRecSplitFedAvg (D-12); SplitFedProx removed (D-07)"
    - "set_local_parameters(strict=True) raises RuntimeError on shape mismatch with rm -rf hint (D-21)"
    - "PFedRecSplitFedAvg.aggregate_evaluate sums sufficient stats and divides once (BSL-06 / D-26)"
    - "set(GLOBAL_PARAM_KEYS) == set(_GLOBAL_PARAMS); set(LOCAL_PARAM_KEYS) == set(_LOCAL_PARAMS) (Pitfall 1 guard)"
  artifacts:
    - path: "federated-pfedrec/federated_pfedrec/strategy.py"
      provides: "PFedRecSplitFedAvg class + GLOBAL_PARAM_KEYS / LOCAL_PARAM_KEYS frozensets per D-01"
      contains: "class PFedRecSplitFedAvg(BaseFedAvg)"
    - path: "federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py"
      provides: "PFedRecMLP with D-01 param tuples + D-21 strict=True default"
      contains: "_GLOBAL_PARAMS = ('embedding_item.weight', 'affine_output.bias')"
    - path: "federated-pfedrec/tests/test_strategy.py"
      provides: "Strategy + frozenset symmetry tests (Pitfall 1 guard)"
    - path: "federated-pfedrec/tests/test_pfedrec_mlp.py"
      provides: "Model param tuple + strict-load + Kaiming init regression tests"
  key_links:
    - from: "federated_pfedrec.strategy.GLOBAL_PARAM_KEYS"
      to: "federated_pfedrec.models.pfedrec_mlp.PFedRecMLP._GLOBAL_PARAMS"
      via: "test_strategy.py::test_global_param_keys_matches_model_tuple"
      pattern: "set\\(GLOBAL_PARAM_KEYS\\) == set\\(PFedRecMLP._GLOBAL_PARAMS\\)"
    - from: "federated_pfedrec.strategy.PFedRecSplitFedAvg.aggregate_evaluate"
      to: "fedrec_foundation.fit_metrics.EvaluateMetricsContract sufficient stats"
      via: "_sum_sufficient_stats + _sufficient_stats_to_thesis_metrics module helpers"
      pattern: "hit_count_overall_at10|ndcg_sum_overall_at10|evaluated_users"
---

<objective>
Strategy + model parameter classification flip per PFR-02 D-01 (the highest-impact divergence audit decision).

Purpose:
  - Close PFR-02 row 1 (`affine_output.bias` GLOBAL not LOCAL) — the headline `IJCAI-23-PFedRec/engine.py:143` divergence and the strongest single lever for landing PFR-08 within ±2 points.
  - Close PFR-03 model-side contract (D-21 strict=True hard-fail with rm -rf hint).
  - Close PFR-02 audit row 2 (uniform weight policy via sufficient-stat aggregate_evaluate clone of Phase 3).
  - Drop FedProx (D-07) — reference uses FedAvg only; cross-module ablation surface unchanged.
  - Rename `SplitFedAvg` to `PFedRecSplitFedAvg` (D-12) for cross-module module-prefixed naming.

Output:
  - `federated-pfedrec/federated_pfedrec/strategy.py`: rewritten with `PFedRecSplitFedAvg` (no FedProx variant), `GLOBAL_PARAM_KEYS = frozenset({'embedding_item.weight', 'affine_output.bias'})`, `LOCAL_PARAM_KEYS = frozenset({'affine_output.weight'})`, plus Phase-3-style `aggregate_evaluate` override using `_sum_sufficient_stats` / `_sufficient_stats_to_thesis_metrics` module-level helpers.
  - `federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py`: `_GLOBAL_PARAMS = ('embedding_item.weight', 'affine_output.bias')`, `_LOCAL_PARAMS = ('affine_output.weight',)`, `set_local_parameters(strict=True)` default with RuntimeError + per-field delta + literal `rm -rf .embedding_cache/{run_id}/` hint.
  - 2 new test files (~6 GREEN tests covering D-01 symmetry, D-12 rename, D-07 removal, D-21 strict, D-19 Kaiming init, sum-not-mean aggregate_evaluate).
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/PROJECT.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/ROADMAP.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/STATE.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-RESEARCH.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/phases/05-pfedrec-migration-reproduction/05-VALIDATION.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.planning/codebase/CONCERNS.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/IJCAI-23-PFedRec/engine.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/IJCAI-23-PFedRec/mlp.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec/federated_pfedrec/strategy.py
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py

<interfaces>
<!-- Phase 3 PersonalizedSplitFedAvg pattern (clone for PFedRecSplitFedAvg) -->
<!-- Source: federated-personalized-cf/federated_personalized_cf/strategy.py -->
<!-- The aggregate_evaluate override + module-level _sum_sufficient_stats / _sufficient_stats_to_thesis_metrics helpers are the canonical sufficient-stat aggregator pattern. PFedRec clones this verbatim with the GLOBAL/LOCAL frozensets flipped per D-01. -->

```python
# fedrec_foundation contract (Phase 1 Plan 03/04)
from fedrec_foundation.fit_metrics import (
    EvaluateMetricsContract,           # 15+ field wire payload
    validate_evaluate_metrics,
)

# EvaluateMetricsContract.to_dict() emits these keys (excerpts):
# - hit_count_overall_at10: int
# - ndcg_sum_overall_at10: float
# - evaluated_users: int
# - hit_count_sparse_at10 / hit_count_medium_at10 / hit_count_dense_at10
# - ndcg_sum_sparse_at10 / ndcg_sum_medium_at10 / ndcg_sum_dense_at10
# - evaluated_users_sparse / evaluated_users_medium / evaluated_users_dense
# - eval_loss: Optional[float]   (diagnostic, optional)
# - sampled_hr_at10: Optional[float]  (diagnostic, optional)
# - sampled_ndcg_at10: Optional[float]  (diagnostic, optional)
# - partition_id: Optional[int]   (G-03-01 carry-forward)
```

```python
# IJCAI-23-PFedRec/engine.py:66-81 (D-24 source-of-truth — uniform mean)
def aggregate_clients_params(self, round_user_params):
    t = 0
    for user in round_user_params.keys():
        user_params = round_user_params[user]
        if t == 0:
            self.server_model_param = copy.deepcopy(user_params)
        else:
            for key in user_params.keys():
                self.server_model_param[key].data += user_params[key].data
        t += 1
    for key in self.server_model_param.keys():
        self.server_model_param[key].data = self.server_model_param[key].data / len(round_user_params)
```

```python
# IJCAI-23-PFedRec/engine.py:143 (D-01 source-of-truth — only weight is deleted; bias stays GLOBAL)
del round_participant_params[user]['affine_output.weight']  # bias NOT deleted -> aggregated server-side
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Rewrite strategy.py with PFedRecSplitFedAvg + D-01 frozensets + sufficient-stat aggregate_evaluate; ship test_strategy.py with 4 GREEN tests</name>
  <files>federated-pfedrec/federated_pfedrec/strategy.py, federated-pfedrec/tests/test_strategy.py</files>
  <read_first>
    - federated-pfedrec/federated_pfedrec/strategy.py — current SplitFedAvg + SplitFedProx state to replace
    - federated-personalized-cf/federated_personalized_cf/strategy.py — Phase 3 PersonalizedSplitFedAvg pattern + _sum_sufficient_stats / _sufficient_stats_to_thesis_metrics helpers (clone verbatim with frozensets flipped)
    - IJCAI-23-PFedRec/engine.py (lines 66-81 + 143) — D-01 + D-24 source of truth
    - .planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md §decisions D-01, D-07, D-12, D-24, D-26
    - scripts/foundation/fedrec_foundation/fit_metrics.py — EvaluateMetricsContract.to_dict() field names
  </read_first>
  <behavior>
    - Test 1 (test_strategy_class_renamed_pfedrecsplitfedavg): import `from federated_pfedrec.strategy import PFedRecSplitFedAvg`; assert class exists; assert old name `SplitFedAvg` is NOT importable (`with pytest.raises(ImportError): from federated_pfedrec.strategy import SplitFedAvg`); assert `PFedRecSplitFedProx` is NOT in module dir (D-07 drop).
    - Test 2 (test_global_param_keys_includes_bias): `from federated_pfedrec.strategy import GLOBAL_PARAM_KEYS, LOCAL_PARAM_KEYS`; assert `GLOBAL_PARAM_KEYS == frozenset({'embedding_item.weight', 'affine_output.bias'})`; assert `LOCAL_PARAM_KEYS == frozenset({'affine_output.weight'})`; assert disjoint.
    - Test 3 (test_local_param_keys_excludes_bias): assert `'affine_output.bias' not in LOCAL_PARAM_KEYS`; assert `'affine_output.bias' in GLOBAL_PARAM_KEYS`.
    - Test 4 (test_aggregate_evaluate_sufficient_stat_uniform): build 3 mock EvaluateRes with `hit_count_overall_at10` ∈ {1, 0, 1}, `ndcg_sum_overall_at10` ∈ {0.6, 0.0, 0.4}, `evaluated_users` = {1, 1, 1}, `eval_loss` ∈ {0.5, 0.7, 0.3}; call `PFedRecSplitFedAvg().aggregate_evaluate(round_num=1, results=[...], failures=[])`; assert returned thesis dict has `hit_rate_at10 == 2/3`, `sampled_ndcg_at10 == 1.0/3.0`; assert returned loss == 0.5 (mean of eval_loss).
  </behavior>
  <action>
Replace `federated-pfedrec/federated_pfedrec/strategy.py` ENTIRELY with the following file. Preserve no legacy classes:

```python
"""PFedRec split-learning strategy with D-01 bias-GLOBAL classification.

Phase 5 changes vs prior SplitFedAvg:
- D-01: `affine_output.bias` moves from LOCAL_PARAM_KEYS to GLOBAL_PARAM_KEYS.
  Source of truth: IJCAI-23-PFedRec/engine.py:143 deletes ONLY
  `affine_output.weight` from per-user dict before aggregation; bias is
  aggregated server-side. Closes CONCERNS divergence #9.
- D-07: SplitFedProx variant DROPPED. Reference uses FedAvg only; PFedRec's
  per-user score function does not benefit from a global proximal term.
- D-12: Class renamed to PFedRecSplitFedAvg (module-prefix convention).
- D-24/D-26: aggregate_evaluate sums sufficient stats and divides once
  (uniform weighting; matches engine.py:81 `len(round_user_params)`).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from flwr.server.strategy import FedAvg as BaseFedAvg


# D-01: bias is GLOBAL (engine.py:143 only deletes affine_output.weight).
GLOBAL_PARAM_KEYS = frozenset({
    'embedding_item.weight',
    'affine_output.bias',
})

# D-01: only affine_output.weight is LOCAL (per-user score function).
LOCAL_PARAM_KEYS = frozenset({
    'affine_output.weight',
})


# Module-level sufficient-stat helpers (Phase 3 PersonalizedSplitFedAvg pattern).
# These produce the same headline (hit_rate_at10, sampled_ndcg_at10) regardless
# of the FitRes.num_examples weighting because they sum sufficient stats and
# divide ONCE at the end.

_SUFF_STAT_FIELDS = (
    'hit_count_overall_at10',
    'ndcg_sum_overall_at10',
    'evaluated_users',
    'hit_count_sparse_at10', 'ndcg_sum_sparse_at10', 'evaluated_users_sparse',
    'hit_count_medium_at10', 'ndcg_sum_medium_at10', 'evaluated_users_medium',
    'hit_count_dense_at10',  'ndcg_sum_dense_at10',  'evaluated_users_dense',
)


def _sum_sufficient_stats(metrics_list: List[Dict]) -> Dict[str, float]:
    """Sum the 12 sufficient-stat fields across all client metric dicts.

    Parameters
    ----------
    metrics_list : List[Dict]
        Per-client metric dicts (each carrying the EvaluateMetricsContract keys).

    Returns
    -------
    Dict[str, float]
        Map field-name -> summed value across clients.
    """
    sums: Dict[str, float] = {f: 0.0 for f in _SUFF_STAT_FIELDS}
    for m in metrics_list:
        for f in _SUFF_STAT_FIELDS:
            sums[f] += float(m.get(f, 0) or 0)
    return sums


def _sufficient_stats_to_thesis_metrics(sums: Dict[str, float]) -> Dict[str, float]:
    """Convert summed sufficient stats to ratio-style thesis metrics.

    Returns
    -------
    Dict[str, float]
        Keys: sampled_hr@10, sampled_ndcg@10, evaluated_users plus per-group
        variants (sparse/medium/dense). Zero-safe (returns 0.0 on zero
        denominator).
    """

    def _ratio(num: float, den: float) -> float:
        return float(num) / float(den) if den else 0.0

    out: Dict[str, float] = {
        'sampled_hr@10': _ratio(sums['hit_count_overall_at10'], sums['evaluated_users']),
        'sampled_ndcg@10': _ratio(sums['ndcg_sum_overall_at10'], sums['evaluated_users']),
        'evaluated_users': float(sums['evaluated_users']),
    }
    for g in ('sparse', 'medium', 'dense'):
        out[f'sampled_hr@10/{g}'] = _ratio(
            sums[f'hit_count_{g}_at10'], sums[f'evaluated_users_{g}']
        )
        out[f'sampled_ndcg@10/{g}'] = _ratio(
            sums[f'ndcg_sum_{g}_at10'], sums[f'evaluated_users_{g}']
        )
        out[f'evaluated_users/{g}'] = float(sums[f'evaluated_users_{g}'])
    return out


class PFedRecSplitFedAvg(BaseFedAvg):
    """FedAvg variant for PFedRec with D-01 bias-GLOBAL + uniform weighting.

    Notes
    -----
    - aggregate_fit is INHERITED from BaseFedAvg unchanged. Phase 5 server_app.py
      (Plan 04) sets `FitRes.num_examples = 1` per client (uniform under D-24)
      so FedAvg's existing num_examples-weighted average is mathematically
      uniform — no aggregate_fit override needed.
    - aggregate_evaluate is OVERRIDDEN to sum sufficient stats and compute the
      ratio once at the end (BSL-06 / PSN-04 / ADP-06 carry-forward).
    """

    def __init__(self, fraction_fit: float = 1.0, **kwargs):
        super().__init__(fraction_fit=fraction_fit, **kwargs)
        self.global_param_keys = GLOBAL_PARAM_KEYS
        self.local_param_keys = LOCAL_PARAM_KEYS
        self._is_split_learning = True

    def __repr__(self) -> str:
        return f'PFedRecSplitFedAvg(fraction_fit={self.fraction_fit})'

    def aggregate_evaluate(self, server_round, results, failures):
        """Sum sufficient stats + compute ratio once. Returns (loss, thesis_metrics)."""
        if not results:
            return None, {}
        metric_dicts = []
        eval_losses = []
        for client_proxy, eval_res in results:
            m = dict(eval_res.metrics) if hasattr(eval_res, 'metrics') else dict(eval_res)
            metric_dicts.append(m)
            if 'eval_loss' in m and m['eval_loss'] is not None:
                eval_losses.append(float(m['eval_loss']))
        sums = _sum_sufficient_stats(metric_dicts)
        thesis = _sufficient_stats_to_thesis_metrics(sums)
        loss = (sum(eval_losses) / len(eval_losses)) if eval_losses else 0.0
        return loss, thesis


__all__ = [
    'PFedRecSplitFedAvg',
    'GLOBAL_PARAM_KEYS',
    'LOCAL_PARAM_KEYS',
    '_sum_sufficient_stats',
    '_sufficient_stats_to_thesis_metrics',
]
```

Then create `federated-pfedrec/tests/test_strategy.py` with 4 tests (Test 1-4 from the behavior block above). Use `unittest.mock.MagicMock` for EvaluateRes objects. Build a tiny conftest.py at `federated-pfedrec/tests/conftest.py` if it does not yet exist (one-line `# Phase 5 test infra placeholder` is fine; Plan 02 may extend it).

Verify by running `cd federated-pfedrec && pip install -e ".[dev]"` (after Plan 02 lands the dev extra; if not yet, install pytest manually for now) and `pytest tests/test_strategy.py -x -v`. All 4 tests MUST pass.
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec && pytest tests/test_strategy.py -x -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "class PFedRecSplitFedAvg(BaseFedAvg)" federated-pfedrec/federated_pfedrec/strategy.py` returns 1
    - `grep -c "class SplitFedAvg" federated-pfedrec/federated_pfedrec/strategy.py` returns 0
    - `grep -c "class PFedRecSplitFedProx" federated-pfedrec/federated_pfedrec/strategy.py` returns 0
    - `grep -c "class SplitFedProx" federated-pfedrec/federated_pfedrec/strategy.py` returns 0
    - `grep -c "'embedding_item.weight'" federated-pfedrec/federated_pfedrec/strategy.py` returns at least 1 inside the GLOBAL_PARAM_KEYS frozenset literal
    - `grep -c "'affine_output.bias'" federated-pfedrec/federated_pfedrec/strategy.py` returns at least 1 inside the GLOBAL_PARAM_KEYS frozenset literal
    - `grep -c "'affine_output.weight'" federated-pfedrec/federated_pfedrec/strategy.py` returns at least 1 inside the LOCAL_PARAM_KEYS frozenset literal
    - `grep -c "_sum_sufficient_stats" federated-pfedrec/federated_pfedrec/strategy.py` returns at least 2 (def + call site)
    - `grep -c "def aggregate_evaluate" federated-pfedrec/federated_pfedrec/strategy.py` returns 1
    - `grep -c "BaseFedProx" federated-pfedrec/federated_pfedrec/strategy.py` returns 0 (D-07 — no FedProx import)
    - `pytest federated-pfedrec/tests/test_strategy.py -x -v` exits 0 with 4 tests passed
  </acceptance_criteria>
  <done>
    - strategy.py rewritten: PFedRecSplitFedAvg replaces SplitFedAvg, SplitFedProx removed (D-07), GLOBAL_PARAM_KEYS includes affine_output.bias (D-01), aggregate_evaluate sums sufficient stats per BSL-06/D-26
    - 4 GREEN tests in test_strategy.py covering D-12 rename, D-01 frozensets, D-07 FedProx removal, D-24 sufficient-stat aggregation
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Update PFedRecMLP _GLOBAL_PARAMS / _LOCAL_PARAMS tuples + strict=True default; ship test_pfedrec_mlp.py with 3 GREEN tests including D-01 / D-19 / D-21 coverage</name>
  <files>federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py, federated-pfedrec/tests/test_pfedrec_mlp.py</files>
  <read_first>
    - federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py — current PFedRecMLP state (D-18 surgical: forward/predict/init untouched)
    - federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py — Phase 3 D-21 strict=True idiom for set_local_parameters reference
    - .planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md §decisions D-01, D-19, D-20, D-21
    - IJCAI-23-PFedRec/mlp.py — Kaiming-default init (no Xavier) reference for D-19
  </read_first>
  <behavior>
    - Test 1 (test_local_params_tuple_only_affine_weight): import PFedRecMLP; assert `PFedRecMLP._GLOBAL_PARAMS == ('embedding_item.weight', 'affine_output.bias')`; assert `PFedRecMLP._LOCAL_PARAMS == ('affine_output.weight',)`; assert `'affine_output.bias' not in PFedRecMLP._LOCAL_PARAMS`; build a model, call `model.get_local_parameters()`, assert returned dict has exactly one key `'affine_output.weight'` with shape `(1, latent_dim)` (D-20 native PyTorch shape).
    - Test 2 (test_set_local_parameters_strict_true_hard_fails): build model with latent_dim=32; construct a state_dict with `affine_output.weight` shape `(1, 16)` (deliberately wrong); call `model.set_local_parameters(bad_state)` with default strict=True; assert raises `RuntimeError` whose message contains all 4 substrings: `'affine_output.weight'`, `'(1, 16)'` (saved shape), `'(1, 32)'` (current shape), AND `'rm -rf'` (literal hint per D-21).
    - Test 3 (test_kaiming_default_init_paper_faithful): build PFedRecMLP(num_items=100, latent_dim=32); confirm `model.affine_output.weight` is initialized to PyTorch nn.Linear default (Kaiming-uniform). Assert the weight tensor is non-zero AND that its standard deviation is in the Kaiming range `sqrt(1/in_features) ≈ 0.177` plus a 5x slack band (i.e. `0.0 < weight.std() < 1.0`). Assert NO Xavier-uniform reset has been applied (the existing module-level forbidden token `nn.init.xavier_uniform_` does not appear in the source under D-19).
  </behavior>
  <action>
Modify `federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py` with surgical edits (D-18 — preserve forward/predict/__init__ verbatim except for the param-tuple constants and `set_local_parameters` body):

1. Update class-level constants at line 36-37 to:
```python
    _GLOBAL_PARAMS = ('embedding_item.weight', 'affine_output.bias')
    _LOCAL_PARAMS = ('affine_output.weight',)
```

2. Update the module docstring at lines 11-17 to reflect the D-01 flip:
```
Split Learning Parameter Classification:
    GLOBAL (aggregated via FedAvg):
        - embedding_item.weight: Item latent factors
        - affine_output.bias: User score function bias (D-01:
          IJCAI-23-PFedRec/engine.py:143 deletes only `affine_output.weight`
          before aggregation, so bias is aggregated server-side; updated from
          prior LOCAL classification to align with reference).

    LOCAL (private, per-user, never sent to server):
        - affine_output.weight: User's personalized score function weight only.
```

3. Replace the `set_local_parameters` body (current lines 140-183) with a strict=True default implementation that hard-fails with rm -rf hint:

```python
    def set_local_parameters(
        self,
        local_state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
        run_id: str = "<run_id>",
    ) -> Tuple[List[str], List[str]]:
        """Load local parameters with D-21 strict=True hard-fail semantics.

        Parameters
        ----------
        local_state_dict : Dict[str, torch.Tensor]
            Saved local parameters. MUST contain exactly the keys in
            ``self._LOCAL_PARAMS`` (after D-01: only ``affine_output.weight``).
        strict : bool, optional
            If True (D-21 default), raise ``RuntimeError`` on shape mismatch
            with per-field delta and a literal ``rm -rf .embedding_cache/{run_id}/``
            hint. If False (legacy back-compat — NOT used by Phase 5
            client_app), partial-load and report missing keys.
        run_id : str, optional
            Threaded through into the rm -rf hint when strict=True. Defaults
            to placeholder ``"<run_id>"``; client_app passes the real run_id.

        Returns
        -------
        Tuple[List[str], List[str]]
            (loaded_keys, missing_keys). Under strict=True, missing_keys is
            always empty when this method returns (errors raise instead).

        Raises
        ------
        RuntimeError
            (strict=True) When any LOCAL key is missing from ``local_state_dict``
            or when its shape does not match the live model parameter.
        """
        loaded_keys: List[str] = []
        missing_keys: List[str] = []
        current_state = self.state_dict()

        for name in self._LOCAL_PARAMS:
            if name not in local_state_dict:
                if strict:
                    raise RuntimeError(
                        f"D-21 missing local key {name!r} on cache load. "
                        f"Run: rm -rf .embedding_cache/{run_id}/"
                    )
                missing_keys.append(name)
                continue

            saved = local_state_dict[name]
            current = current_state[name]
            if saved.shape != current.shape:
                if strict:
                    raise RuntimeError(
                        f"D-21 shape mismatch for {name!r}: "
                        f"saved shape {tuple(saved.shape)} vs current shape "
                        f"{tuple(current.shape)}. "
                        f"Run: rm -rf .embedding_cache/{run_id}/"
                    )
                missing_keys.append(name)
                continue

            current_state[name] = saved
            loaded_keys.append(name)

        self.load_state_dict(current_state, strict=True)
        return loaded_keys, missing_keys
```

4. CONFIRM (do NOT add) that nowhere in `pfedrec_mlp.py` does the source contain `nn.init.xavier_uniform_` or `nn.init.xavier_normal_` — D-19 forbids Xavier; PyTorch's nn.Linear / nn.Embedding default Kaiming-uniform init is paper-faithful.

Then create `federated-pfedrec/tests/test_pfedrec_mlp.py` implementing Test 1, Test 2, Test 3 from the behavior block. Use `pytest.raises(RuntimeError, match=...)` with regex assertions for all 4 substrings in Test 2.

Verify: `cd federated-pfedrec && pytest tests/test_pfedrec_mlp.py -x -v` — 3 tests pass.
  </action>
  <verify>
    <automated>cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/federated-pfedrec && pytest tests/test_pfedrec_mlp.py -x -v</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "_GLOBAL_PARAMS = ('embedding_item.weight', 'affine_output.bias')" federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py` returns 1
    - `grep -c "_LOCAL_PARAMS = ('affine_output.weight',)" federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py` returns 1
    - `grep -c "strict: bool = True" federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py` returns 1 (D-21 default flipped)
    - `grep -c "D-21" federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py` returns at least 2
    - `grep -c "rm -rf" federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py` returns at least 1
    - `grep -c "xavier_uniform_\|xavier_normal_" federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py` returns 0 (D-19 paper-faithful Kaiming default)
    - `pytest federated-pfedrec/tests/test_pfedrec_mlp.py -x -v` exits 0 with 3 tests passed
  </acceptance_criteria>
  <done>
    - PFedRecMLP._GLOBAL_PARAMS includes affine_output.bias (D-01); _LOCAL_PARAMS contains only affine_output.weight
    - set_local_parameters defaults to strict=True; raises RuntimeError with per-field delta + literal rm -rf hint on shape mismatch (D-21)
    - 3 GREEN tests confirm D-01 tuple shapes, D-21 strict-fail behavior, D-19 Kaiming default preserved
  </done>
</task>

</tasks>

<verification>
- Strategy.py imports cleanly: `python -c "from federated_pfedrec.strategy import PFedRecSplitFedAvg, GLOBAL_PARAM_KEYS, LOCAL_PARAM_KEYS"` exits 0
- Model.py imports cleanly: `python -c "from federated_pfedrec.models.pfedrec_mlp import PFedRecMLP; m = PFedRecMLP(num_items=100, latent_dim=32); assert tuple(m.get_local_parameters().keys()) == ('affine_output.weight',); print('ok')"` prints "ok"
- Pitfall 1 symmetry guard: `python -c "from federated_pfedrec.strategy import GLOBAL_PARAM_KEYS, LOCAL_PARAM_KEYS; from federated_pfedrec.models.pfedrec_mlp import PFedRecMLP; assert set(GLOBAL_PARAM_KEYS) == set(PFedRecMLP._GLOBAL_PARAMS); assert set(LOCAL_PARAM_KEYS) == set(PFedRecMLP._LOCAL_PARAMS); print('ok')"`
- Total tests added: 4 (test_strategy.py) + 3 (test_pfedrec_mlp.py) = 7 GREEN
- D-18 surgical scope: `git diff --stat` shows ONLY changes to strategy.py + pfedrec_mlp.py + 2 new test files; pyproject.toml / dataset.py / client_app.py / server_app.py / task.py UNTOUCHED (those are owned by Plans 02-04)
</verification>

<success_criteria>
- strategy.py: PFedRecSplitFedAvg class, GLOBAL_PARAM_KEYS frozenset has both 'embedding_item.weight' AND 'affine_output.bias', LOCAL_PARAM_KEYS frozenset has only 'affine_output.weight', NO SplitFedProx class, NO BaseFedProx import, aggregate_evaluate override sums sufficient stats and computes ratio once
- pfedrec_mlp.py: _GLOBAL_PARAMS = ('embedding_item.weight', 'affine_output.bias'); _LOCAL_PARAMS = ('affine_output.weight',); set_local_parameters defaults strict=True with RuntimeError + rm -rf hint; no Xavier init reset (D-19)
- 7 GREEN tests across 2 new test files
- Pitfall 1 symmetry preserved: strategy frozensets and model param tuples agree element-wise
- Files outside the listed `files_modified` list remain byte-identical to pre-Plan-01 state (D-18 surgical)
</success_criteria>

<output>
After completion, create `.planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-01-SUMMARY.md` covering:
- D-01 bias-GLOBAL flip + D-12 rename + D-07 FedProx drop + D-21 strict default
- Test counts and which decisions each test pins
- Confirmation Pitfall 1 (strategy/model frozenset symmetry) is mechanically enforced by test_strategy.py
</output>
