---
phase: 03-personalized-migration
plan: 01
type: execute
subsystem: infra
tags: [strategy, personalized-split-fedavg, personalized-split-fedprox, model-single-row, bpr-mf, basic-mf, local-user-row, sufficient-stats, psn-06, d-01, d-02, d-03, d-20, d-23, tdd, wave-1]
wave: 1
depends_on: []
files_modified:
  - federated-personalized-cf/federated_personalized_cf/strategy.py
  - federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py
  - federated-personalized-cf/federated_personalized_cf/models/basic_mf.py
  - federated-personalized-cf/tests/__init__.py
  - federated-personalized-cf/tests/conftest.py
  - federated-personalized-cf/tests/test_strategy.py
  - federated-personalized-cf/tests/test_single_row_model.py
autonomous: true
requirements: [PSN-06]

must_haves:
  truths:
    - "PersonalizedSplitFedAvg(BaseFedAvg) and PersonalizedSplitFedProx(BaseFedProx) subclasses exist in federated-personalized-cf/federated_personalized_cf/strategy.py and override aggregate_evaluate to emit thesis metrics from summed sufficient stats (sum(hit_count)/sum(evaluated_users)) instead of averaging per-client ratios."
    - "Module-level frozensets _GLOBAL_PARAM_KEYS = {item_embeddings.weight, item_bias.weight, global_bias} and _LOCAL_PARAM_KEYS = {local_user_row, local_user_bias} are declared in strategy.py; aggregate_fit is INHERITED UNCHANGED from parent FedAvg/FedProx (D-23)."
    - "BPRMF model's user-row representation collapses from nn.Embedding(num_users, d) to two nn.Parameter tensors: local_user_row (shape=(d,)) and local_user_bias (shape=(1,) or scalar); _LOCAL_PARAMS tuple = ('local_user_row', 'local_user_bias'); forward() no longer requires user_ids — the client IS the user."
    - "BasicMF model mirrors BPRMF's single-row refactor with the same _LOCAL_PARAMS = ('local_user_row', 'local_user_bias') contract."
    - "get_local_parameters() returns OrderedDict([('local_user_row', tensor(d,)), ('local_user_bias', tensor(1,))]) — the on-disk payload is per-client-single-row, never the old (num_users, d) blob."
    - "federated-personalized-cf/tests/ pytest package exists with pytest fixtures (fake_evaluate_res, fake_client_proxy) and 8+ GREEN tests covering strategy sufficient-stat aggregation (5 tests) + single-row model contract (3+ tests)."
  artifacts:
    - path: "federated-personalized-cf/federated_personalized_cf/strategy.py"
      provides: "PersonalizedSplitFedAvg + PersonalizedSplitFedProx with aggregate_evaluate override, _GLOBAL_PARAM_KEYS + _LOCAL_PARAM_KEYS frozensets, _sum_sufficient_stats + _sufficient_stats_to_thesis_metrics module-level helpers"
      contains: "class PersonalizedSplitFedAvg, class PersonalizedSplitFedProx, frozenset({'local_user_row', 'local_user_bias'})"
    - path: "federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py"
      provides: "BPRMF refactored: local_user_row nn.Parameter + local_user_bias nn.Parameter; forward()/forward_item_only() no longer takes user_ids; get/set_local_parameters use 2-key dict"
      contains: "self.local_user_row = nn.Parameter, self.local_user_bias = nn.Parameter, _LOCAL_PARAMS_WITH_BIAS = ('local_user_row', 'local_user_bias')"
    - path: "federated-personalized-cf/federated_personalized_cf/models/basic_mf.py"
      provides: "BasicMF refactored with same single-row contract as BPRMF"
      contains: "self.local_user_row = nn.Parameter"
    - path: "federated-personalized-cf/tests/test_strategy.py"
      provides: "5 GREEN pytest tests mirroring baseline test_strategy.py (sufficient-stat sums, per-group ratios, zero-division, FedProx inherit, aggregate_fit identity check)"
    - path: "federated-personalized-cf/tests/test_single_row_model.py"
      provides: "3+ GREEN pytest tests verifying D-01 refactor (single-row shape, _LOCAL_PARAMS contract, absence of nn.Embedding(num_users, d))"
  key_links:
    - from: "federated-personalized-cf/federated_personalized_cf/strategy.py"
      to: "fedrec_foundation.fit_metrics.EvaluateMetricsContract"
      via: "_sum_sufficient_stats reads the 12 _at10 / evaluated_users keys emitted client-side"
      pattern: "hit_count_overall_at10|ndcg_sum_overall_at10|evaluated_users"
    - from: "federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py"
      to: "federated-personalized-cf/federated_personalized_cf/strategy.py"
      via: "_LOCAL_PARAM_KEYS frozenset in strategy.py matches tuple on the model"
      pattern: "'local_user_row'"
---

<objective>
Lay the Phase 3 foundation in parallel with Plan 02: ship the PersonalizedSplitFedAvg/FedProx sufficient-stat aggregator (mirroring Plan 02 Phase-2 Plan 01) with a flipped GLOBAL/LOCAL param split (item_* GLOBAL; local_user_* LOCAL), and collapse the BPRMF / BasicMF user-row representation from an nn.Embedding(num_users, d) ghost table to a single-row nn.Parameter per D-01..D-03. Adds a pytest tests/ package with 8+ GREEN tests that fingerprint the new contract.

Purpose: Closes PSN-06 (single-row model, no ghost table) and stages PSN-04's server-side aggregation so Plan 04 can drop the new strategy into server_app.py without touching models. Preserves the split-learning D-23 invariant (aggregate_fit inherited unchanged) exactly as Plan 02 Phase-2 Plan 01 did.

Output:
- federated-personalized-cf/federated_personalized_cf/strategy.py (rewritten)
- federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py (D-01 refactor)
- federated-personalized-cf/federated_personalized_cf/models/basic_mf.py (D-01 refactor)
- federated-personalized-cf/tests/{__init__.py, conftest.py, test_strategy.py, test_single_row_model.py}
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@.planning/phases/03-personalized-migration/03-CONTEXT.md
@.planning/phases/02-baseline-migration/02-baseline-migration-01-SUMMARY.md

<interfaces>
<!-- Baseline module template to clone (federated-baseline-cf/federated_baseline_cf/strategy.py) -->
```python
# Lines ~27-105 — the module-level helpers and constants already shipped in Phase 2
from flwr.server.strategy import FedAvg as BaseFedAvg, FedProx as BaseFedProx

_SUFFICIENT_STAT_KEYS = (
    "hit_count_overall_at10", "ndcg_sum_overall_at10", "evaluated_users",
    "hit_count_sparse_at10", "ndcg_sum_sparse_at10", "evaluated_users_sparse",
    "hit_count_medium_at10", "ndcg_sum_medium_at10", "evaluated_users_medium",
    "hit_count_dense_at10",  "ndcg_sum_dense_at10",  "evaluated_users_dense",
)

def _sum_sufficient_stats(results):
    totals = {k: 0 for k in _SUFFICIENT_STAT_KEYS}
    for _proxy, eval_res in results:
        metrics = eval_res.metrics or {}
        for k in _SUFFICIENT_STAT_KEYS:
            totals[k] += metrics.get(k, 0) or 0
    return totals

def _sufficient_stats_to_thesis_metrics(totals):
    def _safe_ratio(num, den): return (num / den) if den else 0.0
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

class BaselineFedAvg(BaseFedAvg):
    def aggregate_evaluate(self, server_round, results, failures):
        if not results: return (None, {})
        loss_num = sum(r.loss * r.num_examples for _, r in results)
        loss_den = sum(r.num_examples for _, r in results) or 1
        totals = _sum_sufficient_stats(results)
        return (loss_num / loss_den, _sufficient_stats_to_thesis_metrics(totals))
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: PersonalizedSplitFedAvg + PersonalizedSplitFedProx strategy subclasses with sufficient-stat aggregate_evaluate (D-20, D-23, PSN-06 strategy half)</name>
  <files>federated-personalized-cf/federated_personalized_cf/strategy.py, federated-personalized-cf/tests/__init__.py, federated-personalized-cf/tests/conftest.py, federated-personalized-cf/tests/test_strategy.py</files>
  <read_first>
    - federated-personalized-cf/federated_personalized_cf/strategy.py (PRE-EXISTING content that will be REPLACED; read to preserve WIP patterns per D-18)
    - federated-baseline-cf/federated_baseline_cf/strategy.py (CANONICAL TEMPLATE to mirror; the `_sum_sufficient_stats`, `_sufficient_stats_to_thesis_metrics`, `BaselineFedAvg`, `BaselineFedProx` shape is copied nearly verbatim — ONLY the frozenset contents flip)
    - federated-baseline-cf/tests/conftest.py (fake_evaluate_res + fake_client_proxy fixtures — copy EXACTLY to personalized tests/)
    - federated-baseline-cf/tests/test_strategy.py (5 test names + arithmetic to clone with strategy class names substituted)
    - scripts/foundation/fedrec_foundation/fit_metrics.py (EvaluateMetricsContract + 12 sufficient-stat keys — these are the keys the strategy reads)
    - .planning/phases/02-baseline-migration/02-baseline-migration-01-SUMMARY.md (context on the iter-2 `_at10` lock-in and the `aggregate_fit is FedAvg.aggregate_fit` identity invariant)
  </read_first>
  <behavior>
    Tests to write FIRST (RED), then implement:
    - test_aggregate_evaluate_sums_sufficient_stats: 3 clients with overall hit_count=(10, 5, 7), evaluated_users=(20, 15, 25) → PersonalizedSplitFedAvg.aggregate_evaluate returns sampled_hr@10 ≈ 22/60 ≈ 0.3667, evaluated_users == 60.
    - test_aggregate_evaluate_per_group_ratios: 2 clients with per-group sparse/medium/dense hit + evaluated_users → asserts 3 per-group ratios + 3 per-group evaluated_users match arithmetic.
    - test_aggregate_evaluate_zero_division_safe: 1 client with evaluated_users_sparse=0 → sampled_hr@10/sparse == 0.0 (no ZeroDivisionError).
    - test_personalized_split_fedprox_inherits_aggregate_evaluate: instantiate PersonalizedSplitFedProx(fraction_fit=0.1, proximal_mu=0.01) and verify sum-based ratio logic still works.
    - test_aggregate_fit_inherited_unchanged: identity check PersonalizedSplitFedAvg.aggregate_fit is BaseFedAvg.aggregate_fit (D-23 split-learning invariant — only evaluate-side is overridden; fit-side aggregation of GLOBAL params flows through the parent untouched).
  </behavior>
  <action>
    Step 1 — Rip-and-replace federated-personalized-cf/federated_personalized_cf/strategy.py with the sufficient-stat aggregator. Use this exact skeleton (clone from baseline then flip frozensets):

    ```python
    """Split Learning Strategies for Federated Personalized Collaborative Filtering (Phase 3 Plan 01).

    PersonalizedSplitFedAvg / PersonalizedSplitFedProx subclass Flower's FedAvg/FedProx
    and override aggregate_evaluate to compute thesis metrics ONCE from summed sufficient
    stats (sum(hit_count)/sum(evaluated_users)) instead of averaging per-client ratios.

    aggregate_fit is INHERITED UNCHANGED from parent (D-23 split-learning invariant —
    the client only sends GLOBAL params so FedAvg's weighted average of GLOBAL params is correct).

    Parameter split vs Phase 2 baseline:
      - baseline: ALL params GLOBAL (aggregate_fit averages everything)
      - personalized: item_* GLOBAL, local_user_* LOCAL (D-03, aggregate_fit averages only GLOBAL)
    """
    from typing import Dict, List, Tuple
    from flwr.common import EvaluateRes
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import FedAvg as BaseFedAvg, FedProx as BaseFedProx

    # D-03: flipped frozensets vs baseline (item_* GLOBAL, local_user_* LOCAL)
    _GLOBAL_PARAM_KEYS = frozenset({
        "item_embeddings.weight",
        "item_bias.weight",
        "global_bias",
    })
    _LOCAL_PARAM_KEYS = frozenset({
        "local_user_row",
        "local_user_bias",
    })

    _SUFFICIENT_STAT_KEYS = (
        "hit_count_overall_at10", "ndcg_sum_overall_at10", "evaluated_users",
        "hit_count_sparse_at10", "ndcg_sum_sparse_at10", "evaluated_users_sparse",
        "hit_count_medium_at10", "ndcg_sum_medium_at10", "evaluated_users_medium",
        "hit_count_dense_at10",  "ndcg_sum_dense_at10",  "evaluated_users_dense",
    )

    def _sum_sufficient_stats(results: List[Tuple[ClientProxy, EvaluateRes]]) -> Dict[str, float]:
        totals: Dict[str, float] = {k: 0 for k in _SUFFICIENT_STAT_KEYS}
        for _proxy, eval_res in results:
            metrics = eval_res.metrics or {}
            for k in _SUFFICIENT_STAT_KEYS:
                totals[k] += metrics.get(k, 0) or 0
        return totals

    def _sufficient_stats_to_thesis_metrics(totals: Dict[str, float]) -> Dict[str, float]:
        def _safe_ratio(num, den): return (num / den) if den else 0.0
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

    class PersonalizedSplitFedAvg(BaseFedAvg):
        """FedAvg variant for split learning with sufficient-stat aggregate_evaluate (D-20, PSN-04 server half).

        GLOBAL params: item_embeddings.weight, item_bias.weight, global_bias.
        LOCAL params: local_user_row, local_user_bias (on client only; never aggregated).
        aggregate_fit is inherited UNCHANGED — parent FedAvg averages the GLOBAL params the client sends.
        """
        def aggregate_evaluate(self, server_round, results, failures):
            if not results:
                return (None, {})
            loss_num = sum(r.loss * r.num_examples for _, r in results)
            loss_den = sum(r.num_examples for _, r in results) or 1
            totals = _sum_sufficient_stats(results)
            return (loss_num / loss_den, _sufficient_stats_to_thesis_metrics(totals))

    class PersonalizedSplitFedProx(BaseFedProx):
        """FedProx variant that reuses the sum-based aggregate_evaluate (D-20).

        aggregate_evaluate is an EXACT COPY of PersonalizedSplitFedAvg.aggregate_evaluate
        (not super() call) to avoid diamond-inheritance with BaseFedProx; both use the
        module-level _sum_sufficient_stats + _sufficient_stats_to_thesis_metrics helpers.
        """
        def aggregate_evaluate(self, server_round, results, failures):
            if not results:
                return (None, {})
            loss_num = sum(r.loss * r.num_examples for _, r in results)
            loss_den = sum(r.num_examples for _, r in results) or 1
            totals = _sum_sufficient_stats(results)
            return (loss_num / loss_den, _sufficient_stats_to_thesis_metrics(totals))

    __all__ = [
        "PersonalizedSplitFedAvg",
        "PersonalizedSplitFedProx",
        "_GLOBAL_PARAM_KEYS",
        "_LOCAL_PARAM_KEYS",
    ]
    ```

    Step 2 — D-18 surgical guard: any PRE-EXISTING uncommitted WIP in strategy.py unrelated to the aggregator / frozensets (e.g. the pre-existing SplitFedAvg(fraction_fit) __init__ comment block) may be dropped as part of the rip-and-replace, BUT note in the commit message which pre-existing lines were removed. The Phase-2 precedent says "replace OUR helpers; DON'T touch the user's WIP" — here OUR helpers ARE the SplitFedAvg/SplitFedProx classes, so rip-and-replace is authorized.

    Step 3 — Create federated-personalized-cf/tests/__init__.py as an empty file (marker — makes tests/ a package).

    Step 4 — Create federated-personalized-cf/tests/conftest.py. COPY VERBATIM from federated-baseline-cf/tests/conftest.py: the fake_evaluate_res and fake_client_proxy fixtures. No changes beyond the module docstring reference (say "personalized" instead of "baseline" in the header).

    Step 5 — Create federated-personalized-cf/tests/test_strategy.py with the 5 tests named in <behavior>, cloned from federated-baseline-cf/tests/test_strategy.py with:
      - `from federated_baseline_cf.strategy import BaselineFedAvg, BaselineFedProx` → `from federated_personalized_cf.strategy import PersonalizedSplitFedAvg, PersonalizedSplitFedProx`
      - Class names substituted throughout
      - Identity check in test_aggregate_fit_inherited_unchanged uses `BaseFedAvg.aggregate_fit` (the parent) as the right-hand side

    Step 6 — Verify GREEN: `cd federated-personalized-cf && pip install -e .[dev] && pytest tests/test_strategy.py -v` → 5 passed. (Task 2 adds another file alongside; do not skip pytest on intermediate state.)

    Step 7 — Commit (--no-verify per Wave-1 parallel rule):
    ```
    git add federated-personalized-cf/federated_personalized_cf/strategy.py \
            federated-personalized-cf/tests/__init__.py \
            federated-personalized-cf/tests/conftest.py \
            federated-personalized-cf/tests/test_strategy.py
    git commit --no-verify -m "feat(03-01): PersonalizedSplitFedAvg + PersonalizedSplitFedProx (D-20, PSN-06)"
    ```
  </action>
  <acceptance_criteria>
    - `grep -c "^class PersonalizedSplitFedAvg" federated-personalized-cf/federated_personalized_cf/strategy.py` returns 1
    - `grep -c "^class PersonalizedSplitFedProx" federated-personalized-cf/federated_personalized_cf/strategy.py` returns 1
    - `grep -c "_GLOBAL_PARAM_KEYS = frozenset" federated-personalized-cf/federated_personalized_cf/strategy.py` returns 1
    - `grep -c "'local_user_row'" federated-personalized-cf/federated_personalized_cf/strategy.py` returns at least 1 (inside _LOCAL_PARAM_KEYS)
    - `grep -c "'item_embeddings.weight'" federated-personalized-cf/federated_personalized_cf/strategy.py` returns at least 1 (inside _GLOBAL_PARAM_KEYS)
    - `grep -c "'user_embeddings.weight'" federated-personalized-cf/federated_personalized_cf/strategy.py` returns 0 (old ghost-table key MUST be absent from new constants)
    - `grep -c "def _sum_sufficient_stats" federated-personalized-cf/federated_personalized_cf/strategy.py` returns 1
    - `grep -c "def _sufficient_stats_to_thesis_metrics" federated-personalized-cf/federated_personalized_cf/strategy.py` returns 1
    - `grep -c "def aggregate_fit" federated-personalized-cf/federated_personalized_cf/strategy.py` returns 0 (inherited UNCHANGED — D-23)
    - `cd federated-personalized-cf && pytest tests/test_strategy.py -v` exits 0 with "5 passed"
    - `cd federated-personalized-cf && pytest tests/test_strategy.py::test_aggregate_fit_inherited_unchanged -v` passes (parent identity check)
    - `git diff --stat federated-personalized-cf/pyproject.toml` returns empty after commit (Plan 02 owns that file)
    - `git diff --stat federated-personalized-cf/federated_personalized_cf/models/` returns empty after commit (Task 2 owns models)
  </acceptance_criteria>
  <done>PersonalizedSplitFedAvg + PersonalizedSplitFedProx shipped with sufficient-stat aggregate_evaluate; _GLOBAL_PARAM_KEYS + _LOCAL_PARAM_KEYS frozensets declare the flipped split (item_* GLOBAL, local_user_* LOCAL); 5 GREEN strategy tests; D-23 preserved via aggregate_fit inheritance identity check.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: BPRMF + BasicMF single-row refactor — local_user_row nn.Parameter + local_user_bias nn.Parameter (D-01, D-03, PSN-06)</name>
  <files>federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py, federated-personalized-cf/federated_personalized_cf/models/basic_mf.py, federated-personalized-cf/tests/test_single_row_model.py</files>
  <read_first>
    - federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py (ENTIRE file; this is a structural refactor — need to see every use of user_embeddings / user_bias / user_ids / num_users)
    - federated-personalized-cf/federated_personalized_cf/models/basic_mf.py (ENTIRE file; same refactor)
    - federated-personalized-cf/federated_personalized_cf/models/__init__.py (adjust exports if needed)
    - .planning/phases/03-personalized-migration/03-CONTEXT.md (D-01/D-03 exact wording: `nn.Embedding(num_users, d)` → `nn.Parameter(shape=(d,))` + scalar `local_user_bias`; forward() no longer accepts user_ids)
  </read_first>
  <behavior>
    Tests to write FIRST (RED), then implement:
    - test_bpr_mf_single_row_shape: construct BPRMF(num_users=6040, num_items=3706, embedding_dim=64, use_bias=True). Assert: (a) model.local_user_row.shape == torch.Size([64]); (b) model.local_user_bias.shape in (torch.Size([]), torch.Size([1])); (c) model has NO attribute `user_embeddings`; (d) model has NO attribute `user_bias` (the name `user_bias` is retired; `local_user_bias` replaces it).
    - test_bpr_mf_local_params_contract: call model.get_local_parameters(); assert returned OrderedDict keys == {'local_user_row', 'local_user_bias'} exactly; tensor at 'local_user_row' has shape (64,); tensor at 'local_user_bias' has shape () or (1,).
    - test_bpr_mf_global_params_contract: call model.get_global_parameters(); assert returned OrderedDict keys == {'item_embeddings.weight', 'item_bias.weight', 'global_bias'} when use_bias=True; {'item_embeddings.weight'} when use_bias=False.
    - test_bpr_mf_no_ghost_table: assert `grep -c "nn.Embedding(num_users" federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py` via file-read (no subprocess grep needed — just open the file and `assert "nn.Embedding(num_users" not in src`).
    - test_basic_mf_single_row_shape: same shape assertions for BasicMF.
    - test_set_local_parameters_single_row_roundtrip: construct fresh model; save local params to dict; modify local_user_row values; set_local_parameters back; assert local_user_row has been restored.
  </behavior>
  <action>
    Step 1 — bpr_mf.py refactor. Change the parameter classification constants AT LINES 41-46 to:
    ```python
    # Parameter classification for split learning (with bias) — D-03 single-row contract
    _GLOBAL_PARAMS_WITH_BIAS = ('item_embeddings.weight', 'item_bias.weight', 'global_bias')
    _LOCAL_PARAMS_WITH_BIAS = ('local_user_row', 'local_user_bias')

    # Parameter classification for split learning (without bias)
    _GLOBAL_PARAMS_NO_BIAS = ('item_embeddings.weight',)
    _LOCAL_PARAMS_NO_BIAS = ('local_user_row',)
    ```

    Step 2 — In BPRMF.__init__, REPLACE the two lines that construct `self.user_embeddings = nn.Embedding(num_users, embedding_dim)` and `self.user_bias = nn.Embedding(num_users, 1)` with (D-01):
    ```python
    # D-01 single-user local row (was nn.Embedding(num_users, embedding_dim))
    self.local_user_row = nn.Parameter(torch.empty(embedding_dim))
    init.xavier_uniform_(self.local_user_row.view(1, -1))  # Xavier-uniform per D-11
    if use_bias:
        self.local_user_bias = nn.Parameter(torch.zeros(1))
    else:
        # Register as an unused buffer so get_local_parameters returns a consistent key set.
        self.register_buffer("local_user_bias", torch.zeros(1), persistent=False)
    ```
    KEEP `self.item_embeddings = nn.Embedding(num_items, embedding_dim)`, `self.item_bias = nn.Embedding(num_items, 1)`, `self.global_bias = nn.Parameter(torch.zeros(1))` (when use_bias) UNCHANGED — these stay GLOBAL and are the only params sent to the server.
    Retire `num_users` as a stored attribute of the model (still accepted as a constructor arg for API compat, but not stored — the model is per-user; there is no user table).

    Step 3 — In BPRMF.forward / forward_item_only, DROP the `user_ids` argument. The new signature is `forward(self, item_ids: torch.Tensor) -> torch.Tensor`. Internally, replace `user_vec = self.user_embeddings(user_ids)` with `user_vec = self.local_user_row.unsqueeze(0).expand(item_ids.size(0), -1)` (broadcast the single row across the batch). Replace `user_bias_vec = self.user_bias(user_ids).squeeze(-1)` with `user_bias_vec = self.local_user_bias` (scalar broadcast). Preserve the per-(item) prediction arithmetic: `score = global_bias + local_user_bias + item_bias[item] + dot(local_user_row, item_embeddings[item])`.

    Step 4 — In BPRMF.get_local_parameters (around line 397), replace the loop body with:
    ```python
    state = OrderedDict()
    state["local_user_row"] = self.local_user_row.detach().clone()
    state["local_user_bias"] = self.local_user_bias.detach().clone() if isinstance(self.local_user_bias, nn.Parameter) else self.local_user_bias.detach().clone()
    return state
    ```

    Step 5 — In BPRMF.set_local_parameters (around line 428), replace the shape-mismatch-handling loop with:
    ```python
    loaded, missing = [], []
    if "local_user_row" in local_state_dict:
        self.local_user_row.data.copy_(local_state_dict["local_user_row"])
        loaded.append("local_user_row")
    else:
        missing.append("local_user_row")
    if "local_user_bias" in local_state_dict and isinstance(self.local_user_bias, nn.Parameter):
        self.local_user_bias.data.copy_(local_state_dict["local_user_bias"])
        loaded.append("local_user_bias")
    else:
        missing.append("local_user_bias")
    return loaded, missing
    ```

    Step 6 — basic_mf.py: apply the SAME refactor (single-row local_user_row + local_user_bias; forward signature drops user_ids; get/set_local_parameters use the 2-key contract). MSE prediction arithmetic stays the same modulo the user-vec broadcast.

    Step 7 — BPRMF.sample_negatives method (if present): leaves the items-domain sampling untouched; it never needed user_ids because sampling draws negatives from `range(num_items)` regardless. If the current implementation takes a `user_rated_items` dict keyed by user_idx, drop the user_idx key — the client IS one user, and `user_rated_items` becomes a single set.

    Step 8 — Create federated-personalized-cf/tests/test_single_row_model.py with the 6 tests listed in <behavior>. Use direct file reads (not subprocess grep) for the "no ghost table" assertion:
    ```python
    def test_bpr_mf_no_ghost_table():
        src_path = Path(__file__).resolve().parents[1] / "federated_personalized_cf" / "models" / "bpr_mf.py"
        src = src_path.read_text()
        assert "nn.Embedding(num_users" not in src, "D-01 violated: found ghost user table"
        assert "self.user_embeddings" not in src, "D-01 violated: attribute user_embeddings still present"
    ```

    Step 9 — Verify GREEN: `cd federated-personalized-cf && pytest tests/test_single_row_model.py -v` → 6 passed. Then run the full new suite: `pytest tests/ -v` → 11 passed (5 strategy + 6 model). Guard: the pre-existing `federated-personalized-cf/test_dataset.py` / `test_models.py` at module ROOT are ad-hoc scripts and are NOT discovered by `tests/` pytest; leave them alone.

    Step 10 — Commit (--no-verify per Wave-1 parallel rule):
    ```
    git add federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py \
            federated-personalized-cf/federated_personalized_cf/models/basic_mf.py \
            federated-personalized-cf/tests/test_single_row_model.py
    git commit --no-verify -m "feat(03-01): BPRMF + BasicMF single-row refactor (D-01, D-03, PSN-06)"
    ```
  </action>
  <acceptance_criteria>
    - `grep -c "self.local_user_row = nn.Parameter" federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py` returns 1
    - `grep -c "self.local_user_bias = nn.Parameter" federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py` returns 1 (use_bias=True branch)
    - `grep -c "self.local_user_row = nn.Parameter" federated-personalized-cf/federated_personalized_cf/models/basic_mf.py` returns 1
    - `grep -cE "nn\\.Embedding\\(num_users" federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py` returns 0 (D-01: no ghost table)
    - `grep -cE "nn\\.Embedding\\(num_users" federated-personalized-cf/federated_personalized_cf/models/basic_mf.py` returns 0
    - `grep -c "self.user_embeddings" federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py` returns 0
    - `grep -c "_LOCAL_PARAMS_WITH_BIAS = ('local_user_row', 'local_user_bias')" federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py` returns 1
    - `grep -c "_GLOBAL_PARAMS_WITH_BIAS = ('item_embeddings.weight', 'item_bias.weight', 'global_bias')" federated-personalized-cf/federated_personalized_cf/models/bpr_mf.py` returns 1
    - `cd federated-personalized-cf && pytest tests/test_single_row_model.py -v` exits 0 with "6 passed"
    - `cd federated-personalized-cf && pytest tests/ -v` exits 0 with "11 passed" (combined Task 1 + Task 2 suite)
    - `python -c "from federated_personalized_cf.models.bpr_mf import BPRMF; m = BPRMF(num_users=6040, num_items=3706, embedding_dim=64); assert m.local_user_row.shape == (64,); assert set(m.get_local_parameters().keys()) == {'local_user_row', 'local_user_bias'}; assert set(m.get_global_parameters().keys()) == {'item_embeddings.weight', 'item_bias.weight', 'global_bias'}; print('ok')"` prints `ok`
    - `git diff --stat federated-personalized-cf/pyproject.toml` returns empty after commit
    - `git diff --stat federated-personalized-cf/federated_personalized_cf/dataset.py federated-personalized-cf/federated_personalized_cf/client_app.py federated-personalized-cf/federated_personalized_cf/server_app.py federated-personalized-cf/federated_personalized_cf/task.py` returns empty (Plans 02/03/04 own these)
  </acceptance_criteria>
  <done>BPRMF and BasicMF collapsed to single-user-row representation (D-01); local_user_row + local_user_bias nn.Parameter pair replaces nn.Embedding(num_users, d) ghost table; get/set_local_parameters contract uses 2-key OrderedDict (D-03); forward() no longer takes user_ids; 6 GREEN model tests; combined plan suite 11 passing.</done>
</task>

</tasks>

<verification>
- `cd federated-personalized-cf && pytest tests/ -v` exits 0 with "11 passed" (5 strategy + 6 single-row model)
- `python -c "from federated_personalized_cf.strategy import PersonalizedSplitFedAvg, PersonalizedSplitFedProx, _GLOBAL_PARAM_KEYS, _LOCAL_PARAM_KEYS; assert _LOCAL_PARAM_KEYS == frozenset({'local_user_row', 'local_user_bias'}); assert _GLOBAL_PARAM_KEYS == frozenset({'item_embeddings.weight', 'item_bias.weight', 'global_bias'}); from flwr.server.strategy import FedAvg; assert PersonalizedSplitFedAvg.aggregate_fit is FedAvg.aggregate_fit; print('ok')"` prints `ok`
- `python -c "from federated_personalized_cf.models.bpr_mf import BPRMF; m = BPRMF(num_users=6040, num_items=3706, embedding_dim=128, use_bias=True); assert m.local_user_row.shape == (128,); assert not hasattr(m, 'user_embeddings'); print('ok')"` prints `ok`
- `git log --oneline -3` shows the 2 task commits (Task 1 `feat(03-01): PersonalizedSplit...`, Task 2 `feat(03-01): BPRMF + BasicMF single-row...`)
- `git diff --stat federated-personalized-cf/pyproject.toml federated-personalized-cf/federated_personalized_cf/dataset.py federated-personalized-cf/federated_personalized_cf/client_app.py federated-personalized-cf/federated_personalized_cf/server_app.py federated-personalized-cf/federated_personalized_cf/task.py` returns empty (D-18 scope preserved; Plans 02/03/04 own those)
</verification>

<success_criteria>
- PSN-06 is observable at the MODEL layer: single-row nn.Parameter replaces the 6040×d ghost table; get_local_parameters returns a 2-key OrderedDict with per-user-single-row tensors; disk payload size-per-client drops from ~3 MB (6040×128×4B) to ~516 B (128+1 floats).
- PersonalizedSplitFedAvg/FedProx ship with sufficient-stat aggregate_evaluate (mirroring Phase-2 Plan 01's BaselineFedAvg/FedProx); aggregate_fit is inherited UNCHANGED preserving D-23 — the parent's weighted average of GLOBAL params (item embeddings, item bias, global bias) is correct for the split-learning contract.
- Test tree exists at federated-personalized-cf/tests/ (package with __init__.py + conftest.py + test_strategy.py + test_single_row_model.py) and reports 11 GREEN tests.
- Wave-1 write-race prevented: pyproject.toml and dataset.py/client_app.py/server_app.py/task.py are untouched by Plan 01 (Plan 02 and later own them).
</success_criteria>

<output>
After completion, create `.planning/phases/03-personalized-migration/03-personalized-migration-01-SUMMARY.md` using the standard summary template with: file list (7 files: strategy.py + 2 models + 4 test files), decisions made (D-18 strategy.py rip justification if that pre-existing SplitFedAvg code was removed), deviations, issues encountered, test counts (11 GREEN), commit SHAs, next-plan readiness (Plans 03 + 04 depend on this strategy; Plan 03 depends on this model layout).
</output>
