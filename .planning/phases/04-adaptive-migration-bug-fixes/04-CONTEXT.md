# Phase 4: Adaptive Migration & Bug Fixes - Context

**Gathered:** 2026-04-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Migrate `federated-adaptive-personalized-cf/` (the thesis-contribution module) to a correct
cross-device benchmark AND fix the three bugs that currently prevent the adaptive machinery
from accumulating correctly across rounds:

1. Per-user learned alpha (`_logit_alpha.weight`) is re-initialized from the heuristic every
   round because `enable_per_user_alpha()` is called AFTER `load_local_user_embeddings()`
   (CONCERNS.md). Same ordering bug for `item_perturbation`.
2. Server-side prototype EMA (`p_global`) is a live state that is never snapshotted at the
   best round — early-stopping restores the best `ArrayRecord` against the last-round
   prototype, so reported metrics don't correspond to the restored state.
3. Training negatives for a user can include the held-out test positive (shared with
   baseline/personalized; fixed by FND-03 exclusion).

This phase is the **thesis-headline rung** of the comparison ladder (baseline → personalized →
adaptive): it must win on overall AND sparse-user NDCG@10 under the cross-device protocol.

**In scope:** ADP-01..08 requirements — cross-device defaults, benchmark-mode one-user
assertion, exclusion-set training negatives, bug-fix ordering, prototype EMA best-round
restore, alpha-factory unit tests, FND-07 protocol fingerprint manifest, server-side seeded
sampling + RNG-fixed evaluator + sufficient-stat metrics + run-scoped cache (PSN-05 pattern
with schema_version bumped to 2).

**Out of scope:** PFedRec reproduction (Phase 5); thesis ablation sweeps (Phase 7); DP /
privacy quantification (v2); shared `fedrec_common/` extraction (v2 REF-01). Cross-silo
(Dirichlet) runs for this module are FROZEN — pre-Phase-4 commits remain the authoritative
artifact for any legacy re-run.

</domain>

<decisions>
## Implementation Decisions

### ADP-02 cache layout + enable-before-load ordering
- **D-01:** Atomic single-file cache per partition — `.embedding_cache/{run_id}/partition_{pid}.pt`
  (or `sig_*` when `reuse-cache=true` per Phase 3 D-09) — containing ALL local keys in one
  `torch.save(state_dict, weights_only=True)` blob: `local_user_row`, `local_user_bias`,
  `personal_mlp.*` (every sublayer weight + bias per `mlp-hidden-dims`), `fusion_gate.*` /
  `fusion_layer.*` (whichever the active `fusion-type` creates), `logit_alpha.weight`,
  `item_perturbation.weight`. Same atomic tempfile+rename pattern as Phase 3.
- **D-02:** `schema_version=2` in `manifest.json` — full adaptive fingerprint. Phase 3 v1
  fields (`run_id`, `method`, `num_users`, `num_items`, `dim`, `split_hash`) PLUS:
  `alpha_method`, `fusion_type`, `mlp_hidden_dims` (joined string), `per_user_alpha_enabled`,
  `item_perturbation_enabled`, `contrastive_lambda`. Any knob that changes cached tensor shape
  OR semantics is a signature field → silent cross-experiment contamination blocked. Swapping
  `alpha-method=hierarchical_conditional` → `alpha-method=multi_factor` mid-cache hard-fails
  instead of silently reusing tensors with a different semantic meaning.
- **D-03:** In `mode="benchmark_cross_device"`, `enable_per_user_alpha(True)` AND
  `enable_item_perturbation(True)` are called **unconditionally** in `client_app.py` BEFORE
  `load_local_user_embeddings(...)`, so `_logit_alpha.weight` and `item_perturbation.weight`
  are in `_LOCAL_PARAMS` at load time. Per-round cached values are restored instead of
  re-initialized from the heuristic. Run-config flags (`enable-per-user-alpha=false`, etc.)
  become **ablation-only overrides** — they turn the component OFF for a specific sweep cell
  but do not leave it uninitialized when ON.
- **D-04:** Schema-version mismatch (e.g., trying to load a Phase-3 `schema_version=1` cache
  under Phase-4 code) → mirror Phase 3 D-05: raise `RuntimeError` with per-field delta and an
  explicit `"Run: rm -rf .embedding_cache/{run_id}/"` hint. No auto-migration. No silent
  cold-start. Consistent with "never auto-delete or silently reshape" cross-module policy.

### ADP-03 server prototype EMA best-round restore
- **D-05:** `SplitFedAvg` holds `self.best_prototype` (numpy ndarray, shape=(d,)) alongside
  `self.best_arrays` (already from Phase 2 D-27). Both are snapshotted at the **same moment**
  — when `current_ndcg > best_metric` on the aggregate_evaluate hook. Pure in-memory state; no
  extra per-round I/O.
- **D-06:** The final best-round prototype is embedded in the result JSON as a `float[]` under
  the `_manifest.best_prototype` key (D-15 double-write). Payload is tiny at `dim=128`
  (~4KB). Enables post-hoc verification: "was the reported NDCG actually computed against the
  restored EMA?" Satisfies ADP-08's "full protocol fingerprint" requirement.
- **D-07:** For the FINAL centralized evaluation after best-round restore, set
  `self._global_prototype = self.best_prototype` BEFORE broadcasting the last-round
  `train_config_dict` (which carries the prototype to clients via `global_prototype` field).
  Clients receiving the config see the RESTORED prototype, not the last-round one. Ensures
  `best_*` metrics truly correspond to the restored state.
- **D-08:** Degenerate case — best round fires before any prototype was aggregated (round 0,
  or every selected client was cold-start with no prototype to contribute): snapshot
  `np.zeros(embedding_dim, dtype=np.float32)` as the `best_prototype`, log a warning
  `"Prototype snapshot at best round R=X is zero vector — no prior prototype aggregation
  yet."` Zero vector is semantically "no distributional information" and is a safe neutral.

### Benchmark-mode thesis defaults (in `pyproject.toml` + mode resolver)
- **D-09:** `model-type=dual` is the default under `mode="benchmark_cross_device"` —
  `DualPersonalizedBPRMF` with both Level-1 statistical blend and Level-2 PersonalMLP.
  `bpr` / `basic` remain available as ablation overrides via `--run-config`.
- **D-10:** `alpha-method=hierarchical_conditional` is the benchmark default — the thesis
  contribution's two-stage aggregation (geometric mean for data_volume, harmonic mean for
  preference_quality, conditional rules for edge cases). `multi_factor` / `data_quantity` are
  ablation overrides.
- **D-11:** `fusion-type=concat` is the benchmark default — `Linear([score_cf; score_mlp])`
  learnable weighted combination. `gate` / `add` are ablation overrides. `fusion_layer` /
  `fusion_gate` parameters are LOCAL (per client, never aggregated).
- **D-12:** `contrastive-lambda=0.1` is the benchmark default — InfoNCE loss on
  (p_local, p_effective) positive pair with batch-negative sampling. `0.0` is an ablation
  override. With D-03 forcing `enable-per-user-alpha=true` + `enable-item-perturbation=true`
  unconditionally, benchmark mode IS the thesis config: no "silent default-off thesis cell"
  exists.

### Cold-start blend behavior (primary thesis-claim reinforcement on sparse users)
- **D-13:** On cache-miss (first round for this partition), override the blend to
  prototype-only for that round: `p_effective = p_global` (effective `α = 0`). Client trains
  local params from a neutral starting point instead of Xavier-noisy `p_local`. Next round
  (cache exists after save) the normal `p_effective = α·p_local + (1-α)·p_global` blend
  resumes. Directly benefits sparse users, whose entire 50-round evaluation is mostly
  cold-start rounds — primary thesis claim is on sparse NDCG@10.
- **D-14:** InfoNCE contrastive loss is **skipped** in cold-start rounds. Positive pair would
  be `(Xavier_noise, p_global)` which is either a noise anchor (hurts) or trivial (doesn't
  help). Compute `L = L_BPR + reg·||item_perturbation||²` only on cold rounds. Contrastive
  resumes the next round once `p_local` has had one training pass.
- **D-15:** Cold-start detection reuses the Phase 3 D-13 signal: before
  `load_local_user_embeddings`, check if `partition_{pid}.pt` exists. If not → cold round.
  Pass `is_cold_round: bool` through to `train()`, used to (a) force `α = 0` in the blend
  and (b) skip contrastive. Same signal drives the server-side `cold_starts_per_round`
  metric (Phase 3 D-13) — zero extra bookkeeping.
- **D-16:** Per-round **alpha diagnostics** are first-class fields in
  `eval_metrics_history[round_num]` + W&B round logs:
  `alpha_clip_hit_rate` (fraction at `min_alpha=0.1` OR `max_alpha=0.95`), `alpha_mean`,
  `alpha_std`, `alpha_p25`, `alpha_p50`, `alpha_p75`. Thesis artifact answers the CONCERNS.md
  critique directly from the run JSON (no post-hoc `alpha_analysis.py` invocation needed).
  `alpha_analysis.py` retained for deeper visualization but is no longer the only path to the
  headline numbers.

### Claude's Discretion
These hyperparameters inherit the existing CLAUDE.md / pyproject defaults. Claude has
flexibility to revisit during planning/execution if empirical evidence warrants:
- `prototype-momentum=0.9` (EMA half-life ~6 rounds at default)
- `item-perturbation-reg=0.01` (L2 strength on `||item_perturbation||²`)
- `alpha` floor/ceiling `[0.1, 0.95]` (from `HierarchicalConditionalAlphaConfig`)
- `mlp-hidden-dims="512,256,128"`
- Cross-silo legacy freeze: follow Phase 3 D-02 pattern — `dataset.py` raises
  `NotImplementedError` on `partition_mode != "natural"` with a clear pointer to
  pre-Phase-4 commits for legacy reproduction
- FedProx proximal term scope: the proximal penalty applies ONLY to GLOBAL params
  (`item_embeddings.weight`, `item_bias.weight`, `global_bias`). The expanded local-param
  set (`personal_mlp.*`, `fusion_*`, `logit_alpha`, `item_perturbation`) is NEVER touched by
  the proximal term — architectural property of split learning, not a discussion point
- Exact code layout of the cold-start branch (in `client_app.py`, `task.py`, or as a helper in
  `models/dual_personalized_bpr_mf.py`) — Claude picks cleanest placement

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Foundation contract (Phase 1)
- `scripts/foundation/fedrec_foundation/mapping.py` — canonical user/item ID maps (6040×3706)
- `scripts/foundation/fedrec_foundation/split.py` — LOO split manifest, `split_hash`
- `scripts/foundation/fedrec_foundation/exclusion.py` — FND-03 `ExclusionTable` (held-out test item excluded from train negatives, ADP-05)
- `scripts/foundation/fedrec_foundation/evaluator.py` — `sampled_loo_99` primary evaluator (ADP-07 alpha-factory output is independent but the evaluation harness is shared)
- `scripts/foundation/fedrec_foundation/weight_policy.py` — server-side aggregation weight policy (num_positives)
- `scripts/foundation/fedrec_foundation/fit_metrics.py` — `FitMetricsContract` + `EvaluateMetricsContract` wire payloads (ADP-06)
- `scripts/foundation/fedrec_foundation/rng.py` — FND-06 `np_rng` / `torch_gen` / `py_rng` factories (ADP-06)
- `scripts/foundation/fedrec_foundation/manifest.py` — FND-07 `build_run_manifest` + `embed_manifest_in_result` + `write_manifest_sibling` (ADP-08, with `module="adaptive"`)
- `scripts/foundation/fedrec_foundation/mode.py` — mode-profile resolver (`benchmark_cross_device` carries canonical hyperparams)

### Phase 2 template (what to clone)
- `.planning/phases/02-baseline-migration/02-baseline-migration-01-SUMMARY.md` — `BaselineFedAvg/FedProx` sufficient-stat aggregate_evaluate pattern
- `.planning/phases/02-baseline-migration/02-baseline-migration-03-SUMMARY.md` — client_app.py FND-06 RNG wiring + FND-03 exclusion threading + D-24 gradient-isolation (snapshot/restore around optimizer.step for non-user rows)
- `.planning/phases/02-baseline-migration/02-baseline-migration-04-SUMMARY.md` — server_app.py mode resolver + seeded sampling + best-round restore + D-15 double-write
- `.planning/phases/02-baseline-migration/02-baseline-migration-05-SUMMARY.md` — G-03-01 discovery round + partition-id-space sampling + subprocess determinism regression guard

### Phase 3 template (split-learning + cache + cold-start counter)
- `.planning/phases/03-personalized-migration/03-CONTEXT.md` — all D-04..D-13 decisions Phase 4 inherits (manifest-sidecar cache with `schema_version=2` bump, Xavier first-use init, cold-start counter, single-row forward contract)
- `.planning/phases/03-personalized-migration/03-personalized-migration-01-SUMMARY.md` — `PersonalizedSplitFedAvg/FedProx` + single-row BPRMF/BasicMF contract (adaptive `DualPersonalizedBPRMF` extends this pattern with additional LOCAL keys)
- `.planning/phases/03-personalized-migration/03-personalized-migration-03-SUMMARY.md` — client_app.py + task.py cross-device wire + D-04..D-10 manifest-sidecar cache implementation
- `.planning/phases/03-personalized-migration/03-personalized-migration-04-SUMMARY.md` — server_app.py migration with D-13 cold-start counter + D-15 double-write (`module="personalized"`; Phase 4 bumps to `"adaptive"`)
- `.planning/phases/03-personalized-migration/03-personalized-migration-05-SUMMARY.md` — `scripts/clean_cache.py` + subprocess determinism regression guard (extend to cover adaptive's schema_version=2 cache)

### Project-level & module-level
- `.planning/PROJECT.md` — Core Value, Validated requirements (Phases 1-3 shipped), Active (ADP + EVL + THS)
- `.planning/REQUIREMENTS.md` §ADP — ADP-01..08 detailed text
- `.planning/codebase/CONCERNS.md` — the adaptive-module bug list: enable-after-load ordering (line ~340), clip-floor effect, global `random.seed` in `evaluate_ranking_sampled`, silent shape-mismatch on embedding-dim change (line ~141-142)
- `CLAUDE.md` — thesis defaults (dim=128, BPR, Adam, 5-10 local epochs), alpha clip range [0.1, 0.95], EMA momentum 0.9
- `federated-adaptive-personalized-cf/claude.md` — module architecture (dual-level personalization, hierarchical conditional alpha, global prototype, per-user learned alpha + item perturbation + contrastive InfoNCE)

### Paper references (thesis context)
- `Papers/digested/_INDEX.md` — paper knowledge base index
- `Papers/digested/zhang_2024_fedca.md` — adaptive personalization survey context
- `Papers/digested/zhang_2025_survey_personalized_fedrec.md` — personalized FedRec taxonomy; positions this thesis contribution
- `Papers/digested/arivazhagan_2019_fedper.md` — FedPer (personalization layers as LOCAL params; analogous to our PersonalMLP)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`models/dual_personalized_bpr_mf.py`** — `DualPersonalizedBPRMF` class already exists with
  `set_alpha()`, `set_global_prototype()`, `enable_per_user_alpha()`,
  `enable_item_perturbation()`, `compute_user_prototype()`, `get/set_local_parameters`,
  `get/set_global_parameters`. The refactor is an **ordering fix + local-param expansion**,
  not a ground-up rewrite.
- **`models/adaptive_alpha.py`** — `create_alpha_computer(config, hc_config)` factory already
  produces all three alpha methods with clip range `[0.1, 0.95]`. ADP-07 unit test extends
  the existing test suite in-place.
- **`evaluation/alpha_analysis.py`** — post-hoc alpha analysis already exists; refactor to
  expose the scalar summary (mean/std/quartiles/clip-hit-rate) as a callable used per-round
  by `aggregate_evaluate` (D-16).
- **Phase 2+3 shipped patterns** — `BaselineFedAvg` (Phase 2) and `PersonalizedSplitFedAvg`
  (Phase 3) both ship the sufficient-stat aggregate_evaluate contract. `AdaptiveSplitFedAvg`
  clones `PersonalizedSplitFedAvg` and adds the prototype EMA (`_global_prototype` +
  `_aggregate_prototypes` method).
- **`fedrec_foundation.manifest.embed_manifest_in_result`** — already ships; extend the
  manifest dict to accept `best_prototype: List[float]` under `_manifest` (D-06).
- **`scripts/clean_cache.py` (Phase 3)** — already handles `schema_version=1`; Phase 4 only
  needs to confirm it doesn't break on `schema_version=2` manifests (no code change expected
  unless the cleaner parses manifest fields).

### Established Patterns
- **Split-learning LOCAL/GLOBAL frozenset declaration in `strategy.py`** (Phase 3 lock):
  `_GLOBAL_PARAM_KEYS` = `{'item_embeddings.weight', 'item_bias.weight', 'global_bias'}`
  (same as personalized — the item side is unchanged);
  `_LOCAL_PARAM_KEYS` = personalized's two keys PLUS the adaptive-specific keys (personal_mlp,
  fusion, logit_alpha, item_perturbation). `aggregate_fit` inherited unchanged (D-23 Phase 3).
- **D-24 gradient isolation (Phase 2 Plan 03)** — for the single-row local user param, wrap
  `optimizer.step()` with `_snapshot_non_user_rows` / `_restore_non_user_rows`. In Phase 4
  this pattern applies to MORE rows (personal_mlp, fusion, logit_alpha, item_perturbation) —
  near-duplicate helper lives per-module (not extracted to foundation — v2 REF-01).
- **Server-side EMA update pattern (existing in adaptive/strategy.py)** — `_aggregate_prototypes`
  collects `user_prototype` from client `FitRes.metrics`, weighted mean, applies
  `p_global = momentum * p_old + (1 - momentum) * new_prototype`. Phase 4 adds the
  `best_prototype` snapshot branch alongside (D-05).

### Integration Points
- `federated-adaptive-personalized-cf/pyproject.toml` — mode-resolver entries for
  `benchmark_cross_device` default (num-supernodes=6040, natural partition, run-seed,
  weight-policy). Dev-dep `pytest` addition (mirror Phase 3 Plan 02).
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/dataset.py` —
  rip-and-replace as foundation adapter (D-17 from Phase 2); raise NotImplementedError on
  `partition_mode != "natural"` (D-02 mirror); keep `MovieLensDataset` / `download_movielens_1m`
  / `load_movielens_1m` verbatim (D-18).
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` — the
  ordering fix lives here: `enable_per_user_alpha(True)` + `enable_item_perturbation(True)`
  BEFORE `load_local_user_embeddings` (D-03). Cold-start detection (D-15) threads
  `is_cold_round: bool` into `train(..., is_cold_round=...)`.
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py` — FND-06
  RNG factories wired through DataLoader + negative sampling; FND-03 `ExclusionTable` threaded
  into negative-sampling pool; cold-round branch in `train_dual_personalized` / train pathway:
  override `alpha = 0` and skip contrastive when `is_cold_round=True` (D-13, D-14).
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py` —
  `AdaptiveSplitFedAvg` (or extended `SplitFedAvg`) with `best_prototype` snapshot (D-05);
  final centralized eval sets `self._global_prototype = self.best_prototype` before broadcast
  (D-07); D-15 double-write with `module="adaptive"` and `best_prototype` embedded (D-06).
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` —
  extends Phase 3 `PersonalizedSplitFedAvg` with `self.best_prototype` field and the
  snapshot-on-best-metric branch inside the override.

</code_context>

<specifics>
## Specific Ideas

No specific external references or "I want it like X" moments — all four discussed areas
landed on the recommended / thesis-config-coherent option. Locks Phase 4 as the direct
extension of Phase 3's split-learning contract with:

(a) expanded local-param set persisted atomically in one `.pt` blob under `schema_version=2`
    with a full-fingerprint manifest,
(b) `enable_per_user_alpha` + `enable_item_perturbation` unconditional in benchmark mode,
(c) server prototype EMA participating in the D-27 best-round restore symmetry,
(d) prototype-only cold-start blend that benefits sparse users at the first round of every
    truly-cold partition selection.

</specifics>

<deferred>
## Deferred Ideas

- **Sweep over `prototype-momentum`** (currently 0.9) — out of scope for Phase 4; thesis
  evaluation Phase 7 handles sweeps via `sweep.yaml`.
- **Calibration of alpha floor/ceiling `[0.1, 0.95]`** — codebase concern flagged potential
  clip-floor effect on sparse users. D-16 logs `alpha_clip_hit_rate`; if the metric reveals a
  problem at Phase-7 eval time, that's a follow-up phase (e.g., 7.1 gap-closure), not Phase 4.
- **Shared `fedrec_common/` extraction** — adaptive module duplicates dataset.py / early_stopping.py
  / bpr_mf.py / basic_mf.py patterns with baseline + personalized. v2 REF-01.
- **Differential privacy (DP-SGD)** — v2 DP-01.
- **ML-10M / ML-20M generalization** — v2 EXT-01.
- **PFedRec reproduction** — Phase 5 (PFR-01..09).

</deferred>

---

*Phase: 04-adaptive-migration-bug-fixes*
*Context gathered: 2026-04-20*
