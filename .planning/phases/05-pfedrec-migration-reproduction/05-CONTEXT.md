# Phase 5: PFedRec Migration & Reproduction - Context

**Gathered:** 2026-04-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Re-audit `federated-pfedrec/` against `IJCAI-23-PFedRec/`, migrate to cross-device
(1 user = 1 client, N=6040), and reproduce the published IJCAI-23 PFedRec numbers
within ±2 absolute points on HR@10 and NDCG@10 under `paper_compat_pfedrec` mode.

This is the **calibration baseline** of the comparison ladder
(baseline → personalized → adaptive → **pfedrec**): it is NOT a thesis-contribution
rung. Its job is to show the Flower re-implementation faithfully reproduces a
published reference, so that any thesis claim against PFedRec is methodologically
defensible.

**In scope:** PFR-01..09 requirements — cross-device defaults, PFR-02 reference audit
(divergence diff table with align-to-reference / keep-flower decisions), PFR-03 atomic
per-user cache (single artifact per user, hard-fail on schema/shape mismatch),
PFR-04 training negatives exclude held-out test positive (FND-03), PFR-05 single-user
client path collapse, PFR-06 server sampling + RNG-fixed evaluator + sufficient-stat
aggregation match the foundation contract, PFR-07 training negatives re-sampled every
round, PFR-08 ±2-point reproduction under `paper_compat_pfedrec`, PFR-09 FND-07
protocol fingerprint.

**Out of scope:** Thesis comparison sweep (Phase 7 THS-01..07); evaluation harness
unification (Phase 6 EVL-01..06); DP / privacy quantification (v2 DP-01..02);
shared-code refactor (v2 REF-01..02). Cross-silo (Dirichlet) PFedRec runs are FROZEN —
pre-Phase-5 commits remain the authoritative artifact for any legacy re-run.

</domain>

<decisions>
## Implementation Decisions

### PFR-02 Reference divergence resolution

- **D-01:** `affine_output.bias` is GLOBAL, not LOCAL. Matches IJCAI-23 reference
  (`engine.py:143` deletes only `affine_output.weight` from `round_participant_params`,
  so `affine_output.bias` is aggregated across users on the server). Updates Flower's
  current strategy.py classification (which treats both as LOCAL, the documented
  CONCERNS divergence #9). Post-decision: `_GLOBAL_PARAMS = ('embedding_item.weight',
  'affine_output.bias')`; `_LOCAL_PARAMS = ('affine_output.weight',)`. This is the
  highest-impact divergence and the strongest single lever for landing PFR-08 within
  ±2 points.
- **D-02:** Training-negative resampling uses the FND-06 RNG factory per round.
  Replace `task.py:130 rng = random.Random(seed)` with
  `np_rng(run_seed, user_idx, round_num, "train_neg")`. Same pattern as Phase 2-4;
  closes CONCERNS bug #5 (same training negatives every round) and gives byte-identical
  reruns under fixed seed. Closes PFR-07.
- **D-03:** Cold-start handling matches Phase 3/4 pattern — cache-existence probe in
  `client_app.py` before `load_local_user_params`, plus per-round `cold_starts_per_round`
  scalar in `eval_metrics_history` and W&B (Phase 3 D-13 idiom). Reference's explicit
  `if round_id != 0` gate (engine.py:104-110) is achieved implicitly via the probe:
  cache miss on round 0 → falls back to cold-round init.
- **D-04:** Eval-time BCE loss is computed over positives + 99 negatives, matching
  reference `engine.py:195-196`: `ratings_pred = torch.cat((test_score, negative_score))`.
  Closes the PFR-02 audit row "eval BCE scope". HR@10 / NDCG@10 (the thesis numbers)
  are unaffected; eval BCE is a diagnostic that becomes directly comparable to reference
  logs.

### Mode profile design

- **D-05:** Ship `paper_compat_pfedrec` only; do NOT add a `benchmark_cross_device`
  variant for PFedRec. PFedRec is what the IJCAI-23 paper says PFedRec is. Phase 7
  thesis-comparison table reports PFedRec at its paper-faithful config and footnotes
  the per-module config differences ("PFedRec at paper-compat: dim=32, SGD lr=0.1,
  BCE; baseline/personalized/adaptive at benchmark_cross_device: dim=64, Adam, BPR").
  Avoids the philosophically incoherent "PFedRec at non-PFedRec hyperparams" position.
- **D-06:** `fraction-train = 1.0` locked under `paper_compat_pfedrec`. All 6040 users
  selected each round. Required for PFR-08 reproduction (reference uses
  `clients_sample_ratio = 1.0` / full participation). Wallclock ~3 hours per run on
  RTX 5090 per the reference's logged run time; acceptable for a one-shot reproduction.
- **D-07:** Drop FedProx for PFedRec. Ship only `PFedRecSplitFedAvg`; do NOT ship
  `PFedRecSplitFedProx`. Reference doesn't use FedProx; PFedRec's per-user score
  function doesn't benefit from a global proximal term in the same way (proximal scope
  would be a single tensor, `embedding_item.weight`); fewer code paths, fewer ablations.
- **D-08:** No held-out validation split. Carry forward Phase 2/3/4 D-27 in-memory
  best-round-restore against `sampled_ndcg@10` on the test set. Reference IJCAI-23 also
  monitors test (paper-faithful). Documented information leak (CONCERNS bug #2) is
  accepted in this thesis cycle; val-split is deferred to v2.
- **D-09:** Cross-silo path frozen via D-02 NotImplementedError mirror — both
  `load_partition_data` and `load_full_data` raise `NotImplementedError(...)` when
  `partition_mode != "natural"`. Pre-Phase-5 commits are the authoritative cross-silo
  PFedRec artifact. Mirrors Phase 3/4 D-02; the PFedRec REFERENCE itself is cross-device,
  so freezing legacy cross-silo costs nothing for the thesis story.
- **D-10:** W&B project = `federated-cf-cross-device` (shared with Phase 2/3/4
  cross-device runs). Cross-module dashboards plot all four modules together; Phase 7
  thesis-table queries filter by `_manifest.module`.
- **D-11:** Standard Phase 1 D-10 'allow + log loudly' for any CLI overrides under
  `paper_compat_pfedrec`. No extra restriction layer; the user is trusted to respect
  paper-compat for PFR-08 reproduction. All overrides captured in `manifest.overrides`
  and surfaced as `[MODE OVERRIDE]` log lines.
- **D-12:** Strategy class renamed to `PFedRecSplitFedAvg` (currently `SplitFedAvg`).
  Matches Phase 3/4 module-prefixed convention (`PersonalizedSplitFedAvg`,
  `AdaptiveSplitFedAvg`, `BaselineFedAvg`). Drops `SplitFedProx` per D-07.
- **D-13:** Best-round-restore metric = `sampled_ndcg@10`. Matches Phase 2/3/4 D-27;
  cross-module symmetry. NDCG@10 and HR@10 rankings are highly correlated in PFedRec,
  so picking by NDCG@10 also produces near-best HR@10.
- **D-14:** PFR-08 verification = single-seed run (`run-seed=42`) with auto-verify at
  run end. Server_app reads `IJCAI-23-PFedRec/sh_result/ml-1m.txt`, parses HR@10 /
  NDCG@10 reference numbers (target: HR=0.7286-0.7315, NDCG=0.4407-0.4453, taking the
  most recent / best of the two reference runs as the target — final choice deferred to
  planner), asserts |our - reference| ≤ 2.0 absolute points, prints
  `[PFR-08 VERIFIED]` or `[PFR-08 FAILED: Δhr=X Δndcg=Y]`. Multi-seed reporting deferred
  to Phase 7 THS-02.
- **D-15:** Strict hyperparam lock under `paper_compat_pfedrec`. Pyproject.toml carries
  paper-compat values only; sensitivity ablations use `--run-config` overrides per D-11.
  No parallel "ablation knob" rows in pyproject.toml — single source of truth for the
  reproduction config.

### Per-user cache layout

- **D-16:** Cache file path = `.embedding_cache/{run_id}/partition_{pid}.pt` (Phase 3/4
  uniform). In cross-device, `partition_id == user_idx`, so each file is one user's
  state dict. One fewer directory level than current PFedRec layout
  (`partition_{id}/user_{uid}/affine_output.pt` is replaced); `clean_cache.py` works
  unchanged; cross-module idiom uniformity for downstream tooling.
- **D-17:** Manifest sidecar at `.embedding_cache/{run_id}/manifest.json` with
  `schema_version=3` and 9 fields: `run_id`, `method='pfedrec'`, `num_users`,
  `num_items`, `latent_dim`, `split_hash`, `loss='bce'`, `num_train_negatives`,
  `bias_classification='global'`. The `bias_classification` field is a sentinel that
  catches any future regression that reverts D-01 — schema-mismatch hard-fails
  immediately. Written via `fedrec_foundation.atomic.atomic_write_json`.
- **D-18:** Carry forward Phase 3 D-08/D-09 reuse-cache pattern. Default
  `reuse-cache=false` per-run (each `flwr run .` creates a fresh
  `.embedding_cache/{new_run_id}/`). Opt-in `--run-config 'reuse-cache=true'` switches
  the path to `.embedding_cache/sig_<sha256(signature_fields)[:16]>/` for fast
  reproduction-cycle iteration. PFR-08 retries during the audit phase benefit from this.
- **D-19:** First-round `affine_output` initialization = PyTorch nn.Linear default
  (Kaiming-uniform). Reference `engine.py:104-110` doesn't apply Xavier; PFR-08
  reproduction is sensitive to init scale (CONCERNS flags 50% performance variance with
  poor init per RecSys 2024). Cross-module Xavier in BPRMF/BasicMF/DualPersonalizedBPRMF
  is intentionally NOT mirrored here — paper-faithfulness wins.
- **D-20:** Persisted `affine_output.weight` tensor shape = native PyTorch
  `(1, latent_dim)`. No model refactor; existing `PFedRecMLP` forward path untouched;
  `set_local_parameters` loads `(1, latent_dim)` directly into `self.affine_output.weight`.
  Phase 3's D-01 single-row collapse pattern does NOT apply here — PFedRec is already
  per-user by construction (out_features=1).
- **D-21:** `set_local_parameters(strict=True)` — hard-fail on shape mismatch with
  RuntimeError carrying per-field delta + literal `Run: rm -rf .embedding_cache/{run_id}/`
  hint. Phase 3 D-05 idiom. PFR-03 explicitly mandates this; replaces the current
  PFedRec model's `strict=False` partial-load semantics.
- **D-22:** Cold-round client behavior = probe-then-load.
  `if cache_path.exists(): load_local_user_params(); else: cold_round=True; skip load`.
  Init falls back to PyTorch Linear default per D-19. Clean separation of cache-miss
  vs cache-corruption: cache-miss → cold round; cache-load failure → hard-fail per D-21
  (no fail-open).
- **D-23:** `scripts/clean_cache.py` (Phase 3 D-10) handles `schema_version=3`
  unchanged. The script globs `.embedding_cache/{run_id}/` and sorts by mtime; doesn't
  read manifest contents. No code change needed for Phase 5.

### Aggregation weight policy

- **D-24:** `weight_policy = "uniform"` under `paper_compat_pfedrec`. Required for
  PFR-08 ±2 reproduction — reference `engine.py:81` divides by `len(round_user_params)`
  (mean over participating clients, weight=1 each). Diverges from cross-module
  `num_positives` convention but that's expected and correct: paper_compat is the only
  shipped mode for PFedRec (D-05), so cross-module asymmetry doesn't propagate.
- **D-25:** Update the registered `_PAPER_COMPAT_PFEDREC` profile in
  `scripts/foundation/fedrec_foundation/mode.py` — change
  `weight_policy='num_positives'` to `weight_policy='uniform'` and remove the
  "Deferred confirmation to PFR-02" comment. Single source of truth for paper_compat
  semantics; closes the deferred-decision marker introduced in Phase 1.
- **D-26:** Eval-metric aggregation (HR@10 / NDCG@10) carries forward Phase 2 BSL-06
  sufficient-stat ratio: server sums `hit_count_at_10`, `ndcg_sum_at_10`,
  `evaluated_users` across clients; final ratio = sum_hit / sum_users. In cross-device
  with 1 user = 1 client, this is mathematically uniform per-user — each user
  contributes 1 to the denominator — so it matches reference's MetronAtK behavior
  without special-casing. No change to the BSL-06 contract.
- **D-27:** D-10 standard override behavior for `weight-policy` overrides under
  `paper_compat_pfedrec`. User can run
  `flwr run . --run-config "weight-policy=num_positives"` for ablation; the override
  prints a `[MODE OVERRIDE]` line and is captured in `manifest.overrides`. Phase 1 D-10
  contract uniform across all modes.

### Claude's Discretion

These hyperparameters / implementation details inherit existing CLAUDE.md / pyproject /
mode.py defaults. Claude has flexibility to revisit during planning/execution if
empirical evidence warrants:

- `lr=0.1`, `lr-eta=80`, `local-epochs=1`, `num-server-rounds=100`, `latent-dim=32`,
  `num-train-negatives=4`, `optimizer="sgd"`, `l2-regularization=0.0` — all in the
  registered `_PAPER_COMPAT_PFEDREC` profile in `mode.py`. Locked by the profile, not
  re-discussed.
- Exact code partition between Plans 01/02/03/... — planner decides based on Wave-1
  disjoint-file ownership pattern (Phase 2/3/4 precedent).
- Exact name of the new model class (e.g., refactor `PFedRecMLP` in place vs new
  `PFedRecSingleUserMLP`) — planner picks based on diff size.
- Exact `[PFR-08 VERIFIED]` / `[PFR-08 FAILED]` log-line formatting and where in
  `server_app.py` the auto-verify hook fires (after `embed_manifest_in_result` and
  before W&B summary write seems natural; planner confirms).
- Decision on which of the two `IJCAI-23-PFedRec/sh_result/ml-1m.txt` reference runs
  is the "canonical" PFR-08 target (both are within tight tolerance: HR=0.7286 / 0.7315,
  NDCG=0.4407 / 0.4453). Planner picks: most recent (last line) is reasonable;
  alternative is the higher-of-two (paper-tightest reproduction).
- Test surface / Wave layout — Phase 5 mirrors Phase 3/4: a `tests/` dir per module
  plus subprocess regression guards in `scripts/foundation/tests/`. Planner picks
  exact test count + parametrization.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### PFedRec reference implementation (PFR-02 audit anchor)

- `IJCAI-23-PFedRec/engine.py` — Reference `Engine` class. Key sections:
  §66-81 `aggregate_clients_params` (uniform mean policy, D-24 source of truth);
  §84-146 `fed_train_a_round` (random.sample participation §89-91, dual-LR optimizers
  §114-119, `del round_participant_params[user]['affine_output.weight']` at §143
  → bias is GLOBAL, D-01 source of truth);
  §149-212 `fed_evaluate` (BCE loss over positives + 99 negatives §195-196, D-04
  source of truth)
- `IJCAI-23-PFedRec/mlp.py` — `MLP` architecture: `embedding_item` + `affine_output` +
  `Sigmoid`. No init logic (Kaiming default), D-19 source of truth.
- `IJCAI-23-PFedRec/data.py` — `UserItemRatingDataset` BCE binary label format
  (1 positive + N negatives per batch).
- `IJCAI-23-PFedRec/metrics.py` — `MetronAtK` uniform per-user HR@10 / NDCG@10
  computation (D-26 carries forward Phase 2 BSL-06 sufficient-stat which equals this
  semantics in cross-device).
- `IJCAI-23-PFedRec/sh_result/ml-1m.txt` — Reference run results
  (HR@10 ∈ [0.7286, 0.7315], NDCG@10 ∈ [0.4407, 0.4453], both at 100 rounds).
  D-14 auto-verify target.
- `IJCAI-23-PFedRec/train.py` — Reference run entry point with paper-default config.

### Foundation contract (Phase 1, locked)

- `scripts/foundation/fedrec_foundation/mapping.py` — Canonical user/item ID maps
  (6040 users × 3706 items)
- `scripts/foundation/fedrec_foundation/split.py` — LOO split manifest, `split_hash`
  (6 fields including raw_data_hash + mapping_sha256)
- `scripts/foundation/fedrec_foundation/exclusion.py` — FND-03 `ExclusionTable`
  (PFR-04: held-out test item excluded from training negatives)
- `scripts/foundation/fedrec_foundation/evaluator.py` — `sampled_loo_99` primary
  evaluator (PFR-06)
- `scripts/foundation/fedrec_foundation/weight_policy.py` — `WeightPolicy` enum
  (D-24 `uniform` value)
- `scripts/foundation/fedrec_foundation/fit_metrics.py` — `FitMetricsContract` /
  `EvaluateMetricsContract` wire payloads with optional `partition_id` field (PFR-06)
- `scripts/foundation/fedrec_foundation/rng.py` — FND-06 `np_rng` / `torch_gen` /
  `py_rng` / `server_rng` factories (D-02, PFR-06)
- `scripts/foundation/fedrec_foundation/manifest.py` — FND-07 `build_run_manifest` /
  `embed_manifest_in_result` / `write_manifest_sibling` (PFR-09 with `module="pfedrec"`)
- `scripts/foundation/fedrec_foundation/mode.py` — Mode-profile resolver (D-25:
  update `_PAPER_COMPAT_PFEDREC.weight_policy` from `'num_positives'` to `'uniform'`)
- `scripts/foundation/fedrec_foundation/atomic.py` — `atomic_write_json` for D-17
  manifest sidecar

### Phase 2/3/4 templates (clone these patterns)

- `.planning/phases/02-baseline-migration/02-CONTEXT.md` — D-17 dataset rip-and-replace,
  D-18 surgical-edit discipline, D-25 mode-resolver canonical hyperparam source,
  D-27 in-memory best-round-restore (Phase 5 inherits all)
- `.planning/phases/02-baseline-migration/02-baseline-migration-04-SUMMARY.md` —
  server_app pattern (mode resolver + seeded sampling + sufficient-stat strategy
  wire-up + D-15 manifest double-write); Phase 5 server_app clones this shape
- `.planning/phases/02-baseline-migration/02-baseline-migration-05-SUMMARY.md` —
  G-03-01 discovery-round protocol + partition-id-space sampling (REQUIRED for Phase 5
  cross-device server_app)
- `.planning/phases/03-personalized-migration/03-CONTEXT.md` — D-04..D-13
  manifest-sidecar cache pattern (D-16, D-17, D-18 inherit; bumped to schema_version=3
  for PFedRec); D-13 cold-start counter (D-22 inherits)
- `.planning/phases/03-personalized-migration/03-personalized-migration-03-SUMMARY.md`
  — client_app + task.py contract wire pattern; D-04..D-10 manifest-sidecar cache
  implementation; Phase 5 client_app refactor clones this shape
- `.planning/phases/03-personalized-migration/03-personalized-migration-04-SUMMARY.md`
  — server_app cross-device migration with D-13 cold-start counter; Phase 5 server_app
  inherits the cold-start probe + counter exactly
- `.planning/phases/04-adaptive-migration-bug-fixes/04-CONTEXT.md` — Adaptive bug-fix
  pattern (PFR-02's `affine_output.bias` audit is structurally analogous to ADP-02's
  `_logit_alpha` enable-before-load fix: a documented bug in CONCERNS.md that requires
  param-classification or ordering correction)
- `.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-05-SUMMARY.md`
  — server_app schema_v2 cache + best-round restore; Phase 5 schema_v3 inherits the
  schema-bump pattern verbatim

### Project-level

- `.planning/PROJECT.md` — Validated Phases 1-4; Active PFR + EVL + THS; Key
  Decisions (re-audit PFedRec from reference; cross-device migration; centralized
  baselines stay as-is)
- `.planning/REQUIREMENTS.md` §PFR-01..09 — Acceptance criteria text
- `.planning/ROADMAP.md` §Phase 5 — Goal + 4 success criteria
- `.planning/codebase/CONCERNS.md` — PFedRec-specific bug audit: 8+1 known bugs
  (test-positive leak, no val split, cross-silo aggregation weighting, per-user cache
  contamination, frozen training negatives, unseeded participation, no best-model
  checkpoint, BCE on positives only, **bias-LOCAL mis-classification ←
  PFR-02 anchor**)
- `CLAUDE.md` — Project-wide conventions: notation (`w` global / `theta_i` local),
  code standards (type hints, NumPy docstrings, dataclasses, seed+config
  reproducibility), tech stack
- `federated-pfedrec/claude.md` — Module-specific architecture notes (will need
  updating after Phase 5 lands; planner scopes the doc update)

### Existing PFedRec code (refactor target — D-17 surgical-edit applies)

- `federated-pfedrec/pyproject.toml` — Currently `num-supernodes=5`, `partition-mode="natural"`
  (the inconsistency CONCERNS doc flags); PFR-01 flips num-supernodes to 6040, adds
  Phase-3-style mode + run-seed + weight-policy + reuse-cache config keys, adds
  `[project.optional-dependencies] dev = ['pytest>=7.0']` (mirror Phase 3 Plan 02)
- `federated-pfedrec/federated_pfedrec/strategy.py` — Current `SplitFedAvg` /
  `SplitFedProx`; rename to `PFedRecSplitFedAvg` (D-12); update `GLOBAL_PARAM_KEYS`
  / `LOCAL_PARAM_KEYS` per D-01 (bias GLOBAL); drop `PFedRecSplitFedProx` per D-07;
  override `aggregate_evaluate` for sufficient-stat aggregation (BSL-06 / PSN-04 /
  ADP-06 carry-forward)
- `federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py` — Update
  `_GLOBAL_PARAMS` / `_LOCAL_PARAMS` tuples per D-01; `set_local_parameters`
  strict=True per D-21; tensor shape native (1, latent_dim) per D-20
- `federated-pfedrec/federated_pfedrec/client_app.py` — Refactor per-user cache from
  `partition_{id}/user_{uid}/affine_output.pt` to `partition_{pid}.pt` (D-16) +
  manifest sidecar (D-17); add benchmark-mode one-user assertion (PFR-05); thread
  FND-03 exclusion (PFR-04); thread FND-06 RNG (D-02); cold-round probe (D-22)
- `federated-pfedrec/federated_pfedrec/dataset.py` — D-17 rip-and-replace as
  foundation adapter (delegate mapping/split/exclusion to `fedrec_foundation`); raise
  `NotImplementedError` per D-09 on `partition_mode != "natural"` at both
  `load_partition_data` and `load_full_data` entry points; preserve `MovieLensDataset`
  / `download_movielens_1m` / `load_movielens_1m` / `natural_partition_users` per D-18
- `federated-pfedrec/federated_pfedrec/server_app.py` — Currently 587 LOC; refactor
  with: D-25 mode-resolver header, G-03-01 discovery round, ADP-06-style seeded
  partition-id sampling, `PFedRecSplitFedAvg` wire-up, D-15 double-write manifest
  with `module="pfedrec"`, D-13 cold-start counter, D-14 auto-verify hook reading
  `IJCAI-23-PFedRec/sh_result/ml-1m.txt`, D-27 best-round-restore against
  `sampled_ndcg@10`
- `federated-pfedrec/federated_pfedrec/task.py` — D-02 FND-06 RNG factories wired
  through `prepare_user_train_data` and any DataLoader; PFR-04 exclusion set threaded
  into negative-sampling pool; PFR-07 negatives re-sampled every round (no cached
  RNG); D-04 eval BCE loss over positives + 99 negatives

### Paper references (thesis context)

- `Papers/digested/_INDEX.md` — paper knowledge-base index
- `Papers/digested/zhang_2025_survey_personalized_fedrec.md` — personalized FedRec
  taxonomy (positions PFedRec relative to thesis contribution)
- `Papers/digested/yin_2025_devicers_survey.md` — device-recommendation survey

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **`PFedRecMLP` class** (`federated-pfedrec/federated_pfedrec/models/pfedrec_mlp.py`)
  — architecture is unchanged for Phase 5; the Phase 5 work is parameter
  classification (`_GLOBAL_PARAMS` / `_LOCAL_PARAMS` tuples updated per D-01) and
  load-policy strictness (D-21), not a rewrite of the model. `forward()`,
  `predict()`, the dual-LR-aware optimizer logic in task.py — all preserved.
- **`evaluate_pfedrec_sampled()` in task.py** — already implements `sampled_loo_99`;
  Phase 5 changes are: drop `random.seed(seed)` (FND-06 inherits Phase 2 BSL-05 fix
  pattern), accept `rng: np.random.Generator` parameter, and include positives + 99
  negatives in the BCE loss (D-04).
- **`prepare_user_train_data()` in task.py** — already builds per-user training
  batches; Phase 5 changes are RNG injection (D-02) and exclusion-set wiring
  (PFR-04).
- **Phase 2/3/4 sufficient-stat aggregator pattern** — `BaselineFedAvg`,
  `PersonalizedSplitFedAvg`, `AdaptiveSplitFedAvg` all expose the same
  `aggregate_evaluate` signature (sums `hit_count_at_10`, `ndcg_sum_at_10`,
  `evaluated_users`; computes ratio once). `PFedRecSplitFedAvg` clones this verbatim
  with the GLOBAL/LOCAL frozensets flipped per D-01.
- **`fedrec_foundation.manifest.embed_manifest_in_result` /
  `write_manifest_sibling`** — already ship; D-15 double-write applies with
  `module="pfedrec"`, no foundation changes needed.
- **`scripts/clean_cache.py` (Phase 3 D-10)** — handles `schema_version=3` unchanged
  (D-23). No code change.
- **`scripts/foundation/fedrec_foundation/atomic.py::atomic_write_json`** — used by
  D-17 manifest sidecar.
- **G-03-01 discovery round + partition-id sampling**
  (`federated-baseline-cf/.../server_app.py` post-Plan-05; mirrored in
  `federated-personalized-cf/.../server_app.py` and adaptive) — cut-paste reusable.

### Established Patterns

- **D-17 + D-18 (Phase 2):** Rip-and-replace dataset.py helpers with foundation
  loaders; surgical-edit discipline preserves pre-existing uncommitted WIP. Phase 5
  follows.
- **Phase 3/4 manifest-sidecar cache** (`.embedding_cache/{run_id}/manifest.json` +
  `partition_{pid}.pt`, atomic tempfile + `os.replace`, hard-fail on signature
  mismatch with `rm -rf` hint). Phase 5 inherits the structure; bumps
  `schema_version` to 3 and tweaks the field set.
- **D-25 mode-resolver as canonical hyperparam source.** `int(context.run_config.get(
  key, profile.field))` everywhere; pyproject.toml is override-only surface.
  D-10 'allow + log loudly' override visibility.
- **D-27 in-memory best-round-restore.** Snapshot `ArrayRecord` when `current_ndcg >
  best_metric`; restore before final centralized eval / W&B summary write.
- **D-15 double-write manifest.** `embed_manifest_in_result(manifest, results_data)`
  + `write_manifest_sibling(manifest, json_path)`. Phase 5 sets `module="pfedrec"`.
- **Phase 3 Rule-1 fix:** atomic-write tempfile prefix MUST NOT start with `.`
  (PyTorchFileWriter rejects). Use `partition_tmp_*` prefix per Phase 3 precedent.

### Integration Points

- **Installation order**: `pip install -e scripts/foundation/` BEFORE
  `pip install -e federated-pfedrec/`. Already documented in `docs/setup.md`; no
  change for Phase 5.
- **Launcher**: `scripts/run.py pfedrec paper_compat_pfedrec` (the launcher already
  knows the `pfedrec` module dir). Phase 5 just needs server+client+model side to
  consume `mode=paper_compat_pfedrec` correctly.
- **Test location**: `scripts/foundation/tests/` for cross-module subprocess
  determinism guards; `federated-pfedrec/tests/` for module-internal pytest. Same
  layout as Phase 3/4. Phase 5 ships a sibling subprocess test asserting (a)
  `selected_clients_per_round` byte-identity and (b) per-key `torch.equal` on
  `partition_{pid}.pt` cache contents (`affine_output.weight` only after D-01 bias
  move).
- **W&B project**: `federated-cf-cross-device` (D-10). Same bucket as Phases 2/3/4
  cross-device runs. New run_config keys (`reuse-cache`, `weight-policy` in mode
  resolution, `bias_classification` in manifest signature) are logged uniformly.
- **Result file location**: `results/federated/<run_id>_results.json` (flat, Phase 2
  D-28). Sibling manifest `<run_id>-manifest.json` beside it.

### Known Anti-Patterns to Retire (CONCERNS audit closure)

- `federated-pfedrec/federated_pfedrec/server_app.py:250` —
  `random.sample(node_ids, ...)` unseeded. Replace with FND-06 `server_rng(run_seed)`
  and partition-id-space sampling (PFR-06; G-03-01 fix).
- `federated-pfedrec/federated_pfedrec/task.py:130` —
  `rng = random.Random(seed)` re-seeded per call (frozen training negatives across
  rounds). Replace with `np_rng(run_seed, user_idx, round_num, "train_neg")` per
  D-02 / PFR-07.
- `federated-pfedrec/federated_pfedrec/strategy.py:19-22` —
  `affine_output.bias` in `LOCAL_PARAM_KEYS`. Move to `GLOBAL_PARAM_KEYS` per D-01.
- `federated-pfedrec/federated_pfedrec/client_app.py:60-66` —
  `cache_dir = _MODULE_DIR.parent / ".embedding_cache"` unscoped, contaminates new
  experiments. Replace with run-id-scoped path per D-16.
- `federated-pfedrec/federated_pfedrec/client_app.py:147` —
  `torch.load(..., weights_only=False)`. Switch to `weights_only=True` (PyTorch 2.6+
  default-safe; payload is plain tensors).
- `federated-pfedrec/federated_pfedrec/task.py` user_positives building —
  builds from trainloader only; held-out test positive can be drawn as training
  negative. Thread FND-03 `ExclusionTable` per PFR-04.

</code_context>

<specifics>
## Specific Ideas

- "PFedRec is what the IJCAI-23 paper says PFedRec is." Strong preference for
  paper-faithfulness over apples-to-apples cross-module hyperparam unification: D-05
  (paper_compat only), D-06 (fraction-train=1.0), D-07 (drop FedProx), D-19
  (Kaiming default init), D-24 (uniform weight policy). Each individually defensible;
  taken together they say "the Flower PFedRec is the published reference, not a
  PFedRec-shaped variant."
- The `affine_output.bias` GLOBAL/LOCAL flip (D-01) is treated as the headline PFR-02
  audit row — confirmed by tracing `engine.py:143` (`del round_participant_params[
  user]['affine_output.weight']`) which leaves `affine_output.bias` intact in the
  per-user dict that gets aggregated server-side. CONCERNS.md flagged this as
  divergence #9; Phase 5 closes it.
- Auto-verify against `IJCAI-23-PFedRec/sh_result/ml-1m.txt` (D-14) makes PFR-08
  closure a one-shot machine-checkable event — "did this run reproduce the paper
  within ±2?" The reference file already has two runs to compare against, both within
  tight tolerance, so picking the canonical target is a small planner decision.
- The `bias_classification='global'` sentinel field in the schema_v3 manifest (D-17)
  is a deliberate regression-guard: any future maintainer who reverts D-01 (moves
  bias back to LOCAL) gets a hard-fail on the cache load with a clear error pointing
  at the audit decision they're undoing. Same defensive pattern as Phase 4's
  `bias_classification`-style fields in schema_v2.

</specifics>

<deferred>
## Deferred Ideas

### Belongs to Phase 6 (Evaluation & Reporting Harness)

- Per-user-group (sparse / medium / dense) HR@10 / NDCG@10 first-class fields in
  PFedRec result artifacts. PFR-08 reproduction is overall-only; Phase 6 EVL-02 makes
  per-group reporting uniform across all 4 modules.
- `ndcg@10/sparse` / `ndcg@10/medium` / `ndcg@10/dense` named keys in the result
  JSON. PFedRec emits sufficient stats (D-26 carry-forward) but the per-group
  splitting harness is owned by Phase 6.

### Belongs to Phase 7 (Thesis Evaluation Run)

- Multi-seed reproduction (≥3 seeds, mean ± std) for PFedRec — Phase 7 THS-02
  explicitly handles this. Phase 5 PFR-08 is single-seed.
- PFedRec sensitivity ablations (dim sweep, lr sweep, lr-eta sweep) for thesis
  discussion section. D-15 strictly locks the paper-compat config; ablations use
  D-11 CLI overrides, but the SWEEP is Phase 7 territory.
- Standardized `thesis_crossdevice_main` config that all 4 modules run under. Phase 5
  ships paper_compat_pfedrec only (D-05); Phase 7 decides how PFedRec's row appears
  in the comparison table given the per-module config asymmetry.

### Belongs to v2 (deferred beyond this thesis cycle)

- Held-out validation split for early stopping (CONCERNS bug #2: monitor-test
  information leak). D-08 carries forward Phase 2/3/4 D-27 monitor-test pattern;
  val-split is correct ML practice but disrupts the foundation contract (FND-02 split
  manifest) and would invalidate cached Phases 2/3/4 results. v2.
- Differential privacy / privacy quantification — v2 DP-01..02.
- Shared `fedrec_common/` extraction — v2 REF-01..02. Currently the four modules
  duplicate dataset.py / early_stopping.py / strategy.py patterns.
- ML-10M / ML-20M generalization — v2 EXT-01.

### Frozen (not re-derived under cross-device)

- Cross-silo (`partition_mode="dirichlet"`) results for `federated-pfedrec`. D-09
  raises `NotImplementedError` in that mode; pre-Phase-5 commits remain the
  authoritative artifact for any legacy re-run.

### Reviewed Todos (not folded)

- **`phase2-baseline-determinism-path-bug.md`** (score 0.6, keyword-matched) —
  Phase 2 baseline `test_selected_partitions_byte_identical_across_subprocess_reruns`
  asserts `repo_root/results/federated/` but `scripts/run.py baseline
  benchmark_cross_device` writes to `parents[3]/results/federated/`. The todo's own
  text scopes it to Phase 2 baseline (`federated-baseline-cf/tests/test_server_integration.py`)
  and explicitly notes "Not a Phase 3 regression"; not a PFedRec issue. Belongs to
  a future `/gsd:plan-phase 2 --gaps` if the slow gate is re-enabled in CI.

</deferred>

---

*Phase: 05-pfedrec-migration-reproduction*
*Context gathered: 2026-04-28*
