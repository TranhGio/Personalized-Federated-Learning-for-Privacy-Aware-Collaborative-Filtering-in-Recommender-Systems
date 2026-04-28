# Phase 5: PFedRec Migration & Reproduction - Research

**Researched:** 2026-04-28
**Domain:** Cross-device migration + reference-implementation reproduction (IJCAI-23 PFedRec)
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### PFR-02 Reference divergence resolution

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

#### Mode profile design

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

#### Per-user cache layout

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

#### Aggregation weight policy

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

### Deferred Ideas (OUT OF SCOPE)

#### Belongs to Phase 6 (Evaluation & Reporting Harness)

- Per-user-group (sparse / medium / dense) HR@10 / NDCG@10 first-class fields in
  PFedRec result artifacts. PFR-08 reproduction is overall-only; Phase 6 EVL-02 makes
  per-group reporting uniform across all 4 modules.
- `ndcg@10/sparse` / `ndcg@10/medium` / `ndcg@10/dense` named keys in the result
  JSON. PFedRec emits sufficient stats (D-26 carry-forward) but the per-group
  splitting harness is owned by Phase 6.

#### Belongs to Phase 7 (Thesis Evaluation Run)

- Multi-seed reproduction (≥3 seeds, mean ± std) for PFedRec — Phase 7 THS-02
  explicitly handles this. Phase 5 PFR-08 is single-seed.
- PFedRec sensitivity ablations (dim sweep, lr sweep, lr-eta sweep) for thesis
  discussion section. D-15 strictly locks the paper-compat config; ablations use
  D-11 CLI overrides, but the SWEEP is Phase 7 territory.
- Standardized `thesis_crossdevice_main` config that all 4 modules run under. Phase 5
  ships paper_compat_pfedrec only (D-05); Phase 7 decides how PFedRec's row appears
  in the comparison table given the per-module config asymmetry.

#### Belongs to v2 (deferred beyond this thesis cycle)

- Held-out validation split for early stopping (CONCERNS bug #2: monitor-test
  information leak). D-08 carries forward Phase 2/3/4 D-27 monitor-test pattern;
  val-split is correct ML practice but disrupts the foundation contract (FND-02 split
  manifest) and would invalidate cached Phases 2/3/4 results. v2.
- Differential privacy / privacy quantification — v2 DP-01..02.
- Shared `fedrec_common/` extraction — v2 REF-01..02. Currently the four modules
  duplicate dataset.py / early_stopping.py / strategy.py patterns.
- ML-10M / ML-20M generalization — v2 EXT-01.

#### Frozen (not re-derived under cross-device)

- Cross-silo (`partition_mode="dirichlet"`) results for `federated-pfedrec`. D-09
  raises `NotImplementedError` in that mode; pre-Phase-5 commits remain the
  authoritative cross-silo PFedRec artifact.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PFR-01 | `pyproject.toml` defaults to `num-supernodes=6040` and `partition-mode="natural"` | §Existing PFedRec State (current=5); §Phase 2/3/4 Plan 02 pattern (clone the toml block) |
| PFR-02 | Re-audit divergence; produce diff table; decide each row keep-flower / align-to-reference | §Reference Audit (Definitive Diff Table); 8 rows traced cell-by-cell to engine.py line numbers |
| PFR-03 | Per-user head saved/loaded as one atomic artifact keyed by stable `user_idx`; hard-fail on schema/shape mismatch | §Cache Layout Migration (D-16, D-17, D-21); Phase 3 manifest-sidecar pattern verbatim with schema_version=3 |
| PFR-04 | Training negatives exclude held-out test positive (FND-03); unit test asserts | §FND-03 ExclusionTable wiring; Phase 2/3/4 Plan 03 thread-through pattern |
| PFR-05 | Client-side partition-loop collapses to single-user path in benchmark mode | §Single-User Client Refactor; current loop at `client_app.py:249` (over `user_train_data.items()`) collapses to one iteration |
| PFR-06 | Server sampling + evaluator RNG + sufficient-stat aggregation match foundation contract | §Server Migration Pattern (G-03-01 + ADP-06 + BSL-06); Phase 4 Plan 5 6-delta template |
| PFR-07 | Training negatives re-sampled every round (not cached) | §FND-06 RNG Threading (D-02); per-round `np_rng(run_seed, user_idx, round_num, "train_neg")` replaces module-level `random.Random(seed)` |
| PFR-08 | HR@10 / NDCG@10 within ±2 absolute points of IJCAI-23 reference | §Reference Targets (sh_result/ml-1m.txt); §Auto-Verify Hook (D-14); §Reproduction Sensitivity Analysis |
| PFR-09 | Module logs FND-07 protocol fingerprint | §D-15 Double-Write Manifest with `module="pfedrec"` |
</phase_requirements>

## Summary

This is a **migration + reproduction phase**, not a greenfield phase. The federated-pfedrec module already exists and approximates the IJCAI-23 reference, but with eight known bugs (CONCERNS.md §1-9) — one of them (`affine_output.bias` LOCAL vs GLOBAL, divergence #9) is the headline PFR-02 audit item.

The work has three clean layers:

1. **PFR-02 audit table** — trace `IJCAI-23-PFedRec/engine.py` and `mlp.py` end-to-end and document every divergence between Flower and reference. CONTEXT.md already locks 4 audit decisions (D-01 bias GLOBAL, D-02 RNG-per-round, D-04 eval BCE over 99 negs, D-24 uniform weight); this research confirms those + surfaces the residual divergences.
2. **Cross-device migration** — the Phase 2/3/4 template applies almost verbatim. The migration touches the same six files (pyproject.toml, dataset.py, strategy.py, models/pfedrec_mlp.py, client_app.py, server_app.py, task.py) with the same five Wave-1 disjoint-file ownership splits. Phase 4 Plan 5 documented the canonical "6-delta-over-Phase-3" migration shape for server_app.py; Phase 5 will be 5-delta-over-Phase-4 (different strategy class, different manifest module flag, different cache schema, no per-round prototype/alpha bookkeeping, plus the unique D-14 auto-verify hook).
3. **PFR-08 reproduction validation** — auto-verify against `IJCAI-23-PFedRec/sh_result/ml-1m.txt` (HR=0.7286/0.7315, NDCG=0.4407/0.4453, ±2 tolerance is generous given reference run-to-run variance is <0.005).

**Primary recommendation:** Plan 01 owns strategy.py + models/pfedrec_mlp.py (D-01 bias-GLOBAL flip + strict=True); Plan 02 owns pyproject.toml + dataset.py + foundation mode.py D-25 update; Plan 03 owns client_app.py + task.py (D-02/D-03/D-04/D-16/D-17/D-21/D-22 + FND-03 + FND-06 + per-user one-iteration collapse); Plan 04 owns server_app.py main loop + D-14 auto-verify hook; Plan 05 owns subprocess determinism guard. Five plans, three waves (W1: 01+02 parallel; W2: 03; W3: 04+05 parallel). The single highest-risk lever for PFR-08 is D-01 bias-GLOBAL — every other change is a determinism / correctness fix, not a numerics lever.

## Standard Stack

### Core (already installed; no version verification needed — these are pinned by Phase 1)

| Library | Version (verified) | Purpose | Why Standard |
|---------|--------------------|---------|--------------|
| `flwr[simulation]` | `>=1.22.0` | Federated orchestration via `Grid.send_and_receive` API | Used by all four modules; Phase 1 locked |
| `torch` | `>=2.7.1` | `nn.Module` (Embedding + Linear), `optim.SGD`, `BCELoss` | Reference implementation uses torch; matching = paper-faithful |
| `pandas` | `>=2.0.0` | ML-1M `ratings.dat` parsing (kept verbatim in dataset.py per D-18) | Already in dataset.py |
| `numpy` | `>=1.24.0` | `np.random.default_rng` via `fedrec_foundation.rng.np_rng` (FND-06) | Foundation locked Phase 1 |
| `wandb` | `>=0.16.0` | Cross-device runs log to `federated-cf-cross-device` project (D-10) | Phase 2/3/4 default |
| `pytest` | `>=7.0` | Tests under `federated-pfedrec/tests/` and `scripts/foundation/tests/` | Inherited via Plan 02 `[project.optional-dependencies] dev` block |
| `fedrec-foundation` | local-path dep | `mode`, `manifest`, `rng`, `evaluator`, `weight_policy`, `fit_metrics`, `bundle`, `split`, `exclusion`, `atomic` modules | Phase 1 locked; consumed verbatim |

**Version verification:** No new packages are added in Phase 5. Every dependency in the stack already ships in `federated-pfedrec/pyproject.toml` (verified at line 14-26 of the existing file). Phase 5 ADD only `fedrec-foundation` as a local-path dep (already declared at line 17) and `pytest>=7.0` under `[project.optional-dependencies] dev` (mirror Phase 3 Plan 02 pattern).

### Supporting (foundation modules already available; pure consumer)

| Module | Provided by | Used in Phase 5 for |
|--------|-------------|----------------------|
| `fedrec_foundation.mode.{resolve_mode_defaults, log_mode_and_overrides, assert_benchmark_one_user_per_client}` | Phase 1 Plan 5 | D-25 ModeProfile resolution; D-11 override visibility; PFR-05 single-user assertion |
| `fedrec_foundation.rng.{np_rng, server_rng, torch_gen}` | Phase 1 Plan 4 | D-02 per-round training negatives (PFR-07); seeded server sampler (PFR-06); deterministic DataLoader |
| `fedrec_foundation.manifest.{build_run_manifest, embed_manifest_in_result, write_manifest_sibling, generate_run_id}` | Phase 1 Plan 4 | PFR-09 protocol fingerprint with `module="pfedrec"`; D-15 double-write |
| `fedrec_foundation.evaluator.get_primary_evaluator` | Phase 1 Plan 3 | PFR-06 primary-evaluator assertion (`sampled_loo_99`) |
| `fedrec_foundation.weight_policy.{WeightPolicy, compute_aggregation_weight}` | Phase 1 Plan 3 | D-24 `weight_policy="uniform"` resolution |
| `fedrec_foundation.fit_metrics.{FitMetricsContract, EvaluateMetricsContract, validate_*}` | Phase 1 Plan 3 + Phase 2 Plan 1 | D-21 wire-payload contracts (FitMetricsContract for train, EvaluateMetricsContract for evaluate); G-03-01 partition_id field |
| `fedrec_foundation.bundle.verify_bundle` + `fedrec_foundation.split.load_split_manifest` + `fedrec_foundation.exclusion.{ExclusionTable, load_exclusion}` | Phase 1 Plan 2 | FND-03 exclusion threading (PFR-04); FND-07 fingerprints (PFR-09) |
| `fedrec_foundation.atomic.atomic_write_json` | Phase 1 Plan 2 | D-17 manifest sidecar write |
| `fedrec_foundation.paths.data_derived` | Phase 1 Plan 2 | Foundation bundle path resolution |

### Alternatives Considered

| Instead of | Could Use | Why we don't |
|------------|-----------|--------------|
| `weight_policy="uniform"` (D-24) | `weight_policy="num_positives"` | Reference `engine.py:81` divides by `len(round_user_params)` — exactly uniform. Reproducing that is the whole point of paper_compat. |
| Move `affine_output.bias` to GLOBAL (D-01) | Keep both LOCAL (current Flower) | Reference `engine.py:143` deletes ONLY `affine_output.weight` from the per-user dict before aggregation; the bias travels and is averaged. The ±2-point reproduction depends on this. |
| Drop FedProx (D-07) | Keep `SplitFedProx` | Reference uses FedAvg only. PFedRec proximal term scope would be a single tensor (`embedding_item.weight`) — not interesting; not in any sensitivity ablation worth shipping. |
| `fraction-train=1.0` (D-06) | `fraction-train=0.1` (cross-module default) | Reference uses `clients_sample_ratio=1.0`. ±2 reproduction requires full participation; partial participation injects sampling variance the reference doesn't have. |
| Eval BCE over positives + 99 negs (D-04) | Eval BCE over positives only (current Flower) | Reference `engine.py:195-196` concatenates `(test_score, negative_score)` before BCE — 100 items per user. This is a diagnostic fix; HR@10/NDCG@10 are unchanged. |

**Installation:**
```bash
# Foundation must be installed first (already documented in docs/setup.md)
pip install -e scripts/foundation/
pip install -e federated-pfedrec/
# Or with dev dependencies after Plan 02 lands:
pip install -e "federated-pfedrec[dev]"
```

## Architecture Patterns

### Phase 5 Folder Layout (mirrors Phase 4)

```
federated-pfedrec/
├── pyproject.toml                       # Plan 02: cross-device defaults + mode/run-seed/weight-policy/reuse-cache + [dev] pytest
├── federated_pfedrec/
│   ├── __init__.py
│   ├── strategy.py                      # Plan 01: PFedRecSplitFedAvg (drop FedProx D-07); GLOBAL_PARAM_KEYS adds affine_output.bias D-01
│   ├── dataset.py                       # Plan 02: rip-and-replace as foundation adapter; D-09 NotImplementedError
│   ├── client_app.py                    # Plan 03: mode resolve + one-user assert + manifest cache + FND-03 + FND-06 + discover_only + per-user single-iteration collapse PFR-05
│   ├── server_app.py                    # Plan 04: G-03-01 discovery + ADP-06 sampler + PFedRecSplitFedAvg + D-13 cold-start + D-14 PFR-08 auto-verify + D-15 manifest module=pfedrec
│   ├── task.py                          # Plan 03: prepare_user_train_data + train_pfedrec_single_user with FND-06 RNG + FND-03 exclusion + D-04 eval BCE; D-19 Linear-default init
│   ├── early_stopping.py                # Untouched (D-18 surgical)
│   └── models/
│       ├── __init__.py
│       ├── pfedrec_mlp.py              # Plan 01: _GLOBAL_PARAMS adds affine_output.bias D-01; set_local_parameters strict=True D-21
│       └── losses.py                    # Untouched (BCE only)
└── tests/
    ├── conftest.py                      # Plan 02: pytestmark skip-if-foundation-missing
    ├── test_strategy.py                 # Plan 01: bias-GLOBAL invariant + sum-not-mean aggregate_evaluate (uniform weight)
    ├── test_models.py                   # Plan 01: D-21 strict=True shape mismatch fires; D-19 Linear-default init regression guard
    ├── test_dataset_adapter.py          # Plan 02: D-09 NotImplementedError fires at both load_partition_data + load_full_data
    ├── test_task_rng.py                 # Plan 03: stdlib random eradication; FND-06 threading; FND-03 exclusion-in-training-negatives; D-04 eval BCE includes negs
    ├── test_client_assertion.py         # Plan 03: PFR-05 one-user assert; primary-evaluator==sampled_loo_99; D-21 strict-contract; partition_id; bias-classification sentinel D-17
    ├── test_embedding_cache_manifest.py # Plan 03: D-16/D-17/D-21 + bias_classification field; reuse-cache sig path; affine_output.weight (1, latent_dim) shape
    └── test_server_integration.py       # Plan 04: ADP-06 RNG + uniform-weight strategy + D-14 auto-verify pass/fail synthetic + D-15 manifest module=pfedrec + D-13 cold-start + D-02-mirror cross-silo guard
```

### Pattern 1: PFR-02 Reference Audit Table (the canonical PFR-02 deliverable)

**What:** A markdown table comparing Flower PFedRec to `IJCAI-23-PFedRec/engine.py` + `mlp.py` row by row.
**When to use:** PFR-02 explicitly asks for this artifact in the result. Lives at `.planning/phases/05-pfedrec-migration-reproduction/PFR-02-AUDIT.md` (planner picks final path) and is referenced from the result JSON's `_manifest.audit_doc` field (planner discretion).
**Definitive table** (each row traced to a specific reference line + a specific Flower line):

| # | Row | Flower (current) | Reference (engine.py / mlp.py) | Decision | Action | CONTEXT pin |
|---|-----|------------------|--------------------------------|----------|--------|-------------|
| 1 | `affine_output.bias` classification | `LOCAL_PARAM_KEYS` (strategy.py:19-22) | GLOBAL — `engine.py:143` deletes only `affine_output.weight` from `round_participant_params`; `affine_output.bias` is aggregated server-side via `aggregate_clients_params` (engine.py:66-81) | **align-to-reference** | Move to `_GLOBAL_PARAMS` (model) + `GLOBAL_PARAM_KEYS` (strategy) | D-01 |
| 2 | Aggregation weight policy | inherited from FedAvg = num_examples-weighted | uniform — `engine.py:81` divides by `len(round_user_params)`; every contributing user weight=1 | **align-to-reference** | `weight_policy="uniform"` in mode.py profile + uniform-weighted `aggregate_evaluate` override in PFedRecSplitFedAvg | D-24, D-25 |
| 3 | Per-round client participation | `fraction_train` config-driven (currently 1.0; potentially other defaults) | full-participation — `engine.py:87-91` + reference uses `clients_sample_ratio=1.0` (train.py:14) | **align-to-reference** | `fraction_train=1.0` locked in profile | D-06 |
| 4 | Eval BCE loss scope | computed on positives only — `task.py:432` `predictions = model(item_tensor)` over `test_items` only | computed over (positives + 99 negatives) — `engine.py:195-196` `ratings_pred = torch.cat((test_score, negative_score))` | **align-to-reference** | Extend `test_pfedrec` (or replace with eval-time loss path) to compute BCE over the same 100-item candidate pool that drives HR/NDCG | D-04 |
| 5 | Training-negative resampling | static — `task.py:130` `rng = random.Random(seed)` re-seeded per call → frozen across rounds | per-epoch — `engine.py` calls `instance_user_train_loader` from `store_all_train_data` (data.py:83-117) which calls `random.sample(...)` each round (train.py:98 `all_train_data = sample_generator.store_all_train_data(...)` is INSIDE the round loop) | **align-to-reference** | Replace with `np_rng(run_seed, user_idx, round_num, "train_neg")` per-round per-user | D-02, PFR-07 |
| 6 | Held-out test positive in training-negative pool | trainloader-only (`task.py:137-142`) — held-out test item CAN be drawn as a negative | reference's `_split_loo` (data.py:65-73) removes the test item from train; `_sample_negative` (data.py:75-81) operates on `interacted_items` set which DOES include the test item — so reference ALSO has this leak by construction | **align-to-reference** (BUT reference is wrong here per FedRec literature; the foundation contract FND-03 fixes it) | Thread `ExclusionTable.for_user(partition_id)` into `prepare_user_train_data` neg pool — strictly stricter than reference but PFR-04 mandates it | PFR-04, FND-03 |
| 7 | Server-side client sampling RNG | unseeded `random.sample` (server_app.py:250) | unseeded `random.sample` (engine.py:89-91) | **strictly-better-than-reference** | Replace with `_server_sampler = server_rng(run_seed)` (PFR-06; G-03-01 partition-id space) | PFR-06 |
| 8 | Best-round checkpoint / metric reported | last-round metrics; early-stopping records `best_round` but doesn't restore arrays (CONCERNS bug #7) | best validation HR@10 round (train.py:123-125 `if val_hit_ratio >= best_val_hr: final_test_round = round`) — reference reports `hit_ratio_list[final_test_round]` | **align-to-reference (with adaptation)** — reference uses val split; we don't (D-08), so monitor test directly via D-13 best-round-restore on `sampled_ndcg@10` | D-08 + D-13 (D-27 carry-forward); accept val-split deferral to v2 | D-08, D-13 |
| 9 | Init scheme | currently no explicit reset (PFedRecMLP uses nn.Linear/nn.Embedding defaults — Kaiming) | no Xavier (mlp.py: nn.Linear/nn.Embedding defaults — Kaiming) | **align-to-reference** | Don't add Xavier (cross-module convention NOT mirrored here per D-19) | D-19 |
| 10 | Loss function | BCE — `task.py:226` `criterion = nn.BCELoss()` | BCE — `engine.py:30` `self.crit = torch.nn.BCELoss()` | **already-aligned** | No change | — |
| 11 | Optimizer | dual SGD with `lr * num_items * lr_eta` for item embedding — `task.py:229-238` | dual SGD with same scheme — `engine.py:114-119` | **already-aligned** | No change | — |
| 12 | Per-batch update order | (a) update affine_output, (b) update embedding_item — `task.py:254-274` | (a) update affine_output, (b) update embedding_item — `engine.py:51-63` | **already-aligned** | No change | — |
| 13 | Local epochs | 1 (config default) | 1 (train.py:17 default) | **already-aligned** | No change | — |
| 14 | Per-user state persistence | per-user `affine_output.pt` files at `partition_{id}/user_{uid}/affine_output.pt` (client_app.py:60-66) | in-process `self.client_model_params[user]` dict (engine.py:138-140 + 26) | **align-to-reference (storage layer agnostic)** — semantically equivalent (per-user persistence between rounds) but Flower simulation needs a disk artifact since per-user state can't all live in RAM at 6040 supernodes | Use D-16/D-17 manifest-sidecar cache; one file per partition (= one user in cross-device) | D-16, D-17 |

**Anti-Pattern (caught by audit):** Plain `BPRMF`-style `set_local_parameters(strict=False)` partial-load semantics under shape mismatch. PFedRec's per-user `affine_output.weight` is `(1, latent_dim)` and shape mismatch means a config drift that MUST hard-fail (D-21). The current `pfedrec_mlp.py:171-180` has a 3-way branch (exact / strict raises / silent skip) — replace with strict=True default.

### Pattern 2: Cross-Device Server Migration (the 5-delta-over-Phase-4 template)

**What:** server_app.py refactor template inheriting from Phase 4 Plan 5 with 5 PFedRec-specific deltas.
**When to use:** Plan 04 of Phase 5.
**Source pattern (Phase 4 Plan 5 SUMMARY):**
```python
# Phase 4 server_app.py shape (clone this):
@app.main()
def main(grid: Grid, context: Context):
    # 1. Mode resolve
    mode = str(context.run_config.get("mode", "cross_silo_legacy"))
    profile = resolve_mode_defaults(mode)
    overrides = log_mode_and_overrides(mode, profile, context.run_config)

    # 2. D-02 cross-silo guard (FIRST, before any heavy work)
    if mode == "cross_silo_legacy":
        raise NotImplementedError(
            "PFedRec cross-silo path is FROZEN per Phase 5 D-09. "
            "See .planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md §Deferred."
        )

    # 3. Hyperparam reads (profile fallback)
    num_rounds = int(context.run_config.get("num-server-rounds", profile.num_server_rounds))
    fraction_train = float(context.run_config.get("fraction-train", profile.fraction_train))
    run_seed = int(context.run_config.get("run-seed", 42))
    # ... etc

    # 4. run_id materialize EARLY (Phase 3 Plan 04 pattern)
    run_id = context.run_config.get("run-id") or generate_run_id()

    # 5. W&B init (federated-cf-cross-device default)
    default_project = "federated-cf-cross-device" if mode in ("benchmark_cross_device", "paper_compat_pfedrec") else "federated-cf"
    wandb_project = context.run_config.get("wandb-project") or default_project

    # 6. Initial arrays (split learning: GLOBAL only)
    global_model = get_model(latent_dim=latent_dim)
    arrays = ArrayRecord(global_model.get_global_parameters())  # adds affine_output.bias under D-01

    # 7. Strategy wire-up (PFedRec-SPECIFIC delta 1)
    strategy = PFedRecSplitFedAvg(
        fraction_fit=fraction_train,
        # PFedRec aggregate_evaluate uses uniform-weight (num_examples=1 each)
    )

    # 8. G-03-01 discovery round (Phase 2 Plan 5 pattern - clone)
    all_node_ids = sorted(grid.get_node_ids())
    expected_n = profile.num_supernodes
    discovery_messages = [...]  # broadcast discover_only=True
    discovery_responses = list(grid.send_and_receive(discovery_messages))
    partition_to_node_id = {response.partition_id: response.node_id for response in discovery_responses}
    missing = sorted(set(range(expected_n)) - set(partition_to_node_id.keys()))
    assert not missing, f"Discovery failed for partitions {missing[:5]}..."

    # 9. _server_sampler (single instance, partition-id space)
    _server_sampler = server_rng(run_seed)

    # 10. FL loop with D-13 cold-start + D-26 selected_clients_per_round + D-27 best-round
    selected_clients_per_round: List[List[int]] = []
    cold_starts_per_round: List[int] = []
    best_metric = float("-inf")
    best_round_num = 0
    best_arrays = arrays

    for round_num in range(1, num_rounds + 1):
        # Sample partitions (NOT node_ids)
        selected_pids = _server_sampler.sample(range(expected_n), int(expected_n * fraction_train))
        selected_node_ids = [partition_to_node_id[pid] for pid in selected_pids]
        selected_clients_per_round.append([int(p) for p in selected_pids])

        # D-13 cold-start probe
        cache_root = Path(".embedding_cache") / run_id
        cold_count = sum(1 for pid in selected_pids if not (cache_root / f"partition_{pid}.pt").exists())
        cold_starts_per_round.append(cold_count)
        # ... D-09 reuse-cache short-circuit to 0

        # Train messages
        train_config = ConfigRecord({
            "lr": lr, "lr_eta": lr_eta, "round_num": round_num,
            "run_id": run_id, "reuse_cache": reuse_cache,
        })
        # ... grid.send_and_receive ...

        # Aggregate fit (FedAvg-style)
        fit_results = [(DummyClientProxy(...), FitRes(...)) for response in train_responses]
        aggregated_params, agg_metrics = strategy.aggregate_fit(round_num, fit_results, [])
        # apply aggregated_params to arrays

        # Evaluate messages
        eval_results = [(DummyClientProxy(...), EvaluateRes(...)) for response in eval_responses]
        loss, thesis_metrics = strategy.aggregate_evaluate(round_num, eval_results, [])
        eval_metrics_history[round_num] = dict(thesis_metrics)

        # D-27 best-round snapshot (PFedRec-SPECIFIC delta 4: monitor-test per D-08)
        current_ndcg = thesis_metrics.get("sampled_ndcg@10", float("-inf"))
        if checkpoint_rule in ("best_round_restore", "best_round") and current_ndcg > best_metric:
            best_metric = current_ndcg
            best_round_num = round_num
            best_arrays = ArrayRecord({k: v.detach().clone() for k, v in arrays.to_torch_state_dict().items()})

    # 11. Restore best-round arrays
    if checkpoint_rule in ("best_round_restore", "best_round") and best_round_num > 0:
        arrays = best_arrays

    # 12. Build result + manifest
    # NOTE: NO centralized eval (split learning — server has no LOCAL params)
    final_metrics = eval_metrics_history.get(best_round_num, eval_metrics_history.get(num_rounds, {}))

    # 13. PFedRec-SPECIFIC delta 5: D-14 PFR-08 auto-verify
    target_hr, target_ndcg = _parse_reference_results("IJCAI-23-PFedRec/sh_result/ml-1m.txt")
    delta_hr = abs(final_metrics["sampled_hr@10"] - target_hr)
    delta_ndcg = abs(final_metrics["sampled_ndcg@10"] - target_ndcg)
    if delta_hr <= 2.0 and delta_ndcg <= 2.0:
        print(f"[PFR-08 VERIFIED] Δhr={delta_hr:.4f} Δndcg={delta_ndcg:.4f}")
    else:
        print(f"[PFR-08 FAILED] Δhr={delta_hr:.4f} Δndcg={delta_ndcg:.4f} (tolerance=2.0)")

    # 14. D-15 double-write manifest with module='pfedrec'
    manifest = build_run_manifest(run_id=run_id, mode_profile=profile, run_seed=run_seed,
        ..., module="pfedrec", overrides=overrides)
    embed_manifest_in_result(manifest, results_data)
    write_manifest_sibling(manifest, results_filename)
```

**The 5 PFedRec-specific deltas over the Phase 4 server template:**

1. **Strategy class** — `PFedRecSplitFedAvg` (no FedProx variant per D-07).
2. **Initial arrays composition** — `global_model.get_global_parameters()` returns BOTH `embedding_item.weight` AND `affine_output.bias` under D-01.
3. **No prototype** — no `best_prototype` snapshot/restore (Phase 4 unique). No alpha diagnostics aggregation.
4. **No centralized eval block** — same as Phase 3/4 (split learning); final metrics come from `eval_metrics_history[best_round_num]`.
5. **D-14 auto-verify hook** — NEW to Phase 5, fires after the FL loop and before the JSON write. Reads `IJCAI-23-PFedRec/sh_result/ml-1m.txt`, parses `hr: <float>` and `ndcg: <float>` tokens, asserts `|our - ref| ≤ 2.0`, prints PASS/FAIL line. Final metric source = `final_metrics` (best-round-restored). The `sh_result/ml-1m.txt` file format is line-oriented:
   ```
   2026-04-03 18-13-01-latent_dim: 32-lr: 0.1-clients_sample_ratio: 1.0-num_round: 100-negatives: 4-lr_eta: 80-batch_size: 256-hr: 0.7314569536423841-ndcg: 0.4453293898861762-best_round: 95-optimizer: sgd-l2_regularization: 0.0
   2026-04-03 19-47-11-latent_dim: 32-lr: 0.1-clients_sample_ratio: 1.0-num_round: 100-negatives: 4-lr_eta: 80-batch_size: 256-hr: 0.7286423841059603-ndcg: 0.4407401988138434-best_round: 89-optimizer: sgd-l2_regularization: 0.0
   ```
   Parsing: split on `-`, find `hr: <val>` and `ndcg: <val>` tokens. Two reference runs present; CONTEXT D-14 leaves the canonical-target choice to the planner (recommendation: use the most recent / last line — line 2 — `HR=0.7286, NDCG=0.4407`; tighter target than line 1 because the older run got slightly lucky on the val-split's `best_round` selection).

### Pattern 3: D-17 Manifest Sidecar Schema (schema_version=3 with `bias_classification` sentinel)

**Source pattern:** Phase 3 Plan 03 + Phase 4 Plan 03.

**Phase 5 schema:**
```python
# .embedding_cache/{run_id}/manifest.json (atomic_write_json)
{
  "schema_version": 3,
  "run_id": "20260428-103000-a1b2c3",
  "method": "pfedrec",
  "num_users": 6040,
  "num_items": 3706,
  "latent_dim": 32,
  "split_hash": "5685bed7e4b6...",
  "loss": "bce",
  "num_train_negatives": 4,
  "bias_classification": "global"   # D-01 sentinel — catches future regression
}
```

**`.embedding_cache/{run_id}/partition_{pid}.pt` payload:**
```python
# torch.save with weights_only=True compatible (plain tensors)
{
  "affine_output.weight": Tensor(shape=(1, 32))   # ONLY this key under D-01
}
```

**Hard-fail on mismatch (D-21):** `set_local_parameters(strict=True)` raises `RuntimeError` with per-field delta + literal `Run: rm -rf .embedding_cache/{run_id}/` hint when (a) any of the 9 manifest fields diverges from the live signature, OR (b) `affine_output.weight` shape ≠ `(1, latent_dim)`.

### Pattern 4: PFR-05 Single-User Client Refactor

**Current code (federated-pfedrec/federated_pfedrec/client_app.py:249):**
```python
for user_idx, (user_items, user_ratings) in user_train_data.items():
    # Train this user
    model = get_model(...)
    model.set_global_parameters(global_state)
    load_user_local_params(model, partition_id, user_idx)
    train_pfedrec_single_user(model, user_items, ...)
    save_user_local_params(model, partition_id, user_idx)
    # accumulate item embedding for averaging
```

**After PFR-05 (cross-device, 1 user = 1 partition):**
```python
# Assert exactly one user (PFR-05 + Phase 1 D-11 helper)
assert_benchmark_one_user_per_client(profile, num_users_in_partition, overrides)

user_idx = partition_id  # In cross-device, partition_id IS the user_idx
user_items, user_ratings = next(iter(user_train_data.items()))[1]  # collapse loop

model = get_model(num_items=num_items, latent_dim=latent_dim)
model.set_global_parameters(global_state)  # adds affine_output.bias under D-01

# D-22 probe-then-load
cache_path = _cache_dir_for_run(run_id, reuse_cache, signature) / f"partition_{partition_id}.pt"
cold_round = not cache_path.exists()
if not cold_round:
    state = _load_local_user_state(partition_id, run_id, reuse_cache, signature)
    model.set_local_parameters(state, strict=True)  # D-21
# else: keep PyTorch nn.Linear default init per D-19

# Train (no inner loop)
train_pfedrec_single_user(
    model=model,
    user_items=user_items, user_ratings=user_ratings,
    lr=lr, lr_eta=lr_eta, num_items=num_items,
    local_epochs=local_epochs, batch_size=batch_size, device=device,
    run_seed=run_seed, user_idx=user_idx, round_num=round_num,
    exclude_items=exclusion_table.for_user(user_idx),  # PFR-04 (FND-03)
)

# Persist single-user single-file (D-16)
_save_local_user_state(
    partition_id=partition_id,
    state_dict={"affine_output.weight": model.affine_output.weight.data.cpu().clone()},
    run_id=run_id, reuse_cache=reuse_cache, signature=signature,
)

# Return GLOBAL params (item embeddings + affine_output.bias under D-01)
return ArrayRecord(model.get_global_parameters())
```

**Note:** The current code's `item_embedding_accum / num_trained_users` averaging step at `client_app.py:288-290` becomes UNNECESSARY in cross-device — there's only one user per partition, so no in-partition averaging. The averaging is moved to the server (uniform `aggregate_fit` over 6040 partition-level item embeddings).

### Anti-Patterns to Avoid

- **Don't preserve current `affine_output/<user_uid>/` per-user-subdirectory layout.** D-16 specifies `partition_{pid}.pt` (one file per partition = one user). The 2-level layout is a cross-silo artifact (partition contained many users); cross-device flattens it. `clean_cache.py` already works this way.
- **Don't keep `strict=False` partial-load semantics.** D-21 mandates strict=True. Silent partial-loads under shape mismatch are CONCERNS bug #4 ("per-user cache contaminates new experiments"); the manifest-sidecar D-17 + strict-load D-21 close that bug.
- **Don't carry FedProx into Phase 5.** D-07 drops it. The line in pyproject.toml `proximal-mu = 0.0` should be removed; the `strategy = "fedprox"` config key should be DROPPED (not "allowed but discouraged"); `PFedRecSplitFedProx` class should NOT exist.
- **Don't skip the discovery round.** Cross-device with 6040 partitions and `fraction_train=1.0` still benefits from discovery: the server needs the partition_to_node_id map for ANY partition-id-space sampling, even when "sampling" all 6040. The G-03-01 fix is REQUIRED for byte-identical reruns under the subprocess regression guard.
- **Don't add Xavier init.** D-19 explicitly forbids it. Reference uses Kaiming defaults. Cross-module Xavier in BPRMF/BasicMF/DualPersonalizedBPRMF is paper-COMPETING, not paper-COMPATIBLE; PFedRec stays paper-faithful.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Run-scoped client RNG | Custom `random.Random(seed + user_id + round)` | `np_rng(run_seed, user_idx, round_num, "train_neg")` | FND-06 contract uses `hashlib.sha256` not Python `hash()` for cross-process stability; `_ALLOWED_PURPOSES` whitelist catches typos |
| Manifest fingerprinting | Custom JSON-with-hashes file | `build_run_manifest(...) + embed_manifest_in_result + write_manifest_sibling` | FND-07 uses 4 IMP-2 fingerprints + atomic writes + git_commit; hand-rolled equivalent would re-derive all this |
| Atomic JSON write | Custom tempfile + rename loop | `fedrec_foundation.atomic.atomic_write_json` | Already handles numpy scalars + Path objects; same parent-dir requirement; mkstemp prefix=`.tmp-` |
| Cross-silo guard | Comment "do not run cross-silo" | `if mode == "cross_silo_legacy": raise NotImplementedError(...)` | D-09 mirrors Phase 3/4 D-02; freezes the legacy path before any work happens |
| Per-user mapping | Custom `user2idx` rebuild | `fedrec_foundation.mapping.build_mapping(...)` via `verify_bundle(data_derived())` | FND-01 — canonical 6040-user / 3706-item mapping, locked by foundation_index.json (committed Phase 1 Plan 02) |
| Held-out test exclusion | Compare against `user_test_items` set inline | `ExclusionTable.for_user(user_idx)` from `load_exclusion(data_derived() / "exclusion_items.npz")` | FND-03 — flat int32 + indptr CSR layout; O(1) per-user; pre-computed at Phase 1 Plan 02 commit |
| Cross-process determinism guard | Compare two result JSONs by hand | Subprocess test that runs `scripts/run.py pfedrec paper_compat_pfedrec` twice and asserts byte-identity on `selected_clients_per_round` + `partition_{pid}.pt` files | Phase 2/3/4 Plan 05 precedent (`@pytest.mark.slow` + `FEDREC_SKIP_SLOW=1` escape) |
| Discovery round | Lazy round-1 mapping | Explicit one-shot `discover_only=True` broadcast pre-loop | Plan 02-Plan-05 chose discovery over lazy because lazy creates a chicken-and-egg problem (server has to sample SOME node_ids before knowing partition_ids) |
| Reference parsing | Hand-coded regex over `sh_result/ml-1m.txt` | Same approach BUT keep it in a dedicated `_parse_reference_results(path) -> Tuple[float, float]` helper inside server_app.py | The file format is stable; one-shot reads the last line (or the most-recent-by-timestamp line) and returns `(hr, ndcg)`. NOT something that justifies a separate module — but DOES justify a unit test |

**Key insight:** Phase 5 is a pure CONSUMER of foundation contracts and Phase 2/3/4 patterns. There are no new abstractions to invent. Any "let's build a tiny helper for X" instinct should be checked against existing foundation modules first.

## Runtime State Inventory

> Phase 5 is a migration phase. Several existing artifacts will become stale and must be invalidated explicitly.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| **Stored data** | (1) Existing `.embedding_cache/partition_<pid>/user_<uid>/affine_output.pt` files in `federated-pfedrec/.embedding_cache/` (cross-silo legacy layout). (2) Cross-silo result JSONs at `results/federated/pfedrec/pfedrec_mlp_*.json` from pre-Phase-5 runs. | (1) **Code edit** — D-16 changes the path layout to `.embedding_cache/{run_id}/partition_{pid}.pt`. Existing legacy cache files become orphaned; no automatic migration. Document in module-level claude.md update: "delete old .embedding_cache/ before first cross-device run." (2) **No action** — cross-silo result JSONs are FROZEN per D-09; pre-Phase-5 commits remain authoritative for any legacy re-run. |
| **Live service config** | None. PFedRec doesn't write to any external service (no n8n / Datadog / Cloudflare / pm2). | None — verified by grepping module for HTTP / external-service hooks. |
| **OS-registered state** | None. No Windows Task Scheduler / launchd / systemd hooks. | None. |
| **Secrets/env vars** | (1) `WANDB_API_KEY` (env var, no rename). (2) `WANDB_ENTITY` / `WANDB_PROJECT` (env vars, no rename). (3) `PYTHONHASHSEED` (relevant for FND-06; no rename). | None — Phase 5 doesn't change any env-var names or secret keys; the W&B project DEFAULT changes from `federated-pfedrec` to `federated-cf-cross-device` per D-10, but this is a `wandb_project = context.run_config.get(..., default_project)` change, not an env-var change. |
| **Build artifacts / installed packages** | (1) `federated_pfedrec.egg-info/` (pip's editable-install metadata) — stale if pyproject.toml dependency list changes (Plan 02 adds `[dev]` extra). (2) Foundation package's editable install at `scripts/foundation/` — already installed; no change. | (1) **User setup** — after Plan 02 lands, `pip install -e "federated-pfedrec[dev]"` regenerates the egg-info. Document in PLAN.md verification step. (2) None — foundation has been installed since Phase 1 Plan 06. |

**The canonical question: "After every file in the repo is updated, what runtime systems still have the old string cached, stored, or registered?"**

Answer for Phase 5: only the `.embedding_cache/` directory is at risk. The migration is otherwise self-contained inside one Python module + one foundation module update + one mode-profile field flip.

## Common Pitfalls

### Pitfall 1: D-01 bias-GLOBAL flip without strategy/model symmetry

**What goes wrong:** Move bias to `_GLOBAL_PARAMS` in the model but forget to add it to `GLOBAL_PARAM_KEYS` in strategy.py. Server aggregates only `embedding_item.weight`; client sends both but server discards the bias on round-trip.
**Why it happens:** Two declarations in two files; Phase 3 had a similar structure for `local_user_row` (single-row model in models, frozenset in strategy) and the symmetry was only enforced by tests.
**How to avoid:** Land BOTH changes in Plan 01 (single plan owns both files); add a `test_strategy.py` test asserting `set(GLOBAL_PARAM_KEYS) == set(PFedRecMLP._GLOBAL_PARAMS)` AND that `'affine_output.bias'` is in BOTH.
**Warning signs:** First-round eval HR@10 ≈ 0.10 (random-baseline level); training BCE loss not decreasing.

### Pitfall 2: Reference's `random.sample(... 198)` in `_sample_negative` (data.py:80)

**What goes wrong:** Reference samples 198 negatives at dataset-construction time and splits them 99 (val) + 99 (test) per `validate_data` (data.py:127) and `test_data` (data.py:146). Flower's foundation contract assumes 99 test negatives only and reads them per-call (no val split). If we mistakenly assume the reference samples 99-per-round at the same eval round, we'll inject extra variance.
**Why it happens:** Reading `data.py:75-81` in isolation suggests "reference samples 198 every round" — but actually it samples 198 ONCE at `SampleGenerator.__init__` and the same 99 are reused every round.
**How to avoid:** The eval-side negatives are FROZEN by the reference (sampled once). Our foundation FND-04 evaluator's 99-negs sampling under `np_rng(..., "eval_neg")` is per-round per-user — strictly more variance than reference, but mathematically unbiased and reproducible. Document this in PFR-02 audit row 12 (eval negatives — strictly-more-variance-than-reference). PFR-08 ±2-point tolerance generously absorbs this.
**Warning signs:** Single-seed reproduction lands at HR@10=0.71-0.74 instead of exactly 0.7286-0.7315. This is acceptable per ±2 tolerance but worth documenting.

### Pitfall 3: `lr * num_items * lr_eta = 29,648` looks like an LR bug, isn't

**What goes wrong:** A reviewer (or future Claude) sees the huge effective item LR and "fixes" it by removing the multipliers, breaking reproduction.
**Why it happens:** CLAUDE.md flags this in `Performance Bottlenecks > Large item-embedding learning rate in PFedRec is an intentional trick`. It compensates for sparse item gradients in the large embedding table — reference behavior.
**How to avoid:** Add a comment in `train_pfedrec_single_user` (task.py:234-238) explicitly stating "DO NOT change — matches reference engine.py:117-119 (effective LR = lr * num_items * lr_eta)". The current code already has this; verify Plan 03 doesn't accidentally drop the comment during the rip-and-replace.
**Warning signs:** Item embeddings flatlined at init; HR@10 drops from 0.73 to 0.05; reviewer comment "removed numerically-suspicious multiplier".

### Pitfall 4: Discovery round at 6040 supernodes is slow — but required

**What goes wrong:** "fraction_train=1.0 means we're sampling everyone every round; why do we need a discovery round?" → skip it → subprocess regression guard fails.
**Why it happens:** D-06 says fraction_train=1.0 (full participation), so partition-id-space sampling looks like an over-engineering. But (a) the byte-identity test asserts `selected_clients_per_round` partition-id, NOT node-id, and node_ids are os.urandom-ephemeral, so the ID translation IS load-bearing; (b) Plan 05's subprocess regression guard depends on partition-id stability.
**How to avoid:** Keep the discovery round in server_app.py even at fraction_train=1.0. Phase 2 Plan 05's note ("Discovery-round scale: The first discovery broadcast sends 6040 messages over the Flower grid") confirms it works at 6040 — both runs in the regression-guard tests succeeded with 0 missing partitions.
**Warning signs:** `subprocess regression guard FAILS: selected_clients_per_round bytes differ`; or the result JSON has different `selected_clients_per_round` lists across two same-seed runs.

### Pitfall 5: `weight_policy="uniform"` interaction with FedAvg.aggregate_fit

**What goes wrong:** FedAvg's `aggregate_fit` is num_examples-weighted; setting `weight_policy="uniform"` only affects the strategy's `aggregate_evaluate` (which we override). The server-side AGGREGATION of training updates ALSO needs to be uniform per the reference.
**Why it happens:** Phase 2/3/4 plans inherit `aggregate_fit` from Flower's FedAvg (D-23 identity check). The reference (engine.py:81) divides by `len(round_user_params)` for ALL params, which is uniform-weight aggregation of training updates too.
**How to avoid:** In cross-device with `fraction_train=1.0` + `1 user = 1 partition`, every client's `num_examples` is the count of training samples for that one user (positives + negatives). FedAvg's num_examples weighting therefore weights each partition by `n_user_positives * (1 + num_negatives) ≈ n_user_positives * 5`. Sparse users contribute less weight than the reference's uniform behavior. **Two options for Plan 01:**
  - **Option A (clone):** OVERRIDE `aggregate_fit` in `PFedRecSplitFedAvg` to do uniform mean (mirror Phase 4 Plan 1's `aggregate_fit` override pattern, but for uniform weight instead of prototype-bonus).
  - **Option B (cheaper):** Have clients return `num_examples=1` (uniform) under `weight_policy="uniform"`. Then FedAvg's existing num_examples-weighted aggregate is mathematically uniform. This is what `compute_aggregation_weight(metrics, "uniform")` returns (weight_policy.py:85-86). The cleanest implementation: `FitRes.num_examples = int(compute_aggregation_weight(metrics, profile.weight_policy))`.

  CONTEXT.md doesn't prescribe; planner picks. **Recommendation: Option B** (cleaner: no aggregate_fit override; mirrors how `weight_policy` already works for evaluate aggregation).

  Phase 2 Plan 01 already has the precedent: BaselineFedAvg.aggregate_fit is FedAvg.aggregate_fit unchanged; the weight_policy convention enters via `num_examples` on the FitRes wrapper at server_app.py.

**Warning signs:** PFR-08 reproduction off by ~2-5 points despite all other audit rows being closed; sparse users underperform vs reference.

### Pitfall 6: `torch.load(weights_only=False)` security smell + PyTorch 2.6+ warning

**What goes wrong:** Current code uses `torch.load(filepath, weights_only=False)` (client_app.py:147). PyTorch 2.6+ emits a deprecation warning for `weights_only=False` and will switch the default to `True` in 2.7+. With our pinned `torch>=2.7.1`, the default is already `True` — but explicit `False` still works and emits the warning; keeping it leaks pickle's arbitrary-code-execution semantics.
**Why it happens:** Pre-PyTorch-2.6 code; nobody's flipped it.
**How to avoid:** Plan 03 should switch to `torch.load(filepath, weights_only=True)` (or omit the arg — same effect under torch≥2.7.1). The cache payload is plain tensors (D-20 `affine_output.weight` shape `(1, latent_dim)`), so `weights_only=True` works without pickle hooks. Phase 4 Plan 3 already did this for `_logit_alpha.weight` + `_item_perturbation.weight`; mirror.
**Warning signs:** `UserWarning: torch.load received weights_only=False` in test logs; CI security scan flags pickle loader.

### Pitfall 7: `sh_result/ml-1m.txt` parsing fragility

**What goes wrong:** Auto-verify hook (D-14) parses the txt file with a brittle regex; reference file format changes silently (e.g., adds a new field) and parser breaks.
**Why it happens:** The file is a custom dash-separated format from `train.py:128-133`. There's no schema. New runs append to the file (`with open(file_name, 'a')`) so the file grows but format stays identical.
**How to avoid:** (a) Pin to the LAST line (`with open(...) as f: last = f.readlines()[-1]`); (b) split on `-` and search for tokens `hr: <float>` and `ndcg: <float>` rather than positional indexing; (c) emit clear error message when the file is missing or malformed (D-14: `[PFR-08 FAILED: cannot parse reference at <path>]`); (d) ship a unit test with two synthetic lines (mirror the actual sh_result/ml-1m.txt content) to lock the parser.
**Warning signs:** D-14 hook returns 0.0 / 0.0 silently; or raises uncaught exception killing the run before W&B summary writes.

## Code Examples

Verified patterns from official sources and the existing codebase.

### Reference: Reference Aggregation Step (engine.py:66-81 — D-01 source-of-truth + D-24 source-of-truth)

```python
# Source: IJCAI-23-PFedRec/engine.py
def aggregate_clients_params(self, round_user_params):
    """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
    # aggregate item embedding and score function via averaged aggregation.
    t = 0
    for user in round_user_params.keys():
        # load a user's parameters.
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

Key observations: `for key in user_params.keys()` iterates over EVERY key in the per-user dict. The dict is built earlier at engine.py:138-143 with `del round_participant_params[user]['affine_output.weight']` — so only `affine_output.bias` and `embedding_item.weight` survive. Both get aggregated. Division by `len(round_user_params)` is uniform mean (D-24).

### Reference: Per-User Round Update (engine.py:84-146 — full sampling + dual-LR + bias-GLOBAL trace)

```python
# Source: IJCAI-23-PFedRec/engine.py
def fed_train_a_round(self, all_train_data, round_id):
    if self.config['clients_sample_ratio'] <= 1:
        num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
        participants = random.sample(range(self.config['num_users']), num_participants)
    else:
        participants = random.sample(range(self.config['num_users']), self.config['clients_sample_num'])

    round_participant_params = {}
    all_loss = {}

    for user in participants:
        loss = 0
        model_client = copy.deepcopy(self.model)
        if round_id != 0:
            user_param_dict = copy.deepcopy(self.model.state_dict())
            if user in self.client_model_params.keys():
                for key in self.client_model_params[user].keys():
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
            user_param_dict['embedding_item.weight'] = copy.deepcopy(self.server_model_param['embedding_item.weight'].data).cuda()
            model_client.load_state_dict(user_param_dict)
        # NOTE: affine_output.bias is NOT loaded from server here on round_id==0;
        # it inherits the model's nn.Linear default. On round_id>=1, it's loaded
        # from `self.client_model_params[user]['affine_output.bias']` (the per-user
        # cache) — which is the ROUND-K-1 server-aggregated value, because at the
        # end of round K-1, line 142 stores the post-aggregation server param into
        # `self.client_model_params[user]` via `copy.deepcopy(self.client_model_params[user])`
        # AFTER the server aggregated. So bias travels server <-> client every round.
        optimizer = torch.optim.SGD(model_client.affine_output.parameters(),
                                    lr=self.config['lr'], weight_decay=self.config['l2_regularization'])
        optimizer_i = torch.optim.SGD(model_client.embedding_item.parameters(),
                                      lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'],
                                      weight_decay=self.config['l2_regularization'])
        # ... train ...
        client_param = model_client.state_dict()
        self.client_model_params[user] = copy.deepcopy(client_param)
        for key in self.client_model_params[user].keys():
            self.client_model_params[user][key] = self.client_model_params[user][key].data.cpu()
        round_participant_params[user] = copy.deepcopy(self.client_model_params[user])
        del round_participant_params[user]['affine_output.weight']  # KEY LINE: only weight is deleted; bias stays
    self.aggregate_clients_params(round_participant_params)
    return all_loss
```

### Reference: Eval BCE Includes 99 Negatives (engine.py:195-196 — D-04 source-of-truth)

```python
# Source: IJCAI-23-PFedRec/engine.py
# (from inside fed_evaluate, per user)
test_score = user_model(test_item)
negative_score = user_model(negative_item)
# ... store in test_scores / negative_scores for HR/NDCG ...
ratings_pred = torch.cat((test_score, negative_score))  # 100 items: 1 positive + 99 negatives
loss = self.crit(ratings_pred.view(-1), ratings)  # ratings is [1, 0, 0, ..., 0] (100 elements)
all_loss[user] = loss.item()
```

`ratings` is constructed at engine.py:154-156: `temp = [0] * 100; temp[0] = 1; ratings = torch.FloatTensor(temp)`.

### Phase 4 Plan 5 Pattern: Strategy aggregate_evaluate Override (clone for PFedRecSplitFedAvg)

```python
# Source: federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py
# (canonical sufficient-stat aggregator — Phase 5 clones with weight_policy="uniform")

class PFedRecSplitFedAvg(BaseFedAvg):
    """FedAvg variant with uniform-weight aggregation of (a) item embeddings + bias and (b) sufficient-stat thesis metrics.

    Notes
    -----
    - aggregate_fit is INHERITED from BaseFedAvg unchanged (D-23). FitRes.num_examples
      from clients is set to compute_aggregation_weight(metrics, "uniform") = 1.0,
      so FedAvg's existing num_examples-weighted average is mathematically uniform.
    - aggregate_evaluate is OVERRIDDEN to sum sufficient stats (BSL-06 / PSN-04 / ADP-06).
    """

    def __init__(self, fraction_fit: float = 1.0, **kwargs):
        super().__init__(fraction_fit=fraction_fit, **kwargs)
        self.global_param_keys = GLOBAL_PARAM_KEYS  # frozenset({'embedding_item.weight', 'affine_output.bias'})
        self.local_param_keys = LOCAL_PARAM_KEYS    # frozenset({'affine_output.weight'})
        self._is_split_learning = True

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        sums = _sum_sufficient_stats([metrics for _, metrics in results])
        thesis = _sufficient_stats_to_thesis_metrics(sums)
        loss = sum(m.get("eval_loss", 0.0) for _, m in results) / len(results) if results else 0.0
        return loss, thesis
```

### Phase 3 Pattern: Manifest Sidecar Cache Helpers (clone for Phase 5 with schema_version=3)

```python
# Source: federated-personalized-cf/federated_personalized_cf/client_app.py
# (Phase 5 clones with schema_version=3 + bias_classification field + 9 signature fields)

def _signature_fields(*, run_id, method, num_users, num_items, latent_dim, split_hash,
                      loss="bce", num_train_negatives=4, bias_classification="global"):
    """Schema-v3 signature for PFedRec. 9 fields including the D-01 sentinel."""
    return {
        "schema_version": 3,
        "run_id": run_id,
        "method": method,
        "num_users": num_users,
        "num_items": num_items,
        "latent_dim": latent_dim,
        "split_hash": split_hash,
        "loss": loss,
        "num_train_negatives": num_train_negatives,
        "bias_classification": bias_classification,
    }


def _cache_dir_for_run(*, run_id: str, reuse_cache: bool, signature: Dict) -> Path:
    """D-18: under reuse_cache=True, dir is sig_<sha256[:16]> instead of {run_id}.
    Schema_version + run_id are EXCLUDED from the sig hash so two different run_ids
    with otherwise-matching signatures collide on the same dir."""
    if reuse_cache:
        sig_keys = sorted(signature.keys() - {"run_id", "schema_version"})
        sig_str = json.dumps({k: signature[k] for k in sig_keys}, sort_keys=True)
        sig_hash = hashlib.sha256(sig_str.encode()).hexdigest()[:16]
        return _CACHE_BASE_DIR / f"sig_{sig_hash}"
    return _CACHE_BASE_DIR / run_id


def _save_local_user_state(*, partition_id, state_dict, run_id, reuse_cache, signature):
    cache_dir = _cache_dir_for_run(run_id=run_id, reuse_cache=reuse_cache, signature=signature)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # D-21 shape guard BEFORE disk write
    assert set(state_dict.keys()) == {"affine_output.weight"}, \
        f"D-21 expected single-key payload {{'affine_output.weight'}}, got {set(state_dict.keys())}"
    assert state_dict["affine_output.weight"].shape == (1, signature["latent_dim"]), \
        f"D-21 expected shape (1, {signature['latent_dim']}), got {state_dict['affine_output.weight'].shape}"

    # Manifest first (sidecar)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        atomic_write_json(str(manifest_path), signature)

    # Atomic .pt write (Phase 3 Rule 1 — prefix MUST NOT start with .)
    pt_path = cache_dir / f"partition_{partition_id}.pt"
    fd, tmp = tempfile.mkstemp(dir=str(cache_dir), prefix="partition_tmp_", suffix=".pt")
    os.close(fd)
    try:
        torch.save(state_dict, tmp)
        os.replace(tmp, str(pt_path))
    except Exception:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise


def _load_local_user_state(*, partition_id, run_id, reuse_cache, signature):
    cache_dir = _cache_dir_for_run(run_id=run_id, reuse_cache=reuse_cache, signature=signature)
    pt_path = cache_dir / f"partition_{partition_id}.pt"
    manifest_path = cache_dir / "manifest.json"

    if not pt_path.exists():
        return None  # D-22 cold round (cache miss is silent)

    # D-17 manifest signature check
    with open(manifest_path) as f:
        on_disk = json.load(f)
    diffs = [(k, on_disk.get(k), signature[k]) for k in signature if on_disk.get(k) != signature[k]]
    if diffs:
        msg = "; ".join(f"{k}: on-disk={od!r} vs live={live!r}" for k, od, live in diffs)
        raise RuntimeError(
            f"D-17 manifest mismatch for {pt_path}: {msg}. "
            f"Run: rm -rf {cache_dir}/"
        )

    state = torch.load(pt_path, weights_only=True, map_location="cpu")
    # D-21 shape guard AFTER load
    assert set(state.keys()) == {"affine_output.weight"}
    assert state["affine_output.weight"].shape == (1, signature["latent_dim"])
    return state
```

### D-14 PFR-08 Auto-Verify Hook Sketch

```python
# Source: NEW for Phase 5 — sketch (planner finalizes location and exact format)

def _parse_reference_results(reference_path: Path) -> Tuple[float, float]:
    """Parse IJCAI-23-PFedRec/sh_result/ml-1m.txt and return (HR@10, NDCG@10).

    Picks the LAST line per CONTEXT D-14 'most recent / best of the two reference runs'.
    Format example:
      2026-04-03 19-47-11-...-hr: 0.7286423841059603-ndcg: 0.4407401988138434-best_round: 89-...

    Raises
    ------
    RuntimeError
        If the file is missing or malformed.
    """
    if not reference_path.exists():
        raise RuntimeError(f"PFR-08 reference file not found: {reference_path}")

    with open(reference_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise RuntimeError(f"PFR-08 reference file is empty: {reference_path}")

    target = lines[-1]
    tokens = target.split("-")  # the "-" is the field separator from train.py:128-133
    hr, ndcg = None, None
    for token in tokens:
        if token.lstrip().startswith("hr:"):
            hr = float(token.split(":")[1].strip())
        elif token.lstrip().startswith("ndcg:"):
            ndcg = float(token.split(":")[1].strip())
    if hr is None or ndcg is None:
        raise RuntimeError(f"PFR-08 reference parse failed: {target!r}")
    return hr, ndcg


def _emit_pfr_08_verification(final_metrics: Dict, reference_path: Path,
                               tolerance: float = 2.0) -> Tuple[bool, str]:
    """Emit [PFR-08 VERIFIED] / [PFR-08 FAILED] log line.

    Returns
    -------
    (passed, log_line)
    """
    try:
        ref_hr, ref_ndcg = _parse_reference_results(reference_path)
    except RuntimeError as e:
        return False, f"[PFR-08 FAILED: {e}]"

    our_hr = final_metrics.get("sampled_hr@10", float("nan"))
    our_ndcg = final_metrics.get("sampled_ndcg@10", float("nan"))
    if any(v != v for v in (our_hr, our_ndcg)):  # NaN check
        return False, f"[PFR-08 FAILED: missing metric our_hr={our_hr} our_ndcg={our_ndcg}]"

    # Reference HR/NDCG are 0..1; tolerance is in absolute points (0..100)
    delta_hr_pts = abs(our_hr - ref_hr) * 100.0
    delta_ndcg_pts = abs(our_ndcg - ref_ndcg) * 100.0
    passed = delta_hr_pts <= tolerance and delta_ndcg_pts <= tolerance
    tag = "VERIFIED" if passed else "FAILED"
    return passed, (
        f"[PFR-08 {tag}] our_hr@10={our_hr:.4f} ref_hr@10={ref_hr:.4f} Δhr={delta_hr_pts:.2f}pts | "
        f"our_ndcg@10={our_ndcg:.4f} ref_ndcg@10={ref_ndcg:.4f} Δndcg={delta_ndcg_pts:.2f}pts | "
        f"tolerance={tolerance:.1f}pts"
    )
```

**Note:** The ±2 absolute points target in REQUIREMENTS.md PFR-08 ("HR@10 and NDCG@10 within ±2 points of paper numbers") is ambiguous between 2 absolute fractional units (0.02) and 2 absolute percentage-points (multiplying by 100). Reference HR is 0.7286 (= 72.86%); ±2 percentage points means HR ∈ [0.7086, 0.7486]. ±0.02 in absolute fractional terms is the same range. The auto-verify hook should multiply by 100 and compare to 2.0 to make the log line readable in percentage-point form ("Δ=1.4pts").

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Cross-silo PFedRec (5 supernodes) — current default | Cross-device PFedRec (6040 supernodes) | Phase 5 (this phase) | Reproduces published baseline; methodologically defensible per project memory `crossdevice_research.md` |
| `affine_output.bias` LOCAL — current default | `affine_output.bias` GLOBAL — D-01 | Phase 5 | Closes CONCERNS divergence #9; aligns with reference engine.py:143; ±2-point reproduction lever |
| Static training negatives across rounds (CONCERNS bug #5) | Per-round per-user FND-06 RNG (D-02) | Phase 5 | Closes PFR-07; gives byte-identical reruns under fixed seed |
| Per-user-subdir cache `partition_{id}/user_{uid}/affine_output.pt` | Single-file per partition `partition_{pid}.pt` (D-16) | Phase 5 | Cross-module idiom uniformity; clean_cache.py works unchanged |
| `set_local_parameters(strict=False)` partial-load | `strict=True` hard-fail with rm -rf hint (D-21) | Phase 5 | Closes CONCERNS bug #4 (per-user cache contamination) |
| `weight_policy="num_positives"` placeholder (Phase 1 deferred-confirmation comment in mode.py) | `weight_policy="uniform"` (D-25) | Phase 5 Plan 02 (mode.py update) | Closes Phase 1 deferred decision; aligns with reference engine.py:81 |
| Eval BCE on positives only (CONCERNS bug #8) | Eval BCE on positives + 99 negatives (D-04) | Phase 5 | Diagnostic fix; HR/NDCG numbers unchanged |

**Deprecated/outdated (in this codebase):**

- `SplitFedProx` / `PFedRecSplitFedProx` — removed entirely in Phase 5 per D-07.
- `random.sample(node_ids, ...)` server-side (server_app.py:250) — replaced with seeded sampler.
- `random.Random(seed)` per-call training negatives (task.py:130) — replaced with FND-06 RNG.
- `torch.load(weights_only=False)` (client_app.py:147) — replaced with `weights_only=True` per Pitfall 6.

## Open Questions

1. **Reference target choice (D-14 leaves to planner):** Two reference runs in `sh_result/ml-1m.txt`:
   - Line 1: HR=0.7314569536, NDCG=0.4453293898, best_round=95
   - Line 2: HR=0.7286423841, NDCG=0.4407401988, best_round=89
   - **What we know:** Both runs are within ±0.005 absolute (very tight; well within ±2-point tolerance).
   - **What's unclear:** Which is "canonical"? Most-recent (line 2) is reasonable; higher-of-two (line 1) is paper-tightest reproduction.
   - **Recommendation:** Most-recent (line 2) — it's the one that got `best_round=89` which is closer to the paper's reported "round 89" (REQUIREMENTS.md PFR-08 says "HR@10 ≈ 0.729, NDCG@10 ≈ 0.441 at round 89"). Pinning to line 2 also makes the test deterministic if a future maintainer appends a third run; the parser reads `lines[-1]` and the test assertions are stable.

2. **Aggregate_fit weighting strategy (Pitfall 5):** `weight_policy="uniform"` semantics on the EVALUATE side is clear (we override `aggregate_evaluate`). On the FIT side, it's between Option A (override `aggregate_fit` for uniform mean) and Option B (set `FitRes.num_examples = 1` so FedAvg's existing weighting is mathematically uniform).
   - **What we know:** Both produce the same aggregated tensor.
   - **What's unclear:** Phase 4 Plan 1 OVERRODE `aggregate_fit` (for prototype-bonus side-effect), but Phase 5 doesn't have that need — uniform weight is purely a weighting choice. Option B is cleaner (no override), Option A is more explicit.
   - **Recommendation:** Plan 01 picks Option B. Set `num_examples = int(compute_aggregation_weight(metrics, profile.weight_policy))` on the server's FitRes wrapper inside server_app.py per-round. Test the equivalence with a `test_pfedrec_split_fedavg_aggregate_fit_uniform_with_num_examples_1` test.

3. **Auto-verify hook placement (CONTEXT discretion):** Where in server_app.py does the D-14 hook fire?
   - **What we know:** CONTEXT says "after `embed_manifest_in_result` and before W&B summary write seems natural; planner confirms."
   - **What's unclear:** Should the verification result be embedded in the manifest (`manifest.pfr_08_verification = {...}`) or only printed to stdout / W&B?
   - **Recommendation:** BOTH. (a) Print to stdout as `[PFR-08 VERIFIED]` / `[PFR-08 FAILED: ...]` — visible in CI logs. (b) Embed in result_data as `results_data["pfr_08_verification"] = {"passed": bool, "delta_hr_pts": float, "delta_ndcg_pts": float, "ref_hr": float, "ref_ndcg": float, "ref_path": str}` — survives in the JSON artifact for post-hoc audit. (c) Push to W&B summary as `wandb.run.summary["pfr_08_passed"] = bool` so the dashboard can filter. The hook itself does NOT raise — a failed reproduction should NOT abort the run; the JSON artifact and the log are the audit trail.

4. **Cross-silo-legacy frozen path test:** D-09 says load_partition_data and load_full_data both raise NotImplementedError when `partition_mode != "natural"`. The current Phase 3 / Phase 4 SUMMARY notes show this guard is at the dataset level. But the server_app.py D-02 mirror also fires when `mode == "cross_silo_legacy"` (Phase 3 Plan 4 pattern).
   - **What we know:** Two layers of guard. Both fire BEFORE any work.
   - **What's unclear:** Should the Phase 5 server_app.py also fire on `partition_mode="dirichlet"` directly (before mode resolve), or rely on the mode-level guard?
   - **Recommendation:** Mirror Phase 3 Plan 4. Server raises on `mode == "cross_silo_legacy"` after `log_mode_and_overrides`. Dataset raises on `partition_mode != "natural"` deeper in the call chain. Both layers in place is defensive, not redundant.

5. **Wave layout (CONTEXT discretion):** Phase 4 was 6 plans across 3 waves. Phase 5 has fewer moving parts (no per-user alpha, no item perturbation, no contrastive, no prototype EMA) — likely 5 plans across 3 waves.
   - **Recommendation:**
     - **Wave 1 (parallel):** Plan 01 (strategy + model — D-01 bias-GLOBAL, D-21 strict, model param tuples) + Plan 02 (pyproject.toml + dataset.py + foundation mode.py D-25 update).
     - **Wave 2 (single):** Plan 03 (client_app.py + task.py — FND-03 + FND-06 + D-02/D-03/D-04 + manifest cache D-16/D-17/D-21/D-22 + PFR-05 single-user collapse).
     - **Wave 3 (parallel):** Plan 04 (server_app.py main loop + D-14 PFR-08 auto-verify hook + D-15 manifest module=pfedrec) + Plan 05 (subprocess determinism regression guard + scripts/clean_cache.py schema_v3 sanity check).
   - Test count estimate: Plan 01 (5-6 tests strategy + model), Plan 02 (3-4 tests dataset_adapter + foundation mode update), Plan 03 (12-13 tests like Phase 3), Plan 04 (7-8 tests server integration + D-14 parser + auto-verify), Plan 05 (1 slow subprocess test). Total ~30-32 GREEN tests.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest >= 7.0 |
| Config file | None (existing convention; pytestmark skip-if-bundle-missing in conftest.py per Phase 3 precedent) |
| Quick run command | `cd federated-pfedrec && pytest tests/ -v` |
| Full suite command | `pytest scripts/foundation/tests/ federated-pfedrec/tests/ federated-baseline-cf/tests/ federated-personalized-cf/tests/ federated-adaptive-personalized-cf/tests/` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PFR-01 | pyproject.toml `num-supernodes=6040`, `partition-mode=natural` in BOTH federation blocks | unit (config grep) | `pytest federated-pfedrec/tests/test_dataset_adapter.py::test_pyproject_cross_device_defaults -x` | ❌ Wave 1 (Plan 02) |
| PFR-02 | Audit table exists; bias-GLOBAL invariant + GLOBAL/LOCAL frozenset symmetry | unit | `pytest federated-pfedrec/tests/test_strategy.py::test_bias_global_invariant -x` | ❌ Wave 1 (Plan 01) |
| PFR-02 | Reference parser matches sh_result/ml-1m.txt last line | unit | `pytest federated-pfedrec/tests/test_server_integration.py::test_parse_reference_results_last_line -x` | ❌ Wave 3 (Plan 04) |
| PFR-03 | D-21 strict=True hard-fails on shape mismatch with rm -rf hint | unit | `pytest federated-pfedrec/tests/test_models.py::test_set_local_parameters_strict_true_shape_mismatch -x` | ❌ Wave 1 (Plan 01) |
| PFR-03 | D-17 manifest sidecar fields = 9 with bias_classification='global' sentinel | unit | `pytest federated-pfedrec/tests/test_embedding_cache_manifest.py::test_schema_v3_includes_bias_classification -x` | ❌ Wave 2 (Plan 03) |
| PFR-04 | Held-out test positive never drawn as training negative (FND-03 thread) | unit | `pytest federated-pfedrec/tests/test_task_rng.py::test_train_negatives_exclude_test_positive -x` | ❌ Wave 2 (Plan 03) |
| PFR-05 | client_app.py train path collapses to single-user (no inner loop over user_test_items) | unit (assertion) | `pytest federated-pfedrec/tests/test_client_assertion.py::test_benchmark_one_user_assert_fires -x` | ❌ Wave 2 (Plan 03) |
| PFR-06 | server_rng reproducibility byte-identity (RNG-only, fast) | unit | `pytest federated-pfedrec/tests/test_server_integration.py::test_server_rng_reproducible -x` | ❌ Wave 3 (Plan 04) |
| PFR-06 | PFedRecSplitFedAvg.aggregate_evaluate sum-not-mean (uniform weight semantics) | unit | `pytest federated-pfedrec/tests/test_strategy.py::test_aggregate_evaluate_uniform_sum_not_mean -x` | ❌ Wave 1 (Plan 01) |
| PFR-06 | get_primary_evaluator(paper_compat_pfedrec) == 'sampled_loo_99' | unit | `pytest federated-pfedrec/tests/test_client_assertion.py::test_primary_evaluator_resolver -x` | ❌ Wave 2 (Plan 03) |
| PFR-07 | Training negatives differ across rounds (FND-06 round_num threading) | unit | `pytest federated-pfedrec/tests/test_task_rng.py::test_train_negatives_change_per_round -x` | ❌ Wave 2 (Plan 03) |
| PFR-07 | stdlib random eradicated from task.py + client_app.py + server_app.py | unit (cross-file grep) | `pytest federated-pfedrec/tests/test_task_rng.py::test_random_seed_calls_stripped -x` | ❌ Wave 2 (Plan 03) |
| PFR-08 | Auto-verify pass — synthetic final_metrics within ±2pts of synthetic ref | unit | `pytest federated-pfedrec/tests/test_server_integration.py::test_pfr_08_auto_verify_passed -x` | ❌ Wave 3 (Plan 04) |
| PFR-08 | Auto-verify fail — synthetic final_metrics outside ±2pts | unit | `pytest federated-pfedrec/tests/test_server_integration.py::test_pfr_08_auto_verify_failed -x` | ❌ Wave 3 (Plan 04) |
| PFR-08 (full) | End-to-end paper reproduction with HR ∈ [0.71, 0.75], NDCG ∈ [0.42, 0.46] | manual-only | `python scripts/run.py pfedrec paper_compat_pfedrec` (~3 hours on RTX 5090) | N/A — manual run; no test harness |
| PFR-09 | manifest.json contains module='pfedrec' + 4 IMP-2 fingerprints | unit | `pytest federated-pfedrec/tests/test_server_integration.py::test_build_run_manifest_module_pfedrec -x` | ❌ Wave 3 (Plan 04) |
| PFR-09 | D-15 double-write — both result JSON has _manifest key AND sibling file exists | unit | `pytest federated-pfedrec/tests/test_server_integration.py::test_d15_double_write -x` | ❌ Wave 3 (Plan 04) |
| Determinism (subprocess) | Two same-seed subprocess runs produce byte-identical selected_clients_per_round AND byte-identical partition_{pid}.pt cache files | slow / subprocess | `pytest scripts/foundation/tests/test_pfedrec_determinism.py -v` (with FEDREC_SKIP_SLOW=0) | ❌ Wave 3 (Plan 05) |

**Manual-only justification (PFR-08 full reproduction):** Running 100 rounds × 6040 users at fraction_train=1.0 takes ~3 hours on RTX 5090 (per the reference's logged run time). Embedding this in pytest would either need (a) a full `--keep-after-test` artifact for CI to inspect, or (b) a heavy `@pytest.mark.slow` marker that only runs on a developer's local box. The auto-verify hook in server_app.py is the in-process verification surface. The pytest-side regression coverage is the synthetic-final_metrics tests (PFR-08 auto_verify pass/fail), NOT the actual 3-hour run. The manual run produces the artifact + the log line that closes PFR-08 in the UAT.

### Sampling Rate

- **Per task commit:** `cd federated-pfedrec && pytest tests/ -v` (expected: ~30 tests, < 5 seconds)
- **Per wave merge:** `pytest scripts/foundation/tests/ federated-pfedrec/tests/ federated-baseline-cf/tests/ federated-personalized-cf/tests/ federated-adaptive-personalized-cf/tests/ -v` (full project test surface; ~150-180 tests; < 30 seconds excluding @pytest.mark.slow)
- **Phase gate:** Full suite green + manual `python scripts/run.py pfedrec paper_compat_pfedrec` produces a result JSON with `pfr_08_verification.passed=true` AND `[PFR-08 VERIFIED]` log line (PFR-08 closure).

### Wave 0 Gaps

- [ ] `federated-pfedrec/tests/conftest.py` — pytestmark skip-if-foundation-missing (clone Phase 3 conftest)
- [ ] `federated-pfedrec/tests/test_strategy.py` — bias-GLOBAL invariant (PFR-02) + uniform-weight aggregate_evaluate (PFR-06)
- [ ] `federated-pfedrec/tests/test_models.py` — D-21 strict=True (PFR-03) + D-19 Linear-default init regression
- [ ] `federated-pfedrec/tests/test_dataset_adapter.py` — D-09 NotImplementedError (PFR-01) + foundation bundle integration
- [ ] `federated-pfedrec/tests/test_task_rng.py` — FND-06 RNG threading + FND-03 exclusion + per-round resampling (PFR-04, PFR-07)
- [ ] `federated-pfedrec/tests/test_client_assertion.py` — PFR-05 one-user assert + primary-evaluator + D-21 strict-contract payloads
- [ ] `federated-pfedrec/tests/test_embedding_cache_manifest.py` — D-16/D-17/D-21 + bias_classification sentinel (PFR-03)
- [ ] `federated-pfedrec/tests/test_server_integration.py` — server_rng + uniform-weight strategy + D-14 PFR-08 auto-verify pass/fail + D-15 manifest + D-13 cold-start + D-09 cross-silo guard
- [ ] `scripts/foundation/tests/test_pfedrec_determinism.py` — slow subprocess regression guard (mirror Phase 3 Plan 05's `test_personalized_determinism.py`)
- [ ] Framework install (Plan 02): `pip install -e "federated-pfedrec[dev]"` after pyproject.toml gains `[project.optional-dependencies] dev = ["pytest>=7.0"]` (mirror Phase 3 Plan 02)

## Sources

### Primary (HIGH confidence)

- `IJCAI-23-PFedRec/engine.py` (lines 66-81 aggregate; 84-146 train round; 149-212 evaluate; 195-196 BCE-with-99-negs) — D-01, D-04, D-24 ground truth
- `IJCAI-23-PFedRec/mlp.py` (lines 5-20 MLP class; D-19 Kaiming-default init source)
- `IJCAI-23-PFedRec/data.py` (lines 65-117 split + train data; 75-81 negative sampling; 119-154 val/test data) — eval-negs-frozen-at-init pitfall #2 source
- `IJCAI-23-PFedRec/metrics.py` (full file: MetronAtK uniform per-user HR/NDCG) — D-26 source
- `IJCAI-23-PFedRec/sh_result/ml-1m.txt` (2 lines: HR=0.7286/0.7315, NDCG=0.4407/0.4453) — D-14 PFR-08 target
- `IJCAI-23-PFedRec/train.py` (lines 14-30 args; 94-125 round loop; 128-133 result-line format) — file-format source for D-14 parser
- `scripts/foundation/fedrec_foundation/{mode,manifest,rng,evaluator,weight_policy,fit_metrics,exclusion,atomic}.py` — Phase 1 contracts (read end-to-end)
- `.planning/phases/02-baseline-migration/02-baseline-migration-04-SUMMARY.md` — server template patterns
- `.planning/phases/02-baseline-migration/02-baseline-migration-05-SUMMARY.md` — G-03-01 discovery + subprocess regression guard
- `.planning/phases/03-personalized-migration/03-personalized-migration-03-SUMMARY.md` — manifest-sidecar cache canonical implementation
- `.planning/phases/03-personalized-migration/03-personalized-migration-04-SUMMARY.md` — server cross-device migration with D-13 cold-start
- `.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-05-SUMMARY.md` — adaptive server migration (template for Phase 5 Plan 04)
- `.planning/codebase/CONCERNS.md` — 8+1 PFedRec bugs end-to-end audit (divergence #9 = D-01 anchor)
- `federated-pfedrec/federated_pfedrec/{strategy,client_app,server_app,task,models/pfedrec_mlp,dataset}.py` — current state (refactor target)
- `federated-pfedrec/pyproject.toml` — config surface (refactor target)
- `CLAUDE.md` + `federated-pfedrec/claude.md` — project conventions + module-specific architecture

### Secondary (MEDIUM confidence)

- Memory file `project_crossdevice_research.md` (referenced in compaction memory, not loaded — flagged in init context as known prior research)
- Memory file `project_config_comparison.md` (referenced in compaction memory) — pre-Phase-1 PFedRec ref-vs-Flower comparison; superseded by PFR-02 audit but informs decisions

### Tertiary (LOW confidence)

- None. This is a migration phase grounded in existing code; no external library docs were needed (Flower API is consumed identically to Phase 2/3/4; PyTorch APIs are stable; no new package introduced).

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — all dependencies already present in pyproject.toml; foundation modules already shipped Phase 1; no new packages to verify.
- Architecture (audit table): HIGH — every row traced to a specific reference line + specific Flower line; CONTEXT.md decisions D-01/D-04/D-07/D-19/D-24 align cell-for-cell with the audit.
- Architecture (server template): HIGH — Phase 4 Plan 5 template is documented end-to-end with 5 explicit deltas; pattern is proven across Phase 2/3/4.
- Cache layout (D-16, D-17, D-21): HIGH — Phase 3 Plan 03 + Phase 4 Plan 03 cache-helpers code is direct cloneable; schema_version=3 + bias_classification sentinel is a 1-field add.
- Pitfalls: HIGH — every pitfall is sourced from CONCERNS.md or Phase 2/3/4 SUMMARY findings (not speculation).
- D-14 reference parser: MEDIUM — file format is stable but Python parsing is one-shot; recommend pinning with a unit test on synthetic input. Risk: unexpected file format change breaks parse silently.
- PFR-08 absolute-points unit ambiguity: MEDIUM — REQUIREMENTS.md says "±2 points" without unit; recommendation is to multiply by 100 for percentage-points (Pitfall in §Code Examples §D-14). Planner should confirm the canonical convention with the user if any doubt remains.

**Research date:** 2026-04-28
**Valid until:** 2026-05-28 (30 days; the codebase is stable on this branch and the IJCAI-23 reference is frozen).
