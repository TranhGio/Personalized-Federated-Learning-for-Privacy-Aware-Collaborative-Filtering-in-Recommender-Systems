# Phase 5: PFedRec Migration & Reproduction - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-28
**Phase:** 05-pfedrec-migration-reproduction
**Areas discussed:** Reference divergence resolution (PFR-02), Mode profile for thesis comparison, Per-user cache layout, Aggregation weight policy

---

## Reference divergence resolution (PFR-02)

### Q1 — `affine_output.bias` classification (Bias scope)

| Option | Description | Selected |
|--------|-------------|----------|
| Align-to-reference: bias GLOBAL | Move `affine_output.bias` to GLOBAL_PARAM_KEYS; only `affine_output.weight` stays LOCAL. Matches IJCAI-23 reference exactly. Required for PFR-08 ±2-point reproduction. | ✓ |
| Keep-flower: both LOCAL | Preserve current Flower classification; document deliberate divergence in PFR-02 audit. Risks PFR-08 landing outside ±2. | |
| Mode-conditional | GLOBAL in paper_compat_pfedrec, LOCAL in benchmark_cross_device. Runtime branch in strategy.py. | |

**User's choice:** Align-to-reference: bias GLOBAL
**Notes:** User's free-text annotation: "Which one you recommend?" — clarifying confirmation that the recommended option matches their intent. Recommendation reaffirmed; user's pick aligned with recommendation.

### Q2 — Training-negative resampling (Train neg / PFR-07)

| Option | Description | Selected |
|--------|-------------|----------|
| FND-06 RNG factory per round | Replace static `random.Random(seed)` with `np_rng(run_seed, user_idx, round_num, 'train_neg')`. Phase 2-4 pattern. | ✓ |
| Advance seed by round_num | `random.Random(seed + round_num)` style. Loses cross-process determinism. | |
| Foundation-cached negatives | Pre-sample once per run, cache, reuse. Removes round-to-round variance. | |

**User's choice:** FND-06 RNG factory per round
**Notes:** None.

### Q3 — Cold-start / first-round behavior (Cold start)

| Option | Description | Selected |
|--------|-------------|----------|
| Match Phase 3/4 pattern | Cache-existence probe + `cold_starts_per_round` counter (Phase 3 D-13). | ✓ |
| Reference-faithful round_id gate | Match engine.py:104-110 `if round_id == 0` short-circuit; drop counter. | |
| Both | Phase 3/4 pattern + reference's explicit round_id == 0 branch. | |

**User's choice:** Match Phase 3/4 pattern
**Notes:** None.

### Q4 — Eval-time BCE loss scope (Eval loss)

| Option | Description | Selected |
|--------|-------------|----------|
| Align-to-reference: positives + 99 negatives | Match engine.py:195-196 `ratings_pred = torch.cat((test_score, negative_score))`. Closes PFR-02 audit row. | ✓ |
| Keep-flower: positives only | Document as deliberate deviation; eval BCE is diagnostic, doesn't affect HR@10/NDCG@10. | |
| You decide | Claude's discretion. | |

**User's choice:** Align-to-reference: positives + 99 negatives
**Notes:** None.

---

## Mode profile for thesis comparison

### Q1 — PFedRec mode shipping policy (Mode ship)

| Option | Description | Selected |
|--------|-------------|----------|
| paper_compat_pfedrec only | Ship only paper_compat (dim=32, SGD, BCE). Phase 7 thesis-table reports PFedRec at paper-faithful config; footnote per-module config differences. | ✓ |
| paper_compat + benchmark_cross_device (dual) | Ship both modes; paper_compat for PFR-08, benchmark_cross_device for Phase 7 apples-to-apples row. 2x sweep budget. | |
| benchmark_cross_device only | Drop paper_compat. Risky — PFR-08 reproduction needs paper config. | |

**User's choice:** paper_compat_pfedrec only
**Notes:** None.

### Q2 — `fraction-train` under paper_compat_pfedrec (Fraction)

| Option | Description | Selected |
|--------|-------------|----------|
| Keep 1.0 (paper-faithful, full participation) | All 6040 users selected each round. Required for PFR-08 reproduction. ~3 hours wallclock. | ✓ |
| Drop to 0.1 (sweep-tunable) | 604 users/round; faster but no longer paper-faithful. | |
| 1.0 default + sweep override flag | Lock 1.0 in profile; document --run-config 'fraction-train=0.1' for fast iteration. | |

**User's choice:** Keep 1.0 (paper-faithful, full participation)
**Notes:** None.

### Q3 — FedProx support for PFedRec (FedProx)

| Option | Description | Selected |
|--------|-------------|----------|
| Drop FedProx for PFedRec | Ship only PFedRecSplitFedAvg. Reference doesn't use FedProx; per-user score function doesn't benefit. | ✓ |
| Keep FedProx as opt-in | Ship both PFedRecSplitFedAvg and PFedRecSplitFedProx; default to fedavg. Cross-module symmetry. | |
| FedAvg + FedProx with proximal-mu=0 default | Single strategy class; semantic confusion. | |

**User's choice:** Drop FedProx for PFedRec
**Notes:** None.

### Q4 — Validation split for early stopping (Val split)

| Option | Description | Selected |
|--------|-------------|----------|
| Carry forward Phase 2/3/4: monitor test (no val split) | Match D-27 in-memory best-round-restore against `sampled_ndcg@10` on test set. Reference also monitors test. | ✓ |
| Introduce held-out val split | Correct ML practice but disrupts FND-02 contract; invalidates Phases 2/3/4 cached results; PFR-08 numbers shift. | |
| Add val split for PFedRec only | Local-to-pfedrec val split; defeats PFR-08 reproduction protocol. | |

**User's choice:** Carry forward Phase 2/3/4: monitor test (no val split)
**Notes:** None.

### Q5 — Cross-silo path in `federated-pfedrec` (Cross-silo)

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror Phase 3/4: D-02 NotImplementedError | Raise NotImplementedError at both load_partition_data and load_full_data. Pre-Phase-5 commits = legacy artifact. | ✓ |
| Keep cross-silo operational | Preserve existing num-supernodes=5 / partition_mode=dirichlet path. Diverges from Phase 3/4 freeze. | |
| Mode-conditional | Cross-silo operational only under cross_silo_legacy mode. Adds test surface. | |

**User's choice:** Mirror Phase 3/4: D-02 NotImplementedError
**Notes:** None.

### Q6 — W&B project (W&B proj)

| Option | Description | Selected |
|--------|-------------|----------|
| Shared `federated-cf-cross-device` | Same W&B project Phase 2/3/4 use. Cross-module dashboards plot all 4 modules together. | ✓ |
| Dedicated `federated-cf-pfedrec-reproduction` | Isolate PFR-08 reproduction runs. Cleaner audit but breaks cross-module dashboarding. | |
| Both | Log to BOTH projects. Doubles W&B storage cost. | |

**User's choice:** Shared `federated-cf-cross-device`
**Notes:** None.

### Q7 — CLI override policy under paper_compat_pfedrec (Mode lock)

| Option | Description | Selected |
|--------|-------------|----------|
| Standard D-10 'allow + log loudly' overrides | Phase 1 D-10 contract: any override applied + visible warning + manifest capture. | ✓ |
| Strict lock: refuse all overrides | Block any --run-config override under paper_compat. Tighter reproduction but breaks D-10 cross-module pattern. | |
| Allow but mark run as 'non-reproduction' | Set _manifest.reproduction_attempt=false on override. Adds manifest field. | |

**User's choice:** Standard D-10 'allow + log loudly' overrides
**Notes:** None.

### Q8 — Strategy class naming (Strategy)

| Option | Description | Selected |
|--------|-------------|----------|
| PFedRecSplitFedAvg (rename) | Match Phase 3/4 module-prefixed convention (PersonalizedSplitFedAvg, AdaptiveSplitFedAvg). | ✓ |
| Keep existing SplitFedAvg name | Less rename churn; ambiguous module ownership. | |

**User's choice:** PFedRecSplitFedAvg (rename)
**Notes:** None.

### Q9 — Best-round-restore metric (Best metric)

| Option | Description | Selected |
|--------|-------------|----------|
| sampled_ndcg@10 (carry-forward) | Match Phase 2/3/4 D-27 convention. Cross-module symmetry; thesis-table primary metric. | ✓ |
| sampled_hr@10 (paper-faithful) | Match IJCAI-23 paper's reported metric. Diverges from cross-module convention. | |
| Composite best (NDCG@10 + HR@10) | Track both, restore at max(ndcg+hr) round. Unconventional. | |

**User's choice:** sampled_ndcg@10 (carry-forward)
**Notes:** None.

### Q10 — PFR-08 reproduction protocol (Repro N)

| Option | Description | Selected |
|--------|-------------|----------|
| Single seed for PFR-08 verification | Run paper_compat_pfedrec once with run-seed=42, assert within ±2 of reference. Phase 7 handles multi-seed. | ✓ |
| Multi-seed reproduction (≥3 seeds, mean±std) | Match reference's multi-seed reporting. Adds 3x wallclock. | |
| Single seed for PFR-08 + multi-seed in Phase 7 | Two stages — Phase 5 single-seed close, Phase 7 already-planned multi-seed. | |

**User's choice:** Single seed for PFR-08 verification
**Notes:** None.

### Q11 — Auto-verify against IJCAI-23 reference (Auto verify)

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — emit pass/fail at run end | Server_app reads sh_result/ml-1m.txt at run end, asserts within ±2, prints `[PFR-08 VERIFIED]` or `[PFR-08 FAILED: Δhr Δndcg]`. | ✓ |
| No — manual verification step | Run produces metrics; verification is human task. Avoids file-parse path. | |
| Yes but as a separate script | Server_app unchanged; ship `scripts/verify_pfr08.py`. More flexible; one extra step. | |

**User's choice:** Yes — emit pass/fail at run end
**Notes:** None.

### Q12 — Ablation knob exposure (Ablation)

| Option | Description | Selected |
|--------|-------------|----------|
| Strictly locked; sweeps via D-10 CLI overrides | Pyproject.toml carries paper_compat values only; ablations use --run-config (D-10 logs visibly). | ✓ |
| Expose dim/lr/lr-eta as first-class pyproject knobs | Easier to discover; risks future maintainer changing default and silently invalidating PFR-08. | |
| Strictly locked + separate ablation mode profile | Add `pfedrec_ablation` mode. Defeats Q1's paper-compat-only decision. | |

**User's choice:** Strictly locked; sweeps via D-10 CLI overrides
**Notes:** None.

---

## Per-user cache layout

### Q13 — Cache file path layout (File path)

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 3/4 uniform: `.embedding_cache/{run_id}/partition_{pid}.pt` | Single .pt per partition; cross-module idiom uniformity; clean_cache.py works unchanged. | ✓ |
| PFedRec-specific: `.embedding_cache/{run_id}/partition_{pid}/user_{user_idx}/affine_output.pt` | Preserve current PFedRec layout. Filesystem overhead at 6040 dirs; diverges from PSN-05/ADP-06. | |
| Flat per-user: `.embedding_cache/{run_id}/user_{user_idx}.pt` | Drop partition_{pid} indirection. Breaks Phase 3/4 uniformity. | |

**User's choice:** Phase 3/4 uniform: `.embedding_cache/{run_id}/partition_{pid}.pt`
**Notes:** None.

### Q14 — Cache-signature manifest fields (Signature)

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 3 base + PFedRec-specific (loss, num_train_negatives, bias_classification) | schema_version=3. Most fingerprints; bias_classification sentinel catches D-01 regression. | ✓ |
| Phase 3 base only, schema_version=3 sentinel | (run_id, method, num_users, num_items, latent_dim, split_hash). Lighter. | |
| Maximal: every paper-relevant hyperparam | Adds lr=0.1, lr_eta=80, optimizer='sgd', local_epochs=1. Forces cache delete between every override. | |

**User's choice:** Phase 3 base + PFedRec-specific (loss, num_train_negatives, bias_classification)
**Notes:** None.

### Q15 — Cache reuse policy (Reuse)

| Option | Description | Selected |
|--------|-------------|----------|
| Carry forward Phase 3 D-08/D-09 | Default reuse-cache=false per-run; opt-in `reuse-cache=true` for sig_<hash> dirs. | ✓ |
| Drop reuse-cache for PFedRec | Default behavior only; safer for thesis. Costs ~3 hours per PFR-08 retry. | |
| Reuse-cache always-on (default true) | Skip run-id namespacing. Matches reference but risky. | |

**User's choice:** Carry forward Phase 3 D-08/D-09
**Notes:** None.

### Q16 — `affine_output` first-round init (Init)

| Option | Description | Selected |
|--------|-------------|----------|
| Paper-faithful: PyTorch default (Kaiming-uniform) | Match reference engine.py:104-110 exactly. PFR-08 sensitivity to init scale per CONCERNS. | ✓ |
| Cross-module: Xavier-uniform | Apply init.xavier_uniform_; matches BPRMF/BasicMF. Likely shifts PFR-08 outside ±2. | |
| Both: Xavier in benchmark mode, Kaiming in paper_compat | Mode-conditional init; defeats Q1 (paper-compat only ships). | |

**User's choice:** Paper-faithful: PyTorch default (Kaiming-uniform)
**Notes:** None.

### Q17 — Persisted tensor shape (Tensor shape)

| Option | Description | Selected |
|--------|-------------|----------|
| Native PyTorch shape (1, latent_dim) | Persist `affine_output.weight` as nn.Linear creates it. Minimal diff. | ✓ |
| Collapse to (latent_dim,) per-row | Refactor to nn.Parameter(shape=(latent_dim,)); broader model surgery. | |
| Save as (latent_dim,), load as (1, latent_dim) | Reshape on serialize/load. Saves 0 bytes; unnecessary. | |

**User's choice:** Native PyTorch shape (1, latent_dim)
**Notes:** None.

### Q18 — `set_local_parameters` strict policy (Strict load)

| Option | Description | Selected |
|--------|-------------|----------|
| strict=True (hard-fail with RuntimeError) | Per-field delta + 'rm -rf' hint. Phase 3 D-05 idiom. PFR-03 mandates. | ✓ |
| strict=False with partial-load (current PFedRec) | Allows runtime to limp along. Silent contamination per CONCERNS. PFR-03 forbids. | |
| strict=True at signature + strict=False at tensor | Manifest catches most cases; tensor-layer slips through. Defeats PFR-03 'shape or schema' coverage. | |

**User's choice:** strict=True (hard-fail with RuntimeError)
**Notes:** None.

### Q19 — Cold-round client behavior (Cold load)

| Option | Description | Selected |
|--------|-------------|----------|
| Probe-then-load: skip load entirely if probe fails | `if cache_path.exists(): load_local_user_params(); else: cold_round=True`. | ✓ |
| Unconditional-load with try/except | Always attempt; catch FileNotFoundError. Masks real corruption as 'cold rounds'. | |
| Probe + load with fail-open | Probe; fall back to cold-round init on RuntimeError. Defeats PFR-03 hard-fail. | |

**User's choice:** Probe-then-load: skip load entirely if probe fails
**Notes:** None.

### Q20 — `clean_cache.py` schema_v3 handling (Cleanup)

| Option | Description | Selected |
|--------|-------------|----------|
| Unchanged — glob-based mtime sort handles all schema versions | Phase 3 script globs and sorts by mtime; doesn't read manifest contents. Works for v1/v2/v3. | ✓ |
| Schema-aware: refuse to clean unknown schema versions | Read manifest.json, only clean schemas in {1,2,3}. Defensive but adds complexity. | |
| Add `--module pfedrec` filter flag | Clean only PFedRec caches. Marginal benefit; sig_* dirs already aren't touched. | |

**User's choice:** Unchanged — glob-based mtime sort handles all schema versions
**Notes:** None.

---

## Aggregation weight policy

### Q21 — Param aggregation weight policy (Weight pol)

| Option | Description | Selected |
|--------|-------------|----------|
| uniform (align-to-reference) | Match engine.py:81 mean-over-N-clients. Required for PFR-08 ±2 reproduction. | ✓ |
| num_positives (cross-module convention) | Each client weighted by its number of positive interactions. Diverges from reference. | |
| num_training_examples | Weight = num_positives * (1 + num_train_negatives). Strongest weighting toward dense users. | |

**User's choice:** uniform (align-to-reference)
**Notes:** None.

### Q22 — Where to apply weight_policy=uniform (Override loc)

| Option | Description | Selected |
|--------|-------------|----------|
| Update registered _PAPER_COMPAT_PFEDREC profile in mode.py | Change weight_policy='num_positives' to 'uniform'; remove 'Deferred to PFR-02' comment. Single source of truth. | ✓ |
| Apply via module_overrides at PFedRec call site | Keep registered profile as num_positives; PFedRec call passes module_overrides. Misleading registered profile. | |
| Both: update profile AND set explicit module_overrides | Belt-and-suspenders. Slight redundancy. | |

**User's choice:** Update registered _PAPER_COMPAT_PFEDREC profile in mode.py
**Notes:** None.

### Q23 — Eval-metric aggregation policy (Eval ratio)

| Option | Description | Selected |
|--------|-------------|----------|
| Carry-forward Phase 2 BSL-06 sufficient-stat ratio (uniform-equivalent) | Server sums sufficient stats; final ratio = sum_hit / sum_users. In cross-device, mathematically uniform per-user. | ✓ |
| Explicitly weight by num_positives in eval | Diverges from BSL-06; cross-module asymmetry; doesn't match reference. | |
| You decide | Claude's discretion — sufficient-stat already gives uniform per-user. | |

**User's choice:** Carry-forward Phase 2 BSL-06 sufficient-stat ratio (uniform-equivalent)
**Notes:** None.

### Q24 — `weight-policy` CLI override visibility (Override viz)

| Option | Description | Selected |
|--------|-------------|----------|
| Standard D-10 — allow + visible warning + manifest capture | Phase 1 D-10 contract uniform across all modes. | ✓ |
| Refuse weight_policy overrides under paper_compat_pfedrec | Tighter PFR-08 guardrail; breaks D-10 cross-module pattern. | |
| Allow but mark as 'reproduction_attempt=false' | Same as Mode-Lock decision in Area 2 (already declined). | |

**User's choice:** Standard D-10 — allow + visible warning + manifest capture
**Notes:** None.

---

## Claude's Discretion

User did not explicitly say "you decide" on any question — every question received a
positive selection. Implicit Claude-discretion areas (per CONTEXT.md):

- Exact code partition between plans (Wave-1 disjoint-file ownership pattern)
- Exact name of refactored model class
- `[PFR-08 VERIFIED]` log-line formatting and hook position in server_app.py
- Choice between the two reference runs in `IJCAI-23-PFedRec/sh_result/ml-1m.txt`
  as the canonical PFR-08 target (HR=0.7286 vs 0.7315; NDCG=0.4407 vs 0.4453)
- Exact test count + parametrization in module-internal pytest

## Deferred Ideas

(Captured in CONTEXT.md `<deferred>` section. Highlights:)

- Per-user-group (sparse/medium/dense) reporting → Phase 6 EVL-02
- Multi-seed reproduction → Phase 7 THS-02
- PFedRec sensitivity ablations (dim sweep, lr sweep) → Phase 7
- Held-out validation split → v2 (CONCERNS bug #2 acknowledged but accepted)
- DP / privacy → v2 DP-01..02
- Shared `fedrec_common/` extraction → v2 REF-01..02
- ML-10M / ML-20M generalization → v2 EXT-01
- Cross-silo PFedRec results — frozen via D-09 NotImplementedError; pre-Phase-5
  commits are the authoritative artifact

## Reviewed Todos (not folded)

- `phase2-baseline-determinism-path-bug.md` (score 0.6, keyword-matched but
  explicitly Phase 2 baseline scope) — belongs to a future
  `/gsd:plan-phase 2 --gaps` if the slow gate is re-enabled in CI; not a PFedRec
  issue.
