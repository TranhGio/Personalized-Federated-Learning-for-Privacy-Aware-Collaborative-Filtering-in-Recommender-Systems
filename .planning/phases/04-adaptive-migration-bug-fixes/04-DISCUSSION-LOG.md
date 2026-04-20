# Phase 4: Adaptive Migration & Bug Fixes - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-20
**Phase:** 04-adaptive-migration-bug-fixes
**Areas discussed:** ADP-02 cache layout + enable-before-load ordering, ADP-03 server prototype EMA best-round restore, Benchmark-mode thesis defaults, Cold-start blend behavior

---

## ADP-02: Cache Layout + Enable-Before-Load Ordering

### Q1: Cache layout

| Option | Description | Selected |
|--------|-------------|----------|
| Atomic single-file | One `partition_{pid}.pt` per client containing ALL local keys (local_user_row, local_user_bias, personal_mlp.*, fusion_gate/fusion_layer.*, logit_alpha.weight, item_perturbation.weight). torch.save(state_dict, weights_only=True). Matches Phase 3. | ✓ |
| Sharded per-component | Per-partition directory with user.pt, mlp.pt, fusion.pt, alpha.pt, perturbation.pt. Selective reload but breaks atomicity. | |
| You decide | Claude picks based on simplicity. | |

**User's choice:** Atomic single-file (Recommended)
**Notes:** Locks the "single atomic tempfile+rename" pattern; mirrors Phase 3's disk-layout philosophy.

### Q2: Schema v2

| Option | Description | Selected |
|--------|-------------|----------|
| Full adaptive fingerprint | Adds alpha_method, fusion_type, mlp_hidden_dims, per_user_alpha_enabled, item_perturbation_enabled, contrastive_lambda. Any semantic-affecting knob is a signature field. | ✓ |
| Minimal (shape-only) | Adds only fields that change tensor shapes. Omits semantic knobs. | |
| You decide | Claude picks. | |

**User's choice:** Full adaptive fingerprint (Recommended)
**Notes:** Prevents silent contamination when swapping alpha_method / fusion_type / contrastive_lambda mid-cache.

### Q3: Enable order (unconditional vs flag-gated)

| Option | Description | Selected |
|--------|-------------|----------|
| Unconditional in benchmark mode | enable_per_user_alpha(True) + enable_item_perturbation(True) called unconditionally in client_app BEFORE load. Run-config flags become ablation-only overrides. | ✓ |
| Flag-gated as today | Keep run-config flags controlling module attachment; ensure order bug fixed but default remains off. | |
| You decide | Claude picks. | |

**User's choice:** Unconditional in benchmark mode (Recommended)
**Notes:** Benchmark mode = thesis config; removes the silent-misconfiguration risk.

### Q4: Phase-3 v1 cache encounter

| Option | Description | Selected |
|--------|-------------|----------|
| Hard-fail with rm -rf hint | Mirror Phase 3 D-05: raise RuntimeError with explicit reset instruction. No auto-migration. | ✓ |
| Treat as cold-start (silent) | Log warning, Xavier-init fresh. Violates D-05 philosophy. | |
| You decide | Claude picks. | |

**User's choice:** Hard-fail with rm -rf hint (Recommended)
**Notes:** Preserves cross-module D-05 consistency.

---

## ADP-03: Server Prototype EMA Best-Round Restore

### Q1: Snapshot location

| Option | Description | Selected |
|--------|-------------|----------|
| Alongside best_arrays in server memory | SplitFedAvg.best_prototype (numpy) snapshotted with best_arrays on current_ndcg > best_metric. Restored together before final eval. | ✓ |
| Separate sidecar file during training | Write best_prototype.npy each time best round updates. Extra I/O, post-hoc diffability. | |
| You decide | Claude picks. | |

**User's choice:** Alongside best_arrays in server memory (Recommended)
**Notes:** Minimum I/O; symmetric with D-27 best_arrays snapshot.

### Q2: Manifest embed

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — serialize as list in _manifest | Embed best_prototype as float[] in D-15 double-write _manifest block. Tiny payload (~4KB at d=128). | ✓ |
| No — keep runtime-only | Shorter artifact but no post-hoc diffability. | |
| You decide | Claude picks. | |

**User's choice:** Yes — serialize as list in _manifest (Recommended)
**Notes:** Satisfies ADP-08 full protocol fingerprint.

### Q3: Final eval prototype

| Option | Description | Selected |
|--------|-------------|----------|
| Restored best-round prototype | Set self._global_prototype = best_prototype BEFORE broadcasting final eval config. Clients see restored value. | ✓ |
| Live last-round prototype | Final eval uses whatever _global_prototype is at round R. Inconsistent with best_arrays. | |
| You decide | Claude picks. | |

**User's choice:** Restored best-round prototype (Recommended)
**Notes:** Reported NDCG genuinely corresponds to restored state.

### Q4: Degenerate case

| Option | Description | Selected |
|--------|-------------|----------|
| Zero vector of shape (d,) | Snapshot np.zeros(d) if no prototype has been set when best_round fires. Log warning. | ✓ |
| Skip prototype restore this time | Restore best_arrays only. Inconsistent state. | |
| You decide | Claude picks. | |

**User's choice:** Zero vector of shape (d,) (Recommended)
**Notes:** Semantically neutral fallback.

---

## Benchmark-Mode Thesis Defaults

### Q1: model-type

| Option | Description | Selected |
|--------|-------------|----------|
| dual | DualPersonalizedBPRMF (Level-1 blend + Level-2 PersonalMLP + fusion). The thesis contribution. | ✓ |
| bpr | Plain BPR-MF with adaptive alpha blend only; no PersonalMLP / fusion. | |
| You decide | Claude picks. | |

**User's choice:** dual (Recommended)
**Notes:** Thesis-headline architecture; bpr/basic remain available via --run-config.

### Q2: alpha-method

| Option | Description | Selected |
|--------|-------------|----------|
| hierarchical_conditional | Two-stage (geometric mean + harmonic mean) + conditional rules. Thesis contribution. | ✓ |
| multi_factor | 0.40·quantity + 0.25·diversity + 0.20·coverage + 0.15·consistency. Known issues. | |
| You decide | Claude picks. | |

**User's choice:** hierarchical_conditional (Recommended)
**Notes:** Thesis-headline alpha method.

### Q3: fusion-type

| Option | Description | Selected |
|--------|-------------|----------|
| concat | Linear([score_cf; score_mlp]). Most expressive; pyproject + CLAUDE.md default. | ✓ |
| gate | sigmoid(gate_logit)·score_cf + (1-·)·score_mlp. Fewer params. | |
| add | Unweighted sum. Cheapest sanity baseline. | |
| You decide | Claude picks. | |

**User's choice:** concat (Recommended)
**Notes:** Most expressive learned mix.

### Q4: contrastive-lambda

| Option | Description | Selected |
|--------|-------------|----------|
| 0.1 (thesis 'on') | InfoNCE on (p_local, p_effective) with λ=0.1 alongside BPR. Thesis config. | ✓ |
| 0.0 (off) | No contrastive loss. Cleaner benchmark but contradicts thesis-config framing. | |
| You decide | Claude picks. | |

**User's choice:** 0.1 (Recommended — thesis 'on')
**Notes:** Completes "benchmark mode = thesis config" framing.

---

## Cold-Start Blend Behavior

### Q1: Cold-blend strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Prototype-only override for cold rounds | p_effective = p_global (α=0) on cache-miss round. Next round resumes normal blend. Benefits sparse users. | ✓ |
| Noisy blend as today | p_effective = α·Xavier_noise + (1-α)·p_global. Hurts first-round sparse-user convergence. | |
| Hybrid: prototype-only if user_stats.n < threshold | Second threshold to sweep. | |
| You decide | Claude picks. | |

**User's choice:** Prototype-only override for cold rounds (Recommended)
**Notes:** Directly reinforces sparse-user NDCG@10 primary thesis claim.

### Q2: Contrastive in cold rounds

| Option | Description | Selected |
|--------|-------------|----------|
| Skip contrastive in cold rounds | Positive pair (noise, prototype) would be a noise anchor. Skip L_contrastive in cold rounds. | ✓ |
| Apply contrastive as today | May actively hurt first-round sparse-user training. | |
| You decide | Claude picks. | |

**User's choice:** Skip contrastive in cold rounds (Recommended)
**Notes:** Eliminates noise-anchor pathology.

### Q3: Cold-start detection

| Option | Description | Selected |
|--------|-------------|----------|
| Cache-absence signal from D-13 | Before load_local_user_embeddings, check if partition_{pid}.pt exists. Same signal drives D-13 counter. | ✓ |
| Separate first-round boolean per partition | New state that doesn't survive restart. D-13 already has the signal. | |
| You decide | Claude picks. | |

**User's choice:** Cache-absence signal from D-13 (Recommended)
**Notes:** Reuses Phase 3's mechanism; zero bookkeeping.

### Q4: Alpha diagnostics

| Option | Description | Selected |
|--------|-------------|----------|
| First-class metrics each round | Emit alpha_clip_hit_rate + alpha_mean/std/quartiles to round logs + W&B. Answers CONCERNS.md clip-floor critique directly from artifact. | ✓ |
| Post-hoc only via alpha_analysis.py | Manual analysis; no run-time surfacing. | |
| You decide | Claude picks. | |

**User's choice:** First-class metrics each round (Recommended)
**Notes:** Thesis artifact is self-contained.

---

## Claude's Discretion

- `prototype-momentum=0.9` (inherit CLAUDE.md default)
- `item-perturbation-reg=0.01` (inherit pyproject default)
- `alpha` floor/ceiling `[0.1, 0.95]` (inherit `HierarchicalConditionalAlphaConfig` default)
- `mlp-hidden-dims="512,256,128"` (inherit pyproject default)
- Cross-silo legacy freeze pattern (mirror Phase 3 D-02: NotImplementedError in dataset.py)
- FedProx proximal scope (architectural — only GLOBAL params; expanded local-param set never penalized)
- Exact code-layout placement of the cold-start branch (Claude picks cleanest)

## Deferred Ideas

- Sweep over `prototype-momentum` — Phase 7 thesis evaluation
- Alpha floor/ceiling calibration — follow-up gap phase if D-16 reveals a problem
- Shared `fedrec_common/` extraction — v2 REF-01
- DP-SGD — v2 DP-01
- ML-10M / ML-20M generalization — v2 EXT-01
- PFedRec reproduction — Phase 5 (PFR-01..09)
