# Phase 4: Adaptive Migration & Bug Fixes - Research

**Researched:** 2026-04-20
**Domain:** Split-learning cross-device migration + per-client stateful personalization (per-user alpha, item perturbation, server EMA prototype) + in-memory `nn.Embedding` component lifecycle fixes
**Confidence:** HIGH

## Summary

Phase 4 is a **surgical extension of the Phase 2/3 cross-device migration template** applied to the
thesis-contribution module, with three load-bearing bug fixes layered on top. The canonical work
has been shipped twice already: Phase 2 locked the mode-resolver + RNG + exclusion + D-15 +
D-27 pattern; Phase 3 locked the split-learning variant with manifest-sidecar cache and cold-start
counter. Phase 4 reuses every one of those mechanisms — same `fedrec_foundation` APIs, same
shapes of `@app.train` / `@app.evaluate` / `@app.main`, same strict-contract wire payloads — and
bumps three specific things: (a) cache schema version from 1 to 2 with six new signature fields,
(b) enables two `nn.Module` components before loading from cache (the actual bug fix), (c) extends
the best-round restore pattern to snapshot the server-side EMA prototype alongside `best_arrays`.

All three bug fixes are re-orderings or state-tracking additions, not new algorithms. The
`DualPersonalizedBPRMF` class already has the ordering-sensitive helpers (`enable_per_user_alpha`,
`enable_item_perturbation`); the `_LOCAL_PARAMS` property already includes the extended keys when
those flags are on. The EMA `_global_prototype` lives on `SplitFedAvg` / `SplitFedProx` and
already updates each round inside `aggregate_fit`; Phase 4 just needs one new snapshot branch
(5-10 lines) to capture it at the same moment `best_arrays` is captured.

**Primary recommendation:** Clone Phase 3 Plans 01-05 verbatim into the adaptive module; apply
four targeted modifications — (1) expand LOCAL_PARAM_KEYS in strategy.py with adaptive keys,
(2) call `enable_*` BEFORE `load_local_user_embeddings` in client_app.py under `benchmark_cross_device`,
(3) add `best_prototype` snapshot branch to the override `aggregate_fit` in the adaptive strategy,
(4) bump manifest schema_version to 2 with six new signature fields. Leave
`DualPersonalizedBPRMF` / `adaptive_alpha.py` / `evaluation/alpha_analysis.py` largely untouched;
their content is correct, only their integration points need refactoring.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**ADP-02 cache layout + enable-before-load ordering:**
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
  OR semantics is a signature field → silent cross-experiment contamination blocked.
- **D-03:** In `mode="benchmark_cross_device"`, `enable_per_user_alpha(True)` AND
  `enable_item_perturbation(True)` are called **unconditionally** in `client_app.py` BEFORE
  `load_local_user_embeddings(...)`, so `_logit_alpha.weight` and `item_perturbation.weight`
  are in `_LOCAL_PARAMS` at load time. Run-config flags (`enable-per-user-alpha=false`, etc.)
  become **ablation-only overrides** — they turn the component OFF for a specific sweep cell
  but do not leave it uninitialized when ON.
- **D-04:** Schema-version mismatch (e.g., loading Phase-3 `schema_version=1` under Phase-4)
  → mirror Phase 3 D-05: raise `RuntimeError` with per-field delta and an explicit
  `"Run: rm -rf .embedding_cache/{run_id}/"` hint. No auto-migration. No silent cold-start.

**ADP-03 server prototype EMA best-round restore:**
- **D-05:** `SplitFedAvg` holds `self.best_prototype` (numpy ndarray, shape=(d,)) alongside
  `self.best_arrays`. Both are snapshotted at the **same moment** — when
  `current_ndcg > best_metric` on the aggregate_evaluate hook. Pure in-memory state; no
  extra per-round I/O.
- **D-06:** The final best-round prototype is embedded in the result JSON as a `float[]` under
  the `_manifest.best_prototype` key (D-15 double-write). Payload ~4KB at `dim=128`.
- **D-07:** For the FINAL centralized evaluation after best-round restore, set
  `self._global_prototype = self.best_prototype` BEFORE broadcasting the last-round
  `train_config_dict` so clients receive the RESTORED prototype, not the last-round one.
- **D-08:** Degenerate case — best round fires before any prototype was aggregated
  (round 0, or every selected client was cold-start with no prototype):
  snapshot `np.zeros(embedding_dim, dtype=np.float32)` as the `best_prototype`; log a warning
  `"Prototype snapshot at best round R=X is zero vector — no prior prototype aggregation yet."`

**Benchmark-mode thesis defaults:**
- **D-09:** `model-type=dual` is the default under `mode="benchmark_cross_device"`.
- **D-10:** `alpha-method=hierarchical_conditional` is the benchmark default.
- **D-11:** `fusion-type=concat` is the benchmark default.
- **D-12:** `contrastive-lambda=0.1` is the benchmark default.

**Cold-start blend behavior:**
- **D-13:** On cache-miss (first round for this partition), override the blend to
  prototype-only for that round: `p_effective = p_global` (effective `α = 0`). Client trains
  local params from a neutral starting point instead of Xavier-noisy `p_local`.
- **D-14:** InfoNCE contrastive loss is **skipped** in cold-start rounds.
  Compute `L = L_BPR + reg·||item_perturbation||²` only.
- **D-15:** Cold-start detection reuses the Phase 3 D-13 signal: before
  `load_local_user_embeddings`, check if `partition_{pid}.pt` exists. If not → cold round.
  Pass `is_cold_round: bool` through to `train()`.
- **D-16:** Per-round alpha diagnostics (`alpha_clip_hit_rate`, `alpha_mean`, `alpha_std`,
  `alpha_p25`, `alpha_p50`, `alpha_p75`) logged to `eval_metrics_history[round_num]` and W&B.

### Claude's Discretion

- `prototype-momentum=0.9` (EMA half-life ~6 rounds at default)
- `item-perturbation-reg=0.01` (L2 strength on `||item_perturbation||²`)
- `alpha` floor/ceiling `[0.1, 0.95]` (from `HierarchicalConditionalAlphaConfig`)
- `mlp-hidden-dims="512,256,128"`
- Cross-silo legacy freeze: follow Phase 3 D-02 pattern — `dataset.py` raises
  `NotImplementedError` on `partition_mode != "natural"`.
- FedProx proximal term scope: proximal penalty applies ONLY to GLOBAL params
  (`item_embeddings.weight`, `item_bias.weight`, `global_bias`). The expanded local-param
  set is NEVER touched by the proximal term.
- Exact code layout of the cold-start branch (in `client_app.py`, `task.py`, or as a helper in
  `models/dual_personalized_bpr_mf.py`) — Claude picks cleanest placement.

### Deferred Ideas (OUT OF SCOPE)

- **Sweep over `prototype-momentum`** — Phase 7 handles sweeps via `sweep.yaml`.
- **Calibration of alpha floor/ceiling `[0.1, 0.95]`** — if D-16 metric reveals a problem at
  Phase-7 eval time, that's a follow-up phase, not Phase 4.
- **Shared `fedrec_common/` extraction** — v2 REF-01.
- **Differential privacy (DP-SGD)** — v2 DP-01.
- **ML-10M / ML-20M generalization** — v2 EXT-01.
- **PFedRec reproduction** — Phase 5 (PFR-01..09).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **ADP-01** | `pyproject.toml` defaults: `num-supernodes=6040`, `partition-mode="natural"` in BOTH `local-simulation` and `local-sim-gpu` federation blocks | Phase 2 Plan 02 + Phase 3 Plan 02 established the exact-duplicate pyproject shape. Foundation contract keys: `mode="benchmark_cross_device"`, `run-seed`, `weight-policy="num_positives"`, `checkpoint-rule="best_round_restore"`, `reuse-cache=false`. Phase 4 adds six NEW keys that drive `schema_version=2`: `alpha-method`, `fusion-type`, `mlp-hidden-dims`, `enable-per-user-alpha=true`, `enable-item-perturbation=true`, `contrastive-lambda=0.1` (thesis defaults per D-09..D-12). Dev dep `pytest>=7.0` added (Phase 3 Plan 02 pattern). |
| **ADP-02** | `enable_per_user_alpha()` + `enable_item_perturbation()` called BEFORE `load_local_user_embeddings()`; cached values restored, not re-initialized from heuristic | Fix is re-ordering in `client_app.py` lines 247-260 (current code creates model → loads cache → enables components overwriting the loaded cache on line 351, 362). CONTEXT D-03 locks: call both `enable_*` unconditionally under benchmark mode BEFORE the load so `_LOCAL_PARAMS` has `_logit_alpha.weight` and `_item_perturbation.weight` at load time. Signature-verified manifest schema_version=2 (D-02) prevents stale Phase-3 caches from silently loading. Existing `DualPersonalizedBPRMF._LOCAL_PARAMS` property at `models/dual_personalized_bpr_mf.py:542-569` already adds these keys to the tuple when the enable flags are set — no model surgery required. |
| **ADP-03** | Server prototype EMA (`p_global`) saved as part of best-round checkpoint and restored at final evaluation time | D-05 locks: mirror D-27 (Phase 2 Plan 04); add `self.best_prototype: Optional[np.ndarray]` field to `AdaptiveSplitFedAvg`/`FedProx`; snapshot alongside `best_arrays` when `current_ndcg > best_metric`. D-07 locks: set `self._global_prototype = self.best_prototype` before the final `train_config_dict["global_prototype"]` broadcast. D-08 locks: degenerate case (no prototype aggregated yet at best round) snapshots `np.zeros(dim, dtype=float32)` + warning. Existing `_aggregate_prototypes` at `strategy.py:138-186` is unchanged; only one new branch is added in `aggregate_fit` AFTER `_aggregate_prototypes` has run this round. |
| **ADP-04** | Benchmark-mode one-user assertion in `client_app.py` | Identical to BSL-02 / PSN-02. Use `fedrec_foundation.mode.assert_benchmark_one_user_per_client(profile, num_users, overrides)` in BOTH `@app.train` and `@app.evaluate` handlers; visible `num-supernodes` override bypass via D-10 Phase 1. |
| **ADP-05** | Training negatives exclude held-out test positive (FND-03) | Identical to BSL-03 / PSN-03. Thread `ExclusionTable.for_user(partition_id)` through `train_dual_personalized` / `train_bpr_mf`; merge into `user_rated_items` (or collapse to flat `Set[int]` as Phase 3 did — the adaptive client is also one user under cross-device). `evaluate_ranking_sampled` at `task.py:918-1074` currently uses `import random; random.seed(seed)` at lines 952-953 and `random.sample` at line 1012 — all three must be stripped and replaced with `np_rng(run_seed, user_idx, round_num, "eval_neg")` per BSL-05. |
| **ADP-06** | Server-side sampling seeded; evaluator RNG fixed; sufficient-stat metrics; run-scoped cache | Server-side sampling: copy Phase 3 Plan 04 pattern verbatim — `_server_sampler = server_rng(run_seed)`, discovery round, partition-id-space sampling. Evaluator RNG: strip `random.seed`/`random.sample` from `task.py:918-1074` as in BSL-05. Sufficient-stat metrics: extend `AdaptiveSplitFedAvg.aggregate_evaluate` to sum `hit_count_*`/`ndcg_sum_*`/`evaluated_users_*` per group, same signature as `PersonalizedSplitFedAvg.aggregate_evaluate`. Run-scoped cache: PSN-05 pattern — `.embedding_cache/{run_id}/manifest.json` + `partition_{pid}.pt`, with schema_version=2 and six new fields (D-02). |
| **ADP-07** | Hierarchical-conditional / multi-factor / data-quantity alpha factory returns values in `[0.1, 0.95]` for edge-case user-stats inputs | `create_alpha_computer` factory at `adaptive_alpha.py` already clips to `[min_alpha, max_alpha]` inside `compute_from_stats` (lines 208, 306, 339, 486). Unit test exercises each branch with crafted inputs: sparse (n=5, 15), niche (n=200, genre_entropy=0.5), inconsistent (n=100, rating_std=1.45), completionist (n=150, n_unique_items=95, genre_entropy=0.5). Verifies clip bounds AND that each conditional rule fires (via `compute_factors` helper at line 488 which returns `applied_rules` list). |
| **ADP-08** | Module logs FND-07 protocol fingerprint | Identical to BSL-08 / PSN-07. `build_run_manifest(..., module="adaptive")` + `embed_manifest_in_result` + `write_manifest_sibling`. Phase 4 EXTENDS the embedded _manifest dict with `best_prototype: List[float]` (D-06) after `embed_manifest_in_result` mutates `results_data["_manifest"]`. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `flwr[simulation]` | ≥1.22.0 | Federated orchestration; `ServerApp`/`ClientApp`/`Grid` API; `FedAvg`/`FedProx` base strategies | Existing codebase-wide dependency; Phase 2+3 wired unchanged |
| `torch` | ≥2.7.1 | `DualPersonalizedBPRMF` is a `torch.nn.Module`; the three enable-* helpers mutate `nn.Embedding` sub-modules | Existing |
| `numpy` | ≥1.24.0 | Server EMA `_global_prototype: np.ndarray`; user-stat computation | Existing |
| `fedrec-foundation` | local editable | Source of `mode.resolve_mode_defaults`, `rng.np_rng/torch_gen/server_rng`, `evaluator.get_primary_evaluator`, `fit_metrics.*Contract`, `manifest.build_run_manifest/embed_manifest_in_result/write_manifest_sibling`, `bundle.verify_bundle`, `atomic.atomic_write_json` | Phase 1 shipped as local-path dep |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pytest` | ≥7.0 (dev) | ADP-07 unit tests; client/server integration tests | Added as `[project.optional-dependencies] dev = [...]` per Phase 2+3 precedent |
| `wandb` | ≥0.19.0 | Per-round alpha-diagnostic logging (D-16) | Existing |
| `pandas` | ≥2.0.0 | Foundation dataset adapter consumes pandas via existing `dataset.py` helpers | Existing |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| In-memory `best_prototype` snapshot | Disk-backed snapshot on every improvement | Disk-backed gives crash-recovery but adds N_round I/O; CONTEXT D-05 locks in-memory. Phase 2 D-27 is in-memory for the same reason — snapshot is a cheap numpy copy. |
| Single `.pt` file with all LOCAL keys (D-01) | Multi-file per-component cache | Multi-file eases partial read but explodes the inode count at N=6040. Single-file plus manifest sidecar keeps Phase 3 atomicity pattern. |
| New `AdaptiveSplitFedAvg` subclass | Extend existing `SplitFedAvg` in-place | Phase 3 established module-owned strategy subclasses; a new class parallels `PersonalizedSplitFedAvg` and avoids cross-module import surfaces. |

**Installation:**
```bash
# already installed per Phase 1 + Phase 2 + Phase 3
pip install -e scripts/foundation/
pip install -e "federated-adaptive-personalized-cf[dev]"
```

**Version verification:** All stack entries are existing dependencies of Phase 2+3 modules; no
version bumps for Phase 4. Check `scripts/foundation/fedrec_foundation/` for exported symbols
before writing plans; `grep -r "from fedrec_foundation"` across Phase 2+3 shipped code is the
authoritative list.

## Architecture Patterns

### Recommended Project Structure

```
federated-adaptive-personalized-cf/
├── pyproject.toml                     # [ADP-01] num-supernodes=6040, partition-mode="natural",
│                                      #          schema_version=2 driver keys, [dev] pytest extra
├── federated_adaptive_personalized_cf/
│   ├── dataset.py                     # [ADP-06] rip-and-replace foundation adapter (D-17);
│   │                                  #          NotImplementedError on partition_mode!="natural" (D-02 mirror)
│   ├── strategy.py                    # [ADP-03, ADP-06] AdaptiveSplitFedAvg/FedProx
│   │                                  #          with expanded LOCAL_PARAM_KEYS, best_prototype snapshot,
│   │                                  #          sufficient-stat aggregate_evaluate, aggregate_fit override
│   ├── client_app.py                  # [ADP-02, ADP-04, ADP-05, ADP-06]
│   │                                  #          mode resolve → enable_per_user_alpha + enable_item_perturbation
│   │                                  #          BEFORE load → signature v2 cache load → cold-start detection →
│   │                                  #          FND-03 exclusion → FND-06 RNG → save → strict-contract payload
│   ├── task.py                        # [ADP-05, ADP-06] FND-06 np_rng/torch_gen,
│   │                                  #          ExclusionTable threading, strip random.seed/random.sample,
│   │                                  #          is_cold_round kwarg → α=0 + skip contrastive
│   ├── server_app.py                  # [ADP-06, ADP-08] mode resolver, discovery round,
│   │                                  #          partition-id sampling, AdaptiveSplitFedAvg wire-up,
│   │                                  #          D-27 best-round restore for arrays AND prototype (D-07),
│   │                                  #          D-15 double-write with module="adaptive",
│   │                                  #          best_prototype embedded (D-06), D-16 alpha diagnostics,
│   │                                  #          cold-start counter (D-13/D-15 reuse Phase-3 pattern)
│   ├── models/
│   │   ├── dual_personalized_bpr_mf.py   # UNTOUCHED (or minor tweak for is_cold_round kwarg
│   │   │                                 # — see Pattern 3)
│   │   ├── adaptive_alpha.py             # UNTOUCHED
│   │   ├── bpr_mf.py / basic_mf.py       # Optionally collapsed to single-row (PSN-06 mirror) —
│   │   │                                 # adaptive module's primary thesis model is dual, so
│   │   │                                 # single-row refactor of these fallbacks is OPTIONAL
│   │   │                                 # (Claude's discretion; out-of-scope if time-boxed)
│   │   └── losses.py                     # UNTOUCHED (InfoNCE has batch<=1 guard at line 173)
│   └── evaluation/
│       └── alpha_analysis.py             # Refactor ONE-LINER: expose AlphaAnalyzer.compute_scalar_summary
│                                          # so aggregate_evaluate can call it per-round for D-16
└── tests/
    ├── __init__.py
    ├── conftest.py                    # fake_evaluate_res + fake_client_proxy fixtures (copy from Phase 3)
    ├── test_strategy.py               # AdaptiveSplitFedAvg sufficient-stat aggregator + aggregate_fit
    │                                  # inherited unchanged (aggregate_fit super().aggregate_fit
    │                                  # still runs weighted average of GLOBAL params) + best_prototype
    │                                  # snapshot at best round + D-07 broadcast swap
    ├── test_dual_model.py             # enable_per_user_alpha + enable_item_perturbation before
    │                                  # load: _LOCAL_PARAMS membership + set_local_parameters roundtrip
    ├── test_task_rng.py               # BSL-05-style RNG strip + FND-03 exclusion + cold-round α=0
    ├── test_client_assertion.py       # ADP-04 one-user assert + FitMetricsContract+partition_id shape
    ├── test_embedding_cache_manifest_v2.py   # schema_version=2 + 12 fields + hard-fail load on delta
    ├── test_alpha_factory.py          # ADP-07: edge-case inputs at [0.1, 0.95] boundaries + each rule branch
    └── test_server_integration.py     # PSN-04-style seeded sampling + D-15 + D-13 cold-start +
                                       # NEW: best_prototype snapshot + D-07 restored broadcast
```

### Pattern 1: Enable-Before-Load Ordering (ADP-02)

**What:** Extend `DualPersonalizedBPRMF`'s `_LOCAL_PARAMS` keyset BEFORE `load_local_user_embeddings`
reads the cache manifest, so `_logit_alpha.weight` and `_item_perturbation.weight` are matched
in the saved state dict and restored, rather than created fresh after-the-fact.

**When to use:** Only in `benchmark_cross_device` mode (D-03). In ablation mode
(`enable-per-user-alpha=false` override), skip the call — but that branch is expected to never
fire in a thesis run.

**Example:**
```python
# Source: CONTEXT D-03 + existing client_app.py flow (lines 247-363)
# Location: federated_adaptive_personalized_cf/client_app.py @app.train() and @app.evaluate()

# Step 1: construct bare model (Xavier init)
model = get_model(
    model_type=model_type,  # "dual" under benchmark_cross_device (D-09)
    embedding_dim=embedding_dim,
    dropout=dropout,
    mlp_hidden_dims=mlp_hidden_dims,  # "512,256,128" per D-11
    fusion_type=fusion_type,          # "concat" per D-11
)

# Step 2: load GLOBAL params from server message
model.set_global_parameters(msg.content["arrays"].to_torch_state_dict())

# Step 3 (FIX): enable_* BEFORE load_local_user_embeddings. In benchmark_cross_device mode,
# both flags are unconditionally True (D-03); run-config overrides are ablation-only.
enable_per_user_alpha = bool(context.run_config.get("enable-per-user-alpha",
                                                    profile.enable_per_user_alpha))
enable_item_perturbation = bool(context.run_config.get("enable-item-perturbation",
                                                       profile.enable_item_perturbation))

# Compute heuristic alphas ONCE at enable-time; refinement happens via BPR grads thereafter.
if enable_per_user_alpha:
    per_user_alphas = compute_per_user_alpha(user_stats, alpha_config, hc_config)
    model.enable_per_user_alpha(num_users=model.num_users, init_alphas=per_user_alphas)
    # ^ creates nn.Embedding _logit_alpha; model._LOCAL_PARAMS now includes '_logit_alpha.weight'

if enable_item_perturbation:
    reg = float(context.run_config.get("item-perturbation-reg", 0.01))
    model.enable_item_perturbation(reg_lambda=reg)
    # ^ creates nn.Embedding _item_perturbation; model._LOCAL_PARAMS now includes
    #   '_item_perturbation.weight'

# Step 4: NOW load — cache manifest signature includes per_user_alpha_enabled +
# item_perturbation_enabled flags (D-02), so a mismatch with the on-disk cache hard-fails
# loud with a per-field delta (D-04).
is_cold_round = not _cache_partition_path(partition_id, run_id, reuse_cache).exists()
if not is_cold_round:
    loaded_keys, missing_keys = model.set_local_parameters(cached_state_dict, strict=False)
    # loaded_keys NOW includes '_logit_alpha.weight' + '_item_perturbation.weight' because
    # the keys were added to _LOCAL_PARAMS in Step 3. The bug is fixed.
```

### Pattern 2: Best-Round Prototype Snapshot (ADP-03)

**What:** Extend the Phase 2 D-27 in-memory best-round pattern to symmetrically track both
the global ArrayRecord AND the EMA prototype. Both are snapshotted at the same moment —
when the current round's federated NDCG@10 exceeds the best-so-far.

**When to use:** In the adaptive strategy's `aggregate_fit` OR in the server_app per-round
loop. Either location works because Flower runs `aggregate_fit` deterministically before
`aggregate_evaluate` within each round. The cleanest placement is on the strategy object
(it already owns `_global_prototype`).

**Lifecycle verification (Flower):** Per-round order is `aggregate_fit` → per-client evaluate
messages are dispatched → `aggregate_evaluate`. `_global_prototype` is updated exactly once
per round inside `aggregate_fit` via `_aggregate_prototypes`. The `aggregate_evaluate` hook
observes the freshly-updated prototype. The "best round" check runs AFTER aggregate_evaluate
produces thesis_metrics — at that point, `self._global_prototype` already reflects this
round's post-aggregation state. This ordering is safe; no additional synchronization needed.

**Example:**
```python
# Source: CONTEXT D-05, D-07, D-08 + Phase 2 Plan 04 D-27 pattern
# Location: federated_adaptive_personalized_cf/strategy.py (AdaptiveSplitFedAvg)
#           federated_adaptive_personalized_cf/server_app.py @app.main() loop

# In AdaptiveSplitFedAvg.__init__:
self.best_prototype: Optional[np.ndarray] = None  # D-05 mirror

# In server_app.py per-round loop, AFTER strategy.aggregate_evaluate produces thesis_metrics:
if checkpoint_rule in ("best_round_restore", "best_round") and thesis_metrics:
    current_ndcg = float(thesis_metrics.get("sampled_ndcg@10", 0.0))
    if round_num == 1 or current_ndcg > best_metric:
        best_metric = current_ndcg
        best_round_num = round_num
        best_arrays = ArrayRecord({
            k: v.detach().clone() for k, v in arrays.to_torch_state_dict().items()
        })
        # D-05: snapshot prototype at the same moment
        current_proto = strategy.get_global_prototype()
        if current_proto is not None:
            strategy.best_prototype = current_proto.copy()  # np.ndarray copy
        else:
            # D-08 degenerate case: best round fires before prototype aggregated
            strategy.best_prototype = np.zeros(embedding_dim, dtype=np.float32)
            log(WARNING, f"Prototype snapshot at best round R={round_num} is zero vector "
                         f"— no prior prototype aggregation yet.")
        print(f"  [CHECKPOINT] New best sampled_ndcg@10={best_metric:.4f} at round {best_round_num}")

# After the FL loop completes, BEFORE the final centralized-eval broadcast (D-07):
if checkpoint_rule in ("best_round_restore", "best_round") and best_round_num > 0:
    arrays = best_arrays
    if strategy.best_prototype is not None:
        strategy._global_prototype = strategy.best_prototype  # D-07 swap
    # Now the last-broadcast train_config will carry the RESTORED prototype to clients.

# D-06: embed best_prototype in the result JSON's _manifest dict
results_data = embed_manifest_in_result(manifest, results_data)
if strategy.best_prototype is not None:
    results_data["_manifest"]["best_prototype"] = strategy.best_prototype.tolist()
else:
    results_data["_manifest"]["best_prototype"] = None
```

### Pattern 3: Cold-Start Blend Override (D-13 / D-14 / D-15)

**What:** On a client's first round (cache miss), force `p_effective = p_global` (effective α=0)
and skip InfoNCE contrastive — because `p_local` is still Xavier-random noise and would either
corrupt the contrastive anchor or add noise to the blend.

**When to use:** Inside `@app.train()` when `is_cold_round=True`. The detection is cheap:
probe `Path(cache_dir) / f"partition_{pid}.pt"` existence.

**Cold-start signal:** The Phase 3 D-13 pattern checks the cache file existence BEFORE
load_local_user_embeddings. That same check also drives the cold-start counter on the server
side. In Phase 4 the check is identical; the client just threads `is_cold_round` into
the training loop so the blend + contrastive behavior changes.

**Placement choice:** Two options —
  (a) `task.py::train_dual_personalized(..., is_cold_round=False)` kwarg branch at the
      top: `if is_cold_round: model.set_alpha(0.0); contrastive_lambda_eff = 0.0` then reset
      `set_alpha(client_alpha)` after training completes.
  (b) Put `cold_round=bool` as a context-dict field passed via `model.set_cold_round(True)` and
      have `get_effective_embedding` check it before applying alpha.

Option (a) is simpler and matches Phase 3's "thread kwarg through the handler" idiom. Option
(b) requires mutating the model class. **Recommendation: Option (a)**.

`compute_user_prototype` design choice: on cold rounds, `p_local` is Xavier noise post-one-
training-pass. Should that still contribute to the server EMA? The mean of Xavier-init +
one-pass BPR gradient is not pathological, but it's less informative than a mature p_local.
Since `_aggregate_prototypes` already weights by `num_examples` and clients naturally contribute
proportional to their data, no special cold-round handling is needed — existing code is correct.
**Recommendation: leave `compute_user_prototype` unchanged.** Cold-round p_local-derived
prototype is noisy but weighted by num_examples, which is usually small for sparse users, so
EMA absorbs the noise.

### Pattern 4: Schema-Version=2 Manifest (D-02 / D-04)

**What:** Phase 3 shipped a 6-field signature (`run_id`, `method`, `num_users`, `num_items`,
`dim`, `split_hash`) at `schema_version=1`. Phase 4 bumps to `schema_version=2` and adds six
adaptive-specific signature fields: `alpha_method`, `fusion_type`, `mlp_hidden_dims` (joined
string), `per_user_alpha_enabled`, `item_perturbation_enabled`, `contrastive_lambda`.

**Extension pattern:** The existing `_signature_fields` helper in Phase 3's
`federated-personalized-cf/federated_personalized_cf/client_app.py` is a pure dict-builder
that consumes keyword-only args and returns the dict written into `manifest.json` via
`atomic_write_json`. Phase 4 near-clones this helper and adds the six new fields. The
shape-guard in `_save_local_user_state` / `_load_local_user_state` also extends to check
the full set of LOCAL keys (base 2 + MLP sublayers + fusion + optional logit_alpha +
optional item_perturbation). Per-field mismatch raises `RuntimeError` with per-field delta
+ literal `rm -rf .embedding_cache/{run_id}/` hint (D-04 mirror of Phase 3 D-05).

**Cache load failure mode:** Swapping `alpha-method=hierarchical_conditional` →
`alpha-method=multi_factor` mid-cache hard-fails because `alpha_method` is a signature
field. The user gets a clear error + the rm hint — not a silently-wrong-semantics run.

### Pattern 5: Per-Round Alpha Diagnostics (D-16)

**What:** Surface alpha distribution statistics (mean, std, quartiles, clip-hit-rate) as
first-class fields on `eval_metrics_history[round_num]` + W&B logs, so the CONCERNS.md clip-
floor critique can be answered directly from the run JSON without post-hoc analysis.

**Implementation seam:** `evaluation/alpha_analysis.py::AlphaAnalyzer` already computes these
via `AlphaStatistics`. Refactor ONE method — expose a `compute_scalar_summary(per_user_alphas) →
Dict[str, float]` callable that `aggregate_evaluate` invokes per-round. The dict contains:
`alpha_clip_hit_rate` (fraction at floor OR ceiling), `alpha_mean`, `alpha_std`, `alpha_p25`,
`alpha_p50`, `alpha_p75`.

**Wire path:** Each client includes `per_user_alpha_*` stats in `FitMetricsContract` metrics
(client_app.py already does this at lines 420-426). Server aggregates by num_positives-
weighted mean and emits the D-16 dict into `eval_metrics_history`. Alternatively, server
does a post-hoc single-shot summary from the broadcast train_config (uses last-round
_global_prototype and forward model). Either works; client-side emission is cheaper.

### Anti-Patterns to Avoid

- **Disk-backed best_prototype snapshot on every improvement:** Adds N_rounds of I/O for
  near-zero benefit. In-memory snapshot is ~4KB at dim=128 and survives the process lifetime.
- **Touching `models/dual_personalized_bpr_mf.py` for the ADP-02 fix:** The fix is re-ordering
  in `client_app.py`, not a model change. The model's `_LOCAL_PARAMS` property already reacts
  correctly to the enable flags — it just needed to be called in the right order.
- **Porting D-24 gradient isolation:** The adaptive module uses the `num_users × d` ghost
  table (`nn.Embedding(num_users, embedding_dim)` at `dual_personalized_bpr_mf.py:112`),
  unlike Phase 3 which collapsed to single-row. So D-24 snapshot-restore IS still needed
  here. **But only for `user_embeddings.weight` and `user_bias.weight`** — NOT for
  `_logit_alpha.weight` or `_item_perturbation.weight` because those are full-table-updated
  per user by design (per-user alpha IS indexed by user id; perturbation is indexed by item
  id and is a full N_items × d update). Confirm during planning by reading `task.py` training
  loop: whether the dual-model single-user batch training leaks cross-row updates on user_
  embeddings depends on whether the client has >1 user in it. Under benchmark_cross_device
  (1 user per client) the single-row pattern (PSN-06 mirror) would eliminate D-24 need, but
  collapsing `DualPersonalizedBPRMF`'s embeddings to single-row is substantial surgery. The
  conservative choice: keep D-24 snapshot/restore on user_embeddings + user_bias; accept the
  ghost table; document in plan that PSN-06 collapse for adaptive is Phase 4.5 follow-up.
- **Treating the cold-start override as an ablation feature:** D-13/D-14 is load-bearing for
  the sparse-user thesis claim (THS-04). Every partition's first round is cold; sparse users
  with <20 interactions might have MOST of their rounds be cold because they're not sampled
  often. The cold-start blend override is what prevents Xavier noise from dominating their
  first-round prediction — skipping it is silently dropping the thesis claim on the weakest
  user group.
- **Making the enable-before-load re-ordering conditional on the `enable-*` run-config flags:**
  CONTEXT D-03 is explicit — in benchmark mode, always True. Keeping a conditional branch
  opens the door to forgetting the fix, which is the entire Phase 4 hazard. Flags become
  ablation overrides only.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Seeded client sampling per round | Custom `random.Random(seed + round_num).sample(...)` | `server_rng(run_seed).sample(...)` (`fedrec_foundation.rng`) | Phase 1 FND-06 shipped the sha256-namespaced per-purpose factory; re-inventing gives inconsistent seeding across modules |
| Per-user FND-06 RNG | Fresh `np.random.default_rng(seed)` inline | `np_rng(run_seed, user_idx, round_num, "eval_neg"/"train_neg")` | Keeps cross-module determinism with the same tuple contract Phase 2+3 use |
| LOO exclusion set lookup | Re-build `train_positives ∪ {test_item}` in every call site | `ExclusionTable.for_user(partition_id)` | Phase 1 FND-03 owns this with a flat CSR disk layout; avoids per-module drift |
| Cache manifest atomic write | `open(path, 'w') + json.dump()` | `fedrec_foundation.atomic.atomic_write_json(path, dict)` | Atomic via tempfile + os.replace; matches Phase 3's signature-write pattern |
| Run manifest assembly | Hand-roll a dict with fingerprints | `build_run_manifest(run_id, mode_profile, verify_bundle_result, overrides, module="adaptive")` | One-line call populates all 6 IMP-2 fingerprints + schema + git commit |
| Protocol-fingerprint double-write | Single JSON write | `embed_manifest_in_result(...) + write_manifest_sibling(...)` | D-15 belt-and-suspenders; survives partial-failure runs |
| Sufficient-stat aggregation | Per-client-ratio weighted mean | `strategy.aggregate_evaluate` returning `sum(hit_count)/sum(evaluated_users)` per group | Phase 2 Plan 01 + Phase 3 Plan 01 shipped this pattern; double-aggregation bug is specifically what the strict contract prevents |
| Strict-contract wire payload | Loose `dict` metric payload | `FitMetricsContract(...)` + `EvaluateMetricsContract(...)` + `validate_*_metrics(dict)` | Phase 1 Plan 03 owns the contract; defense-in-depth validate rejects free-form extras |
| Alpha factory | Hand-roll hierarchical conditional formula | `create_alpha_computer(config, hc_config)` | Existing at `adaptive_alpha.py`; ADP-07 extends the test suite, not the implementation |
| Cold-start cache probe | Manual JSON manifest parsing | Existing `_load_local_user_state` returning `None` on miss (Phase 3 pattern) | Same signature mismatch path, same atomic write discipline |
| Atomic single-file torch.save | DIY tempfile + torch.save | `tempfile.mkstemp(prefix="partition_tmp_", suffix=".pt") + torch.save + os.replace` | Phase 3 Plan 03 discovered PyTorch rejects dot-prefixed tempfile names; use `partition_tmp_` prefix (not `.partition_`) |

**Key insight:** Phase 4 is "integration surface work" — every algorithm that matters (alpha
factory, EMA, dual-level fusion, contrastive loss, item perturbation) is already written and
correct. The bugs are at the integration boundaries (ordering, state persistence, RNG
determinism, cache contamination). Don't hand-roll; thread foundation APIs through the right
sequence of calls.

## Runtime State Inventory

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| **Stored data** | `.embedding_cache/` directories across all 4 modules (federated-baseline-cf, federated-personalized-cf, federated-adaptive-personalized-cf, federated-pfedrec). Adaptive module's cache at `federated-adaptive-personalized-cf/.embedding_cache/partition_{id}/user_embeddings.pt`. Contains old-schema (pre-Phase-3) state dicts with ghost-table shapes. W&B run artifacts at `federated-adaptive-personalized-cf/wandb/` (8.9 MB) from cross-silo legacy runs. | **Code edit:** Phase 4 Plan 03 (or equivalent) invokes `scripts/clean_cache.py` documentation pattern — user must `rm -rf federated-adaptive-personalized-cf/.embedding_cache/` BEFORE first Phase-4 run (documented in plan). Or: schema_version=2 hard-fails on schema_version=1 load (D-04) so the error message tells the user what to do. **No data migration needed** — caches are per-experiment and regenerated. |
| **Live service config** | W&B project `federated-adaptive-personalized-cf` (cross-silo legacy) exists remotely. Existing run dashboards on that project mix cross-silo and (upcoming) cross-device runs if we don't separate. | **Code edit:** Set default W&B project to `federated-cf-cross-device` for `mode="benchmark_cross_device"` (mirror Phase 3 Plan 04). Legacy project name stays for `mode="cross_silo_legacy"` — but that branch is frozen by Phase 4 D-17-style NotImplementedError. |
| **OS-registered state** | None — nothing in Task Scheduler, systemd, launchd, pm2, or similar OS-level registries references the adaptive module. Federated simulation is a one-shot `flwr run .` invocation. | **None — verified by** checking for any `scripts/run_*.sh` that schedule cron or daemonize; only ad-hoc sweep scripts exist. |
| **Secrets / env vars** | `WANDB_API_KEY` env var or `~/.netrc` from `wandb login` (user-managed outside repo). `PYTHONHASHSEED` recommended for full determinism but not required (FND-06 is already hash-stable). | **None — no rename.** W&B auth unchanged; Phase 4 doesn't touch auth surface. |
| **Build artifacts / installed packages** | `federated-adaptive-personalized-cf.egg-info/` (pip install -e output; may carry stale entry points if pyproject.toml changes). `scripts/foundation/fedrec_foundation.egg-info/` (editable install metadata). `__pycache__/` across all federated modules. | **Code edit:** Phase 4 Plan 01 (pyproject.toml edit) may require a `pip install -e .` re-run to refresh egg-info. Standard practice; plan should document. No disk-level rename needed. |

**Nothing found in category:** Explicitly verified for OS-registered state and secrets/env-vars;
the rename/refactor surface is entirely in-repo with a per-run cache regime that self-invalidates
via schema_version bump (D-04).

## Common Pitfalls

### Pitfall 1: `enable_*` called AFTER `load_local_user_embeddings` silently re-initializes state
**What goes wrong:** `_logit_alpha` and `_item_perturbation` are `nn.Embedding` sub-modules that
exist only when the `enable_*` method has run. `_LOCAL_PARAMS` is a dynamic property that only
includes those keys when the flags are set. If `load_local_user_embeddings` runs first, the model
has no sub-module to copy the saved weights into; the load silently matches only the base 2 keys
(`user_embeddings.weight`, `user_bias.weight`) and the adaptive-specific cached tensors are
ignored. Then `enable_per_user_alpha(init_alphas=...)` creates a fresh `nn.Embedding` and
re-initializes from the heuristic. Every round the model "learns" from scratch in alpha space.
**Why it happens:** The enable-method pattern was designed for one-time-at-startup
initialization, not round-by-round re-entry with a persistence layer behind it. The method
mutates the model in place but does not check if a saved cache exists.
**How to avoid:** Reorder unconditionally under benchmark mode (D-03). In ablation mode the
load still has a choice — if the flag is off, no sub-module is created, no cached tensor is
loaded, no bug.
**Warning signs:** Per-user alpha std stays near zero across rounds (the model keeps reverting
to the near-uniform heuristic init); item perturbation norm stays at zero (ditto).

### Pitfall 2: Server prototype EMA state mismatched with restored arrays
**What goes wrong:** Server restores `best_arrays` at end-of-training but `self._global_prototype`
is the LAST round's prototype, not the best round's. Final centralized evaluation uses the
best item embeddings but the latest (possibly drifted) user prototype. The reported
`sampled_ndcg@10` technically doesn't correspond to any run state that existed during training
— it's a Frankenstein of best global items + latest prototype.
**Why it happens:** Best-round restore pattern (Phase 2 D-27) was built for an all-global
module where there was no extra server state to snapshot. Split learning with EMA prototype
has TWO pieces of server state; both must be snapshot together.
**How to avoid:** Symmetrize the snapshot (D-05). Snapshot `best_prototype` in the same branch
as `best_arrays`. At final restore, swap both.
**Warning signs:** `last_sampled_ndcg@10` and `best_sampled_ndcg@10` differ by more than random
fluctuation; post-final centralized eval metric is off by more than ~0.005 from what the FL-
aggregated best-round value reported.

### Pitfall 3: Training negatives include held-out test positive (FND-03 leak)
**What goes wrong:** `train_bpr_mf` builds `user_rated_items[user]` from the trainloader only
(`task.py:407-414`). The held-out test positive is excluded from trainloader by LOO construction,
so it's not in `user_rated_items`; the random negative sampler can select it. The model then
trains on "push the test item's score DOWN", which is literally the opposite of the thing the
test metric measures.
**Why it happens:** Same pattern as BSL-03 / PSN-03 — fixed in baseline + personalized, not
yet in adaptive.
**How to avoid:** Fold `ExclusionTable.for_user(partition_id)` into `user_rated_items` before
the sampling loop runs. Exclusion table already contains `train_positives ∪ {test_item}`.
**Warning signs:** Early-round NDCG@10 artificially LOW (the model is trained to down-rank the
test item); slow convergence; high per-round NDCG variance.

### Pitfall 4: Global `random.seed(seed)` in `evaluate_ranking_sampled`
**What goes wrong:** `task.py:952-953` calls `import random; random.seed(seed)` with
default `seed=42`. Every round, same 99 negatives per user. Evaluation variance is artificially
low; if the 99 negatives include an atypically easy or hard item for the user, the bias is
baked into every round's metric.
**Why it happens:** Same pattern as BSL-05 — fixed in baseline + personalized, not yet in
adaptive. Also: globally re-seeding `random` affects any other code path that uses stdlib
random elsewhere in the same process.
**How to avoid:** Strip `import random`, `random.seed`, `random.sample` from `task.py`; replace
the negative candidate sampling with `np_rng(run_seed, user_idx, round_num, "eval_neg").choice(
negative_candidates, num_negatives, replace=False)`. Backward-compat: keep `seed: int = 42`
param with a documented "IGNORED" note (Phase 2+3 precedent).
**Warning signs:** `sampled_ndcg@10` variance across rounds near zero; cross-run byte-identical
negatives (a deterministic-over-non-deterministic-domain regression).

### Pitfall 5: Silent cache contamination across experiments with different hyperparameters
**What goes wrong:** Current `.embedding_cache/partition_X/user_embeddings.pt` has no signature
check. Switching `embedding-dim=128` → `64`, or `alpha-method=hierarchical_conditional` →
`multi_factor`, or turning `enable-per-user-alpha` on/off, silently loads stale tensors with
mismatched shapes/semantics. The model's `set_local_parameters` has a "truncate to fit" branch
(`models/dual_personalized_bpr_mf.py:635`) that silently zero-pads or truncates.
**Why it happens:** Phase 3 shipped the fix for 2-key single-row caches at schema_version=1.
Phase 4's extended key set has never had signature protection.
**How to avoid:** D-02 schema_version=2 with 12 signature fields (6 Phase-3 + 6 adaptive-
specific). Hard-fail on delta with `rm -rf` hint (D-04).
**Warning signs:** Two runs with "clean cache" and otherwise identical config but different
hyperparameters produce nearly-identical results (cache wasn't actually clean); sudden drop in
metric after changing an apparently-minor config.

### Pitfall 6: Cold-start round contrastive loss on Xavier noise
**What goes wrong:** InfoNCE computes `sim(p_local, p_effective)` where the positive pair is
`(p_local[u], p_effective[u])`. On round 1 for a cold-start user, `p_local[u]` is Xavier-init
noise. Contrastive loss pulls Xavier noise toward the alpha-blended version of itself (which
is also partially noise-derived) or pushes it away from other users' similarly-noisy embeddings.
Neither signal is useful; both consume gradient budget.
**Why it happens:** D-14 skip-contrastive-on-cold-round not currently implemented.
**How to avoid:** D-14 locks the skip. When `is_cold_round=True`, set `contrastive_lambda_eff =
0.0` for that round. Next round, cache exists, resume normal contrastive loss.
**Warning signs:** First-round train loss dominated by contrastive term; NDCG@10 WORSE than
pure BPR on the first few rounds.

### Pitfall 7: InfoNCE contrastive on batch-size=1 cross-device
**What goes wrong:** In cross-device (1 user per client), `torch.unique(user_ids)` produces a
length-1 tensor; `InfoNCEContrastiveLoss.forward` returns 0.0 for `batch_size <= 1`
(`losses.py:173-174`). The contrastive regularization effectively never fires in benchmark mode.
**Why it happens:** InfoNCE needs cross-user negatives to discriminate; in cross-device with
1-user clients there are no other users in the same batch.
**How to avoid:** **This is an open design question.** CONTEXT.md locks `contrastive-lambda=0.1`
as the thesis default (D-12), but under cross-device this lambda contributes 0.0 to the loss.
Three options:
  (a) **Document the emergent "no-op" behavior** — the flag is reserved for ablations that
      run multi-user-per-client or for potential multi-batch aggregation; in the thesis
      benchmark row contrastive is effectively disabled. Update `federated-adaptive-
      personalized-cf/claude.md` to reflect this.
  (b) **Use intra-user temporal negatives** — different item positives for the same user at
      different mini-batch positions serve as in-batch negatives. Non-trivial rewrite of
      `losses.py` + `task.py` call site; larger scope than Phase 4.
  (c) **Server-side negative broadcast** — server injects other users' embeddings via the
      per-round train_config. Fundamentally breaks privacy (broadcasting user embeddings to
      clients); out of scope.
**Recommendation:** Option (a) — document + move on. The contrastive ablation story remains
valid for cross-silo (documented as deferred). Phase 4 plan should call this out explicitly
and NOT write code that pretends contrastive fires.
**Warning signs:** `contrastive_lambda` positive in config but train_loss shows no contrastive
contribution; alpha-mean metric shows no signal of contrastive regularization.

### Pitfall 8: Per-user alpha `nn.Embedding(num_users=6040)` explodes disk cache size
**What goes wrong:** `enable_per_user_alpha(num_users=6040)` creates an `nn.Embedding(6040, 1)`
= 24 KB per client. With the full adaptive LOCAL key set (user_embeddings 3 MB, user_bias 24 KB,
personal_mlp ~400 KB at mlp_hidden_dims=[512,256,128], fusion_layer 16 B, logit_alpha 24 KB,
item_perturbation 3706 * 128 * 4 = 1.8 MB), a single partition's `.pt` file is ~5.3 MB.
At 6040 partitions that's 32 GB of disk cache — feasible but not free.
**Why it happens:** D-01 locks single-file-per-partition. `num_users=6040` in `logit_alpha`
means every client is storing a FULL user-space alpha table even though only index `user_idx`
is active for that client.
**How to avoid:** The benchmark cross-device case (1 user per client) could use `num_users=1`
in `enable_per_user_alpha` — but then `compute_per_user_alpha(user_stats)` returns a 1-entry
dict, and `set_alpha(client_alpha)` already does the right thing as a scalar. The per-user
alpha table is ACTUALLY a cross-silo-era design; under cross-device it collapses to the scalar
alpha already provided by `set_alpha`. **Recommendation:** For cross-device benchmark mode,
consider: `enable_per_user_alpha(num_users=1)` — one scalar learnable alpha, 16 B instead of
24 KB. Same for `item_perturbation`: 1.8 MB is unavoidable if perturbation is full-item-space
(which is correct — any user may touch any item). This is NOT a blocker for Phase 4 but is
a disk-efficiency win worth flagging.
**Warning signs:** `.embedding_cache/{run_id}/` directory size grows to tens of GB; slow
per-round disk I/O; W&B upload of manifests balloons.

## Code Examples

### Loading signature-v2 cache with hard-fail
```python
# Source: Phase 3 Plan 03 _load_local_user_state pattern, extended with v2 fields
# Location: federated_adaptive_personalized_cf/client_app.py

def _signature_fields_v2(*, run_id: str, method: str, num_users: int, num_items: int,
                         dim: int, split_hash: str, alpha_method: str, fusion_type: str,
                         mlp_hidden_dims: str, per_user_alpha_enabled: bool,
                         item_perturbation_enabled: bool, contrastive_lambda: float
                         ) -> Dict[str, Any]:
    """D-02: schema_version=2 with 12 signature fields (6 Phase 3 + 6 adaptive)."""
    return {
        "schema_version": 2,
        "run_id": run_id,
        "method": method,                       # e.g., "dual"
        "num_users": num_users,
        "num_items": num_items,
        "dim": dim,
        "split_hash": split_hash,
        "alpha_method": alpha_method,           # NEW
        "fusion_type": fusion_type,             # NEW
        "mlp_hidden_dims": mlp_hidden_dims,     # NEW, joined "512,256,128"
        "per_user_alpha_enabled": per_user_alpha_enabled,   # NEW
        "item_perturbation_enabled": item_perturbation_enabled,  # NEW
        "contrastive_lambda": float(contrastive_lambda),    # NEW
    }

def _load_local_user_state_v2(*, partition_id: int, run_id: str, reuse_cache: bool,
                              signature: Dict[str, Any]) -> Optional[Dict]:
    """D-04 mirror: loud per-field delta on mismatch."""
    cache_dir = _cache_dir_for_run(run_id=run_id, reuse_cache=reuse_cache,
                                   signature=signature)
    pt_path = cache_dir / f"partition_{partition_id}.pt"
    manifest_path = cache_dir / "manifest.json"
    if not pt_path.exists():
        return None  # cold-start (D-15)
    with open(manifest_path) as f:
        on_disk = json.load(f)
    deltas = [k for k in signature if on_disk.get(k) != signature.get(k)]
    if deltas:
        delta_lines = "\n".join(f"  {k}: on-disk={on_disk.get(k)!r} vs current={signature[k]!r}"
                                for k in deltas)
        raise RuntimeError(
            f"D-04: .embedding_cache/{run_id}/ manifest mismatch (schema_version=2):\n"
            f"{delta_lines}\n"
            f"Run: rm -rf .embedding_cache/{run_id}/"
        )
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    # D-10 extended shape guard: verify LOCAL key set matches the model's _LOCAL_PARAMS
    # (computed from signature flags — deterministic reconstruction).
    return state
```

### Adaptive strategy override with best_prototype snapshot
```python
# Source: CONTEXT D-05 + Phase 3 Plan 01 PersonalizedSplitFedAvg
# Location: federated_adaptive_personalized_cf/strategy.py

_GLOBAL_PARAM_KEYS = frozenset([
    "item_embeddings.weight", "item_bias.weight", "global_bias",
])

# LOCAL keys are dynamically extended at runtime in DualPersonalizedBPRMF._LOCAL_PARAMS;
# the frozenset here is the BASE set. For manifest signature fields we use the
# per_user_alpha_enabled / item_perturbation_enabled flags instead.
_LOCAL_PARAM_KEYS_BASE = frozenset([
    "user_embeddings.weight", "user_bias.weight",
    # "_logit_alpha.weight" appended if enable-per-user-alpha
    # "_item_perturbation.weight" appended if enable-item-perturbation
    # "personal_mlp.*" per mlp-hidden-dims
    # "fusion_gate" | "fusion_layer.weight" + "fusion_layer.bias" per fusion-type
])


class AdaptiveSplitFedAvg(BaseFedAvg):
    """FedAvg for adaptive split learning with prototype EMA + best-round snapshot."""

    def __init__(self, fraction_fit: float = 1.0, prototype_momentum: float = 0.9, **kwargs):
        super().__init__(fraction_fit=fraction_fit, **kwargs)
        self.global_param_keys = _GLOBAL_PARAM_KEYS
        self._is_split_learning = True
        self.prototype_momentum = prototype_momentum
        self._global_prototype: Optional[np.ndarray] = None
        # D-05: best-round prototype snapshot, mirrors best_arrays pattern
        self.best_prototype: Optional[np.ndarray] = None

    def get_global_prototype(self) -> Optional[np.ndarray]:
        return self._global_prototype

    # aggregate_fit INHERITED UNCHANGED from BaseFedAvg per D-23; the split happens at the
    # client. BUT: we need to aggregate prototypes too, so we DO override aggregate_fit
    # here (contrast with Phase 3 where aggregate_fit was pure-inherited). _aggregate_prototypes
    # is called AFTER super().aggregate_fit runs the weighted-average of GLOBAL params.
    def aggregate_fit(self, server_round, results, failures):
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)
        self._aggregate_prototypes(results)
        if self._global_prototype is not None:
            metrics["global_prototype_norm"] = float(np.linalg.norm(self._global_prototype))
        return aggregated_params, metrics

    def _aggregate_prototypes(self, results):
        """EMA update on weighted mean of client prototypes. UNCHANGED from existing code."""
        # ... identical to existing strategy.py:138-186 ...

    # aggregate_evaluate: sufficient-stat sum + per-group D-22 routing + D-16 alpha diagnostics
    def aggregate_evaluate(self, server_round, results, failures):
        """Sum sufficient stats across clients; emit thesis metrics dict."""
        # ... cloned from Phase 3 PersonalizedSplitFedAvg.aggregate_evaluate ...
        # PLUS: D-16 alpha diagnostics from client FitMetricsContract per_user_alpha_* fields
        # (already populated by client_app.py at lines 420-426).
```

### Cold-start branch in training loop (D-13 + D-14)
```python
# Source: CONTEXT D-13, D-14
# Location: federated_adaptive_personalized_cf/task.py (inside train_bpr_mf or train_dual_personalized)

def train_bpr_mf_adaptive(model, trainloader, epochs, lr, device, *,
                          run_seed: int, user_idx: int, round_num: int,
                          exclude_items: Set[int], rng: np.random.Generator,
                          is_cold_round: bool = False,           # NEW: D-13/D-14 signal
                          contrastive_lambda: float = 0.1,
                          contrastive_tau: float = 0.1,
                          proximal_mu: float = 0.0, global_params=None,
                          global_param_names=None, weight_decay: float = 1e-5,
                          num_negatives: int = 1) -> float:
    """..."""
    # D-13: override alpha to 0 for this round (prototype-only blend).
    # D-14: disable contrastive this round (Xavier noise anchor is useless).
    if is_cold_round and hasattr(model, "set_alpha"):
        saved_alpha = model.get_alpha()
        model.set_alpha(0.0)
        contrastive_lambda_eff = 0.0
    else:
        contrastive_lambda_eff = contrastive_lambda

    try:
        # ... existing training loop ...
        # Negatives use exclude_items (ADP-05 + FND-03):
        user_rated = set()
        for batch in trainloader:
            for u, i in zip(batch['user'].numpy(), batch['item'].numpy()):
                user_rated.add(i)
        user_rated |= exclude_items  # fold in held-out test positive
        # ... sample negatives via rng, NOT model.sample_negatives (ADP-06) ...
        # ... contrastive branch checks contrastive_lambda_eff > 0 ...
    finally:
        if is_cold_round and hasattr(model, "set_alpha"):
            model.set_alpha(saved_alpha)  # restore so eval uses original alpha next round
```

### Alpha factory unit test covering every conditional rule branch (ADP-07)
```python
# Source: CONTEXT ADP-07 + existing adaptive_alpha.py compute_factors (line 488)
# Location: federated-adaptive-personalized-cf/tests/test_alpha_factory.py

import pytest
from federated_adaptive_personalized_cf.models.adaptive_alpha import (
    AlphaConfig, HierarchicalConditionalAlphaConfig, create_alpha_computer,
    DataQuantityAlpha, MultiFactorAlpha, HierarchicalConditionalAlpha,
)


# ======================================================================================
# DataQuantityAlpha — floor/ceiling clip hits
# ======================================================================================
def test_data_quantity_min_clip_at_very_sparse():
    config = AlphaConfig(method="data_quantity", min_alpha=0.1, max_alpha=0.95,
                         quantity_threshold=100, quantity_temperature=0.05)
    computer = DataQuantityAlpha(config)
    # n=0: sigmoid((0-100)*0.05) = sigmoid(-5) ≈ 0.0067 → clipped to 0.1
    assert computer.compute(0) == pytest.approx(0.1)
    # n=50: sigmoid((50-100)*0.05) = sigmoid(-2.5) ≈ 0.0759 → clipped to 0.1 (still clipped)
    assert computer.compute(50) == pytest.approx(0.1)

def test_data_quantity_max_clip_at_dense():
    config = AlphaConfig(method="data_quantity", min_alpha=0.1, max_alpha=0.95,
                         quantity_threshold=100, quantity_temperature=0.05)
    computer = DataQuantityAlpha(config)
    # n=200: sigmoid(5) ≈ 0.9933 → clipped to 0.95
    assert computer.compute(200) == pytest.approx(0.95)

def test_data_quantity_midpoint():
    config = AlphaConfig(method="data_quantity", min_alpha=0.1, max_alpha=0.95,
                         quantity_threshold=100, quantity_temperature=0.05)
    computer = DataQuantityAlpha(config)
    assert computer.compute(100) == pytest.approx(0.5)


# ======================================================================================
# HierarchicalConditionalAlpha — each rule branch fires on crafted inputs
# ======================================================================================
def test_hc_sparse_penalty_applies():
    """Rule 1: n < sparse_threshold (20) → 50% penalty at n=0."""
    config = HierarchicalConditionalAlphaConfig(sparse_threshold=20, sparse_penalty_max=0.5)
    computer = HierarchicalConditionalAlpha(config)
    factors = computer.compute_factors({"n_interactions": 5, "genre_entropy": 1.5,
                                        "n_unique_items": 5, "rating_std": 0.75})
    assert "sparse" in factors["applied_rules"]
    assert factors["alpha"] >= 0.1 and factors["alpha"] <= 0.95  # clipped in range

def test_hc_niche_bonus_applies():
    """Rule 2: low diversity + high quantity → +0.15 bonus."""
    config = HierarchicalConditionalAlphaConfig(niche_diversity_threshold=0.25,
                                                niche_quantity_threshold=0.6,
                                                niche_bonus=0.15, max_entropy=3.0)
    computer = HierarchicalConditionalAlpha(config)
    # High quantity: n=200 → f_quantity ≈ 0.99 > 0.6
    # Low diversity: genre_entropy=0.5 → f_diversity = 0.5/3.0 ≈ 0.17 < 0.25
    factors = computer.compute_factors({"n_interactions": 200, "genre_entropy": 0.5,
                                        "n_unique_items": 200, "rating_std": 0.75})
    assert "niche" in factors["applied_rules"]
    assert factors["alpha"] >= 0.1 and factors["alpha"] <= 0.95

def test_hc_inconsistent_penalty_applies():
    """Rule 3: low consistency → 30% penalty."""
    config = HierarchicalConditionalAlphaConfig(inconsistent_threshold=0.3,
                                                inconsistent_penalty=0.3,
                                                max_rating_std=1.5)
    computer = HierarchicalConditionalAlpha(config)
    # High rating_std=1.45 → f_consistency = 1 - 1.45/1.5 ≈ 0.033 < 0.3
    factors = computer.compute_factors({"n_interactions": 100, "genre_entropy": 2.0,
                                        "n_unique_items": 100, "rating_std": 1.45})
    assert "inconsistent" in factors["applied_rules"]
    assert factors["alpha"] >= 0.1 and factors["alpha"] <= 0.95

def test_hc_completionist_bonus_applies():
    """Rule 4: high coverage + low diversity → +0.1 bonus."""
    config = HierarchicalConditionalAlphaConfig(completionist_coverage=0.7,
                                                completionist_diversity=0.3,
                                                completionist_bonus=0.1,
                                                coverage_threshold=100, max_entropy=3.0)
    computer = HierarchicalConditionalAlpha(config)
    # n_unique=90 → f_coverage = 0.9 > 0.7; genre_entropy=0.5 → f_diversity ≈ 0.17 < 0.3
    factors = computer.compute_factors({"n_interactions": 90, "genre_entropy": 0.5,
                                        "n_unique_items": 90, "rating_std": 0.75})
    assert "completionist" in factors["applied_rules"]
    assert factors["alpha"] >= 0.1 and factors["alpha"] <= 0.95

def test_hc_min_max_clip_bounds():
    """Every input produces alpha in [min_alpha, max_alpha]."""
    config = HierarchicalConditionalAlphaConfig(min_alpha=0.1, max_alpha=0.95)
    computer = HierarchicalConditionalAlpha(config)
    # Adversarial edge cases
    for n, ge, nu, rs in [(0, 0.0, 0, 0.0), (0, 3.0, 0, 1.5),
                          (10000, 0.0, 10000, 0.0), (10000, 3.0, 10000, 1.5),
                          (5, 1.5, 5, 0.75), (1000, 2.0, 100, 1.0)]:
        alpha = computer.compute_from_stats({"n_interactions": n, "genre_entropy": ge,
                                             "n_unique_items": nu, "rating_std": rs})
        assert 0.1 <= alpha <= 0.95, f"Alpha {alpha} out of bounds for n={n}, ge={ge}"


# ======================================================================================
# MultiFactorAlpha — clip bounds on extremes (same adversarial grid)
# ======================================================================================
def test_multi_factor_clip_bounds():
    config = AlphaConfig(method="multi_factor", min_alpha=0.1, max_alpha=0.95,
                         factor_weights={"quantity": 0.4, "diversity": 0.25,
                                         "coverage": 0.2, "consistency": 0.15})
    computer = MultiFactorAlpha(config)
    for n, ge, nu, rs in [(0, 0.0, 0, 0.0), (10000, 3.0, 10000, 0.0)]:
        alpha = computer.compute_from_stats({"n_interactions": n, "genre_entropy": ge,
                                             "n_unique_items": nu, "rating_std": rs})
        assert 0.1 <= alpha <= 0.95


# ======================================================================================
# Factory dispatch
# ======================================================================================
def test_factory_returns_correct_computer_class():
    assert isinstance(create_alpha_computer(AlphaConfig(method="data_quantity")),
                      DataQuantityAlpha)
    assert isinstance(create_alpha_computer(AlphaConfig(method="multi_factor")),
                      MultiFactorAlpha)
    assert isinstance(create_alpha_computer(AlphaConfig(method="hierarchical_conditional"),
                                            hc_config=HierarchicalConditionalAlphaConfig()),
                      HierarchicalConditionalAlpha)

def test_factory_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        AlphaConfig(method="invalid_method")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Cross-silo with `num-supernodes=5` | Cross-device with `num-supernodes=6040`, 1 user = 1 client | Phase 2 (baseline) + Phase 3 (personalized); Phase 4 closes this for adaptive | Methodologically required for thesis review; every published FedRec paper is cross-device |
| Global `random.seed()` in `evaluate_ranking_sampled` | FND-06 per-call `np_rng(run_seed, user_idx, round_num, "eval_neg")` | Phase 2 Plan 03 (baseline); carried forward | Deterministic, reproducible, cross-run byte-identical negative sets |
| Per-client-ratio mean in `aggregate_evaluate` | `sum(hit_count) / sum(evaluated_users)` sufficient-stat | Phase 2 Plan 01 (baseline) | Eliminates per-user double-counting bug; thesis metrics are now mathematically sound |
| `strategy.aggregate_fit` as-is (no prototype snapshot) | `aggregate_fit` + `_aggregate_prototypes` + D-05 best-prototype tracking | Phase 4 (this) | Prototype EMA state matches restored global params |
| `enable_per_user_alpha` AFTER `load_local_user_embeddings` | `enable_per_user_alpha` BEFORE load (D-03) | Phase 4 (this) | Per-user alpha actually accumulates across rounds |
| Schema-less `.embedding_cache/partition_X/user_embeddings.pt` | schema_version=2 manifest-sidecar with 12 signature fields | Phase 4 (this, extending Phase 3 schema_version=1) | Silent cross-experiment contamination impossible; `rm -rf` hint on any mismatch |
| No cold-start blend override | Prototype-only blend (α=0) on first round (D-13), skip contrastive (D-14) | Phase 4 (this) | Sparse-user NDCG@10 benefits directly; Xavier noise stops dominating the first-round embedding |
| Post-hoc `alpha_analysis.py` invocation for clip-hit-rate diagnostic | Per-round `alpha_clip_hit_rate` + quartiles in `eval_metrics_history` (D-16) | Phase 4 (this) | Thesis artifact answers CONCERNS.md clip-floor critique directly from JSON |

**Deprecated/outdated:**
- Cross-silo `num-supernodes=5` — kept as `mode="cross_silo_legacy"` but frozen by D-02-style
  NotImplementedError at dataset/server entry; pre-Phase-4 commits are the reproduction oracle.
- `save_local_user_embeddings` / `load_local_user_embeddings` with metadata `_round` / `_timestamp`
  prefix keys — replaced by manifest.json sidecar.
- `test`/`evaluate_ranking` (all-items, rating-pred) on the wire — kept for client-side
  diagnostic but server strategy contract ignores `allrank_*` metrics; BSL-07 invariant.
- Model path `torch.load(..., weights_only=False)` — replaced by `weights_only=True` default.

## Open Questions

1. **Should `DualPersonalizedBPRMF` collapse to single-row (PSN-06 mirror)?**
   - What we know: Phase 3 shipped the single-row collapse for BPRMF + BasicMF in the
     personalized module, saving ~3 MB per client at disk. The adaptive module's
     `DualPersonalizedBPRMF` still has the `num_users × d` ghost table at
     `dual_personalized_bpr_mf.py:112`.
   - What's unclear: The dual-model training loop references `user_ids` throughout (forward,
     compute_score, get_effective_embedding). Collapsing to single-row requires rewriting
     that indexing to collapse too. Substantial surgery.
   - Recommendation: **Phase 4 does NOT collapse.** D-24 gradient isolation snapshot/restore
     (from Phase 2 Plan 03) is already the mitigation. Plan the collapse as Phase 4.5 follow-up
     if disk pressure becomes real (see Pitfall 8). Phase 4's scope is bug fix + migration,
     not a model-class rewrite.

2. **How does `per_user_alpha` interact with cross-device (1 user per client)?**
   - What we know: `enable_per_user_alpha(num_users=model.num_users)` creates an `nn.Embedding(6040, 1)`
     per client. At 6040 clients that's 6040^2 · 4 B = 146 MB total cross-repo (acceptable).
     But each client stores its own full 6040-entry table, only index `user_idx` of which is
     semantically active.
   - What's unclear: Collapsing `logit_alpha` to a scalar per-client under cross-device would
     be architecturally cleaner (eliminates the ghost-column-of-the-ghost-table) but breaks
     symmetry with cross-silo ablation.
   - Recommendation: **Keep the table for Phase 4.** Document in the plan that cross-device
     uses full-table for symmetry; follow-up optimization is Phase 4.5. The storage is non-
     critical.

3. **When does contrastive loss actually contribute under cross-device 1-user-per-client?**
   - What we know: InfoNCE batch-size guard returns 0.0 when batch has ≤1 unique user.
     In cross-device, every batch has exactly 1 unique user.
   - What's unclear: Is `contrastive-lambda=0.1` in D-12 a thesis-benchmark default that
     we know to be a no-op in benchmark mode but is carried forward for ablation sweeps?
   - Recommendation: **Document the no-op in the plan and in `federated-adaptive-personalized-
     cf/claude.md`.** D-12's `contrastive-lambda=0.1` becomes a "reserved for ablation"
     marker in the benchmark row; the thesis comparison table should note that contrastive
     loss is inactive under cross-device in the default run.

4. **How does the D-24 gradient isolation interact with `_logit_alpha.weight` and
   `_item_perturbation.weight`?**
   - What we know: In Phase 2, D-24 snapshot/restore protected non-user rows of user_embeddings
     from Adam weight-decay drift. In Phase 4 the LOCAL key set expands.
   - What's unclear: Does `_logit_alpha.weight` (num_users × 1 embedding) need D-24 protection?
     Does `_item_perturbation.weight` (num_items × d embedding)?
   - Recommendation: `_logit_alpha` is per-user-indexed — same ghost-table concern as
     user_embeddings; needs D-24 protection. `_item_perturbation` is per-ITEM-indexed and gets
     full-table gradients for every training batch (every item appears in some batch, either as
     positive or negative), so D-24 does NOT apply. Plan should thread BOTH user_embeddings
     AND _logit_alpha.weight through the snapshot/restore bracket; explicitly DOCUMENT that
     _item_perturbation is outside D-24 scope by design.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `pytest>=7.0` (dev extra) |
| Config file | none — implicit `[pytest]` via per-module `[tool.pytest.ini_options]` if added, else ad-hoc. Phase 2+3 precedent: no config file, tests collected by default discovery. |
| Quick run command | `pytest federated-adaptive-personalized-cf/tests/ -v -x` |
| Full suite command | `pytest federated-adaptive-personalized-cf/tests/ scripts/foundation/tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ADP-01 | `pyproject.toml` carries `num-supernodes=6040 partition-mode="natural"` in both federation blocks + schema_version=2 driver keys + `[dev]` extra | grep regression | `pytest tests/test_pyproject_shape.py` (source-level grep) | ❌ Wave 0 |
| ADP-02 | `_LOCAL_PARAMS` contains `_logit_alpha.weight` and `_item_perturbation.weight` at load time; cached values restored (not re-inited) | unit | `pytest tests/test_dual_model.py::test_enable_before_load_restores_cached_alpha -x` | ❌ Wave 0 |
| ADP-03 | `AdaptiveSplitFedAvg.best_prototype` snapshotted when `current_ndcg > best_metric`; `_global_prototype = best_prototype` at end-of-training | unit + integration | `pytest tests/test_strategy.py::test_best_prototype_snapshot_at_best_round` + `pytest tests/test_server_integration.py::test_d07_best_prototype_restored_before_final_broadcast` | ❌ Wave 0 |
| ADP-04 | Benchmark mode: `assert_benchmark_one_user_per_client(profile, num_users_in_client, overrides)` raises on >1 user, passes on =1 | unit | `pytest tests/test_client_assertion.py::test_benchmark_mode_asserts_one_user` | ❌ Wave 0 |
| ADP-05 | ExclusionTable.for_user merged into user_rated_items; train negatives exclude held-out test positive | unit | `pytest tests/test_task_rng.py::test_train_negatives_exclude_test_positive` | ❌ Wave 0 |
| ADP-06 | server_rng(run_seed) byte-identical across subprocess reruns + stdlib random eradicated + sufficient-stat aggregator + signature v2 cache | unit + grep + integration | `pytest tests/test_task_rng.py::test_random_seed_calls_stripped` (grep task.py + client_app.py + server_app.py for `^import random$\|random.seed(\|random.sample(`) + `pytest tests/test_server_integration.py::test_server_rng_reproducible_per_round_selection` + `pytest tests/test_embedding_cache_manifest_v2.py` | ❌ Wave 0 |
| ADP-07 | All three alpha factory classes return values in `[min_alpha, max_alpha]` for a grid of edge-case inputs; each HC rule branch fires for crafted inputs | unit | `pytest tests/test_alpha_factory.py` | ❌ Wave 0 |
| ADP-08 | `_manifest.module == "adaptive"` + all four IMP-2 fingerprints + `best_prototype: List[float]` field present in result JSON | unit + integration | `pytest tests/test_server_integration.py::test_build_run_manifest_module_adaptive_with_best_prototype` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest federated-adaptive-personalized-cf/tests/ -v -x` (tests ship with
  `pytestmark = pytest.mark.skipif(not foundation_index.json.exists(), ...)` so a minimal clone
  without `data/derived/` cleanly skips; quick-run under 10s)
- **Per wave merge:** `pytest federated-adaptive-personalized-cf/tests/ scripts/foundation/tests/`
- **Phase gate:** Full suite green + `grep -rnE "^import random$|random.seed\(|random.sample\(" federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/` returns 0 + `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `federated-adaptive-personalized-cf/tests/__init__.py` — package marker
- [ ] `federated-adaptive-personalized-cf/tests/conftest.py` — fake_evaluate_res + fake_client_proxy fixtures (copy from Phase 3 `federated-personalized-cf/tests/conftest.py`)
- [ ] `federated-adaptive-personalized-cf/tests/test_strategy.py` — AdaptiveSplitFedAvg sufficient-stat + best_prototype snapshot + aggregate_fit override (covers ADP-03 in unit form)
- [ ] `federated-adaptive-personalized-cf/tests/test_dual_model.py` — enable_per_user_alpha + enable_item_perturbation BEFORE load restores cache (covers ADP-02)
- [ ] `federated-adaptive-personalized-cf/tests/test_task_rng.py` — BSL-05-style cross-file strip + FND-03 exclusion + cold-round α=0 (covers ADP-05 + ADP-06 RNG half)
- [ ] `federated-adaptive-personalized-cf/tests/test_client_assertion.py` — one-user assert + FitMetricsContract + partition_id (covers ADP-04 + client half of ADP-06)
- [ ] `federated-adaptive-personalized-cf/tests/test_embedding_cache_manifest_v2.py` — schema_version=2 + 12 signature fields + hard-fail delta (covers ADP-06 cache half)
- [ ] `federated-adaptive-personalized-cf/tests/test_alpha_factory.py` — ADP-07 crafted-input edge cases (covers ADP-07)
- [ ] `federated-adaptive-personalized-cf/tests/test_server_integration.py` — server_rng + best_prototype snapshot + D-07 restored broadcast + cold-start counter + D-15 manifest (covers ADP-03 integration + ADP-06 server half + ADP-08)
- [ ] `federated-adaptive-personalized-cf/pyproject.toml` `[project.optional-dependencies] dev = ["pytest>=7.0"]` — same pattern as Phase 2 Plan 02 + Phase 3 Plan 02
- [ ] Framework install: `pip install -e "federated-adaptive-personalized-cf[dev]"` (documented in docs/setup.md)

**Note on foundation bundle:** The `pytestmark = pytest.mark.skipif(not
(data_derived() / "foundation_index.json").exists(), reason="foundation bundle not committed")`
guard mirrors the Phase 2+3 precedent — if `data/derived/` is missing, tests collect + skip.
The bundle is already committed per Phase 1 Plan 02 D-04 lock-forever decision.

## Sources

### Primary (HIGH confidence)
- `.planning/phases/04-adaptive-migration-bug-fixes/04-CONTEXT.md` — Locked D-01..D-16 + carried-forward decisions.
- `.planning/REQUIREMENTS.md` §ADP — ADP-01..08 canonical text.
- `.planning/codebase/CONCERNS.md` — bug list with line numbers (enable-after-load at `client_app.py:247-351`, clip-floor at `adaptive_alpha.py:208/306/339/486`, global random.seed at `task.py:952-953`, silent shape-mismatch at `client_app.py:141-142`).
- `.planning/phases/02-baseline-migration/02-baseline-migration-03-SUMMARY.md` — client_app + task FND-06 + FND-03 + D-24 template.
- `.planning/phases/02-baseline-migration/02-baseline-migration-04-SUMMARY.md` — server_app mode resolver + best-round restore + D-15 template.
- `.planning/phases/03-personalized-migration/03-personalized-migration-01-SUMMARY.md` — PersonalizedSplitFedAvg/FedProx base for AdaptiveSplitFedAvg/FedProx clone.
- `.planning/phases/03-personalized-migration/03-personalized-migration-03-SUMMARY.md` — client_app + task + manifest-sidecar v1 template for v2 extension.
- `.planning/phases/03-personalized-migration/03-personalized-migration-04-SUMMARY.md` — server_app + D-13 cold-start counter pattern.
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/dual_personalized_bpr_mf.py` (lines 100-175, 400-570) — enable-method behaviors, `_LOCAL_PARAMS` property, `compute_user_prototype`.
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/strategy.py` (lines 90-186, 232-330) — existing `_global_prototype` EMA, `_aggregate_prototypes` logic.
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/adaptive_alpha.py` (lines 98-170, 380-550) — hierarchical conditional config + `compute_factors` for ADP-07 test design.
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/losses.py` (lines 135-190) — InfoNCE batch<=1 guard confirms cross-device no-op.
- `scripts/foundation/fedrec_foundation/manifest.py` (lines 1-250) — RunManifest dataclass + embed_manifest_in_result mutation pattern confirming `_manifest` dict is extensible.

### Secondary (MEDIUM confidence)
- `federated-personalized-cf/federated_personalized_cf/server_app.py` (lines 490-800) — canonical Phase 3 server-side pattern (discovery round, D-27 best-round, cold-start counter) to mirror with best-prototype extension.
- `federated-personalized-cf/federated_personalized_cf/client_app.py` — Phase 3 `_signature_fields` + `_cache_dir_for_run` + `_save/load_local_user_state` pattern for schema_version=2 extension.
- `federated-personalized-cf/tests/` — 7 test files exemplifying Phase-3 TDD structure that Phase 4 clones and extends.

### Tertiary (LOW confidence — NONE in this research)
No LOW-confidence claims remain. Every finding is cross-verified against (a) CONTEXT.md decisions
or (b) shipped Phase-2/3 code + SUMMARY docs or (c) direct inspection of adaptive module source.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all deps existing across Phase 2+3; no new libraries or versions.
- Architecture: HIGH — patterns 1-5 are cloned+extended from Phase 2+3 shipped templates with
  CONTEXT.md-locked delta at each extension point.
- Pitfalls: HIGH — Pitfalls 1-5 are documented in CONCERNS.md with line numbers; Pitfall 6-8
  are architectural consequences verified by direct code inspection.
- ADP-07 test design: HIGH — crafted inputs derived from HierarchicalConditionalAlphaConfig
  defaults + existing `compute_factors` helper (already returns `applied_rules`).
- Validation architecture: HIGH — Nyquist dim 8 mapping follows the Phase 2+3 precedent where
  each requirement has a unit-test + integration-test pair.

**Research date:** 2026-04-20
**Valid until:** 30 days (stable-stack research; adaptive module code is frozen outside
Phase 4's explicit scope; Flower API stable at 1.22.x by Phase 1 pin).

---

*Phase: 04-adaptive-migration-bug-fixes*
*Research gathered: 2026-04-20*
