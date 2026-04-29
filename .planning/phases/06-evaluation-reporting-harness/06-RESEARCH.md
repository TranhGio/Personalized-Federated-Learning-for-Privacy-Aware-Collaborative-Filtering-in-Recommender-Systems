# Phase 6: Evaluation & Reporting Harness - Research

**Researched:** 2026-04-29
**Domain:** Server-side result emission, repo-root path resolution, post-loop final-eval broadcast wiring, manifest schema evolution, per-group metric plumbing, W&B summary keying.
**Confidence:** HIGH (every recommendation traces to existing in-repo code; nine cross-cutting questions answered against concrete file/line refs).

## Summary

Phase 6 is **pure server-side / strategy-side / manifest-side plumbing work**. Every layer Phase 6 needs already exists somewhere in the codebase:

- Repo-root resolution exists at `scripts/foundation/fedrec_foundation/paths.py:16` (`repo_root()`) — reused for `data_derived()` and `ml1m_dir()`. Phase 6 adds one new helper that returns `<repo>/results/federated/<module>/<run_id>/`.
- Per-group sufficient-stat aggregation exists in all four `strategy.py` files (`hit_count_{sparse,medium,dense}_at10`, `ndcg_sum_..._at10`, `evaluated_users_{sparse,medium,dense}` already flow through `_sum_sufficient_stats` and `_sufficient_stats_to_thesis_metrics`). EVL-02 is mostly a re-emission rename, not a new aggregation.
- D-15 manifest double-write (`embed_manifest_in_result` + `write_manifest_sibling`) is already shared by all four modules. Phase 6 changes only **where** the result file is written (and adds new fields inside the manifest); the helper API is untouched.
- D-27 in-memory best-round snapshot (`best_arrays = ArrayRecord(...)`) is already in baseline/personalized/adaptive. D-13 carry-forward shipped the same idiom for pfedrec. D-05/D-07 layered prototype snapshot+restore on top for adaptive.

What is new in Phase 6:
1. **Repo-root-anchored per-run directory layout** (`<repo>/results/federated/<module>/<run_id>/{results.json, manifest.json}`) replacing the four current ad-hoc paths (`../results/federated`, `../results/federated/adaptive`, `../results/federated/pfedrec`, `../results/federated/`).
2. **One extra `@app.evaluate` broadcast after best-arrays restore**, gated on `mode_profile.checkpoint_rule == "best_round_restore"`. The broadcast reuses the existing intra-loop eval pattern (build `eval_messages`, call `grid.send_and_receive`, run `strategy.aggregate_evaluate`). Result becomes the canonical `best_*` block.
3. **`best_*` and `last_*` dual blocks** in `final_metrics`. Today three of four modules (personalized, adaptive, pfedrec) silently lookup `eval_metrics_history[best_round_num]` for their `final_metrics` value — that is exactly what D-06 forbids. Baseline does a centralized eval but on rating-prediction metrics only (RMSE/MAE), not on the thesis sufficient-stat metrics.
4. **Manifest schema bump** from version 1 → 2 to add `final_eval_round_index` and a typed `metrics` block. Bumping is mechanical because `RUN_MANIFEST_SCHEMA_VERSION` is already a module-level constant.

**Primary recommendation:** Land Phase 6 as four small surgical edits to the four `server_app.py` files plus one new foundation helper (`fedrec_foundation.paths.module_run_results_dir(module: str, run_id: str)`) and one `RunManifest` field addition. Do NOT touch `strategy.py`, `task.py`, `client_app.py`, or any `models/`. Update the four subprocess determinism guards in lockstep with the path change.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Results path schema (EVL-04)**
- **D-01:** Per-run directory layout: `results/federated/<module>/<run_id>/` containing `results.json` + `manifest.json` (and optional sidecars like `alpha_diagnostics.json` for adaptive). One directory per run; the directory IS the run identifier.
- **D-02:** Results root is **repo-root anchored**: `<repo>/results/federated/...` resolved via the foundation package (walk-up from `scripts/foundation/` or equivalent helper). Server_app must NOT use module-relative `../results/federated/`. Resolves the folded `phase2-baseline-determinism-path-bug.md` todo.
- **D-03:** Existing pre-Phase-6 result artifacts under `results/federated/` are **left untouched**. Cross-silo reproducibility from PROJECT.md constraint is preserved. New Phase-6 runs go to the new `<module>/<run_id>/` layout; legacy flat files coexist.
- **D-04:** Clean filenames inside the per-run directory: `results.json` + `manifest.json` (no run_id prefix, no best_round_<N> infix). best_round and run_id live INSIDE the manifest, not in the filename.

**W&B project routing (EVL-05)**
- **D-05:** Keep current default project `federated-cf-cross-device` for all four modules in cross-device modes. Already wired in every `server_app.py`; zero churn.

**Best-round restore semantics (EVL-01, EVL-06)**
- **D-06:** After restoring best-round arrays (and best_prototype for adaptive), **broadcast one extra `@app.evaluate` round** and emit those numbers as the canonical `best_*` block.
- **D-07:** Both `best_*` and `last_*` blocks live in the canonical artifact. `best_*` is the canonical reported metric. `last_*` is preserved as a diagnostic field for spotting overfitting / late-round drift.

**Per-user-group reporting (EVL-02, EVL-03)**
- **D-08:** Per-group fields are **HR@10 and NDCG@10 only**, for sparse / medium / dense. No per-group `eval_loss`, no per-group `alpha_*`.
- **D-09:** Sampling-exposure counts (`evaluated_users_{overall,sparse,medium,dense}`) reported **per-round AND in the canonical block**.

### Claude's Discretion

- Exact name of the foundation helper that resolves the per-run results dir (e.g., `fedrec_foundation.paths.module_run_results_dir()`). Planner picks naming to match Phase 1 conventions.
- Internal wiring of the "extra final eval round" in each module's `server_app.py` main loop (after the main FL loop exits, before W&B summary write, before manifest double-write). Standard pattern preferred but per-module adaptation acceptable.
- Schema of `manifest.json` evolution (adding new fields like `last_round_metrics`, `final_eval_round_index`) — must remain backward-readable but Phase 6 may bump a manifest schema version field.

### Deferred Ideas (OUT OF SCOPE)

- **Per-group eval_loss / per-group alpha breakdown** — Belongs in Phase 7 (Thesis Evaluation Run) ablations or as adaptive-module-internal logging.
- **W&B project rename to `thesis-crossdevice-*`** — Could revisit at thesis-paper time.
- **Per-mode W&B project split** — Considered, deferred. Single project + run tags is sufficient for now.
- **Migrating legacy flat result files into `_legacy/` subtree** — Deferred. Coexistence with new layout is acceptable.
- **Encoding `best_round` in filename** — Rejected by D-04 in favor of clean filenames + manifest-internal field.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVL-01 | Best-round restore: for every module, `best_*` metrics and the corresponding global + local + strategy state are saved; after the last round, the best-round state is restored and ONE final evaluation is written as the canonical result artifact. | §"Question 2: Extra-eval-round wiring per module" + §"Don't Hand-Roll" Pattern Reuse #2 (existing `grid.send_and_receive` eval pattern; D-27 best_arrays + D-05/D-07 best_prototype already in place). |
| EVL-02 | Per-user-group (sparse/medium/dense) NDCG@10 and HR@10 are emitted as first-class fields in every module's result artifact + W&B run. | §"Question 5: Per-round round_metrics_history schema" — strategy aggregation already emits these keys in all 4 modules; Phase 6 only needs to surface them in the canonical block (no aggregation work). |
| EVL-03 | Per-user and per-group sampling-exposure counts logged each round; reports surface support counts. | §"Question 5" — `evaluated_users_{,sparse,medium,dense}` already emitted by every strategy.aggregate_evaluate. Phase 6 carries them through the canonical block. |
| EVL-04 | Results written to `results/federated/<module>/<run_id>/` with FND-07 manifest; legacy locations untouched. | §"Question 1: Repo-root resolver design" — extend `fedrec_foundation.paths` with `module_run_results_dir(module, run_id)`. §"Question 7: Test surface for path migration" — 4 subprocess-determinism guards need their `_RESULTS_DIR` probes updated. |
| EVL-05 | All cross-device W&B runs log to `federated-cf-cross-device`. | Already wired in every `server_app.py` (lines ~285-302 in baseline, mirrored in others). D-05 says zero churn. |
| EVL-06 | Canonical reporting uses `best_*`; `last_*` is diagnostic only. | §"Question 4: `last_*` block schema" — propose nested structure `final_metrics: {best: {...}, last: {...}, best_round, last_round, final_eval_round_index}`. |
</phase_requirements>

## Standard Stack

### Core (already installed; no new dependencies needed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `flwr` | ≥1.22.0 | `Grid.send_and_receive`, `MessageType="evaluate"`, `ServerApp` | Pinned by project; cannot change. Phase 6 uses the existing eval-broadcast pattern from inside the FL loop, replayed once after best-arrays restore. |
| `torch` | ≥2.7.1 | `ArrayRecord.to_torch_state_dict()` for reading restored params | Already used; no surface change. |
| `wandb` | ≥0.16 (≥0.19 for adaptive) | `wandb.log` per-round + `wandb.run.summary[k]` for final keys | Already wired; only the keying for `best/`/`last/` namespaces is new. |
| `pathlib.Path` | stdlib | All file path work, including the new per-run dir helper | Already used everywhere; no `os.path` introduction. |
| `pytest` | ≥7 | Test runner inside each module's `tests/` (declared in `[project.optional-dependencies]`); `@pytest.mark.slow` + `FEDREC_SKIP_SLOW=1` escape hatch idiom | Same skip-gate convention as existing 4 subprocess determinism guards — copy verbatim. |

### Supporting (already in `fedrec_foundation/`)
| Module | Purpose | Used As |
|--------|---------|---------|
| `fedrec_foundation.paths.repo_root()` | Walk-up to repo root (anchor: `data/ml-1m/` exists) | Foundation for the new `module_run_results_dir(module, run_id)` helper. Existing `data_derived()` is the one-call precedent. |
| `fedrec_foundation.atomic.atomic_write_json` | Atomic JSON write via tempfile + os.replace | Reuse for both `results.json` and `manifest.json` writes inside the per-run dir. The existing manifest-sibling helper already uses this. |
| `fedrec_foundation.manifest.{build_run_manifest, embed_manifest_in_result, write_manifest_sibling, generate_run_id}` | D-15 double-write contract | Phase 6 keeps this contract verbatim. Only the `result_json_path` argument to `write_manifest_sibling` changes (now points inside the per-run dir, so the sibling is `manifest.json` per D-04). |
| `fedrec_foundation.mode.ModeProfile.checkpoint_rule` | `"best_round"` / `"last_round"` (legacy spells `"best_round_restore"`) | Phase 6 reads it in `server_app.py` to gate D-06's extra eval round. Both spellings already accepted by every module — match that pattern. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| New foundation helper `module_run_results_dir(module, run_id)` | Inline `repo_root() / "results" / "federated" / module / run_id` in each `server_app.py` | Inline duplicates four times → drifts. Helper is one place to fix `mkdir(parents=True, exist_ok=True)` consistently. **Choose the helper.** |
| Bump manifest `schema_version` from 1 → 2 | Add fields silently while keeping schema_version=1 | Manifest is an audit artifact; lying about the schema version breaks downstream parsers that branch on it. **Bump to 2; document field additions in a comment block.** |
| Use Flower's `aggregate_evaluate` from inside `strategy.py` for the extra eval round | Inline aggregation in `server_app.py` | Reusing strategy.aggregate_evaluate is the whole point of D-06 — same code path means same metric semantics as in-loop rounds. **Reuse strategy.aggregate_evaluate** (no new aggregation site). |
| `best_*` / `last_*` as flat keys (`final_metrics["best/sampled_ndcg@10"]`) | Nested dict (`final_metrics["best"]["sampled_ndcg@10"]`) | Flat keys keep W&B summary readers simple (no nested dict logic). Nested dict reads cleanly in JSON but breaks `wandb.run.summary[f"final/{key}"] = v` loops. **Use nested dict for the JSON; flatten with a `f"final/best/{key}"` prefix at W&B summary write time.** |

**No new pip installs are required.** Phase 6 only edits existing `server_app.py` files, adds a foundation helper, bumps a schema version, and updates four test files.

## Architecture Patterns

### Recommended Project Structure (no new dirs)

```
scripts/foundation/fedrec_foundation/
├── paths.py            # ADD: module_run_results_dir(module: str, run_id: str) -> Path
└── manifest.py         # MODIFY: bump RUN_MANIFEST_SCHEMA_VERSION = 1 -> 2; add `final_eval_round_index: int` and `metrics: Dict[str, Any]` fields to RunManifest

federated-baseline-cf/federated_baseline_cf/server_app.py    # EDIT: extra-eval-round + per-run dir + best/last block
federated-personalized-cf/federated_personalized_cf/server_app.py  # EDIT: same
federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py  # EDIT: same + alpha_diagnostics sidecar moves into per-run dir
federated-pfedrec/federated_pfedrec/server_app.py            # EDIT: same + D-14 PFR-08 hook reads POST-extra-eval canonical metrics

scripts/foundation/tests/
├── test_adaptive_determinism.py        # EDIT: _RESULTS_DIR probe glob
├── test_personalized_determinism.py    # EDIT: same
├── test_pfedrec_subprocess_determinism.py  # EDIT: same
└── test_baseline_subprocess_determinism.py # NEW or RE-ENABLE: phase2 path bug regression guard

results/federated/                                         # WRITE TARGET (new layout)
├── baseline/
│   └── 20260429-104530-a1b2c3/
│       ├── results.json          # canonical artifact (clean filename per D-04)
│       └── manifest.json         # D-15 sibling (clean filename per D-04)
├── personalized/
│   └── 20260429-104812-d4e5f6/
│       ├── results.json
│       └── manifest.json
├── adaptive/
│   └── 20260429-105002-789abc/
│       ├── results.json
│       ├── manifest.json
│       └── alpha_diagnostics.json   # adaptive-only optional sidecar
└── pfedrec/
    └── 20260429-105230-deadbe/
        ├── results.json
        └── manifest.json
# Pre-Phase-6 flat files (`*_results.json`, `*-manifest.json`) coexist untouched (D-03).
```

### Pattern 1: Repo-Root-Anchored Per-Run Directory (D-01 + D-02)

**What:** A single foundation helper resolves the absolute write path for any module/run_id pair. Every `server_app.py` calls it identically.

**When to use:** Replace every `Path("../results/federated...")` site in the four modules.

**Example:**
```python
# In scripts/foundation/fedrec_foundation/paths.py (NEW helper, ~12 lines)
def module_run_results_dir(module: str, run_id: str) -> Path:
    """Return <repo>/results/federated/<module>/<run_id>/.

    The directory is created (parents=True, exist_ok=True). One directory
    per run; the directory IS the run identifier (D-01).

    Parameters
    ----------
    module : str
        One of "baseline", "personalized", "adaptive", "pfedrec" — matches
        the literal value already passed to build_run_manifest(..., module=...).
    run_id : str
        The same run_id the manifest carries (from generate_run_id()).

    Returns
    -------
    pathlib.Path
        Absolute, resolved path to the per-run directory.
    """
    _ALLOWED_MODULES = frozenset({"baseline", "personalized", "adaptive", "pfedrec"})
    if module not in _ALLOWED_MODULES:
        raise ValueError(
            f"Unknown module {module!r}. Expected one of {sorted(_ALLOWED_MODULES)}."
        )
    out = repo_root() / "results" / "federated" / module / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out

# In every server_app.py (REPLACES `results_dir = Path("../results/federated[/module]")`):
from fedrec_foundation.paths import module_run_results_dir

run_dir = module_run_results_dir(module="baseline", run_id=run_id)
results_filename = run_dir / "results.json"   # D-04 clean filename
# write_manifest_sibling will derive <run_dir>/<run_id>-manifest.json BUT we want manifest.json (D-04).
```

**Note on the manifest sibling filename:** `write_manifest_sibling` currently derives `<parent>/<run_id>-manifest.json` from the result path's parent. D-04 says clean filename `manifest.json`, so the planner has two options:
- (A) Add an optional `sibling_name: Optional[str] = None` kwarg to `write_manifest_sibling` that overrides the default `f"{run_id}-manifest.json"`. Defaults to existing behavior; cross-silo callers unaffected.
- (B) After calling `write_manifest_sibling`, `os.replace` the file to `manifest.json`. Simpler at the call site but introduces a tempfile-rename window.

**Recommendation: Option A** — the helper signature is the contract; threading a kwarg keeps tests deterministic and avoids the rename window.

### Pattern 2: Extra Eval Round After Best-Arrays Restore (D-06)

**What:** After the FL training loop exits and `arrays = best_arrays` runs (and for adaptive, `strategy._global_prototype = strategy.best_prototype`), broadcast one more `@app.evaluate` to all eligible clients. Aggregate via the existing `strategy.aggregate_evaluate`. The result becomes the canonical `best_*` block.

**When to use:** Only when `mode_profile.checkpoint_rule in ("best_round", "best_round_restore")` AND `best_round_num > 0`. Modes like `cross_silo_legacy` (`checkpoint_rule="last_round"`) skip this entirely; they emit `last_*` only.

**Important: which clients participate in the extra eval?**

The simplest choice — and the one that matches D-06's "exactly the restored state" intent — is to broadcast to **every node in the federation** (`fraction_eval=1.0`-style). Three reasons:
1. The whole point is a clean, reproducible canonical block. Sampling a subset reintroduces sampling noise, which is exactly what the per-round in-loop eval already had.
2. Cost is bounded: 6040 client × ~1 eval second each ≈ 1 minute per run. For a multi-round 100-round paper-compat run that's a ~1% overhead. Acceptable per CONTEXT.md "Costs one extra evaluation round per run (~10–60s on 6040 clients)".
3. Strategy aggregation already handles "all clients" in a single `aggregate_evaluate` call without modification.

**Pseudocode skeleton (drop after `arrays = best_arrays` block in every server_app.py):**
```python
# D-06: extra eval round on the restored best-round state.
final_eval_round_metrics: Dict[str, Any] = {}
final_eval_round_index: int = 0   # 0 means "no extra round ran" (manifest sentinel)

if checkpoint_rule in ("best_round_restore", "best_round") and best_round_num > 0:
    final_eval_round_index = actual_rounds + 1   # one past the last training round

    # Broadcast to all 6040 nodes (full federation eval, no sampling).
    eval_node_ids = sorted(partition_to_node_id.values())   # already populated at G-03-01 discovery
    eval_messages = [
        grid.create_message(
            content=RecordDict({"arrays": arrays, "config": ConfigRecord({"lr": lr})}),
            message_type="evaluate",
            dst_node_id=nid,
            group_id=f"final_eval_round_{final_eval_round_index}",
        )
        for nid in eval_node_ids
    ]
    eval_responses = list(grid.send_and_receive(eval_messages))

    eval_results: List[Tuple[ClientProxy, EvaluateRes]] = []
    for response in eval_responses:
        if response.has_error():
            continue
        m = dict(response.content.get("metrics", MetricRecord())) or {}
        num_examples = int(m.get("num_training_examples", m.get("evaluated_users", m.get("num-examples", 1))))
        eval_results.append((
            DummyClientProxy(str(response.metadata.src_node_id)),
            EvaluateRes(
                status=Status(code=Code.OK, message="ok"),
                loss=float(m.get("eval_loss", 0.0)),
                num_examples=num_examples,
                metrics=m,
            ),
        ))
    if eval_results:
        _agg_loss, thesis = strategy.aggregate_evaluate(final_eval_round_index, eval_results, [])
        final_eval_round_metrics = dict(thesis) if thesis else {}

# Diagnostic last_round_metrics: in-loop sufficient stats from the actual final training round.
last_round_metrics = dict(eval_metrics_history.get(actual_rounds, {}))
# If checkpoint_rule disabled the extra eval, the `best_*` block falls back to whatever
# the in-loop best-round had — which under last_round modes equals last_round_metrics by definition.
best_round_metrics = final_eval_round_metrics or dict(eval_metrics_history.get(best_round_num, {}))
```

**Per-module integration points** (where to drop this block):

| Module | Drop site | Notes |
|--------|-----------|-------|
| baseline | After lines 626 (`arrays = best_arrays`) and BEFORE line 635 (`final_model = get_model(...)`). The existing centralized-eval block on RMSE/MAE stays as-is — those are rating-prediction metrics, NOT the thesis sufficient-stat metrics, and they fall under the `last_*` diagnostic per D-07. | Baseline already runs a centralized eval for rating metrics; keep it. The new federated extra-eval populates the thesis `best_*` block. |
| personalized | After line 775 (`arrays = best_arrays`) and BEFORE line 783 (the "Using federated evaluation metrics..." print). Replaces the lookup `final_metrics = dict(eval_metrics_history.get(final_round_for_metrics, {}))` at line 796. | Today personalized SILENTLY pulls `eval_metrics_history[best_round_num]` for `final_metrics`. That's exactly what D-06 forbids. Replace with the broadcast result. |
| adaptive | After line 956 (`strategy._global_prototype = strategy.best_prototype`) and BEFORE the "Using federated evaluation metrics..." print at line 965. Replaces line 978 (`final_metrics = dict(eval_metrics_history.get(final_round_for_metrics, {}))`). | The prototype broadcast must happen FIRST (before extra-eval-round messages go out) so clients see the restored prototype. The existing D-07 ordering already does this — the extra-eval-round just slots after it. |
| pfedrec | After line 901 (`arrays = best_arrays`) and BEFORE line 924 (`final_metrics: Dict[str, Any] = dict(eval_metrics_history.get(...))`). The D-14 PFR-08 auto-verify hook at lines 1062-1067 must read POST-extra-eval `final_metrics` (i.e., `best_round_metrics`). | **D-14 hook ordering is delicate**: today the hook fires on `final_metrics` which comes from `eval_metrics_history[best_round_num]`. After Phase 6, the hook MUST read `best_round_metrics` (extra-eval result). The hook's existing position (after `embed_manifest_in_result`, before W&B summary) is unchanged; only the input dict changes. |

**Crucial subtlety for adaptive:** `strategy._global_prototype` is broadcast to clients via `train_config_dict["global_prototype"] = list(...)` inside the FL loop. The extra eval round must construct its `ConfigRecord` the same way so clients receive the restored prototype. Look at the existing eval-config build site (line 502 in baseline, equivalent in adaptive) — adaptive's eval-config additionally carries `global_prototype` for the same reason. Phase 6 mirrors that.

**Cost note:** A full-federation extra eval (N=6040, ~1s/client serialized via Grid) is acceptable per CONTEXT.md. If a module wants `fraction_eval < 1.0`, that's a future ablation toggle — not Phase 6's concern.

### Pattern 3: `final_metrics` Block Schema (D-07)

**What:** Replace the current flat `final_metrics: Dict[str, Any]` with a structured block carrying `best`, `last`, and bookkeeping.

**Current (across all four server_apps):**
```json
{
  "final_metrics": {
    "sampled_ndcg@10": 0.4413,
    "sampled_hr@10": 0.7287,
    "sampled_ndcg@10/sparse": 0.21,
    "evaluated_users": 6040,
    ...
  }
}
```

**Phase 6 (recommended, nested for the canonical artifact):**
```json
{
  "final_metrics": {
    "best": {
      "sampled_ndcg@10": 0.4413,
      "sampled_hr@10": 0.7287,
      "sampled_ndcg@10/sparse": 0.21, "sampled_ndcg@10/medium": 0.41, "sampled_ndcg@10/dense": 0.51,
      "sampled_hr@10/sparse":   0.65, "sampled_hr@10/medium":   0.78, "sampled_hr@10/dense":   0.85,
      "evaluated_users": 6040,
      "evaluated_users_sparse": 412,
      "evaluated_users_medium": 1230,
      "evaluated_users_dense": 4398
    },
    "last": {
      "sampled_ndcg@10": 0.4321,
      "sampled_hr@10": 0.7102,
      "...same per-group + exposure keys..."
    },
    "best_round": 87,
    "last_round": 100,
    "final_eval_round_index": 101
  }
}
```

**Why nested rather than `final_metrics["best/sampled_ndcg@10"]` flat keys:**
- JSON consumers (Phase 7 thesis tables, ablation aggregators) can `metrics["best"]` and iterate the inner dict mechanically.
- The nested layout makes `last_*` truly diagnostic — readers explicitly reach for it.
- W&B summary write loop becomes:
  ```python
  for key, value in final_metrics["best"].items():
      if isinstance(value, (int, float)):
          wandb.run.summary[f"best/{key}"] = value
  for key, value in final_metrics["last"].items():
      if isinstance(value, (int, float)):
          wandb.run.summary[f"last/{key}"] = value
  ```
  ...which is symmetric and unambiguous.

**Implication for existing W&B summary readers:** Today every server_app does `wandb.run.summary[f"final/{key}"] = value`. After Phase 6 this changes to `best/` and `last/` namespaces (with `final/` going away). **The adaptive sweep.yaml metric optimizer key (`final/sampled_ndcg@10` at sweep.yaml:18) MUST be migrated** — see §"Question 8" for the precise change. Recommend a one-line Phase 6 task: bump sweep.yaml's `metric.name` from `final/sampled_ndcg@10` → `best/sampled_ndcg@10`.

### Pattern 4: Manifest Schema Bump (1 → 2)

**What:** Add `final_eval_round_index: int` and `metrics: Dict[str, Any]` fields to `RunManifest`. Bump `RUN_MANIFEST_SCHEMA_VERSION` from 1 to 2.

**Why bump:** The four existing subprocess determinism guards do dict-equality checks on `_manifest.pfr08_verification` (pfedrec) and `_manifest.best_prototype` (adaptive). Adding fields to a v1 manifest while keeping schema_version=1 creates a silent contract break. A version bump is mechanical (it's a single line in `manifest.py`) and self-documenting.

**Backward compatibility for existing readers:**
- `RunManifest` is a dataclass; adding required fields would break `from_dict`-style deserialization. **Workaround: add new fields with safe defaults.**
- The four subprocess determinism tests do `bp_a == bp_b` (dict equality). Adding an immutable `metrics` block that's deterministic across reruns (because it's derived from sufficient stats which are themselves byte-identical under FND-06 RNG) keeps that invariant. **Recommendation: add a Phase 6 invariant test that asserts `manifest_a["metrics"] == manifest_b["metrics"]` across two same-seed runs.**

**Minimal change to `RunManifest`:**
```python
RUN_MANIFEST_SCHEMA_VERSION: int = 2   # was 1; bump per Phase 6 D-07 + final_eval_round_index addition

@dataclass
class RunManifest:
    schema_version: int
    run_id: str
    # ... existing fields unchanged ...
    git_commit: str
    # NEW (Phase 6):
    final_eval_round_index: int = 0   # 0 sentinel = no extra eval ran (last_round modes); >=1 = round index of the post-restore broadcast
    metrics: Dict[str, Any] = field(default_factory=dict)   # mirrors final_metrics block (best + last + bookkeeping)
```

**Note:** `field(default_factory=dict)` requires `from dataclasses import field`. Adding a default to `final_eval_round_index` keeps `build_run_manifest` calls in older test stubs working.

### Anti-Patterns to Avoid

- **Don't write the result JSON via `json.dump` directly.** Use `atomic_write_json` from `fedrec_foundation.atomic`. The current four server_apps use `json.dump`; Phase 6 should standardize on atomic_write to match the manifest write path. This avoids partial writes if the process is killed mid-write.
- **Don't read `final_metrics` from `eval_metrics_history[best_round_num]`.** That's the bug D-06 is fixing. The canonical block must come from the broadcast.
- **Don't hard-code `module="baseline"` strings in 4 places.** The literal already lives in each module's `build_run_manifest(..., module=...)` call site — extract it to a single local variable per module so it's used by both `module_run_results_dir(module=..., run_id=run_id)` and `build_run_manifest`.
- **Don't change `strategy.aggregate_evaluate` signatures.** Phase 6 reuses them verbatim. Touching `strategy.py` would invalidate the per-module sufficient-stat tests; leave the strategy frozen.
- **Don't break the D-14 PFR-08 hook ordering.** It must fire AFTER `embed_manifest_in_result` (so the audit dict can be injected) AND BEFORE the W&B summary write (so failure surfaces in W&B). Phase 6 keeps that order; only the *input* (`final_metrics` dict it reads from) changes.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Repo-root resolution from a Flower subprocess | `Path(__file__).resolve().parents[N]` walks | `fedrec_foundation.paths.repo_root()` | Already exists, anchored on `data/ml-1m/` existence (more robust than parent-counting; Flower may chdir). |
| Per-run results directory creation | Manual `Path("../results/federated") / module / run_id` per module | `module_run_results_dir(module, run_id)` (NEW one-liner helper in foundation) | DRY — four duplicate sites today. Helper centralizes mkdir, validation, and future schema changes. |
| Atomic JSON write | `json.dump(...)` then hope the process doesn't crash | `fedrec_foundation.atomic.atomic_write_json` | Tempfile + os.replace pattern; manifest sibling already uses it. Apply same to `results.json`. |
| Aggregating per-group metrics across clients | Build a per-server custom summer | Reuse `strategy.aggregate_evaluate` from each module's existing `_sum_sufficient_stats` + `_sufficient_stats_to_thesis_metrics` | Already correct; the D-22 fit-metrics contract guarantees the wire payload carries the keys. Phase 6's extra eval round wraps responses into `EvaluateRes` and calls the same strategy method. |
| Per-round W&B logging of exposure counts | A new `wandb.log({"exposure/sparse": ...})` block | The existing per-round `eval_metrics_history[round_num]` already includes `evaluated_users_sparse` / `_medium` / `_dense` from `_sufficient_stats_to_thesis_metrics`. The existing wandb.log loop iterates this dict. | Already works; D-09's per-round emission is satisfied as-is. Only the per-round W&B key prefix may need normalization (currently `eval/evaluated_users_sparse`; some readers may expect `exposure/sparse`). Recommend keeping the current `eval/` prefix for zero churn. |
| Best-round snapshot/restore plumbing | New `_BestRoundCheckpoint` class | The existing `best_arrays = ArrayRecord({...})` snapshot in D-27 + adaptive's `best_prototype` in D-05 is sufficient for Phase 6. | Phase 6 adds NO new snapshotted state. The "extra eval round" reuses `arrays = best_arrays` and `strategy._global_prototype = strategy.best_prototype`. |
| Filename encoding of `best_round` | Glob-friendly `f"{run_id}_best_round_{N}_results.json"` | `manifest.json` carries `best_round` as a typed field (D-04 + D-07) | Filenames-as-data is brittle (parsing reverses). Manifest field is the single source of truth. |

**Key insight:** Phase 6 is intentionally a **plumbing phase**. Every algorithmic primitive it needs (best-round snapshot, sufficient-stat aggregation, manifest fingerprint, atomic writes, repo-root walk-up) already exists in the codebase from Phases 1–5. The risk is not "we need new algorithms" — it's "we need to wire four nearly-identical edits without breaking 100+ existing tests across four module suites and one foundation suite". Surgical edits + lockstep test updates is the discipline.

## Common Pitfalls

### Pitfall 1: D-14 PFR-08 hook reads stale `final_metrics`
**What goes wrong:** Today pfedrec's `_emit_pfr_08_verification` (line 1062) reads `final_metrics` populated from `eval_metrics_history[best_round_num]`. After Phase 6, `final_metrics` becomes a nested `{best: {...}, last: {...}}` block — the hook will silently read `final_metrics["sampled_ndcg@10"]` and get `None`, marking the run as "PFR-08 FAILED" for purely structural reasons.
**Why it happens:** Schema change without updating the consumer.
**How to avoid:** When restructuring `final_metrics` to nested form, update `_emit_pfr_08_verification(final_metrics, ...)` call site to pass `final_metrics["best"]` instead. Add a Phase 6 task: "rewire D-14 PFR-08 hook to consume `best_round_metrics` (the post-extra-eval canonical dict)". Verify with the existing `test_pfr08_autoverify_pass_within_2pts` synthetic test by switching its synthetic input to the new schema.
**Warning signs:** First post-Phase-6 pfedrec run logs `[PFR-08 FAILED]` with `delta_hr_pts=nan delta_ndcg_pts=nan`.

### Pitfall 2: Module-relative path passes to `flwr run` from CWD other than the module dir
**What goes wrong:** The four `_RESULTS_DIR = _REPO_ROOT / "results" / "federated"` probes in subprocess determinism tests assume the launcher writes inside `repo_root()/results`. Today only the personalized + adaptive tests actually find files there because they `rglob("*_results.json")`. Baseline currently writes to `<repo_root>/../results/federated/` (one level above the repo root) — outside the test's probe — which is exactly the folded `phase2-baseline-determinism-path-bug.md` bug.
**Why it happens:** `Path("../results/federated")` resolves relative to CWD at write time. When `flwr run .` is called inside `federated-baseline-cf/`, that's `<repo>/federated-baseline-cf/../results/federated/` = `<repo>/results/federated/` (correct). When `scripts/run.py` is called from `<repo>/`, that's `<repo>/../results/federated/` = `<repo's_parent>/results/federated/` (WRONG, above the repo root).
**How to avoid:** D-02 mandates the foundation helper. Use `module_run_results_dir(module, run_id)` everywhere; never use module-relative paths.
**Warning signs:** A baseline run completes successfully but the result JSON is written to a path outside the repo (and the subprocess determinism test pytest.skip's because it can't find the artifact). The fix lands the missing path-bug todo at the same time.

### Pitfall 3: Manifest schema bump breaks existing test fixtures
**What goes wrong:** `scripts/foundation/tests/test_manifest.py` and the four module-level `test_server_integration.py` tests construct `RunManifest` instances directly. Adding required fields (without defaults) breaks all of them.
**Why it happens:** Dataclass positional/keyword ordering.
**How to avoid:** Add new fields with safe defaults: `final_eval_round_index: int = 0`, `metrics: Dict[str, Any] = field(default_factory=dict)`. Existing test fixtures pass through unchanged. Phase 6 plan should explicitly enumerate the test files that touch RunManifest construction so the planner can verify fixtures one-by-one.
**Warning signs:** `TypeError: __init__() missing 2 required positional arguments` in foundation test suite.

### Pitfall 4: Adaptive's `strategy._global_prototype` not in scope at extra-eval-round
**What goes wrong:** Adaptive's eval-config inside the FL loop attaches `global_prototype` to the ConfigRecord (so clients can blend `p_global` into `p_effective` during eval). The extra eval round must do the same; if it builds an eval ConfigRecord without `global_prototype`, every client falls back to a zero or stale prototype and the canonical metrics are wrong.
**Why it happens:** Copy-pasting the pattern from baseline (which has no prototype) into adaptive without including the prototype attachment.
**How to avoid:** When writing the extra-eval-round block, **first read each module's existing in-loop eval ConfigRecord build site** and replicate it verbatim. For adaptive, that includes `train_config["global_prototype"] = strategy._global_prototype.tolist()` (already restored to `best_prototype` per D-07).
**Warning signs:** Adaptive's `best_*` block reports lower NDCG@10 than the in-loop best_round_num round did — a contradiction that signals the prototype was missing from the broadcast.

### Pitfall 5: `selected_clients_per_round` byte-identity test breaks under extra-eval-round
**What goes wrong:** The four subprocess determinism tests assert `result_a["selected_clients_per_round"] == result_b["selected_clients_per_round"]`. If the extra eval round uses `_server_sampler.sample(...)` (consuming RNG state), and the sampler is then NOT reset between runs, the saved `selected_clients_per_round` would shift. But: the extra eval round does NOT sample — it broadcasts to ALL nodes. So this is a non-issue **as long as** the planner specifies "no sampling, broadcast to all eligible nodes".
**Why it happens:** Defensively assuming the extra eval also samples.
**How to avoid:** The planner must specify the extra eval is full-federation (`fraction_eval=1.0`-equivalent). Document this in PLAN.md as a non-negotiable.
**Warning signs:** All four subprocess determinism tests start failing with mysterious `selected_clients_per_round` mismatches.

### Pitfall 6: `module_run_results_dir` and module string drift between server_app and helper
**What goes wrong:** `build_run_manifest(..., module="baseline")` and `module_run_results_dir(module="basline", ...)` (typo). Today the literal `"baseline"` lives in only one place per server_app; adding a second use-site doubles the typo surface.
**How to avoid:** Extract the module name to a local constant at the top of each `@app.main()`, e.g.,:
```python
_MODULE: str = "baseline"   # cross-references: build_run_manifest, module_run_results_dir, default W&B project switch
```
Then use `_MODULE` in both `module_run_results_dir(_MODULE, run_id)` and `build_run_manifest(..., module=_MODULE)`. The `module_run_results_dir` helper's whitelist (`_ALLOWED_MODULES = frozenset({...})`) catches typos at runtime.
**Warning signs:** Run completes but results land in `/results/federated/basline/<run_id>/`.

### Pitfall 7: Sweep.yaml metric optimizer key change breaks active sweeps
**What goes wrong:** The `federated-adaptive-personalized-cf/sweep.yaml` currently optimizes `final/sampled_ndcg@10`. After Phase 6 the W&B summary key becomes `best/sampled_ndcg@10`. Active wandb agents pulling from the sweep config will start reporting NaN for the metric.
**How to avoid:** Migrate sweep.yaml as a Phase 6 sub-task. Bump `metric.name` from `final/sampled_ndcg@10` → `best/sampled_ndcg@10`. Communicate this in the plan.
**Warning signs:** Bayesian sweep stops converging; agents log "metric not found" warnings.

### Pitfall 8: Cross-silo runs trip the new path layout
**What goes wrong:** `cross_silo_legacy` mode has `checkpoint_rule="last_round"`. Phase 6's per-run dir + extra-eval-round logic should NOT activate for cross-silo runs (D-03 requires legacy paths untouched). If `module_run_results_dir` is called unconditionally, cross-silo runs start writing to the new layout.
**How to avoid:** The path resolution for cross-silo can stay on the legacy flat layout (`<repo>/results/federated/<run_id>_results.json`). Add a one-line branch:
```python
if mode in ("benchmark_cross_device", "paper_compat_pfedrec"):
    run_dir = module_run_results_dir(_MODULE, run_id)
    results_filename = run_dir / "results.json"
else:  # cross_silo_legacy
    results_dir = repo_root() / "results" / "federated"  # legacy flat layout per D-03
    results_dir.mkdir(parents=True, exist_ok=True)
    results_filename = results_dir / f"{run_id}_results.json"
```
The extra-eval-round is also gated by `checkpoint_rule == "best_round[_restore]"` so cross-silo (last_round) skips it naturally.
**Warning signs:** Pre-Phase-6 cross-silo dashboards start sprouting `results/federated/<module>/<run_id>/` directories that didn't exist before.

### Pitfall 9: `last_round_metrics` is empty when the FL loop exits early (early stopping fires)
**What goes wrong:** `eval_metrics_history.get(actual_rounds, {})` returns `{}` if `actual_rounds` was set to the early-stopper's stop round but no eval ran on that exact round (edge case if early stopping triggers before the eval block on a given round).
**How to avoid:** Define `last_round` as `max(eval_metrics_history.keys())` rather than `actual_rounds`. This guarantees a real eval result exists for that round.
```python
if eval_metrics_history:
    last_round = max(eval_metrics_history.keys())
    last_round_metrics = dict(eval_metrics_history[last_round])
else:
    last_round = 0
    last_round_metrics = {}
```
**Warning signs:** Result JSON has `final_metrics["last"] = {}` and `final_metrics["last_round"] = N` where N is `actual_rounds` but `eval_metrics_history` doesn't have key N.

### Pitfall 10: Existing centralized eval in baseline computes RMSE/MAE, NOT thesis sufficient-stat metrics
**What goes wrong:** Baseline's `server_app.py` lines 632–700 run `evaluate_ranking` and `evaluate_ranking_sampled` centrally — but those write into the same `final_metrics` dict that today gets the `final/sampled_ndcg@10` key. After Phase 6 the centralized eval's outputs go into `final_metrics["last"]` (rating-prediction diagnostics) — but the thesis sufficient-stat metric must come from the federated extra-eval-round.
**How to avoid:** The centralized eval results (`rmse`, `mae`, `ranking_metrics`, `sampled_metrics`) become `last`-block diagnostics. The federated extra-eval-round populates `best`. Be explicit about which dict each metric source feeds.
**Warning signs:** Baseline reports two NDCG@10 numbers — one from centralized full-rank evaluator and one from federated sufficient-stat. The thesis table must use the federated one.

## Code Examples

### Example 1: Foundation helper (NEW)
```python
# scripts/foundation/fedrec_foundation/paths.py — append after data_derived/ml1m_dir.
# Source: existing repo_root() at paths.py:16; mirrors data_derived() shape.

_ALLOWED_MODULES = frozenset({"baseline", "personalized", "adaptive", "pfedrec"})

def module_run_results_dir(module: str, run_id: str) -> Path:
    """Return the per-run output dir, creating it if needed (D-01 + D-02).

    Layout: ``<repo>/results/federated/<module>/<run_id>/``. The directory IS
    the run identifier (one directory per run). Inside it the canonical
    artifacts are ``results.json`` + ``manifest.json`` (D-04 clean filenames).

    Parameters
    ----------
    module : str
        One of ``baseline`` / ``personalized`` / ``adaptive`` / ``pfedrec``.
    run_id : str
        Same string as the ``RunManifest.run_id`` field (from generate_run_id).

    Returns
    -------
    pathlib.Path
        Absolute path to the per-run directory.

    Raises
    ------
    ValueError
        If ``module`` is not in the allowed-modules whitelist.
    """
    if module not in _ALLOWED_MODULES:
        raise ValueError(
            f"Unknown module {module!r}. Expected one of {sorted(_ALLOWED_MODULES)}."
        )
    out = repo_root() / "results" / "federated" / module / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out
```

### Example 2: Manifest schema bump (MODIFY)
```python
# scripts/foundation/fedrec_foundation/manifest.py — diff-style.
# Source: existing manifest.py at lines 28-84.

from dataclasses import asdict, dataclass, field   # ADD `field` to existing imports
from typing import Any, Dict

# Bump from 1 to 2 (Phase 6 adds final_eval_round_index + metrics fields).
RUN_MANIFEST_SCHEMA_VERSION: int = 2   # was 1

@dataclass
class RunManifest:
    schema_version: int
    run_id: str
    # ... 22 existing fields unchanged ...
    git_commit: str
    # NEW (Phase 6 — both with safe defaults so existing test fixtures still construct):
    final_eval_round_index: int = 0   # 0 = no extra eval ran (last_round modes); >=1 = post-restore broadcast index
    metrics: Dict[str, Any] = field(default_factory=dict)   # mirrors result JSON `final_metrics` block
```

### Example 3: Per-run dir + atomic write in server_app.py (REPLACE the four legacy sites)
```python
# Replaces e.g. federated-baseline-cf/federated_baseline_cf/server_app.py:786-794.
# Mirror in personalized (lines 895-906), adaptive (lines 1181-1188), pfedrec (lines 1071-1077).
# Source: existing manifest.py write_manifest_sibling pattern + atomic.atomic_write_json.

from fedrec_foundation.paths import module_run_results_dir
from fedrec_foundation.atomic import atomic_write_json

_MODULE: str = "baseline"   # extract once; used by both module_run_results_dir AND build_run_manifest

# Build manifest (existing call, unchanged signature aside from new fields filled in below).
manifest = build_run_manifest(
    run_id=run_id,
    mode_profile=profile,
    run_seed=run_seed,
    mapping_sha256=foundation_idx.mapping_sha256,
    split_hash=foundation_idx.split_hash,
    exclusion_sha256=foundation_idx.exclusion_sha256,
    foundation_contract_sha256=foundation_idx.foundation_contract_sha256,
    raw_data_hash=split_manifest.raw_data_hash,
    builder_version=split_manifest.builder_version,
    overrides=overrides,
    module=_MODULE,
)
# Phase-6 additions to the manifest (post-build mutation pattern, same as adaptive's best_prototype):
manifest = replace(   # dataclasses.replace → returns a fresh RunManifest with new fields populated
    manifest,
    final_eval_round_index=final_eval_round_index,   # 0 if checkpoint_rule != best_round
    metrics=results_data["final_metrics"],            # nested {best, last, best_round, last_round, final_eval_round_index}
)
embed_manifest_in_result(manifest, results_data)

# D-04 clean per-run dir + filenames.
run_dir = module_run_results_dir(_MODULE, run_id)
results_filename = run_dir / "results.json"

atomic_write_json(str(results_filename), results_data)

# D-15 sibling. Pass sibling_name="manifest.json" to override the default <run_id>-manifest.json.
sibling_path = write_manifest_sibling(manifest, results_filename, sibling_name="manifest.json")
print(f"Results saved to: {results_filename.resolve()}")
print(f"Manifest sibling: {sibling_path.resolve()}")
```

### Example 4: Extra-eval-round block (DROP after `arrays = best_arrays` in every server_app)
```python
# Source: existing intra-loop eval pattern (baseline server_app.py lines 500-547).

# D-06: extra eval round. Gated on best_round_num > 0 (zero means we never recorded a best round —
# happens for last_round modes or pathological 1-round runs).
final_eval_round_index: int = 0
best_round_metrics: Dict[str, Any] = {}

if checkpoint_rule in ("best_round_restore", "best_round") and best_round_num > 0:
    final_eval_round_index = actual_rounds + 1
    print(f"\n[D-06] Broadcasting extra eval round {final_eval_round_index} on restored best-round state...")

    # All-nodes broadcast (no sampling — reproducibility > latency per D-06).
    eval_node_ids = sorted(partition_to_node_id.values())
    extra_eval_messages = []
    for nid in eval_node_ids:
        # Build eval-config the SAME way as the in-loop eval.
        # For adaptive: include strategy._global_prototype (already restored to best_prototype).
        eval_config = ConfigRecord({"lr": lr})
        # ===== ADAPTIVE-ONLY ADDITION =====
        # if strategy._global_prototype is not None:
        #     eval_config["global_prototype"] = strategy._global_prototype.tolist()
        content = RecordDict({"arrays": arrays, "config": eval_config})
        extra_eval_messages.append(grid.create_message(
            content=content, message_type="evaluate", dst_node_id=nid,
            group_id=f"final_eval_round_{final_eval_round_index}",
        ))
    extra_eval_responses = list(grid.send_and_receive(extra_eval_messages))

    extra_results: List[Tuple[ClientProxy, EvaluateRes]] = []
    for response in extra_eval_responses:
        if response.has_error():
            continue
        m = dict(response.content.get("metrics", MetricRecord()))
        num_examples = int(m.get("num_training_examples", m.get("evaluated_users", m.get("num-examples", 1))))
        extra_results.append((
            DummyClientProxy(str(response.metadata.src_node_id)),
            EvaluateRes(
                status=Status(code=Code.OK, message="ok"),
                loss=float(m.get("eval_loss", 0.0)),
                num_examples=num_examples,
                metrics=m,
            ),
        ))
    if extra_results:
        _agg_loss, thesis = strategy.aggregate_evaluate(final_eval_round_index, extra_results, [])
        best_round_metrics = dict(thesis) if thesis else {}
        print(f"[D-06] Extra eval round complete. Canonical sampled_ndcg@10={best_round_metrics.get('sampled_ndcg@10')}")

# Always populate last_round (diagnostic per D-07). Use max-key, NOT actual_rounds, to dodge
# the early-stopping pitfall (Pitfall 9).
if eval_metrics_history:
    last_round = max(eval_metrics_history.keys())
    last_round_metrics = dict(eval_metrics_history[last_round])
else:
    last_round = 0
    last_round_metrics = {}

# D-07 nested final_metrics block.
final_metrics = {
    "best": best_round_metrics or last_round_metrics,   # for last_round modes, best collapses to last
    "last": last_round_metrics,
    "best_round": best_round_num if best_round_num > 0 else last_round,
    "last_round": last_round,
    "final_eval_round_index": final_eval_round_index,
}
```

### Example 5: W&B summary write under the new schema
```python
# Source: existing `wandb.run.summary[f"final/{key}"] = value` loop at server_app.py:721-723 (baseline).

if wandb_enabled and wandb_run is not None:
    # best/* (canonical thesis metrics).
    for key, value in final_metrics["best"].items():
        if isinstance(value, (int, float)):
            wandb.run.summary[f"best/{key}"] = value
    # last/* (diagnostic — overfitting / late-round drift detection).
    for key, value in final_metrics["last"].items():
        if isinstance(value, (int, float)):
            wandb.run.summary[f"last/{key}"] = value
    # Bookkeeping.
    wandb.run.summary["best_round"] = final_metrics["best_round"]
    wandb.run.summary["last_round"] = final_metrics["last_round"]
    wandb.run.summary["final_eval_round_index"] = final_metrics["final_eval_round_index"]
```

## State of the Art

| Old Approach (pre-Phase-6) | Current Approach (Phase 6) | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `final_metrics = dict(eval_metrics_history.get(best_round_num, {}))` | Federated extra-eval-round broadcasts to all 6040 clients on restored state; `final_metrics["best"]` from `strategy.aggregate_evaluate` of those responses | Phase 6 | Canonical metric matches the actually-restored state, not stale per-round sufficient stats. Cost: ~10–60s extra per run. |
| `Path("../results/federated/<module>")` (module-relative; CWD-dependent) | `module_run_results_dir(_MODULE, run_id)` (repo-root-anchored via `repo_root()` walk-up) | Phase 6 (D-02) | Tests find the artifact regardless of where launcher runs from. Closes folded `phase2-baseline-determinism-path-bug.md`. |
| Flat `{run_id}_results.json` + `{run_id}-manifest.json` siblings in shared dir | Per-run directory `<module>/<run_id>/{results.json, manifest.json}` | Phase 6 (D-01 + D-04) | Co-location is unambiguous; future sidecars (`alpha_diagnostics.json`) drop into the same dir without filename collision. |
| `wandb.run.summary[f"final/{k}"] = v` | `wandb.run.summary[f"best/{k}"] = v` and `wandb.run.summary[f"last/{k}"] = v` | Phase 6 (D-07) | Sweep optimizers (e.g. `final/sampled_ndcg@10` in adaptive's sweep.yaml:18) MUST migrate to `best/sampled_ndcg@10`. Schedule in plan. |
| `RunManifest` schema_version=1 | schema_version=2 with `final_eval_round_index`, `metrics` fields (defaults; backward-readable) | Phase 6 | Subprocess determinism tests now also assert `manifest["metrics"]` byte-identity. |

**Deprecated/outdated (within this repo):**
- Reading `final_metrics` from `eval_metrics_history[best_round_num]` (personalized, adaptive, pfedrec server_apps) — keep as a fallback when `checkpoint_rule != "best_round[_restore]"` only.
- The `final/*` W&B summary namespace — migrating to `best/*` and `last/*`.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `pytest >=7.0` (declared in each module's `[project.optional-dependencies] dev`) |
| Config file | None at repo root; per-module `tests/conftest.py` exists (e.g., `federated-baseline-cf/tests/conftest.py`); `scripts/foundation/tests/conftest.py` exists. No `pyproject.toml [tool.pytest.ini_options]` block. |
| Quick run command | `pytest scripts/foundation/tests/test_manifest.py federated-baseline-cf/tests/test_server_integration.py` — runs <30s |
| Full suite command | `pytest scripts/foundation/tests/ federated-*/tests/ -m "not slow"` (skips subprocess determinism guards which can take 5-15min); add `-m slow` (and unset `FEDREC_SKIP_SLOW`) to include them. |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| EVL-01 | Best-round restore: canonical `best_*` block comes from a fresh broadcast on restored arrays (NOT `eval_metrics_history[best_round_num]`) | unit + integration | `pytest -k test_extra_eval_round_uses_restored_arrays scripts/foundation/tests/ federated-*/tests/` | NEW file per module (planner adds: `tests/test_extra_eval_integration.py` × 4) |
| EVL-01 | Manifest carries `final_eval_round_index >= 1` for `best_round_restore` modes; `== 0` for `last_round` modes | unit | `pytest -k test_manifest_final_eval_round_index scripts/foundation/tests/test_manifest.py` | NEW assertion in existing `scripts/foundation/tests/test_manifest.py` |
| EVL-02 | `final_metrics["best"]` contains `sampled_ndcg@10/sparse`, `/medium`, `/dense` and `sampled_hr@10/{sparse,medium,dense}` | unit | `pytest -k test_per_group_keys_in_best_block federated-*/tests/test_server_integration.py` | NEW assertion in existing 4 `tests/test_server_integration.py` files |
| EVL-03 | `final_metrics["best"]` carries `evaluated_users_{,sparse,medium,dense}`; eval_metrics_history per-round entries also carry these | unit | `pytest -k test_exposure_counts_per_round_and_canonical federated-*/tests/test_server_integration.py` | NEW assertion in existing 4 test files |
| EVL-04 | Result JSON path is `<repo>/results/federated/<module>/<run_id>/results.json`; manifest sibling is `manifest.json`; cross-silo modes still write the legacy flat path | unit | `pytest scripts/foundation/tests/test_paths.py` | NEW file: `scripts/foundation/tests/test_paths.py` |
| EVL-04 | Subprocess byte-identity invariants survive the path migration (selected_clients_per_round, partition_*.pt, manifest.best_prototype, manifest.pfr08_verification) | integration (slow) | `pytest -m slow scripts/foundation/tests/test_*_subprocess_determinism.py` | EXISTING — UPDATE `_RESULTS_DIR` probes to find files in the new layout |
| EVL-05 | `wandb.init(project=...)` resolves to `federated-cf-cross-device` for benchmark_cross_device + paper_compat_pfedrec, regardless of module | unit | `pytest -k test_wandb_project_routing federated-*/tests/test_server_integration.py` | EXISTING — add a one-liner assertion (each module already has wandb-project tests). |
| EVL-06 | `final_metrics["last"]` is preserved as a diagnostic; nested schema (`{best, last, best_round, last_round, final_eval_round_index}`) is byte-identical across same-seed reruns | integration (slow) | `pytest -m slow -k test_final_metrics_schema_byte_identical scripts/foundation/tests/test_*_subprocess_determinism.py` | EXISTING — extend the 4 subprocess determinism guards with the schema-byte-identity assertion |

**Manual-only checks** (justified — require full-scale 6040-client run):
- "Adaptive `best_*` NDCG@10 ≥ in-loop `eval_metrics_history[best_round_num]['sampled_ndcg@10']`" — Verified by hand on the first full Phase 6 + paper_compat_pfedrec run; recorded in `.planning/phases/06-evaluation-reporting-harness/06-VALIDATION.md` as a one-time observation, not a regression test (smoke at full scale would dominate CI cost).

### Sampling Rate
- **Per task commit:** `pytest scripts/foundation/tests/test_paths.py scripts/foundation/tests/test_manifest.py federated-baseline-cf/tests/test_server_integration.py` (~10s)
- **Per wave merge:** `pytest scripts/foundation/tests/ federated-*/tests/ -m "not slow"` (~60s)
- **Phase gate:** Full suite green INCLUDING `pytest -m slow` against all 4 subprocess determinism guards (~15min on warm cache; can be parallelized with `-n auto` if pytest-xdist is added — currently not declared, leave as serial for Phase 6).

### Wave 0 Gaps
- [ ] `scripts/foundation/tests/test_paths.py` — covers EVL-04 (new helper unit tests: whitelist enforcement, mkdir, repo-root anchoring under chdir)
- [ ] `federated-baseline-cf/tests/test_server_integration.py` — extend with `test_extra_eval_round_*` and `test_per_group_keys_in_best_block` (REQ EVL-01, EVL-02)
- [ ] `federated-personalized-cf/tests/test_server_integration.py` — same extensions (NEW file? Verify existence: see Q7 below)
- [ ] `federated-adaptive-personalized-cf/tests/test_server_integration.py` — same extensions + an adaptive-specific assertion that `eval_config["global_prototype"]` is attached to the extra-eval-round broadcast
- [ ] `federated-pfedrec/tests/test_server_integration.py` — same + assert D-14 PFR-08 hook reads `final_metrics["best"]` (Pitfall 1 regression guard)
- [ ] Existing `scripts/foundation/tests/test_*_subprocess_determinism.py` × 4 — UPDATE `_RESULTS_DIR.glob(...)` probes from flat `*_results.json` to per-run `*/results.json`
- [ ] NEW `scripts/foundation/tests/test_baseline_subprocess_determinism.py` — re-enable the folded path-bug regression guard once D-02 lands (the original `test_selected_partitions_byte_identical_across_subprocess_reruns` from Phase 2 Plan 5)

*(All gaps surface in Wave 0 of the plan; all four module suites already have a `tests/` directory with `conftest.py`.)*

## Open Questions

1. **Should `module_run_results_dir` whitelist module names, or accept any string?**
   - What we know: All four current `build_run_manifest(..., module=...)` call sites pass a literal from the set `{"baseline", "personalized", "adaptive", "pfedrec"}`. A typo in one of these literal strings is a real risk.
   - What's unclear: Whether to enforce the whitelist or stay permissive (in case Phase 7 introduces a new module).
   - Recommendation: **Whitelist** for Phase 6. Phase 7 doesn't add new modules (only thesis-eval scripts); the whitelist is a cheap typo-catcher. Phase 8+ can grow the set.

2. **Should the extra eval round broadcast to ALL nodes or sample per `fraction_eval`?**
   - What we know: D-06 says "broadcast one extra `@app.evaluate` round". CONTEXT.md latency budget assumes ~1 minute on 6040 clients (full federation).
   - What's unclear: Whether reusing `fraction_eval` would let users opt into a faster (sampled) canonical eval.
   - Recommendation: **All nodes**, no sampling. Sampling reintroduces the noise the canonical block is meant to eliminate. Document this choice as a Phase 6 invariant in PLAN.md.

3. **What happens to `final_metrics` for `cross_silo_legacy` runs (which have `checkpoint_rule="last_round"`)?**
   - What we know: `last_round` modes never run the extra eval round. The natural fallback is `final_metrics["best"] = final_metrics["last"]` (collapse).
   - What's unclear: Whether `final_metrics["best"]` should be `None` instead, signaling "no canonical block" for legacy runs.
   - Recommendation: **Collapse**. Setting `best == last` keeps downstream JSON parsers' branching simple (always read `best`). Documents this in the manifest comment block + plan VALIDATION.md.

4. **Should the wandb sweep.yaml metric.name change be a Phase 6 task or deferred?**
   - What we know: The metric currently reads `final/sampled_ndcg@10`. After Phase 6 the W&B summary key becomes `best/sampled_ndcg@10`. Active sweeps (none currently mid-flight per STATE.md) would silently break.
   - What's unclear: Whether Phase 7 is the better place to migrate (since Phase 7 owns the thesis sweep config).
   - Recommendation: **Bump in Phase 6**. The migration is one line; deferring creates a Phase-7 surprise. Add a Phase 6 task: "update `federated-adaptive-personalized-cf/sweep.yaml` metric.name from `final/sampled_ndcg@10` to `best/sampled_ndcg@10`".

5. **Should we keep `final_metrics` as the top-level key, or rename to `metrics`?**
   - What we know: Today every module's result JSON has `final_metrics`. Phase 7 + manifest both refer to it by that name.
   - What's unclear: Whether the nested `{best, last, ...}` block warrants a rename.
   - Recommendation: **Keep `final_metrics`**. Renaming forces every downstream reader to update; the schema change is already disruptive enough. The nested layout is sufficient signal that the structure has evolved.

## Sources

### Primary (HIGH confidence — in-repo source code)
- `scripts/foundation/fedrec_foundation/paths.py:16` — `repo_root()` walk-up function (anchor: `data/ml-1m/` exists). Phase 6's new helper extends this.
- `scripts/foundation/fedrec_foundation/manifest.py:29-84` — `RunManifest` dataclass + `RUN_MANIFEST_SCHEMA_VERSION`. Phase 6 bumps version, adds two fields.
- `scripts/foundation/fedrec_foundation/manifest.py:203-226` — `write_manifest_sibling`. Phase 6 adds an optional `sibling_name` kwarg.
- `scripts/foundation/fedrec_foundation/atomic.py` — `atomic_write_json` (already imported by `manifest.py`). Phase 6 reuses for `results.json`.
- `scripts/foundation/fedrec_foundation/mode.py:80-103` — `ModeProfile.checkpoint_rule` field. Gates the extra-eval-round.
- `federated-baseline-cf/federated_baseline_cf/server_app.py:786-794` — current `Path("../results/federated")` write site (D-02 target). Plus lines 617-628 (D-27 best_arrays restore) and lines 500-547 (intra-loop eval pattern).
- `federated-baseline-cf/federated_baseline_cf/strategy.py:27-104` — Per-group sufficient-stat aggregation. Phase 6 reuses.
- `federated-personalized-cf/federated_personalized_cf/server_app.py:770-799, 895-906` — Today silently looks up `eval_metrics_history[best_round_num]` for `final_metrics`. Phase 6 replaces with extra-eval-round.
- `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py:947-958, 978, 1183-1188` — D-07 prototype paired restore + `eval_metrics_history` lookup + `../results/federated/adaptive` path.
- `federated-pfedrec/federated_pfedrec/server_app.py:894-901, 924-925, 1062-1067, 1073-1077` — D-13 best-round restore + `eval_metrics_history` lookup + D-14 PFR-08 hook (must rewire to consume `best_round_metrics`).
- `scripts/foundation/tests/test_personalized_determinism.py:47-118` — Subprocess determinism guard reference; same shape applies to baseline/adaptive/pfedrec; Phase 6 must update `_RESULTS_DIR.glob` patterns.
- `scripts/foundation/tests/test_adaptive_determinism.py:189-211` — `_manifest.best_prototype` byte-identity invariant; Phase 6 extends to assert `_manifest.metrics` and `final_metrics` schema byte-identity.
- `scripts/foundation/tests/test_pfedrec_subprocess_determinism.py:56-67` — `_manifest.pfr08_verification` byte-identity; Phase 6 must keep this invariant after rewiring D-14 hook input.
- `federated-adaptive-personalized-cf/sweep.yaml:18` — `metric.name = final/sampled_ndcg@10` (must migrate to `best/sampled_ndcg@10` per Pitfall 7).
- `.planning/phases/02-baseline-migration/02-baseline-migration-04-SUMMARY.md` — D-15, D-25, D-26, D-27 patterns. Phase 6 carries D-15 + D-27 verbatim.
- `.planning/phases/04-adaptive-migration-bug-fixes/04-adaptive-migration-bug-fixes-05-SUMMARY.md` — D-05 + D-07 prototype paired restore pattern. Phase 6's extra-eval-round must run AFTER this restore.
- `.planning/phases/05-pfedrec-migration-reproduction/05-pfedrec-migration-reproduction-04-SUMMARY.md` — D-14 PFR-08 hook ordering (after `embed_manifest_in_result`, before W&B summary). Phase 6 keeps order; only the input changes.

### Secondary (MEDIUM confidence — existing patterns inferred from multiple files)
- `f"final/{key}"` W&B summary key convention is consistent across all four modules (lines 723, 814, 996, 1100) → migrating to `best/`/`last/` namespaces is a single, mechanical sed across all four.
- `Path("../results/federated[/<module>]")` is the only divergent point in the four servers' result-write blocks (baseline = `..`, personalized = `..`, adaptive = `../adaptive`, pfedrec = `../pfedrec`) → the foundation helper `module_run_results_dir(module, run_id)` collapses all four into one.
- The `dataclasses.replace` post-build mutation pattern is explicitly documented as "Phase-3/Phase-4 idiom" in the Phase 4 Plan 5 SUMMARY and Phase 5 Plan 4 SUMMARY → Phase 6 reuses it for `final_eval_round_index` and `metrics` fields.

### Tertiary (LOW confidence — assumptions to validate during planning)
- The cost estimate "~10–60s extra per run on 6040 clients" comes from CONTEXT.md and is unsubstantiated by a measurement; planner should confirm with one timed paper_compat_pfedrec smoke before locking the design. (Mitigation: even if the cost is 5× higher at 5min, it's still a one-time tax per run.)
- The assumption that `fraction_eval=1.0` is the right semantic for the extra eval round (vs `fraction_eval=mode_profile.fraction_eval`) — recommend "all nodes" but flag for planner review.

## Metadata

**Confidence breakdown:**
- Standard stack: **HIGH** — every dependency already declared and used in repo.
- Architecture: **HIGH** — every primitive (paths, atomic write, manifest, sufficient-stat aggregation, best-arrays snapshot) exists; Phase 6 only wires.
- Pitfalls: **HIGH** — six of ten pitfalls are direct consequences of code changes, traced to specific file/line refs (Pitfalls 1, 2, 4, 7, 8, 10).
- W&B sweep migration: **MEDIUM** — depends on whether Phase 7 plans to rewrite sweep.yaml anyway; recommend bumping in Phase 6 to avoid surprise.

**Research date:** 2026-04-29
**Valid until:** 2026-05-29 (30 days; in-repo code references are stable barring upstream Phase 5 hot-fixes).
