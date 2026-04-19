---
phase: 02-baseline-migration
plan: 04
type: execute
wave: 2
depends_on:
  - 02-baseline-migration-01
  - 02-baseline-migration-02
files_modified:
  - federated-baseline-cf/federated_baseline_cf/server_app.py
  - federated-baseline-cf/tests/test_server_integration.py
autonomous: true
requirements:
  - BSL-04
  - BSL-06
  - BSL-08

must_haves:
  truths:
    - "Server at startup calls `resolve_mode_defaults(mode)` + `log_mode_and_overrides(mode, profile, context.run_config)` — any value in context.run_config that overrides a locked mode default prints `⚠ OVERRIDE: <key>=<value> (mode default=<default>). Run is NOT comparable to benchmark thesis table.` and is captured in `manifest.overrides`."
    - "Server strategy is `BaselineFedAvg` or `BaselineFedProx` (from Plan 01's strategy.py) — NOT raw `FedAvg` / `FedProx`. Thesis-table metrics in `eval_metrics_history` come from `strategy.aggregate_evaluate(...)` (summed sufficient stats → one server-side ratio), not from `weighted_average_metrics(...)` of per-client ratios."
    - "Server-side per-round client sampling uses `server_rng(run_seed).sample(sorted_node_ids, num_selected)` — NOT `random.sample(node_ids, ...)` — and `selected_clients_per_round: List[List[int]]` is persisted to the result JSON (D-26) and logged to W&B (`wandb.log({'round/selected_clients': ids_list}, step=round_num)`)."
    - "`build_run_manifest(...)` is called ONCE at startup with run_id, profile, run_seed, and all four foundation fingerprints (mapping_sha256, split_hash, exclusion_sha256, foundation_contract_sha256) from `data/derived/foundation_index.json`; at result-writing time `embed_manifest_in_result(manifest, results_data)` + `write_manifest_sibling(manifest, json_path)` are called (D-15 double-write, BSL-08)."
    - "D-27 best-round restore: server tracks `best_sampled_ndcg_at10` and `best_round_state_dict` in-memory (no disk writes); at training end restores `arrays = ArrayRecord(best_round_state_dict)` before running centralized evaluation; result JSON records `best_round` in early_stopping/checkpoint-rule sections."
  artifacts:
    - path: "federated-baseline-cf/federated_baseline_cf/server_app.py"
      provides: "Cross-device server: mode resolver + seeded sampling + BaselineFedAvg + best-round restore + manifest"
      contains: "resolve_mode_defaults"
    - path: "federated-baseline-cf/tests/test_server_integration.py"
      provides: "BSL-04 seeded-sampling reproducibility + BSL-06 sufficient-stat + BSL-08 manifest tests"
      contains: "def test_server_rng_reproducible_per_round_selection"
  key_links:
    - from: "federated_baseline_cf.server_app::main"
      to: "fedrec_foundation.mode.resolve_mode_defaults"
      via: "mode = context.run_config['mode']; profile = resolve_mode_defaults(mode)"
      pattern: "resolve_mode_defaults\\("
    - from: "federated_baseline_cf.server_app::main"
      to: "fedrec_foundation.rng.server_rng"
      via: "server_rng(run_seed) with .sample(sorted_node_ids, num_selected)"
      pattern: "server_rng\\("
    - from: "federated_baseline_cf.server_app::main"
      to: "federated_baseline_cf.strategy.BaselineFedAvg"
      via: "strategy = BaselineFedAvg(fraction_fit=..., ...) replacing flwr FedAvg"
      pattern: "BaselineFedAvg\\("
    - from: "federated_baseline_cf.server_app::main"
      to: "fedrec_foundation.manifest.build_run_manifest"
      via: "build_run_manifest(run_id, profile, run_seed, ..., foundation_contract_sha256=..., module='baseline')"
      pattern: "build_run_manifest\\("
    - from: "federated_baseline_cf.server_app::main"
      to: "fedrec_foundation.manifest.embed_manifest_in_result + write_manifest_sibling"
      via: "D-15 double-write before closing the run"
      pattern: "embed_manifest_in_result|write_manifest_sibling"
---

<objective>
Migrate `federated-baseline-cf/federated_baseline_cf/server_app.py` to the cross-device contract. Closes three BSL requirements in a single plan because they all touch the same file on the same hot path (the `@app.main()` training loop):

- **BSL-04**: `random.sample(node_ids, num_selected)` (line ~297) replaced with `server_rng(run_seed).sample(sorted(node_ids), num_selected)`. Selected client IDs per round embedded in the result JSON (D-26) and logged to W&B per round.
- **BSL-06**: Custom `BaselineFedAvg` / `BaselineFedProx` strategy (from Plan 01) replaces raw `FedAvg` / `FedProx`. `eval_metrics_history[round_num]` comes from `strategy.aggregate_evaluate(...)` — server-side sum-based ratio — not from `weighted_average_metrics(...)` of per-client ratios.
- **BSL-08**: `build_run_manifest` called at startup; `embed_manifest_in_result` + `write_manifest_sibling` at result-write time (D-15 double-write).

Also implements D-25 (mode resolver owns canonical hyperparams; pyproject values are fallback; overrides captured per D-19), D-26 (selected clients per round persisted + W&B logged), and D-27 (in-memory best-round restore).

Purpose: Plans 01, 02, 03 set up the contract types, dataset layer, and client behaviors. Plan 04 is the orchestrator — once it lands, `python scripts/run.py baseline benchmark_cross_device` spawns 6040 supernodes, selects clients with a deterministic seeded RNG, aggregates sufficient stats the right way, restores the best round at training end, and writes a result JSON with a full protocol fingerprint.

D-18 surgical migration guard: server_app.py currently has pre-existing uncommitted hunks. Executor MUST run `git diff federated-baseline-cf/federated_baseline_cf/server_app.py` first to inventory them. This plan's rip targets are (1) line ~7 `import random`, (2) line ~297 `random.sample(node_ids, num_selected)`, (3) lines ~261-271 strategy instantiation, (4) lines ~260-590 eval-metric aggregation + result JSON construction (manifest embed), (5) early stopping integration for best-round restore. Everything else in the file stays as-is unless a specific BSL line in this task demands otherwise.

Output: (1) migrated `server_app.py` (~650 LOC, up from 587) with mode resolver, seeded sampling, BaselineFedAvg strategy, best-round restore, and D-15 double-write manifest. (2) `test_server_integration.py` with 4 tests covering BSL-04/06/08.
</objective>

<execution_context>
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/workflows/execute-plan.md
@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/REQUIREMENTS.md
@.planning/phases/02-baseline-migration/02-CONTEXT.md
@.planning/phases/01-foundation-contract/01-foundation-contract-04-SUMMARY.md
@.planning/phases/01-foundation-contract/01-foundation-contract-05-SUMMARY.md
@.planning/phases/02-baseline-migration/02-baseline-migration-01-PLAN.md
@.planning/phases/02-baseline-migration/02-baseline-migration-02-PLAN.md
@CLAUDE.md
@federated-baseline-cf/claude.md

@scripts/foundation/fedrec_foundation/rng.py
@scripts/foundation/fedrec_foundation/mode.py
@scripts/foundation/fedrec_foundation/manifest.py
@scripts/foundation/fedrec_foundation/bundle.py
@scripts/foundation/fedrec_foundation/evaluator.py
@scripts/foundation/fedrec_foundation/weight_policy.py
@scripts/run.py

@federated-baseline-cf/federated_baseline_cf/server_app.py

<interfaces>
<!-- Foundation + strategy surface consumed by server_app. -->

From fedrec_foundation.rng:
```python
def server_rng(run_seed: int) -> random.Random: ...
#   Usage: rng = server_rng(run_seed); rng.sample(sorted(node_ids), k) per round.
```

From fedrec_foundation.mode:
```python
def resolve_mode_defaults(mode: str, module_overrides=None) -> ModeProfile: ...
def log_mode_and_overrides(mode, profile, run_config) -> Dict[str, object]: ...
# ModeProfile attributes read here: num_server_rounds, fraction_train, fraction_eval,
#   embedding_dim, lr, local_epochs, weight_policy, primary_evaluator, checkpoint_rule,
#   num_train_negatives, num_eval_negatives
```

From fedrec_foundation.manifest:
```python
def generate_run_id() -> str: ...
def build_run_manifest(run_id, mode_profile, run_seed, mapping_sha256, split_hash,
                      exclusion_sha256, foundation_contract_sha256, raw_data_hash,
                      builder_version, overrides, module) -> RunManifest: ...
def embed_manifest_in_result(manifest, result_dict) -> dict: ...     # mutates
def write_manifest_sibling(manifest, result_json_path: Path) -> Path: ...
```

From fedrec_foundation.bundle:
```python
def verify_bundle(derived_dir) -> FoundationIndex: ...
# FoundationIndex fields: schema_version, builder_version, created_at,
#   mapping_sha256, split_hash, exclusion_sha256, foundation_contract_sha256
```

From fedrec_foundation.split:
```python
def load_split_manifest(path) -> SplitManifest: ...
# SplitManifest fields consumed: raw_data_hash, builder_version
```

From federated_baseline_cf.strategy (Plan 01 output):
```python
class BaselineFedAvg(BaseFedAvg):
    def aggregate_evaluate(self, server_round, results, failures)
        -> Tuple[Optional[float], Dict[str, Scalar]]: ...
class BaselineFedProx(BaseFedProx):
    def aggregate_evaluate(...): ...
```

Existing server_app shape to preserve (D-18):
- `DummyClientProxy(ClientProxy)` — thin stub that lets the strategy accept FitRes/EvaluateRes wrapped from Flower responses. Keep.
- `print_evaluation_metrics(round_num, metrics, context)` — pretty printer. Keep.
- `weighted_average_metrics(...)` — keep but DO NOT call on the primary metrics path; retained for rating RMSE/MAE which don't go through the sufficient-stat path.
</interfaces>

</context>

<tasks>

<task type="auto">
  <name>Task 1: Mode resolver + seeded sampling + BaselineFedAvg wiring + D-26 selected-clients logging (BSL-04, BSL-06 wire-up, D-25, D-26)</name>
  <files>
    federated-baseline-cf/federated_baseline_cf/server_app.py
  </files>
  <read_first>
    - federated-baseline-cf/federated_baseline_cf/server_app.py (ENTIRE file, 587 LOC — inventory pre-existing WIP via `git diff`; rip targets explicitly named above in objective)
    - federated-baseline-cf/federated_baseline_cf/strategy.py (Plan 01 output: BaselineFedAvg + BaselineFedProx)
    - scripts/foundation/fedrec_foundation/rng.py (server_rng signature)
    - scripts/foundation/fedrec_foundation/mode.py (resolve_mode_defaults + log_mode_and_overrides)
    - scripts/foundation/fedrec_foundation/manifest.py (build_run_manifest + embed + sibling signatures)
    - scripts/foundation/fedrec_foundation/bundle.py (verify_bundle + FoundationIndex fields)
    - .planning/phases/02-baseline-migration/02-CONTEXT.md §decisions (D-19, D-20, D-25, D-26, D-28)
    - CLAUDE.md "Code Standards" + "Logging and Metric Reporting"
  </read_first>
  <action>
**Pre-edit inventory.** `git diff federated-baseline-cf/federated_baseline_cf/server_app.py > /tmp/server_app_diff.txt`. Surgical rip targets:

1. Line ~7: `import random` — stripped.
2. Line ~23: `from flwr.server.strategy import FedAvg, FedProx` — keep (needed for isinstance checks elsewhere if any) but ADD `from federated_baseline_cf.strategy import BaselineFedAvg, BaselineFedProx`.
3. Lines ~261-271: strategy instantiation — replace `FedAvg(...)` with `BaselineFedAvg(...)`, `FedProx(...)` with `BaselineFedProx(...)`.
4. Line ~297: `selected_node_ids = random.sample(node_ids, num_selected)` — replace with `selected_node_ids = server_rng(run_seed).sample(sorted(node_ids), num_selected)`; IMPORTANT: use a SINGLE `server_rng` instance at loop-start so the sequence across rounds is deterministic.
5. After eval aggregation (around line ~402-405): replace `eval_metrics_history[round_num] = weighted_average_metrics(round_eval_metrics)` with the strategy call path (use `BaselineFedAvg.aggregate_evaluate`-computed metrics).
6. End-of-loop: insert best-round tracking for D-27 (in-memory `best_arrays`, `best_round_num`, `best_metric`). At training end, restore `arrays = best_arrays`.
7. Before writing JSON: call `build_run_manifest` + `embed_manifest_in_result` + `write_manifest_sibling`.

Scope **OUT** (D-18): `DummyClientProxy`, `weighted_average_metrics`, `print_evaluation_metrics`, wandb `init()` boilerplate (only ADD `wandb.config.update({'_manifest': manifest_dict})` per D-15), centralized-eval code at lines ~458-534 (kept; final_metrics stays), early_stopping boilerplate (kept; we augment, not replace).

**Step 1 — imports:** Add at the top of server_app.py (after existing imports):
```python
# Phase 2 Plan 04: foundation imports.
from fedrec_foundation.bundle import verify_bundle
from fedrec_foundation.manifest import (
    build_run_manifest,
    embed_manifest_in_result,
    generate_run_id,
    write_manifest_sibling,
)
from fedrec_foundation.mode import (
    log_mode_and_overrides,
    resolve_mode_defaults,
)
from fedrec_foundation.paths import data_derived
from fedrec_foundation.rng import server_rng
from fedrec_foundation.split import load_split_manifest

from federated_baseline_cf.strategy import BaselineFedAvg, BaselineFedProx
```

Remove the module-top `import random` line.

**Step 2 — mode resolution block** (insert immediately after `fraction_train = context.run_config[...]` lines, around line ~185 inside `@app.main()`):
```python
# =========================================================================
# Phase 2: mode resolver owns canonical hyperparams; pyproject values are
# fallback only (D-25). Overrides visible per D-19, captured in manifest.overrides.
# =========================================================================
mode = str(context.run_config.get("mode", "cross_silo_legacy"))
profile = resolve_mode_defaults(mode)
print(
    f"\n[MODE] Resolved profile mode={profile.mode!r} "
    f"num_supernodes={profile.num_supernodes} "
    f"weight_policy={profile.weight_policy!r} "
    f"primary_evaluator={profile.primary_evaluator!r}"
)
overrides = log_mode_and_overrides(mode, profile, context.run_config)
if overrides:
    # D-19 loud warning per key — already printed inside log_mode_and_overrides; add a SUMMARY line.
    print(
        f"⚠ OVERRIDE: {len(overrides)} key(s) diverge from mode default. "
        f"Run is NOT comparable to benchmark thesis table."
    )

run_seed = int(context.run_config.get("run-seed", 42))

# Resolve hyperparameters: profile is the source of truth, context.run_config overrides win.
num_rounds = int(context.run_config.get("num-server-rounds", profile.num_server_rounds))
fraction_train = float(context.run_config.get("fraction-train", profile.fraction_train))
lr = float(context.run_config.get("lr", profile.lr))
model_type = context.run_config.get("model-type", "bpr")
embedding_dim = int(context.run_config.get("embedding-dim", profile.embedding_dim))
dropout = float(context.run_config.get("dropout", 0.1))
strategy_name = str(context.run_config.get("strategy", "fedavg")).lower()
proximal_mu = float(context.run_config.get("proximal-mu", 0.0))
weight_policy = str(context.run_config.get("weight-policy", profile.weight_policy))
checkpoint_rule = str(context.run_config.get("checkpoint-rule", profile.checkpoint_rule))
```

**Step 3 — strategy instantiation.** Replace the `if strategy_name == "fedprox":` block at lines ~261-271 with:
```python
# BSL-06: BaselineFedAvg / BaselineFedProx overrides aggregate_evaluate to
# emit headline metrics from SUMMED sufficient stats (not averaged per-client ratios).
if strategy_name == "fedprox":
    strategy = BaselineFedProx(
        fraction_fit=fraction_train,
        proximal_mu=proximal_mu,
    )
    print(f"  Strategy: BaselineFedProx (proximal_mu={proximal_mu})")
else:
    strategy = BaselineFedAvg(
        fraction_fit=fraction_train,
    )
    print(f"  Strategy: BaselineFedAvg")
```

**Step 4 — seeded client sampling.** At the top of the FL loop, BEFORE the `for round_num in range(1, num_rounds + 1):` line, instantiate the server sampler ONCE:
```python
# BSL-04: replace random.sample(node_ids, ...) with a seeded RNG derived from run_seed.
# Single instance = deterministic sequence across rounds (like random.Random(seed).sample in a loop).
_server_sampler = server_rng(run_seed)
selected_clients_per_round: List[List[int]] = []  # D-26: persisted in result JSON
```

Replace line ~297 `selected_node_ids = random.sample(node_ids, num_selected)` with:
```python
selected_node_ids = _server_sampler.sample(sorted(node_ids), num_selected)
# D-26: persist + log selected client IDs for reproducibility + W&B audit.
selected_clients_per_round.append([int(x) for x in selected_node_ids])
if wandb_enabled:
    wandb.log({"round/selected_clients": [int(x) for x in selected_node_ids]}, step=round_num)
```

**Step 5 — aggregate_evaluate + best-round tracking.** Replace the existing eval-metrics aggregation block (around lines ~392-416) with:
```python
# =====================================================================
# EVALUATION AGGREGATION (BSL-06 via BaselineFedAvg.aggregate_evaluate)
# =====================================================================
eval_results = []
for response in eval_responses:
    if response.has_error():
        continue
    resp_metrics = response.content.get("metrics", MetricRecord())
    metrics_dict = dict(resp_metrics) if resp_metrics else {}
    num_examples = int(metrics_dict.get("num_training_examples", metrics_dict.get("evaluated_users", 1)))
    eval_res = EvaluateRes(
        status=Status(code=Code.OK, message="ok"),
        loss=float(metrics_dict.get("eval_loss", 0.0)),
        num_examples=num_examples,
        metrics=metrics_dict,
    )
    client_id = str(response.metadata.src_node_id)
    proxy = DummyClientProxy(client_id)
    eval_results.append((proxy, eval_res))

# BSL-06: strategy returns (loss, metrics_dict) with sum-based ratios + per-group ratios.
_loss, thesis_metrics = strategy.aggregate_evaluate(round_num, eval_results, [])
eval_metrics_history[round_num] = dict(thesis_metrics) if thesis_metrics else {}

# D-27 best-round tracking. Tracks sampled_ndcg@10 overall (profile.checkpoint_rule).
if checkpoint_rule == "best_round_restore" and thesis_metrics:
    current_ndcg = float(thesis_metrics.get("sampled_ndcg@10", 0.0))
    if round_num == 1 or current_ndcg > best_metric:
        best_metric = current_ndcg
        best_round_num = round_num
        best_arrays = ArrayRecord({k: v.detach().clone() for k, v in arrays.to_torch_state_dict().items()})
        print(f"  [CHECKPOINT] New best sampled_ndcg@10={best_metric:.4f} at round {best_round_num}")
```

Initialize the tracking variables BEFORE the FL loop:
```python
best_metric: float = float("-inf")
best_round_num: int = 0
best_arrays = arrays  # fallback to init if no rounds improve
```

Also add `from flwr.common import EvaluateRes, Status, Code` near the existing flwr imports (needed to wrap eval responses).

**Step 6 — restore best round before centralized eval.** Immediately before the "CENTRALIZED EVALUATION" section (line ~458), insert:
```python
# D-27: restore best-round global params before running the final centralized eval.
if checkpoint_rule == "best_round_restore" and best_round_num > 0:
    print(f"\n[CHECKPOINT] Restoring global params from best round {best_round_num} "
          f"(sampled_ndcg@10={best_metric:.4f}) before centralized evaluation")
    arrays = best_arrays
elif checkpoint_rule != "best_round_restore":
    print(f"\n[CHECKPOINT] checkpoint_rule={checkpoint_rule!r}: keeping last-round params")
```

**Step 7 — manifest assembly + D-15 double-write.** Between the centralized-eval JSON dump block (around line ~549) and the actual `with open(results_filename, 'w') as f: json.dump(...)`, insert the manifest construction:
```python
# =========================================================================
# BSL-08: protocol fingerprint manifest (FND-07 + D-15 double-write).
# =========================================================================
run_id = generate_run_id()
# Verify the bundle ONCE; raises if tampered. Reads fingerprints from foundation_index.json.
foundation_idx = verify_bundle(data_derived())
# raw_data_hash + builder_version live on the SplitManifest (single source of truth per IMP-2).
split_manifest = load_split_manifest(data_derived() / "split_manifest.json")
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
    module="baseline",
)

# D-26: selected_clients_per_round is a first-class field in the JSON.
results_data["selected_clients_per_round"] = selected_clients_per_round
results_data["checkpoint"] = {
    "rule": checkpoint_rule,
    "best_round": best_round_num,
    "best_sampled_ndcg@10": best_metric if best_metric != float("-inf") else None,
}

# D-15: double-write (embedded in result JSON + sibling file).
embed_manifest_in_result(manifest, results_data)  # mutates in place

# Write result JSON (unchanged path pattern; D-28 flat results/federated/).
results_filename = results_dir / f"{run_id}_results.json"
with open(results_filename, 'w') as f:
    json.dump(results_data, f, indent=4, default=str)
# D-15 sibling.
sibling_path = write_manifest_sibling(manifest, results_filename)
print(f"Results saved to: {results_filename.resolve()}")
print(f"Manifest sibling: {sibling_path.resolve()}")

# W&B: attach manifest to the run's config for dashboard filtering.
if wandb_enabled:
    wandb.config.update({"_manifest": {
        "run_id": manifest.run_id,
        "mode": manifest.mode,
        "num_supernodes": manifest.num_supernodes,
        "foundation_contract_sha256": manifest.foundation_contract_sha256,
        "split_hash": manifest.split_hash,
        "run_seed": manifest.run_seed,
        "checkpoint_rule": manifest.checkpoint_rule,
    }})
```

Ensure that `results_dir` is set to `Path("../results/federated")` (unchanged from pre-existing code — D-28 flat directory).

**Step 8 — update wandb config.** The existing `wandb.init(config=wandb_config)` block at the top of `@app.main()` stays, but `wandb_config` grows to include:
```python
wandb_config.update({
    "mode": mode,
    "run_seed": run_seed,
    "weight_policy": weight_policy,
    "partition_mode": profile.partition_mode,
    "checkpoint_rule": checkpoint_rule,
})
```

**Do-not-touch ranges** (D-18): `DummyClientProxy` class, `weighted_average_metrics` (kept for RMSE/MAE path), `print_evaluation_metrics`, centralized eval code at lines ~458-534 (final_metrics construction stays; only the post-final-metrics block adds the manifest), early_stopping config parsing at lines ~197-211 (kept; we only AUGMENT with best-round tracking).
  </action>
  <verify>
    <automated>grep -c "BaselineFedAvg\\|BaselineFedProx" federated-baseline-cf/federated_baseline_cf/server_app.py && grep -c "server_rng\\|resolve_mode_defaults\\|build_run_manifest\\|embed_manifest_in_result\\|write_manifest_sibling" federated-baseline-cf/federated_baseline_cf/server_app.py && ! grep -E "^import random$|random\\.sample\\(" federated-baseline-cf/federated_baseline_cf/server_app.py</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c "^import random$" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 0.
    - `grep -c "random.sample(" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 0.
    - `grep -c "BaselineFedAvg(" federated-baseline-cf/federated_baseline_cf/server_app.py` returns at least 1.
    - `grep -c "BaselineFedProx(" federated-baseline-cf/federated_baseline_cf/server_app.py` returns at least 1.
    - `grep -c "server_rng(run_seed)" federated-baseline-cf/federated_baseline_cf/server_app.py` returns at least 1.
    - `grep -c "resolve_mode_defaults(" federated-baseline-cf/federated_baseline_cf/server_app.py` returns at least 1.
    - `grep -c "log_mode_and_overrides(" federated-baseline-cf/federated_baseline_cf/server_app.py` returns at least 1.
    - `grep -c "build_run_manifest(" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 1.
    - `grep -c "embed_manifest_in_result(" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 1.
    - `grep -c "write_manifest_sibling(" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 1.
    - `grep -c "selected_clients_per_round" federated-baseline-cf/federated_baseline_cf/server_app.py` returns at least 2 (init + append + json field).
    - `grep -c "best_round_restore\\|best_metric\\|best_round_num\\|best_arrays" federated-baseline-cf/federated_baseline_cf/server_app.py` returns at least 3 (D-27 tracking).
    - `grep -c "strategy.aggregate_evaluate(" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 1.
    - `python -c "import ast; ast.parse(open('federated-baseline-cf/federated_baseline_cf/server_app.py').read()); print('syntax ok')"` exits 0.
    - `python -c "from federated_baseline_cf.server_app import app; print('import ok')"` exits 0.
  </acceptance_criteria>
  <done>server_app.py: BSL-04 seeded sampling + D-26 selected-clients log, BSL-06 BaselineFedAvg wire-up, BSL-08 double-write manifest, D-25 mode resolver, D-27 in-memory best-round restore. No `random.seed`/`random.sample`/`import random`. Module still imports.</done>
</task>

<task type="auto">
  <name>Task 2: Server integration tests — BSL-04 reproducibility, BSL-06 aggregation path, BSL-08 manifest shape</name>
  <files>
    federated-baseline-cf/tests/test_server_integration.py
  </files>
  <read_first>
    - federated-baseline-cf/federated_baseline_cf/server_app.py (post-Task-1 state)
    - federated-baseline-cf/federated_baseline_cf/strategy.py (Plan 01 output)
    - scripts/foundation/fedrec_foundation/rng.py (server_rng reproducibility contract)
    - scripts/foundation/fedrec_foundation/manifest.py (RunManifest required fields)
    - scripts/foundation/tests/test_rng.py (existing reproducibility-test patterns to mirror)
    - scripts/foundation/tests/test_manifest.py (manifest shape tests to mirror)
  </read_first>
  <action>
Create `federated-baseline-cf/tests/test_server_integration.py` with 4 tests. No Flower federation is spawned — these are unit/integration tests that exercise the foundation contract + strategy + manifest path server_app will use.

```python
"""Server integration tests (Phase 2 Plan 04)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "derived" / "foundation_index.json").exists(),
    reason="foundation bundle not committed",
)


def test_server_rng_reproducible_per_round_selection() -> None:
    """BSL-04: server_rng(run_seed) produces identical client sequences across processes."""
    from fedrec_foundation.rng import server_rng
    # Two separate rng instances with the same seed -> same sequence.
    rng1 = server_rng(42)
    rng2 = server_rng(42)
    ids = list(range(6040))
    seq1 = [tuple(rng1.sample(sorted(ids), 50)) for _ in range(3)]
    seq2 = [tuple(rng2.sample(sorted(ids), 50)) for _ in range(3)]
    assert seq1 == seq2, "server_rng(42) must produce byte-identical sequences across instances"


def test_server_rng_different_seeds_different_selections() -> None:
    """BSL-04 negative guard: different run_seeds -> different sequences."""
    from fedrec_foundation.rng import server_rng
    rng1 = server_rng(42)
    rng2 = server_rng(43)
    ids = list(range(6040))
    s1 = rng1.sample(sorted(ids), 50)
    s2 = rng2.sample(sorted(ids), 50)
    assert s1 != s2, "Different seeds MUST yield different client-selection sequences"


def test_aggregate_evaluate_uses_sum_not_average() -> None:
    """BSL-06: BaselineFedAvg.aggregate_evaluate returns sum-based ratios, not mean-of-ratios."""
    from unittest.mock import MagicMock
    from flwr.common import EvaluateRes, Status, Code
    from federated_baseline_cf.strategy import BaselineFedAvg

    strategy = BaselineFedAvg(fraction_fit=0.1)
    proxy = MagicMock(); proxy.cid = "c"

    # Client A: 1 hit on 1 user (HR=1.0), NDCG=1.0. Client B: 0 hits on 99 users (HR=0).
    # Per-client AVERAGE of ratios = (1.0 + 0.0) / 2 = 0.5  <-- WRONG
    # SUM-BASED ratio           = 1 / 100 = 0.01                          <-- CORRECT
    results = [
        (proxy, EvaluateRes(Status(Code.OK, "ok"), 0.5, 1, {
            "eval_loss": 0.5,
            "hit_count_overall_at10": 1, "ndcg_sum_overall_at10": 1.0, "evaluated_users": 1,
            "hit_count_sparse_at10": 0, "ndcg_sum_sparse_at10": 0.0, "evaluated_users_sparse": 0,
            "hit_count_medium_at10": 0, "ndcg_sum_medium_at10": 0.0, "evaluated_users_medium": 0,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.0, "evaluated_users_dense": 0,
        })),
        (proxy, EvaluateRes(Status(Code.OK, "ok"), 0.5, 99, {
            "eval_loss": 0.5,
            "hit_count_overall_at10": 0, "ndcg_sum_overall_at10": 0.0, "evaluated_users": 99,
            "hit_count_sparse_at10": 0, "ndcg_sum_sparse_at10": 0.0, "evaluated_users_sparse": 0,
            "hit_count_medium_at10": 0, "ndcg_sum_medium_at10": 0.0, "evaluated_users_medium": 0,
            "hit_count_dense_at10": 0, "ndcg_sum_dense_at10": 0.0, "evaluated_users_dense": 0,
        })),
    ]
    _loss, metrics = strategy.aggregate_evaluate(1, results, [])
    assert metrics["sampled_hr@10"] == pytest.approx(1.0 / 100.0, abs=1e-9), (
        f"BSL-06: sum-based ratio must be 1/100=0.01, got {metrics['sampled_hr@10']}"
    )
    # Sanity: mean-of-ratios would be 0.5; we MUST not be 0.5.
    assert metrics["sampled_hr@10"] < 0.5, "BSL-06: aggregation is NOT averaging per-client ratios"


def test_build_run_manifest_integrates_foundation_index() -> None:
    """BSL-08: build_run_manifest integrates all four foundation fingerprints."""
    from fedrec_foundation.bundle import verify_bundle
    from fedrec_foundation.mode import resolve_mode_defaults
    from fedrec_foundation.manifest import build_run_manifest, generate_run_id, embed_manifest_in_result, write_manifest_sibling
    from fedrec_foundation.paths import data_derived
    from fedrec_foundation.split import load_split_manifest

    derived = data_derived()
    idx = verify_bundle(derived)
    split = load_split_manifest(derived / "split_manifest.json")
    profile = resolve_mode_defaults("benchmark_cross_device")

    manifest = build_run_manifest(
        run_id=generate_run_id(),
        mode_profile=profile,
        run_seed=42,
        mapping_sha256=idx.mapping_sha256,
        split_hash=idx.split_hash,
        exclusion_sha256=idx.exclusion_sha256,
        foundation_contract_sha256=idx.foundation_contract_sha256,
        raw_data_hash=split.raw_data_hash,
        builder_version=split.builder_version,
        overrides={"lr": 0.005},
        module="baseline",
    )
    # All 4 IMP-2 fingerprints present.
    assert manifest.mapping_sha256 == idx.mapping_sha256
    assert manifest.split_hash == idx.split_hash
    assert manifest.exclusion_sha256 == idx.exclusion_sha256
    assert manifest.foundation_contract_sha256 == idx.foundation_contract_sha256
    assert manifest.raw_data_hash == split.raw_data_hash
    # Mode profile propagated.
    assert manifest.mode == "benchmark_cross_device"
    assert manifest.num_supernodes == 6040
    assert manifest.weight_policy == "num_positives"
    assert manifest.primary_evaluator == "sampled_loo_99"
    # Overrides captured.
    assert manifest.overrides == {"lr": 0.005}
    assert manifest.module == "baseline"


def test_embed_and_sibling_double_write_roundtrip(tmp_path) -> None:
    """BSL-08 + D-15: double-write (embedded + sibling) roundtrips to JSON cleanly."""
    from fedrec_foundation.bundle import verify_bundle
    from fedrec_foundation.mode import resolve_mode_defaults
    from fedrec_foundation.manifest import (
        build_run_manifest, generate_run_id, embed_manifest_in_result, write_manifest_sibling,
    )
    from fedrec_foundation.paths import data_derived
    from fedrec_foundation.split import load_split_manifest

    derived = data_derived()
    idx = verify_bundle(derived)
    split = load_split_manifest(derived / "split_manifest.json")
    profile = resolve_mode_defaults("benchmark_cross_device")
    manifest = build_run_manifest(
        run_id=generate_run_id(), mode_profile=profile, run_seed=42,
        mapping_sha256=idx.mapping_sha256, split_hash=idx.split_hash,
        exclusion_sha256=idx.exclusion_sha256,
        foundation_contract_sha256=idx.foundation_contract_sha256,
        raw_data_hash=split.raw_data_hash, builder_version=split.builder_version,
        overrides={}, module="baseline",
    )
    result = {"training_rounds": 10, "final_metrics": {"sampled_ndcg@10": 0.25}}
    embed_manifest_in_result(manifest, result)  # mutates
    result_path = tmp_path / f"{manifest.run_id}_results.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4, default=str)
    sibling = write_manifest_sibling(manifest, result_path)
    # Both artifacts exist + contain the foundation_contract_sha256.
    assert result_path.exists()
    assert sibling.exists()
    with open(result_path) as f:
        roundtrip = json.load(f)
    assert roundtrip["_manifest"]["foundation_contract_sha256"] == idx.foundation_contract_sha256
    with open(sibling) as f:
        sibling_json = json.load(f)
    assert sibling_json["foundation_contract_sha256"] == idx.foundation_contract_sha256
```
  </action>
  <verify>
    <automated>cd federated-baseline-cf && pytest tests/test_server_integration.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `pytest federated-baseline-cf/tests/test_server_integration.py -v 2>&1 | grep -E "passed|failed"` shows 5 passed, 0 failed.
    - `grep -c "def test_" federated-baseline-cf/tests/test_server_integration.py` returns 5.
    - `grep -c "server_rng\\|aggregate_evaluate\\|build_run_manifest\\|embed_manifest_in_result\\|write_manifest_sibling" federated-baseline-cf/tests/test_server_integration.py` returns at least 5.
  </acceptance_criteria>
  <done>4 tests cover BSL-04 (reproducible + distinguishable seeds), BSL-06 (sum-not-average aggregation), BSL-08 (manifest field completeness + double-write roundtrip). 5 GREEN tests total.</done>
</task>

</tasks>

<verification>
Full-phase verification for Plan 04:

1. `pytest federated-baseline-cf/tests/ -v` shows 5 + 8 + 3 + 5 = 21 passed across the four test files (aggregating Plans 01-04 suites). Run: `cd federated-baseline-cf && pytest tests/ -v`.
2. Anti-pattern regression: `grep -rn "random.seed\|random.sample\|^import random$" federated-baseline-cf/federated_baseline_cf/` returns 0 matches.
3. BSL-06 invariant regression: `grep "weighted_average_metrics(round_eval_metrics)" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 0 matches on the thesis-metric code path (may still exist for RMSE/MAE on the full-rank path — D-18 preserves that WIP).
4. BSL-08 end-to-end smoke test (manual, executor runs once): `python scripts/run.py --dry-run baseline benchmark_cross_device` — stdout prints `num-supernodes=6040 mode=benchmark_cross_device`. (We cannot run a full 6040-supernode simulation in CI, but --dry-run proves the launcher + app contract agree.)
5. D-18 guard: `git diff --stat federated-baseline-cf/federated_baseline_cf/dataset.py federated-baseline-cf/federated_baseline_cf/client_app.py federated-baseline-cf/federated_baseline_cf/task.py` shows NO new changes attributable to Plan 04 (only Plan 02/03 owned those files).
</verification>

<success_criteria>
- BSL-04 observable: `server_rng(run_seed).sample(sorted(node_ids), ...)` replaces `random.sample(node_ids, ...)`; two back-to-back runs with the same `run-seed` select the same client sequence. `selected_clients_per_round` appears in the result JSON (D-26) and is logged per round to W&B.
- BSL-06 observable: Strategy is `BaselineFedAvg` / `BaselineFedProx`; `strategy.aggregate_evaluate(...)` returns sum-based ratios for `sampled_hr@10` / `sampled_ndcg@10` + per-group variants; eval_metrics_history is populated from this path.
- BSL-08 observable: `build_run_manifest` + `embed_manifest_in_result` + `write_manifest_sibling` called once per run; results JSON has a `_manifest` key with all 23 fields; a sibling `<run_id>-manifest.json` file is written.
- D-25 observable: `resolve_mode_defaults` + `log_mode_and_overrides` at startup; `[MODE OVERRIDE]` log lines appear whenever a `context.run_config` key diverges from the mode default; overrides captured in `manifest.overrides`.
- D-27 observable: `best_metric` / `best_round_num` / `best_arrays` tracked in-memory; at training end, `arrays = best_arrays` is set before centralized eval; `checkpoint.best_round` + `checkpoint.best_sampled_ndcg@10` appear in result JSON.
- 5 new GREEN tests in `test_server_integration.py` bring module total to 21 GREEN tests (aggregating Plans 01-04).
- D-18 surgical guard preserved: `dataset.py` / `client_app.py` / `task.py` untouched by this plan.
</success_criteria>

<output>
After completion, create `.planning/phases/02-baseline-migration/02-baseline-migration-04-SUMMARY.md` following the template in `@/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system/.claude/get-shit-done/templates/summary.md`.
</output>
