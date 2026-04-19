---
phase: 02-baseline-migration
plan: 05
type: execute
wave: 3
gap_closure: true
depends_on:
  - 02-baseline-migration-03
  - 02-baseline-migration-04
files_modified:
  - scripts/foundation/fedrec_foundation/fit_metrics.py
  - federated-baseline-cf/federated_baseline_cf/server_app.py
  - federated-baseline-cf/federated_baseline_cf/client_app.py
  - federated-baseline-cf/tests/test_server_integration.py
  - .planning/phases/02-baseline-migration/02-UAT.md
autonomous: true
requirements:
  - G-03-01

must_haves:
  truths:
    - "`selected_clients_per_round` stores PARTITION IDs (stable 0..N-1, tied to user identity via the canonical mapping), NOT Flower's ephemeral per-boot `node_id`s. Two reruns of the same config with the same `run-seed` produce byte-identical `selected_clients_per_round` lists."
    - "Server builds a `partition_to_node_id: Dict[int, int]` via a single discovery round before the main loop: fires one lightweight `@app.evaluate()` message to every node_id returned by `grid.get_node_ids()` with `discover_only=true` in the config; clients short-circuit and return only `{'partition_id': int(context.node_config['partition-id'])}`; server reads `partition_id` from each response and builds the mapping. Mapping is complete (all `N == num_supernodes` entries) before the first training round."
    - "Main training loop samples in partition-id space: `selected_pids = _server_sampler.sample(range(profile.num_supernodes), num_selected)`; `selected_node_ids = [partition_to_node_id[pid] for pid in selected_pids]`. `_server_sampler` is a single `server_rng(run_seed)` instance at loop-start. Raw node_ids are still used for message addressing but are NOT the recorded identifier."
    - "FitMetricsContract and EvaluateMetricsContract gain an OPTIONAL `partition_id: Optional[int] = None` field; `validate_evaluate_metrics` adds `partition_id` to its known-fields set so the strict contract still passes."
    - "Baseline client_app.py populates `partition_id=partition_id` in both `FitMetricsContract` and `EvaluateMetricsContract` outputs. Under `discover_only=true`, evaluate handler returns early with ONLY the required sufficient-stat zeros + partition_id (no model load, no data load, no scoring)."
  artifacts:
    - path: "scripts/foundation/fedrec_foundation/fit_metrics.py"
      provides: "partition_id field on both contracts; validate_evaluate_metrics updated"
      contains: "partition_id"
    - path: "federated-baseline-cf/federated_baseline_cf/server_app.py"
      provides: "Discovery-round handshake + partition-space sampling + partition_id-valued selected_clients_per_round"
      contains: "partition_to_node_id"
    - path: "federated-baseline-cf/federated_baseline_cf/client_app.py"
      provides: "partition_id echo in both contract outputs; discover_only short-circuit branch"
      contains: "discover_only"
    - path: "federated-baseline-cf/tests/test_server_integration.py"
      provides: "Subprocess-based cross-run reproducibility test (not pure-RNG)"
      contains: "test_selected_partitions_byte_identical_across_subprocess_reruns"
  key_links:
    - from: "federated_baseline_cf.server_app::main"
      to: "discovery round → partition_to_node_id mapping"
      via: "one-shot @app.evaluate broadcast with discover_only=true BEFORE round 1"
      pattern: "partition_to_node_id"
    - from: "federated_baseline_cf.server_app::main"
      to: "fedrec_foundation.rng.server_rng"
      via: "_server_sampler.sample(range(num_supernodes), k) — partition-id space"
      pattern: "sample\\(range\\(.*num_supernodes"
    - from: "federated_baseline_cf.client_app::evaluate"
      to: "discover_only short-circuit"
      via: "if msg.content['config'].get('discover_only'): return tiny EvaluateMetricsContract(... partition_id=partition_id)"
      pattern: "discover_only"
---

<objective>
Close G-03-01 (documented in `.planning/phases/02-baseline-migration/02-UAT.md` Gaps section): `selected_clients_per_round` is not byte-identical across reruns with the same `run-seed` because Flower's supernode node_ids are fresh per-boot random 64-bit values (os.urandom, not seedable).

Fix direction: stop recording Flower's ephemeral `node_id`. Record the stable `partition_id` (0..N-1, bound to user identity via the canonical mapping). Sample in partition-id space with `server_rng(run_seed).sample(range(N), k)` and translate to node_ids at send-time via a `partition_to_node_id` map built from a one-shot discovery round at server startup.

This brings Test 3 of `02-UAT.md` back to PASS (byte-identical `selected_clients_per_round` across two subprocess reruns with the same seed) and collapses the observed `sampled_ndcg@10` cross-run drift from ~1.05e-3 to ≤1e-4 (true GPU non-determinism). It also changes the semantics of `selected_clients_per_round` from "sequence of meaningless 64-bit handles" to "sequence of user-identifying partition indices" — a strict improvement for thesis audit trails.

Scope is contained to the baseline module plus a small foundation-contract extension (one optional field on each metrics contract). Other modules (`federated-personalized-cf`, `federated-adaptive-personalized-cf`, `federated-pfedrec`) are NOT touched in this plan; they inherit the optional contract field for free and can opt into the discovery+partition-sampling pattern in a later chore once this plan proves out on baseline.
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
@.planning/phases/02-baseline-migration/02-UAT.md
@.planning/phases/02-baseline-migration/02-CONTEXT.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Foundation contract — add optional `partition_id` field to both metrics contracts (D-21 extension)</name>
  <files>
    scripts/foundation/fedrec_foundation/fit_metrics.py
    scripts/foundation/tests/test_fit_metrics.py
  </files>
  <read_first>
    - scripts/foundation/fedrec_foundation/fit_metrics.py (entire file — dataclass + validators)
    - scripts/foundation/tests/test_fit_metrics.py (existing contract tests)
  </read_first>
  <action>
Surgical addition to both contracts:

1. `FitMetricsContract`: add `partition_id: Optional[int] = None` at the bottom of the Phase-2 extension block (keep alphabetical-ish grouping of per-group fields intact; place new field after `evaluated_users_dense` with a `# --- G-03-01 extension: client echoes its partition_id ---` comment). Update the class docstring's `Attributes` section with a one-line entry. `to_dict` already drops None so no change there. `from_dict`'s `known` set auto-picks it up via `fields(cls)`.

2. `EvaluateMetricsContract`: same addition (`partition_id: Optional[int] = None`) in the same position. `to_dict` / `from_dict` pick it up automatically.

3. `validate_evaluate_metrics`: no code change required — its `known` set is computed from `fields(EvaluateMetricsContract)` so `partition_id` is now a permitted field. `EVAL_METRICS_REQUIRED_KEYS` stays unchanged (partition_id is OPTIONAL — round-0 discovery clients still populate it, but omitting it does not break existing Phase-1 tests).

Tests (extend `test_fit_metrics.py`):

- `test_fit_metrics_contract_accepts_partition_id` — `FitMetricsContract(train_loss=0.1, num_positives=5, num_training_examples=25, partition_id=1234).to_dict()` includes `"partition_id": 1234`.
- `test_evaluate_metrics_contract_accepts_partition_id` — same shape for `EvaluateMetricsContract`.
- `test_validate_evaluate_metrics_allows_partition_id` — a payload with partition_id plus the three required sufficient-stat keys passes `validate_evaluate_metrics` without raising.
- `test_validate_evaluate_metrics_still_rejects_unknown_extras` — a payload with `foo="bar"` (not a known field) still raises `ValueError` as before. This is the regression guard: partition_id is known, anything else isn't.

Run: `pytest scripts/foundation/tests/test_fit_metrics.py -x`.
  </action>
  <acceptance_criteria>
    - `grep -c "partition_id: Optional\[int\]" scripts/foundation/fedrec_foundation/fit_metrics.py` returns 2 (one per contract).
    - All four new pytest cases pass; pre-existing `test_fit_metrics.py` tests still pass.
    - `validate_evaluate_metrics({"hit_count_overall_at10": 0, "ndcg_sum_overall_at10": 0.0, "evaluated_users": 0, "partition_id": 42})` returns `None` (no raise).
  </acceptance_criteria>
  <done>Optional `partition_id: int` field accepted by both `FitMetricsContract` and `EvaluateMetricsContract`; strict-contract validator whitelists it automatically; 4 new tests pass; no existing test regressed.</done>
</task>

<task type="auto">
  <name>Task 2: Baseline client — echo partition_id + `discover_only` short-circuit branch (G-03-01 client side)</name>
  <files>
    federated-baseline-cf/federated_baseline_cf/client_app.py
  </files>
  <read_first>
    - federated-baseline-cf/federated_baseline_cf/client_app.py (entire file, 425 LOC — understand @app.train and @app.evaluate current shape)
    - scripts/foundation/fedrec_foundation/fit_metrics.py (post-Task-1 shape)
  </read_first>
  <action>
Two edits to client_app.py (SURGICAL; preserve everything else):

**Edit A — `@app.train()` handler** (at the `FitMetricsContract(...)` construction, ~line 224):

Add `partition_id=partition_id` as the last kwarg passed to `FitMetricsContract(...)`. No other change to the train handler.

**Edit B — `@app.evaluate()` handler**: add a `discover_only` short-circuit BEFORE any heavy work (before mode resolution, before model creation, before `load_data`). Insert this block right after the `@app.evaluate()` def signature and the `partition_id = int(context.node_config["partition-id"])` line — move that partition_id read to the TOP of the handler body:

```python
@app.evaluate()
def evaluate(msg: Message, context: Context):
    # G-03-01 discovery-round short-circuit: server uses this to build
    # partition_id -> node_id mapping before round 1 so the per-round
    # sampler can work in partition-id space (stable 0..N-1) instead of
    # Flower's ephemeral node_id space (os.urandom, not seedable).
    partition_id = int(context.node_config["partition-id"])
    config = msg.content.get("config") or ConfigRecord()
    if bool(config.get("discover_only", False)):
        payload = EvaluateMetricsContract(
            hit_count_overall_at10=0,
            ndcg_sum_overall_at10=0.0,
            evaluated_users=0,
            partition_id=partition_id,
        ).to_dict()
        validate_evaluate_metrics(payload)
        content = RecordDict({"metrics": MetricRecord(payload)})
        return Message(content=content, reply_to=msg)
    # ... (existing evaluate body continues here, but DELETE the old
    # `partition_id = int(context.node_config["partition-id"])` line
    # further down because it's now already set) ...
```

Required imports (already present): `EvaluateMetricsContract`, `validate_evaluate_metrics`, `MetricRecord`, `RecordDict`, `Message`, `ConfigRecord`. Check with `head -60 federated-baseline-cf/federated_baseline_cf/client_app.py`; if `ConfigRecord` is not imported in this file, add `from flwr.app import ConfigRecord` (adjust the existing `from flwr.app import ...` line to include it).

Add `partition_id=partition_id` to the `EvaluateMetricsContract(...)` construction in the existing evaluate body (~line 402).

**IMPORTANT — no FitMetricsContract breakage:** `partition_id` is OPTIONAL on both contracts. The existing test suite (`tests/test_client_app*.py` if any) that asserts contract payload shape MUST continue to pass.
  </action>
  <acceptance_criteria>
    - `grep -c "partition_id=partition_id" federated-baseline-cf/federated_baseline_cf/client_app.py` returns at least 2 (one in train, one in evaluate).
    - `grep -c "discover_only" federated-baseline-cf/federated_baseline_cf/client_app.py` returns at least 1 (the short-circuit branch).
    - Running `python -c "from federated_baseline_cf.client_app import app; print('ok')"` from the `federated-baseline-cf/` directory succeeds (module imports).
    - `grep -n "partition_id = int(context.node_config\[" federated-baseline-cf/federated_baseline_cf/client_app.py` shows partition_id read at the top of BOTH train and evaluate handlers (no duplicate read further down in evaluate).
  </acceptance_criteria>
  <done>Client echoes partition_id in both contract outputs; evaluate handler short-circuits on `discover_only=true` and returns a tiny payload with partition_id only; no other behavior changed.</done>
</task>

<task type="auto">
  <name>Task 3: Baseline server — discovery round + partition-space sampling + partition_id-valued selected_clients_per_round (G-03-01 server side)</name>
  <files>
    federated-baseline-cf/federated_baseline_cf/server_app.py
  </files>
  <read_first>
    - federated-baseline-cf/federated_baseline_cf/server_app.py (entire file — identify exact insertion points)
    - scripts/foundation/fedrec_foundation/rng.py (server_rng signature — unchanged)
    - .planning/phases/02-baseline-migration/02-UAT.md Gap G-03-01 (fix direction)
  </read_first>
  <action>
Three surgical edits to `@app.main()`:

**Edit A — discovery round** (insert right after the `arrays = ArrayRecord(global_model.state_dict())` line and BEFORE the strategy-instantiation block, ~line 318):

```python
# =========================================================================
# G-03-01: discovery round. Build partition_id -> node_id mapping BEFORE the
# main loop so per-round sampling can work in stable partition-id space.
# Flower's node_ids are os.urandom-seeded per boot; partition_ids are the
# stable 0..N-1 identity we actually want to record for thesis reproducibility.
# =========================================================================
all_node_ids = list(grid.get_node_ids())
expected_n = int(profile.num_supernodes)
assert len(all_node_ids) == expected_n, (
    f"G-03-01 invariant: grid.get_node_ids() returned {len(all_node_ids)} "
    f"node_ids, expected num_supernodes={expected_n} from profile {profile.mode!r}."
)
print(f"\n[G-03-01] Running discovery round over {expected_n} supernodes...")
discovery_config = ConfigRecord({"discover_only": True})
discovery_messages = [
    grid.create_message(
        content=RecordDict({"arrays": ArrayRecord(), "config": discovery_config}),
        message_type="evaluate",
        dst_node_id=nid,
        group_id="discovery",
    )
    for nid in all_node_ids
]
discovery_responses = list(grid.send_and_receive(discovery_messages))
partition_to_node_id: Dict[int, int] = {}
for r in discovery_responses:
    if r.has_error():
        continue
    m = dict(r.content.get("metrics", MetricRecord()))
    pid = m.get("partition_id")
    if pid is None:
        continue
    partition_to_node_id[int(pid)] = int(r.metadata.src_node_id)
missing = sorted(set(range(expected_n)) - set(partition_to_node_id.keys()))
assert not missing, (
    f"G-03-01 invariant: discovery round did not collect partition_ids "
    f"for {len(missing)} nodes (first 5 missing: {missing[:5]}). "
    f"Cannot proceed — partition-space sampling would KeyError."
)
print(f"[G-03-01] Discovery complete: {len(partition_to_node_id)} partition -> node_id entries.")
```

**Edit B — per-round sampling** (in the main `for round_num in range(1, num_rounds + 1):` loop, replace the current `node_ids = sorted(grid.get_node_ids()); num_selected = ...; selected_node_ids = _server_sampler.sample(...)` block at ~lines 369-374):

```python
# G-03-01: sample in partition-id space (stable 0..N-1), translate to node_ids
# for message addressing. Deterministic across runs for a given run_seed.
num_selected = max(1, int(expected_n * fraction_train))
selected_pids: List[int] = _server_sampler.sample(range(expected_n), num_selected)
selected_node_ids = [partition_to_node_id[pid] for pid in selected_pids]

# D-26: persist + log PARTITION IDs (user-identifying, stable) — not node_ids.
selected_clients_per_round.append([int(pid) for pid in selected_pids])
if wandb_enabled:
    wandb.log({"round/selected_clients": [int(pid) for pid in selected_pids]}, step=round_num)
```

**Edit C — optional diagnostic** (defense-in-depth): right before the JSON write, append a parallel `selected_node_ids_per_round` only if a local debug flag is set (skip this for now; one-line comment noting the possibility is enough).

Also delete the obsolete `selected_node_ids = _server_sampler.sample(sorted(node_ids), num_selected)` code path. Keep `_server_sampler = server_rng(run_seed)` instantiation at loop-start — unchanged.

Verify the surrounding loop (`train_messages = [...dst_node_id=node_id...]`) still uses `selected_node_ids` correctly — it does, because `selected_node_ids` is still a `List[int]` of real Flower node_ids, just sourced via `partition_to_node_id`.

**Surgical discipline:** before editing, `git diff federated-baseline-cf/federated_baseline_cf/server_app.py > /tmp/sa_diff.txt` and confirm the only changes are the three surgical edits above. Do NOT touch the centralized-eval block, manifest block, D-27 best-round restore, weighted_average_metrics helper, print_evaluation_metrics, or the DummyClientProxy stub.
  </action>
  <acceptance_criteria>
    - `grep -c "partition_to_node_id" federated-baseline-cf/federated_baseline_cf/server_app.py` returns at least 3 (declaration + at-least-one-lookup + assertion).
    - `grep -c "discover_only" federated-baseline-cf/federated_baseline_cf/server_app.py` returns at least 1 (the ConfigRecord line).
    - `grep -c "sorted(node_ids)" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 0 (old code path eliminated).
    - `grep -c "_server_sampler.sample(range(" federated-baseline-cf/federated_baseline_cf/server_app.py` returns 1.
    - Dry-run sanity: `python -c "from federated_baseline_cf.server_app import app; print('ok')"` from `federated-baseline-cf/` succeeds.
    - `git diff --stat federated-baseline-cf/federated_baseline_cf/server_app.py` delta ≤ 80 lines (discovery block + sampling swap).
  </acceptance_criteria>
  <done>Server runs a discovery round over all 6040 supernodes before round 1; per-round sampling is partition-space deterministic; `selected_clients_per_round` is a list of partition_ids (0..N-1); raw node_ids never leak into the result JSON.</done>
</task>

<task type="auto">
  <name>Task 4: Subprocess-based reproducibility test (the regression guard Plan-04 missed)</name>
  <files>
    federated-baseline-cf/tests/test_server_integration.py
  </files>
  <read_first>
    - federated-baseline-cf/tests/test_server_integration.py (existing tests; add, do not replace)
    - scripts/run.py (launcher interface)
  </read_first>
  <action>
Add one new pytest test:

```python
def test_selected_partitions_byte_identical_across_subprocess_reruns(tmp_path):
    """G-03-01 regression guard: real-loop reproducibility, not pure-RNG.

    Plan-04's test_server_rng_reproducible_per_round_selection asserted
    `rng.sample(sorted(fixed_ids), k)` is stable — that's a pure-RNG property
    and always held. The REAL invariant the thesis needs is:

        subprocess run 1: python scripts/run.py baseline benchmark_cross_device ...
        subprocess run 2: python scripts/run.py baseline benchmark_cross_device ...
        => json_run1["selected_clients_per_round"] == json_run2["selected_clients_per_round"]

    This test runs the launcher twice in a child process, parses both result
    JSONs, and asserts byte-identity of the `selected_clients_per_round`
    field (which as of Plan-05 stores partition_ids, not node_ids).
    """
    # ... (test body: use subprocess.run with minimal config:
    #      num-server-rounds=1, fraction-train=0.01, local-epochs=1;
    #      skim the two latest results JSONs from results/federated/;
    #      assert selected_clients_per_round fields are equal)
```

Use a short configuration to keep the test fast:
- `num-server-rounds=1`
- `fraction-train=0.005` (~30 clients/round)
- `local-epochs=1`
- `wandb-enabled=false` (skip wandb in tests)
- `--federation local-sim-gpu` if available, fall back to CPU via env marker.

Mark the test with `@pytest.mark.slow` (≥ 30s wall time on CPU, ≥ 5s on GPU). Skip if `FEDREC_SKIP_SLOW=1` env var is set.

Also update the EXISTING `test_server_rng_reproducible_per_round_selection` with a docstring note: "This is a necessary-but-not-sufficient check. The load-bearing invariant is tested by `test_selected_partitions_byte_identical_across_subprocess_reruns`."
  </action>
  <acceptance_criteria>
    - `pytest federated-baseline-cf/tests/test_server_integration.py::test_selected_partitions_byte_identical_across_subprocess_reruns -x` passes on GPU (and optionally on CPU when the slow marker isn't skipped).
    - Pre-existing tests in the same file still pass.
  </acceptance_criteria>
  <done>Subprocess-based reproducibility test green; it WOULD have caught G-03-01 had it existed in Plan-04.</done>
</task>

<task type="auto">
  <name>Task 5: UAT update — Test 3 expectation + rerun, move result back to pass</name>
  <files>
    .planning/phases/02-baseline-migration/02-UAT.md
  </files>
  <read_first>
    - .planning/phases/02-baseline-migration/02-UAT.md (existing Gap G-03-01 and Test 3 block)
  </read_first>
  <action>
After Tasks 1-4 land and commits are clean, rerun the exact Test 3 command twice back-to-back and verify:

```bash
python scripts/run.py baseline benchmark_cross_device \
    --run-config 'num-server-rounds=2 fraction-train=0.01 model-type=bpr'
python scripts/run.py baseline benchmark_cross_device \
    --run-config 'num-server-rounds=2 fraction-train=0.01 model-type=bpr'

python -c "
import json, glob
files = sorted(glob.glob('/home/bes/Desktop/vinh/federated-learning/results/federated/*_results.json'))[-2:]
a, b = [json.load(open(f)) for f in files]
assert a['selected_clients_per_round'] == b['selected_clients_per_round'], 'G-03-01 fix broken'
diff = abs(a['final_metrics']['sampled_ndcg@10'] - b['final_metrics']['sampled_ndcg@10'])
assert diff <= 1e-3, f'ndcg@10 cross-run diff too large: {diff}'
print('G-03-01 PASS')
"
```

On green: flip Test 3 `result: fail` → `result: pass`; replace the failure notes with the new rerun summary (run_ids, byte-identity confirmation, new NDCG diff). Move G-03-01 from the Gaps section to a new `## Closed Gaps` section with a one-line closure note pointing at this plan's SUMMARY. Update `Summary` counts: `issues: 0`, `pending: 1`.
  </action>
  <acceptance_criteria>
    - `grep "result: pass" .planning/phases/02-baseline-migration/02-UAT.md | wc -l` returns at least 3 (Tests 1, 2, 3).
    - `grep -A1 "### G-03-01" .planning/phases/02-baseline-migration/02-UAT.md` is under a `## Closed Gaps` heading (not the open `## Gaps` heading).
    - Summary counts reflect 3 pass, 0 issues, 1 pending (Test 4).
  </acceptance_criteria>
  <done>Test 3 rerun passes; UAT reflects closure; G-03-01 moved to Closed Gaps.</done>
</task>

</tasks>

<success_criteria>
- G-03-01 observable: two back-to-back runs with the same `run-seed` produce byte-identical `selected_clients_per_round` lists and NDCG@10 cross-run diff ≤ 1e-3.
- Contract extension observable: both `FitMetricsContract` and `EvaluateMetricsContract` accept and serialize `partition_id`; `validate_evaluate_metrics` permits it without loosening strict-extras rejection.
- Discovery round observable: server-side log shows `[G-03-01] Discovery complete: 6040 partition -> node_id entries.` before round 1.
- Test coverage: the new subprocess reproducibility test would fail on pre-Plan-05 code (regression guard).
- No spill: personalized / adaptive / pfedrec modules are unchanged; they inherit the optional contract field without needing to populate it.
- UAT Test 3 flipped back to pass; Gap G-03-01 moved to Closed Gaps; Test 4 remains the only pending item (W&B verification — user-action).
</success_criteria>
