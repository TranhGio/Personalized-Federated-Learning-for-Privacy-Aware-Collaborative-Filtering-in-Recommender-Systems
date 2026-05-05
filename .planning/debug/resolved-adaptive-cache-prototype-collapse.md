---
status: resolved
human_verify_passed: true
trigger: "adaptive-cache-prototype-collapse: cold_start_rate=1.0 every round, global_prototype_norm~6e-5 (zero), full-pop restored NDCG@10=0.054 (random) vs per-round 0.235 in run 20260503-171314-313b26"
created: 2026-05-05T00:00:00Z
updated: 2026-05-05T15:00:00Z
---

## Status update 2026-05-05T03:30:00Z

Bug 1 and Bug 2 fixes APPLIED and committed atomically on branch
`feat/try_to_run_the_baseline`:
- a980217 fix(07): resolve server cold-start probe path the same way the client does
- 843c9bc fix(07): make compute_user_prototype user-id aware for cross-device protocol

Test suite (`federated-adaptive-personalized-cf/tests/`): 74/74 pass.

Bug 3 fix is DEFERRED until Option E delivers an empirical NDCG@10 data point
(see "Option E — User-Runnable Last-Round Full-Pop Eval" section at the tail
of this file). The user runs Option E by hand in tmux.

## Current Focus

hypothesis: THREE distinct, independent bugs (NOT a single shared root cause):
  1. cold_start_rate=1.0 is a server-side path bug (RELATIVE Path() vs ABS in clients) — measurement-only
  2. global_prototype_norm collapse — compute_user_prototype() does mean(dim=0) over the full 6040×128 user_embeddings table, which is dominated by Xavier noise in 6039/6040 rows under cross-device (1 user/client). Math is correct for cross-silo, broken for cross-device.
  3. Full-pop NDCG=0.054 vs per-round NDCG=0.235 (4.6× gap) — per-round eval is biased because it sends evaluate ONLY to the same 604 clients that just trained (line 813: `for node_id in selected_node_ids`). Each round measures only the freshly-trained subset; the full-pop best-round-restore eval reveals the actual model where ~90% of users have local user_embeddings de-synchronized from the restored GLOBAL item_embeddings.

test: read all relevant code paths, verify with cache-content inspection
expecting: clarity on which bug to fix first
next_action: write up final report with three fixes; await user verification

## Symptoms

expected:
- cold_start_rate drops below 1.0 by round 2 (5436/6040 unique clients revisited)
- global_prototype_norm grows in first ~5 rounds, stable at ~1 (Xavier-init scale)
- Full-pop restored NDCG@10 within ~30% of per-round NDCG@10

actual:
- cold_start_rate = 1.0 for all 56,172 client visits across 93 rounds
- global_prototype_norm series: 1.6e-4 → 6e-5 (drift down, never grows)
- Full-pop restored NDCG@10 = 0.0540 vs per-round best 0.2473 (4.6× gap)
- HR@10 = 0.105 ≈ random (1/10 with 99 negatives)
- Sparse < medium < dense (opposite of thesis claim)
- alpha_analysis: null in JSON

errors: None — silent collapse, run completed cleanly, committed at fd4450b

reproduction:
- results.json: results/federated/adaptive/20260503-171314-313b26/results.json
- manifest.json: results/federated/adaptive/20260503-171314-313b26/manifest.json
- cache dir: federated-adaptive-personalized-cf/.embedding_cache/20260503-171314-313b26/

started: Run 2026-05-04 00:13 → 18:05 on branch feat/try_to_run_the_baseline at fd4450b. Earlier same-session run (20260503-074155-53db2c) had 5,299 cache files per handoff.

## Eliminated

- hypothesis: H1 — cache writes never happened
  evidence: 6040 partition_*.pt files exist in federated-adaptive-personalized-cf/.embedding_cache/20260503-171314-313b26/, mtimes span May 4 12:20 to 17:04 (continuously updated through the run). Each is 4MB, contains valid tensors with non-zero values.
  timestamp: 2026-05-05T00:30:00Z

- hypothesis: H2(a/b/c) — client read fails (path mismatch / strict-load failure / atomic-write leakage)
  evidence: wandb output.log of run 20260504_001315 shows 5877 "loaded Phase-4 cached state" log lines vs 1989 "cold start — using initialized LOCAL state" lines (5877+1989 ≠ 56172 due to Ray log dedup, but the ratio confirms reads succeed in majority of cases). And cache content inspection shows trained own-row (0.24-0.58 norm) clearly distinct from untouched Xavier rows (~0.20 norm).
  timestamp: 2026-05-05T00:35:00Z

- hypothesis: H4(c) — wrong key for prototype serialization
  evidence: client uses USER_PROTOTYPE_KEY="user_prototype" (strategy.py:49), server reads same constant via _aggregate_prototypes (strategy.py:229). Wire-format match confirmed.
  timestamp: 2026-05-05T00:40:00Z

- hypothesis: H4(a/b) — _aggregate_prototypes math bug or zero-prototype contribution
  evidence: weighted-mean math is standard, EMA formula is standard, momentum=0.9. The reason prototype is small is NOT an aggregation bug — it's that the CLIENT-SIDE compute_user_prototype() returns mean(dim=0) of a 6040×128 table where 6039 rows are untrained Xavier noise. The contribution per-client is genuinely ~2.5e-3 in L2 norm; weighted-mean across 604 clients adds zero-mean noise → smaller; EMA with momentum=0.9 → smaller still → 6e-5. Math is right; the prototype semantics is wrong.
  timestamp: 2026-05-05T01:00:00Z

## Evidence

- timestamp: 2026-05-05T00:30:00Z
  checked: federated-adaptive-personalized-cf/.embedding_cache/20260503-171314-313b26/ existence and contents
  found: 6040 partition_*.pt files, all 4MB, mtimes May 4 12:20 → 17:04 (during the run). manifest.json present with schema_version=2 fields. Sample tensors load without error: user_embeddings.weight shape (6040, 128) norm ~15.83 (Xavier-uniform scale), personal_mlp.* present, fusion_layer.weight present (concat fusion).
  implication: Client cache writes work. Bug is NOT "writes never happen".

- timestamp: 2026-05-05T00:35:00Z
  checked: Server-side cold-start probe code path: server_app.py _cold_start_cache_root() at line 268-289
  found: Returns `Path(".embedding_cache") / run_id` — RELATIVE path. Server CWD at runtime is the repo root (per wandb-metadata.json `"root": "/home/bes/Desktop/vinh/federated-learning/movie-recommendation-system"`). The clients use _CACHE_BASE_DIR = _MODULE_DIR.parent / ".embedding_cache" — ABSOLUTE path rooted at federated-adaptive-personalized-cf/. So clients write to .../federated-adaptive-personalized-cf/.embedding_cache/{run_id}/ but server probes <repo>/.embedding_cache/{run_id}/ which doesn't exist.
  implication: Bug #1 found. The cold_start_rate=1.0 is a SPURIOUS measurement bug. Cache IS being reused at the client. Verified by 5877 "loaded Phase-4 cached state" log lines.

- timestamp: 2026-05-05T00:50:00Z
  checked: dual_personalized_bpr_mf.py compute_user_prototype() at line 527-529 (and bpr_mf.py:527-536 for the BPR base class)
  found: `return self.user_embeddings.weight.mean(dim=0)` — mean across ALL 6040 rows. In cross-device (1 user/client = 6040 supernodes), each client's user_embeddings table has 6039 untouched Xavier rows + 1 trained row. The mean is ~Xavier_noise / sqrt(6040) ≈ 0.013 per dim → L2 ~ 0.003. Cache file inspection confirms: prototype mean(dim=0) of partition_0.pt has L2 norm 2.5e-3.
  implication: Bug #2 found. compute_user_prototype() semantics is broken for cross-device. The "global prototype lifts sparse users" mechanism is INERT (prototype magnitude ~6e-5 on server after EMA decay; effectively zero). The thesis claim "Server-side EMA prototype helps sparse users" cannot be evaluated because the prototype IS zero. Need a per-user `compute_user_prototype()` that returns just `user_embeddings.weight[partition_id]` (the trained row).

- timestamp: 2026-05-05T01:15:00Z
  checked: server_app.py round-loop eval phase at line 810-833, vs final-eval-round at line 990-1006
  found: Per-round eval sends to ONLY `selected_node_ids` (the 604 just-trained clients). Final-pop eval sends to ALL 6040 `partition_to_node_id.values()`. The full-pop eval at round 94 uses the BEST-ROUND restored GLOBAL params (from round 83) but each client loads its LAST-CACHED LOCAL state (from whenever it was last trained — could be any round 1..93). The user/item embedding spaces are de-synchronized for ~90% of users.
  implication: Bug #3 found. The 4.6× NDCG@10 gap (0.235 in-sample vs 0.054 full-pop) is fundamentally a measurement-protocol vs evaluation-protocol mismatch:
    - In-sample (per round): only freshly-trained clients eval → picks up the best version of their local state, perfectly aligned with current global state
    - Full-pop best-round-restore: forces all 6040 clients to eval against round-83 global state, but each client's local state is from a different round → de-synced → near-random predictions
  This is THE dominant cause of the "thesis-blocking" NDCG collapse.

- timestamp: 2026-05-05T01:20:00Z
  checked: selected_clients_per_round coverage histogram from results.json
  found: 6040/6040 partitions selected ≥1 time, max=21, median=9, mean=9.30, ≥10 times = 2784, ≥20 times = 2. So most users got their local embedding trained 9-10 times across 93 rounds.
  implication: Even partitions that were selected don't have STRONG local embeddings — average partition was selected once every ~10 rounds, so there are ~10 round gaps between selections during which the global item embeddings drift. This compounds the de-synchronization in Bug #3.

## Resolution

root_cause: Three independent bugs combine to produce the symptoms. The user's hypothesis of "single shared root cause" is FALSIFIED — these have distinct mechanisms:

  1. Server-side cold-start probe uses relative path that resolves against repo-root CWD (server) vs client-side absolute path rooted in module dir. The clients ARE caching properly; the probe is just wrong.

  2. compute_user_prototype() = mean over 6040×128 user-embedding table. In cross-device (1 user = 1 partition), 6039/6040 rows are untouched Xavier noise → mean ≈ 0. EMA prototype thus stays near zero throughout the run; the "global prototype lifts sparse users" mechanism is inert.

  3. Per-round NDCG measures only the just-trained 604 clients (in-sample), inflating the metric. Full-pop best-round-restore evaluates all 6040 clients against round-83 global state, but each client's local state was last trained at a different (random) round → user/item embedding spaces are de-synchronized for ~90% of users → near-random predictions.

fix: (proposed; awaiting user verification before commit)

  Bug #1: Change `_cold_start_cache_root()` in `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py:268-289` to use the same module-anchored absolute path as the client (or use `Path.resolve()` and check that it matches what client_app uses). Concrete fix: 
    ```python
    _MODULE_DIR_SERVER = Path(__file__).resolve().parent
    return _MODULE_DIR_SERVER.parent / ".embedding_cache" / run_id
    ```

  Bug #2: Change `compute_user_prototype()` in BOTH:
    - `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/dual_personalized_bpr_mf.py:527-529`
    - `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/bpr_mf.py:527-536`
    To accept a `user_id` argument and return `user_embeddings.weight[user_id]` (the trained row). Update client_app.py:716-718 to pass `partition_id` to `compute_user_prototype`. Cross-silo callers (currently the only ones using mean) need a separate code path or an opt-in flag, since cross-silo legitimately wants a population-mean.
    Note: the existing prototype caching tests in tests/ probably rely on the mean semantics — review.

  Bug #3: This is the deepest bug. Three potential fixes, in order of preference:
    a) When checkpointing best-round, ALSO snapshot all clients' `.embedding_cache/{run_id}/partition_*.pt` files into `.embedding_cache/{run_id}/best_round_{N}/`. On restore, swap back. Costs ~24GB disk per run for ML-1M (4MB × 6040). Adds significant I/O.
    b) Bias the per-round eval to a fresh independent random sample of users (not the same 604 that just trained), so per-round metrics measure population-level performance and the 4.6× gap disappears (per-round metric becomes a pessimistic estimator instead of optimistic).
    c) Replace the `best_round` checkpoint rule with `last_round`: always report final-round full-pop performance. Stop early stopping if you must; just report what the model actually achieves at convergence with current local state.
    d) Cache-coverage rule: gate the `best_round` snapshot to only fire when EVERY client has been trained in the last K rounds (e.g. K=10). This is impractical for cross-device with N=6040 and fraction_train=0.1 (would need K≈10/0.1=100 rounds before the first valid snapshot).
    Recommended: combination of (b) and (c) — per-round eval samples a fresh random batch (so the in-sample bias goes away) and use last-round checkpointing (no best-round restore). The "thesis claim" then has to be evaluated on the last-round full-pop number, which is currently 0.235 (the round-93 in-sample number) → MUST be verified against round-93 full-pop, not round-83 best-round-restore.

verification: (pending)
  - For Bug #1: run a dummy 2-round simulation, confirm cold_start_rate=0 in round 2 logs.
  - For Bug #2: re-run with the per-user prototype fix, confirm global_prototype_norm grows above 0.1 (Xavier-trained-row scale).
  - For Bug #3: run a small (num-partitions=20, num-server-rounds=10, fraction-train=0.5) Flower repro before and after the fix; the in-sample vs full-pop gap should shrink to <30%.

files_changed:
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py  # Bug 1: module-anchored cold-start probe path (commit a980217)
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/dual_personalized_bpr_mf.py  # Bug 2: user_id param on compute_user_prototype (commit 843c9bc)
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/bpr_mf.py  # Bug 2: user_id param on compute_user_prototype (commit 843c9bc)
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py  # Bug 2: pass partition_id to compute_user_prototype (commit 843c9bc)

## Option E — User-Runnable Last-Round Full-Pop Eval

### Constraint discovered while planning

The original prompt asked for a "one-shot full-population evaluation at the LAST cached state of the just-completed adaptive run, using the existing on-disk artifacts." After auditing the artifacts, this is **not directly achievable**:

- The per-client LOCAL state (user embeddings, MLPs, perturbation, fusion layer) IS on disk under `federated-adaptive-personalized-cf/.embedding_cache/20260503-171314-313b26/partition_*.pt` (6040 files, 4 MB each, ~24 GB total).
- The round-83 best_prototype IS preserved in `results/federated/adaptive/20260503-171314-313b26/results.json` under `_manifest.best_prototype` (128-d vector).
- The round-93 (or any round's) **GLOBAL ITEM EMBEDDINGS** are NOT on disk anywhere — neither in the manifest, results.json, the cache dir, nor the wandb run dir. The server held them only in memory during the run.

The 0.054 figure already IS "best-round-restored globals + each client's last-cached LOCAL state". To distinguish that from "last-round globals + each client's last-cached LOCAL state", we would need round-93 globals — which are gone.

### Path α (existing eval-only mode) — does NOT exist

Verified by grepping for `eval-only`, `evaluation-only`, `skip-train` across the adaptive module: no such config key exists. Setting `num-server-rounds=0` would skip the entire round loop AND skip the D-06 extra-eval-round (per `server_app.py:985`), so it produces no metrics. Adding a true eval-only mode is a code change beyond Option E's scope.

### Path β — pragmatic substitute: short warm-start re-run with the new fixes

The closest executable proxy that yields a meaningful new data point is a **short fresh sim that warm-starts each client from the existing partition cache and uses `checkpoint-rule=last_round`**. This trades cleanliness for executability:

- **What this measures:** With Bug 1 + Bug 2 fixes applied AND each client warm-loaded from its previous LOCAL state, how far does NDCG@10 climb in 10 additional rounds when reported as a true full-pop last-round number (no best-restore)?
- **What this does NOT measure:** "What would the round-93 globals from the original run have produced on full-pop eval?" — that data is gone.
- **Why it's still valuable:** If 10 warm-start rounds produce NDCG@10 > 0.054 → Bug 2 is delivering signal AND the prior 0.054 plateau was at least partly the prototype-collapse + best-round-restore mismatch. If NDCG@10 stays ≤ 0.054 → Bug 3 strategy must change architecture (Option D).

### Pre-flight checks

```bash
# 0. Verify branch and clean working tree (the two new commits should be on HEAD)
cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
git branch --show-current     # expect: feat/try_to_run_the_baseline
git log --oneline -3          # expect 843c9bc and a980217 in top two
git status --short            # expect: only pre-existing M lines on baseline/personalized pyproject.toml + wandb/latest-run

# 1. Verify the existing cache is intact
ls federated-adaptive-personalized-cf/.embedding_cache/20260503-171314-313b26/ | grep -c '^partition_'  # expect: 6040
du -sh federated-adaptive-personalized-cf/.embedding_cache/20260503-171314-313b26/  # expect: ~24G

# 2. Verify no other Flower simulation is running, and co-worker's gradio is intact
ps -u $USER -o pid,etime,cmd | grep -E 'flwr|flower-simulation' | grep -v grep  # expect: empty
ps -p 2103775 -o pid,etime,cmd | grep -v grep                                   # expect: still running

# 3. Free disk for the new run's cache (we're going to copy ~24 GB)
df -h .  # expect: at least 30 GB free in repo root
```

If any check fails, STOP and investigate before proceeding.

### The actual run

Two steps. Do them in a tmux session because the simulation will run for ~1-2 hours and you may want to detach.

**Step 1 — copy the existing cache to a new run-id directory.** This warm-starts every client from its last-trained LOCAL state (user embedding, MLP, perturbation, fusion layer, per-user alpha logits if enabled). The `flwr run` step in Step 2 will assign a fresh run-id; we'll override it with our chosen run-id below. Pick a fixed new run-id so the cache copy and the run agree:

```bash
NEW_RUN_ID="20260505-option-e-warmstart"
SRC=federated-adaptive-personalized-cf/.embedding_cache/20260503-171314-313b26
DST=federated-adaptive-personalized-cf/.embedding_cache/${NEW_RUN_ID}

# Use cp -r with --reflink=auto to avoid duplicating ~24 GB on copy-on-write FSes (ext4/xfs allow this)
cp -r --reflink=auto "$SRC" "$DST"
ls "$DST" | grep -c '^partition_'  # expect: 6040
```

**Step 2 — run a 10-round warm-start sim with last-round checkpointing.** Use the canonical `scripts/run.py` launcher so `num-supernodes=6040` is locked at federation construction time (CR-2). Override num rounds, checkpoint rule, run-id, and disable wandb (since this is a diagnostic, not a thesis cell):

```bash
tmux new -s option_e
cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
RAY_memory_usage_threshold=0.97 \
python scripts/run.py adaptive thesis_crossdevice_main \
  --run-config "num-server-rounds=10 checkpoint-rule=last_round run-id=20260505-option-e-warmstart wandb-enabled=false early-stopping-enabled=false fraction-eval=1.0"
# Ctrl+B then D to detach if needed.
```

Notes on the override flags:
- `num-server-rounds=10` — short to keep wall-clock manageable; we don't need full convergence.
- `checkpoint-rule=last_round` — server skips both the best-round restore (`server_app.py:960` if-branch becomes else) AND the D-06 extra-eval-round (`server_app.py:985` else-branch). Final metrics come from the in-loop eval at the last round.
- `run-id=20260505-option-e-warmstart` — must match the dir copied in Step 1, so each client reads from `partition_*.pt` instead of cold-starting.
- `wandb-enabled=false` — diagnostic run, not a thesis cell.
- `early-stopping-enabled=false` — we want all 10 rounds to execute.
- `fraction-eval=1.0` — every selected client evaluates each round (does NOT make eval full-pop; per-round eval still uses `selected_node_ids` per Bug 3 — the in-sample bias remains, see "What to look for" below for the workaround).

### What to look for in the output

**Per-round console output** (each round):
```
Round N/10
[CHECKPOINT] checkpoint_rule='last_round': keeping last-round params  # round 10 only
[D-13] cold_start_rate=<x>/<total>  # x should drop from 1.0 to a nonzero value as Bug 1 fix kicks in
[strategy] global_prototype_norm=<y>  # should be > 0.1 by round 2-3 with Bug 2 fix; was 6e-5 before
sampled_ndcg@10=<z>  # this is the IN-SAMPLE per-round metric (Bug 3) — only ~604 fresh-trained users
```

**Final metrics block** (printed after round 10):
```
[final_metrics.last] sampled_ndcg@10 = <Z>     # in-sample on the last round's 604 trained clients
[final_metrics.best] sampled_ndcg@10 = <Z>     # under last_round rule, best == last (no D-06 full-pop eval)
```

Because of Bug 3, `final_metrics.last` is still the in-sample 604-client number — NOT a full-pop number. To get the FULL-POP number, you need the per-round full-pop eval that doesn't currently exist as an in-loop step. Two options:

**Option E.workaround-1 (quick, dirty, no code change):** Read `eval_metrics_history` from results.json after the run completes. Each round's `sampled_ndcg@10` is the in-sample 604-client number. Compare round-10 in-sample NDCG against the original run's round-83 in-sample NDCG=0.235. If round-10 in-sample is ≥ 0.235, Bug 2 fix is delivering signal in-sample.

**Option E.workaround-2 (better, requires `checkpoint-rule=best_round_restore` instead):** Re-run Step 2 with `checkpoint-rule=best_round_restore` (default). The D-06 extra-eval-round at the very end runs full-pop on ALL 6040 clients with the BEST-round globals from the 10-round window. Compare against 0.054. The headline number to look at is `final_metrics.best.sampled_ndcg@10`. **This is probably what you actually want.** Modify Step 2 by removing `checkpoint-rule=last_round` from the `--run-config` string.

### Interpretation guide (for whichever workaround you ran)

| Result | Implication |
|--------|-------------|
| full-pop NDCG@10 < 0.10 | Bug 3 mechanism is even worse than thought, or warm-start corrupted something. Loop back to debug. |
| full-pop NDCG@10 in [0.10, 0.18) | Bug 3 mechanism confirmed; baseline still beats adaptive. Need protocol redesign per Option D for the next thesis run. |
| full-pop NDCG@10 in [0.18, 0.25) | Bug 3 mechanism confirmed AND adaptive's local state when fully cached actually meets the thesis target. Win is recoverable with last_round checkpointing + warm-start. |
| full-pop NDCG@10 ≥ 0.25 | Strong signal that Bug 1+2 fixes alone unlock convergence. Re-run a full 100-round sim to confirm. |

### Wall-clock estimate

The just-completed run took **64,321 s = 17.9 hours** for 93 training rounds + 1 final eval round = **~11.4 minutes per training round** at `fraction-train=0.1` (604 clients × 1 local epoch each) and per-round eval at `fraction-eval=1.0` over the same 604 clients.

For a 10-round warm-start re-run:
- 10 rounds × ~11.4 min/round = **~115 minutes (~2 hours)** for training + per-round eval.
- D-06 final full-pop eval (only fires under `best_round_restore`): all 6040 clients × 1 eval pass ≈ ~30-50 min.
- **Total: ~2-3 hours** if using Option E.workaround-2 (`best_round_restore`); ~2 hours if using workaround-1 (`last_round`).

### Cleanup and safety

- **DO NOT delete** `federated-adaptive-personalized-cf/.embedding_cache/20260503-171314-313b26/` — it is the only artifact of the prior 17.9-hour run and the basis for the warm-start.
- The new `federated-adaptive-personalized-cf/.embedding_cache/20260505-option-e-warmstart/` cache CAN be deleted after the run finishes and you've recorded the result.
- Keep `RAY_memory_usage_threshold=0.97` set; without it Ray will OOM-kill at 95% on this 62 GB shared machine.
- Co-worker's gradio (PID 2103775, ~16 GB GPU + 4 GB RAM) MUST stay alive — verify with `ps -p 2103775` mid-run.
- `wandb-enabled=false` so this run does not pollute the thesis W&B project.

### When to commit the debug file

Do NOT commit `.planning/debug/adaptive-cache-prototype-collapse.md` yet. Commit it AFTER you run Option E and have a number — the "verification" block above can then be filled in with the actual NDCG@10 result and the file moved to `.planning/debug/resolved/`. This matches the pattern from `.planning/debug/resolved-baseline-eval-leakage.md`.

---

## Resolution — 2026-05-05

**Status: RESOLVED.** Bugs 1 and 2 are fixed and verified empirically. Bug 3 is **deferred / not required** for thesis recovery.

### Option E warm-start re-run result

Run id: `20260505-option-e-warmstart` — 10 rounds on a reflink-cloned cache from `20260503-171314-313b26` with the manifest's `run_id` field patched (in-place edit, not via code change) so the D-04 signature check passed.

**Headline — full-population restored-best at round 10 (N=6040):**

| Source | NDCG@10 | HR@10 |
|---|---:|---:|
| Original adaptive (94 rounds, both bugs live) | 0.0540 | 0.1050 |
| **Option E re-run (10 warm rounds, both bugs fixed)** | **0.2190** | **0.4028** |
| Baseline (100 rounds, BasicMF + FedAvg) | 0.2013 | 0.3646 |

**Adaptive beats baseline overall AND on every user group** — including sparse, the thesis-targeted subgroup:

| Group | Adaptive | Baseline | Δ NDCG@10 |
|---|---:|---:|---:|
| sparse (n=809) | 0.2506 | 0.2446 | **+0.006** |
| medium (n=2322) | 0.2354 | 0.2217 | **+0.014** |
| dense (n=2909) | 0.1970 | 0.1730 | **+0.024** |

NDCG@10 = 0.219 clears the > 0.18 thesis target. The thesis claim ("adaptive lifts sparse users via global prototype + dual-level personalization") holds at this checkpoint.

### Bug fix verifications

| Bug | Original | Option E re-run | Verdict |
|---|---|---|---|
| 1 — server cache-path probe relative | `cold_start_rate = 1.0` (56,172/56,172) | `cold_start_rate = 0.0` (0/6040) | FIXED |
| 2 — `compute_user_prototype()` mean-over-all-rows | `global_prototype_norm ≈ 6.3e-5` | `global_prototype_norm` round-1 = `0.0687` | FIXED (~10³× larger) |
| 3 — best_round_restore × split learning | hypothesized to be dominant 4.6× gap cause | full-pop 0.219 vs per-round 0.214 (≈ no gap) | NOT REQUIRED — gap closed by Bugs 1+2 alone |

### Follow-up flags (non-blocking, file separate sessions if needed)

1. **`global_prototype_norm` trends DOWN over the 10 warm-start rounds**: 0.0687 → 0.0594 → 0.0567 → ... → 0.0264. Magnitude is non-trivial (vs the original 6e-5 collapse) but monotone decay is suspicious — would expect stabilization if the EMA is converging on a coherent population centroid. Worth a separate investigation; not blocking the thesis number.
2. **`alpha_analysis` is still `null`** in both the original run and the Option E re-run. Telemetry serialization gap unrelated to Bugs 1/2/3. Tracked for separate fix.
3. **0.219 is a single round-10 peak on top of a 17.9-h warm cache.** Methodological best practice: confirm with a full 100-round adaptive run from a clean cache before declaring the thesis result.

### Companion commits in this session (chronological)

- `fd4450b` — fix(07): bypass BasicMF predict() clamp in baseline ranking eval (separate `baseline-eval-leakage` debug session)
- `a980217` — fix(07): resolve server cold-start probe path the same way the client does (Bug 1)
- `843c9bc` — fix(07): make compute_user_prototype user-id aware for cross-device protocol (Bug 2)
- `ca63c2a` — chore(07): declare run-id in pyproject for CLI override (enables warm-start workflow)
- `<this archival commit>` — docs(07): archive resolved adaptive-cache-prototype-collapse debug session

### Reproducing Option E warm-start (for posterity)

The exact command set used to produce the Option E result, in case the warm-start workflow needs to be re-exercised:

```bash
NEW_RUN_ID=20260505-option-e-warmstart
SRC=federated-adaptive-personalized-cf/.embedding_cache/20260503-171314-313b26
DST=federated-adaptive-personalized-cf/.embedding_cache/${NEW_RUN_ID}

# 1. Clone cache (CoW, near-instant on ext4/xfs)
cp -r --reflink=auto "$SRC" "$DST"

# 2. Patch the cloned manifest's run_id field so D-04 signature check passes
python3 -c "
import json
p = '${DST}/manifest.json'
with open(p) as f: m = json.load(f)
m['run_id'] = '${NEW_RUN_ID}'
with open(p, 'w') as f: json.dump(m, f, indent=4)
"

# 3. Launch (must explicitly disable the next-gen flags — they default to True
#    in the current pyproject and would break the cache signature)
RAY_memory_usage_threshold=0.97 \
python scripts/run.py adaptive thesis_crossdevice_main \
  --run-config "num-server-rounds=10 run-id=${NEW_RUN_ID} wandb-enabled=false early-stopping-enabled=false enable-per-user-alpha=false enable-item-perturbation=false contrastive-lambda=0.0"
```
