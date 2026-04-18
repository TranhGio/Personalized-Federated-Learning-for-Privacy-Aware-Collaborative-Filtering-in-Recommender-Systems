# Features Research — Correct Cross-Device FedRec Protocol

**Source:** Codex MCP research (2026-04-19) distilled from the published FedRec literature (FedAvg, NCF, FedPer, FedProx, SCAFFOLD, MetaMF, FedRec, FedRecon, Adaptive FedOpt, FedNCF, FedPerGNN, LightFR, PFedRec, FedRAP, CoFedRec, FedCA, GPFedRec, P²FedRec, 2025 personalized-FedRec survey) plus Flower systems practice.

Phase legend for the `Phase` column:
- `P0` Protocol spec (benchmark mode, paper-compat mode, metric contract, aggregation contract)
- `P1` Data/splits (natural partitioning, ID mapping, train/val/test, negative exclusion sets)
- `P2` Server/orchestration (sampling, aggregation, Flower runtime, optimizer/strategy state)
- `P3` Client state/runtime (local params, RNGs, caches, per-user heads, resume)
- `P4` Evaluation/reporting (evaluators, early stopping, checkpoint restore, subgroup metrics)
- `P5` Experiment ops (W&B, manifests, artifacts, throughput telemetry, run isolation)

"Correct cross-device" here means the benchmark protocol the four modules should standardize on. Paper-exact reproduction modes that deviate (e.g., PFedRec full participation) exist only as explicitly labeled compatibility modes, never as the default benchmark.

## Table Stakes — must have or the setup isn't cross-device

| Feature | Description | Complexity | Phase | Basis |
|---|---|---:|---|---|
| Natural user partitioning | `1 user = 1 client` in benchmark mode; clients never contain multiple users. | med | P1 | NCF PFedRec CoFedRec P²FedRec Flower |
| Identity equality | `partition_id == user_id == cache/eval key`, or an explicit stable map making them equivalent. | med | P1 | FedPer FedRecon PFedRec Survey |
| `num-supernodes == 6040` | Simulation client universe equals the ML-1M user universe; the old `num-supernodes=5` default is gone. | low | P0 | PFedRec LightFR CoFedRec P²FedRec Flower |
| Partial participation by default | Default cross-device runs use `C < 1`; `C = 1` is a labeled paper-compat ablation, not the benchmark default. | low | P2 | FedAvg FedProx SCAFFOLD FedOpt Flower |
| Separate train/eval sampling fractions | Training and evaluation sample different configurable client fractions instead of implicitly evaluating whatever trained. | low | P2 | FedAvg FedOpt Flower |
| Stable global ID maps | One persisted `raw_user_id → user_idx` and `raw_item_id → item_idx` mapping consumed by every module, cache, and evaluator. | med | P1 | NCF PFedRec GPFedRec P²FedRec |
| Per-user LOO test split before training | Hold out each user's last interaction before any local training, cache write, or negative generation touches it. | low | P1 | NCF PFedRec CoFedRec P²FedRec FedCA |
| Test item excluded from train negatives | Training negatives for a user must exclude the held-out test positive. | low | P1 | NCF PFedRec CoFedRec P²FedRec |
| All observed positives excluded from eval negatives | Sampled eval negatives exclude all train/val/test positives except the target positive being ranked. | low | P1 | NCF PFedRec CoFedRec P²FedRec |
| Primary evaluation protocol locked | Every benchmark run declares ONE primary protocol (e.g. `sampled_loo_99`) and every compared module uses it. | low | P0 | NCF PFedRec CoFedRec FedCA P²FedRec |
| Final table metrics are protocol-scoped | Metric names encode the protocol: `sampled_ndcg@10` vs `allrank_ndcg@10` are never confused. | low | P4 | NCF CoFedRec FedCA P²FedRec |
| Local/global parameter mask is explicit | Each method publishes exactly which tensors are global, local, reconstructed, or server-only. | med | P0 | FedPer FedRecon PFedRec FedRAP Survey |
| Personalized params stay local | User-local tensors never aggregated unless the paper explicitly defines them as shareable. | med | P2 | FedPer FedRecon PFedRec FedRAP GPFedRec |
| Aggregation weight policy is explicit | `uniform` / `by_positive_count` / `by_training_examples` — method-specific and logged per run. | med | P0 | FedAvg FedPer FedProx FedRecon PFedRec |
| Metric aggregation uses sufficient statistics | Aggregate counts/sums/denominators, not already-averaged per-client HR/NDCG values. | med | P4 | FedAvg FedRecon Survey Flower |
| Final evaluation coverage is explicit | Final benchmark numbers use all eligible eval users or are clearly labeled as sampled-user estimates with coverage counts. | low | P4 | NCF PFedRec CoFedRec P²FedRec |
| Seeds are run-scoped, not round-scoped | Set Python/NumPy/Torch seeds once per run; derive child RNG streams from them rather than reseeding globals every round. | low | P0 | FedProx SCAFFOLD FedOpt Survey |
| Sampling pool is deterministic | Server samples from a stable sorted client universe with a seeded RNG so the run is replayable. | low | P2 | FedAvg FedProx SCAFFOLD FedOpt Flower |
| Client cache is run-namespaced | Every persisted local-state path includes run id plus data/model signature so two experiments cannot silently share state. | med | P3 | FedRecon PFedRec SCAFFOLD FedOpt |
| Cache loads are signature-checked | Loading cached state hard-fails on mismatched method, split hash, `num_users`, `num_items`, or embedding dimension. | med | P3 | FedRecon PFedRec SCAFFOLD FedOpt |
| Negative sampling is user-scoped | Every client samples negatives from that user's unseen catalog only; no pooled cross-user shortcut. | low | P3 | PFedRec FedRAP CoFedRec GPFedRec P²FedRec |
| Runtime only materializes active clients | Cross-device simulation schedules sampled clients on demand rather than instantiating all 6,040 workers eagerly. | med | P2 | FedAvg FedRecon Flower |
| One-user assertion in benchmark mode | Client code asserts exactly one local user in benchmark mode and fails if partition-scope logic leaks through. | low | P3 | PFedRec FedRecon Flower |
| PFedRec per-user head persistence | `affine_output.weight` and `affine_output.bias` stored and restored per user, not per old silo partition. | med | P3 | PFedRec FedRAP |
| Best-round reporting is correct | If early stopping exists, final benchmark numbers come from the restored best round, not the last in-memory round. | med | P4 | FedProx FedOpt PFedRec CoFedRec |
| Round logs carry enough audit data | Every round logs selected client IDs (or counts), weight denominators, evaluated users, and protocol name. | low | P5 | FedAvg FedOpt Flower |

## Differentiating — adds rigor, not mandatory for cross-device correctness

| Feature | Description | Complexity | Phase | Basis |
|---|---|---:|---|---|
| Per-user validation split | Hold out the second-most-recent interaction per user for model selection so test stays untouched. | med | P1 | PFedRec CoFedRec P²FedRec |
| Benchmark mode vs paper-compat mode | Ship explicit modes for `benchmark_cross_device` and method-specific reproduction settings like PFedRec full participation. | low | P0 | PFedRec FedRAP CoFedRec Survey |
| Seeded per-round server sampling | `Random(base_seed + round_idx)` or persisted RNG state so round samples are deterministic and replayable. | low | P2 | FedAvg SCAFFOLD FedOpt Flower |
| Fresh train negatives per round | Re-sample training negatives each round or epoch instead of caching one static set for the whole run. | med | P3 | PFedRec FedRAP CoFedRec GPFedRec |
| Per-user RNG streams | Derive RNG streams from `(run_seed, user_id, round, purpose)` so clients do not collide. | low | P3 | SCAFFOLD FedOpt Survey |
| Best-checkpoint restore includes local state | Restoring the best checkpoint also restores personalized user-local state required for fair evaluation. | high | P4 | FedRecon PFedRec GPFedRec |
| Secondary full-rank eval track | Add all-item ranking as a secondary track, but namespaced and never mixed with the primary sampled protocol. | med | P4 | CoFedRec FedCA P²FedRec |
| Sampled-candidate audit artifact | Persist the exact candidate pools used in sampled eval so a result can be re-ranked and checked offline. | med | P5 | NCF PFedRec CoFedRec |
| Sparse/medium/dense subgroup reporting | Report user-group metrics by interaction count to expose whether gains only come from dense users. | med | P4 | FedCA GPFedRec P²FedRec Survey |
| Macro and micro averages | Publish both per-user average metrics and example-weighted metrics so skewed participation is visible. | low | P4 | Survey |
| Multi-seed confidence intervals | Report mean ± std or CIs across multiple seeds, as many FedRec papers do. | med | P5 | PFedRec CoFedRec LightFR |
| Sampling-exposure tracking | Track how often each user/group is sampled during training and evaluation. | low | P2 | FedAvg SCAFFOLD FedOpt |
| Sparse-user sampling floor | Add a quota or reweighting rule if sparse users are materially under-sampled. | med | P2 | SCAFFOLD FedCA Survey |
| Protocol fingerprint manifest | Save a manifest containing split hash, seeds, eval protocol, aggregation weights, negative counts, runtime config. | low | P5 | Survey Flower |
| Dedicated cross-device W&B project | Separate W&B project namespace so cross-device results cannot be confused with old cross-silo runs. | low | P5 | Ops |
| Unit tests for tensor masks | Tests that assert local tensors are excluded from aggregation and global tensors are present. | med | P5 | FedPer FedRecon PFedRec |
| Unit tests for evaluator equivalence | Check that server-side global evaluation and federated sufficient-stat aggregation agree on the same checkpoint. | med | P5 | FedRecon Flower |
| Resume-safe server optimizer state | Persist FedOpt momentum/variance and any server-side adaptive state when resuming. | med | P5 | FedOpt |
| Resume-safe control variates | Persist SCAFFOLD-style client/server control variates if you compare against drift-corrected baselines. | high | P5 | SCAFFOLD |
| Throughput telemetry | Log round wall time, active workers, queue depth, OOM/retry counts to catch scaling failures early. | low | P5 | Flower |
| Unseen-user reconstruction benchmark | Add a separate FedRecon-style track for stateless personalization on unseen users. | med | P4 | FedRecon |

## Anti-features — deliberately NOT built

| Anti-feature | Why it is excluded | Phase | Basis |
|---|---|---|---|
| Keep `num-supernodes = 5` as the default | Preserving the old cross-silo client universe makes the setup non-cross-device. | P0 | PFedRec CoFedRec P²FedRec Flower |
| Full participation everywhere | Defaulting every method to `fraction-train = 1.0` after migration and calling it representative cross-device FL. | P2 | FedAvg FedProx SCAFFOLD FedOpt |
| Shared project-root `.embedding_cache/` | Letting different runs silently reuse one cache directory. | P3 | FedRecon PFedRec |
| Cache key missing run/data/model signature | Keying caches only by `partition_id` or `user_id`. | P3 | FedRecon PFedRec |
| Silent cache shape fallback | Ignoring mismatched tensor shapes or missing keys and continuing. | P3 | FedRecon SCAFFOLD |
| Unseeded `random.sample(...)` on server | Relying on process-global randomness for client sampling. | P2 | FedAvg SCAFFOLD FedOpt |
| Round-level global reseeding | Calling `random.seed(...)` or `np.random.seed(...)` at the start of each round/eval path. | P0 | SCAFFOLD FedOpt |
| Every client uses `seed=42` | Giving all users identical RNG streams. | P3 | SCAFFOLD FedOpt |
| Train negatives built from train-only positives | Excluding only train positives and leaving held-out positives eligible as negatives. | P1 | NCF PFedRec P²FedRec |
| Test positive appears in training negatives | Poisoning the ranking loss with the user's held-out positive as a negative sample. | P1 | NCF PFedRec CoFedRec |
| Mixed all-items and sampled metrics in one headline | Placing `ndcg@10` and `sampled_ndcg@10` side-by-side without protocol labels. | P4 | NCF CoFedRec FedCA P²FedRec |
| Full-rank metrics labeled paper-comparable | Comparing full-catalog metrics directly against published sampled-LOO+99 numbers. | P0 | NCF PFedRec CoFedRec P²FedRec |
| Report last-round metrics after early stopping | Stopping early at round `t*` and publishing round `T` as the main result. | P4 | FedProx FedOpt |
| Track best metric but not best tensors | Remembering `best_round` without restoring global and local state from that round. | P4 | FedRecon PFedRec |
| Aggregate personalized tensors by accident | Letting user-local heads, embeddings, or recon variables leak into server aggregation. | P2 | FedPer FedRecon PFedRec |
| Omit PFedRec bias from local-state handling | Treating `affine_output.weight` as local but forgetting `affine_output.bias`. | P3 | PFedRec FedRAP |
| Leave partition-scope loops in user-scope clients | Keeping "many users per client" logic alive in benchmark mode without assertions. | P3 | FedRecon PFedRec |
| Instantiate all 6,040 clients eagerly | Spinning up one worker/process per user before sampling. | P2 | FedAvg Flower |
| One-client-one-vote as hidden default | Letting equal client weights survive by accident after switching to per-user clients. | P0 | FedAvg FedPer FedProx |
| Use test for tuning | Driving early stopping, sweeps, or LR selection from test metrics. | P4 | FedRecon PFedRec CoFedRec |
| Recreate identical 99 negatives every eval | Reseeding inside the evaluator so every checkpoint is scored on the same accidental RNG path unless explicitly frozen and recorded. | P4 | NCF PFedRec |
| Let user remapping drift across modules | Recomputing user indices independently in dataset, cache, and evaluation code. | P1 | FedPer FedRecon PFedRec |

## Current Repo Anti-feature Audit (direct code citations)

The following anti-features are currently present in the codebase and must be corrected by the migration:

- Cross-silo defaults in all four `pyproject.toml` files: `federated-baseline-cf/pyproject.toml:92`, `federated-personalized-cf/pyproject.toml:92`, `federated-adaptive-personalized-cf/pyproject.toml:207`, `federated-pfedrec/pyproject.toml:85`.
- Unseeded server-side client sampling in `federated-baseline-cf/federated_baseline_cf/server_app.py:297`, `federated-personalized-cf/federated_personalized_cf/server_app.py:303`, `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/server_app.py:371`, `federated-pfedrec/federated_pfedrec/server_app.py:250`.
- Global RNG reseeding inside sampled evaluators: `federated-baseline-cf/federated_baseline_cf/task.py:721`, `federated-personalized-cf/federated_personalized_cf/task.py:756`, `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/task.py:918`.
- Shared project-root `.embedding_cache/` paths: `federated-pfedrec/federated_pfedrec/client_app.py:61`, `federated-personalized-cf/federated_personalized_cf/client_app.py:49`, `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py:60`.
- Dual evaluators called in the same client path: `federated-baseline-cf/federated_baseline_cf/client_app.py:170`, `federated-personalized-cf/federated_personalized_cf/client_app.py:387`, `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py:616`.

## References

- See `PITFALLS.md` for how each anti-feature tends to manifest and how to detect it early.
- See `.planning/codebase/CONCERNS.md` for additional already-catalogued bugs in the current code.
- Paper-level references cited by short tag (PFedRec, FedRecon, …); full citations in `SUMMARY.md`.
