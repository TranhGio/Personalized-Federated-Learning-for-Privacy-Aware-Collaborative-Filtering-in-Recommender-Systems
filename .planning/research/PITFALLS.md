# Pitfalls Research — Cross-Device FedRec Migrations

**Source:** Codex MCP research (2026-04-19), reinforced by the existing `.planning/codebase/CONCERNS.md`. Each pitfall gives warning signs, prevention strategy, the phase that must address it, and reference tags.

Phase legend: P0 protocol spec · P1 data/splits · P2 server/orchestration · P3 client state · P4 eval/reporting · P5 ops.

---

## 1. `num-supernodes=5` survives the migration

**Warning signs:** Round logs show 5 clients available or selected. A client owns many users, so per-client eval reports multiple test users. Current repo evidence: all four `pyproject.toml` files still pin `options.num-supernodes = 5`.

**Prevention:** Add a benchmark-mode startup assertion — `num_partitions == 6040` and `users_per_client == 1`. Keep `paper_compat_*` modes separate from `benchmark_cross_device`; never overload one config for both.

**Phase:** P0 · **Refs:** PFedRec CoFedRec LightFR P²FedRec Flower

---

## 2. Evaluation protocol drifts during the move from cross-silo to cross-device

**Warning signs:** Same checkpoint looks much better or worse solely because one module switched from sampled LOO+99 to all-item ranking. Result JSONs or W&B dashboards contain both `ndcg@10` and `sampled_ndcg@10`, but only one is used in the thesis table.

**Prevention:** Define a single `primary_eval_protocol` enum and route all table-generation code through it. If you also run full-rank evaluation, namespace it as `allrank_*` and exclude it from apples-to-apples comparisons with published FedRec baselines.

**Phase:** P0 · **Refs:** NCF PFedRec CoFedRec FedCA P²FedRec

---

## 3. Train-negative sampling leaks the held-out test positive

**Warning signs:** Negative pools are built as `all_items - train_positives`; the test item is not excluded because it is absent from training. Ranking quality degrades mysteriously even when the model appears to fit training. In this repo, PFedRec's `prepare_user_train_data` builds the negative pool from `all_items - user_positives` where `user_positives` is train-only: `federated-pfedrec/federated_pfedrec/task.py:108`. Same pattern in the other three modules' `task.py`.

**Prevention:** Precompute `exclude_items[user] = train_pos ∪ val_pos ∪ test_pos` before any negative sampling. Add a unit test that asserts the user's LOO test item never appears among sampled training negatives.

**Phase:** P1 · **Refs:** NCF PFedRec CoFedRec P²FedRec

---

## 4. Aggregation is still "one client, one vote"

**Warning signs:** Every selected user gets the same aggregation coefficient regardless of whether they have 20 or 2,000 interactions. Strategy logs show only the number of selected clients, never the per-client denominator. Metrics shift materially when a few dense users are sampled, but the server weight policy is undocumented.

**Prevention:** Encode aggregation as an explicit policy: `uniform` / `num_positives` / `num_training_examples`. Use `num_positives` (or `num_examples`) for FedAvg/FedProx/FedOpt-style baselines; only use `uniform` when reproducing a paper that explicitly does so, and label it.

**Phase:** P2 · **Refs:** FedAvg FedPer FedProx FedRecon PFedRec

---

## 5. The server averages already-averaged ranking metrics

**Warning signs:** `sampled_num_users` becomes non-integer in experiment summaries (impossible if it is a real count). Per-client HR/NDCG are averaged directly instead of aggregating hits, NDCG sums, and user counts.

**Prevention:** Clients return sufficient statistics: `hit_count@k`, `ndcg_sum@k`, `mrr_sum`, `evaluated_users`. Server computes final metrics exactly once from summed numerators and denominators.

**Phase:** P4 · **Refs:** FedAvg FedRecon Survey Flower

---

## 6. Flower simulation throughput tanks because 6,040 clients are materialized eagerly

**Warning signs:** Startup time explodes before round 1. Memory spikes as if all clients are alive simultaneously. CPU/GPU saturated even when `fraction-train` is small.

**Prevention:** Let Flower manage a large virtual client universe with small training and evaluation fractions; only sampled clients run concurrently. Cap client resources and log active-worker counts per round.

**Phase:** P2 · **Refs:** FedAvg FedRecon Flower

---

## 7. `.embedding_cache/` silently contaminates later experiments

**Warning signs:** A fresh run starts with implausibly good round-1 metrics. Deleting `.embedding_cache/` changes outcomes without any code change. Current repo evidence: project-level cache roots in the three split/personalized modules and PFedRec.

**Prevention:** Namespace cache by `run_id/method/num_users/num_items/dim/split_hash`. Require an explicit resume flag before any cache read; otherwise start clean.

**Phase:** P3 · **Refs:** FedRecon PFedRec

---

## 8. Cached local state from another `num_users`/shape regime is loaded anyway

**Warning signs:** Cache loads succeed with missing keys, silent fallbacks, or shape warnings. After changing from 5 grouped clients to 6,040 user clients, local-state cardinality or tensor shapes no longer match but the code continues. Resumed runs diverge unexpectedly after changing embedding dim or model head.

**Prevention:** Persist a cache manifest with method name, split hash, `num_users`, `num_items`, tensor shapes, schema version. Hard-fail on mismatch; do not "best effort" load personalized state.

**Phase:** P3 · **Refs:** FedRecon PFedRec SCAFFOLD FedOpt

---

## 9. PFedRec's per-user `affine_output` layer is misclassified or miskeyed

**Warning signs:** `affine_output.weight` is kept local but `affine_output.bias` is forgotten, uploaded, or not restored. Cache keys still reflect old partition semantics, so a user can load the wrong local head. Current repo evidence: PFedRec treats both tensors as local in `federated-pfedrec/federated_pfedrec/strategy.py:20` but the cache layout is still partition-centric in `federated-pfedrec/federated_pfedrec/client_app.py:52`.

**Prevention:** Treat `affine_output.weight` and `affine_output.bias` as one atomic per-user artifact keyed by stable `user_idx`. Add tests that confirm those tensors never enter server aggregation and are restored for best-checkpoint evaluation. Cross-check against the IJCAI reference — `IJCAI-23-PFedRec/engine.py:143` deletes only `affine_output.weight` before aggregation, so the reference DOES aggregate the bias globally. This is a semantic choice that must be explicitly decided, not left implicit.

**Phase:** P3 · **Refs:** PFedRec FedRAP FedRecon

---

## 10. The old global user embedding table survives per-user migration

**Warning signs:** Each client still instantiates an embedding table of shape `num_users × d` although it represents one user. Clients upload thousands of untouched user rows or massive sparse arrays. Memory and network costs scale with global user count instead of local-user count.

**Prevention:** Collapse user-specific state to one local row or one local head per client. If you keep a global-user-table baseline for reproduction, use sparse row masking and aggregate only the visited row.

**Phase:** P3 · **Refs:** FedPer FedRecon PFedRec LightFR

---

## 11. Partition-scope code paths remain in user-scope clients

**Warning signs:** Client logic loops over multiple local users even in benchmark mode. `sampled_num_users` per client often > 1 when it should be 1. PFedRec client evaluation currently iterates over `user_test_items.keys()` inside a client: `federated-pfedrec/federated_pfedrec/client_app.py:354`.

**Prevention:** In benchmark mode, assert exactly one raw user in every client loader and short-circuit otherwise. Keep old partition-wide loops only in explicit cross-silo compatibility mode.

**Phase:** P3 · **Refs:** FedRecon PFedRec Flower

---

## 12. Server-side client sampling is not seeded

**Warning signs:** Same config and seed produce different selected-client sequences and different `best_round`s. The code calls `random.sample(node_ids, num_selected)` without a dedicated server RNG. Current repo evidence: unseeded sampling in all four `server_app.py` files.

**Prevention:** Create a server RNG from the run seed and derive per-round samples deterministically. Persist or log the selected client ids for each round so the run is replayable.

**Phase:** P2 · **Refs:** FedAvg FedProx SCAFFOLD FedOpt Flower

---

## 13. Every client uses the same RNG stream

**Warning signs:** Different clients generate identical negative items in the same order. Training traces across users look unnaturally synchronized. Client-specific randomness is effectively `seed=42` everywhere.

**Prevention:** Derive independent RNG streams from `(run_seed, user_id, round, purpose)`. Keep separate streams for model init, train negatives, eval negatives, any local augmentation.

**Phase:** P3 · **Refs:** SCAFFOLD FedOpt Survey

---

## 14. The evaluator reseeds to the same value every time

**Warning signs:** Evaluator contains `random.seed(seed)` internally. Sampled candidate pools are identical on every eval call unless code elsewhere mutates the RNG. Current repo evidence: sampled evaluators in baseline, personalized, and adaptive modules reseed internally.

**Prevention:** Pass an RNG object into evaluation instead of reseeding globals. If you want fixed eval candidates for a paper-compat mode, generate them once at split-build time and save them with the split manifest.

**Phase:** P4 · **Refs:** NCF PFedRec CoFedRec P²FedRec

---

## 15. Partial participation starves sparse users

**Warning signs:** Exposure histograms show sparse users are sampled rarely. Sparse/medium/dense subgroup NDCG@10 is far noisier than overall NDCG@10. Early stopping decisions dominated by dense users.

**Prevention:** Track per-user and per-group sampling exposure; publish the support counts. Add a floor or stratified sampling rule for sparse users, or evaluate all users on checkpoints used for model selection.

**Phase:** P2 · **Refs:** FedAvg SCAFFOLD FedOpt FedCA Survey

---

## 16. Early stopping records `best_round` but does not restore the best checkpoint

**Warning signs:** `best_round` is logged, but the final evaluation runs on the last model in memory. `final/sampled_ndcg@10` is worse than `best_sampled_ndcg@10` without an explicit statement that "final" means last round. Personalized methods restore global arrays but not local per-user state.

**Prevention:** Save and restore all state required for evaluation: global parameters, local personalized state, any strategy state that affects evaluation. Run one final post-restore evaluation and write that to the canonical result artifact.

**Phase:** P4 · **Refs:** FedRecon FedOpt PFedRec

---

## 17. Last-round metrics are reported when early stopping found a better round

**Warning signs:** Headline number in the thesis table matches the last round, not the best round. Result filenames encode total rounds but not `best_round`. W&B or JSON contains both `best_*` and `final_*`, but downstream scripts use `final_*` by default.

**Prevention:** Promote `best_*` metrics to the canonical table fields and keep `last_*` as diagnostics only. Include `best_round` in filenames, manifests, and comparison scripts.

**Phase:** P4 · **Refs:** FedProx FedOpt PFedRec CoFedRec

---

## 18. The test split is used for model selection

**Warning signs:** Test NDCG@10 drives patience, sweep selection, or LR scheduling. No validation split exists, but early stopping is still enabled. The best paper-looking test number appears during tuning, then never again.

**Prevention:** If you need early stopping, create a per-user validation split or a user-level validation fold. Freeze hyperparameters before touching the final test once.

**Phase:** P4 · **Refs:** FedRecon PFedRec CoFedRec P²FedRec

---

## 19. `evaluate_ranking` and `evaluate_ranking_sampled` are mixed across modules

**Warning signs:** Some modules optimize or select checkpoints on `ndcg@10`, others on `sampled_ndcg@10`. Current repo evidence: baseline, personalized, and adaptive client apps call both evaluators in the same path. Comparison scripts flatten these into one leaderboard.

**Prevention:** Choose ONE evaluator as the benchmark primary; every module uses the same one for table numbers and checkpoint selection. Keep the other evaluator in a secondary namespaced metric family and exclude it from apples-to-apples claims.

**Phase:** P4 · **Refs:** NCF PFedRec CoFedRec P²FedRec

---

## 20. Full-rank and sampled metrics are presented as comparable

**Warning signs:** One module's `NDCG@10` is an all-item rank, another's is `1 positive + 99 negatives`, and both appear in the same row. A migration note says "cross-device got worse" but the evaluator changed at the same time.

**Prevention:** Embed the protocol in the metric name, artifact name, chart legend. Never compare across protocols inside one ablation or one thesis table.

**Phase:** P0 · **Refs:** NCF CoFedRec FedCA P²FedRec

---

## 21. User-ID mapping drifts across split, cache, and evaluation

**Warning signs:** Cache hits disappear after restart even though the run id is unchanged. Same raw MovieLens user ends up with different `user_idx` values across modules. Filtering users with too few interactions changes partition order and silently breaks cache identity.

**Prevention:** Persist one canonical `raw_user_id → user_idx` mapping artifact and import it everywhere. Key caches and evaluators by canonical `user_idx`, not local partition order.

**Phase:** P1 · **Refs:** FedPer FedRecon PFedRec P²FedRec

---

## 22. Resume logic restores model weights but not strategy state

**Warning signs:** FedOpt resumes with the same model weights but very different convergence because momentum/variance state was dropped. SCAFFOLD resumes but client/server control variates reset to zero. Reported "resume support" only covers arrays, not optimizer state.

**Prevention:** Version and checkpoint all strategy state alongside model arrays. Disallow resume across changes to user universe, split manifest, or sampling protocol.

**Phase:** P5 · **Refs:** SCAFFOLD FedOpt FedRecon

---

## 23. Timestamp ties make leave-one-out splitting nondeterministic

**Warning signs:** Re-running the same preprocessing yields a different held-out item for some users. Different pandas/python versions change which same-timestamp interaction becomes test.

**Prevention:** Stable-sort by a deterministic key like `(user_id, timestamp, movie_id)`. Persist the split manifest once and reuse it across all methods.

**Phase:** P1 · **Refs:** NCF PFedRec P²FedRec

---

## 24. Paper deviations are real but undocumented

**Warning signs:** A run is labeled "PFedRec" but changes participation fraction, eval mode, negative count, aggregation policy, or checkpoint rule. Months later you cannot explain why your number differs from the paper.

**Prevention:** Store a protocol fingerprint in every result artifact: partition mode, client fractions, aggregation weight policy, eval protocol, negative counts, seeds, checkpoint rule. Use explicit names such as `benchmark_cross_device` and `paper_compat_pfedrec_c1`.

**Phase:** P0 · **Refs:** Survey PFedRec FedRAP CoFedRec FedCA P²FedRec

---

## Additional Pitfalls Specific to the Adaptive Module

### 25. Per-user learned alpha silently re-initializes every round

Already documented in `.planning/codebase/CONCERNS.md`. The current `federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` loads local params before `enable_per_user_alpha()` has constructed them, so the cached `_logit_alpha.weight` is never restored. Fix: call `enable_per_user_alpha` / `enable_item_perturbation` BEFORE `load_local_user_embeddings`.

**Phase:** P3 · **Refs:** PFedRec (same cache-load-ordering pattern)

### 26. Server prototype EMA does not survive best-round restore

**Warning signs:** After restoring to `best_round`, the adaptive module's `p_global` EMA state still reflects the last-round value, silently biasing the final evaluation.

**Prevention:** Save `p_global` as part of the best-round checkpoint; restore it alongside the arrays.

**Phase:** P4 · **Refs:** FedRecon PFedRec

---

## References

- Paper tags as in `FEATURES.md`; full citations in `SUMMARY.md`.
- Local evidence: `.planning/codebase/CONCERNS.md`.
- Codex MCP research session: 2026-04-19.
