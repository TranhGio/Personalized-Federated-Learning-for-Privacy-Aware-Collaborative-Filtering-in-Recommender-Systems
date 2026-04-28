# PFR-02 Reference Audit — federated-pfedrec vs IJCAI-23-PFedRec

> Closes ROADMAP §Phase 5 SC-1. CONTEXT.md §decisions ARE the locked outcomes;
> this file is the human-readable cross-walk traced row-by-row to a specific
> reference line and a specific Flower line.

**Phase:** 05-pfedrec-migration-reproduction
**Authored:** 2026-04-28
**Status:** Closed (every row carries a Decision + Rationale + CONTEXT D-XX pin)

## Anchors

The three reference lines that drive the highest-impact decisions:

- `IJCAI-23-PFedRec/engine.py:143` — `del round_participant_params[user]['affine_output.weight']` (D-01: bias travels and is aggregated server-side).
- `IJCAI-23-PFedRec/engine.py:81` — `self.server_model_param[key].data / len(round_user_params)` (D-24: uniform weight = 1 per participating client).
- `IJCAI-23-PFedRec/engine.py:195-196` — `ratings_pred = torch.cat((test_score, negative_score))` followed by BCE over the 100-item pool (D-04: eval BCE includes positives + 99 negs).

## Audit Table

| Topic | Reference Behavior (engine.py:LINE) | Flower Current (file:LINE) | Decision | Rationale | CONTEXT D-XX |
|-------|--------------------------------------|----------------------------|----------|-----------|---------------|
| 1. `affine_output.bias` classification | GLOBAL — `engine.py:143` deletes only `affine_output.weight` from `round_participant_params`; bias is aggregated server-side via `aggregate_clients_params` (`engine.py:66-81`) | LOCAL — `federated-pfedrec/federated_pfedrec/strategy.py:19-22` (pre-Phase-5) lists both weight + bias in `LOCAL_PARAM_KEYS` | align-to-reference | Closes CONCERNS divergence #9; the headline lever for landing PFR-08 within ±2 points; without it the bias never aggregates and the per-user score functions cannot recover the reference HR/NDCG. | D-01 |
| 2. Aggregation weight policy | uniform — `engine.py:81` divides by `len(round_user_params)`; every contributing user weight=1 | num_examples-weighted (inherited from FedAvg) — sparse users contribute less | align-to-reference | `weight_policy="uniform"` in `_PAPER_COMPAT_PFEDREC` profile + sufficient-stat `aggregate_evaluate` override; on the FIT side, `FitRes.num_examples = 1` per client (Pitfall 5 Option B) makes existing FedAvg num_examples-weighted aggregation mathematically uniform without overriding `aggregate_fit`. | D-24, D-25 |
| 3. Per-round client participation | full — `engine.py:87-91` + `train.py:14` `clients_sample_ratio=1.0` | `fraction_train` config-driven; cross-silo defaults previously diverged | align-to-reference | `fraction_train=1.0` locked in profile; required for PFR-08 ±2 reproduction; partial participation injects sampling variance the reference does not have. | D-06 |
| 4. Eval BCE loss scope | computed over (positive + 99 negatives) — `engine.py:195-196` `ratings_pred = torch.cat((test_score, negative_score))` | computed on positives only — `federated-pfedrec/federated_pfedrec/task.py:432` | align-to-reference | Diagnostic alignment so eval BCE is directly comparable to reference logs; HR@10 / NDCG@10 (the thesis numbers) are unaffected. | D-04 |
| 5. Training-negative resampling | per-round — reference's `store_all_train_data` (called inside `train.py`'s round loop) re-samples via `random.sample` each round | static — `federated-pfedrec/federated_pfedrec/task.py:130` `rng = random.Random(seed)` re-seeded per call → frozen across rounds | align-to-reference | Closes CONCERNS bug #5; replaces stdlib `random.Random(seed)` with `np_rng(run_seed, user_idx, round_num, "train_neg")` per FND-06; gives byte-identical reruns under fixed seed. | D-02, PFR-07 |
| 6. Held-out test positive in training-negative pool | leak — reference's `_sample_negative` (`data.py:75-81`) operates on `interacted_items` which DOES include the held-out test item | leak — `federated-pfedrec/federated_pfedrec/task.py:137-142` builds from trainloader only | align-to-reference (BUT strictly stricter — FND-03 fixes the leak that BOTH codebases share) | Reference's behavior is wrong per modern FedRec literature; FND-03 `ExclusionTable.for_user(user_idx)` mandates removing the held-out test positive from the training-negative pool. PFR-04 enforces this. | PFR-04, FND-03 |
| 7. Server-side client-sampling RNG | unseeded — `engine.py:89-91` uses unseeded `random.sample` | unseeded — `federated-pfedrec/federated_pfedrec/server_app.py:250` uses unseeded `random.sample` | strictly-better-than-reference | Replace with `_server_sampler = server_rng(run_seed)` (FND-06) and partition-id-space sampling (G-03-01); enables byte-identical reruns under fixed seed (PFR-06). | PFR-06 |
| 8. Best-round checkpoint / metric reported | best validation HR@10 — `train.py:123-125` `if val_hit_ratio >= best_val_hr: final_test_round = round` | last-round metrics; early-stopping records `best_round` but does not restore arrays (CONCERNS bug #7) | align-to-reference (with adaptation) | Reference uses val split; we don't (D-08). Carry forward Phase 2/3/4 D-27 in-memory best-round-restore against `sampled_ndcg@10` on the test set. Documented information leak accepted in this thesis cycle; val-split deferred to v2. | D-08, D-13 |
| 9. `affine_output` init scheme | Kaiming default — `mlp.py` uses `nn.Linear` defaults (no Xavier) | Kaiming default — `pfedrec_mlp.py` uses `nn.Linear` defaults (no Xavier reset) | already-aligned | PFR-08 reproduction is sensitive to init scale (RecSys 2024 reports ~50% variance with poor init). The cross-module Xavier reset used by BPR-MF / BasicMF / DualPersonalizedBPRMF is intentionally NOT mirrored here per D-19 — paper-faithfulness wins. | D-19 |

## Closure Note

- ROADMAP §Phase 5 SC-1 reads: *"A diff table comparing Flower PFedRec to `IJCAI-23-PFedRec/` … exists in the repository with a keep-flower or align-to-reference decision and rationale for every row."* Every row above carries one of {`align-to-reference`, `align-to-reference (with adaptation)`, `align-to-reference (BUT strictly stricter — FND-03 fixes the leak that BOTH codebases share)`, `strictly-better-than-reference`, `already-aligned`} as its Decision column. SC-1 is closed.
- ROADMAP §Phase 5 SC-2 reads: *"each user's `(affine_output.weight, affine_output.bias)` is persisted/restored as one atomic per-user artifact keyed by stable `user_idx`."* Reconciliation with D-01 (bias is GLOBAL): the atomicity contract is preserved (per-round, per-user) but the bias channel moves from per-user disk to server-side aggregation per `engine.py:143`. Plan 03 carries the explicit reconciliation note in the cache-layout task; Plan 04 server_app surfaces it in the `_manifest` block. The verifier must accept this reconciliation when evaluating SC-2.
- For each row, the CONTEXT D-XX column points at the canonical decision in `.planning/phases/05-pfedrec-migration-reproduction/05-CONTEXT.md`. Locked decisions are NON-NEGOTIABLE; this audit document does not introduce new decisions, it only cross-walks them to specific reference + Flower lines.
