---
status: resolved
trigger: "federated-baseline-cf reports Sampled HR@10 = Sampled NDCG@10 = 1.0000 every round during a thesis_crossdevice_main run, while same protocol on adaptive returns realistic ~0.31 / ~0.18"
created: 2026-05-04T10:10:00Z
updated: 2026-05-04T11:30:00Z
human_verify_passed: true
---

## Current Focus

hypothesis: RESOLVED. `BasicMF.predict()` clamps to [1.0, 5.0]; tied scores under that clamp degenerate ranking eval. Fix bypasses the clamp inside `evaluate_ranking_sampled` via class-name dispatch on raw `forward()`.
test: Live multi-round Flower run confirms recovery — baseline NDCG@10 climbs realistically (0.1213 at round 12/100); adaptive untouched and healthy (0.2110 at round 73/100).
expecting: Session archived; fix shipped; downstream consumers (rating-prediction RMSE/MAE path) preserved.
next_action: COMPLETE. Session archived to `.planning/debug/resolved-baseline-eval-leakage.md`. Fix committed atomically on `feat/try_to_run_the_baseline`.

## Symptoms

expected: Baseline BPR-MF on NCF protocol (LOO + 99 neg) should yield HR@10 ~0.6-0.75, NDCG@10 ~0.15-0.25 after ~50 rounds. Adaptive on same protocol at round 54: HR@10 ~0.31, NDCG@10 ~0.18 — sane.
actual: Baseline returns Sampled HR@10 = Sampled NDCG@10 = 1.0000 from at least round 51 onward, sustained, not transient.
errors: No exceptions — metric computes cleanly, just always equals 1.0
reproduction: |
  1. cd /home/bes/Desktop/vinh/federated-learning/movie-recommendation-system
  2. git checkout feat/try_to_run_the_baseline
  3. python scripts/run.py baseline thesis_crossdevice_main --run-config "early-stopping-enabled=false num-server-rounds=2"
  4. Observe Sampled HR@10/NDCG@10 = 1.0000 in stdout
started: Visible after D-01 alignment fix unblocked eval pipeline. Likely always-present bug, masked by prior bugs (Fix 1 evaluator mode, Fix 2/3 run_id propagation).

## Eliminated

- hypothesis: H1 (off-by-one — only positive scored, no negatives concatenated)
  evidence: line 1243 of baseline task.py concatenates [positive] + 99 negatives correctly. Functionally identical to adaptive task.py:1116. Direct probe shows 100 candidates being scored.
  timestamp: 2026-05-04T10:25:00Z
- hypothesis: H2 (positive item leaks into negative pool)
  evidence: Direct probe shows test item 47 is NOT in trainloader; FND-03 exclude_items contains both train items and the held-out test positive (53 items total = 52 train + 1 test); all_user_items in eval = train ∪ test ∪ exclude is correctly disjoint from sampled negatives.
  timestamp: 2026-05-04T10:30:00Z
- hypothesis: H3 (test item leaks into trainloader)
  evidence: Verified at dataset level — train_items={...} (52 items), test_items={47}, intersection is empty.
  timestamp: 2026-05-04T10:30:00Z
- hypothesis: H4 (model collapsed to constant prediction giving HR=10/100=0.1)
  evidence: Disproved by observation that HR=1.0 (not 0.1). But this hypothesis is closely related to the actual root cause.
  timestamp: 2026-05-04T10:30:00Z

## Evidence

- timestamp: 2026-05-04T10:25:00Z
  checked: evaluate_ranking_sampled in both baseline and adaptive task.py, side-by-side
  found: The two functions are functionally identical — same candidate construction, same sort, same scoring loop. Bug is NOT in this function.
  implication: Bug must be upstream (in inputs to the eval function) or in the model's predict() method.
- timestamp: 2026-05-04T10:30:00Z
  checked: Direct probe of evaluate_ranking_sampled with fresh untrained BPRMF (embedding_dim=128) on user 0
  found: HR@10=0.0, NDCG@10=0.0, MRR=0.0147 (positive ranks 68th of 100). Eval pipeline functionally correct in isolation.
  implication: Bug is NOT in eval data construction. Suspicion shifts to model.predict() behavior.
- timestamp: 2026-05-04T10:35:00Z
  checked: federated-baseline-cf/pyproject.toml [tool.flwr.app.config]
  found: `model-type = "basic"` is the pyproject default (BasicMF, not BPRMF). The thesis run uses `model-type=basic` because thesis_crossdevice_main mode profile does NOT override model-type, and pyproject value wins over fallback.
  implication: BasicMF is the model actually being trained/evaluated. Need to inspect BasicMF.predict().
- timestamp: 2026-05-04T10:38:00Z
  checked: BasicMF.predict() at federated-baseline-cf/federated_baseline_cf/models/basic_mf.py:135-151
  found: predict() does `predictions = torch.clamp(predictions, min=1.0, max=5.0)` before returning. With Xavier-initialized embeddings outputting scores in roughly [-0.03, 0.03], every score is below 1.0 and gets clamped to EXACTLY 1.0.
  implication: All 100 candidates tie at score=1.0. evaluate_ranking_sampled then sorts via Python's stable list.sort — the positive (always at input index 0) keeps rank 1.
- timestamp: 2026-05-04T10:40:00Z
  checked: Direct probe — initialized BasicMF on user 0, ran evaluate_ranking_sampled BEFORE any training
  found: HR@10=1.0000, NDCG@10=1.0000, MRR=1.0000. predict() output: min=1.0, max=1.0, std=0.0, unique values=[1.0]. forward() output (pre-clamp): min=-0.0318, max=0.0218 — confirming clamp degenerates 100 distinct floats into a single value.
  implication: Bug reproducible WITHOUT any training, isolated entirely to BasicMF.predict()'s clamp.
- timestamp: 2026-05-04T10:42:00Z
  checked: Same probe after 1 epoch of training (BasicMF, lr=0.005, MSE loss)
  found: train_loss=17.96 (model trying to predict ratings 4-5 from initial near-zero output → huge loss; learns slowly under L2). Eval still HR@10=1.0. predict() still returns all 1.0.
  implication: Even after training the clamp dominates because the model's pre-clamp scores are still small. The bug is permanent for any state where forward() < 1.0.
- timestamp: 2026-05-04T10:43:00Z
  checked: Sibling implementations BPRMF.predict (baseline) and BasicMF.predict (personalized, adaptive)
  found: BPRMF.predict returns raw forward scores (no clamp) — that's why adaptive's `model-type=bpr` run is healthy. BasicMF.predict in BOTH personalized and adaptive ALSO clamps to [1, 5] (same bug, dormant because their pyproject defaults are bpr/dual).
  implication: Fix needed in baseline; siblings are unaffected for the current thesis runs but should also be fixed defensively. Per constraint, do NOT touch adaptive — leave its dormant copy alone.

## Resolution

root_cause: |
  `BasicMF.predict()` clamps predictions to [1.0, 5.0] for rating-prediction (RMSE/MAE).
  `evaluate_ranking_sampled` calls `model.predict()` for ranking. With Xavier init the
  model's forward() scores fall well below 1.0, so all 100 candidates clamp to exactly
  1.0. Python's stable `list.sort` keeps the positive (always at input index 0) at
  rank 1, yielding HR@10 = NDCG@10 = MRR = 1.0 every round.

  Adaptive is unaffected because its pyproject defaults to `model-type=bpr`, and
  `BPRMF.predict()` returns raw forward scores without clamping.
fix: |
  In `federated-baseline-cf/federated_baseline_cf/task.py` at the score-computation
  step inside `evaluate_ranking_sampled` (around line 1245-1263), replaced the call
  `model.predict(user_tensor, item_tensor)` with a class-name dispatch that uses
  raw `forward()` instead — preserving BasicMF's predict()/clamp behavior for the
  rating-prediction (RMSE/MAE) path while ensuring ranking eval sees uncoerced
  scores. BPRMF.forward needs `neg_item_ids=None`; BasicMF.forward takes only
  (user, item).

  Concretely:
      if type(model).__name__ == "BPRMF":
          candidate_scores = model(user_tensor, item_tensor, neg_item_ids=None)
      else:
          candidate_scores = model(user_tensor, item_tensor)

  Did NOT remove the clamp from BasicMF.predict() — that path is still consumed by
  the rating-prediction RMSE/MAE eval (`task.test()`) and any external consumer
  who explicitly wants clamped 1-5 outputs. The fix is surgical: only the ranking
  call path bypasses the clamp.

  Did NOT touch federated-personalized-cf or federated-adaptive-personalized-cf
  per the constraint that adaptive must not be modified during the active thesis
  run. Their dormant clamp bugs (in their own BasicMF.predict) remain — but they
  default to `model-type=bpr|dual`, so they are not exposed.
verification: |
  1) Direct unit-style probe (no Flower):
     - Untrained BasicMF, user 0: HR@10 = 0.0000 (was 1.0000), MRR = 0.0185 — chance-level.
     - Untrained BPRMF, user 0: HR@10 = 0.0000 (unchanged), MRR = 0.0132 — regression-clean.
     - Trained BasicMF (10 epochs MSE on user 0): HR@10 = 0.0000, MRR = 0.0135 — eval no longer pinned to 1.0; loss decreases as expected.
     - Avg over 50 random users (untrained BasicMF): HR@10 = 0.16, NDCG@10 = 0.077 — exactly the expected chance-level distribution for 1-of-100 (theoretical 0.10).
  2) End-to-end Flower 1-round run:
     - `python scripts/run.py baseline thesis_crossdevice_main --run-config "early-stopping-enabled=false num-server-rounds=1 wandb-enabled=false fraction-train=0.005"`
     - Result: Sampled HR@10 = 0.0333, Sampled NDCG@10 = 0.0105 — non-degenerate, in the chance-level range expected for 1 round of MSE training under near-zero init (no longer pinned to 1.0000).
  3) Live multi-round run (HUMAN-VERIFY GATE PASSED, 2026-05-04):
     - User relaunched the baseline run with the fix in place.
     - Baseline at round 12/100: Sampled HR@10 = 0.2020, Sampled NDCG@10 = 0.1213 — climbing realistically, no longer pinned to 1.0000.
     - Adaptive comparison reference (untouched) at round 73/100: Sampled HR@10 = 0.3891, NDCG@10 = 0.2110 — confirms BPR path was never affected.
     - Both runs healthy; no regressions on either model path.
     - User confirmation: "confirmed fixed."
files_changed:
  - federated-baseline-cf/federated_baseline_cf/task.py

## Lesson Learned

`BasicMF.predict()` clamps predictions to [1.0, 5.0] for rating prediction (RMSE/MAE).
Ranking metrics (HR@K, NDCG@K, MRR) need raw `forward()` output — the clamp collapses
distinct candidate scores to a single tied value when the model's pre-clamp range is
below 1.0, which is the typical regime under Xavier initialization or early training.
The same dormant bug exists in:
  - federated-personalized-cf/federated_personalized_cf/models/basic_mf.py
  - federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/models/basic_mf.py

Both modules' `BasicMF.predict()` clamp to [1, 5] and both modules' `evaluate_ranking_sampled`
calls `model.predict()`. The bug is latent because their pyproject defaults are
`model-type=bpr` and `model-type=dual` respectively, never exercising the BasicMF path
during current thesis runs. If anyone later runs those modules with `model-type=basic`,
the same degenerate HR@10 = NDCG@10 = 1.0 will appear. Fix should be replicated
defensively in a follow-up phase, but is intentionally OUT OF SCOPE for this debug
session per the constraint that the active adaptive thesis run must not be modified
mid-flight.

Generalizable rule: any time a `predict()` method applies a domain-specific clamp/
sigmoid/sign transform that loses ordering information, downstream ranking
evaluation must call `forward()` directly — `predict()` is for the prediction task
the clamp serves, not for arbitrary scoring.
