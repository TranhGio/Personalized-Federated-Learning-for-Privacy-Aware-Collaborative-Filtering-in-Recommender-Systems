---
name: methodology-review
description: "Project-specific checklist for reviewing thesis methodology: known pitfalls of this codebase and protocol (test-item leakage in training negatives, no validation item / best-round inflation, cold eval and cache misses, calibration and config asymmetries between runs, user-group definition, module differences that break single-factor comparisons, communication-size counts, privacy boundary of Adaptive, statistics rules for paired per-user tests) plus the finding format. Use when reviewing a pre-registration README, a result write-up, an advisor report, a thesis section, or a protocol-affecting diff; also when refuting another reviewer's finding. Not for recomputing every number (numbers-auditor uses analyze-results)."
---

# Reviewing methodology in this project

Each pitfall below cost real time once. Check them first; they are where errors cluster. Verify facts against files when the machine has them — some figures here were measured on specific dates and may be superseded by later runs.

## 1. Evaluation protocol facts (verify against `data/derived/`, `scripts/thesis/`)
- Leave-one-out: test item = latest interaction; **no validation item**. Best round is chosen on the in-loop NDCG of the sampled clients' test items → the "best" number is inflated. Measured inflation (in-loop max − last-10 mean, 2026-10): PFedRec 1.8 pts, Personalized 2.9, Adaptive 2.1, Baseline-BPR 5.5. Round-100 checkpoints (`global_state_last.pt`) exist only for runs after commit 1aba5fa (cf72e0, e81518).
- Sampled eval: test item + 99 negatives seeded by `round_num` (D-06 uses `final_eval_round_index`, 101 for 100-round runs). Any re-score must reproduce the run's own candidate lists (trust gate: `cached` replay equals `final_metrics.best` to 1e-6).
- **Training-negative leakage**: the exclusion table (D-13) and the official PFedRec `data.py` both exclude the test item from training negatives. Harmless-ish with a shared global table; decisive for any per-user item-table fine-tuning (post-tuning leaky E=3 reaches 0.4520 vs clean 0.2762 vs published 0.4436). Ask of any per-user table method: which pool, and does the test row ever get pushed down?
- **Cold eval**: D-06 must stamp `run_id`/`reuse_cache`; without it every user is scored with a cold head (0.0150 vs 0.3447 on cf72e0). `EVAL_VALIDITY.json` status must be `valid`; `d06_cache_misses` must be 0.
- User groups: train-interaction counts, half-open [0,30) / [30,100) / [100,∞) → 809 / 2322 / 2909. ML-1M has ≥ 20 ratings per user, so "sparse" is not cold-start; the α rule `n < 20` fires for ≤ 86 users.

## 2. Comparability between runs/modules
- Only PFedRec cf72e0 has final calibration; Personalized f18e64 and Adaptive 869730 do not. ef41ab (0.3352) vs cf72e0 (0.3447) differ also in checkpoint rule (`best_round` vs `best_round_restore`), best round (100 vs 83) and code commit — the difference is **not** the "stale-head cost".
- Modules differ beyond placement: PFedRec = BCE, dim 32, no user vector; the other three = BPR, dim 64. The placement study is a comparison at each method's default configuration, not a single-factor ablation — say so.
- Communication per round (architecture counts for the cited runs): Baseline-BPR ≈ 633K params, Personalized/Adaptive globals ≈ 241K, PFedRec ≈ 118.6K. The 874K / 485K figures in module docs are wrong. Byte-level numbers from real Flower messages do not exist yet.
- Adaptive uploads the user's own `p_u` (64 floats) each participating round as `user_prototype` — the "user embeddings never transmitted" boundary in module docs is false for Adaptive.
- Centralized BPR-MF comparator (0.3332) is untuned (dim 128, 50 epochs, 1 negative); do not let "federated beats centralized" through.
- Published PFedRec (ML-1M, C = 1.0, own post-tuned table at inference, leaky negatives, validation split): HR@10 73.26 / NDCG@10 44.36. The CLAUDE.md "paper band" HR 0.65–0.70 / NDCG 0.35–0.42 is not the paper's number. Our C = 0.1.

## 3. Statistics rules (paired per-user)
- Unit = user (6040); test = Wilcoxon signed-rank on per-user NDCG@10 differences; binding sidedness must be pre-registered; `zero_method="wilcox"` drops ties (many users share NDCG levels — report `n_nonzero`); Holm across cells; bootstrap CI 10 000 resamples seed 42; rank-biserial r. Tests measure between-user variation, not seed variation — never write "significant across seeds".
- A GO/NO-GO or reporting rule must be fixed before the run; a "winner" needs a tie-break rule.
- Deltas in points (×100) need the baseline next to them; "vượt" requires the paired test, not a larger mean.
- Float32: blending every row of an item table is not bit-exact at α ∈ {0.3, 0.7}; masked blends on touched rows only.

## 4. Provenance
- Every number → path + key (`results.json: final_metrics.best.sampled_ndcg@10/sparse`, `reeval_<v>_best.json: metrics_by_cell.<cell>`, `stats_wilcoxon.json: cells.<cell>.by_scope.overall.p_one_sided`, CSV mean). No key → finding.
- Heavy inputs (`global_state_*.pt`, `.embedding_cache/`) are only on the training machine; a claim that needs them is `uncertain` from the laptop, not `confirmed`.

## 5. Finding format
`title` (≤ 12 words) · `location` (file + section/line) · `claim` (what the document says) · `evidence` (what the files say, with the recomputation) · `severity` ∈ {blocking, major, minor} · `suggested_fix` (smallest change that makes the claim true). Return as data when a workflow asks for a schema; otherwise a Markdown table. Refutation: `status` ∈ {confirmed, refuted, uncertain} + `reason` — default to `refuted` when the evidence does not reproduce.

## Angles the orchestrator assigns
`leakage-and-protocol`, `statistics`, `comparability-and-asymmetry`, `provenance-and-reproducibility`, `advisor-alignment` (does the document do what the advisor's latest instruction says — compare against the quoted text), `writing` (only for advisor-facing drafts: terminology, diacritics, no infrastructure).
