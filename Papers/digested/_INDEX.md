# Paper Knowledge Base — Index

> **Last updated**: 2026-10-27
>
> This index is the entry point for the AI agent to find relevant papers.
> Read this file FIRST when you need to find prior work on a specific topic.

---

## Quick Stats
- **Total papers digested**: 20
- **Topics covered**: 8 (Surveys, FL Core Algorithms, FL Personalization, FedRecSys, MF & BPR, Privacy, Communication Efficiency, Non-IID)

---

## Papers by Topic

### Surveys
- [zhang_2025_survey_personalized_fedrec](zhang_2025_survey_personalized_fedrec.md) — Personalization in FedRecSys: taxonomy of 100+ papers, bi-level optimization framework, five future directions (arXiv 2025)
- [yin_2025_devicers_survey](yin_2025_devicers_survey.md) — On-device RS: deployment/compression + federated training + security/attacks; covers FedRS pipeline taxonomy (DSE 2025)

### Federated Learning — Core Algorithms
- [li_2020_fedprox](li_2020_fedprox.md) — FedProx: proximal term for heterogeneous networks, tolerates partial work from stragglers (MLSys 2020)
- [karimireddy_2020_scaffold](karimireddy_2020_scaffold.md) — SCAFFOLD: control variates eliminate client drift; convergence unaffected by heterogeneity (ICML 2020)
- [reddi_2021_adaptive_fedopt](reddi_2021_adaptive_fedopt.md) — FedOpt/FedAdam/FedYogi/FedAdagrad: adaptive server optimizers, 2.2x on sparse tasks (ICLR 2021)

### Federated Learning — Personalization
- [arivazhagan_2019_fedper](arivazhagan_2019_fedper.md) — FedPer: base layers global + personalization layers local; conceptual ancestor of split learning (AISTATS 2020)
- [deng_2020_apfl](deng_2020_apfl.md) — APFL: personalized model = α_i·local + (1−α_i)·global, α_i learned by gradient on local loss; generalization bound with closed-form α* and local-SGD convergence rates (arXiv 2020)
- [collins_2021_fedrep](collins_2021_fedrep.md) — FedRep: global representation + low-dim local heads; head-first alternation (τ_h=10 head epochs ≫ τ_φ=1–5 body epochs); linear convergence in the low-rank linear setting (ICML 2021)
- [singhal_2021_fedrecon](singhal_2021_fedrecon.md) — FedRecon: partially local FL with stateless clients — local params (MF user embeddings) reconstructed from scratch each round and at inference with globals frozen; ≈ first-order MAML (NeurIPS 2021)
- [zhang_2023_pfedrec](zhang_2023_pfedrec.md) — PFedRec: dual personalization — local score function + post-tuned item embeddings (IJCAI 2023)
- [jiang_2023_fedthe](jiang_2023_fedthe.md) — FedTHE/FedTHE+: FedAvg'd global head + local head on a shared extractor, per-test-sample ensemble weight via unsupervised entropy minimization + feature alignment; gains mainly under test-time shift, small in-distribution (ICLR 2023)
- [li_2024_fedrap](li_2024_fedrap.md) — FedRAP: additive local item table D_i on top of PFedRec's local head, sparse global C, tanh-curriculum regularizers (ICLR 2024)
- [zhang_2024_gpfedrec](zhang_2024_gpfedrec.md) — GPFedRec: graph-guided per-client aggregation of item tables + MSE regularizer toward the neighbourhood table (KDD 2024)
- [he_2024_cofedrec](he_2024_cofedrec.md) — CoFedRec: co-clustering by item category + supervised contrastive learning (WWW 2024)
- [zhang_2024_fedca](zhang_2024_fedca.md) — FedCA: composite aggregation — similarity + complementarity; identifies embedding skew problem (arXiv 2024)

### Federated Recommender Systems
- [he_2024_cofedrec](he_2024_cofedrec.md) — **SOTA**: HR@10=77.75, NDCG@10=48.81 on ML-1M (WWW 2024)
- [zhang_2023_pfedrec](zhang_2023_pfedrec.md) — PFedRec: HR@10=73.26, NDCG@10=44.36 on ML-1M (IJCAI 2023)
- [zhang_2024_gpfedrec](zhang_2024_gpfedrec.md) — GPFedRec: HR@10=72.17, NDCG@10=43.61 on ML-1M (Light variant 72.95/45.48), C=1.0 — does NOT beat its own PFedRec re-run (73.03/44.49) (KDD 2024)
- [li_2024_fedrap](li_2024_fedrap.md) — FedRAP: claims HR@10=93.24, NDCG@10=71.87 on ML-1M — **not comparable / not a target** (best round selected on test HR in released code; full participation) (ICLR 2024)
- [singhal_2021_fedrecon](singhal_2021_fedrecon.md) — FedRecon: ML-1M explicit-rating MF (RMSE 0.907 on unseen users) + Gboard deployment; its LOO+99neg numbers come only from PFedRec's re-run: 63.28/36.59 (NeurIPS 2021)
- [hu_2025_p2fedrec](hu_2025_p2fedrec.md) — P2FedRec: privacy-preserving via relationship graphs + two-server ASS+HE + edge-LDP (SIGMOD 2025)
- [zhang_2024_fedca](zhang_2024_fedca.md) — FedCA: HR@10=83.48, NDCG@10=71.18 on ML-1M with PMF backbone (arXiv 2024)
- [zhang_2022_lightfr](zhang_2022_lightfr.md) — LightFR: binary codes for 8x communication reduction; HR@10=50.14 on ML-1M (TOIS 2022)

### Matrix Factorization & BPR
- [he_2017_ncf](he_2017_ncf.md) — NCF/NeuMF: neural interaction function replacing inner product; HR@10=73.0, NDCG@10=44.7 on ML-1M (WWW 2017)
- [bayer_2016_implicit](bayer_2016_implicit.md) — iCD: efficient coordinate descent for implicit feedback via k-separability decomposition (arXiv 2016)
- [he_2020_lightgcn](he_2020_lightgcn.md) — LightGCN: simplified GCN for CF — only neighbor aggregation + layer combination; BPR loss (SIGIR 2020)

### Privacy in Federated Learning
- [hu_2025_p2fedrec](hu_2025_p2fedrec.md) — Multi-level privacy (data + edge) via ASS, HE, and edge-LDP
- [zhang_2022_lightfr](zhang_2022_lightfr.md) — Binary codes prevent gradient inversion attacks
- [singhal_2021_fedrecon](singhal_2021_fedrecon.md) — Local params never transmitted nor persisted server-side; uploading user embeddings is trivially exploitable (p_u·Qᵀ) — threat-model reference
- [zhang_2024_gpfedrec](zhang_2024_gpfedrec.md) — Server materializes an N×N user-similarity graph; r_i leaks neighbours' item tables to clients; Laplace LDP with no clipping and no ε reported
- [li_2024_fedrap](li_2024_fedrap.md) — DP on the global table C (clip τ=0.1, noise z=1), ε never reported; uploaded C⁽ⁱ⁾ carries a λ(D⁽ⁱ⁾ − C) term

### Communication Efficiency
- [zhang_2022_lightfr](zhang_2022_lightfr.md) — 8x reduction via binary codes (1 bit vs 64-bit float per dimension)
- [reddi_2021_adaptive_fedopt](reddi_2021_adaptive_fedopt.md) — Server-side adaptivity reduces rounds to convergence
- [singhal_2021_fedrecon](singhal_2021_fedrecon.md) — Only global params transmitted (2|w| per client per round); DP noise grows with √d → small global model is DP-friendly
- [li_2024_fedrap](li_2024_fedrap.md) — Claims L1-sparse global table cuts communication; uploads are dense in released code, bytes never measured

### Non-IID Data & Heterogeneity
- [karimireddy_2020_scaffold](karimireddy_2020_scaffold.md) — Control variates make convergence independent of gradient dissimilarity G
- [li_2020_fedprox](li_2020_fedprox.md) — Proximal term handles statistical + systems heterogeneity
- [deng_2020_apfl](deng_2020_apfl.md) — α* grows with local–global distribution divergence; heterogeneity residual vanishes only at α=1
- [collins_2021_fedrep](collins_2021_fedrep.md) — Heterogeneity concentrated in heads; single-model FL error grows with head dispersion
- [jiang_2023_fedthe](jiang_2023_fedthe.md) — Test-time label/covariate shift; per-sample global/local head ensembling
- [he_2024_cofedrec](he_2024_cofedrec.md) — Co-clustering addresses client heterogeneity via item-category grouping
- [zhang_2024_fedca](zhang_2024_fedca.md) — Embedding skew: similarity-only aggregation harms non-interacted item embeddings

---

## Papers by Relevance to Thesis

### Tier 0 — Surveys
- [zhang_2025_survey_personalized_fedrec](zhang_2025_survey_personalized_fedrec.md) — Comprehensive FedRecSys survey. Validates thesis: multi-granular personalization and adaptive fusion are key open gaps.

### Tier 1 — Must-implement baselines
- [zhang_2023_pfedrec](zhang_2023_pfedrec.md) — **Primary baseline**. ML-1M: HR@10=73.26, NDCG@10=44.36 (cross-device, emb=32, BCE, 100 rounds)
- [he_2024_cofedrec](he_2024_cofedrec.md) — **Newer SOTA** (WWW 2024). ML-1M: HR@10=77.75, NDCG@10=48.81
- [he_2017_ncf](he_2017_ncf.md) — **Evaluation protocol source**. Leave-one-out + 99 negatives. Centralized BPR: HR@10≈67, NDCG@10≈36

### Tier 2 — Core techniques to incorporate
- [hu_2025_p2fedrec](hu_2025_p2fedrec.md) — User-specific item embedding aggregation via relationship graphs
- [zhang_2024_fedca](zhang_2024_fedca.md) — Composite aggregation (similarity + complementarity). ML-1M: HR@10=83.48
- [arivazhagan_2019_fedper](arivazhagan_2019_fedper.md) — FedPer split = thesis split (base=items, personal=users). Cite as conceptual ancestor.
- [li_2020_fedprox](li_2020_fedprox.md) — Already implemented. Proximal term on global params only.
- [karimireddy_2020_scaffold](karimireddy_2020_scaffold.md) — Potential upgrade: control variates for global item embeddings
- [reddi_2021_adaptive_fedopt](reddi_2021_adaptive_fedopt.md) — Potential upgrade: FedAdam/FedYogi for sparse item embedding gradients

### Tier 2 — Must-cite prior art for the thesis's own mechanisms
- [collins_2021_fedrep](collins_2021_fedrep.md) — Formal ancestor of PFedRec + split learning (item table = representation φ, per-user head/embedding = h_i). Final-calibration = FedRep's head-only phase / "+FT" protocol.
- [singhal_2021_fedrecon](singhal_2021_fedrecon.md) — Final-calibration = FedRecon's reconstruct-at-eval with globals frozen (warm-started). Cold-eval bug class = FedRecon's "no reconstruction" case.
- [deng_2020_apfl](deng_2020_apfl.md) — Direct ancestor of the adaptive module's α-blend and per-user learnable α. Its theory is consistent with the thesis finding α* = 1.0.
- [jiang_2023_fedthe](jiang_2023_fedthe.md) — **Closest prior art to the capstone α-PFedRec head blend** (Stage-A NO-GO). Its in-distribution evidence (weights ≈0.93–0.98 on local head) predicts the NO-GO.
- [li_2024_fedrap](li_2024_fedrap.md) — Prior art for the adaptive module's `enable-item-perturbation` flag (q_global + δ_i); opposite regularizer direction.
- [zhang_2024_gpfedrec](zhang_2024_gpfedrec.md) — "Borrow from similar users" rung of the counterfactual chain; its population-mean ablation mirrors the thesis's −9.4-pt mean-head result.

### Tier 3 — Supplementary references
- [he_2020_lightgcn](he_2020_lightgcn.md) — LightGCN validates BPR-MF as strong foundation; graph convolution for local enhancement
- [bayer_2016_implicit](bayer_2016_implicit.md) — Theoretical: implicit regularizer concept for implicit feedback
- [zhang_2022_lightfr](zhang_2022_lightfr.md) — Communication efficiency via binary codes; orthogonal to personalization

---

## Tag Registry
**Paper types**: `#survey` `#method` `#benchmark` `#position-paper`
**Topics**: `#federated-learning` `#personalization` `#recommender-system` `#matrix-factorization`
`#bpr` `#privacy` `#differential-privacy` `#non-iid` `#communication-efficiency`
`#meta-learning` `#regularization` `#model-splitting` `#aggregation`
`#implicit-feedback` `#collaborative-filtering` `#knowledge-distillation`
`#foundation-model` `#fairness` `#security` `#robustness` `#cold-start`
`#cross-domain` `#social-recommendation` `#graph-neural-network`
`#homomorphic-encryption` `#secret-sharing` `#two-server` `#clustering`
`#optimization` `#variance-reduction` `#control-variates` `#coordinate-descent`
`#neural-network` `#deep-learning` `#test-time-adaptation` `#model-interpolation`

---

## Cross-Reference Map
- zhang_2025_survey → covers → zhang_2023_pfedrec, recommends li_2024_fedrap [38], zhang_2024_gpfedrec [63]
- he_2024_cofedrec → builds on → zhang_2023_pfedrec, SupCon (Khosla NeurIPS'20)
- hu_2025_p2fedrec → builds on → zhang_2023_pfedrec, zhang_2024_gpfedrec, FedMF (Chai 2020)
- hu_2025_p2fedrec → hardens → zhang_2024_gpfedrec (ASS+HE similarity + edge-LDP graph; same threshold/reg structure)
- zhang_2023_pfedrec → builds on → Per-FedAvg, FedMF, singhal_2021_fedrecon
- zhang_2023_pfedrec → is a per-batch instance of → collins_2021_fedrep (item table = φ, affine_output = h_u; PFedRec cites neither FedRep nor FedPer)
- singhal_2021_fedrecon → compared/extended by → zhang_2023_pfedrec (ML-1M LOO+99neg at C=1.0: FedRecon 63.28/36.59, FedRecon+DualPer 68.89/40.04)
- singhal_2021_fedrecon → critiques (stateful clients) → arivazhagan_2019_fedper
- singhal_2021_fedrecon, collins_2021_fedrep → prior art for → thesis final-calibration (head-only refit on frozen globals)
- collins_2021_fedrep → refines → arivazhagan_2019_fedper (same body/head split; head-first τ_h ≫ τ_φ; linear-convergence theory)
- deng_2020_apfl → is ancestor of → thesis adaptive α-blend (p_eff = α·p_local + (1−α)·p_global) + per-user learnable α
- deng_2020_apfl → contrasts with → arivazhagan_2019_fedper (soft whole-model interpolation vs hard layer split)
- jiang_2023_fedthe → builds on → FedRoD (Chen & Chao ICLR'22), collins_2021_fedrep, MEMO
- jiang_2023_fedthe → closest prior art to → thesis capstone α-blend of personal + population head (cf72e0 Stage-A NO-GO)
- jiang_2023_fedthe → contrasts with → deng_2020_apfl (APFL underperforms in FedTHE's tables)
- li_2024_fedrap → extends → zhang_2023_pfedrec (same local Linear(k,1) head + persistent additive local item table D_i)
- li_2024_fedrap → prior art for → adaptive `enable-item-perturbation` (q_global + δ_i)
- zhang_2024_gpfedrec → builds on → zhang_2023_pfedrec, he_2020_lightgcn (1-layer propagation)
- zhang_2024_gpfedrec → contrasts with → thesis adaptive EMA global prototype (per-client neighbourhood aggregate vs single population prototype)
- zhang_2024_fedca → extends → pFedGraph (similarity), FedFast (complementarity)
- zhang_2022_lightfr → extends → FCF (Ammad-Ud-Din 2019), hashing methods
- reddi_2021_adaptive_fedopt → generalizes → FedAvg, combinable with FedProx
- karimireddy_2020_scaffold → improves on → FedAvg, FedProx (proves same theoretical complexity)
- arivazhagan_2019_fedper → inspires → thesis split learning (user=personal, item=base)
- he_2017_ncf → establishes → evaluation protocol (leave-one-out + 99 neg), NeuMF architecture
- he_2020_lightgcn → uses → BPR loss (same as thesis), validates embedding-only models
- bayer_2016_implicit → provides → theoretical framework for implicit feedback optimization
