# Paper Knowledge Base — Index

> **Last updated**: 2026-04-03
>
> This index is the entry point for the AI agent to find relevant papers.
> Read this file FIRST when you need to find prior work on a specific topic.

---

## Quick Stats
- **Total papers digested**: 14
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
- [zhang_2023_pfedrec](zhang_2023_pfedrec.md) — PFedRec: dual personalization — local score function + post-tuned item embeddings (IJCAI 2023)
- [he_2024_cofedrec](he_2024_cofedrec.md) — CoFedRec: co-clustering by item category + supervised contrastive learning (WWW 2024)
- [zhang_2024_fedca](zhang_2024_fedca.md) — FedCA: composite aggregation — similarity + complementarity; identifies embedding skew problem (arXiv 2024)

### Federated Recommender Systems
- [he_2024_cofedrec](he_2024_cofedrec.md) — **SOTA**: HR@10=77.75, NDCG@10=48.81 on ML-1M (WWW 2024)
- [zhang_2023_pfedrec](zhang_2023_pfedrec.md) — PFedRec: HR@10=73.26, NDCG@10=44.36 on ML-1M (IJCAI 2023)
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

### Communication Efficiency
- [zhang_2022_lightfr](zhang_2022_lightfr.md) — 8x reduction via binary codes (1 bit vs 64-bit float per dimension)
- [reddi_2021_adaptive_fedopt](reddi_2021_adaptive_fedopt.md) — Server-side adaptivity reduces rounds to convergence

### Non-IID Data & Heterogeneity
- [karimireddy_2020_scaffold](karimireddy_2020_scaffold.md) — Control variates make convergence independent of gradient dissimilarity G
- [li_2020_fedprox](li_2020_fedprox.md) — Proximal term handles statistical + systems heterogeneity
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
`#neural-network` `#deep-learning`

---

## Cross-Reference Map
- zhang_2025_survey → covers → zhang_2023_pfedrec, recommends FedRAP [38], GPFedRec [63]
- he_2024_cofedrec → builds on → zhang_2023_pfedrec, SupCon (Khosla NeurIPS'20)
- hu_2025_p2fedrec → builds on → zhang_2023_pfedrec, GPFedRec (Zhang KDD'24), FedMF (Chai 2020)
- zhang_2023_pfedrec → builds on → Per-FedAvg, FedMF, FedRecon
- zhang_2024_fedca → extends → pFedGraph (similarity), FedFast (complementarity)
- zhang_2022_lightfr → extends → FCF (Ammad-Ud-Din 2019), hashing methods
- reddi_2021_adaptive_fedopt → generalizes → FedAvg, combinable with FedProx
- karimireddy_2020_scaffold → improves on → FedAvg, FedProx (proves same theoretical complexity)
- arivazhagan_2019_fedper → inspires → thesis split learning (user=personal, item=base)
- he_2017_ncf → establishes → evaluation protocol (leave-one-out + 99 neg), NeuMF architecture
- he_2020_lightgcn → uses → BPR loss (same as thesis), validates embedding-only models
- bayer_2016_implicit → provides → theoretical framework for implicit feedback optimization
