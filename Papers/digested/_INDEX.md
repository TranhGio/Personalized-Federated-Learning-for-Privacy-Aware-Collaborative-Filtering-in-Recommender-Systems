# Paper Knowledge Base — Index

> **Last updated**: 2026-03-23
>
> This index is the entry point for the AI agent to find relevant papers.
> Read this file FIRST when you need to find prior work on a specific topic.

---

## Quick Stats
- **Total papers digested**: 3
- **Topics covered**: 4 (Federated RecSys, Privacy, Personalization, Clustering)

---

## Papers by Topic

### Federated Learning — Core Algorithms
<!-- Papers on FedAvg, FedProx, SCAFFOLD, FedOpt, etc. -->

### Federated Learning — Personalization
- [zhang_2023_pfedrec](zhang_2023_pfedrec.md) — Dual personalization: local score function + post-tuned item embeddings (IJCAI 2023)
- [he_2024_cofedrec](he_2024_cofedrec.md) — Co-clustering by item category + supervised contrastive learning; beats PFedRec (WWW 2024)

### Federated Recommender Systems
- [he_2024_cofedrec](he_2024_cofedrec.md) — **Current SOTA**: CoFedRec co-clustering, HR@10=77.75, NDCG@10=48.81 on ML-1M (WWW 2024)
- [zhang_2023_pfedrec](zhang_2023_pfedrec.md) — PFedRec dual personalization, HR@10=73.26, NDCG@10=44.36 on ML-1M (IJCAI 2023)
- [hu_2025_p2fedrec](hu_2025_p2fedrec.md) — P2FedRec: privacy-preserving via relationship graphs + two-server ASS+HE + edge-LDP (SIGMOD 2025)

### Matrix Factorization & BPR
<!-- Papers on BPR-MF, NeuMF, implicit feedback methods, etc. -->

### Privacy in Federated Learning
- [hu_2025_p2fedrec](hu_2025_p2fedrec.md) — Multi-level privacy (data + edge) in FedRec via ASS, HE, and edge-LDP

### Communication Efficiency
<!-- Papers on compression, sparsification, quantization in FL -->

### Non-IID Data & Heterogeneity
- [he_2024_cofedrec](he_2024_cofedrec.md) — Addresses client heterogeneity via item-category co-clustering instead of K-Means on embeddings

---

## Papers by Relevance to Thesis

### Tier 1 — Must-implement baselines
- [zhang_2023_pfedrec](zhang_2023_pfedrec.md) — **Primary SOTA baseline**. Must implement in cross-silo setting. ML-1M: HR@10=73.26, NDCG@10=44.36 (cross-device, emb=32, BCE, 100 rounds)
- [he_2024_cofedrec](he_2024_cofedrec.md) — **Newer SOTA** (WWW 2024). ML-1M: HR@10=77.75, NDCG@10=48.81. Builds on PFedRec + co-clustering + contrastive. Cross-device, emb=32.

### Tier 2 — Core techniques to incorporate
- [hu_2025_p2fedrec](hu_2025_p2fedrec.md) — User-specific item embedding aggregation; regularization toward personalized embeddings. Confirms cross-device is standard.

### Tier 3 — Supplementary references
<!-- Useful context but not directly implemented -->

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

---

## Cross-Reference Map
- he_2024_cofedrec → builds on → zhang_2023_pfedrec, SupCon (Khosla NeurIPS'20)
- hu_2025_p2fedrec → builds on → zhang_2023_pfedrec, GPFedRec (Zhang KDD'24), FedMF (Chai 2020)
- zhang_2023_pfedrec → builds on → Per-FedAvg (Fallah NeurIPS'20), FedMF (Chai 2020), FedRecon (Singhal NeurIPS'21)
