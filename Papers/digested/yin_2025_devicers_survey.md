# [SURVEY] On-Device Recommender Systems: A Comprehensive Survey

- **Authors**: Hongzhi Yin, Liang Qu, Tong Chen, Wei Yuan, Ruiqi Zheng, Jing Long, Xin Xia, Yuhui Shi, Chengqi Zhang
- **Venue**: Data Science and Engineering, 2025 (Vol. 10, pp. 591-620)
- **Paper ID**: yin_2025_devicers_survey
- **Tags**: #survey, #recommender-system, #on-device, #federated-learning, #privacy, #model-compression, #communication-efficiency

---

## 1. Core Contribution

First comprehensive survey on **on-device recommender systems (DeviceRSs)** covering three pillars: (1) **Deployment & Inference** — model compression for resource-constrained devices (embedding binarization, sparsification, compositional embeddings); (2) **Training & Updating** — federated, decentralized, and on-device finetuning; (3) **Security & Privacy** — behavior/attribute leakage, poisoning attacks and defenses. Distinguishes from prior FL-for-RecSys surveys by additionally covering deployment/compression and security/attack dimensions.

## 2. Taxonomy / Classification

```
DeviceRSs
├── Deployment and Inference
│   ├── Binary Code-based Methods (binarization, quantization)
│   ├── Embedding Sparsification
│   ├── Variable Size Embedding Methods
│   ├── Compositional Embedding Methods
│   └── Sustainable Deployment Methods
├── Training and Updating
│   ├── Federated Recommendation (FedRSs)
│   │   ├── Client selection (random, full, clustering-based)
│   │   ├── Local training (MF-FedRSs, DNN-FedRSs, GNN-FedRSs, RL-FedRSs)
│   │   ├── Model upload (HE, secret sharing, MPC, LDP, fake items, mask matrix)
│   │   └── Global aggregation (SGD, FedAvg, average, others)
│   ├── Decentralized Recommendation (DecRSs)
│   └── On-device Finetuning (whole, partial, patch learning)
└── Security and Privacy
    ├── Privacy Risks (behavior leakage, attribute leakage, data ownership)
    └── Poisoning Attacks (data poisoning, model poisoning + defenses)
```

**FedRS sub-types**: Cross-client (users as clients, dominant) vs Cross-platform (service providers as clients).

## 3. Comparative Tables

### FedRS Methods (Table 2, selected)

| Method | Year | Venue | Task |
|--------|------|-------|------|
| FCF | 2019 | ArXiv | Top-k |
| FedMF | 2020 | IEEE IS | Rating pred |
| FedFast | 2020 | KDD | Top-k |
| PFedRS (PFedRec) | 2023 | IJCAI | Top-k |
| LightFR | 2023 | TOIS | Top-k |
| FedRAP | 2024 | ICLR | Top-k |
| GPFedRec | 2024 | KDD | Top-k |
| CoFedRec | 2024 | WWW | Top-k |
| DGFedRS | 2025 | TOIS | Top-k |
| FedDAE | 2025 | AAAI | Top-k |

### FedRS Pipeline Strategies (Table 3)

| Stage | Strategy | Methods |
|-------|----------|---------|
| Local Training | MF-FedRSs | FCF, FedMF, MetaMF, FedRS++, LightFR |
| | DNN-FedRSs | PFedRS, FedNCF, FedDSR |
| | GNN-FedRSs | FedPerGNN, SemiDFEGL, GPFedRec |
| Model Upload | HE | FedMF, HPFL, FedPerGNN |
| | LDP | PrivRec, F2MF, PFedRS |
| | Fake items | SemiDFEGL, FedRS |
| Aggregation | FedAvg | SemiDFEGL, F2MF, FedPerGNN |
| | Gradient descent | FCF, MetaMF, FedRS++ |

## 4. Key Methods Identified

**MF-based FedRSs** (most relevant): FCF, FedMF, MetaMF, PFedRS, LightFR, FedRAP, FedRecon
**DNN-based**: FedNCF, PFedRec, FedDSR, FedFast
**GNN-based**: FedPerGNN, GPFedRec, SemiDFEGL
**Key pattern**: User embedding LOCAL (never uploaded), item embeddings GLOBAL — the split-learning paradigm thesis implements.

## 5. Research Gaps Identified

1. **Heterogeneity**: Data non-IID + system + privacy heterogeneity across clients
2. **Fairness**: FedAvg biases toward data-rich clients; sparse users underserved
3. **Evolving Users**: Cold-start harder on-device; federated unlearning unexplored
4. **Model Copyright**: Exposed models need IP protection
5. **Foundation Models**: Cloud-centric; device deployment infeasible currently
6. **Benchmarking**: No standardized DeviceRS benchmark; inconsistent protocols

## 6. Connections to My Thesis

### 6.1 Where My Work Fits in the Taxonomy

**Training & Updating → Federated Recommendation → Cross-client MF-FedRSs** with split learning. Thesis adds adaptive personalization (hierarchical alpha, dual-level) not covered by any method in the survey.

### 6.2 Key Baselines Mentioned

| Method | Status in Thesis |
|--------|-----------------|
| PFedRS (IJCAI 2023) | **Implemented** as `federated-pfedrec/` |
| CoFedRec (WWW 2024) | **Digested** |
| FedMF | Conceptual ancestor of baseline |
| FedRAP (ICLR 2024) | Not implemented; closest to item perturbation |
| GPFedRec (KDD 2024) | Not implemented; closest to EMA prototype |

### 6.3 Identified Gaps My Thesis Could Address

1. **Heterogeneity (Gap 1)**: Hierarchical conditional alpha adapts personalization per user data characteristics
2. **Fairness (Gap 2)**: Global prototype helps sparse users; alpha gives them more global knowledge
3. **Benchmarking (Gap 6)**: 4-module progression with consistent eval protocol on ML-1M
4. **Not addressed**: Model compression, decentralized RS, formal DP, poisoning robustness

## 7. Key References to Follow

- **FedRAP (ICLR 2024)** — Dual-view personalized item embeddings; closest to thesis item perturbation
- **GPFedRec (KDD 2024)** — Global prototypes; closest to thesis EMA prototype
- **DGFedRS (TOIS 2025)** — Very recent top-k FedRS
- **FedDAE (AAAI 2025)** — Latest FedRS method
- **PipAttack** — First model poisoning attack on FedRS; relevant for security discussion
