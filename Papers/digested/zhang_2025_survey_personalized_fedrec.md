# [SURVEY] Personalized Recommendation Models in Federated Settings: A Survey

- **Authors**: Zhang, Chunxu; Long, Guodong; Zhang, Zijian; Li, Zhiwei; Zhang, Honglei; Yang, Qiang; Yang, Bo
- **Venue**: arXiv preprint 2504.07101, March 2025
- **Paper ID**: zhang_2025_survey_personalized_fedrec
- **Tags**: #survey, #federated-learning, #recommender-system, #personalization, #collaborative-filtering, #privacy, #security, #robustness, #communication-efficiency, #foundation-model, #graph-neural-network, #matrix-factorization
- **Scope**: First survey to systematically examine **user personalization modeling** in Federated Recommender Systems (FedRecSys), charting the evolution from centralized paradigms to federated-specific personalized models, with formal definitions, a four-dimensional challenge framework, and five future directions.

---

## 1. Survey Landscape

### 1.1 What This Survey Covers

This survey is the first to focus specifically on **personalized models** in FedRecSys -- prior surveys covered model architectures, security, or efficiency but none addressed the critical question of how to build client-specific model components within federated constraints. It provides: (1) a formal definition of personalization in FedRecSys with a bi-level optimization objective, (2) a comprehensive taxonomy of 100+ FedRecSys papers organized by RecSys Adaptation and FL Enhancement, (3) a structured analysis of four challenge dimensions for deploying personalized models with mapped solutions, and (4) five future research directions. The survey also provides an open-source paper repository at https://anonymous.4open.science/r/Personalized_FedRecSys.

### 1.2 Taxonomy Overview

```
FedRecSys
├── RecSys Adaptation (adapting centralized RecSys to FL)
│   ├── Model Architecture
│   │   ├── Matrix Factorization (MF)
│   │   │   ├── Implicit feedback: FCF, FED-MVMF, P-NSMF, FedRAP
│   │   │   └── Explicit feedback: FedMF, FedRec++, FedRec, MetaMF, Fedmf, FCMF, F2MF, ElFedMF, LightFR, FMFSS, FedRecon
│   │   └── Deep Neural Networks
│   │       ├── MLP: PFedRec, FedNCF, FedFast, UC-FedRec, IFedRec, HPFL, FedPA
│   │       ├── CNN: Dual-CPMF, FedPOIRec
│   │       ├── GNN: FedPerGNN, FedHGNN, SemiDFEGL, P-GCN, F2PGNN, PPCDR, DCI-PFGL, FedHGNN, FeSoG, FedGST, GPFedRec
│   │       └── Transformer: KG-FedTrans4Rec, FLT-PR, RP3FL, MRFF
│   └── Recommendation Scenarios
│       ├── Cross-Domain: PPCDR, FedCDR, P2FCDR, FPPDM, FedDCSR, PFCR, FedHCDR
│       ├── Fairness: F2MF, F2PGNN, RF2, Cali3F, CF-FedSR, FPFR
│       ├── Social: FedHGNN, FeSoG, T-PriDO, DFSR
│       ├── News: FedNewsRec, Efficient-FedRec, UA-FedRec, PrivateRec, FINDING, RD-FedRec
│       └── POI: FedPOIRec, FedGST, PriRec, RFPG, PrefFedPOI, CPF-POI
│
├── FL Enhancement (addressing FL-specific challenges)
│   ├── Security
│   │   ├── Homomorphic Encryption: FedMF, Fedmf, ElFedMF, FedPOIRec, FINDING, FedGNN
│   │   ├── Differential Privacy: PFedRec, FedRAP, IFedRec, GPFedRec, FL-MV-DSSM
│   │   ├── Secret Sharing: FedPOIRec, Efficient-FedRec, Federated CF, FR-FMSS
│   │   ├── Pseudo Item Generation: FedRec++, FedRec, SemiDFEGL
│   │   ├── Personalized Mask: FedMMF
│   │   └── Combined approaches: FedPerGNN (DP+Pseudo), FMFSS (SS+Pseudo), FeSoG (DP+Pseudo)
│   ├── Robustness
│   │   ├── Attack: UA-FedRec, PipAttack, FedAttack, FedRecAttack, IMIA, ClusterAttack, PIECK, A-ra/A-hum, PSMU, PoisonFRS, HMTA, HidAttack
│   │   └── Defense: ElFedMF, UC-FedRec, UNION, APM, CIRDP
│   └── Efficiency
│       ├── Hash Binary Code: LightFR
│       ├── Cluster-based Client Selection: FedFast, CF-FedSR
│       ├── Reduce Transmission Parameters: ElFedMF, MOEFR, FCIS, FNCF-MAB, FCF-BTS
│       ├── Contribution Oriented Client Selection: FedGST
│       ├── Decompose Model: Efficient-FedRec, FedMMR
│       ├── Knowledge Distillation: FedKD, AeroRec
│       ├── Low Rank Decomposition: CoLR
│       ├── Fast-Convergent Aggregation: FedIS
│       └── Refined Optimization: RFRecF
│
├── Personalization Modeling (the survey's NOVEL contribution)
│   ├── Personalized User Embeddings (parameter decoupling: local user emb, global item emb)
│   │   └── Representative: FedMF, FCF, PFedRec, FedNCF (most existing FedRecSys)
│   ├── Personalized Models (full client-specific model components beyond user embeddings)
│   │   └── Representative: PFedRec (local score function), FedRAP (dual-view), GPFedRec (prototypes)
│   └── Challenges & Solutions (Section V -- four-dimensional analysis)
│       ├── Static Item Identifiers: memory/communication overhead
│       ├── Semantic-Aware Representations: overly specific item representations
│       ├── Challenging Scenarios: fairness (overfitted models) + security (disturbed models)
│       └── Foundation Model-based: balance general vs personalized knowledge + inherent bias
│
└── Future Directions
    ├── New Personalized FedRecSys Modeling Methods (multi-granular, cluster-level)
    ├── Personalization Interpretability
    ├── Recommendation Diversity
    ├── Practical Scenarios Evaluation (real-world deployment)
    └── Benchmark Construction (standardized comparison framework)
```

### 1.3 Key Figures and Diagrams

- **Figure 1 (p.1)**: Side-by-side comparison of personalization in centralized vs federated RecSys. Centralized: single model with all user embeddings on server. Federated: per-client models with local user embeddings and shared item embeddings. Clearly shows the parameter decoupling paradigm.
- **Figure 2 (p.2)**: **Master taxonomy diagram** -- the complete survey organization as a tree. Three main branches (RecSys Adaptation, FL Enhancement, Personalization Modeling) with all sub-branches. Essential reference for understanding survey structure.
- **Figure 3 (p.3)**: Standard FedRecSys framework showing client-server communication cycle.
- **Figure 4 (p.11)**: **Challenges and solutions summary** for personalized model-driven FedRecSys. Four settings (Static Item IDs, Semantic-Aware, Challenging Scenarios, Foundation Models) mapped to specific challenges and solutions. Critical framework for thesis positioning.
- **Figure 5 (p.12)**: Solution schematic for memory/communication overhead in static item identifier methods (S1: preserve partial items, S2: matrix decomposition).
- **Figure 6 (p.13)**: Solution schematic for overly specific item representations (S1: partially personalized attribute embeddings, S2: same embeddings for similar users).
- **Figure 7 (p.13)**: Solution schematic for fairness (predict with shared + personalized models) and security (reuse historical models).
- **Figure 8 (p.14)**: Solution schematic for foundation model challenges (S1: adaptive fusion, S2: bias detection/mitigation).

---

## 2. Formal Definitions & Unified Objectives

### Definition 1: FedRecSys
FedRecSys is a privacy-preserving machine learning paradigm that trains decentralized recommendation models through coordinated parameter aggregation across distributed clients (e.g., user devices). By maintaining raw data localized on client nodes and exchanging encrypted model updates during collaborative training, the system achieves dual objectives: (a) enhancing recommendation accuracy through knowledge fusion from heterogeneous user behaviors, and (b) ensuring data sovereignty via cryptographic protocols.

### Definition 2: Personalization in FedRecSys
Personalization in FedRecSys refers to the capability of learning user-specific model components while collaboratively training a global recommendation model under federated constraints. Specifically, each client $u \in \mathcal{U}$ maintains a personalized model $\mathcal{F}_u = \{\theta, \phi_u\}$, where $\theta$ is the global parameters shared across all clients and $\phi_u$ is the personalized parameters unique to client $u$.

This dual-parameter architecture enables:
1. **Knowledge Sharing**: Global parameters $\theta$ capture cross-user patterns through federated aggregation
2. **Local Adaptation**: Personalized parameters $\phi_u$ encode client-specific preferences inferred from private interaction data $\mathcal{Y}_u$

### Unified Optimization Objective (Base FedRecSys)

$$\min_\theta \sum_{u \in U} \alpha_u \mathcal{L}_u(\theta; \mathcal{Y}_u) \tag{1}$$

Where:
- $\theta$: recommendation model parameters
- $\mathcal{L}_u$: local loss function (MSE for explicit, BCE for implicit feedback)
- $\alpha_u = |\mathcal{Y}_u| / |\mathcal{Y}|$: aggregation weight proportional to client data size
- $\mathcal{Y}_u$: client $u$'s private interaction records

### Personalized FedRecSys Objective (Bi-level)

$$\min_\theta \sum_{u \in U} \alpha_u \mathcal{L}_u(\theta, \phi_u^*; \mathcal{Y}_u) \tag{17}$$

$$\text{where} \quad \phi_u^* = \arg\min_{\phi_u} \mathcal{L}_u(\theta, \phi_u; \mathcal{Y}_u)$$

### Objective Variants by Method Category

| Category | How it instantiates the objective | Key addition to Eq. 17 |
|---|---|---|
| Matrix Factorization | $\hat{y}_{ui} = \theta_u^\top \theta_i$; $\theta_i$ aggregated, $\theta_u$ local | $+\lambda(\|\theta_u\|_2^2 + \|\theta_i\|_2^2)$ regularization (Eq. 3) |
| Deep Neural Network | $\hat{y}_{ui} = \sigma(W(\theta_u \oplus \theta_i))$; adds neural weights $W$ | $+\|W\|_F^2$ Frobenius norm on network weights (Eq. 5) |
| Scenario-specific | Base loss + scenario loss | $+\mathcal{L}_\text{scenario}$ with $\mathcal{L}_\text{scenario} < \delta_\text{scenario}$ (Eq. 6) |
| Security-enhanced | Base loss + privacy loss | $+\mathcal{L}_\text{security}$ (HE reversibility, DP noise, etc.) (Eq. 10) |
| Robustness-enhanced | Base loss + attack expectation + defense regularizer | $+\mathbb{E}[\mathcal{A}(\theta)] + \mathcal{D}(\theta)$ (Eq. 13) |
| Efficiency-enhanced | Base loss + comm + memory + computation losses | $+\mathcal{L}_\text{comm} + \mathcal{L}_\text{mem} + \mathcal{L}_\text{comp}$ (Eq. 16) |
| Personalized (memory/comm) | Bi-level + memory/communication loss on personalized item emb $\phi_u^I$ | $+\mathcal{L}_\text{mem}(\phi_u^I) + \mathcal{L}_\text{comm}(\phi_u^I)$ (Eq. 18) |
| Personalized (generality) | Bi-level + generality loss on personalized item emb | $+\mathcal{L}_\text{gene}(\phi_u^I)$ prevents overly specific representations (Eq. 19) |
| Personalized (versatility) | Bi-level + versatility loss on personalized params | $+\mathcal{L}_\text{vers}(\phi_u)$ stability for challenging scenarios (Eq. 20) |
| Foundation model-based | Bi-level + balance loss + bias detection loss | $+\mathcal{L}_\text{bal}(\phi_u) + \mathcal{L}_\text{det}(\phi_u)$ (Eq. 21-22) |

---

## 3. Method Catalog

### 3.1 Matrix Factorization-Based FedRecSys

**Overview**: The most prevalent architectural paradigm in FedRecSys. Decomposes user-item interactions into low-dimensional latent embeddings. Prediction: $\hat{y}_{ui} = \theta_u^\top \theta_i$. Key innovation for FL: item embeddings $\theta_i$ are aggregated across clients to share common knowledge, while user embeddings $\theta_u$ are retained privately on each device.

**Methods Table** (Table II in survey):

| Method | Year | Key Technique | Task | Datasets | Metrics | Code? |
|---|---|---|---|---|---|---|
| FCF [18] | - | Pioneering federated MF | Implicit | Simulated, **ML** | Precision, Recall, F1, MAP, RMSE | No |
| FED-MVMF [36] | - | Multi-view federated MF | Implicit | **ML-1M**, BookCrossings | Precision, Recall, F1, MAP, NMR | No |
| P-NSMF [37] | - | Negative sampling MF | Implicit | **ML-1M**, Netflix5K5K, XING5K5K, AMZ-KindleStore | Precision, NDCG | Yes |
| **FedRAP [38]** | - | **Dual-view: global + personalized item emb** | Implicit | ML-100K, **ML-1M**, AMZ-Instant-Video, LastFM-2K, TaFeng, QB-article | **HR, NDCG** | Yes |
| FedMF [1] | 2020 | HE-protected federated MF | Explicit | ML | Computation Time | Yes |
| FedRec++ [39] | - | Pseudo item generation | Explicit | ML-100K, **ML-1M**, NF5K5K | MAE, RMSE | No |
| FedRec [40] | - | Pseudo item generation | Explicit | ML-100K, **ML-1M** | MAE, RMSE | No |
| MetaMF [41] | - | Meta-learning for MF | Explicit | DB, Hetrec, **ML-1M**, Ciao | MAE, MSE | No |
| Fedmf [42] | - | HE-protected MF | Explicit | Filmtrust, ML-100K | RMSE, CDF | No |
| FCMF [43] | - | Clustered federated MF | Explicit | ML-100K, **ML-1M**, ML-10M, Netflix | MAE, RMSE | No |
| F2MF [44] | - | Fairness-aware federated MF | Explicit | **ML-1M**, AMZ-Movies | Recall, F1, **NDCG** | No |
| ElFedMF [45] | - | Efficient + lightweight MF | Explicit | ML, NYC | RMSE | No |
| LightFR [24] | - | Hash binary codes for efficiency | Explicit | **ML-1M**, Filmtrust, DB-Movie, Ciao | **HR, NDCG** | No |
| FMFSS [46] | - | Secret sharing MF | Explicit | ML-100K, filmTrust, Epinions | RMSE, MAE | No |
| FedRecon [47] | 2021 | Reconstruction-based split learning | Explicit | **ML-1M** | RMSE, Accuracy | No |

**Key Equations**:
- MF prediction: $\hat{y}_{ui} = \theta_u^\top \theta_i$ (Eq. 2)
- Federated MF objective: $\min_\theta \sum_{u \in U} \alpha_u \left[\sum_{(i,y_{ui}) \in \mathcal{Y}_u} L(y_{ui}, \hat{y}_{ui}) + \lambda(\|\theta_u\|_2^2 + \|\theta_i\|_2^2)\right]$ (Eq. 3)

**Key insight**: Item embeddings $\theta_i$ are aggregated across clients for global knowledge; user embeddings $\theta_u$ are retained privately. Two main extension directions: (1) privacy enhancement via DP or HE, and (2) efficiency optimization via communication-efficient protocols.

### 3.2 Deep Neural Network-Based FedRecSys

**Overview**: Enhance FedRecSys by learning hierarchical representations of user-item interactions via nonlinear neural networks. Prediction: $\hat{y}_{ui} = \sigma(W(\theta_u \oplus \theta_i))$ where $\oplus$ is concatenation, $W$ are neural weights, and $\sigma$ is activation. Adds $\|W\|_F^2$ to regularization.

**Methods Table** (Table III in survey):

| Method | Arch | Year | Key Technique | Task | Datasets | Metrics | Code? |
|---|---|---|---|---|---|---|---|
| **PFedRec [6]** | MLP | 2023 | **Local score function + global item emb** | Implicit | ML-100K, **ML-1M**, Lastfm-2K, AMZ-Video | **HR, NDCG** | **Yes** |
| FedNCF [19] | MLP | 2022 | First federated neural CF | Implicit | ML-100K, **ML-1M**, Lastfm-2K, Foursquare NY | **HR, NDCG** | No |
| FedFast [49] | MLP | - | Cluster-based client selection | Implicit | **ML-1M**, ML-100K, TripAdvisor, Yelp | **HR, NDCG** | No |
| UC-FedRec [50] | MLP | - | User-centric defense | Implicit | ML, DB | **HR, NDCG** | Yes |
| IFedRec [51] | MLP | - | DP-protected federated rec | Implicit | CiteULike, XING | Recall, Precision, **NDCG** | Yes |
| HPFL [52] | MLP | - | Hierarchical personalized FL | Explicit | ASSIST, ML | AUC, ACC, MAE, RMSE, DOA, **NDCG** | Yes |
| FedPA [53] | MLP | - | Foundation model + personalized adaptation | Implicit | KuaiRand-Pure, KuaiSAR-S/R | AUC, Precision | Yes |
| Dual-CPMF [54] | CNN | - | Convolutional MF | Explicit | ML | RMSE, Recall, Precision | No |
| FedPOIRec [55] | CNN | - | POI recommendation | Implicit | Foursquare | Precision, Recall, MAP, F1 | No |
| FedPerGNN [5] | GNN | 2022 | Graph + local DP | Explicit | ML-100K, **ML-1M**, ML-10M, Flixster, DB, Yahoo | RMSE | Yes |
| FedHGNN [56] | GNN | - | Heterogeneous GNN | Explicit | ACM, DBLP, Yelp, DB-Book | **HR, NDCG** | No |
| SemiDFEGL [22] | GNN | - | Semi-decentralized ego graph + pseudo gradients | Explicit | **ML-1M**, Yelp2018, Gowalla | Recall, **NDCG** | No |
| P-GCN [57] | GNN | - | Personalized GCN | Implicit | Gowalla, Yelp2018, AMZ-Book | Recall, **NDCG** | No |
| F2PGNN [58] | GNN | - | Fair + personalized GNN | Explicit | ML-100K, **ML-1M**, AMZ-Movies | RMSE | Yes |
| GPFedRec [63] | GNN | - | **Global prototypes + personalized graph** | Implicit | ML-100K, **ML-1M**, Lastfm-2K, HetRec2011, DB | **HR, NDCG** | Yes |
| KG-FedTrans4Rec [64] | Transformer | - | Knowledge graph + Transformer | Implicit | ML, Last FM, Book-Crossing | **HR, NDCG** | No |
| FLT-PR [65] | Transformer | - | Transformer for personalized rec | Implicit | **ML-1M**, AMZ-book | Recall, **NDCG** | No |
| RP3FL [66] | Transformer | - | Multimodal fusion | Implicit | **ML-1M**, Jester | F1-score, Accuracy, AUC | No |
| MRFF [67] | Transformer | - | Multi-modal recommendation | Implicit | KuaiRand-Pure, KuaiSAR-R/S | AUC, LogLoss | Yes |

### 3.3 Recommendation Scenario-Based FedRecSys

**Overview**: Extensions of FedRecSys to specific recommendation domains, each adding a scenario-specific loss term: $\min_\theta \left[\sum_{u \in U} \alpha_u \mathcal{L}_u(\theta; \mathcal{Y}_u) + \mathcal{L}_\text{scenario}\right]$.

**Methods Table** (Table IV in survey, selected):

| Method | Scenario | Key Technique | Datasets | Metrics | Code? |
|---|---|---|---|---|---|
| PPCDR [59] | Cross-domain | Privacy-preserving CDR | AMZ, DB | Recall, **NDCG** | No |
| FedCDR [20] | Cross-domain | Cold-start via auxiliary domain | AMZ-review | MAE, RMSE | No |
| P2FCDR [69] | Cross-domain | Personalized privacy-preserving CDR | AMZ | **HR, NDCG** | No |
| FedDCSR [71] | Cross-domain | Disentangled CDR | AMZ | **HR, NDCG** | Yes |
| F2MF [44] | Fairness | Fairness-aware MF | **ML-1M**, AMZ-Movies | Recall, F1, **NDCG** | No |
| RF2 [74] | Fairness | Re-ranking for fairness | Taobao Ad, ML-20M | AUC, MDAC | Yes |
| FeSoG [21] | Social | Social graph recommendation | Ciao, Epinions, Filmtrust | MAE, RMSE | Yes |
| FedNewsRec [80] | News | Federated news recommendation | Adressa | AUC, MRR, **NDCG** | Yes |
| FedPOIRec [55] | POI | Point-of-interest | Foursquare | Precision, Recall, MAP, F1 | No |

**Key Scenario Loss Functions**:
- Cross-domain: $\mathcal{L}_\text{cross\_domain} = \|M\theta_c^{(s)} - \theta_c^{(t)}\|_2^2$ (Eq. 7) where $M$ is transfer matrix
- Fairness: $\mathcal{L}_\text{fair} = \sum_{k=1}^{K} \Omega(\{\hat{y}_{ui}\}_{u \in \mathcal{G}_k})$ (Eq. 8) where $\Omega$ is fairness metric over protected groups
- Social: $\mathcal{L}_\text{social} = \sum_{v \in S_u} \|\theta^{(u)} - \theta^{(v)}\|_2^2$ (Eq. 9) over social neighbors

### 3.4 Security-Enhanced FedRecSys

**Overview**: Although FL avoids direct data upload, model updates can still leak sensitive information. Security methods add privacy-preserving mechanisms. Objective: $\min_\theta \left[\sum_{u \in U} \alpha_u \mathcal{L}_u(\theta; \mathcal{Y}_u) + \mathcal{L}_\text{security}\right]$ with $\mathcal{L}_\text{security} < \delta_\text{security}$.

**Methods Table** (Table V in survey):

| Method | Technique | Dataset | Code? |
|---|---|---|---|
| FedMF [1] | Homomorphic Encryption | ML | Yes |
| Fedmf [42] | Homomorphic Encryption | Filmtrust, ML-100K | No |
| ElFedMF [45] | Homomorphic Encryption | ML, NYC | No |
| FedPOIRec [55] | Homomorphic Encryption | Foursquare | No |
| FINDING [84] | Homomorphic Encryption | Adressa, MIND | Yes |
| FedGNN [94] | Homomorphic Encryption | Flixster, DB, Yahoo, **ML-1M**, ML-10M | No |
| **PFedRec [6]** | **Differential Privacy** | ML-100K, **ML-1M**, Lastfm-2K, AMZ-Video | **Yes** |
| **FedRAP [38]** | **Differential Privacy** | ML-100K, **ML-1M**, AMZ-Instant-Video, LastFM-2K, etc. | **Yes** |
| IFedRec [51] | Differential Privacy | CiteULike, XING | Yes |
| GPFedRec [63] | Differential Privacy | ML-100K, **ML-1M**, Lastfm-2K, HetRec2011, DB | Yes |
| FedPOIRec [55] | Secret Sharing | Foursquare | No |
| Efficient-FedRec [81] | Secret Sharing | MIND, Adressa | Yes |
| FedRec++ [39] | Pseudo Item Generation | ML-100K, **ML-1M**, NF5K5K | No |
| FedRec [40] | Pseudo Item Generation | ML-100K, **ML-1M** | No |
| FedMMF [98] | Personalized Mask Generation | ML-100K, ML-10M, LastFM | No |
| FedPerGNN [5] | DP + Pseudo Item Generation | ML-100K, **ML-1M**, ML-10M, Flixster, DB, Yahoo | Yes |
| FMFSS [46] | Secret Sharing + Pseudo Item | ML-100K, filmTrust, Epinions | No |

**Key Equations**:
- HE objective: $\mathcal{L}_\text{HE} = \|\text{Decrypt}(\text{Encrypt}(\theta)) - \theta\|_2^2$ (Eq. 11) -- reversibility constraint
- LDP objective: $\mathcal{L}_\text{LDP} = \lambda_1 \cdot \text{PrivacyCost}(\theta; \epsilon) + \lambda_2 \cdot \text{NoisePenalty}(\theta; \epsilon)$ (Eq. 12) where $\epsilon$ is privacy budget

### 3.5 Robustness-Enhanced FedRecSys

**Overview**: Addresses attack and defense in FedRecSys. Unified objective: $\min_\theta \left[\sum_u \alpha_u \mathcal{L}_u + \mathbb{E}[\mathcal{A}(\theta)] + \mathcal{D}(\theta)\right]$ where $\mathcal{A}$ is attack expectation and $\mathcal{D}$ is defense regularizer.

**Methods Table** (Table VI in survey):

| Method | Type | Target | Dataset | Code? |
|---|---|---|---|---|
| UA-FedRec [82] | Attack | Degrade Model Performance | MIND, Feeds | Yes |
| PipAttack [33] | Attack | Promote Targeted Item | **ML-1M**, AMZ | No |
| FedAttack [102] | Attack | Degrade Model Performance | **ML-1M**, Beauty | Yes |
| FedRecAttack [103] | Attack | Promote Targeted Item | ML-100K, **ML-1M**, Steam-200K | Yes |
| IMIA [104] | Attack | Infer User-Item Interactions | ML-100K, Steam-200K, Amazon Cell Phone | No |
| ClusterAttack [105] | Attack | Degrade Model Performance | **ML-1M**, Gowalla | No |
| PIECK [106] | Attack | Promote Targeted Item | ML-100K, **ML-1M**, Amazon Digital Music | No |
| A-ra & A-hum [107] | Attack | Generate Poisoned User Embedding | ML, AmazonDigitalMusic | Yes |
| PSMU [108] | Attack | Promote Targeted Item | **ML-1M**, AMZ Digital Music | No |
| PoisonFRS [109] | Attack | Promote Targeted Item | Steam-200, Yelp, ML-10M, ML-20M | No |
| HMTA [110] | Attack | Promote Targeted Item | ML, AMZ, IJCAI | No |
| HidAttack [111] | Attack | Promote Targeted Item | Amazon Appliances, **ML-1M**, YahooMusic | No |
| ElFedMF [45] | Defense | Defense Inference Attacks | ML, NYC | No |
| UC-FedRec [50] | Defense | Safeguard Users' Attributes | ML, DB | Yes |
| UNION [103] | Defense | Safeguard Model Performance | **ML-1M**, Gowalla | Yes |
| APM [112] | Defense | Safeguard Users' Attributes | ML-100K, **ML-1M** | No |
| CIRDP [113] | Defense | Defense Inference Attacks | **ML-1M**, Lastfm-360K | No |

### 3.6 Efficiency-Enhanced FedRecSys

**Overview**: Addresses communication, memory, and computation bottlenecks. Multi-objective: $\min_\theta \left[\sum_u \alpha_u \mathcal{L}_u + \mathcal{L}_\text{comm} + \mathcal{L}_\text{mem} + \mathcal{L}_\text{comp}\right]$ subject to all three costs being within thresholds.

**Methods Table** (Table VII in survey):

| Method | Technique | Dataset | Code? |
|---|---|---|---|
| LightFR [24] | Hash Binary Code | **ML-1M**, Filmtrust, DB-Movie, Ciao | No |
| FedFast [49] | Cluster-based Client Selection | **ML-1M**, ML-100K, TripAdvisor, Yelp | No |
| CF-FedSR [76] | Cluster-based Client Selection | AMZ, Wikipedia | No |
| ElFedMF [48] | Reduce Transmission Parameters | ML, NYC | No |
| MOEFR [118] | Reduce Transmission Parameters | ML-100K, Epinions | No |
| FCIS [119] | Reduce Transmission Parameters | Citeulike-a, LastFM, Steam, **ML-1M** | Yes |
| FNCF-MAB [120] | Reduce Transmission Parameters | **ML-1M**, ML-100K, FilmTrust, YahooMusic | Yes |
| FCF-BTS [121] | Reduce Transmission Parameters | **ML-1M**, Last-FM, MIND | No |
| FedGST [62] | Contribution Oriented Client Selection | FourSquare | Yes |
| Efficient-FedRec [81] | Decompose Model | MIND, Adressa | Yes |
| FedMMR [122] | Decompose Model | Baby, Sports and Clothing | No |
| FedKD [123] | Knowledge Distillation | MIND, ADR | Yes |
| FedIS [124] | Fast-Convergent Aggregation | **ML-1M**, Lastfm-2K, Steam, Foursquare | Yes |
| CoLR [125] | Low Rank Decomposition | **ML-1M**, Pinterest | Yes |
| AeroRec [126] | Self-Supervised Knowledge Distillation | ML, ML-20M, Yelp | No |
| RFRecF [127] | Refined Optimization Algorithm | ML-100K, **ML-1M**, KuaiRec, Jester | Yes |

### 3.7 Personalized FL Paradigms (Section IV-D)

**Overview**: Two main paradigms for achieving personalization in FL, applicable to FedRecSys:

**Global Model Personalization** (train global first, then adapt locally):
- Data-based methods: Reduce data heterogeneity among clients [158-160]
- Model-based methods: Learn a more capable global model for better local adaptation [161-163]

**Learning Personalized Models** (architecture inherently supports client-specific models):
- Architecture-based methods: Decouple layers or deploy customized models per client [164-166]
- Similarity-based methods: Discover client relationships and share among similar clients [167-168]

---

## 4. Challenges & Solutions Framework

| Challenge | Description | Proposed Solutions | Relevant Methods |
|---|---|---|---|
| **C: Memory & Communication Overhead** (Static Item IDs) | Client devices cannot store entire item embedding table; repeated full model transfer is expensive | **S1**: Store only embeddings of interacted items locally (partial item retention). **S2**: Decompose item embedding matrix into smaller sub-matrices [125, 174] | CoLR [125], partial item caching methods |
| **C: Overly Specific Item Representations** (Semantic-Aware) | Learning fully personalized attribute embeddings per user leads to overfitting and impedes collaborative knowledge transfer | **S1**: Learn only a *subset* of personalized attribute embeddings; shared embeddings for general attributes, personalized for preference-sensitive attributes [47, 165]. **S2**: Group similar users to share identical personalized attribute embeddings [183] | FedRecon [47], similarity-based grouping |
| **C1: Overfitted Models** (Fairness scenario) | Personalized models for high-capability clients over-converge, reducing fairness; more local updates = more overfitting | **S1**: Predict with global shared models AND personalized models in tandem [38, 186]. Global models contain general information that regularizes overfitted local models | FedRAP [38] |
| **C2: Disturbed Models** (Security scenario) | DP noise added to shared parameters diminishes quality of personalized models | **S2**: Collect and reuse unperturbed local personalized models from previous iterations [187]. Historical clean models counter noise | Historical model ensembling |
| **C1: Balance General & Personalized Knowledge** (Foundation Models) | Foundation models have broad knowledge; striking balance with user-specific personalized models is hard | **S1**: Adaptive fusion of general knowledge with personalized models [53]. Hybrid architecture that seamlessly integrates both | FedPA [53] |
| **C2: Inherent Bias from Foundation Models** | Foundation models harbor biases that adversely affect personalized model learning | **S2**: Bias detection and mitigation techniques (adversarial debiasing, calibrated data augmentation, bias-aware loss functions) [199, 200] | Debiasing methods |

### Challenge Formulations

**Challenge: Memory & Communication for Personalized Item Embeddings**
$$\min_\theta \sum_{u \in U} \alpha_u \mathcal{L}_u(\theta, \phi_u^*; \mathcal{Y}_u) \tag{18}$$
$$\text{where } \phi_u^* = \arg\min_{\phi_u} \left[\mathcal{L}_u(\theta, \phi_u; \mathcal{Y}_u) + \mathcal{L}_\text{mem}(\phi_u^I) + \mathcal{L}_\text{comm}(\phi_u^I)\right]$$
$$\text{s.t. } \mathcal{L}_\text{mem}(\phi_u^I) < \delta_\text{mem}, \quad \mathcal{L}_\text{comm}(\phi_u^I) < \delta_\text{comm} \quad (\forall u \in U)$$

Where $\phi_u^I$ denotes personalized item embeddings for user $u$.

**Challenge: Overly Specific Representations**
$$\min_\theta \sum_{u \in U} \alpha_u \mathcal{L}_u(\theta, \phi_u^*; \mathcal{Y}_u) \tag{19}$$
$$\text{where } \phi_u^* = \arg\min_{\phi_u} \left[\mathcal{L}_u(\theta, \phi_u; \mathcal{Y}_u) + \mathcal{L}_\text{gene}(\phi_u^I)\right]$$
$$\text{s.t. } \mathcal{L}_\text{gene}(\phi_u^I) < \delta_\text{gene} \quad (\forall u \in U)$$

Where $\mathcal{L}_\text{gene}$ enforces generality of attribute embeddings to avoid overfitting.

**Challenge: Versatility for Challenging Scenarios**
$$\min_\theta \sum_{u \in U} \alpha_u \mathcal{L}_u(\theta, \phi_u^*; \mathcal{Y}_u) \tag{20}$$
$$\text{where } \phi_u^* = \arg\min_{\phi_u} \left[\mathcal{L}_u(\theta, \phi_u; \mathcal{Y}_u) + \mathcal{L}_\text{vers}(\phi_u)\right]$$
$$\text{s.t. } \mathcal{L}_\text{vers}(\phi_u) < \delta_\text{vers} \quad (\forall u \in U)$$

Where $\mathcal{L}_\text{vers}$ enhances stability of personalized parameters across diverse scenarios.

**Challenge: Foundation Model Balance + Bias**
$$\min_\theta \sum_{u \in U} \alpha_u \mathcal{L}_u(\theta, \phi_u^*; \mathcal{Y}_u) \tag{21}$$
$$\text{where } \phi_u^* = \arg\min_{\phi_u} \left[\mathcal{L}_u(\theta, \phi_u; \mathcal{Y}_u) + \mathcal{L}_\text{bal}(\phi_u) + \mathcal{L}_\text{det}(\phi_u)\right]$$
$$\text{s.t. } \mathcal{L}_\text{bal}(\phi_u) < \delta_\text{bal}, \quad \mathcal{L}_\text{det}(\phi_u) < \delta_\text{det} \quad (\forall u \in U) \tag{22}$$

---

## 5. Research Gaps & Future Directions

| Future Direction | Survey's Description | Relevance to My Thesis | Why |
|---|---|---|---|
| **New Personalized FedRecSys Modeling Methods** | Current methods are user-level and may over-specialize. Need cluster-level, multi-granular, hierarchical personalization. Balance personalization vs generalization. | HIGH | Thesis's hierarchical conditional alpha provides multi-granular personalization (data volume + preference quality). Dual-level personalization (statistical alpha + neural PersonalMLP) directly addresses this gap. |
| **Personalization Interpretability** | Personalized models are opaque. Need explainable personalized FL for user trust, debugging, and user-controlled personalization. | MEDIUM | Thesis's alpha values are inherently interpretable (higher alpha = more local, lower = more global). Could extend with explanations of why alpha is set to certain levels per user group. |
| **Recommendation Diversity** | Personalized models may create filter bubbles. Need to balance personalization with exploration, serendipity, and equal exposure. | MEDIUM | Thesis uses genre diversity (entropy) as an alpha factor -- users with low diversity get different alpha treatment. Could extend to explicitly optimize for diversity. |
| **Practical Scenarios Evaluation** | Only public datasets used. No real-world online validation. Cannot replicate real complexities: diverse profiles, real-time requirements, scalability. | LOW | Thesis uses MovieLens-1M (public dataset). Practical evaluation is out of scope but could be discussed as future work. |
| **Benchmark Construction** | No standardized benchmark for FedRecSys. Each paper reimplements own setup, hindering reproducibility. Need shared datasets, metrics, protocols. | MEDIUM | Thesis's Flower-based implementation with standardized evaluation (leave-one-out, 99 negatives, NCF protocol) contributes to reproducibility. Could release as benchmark. |

---

## 6. Key References Extracted

### Tier 1 -- Must-read (directly relevant to thesis)

| Ref | Paper | Year | Why important | Already in KB? |
|---|---|---|---|---|
| [6] | **PFedRec** (Zhang et al.) | IJCAI 2023 | Local score function + global item embeddings. Already implemented as calibration baseline. Survey highlights as key MLP-based personalization method. | **Yes** (`zhang_2023_pfedrec`) |
| [38] | **FedRAP** | - | Dual-view: global + personalized item embeddings with post-aggregation local fine-tuning. Most architecturally similar to thesis's item perturbation. Uses **HR, NDCG** on **ML-1M**. Code available. | No -- **should digest** |
| [63] | **GPFedRec** | - | Global prototypes in graph-enhanced personalized federated rec. Directly related to thesis's EMA prototype. Uses **HR, NDCG** on **ML-1M**. Code available. | No -- **should digest** |
| [47] | **FedRecon** (Singhal et al.) | NeurIPS 2021 | Reconstruction-based split learning for MF. Foundation of the split learning approach used in thesis. Uses **ML-1M**. | No -- **should digest** |
| [1] | **FedMF** (Chai et al.) | 2020 | Pioneering federated MF. Foundational work that thesis builds upon. | No |
| [19] | **FedNCF** (Perifanis et al.) | 2022 | First federated neural CF. Uses **HR, NDCG** on **ML-1M**. Thesis has centralized NCF baseline. | No |

### Tier 2 -- Should-read (useful techniques or baselines)

| Ref | Paper | Year | Why important | Already in KB? |
|---|---|---|---|---|
| [53] | **FedPA** | - | Foundation model + personalized adaptation. Relevant for adaptive fusion (similar to thesis's alpha-blended approach). Code available. | No |
| [49] | **FedFast** | - | Cluster-based client selection for faster convergence on **ML-1M**. Could complement thesis's approach. | No |
| [5] | **FedPerGNN** (Wu et al.) | 2022 | GNN-based federated rec with LDP on **ML-1M**. Privacy-aware approach. Code available. | No |
| [44] | **F2MF** | - | Fairness-aware federated MF on **ML-1M**. Relevant if extending to fairness. | No |
| [125] | **CoLR** | - | Low rank decomposition for communication efficiency on **ML-1M**. Code available. | No |
| [124] | **FedIS** | - | Fast-convergent aggregation on **ML-1M**. Code available. | No |
| [50] | **UC-FedRec** | - | User-centric defense against attribute inference. Code available. | No |

### Tier 3 -- Nice-to-know (broader context)

| Ref | Paper | Year | Why important |
|---|---|---|---|
| [18] | FCF | - | Pioneering federated MF framework |
| [24] | LightFR | - | Hash binary codes for efficiency in FedRecSys |
| [22] | SemiDFEGL | - | Semi-decentralized FL with pseudo gradients for security |
| [41] | MetaMF | - | Meta-learning approach to federated MF |
| [98] | FedMMF | - | Personalized mask mechanism for security |
| [153-156] | PFL surveys | Various | General personalized FL surveys for broader context |
| [157] | Per-FedAvg (Fallah) | NeurIPS 2020 | MAML-based personalization via local fine-tuning |
| [164-166] | Architecture-based PFL | Various | Layer decoupling approaches in personalized FL |

---

## 7. Connections to My Thesis

### 7.1 Directly Applicable Taxonomy Mappings

The thesis maps to the survey's taxonomy as follows:

```
FedRecSys
├── RecSys Adaptation > Model Architecture > MF
│   └── federated-baseline-cf (FedAvg/FedProx on BPR-MF, all params global)
│
├── RecSys Adaptation > Model Architecture > MLP
│   └── federated-pfedrec (PFedRec calibration baseline, [6] in survey)
│
├── Personalization Modeling > Personalized User Embeddings
│   └── federated-personalized-cf (split learning: local user emb, global item emb)
│
└── Personalization Modeling > Personalized Models
    └── federated-adaptive-personalized-cf (thesis contribution)
        ├── Hierarchical conditional alpha (multi-factor adaptive personalization)
        ├── Dual-level personalization (alpha-blended + PersonalMLP)
        ├── Global prototype (EMA-based sparse user support)
        ├── Per-user learned alpha (gradient-refined personalization)
        ├── Item perturbation (local item embedding adjustments)
        └── Contrastive local-global alignment
```

The thesis's **closest existing methods** in this survey's catalog:
1. **PFedRec [6]** -- Already implemented. Thesis extends beyond local score function to full personalized model.
2. **FedRAP [38]** -- Dual-view (global + personalized item embeddings). Thesis's item perturbation is architecturally similar but uses additive perturbation rather than separate embedding tables.
3. **GPFedRec [63]** -- Global prototypes. Thesis uses EMA-based global prototype for similar purpose (sparse user support).

The survey's paradigm shift argument (Section IV-E) -- from "personalized user embeddings only" to "full personalized models" -- directly validates the thesis's progression from `federated-personalized-cf` (user embeddings only) to `federated-adaptive-personalized-cf` (full personalized model with MLP, alpha, perturbation).

### 7.2 Methods to Prioritize for Implementation

Based on the survey's comparative analysis:

1. **FedRAP [38]** -- Highest priority new baseline. Dual-view personalized item embeddings are the closest published approach to thesis's item perturbation. Code available. Uses HR/NDCG on ML-1M. Would strengthen the thesis comparison table.
2. **GPFedRec [63]** -- Important comparison for the global prototype component. Code available. Uses HR/NDCG on ML-1M.
3. **FedRecon [47]** -- Theoretical foundation for split learning in MF. Should cite and compare conceptually even if not reimplemented.
4. **FedNCF [19]** -- Could extend centralized NCF baseline to federated setting for additional comparison point.

### 7.3 Gaps My Thesis Can Address

The thesis directly addresses these survey-identified gaps:

1. **"Multi-granular personalization"** (Section VI-A): The survey calls for "designing models at different granularities and using hierarchical compositions." The thesis's hierarchical conditional alpha provides exactly this -- two-stage computation with data volume (geometric mean of quantity + coverage) and preference quality (harmonic mean of diversity + consistency), plus conditional rules for edge cases.

2. **"Balance personalization and generalization"** (Section VI-A + Section V-D): The survey identifies this as the core open problem. The thesis's adaptive alpha mechanism ($p_\text{effective} = \alpha \cdot p_\text{local} + (1-\alpha) \cdot p_\text{global}$) with per-client alpha computed from user statistics is a direct solution. The alpha is clipped to [0.1, 0.95], never fully local or fully global.

3. **"Adaptive Fusion of General and Personalized Knowledge"** (Figure 4, Foundation Model challenge S1): While the survey discusses this in the context of foundation models, the thesis applies the same principle to embedding-based models. The dual-level personalization (Level 1: statistical alpha-blending, Level 2: neural PersonalMLP) is an instance of adaptive fusion.

4. **"Overfitted personalized models"** (Section V-C, Challenge C1): The survey's Solution S1 ("predict with shared and personalized models in tandem") maps to the thesis's dual-level prediction. The contrastive local-global alignment loss ($L_\text{total} = L_\text{BPR} + \lambda \cdot L_\text{contrastive}$) provides additional regularization.

5. **"Cluster-level personalization"** (Section VI-A): The thesis's user grouping (sparse 0-30, medium 30-100, dense 100+) with group-specific evaluation and alpha behavior analysis is a step toward cluster-level personalization, though not fully cluster-based model sharing.

### 7.4 Optimization Objectives to Build Upon

The thesis formulation can be grounded in the survey's unified personalized FedRecSys objective (Eq. 17):

$$\min_\theta \sum_{u \in U} \alpha_u \mathcal{L}_u(\theta, \phi_u^*; \mathcal{Y}_u)$$

Where for the thesis:
- $\theta$ = {item_embeddings, item_bias, global_bias} (global, aggregated via FedAvg/FedProx)
- $\phi_u$ = {user_embeddings, user_bias, PersonalMLP, logit_alpha, item_perturbation} (local, never shared)
- $\mathcal{L}_u$ = BPR loss with alpha-blended embeddings + contrastive loss + perturbation regularization

The thesis's full objective maps to the survey's variant framework:
- **Base**: Bi-level personalized FedRecSys objective (Eq. 17)
- **+ Generality constraint**: Alpha mechanism prevents overly specific local models (related to Eq. 19)
- **+ Versatility constraint**: Conditional rules for sparse/niche users (related to Eq. 20)
- **+ Efficiency**: Split learning reduces communication by ~44% vs baseline (related to Eq. 16 comm term)
