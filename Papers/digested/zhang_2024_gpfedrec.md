# GPFedRec: Graph-Guided Personalization for Federated Recommendation

- **Authors**: Chunxu Zhang, Guodong Long, Tianyi Zhou, Zijian Zhang, Peng Yan, Bo Yang
- **Venue**: KDD 2024 (30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Barcelona). DOI 10.1145/3637528.3671702. arXiv 2305.07866v2
- **Paper ID**: zhang_2024_gpfedrec
- **Tags**: #method, #federated-learning, #personalization, #recommender-system, #collaborative-filtering, #implicit-feedback, #graph-neural-network, #aggregation, #regularization, #differential-privacy

---

## 1. Core Idea
The server builds a **user-relation graph** from the cosine similarity of the item-embedding tables that clients upload, so it never needs raw interactions. It then runs a one-layer graph convolution over that graph, producing a **user-specific item table $r_i$** for every client (the mean of its neighbours' tables) and a **global item table $w$** (the mean of all $r_i$, which up-weights well-connected users). Each client starts its next round from $w$ and adds an MSE regulariser $\lambda\,\mathrm{MSE}(q_i, r_i)$ that pulls its item table toward its graph neighbourhood. In short, the server replaces the population average with "borrow from similar users". The same authors' earlier PFedRec is the closest backbone, and on ML-1M GPFedRec does **not** beat it: 72.17/43.61 vs 73.03/44.49. PFedRec with GraphAgg added gains only +0.04 NDCG points.

## 2. Problem & Motivation
Existing FedRec methods (FedMF, FedNCF, FedRecon, MetaMF, PFedRec) treat users as independent and aggregate item embeddings without discrimination, so the common user-user correlations used in centralized CF are lost. Graph-based FedRec either needs side social graphs (FedSoG, Liu et al. TIST'22) or a third-party server that matches interactions to find high-order neighbours (FedPerGNN, Wu et al. Nat. Comms'22), which is costly and leaks interaction information. A pilot on ML-100K (Fig. 1, PFedRec backbone) shows the trade-off. Random user graphs hurt by about 2 NDCG points. Graphs built from interactions help, but they cannot be built under FL. The goal is therefore to get an informative graph **without** interaction data.

## 3. Method

### 3.1 Objective Function
Each user is one client, and $\theta_i = \{p_i, q_i, o_i\}$: user embedding $p_i$ (local), item table $q_i \in \mathbb{R}^{M\times H}$ (initialised from the global table each round, then regularised), and score function $o_i$ (local MLP).

$$
\min_{\{\theta_1,\dots,\theta_N\}} \sum_{i=1}^{N} \Big[\mathcal{L}_i(\theta_i;\mathcal{D}_i) + \lambda\,\mathcal{R}(q_i, r_i)\Big],\qquad \mathcal{R}(q_i,r_i)=\mathrm{MSE}(q_i,r_i)=\tfrac{1}{MH}\lVert q_i-r_i\rVert_F^2
$$

$$
\mathcal{L}_i = -\sum_{(u,m)\in\mathcal{D}_i}\log \hat{y}_{um} \;-\; \sum_{(u,m')\in\mathcal{D}_i^-}\log\big(1-\hat{y}_{um'}\big),\qquad \hat{y}_{um} = \sigma\big(o_i([p_i \,\Vert\, q_{i,m}])\big)
$$

**Server graph construction** (Eqs. 3–4). $\mathrm{vec}(q_i)$ is the flattened $M\cdot H$ table, and $\bar{S}$ is the mean of **all** entries of $S$, including the diagonal:

$$
S_{ij} = \frac{\langle \mathrm{vec}(q_i), \mathrm{vec}(q_j)\rangle}{\lVert q_i\rVert\,\lVert q_j\rVert},\qquad A_{ij} = \mathbb{1}\big[S_{ij} > \gamma\,\bar{S}\big]
$$

**Graph-guided aggregation** (Eq. 5, with $l=1$). The paper writes $R = A^{l}Q$ with "vanilla adjacency". The released code row-normalises ($\hat{A} = D^{-1}A$), so each $r_i$ is a neighbour **mean**. If a client has no neighbours, its row falls back to a self-loop:

$$
r_i = \frac{1}{d_i}\sum_{j:\,A_{ij}=1} q_j \quad (d_i=\textstyle\sum_j A_{ij}),\qquad r_i = q_i \text{ if } d_i = 0
$$

**Global ("popular preference") table** (Eq. 6). The paper writes $w = DQ$ ("degree-weighted"). The code takes the mean of the $r_i$, so client $j$ ends up with weight $c_j$ that grows with how often it appears in others' neighbourhoods:

$$
w = \frac{1}{N}\sum_{i=1}^{N} r_i = \sum_{j} c_j\, q_j,\qquad c_j = \frac{1}{N}\sum_i \frac{A_{ij}}{d_i}
$$

**LDP variant** (Eq. 9). Before upload, $q_i \leftarrow q_i + \mathrm{Laplace}(0,\delta)$ elementwise, where $\delta$ is the Laplace **scale**. There is no clipping, no sensitivity bound and no $\varepsilon$ accounting.

Where:
- $w$: global item table (the paper's $q_{global}$), used to initialise every client's $q_i$ each round
- $r_i$: user-specific item table for client $i$, kept on the server and sent only to client $i$
- $\lambda$: regularisation coefficient. $\gamma$: neighbour-threshold factor. $l$: number of propagation layers (always 1)
- $\mathcal{D}_i^-$: 4 uniform negatives per positive, drawn from items $i$ never interacted with and **resampled every round**

### 3.2 Algorithm
This follows Algorithm 1 plus the released code (github.com/Zhangcx19/GPFedRec). Note that the equation numbers cited in Alg. 1 are shifted by one from the numbers in the text.

**Server-side:**
1. Initialise one random item table and set $w^{(0)} = r_i^{(0)} =$ that table for all $i$.
2. For each round $t = 1..R$, select participants. The paper and code use **$C = 1.0$**: Alg. 1 loops over all $N$ clients, and `clients_sample_ratio` defaults to 1.0.
3. Send $w^{(t-1)}$ and $r_i^{(t-1)}$ to client $i$, which is $2MH$ floats.
4. Receive each client's updated, optionally noised, item table $q_i$ ($MH$ floats).
5. Flatten the tables into $Q \in \mathbb{R}^{N \times MH}$ and compute the $N\times N$ similarity $S$.
6. Threshold at $\gamma\bar{S}$ to get $A$, then row-normalise.
7. Compute $r_i = \sum_j \hat{A}_{ij} q_j$ for every $i$ (one layer), then $w = \frac{1}{N}\sum_i r_i$.
8. **Light_GPFedRec**: same procedure, but steps 5–6 run less often than every round. The paper gives no update interval, and the released code has no such option.

**Client-side (client $i$, round $t$):**
1. Load the local $p_i$ (`Embedding(1,32)`), MLP layers and `affine_output` from the previous round, and set $q_i \leftarrow w^{(t-1)}$.
2. For $K$ epochs ($K=1$), and for each batch of 256 drawn from $\mathcal{D}_i \cup \mathcal{D}_i^-$:
   - do **one joint** forward/backward pass on $\mathcal{L}_{\text{BCE}} + \lambda\,\mathrm{MSE}(q_i, r_i)$ over the **whole** $M\times H$ table, then take three SGD steps:
   - MLP and head at lr $\eta=0.1$;
   - $p_i$ at lr $\eta\,\eta_e/C - \eta = 7.9$;
   - $q_i$ at lr $\eta M\eta_e - \eta \approx 29{,}648$ on ML-1M, which is PFedRec's huge item learning rate.
   - Unlike PFedRec, there is **no** alternating two-pass update.
3. Store $p_i$, the MLP and $q_i$ locally. Upload $q_i$ only (plus Laplace noise if $\delta>0$).

### 3.3 Key Hyperparameters
| Hyperparameter | Symbol | Default Value | Role |
|---|---|---|---|
| Threshold factor | $\gamma$ | 0.5 (best of {0, 0.5, 1.0, 1.5, 2.0} on ML-100K); code default 1.0 | Neighbour if $S_{ij} > \gamma\bar{S}$. $\gamma=0$ gives a fully connected graph (population mean) |
| Reg. coefficient | $\lambda$ | 0.5 (best of {0..2}, step 0.5, ML-100K); code default 1.0 | Pull of $q_i$ toward $r_i$ |
| Propagation layers | $l$ | 1 | GCN depth (`mp_layers`) |
| Similarity | — | cosine on flattened tables | Euclidean, Jaccard and Pearson tried for interaction graphs (App. E) |
| Embedding dim | $H$ | 32 (grid 16/32/64/128) | Same for all baselines |
| Score function | $o_i$ | MLP $64\to32\to16\to8\to1$, ReLU, sigmoid | Input is $[p_i \Vert q_{i,m}]$ (the paper writes "32→16→8→1") |
| Base LR | $\eta$ | 0.1 SGD (searched over {1e-4, 1e-3, 1e-2, 0.1}) | MLP and head |
| Item-LR multiplier | $\eta_e$ | 80 | Item LR $=\eta M\eta_e-\eta$ |
| Batch / negatives | — | 256 / 4 per positive | — |
| Rounds / local epochs | $R$ / $K$ | 100 / 1 | — |
| Participation | $C$ | 1.0 | Full participation only |
| LDP scale | $\delta$ | 0–0.5 (0.3 recommended) | Laplace noise on the upload |

### 3.4 Architectural Decisions
- **Why similarity of item tables?** "Similar users share a similar view of the same items." The tables are already uploaded under FedAvg-style FedRec, so the graph costs no extra upload.
- **Adaptive threshold instead of top-k.** A fixed $k$ is hard to set. Thresholding against the global mean similarity lets degrees vary by user. The code also has a top-k mode (`neighborhood_size>0`, weights $1/k$), but the paper does not use it.
- **Why regularise instead of replacing $q_i$ with $r_i$?** Replacing would throw away local signal. The neighbourhood acts as a prior and $\mathcal{L}_i$ keeps the personal view.
- **Why a degree-weighted global table?** It favours "popular preference". The $\lambda=0$ point in Fig. 4 (initialisation from $w$ only) still roughly matches PFedRec on ML-100K.
- **Implicit α (derived from the released code; not stated in the paper).** The MSE uses mean reduction over $MH$ entries, and the item LR is about $\eta M\eta_e$. A row $m$ with no BCE gradient in a batch therefore moves by $q_{i,m} \leftarrow q_{i,m} - \rho\,(q_{i,m}-r_{i,m})$, where $\rho = 2\eta\eta_e\lambda/H = 0.5\lambda$ (with $\eta=0.1$, $\eta_e=80$, $H=32$; $M$ cancels). After $B_i$ batches, the untouched rows equal $(1-\rho)^{B_i} w + [1-(1-\rho)^{B_i}]\, r_i$. This is an **implicit per-user blend weight toward the neighbourhood that grows with data volume**. At $\lambda=0.5$:
  - a 20-interaction user (≈90 samples, 1 batch) gets ≈0.25;
  - an ML-1M mean user (≈166 interactions, 4 batches) gets ≈0.68;
  - a dense user gets close to 1.

  At $\lambda=2$, $\rho=1$, so $q_i$ is overwritten by $r_i$ every batch. This matches the collapse in Fig. 4 (HR about 59 at $\lambda=2$). It is also why $\lambda$ does not transfer across $H$, $\eta$ or $\eta_e$.
- **The graph feeds back on itself.** Untouched rows of the uploaded $q_i$ already lean toward the previous $r_i$, so $S^{(t)}$ partly re-encodes $A^{(t-1)}$.

## 4. Implementation Notes
- **Framework used in paper**: PyTorch. Plain-loop simulation with no FL framework (`train.py`, `engine.py`, `utils.py`, `mlp.py`). Server similarity uses `sklearn.metrics.pairwise_distances`.
- **Model architecture**: `Embedding(1,32)` user (local) and `Embedding(M,32)` item. The MLP is `Linear(64,32)`→ReLU→`Linear(32,16)`→ReLU→`Linear(16,8)`→ReLU→`affine_output Linear(8,1)`→Sigmoid, all local. Only `embedding_item.weight` is uploaded.
- **Training details**: SGD without momentum or weight decay, lr 0.1, item lr $\eta M\eta_e-\eta$, user lr $\eta\eta_e/C-\eta$, batch 256, $K=1$, 100 rounds, $C=1.0$. Reported test numbers are taken at the round with the **best validation HR@10**; ties go to the later round.
- **Data preprocessing**: ratings are binarised to 1. Leave-one-out: the latest interaction is the test item, the second-latest the validation item, the rest are training. 198 negatives per user are drawn once (`random.seed(0)`) from items the user never interacted with: the first 99 for validation, the next 99 for test. HR@10 and NDCG@10 are computed over 100 candidates, with ties broken against the positive. ML-1M is used as-is (≥20 ratings per user). Lastfm-2K keeps users with ≥5 interactions.
- **Communication / compute (ML-1M, $M=3706$, $H=32$)**:
  - Per client per round: 118,592 floats up and 237,184 floats down ($r_i$ plus $w$). PFedRec sends 118,592 each way.
  - Server state: $Q$ and $R$ at $N\times MH$ each, about 2.9 GB fp32 apiece.
  - Similarity: about $N^2MH \approx 4.3\times10^{12}$ MACs per round, plus a pure-Python $N^2$ threshold loop. This cost is why Light_GPFedRec exists.
- **Tricks & gotchas** (from the released code, fetched 2026-10-27; it may differ from the code behind the paper's tables):
  1. **Similarity sign bug.** `pairwise_distances(metric='cosine')` returns cosine **distance**, but the cosine branch returns it unchanged as "similarity". The Euclidean branch correctly negates. As released, cosine "neighbours" are therefore pairs with distance $> \gamma\cdot$mean distance: the *least* similar users, with the self-loop excluded. Anyone reimplementing should use `1 - pairwise_distances` or `cosine_similarity`.
  2. **$C<1$ breaks.** Client tables are written into an array of length |participants| indexed by user id, and the server's $r_i$ dictionary is replaced with only the current round's participants. A client absent in round $t-1$ has no $r_i$.
  3. **Dense regulariser.** The MSE spans all $M$ rows every batch, while BCE touches only the batch's items.
  4. **Eval uses each user's own trained item table.** It does not use $w$ or $r_i$; the line that would substitute $w$ is commented out.

## 5. Experimental Results

### 5.1 Datasets Used
| Dataset | Users | Items | Interactions | Sparsity |
|---|---|---|---|---|
| MovieLens-100K | 943 | 1,682 | 100,000 | 93.70% |
| **MovieLens-1M** | **6,040** | **3,706** | **1,000,209** | **95.53%** |
| Lastfm-2K | 1,600 | 12,454 | 185,650 | 99.07% |
| HetRec2011 | 2,113 | 10,109 | 855,598 | 95.99% |
| Douban | 2,509 | 39,576 | 893,575 | 99.10% |

### 5.2 Key Results
Table 3 reports HR@10 / NDCG@10 (×100), with LOO + 99 negatives, H=32, 100 rounds and **C=1.0**. "*" marks two-sided t-test p<0.05. The number of seeds is not stated.

| Method | ML-100K | **ML-1M** | Lastfm-2K | HetRec2011 | Douban |
|---|---|---|---|---|---|
| MF (central) | 64.48 / 38.61 | 68.69 / 41.45 | 83.13 / 71.78 | 66.07 / 41.21 | 87.17 / 61.75 |
| NCF (central) | 64.21 / 37.13 | 64.02 / 38.16 | 82.57 / 68.26 | 64.74 / 39.55 | 87.49 / 62.51 |
| SGL (central) | 64.90 / 40.02 | 62.60 / 34.13 | 82.37 / 68.59 | 65.12 / 40.18 | – |
| FedMF | 66.17 / 38.73 | 67.91 / 40.81 | 81.63 / 68.18 | 64.69 / 40.29 | 87.17 / 61.00 |
| FedNCF | 60.66 / 33.93 | 60.38 / 34.13 | 81.44 / 61.95 | 60.86 / 36.27 | 86.01 / 59.94 |
| FedRecon | 65.22 / 38.49 | 62.78 / 36.82 | 82.06 / 67.37 | 61.57 / 34.20 | 87.52 / 60.38 |
| MetaMF | 66.21 / 41.02 | 44.98 / 26.31 | 81.04 / 64.13 | 54.52 / 32.36 | 82.58 / 55.44 |
| **PFedRec** | 71.37 / 42.59 | **73.03 / 44.49** | 82.38 / 73.19 | 67.20 / 42.70 | 87.40 / 61.90 |
| FedLightGCN | 24.53 / 12.78 | 37.53 / 15.01 | 43.75 / 15.17 | 22.65 / 7.96 | 35.66 / 12.33 |
| FedPerGNN | 11.52 / 5.08 | 9.31 / 4.09 | 10.56 / 4.25 | OOM | OOM |
| **GPFedRec** | 72.85* / 43.77* | **72.17 / 43.61** | 83.44* / 74.11* | 69.41* / 43.34* | 88.04* / 63.87* |
| Light_GPFedRec | 72.00* / 43.92* | 72.95 / **45.48*** | 83.44* / 74.33* | 69.47* / 43.21* | 88.04* / 64.00* |

- **ML-1M:** full GPFedRec loses to PFedRec on both metrics. Only the Light variant wins NDCG (+0.99) while still losing HR. The Light-vs-full gap (+1.87 NDCG) is larger than most claimed improvements, which suggests run-to-run variance of about 1–2 points.
- **PFedRec is re-run here:** 73.03/44.49, against 73.26/44.36 in its own paper.

**GraphAgg as a plug-in (Table 5)**, swapping each method's plain item aggregation for graph-guided aggregation:

| Backbone | ML-100K | ML-1M | Δ ML-1M (NDCG pts) |
|---|---|---|---|
| FedMF → +GraphAgg | 66.17/38.73 → 71.79/44.20 | 67.91/40.81 → 72.15/43.69 | +2.88 |
| FedRecon → +GraphAgg | 65.22/38.49 → 70.78/41.10 | 62.78/36.82 → 69.03/40.15 | +3.33 |
| **PFedRec → +GraphAgg** | 71.37/42.59 → 72.38/43.75 | **73.03/44.49 → 73.50/44.53** | **+0.04** |

Large gains appear only for backbones that do not personalise the item view. With PFedRec on ML-1M the gain is about zero, and on Lastfm-2K NDCG drops (73.19 → 73.11).

**LDP (Table 6, ML-100K)**, as HR/NDCG for each noise scale $\delta$:

| $\delta$ | HR | NDCG |
|---|---|---|
| 0 | 72.85 | 43.77 |
| 0.1 | 71.89 | 42.58 |
| 0.2 | 71.32 | 41.79 |
| 0.3 | 70.41 | 41.78 |
| 0.4 | 69.99 | 40.68 |
| 0.5 | 69.35 | 39.89 |

At $\delta=0.3$ the cost is −1.99 NDCG points. PFedRec's own LDP table on ML-1M lost only 0.13 at the same scale, so noise hurts GPFedRec more, plausibly because it also corrupts the graph.

### 5.3 Ablation Highlights
- **Components (Table 4, ML-1M).** The backbone is FedNCF, and PSF means personalised score function:

  | Variant | HR@10 / NDCG@10 |
  |---|---|
  | FedNCF | 60.38 / 34.13 |
  | + PSF | 67.14 / 40.22 |
  | + PSF + Init from $w$ | 68.26 / 41.10 |
  | + PSF + Reg to $r_i$ | 67.79 / 42.98 |
  | Full | 72.17 / 43.61 |

  Init and Reg are complementary on every dataset. GPFedRec's own full model, built on FedNCF, still trails the PFedRec backbone on ML-1M. Backbone choice matters more than the graph.
- **Threshold $\gamma$ (Fig. 3, ML-100K; values read off the bars, about ±0.5):**

  | $\gamma$ | HR | NDCG | Note |
  |---|---|---|---|
  | 0 | 68.5 | 40.2 | Fully connected, i.e. borrow from everyone |
  | 0.5 | 72.85 | 43.77 | Best |
  | 1.0 | 69.7 | 41.3 | — |
  | 1.5 | 67.7 | 40.6 | — |
  | 2.0 | 63.6 | 36.3 | — |

  Regularising toward the **population mean** ($\gamma=0$) is **worse than PFedRec** (42.59) and worse than no regularisation at all ($\lambda=0$ gives about 42.5). Borrowing from everyone hurts, and borrowing from a neighbourhood helps a little.
- **Regulariser $\lambda$ (Fig. 4, ML-100K; read off the bars):**

  | $\lambda$ | HR | NDCG | Note |
  |---|---|---|---|
  | 0 | 71.0 | 42.5 | — |
  | 0.5 | 72.85 | 43.77 | Best |
  | 1.0 | 67.2 | 39.7 | The code's default |
  | 1.5 | 61.9 | 34.8 | — |
  | 2.0 | 58.8 | 32.9 | $q_i$ overwritten by $r_i$ every batch (see §3.4) |
- **Embedding size (Fig. 5).** $H=32$ is best. HR/NDCG is about 69.7/41.8 at $H=16$ and about 70.1/43.1 at $H=128$.
- **Graph source (Fig. 1 and Table 7, ML-100K, densities 10–90%):**
  - Random graphs (BA/WS/ER/Regular) all fall **below** the backbone, by roughly 1.7–2.6 NDCG points.
  - Interaction graphs built with cosine or Jaccard roughly match the backbone. Euclidean at 80% density gives the best result, 73.17/44.73.
  - GPFedRec's embedding-derived graph lands between those two groups.
  - Denser interaction graphs are generally better.
- **Convergence (App. C).** Converges fast on the sparse datasets (Lastfm, Douban). On MovieLens it tracks FedNCF early and overtakes later.

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas
1. **"Borrow from similar users" vs "borrow from everyone".** GPFedRec's own ablation reproduces the thesis's counterfactual one level down, on the item table instead of the head. On ML-100K, regularising toward the population mean ($\gamma=0$, about 40.2) is worse than not borrowing at all (about 42.5). That is the item-table analogue of the thesis result where the population-mean head scores 0.2510 against the cached head's 0.3447 and the best α is 1.0. A thresholded neighbourhood ($\gamma=0.5$) turns this into a +1.2-point gain. The paper therefore changes the *sign* of the prediction for neighbourhood borrowing, but not its *size*: on ML-1M, PFedRec+GraphAgg gains only +0.04 NDCG points (Table 5). ML-1M users have at least 20 ratings and 166 on average, so local data already pins down each user's view.
2. **Implicit α.** GPFedRec's regulariser behaves like a per-user blend weight that grows with the number of local batches (§3.4). That runs opposite to the thesis's hierarchical-conditional heuristic, which makes sparse users more global. It is worth one sentence in the related-work discussion of α design: neither direction is shown to win on ML-1M.
3. **Threat-model entry (privacy).**
   - **What the server sees.** It receives each client's full item table every round. Changed rows reveal the client's positives and sampled negatives (Yuan et al. WWW'23 [46]). It also *materialises* an $N\times N$ user-similarity graph, which is an edge-level leak.
   - **A new client-to-client channel.** Client $i$ receives $r_i$, the mean of its neighbours' tables. $r_i - w$ concentrates on the neighbours' interacted rows. When $d_i = 1$, $r_i$ *is* one other user's table.
   - **Weak LDP.** The Laplace noise has no clipping and no $\varepsilon$, so it is not a formal guarantee.
   - **Later hardening.** P2FedRec (hu_2025_p2fedrec) addresses exactly these two leaks with ASS+HE similarity across two servers plus an edge-LDP graph, with the same threshold/regulariser structure ($\alpha\equiv\gamma$, $\beta\equiv\lambda$). For the thesis: FedAvg, PFedRec and split learning expose only one table (or one 32-d head) per client to a server that sees a population mean. GPFedRec-style methods add graph and neighbour-aggregate exposure.
4. **Per-method communication table row.** GPFedRec sends $MH$ floats up and $2MH$ down (118,592 / 237,184 on ML-1M) and keeps $2NMH$ floats of server state (about 5.7 GB fp32). PFedRec sends $MH$ each way.

### 6.2 Potential Integration Points
1. **Offline neighbourhood-head counterfactual.** This is cheap and fits the existing machinery: add a variant to `scripts/thesis/reeval.py` `run_pfedrec_suite`, alongside `global_head`, `alpha_grid` and `alpha_oracle_*`, on run `20260726-083121-cf72e0`.
   - **The new head.** For each user, $w_i^{nbr} = \mathrm{mean}_{j\in\mathcal{N}_k(i)} \hat{w}_j$ over the cached `affine_output.weight` rows. Score with $\alpha\hat{w}_i + (1-\alpha)w_i^{nbr}$, reusing the bit-exact trust gate and the cross-fitted bucket oracle.
   - **Use unit-normalised heads.** `affine_output.bias` is global in this module, so a user's ranking depends only on the *direction* of $w_i$. Raw averaging would let high-norm heads dominate.
   - **Neighbour spaces (≤3, pre-registered, $k\in\{20,100\}$):**
     - (a) **Interaction Jaccard** on training positives. This mirrors GPFedRec's interaction graph and is privacy-violating, so treat it as an upper bound on graph quality.
     - (b) **Profile space.** The mean of restored global item embeddings over the user's training positives approximates what GPFedRec's uploaded $\Delta q_i$ encodes.
     - (c) **Head space.** This is near-tautological: neighbours are chosen *because* their heads are close, so it reduces to directional shrinkage.
   - **Why the GPFedRec signal cannot be replayed exactly.** The thesis module caches only `affine_output.weight`. Per-client item tables are never persisted, and the $N\cdot MH$ (about 2.9 GB) bank would have to be added at training time.
   - **Honest expected gain: about null.**
     - The paper's own nearest configuration (PFedRec+GraphAgg on ML-1M) gives +0.04 points.
     - The per-user oracle already picks α=1.0 for 74% of users, and the cross-fitted bucket Δ against the population head is −0.0002.
     - A neighbourhood target sits closer to $w_i$, so the α-curve flattens near 1 rather than gaining.
     - Sparse users (809 users, 20–30 interactions) score *higher* than average (0.4270 vs 0.3447), so they are not the bottleneck.
     - Predicted results: cross-fitted Δ in about [−0.003, +0.003] overall, and sparse Δ ≤ +0.005, both below the +0.010 GO threshold. A pure neighbourhood head (α=0) should land between 0.2510 and 0.3447.
   - **Value.** It closes the obvious reviewer question ("you only tried the population mean; GPFedRec shows similar users help"). It also adds a rung to the "where should personalization live" chain: own > neighbourhood > population > shuffled > random. Run it once, pre-registered, after the paired significance tests. **Do not** build a training-time GPFedRec module.
2. **Under cross-device C=0.1** the per-round graph degrades. Only about 604 clients upload per round, so:
   - each client's neighbours come from a random 10% of the population and change every round;
   - $\bar{S}$ and the degrees are computed on a 604×604 block;
   - each client participates about 10 times in 100 rounds.

   A faithful port needs a server **bank** of each client's last upload:
   - entries are on average about 10 rounds stale;
   - clients that never uploaded must be masked, since $0.9^{t}$ of clients are still at the initial table after $t$ rounds, about 10% at round 22, and identical cold entries would cluster;
   - computing $r_i$ on demand for the sampled clients against the bank costs about $604\cdot6040\cdot MH \approx 4.3\times10^{11}$ MACs per round.

   None of this is specified in the paper, and the released code cannot run at $C<1$. A thesis implementation would be a GPFedRec-inspired method, not a reproduction.
3. **Adaptive module (future work only).** Replace the single server EMA prototype $p_{global}$ with a per-client neighbourhood prototype from the bank. Given that adaptive (0.2144) already trails split learning (0.2275), and given the +0.04 result above, list it as future work rather than a run.

### 6.3 Limitations & Gaps
- **Not comparable to thesis numbers.**
  - GPFedRec's ML-1M results (72.17/43.61; Light 72.95/45.48) use **C=1.0** and evaluate each user's own locally trained item table. Test numbers come from the best-validation-HR round, with an unstated number of seeds.
  - The thesis's PFedRec uses C=0.1, scores the restored global item table plus a cached head, and evaluates the full population (0.3352, or 0.3447 calibrated; HR about 0.59).
  - GPFedRec belongs in the "paper-compat full participation" band, alongside PFedRec's 73.26/44.36 and the re-run 73.03/44.49. Its ML-1M numbers are **below** its own PFedRec re-run, which reinforces that PFedRec is the right calibration bar.
- **Loss and model differ.** BCE with an NCF-MLP head and a user embedding; no BPR and no MF dot-product. Large server memory and compute.
- **Unverifiable sources of gain.** The released cosine graph is sign-inverted (§4). Light_GPFedRec's update interval is unreported. There is no per-user-group (sparse/dense) breakdown, so the sparse-user claim (cold-start, §4.6.2) is argued, not measured.
- **Privacy is not analysed.** There is no attack evaluation, the graph and $r_i$ leakage go unanalysed, and the LDP has no formal $\varepsilon$.

## 7. Key References to Follow
- [Wu et al., Nature Comms 2022] FedPerGNN: third-party-server neighbour expansion. It is the graph-FedRec baseline that GPFedRec replaces, and it scores 9.31 HR on ML-1M here.
- [Yuan et al., WWW 2023] Interaction-level membership inference against FedRecs. It shows uploaded item tables leak interactions, which the thesis's threat model should cite.
- [Li, Long, Zhou, arXiv 2301.09109 / ICLR 2024] FedRAP, additive personalisation of item embeddings from the same group. The PDF is already in `Papers/raw/FedRAP-Zhiwei-2024.pdf`, not yet digested.
- [Zhang, Yuan, Yin, TKDE 2023] Attribute-inference attacks on FedRecs, for the threat model.
- [Zhang et al., WWW 2024] IFedRec, cold-start FedRec from the same group: the GPFedRec cold-start discussion made concrete.
