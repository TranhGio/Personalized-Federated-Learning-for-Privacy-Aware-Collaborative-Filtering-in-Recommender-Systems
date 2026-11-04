# Federated Recommendation with Additive Personalization

- **Authors**: Zhiwei Li, Guodong Long (University of Technology Sydney), Tianyi Zhou (University of Maryland, College Park)
- **Venue**: ICLR 2024 (arXiv:2301.09109v4). Code: https://github.com/mtics/FedRAP
- **Paper ID**: li_2024_fedrap
- **Tags**: #method, #federated-learning, #personalization, #recommender-system, #collaborative-filtering, #implicit-feedback, #model-splitting, #regularization, #differential-privacy, #communication-efficiency

> **Notation clash**: the paper calls the global item matrix $\mathbf{C}$. In the thesis, $C$ is the client sampling fraction. In this digest the item matrix is always bold $\mathbf{C}$, and the sampling fraction is written $n_s/n$.

---

## 1. Core Idea
FedRAP gives each user its own view of the items by adding two tables: a sparse, server-aggregated **global item embedding** $\mathbf{C}\in\mathbb{R}^{m\times k}$ and a never-uploaded **local item embedding** $\mathbf{D}^{(i)}\in\mathbb{R}^{m\times k}$. User $i$ scores item $j$ with $(\mathbf{D}^{(i)}+\mathbf{C})_j$. Two regularizers shape the tables. An $L_1$ term keeps $\mathbf{C}$ sparse, which the paper says cuts communication. A difference term pushes $\mathbf{D}^{(i)}$ away from $\mathbf{C}$ so the two views complement each other. Both weights ramp up with a $\tanh$ curriculum. **What the code actually builds**: PFedRec's per-user `Linear(k,1)` score head plus one persistent local item table added on top. The paper calls this head the "user embedding" $u_i$. There is no user embedding in the code.

## 2. Problem & Motivation
Partially-local FedRecs (FedMF, FedNCF, FedRecon) personalize only the user side and share one item table, so all users see an item the same way. PFedRec (zhang_2023_pfedrec) personalizes item embeddings by post-tuning, but the tuned copy is thrown away after upload. FedRAP's authors (same UTS group) argue this "neglects knowledge sharing" on the item side. They also say a dense item table makes communication expensive. FedRAP keeps a shared part and a persistent personal part of each item embedding, and sparsifies the shared part.

## 3. Method

### 3.1 Objective Function
Data: $R\in\{0,1\}^{n\times m}$, with every observed rating binarized to 1. $\Omega$ is the set of observed entries plus sampled negatives (4 per positive in the experiments). There is one user per client. The prediction is

$$\hat r_{ij}=\sigma\big(\langle u_i,\ (\mathbf{D}^{(i)}+\mathbf{C})_j\rangle\big)\qquad\text{(code: } \sigma(W_i(\mathbf{D}^{(i)}_j+\mathbf{C}_j)+b_i),\ W_i\in\mathbb{R}^{1\times k}\text{)}$$

The full objective (Eq. 3) is

$$
\min_{U,\mathbf{C},\{\mathbf{D}^{(i)}\}}\ \sum_{i=1}^{n}\Big(\sum_{(i,j)\in\Omega}-\big[r_{ij}\log\hat r_{ij}+(1-r_{ij})\log(1-\hat r_{ij})\big]\;-\;\lambda_{(a,v_1)}\,\|\mathbf{D}^{(i)}-\mathbf{C}\|_F^2\Big)\;+\;\mu_{(a,v_2)}\,\|\mathbf{C}\|_1
$$

with the curriculum (Sec. 4.3)

$$\lambda_{(a,v_1)}=\tanh\!\big(\tfrac{a}{10}\big)\,v_1,\qquad \mu_{(a,v_2)}=\tanh\!\big(\tfrac{a}{10}\big)\,v_2$$

Where:
- $u_i\in\mathbb{R}^k$: the user's private vector. In the code it is the per-user `affine_output = Linear(k,1)` with bias, the same component as PFedRec's local head.
- $\mathbf{C}\in\mathbb{R}^{m\times k}$: the global item embedding (`item_commonality`). It is the only global parameter, so thesis $w=\mathbf{C}$.
- $\mathbf{D}^{(i)}\in\mathbb{R}^{m\times k}$: the local item embedding (`item_personality`). It persists across rounds and is never uploaded. Thesis $\theta_i=(u_i,\mathbf{D}^{(i)})$.
- $-\lambda\|\mathbf{D}^{(i)}-\mathbf{C}\|_F^2$: the difference term. It **maximizes** the distance between the two views (Eq. 2: $\max\sum_i\|\mathbf{D}^{(i)}-\mathbf{C}\|_F^2$).
- $\mu\|\mathbf{C}\|_1$: the sparsity term on the shared table. It is meant to drop redundant shared information and save communication.
- $a$: the round index (`iteration` in code). $v_1, v_2$: the maximum weights. $\tanh(a/10)$ reaches 0.76 at round 10, 0.96 at round 20 and 0.995 at round 30. **The ramp is effectively over by round about 30.**

**Sign inconsistency (important if you implement this)**: the main text (Eqs. 2-3) and the official `model/loss.py` both use $-\lambda$, which pushes the tables apart. The appendix gradients (Eqs. 11-15) are written for $+\lambda$, which pulls them together: $\nabla_{\mathbf{C}}\mathrel{+}=2\lambda(\mathbf{C}-\mathbf{D}^{(i)})$ and $\nabla_{\mathbf{D}^{(i)}}\mathrel{+}=2\lambda(\mathbf{D}^{(i)}-\mathbf{C})$. With $-\lambda$ the objective has no lower bound in the rows of $\mathbf{D}^{(i)}$ that the BCE term never touches. The code keeps this in check only because it uses **mean**-reduced losses: `- lam * MSELoss(D, C) + mu * L1Loss(C, 0)`. That divides the effective per-entry weights by $mk$ (118,592 on ML-1M at $k=32$).

**Gradients (Appendix A.2)**: the BCE part simplifies to $\partial/\partial(\cdot)_j=(\hat r_{ij}-r_{ij})\,u_i$ for both $\mathbf{C}_j$ and $\mathbf{D}^{(i)}_j$, and to $\sum_j(\hat r_{ij}-r_{ij})(\mathbf{D}^{(i)}+\mathbf{C})_j$ for $u_i$. The paper prints this in a garbled expanded form, Eqs. 9/12/15. The appendix handles $L_1$ with a proximal soft-threshold step, $\mathbf{C}\leftarrow S_\theta(\mathbf{C}-\eta\nabla_{\mathbf{C}}L)$ with $S_\theta(v)=\max(v-\theta,0)-\max(-v-\theta,0)$, but never gives $\theta$. **The code does not soft-threshold.** It takes an autograd subgradient of `L1Loss`, so $\mathbf{C}$ is never exactly zero.

### 3.2 Algorithm
**Server-side** (Alg. 1):
1. Initialize $\mathbf{C}$. Each client initializes $u_i$ and $\mathbf{D}^{(i)}$. In the code every user starts from the same `nn.Embedding` default init, $\mathcal{N}(0,1)$, so all $\mathbf{D}^{(i)}$ begin identical and non-zero.
2. For $a=1,\dots,t_1$ ($R=100$): sample $n_s$ of the $n$ clients at random (code: `clients_sample_ratio`, default **1.0**, i.e. full participation, which the paper never states).
3. Broadcast $\mathbf{C}$. Each sampled client returns $\mathbf{C}^{(i)}$.
4. Aggregate with an **unweighted** mean, $\mathbf{C}\leftarrow\frac{1}{n_s}\sum_{i\in S_a}\mathbf{C}^{(i)}$. Set $\lambda(a)$ and $\mu(a)$ for the next round. Multiply both learning rates by `decay_rate`=0.97.
5. Evaluate all $n$ users every round. Each user's stored $(u_i,\mathbf{D}^{(i)})$ is combined with the latest aggregated $\mathbf{C}$. Users with no stored copy fall back to the initial model, which is a cold start. This is the same failure mode as the thesis's D-06 bug.

**Client-side** (ClientUpdate for client $i$):
1. Load $\mathbf{C}$ from the server. Restore $u_i$ (head) and $\mathbf{D}^{(i)}$ from the per-user store, or from the initial state on first contact.
2. Sample 4 negatives per positive from all items minus the user's **full** interaction set, which includes the val and test items. Resample every round.
3. For $b=1,\dots,t_2$ ($K=10$ local epochs, mini-batch 2048, stop early if the relative loss change is below `tol`=1e-4), take one **joint** SGD step on all three blocks. The appendix calls this "alternating", but all three gradients are evaluated at the same point:
   $u_i\leftarrow u_i-\eta_u\nabla_{u_i}L$, $\ \mathbf{C}^{(i)}\leftarrow\mathbf{C}-\eta_{\text{emb}}\nabla_{\mathbf{C}}L$, $\ \mathbf{D}^{(i)}\leftarrow\mathbf{D}^{(i)}-\eta_{\text{emb}}\nabla_{\mathbf{D}^{(i)}}L$.
   The code uses one `torch.optim.SGD` with three param groups and `weight_decay`=1e-4 on everything. A fresh `ExponentialLR(γ=0.95)` steps once per batch.
4. Upload $\mathbf{C}^{(i)}$ as a dense $m\times k$ state dict. `affine_output.*` and `item_personality` are deleted before upload.
5. Inference: $\hat r_i=\sigma(\langle u_i,\mathbf{D}^{(i)}+\mathbf{C}\rangle)$.

**Optional DP (Sec. 3.3)**: clip the client gradient with respect to $\mathbf{C}$, $\nabla\mathbf{C}^{(i)}\leftarrow\nabla\mathbf{C}^{(i)}\min\{1,\tau/\|\nabla\mathbf{C}^{(i)}\|_F\}$. The sensitivity of $\mathbf{C}$ is at most $2\eta\tau/n_s$. Add Gaussian noise (multiplier $z$) through Opacus and account for it with the moments accountant (Abadi 2016). **No $\varepsilon$ value is ever reported.**

### 3.3 Key Hyperparameters
| Hyperparameter | Symbol | Default Value | Role |
|---|---|---|---|
| Latent dim | $k$ | 32 (all methods) | Shared and local item tables, head input |
| Rounds | $t_1=R$ | 100 | Communication rounds |
| Local epochs | $t_2=K$ | 10 | Per participation. PFedRec's paper uses 1 |
| Batch size | $B$ | 2048 | All methods |
| Training negatives | — | 4 per positive | Resampled every round |
| Client fraction | $n_s/n$ | 1.0 (code default, not stated in paper) | Full participation |
| Difference-weight max | $v_1$ | grid $\{10^{-6},\dots,10^{0}\}$; code default 0.1 | Weight on $-\lambda\,\mathrm{MSE}(\mathbf{D},\mathbf{C})$ |
| Sparsity-weight max | $v_2$ | grid $\{10^{-3},\dots,10^{3}\}$; code default 0.1 | Weight on $\mu\,\mathrm{mean}|\mathbf{C}|$ |
| Curriculum | $\lambda,\mu$ | $\tanh(a/10)\cdot v$ | Ablations: fixed, sin, square, frac |
| Head LR | `lr_network` | 0.5 (code); README ML-100K example 1e-4 | SGD on $u_i$ / `affine_output` |
| Item-table LR | `lr_args` | 200 (code); README example 1e3 | SGD on $\mathbf{C}$ and $\mathbf{D}^{(i)}$. Huge, like PFedRec's $\eta\cdot m\cdot80$ |
| LR decay | — | ×0.97 per round; ExponentialLR 0.95 per local batch | Both on |
| Weight decay | `l2_regularization` | 1e-4 (README example 1e-6) | All parameters |
| DP clip / noise | $\tau$, $z$ | 0.1, 1.0 | $\varepsilon$ not reported |
| Eval | — | LOO, **99 sampled negatives**, K=10 | HR@10, NDCG@10, 5 repeats |
| Final per-dataset $v_1,v_2$ | — | **Not reported in paper** | README warns that results are "highly sensitive" to the hyperparameter combination |

### 3.4 Architectural Decisions
- **Additive rather than replacement personalization**: $\mathbf{D}^{(i)}+\mathbf{C}$ keeps collaborative item knowledge in $\mathbf{C}$ and adds a persistent per-user view. The paper presents this as the fix for PFedRec, which loses its item personalization each round.
- **Push apart rather than pull together**: Ditto, pFedMe and the thesis's $\delta$ pull the personal model toward the global one. FedRAP does the opposite. The paper says $\mathbf{C}$ should hold only user-agnostic structure; its t-SNE (Fig. 8) shows rated and unrated items mixed in $\mathbf{C}$ and separated in $\mathbf{D}^{(i)}$.
- **$L_1$ on $\mathbf{C}$ only**: $\mathbf{C}$ is the only uploaded table, so the paper treats sparsity as a communication lever. $L_1$ was chosen over squared Frobenius $\|\mathbf{C}\|_F^2$ ("FedRAP-L2") because of a marginal NDCG gain.
- **Curriculum**: early on, both tables are poorly trained, and the paper says forcing them apart then hurts. Starting from $\lambda=\mu=0$ ("full personalization") and ramping up is meant to avoid that.
- **No user embedding**: the "user embedding" is the score-head weight vector, as in PFedRec. Personalization therefore lives in the head **and** in the item table.

## 4. Implementation Notes
- **Framework**: PyTorch 1.13. Experiments ran on one RTX 3070 Ti laptop GPU.
- **Model**: `item_commonality = Embedding(m,k)` (global), `item_personality = Embedding(m,k)` (local), `affine_output = Linear(k,1)` (local), then a sigmoid. Per-client state is $mk+k+1$ floats, which is 118,625 on ML-1M at $k=32$. PFedRec's is 33.
- **Training**: BCE on binarized implicit feedback. Joint SGD with per-group LRs (0.5 for the head, 200 for both item tables), decayed per round and per batch. 10 local epochs, batch 2048, 100 rounds. Sparsity is logged as the fraction of $|\mathbf{C}|\le10^{-4}$.
- **Data**: ratings above 0 become 1. Users with fewer than 10 ratings are dropped. LOO by timestamp: latest interaction = test, second latest = validation, rest = train. The 99 eval negatives are drawn once from items the user never interacted with, and validation reuses them.
- **Code-audit findings** (from the GitHub repo, 2026-10-27; none of this is in the paper):
  1. **Model selection uses the test set**: `train.py` keeps the round with the highest *test* HR (`if hr >= best_test_hr`) and reports the *validation* HR/NDCG at that round.
  2. **The held-out item is never a training negative**: the training-negative pool is all items minus the user's full interaction set, which includes the val and test items, while the 99 eval negatives are in that pool. A free per-user, per-item table like $\mathbf{D}^{(i)}$ could in principle learn "items sampled as negatives go down", which would lower the 99 eval negatives and never the held-out item. **This is a hypothesis, not tested.** The paper's own ablation cuts against it: FedRAP-D, which has a local table and no $\mathbf{C}$, stays at PFedRec level. The NCF and PFedRec pipelines share this exclusion, and so does the thesis's `ExclusionTable`.
  3. `setItemCommonality` sets `.freeze = True`, which is not a PyTorch flag and does nothing. $\mathbf{C}$ is trained locally, as Alg. 1 says.
  4. **Uploads are dense**: the full $m\times k$ $\mathbf{C}^{(i)}$ goes up as a state dict. The $L_1$ term does not change the bytes sent.
- **Errata in the paper**: Tables 5/6 have the "w/ DP" and "w/o DP" columns swapped (the "w/ DP" FedRAP 0.9709 equals the no-noise value in Table 1). TaFeng's sparsity is printed as 78.88%, but its own counts give 97.4%. Appendix A cites a "Theorem 1 in Sec. 3.4" that does not exist; it means Property 1.

## 5. Experimental Results

### 5.1 Datasets Used
| Dataset | Users | Items | Interactions | Density |
|---|---|---|---|---|
| TaFeng (implicit) | 120 | 32,266 | 100,000 | 2.58% (paper: "sparsity 78.88%", inconsistent) |
| ML-100K | 943 | 1,682 | 100,000 | 6.30% |
| **ML-1M** | **6,040** | **3,706** | **1,000,209** | **4.47%** |
| Amazon-Instant-Video | 1,372 | 7,957 | 23,181 | 0.21% |
| LastFM-2K | 1,874 | 17,612 | 92,780 | 0.28% |
| QB-article (Tenrec, implicit) | 24,516 | 7,455 | 266,356 | 0.15% |

### 5.2 Key Results
**ML-1M, HR@10 / NDCG@10 (%), Tables 3/4.** Mean ± std over 5 runs. LOO with 99 sampled negatives. Full participation by default. 10 local epochs. Round chosen by test HR.

| Method | Type | HR@10 | NDCG@10 |
|---|---|---|---|
| NCF | Cent | 62.52 ± 0.70 | 35.60 ± 0.56 |
| LightGCN | Cent | 99.43 ± 0.31 | 90.43 ± 0.72 |
| CentRAP (centralized FedRAP) | Cent | 99.17 ± 0.43 | 86.89 ± 0.46 |
| FedMF (no encryption) | Fed | 65.95 ± 0.37 | 38.77 ± 0.29 |
| FedNCF (own reimplementation) | Fed | 61.31 ± 0.32 | 34.94 ± 0.42 |
| PFedRec | Fed | 68.59 ± 0.21 | 40.19 ± 0.21 |
| **FedRAP** | Fed | **93.24 ± 0.26** | **71.87 ± 0.28** |
| FedRAP + DP ($\tau$=0.1, $z$=1) | Fed | 91.80 | 70.35 |

**Other datasets, FedRAP vs PFedRec (HR@10 / NDCG@10, %)**: TaFeng 99.43/99.11 vs 98.33/98.03. ML-100K 97.09/87.81 vs 70.03/41.38. Video 46.53/28.17 vs 40.87/24.04. LastFM 23.29/10.99 vs 21.61/10.88. QB-article 53.98/24.75 vs 46.00/22.28. **FedRAP's lead is huge on MovieLens (+25 to +46 points) and small on the sparse datasets (+0.1 to +8 points).**

**Comparability with the thesis protocol**: the metric definitions match the thesis exactly: LOO by timestamp, 99 sampled negatives, HR@10 and NDCG@10. Four things differ. (i) Every client participates every round (the thesis samples 0.1). (ii) $K$=10. (iii) The best round is picked on the test set. (iv) There are no user groups. **The FedRAP, CentRAP and LightGCN magnitudes are not credible as targets.** Under this protocol, strong ML-1M models reach about HR@10 0.70-0.78 (NCF paper 73.0; PFedRec paper 73.26; CoFedRec 77.75). A centralized LightGCN at 99.43 is 37 points above NCF in the same table, which no other source supports. This digest does not explain the gap (see §4, audit item 2). PFedRec's row (68.59/40.19) is usable as a second external PFedRec reference point under full participation, about 4.7/4.2 points below PFedRec's own paper.

**Dataset-level sparsity, ML-1M subsets (Table 7)**: performance falls as sparsity rises. With 500 users: sparsity 81.33% / 95.49% / 98.62% gives HR 1.000 / 1.000 / 0.902 and NDCG 0.945 / 0.823 / 0.569. With 1000 items: 87.62% / 94.86% / 96.85% gives NDCG 0.893 / 0.776 / 0.700. **There is no per-user-group (sparse/medium/dense) breakdown.**

### 5.3 Ablation Highlights
On ML-100K, $v_1=v_2=0.1$, round 100 (values read off Figs. 5/6, ±0.02):

| Variant | What changes | HR@10 | NDCG@10 |
|---|---|---|---|
| FedRAP | full | ~0.94 | ~0.92 |
| FedRAP-No | no $\|\mathbf{C}\|_1$ | ~0.94 | ~0.89 |
| FedRAP-L2 | $\|\mathbf{C}\|_F^2$ instead of $L_1$ | ~0.94 | ~0.89 |
| FedRAP-D | local $\mathbf{D}^{(i)}$ only, no shared $\mathbf{C}$ | ~0.71 | ~0.43 |
| PFedRec | — | ~0.69 | ~0.41 |
| FedRAP-C | shared $\mathbf{C}$ only | ~0.46 | ~0.26 |

Curricula, on the validation set: tanh ~0.98/0.97, sin ~0.97/0.84, fixed ~0.95/0.67, square ~0.93/0.61, frac (decreasing) ~0.80/0.46.
- **What matters is having both tables.** Each one alone is no better than PFedRec. Sparsity adds almost nothing to accuracy: the No, L2 and L1 variants are within about 0.03 NDCG of each other.
- **The curriculum matters a lot for NDCG.** Fixed weights lose about 0.30 NDCG, and a decreasing schedule loses about 0.5.
- **Sparsity is weak.** In Fig. 7 (ML-100K, $k$=16), 67.36% of the $L_1$ $\mathbf{C}$ entries are above 1e-2 (88.94% for L2), and 4.19% are above 1e-1 (8.07% for L2). No bytes-transmitted figure is ever reported.
- **DP cost on ML-1M is small**: −1.54% HR and −2.11% NDCG. The paper says FedRAP loses less than PFedRec under DP: on ML-100K NDCG drops 8.72% for FedRAP against 14.23% for PFedRec (Tables 5/6, column labels corrected).

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas
1. **The item-perturbation flag needs this citation.** The adaptive module's `enable-item-perturbation` (`models/bpr_mf.py:455-486`: $q_{\text{eff}}[j]=q_{\text{global}}[j]+\delta_i[j]$, LOCAL, zero-init, `reg·‖δ‖²`, reg = 0.01) is a form of FedRAP's additive item personalization. Cite FedRAP as prior art, and cite PFedRec for transient post-tuning. Differences to state in the thesis:

   | | FedRAP | Thesis $\delta_i$ flag |
   |---|---|---|
   | Local table init | $\mathcal{N}(0,1)$, identical for all users | zeros |
   | Regularizer on local table | $-\lambda\,\mathrm{MSE}(\mathbf{D}^{(i)},\mathbf{C})$: push **away** from the global table | $+0.01\|\delta_i\|^2$: pull toward **zero**, i.e. toward the global table |
   | Sparsity on global table | $\mu\,\mathrm{mean}|\mathbf{C}|$ | none |
   | Curriculum | $\tanh(a/10)$ ramp on both weights | constant |
   | User side | score head only (`Linear(32,1)`) | local $p_u$, α-blended with the EMA prototype, plus PersonalMLP |
   | Loss / LR | BCE with 4 negatives, SGD with item LR 200 | BPR |
   | Reported outcome | +25 HR points over PFedRec (ML-1M, full participation) | collapse in cross-device runs: in-loop NDCG@10 about 0.07 vs about 0.24 with the flag off (BUG 1 FIX, `client_app.py:448`) |

   FedRAP's curriculum ramps the *regularizers*, not the additive term itself, so it would not have prevented the thesis's collapse. That collapse came from $\delta_i$ being active from round 1 with most rows untouched. The thesis's negative result stands as a cross-device finding, and it should be reported next to FedRAP's full-participation claim, not hidden.
2. **Place FedRAP in the "where should personalization live" chain.** Ordered by how much local state each method keeps per user at ML-1M: baseline (nothing) → split learning ($p_u$ + $b_u$, 129 floats at $d=128$) → PFedRec (score head, 33 floats; item post-tuning is discarded each round) → **FedRAP (the same 33-float head plus a persistent additive item table, 118,592 floats)**. FedRAP is "PFedRec + persistent item residual", minus PFedRec's two-pass alternating update. The thesis's counterfactual (PFedRec's population-mean head costs −9.4 points; shuffled heads 0.1973; random head 0.0741) shows the head carries PFedRec's personalization. FedRAP's own ablations point the same way: the item table alone (FedRAP-D) does not beat PFedRec, and the gain appears only when it is combined with the shared table and the head.
3. **Extend the counterfactual battery if FedRAP is ever run**: (own head, own $\mathbf{D}$), (own head, $\mathbf{D}=0$), (population-mean head, own $\mathbf{D}$), (own head, shuffled $\mathbf{D}$), (own head, own $\mathbf{D}$, $\mathbf{C}$ = initial). This separates head-side from item-side personalization, using the same offline-oracle machinery that killed the α-blend capstone.

### 6.2 Potential Integration Points
1. **Implementing FedRAP as a comparator.** Build it as a variant of `federated-pfedrec/`, not of the adaptive module.
   - Add `item_personality = Embedding(3706,32)` as LOCAL, cached next to `affine_output.pt`.
   - Use the loss `BCE − λ(r)·MSE(D,C) + μ(r)·mean|C|`, with $\lambda(r)=\tanh(r/10)\cdot0.1$ computed from `server_round` in the fit config.
   - Upload only `embedding_item`.

   **Cost**:
   - (a) The per-user cache is $6040\times118{,}592\times4$ B ≈ **2.87 GB** in fp32, against about 0.8 MB for PFedRec heads, on an NVMe the thesis notes say is near-full. It must not go to the HDD while a run is live. Storing only touched rows does not work: the dense init and the dense $-\lambda$ gradient change every row. That would need zero-init, which is a deviation from FedRAP.
   - (b) The 474 KB read/write per participation and the 2.87 GB D-06 full-population reload slow every round.
   - (c) There is no published per-dataset $(v_1,v_2,\text{lr})$ and the README warns of sensitivity, so this needs a search on the shared GPU.

   **Verdict**: possibly a stronger comparator on paper, but its published numbers cannot serve as a target, and a fair run costs about 1-2 days of code plus several 100-round runs. Given what the thesis still owes (paired significance tests, a threat model, a communication table), **cite and discuss; do not implement unless a reviewer asks.** If it is implemented, run the §6.1(3) counterfactuals and the eval-negative overlap diagnostic (point 2 below) before believing any win.
2. **Audit the evaluation protocol.** The thesis protocol has the same channel FedRAP's code has: the held-out item is excluded from training negatives. Any per-user, per-item local parameter can exploit that, including the $\delta_i$ flag and a FedRAP port. A cheap diagnostic: for each user, log the fraction of the 99 eval negatives that were drawn as training negatives during training, and correlate it with that user's hit. Mention this in the threat-to-validity section if any item-side local table is reported.
3. **Rows for the per-method communication table** (per participating client, per direction, fp32):

   | Method | Uploaded | Floats |
   |---|---|---|
   | FedRAP | dense $\mathbf{C}$ | 118,592 (474 KB) at $k=32$ |
   | PFedRec | item table | the same 118,592 |
   | Split learning | item embedding + item bias + global bias | about 478K at $d=128$ |
   | Baseline | everything | about 874K |

   FedRAP's $L_1$ "communication saving" is **not realized** in its code (dense upload) and is not measured in the paper. It is at most about 30% with a lossy 1e-2 threshold plus a bitmap. In the comparison table, FedRAP is the same as PFedRec.
4. **Privacy rows for the threat-model section.**
   - Property 1 is weaker than the paper suggests. The server knows the $\mathbf{C}$ it broadcast, so a **single** upload already reveals the accumulated update $\mathbf{C}^{(i)}-\mathbf{C}$; consecutive participation is not needed.
   - That update contains the dense term $\propto\lambda(\mathbf{D}^{(i)}-\mathbf{C})$ on **every** row. A server that knows $(\eta,\lambda(a))$ could in principle invert the linear dynamics on rows the BCE term never touched and recover the "private" $\mathbf{D}^{(i)}$ there. The rows that deviate from those dynamics expose the user's set of positive and sampled items. (This is my analysis, not the paper's.)
   - The DP add-on reports no $\varepsilon$.
   - PFedRec's upload carries no term that depends on the local head.

### 6.3 Limitations & Gaps
- **No sparse/medium/dense user-group results.** Table 7 varies sparsity only at the dataset level, and it shows FedRAP degrading as sparsity rises. That says nothing about the thesis's sparse-user claim.
- Only full participation was evaluated (code default 1.0). There are no C=0.1 cross-device results. Under C=0.1 each user's $\mathbf{D}^{(i)}$ trains in about 10 of 100 rounds, which is the regime where the thesis's $\delta_i$ collapsed.
- **The headline numbers cannot be relied on.** Rounds are selected on the test set. The ML-1M magnitudes are implausible. The objective's sign is inconsistent (§3.1). The DP table's columns are swapped. The per-dataset hyperparameters are missing. There are only 5 seeds and no significance tests.
- BCE pointwise loss only. No BPR or user-embedding variant.
- The paper's own limitation (App. C): every device must store two full $m\times k$ tables, and cold start for new items is not addressed.

## 7. Key References to Follow
- **[Zhang et al., IJCAI 2023] PFedRec** (zhang_2023_pfedrec) — FedRAP's code is PFedRec's head plus a local item table. The direct parent.
- **[Singhal et al., NeurIPS 2021] FedRecon** — partially-local FL, where the local part is reconstructed rather than stored. The alternative to caching a 118K-float $\mathbf{D}^{(i)}$ per user.
- **[Pillutla et al., ICML 2022] Partial model personalization** — convergence theory for split global/local parameters (FedAlt/FedSim). It covers both split learning and FedRAP.
- **[Li et al., ICML 2021] Ditto** — the opposite regularizer: it pulls the personal model toward the global one. This is the conceptual twin of the thesis's $\|\delta\|^2$, while FedRAP pushes the tables apart.
- **[Minto et al., RecSys 2021] Stronger privacy for federated CF with implicit feedback** — the DP mechanism FedRAP adopts. Relevant to the thesis threat-model section.
