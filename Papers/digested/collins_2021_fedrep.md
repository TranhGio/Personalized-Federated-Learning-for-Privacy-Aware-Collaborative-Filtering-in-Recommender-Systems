# Exploiting Shared Representations for Personalized Federated Learning

- **Authors**: Liam Collins (UT Austin), Hamed Hassani (UPenn), Aryan Mokhtari (UT Austin), Sanjay Shakkottai (UT Austin)
- **Venue**: ICML 2021 (arXiv:2102.07078v3, revised Mar 2023)
- **Paper ID**: collins_2021_fedrep
- **Tags**: #method, #federated-learning, #personalization, #model-splitting, #non-iid, #optimization, #deep-learning

---

## 1. Core Idea

FedRep splits each client's model into a **global representation** $\phi:\mathbb{R}^d\to\mathbb{R}^k$ (averaged by the server) and a **low-dimensional local head** $h_i:\mathbb{R}^k\to\mathcal{Y}$ (never transmitted), with $k \ll d$. Each sampled client **first runs many steps on its head with $\phi$ frozen ($\tau_h$ large), then a few steps on $\phi$ with the fresh head frozen ($\tau_\phi$ small)**, and uploads only $\phi$. In the linear-regression case the paper proves linear convergence to the ground-truth subspace with per-client sample complexity $\Theta\big((d/n+\log n)\log(1/\epsilon)\big)$, versus $\Theta(d)$ for local-only learning. The paper calls the schedule "alternating minimization (head) – descent (representation)". Unlike FedPer, FedRep orders the updates head-first and gives the head many more steps, and it backs the design with theory.

## 2. Problem & Motivation

Under heterogeneous data, a single FedAvg model fits poorly, and with few samples per client local-only training fails (Fig. 1). Earlier personalized-FL methods (fine-tuning, Per-FedAvg, APFL/L2GD/Ditto interpolation or regularization, MOCHA) keep each client's subproblem **full-dimensional**. LG-FedAvg (Liang 2020) does the opposite split (local representation, global head). **FedPer (Arivazhagan 2019) uses the same body/head split as FedRep but updates head and body jointly with the same number of steps and gives no theory.** FedRep's premise comes from centralized deep learning: tasks share a feature representation, and the heterogeneity sits in the labels, which a client-specific head can capture.

## 3. Method

### 3.1 Objective Function

General personalized FL problem: $\min_{(q_1,\dots,q_n)} \frac1n\sum_i f_i(q_i)$, where $f_i(q_i)=\mathbb{E}_{(x,y)\sim\mathcal{D}_i}[\ell(q_i(x),y)]$. FedRep restricts $q_i = h_i\circ\phi$ and solves the **nested** objective:

$$
\min_{\phi\in\Phi}\; \frac{1}{n}\sum_{i=1}^{n}\; \min_{h_i\in\mathcal{H}}\; f_i(h_i\circ\phi) \qquad (2)
$$

**Linear instance (theory, §4):** $\phi \equiv B\in\mathbb{R}^{d\times k}$ and $h_i \equiv$ a vector in $\mathbb{R}^k$. The paper writes $\mathbf{w}_i$; it is renamed $h_i$ here so it does not clash with the thesis's global model $w$. The data follow $y_i = h_i^{*\top}B^{*\top}x_i$:

$$
\min_{B\in\mathbb{R}^{d\times k},\,H\in\mathbb{R}^{n\times k}} F(B,H) := \frac{1}{2n}\sum_{i=1}^{n}\mathbb{E}_{(x_i,y_i)}\big[(y_i - h_i^\top B^\top x_i)^2\big] \qquad (6)
$$

Per-round batch loss on $m$ fresh samples: $\hat f_i^t(h,B)=\frac{1}{2m}\sum_{j=1}^m (y_i^{t,j}-h^\top B^\top x_i^{t,j})^2$.

**Why a single model fails (§2.1):** with $f_i = \tfrac12\|Bh_i - B^*h_i^*\|^2$, FedRep's problem (3) has zero error, $\hat B\hat h_i = B^*h_i^*$, and recovers $\mathrm{col}(B^*)$ whenever the $h_i^*$ span $\mathbb{R}^k$. The single-model problem (4) is minimized iff $Bh = B^*\bar h^*$, where $\bar h^* = \frac1n\sum_i h_i^*$ (the **mean head**). Its residual error is $\frac{1}{2n}\sum_i\|\frac1n B^*\sum_{i'}(h_{i'}^*-h_i^*)\|_2^2$, which grows with head heterogeneity, and it cannot recover $\mathrm{col}(B^*)$ for $k>1$.

Notation mapping (paper → thesis):
- $\phi$ (or $B$) → the server's global model $w$. Only $\phi$ is server-side.
- $h_i$ → the local part of $\theta_i$, with $\theta_i = h_i\circ\phi$ (local head plus current global representation).
- $\mathcal{D}_i$ → $\mathcal{D}_i$; $M_i$ / $m$ = local sample count / per-round fresh batch.
- $n$ → $N$ (clients); $r$ → $C$ (participation fraction); $T$ → $R$ (rounds); $\tau_h+\tau_\phi$ → $K$ (local steps, split into two phases).
- **Paper's $\alpha$ = step size, NOT personalization level.** In thesis terms FedRep is a hard split: $\alpha\equiv1$ for $h_i$ (fully local) and $\alpha\equiv0$ for $\phi$ (fully global).
- $k$ = representation (= head input) dimension; $d$ = input dimension.

### 3.2 Algorithm

**Algorithm 1 (FedRep, general).** Parameters: $r$, step size $\alpha$, $\tau_h$, $\tau_\phi$, $T$. Initialize $\phi^0, h_1^0,\dots,h_n^0$ randomly.

**Server-side (each round $t=1..T$):**
1. Sample a client set $\mathcal{I}^t$ of size $rn$ uniformly.
2. Send the current representation ($\phi^{t-1}$) to the clients in $\mathcal{I}^t$.
3. Receive $\phi_i^{t,\tau_\phi}$ and aggregate uniformly: $\phi^t = \frac{1}{rn}\sum_{i\in\mathcal{I}^t}\phi_i^{t,\tau_\phi}$. The average is not weighted by data size.
4. For $i\notin\mathcal{I}^t$, the head is **frozen**: $h_i^{t,\tau_h}\leftarrow h_i^{t-1,\tau_h}$. Heads of non-participants go stale exactly as in the thesis's C=0.1 setting.

**Client-side ($i\in\mathcal{I}^t$):**
1. $h_i^{t,0}\leftarrow h_i^{t-1,\tau_h}$ (the cached local head).
2. **Head phase**, $\phi$ frozen at the received global: for $s=1..\tau_h$, $h_i^{t,s}\leftarrow \mathrm{GRD}\big(f_i(h_i^{t,s-1},\phi^{t-1}),\,h_i^{t,s-1},\alpha\big)$. GRD can be GD, SGD, or SGD with momentum. $\tau_h$ is chosen **large** so the inner $\min_{h_i}$ in (2) is nearly solved and the representation gradient is accurate.
3. $\phi_i^{t,0}\leftarrow\phi^{t-1}$.
4. **Representation phase**, head frozen at the fresh $h_i^{t,\tau_h}$: for $s=1..\tau_\phi$, $\phi_i^{t,s}\leftarrow\mathrm{GRD}\big(f_i(h_i^{t,\tau_h},\phi_i^{t,s-1}),\,\phi_i^{t,s-1},\alpha\big)$.
5. Upload only $\phi_i^{t,\tau_\phi}$. The head stays on the device.

**Algorithm 2 (linear regression, the analyzed variant):**
1. *Method-of-Moments init*: each client sends $Z_i=\frac1m\sum_j (y_i^{0,j})^2 x_i^{0,j}x_i^{0,j\top}$. The server sets $B^0\leftarrow U$, where $UDU^\top = \text{rank-}k\ \mathrm{SVD}(\frac1n\sum_i Z_i)$.
2. Each round, each sampled client draws **fresh** $m$ samples and computes the **exact** head $h_i^{t+1}=\arg\min_h \hat f_i^t(h,B^t)$ in closed form, at a cost of $O(mk^2+k^3)$.
3. It takes **one** GD step on the representation: $B_i^{t+1}= B^t-\eta\nabla_B\hat f_i^t(h_i^{t+1},B^t)$.
4. The server averages and orthonormalizes: $\bar B^{t+1}=\frac{1}{rn}\sum_{i\in\mathcal{I}^t}B_i^{t+1}$, then $(B^{t+1},R^{t+1})\leftarrow \mathrm{QR}(\bar B^{t+1})$.

**Theory (Theorem 1).** Assumptions:
- (A1) $x_i$ i.i.d., mean 0, covariance $I_d$, $I_d$-sub-Gaussian.
- (A2, client diversity) $\bar\sigma_{\min,*}:=\min_{|\mathcal{I}|=rn}\sigma_{\min}(\frac{1}{\sqrt{rn}}H^*_{\mathcal{I}})>0$, i.e. any $rn$ ground-truth heads span $\mathbb{R}^k$.
- (A3) $\|h_i^*\|_2=\sqrt{k}$ and $B^*$ orthonormal. This gives row-wise incoherence and is relaxable to $\le O(\sqrt k)$.

Let $E_0 := 1-\mathrm{dist}^2(B^0,B^*)$, where $\mathrm{dist}(B_1,B_2)=\|\hat B_{1,\perp}^\top\hat B_2\|_2$ is the principal-angle distance, and let $\kappa=\bar\sigma_{\max,*}/\bar\sigma_{\min,*}$. If

$$
m \ge c\Big(\frac{\kappa^4k^2 d}{E_0^2\, rn} + \frac{\kappa^4 k^3\log(rn)}{E_0^2}\Big),\quad \eta\le \frac{1}{4\bar\sigma_{\max,*}^2},
$$

then $\mathrm{dist}(B^T,B^*)\le(1-\eta E_0\bar\sigma_{\min,*}^2/2)^{T/2}\,\mathrm{dist}(B^0,B^*)$ with probability $\ge 1-Te^{-100\min(k^2\log(rn),\,d)}$. This is **linear (exponential) convergence**.

What this implies:
- Total samples per client to reach $\epsilon$ accuracy: $\Theta\big(\kappa^4k^2(d/rn + k\log(rn))\log(1/\epsilon)\big)$.
- Treating $k,\kappa,r$ as constants, this is $\Theta((d/n+\log n)\log(1/\epsilon))$, versus $\Theta(d)$ for local-only. **Federation helps whenever $1\ll n\ll e^{\Theta(d)}$.**
- A **new client** needs only $\Theta(k)$ samples, since it fits just a $k$-dim head.
- MoM init (Thm 2, adapted from Tripuraneni et al. 2020a, needs Gaussian $x$) keeps $E_0$ bounded away from 0 without changing the complexity beyond log factors.
- Proof novelty: the problem is matrix sensing with rank-1, **row-wise sparse** measurements. RIP fails when $m\le d/2$ (Claim 1), so the proof uses row-wise incoherence instead. It is the paper's claimed first convergence proof for alternating minimization–descent in matrix sensing, and it matches Zhong et al. (2015) up to $O(\kappa^2)$.

### 3.3 Key Hyperparameters

| Hyperparameter | Symbol | Default Value | Role |
|---|---|---|---|
| Participation rate | $r$ ($C$) | 0.1 (all real-data runs); 0.1 or 1.0 (synthetic) | Clients per round = $rn$ |
| Head local updates | $\tau_h$ | **10 local epochs** (real data); exact argmin (linear theory) | Nearly solve the inner $\min_h$ |
| Representation local updates | $\tau_\phi$ | **1 epoch** (CIFAR10, n=100); **5 epochs** (all others); 1 GD step (theory) | Few steps on the shared part |
| Learning rate | $\alpha$ / $\eta$ | 0.01, tuned in {0.001, 0.01, 0.1}, same for head and body; Fig. 5 uses 0.1 | Theory: $\eta\le 1/(4\bar\sigma^2_{\max,*})$ |
| Optimizer | — | SGD with momentum 0.5 | Same for all methods |
| Batch size | — | 10 (CIFAR, FEMNIST), 4 (Sent140); 50 in Fig. 5 | — |
| Rounds | $T$ ($R$) | 100 (CIFAR), 200 (FEMNIST), 50 (Sent140); 200 in Fig. 5 | — |
| Head-only fine-tune (+FT baselines, new clients) | — | 10 epochs SGD, representation frozen | Evaluation protocol |
| Synthetic dims | $d,k,m$ | $(10,2,5)$ Fig. 3; $(20,2,\cdot)$ Figs. 1/4 with $n=100$ | Label noise variance $10^{-3}$ |

### 3.4 Architectural Decisions

1. **Head = final FC layer only (weights + bias)**, for every architecture. The head is deliberately low-dimensional ($k\ll d$), so each client's local problem is small and nearly solvable every round.
2. **Head first, then representation.** The representation gradient is taken with a head that is (nearly) optimal for the current $\phi$, which makes the $\phi$-update an accurate step on the outer objective of (2). Fig. 3 shows that replacing the exact head solve with 1 or 10 GD steps (GD-GD, 10GD-GD) slows convergence and **removes the benefit of more clients $n$**.
3. **Many head steps are safe; many body steps are not.** Local head steps cannot cause client drift on the shared part, because the head is never averaged. FedRep is never hurt by more head epochs (Fig. 5), whereas FedAvg does not improve with more local epochs.
4. **Hard split, no interpolation.** There is no $\alpha$-mixing of local and global models, unlike APFL, and no proximal coupling, unlike Ditto/L2GD.
5. **Server QR orthonormalization** appears only in the linear variant, to remove the $B\to BQ$ ambiguity. The real-data runs use plain averaging.

## 4. Implementation Notes

- **Framework used in paper**: PyTorch, built on the LG-FedAvg codebase. Code: https://github.com/lgcollins/FedRep. All baselines (FedPer, APFL, Ditto, L2GD, Per-FedAvg, SCAFFOLD, FedProx) were reimplemented in that one codebase.
- **Model architecture**:
  - CIFAR10/100: 5-layer CNN (2 conv + 3 FC).
  - FEMNIST: MLP with 2 hidden layers.
  - Sent140: frozen 300-d GloVe embeddings, then an LSTM, then 2 FC decoding layers.
  - Head = final FC layer. The paper reports no head-size ablation.
- **Training details**: SGD with momentum 0.5, lr 0.01, batch 10 (4 for Sent140), r = 0.1, random init for all methods (no FedAvg warm-start, unlike the LG-FedAvg paper). Each round: 10 epochs on the head, then 1 or 5 epochs on the representation. **Every other method runs the same number of epochs as FedRep's representation phase**, so FedRep spends 10 extra head epochs per round compared with FedPer.
- **Data preprocessing**:
  - CIFAR: split into $Sn$ single-class shards, $S$ classes per client, $50000/n$ samples per client.
  - FEMNIST: 10-letter subset, log-normal allocation (Li et al. 2019), 150 clients.
  - Sent140: natural per-author split, 183 users with at least 50 tweets.
  - Data via LEAF.
- **Evaluation**: average local test accuracy over **all** users, averaged over the **final 10 rounds**, then over 5 repetitions (Table 1 reports **no std**). +FT methods: train the global model fully, then fine-tune the head for 10 epochs per client. The paper does not say whether the stale heads of non-sampled clients are refreshed before evaluation.
- **Tricks & gotchas**:
  - In the synthetic runs, FedRep converges the same from random or projected-GD initialization, while GD-GD is highly init-sensitive and often fails from random init.
  - The Fig. 5 runs use (batch 50, lr 0.1) and score lower than Table 1's (10, 0.01).
  - Fig. 3's text says "four values of n" but only three are plotted (10, 100, 1000).

## 5. Experimental Results

### 5.1 Datasets Used

**No recommendation, CF, or MF dataset and no ranking metric are used.** Columns are adapted: clients / samples per client / heterogeneity.

| Dataset | Clients ($n$) | Samples/client | Heterogeneity | Model |
|---|---|---|---|---|
| Synthetic linear regression | 5–1000 | $m$ = 5–10 per round (fresh) | Distinct $h_i^*$, shared $B^*$ | Two-layer linear |
| CIFAR10 | 100 / 1000 | 500 / 50 | $S\in\{2,5\}$ classes per client | 5-layer CNN |
| CIFAR100 | 100 | 500 | $S\in\{5,20\}$ | 5-layer CNN |
| FEMNIST (10 letters) | 150 | avg 148, min 50 | log-normal, ~3 classes | 2-hidden-layer MLP |
| Sent140 | 183 | avg 72, min 50 | natural per-user, 2 classes | GloVe + LSTM |
| FEMNIST-digits (new clients) | 100 new | 500 for fine-tuning + 100 for test | 5 classes per client | Head re-fit only |

### 5.2 Key Results

**Table 1: average local test accuracy (%), r = 0.1, mean of 5 runs.**

| Method | C10 (100,2) | C10 (100,5) | C10 (1000,2) | C100 (100,5) | C100 (100,20) | Sent140 | FEMNIST |
|---|---|---|---|---|---|---|---|
| Local Only | **89.79** | 70.68 | 78.30 | 75.29 | 41.29 | 69.88 | 60.86 |
| FedAvg | 42.65 | 51.78 | 44.31 | 23.94 | 31.97 | 52.75 | 51.64 |
| FedAvg+FT | 87.65 | 73.68 | 82.04 | **79.34** | 55.44 | 71.92 | 72.41 |
| FedProx / +FT | 39.92 / 85.81 | 50.99 / 72.75 | 21.93 / 75.41 | 20.17 / 78.52 | 28.52 / 55.09 | 52.33 / 71.21 | 18.89 / 53.54 |
| SCAFFOLD / +FT | 37.72 / 86.35 | 47.33 / 68.23 | 33.79 / 78.24 | 20.32 / 78.88 | 22.52 / 44.34 | 51.31 / 71.49 | 17.65 / 52.11 |
| Fed-MTL | 80.46 | 58.31 | 76.53 | 71.47 | 41.25 | 71.20 | 54.11 |
| PerFedAvg | 82.27 | 67.20 | 67.36 | 72.05 | 52.49 | 68.45 | 71.51 |
| LG-FedAvg | 84.14 | 63.02 | 77.48 | 72.44 | 38.76 | 70.37 | 62.08 |
| L2GD | 81.04 | 59.98 | 71.96 | 72.13 | 42.84 | 70.67 | 66.18 |
| APFL ($\alpha$ = 0.25 tuned) | 83.77 | 72.29 | 82.39 | 78.20 | 55.44 | 69.87 | 70.74 |
| Ditto | 85.39 | 70.34 | 80.36 | 78.91 | **56.34** | 71.04 | 68.28 |
| FedPer | 87.13 | 73.84 | 81.73 | 76.00 | 55.68 | 72.12 | 76.91 |
| **FedRep** | 87.70 | **75.68** | **83.27** | 79.15 | 56.10 | **72.41** | **78.56** |

- FedRep is best in 4 of 7 settings and within 2.1 points of the best in the other 3. It loses to Local Only on CIFAR10 (100,2), to FedAvg+FT on CIFAR100 (100,5), and to Ditto on CIFAR100 (100,20).
- **FedRep − FedPer: +0.57, +1.84, +1.54, +3.15, +0.42, +0.29, +1.65.** FedRep wins all 7, but by small margins, with no std reported and with 10 extra head epochs per round.
- **FedRep − FedAvg+FT: +0.05, +2.00, +1.23, −0.19, +0.66, +0.49, +6.15.** Head-only fine-tuning of a global model recovers most of FedRep's gain, except on FEMNIST.
- FedRep beats APFL ($\alpha$-interpolation) in all 7 settings, by 0.66 to 7.82 points.
- **Fig. 5 (CIFAR10, n = 100, S = 2, head epochs $E\in\{1,10,20,40\}$)**: FedRep reaches about 83–84% with $E=1$ and about 86–87% with $E\ge10$. Accuracy is monotonically non-decreasing in $E$. FedAvg stays at about 35–50% and oscillates, and more epochs do not help it.
- **Fig. 6 (new clients, FEMNIST letters → digits, head re-fit on 500 samples)**: after 1 → 10 head epochs, FedRep goes from about 36% to about 47%, L2GD about 35% to 45%, LG-FedAvg about 34% to 44.7%, FedProx about 30% to 36%, PerFedAvg about 26% to 35%, FedAvg about 28% to 32%, and APFL about 22% to 30%. FedPer is not shown.
- **Synthetic**:
  - Fig. 1 ($d=20$, $k=2$, $n=100$): FedRep has low MSE at every sample size; Local Only fails when $m<d$; FedSGD fails everywhere.
  - Fig. 4: a new client using FedRep's $B$ is accurate once $m_{new}\ge k$.
  - Fig. 3 ($d=10$, $k=2$, $m=5$, $r=0.1$): FedRep reaches $\mathrm{dist}\approx10^{-5}$ in about 50 rounds for $n\in\{100,1000\}$, while GD-GD and 10GD-GD are far slower. At $n=10$ nothing converges well.

### 5.3 Ablation Highlights

1. **Number of head steps ($\tau_h$)** is the paper's main lever. Exact argmin beats 10GD-GD, which beats GD-GD (Fig. 3). The gap **grows with $n$** (Fig. 3; Fig. 7 with $r=1$, $n$ = 5 to 500, $m$ = 5 or 10, 10 trials, 95% CIs). On real data, head epochs 1 → 10 add about 3 points, and more epochs never hurt (Fig. 5, one setting only).
2. **Number of clients**: more clients means faster representation recovery, as long as the head is solved.
3. **Heterogeneity and data size** (Table 1): personalized methods help most at low $S$. With 50 samples per client (C10, n = 1000), FedRep stays on top (83.27) while Local Only drops to 78.30.
4. **Not ablated**: $\tau_\phi$, head size and which layers form the head (always the final FC layer), the joint vs head-first order at matched compute, and eval-time head staleness under partial participation.

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas

1. **PFedRec and split learning are FedRep models; with one-hot item input, FedRep's linear model is MF.** With $x=e_j$, $h_u^\top B^\top e_j = h_u^\top q_j$: $B$ is the item-embedding table and the head is a user vector.

   | Module | $\phi$ (global) | $h_u$ (local head) | Schedule per participation |
   |---|---|---|---|
   | FedRep | body ($d\to k$) | final FC layer | $\tau_h$ = 10 head epochs, **then** $\tau_\phi$ = 1–5 body epochs |
   | `federated-pfedrec` | `embedding_item` (3706×32) | `affine_output` (32 + 1 params) + sigmoid | **per batch**: 1 head step, then 1 item step, so $\tau_h=\tau_\phi=$ #batches, interleaved |
   | `federated-personalized-cf` | item emb (3706×128) + item bias + global bias | $p_u\in\mathbb{R}^{128}$, $b_u$ | joint BPR step (**FedPer-style**) |
   | `federated-adaptive-personalized-cf` | same as split | $\alpha p_{local}+(1-\alpha)p_{global}$ + PersonalMLP | joint; the head is pulled toward the population mean |
   | `federated-baseline-cf` | everything | none | — |

   Consequence: PFedRec (0.3352) and split learning (0.2275) are the **same architecture family**: a linear head on a shared item table. Their gap therefore comes from the **training recipe** (BCE vs BPR, alternating vs joint updates, the ~29,648 item-LR boost, dim 32 vs 128), not from where personalization lives. Say this explicitly in the "where should personalization live" chapter.

2. **The stale-head problem has a precise reading in FedRep.**
   - Algorithm 1 freezes non-participants' heads ($h_i^{t}\leftarrow h_i^{t-1}$), the same as PFedRec under C = 0.1, where each user is sampled about 10 of 100 rounds and $\phi$ moves for about 9 rounds between visits.
   - FedRep's fix on re-entry is to **re-solve the head against the current $\phi$ before touching $\phi$**.
   - PFedRec does something much weaker. With batch 256 and 4 negatives, any user with ≲50 interactions has one batch per epoch, so a returning sparse user gets **one** head SGD step (lr 0.1) and then immediately one item step. That is exactly FedRep's **GD-GD** comparator: slowest in Fig. 3, and it does not improve with more clients.
   - So FedRep predicts that staleness hurts in **two** places: (a) at evaluation, where the stale head meets the final $\phi$, and (b) **during training**, where returning users push item gradients through a head that has not been re-fit. That can degrade $\phi$ itself.

3. **Final calibration = FedRep's head phase = the "+FT" protocol.**
   - The thesis fix (one local epoch for all 6040 users, `lr_eta=0` so items are frozen; `federated-pfedrec/federated_pfedrec/server_app.py` D-06.7) is "train $h_u$ with $\phi$ frozen".
   - In the paper, head-only fine-tuning (FedAvg+FT: 10 epochs) is within 0.05–2.0 points of FedRep on CIFAR10 and even beats it on CIFAR100 (100,5).
   - The thesis calibration closes 45% of the staleness gap (0.3352 → 0.3447 against the in-loop ceiling 0.3563) with roughly **one SGD step** for most users. That suggests the heads are still under-fit to the final $\phi$.
   - FedRep's Fig. 5 (more head epochs never hurt) predicts that more calibration epochs should close more of the gap.

4. **The counterfactual chain is an empirical test of FedRep's premise.** On a fixed item table $\phi$, the personal head scores 0.3447. The population-mean head (literally the solution of FedRep's single-model problem (4), $\bar h$) scores 0.2510 (−27.2%). Shuffled heads score 0.1973 (−42.8%), and a random head 0.0741 (−78.5%).
   - The mean-head drop is FedRep's "heterogeneity error" measured in CF.
   - Shuffled heads doing worse than the mean head shows the heads carry user-specific information, not generic calibration.
   - The killed capstone (blending the personal head toward the mean head, oracle $\alpha^*=1.0$, Δ −0.0002) agrees with FedRep's hard-split formulation and with APFL losing to FedRep in all 7 Table 1 settings.

5. **Theoretical regime check: the thesis is inside the structural regime but outside the theoretical one.**

   | FedRep condition | Thesis (PFedRec, k = 32, d = 3706, rn = 604) | Holds? |
   |---|---|---|
   | $k\ll d$ | 32/3706 = 0.9% (split: 128/3706 = 3.5%) | yes |
   | A2: any $rn$ heads span $\mathbb{R}^k$ | 604 ≫ 32 | plausible |
   | A1: isotropic sub-Gaussian $x$ | one-hot items with popularity-skewed covariance. This is **matrix completion**, which needs incoherence of the item factors too, and the paper does not cover it | no |
   | MSE, realizable, fresh samples each round | BCE / BPR on a fixed local set with resampled negatives | no |
   | Exact head minimization | 1 SGD step for users with ≲50 interactions | no |
   | $m \gtrsim k^2d/rn + k^3\log rn$ | ≈ 6.3K + 210K ≈ **2.2×10⁵** samples per round, even with $c=\kappa=E_0=1$; ML-1M mean is ≈ 166 ratings per user | no (off by about 3 orders of magnitude) |
   | MoM init + server QR | random init + plain FedAvg | no |

   Cite the theory as **motivation** (low-dimensional heads make the local problem easy; collaboration reduces per-client sample complexity), never as a guarantee for the thesis models.

### 6.2 Potential Integration Points

1. **Cheap ablation, no code change:** sweep `final-calibration-epochs` ∈ {1, 3, 10} with `final-calibration-freeze-items=true`. The knob already exists in `federated-pfedrec/pyproject.toml`, and also in `federated-personalized-cf/pyproject.toml`. Ten epochs mirrors FedRep's +FT protocol. The cost is about one extra round per epoch.
   - Report NDCG@10 overall and per sparse/medium/dense group.
   - Watch for a 33-parameter head over-fitting about 90 BCE samples for sparse users.
   - Keep the paper-compatible PFedRec numbers as the calibration-baseline row.
2. **In-loop FedRep schedule (opt-in flag, e.g. `head-first-epochs`)** in `train_pfedrec_single_user` (`federated_pfedrec/task.py`):
   - Before the existing per-batch alternation, run $\tau_h$ epochs with only `optimizer_local` stepping (items frozen, the same path calibration uses).
   - The default must stay 0, because the module's claude.md forbids changing PFedRec's alternation for the reproduction. Label the result "PFedRec + FedRep schedule".
   - This tests whether in-loop staleness (point 2(b) above) degrades $\phi$. If in-loop and full-population NDCG both rise, staleness was hurting representation learning, not just evaluation.
3. **Split-learning analogue:** freeze `item_embeddings` / `item_bias` / `global_bias` and fit $p_u$, $b_u$ for $\tau_h$ epochs before the joint BPR step. This is also FedRecon's "reconstruct locals, then update globals" scheme. It moves the module from FedPer-style joint updates to FedRep-style head-first updates, a clean single-variable ablation.
4. **Threat-model section:** FedRep's MoM init uploads $Z_i=\frac1m\sum_j y_j^2\,x_jx_j^\top$, and the paper claims this "does not compromise privacy". With one-hot items, $x_jx_j^\top=e_je_j^\top$, so $Z_i$ is a diagonal matrix whose support is **exactly the user's interacted-item set**. Cite this as an example of why image-domain privacy claims do not carry over to CF. More generally, FedRep, PFedRec, and the thesis modules never transmit the head, but the uploaded $\phi$-delta (item rows) still leaks which items were touched, softened in PFedRec by the sampled negatives.
5. **Suggested citation framing:** "PFedRec [Zhang et al., 2023] and our split-learning model instantiate the shared-representation / local-head decomposition $\min_\phi \frac1N\sum_i\min_{h_i} f_i(h_i\circ\phi)$ formalized by FedRep [Collins et al., 2021]. The item-embedding table plays the role of the shared representation and the per-user score head (or user embedding) the local head. FedPer [Arivazhagan et al., 2019] first proposed this base/personalization-layer split, with joint updates. Our final-calibration pass is FedRep's head-only phase applied once to every client after training."
   - Cite **both**: FedPer as the origin of the split, FedRep as the formal ancestor of PFedRec's alternating head-then-representation update and of the calibration fix.
   - For the split-learning module (joint BPR), FedPer is the closer *algorithmic* match and FedRep the closer *model* match (linear head on a shared table = MF).
   - Note that the PFedRec paper itself cites neither FedRep nor FedPer.

### 6.3 Limitations & Gaps

1. **No recommender experiment at all.** There is no CF/MF dataset, implicit feedback, negative sampling, ranking metric, or cold-start in the RecSys sense. The only link to MF is theoretical (matrix sensing).
2. **Scale mismatch.** At most 1000 clients and at most 100 per round, with 50–500 samples per client, against 6040 clients, 604 per round, and sparse users with 20–30 positives.
3. **Weak statistics.** Table 1 has no std or significance tests. The FedRep − FedPer margins are 0.3–3.2 points and are confounded by FedRep's 10 extra head epochs (head-first vs joint is never compared at matched compute). The $\tau_h$ ablation covers a single setting (Fig. 5).
4. **Evaluation-time staleness is unaddressed.** With r = 0.1, "average over all users" mixes fresh and stale heads, and the paper never says which. The thesis's full-population evaluation and calibration gap quantify something FedRep leaves implicit.
5. **The theory assumes away CF's key features:** non-isotropic one-hot inputs, a pairwise or BCE loss, fixed small local data, and a non-exact head (see the table in 6.1 item 5).
6. **No per-client adaptivity** (one $\tau_h$, one head size for every client) and no privacy mechanism (no DP or secure aggregation).

## 7. Key References to Follow

- **Arivazhagan et al., 2019 (FedPer)**: already digested; the joint-update predecessor that FedRep explicitly contrasts with.
- **Deng et al., 2020 (APFL)**: adaptive local/global model interpolation, the closest prior to the thesis's α-blending. It loses to FedRep in all 7 settings (PDF in `Papers/raw/APFL-Yuyang-2020.pdf`).
- **Liang et al., 2020 (LG-FedAvg)**: the inverse split (local representation, global head); FedRep's codebase.
- **Jain, Netrapalli & Sanghavi, 2013**: alternating minimization for low-rank matrix completion. It is the CF-native analogue of FedRep's analysis and covers the one-hot / incoherent setting FedRep does not.
- **Tripuraneni et al., 2020a; Du et al., 2020**: linear multi-task representation-learning theory and the Method-of-Moments initialization.
- *(Not cited by FedRep; contemporaneous)* **Singhal et al., 2021 (FedRecon)**: a FedRep-like "reconstruct local params with globals frozen, then update globals" scheme applied to MF. A PDF is in `Papers/raw/`. **FedRAP (2024) cites FedRep, and FedTHE (2023) benchmarks against it.**
