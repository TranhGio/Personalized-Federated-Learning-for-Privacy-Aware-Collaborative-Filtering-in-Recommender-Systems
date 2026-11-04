# Test-Time Robust Personalization for Federated Learning

- **Authors**: Liangze Jiang (EPFL), Tao Lin (Westlake University). Equal contribution.
- **Venue**: ICLR 2023 (arXiv:2205.10920v4). Code: https://github.com/LINs-lab/FedTHE
- **Paper ID**: jiang_2023_fedthe
- **Tags**: #method, #federated-learning, #personalization, #model-splitting, #robustness, #non-iid, #test-time-adaptation (new tag)

---

## 1. Core Idea
FedTHE trains a two-head FL network: a shared feature extractor, a FedAvg'd **global head** and a local **personalized head**. At test time it blends the two heads' **logits** with a scalar weight $e$. The weight is optimized **separately for every single unlabeled test sample** by a label-free loss. That loss is entropy minimization plus feature alignment to stored train/test descriptors, mixed by a similarity-based weight. FedTHE+ then adds MEMO-style test-time fine-tuning (an unsupervised "linear-probe then fine-tune" scheme). The selling point is robustness to **test-time distribution shift**, especially label shift. In-distribution gains are small: +0.3 to +1.9 points on CIFAR10, depending on the comparator.

## 2. Problem & Motivation
Personalized FL methods (FedRep, FedRoD, APFL, Ditto, FedAvg+FT, kNN-Per) fit each client's *training* distribution. When test samples drift, accuracy collapses. The worst case is label shift: on Out-of-Client (OoC) tests at Dir(0.1) they score 22–33% vs 64% for plain FedAvg (Table 1). Test-time adaptation methods (TTT, MEMO, T3A, Tent) were designed for centralized i.i.d. settings and give little gain under FL label shift. The paper asks: *how to keep the global model's robustness while keeping personalized accuracy on the local distribution?*

## 3. Method

Notation used here: FedTHE's $e$ is the weight on the **global** head. The thesis's personalization level is therefore $\alpha = 1-e$.

### 3.1 Objective Function

**Federated training.** The global part $w = (w^e, w^g)$ (extractor + global head) solves standard FedAvg:

$$
\min_{w}\; \sum_{i=1}^{N} p_i\, \mathcal{L}_i(w), \qquad p_i = \frac{|\mathcal{D}_i|}{\sum_j |\mathcal{D}_j|}, \qquad \mathcal{L}_i(w) = \frac{1}{|\mathcal{D}_i|}\sum_{\xi \in \mathcal{D}_i} \ell(x_\xi, y_\xi; w)
$$

Client $i$'s personal model is $\theta_i = (w^e, w^{l}_i)$. Its head $w^l_i$ is trained on $\mathcal{D}_i$ **on top of the received, frozen extractor** $w^e$:

$$
w^{l}_i \leftarrow \arg\min_{w^l}\; \mathcal{L}_i(w^e_{\text{frozen}}, w^l) \quad (p \text{ SGD steps, 1 epoch})
$$

As a side study, the global part is trained with BalancedSoftmax (Ren et al. 2020), as in FedRoD.

**Test-time head ensemble (FedTHE) for one test sample $x_n$:**

$$
h_n = f_{w^e}(x_n), \quad \hat y^g = f_{w^g}(h_n), \quad \hat y^l = f_{w^l}(h_n), \quad \hat y(e) = e\,\hat y^g + (1-e)\,\hat y^l
$$

$$
\mathcal{L}_{EM}(e) = -\sum_{c} p_c(\hat y(e)) \log p_c(\hat y(e))
$$

$$
\mathcal{L}_{FA}(e) = e\,\lVert h'_n - h^g \rVert_2 + (1-e)\,\lVert h'_n - h^l \rVert_2, \qquad h'_n = \beta h_n + (1-\beta)\, h^{hist}
$$

$$
e^\star = \arg\min_{e}\; \mathcal{L}_{SLW} := \lambda_s\, \mathcal{L}_{EM} + (1-\lambda_s)\, \mathcal{L}_{FA}, \qquad \lambda_s = \cos\!\big(p(\hat y^g),\, p(\hat y^l)\big) \in [0,1]
$$

Where:
- $w^e, w^g$: feature extractor and global ("generic") head. Both are aggregated by FedAvg and are part of $w$.
- $w^l$ ($= w^l_i$): client $i$'s personalized head. It is the last FC layer and never leaves the client.
- $e \in (0,1)$: head-ensemble weight on the global head. It is parameterized as a softmax over **two learnable scalars**, $e = \sigma(a_g - a_l)$, initialized to 0.5 for every test sample.
- $p_c(\cdot)$: softmax class probability. $\mathcal{L}_{EM}$ is Shannon entropy over classes.
- $h^l_i = \frac{1}{|\mathcal{D}_i|}\sum_{x\in\mathcal{D}_i} f_{w^e}(x)$: **local descriptor**, the mean training feature under the *received global* extractor.
- $h^g = \sum_{m\in\mathcal{S}} \frac{|\mathcal{D}_m|}{\sum_{l}|\mathcal{D}_l|}\, h^l_m$: **global descriptor**. The server computes this data-size-weighted mean of uploaded local descriptors and syncs it every round with the model.
- $h^{hist}$: **test-history descriptor**, an EMA over the client's stream of test features. It is initialized to the first test feature and updated *after* optimizing $e$ as $h^{hist} \leftarrow \alpha_{h} h_n + (1-\alpha_{h}) h^{hist}$, using the unsmoothed $h_n$.
- $\beta$: smoothing of the test feature. With $\beta = 0.1$ the "test feature" in $\mathcal{L}_{FA}$ is **90% client test-stream history**, so FA is effectively a per-stream signal and not a per-sample one.
- $\mathcal{L}_{FA}$ is linear in $e$. Its gradient pushes weight toward whichever head's descriptor is closer to the (smoothed) test feature.
- $\lambda_s$: cosine similarity of the two heads' probability vectors. It does not depend on $e$. When the heads agree, the confident head wins (EM). When they disagree, feature alignment arbitrates (FA).

**FedTHE+ (test-time fine-tuning, MEMO).** With $e^\star$ fixed, it updates $(w^e, w^g, w^l)$ for $\tau$ steps on the marginal entropy over $B$ augmentations, $\mathcal{L} = H\big(\tfrac{1}{B}\sum_{b} p(f(\tilde x_b))\big)$ with $\tilde x_b = a_b(x_n)$. It then predicts with a **weight-space** head interpolation (Alg. 2, line 11):

$$
y^\star = f_{\,e^\star \tilde w^g + (1-e^\star)\tilde w^l}\big(f_{\tilde w^e}(x_n)\big)
$$

This is episodic: the weights are reset to $(w^e, w^g, w^l)$ for every test sample (Alg. 2, line 4). For linear heads on a shared feature, the weight-space form in line 11 is identical to the logit ensemble $\hat y(e)$. The authors use both forms interchangeably.

### 3.2 Algorithm

**Server-side (Alg. 1, per round $t = 1..R$):**
1. Sample clients $\mathcal{S}$ of size $C\cdot N$ (default $C = 1.0$).
2. Send $w^e, w^g$ (and $h^g$) to $\mathcal{S}$.
3. Aggregate. $w^e \leftarrow \sum_{m\in\mathcal{S}} \frac{|\mathcal{D}_m|}{\sum_l|\mathcal{D}_l|}\tilde w^e_m$. Same weighting for $w^g$ and for $h^g$ from the $\tilde h^l_m$.

**Client-side training (sampled client $m$):**
1. Initialize $\tilde w^e_m := w^e$, $\tilde w^g_m := w^g$, and $\tilde w^l_m :=$ its stored personal head.
2. **Local training**: run $K$ steps (5 epochs) of mini-batch SGD on $(\tilde w^e_m, \tilde w^g_m)$, the extractor and global head.
3. **Personalization**: run $p$ steps (1 epoch) of SGD on $\tilde w^l_m$ only, using the **received** $w^e$ (not the locally updated one) as a frozen extractor.
4. Compute $\tilde h^l_m$. Send $\tilde w^e_m, \tilde w^g_m, \tilde h^l_m$. Keep $\tilde w^l_m$ local.
5. Clients that are not sampled do nothing. Their heads go stale under $C < 1$. **No end-of-training re-personalization is described.**

**Client-side test time (Alg. 2, per unlabeled test sample $x_n$, online):**
1. $h_n = f_{w^e}(x_n)$. Then $h'_n = \beta h_n + (1-\beta) h^{hist}$.
2. $\hat y^g = f_{w^g}(h_n)$ and $\hat y^l = f_{w^l}(h_n)$ (the prediction uses $h_n$, not $h'_n$).
3. Set $e = 0.5$. Run $s = 20$ Adam steps (lr 0.1) on $\mathcal{L}_{SLW}$ with respect to $e$ (the two scalars). This gives $e^\star$.
4. $h^{hist} \leftarrow \alpha_h h_n + (1-\alpha_h) h^{hist}$.
5. FedTHE: predict $\hat y(e^\star)$. FedTHE+: run MEMO FT for $\tau = 3$ steps (16 augmentations, lr 5e-4) from a fresh copy of the weights, then predict $y^\star$ as above.

### 3.3 Key Hyperparameters
| Hyperparameter | Symbol | Default Value | Role |
|---|---|---|---|
| Clients | $N$ | 20 (ablation: 100) | Cross-silo scale (CIFAR10.1 has only 2000 samples) |
| Participation | $C$ | 1.0 (ablation: 0.1, 0.4) | Default is full participation |
| Rounds | $R$ | 100 (CIFAR10), 300 (ImageNet-32) | — |
| Local training | $K$ | 5 epochs on $(w^e, w^g)$ | Global objective |
| Personalization | $p$ | 1 epoch on $w^l$, frozen received $w^e$ | 2/4/8 epochs change little (Table 9) |
| Optimizer (train) | — | SGD lr 0.01, wd 5e-4 (CNN); SGDm 0.9 (ResNet20-GN); Adam 1e-3 (CCT4) | — |
| Batch size | — | 32 (CIFAR10), 128 (ImageNet-32) | — |
| Ensemble weight init | $e_0$ | 0.5 (via two equal logits) | Reset for every test sample |
| Ensemble optimizer | — | Adam, lr 0.1 (CNN/ResNet), 0.01 (CCT4) | — |
| Ensemble steps | $s$ | 20 | 20 Adam steps of lr 0.1 move the logit gap by at most about 4, so $e \in [\approx0.02, \approx0.98]$. This is our inference and matches Fig. 16, where ID $1-e$ peaks at about 0.95–0.98 |
| History EMA | $\alpha_h$ | 0.1 | Grid {0.01, 0.05, 0.1, 0.2} |
| Test-feature smoothing | $\beta$ | **0.1** (v4 footnote: a code typo fix changed it from 0.3) | Alg. 2, App. C.1 and E.4 still print 0.3 (stale) |
| MEMO (FedTHE+) | $B, \tau$, lr | 16 augs, 3 steps, 5e-4, no decay | Test-time FT |
| Heads | — | Last FC layer | More FC layers were sub-optimal |
| Feature dim | $d$ | 64 (CNN, ResNet20 CIFAR), 256 (ResNet20×4 ImageNet), 128 (CCT4) | — |
| Seeds | — | 3 | — |

### 3.4 Architectural Decisions
- **LP-FT theory (Kumar et al. 2022)**: full fine-tuning with a badly initialized head distorts features and raises OOD error. So the method first probes, by optimizing only a scalar, and then fine-tunes. App. E.1 checks this empirically on MoCo-pretrained ResNet-50.
- **Scalar per-sample weight**: one degree of freedom per sample is cheap and hard to overfit to a single unlabeled input. The feature extractor is never touched in FedTHE, which is the opposite of TTT.
- **Decoupled heads**: the personal head is trained on the *received, frozen* extractor, so local fitting cannot distort the shared features. Only $(w^e, w^g)$ shape the global model.
- **EM alone is harmful**: the overfit personal head is overconfident, and entropy minimization picks it even when wrong. Ablation row (7): ID 71.37 vs 88.68. FA is the counterweight, giving the global head more weight for tail or unseen-class features.
- **SLW instead of a tuned coefficient**: there are no labels at deployment to tune an EM/FA trade-off, so the heads' own agreement $\lambda_s$ sets it.
- **Test history**: single-sample features are too noisy. $h^{hist}$ is a variance-reduction device and is load-bearing (row 11).
- **BalancedSoftmax** on global training gives a better global head. It is orthogonal to the method (+0.3–0.9).

## 4. Implementation Notes
- **Framework used in paper**: Not stated in the text. Code is at github.com/LINs-lab/FedTHE.
- **Model architecture**: CNN with 2 conv layers (32 and 64 channels, 5×5) and 2 FC layers (64 hidden), no BN. ResNet20 with GroupNorm (2 groups; width ×4 for ImageNet). CCT-4/3x2 with LayerNorm and learnable positional embedding. Both heads are a single linear layer.
- **Training details**: see 3.3. Local training (5 epochs) and personalization (1 epoch) both start from the same received global model, and only the local-training parameters are uploaded.
- **Data preprocessing**: Dirichlet non-IID partition ($\text{Dir}(0.1/1.0/10.0)$), with each client split uniformly into local train/val/test. CIFAR10 augmentation is flip + random crop, normalized to [−1, 1]. ImageNet is downsampled to 32×32 and restricted to the 86 classes shared with ImageNet-A/R.
- **Tricks & gotchas**:
  - The value of $\beta$ is inconsistent across the paper; 0.1 is authoritative per the v4 footnote.
  - The FA "test feature" is 90% history, so FedTHE needs a **stream** of test samples per client. Table 10 shows robustness to stream length (0.5× to 1×), but this is never zero history.
  - Each client uploads $h^l_i$, the mean feature of its private data, every round. This is an extra information leak that the paper does not analyze.
  - The authors admit that hyperparameter tuning for test-time methods "is still an open question" (App. C.2).

## 5. Experimental Results

### 5.1 Datasets Used
| Dataset | Clients | Classes | Role in BRFL benchmark | Notes |
|---|---|---|---|---|
| CIFAR10 | 20 (100 in ablation) | 10 | ID train / (a) original local test | 60K images; about 2.5K per client before the split (our estimate) |
| CIFAR10-C | — | 10 | (b) corrupted local test (synthetic covariate shift) | 15 corruptions, severity 5, random corruption per sample |
| CIFAR10.1 | Split by each client's class distribution | 10 | (c) naturally shifted test | Only 2000 samples |
| Other clients' tests | — | 10 | (d) Out-of-Client (OoC): label distribution shift | Same size as the local test |
| Mixture | — | — | (e) random-order mix of (a)–(d) | The "realistic" headline column |
| ImageNet-32 + ImageNet-A / -R / -V2 | 20 | 86 | ID + 3 natural shifts | 300 rounds |

No recommender, collaborative-filtering or implicit-feedback data is used. The only task is classification.

### 5.2 Key Results

**CIFAR10, CNN, $N$=20, $C$=1.0 (Table 1). Accuracy %; "Orig" is the in-distribution (ID) column.**

| Method | Dir(0.1) Orig (ID) | Dir(0.1) OoC | Dir(0.1) Mixture | Dir(1.0) Orig (ID) | Dir(1.0) OoC | Dir(1.0) Mixture |
|---|---|---|---|---|---|---|
| FedAvg (global) | 64.25 | 64.31 | 57.68 | 73.23 | 73.13 | 64.97 |
| FedAvg + FT | 87.82 | 32.59 | 70.55 | 78.21 | 60.01 | 67.38 |
| APFL (adaptive model interpolation) | 87.85 | 27.02 | 69.36 | 68.73 | 57.30 | 55.74 |
| FedRep (personal head) | 88.73 | 26.70 | 69.63 | 79.25 | 54.21 | 66.50 |
| FedRoD (two-head) | 88.78 | 27.35 | 69.84 | 78.55 | 54.94 | 66.25 |
| kNN-Per (per-client tuned $\lambda$) | 89.18 | 32.50 | 70.83 | 79.66 | 64.38 | 68.78 |
| MEMO (P) | 88.19 | 28.25 | 69.73 | 80.20 | 57.80 | 68.41 |
| **FedTHE** | **89.58** | 57.12 | 72.07 | **80.49** | 68.39 | 69.88 |
| **FedTHE+** | **90.55** | **58.29** | **72.90** | **82.05** | **69.49** | **71.59** |

- The big wins are under **label shift (OoC)**: +25 to +30 points over the personal-head methods.
- In-distribution at Dir(0.1), FedTHE beats the strongest other method (kNN-Per) by **+0.40** and the pure personal-head methods by **+0.80** (FedRoD) to **+0.85** (FedRep).
- At Dir(1.0) the margins are **+0.29** over MEMO(P), **+0.83** over kNN-Per and **+1.24/+1.94** over FedRep/FedRoD. Seed std is ±0.3–1.0.
- At Dir(10.0) (Table 4), the global FedAvg (76.28) beats every PFL baseline, and FedTHE adds **+0.89** (77.17).

**Partial participation $C$=0.1, $N$=20, Dir(0.1) (Table 14).** Each client takes part in about 10 of 100 rounds, the same staleness ratio as the thesis.

| Method | ID | OoC | Mixture |
|---|---|---|---|
| FedAvg + FT | 87.13 ±0.97 | 30.92 | 69.32 |
| FedRoD | 88.23 ±0.66 | 26.80 | 69.46 |
| MEMO (P) | 88.34 ±0.54 | 27.52 | 69.57 |
| FedTHE | 88.70 ±0.71 | 53.58 | 70.07 |
| FedTHE+ | 89.48 ±0.58 | 53.69 | 70.40 |

The ID gain of FedTHE over FedRoD shrinks to **+0.47**, which is **within one seed std**. Compared with full participation, FedTHE loses 0.88 ID points and 3.5 OoC points.

**$N$=100 (Table 12)**: ID is FedTHE 88.55 vs FedRoD 87.07 (+1.48).

**ImageNet-32, ResNet20-GN×4 (Tables 2, 5)**: ID is FedTHE 86.56 vs FedRoD 70.52 at Dir(1.0), and 93.00 vs 78.99 at Dir(0.1). These ID gains of 14–16 points are outliers. The paper explains them only with a hypothesis ("fast adaptation nature" of the LP phase). On ImageNet-A every method fails (FedTHE 6.76, below the baselines' 7.1–8.3).

### 5.3 Ablation Highlights
**Table 3, step-by-step build-up (CNN, CIFAR10, Dir(0.1)). ID / OoC / Mixture accuracy, %:**

| Row | Variant | ID | OoC | Mixture |
|---|---|---|---|---|
| (4) | Pure 2-head model, fixed combination (the paper does not state the fixed $e$) | 87.49 | 33.34 | 69.90 |
| (5) | + BalancedSoftmax | 88.68 | 32.92 | 70.72 |
| (6) | + FA only | 87.41 (−1.27) | 63.39 | 67.98 |
| (7) | + EM only | **71.37 (−17.31)** | 49.62 | 59.98 |
| (8) | + 0.5 FA + 0.5 EM | 88.54 (−0.14) | 41.45 | 70.57 |
| (9) | + SLW (= FedTHE) | 89.58 (**+0.90**) | 57.12 | 72.07 |
| (10) | Batch-wise FedTHE | 89.97 | 54.12 | 70.10 |
| (11) | FedTHE without $h^{hist}$ | **87.05 (−1.63 vs row 5)** | 50.45 | 71.89 |
| (13) | + MEMO FT (= FedTHE+) | 90.55 | 58.29 | 72.90 |

- **On ID, only the full recipe helps.** Every partial unsupervised objective (FA only, EM only, uniform mix, no history) is at or below the un-ensembled two-head model on ID. EM alone is catastrophic.
- **Shared-$e$ vs per-client-oracle $e$ (Fig. 13, head-only interpolation, mixed test; values read off the plot, approximate).**
  - Dir(0.1): local head alone ($e$=0) ≈ 56–57; the shared-$e$ curve peaks at ≈71.5 around $e\approx0.25$; global head alone ($e$=1) ≈ 70; the average of per-client optimal $e$ ≈ 72.6; FedTHE 72.07; FedTHE+ 72.90.
  - Dir(1.0): local ≈ 65.5, shared peak ≈ 68.7, oracle ≈ 69.9, FedTHE 69.88.
  - So the per-client oracle beats the best shared $e$ by only about 1–1.2 points. The interior optimum exists *because* the mixed test contains OoC/OOD samples. Whole-model interpolation (Fig. 14, APFL/Mansour style) is worse.
- **Learned $1-e$ (Fig. 16)**: on the in-distribution test, almost all samples put **about 0.93–0.98 weight on the local head**. Only a small mass of tail-class samples leans global. Under OoC, most samples go to the global head.
- **Smaller local training sets (Table 11)**: FedTHE's ID margin over MEMO(P) grows as local data shrinks. At Dir(1.0) it goes +0.29 → +3.59 → +3.87 for 1.0×/0.75×/0.5× data; at Dir(0.1), +1.39 → +1.89 → +3.52. "Small" here still means hundreds or more samples per client.
- **Robust to α/β** (Fig. 15), to corruption severity (Table 8), and to personalization epochs (Table 9).

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas
1. **This is the nearest prior art to the killed capstone.** The capstone blended each user's personal score head with the population-mean head, $w_{\text{eff}} = \alpha w_{\text{local}} + (1-\alpha)\bar w$, at the eval step. FedTHE does the classification counterpart with a *per-sample, unsupervised* weight. The thesis must cite it when telling the α-blend story.
2. **Weight space and logit space are the same thing here.** In the thesis PFedRec, only `affine_output.weight` is local; the bias is global (D-01, see `scripts/thesis/reeval.py` PfedrecAdapter docstring). The score is $\sigma\big((\alpha w_l + (1-\alpha) w_g)^\top q_i + b\big)$. Ranking over the candidates is therefore by $\alpha\, w_l^\top q_i + (1-\alpha)\, w_g^\top q_i$, which is exactly FedTHE's logit ensemble with $e = 1-\alpha$, because the shared $b$ does not change ranks. FedTHE itself writes the final predictor in weight space (Alg. 2, line 11). The "weight-space vs output-space" distinction is **not** a real difference for the capstone. It would matter only for the adaptive module, whose α-blend acts on the *user embedding* before a nonlinear PersonalMLP.
3. **Final-calibration has a precedent.** FedTHE's personalization phase trains $w^l$ for 1 epoch on a frozen received extractor (Alg. 1, lines 8–9; also FedPer's fine-tune variant). Thesis final-calibration does the same once, for all 6040 users after best-round restore, with item embeddings frozen. FedTHE runs at $C$=0.1 (Table 14) without any end-of-training re-personalization and loses 0.88 ID points versus $C$=1.0. The thesis result, that calibration closes 45% of the staleness gap, addresses something FedTHE leaves open.
4. **The adaptive module's server EMA prototype has a precedent.** The global descriptor $h^g$ (a data-size-weighted mean of client feature means, synced each round) and the test-history EMA ($\alpha_h$=0.1) are close cousins of `p_global = 0.9·p_old + 0.1·weighted_avg(client_protos)`. The difference is use: FedTHE uses descriptors to *choose* the head weight, while the adaptive module *blends* $p_{\text{local}}$ toward the prototype.
5. **A figure template for the NO-GO.** Fig. 13 plots a shared-weight curve, a per-client-oracle line and the method. The Stage-A `reeval_*.json` files already contain everything needed for the same plot: NDCG@10 vs global α over {0, 0.05, …, 1}, with horizontal lines for cached α=1 (0.34471), the cross-fitted bucket oracle (−0.0002), the in-sample per-user oracle (+0.0712) and the HC policy (0.30612). Putting it next to FedTHE's Fig. 13 makes the contrast visual. FedTHE's curve has an interior optimum because its test set mixes in OOD; the thesis curve is monotone up to α=1 on in-distribution data.

### 6.2 Potential Integration Points

**(a) Exact differences between FedTHE and the capstone (for the related-work paragraph):**

| Axis | FedTHE | Thesis capstone gate (run cf72e0) |
|---|---|---|
| Blend space | Logits of two linear heads; also written in weight space (Alg. 2, l.11) | Weights of a Linear(32→1) head. **Rank-equivalent** to a logit blend because the bias is global |
| Granularity | Per test sample, many samples per client, online stream | Global / per-bucket / per-user α. Under leave-one-out each user has **exactly one** test query (1 positive + 99 negatives), so "per-sample" **collapses to per-user** |
| How the weight is chosen | Unsupervised: entropy minimization + feature alignment weighted by head agreement, 20 Adam steps from $e$=0.5 | Supervised oracle: grid search that sees the label; cross-fitted for buckets. The per-user oracle is an **upper bound on any label-free per-user rule**, up to the 0.05 grid resolution |
| Global head | *Trained* by FedAvg (+BalancedSoftmax) together with the extractor; a genuine generic expert | Post-hoc **mean of 6040 personal heads**, never trained as a head. Alone it scores 0.2510 vs 0.3447 |
| Setting | Test-time shift: label shift (OoC), corruptions, natural shift, mixture | In-distribution leave-one-out (most recent item held out, 99 uniform negatives), after final-calibration |
| Task | 10- or 86-class softmax classification | 1-output BCE scorer ranking 100 candidates |
| Scale | $N$=20 (100), cross-silo, $C$=1.0 by default | $N$=6040 cross-device, $C$=0.1 |

**(b) Does FedTHE's own evidence predict the NO-GO? Mostly yes.**
- **Predicts it:**
  - On in-distribution data FedTHE gives about 95% weight to the local head (Fig. 16). This mirrors the thesis result α\*=1.0, with buckets at 0.7–1.0 and 74% of users choosing α=1.0 under the per-user oracle.
  - FedTHE's in-distribution gains at $C$=1 are +0.8 to +1.9 points over pure personal-head baselines (FedRep/FedRoD), and only +0.3 to +0.4 over the strongest other method. At $C$=0.1 the gain is +0.47 over FedRoD, which is within one seed std (Table 14).
  - The per-client oracle $e$ beats the best shared $e$ by only about 1 point, and only on a test set that includes OOD (Fig. 13).
  - The mechanism behind FedTHE's in-distribution gain is local label-prior skew: tail-class samples lean on the global head (App. E.5). The rank-only, 1-output setting has no analog of this, because a per-user bias or prior shifts all 100 candidates equally.
  - Row (11) of the ablation is the closest prediction. **Without test history, FedTHE is 1.63 points *worse* on in-distribution data than not ensembling at all** (87.05 vs 88.68). Leave-one-out is exactly the no-history regime: one query per user.
- **Cuts the other way (be honest):**
  - In-distribution gains on ImageNet-32 are large (+14 to +16 points) and unexplained.
  - Table 11 suggests the global head helps *more* as local data shrinks, which would predict a sparse-user gain. The thesis measured cross-fitted sparse Δ = −0.0015. Three caveats on this comparison: FedTHE's comparator there is MEMO(P), not a pure personal head; its "small" clients still have hundreds or more samples, while thesis sparse users have 30 or fewer interactions; and FedTHE's global head is a trained expert, not a mean of heads.
  - So the thesis can say that its negative result is *consistent with* FedTHE's in-distribution evidence. It should not say FedTHE proves it.

**(c) Could a FedTHE-style per-query ensemble be tested offline with `scripts/thesis/reeval.py`? Yes, cheaply, but the expected value is low.**
- **Mechanics.** Add one arm to the pfedrec suite (e.g. `alpha_tta_em`), next to `alpha_oracle_user`. For each user:
  1. Compute the logits over the bit-paired 100 candidates: $z_l = Q w_l$ and $z_g = Q \bar w$, with $\bar w$ = the `global_head` arm.
  2. Optimize $e = \sigma(a_g - a_l)$ from 0.5 with 20 Adam steps at lr 0.1.
  3. Write $\alpha = 1-e^\star$ through the existing `blend_local`, then score with the module's own `evaluate_pfedrec_sampled`.
- **Cost.** Compute is seconds: 6040 users × 20 steps × 100 logits on CPU. The heavy state (cached heads, global item table) lives on the workstation, so it would run there as a reeval.
- **The one non-trivial piece.** The optimization must see the *same* 100 candidates that D-06 scores. That means exposing the candidate pool from `evaluate_pfedrec_sampled`, or replaying its `np_rng(..., "eval_neg")` draw, without breaking the bit-exact trust gate.
- **What the unsupervised objective could be:**
  - **EM**: the entropy of the softmax over the 100 candidate logits, i.e. "which candidate is the positive". A mean per-candidate binary entropy of $\sigma(z+b)$ is the alternative, but it is dominated by the 99 negatives and by $b$.
  - **FA**: does **not** transfer. The PFedRec model has no query or user feature (`forward(item_indices)` only). A local descriptor could be the mean item embedding of the user's training positives, but the only per-query feature that tells users apart is the held-out positive's embedding, which is label leakage. The candidate-set mean is about the uniform item mean for every user, because the 99 negatives are uniform draws. $h^{hist}$ is degenerate, since there is one query per user.
  - **SLW therefore reduces to EM.** EM-only is exactly the variant FedTHE's own ablation found worst on in-distribution data (row 7, −17 points).
- **Predicted outcome.** EM picks the head whose logits are more spread out. A mean of 6040 heads should have a smaller norm than a typical personal head, because the directions cancel; this is unverified. If so, EM drives $e \to \approx0.02$, i.e. α ≈ 0.98, which is essentially the cached α=1 result (Δ ≈ 0). Check this before writing any code: compare $\lVert\bar w\rVert$ with the median $\lVert w_l\rVert$ in a one-line check.
- **A faithful port** would need a *trained* global head: a FedAvg'd second `affine_output` trained jointly with item embeddings, in the FedRoD/FedTHE two-head style. That is a GPU run (Stage B/C-class) and is not justified after the NO-GO.
- **Recommendation.** Optional and low priority, after the mandatory paired significance tests and the threat-model work. It is worth doing only as a half-day "we tested the nearest prior-art mechanism" appendix arm. If done, pre-register it against cached 0.34471, report it beside the per-user oracle (which bounds it), and expect roughly zero.

**(d) Suggested citation framing (1–2 sentences):**
> *FedTHE (Jiang & Lin, 2023) ensembles a federated global head with a local personalized head using a per-test-sample weight optimized by unsupervised entropy and feature-alignment objectives. Its gains come mainly under test-time distribution shift, especially label shift, and are modest in-distribution, where the learned weight concentrates on the local head. Our offline gate tests the in-distribution counterpart for implicit-feedback ranking, where blending the linear score head in weight space is rank-equivalent to FedTHE's logit ensemble. Even a cross-fitted, label-informed bucket oracle gains nothing (ΔNDCG@10 = −0.0002), consistent with FedTHE's observation that in-distribution samples overwhelmingly select the personal head.*

### 6.3 Limitations & Gaps
- **No recommender, implicit-feedback or ranking evaluation.** It is classification only, and the EM/FA objectives assume a multi-class softmax and a per-sample input feature. A BCE scorer over item IDs has neither a natural per-query feature nor augmentations, so MEMO, and with it FedTHE+, has no analog for ID inputs.
- **Cross-silo scale.** $N$=20, or 100 in one ablation; the default $C$=1.0. The thesis protocol is $N$=6040, $C$=0.1.
- **Assumes a test stream per client.** The FA term is about 90% history. Leave-one-out gives one query per user.
- **No privacy analysis.** Uploading $h^l_i$, the mean feature of the client's private data, is an unanalyzed leak. In the RecSys analog it would be the user's preference centroid, essentially a user embedding. The thesis threat-model section should list it, together with the capstone's own proposed upload of `affine_output.weight` (32 floats = the user's preference vector).
- **Partial participation is handled naively.** Heads go stale, and there is no re-personalization before deployment.
- **Test-time hyperparameters** (Adam lr, 20 steps, α, β) are fixed globally, and the authors call fair tuning for test-time methods an open question.

## 7. Key References to Follow
- **Chen & Chao, ICLR 2022 (FedRoD)**: the two-head (generic + personalized) design with BalancedSoftmax that FedTHE builds on. It is the template for a "trained global head" port.
- **Collins et al., ICML 2021 (FedRep)**: shared representation + local head. This is exactly the PFedRec structure (shared item table + local Linear head). Digest: [collins_2021_fedrep](collins_2021_fedrep.md).
- **Deng et al., 2020 (APFL)**: adaptive per-client α interpolation of local and global *models*. It hurts at mild heterogeneity in FedTHE's tables (ID 68.73 vs 78.21 for FedAvg+FT at Dir(1.0)). It is the direct ancestor of the thesis's α-blend. Queued: `Papers/raw/APFL-Yuyang-2020.pdf`.
- **Marfoq et al., ICML 2022 (kNN-Per)**: global model + local kNN, interpolated with a per-client tuned λ. It is within 0.4–0.8 points of FedTHE on ID, which shows that tuned per-client interpolation ≈ per-sample TTA in-distribution.
- **Kumar et al., ICLR 2022 (LP-FT)** and **Zhang et al., 2021 (MEMO)**: the theory and the test-time FT component behind FedTHE+.
