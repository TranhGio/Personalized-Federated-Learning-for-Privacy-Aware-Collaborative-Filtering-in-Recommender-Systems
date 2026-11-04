# Adaptive Personalized Federated Learning

- **Authors**: Yuyang Deng, Mohammad Mahdi Kamani, Mehrdad Mahdavi (The Pennsylvania State University)
- **Venue**: arXiv preprint arXiv:2003.13461v3, 2020 (no venue stated in the PDF)
- **Paper ID**: deng_2020_apfl
- **Tags**: #method, #federated-learning, #personalization, #non-iid, #optimization, #model-interpolation

---

## 1. Core Idea
APFL gives each client a personalized model that is a **convex combination of a local model and the global model**, $\theta_i = \alpha_i v_i + (1-\alpha_i) w$. The local model $v_i$ is trained *through* the mixture, so it learns a residual on top of $w$. The global $w$ is trained exactly as in FedAvg. The mixing weight $\alpha_i$ is learned per client by gradient descent on the local loss. The paper adds a generalization bound for the mixed model. That bound is quadratic in $\alpha_i$ and gives a closed-form $\alpha_i^*$ that grows with the local–global distribution shift $\|\bar{\mathcal{D}}-\mathcal{D}_i\|_1$ and shrinks as the local sample size $m_i$ gets smaller. It also gives local-SGD-style convergence rates for the strongly convex and nonconvex cases.

## 2. Problem & Motivation
Under non-IID data, the global model learned by FedAvg/SCAFFOLD generalizes poorly on each client's own distribution. In the paper's Fig. 1 (MNIST, logistic regression), the global-model loss grows as the number of classes per client falls from 10 to 2. Local fine-tuning (Per-FedAvg/MAML-style, FedPer) optimizes training error and can overfit. Regularization methods (Hanzely & Richtárik 2020, pFedMe) and the model interpolation of Mansour et al. (2020) do not show *how* the mixing weight affects generalization; Mansour's bound does not depend on $\alpha$ at all. APFL asks "what degree of personalization is best for each client?", which Mohri et al. (2019) left open.

## 3. Method

**Notation map (APFL → thesis).** APFL $v_i$ (local model) → $v_i$. APFL $\bar v_i$ (mixed model) → $\theta_i$. APFL $w$ → $w$. APFL $n$ (clients) → $N$. APFL $K$ (clients sampled per round) → $CN$, so APFL's $K/n = C$. APFL $\tau$ (local steps between syncs) → $K$. APFL $T$ = total local iterations ≈ $R\cdot K$. Dataset $\mathcal{D}_i$ is unchanged. $\alpha_i$ has the **same orientation as the thesis**: $\alpha_i=1$ is fully local and $\alpha_i=0$ is fully global. All formulas below use thesis symbols.

### 3.1 Objective Function

Hypothesis level (Sec. 3.1). The global model minimizes empirical risk on the average distribution $\bar{\mathcal{D}} = \frac1N\sum_i \mathcal{D}_i$. The local model is fit **with the global model mixed in**:

$$
\bar h^* = \arg\min_{h\in\mathcal H}\hat{\mathcal L}_{\bar{\mathcal D}}(h),\qquad
\hat h^*_{loc,i} = \arg\min_{h\in\mathcal H}\hat{\mathcal L}_{\mathcal D_i}\big(\alpha_i h + (1-\alpha_i)\bar h^*\big),\qquad
h_{\alpha_i} = \alpha_i \hat h^*_{loc,i} + (1-\alpha_i)\bar h^* \quad (1)
$$

Parameter level (Sec. 4, Eqs. 4, 5, 9). This is a bilevel problem: $w$ does not depend on $v_i$ or $\alpha_i$.

$$
\min_{w\in\mathbb R^d} F(w) := \frac1N\sum_{i=1}^N f_i(w),\quad f_i(w)=\mathbb E_{\xi_i}[f_i(w;\xi_i)]
\qquad\text{(server, = FedAvg objective)}
$$

$$
\min_{v\in\mathbb R^d} f_i\big(\alpha_i v + (1-\alpha_i) w^*\big)\qquad\text{(client } i\text{)},\qquad
\alpha_i^* = \arg\min_{\alpha_i\in[0,1]} f_i\big(\alpha_i v + (1-\alpha_i) w\big)\quad (9)
$$

Where:
- $f_i$: empirical risk of client $i$ on $\mathcal D_i$; $\xi_i$ is a minibatch from $\mathcal D_i$.
- $w^* = \arg\min_w F(w)$: optimal global model.
- $v_i$: client $i$'s local model. It has the **same architecture and dimension as $w$** (whole-model mixing).
- $\theta_i = \bar v_i = \alpha_i v_i + (1-\alpha_i) w_i$: the personalized model used for inference.
- $\alpha_i\in[0,1]$: per-client mixing weight. The paper notes that $\theta_i$ is generally *not* the empirical risk minimizer on $\mathcal D_i$; it carries a residual risk.

### 3.2 Algorithm (Algorithm 1, "Local Descent APFL")

**Client-side.** Each selected client $i\in U_t$ holds three models: $w_i$ (its local copy of the global model), $v_i$ (persistent local model) and $\theta_i$ (the mixture). For each local iteration $t$ with $t \bmod K \neq 0$:
1. Update the global copy on local data, ignoring $v_i$: $\;w_i^{(t)} = w_i^{(t-1)} - \eta_t \nabla f_i(w_i^{(t-1)};\xi_i^t)$ (Eq. 6).
2. Update the local model through the mixture: $\;v_i^{(t)} = v_i^{(t-1)} - \eta_t \nabla_v f_i(\theta_i^{(t-1)};\xi_i^t)$ (Eq. 7). By the chain rule $\nabla_v f_i(\theta_i) = \alpha_i\nabla f_i(\theta_i)$, so the mixture moves by $\alpha_i^2\eta_t\nabla f_i$ through the $v$ path. The proofs (App. B) confirm this $\alpha_i^2$ factor.
3. Re-mix: $\;\theta_i^{(t)} = \alpha_i v_i^{(t)} + (1-\alpha_i) w_i^{(t)}$.
4. If $\alpha$ is adaptive, take one gradient step on $\alpha_i$ (Eq. 10). The closed-form gradient is an inner product:
$$\alpha_i^{(t)} = \alpha_i^{(t-1)} - \eta_t\nabla_\alpha f_i(\theta_i^{(t-1)};\xi_i^t) = \alpha_i^{(t-1)} - \eta_t\Big\langle v_i^{(t-1)} - w_i^{(t-1)},\; \nabla f_i(\theta_i^{(t-1)};\xi_i^t)\Big\rangle$$
   $\alpha_i$ grows when the descent direction $-\nabla f_i$ points from $w$ toward $v$, and stays put when $v_i\approx w_i$ (IID data). The text says "at every communication round", but Eq. 10 is indexed per iteration with $\eta_t$ and $\xi_i^t$, so the update frequency is ambiguous. Eq. 9 constrains $\alpha_i\in[0,1]$, but **Eq. 10 has no explicit projection or clipping**. Public re-implementations (FedTorch, PFLlib `clientapfl.py`) reportedly clip to $[0,1]$ and add a small L2 term ($0.02\,\alpha$) to the $\alpha$ gradient. This has not been checked here; verify in the code before citing.
5. When $t \bmod K = 0$, send only $w_i^{(t)}$ to the server. $v_i$ and $\alpha_i$ never leave the device.
6. Clients not in $U_t$ keep $v_i^{(t)} = v_i^{(t-1)}$. Clients are stateful.

**Server-side.**
1. Average: $w^{(t)} = \frac{1}{|U_t|}\sum_{j\in U_t} w_j^{(t)}$ (Eq. 8). This is plain FedAvg, unweighted in the paper.
2. Sample a new set $U_t$ of $CN$ clients uniformly and broadcast $w^{(t)}$. Each $w_i$ is overwritten by this broadcast; $v_i$ persists.
3. Output used in the theory only: time-averaged models with $p_t=(t+a)^2$ and $S_T=\sum_t p_t$:
$$\hat\theta_i=\frac{1}{S_T}\sum_{t=1}^T p_t\big(\alpha_i v_i^{(t)} + (1-\alpha_i)\tfrac{1}{CN}\textstyle\sum_{j\in U_t} w_j^{(t)}\big),\qquad \hat w=\frac{1}{CN\,S_T}\sum_t p_t\sum_{j\in U_t} w_j^{(t)}$$

**New-client personalization (Eq. 17).** A device $k$ joining after training runs a few steps of $v_k \leftarrow v_k - \eta_t\nabla_v f_k(\alpha_k v_k + (1-\alpha_k)\hat w;\xi_k)$ with $\hat w$ frozen. In Fig. 6 this beats fine-tuning FedAvg's model.

**Cost.** Communication per round is the same as FedAvg (only $w$). Each client stores and computes on **two full models** (2× forward/backward per step) plus a scalar $\alpha_i$.

### 3.3 Key Hyperparameters
| Hyperparameter | Symbol (thesis) | Default Value | Role |
|---|---|---|---|
| Mixing weight | $\alpha_i$ | Fixed ∈ {0.25, 0.5, 0.75}, or adaptive with $\alpha_i^{(0)}=0.01$ (synthetic) / $0.5$ (EMNIST) | 1 = local, 0 = global; $\alpha=0$ is FedAvg's localized model |
| $\alpha$ domain | — | $[0,1]$ (Eq. 9); no clip stated in Eq. 10 | Theory (Thm 3) needs $\alpha_i \ge 1-\frac{1}{4\sqrt{6\kappa}}$ for the tight rate |
| Local steps between syncs | $K$ (APFL $\tau$) | 10 (default); 20 (Table 2) | Synchronization gap |
| Learning rate | $\eta_t$ | 0.1 initial, decayed 1% per iteration (Bottou 2012); SCAFFOLD 0.05 with 5% decay | The same $\eta_t$ drives $w$, $v$ and $\alpha$ updates |
| LR (theory) | $\eta_t$ | $\frac{16}{\mu(t+a)}$, $a=\max\{128\kappa,K\}$ (strongly convex); $\frac{1}{2\sqrt5L\sqrt T}$ (nonconvex) | — |
| Clients | $N$ | 100 (1000 for EMNIST) | — |
| Sampling fraction | $C$ | 1.0 (MNIST LR); {0.3, 0.5, 0.7} (sampling study); 0.1 (EMNIST) | — |
| Rounds | $R$ | 100 (250 for EMNIST) | — |
| Batch size | $B$ | 20 (Table 2); otherwise not reported | — |
| Local epochs | — | 1 epoch per round (EMNIST, Fig. 6) | — |
| L2 on logistic regression | $\lambda$ | Not reported in paper | Makes the loss strongly convex |
| Baseline settings | — | Per-FedAvg: $\alpha=0.01$, $\beta=0.001$; pFedMe: $\lambda=15$, $\eta=0.01$, $K=5$ | — |

### 3.4 Architectural Decisions
1. **Mix models, not just regularize.** $\theta_i$ is an explicit convex combination, so the bound can show how $\alpha_i$ changes generalization. The authors say this is what Mansour et al. 2020 lacks.
2. **The global model is trained independently of the mixture.** The $w$ update uses $\nabla f_i(w_i)$, not $\nabla f_i(\theta_i)$, so $w$ is exactly FedAvg's model and the method stays FedAvg-compatible (same communication, same server). All personalization lives in $v_i$ and $\alpha_i$.
3. **The local model is trained under the mixture.** $v_i$ is fit to minimize $f_i(\alpha_i v + (1-\alpha_i)w)$, so it learns to *correct* $w$ rather than replace it. For model classes closed under affine combinations, any fixed $\alpha_i>0$ loses no expressiveness. $\alpha_i$ then mainly sets (a) the $\alpha_i^2$ scaling of the effective step on $\theta_i$ through $v_i$, and (b) how far $\theta_i$ starts from $w$.
4. **Adaptive $\alpha$.** The bound-optimal $\alpha^*$ depends on $\|\bar{\mathcal D}-\mathcal D_i\|_1$, which no client can observe. The authors therefore learn $\alpha_i$ from the local loss instead of setting it from statistics.
5. **Tight convergence theory lives near $\alpha=1$.** Theorem 3 needs $\alpha_i \ge \max\{1-\frac{1}{4\sqrt{6\kappa}},1-\frac{1}{4\sqrt{6\kappa}\sqrt\mu}\}$; for $\kappa=10$ that is about 0.97. Theorem 4 drops this assumption but needs bounded gradients $G$ and projection onto a bounded domain.

**Theory, stated precisely.**

*Theorem 1 (generalization).* Assumptions: $\mathcal H$ has VC dimension $d$; the loss is $G$-Lipschitz and bounded in $[0,B]$ (proved for squared hinge and MSE). Then with probability $\ge 1-\delta$:

$$
\mathcal L_{\mathcal D_i}(h_{\alpha_i}) \le 2(1-\alpha_i)^2\underbrace{\Big(\hat{\mathcal L}_{\bar{\mathcal D}}(\bar h^*) + B\|\bar{\mathcal D}-\mathcal D_i\|_1 + C\sqrt{\tfrac{d+\log(1/\delta)}{m}}\Big)}_{\text{global term }G_i}
+ 2\alpha_i^2\underbrace{\Big(\mathcal L_{\mathcal D_i}(h_i^*) + 2C\sqrt{\tfrac{d+\log(1/\delta)}{m_i}} + G\lambda_{\mathcal H}(\mathcal S_i)\Big)}_{\text{local term }L_i}
$$

The symbols are:
- $m_i$: client $i$'s sample count; $m=\sum_i m_i$.
- $\|\bar{\mathcal D}-\mathcal D_i\|_1=\int_\Xi|P_{\bar{\mathcal D}}-P_{\mathcal D_i}|$: total-variation-type divergence on the **joint** $(x,y)$.
- $h_i^*=\arg\min_h\mathcal L_{\mathcal D_i}(h)$.
- $\lambda_{\mathcal H}(\mathcal S)=\sup_{h,h'\in\mathcal H}\frac{1}{|\mathcal S|}\sum_{(x,y)\in\mathcal S}|h(x)-h'(x)|$: worst-case disagreement on the local training set (Def. 1).
- $C$: a constant.

The proof chains three steps: Jensen on the convex loss, $(a+b)^2\le 2a^2+2b^2$, and then Lemma 1, $\mathcal L_{\mathcal D}(h)\le\mathcal L_{\mathcal D'}(h)+B\|\mathcal D-\mathcal D'\|_1$.

*Optimal mixing (Eq. 3).* The right-hand side is quadratic in $\alpha_i$, so
$$\alpha_i^*=\frac{G_i}{G_i+L_i}=\frac{\hat{\mathcal L}_{\bar{\mathcal D}}(\bar h^*)+B\|\bar{\mathcal D}-\mathcal D_i\|_1+C\sqrt{\frac{d+\log(1/\delta)}{m}}}{\hat{\mathcal L}_{\bar{\mathcal D}}(\bar h^*)+B\|\bar{\mathcal D}-\mathcal D_i\|_1+C\sqrt{\frac{d+\log(1/\delta)}{m}}+\mathcal L_{\mathcal D_i}(h_i^*)+2C\sqrt{\frac{d+\log(1/\delta)}{m_i}}+G\lambda_{\mathcal H}(\mathcal S_i)}.$$
The paper reads it this way:
- A large divergence pushes $\alpha^*\to1$ (more local).
- A small $m_i$ pushes $\alpha^*$ down (more global).
- A large $m_i$ pushes $\alpha^*\to1$.

**Caveat.** Because $G_i, L_i > 0$, this $\alpha^*$ is **always strictly interior**. The bound can never recommend $\alpha=1$ or $\alpha=0$.

*Corollary 3 (vs local ERM $\hat h_i^*$).* $\mathcal L_{\mathcal D_i}(h_{\alpha_i})-\mathcal L_{\mathcal D_i}(\hat h_i^*)\le(2\alpha_i^2-1)\mathcal L_{\mathcal D_i}(h_i^*)+(2\alpha_i^2C_1-C_2)\sqrt{\frac{d+\log(1/\delta)}{m_i}}+2\alpha_i^2G\lambda_{\mathcal H}(\mathcal S_i)+2(1-\alpha_i)^2\big(\hat{\mathcal L}_{\bar{\mathcal D}}(\bar h^*)+B\|\bar{\mathcal D}-\mathcal D_i\|_1+C_1\sqrt{\frac{d+\log(1/\delta)}{m}}\big)$. Mixing beats the local ERM only when three conditions hold together:
- $\alpha_i\le\min\{\frac{\sqrt2}{2},\sqrt{C_2/(2C_1)}\}$;
- $m_i$ is small;
- **both** the divergence $\|\bar{\mathcal D}-\mathcal D_i\|_1$ and the global model's error are small.

*Convergence.* The theorems use these heterogeneity measures:
- $\zeta_i=\sup_w\|\nabla F(w)-\nabla f_i(w)\|^2$, with $\zeta=\sum_i\zeta_i$;
- $\Delta_i=\|v_i^*-w^*\|^2$;
- $\Gamma=\sup_{x_1,x_2}\|\nabla F(x_1)-\nabla F(x_2)\|^2$;
- $\kappa=L/\mu$, gradient variance $\le \sigma^2$, and $b=\min\{C,\tfrac12\}$.

The results, in thesis symbols:
- **Thm 2** (global $w$, $\mu$-strongly convex, sampled): $\mathbb E[F(\hat w)]-F^*\le O\big(\tfrac{\mu\|w^{(1)}-w^*\|^2}{T^3}\big)+O\big(\tfrac{\kappa^2K(\sigma^2+2K\zeta/CN)}{\mu T^2}\big)+O\big(\tfrac{\kappa^2K(\sigma^2+2K\zeta/CN)\ln T}{\mu T^3}\big)+O\big(\tfrac{\sigma^2}{CN\,T}\big)$. With $K=\sqrt{T/(CN)}$ this is $O(1/(CN\,T))$, the local-SGD rate.
- **Thm 3 / Cor. 1** (personalized $\hat\theta_i$, strongly convex, $K=\sqrt{T/(CN)}$): $\mathbb E[f_i(\hat\theta_i)]-f_i^*\le\alpha_i^2O\big(\tfrac{\sigma^2}{\mu T}\big)+(1-\alpha_i)^2\Big(O\big(\tfrac{\kappa^2\sigma^2}{\mu CNT}\big)+O\big(\tfrac{\kappa^2(\zeta_i+\zeta/CN)}{\mu CNT}\big)\Big)+(1-\alpha_i)^2O\big(\tfrac{\zeta_i+\zeta/CN}{\mu}+\kappa L\Delta_i\big)$. The last term is a **non-vanishing residual** that disappears only at $\alpha_i=1$ or with no heterogeneity.
- **Thm 4** (no $\alpha$ assumption): the residual becomes $(1-\alpha_i)^2O(G^2/(\mu b))$, and the iterates need projection onto a bounded domain.
- **Thm 5** (global, nonconvex, $C=1$): $\frac1T\sum_t\mathbb E\|\nabla F(w^{(t)})\|^2\le O\big(\tfrac{L}{\sqrt T}\big)+O\big(\tfrac{\sigma^2+\sigma^2/N+\zeta/N}{T}\big)+O\big(\tfrac{\sigma^2}{\sqrt T}\big)$.
- **Thm 6 / Cor. 2** (personalized, nonconvex, $C=1$, $K=N^{-1/4}T^{1/4}$): $O(\tfrac{1}{\sqrt T})+(1-\alpha_i)^2O(\tfrac{1}{\sqrt T}+\tfrac{1}{\sqrt{NT}})+(1-\alpha_i^2)^2(\zeta_i+\Gamma)$ with $N^{3/4}T^{3/4}$ communication. The residual again vanishes only at $\alpha_i=1$.

Neither rate analyzes the *adaptive*-$\alpha$ update. Every rate assumes a fixed $\alpha$.

## 4. Implementation Notes
- **Framework used in paper**: PyTorch `distributed` with MPI on Azure (Standard F64s VMs, 64 vCPUs), in a custom Docker image. The paper gives no code link; the authors' FedTorch library and PFLlib contain APFL implementations (not verified here).
- **Model architecture**: L2-regularized logistic regression (MNIST, synthetic; strongly convex). CNN with 2 conv + 2 FC layers (CIFAR-10). MLP with 2×200 ReLU hidden layers (EMNIST; MNIST comparison). Cross-entropy loss.
- **Training details**: SGD, LR 0.1 decayed 1% per iteration. $K=10$ local steps (20 in the comparison run). 100 rounds (250 for EMNIST). Batch size 20 in the comparison run.
- **Data preprocessing**: Pathological label partitions following McMahan 2017: MNIST with 2 classes/client, 4 classes/client, or IID; CIFAR-10 with 2 classes/client. EMNIST digits are split by writer (1000 writers). Synthetic($\gamma,\beta$) follows FedProx: $W_i\sim\mathcal N(\mu_i,1)$, $\mu_i\sim\mathcal N(0,\gamma)$; $x_i\sim\mathcal N(\nu_i,\Sigma)$, $\nu_i\sim\mathcal N(V_i,1)$, $V_i\sim\mathcal N(0,\beta)$; $\Sigma_{kk}=k^{-1.2}$. Each client has a local validation split from its own distribution. Reported numbers average the personalized models over online clients each round.
- **Tricks & gotchas**: (i) $\alpha=0$ reduces to FedAvg's *localized* model, which is how Fig. 2 labels it. (ii) $v_i$ must persist across rounds on the device, so clients are stateful. (iii) In the CIFAR-10 run, adaptive $\alpha$ wins on *training* loss but not on validation, because the learned $\alpha$ minimizes training loss. (iv) The paper **never reports the learned $\alpha_i$ values**: no histogram, no trajectory, and no plot against $m_i$ or heterogeneity. (v) The nonconvex theory assumes full participation ($C=1$).

## 5. Experimental Results

### 5.1 Datasets Used
| Dataset | Clients | Heterogeneity | Model | Loss |
|---|---|---|---|---|
| MNIST | 100 | 2 or 4 classes/client, or IID | Logistic regression; MLP 2×200 | Logistic + L2; CE |
| CIFAR-10 | 100 | 2 classes/client | CNN (2 conv + 2 FC) | CE |
| EMNIST (digits) | 1000 writers, $C=0.1$ | Natural (writer style) | MLP 2×200 | CE |
| Synthetic($\gamma,\beta$) | 100 | (0,0) IID, (0.5,0.5), (1,1) | Logistic regression | Logistic |

There is no recommendation, implicit-feedback, or ranking data.

### 5.2 Key Results

**Table 1. CIFAR-10, 2 classes/client, CNN, 100 rounds (mean ± std).**
| Method | Training Loss | Validation Acc |
|---|---|---|
| APFL $\alpha=0.25$ | 0.154 ± 0.003 | **89.33% ± 0.26** |
| APFL $\alpha=0.5$ | 0.113 ± 0.008 | 88.74% ± 0.14 |
| APFL $\alpha=0.75$ | 0.103 ± 0.007 | 89.04% ± 0.22 |
| APFL adaptive $\alpha$ | **0.101 ± 0.013** | 88.87% ± 0.51 |
| FedAvg global | 1.789 ± 0.004 | 32.51% ± 0.47 |
| FedAvg localized | 0.369 ± 0.005 | 83.16% ± 0.37 |
| SCAFFOLD global | 1.70 ± 0.001 | 37.16% ± 0.3 |
| SCAFFOLD localized | 0.593 ± 0.012 | 85.25% ± 0.2 |

**Table 2. MNIST, 2 classes/client, MLP, 100 rounds, $K=20$, $B=20$.**
| Method | Training Loss | Validation Acc |
|---|---|---|
| APFL $\alpha=0.25$ | 0.011 ± 0.0007 | 98.07% ± 0.10 |
| APFL $\alpha=0.5$ | 0.004 ± 0.0004 | 98.04% ± 0.08 |
| APFL $\alpha=0.75$ | **0.002 ± 0.0001** | 97.86% ± 0.09 |
| APFL adaptive $\alpha$ | 0.004 ± 0.0008 | **98.10% ± 0.10** |
| FedAvg global | 0.240 ± 0.006 | 93.81% ± 0.29 |
| FedAvg localized | 0.041 ± 0.002 | 97.75% ± 0.15 |
| Per-FedAvg | 0.039 ± 0.002 | 97.83% ± 0.12 |
| pFedMe | 0.182 ± 0.004 | 95.92% ± 0.1 |

**Figures (qualitative).**
- **Fig. 2 (MNIST, logistic regression).** Larger $\alpha$ gives lower training loss. On non-IID data (2 or 4 classes) the best validation comes from $\alpha=0.25$, ahead of FedAvg/SCAFFOLD global and localized. On IID data FedAvg/SCAFFOLD win, so personalization helps only under heterogeneity.
- **Fig. 3 (sampling).** APFL beats localized FedAvg at $C\in\{0.3,0.5,0.7\}$, and a larger $\alpha$ softens the harm from sampling. The text says "larger α values are preferred" here, which contradicts the Fig. 2 statement that $\alpha=0.25$ is best.
- **Fig. 4 (adaptive $\alpha$, synthetic).** With adaptive $\alpha$ (init 0.01), all three heterogeneity levels reach almost the same validation accuracy, while localized FedAvg falls well behind.
- **Fig. 5 (EMNIST).** Adaptive APFL (init 0.5) matches localized FedAvg's training loss and beats its validation accuracy.

### 5.3 Ablation Highlights
- **The existence of a jointly trained local component matters far more than the value of $\alpha$.** Among APFL variants, validation accuracy spans only 0.59 pt on CIFAR-10 (88.74–89.33) and 0.24 pt on MNIST (97.86–98.10), roughly 1–2 std. The gap to global FedAvg is 4–57 pts.
- **Adaptive $\alpha$ is about as good as the best fixed $\alpha$, not clearly better.** It is best on MNIST validation (+0.03 pt over $\alpha=0.25$) and best on CIFAR-10 training loss, but worse than $\alpha=0.25$ on CIFAR-10 validation. Learning $\alpha$ by training loss tends toward more local mixing.
- **Under IID data, personalization does not help.** The paper's answer to "when is personalization better?" is: only under heterogeneity.

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas
1. **APFL is the direct conceptual ancestor of the thesis α-machinery, and should be cited as such.** All three thesis α-mechanisms instantiate Eq. (1) on **one component** instead of the whole model:
   - (a) The adaptive module's Level-1 blend $p_{eff}=\alpha p_{local}+(1-\alpha)p_{global}$, in `bpr_mf.py:get_effective_embedding`.
   - (b) The optional per-user learnable $\alpha=\sigma(\ell_u)$.
   - (c) The killed capstone, which blended the personal PFedRec `affine_output` head with the population-mean head in weight space.

   The head is Linear(32→1) before the sigmoid, and $p\cdot q$ is linear in $p$. So in (a) without the MLP, and in (c), weight-space mixing equals APFL's output-space mixing $\alpha z_{own}+(1-\alpha)z_{global}$ on the logit.
2. **The per-user learnable α is APFL's Eq. (10), reparametrized.** The prototype is a non-trainable tensor, so autograd gives $\partial L/\partial\ell_u=\sigma(\ell_u)(1-\sigma(\ell_u))\,\langle p_{local,u}-p_{global},\nabla_{p_{eff}}L_{BPR}\rangle$. That is APFL's inner-product gradient times $\sigma'$. The sigmoid replaces APFL's $[0,1]$ constraint. Note that this path is *unclipped*; the logit init is clamped to [0.01, 0.99], while the heuristic path clips to [0.1, 0.95].
3. **Use APFL's α-gradient as an analytic boundary test in the capstone write-up.** The held-out risk of a convex combination is convex in α. So $\alpha=1$ is optimal for user $u$ exactly when the one-sided derivative at α=1 is non-positive, i.e. $\langle z_{own}-z_{global},\nabla_z\ell\rangle\le0$ evaluated at $\alpha=1$. On the bit-exact replay infrastructure, one forward/backward per user gives the *fraction of users for whom any global mixing is a descent direction*. This is a grid-free companion to "α* = 1.0" and "74% of per-user oracle choices are α=1.0".
4. **External corroboration that α is a second-order knob.** Tables 1–2 show a ≤0.6-pt spread across α once a jointly trained local model exists. That matches the thesis bucket-oracle Δ = −0.0002 (GO threshold +0.010). APFL's own adaptive α never clearly beats a fixed α.

### 6.2 Potential Integration Points
**Precise mapping: what is the same, what differs.**

| Aspect | APFL | Thesis adaptive module | Thesis killed capstone |
|---|---|---|---|
| Mixed object | Whole model ($v_i$ vs $w$, same dim) | User embedding only ($p_{local}$ vs $p_{global}$, dim 128); item side already global (split learning) | 33-param score head only, on top of global item embeddings |
| "Global" side | $w$ = FedAvg model, **retrained on $\mathcal D_i$ each local step** ($w_i$) | Fixed server prototype: EMA, $0.9\,p_{old}+0.1\cdot$ the `num_examples`-weighted mean of client $p_{local}$ (dense users dominate); not trained on $\mathcal D_i$ | Population-mean head (mean of personal heads) |
| Local side trained under the mixture? | **Yes** (Eq. 7), so $v_i$ learns a residual | Yes: gradient $\alpha\nabla_{p_{eff}}$ flows to $p_{local}$ | **No**: post-hoc interpolation of two fixed models (head trained at α=1, then mixed). This is *not* APFL, which fits $v_i$ under the mixture. |
| How α is set | Learned by SGD on local training loss, init 0.01 or 0.5, domain [0,1] | Hierarchical-conditional heuristic from user statistics, clipped [0.1, 0.95]; optional learnable logit initialized from it | Grid / bucket oracle / heuristic policy, offline |
| Can be fully local? | Yes (α=1 allowed; Thm 3 even assumes α≈1) | **No**: α ≤ 0.95 ceiling, though the oracle α* = 1.0 | Yes (grid includes 1.0) |

**Integration (write-up only; no new training needed).**

1. **Explain the −3.9-pt capstone result and α* = 1.0 with Corollary 3, not Theorem 1.** Corollary 3 says mixing beats the local ERM only if the *global model's error and the divergence $\|\bar{\mathcal D}-\mathcal D_i\|_1$ are both small*. In recommendation that condition fails for every user group:
   - $\mathcal D_i$ is a joint over (item, label), and $P(y\mid\text{item})$ is user-specific. This is concept shift, not covariate shift, so the "average user" reduces to a popularity scorer.
   - The thesis counterfactuals measure it directly: the population-mean head scores 0.2510 against 0.3447 for the user's own head (−9.4 pts). It is far better than a random head (0.0741), but that population signal is already carried by the shared item embeddings.
   - On the $m_i$ side, the local model is tiny: a 33-param head, or one 128-d embedding, sitting on a global item backbone. Even a "sparse" ML-1M user (20–30 ratings, ×(1+4 negatives) per epoch) has ~100+ training examples for 33 parameters. So the estimation-error term that would favor small α is small. APFL's mechanism predicts α* near 1 here, which is what the gate found (sparse bucket Δ −0.0015).
2. **Do not claim that Theorem 1 "predicts" α* = 1 or any specific α*.** Its minimizer $G_i/(G_i+L_i)$ is always interior, and both terms are vacuous at ML-1M scale: $B\|\cdot\|_1$ can reach $2B$, and $\lambda_{\mathcal H}$ is a sup over the class that does not shrink with $m_i$. Theorems 3 and 6 point the other way: their heterogeneity residuals, $(1-\alpha_i)^2(\zeta_i/\mu+\kappa L\Delta_i)$ and $(1-\alpha_i^2)^2(\zeta_i+\Gamma)$, vanish **only at α = 1**. On the optimization side, APFL's theory favors fully local whenever heterogeneity is high.
3. **A candidate mechanistic hypothesis for "adaptive worst on sparse users" (sparse 0.2355 vs split-learning 0.2726).** As in APFL's Eq. 7, $p_{local}$ receives $\alpha\nabla$. Because the thesis prototype is frozen, $p_{eff}$ moves only through $\alpha\,\Delta p_{local}$. That gives an $\alpha^2$-damped step under SGD, and ~α-damped under Adam, which the module uses (lr 0.005). Unlike APFL, nothing on the global side keeps learning from $\mathcal D_i$. A sparse-penalty α of 0.1–0.3, applied to users with only ~10 participations at C=0.1, would sharply slow how fast their effective embedding moves away from the prototype. A cheap pre-thesis check: compare $\|p_{eff}-p_{global}\|$ for sparse users in the adaptive run against $\|p_u\|$ movement in the split-learning run.
4. **The learnable-α path cannot escape the heuristic within the cross-device budget.** This is an estimate under assumptions. With Adam lr=0.005, the logit moves ≲0.005 per step. A sparse user gets $O(10^2)$ local steps over the whole run, so $|\Delta\ell|\lesssim0.5$. In addition, `weight_decay` applies to $\ell_u$ and pulls α toward 0.5. So "APFL-style learned α" in the thesis stays heuristic-initialized in practice. *(Verified 2026-10-27: the cited adaptive run `20260613-152226-869730` has `adaptive_config.enable_per_user_alpha = False`, with `model_type = dual` and `contrastive_lambda = 0.1`. The headline adaptive number therefore comes from the pure heuristic α, and the learnable-α path was never part of the reported result.)*
5. **Does APFL make the heuristic look unjustified?** Partly yes, and the thesis should say so.
   - APFL argues that the optimal α depends on an unobservable distribution gap and should be *learned from the local loss*. The hierarchical-conditional heuristic uses user statistics (n_interactions, coverage, genre entropy, rating std).
   - The heuristic's *signs* are APFL-consistent. The sparse penalty corresponds to small $m_i$ (more global). The niche-specialist +0.15 for low diversity corresponds to large divergence (more local).
   - What it gets wrong is the *magnitude*. In CF the divergence side dominates for all groups, and the [0.1, 0.95] clip forbids the α = 1 that both APFL's convergence residuals and the thesis oracle favor.
   - The heuristic's only defensible advantages are that it is gradient-free and computable at round 0.
6. **Suggested citation framing (1–2 sentences).** "Our α-blending instantiates APFL's model interpolation $\theta_i=\alpha_i v_i+(1-\alpha_i)w$ (Deng et al., 2020) at the level of a single component — the user representation or score head — on top of a shared item-embedding backbone. APFL's analysis predicts that α should shrink with local sample size but grow with local–global distribution shift; under 1-user-1-client MovieLens-1M we find the shift term dominates for every user group (α* = 1.0; bucket-oracle Δ ≈ 0), so mixing in a population-level component does not help even sparse users."

### 6.3 Limitations & Gaps
1. **No recommendation setting.** There is no implicit feedback, no ranking loss (BPR/BCE with sampled negatives), and no NDCG/HR. Theorem 1 is proved for squared hinge and MSE; the claim that it extends to "any convex smooth loss" is not shown for pairwise BPR.
2. **Whole-model mixing is ill-defined for MF.** A global model's user-$u$ row is trained only by user $u$ and then diluted by averaging. That is why the thesis substitutes a population prototype. APFL offers no guidance for this case of partially global, partially local structure; FedRep/FedRecon and FedPer cover it.
3. **No per-client α evidence.** The paper never shows learned $\alpha_i$ against $m_i$ or heterogeneity. Client data sizes are near-uniform in most experiments. The thesis cannot cite APFL for the claim "learned α tracks data volume".
4. **Weak or vacuous theory for this regime.**
   - The bound is vacuous at small $m_i$ and always prefers interior α.
   - The convergence results hold only for fixed α.
   - The nonconvex results need $C=1$, while the thesis is cross-device with C=0.1 and ~10 participations per user.
   - The adaptive-α update is never analyzed.
5. **Cost and privacy.** Each client needs 2× memory and compute. There is no privacy analysis: the full $w_i$ is uploaded, though $v_i$ and $\alpha_i$ stay local, consistent with the thesis threat model. There is no communication saving over FedAvg.
6. **Small scale.** 100–1000 clients; no 6040-client cross-device test.

## 7. Key References to Follow
- **Mansour, Mohri, Ro, Suresh, 2020** ("Three approaches for personalization…"): model interpolation with an α-independent bound; the closest prior formulation. Before using it as the label for the capstone's post-hoc head mixing, check whether its model-interpolation algorithm also fits the local model under the mixture.
- **Hanzely & Richtárik, 2020** ("Federated learning of a mixture of global and local models"): an L2-coupled local/global mixture with a mixing parameter. Precursor of Ditto/pFedMe-style regularization.
- **Dinh, Tran, Nguyen, 2020 (pFedMe)** and **Fallah, Mokhtari, Ozdaglar, 2020 (Per-FedAvg)**: the personalization baselines APFL beats in Table 2. Per-FedAvg is also a PFedRec ancestor.
- **Ben-David et al., 2010; Mansour, Mohri, Rostamizadeh, 2009**: the domain-adaptation theory behind the $\|\bar{\mathcal D}-\mathcal D_i\|_1$ divergence term. Useful for arguing concept shift in CF.
- **Mohri, Sivek, Suresh, 2019 (Agnostic FL)**: posed the open question of how to mix global and local models; APFL's agnostic extension (Eq. 18) is left as future work.
