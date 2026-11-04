# Federated Reconstruction: Partially Local Federated Learning

- **Authors**: Karan Singhal, Hakim Sidahmed, Zachary Garrett, Shanshan Wu, Keith Rush, Sushant Prakash (Google Research)
- **Venue**: NeurIPS 2021 (arXiv 2102.03448v6)
- **Paper ID**: singhal_2021_fedrecon
- **Tags**: #method, #federated-learning, #personalization, #model-splitting, #meta-learning, #matrix-factorization, #collaborative-filtering, #communication-efficiency, #privacy, #cold-start

---

## 1. Core Idea
FedRecon is **partially local FL with stateless clients**: the model is split into global parameters $w$ (aggregated) and local parameters $\theta_i$ (never sent to the server), but $\theta_i$ is **not stored across rounds** — every time client $i$ participates it **reconstructs** $\theta_i$ from scratch (random init, $K_r$ gradient steps on a support set with $w$ frozen), then updates $w$ on a query set with $\theta_i$ frozen, sends only $\Delta w$, and discards $\theta_i$. The same reconstruction runs at inference (RECONEVAL), so clients never seen in training (or whose state is lost) still get a full personalized model; with one client-update step this is provably first-order MAML, i.e. $w$ is trained *to be easy to reconstruct from*.

## 2. Problem & Motivation
For models with user-specific parameters (MF user embeddings $P_u$), fully-global FL (FedAvg, Per-FedAvg, pFedMe, APFL…) must send per-user parameters to the server, which "can be trivially used to recover sensitive user-item affinities" (§3). Partial-aggregation methods that keep them local — FedPer (Arivazhagan 2019), LG-FedAvg (Liang 2020), FCF (Ammad-Ud-Din 2019), FedMF (Chai 2020), FedBN — are **stateful**: (1) in cross-device FL a client is rarely re-sampled, so state is missing or **stale** (citing Reddi et al. 2021 §5.1 on SCAFFOLD); (2) non-sampled clients have no trained local parameters, so 99%+ of a production population could not run inference; (3) they are architecture-specific. FedRecon targets four criteria: model-agnostic, scalable to partial participation, inference for unseen clients, fast local adaptation.

## 3. Method

### 3.1 Objective Function
Parameters are partitioned into global $w \in \mathcal{G}$ (paper: $g$) and local $\theta_i \in \mathcal{L}$ (paper: $l_i$); $w \| \theta_i$ is concatenation. Each client's data is split $(\mathcal{D}_{i,s}, \mathcal{D}_{i,q}) = S(\mathcal{D}_i)$ into support (reconstruction) and query (global update) sets. The meta-learning objective (Eq. 2/3):

$$
\min_{w} \; \mathbb{E}_{i \sim \mathcal{P}}\Big[ f_i\big(w \,\|\, \mathcal{R}(\mathcal{D}_{i,s}, \mathcal{L}, w)\big) \Big], \qquad f_i(w\|\theta_i) = \mathbb{E}_{\xi \in \mathcal{D}_{i,q}}\big[f_i(w\|\theta_i, \xi)\big]
$$

Reconstruction $\mathcal{R}$ = $K_r$ steps of GD on the local parameters from a random init, globals frozen:

$$
\theta_i^{(t,0)} \sim \text{random init}, \qquad \theta_i^{(t,j+1)} = \theta_i^{(t,j)} - \eta_r \nabla_{\theta} f_i\big(w^{(t)} \| \theta_i^{(t,j)}; \mathcal{D}_{i,s}\big), \qquad \theta_i^{(t)} = \theta_i^{(t,K_r)}
$$

Client update $\mathcal{U}$ = $K_u$ steps of GD on $w$ using $\mathcal{D}_{i,q}$, locals frozen; server step (generalized FedAvg of Reddi et al., SGD shown):

$$
\Delta_i^{(t)} = \mathcal{U}(\mathcal{D}_{i,q}, \theta_i^{(t)}, w^{(t)}) - w^{(t)}, \qquad w^{(t+1)} = w^{(t)} + \eta_s \sum_{i \in \mathcal{S}^{(t)}} \frac{n_i}{n} \Delta_i^{(t)}, \quad n_i = |\mathcal{D}_{i,q}|,\; n = \sum_{i\in\mathcal{S}^{(t)}} n_i
$$

**MF instantiation** (§3, §5.1.1): $R \approx P Q^\top$; local $\theta_i = P_u \in \mathbb{R}^{K}$ (user embedding), global $w = Q \in \mathbb{R}^{I \times K}$; loss = squared error on explicit ratings (Fig. 4 plots MSE). No bias terms are mentioned.

**First-order MAML proof (App. B).** With $K_u = 1$ (lr $\eta_u$), the server step equals $w^{(t)} - \eta_s\eta_u \sum_i \frac{n_i}{n} \nabla_w f_i\big(w^{(t)} \| \mathcal{R}(\mathcal{D}_{i,s}, \mathcal{L}, w^{(t)})\big)$ **treating $\mathcal{R}$'s output as a constant**. The exact chain-rule gradient is

$$
\nabla_w f_i(w\|\theta_i^{(K_r)}) - \eta_r\, \nabla_{\theta} f_i(w\|\theta_i^{(K_r)}) \sum_{j=0}^{K_r-1} \nabla_w \nabla_{\theta} f_i(w\|\theta_i^{(j)})
$$

(using $\theta_i^{(0)}$ independent of $w$); dropping the mixed second-order (Jacobian) terms leaves the FedRecon update → FedRecon with $K_r \ge 1$, $K_u = 1$ is first-order MAML: reconstruction = inner loop, global update = outer loop, "test the same way we train". Keeping the dropped terms costs memory quadratic in $|\theta_i|$ — impractical on device. $K_u > 1$ is to $K_u = 1$ what FedAvg is to FedSGD.

Where:
- $w$ — global params (MF: item matrix $Q$; NWP: LSTM + core-vocab embeddings); $\theta_i$ — local params (MF: $P_u$; NWP: OOV embeddings)
- $\mathcal{P}$ — client distribution; $\mathcal{S}^{(t)}$ — $m$ clients sampled in round $t$ ($C = m/N$)
- $\eta_r, \eta_u, \eta_s$ — reconstruction, client-update, server learning rates
- $K_r, K_u$ — reconstruction steps / client-update steps (paper: $k_r$, $k_u$)

### 3.2 Algorithm

**Server-side (Algorithm 1):**
1. Initialize $w^{(0)}$.
2. For each round $t = 0, \dots, R-1$: sample $m$ clients $\mathcal{S}^{(t)}$ uniformly; send $w^{(t)}$ (only globals).
3. Receive $(\Delta_i^{(t)}, n_i)$; aggregate $w^{(t+1)} = w^{(t)} + \eta_s \sum_i \frac{n_i}{n}\Delta_i^{(t)}$ (or feed the averaged $\Delta$ as an "antigradient" to Adagrad/Yogi/Adam ServerOpt).
4. **Server never receives, stores, or initializes any $\theta_i$.**

**Client-side (ClientUpdate):**
1. Split $(\mathcal{D}_{i,s}, \mathcal{D}_{i,q}) \leftarrow S(\mathcal{D}_i)$ (MF: half/half; NWP: half/half **by timestamp** — random splits make support and query too similar and inflate results).
2. **Reconstruct**: random-init $\theta_i$, run $K_r$ GD steps on $\mathcal{D}_{i,s}$ with $w^{(t)}$ frozen.
3. **Update**: $K_u$ GD steps on $w$ using $\mathcal{D}_{i,q}$ with $\theta_i^{(t)}$ frozen → $w_i^{(t)}$.
4. Return $\Delta_i = w_i^{(t)} - w^{(t)}$ and $n_i = |\mathcal{D}_{i,q}|$. **Discard $\theta_i^{(t)}$.**

**Evaluation / inference (RECONEVAL, §4.1):** given final $w$, any client (seen or unseen) runs the same $\mathcal{R}$ on its support data, then predicts with $w\|\theta_i$ on its query data. Done offline, independent of any federated process; the client may store the reconstructed $\theta_i$ for repeated inference and refresh it when it has new data (App. A explicitly allows this — the objection is only to state carried *across training rounds*).

**Variants (App. D):** *No Split* ($\mathcal{D}_{i,s} = \mathcal{D}_{i,q} = \mathcal{D}_i$; used in production for sparse clients); *Joint Training* (update $w$ and $\theta_i$ concurrently after reconstruction — slightly worse); *FedRecon+Finetune* (after reconstruction, also finetune $w$ on the support set before evaluation).

### 3.3 Key Hyperparameters
| Hyperparameter | Symbol | Default Value | Role |
|---|---|---|---|
| Embedding dim (MF) | $K$ | 50 | $P_u, Q_j \in \mathbb{R}^{50}$ |
| Support/query split | $S$ | 50% / 50% | Disjoint by default; "No Split" ablation reuses all data |
| Reconstruction steps | $K_r$ | up to 50 (MF), up to 100 (NWP) | Plateau ~10 steps; no gain past 50/100 |
| Client update steps | $K_u$ | up to 50 (MF), up to 100 (NWP) | $K_u = 1$ is exact FOMAML but trains slower |
| Local batch size | $B$ | 5 (MF), 16 (NWP) | Per GD step |
| Server LR | $\eta_s$ | grid {0.1, 0.5, 1.0} (MF); {0.01, 0.1, 0.3} (NWP) | Server optimizer step |
| Reconstruction LR | $\eta_r$ | grid {0.1, 0.5} (MF); {0.1, 0.3} (NWP) | Heuristic: set $\eta_r \approx \eta_u$ or slightly lower |
| Client update LR | $\eta_u$ | grid {0.1, 0.5} (MF); {0.1, 0.3} (NWP) | = client LR of generalized FedAvg |
| Rounds | $R$ | 500 (MF), 2500 (NWP) | — |
| Clients per round | $m$ | 100 (MF), 200 (NWP) | MF: $C \approx 100/6040 = 0.017$ (seen-user eval) or $100/4832 = 0.021$ (unseen-user eval) → each user participates **~8–10 times** |
| Server optimizer | — | SGD (MF main), Adagrad (MF best), Yogi (NWP) | Generalized FedOpt |
| Local params init | $\theta_i^{(t,0)}$ | random | Every round (stateless) |
| Centralized baseline | — | 20 epochs, batch 300 | MF only |
| Seeds | — | 3 reruns, averaged | Best config by final validation |
| OOV buckets (NWP) | — | 500 (local) | Plateau at 500 for all vocab sizes |

### 3.4 Architectural Decisions
- **Reconstruct instead of store.** In cross-device FL a client is rarely re-reached, so reconstruction costs about the same as initializing and training stateful local params; it also keeps $\theta_i$ always **fresh** (no staleness) and gives unseen clients a model without communication (App. A).
- **Alternating, not joint.** Locals are trained with globals frozen (reconstruction), then globals with locals frozen (update). Freezing $\theta_i$ during the update "is useful for ensuring that global parameters are updated significantly" — otherwise progress across rounds stalls because locals are not aggregated (Joint Training ablation is worse). Related to gradient-based alternating minimization for factored models (Gunawardana 2005; Jain 2013).
- **Disjoint support/query** maximizes meta-generalization (MAML-style); relaxable at small cost.
- **Model-agnostic split.** For MF the split is forced by privacy (user embedding local). For other models it's chosen by use case / communication budget: more local params → less communication (NWP: local OOV embeddings let the core vocabulary shrink).
- **Server optimizer pluggable** (FedOpt): Adagrad improves MF.
- **Privacy argument (App. E):** local params never leave the device; global updates are computed only on the query subset; no formal guarantee — gradient-leakage attacks (DLG, Inverting Gradients, GradInversion) still apply to $\Delta w$, so pair with DP or SecAgg. Splitting reduces the dimensionality $d$ of aggregated params, and DP-SGD noise scales with $\sqrt{d}$ → fewer global params = less noise for user-level DP ("composes rather than competes" with DP advances).

## 4. Implementation Notes
- **Framework used in paper**: TensorFlow Federated (Keras models plugged into `tff.learning.reconstruction`); code: `github.com/google-research/federated/tree/master/reconstruction` (Apache 2.0); TFF tutorial "Federated Reconstruction for Matrix Factorization".
- **Model architecture**: MF $\hat r_{uj} = P_u^\top Q_j$, $K = 50$. NWP: Reddi et al. LSTM (input embedding dim 96, output dim 670), vocab {1K, 5K, 10K} + up to 500 local OOV buckets (hashing trick), 20-token sentences, ≤1000 sentences/client.
- **Training details**: local SGD for both $\mathcal{R}$ and $\mathcal{U}$; MF batch 5, ≤50 steps each, 500 rounds × 100 clients; server SGD (Adagrad variant reported separately). Best config chosen by final validation; mean of 3 runs.
- **Data preprocessing (MF)**: ML-1M **explicit ratings**, MSE. Two protocols: (1) STANDARD EVAL on seen users — each user's ratings split 80/10/10 **by timestamp**, report on test ratings; (2) RECONEVAL on unseen users — **users** split randomly 80/10/10 (train/val/test users), train on train users, reconstruct + evaluate test users. Metrics: RMSE and **rating accuracy** (round prediction to nearest integer, exact match). **No ranking metrics, no negatives, no LOO.**
- **Tricks & gotchas**:
  - $K_r = 0$ (skip reconstruction) gives MovieLens accuracy **0.0** — random user embeddings are garbage; NWP keeps >90% relative accuracy without reconstruction (shared model already useful).
  - Production (Gboard expression recommendation, hundreds of millions of clients, MF with hundreds–tens of thousands of items): oversample clients and **filter out clients below a minimum example count** (1-example clients add noise); reuse examples across support/query (No Split); drop stragglers. Result: **+29.3% CTR**.
  - Timestamp-based support/query split matters for NWP (random split inflates results).
  - Centralized MF + RECONEVAL is extremely high-variance across users (500 draws × 50 clients: std 1.7 pts accuracy, 0.53 RMSE) — some users reconstruct no better than random init.

## 5. Experimental Results

### 5.1 Datasets Used
| Dataset | Users | Items | Interactions | Density |
|---|---|---|---|---|
| MovieLens-1M (explicit ratings) | 6,040 | 3,706 | 1,000,209 | 4.47% |
| Stack Overflow (TFF NWP) | 342,477 train + 38,758 held-out clients | vocab 1K/5K/10K + OOV | 135.8M train + 16.5M held-out examples | N/A |

### 5.2 Key Results

**MovieLens-1M matrix factorization (Tables 1 & 3; explicit ratings, NOT the thesis protocol):**

| Method | Eval population | RMSE ↓ | Rating Acc. ↑ (%) |
|---|---|---|---|
| Centralized + STANDARD EVAL | seen users, held-out ratings | 0.923 | 43.2 |
| Centralized + STANDARD EVAL (Unseen) | unseen users, random $P_u$ | 3.80 | 0.0 |
| Centralized + RECONEVAL | unseen users, reconstructed $P_u$ | 1.36 | 40.8 |
| FedAvg + STANDARD EVAL (≈ stateful FedPer) | seen users | 0.939 | 41.5 |
| FedAvg + RECONEVAL | unseen users | 0.934 | 40.0 |
| **FedRecon** | **unseen users (RECONEVAL)** | **0.907** | **43.3** |
| FedRecon (No Split) | unseen users | 0.912 | 42.5 |
| FedRecon (Joint Training) | unseen users | 0.915 | 42.1 |
| FedRecon (Adagrad server) | unseen users | **0.883** | **44.1** |

FedRecon on *unseen* users matches/beats centralized MF on *seen* users, and beats FedAvg+RECONEVAL (0.907 vs 0.934) → $w$ trained by FedRecon is more "reconstructable". Footnote 4: FedAvg here is equivalent in result to stateful FedPer (each client only updates its own user row; rows stored on server); its gap "may be caused by user embeddings getting stale across rounds" — a hypothesis, not isolated by an ablation (eval populations differ).

**ML-1M FedRecon numbers reported by others under the thesis-style LOO + 99-negatives protocol (implicit):** PFedRec (IJCAI-23, Table 2, full participation): FedRecon HR@10 63.28 / NDCG@10 36.59; FedRecon + DualPer 68.89 / 40.04; PFedRec 73.26 / 44.36. CoFedRec (2024): FedRecon 60.43 / 34.89.

**Stack Overflow NWP (Table 2; accuracy %, comm per client per round):**

| Method | 1K | 5K | 10K | Comm |
|---|---|---|---|---|
| FedYogi | 24.3 | 26.3 | 26.7 | $2|\theta|+2|w|$ |
| FedRecon (1 OOV) | 24.1 | 26.2 | 26.4 | $2|w|$ |
| FedRecon (500 OOV) | 29.6 | 28.1 | 27.7 | $2|w|$ |
| OOV Finetuning (500 OOV) | 30.0 | 28.1 | 27.9 | $2|\theta|+2|w|$ |
| Full Finetuning (500 OOV) | 30.8 | 29.2 | 28.8 | $2|\theta|+2|w|$ |
| FedRecon+Finetune (500 OOV) | 30.7 | 28.9 | 28.6 | $2|w|$ |

Reconstruction ≈ finetuning pre-trained local params, without ever communicating them; best accuracy-per-communicated-parameter (Fig. 2). Smaller core vocab *helps* FedRecon (500 OOV) because local OOV embeddings get more training signal.

### 5.3 Ablation Highlights
- **Reconstruction steps (Fig. 3 left, MovieLens, relative accuracy, read from plot):** $K_r$=0 → 0.0; 1 → ~0.81; 2 → ~0.93; 5 → ~0.94; 10 → ~0.98. **One step recovers most performance**; MF plateaus ~10 steps.
- **Client update steps (Fig. 3 right, MovieLens):** $K_u$=1 → ~0.97, rising to ~1.0 at 10; $K_u$=1 trains noticeably slower per round.
- **No Split** costs little (RMSE 0.907 → 0.912) → disjoint support/query is optional for sparse clients.
- **Joint Training** after reconstruction is worse (0.915) → freeze locals during the global update.
- **Server optimizer:** Adagrad > SGD (0.883 vs 0.907 RMSE).
- **Without reconstruction, unseen-user MF is unusable** (RMSE 3.80, acc 0.0) — the same failure the thesis hit as its cold-eval bug.

## 6. Connections to My Thesis

### 6.1 Directly Applicable Ideas
1. **The thesis's final-calibration (D-06.7) is FedRecon's RECONEVAL reconstruction step, warm-started, so FedRecon must be cited as prior art.** `final-calibration-enabled=true` + `final-calibration-freeze-items=true` (lr_eta=0) runs local-only GD on each user's `affine_output` with the restored global item embeddings frozen, for all 6040 users, once after training. That is exactly $\mathcal{R}(\mathcal{D}_i, \mathcal{L}, w^\star)$ from §4.1, done offline. There are three differences. (a) $\theta_i^{(0)}$ is the cached head, not a random init. (b) The support set is the whole train set (FedRecon "No Split"). (c) The budget is 1 epoch, not a fixed step count. Run `cf72e0`'s result (0.3352 → 0.3447, closing 45.2% of the 0.0211 gap to the 0.3563 in-loop ceiling, sparse users +2.1 pts → 0.4270) is therefore a **measurement of FedRecon-style reconstruction applied to a stateful method under C=0.1**, not a new mechanism. Cite FedRecon §4.1 + App. A ("reconstruction ensures local parameters are always fresh, avoiding staleness"), FedPer's fine-tuning variant (freeze base, 1 epoch on personal layers), and Reddi et al. 2021 §5.1 for stale client state.
2. **The thesis turns FedRecon's staleness hypothesis into a measurement.** FedRecon only asserts that stateful FL degrades from staleness. Its MF evidence (footnote 4) compares different eval populations, so it never isolates the effect. The thesis has a clean, bit-exact-verified number: same run, same restored $w$, cached vs calibrated heads, gap 0.0211, 45% recoverable by one reconstruction epoch. Frame it as "we quantify the staleness cost FedRecon (2021) predicted for stateful partially-local FL." The regime matches FedRecon's own ML-1M simulation. FedRecon's 100 clients/round × 500 rounds means each user participates ~8–10 times. The thesis's C=0.1 × 100 rounds gives ~10 participations, with P(never sampled) = $0.9^{100} \approx 2.7\times10^{-5}$ (~0.16 users expected). The cross-device "state rarely exists" argument does not apply at thesis scale. The "state goes stale" argument does.
3. **The cold-eval bug class is FedRecon's $K_r = 0$ row.** FedRecon: unseen users with random $P_u$ give RMSE 3.80 and accuracy 0.0, and $K_r$=0 gives relative accuracy 0.0. Thesis: when the cache is not loaded, full-pop NDCG@10 craters to 0.0711, matching the random-Kaiming-head counterfactual (0.0741). FedRecon's stateless design makes evaluation self-contained (reconstruct-then-score), so this bug class becomes **structurally impossible** rather than something guarded by `run_id`/`reuse_cache` stamps and `d06_cache_misses=0` gates. The methodology chapter can cite FedRecon for this: stateful personalization has a silent failure mode at eval; stateless reconstruction removes it at the cost of $K_r$ local steps per inference.
4. **The calibration budget is tiny and non-uniform, and FedRecon's Fig. 3 points to the next lever.** At batch 256 with 4 negatives, 1 epoch is about $\lceil 5|\mathcal{D}_i|/256 \rceil$ head SGD steps. Sparse users (≤30 interactions, so ≤~145 samples) get **one step**, medium users 1–2, dense users ~2–8+. On ML-1M MF, FedRecon finds $K_r$=1 recovers ~81% and $K_r$=10 ~98%. The knob already exists (`final-calibration-epochs`, default 1). Raising it, or switching to a fixed step budget, is the obvious way to chase the remaining ~0.0115 headroom, especially for sparse users. Caveat: FedRecon's curve is for random init with $w$ meta-trained for reconstruction, so it is an analogy, not a prediction, for warm-started PFedRec heads.
5. **PFedRec is a stateful, per-batch-interleaved FedRecon.** PFedRec's alternating update (head step with items fixed, then item step with head fixed) is FedRecon's $\mathcal{R}$/$\mathcal{U}$ alternation, interleaved per batch, with the head carried across rounds instead of reconstructed. FedRecon's Joint-Training ablation (worse) supports PFedRec's choice to alternate. The thesis should state that PFedRec's +7.8 NDCG pts over FedRecon (44.36 vs 36.59) was measured at **full participation (C=1.0)**, the one regime where stateful methods have zero staleness. Under the thesis's C=0.1, the stateful head pays a measured staleness cost.

### 6.2 Potential Integration Points
1. **Offline "stateless eval" sweep (cheap, reuses the bit-exact `scripts/thesis/reeval.py` adapter; run on the workstation where `cf72e0`'s `global_state_*.pt` and cache live).** Freeze the restored item embeddings. Reconstruct every user's `affine_output` with init ∈ {cached head, population-mean head (the α=0 arm, 0.2510), Kaiming random (0.0741)} × budget ∈ {1 epoch, 5 epochs, to convergence}. Fix the negative sampler with seed 42 *inside* the sampler. Prediction: with items frozen, the PFedRec head is a 33-parameter logistic regression (BCE), which is convex in the head. Run to convergence, the result should be **init-independent**. If random-init-to-convergence ≈ 0.3447, the per-user cache is unnecessary for evaluation. PFedRec's eval can then be made FedRecon-stateless, which kills the cold-eval bug class and strengthens the privacy story (no between-round device state is needed for inference). The same convexity holds for split learning: BPR loss in $p_u$ with $Q$ frozen is convex, and $b_u$ cancels in BPR. The adaptive module's PersonalMLP is not convex.
2. **Population head as reconstruction *init*, not as blend target.** FedRecon footnote 2 suggests learning "a global default for the local parameters." In MAML terms, a population prior matters as the **starting point for few-step adaptation**. The killed capstone used it as a score-time blend target ($w = \alpha w_{local} + (1-\alpha) w_{global}$; oracle α* = 1.0, Δ = −0.0002). The init arm of the sweep above tests the MAML-style use directly, at 1-step budget, where sparse users live. This adds one more rung to the "where should personalization live" chain: state vs reconstruction vs blend. The expected outcome is honest either way: with α* = 1.0 the cached head should still win at a 1-step budget, and init should not matter at convergence.
3. **Threat-model section: adopt FedRecon's partially-local framing, and disclose a leak in the adaptive module.** FedRecon §3: if the server holds a user's embedding and the item matrix, it can trivially recover that user's affinities ($p_u Q^\top$). Per module:
   - **baseline (all-global):** the upload includes the user-embedding table, whose only trained row is $p_u$. Server learns $p_u$ → **full leak** (FedRecon's "unreasonable" case).
   - **pfedrec:** uploads only item embeddings (3706×32 = 118,592 params). The 33-param head never leaves the device ✓.
   - **personalized (split):** uploads item embeddings, item bias, and global bias. $p_u$ and $b_u$ stay local ✓.
   - **adaptive:** ✗ under cross-device. `compute_user_prototype(user_id=partition_id)` returns the **trained row $p_u$**, and the client uploads it every participation as the `USER_PROTOTYPE_KEY` MetricRecord (`federated-adaptive-personalized-cf/federated_adaptive_personalized_cf/client_app.py` ~L740–786) to feed the server EMA. The adaptive module therefore transmits exactly the parameter split learning is meant to protect, so the server can compute $p_u Q^\top$. Mitigation: the EMA only needs the weighted *sum*, so secure aggregation (Bonawitz 2017) or DP noise on the prototype restores split-learning privacy. Disclose this regardless, since the adaptive module is already falsified on accuracy.
   - **all four:** the item-embedding $\Delta$ is row-sparse (only positives + 4× sampled negatives are touched), so the upload reveals a superset of the interacted items. This is the classic federated-MF gradient leak that motivated FedMF's homomorphic encryption (Chai et al. 2020; see also Gao et al. 2020), and FedRecon does not address it. Every method also uploads the example count $n_i$ (FedRecon does too).
   - **Communication table:** use FedRecon's per-client-per-round accounting, $2|w|$ for partially local methods vs $2|w| + 2|\theta|$ for fully global ones. The App. E.1 point ($\sqrt{d}$ DP noise) means PFedRec's small $|w|$ is also a DP-friendliness argument.
4. **Re-read Exp 1 (+20.9% split vs all-global) through FedRecon footnote 4 (hypothesis to verify, CPU-only).** FedRecon claims FedAvg on MF ≈ stateful FedPer because each client updates only its own row. The thesis baseline enforces this structure: the D-24 row-grad mask plus snapshot/restore of non-own rows in `federated-baseline-cf/federated_baseline_cf/task.py`. But Flower's weighted parameter averaging gives row $u$ an aggregated update of only $(n_u/n)\Delta_u$, roughly $\Delta_u/604$ at C=0.1 with $\eta_s = 1$. The gap may then mostly measure **FedAvg dilution of per-user rows**, not an intrinsic cost of "averaging user embeddings". Check by logging $\|p_u^{\text{after agg}} - p_u^{\text{before}}\| / \|\Delta_u\|$ for one round. If it is ≈ $n_u/n$, state that mechanism in the thesis; it matches FedRecon §3's argument that per-user params cannot be trained by global FL.
5. **Optional, not recommended now:** a training-time stateless PFedRec (reconstruct the head each round, then alternating update) is "FedRecon + DualPer" (68.89/40.04 vs PFedRec 73.26/44.36 at C=1.0). It costs ~8h GPU, and the open thesis needs (significance tests, threat model) are higher priority.

### 6.3 Limitations & Gaps
- **Different task and protocol:** explicit ratings, MSE, RMSE/rating accuracy. No implicit feedback, BPR/BCE, LOO+99 negatives, or HR/NDCG. FedRecon's own ML-1M numbers **cannot** go in the thesis comparison table. If a FedRecon NDCG is cited, use PFedRec's 63.28/36.59 (C=1.0) or CoFedRec's 60.43/34.89. **Repo issue:** `results/comparison_latest.md`, `results/comparison_loo_*.md`, `results/federated/personalized/BENCHMARK_COMPARISON.md`, and `scripts/compare_all_results.py` list "FedRecon 64.45% / 37.78% (TFF)". That number appears neither in FedRecon (no ranking metrics) nor in PFedRec/CoFedRec, so it is unsourced and should be replaced or removed.
- **The staleness evidence is confounded:** FedRecon rows are RECONEVAL on unseen users while FedAvg/centralized rows are on seen users' held-out ratings, so "stateful is worse" is not isolated. The thesis's own same-run measurement is stronger evidence.
- **No per-user-group analysis:** only an anecdote that some users reconstruct no better than random. No evidence on whether reconstruction from a small support set hurts sparse users, which is the thesis's load-bearing group.
- **No convergence theory:** only the first-order MAML correspondence (exact for $K_u = 1$) plus empirical loss curves.
- **No formal privacy:** DP and SecAgg are only discussed. Leakage through the row-sparsity of item-matrix updates is not addressed.
- **Unseen-user generalization (RECONEVAL on held-out users) is not part of the thesis protocol:** every one of the 6040 users is a training participant. Say this explicitly as a scope limitation rather than claiming cross-device deployability.
- **Simulation regime:** the "a client participates at most once" premise is production-only. FedRecon's ML simulation and the thesis both have ~10 participations per user.

## 7. Key References to Follow
- **Reddi et al., ICLR 2021 (Adaptive Federated Optimization)** — source of the stale-client-state argument (§5.1) and the generalized FedAvg server step FedRecon uses; already digested as `reddi_2021_adaptive_fedopt`.
- **Gao et al., 2020 (Privacy Threats Against Federated Matrix Factorization)** — the attack model FedRecon cites for why per-user params must stay local; needed for the thesis threat-model section; pair it with Chai et al. 2020 (FedMF) for the item-gradient leakage argument.
- **Jiang et al., 2019 (Improving FL personalization via MAML)** and **Fallah et al., NeurIPS 2020 (Per-FedAvg)** — the FL↔MAML link FedRecon builds on; useful when framing calibration as few-step adaptation.
- **Bonawitz et al., CCS 2017 (Practical Secure Aggregation)** — the mitigation for the adaptive module's per-user prototype upload.
- **Ammad-Ud-Din et al., 2019 (FCF)** — original federated CF with local user embeddings (stateful), the design lineage of `federated-personalized-cf/`.
