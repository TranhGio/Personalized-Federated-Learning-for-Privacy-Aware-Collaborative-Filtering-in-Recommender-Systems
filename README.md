# Federated Movie Recommendation System

Master's thesis · **Personalized Federated Learning for Privacy-Aware Collaborative Filtering**
Author: Dang Vinh (2026) · Dataset: MovieLens 1M · Stack: Flower (flwr) ≥ 1.22 + PyTorch ≥ 2.7

## What This Repository Is

A thesis research codebase comparing **four federated collaborative-filtering approaches** on MovieLens 1M under a **cross-device protocol** (1 user = 1 client, N = 6040). The four implementations sit at different points on the personalization-vs-privacy axis and are evaluated apples-to-apples under a single standardized protocol.

## The Four Federated Implementations

| Module | Approach | Role in thesis |
|--------|----------|----------------|
| [`federated-baseline-cf/`](./federated-baseline-cf/) | All parameters global (FedAvg / FedProx), BPR-MF | Lower-bound baseline |
| [`federated-pfedrec/`](./federated-pfedrec/) | PFedRec (IJCAI-23): local per-user affine head, global item embeddings | Published calibration baseline |
| [`federated-personalized-cf/`](./federated-personalized-cf/) | Split learning: local user embeddings, global item embeddings | Privacy + personalization baseline |
| [`federated-adaptive-personalized-cf/`](./federated-adaptive-personalized-cf/) | Hierarchical-conditional α + dual-level personalization + prototype EMA | **Thesis contribution** |

Each module is an independent Flower app (`flwr run .` from its directory). The upstream reference implementation is preserved unmodified in [`IJCAI-23-PFedRec/`](./IJCAI-23-PFedRec/) for calibration.

Centralized reference baselines (kept as-is, already-correct upper bound): `centralized_baseline_svd.ipynb`, `centralize_baseline_ncf.py`.

## Thesis Claim

Under a correct cross-device protocol, the adaptive/hierarchical-conditional method must:

1. Beat all three baselines on **overall NDCG@10**.
2. Beat all three baselines on **sparse-user NDCG@10** (the strongest and most defensible thesis claim).
3. While Flower PFedRec reproduces the published IJCAI-23 numbers (HR@10 ≈ 0.729, NDCG@10 ≈ 0.441) within ±2 points.

Methodological correctness is non-negotiable. If the adaptive method does not win under the corrected protocol, the thesis contribution is rethought — not the protocol.

## Current Status

Active migration from cross-silo (`num-supernodes = 5`) to cross-device (`num-supernodes = 6040`). Plan is laid out in `.planning/ROADMAP.md` (7 phases, M1 = Phases 1–6 migration, M2 = Phase 7 thesis evaluation). Every phase's requirements and success criteria are observable, protocol-level behaviors — not code-internal metrics.

## Quick Start

```bash
# Install any of the four modules (editable)
cd federated-baseline-cf              && pip install -e .
cd ../federated-pfedrec                && pip install -e .
cd ../federated-personalized-cf        && pip install -e .
cd ../federated-adaptive-personalized-cf && pip install -e .

# Run the thesis module (defaults change post-migration)
cd ../federated-adaptive-personalized-cf
flwr run .
```

Post-migration, each module exposes a single top-level `mode` selector:

```bash
flwr run . --run-config "mode=benchmark_cross_device"   # thesis default (N=6040, natural partitioning)
flwr run . --run-config "mode=paper_compat_pfedrec"     # federated-pfedrec/ only — reproduces IJCAI-23
flwr run . --run-config "mode=cross_silo_legacy"        # reproduce pre-migration appendix results
```

Mode-locked settings (`num-supernodes`, `partition-mode`, `weight-policy`, `eval-protocol`, and training hyperparameters) can still be overridden at the CLI for debugging; every override is captured in the run manifest and logged loudly at startup.

## Evaluation Protocol (Locked)

- **Primary metric:** NDCG@10
- **Protocol:** leave-one-out + 99 negative samples (NCF convention), selector name `sampled_loo_99`
- **User-group slicing:** sparse (≤ 30 interactions), medium (30–100), dense (> 100) — first-class reported fields
- **Reporting:** best-round-restored metrics are canonical; last-round kept as diagnostic
- **Artifacts:** `results/federated/<module>/<run_id>/` with a full protocol fingerprint manifest

## Documentation Map

- **Developer guide (authoritative):** [`CLAUDE.md`](./CLAUDE.md)
- **Thesis scope & decisions:** [`.planning/PROJECT.md`](./.planning/PROJECT.md)
- **Requirements (52 v1 items):** [`.planning/REQUIREMENTS.md`](./.planning/REQUIREMENTS.md)
- **Migration roadmap (7 phases):** [`.planning/ROADMAP.md`](./.planning/ROADMAP.md)
- **Research summary (26 pitfalls, table-stakes features):** [`.planning/research/SUMMARY.md`](./.planning/research/SUMMARY.md)
- **Codebase map:** [`.planning/codebase/ARCHITECTURE.md`](./.planning/codebase/ARCHITECTURE.md) · [`CONCERNS.md`](./.planning/codebase/CONCERNS.md)
- **Paper knowledge base:** [`Papers/digested/_INDEX.md`](./Papers/digested/_INDEX.md) (see also [`README_PAPER_KB.md`](./README_PAPER_KB.md))
- **Per-module internals:** `<module>/claude.md` alongside each of the four modules

## Repository Layout

```
.
├── federated-baseline-cf/              # All-global baseline
├── federated-pfedrec/                  # PFedRec calibration baseline
├── federated-personalized-cf/          # Split-learning baseline
├── federated-adaptive-personalized-cf/ # Thesis contribution
├── IJCAI-23-PFedRec/                   # Upstream reference (unmodified)
├── Papers/                             # Paper knowledge base (raw/, digested/)
├── data/
│   ├── ml-1m/                          # Auto-downloaded MovieLens 1M
│   └── derived/                        # Foundation artifacts (mapping, split manifest, exclusion set) — Phase 1+
├── results/
│   ├── centralized/                    # SVD / NCF centralized reference
│   └── federated/<module>/<run_id>/    # Cross-device federated runs
├── scripts/                            # Sweep / comparison / orchestration scripts
├── .planning/                          # GSD thesis planning (PROJECT, REQUIREMENTS, ROADMAP, phases, research)
├── centralized_baseline_svd.ipynb
├── centralize_baseline_ncf.py
├── CLAUDE.md                           # Developer guide
└── README.md                           # This file
```

## Citation

```bibtex
@mastersthesis{vinh2026personalizedfl,
  title={Personalized Federated Learning for Privacy-Aware Collaborative Filtering in Recommender Systems},
  author={Dang Vinh},
  year={2026}
}
```

## License

Apache License 2.0.

## Acknowledgments

- [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) — dataset
- [Flower](https://flower.ai) — federated-learning framework
- [PFedRec (IJCAI-23)](https://www.ijcai.org/proceedings/2023/507) — calibration baseline
