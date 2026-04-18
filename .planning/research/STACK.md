# Stack Research

**Brownfield note:** The stack is already locked by the existing codebase. This document points to the authoritative map rather than re-deriving it.

## Canonical Stack (from `.planning/codebase/STACK.md`)

- **FL framework**: Flower (flwr) ≥ 1.22.0 — `Grid.send_and_receive(messages)` pub-sub API, `ClientApp`/`ServerApp` hooks, `flwr.server.strategy.FedAvg`/`FedProx` subclassed in the three split modules.
- **ML framework**: PyTorch ≥ 2.7.1 — `nn.Module` with `get_/set_global_parameters()` / `get_/set_local_parameters()` on split-aware models; `Adam` for BPR-MF modules, `SGD` for PFedRec (per-paper).
- **Data**: MovieLens 1M (6,040 users, 3,706 items, ~1M ratings); `pd.read_csv(..., sep="::", engine="python")` loader; Dirichlet + natural partitioners in every module's `dataset.py`.
- **Models**: `BasicMF` (MSE), `BPRMF` (BPR loss, ranking), `PFedRecMLP` (no user embedding, per-user `affine_output`), `DualPersonalizedBPRMF` (thesis).
- **Tracking**: Weights & Biases (wandb) — will use a NEW project for cross-device runs (see PROJECT.md key decisions).
- **Python**: ≥ 3.9 per `pyproject.toml`.

## No Stack Changes in This Cycle

Migration to cross-device is a *configuration* and *semantics* change, not a stack change. No library swaps or version bumps required.

## Confidence

- **High**: Flower + PyTorch + ML-1M stay. The published cross-device FedRec literature (PFedRec, FedRAP, CoFedRec, GPFedRec, P²FedRec) all use this same stack profile.
- **Medium**: Embedding dim / optimizer / negatives are currently inconsistent across modules. Standardizing these is part of the migration (see FEATURES.md "Primary evaluation protocol locked" + "Benchmark vs paper-compat modes").

## References

- `.planning/codebase/STACK.md` — full list of dependencies and versions
- `.planning/codebase/ARCHITECTURE.md` — module structure
- Codex research (2026-04-19) — confirmed this stack is standard for cross-device FedRec work
