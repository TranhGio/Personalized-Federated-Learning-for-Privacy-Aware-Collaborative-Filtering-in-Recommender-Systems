# federated-personalized-cf

**Role:** Split-learning baseline in the thesis comparison.
**Approach:** User embeddings stay **local** and are cached per client; only item embeddings + biases are aggregated on the server. FedAvg / FedProx strategy with split-aware proximal term (regularizes global params only).
**Model:** BPR-MF (ranking, default) or BasicMF (MSE) on MovieLens 1M.

See [`../README.md`](../README.md) for the four-way thesis comparison context and [`claude.md`](./claude.md) for module-level architecture and parameter-classification detail.

## What This Module Does

Under cross-device (`num-supernodes = 6040`, one user per client), each selected client:

1. Receives global params (item embeddings + biases) from the server.
2. Loads its user's local embedding from `.embedding_cache/<run-scoped-path>/user_embeddings.pt`.
3. Trains locally on its single user's data, including proximal term on global params only.
4. Saves the updated local user embedding back to disk.
5. Returns only the global params to the server.

**Personalization boundary:**

| Parameter | Where it lives | Aggregation |
|-----------|----------------|-------------|
| User embedding row (1 × d) | Local (private, cached) | None — never transmitted |
| User bias (scalar) | Local (private, cached) | None — never transmitted |
| Item embeddings (3706 × d) | Global | FedAvg / FedProx |
| Item biases | Global | FedAvg / FedProx |
| Global bias | Global | FedAvg / FedProx |

Communication per round: ~485K params (≈ 44% of the all-global baseline). User preferences never leave the client.

## Quick Start

```bash
pip install -e .

# Default cross-device benchmark
flwr run .

# FedProx with split-aware proximal term
flwr run . --run-config "strategy=fedprox proximal-mu=0.01"

# Reproduce pre-migration cross-silo run
flwr run . --run-config "mode=cross_silo_legacy"

# Purge cached user embeddings (forces cold start)
rm -rf .embedding_cache/
```

## Configuration Surface (`pyproject.toml`)

Mode-locked by the top-level `mode` selector. Key overrides:

| Key | Default (benchmark) | Purpose |
|-----|---------------------|---------|
| `num-supernodes` | 6040 | Client universe (= `num_users`) |
| `partition-mode` | `natural` | `natural` = 1 user / 1 client |
| `fraction-train` | swept | Per-round client sampling fraction `C` |
| `weight-policy` | `num_positives` | Aggregation weighting (global params only) |
| `strategy` | `fedavg` | `fedavg` (`SplitFedAvg`) / `fedprox` (`SplitFedProx`) |
| `model-type` | `bpr` | `bpr` / `basic` |
| `embedding-dim` | 128 | Latent factor dimensionality |

## Gotchas

- **`.embedding_cache/` is run-namespaced.** Path includes `run_id / method / num_users / num_items / dim / split_hash`. Cache loads hard-fail on shape or schema mismatch instead of silently partial-loading. Delete the folder to force a cold start.
- **Split-aware FedProx.** Proximal term applied **only to global parameters**. Local user embeddings are NOT proximally regularized — they're meant to personalize freely.
- **Test-positive exclusion.** Training negatives exclude the held-out LOO test positive (fix inherited from Phase 1 foundation).
- **Local user-row collapse.** Under cross-device, the client holds a single 1 × d row, not the full 6040 × d table ghost. Memory and I/O scale with local state, not global user count.
- **Split parameter API.** All split-aware models expose `get_global_parameters()`, `set_global_parameters()`, `get_local_parameters()`, `set_local_parameters()` — these are the critical seams for the server / client split. See `strategy.py` for `GLOBAL_PARAM_KEYS` / `LOCAL_PARAM_KEYS` frozensets.

## Testing

```bash
python test_dataset.py
python test_models.py
```

## Results Location

`results/federated/personalized/<run_id>/` with full protocol fingerprint manifest.

## References

- Arivazhagan et al., "Federated Learning with Personalization Layers (FedPer)," 2019.
- Singhal et al., "Federated Reconstruction: Partially Local Federated Learning," NeurIPS 2021.
- Li et al., "FedProx," MLSys 2020 — split-aware proximal term variant is a thin modification.
