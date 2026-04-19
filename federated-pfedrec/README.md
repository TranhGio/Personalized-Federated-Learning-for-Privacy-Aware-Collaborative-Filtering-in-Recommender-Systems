# federated-pfedrec

**Role:** Published calibration baseline in the thesis comparison.
**Approach:** PFedRec (IJCAI-23) — per-user local affine head, global item embeddings, alternating optimization, BCE loss, SGD.
**Calibration target:** Reproduce the IJCAI-23 paper numbers on ML-1M (HR@10 ≈ 0.729, NDCG@10 ≈ 0.441) within ±2 points under `mode = "paper_compat_pfedrec"`.

See [`../README.md`](../README.md) for the four-way thesis comparison context and [`claude.md`](./claude.md) for this module's architecture, parameter classification, and the dual-LR alternating-optimization protocol.

The unmodified upstream reference lives at [`../IJCAI-23-PFedRec/`](../IJCAI-23-PFedRec/) and is the calibration oracle for Phase 5 of the migration.

## Quick Start

```bash
pip install -e .
flwr run .                                                # default benchmark_cross_device
flwr run . --run-config "mode=paper_compat_pfedrec"       # reproduce IJCAI-23 numbers
flwr run . --run-config "mode=cross_silo_legacy"          # pre-migration config
```

Full implementation details, configuration surface, gotchas (alternating optimization, dual LR `lr * num_items * lr_eta`, per-user `affine_output` cache layout, `affine_output.bias` aggregation-scope decision) are in [`claude.md`](./claude.md).
