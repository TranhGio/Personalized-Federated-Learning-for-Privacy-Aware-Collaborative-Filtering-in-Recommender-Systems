# Federated Personalized (Split Learning)

> Local user embeddings + global item embeddings. Middle step: baseline → **this** → adaptive.

**Role in thesis**: shows that keeping user embeddings local (privacy) already helps before any α-machinery enters the picture. If split learning alone doesn't beat baseline, the adaptive module isn't winning for the reason we think.

## Split Learning Boundary

| Scope | Params | Notes |
|---|---|---|
| **Local** (private, cached in `.embedding_cache/`) | `user_embeddings.weight`, `user_bias.weight` | Never transmitted. Accumulate across rounds. |
| **Global** (aggregated) | `item_embeddings.weight`, `item_bias.weight`, `global_bias` | ~485K params (-44% vs baseline) |

### Round Flow
1. Server sends globals → client.
2. Client loads locals from `.embedding_cache/partition_{id}/user_embeddings.pt`.
3. Train all params locally.
4. **FedProx proximal term applies to globals ONLY** — user embeddings are NOT regularized (that's the whole point).
5. Save locals to cache; send only globals back.

## Run

```bash
flwr run .                                              # default: BPR-MF + SplitFedAvg
flwr run . --run-config "strategy=fedprox proximal-mu=0.01"
flwr run . --run-config "model-type=basic"
rm -rf .embedding_cache/                                # reset local embedding cache
```

Defaults: `num-server-rounds=10`, `local-epochs=5`, `embedding-dim=128`, `alpha=0.5`.

## Gotchas

- Models expose the split-learning interface: `get_global_parameters() / set_global_parameters() / get_local_parameters() / set_local_parameters()`. The aggregation surface is defined by `GLOBAL_PARAM_KEYS` and `LOCAL_PARAM_KEYS` frozensets in `strategy.py`.
- `set_local_parameters(strict=False)` does partial loads when the user population grows between rounds.
- `.embedding_cache/` is created at runtime; delete to start fresh.

## Expected (BPR-MF, 10 rounds)

HR@10: 0.70-0.80 (+5-7% vs baseline) | NDCG@10: 0.18-0.28 | Comm: -44%

A win over baseline here is required before claiming anything about the adaptive module.
