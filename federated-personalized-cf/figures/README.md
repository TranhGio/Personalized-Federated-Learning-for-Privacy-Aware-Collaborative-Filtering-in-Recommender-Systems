# figures/

Visualizations produced by `python ../visualize_partitions.py` for this module.

## What's in here

The visualizer illustrates the **Dirichlet cross-silo partitioning** — how MovieLens 1M users are assigned to a small number of clients (`num-supernodes = 5` in the cross-silo legacy mode) based on genre-preference similarity. Under the current thesis setup, this partitioning strategy is **cross-silo legacy** — retained for reproducing pre-migration appendix results, not for the thesis-table runs.

**Under `mode = "benchmark_cross_device"`** (the thesis default) the partitioning is `natural` (1 user = 1 client, N = 6040), so Dirichlet α has no effect and these visualizations do not apply.

## Generated files

| File | What it shows |
|------|---------------|
| `partition_sizes_alpha_*.png` | Ratings / users / movies per client for each α |
| `genre_distribution_alpha_*.png` | Heatmap of per-client genre proportions |
| `rating_distribution_alpha_*.png` | Rating-value (1–5) distribution per client |
| `user_activity_alpha_*.png` | Ratings-per-user histogram per client |
| `alpha_comparison.png` | Side-by-side comparison of α ∈ {0.1, 0.5, 1.0} |
| `partition_summary_alpha_*.csv` | Per-partition statistics (ratings, users, movies, top genres) |

## Regenerating

```bash
python ../visualize_partitions.py        # overwrites everything under figures/
```

## Dirichlet α quick reference (cross-silo legacy)

- α = 0.1 — highly non-IID, one client dominates, some clients may be empty. Not recommended.
- α = 0.5 — recommended for cross-silo experiments; moderate heterogeneity.
- α = 1.0 — less non-IID; nearly balanced but still heterogeneous.

## Notes

- Dirichlet partitioning is methodologically indefensible for a FedRec thesis under the literature convention; see `../../.planning/research/PITFALLS.md` for the full rationale and `../../.planning/PROJECT.md` for the migration decision.
- For cross-device diagnostics (per-user interaction histograms, sparse/medium/dense bucket distribution), see the eval-harness outputs under `../../results/federated/personalized/<run_id>/` once Phase 6 lands.
