# INVALID — cold-eval artifact

Run `20260527-085237-573e19` (pfedrec) was produced BEFORE the module's run_id eval fix (`01d8b72`). Its D-06 full-population eval omitted `run_id`, so the client read per-user local state from `.embedding_cache/default/` (nonexistent) and scored every user with COLD init.

Poisoned: `final_metrics.best`, `_manifest.metrics.best` (and per-group variants).
Trustworthy: in-loop `final_metrics.last` / `eval_metrics_history` (in-loop eval stamped run_id).

Do NOT cite the best block in any claim table. See EVAL_VALIDITY.json.
