# Phase 2 Deferred Items

Scope-boundary discoveries logged during plan execution. Items here are NOT auto-fixed
by the current plan — they belong to other plans' owned files.

## From Plan 04 execution (2026-04-19)

### Test failure in Plan 03 territory

- **File:** `federated-baseline-cf/tests/test_task_rng.py::test_gradient_mask_zeros_non_user_rows`
- **Current status:** FAILED
- **Reason it's out of scope for Plan 04:** Plan 04 owns `server_app.py` + `test_server_integration.py`
  exclusively. The failing test exercises `federated_baseline_cf.task.train_bpr_mf`'s gradient-mask
  contract (D-24) which is owned by Plan 03 (task.py + test_task_rng.py). The parallel execution
  directive in Plan 04 explicitly lists client_app.py and task.py as Plan 03's files.
- **Observed error:**
  ```
  AssertionError: D-24 violation: user_idx=1 row of user_embeddings changed but
  shouldn't have. diff_norm=3.964839e-01
  ```
- **Action:** Plan 03 should verify/fix its own `train_bpr_mf` gradient masking before closing.
  Plan 04 does NOT modify task.py; this failure is pre-existing at the start of Plan 04 execution
  and must be addressed by Plan 03 or a follow-up plan.
