---
created: 2026-04-20
source: Phase 3 regression gate
priority: medium
tags: [phase-2, baseline, test-path, determinism, env]
---

# Phase 2 baseline determinism test fails on path mismatch

`federated-baseline-cf/tests/test_server_integration.py::test_selected_partitions_byte_identical_across_subprocess_reruns` asserts result JSONs land at `repo_root/results/federated/` (where `repo_root = movie-recommendation-system/`), but `scripts/run.py baseline benchmark_cross_device` actually writes to `/home/bes/Desktop/vinh/federated-learning/results/federated/` (one dir above repo_root) because baseline `server_app.py` uses a relative `../results/federated/` path consistent with its historical CLAUDE.md.

Surfaced: Phase 3 regression gate (2026-04-20) — test is `@pytest.mark.slow` and was never actually exercised during Phase 2 verification.

Fix options:
1. Update baseline `server_app.py` to write to `repo_root/results/federated/` (breaks historical parity with claude.md examples).
2. Update the test's `results_dir` computation to match the current write location (`parents[3] / "results" / "federated"`).
3. Inject `FEDREC_RESULTS_DIR` env var in the test.

Not a Phase 3 regression — Phase 3 did not touch baseline server_app.py or the results-writing path. Fold into a future `/gsd:plan-phase 2 --gaps` if the slow gate is re-enabled in CI.
