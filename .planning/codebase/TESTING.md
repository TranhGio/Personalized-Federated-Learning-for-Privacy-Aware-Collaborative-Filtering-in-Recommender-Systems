# Testing Patterns

**Analysis Date:** 2026-04-17

Testing in this codebase is **light and manual** — there is no `pytest`/`unittest` framework, no CI test pipeline, and no mocking. What exists are plain `python test_*.py` smoke scripts that exercise dataset loading and model forward passes with live downloads and real tensors. Treat this as a verification scratchpad, not a regression safety net.

## Test Framework

**Runner:**
- **None.** Tests are plain scripts run with `python test_dataset.py` and `python test_models.py`.
- No `pytest` dependency anywhere in the four `pyproject.toml` files (`federated-baseline-cf/pyproject.toml`, `federated-personalized-cf/pyproject.toml`, `federated-pfedrec/pyproject.toml`, `federated-adaptive-personalized-cf/pyproject.toml`). A repo-wide search for `import pytest` / `pytest.` returns zero matches.
- No `unittest`, no `nose`, no `hypothesis`.

**Assertion Library:**
- **None.** Tests "pass" if they do not raise. There are zero `assert` statements in the test files and zero calls to `assertEqual`/`pytest.raises`. Verification is performed by `print()`ing shapes, sample values, and summary statistics which the developer inspects manually.

**Configuration Files:**
- No `pytest.ini`, no `conftest.py`, no `[tool.pytest.ini_options]` section, no `tox.ini`.

**Run Commands:**
```bash
# From the project root (CLAUDE.md > "Commands")
python test_dataset.py
python test_models.py

# Or per-module (tests live inside the Flower app subdirectory)
cd federated-baseline-cf && python test_dataset.py
cd federated-baseline-cf && python test_models.py
cd federated-personalized-cf && python test_dataset.py    # NOTE: broken import — see "Known Test Issues"
cd federated-personalized-cf && python test_models.py     # NOTE: broken import — see "Known Test Issues"

# No coverage command, no watch mode.
```

## Test File Organization

**Location — per-module, at the Flower app root (not in a `tests/` dir):**

| File | Exists | Notes |
|------|--------|-------|
| `federated-baseline-cf/test_dataset.py` | Yes (115 lines) | Canonical, imports work |
| `federated-baseline-cf/test_models.py` | Yes (206 lines) | Canonical, imports work |
| `federated-personalized-cf/test_dataset.py` | Yes (115 lines) | **Byte-identical copy** of baseline version — still imports from `federated_baseline_cf` package |
| `federated-personalized-cf/test_models.py` | Yes (206 lines) | **Byte-identical copy** of baseline version — still imports from `federated_baseline_cf.models` |
| `federated-pfedrec/test_*.py` | **Missing** | No tests for this module |
| `federated-adaptive-personalized-cf/test_*.py` | **Missing** | No tests for the thesis-contribution module |

**Directory layout:**
- No `tests/` directories anywhere in the repo (repo-wide glob for `**/tests/**/*.py` returns no matches).
- Tests sit alongside `pyproject.toml` inside each Flower app directory.

**Naming:**
- Files: `test_<area>.py` where `<area>` is the module being smoke-tested (`dataset`, `models`).
- Functions: `test_<specific_case>()` (e.g., `test_basic_loading`, `test_dirichlet_partitioning`, `test_dataloader`, `test_basic_mf`, `test_bpr_mf`, `test_with_movielens_shape`).

## Test Structure

**Script shape (consistent across all four test files):**

```python
# federated-baseline-cf/test_dataset.py
"""Test script for MovieLens 1M dataset loading and Dirichlet partitioning."""

from federated_baseline_cf.dataset import (
    download_movielens_1m,
    load_movielens_1m,
    dirichlet_partition_users,
    create_global_mappings,
    load_partition_data,
)


def test_basic_loading():
    """Test basic dataset loading."""
    print("=" * 80)
    print("TEST 1: Basic Dataset Loading")
    print("=" * 80)

    data_dir = "./data"
    download_movielens_1m(data_dir)
    ratings_df, movies_df, users_df = load_movielens_1m(data_dir)

    print("\nDataset Info:")
    print(f"  Ratings shape: {ratings_df.shape}")
    ...


if __name__ == "__main__":
    test_basic_loading()
    test_dirichlet_partitioning()
    test_dataloader()
    print("All tests completed successfully!")
```

**Patterns:**
- **Setup/teardown:** None. Each function builds its own inputs inline.
- **Fixtures:** None (no `@pytest.fixture`, no factory modules). Test data is either (a) downloaded live from `https://files.grouplens.org/datasets/movielens/ml-1m.zip` via `download_movielens_1m("./data")` or (b) constructed with `torch.randint` / `torch.rand` directly in the test body.
- **Banner style:** `print("=" * 80)` before and after each test section; checkmarks `✅` at the end of each `test_*()` function (e.g., `federated-baseline-cf/test_models.py:60,138,190`).
- **Invocation:** functions are listed in order inside `if __name__ == "__main__":` — they are NOT auto-discovered; adding a new function requires appending to the main block.
- **Success signal:** completing without an uncaught exception. No explicit pass/fail return.

## What Is Covered

**`test_dataset.py` exercises:**
- `download_movielens_1m(data_dir)` — network download and extraction of ML-1M.
- `load_movielens_1m(data_dir)` — returns `(ratings_df, movies_df, users_df)`; prints shapes and rating distribution.
- `dirichlet_partition_users(ratings_df, movies_df, num_clients=10, alpha={0.1, 0.5, 1.0}, seed=42)` — three alpha values, prints partition sizes and per-client user/movie counts.
- `load_partition_data(partition_id=0, num_partitions=10, alpha=0.5, test_ratio=0.2, batch_size=32, data_dir="./data")` — returns `(trainloader, testloader, num_users, num_items, user2idx, item2idx)`; iterates one batch and prints shapes.

**`test_models.py` exercises:**
- `BasicMF(num_users, num_items, embedding_dim, dropout)` — instantiation, forward pass, `MSELoss`, `.predict()`, `.recommend(user_id=0, top_k=10)`.
- `BPRMF(num_users, num_items, embedding_dim, dropout, use_bias=True)` — instantiation, forward with `(user_ids, pos_items, neg_items)`, `BPRLoss`, `.sample_negatives()` with `user_rated_items` exclusion dict, both single-negative and `num_negatives=4` paths, `.recommend()`.
- Realistic shape sanity: runs once more with `num_users=6040, num_items=3706, embedding_dim=64` (actual ML-1M dimensions) and reports parameter count + MB footprint.

## What Is NOT Covered

No tests exist for any of the following, even though they contain the bulk of the thesis logic:

- **Federated orchestration:** `client_app.py`, `server_app.py` (neither the Flower `@app.train()` / `@app.evaluate()` entrypoints nor the end-to-end round flow).
- **Split-learning plumbing:** `strategy.py`, `SplitFedAvg`, `SplitFedProx`, `GLOBAL_PARAM_KEYS` / `LOCAL_PARAM_KEYS` sets, `get_local_parameters` / `set_local_parameters` shape-mismatch recovery, embedding cache round-trip, prototype EMA aggregation.
- **Adaptive alpha (thesis contribution):** `models/adaptive_alpha.py` — `DataQuantityAlpha`, `MultiFactorAlpha`, `HierarchicalConditionalAlpha`, `create_alpha_computer`, hierarchical conditional rule application (sparse / niche / inconsistent / completionist).
- **Dual-level personalization:** `models/dual_personalized_bpr_mf.py`, `PersonalMLP`, fusion modes (`add`, `gate`, `concat`), `set_alpha()`, `set_global_prototype()`.
- **Next-gen techniques:** per-user learned alpha (`logit_alpha`), item perturbation, contrastive InfoNCE loss.
- **PFedRec baseline:** the entire `federated-pfedrec/` module has zero tests (no `test_dataset.py`, no `test_models.py`).
- **Evaluation:** `evaluation/alpha_analysis.py` (`AlphaAnalyzer`, `AlphaStatistics`, `HierarchicalAlphaAnalyzer`), `evaluation/user_groups.py` (`classify_user_group`, `classify_users_by_group`), ranking metrics (`evaluate_ranking`, `evaluate_ranking_sampled`), per-group NDCG/HR aggregation.
- **Dataclass validation:** `__post_init__` `ValueError` paths in `AlphaConfig`, `HierarchicalConditionalAlphaConfig`, `UserGroupConfig` — no parametric tests confirming the invariants hold.
- **Early stopping:** `EarlyStopping.step()`, patience / mode / min_delta logic.
- **Negative sampling exclusions:** while `BPRMF.sample_negatives` is called, there is no assertion that the returned negatives actually avoid `user_rated_items`.

## Mocking

- **No mocking framework in use.** Repo-wide searches for `unittest.mock`, `from mock`, `MagicMock`, `patch(` return zero matches.
- **What gets mocked in practice:** nothing. Tests hit the real filesystem (`./data/ml-1m/`), make real HTTP downloads the first time they run, and construct real PyTorch tensors on whatever device is default.
- **Isolation:** none — running `test_dataset.py` creates `./data/` in the CWD and leaves it behind; running `test_models.py` depends only on PyTorch but prints random-seeded-at-import results.

**When adding tests, follow the existing pragmatism unless stronger guarantees are needed:**
- Real PyTorch tensors are fine for numerical behavior.
- If a test needs to avoid the 6 MB ML-1M download, construct a tiny synthetic `pd.DataFrame` with the same columns (`user_id`, `movie_id`, `rating`, `timestamp`) and pass it directly to `dirichlet_partition_users` — this is already the easier path.

## Fixtures and Factories

**Test data:**
- Inline, ad-hoc construction inside each test function:
  ```python
  # federated-baseline-cf/test_models.py:34-36
  user_ids = torch.randint(0, num_users, (batch_size,))
  item_ids = torch.randint(0, num_items, (batch_size,))
  ratings = torch.rand(batch_size) * 4 + 1  # Ratings between 1-5

  # federated-baseline-cf/test_models.py:110
  user_rated_items = {0: {1, 5, 10}, 1: {2, 6, 11}}
  ```
- No shared fixture module, no factory_boy / faker.

**Location:** inline in the test function that needs it.

## Coverage

- **No coverage tool is configured** (no `coverage.py`, no `pytest-cov`, no `.coveragerc`).
- **No enforced coverage target.**
- Realistically, the existing scripts cover only `dataset.py` data-loading happy paths and a handful of `models/*.py` forward-pass shapes in two of the four modules. Everything specific to federated learning, split architecture, and the thesis contribution (adaptive alpha, dual-level personalization, prototype aggregation) is unverified by automated checks.

## Test Types

**Unit tests:**
- Loosely: the smoke-script functions are the only thing resembling unit tests. They target single public functions per test but omit boundary / error cases entirely.

**Integration tests:**
- **None.** The closest thing to integration verification is running the full Flower experiment: `flwr run .` with different `--run-config` flags. That is used for empirical validation, not regression testing, and the pass criterion is "metrics look reasonable on W&B", not an automated assertion.

**E2E tests:**
- **None.** No test framework drives a full federated round. Validation of end-to-end behavior is manual via shell sweep scripts in `scripts/` and `federated-<module>-cf/scripts/` (e.g., `scripts/run_baseline_sweep_loo.sh`, `federated-adaptive-personalized-cf/scripts/run_fedprox_sweep.sh`, `federated-adaptive-personalized-cf/scripts/run_ablation.sh`).

## Common Patterns (Current)

**Running a test:**
```bash
cd federated-baseline-cf
python test_dataset.py   # downloads ML-1M to ./data on first run, prints partition tables
python test_models.py    # instantiates BasicMF and BPRMF, prints shapes and param counts
```

**Adding a new test function** (follow the existing style so the file stays runnable as a plain script):
```python
def test_new_case():
    """Test one sentence description."""
    print("=" * 80)
    print("TEST N: <name>")
    print("=" * 80)
    # arrange
    ...
    # act
    ...
    # print shapes / samples for manual inspection
    print(f"  result: {result}")
    print("\n✅ New case test passed!")


if __name__ == "__main__":
    test_basic_loading()
    test_dirichlet_partitioning()
    test_dataloader()
    test_new_case()   # <-- append here; otherwise it will not run
```

## Known Test Issues (Gaps / Bugs)

These are verified issues in the test suite as of 2026-04-17. Treat them as tech debt rather than working tests:

1. **`federated-personalized-cf/test_dataset.py` and `test_models.py` are byte-identical copies of the `federated-baseline-cf` versions.** They still import from `federated_baseline_cf.dataset` and `federated_baseline_cf.models` (lines 3 and 4 of each file) instead of `federated_personalized_cf.*`. As a result they verify the baseline module's behavior, not the personalized module's split-learning API (`get_local_parameters`, `set_local_parameters`, `get_global_parameters`, `set_global_parameters`). If the baseline package is not installed alongside the personalized one, these tests fail at import time.
2. **`federated-pfedrec/` has no tests at all.** The PFedRec calibration baseline — including its unusual dual-learning-rate alternating optimization, per-user `affine_output` cache under `.embedding_cache/partition_{id}/user_{uid}/affine_output.pt`, and BCE loss on binarized implicit feedback — is entirely unchecked by any automated script.
3. **`federated-adaptive-personalized-cf/` has no tests at all.** The thesis-contribution module (hierarchical conditional alpha, dual-level personalization, PersonalMLP fusion, global prototype EMA, per-user learned alpha, item perturbation, contrastive loss) has no `test_*.py` files — none of its novel behavior is exercised by the current test suite.
4. **No CI runs tests.** The only workflow in `.github/workflows/` is `claude.yml` (Anthropic Claude Code PR assistant); there is no workflow that invokes `python test_dataset.py` or anything equivalent. A broken import, a crashed model instantiation, or a changed function signature will not be caught until a developer runs the script manually.
5. **Tests have side effects on the CWD.** `test_dataset.py` downloads and unzips ML-1M into `./data/` relative to the invocation directory. If run from the repo root, that is the same `data/` directory the real experiments use; if run from a subdirectory, it creates a duplicate.

---

*Testing analysis: 2026-04-17*
