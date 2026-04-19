# Development Environment Setup

## Install order (REQUIRED)

The shared `fedrec-foundation` package MUST be installed before any of the four federated modules. Each module's `pyproject.toml` declares it as a local-path dependency, but editable-install semantics require foundation to exist on disk first.

```bash
# 1. Test framework (one-off)
pip install pytest

# 2. Shared foundation package
pip install -e scripts/foundation/

# 3. The four federated modules (any order, all four)
for m in federated-baseline-cf federated-pfedrec federated-personalized-cf federated-adaptive-personalized-cf; do
  pip install -e "$m"
done

# 4. Smoke test
for m in federated-baseline-cf federated-pfedrec federated-personalized-cf federated-adaptive-personalized-cf; do
  (cd "$m" && python -c "import fedrec_foundation; print(fedrec_foundation.__version__)")
done
```

## Running foundation tests

```bash
cd scripts/foundation && pytest -x tests/         # fail-fast quick run
cd scripts/foundation && pytest tests/ -v         # full suite
```

## Notes

- Plan 01 (Wave 0) creates the `fedrec-foundation` scaffold with real implementations for `paths`, `atomic`, and `hashing`; the rest of the modules (`mapping`, `split`, `exclusion`, `evaluator`, `weight_policy`, `rng`, `mode`, `manifest`) land in Plans 02–05 as their corresponding tests flip from SKIPPED to real assertions.
- Plan 06 adds `fedrec-foundation` as a local-path dependency to each `federated-*-cf/pyproject.toml` so a single `pip install -e <module>/` pulls the foundation automatically. Until then, install foundation explicitly with the command above.
- The env var `FEDREC_FOUNDATION_DATA_DIR` overrides `data_derived()` for CI or remote environments (leaves `ml1m_dir()` untouched).
