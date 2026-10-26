# Protocol pins, modes, reference runs

## Mode per module

| Module (`scripts/run.py` name) | Mode | Package to import |
|---|---|---|
| `baseline` | `thesis_crossdevice_main` | `federated_baseline_cf` |
| `personalized` | `thesis_crossdevice_main` | `federated_personalized_cf` |
| `adaptive` | `thesis_crossdevice_main` | `federated_adaptive_personalized_cf` |
| `pfedrec` | `paper_compat_pfedrec` (with `fraction-train=0.1`) | `federated_pfedrec` |

`paper_compat_pfedrec` defaults to `fraction-train=1.0` (~6 days); the thesis protocol always overrides it to 0.1.

## Pins for any claim run (all modules)

- `run-seed=42` (extra seeds: 1337, 2026)
- `fraction-train=0.1`, `num-server-rounds=100`
- `checkpoint-rule="best_round_restore"`
- `early-stopping-enabled=false` — ES once stopped a long run at round 34 and made it unusable
- `strategy="fedavg"`
- thesis tags: `thesis-run-label` (`"main"` for credited cells, `"ablation_<dim>=<value>"` otherwise), `ablation-dimension`, `ablation-value`, `wandb-run-name`

Module-specific (from `scripts/thesis/run_thesis_sweep.py` THESIS_BASE_OVERRIDES):

| Module | Extra pins |
|---|---|
| baseline | `embedding-dim=128` for the credited MSE cell (BPR ablation used 64 + `model-type="bpr"`) |
| personalized | `embedding-dim=64 lr=0.005 final-calibration-enabled=false strict-d06-cache=true` |
| adaptive | `model-type="dual" alpha-method="hierarchical_conditional" enable-per-user-alpha=false enable-item-perturbation=false contrastive-lambda=0.0 local-epochs=1 embedding-dim=64 lr=0.005 final-calibration-enabled=true strict-d06-cache=true` |
| pfedrec | `final-calibration-enabled=true final-calibration-freeze-items=true strict-d06-cache=true` (current comparator protocol, run cf72e0) |

`strict-d06-cache=true` makes a cold full-pop eval fail loudly instead of producing a silently wrong number. The baseline module has no per-user state and does not declare it.

## Reference runs (mirror one, change one knob)

| Module | Run id | Full-pop NDCG@10 | Federation | Wrapper / notes |
|---|---|---|---|---|
| pfedrec | `20260726-083121-cf72e0` | 0.3447 | default (`local-simulation`) | `run_pfedrec_calibration.sh` — + final calibration, current comparator bar |
| pfedrec | `20260608-071106-ef41ab` | 0.3352 | `local-sim-gpu` | `run_c3_run1.sh` — no calibration |
| personalized | `20260610-064423-f18e64` | 0.2275 | `local-sim-gpu` | `run_rerun_personalized_warm.sh` — split learning, BPR, dim 64 |
| adaptive | `20260613-152226-869730` | 0.2144 | `local-sim-gpu` | `run_rerun_adaptive_dual_warm_v2.sh` — dual + calibration (contrastive 0.1 was ON) |
| baseline | `20260504-203751-1bf513` | 0.2013 | default (`local-sim-gpu`) | MSE / BasicMF, dim 128 |
| baseline | `20260725-093704-e81518` | 0.1882 | default (`local-sim-gpu`) | BPR, dim 64 — knob-for-knob pair of personalized |

**Federation:** mirror the reference run's backend exactly — copy its `--federation` flag, or omit the flag if the reference omitted it. Pyproject defaults differ per module (pfedrec/personalized default to the CPU `local-simulation`; baseline/adaptive to `local-sim-gpu`), and switching backend is an unrecorded second difference between two runs. (cf72e0 vs ef41ab already differ on this; their in-loop trajectories still matched within 0.0001.)

Read the reference run's `results.json` (`federated_config`, `_manifest.overrides`) **and its wrapper in `.bug-investigation/`** to copy its exact knobs and federation.

## Expected wall-clock on the RTX 5090

pfedrec ~8 h · baseline ~15 h · personalized ~15 h · adaptive ~15 h (+ cache ~14 GB) · reeval: minutes on CPU.

## Wrapper template

```bash
#!/usr/bin/env bash
# <What this run tests, one line.>
# Mirrors <reference run_id> EXCEPT: <the one knob that changes>.
# Pre-flight: <date>, 5/5 passed. Expected wall-clock: ~<h> h.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
exec python3 scripts/run.py <module> <mode> [--federation <as in the reference wrapper>] --run-config '<pins + the change + thesis tags>'
```
