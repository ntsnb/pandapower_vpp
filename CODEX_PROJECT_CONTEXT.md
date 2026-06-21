# Codex Project Context

Updated: 2026-05-11 Asia/Shanghai

## Project Root

- Canonical root: `/mnt/sda/home/niutiansen/panda power/pandapower-vpp-dso-sim`
- Outer folder: `/mnt/sda/home/niutiansen/panda power`
- The outer `pytest.ini` points to `pandapower-vpp-dso-sim/tests` and `pandapower-vpp-dso-sim/src`.
- The project-level `.git` directory is present but invalid/empty after migration, so `git status`, branches, and commit history are unavailable until Git metadata is restored.

## Active Server Environment

- Created Linux virtual environment: `.venv-server`
- Creation mode: `python -m venv --system-site-packages .venv-server`
- Reason: migrated `.venv` is Windows-style (`Scripts/python.exe`) and is not usable as a Linux venv.
- The venv inherits system site-packages to reuse existing `torch 2.7.1+cu128` from the server Anaconda installation.

Install command used:

```bash
./.venv-server/bin/python -m pip install --cache-dir .pip-cache -e '.[dev]'
```

Verified imports:

```text
pandapower 3.4.0
gymnasium 1.3.0
torch 2.7.1+cu128
package /mnt/sda/home/niutiansen/panda power/pandapower-vpp-dso-sim/src/vpp_dso_sim/__init__.py
```

Known environment notes:

- `pandapower` and `gymnasium` were missing from the base Python before setup.
- `.venv-server` intentionally prioritizes project correctness over unrelated inherited packages: `pandapower 3.4.0` requires `packaging~=25.0`, so the venv keeps `packaging 25.0`.
- `python -m pip check` is not clean because inherited system packages have mutually incompatible constraints:
  - `streamlit 1.37.1` requires `packaging<25,>=20`, but `.venv-server` resolves `packaging 25.0`.
  - `s3fs 2024.6.1` requires `fsspec==2024.6.1.*`, but inherited `fsspec` is `2024.2.0`.
- A temporary attempt to satisfy `streamlit/s3fs` broke `pandapower`/`datasets` constraints, so it was reverted. Use a non-`--system-site-packages` conda/venv if a fully clean `pip check` is required.
- Matplotlib/fontconfig may warn if default home cache/config folders are not writable. For tests and demos, use temporary cache variables:

```bash
MPLCONFIGDIR=/tmp/pandapower_vpp_mplconfig XDG_CACHE_HOME=/tmp/pandapower_vpp_cache
```

## Minimal Verification Commands

Minimal pytest subset run successfully:

```bash
MPLCONFIGDIR=/tmp/pandapower_vpp_mplconfig XDG_CACHE_HOME=/tmp/pandapower_vpp_cache \
  ./.venv-server/bin/python -m pytest \
  tests/test_network_build.py \
  tests/test_der_constraints.py \
  tests/test_env_smoke.py
```

Result:

```text
8 passed, 1 warning in 6.35s
```

Additional data/SMART-DS smoke tests also passed:

```bash
MPLCONFIGDIR=/tmp/pandapower_vpp_mplconfig XDG_CACHE_HOME=/tmp/pandapower_vpp_cache \
  ./.venv-server/bin/python -m pytest \
  tests/test_dataset_registry.py \
  tests/test_smart_ds_analysis.py
```

Result:

```text
4 passed in 13.01s
```

Combined minimal suite after final dependency reconciliation:

```bash
MPLCONFIGDIR=/tmp/pandapower_vpp_mplconfig XDG_CACHE_HOME=/tmp/pandapower_vpp_cache \
  ./.venv-server/bin/python -m pytest \
  tests/test_network_build.py \
  tests/test_der_constraints.py \
  tests/test_env_smoke.py \
  tests/test_dataset_registry.py \
  tests/test_smart_ds_analysis.py
```

Result:

```text
12 passed, 1 warning in 15.83s
```

Four-step environment demo run successfully using `configs/ieee33_multi_vpp.yaml` and `VPPDSOEnv(horizon_steps=4)`.

Observed output:

```text
reset_obs_shape=(85,) info_keys=['step']
step=0 reward=0.207 truncated=False total_cost=20.856
step=1 reward=0.053 truncated=False total_cost=23.934
step=2 reward=0.167 truncated=False total_cost=21.658
step=3 reward=0.025 truncated=True total_cost=24.499
```

## Long Experiment Artifact Status

Readable root:

```text
outputs/paper_training_long_current
```

Key files present:

- `experiment_manifest.json`
- `run_index.csv`
- `evaluation_seed_metrics.csv`
- `aggregate_metrics.csv`
- `baseline_comparison.csv`
- `training_episode_metrics.csv`
- `training_loss_metrics.csv`
- `architecture_diagnostics.csv`
- `long_training_report.html`
- `long_training_report_data.json`
- per-run folders under `runs/`

Manifest summary:

- Preset: `paper_long`
- Algorithms: `rule_based`, `no_flex`, `opf_oracle_proxy`, `happo`, `matd3`, `hasac`
- Seeds: `9401`, `9402`, `9403`, `9404`, `9405`
- Train variant: `train_mixed`
- Eval variants: `holdout_peak`, `holdout_cloudy`, `holdout_reverseflow`
- Horizon and eval horizon: `672` steps
- Step length: `0.25` h
- Training episodes: `120`
- Hidden dim: `256`
- Learning rate: `0.0003`
- Batch size: `256`
- Claim boundary: the OPF baseline is an oracle proxy, not a complete AC OPF proof.

Recomputed aggregate checks:

- `evaluation_seed_metrics.csv`: 225 rows
- `aggregate_metrics.csv`: 45 rows
- `run_index.csv`: 285 rows, all `completed`
- Group keys matched exactly between recomputed seed metrics and `aggregate_metrics.csv`.
- Max absolute recomputation differences:
  - `eval_total_reward_mean`: `2.91e-11`
  - `eval_total_cost_mean`: `9.31e-10`
  - `total_violation_cells_mean`: `0`
  - `security_pass_mean`: `0`
  - `horizon_steps_count`: `0`

Algorithm-level evaluation means from `evaluation_seed_metrics.csv`:

| Algorithm | Reward Mean | Cost Mean | Violation Mean | Safety Mean | Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rule_based` | -116070.767806 | 2338215.356113 | 0.000000 | 1.000000 | 15 |
| `no_flex` | -120441.230798 | 2425624.615958 | 0.000000 | 1.000000 | 15 |
| `opf_oracle_proxy` | -122733.622042 | 2471367.107509 | 29.666667 | 0.466667 | 15 |
| `hasac_continuous_dispatch` | -134213.030648 | 2700014.268934 | 31.550000 | 0.666667 | 60 |
| `happo_sequential_ctde` | -135700.587944 | 2746137.699403 | 7.966667 | 0.750000 | 60 |
| `matd3_continuous_dispatch` | -138523.618960 | 2786059.735839 | 16.850000 | 0.750000 | 60 |

Artifact caveats:

- `run_index.csv` and `experiment_manifest.json` contain Windows-style paths such as `outputs\paper_training_long_current` and `C:\Users\admin\Desktop\...`. Relative Linux files exist and are readable, but absolute Windows paths should not be used for server reruns.
- `experiment_progress.csv` is not a strict rectangular CSV: most rows have more fields than the header because progress rows append phase-specific metrics. Use tolerant CSV parsing or `experiment_progress.jsonl` for reliable progress replay.
- `training_loss_metrics.csv` is large, about 1.3 GB. Avoid loading it entirely unless needed; prefer chunked reads.

## Repository Organization Notes

Canonical configs now live under:

- `configs/scenarios/demo/`
- `configs/scenarios/benchmark/`
- `configs/algorithms/dso_sensitivity_attention/v1/`
- `configs/rewards/v2_minimal/`
- `configs/experiments/paper_long/sensitivity_attention_v1/`

Root-level `configs/*.yaml` files are compatibility wrappers so older tests, scripts, docs, and experiment manifests can still use historical paths. New code can also use aliases from `configs/registry.yaml`; `load_yaml()` resolves aliases, canonical paths, legacy wrappers, package resources, and nested `extends` chains.

Inactive output directories were moved under `outputs/_archive/` with a manifest at `outputs/_manifests/output_archive_manifest.csv`. The active paper-long run stays at `outputs/paper_training_long_reward_v2_minimal_20260604_gpu_decoder_bounds_happo_hatrpo_logfix` while the training process is still writing progress files.

Root-level generated files were moved:

- `paper_method_experiment_cn.*` -> `docs/reports/paper_method_experiment_cn/`
- `task_0.md` -> `docs/tasks/reward_v2_minimal_task.md`
- `task2.md` -> `docs/tasks/current_training_diagnosis_task.md`

## Recommended Operating Commands

Activate environment:

```bash
source .venv-server/bin/activate
```

Verify imports:

```bash
python -c "import pandapower, gymnasium, torch, vpp_dso_sim; print(pandapower.__version__, gymnasium.__version__, torch.__version__)"
```

Run minimal tests:

```bash
MPLCONFIGDIR=/tmp/pandapower_vpp_mplconfig XDG_CACHE_HOME=/tmp/pandapower_vpp_cache \
  python -m pytest tests/test_network_build.py tests/test_der_constraints.py tests/test_env_smoke.py
```

Run a short RL environment smoke demo:

```bash
MPLCONFIGDIR=/tmp/pandapower_vpp_mplconfig XDG_CACHE_HOME=/tmp/pandapower_vpp_cache \
  python examples/05_random_rl_env_rollout.py
```

Note: `examples/05_random_rl_env_rollout.py` currently runs 10 steps. For a strict 2-8 step smoke test, instantiate `VPPDSOEnv(..., horizon_steps=4)` directly.

## Immediate Follow-Up Tasks

1. Restore or reinitialize valid Git metadata so future edits can be tracked safely.
2. Decide whether `.venv-server` should remain the standard server venv or be replaced by a conda environment.
3. Consider pinning `pandapower<3.0` if future tests expose API incompatibilities; current minimal smoke tests pass on `pandapower 3.4.0`.
4. Normalize server-safe output paths in future experiment manifests and run indices.
5. Make `experiment_progress.csv` rectangular or rely on `experiment_progress.jsonl`.
6. Add a scripted 4-step smoke command or example if repeated environment validation is expected.
7. Keep archived `outputs/_archive/paper_long/paper_training_long_current` as readable historical evidence, but do not treat it as paper-grade proof that RL beats rule-based baselines.
