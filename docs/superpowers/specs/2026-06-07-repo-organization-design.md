# Repository Organization Design

## Goal

Organize the migrated research repository so code, configs, documents, and experiment artifacts are grouped by type and version while preserving the old paths that existing scripts, tests, and running experiments still use.

## Current Evidence

- The repository root is `/mnt/sda/home/niutiansen/panda power/pandapower-vpp-dso-sim`.
- The worktree contains many uncommitted and untracked research changes. This reorganization must not revert or discard them.
- `outputs/` is about 50 GB and is ignored by Git.
- PID `715442` is actively writing `outputs/paper_training_long_reward_v2_minimal_20260604_gpu_decoder_bounds_happo_hatrpo_logfix`. That directory must stay in place until the run stops.
- Many tests, scripts, docs, and experiment manifests directly reference `configs/<name>.yaml`. Moving those files without compatibility would break the project.

## Design Decision

Use a compatibility-preserving reorganization:

1. Move real configuration bodies into versioned directories.
2. Keep root-level `configs/<legacy-name>.yaml` files as thin compatibility wrappers.
3. Teach the YAML loader to resolve canonical paths, aliases, and nested `extends` paths.
4. Move root-level generated documents into a documented report directory.
5. Archive only inactive output directories. Active experiment directories stay frozen at their original path.
6. Add manifests and README files so future agents can tell which paths are canonical and which are compatibility aliases.

## Config Layout

Canonical config directories:

```text
configs/
  README.md
  registry.yaml
  scenarios/
    demo/
    benchmark/
  algorithms/
    dso_sensitivity_attention/
      v1/
  rewards/
    v2_minimal/
  experiments/
    paper_long/
      sensitivity_attention_v1/
```

Compatibility files remain at `configs/*.yaml`. Each wrapper should contain only `extends: <canonical-path>` plus a short comment. This keeps existing commands such as:

```text
python examples/17_paper_training_experiment.py --config-path configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml
```

working after the reorganization.

## Config Categories

| Category | Canonical Directory | Files |
|---|---|---|
| Demo scenarios | `configs/scenarios/demo/` | `ieee33_multi_vpp.yaml`, `lv_taiqu_demo.yaml`, `european_lv_mixed_vpp.yaml`, `default_profiles.yaml` |
| Benchmark scenarios | `configs/scenarios/benchmark/` | `european_lv_benchmark_v2.yaml`, `european_lv_benchmark_v2_safety_tight.yaml` |
| DSO sensitivity attention v1 | `configs/algorithms/dso_sensitivity_attention/v1/` | `baseline_rule_v0.yaml`, `happo_legacy_mlp.yaml`, `happo_sensitivity_attention_v1.yaml`, `ablation_no_*.yaml` |
| Reward v2 minimal | `configs/rewards/v2_minimal/` | `reward_v2_minimal.yaml` |
| Paper-long sensitivity v1 experiments | `configs/experiments/paper_long/sensitivity_attention_v1/` | `european_lv_benchmark_v2_sensitivity_attention_v1*.yaml` |

## YAML Loader Requirements

`vpp_dso_sim.utils.config.load_yaml()` must support:

1. Absolute paths.
2. Project-root relative paths.
3. Legacy root config paths.
4. Canonical nested config paths.
5. Aliases from `configs/registry.yaml`.
6. Package resources under `vpp_dso_sim.resources.configs`.
7. Nested `extends` chains without changing merge behavior.

It must not silently invent a file. If a path or alias cannot be resolved, it should raise `FileNotFoundError` with the original requested path and the attempted aliases.

## Output Layout

Keep `outputs/` as the runtime output root because current code and README assume it. Add organization under it for inactive artifacts only:

```text
outputs/
  _archive/
    paper_long/
    smoke/
    tests/
    audits/
    reports/
  _manifests/
    output_archive_manifest.csv
```

Rules:

- Do not move active directory `outputs/paper_training_long_reward_v2_minimal_20260604_gpu_decoder_bounds_happo_hatrpo_logfix` while PID `715442` is running or while files in it update recently.
- Do not move `outputs/dashboard_data`, `outputs/figures`, `outputs/interactive_report.html`, `outputs/rl_architecture.html`, or `outputs/vpp_first_person` during the first pass because dashboard/report code reads these conventional locations.
- Move obvious test outputs such as `outputs/test_*`, `outputs/pytest_tmp_*`, and `outputs/tmp_*` into `_archive/tests/`.
- Move paper-long and reward experiment runs into `_archive/paper_long/` only after they are inactive.
- Move audit/probe/report-only outputs into `_archive/audits/` or `_archive/reports/`.
- Write a manifest row for every moved path: old path, new path, category, reason, timestamp.

## Root-Level Generated Files

Move generated or ad hoc root files out of the project root:

```text
paper_method_experiment_cn.* -> docs/reports/paper_method_experiment_cn/
task_0.md                  -> docs/tasks/reward_v2_minimal_task.md
task2.md                   -> docs/tasks/current_training_diagnosis_task.md
```

Update `.gitignore` for common LaTeX build byproducts so generated `.aux`, `.fls`, `.fdb_latexmk`, `.xdv`, `.synctex.gz`, `.toc`, and `.bbl` files do not pollute `git status`.

## Documentation Updates

Update:

- `README.md`: add a short "Repository layout and path compatibility" section.
- `CODEX_PROJECT_CONTEXT.md`: record canonical config layout and the active output freeze.
- `configs/README.md`: explain categories, wrappers, and registry aliases.
- `outputs/_manifests/README.md`: explain archive manifest fields. This file is under ignored `outputs/`, so it is operational documentation, not source documentation.

## Testing Strategy

Run focused tests first:

```text
python -m pytest tests/test_config_path_registry.py tests/test_structured_happo_training.py tests/test_paper_training_experiment.py -q --basetemp=outputs/pytest_tmp_repo_org -o cache_dir=outputs/pytest_cache_repo_org
```

Then run a minimal scenario load smoke:

```text
./.venv-server/bin/python -c "from vpp_dso_sim.simulation.scenario import load_scenario; names=['configs/ieee33_multi_vpp.yaml','configs/scenarios/demo/ieee33_multi_vpp.yaml','happo_sensitivity_attention_v1']; print([len(load_scenario(n).vpps) for n in names])"
```

Finally run the project-level lightweight test set used by this repository:

```text
python -m pytest tests/test_env_smoke.py tests/test_timeseries_smoke.py tests/test_config_path_registry.py -q --basetemp=outputs/pytest_tmp_repo_org_core -o cache_dir=outputs/pytest_cache_repo_org_core
```

## Non-Goals

- Do not change research semantics, reward formulas, algorithm behavior, or power-system modeling.
- Do not move active training outputs.
- Do not delete experiment artifacts.
- Do not rewrite all historical docs to use only canonical paths; historical evidence should remain interpretable.
- Do not commit automatically while the worktree contains broad unrelated research changes.

## Acceptance Criteria

- Root-level config wrappers still let existing tests and scripts use old `configs/<name>.yaml` paths.
- Canonical nested config paths load the same scenario content as the old paths.
- Alias names in `configs/registry.yaml` load correctly.
- `extends` still works for nested canonical configs.
- Root-level generated files are removed from project root and placed in documented directories.
- Inactive output directories are archived with a manifest, while the active paper-long directory stays in place.
- README and project context explain the new layout.
- Focused tests and scenario-load smoke commands pass.
