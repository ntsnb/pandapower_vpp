# 00 Repository Map

## 结论

当前实际项目根目录是 `/mnt/sda/home/niutiansen/panda power/pandapower-vpp-dso-sim`，不是初始工作目录 `/mnt/sda/home/niutiansen`。项目是 `src/` layout 的 Python 包 `vpp_dso_sim`，强化学习、仿真、DSO/VPP、dashboard、visualization、experiments 都在同一仓库内。

正式训练入口按证据排序：

1. `examples/17_paper_training_experiment.py`：论文实验主入口，调用 `paper_training_preset()` 与 `run_paper_training_experiment()`。
2. `src/vpp_dso_sim/experiments/paper_training.py`：实际训练/评估/报告总编排器。
3. `examples/16_train_happo_hasac.py`：HAPPO/HATRPO/HASAC 单算法入口。
4. `examples/14_train_matd3.py`：MATD3 单算法入口。
5. `examples/10_train_deep_rl.py`：旧 CTDE actor-critic 入口。
6. `scripts/run_reward_v2_matrix.py`：批量包装 `examples/17` 的矩阵脚本。
7. `scripts/run_smoke.py`、`scripts/run_short_train.py`：smoke/sanity 验证入口。

Gate 前安全结论：工作树很脏，后续平台建设必须在独立分支或 worktree 中进行，且不要覆盖未提交训练、reward、环境、测试改动。

## 证据

- `git branch --show-current`：`main`。
- `git status --short`：33 个 tracked 文件已修改，大量 untracked `configs/`、`docs/`、`scripts/`、`src/`、`tests/` 文件。
- `git diff --stat`：33 files changed, 6570 insertions(+), 1368 deletions(-)。
- `pyproject.toml`：项目名 `pandapower-vpp-dso-sim`，`requires-python = ">=3.10"`，包发现 `where = ["src"]`。
- `pyproject.toml`：核心依赖 `pandapower`、`numpy`、`pandas`、`scipy`、`matplotlib`、`pyyaml`、`gymnasium`、`tqdm`。
- `pyproject.toml`：可选依赖 `dev`、`opt`、`viz`、`rl`；`viz` 已有 `plotly`/`dash`，未发现 FastAPI、DuckDB、Parquet、Vite/React。
- `examples/17_paper_training_experiment.py`：CLI presets 包含 `smoke`、`pilot`、`paper_lite`、`paper_long`、`paper_long_sensitivity_v1`。
- `src/vpp_dso_sim/experiments/paper_training.py`：`paper_long_sensitivity_v1` preset 使用 `configs/european_lv_benchmark_v2_sensitivity_attention_v1.yaml`，算法默认含 `happo`，`horizon_steps=672`，`train_episodes=120`，`require_cuda_for_trainable=True`。

## 仓库结构图

```text
pandapower-vpp-dso-sim/
  pyproject.toml
  README.md
  AGENTS.md
  configs/
    registry.yaml
    scenarios/
    algorithms/
    rewards/
    experiments/
  data/
    profiles/
      load_profile.csv
      pv_profile.csv
      price_profile.csv
    external/raw/smart_ds/...
  examples/
    03_timeseries_multi_vpp.py
    08_run_dashboard.py
    10_train_deep_rl.py
    14_train_matd3.py
    16_train_happo_hasac.py
    17_paper_training_experiment.py
  scripts/
    run_smoke.py
    run_short_train.py
    run_reward_v2_matrix.py
    agent_harness.py
  src/vpp_dso_sim/
    dashboard/
    data_sources/
    der/
    dso/
    entities/
    envs/
    experiments/
    learning/
    network/
    optimization/
    simulation/
    utils/
    visualization/
  tests/
```

## 相关文件路径

- `pyproject.toml`
- `README.md`
- `AGENTS.md`
- `configs/registry.yaml`
- `configs/scenarios/`
- `configs/algorithms/`
- `configs/rewards/`
- `data/profiles/*.csv`
- `examples/17_paper_training_experiment.py`
- `examples/16_train_happo_hasac.py`
- `examples/14_train_matd3.py`
- `examples/10_train_deep_rl.py`
- `examples/08_run_dashboard.py`
- `src/vpp_dso_sim/experiments/paper_training.py`
- `src/vpp_dso_sim/dashboard/app.py`
- `tests/test_paper_training_experiment.py`
- `tests/test_dashboard_smoke.py`

## 相关类/函数/变量

- `PaperTrainingExperimentConfig`
- `paper_training_preset`
- `run_paper_training_experiment`
- `_validate_trainable_cuda_requirement`
- `_guard_output_protocol`
- `load_yaml`
- `resolve_config_path`
- `load_dashboard_frames`
- `create_dashboard_app`

## 运行与测试入口

当前环境中未发现 `setup.py`、`setup.cfg`、`requirements.txt`、`environment.yml`、`poetry.lock`、`uv.lock`、`package.json`、`Makefile`、`Dockerfile`、`docker-compose.yml`。包管理事实来源是 `pyproject.toml`。

`python3 examples/17_paper_training_experiment.py --help` 和 `python3 examples/08_run_dashboard.py --help` 在当前 Anaconda Python 3.12.7 下失败，原因是 `ModuleNotFoundError: No module named 'pandapower'`。这说明当前 shell 不是完整训练环境，不说明脚本本身不可运行。

`python3 -m pytest --collect-only` 收集到 65 个测试项后，被 43 个 `pandapower` 导入错误中断。

## 风险

- High：工作树脏且包含训练、环境、reward、算法、测试的核心改动；新平台开发若直接修改这些文件，极易覆盖用户工作。
- Medium：项目路径包含空格 `panda power`，后续前端/脚本/CI 必须使用 `Path` 或参数数组，避免未引用 shell 路径。
- Medium：当前没有 Node 前端工程；如果后续引入 React/Vite，应新建独立可选模块，不应把 build 产物写入训练 `outputs/`。
- Medium：当前 shell 缺核心依赖 `pandapower`，本阶段只能做静态审计和有限 pytest collect-only。

## 建议

- 后续建设使用独立分支或 git worktree。
- 本阶段只新增 `docs/dashboard_audit/`，不改训练、环境、算法、reward。
- dashboard 依赖作为 optional extras，不污染纯训练依赖。
- 新实时平台命名与现有 `src/vpp_dso_sim/dashboard` 分离，例如 `src/vpp_dso_sim/realtime_dashboard` 或 `src/vpp_dso_sim/dashboard_realtime`。
- 保留现有 Dash/CSV dashboard 作为静态报告查看器，不把它改造成训练控制中心。

## 待用户确认项

- 是否接受后续平台建设新开 worktree/分支。
- 当前大量未提交修改是否都属于应保留的用户工作。
- 后续实时平台是否必须与现有 Dash 并存，还是可以逐步替换。
- 是否需要恢复或提供含 `pandapower` 的标准 Python 环境用于后续 smoke test。
