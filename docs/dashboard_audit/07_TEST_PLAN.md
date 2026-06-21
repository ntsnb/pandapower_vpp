# 07 Test Plan

## 结论

当前阶段未运行训练，只做静态审计和轻量命令。由于当前 shell 缺 `pandapower`，`--help` 和 `pytest --collect-only` 无法完整通过。后续平台接入测试重点应是：schema 稳定、logger failure isolation、平台接入不改变训练结果、多进程日志不冲突、dashboard 不阻塞训练。

## 证据

- `python3 --version`：Python 3.12.7。
- `python3 examples/17_paper_training_experiment.py --help`：导入阶段失败，`ModuleNotFoundError: No module named 'pandapower'`。
- `python3 examples/08_run_dashboard.py --help`：导入阶段失败，`ModuleNotFoundError: No module named 'pandapower'`。
- `python3 -m pytest --collect-only`：收集到 65 个 tests 后被 43 个 `pandapower` import errors 中断。
- `pyproject.toml`：pytest 配置 `testpaths=["tests"]`、`pythonpath=["src"]`、`addopts="-q -p no:cacheprovider"`。
- `tests/test_dashboard_smoke.py`：已有 dashboard CSV load 和 optional Dash dependency behavior 测试。
- `tests/test_paper_training_experiment.py`：已有 paper preset、checkpoint、CUDA guard、非空目录保护测试。

## 相关文件路径

- `pyproject.toml`
- `tests/test_dashboard_smoke.py`
- `tests/test_paper_training_experiment.py`
- `tests/test_multi_agent_env.py`
- `tests/test_hasac_happo.py`
- `tests/test_hatrpo_training.py`
- `tests/test_matd3_training.py`
- `tests/test_deep_rl_training.py`
- `tests/test_runtime_thread_limits.py`
- `examples/17_paper_training_experiment.py`
- `examples/08_run_dashboard.py`

## 相关类/函数/变量

- `load_dashboard_frames`
- `export_dashboard_frames`
- `run_paper_training_experiment`
- `paper_training_preset`
- `train_happo`
- `train_hasac`
- `train_matd3`
- `train_hatrpo`
- `MultiAgentVPPDSOEnv`
- `VPPDSOEnv`
- `configure_numeric_thread_limits`

## 已执行命令

```text
git branch --show-current
git status --short
git diff --stat
python3 --version
python3 examples/17_paper_training_experiment.py --help
python3 examples/08_run_dashboard.py --help
python3 -m pytest --collect-only
```

结果：

- Git 分支：`main`。
- 工作树：脏。
- `--help`：当前环境缺 `pandapower`，未能显示帮助。
- pytest collect-only：当前环境缺 `pandapower`，未能完整收集。

## 后续 Smoke Test 设计

在含完整依赖的环境中运行：

```text
python3 examples/17_paper_training_experiment.py --help
python3 examples/08_run_dashboard.py --help
python3 -m pytest --collect-only
python3 -m pytest tests/test_dashboard_smoke.py -q
python3 -m pytest tests/test_multi_agent_env.py tests/test_env_smoke.py -q
```

若要跑最小训练 smoke：

```text
python3 examples/17_paper_training_experiment.py \
  --preset smoke \
  --output-dir outputs/dashboard_audit_smoke_<run_id> \
  --no-html \
  --no-tensorboard
```

约束：

- 使用新 output dir。
- 不覆盖已有 `outputs/paper_training*`。
- 不启用 long preset。
- 不下载数据。
- 不启动长时间训练。

## Dashboard Logger 单元测试

建议新增：

- `tests/test_dashboard_logger_schema.py`
  - Metric row 必含 contract fields。
  - nullable `timestamp/date/batch_id` 可通过 validation。
  - `unit/formula_latex/description` 保留。

- `tests/test_dashboard_logger_failure_isolation.py`
  - writer 抛异常时训练 hook 不抛出。
  - queue full 时低优先级 step metrics 可丢弃，高优先级 train/error/checkpoint 保留。

- `tests/test_dashboard_environment_adapter.py`
  - fake env transition 能提取 `time_index`、`vpp_id`、`agent_id`、`reward`、`done`。
  - raw/decoded/projected/landed action 字段不混淆。

- `tests/test_dashboard_algorithm_adapter.py`
  - HAPPO update row 映射 actor/critic/entropy/KL。
  - MATD3/HASAC replay update row 映射 q/actor/critic/alpha。
  - HATRPO trust-region metrics 不依赖 optimizer step。

## 不改变训练结果回归测试

目标：证明 logger 是旁路副作用。

测试方案：

1. 固定 seed、config、horizon=2 或 4、episode=1。
2. 运行 baseline no-op logger。
3. 运行 dashboard logger enabled，但 sink 为 in-memory 或 temp dir。
4. 比较：
   - 初始 actor parameters hash。
   - 同 step action values。
   - reward/cost components。
   - checkpoint summary 中关键 scalar。
5. 允许浮点容差，但不允许随机数序列变化。

## 多进程测试

目标：shared rollout 不写坏日志。

测试方案：

- `happo_shared_rollout_backend=serial` 与 `subprocess` 各跑 horizon=2/fragment=1。
- 每个 worker 写独立 `env_id/worker_id`。
- 合并后 `global_env_step` 无重复冲突，或重复时带不同 `env_id`。
- logger shutdown 后无 zombie writer。

## Dashboard 服务测试

- 默认 host 必须是 `127.0.0.1`。
- 端口占用时给出可读错误或自动换端口。
- dashboard 读取 incomplete run manifest 时显示 running/incomplete，不误报 complete。
- dashboard 读取 failed run manifest 时展示 error，不重放训练。

## 风险

- 当前本地依赖不完整，不能证明现有测试全通过。
- 长训练 smoke 可能消耗 GPU，不应在本阶段运行。
- 如果未来引入 React/Vite，需要额外 Node 测试，但当前仓库无 `package.json`。

## 建议

- 后续先修复/指定 Python 环境，再跑 collect-only。
- 第一版 logger 使用纯 Python stdlib JSONL，可避免新增依赖导致测试环境扩大。
- 对 dashboard optional deps 使用跳过策略：未安装 FastAPI/Dash 时 schema/logger 测试仍应通过。

## 待用户确认项

- 提供或确认含 `pandapower` 的项目解释器路径。
- 是否允许后续执行 smoke preset。
- 是否要求后续引入 Node 前端测试。
- 接入后回归比较以哪些指标作为“训练结果不变”的判据。
