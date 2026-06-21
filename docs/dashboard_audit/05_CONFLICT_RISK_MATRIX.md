# 05 Conflict Risk Matrix

## 结论

准入 gate 建议为 YELLOW。可以建设平台的只读 logger/adapter/schema 层，但不建议立即把 Web 服务同进程深度嵌入长训练，也不建议改环境/算法/reward 核心接口。高风险点集中在：脏工作树、实时 CSV 读写、reward/cost 单位、epoch/episode 概念、多进程日志、CUDA/GPU 资源、现有 Dash dashboard 边界。

## 证据

- `git status --short` 显示工作树存在大量未提交修改和未跟踪文件。
- `src/vpp_dso_sim/dashboard/app.py` 的 `load_dashboard_frames()` 直接 `pd.read_csv()`。
- `src/vpp_dso_sim/visualization/dashboard_data.py` 的 `export_dashboard_frames()` 直接 `frame.to_csv(path)`。
- `examples/08_run_dashboard.py` 默认 `host=127.0.0.1`、`port=8050`、Dash app 一次性加载 frames。
- `examples/17_paper_training_experiment.py` 同步调用 `run_paper_training_experiment()`。
- `paper_training.py` 的 paper-long family 强制 CUDA guard。
- `learning/shared_rollout_workers.py` 使用 subprocess/spawn 共享 rollout worker。
- `pyproject.toml` 只声明 Dash/Plotly 可选 visualization stack，未声明 FastAPI/React/DuckDB。
- 当前 `python3 --help`/pytest collect-only 因 `pandapower` 缺失无法完整执行。

## 相关文件路径

- `examples/08_run_dashboard.py`
- `examples/17_paper_training_experiment.py`
- `src/vpp_dso_sim/dashboard/app.py`
- `src/vpp_dso_sim/visualization/dashboard_data.py`
- `src/vpp_dso_sim/experiments/paper_training.py`
- `src/vpp_dso_sim/envs/gym_env.py`
- `src/vpp_dso_sim/envs/multi_agent_env.py`
- `src/vpp_dso_sim/entities/dso.py`
- `src/vpp_dso_sim/envs/reward_design.py`
- `src/vpp_dso_sim/learning/advanced_marl.py`
- `src/vpp_dso_sim/learning/shared_rollout_workers.py`
- `pyproject.toml`
- `CODEX_PROJECT_CONTEXT.md`
- `AGENTS.md`

## 相关类/函数/变量

- `load_dashboard_frames`
- `export_dashboard_frames`
- `create_dashboard_app`
- `run_paper_training_experiment`
- `_guard_output_protocol`
- `_validate_trainable_cuda_requirement`
- `VPPDSOEnv.step`
- `MultiAgentVPPDSOEnv.step`
- `DSO.calculate_reward_or_cost`
- `vpp_dispatch_reward_components`
- `train_happo`
- `train_hasac`
- `train_matd3`
- `train_hatrpo`
- `happo_shared_rollout_backend`
- `require_cuda_for_trainable`

## 风险矩阵

| 风险 | 等级 | 影响范围 | 触发条件 | 证据 | 建议规避方案 | 是否阻塞平台建设 |
|---|---|---|---|---|---|---|
| 未提交修改被覆盖 | High | 全仓库 | 在脏 worktree 直接改核心训练/环境/算法 | `git status --short` 大量 M/??，核心文件包括 `advanced_marl.py`、`paper_training.py`、`reward_design.py` | 独立分支/worktree；首阶段只新增 docs/logger/adapter | 阻塞大规模开发 |
| 训练主循环冲突 | High | paper training | Web callback 同步调用长训练 | `examples/17` 同步调用；paper-long 672×120×多 seed | 训练 worker/独立进程；dashboard 仅读日志 | 阻塞同进程控制训练 |
| 环境 step/reset 语义冲突 | High | env/trainer/tests | 改 `reset/step` 返回值或 done 语义 | `VPPDSOEnv`/`MultiAgentVPPDSOEnv` 已被多训练器依赖 | adapter 只读；不改 env API | 阻塞直接修改 |
| reward/cost 命名冲突 | High | metrics/plots/paper | 把 `reward`、`total_cost`、VPP profit 混为同一单位 | `DSO.calculate_reward_or_cost`、`reward_design.py` 多版本 reward | long schema，字段含 unit/formula/source | 不阻塞只读接入，阻塞公式展示 |
| epoch/episode/batch 概念冲突 | Medium | dashboard axes | 把 PPO epoch 当 campaign epoch | `advanced_marl.py` update rows 写 `epoch` | schema 区分 `episode_id`、`optimizer_epoch`、`batch_type` | 不阻塞 |
| 数据单位和归一化冲突 | High | frontend/metric | load multiplier 标 MW，actor feature 当物理量 | `profiles.py`、`deep_rl.py`、structured observation scaling | variable dictionary；normalized 标记 | 阻塞高可信物理展示 |
| 并行环境日志冲突 | High | HAPPO shared rollout | 多 worker 同写一个文件或相同 time_index | `shared_rollout_workers.py`，update metrics 有 worker fields | 单 writer 队列；per-worker file + merge；强制 `env_id` | 阻塞多进程实时写 |
| GPU 训练与 Web 服务资源冲突 | Medium | CUDA/CPU | Web 与训练同进程/同 GPU | paper-long `require_cuda_for_trainable=True` | Web 独立进程，CPU only；训练 preflight | 条件阻塞 |
| 日志写入性能冲突 | Medium | training throughput | 每 step 同步写 CSV/DB | long training loss/history 大 | async queue，batch flush，tail/aggregate | 不阻塞低频 hook |
| 多进程写文件冲突 | High | logs/artifacts | worker 进程并发写 JSONL/CSV | shared rollout subprocess | single writer process；atomic rename | 阻塞多 worker live logger |
| 端口冲突 | Low/Medium | local service | 固定 8050 已被占用 | `examples/08` 默认 8050 | 端口探测；配置化；默认 127.0.0.1 | 不阻塞 |
| 包依赖冲突 | Medium | env/CI | 引入 FastAPI/React/DuckDB 到核心 deps | `pyproject.toml` 无这些 deps，已有 `viz` | optional extras `[dashboard]`，独立 frontend | 不阻塞可选模块 |
| 前端构建产物路径冲突 | Medium | outputs/reports | build 写入 `outputs/` 或覆盖 dashboard_data | 无 `package.json`，现有 `outputs/dashboard_data` | `frontend_dist/` 或 package data；不写训练 outputs | 不阻塞 |
| 现有 checkpoint/logging 冲突 | Medium | resume/eval | dashboard 额外文件污染 output dir 或 resume 旧 artifact | `_guard_output_protocol`、`resume_completed` | 每 run 独立 `dashboard_logs/`，manifest/hash | 不阻塞只读 |
| Windows/Linux 路径兼容 | Medium | scripts/frontend | 路径含空格或 Windows `.ps1` | 项目路径 `panda power`，scripts 有 PowerShell | 使用 `Path`，quote shell args | 不阻塞 |
| 隐私边界冲突 | Medium/High | CTDE/data | 前端暴露 private cost 或 critic state | `critic_global_state`、dashboard asset registry includes cost coefficients | public/private schema，默认本地 127.0.0.1，脱敏 | 阻塞远程访问 |
| 已有 Dash dashboard 边界冲突 | Medium | dashboard UX | 新平台覆盖 `src/vpp_dso_sim/dashboard` | 当前 Dash 读取 CSV 静态 frames | 新建 realtime module，兼容现有 app | 不阻塞 |

## 风险

核心风险不是“不能建设 dashboard”，而是不能把 dashboard 当作训练框架重构入口。实时平台必须旁路化、可选化、失败隔离化。

## 建议

- 采用混合模式：训练侧轻量 logger + 可独立启动 dashboard；本地 debug 可自动启动 127.0.0.1 服务。
- 写入采用 JSONL/SQLite/DuckDB/Parquet 中一种稳定 schema，不与现有 `outputs/dashboard_data/*.csv` 抢写。
- 首轮只读 hook 放在 paper training、env.step 返回后、algorithm update metrics 形成后。
- Dashboard 崩溃不影响训练；logger 异常只 warning。
- 不默认引入 Ray/RLlib/PettingZoo/Hydra/Lightning/WandB。

## 待用户确认项

- 是否接受 Gate = YELLOW，即先做最小 logger/adapter/schema，而不是直接做完整 Web 平台。
- 是否需要支持远程访问；若需要，隐私脱敏和鉴权必须前置。
- 是否要求 dashboard 从 UI 启停训练；若要求，必须引入训练 worker/队列边界。
- 是否允许新增 optional dependency group。
