# 08 Platform Build Readiness

## 结论

准入 gate 为 YELLOW：可以开始第一阶段低侵入 logger/adapter/schema 建设，但暂时不应直接实现完整实时 Web 平台、UI 启停训练、远程访问或多进程长训控制。

## 证据

- `git status --short`：工作树有 33 个 tracked 修改和大量 untracked 文件。
- `pyproject.toml`：项目已有 optional `viz`/`rl`，未发现 FastAPI/React/DuckDB/Parquet 依赖。
- `examples/17_paper_training_experiment.py`：正式训练入口同步调用 `run_paper_training_experiment()`。
- `src/vpp_dso_sim/experiments/paper_training.py`：`paper_long_sensitivity_v1` 为 672 steps、120 episodes、CUDA required。
- `src/vpp_dso_sim/envs/multi_agent_env.py`：多智能体 reset/step 接口已被训练器依赖。
- `src/vpp_dso_sim/dashboard/app.py`：现有 dashboard 是 Dash 读取 CSV frames。
- `python3 -m pytest --collect-only`：当前 shell 缺 `pandapower`，收集到 65 项后被 43 个 import errors 中断。

## 相关文件路径

- `docs/dashboard_audit/00_REPO_MAP.md`
- `docs/dashboard_audit/01_ENVIRONMENT_ANALYSIS.md`
- `docs/dashboard_audit/02_ALGORITHM_ANALYSIS.md`
- `docs/dashboard_audit/03_TRAINING_LIFECYCLE.md`
- `docs/dashboard_audit/04_DATA_AND_METRIC_SCHEMA.md`
- `docs/dashboard_audit/05_CONFLICT_RISK_MATRIX.md`
- `docs/dashboard_audit/06_INTEGRATION_CONTRACT.md`
- `docs/dashboard_audit/07_TEST_PLAN.md`
- `examples/17_paper_training_experiment.py`
- `src/vpp_dso_sim/experiments/paper_training.py`
- `src/vpp_dso_sim/envs/multi_agent_env.py`
- `src/vpp_dso_sim/learning/advanced_marl.py`
- `src/vpp_dso_sim/dashboard/app.py`

## 相关类/函数/变量

- `PaperTrainingExperimentConfig`
- `paper_training_preset`
- `run_paper_training_experiment`
- `_write_profile_config`
- `_train_algorithm`
- `_checkpoint_choices`
- `MultiAgentVPPDSOEnv`
- `Simulator`
- `DSO.calculate_reward_or_cost`
- `build_role_reward_maps`
- `train_happo`
- `train_hasac`
- `train_matd3`
- `train_hatrpo`
- `load_dashboard_frames`
- `export_dashboard_frames`

## 1. Executive Summary

当前项目是一个基于 pandapower 的 DSO/VPP 多智能体强化学习仿真与实验仓库。核心结构是 `src/vpp_dso_sim`，训练主入口是 `examples/17_paper_training_experiment.py`，环境在 `envs/` 与 `simulation/`，算法在 `learning/`，数据 profile 在 `data/profiles/` 和 paper runner 物化目录，reward/cost 在 `entities/dso.py` 与 `envs/reward_design.py`，loss/update 在各算法训练器中。

是否可以开始平台建设：可以开始最小低侵入 logger/adapter/schema 建设，但不建议立即构建完整实时 Web 控制平台。

Gate 结论：YELLOW。

## 2. Repository Map

详见 `00_REPO_MAP.md`。

核心地图：

```text
configs -> scenario/reward/algorithm config
data -> demo and external profiles
examples/17 -> formal paper campaign CLI
src/vpp_dso_sim/envs -> Gym/multi-agent wrappers
src/vpp_dso_sim/simulation -> scenario/simulator/records/settlement
src/vpp_dso_sim/learning -> HAPPO/HATRPO/MATD3/HASAC/deep RL
src/vpp_dso_sim/dashboard -> existing Dash CSV dashboard
src/vpp_dso_sim/visualization -> CSV/HTML/Plotly exporters
tests -> env/algorithm/dashboard/paper tests
```

## 3. Environment Analysis

详见 `01_ENVIRONMENT_ANALYSIS.md`。

当前环境是 `SimulationScenario + Simulator + env wrapper`。多智能体 agent ids 为 `dso_global_guidance`、`{vpp}_dispatch`、`{vpp}_portfolio`。`reset/step` 接口语义稳定，后续禁止修改。真实时间只有 `time_index/time_hours`，无真实 `date/timestamp`。

## 4. Algorithm & Model Analysis

详见 `02_ALGORITHM_ANALYSIS.md`。

当前主算法是 HAPPO/HATRPO/MATD3/HASAC。CTDE privacy 边界很重要：actor 执行期本地观测，critic 训练期全局/privileged state。MAPPO/QMIX 未发现完整 neural learner/mixer。

## 5. Training Lifecycle Analysis

详见 `03_TRAINING_LIFECYCLE.md`。

训练主循环由 `run_paper_training_experiment()` 总编排，支持 profile materialization、baseline、trainable algorithms、checkpoint、frozen eval、CSV/HTML/TensorBoard 输出。`episode` 是 horizon trajectory；`epoch` 在 HAPPO 中主要是 PPO epoch。

## 6. Data & Metric Schema Analysis

详见 `04_DATA_AND_METRIC_SCHEMA.md`。

数据 profile 是 CSV/pack，`load` 是 multiplier，`pv` 是 factor，`price` 是 currency/MWh proxy 待确认。前端应展示反归一化物理量；encoded actor features 必须标记 normalized。

## 7. Concept Alignment

| 当前代码名字 | 当前实际含义 | Dashboard 建议命名 | 是否需要重命名或注释 |
|---|---|---|---|
| `epoch` | HAPPO PPO update epoch；HATRPO value fitting epoch | `optimizer_epoch`/`ppo_epoch` | 需要注释 |
| `episode` | env reset 到 horizon truncation 的轨迹 | `episode_id` | 不必重命名，但需定义 |
| `batch` | rollout tensor 或 replay minibatch | `batch_id` + `batch_type` | 需要 adapter |
| `rollout` | on-policy episode/fragment | `rollout_id`/`fragment_id` | shared rollout 需补 |
| `gradient_step` | 未统一；算法内 counters 分散 | `gradient_step` | adapter 合成 |
| `global_env_step` | 未统一；可合成 | `global_env_step` | shared rollout 要带 env_id |
| `date` | 未发现 | `date` nullable | 需用户确认 |
| `time_index` | scenario step | `time_index` | 保留 |
| `vpp_id` | VPP 实体 id | `vpp_id` | 保留 |
| `agent_id` | env agent id | `agent_id` | 保留 |
| `policy_id` | actor/policy id，可能共享 | `policy_id` | 必须与 agent_id 区分 |

## 8. Conflict Risk Matrix

详见 `05_CONFLICT_RISK_MATRIX.md`。

最高风险：脏工作树、实时 CSV 读写、多进程日志、reward/cost 单位、env API 语义、GPU/Web 资源、privacy boundary。

## 9. Recommended Integration Architecture

推荐：策略 C，混合模式。

理由：

- 训练侧需要低侵入 logger/hook。
- 长训练和 shared rollout 需要资源隔离。
- 用户希望本地启动服务，可以做可选本地服务进程。
- 现有 Dash dashboard 可继续静态查看 CSV，不应被强行改成实时训练控制器。

接入点：

- `run_paper_training_experiment()` train start/end/error。
- `_write_profile_config()` profile metadata。
- `MultiAgentVPPDSOEnv.step()` 返回后的只读 transition。
- algorithm update metrics 形成后。
- checkpoint 保存后。

日志写入：

- 第一版 JSONL + manifest。
- 后续可选 DuckDB/Parquet。
- 单 writer 队列，批量 flush。

Web 服务：

- 默认独立 CLI 读取日志目录。
- 可选由训练入口启动子进程。
- 默认绑定 `127.0.0.1`。

## 10. Integration Contract

详见 `06_INTEGRATION_CONTRACT.md`。

最小接口包括 `TrainingLifecycleContract`、`MetricSchemaContract`、`EnvironmentAdapterContract`、`AlgorithmAdapterContract`、`VariableDictionaryContract`。所有 hook 都是旁路副作用，不允许改变训练结果。

## 11. Test Plan

详见 `07_TEST_PLAN.md`。

当前验证受限：本 shell 缺 `pandapower`。后续先跑 help、pytest collect-only、dashboard smoke，再做 logger schema/failure isolation/regression tests。

## 12. Files Safe to Add

- `docs/dashboard_audit/`
- `src/vpp_dso_sim/dashboard_realtime/` 或 `src/vpp_dso_sim/realtime_dashboard/`
- `src/vpp_dso_sim/logging/` 或 `src/vpp_dso_sim/training_hooks/`
- `src/vpp_dso_sim/adapters/`
- `tests/test_dashboard_logger_schema.py`
- `tests/test_dashboard_logger_failure_isolation.py`
- `tests/test_dashboard_environment_adapter.py`
- `tests/test_dashboard_algorithm_adapter.py`
- `examples/18_run_realtime_dashboard.py`，若需要独立 CLI
- `docs/dashboard/`

## 13. Files Safe to Modify

仅允许最小修改：

- `pyproject.toml`：增加 optional dependency group，不改核心 dependencies。
- `examples/17_paper_training_experiment.py`：增加 opt-in CLI flags，例如 `--dashboard-log-dir`、`--dashboard-start-local`，默认关闭。
- `src/vpp_dso_sim/experiments/paper_training.py`：在生命周期边界加入 no-op logger hook，默认关闭。
- `src/vpp_dso_sim/learning/advanced_marl.py`、`matd3.py`、`hatrpo.py`：只在已有 update metrics 形成后调用 logger，不改 loss/update。
- `tests/`：新增或局部扩展测试。

## 14. Files Do Not Modify

第一批建设禁止修改或高风险：

- `src/vpp_dso_sim/envs/gym_env.py`
- `src/vpp_dso_sim/envs/multi_agent_env.py` 的 reset/step 语义
- `src/vpp_dso_sim/simulation/simulator.py` 的物理推进逻辑
- `src/vpp_dso_sim/entities/dso.py` 的 reward/cost 数学定义
- `src/vpp_dso_sim/envs/reward_design.py` 的 reward 数学定义
- `src/vpp_dso_sim/learning/ctde_networks.py` 的 network forward
- `src/vpp_dso_sim/learning/advanced_marl.py` 的 optimizer/loss 顺序
- `src/vpp_dso_sim/learning/matd3.py` 的 replay/update 语义
- `src/vpp_dso_sim/learning/hatrpo.py` 的 trust-region update
- 已有 `outputs/`、`runs/`、`checkpoints/`

## 15. User Confirmation Needed

- reward 公式：DSO v2/v3、VPP dispatch、portfolio 是否作为正式论文公式。
- cost 公式：price 单位、operation/loss/procurement 是否统一乘 `dt_hours`。
- loss 拆项：HAPPO/HATRPO/MATD3/HASAC dashboard 展示哪些字段。
- 单位：price 币种、comfort、SOC penalty、constraint penalty 的业务单位。
- episode 周期定义：是否正式定义为 one horizon trajectory。
- epoch 定义：是否接受 dashboard 把 HAPPO `epoch` 命名为 `ppo_epoch`。
- 是否需要并行训练支持：shared rollout/subprocess 是否第一版必须支持。
- 是否需要 demo 数据：若真实 5 周数据不可用，是否先用 synthetic/SMART-DS proxy。
- 是否需要真实数据反归一化：若需要，请提供 timestamp/start date/unit metadata。
- 是否支持 wind：当前未发现 wind schema。
- 是否允许 dashboard 从 UI 启停训练：若允许，需要 worker/queue，不应同进程直接调用长训练。
- 是否需要远程访问：若需要，隐私脱敏和鉴权前置。

## 16. Next Step

Gate 是 YELLOW。

最小修复/准备任务顺序：

1. 确认 Python 环境，能通过 `python3 -m pytest --collect-only`。
2. 确认 `episode/epoch/price unit/date timestamp/5-week dataset` 口径。
3. 新增只读 logger schema 与 variable dictionary，不接 Web。
4. 在 smoke preset 上证明 logger enabled/disabled 不改变训练结果。
5. 再新增独立 dashboard service 读取日志目录。

下一条可交给 Codex 的建设 prompt 建议：

```text
请基于 docs/dashboard_audit/ 的 YELLOW gate 结论，只实现第一阶段低侵入 dashboard logger：
1. 新增可选的 JSONL MetricLogger、MetricSchema、VariableDictionary。
2. 新增 EnvironmentAdapter 和 AlgorithmAdapter 的最小只读实现。
3. 不启动 Web 服务，不改 reward/cost/loss/reset/step/forward/update 语义。
4. 只在 paper_training 生命周期边界加入默认关闭的 hook。
5. 新增 schema/failure-isolation/no-op regression 测试。
6. 所有 dashboard 依赖保持 optional，默认纯训练路径行为不变。
```

## 风险

YELLOW 的含义是可以建设，但必须从 logger/adapter/schema 开始。直接建设“训练启动即实时 Web 平台 + UI 控制训练 + 多进程长训 + 远程访问”会踩中当前多个 High 风险。

## 建议

- 第一阶段不要引入 FastAPI/React/DuckDB，先把日志契约和测试立住。
- 如果用户坚持“训练启动时自动启动 Web 服务”，也应作为 opt-in 子进程读取日志，而不是阻塞训练主线程。
- 后续任何代码接入前先处理脏工作树和测试环境。

## 待用户确认项

同第 15 节。
