# pandapower-vpp-dso-sim

## MARL VPP Dashboard / 多智能体 VPP 实验平台

本仓库包含本地实时训练可视化平台 `marl_dashboard`。默认只绑定 `127.0.0.1`，用于展示 run、epoch、episode、date、time_index、VPP、dataset、reward/cost/loss 拆项、变量字典、公式、pandapower 拓扑和 VPP 配置。

快速 demo：

```bash
marl-dashboard demo --data-dir runs --host 127.0.0.1 --port 8765
```

真实训练接入使用 `ExperimentLogger` 和 `start_dashboard`，训练侧在 `env.step` 后记录 dataset/reward/cost，在 learner update 后记录 loss，在 episode/epoch 结束后记录 scalar/event。完整启动方式、logger API、Parquet/DuckDB schema、epoch/episode/batch/trajectory/gradient_step/global_env_step 概念说明见 [`src/marl_dashboard/README.md`](src/marl_dashboard/README.md)。

## 2026-05-03 Hierarchical MARL Final-Shape Update / 分层 MARL 最终形态更新

本轮将默认 HAPPO 路线调整为更贴近研究设定的异质分层结构，而不是共享 VPP actor 的工程原型：

- `DSO global agent`：独立连续 actor，基于 DSO 可见的全局电网状态、VPP 报量/报价、FR/DOE 边界和安全状态生成全局引导动作。
- `VPP dispatch agents`：每个 VPP 一个独立连续 dispatch actor，用各自本地观测进行聚合目标与 DER 级解聚合动作决策；默认不共享参数，避免把不同 VPP 误建模为同质合作体。
- `VPP portfolio agents`：每个 VPP 一个独立离散 portfolio actor，按 `portfolio_decision_interval_steps` 慢周期更新组合配置建议；非决策步强制 `keep`，防止每 15 分钟频繁改变 VPP 资产组合。
- `Centralized value critic`：仅训练期可见，输出 `dso_global_guidance + each <vpp_id>_dispatch + each <vpp_id>_portfolio` 多头 value baseline，用于 HAPPO 顺序更新与 importance correction。
- 参数共享保留为显式消融项：`share_vpp_dispatch_parameters=True` 或 `share_vpp_portfolio_parameters=True` 时才启用，不再是默认主模型。
- `MATD3` 与 `HASAC` 作为连续调度 baseline 也同步调整为默认每个 VPP 独立 dispatch actor；共享 VPP dispatch actor 只作为 `share_vpp_dispatch_parameters=True` 消融项。
- 新增 `src/vpp_dso_sim/learning/reward_contracts.py`，显式记录 `r_dso / r_dispatch_i / r_portfolio_i` 的奖励边界：dispatch 不接收 raw DSO global reward，portfolio 接收的是局部化 DSO-alignment/service 信号而不是直接共享全局奖励。
- 新增 `src/vpp_dso_sim/envs/heterogeneous_multi_agent_env.py` 作为异质多智能体环境入口别名，避免论文/文档里的 `heterogeneous_multi_agent_env` 与实际 `multi_agent_env` 命名脱节。

可运行示例：

```powershell
python examples\16_train_happo_hasac.py --algorithm happo --episodes 1 --horizon-steps 3 --hidden-dim 16 --output-dir outputs\happo_final_shape
```

输出中应看到 `happo_training_summary.json/csv`，其中 `per_vpp_dispatch_actors=true`、`per_vpp_portfolio_actors=true`、`portfolio_agent_timescale=slow_loop`。

## 2026-05-02 Advanced MARL Update / 高级 MARL 算法更新

本轮把算法与可视化同步推进到更清晰的三路线结构：

- `MATD3`: `src/vpp_dso_sim/learning/matd3.py` 已从“DSO critic + 平均 VPP dispatch critic”升级为 **role multi-head twin Q**。第 0 个 Q head 对应 `dso_global_guidance`，后续每个 head 分别对应一个 `<vpp_id>_dispatch`，因此 VPP 的 general-sum/self-interested reward 不再被先求均值后训练。
- `HAPPO`: `src/vpp_dso_sim/learning/advanced_marl.py` 新增可运行 HAPPO research trainer，包含 `sequential_role_update`、`importance_correction`、PPO clipped surrogate 与固定 role order。2026-05-03 起，默认 HAPPO 进一步升级为 DSO 独立 actor、每个 VPP 独立 dispatch actor、每个 VPP 独立慢周期 portfolio actor。
- `HASAC`: `src/vpp_dso_sim/learning/advanced_marl.py` 新增可运行 HASAC 连续调度 trainer，包含 squashed Gaussian soft actor、entropy temperature、twin soft Q、soft Bellman backup 和 off-policy replay buffer。慢周期 portfolio action 仍保持在 HASAC 之外。
- UI 同步：`outputs/rl_architecture.html` 和 `outputs/dashboard_data/rl_algorithm_capabilities.csv` 已展示 MATD3/HAPPO/HASAC 的机制、critic head 和实现边界。

相关测试：

```powershell
python -m pytest tests\test_matd3_training.py tests\test_happo_hasac_trainers.py -q
python -m pytest tests\test_advanced_marl.py tests\test_hasac_happo.py -q
python -m pytest tests\test_dashboard_smoke.py tests\test_visualization_data.py -q
```

## 2026-05-02 Real-Training Preparation / 真实训练准备更新

本轮更新不再停留在玩具 profiles。项目已经新增公开大数据 registry、可恢复并发下载脚本，并已把第一批真实配电网大数据下载到本地：

```powershell
python scripts\download_open_datasets.py --dataset smart_ds_aus_p1u_base_opendss --output-root data\external\raw --workers 16
```

已下载数据：

- 数据集：NREL SMART-DS Austin `AUS/P1U/base_timeseries/opendss`
- 本地目录：`data/external/raw/smart_ds/v1.0/2018/AUS/P1U/base_timeseries/opendss`
- 文件数：2369
- 大小：127.059 MiB
- 失败文件：0
- 审计文件：`data/external/raw/smart_ds_aus_p1u_base_opendss_download_manifest.csv`、`data/external/raw/smart_ds_aus_p1u_base_opendss_download_summary.json`

新增文件：

- `scripts/download_open_datasets.py`：公开数据集下载器，当前支持 OEDI/NREL SMART-DS OpenDSS 子集，可恢复、可并发、会生成 manifest/summary。
- `src/vpp_dso_sim/data_sources/registry.py`：真实训练候选数据源 registry，覆盖 SMART-DS、SimBench、IEEE PES feeders、NREL End-Use Load Profiles、ACN-Data、Pecan Street、Ausgrid、Low Carbon London、OpenEI、CAISO、ENTSO-E。
- `docs/dataset_landscape.md`：数据集适配矩阵、下载记录、后续 adapter 计划。
- `docs/real_training_readiness_audit.md`：真实训练前的阻塞项、最低训练协议和下一步工程计划。

历史兼容训练器也同步升级：`privacy_separated_ctde_actor_critic` 保持原算法 ID 以兼容已有 benchmark/report，但训练规则已从简单 actor-critic 升级为 **MAPPO/HAPPO-lite**：

- centralized critic 仍只在训练期可见，但现在输出 `V_dso`、`V_dispatch`、`V_portfolio` 三个角色级 value head。
- actor loss 使用 GAE-lambda advantage 和 PPO clipped surrogate。
- loss/summary 输出新增 `gae_lambda`、`ppo_clip_ratio`、`dso_value_loss`、`dispatch_value_loss`、`portfolio_value_loss`、`policy_update_rule=mappo_happo_lite_gae_single_epoch_clipped_surrogate`。

边界说明：这一段描述的是历史兼容训练器，不是当前推荐的最终形态 HAPPO 路线。当前 `advanced_marl.py` 已提供 HAPPO 顺序更新、importance correction、每 VPP 独立调度 actor 与慢周期 portfolio actor；`matd3.py` 已提供连续调度 MATD3 核心；HASAC 已提供连续调度 soft actor-critic 核心。正式论文级训练前仍需阅读 `docs/real_training_readiness_audit.md`，因为长周期、多 seed、真实数据 split、OPF/oracle baseline 仍是训练证据的关键。

## 2026-05-02 MATD3 完整连续调度核心

项目新增了独立 MATD3 路线，不再把 MATD3 只作为候选算法名字：

```powershell
python examples\14_train_matd3.py --episodes 1 --horizon-steps 6 --batch-size 2 --warmup-steps 2 --hidden-dim 16 --output-dir outputs\matd3_smoke --eval
```

新增实现：

- `src/vpp_dso_sim/learning/matd3.py`
- `examples/14_train_matd3.py`
- `tests/test_matd3_training.py`

当前 MATD3 覆盖连续动作子问题：

- DSO envelope preference
- VPP aggregate dispatch point
- VPP 内部 DER 归一化连续解聚合动作

已实现 MATD3 核心机制：

- replay buffer
- centralized twin Q critics
- target actor / target critic networks
- target policy smoothing
- delayed actor update
- frozen deterministic evaluation

边界说明：`<vpp_id>_portfolio` 是慢周期离散组合配置动作，不强行塞进 TD3。它仍应由 HAPPO/MAPPO 类随机策略或后续组合优化器训练。

## 2026-05-02 SMART-DS 数据分析结果

已对下载的 NREL SMART-DS OpenDSS 数据进行结构化扫描：

```powershell
python examples\15_analyze_smart_ds_dataset.py --output-dir outputs\smart_ds_analysis
```

输出：

- `outputs/smart_ds_analysis/smart_ds_summary.json`
- `outputs/smart_ds_analysis/smart_ds_feeders.csv`
- `outputs/smart_ds_analysis/smart_ds_suites.csv`
- `outputs/smart_ds_analysis/smart_ds_dataset_report.md`

关键统计：

- 文件数：2369
- DSS 文件：917
- 年度 15 分钟 profile CSV：3021 个，约 1914.933 MiB
- OpenDSS LoadShapes 引用的唯一 profile：2912 个
- 本地已解析到的 profile 引用：62704 / 62704，缺失 0
- 主馈线目录：25
- 配变/低压组合目录：93
- load 定义约 127138 条
- line 定义约 156683 条
- transformer 定义约 25350 条

推荐形成三套训练数据组合：

- `smart_ds_full_feeder_ctde`：以 25 个主馈线为拓扑迁移/泛化任务，训练 DSO 全局引导和 VPP dispatch。
- `smart_ds_lv_portfolio_suite`：以 93 个配变/低压目录作为 VPP 组合配置候选，用于慢周期 portfolio agent。
- `hybrid_der_market_suite`：SMART-DS 拓扑 + NREL EULP 建筑负荷/HVAC + ACN-Data EVCS + OpenEI/CAISO 价格，用于论文级收益、服务和安全联合评估。

## 2026-05-02 Reward / MARL 架构更新

当前 MARL 目标已经从“所有 agent 共享一个全局 reward”的合作式原型，调整为更符合 DSO-VPP 机制的 **role-specific general-sum reward**：

- `dso_global_guidance`：使用 `r_dso`，关注电网安全、采购/代理运行成本、目标跟踪和动作投影惩罚。
- `<vpp_id>_dispatch`：使用 `r_dispatch_i`，关注该 VPP 自身的能量收益、灵活性服务收益、可用性收益、DER 成本、履约跟踪、SOC/舒适度和投影惩罚。它不直接接收原始 DSO 全局 reward。
- `<vpp_id>_portfolio`：使用 `r_portfolio_i`，关注长周期利润代理、履约可靠性、配置切换成本、履约风险，以及“局部化 DSO 对齐收益”。这里的 DSO 项是可解释的激励/结算代理，不是把原始全局 reward 直接共享给 VPP。

对应实现位置：

- `src/vpp_dso_sim/envs/reward_design.py`：角色级 reward 公式。
- `src/vpp_dso_sim/envs/multi_agent_env.py`：`step()` 返回每个 agent 的不同 reward，并在 `info[agent]["agent_reward_components"]` 中记录分量。
- `src/vpp_dso_sim/learning/deep_rl.py`：训练循环使用 DSO、VPP dispatch、VPP portfolio 的分离回报更新对应 loss。
- `outputs/interactive_report.html`、`outputs/rl_architecture.html`、`outputs/vpp_first_person/index.html`：重新生成后会展示新的 reward 结构。

## 2026-05-02 Advanced MARL 候选算法筛选

为回应“至少同步尝试 20 种以上思路”的需求，项目新增了一个轻量级算法筛选层。它不会把代理评分伪装成正式训练收敛结果，而是先把当前 DSO/VPP 隐私边界、连续动作、异构智能体和 general-sum reward 约束转成可审计的候选算法评分表，用于决定下一步应该优先实现哪些深度 MARL baseline。

新增实现：

- `src/vpp_dso_sim/learning/advanced_marl.py`：高级 MARL 候选库与 MATD3/TD3 风格 twin critic 结构规格。
- `src/vpp_dso_sim/experiments/algorithm_search.py`：至少 20 个候选思路的代理筛选实验，输出保留/暂缓原因。
- `src/vpp_dso_sim/visualization/algorithm_search_report.py`：中英双语动态 HTML 报告，展示 top candidates、拒绝原因、MATD3/HAPPO/HASAC/FACMAC 适配说明。
- `examples/12_algorithm_search.py`：一键运行候选算法筛选。

运行方式：

```powershell
python examples/12_algorithm_search.py --output-dir outputs/algorithm_search --top-k 5 --min-candidates 20
python -m vpp_dso_sim.visualization.algorithm_search_report --output-dir outputs/algorithm_search
```

输出文件：

- `outputs/algorithm_search/candidate_scores.csv`
- `outputs/algorithm_search/summary.json`
- `outputs/algorithm_search/algorithm_search_report.html`

当前代理筛选的保留逻辑优先考虑：

- DSO/VPP 执行期隐私分离：actor 只能读本地观测或公开报量/包络。
- 连续控制适配：DSO envelope、VPP dispatch、DER 动作 head 都更接近连续动作。
- 异构智能体适配：DSO 全局引导、VPP dispatch、VPP portfolio 不应被强行塞进同一个共享 MLP。
- general-sum reward：VPP 不再是纯合作共享 reward，而是局部收益最大化，同时通过 DSO 对齐项和安全约束被引导。
- 工程风险：短预算下优先保留可落地、可验证、可逐步替换现有 CTDE trainer 的算法。

默认情况下，`examples/12_algorithm_search.py` 会优先读取 `learning/advanced_marl.py` 中的高级候选注册库。只有注册库候选数量不足 `--min-candidates` 时，才会用内置 curated search space 补齐，避免“补充候选”覆盖正式注册库候选。当前默认注册库包含 25 个候选，最近一次代理筛选保留的近期优先实现候选为 `matd3`、`mappo`、`maddpg`、`mappo_gnn_critic`、`happo`。

注意：这一步是 `metadata/proxy algorithm search`，不是 20 个完整深度强化学习训练实验。正式论文级比较仍需要后续对 MAPPO/HAPPO/MATD3/HASAC 等候选按相同场景、相同 seed、相同训练预算、相同 holdout split 和 oracle/OPF baseline 做真实训练与统计检验。

## 2026-05-02 Deep RL 候选训练 Campaign

项目新增了深度强化学习候选训练编排入口：

```powershell
python examples/13_deep_rl_candidate_campaign.py --preset pilot_1d --output-dir outputs/deep_rl_candidate_campaign_plan --plan-only
```

该脚本会读取 `learning/advanced_marl.py` 中的全部候选算法，输出每个候选的训练状态：

- `true_implemented`：当前工程已有该算法的真实专用训练器。
- `ctde_adapter_training`：可以运行真实 PyTorch 隐私分离 CTDE 训练闭环，但这不是该候选算法的完整专用实现。例如 `matd3` 行如果是这个状态，表示它只用了 CTDE adapter 训练，不能宣称已经实现完整 MATD3。
- `not_yet_implemented`：候选算法仍只在 registry/设计层存在，尚未实现对应 update rule。

输出文件：

- `outputs/deep_rl_candidate_campaign_*/candidate_training_plan.csv`
- `outputs/deep_rl_candidate_campaign_*/candidate_training_results.csv`
- `outputs/deep_rl_candidate_campaign_*/campaign_summary.json`
- `outputs/deep_rl_candidate_campaign_*/deep_rl_candidate_campaign.html`

训练预算说明：

- `pilot_1d`：工程连通性检查，不用于论文结论。常见设置是 `1-2 episodes × 16-96 steps`。
- `long_7d`：建议的研究起点，自动使用至少 `100 episodes × 672 steps × 5 seeds`，即 15 分钟步长下的 7 天训练窗口。
- `long_14d`：更强预算，至少 `100 episodes × 1344 steps × 5 seeds`。

示例：

```powershell
python examples/13_deep_rl_candidate_campaign.py --preset long_7d --train-top-k 3 --output-dir outputs/deep_rl_candidate_campaign_long_7d
```

当前默认经济结算参数是保守现实近似，不以“让 reward/profit 变正”为目标。`private_profit_proxy` 可能为负，这通常表示吸收/充电服务、DER 成本、EV/HVAC/flexible load 成本和服务补偿之间尚未形成正现金流。正式分析必须分开报告：

- DSO 侧：`total_cost`、`dso_reward`、越限、投影、采购代理成本。
- VPP 侧：`private_profit_proxy`、`positive_private_profit_step_rate`、`profit_by_vpp`、服务支付、DER 成本。
- 训练侧：episode reward、frozen eval reward、projection gap、violation count、收益分布。

## Benchmark HTML 同步机制（2026-05-01 更新）

现在项目明确区分两类静态 HTML：

- 通用仿真报告：`outputs/interactive_report.html`、`outputs/rl_architecture.html`、`outputs/vpp_first_person/*.html`，由 `examples/07_interactive_report.py` 或 `examples/10_train_deep_rl.py` 刷新，适合查看一次普通 rollout 的拓扑、潮流、VPP 第一视角和 RL 架构。
- Benchmark 专用报告：`outputs/<benchmark_dir>/benchmark_report.html`、`outputs/<benchmark_dir>/interactive_report.html`、`outputs/<benchmark_dir>/rl_architecture.html`、`outputs/<benchmark_dir>/vpp_first_person/index.html`，由 `examples/11_run_benchmark_experiment.py` 默认自动刷新，直接读取该 benchmark 目录下的 `seed_metrics.csv`、`aggregate_metrics.csv`、`profile_quality.csv`、每个 run 的 `step_summary.csv` 和 CTDE `frozen_eval_summary.csv`。

运行 Benchmark V2.1 时推荐使用：

```powershell
python examples/11_run_benchmark_experiment.py --output-dir outputs/benchmark_v21_pilot --horizon-steps 48 --seeds 5201 --train-variants train_mixed --eval-variants holdout_peak,holdout_cloudy,holdout_reverseflow --algorithms rule_based,privacy_separated_ctde_actor_critic --ctde-train-episodes 2 --ctde-train-horizon-steps 48 --ctde-eval-horizon-steps 48
```

运行结束后优先打开：

- `outputs/benchmark_v21_pilot/benchmark_report.html`
- `outputs/benchmark_v21_pilot/interactive_report.html`
- `outputs/benchmark_v21_pilot/rl_architecture.html`
- `outputs/benchmark_v21_pilot/vpp_first_person/index.html`

这些页面会展示 `holdout_reverseflow`、`safety_tight_limits`、`privacy_preserving_proxy`、`policy_evaluation_mode=frozen_deterministic_mean_policy`、`projection_gap_mw`、`fr_binding_rate`、`reverse_flow_step_rate` 等 Benchmark V2.1 字段。对应的 Dash/HTML 数据表写入：

```text
outputs/<benchmark_dir>/dashboard_data/
  benchmark_seed_metrics.csv
  benchmark_aggregate_metrics.csv
  benchmark_profile_quality.csv
  benchmark_run_index.csv
  benchmark_focus_step_summary.csv
  model_update_summary.csv
```

如果只想产出 benchmark CSV 而不刷新 HTML，可以使用：

```powershell
python examples/11_run_benchmark_experiment.py --skip-report-refresh
```

如果还想额外刷新根目录的通用 rollout 页面，可以显式添加：

```powershell
python examples/11_run_benchmark_experiment.py --refresh-root-report
```

协作规则：以后只要算法、训练协议、reward、agent 架构或 benchmark 指标发生变化，必须同步更新 `visualization/` 中对应 HTML 生成器，并增加测试断言，不能只更新 Python 训练代码。

`pandapower-vpp-dso-sim` 是一个面向科研迭代的配电网 DSO-VPP 协同运行仿真平台。项目用 `pandapower` 建模馈线/台区网络，用 Python 逻辑层建模 DSO、多个 VPP 聚合商、PV、微型燃机、储能、柔性负荷、HVAC、EV/EVCS 等资源，并预留深度强化学习、分层强化学习、多智能体强化学习和 LLM 辅助调度接口。

当前版本重点不是一次性实现复杂市场出清或三相不平衡，而是形成一个可以运行、可以测试、可以可视化、可以继续扩展的工程骨架。

## 当前实现状态

已实现的核心能力：

- IEEE 33 风格配电馈线和简化低压台区网络构建。
- 多 VPP 场景配置，每个 VPP 可管理单 PCC 或多节点 DER 组合。
- DER 到 pandapower 元件的映射：`load`、`sgen`、`storage` 等。
- 72 小时默认时序仿真，15 分钟间隔，共 288 个 time steps；测试和示例可使用短 horizon 快速验证。
- 结果输出：电压、线路负载、变压器负载、VPP 出力、DER 调度、SOC、HVAC 温度、约束违规、reward 分量。
- 静态拓扑图、Plotly 离线交互 HTML、Dash 只读仪表盘数据层。
- VPP 第一视角报告：按 step 展示“看到什么、推断什么、收到什么包络/指令、做出什么 DER 调度、结果如何”。
- Gymnasium 风格环境 `VPPDSOEnv` 与多智能体环境 `MultiAgentVPPDSOEnv`。
- PyTorch Actor-Critic 深度强化学习训练闭环。

最新算法口径：

- DSO 不直接控制所有 DER。DSO 根据 VPP 日前报量/报价、当前网络状态、线路/电压压力和价格上下文，为每个 VPP 生成 operating envelope、preferred target、service request。
- VPP dispatch agent 已不再只是 `normalized_setpoint_bias`。当前训练路径支持输出：
  - `selected_p_mw`：VPP 在 DSO 包络内选择的聚合运行点。
  - `der_actions`：面向 PV/ESS/EVCS/HVAC/柔性负荷/MT 等 DER 的归一化动作。
- 环境在写入 pandapower 前执行 FR/DOE 投影、DER 边界裁剪、聚合残差修复和安全校核。
- 当前默认训练器是 `privacy_separated_ctde_actor_critic`：DSO actor、VPP 本地调度 actor、VPP 慢周期组合配置 actor 与 centralized critic 已经拆成独立神经模块。`shared_actor_critic_benchmark` 不参与默认实验，只在显式传入 `--algorithm shared` 时作为消融/回归基线使用。
- 当前 reward 是整形后的控制目标，不是市场净利润：

```text
reward = -0.05 * total_cost + feasibility_bonus + tracking_bonus
raw_objective_reward = -total_cost
```

`raw_objective_reward` 保留用于诊断，避免 reward shaping 掩盖真实成本。

`total_cost` 已包含 `action_projection_penalty`。如果 DSO/VPP/RL 给出的原始目标超过 DSO 包络或 VPP/DER 当前可行域，即使安全投影把动作修回可执行范围，模型也会因为投影修正量付出学习成本。

### 顶会规格 CTDE 隐私边界

用户需要研究的是 DSO 与多个 VPP 在隐私边界下的协同控制，因此当前项目把“共享骨干基线”和“隐私分离 CTDE 主训练器”明确分开：

- 共享骨干 benchmark：`shared_actor_critic_benchmark` 把 DSO 观测编码成一个共享 latent，再由 DSO head、VPP 聚合 head、DER 解聚合 head、组合配置 head 和 value head 共同读取。它是历史中间产物和对照基线，不参与当前默认实验；只有运行 `examples/10_train_deep_rl.py --algorithm shared` 时才会启用。
- 隐私分离 CTDE 主训练器：`privacy_separated_ctde_actor_critic` 已实现为可运行 PyTorch 训练闭环。DSO actor、VPP dispatch actor、VPP portfolio actor 和 centralized critic 是分离神经模块。示例脚本 `examples/10_train_deep_rl.py` 默认运行该训练器。
- 执行期隐私边界：DSO actor 只读取电网拓扑、安全状态、市场上下文和 VPP 上报的报量/报价/可行域摘要；VPP actor 只读取自身 DER 状态、成本、预测、SOC/舒适度、历史履约和 DSO 下发的运行包络；VPP 之间不共享原始私有观测。
- 训练期 centralized critic：只有训练期 critic 可以在仿真可信边界内读取 `critic_global_state`，用于估计 `V(s)`/`Q(s,a)` 和 advantage。执行期 DSO/VPP actor 不读取 `critic_global_state`。
- 同质 VPP 参数共享：允许同类型 VPP actor 共享网络参数以提高样本效率，但每个 VPP 的输入仍是本地 observation，不把其他 VPP 私有 DER 成本或状态拼入执行 actor。
- 安全投影不是 RL agent：`raw action -> DER/device bounds -> FR/DOE projection -> residual repair -> pandapower write -> runpp check` 是确定性安全层，不参与 actor 身份定义；投影残差可以作为 penalty 反馈给学习器。

相关 UI 已同步：

- `outputs/rl_architecture.html` 与 `outputs/interactive_report.html` 会把 `Current Implemented Privacy-Separated CTDE Neural Network Architecture / 当前已实现的隐私分离 CTDE 神经网络架构` 作为主图展示。该图用不同泳道展示 DSO 全局引导、VPP 本地调度、VPP 慢周期组合配置、训练期 centralized critic 和非 RL 安全投影。
- 原来的共享 MLP 图不再出现在默认 CTDE 报告主页面中；只有在显式运行 shared benchmark 后才作为基线结构展示。
- `outputs/dashboard_data/rl_target_ctde_architecture.csv` 会记录目标 CTDE 组件、隐私范围、执行可见性、损失信号和论文角色，便于后续算法更新时 UI 自动跟随。

### 当前 RL/MARL 边界说明

- `VPP dispatch agent` 是快周期调度/解聚合智能体：输入本 VPP 的 envelope、DER 状态和服务信号，输出 `selected_p_mw` 与 `der_actions`。
- `VPP portfolio agent` 是慢周期组合配置智能体：当前已经进入 PyTorch policy-gradient loss，动作是 `keep`、`reweight`、`propose_membership_change`。为了保证物理模型安全，它现在只学习商业组合配置提议；DER 的物理母线和 pandapower 元件行不会被神经网络动作直接移动，只有受控的 `portfolio_events` 才会改变商业归属。
- `Safety projection` 不是 MARL 智能体，也不是学习策略。它是确定性安全层，执行链路为：`raw action -> device bounds -> FR/DOE clip -> DER residual repair -> pandapower write -> runpp check`。
- `Training supervisor` 也不是环境内的 MARL agent，更不是 LLM agent。它是实验级编排/监督模块，负责跟踪 trial、reward 趋势、超参数和收敛状态；它不在 `MultiAgentVPPDSOEnv.step()` 内接收 observation/action/reward。

`outputs/rl_architecture.html` 和 `outputs/interactive_report.html` 的论文总图智能体卡片已经补充“结果来源/计算过程”审计字段。点击任意智能体可以看到：

- `Uses RL?`：说明该模块是否真的使用强化学习策略头，还是确定性投影/实验监督模块。
- `Output formula`：说明该智能体输出如何从观测编码、策略分布、采样动作或规则约束计算得到。
- `Result calculation`：用自然语言解释“页面上这个结果到底是怎么来的”，避免把神经网络动作、启发式包络和安全投影混在一起。
- `Result source`：列出结果来自 `deep_rl.py` 训练轨迹、`Simulator` 基线、dashboard CSV，还是实验监督器输出。
- `Training signal`：说明该模块是否进入 policy-gradient / value loss / entropy loss。
- `Audit outputs`：列出可追溯的 CSV，例如 `dso_operating_envelope.csv`、`vpp_rl_disaggregation.csv`、`projection_trace.csv`、`deep_rl_trajectory.csv`。
- `Non-RL guardrails`：列出不参与学习但必须保证物理可行性的边界、SOC/舒适度约束、FR/DOE 投影和 `pandapower runpp` 校核。

当前默认深度 RL 神经网络已经不是共享骨干，而是隐私分离 CTDE actor-critic。它仍然不是最终论文级的 GNN/Transformer/MAPPO 实现，但已经完成两项关键升级：VPP dispatch actor 使用 DER token 的 Deep Sets 编码器，不再把补零后的 DER 列表直接交给 flat MLP；centralized critic 使用 `critic_global_state + joint_action_summary`，不再只看状态而忽略当前联合动作。

```text
DSO actor:
  o_dso in R^(5 + 7*N_vpp)
    -> LayerNorm -> Linear(D_dso,64) -> Tanh -> Linear(64,64) -> Tanh
    -> z_dso in R^64
    -> mean=tanh(Linear(64,N_vpp)) + trainable dso_log_std[N_vpp]

VPP dispatch actor, per VPP:
  o_vpp_i in R^(16 + 15*Kmax)
    -> split into context R^16 and Kmax DER tokens, each R^15
    -> context LayerNorm(16) -> Linear(16,64) -> Tanh -> Linear(64,64) -> Tanh
    -> shared DER token MLP: LayerNorm(15) -> Linear(15,64) -> Tanh -> Linear(64,64) -> Tanh
    -> masked mean pooling + masked max pooling + token-count ratio
    -> fusion Linear(64*3+1,64) -> Tanh -> Linear(64,64) -> Tanh
    -> z_vpp_i in R^64
    -> aggregate_mean=tanh(Linear(64,1)) + der_mean=tanh(Linear(64,Kmax))
    -> trainable log_std in R^1 and R^Kmax

VPP portfolio actor, per VPP:
  h_vpp_i in R^9
    -> LayerNorm -> Linear(9,64) -> Tanh -> Linear(64,64) -> Tanh
    -> z_portfolio_i in R^64
    -> Linear(64,3) -> Categorical(keep, reweight, propose_membership_change)

Centralized critic, training only:
  critic_global_state in R^(5 + 9*N_vpp)
  joint_action_summary in R^(16 + 8*N_vpp)
    -> state encoder:  LayerNorm(D_critic) -> Linear(D_critic,64) -> Tanh -> Linear(64,64) -> Tanh
    -> action encoder: LayerNorm(D_action) -> Linear(D_action,64) -> Tanh -> Linear(64,64) -> Tanh
    -> fusion Linear(128,64) -> Tanh -> Linear(64,1) = V(s, a_summary)
```

其中 `N_vpp` 是场景中的 VPP 数量，`Kmax` 是单个 VPP 最大 DER 数量。当前默认 demo 中 `N_vpp=6`、`Kmax=6`，所以 `o_dso` 是 47 维，`o_vpp_i` 是 106 维，`critic_global_state` 是 59 维，`joint_action_summary` 是 64 维。`outputs/rl_architecture.html` 和 `outputs/interactive_report.html` 会展示逐层神经网络图、CTDE nodes/edges/feedback 表，以及各 actor 的 reward/loss 回传路径。

## 快速开始

进入项目目录：

```powershell
cd "C:\Users\admin\Desktop\panda power\pandapower-vpp-dso-sim"
```

推荐创建独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

安装基础包：

```powershell
pip install -e .
```

安装测试、可视化和强化学习依赖：

```powershell
pip install -e ".[dev,viz,rl]"
```

如果本机 `pandapower` 导入很慢，可以临时关闭 numba JIT：

```powershell
$env:NUMBA_DISABLE_JIT = "1"
```

## 运行测试

在项目根目录运行：

```powershell
python -m pytest
```

项目的 `pyproject.toml` 已配置：

- `testpaths = ["tests"]`，只收集 `tests/` 下的测试。
- `addopts = "-q -p no:cacheprovider"`，避免 pytest 在项目中生成 `.pytest_cache`。

如果你之前从父目录 `C:\Users\admin\Desktop\panda power` 运行过 pytest，可能会扫描到 `outputs/` 或临时虚拟环境并报权限错误。请改为进入项目目录后运行：

```powershell
cd "C:\Users\admin\Desktop\panda power\pandapower-vpp-dso-sim"
python -m pytest
```

常用局部测试：

```powershell
python -m pytest tests/test_network_build.py
python -m pytest tests/test_timeseries_smoke.py
python -m pytest tests/test_multi_agent_env.py tests/test_deep_rl_training.py
python -m pytest tests/test_visualization_data.py
```

## 常用示例

构建网络并运行一次潮流：

```powershell
python examples/01_build_network.py
```

运行静态潮流 demo：

```powershell
python examples/02_static_powerflow.py
```

运行多 VPP 时序仿真并生成 `outputs/`：

```powershell
python examples/03_timeseries_multi_vpp.py
```

运行日前/实时协同 demo：

```powershell
python examples/04_day_ahead_realtime_demo.py
```

运行 Gymnasium 环境随机 rollout：

```powershell
python examples/05_random_rl_env_rollout.py
```

生成交互式 HTML 报告：

```powershell
python examples/07_interactive_report.py
```

检查 Dash dashboard 依赖和数据：

```powershell
python examples/08_run_dashboard.py --check
```

启动 Dash dashboard：

```powershell
python examples/08_run_dashboard.py
```

运行深度 RL 小规模训练：

```powershell
python examples/10_train_deep_rl.py --episodes 2 --horizon-steps 4 --output-dir outputs/deep_rl
```

默认情况下，`examples/10_train_deep_rl.py` 在训练结束后会同步刷新所有可视化文件：

- `outputs/interactive_report.html`
- `outputs/rl_architecture.html`
- `outputs/vpp_first_person/*.html`
- `outputs/dashboard_data/model_update_summary.csv`
- `outputs/dashboard_data/rl_*.csv`

如果只想训练、不想重新生成 HTML，可以添加：

```powershell
python examples/10_train_deep_rl.py --episodes 2 --horizon-steps 4 --output-dir outputs/deep_rl --skip-report-refresh
```

## Benchmark V2 二次实验

当前新增的研究级候选实验入口是：

```powershell
python examples/11_run_benchmark_experiment.py
```

默认会运行：

- 主场景：`configs/european_lv_benchmark_v2.yaml`
- 安全裕度场景：`configs/european_lv_benchmark_v2_safety_tight.yaml`
- 拓扑 holdout：`configs/ieee33_multi_vpp.yaml`
- `288` 个 15 分钟步长，也就是 3 天。
- `5` 个 seed：`3101,3102,3103,3104,3105`
- split：`train_profile`、`eval_profile`、`safety_tight_limits`、`topology_holdout`
- 默认可执行算法：`rule_based` 基线。
- 可选训练算法：`privacy_separated_ctde_actor_critic`，运行方式为 train split 训练 checkpoint，然后 frozen eval 到 eval/safety split。

本轮已经执行过的正式候选命令：

```powershell
python examples/11_run_benchmark_experiment.py --output-dir outputs/benchmark_v2_research_candidate
```

输出包括：

- `outputs/benchmark_v2_research_candidate/seed_metrics.csv`
- `outputs/benchmark_v2_research_candidate/aggregate_metrics.csv`
- `outputs/benchmark_v2_research_candidate/profile_quality.csv`
- `outputs/benchmark_v2_research_candidate/experiment_manifest.json`
- `outputs/benchmark_v2_research_candidate/benchmark_report.html`

该实验不是 smoke test。它包含多 seed、288 step、非重复多日 profile、同场景评估、安全裕度评估和 IEEE33 拓扑 sanity holdout。
v2.0 结果：`min_voltage≈0.9338 pu`、`max_line_loading≈92.04%`、`security_pass_rate=1.000`。

v2.1 pilot 已新增：

- `holdout_reverseflow`：低负荷、高 PV、PV 容量放大的反向潮流 split。
- `safety_tight_limits`：`[0.95, 1.05]` 电压约束和 95% 线路/变压器限值。
- `privacy_preserving_proxy` reward：默认训练/评估不再直接用 VPP 私有真实运行成本作为主 reward。
- `step_summary.csv` 和扩展 `seed_metrics.csv`：包含 near-voltage rate、near-line rate、projection gap、FR binding、service-request count、reverse-flow rate 等字段。

可运行一个短预算 rule-based + CTDE pilot：

```powershell
python examples/11_run_benchmark_experiment.py --output-dir outputs/benchmark_v21_pilot --horizon-steps 48 --seeds 5201 --train-variants train_mixed --eval-variants holdout_peak,holdout_cloudy,holdout_reverseflow --algorithms rule_based,privacy_separated_ctde_actor_critic --ctde-train-episodes 2 --ctde-train-horizon-steps 48 --ctde-eval-horizon-steps 48
```

注意：这仍然是 `research-grade candidate benchmark`，不是论文最终证据。原因是当前 profile 仍可能降级为合成数据，CTDE 已接入 train-then-frozen-eval 协议但默认训练预算仍偏小，MAPPO/IPPO 还没有按同一协议接入，且仍缺少严格 AC OPF/MILP 最优参考和真实市场结算。详细设计见 `docs/benchmark_v2_experiment_plan.md`。

## Paper Training 长周期实验入口

2026-05-03 起，项目新增论文级训练编排入口：

```powershell
python examples/17_paper_training_experiment.py --preset smoke
```

该入口会统一完成：

- SMART-DS Austin 年度 15 分钟 profile 接入：优先读取 `data/external/raw/smart_ds/v1.0/2018/AUS/P1U/profiles` 下的 `res_kw/com_kw/AUS_*` 曲线；若本地数据缺失，会显式降级到 synthetic benchmark pack，并在 `profile_metadata.json` 中记录。
- 真实 split 形态：训练使用 `train_mixed`，冻结评估使用 `holdout_peak/holdout_cloudy/holdout_reverseflow` 等独立 profile variant。
- 多 seed 与超参 case：`base/lower_lr/higher_entropy/larger_network` 可作为调参网格。
- Baseline：默认使用 `rule_based`、`no_flex`、`ac_validated_search_reference`。其中 `ac_validated_search_reference` 是有限候选集 + AC 潮流校验的安全参考，不是穷举 AC OPF/MILP，也不能写成最优上界。旧 `opf_oracle_proxy` 仅保留为兼容别名，不推荐用于新实验。
- Trainable MARL：`happo`、`hatrpo`、`matd3`、`hasac` 都采用 train split 训练 checkpoint，再用 frozen actor 在 eval split 评估。
- Checkpoint 对照：可通过 `--checkpoint-selection final|train_best|both` 指定冻结评估使用最终 checkpoint、训练集最佳 checkpoint，或两者都评估。`paper_long` 默认使用 `both`，并在 run id、汇总表和 baseline comparison 中写入 `checkpoint_label`，避免 final 与 train_best 混在一起。
- AC 安全证书汇总：`evaluation_seed_metrics.csv` 会记录 post-AC 安全通过率、潮流收敛率、AC certificate safe rate、backoff/rollback rate、accepted alpha 和 certified projection gap，避免只用 projection gap 代替真实 AC 安全。
- TensorBoard：同步写入 `outputs/<run>/tensorboard/<run_id>/events.*`，并导出 PNG 图像。
- 静态 HTML：生成 `outputs/<run>/long_training_report.html`，双击即可查看训练曲线、冻结评估、baseline 对比、收敛摘要、论文声明门禁、架构诊断和产物索引，不需要启动服务。

快速 smoke 验证：

```powershell
python examples/17_paper_training_experiment.py --preset smoke --output-dir outputs/paper_training_smoke
```

研究 pilot：

```powershell
python examples/17_paper_training_experiment.py --preset pilot --algorithms rule_based,no_flex,ac_validated_search_reference,happo,hatrpo,matd3,hasac
```

论文级长周期训练建议从以下命令开始，运行时间会显著增加。请使用全新的输出目录：

```powershell
python examples/17_paper_training_experiment.py --preset paper_long --output-dir outputs/paper_training_long_20260512_fresh
```

`ac_validated_search_reference` 每步会跑多个候选调度的 AC 潮流校验。默认 paper-long 使用有限候选预算；若要提高参考质量、并接受更长运行时间，可以显式设置：

```powershell
python examples/17_paper_training_experiment.py --preset paper_long --output-dir outputs/paper_training_long_current --ac-reference-max-candidates 24
```

如果要显式说明同一次 paper-long 中同时比较最终 checkpoint 和训练集最佳 checkpoint：

```powershell
python examples/17_paper_training_experiment.py --preset paper_long --output-dir outputs/paper_training_long_fresh --checkpoint-selection both --progress-interval-seconds 60
```

当前 DSO `sensitivity_attention_v1` + 结构化 HAPPO 主线请优先使用新的 paper-long preset：

```powershell
python examples/17_paper_training_experiment.py --preset paper_long_sensitivity_v1 --output-dir outputs/paper_training_long_sensitivity_v1_20260528 --progress-interval-seconds 60
```

正式长周期前建议先跑 2-step preflight：

```powershell
python examples/17_paper_training_experiment.py --preset paper_long_sensitivity_v1 --output-dir outputs/paper_training_long_sensitivity_v1_preflight_smoke --seeds 9401 --horizon-steps 2 --eval-horizon-steps 2 --train-episodes 1 --hparam-cases base --algorithms rule_based,no_flex,happo --checkpoint-selection final --no-html --no-tensorboard --progress-interval-seconds 60
```

改造过程、实验过程、reward/loss/KL/entropy/grad norm 曲线口径和产物位置记录在 `docs/experiments/paper_long_sensitivity_v1_protocol.md`。

不要直接复用旧的 `outputs/paper_training_long_current` 结果作为论文指标，尤其是其中仍包含旧 `opf_oracle_proxy` 或旧诊断结果时。新协议会在发现旧 manifest 含 legacy oracle proxy 时拒绝继续复用；`paper_long` 在未开启 `--resume-completed` 时也会拒绝写入非空输出目录，避免 TensorBoard/event/csv 混入旧实验。

长周期训练默认不会刷屏。若当前终端支持交互刷新并已安装 `tqdm`，脚本会自动显示两层动态进度条：

- `Campaign`：完整实验 campaign 总进度，右侧用 `B/T/E` 汇总 baseline、training checkpoint 和 frozen eval 的完成数量。
- `HAPPO` / `HATRPO` / `MATD3` / `HASAC`：当前正在训练的算法 episode 进度，右侧刷新 reward、cost、violations、projection gap、replay/update 计数。

Baseline、RL train、Frozen eval 的分阶段彩色进度条保留在 `live_progress.html`，避免终端同时出现过多条形图。

如果终端不是交互式环境，脚本会自动退回到低频汇总日志。若希望每 60 秒汇总一次：

```powershell
python examples/17_paper_training_experiment.py --preset paper_long --output-dir outputs/paper_training_long_current --progress-interval-seconds 60
```

实时查看进度：

- 打开 `outputs/paper_training_long_current/live_progress.html`：显示总体进度、baseline/RL/eval 分阶段彩色进度条和最近指标，页面每 30 秒自动刷新。
- PowerShell 监控 JSONL：

```powershell
Get-Content outputs\paper_training_long_current\experiment_progress.jsonl -Tail 20 -Wait
```

- PowerShell 监控 CSV：

```powershell
Get-Content outputs\paper_training_long_current\experiment_progress.csv -Tail 20 -Wait
```

如果需要恢复逐事件刷屏调试，可以加：

```powershell
python examples/17_paper_training_experiment.py --preset paper_long --output-dir outputs/paper_training_long_current --verbose-progress
```

核心输出：

- `outputs/<run>/experiment_manifest.json`
- `outputs/<run>/live_progress.html`
- `outputs/<run>/experiment_progress.jsonl`
- `outputs/<run>/experiment_progress.csv`
- `outputs/<run>/run_index.csv`
- `outputs/<run>/training_episode_metrics.csv`
- `outputs/<run>/training_loss_metrics.csv`
- `outputs/<run>/evaluation_seed_metrics.csv`
- `outputs/<run>/aggregate_metrics.csv`
- `outputs/<run>/baseline_comparison.csv`
- `outputs/<run>/convergence_summary.csv`
- `outputs/<run>/architecture_diagnostics.csv`
- `outputs/<run>/claim_guardrails.csv`
- `outputs/<run>/claim_readiness.json`
- `outputs/<run>/tensorboard/`
- `outputs/<run>/tensorboard_images/`
- `outputs/<run>/long_training_report.html`

结论边界：`smoke` 和 `pilot` 只证明流水线可运行，不能作为论文结论。论文表格至少应使用 `paper_long` 或更大预算，并补充真实价格/结算数据、严格 AC OPF 或 MILP 最优参考、显著性检验和隐私泄漏审计。当前 `ac_validated_search_reference` 只能作为 AC 可行性参考，不能作为最优性上界。

## 输出文件

主要输出目录为 `outputs/`。

时序结果：

- `outputs/bus_voltage.csv`
- `outputs/line_loading.csv`
- `outputs/trafo_loading.csv`
- `outputs/vpp_power.csv`
- `outputs/der_dispatch.csv`
- `outputs/storage_soc.csv`
- `outputs/evcs_soc.csv`
- `outputs/hvac_temperature.csv`
- `outputs/constraint_violations.csv`
- `outputs/reward_components.csv`，其中包含 `reward`、`raw_objective_reward`、`feasibility_bonus`、`tracking_bonus` 和各类 penalty/cost 分量
- `outputs/summary.json`

可视化与 dashboard 数据：

- `outputs/interactive_report.html`
- `outputs/rl_architecture.html`
- `outputs/vpp_first_person/index.html`
- `outputs/vpp_first_person/<vpp_id>.html`
- `outputs/vpp_first_person/long_cycle.html`
- `outputs/vpp_first_person/economic_explanation.html`
- `outputs/algorithm_search/algorithm_search_report.html`
- `outputs/deep_rl_candidate_campaign_*/deep_rl_candidate_campaign.html`
- `outputs/dashboard_data/*.csv`
- `outputs/figures/*.png`

可视化同步机制：

- `outputs/dashboard_data/model_update_summary.csv` 是所有 UI 页面共享的算法更新摘要。
- `outputs/interactive_report.html`、`outputs/rl_architecture.html`、`outputs/vpp_first_person/*.html` 和 Dash 面板都会读取同一组 dashboard CSV。
- 算法结构更新后，运行 `python examples/10_train_deep_rl.py --episodes 2 --horizon-steps 4 --output-dir outputs/deep_rl` 会训练并刷新全部静态 HTML。
- 如果只运行训练而不刷新报告，使用 `--skip-report-refresh`。

深度 RL 训练产物：

- `outputs/deep_rl/actor_critic_checkpoint.pt`
- `outputs/deep_rl/deep_rl_training_summary.csv`
- `outputs/deep_rl/deep_rl_training_summary.json`
- `outputs/deep_rl/deep_rl_episode_metrics.csv`
- `outputs/deep_rl/deep_rl_step_metrics.csv`
- `outputs/deep_rl/deep_rl_loss_metrics.csv`
- `outputs/deep_rl/deep_rl_trajectory.csv`

与当前算法结构直接相关的 dashboard 数据：

- `outputs/dashboard_data/rl_algorithm_overview.csv`
- `outputs/dashboard_data/rl_agent_groups.csv`
- `outputs/dashboard_data/rl_agent_architecture.csv`
- `outputs/dashboard_data/rl_agent_relationships.csv`
- `outputs/dashboard_data/rl_step_workflow.csv`
- `outputs/dashboard_data/rl_reward_design.csv`
- `outputs/dashboard_data/rl_loss_components.csv`
- `outputs/dashboard_data/rl_ctde_assessment.csv`
- `outputs/dashboard_data/rl_implementation_gaps.csv`
- `outputs/dashboard_data/model_update_summary.csv`
- `outputs/dashboard_data/vpp_day_ahead_bid.csv`
- `outputs/dashboard_data/dso_operating_envelope.csv`
- `outputs/dashboard_data/vpp_rl_disaggregation.csv`

## 仓库布局与路径兼容

配置文件已经按用途和版本整理到规范目录：

- `configs/scenarios/demo/`：demo、smoke 和小型场景。
- `configs/scenarios/benchmark/`：benchmark 主场景和 holdout/safety-tight 变体。
- `configs/algorithms/dso_sensitivity_attention/v1/`：DSO sensitivity-attention v1、legacy MLP、rule baseline 和 ablation 配置。
- `configs/rewards/v2_minimal/`：可复用 reward 配置片段。
- `configs/experiments/paper_long/sensitivity_attention_v1/`：paper-long sensitivity-v1 实验配置和 reward 变体。

根目录下的 `configs/*.yaml` 仍然保留为兼容 wrapper，旧命令和旧实验 manifest 可以继续使用原路径。新脚本优先使用规范路径或 `configs/registry.yaml` 中的 alias，例如 `happo_sensitivity_attention_v1`、`reward_v2_minimal`、`paper_long_sensitivity_v1_reward_v2_minimal`。

已结束或临时的输出目录归档到 `outputs/_archive/`，迁移记录在 `outputs/_manifests/output_archive_manifest.csv`。当前仍在运行的 paper-long 目录会保持在 `outputs/` 根层，避免破坏训练写入。`outputs/dashboard_data/`、`outputs/figures/`、`outputs/interactive_report.html`、`outputs/rl_architecture.html` 和 `outputs/vpp_first_person/` 仍作为可视化固定入口保留。

## 项目结构

```text
pandapower-vpp-dso-sim/
  configs/                 YAML 兼容入口、registry 和规范化配置目录
  agents/                  项目级 subagent 角色、重叠整合和 ppvpp-* 可注册草案
  data/profiles/           负荷、PV、电价等曲线
  docs/                    架构、符号约定、建模假设、实验审查、路线图、报告和任务记录
  examples/                可运行示例脚本
  memory/                  长期规则、决策、经验、实验和用户偏好沉淀
  outputs/                 仿真、训练、可视化输出；已结束产物位于 outputs/_archive/
  src/vpp_dso_sim/
    dashboard/             Dash 只读仪表盘
    der/                   DER 物理/逻辑模型
    entities/              DSO、VPP、PCC、市场等逻辑主体
    envs/                  Gymnasium 与多智能体环境
    learning/              agent 角色、深度 RL、训练监督、架构数据源
    network/               pandapower 网络构建、潮流、约束、灵敏度
    optimization/          灵活性聚合、解聚合、安全投影、基线策略
    simulation/            场景加载、时序仿真、日志和 profile 管理
    visualization/         静态图、交互式 HTML、第一视角报告
    utils/                 配置、IO、随机数、单位工具
  tests/                   pytest 测试
```

`agents/` 是本项目的 subagent 复用层：它把全局已注册但职责重叠的 agent
收敛为 `ppvpp-architect`、`ppvpp-domain-guardian`、`ppvpp-agent-engineer`、
`ppvpp-experiment-critic`、`ppvpp-memory-keeper` 等项目角色。以后复杂任务应先读
`agents/project_agent_registry.yaml`，再决定实际调用哪个全局 backing role。

`memory/user_preferences.md` 记录用户长期偏好和协作规则，包括：算法更新后必须同步
HTML/UI、网络/算法设计后必须进行架构与实验审查、短训练只能算 smoke 不能算论文证据。

## pandapower 符号约定

项目内部采用：

- DER 内部有功 `P > 0` 表示向电网注入。
- DER 内部有功 `P < 0` 表示从电网吸收。

pandapower 元件约定：

- `load.p_mw > 0` 表示消耗有功。
- `sgen.p_mw > 0` 表示发电注入。
- `storage.p_mw > 0` 表示充电，`storage.p_mw < 0` 表示放电。

因此储能必须做符号转换：

```text
internal P > 0 discharge  -> pandapower storage.p_mw < 0
internal P < 0 charge     -> pandapower storage.p_mw > 0
```

储能 SOC 不会由 pandapower 自动更新，项目在每个 time step 后由 `StorageModel.update_soc()` 自行更新。

更完整说明见：

- `docs/sign_conventions.md`
- `tests/test_sign_conventions.py`

## DSO-VPP-RL 算法框架

一个仿真 step 的数据流：

1. VPP 根据本地 DER 状态、PV 预测、SOC、负荷和成本，上报聚合可行域与报价摘要。
2. DSO 读取全局网络状态、VPP 报告、线路/电压压力和价格上下文。
3. DSO 生成每个 VPP 的 operating envelope、preferred target、service request。
4. VPP dispatch agent 读取本 VPP 包络和 DER 状态，输出 `selected_p_mw + der_actions`。
5. 安全层将动作投影到 FR/DOE 与 DER 物理边界内，并修复聚合残差。
6. 仿真器写入 pandapower 元件并运行 `runpp`。
7. DSO 汇总电压越限、线路过载、变压器过载、跟踪误差、成本、舒适度/SOC 等分量。
8. 深度 RL trainer 用 reward、log_prob、critic value 和 entropy 更新隐私分离 CTDE actor-critic 网络。

当前智能体分组：

- 全局引导智能体：`dso_global_guidance`
- VPP 调度/解聚合智能体：`<vpp_id>_dispatch`
- VPP 慢周期组合配置智能体：`<vpp_id>_portfolio`
- 训练监督智能体：`deep_training_supervisor`

当前实现边界：

- 已有真实 PyTorch 训练闭环和 DER 级 VPP action head。
- 已有多智能体 observation/action 字典接口。
- 已有 `learning/ctde_interface.py`，用于描述 shared benchmark 与 independent actor CTDE 的 actor/module/critic/action schema contract。
- 已有 `learning/ctde_networks.py`，实现 VPP DER token Deep Sets 编码器、联合动作摘要和 action-conditioned centralized critic。
- 已有 `train_privacy_separated_ctde()`，其中 DSO actor、VPP dispatch actor、VPP portfolio actor 和 centralized critic 是分离模块。
- 当前 centralized critic 使用 global reward、`critic_global_state` 和 `joint_action_summary` 训练；VPP 本地 reward 仍是简化结算感知版本，需要继续增强。
- DSO 包络当前是能力感知启发式包络，尚未升级为 OPF 或机会约束认证包络。
- `docs/experiment_audit.md` 已记录配电网/VPP 实验审查结论：当前默认数据、训练轮数、经济结算和 FR/DOE 仍是 demo/smoke 级；论文级实验前需要补充公开 benchmark feeder、真实负荷/EV/HVAC/价格数据、多 seed 评估、holdout 场景、oracle baseline 和完整经济指标。

## 可视化说明

推荐先运行：

```powershell
python examples/03_timeseries_multi_vpp.py
python examples/07_interactive_report.py
```

然后打开：

- `outputs/interactive_report.html`：综合交互报告。
- `outputs/rl_architecture.html`：强化学习 / MARL 框架图。页面顶部用论文总图风格箭头展示 `VPP 日前报量/报价 -> DSO 运行包络 -> VPP 调度/解聚合 actor -> DER actions -> 安全投影 -> pandapower -> reward/critic/training update`；页面中部的 `Current Implemented Privacy-Separated CTDE Neural Network Architecture` 展示当前主训练器的隐私分离神经网络结构。共享 MLP 基线不会在默认 CTDE 报告中展示。
- `outputs/dashboard_data/rl_target_ctde_architecture.csv`：目标 CTDE 组件表，包含每个 encoder/actor/critic/safety layer 的隐私范围、执行可见性、损失信号和论文角色。
- `outputs/vpp_first_person/index.html`：VPP 第一视角入口。
- `outputs/vpp_first_person/<vpp_id>.html`：单个 VPP 的逐时刻第一视角。
- `outputs/vpp_first_person/economic_explanation.html`：解释 reward、profit proxy 和负收益来源。

如果页面未更新，先重新运行生成脚本；HTML 是静态文件，不会自动热更新。

## 后续路线

下一阶段建议按下面顺序推进：

1. 用 OPF、灵敏度或安全投影算法认证 DSO operating envelope。
2. 将 VPP dispatch reward 从简化 global/shared reward 升级为更完整的本地 settlement-aware reward。
3. 将 centralized critic 从 MLP 升级为拓扑 GNN / set encoder，并保留执行期隐私隔离。
4. 训练慢周期 VPP portfolio agent，使其真实决定聚合配置调整。
5. 扩展 IEEE 123、European LV 或更复杂台区模型。
6. 加入 PettingZoo / RLlib 兼容接口。
7. 建立多种异构 MARL baseline：IPPO、MAPPO、QMIX/VDN、MADDPG/MATD3 等。
8. 将 LLM 接口接入为调度解释、越限诊断和安全动作建议层。
