# User Preferences And Collaboration Memory

Date: 2026-05-01

## Language And Communication

- 默认用中文解释项目进展、设计理由、实验结论和风险。
- 文件名、代码符号、函数名、模型名可以保留英文，但中文模式下 UI 图表和符号说明必须配中文解释。
- 用户希望回答直接、工程化、具体，不要只给抽象计划。
- 对复杂任务可以使用 subagent，但最终集成、验证和结论需要主 agent 负责。

## Project Direction

- 项目核心是配电网 DSO 与多个 VPP 在隐私边界下的协同运行，不是单一集中式 DSO 控制。
- DSO 给 VPP 的应是安全可行域、运行包络、局部灵活性需求或价格/服务信号，而不是直接控制所有 DER。
- VPP 需要有自身调度 agent，负责在包络内解聚合到 DER。
- 每个 VPP 还需要慢周期 portfolio/configuration agent，用于判断是否调整自身聚合配置。
- 训练监督器不是环境内 MARL agent，也不是 LLM agent；它是实验级调参和收敛监督模块。

## Algorithm Expectations

- 不满足于“能跑一轮”的 smoke test。短训练只能验证闭环，不能作为科研结论。
- DRL/MARL 是项目核心，应持续向 CTDE、HRL、MAPPO/MADDPG/QMIX、GNN/Transformer/Set
  encoder、settlement-aware reward 和论文级实验推进。
- 神经网络结构图必须清楚展示：
  - 每个 agent 属于全局引导、VPP 调度、慢周期配置、critic 或安全层。
  - 输入张量、编码器、策略分布、输出动作、critic、loss 和梯度方向。
  - 哪些模块真的用 RL，哪些是规则、安全投影或实验监督。

## Visualization Expectations

- 算法模型、reward、训练流程、agent 结构更新后，必须同步更新动态 HTML、Dash 数据、
  架构图和 VPP 第一视角页面。
- 页面不能只给表格，要有论文总图风格箭头、模块分组、可点击卡片、输入输出解释和奖励函数。
- 电力拓扑图需要像电力系统图，而不是抽象点线图。PV、储能、EVCS、柔性负荷、HVAC、
  MT、母线、馈线、PCC、电压等级、线路潮流和 VPP 归属要清楚。
- 中文模式下图表标题、图例、表头、卡片说明、符号含义都要同步中文化。
- 避免文字重叠、字体太小、动态图卡顿；必要时拆成多个 HTML 文件。

## Subagent Expectations

- 当工具允许显式模型选择时，项目 subagent 优先使用 `gpt-5.5` 和 `xhigh`。
- 每次设计或修改算法模型后，需要 `marl_architecture_engineer` 和
  `grid_vpp_experiment_auditor` 交叉核查。
- 每次设计或修改网络拓扑、DER 映射、FR/DOE、潮流约束后，需要
  `power_system_modeling_auditor` 和 `grid_vpp_experiment_auditor` 核查。
- 每次算法或场景更新后，需要 `ui_visualization_sync_agent` 确认可视化已同步。
- 每个阶段结束后，需要 `memory_curator_agent` 总结经验、偏好、规则和下一步方向。

## Testing Preferences

- 测试应在项目根目录运行：
  `C:\Users\admin\Desktop\panda power\pandapower-vpp-dso-sim`
- 为避免 pytest 扫描临时目录或权限错误，优先使用：
  `python -m pytest -q --basetemp=outputs\pytest_tmp_<name> -o cache_dir=outputs\pytest_cache_<name>`
- 重要算法/UI改动后至少运行：
  - 深度 RL 相关测试。
  - visualization/dashboard smoke tests。
  - 必要时全量 pytest。

## Paper-Grade Standard

- 论文级实验需要多 seed、长 horizon、train/eval split、holdout scenario、oracle baseline、
  rule-based 和 MARL baselines、真实或公开数据集、完整经济结算和网络安全指标。
- 当前 demo 结果不能直接声称 optimal、market-realistic 或 publication-grade。
