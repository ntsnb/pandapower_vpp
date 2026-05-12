# Project Agents

本目录是 `pandapower-vpp-dso-sim` 的项目级 subagent 复用层。它不替代全局
`C:\Users\admin\.codex\agents` 注册表，而是在本项目中把重叠的通用 subagent
整理成少量稳定角色，便于后续 Codex 反复识别、调用和审查。

## 读取顺序

后续处理本项目的复杂任务时，先读：

1. `AGENTS.md`
2. `agents/project_agent_registry.yaml`
3. `memory/rules.md`
4. `memory/user_preferences.md`
5. 与当前任务相关的项目 agent 文件

## 项目级固定角色

| 中文调用名 | 稳定英文 ID / 文件 | 适合调用的任务 |
| --- | --- | --- |
| 项目总协调官 | `main_project_supervisor` / `main_project_supervisor.md` | 主监督与阶段推进，负责把算法、仿真、UI、测试、memory 串起来。 |
| 强化学习算法架构师 | `marl_architecture_engineer` / `marl_architecture_engineer.md` | 深度强化学习 / MARL / CTDE / HRL 结构、收敛、reward、critic、训练循环。 |
| 实验可信度审计员 | `grid_vpp_experiment_auditor` / `grid_vpp_experiment_auditor.md` | 配电网、VPP、数据集、实验设置、指标、论文级可信度审查。 |
| 配电网物理建模审计员 | `power_system_modeling_auditor` / `power_system_modeling_auditor.md` | pandapower 拓扑、DER 映射、符号约定、FR/DOE、潮流与物理约束审查。 |
| 可视化同步设计师 | `ui_visualization_sync_agent` / `ui_visualization_sync_agent.md` | 动态 HTML / Dash / 双语图表 / 拓扑图 / 算法结构图同步。 |
| 项目记忆管理员 | `memory_curator_agent` / `memory_curator_agent.md` | 长期协作经验、用户偏好、规则、阶段结论、交接文档沉淀。 |

## 中文别名调用建议

为了方便自然语言调用，可以直接使用以下中文名；Codex 应映射回稳定英文 ID：

- “请调用**项目总协调官**”：对应 `main_project_supervisor`。
- “请调用**强化学习算法架构师**”或“MARL 收敛诊断师”：对应 `marl_architecture_engineer`。
- “请调用**实验可信度审计员**”：对应 `grid_vpp_experiment_auditor`。
- “请调用**配电网物理建模审计员**”或“电力系统场景专家”：对应 `power_system_modeling_auditor`。
- “请调用**可视化同步设计师**”或“图示结构设计师”：对应 `ui_visualization_sync_agent`。
- “请调用**项目记忆管理员**”：对应 `memory_curator_agent`。

## 已注册线程实例中文名

下面是 `/subagents` 或本会话中已经出现过的具体线程实例。它们不是新的稳定角色文件，而是已经开过的
subagent 会话；如果仍在 `/subagents` 中可见，应优先复用对应线程。

| 中文线程名 | 系统原昵称 | 线程 ID | 对应稳定角色 | 适合调用的任务 |
| --- | --- | --- | --- | --- |
| 强化学习收敛诊断线程 | `Boole` | `019e165d-1113-76c3-8120-d9274b8bd071` | `marl_architecture_engineer` | RL/MARL 不收敛、reward/action/critic/训练设置诊断。 |
| 配电网运行场景专家线程 | `Sagan` | `019e165d-3af3-79d1-9280-944ed29096f8` | `power_system_modeling_auditor` | DSO/VPP/DER 物理建模、pandapower 潮流、FR/DOE、AC 安全审查。 |
| 安全包络咨询解释线程 | `Hume` | `019e167a-3459-75d3-bb23-5af8b2181e40` | `power_system_modeling_auditor` | FR/DOE、projection gap、post-AC violation、安全过滤层的概念解释。 |
| 强化学习算法修复实现线程 | `Meitner` | `019e1682-a414-7d71-825a-c394671d2604` | `marl_architecture_engineer` | reward 尺度修复、AC-aware DOE、安全投影、算法补丁和测试。 |
| 图示结构可视化线程 | `Averroes` | `019e169b-8ba7-79c0-9280-dd28c77de4a2` | `ui_visualization_sync_agent` | 修改前后结构图、Mermaid/HTML 图示、流程说明可视化。 |

调用示例：

- “请复用**强化学习收敛诊断线程**检查 HASAC 是否仍有 mask 问题。”
- “请复用**配电网运行场景专家线程**审查 AC-aware DOE 是否符合真实配电网运行。”
- “请让**图示结构可视化线程**更新论文结构图。”

## 强制联动规则

- 算法模型、agent 架构、reward、训练流程发生变化时，必须同步更新相关可视化：
  `outputs/interactive_report.html`、`outputs/rl_architecture.html`、
  `outputs/vpp_first_person/*.html` 和 `outputs/dashboard_data/*.csv`。
- 新网络拓扑、VPP 场景、DER 参数或实验设定完成后，必须触发：
  `power_system_modeling_auditor` 和 `grid_vpp_experiment_auditor`。
- 新 DRL/MARL/CTDE/HRL 架构完成后，必须触发：
  `marl_architecture_engineer`、`grid_vpp_experiment_auditor`、
  `ui_visualization_sync_agent` 和 `memory_curator_agent`。
- 每个阶段结束时，必须更新 `memory/progress.md`；若形成原则或偏好，更新
  `memory/user_preferences.md` 或 `memory/rules.md`。

## 全局 Agent 重叠处理

全局已注册 subagent 中存在较多重叠。本项目不按名字机械调用，而按项目职责路由：

- UI 类：`ui-designer`、`frontend-developer`、`ui-fixer`、`ux-researcher`、`browser-debugger`
  统一收敛到 `ui_visualization_sync_agent`。
- AI/ML 类：`machine-learning-engineer`、`ml-engineer`、`ai-engineer`、`data-scientist`
  统一收敛到 `marl_architecture_engineer`。
- 实验/研究类：`research-analyst`、`data-researcher`、`docs-researcher`、`search-specialist`
  统一收敛到 `grid_vpp_experiment_auditor`，其中外部资料检索再用 `docs-researcher` 或
  `search-specialist`。
- 编排类：`agent-organizer`、`multi-agent-coordinator`、`workflow-orchestrator`、
  `task-distributor`、`project-manager` 统一收敛到 `main_project_supervisor`。
- 记忆/总结类：`context-manager`、`knowledge-synthesizer` 统一收敛到
  `memory_curator_agent`。

当工具允许显式模型选择时，优先使用 `model="gpt-5.5"` 与
`reasoning_effort="xhigh"`。如果某个全局角色固定模型不可覆盖，则使用该固定角色，
并在 prompt 中明确项目级职责边界。
