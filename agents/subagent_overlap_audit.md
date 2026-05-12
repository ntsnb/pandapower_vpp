# Subagent Overlap Audit

Date: 2026-05-01

## Summary

已注册全局 subagent 覆盖面很广，存在明显功能重叠。重叠本身不是问题，问题是如果
每次都直接按全局名字调用，会导致同一任务在 `ui-designer`、`frontend-developer`、
`ui-fixer` 或 `machine-learning-engineer`、`ml-engineer`、`ai-engineer` 之间摇摆。

本项目采用“项目级角色收敛”：保留全局 agent 作为 backing roles，在 `agents/`
中定义少量面向 DSO/VPP/DRL/UI/实验审查的稳定职责。

## Overlap Groups

### UI / Visualization

重叠角色：

- `ui-designer`
- `frontend-developer`
- `ui-fixer`
- `ux-researcher`
- `browser-debugger`
- `accessibility-tester`

项目收敛：`ui_visualization_sync_agent`

原因：本项目 UI 不是普通网页，而是算法、拓扑、VPP 第一视角、双语图表和 dashboard
CSV 的同步展示。只做视觉设计或只修 CSS 都不够。

### AI / ML / MARL

重叠角色：

- `machine-learning-engineer`
- `ml-engineer`
- `ai-engineer`
- `data-scientist`
- `research-analyst`

项目收敛：`marl_architecture_engineer`

原因：本项目的核心是隐私分离 CTDE、多智能体分层控制、VPP 解聚合和慢周期 portfolio
agent。普通 ML 工程、应用 AI 工程和数据科学审查会交叠，需统一成 MARL 架构职责。

### Research / Experiment Audit

重叠角色：

- `research-analyst`
- `data-researcher`
- `docs-researcher`
- `search-specialist`
- `trend-analyst`
- `competitive-analyst`

项目收敛：`grid_vpp_experiment_auditor`

原因：本项目实验审查必须同时看 feeder benchmark、数据集、VPP 参数、reward/economics、
训练规模和论文可主张性，不是普通文献搜索。

### Orchestration

重叠角色：

- `agent-organizer`
- `multi-agent-coordinator`
- `workflow-orchestrator`
- `task-distributor`
- `project-manager`
- `scrum-master`

项目收敛：`main_project_supervisor`

原因：用户希望长期推进，不能只分任务。该角色需要阶段目标、subagent 边界、集成验证、
UI 同步和 memory 更新一起把控。

### Memory / Knowledge

重叠角色：

- `context-manager`
- `knowledge-synthesizer`
- `documentation-engineer`
- `technical-writer`

项目收敛：`memory_curator_agent`

原因：项目需要持续沉淀用户偏好、规则、经验、阶段成果和反复出现的问题，不只是生成
一段上下文摘要或写普通文档。

### Review / Quality

重叠角色：

- `reviewer`
- `code-reviewer`
- `architect-reviewer`
- `qa-expert`
- `test-automator`
- `debugger`

项目处理：不固定收敛为一个常驻项目 agent。只有在具体 patch、失败测试、release gate
或架构风险出现时按问题调用。

## Project-Level Minimum High-Quality Team

日常复杂推进默认使用：

1. `main_project_supervisor`
2. `marl_architecture_engineer`
3. `grid_vpp_experiment_auditor`
4. `ui_visualization_sync_agent`
5. `memory_curator_agent`

涉及 pandapower 拓扑、DER 映射、FR/DOE 或潮流约束时，再加入：

6. `power_system_modeling_auditor`

## Non-Goals

- 不删除或修改全局 `C:\Users\admin\.codex\agents` 注册表。
- 不让项目 agent 取代工具层的真实 `agent_type`。
- 不把 training supervisor 误认为环境内 MARL agent；它只是实验编排/调参监督模块。
