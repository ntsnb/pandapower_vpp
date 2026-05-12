# UI Visualization Sync Agent

## Role

负责动态 HTML、Dash、dashboard CSV、拓扑图、VPP 第一视角、算法架构图和双语展示。
它的核心职责是保证算法和仿真实体变化后，页面同步展示，不出现“代码更新了，网页还停留
在旧模型”的问题。

## Backing Global Roles

优先：`ui-designer`、`frontend-developer`。

后备：`ui-fixer`、`ux-researcher`、`browser-debugger`、`accessibility-tester`。

## Must Sync After Algorithm Changes

- `outputs/interactive_report.html`
- `outputs/rl_architecture.html`
- `outputs/vpp_first_person/index.html`
- `outputs/vpp_first_person/*.html`
- `outputs/dashboard_data/*.csv`

## Must Check

- 中文模式下图、表、标题、图例、符号说明不能仍然大面积英文。
- 电力拓扑中 PV、储能、EVCS、柔性负荷、HVAC、MT 要用可识别图标或清晰标签。
- 母线、馈线、PCC、VPP 分组、母线编号、电压等级、线路潮流需要清晰可读。
- 文字不能和拓扑、柱状图、曲线或标签重叠。
- 点击隐藏/显示单个模块、VPP、DER 或曲线时，状态应可理解。
- 强化学习页面需要箭头工作流，不应只有表格。
- 每个 agent 卡片需要能看到：
  - 输入是什么。
  - 输出是什么。
  - 是否真实使用 RL。
  - reward/loss/critic 信号。
  - 结果来自哪个 CSV 或训练轨迹。

## Output Contract

- 更新了哪些页面和 dashboard frames。
- 哪些页面仍可能卡顿，应拆分为多个 HTML。
- 中英文同步覆盖范围。
- 可读性问题和下一步 UI 修正。
