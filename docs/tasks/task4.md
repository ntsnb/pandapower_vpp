你现在是一个资深全栈工程师、强化学习工程师、能源系统仿真平台架构师。请在当前代码仓库中，为“多智能体强化学习 VPP 实验平台”搭建一个本地实时 Web 可视化训练平台。

项目背景：
我正在研发一个多智能体强化学习实验平台，场景是多个 VPP（Virtual Power Plant，虚拟电厂）智能体在电价、充电桩、储能、光伏、风电、负荷等数据下进行训练。训练一启动，就需要在本地 127.0.0.1 启动一个 Web 服务，实时记录、计算并展示所有关键数据。平台既要服务算法调试，也要服务论文实验分析，因此必须有清晰的数据结构、漂亮的前端界面、可扩展的指标系统和可靠的实时日志系统。

请你先扫描当前仓库结构，理解已有训练入口、环境、智能体、reward、cost、loss、dataset 的代码位置。不要破坏已有训练逻辑。采用渐进式重构：先新增 dashboard 模块和 logging adapter，再把已有训练循环接入该模块。如果仓库中暂时缺少某些真实数据或指标，请实现 demo/synthetic 模式，但必须清晰区分 demo 数据和真实训练数据。

总体目标：
1. 当训练启动时，自动在本地 127.0.0.1 启动 Web 服务。
2. 训练过程中实时记录所有需要展示的数据。
3. 前端网页实时展示训练状态、数据集曲线、reward、cost、loss、变量物理意义、数值、单位、公式。
4. 支持按 run、epoch、episode、日期、时刻、VPP 进行筛选和对比。
5. 支持同一时刻不同 VPP 对比、同一 VPP 不同 epoch 对比、同一日期下逐时刻曲线分析。
6. 页面要美观、现代、清晰，适合科研演示和实验复盘。

推荐技术栈：
后端：
- Python 3.11+
- FastAPI
- Uvicorn
- Pydantic v2
- DuckDB
- PyArrow / Parquet
- Pandas 或 Polars，优先考虑 Polars 处理大规模时间序列
- WebSocket 用于实时推送训练状态和最新指标
- REST API 用于历史数据查询
- pytest、ruff、mypy 用于质量控制

前端：
- React + TypeScript + Vite
- Tailwind CSS
- shadcn/ui 风格组件，或自行实现同等水平的现代 UI 组件
- Apache ECharts 或 Plotly，优先选择 ECharts，要求支持多曲线、tooltip、legend、缩放、brush、导出图像
- TanStack Table 或等价方案，用于大表格、排序、筛选、虚拟滚动
- KaTeX 或 react-katex，用于 reward/cost/loss 公式渲染
- Zustand 或 TanStack Query，用于前端状态管理和 API 请求缓存

核心架构要求：
请实现如下模块结构，具体路径可根据当前仓库调整，但要保持清晰分层：

marl_dashboard/
  backend/
    app.py                    # FastAPI app 创建入口
    server.py                 # 启动/停止本地 dashboard 服务
    api/
      runs.py                 # run 查询接口
      selectors.py            # epoch/date/time/vpp 可选项接口
      datasets.py             # 数据集曲线和表格接口
      rewards.py              # reward 接口
      costs.py                # cost 接口
      losses.py               # loss 接口
      variables.py            # 变量字典和物理意义接口
      compare.py              # 同时刻、跨 VPP、跨 epoch 对比接口
      websocket.py            # WebSocket 实时推送
    schemas/
      common.py
      run.py
      selector.py
      metric.py
      formula.py
      variable.py
    storage/
      duckdb_store.py         # DuckDB 查询封装
      parquet_writer.py       # Parquet 分区写入
      metadata_store.py       # run metadata、schema、变量字典
      query_service.py        # 图表和表格查询服务
    logging/
      experiment_logger.py    # 训练侧调用的统一 logger
      event_bus.py            # 内存事件队列和 WebSocket 广播
      adapters.py             # 对接不同训练框架/自定义训练循环
    demo/
      generate_demo_run.py    # 生成一个可完整演示的 VPP 多智能体训练数据
  frontend/
    package.json
    vite.config.ts
    src/
      main.tsx
      App.tsx
      api/
        client.ts
        types.ts
        hooks.ts
      components/
        layout/
          AppShell.tsx
          Sidebar.tsx
          TopBar.tsx
          FilterBar.tsx
        selectors/
          RunSelector.tsx
          CalendarPicker.tsx
          WheelPicker.tsx
          VppWheelPicker.tsx
          EpochWheelPicker.tsx
          TimeStepSlider.tsx
        charts/
          MetricLineChart.tsx
          MultiPanelChart.tsx
          CombinedChart.tsx
          CompositionChart.tsx
          SameTimeCompareChart.tsx
        tables/
          MetricTable.tsx
          FormulaTable.tsx
          VariableDictionaryTable.tsx
        cards/
          StatusCard.tsx
          CurrentValueCard.tsx
          VppSummaryCard.tsx
      pages/
        OverviewPage.tsx
        DatasetPage.tsx
        RewardCostPage.tsx
        LossPage.tsx
        ComparePage.tsx
        VariableDictionaryPage.tsx
        RunConfigPage.tsx
      styles/
        globals.css
  cli.py                      # marl-dashboard 命令行入口
  README.md

数据层要求：
使用“高频训练日志 + 可查询实验湖”的设计。训练侧不能因为写日志阻塞训练，因此 logger 应该先把事件写入队列，由后台 writer 异步批量写入 Parquet，并同步维护轻量 metadata。查询层使用 DuckDB 直接查询 Parquet 分区。

数据目录建议：
runs/
  {run_id}/
    metadata.json
    config.json
    variable_dictionary.json
    formulas.json
    tables/
      dataset_timeseries/
        epoch_id=000001/
          vpp_id=vpp_001/
            part-000.parquet
      reward_terms/
      cost_terms/
      loss_terms/
      scalar_metrics/
      events/
      model_metrics/

必须支持如下核心维度：
- run_id：一次训练运行 ID
- epoch_id：训练轮次或 learner update round
- episode_id：环境从 reset 到结束的一条轨迹编号
- batch_id：用于一次或多次梯度更新的数据批次编号
- gradient_step：真实梯度更新步数
- global_env_step：所有并行环境累计交互步数
- env_id：并行环境编号
- vpp_id：VPP 智能体编号
- agent_id：智能体编号，可与 vpp_id 相同
- policy_id：多智能体算法中的策略编号
- date：数据集对应日期
- time_index：当天第几个时刻，例如 0-23 或 0-95
- timestamp：真实时间戳
- metric_name：指标名称
- metric_group：dataset / reward / cost / loss / variable / event
- value：数值
- unit：单位
- formula_latex：公式的 LaTeX 表达
- description：物理意义或说明

变量字典要求：
请实现 variable_dictionary.json 或同等结构，用于管理所有变量的物理意义、单位、符号、取值范围、来源、公式。前端“变量字典”页面必须可搜索、可筛选、可查看当前选中时刻的变量值。

变量字典字段示例：
{
  "name": "pv_power",
  "display_name": "光伏出力",
  "symbol": "P^{PV}_{i,t}",
  "unit": "kW",
  "group": "dataset",
  "physical_meaning": "第 i 个 VPP 在 t 时刻的光伏发电功率",
  "formula_latex": "P^{PV}_{i,t}",
  "min_value": 0,
  "max_value": null,
  "source": "dataset",
  "notes": "若数据为 MW，请在 ingestion 阶段统一换算或标明单位"
}

必须支持的数据集指标：
- electricity_price：电价，单位可配置，例如 元/kWh 或 $/MWh
- ev_charging_load：充电桩负荷，单位 kW 或 MW
- storage_power：储能充放电功率，单位 kW 或 MW，正负号含义必须写入变量字典
- storage_soc：储能 SOC，单位 %
- pv_power：光伏出力，单位 kW 或 MW
- wind_power：风电出力，单位 kW 或 MW
- base_load：基础负荷，单位 kW 或 MW
- net_load：净负荷，可由 base_load + ev_charging_load - pv_power - wind_power 等计算，公式必须可配置

Reward 和 cost 要求：
Reward 和 cost 必须支持拆项记录，不要只记录 total_reward。每个 VPP、每个时刻、每个 episode、每个 epoch 都应能看到各项 reward/cost 的数值、占比、公式。

示例 reward_terms：
- total_reward
- profit_reward
- grid_balance_reward
- storage_degradation_penalty
- carbon_penalty
- ev_satisfaction_reward
- curtailment_penalty
- constraint_violation_penalty

示例 cost_terms：
- energy_purchase_cost
- energy_sell_revenue
- storage_degradation_cost
- carbon_emission_cost
- ev_unserved_cost
- renewable_curtailment_cost
- power_imbalance_cost
- constraint_violation_cost

每个 term 需要：
- term_name
- value
- unit
- sign_convention，例如 reward 越大越好，cost 越小越好
- formula_latex
- formula_description
- component_ratio，占 total_reward 或 total_cost 的比例，注意 total 接近 0 时要避免除零

Loss 要求：
Loss 也必须支持拆项记录。兼容 MAPPO、MADDPG、QMIX、VDN、IPPO、IQL 等常见多智能体算法，但不要强依赖某一个算法。先做通用 schema。

示例 loss_terms：
- total_loss
- actor_loss
- critic_loss
- value_loss
- policy_loss
- entropy_loss
- q_loss
- td_error
- mixer_loss
- imitation_loss
- regularization_loss
- constraint_loss
- auxiliary_loss

每个 loss term 需要：
- term_name
- value
- unit，通常为 scalar
- formula_latex
- formula_description
- optimizer_name
- network_name，例如 actor、critic、mixer
- policy_id
- component_ratio

ExperimentLogger 训练侧 API：
请实现一个非常容易接入训练代码的 logger。示例用法如下：

from marl_dashboard.logging import ExperimentLogger, start_dashboard

dashboard = start_dashboard(
    data_dir="runs",
    host="127.0.0.1",
    port=8765,
    open_browser=True,
)

logger = ExperimentLogger(
    run_id=run_id,
    data_dir="runs",
    config=train_config,
    variable_dictionary=variable_dictionary,
    formulas=formulas,
)

logger.log_dataset(
    epoch_id=epoch,
    episode_id=episode,
    env_id=env_id,
    vpp_id=vpp_id,
    date=date,
    time_index=t,
    timestamp=timestamp,
    values={
        "electricity_price": price,
        "ev_charging_load": ev_load,
        "storage_power": storage_power,
        "storage_soc": soc,
        "pv_power": pv,
        "wind_power": wind,
        "base_load": load,
        "net_load": net_load,
    },
    units={
        "electricity_price": "元/kWh",
        "ev_charging_load": "kW",
        "storage_power": "kW",
        "storage_soc": "%",
        "pv_power": "kW",
        "wind_power": "kW",
        "base_load": "kW",
        "net_load": "kW",
    },
)

logger.log_reward_terms(
    epoch_id=epoch,
    episode_id=episode,
    env_id=env_id,
    vpp_id=vpp_id,
    date=date,
    time_index=t,
    terms={
        "profit_reward": profit_reward,
        "grid_balance_reward": balance_reward,
        "storage_degradation_penalty": degradation_penalty,
        "constraint_violation_penalty": violation_penalty,
        "total_reward": total_reward,
    },
)

logger.log_cost_terms(...)
logger.log_loss_terms(...)
logger.log_scalar("episode_return", value, epoch_id=epoch, episode_id=episode)
logger.log_event("training_status", {"message": "epoch finished", "epoch_id": epoch})
logger.flush()

启动要求：
1. 提供 Python API：start_dashboard(...)
2. 提供 CLI：
   - marl-dashboard serve --data-dir runs --host 127.0.0.1 --port 8765
   - marl-dashboard demo --data-dir runs --port 8765
3. 训练脚本调用 start_dashboard 后，控制台必须打印：
   Dashboard running at http://127.0.0.1:8765
4. 必须默认只绑定 127.0.0.1，不允许默认暴露到 0.0.0.0。
5. 如果端口被占用，给出清晰错误；可以支持 auto_port=True 自动寻找可用端口。
6. 训练停止后，logger 必须安全 flush，避免数据损坏。

后端 API 要求：
请实现如下 REST API 和 WebSocket API：

GET /api/health
返回服务状态、版本、当前 data_dir。

GET /api/runs
返回所有 run 列表，包括 run_id、开始时间、结束时间、状态、算法、环境、VPP 数量、epoch 数量。

GET /api/runs/{run_id}/metadata
返回 config、训练状态、指标摘要、数据时间范围。

GET /api/runs/{run_id}/variables
返回变量字典，包括物理意义、单位、公式、当前选中值可选。

GET /api/runs/{run_id}/selectors
返回可选 date、vpp_id、epoch_id、time_index、episode_id、policy_id。

GET /api/runs/{run_id}/dataset
查询数据集曲线和表格。
参数：
- epoch_id
- date
- vpp_id
- start_time_index
- end_time_index
- metrics，可多选
返回：
- chart_series
- table_rows
- units
- formulas
- summary

GET /api/runs/{run_id}/rewards
查询 reward 曲线、拆项、表格、公式。

GET /api/runs/{run_id}/costs
查询 cost 曲线、拆项、表格、公式。

GET /api/runs/{run_id}/losses
查询 loss 曲线、拆项、表格、公式。

GET /api/runs/{run_id}/compare
用于同时刻对比。
参数：
- scope: dataset | reward | cost | loss
- fixed_epoch_id
- fixed_date
- fixed_time_index
- metric_names
- group_by: vpp_id | epoch_id | policy_id | agent_id
返回适合画柱状图、雷达图、热力图和表格的数据。

GET /api/runs/{run_id}/formulas
返回所有 reward/cost/loss 的公式，LaTeX 格式。

WebSocket /ws/runs/{run_id}/live
实时推送：
- run_status
- latest_scalar_metrics
- latest_dataset_point
- latest_reward_terms
- latest_cost_terms
- latest_loss_terms
- latest_event
要求前端可以切换 live/frozen 模式。live 模式自动跟随最新 epoch 和 time_index；frozen 模式固定用户选择的 epoch/date/time/vpp。

前端页面要求：
整体界面风格：
- 科研级实时实验 dashboard
- 简洁、现代、清晰
- 支持深色/浅色主题
- 顶部显示当前 run、训练状态、epoch 进度、env steps、gradient steps、当前 episode、当前 VPP、WebSocket 连接状态
- 左侧 Sidebar 页面导航
- 主区域卡片化布局
- 图表必须有清晰坐标轴、单位、legend、tooltip、缩放、导出
- 表格必须支持排序、筛选、列显隐、CSV 导出
- 公式必须用 KaTeX 渲染，而不是普通字符串
- 所有变量名旁边提供 tooltip，解释物理意义和单位

全局筛选器 FilterBar：
必须在 Overview、Dataset、Reward/Cost、Loss、Compare 页面复用。
包含：
1. RunSelector：选择训练 run
2. CalendarPicker：日期日历，只有有数据的日期可选，并高亮
3. VppWheelPicker：VPP 轮盘选择器，可选择单个或多个 VPP
4. EpochWheelPicker：epoch 轮盘选择器，支持大量 epoch 虚拟滚动
5. TimeStepSlider：选择当天具体时刻，例如 0-23 或 0-95
6. Live/Frozen Toggle：实时跟随或固定查看
7. Compare Mode：同时刻对比开关
8. Reset Filters

轮盘选择器要求：
- 支持鼠标滚轮、拖拽、键盘方向键
- 当前选中项居中高亮
- 数据量大时不能卡顿，应使用虚拟列表或窗口化渲染
- VPP 显示格式：VPP-001 / VPP-002
- Epoch 显示格式：Epoch 000001
- 保留普通 Select/Combobox 的 fallback，保证可访问性

页面 1：OverviewPage 总览
目标：我可以查看所有变量的物理意义及其当前数值。
内容：
- 当前训练状态卡片：Running / Finished / Error
- 当前 epoch、episode、global_env_step、gradient_step
- 当前选择的 date、time_index、timestamp、vpp_id
- 当前 VPP 的关键数据卡片：
  - 电价
  - 充电桩负荷
  - 储能功率
  - 储能 SOC
  - 光伏
  - 风电
  - 负荷
  - total_reward
  - total_cost
  - total_loss
- 变量字典摘要卡片：显示变量名、符号、单位、物理意义
- 最近事件日志
- 如果开启同时刻对比，显示该时刻所有 VPP 的核心指标对比表和柱状图

页面 2：DatasetPage 数据集可视化
目标：逐时刻展示电价、充电桩、储能、光伏、风电、负荷数据。
筛选条件：
- date
- vpp_id
- epoch_id
- time_index
功能：
1. 六个分子图：
   - 电价曲线
   - 充电桩负荷曲线
   - 储能曲线，最好同时展示 storage_power 和 storage_soc
   - 光伏曲线
   - 风电曲线
   - 负荷曲线，最好支持 base_load/net_load
2. 一个总图：
   - 将六类指标放在一起
   - 多 y 轴或归一化模式可切换
   - 鼠标 hover 时显示具体数值、单位、曲线名称、日期、时刻、VPP
3. 逐时刻表格：
   - 行代表时间
   - 列代表 electricity_price、ev_charging_load、storage_power、storage_soc、pv_power、wind_power、base_load、net_load
   - 列名显示单位
   - 支持导出 CSV
4. 同时刻对比：
   - 当用户固定 epoch/date/time_index 后，可显示所有 VPP 在该时刻的各数据项对比
   - 使用柱状图或热力图

页面 3：RewardCostPage
目标：逐个 VPP 展示各项 reward 和 cost 的占比、逐时刻曲线、表格和公式。
筛选条件：
- date
- vpp_id
- epoch_id
- time_index
Reward 区域：
1. Reward 拆项占比图：
   - 支持饼图、环形图或堆叠条形图
   - 显示各项 reward 对 total_reward 的贡献
2. Reward 分子图：
   - 每个 reward term 一个小图
   - 另有 total_reward 曲线
3. Reward 总图：
   - 所有 reward term 放在同一张图
   - 鼠标 hover 显示值、单位、公式名称
4. Reward 表格：
   - 行代表 time_index
   - 列代表 reward terms
   - 附带公式列，公式必须用 KaTeX 渲染
Cost 区域：
1. Cost 拆项占比图
2. Cost 分子图
3. Cost 总图
4. Cost 表格，包含公式渲染
5. Reward 和 cost 的关系图，例如 reward vs cost、profit vs penalty

页面 4：LossPage
目标：逐个 VPP 或 policy 展示各类型 loss 以及各 loss 项占比、逐时刻或逐 update 曲线。
筛选条件：
- epoch_id
- policy_id
- vpp_id，可选
- gradient_step
功能：
1. Loss 拆项占比图：
   - actor_loss、critic_loss、entropy_loss、value_loss、q_loss、mixer_loss 等
2. Loss 分子图：
   - 每个 loss term 一个小图
3. Loss 总图：
   - 所有 loss term 放在同一张图
4. Loss 表格：
   - 行代表 gradient_step 或 time_index，根据数据实际情况选择
   - 列代表各 loss term
   - 公式列用 KaTeX 渲染
5. 支持按 policy_id 对比 loss
6. 支持查看 loss 是否 NaN、inf、爆炸，并在 UI 中给出警告 badge

页面 5：ComparePage 同时刻对比
目标：专门处理“同时刻对比”。
用户固定：
- epoch_id
- date
- time_index
可选择：
- 多个 VPP
- 多个指标
展示：
1. 所选时刻所有 VPP 的指标矩阵热力图
2. 每个指标的 VPP 横向柱状图
3. 雷达图，可选
4. 表格：
   - 行为 VPP
   - 列为所选指标
5. 支持跨 epoch 对比：
   - 固定 VPP/date/time_index
   - 比较不同 epoch 下该时刻的 reward/cost/loss 变化

页面 6：VariableDictionaryPage
目标：查看所有变量的物理意义及当前数值。
功能：
- 搜索变量
- 按 group 筛选：dataset / reward / cost / loss / action / observation / state / constraint
- 显示：
  - 变量名
  - 展示名
  - 数学符号
  - 单位
  - 物理意义
  - 公式
  - 当前选中时刻的数值
  - 来源
- 公式用 KaTeX 渲染

页面 7：RunConfigPage
目标：查看实验配置、训练日志和运行元信息。
功能：
- 展示训练 config
- 展示算法名称、环境名称、seed、VPP 数量、数据集范围、episode horizon、batch size、learning rate 等
- 展示事件日志
- 支持下载 metadata/config/formulas/variable dictionary

实时性和性能要求：
1. 训练写日志不能明显拖慢训练。
2. 使用队列和后台批量写入。
3. WebSocket 推送需要限频，例如默认 2-5 Hz，避免前端过载。
4. 图表数据量过大时后端要支持 downsample，例如 max_points=2000。
5. 表格数据要分页或虚拟滚动。
6. DuckDB 查询要使用 Parquet 分区过滤，避免全量扫描。
7. NaN、inf、None 必须在后端统一处理，前端不能崩溃。
8. 所有 API 返回结构必须稳定，由 Pydantic schema 定义。
9. 单位必须贯穿后端和前端，不允许图表没有单位。

多智能体强化学习术语要求：
请在代码和 UI 中区分以下概念，不要混用：
- epoch_id：训练轮次或 learner update round，不一定等于一个 episode，也不一定等于遍历完整数据集一次
- episode_id：环境从 reset 到 terminated/truncated 的完整轨迹
- rollout / trajectory：由多个 step 组成的轨迹片段或完整轨迹
- batch_id：用于训练更新的数据批次
- gradient_step：优化器实际更新次数
- global_env_step：环境交互总步数
- time_index：能源数据集中一天内的时刻
- date：能源数据集日期
- vpp_id / agent_id：VPP 智能体编号
前端 tooltip 中也要解释这些概念。

推荐默认定义：
对于 VPP 日前调度问题，如果数据是逐小时或 15 分钟粒度，默认 episode 可以设为一个调度周期，例如一天 24 步或 96 步。5 周数据集不应默认等于一个 epoch；5 周更适合作为训练样本池。epoch_id 建议表示训练外层迭代或 learner update round。请在 README 中解释这一点。

Demo 数据要求：
如果当前仓库没有完整训练数据，请生成 demo run：
- 5 个 VPP
- 5 周数据
- 每天 24 个时刻，或通过参数支持 96 个时刻
- 生成电价、充电桩、储能、SOC、光伏、风电、负荷
- 生成 reward/cost/loss 拆项
- 生成多个 epoch，让前端 wheel picker 能演示选择 epoch
- 生成公式和变量字典
- demo 模式必须可通过 marl-dashboard demo 启动

代码质量要求：
1. 所有新增 Python 代码要有类型标注。
2. 所有 API schema 用 Pydantic。
3. 前端 TypeScript 不能用 any，除非有明确理由并加注释。
4. 关键模块写单元测试。
5. 后端测试：
   - logger 能写入 Parquet
   - DuckDB 能查询 dataset/reward/cost/loss
   - selectors API 正确返回可选日期、VPP、epoch
   - WebSocket 能收到 live event
6. 前端测试：
   - FilterBar 状态联动
   - DatasetPage 能渲染六个分图和总图
   - FormulaTable 能渲染 LaTeX
7. 提供 README：
   - 如何启动 demo
   - 如何接入真实训练
   - logger API 示例
   - 数据 schema 说明
   - epoch/episode/batch/trajectory 概念说明

集成到现有训练代码：
请在扫描仓库后，找到最合适的训练入口，例如 train.py、main.py、runner.py、trainer.py 或 scripts/train_*.py。
如果找到训练循环，请做最小侵入式集成：
- 在训练开始前 start_dashboard
- 初始化 ExperimentLogger
- 在 env.step 后 log dataset/reward/cost
- 在每次 learner update 后 log loss
- 在 episode 结束后 log scalar episode_return、episode_length
- 在 epoch/iteration 结束后 log summary
如果无法安全判断训练循环位置，请不要盲目修改核心算法。请新增 examples/integrate_logger_example.py，展示如何接入，并在 README 说明需要用户手动放置的位置。

验收标准：
1. 运行 marl-dashboard demo 后，浏览器打开 http://127.0.0.1:8765，能看到完整 dashboard。
2. Overview 能显示当前 run、当前 epoch、当前日期、当前 VPP、关键变量当前值和物理意义。
3. Dataset 页面能按 date/vpp/epoch 选择，显示六个分图、一个总图、逐时刻表格。
4. Reward/Cost 页面能显示拆项占比、逐时刻曲线、总图、公式表格。
5. Loss 页面能显示 actor/critic/value/entropy 等 loss 曲线和表格。
6. Compare 页面能固定同一时刻，对比多个 VPP 或多个 epoch。
7. Variable Dictionary 页面能搜索变量，显示单位、物理意义、公式和当前值。
8. 真实训练调用 logger 时，前端能通过 WebSocket 实时更新。
9. 所有核心测试通过。
10. 不默认暴露外网，只绑定 127.0.0.1。

请按以下步骤执行：
第一步：扫描仓库，输出简短实现计划和你发现的训练入口、数据结构、reward/loss 位置。
第二步：新增 dashboard 后端、数据 schema、logger、demo 数据生成器。
第三步：新增 React/Vite 前端，实现页面、筛选器、图表、表格、公式渲染。
第四步：接入训练入口；如果不能安全接入，则提供清晰示例。
第五步：补测试和 README。
第六步：运行 lint/test/build，修复错误。
最后请汇报：
- 新增和修改了哪些文件
- 如何启动 demo
- 如何接入真实训练
- 还有哪些需要用户确认的变量公式或单位
