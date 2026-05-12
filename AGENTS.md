# AGENTS.md — 配电网台区多 VPP 双向引导最优聚合仿真项目

> 面向 Codex / AI coding agents。本文是本仓库的长期研究工程主控提示，用于约束后续代码改造、实验设计、memory 沉淀与多智能体流程编排。目标不是重写已有 pandapower 仿真框架，而是在现有台区/馈线仿真、VPP/DER 逻辑对象、可视化和最小 RL 环境之上，逐步实现 **DSO↔多 VPP 双向引导、隐私保护、三时标、位置感知、多智能体/分层强化学习** 的最优聚合闭环。

若子目录存在更近的 `AGENTS.md`，更近文件对该子树优先。用户当前任务指令优先，但任何会破坏物理一致性、隐私边界、研究主线或实验口径的操作，必须先说明风险并请求确认。

---

## 0. 研究主线：不要把本项目做偏

本项目的核心不是“给已有仿真器接一个普通 RL 算法”，也不是“让 DSO 集中式最优控制所有 DER”。核心研究假设是：

> **在 DSO 不掌握 VPP 内部真实成本与逐 DER 私有状态、VPP 不掌握完整配电网拓扑与其他 VPP 私有策略的条件下，DSO 通过日前功率可行域（FR/DOE）和日内位置灵活性价格/报价（LFP/LFB）自上而下引导 VPP；VPP 在可行域内追求自身收益最大化，并通过响应轨迹自下而上向 DSO 暴露“可验证的支撑能力”。长期看，DSO 可由历史响应学习节点/区域需求状态与 VPP 能力画像，VPP 也可由 DSO 的需求与价格历史学习区域电气特征，进而调整 DER 聚合组合，减少资源错配和无效响应。**

因此，所有后续开发必须服务于以下闭环：

```text
DSO 历史运行数据 + 网络状态
        ↓
节点/区域需求表征 NodeNeedEmbedding
        ↓
日前 FR/DOE：告诉 VPP_i 在哪些 bus/zone/时段内“能做什么”
        ↓
日内 LFP/LFB / LocalFlexNeed：告诉 VPP_i “哪里、何时、什么方向的灵活性更有价值”
        ↓
VPP_i 在 FR/DOE 内进行收益最大化、报价、调度和服务响应
        ↓
真实 bus/zone 注入 + pandapower 潮流 + 约束/结算/履约结果
        ↓
DSO 根据响应轨迹更新 VPP 能力画像 VppCapabilityEmbedding
        ↓
慢循环中 VPP 根据 DSO 需求历史与收益/履约反馈更新 DER 聚合组合
```

**研究贡献必须体现“双向学习与双向引导”：**

- DSO → VPP：FR/DOE 是安全硬边界；LFP/LFB 是经济软激励；LocalFlexNeed 是服务需求。
- VPP → DSO：报价、响应曲线、实际交付、偏差、可靠性和位置化响应轨迹用于 DSO 判断 VPP 对不同配电网问题的支撑能力。
- DSO 学习：从历史潮流、告警、灵活性调用和 VPP 响应中形成节点/区域需求状态。
- VPP 学习：从 DSO 下发的 FR/LFP/服务调用历史中形成对区域电气特征和需求偏好的隐式认知。
- 慢循环聚合：VPP 的 DER 组合不能每个时步随意改变，只能在较长周期更新；调度在快循环完成。

任何把本项目退化为以下方向的改动，必须在 `memory/pitfalls.md` 记录并请求用户确认：

```text
1. 纯 DSO 集中式调度，VPP 没有自利目标。
2. 纯 VPP 收益最大化，不考虑 DSO 网络安全和位置约束。
3. 单 PCC 总 MW 聚合，忽略多节点 VPP 的 DER 真实注入位置。
4. 只有 FR/DOE，无 LFP/LFB、报价、响应、能力画像和聚合更新。
5. 只有 RL 训练曲线，没有 oracle baseline、隐私边界和物理一致性审计。
```

---

## 1. 当前项目状态：基于已有仓库增量改造

本项目不是空仓库。README 描述的现有能力包括：

- IEEE 33 节点风格配电馈线和简化低压台区网络。
- PV、微型燃机、储能、柔性负荷、HVAC、EV、EVCS 映射到 pandapower。
- 逻辑层 DSO 与 VPP；VPP 不是 pandapower 原生元件。
- 默认 72 小时、15 min 间隔、288 步多 VPP 时序潮流仿真。
- 输出母线电压、线路/变压器负载、VPP 出力、DER 调度、SOC、温度、约束违规、reward 分量。
- dashboard 标准化长表、静态拓扑图、Plotly 离线报告、只读 Dash dashboard。
- 最小 Gymnasium 风格环境 `VPPDSOEnv`，后续可接入 DRL、HRL、MARL、RLlib、SB3、PettingZoo。

因此后续默认执行策略是：

```text
保留 network / der / entities / simulation / visualization / dashboard 基础能力。
优先在 optimization / envs / learning / memory 中增量加入 FR、LFP、出清、内层求解器、MARL/HRL 和表征学习。
不要重写已有仿真主干，除非测试证明旧接口无法承载研究主线。
```

常用验证命令：

```powershell
python -m pytest
python -m pytest tests/test_env_smoke.py tests/test_timeseries_smoke.py
python examples/03_timeseries_multi_vpp.py
python examples/05_random_rl_env_rollout.py
python examples/08_run_dashboard.py --check
```

涉及 dashboard/export/可视化字段时，必须确认 `outputs/dashboard_data/*.csv`、Plotly 报告和 Dash `--check` 不被破坏。

---

## 2. 领域硬约束：物理一致性与 VPP 含义

### 2.1 VPP 聚合不改变 DER 的物理接入节点

每个 DER 必须保留真实物理位置：

```text
der_id, der_type, bus_id, phase, pp_element_type, pp_element_index,
p_min_mw, p_max_mw, q_min_mvar, q_max_mvar,
owner_vpp_id, controllability, private_cost_fields
```

VPP 聚合只能改变：

```text
DER -> VPP 的商业归属、控制权、预测、报价、结算和风险承担
```

不能改变：

```text
DER -> bus_id / phase / pp_element 的真实物理注入位置
```

禁止错误建模：

```text
VPP_A 名义 PCC 在 bus5，但它聚合 bus20 储能和 bus30 光伏。
错误：P_VPP_A_at_bus5 = P_bus5 + P_bus20 + P_bus30
正确：bus5、bus20、bus30 分别真实注入；VPP_A 只是商业/控制主体。
```

### 2.2 显式区分单 PCC VPP 与多节点 VPP

```text
single_pcc VPP:
  园区、微网、充电站、工商业用户。
  DER 通常位于同一 PCC 后的本地电力系统。
  可使用单点 FR/DOE: (P_i^PCC, Q_i^PCC)。

multi_node VPP:
  居民侧 DER、EV、户用 PV/BESS、区域负荷聚合商。
  DER 分布在多个 bus/zone。
  必须使用 bus/zone/der-vector FR/DOE，不能用单个总 MW 替代网络安全校核。
```

所有 VPP 对象必须显式包含：

```text
physical_mode: single_pcc | multi_node
pcc_bus_id: optional for single_pcc
connection_buses: list[bus_id]
zone_ids: list[zone_id]
der_ids: list[der_id]
```

### 2.3 功率符号约定

项目内部统一：

```text
P > 0：向电网注入
P < 0：从电网吸收
```

pandapower 原生：

```text
load.p_mw > 0：负荷消费
sgen.p_mw > 0：发电注入
storage.p_mw > 0：储能充电吸收
storage.p_mw < 0：储能放电注入
```

储能转换：

```text
pp_storage_p_mw = -internal_grid_injection_p_mw
internal_grid_injection_p_mw = -pp_storage_p_mw
```

新增 DER、VPP、RL action、inner solver 输出必须测试符号转换。

### 2.4 智能体动作进入潮流前必须投影

```text
raw_action
  -> action_decoder
  -> device_constraints
  -> FR/DOE or DispatchAward projection
  -> sign conversion
  -> true bus injection
  -> pandapower.runpp
```

潮流不收敛、电压越限、线路/变压器过载是环境反馈，进入 reward/cost/log；只有 API、维度、依赖和数据结构错误才是程序异常。

---

## 3. 隐私边界与 CTDE

### 3.1 DSO 默认可见

```text
网络拓扑、参数、运行状态、线路/变压器约束
DER/VPP 接入位置和 zone 标签
VPP 申报能力、报价、响应曲线参数
实际响应、计量结果、历史履约率
```

### 3.2 DSO 默认不可见

```text
VPP 真实成本函数
用户舒适度和私有偏好
VPP 内部逐 DER 调度优先级
竞争性 VPP 私有策略和私有状态
```

如需使用上帝视角，命名必须包含：

```text
oracle / centralized / privileged / full_information_baseline
```

### 3.3 VPP_i 默认可见

```text
自身 DER 状态、成本、可用性、收益、偏差
DSO 下发给自身的 FR/DOE
DSO 下发给自身的 LFP/LFB / LocalFlexNeed / DispatchAward
自身允许可见的本地 bus/zone 标签
自身历史响应、收益、履约、惩罚
```

### 3.4 VPP_i 默认不可见

```text
完整拓扑
其他 VPP 成本、DER 状态、报价策略
全网安全裕度
oracle 解和 DSO 全局内部目标分解
```

### 3.5 CTDE 规则

允许 centralized training decentralized execution：

```text
critic 可使用全局状态和所有动作。
actor_i 执行时只能使用 local observation_i。
```

如果 actor_i 接收完整拓扑、其他 VPP 私有成本/状态、全局 oracle 解，视为 bug，除非实验明确标记 `oracle_actor`。

---

## 4. 双向状态表征：必须体现在接口中

### 4.1 DSO 侧节点/区域需求状态

DSO 应基于历史运行数据为每个 bus/zone 构造需求状态向量：

```text
NodeNeedState[z,t] = {
  voltage_deviation_history,
  voltage_violation_frequency,
  line_loading_history,
  transformer_loading_history,
  reverse_flow_pressure,
  curtailment_history,
  local_flex_call_history,
  unserved_or_resilience_event_history,
  forecast_load_pv_ev_features,
  time_calendar_features
}
```

对应编码器输出：

```text
node_or_zone_need_embedding[z,t]
```

含义：该节点/区域在当前电气环境下更需要哪类支撑，例如：

```text
voltage_high_mitigation
voltage_low_support
congestion_relief
reverse_flow_absorption
peak_shaving
resilience_support
reactive_power_support
```

### 4.2 DSO 侧 VPP 能力状态

DSO 不知道 VPP 内部真实成本，但可由历史响应识别 VPP 能力画像：

```text
VppCapabilityState[i,t] = {
  delivered_response_by_service_type,
  response_delay,
  non_delivery_rate,
  deviation_statistics,
  bid_history,
  settlement_history,
  location_effectiveness_history,
  reliability_score,
  inferred_service_preference
}
```

对应编码器输出：

```text
vpp_capability_embedding[i,t]
```

它用于 DSO 判断：

```text
VPP_i 更擅长支撑电压？缓解拥塞？吸收反向潮流？提供无功？保障韧性？
```

### 4.3 VPP 侧区域需求信念

VPP 应基于 DSO 下发历史学习区域电气需求，而不是直接读取完整拓扑：

```text
VppBeliefAboutGrid[i,t] = {
  historical_FR_shape_received,
  historical_LFP_or_bid_award_by_zone,
  LocalFlexNeed_history,
  DispatchAward_history,
  own_profit_and_penalty_history,
  own_delivery_success_history
}
```

对应输出：

```text
vpp_grid_need_belief[i,t]
```

含义：VPP_i 对“哪些位置/时段/服务类型更有价值”的隐式认知。

### 4.4 表征学习更新时机

表征模型不要在每个快循环 step 中随意在线抖动。默认：

```text
慢循环：每 N episodes / 每周 / 每月离线 fine-tune
中循环：使用最新 encoder 生成日前状态
快循环：只读 embedding，供 actor 或机制使用
```

可以后续扩展 online encoder，但必须单独标记实验并与离线版本对照。

优先方法：

```text
统计特征 baseline
监督标签学习
对比学习 encoder
序列编码器 RNN/Transformer
```

不默认使用 VAE。若使用生成式模型，必须解释为何重构目标不会学习无关细节。

---

## 5. 三时标架构：慢聚合、中日前、快调度

不要把设备签约、日前可行域、日内价格和调度响应塞进同一个 `env.step()`。

### 5.1 慢循环：DER 聚合组合与表征更新

周期：周、月、若干 episode 后。

职责：

```text
VPP 更新 DER -> VPP 的商业聚合关系
VPP 根据 DSO 需求历史、收益、惩罚和履约结果调整 portfolio
DSO 更新 VPP 能力画像
DSO/VPP 双向编码器离线 fine-tune
评估关键 DER 垄断、长期可靠性、资源浪费和市场力
```

输出：

```text
AggregationPlan
VPPPortfolio
EncoderVersion
LongTermReliability
MarketPowerMetric
PortfolioChangeLog
```

建议算法：

```text
小规模：穷举、启发式、局部搜索
中等规模：贝叶斯优化、进化策略、模拟退火
不默认用 RL 学大规模整数聚合动作
```

慢循环选择 DER 时仍必须保留 DER 真实 bus 注入位置。优化的是 `DER -> VPP`，不是 `DER -> PCC`。

### 5.2 中循环：日前 FR/DOE 与能力申报

周期：日前或日内滚动预测窗口。

职责：

```text
DSO 基于预测负荷/PV/EV、网络状态和安全裕度生成 FR/DOE
VPP 上报聚合能力、可用容量、报价参数或响应曲线
DSO 做安全校核并形成 DayAheadFlexNeed
```

输出：

```text
FRObject / DOEObject
VPPAbilityReport
DayAheadFlexNeed
RiskMargin
```

### 5.3 快循环：日内 LFP/LFB、MARL/HRL 调度与响应

周期：5 min、15 min、30 min 或 1 h；当前项目默认 15 min。

职责：

```text
DSO 根据当前网络状态生成 LocalFlexNeed 和 LFP/LFB
VPP actor 根据本地观测、FR/DOE、LFP/LFB 和自身目标响应
DSO 按资格、网络有效性、报价、可靠性和安全约束组合出清
VPP inner solver 将高层动作派发到 DER
pandapower 执行潮流
环境返回 reward、cost、结算、约束违反和解释标签
```

输出：

```text
LocalFlexNeed
VPPFlexBid
DispatchAward
VPPDispatchAction
PowerFlowResult
RewardComponents
SettlementRecord
ExplanationRecord
```

---

## 6. FR/DOE：动态功率可行域

FR/DOE 是安全硬边界，不是价格信号。

```text
ExecutableFeasibleSet = DeviceCapability ∩ VPPPortfolioCapability ∩ DSOSecurityFeasibleSet
```

### 6.1 单 PCC VPP

```text
FR_i,t = {(P_i^PCC, Q_i^PCC) | voltage/line/trafo/reverse-flow/device constraints}
```

可表示为：box、多面体、椭球、方向扫描凸包、OPF 投影近似。

### 6.2 多节点 VPP

```text
FR_i,t = {(P_i,b,t, Q_i,b,t) for b in connection_buses_i}
```

维度过高时可降维为分区级：

```text
FR_i,z,t = {(P_i,z,t, Q_i,z,t) for z where VPP_i has resources}
```

多节点 FR/DOE 不得退化成单个总功率上限，除非该上限只用于商业摘要，不用于网络安全校核。

### 6.3 生成方法版本

```text
v0 静态 box：额定容量、历史安全上限、固定裕度。
v1 灵敏度多面体：ΔV≈S_PΔP+S_QΔQ，ΔS≈H_PΔP+H_QΔQ，Ax≤b。
v2 OPF 投影：参数化 OPF / 方向扫描边界点 + 凸包/盒/椭球拟合。
v3 数据驱动近似：历史潮流和扰动响应学习 + 安全裕度 + oracle 对比。
```

所有版本必须支持：

```text
V_min + eps_v ≤ V ≤ V_max - eps_v
|S_l| ≤ S_l_max - eps_s
```

裕度覆盖预测误差、VPP 履约偏差、模型误差、通信延迟和线性化误差。

---

## 7. LFP/LFB：位置灵活性价格/报价

LFP/LFB 是经济软激励，不是普通电能价格，也不默认等同于 LMP。

普通电价回答：

```text
1 MWh 电能值多少钱？
```

位置灵活性价格/报价回答：

```text
某 VPP/DER 在某个配电位置、某个时段、某个方向提供 1 MW 或 1 MWh 调节能力，
对缓解电压越限、线路拥塞、变压器过载、反向潮流、弃光或韧性问题值多少钱？
```

推荐代码命名：

```text
local_flex_price
local_flex_bid
locational_flexibility_payment
network_service_price
network_service_adder
```

谨慎使用 `lmp`。若使用，必须注明 `wholesale_lmp`、`distribution_lmp` 或 `network_adder`。

### 7.1 LFP/LFB 建模层级

```text
v0 外生 LFP：local_flex_price[z,t] 来自场景参数。
v1 紧迫度驱动 LFP：base + α_v*voltage_stress + α_l*line_stress + α_tr*trafo_stress。
v2 VPP 报价出清 LFB：VPP 提交 bid，DSO 做组合出清。
v3 能量价格 + 网络服务附加费：energy_price + local_network_adder。
```

不要在基础版本中强制要求从 OPF 对偶变量直接得到 LMP。配电 OPF 对偶变量可作为高级解释性网络影子价格，但不是 v0/v1 的必要条件。

### 7.2 多 VPP 同时报价的出清规则

不是最低价者通吃。DSO 应按以下顺序处理：

```text
1. 资格筛选：VPP_i 是否位于 zone z、能提供服务类型 s、满足计量/通信/容量/响应时间。
2. 网络有效性：kappa_i,z,t 表征 VPP_i 对目标约束的缓解能力。
3. 报价与可靠性：effective_score_i = bid_price_i / max(eps, kappa_i * reliability_i)。
4. 组合出清：选择多个 VPP 的调用量 u_i，使 Σ kappa_i*u_i ≥ flex_need_z,t。
5. 结算与惩罚：按实际交付结算，未履约进入 non_delivery_penalty。
```

简化优化：

```text
minimize Σ bid_i*u_i + risk_penalty + unfairness_penalty
subject to:
  Σ kappa_i*u_i ≥ required_effective_flex
  0 ≤ u_i ≤ available_i
  u_i respects FR/DOE and network constraints
```

未中标 VPP 不得自行响应并获得补偿。过度响应应被惩罚或安全投影，否则可能造成新的电压/潮流问题。

### 7.3 DSO 不知道 VPP 真实成本

DSO 不需要知道 VPP 的真实成本函数。VPP 提交服务报价或响应曲线参数。

可简化为：

```text
bid_i = marginal_cost_proxy_i * (1 + markup_i) + risk_premium_i
```

或分段线性报价。策略性报价、市场力和反垄断约束是扩展实验，不要在最小闭环中同时引入。

---

## 8. 优化主体与目标函数

### 8.1 总体目标

顶层目标是通过机制使 VPP 个体理性行为逼近系统最优，而不是把 VPP 写成无私调度对象。

```text
min J_sys = C_operation + C_flex_procurement + C_loss + C_voltage_violation
          + C_line_overload + C_transformer_risk + C_curtailment
          + C_load_shedding + C_unfairness
          - B_hosting_capacity - B_resilience
```

### 8.2 DSO 目标

```text
min J_DSO = C_flex_procurement + C_loss + C_grid_violation
          + C_curtailment + C_shedding + C_risk + C_unfairness
```

DSO 决策：

```text
FR/DOE, LocalFlexNeed, LFP/LFB 或 clearing rule,
DispatchAward, risk_margin, fairness_weight, capability_label_update
```

### 8.3 VPP 目标

VPP 在 FR/DOE 内追求自身收益最大化：

```text
max Profit_i = energy_revenue_i + flex_service_revenue_i + locational_payment_i
             - operation_cost_i - battery_degradation_i - comfort_loss_i
             - imbalance_penalty_i - non_delivery_penalty_i - contract_cost_i
```

VPP actor 不应默认直接输出每台 DER 出力。优先输出高层策略参数：

```text
response_curve_params
price_sensitivity
reserve_preference
der_group_weight
risk_aversion
bid_markup
portfolio_update_preference
```

再由 `vpp_inner_solver` 结合 FR/DOE、LFP/LFB、DispatchAward 和设备约束求实际 DER 派发。

---

## 9. 多智能体/分层强化学习设计边界

### 9.1 推荐训练结构

```text
慢循环：聚合组合搜索 + encoder fine-tune，不默认用 RL 学整数动作。
快循环：MARL/HRL 学日内调度、报价响应或高层策略参数。
中循环：可先用规则/优化生成 FR/DOE，不急于训练 DSO agent。
```

### 9.2 最小可行闭环优先级

不要一开始同时训练 DSO 和多个 VPP。推荐顺序：

```text
Stage A: DSO 固定规则生成 FR v0 + 外生 LFP，单 VPP 响应。
Stage B: 多 VPP 在 FR 内响应，VPP actor 本地观测，DSO 规则出清。
Stage C: 加入 LFB 报价、组合出清和结算。
Stage D: 加入 VPP 能力画像和节点需求画像。
Stage E: 加入慢循环 DER 聚合组合更新。
Stage F: 再考虑 DSO agent 学习价格、风险裕度或公平权重。
```

### 9.3 Observation 规则

Actor_i 观测可含：

```text
自身 DER 聚合状态
自身可用容量/SOC/舒适度/成本代理
自身收到的 FR/DOE
自身收到的 LFP/LFB/LocalFlexNeed/DispatchAward
自身历史收益、偏差、履约结果
自身 vpp_grid_need_belief
时间特征和本地 profile
```

Actor_i 不可含：

```text
完整拓扑
其他 VPP 私有成本/DER 状态
oracle centralized OPF 解
DSO 全局内部目标分解
```

Critic 可含全局状态，但要通过接口明确区分：

```text
actor_observation_i
critic_global_state
```

---

## 10. 通信 schema：核心路径禁止散乱 dict

优先使用 dataclass；需要强校验再使用 pydantic。所有 schema 都应支持序列化到 CSV/JSON 以便实验回放。

```text
DERSpec:
  der_id, der_type, bus_id, phase, pp_element_type, pp_element_index,
  p_min_mw, p_max_mw, q_min_mvar, q_max_mvar,
  energy_capacity_mwh?, soc?, owner_vpp_id?, controllable

VPPPortfolio:
  vpp_id, physical_mode, pcc_bus_id?, connection_buses, zone_ids, der_ids,
  max_import_mw?, max_export_mw?, portfolio_version

FRObject:
  fr_id, vpp_id, time_window, scope[pcc,bus_vector,zone_vector,der_vector],
  representation[box,polytope,ellipsoid,convex_hull,sampled_boundary],
  variables, constraints, safety_margin, valid_until, source_method, encoder_version?

LocalFlexNeed:
  need_id, zone_id, target_constraint[voltage_high,voltage_low,line_overload,transformer_overload,reverse_flow,resilience],
  direction[up,down,absorb_p,inject_p,absorb_q,inject_q],
  required_effective_mw_or_mvar, start_time, duration_min, response_time_min, severity

VPPFlexBid:
  bid_id, vpp_id, portfolio_version, zone_id, direction,
  quantity_mw_or_mvar, duration_min, response_time_min,
  price, price_unit, reliability, location_effectiveness?

DispatchAward:
  award_id, vpp_id, need_id, awarded_quantity, settlement_price,
  expected_effective_contribution, dispatch_instruction, valid_time_window

MeasurementReport:
  report_id, vpp_id, time, delivered_by_bus_or_zone, deviation,
  voltage_violations, line_violations, non_delivery_penalty

ExplanationRecord:
  step, vpp_id, support_type_label, reason, DSO_signal_seen,
  VPP_response, network_effect, settlement, reliability_update
```

---

## 11. Repository map 与推荐新增模块

当前核心包：`src/vpp_dso_sim/`。不要默认新建平行大目录如 `src/grid/` 或 `src/assets/`。新增功能优先落到现有结构：

```text
network/        网络构建、潮流、约束、灵敏度接口
DER/der/        PV、MT、ESS、Flex、HVAC、EV、EVCS 模型
entities/       DSO、VPP、PCC、market 等逻辑主体
optimization/   聚合、分解、成本、安全投影、baseline、FR/LFP/clearing/inner solver
envs/           Gym/MARL/CTDE/HRL 接口
learning/       可选：节点需求/VPP 能力编码器、课程、训练日志
simulation/     scenario、profiles、Simulator 主循环、结果导出
visualization/  dashboard 数据、拓扑图、Plotly 报告
dashboard/      只读 Dash dashboard
memory/         项目长期记忆
```

推荐新增/扩展：

```text
optimization/feasibility_region.py     # FR/DOE v0-v3
optimization/local_flex_market.py      # LFP/LFB、LocalFlexNeed、VPPFlexBid、DispatchAward
optimization/inner_solver.py           # VPP 在 FR 内收益最大化求解器
optimization/portfolio_search.py       # 慢循环 DER->VPP 聚合搜索
optimization/oracle_baseline.py        # 集中式 OPF / full-information baseline
envs/observations.py                   # actor/critic observation 与隐私过滤
envs/rewards.py                        # reward/cost/settlement 分量
envs/multi_agent_env.py                # PettingZoo/RLlib/CTDE 接口
learning/encoders.py                   # NodeNeedEncoder, VppCapabilityEncoder
learning/capability_labeler.py         # VPP 支撑能力标签
learning/curriculum.py                 # 训练课程
memory/*.md                            # 长期记忆
```

注意：若当前包名大小写已有约定，必须遵守现有目录实际名称，不凭本文大小写重命名。

---

## 12. Harness engineering：subagents

Codex 在内部应按以下角色切换；不需要真的创建多个进程，但产出要体现角色职责。

```text
Architect:
  负责模块边界、schema、三时标接口、技术路线。

DomainGuardian:
  负责配电网、VPP、FR/DOE、LFP/LFB、pandapower 符号和物理一致性审查。

AgentEngineer:
  负责 Gym/MARL/HRL/CTDE observation、action、reward、policy 接口。

MarketMechanismEngineer:
  负责 LFP/LFB、报价、组合出清、结算、偏差惩罚和公平性。

SimulatorEngineer:
  负责 pandapower 写入、runpp、结果抽取、时序闭环和测试。

MemoryKeeper:
  负责 concepts/decisions/rules/pitfalls/progress/open_questions/experiments。

Critic:
  负责指出正确性、隐私、物理一致性、研究漂移、可解释性和可测试性问题。
```

重大架构改动要经过：

```text
Architect -> DomainGuardian -> Critic -> Implementer
```

---

## 13. Harness engineering：hooks

### 13.1 boot hook

每次接到任务先执行：

```text
1. 读 AGENTS.md。
2. 读 README.md。
3. 读与任务相关的 memory 文件。
4. 浅扫 pyproject.toml、configs、src/vpp_dso_sim、tests。
5. 判断任务属于 research-design / schema / implementation / test / experiment / memory。
6. 给出最小计划；低风险实现可直接执行，高风险先确认。
```

### 13.2 pre-write hook

写代码或创建文件前检查：

```text
是否破坏 DER 真实 bus 注入？
是否把多节点 VPP 压成单 PCC 总功率？
是否让 VPP actor 看见不该看的全局/其他 VPP 私有信息？
是否把 LFP 等同于 LMP？
是否把 VPP 写成无私服从 DSO？
是否跳过 FR/DOE 或安全投影直接写 pandapower？
是否需要新增或更新测试？
```

### 13.3 post-write hook

写完后检查：

```text
运行最小相关测试。
更新 memory/progress.md。
若改了架构或 schema，写入 memory/decisions.md。
若发现领域错误或幻觉，写入 memory/pitfalls.md。
```

### 13.4 research-drift hook

每次较大改动后问：

```text
这个改动是否强化了“双向引导 + 三时标 + 隐私 + 物理一致性”？
还是只是在做普通仿真可视化、普通 RL 或普通 OPF？
```

若偏离，先暂停并说明。

### 13.5 privacy hook

任何 DSO↔VPP 通信 schema 变更都必须列字段可见性：

```text
field | visible_to_DSO | visible_to_VPP_i | visible_to_other_VPP | oracle_only
```

### 13.6 physical-consistency hook

任何涉及聚合、动作、FR、出清或潮流写入的变更都必须验证：

```text
DER setpoint writes to true bus/pp element.
VPP total power is not used as fake physical injection unless single_pcc.
Multi-node VPP report includes bus/zone distribution.
```

---

## 14. Harness engineering：skills

### skill: repo-orientation

触发：新会话、任务不清、代码状态需要盘点。

产出：

```text
已实现 / 半实现 / 未实现
相关文件清单
不应重写的稳定模块
本次最小改造点
```

### skill: bidirectional-guidance-loop-build

触发：实现主闭环时。

步骤：

```text
1. DSO 生成 FR/DOE。
2. DSO 生成 LocalFlexNeed 或 local_flex_price。
3. VPP 根据本地观测生成响应/报价。
4. DSO 出清/调用。
5. VPP inner solver 派发 DER。
6. pandapower 跑潮流。
7. 记录响应、结算、约束与能力标签。
8. 更新可用于 encoder 的 trajectory buffer。
```

### skill: feasibility-region-compute

触发：中循环日前 FR/DOE。

版本：

```text
v0: static box
v1: sensitivity polytope
v2: OPF projection
v3: data-driven + safety margin
```

接口建议：

```text
compute_feasible_region(vpp: VPPPortfolio, window, grid_state, method) -> FRObject
```

### skill: local-flex-clearing

触发：日内 LFP/LFB 或多个 VPP 同时报价。

步骤：

```text
1. build LocalFlexNeed from network stress.
2. collect VPPFlexBid from qualified VPPs.
3. compute kappa/location_effectiveness.
4. solve least effective-cost combination.
5. emit DispatchAward.
6. log settlement and non-delivery.
```

### skill: vpp-inner-solve

触发：VPP 收到 FR/DOE、LFP/LFB、DispatchAward 后。

目标：

```text
max Profit_i subject to DER constraints, FR/DOE, award, SOC, comfort, ramp, duration.
```

v0 可使用规则/启发式；后续可升级到 scipy/cvxpy/MILP。

### skill: response-capability-inference

触发：每次 episode 或一段时间结束。

产出：

```text
VPP_i 对 voltage_high / voltage_low / congestion / reverse_flow / reactive_support / resilience 的支撑能力标签。
```

输入：

```text
requested service, bid, award, delivered response, network effect, deviation, reliability.
```

### skill: bidirectional-encoder-train

触发：慢循环。

步骤：

```text
1. 从 trajectory buffer 读取历史网络状态、FR、LFP、VPP 响应、结算和约束结果。
2. 训练/更新 NodeNeedEncoder。
3. 训练/更新 VppCapabilityEncoder。
4. 训练/更新 VPP 侧 GridNeedBeliefEncoder。
5. 保存 encoder version。
6. 记录到 memory/experiments.md。
```

### skill: aggregation-search

触发：慢循环 DER 组合更新。

约束：

```text
改变 owner_vpp_id / portfolio membership。
不改变 bus_id / pp element / physical injection.
每次更新必须形成 portfolio_version。
```

默认算法：启发式、局部搜索、贝叶斯优化、进化策略。不要默认直接用 RL 学大规模整数动作。

### skill: marl-interface-build

触发：接入 MARL/HRL。

必须分离：

```text
actor_observation_i
critic_global_state
action_decoder
safety_projection
reward_components
info/debug dict
```

### skill: oracle-gap-eval

触发：阶段性评估、论文结果、重大改动后。

任务：

```text
跑 centralized/full-information/oracle baseline。
比较 distributed bidirectional framework 的 cost gap、constraint violation gap、profit/fairness gap。
```

没有 oracle baseline，不要宣称“最优”。只能说“相对某些基线改进”。

### skill: privacy-audit

触发：任何通信 schema、observation、critic/actor 改动。

输出字段可见性审计表。

---

## 15. Memory：长期自进化项目记忆

若 `memory/` 不存在，创建：

```text
memory/concepts.md       # 概念、术语、符号、公式、定义
memory/decisions.md      # ADR：架构/算法/实验口径决策
memory/rules.md          # 工程规则、命名、测试、依赖
memory/pitfalls.md       # 幻觉、方向偏移、踩坑、错误假设
memory/progress.md       # 时间倒序进度日志
memory/open_questions.md # 未决问题，按优先级
memory/experiments.md    # 实验、超参、结果、是否采纳
```

写入触发：

```text
新概念 -> concepts.md
关键决策 -> decisions.md
新工程约定 -> rules.md
错误/幻觉/方向偏移 -> pitfalls.md
任务完成 -> progress.md
新未决问题 -> open_questions.md
实验运行 -> experiments.md
```

Memory 原则：

```text
只增补，不无声重写历史。
过时条目标注 SUPERSEDED。
所有 ADR 写背景、选项、选择、放弃理由、回滚条件。
```

必记概念包括：

```text
FR/DOE
LFP/LFB
LocalFlexNeed
DispatchAward
NodeNeedEmbedding
VppCapabilityEmbedding
VppGridNeedBelief
single_pcc vs multi_node VPP
DER physical bus vs VPP commercial ownership
CTDE privacy boundary
```

---

## 16. 测试与验收

新增模块至少具备以下测试之一：

```text
schema serialization/deserialization test
sign convention test
single_pcc vs multi_node physical injection test
FR/DOE scope and bounds test
LFP/LFB clearing test
privacy observation filter test
VPP inner solver feasibility test
env smoke test
oracle/baseline comparison smoke test
```

核心验收指标：

```text
物理安全：电压越限次数/幅度、线路过载、变压器风险、潮流收敛率。
经济性：DSO 采购成本、VPP 利润、结算金额、偏差惩罚。
承载力：弃光减少、反向潮流吸收、新 DER 可接入能力 proxy。
韧性：关键负荷保障、故障/扰动场景下供电能力 proxy。
公平性：VPP 调用频率、收益分布、削减比例、关键节点垄断指标。
隐私性：actor 可见字段审计、DSO 不读取 VPP 真实成本。
最优性：与 oracle centralized baseline 的 gap。
学习性：NodeNeed/VppCapability/VppGridBelief 表征是否提高响应质量。
```

---

## 17. 推荐开发路线

按以下顺序推进，不要跳到复杂版本：

```text
Phase 1: Repo 盘点 + memory 初始化
  产出：已实现/未实现清单，写入 progress.md。

Phase 2: Schema 与物理一致性
  产出：DERSpec, VPPPortfolio, FRObject, LocalFlexNeed, VPPFlexBid, DispatchAward。
  测试：多节点 VPP 不把远端 DER 注入 fake PCC。

Phase 3: FR/DOE v0
  产出：single_pcc box FR + multi_node bus/zone box FR。
  测试：action projection 后不越设备 bounds。

Phase 4: LFP/LFB v0-v1
  产出：外生 local_flex_price + 紧迫度驱动 local_flex_price。
  测试：VPP reward 中出现 flex_service_revenue。

Phase 5: VPP inner solver v0
  产出：VPP 在 FR 内根据 LFP 最大化收益的启发式/凸求解。
  测试：可行、符号正确、SOC/comfort 不破坏。

Phase 6: 多 VPP 组合出清
  产出：资格筛选 + kappa + effective_score + DispatchAward。
  测试：低价但无效 VPP 不必然中标；多个 VPP 可共同中标。

Phase 7: MARL/HRL 快循环
  产出：actor_observation_i, critic_global_state, action_decoder, reward_components。
  测试：actor 隐私过滤，critic 可见全局但不泄漏到 actor。

Phase 8: 双向表征与能力画像
  产出：NodeNeedEncoder, VppCapabilityEncoder, VppGridNeedBeliefEncoder, ExplanationRecord。
  测试：trajectory buffer 可回放，标签与响应类型一致。

Phase 9: 慢循环聚合更新
  产出：portfolio_search, AggregationPlan, portfolio_version。
  测试：改变 owner_vpp_id，不改变 bus_id；组合更新周期与快循环解耦。

Phase 10: Oracle baseline 与论文指标
  产出：centralized/full-information baseline，gap 评估。
```

---

## 18. 反幻觉与方向偏移红线

高风险断言必须核验本地代码、文档或用户提供材料：

```text
pandapower API 和符号
Gym/PettingZoo/RLlib/SB3/torch 版本相关接口
配电网市场机制的制度性描述
标准条款与政策细节
```

禁止输出以下未经验证或不符合研究方向的内容：

```text
“DSO 直接知道 VPP 真实成本。”
“VPP 可以把任意 bus 的 DER 等效到自己的 PCC。”
“多节点 VPP 的动态可行域就是总 P 上限。”
“LFP 就是 LMP。”
“谁报价最低谁一定中标。”
“RL 动作可以直接写入 pandapower，不需要安全投影。”
“没有 oracle baseline 也可称总体最优。”
```

---

## 19. 交付风格

每次代码任务交付时说明：

```text
修改了哪些文件
为什么这些修改服务于双向引导主线
是否触碰 FR/DOE、LFP/LFB、聚合更新、表征学习、MARL/HRL、隐私边界
运行了哪些测试
memory 更新情况
下一步最小切片
```

默认中文回复。术语首次出现给中英对照。对研究方向有冲突时，先指出冲突，不要为了“能实现”而默默改研究问题。
