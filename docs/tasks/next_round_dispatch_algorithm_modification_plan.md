# 新一轮算法修改意见：Dispatch Action Landing、Projection Policy 与 Dispatch Actor 结构升级

**版本**：v0.1
**日期**：2026-06-11
**依据**：`dispatch_reward_settlement_actor_diagnosis_20260611.md`

---

## 0. 诊断结论转化为算法修改目标

本轮修改不应优先从“继续加深网络”开始，而应先修复两个控制闭环问题：

1. **reward / settlement 口径必须可解释。**
   旧实验 episode 0015 中，`private_profit_proxy` 总和为 `-2,876,729.16`，而可见电费运行差额只有 `-2,953.25`，二者残差达到 `-2,873,775.91`；同时残差高度集中在带 HVAC 的 VPP，说明旧口径很可能把 raw comfort / unserved service-quality penalty 直接混入经济利润，造成量纲混用。fileciteturn17file2

2. **dispatch 动作必须真实落地。**
   旧 episode 中 `accepted_delta_p_mw` 非零比例为 `6.68%`，但 `actual_delta_p_mw` 非零比例为 `0.00%`，最大绝对值也是 `0.0`。这意味着 actor 即使输出了动作，最终进入环境的实际功率变化仍然为 0；在这种情况下，actor 无法从 reward 中学到有效因果控制。fileciteturn17file2

因此，新一轮修改的主目标是：

```text
先保证 actor 输出动作能够真实改变 DER/VPP 执行功率；
再保证 reward 能被 settlement 分项解释；
最后再升级 dispatch agent 网络结构。
```

---

## 1. 是否应该只干预影响 AC 潮流的动作？

### 1.1 结论

**不应简单表述为“只干预影响 AC 潮流的动作”。更准确的原则是：**

```text
只允许对“物理不可执行”和“AC 安全紧急风险”做硬干预；
对 comfort、contract、service-quality、preferred target、tracking、非硬性 envelope preference 等约束，尽量不做 post-hoc projection，而是放入 reward penalty / constraint penalty / auxiliary loss。
```

原因是：

- 所有 DER 的有功/无功注入最终都会影响 AC 潮流；
- 但不是所有约束都应该通过 projection 覆盖 actor 动作；
- 旧实验的 `actual_delta_p_mw = 0` 说明后处理链路可能把 actor 动作抹掉了；
- 如果大量非 AC 约束都通过 projection 纠正，actor 会看到“我输出了动作，但环境执行为 0”的反馈，导致 credit assignment 断裂。

---

## 2. 新的动作干预原则：三层约束体系

建议把所有约束分为三类。

---

### 2.1 第一类：硬物理约束，必须保证，但不应通过隐藏 projection 实现

这类约束包括：

| 约束 | 示例 | 处理方式 |
|---|---|---|
| DER 功率上下限 | PV 不能负发电；EVCS 不能放电；储能充放电限值 | 用 action parameterization 保证输出落在可行域 |
| SOC 硬边界 | 储能 SOC 不得小于 0 或大于 1 | 用可行功率区间动态缩放动作 |
| ramp 物理边界 | 微型燃机爬坡限制 | 用动作解码器内置 ramp-aware bound |
| 设备在线状态 | EV 未连接时不能充电 | 用 action mask |

**关键原则：**

```text
不要先让 actor 输出非法动作，再通过 projection 抹掉；
而是让 actor 的动作参数化天然落在设备硬物理可行域内。
```

例如储能：

```math
u^{ESS}_t \in [-1, 1]
```

动态映射为：

```math
P^{ESS,target}_t =
\begin{cases}
u_t P^{dis,max}_t, & u_t \ge 0 \\
u_t P^{ch,max}_t, & u_t < 0
\end{cases}
```

其中：

```math
P^{dis,max}_t
=
\min(P^{rated,dis}, \frac{SOC_t - SOC^{min}}{\Delta t})
```

```math
P^{ch,max}_t
=
\min(P^{rated,ch}, \frac{SOC^{max} - SOC_t}{\Delta t})
```

这样 actor 输出的目标天然不会因为 SOC 物理边界被后处理改成 0。

---

### 2.2 第二类：AC 安全约束，允许 emergency intervention

这类约束包括：

| 约束 | 示例 | 处理方式 |
|---|---|---|
| 电压越限 | bus voltage 超出上下限 | AC shield 可硬干预 |
| 线路过载 | line loading > 100% | AC shield 可硬干预 |
| 变压器过载 | transformer loading > 100% | AC shield 可硬干预 |
| 潮流不收敛 | pandapower power flow failed | AC shield 可硬干预 |

这里可以保留 safety shield，但必须满足三个要求：

1. **只在 raw action 可能导致 AC 不安全时干预；**
2. **必须记录 raw action safety cost、projected action safety cost、projection gap；**
3. **必须在 reward 中惩罚 raw action 到 projected action 的差距。**

建议定义：

```math
C^{AC,raw}_t =
C^{V,raw}_t + C^{L,raw}_t + C^{T,raw}_t + C^{PF,raw}_t
```

```math
C^{AC,proj}_t =
C^{V,proj}_t + C^{L,proj}_t + C^{T,proj}_t + C^{PF,proj}_t
```

```math
C^{AC,gap}_t =
\frac{\|a^{exec}_t - a^{phys}_t\|_1}
{\|a^{span}_t\|_1 + \epsilon}
```

其中：

- `a_phys` 是经过设备物理参数化后的动作；
- `a_exec` 是 AC shield 后实际执行动作；
- `a_span` 是设备可调跨度。

然后在 dispatch reward 中加：

```math
r^{dispatch}_t
\leftarrow
r^{dispatch}_t
-
\lambda_{AC,raw} C^{AC,raw}_t
-
\lambda_{AC,gap} C^{AC,gap}_t
```

这样 shield 仍然保护系统，但 actor 会知道：

```text
我的原始动作离 AC-safe 可执行动作有多远。
```

---

### 2.3 第三类：非 AC 软约束，不应硬投影

这类约束包括：

| 约束 | 示例 | 推荐处理方式 |
|---|---|---|
| comfort | HVAC 室温偏离设定值 | service-quality penalty |
| unserved energy | EV 离站时未满足充电需求 | deadline / unserved penalty |
| contract tracking | 没有交付承诺服务量 | contract penalty |
| preferred target | 没贴近 DSO preferred midpoint | tracking penalty，不要 hard clamp |
| envelope midpoint preference | DSO 偏好点 | 只作为 guidance，不应覆盖动作 |
| dispatch smoothness | 动作变化太快 | smoothness penalty |
| SoC target | 期望末端 SOC | terminal target penalty |

这些约束不应该在环境执行前把动作直接改成 0。
否则会出现旧 episode 中的问题：

```text
actor 输出动作 → 被非 AC projection / baseline override 抹掉 → actual_delta_p_mw = 0
```

推荐做法是：

```math
r^{dispatch}_t
=
\lambda_{\Pi} \Pi^{private,economic}_{i,t}
+
F^{storage}_t
+
I^{terminal}_t R^{storage,T}_{i,t}
-
\lambda_{quality} C^{quality,norm}_{i,t}
-
\lambda_{contract} C^{contract,norm}_{i,t}
-
\lambda_{smooth} C^{smooth}_{i,t}
-
\lambda_{AC,gap} C^{AC,gap}_{i,t}
```

其中：

```math
C^{quality,norm}_{i,t}
=
normalize(C^{comfort}_{i,t} + C^{unserved}_{i,t})
```

注意：不要再把 raw comfort / unserved 直接混进 economic profit。诊断报告已经显示旧 private profit 与可见电费运行差额存在巨额残差，且集中在带 HVAC 的 VPP；新的 settlement 已经把经济利润和服务质量惩罚拆开。fileciteturn17file5

---

## 3. 建议的新动作执行链路

### 3.1 旧链路的问题

旧链路大概率类似：

```text
actor raw action
→ local bounds projection
→ DOE / envelope projection
→ AC projection
→ baseline / actual execution
→ actual_delta_p_mw = 0
```

问题是 actor 不知道动作在哪一级被抹掉。

---

### 3.2 新链路建议

改成：

```text
actor latent action z
→ differentiable physical action decoder
→ device-feasible target a_phys
→ optional DSO envelope guidance penalty
→ AC shield only if raw AC unsafe
→ a_exec
→ DER state update
→ settlement + reward
```

关键点：

1. **设备物理可行性由 decoder 保证，不靠隐藏 projection。**
2. **非 AC 约束进入 reward，不覆盖动作。**
3. **AC shield 是唯一主要 post-hoc hard intervention。**
4. **所有动作差异都要落盘。**

---

## 4. 必须新增的 action landing 审计字段

下一轮训练必须逐 step、逐 VPP、逐 DER 记录：

```text
raw_action_norm
decoded_target_p_mw
device_feasible_target_p_mw
pre_ac_target_p_mw
ac_projected_target_p_mw
actual_target_p_mw
baseline_p_mw
raw_delta_p_mw
decoded_delta_p_mw
device_feasible_delta_p_mw
pre_ac_delta_p_mw
ac_projected_delta_p_mw
actual_delta_p_mw
raw_to_device_gap_mw
device_to_ac_gap_mw
ac_to_actual_gap_mw
accepted_to_actual_gap_mw
actual_delta_nonzero_flag
actual_delta_nonzero_rate
action_landing_ratio
```

其中：

```math
action\_landing\_ratio =
\frac{|actual\_delta\_p\_mw|}
{|decoded\_delta\_p\_mw| + \epsilon}
```

如果：

```text
decoded_delta_p_mw 非零，但 actual_delta_p_mw 长期为 0
```

则必须自动生成 attribution：

```text
drop_reason =
local_physical_limit
| dso_envelope_clip
| ac_shield_projection
| soc_limit
| ramp_limit
| comfort_override
| baseline_override
| not_applied_bug
| logging_bug
```

---

## 5. 训练前硬性验收条件

在继续 paper-long 前，建议增加以下 gate：

```text
1. private_profit_vs_visible_energy_residual_ratio < 5%
2. actual_delta_nonzero_rate > 10%   # smoke 阶段先设低门槛
3. action_landing_ratio_mean > 0.3
4. baseline_override_rate < 1%
5. non_ac_projection_zeroing_rate < 5%
6. ac_shield_projection_rate 可以非零，但必须有 raw/proj gap penalty
7. settlement_audit_complete = 1
8. settlement_power_balance_ok = 1
```

如果第 2 条不满足，说明 dispatch actor 仍然没有真实控制权，不能进入长训练。

---

## 6. Dispatch reward 新建议

### 6.1 不再把 raw comfort / unserved 混入 economic profit

诊断报告已经说明，旧 trace 中 private profit 与 visible energy margin 之间的巨大残差高度集中在带 HVAC 的 VPP，最可信原因是 raw comfort / unserved penalty 混入经济利润。fileciteturn17file3

建议：

```math
\Pi^{economic}_{i,t}
=
R^{PV,export}
+
R^{MT,export}
+
R^{ESS,discharge}
+
R^{EVCS,user}
-
C^{EVCS,wholesale}
-
C^{ESS,charge}
-
C^{HVAC,energy}
-
C^{flex,energy}
-
C^{DER,op}
-
C^{deg}
```

服务质量惩罚单独进入：

```math
C^{quality}_{i,t}
=
C^{comfort,norm}_{i,t}
+
C^{unserved,norm}_{i,t}
```

dispatch reward：

```math
r^{dispatch}_{i,t}
=
\lambda_{\Pi}\Pi^{private,economic}_{i,t}
+
\lambda_{storage}F^{storage}_{i,t}
+
I^{terminal}_t R^{storage,T}_{i,t}
-
\lambda_{quality} C^{quality}_{i,t}
-
\lambda_{contract} C^{contract}_{i,t}
-
\lambda_{landing} C^{landing}_{i,t}
-
\lambda_{AC} C^{AC,gap}_{i,t}
```

其中：

```math
C^{landing}_{i,t}
=
\frac{\|a^{exec}_{i,t} - a^{decoded}_{i,t}\|_1}
{\|a^{span}_{i,t}\|_1+\epsilon}
```

---

## 7. Dispatch Agent 网络结构修改建议

诊断报告指出，当前 dispatch actor 使用 VPP context encoder、DER token encoder、masked mean/max pooling、aggregate action head 和 DER action head，不是玩具 MLP；但它仍然是 DeepSet 聚合，不建模电气拓扑和长时序，也难以表达异质 VPP 结构差异。fileciteturn17file1

因此，网络升级应在 action landing 修复之后进行。

---

# 7.1 新结构名称建议

建议命名为：

```text
VPPTemporalSensitivityDispatchActor
```

或者：

```text
TopologyAwareTemporalDispatchActor
```

核心思想：

```text
DER token + VPP context + DSO envelope + market forecast + grid sensitivity + temporal state
→ type-specific transformer / attention encoder
→ hierarchical aggregate-to-DER action decoder
→ physical-feasible action parameterization
```

---

## 7.2 输入结构升级

### VPP-level context

```text
current_vpp_p_mw
baseline_p_mw
dso_envelope_p_min_mw
dso_envelope_p_max_mw
preferred_target_p_mw
price_buy
price_sell
evcs_retail_tariff
time_of_day_sin/cos
day_type
recent_actual_delta_p_mw
recent_action_landing_ratio
recent_projection_gap
```

### DER token features

每个 DER 一个 token：

```text
der_type_onehot
bus_id_embedding
zone_id_embedding
p_current
p_min_dynamic
p_max_dynamic
p_baseline
soc
soc_target
time_to_deadline
required_energy_remaining
pv_available
hvac_temp_deviation
comfort_deadband
ramp_up_limit
ramp_down_limit
operation_cost_coeff
battery_degradation_coeff
```

### Grid sensitivity features

每个 DER token 增加：

```text
dV_dP_topk
dLineLoading_dP_topk
dTrafoLoading_dP
dLoss_dP
electrical_distance_to_pcc
reverseflow_sensitivity
voltage_risk_score
congestion_risk_score
```

这些可以来自线性灵敏度、Jacobian、historical finite difference 或已有 sensitivity attention 模块。

### Temporal / forecast features

建议至少加入 8 到 16 个 step 的短窗：

```text
price_forecast_buy[0:K]
price_forecast_sell[0:K]
pv_forecast[0:K]
load_forecast[0:K]
ev_arrival_departure_pressure[0:K]
voltage_margin_forecast_proxy[0:K]
```

K 建议先取：

```text
K = 8 或 16
```

因为 15 分钟 step 下，K=8 代表 2 小时，K=16 代表 4 小时。

---

## 7.3 Encoder 结构建议

### 当前 DeepSet 的问题

DeepSet mean/max pooling 会丢失 DER 之间的互斥和互补关系，例如：

```text
储能充电 vs EVCS 充电
PV 出力 vs 储能充电
HVAC 舒适度 vs 电价高峰
微型燃机爬坡 vs 反向潮流
```

### 推荐结构

```text
DER type-specific embedding
→ self-attention / Set Transformer block
→ cross-attention with VPP context token
→ cross-attention with grid sensitivity token
→ temporal GRU / Transformer encoder
→ action heads
```

一个可实施的中等复杂版本：

```python
DERTokenEncoder
  → TypeSpecificLinear
  → SetTransformerBlock(num_heads=4, layers=2)
  → ContextCrossAttention(VPPContextToken)
  → TemporalGRU(hidden=128, window=8)
  → Heads
```

---

## 7.4 动作头设计：必须设备类型专用

不要让所有 DER 共用一个 generic normalized P head。建议拆成：

| DER 类型 | 动作头 |
|---|---|
| PV | `curtailment_fraction ∈ [0,1]`，可选 `q_support_fraction ∈ [-1,1]` |
| Storage | `charge_discharge_mode logits + magnitude` 或 `signed_power ∈ [-1,1]` |
| EVCS | `charging_fraction ∈ [0,1]`，受连接 EV、deadline、requested energy mask 约束 |
| HVAC | `load_adjustment_fraction ∈ [0,1]` 或 setpoint adjustment |
| Flexible load | `curtail_or_shift_fraction ∈ [0,1]` |
| Microturbine | `p_setpoint_fraction ∈ [0,1]`，带 ramp-aware decoder |
| Inverter DER | 后续加 `q_fraction ∈ [-1,1]` |

其中所有动作都应通过 **physical-feasible decoder** 映射到 MW，而不是后处理 projection。

---

## 7.5 层级式动作结构

建议使用：

```text
aggregate head + allocation head
```

但不要让 aggregate head 和 DER heads 互相矛盾。

### Step 1：actor 输出 VPP aggregate target

```math
P^{agg,target}_{i,t}
\in
[P^{phys,min}_{i,t}, P^{phys,max}_{i,t}]
```

### Step 2：DER allocation head 分配功率

```math
P^{DER,target}_{j,t}
=
Allocator(P^{agg,target}_{i,t}, DERTokens)
```

### Step 3：做 power balance repair，但不是硬覆盖为 0

如果：

```math
\sum_j P^{DER,target}_{j,t}
\neq P^{agg,target}_{i,t}
```

使用 differentiable allocator 或 proportional scaling，让差额在可调 DER 中分配。

记录：

```text
aggregate_to_der_allocation_gap_mw
allocation_repair_gap_mw
```

---

## 7.6 Critic 输入也要升级

Dispatch critic 需要看到 actor 看不到但训练期可用的全局信息。建议 centralized critic 输入包括：

```text
VPP private settlement components
economic_operational_surplus
service_quality_penalty_total
actual_delta_p_mw
action_landing_ratio
raw/proj AC safety cost
DSO envelope
top-k voltage margin
top-k line loading
time/price/PV/load forecast
```

尤其要包含：

```text
actual_delta_p_mw
accepted_to_actual_gap_mw
```

否则 critic 仍然会把 reward 归因给 raw actor action，而不是实际执行动作。

---

## 8. 最小实现路线

建议不要一次性上 GNN + Transformer + Q 控制。按下面顺序执行。

---

### Phase 1：动作落地与投影策略修复

目标：

```text
actual_delta_p_mw 不再长期为 0。
```

修改：

1. 引入 physical-feasible action decoder；
2. 移除非 AC 约束的硬 projection；
3. 保留 AC shield，但记录并惩罚 AC projection gap；
4. 添加 action landing audit；
5. 添加 fail gate。

验收：

```text
actual_delta_nonzero_rate > 10%
action_landing_ratio_mean > 0.3
accepted_to_actual_gap_mw 可解释
```

---

### Phase 2：reward / settlement 完整闭环

目标：

```text
private_profit_vs_visible_energy_residual 接近 0。
```

修改：

1. 使用 `economic_operational_surplus` 作为经济主项；
2. 服务质量惩罚归一化并单独进入 reward；
3. settlement fields 全量落盘；
4. residual ratio 每 episode 报告。

验收：

```text
private_profit_vs_visible_energy_residual_ratio < 5%
settlement_audit_complete = 1
settlement_power_balance_ok = 1
```

---

### Phase 3：dispatch actor 结构升级

目标：

```text
网络能表达 DER 异质性、时序价值和电气灵敏度。
```

修改：

1. Type-specific DER encoders；
2. Set Transformer 替代 mean/max DeepSet；
3. GRU/Transformer temporal encoder；
4. grid sensitivity token；
5. type-specific heads；
6. hierarchical aggregate-to-DER allocator。

验收：

```text
KL 不爆炸
actual_delta_nonzero_rate 稳定
economic_operational_surplus 改善
service_quality_penalty 不恶化
no-shield safety 不显著恶化
```

---

## 9. 建议新增测试

### 9.1 非 AC 约束不应把动作覆盖为 0

```python
def test_non_ac_penalty_does_not_zero_executed_action():
    raw_action = nonzero_dispatch_action()
    env = make_env(disable_ac_violation=True)
    result = env.step(raw_action)

    assert result["decoded_delta_p_mw"] != 0
    assert result["actual_delta_p_mw"] != 0
    assert result["comfort_penalty"] >= 0
    assert result["non_ac_projection_zeroing_flag"] == 0
```

### 9.2 AC shield 可以干预，但必须有 projection gap penalty

```python
def test_ac_projection_gap_penalizes_actor():
    raw_action = action_that_causes_voltage_violation()
    result = env.step(raw_action)

    assert result["ac_projected_delta_p_mw"] != result["pre_ac_delta_p_mw"]
    assert result["device_to_ac_gap_mw"] > 0
    assert result["dispatch_projection_penalty"] > 0
```

### 9.3 动作落地链路不允许 silent baseline override

```python
def test_no_silent_baseline_override():
    result = env.step(nonzero_action)

    if result["actual_delta_p_mw"] == 0:
        assert result["drop_reason"] != ""
        assert result["accepted_to_actual_gap_mw"] > 0
```

### 9.4 actor 输出与 settlement 分项相关

```python
def test_actual_delta_changes_settlement_components():
    result0 = env.step(zero_action)
    result1 = env.step(nonzero_action)

    assert result1["actual_delta_p_mw"] != result0["actual_delta_p_mw"]
    assert result1["economic_operational_surplus"] != result0["economic_operational_surplus"]
```

---

## 10. 下一轮实验矩阵

先跑小矩阵，不要直接 paper-long：

| 实验 | horizon | episodes | 目的 |
|---|---:|---:|---|
| `landing_smoke` | 8 | 1 | 检查动作链路是否落地 |
| `settlement_smoke` | 96 | 1 | 检查 residual 是否接近 0 |
| `dispatch_profit_only` | 96 | 3 | 验证 VPP 经济闭环 |
| `no_non_ac_projection` | 96 | 3 | 验证非 AC 约束改 penalty 后是否改善 action landing |
| `ac_shield_only_projection` | 96 | 3 | 验证仅 AC shield hard intervention |
| `type_head_actor` | 96 | 3 | 验证类型专用动作头 |
| `temporal_actor` | 288 | 3 | 验证短时序 encoder |
| `sensitivity_actor` | 288 | 3 | 验证灵敏度 token |

---

## 11. 最终建议

### 11.1 关于投影与干预

最终策略应为：

```text
设备硬物理约束：用 action decoder 参数化保证，不做隐藏 projection；
AC 安全约束：允许 hard shield，但必须记录 raw/proj gap 并惩罚；
非 AC 经济/舒适/合同/偏好约束：放入 reward penalty 或 constraint penalty，不要把动作覆盖为 0。
```

这能最大限度避免：

```text
actor 输出动作，但进入环境后 actual_delta_p_mw = 0
```

的问题。

---

### 11.2 关于 dispatch agent 网络

不建议立即把当前 DeepSet actor 替换成非常复杂的 GNN/Transformer。推荐路线是：

1. **先修 action decoder 和 action landing；**
2. **再加 type-specific heads；**
3. **再把 DeepSet pooling 换成 Set Transformer / attention pooling；**
4. **再加 GRU/Transformer temporal encoder；**
5. **最后加 topology/sensitivity-aware features。**

因为诊断报告已经指出，旧实验不能证明网络本身把 VPP 调坏；实际动作没有落地时，网络再复杂也无法学到有效因果信号。fileciteturn17file4

---

## 12. Paper-long 启动前硬性条件

```text
1. actual_delta_nonzero_rate > 10%
2. action_landing_ratio_mean > 0.3
3. private_profit_vs_visible_energy_residual_ratio < 5%
4. settlement_audit_complete = 1
5. settlement_power_balance_ok = 1
6. non_ac_projection_zeroing_rate < 5%
7. ac_projection_gap 有独立 penalty
8. no-shield evaluation 可运行
9. role_approx_kl_dispatch <= target_kl_dispatch
10. reward 可由 settlement 分项解释
```

如果这些条件不满足，不建议启动 paper-long。
