# Reward V3 反馈问题审查报告：SOC、储能潜在价值、安全成本、灵活性合同与真实数据

日期：2026-06-09
适用范围：`pandapower-vpp-dso-sim` 当前 reward-v2 代码、Reward V3 计划文档、后续 paper-long 实验前的机制审查
核心结论：你提出的问题大多成立。当前应把 Reward V3 的目标从“加入更多奖励项”进一步收紧为“只保留有物理/市场依据、可审计、不会制造错误激励的奖励项”。特别是 SOC target、可用性奖励、服务费、合同偏差、未履约成本，在没有明确外部合同或市场清算机制前，不能作为论文主实验里的真实经济奖励来讲。

---

## 0. 当前代码与计划状态先澄清

这份报告必须先区分三层东西，否则很容易混淆。

### 0.1 已在当前代码中存在的机制

当前代码中已经存在：

1. DSO 安全约束检查和安全惩罚。
   - 文件：`src/vpp_dso_sim/network/constraints.py`
   - 电压越限、线路过载、变压器过载、潮流失败都有明确计算。
2. DSO reward-v1/v2 路径。
   - 文件：`src/vpp_dso_sim/entities/dso.py`
   - 当前 DSO reward 仍包含 v2 的 `dso_curtailment_cost`、`dso_safe_capacity_utilization_reward` 等字段。
3. VPP dispatch reward 中的 service payment、availability payment、contract shortfall 等 proxy。
   - 文件：`src/vpp_dso_sim/envs/reward_design.py`
   - 这些目前更像“灵活性服务代理奖励”，不是完整市场清算。
4. 储能 SOC 物理状态更新。
   - 文件：`src/vpp_dso_sim/der/storage.py`
   - 当前 storage 有 `soc_min/soc_max`，但没有真实 terminal SOC contract。
5. EVCS 是合成会话。
   - 文件：`src/vpp_dso_sim/der/evcs.py`
   - 当前 EVCS 在 `__post_init__` 中生成固定模板 EV，不是 ACN-Data 真实接入。

### 0.2 目前仍主要是计划的机制

以下内容在 Reward V3 计划中已经被设计，但当前代码尚未完整实现：

1. `v3_market_safety` reward version。
2. VPP true settlement。
3. EVCS 真实用户充电收入。
4. per-DER settlement audit。
5. welfare calibration。
6. storage potential shaping / terminal value。
7. no-shield evaluation。
8. projected unsafe safety gate。
9. 完整的 HAPPO/HATRPO reward-v3 稳定性配置。

因此，后续文档和论文不能说这些已经在当前实验代码里完全实现。更准确的表述是：

> Reward V3 是下一阶段计划实现的可审计 reward 机制；当前代码已经有 reward-v2 的若干 proxy 和安全惩罚，但还没有完整的真实市场结算与真实 EVCS/电价数据接入。

---

## 1. 问题一：没有设置 SOC target，是否还需要设计 `soc_target`？

### 1.1 你的质疑是成立的

如果项目没有外部合同、调度规则、运营约束或实验边界要求储能在某个时刻达到某个 SOC，那么把 `SOC target` 作为 reward 项是不严谨的。

储能的 SOC target 只有在下面几种情况下才有明确含义：

1. 合同约束：
   - 例如 VPP 承诺次日早上 8:00 储能必须保留 50% 容量用于备用。
2. 循环边界条件：
   - 例如每个 episode 表示一天，要求日末 SOC 回到日初 SOC，防止算法把电池在最后一步放空来刷收益。
3. 备用/韧性要求：
   - 例如配电网要求 VPP 保持最低备用能量用于故障支撑。
4. 电池健康或运营策略：
   - 例如长期运营中不希望 SOC 长期贴近 0 或 1。

如果这些都没有定义，那么 `SOC target` 不应该作为默认 reward 目标。

### 1.2 当前项目更适合怎么处理 SOC？

当前更合理的处理方式是三层：

第一层：硬物理边界。

```math
SOC_{min} \le SOC_t \le SOC_{max}
```

当前 `StorageModel.get_bounds()` 已经根据 SOC 限制充放电：

```text
SOC >= soc_max 时，不能继续充电；
SOC <= soc_min 时，不能继续放电。
```

这属于设备物理约束，必须保留。

第二层：越界惩罚或不可行动作惩罚。

如果动作导致 SOC 越界，应该通过 action projection / infeasible action cost 处理，而不是通过一个人为 target 拉回。

第三层：可选的 terminal/cyclic condition。

如果 paper-long 实验希望防止 episode 末端策略“把电池榨干”，可以加一个明确标记的实验边界：

```math
SOC_T \approx SOC_0
```

这不是合同 target，而是 cyclic episode boundary condition，应该命名为：

```text
storage_cyclic_terminal_soc_weight
```

而不是泛泛叫 `storage_terminal_soc_target_weight`。

### 1.3 对 Reward V3 计划的修正建议

不建议默认启用：

```yaml
storage_terminal_soc_target: 0.5
storage_terminal_soc_target_weight: 0.01
```

建议改为：

```yaml
storage_terminal_soc_reference_mode: disabled
storage_terminal_soc_reference: null
storage_terminal_soc_reference_weight: 0.0
```

如果要做日循环实验，再显式切换：

```yaml
storage_terminal_soc_reference_mode: initial_soc
storage_terminal_soc_reference_weight: 0.01
```

如果要做备用容量实验，再显式切换：

```yaml
storage_terminal_soc_reference_mode: reserve_requirement
storage_terminal_soc_reference: 0.5
storage_terminal_soc_reference_weight: 0.01
```

### 1.4 结论

当前没有合同或运营规则时，SOC target 不应作为主 reward。它最多是：

1. 诊断字段；
2. 消融实验项；
3. cyclic episode 的边界条件；
4. 明确备用服务实验中的约束。

Reward V3 的默认实现应关闭 SOC target reward。

---

## 2. 问题二：储能潜在价值如何计算，如何防止每一步为了潜在奖励不断充电？

### 2.1 储能为什么需要跨时价值？

储能和普通负荷、PV 不一样。充电时，当前时刻通常是负现金流：

```math
C^{charge}_t = p^{buy}_t P^{charge}_t \Delta t
```

如果 reward 只看即时利润，充电永远像亏钱。可是真实储能的价值来自未来：

```text
低价充电 -> 高价放电
低压/高压问题发生前充电或放电 -> 未来提供灵活性服务
保留 SOC -> 未来响应 DSO 调度需求
```

所以储能需要某种未来价值估计。

### 2.2 最朴素的储能潜在价值

设：

- `SOC_t`：当前 SOC；
- `E_cap`：电池容量，MWh；
- `E_t = SOC_t * E_cap`：当前储存能量；
- `p_buy_t`：当前充电购电价格；
- `p_future_sell_t`：预测未来可卖电价或服务价值；
- `eta_ch`：充电效率；
- `eta_dis`：放电效率；
- `c_deg`：电池退化成本，按 MWh 计。

储能中每 1 MWh 当前储存能量的未来可兑现价值可以近似写成：

```math
\kappa_t =
\max
\left(
0,\,
\eta_{dis} \cdot \mathbb{E}_{\tau > t}[p^{sell/service}_\tau]
- c^{deg}
\right)
```

那么储能状态势函数可以写成：

```math
\Phi(s_t) = \kappa_t \cdot E_t
```

即：

```math
\Phi(s_t) =
\kappa_t \cdot SOC_t \cdot E_{cap}
```

直观解释：

> 电池里有多少电，以及这些电未来大概值多少钱。

### 2.3 Potential-based shaping 的形式

经典 potential-based shaping 是：

```math
F_t = \gamma \Phi(s_{t+1}) - \Phi(s_t)
```

把它加到 reward：

```math
r'_t = r_t + F_t
```

如果 `Phi` 设计合理，这种 shaping 的作用是帮助 credit assignment，而不是直接改变最优策略。

### 2.4 为什么每一步可能诱导不断充电？

如果简单设：

```math
\Phi(s_t) = \kappa \cdot SOC_t \cdot E_{cap}
```

且 `kappa` 总是正数，那么只要充电导致：

```math
SOC_{t+1} > SOC_t
```

就可能得到：

```math
F_t > 0
```

这会诱导智能体：

> 只要能充电就充电，因为每一步 SOC 增加都带来 shaping reward。

但真实世界中，充电并不总是好事。充电需要付当前购电成本，还会产生效率损耗和电池退化。如果未来价格不高，充电应该是负价值。

### 2.5 防止“为了潜在奖励不断充电”的方法

#### 方法 A：潜在价值必须使用净边际价值

不要让 `kappa` 总是正数。应使用未来收益减去当前充电成本后的净价值。

对一次充电动作，边际价值应该近似为：

```math
\Delta V^{charge}_t =
\eta_{ch}\eta_{dis}
\mathbb{E}_{\tau > t}[p^{sell/service}_\tau]
- p^{buy}_t
- c^{deg}
```

如果：

```math
\Delta V^{charge}_t < 0
```

充电应该得到负的或至少非正的 shaping。

所以更稳的定义是：

```math
\kappa^{net}_t =
\eta_{dis}
\mathbb{E}_{\tau > t}[p^{sell/service}_\tau]
- c^{deg}
```

但在奖励中同时保留当前购电成本：

```math
- p^{buy}_t P^{charge}_t \Delta t
```

这样，只有未来价值超过当前成本时，充电才有正总收益。

#### 方法 B：不把 `terminal_value_proxy` 每步加入 reward

这点非常关键。

可以每步记录：

```text
storage_terminal_value_proxy
```

但不要每步加到 reward。

否则 reward 会变成：

```math
r_t \leftarrow r_t + \kappa E_t
```

这会让智能体每一步都因为高 SOC 受奖励，容易学成“囤电”。

正确做法：

```math
F_t = \gamma \Phi(s_{t+1}) - \Phi(s_t)
```

每步可以加入 `F_t`。

如果确实需要 terminal value，则只在 episode 结束时：

```math
R_T^{storage}
=
\mathbf{1}_{done}
\cdot
V_T(SOC_T)
```

并且必须做消融，检查它是否和 potential shaping 双重计算。

#### 方法 C：关闭默认 SOC target

如果没有 SOC target 合同，就不应该加：

```math
- \lambda_{socT}(SOC_T - SOC^{target})^2
```

否则算法会被人为拉向某个 SOC，而不是根据价格、服务价值和安全需求自主优化。

#### 方法 D：加入反囤电测试

必须有测试：

```text
未来价格低于当前买电价
SOC 已经偏高
继续充电不应该得到正奖励
```

数学上：

```math
p^{future,sell} \eta_{ch}\eta_{dis}
<
p^{buy} + c^{deg}
```

则：

```math
Reward(charge) < Reward(no\ charge)
```

或者至少：

```math
storage_potential_shaping_reward < 0
```

#### 方法 E：潜在价值必须被 action cost 抵消

储能 reward 不能只写：

```math
+ storage_potential_shaping
```

还必须有：

```math
- energy_buy_cost
- battery_degradation_cost
- efficiency_loss_cost
```

完整一点：

```math
r^{ESS}_t =
R^{discharge}_t
- C^{charge}_t
- C^{deg}_t
+ \lambda_F F_t
+ \mathbf{1}_{done}R^{terminal}_T
```

### 2.6 推荐的 Reward V3 储能默认设置

建议默认：

```yaml
storage_potential_shaping_weight: 0.02
storage_terminal_value_weight: 0.0
storage_terminal_soc_reference_mode: disabled
storage_terminal_soc_reference_weight: 0.0
storage_future_value_mode: price_forecast_window
storage_anti_hoarding_test_required: true
```

只有在专门做 cyclic horizon 或备用容量实验时，才启用 terminal SOC reference。

### 2.7 结论

储能潜在价值可以用，但必须满足：

1. 它是未来净价值，不是单纯奖励高 SOC；
2. 每步只能加 potential difference，不能每步加 terminal value；
3. 当前充电成本、效率损失、退化成本必须进入 reward；
4. 默认不使用 SOC target；
5. 必须有 anti-hoarding 测试；
6. 必须有 `v3_no_storage_temporal_value` 消融。

---

## 3. 问题三：电路安全成本，包括电压越限、变压器、线路过载，是怎么计算的？

### 3.1 当前代码中的约束检查

当前安全约束在：

```text
src/vpp_dso_sim/network/constraints.py
```

核心函数：

```python
check_network_constraints(...)
violation_penalties(...)
```

默认约束：

```text
电压范围：0.95 p.u. 到 1.05 p.u.
线路 loading 上限：100%
变压器 loading 上限：100%
```

DSO 类中也有同样默认：

```python
voltage_limits = (0.95, 1.05)
line_loading_limit_percent = 100.0
trafo_loading_limit_percent = 100.0
```

### 3.2 电压越限如何计算？

对每个 bus：

如果：

```math
V_b < V_{min}
```

越限幅度：

```math
m_b = V_{min} - V_b
```

如果：

```math
V_b > V_{max}
```

越限幅度：

```math
m_b = V_b - V_{max}
```

当前代码中的原始电压惩罚：

```math
C^V =
10000
\sum_b m_b^2
```

例子：

1. 电压 1.055 p.u.，上限 1.05。

```math
m = 1.055 - 1.05 = 0.005
```

```math
C^V = 10000 \times 0.005^2 = 0.25
```

2. 电压 1.06 p.u.，上限 1.05。

```math
m = 0.01
```

```math
C^V = 10000 \times 0.01^2 = 1.0
```

3. 电压 1.10 p.u.，上限 1.05。

```math
m = 0.05
```

```math
C^V = 10000 \times 0.05^2 = 25.0
```

这种设计含义是：

> 小越限有惩罚，大越限的惩罚按平方快速增长。

### 3.3 线路过载如何计算？

对每条 line：

如果：

```math
Loading_l > Loading^{max}
```

越限幅度：

```math
m_l = Loading_l - Loading^{max}
```

单位是百分点。

当前代码中的线路过载惩罚：

```math
C^L =
5
\sum_l m_l^2
```

例子：

1. 线路 loading = 105%，上限 100%。

```math
m = 5
```

```math
C^L = 5 \times 5^2 = 125
```

2. 线路 loading = 115%，上限 100%。

```math
m = 15
```

```math
C^L = 5 \times 15^2 = 1125
```

这比 1% 电压越限大很多，因为单位不同：电压越限是 p.u.，线路过载是百分点。

### 3.4 变压器过载如何计算？

变压器和线路类似：

```math
m_{tr} = Loading_{tr} - Loading^{max}_{tr}
```

当前代码：

```math
C^T =
5
\sum_{tr} m_{tr}^2
```

例子：

变压器 loading = 110%，上限 100%：

```math
C^T = 5 \times 10^2 = 500
```

### 3.5 潮流失败如何计算？

如果 pandapower 潮流不收敛：

```math
C^{PF} = 1000
```

当前代码直接设置：

```python
powerflow_penalty = 1000.0
```

### 3.6 DSO 中还有一层缩放

在 `DSO.calculate_reward_or_cost()` 里，原始 penalty 会再经过 scaling：

默认 scale：

```text
voltage_violation_penalty scale = 1.0
line_overload_penalty scale = 100.0
transformer_overload_penalty scale = 100.0
powerflow_penalty scale = 1000.0
```

缩放公式近似：

```math
\widetilde{C}_k =
\min
\left(
clip,\,
\frac{C_k}{scale_k}
\right)
```

默认 clip：

```text
reward_component_clip = 10
```

所以：

1. 电压 penalty 25，scale 1。

```math
\widetilde{C} = \min(10,25/1)=10
```

2. 线路 penalty 1125，scale 100。

```math
\widetilde{C} = \min(10,11.25)=10
```

3. 潮流失败 penalty 1000，scale 1000。

```math
\widetilde{C} = 1
```

这说明当前 reward 的安全尺度并不直观。原始 penalty 和 scaled penalty 的相对大小不同，必须在报告里同时输出。

### 3.7 Guard band 是什么？

DSO 还有一类 “guard band penalty”：

```text
dso_voltage_guard_penalty
dso_line_guard_penalty
dso_trafo_guard_penalty
```

它不是已经越限，而是接近边界时的提前惩罚。

默认：

```text
voltage_guard_band_pu = 0.02
line_guard_band_percent = 5.0
trafo_guard_band_percent = 5.0
```

例如电压上限 1.05，当前电压 1.04，还没越限，但距离上限只有 0.01 p.u.，小于 guard band 0.02，于是会产生安全裕度惩罚。

数学上类似：

```math
C^{guard,V}
=
\sum_b
\max(0, guard_V - margin_b)^2
```

其中：

```math
margin_b =
\min(V_b - V_{min}, V_{max} - V_b)
```

### 3.8 对 Reward V3 的建议

安全成本应该拆成四个层次：

1. 原始 post-AC violation：

```text
post_ac_voltage_violation_cost
post_ac_line_overload_cost
post_ac_trafo_overload_cost
post_ac_powerflow_failed
```

2. raw action safety：

```text
raw_action_voltage_violation_cost
raw_action_line_overload_cost
raw_action_trafo_overload_cost
raw_action_powerflow_failed
```

3. projected action safety：

```text
projected_action_voltage_violation_cost
projected_action_line_overload_cost
projected_action_trafo_overload_cost
projected_action_powerflow_failed
```

4. normalized safety gate input：

```math
C^{gate}
=
\max(C^{raw,norm}, C^{projected,norm})
```

这样才能回答：

> 是智能体自己学会安全，还是 safety shield 替它修好了动作？

---

## 4. 问题四：可用性奖励的存在意义是什么？

### 4.1 真实市场中的 availability payment

在真实灵活性服务或备用服务中，可用性奖励一般表示：

> VPP 即使没有被实际调用，也因为承诺在某段时间保持可调能力而获得容量/可用性补偿。

例如：

```text
VPP 承诺 14:00-16:00 可提供 0.5 MW 上调能力。
DSO 不一定每分钟调用它。
但 VPP 为了保持这 0.5 MW 能力，可能牺牲其他市场机会。
所以 DSO 支付 availability payment。
```

### 4.2 availability payment 在本项目里的潜在正面作用

它可以表达：

1. VPP 不是只在被调用时赚钱；
2. 备用容量本身有价值；
3. VPP 需要保留储能 SOC、柔性负荷空间或可调机组余量；
4. DSO 可以用价格引导 VPP 提前准备可调能力。

### 4.3 当前代码中的 availability payment 更像 proxy

当前代码中：

```python
availability_payment = AVAILABILITY_PAYMENT_RATE * price * flex_span_mw * dt_hours
```

含义是：

```math
R^{avail}_t
=
\alpha_{avail}
p_t
(P^{max}_t - P^{min}_t)
\Delta t
```

这不是完整市场合同，而是用可行域宽度 `flex_span_mw` 近似“可用能力”。

问题是：

> 如果没有 DSO award、没有容量合同、没有被占用的机会成本，availability payment 可能诱导 VPP 单纯扩大可行域宽度，而不是提供真实有价值的服务。

### 4.4 是否应该保留？

批判性结论：

1. 作为真实市场奖励：当前不应默认保留。
2. 作为 proxy 实验：可以保留，但必须标记 `baseline_proxy`。
3. 作为论文主结论：不能声称这是完整可用性市场。
4. 作为消融项：应该保留一个 `v3_no_availability_payment` 或 `v3_no_service_payment` 对照。

### 4.5 推荐默认设置

Reward V3 主实验建议：

```yaml
availability_payment_weight: 0.0
availability_payment_source: disabled
```

如果想研究 availability proxy：

```yaml
availability_payment_weight: 1.0
availability_payment_source: baseline_proxy
```

如果以后实现真实容量合同：

```yaml
availability_payment_source: capacity_contract
```

### 4.6 结论

availability reward 有市场含义，但当前项目尚未完整实现容量合同与机会成本。因此不应默认作为 Reward V3 主训练奖励。它应该是：

1. 诊断项；
2. proxy ablation；
3. 后续市场机制增强项。

---

## 5. 问题五：服务费收入、合同偏差、未履约成本是否应纳入奖励？

你这里的判断需要分开看。

### 5.1 真实灵活性服务合同中，它们应该存在

如果系统有完整合同，则这些项合理：

1. 服务费收入：

```math
R^{service}_{i,t}
=
\pi^{service}_{i,t}
Q^{delivered}_{i,t}
\Delta t
```

2. 合同偏差：

```math
e^{delivery}_{i,t}
=
Q^{awarded}_{i,t}
-
Q^{delivered}_{i,t}
```

3. 未履约成本：

```math
C^{shortfall}_{i,t}
=
\lambda_{shortfall}
\left[e^{delivery}_{i,t}\right]_+^2
```

这些在真实 DSO-VPP 灵活性市场中是合理的。

### 5.2 当前项目的问题是：完整合同/出清/履约链条还没实现

要真正纳入这些 reward，至少需要：

1. VPP bid：

```text
VPP 报价：愿意提供多少 MW、什么方向、什么价格、在哪个节点/区域。
```

2. DSO award：

```text
DSO 出清：接受多少 MW，服务类型是什么，价格是多少。
```

3. delivery measurement：

```text
实际交付量如何从 baseline 中计算。
```

4. baseline rule：

```text
没有服务时 VPP 应该是多少功率。
```

5. non-delivery penalty：

```text
未履约如何惩罚，是否对称，是否有容差。
```

6. settlement record：

```text
谁付钱给谁，转移支付是否进入 DSO welfare。
```

当前代码中有 baseline/service/contract 的 proxy，但还不能说是完整灵活性服务合同。

### 5.3 当前代码中这些项是什么状态？

当前 `reward_design.py` 中存在：

```text
flexibility_service_payment
service_payment
availability_payment
verified_delivery_mw
contract_shortfall_mw
contract_delivery_penalty
```

但它们主要来自：

```text
baseline_p_mw
target_p_mw
delivered_p_mw
price
flex_span_mw
```

这更像 reward proxy，不是完整 market clearing。

### 5.4 Reward V3 主实验应该怎么做？

建议主实验默认禁用：

```yaml
service_payment_weight: 0.0
availability_payment_weight: 0.0
contract_delivery_weight: 0.0
```

或者仅保留非常明确的投影/跟踪物理惩罚：

```yaml
dispatch_projection_penalty
device_constraint_penalty
unserved_ev_energy_penalty
```

如果要保留 service/contract 项，必须标记为：

```text
settlement_proxy
baseline_proxy
not_full_market_clearing
```

论文表述应改成：

> 本阶段使用 settlement-aware proxy reward，而不是完整 DSO-VPP 灵活性市场清算。

### 5.5 对 DSO reward 的额外提醒

如果 VPP 的 private profit 包含 service payment，而 DSO reward 又最大化 VPP private profit，就会出现转移支付错误：

```text
DSO 多付钱 -> VPP 利润增加 -> DSO reward 增加
```

这在社会福利上是不对的。

DSO welfare 应使用：

```math
\Pi^{op}_{i,t}
=
\text{VPP operational surplus excluding transfer payments}
```

而不是完整 private profit。

### 5.6 结论

服务费、合同偏差、未履约成本不是永远不能用，而是当前不能作为主实验真实市场 reward。当前应：

1. 主实验禁用或降权；
2. 如启用，明确标记为 proxy；
3. 单独做 `v3_no_service_payment`、`v3_proxy_service_payment` 消融；
4. 未来实现 bid-award-delivery-settlement 后，再升级为真实合同 reward。

---

## 6. 问题六：`min_raw_unsafe_penalty` 是否还是偏大？

### 6.1 是的，`1.0` 作为默认 floor 可能偏大

计划中曾经设计：

```python
min_raw_unsafe_penalty = 1.0
raw_action_safety_weight = 20.0
```

如果 raw safety 只要大于 0 就触发：

```math
C^{raw,penalty}
\ge
1.0
```

则安全惩罚至少：

```math
20 \times 1.0 = 20
```

而 welfare 通常被 clip：

```math
\widetilde{W} \in [-5, 5]
```

这意味着任何微小 raw violation 都可能压倒全部经济信号。

### 6.2 这有什么优点？

优点：

1. 强安全优先；
2. 可以防止智能体依赖 safety shield；
3. safe-first proof 容易成立。

### 6.3 这有什么问题？

问题：

1. 不连续：
   - raw violation 从 0 到 0.000001，reward 突然大幅下降。
2. 不区分轻微和严重：
   - 轻微越限和严重越限都先吃一个大 floor。
3. 训练不平滑：
   - actor 很难学习“离安全边界还有多远”。
4. 可能阻断经济学习：
   - 早期策略经常微小不安全时，welfare 信号几乎消失。

### 6.4 更合理的设计

建议用：

```yaml
raw_safety_epsilon: 1.0e-5
min_raw_unsafe_penalty: 0.05
raw_action_safety_weight: 10.0
```

或者平滑函数：

```math
C^{raw,floor}
=
C^{raw,norm}
+
\alpha
\cdot
\sigma(k(C^{raw,norm} - \epsilon))
```

更简单的工程版本：

```python
if raw_norm > raw_safety_epsilon:
    raw_penalty_input = raw_norm + min_raw_unsafe_penalty
else:
    raw_penalty_input = 0.0
```

但 `min_raw_unsafe_penalty` 建议从 `0.05` 或 `0.1` 开始，而不是 `1.0`。

### 6.5 推荐实验矩阵

必须做消融：

| 实验 | min_raw_unsafe_penalty | raw_action_safety_weight | 目的 |
|---|---:|---:|---|
| A | 0.0 | 10 | 看没有 floor 时是否依赖 shield |
| B | 0.05 | 10 | 温和安全优先 |
| C | 0.1 | 10 | 推荐初始值 |
| D | 0.5 | 10 | 强安全优先 |
| E | 1.0 | 20 | 极强安全优先，只作上界压力测试 |

指标：

```text
raw_action_violation_rate
projected_action_violation_rate
no_shield_eval_violation_rate
dso_safety_gate_mean
dso_safety_gate_p10
welfare_clip_saturation_rate
policy_kl
reward_improvement
```

### 6.6 结论

`min_raw_unsafe_penalty=1.0` 不应作为默认 paper-long 设置。它适合做强安全 ablation，但主实验建议从：

```yaml
min_raw_unsafe_penalty: 0.1
raw_action_safety_weight: 10.0
raw_safety_epsilon: 1.0e-5
```

开始。

---

## 7. 问题七：EVCS 和电价真实数据集是否已经调查完毕？

### 7.1 简短结论

没有完全完成。

当前状态应该分成四级：

| 层级 | 状态 | 说明 |
|---|---|---|
| 数据源调查 | 部分完成 | 已登记 ACN-Data、OpenEI URDB、SMART-DS、NREL EULP 等候选源。 |
| 数据下载 | 部分完成 | SMART-DS 子集已下载；ACN-Data 和 OpenEI/URDB 未确认已下载到本项目。 |
| 适配器实现 | 未完成或仅计划 | Reward V3 计划中提出 `ACNDataEVCSAdapter`、`TariffAdapter`，但当前代码主要只有 registry。 |
| reward 接入 | 未完成 | 当前 EVCS reward 还没有真实用户充电收入；电价仍主要是 price profile/proxy。 |

### 7.2 ACN-Data 是否适合 EVCS？

适合，但要注意边界。

ACN-Data 官方页面说明它是为研究者提供真实 EV 充电数据的数据集；每条记录对应一次充电 session。它提供 Web Interface、REST API 和 Python API Client 三种访问方式。官方字段包括：

```text
connectionTime
disconnectTime
doneChargingTime
kWhDelivered
chargingCurrent time series
pilotSignal time series
siteID
stationID
spaceID
userInputs
kWhRequested
minutesAvailable
paymentRequired
requestedDeparture
```

这些字段很适合构造 EVCS：

```text
arrival_step       <- connectionTime
departure_step     <- disconnectTime 或 requestedDeparture
requested_energy   <- userInputs.kWhRequested
delivered_energy   <- kWhDelivered
max_charge_power   <- pilotSignal / station limit
payment_required   <- userInputs.paymentRequired
```

但 ACN-Data 有几个限制：

1. 需要注册 token 才能用 API。
2. 站点类型主要是 Caltech、JPL、Office 等工作场景，不代表所有居民区夜间充电。
3. 官方提示 Web Interface 在 2019-10-10 之前曾有 timezone bug；应优先用 API 或当前 Python client。
4. 数据接入后必须保留 site type，否则论文中不能泛化为所有 EVCS。

来源：ACN-Data 官方数据页，包含 API endpoint、字段和站点说明：
https://ev.caltech.edu/dataset.html

### 7.3 OpenEI / URDB 是否适合真实电价？

适合零售 tariff，但不等于简单一列电价。

OpenEI Utility Rate API 官方文档说明它提供美国 Utility Rate Database 中复杂 utility rate structure 信息，并且 API 需要 `api_key`。它支持 sector、country、address、lat/lon、utility 等查询。返回字段包含：

```text
utility
rate name
sector
service type
demand charge structure
energy charge structure
season/month period
TOU period
voltage category
approved/default status
```

这说明 URDB 可以用于：

1. EVCS 零售电价；
2. TOU 分时电价；
3. demand charge；
4. commercial/industrial tariff。

但它不是直接给你一个干净的 `price[t]` 数组。必须写 TariffAdapter，把复杂 tariff 结构转换成：

```text
retail_buy_price_per_mwh[t]
wholesale_buy_price_per_mwh[t]
export_sell_price_per_mwh[t]
evcs_retail_price_per_mwh[t]
demand_charge_per_mw[t]
```

来源：OpenEI Utility Rates API 文档：
https://apps.openei.org/services/doc/rest/util_rates/

EIA FAQ 也说明 DOE/OpenEI 是美国 utility rate structure 的免费来源：
https://www.eia.gov/tools/faqs/faq.php?id=20&t=6

### 7.4 SMART-DS 当前是什么状态？

本项目 `docs/dataset_landscape.md` 记录：

1. NREL SMART-DS Austin 子集已下载到：

```text
data/external/raw/smart_ds/v1.0/2018/AUS/P1U/base_timeseries/opendss
data/external/raw/smart_ds/v1.0/2018/AUS/P1U/profiles
```

2. 该子集包含：

```text
2369 个 opendss 文件
3021 个 profile 文件
约 1.9 GiB profiles
25 个 primary feeder directories
93 个 LV portfolio directories
```

SMART-DS 适合做真实/合成配电网拓扑与负荷曲线，不是 EVCS session 数据，也不是 retail tariff 数据。

来源：NREL SMART-DS 官方页面：
https://www.nrel.gov/grid/smart-ds

### 7.5 当前是否可以说“真实 EVCS 和电价数据已经接入 reward”？

不能。

当前只能说：

> 项目已经完成真实数据源 landscape 和 registry 的初步整理，并下载了 SMART-DS 子集；ACN-Data 和 OpenEI/URDB 是明确的下一步 P0/P1 数据源，但尚未完成 adapter、清洗、缓存、场景生成和 reward 接入。

### 7.6 下一步必须做什么？

#### ACN-Data 接入任务

1. 获取 API token。
2. 下载指定 site 和时间范围 session。
3. 标准化字段：

```text
session_id
site_id
station_id
arrival_time
departure_time
requested_departure
kwh_requested
kwh_delivered
payment_required
pilot_signal
charging_current
timezone
```

4. 转换为 15-min step：

```text
arrival_step
departure_step
deadline_step
requested_energy_mwh
max_charge_mw
```

5. 生成 EVCS audit：

```text
evcs_grid_p_mw
evcs_user_energy_mwh
evcs_unserved_energy_mwh
evcs_deadline_pressure
```

#### OpenEI/URDB 接入任务

1. 获取 API key。
2. 选择 utility、sector、rate。
3. 解析 TOU energy charges。
4. 解析 demand charges。
5. 转成统一 price streams：

```text
retail_buy_price_per_mwh
wholesale_buy_price_per_mwh
export_sell_price_per_mwh
evcs_retail_price_per_mwh
demand_charge_per_mw
```

6. 明确地区和电价假设。

---

## 8. 对 Reward V3 的修订建议汇总

### 8.1 应该删除或默认关闭的项

默认关闭：

```yaml
storage_terminal_soc_reference_weight: 0.0
service_payment_weight: 0.0
availability_payment_weight: 0.0
contract_delivery_weight: 0.0
```

除非有明确合同或实验设定。

### 8.2 应该保留的项

保留：

```yaml
energy_buy_cost
energy_sell_revenue
evcs_user_charging_revenue
evcs_wholesale_energy_cost
battery_degradation_cost
storage_potential_shaping_reward
dispatch_projection_penalty
device_constraint_penalty
raw_action_safety_cost
projected_action_safety_cost
```

### 8.3 应该改名的项

把：

```text
storage_terminal_soc_target
```

改成：

```text
storage_terminal_soc_reference
```

并加：

```text
storage_terminal_soc_reference_mode:
  disabled | initial_soc | reserve_requirement | explicit_contract
```

把：

```text
service_payment
availability_payment
```

改成带 source：

```text
service_payment_source:
  disabled | baseline_proxy | cleared_award

availability_payment_source:
  disabled | baseline_proxy | capacity_contract
```

### 8.4 推荐默认 Reward V3 配置

```yaml
reward:
  version: v3_market_safety
  dso:
    raw_action_safety_weight: 10.0
    projected_action_safety_weight: 5.0
    min_raw_unsafe_penalty: 0.1
    raw_safety_epsilon: 1.0e-5
    safety_gate_input_mode: max_raw_projected
    welfare_normalization_mode: per_mwh_baseline_zscore
    welfare_clip: 5.0
  vpp:
    dispatch:
      private_profit_weight: 1.0
      service_payment_weight: 0.0
      availability_payment_weight: 0.0
      contract_delivery_weight: 0.0
      storage_potential_shaping_weight: 0.02
      storage_terminal_value_weight: 0.0
      storage_terminal_soc_reference_mode: disabled
      storage_terminal_soc_reference_weight: 0.0
      battery_degradation_weight: 0.01
```

---

## 9. 最小验证实验建议

### 9.1 储能反囤电实验

目标：证明储能不会为了 shaping reward 无脑充电。

设置：

```text
当前买电价高
未来卖电价低
SOC 已经高于 0.7
动作 A：继续充电
动作 B：不动作
```

期望：

```math
r(B) > r(A)
```

至少：

```math
storage_potential_shaping_reward(A) < 0
```

### 9.2 无 SOC target 实验

目标：证明关闭 SOC target 后，储能仍可根据价格和安全需求学习。

配置：

```yaml
storage_terminal_soc_reference_mode: disabled
storage_terminal_soc_reference_weight: 0.0
```

观察：

```text
SOC 是否撞边界
是否仍有合理充放电
是否出现末端放空
```

如果末端放空，再启用 cyclic boundary，而不是随便设 0.5 target。

### 9.3 服务费关闭实验

配置：

```yaml
service_payment_weight: 0.0
availability_payment_weight: 0.0
contract_delivery_weight: 0.0
```

观察：

```text
VPP 是否仍能通过 energy/EVCS/storage 获得合理利润
DSO 是否仍能通过 safety reward 学到安全包络
```

### 9.4 raw unsafe penalty 扫描

配置矩阵：

```text
min_raw_unsafe_penalty = 0, 0.05, 0.1, 0.5, 1.0
```

观察：

```text
raw_action_violation_rate
no_shield_eval_violation_rate
dso_safety_gate_mean
policy KL
reward improvement
```

### 9.5 EVCS 真实数据接入 smoke

目标：证明 ACN session 能转成 EVCS episode。

输出：

```text
arrival_step
departure_step
requested_energy_mwh
delivered_energy_mwh
evcs_grid_p_mw
evcs_user_energy_mwh
unserved_energy_mwh
payment_required
```

### 9.6 电价接入 smoke

目标：证明 OpenEI/URDB tariff 能转成 15-min price stream。

输出：

```text
retail_buy_price_per_mwh[t]
evcs_retail_price_per_mwh[t]
demand_charge_per_mw[t]
export_sell_price_per_mwh[t]
```

---

## 10. 最终结论

### 10.1 你的反馈中应采纳的部分

应采纳：

1. 没有合同或规则时，不应默认设置 SOC target。
2. 储能 potential shaping 必须防止每步充电刷奖励。
3. 电路安全成本必须明确原始量、缩放量和 gate 输入。
4. availability reward 当前只能算 proxy，不应默认当真实市场收益。
5. service payment、contract deviation、non-delivery cost 在完整灵活性合同没实现前，不应作为主 reward。
6. `min_raw_unsafe_penalty=1.0` 偏大，适合作上界消融，不适合作默认。
7. EVCS 与电价真实数据源只是完成了初步调查和登记，没有完成接入。

### 10.2 当前最应该修改的 Reward V3 默认方向

Reward V3 不应该继续堆奖励项，而应该收敛为：

```text
DSO:
  安全成本 + raw/projected safety audit + calibrated welfare

VPP dispatch:
  真实能量收支 + EVCS 用户收入 + 设备成本 + 储能净跨时价值

暂不默认包含:
  SOC target
  availability payment
  service payment
  contract shortfall
  non-delivery penalty
```

### 10.3 论文中应如何表述

不能写：

> 本文实现了完整 DSO-VPP 灵活性服务市场结算。

更准确：

> 本文当前阶段实现了 safety-first、settlement-aware 的多智能体 reward 设计。VPP 侧结算显式区分能量收入、EVCS 用户收入、储能跨时价值和设备成本；灵活性服务费、可用性费和合同履约项在未实现完整 bid-award-delivery-settlement 前作为可关闭的 proxy/ablation，不作为主实验真实市场收益。

### 10.4 最优先下一步

建议按这个顺序改 Reward V3 计划和代码：

1. 把 SOC target 默认关闭，改成 optional `soc_reference_mode`。
2. 把 service/availability/contract 默认关闭或标记 proxy。
3. 把 storage potential 改成净未来价值，并加入 anti-hoarding 测试。
4. 把 `min_raw_unsafe_penalty` 默认降到 0.1，并做扫描。
5. 明确安全成本 raw/post/projected 三套日志。
6. 实现 ACN-Data adapter 和 TariffAdapter smoke test，但不要声称已经完成真实数据接入。

---

## 11. 使用到的本地证据

本地文件：

- `src/vpp_dso_sim/der/storage.py`
- `src/vpp_dso_sim/der/evcs.py`
- `src/vpp_dso_sim/network/constraints.py`
- `src/vpp_dso_sim/entities/dso.py`
- `src/vpp_dso_sim/envs/reward_design.py`
- `src/vpp_dso_sim/learning/reward_config.py`
- `src/vpp_dso_sim/data_sources/registry.py`
- `docs/dataset_landscape.md`
- `docs/superpowers/plans/2026-06-09-reward-v3-market-safety-terminal-value.md`

外部来源：

- ACN-Data 官方数据页：https://ev.caltech.edu/dataset.html
- OpenEI Utility Rates API：https://apps.openei.org/services/doc/rest/util_rates/
- EIA 关于 utility rate/tariff/demand-charge 数据的 FAQ：https://www.eia.gov/tools/faqs/faq.php?id=20&t=6
- NREL SMART-DS 官方页面：https://www.nrel.gov/grid/smart-ds
