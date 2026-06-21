#  最大风险一：DSO welfare 归一化在实现计划里不充分

设计文档明确写了：

$$
\widetilde{W}_t
===============

\operatorname{clip}
\left(
\frac{W_t-\mu_W}{\sigma_W+\epsilon},
-W_{clip},
W_{clip}
\right)
$$

也就是 VPP 总收益进入 DSO reward 前，应该先做 **标准化 + clip**。

但 implementation plan 里的 `_dso_v3_components` 是：

```python
bounded_welfare = max(-welfare_clip, min(welfare_clip, vpp_welfare_proxy))
welfare_reward = welfare_weight * safety_gate * bounded_welfare
```

这只是对原始 `vpp_welfare_proxy` 做 clip，没有做：

$$
\frac{W_t-\mu_W}{\sigma_W+\epsilon}
$$

这会有两个问题。

---

## 问题 1：如果 VPP welfare 数值很大，会被 clip 成常数

例如真实私有利润每步可能是：

$$
W_t = 200, 500, 1000
$$

而 `welfare_clip = 5.0`。

那么它们都会变成：

$$
\widetilde{W}_t = 5.0
$$

DSO 看不到 200 和 1000 的区别。

这会导致：

> DSO reward 中 VPP welfare 失去分辨率，无法学习更优经济调度。

---

## 问题 2：如果 VPP welfare 数值很小，信号又可能过弱

如果 step-level welfare 是：

$$
W_t = 0.02
$$

而 safety penalty 是：

$$
C^{safe}_t \approx 1
$$

那么 welfare 项几乎没有作用。

所以这里不能只 clip，必须引入 running normalizer：

```python
welfare_normalizer.update(vpp_welfare_proxy)
welfare_z = (vpp_welfare_proxy - mean) / (std + eps)
bounded_welfare = clamp(welfare_z, -welfare_clip, welfare_clip)
```

更稳妥的是按 **MWh、VPP 数量、时间步长** 标准化：

$$
W_{norm,t}
==========

\frac{W_t}{N_{VPP}\Delta t}
$$

然后再做 running mean/std。

**建议修改：**

新增：

```python
welfare_normalization_mode: "running_zscore" | "baseline_rule" | "per_mwh"
welfare_running_mean
welfare_running_std
welfare_clip
```

并在日志中同时记录：

```text
dso_vpp_welfare_raw
dso_vpp_welfare_per_mwh
dso_vpp_welfare_zscore
dso_vpp_welfare_bounded
```

否则你后面又会遇到 reward 分项不可解释的问题。

---

# 4. 最大风险二：安全门控可能过度饱和

当前 implementation plan 里：

```python
safety_gate = exp(-kappa * safety_cost)
```

默认：

```text
soft_safety_gate_kappa = 10.0
```

测试里甚至用了：

```text
soft_safety_gate_kappa = 25.0
```

问题是 `safety_cost` 已经包含：

```python
hard_violation_weight * bounded_training_penalty(hard_violation_raw)
powerflow_failure_weight * post_ac_powerflow_failed
raw_action_safety_weight * bounded_training_penalty(raw_action_safety_cost)
```

其中 `hard_violation_weight = 20.0`，`powerflow_failure_weight = 50.0`。

如果 `safety_cost = 2`，则：

$$
G^{safe}
========

# \exp(-10 \times 2)

e^{-20}
\approx 2.06 \times 10^{-9}
$$

如果 `kappa = 25`，那就更接近 0。

这会导致：

> 只要有一点安全问题，VPP welfare 项完全消失，DSO 只剩安全惩罚信号。

短期看这符合“安全优先”，但训练上可能出现两个问题：

1. 早期策略不安全时，经济信号完全消失，DSO 学不到安全与经济之间的边界；
2. 如果 safety shield 把 post-action 修安全，而 raw safety cost 又不准确，gate 可能长期接近 1 或长期接近 0，二者都会造成学习信号异常。

---

## 建议修改

不要用已经加权后的 `safety_cost` 直接进 gate，而是把 gate 建立在 **归一化未加权安全指标** 上：

$$
G^{safe}_t
==========

\exp
\left(
-\kappa
\cdot
C^{safe,norm}_t
\right)
$$

其中：

$$
C^{safe,norm}_t
===============

C^V_t + C^L_t + C^T_t + C^{PF}_t
$$

然后 reward 本身仍然保留：

$$
-\lambda_{safe} C^{safe}_t
$$

也就是说：

```python
safety_penalty = lambda_safe * weighted_safety_cost
safety_gate = exp(-kappa * normalized_safety_cost)
reward = -safety_penalty + safety_gate * bounded_welfare - loss - smooth
```

这样 gate 不会因为 `hard_violation_weight=20` 被放大到完全饱和。

还可以设置课程学习：

| 阶段     | kappa |
| ------ | ----: |
| early  |   1.0 |
| middle |   3.0 |
| late   |  10.0 |

这比一开始就用 10 或 25 更稳。

---

# 5. 最大风险三：实现计划里的 “safe-first proof” 不能自动成立

设计文档给出了安全优先条件：

$$
\lambda_{safe}c_{min}

>

2\beta W_{clip}
+
\lambda_{loss}\Delta P_{loss}
+
\lambda_{smooth}\Delta S
$$

然后就可以保证不安全动作不会优于安全动作。

这个证明逻辑是对的，但 implementation plan 里的代码不一定满足这个条件。

原因：

1. `c_min` 没有定义；
2. `bounded_training_penalty` 可能把大 violation 压小；
3. `loss_cost` 和 `smoothness_penalty` 的范围没有严格上界；
4. `vpp_welfare_proxy` 没有标准化，只是 clip；
5. `raw_action_safety_cost` 默认是 0，如果没有真实传入，安全优先会退化成 post-shield 安全；
6. safety shield 可能让 `post_ac_violation_count = 0`，从而让 DSO 误以为动作安全。

因此，不能只靠单元测试：

```python
unsafe["dso_reward_train"] < safe["dso_reward_train"]
```

这个测试太弱。

---

## 建议新增强测试

应该加一个参数扫描测试：

```python
@pytest.mark.parametrize("unsafe_welfare", [5, 50, 500, 5000])
@pytest.mark.parametrize("violation_magnitude", [0.01, 0.05, 0.1, 0.2])
def test_unsafe_action_never_beats_safe_action_under_worst_case_profit(...):
    ...
```

并验证：

$$
r^{DSO}*{unsafe} < r^{DSO}*{safe}
$$

还应该测试：

```text
raw unsafe + post safe
```

也就是 safety shield 修正后安全，但 raw action 不安全时，DSO reward 必须仍然下降。

否则又会回到原来的问题：

> safety shield 替 MARL 学安全。

---

# 6. 最大风险四：VPP settlement 有重复计入服务收入的问题

implementation plan 里 v3 的 private profit 写成：

```python
private_profit_proxy = (
    energy_market_revenue
    + evcs_user_charging_revenue
    + flexibility_service_payment
    + availability_payment
    - evcs_wholesale_energy_cost
    - evcs_demand_charge_cost
    - der_operation_cost
    - battery_degradation_cost
)
```

然后 reward 又写成：

```python
reward = (
    private_profit_weight * private_profit_proxy
    + service_payment_weight * flexibility_service_payment
    + availability_payment_weight * availability_payment
    + preferred_region_bonus
    - contract_delivery_penalty
    - dispatch_projection_penalty
    - comfort_soc_weight * scaled_comfort_soc_penalty
)
```

这会导致：

> `flexibility_service_payment` 和 `availability_payment` 既进入了 `private_profit_proxy`，又作为单独 reward 项进入了一次。

也就是说，服务收入被重复奖励。

尤其默认配置里：

```yaml
private_profit_weight: 0.05
service_payment_weight: 1.0
availability_payment_weight: 1.0
```

这意味着 private profit 仍然只是一个很小的项，而 service / availability 仍然可能主导 dispatch reward。

这和 reward-v3 的目标冲突。

---

## 两种正确写法，只能选一种

### 写法 A：真实利润统一入口

定义：

$$
\Pi_i
=====

R^{energy}
+
R^{EVCS}
+
R^{service}
+
R^{availability}
----------------

## C^{buy}

## C^{op}

## C^{deg}

## C^{comfort}

## C^{unserved}

C^{deviation}
$$

然后 reward 写成：

$$
r_i^{dispatch}
==============

## \lambda_{\Pi}\Pi_i

## \lambda_{proj}C^{proj}

\lambda_{constraint}C^{constraint}
$$

代码上就是：

```python
reward = (
    private_profit_weight * true_private_profit
    - dispatch_projection_penalty
    - comfort_soc_weight * scaled_comfort_soc_penalty
    - contract_delivery_penalty
)
```

这时不要再额外加：

```python
+ service_payment_weight * service_payment
+ availability_payment_weight * availability_payment
```

---

### 写法 B：利润代理不含服务收入，服务收入单独加

定义：

```python
private_profit_proxy = (
    energy_sell_revenue
    + evcs_user_charging_revenue
    - energy_buy_cost
    - evcs_wholesale_energy_cost
    - der_operation_cost
    - battery_degradation_cost
)
```

然后：

```python
reward = (
    private_profit_weight * private_profit_proxy
    + service_payment_weight * service_payment
    + availability_payment_weight * availability_payment
    - penalties
)
```

这个也可以，但需要明确：

> private_profit_proxy 不是完整利润，而是 energy/retail margin。

我更建议采用 **写法 A**，因为论文里更容易解释：

$$
\text{VPP dispatch reward 以真实结算利润为主体。}
$$

---

# 7. 最大风险五：energy_market_revenue 与 EVCS wholesale cost 可能双重扣费

当前测试里：

```python
delivered_p_mw = -0.05
price = 400
retail_evcs_tariff = 1000
evcs_charging_p_mw = 0.05
```

如果原有代码保留：

$$
energy_market_revenue
=====================

price \cdot delivered_p \cdot \Delta t
$$

那么：

$$
energy_market_revenue
=====================

# 400 \times (-0.05) \times 0.25

-5
$$

同时 EVCS settlement 又计算：

$$
evcs_wholesale_energy_cost
==========================

# 400 \times 0.05 \times 0.25

5
$$

如果 private profit 写成：

```python
energy_market_revenue
+ evcs_user_charging_revenue
- evcs_wholesale_energy_cost
```

那 EVCS 购电成本被扣了两次：

$$
-5 -5
$$

这会把 EVCS 利润从理论上的：

$$
11.875 - 5 = 6.875
$$

变成：

$$
11.875 - 5 - 5 = 1.875
$$

测试里只要求：

```python
private_profit_proxy > 0
```

所以这个 bug 可能不会被测出来。

---

## 必须重构成买卖分离

不要再用一个 `energy_market_revenue = price * P * dt` 混合正负功率。

应该拆成：

$$
R^{energy,sell}_t
=================

p^{sell}*t [P_t]*+ \Delta t
$$

$$
C^{energy,buy}_t
================

p^{buy}*t [-P_t]*+ \Delta t
$$

代码应是：

```python
p_export = max(0.0, delivered_p_mw)
p_import = max(0.0, -delivered_p_mw)

energy_sell_revenue = export_sell_price * p_export * dt_hours
energy_buy_cost = wholesale_buy_price * p_import * dt_hours
```

然后 EVCS 的 wholesale cost 不应该再和全 VPP 的 `energy_buy_cost` 重复。
更稳妥做法是按 DER 类型拆分：

```text
pv_export_revenue
ess_charge_cost
ess_discharge_revenue
evcs_wholesale_cost
hvac_energy_cost
flex_load_energy_cost
microturbine_fuel_cost
```

---

# 8. 最大风险六：EVCS fallback 会把所有负功率都当成 EVCS 充电

implementation plan 里 EVCS settlement helper 有这个逻辑：

```python
charging_p_mw = audit.get("evcs_charging_p_mw", max(0.0, -delivered_p_mw))
```

这很危险。

如果某个 VPP 的负功率来自：

* 储能充电；
* HVAC 用电；
* 柔性负荷；
* EVCS；
* safety projection 后的吸收；

那么 `max(0, -delivered_p_mw)` 会把所有吸收功率都当成 EVCS 充电。

这样会出现严重错误：

> 本来是储能在充电，却被记录成 EV 用户充电收入。
> 本来是 HVAC 在用电，也被记录成 EVCS 收入。
> 本来是柔性负荷在吸收，也被算成用户充电收入。

这会把 VPP 利润虚高，并且让策略错误地偏好负功率。

---

## 必须改成 per-DER audit

EVCS settlement 必须只使用 EVCS 实际功率：

```python
evcs_charging_p_mw = sum(
    max(0.0, -der.p_mw)
    for der in vpp.der_list
    if der.type == "evcs"
)
```

或者在 dispatch audit 里明确写入：

```text
evcs_grid_p_mw
evcs_user_energy_mwh
ess_charge_p_mw
hvac_p_mw
flex_load_p_mw
```

不能用 VPP 总功率兜底。

---

# 9. 最大风险七：storage terminal value 可能被每一步重复加

implementation plan 里 storage helper 返回：

```python
storage_potential_shaping_reward
storage_terminal_value_proxy
storage_terminal_soc_target_penalty
```

然后在 v3 `private_profit_proxy` 中加：

```python
+ storage_terminal_value_proxy
+ storage_potential_shaping_reward
- storage_terminal_soc_target_penalty
```

如果这个逻辑每一步都执行，那么 `storage_terminal_value_proxy` 会在每个 step 奖励高 SOC。

这会导致 agent 学到：

$$
\text{SOC 越高越好}
$$

甚至：

> 储能永远充满，不愿放电。

设计文档自己也警告了这一点：terminal value 不能无脑给高，否则 agent 会学到“永远充满不放”；`kappa` 必须来自未来价格/服务价格，末端 SOC 要有目标区间或机会成本，放电收益、退化成本、效率损失都要进入真实利润。

---

## 正确做法

应该分清：

### potential shaping：每一步加

$$
F_t
===

## \gamma \Phi(s_{t+1})

\Phi(s_t)
$$

这个可以每一步加。

### terminal value：只在 episode 结束或截断时加

$$
R_T^{storage}
=============

## \kappa_T E^{stored}_T

\lambda_{socT}(SOC_T-SOC^{target})^2
$$

代码应该是：

```python
if done or truncated:
    terminal_value_reward = storage_terminal_value_weight * terminal_value
else:
    terminal_value_reward = 0.0
```

日志可以每一步记录：

```text
storage_terminal_value_proxy
```

但 reward 不能每一步都加。

---

# 10. 最大风险八：DSO reward 中使用 VPP 私有利润可能会把“转移支付”误当成社会福利

设计文档中 DSO reward 使用：

$$
W_t = \sum_i \Pi_{i,t}
$$

也就是所有 VPP 私有利润之和。这个作为过渡版本可以，但论文上要小心。

如果 VPP 利润里包含：

```text
service_payment
availability_payment
```

这些钱本质上是 DSO 支付给 VPP 的转移支付。

如果 DSO reward 又最大化 VPP 利润，而没有把 DSO 支付成本扣回来，就会出现：

$$
\text{DSO 多付钱}
\Rightarrow
\text{VPP 利润增加}
\Rightarrow
\text{DSO reward 增加}
$$

这在经济学上是错的。

真正的社会福利应该更接近：

$$
SW_t
====

\text{用户效用}
+
\text{能源服务价值}
-------------

## \text{发电/购电成本}

## \text{DER运行成本}

## \text{网损成本}

## \text{舒适损失}

## \text{未服务惩罚}

\text{安全风险成本}
$$

其中 service payment 和 availability payment 是 DSO 与 VPP 之间的内部转移，不能同时作为 DSO 成本之外的福利增加。

---

## 建议改法

DSO reward 里的 welfare 不要直接用完整 VPP private profit，而应该用：

```python
vpp_operational_surplus_ex_transfer
```

即：

$$
\Pi_i^{op}
==========

R^{energy,sell}
+
R^{EVCS,user}
-------------

## C^{energy,buy}

## C^{DER}

## C^{deg}

## C^{comfort}

C^{unserved}
$$

不要把 DSO 支付的：

```text
service_payment
availability_payment
```

直接加进 DSO welfare。

如果要计入服务价值，则应该计入 DSO 侧的物理/经济收益，例如：

$$
\text{avoided violation cost}
$$

$$
\text{avoided network reinforcement proxy}
$$

$$
\text{reduced loss cost}
$$

而不是简单用 VPP 收到的钱。

---

# 11. 最大风险九：Task 4 的 DSO welfare 传递可能无效

implementation plan 的 Task 4 写道：

1. 先计算 dispatch components；
2. 如果 reward-v3 active，就：

```python
dso_components = dict(dso_components)
dso_components["dso_vpp_welfare_proxy"] = aggregate_vpp_welfare(...)
```

3. 然后：

```python
dso_components_local = dso_reward_from_components(dso_components)
```

这可能有问题。

如果 `dso_reward_train` 已经在 `scenario.dso.calculate_reward_or_cost()` 里计算完了，那么你后面只是改了：

```python
dso_components["dso_vpp_welfare_proxy"]
```

但没有重新计算：

```python
dso_safe_gated_welfare_reward
dso_reward_train
dso_reward_env
reward
```

那 DSO reward 实际不会变。

---

## 正确流程

应该是：

```python
dispatch_components = compute_all_vpp_dispatch_components(...)
vpp_welfare_proxy = aggregate_vpp_welfare(dispatch_components)

dso_components = dso.calculate_reward_or_cost(
    ...,
    vpp_welfare_proxy=vpp_welfare_proxy,
    raw_action_safety_cost=raw_action_safety_cost,
)
```

也就是说：

> VPP settlement 必须先于 DSO reward-v3 计算完成。

如果当前框架已经先算 DSO 后算 VPP，就要重构 reward map 构建顺序。
不能只在 components dict 里补一个字段。

这是 implementation plan 里最需要优先检查的工程风险之一。

---

# 12. 最大风险十：balanced generation scenario 的验收指标还不够

implementation plan 的场景测试要求：

```python
assert positive_midpoints / len(scenario.vpps) >= 0.4
assert generation_capable >= 4
assert storage_capable >= 4
```

这是好的开始。

但还不够。

因为一个 VPP 静态中点为正，不代表系统真的会出现：

1. 反向潮流；
2. 高电压风险；
3. 线路/变压器过载；
4. DSO 需要协调 VPP 出力；
5. 储能有套利价值；
6. EVCS 有真实充电服务压力。

设计文档里其实还提出了：

```text
reverseflow_candidate_steps > 0
```

但 implementation plan 的测试没有写进去。

---

## 建议新增场景验收指标

至少增加：

```text
reverseflow_candidate_steps >= 某个阈值
high_voltage_risk_steps >= 某个阈值
peak_import_stress_steps >= 某个阈值
pv_available_peak_mw > load_low_period_mw
storage_arbitrage_spread > degradation_cost_threshold
evcs_deadline_pressure_steps > 0
```

尤其是：

$$
\text{reverseflow_candidate_steps} > 0
$$

必须加进测试。否则你只是加了几个发电型 VPP，但系统不一定真的进入高 PV / 低负荷的反送电压力场景。

---

# 13. 算法层面：HAPPO/HATRPO 方向正确，但计划缺少 KL/梯度稳定性修复

你之前的训练记录已经显示 dispatch actor 有严重 KL 超标问题。
这次 reward-v3 改造会显著改变 reward scale，如果不同时修训练稳定性，可能出现：

```text
reward 语义变正确了，但 actor 更新仍然发散。
```

当前 implementation plan 里主要关注 reward 和场景，没有明确加入：

```text
per-role learning rate
target KL
early stopping
advantage normalization
reward normalization
gradient clipping audit
```

这是不够的。

---

## 建议加入 reward-v3 专用训练配置

例如：

```yaml
algorithm:
  happo:
    dso_actor_lr: 5.0e-5
    dispatch_actor_lr: 5.0e-5
    critic_lr: 1.0e-4
    target_kl_dso: 0.01
    target_kl_dispatch: 0.005
    ppo_epochs: 4
    minibatches: 4
    max_grad_norm: 0.5
    per_role_advantage_norm: true
    value_norm: true
    reward_component_norm: true
```

并且每轮记录：

```text
role_approx_kl
role_clip_fraction
role_entropy
role_grad_norm_before_clip
role_grad_norm_after_clip
role_advantage_mean/std
role_value_loss
role_policy_loss
```

否则 reward-v3 后的实验仍然可能出现：

> critic loss 下降，但 reward/cost 不改善。

---

# 14. Safety shield 相关：计划方向正确，但实现要求更严格

设计文档明确指出，安全投影可以继续作为运行保护，但 reward 必须记录投影前和投影后的安全代价：

$$
C_t^{safe,raw}
==============

C^{safe}(a_t^{raw})
$$

$$
C_t^{safe,proj}
===============

C^{safe}(a_t^{proj})
$$

并且 DSO actor 训练 reward 至少要惩罚 raw action 导致的安全风险。

这是完全正确的。

但是 implementation plan 只是在函数签名里加了：

```python
raw_action_safety_cost: float = 0.0
```

这不够。关键问题是：

> raw_action_safety_cost 到底怎么计算？

如果它只是默认 0，或者只是从 post-projection report 里来，那 reward-v3 仍然没有解决 safety shield credit assignment 问题。

---

## 必须新增这些日志和测试

```text
raw_action_voltage_violation_cost
raw_action_line_overload_cost
raw_action_trafo_overload_cost
raw_action_powerflow_failed
projected_action_voltage_violation_cost
projected_action_line_overload_cost
projected_action_trafo_overload_cost
shield_intervention_frequency
projection_gap_local_bounds
projection_gap_ac_aware
projection_gap_ac_certificate
no_shield_eval_violation_rate
```

并且必须有测试：

```python
raw unsafe + projected safe => dso_reward_train decreases
```

否则安全外壳仍然会替策略兜底。

---

# 15. 需要补充的关键消融实验

当前 Task 6 的 staged experiments 是对的，包括 8-step smoke、96-step sanity、1-episode MARL smoke、96-step HATRPO sanity。

但对于论文和诊断还不够。必须加入以下消融：

| 消融                                       | 目的                     |
| ---------------------------------------- | ---------------------- |
| v2 vs v3                                 | 证明 reward-v3 确实修复原问题   |
| v3 without EVCS revenue                  | 证明 EVCS 用户收入是必要项       |
| v3 without storage terminal/shaping      | 证明储能跨时价值是必要项           |
| v3 without generation VPP mix            | 证明场景修复有效               |
| soft gate vs hard gate                   | 检验安全门控稳定性              |
| raw safety cost on/off                   | 证明 actor 不是靠 shield 安全 |
| availability/service double-count off/on | 检验是否被服务支付主导            |
| no-shield evaluation                     | 证明策略本身学会安全边界           |
| profit-only dispatch                     | 检验 VPP 经济逻辑是否闭合        |
| safety-only DSO                          | 检验 DSO 是否能学安全包络        |

尤其是 no-shield evaluation 必须做。否则审稿人会继续问：

> 到底是 MARL 学会了安全调度，还是 safety layer 把坏动作修好了？

---

# 16. 我建议你如何修改 implementation plan

我建议在正式执行前，把任务顺序改成下面这样。

---

## 修订版 Task 0：统一 reward 数学合同

先不要写代码，先冻结 reward-v3 公式：

### DSO reward

$$
r_t^{DSO}
=========

-\lambda_{safe} C_t^{safe,raw}
-\lambda_{proj} C_t^{proj}
-\lambda_{loss} \widetilde{P}*{loss,t}
-\lambda*{smooth}|E_t-E_{t-1}|^2
+
\beta G_t^{safe,raw}\widetilde{SW}_t
$$

其中：

* `SW_t` 是社会福利或 operational surplus，不是简单 VPP private profit；
* `G_safe` 用 normalized safety cost，不用加权 safety penalty；
* `curtailment/safe_capacity/envelope_width` 全部只做 diagnostic。

---

## 修订版 Task 1：先做 reward component instrumentation

先加日志，不急着训练：

```text
energy_sell_revenue
energy_buy_cost
evcs_user_revenue
evcs_wholesale_cost
ess_charge_cost
ess_discharge_revenue
storage_potential_shaping
storage_terminal_value_reward
service_payment
availability_payment
dso_payment_cost
vpp_operational_surplus
social_welfare_proxy
```

没有这些日志，后面 reward 又会变成黑箱。

---

## 修订版 Task 2：修 VPP settlement，避免重复计入

必须统一选择：

```text
真实利润统一入口
```

也就是：

$$
r_i^{dispatch}
==============

## \lambda_{\Pi}\Pi_i

## \text{projection penalty}

\text{constraint penalty}
$$

不要再把 service/availability 在 private profit 里加一次，在 reward 外再加一次。

---

## 修订版 Task 3：修 storage terminal value

把：

```text
storage_potential_shaping_reward
```

作为 step-level reward。

把：

```text
storage_terminal_value_reward
```

只在 episode done/truncated 时加入。

并新增：

```text
terminal_value_applied_flag
```

防止误加。

---

## 修订版 Task 4：重构 DSO reward 计算顺序

必须保证：

```text
先算 VPP settlement
再算 DSO reward-v3
```

而不是先算 DSO，再事后塞一个 welfare 字段。

---

## 修订版 Task 5：balanced generation scenario 增加真实压力验收

除了：

```text
positive_midpoint_ratio >= 0.4
generation_capable >= 4
storage_capable >= 4
```

还要加：

```text
reverseflow_candidate_steps > 0
high_voltage_risk_steps > 0
peak_import_stress_steps > 0
```

---

## 修订版 Task 6：加 reward-v3 专用训练稳定性配置

明确写入：

```text
target_kl_dispatch <= 0.005 or 0.01
per-role advantage normalization
reward/value normalization
gradient norm before/after clipping
early stopping on KL
```

这是你当前实验最需要补的算法稳定性约束。

---

# 17. 对两份文件的最终评价

## 设计文件评价

设计文件质量较高，核心方向正确：

1. 准确识别了 DSO curtailment proxy 主导问题；
2. 明确提出安全优先；
3. 正确补齐 EVCS 用户收入；
4. 正确引入储能 terminal value / potential shaping；
5. 正确要求增加发电型 VPP；
6. 正确提出 raw/projection safety cost；
7. 给出了比较完整的论文叙事。

但它仍需要补充：

1. DSO welfare 不能简单用 VPP private profit，因为 service/availability payment 是转移支付；
2. safety-first proof 需要映射到实际代码中的 bounded penalty 和 reward scale；
3. potential shaping 的时间变势函数和 terminal value 边界要更严格；
4. 需要明确 reward-v3 是 settlement proxy 还是完整市场出清闭环。

---

## 实现计划评价

实现计划的优点是工程拆解很清晰：

1. tests-first；
2. 保持 v1/v2 兼容；
3. 每个任务有文件清单；
4. 有 smoke test、96-step sanity、MARL smoke；
5. 有 scenario mix test；
6. 有 reward trace 扩展。

但它目前不应原样执行，主要问题是：

1. VPP welfare 只 clip、不标准化；
2. safety gate 可能严重饱和；
3. service/availability payment 可能重复计入；
4. energy_market_revenue 和 EVCS wholesale cost 可能双重扣费；
5. EVCS fallback 可能把所有负功率误算成 EVCS 充电；
6. storage terminal value 可能每一步重复加；
7. DSO welfare 传递顺序可能无效；
8. scenario test 未覆盖 reverseflow candidate；
9. 没有补充 KL/梯度/advantage 稳定性配置；
10. 没有完整 raw/no-shield safety evaluation。

---

# 18. 最终建议

我的建议是：

> **采用 reward-v3 作为下一阶段主线，但先修改 implementation plan，再让 agent 执行。**

优先级如下：

1. **先修 settlement 公式，避免重复计入和双重扣费。**
2. **先修 DSO welfare normalization，不要只 clip。**
3. **先修 storage terminal value，只在 terminal 加。**
4. **先修 DSO reward 计算顺序，确保 VPP welfare 真正进入 DSO reward。**
5. **加入 raw action safety cost 和 no-shield evaluation。**
6. **加入 reward-v3 专用 HAPPO/HATRPO 稳定性参数。**
7. **最后再跑 balanced generation paper-long。**

一句话总结：

> **这两份文件已经把研究方向从“reward proxy 调参”推进到了“安全-市场-跨时价值闭环”的正确路线，但 implementation plan 还需要一次严格的 reward 会计审计和 MARL 稳定性审计。否则 reward-v3 可能只是把旧的 curtailment proxy 问题，替换成新的 service payment 重复计入、EVCS 收入误归因和储能囤电问题。**
