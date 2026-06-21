
# 4. 仍然存在的关键问题与批判性建议

下面是我认为还必须修的地方。

---

# 问题 1：$\Pi^{op}$ 公式中 `R_export` 和 `R_ESS_discharge` 可能重复

你现在写的是：

$$
\Pi^{op}_{i,t}
==============

R^{export}*{i,t}
+
R^{EVCS,user}*{i,t}
+
R^{ESS,discharge}_{i,t}
-----------------------

...
$$

但是后面的 settlement helper 里：

```python
export_energy = (
    pv_export_p_mw
    + microturbine_export_p_mw
    + storage_discharge_p_mw
) * dt_hours
```

也就是说，`R_export` 已经包含了 storage discharge revenue。然后你又单独写了：

$$
R^{ESS,discharge}_{i,t}
$$

这在数学定义上有重复风险。

虽然你的代码返回里有：

```python
storage_discharge_revenue
```

但 operational surplus 里实际只用了 `export_revenue`，没有再加 `storage_discharge_revenue`。

所以问题是：

> **公式层面写法容易让后续实现者误以为 storage discharge 要加两次。**

建议把公式改成互斥分解：

$$
\Pi^{op}_{i,t}
==============

R^{PV,export}*{i,t}
+
R^{MT,export}*{i,t}
+
R^{ESS,discharge}*{i,t}
+
R^{EVCS,user}*{i,t}
-------------------

## C^{EVCS,wholesale}_{i,t}

## C^{ESS,charge}_{i,t}

## C^{HVAC,energy}_{i,t}

## C^{flex,energy}_{i,t}

## C^{DER,op}_{i,t}

## C^{deg}_{i,t}

## C^{comfort}_{i,t}

C^{unserved}_{i,t}
$$

或者保留：

$$
R^{export}
$$

但明确：

$$
R^{export}
==========

R^{PV,export}
+
R^{MT,export}
+
R^{ESS,discharge}
$$

并删除单独的 $R^{ESS,discharge}$。

---

# 问题 2：`C_import` 的定义仍然容易和 EVCS / ESS / HVAC 成本重复

你现在公式中有：

$$
-C^{import}_{i,t}
-----------------

## C^{EVCS,wholesale}_{i,t}

C^{ESS,charge}_{i,t}
$$

如果 $C^{import}$ 表示 VPP 总进口电费，那么 EVCS wholesale 和 ESS charge 又会重复扣费。

后面的代码实现中 `energy_buy_cost` 实际上只等于：

```python
load_energy_cost = wholesale_buy * (hvac_p_mw + flex_load_p_mw) * dt_hours
```

也就是它不是总 import cost，而更像 HVAC/flex load energy cost。

建议把 `C_import` 改名，不要叫 import。

改成：

$$
C^{load,energy}_{i,t}
=====================

C^{HVAC,energy}*{i,t}
+
C^{flex,energy}*{i,t}
$$

最终公式写成：

## $$

## C^{EVCS,wholesale}_{i,t}

## C^{ESS,charge}_{i,t}

## C^{HVAC,energy}_{i,t}

## C^{flex,energy}_{i,t}

C^{unclassified,buy}_{i,t}
$$

其中 `C_unclassified_buy` 只用于 audit 不完整时，而且 paper-long 中必须为 0 或不能使用。

---

# 问题 3：`settlement_audit_complete` 的判定过弱

你当前代码草案是：

```python
"settlement_audit_complete": float("evcs_grid_p_mw" in audit or evcs_grid_p_mw == 0.0)
```

这个逻辑有漏洞。

如果 audit 里没有 `evcs_grid_p_mw`，那么：

```python
evcs_grid_p_mw = 0.0
```

于是：

```python
"evcs_grid_p_mw" in audit or evcs_grid_p_mw == 0.0
```

仍然为 True。

也就是说：

> audit 缺失时，反而可能被判定为 complete。

这是严重问题。

建议改成：

```python
required_fields = required_settlement_fields_for_vpp(vpp)
settlement_audit_complete = all(field in audit for field in required_fields)
```

并且根据 VPP 设备类型动态确定 required fields：

| DER 类型        | 必须字段                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------ |
| EVCS          | `evcs_grid_p_mw`, `evcs_charge_efficiency`, `retail_evcs_tariff`                           |
| ESS           | `storage_charge_p_mw`, `storage_discharge_p_mw`, `storage_soc_before`, `storage_soc_after` |
| PV            | `pv_export_p_mw`                                                                           |
| Microturbine  | `microturbine_export_p_mw`, `der_operation_cost`                                           |
| HVAC          | `hvac_p_mw`, `comfort_cost`                                                                |
| flexible load | `flex_load_p_mw`, `comfort_cost` 或 `unserved_energy_cost`                                  |

然后加一条硬性规则：

> **paper-long reward-v3 MARL training 中，如果 `settlement_audit_complete=0`，直接 fail，不允许继续训练。**

你的文档已经说 simulator fallback 不能用于 paper-long MARL training，因为它无法区分 EVCS、储能、HVAC、柔性负荷和 projection-induced absorption。这个警告是正确的，但还需要强制机制。

---

# 问题 4：缺少 DER power balance audit

你现在 per-DER audit 很关键，但还需要一个功率平衡检查。

否则可能出现：

```text
evcs_grid_p_mw = 0.04
storage_charge_p_mw = 0.06
delivered_p_mw = -0.05
```

这三个数不一致。

建议新增：

$$
P^{audit}_{i,t}
===============

P^{PV}*{i,t}
+
P^{MT}*{i,t}
+
P^{ESS,dis}_{i,t}
-----------------

## P^{ESS,ch}_{i,t}

## P^{EVCS}_{i,t}

## P^{HVAC}_{i,t}

P^{flex}_{i,t}
$$

然后检查：

$$
|P^{audit}*{i,t} - P^{delivered}*{i,t}| \le \epsilon_P
$$

新增日志：

```text
settlement_power_balance_gap_mw
settlement_power_balance_ok
```

并新增测试：

```python
def test_v3_settlement_power_balance_gap_is_logged_and_bounded():
    ...
```

没有这个检查，per-DER audit 虽然看起来严谨，但可能和环境实际执行功率不一致。

---

# 问题 5：`min_raw_unsafe_penalty=1.0` 可能太硬

你在 DSO safety helper 里写：

```python
raw_penalty_input = max(raw_norm, min_raw_unsafe_penalty if raw_unsafe else 0.0)
```

这意味着只要 raw_norm 大于 0，不管是：

$$
0.0001
$$

还是：

$$
0.5
$$

至少都按 1.0 处罚。

优点是保证 safe-first proof 更容易成立。
缺点是会损失连续学习信号。

这可能导致：

> agent 无法区分“轻微越限”和“严重越限”，早期训练中安全项像硬墙一样，让策略更新不平滑。

建议改成平滑地板：

$$
C^{raw,penalty}
===============

C^{raw,norm}
+
\alpha \cdot \sigma(k(C^{raw,norm}-\epsilon))
$$

或者简单点：

```python
if raw_norm > raw_safety_epsilon:
    raw_penalty_input = raw_norm + min_raw_unsafe_penalty
else:
    raw_penalty_input = 0.0
```

也就是说，不要对浮点噪声产生硬惩罚。

建议新增配置：

```yaml
raw_safety_epsilon: 1.0e-5
min_raw_unsafe_penalty: 1.0
```

---

# 问题 6：safety gate 只看 raw safety，不看 projected safety，有边界问题

你现在定义：

$$
G^{safe}_t
==========

\exp(-\kappa C^{safe,raw,norm}_t)
$$

weighted penalty 里包含 projected cost：

$$
C^{safe,penalty}
================

...
+
\lambda_{proj} C^{proj}_t
$$

正常情况下没问题，因为 projected unsafe 会被 penalty 惩罚。

但有一种边界情况：

* raw safety cost 很低；
* projection / certificate 后仍然有 projected safety violation；
* 但 gate 仍然接近 1。

这时 VPP welfare 仍然可以较大程度进入 DSO reward。

建议 gate 使用：

$$
G^{safe}_t
==========

\exp
\left(
-\kappa
\max(C^{safe,raw,norm}_t, C^{safe,proj,norm}_t)
\right)
$$

或者：

$$
G^{safe}_t
==========

\exp
\left(
-\kappa
(C^{safe,raw,norm}_t + \rho C^{safe,proj,norm}_t)
\right)
$$

其中 $\rho$ 可以小于 1。

更保守的写法：

```python
gate_input = max(
    safety["dso_raw_safety_cost_norm"],
    safety["dso_projected_safety_cost_norm"],
)
```

这样只要最终执行动作仍然不安全，VPP welfare 就不会放大 DSO reward。

---

# 问题 7：welfare baseline 默认值可能导致过早 clip

你配置里默认：

```yaml
welfare_baseline_mean: 0.0
welfare_baseline_std: 1.0
welfare_clip: 5.0
```

如果 operational surplus per MWh 的数值天然是几十、几百，z-score 会马上被 clip 到 5。

这会导致：

> 不同经济方案又被压成同一个值，DSO 仍然分不清好坏。

你的文档已经规定可以用 baseline calibration 或 online running z-score。
但实现计划里还缺一个明确的 **baseline calibration task**。

建议在 Task 2 和 Task 9 之间加入：

## Task 2.5: Welfare Normalization Calibration

先用 rule-based / random / no-flex 跑 1 到 3 个短 episode，统计：

```text
vpp_operational_surplus_ex_transfer_per_mwh_mean
vpp_operational_surplus_ex_transfer_per_mwh_std
```

然后写入配置：

```yaml
welfare_baseline_mean: <calibrated_mean>
welfare_baseline_std: <calibrated_std>
```

否则 `baseline_std=1.0` 只是测试方便，不适合真实训练。

---

# 问题 8：potential shaping 和 terminal value 可能仍然双重计算终端储能价值

你现在有：

$$
F_t^{storage}
=============

\gamma \Phi(s_{t+1})-\Phi(s_t)
$$

又在 terminal step 加：

$$
R_T^{storage}
=============

## \kappa_T E_T^{stored}

\lambda_{socT}(SOC_T-SOC^{target})^2
$$

如果 potential shaping 从 $t=0$ 一直加到 $T-1$，折扣累加后会留下：

$$
\gamma^T \Phi(s_T)-\Phi(s_0)
$$

然后你又额外加了 terminal value：

$$
R_T^{storage}
$$

这可能导致终端储能价值被计算两次。

不是说一定错，但必须明确：

1. 如果 terminal value 是真实目标的一部分，那么 potential shaping 的终端残差要一起考虑；
2. 如果 potential shaping 只是辅助学习，不想改变最优策略，就应该设置 terminal potential 处理方式；
3. 如果两者都保留，就需要做消融验证是否诱导囤电。

建议新增测试：

```python
def test_storage_potential_plus_terminal_does_not_reward_hoarding_when_future_price_low():
    ...
```

以及一个场景：

* 未来价格低；
* SOC 高；
* terminal target 是 0.5；
* agent 不应继续充电。

---

# 问题 9：VPP dispatch reward 仍可能被 service/availability 诱导，而不是真实市场成交

你现在避免了重复计入，这是进步。
但是 private profit 里仍然包含：

$$
R^{service}*{i,t}
+
R^{availability}*{i,t}
$$

这没错，因为这是 VPP 私有收入。

但问题是：

> 这些 payment 从哪里来？

如果它们仍然是 baseline proxy，不是真正经过 DSO award / bid / clearing / delivery settlement 得来的，那么 VPP 仍可能被“可用性补贴”牵引，而不是真实市场成交牵引。

建议新增日志字段：

```text
service_payment_source = baseline_proxy | cleared_award | disabled
availability_payment_source = baseline_proxy | capacity_contract | disabled
```

并在论文实验中区分：

1. `v3_proxy_service_payment`
2. `v3_no_service_payment`
3. `v3_cleared_service_payment`

如果当前还没有完整市场出清，论文中就不能说已经实现完整 DSO-VPP 市场结算，只能说：

> reward-v3 uses a settlement-aware proxy with explicit EVCS retail revenue, DER operation cost, and transfer-payment separation.

---

# 问题 10：scenario physical stress acceptance 还需要 AC power flow 验证

你现在新增了：

```text
reverseflow_candidate_steps
high_voltage_risk_steps
peak_import_stress_steps
storage_arbitrage_spread
evcs_deadline_pressure_steps
```

这个比上一版只看 positive midpoint ratio 好得多。

但是 `profile_diagnostics()` 如果只是基于 profile 计算候选窗口，仍然不能证明系统真的出现了 AC 层面的高电压或反送电压力。

建议增加一个 **pre-control AC stress scan**：

在没有 DSO/VPP 控制或用 rule-based dispatch 的情况下跑一遍 AC 潮流，输出：

```text
base_reverseflow_steps
base_high_voltage_violation_or_near_violation_steps
base_line_loading_topk
base_trafo_loading_topk
base_voltage_max_pu
base_voltage_min_pu
```

然后场景验收不只看 profile，还看 AC 结果：

```python
assert diagnostics["base_reverseflow_steps"] > 0
assert diagnostics["base_voltage_max_pu"] > 1.04  # near-risk threshold
```

这样才能证明这个 balanced generation scenario 真的对 DSO 有物理压力。

---

# 问题 11：Task 8 的稳定性配置测试太静态

你现在有 YAML 测试，检查：

```python
target_kl_dispatch <= 0.005
per_role_advantage_norm is True
early_stop_on_kl is True
```

这个很好，但还不够。

因为 YAML 里有，不代表 trainer 真用了。

建议再加两个动态测试：

### 测试 1：配置被正确传入 trainer

```python
def test_reward_v3_stability_config_is_loaded_into_happo_config():
    ...
```

### 测试 2：人为制造大 KL 时 early stop 触发

```python
def test_happo_update_early_stops_when_dispatch_kl_exceeds_target():
    ...
```

并在输出里检查：

```text
role_kl_early_stop = 1
role_update_epochs_completed < ppo_epochs
```

否则可能出现：

> 配置文件写得很好，但实际训练循环没有用。

---

# 问题 12：Task 数量已经很多，建议拆成两个执行阶段

你现在 Task 0 到 Task 10 非常完整，但一次性让 agent 执行会很重，容易产生接口冲突。

我建议分成两个阶段。

---

## 阶段 A：reward 会计与安全诊断闭环

只做：

1. Task 0：Freeze Reward V3 Accounting Contract；
2. Task 1：Reward V3 Contract Tests；
3. Task 2：DSO Reward；
4. Task 3：VPP Settlement；
5. Task 4：Storage Value；
6. Task 5：Reward computation order；
7. Task 6：raw/projected safety diagnostics。

阶段 A 的目标不是训练，而是让 reward trace 可信。

验收标准：

```text
reward_components.csv 中所有 v3 字段完整；
无 DSO curtailment training 字段；
service/availability 不重复；
EVCS revenue 来自 audit；
terminal value 非终止步为 0；
raw unsafe/projected safe 会扣 DSO reward。
```

---

## 阶段 B：场景、算法稳定性与实验

再做：

1. Task 7：balanced generation scenario；
2. Task 8：HAPPO/HATRPO stability；
3. Task 9：minimal verification；
4. Task 10：ablation suite。

这样可以减少一次性改动造成的大规模接口冲突。

---

# 5. 我建议你在文档中直接补充的修改

下面这些可以直接加进你当前计划。

---

## 建议补充 1：Settlement power balance test

```python
def test_v3_settlement_der_audit_matches_delivered_power() -> None:
    components = vpp_dispatch_reward_components(
        ...
        audit={
            "pv_export_p_mw": 0.03,
            "storage_discharge_p_mw": 0.02,
            "storage_charge_p_mw": 0.04,
            "evcs_grid_p_mw": 0.01,
            "hvac_p_mw": 0.00,
            "flex_load_p_mw": 0.00,
        },
        delivered_p_mw=0.00,
    )

    assert components["settlement_power_balance_gap_mw"] == pytest.approx(0.0)
    assert components["settlement_power_balance_ok"] == pytest.approx(1.0)
```

---

## 建议补充 2：paper-long 禁止 incomplete audit

```python
def test_reward_v3_paper_long_rejects_incomplete_settlement_audit() -> None:
    ...
    assert raises_or_fails_when(settlement_audit_complete == 0)
```

---

## 建议补充 3：projected unsafe gate test

```python
def test_projected_unsafe_action_closes_safety_gate_even_if_raw_cost_low() -> None:
    ...
    assert unsafe["dso_safety_gate"] < safe["dso_safety_gate"]
```

---

## 建议补充 4：welfare calibration task

在 Task 2 后加：

```text
Task 2.5: Welfare normalization calibration
```

输出：

```text
welfare_baseline_mean
welfare_baseline_std
welfare_clip_saturation_rate
```

并要求：

```text
welfare_clip_saturation_rate < 10%
```

否则说明 clip 太频繁，经济信号失真。

---

## 建议补充 5：storage anti-hoarding test

```python
def test_storage_temporal_value_does_not_reward_charging_when_future_price_below_buy_price() -> None:
    ...
    assert storage_potential_shaping_reward < 0
```

这能防止储能 shaping 变成无脑充电奖励。

---

# 6. 最终评价

这版文档已经从“能不能改 reward”升级到了“如何建立一个可审计、可复现、可解释的 MARL reward-v3 实验体系”。

我给它的评价是：

| 维度      | 评价                                                                                                     |
| ------- | ------------------------------------------------------------------------------------------------------ |
| 研究方向    | 很好，已经对准 safety-first + market-aware + terminal value                                                   |
| 相比上一版   | 明显改进，修复了主要重复奖励和安全 gate 问题                                                                              |
| 工程可执行性  | 较高，但一次性执行风险较大                                                                                          |
| 论文可信度   | 比 v2 强很多，尤其是 no-shield eval 和 transfer-payment separation                                              |
| 仍需修正    | settlement 公式歧义、audit 完整性、power balance、welfare calibration、storage double-count、projected unsafe gate |
| 是否可作为主线 | 可以，但建议先分阶段执行                                                                                           |

最终建议：

> **这版 reward-v3 计划可以作为下一阶段主实施方案，但在执行前应补充 settlement power balance、audit completeness、welfare calibration、projected unsafe gate、storage anti-hoarding 这五类测试。否则 v3 虽然修掉了旧 reward 的 curtailment proxy 问题，但仍可能引入新的结算误差和储能激励偏差。**
