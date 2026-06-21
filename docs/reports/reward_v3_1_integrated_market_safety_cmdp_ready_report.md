# Reward V3.1 Market-Safety-CMDP-Ready 研究与实施报告

## 面向 VPP-DSO 协同调度的安全优先、结算可审计、多智能体强化学习框架

**版本**：Integrated Revision 1.0
**日期**：2026-06-09
**适用对象**：`pandapower-vpp-dso-sim`、Reward V3 机制设计、paper-long 实验、论文方法章节与实验章节
**整合来源**：

1. `2026-06-09-reward-v3-market-safety-terminal-value.md`
2. `reward_v3_feedback_soc_storage_safety_market_data_review_20260609.md`

---

# 摘要

本报告将原始 **Reward V3 Market-Safety Implementation Plan** 与后续 **Reward V3 反馈问题审查报告** 整合为一版更适合论文主实验、工程实现和后续安全 MARL 扩展的完整研究计划。整合后的 Reward V3.1 不再把目标理解为“增加更多 reward 项”，而是收敛为：

> **只保留具有明确物理意义、市场结算依据、可审计证据链且不会制造错误激励的 reward 机制。**

本报告的核心修订包括：

1. **DSO reward 保持安全优先，但避免把 safety shield 的修正效果误认为智能体学习成果。**
2. **VPP dispatch reward 使用 DER-level settlement，避免 service/availability/payment double counting。**
3. **EVCS 收入必须来自 EVCS 审计字段，不能由 VPP 总负功率推断。**
4. **储能只保留净跨时价值，不默认启用无合同依据的 SOC target。**
5. **service payment、availability payment、contract shortfall、non-delivery penalty 默认关闭或标记为 proxy。**
6. **paper-long 训练必须通过 settlement audit、power balance、welfare calibration、raw/projected/no-shield safety diagnostics、KL/gradient stability logging。**
7. **为不落后于先进 VPP-MARL 方向，新增 CMDP/Lagrangian-ready 扩展、GNN/Transformer 编码器、partial observability、uncertainty-aware evaluation、多种强基线和标准化 MARL 评估协议。**

---

# 1. 整合后的研究定位

## 1.1 论文方法定位

本研究不应表述为：

> 本文实现了完整 DSO-VPP 灵活性市场清算。

更准确的定位应为：

> 本文提出一种 safety-first、settlement-aware、audit-driven 的多智能体强化学习 reward 设计框架。该框架在 VPP 侧显式区分能量收入、EVCS 用户收入、储能跨时价值和设备成本；在 DSO 侧使用剔除转移支付后的 operational surplus 作为福利输入；同时通过 raw/projected/no-shield safety diagnostics 审计策略本身的安全性。灵活性服务费、可用性费和合同履约项在完整 bid-award-delivery-settlement 机制实现前，仅作为可关闭的 proxy 或 ablation，不作为主实验真实市场收益。

这一定义比“完整市场清算”更严谨，也比普通 reward shaping 更可审计。它承认当前阶段尚未完成完整合同出清，但已经建立了可追踪的结算口径、安全诊断和训练稳定性框架。

## 1.2 与先进 VPP-MARL 方法的对齐

Reward V3.1 应从一开始保留以下先进方法接口：

1. **CTDE，Centralized Training with Decentralized Execution**
   训练时 critic 可以访问全局网络状态和多 VPP 信息；执行时每个 agent 只使用自己可观测的信息。

2. **HAPPO/HATRPO/MAPPO 作为主 MARL 算法族**
   DSO、VPP dispatch、EVCS、storage 等智能体具有异构动作空间和异构目标，因此需要支持异构多智能体算法。

3. **Constrained MARL / CMDP / Lagrangian-ready 接口**
   当前主线仍是 safety-first reward shaping，但必须为后续成本 critic、约束乘子、primal-dual 更新预留接口。

4. **GNN 编码器用于网络拓扑建模**
   配电网天然是图结构。DSO policy 或 DSO critic 应支持 bus/line/transformer/VPP-PCC 图编码。

5. **Transformer 或 temporal encoder 用于 EVCS、价格和负荷序列**
   EV 到达、离开、充电 deadline、电价和 PV/负荷预测具有强时序性，应为时序编码器预留接口。

6. **Partial observability 与隐私机制**
   VPP 不应默认暴露所有 DER 内部状态给 DSO。需要支持 full information、representative data、privacy-preserving envelope 等多种信息模式。

7. **标准化多 seed 评估**
   paper-long 不能只报单 seed 最优曲线，应报告均值、标准差、置信区间、失败率和 NaN 率。

---

# 2. 两份文档的取舍原则

## 2.1 保留原始 Reward V3 计划中的核心内容

保留：

1. 不删除 v1/v2 reward，新增严格的 `v3_market_safety` 路径。
2. VPP settlement 必须先于 DSO reward 计算。
3. DSO welfare 使用剔除 transfer payment 的 operational surplus。
4. welfare 需要 per-MWh normalization、baseline/running z-score、clip saturation 监控。
5. safety gate 使用 raw/projected normalized safety cost，而不是 weighted safety penalty。
6. EVCS 收入必须来自 per-DER EVCS audit。
7. settlement audit 必须检查 required fields 和功率平衡。
8. storage terminal value 不能每步加入 reward。
9. paper-long 前必须通过 Phase A / Phase B gate。
10. HAPPO/HATRPO 需要 KL early stop、per-role advantage/value normalization 和 gradient audit。

## 2.2 修正原始计划中不合理的默认项

| 原始设计 | 问题 | Reward V3.1 修正后默认 |
|---|---|---|
| `storage_terminal_soc_target: 0.5` | 无合同或循环边界依据时不严谨 | `storage_terminal_soc_reference_mode: disabled` |
| `storage_terminal_value_weight: 0.05` | 可能与 potential shaping 重复激励囤电 | 主实验默认 `0.0`，作为 ablation 启用 |
| `storage_potential_shaping_weight: 0.05` | 偏强，可能诱导充电 | 默认降为 `0.02` |
| `min_raw_unsafe_penalty: 1.0` | 对微小 raw violation 过强 | 默认降为 `0.1` |
| `raw_action_safety_weight: 20.0` | 与 floor 叠加后经济信号可能被压制 | 默认改为 `10.0` |
| `service_payment_source: baseline_proxy` | 容易被误解为真实市场收入 | 主实验默认 `disabled` |
| `availability_payment_source: baseline_proxy` | 没有容量合同和 opportunity cost | 主实验默认 `disabled` |
| `contract_delivery_weight > 0` | 没有 bid-award-delivery-settlement 链条 | 默认 `0.0` |

---

# 3. Reward V3.1 总体架构

## 3.1 智能体角色

```text
DSO agent
  输出 network-aware operating envelope / flexibility guidance
  目标：电压安全、线路/变压器不过载、网损低、包络平滑、系统福利可控

VPP dispatch agents
  在 DSO envelope 内控制各自 DER 聚合行为
  目标：真实结算利润、EVCS 用户服务、储能跨时价值、设备约束满足

DER asset models
  PV、ESS、EVCS、HVAC、flex load、microturbine
  负责物理状态更新、审计字段输出、功率平衡
```

DSO 不直接控制 VPP 内部 DER；VPP dispatch agent 不能篡改 SOC target 或结算价格；DER asset model 负责真实物理状态和 settlement audit。

## 3.2 计算顺序

Reward V3.1 的计算顺序必须固定：

```text
Step 1: agent raw actions
Step 2: local bound / DER physical feasibility check
Step 3: AC-aware safety projection / shield
Step 4: DER physical state update
Step 5: DER-level settlement audit
Step 6: VPP private settlement and operational surplus
Step 7: aggregate VPP operational surplus excluding transfer payments
Step 8: DSO raw/projected/post/no-shield safety diagnostics
Step 9: DSO safety-gated welfare reward
Step 10: reward trace export and training stability logs
```

核心原则：

> **先算 VPP 真实结算，再算 DSO welfare；先记录 raw action 安全，再允许 safety projection 修正；先证明 reward 口径可信，再做 paper-long 训练。**

---

# 4. 功率符号与 DER-level settlement audit

## 4.1 功率符号约定

继续使用项目内部约定：

```text
P > 0：向电网注入功率
P < 0：从电网吸收功率
```

例如：

| 行为 | 功率符号 |
|---|---|
| PV 发电上网 | 正 |
| 储能放电 | 正 |
| 微型燃机发电 | 正 |
| EVCS 充电 | 负 |
| 储能充电 | 负 |
| HVAC 用电 | 负 |
| 柔性负荷用电 | 负 |

不能仅凭 aggregate VPP power 推断 DER 类型。一个负的 VPP net power 可能来自 EVCS 充电、储能充电、HVAC、flex load 或 projection-induced absorption。因此 reward-v3 必须使用 per-DER audit 字段进行收入和成本分配。

## 4.2 DER-level settlement audit 的意义

DER-level settlement audit 是：

> 每个 VPP 在每个时间步都必须提交一份“分 DER 账本”，说明 PV 发了多少、储能充/放了多少、EVCS 从电网取了多少、HVAC 和柔性负荷消耗了多少，以及这些功率对应的收入和成本。

它不是普通 logging，而是 reward-v3 的有效性前提。

没有 DER audit，系统会出现三类错误：

1. **收入归因错误**：例如把储能充电视为 EVCS 用户充电收入。
2. **成本重复计算**：例如 aggregate import cost 已经扣过，又单独扣 EVCS wholesale cost。
3. **功率不守恒**：例如 delivered power = -0.1 MW，但 audit 中 EVCS、ESS、HVAC 加总只有 -0.04 MW。

## 4.3 功率平衡公式

对第 \(i\) 个 VPP，第 \(t\) 个时间步：

```math
P^{audit}_{i,t}
=
P^{PV}_{i,t}
+
P^{MT}_{i,t}
+
P^{ESS,dis}_{i,t}
-
P^{ESS,ch}_{i,t}
-
P^{EVCS}_{i,t}
-
P^{HVAC}_{i,t}
-
P^{flex}_{i,t}
```

并要求：

```math
\left|P^{audit}_{i,t}-P^{delivered}_{i,t}\right|
\le
\epsilon_P
```

| 符号 | 含义 |
|---|---|
| \(P^{audit}_{i,t}\) | 由 DER 审计字段重构出的 VPP 净功率 |
| \(P^{PV}_{i,t}\) | PV 注入电网功率 |
| \(P^{MT}_{i,t}\) | microturbine / CHP 注入电网功率 |
| \(P^{ESS,dis}_{i,t}\) | 储能放电功率 |
| \(P^{ESS,ch}_{i,t}\) | 储能充电功率 |
| \(P^{EVCS}_{i,t}\) | EVCS 充电取电功率 |
| \(P^{HVAC}_{i,t}\) | HVAC 用电功率 |
| \(P^{flex}_{i,t}\) | 柔性负荷用电功率 |
| \(P^{delivered}_{i,t}\) | VPP 对配电网实际交付的净功率 |
| \(\epsilon_P\) | 功率平衡容差 |

paper-long 训练中必须满足：

```text
settlement_audit_complete = 1
settlement_power_balance_ok = 1
```

---

# 5. VPP 结算合同

## 5.1 Operational surplus

第 \(i\) 个 VPP 在第 \(t\) 步的运行盈余定义为：

```math
\begin{aligned}
\Pi^{op}_{i,t}
=&
R^{PV,export}_{i,t}
+
R^{MT,export}_{i,t}
+
R^{ESS,discharge}_{i,t}
+
R^{EVCS,user}_{i,t}
\\
&
-
C^{EVCS,wholesale}_{i,t}
-
C^{ESS,charge}_{i,t}
-
C^{HVAC,energy}_{i,t}
-
C^{flex,energy}_{i,t}
-
C^{unclassified,buy}_{i,t}
\\
&
-
C^{DER,op}_{i,t}
-
C^{deg}_{i,t}
-
C^{comfort}_{i,t}
-
C^{unserved}_{i,t}
\end{aligned}
```

这是 **不包含 DSO-to-VPP transfer payment 的真实运行盈余**。

| 符号 | 含义 |
|---|---|
| \(\Pi^{op}_{i,t}\) | VPP operational surplus，不含转移支付 |
| \(R^{PV,export}_{i,t}\) | PV 上网收入 |
| \(R^{MT,export}_{i,t}\) | 微型燃机/CHP 上网收入 |
| \(R^{ESS,discharge}_{i,t}\) | 储能放电卖电收入 |
| \(R^{EVCS,user}_{i,t}\) | EVCS 向 EV 用户收取的充电收入 |
| \(C^{EVCS,wholesale}_{i,t}\) | EVCS 从电网购电的批发成本 |
| \(C^{ESS,charge}_{i,t}\) | 储能充电购电成本 |
| \(C^{HVAC,energy}_{i,t}\) | HVAC 用电成本 |
| \(C^{flex,energy}_{i,t}\) | 柔性负荷用电成本 |
| \(C^{unclassified,buy}_{i,t}\) | 未分类购电成本，仅允许 smoke test 使用 |
| \(C^{DER,op}_{i,t}\) | DER 运行成本，例如燃料成本 |
| \(C^{deg}_{i,t}\) | 电池退化成本 |
| \(C^{comfort}_{i,t}\) | 舒适度损失成本 |
| \(C^{unserved}_{i,t}\) | 未服务负荷或未满足 EV 需求成本 |

## 5.2 Export revenue 不得重复计数

如果实现中保留 aggregate export 字段，只能作为日志汇总：

```math
R^{export,total}_{i,t}
=
R^{PV,export}_{i,t}
+
R^{MT,export}_{i,t}
+
R^{ESS,discharge}_{i,t}
```

不得在 reward 中同时加入：

```text
R_PV_export + R_MT_export + R_ESS_discharge + R_export_total
```

否则就是重复奖励上网电量。

## 5.3 Private profit

VPP 私有利润定义为：

```math
\Pi^{private}_{i,t}
=
\Pi^{op}_{i,t}
+
R^{service}_{i,t}
+
R^{availability}_{i,t}
-
C^{contract}_{i,t}
```

Reward V3.1 对该式作如下修订：

```text
主实验默认：
R_service = 0
R_availability = 0
C_contract = 0
```

只有当实验明确启用 proxy 或真实市场清算模块时，才允许这些项非零。

| 项 | 默认处理 | 原因 |
|---|---|---|
| \(R^{service}_{i,t}\) | disabled | 当前未实现 bid-award-delivery-settlement |
| \(R^{availability}_{i,t}\) | disabled | 当前未实现容量合同和机会成本 |
| \(C^{contract}_{i,t}\) | disabled | 当前没有真实 contract award 和 baseline rule |

## 5.4 VPP dispatch reward

主实验默认：

```math
r^{dispatch}_{i,t}
=
\lambda_\Pi \Pi^{private}_{i,t}
-
\lambda_{proj}C^{proj}_{i,t}
-
\lambda_{constraint}C^{constraint}_{i,t}
+
\lambda_F F^{storage}_{i,t}
+
I^{terminal}_{t}R^{storage,T}_{i,t}
```

在默认配置中，因为 service/availability/contract 关闭：

```math
\Pi^{private}_{i,t}
=
\Pi^{op}_{i,t}
```

关键规则：

```text
service_payment 和 availability_payment 如果已经进入 Pi_private，
dispatch reward 不能再额外单独加一次。
```

---

# 6. DSO reward 合同

## 6.1 DSO welfare 输入

DSO 不使用 VPP private profit，而使用所有 VPP operational surplus 的总和：

```math
SW^{proxy}_t
=
\sum_i
\Pi^{op}_{i,t}
```

原因是：

```text
service_payment / availability_payment
对 VPP 是收入，
但对 DSO 或系统整体是转移支付或成本，
不能被误当成社会福利。
```

## 6.2 Welfare normalization

先做 per-MWh normalization：

```math
W^{perMWh}_t
=
\frac{SW^{proxy}_t}
{\max(N_{VPP}\Delta t,\epsilon)}
```

再做 z-score 与裁剪：

```math
\widetilde{W}_t
=
clip
\left(
\frac{
W^{perMWh}_t-\mu_W
}{
\sigma_W+\epsilon
},
-
W_{clip},
W_{clip}
\right)
```

paper-long 前必须完成 welfare calibration，要求：

```text
welfare_clip_saturation_rate < 10%
```

## 6.3 安全成本分层

Reward V3.1 必须记录四套安全成本：

```text
1. raw_action safety
2. projected_action safety
3. post_AC execution safety
4. no_shield evaluation safety
```

### Raw safety

```math
C^{safe,raw,norm}_t
=
C^{V,raw}_t
+
C^{L,raw}_t
+
C^{T,raw}_t
+
C^{PF,raw}_t
```

### Projected safety

```math
C^{safe,proj,norm}_t
=
C^{V,proj}_t
+
C^{L,proj}_t
+
C^{T,proj}_t
+
C^{PF,proj}_t
```

### Weighted safety penalty

```math
C^{safe,penalty}_t
=
\lambda_V C^{V,raw}_t
+
\lambda_L C^{L,raw}_t
+
\lambda_T C^{T,raw}_t
+
\lambda_{PF} C^{PF,raw}_t
+
\lambda_{proj}C^{proj}_t
```

## 6.4 Safety gate

```math
G^{safe}_t
=
\exp
\left(
-
\kappa
\max
\left(
C^{safe,raw,norm}_t,
C^{safe,proj,norm}_t
\right)
\right)
```

使用 `max(raw, projected)` 的原因是：

```text
raw 不安全：说明智能体还没学会安全动作；
projected 不安全：说明修正后执行仍有风险；
两者任一不安全，welfare 都不应被放大。
```

## 6.5 DSO reward

```math
r^{DSO}_t
=
-
C^{safe,penalty}_t
-
\lambda_{loss}\widetilde{P}_{loss,t}
-
\lambda_{smooth}
\left\|
E_t-E_{t-1}
\right\|^2
+
\beta
G^{safe}_t
\widetilde{W}_t
```

在 v3 中，curtailment、safe capacity utilization、envelope width 只作为 diagnostic，不进入训练 reward。

---

# 7. `min_raw_unsafe_penalty` 与 `raw_safety_epsilon` 修订

## 7.1 原始问题

原计划曾建议：

```yaml
raw_action_safety_weight: 20.0
min_raw_unsafe_penalty: 1.0
raw_safety_epsilon: 1.0e-5
```

这意味着只要 raw safety cost 超过 epsilon，就至少触发：

```math
20.0 \times 1.0 = 20.0
```

而 welfare 通常被裁剪在：

```math
\widetilde{W}_t \in [-5,5]
```

因此微小 raw unsafe 可能完全压制经济信号。

## 7.2 修正后默认

建议主实验默认：

```yaml
raw_safety_epsilon: 1.0e-5
min_raw_unsafe_penalty: 0.1
raw_action_safety_weight: 10.0
projected_action_safety_weight: 5.0
```

对应逻辑：

```python
if raw_norm > raw_safety_epsilon:
    raw_penalty_input = raw_norm + min_raw_unsafe_penalty
else:
    raw_penalty_input = 0.0
```

## 7.3 参数扫描

| 实验 | `min_raw_unsafe_penalty` | `raw_action_safety_weight` | 目的 |
|---|---:|---:|---|
| A | 0.0 | 10.0 | 检查没有 floor 时是否依赖 shield |
| B | 0.05 | 10.0 | 温和安全优先 |
| C | 0.1 | 10.0 | 推荐主实验 |
| D | 0.5 | 10.0 | 强安全优先 |
| E | 1.0 | 20.0 | 极强安全优先，仅作压力上界 |

观察指标：

```text
raw_action_violation_rate
projected_action_violation_rate
post_ac_violation_rate
no_shield_eval_violation_rate
dso_safety_gate_mean
dso_safety_gate_p10/p50/p90
welfare_clip_saturation_rate
policy_entropy
role_approx_kl
reward_improvement
```

---

# 8. 储能 temporal value 修订

## 8.1 储能为什么需要跨时价值

储能与普通负荷不同，它的价值来自跨时间搬运能量：

```text
低价充电 → 高价放电
高电压时吸收功率 → 后续释放
保留 SOC → 后续响应 DSO flexibility need
```

但是如果简单奖励高 SOC，智能体会学会囤电。

## 8.2 修订后的 potential function

设：

| 符号 | 含义 |
|---|---|
| \(SOC_t\) | 当前储能 SOC |
| \(E^{cap}\) | 储能容量，MWh |
| \(E_t=SOC_tE^{cap}\) | 当前储存能量 |
| \(p^{buy}_t\) | 当前购电价格 |
| \(\hat{p}^{sell/service}_{t:H}\) | 未来窗口内可售电价或服务价值预测 |
| \(\eta_{ch}\) | 充电效率 |
| \(\eta_{dis}\) | 放电效率 |
| \(c^{deg}\) | 电池退化成本，按 MWh 计 |

定义未来净储能价值系数：

```math
\kappa^{net}_t
=
\max
\left(
0,
\eta_{dis}
\mathbb{E}_{\tau \in (t,t+H]}
[
p^{sell/service}_{\tau}
]
-
c^{deg}
\right)
```

势函数：

```math
\Phi(s_t,t)
=
\kappa^{net}_t
SOC_t
E^{cap}
```

potential shaping：

```math
F^{storage}_t
=
\gamma
\Phi(s_{t+1},t+1)
-
\Phi(s_t,t)
```

完整储能 reward 必须同时包含当前成本：

```math
r^{ESS}_t
=
R^{discharge}_t
-
C^{charge}_t
-
C^{deg}_t
-
C^{effloss}_t
+
\lambda_F F^{storage}_t
+
I^{terminal}_t R^{storage,T}_t
```

## 8.3 反囤电条件

如果：

```math
\eta_{ch}\eta_{dis}
\mathbb{E}_{\tau>t}
[
p^{sell/service}_{\tau}
]
<
p^{buy}_t
+
c^{deg}
```

且当前 SOC 已经偏高，则继续充电的 reward 应小于不动作：

```math
r(charge) < r(no\ action)
```

至少必须满足：

```math
F^{storage}_t(charge) < 0
```

## 8.4 Terminal value 修订

原始公式：

```math
R^{storage,T}_t
=
\mathbf{1}_{done\lor truncated}
\left(
\kappa_T E^{stored}_T
-
\lambda_{socT}
(SOC_T-SOC^{target})^2
\right)
```

Reward V3.1 修订为默认关闭 SOC target：

```math
R^{storage,T}_t
=
I^{terminal}_t
\left(
\kappa_T E^{stored}_T
-
\lambda_{socRef}
\cdot
\mathbf{1}_{mode\neq disabled}
(SOC_T-SOC^{ref})^2
\right)
```

默认：

```yaml
storage_terminal_value_weight: 0.0
storage_terminal_soc_reference_mode: disabled
storage_terminal_soc_reference: null
storage_terminal_soc_reference_weight: 0.0
```

只有在以下场景启用：

| mode | 含义 |
|---|---|
| `disabled` | 默认，无终端 SOC reference |
| `initial_soc` | 循环 episode，要求 \(SOC_T \approx SOC_0\) |
| `reserve_requirement` | 明确备用容量实验 |
| `explicit_contract` | 明确合同要求 |

## 8.5 必须记录的储能审计字段

```text
storage_soc_before
storage_soc_after
storage_capacity_mwh
storage_charge_p_mw
storage_discharge_p_mw
storage_future_value_kappa
storage_phi_before
storage_phi_after
storage_potential_shaping_reward
storage_terminal_value_proxy
storage_terminal_value_reward
storage_terminal_value_applied_flag
storage_terminal_potential_residual
storage_anti_hoarding_pass
```

---

# 9. Service / availability / contract 机制修订

## 9.1 主实验默认关闭

主实验配置：

```yaml
service_payment_weight: 0.0
availability_payment_weight: 0.0
contract_delivery_weight: 0.0
service_payment_source: disabled
availability_payment_source: disabled
contract_settlement_source: disabled
```

## 9.2 为什么关闭

真实灵活性服务市场至少需要：

```text
VPP bid
DSO award
baseline rule
delivery measurement
shortfall tolerance
non-delivery penalty
settlement record
DSO budget accounting
```

当前 service payment、availability payment 和 contract shortfall 更像 reward proxy，而不是完整 market clearing。因此不能作为论文主实验真实市场收益。

## 9.3 允许作为 ablation

可以保留：

```yaml
service_payment_source:
  disabled | baseline_proxy | cleared_award

availability_payment_source:
  disabled | baseline_proxy | capacity_contract
```

但论文必须标注：

```text
baseline_proxy is not full market clearing.
```

## 9.4 后续 Phase D：真实灵活性市场扩展

如果后续要升级为真实市场机制，需要新增：

```math
b_{i,t}
=
(q^{up}_{i,t},q^{down}_{i,t},\pi^{up}_{i,t},\pi^{down}_{i,t},node_i,duration)
```

其中 \(b_{i,t}\) 是 VPP bid。

DSO clearing：

```math
q^{award}_{i,t}
=
\arg\min
\left[
C^{DSO}_{network}
+
\sum_i
\pi_{i,t}q_{i,t}
\right]
```

subject to AC constraints。

交付偏差：

```math
e^{delivery}_{i,t}
=
q^{award}_{i,t}
-
q^{delivered}_{i,t}
```

未履约成本：

```math
C^{shortfall}_{i,t}
=
\lambda_{shortfall}
[
e^{delivery}_{i,t}
]_+^2
```

在 Phase D 前，这些都不作为真实市场 reward。

---

# 10. 数据接入策略

## 10.1 EVCS 数据

建议优先使用 ACN-Data 作为 EVCS 真实会话来源。

Reward V3.1 需要 ACN adapter 输出：

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

转成 15-min step：

```text
arrival_step
departure_step
deadline_step
requested_energy_mwh
delivered_energy_mwh
max_charge_mw
payment_required
```

EVCS audit：

```text
evcs_grid_p_mw
evcs_user_energy_mwh
evcs_unserved_energy_mwh
evcs_deadline_pressure
evcs_charge_efficiency
retail_evcs_tariff
```

## 10.2 电价数据

OpenEI/URDB 适合 retail tariff，但不能直接当成一列 `price[t]`。需要 TariffAdapter 转换复杂 tariff 结构。

TariffAdapter 统一输出：

```text
retail_buy_price_per_mwh[t]
wholesale_buy_price_per_mwh[t]
export_sell_price_per_mwh[t]
evcs_retail_price_per_mwh[t]
demand_charge_per_mw[t]
```

## 10.3 配电网与负荷场景

SMART-DS 可作为 realistic-but-synthetic distribution network 和 profile 数据源。

| 数据类型 | 推荐来源 | 用途 |
|---|---|---|
| EVCS session | ACN-Data | 真实 EV 到达、离开、需求、已充电量 |
| Tariff | OpenEI/URDB | 零售电价、TOU、电需量费 |
| 配电网拓扑/负荷 | SMART-DS | feeder、load、DER 场景 |
| DER 标准约束 | IEEE 1547 / 本地配置 | 电压、逆变器能力、DER 并网约束 |

---

# 11. MARL 算法设计

## 11.1 主算法族

建议保留以下算法族：

```text
MAPPO
HAPPO
HATRPO
MATD3 / HATD3
MASAC / graph-SAC optional
Constrained MAPPO / Lagrangian MAPPO optional
```

## 11.2 为什么不能只做 HAPPO/HATRPO

HAPPO/HATRPO 适合 DSO、VPP dispatch、portfolio 等异构 agent，但当前电力系统安全问题还具有显式约束特征。因此应保留两条线：

```text
Reward V3.1 主线：
  safety-first reward shaping + raw/projected/no-shield audit

Reward V3.5 / V4 扩展：
  constrained Markov game + Lagrangian cost critic
```

论文中不应声称：

```text
guaranteed safe MARL
```

更合理的说法是：

```text
safety-first reward shaping with auditable shield-dependence metrics
```

或：

```text
settlement-aware MARL with raw/projected/no-shield safety diagnostics, designed to be extensible to constrained MARL.
```

## 11.3 CTDE 信息结构

训练时 critic 可以访问：

```text
network state
bus voltages
line loading
trafo loading
all VPP envelope
aggregated DER audit
price streams
forecast window
constraint report
```

执行时：

```text
DSO agent:
  可访问网络聚合状态、局部监测点、VPP flexibility bids/envelopes

VPP dispatch agent:
  可访问自己的 DER 状态、DSO envelope、局部价格、EVCS session、forecast

EVCS / storage sub-controller:
  可访问本资产状态、deadline、SOC、局部 tariff
```

## 11.4 GNN / Transformer 先进扩展

### Graph encoder

用于 DSO critic 或 DSO policy：

```text
nodes: buses / VPP PCC / DER buses
edges: lines / transformers
node features: voltage, load, PV, ESS SOC, EVCS demand, envelope bounds
edge features: impedance, loading, thermal limit
```

### Transformer temporal encoder

用于 VPP dispatch 或 EVCS agent：

```text
input sequence:
  price[t:t+H]
  load forecast[t:t+H]
  PV forecast[t:t+H]
  EVCS arrival/departure/deadline features
  SOC trajectory features
```

---

# 12. Reward V3.1 推荐默认 YAML

```yaml
reward:
  version: v3_market_safety
  critic_reward_scale: 0.01

  dso:
    # v3 不使用削减类训练项
    curtailment_cost_weight: 0.0
    safe_capacity_utilization_weight: 0.0
    over_conservative_curtailment_weight: 0.0
    envelope_width_penalty_weight: 0.0

    # 安全优先，但默认不过度惩罚微小数值噪声
    safety_margin_weight: 1.0
    hard_violation_weight: 30.0
    powerflow_failure_weight: 80.0
    raw_action_safety_weight: 10.0
    projected_action_safety_weight: 5.0
    min_raw_unsafe_penalty: 0.1
    raw_safety_epsilon: 1.0e-5
    safety_gate_input_mode: max_raw_projected
    soft_safety_gate_kappa: 2.0

    # welfare normalization
    welfare_weight: 1.0
    welfare_clip: 5.0
    welfare_normalization_mode: per_mwh_baseline_zscore
    welfare_baseline_mean: 0.0
    welfare_baseline_std: 10.0
    welfare_clip_saturation_rate_max: 0.10

    # 工程运行项
    loss_cost_weight: 0.2
    smoothness_weight: 0.02

  vpp:
    dispatch:
      # 真实结算主线
      private_profit_weight: 1.0
      use_unified_private_profit_v3: true
      require_per_der_settlement_audit: true
      paper_long_fail_on_incomplete_settlement_audit: true
      settlement_power_balance_tolerance_mw: 1.0e-6

      # 主实验默认关闭 proxy 市场项
      service_payment_weight: 0.0
      availability_payment_weight: 0.0
      contract_delivery_weight: 0.0
      service_payment_source: disabled
      availability_payment_source: disabled
      contract_settlement_source: disabled

      # 投影与设备约束
      projection_linear_weight: 2.0
      projection_quadratic_weight: 5.0
      dispatch_constraint_weight: 1.0
      battery_degradation_weight: 0.01
      unserved_ev_energy_weight: 1.0

      # 储能跨时价值
      storage_potential_shaping_weight: 0.02
      storage_future_value_mode: price_forecast_window
      storage_future_value_window_steps: 16
      storage_charge_efficiency: 0.95
      storage_discharge_efficiency: 0.95
      storage_discount: 0.99
      storage_anti_hoarding_test_required: true

      # 默认不启用 terminal SOC target
      storage_terminal_value_weight: 0.0
      storage_terminal_soc_reference_mode: disabled
      storage_terminal_soc_reference: null
      storage_terminal_soc_reference_weight: 0.0
      storage_terminal_potential_residual_mode: log_and_ablate

  shield:
    enabled: true
    dso_penalty_coef: 1.0
    dispatch_penalty_coef: 1.0
    portfolio_future_penalty_coef: 1.0
    export_raw_projected_no_shield_diagnostics: true
```

---

# 13. 实施阶段重构

## Phase A：会计合同与安全诊断闭环

目标：先修正 reward 口径，不训练 paper-long。

任务：

```text
A0. Freeze Reward V3.1 accounting contract
A1. Reward V3.1 contract tests
A2. Implement v3 DSO reward without curtailment training terms
A3. Welfare normalization and calibration
A4. VPP settlement without double counting
A5. DER-level settlement audit and power balance
A6. EVCS audit revenue and wholesale cost separation
A7. Storage net potential shaping and anti-hoarding test
A8. Raw/projected/post/no-shield safety trace
A9. Dispatch settlement before DSO reward ordering
```

验收：

```text
1. v3 reward output 不含 dso_curtailment_cost / safe_capacity training fields
2. service/availability/contract 主实验为 disabled
3. EVCS revenue 只来自 evcs_grid_p_mw 与 retail_evcs_tariff
4. storage_terminal_value_reward 非终端步为 0
5. SOC target 默认 disabled
6. settlement_audit_complete = 1
7. settlement_power_balance_ok = 1
8. raw unsafe/projected safe 仍降低 DSO reward
9. projected unsafe 会关闭 safety gate
10. welfare_clip_saturation_rate < 10%
11. no_shield_eval metrics 可导出
```

## Phase B：真实数据与场景压力

任务：

```text
B1. ACN-Data adapter smoke
B2. TariffAdapter / OpenEI-URDB smoke
B3. SMART-DS / benchmark feeder scenario alignment
B4. balanced generation + absorption scenario
B5. pre-control AC stress scan
B6. EVCS deadline pressure profile
B7. storage arbitrage stress windows
```

验收：

```text
1. ACN session 能转换为 EVCS episode
2. tariff 能转换为 15-min price stream
3. 场景包含 reverse-flow、high-voltage、peak-import、EVCS deadline pressure
4. pre-control AC scan 证明场景有真实物理压力
```

## Phase C：MARL 稳定性与先进算法接口

任务：

```text
C1. HAPPO/HATRPO/MAPPO config loader
C2. KL early stop
C3. per-role advantage normalization
C4. value normalization
C5. reward component normalization
C6. role-specific gradient norm logging
C7. optional graph encoder for DSO critic
C8. optional transformer encoder for EVCS/VPP dispatch
C9. optional constrained MARL cost critic interface
```

验收：

```text
1. role_approx_kl exported
2. role_grad_norm_before_clip / after_clip exported
3. role_advantage_mean/std exported
4. policy entropy 不塌缩
5. KL early stop 可触发
6. 多 seed 无 NaN
```

## Phase D：市场机制升级，可选

仅在完成真实 bid-award-delivery-settlement 后启用：

```text
D1. VPP flexibility bid
D2. DSO award / clearing
D3. baseline rule
D4. delivery measurement
D5. non-delivery penalty
D6. DSO budget objective
D7. cleared_service_payment
D8. capacity_contract availability_payment
```

在 Phase D 前，论文不得声称完整灵活性市场结算。

## Phase E：paper-long 训练与消融

必须先完成：

```text
1. reward-v3.1 unit tests
2. simulator 8-step smoke
3. simulator 96-step sanity
4. HAPPO/HATRPO 1-episode smoke
5. 96-step MARL sanity
6. short ablations
7. no-shield evaluation
8. multiple seeds
9. standardized MARL reporting
```

---

# 14. 必做消融实验

| 消融 | 目的 |
|---|---|
| `v2_baseline` | 比较旧 reward-v2 与 reward-v3.1 |
| `v3_no_evcs_revenue` | 验证 EVCS 用户收入的贡献 |
| `v3_no_storage_temporal_value` | 验证储能跨时价值是否必要 |
| `v3_potential_only` | 检查只用 potential shaping 的效果 |
| `v3_terminal_only` | 检查只用 terminal value 的效果 |
| `v3_potential_plus_terminal` | 检查是否重复激励囤电 |
| `v3_no_soc_reference` | 主实验默认项，证明不依赖人为 SOC target |
| `v3_cyclic_soc_reference` | 仅用于 cyclic episode |
| `v3_service_disabled` | 主实验默认项 |
| `v3_proxy_service_payment` | proxy 服务费对照 |
| `v3_no_availability_payment` | 验证不依赖 availability proxy |
| `v3_proxy_availability_payment` | proxy 可用性费对照 |
| `v3_raw_safety_floor_0_0.05_0.1_0.5_1.0` | 扫描 raw unsafe floor |
| `v3_projected_safety_gate_off` | 验证 projected unsafe gate 必要性 |
| `v3_no_shield_eval` | 测试无 safety shield 下策略安全性 |
| `v3_profit_only_dispatch` | 验证 VPP 经济闭环 |
| `v3_safety_only_dso` | 验证 DSO 是否能只学安全包络 |
| `v3_welfare_calibration_off` | 证明 welfare calibration 必要性 |
| `v3_no_generation_mix` | 证明发电/吸收平衡场景必要性 |
| `v3_gnn_critic` | 检查图结构 critic 是否改善拓扑泛化 |
| `v3_transformer_evcs` | 检查 EVCS/价格时序编码是否改善调度 |
| `v3_lagrangian_cost_critic` | CMDP/Lagrangian 扩展对比 |

---

# 15. 评价指标体系

## 15.1 经济指标

```text
vpp_true_private_profit
vpp_operational_surplus_ex_transfer
energy_sell_revenue
energy_buy_cost
evcs_user_charging_revenue
evcs_wholesale_energy_cost
storage_discharge_revenue
storage_charge_energy_cost
battery_degradation_cost
unserved_ev_energy_cost
dso_transfer_payment_cost
```

## 15.2 安全指标

```text
raw_action_voltage_violation_cost
raw_action_line_overload_cost
raw_action_trafo_overload_cost
raw_action_powerflow_failed

projected_action_voltage_violation_cost
projected_action_line_overload_cost
projected_action_trafo_overload_cost
projected_action_powerflow_failed

post_ac_voltage_violation_cost
post_ac_line_overload_cost
post_ac_trafo_overload_cost
post_ac_powerflow_failed

no_shield_eval_voltage_violation_rate
no_shield_eval_line_overload_rate
no_shield_eval_trafo_overload_rate
no_shield_eval_powerflow_failure_rate
```

## 15.3 审计指标

```text
settlement_audit_complete
settlement_required_fields_missing_count
settlement_missing_<field_name>
settlement_power_balance_gap_mw
settlement_power_balance_ok
```

## 15.4 储能指标

```text
storage_soc_before
storage_soc_after
storage_soc_min_violation
storage_soc_max_violation
storage_potential_shaping_reward
storage_terminal_value_reward
storage_terminal_value_proxy
storage_terminal_potential_residual
storage_anti_hoarding_pass
storage_cycle_count
storage_degradation_cost
```

## 15.5 MARL 稳定性指标

```text
role_approx_kl
role_kl_early_stop
role_entropy
role_clip_fraction
role_grad_norm_before_clip
role_grad_norm_after_clip
role_advantage_mean
role_advantage_std
role_value_loss
role_policy_loss
critic_explained_variance
reward_component_norm_stats
```

---

# 16. 论文中可使用的贡献表述

建议写成：

```text
本文提出一种面向 VPP-DSO 协同调度的 safety-first、settlement-aware 多智能体强化学习 reward 框架。该框架首先在 VPP 侧建立 DER-level settlement audit，显式区分 PV、储能、EVCS、HVAC 和柔性负荷的收入与成本，避免 aggregate net power 导致的收入归因错误和 double counting。随后，DSO reward 仅使用剔除 service/availability transfer payments 的 operational surplus，并通过 per-MWh z-score calibration 和 raw/projected safety gate 控制经济福利信号。为避免 safety shield 掩盖策略缺陷，框架同时输出 raw-action、projected-action、post-AC 和 no-shield safety diagnostics。针对储能，本文采用净未来价值驱动的 step-level potential shaping，并默认关闭无合同依据的 terminal SOC target，通过 anti-hoarding 测试验证储能不会为了 shaping reward 无意义囤电。
```

不要写：

```text
guaranteed safe MARL
complete flexibility market clearing
real EVCS and tariff settlement fully implemented
```

除非后续 Phase C/D/E 实际完成并通过验证。

---

# 17. Paper-long 启动门槛

paper-long 训练只能在以下条件全部满足后开始：

```text
1. reward-v3.1 输出没有 DSO curtailment/safe-capacity training fields
2. service/availability/contract 主实验 disabled 或明确 proxy
3. VPP settlement 无 double counting
4. EVCS revenue 来自 EVCS DER audit
5. settlement_audit_complete = 1
6. settlement_power_balance_ok = 1
7. storage_terminal_value_reward 非终端步为 0
8. SOC reference 默认 disabled
9. anti-hoarding test 通过
10. raw unsafe/projected safe 仍降低 DSO reward
11. projected unsafe 会降低 safety gate
12. welfare_clip_saturation_rate < 10%
13. no-shield evaluation 可用
14. role KL、entropy、gradient norm、advantage stats 可导出
15. balanced generation/absorption 场景通过 profile diagnostics
16. balanced scenario 通过 pre-control AC stress scan
17. 至少 5 个 random seeds，无 NaN
18. 至少包含 MAPPO、HAPPO/HATRPO、rule-based/MPC/OPF baseline
19. 关键指标报告均值、标准差、置信区间
20. 论文表述不夸大为完整市场或严格安全保证
```

---

# 18. 最终结论

整合后的 Reward V3.1 应该从原来的 “market-safety implementation plan” 升级为：

> **面向 VPP-DSO 协同调度的安全优先、结算可审计、数据可追溯、CMDP/Lagrangian-ready 的多智能体强化学习框架。**

最重要的修订是：

1. 保留原计划的 settlement-before-DSO、DER audit、safety gate、raw/projected/no-shield diagnostics、HAPPO/HATRPO stability instrumentation。
2. 采纳反馈报告的机制纠偏：SOC target 默认关闭，service/availability/contract 默认关闭或 proxy 标注，`min_raw_unsafe_penalty` 默认降为 0.1，storage potential 改为净未来价值，必须 anti-hoarding。
3. 增加与先进 MARL/VPP 研究对齐的接口：constrained Markov game、Lagrangian cost critic、GNN topology encoder、Transformer temporal encoder、partial observability、真实数据 smoke test 和标准化多 seed 评估。
4. 论文主实验只声称 settlement-aware proxy reward 和 safety-first auditable MARL，不声称完整市场清算或 guaranteed safe MARL。

建议将这版作为新的正式计划，命名为：

```text
Reward V3.1 Market-Safety-CMDP-Ready Implementation Plan
```

主实验默认目标应收敛为：

```text
DSO:
  calibrated welfare
  raw/projected/post/no-shield safety audit
  safety-first envelope learning
  no curtailment training terms

VPP dispatch:
  DER-level true energy settlement
  EVCS user revenue
  storage net temporal value
  equipment and service quality cost
  no default proxy service/availability/contract revenue

MARL:
  CTDE + HAPPO/HATRPO/MAPPO baselines
  KL/gradient/value/advantage diagnostics
  optional GNN/Transformer encoder
  future Lagrangian constrained MARL extension
```

---

# 附录 A：核心字段清单

```text
reward_version_code
reward_scaling_version

# DSO welfare
vpp_operational_surplus_ex_transfer
dso_vpp_welfare_raw
dso_vpp_welfare_per_mwh
dso_vpp_welfare_zscore
dso_vpp_welfare_bounded
dso_safety_gate
dso_reward_train

# VPP settlement
vpp_true_private_profit
energy_sell_revenue
energy_buy_cost
evcs_user_charging_revenue
evcs_wholesale_energy_cost
storage_discharge_revenue
storage_charge_energy_cost
battery_degradation_cost
comfort_cost
unserved_energy_cost

# Settlement audit
settlement_audit_complete
settlement_required_fields_missing_count
settlement_missing_<field_name>
settlement_power_balance_gap_mw
settlement_power_balance_ok

# Safety diagnostics
raw_action_voltage_violation_cost
raw_action_line_overload_cost
raw_action_trafo_overload_cost
raw_action_powerflow_failed
projected_action_voltage_violation_cost
projected_action_line_overload_cost
projected_action_trafo_overload_cost
projected_action_powerflow_failed
post_ac_voltage_violation_cost
post_ac_line_overload_cost
post_ac_trafo_overload_cost
post_ac_powerflow_failed
no_shield_eval_violation_rate

# Storage
storage_soc_before
storage_soc_after
storage_potential_shaping_reward
storage_terminal_value_reward
storage_terminal_value_proxy
storage_terminal_potential_residual
storage_anti_hoarding_pass

# Training stability
role_approx_kl
role_kl_early_stop
role_entropy
role_grad_norm_before_clip
role_grad_norm_after_clip
role_advantage_mean
role_advantage_std
role_value_loss
role_policy_loss
```

---

# 附录 B：最小验证实验清单

```text
1. Reward V3.1 unit tests
2. DSO reward no-curtailment smoke
3. Welfare calibration smoke
4. VPP settlement no-double-counting test
5. EVCS audit revenue test
6. Storage anti-hoarding test
7. Storage terminal-only reward test
8. Settlement power balance test
9. Raw/projected safety gate test
10. No-shield evaluation smoke
11. Balanced generation scenario stress test
12. AC pre-control stress scan
13. HAPPO/HATRPO KL early stop test
14. 8-step simulator smoke
15. 96-step MARL sanity
16. Short ablation sanity
17. Multi-seed paper-long readiness check
```
