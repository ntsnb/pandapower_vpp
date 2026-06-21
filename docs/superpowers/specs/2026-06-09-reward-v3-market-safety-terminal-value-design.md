# Reward V3 市场安全与储能终值整改设计

本文档是 reward-v3 的设计规格，不是代码实现记录。它的目的，是把当前项目从“DSO curtailment proxy 主导 + VPP 缺少真实结算 + 储能充电无即时信用”的 reward 结构，改成“配电网安全优先、VPP 真实利润驱动、储能跨时价值可学习、场景中包含吸收型与发电型 VPP”的研究级设计。

## 1. 本次整改的硬性目标

1. DSO reward 中去掉所有 curtailment 类训练项。
   - 去掉 `dso_curtailment_cost` 对训练 reward 的扣分。
   - 去掉 `dso_over_conservative_curtailment_penalty` 作为训练 penalty 的含义。
   - 去掉 `dso_safe_capacity_utilization_reward` 作为训练 reward 的含义。
   - 可以保留“包络宽度、平均包络利用率”等中性审计字段，但字段名和用途必须明确是诊断，不进入 reward。
2. DSO 的首要目标是配电网安全。
   - 电压越限、线路过载、变压器过载、潮流不收敛是最高优先级。
   - 在安全状况下，再最大化所有 VPP 的真实总收益或社会福利。
3. EVCS 要加入真实用户充电售电收入。
   - EV 充电需求使用真实或公开数据集生成，例如 ACN-Data。
   - 如果使用真实电价，零售电价应来自明确来源，例如 OpenEI/URDB，批发电价和零售电价要分开。
4. 当前所有 VPP 平均可行域中点偏负，要增加发电型 VPP。
   - 新场景不能再全是净吸收型 VPP。
   - 至少要包含 PV park、PV+BESS、CHP/微型燃机、BESS 套利、社区光储等发电/可注入型组合。
5. 储能缺少 terminal value，需要补上跨时价值。
   - 不能只靠即时电价收入，否则充电时刻天然是负收益。
   - 要用终端价值、势函数 shaping、价值函数 bootstrap 或它们的组合，让“低价充电、高价放电”的长期收益能被 actor/critic 学到。

## 2. 当前代码证据

| 问题 | 当前证据 | 结论 |
|---|---|---|
| DSO reward 仍含 curtailment 类项 | `src/vpp_dso_sim/learning/reward_config.py` 中 `curtailment_cost_weight=0.5`、`safe_capacity_utilization_weight=0.2`、`over_conservative_curtailment_weight=0.5` | 当前配置没有满足“全部去掉 curtailment penalty”的目标。 |
| DSO 训练 reward 直接使用 safe capacity 和 curtailment | `src/vpp_dso_sim/entities/dso.py` 中 `dso_reward_env = ... + dso_safe_capacity_utilization_reward - dso_curtailment_cost ...` | DSO actor 会被“释放包络宽度/保守程度”代理项牵引，而不是只根据安全和收益学习。 |
| `over_conservative_curtailment_weight` 配置存在但语义混乱 | `dso_over_conservative_curtailment_penalty` 返回的是未加权 `over_conservative`，`curtailment_cost = curtailment_cost_weight * over_conservative` 才实际扣分 | 文档和训练日志容易误读，且已经造成 reward 分项解释困难。 |
| EVCS 当前是合成会话 | `src/vpp_dso_sim/der/evcs.py` 在 `__post_init__` 中用固定到离站模板生成 EV | 不能声称 EVCS 需求来自真实数据。 |
| EVCS 没有独立用户售电收入 | `src/vpp_dso_sim/envs/reward_design.py` 中 `energy_market_revenue = price * delivered_p_mw * dt_hours`，v2 下 `private_profit_proxy = energy_market_revenue - der_operation_cost` | 当 EVCS 用电为负功率时，VPP 会主要看到购电成本，而不是用户缴费收入。 |
| 储能没有显式 terminal value | `src/vpp_dso_sim/der/storage.py` 只更新 SOC；reward trace 中没有 stored-energy value | 充电动作的未来放电收益只能通过很长 horizon 的 critic 间接传播，早期训练非常弱。 |
| 当前 VPP 组合整体偏净吸收 | `docs/reports/vpp_portfolio_composition_and_profit_audit_20260608.md` 中 7 个 VPP 的平均中点均为负 | 场景更像需求响应压力测试，不足以支撑“多类型 VPP 市场响应”论文叙事。 |

## 3. 外部依据

本文档只使用这些外部依据作为设计支撑：

- Constrained Policy Optimization 将安全问题建模为“reward + constraints”比单纯调 reward 更自然，尤其适合带物理安全约束的控制问题。来源：PMLR 的 CPO 论文页面说明了 constrained RL 用于带安全约束任务，并给出 near-constraint satisfaction 的算法目标：https://proceedings.mlr.press/v70/achiam17a
- HATRPO/HAPPO 适合异构多智能体。其论文强调不要求所有 agent 共享参数，也不要求 joint value function 可分解，并给出 trust-region 下的单调改进理论。来源：https://arxiv.org/abs/2109.11251
- MAPPO 是强基线，使用 centralized value function 支撑 cooperative MARL。来源：https://arxiv.org/abs/2103.01955
- Potential-based reward shaping 的经典理论来自 Ng、Harada、Russell 1999。它给出一种不改变最优策略的 shaping 形式：`F(s,s') = gamma * Phi(s') - Phi(s)`。来源：https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html
- ACN-Data 提供 EV 充电会话字段，例如 connection/disconnect、kWhDelivered、userInputs 中的 kWhRequested、requestedDeparture、paymentRequired 等，可用于 EVCS 真实需求建模。来源：https://ev.caltech.edu/dataset.html
- OpenEI/URDB 提供美国 utility rate 结构，包括 sector、TOU、demand charge 等字段；EIA FAQ 也说明 DOE/OpenEI 是 utility rate structure 的免费来源。来源：https://apps.openei.org/services/doc/rest/util_rates/ 与 https://www.eia.gov/tools/faqs/faq.php?id=20&t=6

## 4. 设计路线选择

### 4.1 方案 A：只删 curtailment，继续用单一加权 reward

形式：

```text
DSO reward = -安全惩罚 - 损耗成本 + VPP 总利润
```

优点是改动小，能快速跑实验。缺点是安全和收益仍靠权重调平，一旦收益尺度变大，DSO 可能重新学到“用收益覆盖安全风险”的行为。

### 4.2 方案 B：CMDP / Lagrangian 安全约束

形式：

```text
maximize    E[sum gamma^t welfare_t]
subject to  E[sum gamma^t voltage_violation_t] <= epsilon_v
            E[sum gamma^t line_overload_t] <= epsilon_l
            E[sum gamma^t trafo_overload_t] <= epsilon_t
```

优点是理论上更符合配电网运行：安全不是普通奖励项，而是约束。缺点是训练实现复杂，需要记录 cost return、更新拉格朗日乘子，并让 critic 分别估计 reward value 和 cost value。

### 4.3 方案 C：安全优先门控 reward + 势函数储能终值

形式：

```text
DSO reward = -大权重安全惩罚 + safe_gate * VPP 总利润 - 小权重平滑/损耗
VPP reward = 真实结算利润 + EVCS 用户收入 + 储能终值 shaping - 履约/舒适/SOC/投影惩罚
```

优点是最适合当前代码阶段：比纯 CMDP 更容易接入现有 HAPPO/HATRPO/MAPPO/MATD3 训练栈，又能显式表达“安全下才谈利润”。缺点是仍需要正确归一化安全项和利润项。

### 4.4 推荐路线

推荐采用“方案 C 作为 reward-v3 第一版，方案 B 作为 paper-long 后续增强”的组合。

原因：

1. 你现在最急的问题是 reward 方向错误，先修 reward 语义比直接上完整 CMDP 更稳。
2. 当前项目已有 HAPPO/HATRPO/HASAC/MATD3 家族，安全优先门控 reward 可以较快接入这些 actor-critic。
3. 若后续要写成更强论文，可以把 reward-v3 升级为 Lagrangian safe MARL：reward critic 学利润，cost critic 学电压/线路/变压器风险。

## 5. DSO Reward V3 数学定义

### 5.1 物理符号

对每个时刻 `t`：

- `V_b,t`：母线 `b` 的电压幅值，单位 p.u.
- `L_l,t`：线路 `l` 的 loading percent。
- `T_m,t`：变压器 `m` 的 loading percent。
- `P_loss,t`：网损。
- `F_t`：潮流是否成功，成功为 1，失败为 0。
- `Pi_i,t`：VPP `i` 在时刻 `t` 的真实私有利润。
- `a_DSO,t`：DSO actor 输出的包络动作。
- `E_i,t = [P_i,min,t, P_i,max,t]`：DSO 给 VPP `i` 的运行包络。

### 5.2 安全代价

电压越限代价：

```math
C^V_t =
\frac{1}{|B|}
\sum_{b \in B}
\left(
\left[\frac{V_{\min} - V_{b,t}}{\Delta V}\right]_+^2
+
\left[\frac{V_{b,t} - V_{\max}}{\Delta V}\right]_+^2
\right)
```

其中 `[x]_+ = max(0,x)`，`Delta V` 是归一化尺度，例如 0.01 p.u.。

线路过载代价：

```math
C^L_t =
\frac{1}{|Lines|}
\sum_{l}
\left[
\frac{L_{l,t} - L^{max}_{l}}{L^{max}_{l}}
\right]_+^2
```

变压器过载代价：

```math
C^T_t =
\frac{1}{|Trafos|}
\sum_{m}
\left[
\frac{T_{m,t} - T^{max}_{m}}{T^{max}_{m}}
\right]_+^2
```

潮流失败代价：

```math
C^{PF}_t =
\begin{cases}
0, & \text{pandapower 潮流收敛}\\
1, & \text{pandapower 潮流失败}
\end{cases}
```

总安全代价：

```math
C^{safe}_t =
w_V C^V_t
+ w_L C^L_t
+ w_T C^T_t
+ w_{PF} C^{PF}_t
```

### 5.3 安全门控

硬门控：

```math
G^{safe}_t =
\mathbf{1}\left[
C^V_t = 0,\,
C^L_t = 0,\,
C^T_t = 0,\,
C^{PF}_t = 0
\right]
```

软门控：

```math
G^{safe}_t =
\exp\left(-\kappa C^{safe}_t\right)
```

建议训练初期使用软门控，评估和论文指标使用硬门控。软门控的好处是 violation 接近 0 时仍有梯度，不会让 reward 完全断掉。

### 5.4 VPP 总收益

```math
W_t =
\sum_{i=1}^{N_{VPP}}
\Pi_{i,t}
```

为了避免利润尺度压过安全项，进入 DSO reward 前需要归一化：

```math
\widetilde{W}_t =
\operatorname{clip}
\left(
\frac{W_t - \mu_W}{\sigma_W + \epsilon},
-W_{clip},
W_{clip}
\right)
```

`mu_W` 和 `sigma_W` 可以来自随机策略/规则策略的 warmup 统计，也可以用在线 running mean/std。

### 5.5 DSO reward

Reward-v3 的 DSO reward 定义为：

```math
r^{DSO}_t =
- \lambda_{safe} C^{safe}_t
- \lambda_{loss} \widetilde{P}_{loss,t}
- \lambda_{smooth} \|E_t - E_{t-1}\|_2^2
+ \beta \, G^{safe}_t \, \widetilde{W}_t
```

这里有四个关键点：

1. 没有 `curtailment_cost`。
2. 没有 `over_conservative_curtailment_penalty`。
3. 没有 `safe_capacity_utilization_reward`。
4. VPP 总收益只在安全门控下进入 DSO reward。

如果使用硬门控，当配电网不安全时：

```math
G^{safe}_t = 0
```

于是 DSO 只会受到安全惩罚，不会因为 VPP 利润高而被鼓励冒险。

### 5.6 安全优先证明

假设归一化后的 VPP 总收益有界：

```math
|\widetilde{W}_t| \le W_{clip}
```

任意两个动作 `a` 和 `a'`，如果 `a` 安全，`a'` 不安全，且不安全动作最小安全代价为：

```math
C^{safe}(a') \ge c_{min} > 0
```

最坏情况下，不安全动作得到最大的利润项，安全动作得到最小的利润项，则两者 reward 差的上界为：

```math
r^{DSO}(a') - r^{DSO}(a)
\le
-\lambda_{safe} c_{min}
+ 2 \beta W_{clip}
+ \lambda_{loss} \Delta P_{loss}
+ \lambda_{smooth} \Delta S
```

只要选择：

```math
\lambda_{safe} c_{min}
>
2 \beta W_{clip}
+ \lambda_{loss} \Delta P_{loss}
+ \lambda_{smooth} \Delta S
```

就有：

```math
r^{DSO}(a') < r^{DSO}(a)
```

也就是说，不安全动作无论带来多高 VPP 利润，都不会优于安全动作。这就是“安全优先”的数学条件。

### 5.7 与安全外壳的关系

安全投影可以继续作为运行保护，但 reward 必须记录“投影前”和“投影后”两类安全代价：

```math
C^{safe,raw}_t = C^{safe}(a^{raw}_t)
```

```math
C^{safe,proj}_t = C^{safe}(a^{proj}_t)
```

DSO actor 的训练 reward 至少要惩罚 raw action 导致的安全风险：

```math
r^{DSO}_t
\leftarrow
r^{DSO}_t
- \lambda_{raw} C^{safe,raw}_t
```

否则安全外壳会把危险动作修好，actor 只看到修正后的安全结果，从而学不到配电网物理边界。

## 6. EVCS 真实售电收入设计

### 6.1 当前问题

当前 EVCS 在项目中是吸收型 DER：

```math
P^{EVCS}_{j,t} \le 0
```

当 EVCS 充电时：

```math
E^{grid}_{j,t}
=
[-P^{EVCS}_{j,t}]_+ \Delta t
```

当前 v2 里的市场收入大致是：

```math
R^{market}_{i,t}
=
price_t \cdot P^{delivered}_{i,t} \cdot \Delta t
```

如果 VPP 从电网吸收功率，`P_delivered < 0`，这个项会变成负数。它表示“按批发/市场口径购电的成本”，但没有对应的“EV 用户给充电站付钱”的收入。

### 6.2 EVCS 用户收入

对 EVCS `j`：

```math
E^{grid}_{j,t}
=
[-P^{EVCS}_{j,t}]_+ \Delta t
```

考虑充电效率：

```math
E^{user}_{j,t}
=
\eta^{EVCS}_{ch} E^{grid}_{j,t}
```

用户售电收入：

```math
R^{EVCS,user}_{j,t}
=
\tau^{retail}_{j,t}
E^{user}_{j,t}
+ R^{session}_{j,t}
```

其中：

- `tau_retail` 是零售充电价格，可以来自 OpenEI/URDB、站点 tariff CSV 或经过说明的本地价格。
- `R_session` 是会话服务费、停车费或固定充电服务费。如果真实数据没有这部分，就设为 0，并在论文中明确说明。
- ACN-Data 的 `paymentRequired` 可以用于判断某会话是否应计用户付费，但 ACN-Data 本身不等于零售电价数据源。

### 6.3 EVCS 购电成本

批发购电成本：

```math
C^{EVCS,buy}_{j,t}
=
\lambda^{buy}_t E^{grid}_{j,t}
```

如果考虑需量电费：

```math
C^{demand}_{j,m}
=
\kappa^{demand}_m
\max_t P^{EVCS,grid}_{j,t}
```

在 step-level reward 中，需量电费可以用增量形式近似：

```math
\Delta C^{demand}_{j,t}
=
\kappa^{demand}_m
\left[
P^{EVCS,grid}_{j,t}
-
P^{peak}_{j,t-1}
\right]_+
```

### 6.4 EVCS 利润

```math
\Pi^{EVCS}_{j,t}
=
R^{EVCS,user}_{j,t}
- C^{EVCS,buy}_{j,t}
- \Delta C^{demand}_{j,t}
- C^{EVCS,unmet}_{j,t}
- C^{EVCS,op}_{j,t}
```

未满足用户充电需求的惩罚：

```math
C^{EVCS,unmet}_{j,t}
=
\lambda_{unmet}
\sum_{e \in EV(j)}
\mathbf{1}[t = d_e]
\left[
E^{req}_{e}
-
E^{delivered}_{e}
\right]_+^2
```

这样 EVCS 充电不再天然是“亏钱动作”。如果用户零售电价高于购电成本和运维成本，充电会给 VPP 正收益；如果价格倒挂或需量电费过高，VPP 也会学会推迟/削减充电。

### 6.5 VPP 真实私有利润

对 VPP `i`：

```math
\Pi_{i,t}
=
R^{energy,sell}_{i,t}
+ R^{EVCS,user}_{i,t}
+ R^{flex}_{i,t}
+ R^{availability}_{i,t}
- C^{energy,buy}_{i,t}
- C^{DER,op}_{i,t}
- C^{battery,deg}_{i,t}
- C^{comfort}_{i,t}
- C^{unserved}_{i,t}
- C^{deviation}_{i,t}
```

其中：

- `R_energy_sell`：PV、储能、微型燃机向电网或市场出售电能的收入。
- `R_EVCS_user`：EV 用户充电售电收入。
- `R_flex`：响应 DSO 灵活性服务请求的服务收入。
- `R_availability`：可用容量或备用容量收入。
- `C_energy_buy`：从上级电网/市场购电成本。
- `C_DER_op`：燃机、储能、柔性负荷等运行成本。
- `C_battery_deg`：储能循环退化成本。
- `C_comfort`：HVAC/柔性负荷舒适度损失。
- `C_unserved`：EV 未满足、负荷未满足等违约成本。
- `C_deviation`：未按包络/合同交付的偏差成本。

VPP dispatch reward 应该以这个 `Pi_i,t` 为主，而不是只使用 `energy_market_revenue - der_operation_cost`。

## 7. 发电型 VPP 场景设计

### 7.1 当前场景问题

当前报告显示 7 个 VPP 的平均可行域中点均为负：

```text
vpp_commercial_multi  -0.003 MW
vpp_community_multi   -0.012 MW
vpp_f3_mixed_multi    -0.032 MW
vpp_residential_multi -0.006 MW
vpp_single_campus     -0.027 MW
vpp_single_ev_hub     -0.041 MW
vpp_single_industrial -0.038 MW
```

这说明当前场景更像“需求响应/充电/柔性负荷压力测试”，不是均衡的“多类型 VPP 市场协同”场景。

### 7.2 新增 VPP 类型

reward-v3 场景应新增至少这些 VPP：

| VPP 类型 | DER 组成 | 物理意义 | 预期平均中点 |
|---|---|---|---|
| Solar Park VPP | 大容量 PV + 小储能 | 典型光伏发电聚合商 | 正 |
| PV+BESS Arbitrage VPP | PV + 中/大储能 | 白天发电，低价充电，高价放电 | 接近 0 或正 |
| CHP / Microturbine Industrial VPP | CHP/微型燃机 + 工业负荷 + 储能 | 可调发电与热电联产 | 正或接近 0 |
| Community Export VPP | 多节点屋顶 PV + 社区储能 | 居民侧反向潮流场景 | 白天正，夜间负 |
| BESS Merchant VPP | 独立储能 | 套利和安全服务 | 中点 0，但高可调跨度 |
| Wind/PV Hybrid VPP | 风/光 + 储能 | 可再生发电不确定性 | 正但波动 |

### 7.3 场景验收指标

新 benchmark 场景应满足：

```math
\frac{
\#\{i: \bar{m}_i > 0\}
}{
N_{VPP}
}
\ge 0.4
```

其中 `m_i` 是 VPP `i` 的平均可行域中点：

```math
\bar{m}_i
=
\frac{1}{T}
\sum_t
\frac{P^{max}_{i,t} + P^{min}_{i,t}}{2}
```

同时：

```math
\sum_i \bar{P}^{inj}_{i}
>
0
```

至少要有一部分时间段出现净注入或反向潮流压力，否则无法检验 DSO 对发电型 VPP 的包络推荐能力。

### 7.4 论文场景分层

建议把场景分为四类：

1. `train_mixed_balanced`：吸收型和发电型 VPP 都有。
2. `holdout_peak_import`：峰值负荷和 EVCS 充电压力。
3. `holdout_cloudy_low_pv`：阴天低 PV，检验储能和燃机。
4. `holdout_reverseflow_generation`：高 PV/低负荷，检验反向潮流和电压越限。

## 8. 储能 Terminal Value 与多智能体强化学习处理

### 8.1 当前问题

储能充电在即时 reward 中通常是负的：

```math
\Delta r_t^{charge}
\approx
- \lambda^{buy}_t E^{grid}_t
- C^{deg}_t
```

但它可能在未来产生收益：

```math
\Delta r_{\tau}^{discharge}
\approx
\lambda^{sell}_{\tau} E^{sell}_{\tau}
```

如果 horizon 很长，例如 672 steps，且 critic 尚未学好，充电动作的收益要经过很长的 temporal credit assignment 才能反馈到 actor。这会导致早期训练中“充电总是看起来亏”。

### 8.2 现代 MARL 的处理方式

在 HAPPO/HATRPO/MAPPO 这类 actor-critic 中，常见处理不是让 actor 直接等到未来几百步，而是让 critic 估计未来价值：

```math
Q^{\pi}(s_t, a_t)
=
\mathbb{E}_{\pi}
\left[
\sum_{k=0}^{\infty}
\gamma^k r_{t+k}
\middle|
s_t,a_t
\right]
```

如果 critic 的输入包含：

- 储能 SOC；
- 当前时间；
- 电价预测；
- PV/负荷预测；
- DSO 安全裕度；
- VPP 包络上下限；

那么 critic 可以学到“现在充电，未来高价放电”的长期价值。问题是：如果 reward 里根本没有未来储能价值，或者 episode 末端截断时没有 terminal value，critic 也很难凭空学出来。

### 8.3 方案一：显式 terminal value

在 episode 末端加入：

```math
R_T^{storage}
=
\sum_{j \in Storage}
\kappa_{j,T}
E_{j,T}^{stored}
```

其中：

```math
E_{j,T}^{stored}
=
SOC_{j,T} \cdot E^{cap}_{j}
```

`kappa` 是末端储能电量的边际价值。最简单选择：

```math
\kappa_{j,T}
=
\eta^{dis}_{j}
\mathbb{E}
\left[
\lambda^{sell}_{T:T+H}
\right]
```

更稳妥的价格套利值：

```math
\kappa_{j,t}
=
\max_{\tau \in [t,t+H]}
\gamma^{\tau-t}
\eta^{dis}_{j}
\lambda^{sell}_{\tau}
```

为了防止储能永远囤电，还要加入终端 SOC 目标或衰减：

```math
C_T^{soc-target}
=
\lambda_{socT}
\left(SOC_{j,T} - SOC^{target}_{j}\right)^2
```

于是末端项为：

```math
R_T^{storage-final}
=
\sum_j
\kappa_{j,T} E_{j,T}^{stored}
-
\lambda_{socT}
\left(SOC_{j,T} - SOC^{target}_{j}\right)^2
```

### 8.4 方案二：势函数 shaping

定义势函数：

```math
\Phi(s_t)
=
\sum_j
\kappa_{j,t}
E_{j,t}^{stored}
```

对每一步 reward 加：

```math
F_t
=
\gamma \Phi(s_{t+1})
-
\Phi(s_t)
```

新的 reward：

```math
r'_t
=
r_t
+ F_t
```

这会让储能充电后的 SOC 增加在下一步立即体现为正反馈。

### 8.5 势函数 shaping 的 telescoping 证明

考虑从 `t=0` 到 `T-1` 的折扣回报：

```math
\sum_{t=0}^{T-1}
\gamma^t r'_t
=
\sum_{t=0}^{T-1}
\gamma^t r_t
+
\sum_{t=0}^{T-1}
\gamma^t
\left(
\gamma \Phi(s_{t+1}) - \Phi(s_t)
\right)
```

第二项展开：

```math
\sum_{t=0}^{T-1}
\gamma^t
\left(
\gamma \Phi(s_{t+1}) - \Phi(s_t)
\right)
=
\sum_{t=0}^{T-1}
\gamma^{t+1}\Phi(s_{t+1})
-
\sum_{t=0}^{T-1}
\gamma^t \Phi(s_t)
```

中间项两两抵消，得到：

```math
=
\gamma^T \Phi(s_T)
-
\Phi(s_0)
```

如果初始状态 `s0` 固定，且 terminal potential 被设置为 0 或作为同一个终端边界处理，那么 shaping 只改变学习过程中的信用分配，不改变最优策略集合。这就是 potential-based shaping 的核心性质。

如果我们故意保留 `gamma^T Phi(s_T)` 作为储能资产价值，那么它会改变原目标；但这是合理的，因为当前目标本来漏掉了 episode 截断后储能电量的经济价值。

### 8.6 充电即时正反馈条件

假设储能在 `t` 时刻从电网充入 `x` MWh：

```math
E^{stored}_{t+1}
-
E^{stored}_{t}
=
\eta^{ch} x
```

即时购电成本：

```math
C^{buy}_t
=
\lambda^{buy}_t x
```

势函数给出的边际 shaping 增益近似为：

```math
\Delta F_t
\approx
\gamma \kappa_{t+1}
\eta^{ch} x
-
\kappa_t \cdot 0
```

充电动作的净即时反馈：

```math
\Delta r'_t
\approx
-\lambda^{buy}_t x
- C^{deg}_t
+
\gamma \kappa_{t+1} \eta^{ch} x
```

当：

```math
\gamma \kappa_{t+1} \eta^{ch}
>
\lambda^{buy}_t
+
\frac{C^{deg}_t}{x}
```

低价充电就会得到正反馈。这个条件正是储能套利的经济逻辑：未来可卖电价值折现后要高于当前买电成本和退化成本。

### 8.7 防止错误激励

储能 terminal value 不能无脑给高，否则 agent 会学到“永远充满不放”。需要三层约束：

1. `kappa` 必须来自未来电价/服务价格，不是常数大奖励。
2. 末端 SOC 要有目标区间或机会成本。
3. 放电收益、退化成本、效率损失都要进入真实利润。

推荐储能 reward 组合：

```math
r^{storage}_{j,t}
=
R^{discharge}_{j,t}
- C^{charge}_{j,t}
- C^{degradation}_{j,t}
- C^{soc-violation}_{j,t}
+ \alpha_F
\left(
\gamma \Phi_j(s_{t+1})
- \Phi_j(s_t)
\right)
```

其中 `alpha_F` 初期可以为 1，paper-long 前应做消融：

- 无 terminal value；
- 只有 terminal value；
- 只有 potential shaping；
- terminal value + potential shaping；
- learned terminal critic bootstrap。

## 9. 与当前算法家族的对接

### 9.1 推荐算法排序

| 算法 | 适用性 | 原因 |
|---|---|---|
| HAPPO/HATRPO | 高 | DSO、VPP dispatch、VPP portfolio 是异构 agent，不应强制共享参数；trust-region 对 reward 改造后的稳定训练有帮助。 |
| MAPPO | 高 | 适合作为 centralized critic 强基线，能吃全局安全状态和联合动作摘要。 |
| HASAC/MATD3 | 中到高 | 适合连续动作和探索，但要谨慎处理 safety projection 造成的 off-policy 分布偏移。 |
| IPPO | 中 | 可作为低复杂度 baseline，但不适合解释 DSO-VPP 强耦合安全约束。 |
| 规则/优化 baseline | 必须保留 | 用于证明环境和 reward 本身合理，否则 RL 收敛失败无法归因。 |

### 9.2 Critic 输入要求

DSO critic 至少需要：

- 全网电压最小/最大值；
- 线路 loading top-k；
- 变压器 loading；
- post-AC violation；
- raw action 安全代价；
- projection 后安全代价；
- VPP 包络上下限；
- VPP bid / price / capacity 摘要；
- 全部 VPP 的真实利润摘要；
- 时间编码和价格/PV/负荷预测窗口。

VPP dispatch critic 至少需要：

- 本 VPP 的 DER token；
- SOC、EVCS 连接数量、EV deadline pressure；
- PV 可用功率；
- DSO 下发包络；
- 本地价格、零售售电价、批发购电价；
- 投影前/后的动作 gap；
- 本 VPP 的真实结算利润分项。

执行期 actor 仍要遵守隐私边界：DSO actor 不读取 VPP 私有真实成本、用户舒适偏好、私有 SOC 明细；集中式 critic 可以在训练期读取更完整状态，但要在论文中明确 CTDE 假设。

## 10. 后续代码修改清单

| 文件 | 修改内容 | 理由 |
|---|---|---|
| `src/vpp_dso_sim/learning/reward_config.py` | 新增 `version: v3_market_safety`；DSO config 移除或禁用 curtailment/safe-capacity reward 字段；新增 safety/welfare normalization 字段 | 建立 reward-v3 配置入口。 |
| `src/vpp_dso_sim/entities/dso.py` | 新增 `_dso_v3_components`；训练 reward 只使用安全、损耗、平滑、safe-gated VPP welfare | 彻底移除 DSO curtailment proxy。 |
| `src/vpp_dso_sim/envs/reward_design.py` | 把 `private_profit_proxy` 改为 true settlement；加入 EVCS user revenue、buy cost、demand charge、storage terminal value | 让 VPP reward 反映真实利润。 |
| `src/vpp_dso_sim/der/evcs.py` | 支持从真实 session 构建 EV；记录 session/user tariff/paymentRequired/requested energy | 接入 ACN-Data 等真实 EVCS 数据。 |
| `src/vpp_dso_sim/data_sources/acn.py` | 新增 ACN-Data 本地 CSV/API 适配器 | 用真实充电会话替换合成模板。 |
| `src/vpp_dso_sim/data_sources/tariff.py` | 新增 OpenEI/本地 tariff 适配器 | 区分零售电价、批发电价、需量电费。 |
| `configs/scenarios/benchmark/european_lv_benchmark_v3_balanced_generation.yaml` | 新增发电型 VPP 和均衡场景 | 修复所有 VPP 偏净吸收的问题。 |
| `tests/test_reward_v3_market_safety.py` | 增加 DSO 无 curtailment、EVCS 收入、储能 terminal value、安全优先门控测试 | 防止 reward 语义回退。 |
| `tests/test_scenario_generation_mix.py` | 验证发电型 VPP 占比和正中点比例 | 保证场景不再全偏吸收。 |

## 11. 必须先通过的验证实验

### 11.1 Reward contract tests

1. DSO reward-v3 中搜索不到进入训练 reward 的 `curtailment_cost`。
2. DSO reward-v3 中搜索不到进入训练 reward 的 `safe_capacity_utilization_reward`。
3. DSO reward-v3 中搜索不到进入训练 reward 的 `over_conservative_curtailment_penalty`。
4. 电压越限、线路过载、变压器过载、潮流失败会让 DSO reward 显著下降。
5. 安全状态下，VPP 总利润增加会让 DSO reward 增加。

### 11.2 EVCS settlement tests

构造一个 EVCS：

```text
充电功率 = 0.05 MW
dt = 0.25 h
零售售电价 = 1000 元/MWh
批发购电价 = 400 元/MWh
效率 = 0.95
```

则：

```math
E^{grid} = 0.05 \times 0.25 = 0.0125 \text{ MWh}
```

```math
E^{user} = 0.95 \times 0.0125 = 0.011875 \text{ MWh}
```

```math
R^{user} = 1000 \times 0.011875 = 11.875
```

```math
C^{buy} = 400 \times 0.0125 = 5.0
```

如果忽略运维和需量电费：

```math
\Pi^{EVCS} = 11.875 - 5.0 = 6.875
```

这个测试必须证明 EVCS 充电可以产生正利润，而不是必然负利润。

### 11.3 Storage terminal value tests

构造一个储能：

```text
低价买电 = 300 元/MWh
未来高价卖电 = 900 元/MWh
eta_charge = 0.95
eta_discharge = 0.95
gamma = 0.99
```

未来边际价值近似：

```math
\kappa \approx 900 \times 0.95 = 855
```

充电正反馈条件：

```math
0.99 \times 855 \times 0.95 > 300
```

左侧为：

```math
804.6225 > 300
```

所以低价充电应获得正的 shaping 信号。

### 11.4 场景 mix tests

新场景必须满足：

```text
positive_midpoint_vpp_ratio >= 0.4
generation_capable_vpp_count >= 4
storage_capable_vpp_count >= 4
reverseflow_candidate_steps > 0
```

### 11.5 最小训练实验

1. 8-step smoke：验证 reward 字段完整。
2. 96-step single-VPP storage arbitrage：验证储能会低价充、高价放。
3. 96-step EVCS settlement：验证 EVCS 不再因为充电必然负利润。
4. 288-step balanced generation：验证 DSO 在安全下放大利润，在不安全时收紧。
5. paper-long：只有上述实验通过后再跑。

## 12. 论文表达建议

当前项目的研究主线可以表述为：

```text
一个面向配电网安全约束的 DSO-VPP 双向引导多智能体强化学习框架。
DSO 通过 AC 潮流感知的运行包络引导 VPP；
VPP 在真实结算收益、EVCS 用户收入、储能跨时价值和 DER 约束下响应；
训练期使用集中式 critic 学习全局安全-经济权衡，执行期保持 DSO/VPP 信息边界。
```

reward-v3 后，论文创新点会更清晰：

1. DSO 不再因为“包络宽度”这种 proxy 受激励，而是直接学习安全和社会福利。
2. VPP 不再是抽象利润代理，而是含 EVCS 用户售电、购电成本、服务收入、储能退化和终值的真实结算主体。
3. 储能通过 terminal value / potential shaping 解决长 horizon credit assignment。
4. 场景包含吸收型和发电型 VPP，可检验峰值负荷、低 PV、高 PV 反向潮流等多种配电网真实压力。
5. 安全投影不再替代学习，而是作为运行保护；actor 仍通过 raw-action safety cost 学物理边界。

## 13. 设计自审

| 检查项 | 结论 |
|---|---|
| 是否满足“去掉 DSO curtailment penalty” | 满足。DSO reward-v3 明确不含 `curtailment_cost`、`over_conservative_curtailment_penalty`、`safe_capacity_utilization_reward`。 |
| 是否保留 DSO 自主选择包络 | 满足。包络上下限由 DSO actor 输出，但 reward 不再直接奖励“越宽越好”或惩罚“越窄越坏”。 |
| 是否体现安全第一 | 满足。安全代价和安全门控决定利润项是否生效。 |
| 是否加入 EVCS 真实售电收入 | 满足。给出 ACN-Data + tariff 的数据路径和完整结算公式。 |
| 是否增加发电型 VPP | 满足。给出 VPP 类型、验收比例和场景分层。 |
| 是否解释储能 terminal value | 满足。给出 terminal reward、potential shaping、telescoping proof 和充电正反馈条件。 |
| 是否直接改代码 | 未改。本文件只定义后续实现方案。 |

## 14. 下一步执行入口

如果确认采用本设计，下一步应生成 implementation plan，按以下顺序实现：

1. reward-v3 contract tests。
2. DSO reward-v3，无 curtailment/safe-capacity 训练项。
3. EVCS settlement 数据结构和本地 CSV 适配。
4. tariff adapter 和真实/本地电价结算。
5. storage terminal value / potential shaping。
6. balanced generation scenario。
7. 8-step、96-step、288-step、paper-long 分级实验。
