# Reward 全术语逐项解释手册

更新日期：2026-06-04

本文档只解释当前代码仓实际存在的 reward、penalty、loss、训练指标和日志字段。任务草稿里提到但当前主路径尚未完整落地的设计，会在文末单独标为“设计项/未完全落地”，避免把设计设想和当前实验结果混在一起。

核心代码位置：

- `src/vpp_dso_sim/entities/dso.py`：DSO reward、成本缩放、AC 安全惩罚。
- `src/vpp_dso_sim/envs/reward_design.py`：VPP dispatch reward、VPP portfolio reward、role reward map。
- `src/vpp_dso_sim/learning/reward_contracts.py`：安全壳惩罚、VPP reward 常量。
- `src/vpp_dso_sim/learning/advanced_marl.py`：HAPPO/HASAC 的训练 reward、loss、日志字段。
- `src/vpp_dso_sim/learning/hatrpo.py`：HATRPO 的训练 reward、loss、日志字段。
- `src/vpp_dso_sim/network/constraints.py`：电压、线路、变压器、潮流失败违规惩罚。

## 1. 先说结论：reward 不是一个数，而是一条链

当前项目中，reward 的完整链路是：

$$
\text{Actor原始动作}
\rightarrow
\text{动作解码}
\rightarrow
\text{本地边界/FR/DOE投影}
\rightarrow
\text{AC-aware DOE投影}
\rightarrow
\text{AC certificate修复}
\rightarrow
\text{写入DER和pandapower}
\rightarrow
\text{AC潮流校验}
\rightarrow
\text{环境reward分项}
\rightarrow
\text{训练器扣安全壳惩罚}
\rightarrow
\text{乘reward scale}
\rightarrow
\text{critic/value/actor loss}
$$

所以你看到的 `reward_components.csv` 不是神经网络直接吃到的全部内容。它主要是环境结算和审计字段。训练器还会再扣安全壳惩罚，并乘 `reward_scale`。

## 2. 当前代码真实口径，不要和任务草稿混淆

| 容易误解的说法 | 当前代码真实情况 |
|---|---|
| `reward_scaling_version = 2.0` 就是任务草稿里的 `v2_minimal` | 不是。当前 `reward_scaling_version = 2.0` 表示 DSO reward 使用“加权、缩放、裁剪”的缩放口径。它仍然包含 tracking、effective response、comfort、SOC、envelope width、smoothness 等项。 |
| DSO reward 已经完全只剩电网安全和成本 | 不是。当前 DSO reward 仍然有 comfort、SOC、tracking、effective response。 |
| VPP reward 权重都从 YAML 读取 | 当前不是。VPP dispatch/portfolio 的主要权重来自 `reward_contracts.py` 常量。 |
| `projection_gap` 能代表电网安全 | 不能。`projection_gap` 只表示动作被修正了多少。AC 安全必须看 post-AC violation 字段。 |
| reward 和 loss 是一回事 | 不是。reward 是环境给智能体的反馈；loss 是神经网络根据 reward、advantage、Q target 等构造的反向传播目标。 |

## 3. 三类智能体分别吃什么 reward

| 角色 | agent id | 中文说明 | 使用的 actor | reward 归属 |
|---|---|---|---|---|
| DSO 全局引导智能体 | `dso_global_guidance` | 配电网运营方，给 VPP 下发目标、边界、引导强度 | DSO actor | DSO 环境 reward，训练时再扣安全壳惩罚 |
| VPP 快周期调度智能体 | `{vpp_id}_dispatch` | 每个 VPP 的实时调度，决定聚合功率/DER动作 | VPP dispatch actor | VPP dispatch reward，训练时再扣安全壳惩罚 |
| VPP 慢周期组合智能体 | `{vpp_id}_portfolio` | 每个 VPP 的资源组合/重配建议 | VPP portfolio actor | VPP portfolio reward，非决策步通常置 0 |

HAPPO/HATRPO 日志里的总 reward 是：

$$
\text{日志总reward}
=
\text{DSO训练reward}
+
\operatorname{mean}(\text{所有VPP dispatch训练reward})
+
\operatorname{mean}(\text{所有VPP portfolio训练reward})
$$

这只是日志总和，不代表所有 agent 共用同一个 critic。多角色 value/critic 会按角色奖励向量分别学习。

## 4. DSO 全局环境 reward

### 4.1 总公式

当前 DSO 环境 reward 是：

$$
\text{DSO环境reward}
=
\text{可行奖励}
+
\text{跟踪奖励}
+
\text{有效响应奖励}
-
0.05
\times
\text{缩放总成本}
$$

其中：

| 项 | 中文名 | 公式 | 范围 | 解释 |
|---|---|---|---|---|
| 可行奖励 | `feasibility_bonus` | AC 潮流后安全则 1，否则 0 | 0 或 1 | 最终执行动作没有电压/线路/变压器/潮流失败违规，就给奖励 |
| 跟踪奖励 | `tracking_bonus` | \(0.25/(1+\text{DSO目标跟踪误差})\) | 0 到 0.25 | VPP 越接近 DSO 目标，奖励越高 |
| 有效响应奖励 | `effective_response_bonus` | 当前实际为 \(1/(1+\text{DSO目标跟踪误差})\) | 0 到 1 | 当前没有外部有效响应得分输入，因此它是目标跟踪误差的另一个变换 |
| 缩放总成本 | `scaled_total_cost` | \(\sum_k \min(10,\max(0,w_k c_k)/s_k)\) | 非负 | 所有成本项归一化后求和 |
| 成本缩放系数 | `dso_reward_cost_scale` | 0.05 | 固定超参数 | 控制成本项对 DSO reward 的总体扣分强度 |

如果某一步 AC 安全、目标跟踪误差接近 0、缩放成本接近 0，则：

$$
\text{DSO环境reward}
\approx
1 + 0.25 + 1
=
2.25
$$

这解释了为什么当前 sensitivity-v1 的 DSO reward 常在 2.1 到 2.25 附近。

### 4.2 什么是“原始成本”

原始成本是某个物理或经济指标刚被换算成惩罚后的值，还没有乘 reward 权重，也没有缩放。

通用流程：

$$
\text{原始成本}
\xrightarrow{\times \text{权重}}
\text{加权成本}
\xrightarrow{\div \text{缩放因子并裁剪}}
\text{缩放成本}
\xrightarrow{\sum}
\text{缩放总成本}
\xrightarrow{\times 0.05}
\text{DSO reward扣分}
$$

对任意成本项 \(k\)：

$$
\text{加权成本}_k
=
\text{原始成本}_k
\times
\text{权重}_k
$$

$$
\text{缩放成本}_k
=
\min\left(
10,\,
\frac{\max(0,\text{加权成本}_k)}{\text{缩放因子}_k}
\right)
$$

$$
\text{缩放总成本}
=
\sum_k \text{缩放成本}_k
$$

字段命名规则：

| 字段形式 | 含义 |
|---|---|
| `raw_xxx` | 原始成本，没有乘权重，没有缩放 |
| `xxx` | 加权成本，已经乘权重 |
| `scaled_xxx` | 缩放成本，已经除以缩放因子并裁剪 |
| `xxx_weight` | 该项权重 |
| `xxx_scale` | 该项缩放因子 |

例子：

如果舒适度原始成本是 4000，权重是 0.02，缩放因子是 100：

$$
\text{舒适度加权成本}=4000\times0.02=80
$$

$$
\text{舒适度缩放成本}=80/100=0.8
$$

$$
\text{对DSO reward直接扣分}=0.05\times0.8=0.04
$$

### 4.3 什么是“缩放因子”

缩放因子是归一化分母，不是权重，不是学习参数。

它用于把不同单位、不同量级的成本放到相近范围。否则 comfort、线路 loading、潮流失败、经济成本无法直接相加。

当前默认缩放因子：

| 成本项字段 | 中文名 | 缩放因子 | 为什么这样设 |
|---|---|---:|---|
| `operation_cost` | 运行成本 | 1000 | 经济代理成本量级可能几十到几百 |
| `target_tracking_error_penalty` | 目标跟踪误差惩罚 | 10 | 原始形式已有 \(100e^2\) 放大 |
| `action_projection_penalty` | 动作投影惩罚 | 10 | 原始形式含 \(250d^2+2n\) |
| `comfort_violation_penalty` | 舒适度惩罚 | 100 | HVAC comfort 原始值可能很大 |
| `soc_violation_penalty` | SOC惩罚 | 100 | 储能/EV SOC 缺口平方惩罚可能较大 |
| `security_violation_count_penalty` | 安全违规数量惩罚 | 1 | 本身是安全违规计数惩罚 |
| `post_ac_violation_magnitude_penalty` | AC违规幅值惩罚 | 1 | 后验违规幅值直接保留 |
| `voltage_violation_penalty` | 电压越限惩罚 | 1 | 电压偏差已乘 10000 |
| `line_overload_penalty` | 线路过载惩罚 | 100 | 百分比超限平方可能较大 |
| `transformer_overload_penalty` | 变压器过载惩罚 | 100 | 同线路 loading |
| `powerflow_penalty` | 潮流失败惩罚 | 1000 | 潮流失败原始固定为 1000 |
| `envelope_width_penalty` | 边界宽度惩罚 | 1 | 本身通常是 0 到 1 的比例 |
| `smoothness_penalty` | 边界平滑惩罚 | 1 | 本身是 MW 级变化量 |

## 5. 目标跟踪误差

目标跟踪误差不是神经网络 loss，也不是预测误差。它是“最终物理执行结果”和“DSO 目标”之间的偏差。

### 5.1 DSO 全局目标跟踪误差

每个 VPP 的误差：

$$
e_i
=
\left|
P_i^{\text{actual}}
-
P_i^{\text{target}}
\right|
$$

所有 VPP 汇总：

$$
E_{\text{DSO}}
=
\sum_i e_i
$$

单位是 MW。

它进入 DSO reward 的三条路径：

第一，作为惩罚：

$$
\text{目标跟踪误差原始惩罚}
=
100
\times
E_{\text{DSO}}^2
$$

第二，作为跟踪奖励：

$$
\text{跟踪奖励}
=
\frac{0.25}{1+E_{\text{DSO}}}
$$

第三，作为有效响应奖励 fallback：

$$
\text{有效响应奖励}
=
\frac{1}{1+E_{\text{DSO}}}
$$

数值例子：

| VPP | DSO目标功率 | 最终实际功率 | 误差 |
|---|---:|---:|---:|
| VPP 1 | 0.20 MW | 0.18 MW | 0.02 MW |
| VPP 2 | -0.10 MW | -0.16 MW | 0.06 MW |
| VPP 3 | 0.05 MW | 0.05 MW | 0.00 MW |

则：

$$
E_{\text{DSO}}=0.02+0.06+0.00=0.08\text{ MW}
$$

$$
\text{目标跟踪误差原始惩罚}=100\times0.08^2=0.64
$$

$$
\text{跟踪奖励}=0.25/1.08\approx0.231
$$

$$
\text{有效响应奖励}=1/1.08\approx0.926
$$

对应字段：

| 字段 | 中文解释 |
|---|---|
| `raw_target_tracking_error_penalty` | 未加权的 DSO 目标跟踪误差惩罚 |
| `target_tracking_error_penalty` | 加权后的 DSO 目标跟踪误差惩罚 |
| `scaled_target_tracking_error_penalty` | 缩放后的 DSO 目标跟踪误差惩罚 |
| `tracking_bonus` | DSO 跟踪奖励 |
| `effective_response_bonus` | DSO 有效响应奖励 |

### 5.2 VPP 局部目标跟踪偏差

VPP dispatch reward 里还有单个 VPP 自己的跟踪偏差：

$$
e_i^{\text{VPP}}
=
\left|
P_i^{\text{delivered}}
-
P_i^{\text{target}}
\right|
$$

进入 VPP dispatch reward：

$$
\text{VPP目标跟踪惩罚}
=
25
\times
\left(e_i^{\text{VPP}}\right)^2
$$

对应字段：

| 字段 | 中文解释 |
|---|---|
| `delivered_p_mw` | 本 VPP 当前实际聚合功率 |
| `target_p_mw` | 本 VPP 当前目标功率 |
| `target_tracking_penalty` | VPP dispatch 局部跟踪惩罚 |

DSO 跟踪误差和 VPP 跟踪误差的区别：

| 类型 | 看谁 | 用在哪里 |
|---|---|---|
| DSO 全局目标跟踪误差 | 所有 VPP 汇总 | DSO reward |
| VPP 局部目标跟踪偏差 | 单个 VPP | VPP dispatch reward、portfolio reliability |

## 6. 有效响应得分

“有效响应”在当前代码里有两个含义。

### 6.1 DSO reward 里的有效响应奖励

理论上 DSO reward 函数支持传入外部有效响应得分：

$$
\text{有效响应奖励}
=
\operatorname{clip}(\text{外部有效响应得分},0,1)
$$

但当前 simulator 没有传入这个外部得分，所以实际使用：

$$
\text{有效响应奖励}
=
\frac{1}{1+E_{\text{DSO}}}
$$

其中 \(E_{\text{DSO}}\) 是 DSO 全局目标跟踪误差。

这意味着当前代码中：

$$
\text{有效响应奖励不是独立学出来的指标，而是跟踪误差的单调函数}
$$

### 6.2 VPP 偏好区间里的响应有效性门控

VPP dispatch 的偏好区间奖励里有一个 `preferred_bonus_effectiveness_gate`。

如果 envelope 没有显式提供 `effective_response_score`，当前使用：

$$
\text{响应有效性门控}
=
\operatorname{clip}
\left(
1-
\frac{\text{本地投影距离}}{\text{硬约束区间宽度}},
0,
1
\right)
$$

解释：

| 情况 | 门控值 |
|---|---:|
| 动作没有被投影 | 接近 1 |
| 动作被小幅投影 | 0 到 1 |
| 动作被大幅投影 | 接近 0 |

## 7. 动作投影惩罚

动作投影惩罚回答的问题是：

$$
\text{actor给出的动作，离最终可执行动作有多远？}
$$

它不等于 AC 安全惩罚。它惩罚的是“动作需要被边界或安全机制修正”。

### 7.1 四类投影距离

| 投影距离 | 中文解释 | 字段 |
|---|---|---|
| 原始动作投影距离 | 原始动作在进入执行前被预裁剪的距离 | `action_projection_gap_mw` 的一部分 |
| 本地边界投影距离 | 被 VPP/DER 本地上下限、FR/DOE 投影的距离 | `local_bounds_projection_gap_mw` |
| AC-aware DOE 投影距离 | 被灵敏度安全边界进一步收缩的距离 | `ac_aware_projection_gap_mw` |
| AC certificate 修复距离 | 经 pandapower AC 试算后为通过安全检查而修复的距离 | `ac_certified_projection_gap_mw` |

### 7.2 DSO 成本里的动作投影惩罚

先计算综合投影距离：

$$
d
=
\max\left(
\text{总动作投影距离},
\text{本地边界投影距离}
+
\text{AC-aware投影距离}
+
\text{AC证书修复距离}
\right)
$$

原始惩罚：

$$
\text{动作投影原始惩罚}
=
250d^2+2n
$$

其中：

| 符号 | 含义 |
|---|---|
| \(d\) | 综合投影距离，单位 MW |
| \(n\) | 投影次数 |

加权惩罚：

$$
\text{动作投影加权惩罚}
=
w_{\text{projection}}
\times
(250d^2+2n)
$$

缩放惩罚：

$$
\text{动作投影缩放惩罚}
=
\min\left(
10,\,
\frac{\text{动作投影加权惩罚}}{10}
\right)
$$

例子：

若 \(d=0.04\text{ MW}\)、\(n=1\)、权重 \(w=5\)：

$$
\text{原始惩罚}=250\times0.04^2+2=2.4
$$

$$
\text{加权惩罚}=5\times2.4=12
$$

$$
\text{缩放惩罚}=12/10=1.2
$$

$$
\text{DSO reward直接扣分}=0.05\times1.2=0.06
$$

对应字段：

| 字段 | 中文解释 |
|---|---|
| `raw_action_projection_penalty` | 动作投影原始惩罚 |
| `action_projection_penalty` | 动作投影加权惩罚 |
| `scaled_action_projection_penalty` | 动作投影缩放惩罚 |
| `action_projection_gap_mw` | 动作投影距离记录 |
| `local_bounds_projection_gap_mw` | 本地边界投影距离 |
| `ac_aware_projection_gap_mw` | AC-aware DOE 投影距离 |
| `ac_certified_projection_gap_mw` | AC certificate 修复距离 |
| `action_projection_count` | 投影次数 |

## 8. 安全壳惩罚

安全壳惩罚是训练器额外扣的，不属于 DSO `total_cost`。

它的目的：

$$
\text{不要让actor把安全投影当成免费兜底}
$$

### 8.1 安全壳介入量

先取四个非负 gap：

$$
\begin{aligned}
g_{\text{action}} &= \max(0,\text{action projection gap}) \\
g_{\text{local}} &= \max(0,\text{local bounds gap}) \\
g_{\text{ac}} &= \max(0,\text{AC-aware gap}) \\
g_{\text{cert}} &= \max(0,\text{AC certificate gap})
\end{aligned}
$$

如果本地、AC-aware、certificate 任一非零：

$$
g_{\text{shield}}
=
g_{\text{local}}
+
g_{\text{ac}}
+
g_{\text{cert}}
$$

否则：

$$
g_{\text{shield}}
=
\max(
g_{\text{action}},
g_{\text{local}},
g_{\text{ac}},
g_{\text{cert}}
)
$$

### 8.2 安全壳惩罚公式

$$
\text{安全壳惩罚}
=
5
\times
g_{\text{shield}}
+
10
\times
g_{\text{shield}}^2
$$

训练器中：

$$
\text{DSO训练reward}
=
\text{DSO环境reward}
-
\lambda_{\text{DSO shield}}
\times
\text{安全壳惩罚}
$$

$$
\text{VPP dispatch训练reward}
=
\text{VPP dispatch环境reward}
-
\lambda_{\text{dispatch shield}}
\times
\text{安全壳惩罚}
$$

当前 portfolio 不扣安全壳惩罚：

$$
\text{VPP portfolio训练reward}
=
\text{VPP portfolio环境reward}
$$

但非 portfolio 决策步会被置 0。

### 8.3 动作投影惩罚和安全壳惩罚的区别

| 对比项 | 动作投影惩罚 | 安全壳惩罚 |
|---|---|---|
| 位置 | DSO 环境 reward 成本项 | 训练器学习层额外扣分 |
| 是否进入 `total_cost` | 是 | 否 |
| 公式 | \(250d^2+2n\)，再乘权重和缩放 | \(5g+10g^2\) |
| 主要作用 | 记录/惩罚 DSO 或动作不可执行程度 | 防止 actor 依赖投影兜底 |
| 影响谁 | DSO reward | DSO 和 dispatch 训练 reward |

## 9. 电网安全惩罚

### 9.1 AC 后验安全

AC 后验安全指：动作已经写入 pandapower 网络并运行 AC 潮流之后，再检查电压、线路、变压器、潮流是否安全。

真正判断安全要看：

| 字段 | 中文名 | 含义 |
|---|---|---|
| `post_ac_violation_count` | AC 后验违规总数 | 所有违规记录数量 |
| `post_ac_voltage_violation_count` | AC 后验电压违规数量 | 电压越上限/下限的母线数量 |
| `post_ac_line_overload_count` | AC 后验线路过载数量 | 线路 loading 超限数量 |
| `post_ac_trafo_overload_count` | AC 后验变压器过载数量 | 变压器 loading 超限数量 |
| `post_ac_powerflow_failed` | AC 潮流失败 | pandapower 是否不收敛 |
| `post_ac_violation_magnitude` | AC 违规幅值 | 所有违规幅值绝对值之和 |
| `post_ac_security_penalty` | AC 安全惩罚 | 安全相关加权惩罚总和 |
| `scaled_post_ac_security_penalty` | 缩放 AC 安全惩罚 | 安全相关缩放惩罚总和 |

### 9.2 电压越限

如果母线电压低于下限：

$$
\Delta V=V_{\min}-V
$$

如果母线电压高于上限：

$$
\Delta V=V-V_{\max}
$$

电压越限原始惩罚：

$$
\text{电压越限原始惩罚}
=
\sum_{\text{越限母线}}
10000
\times
\Delta V^2
$$

### 9.3 线路过载

$$
\Delta L
=
\text{实际线路loading百分比}
-
\text{线路loading限制百分比}
$$

线路过载原始惩罚：

$$
\text{线路过载原始惩罚}
=
\sum_{\text{过载线路}}
5
\times
\Delta L^2
$$

### 9.4 变压器过载

$$
\Delta T
=
\text{实际变压器loading百分比}
-
\text{变压器loading限制百分比}
$$

变压器过载原始惩罚：

$$
\text{变压器过载原始惩罚}
=
\sum_{\text{过载变压器}}
5
\times
\Delta T^2
$$

### 9.5 潮流失败

$$
\text{潮流失败原始惩罚}
=
\begin{cases}
1000, & \text{pandapower AC潮流不收敛} \\
0, & \text{pandapower AC潮流收敛}
\end{cases}
$$

## 10. 设备层成本：comfort、SOC、DER cost

### 10.1 DER 运行成本

每个 DER 有二次运行成本：

$$
\text{DER运行成本}
=
aP^2+b|P|+c
$$

VPP DER 运行成本：

$$
\text{VPP运行成本}
=
\sum_{\text{DER}\in\text{VPP}}
(aP^2+b|P|+c)
$$

VPP dispatch 的 `der_operation_cost` 会乘时间步长：

$$
\text{dispatch DER成本}
=
\text{VPP运行成本}
\times
\Delta t
$$

### 10.2 HVAC 舒适度惩罚

室温误差：

$$
e_T
=
\text{室内温度}
-
\text{设定温度}
$$

硬越界量：

$$
h
=
\max(0,T_{\min}-T)
+
\max(0,T-T_{\max})
$$

HVAC 舒适度惩罚：

$$
\text{HVAC舒适度惩罚}
=
a e_T^2
+
b |e_T|
+
100h^2
$$

VPP 舒适度惩罚：

$$
\text{VPP舒适度惩罚}
=
\sum_{\text{HVAC}\in\text{VPP}}
\text{HVAC舒适度惩罚}
$$

### 10.3 储能 SOC 惩罚

如果储能 SOC 低于下限：

$$
\text{低SOC惩罚}
=
1000
\times
(\text{SOC下限}-\text{当前SOC})^2
$$

如果储能 SOC 高于上限：

$$
\text{高SOC惩罚}
=
1000
\times
(\text{当前SOC}-\text{SOC上限})^2
$$

储能 SOC 惩罚：

$$
\text{储能SOC惩罚}
=
\text{低SOC惩罚}
+
\text{高SOC惩罚}
$$

### 10.4 EVCS 未满足 SOC 惩罚

EV 只在离站时刻检查是否达到目标 SOC。

如果不是离站时刻：

$$
\text{EV未满足SOC惩罚}=0
$$

如果到达离站时刻：

$$
\text{EV未满足SOC惩罚}
=
500
\times
\max(0,\text{目标SOC}-\text{当前SOC})^2
$$

## 11. VPP dispatch reward

### 11.1 总公式

VPP dispatch reward 是单个 VPP 快周期调度智能体的 reward：

$$
\begin{aligned}
\text{VPP dispatch reward}
=&
0.02
\times
\text{私有利润代理}
+
\text{偏好区间奖励}
\\
&-
25
\times
\text{VPP局部目标跟踪偏差}^2
\\
&-
\left(
5
\times
\text{本地投影距离}
+
10
\times
\text{本地投影距离}^2
\right)
\\
&-
0.001
\times
\text{缩放舒适/SOC惩罚}
\end{aligned}
$$

### 11.2 私有利润代理

$$
\text{私有利润代理}
=
\text{电能市场收入}
+
\text{灵活性服务收入}
+
\text{可用容量收入}
-
\text{DER运行成本}
$$

电能市场收入：

$$
\text{电能市场收入}
=
\text{price}
\times
P_{\text{delivered}}
\times
\Delta t
$$

灵活性服务收入：

$$
\text{灵活性服务收入}
=
1.00
\times
\text{price}
\times
\text{服务功率数量}
\times
\Delta t
$$

可用容量收入：

$$
\text{可用容量收入}
=
0.02
\times
\text{price}
\times
\text{可调容量宽度}
\times
\Delta t
$$

DER 运行成本：

$$
\text{DER运行成本}
=
\sum_{\text{DER}}
(aP^2+b|P|+c)
\times
\Delta t
$$

服务功率数量：

$$
\text{服务功率数量}
=
\begin{cases}
\max(0,-P_{\text{delivered}}), & \text{吸收/充电/增加用电服务} \\
\max(0,P_{\text{delivered}}), & \text{注入/削减负荷/外送服务} \\
|P_{\text{delivered}}|, & \text{未指定方向}
\end{cases}
$$

注意：当前代码的服务收入是方向性代理收入，不是严格的 baseline-based 合同结算。任务草稿里提到的 baseline-based payment 是后续可改造项。

### 11.3 VPP dispatch 字段解释

| 字段 | 中文名 | 公式/含义 |
|---|---|---|
| `vpp_dispatch_reward` | VPP调度奖励 | 上述总公式 |
| `delivered_p_mw` | 实际交付功率 | VPP 内所有 DER 当前功率求和 |
| `target_p_mw` | 目标功率 | 投影后目标或 DSO 偏好目标 |
| `flex_span_mw` | 可调容量宽度 | \(P_{\max}-P_{\min}\) |
| `private_profit_proxy` | 私有利润代理 | 电能收入 + 服务收入 + 可用容量收入 - DER成本 |
| `energy_market_revenue` | 电能市场收入 | \(\text{price}\times P_{\text{delivered}}\times \Delta t\) |
| `flexibility_service_payment` | 灵活性服务收入 | \(\text{price}\times\text{服务功率数量}\times\Delta t\) |
| `availability_payment` | 可用容量收入 | \(0.02\times\text{price}\times\text{flex span}\times\Delta t\) |
| `der_operation_cost` | DER运行成本 | DER 成本求和后乘步长 |
| `target_tracking_penalty` | 局部跟踪惩罚 | \(25\times|P_{\text{delivered}}-P_{\text{target}}|^2\) |
| `envelope_projection_penalty` | envelope投影惩罚 | \(5d+10d^2\) |
| `raw_comfort_soc_penalty` | 原始舒适/SOC惩罚 | 本 VPP 的 comfort + SOC |
| `scaled_comfort_soc_penalty` | 缩放舒适/SOC惩罚 | \(\min(5,\max(0,\text{raw})/100)\) |
| `comfort_soc_penalty` | 舒适/SOC惩罚 | 当前等于 `scaled_comfort_soc_penalty` |
| `private_profit_weight` | 私有利润权重 | 当前为 0.02 |

## 12. 偏好区间奖励

偏好区间奖励：

$$
\text{偏好区间奖励}
=
0.50
\times
\text{inside}
\times
\text{lambda gate}
\times
\text{width gate}
\times
\text{effectiveness gate}
$$

逐项：

| 字段 | 中文名 | 公式/判定 | 解释 |
|---|---|---|---|
| `preferred_inside_range` | 是否在偏好区间内 | 在区间内为 1，否则 0 | VPP 实际功率是否落在 DSO 偏好范围 |
| `preferred_bonus_lambda_gate` | DSO引导强度门控 | `guidance_strength_lambda` 裁剪到 0 到 1 | DSO actor 对这个 VPP 的引导强度 |
| `preferred_bonus_width_gate` | 边界收缩门控 | \(1-\text{偏好宽度}/\text{硬边界宽度}\) | 偏好区间越窄，DSO 指令越明确 |
| `preferred_bonus_effectiveness_gate` | 响应有效性门控 | \(1-\text{投影距离}/\text{硬边界宽度}\)，再裁剪 | 动作被投影越多，说明原始响应越无效 |
| `preferred_region_score` | 偏好区间得分 | 区间内为 1，区间外按距离下降 | portfolio 的局部 DSO 对齐也会用 |

区间外得分：

$$
\text{偏好区间得分}
=
\max\left(
0,\,
1-
\frac{\text{到偏好区间最近边界的距离}}{\text{偏好区间宽度}}
\right)
$$

## 13. VPP portfolio reward

VPP portfolio 是慢周期资源组合/重配智能体。

总公式：

$$
\text{VPP portfolio reward}
=
\text{长期利润代理}
+
\text{局部DSO对齐奖励}
+
\text{可靠性奖励}
-
\text{切换成本}
-
\text{交付风险惩罚}
$$

长期利润代理：

$$
\text{长期利润代理}
=
0.10
\times
\text{私有利润代理}
$$

可靠性奖励：

$$
\text{可靠性奖励}
=
0.50
\times
\max\left(
0,\,
1-
\frac{\text{目标跟踪偏差}}{\max(10^{-6},\text{可调容量宽度})}
\right)
$$

可用容量质量：

$$
\text{可用容量质量}
=
\min\left(
1,\,
\frac{\text{可调容量宽度}}{0.50}
\right)
$$

网络惩罚：

$$
\text{网络惩罚}
=
\text{电压越限惩罚}
+
\text{线路过载惩罚}
+
\text{变压器过载惩罚}
+
\text{潮流失败惩罚}
+
\text{AC违规幅值惩罚}
$$

局部 DSO 对齐奖励：

$$
\begin{aligned}
\text{局部DSO对齐奖励}
=&
0.35
\times
\text{偏好区间得分}
+
0.25
\times
\text{可行奖励}
\\
&+
0.15
\times
\text{可用容量质量}
+
0.10
\times
\text{可靠性奖励}
-
0.001
\times
\text{网络惩罚}
\end{aligned}
$$

切换成本：

$$
\text{切换成本}
=
\begin{cases}
0, & \text{keep，保持当前组合} \\
0.02, & \text{reweight，重新加权} \\
0.08, & \text{propose membership change，建议成员变化}
\end{cases}
$$

交付风险惩罚：

$$
\text{交付风险惩罚}
=
0.50
\times
\text{投影距离}
+
0.20
\times
\text{目标跟踪偏差}
$$

字段解释：

| 字段 | 中文名 | 解释 |
|---|---|---|
| `vpp_portfolio_reward` | VPP组合奖励 | portfolio agent 的环境 reward |
| `long_horizon_profit_proxy` | 长期利润代理 | dispatch 私有利润代理的 0.10 倍 |
| `localized_dso_alignment_reward` | 局部DSO对齐奖励 | 用偏好得分、安全可行、可用容量、可靠性近似 DSO 目标 |
| `reliability_bonus` | 可靠性奖励 | 实际交付越接近目标越高 |
| `availability_quality` | 可用容量质量 | 可调容量达到 0.50 MW 记为满分 |
| `switching_cost` | 切换成本 | 防止频繁重配 |
| `delivery_risk_penalty` | 交付风险惩罚 | 投影越大、跟踪越差，风险越高 |
| `global_reward_variant_weight` | 局部DSO对齐权重 | 当前为 1.0 |
| `raw_dso_reward_shared` | raw DSO reward 共享权重 | 当前为 0，表示不直接共享全局 DSO reward |

## 14. 训练 reward 和日志字段

训练器会先拿环境 reward，然后扣安全壳惩罚。

DSO：

$$
\text{DSO训练reward}
=
\text{DSO环境reward}
-
\lambda_{\text{DSO shield}}
\times
\text{安全壳惩罚}
$$

Dispatch：

$$
\text{dispatch训练reward}
=
\text{dispatch环境reward}
-
\lambda_{\text{dispatch shield}}
\times
\text{安全壳惩罚}
$$

Portfolio：

$$
\text{portfolio训练reward}
=
\begin{cases}
\text{portfolio环境reward}, & \text{portfolio决策步} \\
0, & \text{非portfolio决策步}
\end{cases}
$$

然后乘 reward scale：

$$
\text{进入critic/value的reward}
=
\text{训练reward}
\times
\text{reward scale}
$$

常见：

$$
\text{reward scale}=0.01
$$

字段解释：

| 字段 | 中文名 | 解释 |
|---|---|---|
| `reward` | 日志总奖励 | DSO训练reward + dispatch平均训练reward + portfolio平均训练reward |
| `dso_reward` | DSO训练reward | 已经可能扣过安全壳惩罚 |
| `mean_dispatch_reward` | 平均dispatch训练reward | 所有 VPP dispatch reward 平均 |
| `mean_portfolio_reward` | 平均portfolio训练reward | 所有 VPP portfolio reward 平均，非决策步可为 0 |
| `raw_dso_reward_before_shield_penalty` | 扣安全壳前的 DSO reward | 用来判断 DSO reward 和安全壳扣分差距 |
| `raw_dispatch_reward_before_shield_penalty` | 扣安全壳前的 dispatch reward | 用来判断 VPP 动作是否依赖安全壳 |
| `raw_portfolio_reward_before_decision_mask` | portfolio 掩码前 reward | 用来判断 portfolio 分支本身是否有信号 |
| `portfolio_decision_step` | 是否为 portfolio 决策步 | false 时 portfolio reward 可能被置 0 |
| `reward_scale` | reward缩放系数 | 进入 critic/value 前乘的缩放 |

## 15. 神经网络 loss 指标

### 15.1 reward 和 loss 的区别

reward 是环境反馈：

$$
\text{动作好不好}
$$

loss 是神经网络训练目标：

$$
\text{参数应该怎样更新}
$$

reward 越高通常越好；loss 不是越高越好，也不是简单越低就一定策略越好，要结合 KL、entropy、reward、违规、projection 一起看。

### 15.2 HAPPO/HATRPO 的 on-policy 指标

HAPPO/PPO 类策略损失近似：

$$
\text{policy loss}
=
-
\operatorname{mean}
\left[
\min
\left(
r_t A_t,\,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
-
\beta H(\pi)
$$

其中：

| 符号 | 中文解释 |
|---|---|
| \(r_t\) | 新旧策略概率比 |
| \(A_t\) | advantage，动作比 critic 预期好多少 |
| \(\epsilon\) | PPO clip ratio |
| \(H(\pi)\) | 策略熵，表示探索程度 |
| \(\beta\) | entropy 系数 |

价值损失：

$$
\text{critic loss}
=
\operatorname{mean}
\left[
(V(s)-R_{\text{return}})^2
\right]
$$

字段解释：

| 字段 | 中文名 | 怎么理解 |
|---|---|---|
| `critic_loss` | 价值网络损失 | critic/value 对 return 的拟合误差。长期很大说明 critic 学不稳。 |
| `dso_policy_loss` | DSO策略损失 | DSO actor 的 PPO/HAPPO surrogate loss。正负不直接代表好坏。 |
| `dispatch_policy_loss` | dispatch策略损失 | VPP dispatch actor 的 surrogate loss。 |
| `portfolio_policy_loss` | portfolio策略损失 | portfolio actor 的 surrogate loss。 |
| `policy_loss` | 单角色策略损失 | update metrics 中按角色记录。 |
| `entropy_mean` | 平均熵 | 越大说明动作分布越随机；过快降到 0 说明探索塌缩。 |
| `approx_kl` | 近似KL | 新旧策略差异。太大说明一步更新过猛。 |
| `target_kl` | KL阈值 | 超过后可能 early stop，常见 0.02。 |
| `target_kl_exceeded` | 是否超过KL阈值 | true 多说明策略更新太激进。 |
| `grad_norm` | 梯度范数 | 巨大或 NaN 说明训练不稳定。 |
| `critic_grad_norm` | critic梯度范数 | critic 反向传播梯度大小。 |
| `ratio_mean` | 概率比均值 | 新旧策略概率比均值，长期远离 1 需警惕。 |
| `correction_mean` | 顺序更新修正项 | HAPPO 多角色顺序更新的修正因子。 |

### 15.3 HASAC 的 off-policy 指标

HASAC critic target 近似：

$$
y
=
r
+
\gamma
\left(
\min(Q_1',Q_2')
-
\alpha\log\pi(a'|s')
\right)
$$

critic loss：

$$
\text{critic loss}
=
\operatorname{mean}
\left[
(Q_1-y)^2
+
(Q_2-y)^2
\right]
$$

actor loss：

$$
\text{actor loss}
=
\operatorname{mean}
\left[
\alpha\log\pi(a|s)
-
Q(s,a)
\right]
$$

字段解释：

| 字段 | 中文名 | 怎么理解 |
|---|---|---|
| `critic_loss` | 双Q critic损失 | Q target 拟合误差。 |
| `actor_loss` | 总 actor 损失 | DSO actor loss + dispatch actor loss。 |
| `dso_actor_loss` | DSO actor损失 | DSO 分支的 SAC actor loss。 |
| `dispatch_actor_loss` | dispatch actor损失 | dispatch 分支的 SAC actor loss。 |
| `alpha_loss` | 温度系数损失 | 自动调节探索温度 \(\alpha\)。 |
| `alpha_dso` | DSO entropy温度 | DSO 探索权重。 |
| `alpha_dispatch` | dispatch entropy温度 | VPP dispatch 探索权重。 |
| `target_entropy_dso` | DSO目标熵 | 希望 DSO 策略保持的探索程度。 |
| `target_entropy_dispatch` | dispatch目标熵 | 希望 dispatch 策略保持的探索程度。 |

## 16. 读曲线时最重要的组合判断

不要只看总 `reward`。建议同时看：

| 组合 | 健康现象 | 风险现象 |
|---|---|---|
| `reward` + `dso_reward` + `mean_dispatch_reward` + `mean_portfolio_reward` | 总 reward 上升，且不是单靠 DSO 一个分支撑起来 | 总 reward 高，但 dispatch/portfolio 没变化 |
| `post_ac_violation_count` + `shield_intervention_gap_mw` | 违规为 0，安全壳介入也下降 | 违规为 0，但安全壳介入不下降 |
| `raw_dso_reward_before_shield_penalty` + `dso_reward` | 两者差距逐渐变小 | 差距大，说明安全壳扣分多 |
| `projection_gap_mw` + `action_projection_count` | 趋势下降 | 长期高，说明动作常被修 |
| `critic_loss` + `approx_kl` + `entropy_mean` | critic 稳定，KL 不爆，entropy 缓慢下降 | critic 爆、KL 超限、entropy 快速塌缩 |

判断平台期是否健康：

| 情况 | 解释 |
|---|---|
| reward 高、post-AC违规为 0、shield gap 下降 | 健康平台期，策略可能逐步学会安全边界 |
| reward 高、post-AC违规为 0、shield gap 不下降 | 不健康平台期，可能是安全壳兜底造成高 reward |
| reward 不涨、critic loss 下降、projection下降 | 可能是 reward 初始已高，策略在减少不可见风险 |
| reward 不涨、critic loss 大、KL 大、entropy塌缩 | 训练不稳定 |

## 17. 当前容易过时或容易误读的 reward 术语

这些术语已经在代码中有不同程度的实现，但解释实验结果时必须区分“公式已实现”“审计已落盘”“最新 paper-long 是否已经产生相应数据”。

| 术语 | 设计含义 | 当前状态 |
|---|---|---|
| `v2_minimal` | 将 DSO reward 简化为安全、成本、容量释放、少投影等核心项 | 已在 `RewardConfig.from_dict()`、`DSO.calculate_reward_or_cost()` 和 `vpp_dispatch_reward_components()` 中实现。注意：旧配置仍可能使用 legacy 口径；必须看输出目录里的 `resolved_reward_config.yaml`。 |
| `safe_capacity_utilization` | 安全释放容量 / 名义可行容量 | 已进入 v2_minimal DSO reward，字段为 `dso_safe_capacity_utilization_reward`。 |
| `over_conservative_curtailment` | 电网安全时 DSO 释放容量过少的保守性诊断 | 字段 `dso_over_conservative_curtailment_penalty` 是诊断基量；实际训练扣分项是 `dso_curtailment_cost = curtailment_cost_weight * over_conservative`。二者不能同时放进 reward 分母，否则会双计。 |
| `baseline_based_service_payment` | 服务收入按 baseline 与实际响应差值结算 | v2_minimal dispatch 已启用 `use_baseline_service_payment: true`，公式为 `price * verified_delivery_mw * dt_hours`。若实验中该项为 0，表示服务交付未激活或 verified delivery 为 0，不表示代码缺项。 |
| `contract_shortfall_mw` | 合同交付缺口 | 已是主 dispatch reward 标准输出，并进入 `contract_delivery_penalty = contract_delivery_weight * contract_shortfall_mw^2`。 |
| `attribute_projection_gaps` | 把投影责任拆给 DSO、dispatch、portfolio、外生因素 | 已有 `attribute_projection_gaps()`。当前 dispatch reward 使用 `dispatch_responsible_projection_gap_mw`，DSO reward 使用 AC-aware/certified gap 形成 `dso_responsible_projection_penalty`。仍需注意：审计表中的 safety shield penalty 是单独 source，若要算 after-shield role 占比，必须按训练器系数分配回 DSO/dispatch。 |
| `role_internal_reward_share` | 每个 role 内部各 reward 项占比 | train probe 可对 dispatch/portfolio 较完整计算；最新完整 paper-long 目录若没有 `*_step_metrics.csv`，则不能计算训练 role 内占比。DSO 还需要避免父子诊断项双计。 |

## 18. 最短阅读路径

如果你只想看懂训练结果，按这个顺序读：

1. `DSO环境reward`：看全局调度安全和目标响应。
2. `目标跟踪误差`：看 VPP 是否执行 DSO 目标。
3. `动作投影惩罚` 和 `安全壳惩罚`：看策略是否依赖安全外壳。
4. `post_ac_*` 安全字段：看最终配电网是否真的 AC 安全。
5. `mean_dispatch_reward`：看 VPP 快周期调度是否参与学习。
6. `mean_portfolio_reward` 和 `portfolio_decision_step`：看慢周期组合智能体是否真正介入。
7. `critic_loss`、`approx_kl`、`entropy_mean`：看神经网络训练是否稳定。
