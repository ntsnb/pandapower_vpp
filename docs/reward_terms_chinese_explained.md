# Reward 数学定义、Actor/Critic 归属与实验量级说明

更新日期：2026-06-04

完整逐词解释总表见：`docs/reward_terms_full_glossary_cn.md`。

如果只想查“目标跟踪误差、动作投影惩罚、原始成本、有效响应得分、缩放因子、训练 loss”等术语，请优先阅读这份新文档。本文保留原来的公式、实验占比和历史说明。

本文档回答四个问题：

1. 每一个 reward 项到底怎么算。
2. 每一个 reward 属于哪个 actor，训练时进入哪个 critic 或 value head。
3. 每一个权重来自哪里，当前配置是否真的生效。
4. 已有实验里每一项大致多大，占比大概是多少。

核心代码证据：

- `src/vpp_dso_sim/entities/dso.py:37-57`：DSO reward 的缩放规则。
- `src/vpp_dso_sim/entities/dso.py:123-275`：DSO 成本、AC 安全项、最终 DSO reward。
- `src/vpp_dso_sim/envs/reward_design.py:120-216`：VPP dispatch reward。
- `src/vpp_dso_sim/envs/reward_design.py:219-283`：VPP portfolio reward。
- `src/vpp_dso_sim/envs/reward_design.py:286-318`：多智能体 reward map。
- `src/vpp_dso_sim/learning/reward_contracts.py:36-50`：VPP reward 与安全壳 penalty 常量。
- `src/vpp_dso_sim/learning/advanced_marl.py:1662-1710`：on-policy HAPPO reward vector。
- `src/vpp_dso_sim/learning/advanced_marl.py:2664-2772`：off-policy HASAC reward vector 与 Q target。
- `src/vpp_dso_sim/learning/hatrpo.py:679-713`：HATRPO reward vector。

统计数据来源：

- 旧 long 完整实验：`outputs/paper_training_long_current`
- 当前 sensitivity-v1 进度目录：`outputs/paper_training_long_sensitivity_v1_20260528_thread8_progress`
- 当前 HAPPO/HATRPO smoke 小样本：`outputs/test_paper_training_structured_happo_sensitivity_smoke`、`outputs/test_happo_cuda_smoke_20260601`、`outputs/test_hatrpo_training`
- VPP 子项小样本轨迹：`outputs/test_privacy_separated_ctde_eval/train/deep_rl_trajectory.csv` 等

统计口径说明：

- `abs-cost-share` 表示 `mean(abs(某成本项)) / sum(mean(abs(所有 DSO 成本项)))`。它不是 signed reward 占比，而是用于看哪个成本项在量级上主导 DSO 成本。
- `abs-share-vs-learning` 表示 `mean(abs(角色 reward)) / sum(mean(abs(角色 reward)))`。它用于看 DSO、dispatch、portfolio 哪个角色在学习信号中更强。
- smoke 小样本只用于解释量级，不能代替 paper-long 结论。

## 0. 公式渲染快读版

这一节先不用代码变量名，而是用论文式数学公式说明。后文仍保留 CSV 字段名，方便你追踪实验结果。

### 0.1 DSO 全局奖励

DSO 的核心目标是：电网安全、少投影、低运行成本、VPP 能跟踪 DSO 指令。

最终给 DSO 智能体的环境奖励是：

$$
\text{DSO环境奖励}
=
\text{可行奖励}
+ \text{跟踪奖励}
+ \text{有效响应奖励}
- 0.05 \times \text{缩放总成本}
$$

其中：

$$
\text{可行奖励}
=
\begin{cases}
1, & \text{AC 潮流后无电压/线路/变压器/潮流失败违规} \\
0, & \text{否则}
\end{cases}
$$

$$
\text{跟踪奖励}
=
\frac{0.25}{1 + \text{目标跟踪误差}}
$$

$$
\text{有效响应奖励}
=
\begin{cases}
\text{有效响应得分}, & \text{如果环境已显式计算该得分} \\
\dfrac{1}{1+\text{目标跟踪误差}}, & \text{否则}
\end{cases}
$$

缩放总成本按每个成本项先加权、再缩放、再裁剪：

$$
\text{缩放总成本}
=
\sum_{k \in \mathcal{C}}
\min\left(
10,\,
\frac{\max(0,\, \text{权重}_k \times \text{原始成本}_k)}
{\text{缩放因子}_k}
\right)
$$

成本项集合为：

$$
\mathcal{C}
=
\{
\text{运行成本},
\text{目标跟踪误差惩罚},
\text{动作投影惩罚},
\text{舒适度惩罚},
\text{SOC惩罚},
\text{安全违规数量惩罚},
\text{AC违规幅值惩罚},
\text{电压越限惩罚},
\text{线路过载惩罚},
\text{变压器过载惩罚},
\text{潮流失败惩罚},
\text{边界宽度惩罚},
\text{边界平滑惩罚}
\}
$$

未缩放的加权总成本为：

$$
\begin{aligned}
\text{加权总成本}
=&
\text{运行成本}
+ \text{目标跟踪误差惩罚}
+ \text{动作投影惩罚}
+ \text{舒适度惩罚}
+ \text{SOC惩罚}
\\
&+
\text{安全违规数量惩罚}
+ \text{AC违规幅值惩罚}
+ \text{电压越限惩罚}
+ \text{线路过载惩罚}
+ \text{变压器过载惩罚}
\\
&+
\text{潮流失败惩罚}
+ \text{边界宽度惩罚}
+ \text{边界平滑惩罚}
\end{aligned}
$$

### 0.2 DSO 动作投影惩罚

动作投影惩罚用于惩罚“DSO 或 VPP 的原始动作被安全边界修正了多少”。

先计算综合投影距离：

$$
\text{综合投影距离}
=
\max\left(
\text{原始动作投影距离},
\text{本地边界投影距离}
+ \text{AC-aware边界投影距离}
+ \text{AC证书修复距离}
\right)
$$

DSO 成本里的动作投影惩罚为：

$$
\text{动作投影惩罚}
=
\text{投影权重}
\times
\left(
250 \times \text{综合投影距离}^{2}
+ 2 \times \text{投影次数}
\right)
$$

当前权重：

$$
\text{投影权重}
=
\begin{cases}
2.0, & \text{base/paper-long 旧配置} \\
5.0, & \text{sensitivity-v1 配置}
\end{cases}
$$

### 0.3 VPP 快周期调度奖励

每个 VPP dispatch 智能体的奖励是：

$$
\begin{aligned}
\text{VPP调度奖励}
=&
0.02 \times \text{私有利润代理}
+ \text{偏好区间奖励}
\\
&-
25 \times \text{目标跟踪偏差}^{2}
-
\left(
5 \times \text{投影偏差}
+ 10 \times \text{投影偏差}^{2}
\right)
\\
&-
0.001 \times \text{缩放舒适/SOC惩罚}
\end{aligned}
$$

私有利润代理为：

$$
\text{私有利润代理}
=
\text{电能市场收入}
+ \text{灵活性服务收入}
+ \text{可用容量收入}
- \text{DER运行成本}
$$

其中：

$$
\text{电能市场收入}
=
\text{电价}
\times
\text{VPP实际聚合功率}
\times
\text{步长小时数}
$$

$$
\text{灵活性服务收入}
=
1.00
\times
\text{电价}
\times
\text{服务功率数量}
\times
\text{步长小时数}
$$

$$
\text{可用容量收入}
=
0.02
\times
\text{电价}
\times
\text{VPP可调功率区间宽度}
\times
\text{步长小时数}
$$

服务功率数量按 DSO 服务请求方向取值：

$$
\text{服务功率数量}
=
\begin{cases}
\max(0, -\text{VPP实际聚合功率}), & \text{吸收/充电/增加用电服务} \\
\max(0, \text{VPP实际聚合功率}), & \text{注入/削减负荷/外送服务} \\
|\text{VPP实际聚合功率}|, & \text{未指定方向}
\end{cases}
$$

缩放舒适/SOC惩罚为：

$$
\text{缩放舒适/SOC惩罚}
=
\min\left(
5,\,
\frac{
\max(0,\, \text{舒适度惩罚}+\text{SOC惩罚})
}{100}
\right)
$$

偏好区间奖励为：

$$
\text{偏好区间奖励}
=
0.50
\times
\text{是否落在偏好区间}
\times
\text{DSO引导强度}
\times
\text{边界收缩程度}
\times
\text{响应有效性}
$$

其中：

$$
\text{是否落在偏好区间}
=
\begin{cases}
1, & \text{VPP实际功率在 DSO 偏好区间内} \\
0, & \text{否则}
\end{cases}
$$

### 0.4 VPP 慢周期组合奖励

每个 VPP portfolio 智能体的奖励是：

$$
\begin{aligned}
\text{VPP组合奖励}
=&
0.10 \times \text{私有利润代理}
+ \text{局部DSO对齐奖励}
+ \text{可靠性奖励}
\\
&-
\text{切换成本}
-
\text{交付风险惩罚}
\end{aligned}
$$

局部 DSO 对齐奖励为：

$$
\begin{aligned}
\text{局部DSO对齐奖励}
=&
0.35 \times \text{偏好区间得分}
+ 0.25 \times \text{可行奖励}
+ 0.15 \times \text{可用容量质量}
\\
&+
0.10 \times \text{可靠性奖励}
- 0.001 \times \text{网络惩罚}
\end{aligned}
$$

可靠性奖励为：

$$
\text{可靠性奖励}
=
0.50
\times
\max\left(
0,\,
1 -
\frac{
\text{目标跟踪偏差}
}{
\max(10^{-6},\, \text{VPP可调功率区间宽度})
}
\right)
$$

可用容量质量为：

$$
\text{可用容量质量}
=
\min\left(
1,\,
\frac{\text{VPP可调功率区间宽度}}{0.50}
\right)
$$

网络惩罚为：

$$
\text{网络惩罚}
=
\text{电压越限惩罚}
+ \text{线路过载惩罚}
+ \text{变压器过载惩罚}
+ \text{潮流失败惩罚}
+ \text{AC违规幅值惩罚}
$$

交付风险惩罚为：

$$
\text{交付风险惩罚}
=
0.50 \times \text{投影偏差}
+ 0.20 \times \text{目标跟踪偏差}
$$

切换成本为：

$$
\text{切换成本}
=
\begin{cases}
0.00, & \text{保持当前组合} \\
0.02, & \text{重新加权} \\
0.08, & \text{建议成员关系变化}
\end{cases}
$$

### 0.5 安全壳惩罚

安全壳惩罚用于惩罚“动作本来不安全或不可行，后来被投影/证书修复”。

安全壳介入量为：

$$
\text{安全壳介入量}
=
\begin{cases}
\text{本地边界投影距离}
+ \text{AC-aware边界投影距离}
+ \text{AC证书修复距离},
& \text{如果三者任一非零} \\
\max(
\text{原始动作投影距离},
\text{本地边界投影距离},
\text{AC-aware边界投影距离},
\text{AC证书修复距离}
),
& \text{否则}
\end{cases}
$$

安全壳惩罚为：

$$
\text{安全壳惩罚}
=
5 \times \text{安全壳介入量}
+ 10 \times \text{安全壳介入量}^{2}
$$

训练时：

$$
\text{DSO训练奖励}
=
\text{DSO环境奖励}
- \text{安全壳惩罚}
$$

$$
\text{VPP调度训练奖励}
=
\text{VPP调度环境奖励}
- \text{安全壳惩罚}
$$

当前 portfolio 奖励不扣安全壳惩罚。

### 0.6 训练时真正进入 critic/value 的奖励

HAPPO/HATRPO 每步日志总奖励为：

$$
\text{日志总奖励}
=
\text{DSO训练奖励}
+ \operatorname{mean}(\text{所有VPP调度训练奖励})
+ \operatorname{mean}(\text{所有VPP组合训练奖励})
$$

HASAC/MATD3 连续调度分支主要使用：

$$
\text{角色奖励向量}
=
\left[
\text{DSO训练奖励},
\text{VPP}_1\text{调度训练奖励},
\ldots,
\text{VPP}_N\text{调度训练奖励}
\right]
$$

进入 critic 或 value 之前还会缩放：

$$
\text{critic实际使用奖励}
=
0.01 \times \text{角色训练奖励}
$$

所以你在 CSV 里看到的 reward 数字，和神经网络内部用于更新的 reward 数字差一个 `0.01` 缩放系数。

## 0A. 关键术语逐项解释

这一节专门解释容易混淆的基础量。先记住一条主线：

$$
\text{原始量}
\rightarrow
\text{原始成本}
\rightarrow
\text{加权成本}
\rightarrow
\text{缩放成本}
\rightarrow
\text{最终奖励}
$$

也就是说，reward 不是直接拿“电压”“SOC”“利润”等物理量相加，而是先把它们转换成成本或奖励，再通过权重和缩放进入训练。

### 0A.1 什么是“原始成本”

“原始成本”指的是某个物理或经济指标刚从环境里算出来、还没有乘 reward 权重、还没有除以缩放因子的数值。

例如：

$$
\text{舒适度原始成本}
=
\sum_{\text{所有HVAC}}
\text{室温偏离舒适区的惩罚}
$$

$$
\text{动作投影原始成本}
=
250
\times
\text{综合投影距离}^{2}
+ 2
\times
\text{投影次数}
$$

$$
\text{线路过载原始成本}
=
\sum_{\text{过载线路}}
5
\times
\text{线路超限百分比}^{2}
$$

原始成本的特点：

| 特点 | 解释 |
|---|---|
| 单位不统一 | 运行成本像“电价乘功率”，电压越限是 p.u. 偏差，线路过载是百分比超限，SOC 是比例偏差。 |
| 数值量级差异很大 | 舒适度原始成本可能几千，电压原始成本可能接近 0，潮流失败是 1000。 |
| 不能直接相加后给 RL | 如果直接相加，最大量级的项会压倒其他项，旧 long 实验就是 comfort 项主导。 |
| 需要权重和缩放 | 权重决定科研偏好，缩放因子解决数值稳定。 |

### 0A.2 原始成本、加权成本、缩放成本分别是什么

对任意一个成本项，项目采用三步：

第一步：计算原始成本。

$$
\text{原始成本}
=
\text{从仿真、设备或潮流结果直接计算出来的惩罚}
$$

第二步：乘权重，得到加权成本。

$$
\text{加权成本}
=
\text{原始成本}
\times
\text{该项权重}
$$

第三步：除以缩放因子并裁剪，得到缩放成本。

$$
\text{缩放成本}
=
\min\left(
10,\,
\frac{\max(0,\text{加权成本})}{\text{缩放因子}}
\right)
$$

最后：

$$
\text{缩放总成本}
=
\sum_{\text{所有成本项}}
\text{缩放成本}
$$

DSO 奖励中真正扣的是：

$$
0.05
\times
\text{缩放总成本}
$$

所以一个成本项要影响 DSO reward，需要经过：

$$
\text{原始成本}
\times
\text{权重}
\div
\text{缩放因子}
$$

### 0A.3 一个数值例子：为什么旧实验 comfort 会压倒其他项

假设某一步：

$$
\text{舒适度原始成本}=4000
$$

当前舒适度权重：

$$
\text{舒适度权重}=0.02
$$

加权后：

$$
\text{舒适度加权成本}
=
4000
\times
0.02
=
80
$$

舒适度缩放因子：

$$
\text{舒适度缩放因子}=100
$$

缩放后：

$$
\text{舒适度缩放成本}
=
\frac{80}{100}
=
0.8
$$

如果每个 step 都有类似规模的 comfort 成本，而电压、线路、安全违规多数 step 为 0，那么总 reward 曲线就会主要反映 comfort，而不是网络安全。

旧 `paper_training_long_current` 统计显示：

$$
\text{舒适度惩罚占DSO加权成本}
\approx
97.5\%
$$

这就是之前诊断“reward 尺度失衡”的核心证据。

### 0A.4 什么是“目标跟踪误差”

项目里有两种跟踪误差，必须分清：

| 名称 | 用在哪里 | 含义 |
|---|---|---|
| DSO 全局目标跟踪误差 | DSO reward | 所有 VPP 最终实际响应和 DSO 目标之间的总偏差 |
| VPP 局部目标跟踪偏差 | VPP dispatch reward | 某一个 VPP 自己的实际功率和该 VPP 目标功率之间的偏差 |

#### DSO 全局目标跟踪误差

每个 step，DSO 会给每个 VPP 一个目标功率或目标边界。VPP 动作经过本地可行域、AC-aware DOE、AC certificate 修复后，最后真正写入 DER。

对每个 VPP：

$$
\text{该VPP跟踪误差}
=
\left|
\text{该VPP最终实际聚合功率}
-
\text{该VPP目标功率}
\right|
$$

DSO 全局目标跟踪误差是所有 VPP 的绝对误差求和：

$$
\text{DSO目标跟踪误差}
=
\sum_{\text{所有VPP}}
\left|
\text{最终实际聚合功率}
-
\text{目标功率}
\right|
$$

单位是 MW。

DSO 中对应的原始惩罚为：

$$
\text{目标跟踪误差原始惩罚}
=
100
\times
\text{DSO目标跟踪误差}^{2}
$$

DSO 还会给一个跟踪奖励：

$$
\text{跟踪奖励}
=
\frac{0.25}{1+\text{DSO目标跟踪误差}}
$$

因此：

| 情况 | 结果 |
|---|---|
| VPP 最终实际功率非常接近 DSO 目标 | 目标跟踪误差接近 0，跟踪惩罚接近 0，跟踪奖励接近 0.25 |
| VPP 被安全壳修复后偏离 DSO 原目标 | 目标跟踪误差变大，跟踪惩罚按平方变大，跟踪奖励下降 |
| 目标本身被投影到可行域内 | 这会减少不可行目标带来的误差，但会增加 projection 相关惩罚 |

#### VPP 局部目标跟踪偏差

每个 VPP dispatch 智能体还会单独计算自己的跟踪偏差：

$$
\text{VPP局部目标跟踪偏差}
=
\left|
\text{该VPP当前实际聚合功率}
-
\text{该VPP收到的目标功率}
\right|
$$

VPP dispatch reward 中扣：

$$
25
\times
\text{VPP局部目标跟踪偏差}^{2}
$$

这和 DSO 全局跟踪误差不是同一个量：

- DSO 是所有 VPP 汇总后的系统级误差。
- VPP dispatch 是单个 VPP 自己的履约误差。

### 0A.5 什么是“有效响应得分”

“有效响应得分”最容易误解，因为代码里有两个相关概念。

#### DSO reward 里的有效响应奖励

DSO reward 函数支持传入一个外部的有效响应得分：

$$
\text{有效响应奖励}
=
\operatorname{clip}(\text{外部有效响应得分},0,1)
$$

但是当前 simulator 调用 DSO reward 时没有传入这个外部得分。因此当前实际使用的是 fallback 公式：

$$
\text{有效响应奖励}
=
\frac{1}{1+\text{DSO目标跟踪误差}}
$$

这意味着当前实验中：

$$
\text{有效响应奖励}
\text{不是一个单独学习出来的神秘指标}
$$

而是目标跟踪误差的另一个单调变换：

| DSO 目标跟踪误差 | 有效响应奖励 |
|---:|---:|
| 0 | 1.000 |
| 0.1 | 0.909 |
| 0.5 | 0.667 |
| 1.0 | 0.500 |
| 2.0 | 0.333 |

所以当前 DSO reward 里，`tracking_bonus` 和 `effective_response_bonus` 都在鼓励“VPP 实际响应接近 DSO 目标”，只是权重和数值范围不同：

$$
\text{跟踪奖励}
=
\frac{0.25}{1+\text{DSO目标跟踪误差}}
$$

$$
\text{有效响应奖励}
=
\frac{1}{1+\text{DSO目标跟踪误差}}
$$

#### VPP 偏好区间奖励里的响应有效性

VPP dispatch reward 的偏好区间奖励也有一个“响应有效性门控”：

$$
\text{偏好区间奖励}
=
0.50
\times
\text{是否落在偏好区间}
\times
\text{DSO引导强度}
\times
\text{边界收缩程度}
\times
\text{响应有效性}
$$

如果 envelope 里显式给了有效响应得分，就用它。但当前 envelope policy 没有写入这个字段，因此使用 fallback：

$$
\text{响应有效性}
=
\operatorname{clip}
\left(
1
-
\frac{\text{投影偏差}}{\text{硬约束区间宽度}},
0,
1
\right)
$$

含义：

| 情况 | 响应有效性 |
|---|---|
| VPP 动作没有被投影，或者投影很小 | 接近 1 |
| VPP 动作被明显投影 | 下降 |
| 投影偏差大于硬约束区间宽度 | 被裁剪到 0 |

### 0A.6 什么是“缩放因子”

缩放因子不是 reward 权重，也不是可学习参数。它是数值归一化用的分母。

对某个成本项：

$$
\text{缩放成本}
=
\min\left(
10,\,
\frac{\text{加权成本}}{\text{缩放因子}}
\right)
$$

直观理解：

| 加权成本和缩放因子的关系 | 缩放后贡献 |
|---|---:|
| 加权成本 = 0 | 0 |
| 加权成本 = 0.1 × 缩放因子 | 0.1 |
| 加权成本 = 1 × 缩放因子 | 1 |
| 加权成本 = 5 × 缩放因子 | 5 |
| 加权成本 ≥ 10 × 缩放因子 | 10，触发裁剪上限 |

所以缩放因子的作用是：

1. 把不同单位的成本变成大致可比的无量纲数。
2. 避免某个极端值让 critic target 爆炸。
3. 保留“越大越差”的方向，但限制单项最大影响。

当前默认缩放因子的含义：

| 成本项 | 缩放因子 | 为什么这样设 |
|---|---:|---|
| 运行成本 | 1000.0 | 运行成本由外部电网功率乘电价得到，天然可能是几十到几百，给较大分母防止经济项过强。 |
| 目标跟踪误差惩罚 | 10.0 | 已经是 100 × MW误差平方，给 10 让中等偏差产生可见但不爆炸的惩罚。 |
| 动作投影惩罚 | 10.0 | 投影惩罚包含 250 × MW距离平方，还加投影次数，分母 10 控制其训练量级。 |
| 舒适度惩罚 | 100.0 | HVAC comfort 原始值可能很大，用 100 降低量级。旧实验仍显示它过强。 |
| SOC惩罚 | 100.0 | 储能和 EV 的 SOC 缺口平方惩罚可到几百，使用 100 归一化。 |
| 安全违规数量惩罚 | 1.0 | 原始值已经是违规数量 × 50，安全违规应强烈保留。 |
| AC违规幅值惩罚 | 1.0 | 幅值已经是后验违规严重程度，直接保留。 |
| 电压越限惩罚 | 1.0 | 电压原始惩罚已用 10000 × p.u.偏差平方放大，直接保留。 |
| 线路过载惩罚 | 100.0 | 线路 loading 超限是百分比，平方后可能大，用 100 降尺度。 |
| 变压器过载惩罚 | 100.0 | 同线路过载。 |
| 潮流失败惩罚 | 1000.0 | 潮流失败原始惩罚是 1000，除以 1000 后变成 1。 |
| 边界宽度惩罚 | 1.0 | 本身是 0 到 1 附近的比例。 |
| 边界平滑惩罚 | 1.0 | 本身是 MW 级变化量，通常较小。 |

### 0A.7 每个 DSO 原始成本到底来自哪里

| 原始成本 | 直接来源 | 数学形式 | 物理含义 |
|---|---|---|---|
| 运行成本 | 外部电网交换功率或 VPP DER 成本 | 隐私模式下为 \( |\text{外部电网功率}| \times \text{电价} \) | DSO 为维持系统供需或采购电力付出的代理成本 |
| 目标跟踪误差惩罚 | simulator 对所有 VPP 最终响应求偏差 | \(100 \times \text{DSO目标跟踪误差}^2\) | VPP 最终是否按 DSO 指令执行 |
| 动作投影惩罚 | FR/DOE、本地边界、AC-aware、AC certificate 投影记录 | \(250 \times \text{综合投影距离}^2 + 2 \times \text{投影次数}\) | 策略原始动作有多依赖安全壳修正 |
| 舒适度惩罚 | HVAC 室温模型 | \(a \times \text{室温误差}^2 + b \times |\text{室温误差}| + 100 \times \text{硬越界}^2\) | 空调/柔性负荷是否牺牲用户舒适 |
| SOC惩罚 | 储能和 EVCS | 储能越界平方惩罚 + EV 离站未达目标 SOC 惩罚 | 储能/车辆电量是否违反约束 |
| 安全违规数量惩罚 | AC 后验 constraint report | \(50 \times \text{违规数量}\) | 有多少个电压/线路/变压器/潮流失败违规 |
| AC违规幅值惩罚 | AC 后验违规记录 | \( \sum |\text{违规幅值}| \) | 违规严重程度 |
| 电压越限惩罚 | 母线电压检查 | \( \sum 10000 \times \text{电压越限p.u.}^{2} \) | 电压越上限或下限 |
| 线路过载惩罚 | 线路 loading 检查 | \( \sum 5 \times \text{线路超限百分比}^{2} \) | 线路热稳定/载流越限 |
| 变压器过载惩罚 | 变压器 loading 检查 | \( \sum 5 \times \text{变压器超限百分比}^{2} \) | 变压器过载 |
| 潮流失败惩罚 | pandapower 是否收敛 | 不收敛则 1000，否则 0 | AC 潮流无法求解 |
| 边界宽度惩罚 | DSO envelope 宽度 | 偏好区间宽度 / 硬约束区间宽度，再对 VPP 求平均 | DSO 给的偏好边界是否太宽 |
| 边界平滑惩罚 | 相邻 step 的 DSO 目标变化 | 相邻时刻偏好目标功率差的绝对值，再对 VPP 求平均 | DSO 指令是否跳变过大 |

### 0A.8 设备层成本和惩罚如何进入 DSO reward

#### DER 运行成本

每个 DER 有一个二次成本函数：

$$
\text{DER运行成本}
=
a
\times
\text{当前功率}^{2}
+ b
\times
|\text{当前功率}|
+ c
$$

VPP 运行成本：

$$
\text{VPP运行成本}
=
\sum_{\text{VPP内部DER}}
\text{DER运行成本}
$$

如果 DSO reward 使用非隐私模式，则 DSO 运行成本会取：

$$
\sum_{\text{所有VPP}}
\text{VPP运行成本}
$$

但当前 paper-long 使用隐私代理模式，因此 DSO 运行成本主要是：

$$
|\text{外部电网功率}|
\times
\text{电价}
$$

VPP dispatch 的私有利润代理仍会使用自己的 DER 运行成本。

#### HVAC 舒适度惩罚

每个 HVAC 设备：

$$
\text{室温误差}
=
\text{室内温度}
-
\text{设定温度}
$$

如果室温在允许区间外：

$$
\text{硬越界}
=
\max(0,\text{最低舒适温度}-\text{室内温度})
+
\max(0,\text{室内温度}-\text{最高舒适温度})
$$

HVAC 舒适度惩罚：

$$
\text{HVAC舒适度惩罚}
=
a
\times
\text{室温误差}^{2}
+ b
\times
|\text{室温误差}|
+ 100
\times
\text{硬越界}^{2}
$$

VPP 舒适度惩罚只统计 HVAC：

$$
\text{VPP舒适度惩罚}
=
\sum_{\text{VPP内部HVAC}}
\text{HVAC舒适度惩罚}
$$

#### 储能 SOC 惩罚

如果储能 SOC 低于下限：

$$
\text{低SOC惩罚}
=
1000
\times
(\text{SOC下限}-\text{当前SOC})^{2}
$$

如果储能 SOC 高于上限：

$$
\text{高SOC惩罚}
=
1000
\times
(\text{当前SOC}-\text{SOC上限})^{2}
$$

储能 SOC 惩罚为二者相加。

#### EVCS 未满足 SOC 惩罚

EV 只在离站时刻检查是否达到目标 SOC。

如果当前时刻不是离站时刻：

$$
\text{EV未满足SOC惩罚}=0
$$

如果到达离站时刻：

$$
\text{EV未满足SOC惩罚}
=
500
\times
\max(0,\text{目标SOC}-\text{当前SOC})^{2}
$$

EVCS 惩罚是站内所有 EV 的未满足 SOC 惩罚求和。

### 0A.9 电力系统安全原始惩罚如何计算

#### 电压越限

如果母线电压低于下限：

$$
\text{电压越限幅值}
=
\text{电压下限}
-
\text{实际电压}
$$

如果母线电压高于上限：

$$
\text{电压越限幅值}
=
\text{实际电压}
-
\text{电压上限}
$$

电压原始惩罚：

$$
\text{电压越限原始惩罚}
=
\sum_{\text{越限母线}}
10000
\times
\text{电压越限幅值}^{2}
$$

注意：电压幅值单位是 p.u.。例如电压上限 1.06，实际 1.07，则幅值为 0.01 p.u.。

#### 线路过载

如果线路 loading 超过限制：

$$
\text{线路超限百分比}
=
\text{实际loading百分比}
-
\text{线路loading限制百分比}
$$

线路原始惩罚：

$$
\text{线路过载原始惩罚}
=
\sum_{\text{过载线路}}
5
\times
\text{线路超限百分比}^{2}
$$

例如限制是 100%，实际是 105%，则超限百分比是 5。

#### 变压器过载

同线路：

$$
\text{变压器过载原始惩罚}
=
\sum_{\text{过载变压器}}
5
\times
\text{变压器超限百分比}^{2}
$$

#### 潮流失败

如果 pandapower AC 潮流不收敛：

$$
\text{潮流失败原始惩罚}=1000
$$

否则：

$$
\text{潮流失败原始惩罚}=0
$$

### 0A.10 投影距离和安全壳到底在惩罚什么

投影距离不是电网安全本身，而是“动作被修正了多少”。

项目里有四类投影距离：

| 投影距离 | 说明 |
|---|---|
| 原始动作投影距离 | RL 或外部动作在进入执行前已经被裁剪的距离 |
| 本地边界投影距离 | 目标功率被 VPP/DER 本地上下限或 FR/DOE 可行域裁剪的距离 |
| AC-aware 边界投影距离 | 目标功率被 AC-aware DOE 进一步收缩的距离 |
| AC certificate 修复距离 | 候选 dispatch 经 AC 潮流试算后，为通过安全检查而回退/修复的距离 |

综合投影距离：

$$
\text{综合投影距离}
=
\max\left(
\text{原始动作投影距离},
\text{本地边界投影距离}
+ \text{AC-aware边界投影距离}
+ \text{AC证书修复距离}
\right)
$$

DSO 动作投影原始惩罚：

$$
\text{动作投影原始惩罚}
=
250
\times
\text{综合投影距离}^{2}
+ 2
\times
\text{投影次数}
$$

学习层安全壳惩罚：

$$
\text{安全壳惩罚}
=
5
\times
\text{安全壳介入量}
+ 10
\times
\text{安全壳介入量}^{2}
$$

二者区别：

| 项 | 用在哪里 | 作用 |
|---|---|---|
| 动作投影惩罚 | DSO 原始成本，进入 DSO 环境奖励 | 作为 DSO 成本的一部分，记录原始动作和可执行动作之间的差距 |
| 安全壳惩罚 | 学习层，额外从 DSO/dispatch 训练奖励中扣除 | 专门让 actor 不要依赖安全壳兜底 |

### 0A.11 边界宽度、边界平滑、引导强度分别是什么

#### 边界宽度

DSO 对每个 VPP 会给两个区间：

| 区间 | 含义 |
|---|---|
| 硬约束区间 | VPP/DER 物理上可行的功率上下限 |
| 偏好区间 | DSO 更希望 VPP 落入的功率区间 |

边界宽度比例：

$$
\text{边界宽度比例}
=
\frac{
\text{偏好区间宽度}
}{
\text{硬约束区间宽度}
}
$$

DSO reward 中使用所有 VPP 的平均值：

$$
\text{边界宽度原始惩罚}
=
\operatorname{mean}_{\text{所有VPP}}
\left(
\frac{
\text{偏好区间宽度}
}{
\text{硬约束区间宽度}
}
\right)
$$

含义：

- 比例越大，DSO 给的偏好区间越宽，指导性越弱。
- 比例越小，DSO 给的偏好区间越窄，指导性更强，但可能过度约束 VPP。

#### 边界平滑

对每个 VPP：

$$
\text{边界平滑变化量}
=
\left|
\text{当前step偏好目标功率}
-
\text{上一次该VPP偏好目标功率}
\right|
$$

DSO reward 中使用所有 VPP 的平均值：

$$
\text{边界平滑原始惩罚}
=
\operatorname{mean}_{\text{所有有历史目标的VPP}}
\text{边界平滑变化量}
$$

含义：

- 这个值越大，说明 DSO 对 VPP 的指令跳变越大。
- 指令跳变大可能导致 VPP 难以跟踪，也可能让训练更不稳定。

#### DSO 引导强度

在 sensitivity-attention DSO actor 中，网络会输出一个 0 到 1 之间的引导强度。

$$
\text{DSO引导强度}
\in
[0,1]
$$

含义：

| 引导强度 | 解释 |
|---:|---|
| 接近 0 | DSO 对该 VPP 的偏好引导很弱 |
| 接近 1 | DSO 对该 VPP 的偏好引导很强 |

它进入 VPP dispatch 的偏好区间奖励：

$$
\text{偏好区间奖励}
\propto
\text{DSO引导强度}
$$

但当前 DSO reward 自身只是记录平均引导强度，没有直接把它作为成本项扣分。

## 0B. 全文术语字典：每个词到底是什么意思

这一节是查词用的。后文出现的关键术语都按同一格式解释：

- 中文含义
- 数学形式
- 单位或范围
- 属于哪个 reward
- 影响哪个 actor/critic
- CSV/代码字段

### 0B.1 角色、actor、critic 相关术语

| 术语 | 中文解释 | 数学或训练含义 | 属于谁 | 影响谁 |
|---|---|---|---|---|
| DSO | 配电网运营方，全局调度者 | 给每个 VPP 下发可运行边界、偏好目标或安全引导 | 系统级角色 | DSO actor、DSO critic/value head |
| VPP | 虚拟电厂或聚合商 | 聚合 PV、储能、EVCS、柔性负荷、HVAC、微燃机等 DER | 聚合商角色 | VPP dispatch actor、VPP portfolio actor |
| DER | 分布式能源资源 | VPP 内部真实执行功率的设备 | 环境组件 | 间接影响 VPP reward 和 DSO reward |
| actor | 策略网络 | 输入 observation，输出动作 | DSO/VPP 各自拥有 | 被 reward 通过 policy loss 更新 |
| critic | 价值网络或 Q 网络 | 估计某个动作/状态好不好 | CTDE 或 role-head critic | 用 reward 训练，反过来指导 actor |
| value head | 多角色 value 输出头 | 一个网络输出 DSO、dispatch、portfolio 不同角色的价值估计 | HAPPO/HATRPO 等 | 各角色 advantage 计算 |
| DSO actor | DSO 全局策略 | 输出 DSO envelope 的中心、宽度、方向、引导强度 | `dso_global_guidance` | DSO reward |
| VPP dispatch actor | VPP 快周期策略 | 输出每个 VPP 或 DER 的实时功率调度动作 | `{vpp_id}_dispatch` | VPP dispatch reward |
| VPP portfolio actor | VPP 慢周期组合策略 | 输出 keep、reweight、propose_membership_change 等组合动作 | `{vpp_id}_portfolio` | VPP portfolio reward |
| CTDE | 集中训练、分散执行 | 训练时 critic 可看更多全局信息，执行时 actor 用本地信息 | 算法框架 | 影响 critic 输入，不直接改变 reward 公式 |
| role reward vector | 角色奖励向量 | 把 DSO、各 VPP dispatch、各 VPP portfolio 的奖励排成向量 | 训练器内部 | 多头 critic/value 分别学习 |

### 0B.2 DSO reward 总公式里的术语

DSO 总公式是：

$$
\text{DSO环境奖励}
=
\text{可行奖励}
+ \text{跟踪奖励}
+ \text{有效响应奖励}
- 0.05
\times
\text{缩放总成本}
$$

| 术语 | 中文解释 | 数学形式 | 范围/单位 | 属于谁 | 影响谁 | 字段 |
|---|---|---|---|---|---|---|
| DSO环境奖励 | DSO 在环境中得到的原始角色奖励 | 上式 | 通常约 0 到 2.3，取决于成本和安全 | DSO | DSO actor/critic | `dso_reward` |
| 可行奖励 | 当前 step 最终 AC 潮流是否安全 | 安全则 1，否则 0 | 0 或 1 | DSO | DSO actor/critic | `feasibility_bonus` |
| 跟踪奖励 | VPP 是否按 DSO 目标响应 | \(0.25/(1+\text{DSO目标跟踪误差})\) | 0 到 0.25 | DSO | DSO actor/critic | `tracking_bonus` |
| 有效响应奖励 | 当前代码中等价于“跟踪误差越小越好”的奖励 | \(1/(1+\text{DSO目标跟踪误差})\) | 0 到 1 | DSO | DSO actor/critic | `effective_response_bonus` |
| 缩放总成本 | 所有缩放成本相加 | \(\sum \text{某项缩放成本}\) | 非负，一般 0 到十几 | DSO | DSO actor/critic | `scaled_total_cost` |
| DSO成本缩放系数 | 缩放总成本在 DSO reward 中的系数 | 0.05 | 固定超参数 | DSO | DSO actor/critic | `dso_reward_cost_scale` |
| 加权总成本 | 所有加权成本相加，主要用于日志和审计 | \(\sum \text{某项加权成本}\) | 非负，可能很大 | DSO | 间接影响 reward，直接不进最终公式 | `total_cost` |
| 原始目标奖励 | 未缩放成本的负值 | \(-\text{加权总成本}\) | 负数 | DSO | 日志/审计 | `raw_objective_reward` |
| 缩放目标奖励 | 缩放总成本的负值 | \(-\text{缩放总成本}\) | 负数 | DSO | 日志/审计 | `scaled_objective_reward` |

### 0B.3 “目标跟踪误差”完整解释

目标跟踪误差回答的问题是：

$$
\text{VPP最终实际做出来的功率，和DSO希望它做的目标功率差多少？}
$$

它不是神经网络 loss，也不是预测误差，而是环境执行后的物理响应误差。

#### DSO 全局目标跟踪误差

计算流程：

1. DSO 先给每个 VPP 一个目标功率或偏好目标。
2. 目标功率先经过 FR/DOE、本地上下限、AC-aware DOE。
3. 候选 dispatch 再经过 AC certificate 检查和必要修复。
4. 修复后的 dispatch 写入 DER。
5. 对每个 VPP，求最终实际聚合功率和目标功率之间的绝对差。
6. 对所有 VPP 求和。

数学形式：

$$
\text{DSO目标跟踪误差}
=
\sum_{i=1}^{N_{\text{VPP}}}
\left|
\text{第}i\text{个VPP最终实际聚合功率}
-
\text{第}i\text{个VPP目标功率}
\right|
$$

单位：

$$
\text{MW}
$$

进入 DSO reward 的方式：

$$
\text{目标跟踪误差原始惩罚}
=
100
\times
\text{DSO目标跟踪误差}^{2}
$$

$$
\text{跟踪奖励}
=
\frac{0.25}{1+\text{DSO目标跟踪误差}}
$$

$$
\text{有效响应奖励}
=
\frac{1}{1+\text{DSO目标跟踪误差}}
$$

对应字段：

| 字段 | 含义 |
|---|---|
| `target_tracking_error_penalty` | 加权后的目标跟踪误差惩罚 |
| `raw_target_tracking_error_penalty` | 未乘权重的目标跟踪误差惩罚 |
| `scaled_target_tracking_error_penalty` | 归一化后的目标跟踪误差惩罚 |
| `tracking_bonus` | 跟踪奖励 |
| `effective_response_bonus` | 当前 fallback 下的有效响应奖励 |

数值例子：

如果有 3 个 VPP：

| VPP | DSO目标功率 | 最终实际功率 | 绝对误差 |
|---|---:|---:|---:|
| VPP 1 | 0.20 MW | 0.18 MW | 0.02 MW |
| VPP 2 | -0.10 MW | -0.16 MW | 0.06 MW |
| VPP 3 | 0.05 MW | 0.05 MW | 0.00 MW |

则：

$$
\text{DSO目标跟踪误差}
=
0.02+0.06+0.00
=
0.08\text{ MW}
$$

$$
\text{目标跟踪误差原始惩罚}
=
100
\times
0.08^2
=
0.64
$$

$$
\text{跟踪奖励}
=
\frac{0.25}{1+0.08}
\approx
0.231
$$

$$
\text{有效响应奖励}
=
\frac{1}{1+0.08}
\approx
0.926
$$

#### VPP 局部目标跟踪偏差

VPP dispatch reward 里还有一个局部跟踪偏差：

$$
\text{VPP局部目标跟踪偏差}
=
\left|
\text{本VPP当前实际聚合功率}
-
\text{本VPP收到的目标功率}
\right|
$$

进入 VPP dispatch reward：

$$
\text{VPP目标跟踪惩罚}
=
25
\times
\text{VPP局部目标跟踪偏差}^{2}
$$

对应字段：

| 字段 | 含义 |
|---|---|
| `target_tracking_penalty` | VPP dispatch reward 里的局部跟踪惩罚 |
| `delivered_p_mw` | 本 VPP 当前实际聚合功率 |
| `target_p_mw` | 本 VPP 收到的目标功率 |

注意：

- DSO 的目标跟踪误差是所有 VPP 汇总。
- VPP 的目标跟踪偏差是单个 VPP 自己的履约偏差。
- 二者方向一致，但不是同一个字段。

### 0B.4 “动作投影惩罚”完整解释

动作投影惩罚回答的问题是：

$$
\text{RL或DSO给出的原始动作，被安全边界改了多少？}
$$

它不是电压越限惩罚，也不是线路过载惩罚。它惩罚的是“动作不可执行、必须被修正”。

#### 为什么需要动作投影

RL actor 输出的动作可能有这些问题：

| 问题 | 例子 |
|---|---|
| 超出 DER 本地上下限 | 储能最多放电 0.05 MW，actor 要求 0.10 MW |
| 超出 VPP 可行域 | VPP 总可调区间是 [-0.2, 0.3] MW，actor 要求 0.6 MW |
| 违反 DSO FR/DOE 边界 | DSO 只允许该位置在安全区间内运行，actor 给出区间外目标 |
| AC 潮流后不安全 | 动作本地可行，但写入网络后导致电压或线路问题 |

项目不会直接执行这些危险动作，而是投影到安全或可行范围内。

#### 四类投影距离

| 投影距离 | 数学含义 | 来源 | 字段 |
|---|---|---|---|
| 原始动作投影距离 | 原始动作被预裁剪的距离 | action payload 或 decoder 层 | `action_projection_gap_mw` 的一部分 |
| 本地边界投影距离 | 从原始目标到本地 FR/DER/VPP 可行目标的距离 | FR/DOE、本地上下限 | `local_bounds_projection_gap_mw` |
| AC-aware 边界投影距离 | 从本地可行目标到 AC-aware DOE 目标的距离 | 灵敏度安全边界 | `ac_aware_projection_gap_mw` |
| AC certificate 修复距离 | AC 潮流证书把候选 dispatch 回退/修复的距离 | post-AC 安全检查 | `ac_certified_projection_gap_mw` |

#### 综合投影距离

DSO 成本使用：

$$
\text{综合投影距离}
=
\max\left(
\text{总动作投影距离},
\text{本地边界投影距离}
+ \text{AC-aware边界投影距离}
+ \text{AC证书修复距离}
\right)
$$

其中“总动作投影距离”已经包含若干投影阶段的合计记录。

#### 动作投影原始惩罚

$$
\text{动作投影原始惩罚}
=
250
\times
\text{综合投影距离}^{2}
+ 2
\times
\text{投影次数}
$$

这里有两个部分：

| 部分 | 含义 |
|---|---|
| \(250 \times \text{综合投影距离}^{2}\) | 投影距离越大，惩罚按平方变大 |
| \(2 \times \text{投影次数}\) | 只要发生投影，即使距离很小，也有固定惩罚 |

#### 动作投影加权惩罚

base/paper-long 旧配置：

$$
\text{动作投影加权惩罚}
=
2.0
\times
\text{动作投影原始惩罚}
$$

sensitivity-v1 配置：

$$
\text{动作投影加权惩罚}
=
5.0
\times
\text{动作投影原始惩罚}
$$

#### 动作投影缩放惩罚

动作投影缩放因子是 10：

$$
\text{动作投影缩放惩罚}
=
\min\left(
10,\,
\frac{\text{动作投影加权惩罚}}{10}
\right)
$$

#### 数值例子

假设某一步：

$$
\text{综合投影距离}=0.04\text{ MW}
$$

投影次数：

$$
\text{投影次数}=1
$$

则：

$$
\text{动作投影原始惩罚}
=
250
\times
0.04^2
+ 2
\times
1
=
0.4 + 2
=
2.4
$$

如果使用 sensitivity-v1 权重 5：

$$
\text{动作投影加权惩罚}
=
5
\times
2.4
=
12
$$

缩放后：

$$
\text{动作投影缩放惩罚}
=
\frac{12}{10}
=
1.2
$$

再进入 DSO reward 时，会通过缩放总成本被乘以 0.05：

$$
\text{对DSO奖励的直接扣减}
=
0.05
\times
1.2
=
0.06
$$

对应字段：

| 字段 | 含义 |
|---|---|
| `raw_action_projection_penalty` | 动作投影原始惩罚 |
| `action_projection_penalty` | 动作投影加权惩罚 |
| `scaled_action_projection_penalty` | 动作投影缩放惩罚 |
| `action_projection_gap_mw` | 总动作投影距离记录 |
| `local_bounds_projection_gap_mw` | 本地边界投影距离 |
| `ac_aware_projection_gap_mw` | AC-aware DOE 投影距离 |
| `ac_certified_projection_gap_mw` | AC certificate 修复距离 |
| `action_projection_count` | 投影次数 |

#### 动作投影惩罚和安全壳惩罚的区别

| 项 | 动作投影惩罚 | 安全壳惩罚 |
|---|---|---|
| 位置 | DSO 环境 reward 的成本项 | 学习层额外扣分 |
| 公式 | \(250d^2+2n\)，再乘权重和缩放 | \(5d+10d^2\) |
| 作用对象 | 主要影响 DSO reward | 同时扣 DSO 训练奖励和 VPP dispatch 训练奖励 |
| 是否进入 `total_cost` | 是 | 否 |
| 是否专门防止 actor 依赖安全壳 | 间接 | 直接 |

### 0B.5 DSO 成本项逐项字典

| 字段 | 中文名 | 最详细解释 | 数学形式 | 单位/范围 | Actor/Critic |
|---|---|---|---|---|---|
| `operation_cost` | 运行成本 | DSO 当前 step 的运行/采购代理成本。隐私模式下不看 VPP 私有成本，只用外部电网交换功率乘电价。 | \( |\text{外部电网功率}| \times \text{电价} \) | 经济代理量，非负 | DSO |
| `target_tracking_error_penalty` | 目标跟踪误差惩罚 | VPP 最终执行功率偏离 DSO 目标的系统级惩罚。 | \(100 \times \text{DSO目标跟踪误差}^2\)，再乘权重 | 非负 | DSO |
| `action_projection_penalty` | 动作投影惩罚 | 原始动作被边界、DOE 或 AC certificate 修正的惩罚。 | \(w(250d^2+2n)\) | 非负 | DSO |
| `comfort_violation_penalty` | 舒适度惩罚 | HVAC 室温偏离设定温度或舒适边界的惩罚。 | \(w\sum(ae^2+b|e|+100h^2)\) | 非负 | DSO/VPP dispatch 子项 |
| `soc_violation_penalty` | SOC 惩罚 | 储能 SOC 越界和 EV 离站 SOC 未达标的惩罚。 | 储能 \(1000\Delta^2\)，EV \(500\Delta^2\) | 非负 | DSO/VPP dispatch 子项 |
| `security_violation_count_penalty` | 安全违规数量惩罚 | AC 后验检查发现多少个违规。 | \(50 \times \text{违规数量}\)，再乘权重 | 非负整数倍 | DSO |
| `post_ac_violation_magnitude_penalty` | AC 违规幅值惩罚 | 所有 AC 后验违规幅值的绝对值求和。 | \(\sum|\text{违规幅值}|\) | 非负 | DSO |
| `voltage_violation_penalty` | 电压越限惩罚 | 母线电压低于/高于限制的平方惩罚。 | \(\sum 10000 \times \Delta V^2\)，再乘权重 | p.u. 派生惩罚 | DSO |
| `line_overload_penalty` | 线路过载惩罚 | 线路 loading 超过限制的平方惩罚。 | \(\sum 5 \times \Delta L^2\)，再乘权重 | loading 百分比派生惩罚 | DSO |
| `transformer_overload_penalty` | 变压器过载惩罚 | 变压器 loading 超过限制的平方惩罚。 | \(\sum 5 \times \Delta T^2\)，再乘权重 | loading 百分比派生惩罚 | DSO |
| `powerflow_penalty` | 潮流失败惩罚 | pandapower AC 潮流不收敛时给固定惩罚。 | 不收敛为 1000，否则 0，再乘权重 | 0 或正数 | DSO |
| `envelope_width_penalty` | 边界宽度惩罚 | DSO 给出的偏好区间占硬约束区间比例。比例越大，指令越松。 | \(\operatorname{mean}(\text{偏好宽度}/\text{硬约束宽度})\) | 通常 0 到 1 | DSO |
| `smoothness_penalty` | 边界平滑惩罚 | 相邻 step 的 DSO 偏好目标跳变量。 | \(\operatorname{mean}(|\text{当前目标}-\text{上次目标}|)\) | MW | DSO |

### 0B.6 DSO 奖励项逐项字典

| 字段 | 中文名 | 详细解释 | 数学形式 | 范围 | Actor/Critic |
|---|---|---|---|---|---|
| `feasibility_bonus` | 可行奖励 | 最终执行动作经过 AC 潮流后安全，则奖励 1。 | 安全为 1，否则 0 | 0 或 1 | DSO |
| `tracking_bonus` | 跟踪奖励 | DSO 目标被 VPP 跟踪得越好，该奖励越高。 | \(0.25/(1+\text{DSO目标跟踪误差})\) | 0 到 0.25 | DSO |
| `effective_response_bonus` | 有效响应奖励 | 当前代码里等价于跟踪误差的更大权重奖励。 | \(1/(1+\text{DSO目标跟踪误差})\) | 0 到 1 | DSO |
| `dso_reward` | DSO 最终环境奖励 | DSO 的角色 reward，进入训练前还可能扣安全壳惩罚。 | 可行 + 跟踪 + 有效响应 - 0.05 × 缩放总成本 | 可正可负 | DSO |
| `scaled_reward` | 缩放奖励 | 当前等于 `dso_reward`，保留兼容字段。 | 同 `dso_reward` | 可正可负 | DSO |
| `reward` | 单智能体兼容奖励 | 当前等于 `dso_reward`，用于旧接口。 | 同 `dso_reward` | 可正可负 | DSO/旧接口 |

### 0B.7 VPP dispatch 术语逐项字典

VPP dispatch 总公式：

$$
\begin{aligned}
\text{VPP调度奖励}
=&
0.02 \times \text{私有利润代理}
+ \text{偏好区间奖励}
- 25\times\text{局部目标跟踪偏差}^{2}
\\
&-
\left(5\times\text{投影偏差}+10\times\text{投影偏差}^{2}\right)
-0.001\times\text{缩放舒适/SOC惩罚}
\end{aligned}
$$

| 字段/术语 | 中文名 | 详细解释 | 数学形式 | 范围/单位 | Actor/Critic |
|---|---|---|---|---|---|
| `vpp_dispatch_reward` | VPP 调度环境奖励 | 单个 VPP 快周期调度 agent 的环境 reward。 | 上式 | 可正可负 | VPP dispatch |
| `delivered_p_mw` | 实际交付功率 | VPP 当前所有 DER 功率求和。 | \(\sum_{\text{DER}}P_{\text{DER}}\) | MW | VPP dispatch |
| `target_p_mw` | 目标功率 | DSO envelope 或投影后目标。 | 来自 projected target 或 preferred target | MW | VPP dispatch |
| `flex_span_mw` | 可调功率区间宽度 | VPP 当前最大可行功率减最小可行功率。 | \(P_{\max}-P_{\min}\) | MW | VPP dispatch/portfolio |
| `private_profit_proxy` | 私有利润代理 | VPP 的局部收益代理，不含 raw DSO reward。 | 市场收入 + 服务收入 + 可用容量收入 - DER成本 | 经济代理量 | VPP dispatch/portfolio |
| `energy_market_revenue` | 电能市场收入 | VPP 实际功率按电价结算的收入或代理收益。 | 电价 × 实际功率 × 步长 | 经济代理量 | VPP dispatch |
| `flexibility_service_payment` | 灵活性服务收入 | 按 DSO 服务方向计算的灵活性响应收入。 | 电价 × 服务功率数量 × 步长 | 经济代理量 | VPP dispatch |
| `availability_payment` | 可用容量收入 | VPP 只要提供可调区间就有的容量类收入。 | 0.02 × 电价 × 可调区间宽度 × 步长 | 经济代理量 | VPP dispatch |
| `der_operation_cost` | DER运行成本 | VPP 内 DER 的运行成本求和。 | \(\sum(aP^2+b|P|+c)\) | 经济代理量 | VPP dispatch |
| `target_tracking_penalty` | VPP 局部跟踪惩罚 | 本 VPP 实际功率偏离目标功率。 | \(25\times|\text{实际功率}-\text{目标功率}|^2\) | 非负 | VPP dispatch |
| `envelope_projection_penalty` | VPP 边界投影惩罚 | 本 VPP 动作被本地边界投影的惩罚。 | \(5d+10d^2\) | 非负 | VPP dispatch |
| `raw_comfort_soc_penalty` | 原始舒适/SOC惩罚 | 本 VPP 的 comfort penalty + SOC penalty。 | comfort + SOC | 非负 | VPP dispatch |
| `scaled_comfort_soc_penalty` | 缩放舒适/SOC惩罚 | 把 comfort/SOC 除以 100 并裁剪到 5。 | \(\min(5,\max(0,\text{raw})/100)\) | 0 到 5 | VPP dispatch |
| `preferred_region_bonus` | 偏好区间奖励 | VPP 落在 DSO 偏好区间内并响应有效时得到奖励。 | 0.50 × inside × strength × width_gate × effectiveness | 0 到 0.50 | VPP dispatch |
| `preferred_inside_range` | 是否落在偏好区间 | 实际功率是否在 DSO 偏好上下限内。 | 是为 1，否则 0 | 0 或 1 | VPP dispatch |
| `preferred_bonus_lambda_gate` | DSO 引导强度门控 | DSO actor 输出的引导强度。 | clip 到 0 到 1 | 0 到 1 | VPP dispatch |
| `preferred_bonus_width_gate` | 边界收缩门控 | 偏好区间越窄，说明 DSO 越明确，奖励门控越高。 | \(1-\text{偏好宽度}/\text{硬约束宽度}\) | 0 到 1 | VPP dispatch |
| `preferred_bonus_effectiveness_gate` | 响应有效性门控 | 如果动作被投影很多，该门控下降。 | \(1-\text{投影偏差}/\text{硬约束宽度}\) 后裁剪 | 0 到 1 | VPP dispatch |

### 0B.8 VPP portfolio 术语逐项字典

VPP portfolio 总公式：

$$
\text{VPP组合奖励}
=
\text{长期利润代理}
+ \text{局部DSO对齐奖励}
+ \text{可靠性奖励}
- \text{切换成本}
- \text{交付风险惩罚}
$$

| 字段/术语 | 中文名 | 详细解释 | 数学形式 | 范围/单位 | Actor/Critic |
|---|---|---|---|---|---|
| `vpp_portfolio_reward` | VPP 组合奖励 | 慢周期 portfolio agent 的环境 reward。 | 上式 | 可正可负 | VPP portfolio |
| `long_horizon_profit_proxy` | 长期利润代理 | dispatch 私有利润代理的慢周期版本。 | \(0.10\times\text{私有利润代理}\) | 经济代理量 | VPP portfolio |
| `localized_dso_alignment_reward` | 局部 DSO 对齐奖励 | 不直接共享全局 DSO reward，而是局部化为偏好、可行、可用容量和可靠性。 | 0.35×偏好得分 + 0.25×可行 + 0.15×可用容量质量 + 0.10×可靠性 - 0.001×网络惩罚 | 可正可负 | VPP portfolio |
| `reliability_bonus` | 可靠性奖励 | VPP 实际交付越接近目标，可靠性越高。 | \(0.50\max(0,1-\text{跟踪偏差}/\text{可调区间宽度})\) | 0 到 0.50 | VPP portfolio |
| `availability_quality` | 可用容量质量 | 可调区间是否足够宽。 | \(\min(1,\text{可调区间宽度}/0.50)\) | 0 到 1 | VPP portfolio |
| `switching_cost` | 切换成本 | portfolio 动作改变组合会付成本。 | keep=0, reweight=0.02, membership change=0.08 | 0/0.02/0.08 | VPP portfolio |
| `delivery_risk_penalty` | 交付风险惩罚 | 投影和跟踪偏差越大，说明组合建议越可能不可交付。 | \(0.50\times\text{投影偏差}+0.20\times\text{跟踪偏差}\) | 非负 | VPP portfolio |
| `network_penalty` | 网络惩罚 | 电压、线路、变压器、潮流失败、AC违规幅值的局部安全惩罚汇总。 | 安全项求和 | 非负 | VPP portfolio |

### 0B.9 安全与潮流术语逐项字典

| 术语/字段 | 中文名 | 详细解释 | 数学形式/判定 | 范围/单位 | 影响 |
|---|---|---|---|---|---|
| `post_ac_violation_count` | AC 后验违规数量 | 动作最终写入 pandapower 并跑 AC 潮流后，发现的违规条数。 | 违规记录数量 | 非负整数 | DSO reward、安全评估 |
| `post_ac_voltage_violation_count` | AC 后验电压违规数量 | 电压越上限或下限的母线数量。 | 电压违规记录数量 | 非负整数 | DSO reward、安全评估 |
| `post_ac_line_overload_count` | AC 后验线路过载数量 | loading 超限线路数量。 | 线路过载记录数量 | 非负整数 | DSO reward、安全评估 |
| `post_ac_trafo_overload_count` | AC 后验变压器过载数量 | loading 超限变压器数量。 | 变压器过载记录数量 | 非负整数 | DSO reward、安全评估 |
| `post_ac_powerflow_failed` | AC 潮流失败 | pandapower 是否不收敛。 | 失败为 1，否则 0 | 0 或 1 | DSO reward、安全评估 |
| `post_ac_violation_magnitude` | AC 后验违规幅值 | 所有违规幅值绝对值之和。 | \(\sum|\text{违规幅值}|\) | 非负 | DSO reward、安全评估 |
| `post_ac_security_penalty` | AC 后验安全惩罚 | 安全违规数量、幅值、电压、线路、变压器、潮流失败惩罚汇总。 | 安全成本项求和 | 非负 | DSO reward |
| `voltage_limits` | 电压上下限 | 配电网允许的 p.u. 电压范围。 | 例如 [0.93, 1.06] | p.u. | 电压违规判断 |
| `line_loading_limit_percent` | 线路载流限制 | 线路 loading 允许上限。 | 通常 100% | 百分比 | 线路过载判断 |
| `trafo_loading_limit_percent` | 变压器载流限制 | 变压器 loading 允许上限。 | 通常 100% | 百分比 | 变压器过载判断 |
| FR | Feasible Region，可行域 | VPP/DER 在本地物理约束下能实现的 P/Q 区域。 | 上下界或向量边界 | MW/Mvar | 动作投影 |
| DOE | Dynamic Operating Envelope，动态运行包络 | DSO 给 VPP 的允许或偏好运行边界。 | 上下界、目标、偏好区间 | MW/Mvar | DSO/VPP reward |
| AC certificate | AC 潮流安全证书 | 候选 dispatch 被 AC 潮流验证，若不安全则修复。 | 安全则接受，否则回退/修复 | 审计过程 | 安全壳、投影惩罚 |

### 0B.10 实验统计表术语逐项字典

| 统计术语 | 中文解释 | 怎么读 |
|---|---|---|
| mean | 平均值 | 全部样本的平均水平 |
| p05 | 5% 分位数 | 95% 的样本比它大，代表偏低水平 |
| p50 | 50% 分位数 | 中位数，比平均值更抗异常值 |
| p95 | 95% 分位数 | 只有 5% 样本比它大，代表偏高水平 |
| min | 最小值 | 最极端低值 |
| max | 最大值 | 最极端高值 |
| abs-cost-share | 成本绝对占比 | 某成本项绝对均值 / 所有 DSO 成本项绝对均值之和 |
| abs-share-vs-learning | 学习信号绝对占比 | 某角色 reward 绝对均值 / 所有角色 reward 绝对均值之和 |
| smoke | 烟测 | 很短的小实验，只验证代码能跑，不代表论文结论 |
| paper-long | 论文长实验 | horizon、seed、场景更完整，用于观察长期收敛和泛化 |
| sensitivity-v1 | 当前基于灵敏度和 attention 的 DSO envelope 策略版本 | 代表修正后的 DSO 安全引导机制 |

### 0B.11 读 reward 文档时的推荐顺序

如果你想真正读懂 reward，不建议从长公式开始。推荐顺序：

1. 先看 `0B.3 目标跟踪误差`，理解“DSO 目标”和“VPP 实际响应”的关系。
2. 再看 `0B.4 动作投影惩罚`，理解“安全壳到底改了什么”。
3. 再看 `0B.5 DSO 成本项逐项字典`，理解每个 DSO 成本来自哪里。
4. 再看 `0B.7 VPP dispatch 术语逐项字典`，理解每个 VPP 为什么有自己的收益和惩罚。
5. 最后看实验统计表，判断哪个项在真实实验里占主导。

最核心的判断逻辑是：

$$
\text{如果总reward不涨}
\neq
\text{算法一定没学}
$$

要拆开看：

$$
\text{DSO奖励}
,\quad
\text{VPP调度奖励}
,\quad
\text{VPP组合奖励}
,\quad
\text{安全壳介入量}
,\quad
\text{AC后验违规}
$$

如果 DSO reward 很高但安全壳介入量不下降，说明策略可能仍在依赖安全壳。

如果 DSO reward 很高、AC 后验违规为 0、安全壳介入量也下降，才更接近健康收敛。

## 1. 总体结构

项目里的 reward 不是一个单一标量，而是按角色拆开：

| 角色 | agent id | actor | critic/value 归属 | 主要目标 |
|---|---|---|---|---|
| DSO 全局引导 | `dso_global_guidance` | DSO actor | DSO critic head 或 DSO value head | 网络安全、动作可行、跟踪、运行成本 |
| VPP 快周期调度 | `{vpp_id}_dispatch` | 每个 VPP dispatch actor | 每个 VPP dispatch critic/value head | VPP 自身收益、跟踪 DSO envelope、少被投影 |
| VPP 慢周期组合 | `{vpp_id}_portfolio` | 每个 VPP portfolio actor | portfolio value head，取决于算法分支 | 慢周期组合/重配建议、可靠性、局部 DSO 对齐 |

环境每一步先生成三类角色奖励：DSO 全局引导奖励、每个 VPP 的快周期调度奖励、每个 VPP 的慢周期组合奖励。

然后训练器再按算法分支构造学习 reward：

### HAPPO / HATRPO 分支

HAPPO 和 HATRPO 会把 DSO、dispatch、portfolio 都放进 role reward vector：

$$
\text{DSO训练奖励}
=
\text{DSO环境奖励}
-
\text{DSO安全壳系数}
\times
\text{安全壳惩罚}
$$

$$
\text{第 } i \text{ 个VPP调度训练奖励}
=
\text{第 } i \text{ 个VPP调度环境奖励}
-
\text{调度安全壳系数}
\times
\text{安全壳惩罚}
$$

$$
\text{第 } i \text{ 个VPP组合训练奖励}
=
\text{第 } i \text{ 个VPP组合环境奖励}
$$

如果不是 portfolio 决策步，并且开启 `portfolio_force_keep_between_decisions`，则：

$$
\text{第 } i \text{ 个VPP组合训练奖励}
=
0
$$

训练日志里每步显示的总 reward 是：

$$
\text{每步日志总奖励}
=
\text{DSO训练奖励}
+ \operatorname{mean}(\text{所有VPP调度训练奖励})
+ \operatorname{mean}(\text{所有VPP组合训练奖励})
$$

真正进入 advantage/value 计算前还会乘：

$$
\text{神经网络实际使用奖励}
=
0.01
\times
\text{角色训练奖励}
$$

### HASAC / MATD3 连续调度分支

当前连续 dispatch 分支主要训练 DSO 和 dispatch：

$$
\text{角色奖励向量}
=
\left[
\text{DSO训练奖励},
\text{第1个VPP调度训练奖励},
\ldots,
\text{第N个VPP调度训练奖励}
\right]
$$

其中：

$$
\text{DSO训练奖励}
=
\text{DSO环境奖励}
-
\text{DSO安全壳系数}
\times
\text{安全壳惩罚}
$$

$$
\text{第 } i \text{ 个VPP调度训练奖励}
=
\text{第 } i \text{ 个VPP调度环境奖励}
-
\text{调度安全壳系数}
\times
\text{安全壳惩罚}
$$

Q target 中使用：

$$
\text{目标Q值}
=
0.01
\times
\text{角色奖励向量}
+
(1-\text{是否终止})
\times
\gamma
\times
\left(
\text{下一步目标Q值}
-
\text{熵项}
\right)
$$

当前默认：

$$
\text{奖励缩放系数}=0.01,\quad
\text{DSO安全壳系数}=1.0,\quad
\text{调度安全壳系数}=1.0
$$

## 2. DSO Reward

### 2.1 归属

| 项目 | 说明 |
|---|---|
| actor | `dso_global_guidance` |
| critic/value | DSO critic head 或 DSO value head |
| 训练目标 | 让 DSO 输出更安全、更少被投影、更能被 VPP 跟踪的 envelope/DOE 指令 |
| 是否直接给 VPP | 不直接给 dispatch/portfolio。VPP dispatch reward 不吃 raw global DSO reward，portfolio 只吃 localized DSO alignment |

### 2.2 原始成本项

DSO 先计算一组原始成本：

$$
\mathcal{C}
=
\{
\text{运行成本},
\text{目标跟踪误差惩罚},
\text{动作投影惩罚},
\text{舒适度惩罚},
\text{SOC惩罚},
\text{安全违规数量惩罚},
\text{AC违规幅值惩罚},
\text{电压越限惩罚},
\text{线路过载惩罚},
\text{变压器过载惩罚},
\text{潮流失败惩罚},
\text{边界宽度惩罚},
\text{边界平滑惩罚}
\}
$$

然后做加权：

$$
\text{某项加权成本}
=
\text{该项权重}
\times
\text{该项原始成本}
$$

再做缩放和裁剪：

$$
\text{某项缩放成本}
=
\min\left(
\text{裁剪上限},\,
\frac{\max(0,\,\text{某项加权成本})}{\text{该项缩放因子}}
\right)
$$

当前默认：

$$
\text{裁剪上限}=10.0
$$

默认 scale：

| 成本项 | scale |
|---|---:|
| `operation_cost` | 1000.0 |
| `target_tracking_error_penalty` | 10.0 |
| `action_projection_penalty` | 10.0 |
| `comfort_violation_penalty` | 100.0 |
| `soc_violation_penalty` | 100.0 |
| `security_violation_count_penalty` | 1.0 |
| `post_ac_violation_magnitude_penalty` | 1.0 |
| `voltage_violation_penalty` | 1.0 |
| `line_overload_penalty` | 100.0 |
| `transformer_overload_penalty` | 100.0 |
| `powerflow_penalty` | 1000.0 |
| `envelope_width_penalty` | 1.0 |
| `smoothness_penalty` | 1.0 |

### 2.3 DSO 各项数学形式

#### 2.3.1 运行成本 `operation_cost`

如果 `reward.privacy_mode = privacy_preserving_proxy`：

$$
\text{运行成本}
=
|\text{外部电网交换功率}|
\times
\text{电价}
$$

其中 `P_ext_grid` 是 pandapower 外部电网交换功率，`price` 是电价。

如果不是隐私代理模式：

$$
\text{运行成本}
=
\sum_{\text{所有VPP}}
\text{VPP运行成本}
$$

当前 paper-long 配置使用：

$$
\text{隐私模式}=\text{电网购电代理成本},\quad
\text{运行成本权重}=1.0
$$

#### 2.3.2 目标跟踪误差 `target_tracking_error_penalty`

$$
\text{目标跟踪误差惩罚}
=
\text{跟踪误差权重}
\times
100
\times
\text{目标跟踪误差}^{2}
$$

当前：

$$
\text{跟踪误差权重}=1.0
$$

含义：VPP 实际响应和 DSO 期望 envelope/target 差得越远，该项越大。

#### 2.3.3 动作投影惩罚 `action_projection_penalty`

先定义：

$$
\begin{aligned}
\text{原始动作投影距离} &= \text{原始动作到最终动作的距离} \\
\text{本地边界投影距离} &= \text{动作被VPP/DER本地上下限修正的距离} \\
\text{AC-aware边界投影距离} &= \text{动作被AC-aware DOE边界修正的距离} \\
\text{AC证书修复距离} &= \text{动作经AC潮流试算后被修复的距离}
\end{aligned}
$$

DSO 成本层面使用：

$$
\text{综合投影距离}
=
\max\left(
\text{原始动作投影距离},
\text{本地边界投影距离}
+ \text{AC-aware边界投影距离}
+ \text{AC证书修复距离}
\right)
$$

原始惩罚：

$$
\text{动作投影原始惩罚}
=
250
\times
\text{综合投影距离}^{2}
+ 2
\times
\text{投影次数}
$$

加权后：

$$
\text{动作投影惩罚}
=
\text{投影权重}
\times
\text{动作投影原始惩罚}
$$

旧 base 配置：

$$
\text{投影权重}=2.0
$$

sensitivity-v1 配置通过 `reward.dso.projection_gap_weight` 覆盖为：

$$
\text{投影权重}=5.0
$$

解释：该项不是 AC 安全本身，而是惩罚“原始动作被本地边界、AC-aware DOE 或 AC certificate 改过”。它让策略尽量少依赖安全壳兜底。

#### 2.3.4 舒适度惩罚 `comfort_violation_penalty`

$$
\text{舒适度惩罚}
=
\text{舒适度权重}
\times
\sum_{\text{所有VPP}}
\text{VPP舒适度偏离惩罚}
$$

当前 base 配置：

$$
\text{舒适度权重}=0.02
$$

解释：主要来自 HVAC/柔性负荷舒适区偏离。旧 long 中这个项曾严重主导 DSO 成本，是 reward 尺度失衡的核心证据之一。

#### 2.3.5 SOC 惩罚 `soc_violation_penalty`

$$
\text{SOC惩罚}
=
\text{SOC权重}
\times
\sum_{\text{所有VPP}}
\text{VPP内储能/EV的SOC惩罚}
$$

当前：

$$
\text{SOC权重}=0.25
$$

解释：储能或 EV 等设备 SOC 越界/偏离越大，该项越大。

#### 2.3.6 安全违规数量 `security_violation_count_penalty`

$$
\text{安全违规数量惩罚}
=
\text{安全违规数量权重}
\times
\text{AC后验违规数量}
\times
\text{单个违规惩罚}
$$

当前：

$$
\text{单个违规惩罚}=50.0,\quad
\text{安全违规数量权重}=1.0
$$

解释：这是 AC 潮流之后的后验违规数量惩罚。它比 projection gap 更接近真实安全，但只有发生违规时才非零。

#### 2.3.7 AC 后验违规幅值 `post_ac_violation_magnitude_penalty`

$$
\text{AC违规幅值惩罚}
=
\sum_{j}
\left|
\text{第 }j\text{ 个AC后验违规幅值}
\right|
$$

其中 `j` 遍历 AC 潮流检查后的电压越限、线路过载、变压器过载等违规记录。

解释：这是“违规有多严重”的幅值项。

#### 2.3.8 电压越限 `voltage_violation_penalty`

来自 `violation_penalties(report)`。

$$
\text{电压越限惩罚}
=
\text{电压越限权重}
\times
\text{电压越限原始惩罚}
$$

base 配置：

$$
\text{电压越限权重}=20.0
$$

sensitivity-v1 覆盖为：

$$
\text{电压越限权重}=100.0
$$

解释：母线电压低于下限或高于上限时，该项增大。

#### 2.3.9 线路过载 `line_overload_penalty`

来自 `violation_penalties(report)`。

$$
\text{线路过载惩罚}
=
\text{线路过载权重}
\times
\text{线路过载原始惩罚}
$$

当前：

$$
\text{线路过载权重}=50.0
$$

解释：线路 loading 超过 `line_loading_limit_percent` 时，该项增大。

#### 2.3.10 变压器过载 `transformer_overload_penalty`

来自 `violation_penalties(report)`。

$$
\text{变压器过载惩罚}
=
\text{变压器过载权重}
\times
\text{变压器过载原始惩罚}
$$

当前：

$$
\text{变压器过载权重}=50.0
$$

#### 2.3.11 潮流失败 `powerflow_penalty`

来自 `violation_penalties(report)`。

$$
\text{潮流失败惩罚}
=
\text{潮流失败权重}
\times
\text{潮流失败原始惩罚}
$$

当前：

$$
\text{潮流失败权重}=10.0
$$

解释：pandapower AC 潮流不收敛时，该项非零。

#### 2.3.12 envelope 宽度惩罚 `envelope_width_penalty`

$$
\text{边界宽度惩罚}
=
\text{边界宽度权重}
\times
\text{平均边界宽度比例}
$$

sensitivity-v1 中：

$$
\text{边界宽度权重}=0.1
$$

解释：用于惩罚过宽的 DOE/envelope，防止 DSO 给出过松、安全意义不足的边界。

#### 2.3.13 envelope 平滑惩罚 `smoothness_penalty`

$$
\text{边界平滑惩罚}
=
\text{边界平滑权重}
\times
\text{相邻时刻边界变化量}
$$

sensitivity-v1 中：

$$
\text{边界平滑权重}=0.05
$$

解释：用于惩罚相邻时刻 DOE/envelope 剧烈变化。

### 2.4 DSO 总成本与最终 reward

加权总成本：

$$
\begin{aligned}
\text{加权总成本}
=&
\text{运行成本}
+ \text{目标跟踪误差惩罚}
+ \text{动作投影惩罚}
+ \text{舒适度惩罚}
+ \text{SOC惩罚}
\\
&+
\text{安全违规数量惩罚}
+ \text{AC违规幅值惩罚}
+ \text{电压越限惩罚}
+ \text{线路过载惩罚}
+ \text{变压器过载惩罚}
\\
&+
\text{潮流失败惩罚}
+ \text{边界宽度惩罚}
+ \text{边界平滑惩罚}
\end{aligned}
$$

缩放总成本：

$$
\text{缩放总成本}
=
\sum_{\text{所有成本项}}
\text{某项缩放成本}
$$

AC 后验安全惩罚汇总：

$$
\begin{aligned}
\text{AC后验安全惩罚}
=&
\text{安全违规数量惩罚}
+ \text{AC违规幅值惩罚}
+ \text{电压越限惩罚}
\\
&+
\text{线路过载惩罚}
+ \text{变压器过载惩罚}
+ \text{潮流失败惩罚}
\end{aligned}
$$

最终环境 DSO reward：

$$
\text{DSO环境奖励}
=
-\text{DSO成本缩放系数}
\times
\text{缩放总成本}
+ \text{可行奖励}
+ \text{跟踪奖励}
+ \text{有效响应奖励}
$$

其中：

$$
\text{可行奖励}
=
\begin{cases}
1, & \text{AC后验检查安全} \\
0, & \text{AC后验检查不安全}
\end{cases}
$$

$$
\text{跟踪奖励}
=
\frac{0.25}{1+\text{目标跟踪误差}}
$$

$$
\text{有效响应奖励}
=
\begin{cases}
\text{有效响应得分}, & \text{如果环境提供该得分} \\
\dfrac{1}{1+\text{目标跟踪误差}}, & \text{否则}
\end{cases}
$$

当前 base/paper-long：

$$
\text{DSO成本缩放系数}=0.05
$$

所以 DSO reward 的直观含义是：

$$
\text{DSO奖励}
=
\text{AC可行奖励}
+ \text{跟踪奖励}
+ \text{有效响应奖励}
- 0.05
\times
\text{缩放后的总成本}
$$

## 3. VPP Dispatch Reward

### 3.1 归属

| 项目 | 说明 |
|---|---|
| actor | `{vpp_id}_dispatch` |
| critic/value | 对应该 VPP 的 dispatch critic head 或 value head |
| 时间尺度 | 快周期，每个 step 都有 |
| 信息范围 | 本 VPP 当前功率、成本、comfort/SOC、本 VPP 收到的 DSO envelope/audit |
| 是否直接使用全局 DSO reward | 否，`DISPATCH_RAW_DSO_REWARD_WEIGHT = 0.0` |

### 3.2 数学形式

定义：

$$
\text{目标跟踪偏差}
=
\left|
\text{VPP实际聚合功率}
-
\text{DSO给该VPP的目标功率}
\right|
$$

$$
\text{投影偏差}
=
\text{该VPP动作被本地边界或安全边界修正的功率距离}
$$

$$
\text{VPP可调功率区间宽度}
=
\text{最大可调功率}
-
\text{最小可调功率}
$$

市场收益代理：

$$
\text{电能市场收入}
=
\text{电价}
\times
\text{VPP实际聚合功率}
\times
\text{步长小时数}
$$

灵活性服务付款：

$$
\text{灵活性服务收入}
=
1.00
\times
\text{电价}
\times
\text{服务功率数量}
\times
\text{步长小时数}
$$

其中：

$$
\text{服务功率数量}
=
\begin{cases}
\max(0,-\text{VPP实际聚合功率}), & \text{吸收/充电/进口功率服务} \\
\max(0,\text{VPP实际聚合功率}), & \text{注入/削减/出口功率服务} \\
|\text{VPP实际聚合功率}|, & \text{未指定服务方向}
\end{cases}
$$

可用容量付款：

$$
\text{可用容量收入}
=
0.02
\times
\text{电价}
\times
\text{VPP可调功率区间宽度}
\times
\text{步长小时数}
$$

DER 运行成本：

$$
\text{DER运行成本}
=
\text{VPP内部设备运行成本}
\times
\text{步长小时数}
$$

私有利润代理：

$$
\text{私有利润代理}
=
\text{电能市场收入}
+ \text{灵活性服务收入}
+ \text{可用容量收入}
- \text{DER运行成本}
$$

comfort/SOC 惩罚先缩放：

$$
\text{缩放舒适/SOC惩罚}
=
\min\left(
5,\,
\frac{
\max(0,\text{舒适度惩罚}+\text{SOC惩罚})
}{100}
\right)
$$

跟踪惩罚：

$$
\text{目标跟踪惩罚}
=
25.0
\times
\text{目标跟踪偏差}^{2}
$$

envelope 投影惩罚：

$$
\text{边界投影惩罚}
=
5.0
\times
\text{投影偏差}
+ 10.0
\times
\text{投影偏差}^{2}
$$

preferred region 奖励：

$$
\text{偏好区间奖励}
=
0.50
\times
\text{是否落在偏好区间}
\times
\text{DSO引导强度}
\times
\text{边界收缩程度}
\times
\text{响应有效性}
$$

其中：

$$
\text{是否落在偏好区间}
=
\begin{cases}
1, & \text{VPP实际功率在DSO偏好区间内} \\
0, & \text{否则}
\end{cases}
$$

$$
\text{边界收缩程度}
=
\operatorname{clip}
\left(
1
-
\frac{\text{偏好区间宽度}}{\text{硬约束区间宽度}},
0,
1
\right)
$$

$$
\text{响应有效性}
=
\begin{cases}
\text{有效响应得分}, & \text{如果环境提供该得分} \\
\operatorname{clip}
\left(
1-\dfrac{\text{投影偏差}}{\text{硬约束区间宽度}},
0,
1
\right), & \text{否则}
\end{cases}
$$

最终 VPP dispatch 环境 reward：

$$
\begin{aligned}
\text{VPP调度环境奖励}
=&
0.02 \times \text{私有利润代理}
+ \text{偏好区间奖励}
- \text{目标跟踪惩罚}
\\
&-
\text{边界投影惩罚}
- 0.001
\times
\text{缩放舒适/SOC惩罚}
\end{aligned}
$$

训练时还会扣同一个安全壳惩罚：

$$
\text{VPP调度训练奖励}
=
\text{VPP调度环境奖励}
-
\text{调度安全壳系数}
\times
\text{安全壳惩罚}
$$

当前：

$$
\text{调度安全壳系数}=1.0
$$

### 3.3 VPP dispatch 权重

| 项 | 当前真实权重 | 来源 |
|---|---:|---|
| 私有利润 `private_profit_proxy` | 0.02 | `reward_contracts.py` |
| 跟踪误差平方 | 25.0 | `reward_contracts.py` |
| 投影 gap 线性项 | 5.0 | `reward_contracts.py` |
| 投影 gap 二次项 | 10.0 | `reward_contracts.py` |
| comfort/SOC 缩放惩罚 | 0.001 | `reward_contracts.py` |
| preferred region bonus | 0.50 | `reward_contracts.py` |
| 原始 DSO reward 共享 | 0.0 | `reward_contracts.py` |

重要注意：

`configs/*sensitivity_attention_v1.yaml` 里有：

```yaml
reward:
  vpp:
    preferred_range_weight: 0.5
    projection_gap_weight: 2.0
    private_profit_weight: 0.02
    raw_dso_reward_weight: 0.0
```

但当前代码搜索结果显示，这些 `reward.vpp.*` 配置没有被 `reward_design.py` 动态读取。当前 VPP reward 的真实权重来自 `reward_contracts.py` 常量。也就是说，VPP 配置项目前更像文档性配置或预留配置，不是实际生效的权重入口。

## 4. VPP Portfolio Reward

### 4.1 归属

| 项目 | 说明 |
|---|---|
| actor | `{vpp_id}_portfolio` |
| critic/value | HAPPO/HATRPO 中有 portfolio value/head；HASAC/MATD3 连续 dispatch 分支不作为主要 reward vector |
| 时间尺度 | 慢周期，默认每 24 step 决策一次 |
| 是否直接改变物理 DER membership | 当前更多是慢周期组合建议/动作标签，不应理解为每步真实重聚合所有 DER |
| 是否直接使用 raw global DSO reward | 否，`PORTFOLIO_RAW_DSO_REWARD_WEIGHT = 0.0` |

### 4.2 数学形式

定义：

$$
\text{目标跟踪偏差}
=
\left|
\text{VPP实际聚合功率}
-
\text{DSO给该VPP的目标功率}
\right|
$$

$$
\text{投影偏差}
=
\text{该VPP动作被本地边界或安全边界修正的功率距离}
$$

$$
\text{VPP可调功率区间宽度}
=
\text{最大可调功率}
-
\text{最小可调功率}
$$

preferred score：

$$
\text{偏好区间得分}
=
\begin{cases}
1, & \text{VPP实际功率在偏好区间内} \\
\max\left(
0,\,
1-\dfrac{\text{到偏好区间的距离}}{\text{偏好区间宽度}}
\right), & \text{否则}
\end{cases}
$$

可靠性 bonus：

$$
\text{可靠性奖励}
=
0.50
\times
\max\left(
0,\,
1-
\frac{
\text{目标跟踪偏差}
}{
\max(10^{-6},\,\text{VPP可调功率区间宽度})
}
\right)
$$

可用容量质量：

$$
\text{可用容量质量}
=
\min\left(
1,\,
\frac{\text{VPP可调功率区间宽度}}{0.50}
\right)
$$

网络惩罚：

$$
\text{网络惩罚}
=
\text{电压越限惩罚}
+ \text{线路过载惩罚}
+ \text{变压器过载惩罚}
+ \text{潮流失败惩罚}
+ \text{AC违规幅值惩罚}
$$

局部 DSO 对齐奖励：

$$
\begin{aligned}
\text{局部DSO对齐奖励}
=&
0.35
\times
\text{偏好区间得分}
+ 0.25
\times
\text{可行奖励}
\\
&+
0.15
\times
\text{可用容量质量}
+ 0.10
\times
\text{可靠性奖励}
- 0.001
\times
\text{网络惩罚}
\end{aligned}
$$

切换成本：

$$
\text{切换成本}
=
\begin{cases}
0.00, & \text{保持当前组合} \\
0.02, & \text{重新加权} \\
0.08, & \text{建议成员关系变化}
\end{cases}
$$

交付风险惩罚：

$$
\text{交付风险惩罚}
=
0.50
\times
\text{投影偏差}
+ 0.20
\times
\text{目标跟踪偏差}
$$

长期利润代理：

$$
\text{长期利润代理}
=
0.10
\times
\text{私有利润代理}
$$

最终 portfolio reward：

$$
\begin{aligned}
\text{VPP组合环境奖励}
=&
\text{长期利润代理}
+ \text{局部DSO对齐奖励}
+ \text{可靠性奖励}
\\
&-
\text{切换成本}
- \text{交付风险惩罚}
\end{aligned}
$$

### 4.3 为什么 portfolio 介入不一定让 total reward 大幅跳变

原因有三点：

1. HAPPO/HATRPO 中 portfolio 是慢周期分支，不是每步都有效。如果不是决策步，常被强制为 0。
2. 旧 paper-long 长样本里，DSO reward 绝对量级占学习信号约 99.8%，portfolio 约 0.024%。这意味着 portfolio 即使变化，也很难让总 reward 曲线肉眼上大幅跳变。
3. 当前 portfolio 动作更像组合配置建议或慢变量动作标签，不等价于“每个 VPP 真实 DER membership 每步重新洗牌”。

## 5. 安全壳惩罚 Shield Intervention Penalty

安全壳惩罚用于回答这个问题：

$$
\text{RL原始动作是否被本地边界、AC-aware DOE 或 AC certificate 修改过？}
$$

先取四类 gap：

$$
\begin{aligned}
\text{原始动作投影距离} &= \max(0,\text{原始动作投影距离记录值}) \\
\text{本地边界投影距离} &= \max(0,\text{本地边界投影距离记录值}) \\
\text{AC-aware边界投影距离} &= \max(0,\text{AC-aware边界投影距离记录值}) \\
\text{AC证书修复距离} &= \max(0,\text{AC证书修复距离记录值})
\end{aligned}
$$

如果 local/AC/certificate 任一 gap 非零：

$$
\text{安全壳介入量}
=
\text{本地边界投影距离}
+ \text{AC-aware边界投影距离}
+ \text{AC证书修复距离}
$$

否则：

$$
\text{安全壳介入量}
=
\max\left(
\text{原始动作投影距离},
\text{本地边界投影距离},
\text{AC-aware边界投影距离},
\text{AC证书修复距离}
\right)
$$

安全壳惩罚：

$$
\text{安全壳惩罚}
=
5.0
\times
\text{安全壳介入量}
+ 10.0
\times
\text{安全壳介入量}^{2}
$$

训练时：

$$
\text{DSO训练奖励}
=
\text{DSO环境奖励}
- \text{安全壳惩罚}
$$

$$
\text{VPP调度训练奖励}
=
\text{VPP调度环境奖励}
- \text{安全壳惩罚}
$$

portfolio reward 当前不扣安全壳惩罚。

解释：

- `post_ac_violation_count = 0` 说明最终执行动作安全。
- 安全壳惩罚大于 0，或安全壳介入量大于 0，说明安全来自投影/修复介入，策略原始动作还不够安全。
- 所以不能只看 post-AC violation，还要同时看 projection/shield gap。

## 6. 当前权重总表

### 6.1 DSO 权重

| DSO 项 | base 权重 | sensitivity-v1 权重 | 是否实际生效 |
|---|---:|---:|---|
| `operation_cost` | 1.0 | 1.0 | 是 |
| `target_tracking_error_penalty` | 1.0 | 1.0 | 是 |
| `action_projection_penalty` | 2.0 | 5.0 | 是 |
| `comfort_violation_penalty` | 0.02 | 0.02 | 是 |
| `soc_violation_penalty` | 0.25 | 0.25 | 是 |
| `security_violation_count_penalty` | 1.0 | 1.0 | 是 |
| `voltage_violation_penalty` | 20.0 | 100.0 | 是 |
| `line_overload_penalty` | 50.0 | 50.0 | 是 |
| `transformer_overload_penalty` | 50.0 | 50.0 | 是 |
| `powerflow_penalty` | 10.0 | 10.0 | 是 |
| `envelope_width_penalty` | 默认 1.0 | 0.1 | 是 |
| `smoothness_penalty` | 默认 1.0 | 0.05 | 是 |

其他 DSO 超参数：

| 参数 | 值 | 说明 |
|---|---:|---|
| `dso_reward_cost_scale` | 0.05 | DSO reward 中 `scaled_total_cost` 的系数 |
| `security_violation_count_penalty` | 50.0 | 每个 post-AC 违规数量的 raw 惩罚 |
| `component_clip` | 10.0 | 单项 scaled cost 上限 |

### 6.2 VPP dispatch 权重

| VPP dispatch 项 | 权重 | 是否实际生效 |
|---|---:|---|
| `private_profit_proxy` | 0.02 | 是 |
| `target_tracking_penalty = tracking_gap^2` | 25.0 | 是 |
| `projection_gap` 线性项 | 5.0 | 是 |
| `projection_gap^2` 二次项 | 10.0 | 是 |
| `scaled_comfort_soc_penalty` | 0.001 | 是 |
| `preferred_region_bonus` | 0.50 | 是 |
| raw DSO reward 共享 | 0.0 | 是，表示不共享 |

### 6.3 VPP portfolio 权重

| VPP portfolio 项 | 权重 |
|---|---:|
| `preferred_score` | 0.35 |
| `feasibility_bonus` | 0.25 |
| `availability_quality` | 0.15 |
| `reliability_bonus` 在 localized DSO alignment 中 | 0.10 |
| `network_penalty` | -0.001 |
| `long_horizon_profit_proxy` | 0.10 * `private_profit_proxy` |
| `delivery_risk_penalty` 中 projection gap | 0.50 |
| `delivery_risk_penalty` 中 tracking gap | 0.20 |
| `switching_cost: keep` | 0.00 |
| `switching_cost: reweight` | 0.02 |
| `switching_cost: propose_membership_change` | 0.08 |

## 7. 实验中的大致占比和值域

### 7.1 旧 paper-long 中 DSO 成本项量级

数据源：`outputs/paper_training_long_current/runs/*/simulator_results/reward_components.csv`

样本：45 个 `reward_components.csv`，30240 行。

| 字段 | mean | p05 | p50 | p95 | min | max | abs-cost-share |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dso_reward` | -178.197 | -427.358 | -125.680 | -38.436 | -493.604 | 0.233 |  |
| `total_cost` | 3588.890 | 793.713 | 2538.600 | 8572.170 | 20.331 | 9897.070 |  |
| `operation_cost` | 63.378 | 32.477 | 62.926 | 96.516 | 0.218 | 138.847 | 1.766% |
| `target_tracking_error_penalty` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000% |
| `action_projection_penalty` | 0.042 | 0.000 | 0.000 | 0.000 | 0.000 | 10.143 | 0.001% |
| `comfort_violation_penalty` | 3500.320 | 420.118 | 2475.630 | 8508.940 | 3.866 | 9858.850 | 97.532% |
| `soc_violation_penalty` | 24.328 | 0.562 | 4.034 | 11.698 | 0.000 | 1079.030 | 0.678% |
| `voltage_violation_penalty` | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 4.149 | 0.000% |
| `line_overload_penalty` | 0.822 | 0.000 | 0.000 | 0.000 | 0.000 | 2497.770 | 0.023% |
| `transformer_overload_penalty` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000% |
| `powerflow_penalty` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000% |
| `action_projection_gap_mw` | 0.000040 | 0.000 | 0.000 | 0.000 | 0.000 | 0.028925 |  |

结论：

- 旧 long 的 DSO reward 几乎被 comfort 项支配。
- `comfort_violation_penalty` 平均占加权 DSO 成本约 97.5%。
- 网络安全项虽然存在，但在平均量级上几乎没有形成有效学习信号。

### 7.2 sensitivity-v1 当前输出中 DSO 成本项量级

数据源：`outputs/paper_training_long_sensitivity_v1_20260528_thread8_progress/runs/*/simulator_results/reward_components.csv`

样本：45 个 `reward_components.csv`，30240 行。

| 字段 | mean | p05 | p50 | p95 | min | max | abs-cost-share |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dso_reward` | 2.194 | 2.129 | 2.209 | 2.237 | 0.982 | 2.248 |  |
| `scaled_total_cost` | 1.081 | 0.249 | 0.810 | 2.366 | 0.035 | 15.672 |  |
| `feasibility_bonus` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  |
| `tracking_bonus` | 0.250 | 0.249 | 0.250 | 0.250 | 0.153 | 0.250 |  |
| `effective_response_bonus` | 0.998 | 0.997 | 1.000 | 1.000 | 0.613 | 1.000 |  |
| `total_cost` | 145.352 | 46.661 | 124.763 | 275.275 | 1.093 | 731.925 |  |
| `operation_cost` | 50.896 | 3.750 | 52.336 | 96.472 | 0.000 | 159.156 | 35.016% |
| `target_tracking_error_penalty` | 0.054 | 0.000 | 0.000 | 0.001 | 0.000 | 39.925 | 0.037% |
| `action_projection_penalty` | 0.813 | 0.000 | 0.000 | 0.000 | 0.000 | 509.061 | 0.559% |
| `comfort_violation_penalty` | 80.002 | 10.815 | 57.499 | 191.911 | 0.077 | 223.470 | 55.040% |
| `soc_violation_penalty` | 13.552 | 0.010 | 0.774 | 132.562 | 0.000 | 286.661 | 9.324% |
| `security_violation_count_penalty` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000% |
| `post_ac_violation_magnitude_penalty` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000% |
| `voltage_violation_penalty` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000% |
| `line_overload_penalty` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000% |
| `transformer_overload_penalty` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000% |
| `powerflow_penalty` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000% |
| `envelope_width_penalty` | 0.034 | 0.030 | 0.030 | 0.061 | 0.030 | 0.070 | 0.023% |
| `smoothness_penalty` | 0.000347 | 0.000006 | 0.000089 | 0.002501 | 0.000000 | 0.008028 | 0.000% |
| `post_ac_security_penalty` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `action_projection_gap_mw` | 0.001736 | 0.000 | 0.000 | 0.000 | 0.000 | 0.631861 |  |
| `local_bounds_projection_gap_mw` | 0.000016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.045907 |  |
| `ac_aware_projection_gap_mw` | 0.000000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `ac_certified_projection_gap_mw` | 0.001720 | 0.000 | 0.000 | 0.000 | 0.000 | 0.631861 |  |

结论：

- sensitivity-v1 的 reward 尺度比旧 long 明显健康，`dso_reward` 从旧口径的负数大幅值变成 2.2 左右的稳定正值。
- comfort 成本仍是最大项，但占比从约 97.5% 降到约 55.0%。
- operation cost 占约 35.0%，SOC 占约 9.3%。
- 这批 baseline 输出中 post-AC 安全违规为 0，所以安全违规项占比为 0。安全影响主要体现在 projection/certificate gap，而不是违规罚分。

### 7.3 旧 paper-long 训练 step 中角色 reward 占比

数据源：`outputs/paper_training_long_current/runs/*/train/*_step_metrics.csv`

样本：60 个训练 step metrics 文件，4838400 行。

| 字段 | mean | p05 | p50 | p95 | min | max | abs-share-vs-learning |
|---|---:|---:|---:|---:|---:|---:|---:|
| `reward` | -206.645 | -476.466 | -153.500 | -48.922 | -557.391 | 1.113 |  |
| `dso_reward` | -206.576 | -475.608 | -153.562 | -49.282 | -556.300 | -0.487 | 99.631% |
| `mean_dispatch_reward` / `vpp_dispatch_reward` | -0.086 | -0.861 | 0.063 | 0.354 | -1.063 | 0.503 | 0.172% |
| `mean_portfolio_reward` | 0.051 | 0.000 | 0.000 | 0.000 | 0.000 | 1.681 | 0.024% |
| `projection_gap_mw` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `total_cost` | 4156.470 | 1010.640 | 3096.200 | 9537.170 | 34.744 | 11151.000 |  |
| `violation_count` | 0.012 | 0.000 | 0.000 | 0.000 | 0.000 | 14.000 |  |

按算法拆分：

| algorithm | rows | reward_mean | dso_mean | dispatch_mean | portfolio_mean | dso_abs_share | dispatch_abs_share | portfolio_abs_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `happo_sequential_ctde` | 1612800 | -207.277 | -207.242 | -0.086 | 0.051 | 99.803% | 0.172% | 0.024% |
| `hasac_continuous_dispatch` | 1612800 | -205.324 | -205.241 | -0.083 | 0.000 | 99.827% | 0.173% | 0.000% |
| `matd3_continuous_dispatch` | 1612800 | -207.334 | -207.245 | -0.089 | 0.000 | 99.828% | 0.172% | 0.000% |

结论：

- 旧 long 中学习信号几乎被 DSO reward 单独控制。
- dispatch 和 portfolio 的 reward 量级太小，即使策略在 VPP 层有变化，也很难体现在总 reward 曲线中。

### 7.4 当前 HAPPO/HATRPO smoke 中角色 reward 量级

数据源：

- `outputs/test_paper_training_structured_happo_sensitivity_smoke/runs/*/train/*_step_metrics.csv`
- `outputs/test_happo_cuda_smoke_20260601/train/happo_step_metrics.csv`
- `outputs/test_hatrpo_training/hatrpo_step_metrics.csv`

样本：6 行。这个样本非常小，只能辅助理解。

| 字段 | mean | p05 | p50 | p95 | min | max | abs-share-vs-learning |
|---|---:|---:|---:|---:|---:|---:|---:|
| `reward` | 3.136 | 2.573 | 3.173 | 3.660 | 2.566 | 3.665 |  |
| `dso_reward` | 2.241 | 2.229 | 2.247 | 2.248 | 2.229 | 2.248 | 71.412% |
| `mean_dispatch_reward` | 0.197 | -0.006 | 0.262 | 0.339 | -0.008 | 0.346 | 6.356% |
| `mean_portfolio_reward` | 0.698 | 0.000 | 0.938 | 1.172 | 0.000 | 1.189 | 22.232% |
| `shield_intervention_penalty` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `projection_gap_mw` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `total_cost` | 17.052 | 0.415 | 17.914 | 34.958 | 0.035 | 37.469 |  |
| `violation_count` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

解释：

- 在修正后的短样本里，DSO 不再是 99% 以上的唯一主导项。
- 但样本只有 6 行，不能据此判断 paper-long 的真实长期占比。

### 7.5 reward_v2 train probe 中 role 内部分项占比

这一节是 2026-06-07 对上一版报告的修正。此前的表格把 DSO、dispatch、portfolio、safety 全部混在同一个分母里，所以只能叫“全局已审计项绝对占比”，不能叫“每个 role 内部 reward 百分比”。

当前最可靠的数据源是：

`outputs/_archive/paper_long/reward_v2_audit_train_probe_gpu_logfix_20260605_effective/reward_component_abs_share.csv`

它对应的训练 probe 有 HAPPO/HATRPO 的 `*_step_metrics.csv`，因此可以重建 dispatch 和 portfolio 的大部分有效项。注意：最新完整 paper-long 目录 `outputs/paper_training_long_reward_v2_minimal_20260604_gpu_decoder_bounds_happo_hatrpo_logfix` 的 manifest 仍是 `baseline_complete`，训练目录下没有 `*_step_metrics.csv`，所以它本身不能计算训练 role 内部分项占比。

#### 7.5.1 三种占比的区别

| 口径 | 分母 | 能说明什么 |
|---|---|---|
| 全局绝对占比 | 所有已审计项的绝对值总和 | 哪些字段在总体量级上最大 |
| role 内部审计占比 | 同一 role 已审计项的绝对值总和 | 某个 role 内部已审计项谁更大 |
| 真实训练贡献占比 | 该 role 实际进入训练 reward 的 signed 项 | 最严格，但需要排除诊断项双计并处理 shield penalty |

role 内部审计占比公式为：

$$
\text{某项 role 内部占比}
=
\frac{
\sum_t |\text{该项在第 }t\text{ 步的值}|
}{
\sum_{j \in \text{该 role 的已审计项}}\sum_t |\text{第 }j\text{ 项在第 }t\text{ 步的值}|
}
$$

这里用绝对值，所以负利润项也会产生正的量级占比。它不是 signed contribution。

#### 7.5.2 VPP dispatch 的 7 个有效项

v2_minimal dispatch reward 的完整公式是：

$$
\begin{aligned}
\text{dispatch reward}
=&
0.02 \times \text{私有利润代理}
+ 1.0 \times \text{服务支付}
+ 1.0 \times \text{可用容量支付}
\\
&-10.0 \times \text{合约短缺功率}^2
-(2.0 \times \text{投影距离} + 5.0 \times \text{投影距离}^2)
\\
&-0.02 \times \min(5,\text{舒适度/SOC原始惩罚}/100)
-0.01 \times \text{电池退化成本}
\end{aligned}
$$

train probe 中，dispatch 的 7 项角色内绝对占比是：

| 算法 | dispatch 项 | 平均值 | dispatch 角色内占比 | 说明 |
|---|---|---:|---:|---|
| HAPPO | 舒适度/SOC 惩罚 | 0.073548 | 45.020% | 最大项 |
| HAPPO | 可用容量支付 | 0.072072 | 44.117% | 第二大项 |
| HAPPO | 私有利润奖励 | -0.010529 | 10.785% | 均值为负，但量级仍明显 |
| HAPPO | 电池退化惩罚 | 0.000111 | 0.068% | 很弱 |
| HAPPO | 合约短缺惩罚 | 0.000017 | 0.010% | 几乎未激活 |
| HAPPO | 服务支付 | 0.000000 | 0.000% | 不是缺字段，是 verified delivery 服务支付为 0 |
| HAPPO | dispatch 投影惩罚 | 0.000000 | 0.000% | 几乎未触发 |
| HATRPO | 舒适度/SOC 惩罚 | 0.074216 | 44.839% | 最大项 |
| HATRPO | 可用容量支付 | 0.073374 | 44.330% | 第二大项 |
| HATRPO | 私有利润奖励 | -0.010474 | 10.753% | 均值为负 |
| HATRPO | 电池退化惩罚 | 0.000103 | 0.062% | 很弱 |
| HATRPO | 合约短缺惩罚 | 0.000027 | 0.016% | 很弱 |
| HATRPO | dispatch 投影惩罚 | 约 0 | 约 0% | 可忽略 |
| HATRPO | 服务支付 | 0.000000 | 0.000% | 未激活 |

修正后的解释：

- 不能再只说 dispatch 精确知道三项。对 reward_v2 train probe 来说，7 个公式项都能审计或由已有列派生。
- 但最新完整 paper-long 目录没有训练 step metrics，所以不能给出 paper-long 级别 dispatch 内部百分比。
- 当前 train probe 中真正主导 dispatch 的不是服务履约，而是舒适度/SOC 与可用容量支付；两者合计约 89%。
- `service_payment`、`contract_delivery_penalty`、`dispatch_projection_penalty` 和 `battery_degradation` 不是理论上缺失，而是在该 probe 中为 0 或接近 0，因此几乎不驱动学习。

#### 7.5.3 VPP portfolio 的窗口项

v2_minimal portfolio reward 每 24 step 结算一次：

$$
\begin{aligned}
\text{portfolio window reward}
=&
0.05 \times \text{窗口平均利润}
+0.5 \times \text{窗口平均已验证容量}
-1.0 \times \text{窗口平均合约短缺}
\\
&-1.0 \times \text{窗口平均安全壳修复距离}
-0.5 \times \text{窗口平均投影距离}
-0.02 \times \text{窗口平均舒适度/SOC惩罚}
-\text{切换成本}
\end{aligned}
$$

train probe 中，portfolio 的 role 内部占比如下：

| 算法 | portfolio 项 | 平均值 | portfolio 角色内占比 | 说明 |
|---|---|---:|---:|---|
| HAPPO | 切换成本 | 0.002493 | 42.259% | 最大项 |
| HAPPO | 未来 comfort/SOC 惩罚 | 0.001992 | 33.770% | 第二大项 |
| HAPPO | 长期利润项 | -0.001167 | 23.836% | 均值为负 |
| HAPPO | 合约短缺惩罚 | 0.000008 | 0.133% | 很弱 |
| HAPPO | 未来 shield 惩罚 | 约 0 | 0.003% | 很弱 |
| HAPPO | 已验证容量奖励 | 0.000000 | 0.000% | 未激活 |
| HAPPO | 未来 projection 惩罚 | 0.000000 | 0.000% | 未激活 |
| HATRPO | 切换成本 | 0.003885 | 53.325% | 最大项 |
| HATRPO | 未来 comfort/SOC 惩罚 | 0.002009 | 27.579% | 第二大项 |
| HATRPO | 长期利润项 | -0.001163 | 18.941% | 均值为负 |
| HATRPO | 合约短缺惩罚 | 0.000011 | 0.155% | 很弱 |
| HATRPO | 未来 shield/projection、已验证容量 | 约 0 | 约 0% | 基本未激活 |

结论：portfolio agent 已经参与训练，但当前主要看见的是切换成本、未来 comfort/SOC 和长期利润代理，不是 verified capacity、future projection 或 future shield 风险。若要它真正体现“重新选择聚合 DER 后 reward 大幅变化”，需要让 portfolio 动作更强地影响未来可行域、已验证容量、合约短缺和投影风险。

#### 7.5.4 DSO 不能直接用当前审计表算严格内部占比

DSO v2_minimal 的训练公式是：

$$
\begin{aligned}
\text{DSO reward}
=&
\text{安全容量利用奖励}
-\text{灵活性采购成本}
-\text{网损成本}
-\text{削减成本}
-\text{平滑惩罚}
\\
&-\text{安全裕度惩罚}
-\text{硬安全违规惩罚}
-\text{潮流失败惩罚}
-\text{DSO负责投影惩罚}
\end{aligned}
$$

当前审计表里的 DSO 字段不是干净的代数分解，主要原因：

- `dso_over_conservative_curtailment_penalty` 是诊断基量。
- `dso_curtailment_cost = 0.5 * dso_over_conservative_curtailment_penalty` 才是实际训练扣分项之一。
- 如果把二者同时放进分母，就把同一个保守性信号计算了两次。
- `dso_safety_margin_penalty` 是总父项，`dso_voltage_guard_penalty`、`dso_line_guard_penalty`、`dso_trafo_guard_penalty` 是子诊断项，也不能父子同算。

所以严格说法是：

- 当前 DSO 审计字段显示：保守性/削减相关信号最大。
- 但不能把当前 DSO 审计表直接解释成 DSO reward 内部百分比。
- 后续如果要精确 DSO role 内部占比，应新增干净的 `dso_effective_reward_terms.csv`，只落盘公式中实际相加/相减的项，不落盘重复诊断项作为分母成员。

#### 7.5.5 safety shield 不是独立 role

`shield_intervention_penalty` 在审计表里属于 `source = safety`，但 safety shield 不是 MARL actor。训练时它会按系数扣回 DSO 和 dispatch：

$$
\text{DSO训练reward}
=
\text{DSO环境reward}
-
\text{DSO shield系数}
\times
\text{shield penalty}
$$

$$
\text{dispatch训练reward}
=
\text{dispatch环境reward}
-
\text{dispatch shield系数}
\times
\text{shield penalty}
$$

因此，如果要计算 after-shield 的 role 内部百分比，必须把 shield penalty 按系数分配回 DSO/dispatch。当前 `reward_component_abs_share.csv` 没有做这个分配，所以它只能作为 safety 诊断表，而不是最终 role reward 分解表。

## 8. 当前最容易误解的点

### 8.1 `projection_gap` 不是 AC 安全

`projection_gap` 表示动作被投影修正的幅度。它可以反映策略是否依赖安全壳，但不能直接等价于电网安全。

真正的 AC 后验安全要看：

```text
post_ac_violation_count
post_ac_voltage_violation_count
post_ac_line_overload_count
post_ac_trafo_overload_count
post_ac_powerflow_failed
post_ac_violation_magnitude
post_ac_security_penalty
```

### 8.2 `post_ac_violation_count = 0` 也不等于 RL 策略天然安全

如果：

```text
post_ac_violation_count = 0
shield_intervention_gap_mw > 0
```

说明最终安全，但中间经过了安全壳修正。此时应报告为：

```text
最终执行动作 AC 安全，但原始策略动作仍依赖 projection/certificate 兜底。
```

### 8.3 当前 VPP reward 配置项不完全是实际权重入口

DSO nested reward 配置会生效，例如：

```yaml
reward:
  dso:
    voltage_violation_weight: 100.0
```

会映射到：

```text
voltage_violation_penalty
```

但 VPP nested reward 配置目前没有被 reward 计算函数读取。实际 VPP 权重来自 `reward_contracts.py` 常量。

### 8.4 为什么 reward 平台期可能出现

如果初始策略加安全投影后已经得到较高 reward，那么继续训练时 reward 提升会很小。这有两种可能：

1. 健康平台期：策略已经接近当前 reward 定义下的局部最优，且安全违规为 0，projection gap 很低。
2. 不健康平台期：安全壳把错误动作修成安全动作，导致 reward 看起来高，但原始 actor 没学到物理约束。

区分方式：

```text
健康平台期：
  post_ac_violation_count ≈ 0
  shield_intervention_gap_mw 逐步下降
  action_projection_count 逐步下降
  deterministic eval 优于 random/zero/rule baseline

不健康平台期：
  post_ac_violation_count ≈ 0
  但 shield_intervention_gap_mw 不下降
  actor raw action 大量触边或被 certificate 修复
  VPP dispatch/portfolio 分项没有改善
```

## 9. 对论文实验报告的建议

建议后续 paper-long 报告至少同时画这些曲线：

1. `reward` 总曲线。
2. `dso_reward`、`mean_dispatch_reward`、`mean_portfolio_reward` 分角色曲线。
3. `scaled_total_cost` 和 `total_cost`。
4. `comfort_violation_penalty`、`operation_cost`、`soc_violation_penalty`、`action_projection_penalty`。
5. `post_ac_violation_count`、`post_ac_security_penalty`。
6. `shield_intervention_gap_mw`、`shield_intervention_penalty`。
7. `action_projection_gap_mw`、`local_bounds_projection_gap_mw`、`ac_aware_projection_gap_mw`、`ac_certified_projection_gap_mw`。
8. VPP 子项：`private_profit_proxy`、`target_tracking_penalty`、`envelope_projection_penalty`、`preferred_region_bonus`、`localized_dso_alignment_reward`。

其中第 8 类目前在主 paper-long HAPPO/HATRPO 输出中还不完整，建议后续补充日志字段，否则很难用论文级证据解释 VPP 智能体到底学到了什么。
