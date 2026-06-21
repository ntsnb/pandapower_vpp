从你上传的训练记录判断：**当前主要不是 GPU、pandapower、critic loss 或网络是否跑起来的问题，而是 MARL 的“奖励设计 + safety shield 后的 credit assignment + dispatch actor 更新稳定性”出了问题。**

更精确地说：

> **第一主因：reward 结构失衡，DSO 侧 curtailment / over-conservative proxy 主导了训练目标。**
> **第二主因：safety shield 把 post-AC violation 全部修成 0，导致策略几乎学不到真实安全边界。**
> **第三主因：VPP dispatch actor 更新过猛，PPO/HAPPO 的 trust region 已经被严重破坏。**
> **第四主因：DSO actor 不是没有网络结构，而是动作空间太窄、学习信号太弱。**

下面我按 MARL 的各个部分逐项判断。

---

# 1. 当前训练现象说明了什么？

你的 HAPPO full 前 21 个完整 episode 是：

| 阶段         |  reward |  cost | violation | projection gap |
| ---------- | ------: | ----: | --------: | -------------: |
| episode 1  | -216.38 | 28323 |         0 |             很小 |
| episode 14 | -226.67 | 34782 |         0 |            约 0 |
| episode 21 | -223.73 | 33398 |         0 |            约 0 |

也就是说：

1. **reward 前 14 个 episode 明显变差。**
2. **cost 也明显升高。**
3. **后面 15 到 21 episode 有一点恢复，但仍然没有回到 episode 1 水平。**
4. **violations 始终为 0。**
5. **projection gap 基本为 0。**

这说明当前训练不是崩溃，但也不能说策略在有效学习。

最关键的是：

> **物理安全指标一直很好，但经济指标和 reward 没有同步改善。**

这通常意味着：
**安全性不是 MARL 策略自己学出来的，而是 safety shield / AC-aware projection / certificate/backoff 兜底出来的。**

所以现在不能说“MARL 学会了安全调度”，更像是：

> **MARL 输出一个动作，安全层把它修成可行解，然后 reward 又主要被 DSO curtailment 类 proxy 支配，导致 actor 的学习方向不稳定。**

---

# 2. Critic loss 下降，但 reward 变差：说明 critic 不是主矛盾

你的 probe 里 HAPPO critic loss 是：

| episode |  reward |  cost | critic loss |
| ------: | ------: | ----: | ----------: |
|       0 | -216.38 | 28323 |    0.002419 |
|       1 | -216.87 | 28891 |    0.001073 |
|       2 | -217.25 | 29176 |    0.000509 |
|       3 | -218.24 | 29678 |    0.000507 |

现象是：

> **critic loss 明显下降，但 reward/cost 变差。**

这说明 critic 在拟合当前采样到的 return/advantage target，但这不等于 actor 的策略在变好。

在 PPO/HAPPO/MAPPO 里，actor 更新大致依赖：

$$
L_{\pi}
=

\mathbb{E}
\left[
\min
\left(
r_t(\theta) A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

其中：

$$
r_t(\theta)
=

\frac{\pi_{\theta}(a_t|o_t)}
{\pi_{\theta_{\text{old}}}(a_t|o_t)}
$$

critic 的作用是估计 value，然后产生 advantage：

$$
A_t = Q_t - V(s_t)
$$

如果 reward 本身设计偏了，或者 action 经过 projection 后真实执行动作和 actor 采样动作不一致，那么 critic 仍然可以把当前数据拟合得很好，但 actor 学到的是错误方向。

所以这里不是简单的：

> critic loss 下降 = 训练正常。

而应该判断为：

> **critic 在拟合一个已经被 reward proxy、安全层修正、投影动作污染过的学习目标。**

因此 critic 不是首要故障点。

---

# 3. 最大问题一：reward 被 DSO curtailment proxy 主导

你记录里的 reward 分项占比非常关键。

HAPPO 分项占比：

| 分项                                        |   绝对占比 |
| ----------------------------------------- | -----: |
| DSO over-conservative curtailment penalty | 52.70% |
| DSO curtailment cost                      | 26.35% |
| dispatch comfort/SOC penalty              |  5.60% |
| availability payment                      |  5.49% |
| DSO safe capacity utilization reward      |  4.69% |
| DSO flex procurement cost                 |  3.29% |
| dispatch private profit reward            |  1.34% |

这里最重要的结论是：

$$
52.70% + 26.35% = 79.05%
$$

也就是说，**接近 80% 的 reward 绝对量来自 DSO over-conservative curtailment penalty 和 DSO curtailment cost。**

这非常危险。

因为你的 MARL 目标本来应该是多目标折中：

1. 保证电压安全；
2. 保证线路/变压器不过载；
3. 降低 DSO 调度成本；
4. 让 VPP 有利润；
5. 保持 DER 舒适度和 SoC 合理；
6. 减少不必要弃光/削减；
7. 保持 reverseflow 场景下的物理可行性；
8. 减少 safety shield 介入。

但现在 reward 实际上变成了：

> **主要优化 DSO curtailment / over-conservative proxy。**

这会导致策略学偏。

尤其是 reverseflow 场景中，你的记录显示：

| 算法     | 场景          |  reward |     cost | post-AC violation | AC projection gap |
| ------ | ----------- | ------: | -------: | ----------------: | ----------------: |
| HAPPO  | reverseflow | -678.70 | 22835.03 |                 0 |             19.10 |
| HATRPO | reverseflow | -573.63 | 21273.25 |                 0 |             16.32 |

这个现象非常重要：

> **reverseflow 下 cost 不一定差，但 reward 极差。**

HATRPO reverseflow 的 cost 甚至低于 rule_based，但 reward 仍然非常差。这说明 reward 和真实经济/物理目标不一致。

也就是说，问题不是策略真的把系统运行得很差，而是：

> **reward 对 reverseflow 场景的评价函数出了偏差。**

尤其可能存在这几类问题：

1. curtailment cost 和 over-conservative penalty 重复惩罚；
2. reverseflow 场景下本来就需要一定限制，但 reward 把必要限制也惩罚成“过度保守”；
3. safe capacity utilization reward 在 reverseflow 下定义不合理；
4. DSO reward 过大，压制了 dispatch reward 和 portfolio reward；
5. post-shield violation 为 0 后，安全项不再提供有效梯度，reward 只剩经济 proxy 在主导。

所以第一结论是：

> **当前 MARL 最主要的问题在 reward function，尤其是 DSO reward 的 curtailment / over-conservative 相关项。**

---

# 4. 最大问题二：safety shield 掩盖了安全学习信号

你的记录里 violations 始终为 0，projection gap 基本为 0。

表面看这是好事，但在 MARL 训练中这是危险信号。

因为如果所有动作最终都被 safety shield 修成安全动作，那么 agent 看到的是：

$$
a_t^{raw}
\rightarrow
\text{safety shield}
\rightarrow
a_t^{exec}
$$

actor 实际采样的是：

$$
a_t^{raw} \sim \pi_\theta(\cdot|o_t)
$$

但环境真实执行的是：

$$
a_t^{exec} = \Pi_{\mathcal{C}}(a_t^{raw})
$$

其中：

$$
\Pi_{\mathcal{C}}
$$

表示本地边界、DOE、AC-aware boundary、AC certificate/backoff 等投影或修复。

PPO/HAPPO 的 log probability 通常按原始动作算：

$$
\log \pi_\theta(a_t^{raw}|o_t)
$$

但是环境反馈来自执行动作：

$$
r_t = r(s_t, a_t^{exec})
$$

这会造成一个很严重的问题：

> **actor 以为自己执行了 raw action，但 reward 实际对应的是 projected action。**

这就是 credit assignment 断裂。

尤其当 projection 非线性、非光滑、有硬裁剪、有 backoff 搜索时，actor 很难知道：

1. 我原始动作哪里错了？
2. 是哪个 VPP 导致 AC 不可行？
3. 是 P 问题还是 Q 问题？
4. 是线路过载问题还是电压越限问题？
5. 是动作幅度太大，还是方向错了？
6. 是某个 VPP 的局部动作错，还是 DSO envelope 错？

所以你现在的 safety shield 可能在训练中产生了“安全假象”：

> **post-shield violation = 0，不代表 raw policy 安全。**

你必须区分：

| 指标                                  | 含义             |
| ----------------------------------- | -------------- |
| raw action violation rate           | 策略原始动作是否安全     |
| shield intervention frequency       | 安全层是否频繁修正      |
| projection gap trend                | 修正幅度是否变小       |
| AC certificate repair rate          | AC 证书是否经常需要修复  |
| post-shield violation rate          | 执行后是否安全        |
| no-shield evaluation violation rate | 没有安全层时策略是否真的安全 |

当前你只有 post-shield safety 很好，但这不能证明 MARL 学会了安全控制。

所以第二结论是：

> **safety shield 目前更像是在替 MARL 保证安全，而不是给 MARL 提供可学习的安全边界信号。**

---

# 5. 最大问题三：dispatch actor 更新过猛，PPO/HAPPO trust region 失效

你的 HAPPO dispatch update 指标非常异常。

| dispatch actor             | grad norm mean | grad norm max | approx KL abs max | KL 超标次数 |
| -------------------------- | -------------: | ------------: | ----------------: | ------: |
| commercial_multi_dispatch  |          12.80 |         13.67 |            1.8880 |       4 |
| f3_mixed_multi_dispatch    |          22.22 |         31.96 |            1.6053 |       4 |
| single_industrial_dispatch |          12.31 |         12.90 |            1.0278 |       4 |
| single_ev_hub_dispatch     |           5.30 |          7.71 |            1.2303 |       4 |
| community_multi_dispatch   |          10.79 |         12.93 |            0.8365 |       4 |

你的 target KL 是：

$$
\text{target KL} = 0.02
$$

但实际 approx KL 最大到了：

$$
0.8 \sim 1.9
$$

这比目标大了几十倍。

在 PPO/HAPPO 中，KL 过大说明：

> **新策略和旧策略差异太大，一次更新把策略推得太远。**

PPO 的核心思想是小步更新：

$$
\pi_{\theta_{new}} \approx \pi_{\theta_{old}}
$$

如果 KL 爆了，说明 clipping 或 early stopping 没有有效限制策略更新。

这会导致：

1. 策略分布跳变；
2. advantage 估计失效；
3. on-policy 数据很快变成“旧分布数据”；
4. 下一轮采样分布剧烈变化；
5. reward 曲线震荡；
6. 某些 VPP actor 学到极端动作；
7. projection/safety shield 介入更多；
8. critic 继续拟合，但 actor 不稳定。

尤其你的 dispatch actor 是多 VPP、多 DER、多约束动作，动作又会被本地边界、DOE、安全层投影。这个链条本身已经很难学，如果 KL 再大，训练会非常不稳定。

所以第三结论是：

> **dispatch actor 是当前 MARL 中最明显的优化稳定性故障点。**

具体不是说 dispatch 网络完全没用，而是：

> **dispatch actor 的 policy update 太猛，已经不满足 PPO/HAPPO 的小步策略改进假设。**

---

# 6. DSO actor 的问题：不是网络太简单，而是动作空间和学习信号太弱

你的 DSO actor 并不是普通小 MLP，而是：

* structured bipartite observation；
* sensitivity-aware attention；
* action token；
* network object token；
* sensitivity edge；
* global feature；
* flat input dim 7074。

所以不能说 DSO actor 完全是玩具模型。

但是它的问题在于：

> **DSO 可学习动作空间太窄。**

当前 DSO 最终主要输出的是每个 VPP 的 active-power preference / operating envelope 相关变量。

但对于真实 DSO 调度来说，仅靠一个 active-power preference 很难表达完整控制意图。它缺少：

1. 显式 Q/VAR 控制；
2. 显式无功容量调用；
3. locational price adder；
4. service type logits；
5. voltage support / congestion relief 类型区分；
6. safety margin allocation；
7. envelope width / confidence；
8. 对不同网络约束对象的局部分配动作。

同时你的 DSO update 指标是：

* policy loss mean 约 -0.050；
* grad norm mean 约 0.026；
* approx KL max 约 0.0108；
* target KL exceed = 0；
* entropy mean 约 5.036。

这说明 DSO actor 没有爆炸，但也没有强学习信号。

所以 DSO 的问题不是“更新过猛”，而是相反：

> **DSO actor 更新太弱，动作表达能力也不够。**

尤其因为 post-AC violation 始终为 0，DSO 看不到真实越限惩罚，安全相关梯度被 shield 吸收了。DSO 剩下能学的主要就是 curtailment proxy，而这个 proxy 又主导了 reward。

因此 DSO 当前处于一种尴尬状态：

> **网络结构看起来复杂，但动作语义太窄；安全学习信号太弱，经济 proxy 又过强。**

---

# 7. HAPPO 和 HATRPO 对比说明：算法不是根本矛盾

probe eval 显示：

| 算法     | peak reward | cloudy reward | reverseflow reward |
| ------ | ----------: | ------------: | -----------------: |
| HAPPO  |     -225.53 |       -234.46 |            -678.70 |
| HATRPO |     -219.89 |       -228.99 |            -573.63 |

HATRPO 比 HAPPO 好一些，尤其 reverseflow 也没那么差。

这说明 HATRPO 的 trust-region 思想确实缓解了一部分策略更新不稳定问题。

但是：

1. HATRPO reverseflow reward 仍然很差；
2. reverseflow 的 cost 不差但 reward 极差；
3. DSO reward 和 dispatch reward 仍然大幅负值；
4. post-AC violation 仍然为 0；
5. projection gap 仍然存在。

所以不能简单得出：

> 换成 HATRPO 就好了。

更准确的判断是：

> **HATRPO 可以缓解 actor 更新过猛，但解决不了 reward misalignment 和 shield credit assignment 问题。**

因此算法层不是第一优先级。

---

# 8. Portfolio agent 不是当前主故障点

portfolio actor 当前是 slow-loop 商业配置建议，并没有真正改变 pandapower 物理设备归属。

所以你不能把当前问题归因于 portfolio agent。

它的问题更多是建模语义层面的：

> 当前 portfolio 还不是完整意义上的 DER 物理重组智能体，而是慢周期配置建议模块。

但从训练失败的主要证据看，portfolio reward 占比很小，portfolio actor 不是导致 reward/cost 变差的核心原因。

当前主故障还是：

1. DSO reward；
2. safety shield；
3. dispatch update；
4. dispatch 动作映射。

---

# 9. 我对各模块的故障等级判断

| MARL 部分                    | 问题等级 | 判断                                                          |
| -------------------------- | ---: | ----------------------------------------------------------- |
| 环境仿真 / pandapower          |    低 | 训练正常运行，AC 安全层有效，没有挂死                                        |
| GPU / CUDA                 |    低 | 显存占用小但正常，瓶颈主要在 CPU 环境仿真                                     |
| critic/value network       |   中低 | loss 能下降，但可能在拟合错误 reward，不是主矛盾                              |
| DSO actor 网络结构             |    中 | attention 结构存在，不是玩具网络                                       |
| DSO 动作空间                   |    高 | 只有 active-power preference 类动作，表达能力不足                       |
| DSO reward                 |   极高 | curtailment / over-conservative 项占比接近 80%，严重主导训练            |
| dispatch actor 更新          |   极高 | KL 远超 target，梯度很大，PPO/HAPPO 小步更新失效                          |
| dispatch 网络结构              |    高 | Deep Sets 对多 DER、长时序、多约束调度偏弱                                |
| safety shield / projection |   极高 | post-shield violation 为 0，但 raw policy 是否安全未知               |
| portfolio actor            |   中低 | 当前不是主要训练失败来源                                                |
| algorithm choice           |    中 | HATRPO 有帮助，但不能解决根因                                          |
| evaluation design          |    高 | 需要 no-shield eval、raw action violation、projection gap trend |

---

# 10. 最核心判断：到底是 MARL 哪个部分出了问题？

我的结论是：

## 第一层根因：reward function 出问题

尤其是：

$$
R_{\text{DSO-curtailment}}
+
R_{\text{over-conservative}}
$$

占比过大，导致 MARL 主要在优化 DSO curtailment proxy，而不是完整的安全-经济调度目标。

这会导致：

1. reward 方向和 cost 方向不一致；
2. reverseflow 场景 reward 极差但 cost 不一定差；
3. DSO reward 压制 dispatch 和 portfolio；
4. 安全项由于 shield 全部变成 0，失去训练作用；
5. actor 收到的 advantage 方向不稳定。

---

## 第二层根因：safety shield 造成 credit assignment 断裂

现在是：

$$
a^{raw}
\rightarrow
\text{projection/shield}
\rightarrow
a^{exec}
\rightarrow
r
$$

但 policy gradient 仍然以：

$$
\log \pi(a^{raw}|o)
$$

来更新。

这会导致 actor 不知道真实执行动作为什么安全，也不知道原始动作哪里不安全。

所以需要记录并惩罚：

$$
d(a^{raw}, \mathcal{C}_{AC})
$$

也就是 raw action 到 AC-safe feasible set 的距离。

---

## 第三层根因：dispatch actor 的 PPO/HAPPO 更新失控

target KL 是 0.02，但 dispatch KL 到 0.8 到 1.9，这已经不是正常 PPO 小步更新。

所以 dispatch actor 正在大幅跳变，这会直接造成 reward/cost 先恶化、后震荡恢复。

---

## 第四层根因：DSO 动作空间过窄

DSO 不是没网络，而是只能通过很窄的 active power envelope 表达复杂网络约束。

这导致它无法精细地区分：

1. 电压支撑；
2. 线路拥塞；
3. 变压器过载；
4. reverseflow；
5. 局部 VPP 位置价值；
6. 有功/无功协调；
7. 安全裕度分配。

---

# 11. 你现在最应该做的修改顺序

## 第一步：先修 reward，不要先改大网络

当前不要急着加 Transformer、GRU、Q/VAR、复杂市场出清。

先把 reward 拆干净。

尤其要单独输出 reverseflow 场景下这些分项：

| 分项                               | 必须检查的问题                  |
| -------------------------------- | ------------------------ |
| DSO curtailment cost             | 是否和 over-conservative 重复 |
| over-conservative penalty        | 是否把必要安全削减也罚了             |
| safe capacity utilization reward | reverseflow 下是否定义反了      |
| flex procurement cost            | 是否压制 VPP dispatch        |
| dispatch service payment         | 是否和 DSO procurement 重复   |
| contract delivery penalty        | 是否过强                     |
| projection penalty               | 是否只看 post-shield         |
| comfort/SOC penalty              | 是否已经不是主导                 |
| AC backoff count                 | 是否隐藏了安全层频繁介入             |
| accepted candidate AC safe rate  | raw 候选动作是否本来就不可行         |

建议你把 reward 改成三级结构：

$$
R
=

R_{\text{physical}}
+
R_{\text{economic}}
+
R_{\text{learning-signal}}
$$

其中：

### 物理安全项

$$
R_{\text{physical}}
=

-\lambda_V \cdot \text{raw_voltage_violation}
-\lambda_L \cdot \text{raw_line_overload}
-\lambda_T \cdot \text{raw_trafo_overload}
-\lambda_{\text{proj}} \cdot d(a^{raw}, a^{exec})
$$

注意这里要看 raw action，不只看 shield 后结果。

### 经济项

$$
R_{\text{economic}}
=

-\lambda_c C_{\text{DSO}}
+
\lambda_p \Pi_{\text{VPP}}
--------------------------

\lambda_{\text{comfort}} C_{\text{comfort}}
$$

### 学习信号项

$$
R_{\text{learning-signal}}
=

-\lambda_{\text{shield}} I_{\text{shield}}
-\lambda_{\text{gap}} |a^{raw}-a^{exec}|^2
$$

这个项专门告诉 actor：

> 你的原始动作离可执行动作有多远。

---

## 第二步：增加 raw action 安全日志

你必须新增这些指标：

| 指标                             | 作用                     |
| ------------------------------ | ---------------------- |
| raw_voltage_violation_rate     | 判断策略原始动作是否安全           |
| raw_line_overload_rate         | 判断线路约束是否由 shield 修掉    |
| raw_trafo_overload_rate        | 判断变压器约束是否由 shield 修掉   |
| raw_reverseflow_violation_rate | 判断 reverseflow 是否真实不可控 |
| raw_action_to_projection_gap   | 判断策略是否越来越接近可行域         |
| shield_intervention_frequency  | 判断安全层是否频繁介入            |
| AC_certificate_repair_rate     | 判断 AC 修复是否经常发生         |
| no_shield_eval_violation_rate  | 判断策略脱离 shield 后是否安全    |
| post_shield_violation_rate     | 判断最终执行是否安全             |

没有这些指标，你无法证明 MARL 学到了安全调度。

---

## 第三步：控制 dispatch actor 的 KL 和梯度

你现在 dispatch KL 太大，应该立刻做这些事：

1. dispatch actor 单独降低 learning rate，例如：

$$
3 \times 10^{-4}
\rightarrow
1 \times 10^{-4}
\quad \text{或} \quad
5 \times 10^{-5}
$$

2. dispatch actor 单独设置更严格 target KL：

$$
0.02
\rightarrow
0.005 \sim 0.01
$$

3. 开启真正的 KL early stopping：

如果：

$$
D_{KL}(\pi_{old}||\pi_{new}) > 1.5 \times \text{target_KL}
$$

则提前停止当前 actor 的 PPO epoch。

4. 检查 grad norm 是 clipping 前还是 clipping 后。

你设置 max grad norm = 0.5，但记录里 grad norm mean 到 22。这里要确认：

* 如果是 clipping 前：可以接受，但说明梯度源头很不稳定；
* 如果是 clipping 后：说明 clipping 没生效或者日志记录错了。

5. 对 dispatch advantage 做 per-VPP 标准化：

$$
\hat A_{i,t}
=

\frac{A_{i,t}-\mu_i}{\sigma_i+\epsilon}
$$

不要所有 VPP 混在一起标准化，否则大 VPP 会压制小 VPP。

---

## 第四步：修 action projection 和 logprob 对应关系

你必须检查：

> PPO logprob 计算的是 raw action 还是 projected action？

如果环境执行的是 projected action，但 PPO ratio 计算的是 raw action，那么训练会偏。

理想情况下，你至少要记录：

$$
a^{raw},\quad a^{local},\quad a^{DOE},\quad a^{AC},\quad a^{exec}
$$

并分别计算：

$$
|a^{raw}-a^{local}|
$$

$$
|a^{local}-a^{DOE}|
$$

$$
|a^{DOE}-a^{AC}|
$$

$$
|a^{raw}-a^{exec}|
$$

否则你不知道到底是哪一级投影破坏了学习。

---

## 第五步：再考虑增强网络结构

等 reward 和 KL 稳定之后，再增强 dispatch actor。

推荐顺序是：

1. Deep Sets mean/max pooling
   改成 attention pooling 或 Set Transformer；

2. 增加短时序 encoder：

$$
h_t = \operatorname{GRU}(x_{t-k:t})
$$

其中 $k$ 可以先取 4 到 8 个 step。

3. 增加 DER type-specific heads：

| DER 类型        | 动作头                      |
| ------------- | ------------------------ |
| PV            | curtailment / Q support  |
| Battery       | charge / discharge       |
| EV            | charging power           |
| Flexible load | shift / curtail          |
| HVAC          | comfort-aware adjustment |

不要所有 DER 共用一个动作头。

---

# 12. 推荐你马上做的 5 个诊断实验

## 实验 1：reward ablation

分别跑：

1. 原始 reward；
2. 去掉 over-conservative penalty；
3. 去掉 curtailment cost；
4. 只保留物理安全 + 经济成本；
5. 只保留 DSO reward；
6. 只保留 dispatch reward。

看 reward/cost 是否恢复。

如果去掉 over-conservative 后 reverseflow 立刻正常，说明主因就是这个项。

---

## 实验 2：freeze DSO，只训练 dispatch

固定 DSO envelope，让 VPP dispatch 学习。

如果 dispatch 仍然 KL 爆炸，说明 dispatch actor / loss / action mapping 有问题。

---

## 实验 3：freeze dispatch，只训练 DSO

固定 VPP 响应策略，只训练 DSO。

如果 DSO reward 仍然异常，说明 DSO reward/action space 有问题。

---

## 实验 4：no-shield evaluation

训练后关闭 shield，评估：

$$
\text{raw policy} \rightarrow \text{AC power flow}
$$

看 violation rate。

如果 no-shield violation 很高，说明策略没有学会安全，只是 shield 在兜底。

---

## 实验 5：small matrix 20 episodes

不要继续等 full paper-long 跑完。

用小矩阵：

| 参数        | 建议                             |
| --------- | ------------------------------ |
| seed      | 1 到 2 个                        |
| episode   | 20                             |
| horizon   | 168 或 336                      |
| 场景        | peak / cloudy / reverseflow 分开 |
| algorithm | HAPPO + HATRPO                 |
| hparam    | base + low dispatch lr         |

这样更适合定位问题。

---

# 13. 最终判断

你的 MARL 当前不是“没有学习”，而是：

> **学习信号被 reward proxy 和 safety shield 扭曲了；dispatch actor 又因为 KL 过大而更新不稳定；DSO actor 虽然有 attention 结构，但动作空间太窄、梯度太弱。**

按严重程度排序：

1. **Reward function：最严重。**
   DSO over-conservative + curtailment cost 占比接近 80%，reverseflow 场景 reward 与 cost 明显不一致。

2. **Safety shield / projection：非常严重。**
   post-shield violation 为 0 不能证明策略安全，raw policy 是否安全未知。

3. **Dispatch actor update：非常严重。**
   KL 远超 target，PPO/HAPPO 的稳定更新假设已经失效。

4. **DSO action space：严重。**
   DSO 网络不是最弱，但输出动作太少，无法表达复杂调度。

5. **Critic：不是主因。**
   critic loss 下降说明它能拟合当前目标，但当前目标本身可能不对。

6. **Portfolio：不是当前主故障点。**
   当前 portfolio 语义还不完整，但不是导致这轮 reward/cost 异常的核心。

一句话概括：

> **这轮实验最大的问题不是 MARL 算法没跑起来，而是 reward 目标、shield 介入和 actor 更新三者没有对齐。策略梯度正在优化一个被 DSO curtailment proxy 主导、被 safety shield 修正后的间接目标，因此 reward/cost 不稳定，reverseflow 场景尤其失真。**
