# Reward v3 Shield-Credit 设计方案

生成时间：2026-06-06

参考材料：

- `outputs/current_training_loss_and_architecture_diagnostics_20260606_1409.md`
- `task2.md`
- `configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml`
- `src/vpp_dso_sim/entities/dso.py`
- `src/vpp_dso_sim/envs/reward_design.py`
- `src/vpp_dso_sim/learning/reward_config.py`

## 1. 目标

本设计不是直接改代码，而是给出下一轮 reward 修改方案。目标是让训练目标从“被 DSO 过保守/削减容量代理项主导”转为“真实 AC 安全、VPP 经济响应、shield 前后 credit assignment 可解释”的训练目标。

当前证据显示：

- HAPPO full 前 21 到 25 个 episode reward 没有改善，成本上升。
- post-AC violation 始终为 0，主要来自安全投影兜底，不能证明 raw policy 学会了安全边界。
- HAPPO dispatch actor KL 多次远超 `target_kl=0.02`。
- reward 分项审计中，`dso_over_conservative_curtailment_penalty` 和 `dso_curtailment_cost` 合计接近 80% 的有效占比。
- 当前 DSO reward 代码中 `over_conservative_curtailment_weight` 配置存在，但 v2 训练 reward 实际未使用该权重，`dso_over_conservative_curtailment_penalty` 只是未加权诊断值。

## 2. 三种可选方案

### 方案 A：只调低现有权重

做法：

- 把 `curtailment_cost_weight` 从 `0.5` 降到 `0.05-0.10`。
- 把 `safe_capacity_utilization_weight` 从 `0.2` 降到 `0.03-0.08`。
- 不改 reward 结构，只加小规模实验验证。

优点：

- 实现最快，风险低。
- 对现有训练脚本影响小。

缺点：

- 不能修复 shield 前后 credit assignment 断裂。
- 不能证明 raw action 是否安全。
- 不能解决 `over_conservative_curtailment_weight` 配置存在但训练未使用的问题。

结论：可作为临时消融，不建议作为主方案。

### 方案 B：Reward v3 Shield-Credit 重构

做法：

- 新增 `reward.version = v3_shield_credit`。
- 拆分 DSO reward 为物理安全、经济成本、shield 学习信号、容量利用四类。
- 记录并惩罚 raw action 到 executed action 的差距。
- 把 `over_conservative` 拆成“诊断值”和“训练惩罚”，并真正使用 `over_conservative_curtailment_weight`。
- 在 reverseflow 场景下避免把必要的安全收缩误判为过度保守。
- 同步降低 dispatch actor 更新幅度，避免 reward 修正后仍被 KL 爆炸破坏。

优点：

- 直接针对当前不收敛根因。
- 保留安全投影外壳，同时让 agent 为依赖外壳付出训练代价。
- 能形成论文中可解释的“安全强化学习外壳 + raw-policy 安全学习”机制。

缺点：

- 需要改 reward config、DSO reward、simulation diagnostics、trainer logging 和测试。
- 需要小矩阵实验验证，不适合直接上完整 paper-long。

结论：推荐主方案。

### 方案 C：约束强化学习或拉格朗日安全 RL

做法：

- 把电压越限、线路过载、变压器过载、reverseflow 风险作为约束。
- 使用拉格朗日乘子或 CPO/安全 PPO 类方法动态调节惩罚。
- reward 只保留经济目标，安全作为 constraint cost。

优点：

- 理论表达最清晰。
- 更接近安全强化学习论文叙述。

缺点：

- 当前代码尚未准备好约束成本 buffer、乘子更新和稳定验证。
- 直接引入会扩大变量，不适合现在排查不收敛。

结论：作为后续论文增强方向，不作为当前第一步。

## 3. 推荐方案

采用方案 B：Reward v3 Shield-Credit。

原则：

1. post-AC violation 继续作为最终安全验收指标。
2. raw action safety gap 进入训练 reward，让策略不能免费依赖安全投影。
3. DSO 过保守惩罚只在“AC 余量证明可以释放更多容量”时启用。
4. DSO、dispatch、portfolio 三类 agent 的 reward 保持角色分离。
5. critic 使用有界、稳定的训练 reward；论文/汇报保留未缩放原始成本和物理指标。

## 4. DSO Reward v3 结构

推荐将 DSO 训练 reward 写成：

```text
DSO训练奖励 =
    安全容量利用奖励
  - 经济运行成本
  - 必要削减成本
  - 过度保守惩罚
  - post-AC 安全惩罚
  - raw-action 安全惩罚
  - shield 干预惩罚
  - 平滑惩罚
```

### 4.1 经济运行成本

```text
经济运行成本 =
    flex_procurement_cost_weight * bounded(procurement_cost / 1000)
  + loss_cost_weight * bounded(line_loss_cost / 1000)
```

解释：

- `procurement_cost` 是 DSO 为采购 VPP 灵活性付出的代理成本。
- `line_loss_cost` 是线路/变压器有功损耗折算的电价成本。
- `bounded()` 表示用现有 `_bounded_training_penalty()` 做裁剪，防止极端值主导 critic。

### 4.2 必要削减成本

当前 v2：

```text
safe_capacity_utilization = clamp(mean_envelope_width_ratio, 0, 1)
over_conservative = 1 - safe_capacity_utilization
curtailment_cost = curtailment_cost_weight * over_conservative
```

问题：

- 它把所有 DOE/FR 收缩都当作削减成本。
- reverseflow 或接近电压/线路限制时，收缩可能是必要安全动作。
- `over_conservative_curtailment_weight` 没有真正进入训练 reward。

v3 推荐：

```text
必要削减成本 =
  curtailment_cost_weight * necessary_curtailment_score
```

其中：

```text
necessary_curtailment_score =
  envelope_shrink_score * AC_pressure_gate
```

`AC_pressure_gate` 表示当前网络确实接近 AC 安全边界，例如：

- 电压安全裕度低；
- 线路/变压器 loading 接近上限；
- AC certificate 需要 backoff；
- reverseflow 场景触发上游反送风险。

含义：如果 DSO 收缩是为了真实安全，不应被误判为“过度保守”。

### 4.3 过度保守惩罚

v3 推荐：

```text
过度保守惩罚 =
  over_conservative_curtailment_weight
  * over_conservative_score
  * safe_release_gate
```

其中：

```text
over_conservative_score = max(0, 1 - safe_capacity_utilization)
```

`safe_release_gate` 只在以下条件满足时为 1：

- post-AC 安全；
- voltage/line/trafo margin 高于 guard band；
- AC candidate safe rate 高；
- 过去若干步 shield 干预很少；
- 不是明确需要强收缩的 reverseflow 紧急阶段。

解释：只有“明明可以更宽，却给得太窄”才惩罚 DSO。否则会惩罚必要安全动作。

命名要求：

- `dso_over_conservative_score`：未加权诊断值。
- `dso_over_conservative_curtailment_penalty`：加权后、进入训练 reward 的惩罚。
- 不能继续用同一个名字同时表示诊断值和训练惩罚。

### 4.4 post-AC 安全惩罚

post-AC 安全惩罚保留，但不应是唯一安全学习信号。

```text
post_AC_safety_penalty =
    hard_violation_weight * bounded(post_ac_violation_count + violation_magnitude)
  + powerflow_failure_weight * post_ac_powerflow_failed
  + safety_margin_weight * bounded(guard_band_violation_score)
```

解释：

- 这是最终执行动作后的安全验收。
- 如果 safety shield 强，post-AC violation 可能长期为 0，所以它更像底线指标。

### 4.5 raw-action 安全惩罚

新增：

```text
raw_action_safety_penalty =
    raw_voltage_violation_weight * raw_voltage_violation_score
  + raw_line_overload_weight * raw_line_overload_score
  + raw_trafo_overload_weight * raw_trafo_overload_score
  + raw_reverseflow_weight * raw_reverseflow_violation_score
```

解释：

- raw action 是 actor 原始输出未经过安全投影前的动作。
- 该项让策略知道“我原始动作本来会不会造成配电网事故”。
- 它解决 post-AC violation 恒为 0 导致安全梯度消失的问题。

初始权重建议：

| 字段 | 建议初值 |
|---|---:|
| raw_voltage_violation_weight | 4.0 |
| raw_line_overload_weight | 4.0 |
| raw_trafo_overload_weight | 4.0 |
| raw_reverseflow_weight | 2.0 |

### 4.6 shield 干预惩罚

新增或强化：

```text
shield_intervention_penalty =
    shield_gap_linear_weight * raw_to_exec_gap_mw
  + shield_gap_quadratic_weight * raw_to_exec_gap_mw^2
  + shield_intervention_count_weight * I(raw_to_exec_gap_mw > epsilon)
```

解释：

- `raw_to_exec_gap_mw` 是原始动作到最终执行动作的差距。
- 如果 agent 总是乱出动作、由 shield 修正，它会受到惩罚。
- 该惩罚不替代 post-AC safety，而是给学习过程提供 credit assignment。

初始权重建议：

| 字段 | 建议初值 |
|---|---:|
| shield_gap_linear_weight | 0.5 |
| shield_gap_quadratic_weight | 2.0 |
| shield_intervention_count_weight | 0.2 |

### 4.7 DSO v3 初始权重建议

| 项 | 当前 v2 | v3 初始建议 |
|---|---:|---:|
| curtailment_cost_weight | 0.5 | 0.08 |
| over_conservative_curtailment_weight | 0.5，但当前训练未实际使用 | 0.05，并实际使用 |
| safe_capacity_utilization_weight | 0.2 | 0.05 |
| safety_margin_weight | 1.0 | 1.0 |
| hard_violation_weight | 10.0 | 10.0 |
| powerflow_failure_weight | 20.0 | 20.0 |
| projection_gap_weight | 1.0 | 1.0 |
| smoothness_weight | 0.05 | 0.05 |

这些权重不是最终论文参数，而是下一轮诊断矩阵的起点。

## 5. VPP Dispatch Reward v3 结构

dispatch agent 的目标应是：

- 在 DSO 给定边界内提供服务；
- 保持 VPP 私有利润；
- 满足合约交付；
- 避免靠 projection/shield 修正；
- 保持 DER 舒适度、SOC、退化成本可控。

推荐：

```text
VPP调度奖励 =
    private_profit_weight * private_profit_proxy
  + service_payment_weight * verified_service_payment
  + availability_payment_weight * availability_payment
  - contract_delivery_weight * contract_shortfall_mw^2
  - dispatch_projection_penalty
  - dispatch_raw_to_exec_gap_penalty
  - comfort_soc_weight * scaled_comfort_soc_penalty
  - battery_degradation_weight * battery_degradation_cost
```

当前应重点改两点。

### 5.1 加入 dispatch raw-to-exec gap

```text
dispatch_raw_to_exec_gap_penalty =
    dispatch_gap_linear_weight * dispatch_raw_to_exec_gap_mw
  + dispatch_gap_quadratic_weight * dispatch_raw_to_exec_gap_mw^2
```

解释：

- 如果 VPP dispatch actor 输出大量不可行动作，但环境不断裁剪，它会学到错误方向。
- 该项应按 VPP 归因，不能把全局 AC 修复全部平均扣给所有 VPP。

### 5.2 稳定 dispatch actor 更新

这不是 reward 本身，但必须和 reward v3 同步执行。

建议：

| 参数 | 当前 | 建议 |
|---|---:|---:|
| HAPPO actor_learning_rate | 3e-4 | DSO 可保持 3e-4，dispatch 单独降到 1e-4 或 5e-5 |
| target_kl | 0.02 | dispatch 单独 0.005 到 0.01 |
| KL 超标处理 | 记录并 early stop | 对超标 role rollback 或跳过该次更新 |
| advantage normalization | 全局/角色级 | per-VPP dispatch advantage normalization |
| grad norm 日志 | clip 后为主 | 同时记录 pre-clip 与 post-clip |

理由：当前 probe 显示多个 dispatch actor 的 KL 达到 0.6 到 1.9，已经远超 `0.02`。

## 6. Portfolio Reward 暂不作为第一优先级

当前 portfolio actor 只做慢周期商业配置建议，物理 DER membership 仍由场景事件门控。建议 v3 保持 portfolio reward 基本不变，只增加诊断解释：

- `portfolio_window_shield_intervention`
- `portfolio_window_projection_gap`
- `portfolio_window_verified_capacity`
- `portfolio_action_effective_gate`

不要在第一轮把 portfolio 改成真实 DER 重组，否则会和 reward 主问题混在一起。

## 7. 必须新增或整理的日志指标

### 7.1 raw action 指标

- `raw_action_voltage_violation_score`
- `raw_action_line_overload_score`
- `raw_action_trafo_overload_score`
- `raw_action_reverseflow_violation_score`
- `raw_action_powerflow_failed`
- `raw_action_security_pass_rate`

### 7.2 shield/projection 指标

- `raw_to_local_gap_mw`
- `local_to_doe_gap_mw`
- `doe_to_ac_aware_gap_mw`
- `ac_aware_to_ac_certified_gap_mw`
- `raw_to_exec_gap_mw`
- `shield_intervention_frequency`
- `ac_certificate_repair_rate`
- `accepted_candidate_ac_safe_rate`

### 7.3 reward 审计指标

- 每个 reward 项的 raw value；
- 每个 reward 项的 weighted value；
- 每个 reward 项占总绝对 reward 的比例；
- DSO / dispatch / portfolio 分角色占比；
- train reward 与 report-only diagnostic 分离。

## 8. 推荐新增配置文件

建议新增：

- `configs/reward_v3_shield_credit.yaml`
- `configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v3_shield_credit.yaml`
- `configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v3_shield_credit_no_raw_penalty.yaml`
- `configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v3_shield_credit_no_overconservative.yaml`
- `configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v3_shield_credit_no_shield_eval.yaml`

## 9. 推荐代码修改范围

仅在用户确认后执行。

| 文件 | 建议修改 | 原因 |
|---|---|---|
| `src/vpp_dso_sim/learning/reward_config.py` | 添加 v3 字段和 `is_v3_shield_credit` | 让 reward v3 可配置、可哈希、可复现 |
| `src/vpp_dso_sim/entities/dso.py` | 重构 `_dso_v2_diagnostics` 或新增 `_dso_v3_components` | 修复 over_conservative 权重未使用、加入 raw/shield safety |
| `src/vpp_dso_sim/envs/reward_design.py` | 增强 dispatch raw-to-exec gap 和 per-VPP projection attribution | 避免 dispatch actor 靠裁剪学习错误梯度 |
| `src/vpp_dso_sim/simulation/simulator.py` | 输出 raw/local/DOE/AC/exec 分阶段 gap | 为 reward credit 和报告提供证据 |
| `src/vpp_dso_sim/optimization/ac_security_projection.py` | 暴露 candidate safe/backoff/repair 统计 | 支撑 AC-aware shield credit |
| `src/vpp_dso_sim/learning/advanced_marl.py` | dispatch 单独 LR/KL/rollback/pre-clip grad norm | 修复 HAPPO dispatch 更新过猛 |
| `src/vpp_dso_sim/experiments/paper_training.py` | reward v3 audit 与小矩阵实验输出 | 支撑调参和论文报告 |
| `tests/test_reward_v2_minimal.py` 或新建 `tests/test_reward_v3_shield_credit.py` | 单元测试 reward 项和配置解析 | 防止权重配置无效或项名混淆 |

## 10. 验证实验计划

先小矩阵，不要直接上完整 paper-long。

| 实验 | 目的 | horizon | episodes | seed | 验收 |
|---|---|---:|---:|---:|---|
| v2_current_snapshot | 当前版本对照 | 672 | 跑到 30 即可 | 9401 | 保留现有劣化证据 |
| v3_smoke | reward v3 是否可运行 | 24 或 48 | 1 到 2 | 9401 | 无 NaN，字段齐全 |
| v3_short_happo_low_dispatch_lr | reward + dispatch 稳定性 | 168 | 20 | 9401 | dispatch KL 接近目标，reward 不继续恶化 |
| v3_short_hatrpo | HATRPO 对照 | 168 | 20 | 9401 | KL 稳定，reverseflow reward 改善 |
| v3_no_raw_penalty | raw safety 项消融 | 168 | 20 | 9401 | 验证 raw safety 是否必要 |
| v3_no_overconservative | 过保守项消融 | 168 | 20 | 9401 | 验证 DSO proxy 是否仍主导 |
| no_shield_eval | 无 shield 评估 | 168 | eval only | 9401 | raw policy 安全性随训练下降 |
| v3_multiseed_pilot | 稳定性 | 336 | 30 | 9401,9402,9403 | 均值和方差可解释 |

## 11. 验收标准

reward v3 不以“单次 reward 变好”为唯一成功标准。必须同时满足：

1. post-AC violation 仍保持 0 或接近 0。
2. raw/no-shield violation rate 随训练下降。
3. shield intervention frequency 或 raw-to-exec gap 随训练下降。
4. reward 分项中 DSO over-conservative + curtailment 不再超过 30% 到 40%。
5. dispatch KL 不再持续远超 target KL。
6. reverseflow 场景中 AC certified projection gap、backoff count 或 DSO reward 明显改善。
7. HAPPO 与 HATRPO 至少一个算法在短矩阵中呈现 reward/cost 同向改善。

## 12. 当前训练建议

当前正在跑的 v2 full 训练不建议作为主 paper-long 继续跑完。建议：

- 如果需要保留长曲线证据，可跑到 30 episode 后停止。
- 如果主要目标是优化 loss/reward，应尽快停止当前 run，切换到 v3 小矩阵。
- 不建议等完整 120 episode 后再改，因为已有足够证据证明 reward 和 dispatch KL 是主问题。

## 13. 自检

- 已覆盖 `current_training_loss_and_architecture_diagnostics_20260606_1409.md` 中的训练状态、reward 分项、HAPPO/HATRPO 对比、DSO/dispatch 架构问题。
- 已覆盖 `task2.md` 中的四个主因：reward 失衡、safety shield credit assignment、dispatch actor 更新过猛、DSO 动作空间/信号不足。
- 已指出当前代码级证据：`over_conservative_curtailment_weight` 配置存在但 DSO v2 训练 reward 未实际使用。
- 本文档只提出方案，不修改源代码。
