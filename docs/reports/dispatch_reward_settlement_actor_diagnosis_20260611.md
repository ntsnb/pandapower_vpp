# Dispatch Reward / Settlement / Actor 诊断报告

生成日期：2026-06-11

## 1. 本次诊断对象

- 项目目录：`/mnt/sda/home/niutiansen/panda power/pandapower-vpp-dso-sim`
- 旧实验目录：`outputs/paper_training_long_v3_1_happo_only_gpu_20260610_fresh`
- 重点 episode：`happo_base_train_mixed_seed_9401/train/happo_dispatch_private_profit_trace_episode_0015.csv`
- 归因明细：`outputs/paper_training_long_v3_1_happo_only_gpu_20260610_fresh/runs/happo_base_train_mixed_seed_9401/train/reports/happo_dispatch_private_profit_trace_episode_0015_negative_reason_steps.csv`
- 归因报告：`outputs/paper_training_long_v3_1_happo_only_gpu_20260610_fresh/runs/happo_base_train_mixed_seed_9401/train/reports/happo_dispatch_private_profit_trace_episode_0015_absorption_root_cause_report.md`
- 私有利润报告：`outputs/paper_training_long_v3_1_happo_only_gpu_20260610_fresh/reports/dispatch_private_profit_report_happo_episode_0015.md`

## 2. 核心结论

旧实验中 VPP dispatch reward 大幅为负，主因不是 GPU、不是 AC 潮流安全成本，也不是电价本身过高。

更准确的结论是：

1. 旧 trace 缺少完整 DER-level settlement 分项。
2. `private_profit_proxy` 与可见的 `energy_market_revenue - der_operation_cost` 存在约 `-287.38 万` 的巨大残差。
3. 该残差高度集中在带 HVAC 的 VPP。
4. 结合代码公式，最可信的原因是旧 `operational_surplus/private_profit` 口径把 raw comfort / unserved service-quality penalty 直接混入经济利润，造成量纲混用。
5. 旧 episode 里 `actual_delta_p_mw` 全部为 0，不能证明 dispatch 神经网络动作把 VPP 调坏；负 reward 更多来自旧结算口径、场景外生状态、DOE/物理投影后动作未实际落地。

## 3. 旧 episode 0015 的数值证据

| 指标 | 数值 |
|---|---:|
| 行数 | 4704 |
| VPP 数 | 7 |
| `private_profit_proxy` 总和 | -2,876,729.16 |
| 可见电费运行差额总和 | -2,953.25 |
| `private_profit_proxy - 可见电费运行差额` 残差 | -2,873,775.91 |
| `accepted_delta_p_mw` 非零比例 | 6.68% |
| `actual_delta_p_mw` 非零比例 | 0.00% |
| `actual_delta_p_mw` 最大绝对值 | 0.0 |

按 VPP 汇总：

| VPP | private profit | 可见电费运行差额 | 残差 |
|---|---:|---:|---:|
| `vpp_single_campus` | -860,481.33 | -513.17 | -859,968.16 |
| `vpp_f3_mixed_multi` | -706,170.36 | -377.12 | -705,793.24 |
| `vpp_community_multi` | -663,099.00 | -226.72 | -662,872.27 |
| `vpp_residential_multi` | -643,121.46 | -363.25 | -642,758.21 |
| `vpp_single_ev_hub` | -2,512.66 | -38.00 | -2,474.65 |
| `vpp_single_industrial` | -991.17 | -991.17 | 0.00 |
| `vpp_commercial_multi` | -353.18 | -443.81 | 90.63 |

解释：

- `vpp_single_industrial` 基本没有残差，负值可以由可见电费/运行成本解释。
- `vpp_single_campus`、`vpp_f3_mixed_multi`、`vpp_community_multi`、`vpp_residential_multi` 的残差远大于电费和运行成本。
- 这些 VPP 都包含 HVAC 聚合设备，因此与 raw comfort penalty 混入利润口径的假设一致。

## 4. 为什么不是电价主因

旧 episode 中市场电价均值约 80，单步时长 0.25 小时。VPP 单步功率通常在几十 kW 到一百多 kW 量级，因此单步电费通常是 0.x 到数元量级。

但最严重 step 中，`private_profit_proxy` 单步约 -2500 到 -2900，而可见电费运行差额通常只有 -1 到 +1。

这说明：

- 电价可以解释小额亏损；
- DER 运行成本可以解释部分亏损；
- 但不能解释几千级单步亏损；
- 几千级单步亏损来自旧 private profit 口径中的隐藏项。

## 5. reward / settlement 已做的订正

已把经济利润和服务质量惩罚拆开：

1. 经济运行盈余：

   `economic_operational_surplus = export_revenue_total + evcs_user_revenue_total - import_energy_cost_total - der_operating_cost_total - battery_degradation_cost_total`

2. 服务质量惩罚：

   `service_quality_penalty_total = comfort_cost_total + unserved_penalty_total`

3. 质量调整盈余：

   `quality_adjusted_operational_surplus = economic_operational_surplus - service_quality_penalty_total`

4. 主训练口径：

   `operational_surplus/private_profit` 默认使用 `economic_operational_surplus`，不再直接扣 raw service-quality penalty。

关键文件：

- `src/vpp_dso_sim/simulation/settlement.py`
- `src/vpp_dso_sim/envs/reward_design.py`
- `src/vpp_dso_sim/learning/reward_trace.py`
- `src/vpp_dso_sim/learning/advanced_marl.py`
- `src/vpp_dso_sim/learning/hatrpo.py`

## 6. 完整 settlement 分项落盘

新 trace 和 step metrics 会落盘以下核心字段：

- `economic_operational_surplus`
- `quality_adjusted_operational_surplus`
- `service_quality_penalty_total`
- `export_revenue_total`
- `pv_export_revenue_total`
- `mt_export_revenue_total`
- `storage_discharge_revenue_total`
- `evcs_user_revenue_total`
- `import_energy_cost_total`
- `evcs_wholesale_cost_total`
- `storage_charge_cost_total`
- `hvac_energy_cost_total`
- `flex_energy_cost_total`
- `unclassified_import_cost_total`
- `der_operating_cost_total`
- `battery_degradation_cost_total`
- `comfort_cost_total`
- `unserved_penalty_total`
- `visible_energy_minus_operation_cost`
- `market_energy_margin_total`
- `private_profit_vs_visible_energy_residual`

这样下一次 paper-long 训练后，可以直接判断：

- 是 EVCS 收入不足；
- 是购电成本过高；
- 是 DER 运行成本过高；
- 是 HVAC comfort 质量风险；
- 是旧口径残差；
- 还是 dispatch actor 输出不可行动作。

## 7. Dispatch Actor 网络结构判断

当前 HAPPO 使用的 dispatch actor 不是简单玩具 MLP。

当前 paper-long 场景统计：

| 项 | 数值 |
|---|---:|
| VPP 数 | 7 |
| 最大 DER 数 / VPP | 6 |
| dispatch observation 维度 | 172 |
| dispatch action 维度 | 7 |
| HAPPO dispatch actor 参数量 | 407,650 |
| dispatch encoder 类型 | DeepSetDispatchEncoder |
| 是否共享 dispatch policy | 是 |

结构上，dispatch actor 使用：

- VPP 级 context encoder；
- DER token encoder；
- masked token mean pooling；
- masked token max pooling；
- DER 数量比例特征；
- aggregate action head；
- DER action head。

所以它不是“只有几层小 MLP 的玩具实现”。但是它仍然有局限：

1. 它是 DeepSet 聚合，不是图神经网络，不能直接建模电气拓扑距离、线路灵敏度、PCC 到节点的潮流路径。
2. 它不是 RNN/Transformer 时序模型，对 672 step 长时序主要依赖当前观测和少量状态字段。
3. 它是共享 dispatch policy，虽然输入包含 DER token 和 VPP context，但不同 VPP 的结构异质性可能仍然表达不足。
4. 旧 episode 中实际动作没有落地，网络再复杂也无法从这些 step 中形成有效因果学习信号。

因此，当前证据不支持“VPP reward 过低主要是 dispatch actor 网络太简单”这个结论。

更合理的排序是：

1. reward/settlement 口径错误是最高优先级。
2. 动作落地率为 0 是第二优先级，需要检查 action decoder、投影、baseline/target 关系。
3. dispatch actor 表达能力是第三优先级，后续可增强为 topology/sensitivity-aware 或 recurrent encoder，但不是旧实验负 reward 的主要证据。

## 8. 对下一轮实验的要求

下一轮 paper-long 前必须确认：

1. 使用当前修正后的代码重新启动实验，旧 episode 0015 不能作为修正后 reward 的收敛证据。
2. 每个 episode 的 `*_dispatch_private_profit_trace_episode_XXXX.csv` 必须包含完整 settlement 字段。
3. 每个 episode 运行后生成动态 reward cards 和 dispatch attribution report。
4. 重点看 `private_profit_vs_visible_energy_residual`：
   - 如果仍然很大，说明还有隐藏口径混入；
   - 如果接近 0，说明 private profit 已经可由完整 settlement 解释。
5. 重点看 `actual_delta_p_mw`：
   - 如果长期为 0，说明 dispatch actor 动作仍未真正作用到 DER/VPP，算法学习会非常弱；
   - 如果非零且 reward 随动作变化，才说明 dispatch actor 开始产生有效控制。

## 9. 验证记录

已通过测试：

```text
tests/test_dispatch_absorption_reward_report.py
tests/test_reward_v3_1_market_safety.py
tests/test_reward_trace.py
tests/test_hasac_happo.py::test_happo_reward_v2_step_metrics_cover_reward_and_security_columns
tests/test_hatrpo_training.py::test_hatrpo_reward_v2_step_metrics_cover_reward_and_security_columns

15 passed
```
