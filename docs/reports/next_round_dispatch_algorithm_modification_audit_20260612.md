# 新一轮 Dispatch 算法修改审计报告

生成日期：2026-06-12

## 1. 本轮任务边界

本轮按 `docs/tasks/next_round_dispatch_algorithm_modification_plan.md` 做了批判性筛选。结论是：当前最应该先做的是 **action landing 审计闭环** 和 **settlement/reward 报告闭环**，而不是立刻替换成更复杂的 GNN/Transformer dispatch actor。

原因是上一次长实验的关键证据仍然指向两个基础问题：

- 旧 trace 没有完整 settlement 分项，`private_profit_proxy` 与可见电费运行差额存在巨大残差。
- 旧 trace 没有完整 action landing 字段，无法判断动作在哪一级被投影、覆盖或没有执行。

因此本轮采纳计划中的 Phase 1 和报告/gate 部分；暂缓完整网络重构。

## 2. 上一次实验审计结果

审计对象：

- `outputs/paper_training_long_v3_1_happo_only_gpu_20260610_fresh/runs/happo_base_train_mixed_seed_9401/train/happo_dispatch_private_profit_trace_episode_0015.csv`

关键结果：

| 指标 | 数值 |
|---|---:|
| trace 行数 | 4704 |
| settlement 分项完整 | 否 |
| action landing 分项完整 | 否 |
| `private_profit_proxy` 合计 | -2876729.157441 |
| 可见电费运行差额合计 | -2953.252273 |
| `accepted_delta_p_mw` 非零率 | 0.066752 |
| `actual_delta_p_mw` 非零率 | 0.000000 |

更新后的历史审计报告：

- `outputs/paper_training_long_v3_1_happo_only_gpu_20260610_fresh/runs/happo_base_train_mixed_seed_9401/train/reports/happo_dispatch_private_profit_trace_episode_0015_absorption_root_cause_report.md`
- `outputs/paper_training_long_v3_1_happo_only_gpu_20260610_fresh/runs/happo_base_train_mixed_seed_9401/train/reports/happo_dispatch_private_profit_trace_episode_0015_negative_reason_steps.csv`

判断：

- 旧实验不能证明 dispatch actor 已经有效学习，因为实际动作变化长期为 0。
- 旧实验也不能继续用于评价新 reward，因为它缺完整 settlement 分项。
- 旧实验只能作为“为什么必须补审计字段”的反例。

## 3. 对修改计划的取舍

| 计划项 | 本轮处理 | 理由 |
|---|---|---|
| 完整 settlement 分项落盘 | 已继承并补强报告 | 之前已完成主代码修正，本轮把报告入口补齐。 |
| action landing 审计字段 | 已采纳 | 这是判断 dispatch actor 是否真正控制 DER/VPP 的前置条件。 |
| paper-long 前 gate | 部分采纳 | 先在报告中输出 `actual_delta_nonzero_rate` 和 `action_landing_ratio`，不在所有短实验中硬阻断。 |
| 非 AC 约束不硬投影 | 暂缓大改 | 当前还需要更强证据区分 FR/DOE、AC shield、baseline override 的实际占比。 |
| Type-specific heads | 暂缓 | 旧证据不能证明网络结构是第一根因；先确保动作落地。 |
| Temporal / sensitivity actor | 暂缓 | 应在 action landing 和 reward 口径稳定后再做，否则复杂网络无法解决 credit assignment 断裂。 |
| Critic 输入升级 | 暂缓 | 需要先让 step metrics 稳定记录 landing 字段，再评估 critic 是否缺这些输入。 |

## 4. 本轮实际修改

### 4.1 环境层 action landing 审计

修改文件：

- `src/vpp_dso_sim/envs/multi_agent_env.py`
- `src/vpp_dso_sim/envs/reward_design.py`

新增内容：

- `baseline_p_mw` 明确记录为 dispatch 动作前的 VPP 功率。
- 从 simulator 的 `projection_trace` 聚合每个 VPP 的执行阶段：
  - `raw_target_p_mw`
  - `decoded_target_p_mw`
  - `device_feasible_target_p_mw`
  - `pre_ac_target_p_mw`
  - `ac_projected_target_p_mw`
  - `ac_certified_target_p_mw`
  - `actual_target_p_mw`
- 计算每个阶段相对 baseline 的 delta：
  - `decoded_delta_p_mw`
  - `actual_delta_p_mw`
  - `pre_ac_delta_p_mw`
  - `ac_projected_delta_p_mw`
- 计算 gap 和落地指标：
  - `raw_to_device_gap_mw`
  - `device_to_ac_gap_mw`
  - `ac_to_actual_gap_mw`
  - `accepted_to_actual_gap_mw`
  - `actual_delta_nonzero_flag`
  - `action_landing_ratio`
  - `action_landing_drop_reason`
  - `action_landing_drop_reason_code`

### 4.2 训练器输出同步

修改文件：

- `src/vpp_dso_sim/learning/advanced_marl.py`
- `src/vpp_dso_sim/learning/hatrpo.py`
- `src/vpp_dso_sim/learning/reward_trace.py`

影响：

- HAPPO step metrics 写出 action landing 平均字段。
- HATRPO step metrics 写出同样字段。
- HASAC step metrics 也补了基本 action landing 字段，避免实验矩阵字段不一致。
- dispatch private profit trace 会逐 agent/逐 step 写出 action landing 分项和公式。

### 4.3 报告脚本增强

修改文件：

- `scripts/analyze_dispatch_absorption_rewards.py`
- `scripts/watch_dispatch_profit_episode_report.py`

新增能力：

- 识别 `action_landing_trace_complete`。
- 报告 `actual_delta_nonzero_rate`。
- 报告 `mean_action_landing_ratio`。
- 按 `action_landing_drop_reason` 汇总：
  - `landed`
  - `local_physical_limit`
  - `dso_envelope_clip`
  - `ac_shield_projection`
  - `baseline_override`
  - `not_applied_bug`
  - `legacy_trace_missing_landing_audit`
- `watch_dispatch_profit_episode_report.py` 现在同时支持 paper-long 的 `runs/*/train/` trace 和短实验根目录 trace。

## 5. 新 smoke 验证结果

HAPPO smoke：

- 输出目录：`outputs/test_action_landing_happo_smoke_20260612_baselinefix`
- horizon：2 steps
- `action_landing_ratio` 平均值：0.998751
- `actual_delta_nonzero_flag` 平均值：1.0
- `accepted_to_actual_gap_mw` 合计：约 7.93e-18
- `device_to_ac_gap_mw` 合计：0.0

HATRPO smoke：

- 输出目录：`outputs/test_action_landing_hatrpo_smoke_20260612_baselinefix`
- horizon：2 steps
- `action_landing_ratio` 平均值：0.996746
- `actual_delta_nonzero_flag` 平均值：1.0
- `accepted_to_actual_gap_mw` 合计：约 2.97e-18
- `device_to_ac_gap_mw` 合计：0.0

HAPPO 新报告：

- `outputs/test_action_landing_happo_smoke_20260612_baselinefix/reports/dispatch_private_profit_report_happo_episode_0000.md`
- `outputs/test_action_landing_happo_smoke_20260612_baselinefix/reports/happo_dispatch_private_profit_trace_episode_0000_absorption_root_cause_report.md`
- `outputs/test_action_landing_happo_smoke_20260612_baselinefix/reports/dispatch_private_profit_agent_summary_happo_episode_0000.csv`

结论：

- 新字段不是只存在于内存对象，已经实际落盘。
- 在 2-step smoke 中，动作可以真实落到 VPP 执行功率。
- 这不等于 paper-long 已经会收敛，只说明“旧实验 actual_delta 全 0 的审计盲区”已被修复。

## 6. 仍需谨慎的问题

1. 2-step smoke 不能替代 96-step 或 672-step 训练。
2. 当前没有立刻重构 dispatch actor；DeepSet actor 是否足够还要用 action landing 完整 trace 后再判断。
3. 当前没有把 `actual_delta_nonzero_rate < 10%` 做成所有实验的强制异常，因为短实验、baseline 和纯评估阶段可能合法为低动作。
4. 当前只把 landing gap 进入报告和已有 `dispatch_projection_penalty` 体系，没有新增额外 reward 项，避免一次性改动过多导致训练不稳定。

## 7. paper-long 前建议 gate

下一轮进入 paper-long 前，建议先跑 96-step、1-3 episode 的 HAPPO/HATRPO 短实验，并检查：

| 指标 | 建议阈值 |
|---|---:|
| `action_landing_trace_complete` | 1 |
| `settlement_trace_complete` | 1 |
| `actual_delta_nonzero_rate` | > 0.10 |
| `mean_action_landing_ratio` | > 0.30 |
| `private_profit_vs_visible_energy_residual_ratio` | < 5% |
| `device_to_ac_gap_mw` | 不应长期主导 |
| `ac_to_actual_gap_mw` | 不应长期主导 |

如果这些指标失败，不建议直接启动 paper-long。
