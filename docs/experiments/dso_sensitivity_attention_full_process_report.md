# DSO Sensitivity Attention 改造与实验完整报告

Updated: 2026-05-28 Asia/Shanghai

## 1. 报告范围与结论边界

本报告记录本轮 DSO agent 改造的工程过程、算法结构变化、实验验证过程、loss 曲线与归档产物位置。

需要特别说明：这里的 256-step loss 曲线是 `sensitivity_attention_v1` DSO actor 的行为克隆 warm-start sanity 曲线，不是 paper-long 级别强化学习收敛曲线。它证明新结构 actor 可以稳定拟合规则老师输出、无 NaN/Inf、梯度可控；它不能单独证明 HAPPO/MARL 在长周期真实训练中已经收敛。

## 2. 改造前核心问题

| 问题 | 原始风险 | 本轮处理方式 |
|---|---|---|
| DSO envelope 主要依赖规则逻辑 | DSO agent 学不到 ActionUnit 与网络约束之间的空间敏感关系 | 新增 ActionUnit x NetworkObject sensitivity-aware bipartite attention actor |
| FR/DOE 与 AC 潮流安全边界不一致 | projection gap 不能直接代表真实潮流安全 | 在 DSO envelope 路径加入有限差分 AC sensitivity、post-AC violation 相关 refresh 触发与 safe decoder |
| 神经网络如果直接输出 hard bound 风险过高 | 可能破坏 FR/DOE 安全边界 | 神经网络只输出 `center_ratio`、`width_ratio`、`direction_probs`、`guidance_strength_lambda`，hard bound 仍由 safe decoder 和 FR/DOE 约束 |
| reward comfort 项可能压倒网络安全项 | 策略可能更关心舒适/收益而忽视安全 | VPP preferred-range bonus 改为 lambda/width/effectiveness gated，降低错误激励 |
| legacy baseline 容易被新结构破坏 | 后续无法做可复现实验对照 | 保留 `rule_v0`、legacy flat observation、legacy MLP、原有 MARL 家族 |
| sensitivity 计算代价可能过高 | 每步全量 AC perturbation 会拖慢 paper-long | 增加 raw cache、active slice、TTL/update-period/grid-drift/projection-gap 触发、priority ActionUnit partial refresh |

## 3. 改造后的算法框架

本轮将 DSO operating envelope 路径拆成两条版本化路径：

| 路径 | 配置 | 作用 |
|---|---|---|
| 规则基线 | `configs/baseline_rule_v0.yaml` | 原始规则 envelope，作为稳定 baseline |
| legacy MLP 基线 | `configs/happo_legacy_mlp.yaml` | 保留 legacy flat observation / MLP 路径，便于对照 |
| sensitivity attention v1 | `configs/happo_sensitivity_attention_v1.yaml` | 新 DSO 结构化 actor，读取 ActionUnit、NetworkObject、SensitivityEdge |

核心执行链路：

1. VPP/DER 当前可调能力生成 ActionUnit。
2. DSO 根据电压、线路 loading、变压器 loading 等选择关键 NetworkObject。
3. 使用 AC finite-difference 计算 ActionUnit 对 NetworkObject 的灵敏度边。
4. structured observation 编码为 bipartite graph-like tokens。
5. DSO actor 输出每个 ActionUnit 的中心、宽度、方向概率、引导强度。
6. safe decoder 将神经网络输出限制在 FR/DOE hard bound 内，形成 preferred operating range。
7. simulator 将 DSO envelope 下发给 VPP/DER 调度链路。
8. 日志记录 actor output、decoded envelope、sensitivity cache、allocation weights、projection trace、reward components。

## 4. 主要代码改造位置

| 文件/目录 | 改造内容 |
|---|---|
| `src/vpp_dso_sim/dso/envelope/schemas.py` | 新增 ActionUnit、NetworkObject、SensitivityEdgeTensor、StructuredDSOObservation 等结构化 schema |
| `src/vpp_dso_sim/dso/envelope/rule_v0.py` | 将旧规则封装为可切换 baseline |
| `src/vpp_dso_sim/dso/envelope/safe_decoder.py` | 把 actor 输出映射为 FR/DOE hard bound 内的 preferred range |
| `src/vpp_dso_sim/dso/envelope/sensitivity_attention_v1.py` | 新 DSO envelope policy 主入口 |
| `src/vpp_dso_sim/dso/envelope/policy_switch.py` | 根据配置在 `rule_v0` 和 `sensitivity_attention_v1` 之间切换 |
| `src/vpp_dso_sim/dso/sensitivity/finite_difference.py` | AC finite-difference sensitivity tensor，记录 perturbation allocation weights |
| `src/vpp_dso_sim/dso/sensitivity/cache.py` | sensitivity raw cache、active slice、refresh decision、partial priority refresh merge |
| `src/vpp_dso_sim/dso/observation/structured_bipartite.py` | DSO structured observation 编码 |
| `src/vpp_dso_sim/dso/models/bipartite_attention_actor.py` | ActionUnit x NetworkObject 双部注意力 actor |
| `src/vpp_dso_sim/dso/models/structured_happo_actor.py` | HAPPO 兼容的 structured DSO Gaussian actor |
| `src/vpp_dso_sim/envs/reward_design.py` | preferred-range reward 改为安全引导强度/宽度/有效性门控 |
| `src/vpp_dso_sim/learning/advanced_marl.py` | HAPPO structured actor、target KL、normalization、nan guard、frozen eval |
| `scripts/run_smoke.py` | 2-step smoke rollout |
| `scripts/run_short_train.py` | 256-step BC warm-start sanity |
| `scripts/agent_hooks/smoke_training.sh` | 一键运行 baseline、legacy、v1 smoke 和 256-step short train |

## 5. 神经网络模型与 loss 说明

`BipartiteSensitivityDSOActor` 的输入不是一条普通 flat state，而是结构化对象：

| 输入 | 含义 |
|---|---|
| `global_features` | DSO 全局时间/网络运行摘要 |
| `action_tokens` | 每个 ActionUnit 的可调范围、PCC、bus 等公开调度信息 |
| `object_tokens` | 被 DSO 关注的 bus/line/trafo 网络对象 |
| `sensitivity_edges` | ActionUnit 对网络对象的 AC finite-difference 灵敏度 |
| `action_mask` / `object_mask` / `edge_mask` | padding 与可用对象 mask |

actor 输出：

| 输出 | 中文解释 | 是否直接作为硬约束 |
|---|---|---|
| `center_ratio` | preferred range 中心在 hard range 内的位置 | 否 |
| `width_ratio` | preferred range 宽度比例 | 否 |
| `direction_probs` | 吸收/平衡/注入三个方向的概率 | 否 |
| `guidance_strength_lambda` | DSO 引导强度 | 否 |

safe decoder 之后才得到可执行的 preferred range；`p_hard_min_mw` / `p_hard_max_mw` 仍来自 FR/DOE hard bound。

256-step 行为克隆 warm-start 使用规则 envelope 作为老师目标。loss 公式在 `src/vpp_dso_sim/experiments/dso_sensitivity_attention.py` 中：

```text
bc_loss = center_loss + width_loss + 0.25 * direction_loss
```

其中：

| loss 项 | 中文解释 |
|---|---|
| `center_loss` | actor 输出的中心位置与规则老师中心位置的 MSE |
| `width_loss` | actor 输出的宽度比例与规则老师宽度比例的 MSE |
| `direction_loss` | actor 对吸收/平衡/注入方向分类的负对数似然 |
| `grad_norm` | 反向传播后统计的梯度范数；训练中使用 `clip_grad_norm_(..., 0.5)` 控制梯度 |

## 6. 最新实验与验证过程

### 6.1 全量测试

命令：

```bash
./.venv-server/bin/python -m pytest -q
```

结果：

- 退出码：`0`
- 失败测试：无
- 警告：1 个 Jupyter path deprecation warning，和本项目 DSO 改造无直接关系

补充收集命令：

```bash
./.venv-server/bin/python -m pytest --collect-only -q
```

结果：收集到 160 个测试用例。

### 6.2 smoke training hook

命令：

```bash
bash scripts/agent_hooks/smoke_training.sh
```

该 hook 顺序执行：

| 顺序 | 配置 | 目的 | 结果摘要 |
|---|---|---|---|
| 1 | `configs/baseline_rule_v0.yaml` | 验证原始规则 baseline 仍可运行 | 2 steps，`constraint_violations = 0`，无 NaN/Inf |
| 2 | `configs/happo_legacy_mlp.yaml` | 验证 legacy flat/MLP 对照路径仍可运行 | 2 steps，`constraint_violations = 0`，无 NaN/Inf |
| 3 | `configs/happo_sensitivity_attention_v1.yaml` | 验证新 v1 DSO envelope policy 可运行 | 2 steps，`constraint_violations = 0`，无 NaN/Inf |
| 4 | `configs/happo_sensitivity_attention_v1.yaml` | 验证新 actor 256-step BC warm-start | 256 steps 完成，loss 从 `0.5577945709` 降到 `0.0003264844` |

### 6.3 v1 smoke step 指标

归档文件：

```text
docs/experiments/dso_sensitivity_attention_artifacts/current_smoke_step_metrics.csv
```

本轮 2-step v1 smoke：

| step | power flow converged | total_cost | total_reward |
|---|---|---:|---:|
| 0 | True | 1.6619230670 | 2.2481686539 |
| 1 | True | 1.6619230670 | 2.2481686539 |

这说明 2-step sanity 下潮流执行链路正常、reward component 可记录。该结果不等同于长周期训练 reward 已提升。

## 7. Loss 曲线与数值归档

### 7.1 原始文件

| 文件 | 作用 |
|---|---|
| `docs/experiments/dso_sensitivity_attention_artifacts/current_bc_loss_metrics.csv` | 256-step BC loss 原始 CSV |
| `docs/experiments/dso_sensitivity_attention_artifacts/current_bc_loss_curve.svg` | 由 CSV 生成的 log10 loss 曲线 |
| `docs/experiments/dso_sensitivity_attention_artifacts/current_bc_short_train_summary.json` | 训练摘要 |
| `outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0/dso_sensitivity_attention_actor.pt` | 本轮 short train 生成的 actor checkpoint |

### 7.2 loss 下降摘要

| 指标 | 数值 |
|---|---:|
| 更新步数 | 256 |
| 初始 `bc_loss` | 0.5577945709 |
| 最终 `bc_loss` | 0.0003264844 |
| 最小 `bc_loss` | 0.0003264844 |
| 平均 `bc_loss` | 0.0074012345 |
| 最终/初始比例 | 0.0005853129 |

### 7.3 loss 采样点

| update_step | bc_loss | center_loss | width_loss | direction_loss | grad_norm |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.5577945709 | 0.0886369422 | 0.1633713245 | 1.2231453657 | 6.5764207840 |
| 32 | 0.0031630602 | 0.0000260796 | 0.0000036687 | 0.0125332475 | 0.0437851846 |
| 64 | 0.0015608658 | 0.0000128911 | 0.0000006573 | 0.0061892695 | 0.0193774849 |
| 96 | 0.0010502836 | 0.0000006806 | 0.0000004688 | 0.0041965363 | 0.0109868804 |
| 128 | 0.0007735977 | 0.0000002497 | 0.0000001691 | 0.0030927155 | 0.0080269920 |
| 160 | 0.0005970226 | 0.0000001667 | 0.0000000730 | 0.0023871320 | 0.0062221186 |
| 192 | 0.0004761301 | 0.0000000913 | 0.0000000302 | 0.0019040342 | 0.0049693864 |
| 224 | 0.0003893157 | 0.0000000564 | 0.0000000113 | 0.0015569921 | 0.0040715965 |
| 255 | 0.0003264844 | 0.0000000361 | 0.0000000041 | 0.0013057766 | 0.0034202593 |

解释：

- 前 32 步 loss 快速下降，说明新 actor 能快速学习规则老师的中心、宽度和方向输出。
- 32 步之后 loss 继续缓慢下降，主要剩余误差来自方向分类项。
- `center_loss` 和 `width_loss` 后期已经接近 0，说明 preferred range 的连续参数拟合稳定。
- `grad_norm` 从约 6.58 降到约 0.0034，说明这条 sanity 曲线没有梯度爆炸迹象。

## 8. 归档产物索引

小型、可读、适合报告使用的产物已复制到：

```text
docs/experiments/dso_sensitivity_attention_artifacts/
```

关键文件：

| 文件 | 用途 |
|---|---|
| `current_baseline_rule_v0_smoke_summary.json` | rule_v0 baseline smoke 摘要 |
| `current_happo_legacy_mlp_smoke_summary.json` | legacy MLP/flat 路径 smoke 摘要 |
| `current_smoke_summary.json` | sensitivity_attention_v1 smoke 摘要 |
| `current_dso_actor_outputs.csv` | v1 DSO actor 原始输出 |
| `current_decoded_operating_envelope.csv` | safe decoder 后的 preferred range |
| `current_dso_operating_envelope.csv` | DSO envelope 全量诊断字段 |
| `current_bc_loss_metrics.csv` | 256-step loss 曲线原始数据 |
| `current_bc_loss_curve.svg` | 256-step loss 曲线图 |
| `current_bc_short_train_summary.json` | 256-step short train 摘要 |
| `current_happo_training_summary.json` | structured HAPPO minimal training 摘要 |
| `current_happo_update_metrics.csv` | HAPPO update loss/KL/entropy/grad norm |
| `current_happo_frozen_eval_summary.json` | frozen deterministic evaluation 摘要 |

大型原始输出仍保留在：

```text
outputs/dso_sensitivity_attention/
```

`outputs/` 通常被 `.gitignore` 忽略，适合保存大实验原始产物；`docs/experiments/dso_sensitivity_attention_artifacts/` 只保存小型报告级证据。

## 9. 当前能证明什么，不能证明什么

已经能证明：

1. `rule_v0`、legacy MLP、`sensitivity_attention_v1` 三条路径都能跑通最小 smoke。
2. 新 DSO actor 不直接输出 hard bound，仍经过 safe decoder。
3. 新 structured observation 不泄露 private true cost / oracle cost / comfort preference / private SOC internals。
4. finite-difference sensitivity、cache refresh、priority partial refresh、allocation weights 已有单测和 smoke 证据。
5. 256-step BC warm-start 曲线稳定下降，无 NaN/Inf。
6. 全量 pytest 本轮退出码为 0。

尚不能证明：

1. paper-long HAPPO/MARL 已经收敛。
2. 新 DSO actor 在所有 holdout 场景都优于 rule_v0。
3. sensitivity cache refresh 策略在长周期下已经达到最优计算速度。
4. reward 设计在 paper-long 下没有新的平台期或 reward hacking。
5. 多 seed 统计显著性已经满足论文结论要求。

## 10. 下一步建议

paper-long 前建议按以下顺序继续：

1. 先跑 24-step pilot，对比 `rule_v0`、legacy MLP、`sensitivity_attention_v1`。
2. 再跑 96-step pilot，检查 reward 分项、projection gap、shield intervention、KL、entropy、grad norm。
3. 跑三个 ablation：`no_sensitivity_edges`、`no_action_self_attention`、`no_width_penalty`。
4. 对每个 checkpoint 做 frozen deterministic eval，确认评估时没有探索噪声。
5. 最后进入 paper-long，多 seed 保存曲线和安全指标。
