# 模型结构与训练失败原因审查报告

生成日期：2026-05-09  
实验目录：`outputs/paper_training_long_current`  
相关代码更新：reward 可配置再平衡、HASAC entropy override 修复、reward 权重测试

## 1. 核心结论

当前失败不是单一因素导致。按证据强度排序：

1. **reward / penalty 设计是首要问题。**  
   `total_cost` 被 HVAC comfort penalty 主导，导致策略优化目标偏离“配电网-虚拟电厂经济安全调度”。抽查最坏 HASAC reverseflow run：`total_cost=2692741.25`，其中 `comfort_violation_penalty=2661920.79`，`procurement_proxy_cost=28205.61`，电压/线路惩罚只有几十量级。

2. **安全投影不是 AC 可行域，是第二关键问题。**  
   现有 projection 主要处理 DER 边界、FR/DOE 聚合上下界和 residual repair，不能保证潮流后的低电压、过电压、线路 loading 均安全。reverseflow holdout 中 RL 多次触发 bus 10/11/12/20/21 低电压和 bus 73/74/82 高电压。

3. **训练设计与泛化协议不够强，是第三关键问题。**  
   训练只用 `train_mixed`，评估包含 `holdout_peak/cloudy/reverseflow`。策略训练 reward 有提升，但 holdout 全部弱于 `rule_based`。这说明训练目标、场景覆盖和 checkpoint 选择还没有对齐最终评估。

4. **模型表达能力不是当前主瓶颈。**  
   当前 hidden dim=256，VPP dispatch 已使用 DeepSet encoder 处理 DER tokens，DSO/VPP actor 已隐私分离。它确实不够顶会级，例如缺 GNN/时序记忆/拓扑注意力，但本轮结果首先被 reward 尺度和安全约束误导，而不是“神经元太少”。

5. **训练轮数不是当前主瓶颈。**  
   每个 run `120 episodes * 672 steps = 80640` 环境步，5 seeds，已经足以暴露方向性问题。继续盲目加 episode 会把错误目标优化得更彻底。只有在 reward、projection、split、checkpoint 机制修正后，加长训练才有意义。

## 2. 证据链

### 2.1 Baseline 强于 RL

| 算法 | 平均 eval reward | 平均 total cost | 平均越限单元 | 安全通过率 |
|---|---:|---:|---:|---:|
| `rule_based` | -116070.77 | 2338215.36 | 0.00 | 1.000 |
| `no_flex` | -120441.23 | 2425624.62 | 0.00 | 1.000 |
| `HASAC` | -134213.03 | 2700014.27 | 31.55 | 0.667 |
| `HAPPO` | -135700.59 | 2746137.70 | 7.97 | 0.750 |
| `MATD3` | -138523.62 | 2786059.74 | 16.85 | 0.750 |

解释：

- `rule_based` 和 `no_flex` 均无越限，说明系统在保守运行下并不困难。
- RL 策略调用灵活性后反而增加 cost 和越限，说明学习信号或约束边界存在误导。
- `opf_oracle_proxy` 也弱于 rule_based，说明当前 proxy 不是可靠 oracle，不能作为最优性上界。

### 2.2 Reward 尺度不平衡

旧 reward 中：

```text
C_total =
  C_operation
+ C_tracking
+ C_projection
+ C_comfort
+ C_soc
+ C_voltage
+ C_line
+ C_trafo
+ C_powerflow

r_dso = -0.05 * C_total + feasibility_bonus + tracking_bonus
```

问题是 `C_comfort` 的绝对量级远大于其他项。这样 critic 主要学习 HVAC temperature deviation，而不是潮流安全和经济调度。

修复方向：

- 记录 raw cost 和 weighted cost。
- 对各项 cost 引入可配置权重。
- 增加每个 constraint violation cell 的显式惩罚，解决小幅电压越限因平方幅值太小导致惩罚弱的问题。

已实现：

- `DSO.reward_component_weights`
- `DSO.security_violation_count_penalty`
- `DSO.dso_reward_cost_scale`
- `raw_<component>` 诊断列
- `<component>_weight` 诊断列
- `configs/european_lv_benchmark_v2.yaml` 的 balanced reward 配置

### 2.3 reverseflow 泛化失败

主要越限集中：

```text
bus_voltage_low: bus 10/11/12/20/21
bus_voltage_high: bus 73/74/82
line_overload: 少量线路
```

这说明 DSO 给 VPP 的 operating envelope 没有充分表达网络安全边界。仅基于聚合 P 可行域无法保证 AC power flow 约束。

后续应升级为：

```text
FR/DOE local bounds
+ voltage sensitivity margin
+ line loading sensitivity margin
+ reverse-flow stress flag
+ AC power-flow projection / OPF repair
```

### 2.4 HASAC higher_entropy 失效

旧实现中：

- `higher_entropy` case 对 HAPPO 生效：`entropy_coef=0.03`
- 对 MATD3 生效：`exploration_noise=0.22`
- 对 HASAC 不生效：没有传入 target entropy / alpha 初始化

结果是 HASAC `base` 和 `higher_entropy` 完全相同。

已修复：

- `HASACConfig.target_entropy_multiplier`
- `HASACConfig.init_log_alpha_dso`
- `HASACConfig.init_log_alpha_dispatch`
- `HASACConfig.log_alpha_min/max`
- `paper_training.py` 的 `higher_entropy` case 现在会传入 HASAC entropy/alpha override。

## 3. 模型结构审查

### 3.1 当前结构是否太简单

当前结构不是最先进，但不是本轮失败第一原因。

已有能力：

- DSO actor 与 VPP actor 隐私分离。
- VPP dispatch 使用 DeepSet encoder，可处理异构 DER token。
- HAPPO 有 DSO / dispatch / portfolio 三类 actor。
- MATD3/HASAC 有 centralized twin Q 或 value critic。
- MATD3/HASAC 已有 per-VPP dispatch Q heads。

不足：

- 没有 GNN 或 feeder topology encoder。
- 没有 RNN/Transformer 处理 SOC、HVAC、EV 等长时序状态。
- DSO envelope 不是节点/线路约束向量，而是较粗的 VPP 聚合范围。
- HASAC/MATD3 的 portfolio 慢周期没有完整训练。
- critic 缺校准诊断，无法判断 Q/V 是否过估计或失真。

判断：

```text
短期不应优先盲目加深 MLP。
应先修 reward 尺度、安全 envelope、训练 split 和 checkpoint。
之后再引入 topology-aware critic / temporal encoder。
```

### 3.2 训练轮数是否太少

不是首要瓶颈。当前每个 run 80640 steps，60 个训练 checkpoint，总交互量已经能暴露趋势。

现象：

- 训练 episode reward 多数有提升。
- Frozen holdout 仍差于 rule_based。
- 部分 run best episode 明显好于 final episode，说明策略会退化。

判断：

```text
继续增加训练轮数前，应先引入 validation-best checkpoint。
否则长训可能保存退化后的 final checkpoint。
```

### 3.3 实验设计是否不合理

当前设计已经比 smoke 强很多，但仍不够论文级：

- 训练只用 `train_mixed`，holdout 包含更强 stress。
- 没有 validation split 选择 best checkpoint。
- `opf_oracle_proxy` 不是完整 AC OPF oracle。
- price 仍是 derived scarcity proxy，不是真实电价。
- HVAC/EV/ESS 惩罚没有经过物理量纲校准。

建议：

1. train/validation/holdout 三层 split。
2. 每 N episodes 在 validation profile frozen eval。
3. 只保存 validation-best checkpoint。
4. 训练 curriculum 覆盖 mixed、peak、cloudy、reverseflow。
5. 论文主表只使用 AC-safe baselines。

## 4. 本轮已完成的项目完善

### 4.1 Reward 可配置再平衡

新增 DSO 配置字段：

```yaml
reward:
  dso_reward_cost_scale: 0.05
  security_violation_count_penalty: 50.0
  component_weights:
    operation_cost: 1.0
    target_tracking_error_penalty: 1.0
    action_projection_penalty: 2.0
    comfort_violation_penalty: 0.02
    soc_violation_penalty: 0.25
    security_violation_count_penalty: 1.0
    voltage_violation_penalty: 20.0
    line_overload_penalty: 50.0
    transformer_overload_penalty: 50.0
    powerflow_penalty: 10.0
```

新的 reward 输出同时包含：

- weighted component：用于训练和总成本。
- `raw_<component>`：原始物理/代理成本，便于审计。
- `<component>_weight`：当前权重，便于复现实验。

4-step smoke 结果显示，新尺度下 cost 不再被 comfort 主导：

```text
total_cost                       179.5328
operation_cost                   178.6798
raw_comfort_violation_penalty     42.6499
comfort_violation_penalty          0.8530
dso_reward                        -3.9766
```

### 4.2 HASAC entropy 调参修复

`higher_entropy` 现在对 HASAC 生效：

```text
target_entropy_multiplier = 1.50
init_log_alpha_dso = 0.30
init_log_alpha_dispatch = 0.30
log_alpha clamp = [-5.0, 3.0]
```

这避免 `alpha_dispatch` 无限制膨胀，也让 `base` 与 `higher_entropy` 不再是完全相同的训练。

### 4.3 测试

新增测试：

- `tests/test_reward_rebalance.py`

已通过：

- `ruff check`
- `py_compile`
- `tests/test_reward_rebalance.py`
- `tests/test_timeseries_smoke.py`
- `tests/test_env_smoke.py`
- `tests/test_hasac_happo.py`
- `tests/test_paper_training_experiment.py`

## 5. 下一阶段应实现的关键改进

### P0：训练目标与安全性

1. validation-best checkpoint。
2. AC sensitivity / OPF-aware DSO envelope。
3. reward normalization 进一步系统化，生成 `reward_scale_audit.csv`。
4. reverseflow curriculum。

### P1：算法结构

1. HASAC/MATD3 增加 valid DER mask，避免 padded DER action 影响 log_prob / entropy / critic。
2. MATD3/HASAC actor optimizer 分角色隔离，避免 DSO Q 过度更新 VPP actor。
3. HAPPO 连续动作改成 tanh-squashed Gaussian，减少 clamp-logprob mismatch。
4. 加 critic calibration：predicted Q/V vs realized return。

### P2：神经网络升级

1. DSO critic 引入 topology-aware encoder：feeder segment / bus voltage / line loading tokens。
2. VPP encoder 引入 temporal state：SOC、HVAC temperature、EV departure pressure 的短期历史。
3. Operating envelope 改成 vector envelope：按 zone / node / critical line 给边界。

## 6. 对用户问题的直接回答

### 是不是模型设计太简单？

部分是，但不是当前主因。MLP/DeepSet 没有 GNN 和时序记忆，确实不够顶会级；但本轮失败首先由 reward 尺度和安全约束误导造成。

### 是不是深度和神经元不够？

不是首要问题。hidden dim=256 已经足以学习当前启发式策略；`larger_network` case 没有系统性更好，说明加宽网络不能解决核心问题。

### 是不是训练轮数太少？

不是首要问题。训练 reward 提升但 holdout 失败，说明是目标/泛化问题，不是单纯样本不足。

### 是不是实验设计不合理？

是，主要体现在：训练 split 覆盖不足、无 validation-best checkpoint、oracle proxy 不强、reward 未校准、reverseflow 安全边界不足。

### 是不是惩罚设置误导模型？

是，这是最明确的问题。旧 comfort penalty 主导 total cost，安全越限惩罚反而相对弱。已通过可配置 reward 权重和 violation count penalty 做第一轮修复。

## 7. 本轮追加修复：从“只分析”推进到机制改造

结合三个并行审查线程的结论，本轮继续完成了以下工程修改：

1. **DSO operating envelope 改为 grid-aware。**  
   旧版本只按电价阈值生成 `absorb_or_charge / export_or_reduce_load / balanced_operation`。这会在低电压时仍然因为低价而鼓励 VPP 充电/吸收功率。现在 `Simulator.step()` 会在下发 envelope 前先运行一次 pre-dispatch power flow，并将 `min_vm_pu / max_vm_pu / line_loading / trafo_loading / pcc_vm_pu` 写入 envelope。当低电压、高电压或热过载风险接近边界时，`grid_priority_over_price=True`，安全意图覆盖价格意图。

2. **VPP dispatch actor 的本地观测补齐动态 DER 状态。**  
   旧 DER token 只有类型、母线、P/Q 边界、controllable 和成本系数。现在 token 从 `R^15` 升级到 `R^26`，新增当前 P/Q、PV available power、ESS SOC、SOC 边界、EVCS average SOC、HVAC indoor temperature、温度边界和 comfort penalty。这样 VPP agent 不再被 SOC/comfort 惩罚训练，却看不到对应状态。

3. **HAPPO / MATD3 / HASAC 保存 train-best checkpoint。**  
   旧 paper training 默认用 final checkpoint 做 frozen eval；但 `paper_long` 结果显示 MATD3/HASAC 存在明显 best-final gap。现在训练函数同时保存 `{algorithm}_checkpoint.pt` 和 `{algorithm}_best_checkpoint.pt`，summary 中 `checkpoint` 默认指向 train-best，`final_checkpoint` 保留最终权重，`best_checkpoint_episode` 记录被选中的 episode。

4. **paper_training resume/eval 协议同步。**  
   `_load_completed_training()` 会读取 summary 中的 selected checkpoint，避免 resume 后又退回 final checkpoint。

5. **新增测试。**  
   `tests/test_grid_aware_envelope.py` 覆盖低电压覆盖低价吸收、高电压覆盖高价发电，以及 VPP 本地观测中 SOC/HVAC 动态 token 的存在性。

这些修改仍不是完整 AC-OPF projection，也不是论文最终版算法。它们解决的是当前训练失败中最明确的几类“错误学习信号”：安全意图未进入 envelope、VPP actor 看不到被惩罚的动态状态、训练后期退化却评估 final checkpoint。
