你是一个资深多智能体强化学习、配电网优化和 Python 工程专家。请在当前项目代码仓中重构 reward 设计，目标是实现 reward_v2_minimal，并保留 legacy reward_v1 以便 ablation。不要只改 YAML 权重，必须修复 reward 结构、配置生效路径、日志和训练检查。

背景：
当前项目 reward 存在如下问题：
1. DSO reward 中 tracking_bonus、effective_response_bonus、target_tracking_error_penalty 重复表达同一目标。当前 effective_response_bonus 在没有外部 score 时 fallback 为 1/(1+target_tracking_error)，与 tracking_bonus 本质重复。
2. VPP dispatch reward 又有 target_tracking_penalty，portfolio reward 也有 reliability / delivery_risk tracking 项，导致 tracking 在多个层级重复塑形。
3. DSO reward 当前包含 comfort_violation_penalty 和 soc_violation_penalty，但这些更应该属于 VPP/user-side reward。旧实验中 comfort 曾占 DSO 加权成本约 97.5%，sensitivity-v1 中仍约 55%，会支配 DSO reward。
4. reward.vpp.* YAML 配置当前没有被 reward_design.py 动态读取，实际 VPP 权重来自 reward_contracts.py 常量，必须修复。
5. 当前 DSO 和 VPP dispatch 会扣 shield_intervention_penalty，但 portfolio reward 当前不扣，导致 portfolio 可能选择不可执行组合并让 dispatch 或 safety layer 背锅。
6. envelope_width_penalty 和 preferred_region_bonus 的 width_gate 会诱导 DSO 给窄 envelope，不符合 DOE 应在安全前提下释放容量的逻辑。
7. post_ac_violation_count=0 不代表 raw actor 安全；必须同时报告 shield_intervention_gap、projection gaps、raw action violation 等。

请修改以下核心文件，具体文件名以仓库实际结构为准：
- src/vpp_dso_sim/entities/dso.py
- src/vpp_dso_sim/envs/reward_design.py
- src/vpp_dso_sim/learning/reward_contracts.py
- src/vpp_dso_sim/learning/advanced_marl.py
- src/vpp_dso_sim/learning/hatrpo.py
- configs/*sensitivity_attention_v1.yaml 或新增 configs/reward_v2_minimal.yaml
- 相关 logger / metrics / tests 文件

一、实现 versioned reward

新增配置：
reward:
  version: v2_minimal
  critic_reward_scale: 0.01

保留：
reward:
  version: v1_legacy

要求：
1. v1_legacy 行为尽量不变。
2. v2_minimal 使用新的 reward 结构。
3. 所有训练启动时必须保存：
   - resolved_reward_config.yaml
   - reward_config_hash.txt
4. 训练日志第一行打印 reward.version 和所有实际生效权重。
5. 如果 YAML 中有 reward.vpp.*，必须实际生效；不要再让 reward_contracts.py 常量静默覆盖 YAML。

二、建立 RewardConfig dataclass

请建立或扩展 reward config 解析层，例如：
- DSORewardConfig
- VPPDispatchRewardConfig
- VPPPortfolioRewardConfig
- ShieldRewardConfig

建议字段：

DSORewardConfig:
  enable_tracking_bonus: false
  enable_effective_response_bonus: false
  enable_target_tracking_cost: false
  comfort_violation_weight: 0.0
  soc_violation_weight: 0.0
  feasibility_bonus_weight: 0.0
  safety_margin_weight: 1.0
  hard_violation_weight: 10.0
  powerflow_failure_weight: 20.0
  operation_cost_weight: 1.0
  flex_procurement_cost_weight: 1.0
  loss_cost_weight: 1.0
  curtailment_cost_weight: 0.5
  projection_gap_weight: 1.0
  projection_attribution: true
  envelope_width_penalty_weight: 0.0
  safe_capacity_utilization_weight: 0.2
  over_conservative_curtailment_weight: 0.5
  smoothness_weight: 0.05
  voltage_guard_band_pu: 0.02
  line_guard_band_percent: 5.0
  trafo_guard_band_percent: 5.0
  component_clip: 10.0

VPPDispatchRewardConfig:
  private_profit_weight: 0.02
  use_baseline_service_payment: true
  service_payment_weight: 1.0
  availability_payment_weight: 1.0
  contract_delivery_weight: 10.0
  projection_attribution: true
  projection_linear_weight: 2.0
  projection_quadratic_weight: 5.0
  comfort_soc_weight: 0.02
  battery_degradation_weight: 0.01
  preferred_region_bonus_weight: 0.0
  raw_dso_reward_weight: 0.0

VPPPortfolioRewardConfig:
  mode: window_return
  decision_interval_steps: 24
  long_horizon_profit_weight: 0.05
  verified_capacity_weight: 0.5
  delivery_reliability_weight: 1.0
  future_shield_penalty_weight: 1.0
  future_projection_penalty_weight: 0.5
  future_comfort_soc_weight: 0.02
  switching_keep_cost: 0.0
  switching_reweight_cost: 0.05
  switching_membership_change_cost: 0.20
  raw_dso_reward_weight: 0.0

三、修改 DSO reward

在 reward.version == v2_minimal 时，DSO reward 不再包含：
- tracking_bonus
- effective_response_bonus
- target_tracking_error_penalty
- comfort_violation_penalty
- soc_violation_penalty
- envelope_width_penalty

这些项可继续计算和落盘为 diagnostic，但不能进入 dso_reward_train。

新的 DSO reward 公式：

R_DSO_v2 =
  + safe_capacity_utilization_weight * safe_capacity_utilization
  - flex_procurement_cost_weight * flex_procurement_cost
  - loss_cost_weight * loss_cost
  - curtailment_cost_weight * over_conservative_curtailment
  - smoothness_weight * envelope_smoothness
  - safety_margin_weight * safety_margin_penalty
  - hard_violation_weight * hard_violation_penalty
  - powerflow_failure_weight * powerflow_failure_penalty
  - projection_gap_weight * dso_responsible_projection_penalty

实现 safety margin penalty：
1. voltage violation:
   sum(max(0, v_min - v)^2 + max(0, v - v_max)^2)
2. voltage guard band:
   sum(max(0, voltage_guard_band_pu - min(v - v_min, v_max - v))^2)
3. line guard:
   sum(max(0, line_guard_band_percent - (limit_percent - loading_percent))^2)
4. trafo guard:
   同 line guard
5. powerflow failure:
   单独强惩罚，不要只当普通 clipped cost。

保留 post-AC hard shield 执行逻辑，但日志同时保存：
- unclipped safety severity
- clipped training safety penalty
- post_ac_violation_count
- post_ac_security_penalty
- voltage margin
- line margin
- trafo margin

四、修改 DOE/envelope 逻辑

在 v2_minimal 中：
1. envelope_width_penalty_weight = 0.0。
2. preferred_region_bonus_weight 默认 = 0.0。
3. 新增 safe_capacity_utilization:
   safe_released_capacity / nominal_feasible_capacity。
4. 新增 over_conservative_curtailment_penalty：
   当 network 安全裕度允许更多容量但 DSO 释放容量过少时惩罚。
5. 保留 smoothness_penalty，但只在 DSO 层出现，避免 VPP 层重复。

五、修改 VPP dispatch reward

在 v2_minimal 中，VPP dispatch reward 使用：

R_dispatch_v2 =
  + private_profit_weight * private_profit_proxy
  + service_payment_weight * verified_service_payment
  + availability_payment_weight * availability_payment
  - contract_delivery_weight * contract_shortfall_mw^2
  - projection_penalty
  - comfort_soc_weight * scaled_or_normalized_comfort_soc_penalty
  - battery_degradation_weight * battery_degradation_cost

必须把 service payment 改成 baseline-based：
- baseline_p_mw
- requested_delta_p_mw
- accepted_delta_p_mw
- actual_delta_p_mw = delivered_p_mw - baseline_p_mw
- directional_delivery = sign(accepted_delta_p_mw) * actual_delta_p_mw
- verified_delivery = clip(directional_delivery, 0, abs(accepted_delta_p_mw))
- contract_shortfall_mw = max(0, abs(accepted_delta_p_mw) - verified_delivery)

不要再用 abs(delivered_p_mw) 作为 flexibility service payment 的服务量。

如果 baseline 暂时没有现成实现，请先实现一个清晰的 fallback：
baseline_p_mw = previous_uncontrolled_or_last_safe_power
并在日志中标记 baseline_source。

新增电池退化成本：
battery_degradation_cost = degradation_price_per_mwh * abs(p_batt_mw) * step_hours
如果设备粒度可用，对每个 battery 求和。

六、实现 projection / shield 责任归因

新增函数：
attribute_projection_gaps(...)

输出：
- dso_responsible_projection_gap_mw
- dispatch_responsible_projection_gap_mw[vpp_id]
- portfolio_responsible_projection_gap_mw[vpp_id]
- exogenous_projection_gap_mw

初始归因规则：
1. raw VPP dispatch action 超出本 VPP hard bounds：惩罚该 VPP dispatch。
2. DSO envelope / preferred target 本身不可交付：惩罚 DSO。
3. portfolio 声明的容量或组合导致未来 dispatch 经常被修复：惩罚 portfolio。
4. DER 外生 unavailable 或随机扰动导致可行域收缩：记录为 exogenous，不惩罚 actor。
5. AC certificate 修复：若有网络灵敏度，按各 VPP 对违规方向的贡献分摊；若没有，先按 abs(delta_p_mw) 比例分摊，并记录 attribution_method。

七、修改 portfolio reward

实现 window_return 模式：
1. portfolio 每 decision_interval_steps 决策一次。
2. 非 portfolio 决策步 portfolio reward = 0。
3. 每个 dispatch step 更新 portfolio_window_stats[vpp_id]：
   - sum_private_profit
   - sum_contract_shortfall
   - sum_shield_intervention
   - sum_projection_gap
   - sum_comfort_soc_penalty
   - sum_verified_capacity
   - n_steps
4. 到下一个 portfolio 决策步，根据上一窗口计算 portfolio reward：
   R_portfolio_v2 =
     + long_horizon_profit_weight * mean_private_profit
     + verified_capacity_weight * mean_verified_capacity
     - delivery_reliability_weight * mean_contract_shortfall
     - future_shield_penalty_weight * mean_shield_intervention
     - future_projection_penalty_weight * mean_projection_gap
     - future_comfort_soc_weight * mean_comfort_soc_penalty
     - switching_cost
5. portfolio 当前必须承担 future_shield_penalty 和 future_projection_penalty。
6. 将 propose_membership_change 在日志中解释为 portfolio_mode_change 或 resource_priority_mode_change；如果保留 membership_change 名称，则 switching_membership_change_cost 默认提高到 0.20。

八、修改训练层 reward vector

在 advanced_marl.py 和 hatrpo.py 中检查：
1. DSO train reward = DSO env reward - attributed DSO shield penalty。
2. VPP dispatch train reward = dispatch env reward - attributed dispatch shield penalty。
3. VPP portfolio train reward = portfolio env reward - attributed future portfolio shield/projection penalty。
4. critic/value 前继续使用 critic_reward_scale = 0.01，但日志必须分别保存：
   - env_reward
   - train_reward
   - critic_scaled_reward

不要只保存一个 reward 字段。

九、补齐日志字段

必须新增以下日志字段，至少写入 step_metrics.csv 和 reward_components.csv：

DSO：
- dso_reward_env
- dso_reward_train
- dso_reward_critic_scaled
- dso_safety_margin_penalty
- dso_voltage_guard_penalty
- dso_line_guard_penalty
- dso_trafo_guard_penalty
- dso_powerflow_failure_penalty
- dso_flex_procurement_cost
- dso_loss_cost
- dso_curtailment_cost
- dso_safe_capacity_utilization_reward
- dso_over_conservative_curtailment_penalty
- dso_responsible_projection_gap_mw
- dso_responsible_projection_penalty
- tracking_bonus_diagnostic
- effective_response_bonus_diagnostic
- target_tracking_error_to_raw_target
- target_tracking_error_to_projected_target

VPP dispatch：
- mean_dispatch_reward_env
- mean_dispatch_reward_train
- min_dispatch_reward_train
- p05_dispatch_reward_train
- p95_dispatch_reward_train
- private_profit_proxy
- energy_market_revenue
- baseline_p_mw
- requested_delta_p_mw
- accepted_delta_p_mw
- actual_delta_p_mw
- verified_delivery_mw
- contract_shortfall_mw
- contract_delivery_penalty
- availability_payment
- service_payment
- der_operation_cost
- battery_degradation_cost
- comfort_penalty
- soc_penalty
- dispatch_responsible_projection_gap_mw
- dispatch_projection_penalty

Portfolio：
- mean_portfolio_reward_env
- mean_portfolio_reward_train
- portfolio_window_profit
- portfolio_window_contract_shortfall
- portfolio_window_shield_intervention
- portfolio_window_projection_gap
- portfolio_window_comfort_soc_penalty
- portfolio_window_verified_capacity
- portfolio_switching_cost
- portfolio_action_type

Safety：
- raw_action_violation_rate
- post_ac_violation_count
- post_ac_security_penalty
- shield_intervention_gap_mw
- shield_intervention_penalty
- action_projection_gap_mw
- local_bounds_projection_gap_mw
- ac_aware_projection_gap_mw
- ac_certified_projection_gap_mw
- certificate_repair_rate

十、测试要求

新增或更新单元测试：
1. test_vpp_yaml_weights_are_effective：
   修改 reward.vpp.dispatch.private_profit_weight 后，实际 reward 数值必须变化。
2. test_dso_tracking_disabled_in_v2：
   v2_minimal 下 tracking_bonus 和 effective_response_bonus 不进入 dso_reward_train。
3. test_dso_comfort_soc_excluded_in_v2：
   v2_minimal 下 DSO reward 不随 comfort/SOC 权重变化。
4. test_dispatch_contract_delivery：
   baseline-based service delivery 和 shortfall 计算正确。
5. test_portfolio_window_reward：
   portfolio reward 只在决策步结算，且包含 future shield/projection。
6. test_reward_no_nan_inf：
   所有 reward components 不出现 NaN/Inf。
7. test_legacy_v1_backward_compatible：
   v1_legacy 在相同输入下与旧 reward 近似一致。

十一、配置文件

新增 configs/reward_v2_minimal.yaml，内容包括：

reward:
  version: v2_minimal
  critic_reward_scale: 0.01

  dso:
    enable_tracking_bonus: false
    enable_effective_response_bonus: false
    enable_target_tracking_cost: false
    comfort_violation_weight: 0.0
    soc_violation_weight: 0.0
    feasibility_bonus_weight: 0.0
    safety_margin_weight: 1.0
    hard_violation_weight: 10.0
    powerflow_failure_weight: 20.0
    operation_cost_weight: 1.0
    flex_procurement_cost_weight: 1.0
    loss_cost_weight: 1.0
    curtailment_cost_weight: 0.5
    projection_gap_weight: 1.0
    projection_attribution: true
    envelope_width_penalty_weight: 0.0
    safe_capacity_utilization_weight: 0.2
    over_conservative_curtailment_weight: 0.5
    smoothness_weight: 0.05
    voltage_guard_band_pu: 0.02
    line_guard_band_percent: 5.0
    trafo_guard_band_percent: 5.0
    component_clip: 10.0

  vpp:
    dispatch:
      private_profit_weight: 0.02
      use_baseline_service_payment: true
      service_payment_weight: 1.0
      availability_payment_weight: 1.0
      contract_delivery_weight: 10.0
      projection_attribution: true
      projection_linear_weight: 2.0
      projection_quadratic_weight: 5.0
      comfort_soc_weight: 0.02
      battery_degradation_weight: 0.01
      preferred_region_bonus_weight: 0.0
      raw_dso_reward_weight: 0.0

    portfolio:
      mode: window_return
      decision_interval_steps: 24
      long_horizon_profit_weight: 0.05
      verified_capacity_weight: 0.5
      delivery_reliability_weight: 1.0
      future_shield_penalty_weight: 1.0
      future_projection_penalty_weight: 0.5
      future_comfort_soc_weight: 0.02
      switching_keep_cost: 0.0
      switching_reweight_cost: 0.05
      switching_membership_change_cost: 0.20
      raw_dso_reward_weight: 0.0

十二、训练前检查

完成修改后，请按顺序运行：
1. unit tests
2. one-episode deterministic smoke
3. 500-1000 step HAPPO smoke
4. 500-1000 step HATRPO smoke
5. 500-1000 step HASAC/MATD3 smoke
6. reward scale audit

通过条件：
- YAML 中 reward.vpp.* 确认生效；
- DSO comfort/SOC 贡献为 0；
- DSO tracking_bonus/effective_response_bonus 只作为 diagnostics；
- portfolio reward 包含 future shield/projection；
- reward 无 NaN/Inf；
- 各 role reward 都落盘；
- projection/shield/post-AC 全部落盘；
- v2 smoke 中没有单一非核心 reward 项长期占比超过 40%。

十三、全量训练实验矩阵

请生成或更新训练脚本，跑以下实验：

A. legacy_v1_reward
B. v2_minimal
C. v2_minimal_no_shield_eval
D. v2_minimal_no_portfolio_window_penalty
E. v2_minimal_with_preferred_bonus_0p05
F. v2_minimal_contract_delivery_weight_5_10_20

每个实验使用相同 seeds、相同场景、相同 horizon，与旧 paper-long 保持可比。

十四、最终交付

请输出：
1. 修改文件列表；
2. reward_v2_minimal 公式说明；
3. resolved_reward_config.yaml 示例；
4. 单元测试结果；
5. smoke 训练结果摘要；
6. reward component abs-share 表；
7. full training 启动命令；
8. legacy_v1 与 v2_minimal 对比指标列表。

不要删除 legacy reward。不要只改配置不改代码。不要让 reward_contracts.py 常量覆盖 YAML。不要把 tracking、effective_response、reliability、delivery risk 在多个层级重复作为训练 reward。
