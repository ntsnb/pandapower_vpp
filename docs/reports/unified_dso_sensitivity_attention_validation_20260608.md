# Unified DSO Sensitivity Attention 修改与验证报告

日期：2026-06-08

## 1. 修改目标

本轮目标是把原先并行存在的两条 DSO 决策链合并：

1. 旧链路：`dso_global_guidance actor -> normalized_dso_action -> dso_targets -> simulator_actions`
2. envelope 链路：`sensitivity_attention_v1 -> DSO operating envelope -> VPP dispatch`

修改后，`sensitivity_attention_v1` 是 sensitivity 配置下唯一 DSO 决策接口。HAPPO structured DSO actor 输出 action-unit 级 envelope 参数；legacy DSO target payload 在 sensitivity 配置下会被转换为 envelope override，不再直接绕过 envelope。

## 2. 关键代码变化

### 2.1 DSO actor 输出从 VPP 标量改为 action-unit envelope 参数

文件：`src/vpp_dso_sim/dso/models/structured_happo_actor.py`

原来：

```text
StructuredDSOGaussianActor -> 每个 VPP 一个 scalar mean
```

现在：

```text
StructuredDSOGaussianActor -> max_action_units * 6 维动作
```

6 个通道分别是：

```text
center_ratio
width_ratio
guidance_strength
direction_absorb_logit
direction_balanced_logit
direction_inject_logit
```

并新增 `normalized_envelope_action_to_payload()`，把 HAPPO 采样动作转换为 `dso_global_guidance.envelope_action`。

### 2.2 环境层统一 DSO 决策入口

文件：`src/vpp_dso_sim/envs/multi_agent_env.py`

新增内部 simulator action key：

```text
__dso_envelope_guidance__
```

行为：

```text
envelope_action -> 直接传给 sensitivity_attention_v1 policy
legacy targets + sensitivity_attention_v1 config -> 转换成 legacy_targets_by_vpp envelope override
legacy targets + 非 sensitivity config -> 保留旧行为
```

### 2.3 simulator 将 DSO override 传入 envelope policy

文件：`src/vpp_dso_sim/simulation/simulator.py`

`Simulator._build_dso_operating_envelope_for_policy()` 现在只对 `sensitivity_attention_v1` 传入 `actor_override`，避免破坏 `rule_v0`。

### 2.4 rule warm-start 降级为 teacher/fallback

文件：`src/vpp_dso_sim/dso/envelope/sensitivity_attention_v1.py`

当收到 `actor_override` 时：

```text
rule_warmstart_role = teacher_reference_only
```

rule envelope 只用于记录 teacher reference，不参与最终 envelope 混合。

当没有 unified actor override 且 warm-start schedule 未完成时：

```text
rule_warmstart_role = fallback_blend_without_unified_actor_action
```

这保留了无 checkpoint / 无 actor action 时的安全 fallback。

### 2.5 模型容量提升

文件：

```text
configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1.yaml
configs/algorithms/dso_sensitivity_attention/v1/happo_sensitivity_attention_v1.yaml
```

paper-long 配置：

```text
d_model: 256
num_heads: 8
num_layers: 3
action_self_attention_layers: 2
dropout: 0.05
```

短 HAPPO 配置：

```text
d_model: 128
num_heads: 4
num_layers: 2
action_self_attention_layers: 1
dropout: 0.02
```

## 3. 新增验证与诊断

### 3.1 新增测试

文件：

```text
tests/test_envelope_policy_switch.py
tests/test_structured_happo_training.py
tests/test_vpp_feasible_region_bias_diagnostic.py
```

新增覆盖点：

1. `envelope_action` 会进入 unified `sensitivity_attention_v1`。
2. sensitivity 配置下 legacy `targets` 会转换成 unified envelope override。
3. structured DSO actor 输出维度为 `max_action_units * 6`，不再是 VPP scalar。
4. 可行域偏负诊断脚本会输出 detail 和 summary CSV。

### 3.2 测试结果

已通过：

```text
pytest tests/test_envelope_policy_switch.py tests/test_structured_happo_training.py tests/test_bipartite_attention_actor.py tests/test_vpp_feasible_region_bias_diagnostic.py -q
结果：21 passed
```

兼容性测试：

```text
pytest tests/test_multi_agent_env.py tests/test_deep_rl_training.py::test_vpp_dispatch_action_changes_decoded_simulator_targets tests/test_ctde_interface.py -q
结果：10 passed
```

HATRPO sensitivity 小测试：

```text
pytest tests/test_hatrpo_training.py::test_hatrpo_applies_reward_shield_coefficients_from_yaml -q
结果：1 passed
```

### 3.3 GPU smoke 训练

输出目录：

```text
outputs/unified_dso_sensitivity_attention_smoke_20260608/happo_gpu
```

结果摘要：

```text
resolved_device: cuda
cuda_device_name: NVIDIA A800 80GB PCIe
dso_actor_type: sensitivity_attention_v1_structured_happo
dso_actor_observation_mode: structured_bipartite
structured_dso_actor_trainable: True
dso_input_dim: 7074
critic_action_dim: 72
episodes: 1
horizon_steps: 4
violations: 0
projection_gap_mw: 0.0
```

注意：默认 sandbox 内 Python 不能初始化 NVML，所以会误判 `torch.cuda.is_available=False`。非 sandbox 权限运行时，PyTorch 能看到 A800 并解析到 CUDA。

## 4. VPP 可行域偏负诊断

新增脚本：

```text
scripts/analyze_vpp_feasible_region_bias.py
```

输出目录：

```text
outputs/unified_dso_sensitivity_attention_bias_audit_20260608
```

输出文件：

```text
vpp_feasible_region_bias_detail.csv
vpp_feasible_region_bias_summary.csv
vpp_feasible_region_bias_metadata.json
```

8-step 全 variant 诊断结果：

```text
baseline / load_scale_1p2 / load_scale_1p5 / pv_scale_0p8 / pv_scale_1p2 / no_ac_aware
midpoint_negative_rate = 1.0
preferred_target_negative_rate = 1.0
all_negative_rate = 0.0
crosses_zero_rate = 1.0
ac_aware_enabled_rate = 0.0
```

解释：

1. 当前所有 VPP 的 FR midpoint 都偏负，说明 FR 中心确实偏向吸收/少注入。
2. 但 `all_negative_rate = 0.0` 且 `crosses_zero_rate = 1.0`，说明可行域不是只能吸收，仍然允许正向注入。
3. 负荷放大 1.2/1.5、PV 缩放 0.8/1.2 后，短 horizon 内 FR 统计几乎不变。
4. `no_ac_aware` 与 baseline 几乎一致，且 `ac_aware_enabled_rate = 0.0`，说明这次观察到的偏负不是 AC-aware 安全外壳直接造成的。
5. 更可能的原因是当前 DER 组合与默认运行点下，储能/EVCS/柔性负荷等吸收侧能力与当前出力状态共同导致 FR midpoint 偏负。

## 5. 仍需注意

1. 本轮只跑了 smoke 级别训练，不代表 paper-long 收敛结论。
2. HAPPO structured DSO 已统一到 envelope action；HATRPO legacy target 在 sensitivity 配置下已被环境转换到 envelope override，但 HATRPO 自身还不是 structured attention actor。
3. 8-step 诊断说明“偏负”不是 AC-aware 或简单负荷缩放造成的，但要验证区域负荷建模不足，需要进一步加入真实区域负荷/公开负荷曲线并跑 24-96 step 对照。
4. 训练速度瓶颈仍主要在 pandapower、有限差分灵敏度和 AC certificate，不是神经网络 GPU 前向。
