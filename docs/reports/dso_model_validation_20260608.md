# DSO 相关算法模型修改后验证报告

日期：2026-06-08

## 1. 验证目标

本轮验证针对修改后的 DSO 相关算法链路，重点确认：

- sensitivity-attention DSO actor 是否能构造结构化观测、输出运行包络动作并进入 simulator。
- DSO envelope action 与旧的 `dso_targets` 是否已经走统一包络接口，避免重复调度语义。
- HAPPO 与 HATRPO 是否能被 paper-training campaign 调度、训练、保存 checkpoint 并进行 frozen evaluation。
- reward v2 minimal、AC-aware projection、post-AC security 指标是否在训练/评估文件中被记录。
- 训练是否真正使用 GPU，而不是由于沙箱限制退回 CPU。

## 2. 执行的验证

### 2.1 单元与接口测试

命令：

```bash
./.venv-server/bin/python -m pytest \
  tests/test_bipartite_attention_actor.py \
  tests/test_structured_observation_shapes.py \
  tests/test_sensitivity_shapes.py \
  tests/test_envelope_policy_switch.py \
  tests/test_structured_happo_training.py \
  tests/test_training_step_no_nan.py -q
```

结果：29 个测试通过。仅出现 Jupyter 路径迁移 warning。

补充命令：

```bash
./.venv-server/bin/python -m pytest \
  tests/test_hatrpo_training.py \
  tests/test_hasac_happo.py \
  tests/test_paper_training_experiment.py -q
```

结果：26 个测试通过。仅出现 Jupyter 路径迁移 warning。

### 2.2 DSO sensitivity-attention 最小仿真 smoke

输出目录：

```text
outputs/dso_model_validation_20260608/smoke_rollout_8step
```

关键结果：

- `envelope_policy = sensitivity_attention_v1`
- 8 个 simulator step 全部完成。
- 记录：
  - `dso_operating_envelope`: 56 行
  - `fr_envelope_state`: 400 行
  - `projection_trace`: 776 行
  - `vpp_rl_disaggregation`: 264 行
  - `constraint_violations`: 0 行
  - `reward_components`: 8 行
- `nan_or_inf_detected = false`

该 smoke 证明：DSO 包络、FR/DOE、投影、VPP 分解调度、reward 记录和 pandapower 潮流链路能跑通。

### 2.3 DSO attention actor 短监督训练 sanity

输出目录：

```text
outputs/dso_model_validation_20260608/short_train_sanity_32update
```

关键结果：

- 32 次更新完成。
- 初始 loss：0.4317245
- 最终 loss：0.0082190
- 最小 loss：0.0007645
- `nan_or_inf_detected = false`

解释：这不是最终 RL 收敛实验，而是验证 sensitivity-attention DSO actor 至少能在规则包络监督目标上学习 center、width、direction 输出，不存在明显梯度或数值崩溃。

## 3. Paper-training CPU 验证

先在普通沙箱环境中跑了一个小 campaign：

```bash
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset smoke \
  --config-path configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml \
  --output-dir outputs/dso_model_validation_20260608/paper_training_smoke_happo_hatrpo \
  --algorithms happo,hatrpo \
  --seeds 60608 \
  --hparam-cases base \
  --horizon-steps 16 \
  --eval-horizon-steps 16 \
  --train-episodes 3 \
  --progress-interval-seconds 15 \
  --verbose-progress
```

输出目录：

```text
outputs/dso_model_validation_20260608/paper_training_smoke_happo_hatrpo
```

结果：

| 算法 | train episodes | horizon | train final reward | frozen eval reward | eval total cost | violations |
|---|---:|---:|---:|---:|---:|---:|
| HAPPO | 3 | 16 | -2.1556 | -2.4584 | 946.8742 | 0 |
| HATRPO | 3 | 16 | -6.9025 | -6.9343 | 928.2223 | 0 |

重要发现：

- 该 run 能完整完成 train + checkpoint + frozen eval + HTML report。
- HAPPO `critic_loss` 从 0.0751 降到 0.0676。
- HATRPO 三类 actor update 均被接受，KL 受 `max_kl=0.02` 约束。
- 但是该 run 的 summary 显示 `resolved_device=cpu`。

CPU 原因不是显存满，也不是模型没有 GPU 代码，而是受限沙箱内 PyTorch 无法初始化 CUDA runtime。沙箱中直接创建 CUDA tensor 报错：

```text
RuntimeError: No CUDA GPUs are available
```

因此，真正的 GPU 训练必须使用非沙箱授权命令执行。

## 4. Paper-training GPU 验证

非沙箱授权后执行最小 GPU campaign：

```bash
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset smoke \
  --config-path configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml \
  --output-dir outputs/dso_model_validation_20260608/paper_training_smoke_happo_hatrpo_gpu \
  --algorithms happo,hatrpo \
  --seeds 60608 \
  --hparam-cases base \
  --horizon-steps 8 \
  --eval-horizon-steps 8 \
  --train-episodes 1 \
  --progress-interval-seconds 15 \
  --verbose-progress
```

输出目录：

```text
outputs/dso_model_validation_20260608/paper_training_smoke_happo_hatrpo_gpu
```

HTML 报告：

```text
outputs/dso_model_validation_20260608/paper_training_smoke_happo_hatrpo_gpu/long_training_report.html
```

GPU 证据：

- HAPPO train summary:
  - `requested_device = auto`
  - `resolved_device = cuda`
  - `cuda_available = true`
  - `cuda_device_count = 1`
  - `cuda_device_name = NVIDIA A800 80GB PCIe`
  - `dso_actor_type = sensitivity_attention_v1_structured_happo`
  - `structured_dso_actor_trainable = true`
- HAPPO frozen eval summary:
  - `resolved_device = cuda`
  - `cuda_device_name = NVIDIA A800 80GB PCIe`
- HATRPO train summary:
  - `requested_device = auto`
  - `resolved_device = cuda`
  - `cuda_available = true`
  - `cuda_device_name = NVIDIA A800 80GB PCIe`
- HATRPO frozen eval summary:
  - `resolved_device = cuda`
  - `cuda_device_name = NVIDIA A800 80GB PCIe`

GPU run 结果：

| 算法 | train episodes | horizon | train final reward | frozen eval reward | eval total cost | violations | post-AC pass rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| HAPPO | 1 | 8 | -1.4826 | -0.7842 | 449.5061 | 0 | 1.0 |
| HATRPO | 1 | 8 | -3.4693 | -3.4348 | 438.9839 | 0 | 1.0 |

安全相关结果：

- `total_violation_cells = 0`
- `post_ac_violation_count = 0`
- `post_ac_voltage_violation_count = 0`
- `post_ac_line_overload_count = 0`
- `post_ac_trafo_overload_count = 0`
- `post_ac_powerflow_failed = 0`
- `post_ac_security_pass_rate = 1.0`
- `post_ac_powerflow_converged_rate = 1.0`
- `ac_certificate_safe_rate = 1.0`
- `ac_certified_projection_gap_mw = 0.0`

## 5. 审查结论

### 5.1 已通过

- 修改后的 DSO sensitivity-attention actor 能进入真实 simulator。
- 结构化 DSO actor 在 HAPPO 中被识别为可训练：
  - `dso_actor_type = sensitivity_attention_v1_structured_happo`
  - `structured_dso_actor_trainable = true`
- HAPPO 与 HATRPO 都能完成：
  - training
  - checkpoint 保存
  - frozen evaluation
  - report/CSV/JSON 导出
- reward v2 minimal 配置被加载：
  - `reward_version = v2_minimal`
  - `reward_config_hash = e881f550271683a7cd16124bcc4104071e48c908d07d4b929b16a2d3a0a68711`
- GPU 路径已经验证，非沙箱执行时能使用 A800。

### 5.2 仍需谨慎

- 8-step / 16-step smoke 只能证明链路可执行，不能证明 paper-long 收敛。
- HAPPO GPU smoke 只有 1 个 episode，不能解释 reward 趋势。
- HATRPO 已可用，但相对 HAPPO 速度更慢；paper-long 前应先做 HATRPO 小规模扩展，不建议直接全 seed 全 horizon 展开。
- `claim_readiness.json` 显示 `paper_claim_ready=false`，原因包括：
  - 当前 GPU smoke 没有包含 AC-validated reference baseline。
  - 不能把 bounded search baseline 宣称为 OPF 最优或上界。
  - 当前价格仍是 proxy price，不能直接宣称真实市场利润。
  - 安全性是 AC-aware shielded execution，不应宣称 RL 本身保证安全。

### 5.3 关于 GPU 速度

GPU 已经能用于神经网络训练，但整体训练速度不会线性提升，原因是当前主要耗时在：

- pandapower AC 潮流；
- sensitivity finite-difference；
- AC-aware certificate/projection；
- 多 VPP envelope/action-unit 记录；
- CSV/JSON 追踪导出。

这些部分主要在 CPU 上运行。GPU 只能加速 actor/critic 前向、反向和优化器步骤。

## 6. 建议的下一步实验

在直接进入完整 paper-long 前，建议按顺序运行：

1. GPU HAPPO 中等验证：
   - horizon 96
   - train episodes 5 到 10
   - seed 1 个
   - 只跑 HAPPO
   - 目标：确认 reward/loss 不是短 smoke 假象。

2. GPU HATRPO 中等验证：
   - horizon 32 或 96
   - train episodes 3 到 5
   - seed 1 个
   - 目标：确认 HATRPO 信赖域更新耗时、KL、accepted update 比例。

3. 加 baseline 的 paper-lite：
   - 至少包含 `rule_based`、`no_flex`、`ac_validated_search_reference`、`happo`
   - horizon 96
   - eval holdout_peak
   - 目标：让 `claim_readiness` 从执行可用转向论文对照可用。

4. 再进入 paper-long sensitivity v1：
   - 先 1 seed
   - 再扩展到 5 seeds
   - 每次 fresh output dir，不混用旧 checkpoint。

## 7. 本轮生成的主要文件

- `outputs/dso_model_validation_20260608/smoke_rollout_8step/smoke_summary.json`
- `outputs/dso_model_validation_20260608/smoke_rollout_8step/dso_operating_envelope.csv`
- `outputs/dso_model_validation_20260608/smoke_rollout_8step/dso_actor_outputs.csv`
- `outputs/dso_model_validation_20260608/short_train_sanity_32update/short_train_summary.json`
- `outputs/dso_model_validation_20260608/short_train_sanity_32update/dso_sensitivity_attention_short_train_loss_metrics.csv`
- `outputs/dso_model_validation_20260608/paper_training_smoke_happo_hatrpo/long_training_report.html`
- `outputs/dso_model_validation_20260608/paper_training_smoke_happo_hatrpo_gpu/long_training_report.html`
- `outputs/dso_model_validation_20260608/paper_training_smoke_happo_hatrpo_gpu/evaluation_seed_metrics.csv`
- `outputs/dso_model_validation_20260608/paper_training_smoke_happo_hatrpo_gpu/training_episode_metrics.csv`
- `outputs/dso_model_validation_20260608/paper_training_smoke_happo_hatrpo_gpu/training_loss_metrics.csv`
