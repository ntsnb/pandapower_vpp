# paper_long_sensitivity_v1 改造与实验留痕报告

更新时间：2026-05-28

本文档用于保留本轮 paper-long 级实验入口改造、验证过程、曲线文件位置、损失指标口径和后续正式实验命令。它不是最终论文结果报告；它是启动正式长周期实验前的可复现记录。

## 1. 改造目标

旧的 `paper_long` 仍默认使用 `configs/european_lv_benchmark_v2.yaml`，没有启用本轮新增的 DSO `sensitivity_attention_v1`、结构化二部图 observation 和 HAPPO 稳定化训练参数。因此，旧 paper-long 不能代表当前算法框架。

本轮新增 `paper_long_sensitivity_v1`，目标是：

1. 保留原 paper-long 的大场景、多 seed、长 horizon、holdout split 和 frozen evaluation 协议。
2. 将 DSO 全局智能体切换为 `sensitivity_attention_v1_structured_happo`。
3. 使用结构化 action unit / network object 表示，让 DSO actor 看到灵敏度、节点/线路安全裕度和 VPP 位置相关信息。
4. 将 YAML 中的 HAPPO 稳定化参数传入正式训练，避免 `target_kl`、observation normalization、advantage normalization、gradient clipping、nan guard 在 paper training runner 中被静默丢弃。
5. 保留 reward、loss、KL、entropy、gradient norm、projection gap、AC 安全和 frozen evaluation 产物，方便后续判断收敛。

## 2. 改造内容

| 文件 | 改动 | 目的 |
|---|---|---|
| `configs/european_lv_benchmark_v2_sensitivity_attention_v1.yaml` | 新增大场景 sensitivity-v1 配置，继承 `european_lv_benchmark_v2.yaml` | 在 7 VPP / 29 DER 大场景上启用结构化 DSO actor |
| `src/vpp_dso_sim/utils/config.py` | `load_yaml()` 支持顶层 `extends` 和递归 deep merge | 避免复制整份大 YAML，同时保留父配置的 network/VPP/profile/reward 细节 |
| `src/vpp_dso_sim/experiments/paper_training.py` | 新增 `paper_long_sensitivity_v1` preset | 给正式长周期实验一个独立、可审计的入口 |
| `src/vpp_dso_sim/experiments/paper_training.py` | 新增 `happo_use_yaml_trainer_settings` | 让 paper training runner 读取 YAML trainer 稳定化参数 |
| `src/vpp_dso_sim/experiments/paper_training.py` | HAPPO 配置从 YAML 基础值加 paper-long 覆盖值构造 | 保留 `target_kl=0.02`、`normalize_observations=True`、`max_grad_norm=0.5` 等关键设置 |
| `src/vpp_dso_sim/experiments/paper_training.py` | 将非空输出目录保护扩展到整个 `paper_long*` 家族 | 防止新旧实验结果混写 |
| `examples/17_paper_training_experiment.py` | CLI `--preset` 增加 `paper_long_sensitivity_v1` | 允许直接从命令行启动新协议 |
| `tests/test_paper_training_experiment.py` | 新增 preset/config/目录保护/structured HAPPO smoke 测试 | 固化回归检查 |

## 3. 当前 paper-long sensitivity-v1 配置

关键配置：

```yaml
dso:
  envelope_policy: sensitivity_attention_v1
  observation_mode: structured_bipartite
  action_unit_granularity: vpp_bus
  max_action_units: 32
  max_network_objects: 20

trainer:
  name: happo
  gamma: 0.995
  gae_lambda: 0.95
  clip_param: 0.2
  target_kl: 0.02
  entropy_coef: 0.01
  max_grad_norm: 0.5
  normalize_observations: true
  normalize_advantages: true
  nan_guard: true
  critic_use_action_summary: true
```

`paper_long_sensitivity_v1` preset：

| 项目 | 数值 |
|---|---|
| 训练场景 | `train_mixed` |
| 评估场景 | `holdout_peak`, `holdout_cloudy`, `holdout_reverseflow` |
| seeds | `9401, 9402, 9403, 9404, 9405` |
| horizon | `672` steps |
| step 粒度 | 15 min |
| episode 数 | `120` |
| HAPPO hidden dim | `256`，`larger_network` case 为 `512` |
| checkpoint selection | `both`，即 final 与 train_best 都 frozen eval |
| 默认算法 | `rule_based`, `no_flex`, `ac_validated_search_reference`, `happo` |

说明：HATRPO/MATD3/HASAC 仍可通过命令行显式加入，但当前只有 HAPPO 完整接入结构化 DSO `sensitivity_attention_v1`。若把其他算法直接放进本 preset，应在论文中标注为 legacy observation 对照，不能声称同构公平比较。

## 4. 已执行验证

### 4.1 聚焦 pytest

命令：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_paper_training_experiment.py::test_paper_long_sensitivity_v1_preset_uses_structured_happo_config \
  tests/test_paper_training_experiment.py::test_sensitivity_v1_large_benchmark_config_extends_paper_scenario \
  tests/test_paper_training_experiment.py::test_paper_long_sensitivity_v1_rejects_nonempty_output_without_resume \
  tests/test_paper_training_experiment.py::test_paper_training_structured_happo_sensitivity_smoke
```

结果：`4 passed`。

验证含义：

1. 新 preset 可以被 `paper_training_preset()` 正确加载。
2. 新 YAML 能继承大场景，场景包含 7 个 VPP，网络类型为 `european_lv_benchmark_v2`。
3. `paper_long_sensitivity_v1` 和 `paper_long` 一样拒绝写入非空输出目录。
4. paper training runner 能实际训练并 frozen eval 一次结构化 HAPPO。

### 4.2 结构化 HAPPO 相关测试

命令：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_structured_happo_training.py \
  tests/test_structured_smoke_rollout.py \
  tests/test_training_step_no_nan.py
```

结果：`6 passed`。

验证含义：

1. 结构化 HAPPO 训练路径可运行。
2. 结构化 smoke rollout 可运行。
3. 单步训练没有 NaN/Inf。

### 4.3 paper training 完整测试文件

命令：

```bash
./.venv-server/bin/python -m pytest -q tests/test_paper_training_experiment.py
```

结果：`15 passed`。

验证含义：baseline smoke、AC reference 元数据、baseline safety gate、checkpoint label 分离、非空 paper-long 输出目录保护、结构化 HAPPO smoke、legacy HAPPO checkpoint frozen eval 都在同一测试文件中通过。

## 5. 已生成的 smoke 级实验产物

输出目录：

```text
outputs/test_paper_training_structured_happo_sensitivity_smoke/
```

这只是 2-step / 1-episode 的 preflight smoke，不代表收敛结果。

关键训练摘要：

| 指标 | 数值 |
|---|---|
| `dso_actor_type` | `sensitivity_attention_v1_structured_happo` |
| `dso_actor_observation_mode` | `structured_bipartite` |
| `target_kl` | `0.02` |
| `kl_early_stop_count` | `1` |
| `normalize_observations` | `True` |
| `normalize_advantages` | `True` |
| `nan_guard` | `True` |
| `critic_use_action_summary` | `True` |
| `reward_scale` | `0.01` |
| `shield_intervention_step_count` | `0` |

2-step smoke episode 指标：

| 指标 | 数值 |
|---|---|
| `episode_reward` | `6.239091396766145` |
| `episode_cost` | `64.8946727440669` |
| `violation_count` | `0` |
| `projection_gap_mw` | `0.0` |
| `shield_intervention_gap_mw` | `0.0` |
| `critic_loss` | `0.08055782318115234` |

注意：`kl_early_stop_count=1` 出现在极小 smoke 中，不能单独判定训练失败；但 paper-long 过程中必须持续观察 `approx_kl`、`target_kl_exceeded` 和 `grad_norm`。

### 5.1 CLI preflight 记录

为确认真实命令行入口可用，已执行：

```bash
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset paper_long_sensitivity_v1 \
  --output-dir outputs/paper_training_long_sensitivity_v1_preflight_smoke \
  --seeds 9401 \
  --horizon-steps 2 \
  --eval-horizon-steps 2 \
  --train-episodes 1 \
  --hparam-cases base \
  --algorithms rule_based,no_flex,happo \
  --checkpoint-selection final \
  --no-html \
  --no-tensorboard \
  --progress-interval-seconds 60
```

结果：

| 项目 | 数值 |
|---|---|
| 输出目录 | `outputs/paper_training_long_sensitivity_v1_preflight_smoke` |
| run 总数 | `10` |
| evaluation rows | `9` |
| baseline eval | `6/6` |
| HAPPO train | `1/1` |
| HAPPO frozen eval | `3/3` |
| CLI preset | `paper_long_sensitivity_v1` |
| CLI algorithms | `rule_based`, `no_flex`, `happo` |

CLI preflight 的 HAPPO 训练摘要：

| 指标 | 数值 |
|---|---|
| `dso_actor_type` | `sensitivity_attention_v1_structured_happo` |
| `dso_actor_observation_mode` | `structured_bipartite` |
| `target_kl` | `0.02` |
| `kl_early_stop_count` | `1` |
| `normalize_observations` | `True` |
| `normalize_advantages` | `True` |
| `nan_guard` | `True` |
| `critic_use_action_summary` | `True` |
| `final_episode_reward` | `6.249330516242166` |
| `best_episode_reward` | `6.249330516242166` |
| `shield_intervention_step_count` | `0` |

CLI preflight 的 frozen eval 示例：

| split | algorithm | eval total reward | eval total cost | post-AC pass rate |
|---|---|---:|---:|---:|
| `holdout_peak` | `happo_sequential_ctde` | `6.218635653554708` | `90.2350985098177` | `1.0` |
| `holdout_peak` | `rule_based` | `4.491620458779385` | `105.51841928632659` | `1.0` |
| `holdout_peak` | `no_flex` | `4.492998750794968` | `77.9557532094118` | `1.0` |

CLI preflight 暴露的注意点：

| 现象 | 解释 | paper-long 监控方式 |
|---|---|---|
| 部分 VPP dispatch / portfolio 的 `target_kl_exceeded=True` | 2-step 极小样本下 advantage 方差大，KL 会非常敏感；这不等同于长期失败 | 看 `training_loss_metrics.csv` 中 `target_kl_exceeded` 占比是否长期偏高 |
| 部分 VPP `grad_norm` 较大 | 单 episode 更新可能受短样本和 role reward 方差影响 | 若长训持续大于正常区间，应尝试降低 learning rate、减少 `ppo_epochs` 或加严 reward normalization |
| `shield_intervention_step_count=0` | 2-step 中安全外壳没有干预，不说明长周期无干预 | 长训看 `shield_intervention_gap_mw` 和 `projection_gap_mw` 曲线 |

## 6. 曲线、损失和指标文件位置

正式 paper-long 输出目录记为 `outputs/<run>/`。

| 内容 | 文件 |
|---|---|
| 实验配置、版本、profile hash | `outputs/<run>/experiment_manifest.json` |
| 实时过程日志 | `outputs/<run>/experiment_progress.jsonl` |
| 实时过程表格 | `outputs/<run>/experiment_progress.csv` |
| 实时网页 | `outputs/<run>/live_progress.html` |
| 训练 reward/cost/projection/violation 曲线数据 | `outputs/<run>/training_episode_metrics.csv` |
| policy loss / KL / entropy / grad norm 曲线数据 | `outputs/<run>/training_loss_metrics.csv` |
| frozen eval 汇总 | `outputs/<run>/evaluation_seed_metrics.csv` |
| 聚合表 | `outputs/<run>/aggregate_metrics.csv` |
| 收敛摘要 | `outputs/<run>/convergence_summary.csv` |
| baseline 对比 | `outputs/<run>/baseline_comparison.csv` |
| 论文声明门禁 | `outputs/<run>/claim_guardrails.csv`, `outputs/<run>/claim_readiness.json` |
| 架构/机制诊断 | `outputs/<run>/architecture_diagnostics.csv` |
| TensorBoard 事件 | `outputs/<run>/tensorboard/<run_id>/events.*` |
| 自动导出的图片 | `outputs/<run>/tensorboard_images/*.png` |
| 静态总报告 | `outputs/<run>/long_training_report.html` |

smoke 级示例中已生成：

```text
outputs/test_paper_training_structured_happo_sensitivity_smoke/training_episode_metrics.csv
outputs/test_paper_training_structured_happo_sensitivity_smoke/training_loss_metrics.csv
outputs/test_paper_training_structured_happo_sensitivity_smoke/convergence_summary.csv
outputs/test_paper_training_structured_happo_sensitivity_smoke/tensorboard_images/training_reward_curve.png
outputs/test_paper_training_structured_happo_sensitivity_smoke/tensorboard_images/eval_reward_bar.png
outputs/test_paper_training_structured_happo_sensitivity_smoke/tensorboard_images/eval_cost_bar.png
outputs/test_paper_training_structured_happo_sensitivity_smoke/tensorboard_images/eval_violations_bar.png
```

## 7. 如何读取 loss 曲线

`training_loss_metrics.csv` 的关键列：

| 列 | 中文含义 | 正常观察方式 |
|---|---|---|
| `role` | 被更新的智能体角色 | 分别看 DSO、各 VPP dispatch、portfolio |
| `policy_loss` | PPO/HAPPO policy loss | 不要求单调下降；重点看是否爆炸或长期 NaN |
| `ratio_mean` | 新旧策略概率比均值 | 长期远离 1 说明策略更新过猛 |
| `entropy_mean` | 策略熵 | 过快降到很低可能探索不足；长期过高可能没学到确定策略 |
| `approx_kl` | 近似 KL | 应围绕 `target_kl=0.02` 附近受控，频繁超限说明更新太大 |
| `target_kl_exceeded` | 是否触发 KL 早停 | 偶发正常，频繁触发需降学习率或减少 epoch |
| `grad_norm` | 梯度范数 | 长期很大说明不稳定；突然极大要查 reward/obs/action 数值 |
| `correction_mean` | HAPPO 顺序更新修正项 | 过低说明多智能体非平稳性或重要性修正压力较大 |

`training_episode_metrics.csv` 的关键列：

| 列 | 中文含义 | 正常观察方式 |
|---|---|---|
| `episode_reward` | 每个训练 episode 总奖励 | paper-long 才能判断趋势；smoke 不看收敛 |
| `episode_cost` | 每个 episode 成本 | 要和 reward 一起看，不能只看 reward |
| `violation_count` | 违规次数 | 若 reward 升但违规升高，说明 reward hacking |
| `projection_gap_mw` | 投影前后差距 | 长期过大说明 actor 依赖安全外壳兜底 |
| `shield_intervention_gap_mw` | 安全外壳干预量 | 越低越好；高 reward + 高干预不是好策略 |
| `critic_loss` | critic 误差 | 爆炸或长期高位震荡说明 value 学习失败 |

## 8. 建议的正式运行流程

先跑一个短 preflight，确认服务器环境、输出目录、结构化 actor 和日志写入都正常：

```bash
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset paper_long_sensitivity_v1 \
  --output-dir outputs/paper_training_long_sensitivity_v1_preflight_smoke \
  --seeds 9401 \
  --horizon-steps 2 \
  --eval-horizon-steps 2 \
  --train-episodes 1 \
  --hparam-cases base \
  --algorithms rule_based,no_flex,happo \
  --checkpoint-selection final \
  --no-html \
  --no-tensorboard \
  --progress-interval-seconds 60
```

若 preflight 通过，再启动 paper-long：

```bash
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset paper_long_sensitivity_v1 \
  --output-dir outputs/paper_training_long_sensitivity_v1_20260528 \
  --progress-interval-seconds 60
```

如需断点复用已完成 run：

```bash
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset paper_long_sensitivity_v1 \
  --output-dir outputs/paper_training_long_sensitivity_v1_20260528 \
  --resume-completed \
  --progress-interval-seconds 60
```

### 8.1 full paper-long 启动与性能干预记录

#### 第一次 full run：已停止

2026-05-28 首次启动 full paper-long 后，发现普通 `nohup ... &` 后台进程会被当前工具环境清理，因此改用 tmux。随后发现未限制数值库线程时，单个 Python 进程启动了约 89 个线程，CPU 占用超过 3600%，但第一个 672-step baseline 长时间没有可见 step 进度。

| 项目 | 记录 |
|---|---|
| tmux session | `paper_long_sens_v1_20260528` |
| Python PID | `607902` |
| 输出目录 | `outputs/paper_training_long_sensitivity_v1_20260528_full` |
| 状态 | 已停止，未作为正式结果使用 |
| 停止原因 | 数值线程过度展开，且旧版本缺少 baseline step 级进度 |

#### 本轮代码修正

为适配服务器上的 CPU full-run，本轮新增：

1. `src/vpp_dso_sim/utils/runtime.py`：`configure_numeric_thread_limits(default_threads=8)`。
2. `examples/17_paper_training_experiment.py`：在导入实验模块前设置 `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB` 默认线程数。
3. `src/vpp_dso_sim/experiments/paper_training.py`：在 `numpy/pandas` 导入前设置数值线程默认值。
4. baseline rollout step 级进度：每 6 个仿真 step 写一次 `baseline_step`，进度 CSV 新增 `step` 和 `step_progress_pct`。

已验证：

```bash
./.venv-server/bin/python -m pytest -q tests/test_runtime_thread_limits.py
./.venv-server/bin/python -m pytest -q \
  tests/test_paper_training_experiment.py::test_baseline_rollout_reports_step_progress \
  tests/test_runtime_thread_limits.py
./.venv-server/bin/python -m pytest -q \
  tests/test_paper_training_experiment.py::test_paper_long_sensitivity_v1_preset_uses_structured_happo_config \
  tests/test_paper_training_experiment.py::test_sensitivity_v1_large_benchmark_config_extends_paper_scenario \
  tests/test_paper_training_experiment.py::test_paper_training_structured_happo_sensitivity_smoke \
  tests/test_paper_training_experiment.py::test_baseline_rollout_reports_step_progress \
  tests/test_runtime_thread_limits.py
```

上述测试结果均为通过。

#### GPU / CUDA 训练状态

本次已完成 HAPPO 设备路径改造：

| 项目 | 状态 |
|---|---|
| 服务器真实 GPU | 沙箱外 `nvidia-smi` 可见 `NVIDIA A800 80GB PCIe`，Driver `570.124.06`，CUDA `12.8` |
| 项目虚拟环境 PyTorch | 沙箱外 `torch 2.7.1+cu128`，`torch.cuda.is_available() == True`，`cuda_device_count == 1` |
| Codex 默认沙箱 | 默认沙箱内没有 `/dev/nvidia0` / `/dev/nvidiactl` / `/dev/nvidia-uvm`，所以普通 Codex 命令会显示 GPU 不可用 |
| HAPPO 训练/评估 | `HAPPOConfig.device` 支持 `auto` / `cpu` / `cuda` / `cuda:<index>` |
| 配置默认值 | `configs/happo_sensitivity_attention_v1.yaml`、`configs/european_lv_benchmark_v2_sensitivity_attention_v1.yaml`、`configs/happo_legacy_mlp.yaml` 均设置 `trainer.device: auto` |
| CUDA smoke | 2-step HAPPO train + frozen eval 已在 A800 上跑通，训练和评估摘要均记录 `resolved_device=cuda` |
| 训练中间进度 | 2026-06-01 新增 HAPPO `train_step` 进度回调，后续新启动/重启的 paper-long 会向 `experiment_progress.csv/jsonl` 写入 episode、step、reward_so_far、total_cost_so_far、violations_so_far、projection_gap_mw |

重要边界：

- `device: auto` 在你自己的服务器 shell 中会自动选择 CUDA；在 Codex 默认沙箱中会回退 CPU。
- 已经运行中的 paper-long 进程不会热切换到 GPU，因为 actor/critic、optimizer 和张量设备在进程启动时就确定了。
- GPU 只能加速 PyTorch actor/critic 前向、反向传播和张量运算；pandapower AC 潮流、安全投影、候选搜索仍主要是 CPU 串行/并行任务，所以总加速幅度不会等同于纯深度学习训练。
- checkpoint 保存时已转为 CPU state dict，因此 CUDA 训练产物仍可在 CPU 环境下加载与复现。

新启动 GPU 训练时，推荐使用新的输出目录：

```bash
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset paper_long_sensitivity_v1 \
  --output-dir outputs/paper_training_long_sensitivity_v1_gpu_YYYYMMDD \
  --progress-interval-seconds 60
```

如果希望强制 CUDA 而不是自动回退，需要将对应 YAML 的 `trainer.device` 设为 `cuda`；这样当 GPU 设备不可见时会直接报错，避免误以为在 GPU 上训练。

#### 当前 full run：正在运行

```text
tmux session: paper_long_sens_v1_gpu_resume_20260601
python pid: 1134738
output dir: outputs/paper_training_long_sensitivity_v1_20260528_thread8_progress
```

2026-06-01 10:09 CST 曾短暂启动新目录 `outputs/paper_training_long_sensitivity_v1_20260601_gpu`，但该 run 仍要重新读取 45 个 baseline。为避免浪费时间，已停止该新目录 run，改用旧目录 `--resume-completed` 复用已完成 baseline，并从未完成的 HAPPO 训练阶段继续。

当前 GPU resume 启动命令等价于：

```bash
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=8 \
OPENBLAS_NUM_THREADS=8 \
MKL_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 \
VECLIB_MAXIMUM_THREADS=8 \
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset paper_long_sensitivity_v1 \
  --output-dir outputs/paper_training_long_sensitivity_v1_20260528_thread8_progress \
  --resume-completed \
  --progress-interval-seconds 60
```

当前已确认：

| 指标 | 状态 |
|---|---|
| 进程线程数 | 训练进程 PID `1134738`，tmux session `paper_long_sens_v1_gpu_resume_20260601` |
| 数值线程环境 | `CUDA_VISIBLE_DEVICES=0`, `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB = 8` |
| GPU 预检 | `torch 2.7.1+cu128`, `cuda_available=True`, `cuda_device_count=1`, `cuda_device_name=NVIDIA A800 80GB PCIe` |
| GPU 实际占用 | 2026-06-01 10:19:57 CST，`nvidia-smi` 显示 PID `1134738` 占用约 `542 MiB` 显存，说明 HAPPO 进程已加载 CUDA 上下文 |
| 总任务数 | `185 = 45 baseline + 20 train + 120 frozen eval` |
| 当前完成数 | resume 后快速重读/复用 `45` 个 baseline；`train_done=0`, `eval_done=0` |
| 已完成 baseline | `seed=9401` 至 `seed=9405` 的 3 个 holdout 场景全部 baseline 已完成，共 `45 = 5 seeds * 3 scenarios * 3 baselines` |
| 首个 baseline 结果 | `reward_sum=1480.276752998925`, `total_cost=97682.94201531395`, `violations=0` |
| 第二个 baseline 结果 | `no_flex_holdout_peak_seed_9401`: `reward_sum=1476.3000712364324`, `total_cost=114912.90474537118`, `violations=0` |
| 第三个 baseline 结果 | `ac_validated_search_reference_holdout_peak_seed_9401`: `reward_sum=1471.578000246445`, `total_cost=106727.42289528696`, `violations=0` |
| 第四个 baseline 结果 | `rule_based_holdout_cloudy_seed_9401`: `reward_sum=1479.8741787110823`, `total_cost=99389.95599087849`, `violations=0` |
| 最近完成 baseline | `ac_validated_search_reference_holdout_reverseflow_seed_9405`: `reward_sum=1470.703018172785`, `total_cost=82606.55532005102`, `violations=0` |
| 最近 AC reference 摘要 | `steps=672`, `min_voltage_vm_pu=0.9746352752734547`, `max_voltage_vm_pu=1.03`, `max_line_loading_percent=48.71854063396267`, `total_cost=112557.73338393464` |
| 最近 AC reference 证书统计 | `accepted_candidate_ac_safe=672`, `fallback_count=0`; 该参考是 bounded candidate search，不是全局 OPF 上界 |
| Cloudy rule-based 摘要 | `steps=672`, `min_voltage_vm_pu=0.9504759170352529`, `max_voltage_vm_pu=1.0352797888339424`, `max_line_loading_percent=70.6571810599822`; `constraint_violations.csv` 仅表头，无违规记录 |
| Cloudy 对比观察 | `seed=9401`: `no_flex` 比 `rule_based` 成本高约 `14023.04`、无违规；`ac_validated_search_reference` 成本介于二者之间且线路 loading 更低。`seed=9402`: `no_flex` 比 `rule_based` 成本高约 `12668.01`；`ac_validated_search_reference` 比 `no_flex` 成本低约 `1045.95`、比 `rule_based` 成本高约 `11622.06`，但线路 loading 从 `70.56/65.62` 降到 `48.72`，最低电压提高到 `0.9746`，三者均无违规 |
| 当前 baseline 聚合观察 | 已写出 `45` 个 baseline summary；`baseline_safety_gate.csv` 显示 AC reference baseline gate passed。9 组 baseline 聚合均为 `total_violation_cells_mean=0`、`post_ac_security_pass_rate_mean=1.0`。`baseline_claim_readiness.json` 仍为 `paper_claim_ready=false`，原因是 guardrails 禁止把 bounded search 说成 OPF 最优/上界、禁止把代理电价结果说成真实市场利润 |
| 当前阶段 | 旧 CPU run 已在 `episode=51/120` 后被中断；GPU resume 已于 2026-06-01 10:16:07 CST 写入新的 `train_start`，开始 `happo_base_train_mixed_seed_9401` |
| 当前已观测进度 | 截至 2026-06-01 10:49:09 CST，GPU resume 尚未写出第一条 HAPPO episode 日志；当前已运行进程不会热加载新加的 `train_step` 回调，若需要 step 级训练进度需重启 resume |
| 当前吞吐估计 | `no_flex_holdout_peak_seed_9401` 在 8 线程下约 2.2 秒/step；`ac_validated_search_reference_holdout_peak_seed_9401` 全程约 3.5 秒/step |
| 当前训练趋势 | HAPPO episode 1-51 均 `violations=0`；first10 reward mean `1730.9096`、last10 reward mean `1703.7284`；critic loss 从 `0.011433` 降到 `0.003204`；最大 `projection_gap_mw=0.042825`。当前尚不能判断收敛完成，因为只跑完首个训练 run 的 51/120 episode，且 frozen eval 尚未开始 |
| 当前 CPU/GPU 使用 | 2026-06-01 10:49 CST，PID `1134738` 约 `316%` CPU、RSS 约 `1.85 GB`，并在 A800 上占用约 `542 MiB` 显存；`nvidia-smi` 瞬时 GPU util 约 `89%`，tmux pane 无额外未刷新的训练输出 |
| 当前错误状态 | 2026-06-01 10:49 CST 新 resume 日志未见 `Traceback` / `Error` / `baseline_gate_failed` / `nan` / `inf`；训练目录目前仍需等待 HAPPO episode metrics、loss metrics 与 checkpoint 落盘 |
| 进度日志健壮性 | 已修复 `experiment_progress.csv` 字段升级后的历史表头不一致问题：CSV schema 变化时会从 `experiment_progress.jsonl` 自动重建 CSV，避免 25/27/33 列混写导致 `pandas.read_csv` 失败 |

线程限制调优判断：

- 当前设置为 `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB = 8`，不是完全单核运行；但实测单进程约 `380%` CPU，说明 AC 潮流校验和 baseline rollout 仍有明显串行瓶颈。
- 在完成当前证明性/稳定性阶段后，可以做线程上限对照实验，例如 `8 -> 16 -> 32`，用同一个短 baseline 场景比较秒/step。不要直接放开到 `160`，否则 NumPy、pandapower 和 BLAS 线程可能互相抢占，出现 CPU 占用更高但 step 更慢。
- 更可能有效的加速方向是任务级并行：把不同 seed、不同 baseline、不同 holdout 或不同算法 run 分给多个独立进程，每个进程保留 `8-16` 个数值线程；这通常比单进程盲目吃满全部 CPU 更稳定。

若要在服务器 shell 中查看：

```bash
tmux list-sessions
tmux attach -t paper_long_sens_v1_gpu_resume_20260601
```

非交互监控：

```bash
tail -f outputs/paper_training_long_sensitivity_v1_20260528_thread8_progress/logs/gpu_resume_stdout.log
cat outputs/paper_training_long_sensitivity_v1_20260528_thread8_progress/logs/gpu_resume_preflight.log
tail -f outputs/paper_training_long_sensitivity_v1_20260528_thread8_progress/experiment_progress.jsonl
tail -f outputs/paper_training_long_sensitivity_v1_20260528_thread8_progress/experiment_progress.csv
```

若服务器断电或进程退出，优先用 resume 命令接续：

```bash
OMP_NUM_THREADS=8 \
OPENBLAS_NUM_THREADS=8 \
MKL_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 \
VECLIB_MAXIMUM_THREADS=8 \
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset paper_long_sensitivity_v1 \
  --output-dir outputs/paper_training_long_sensitivity_v1_20260528_thread8_progress \
  --resume-completed \
  --progress-interval-seconds 60
```

## 9. 当前仍需谨慎的问题

1. GPU resume 已启动并确认 PID `1134738` 占用 A800 显存；但只有 PyTorch 神经网络部分在 GPU 上，pandapower AC 潮流、安全投影和候选搜索仍主要在 CPU 上。
2. `ac_validated_search_reference` 每步会做多候选 AC 潮流校验；baseline 阶段已经完成并通过 safety gate，但这是本次全流程最慢的已验证阶段之一。
3. `sensitivity_attention_v1` 结构化 DSO actor 每步涉及有限差分灵敏度路径，即使已有缓存设置，也仍是训练阶段性能风险。旧 CPU run 在首个 HAPPO train run 跑到 `51/120` episode 后已中断；GPU resume 已重新进入 `train_start`，需等待新的 episode 日志确认吞吐变化。
4. 2-step smoke 不能证明收敛，只能证明训练/评估链路接通。
5. 当前 `paper_long_sensitivity_v1` 默认只把 HAPPO 作为结构化深度 MARL 主算法；HATRPO/MATD3/HASAC 若要作为同等结构化对照，需要后续升级其 DSO actor 接口。
6. full run 已加数值线程上限；后续如果服务器负载允许，可系统比较 `8/16/32` 线程，但不要在同一输出目录混跑。
7. 不建议直接完全解除线程限制。先前未限制时单进程曾展开约 89 个线程、CPU 超过 3600%，但这不等于更快；pandapower / NumPy / BLAS 可能进入线程争抢或内存带宽瓶颈。正式论文级结果应记录线程配置。
8. 当前正在跑的 PID `1134738` 是新增 `train_step` 进度回调之前启动的进程，所以不会写 step 级训练进度。是否为此重启需要权衡：重启会损失当前已跑的首个 GPU episode 时间，但能获得更细的训练可观测性。由于旧 CPU基准约 `76 min/episode`，当前 GPU run 在 `train_start` 后约 33 分钟尚无 episode 输出仍不能判定异常；建议先等到接近一个旧 CPU episode 基准再决定是否重启。

## 10. 结论

本轮已经把 paper-long 实验入口从旧 flat/legacy 配置升级为当前算法可用的 `paper_long_sensitivity_v1` 协议，并验证：

1. 大场景配置可加载。
2. 结构化 HAPPO actor 真实参与训练。
3. YAML 稳定化参数不会再被 paper training runner 丢弃。
4. reward、loss、KL、entropy、grad norm、收敛摘要、TensorBoard 图片和 HTML 总报告都有明确落盘位置。

full paper-long 已经以线程受限、step 级进度和 GPU 可见环境重新启动；baseline 阶段已经完成并通过 AC reference safety gate，GPU resume 已在 `2026-06-01 10:16:07 CST` 重新进入 HAPPO train。当前不能声明论文级实验结果已经产生。下一步应持续监控 HAPPO train run 是否写出 `training_loss_metrics.csv`、`training_episode_metrics.csv`、checkpoint，以及后续 frozen eval 是否覆盖 5 seeds、3 holdout 场景和 best/final checkpoint。
