# HAPPO subprocess shared rollout 工程报告

日期：2026-06-14

## 1. 目标

把现有 HAPPO shared rollout 从 serial backend 升级为真正的 subprocess backend：

- 主进程只持有一套 HAPPO actor / critic / optimizer。
- worker 进程只持有独立环境实例，只执行 reset / step / close。
- 主进程统一计算 action、old log probability、value。
- 多个 worker 并发执行 env.step。
- 合并 worker fragment 后统一 GAE / return / advantage / HAPPO sequential update。
- 保留 serial backend 作为 debug / regression 模式。

## 2. 修改文件

| 文件 | 修改内容 |
|---|---|
| `src/vpp_dso_sim/learning/shared_rollout_workers.py` | 新增 subprocess worker 接口，使用 multiprocessing spawn；worker 独立创建环境，只处理 reset/step/close；worker 内强制 OMP/MKL/OPENBLAS/NUMEXPR/VECLIB 线程为 1。 |
| `src/vpp_dso_sim/learning/advanced_marl.py` | 新增 `shared_rollout_backend="subprocess"`；主进程统一 policy forward；并发发送 step；合并 transition；记录性能和 on-policy 正确性指标；关闭 worker 并记录 exitcode。 |
| `src/vpp_dso_sim/experiments/paper_training.py` | 将 shared rollout backend、worker 数、fragment steps、HAPPO reward dynamic report 控制项传入 HAPPOConfig。 |
| `examples/17_paper_training_experiment.py` | 新增 CLI 参数：`--happo-shared-rollout-backend subprocess`、动态报告控制参数。 |
| `tests/test_hasac_happo.py` | 新增 subprocess backend 一次 rollout + update 测试；覆盖 worker pid/exitcode、policy version、ratio、NaN、worker offset、性能字段。 |

## 3. 正确性设计

### 3.1 一套权重

worker 进程没有 actor、critic、optimizer，也不会 optimizer.step。主进程负责：

- actor / critic forward；
- action sampling；
- old_log_prob 保存；
- value 估计；
- rollout buffer；
- GAE / return；
- HAPPO actor sequential update；
- centralized critic update；
- checkpoint / logging。

### 3.2 on-policy 保证

每个 merged batch 写入同一个 `policy_version`。如果 batch 中混入不同 policy version，会显式报错。

更新前记录：

- `ratio_mean_before_first_update`
- `ratio_std_before_first_update`
- `old_log_prob_nan_count`
- `new_log_prob_nan_count`
- `advantage_mean/std`
- `return_mean/std`

smoke 结果中 `ratio_mean_before_first_update = 1.0`，说明 rollout 时保存的 old_log_prob 与 update 初始策略一致。

### 3.3 GAE / bootstrap

GAE 沿 worker 自己的时间轴计算。fragment cut 使用对应 worker 的末尾 next_global_state bootstrap；真实 terminal 不 bootstrap。已有测试覆盖：

- fragment cut 使用 next value；
- true terminal 忽略 next value；
- worker 时间轴不会跨 worker 拼接。

### 3.4 centralized critic

每条 transition 保存自己的 `critic_state_vec`，bootstrap 也使用对应 worker 的末尾 `critic_state_vec`。没有对多个 worker 的 global_state 求平均，也没有只使用 worker_0。

## 4. 性能日志

新增/保留字段：

- `shared_rollout_backend`
- `num_workers`
- `worker_count`
- `rollout_fragment_steps`
- `rollout_collect_seconds`
- `policy_forward_seconds`
- `env_step_wall_seconds`
- `env_step_worker_mean_seconds`
- `env_step_worker_max_seconds`
- `wait_for_workers_seconds`
- `update_seconds`
- `total_update_seconds`
- `samples_collected`
- `samples_per_second`
- `slowest_worker_id`

这些字段写入 `happo_update_metrics.csv`，summary 中记录 worker pid / exitcode。

## 5. 验证结果

### 5.1 单元/回归测试

通过：

```bash
./.venv-server/bin/python -m pytest \
  tests/test_hasac_happo.py::test_happo_shared_rollout_multi_worker_completes_one_update \
  tests/test_hasac_happo.py::test_happo_shared_rollout_fragments_continue_worker_time_axis \
  tests/test_hasac_happo.py::test_happo_shared_rollout_workers_start_on_distinct_time_slices \
  tests/test_hasac_happo.py::test_happo_shared_rollout_subprocess_backend_completes_one_update \
  tests/test_paper_training_experiment.py::test_happo_shared_rollout_defaults_are_off \
  tests/test_deep_rl_training.py::test_gae_bootstrap_fragment_cut_uses_next_value \
  tests/test_deep_rl_training.py::test_gae_bootstrap_true_terminal_ignores_next_value \
  -q
```

结果：7 passed。

报告 IO/字段回归：

```bash
./.venv-server/bin/python -m pytest \
  tests/test_hasac_happo.py::test_happo_shared_rollout_subprocess_backend_completes_one_update \
  tests/test_hasac_happo.py::test_happo_reward_v2_step_metrics_cover_reward_and_security_columns \
  tests/test_paper_training_experiment.py::test_happo_shared_rollout_defaults_are_off \
  -q
```

结果：3 passed。

### 5.2 serial vs subprocess smoke 对比

同一配置，2 workers，fragment steps=2，4 samples：

| backend | rollout_collect_seconds | total_update_seconds | samples_per_second | ratio_mean_before_first_update | NaN count |
|---|---:|---:|---:|---:|---:|
| serial | 25.7271 | 26.4162 | 0.1555 | 0.99999994 | 0 |
| subprocess | 14.3143 | 15.0223 | 0.2794 | 1.0 | 0 |

结论：subprocess 在该 smoke 下采样吞吐约提升 1.8 倍。正式收益会受 pandapower 单步耗时、IPC、报告 IO、worker 数影响。

### 5.3 GPU smoke

输出目录：

`outputs/test_shared_rollout_gpu_subprocess_20260614_smoke`

关键结果：

- `resolved_device = cuda`
- `shared_rollout_backend = subprocess`
- worker pids: `3770614`, `3770957`
- worker exitcodes: `0`, `0`
- `rollout_collect_seconds = 14.9507`
- `samples_per_second = 0.2675`
- `ratio_mean_before_first_update = 1.0`
- `ratio_std_before_first_update = 6.1e-7`
- `old_log_prob_nan_count = 0`
- `new_log_prob_nan_count = 0`

结论：主进程 HAPPO 网络在 CUDA 上完成了一次 rollout + update + frozen eval，worker 正常退出，无僵尸进程证据。

## 6. 正式 paper-long 主实验

先前曾短暂启动过：

`outputs/paper_training_long_sensitivity_v1_happo_subprocess_gpu_20260614`

该 run 在第一轮 update 前被停止，原因是启动命令继承了 `OMP/MKL/OPENBLAS=8`，4 个 worker 可能产生过度线程竞争。该目录只保留为半启动痕迹，不作为正式结果目录。

随后又短暂启动过 4-worker 线程限制版：

`outputs/paper_training_long_sensitivity_v1_happo_subprocess_gpu_thread1_20260614`

该 run 完成了 3/120 个 episode 后停止，用于切换到更适合 672-step horizon 的 7-worker 配置。该目录可作为早期 4-worker 对比，不作为当前正式主结果目录。

当前有效 run 已重新启动为 7 workers，数值库线程限制为 1。选择 7 workers 的原因是 `7 workers * 96 fragment steps = 672 steps`，每次 HAPPO update 能覆盖完整 paper-long 日尺度 horizon，避免 8 workers + 96 fragment 的时间片重复。

已启动 tmux：

```bash
tmux list-sessions
# paper_long_happo_subprocess_w7_20260614
```

输出目录：

`outputs/paper_training_long_sensitivity_v1_happo_subprocess_gpu_w7_20260614`

启动命令语义：

```bash
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset paper_long_sensitivity_v1 \
  --config-path configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v3_1_market_safety.yaml \
  --output-dir outputs/paper_training_long_sensitivity_v1_happo_subprocess_gpu_w7_20260614 \
  --algorithms happo \
  --seeds 9401 \
  --hparam-cases higher_entropy \
  --happo-shared-rollout \
  --happo-shared-rollout-workers 7 \
  --happo-rollout-fragment-steps 96 \
  --happo-shared-rollout-backend subprocess \
  --happo-reward-dynamic-report-every-episodes 1 \
  --checkpoint-selection both \
  --progress-interval-seconds 300 \
  --verbose-progress
```

当前进程证据：

- 主训练进程 PID：`4002179`
- subprocess worker PID：`4002634`, `4002991`, `4003386`, `4003760`, `4004149`, `4004508`, `4004788`
- worker 初始环境确认：`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`
- GPU 上主训练进程显存约 468 MiB。
- 截至启动验收时，7 个 worker 已全部进入环境交互，尚未完成第一个 672-step merged update；`happo_episode_metrics.csv` 和 `happo_update_metrics.csv` 仍需等训练函数落盘。若需要实时曲线，应后续把 episode/update metrics 改为每 episode 增量写入。

这是“一个主模型 + 4 个环境 worker”的 shared rollout，不是 4 个独立 shard，也不是 4 套权重。

## 7. 如何查看

实时日志：

```bash
tail -f outputs/paper_training_long_sensitivity_v1_happo_subprocess_gpu_w7_20260614/logs/train_stdout.log
```

tmux：

```bash
tmux attach -t paper_long_happo_subprocess_w7_20260614
```

训练指标：

```bash
python - <<'PY'
import pandas as pd
base = "outputs/paper_training_long_sensitivity_v1_happo_subprocess_gpu_w7_20260614/runs/happo_higher_entropy_train_mixed_seed_9401/train"
print(pd.read_csv(base + "/happo_episode_metrics.csv").tail())
print(pd.read_csv(base + "/happo_update_metrics.csv").tail())
PY
```

## 8. 当前限制

1. 正式 paper-long 主实验刚启动，第一轮长 fragment 尚未完成；最终收敛结论需要等 `happo_episode_metrics.csv` 和 `happo_update_metrics.csv` 持续产生。
2. worker offset 沿用了现有 serial shared rollout 的 `start_step = worker_index * fragment_steps` 逻辑。它能错开 PV/load/price/time index，但不是通过从 episode 起点逐步 warm-up 到 offset 的物理状态回放。SoC/EVCS 物理一致性仍需后续专门审计。
3. 目前主进程仍顺序做 policy forward，subprocess 只并行 env.step。若后续 GPU policy forward 成为瓶颈，再考虑 batched policy forward 优化。
4. 正式扩展到 6 workers 前，应先观察 4 workers 的 `samples_per_second`、`slowest_worker_id`、CPU 占用和 IPC 开销。
5. 本次只启动一套正式 HAPPO `higher_entropy + seed 9401` 主实验。全 5 seeds / 4 hparam cases 会非常重，建议确认主实验稳定后再扩展。

## 9. 下一步建议

1. 等第一轮 update 完成后，检查 `happo_update_metrics.csv`：
   - `shared_rollout_backend == subprocess`
   - `num_workers == 4`
   - `ratio_mean_before_first_update` 接近 1
   - old/new log prob NaN count 为 0
   - `rollout_collect_seconds` 明显小于 serial 估计
2. 若 CPU 使用率低且 worker 等待时间高，检查 IPC payload 和报告 IO。
3. 若 worker 速度差异明显，检查 `slowest_worker_id` 对应场景/潮流是否异常。
4. 若 4 workers 稳定，再尝试 6 workers；不要直接跳到 12 workers。
