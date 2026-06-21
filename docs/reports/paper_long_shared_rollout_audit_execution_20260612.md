# Paper-Long Shared Rollout Audit and Execution Status - 2026-06-12

## 背景

用户任务文件：`docs/tasks/parallel.md`

核心目标是把当前 HAPPO 训练从两类旧模式：

1. 单环境顺序 rollout；
2. 多 shard 独立训练多套权重；

扩展为一种新模式：

> 多个 worker 使用同一套行为策略快照采样，合并 trajectory / fragment 后，对同一套 HAPPO actor / critic 统一更新。

这不是增加 8 个独立实验，而是加速一套主模型。

## 审核结论

### 1. 短跑 shard probe

当前 CLI 已经支持：

- `--train-episodes`
- `--seeds`
- `--hparam-cases`

因此 1 到 3 episode 的短跑 probe 不需要新 launcher。

本次新增：

- `scripts/summarize_happo_probe.py`
- `tests/test_happo_probe_summary.py`

用于从已完成 shard 的 `happo_episode_metrics.csv` 和 `happo_update_metrics.csv` 汇总：

- seed
- hparam case
- final / mean / std / best reward
- final cost
- violation count
- projection gap
- critic loss
- policy loss
- entropy
- approximate KL
- NaN / Inf 状态

当前正在运行的 12 个 shard 尚未写出 `happo_episode_metrics.csv`，因为训练器通常在 run 结束时才落盘。因此当前 summary 输出为：

```text
No completed HAPPO episode metric files found.
```

这不是脚本失败，而是当前训练尚未到落盘点。

### 2. shared rollout 不是命令层改动

当前 HAPPO 代码位于：

- `src/vpp_dso_sim/learning/advanced_marl.py::train_happo`

当前实现是：

1. 采样一个环境 episode；
2. 立即在这个 episode 的 rollout 上更新 critic；
3. 再按 HAPPO 顺序更新 DSO actor、dispatch actors、portfolio actors。

当前 GAE 原始 helper 位于：

- `src/vpp_dso_sim/learning/deep_rl.py::_gae_returns_advantages`

它把 rollout 末尾当成有限 horizon 的零 bootstrap 终点。这对旧 full-episode 训练可接受，但对 `fragment_steps=96` 这种人为截断不正确。

因此，`parallel.md` 中的 shared rollout 必须修改训练器内部 buffer / GAE / update 逻辑，不能只靠新增 CLI 参数完成。

### 3. 资源审核

当前 GPU 状态：

- A800 80GB
- 已占用约 75.8GB
- 12 个 `pl2_sa_*` shard 都还在运行

结论：

- 不应在当前资源状态下再启动新的 GPU paper-long shared rollout smoke；
- 否则极易 OOM；
- 当前可继续做 CPU 单元测试和默认关闭的源码改造。

## 本次已执行修改

### 1. 新增 bootstrap-aware GAE helper

文件：

- `src/vpp_dso_sim/learning/deep_rl.py`
- `tests/test_deep_rl_training.py`

新增：

- `_gae_returns_advantages_bootstrap(...)`

用途：

- 支持 fragment cut bootstrap；
- 支持 true terminal 不 bootstrap；
- 为后续 shared rollout 的 fragment 采样做基础。

验证：

```bash
./.venv-server/bin/python -m pytest \
  tests/test_deep_rl_training.py::test_gae_and_ppo_clip_helpers_are_finite_and_clipped \
  tests/test_deep_rl_training.py::test_gae_bootstrap_fragment_cut_uses_next_value \
  tests/test_deep_rl_training.py::test_gae_bootstrap_true_terminal_ignores_next_value \
  -q
```

结果：通过。

### 2. 新增短跑 probe summary 脚本

文件：

- `scripts/summarize_happo_probe.py`
- `tests/test_happo_probe_summary.py`

命令：

```bash
./.venv-server/bin/python scripts/summarize_happo_probe.py \
  outputs/paper_training_long_dispatch_set_attention_v1_happo_20260612_sharded_v2 \
  --output-dir outputs/paper_training_long_dispatch_set_attention_v1_happo_20260612_sharded_v2/probe_summary
```

当前结果：

```text
No completed HAPPO episode metric files found.
Summary directory: outputs/paper_training_long_dispatch_set_attention_v1_happo_20260612_sharded_v2/probe_summary
```

### 3. 新增 shared rollout 配置和 CLI 字段

文件：

- `src/vpp_dso_sim/learning/advanced_marl.py`
- `src/vpp_dso_sim/experiments/paper_training.py`
- `examples/17_paper_training_experiment.py`
- `tests/test_paper_training_experiment.py`

新增字段：

- `shared_rollout_enabled`
- `shared_rollout_workers`
- `shared_rollout_backend`
- `rollout_fragment_steps`
- `rollout_policy_version_check`

Paper-long CLI 新增参数：

```bash
--happo-shared-rollout
--happo-shared-rollout-workers
--happo-rollout-fragment-steps
--happo-shared-rollout-backend
```

默认值全部保持旧行为：

```text
shared_rollout_enabled = False
shared_rollout_workers = 1
shared_rollout_backend = serial
rollout_fragment_steps = None
```

### 4. 防止 shared rollout 静默 no-op

文件：

- `src/vpp_dso_sim/learning/advanced_marl.py`
- `tests/test_hasac_happo.py`

原因：

当前只是完成配置/CLI 接入，完整 shared rollout collector 还没有实现。如果用户此时传入 `--happo-shared-rollout`，旧代码会悄悄按单 worker HAPPO 训练，造成错误实验结论。

本次加了 guard：

- 启用 shared rollout 时明确抛出 `NotImplementedError`；
- 等下一阶段实现完 collector / merged batch / bootstrap update 后再移除。

验证：

```bash
./.venv-server/bin/python -m pytest \
  tests/test_hasac_happo.py::test_happo_shared_rollout_guard_prevents_silent_noop \
  tests/test_hasac_happo.py::test_happo_training_writes_sequential_update_artifacts \
  tests/test_paper_training_experiment.py::test_happo_shared_rollout_defaults_are_off \
  -q
```

结果：通过。

## 已通过测试

```text
tests/test_deep_rl_training.py::test_gae_and_ppo_clip_helpers_are_finite_and_clipped
tests/test_deep_rl_training.py::test_gae_bootstrap_fragment_cut_uses_next_value
tests/test_deep_rl_training.py::test_gae_bootstrap_true_terminal_ignores_next_value
tests/test_happo_probe_summary.py
tests/test_paper_training_experiment.py::test_happo_shared_rollout_defaults_are_off
tests/test_hasac_happo.py::test_happo_training_writes_sequential_update_artifacts
tests/test_hasac_happo.py::test_happo_shared_rollout_guard_prevents_silent_noop
```

## 追加执行结果：serial shared rollout 已接入

本报告初版写完后，继续按 TDD 实现了最小 serial shared rollout。

### 已新增能力

文件：

- `src/vpp_dso_sim/learning/advanced_marl.py`
- `tests/test_hasac_happo.py`

新增行为：

1. `shared_rollout_enabled=True` 或 `shared_rollout_workers>1` 时进入 shared rollout 分支；
2. 当前只支持 `shared_rollout_backend="serial"`；
3. 每个 worker 使用同一套当前 actor / critic 参数采样；
4. worker 采样阶段不做 optimizer update；
5. 多 worker fragment 合并为一个 batch；
6. 每条样本记录：
   - `worker_index`
   - `terminal`
   - `policy_version`
7. GAE 按 worker 分组计算，不跨 worker 拼接时间轴；
8. 如果 fragment cut 不是真实 terminal，则使用 critic bootstrap；
9. 合并 batch 后复用现有 HAPPO 顺序更新：
   - centralized value critic
   - DSO actor
   - dispatch actors
   - portfolio actors
10. `update_metrics` 新增：
    - `policy_version`
    - `worker_count`
    - `rollout_fragment_steps`
    - `shared_rollout_enabled`
    - `bootstrap_value_mean`
    - `fragment_cut_count`
11. `happo_training_summary.json` 新增：
    - `shared_rollout_enabled`
    - `shared_rollout_workers`
    - `shared_rollout_backend`
    - `rollout_fragment_steps`
    - `shared_rollout_batches`
    - `shared_rollout_total_samples`
    - `shared_rollout_fragment_cut_count`
    - `shared_rollout_policy_version_mismatch_count`
    - `shared_rollout_bootstrap_value_mean`

### 已通过 shared rollout smoke

命令：

```bash
./.venv-server/bin/python -m pytest \
  tests/test_hasac_happo.py::test_happo_shared_rollout_multi_worker_completes_one_update \
  -q
```

测试场景：

- CPU
- 2 workers
- horizon = 4
- rollout fragment = 2
- 1 次 HAPPO update

测试验证：

- `shared_rollout_enabled=True`
- `shared_rollout_workers=2`
- `shared_rollout_total_samples=4`
- `shared_rollout_fragment_cut_count=2`
- `step_metrics.worker_index` 包含 `{0, 1}`
- `update_metrics` 包含 `policy_version` 和 `worker_count`
- `ratio_mean` 有效

### 已通过回归测试

命令：

```bash
./.venv-server/bin/python -m pytest \
  tests/test_hasac_happo.py::test_happo_training_writes_sequential_update_artifacts \
  tests/test_hasac_happo.py::test_happo_shared_rollout_multi_worker_completes_one_update \
  tests/test_paper_training_experiment.py::test_happo_shared_rollout_defaults_are_off \
  tests/test_deep_rl_training.py::test_gae_and_ppo_clip_helpers_are_finite_and_clipped \
  tests/test_deep_rl_training.py::test_gae_bootstrap_fragment_cut_uses_next_value \
  tests/test_deep_rl_training.py::test_gae_bootstrap_true_terminal_ignores_next_value \
  tests/test_happo_probe_summary.py \
  -q
```

结果：

```text
8 passed
```

### 已通过 paper-training CLI shared rollout smoke

为了验证不只是单元测试可用，还从正式实验入口跑了一个 CPU smoke。由于当前 GPU 被旧 shard 占满，本次显式禁用 CUDA：

```bash
CUDA_VISIBLE_DEVICES='' ./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset smoke \
  --output-dir outputs/test_happo_shared_rollout_cli_smoke_20260612 \
  --algorithms happo \
  --seeds 9709 \
  --hparam-cases base \
  --train-episodes 1 \
  --horizon-steps 4 \
  --eval-horizon-steps 4 \
  --checkpoint-selection final \
  --happo-shared-rollout \
  --happo-shared-rollout-workers 2 \
  --happo-rollout-fragment-steps 2 \
  --no-html \
  --no-tensorboard \
  --progress-interval-seconds 60
```

结果：

```text
Exit code: 0
Runs: 2
Evaluation rows: 1
```

训练 summary 关键字段：

```text
shared_rollout_enabled = True
shared_rollout_workers = 2
shared_rollout_backend = serial
rollout_fragment_steps = 2
shared_rollout_batches = 1
shared_rollout_total_samples = 4
shared_rollout_fragment_cut_count = 2
shared_rollout_policy_version_mismatch_count = 0
resolved_device = cpu
final_episode_reward = 6.2096421905087364
```

输出位置：

```text
outputs/test_happo_shared_rollout_cli_smoke_20260612
```

关键证据文件：

- `runs/happo_base_train_mixed_seed_9709/train/happo_training_summary.json`
- `runs/happo_base_train_mixed_seed_9709/train/happo_update_metrics.csv`
- `runs/happo_base_train_mixed_seed_9709/train/happo_step_metrics.csv`
- `run_index.csv`
- `convergence_summary.csv`

## 仍未完成 / 仍需谨慎

1. GPU paper-long shared rollout smoke 尚未启动，因为当前 A800 80GB 已被 12 个旧 shard 占用约 75.8GB。
2. 目前 shared rollout 后端是 serial worker，不是 multiprocessing；它验证算法语义，但不一定立刻带来最大墙钟加速。
3. shared rollout 的 reward dynamic card 已能生成，但 shared 分支的 step metrics 先覆盖关键列，尚未完全等同旧单 worker 路径的全部诊断列。
4. 当前 12 个旧 shard 仍是独立模型训练，不是 shared-weight rollout；它们只适合作为短跑配置筛选。

## 当前建议

当前可以做两件事：

1. 让 12 个独立 shard 跑到足够 probe 数据后停止，用 `scripts/summarize_happo_probe.py` 选择主配置；
2. 释放 GPU 后启动一个小规模 shared rollout GPU smoke，例如 4 workers + 96 steps，再升级到 paper-long 主跑。

## 2026-06-12 23:34 CST 最终审核更新

### 当前正式实验状态

正式 paper-long shared rollout 主实验尚未启动。

当前只完成了审核、修复和 smoke 验证。这样做是必要的，因为在准备启动正式实验前发现一个会影响 paper-long 结论的 shared rollout 时间轴问题。

### 发现的关键问题：fragment 会重复采样 episode 起点

问题位置：

- `src/vpp_dso_sim/learning/advanced_marl.py::train_happo`
- shared rollout 分支中的 worker 采样逻辑

问题表现：

当配置为：

```text
horizon_steps = 672
rollout_fragment_steps = 96
shared_rollout_workers = 4
```

旧实现会在每一轮 HAPPO update 时重新创建 worker 环境并从 `step=0` reset。这样训练并不会逐步覆盖 672 step paper-long 序列，而是反复训练每个 profile 的前 96 step。

这会导致：

1. paper-long 名义上是 672 step，但训练数据主要来自 episode 开头；
2. 后半段的峰荷、阴天、反向潮流等时序状态可能没有被策略充分学习；
3. `fragment cut + bootstrap` 虽然数学上存在，但环境状态没有接续，因此不是真正的连续 rollout fragment；
4. 启动正式实验会得到有偏结果。

### 已按 TDD 修复

新增测试：

```bash
./.venv-server/bin/python -m pytest \
  tests/test_hasac_happo.py::test_happo_shared_rollout_fragments_continue_worker_time_axis \
  -q
```

红灯结果：

```text
first_worker_steps == {0: [0, 1], 1: [0, 1]}
expected           == {0: [0, 1], 1: [2, 3]}
```

修复方式：

1. shared rollout worker 不再每个 update 重新创建并 reset；
2. worker 环境和 observation 在不同 update 之间保留；
3. 每个 worker 从上一次 fragment 结束处继续采样；
4. 只有到达真实 terminal 后才 reset；
5. terminal 前的 fragment cut 继续使用 critic bootstrap；
6. summary 新增 `shared_rollout_worker_terminal_reset_count`，用于记录真实 terminal 后 worker reset 次数。

修复后测试通过：

```text
episode  worker_index
0        0               [0, 1]
         1               [0, 1]
1        0               [2, 3]
         1               [2, 3]
```

### 已补齐 shared rollout 可观测性

新增：

1. `progress_callback` 中的 `shared_rollout_step` 事件；
2. `happo_dispatch_private_profit_trace.csv` 中的：
   - `worker_index`
   - `policy_version`

原因：

paper-long 运行中如果没有 worker 级进度事件，会看起来像进度条长时间不动；如果 trace 不带 worker 标识，则多个 worker 的同一 step 会混在一起，后续 reward 诊断不可靠。

### 最新已通过验证

单元测试：

```bash
./.venv-server/bin/python -m pytest \
  tests/test_hasac_happo.py::test_happo_shared_rollout_multi_worker_completes_one_update \
  tests/test_hasac_happo.py::test_happo_shared_rollout_fragments_continue_worker_time_axis \
  tests/test_paper_training_experiment.py::test_happo_shared_rollout_defaults_are_off \
  tests/test_deep_rl_training.py::test_gae_bootstrap_fragment_cut_uses_next_value \
  tests/test_deep_rl_training.py::test_gae_bootstrap_true_terminal_ignores_next_value \
  -q
```

结果：

```text
5 passed
```

GPU CLI smoke：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 \
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset paper_long_sensitivity_v1 \
  --config-path configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v3_1_market_safety.yaml \
  --output-dir outputs/test_shared_rollout_gpu_cli_smoke_20260612_continuous_patch \
  --algorithms happo \
  --seeds 9401 \
  --hparam-cases higher_entropy \
  --train-episodes 2 \
  --horizon-steps 4 \
  --eval-horizon-steps 4 \
  --checkpoint-selection final \
  --happo-shared-rollout \
  --happo-shared-rollout-workers 2 \
  --happo-rollout-fragment-steps 2 \
  --no-html \
  --no-tensorboard \
  --progress-interval-seconds 1 \
  --verbose-progress
```

结果：

```text
resolved_device = cuda
cuda_available = True
cuda_device_name = NVIDIA A800 80GB PCIe
shared_rollout_enabled = True
shared_rollout_workers = 2
shared_rollout_backend = serial
rollout_fragment_steps = 2
shared_rollout_batches = 2
shared_rollout_total_samples = 8
shared_rollout_fragment_cut_count = 2
shared_rollout_worker_terminal_reset_count = 2
shared_rollout_policy_version_mismatch_count = 0
```

GPU smoke 的 worker 时间轴证据：

```text
episode  worker_index
0        0               [0, 1]
         1               [0, 1]
1        0               [2, 3]
         1               [2, 3]
```

输出目录：

```text
outputs/test_shared_rollout_gpu_cli_smoke_20260612_continuous_patch
```

### 正式 paper-long 启动建议

建议先启动单套主模型，而不是 8 个独立 shard：

- algorithm: `happo`
- seed: `9401`
- hparam case: `higher_entropy`
- train episodes: `120`
- horizon: `672`
- eval horizon: `672`
- shared rollout workers: `4`
- fragment steps: `96`
- checkpoint selection: `final`
- baseline: 跳过，只跑 HAPPO train + frozen eval

注意：

当前 shared rollout backend 仍是 `serial`。它已经满足“同一套权重 + 同一 policy version + 合并 batch 更新”的算法语义，但不是 multiprocessing 并行，因此墙钟加速有限。真正 CPU 墙钟加速需要后续实现 multiprocessing/vectorized worker backend。

### 2026-06-12 23:45 CST 正式主实验早期健康检查

正式主实验已启动：

```text
tmux session = paper_long_shared_happo_main_20260612
pid = 375498
output_dir = outputs/paper_training_long_shared_rollout_happo_20260612_main
```

启动命令语义：

```text
algorithm = happo
seed = 9401
hparam_case = higher_entropy
train_episodes = 120
horizon_steps = 672
eval_horizon_steps = 672
shared_rollout_workers = 4
rollout_fragment_steps = 96
checkpoint_selection = final
baseline algorithms = skipped
```

运行状态证据：

```text
process status = Rl+
elapsed = 08:32
cpu = 233%
gpu process = ./.venv-server/bin/python
gpu memory for training process = 568 MiB
```

当前 progress JSONL 已写入 shared rollout step：

```text
23:39:25 worker=0 local_step=24/96 reward_so_far=-32.0738 total_cost_so_far=1139.1683 violations=0
23:42:01 worker=0 local_step=48/96 reward_so_far=-63.0310 total_cost_so_far=2410.8431 violations=0
23:44:38 worker=0 local_step=72/96 reward_so_far=-81.3201 total_cost_so_far=3378.4145 violations=0
```

判断：

1. 训练没有卡死，worker 0 正在持续推进；
2. 当前仍处于第一个 shared rollout batch，还没有完成第一个 HAPPO update；
3. 因此 `happo_episode_metrics.csv`、`happo_update_metrics.csv`、checkpoint 还未落盘是正常现象；
4. 当前 backend 是 `serial`，所以 4 个 worker 是顺序采样，不是 multiprocessing 并发采样；
5. 按目前速度，完整 paper-long 可能需要很长时间。第一个 update 完成后应立即检查：
   - `critic_loss`
   - `dso_loss`
   - `dispatch_loss`
   - `portfolio_loss`
   - `ratio_mean`
   - `approx_kl`
   - `target_kl_exceeded`
   - `reward_dynamic_cards`
   - `dispatch_private_profit_trace`

当前建议：

先不要中断，等第一个 shared rollout update 产出完整 metrics 后再决定是否：

1. 继续当前 120-update 训练；
2. 缩短为 staged pilot，例如先 7 或 14 updates 覆盖完整 672-step day；
3. 实现 multiprocessing shared rollout backend 后重启正式长实验。

### 2026-06-13 00:28 CST 更正：旧正式 run 已停止

后续核查发现，`paper_long_shared_happo_main_20260612` 的第一批 shared rollout 中 4 个 worker 从相同起点采样，导致第一批轨迹完全重复。典型证据如下：

```text
worker 0: step 0-95, reward/cost 序列 A
worker 1: step 0-95, reward/cost 序列 A
worker 2: step 0-95, reward/cost 序列 A
worker 3: step 0-95, reward/cost 序列 A
```

因此该 run 不能作为正式并行实验结果，只能作为 shared-rollout 审计失败样本保留。会话已停止：

```text
tmux session stopped = paper_long_shared_happo_main_20260612
stopped_at = 2026-06-13 00:28:17 CST
```

### 2026-06-13 shared-rollout 修复

修复点：

1. `MultiAgentVPPDSOEnv.reset()` 增加 `start_step` 参数，默认仍为 0，不影响旧调用；
2. HAPPO shared rollout 初始化每个 worker 时使用错开的起始时间片：
   - worker 0 从 `0` 开始；
   - worker 1 从 `fragment_steps` 开始；
   - worker 2 从 `2 * fragment_steps` 开始；
   - worker 3 从 `3 * fragment_steps` 开始；
3. terminal reset 后，worker 回到自己的固定起始时间片，避免所有 worker 重置到 step 0；
4. `happo_training_summary.json` 增加 `shared_rollout_worker_start_offsets`；
5. `happo_step_metrics.csv` 和 dispatch private profit trace 增加 `worker_start_step`；
6. dispatch private profit trace 增加 actor 行为诊断：
   - `policy_normalized_aggregate_action`
   - `policy_normalized_der_action_mean`
   - `policy_normalized_der_action_std`

新增回归测试：

```text
tests/test_hasac_happo.py::test_happo_shared_rollout_workers_start_on_distinct_time_slices
```

该测试先在旧实现下失败：

```text
actual   = {0: [0, 1], 1: [0, 1]}
expected = {0: [0, 1], 1: [2, 3]}
```

修复后通过，证明 worker 不再复制同一时间片。

### 2026-06-13 GPU smoke 验证

GPU smoke 输出目录：

```text
outputs/test_shared_rollout_gpu_cli_smoke_20260613_staggered_workers
```

关键 summary：

```text
resolved_device = cuda
cuda_available = True
cuda_device_name = NVIDIA A800 80GB PCIe
shared_rollout_enabled = True
shared_rollout_workers = 2
rollout_fragment_steps = 2
shared_rollout_batches = 2
shared_rollout_total_samples = 8
shared_rollout_worker_start_offsets = {"0": 0, "1": 2}
shared_rollout_policy_version_mismatch_count = 0
```

worker 时间轴证据：

```text
episode  worker_index
0        0               [0, 1]
         1               [2, 3]
1        0               [2, 3]
         1               [4, 5]
```

dispatch trace 也已包含行为诊断字段：

```text
worker_start_step
policy_normalized_aggregate_action
policy_normalized_der_action_mean
policy_normalized_der_action_std
```

结论：

1. CUDA 可用；
2. shared rollout 不再复制同一 worker 时间片；
3. policy version 一致；
4. 可以重新启动真正的 paper-long shared-rollout 主实验。

### 2026-06-13 01:08 CST 正式 staggered 主实验已重启

新主实验：

```text
tmux session = paper_long_shared_happo_staggered_20260613
pid = 972128
output_dir = outputs/paper_training_long_shared_rollout_happo_20260613_staggered_main
```

核心设置：

```text
algorithm = happo
seed = 9401
hparam_case = higher_entropy
train_episodes = 120
horizon_steps = 672
eval_horizon_steps = 672
shared_rollout_workers = 4
rollout_fragment_steps = 96
checkpoint_selection = final
baseline = skipped
```

启动健康状态：

```text
process status = Rl+
cpu = about 227%-239%
gpu process = ./.venv-server/bin/python
gpu memory for this training process = about 568 MiB
```

staggered worker 证据：

```text
2026-06-13T01:19:20 worker=0 worker_start_step=0  step=96  local_step=96/96
2026-06-13T01:21:41 worker=1 worker_start_step=96 step=120 local_step=24/96
```

解释：

1. worker 0 正在采样 `0-95`；
2. worker 1 已经从 `96` 起步，当前已采到 `119`，progress 中显示 `step=120`；
3. 这证明正式 long run 已经应用了错开时间片修复，不再是 4 个 worker 复制 step `0-95`；
4. 仍需等第一个 HAPPO update 完成后检查 `happo_training_summary.json`、`happo_step_metrics.csv`、`happo_dispatch_private_profit_trace.csv` 和 reward dynamic cards。
