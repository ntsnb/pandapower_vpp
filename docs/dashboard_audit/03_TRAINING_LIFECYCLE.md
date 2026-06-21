# 03 Training Lifecycle

## 结论

正式训练生命周期由 `examples/17_paper_training_experiment.py` 进入，实际总编排在 `run_paper_training_experiment()`。该编排包括 profile 生成、baseline rollout、安全 gate、trainable algorithm 训练、checkpoint 选择、frozen evaluation、CSV/HTML/TensorBoard artifact 输出。

概念对齐重点：

- `episode`：训练器内部一次 `env.reset()` 到 horizon truncation 的轨迹。
- `epoch`：在 HAPPO 中主要是 PPO update epoch；不是“完整遍历数据集一次”。
- `batch`：HAPPO/HATRPO 是 rollout tensor；MATD3/HASAC 是 replay sample minibatch。
- `horizon_steps`：单个 episode 的环境步数。
- `paper_long_sensitivity_v1`：672 steps × 0.25h = 7 天；未发现 5 周数据按 episode 切分的实现。

## 证据

- `examples/17_paper_training_experiment.py`：解析 CLI，构造 preset，检查 CUDA，调用 `run_paper_training_experiment(cfg)`。
- `src/vpp_dso_sim/experiments/paper_training.py`：`PaperTrainingExperimentConfig` 定义 algorithms、seeds、variants、horizon、train_episodes、checkpoint selection、TensorBoard/export flags。
- `src/vpp_dso_sim/experiments/paper_training.py`：`paper_long_sensitivity_v1` 使用 `horizon_steps=672`、`eval_horizon_steps=672`、`train_episodes=120`、`seeds=(9401..9405)`、`checkpoint_selection="both"`。
- `src/vpp_dso_sim/experiments/paper_training.py`：`_write_profile_config()` 将每个 run profile 物化为 `load_profile.csv`、`pv_profile.csv`、`price_profile.csv`。
- `src/vpp_dso_sim/experiments/paper_training.py`：`_run_baseline_rollout()` 直接创建 `Simulator` 并执行 baseline actions。
- `src/vpp_dso_sim/experiments/paper_training.py`：`_train_algorithm()` 按 `happo|hatrpo|matd3|hasac` 分派训练器。
- `src/vpp_dso_sim/experiments/paper_training.py`：`_checkpoint_choices()` 支持 `final`、`train_best`、`both`。
- `src/vpp_dso_sim/experiments/paper_training.py`：`_write_tensorboard_scalars()` 使用 `torch.utils.tensorboard.SummaryWriter`。
- `src/vpp_dso_sim/experiments/paper_training.py`：最终写 `run_index.csv`、`evaluation_seed_metrics.csv`、`training_episode_metrics.csv`、`training_loss_metrics.csv` 等。
- 当前 Anaconda Python 3.12.7 下 `python3 -m pytest --collect-only` 因缺 `pandapower` 中断；收集到 65 项后 43 个 import errors。

## 相关文件路径

- `examples/17_paper_training_experiment.py`
- `src/vpp_dso_sim/experiments/paper_training.py`
- `src/vpp_dso_sim/learning/advanced_marl.py`
- `src/vpp_dso_sim/learning/hatrpo.py`
- `src/vpp_dso_sim/learning/matd3.py`
- `src/vpp_dso_sim/learning/deep_rl.py`
- `src/vpp_dso_sim/learning/shared_rollout_workers.py`
- `src/vpp_dso_sim/simulation/simulator.py`
- `src/vpp_dso_sim/visualization/reward_dynamic_report.py`
- `tests/test_paper_training_experiment.py`

## 相关类/函数/变量

- `PaperTrainingExperimentConfig`
- `paper_training_preset`
- `run_paper_training_experiment`
- `_write_profile_config`
- `_profile_pack`
- `_run_baseline_rollout`
- `_train_algorithm`
- `_evaluate_algorithm_checkpoint`
- `_checkpoint_choices`
- `_load_completed_training`
- `_guard_output_protocol`
- `_print_progress`
- `_append_progress_event`
- `_write_tensorboard_scalars`
- `progress_callback`
- `train_episodes`
- `horizon_steps`
- `eval_horizon_steps`
- `checkpoint_selection`
- `resume_completed`

## Lifecycle Map

```text
examples/17_paper_training_experiment.py
  -> argparse
  -> paper_training_preset(name)
  -> dataclasses.replace(cfg, CLI overrides)
  -> _validate_trainable_cuda_requirement(cfg)
  -> run_paper_training_experiment(cfg)
       _guard_output_protocol(output_dir, cfg)
       _configure_local_plot_cache(output_dir)
       _print_progress(start)
       for seed:
         for train/eval profile variant:
           _write_profile_config(...)
           profile_quality rows
           baseline eval variants:
             _run_baseline_rollout(...)
       baseline safety gate
       for seed/train variant/algorithm/hparam case:
         _train_algorithm(...)
         checkpoint selection
         frozen evaluation on holdout variants
       aggregate metrics
       tensorboard scalars
       CSV/HTML/manifest output
```

## Config / Seed / Profile

- Config loader 支持 registry alias 与 `extends` deep merge。
- Paper runner 为每个 run 写私有 `scenario_config.yaml`。
- train profile 使用 `seed`；eval profile 使用 `seed + 10000`。
- `resume_completed` 默认 False；开启时复用已有 artifacts。
- `_guard_output_protocol()` 会保护 paper-long 输出目录，防止非空目录被静默污染。

## Episode / Epoch / Batch

| 名称 | 当前实际含义 | dashboard 建议命名 |
|---|---|---|
| `episode` | 一次 env reset 到 horizon 的轨迹，训练器 0-based 计数 | `episode_id` |
| `epoch` | HAPPO PPO update epoch；HATRPO value epoch 是 critic 拟合轮次 | `optimizer_epoch` 或 `ppo_epoch` |
| `batch` | On-policy rollout tensor 或 off-policy replay minibatch | `batch_id` + `batch_type` |
| `trajectory`/`rollout` | HAPPO/HATRPO 每 episode 收集的 step 序列；shared rollout 可能 fragment 化 | `rollout_id`/`fragment_id` |
| `gradient_step` | 未统一；MATD3/HASAC 有 update counters，HAPPO 有 role update rows | adapter 合成 `gradient_step` |
| `global_env_step` | 未统一；可由 episode/step 合成，但 shared rollout 要带 worker/env | `global_env_step` nullable/synthesized |

## Logging / Checkpoint / Evaluation

已发现日志与 artifact：

- `experiment_progress.jsonl`
- `experiment_progress.csv`
- `progress_summary.json`
- `training_episode_metrics.csv`
- `training_loss_metrics.csv`
- `evaluation_seed_metrics.csv`
- `aggregate_metrics.csv`
- `baseline_comparison.csv`
- `tensorboard_runs.csv`
- `tensorboard_assets.csv`
- algorithm-specific `*_episode_metrics.csv`
- algorithm-specific `*_update_metrics.csv`
- `happo_checkpoint.pt`、`happo_best_checkpoint.pt`
- `matd3_checkpoint.pt`、`matd3_best_checkpoint.pt`
- `hasac_checkpoint.pt`、`hasac_best_checkpoint.pt`
- HATRPO checkpoint

现有 dashboard/visualization 读 CSV，不是训练进程内的实时事件总线。

## 安全 Hook 点

推荐 hook 顺序：

1. `on_train_start`：`run_paper_training_experiment()` 开始后，输出目录和 config hash 已确定时。
2. `on_profile_materialized`：`_write_profile_config()` 后记录 profile metadata/hash。
3. `on_env_reset`：训练器 reset 后记录 episode/run/env context。
4. `on_env_step`：env.step 后记录 observations/actions/rewards/infos 和 simulator records。
5. `on_reward_computed`：从 `reward_components`/`agent_reward_components` 只读抽取。
6. `on_loss_computed`：每个算法 update row 形成后抽取。
7. `on_episode_end`：episode metrics row append 后抽取。
8. `on_eval_end`：frozen eval result 形成后抽取。
9. `on_checkpoint_saved`：checkpoint path 写入后抽取。
10. `on_train_error`：外层 try/except 捕获，logger 降级 warning。
11. `on_train_end`：manifest 与 summary 输出后关闭 writer。

## 风险

- High：把 dashboard 服务同步嵌入 `run_paper_training_experiment()` 会阻塞训练或与 GPU/多进程争用资源。
- High：`resume_completed` 复用 artifact 当前不是严格 config hash gate，可能让 dashboard 读到旧结果。
- Medium：`epoch` 概念容易误用，HAPPO PPO epoch 不等于用户理解的训练轮次。
- Medium：非 HAPPO 训练器中途 progress callback 不如 HAPPO 完整，实时粒度不均。
- Medium：shared rollout worker 需要 `worker_id/env_id`，否则 `time_index`/`global_env_step` 会冲突。

## 建议

- 第一阶段只接入 paper training 外围和 algorithm update outputs，不改训练数学。
- 统一 `run_id`：建议由 `preset/algorithm/seed/variant/hparam/checkpoint_label` 组成，再加短 hash。
- `epoch_id` 对 HAPPO 先置为 `ppo_epoch`；campaign 层另用 `iteration_id` 或留空，避免误导。
- `global_env_step` 由 adapter 合成并注明 `source=synthesized`。
- dashboard writer 必须异步/批量，训练进程异常隔离。

## 待用户确认项

- 是否接受 `epoch_id` 在 schema 中保留但很多算法置空，另用 `optimizer_epoch` 记录 PPO/value epoch。
- `episode` 是否正式定义为“一个 horizon trajectory”，而不是一天/一周/5 周。
- 是否需要为 MATD3/HASAC/HATRPO 补齐与 HAPPO 一致的 progress callback。
- `resume_completed` 是否必须加入 config/profile hash 强校验后才允许 dashboard 自动复用。
