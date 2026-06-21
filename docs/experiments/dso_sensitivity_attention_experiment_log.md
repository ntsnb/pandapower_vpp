# DSO 敏感度注意力实验与曲线台账

Updated: 2026-05-28 Asia/Shanghai

本文档专门记录 `sensitivity_attention_v1` 改造过程中的测试、smoke、短训练和后续
paper-long 前置实验。它用于回答：

- 实验是否真的跑过；
- 跑的是哪个配置、哪个 seed、哪个输出目录；
- reward 曲线、loss 曲线、KL、entropy、grad norm、projection gap、AC safety 文件在哪里；
- 哪些结果能说明接口健康，哪些不能说明算法收敛；
- 哪些失败是本轮新增，哪些是迁移前遗留。

## 1. 曲线/损失字段解释

| 字段 | 中文含义 | 用途 |
|---|---|---|
| `episode_reward` / `eval_total_reward` | 每个 episode 或评估总奖励 | 判断策略整体表现趋势，但不能单独判断安全性 |
| `dso_actor_loss` | DSO actor 的策略损失 | 检查 DSO guidance 策略是否发生有效更新 |
| `dispatch_actor_loss` | VPP dispatch actor 的策略损失 | 检查各 VPP 调度策略是否更新 |
| `portfolio_actor_loss` | 慢循环聚合/重组合 actor 损失 | 检查 DER 组合智能体是否参与学习 |
| `critic_loss` / `role_critic_loss` | critic/value 损失 | 判断价值函数是否爆炸或欠拟合 |
| `entropy` | 策略探索熵 | 熵过低可能说明探索被过早压死 |
| `kl` / `approx_kl` | 新旧策略分布差异 | PPO/HAPPO 稳定性约束，过高可能策略崩塌 |
| `grad_norm` | 梯度范数 | 检查梯度爆炸或完全没有更新 |
| `projection_gap_mw` | raw action 到安全投影后动作的差距 | 判断安全外壳是否频繁接管策略 |
| `ac_safe` / `ac_certificate_failed_count` | AC 潮流安全证书状态 | 判断最终动作是否满足潮流安全检查 |
| `preferred_width_mw` | DSO preferred range 宽度 | 检查 DSO 是否靠过宽 envelope 逃避决策 |

## 2. 实验记录模板

每次运行后追加如下记录：

```markdown
### YYYY-MM-DD HH:MM 实验名称

- 目的：
- 命令：
- 配置：
- seed：
- 输出目录：
- checkpoint：
- episode metrics：
- step metrics：
- update/loss metrics：
- reward 曲线：
- loss 曲线：
- projection gap 曲线：
- AC safety 曲线：
- 结果摘要：
- 是否出现 NaN/Inf：
- 是否需要中断或修复：
- 结论边界：
```

## 3. 当前运行记录

### 2026-05-28 改造开始前状态

- 目的：确认迁移后项目中 DSO envelope、HAPPO/HATRPO、FR/DOE、AC projection 的当前接入点。
- 命令：只读 `sed` / `rg` / `find` / `git status --short`。
- 输出目录：无。
- 曲线文件：无。
- 结果摘要：
  - 当前 rule-based envelope 仍由 `Simulator._build_dso_operating_envelope()` 产生。
  - 当前 HAPPO 的 DSO actor 仍是 legacy MLP/Gaussian actor。
  - 当前已有 AC-aware safety shell，但还不是 prompt 要求的 trainable structured DSO actor。
- 结论边界：这一步是只读定位，不证明训练效果。

### 2026-05-28 TDD 红灯：新增 DSO v1 模块测试

- 目的：先用测试固定 `sensitivity_attention_v1` 需要的 schema、selector、sensitivity tensor、structured observation、attention actor、safe decoder 和 policy switch。
- 命令：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_action_units.py \
  tests/test_network_objects.py \
  tests/test_sensitivity_shapes.py \
  tests/test_structured_observation_shapes.py \
  tests/test_bipartite_attention_actor.py \
  tests/test_safe_decoder.py \
  tests/test_legacy_baseline_unchanged.py \
  tests/test_envelope_policy_switch.py \
  tests/test_privacy_no_private_cost_leak.py
```

- 配置：无训练配置，pytest 单元测试。
- 输出目录：无。
- 结果摘要：collection 阶段 9 个错误，均为 `ModuleNotFoundError: No module named 'vpp_dso_sim.dso'`。
- 是否出现 NaN/Inf：未进入数值训练。
- 是否需要中断或修复：需要继续实现新包。
- 结论边界：这是 TDD 预期红灯，证明测试当前确实约束了尚未实现的新模块。

### 2026-05-28 TDD 绿灯：DSO v1 基础模块测试

- 目的：验证新增 `vpp_dso_sim.dso` versioned 包的基础能力。
- 命令：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_action_units.py \
  tests/test_network_objects.py \
  tests/test_sensitivity_shapes.py \
  tests/test_sensitivity_finite_difference.py \
  tests/test_structured_observation_shapes.py \
  tests/test_bipartite_attention_actor.py \
  tests/test_safe_decoder.py \
  tests/test_legacy_baseline_unchanged.py \
  tests/test_envelope_policy_switch.py \
  tests/test_privacy_no_private_cost_leak.py
```

- 配置：pytest 单元测试，使用 `configs/european_lv_benchmark_v2.yaml` 构造小规模 power-flow 场景。
- seed：未进入训练，场景默认 seed。
- 输出目录：无。
- checkpoint：无。
- episode metrics：无。
- step metrics：无。
- update/loss metrics：无。
- reward 曲线：无。
- loss 曲线：无。
- projection gap 曲线：无。
- AC safety 曲线：无。
- 结果摘要：14 个测试通过，1 个 jupyter path deprecation warning。
- 是否出现 NaN/Inf：attention actor forward 测试中输出 finite；无训练 NaN/Inf。
- 是否需要中断或修复：不需要中断，但还未接入 simulator routing / smoke / short training。
- 结论边界：通过的是结构、shape、mask、privacy schema 和有限差分 sanity；不能说明算法收敛。

### 2026-05-28 Simulator routing 绿灯

- 目的：验证 `scenario.config["dso"]["envelope_policy"] = "sensitivity_attention_v1"` 时，
  `Simulator.step()` 不再绕过 policy switch，而是记录结构化 DSO envelope。
- 命令：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_action_units.py \
  tests/test_network_objects.py \
  tests/test_sensitivity_shapes.py \
  tests/test_sensitivity_finite_difference.py \
  tests/test_structured_observation_shapes.py \
  tests/test_bipartite_attention_actor.py \
  tests/test_safe_decoder.py \
  tests/test_legacy_baseline_unchanged.py \
  tests/test_envelope_policy_switch.py \
  tests/test_privacy_no_private_cost_leak.py
```

- 配置：pytest 单元/集成轻量测试。
- 输出目录：无。
- 结果摘要：16 个测试通过，1 个 jupyter path deprecation warning。
- 曲线/损失：无训练曲线。
- 关键验证：
  - `rule_v0` adapter 保持旧 envelope 数值。
  - `sensitivity_attention_v1` policy 能构造 decoded guidance envelope。
  - `Simulator.step()` 能从 config 路由到 `sensitivity_attention_v1`。
  - 记录包含 `action_units`、`selected_network_objects`、`active_sensitivity_edges_shape`、
    `dso_actor_raw_outputs` 和 `decoded_operating_envelope`。
- 结论边界：这证明 simulator routing 跑通，但当前 actor 仍是 deterministic untrained initialization。

### 2026-05-28 脚本级 smoke：rule_v0 baseline

- 目的：确认旧规则 baseline 仍可通过新增 smoke 脚本运行并写出指标。
- 命令：

```bash
./.venv-server/bin/python scripts/run_smoke.py \
  --config configs/baseline_rule_v0.yaml \
  --seed 0 \
  --steps 2 \
  --output-dir outputs/dso_sensitivity_attention/baseline_rule_v0_smoke_seed0
```

- 配置：`configs/baseline_rule_v0.yaml`
- seed：0
- 输出目录：`outputs/dso_sensitivity_attention/baseline_rule_v0_smoke_seed0`
- step metrics：`smoke_step_metrics.csv`
- envelope：`dso_operating_envelope.csv`
- summary：`smoke_summary.json`
- 结果摘要：
  - `envelope_policy = rule_v0`
  - `steps = 2`
  - `dso_operating_envelope` 记录数：2
  - `constraint_violations` 记录数：0
  - `nan_or_inf_detected = False`
- 结论边界：baseline 脚本路径可运行；不涉及学习。

### 2026-05-28 脚本级 smoke：sensitivity_attention_v1

- 目的：确认结构化 DSO policy 可通过 simulator 正常 rollout 并写出 action unit / network object / sensitivity 相关 envelope 字段。
- 命令：

```bash
./.venv-server/bin/python scripts/run_smoke.py \
  --config configs/happo_sensitivity_attention_v1.yaml \
  --seed 0 \
  --steps 2 \
  --output-dir outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0
```

- 配置：`configs/happo_sensitivity_attention_v1.yaml`
- config hash：`e42bb936edcc`
- seed：0
- 输出目录：`outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0`
- step metrics：`smoke_step_metrics.csv`
- envelope：`dso_operating_envelope.csv`
- summary：`smoke_summary.json`
- 结果摘要：
  - `envelope_policy = sensitivity_attention_v1`
  - `steps = 2`
  - `dso_operating_envelope` 记录数：2
  - `constraint_violations` 记录数：0
  - `nan_or_inf_detected = False`
- 结论边界：结构化 policy 可 rollout；actor 仍是未经过正式 HAPPO 训练的初始化网络。

### 2026-05-28 Sensitivity cache refresh / priority refresh

- 目的：补齐 prompt 3.4 要求的 cache 更新策略，避免 `raw_sensitivity_cache` 只有 TTL 复用而没有状态变化触发和优先 ActionUnit。
- 关键改造：
  - 新增 `decide_sensitivity_refresh()`，覆盖 `update_period_steps`、`cache_ttl_steps`、电压漂移、loading 漂移、FR 宽度变化、projection-gap history、缺失 ActionUnit、缺失 NetworkObject。
  - 新增 `merge_sensitivity_update()`，当 raw cache 已覆盖当前 ActionUnit/NetworkObject 时，只对 priority ActionUnit 做有限差分刷新，并合并回 raw cache。
  - finite-difference tensor metadata 记录 `sensitivity_allocation_mode` 和 `sensitivity_allocation_weights`；runtime `dso_operating_envelope.csv` 同步输出这些字段。
  - runtime `dso_operating_envelope.csv` 新增 `sensitivity_refresh_reasons`、`sensitivity_priority_action_units`、`sensitivity_partial_priority_refresh`、`sensitivity_partial_refresh_action_unit_ids`、`sensitivity_update_period_steps`、`sensitivity_cache_ttl_steps`。
  - 主配置和 ablation 配置补齐 `refresh_if_voltage_delta_pu_gt`、`refresh_if_loading_delta_pct_gt`、`refresh_if_fr_width_change_ratio_gt`、`refresh_if_projection_gap_hist_gt_mw`、`projection_gap_history_lookback_steps`。
- 单测命令：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_structured_happo_training.py \
  tests/test_envelope_policy_switch.py \
  tests/test_structured_smoke_rollout.py \
  tests/test_training_step_no_nan.py \
  tests/test_reward_rebalance.py \
  tests/test_multi_agent_env.py \
  tests/test_sensitivity_shapes.py \
  tests/test_sensitivity_finite_difference.py \
  tests/test_privacy_no_private_cost_leak.py \
  tests/test_paper_training_experiment.py::test_happo_checkpoint_frozen_eval_runs
```

- 单测结果：`34 passed`，1 个 jupyter path deprecation warning。
- 完整仓库验证：`./.venv-server/bin/python -m pytest -q`，exit code 0，1 个 jupyter path deprecation warning。
- smoke 命令：

```bash
bash scripts/agent_hooks/smoke_training.sh
```

- smoke 结果：
  - `configs/baseline_rule_v0.yaml`：2 step，`constraint_violations = 0`，`nan_or_inf_detected = False`。
  - `configs/happo_sensitivity_attention_v1.yaml`：2 step，`constraint_violations = 0`，`nan_or_inf_detected = False`，`config_hash = e42bb936edcc`。
  - 256-step BC short train：`initial_loss = 0.5577945709228516`，`final_loss = 0.00032648438354954123`，`nan_or_inf_detected = False`。
- 当前归档文件：
  - `docs/experiments/dso_sensitivity_attention_artifacts/current_dso_operating_envelope.csv`
  - `docs/experiments/dso_sensitivity_attention_artifacts/current_smoke_summary.json`
  - `docs/experiments/dso_sensitivity_attention_artifacts/current_bc_loss_metrics.csv`
- 结论边界：该实验验证 cache refresh 和 priority refresh 的工程链路；不证明 paper-long 收敛，也不证明刷新策略已经达到最优计算效率。

### 2026-05-28 Legacy MLP / flat observation direct smoke

- 目的：为 legacy DSO MLP / legacy flat observation baseline 增加直接 smoke artifact，避免只依赖间接 trainer 测试。
- 命令：

```bash
./.venv-server/bin/python scripts/run_smoke.py \
  --config configs/happo_legacy_mlp.yaml \
  --seed 0 \
  --steps 2 \
  --output-dir outputs/dso_sensitivity_attention/happo_legacy_mlp_smoke_seed0
```

- 配置：`configs/happo_legacy_mlp.yaml`
- config hash：`b5433e8f06f5`
- seed：0
- 输出目录：`outputs/dso_sensitivity_attention/happo_legacy_mlp_smoke_seed0`
- 结果摘要：
  - `envelope_policy = rule_v0`
  - `observation_mode = legacy_flat`
  - `steps = 2`
  - `dso_operating_envelope` 记录数：2
  - `constraint_violations` 记录数：0
  - `reward_components` 记录数：2
  - `nan_or_inf_detected = False`
- 归档：
  - `docs/experiments/dso_sensitivity_attention_artifacts/current_happo_legacy_mlp_smoke_summary.json`
  - `docs/experiments/dso_sensitivity_attention_artifacts/current_happo_legacy_mlp_smoke_step_metrics.csv`
  - `docs/experiments/dso_sensitivity_attention_artifacts/current_happo_legacy_mlp_dso_operating_envelope.csv`
- 结论边界：这是 legacy config 直接运行证据；不代表 legacy MLP 优于新 actor。

### 2026-05-28 短训练 sanity：BC warm-start loss

- 目的：验证 `BipartiteSensitivityDSOActor` 至少可以被优化器更新，且短训练过程没有 NaN/Inf，并写出 loss 曲线。
- 命令：

```bash
./.venv-server/bin/python scripts/run_short_train.py \
  --config configs/happo_sensitivity_attention_v1.yaml \
  --seed 0 \
  --steps 256 \
  --output-dir outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0
```

- 配置：`configs/happo_sensitivity_attention_v1.yaml`
- config hash：`dc87cf068567`
- seed：0
- 输出目录：`outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0`
- checkpoint：`dso_sensitivity_attention_actor.pt`
- loss metrics：`dso_sensitivity_attention_short_train_loss_metrics.csv`
- decoded envelope：`decoded_operating_envelope.csv`
- summary：`short_train_summary.json`
- 结果摘要：
  - `steps_completed = 256`
  - `nan_or_inf_detected = False`
  - `initial_loss = 0.5577945709228516`
  - `final_loss = 0.00032648438354954123`
  - `bc_loss` 从 0.5578 下降到 0.000326。
  - `grad_norm` 从 6.5764 下降到 0.00342。
- 结论边界：这是 rule target 的 behavior cloning warm-start sanity，不是 HAPPO 真实在线收敛结论。

### 2026-05-28 测试集合回归

- 新增结构化测试：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_action_units.py \
  tests/test_network_objects.py \
  tests/test_sensitivity_shapes.py \
  tests/test_sensitivity_finite_difference.py \
  tests/test_structured_observation_shapes.py \
  tests/test_bipartite_attention_actor.py \
  tests/test_safe_decoder.py \
  tests/test_legacy_baseline_unchanged.py \
  tests/test_envelope_policy_switch.py \
  tests/test_privacy_no_private_cost_leak.py \
  tests/test_structured_smoke_rollout.py \
  tests/test_training_step_no_nan.py
```

结果：初次 18 passed；补充 active sensitivity raw-cache slice 测试后复跑为 19 passed，
1 个 jupyter path deprecation warning。

- 旧 baseline / FR / multi-agent smoke：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_grid_aware_envelope.py \
  tests/test_feasibility_region.py \
  tests/test_timeseries_smoke.py \
  tests/test_multi_agent_env.py
```

结果：15 passed，1 个 jupyter path deprecation warning。

- HAPPO/HATRPO trainer 轻量回归：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_hasac_happo.py \
  tests/test_hatrpo_training.py \
  tests/test_happo_hasac_trainers.py
```

结果：11 passed，1 个 jupyter path deprecation warning。

- 完整项目测试：

```bash
./.venv-server/bin/python -m pytest -q
```

结果：exit code 0，完整测试集合通过；仍只有同一个 jupyter path deprecation warning。

### 2026-05-28 当前代码重跑：结构化 smoke artifact 保留

- 目的：按当前代码重新跑 `sensitivity_attention_v1` 2-step smoke，确认报告中列出的结构化 artifact 真实存在。
- 命令：

```bash
./.venv-server/bin/python scripts/run_smoke.py \
  --config configs/happo_sensitivity_attention_v1.yaml \
  --seed 0 \
  --steps 2 \
  --output-dir outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0_current
```

- 配置：`configs/happo_sensitivity_attention_v1.yaml`
- config hash：`dc87cf068567`
- seed：0
- 输出目录：`outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0_current`
- step metrics：`smoke_step_metrics.csv`
- envelope：`dso_operating_envelope.csv`
- action units：`action_units.csv`
- selected network objects：`selected_network_objects.csv`
- sensitivity edge summary：`sensitivity_edges.csv`
- raw actor outputs：`dso_actor_outputs.csv`
- decoded envelope：`decoded_operating_envelope.csv`
- summary：`smoke_summary.json`
- 结果摘要：
  - `envelope_policy = sensitivity_attention_v1`
  - `steps = 2`
  - `dso_operating_envelope` 记录数：2
  - `projection_trace` 记录数：24
  - `constraint_violations` 记录数：0
  - `nan_or_inf_detected = False`
- reward 曲线：`smoke_step_metrics.csv` 中的 `total_reward`。
- projection gap 曲线：本 smoke summary 记录了 `projection_trace` 记录数；完整曲线需要后续把 `projection_trace` 单独导出为 CSV。
- AC safety 曲线：当前通过 `constraint_violations = 0` 间接记录；完整 `ac_safe` 序列仍需后续补充导出。
- 结论边界：这证明当前结构化 envelope 运行和解释 artifact 可保存；不证明策略已经学习收敛。

### 2026-05-28 当前代码重跑：BC warm-start loss 曲线

- 目的：按当前代码重新跑短训练，保留当前版本下的 behavior-cloning loss 曲线和 checkpoint。
- 命令：

```bash
./.venv-server/bin/python scripts/run_short_train.py \
  --config configs/happo_sensitivity_attention_v1.yaml \
  --seed 0 \
  --steps 256 \
  --output-dir outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0
```

- 配置：`configs/happo_sensitivity_attention_v1.yaml`
- config hash：`e42bb936edcc`
- seed：0
- 输出目录：`outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0`
- checkpoint：`dso_sensitivity_attention_actor.pt`
- loss metrics：`dso_sensitivity_attention_short_train_loss_metrics.csv`
- decoded envelope：`decoded_operating_envelope.csv`
- summary：`short_train_summary.json`
- loss 曲线字段：
  - `bc_loss`：总 imitation loss。
  - `center_loss`：preferred center ratio 的 MSE。
  - `width_loss`：preferred width ratio 的 MSE。
  - `direction_loss`：吸收/平衡/注入方向分类损失。
  - `grad_norm`：actor 参数梯度范数。
- 结果摘要：
  - `steps_completed = 256`
  - `nan_or_inf_detected = False`
  - `bc_loss: 0.5577945709228516 -> 0.00032648438354954123`
  - `center_loss: 0.08863694220781326 -> 0.000000036100825440144035`
  - `width_loss: 0.16337132453918457 -> 0.000000004120818797304083`
  - `direction_loss: 1.2231453657150269 -> 0.0013057766482234001`
  - `grad_norm: 6.576420783996582 -> 0.0034202593378722668`
- 结论边界：这只能说明结构化 actor 能拟合 rule target，不能说明 HAPPO paper-long 收敛。

### 2026-05-28 当前代码重跑：结构化 HAPPO 最小训练

- 目的：确认 `train_happo()` 已经能使用结构化 DSO actor，并把 policy loss、entropy、approx KL、grad norm 写入 update metrics。此记录已在 VPP preferred bonus gating 修改后重跑。
- 命令：

```bash
./.venv-server/bin/python -c "from vpp_dso_sim.learning.advanced_marl import HAPPOConfig, train_happo; train_happo(config_path='configs/happo_sensitivity_attention_v1.yaml', output_dir='outputs/dso_sensitivity_attention/happo_structured_minimal_seed0_current', config=HAPPOConfig(horizon_steps=2, episodes=1, hidden_dim=32, ppo_epochs=1, seed=0, critic_use_action_summary=True))"
```

- 配置：`configs/happo_sensitivity_attention_v1.yaml`
- seed：0
- 输出目录：`outputs/dso_sensitivity_attention/happo_structured_minimal_seed0_current`
- checkpoint：
  - `happo_checkpoint.pt`
  - `happo_best_checkpoint.pt`
- episode metrics：`happo_episode_metrics.csv`
- step metrics：`happo_step_metrics.csv`
- update/loss metrics：`happo_update_metrics.csv`
- summary：
  - `happo_training_summary.json`
  - `happo_training_summary.csv`
- reward 曲线：`happo_episode_metrics.csv` 中的 `episode_reward`。
- loss 曲线：
  - `happo_episode_metrics.csv` 中的 `critic_loss`。
  - `happo_update_metrics.csv` 中按 `role` 分组的 `policy_loss`。
- KL/entropy 曲线：`happo_update_metrics.csv` 中的 `approx_kl` 和 `entropy_mean`。
- projection gap 曲线：`happo_episode_metrics.csv` 中的 `projection_gap_mw`、`local_bounds_projection_gap_mw`、`ac_aware_projection_gap_mw`。
- 结果摘要：
  - `algorithm = happo_sequential_ctde`
  - `dso_actor_observation_mode = structured_bipartite`
  - `dso_actor_type = sensitivity_attention_v1_structured_happo`
  - `structured_dso_actor_trainable = True`
  - `dso_input_dim = 1210`
  - `episode_reward = 5.67563214334187`
  - `episode_cost = 1.7059830742162847`
  - `violation_count = 0`
  - `projection_gap_mw = 0.0`
  - `critic_loss = 0.07204630225896835`
  - `param_delta_l2 = 0.0509931817650795`
  - DSO update：`policy_loss = -0.01804180070757866`, `entropy_mean = 0.7192385196685791`, `approx_kl = -0.0005803108215332031`, `grad_norm = 0.05255132168531418`
  - dispatch update：`policy_loss = -0.03949027508497238`, `entropy_mean = 2.4769539833068848`, `approx_kl = -0.03622031211853027`, `grad_norm = 0.3717022240161896`
  - portfolio update：`policy_loss = -0.4796231687068939`, `entropy_mean = 1.0781656503677368`, `approx_kl = -0.027808785438537598`, `grad_norm = 1.0236605405807495`
- 是否出现 NaN/Inf：未发现。
- 是否需要中断或修复：不需要中断当前最小训练；runtime envelope policy checkpoint loading、HAPPO `target_kl`、`nan_guard` 和 deterministic observation normalization 均已在后续记录补齐。
- 结论边界：这是 1 episode / 2 step 的链路验证，不是收敛实验。

### 2026-05-28 当前代码修改：VPP preferred-range bonus gating

- 目的：落实 prompt Phase 9：VPP preferred-range bonus 不能只因为落入推荐区间就给奖励，必须同时受到 DSO 引导强度、推荐区间宽度和响应有效性约束。
- 修改文件：
  - `src/vpp_dso_sim/envs/reward_design.py`
  - `tests/test_reward_rebalance.py`
- 红灯命令：

```bash
./.venv-server/bin/python -m pytest -q tests/test_reward_rebalance.py::test_vpp_preferred_range_bonus_is_gated_by_lambda_width_and_effectiveness
```

- 红灯结果：失败，`KeyError: 'preferred_inside_range'`，证明旧实现没有 gate 日志和 gated bonus。
- 绿灯命令：

```bash
./.venv-server/bin/python -m pytest -q tests/test_reward_rebalance.py::test_vpp_preferred_range_bonus_is_gated_by_lambda_width_and_effectiveness
```

- 绿灯结果：1 passed。
- 回归命令：

```bash
./.venv-server/bin/python -m pytest -q tests/test_reward_rebalance.py tests/test_multi_agent_env.py
```

- 回归结果：9 passed。
- 新增 reward 组件字段：
  - `preferred_inside_range`
  - `preferred_bonus_lambda_gate`
  - `preferred_bonus_width_gate`
  - `preferred_bonus_effectiveness_gate`
  - `preferred_region_score`
- 计算公式：

```text
preferred_region_bonus =
  DISPATCH_PREFERRED_REGION_BONUS_WEIGHT
  * preferred_inside_range
  * preferred_bonus_lambda_gate
  * preferred_bonus_width_gate
  * preferred_bonus_effectiveness_gate
```

- 结论边界：该修改修正 VPP dispatch 的 shaped reward，不代表 DSO policy 已经学会更优 envelope。

### 2026-05-28 当前代码修改：structured HAPPO frozen eval

- 目的：让 `sensitivity_attention_v1_structured_happo` checkpoint 能用 deterministic frozen eval 复现，避免 paper-long 只能看训练 rollout。
- 修改文件：
  - `src/vpp_dso_sim/learning/advanced_marl.py`
  - `tests/test_structured_happo_training.py`
- 红灯命令：

```bash
./.venv-server/bin/python -m pytest -q tests/test_structured_happo_training.py::test_structured_happo_checkpoint_frozen_eval_runs
```

- 红灯结果：失败，`evaluate_happo_checkpoint()` 按 legacy DSO actor 重建模型，加载 structured actor state dict 时出现 missing/unexpected keys。
- 绿灯命令：

```bash
./.venv-server/bin/python -m pytest -q tests/test_structured_happo_training.py::test_structured_happo_checkpoint_frozen_eval_runs
```

- 绿灯结果：1 passed。
- legacy 回归命令：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_structured_happo_training.py \
  tests/test_paper_training_experiment.py::test_happo_checkpoint_frozen_eval_runs
```

- legacy 回归结果：3 passed。
- 当前 frozen eval 命令：

```bash
./.venv-server/bin/python -c "from vpp_dso_sim.learning.advanced_marl import evaluate_happo_checkpoint; evaluate_happo_checkpoint(config_path='configs/happo_sensitivity_attention_v1.yaml', checkpoint_path='outputs/dso_sensitivity_attention/happo_structured_minimal_seed0_current/happo_best_checkpoint.pt', output_dir='outputs/dso_sensitivity_attention/happo_structured_frozen_eval_seed1_current', horizon_steps=2, seed=1)"
```

- 输出目录：`outputs/dso_sensitivity_attention/happo_structured_frozen_eval_seed1_current`
- summary：`happo_frozen_eval_summary.json`
- step metrics：`happo_frozen_eval_step_metrics.csv`
- 结果摘要：
  - `evaluation_mode = frozen_mean_argmax_actor`
  - `dso_actor_observation_mode = structured_bipartite`
  - `dso_actor_type = sensitivity_attention_v1_structured_happo`
  - `structured_dso_actor_loaded = True`
  - `total_reward = 5.655994539527365`
  - `total_cost = 1.6341365382859143`
  - `total_violation_count = 0`
- 结论边界：这是 2-step frozen eval，证明 checkpoint 加载和 deterministic rollout 路径可用，不证明长期泛化。

### 2026-05-28 当前代码修改：sensitivity raw cache 与 active slice

- 目的：落实 prompt Phase 4，避免 `sensitivity_attention_v1` policy 在可复用的情况下每次都重算有限差分敏感度。
- 修改文件：
  - `src/vpp_dso_sim/dso/envelope/sensitivity_attention_v1.py`
  - `tests/test_envelope_policy_switch.py`
- 红灯命令：

```bash
./.venv-server/bin/python -m pytest -q tests/test_envelope_policy_switch.py::test_sensitivity_attention_policy_reuses_raw_sensitivity_cache_for_active_slice
```

- 红灯结果：失败，`KeyError: 'sensitivity_cache_hit'`，证明旧 policy 没有 cache 使用证据。
- 绿灯命令：

```bash
./.venv-server/bin/python -m pytest -q tests/test_envelope_policy_switch.py::test_sensitivity_attention_policy_reuses_raw_sensitivity_cache_for_active_slice
```

- 绿灯结果：1 passed。
- 回归命令：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_sensitivity_shapes.py \
  tests/test_sensitivity_finite_difference.py \
  tests/test_envelope_policy_switch.py \
  tests/test_structured_smoke_rollout.py
```

- 回归结果：9 passed。
- 新增 rollout 字段：
  - `sensitivity_cache_hit`
  - `sensitivity_source`
  - `sensitivity_cache_step`
- current smoke 重跑命令：

```bash
./.venv-server/bin/python scripts/run_smoke.py \
  --config configs/happo_sensitivity_attention_v1.yaml \
  --seed 0 \
  --steps 2 \
  --output-dir outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0_current
```

- current smoke 结果：2 steps，无 NaN/Inf，`constraint_violations = 0`。
- 结论边界：cache hit 行为已经由单测固定；2-step smoke 中 critical object 集合变化时仍会合理 cache miss 并重算，不能把每次 miss 解释为失败。

### 2026-05-28 当前代码修改：runtime envelope policy 加载 structured HAPPO actor checkpoint

- 目的：闭合训练 actor 到在线 DSO envelope policy 的链路，避免 `SensitivityAttentionEnvelopePolicy` 永远使用随机初始化 actor。
- 修改文件：
  - `src/vpp_dso_sim/dso/envelope/sensitivity_attention_v1.py`
  - `tests/test_envelope_policy_switch.py`
- 支持配置：
  - `dso.actor.checkpoint_path`
  - `dso.actor_checkpoint_path`
- 支持 checkpoint key 形式：
  - direct `BipartiteSensitivityDSOActor.state_dict()`
  - `attention_actor.*`
  - `dso_actor.attention_actor.*`
- 红灯命令：

```bash
./.venv-server/bin/python -m pytest -q tests/test_envelope_policy_switch.py::test_sensitivity_attention_policy_loads_structured_happo_checkpoint_actor
```

- 红灯结果：失败，`KeyError: 'dso_actor_checkpoint_loaded'`。
- 绿灯命令：

```bash
./.venv-server/bin/python -m pytest -q tests/test_envelope_policy_switch.py::test_sensitivity_attention_policy_loads_structured_happo_checkpoint_actor
```

- 绿灯结果：1 passed。
- 回归命令：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_envelope_policy_switch.py \
  tests/test_structured_smoke_rollout.py \
  tests/test_structured_happo_training.py
```

- 回归结果：9 passed。
- 新增 rollout 字段：
  - `dso_actor_checkpoint_loaded`
  - `dso_actor_checkpoint_path`
  - `dso_actor_checkpoint_source`
- 结论边界：这证明 runtime policy 可加载 structured HAPPO attention actor 权重；不代表该 checkpoint 已在 paper-long 中收敛。

### 2026-05-28 当前代码回归：HAPPO 日志字段测试

- 目的：防止以后 HAPPO update metrics 又缺失 `entropy_mean` 和 `approx_kl`。
- 命令：

```bash
./.venv-server/bin/python -m pytest -q tests/test_structured_happo_training.py
```

- 结果：1 passed，1 个 jupyter path deprecation warning。
- 验证内容：
  - 结构化 DSO actor 被 `train_happo()` 使用。
  - `happo_update_metrics.csv` 中包含 `policy_loss`、`entropy_mean`、`approx_kl`。
  - 这些字段均非空。

### 2026-05-28 当前代码回归：smoke/short/HAPPO 组合测试

- 目的：把结构化 rollout、短训练无 NaN、HAPPO 日志字段放在同一条回归命令中验证。
- 命令：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_structured_smoke_rollout.py \
  tests/test_training_step_no_nan.py \
  tests/test_structured_happo_training.py
```

- 结果：3 passed，1 个 jupyter path deprecation warning。
- 结论边界：当前小规模链路测试通过；还没有运行 paper-long，因此不能从这条测试推断长期收敛。

### 2026-05-28 小型曲线数据归档

- 目的：`outputs/` 已被 `.gitignore` 排除，为了让本轮报告的关键曲线数据能随文档一起保留，将小型 CSV/JSON 复制到 `docs/experiments/dso_sensitivity_attention_artifacts/`。
- 归档目录：`docs/experiments/dso_sensitivity_attention_artifacts/`
- 归档文件：
  - `current_smoke_summary.json`
  - `current_smoke_step_metrics.csv`
  - `current_dso_operating_envelope.csv`
  - `current_dso_actor_outputs.csv`
  - `current_decoded_operating_envelope.csv`
  - `current_bc_loss_metrics.csv`
  - `current_bc_short_train_summary.json`
  - `current_happo_episode_metrics.csv`
  - `current_happo_update_metrics.csv`
  - `current_happo_frozen_eval_summary.json`
  - `current_happo_frozen_eval_step_metrics.csv`
- 索引说明：`docs/experiments/dso_sensitivity_attention_artifacts/README.md`
- 结论边界：这些归档文件只保存当前小规模 sanity 的证据；paper-long 的大规模输出仍应放在 `outputs/`，再挑选关键 summary/CSV 复制进文档 artifact 目录。

### 2026-05-28 当前代码回归：本轮受影响测试集合

- 目的：验证本轮 VPP reward gating、structured frozen eval、sensitivity cache 接入没有破坏相关训练/评估/环境路径。
- 命令：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_reward_rebalance.py \
  tests/test_multi_agent_env.py \
  tests/test_sensitivity_shapes.py \
  tests/test_sensitivity_finite_difference.py \
  tests/test_envelope_policy_switch.py \
  tests/test_structured_smoke_rollout.py \
  tests/test_training_step_no_nan.py \
  tests/test_structured_happo_training.py \
  tests/test_paper_training_experiment.py::test_happo_checkpoint_frozen_eval_runs
```

- 结果：22 passed，1 个 jupyter path deprecation warning。
- 结论边界：该集合覆盖本轮改动的直接行为和 legacy HAPPO eval 回归；它不是完整项目 pytest，也不是 paper-long 收敛验证。

### 2026-05-28 当前代码修改：HAPPO stability 与 frozen eval normalization

- 目的：把 `target_kl`、observation/advantage normalization、`nan_guard` 和 frozen eval 的配置一致性纳入报告级证据。
- 修改文件：
  - `src/vpp_dso_sim/learning/advanced_marl.py`
  - `tests/test_structured_happo_training.py`
- 红灯命令：

```bash
./.venv-server/bin/python -m pytest -q tests/test_structured_happo_training.py::test_structured_happo_checkpoint_frozen_eval_runs
```

- 红灯结果：失败，`KeyError: 'normalize_observations'`；随后一次补丁定位失败又触发 `NameError: name 'normalize_observations' is not defined`，确认 evaluator 内部没有正确定义和输出该配置。
- 绿灯命令：

```bash
./.venv-server/bin/python -m pytest -q tests/test_structured_happo_training.py::test_structured_happo_checkpoint_frozen_eval_runs
```

- 绿灯结果：1 passed，1 个 jupyter path deprecation warning。
- 当前持久化最小 HAPPO 训练命令：

```bash
./.venv-server/bin/python -c 'from pathlib import Path; from vpp_dso_sim.learning.advanced_marl import HAPPOConfig, evaluate_happo_checkpoint, train_happo; train = train_happo(config_path="configs/happo_sensitivity_attention_v1.yaml", output_dir=Path("outputs/dso_sensitivity_attention/happo_structured_minimal_seed0_current"), config=HAPPOConfig(horizon_steps=2, episodes=1, hidden_dim=32, ppo_epochs=2, seed=0, critic_use_action_summary=True, target_kl=0.02, normalize_observations=True, normalize_advantages=True, nan_guard=True)); ev = evaluate_happo_checkpoint(config_path="configs/happo_sensitivity_attention_v1.yaml", checkpoint_path=train["checkpoint"], output_dir=Path("outputs/dso_sensitivity_attention/happo_structured_frozen_eval_seed1_current"), horizon_steps=2, seed=1); print({"train_reward": train["summary"]["final_episode_reward"], "target_kl": train["summary"]["target_kl"], "eval_reward": ev["summary"]["total_reward"], "eval_normalize_observations": ev["summary"]["normalize_observations"]})'
```

- 输出摘要：

```text
train_reward = 5.67563214334187
target_kl = 0.02
eval_reward = 5.655994539527365
eval_normalize_observations = True
```

- HAPPO training summary：
  - `normalize_observations = true`
  - `normalize_advantages = true`
  - `nan_guard = true`
  - `nan_guard_trigger_count = 0`
  - `kl_early_stop_count = 1`
  - `observation_normalization_stats.dso_obs_std_mean = 18.3441801071167`
- update metrics：
  - `target_kl`
  - `target_kl_exceeded`
  - `nan_guard_triggered`
  - `policy_loss`
  - `entropy_mean`
  - `approx_kl`
  - `grad_norm`
- 归档文件：
  - `docs/experiments/dso_sensitivity_attention_artifacts/current_happo_training_summary.json`
  - `docs/experiments/dso_sensitivity_attention_artifacts/current_happo_update_metrics.csv`
  - `docs/experiments/dso_sensitivity_attention_artifacts/current_happo_frozen_eval_summary.json`
- 中文总报告：

```text
docs/experiments/dso_sensitivity_attention_change_experiment_report.md
```

- 结论边界：该记录说明最小训练/评估链路和稳定性日志已闭合；仍不能作为 paper-long 收敛结论。

### 2026-05-28 当前代码回归：报告生成后的受影响集合

- 目的：在中文总报告和 artifact 归档完成后，重新验证本轮直接影响的训练、评估、reward、sensitivity、smoke 与 legacy HAPPO frozen eval 路径。
- 命令：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_structured_happo_training.py \
  tests/test_envelope_policy_switch.py \
  tests/test_structured_smoke_rollout.py \
  tests/test_training_step_no_nan.py \
  tests/test_reward_rebalance.py \
  tests/test_multi_agent_env.py \
  tests/test_sensitivity_shapes.py \
  tests/test_sensitivity_finite_difference.py \
  tests/test_paper_training_experiment.py::test_happo_checkpoint_frozen_eval_runs
```

- 结果：24 passed，1 个 jupyter path deprecation warning。
- 结论边界：这是当时的本轮直接改动集合回归；最新回归已扩展为 28 passed，见下一节。

### 2026-05-28 当前代码修改：YAML trainer、residual schedule 与 flat privacy audit

- 目的：补齐子代理审计指出的证据弱项。
- 修改文件：
  - `src/vpp_dso_sim/learning/advanced_marl.py`
  - `src/vpp_dso_sim/dso/envelope/sensitivity_attention_v1.py`
  - `src/vpp_dso_sim/dso/observation/happo_structured.py`
  - `tests/test_structured_happo_training.py`
  - `tests/test_envelope_policy_switch.py`
  - `tests/test_privacy_no_private_cost_leak.py`
- 红灯/绿灯：
  - `test_happo_reads_trainer_stability_fields_from_yaml_when_config_not_passed`：先失败于 `summary["episodes"] == 3`，修复后通过。
  - `test_sensitivity_attention_policy_blends_rule_warmstart_with_actor_by_residual_eta`：先失败于缺少 `residual_rule_blend_enabled`，修复后通过。
  - `test_happo_flattened_structured_dso_observation_keeps_privacy_metadata`：先失败于缺少 `privacy_boundary`，修复后通过。
- 最新回归命令：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_structured_happo_training.py \
  tests/test_envelope_policy_switch.py \
  tests/test_structured_smoke_rollout.py \
  tests/test_training_step_no_nan.py \
  tests/test_reward_rebalance.py \
  tests/test_multi_agent_env.py \
  tests/test_sensitivity_shapes.py \
  tests/test_sensitivity_finite_difference.py \
  tests/test_privacy_no_private_cost_leak.py \
  tests/test_paper_training_experiment.py::test_happo_checkpoint_frozen_eval_runs
```

- 结果：28 passed，1 个 jupyter path deprecation warning。
- 新增最终报告：

```text
docs/experiments/dso_sensitivity_attention_final_implementation_report.md
```

- 结论边界：仍不声明完整仓库 pytest 或 paper-long 收敛。

## 4. 后续必须补充的实验

| 实验 | 目的 | 最小验收 |
|---|---|---|
| schema tests | 验证 ActionUnit / NetworkObject / sensitivity tensor 数据结构 | pytest 通过 |
| decoder tests | 验证 safe decoder 不会越过 hard FR/DOE | pytest 通过 |
| actor forward tests | 验证 variable A/K + mask 能前向传播 | 无 NaN/Inf，shape 正确 |
| rule_v0 baseline smoke | 保证旧规则 baseline 未破坏 | 旧测试通过 |
| structured smoke rollout | 验证 `sensitivity_attention_v1` 可跑一个 deterministic rollout | 输出 action units / objects / decoded envelope |
| short training sanity | 验证最短训练更新无 NaN/Inf | 有 update/loss metrics |
| paper-long preflight | paper-long 之前的配置哈希、seed、路径、checkpoint 审计 | 无配置/路径/隐私/shape 风险 |
