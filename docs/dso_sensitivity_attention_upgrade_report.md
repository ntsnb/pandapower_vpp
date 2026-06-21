# DSO 敏感度注意力 operating envelope 改造报告

Updated: 2026-05-28 Asia/Shanghai

## 1. 报告用途

本文档作为本轮 DSO agent 改造的持续台账，记录从规则型
`dso_operating_envelope` 到 `sensitivity_attention_v1` 的实现过程、测试过程、
实验过程、曲线/损失文件位置和未解决风险。

报告不是最后一次性补写的总结，而是随着每个阶段同步更新。需要长期保留：

- 改了哪些文件；
- 为什么这样改；
- 每项修改保护了哪个科研边界；
- 跑了哪些测试和 smoke；
- 短训练 sanity 的结果；
- reward、loss、KL、entropy、grad norm 等曲线文件位置；
- 已知失败、风险和是否属于迁移前遗留问题。

## 2. Source Requirement

Authoritative prompt:

```text
/mnt/sda/home/niutiansen/dso_sensitivity_attention_agent_prompt.md
```

Core target:

```text
Replace the fixed-ratio DSO operating-envelope guidance with a trainable
ActionUnit x NetworkObject sensitivity-aware bipartite-attention DSO actor,
while preserving rule_v0 and legacy flat/MLP baselines.
```

## 3. 不可违反的研究边界

1. FRObject / FR/DOE remains the hard safety boundary.
2. Neural networks must not directly output hard bounds.
3. `dso_operating_envelope` is guidance, not a market clearing award or settlement result.
4. No `zone_id` in v1.
5. No `reliability` field in v1 actor observation.
6. DSO execution actor must not read VPP private true cost, private SOC internals, comfort preferences, or oracle-only fields.
7. Rule-based envelope remains a runnable baseline.
8. Legacy flat DSO observation remains a runnable baseline.
9. Existing HAPPO/HATRPO/MATD3/HASAC paths must not be deleted.
10. AC replay/projection/pandapower checks remain mandatory after dispatch.

## 4. 分阶段执行记录

| Phase | Goal | Status | Evidence |
|---|---|---|---|
| 0 | Repository map | Done | `docs/agents/repo_map.md`，子代理 Cicero 只读扫描 |
| 1 | AGENTS and memory/harness/hooks | Done for current slice | `docs/agents/*`, `scripts/agent_harness.py`, `scripts/agent_hooks/*` |
| 2 | Canonical schemas | Done | `src/vpp_dso_sim/dso/envelope/schemas.py` and tests |
| 3 | ActionUnit builder and NetworkObject selector | Done | `tests/test_action_units.py`, `tests/test_network_objects.py` |
| 4 | Sensitivity matrix/cache | Done for finite-difference slice | `tests/test_sensitivity_shapes.py`, `tests/test_sensitivity_finite_difference.py` |
| 5 | Structured observation | Done | `tests/test_structured_observation_shapes.py`, privacy test |
| 6 | Bipartite attention actor | Done for forward + BC sanity | `tests/test_bipartite_attention_actor.py`, short-train loss CSV |
| 7 | Safe decoder | Done | `tests/test_safe_decoder.py` |
| 8 | Simulator routing | Done | `tests/test_envelope_policy_switch.py` |
| 9 | Reward and training support | Partial | BC sanity + structured HAPPO minimal training/eval; runtime envelope actor checkpoint loading implemented |
| 10 | Configs and experiments | Done for smoke/sanity | baseline/new/ablation configs created |
| 11 | Tests and smoke | Done for current slice | pytest + smoke logs in experiment log |
| 12 | Final report | In progress | this report updated with evidence |

## 5. 核心代码修改前的仓库状态

进入本轮改造前，仓库已经存在以下未提交改动：

```text
M src/vpp_dso_sim/experiments/paper_training.py
M src/vpp_dso_sim/optimization/ac_security_projection.py
M src/vpp_dso_sim/simulation/simulator.py
M tests/test_ac_security_projection.py
M tests/test_paper_training_experiment.py
?? docs/reward_terms_chinese_explained.md
?? paper_method_experiment_cn.*
```

这些文件不是本轮新产生的改动。后续实现必须绕开它们的无关变化，除非用户明确要求，不回滚。

## 6. 改造过程日志

### 2026-05-28 Phase 0 Start

已执行动作：

- 读取用户目标 prompt。
- 读取项目 `AGENTS.md`。
- 确认项目根目录：

```text
/mnt/sda/home/niutiansen/panda power/pandapower-vpp-dso-sim
```

- 启动 multi-agent / superpowers 支持：
  - 已创建只读扫描子代理 `Cicero`，用于定位 DSO envelope、actor、FR/DOE、sensitivity 和测试位置。

初步发现：

- Existing DSO/VPP multi-agent env is in `src/vpp_dso_sim/envs/multi_agent_env.py`.
- Existing operating envelope is currently built through `Simulator._build_dso_operating_envelope(...)`.
- Existing static feasible region is in `src/vpp_dso_sim/optimization/feasibility_region.py`.
- Existing AC-aware projection path is in `src/vpp_dso_sim/optimization/ac_security_projection.py`.
- Existing network sensitivity helper exists at `src/vpp_dso_sim/network/sensitivity.py`, but it is not yet the required ActionUnit x NetworkObject tensor pipeline.

### 2026-05-28 子代理只读扫描结果已合并

子代理 `Cicero` 的结论：

- `Simulator.step()` 在逐 VPP 构造 bid、FR 和 envelope 后，将结果写入
  `records["dso_operating_envelope"]`。
- 当前 envelope 主入口是
  `src/vpp_dso_sim/simulation/simulator.py::_build_dso_operating_envelope()`。
- 当前 AC-aware 收紧入口是
  `Simulator._tighten_dso_envelope_with_ac_sensitivity()`。
- 当前 DSO actor 在 HAPPO/CTDE 路径中来自
  `src/vpp_dso_sim/learning/ctde_networks.py::build_privacy_separated_ctde_modules()`。
- 当前 HAPPO 训练器在 `src/vpp_dso_sim/learning/advanced_marl.py`，并复用上面的
  `dso_actor`。
- 当前 HATRPO 训练器在 `src/vpp_dso_sim/learning/hatrpo.py`，保留为 baseline 家族成员。
- 当前 sensitivity 只有 aggregate VPP active-power finite-difference helper，
  还没有 prompt 要求的 `ActionUnit x NetworkObject x Channel` 张量和 cache。
- 当前 critical object 选择只在旧 helper 中使用最低/最高电压 bus 和最大 line loading，
  还不是可配置 top-k selector。

这些结论决定本轮不直接覆盖旧函数，而是新增 versioned DSO 模块，再通过 feature flag 路由。

### 2026-05-28 Phase 1 文档/记忆/脚手架初始化

已新增或更新：

- `AGENTS.md`：追加 `sensitivity_attention_v1` 的研究边界、feature flags、baseline 保留和实现纪律。
- `docs/agents/subagents.md`：记录子代理职责与当前真实子代理调用结果。
- `docs/agents/MEMORY.md`：记录长期约束，特别是 FR/DOE hard boundary、无 `zone_id`、无 `reliability`、无私有成本泄漏。
- `docs/agents/DECISIONS.md`：记录 ADR-001 到 ADR-007。
- `docs/agents/HANDOFF.md`：记录 Phase 0/1 handoff。
- `docs/agents/KNOWN_FAILURES.md`：记录进入本轮改造前的已有 dirty worktree。
- `docs/agents/RUNBOOK.md`：记录 baseline、structured tests、实验台账位置。
- `scripts/agent_harness.py`：新增阶段声明与 handoff 追加脚手架。
- `scripts/agent_hooks/*.sh`：新增 schema/model/baseline/smoke hook 骨架。

说明：

- `scripts/agent_hooks/smoke_training.sh` 当前通过 `${PYTHON:-./.venv-server/bin/python}`
  调用 `scripts/run_smoke.py` 和 `scripts/run_short_train.py`。
- 本阶段没有改训练主逻辑，也没有改变现有 `rule_v0` 行为。

### 2026-05-28 Phase 2-7 基础模块落地

新增 versioned 包：

- `src/vpp_dso_sim/dso/envelope/schemas.py`
- `src/vpp_dso_sim/dso/sensitivity/selectors.py`
- `src/vpp_dso_sim/dso/sensitivity/finite_difference.py`
- `src/vpp_dso_sim/dso/sensitivity/cache.py`
- `src/vpp_dso_sim/dso/observation/legacy_flat.py`
- `src/vpp_dso_sim/dso/observation/structured_bipartite.py`
- `src/vpp_dso_sim/dso/models/bipartite_attention_actor.py`
- `src/vpp_dso_sim/dso/envelope/safe_decoder.py`
- `src/vpp_dso_sim/dso/envelope/rule_v0.py`
- `src/vpp_dso_sim/dso/envelope/policy_switch.py`
- `src/vpp_dso_sim/dso/envelope/sensitivity_attention_v1.py`

新增测试：

- `tests/test_action_units.py`
- `tests/test_network_objects.py`
- `tests/test_sensitivity_shapes.py`
- `tests/test_sensitivity_finite_difference.py`
- `tests/test_structured_observation_shapes.py`
- `tests/test_bipartite_attention_actor.py`
- `tests/test_safe_decoder.py`
- `tests/test_legacy_baseline_unchanged.py`
- `tests/test_envelope_policy_switch.py`
- `tests/test_privacy_no_private_cost_leak.py`

验证命令和结果已写入 `docs/experiments/dso_sensitivity_attention_experiment_log.md`。

当前边界：

- 这些模块已经能通过 shape/mask/privacy/finite-difference/actor-forward 测试。
- `SensitivityAttentionEnvelopePolicy.build()` 已能基于结构化 observation 和 attention actor 生成
  safe-decoded guidance envelope。
- `Simulator.step()` 已支持从 config 路由到 `sensitivity_attention_v1`。
- short training sanity 已补充，loss 曲线保存在实验输出目录；这仍不是 paper-long 收敛结论。

### 2026-05-28 Phase 9-11 smoke 与短训练 sanity

新增配置：

- `configs/baseline_rule_v0.yaml`
- `configs/happo_legacy_mlp.yaml`
- `configs/happo_sensitivity_attention_v1.yaml`
- `configs/ablation_no_sensitivity_edges.yaml`
- `configs/ablation_no_action_self_attention.yaml`
- `configs/ablation_no_width_penalty.yaml`

新增脚本：

- `scripts/run_smoke.py`
- `scripts/run_short_train.py`

新增说明文档：

- `docs/dso_sensitivity_attention_architecture_plan.md`
- `docs/experiments/baselines.md`
- `docs/experiments/training_protocol.md`

路径说明：

- prompt 原建议 `docs/architecture/dso_sensitivity_attention_plan.md`。
- 本仓库已有 `docs/architecture.md` 文件，无法同时创建同名目录，因此采用
  `docs/dso_sensitivity_attention_architecture_plan.md`。

实际运行：

- `rule_v0` smoke：2 steps，通过，无 NaN/Inf，输出在
  `outputs/dso_sensitivity_attention/baseline_rule_v0_smoke_seed0`。
- `sensitivity_attention_v1` smoke：2 steps，通过，无 NaN/Inf，输出在
  `outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0`。
- `sensitivity_attention_v1` short BC sanity：256 updates，通过，无 NaN/Inf，输出在
  `outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0`。

短训练 loss：

```text
initial bc_loss = 0.5577945709228516
final   bc_loss = 0.00032648438354954123
initial grad_norm = 6.576420783996582
final   grad_norm = 0.0034202593378722668
```

边界说明：

- 这证明 attention actor 可优化、loss 可记录、checkpoint 可保存。
- 这不是 HAPPO 在线训练收敛，也不是 paper-long 结果。
- 当前 HAPPO 训练器已能在 `dso.observation_mode = structured_bipartite` 或
  `dso.envelope_policy = sensitivity_attention_v1` 时使用结构化 DSO actor，并写出
  `policy_loss`、`entropy_mean`、`approx_kl`、`grad_norm`。
- 当前结构化 HAPPO checkpoint 已支持 frozen evaluation。
- 当前 runtime `SensitivityAttentionEnvelopePolicy` 已支持从配置加载 structured HAPPO attention actor 权重。
- 当前 HAPPO 已补充 `target_kl` early stop、`nan_guard`、advantage normalization 和训练/frozen eval 一致的 deterministic observation normalization。

### 2026-05-28 HAPPO stability fields 与 frozen eval normalization

为 paper-long 前的损失曲线和稳定性诊断，`HAPPOConfig` 已新增：

- `target_kl`
- `normalize_observations`
- `normalize_advantages`
- `nan_guard`

训练输出新增：

- `target_kl`
- `target_kl_exceeded`
- `nan_guard_triggered`
- `observation_normalization_stats`
- `kl_early_stop_count`

当前实现说明：

- observation normalization 是逐 observation 向量的 deterministic normalization。
- frozen eval 会读取 checkpoint config 中的 `normalize_observations` 并使用同一处理。
- 这不是长期 running mean/std normalizer，paper-long 前仍建议升级。

红灯测试：

```bash
./.venv-server/bin/python -m pytest -q tests/test_structured_happo_training.py::test_structured_happo_checkpoint_frozen_eval_runs
```

旧实现失败于：

```text
KeyError: 'normalize_observations'
```

补齐 evaluator 后绿灯：

```text
1 passed
```

当前持久化最小训练输出：

```text
outputs/dso_sensitivity_attention/happo_structured_minimal_seed0_current/
```

当前持久化 frozen eval 输出：

```text
outputs/dso_sensitivity_attention/happo_structured_frozen_eval_seed1_current/
```

中文总报告已新增：

```text
docs/experiments/dso_sensitivity_attention_change_experiment_report.md
```

### 2026-05-28 VPP preferred-range bonus gating

按 prompt Phase 9 的要求，VPP dispatch 的 preferred-range bonus 已改成 gated 形式：

```text
preferred_bonus =
  preferred_range_weight
  * inside_preferred_range
  * guidance_strength_lambda
  * width_gate
  * effectiveness_gate
```

其中：

- `inside_preferred_range`：VPP 实际出力是否落在 DSO preferred range 内。
- `guidance_strength_lambda`：DSO actor 输出的引导强度，没有该字段时 legacy 路径默认 1。
- `width_gate`：preferred range 越接近 hard FR/DOE 全宽，gate 越低；如果 DSO 给出几乎无约束的宽区间，则不给 VPP 额外奖励。
- `effectiveness_gate`：优先读 `effective_response_score`；否则按 projection gap 相对 hard span 衰减。

新增记录字段：

- `preferred_inside_range`
- `preferred_bonus_lambda_gate`
- `preferred_bonus_width_gate`
- `preferred_bonus_effectiveness_gate`
- `preferred_region_score`

验证命令：

```bash
./.venv-server/bin/python -m pytest -q tests/test_reward_rebalance.py tests/test_multi_agent_env.py
```

结果：

```text
9 passed
```

### 2026-05-28 sensitivity raw cache 与 active slice 接入

按 prompt Phase 4 的要求，`SensitivityAttentionEnvelopePolicy` 已接入 `SensitivityCache`：

- cache miss 时运行 pandapower finite-difference estimator，写入 raw sensitivity cache。
- cache hit 时调用 `active_sensitivity_slice()` 得到当前 `NetworkObject x ActionUnit` 切片。
- rollout 记录新增 `sensitivity_cache_hit`、`sensitivity_source`、`sensitivity_cache_step`。

验证命令：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_sensitivity_shapes.py \
  tests/test_sensitivity_finite_difference.py \
  tests/test_envelope_policy_switch.py \
  tests/test_structured_smoke_rollout.py
```

结果：

```text
9 passed
```

说明：当前 2-step smoke 中 critical object 集合会随网络状态变化，因此 current smoke
输出里未必出现 cache hit；cache 命中行为由
`test_sensitivity_attention_policy_reuses_raw_sensitivity_cache_for_active_slice` 固定。

### 2026-05-28 runtime envelope policy 加载 structured HAPPO actor checkpoint

为闭合“训练得到的 DSO actor -> 在线 operating envelope policy”链路，`SensitivityAttentionEnvelopePolicy`
现在支持：

- `dso.actor.checkpoint_path`
- `dso.actor_checkpoint_path`

当配置提供 checkpoint 时，policy 会严格提取并加载：

- direct `BipartiteSensitivityDSOActor.state_dict()`
- `attention_actor.*`
- `dso_actor.attention_actor.*`，即 structured HAPPO checkpoint 中的 attention actor 子模块

加载失败会抛出错误，不会 silent fallback 到随机初始化。

新增 rollout 记录字段：

- `dso_actor_checkpoint_loaded`
- `dso_actor_checkpoint_path`
- `dso_actor_checkpoint_source`

红灯测试：

```bash
./.venv-server/bin/python -m pytest -q tests/test_envelope_policy_switch.py::test_sensitivity_attention_policy_loads_structured_happo_checkpoint_actor
```

旧实现失败于：

```text
KeyError: 'dso_actor_checkpoint_loaded'
```

绿灯与回归：

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_envelope_policy_switch.py \
  tests/test_structured_smoke_rollout.py \
  tests/test_structured_happo_training.py
```

结果：

```text
9 passed
```

### 2026-05-28 HAPPO 结构化最小训练与日志字段补齐

本轮为满足“曲线/损失必须保留”的要求，对 HAPPO update metrics 补充：

- `entropy_mean`：每个角色更新时策略分布熵，用于判断探索是否过早消失。
- `approx_kl`：旧策略与新策略的近似 KL，用于判断 HAPPO/PPO 更新步是否过大。

验证命令：

```bash
./.venv-server/bin/python -m pytest -q tests/test_structured_happo_training.py
```

结果：

```text
1 passed
```

同时跑了 1 episode / 2 step 的结构化 HAPPO 最小训练：

```bash
./.venv-server/bin/python -c "from vpp_dso_sim.learning.advanced_marl import HAPPOConfig, train_happo; train_happo(config_path='configs/happo_sensitivity_attention_v1.yaml', output_dir='outputs/dso_sensitivity_attention/happo_structured_minimal_seed0_current', config=HAPPOConfig(horizon_steps=2, episodes=1, hidden_dim=32, ppo_epochs=1, seed=0, critic_use_action_summary=True))"
```

关键结果：

```text
episode_reward = 5.67563214334187
episode_cost = 1.7059830742162847
violation_count = 0
projection_gap_mw = 0.0
critic_loss = 0.07204630225896835
dso_policy_loss = -0.01804180070757866
dispatch_policy_loss = -0.03949027508497238
portfolio_policy_loss = -0.4796231687068939
dso_entropy_mean = 0.7189385294914246
dso_approx_kl = -0.0005803108215332031
param_delta_l2 = 0.0509931817650795
```

输出目录：

```text
outputs/dso_sensitivity_attention/happo_structured_minimal_seed0_current/
```

其中：

- `happo_episode_metrics.csv`：reward、cost、violation、projection gap、critic loss。
- `happo_step_metrics.csv`：step 级环境指标。
- `happo_update_metrics.csv`：DSO/dispatch/portfolio 的 policy loss、entropy、approx KL、grad norm。
- `happo_training_summary.json`：结构化 actor 元数据、config、checkpoint 路径和最终指标。

结构化 frozen eval 已补充：

```bash
./.venv-server/bin/python -c "from vpp_dso_sim.learning.advanced_marl import evaluate_happo_checkpoint; evaluate_happo_checkpoint(config_path='configs/happo_sensitivity_attention_v1.yaml', checkpoint_path='outputs/dso_sensitivity_attention/happo_structured_minimal_seed0_current/happo_best_checkpoint.pt', output_dir='outputs/dso_sensitivity_attention/happo_structured_frozen_eval_seed1_current', horizon_steps=2, seed=1)"
```

关键结果：

```text
evaluation_mode = frozen_mean_argmax_actor
dso_actor_observation_mode = structured_bipartite
dso_actor_type = sensitivity_attention_v1_structured_happo
structured_dso_actor_loaded = true
total_reward = 5.655994539527365
total_cost = 1.6341365382859143
total_violation_count = 0
```

## 7. 实验、曲线和损失记录方案

本轮所有 smoke、短训练、paper-long 之前的 sanity 都必须记录到：

```text
docs/experiments/dso_sensitivity_attention_experiment_log.md
```

每次运行必须保留：

- command;
- config file;
- random seed;
- output directory;
- checkpoint path;
- reward curve path;
- loss curve path;
- projection gap curve path;
- shield/safety curve path;
- post-AC violation counts;
- NaN/Inf check result;
- conclusion.

预期曲线/损失文件：

| Artifact | Meaning |
|---|---|
| `*_episode_metrics.csv` | episode reward, cost, projection gap, safety counters |
| `*_step_metrics.csv` | step-level reward components, DSO/dispatch/portfolio components, post-AC safety |
| `*_loss_metrics.csv` or update metrics | policy loss, value/critic loss, entropy, KL, grad norm |
| `selected_network_objects.csv` | critical bus/line/trafo tokens selected by DSO |
| `action_units.csv` | ActionUnit ids, VPP/PCC/bus mapping |
| `sensitivity_edges.csv` | active sensitivity tensor summaries and confidence |
| `dso_operating_envelope.csv` | envelope guidance plus cache/source/actor/decoder diagnostics |
| `dso_actor_outputs.csv` | raw center/width/direction/lambda outputs |
| `decoded_operating_envelope.csv` | safe-decoded preferred ranges and hard bounds |

当前已经保留的报告与曲线索引：

| 类别 | 路径 | 内容 |
|---|---|---|
| 总改造报告 | `docs/dso_sensitivity_attention_upgrade_report.md` | 改造阶段、文件、测试、风险和验收清单 |
| 实验台账 | `docs/experiments/dso_sensitivity_attention_experiment_log.md` | 每次 smoke/训练命令、输出目录、loss/reward/KL/entropy 文件 |
| 架构方案 | `docs/dso_sensitivity_attention_architecture_plan.md` | ActionUnit x NetworkObject sensitivity-attention 设计 |
| baseline 说明 | `docs/experiments/baselines.md` | rule_v0、legacy MLP、sensitivity_attention_v1、ablation |
| 训练协议 | `docs/experiments/training_protocol.md` | smoke、short sanity、paper-long 前置流程 |
| 当前结构化 smoke | `outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0_current/` | step metrics、envelope、action units、objects、sensitivity、actor outputs |
| 当前 BC loss | `outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0_current/dso_sensitivity_attention_short_train_loss_metrics.csv` | `bc_loss`、`center_loss`、`width_loss`、`direction_loss`、`grad_norm` |
| 当前 HAPPO update loss | `outputs/dso_sensitivity_attention/happo_structured_minimal_seed0_current/happo_update_metrics.csv` | `policy_loss`、`entropy_mean`、`approx_kl`、`grad_norm` |
| 当前 HAPPO episode 曲线 | `outputs/dso_sensitivity_attention/happo_structured_minimal_seed0_current/happo_episode_metrics.csv` | reward、cost、projection gap、critic loss |
| 当前 HAPPO frozen eval | `outputs/dso_sensitivity_attention/happo_structured_frozen_eval_seed1_current/` | frozen deterministic step metrics、summary、post-AC safety |
| 可随文档保留的小型 artifact | `docs/experiments/dso_sensitivity_attention_artifacts/` | 从 `outputs/` 复制出的关键 CSV/JSON，避免因 `outputs/` 被 git ignore 而丢失报告证据 |

## 8. 验收清单

| Requirement | Status | Evidence |
|---|---|---|
| `rule_v0` baseline runs | Done | script smoke + `test_legacy_baseline_unchanged.py` |
| legacy flat DSO observation works | Done | `test_structured_observation_shapes.py` |
| legacy DSO MLP actor works | Done for regression | HAPPO/HATRPO tests passed; no legacy actor code removed |
| `sensitivity_attention_v1` smoke rollout runs | Done | script smoke output directory recorded |
| short training update has no NaN/Inf | Done for BC sanity | 256-step short train, no NaN/Inf |
| no `zone_id` introduced | Done for v1 schema/obs | schema/privacy tests |
| no `reliability` field introduced | Done for v1 schema/obs | schema/privacy tests |
| no private VPP true cost in DSO actor observation | Done for v1 schema/obs | privacy test |
| ActionUnit supports VPP-PCC or VPP-bus | Done | `build_action_units` |
| NetworkObject supports bus/line/trafo | Done | selector and tests |
| sensitivity tensor supports masks | Done | sensitivity shape tests |
| Q channels masked when disabled | Done | sensitivity shape tests |
| safe decoder invariants tested | Done | `test_safe_decoder.py` |
| active sensitivity is slice of raw cache | Done | `active_sensitivity_slice()` and test |
| configs log seed and config hash | Done for scripts | smoke/short-train summaries |
| reward components separately logged | Done for current slice | simulator reward records exist; DSO width/smoothness/effective-response and VPP preferred bonus gates are logged |
| HAPPO update loss/KL/entropy logged | Done for current trainer slice | `happo_update_metrics.csv`, `tests/test_structured_happo_training.py` |
| structured HAPPO frozen eval runs | Done for current slice | `tests/test_structured_happo_training.py`, frozen eval current output |
| runtime envelope policy loads structured actor checkpoint | Done for current slice | `test_sensitivity_attention_policy_loads_structured_happo_checkpoint_actor` |
| AGENTS.md updated | Done | DSO v1 section appended |
| memory/handoff docs exist | Done | `docs/agents/*` |
| baseline and new configs exist | Done | six configs created |
| final changed-files/tests/risks summary exists | In progress | this report |

## 9. 当前验证汇总

已运行：

```text
新增 DSO v1 测试集合：19 passed
旧 grid-aware envelope / FR / timeseries / multi-agent smoke：15 passed
HAPPO / HATRPO / HAPPO trainer 轻量回归：11 passed
完整 pytest：exit code 0
结构化 HAPPO 日志字段测试：1 passed
本轮受影响测试集合：22 passed
runtime checkpoint 加载回归集合：9 passed
```

所有测试均只出现同一个 jupyter path deprecation warning。

已生成实验/曲线文件：

```text
outputs/dso_sensitivity_attention/baseline_rule_v0_smoke_seed0/
outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0/
outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0/
outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0_current/
outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0_current/
outputs/dso_sensitivity_attention/happo_structured_minimal_seed0_current/
outputs/dso_sensitivity_attention/happo_structured_frozen_eval_seed1_current/
docs/experiments/dso_sensitivity_attention_artifacts/
```

短训练 loss 文件：

```text
outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0/dso_sensitivity_attention_short_train_loss_metrics.csv
outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0_current/dso_sensitivity_attention_short_train_loss_metrics.csv
outputs/dso_sensitivity_attention/happo_structured_minimal_seed0_current/happo_update_metrics.csv
outputs/dso_sensitivity_attention/happo_structured_frozen_eval_seed1_current/happo_frozen_eval_step_metrics.csv
```

## 10. 已知风险

- 当前 worktree 已经有历史脏改动，后续每次总结必须区分“本轮新增”和“已有未提交”。
- `docs/architecture.md` already exists as a file; the new prompt asks for `docs/architecture/dso_sensitivity_attention_plan.md`, so both can coexist but links must be explicit.
- 当前项目已经有 `network/sensitivity.py`；新模块不能把旧的 aggregate sensitivity 误当作新的
  ActionUnit x NetworkObject tensor cache。
- prompt 要求研究级集成。smoke 成功只能说明接口跑通，不能声称算法已经收敛。
- 现有 AGENTS.md 仍包含 legacy `zone_ids` 说明；新 v1 DSO actor 不允许新增或依赖
  `zone_id` 字段，文档更新时必须明确 v1 与 legacy 的边界。
- 当前 HAPPO 最小训练只有 1 episode / 2 step，只能证明链路和日志健康，不能证明 paper-long 收敛。
- HAPPO trainer 的 `target_kl` early stop、`nan_guard` fail-fast、advantage normalization、deterministic observation normalization 已有测试覆盖；后续 paper-long 前仍建议升级为 checkpointed running normalizer。

## 11. 下一次报告更新点

下一次更新应填入：

- runtime sensitivity cache 是否接入 envelope policy；
- paper-long preflight 的输出目录、配置哈希、seed、checkpoint；
- paper-long 真实 reward/loss/KL/entropy/projection-gap/AC-safety 曲线。
