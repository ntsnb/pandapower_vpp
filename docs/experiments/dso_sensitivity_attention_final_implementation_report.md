# Final Implementation Report

Updated: 2026-05-28 Asia/Shanghai

## Summary

本轮将 DSO operating envelope 路径从单一规则推荐扩展为版本化双路径：

- `rule_v0`：保留原有规则 baseline。
- `sensitivity_attention_v1`：新增 ActionUnit x NetworkObject sensitivity-aware bipartite-attention DSO actor，经 safe decoder 映射回 FR/DOE hard bounds。

当前报告只声明工程链路、测试、smoke 和短训练 sanity；不声明 paper-long 收敛。

更详细的中文改造/实验/loss 曲线报告见：

- `docs/experiments/dso_sensitivity_attention_full_process_report.md`

## Research boundaries preserved

- FRObject / FR/DOE 仍是 hard safety boundary。
- 神经网络不直接输出 `p_hard_min_mw` / `p_hard_max_mw`。
- `dso_operating_envelope` 仍标记为 `award_status = envelope_guidance`，不是 market award 或 settlement。
- v1 DSO actor observation 不引入 `zone_id` 或 `reliability`。
- v1 DSO actor observation 不读取 VPP private true cost、oracle cost、comfort preference、private SOC internals。
- `rule_v0`、legacy flat observation、legacy MLP actor、HAPPO/HATRPO/MATD3/HASAC 家族未删除。

## Files changed

主要修改集中在：

- `AGENTS.md`
- `configs/*.yaml`
- `scripts/agent_harness.py`
- `scripts/agent_hooks/*.sh`
- `scripts/run_smoke.py`
- `scripts/run_short_train.py`
- `src/vpp_dso_sim/dso/**`
- `src/vpp_dso_sim/envs/reward_design.py`
- `src/vpp_dso_sim/learning/advanced_marl.py`
- `src/vpp_dso_sim/simulation/simulator.py`
- `tests/test_*dso*`
- `tests/test_envelope_policy_switch.py`
- `tests/test_structured_happo_training.py`
- `tests/test_privacy_no_private_cost_leak.py`
- `docs/agents/*`
- `docs/experiments/*`
- `docs/dso_sensitivity_attention_upgrade_report.md`
- `docs/experiments/dso_sensitivity_attention_acceptance_audit.md`

注意：进入本轮前 worktree 已有其他历史脏改动；本报告不把所有 dirty 文件都归因为本轮。

## New modules

| 模块 | 作用 |
|---|---|
| `src/vpp_dso_sim/dso/envelope/schemas.py` | ActionUnit、NetworkObject、SensitivityEdgeTensor、StructuredDSOObservation、DecodedOperatingEnvelopeRecord |
| `src/vpp_dso_sim/dso/envelope/rule_v0.py` | rule baseline adapter |
| `src/vpp_dso_sim/dso/envelope/safe_decoder.py` | actor center/width 到 FR/DOE 内 preferred range |
| `src/vpp_dso_sim/dso/envelope/sensitivity_attention_v1.py` | runtime structured DSO envelope policy |
| `src/vpp_dso_sim/dso/envelope/policy_switch.py` | config feature flag routing |
| `src/vpp_dso_sim/dso/sensitivity/selectors.py` | ActionUnit builder 和 critical NetworkObject selector |
| `src/vpp_dso_sim/dso/sensitivity/finite_difference.py` | pandapower finite-difference sensitivity tensor |
| `src/vpp_dso_sim/dso/sensitivity/cache.py` | raw cache、active slice、refresh decision、priority refresh merge |
| `src/vpp_dso_sim/dso/observation/legacy_flat.py` | legacy flat encoder adapter |
| `src/vpp_dso_sim/dso/observation/structured_bipartite.py` | structured DSO observation builder |
| `src/vpp_dso_sim/dso/observation/happo_structured.py` | HAPPO flattened structured observation and privacy metadata |
| `src/vpp_dso_sim/dso/models/bipartite_attention_actor.py` | bipartite attention actor |
| `src/vpp_dso_sim/dso/models/structured_happo_actor.py` | HAPPO-compatible structured DSO Gaussian actor |
| `src/vpp_dso_sim/experiments/dso_sensitivity_attention.py` | smoke/short-train experiment helpers |

## Modified modules

| 模块 | 修改摘要 |
|---|---|
| `src/vpp_dso_sim/simulation/simulator.py` | 根据 config 调用 `rule_v0` 或 `sensitivity_attention_v1` |
| `src/vpp_dso_sim/envs/multi_agent_env.py` | multi-agent env 使用 simulator policy helper |
| `src/vpp_dso_sim/envs/reward_design.py` | VPP preferred-range bonus 改为 lambda/width/effectiveness gated |
| `src/vpp_dso_sim/learning/advanced_marl.py` | HAPPO structured actor、YAML trainer config、target KL、normalization、nan guard、frozen eval |
| `src/vpp_dso_sim/dso/sensitivity/cache.py` | 增加 update-period、TTL、电压/loading 漂移、FR 宽度变化、projection-gap 历史、缺失对象触发，以及 capped priority ActionUnit partial refresh |
| `src/vpp_dso_sim/dso/sensitivity/finite_difference.py` | 记录 `sensitivity_allocation_mode` 和 `sensitivity_allocation_weights`，使 VPP-PCC/VPP-bus 扰动分配可审计 |
| `scripts/agent_harness.py` | 从阶段声明器升级为可执行 phase tests、记录 touched files、失败即返回非零的 harness |
| `AGENTS.md` | 增加 DSO v1 不可违反边界与测试要求 |

## Tests added

- `tests/test_action_units.py`
- `tests/test_network_objects.py`
- `tests/test_sensitivity_shapes.py`
- `tests/test_sensitivity_finite_difference.py`
- `tests/test_structured_observation_shapes.py`
- `tests/test_bipartite_attention_actor.py`
- `tests/test_safe_decoder.py`
- `tests/test_legacy_baseline_unchanged.py`
- `tests/test_envelope_policy_switch.py`
- `tests/test_structured_smoke_rollout.py`
- `tests/test_training_step_no_nan.py`
- `tests/test_privacy_no_private_cost_leak.py`
- `tests/test_structured_happo_training.py`
- `tests/test_agent_harness.py`

## Tests run

最新受影响集合：

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

最新 hook smoke：

```bash
bash scripts/agent_hooks/smoke_training.sh
```

完整仓库测试：

```bash
./.venv-server/bin/python -m pytest -q
```

测试收集核查：

```bash
./.venv-server/bin/python -m pytest --collect-only -q
```

## Passing tests

上一轮受影响集合回归为 `28 passed`，1 个 jupyter path deprecation warning。

最新 cache-refresh / structured-HAPPO 受影响集合回归为 `34 passed`，1 个 jupyter path deprecation warning。

完整仓库 `pytest -q` 最新结果为 exit code 0，1 个 jupyter path deprecation warning。

`pytest --collect-only -q` 最新收集到 160 个测试用例。

单项红绿证据：

- HAPPO YAML trainer config 读取：`test_happo_reads_trainer_stability_fields_from_yaml_when_config_not_passed`
- residual warm-start schedule：`test_sensitivity_attention_policy_blends_rule_warmstart_with_actor_by_residual_eta`
- flattened structured DSO privacy metadata：`test_happo_flattened_structured_dso_observation_keeps_privacy_metadata`
- sensitivity cache refresh decision：`test_sensitivity_refresh_decision_triggers_on_period_ttl_and_grid_state_changes`
- priority refresh merge：`test_merge_sensitivity_update_overwrites_priority_action_units_without_losing_raw_cache`
- runtime partial refresh logging：`test_sensitivity_attention_policy_refreshes_cache_when_update_period_elapsed`
- finite-difference allocation weights：`test_finite_difference_records_action_unit_allocation_weights`

## Failing tests

当前最近一次受影响集合没有失败测试。

完整 `./.venv-server/bin/python -m pytest -q` 已在本轮末尾重跑，exit code 0；仍需注意这不等同于 paper-long 收敛证明。

## Smoke rollouts

Hook 命令：

```bash
bash scripts/agent_hooks/smoke_training.sh
```

结果摘要：

- `configs/baseline_rule_v0.yaml`
  - `envelope_policy = rule_v0`
  - `steps = 2`
  - `dso_operating_envelope = 2`
  - `constraint_violations = 0`
  - `nan_or_inf_detected = False`
- `configs/happo_sensitivity_attention_v1.yaml`
  - `envelope_policy = sensitivity_attention_v1`
  - `steps = 2`
  - `dso_operating_envelope = 2`
  - `constraint_violations = 0`
  - `nan_or_inf_detected = False`
  - `sensitivity_allocation_mode = equal_pp_element_refs` logged in envelope CSV
- `configs/happo_legacy_mlp.yaml`
  - `envelope_policy = rule_v0`
  - `observation_mode = legacy_flat`
  - `steps = 2`
  - `constraint_violations = 0`
  - `nan_or_inf_detected = False`

## Short training sanity

Hook 命令中的 short train：

```bash
./.venv-server/bin/python scripts/run_short_train.py \
  --config configs/happo_sensitivity_attention_v1.yaml \
  --seed 0 \
  --steps 256 \
  --output-dir outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0
```

结果摘要：

- `steps_completed = 256`
- `nan_or_inf_detected = False`
- `initial_loss = 0.5577945709228516`
- `final_loss = 0.00032648438354954123`
- loss 曲线归档：`docs/experiments/dso_sensitivity_attention_artifacts/current_bc_loss_curve.svg`
- loss CSV 归档：`docs/experiments/dso_sensitivity_attention_artifacts/current_bc_loss_metrics.csv`

该结果只证明 BC warm-start sanity，无 NaN/Inf，不证明 paper-long 收敛。

## Baselines preserved

- `rule_v0` baseline：`configs/baseline_rule_v0.yaml` 和 `tests/test_legacy_baseline_unchanged.py`
- legacy flat DSO observation：`src/vpp_dso_sim/dso/observation/legacy_flat.py`
- legacy MLP DSO actor：保留在现有 privacy-separated network builder 路径，structured mode 通过 feature flag 切换
- HAPPO/HATRPO/MATD3/HASAC：未删除；structured DSO actor 当前接入 HAPPO 路径，其他算法保留 legacy baseline

## New configs

- `configs/baseline_rule_v0.yaml`
- `configs/happo_legacy_mlp.yaml`
- `configs/happo_sensitivity_attention_v1.yaml`
- `configs/ablation_no_sensitivity_edges.yaml`
- `configs/ablation_no_action_self_attention.yaml`
- `configs/ablation_no_width_penalty.yaml`

## Privacy audit result

已覆盖：

- schema 层：`ActionUnitState` / `StructuredDSOObservation` 不定义 private cost / comfort / SOC internals。
- runtime structured observation：metadata 标记 `contains_private_true_cost = False`，field names 不含 `zone_id` / `reliability`。
- HAPPO flattened structured observation：`StructuredDSOFlatSpec.privacy_boundary = dso_execution_actor_no_private_vpp_fields`，field names 不含 `zone_id`、`reliability`、`private_true_cost`、`oracle_cost`、`comfort_preference`、`private_soc_internal`。

相关测试：

```bash
./.venv-server/bin/python -m pytest -q tests/test_privacy_no_private_cost_leak.py
```

## Known limitations

- `docs/architecture.md` 已作为文件存在，因此本轮架构计划采用 `docs/dso_sensitivity_attention_architecture_plan.md`；未强行把它迁移成 `docs/architecture/dso_sensitivity_attention_plan.md`，避免破坏现有文档路径。
- sensitivity cache 已实现 raw cache、active slice、refresh triggers、priority ActionUnit 和 partial priority refresh merge；后续 paper-long 需要观察这些触发是否降低不必要 AC finite-difference 计算。
- DSO reward 的 width/projection/smoothness/effective response 日志已有部分链路；完整论文级 reward ablation 仍需 pilot/paper-long 进一步验证。
- observation normalization 目前为 deterministic per-vector normalization，不是可 checkpoint 的 running normalizer。
- 当前 smoke 和 256-step BC sanity 不代表算法收敛。

## Next recommended experiments

1. 24-step pilot：检查 `reward`、`projection_gap_mw`、`shield_intervention_penalty`、`approx_kl`、`entropy_mean`。
2. 96-step pilot：比较 `rule_v0`、`happo_legacy_mlp`、`happo_sensitivity_attention_v1`。
3. ablation：`no_sensitivity_edges`、`no_action_self_attention`、`no_width_penalty`。
4. frozen evaluation：每个 checkpoint 使用 deterministic policy 重放。
5. paper-long 前置：多 seed，保存 HAPPO update metrics、frozen eval summary、AC safety counters。
