# Benchmark V2 Experiment Plan

本文件记录第二阶段实验编排方案。目标不是再做一个能跑通的 demo，而是建立一个可以持续接入
rule-based、CTDE、MARL、优化分解和可视化报告的研究级实验协议。

## 1. 本轮要修正的问题

1. **重复 profile 问题**：原 288 步实验会把同一天 96 点负荷、PV、电价曲线重复三次，无法证明跨日泛化。
2. **网络压力偏 demo**：原 European-LV 场景安全但偏舒适，不能有效测试安全投影、灵活性调用和调度策略。
3. **实验无 split**：训练、同分布评估、拓扑迁移评估没有统一协议，无法做可信算法比较。
4. **指标口径不足**：只看 reward 不够，必须同时记录安全、电气、经济代理、机制和学习相关指标。
5. **训练闭环未纳入 runner**：当前 benchmark runner 先执行 `rule_based`，但已经预留 `algorithm/split/seed`
   协议，后续 CTDE/MARL trainer 必须按同一协议接入。

## 2. 场景设计

主场景使用 `configs/european_lv_benchmark_v2.yaml`：

- 123 bus European-LV-style 低压台区。
- 6 条馈线，含 trunk / lateral 分支。
- 7 个 VPP，同时包含单 PCC VPP 和多节点 VPP。
- F1-F6 均有至少一个可控 DER/VPP 覆盖，避免空白馈线造成实验解释不清。
- 参数调为临界可行压力：
  - `base_load_scale: 1.60`
  - `transformer_sn_mva: 2.05`
  - `line_resistance_scale: 1.16`
  - `line_reactance_scale: 1.10`

这组参数在二次实验中达到约 `min voltage = 0.9338 pu`、`max line loading = 92.04%`，
仍满足当前安全约束 `[0.93, 1.06]` 和线路/变压器 100% 限值。

拓扑 holdout 使用 `configs/ieee33_multi_vpp.yaml`。它不作为主结果场景，而是用于检查策略/调度协议
是否能迁移到另一类拓扑。

## 3. Profile Split

所有 split 使用 `benchmark_profile_pack()` 生成非重复 profile：

- `train_profile / train_mixed`：训练或基线主场景。
- `eval_profile / holdout_peak`：高峰负荷与高电价评估。
- `eval_profile / holdout_cloudy`：低 PV 和云遮挡评估。
- `topology_holdout / holdout_peak`：IEEE33 拓扑迁移评估。

每个 profile pack 都记录到 `profile_quality.csv`，包含日均负荷、PV 日能量、电价均值和峰值。

## 4. 实验编排

入口：

```bash
python examples/11_run_benchmark_experiment.py
```

默认协议：

- `horizon_steps = 288`，15 分钟间隔，三天。
- `seeds = 3101,3102,3103,3104,3105`。
- `splits = train_profile, eval_profile, topology_holdout`。
- 当前已执行算法：`rule_based`。
- 后续可接入：`privacy_separated_ctde`、`mappo`、`ippo`、`maddpg`、`qp_disaggregation`。

输出目录默认：

```text
outputs/benchmark_v2/
```

本轮正式候选实验输出：

```text
outputs/benchmark_v2_research_candidate/
```

## 5. 指标表

`seed_metrics.csv` 每个 seed/run 一行，至少包含：

- 实验协议：`algorithm`, `split`, `scenario_name`, `config_path`, `seed`, `profile_variant`。
- 网络结构：`network_type`, `bus_count`, `line_count`, `lateral_line_count`, `vpp_count`,
  `single_pcc_vpp_count`, `multi_node_vpp_count`。
- 安全指标：`min_voltage_vm_pu`, `max_voltage_vm_pu`, `max_line_loading_percent`,
  `max_trafo_loading_percent`, `powerflow_fail_count`, `total_violation_cells`, `security_pass`。
- 机制指标：`storage_soc_span`, `pv_utilization_ratio`, `pcc_abs_energy_proxy_mwh`。
- 经济代理：`total_cost`, `operation_cost_sum`, `reward_sum`。
- 压力分级：`network_pressure_level`。

`aggregate_metrics.csv` 按 `algorithm x split x scenario x profile_variant` 汇总 mean/std/min/max。

## 6. 当前结论边界

当前结果可以称为 **research-grade candidate benchmark**：

- 已有多 seed。
- 已有 train/eval/topology holdout。
- 已有非重复多日 profile。
- 已有临界可行网络压力。
- 已有完整 CSV/HTML 报告产物。

但仍不能声称是论文最终 benchmark：

- profile 仍是合成数据，不是公开真实数据。
- `rule_based` 是控制基线，不是训练出的 MARL 策略。
- 经济收益仍是代理成本，不是完整市场结算利润。
- 缺少 OPF/oracle/full-information 上界。
- CTDE/MARL 训练还没有并入 benchmark runner 的 train-then-frozen-eval 协议。

## 7. 下一阶段

1. 将 `privacy_separated_ctde` trainer 接入 `BenchmarkExperimentConfig` 协议。
2. 增加 frozen eval：训练只看 train split，checkpoint 固定后跑 eval/topology holdout。
3. 增加 `privacy_audit.csv`，记录 actor-visible、critic-visible 和 privileged fields。
4. 将 DSO action 从单纯 `target_p_mw` 升级为 `DOE/FR envelope + award + risk margin`。
5. 将 VPP dispatch agent 的 DER-level action 作为主学习对象，而不是规则式解聚合。
6. 增加 settlement-aware `Profit_i`、DSO procurement cost、non-delivery penalty。
7. UI 报告联动更新，展示每次算法更新后的网络、训练、评估和机制变化。

## 8. V2.1 Update

当前代码已经进入 v2.1 协议：

- `privacy_separated_ctde_actor_critic` 已接入 benchmark runner。
- CTDE 运行方式为 `train_profile` 训练一次，然后对 `train/eval/safety` split 执行 frozen deterministic mean-policy eval。
- CTDE 暂不进入 IEEE33 `topology_holdout` 主表，因为当前 policy head 依赖 VPP layout；拓扑迁移需要 VPP slot adapter、graph policy 或兼容签名映射。
- 新增 `holdout_reverseflow` profile variant，用低负荷、高 PV 和 PV 容量放大测试反向潮流。
- 新增 `configs/european_lv_benchmark_v2_safety_tight.yaml`，使用 `[0.95, 1.05]` 电压约束、95% 线路/变压器限值、F1 负荷扰动和 F2 线路降额测试安全裕度。
- `reward.privacy_mode: privacy_preserving_proxy` 已进入 benchmark 配置，默认用上级电网购电代理成本训练/评估；VPP 私有运行成本保留为 reference 字段，不直接进入默认 reward。
- `seed_metrics.csv` 已补充 step-rate、projection、FR binding、service request、reverse-flow 和 privacy-audit 字段。
- 每个 run 目录新增 `step_summary.csv`，用于按 step 检查电压、线载率、投影、调度请求和成本。

v2.1 pilot 命令：

```powershell
python examples/11_run_benchmark_experiment.py `
  --output-dir outputs/benchmark_v21_pilot `
  --horizon-steps 48 `
  --seeds 5201 `
  --train-variants train_mixed `
  --eval-variants holdout_peak,holdout_cloudy,holdout_reverseflow `
  --algorithms rule_based,privacy_separated_ctde_actor_critic `
  --ctde-train-episodes 2 `
  --ctde-train-horizon-steps 48 `
  --ctde-eval-horizon-steps 48
```

Pilot 结果边界：

- `runs=11`
- `security_pass_rate=0.818`
- `safety_tight_limits` 对 rule-based 和 CTDE 都触发安全失败，说明它能测试安全裕度。
- `holdout_reverseflow` 已记录更强的反向潮流信号，例如 `min_line_p_from_mw` 更低。
- CTDE 短预算 pilot 的成本高于 rule-based，不能作为算法效果结论；它只证明 train-then-frozen-eval 闭环已经可运行。
