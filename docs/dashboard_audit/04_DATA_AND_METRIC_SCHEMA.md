# 04 Data And Metric Schema

## 结论

当前数据来源包括 demo CSV、synthetic benchmark profile pack、本地 SMART-DS profile proxy。核心 profile 统一物化为只有 `value` 列的 CSV；`load_profile` 是负荷倍率，`pv_profile` 是 PV 可用系数，`price_profile` 按 currency/MWh 使用但币种未声明。

前端展示应优先使用 `Simulator.records`、settlement audit、dashboard frames 中的反归一化物理量。模型输入中的 normalized/encoded features 只能作为训练诊断展示，必须明确标记 `normalized` 或 `encoded`。

未发现 wind profile/DER schema。未发现真实 `date/timestamp`。未发现“5 周数据集切分为 episode”的代码；当前 paper-long family 是 672 steps，即 7 天。

## 证据

- `src/vpp_dso_sim/simulation/profiles.py`：`load_profile_csv()` 只要求 `value` 列，短于 horizon 时重复补齐。
- `data/profiles/load_profile.csv`、`pv_profile.csv`、`price_profile.csv`：有 `step,value` 两列，但 loader 只使用 `value`。
- `src/vpp_dso_sim/simulation/profiles.py`：`benchmark_profile_pack()` 生成 synthetic load/pv/price。
- `src/vpp_dso_sim/simulation/profiles.py`：`smart_ds_austin_profile_pack()` 从本地 SMART-DS Austin profile 抽样，price 是 derived scarcity proxy。
- `src/vpp_dso_sim/experiments/paper_training.py`：`_write_profile_config()` 将 profile pack 写成 run-local CSV。
- `src/vpp_dso_sim/network/powerflow.py`：`scale_base_loads(net, load_scale)` 表明 load profile 是倍率。
- `src/vpp_dso_sim/der/pv.py`：PV 可用功率由 `p_max_mw * pv_factor` 得到。
- `src/vpp_dso_sim/der/storage.py`：SOC 按 `p_mw`、`dt_hours`、效率、容量更新。
- `src/vpp_dso_sim/simulation/settlement.py`：能量为 `max(0, +/-p_mw) * dt_hours`。
- `src/vpp_dso_sim/envs/reward_design.py`：VPP dispatch reward 中 `energy_market_revenue = price * delivered_p_mw * dt_hours`。

## 相关文件路径

- `data/profiles/load_profile.csv`
- `data/profiles/pv_profile.csv`
- `data/profiles/price_profile.csv`
- `data/external/raw/smart_ds/`
- `src/vpp_dso_sim/simulation/profiles.py`
- `src/vpp_dso_sim/simulation/scenario.py`
- `src/vpp_dso_sim/simulation/simulator.py`
- `src/vpp_dso_sim/simulation/settlement.py`
- `src/vpp_dso_sim/network/powerflow.py`
- `src/vpp_dso_sim/der/base.py`
- `src/vpp_dso_sim/der/pv.py`
- `src/vpp_dso_sim/der/storage.py`
- `src/vpp_dso_sim/envs/reward_design.py`
- `src/vpp_dso_sim/entities/dso.py`
- `src/vpp_dso_sim/visualization/dashboard_data.py`

## 相关类/函数/变量

- `load_profile_csv`
- `default_load_profile`
- `default_pv_profile`
- `default_price_profile`
- `benchmark_profile_pack`
- `smart_ds_austin_profile_pack`
- `profile_quality_summary`
- `_write_profile_config`
- `scale_base_loads`
- `PVModel.get_bounds`
- `StorageModel.update_soc`
- `DERBase.operating_cost`
- `build_settlement_audit`
- `vpp_dispatch_reward_components`
- `vpp_portfolio_reward_components`
- `DSO.calculate_reward_or_cost`
- `build_dashboard_frames`
- `export_dashboard_frames`

## 数据集与时间粒度

| 数据源 | 格式 | 时间粒度 | VPP 维度 | date/timestamp | 备注 |
|---|---|---|---|---|---|
| Demo profiles | CSV `value` | 由 `dt_hours` 解释，默认 0.25h | 无 | 无 | 短于 horizon 会重复 |
| Synthetic benchmark | in-memory pack -> CSV | `dt_hours`，默认 15min | 无 | 无 | profile metadata 表示 synthetic |
| SMART-DS proxy | 本地 CSV 抽样 -> pack -> CSV | 15min proxy | 无 | 未暴露 | price 是 derived scarcity proxy |
| Scenario YAML | YAML | 不直接是时间序列 | VPP/assets | 无 | 定义网络、VPP、DER、reward |
| Simulator records | dict/list frames | step/time_hours | 有 | 无 | 物理状态与结算主来源 |

## 单位和归一化

| 变量 | 当前含义 | 单位 | 是否归一化/倍率 |
|---|---|---|---|
| `p_mw` | 内部有功功率，正值注入 | MW | 物理量 |
| `q_mvar` | 无功功率 | Mvar | 物理量 |
| `vm_pu` | 母线电压 | p.u. | 标幺物理量 |
| `loading_percent` | 线路/变压器负载率 | % | 物理量 |
| `soc` | 电池/EV SOC | fraction | 0..1 |
| `time_index` | 仿真步 | step | 整数索引 |
| `time_hours` | step × dt | h | 物理时间偏移 |
| `load_profile` | base load multiplier | dimensionless | 倍率 |
| `pv_profile` | PV availability factor | dimensionless | 0..1 |
| `price_profile` | 电价 proxy | currency/MWh 待确认 | 未归一化 |
| actor encoded feature | 模型输入 | mixed | 多处显式缩放 |

## 公式可还原清单

- PV 可用功率：`available_p_mw = p_max_mw * pv_factor`。
- Storage SOC 放电：`soc -= p_mw * dt_hours / (eta_discharge * capacity_mwh)`，其中 `p_mw >= 0`。
- Storage SOC 充电：`soc += eta_charge * (-p_mw) * dt_hours / capacity_mwh`，其中 `p_mw < 0`。
- DER operating cost：`a*p^2 + b*abs(p) + c`。
- Energy inject：`max(0, p_mw) * dt_hours`。
- Energy absorb：`max(0, -p_mw) * dt_hours`。
- VPP dispatch market revenue：`price * delivered_p_mw * dt_hours`。
- Shield penalty：代码中存在 `5*gap + 10*gap^2` 形式。
- Constraint penalties：电压/线路/变压器/powerflow failure penalty 可从 `network/constraints.py` 还原。

## 公式待确认清单

- `price` 的币种和单位。
- DSO `operation_cost` 是否应乘以 `dt_hours`。
- `_network_loss_cost()` 使用 `market_price_profile[0]` 是否有意，而不是当前 step price。
- V2 dispatch 中吸收功率导致 signed revenue 为负，是否应前端显示为 import cost。
- comfort/SOC penalty 的单位与业务解释。
- reward v2/v3 的每个 weight 是否作为论文正式公式。

## Proposed Metric Schema

统一 long table 字段必须包含：

```text
run_id
epoch_id
episode_id
batch_id
gradient_step
global_env_step
env_id
vpp_id
agent_id
policy_id
date
time_index
timestamp
metric_group
metric_name
value
unit
formula_latex
description
```

推荐扩展表：

- `profile_timeseries`
- `grid_step_state`
- `asset_dispatch_step`
- `envelope_projection_step`
- `reward_cost_component_long`
- `settlement_step`
- `learning_metrics_long`
- `schema_metadata`
- `variable_dictionary`

## Variable Dictionary 初稿

| name | display_name | unit | group | source |
|---|---|---|---|---|
| `p_mw` | 有功功率 | MW | physical_power | DER/Simulator |
| `q_mvar` | 无功功率 | Mvar | physical_power | DER/Simulator |
| `vm_pu` | 母线电压 | p.u. | grid | pandapower result |
| `loading_percent` | 线路/变压器负载率 | % | grid | pandapower result |
| `soc` | 荷电状态 | fraction | storage/ev | DER state |
| `price` | 电价 | currency/MWh? | market | profile |
| `load_scale` | 负荷倍率 | dimensionless | profile | profile |
| `pv_factor` | PV 可用系数 | dimensionless | profile | profile |
| `reward` | 奖励 | scalar | learning | reward functions |
| `total_cost` | 总成本 | scalar/currency? | objective | DSO reward |
| `actor_loss` | Actor loss | scalar | loss | learner |
| `critic_loss` | Critic loss | scalar | loss | learner |
| `entropy_loss` | Entropy/alpha loss | scalar | loss | learner |

## 风险

- High：无真实 date/timestamp，但用户目标显式包含 date/timestamp；需要 schema 支持 nullable 和 metadata 注入。
- High：price 单位未声明，经济图表可能被误读。
- Medium：load profile 是 multiplier，不是 MW；前端若标 MW 会错误。
- Medium：训练 actor 输入是 normalized/encoded，dashboard 直接展示会失去物理意义。
- Medium：wind 字段在用户目标中出现，但当前代码未发现 wind 支持。

## 建议

- 首轮 dashboard 只展示 raw physical values 和明确 labeled normalized features。
- 所有 metric 都带 `unit`、`source_function`、`formula_latex` 可空字段。
- 增加 `schema_metadata`，记录 config hash、profile hash、reward config hash、sign convention、price unit。
- 不在 dashboard adapter 中重算 reward；只记录代码已计算的 components。

## 待用户确认项

- 5 周数据是否存在；如果存在，数据路径和真实 timestamp schema 是什么。
- price 币种和单位。
- wind 是否属于未来需求而非当前仓库需求。
- reward/cost 公式是否以 v2/v3 config 作为正式论文公式。
- 前端是否需要展示 normalized actor features，还是只展示物理量。
