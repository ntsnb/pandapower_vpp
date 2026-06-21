# 01 Environment Analysis

## 结论

当前项目同时有单智能体 Gymnasium 包装和多智能体 parallel-env 形状包装。真实物理推进不在 env wrapper 内完成，而是在 `Simulator.step()` 内完成。多 VPP 在同一个仿真场景内作为多个实体存在，多智能体层把 DSO、每个 VPP dispatch、每个 VPP portfolio 拆成 agent。

环境接口接近 Gymnasium/PettingZoo parallel：

- `VPPDSOEnv.reset()` 返回 `(obs, info)`；`step()` 返回 `(obs, reward, terminated, truncated, info)`。
- `MultiAgentVPPDSOEnv.reset()` 返回 `(observations, infos)`；`step()` 返回 `(observations, rewards, terminations, truncations, infos)`。

未发现真实 `date`/`timestamp` 推进。能源时间目前是整数 `time_index`/`step` 和 `time_hours = step * dt_hours`。后续 dashboard 的 `date`、`timestamp` 字段应允许为空，或由 profile metadata 显式提供。

## 证据

- `src/vpp_dso_sim/simulation/scenario.py`：`SimulationScenario` 包含 `net`、`dso`、`vpps`、`load_profile`、`pv_profile`、`price_profile`、`horizon_steps`、`dt_hours`。
- `src/vpp_dso_sim/simulation/scenario.py`：`load_scenario()` 默认 `horizon_steps=288`、`dt_hours=0.25`，按 YAML 构造网络、DSO、VPP、DER、reward config。
- `src/vpp_dso_sim/envs/gym_env.py`：`VPPDSOEnv(gym.Env)`，`action_space=Box(-1,1, shape=(n_vpps,))`。
- `src/vpp_dso_sim/envs/gym_env.py`：`reset()` 重建 scenario/simulator，返回 `obs, {"step": 0}`。
- `src/vpp_dso_sim/envs/gym_env.py`：`step()` 把 action 作为每个 VPP 当前功率的 MW delta，执行 `simulator.step()`，返回 `terminated=False`、`truncated=current_step>=horizon_steps`。
- `src/vpp_dso_sim/envs/multi_agent_env.py`：`MultiAgentVPPDSOEnv` 明确是 PettingZoo parallel 形状但无硬依赖。
- `src/vpp_dso_sim/envs/multi_agent_env.py`：agent ids 为 `dso_global_guidance`、`{vpp.id}_dispatch`、`{vpp.id}_portfolio`。
- `src/vpp_dso_sim/simulation/simulator.py`：`Simulator.step()` 完成 profile 读取、FR/DOE envelope、动作解码、AC 修复、pandapower 潮流、状态更新、settlement、reward/cost、records。
- `src/vpp_dso_sim/simulation/simulator.py`：`_time_label()` 返回小时标签；`_record_step()` 写 `time_hours`。

## 相关文件路径

- `src/vpp_dso_sim/envs/gym_env.py`
- `src/vpp_dso_sim/envs/multi_agent_env.py`
- `src/vpp_dso_sim/envs/observations.py`
- `src/vpp_dso_sim/simulation/scenario.py`
- `src/vpp_dso_sim/simulation/simulator.py`
- `src/vpp_dso_sim/entities/dso.py`
- `src/vpp_dso_sim/entities/vpp.py`
- `src/vpp_dso_sim/optimization/feasibility_region.py`
- `src/vpp_dso_sim/optimization/ac_security_projection.py`
- `src/vpp_dso_sim/simulation/settlement.py`
- `src/vpp_dso_sim/envs/reward_design.py`

## 相关类/函数/变量

- `VPPDSOEnv`
- `MultiAgentVPPDSOEnv`
- `SimulationScenario`
- `load_scenario`
- `Simulator`
- `Simulator.reset`
- `Simulator.step`
- `Simulator.records`
- `DSO.calculate_reward_or_cost`
- `build_actor_observation`
- `build_critic_global_state`
- `build_role_reward_maps`
- `dso_global_guidance`
- `{vpp_id}_dispatch`
- `{vpp_id}_portfolio`
- `current_step`
- `horizon_steps`
- `dt_hours`
- `time_index`
- `time_hours`

## Environment Flow Map

```text
YAML config
  -> load_yaml / load_scenario
  -> SimulationScenario(net, dso, vpps, profiles, reward_config)
  -> Simulator(scenario)
  -> VPPDSOEnv or MultiAgentVPPDSOEnv wrapper
  -> reset()
  -> step(action/actions)
       profile[t]
       pre-dispatch powerflow
       DSO FR/DOE envelope
       action decoder
       feasibility projection
       AC security repair
       DER dispatch and dynamic state update
       settlement audit
       reward/cost calculation
       records append
       env return tuple
```

## reset/step 语义

单智能体 `VPPDSOEnv`：

- `reset(seed=None, options=None)`：重建 scenario 和 simulator，`current_step=0`。
- `step(action)`：action shape 是 `(n_vpps,)`，但语义是 MW delta，不是标准归一化 setpoint。
- 返回 reward 是 `result["reward_components"]["reward"]`，供旧 Gym 兼容。
- `terminated` 固定为 `False`，episode 结束通过 `truncated=True` 表示。

多智能体 `MultiAgentVPPDSOEnv`：

- `reset(seed=None, start_step=0)`：调用 `simulator.reset()`，`current_step = start_step % horizon`。
- `step(actions)`：按 role 校验/解码动作，执行 simulator，生成下一步 observations。
- `terminations` 当前固定 `False`，`truncations` 在 `current_step >= horizon_steps` 时全 agent 为 `True`。
- `infos` 包含 decoded action、action validation、critic global state、reward components、agent reward components、violations。

## Observation / Action / Reward / Cost / Done / Info

- DSO observation：网络状态、VPP reports、time_index。
- VPP dispatch observation：本 VPP portfolio、bounds、current_power、local_assets、operating_envelope、service_signal、dispatch_award。
- VPP portfolio observation：慢循环 portfolio 选择，物理 membership 改变仍受 `portfolio_events` gate 控制。
- DSO action：目标或 envelope guidance。
- Dispatch action：`selected_p_mw`/`target_p_mw` 绝对 MW，`normalized_setpoint_bias`/`der_actions` 为归一化局部参数。
- Reward/cost：DSO 系统 cost/reward 在 `DSO.calculate_reward_or_cost()`；多 agent role reward 在 `build_role_reward_maps()`。
- Done：真实故障/严重违规目前进入 cost/info，不直接触发 `terminated=True`。

## VPP 与时间维度

- 一个 VPP 不是一个单一 agent，而是至少对应 `{vpp_id}_dispatch` 和 `{vpp_id}_portfolio` 两个 agent；另有一个全局 `dso_global_guidance` agent。
- 多 VPP 是同一个 env 内的多个实体，不是多个独立 env。
- `time_index` 与 env step 一一对应；并行 shared rollout 时不同 worker 可能从不同 `start_step` 片段运行。
- 当前 `paper_long_sensitivity_v1` 是 `horizon_steps=672`，`dt_hours=0.25`，即 168 小时/7 天。未发现“5 周数据切分为 episode”的直接实现。

## 单位与物理意义

已能从变量名和代码还原：

- `p_mw`：MW，内部正号表示向电网注入。
- `q_mvar`：Mvar。
- `vm_pu`：标幺电压。
- `loading_percent`：线路/变压器负载率百分比。
- `soc`、`average_soc`：0 到 1。
- `dt_hours`：小时。
- `time_hours`：小时。
- `price`：代码按 currency/MWh 使用，但币种和单位未在 schema 中声明。
- `load_profile`：负荷倍率，不是 MW。
- `pv_profile`：PV 可用系数，0 到 1。

## 安全 Hook 点

- `MultiAgentVPPDSOEnv.step()` 返回后读取 `infos`，不改变 action 或 reward。
- `Simulator.records` 派生 frame，适合读取 grid state、projection_trace、settlement、reward_components。
- `reward_components` 和 `agent_reward_components` 只读记录，禁止在 dashboard adapter 中重算并覆盖。
- action 展示应读 decoded/action_validation/projection_trace，不应由前端重新投影。

## 风险

- High：修改 `reset/step` 语义会破坏训练器、测试和 checkpoint eval。
- High：单智能体 action 标称 `[-1,1]` 但实际是 MW delta；dashboard 若统一解释为 normalized action 会误导。
- Medium：`terminated` 固定为 False，严重安全失败不会作为 episode termination；dashboard 应区分 violation/cost 与 terminal。
- Medium：到 horizon 后没有硬 guard，若外部继续调用 step，会继续按 profile modulo 或 records 推进。
- Medium：无真实 timestamp，用户若要求按日历日期展示，需要新增 metadata，不应凭空推断。

## 建议

- 后续平台以 `MultiAgentVPPDSOEnv` 为主口径，`VPPDSOEnv` 标记为兼容旧 Gym 的单智能体入口。
- 环境 adapter 只读取 `observations/actions/rewards/infos` 和 `simulator.records`。
- `date`、`timestamp` 采用 nullable 字段；只有 profile metadata 有真实起始日期时才填充。
- 所有 action 展示必须带 `action_semantics`：`mw_delta`、`absolute_mw`、`normalized_bias`、`normalized_der_action`。
- 前端优先展示 `Simulator.records` 中的物理量，模型输入中的 normalized/encoded feature 必须标记 `normalized=true`。

## 待用户确认项

- 是否以后只支持 `MultiAgentVPPDSOEnv` 作为 dashboard 主环境。
- 是否需要继续展示旧 `VPPDSOEnv` 的单智能体 MW-delta action。
- 严重 powerflow failure 是否仍只作为 cost/violation，还是未来要终止 episode。
- 真实日期/5 周数据是否存在于外部数据源；若存在，需要指定 metadata 来源。
- reward/cost 中所有货币量是否统一为每步能量成本，即全部乘以 `dt_hours`。
