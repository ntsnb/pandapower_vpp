# 02 Algorithm Analysis

## 结论

当前可落地的 MARL 算法主线是 HAPPO、HATRPO、MATD3、HASAC，加上旧 CTDE actor-critic 和 heuristic baseline。MAPPO/QMIX 在仓库中主要作为 registry/baseline/候选概念出现，未发现完整神经 learner/mixer/q-network 训练实现。

总体训练范式是 CTDE：decentralized actors 使用本地/角色化观测，centralized critic 在训练期可见全局状态和 privileged 信息。后续 dashboard 必须明确区分 actor-visible 数据和 training-only critic 数据，尤其不能把 `critic_global_state` 当作 DSO 或 VPP 执行期可见信息展示。

## 证据

- `src/vpp_dso_sim/learning/agent_roles.py`：agent role 映射为 `dso_global_guidance`、`{vpp}_dispatch`、`{vpp}_portfolio`。
- `src/vpp_dso_sim/learning/ctde_interface.py`：`build_ctde_interface_contract()` 记录 `share_dispatch_policy`、`share_portfolio_policy`、`centralized_critic.visible_to_decentralized_actors=False`。
- `src/vpp_dso_sim/learning/ctde_networks.py`：`build_privacy_separated_ctde_modules()` 构造 `DSOActor`、`VPPDispatchActor`、`PortfolioActor`、`CentralizedActionConditionedCritic`。
- `src/vpp_dso_sim/learning/advanced_marl.py`：`train_happo()` 是 on-policy HAPPO 主训练；`train_hasac()` 是 off-policy HASAC 主训练。
- `src/vpp_dso_sim/learning/hatrpo.py`：`train_hatrpo()` 使用 trust-region policy update。
- `src/vpp_dso_sim/learning/matd3.py`：`train_matd3()` 使用 replay buffer、twin critic、target networks、delayed actor update。
- `src/vpp_dso_sim/learning/deep_rl.py`：旧 `train_privacy_separated_ctde_actor_critic()` 使用 GAE/clipped surrogate。
- `src/vpp_dso_sim/learning/marl_baselines.py`：IPPO/MAPPO/QMIX 主要作为 heuristic baseline；未发现 QMIX mixer learner。

## 相关文件路径

- `src/vpp_dso_sim/learning/agent_roles.py`
- `src/vpp_dso_sim/learning/ctde_interface.py`
- `src/vpp_dso_sim/learning/ctde_networks.py`
- `src/vpp_dso_sim/learning/advanced_marl.py`
- `src/vpp_dso_sim/learning/hatrpo.py`
- `src/vpp_dso_sim/learning/matd3.py`
- `src/vpp_dso_sim/learning/deep_rl.py`
- `src/vpp_dso_sim/learning/marl_baselines.py`
- `src/vpp_dso_sim/envs/observations.py`
- `tests/test_hasac_happo.py`
- `tests/test_hatrpo_training.py`
- `tests/test_matd3_training.py`
- `tests/test_structured_happo_training.py`

## 相关类/函数/变量

- `HAPPOConfig`
- `HASACConfig`
- `MATD3Config`
- `HATRPOConfig`
- `train_happo`
- `train_hasac`
- `train_matd3`
- `train_hatrpo`
- `evaluate_happo_checkpoint`
- `evaluate_hasac_checkpoint`
- `evaluate_matd3_checkpoint`
- `evaluate_hatrpo_checkpoint`
- `MATD3ReplayBuffer`
- `OffPolicyReplayBuffer`
- `CentralizedActionConditionedCritic`
- `MultiHeadValueCritic`
- `build_critic_global_state`
- `share_vpp_dispatch_parameters`
- `share_vpp_portfolio_parameters`
- `critic_use_action_summary`

## Algorithm Flow Map

```text
MultiAgentVPPDSOEnv
  -> actor observations by role
  -> action decoder / environment projection
  -> rewards by role
  -> algorithm-specific learner

HAPPO:
  rollout episode -> GAE/value targets -> critic update -> sequential actor role updates -> checkpoint

HATRPO:
  rollout episode -> GAE/value fit -> trust-region policy update -> checkpoint

MATD3:
  env step -> replay buffer -> twin critic update -> delayed actor update -> target soft update -> checkpoint

HASAC:
  env step -> replay buffer -> twin soft-Q update -> actor update -> alpha update -> target soft update -> checkpoint
```

## Policy / Agent / VPP 对齐

- `agent_id`：实际环境 agent，例如 `vpp_1_dispatch`。
- `vpp_id`：物理/商业聚合实体 ID，例如 `vpp_1`。
- `policy_id`：不一定等于 agent_id。共享策略时多个 VPP dispatch agent 可使用同一 dispatch actor。
- DSO 有全局 policy；每个 VPP 有 dispatch actor 和 portfolio actor，但具体是否参数共享取决于 config。

Dashboard schema 必须保留 `vpp_id`、`agent_id`、`policy_id` 三个字段，不能合并。

## Loss / Optimizer / Update 位置

HAPPO：

- actor loss：PPO clipped surrogate，内部函数 `_happo_role_loss()`。
- critic loss：`train_happo()` 中 value critic update。
- optimizer：critic optimizer、DSO actor optimizer、dispatch actor optimizer、portfolio actor optimizer。
- update metrics：`happo_update_metrics.csv`，包含 episode、epoch、role 等字段。
- `epoch` 在这里主要是 PPO update epoch，不是实验 campaign epoch。

HATRPO：

- value loss：critic/value fitting。
- policy update：`hatrpo_trust_region_update()`，关键指标为 KL、CG、line search。
- actor 不走普通 `optimizer.step()`，因此 logger 不能只监听 optimizer step。

MATD3：

- replay buffer：`MATD3ReplayBuffer`。
- critic loss：twin critic TD loss。
- actor loss：`-Q`。
- update counters：`critic_updates`、`actor_updates`。
- portfolio 通常固定为 keep，不训练 `{vpp}_portfolio`。

HASAC：

- replay buffer：`OffPolicyReplayBuffer`。
- critic loss：soft-Q target loss。
- actor loss：soft actor loss。
- entropy/alpha loss：自动 entropy tuning。
- update counters：`critic_updates`、`actor_updates`、`alpha_updates`。

旧 CTDE：

- 单 optimizer、GAE、clipped surrogate，适合 legacy 对照，不建议作为实时平台主契约。

## Buffer / Batch / Gradient Step

- HAPPO/HATRPO：on-policy episode rollout，batch 更接近“整条 episode/trajectory 的 tensor batch”。
- MATD3/HASAC：off-policy replay sample，batch 是优化器 minibatch。
- 原生统一 `gradient_step` 未全局存在；dashboard 应在 AlgorithmAdapter 中按 optimizer update 事件合成。
- `batch_id` 未作为跨算法统一字段存在；应 nullable 或由 adapter 生成。
- `global_env_step` 可由 `episode * horizon_steps + step` 合成，但 shared rollout/subprocess 下必须包含 `worker_id/env_id` 避免冲突。

## Logger Hook 建议

安全 hook：

- 每次 role update 后记录 `role`、`vpp_id`、`agent_id`、`policy_id`、`loss`、`entropy`、`approx_kl`、`ratio_mean`、`grad_norm`。
- 每次 replay update 后记录 `replay_size`、`batch_size`、`critic_loss`、`actor_loss`、`q1_q2_gap`、`target_q`。
- 每次 shared rollout fragment 后记录 `worker_id`、`start_step`、`fragment_steps`、`policy_version`、`samples_per_second`。
- `paper_training.py` 的 `_write_tensorboard_scalars()` 是低侵入统一导出点，但不能作为唯一实时源。

禁止 hook：

- 不包裹或改写 actor/critic `forward()`。
- 不改变 optimizer 调用顺序。
- 不改变 replay/rollout buffer 中的数据结构。
- 不把 dashboard logger 异常向上传播导致训练失败。

## 风险

- High：centralized critic 使用 privileged/global state；前端必须标记 training-only，不能误作 actor 可见。
- High：`policy_id` 与 `agent_id` 不一致，尤其是共享 dispatch/portfolio actor；若合并字段会导致多 VPP 对比错误。
- Medium：HAPPO `epoch` 是 PPO epoch，和 dashboard 常见训练 epoch 概念冲突。
- Medium：MATD3/HASAC replay 存 raw action，而环境执行会经过 projection/shield；展示时必须区分 raw action 与 landed action。
- Medium：HATRPO 更新不一定对应 `optimizer.step()`；只监听 optimizer 会漏掉 policy update。
- Low：MAPPO/QMIX 名称存在但实现不完整；报告/平台应避免把它们列为已完成 learner。

## 建议

- 建立 `AlgorithmAdapterContract`，每个算法独立抽取 loss/update 字段。
- `gradient_step` 由 adapter 统一生成，并保留原算法 counters。
- loss long-table 化：`metric_group=loss`、`metric_name=actor_loss|critic_loss|entropy_loss|value_loss|q_loss|total_loss`。
- CTDE privacy metadata 作为 dashboard 一等字段：`visibility=actor_visible|critic_training_only|private`.
- 每条 learning metric 至少包含 `algorithm`、`role`、`agent_id`、`policy_id`、`episode_id`、`gradient_step`。

## 待用户确认项

- 是否允许 centralized critic 使用 VPP private cost，论文口径是否要称为 privileged critic。
- 长训练 HAPPO 实际 YAML 中 `ppo_epochs`、`critic_use_action_summary`、`share_vpp_dispatch_parameters` 的默认值是否视为正式协议。
- 是否计划补齐真正的 MAPPO learner 和 QMIX mixer；否则 dashboard 不应把 MAPPO/QMIX 作为可训练主算法展示。
- 对共享 policy 的多 VPP 对比，是否展示为同一 `policy_id` 下多个 `agent_id`。
