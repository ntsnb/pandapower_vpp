# DSO sensitivity_attention_v1 架构说明

Updated: 2026-05-28 Asia/Shanghai

> prompt 原建议路径为 `docs/architecture/dso_sensitivity_attention_plan.md`。
> 但本仓库已有 `docs/architecture.md` 文件，文件系统不能同时存在同名目录。
> 因此本轮采用 `docs/dso_sensitivity_attention_architecture_plan.md`。

## 1. 改造目标

将旧的固定比例 DSO preferred range 生成逻辑，升级为：

```text
ActionUnit x NetworkObject sensitivity tensor
  -> structured DSO observation
  -> edge-biased bipartite attention actor
  -> safe decoder
  -> operating envelope guidance
  -> existing FR/DOE projection and AC replay safety checks
```

## 2. 保留的 baseline

- `rule_v0`：旧规则 envelope，仍调用原 simulator 逻辑。
- `legacy_flat`：旧 DSO flat observation，3 VPP 时保持 26 维。
- legacy DSO MLP actor：现有 HAPPO/HATRPO 等 trainer 路径未删除。

## 3. 新增核心模块

| 模块 | 文件 | 作用 |
|---|---|---|
| schema | `src/vpp_dso_sim/dso/envelope/schemas.py` | ActionUnit、NetworkObject、StructuredDSOObservation、DecodedOperatingEnvelopeRecord |
| selector | `src/vpp_dso_sim/dso/sensitivity/selectors.py` | 构造 VPP-PCC / VPP-bus / DER ActionUnit，选择 critical NetworkObject |
| sensitivity | `src/vpp_dso_sim/dso/sensitivity/finite_difference.py` | pandapower finite-difference AC sensitivity tensor |
| cache | `src/vpp_dso_sim/dso/sensitivity/cache.py` | raw cache、active slice、refresh decision、priority partial merge |
| observation | `src/vpp_dso_sim/dso/observation/structured_bipartite.py` | structured DSO actor observation |
| model | `src/vpp_dso_sim/dso/models/bipartite_attention_actor.py` | edge-biased bipartite attention actor |
| decoder | `src/vpp_dso_sim/dso/envelope/safe_decoder.py` | center/width 输出映射到 FR/DOE hard bounds |
| policy | `src/vpp_dso_sim/dso/envelope/sensitivity_attention_v1.py` | config 路由和 simulator build adapter |

## 4. 安全边界

神经网络只输出：

```text
center_ratio
width_ratio
direction_probs
guidance_strength
```

它不输出：

```text
p_hard_min_mw
p_hard_max_mw
正式市场 award
正式结算结果
```

`safe_decoder` 保证：

```text
P_hard_min <= P_pref_lo <= P_pref_target <= P_pref_hi <= P_hard_max
```

最终动作仍走原有：

```text
device bounds -> FR/DOE projection -> AC-aware projection/repair -> pandapower runpp
```

## 5. 当前边界

- 当前已完成 rule baseline smoke、`sensitivity_attention_v1` smoke、256-step BC warm-start sanity、structured HAPPO 最小训练链路和 structured frozen evaluation。
- HAPPO 路径已能使用 structured DSO actor，并记录 `policy_loss`、`entropy_mean`、`approx_kl`、`grad_norm`、seed 和 config hash。
- Runtime `SensitivityAttentionEnvelopePolicy` 可加载 direct 或 structured-HAPPO attention actor checkpoint。
- Sensitivity cache 已支持 update period、TTL、电压/loading 漂移、FR width 变化、projection-gap 历史、缺失对象触发、priority ActionUnit 和 partial priority merge。
- 有限差分扰动分配规则当前为 `equal_pp_element_refs`，并在 metadata / envelope CSV 中记录 `sensitivity_allocation_weights`。
- 当前短训练是 behavior cloning warm-start sanity，不是 paper-long 策略收敛证明。
