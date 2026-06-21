# 123-bus VPP 可行域偏负与区域负荷影响验证报告

日期：2026-06-08

## 1. 先澄清：当前不是 IEEE PES 123 原版

仓库中当前 123 节点网络来自：

```text
src/vpp_dso_sim/network/european_lv.py
configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml
```

代码注释明确说明它是：

```text
European-LV-style 123-bus transformer-area test feeder
not a verbatim copy of an IEEE or CIGRE benchmark
```

因此，严格说当前环境是“项目自建 European-LV-style 123-bus benchmark”，不是 IEEE PES 123-node feeder 原版。

## 2. 当前 123-bus 场景是否设置了区域负荷

实验加载结果：

```text
network_name: european_lv_123_branched_benchmark
bus_count: 123
load_count: 133
base_load_count: 121
base_load_base_p_mw: 0.5576
initial_total_load_p_mw: 0.9086
vpp_count: 7
horizon_steps: 288
dt_hours: 0.25
```

结论：

```text
当前不是“没有设置区域负荷”。
项目网络在 121 个下游 service bus 上设置了 base load。
另外还存在 VPP 内部 flexible load、HVAC、EVCS 等可控负荷类 DER。
```

## 3. 实验设计

为了验证“VPP 可行域偏负是不是区域负荷缺失导致”，我做了两个层次的实验。

### 3.1 负荷倍率扰动

同一 123-bus 场景，完整 288 step，测试：

```text
load_0p0
load_0p5
load_1p0
load_1p5
load_3p0
```

每个 variant 统计：

```text
FR p_min / p_max / midpoint
是否全负
是否跨 0
注入 headroom
吸收 headroom
DSO preferred target
grid pressure mode
service request
AC-aware 是否启用
```

输出目录：

```text
outputs/ieee123_vpp_feasible_region_load_effect_20260608
```

### 3.2 DER 类型贡献分解

按 DER 类型统计每类资源对 FR 的贡献：

```text
PV
Storage
EVCS
FlexibleLoad
HVAC
MicroTurbine
```

输出文件：

```text
ieee123_der_type_fr_contributions_summary.csv
```

## 4. 关键实验结果

### 4.1 硬 FR 的负向偏置不随区域负荷改变

负荷倍率从 0 到 3 倍时，硬 FR 统计如下：

| variant | midpoint_negative_rate | all_negative_rate | crosses_zero_rate | mean_injection_headroom_mw | mean_absorption_headroom_mw |
|---|---:|---:|---:|---:|---:|
| load_0p0 | 0.893353 | 0.0 | 1.0 | 0.105415 | 0.077000 |
| load_0p5 | 0.893353 | 0.0 | 1.0 | 0.105415 | 0.077000 |
| load_1p0 | 0.893353 | 0.0 | 1.0 | 0.105415 | 0.077000 |
| load_1p5 | 0.893353 | 0.0 | 1.0 | 0.105415 | 0.077000 |
| load_3p0 | 0.893353 | 0.0 | 1.0 | 0.105415 | 0.077000 |

结论：

```text
硬 FR midpoint 确实经常偏负。
但硬 FR 不是全负，所有统计都 crosses_zero_rate = 1.0。
平均注入 headroom 还大于平均吸收 headroom。
所以硬 FR 并不是“抗拒注入”或“不允许注入”。
```

最关键的是：

```text
把区域 base load 从 0 倍改到 3 倍，FR 硬边界完全不变。
```

原因是当前 `compute_static_feasible_region()` 只根据 VPP 内部 DER bounds 构造本地 FR，不读取 pandapower base load。

### 4.2 DSO preferred target 会受区域负荷影响

| variant | preferred_target_negative_rate | ac_aware_enabled_rate | grid_priority_rate |
|---|---:|---:|---:|
| load_0p0 | 0.972222 | 0.000000 | 0.000000 |
| load_0p5 | 0.972222 | 0.000000 | 0.000000 |
| load_1p0 | 0.972222 | 0.000000 | 0.000000 |
| load_1p5 | 0.913194 | 0.079861 | 0.079861 |
| load_3p0 | 0.000000 | 1.000000 | 1.000000 |

解释：

```text
低/正常负荷时，网络没有低电压压力，DSO 主要走 price_economic 逻辑。
在当前价格曲线下，很多时段触发 absorb_or_charge，因此 preferred target 偏负。

高负荷 load_3p0 时，低电压压力触发 low_voltage_support。
DSO preferred target 全部转为正向 export_or_reduce_load。
```

所以：

```text
区域负荷不会改变 VPP 本地 FR 硬边界。
但区域负荷会改变 DSO 对 VPP 的方向性需求。
```

如果区域负荷设置过低，DSO 更少触发“需要注入支撑电压”的物理需求，于是经济规则更容易让 target 偏向吸收。

### 4.3 service_request 分布

| variant | absorb_or_charge | balanced_operation | export_or_reduce_load |
|---|---:|---:|---:|
| load_0p0 | 0.552 | 0.427 | 0.021 |
| load_0p5 | 0.552 | 0.427 | 0.021 |
| load_1p0 | 0.552 | 0.427 | 0.021 |
| load_1p5 | 0.552 | 0.368 | 0.080 |
| load_3p0 | 0.000 | 0.000 | 1.000 |

这进一步说明：

```text
正常负荷下 preferred target 偏负，不是因为 AC 安全外壳压制注入。
而是因为 DSO rule/economic guidance 多数时段给出 absorb_or_charge 或 balanced_operation。
```

## 5. 为什么硬 FR midpoint 偏负

DER 类型贡献均值：

| DER 类型 | mean_p_min_mw | mean_p_max_mw | mean_midpoint_mw |
|---|---:|---:|---:|
| EVCSModel | -0.060000 | 0.000000 | -0.030000 |
| FlexibleLoadModel | -0.062167 | -0.015500 | -0.038833 |
| HVACModel | -0.026000 | 0.000000 | -0.013000 |
| MicroTurbineModel | 0.012500 | 0.040000 | 0.026250 |
| PVModel | 0.000000 | 0.012272 | 0.006136 |
| StorageModel | -0.041857 | 0.041857 | 0.000000 |

解释：

```text
PV 和微燃机提供正向注入能力。
储能大致对称，midpoint 接近 0。
EVCS、HVAC、FlexibleLoad 是吸收型/负荷型资源，midpoint 天然为负。
```

因此硬 FR midpoint 偏负的主要原因是：

```text
VPP 内部可控资源组合里，吸收型负荷资源占比较高。
```

而不是：

```text
没有区域负荷。
AC-aware DOE 强行抗拒注入。
FR 本身不允许注入。
```

## 6. 最终结论

先给一个严格判定表：

| 待验证命题 | 判定 | 证据 |
|---|---|---|
| 当前 123-bus 场景没有区域/base load | 否 | `base_load_count=121`，`load_count=133` |
| VPP 硬 FR 的 midpoint 经常偏负 | 是 | `midpoint_negative_rate=0.893353` |
| VPP 硬 FR 全部为负、不允许注入 | 否 | `all_negative_rate=0.0`，`crosses_zero_rate=1.0` |
| VPP 硬 FR 偏负由区域负荷倍率直接造成 | 否 | 负荷从 0 倍到 3 倍，硬 FR 统计不变 |
| DSO preferred target 会受区域负荷影响 | 是 | `load_3p0` 下 `preferred_target_negative_rate=0.0` |
| 正常负荷下 DSO 更常给吸收/充电目标 | 是 | `absorb_or_charge=0.552`，`preferred_target_negative_rate=0.972222` |

### 问题 1：当前 123-bus pandapower 环境对 VPP 的可行域本就偏负吗？

是，但要精确定义。

```text
FR midpoint 偏负：是。
FR 全区间为负：不是。
FR 抗拒注入：证据不足，甚至相反。
```

因为：

```text
all_negative_rate = 0.0
crosses_zero_rate = 1.0
mean_injection_headroom_mw = 0.105415
mean_absorption_headroom_mw = 0.077000
```

### 问题 2：这是因为没有设置区域负荷吗？

不是。

当前 123-bus 场景已经有：

```text
121 个 base load
base_load_base_p_mw = 0.5576
initial_total_load_p_mw = 0.9086
```

并且实验显示：

```text
区域负荷从 0 倍到 3 倍，FR 硬边界完全不变。
```

### 问题 3：区域负荷是否会影响 DSO 给 VPP 的调度方向？

会。

```text
低/正常负荷：DSO 多数走 price_economic，preferred target 偏负。
高负荷：DSO 触发 low_voltage_support，preferred target 转正。
```

因此如果你的论文叙述里说“DSO 因台区负荷不足而不需要 VPP 注入”，这是合理的。

但如果说“VPP FR 因没有区域负荷而本身抗拒注入”，代码和实验不支持。

## 7. 建议

1. 后续论文和代码注释里应严格区分：
   - VPP local feasible region
   - DSO operating envelope
   - DSO preferred target
   - AC-aware DOE
2. 如果你希望 VPP FR midpoint 不那么偏负，需要调整 VPP DER 组合：
   - 增加 PV / microturbine / discharge-capable storage 比例。
   - 降低 EVCS / flexible load / HVAC 在某些 VPP 中的吸收侧权重。
3. 如果你希望 DSO 更频繁要求注入，需要调整电网场景：
   - 增加区域负荷。
   - 降低电压裕度。
   - 增强远端 feeder 负载压力。
   - 增加 holdout_peak / low-voltage scenario。
4. 当前 123-bus 网络不是 IEEE PES 123 原版。若论文要声称 IEEE123，应接入真正 IEEE PES 123-node feeder 或明确写成 European-LV-style 123-bus benchmark。

## 8. 可复现实验证据文件

本报告使用的证据都来自当前工作区下列文件：

```text
outputs/ieee123_vpp_feasible_region_load_effect_20260608/ieee123_load_inventory.csv
outputs/ieee123_vpp_feasible_region_load_effect_20260608/ieee123_vpp_fr_load_effect_summary.csv
outputs/ieee123_vpp_feasible_region_load_effect_20260608/ieee123_vpp_fr_load_effect_detail.csv
outputs/ieee123_vpp_feasible_region_load_effect_20260608/ieee123_grid_load_effect_summary.csv
outputs/ieee123_vpp_feasible_region_load_effect_20260608/ieee123_der_type_fr_contributions_summary.csv
outputs/ieee123_vpp_feasible_region_load_effect_20260608/ieee123_der_type_fr_contributions_detail.csv
```

其中最关键的三类证据是：

1. `ieee123_load_inventory.csv`

   证明当前场景不是无负荷场景：

   ```text
   bus_count = 123
   load_count = 133
   base_load_count = 121
   base_load_base_p_mw = 0.5576
   initial_total_load_p_mw = 0.9086
   ```

2. `ieee123_vpp_fr_load_effect_detail.csv`

   证明硬 FR 与 base-load 倍率无直接关系：

   ```text
   load_0p0 midpoint_negative_rate = 0.893353
   load_0p5 midpoint_negative_rate = 0.893353
   load_1p0 midpoint_negative_rate = 0.893353
   load_1p5 midpoint_negative_rate = 0.893353
   load_3p0 midpoint_negative_rate = 0.893353
   ```

3. `ieee123_der_type_fr_contributions_summary.csv`

   证明 FR midpoint 偏负主要来自 DER 组合：

   ```text
   EVCSModel midpoint < 0
   FlexibleLoadModel midpoint < 0
   HVACModel midpoint < 0
   StorageModel midpoint ≈ 0
   PVModel midpoint > 0
   MicroTurbineModel midpoint > 0
   ```

## 9. 对后续建模的直接影响

如果你的研究问题是“VPP 是否能向台区注入有功以缓解低电压”，那当前场景不是完全不可用，但需要注意：

```text
硬 FR：允许注入，因为每个负荷倍率下 crosses_zero_rate 都是 1.0。
DSO target：正常负荷下较少要求注入，因为网络低电压压力不强。
训练 reward：如果只看 preferred target，智能体会大量学到吸收/充电方向。
```

因此后续 paper-long 实验建议分两条线跑：

1. **DER 组合线**

   目标是验证 FR midpoint 偏负是否影响学习。

   调整方式：

   ```text
   增加 PV / microturbine / storage discharge 能力；
   或降低 EVCS / HVAC / flexible load 的吸收侧容量；
   然后比较 midpoint_negative_rate、injection_headroom、reward、post-AC violation。
   ```

2. **配电网压力线**

   目标是验证 DSO 是否真正需要注入。

   调整方式：

   ```text
   增加 base load；
   增加远端 feeder 负荷；
   设置 holdout_peak / low_voltage_support 场景；
   然后比较 service_request 中 export_or_reduce_load 的比例。
   ```

这两条线不要混在一起。否则会把“VPP 能不能注入”和“DSO 此时需不需要注入”误判成同一个问题。
