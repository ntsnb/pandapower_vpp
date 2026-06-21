from __future__ import annotations

import re
from typing import Any


COMMON_VARIABLES: dict[str, dict[str, Any]] = {
    "reward_so_far": {
        "display_name": "累计奖励 / Episode reward so far",
        "symbol": "R_{episode,sofar}",
        "unit": "score",
        "group": "reward",
        "physical_meaning": "当前 episode 内从 reset 到当前 time_index 已累计获得的奖励；它不是单步 reward。 / Cumulative reward accumulated so far in the active episode; this is not the one-step reward.",
        "formula_latex": "R_{t}^{sofar}=\\sum_{\\tau=0}^{t} r_{\\tau}",
        "notes": "用于观察训练进行中的回报趋势；episode 结束后应与该 episode 的最终累计回报一致或可解释。 / Progress metric for live training diagnostics.",
    },
    "global_env_step": {
        "display_name": "全局环境步 / Global environment step",
        "symbol": "k_{env}",
        "unit": "step",
        "group": "dataset",
        "physical_meaning": "训练运行从开始到当前记录累计推进的环境步数；它用于定位日志顺序，不是一天内时刻。 / Cumulative environment-step counter for locating log order; this is not the within-day time index.",
        "formula_latex": "k_{env}",
        "notes": "同一日内时刻请看 time_index；真实日历请看 date 或 timestamp。 / Use time_index for the within-day slot and date or timestamp for the calendar time.",
    },
    "electricity_price": {
        "display_name": "电价 / Electricity price",
        "symbol": "\\pi_{t}",
        "unit": "currency/MWh",
        "group": "dataset",
        "physical_meaning": "当前时刻用于结算或调度决策的电价。 / Electricity price used for settlement or dispatch decisions at the selected time.",
        "formula_latex": "\\pi_{t}",
        "notes": "单位必须结合数据源确认，例如 元/kWh、$/MWh 或 currency/MWh。 / Confirm the unit from the source dataset.",
    },
    "ev_charging_load": {
        "display_name": "充电桩负荷 / EV charging load",
        "symbol": "P^{EV}_{i,t}",
        "unit": "MW",
        "group": "dataset",
        "physical_meaning": "第 i 个 VPP 在 t 时刻的电动汽车充电需求功率。 / EV charging demand power for VPP i at time t.",
        "formula_latex": "P^{EV}_{i,t}",
    },
    "storage_power": {
        "display_name": "储能功率 / Storage power",
        "symbol": "P^{ESS}_{i,t}",
        "unit": "MW",
        "group": "dataset",
        "physical_meaning": "储能充放电功率；正值通常表示放电/向电网注入，负值表示充电/吸收。 / Storage charge-discharge power; positive usually means discharge/injection and negative means charging/absorption.",
        "formula_latex": "P^{ESS}_{i,t}",
    },
    "storage_soc": {
        "display_name": "储能荷电状态 / Storage SOC",
        "symbol": "SOC_{i,t}",
        "unit": "%",
        "group": "dataset",
        "physical_meaning": "储能系统在当前时刻的荷电状态。 / Battery state of charge at the selected time.",
        "formula_latex": "SOC_{i,t}",
        "min_value": 0,
        "max_value": 100,
    },
    "pv_power": {
        "display_name": "光伏出力 / PV power",
        "symbol": "P^{PV}_{i,t}",
        "unit": "MW",
        "group": "dataset",
        "physical_meaning": "第 i 个 VPP 在 t 时刻的光伏发电功率。 / Photovoltaic generation power for VPP i at time t.",
        "formula_latex": "P^{PV}_{i,t}",
        "min_value": 0,
    },
    "wind_power": {
        "display_name": "风电出力 / Wind power",
        "symbol": "P^{WT}_{i,t}",
        "unit": "MW",
        "group": "dataset",
        "physical_meaning": "第 i 个 VPP 在 t 时刻的风电发电功率。 / Wind generation power for VPP i at time t.",
        "formula_latex": "P^{WT}_{i,t}",
        "min_value": 0,
    },
    "base_load": {
        "display_name": "基础负荷 / Base load",
        "symbol": "P^{load}_{i,t}",
        "unit": "MW",
        "group": "dataset",
        "physical_meaning": "不含可再生出力抵扣的基础用电负荷。 / Base electrical demand before renewable offset.",
        "formula_latex": "P^{load}_{i,t}",
        "min_value": 0,
    },
    "net_load": {
        "display_name": "净负荷 / Net load",
        "symbol": "P^{net}_{i,t}",
        "unit": "MW",
        "group": "dataset",
        "physical_meaning": "综合基础负荷、充电负荷、风光出力和储能功率后的净负荷。 / Net demand after base load, EV load, renewable generation, and storage power.",
        "formula_latex": "P^{net}_{i,t}=P^{load}_{i,t}+P^{EV}_{i,t}-P^{PV}_{i,t}-P^{WT}_{i,t}-P^{ESS}_{i,t}",
    },
    "total_reward": {
        "display_name": "总奖励 / Total reward",
        "symbol": "r_t",
        "unit": "score",
        "group": "reward",
        "physical_meaning": "单步或聚合后的总训练奖励，数值越大通常表示策略表现越好。 / Total training reward; larger is generally better.",
    },
    "profit_reward": {
        "display_name": "收益奖励 / Profit reward",
        "symbol": "r^{profit}_t",
        "unit": "score",
        "group": "reward",
        "physical_meaning": "与市场收益、售电收入或运营利润相关的奖励项。 / Reward component related to market revenue or operating profit.",
    },
    "grid_balance_reward": {
        "display_name": "电网平衡奖励 / Grid balance reward",
        "symbol": "r^{grid}_t",
        "unit": "score",
        "group": "reward",
        "physical_meaning": "鼓励功率平衡、减少偏差或改善并网行为的奖励项。 / Reward for power balance, reduced deviation, or grid-friendly behavior.",
    },
    "dispatch_reward_train": {
        "display_name": "调度训练奖励 / Dispatch training reward",
        "symbol": "r^{train}_{dispatch,i,t}",
        "unit": "score",
        "group": "reward",
        "physical_meaning": "训练时传递给 VPP 调度智能体的 per-VPP reward。 / Per-VPP reward used by the dispatch learner during training.",
    },
    "energy_market_revenue": {
        "display_name": "电能市场收入 / Energy market revenue",
        "symbol": "\\pi_t P^{delivered}_{i,t}\\Delta t",
        "unit": "currency",
        "group": "reward",
        "physical_meaning": "VPP 在电能市场中因净交付功率获得的收入；若为负值通常表示购电成本。 / Market revenue from delivered power; negative values usually indicate import cost.",
        "formula_latex": "\\pi_t P^{delivered}_{i,t}\\Delta t",
    },
    "total_cost": {
        "display_name": "总成本 / Total cost",
        "symbol": "J_t",
        "unit": "currency",
        "group": "cost",
        "physical_meaning": "单步或聚合后的总运营成本，数值越小通常越好。 / Total operating cost; smaller is generally better.",
    },
    "energy_purchase_cost": {
        "display_name": "购电成本 / Energy purchase cost",
        "symbol": "C^{energy}_t",
        "unit": "currency",
        "group": "cost",
        "physical_meaning": "从电网或市场购入电能产生的成本。 / Cost of purchasing energy from the grid or market.",
    },
    "storage_degradation_cost": {
        "display_name": "储能退化成本 / Storage degradation cost",
        "symbol": "C^{deg}_t",
        "unit": "currency",
        "group": "cost",
        "physical_meaning": "储能充放电造成寿命损耗折算的成本。 / Cost proxy for battery degradation caused by cycling.",
    },
    "constraint_violation_cost": {
        "display_name": "约束违背成本 / Constraint violation cost",
        "symbol": "C^{viol}_t",
        "unit": "currency",
        "group": "cost",
        "physical_meaning": "违反功率、安全、SOC 或其他运行约束产生的惩罚性成本。 / Penalty cost for violating operating constraints.",
    },
    "actor_loss": {
        "display_name": "Actor 损失 / Actor loss",
        "symbol": "\\mathcal{L}_{actor}",
        "unit": "scalar",
        "group": "loss",
        "physical_meaning": "策略网络优化目标的损失值。 / Loss for optimizing the policy network.",
        "formula_latex": "\\mathcal{L}_{actor}",
    },
    "critic_loss": {
        "display_name": "Critic 损失 / Critic loss",
        "symbol": "\\mathcal{L}_{critic}",
        "unit": "scalar",
        "group": "loss",
        "physical_meaning": "价值函数或 Q 网络的拟合损失。 / Value-function or Q-network fitting loss.",
        "formula_latex": "\\mathcal{L}_{critic}",
    },
    "entropy_loss": {
        "display_name": "熵损失 / Entropy loss",
        "symbol": "\\mathcal{L}_{entropy}",
        "unit": "scalar",
        "group": "loss",
        "physical_meaning": "与策略熵正则或探索强度相关的 loss 项。 / Loss term associated with policy entropy regularization or exploration.",
    },
    "total_loss": {
        "display_name": "总损失 / Total loss",
        "symbol": "\\mathcal{L}",
        "unit": "scalar",
        "group": "loss",
        "physical_meaning": "一次 learner update 中各项 loss 的总和或主优化目标。 / Total or primary optimization loss for a learner update.",
    },
}


COMMON_VARIABLES.update(
    {
        "ac_projected_target_p_mw": {
            "display_name": "交流潮流校正目标功率 / AC-projected target power",
            "symbol": "P^{AC,target}_{i,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "经过 AC 安全约束或潮流投影后的 VPP 有功功率目标。 / VPP active-power target after AC-security or power-flow projection.",
        },
        "accepted_delta_p_mw": {
            "display_name": "接受调节功率 / Accepted power adjustment",
            "symbol": "\\Delta P^{accepted}_{i,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "VPP 实际接受的有功调节量。 / Active-power adjustment accepted by the VPP.",
        },
        "action_landing_ratio": {
            "display_name": "动作落地比例 / Action landing ratio",
            "symbol": "\\rho^{landing}_{i,t}",
            "unit": "ratio",
            "group": "dataset",
            "physical_meaning": "实际调节量相对策略解码调节量的比例，用于观察动作是否被约束削弱。 / Ratio between realized adjustment and decoded policy adjustment.",
        },
        "actual_delta_p_mw": {
            "display_name": "实际调节功率 / Actual power adjustment",
            "symbol": "\\Delta P^{actual}_{i,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "动作解码和约束投影后真正执行的 VPP 有功功率调节量。 / Realized VPP active-power adjustment after action decoding and projection.",
        },
        "actual_target_p_mw": {
            "display_name": "实际目标功率 / Actual target power",
            "symbol": "P^{actual,target}_{i,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "动作解码和约束投影后真正执行的 VPP 有功功率目标。 / Realized VPP active-power target after action decoding and projection.",
        },
        "baseline_p_mw": {
            "display_name": "基线功率 / Baseline power",
            "symbol": "P^{baseline}_{i,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "调度动作生效前的 VPP 基准有功功率。 / VPP baseline active power before applying the dispatch action.",
        },
        "decoded_target_p_mw": {
            "display_name": "解码目标功率 / Decoded target power",
            "symbol": "P^{decoded,target}_{i,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "策略动作从归一化空间解码后的 VPP 功率目标，尚未完全经过可行性或安全投影。 / VPP power target decoded from the normalized policy action before full feasibility or safety projection.",
        },
        "delivered_p_mw": {
            "display_name": "实际交付功率 / Delivered power",
            "symbol": "P^{delivered}_{i,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "调度轨迹中 VPP 实际交付的净有功功率；负值通常表示净购电或吸收。 / Actual net active power delivered by the VPP; negative values usually mean import or absorption.",
        },
        "device_feasible_target_p_mw": {
            "display_name": "设备可行目标功率 / Device-feasible target power",
            "symbol": "P^{device,target}_{i,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "考虑设备能力边界后的 VPP 有功功率目标。 / VPP active-power target after device capability limits.",
        },
        "raw_target_p_mw": {
            "display_name": "原始目标功率 / Raw target power",
            "symbol": "P^{raw,target}_{i,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "策略动作换算得到的原始 VPP 有功功率目标。 / Raw VPP active-power target converted from the policy action.",
        },
        "requested_delta_p_mw": {
            "display_name": "请求调节功率 / Requested power adjustment",
            "symbol": "\\Delta P^{request}_{i,t}",
            "unit": "MW",
            "group": "dataset",
            "physical_meaning": "DSO 或上层调度请求 VPP 提供的有功调节量。 / Active-power adjustment requested from the VPP by the DSO or upper-level dispatcher.",
        },
        "policy_normalized_aggregate_action": {
            "display_name": "策略归一化聚合动作 / Policy normalized aggregate action",
            "symbol": "a^{norm,agg}_{i,t}",
            "unit": "normalized",
            "group": "action",
            "physical_meaning": "调度策略输出的聚合层归一化动作；不是物理功率，展示时需要结合解码后的 MW 指标理解。 / Aggregate normalized action emitted by the dispatch policy; not a physical power value.",
        },
        "policy_normalized_der_action_mean": {
            "display_name": "DER 动作均值 / DER action mean",
            "symbol": "\\mu(a^{norm,DER}_{i,t})",
            "unit": "normalized",
            "group": "action",
            "physical_meaning": "策略输出的 DER 级归一化动作均值，用于观察内部设备动作整体方向。 / Mean of DER-level normalized policy actions.",
        },
        "policy_normalized_der_action_std": {
            "display_name": "DER 动作标准差 / DER action standard deviation",
            "symbol": "\\sigma(a^{norm,DER}_{i,t})",
            "unit": "normalized",
            "group": "action",
            "physical_meaning": "策略输出的 DER 级归一化动作离散程度，用于观察设备动作是否分散。 / Standard deviation of DER-level normalized policy actions.",
        },
        "episode_rows": {
            "display_name": "Episode 行数 / Episode rows",
            "symbol": "N_{episode\\_rows}",
            "unit": "count",
            "group": "progress",
            "physical_meaning": "训练 episode 指标文件中当前已经写入的行数。 / Number of rows currently written to the training episode metrics file.",
        },
        "loss_rows": {
            "display_name": "Loss 行数 / Loss rows",
            "symbol": "N_{loss\\_rows}",
            "unit": "count",
            "group": "progress",
            "physical_meaning": "训练 loss 指标文件中当前已经写入的行数。 / Number of rows currently written to the training loss metrics file.",
        },
        "progress_rows": {
            "display_name": "进度行数 / Progress rows",
            "symbol": "N_{progress\\_rows}",
            "unit": "count",
            "group": "progress",
            "physical_meaning": "训练 progress 文件中当前已经写入的行数。 / Number of rows currently written to the experiment progress file.",
        },
        "step_progress_pct": {
            "display_name": "步进完成比例 / Step progress percent",
            "symbol": "p^{step}_{progress}",
            "unit": "ratio",
            "group": "progress",
            "physical_meaning": "当前 rollout 或评估流程的步进完成比例。 / Step-completion ratio for the active rollout or evaluation phase.",
        },
        "violations_so_far": {
            "display_name": "累计约束违背次数 / Violations so far",
            "symbol": "N^{viol}_{sofar}",
            "unit": "count",
            "group": "constraint",
            "physical_meaning": "当前进行中的 episode 或 rollout 已累计发现的约束违背次数。 / Cumulative constraint violations observed so far in the active episode or rollout.",
        },
        "projection_gap_mw": {
            "display_name": "投影修正功率差 / Projection gap",
            "symbol": "\\left|P^{decoded}-P^{projected}\\right|",
            "unit": "MW",
            "group": "constraint",
            "physical_meaning": "策略解码目标与约束投影后目标之间的功率差，用于衡量动作被修正的幅度。 / Power gap between decoded and projected targets, measuring how much the action was corrected.",
        },
        "availability_payment": {
            "display_name": "可用容量补偿 / Availability payment",
            "symbol": "R^{avail}_{i,t}",
            "unit": "currency",
            "group": "reward",
            "physical_meaning": "因 VPP 提供可调节容量或可用性而获得的补偿收入；当前 v3.1 若可用容量补偿权重为 0，则仅作诊断，不计入最终训练奖励。 / Payment earned for available flexibility or dispatch capacity; when the v3.1 availability weight is zero, it is diagnostic and excluded from the final training reward.",
        },
        "availability_payment_weight": {
            "display_name": "可用容量补偿权重 / Availability payment weight",
            "symbol": "w^{avail}",
            "unit": "dimensionless",
            "group": "reward_weight",
            "physical_meaning": "可用容量补偿进入最终训练奖励时使用的乘数；当前 v3.1 默认为 0，表示不计入最终训练奖励。 / Multiplier applied to availability payment in the final training reward; current v3.1 defaults to 0, so the term is excluded.",
        },
        "dispatch_private_profit_reward": {
            "display_name": "调度私有收益奖励 / Dispatch private-profit reward",
            "symbol": "r^{private}_{dispatch,i,t}",
            "unit": "score",
            "group": "reward",
            "physical_meaning": "VPP 调度智能体以自身经济收益为核心的奖励分量。 / Reward component for the dispatch agent's private operating profit.",
        },
        "dispatch_reward_env": {
            "display_name": "环境返回调度奖励 / Environment dispatch reward",
            "symbol": "r^{env}_{dispatch,i,t}",
            "unit": "score",
            "group": "reward",
            "physical_meaning": "环境 step 直接返回给调度智能体、尚未经过算法侧再处理的 per-VPP reward。 / Per-VPP reward returned by env.step before algorithm-side processing.",
        },
        "economic_operational_surplus": {
            "display_name": "经济运行盈余 / Economic operational surplus",
            "symbol": "S^{op}_{i,t}",
            "unit": "currency",
            "group": "reward",
            "physical_meaning": "结算拆分中收入扣除运行成本后的经济盈余。 / Operating surplus from settlement revenue minus operating cost.",
        },
        "evcs_user_revenue_total": {
            "display_name": "充电用户收入 / EV charging user revenue",
            "symbol": "R^{EVCS,user}_{i,t}",
            "unit": "currency",
            "group": "reward",
            "physical_meaning": "充电桩向用户提供充电服务获得的总收入。 / Total revenue from EV charging users.",
        },
        "export_revenue_total": {
            "display_name": "总外送收入 / Total export revenue",
            "symbol": "R^{export}_{i,t}",
            "unit": "currency",
            "group": "reward",
            "physical_meaning": "DER 或 VPP 向外部市场/电网送电获得的总收入。 / Total revenue from exporting DER or VPP power.",
        },
        "flexibility_service_payment": {
            "display_name": "灵活性服务补偿 / Flexibility service payment",
            "symbol": "R^{flex}_{i,t}",
            "unit": "currency",
            "group": "reward",
            "physical_meaning": "响应调度或提供灵活性服务获得的真实补偿；当前 v3.1 若服务补偿权重为 0，则仅作诊断，不计入最终训练奖励。 / Real payment for providing flexibility service or accepting dispatch; when the v3.1 service weight is zero, it is diagnostic and excluded from the final training reward.",
        },
        "market_energy_margin_total": {
            "display_name": "市场电能边际收益 / Market energy margin",
            "symbol": "M^{energy}_{i,t}",
            "unit": "currency",
            "group": "reward",
            "physical_meaning": "外送收入和充电用户收入扣除购电成本后的电能市场净收益。 / Energy-market margin after export revenue, EV user revenue, and import energy cost.",
        },
        "mt_export_revenue_total": {
            "display_name": "微燃机外送收入 / Microturbine export revenue",
            "symbol": "R^{MT,export}_{i,t}",
            "unit": "currency",
            "group": "reward",
            "physical_meaning": "微燃机发电外送获得的收入。 / Revenue from exported microturbine generation.",
        },
        "preferred_region_bonus": {
            "display_name": "偏好运行区奖励 / Preferred-region bonus",
            "symbol": "r^{pref}_{i,t}",
            "unit": "score",
            "group": "reward",
            "physical_meaning": "动作或运行点落在偏好区域时给予的奖励。 / Bonus for operating in a preferred region.",
        },
        "private_profit_proxy": {
            "display_name": "私有收益代理值 / Private-profit proxy",
            "symbol": "\\hat{\\Pi}^{private}_{i,t}",
            "unit": "score",
            "group": "reward",
            "physical_meaning": "奖励加权前用于近似 VPP 私有收益的代理指标。 / Proxy for VPP private profit before reward weighting.",
        },
        "private_profit_weight": {
            "display_name": "私有收益权重 / Private-profit weight",
            "symbol": "w^{private}",
            "unit": "dimensionless",
            "group": "reward_weight",
            "physical_meaning": "经济运行盈余或私有收益代理值进入最终训练奖励时使用的乘数。 / Multiplier applied when settlement surplus or private-profit proxy enters the final training reward.",
        },
        "pv_export_revenue_total": {
            "display_name": "光伏外送收入 / PV export revenue",
            "symbol": "R^{PV,export}_{i,t}",
            "unit": "currency",
            "group": "reward",
            "physical_meaning": "光伏发电外送获得的收入。 / Revenue from exported PV generation.",
        },
        "quality_adjusted_operational_surplus": {
            "display_name": "质量修正运行盈余 / Quality-adjusted operational surplus",
            "symbol": "S^{quality}_{i,t}",
            "unit": "currency",
            "group": "reward",
            "physical_meaning": "扣除服务质量、舒适度或未服务惩罚后的经济运行盈余。 / Operating surplus after service-quality or unserved-demand penalties.",
        },
        "service_payment": {
            "display_name": "服务补偿 / Service payment",
            "symbol": "R^{service}_{i,t}",
            "unit": "currency",
            "group": "reward",
            "physical_meaning": "VPP 因接受调度或提供辅助/灵活性服务获得的真实补偿；当前 v3.1 若服务补偿权重为 0，则仅作诊断，不计入最终训练奖励。 / Real payment for accepted dispatch or ancillary/flexibility service; when the v3.1 service weight is zero, it is diagnostic and excluded from the final training reward.",
        },
        "service_payment_weight": {
            "display_name": "服务补偿权重 / Service payment weight",
            "symbol": "w^{service}",
            "unit": "dimensionless",
            "group": "reward_weight",
            "physical_meaning": "服务补偿进入最终训练奖励时使用的乘数；当前 v3.1 默认为 0，表示服务补偿不计入最终训练奖励。 / Multiplier applied to service payment in the final training reward; current v3.1 defaults to 0, so service payment is excluded.",
        },
        "storage_potential_raw": {
            "display_name": "储能潜在价值 / Storage potential value",
            "symbol": "V^{ESS,pot}_{i,t}",
            "unit": "currency",
            "group": "reward",
            "physical_meaning": "根据当前储能充放电、未来价格窗口和退化成本估计的未缩放储能未来价值。 / Unscaled future-value potential of storage based on current storage action, future price window, and degradation cost.",
        },
        "storage_potential_shaping_reward": {
            "display_name": "储能潜在价值塑形奖励 / Storage potential shaping reward",
            "symbol": "r^{ESS,pot}_{i,t}",
            "unit": "score",
            "group": "reward",
            "physical_meaning": "储能潜在价值乘以塑形权重后写入最终训练奖励的项。 / Storage potential value after the shaping weight, added to the final training reward.",
        },
        "storage_potential_shaping_weight": {
            "display_name": "储能塑形权重 / Storage shaping weight",
            "symbol": "w^{ESS,pot}",
            "unit": "dimensionless",
            "group": "reward_weight",
            "physical_meaning": "储能未缩放潜在价值进入最终训练奖励时使用的乘数。 / Multiplier that maps unscaled storage potential into the final training reward.",
        },
        "storage_discharge_revenue_total": {
            "display_name": "储能放电收入 / Storage discharge revenue",
            "symbol": "R^{ESS,dis}_{i,t}",
            "unit": "currency",
            "group": "reward",
            "physical_meaning": "储能放电对外供电获得的收入。 / Revenue from battery discharge delivering energy.",
        },
        "visible_energy_minus_operation_cost": {
            "display_name": "可见电能收益减运行成本 / Visible energy revenue minus operation cost",
            "symbol": "R^{energy}_{i,t}-C^{op}_{i,t}",
            "unit": "currency",
            "group": "reward",
            "physical_meaning": "可观测电能市场收入扣除聚合 DER 运行成本后的净值。 / Visible energy-market revenue net of aggregate DER operating cost.",
        },
        "battery_degradation_cost": {
            "display_name": "电池退化成本 / Battery degradation cost",
            "symbol": "C^{bat,deg}_{i,t}",
            "unit": "currency",
            "group": "cost",
            "physical_meaning": "电池充放电循环导致寿命损耗折算的成本。 / Cost proxy for battery degradation caused by cycling.",
        },
        "battery_degradation_cost_total": {
            "display_name": "总电池退化成本 / Total battery degradation cost",
            "symbol": "C^{bat,deg,total}_{i,t}",
            "unit": "currency",
            "group": "cost",
            "physical_meaning": "结算口径下所有相关储能设备的电池退化成本总和。 / Settlement total of battery degradation costs.",
        },
        "comfort_cost_total": {
            "display_name": "舒适度成本 / Comfort cost",
            "symbol": "C^{comfort}_{i,t}",
            "unit": "currency",
            "group": "cost",
            "physical_meaning": "HVAC 或可调负荷偏离舒适区产生的惩罚成本。 / Penalty cost from HVAC or flexible-load comfort deviation.",
        },
        "contract_delivery_penalty": {
            "display_name": "合同交付违约惩罚 / Contract delivery penalty",
            "symbol": "C^{contract}_{i,t}",
            "unit": "currency",
            "group": "cost",
            "physical_meaning": "未满足合同调度或交付要求产生的惩罚。 / Penalty for shortfall against contracted delivery.",
        },
        "der_operating_cost_total": {
            "display_name": "DER 总运行成本 / Total DER operating cost",
            "symbol": "C^{DER,op,total}_{i,t}",
            "unit": "currency",
            "group": "cost",
            "physical_meaning": "分布式能源设备在结算口径下的总运行成本。 / Total operating cost of distributed energy resources.",
        },
        "der_operation_cost": {
            "display_name": "DER 运行成本 / DER operation cost",
            "symbol": "C^{DER,op}_{i,t}",
            "unit": "currency",
            "group": "cost",
            "physical_meaning": "聚合 DER 运行、启停、燃料或维护相关成本。 / Aggregate DER operating cost such as running, fuel, or maintenance cost.",
        },
        "dispatch_projection_penalty": {
            "display_name": "调度投影惩罚 / Dispatch projection penalty",
            "symbol": "C^{proj}_{i,t}",
            "unit": "score",
            "group": "cost",
            "physical_meaning": "策略动作被可行性或安全投影修正时产生的惩罚项。 / Penalty when a policy action must be corrected by feasibility or safety projection.",
        },
        "reward_scaled_contract_delivery_penalty": {
            "display_name": "训练奖励合同违约惩罚 / Reward-scaled contract delivery penalty",
            "symbol": "\\tilde{C}^{contract}_{i,t}",
            "unit": "score",
            "group": "reward_scaled_cost",
            "physical_meaning": "合同交付惩罚乘以最终训练奖励中的合同权重后的值；当前 v3.1 合同权重为 0 时该项不计入。 / Contract delivery penalty after the reward contract weight; with current v3.1 contract weight 0, this term is excluded.",
        },
        "reward_scaled_dispatch_projection_penalty": {
            "display_name": "环境内调度投影惩罚 / Env reward-scaled projection penalty",
            "symbol": "\\tilde{C}^{proj,env}_{i,t}",
            "unit": "score",
            "group": "reward_scaled_cost",
            "physical_meaning": "环境 reward 中直接扣除的调度投影惩罚；训练侧可能还会额外再扣一次。 / Dispatch projection penalty subtracted inside the environment reward; the learner may subtract an additional training-side penalty.",
        },
        "reward_scaled_training_projection_penalty": {
            "display_name": "训练侧额外投影惩罚 / Training-side projection penalty",
            "symbol": "\\tilde{C}^{proj,train}_{i,t}",
            "unit": "score",
            "group": "reward_scaled_cost",
            "physical_meaning": "算法侧从环境调度奖励到训练调度奖励之间额外扣除的投影惩罚。 / Additional projection penalty subtracted by the algorithm when converting environment dispatch reward into training dispatch reward.",
        },
        "reward_scaled_total_projection_penalty": {
            "display_name": "总投影惩罚 / Total projection penalty",
            "symbol": "\\tilde{C}^{proj,total}_{i,t}",
            "unit": "score",
            "group": "reward_scaled_cost",
            "physical_meaning": "环境内投影惩罚与训练侧额外投影惩罚之和，用于解释最终训练奖励中的总投影惩罚影响。 / Sum of environment and training-side projection penalties, explaining the full projection impact on training reward.",
        },
        "reward_scaled_comfort_soc_penalty": {
            "display_name": "训练奖励舒适度/SOC 惩罚 / Reward-scaled comfort-SOC penalty",
            "symbol": "\\tilde{C}^{comfort,SOC}_{i,t}",
            "unit": "score",
            "group": "reward_scaled_cost",
            "physical_meaning": "缩放舒适度/SOC 惩罚乘以最终训练奖励中的舒适度/SOC 权重后的值。 / Scaled comfort-SOC penalty after the reward comfort-SOC weight.",
        },
        "reward_scaled_battery_degradation_penalty": {
            "display_name": "训练奖励电池退化惩罚 / Reward-scaled battery degradation penalty",
            "symbol": "\\tilde{C}^{bat,deg}_{i,t}",
            "unit": "score",
            "group": "reward_scaled_cost",
            "physical_meaning": "电池退化成本乘以最终训练奖励中的电池退化权重后的值。 / Battery degradation cost after the reward battery-degradation weight.",
        },
        "evcs_wholesale_cost_total": {
            "display_name": "充电站批发购电成本 / EVCS wholesale energy cost",
            "symbol": "C^{EVCS,wholesale}_{i,t}",
            "unit": "currency",
            "group": "cost",
            "physical_meaning": "充电站为满足充电需求从批发市场或电网购电产生的成本。 / Wholesale or grid energy cost for EV charging stations.",
        },
        "flex_energy_cost_total": {
            "display_name": "柔性负荷电能成本 / Flexible-load energy cost",
            "symbol": "C^{flex,energy}_{i,t}",
            "unit": "currency",
            "group": "cost",
            "physical_meaning": "柔性负荷用电产生的电能成本。 / Energy cost associated with flexible loads.",
        },
        "hvac_energy_cost_total": {
            "display_name": "HVAC 电能成本 / HVAC energy cost",
            "symbol": "C^{HVAC,energy}_{i,t}",
            "unit": "currency",
            "group": "cost",
            "physical_meaning": "暖通空调用电产生的电能成本。 / Energy cost from HVAC operation.",
        },
        "import_energy_cost_total": {
            "display_name": "总购电成本 / Total import energy cost",
            "symbol": "C^{import}_{i,t}",
            "unit": "currency",
            "group": "cost",
            "physical_meaning": "VPP 从外部电网或市场购入电能产生的总成本。 / Total cost of imported energy from the grid or market.",
        },
        "scaled_comfort_soc_penalty": {
            "display_name": "缩放舒适度/SOC 惩罚 / Scaled comfort-SOC penalty",
            "symbol": "C^{scaled}_{comfort,SOC}",
            "unit": "score",
            "group": "cost",
            "physical_meaning": "对舒适度偏差或储能 SOC 偏离进行缩放后的惩罚项。 / Scaled penalty for comfort deviation or storage SOC deviation.",
        },
        "storage_charge_cost_total": {
            "display_name": "储能充电成本 / Storage charge cost",
            "symbol": "C^{ESS,ch}_{i,t}",
            "unit": "currency",
            "group": "cost",
            "physical_meaning": "储能充电购电或吸收电能产生的成本。 / Cost of charging storage or absorbing energy.",
        },
        "total_cost_so_far": {
            "display_name": "累计成本 / Total cost so far",
            "symbol": "C^{sofar}_{t}",
            "unit": "currency",
            "group": "cost",
            "physical_meaning": "当前 episode 从 reset 到当前 time_index 已累计的成本；不是单步成本。 / Cumulative cost accumulated so far in the active episode; not one-step cost.",
        },
        "unclassified_import_cost_total": {
            "display_name": "未分类购电成本 / Unclassified import cost",
            "symbol": "C^{import,other}_{i,t}",
            "unit": "currency",
            "group": "cost",
            "physical_meaning": "尚未归入具体 DER 或负荷类别的购电成本。 / Import cost not assigned to a more specific DER or load category.",
        },
        "unserved_penalty_total": {
            "display_name": "未服务负荷惩罚 / Unserved-demand penalty",
            "symbol": "C^{unserved}_{i,t}",
            "unit": "currency",
            "group": "cost",
            "physical_meaning": "EV 或负荷需求未被满足时产生的惩罚成本。 / Penalty cost for unmet EV charging or load demand.",
        },
        "critic_grad_norm": {
            "display_name": "Critic 梯度范数 / Critic gradient norm",
            "symbol": "\\|\\nabla\\mathcal{L}_{critic}\\|",
            "unit": "scalar",
            "group": "loss",
            "physical_meaning": "Critic 网络一次更新中的梯度范数，用于观察训练稳定性。 / Gradient norm of the critic network for a learner update.",
        },
        "dispatch_policy_loss": {
            "display_name": "调度策略损失 / Dispatch policy loss",
            "symbol": "\\mathcal{L}^{dispatch}_{policy}",
            "unit": "scalar",
            "group": "loss",
            "physical_meaning": "VPP 调度策略网络更新时的策略损失。 / Policy loss for the VPP dispatch policy update.",
        },
        "dso_policy_loss": {
            "display_name": "DSO 策略损失 / DSO policy loss",
            "symbol": "\\mathcal{L}^{DSO}_{policy}",
            "unit": "scalar",
            "group": "loss",
            "physical_meaning": "DSO 策略网络更新时的策略损失。 / Policy loss for the DSO policy update.",
        },
        "portfolio_policy_loss": {
            "display_name": "组合策略损失 / Portfolio policy loss",
            "symbol": "\\mathcal{L}^{portfolio}_{policy}",
            "unit": "scalar",
            "group": "loss",
            "physical_meaning": "VPP 组合或投资组合策略网络更新时的策略损失。 / Policy loss for the VPP portfolio policy update.",
        },
    }
)


REWARD_COST_FORMULAS: dict[str, str] = {
    "availability_payment": "R^{\\text{可用容量}}_{i,t}=p^{\\text{可用容量}}_{t}\\,Q^{\\text{可用容量}}_{i,t}",
    "dispatch_private_profit_reward": "r^{\\text{私有收益}}_{i,t}=w^{\\text{私有收益}}\\,\\hat{\\Pi}^{\\text{私有}}_{i,t}",
    "dispatch_reward_env": "r^{\\text{环境调度}}_{i,t}=r^{\\text{收益}}_{i,t}+r^{\\text{安全}}_{i,t}-p^{\\text{违约}}_{i,t}",
    "dispatch_reward_train": "r^{\\text{训练调度}}_{i,t}=f^{\\text{算法处理}}(r^{\\text{环境调度}}_{i,t})",
    "economic_operational_surplus": "S^{\\text{运行}}_{i,t}=R^{\\text{外送}}_{i,t}+R^{\\text{充电用户}}_{i,t}-C^{\\text{购电}}_{i,t}-C^{\\text{DER运行}}_{i,t}-C^{\\text{电池退化}}_{i,t}",
    "energy_market_revenue": "R^{\\text{电能市场}}_{i,t}=\\pi_{t}\\,P^{\\text{交付}}_{i,t}\\,\\Delta t",
    "evcs_user_revenue_total": "R^{\\text{充电用户}}_{i,t}=\\pi^{\\text{充电服务}}_{t}\\,P^{\\text{充电负荷}}_{i,t}\\,\\Delta t",
    "export_revenue_total": "R^{\\text{外送}}_{i,t}=R^{\\text{光伏外送}}_{i,t}+R^{\\text{微燃机外送}}_{i,t}+R^{\\text{储能放电}}_{i,t}",
    "flexibility_service_payment": "R^{\\text{灵活性服务}}_{i,t}=p^{\\text{服务}}_t\\,\\Delta P^{\\text{接受}}_{i,t}\\,\\Delta t",
    "market_energy_margin_total": "M^{\\text{电能市场}}_{i,t}=R^{\\text{外送}}_{i,t}+R^{\\text{充电用户}}_{i,t}-C^{\\text{购电}}_{i,t}",
    "mt_export_revenue_total": "R^{\\text{微燃机外送}}_{i,t}=\\pi_{t}\\,P^{\\text{微燃机外送}}_{i,t}\\,\\Delta t",
    "preferred_region_bonus": "r^{\\text{偏好区}}_{i,t}=b^{\\text{偏好}}\\,\\mathbf{1}\\{x_{i,t}\\in\\Omega^{\\text{偏好}}\\}",
    "private_profit_weight": "w^{\\text{私有收益}}=\\text{经济运行盈余进入训练奖励的乘数}",
    "private_profit_proxy": "\\hat{\\Pi}^{\\text{私有}}_{i,t}=S^{\\text{质量修正}}_{i,t}+R^{\\text{服务}}_{i,t}+R^{\\text{可用容量}}_{i,t}-C^{\\text{合同违约}}_{i,t}",
    "pv_export_revenue_total": "R^{\\text{光伏外送}}_{i,t}=\\pi_{t}\\,P^{\\text{光伏外送}}_{i,t}\\,\\Delta t",
    "quality_adjusted_operational_surplus": "S^{\\text{质量修正}}_{i,t}=S^{\\text{运行}}_{i,t}-C^{\\text{服务质量惩罚}}_{i,t}",
    "reward_so_far": "R^{\\text{累计}}_{i,t}=\\sum_{\\tau=0}^{t} r_{i,\\tau}",
    "service_payment": "R^{\\text{服务}}_{i,t}=p^{\\text{服务}}_t\\,\\Delta P^{\\text{接受}}_{i,t}\\,\\Delta t",
    "service_payment_weight": "w^{\\text{服务}}=\\text{服务补偿进入训练奖励的乘数}",
    "availability_payment_weight": "w^{\\text{可用容量}}=\\text{可用容量补偿进入训练奖励的乘数}",
    "storage_potential_raw": "V^{\\text{储能潜在}}_{i,t}=\\left(\\gamma^{\\text{储能}}\\eta\\,\\pi^{\\text{未来最高}}-\\pi_{t}-c^{\\text{退化}}\\right)E^{\\text{充电}}_{i,t}-\\max(0,s_{i,t})E^{\\text{放电}}_{i,t}",
    "storage_potential_shaping_reward": "r^{\\text{储能塑形}}_{i,t}=w^{\\text{储能塑形}}V^{\\text{储能潜在}}_{i,t}",
    "storage_potential_shaping_weight": "w^{\\text{储能塑形}}=\\text{储能潜在价值进入训练奖励的乘数}",
    "storage_discharge_revenue_total": "R^{\\text{储能放电}}_{i,t}=\\pi_{t}\\,P^{\\text{储能放电}}_{i,t}\\,\\Delta t",
    "total_reward": "R^{\\text{总}}_{i,t}=\\sum_{k} r^{(k)}_{i,t}",
    "visible_energy_minus_operation_cost": "M^{\\text{可见电能}}_{i,t}=R^{\\text{电能市场}}_{i,t}-C^{\\text{DER运行}}_{i,t}",
    "battery_degradation_cost": "C^{\\text{电池退化}}_{i,t}=c^{\\text{退化}}\\,|P^{\\text{储能}}_{i,t}|\\,\\Delta t",
    "battery_degradation_cost_total": "C^{\\text{总电池退化}}_{i,t}=\\sum_{b\\in\\mathcal{B}_i} C^{\\text{电池退化}}_{b,t}",
    "comfort_cost_total": "C^{\\text{舒适度}}_{i,t}=\\lambda^{\\text{舒适}}\\,|x^{\\text{舒适}}_{i,t}-x^{\\text{目标}}_{i,t}|",
    "contract_delivery_penalty": "C^{\\text{合同违约}}_{i,t}=\\lambda^{\\text{合同}}\\,\\max(0,|\\Delta P^{\\text{请求}}_{i,t}-\\Delta P^{\\text{实际}}_{i,t}|-\\epsilon)",
    "der_operating_cost_total": "C^{\\text{DER总运行}}_{i,t}=\\sum_{d\\in\\mathcal{D}_i} C^{\\text{运行}}_{d,t}",
    "der_operation_cost": "C^{\\text{DER运行}}_{i,t}=c^{\\text{运行}}_{i,t}\\,\\Delta t",
    "dispatch_projection_penalty": "C^{\\text{调度投影}}_{i,t}=\\lambda^{\\text{投影}}\\,|P^{\\text{解码目标}}_{i,t}-P^{\\text{投影目标}}_{i,t}|",
    "reward_scaled_contract_delivery_penalty": "\\tilde{C}^{\\text{合同违约}}_{i,t}=w^{\\text{合同}}C^{\\text{合同违约}}_{i,t}",
    "reward_scaled_dispatch_projection_penalty": "\\tilde{C}^{\\text{投影,环境}}_{i,t}=C^{\\text{调度投影}}_{i,t}",
    "reward_scaled_training_projection_penalty": "\\tilde{C}^{\\text{投影,训练}}_{i,t}=r^{\\text{环境调度}}_{i,t}-r^{\\text{训练调度}}_{i,t}",
    "reward_scaled_total_projection_penalty": "\\tilde{C}^{\\text{投影,总}}_{i,t}=\\tilde{C}^{\\text{投影,环境}}_{i,t}+\\tilde{C}^{\\text{投影,训练}}_{i,t}",
    "reward_scaled_comfort_soc_penalty": "\\tilde{C}^{\\text{舒适SOC}}_{i,t}=w^{\\text{舒适SOC}}C^{\\text{缩放舒适SOC}}_{i,t}",
    "reward_scaled_battery_degradation_penalty": "\\tilde{C}^{\\text{电池退化}}_{i,t}=w^{\\text{电池退化}}C^{\\text{电池退化}}_{i,t}",
    "energy_purchase_cost": "C^{\\text{购电}}_{i,t}=\\pi_{t}\\,\\max(P^{\\text{净负荷}}_{i,t},0)\\,\\Delta t",
    "evcs_wholesale_cost_total": "C^{\\text{充电站批发购电}}_{i,t}=\\pi_{t}\\,P^{\\text{充电负荷}}_{i,t}\\,\\Delta t",
    "flex_energy_cost_total": "C^{\\text{柔性负荷电能}}_{i,t}=\\pi_{t}\\,P^{\\text{柔性负荷}}_{i,t}\\,\\Delta t",
    "hvac_energy_cost_total": "C^{\\text{HVAC电能}}_{i,t}=\\pi_{t}\\,P^{\\text{HVAC}}_{i,t}\\,\\Delta t",
    "import_energy_cost_total": "C^{\\text{购电}}_{i,t}=\\pi_{t}\\,P^{\\text{购电}}_{i,t}\\,\\Delta t",
    "scaled_comfort_soc_penalty": "C^{\\text{缩放舒适SOC}}_{i,t}=\\lambda^{\\text{舒适}}\\,d^{\\text{舒适}}_{i,t}+\\lambda^{\\text{SOC}}\\,d^{\\text{SOC}}_{i,t}",
    "storage_charge_cost_total": "C^{\\text{储能充电}}_{i,t}=\\pi_{t}\\,P^{\\text{储能充电}}_{i,t}\\,\\Delta t",
    "storage_degradation_cost": "C^{\\text{储能退化}}_{i,t}=c^{\\text{退化}}\\,|P^{ESS}_{i,t}|\\,\\Delta t",
    "total_cost": "C^{\\text{总}}_{i,t}=\\sum_{k} C^{(k)}_{i,t}",
    "total_cost_so_far": "C^{\\text{累计}}_{i,t}=\\sum_{\\tau=0}^{t} C^{\\text{总}}_{i,\\tau}",
    "constraint_violation_cost": "C^{\\text{约束违背}}_{i,t}=\\lambda^{\\text{约束}}\\,\\max(g_{i,t},0)",
    "unclassified_import_cost_total": "C^{\\text{未分类购电}}_{i,t}=C^{\\text{购电}}_{i,t}-\\sum_{m} C^{\\text{已分类购电},m}_{i,t}",
    "unserved_penalty_total": "C^{\\text{未服务惩罚}}_{i,t}=\\lambda^{\\text{未服务}}\\,E^{\\text{未服务}}_{i,t}",
}


def _formula_lhs(value: Any) -> str:
    text = str(value or "")
    return text.split("=", 1)[0].strip() if "=" in text else ""


for _metric_name, _formula in REWARD_COST_FORMULAS.items():
    _defaults = COMMON_VARIABLES.setdefault(_metric_name, {})
    _defaults["formula_latex"] = _formula
    _lhs = _formula_lhs(_formula)
    if _lhs:
        _defaults["symbol"] = _lhs


_CODE_IDENTIFIER_RE = re.compile(r"\b[a-z][a-z0-9]+_[a-z0-9_]+\b")
_LEGACY_FORMULA_TOKENS = (
    "c_t",
    "P^B",
    "P^{B}",
    "P^W",
    "P^{W}",
    "P^L",
    "P^{L}",
    "C^{energy}",
    "C^{deg}",
    "C^{viol}",
    "R^{sell}",
    "profit_t",
    "penalty_t",
)


def _has_cjk(value: Any) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(value or ""))


def should_use_default_formula(value: Any) -> bool:
    text = str(value or "")
    if text.strip() == "":
        return True
    return (
        bool(_CODE_IDENTIFIER_RE.search(text))
        or " enabled transfers" in text
        or any(token in text for token in _LEGACY_FORMULA_TOKENS)
    )


def should_use_canonical_default_formula(value: Any, default_value: Any) -> bool:
    if default_value in (None, ""):
        return False
    if should_use_default_formula(value):
        return True
    return _has_cjk(default_value) and not _has_cjk(value)


def _should_use_default_symbol(value: Any) -> bool:
    text = str(value or "")
    if text.strip() == "":
        return True
    return any(token in text for token in _LEGACY_FORMULA_TOKENS)


def enrich_variable_definition(variable: dict[str, Any]) -> dict[str, Any]:
    name = str(variable.get("name", ""))
    defaults = COMMON_VARIABLES.get(name)
    if not defaults:
        return variable
    enriched = dict(variable)
    for key, value in defaults.items():
        current = enriched.get(key)
        if current in (None, ""):
            enriched[key] = value
        elif key in {"display_name", "physical_meaning", "notes"} and not _has_cjk(current):
            enriched[key] = value
        elif key == "symbol" and _should_use_default_symbol(current):
            enriched[key] = value
        elif key == "formula_latex" and should_use_canonical_default_formula(current, value):
            enriched[key] = value
    return enriched


def enrich_variable_dictionary(variables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [enrich_variable_definition(variable) for variable in variables]


def variable_defaults(name: str) -> dict[str, Any]:
    return COMMON_VARIABLES.get(str(name), {})


def default_formula_dictionary() -> dict[str, Any]:
    return {
        str(metric_name): defaults["formula_latex"]
        for metric_name, defaults in COMMON_VARIABLES.items()
        if defaults.get("formula_latex") not in (None, "")
    }


def enrich_formula_value(metric_name: str, formula: Any) -> Any:
    defaults = variable_defaults(metric_name)
    default_formula = defaults.get("formula_latex")
    if should_use_canonical_default_formula(formula, default_formula):
        return default_formula
    return formula


def enrich_formula_dictionary(formulas: dict[str, Any]) -> dict[str, Any]:
    return {str(metric_name): enrich_formula_value(str(metric_name), formula) for metric_name, formula in formulas.items()}


def enrich_metric_record(row: dict[str, Any]) -> dict[str, Any]:
    name = str(row.get("metric_name", ""))
    defaults = variable_defaults(name)
    if not defaults:
        return row
    enriched = dict(row)
    display_name = defaults.get("display_name")
    if display_name and not enriched.get("display_name"):
        enriched["display_name"] = display_name
    description = enriched.get("description")
    physical_meaning = defaults.get("physical_meaning")
    if description in (None, "") or not _has_cjk(description):
        enriched["description"] = physical_meaning or display_name
    for key in ("unit", "formula_latex"):
        if enriched.get(key) in (None, "") and defaults.get(key) not in (None, ""):
            enriched[key] = defaults[key]
        elif (
            key == "formula_latex"
            and defaults.get(key) not in (None, "")
            and should_use_canonical_default_formula(enriched.get(key), defaults.get(key))
        ):
            enriched[key] = defaults[key]
    return enriched
