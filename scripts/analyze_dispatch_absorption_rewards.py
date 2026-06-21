from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


SETTLEMENT_BREAKDOWN_COLUMNS = [
    "economic_operational_surplus",
    "quality_adjusted_operational_surplus",
    "service_quality_penalty_total",
    "export_revenue_total",
    "pv_export_revenue_total",
    "mt_export_revenue_total",
    "storage_discharge_revenue_total",
    "evcs_user_revenue_total",
    "import_energy_cost_total",
    "evcs_wholesale_cost_total",
    "storage_charge_cost_total",
    "hvac_energy_cost_total",
    "flex_energy_cost_total",
    "unclassified_import_cost_total",
    "der_operating_cost_total",
    "battery_degradation_cost_total",
    "comfort_cost_total",
    "unserved_penalty_total",
    "legacy_operational_surplus_with_service_quality",
]

REQUIRED_SETTLEMENT_TRACE_COLUMNS = {
    "economic_operational_surplus",
    "quality_adjusted_operational_surplus",
    "service_quality_penalty_total",
    "export_revenue_total",
    "evcs_user_revenue_total",
    "import_energy_cost_total",
    "der_operating_cost_total",
    "battery_degradation_cost_total",
    "comfort_cost_total",
    "unserved_penalty_total",
}

ACTION_LANDING_COLUMNS = [
    "raw_action_norm",
    "raw_target_p_mw",
    "decoded_target_p_mw",
    "device_feasible_target_p_mw",
    "pre_ac_target_p_mw",
    "ac_projected_target_p_mw",
    "ac_certified_target_p_mw",
    "actual_target_p_mw",
    "raw_delta_p_mw",
    "decoded_delta_p_mw",
    "device_feasible_delta_p_mw",
    "pre_ac_delta_p_mw",
    "ac_projected_delta_p_mw",
    "ac_certified_delta_p_mw",
    "raw_to_device_gap_mw",
    "device_to_ac_gap_mw",
    "ac_to_actual_gap_mw",
    "accepted_to_actual_gap_mw",
    "actual_delta_nonzero_flag",
    "action_landing_ratio",
    "action_landing_drop_reason_code",
]

REQUIRED_ACTION_LANDING_TRACE_COLUMNS = {
    "decoded_delta_p_mw",
    "actual_delta_p_mw",
    "actual_delta_nonzero_flag",
    "action_landing_ratio",
    "action_landing_drop_reason",
}


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _classify(row: pd.Series) -> str:
    delivered = float(row["delivered_p_mw"])
    revenue = float(row["energy_market_revenue"])
    cost = float(row["der_operation_cost"])
    proxy = float(row["private_profit_proxy"])
    if proxy >= 0.0:
        return "非负：收入覆盖成本"
    if delivered < 0.0 and revenue < 0.0:
        return "负：VPP净吸收功率导致能量收入为负"
    if revenue >= 0.0 and cost > revenue:
        return "负：VPP净出力/少量出力但DER运行成本高于收入"
    if abs(delivered) < 1e-9 and cost > 0.0:
        return "负：无净交付但仍有DER运行成本"
    return "负：混合原因"


def _direction(value: float, eps: float = 1e-8) -> str:
    if value > eps:
        return "正向出力/减小吸收"
    if value < -eps:
        return "负向吸收/增大负荷"
    return "无有效变化"


def _classify_setting_vs_nn(row: pd.Series) -> str:
    negative = bool(row["negative_private_profit_reward"])
    net_absorption = bool(row["net_absorption"])
    cost_exceeds = bool(row["cost_exceeds_revenue"])
    requested = float(row["requested_delta_p_mw"])
    accepted = float(row["accepted_delta_p_mw"])
    actual = float(row["actual_delta_p_mw"])
    delivered = float(row["delivered_p_mw"])
    baseline = float(row["baseline_p_mw"])
    eps = 1e-8

    if not negative:
        return "非负：收入覆盖成本或总奖励为正"
    if abs(actual) <= eps and abs(delivered - baseline) <= eps:
        if net_absorption and requested > eps:
            return "设定/物理主导：VPP已净吸收，DOE/目标想增出力但实际未改变"
        if net_absorption and requested < -eps:
            return "设定/物理主导：VPP已净吸收，DOE/目标也偏向吸收但实际未改变"
        if net_absorption:
            return "设定/物理主导：VPP已净吸收且动作没有实际落地"
        if cost_exceeds:
            return "成本口径主导：无净吸收或少量出力，但运行成本高于收入且动作未落地"
        return "设定/物理主导：动作没有实际落地"
    if accepted < -eps and actual < -eps:
        return "动作疑似主导：接受目标和实际变化都朝吸收方向"
    if accepted > eps and actual > eps and net_absorption:
        return "动作在纠偏但不足：实际朝出力方向变化，仍处于净吸收"
    if accepted < -eps and abs(actual) <= eps:
        return "动作目标偏吸收但未落地：需查投影/DER物理约束"
    if accepted > eps and abs(actual) <= eps:
        return "动作目标偏出力但未落地：需查投影/DER物理约束"
    if cost_exceeds:
        return "成本口径主导：收入不足以覆盖DER成本"
    return "混合原因：需结合DER级功率和DOE字段进一步判断"


def _classify_settlement_driver(row: pd.Series) -> str:
    private_profit = float(row["private_profit_proxy"])
    visible_residual = float(row["private_profit_vs_visible_energy_residual"])
    visible_margin = float(row["visible_energy_minus_operation_cost"])
    settlement_complete = float(row["settlement_trace_complete"]) >= 1.0
    if not settlement_complete:
        if abs(visible_residual) > max(10.0, 10.0 * abs(visible_margin)):
            return "旧 trace 缺少完整 settlement 分项：大残差疑似旧 operational_surplus 混入 raw comfort/unserved penalty"
        return "旧 trace 缺少完整 settlement 分项：只能按可见电费和DER运行成本粗判"

    economic_surplus = float(row["economic_operational_surplus"])
    market_margin = float(row["market_energy_margin_total"])
    service_quality = float(row["service_quality_penalty_total"])
    der_cost = float(row["der_operating_cost_total"]) + float(row["battery_degradation_cost_total"])
    if private_profit >= 0.0 and service_quality > max(10.0, abs(economic_surplus)):
        return "经济利润非负，但服务质量惩罚很大：不应再把服务质量惩罚混入私有利润"
    if economic_surplus < 0.0 and market_margin < 0.0:
        return "经济利润为负：DER逐项市场收入/EVCS收入不足以覆盖购电成本"
    if economic_surplus < 0.0 and der_cost > max(0.0, market_margin):
        return "经济利润为负：DER运行/退化成本超过市场净收入"
    if economic_surplus < 0.0:
        return "经济利润为负：混合结算成本超过收入"
    if service_quality > 0.0:
        return "经济利润非负：服务质量惩罚仅作为质量审计项"
    return "经济利润非负：收入覆盖成本"


def _table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if frame.empty:
        return "无数据。"
    view = frame.loc[:, columns].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    return view.to_markdown(index=False)


def analyze(trace_path: Path, output_dir: Path) -> tuple[Path, Path]:
    trace = pd.read_csv(trace_path)
    original_columns = set(trace.columns)
    settlement_trace_complete = float(REQUIRED_SETTLEMENT_TRACE_COLUMNS.issubset(original_columns))
    action_landing_trace_complete = float(REQUIRED_ACTION_LANDING_TRACE_COLUMNS.issubset(original_columns))
    for column in [
        "market_price",
        "delivered_p_mw",
        "dt_hours",
        "energy_market_revenue",
        "der_operation_cost",
        "visible_energy_minus_operation_cost",
        "market_energy_margin_total",
        "private_profit_proxy",
        "dispatch_private_profit_reward",
        "service_payment",
        "availability_payment",
        "contract_shortfall_mw",
        "contract_delivery_penalty",
        "dispatch_projection_penalty",
        "scaled_comfort_soc_penalty",
        "battery_degradation_cost",
        "baseline_p_mw",
        "requested_delta_p_mw",
        "accepted_delta_p_mw",
        "actual_delta_p_mw",
        "verified_delivery_mw",
        *SETTLEMENT_BREAKDOWN_COLUMNS,
        *ACTION_LANDING_COLUMNS,
    ]:
        trace[column] = _num(trace, column)
    if "action_landing_drop_reason" not in original_columns:
        trace["action_landing_drop_reason"] = "legacy_trace_missing_landing_audit"

    if "visible_energy_minus_operation_cost" not in original_columns:
        trace["visible_energy_minus_operation_cost"] = trace["energy_market_revenue"] - trace["der_operation_cost"]
    if "market_energy_margin_total" not in original_columns:
        if {"export_revenue_total", "evcs_user_revenue_total", "import_energy_cost_total"}.issubset(original_columns):
            trace["market_energy_margin_total"] = (
                trace["export_revenue_total"] + trace["evcs_user_revenue_total"] - trace["import_energy_cost_total"]
            )
        else:
            trace["market_energy_margin_total"] = trace["visible_energy_minus_operation_cost"]
    if "economic_operational_surplus" not in original_columns:
        trace["economic_operational_surplus"] = trace["private_profit_proxy"]
    if "quality_adjusted_operational_surplus" not in original_columns:
        trace["quality_adjusted_operational_surplus"] = trace["private_profit_proxy"]
    if "service_quality_penalty_total" not in original_columns:
        trace["service_quality_penalty_total"] = trace["comfort_cost_total"] + trace["unserved_penalty_total"]
    trace["settlement_trace_complete"] = settlement_trace_complete
    trace["action_landing_trace_complete"] = action_landing_trace_complete
    trace["private_profit_vs_visible_energy_residual"] = (
        trace["private_profit_proxy"] - trace["visible_energy_minus_operation_cost"]
    )
    trace["economic_surplus_vs_market_margin_residual"] = (
        trace["economic_operational_surplus"] - trace["market_energy_margin_total"]
    )
    trace["negative_private_profit_reward"] = trace["dispatch_private_profit_reward"] < 0.0
    trace["net_absorption"] = trace["delivered_p_mw"] < 0.0
    trace["negative_revenue"] = trace["energy_market_revenue"] < 0.0
    trace["cost_exceeds_revenue"] = trace["der_operation_cost"] > trace["energy_market_revenue"]
    trace["negative_reason"] = trace.apply(_classify, axis=1)
    trace["requested_direction"] = trace["requested_delta_p_mw"].map(_direction)
    trace["accepted_direction"] = trace["accepted_delta_p_mw"].map(_direction)
    trace["actual_direction"] = trace["actual_delta_p_mw"].map(_direction)
    trace["actual_action_effective"] = trace["actual_delta_p_mw"].abs() > 1e-8
    trace["accepted_action_nonzero"] = trace["accepted_delta_p_mw"].abs() > 1e-8
    trace["setting_vs_nn_judgement"] = trace.apply(_classify_setting_vs_nn, axis=1)
    trace["settlement_driver"] = trace.apply(_classify_settlement_driver, axis=1)
    if "action_landing_ratio" in original_columns:
        trace["action_landing_ratio"] = trace["action_landing_ratio"].clip(lower=0.0)

    out = output_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    stem = trace_path.stem
    detail_path = out / f"{stem}_negative_reason_steps.csv"
    report_path = out / f"{stem}_absorption_root_cause_report.md"
    trace.to_csv(detail_path, index=False)

    by_reason = (
        trace.groupby("negative_reason", dropna=False)
        .agg(
            rows=("step", "count"),
            agents=("agent_id", "nunique"),
            total_reward=("dispatch_private_profit_reward", "sum"),
            total_proxy=("private_profit_proxy", "sum"),
            total_revenue=("energy_market_revenue", "sum"),
            total_cost=("der_operation_cost", "sum"),
            total_visible_margin=("visible_energy_minus_operation_cost", "sum"),
            total_private_vs_visible_residual=("private_profit_vs_visible_energy_residual", "sum"),
            total_economic_surplus=("economic_operational_surplus", "sum"),
            total_market_margin=("market_energy_margin_total", "sum"),
            total_service_quality_penalty=("service_quality_penalty_total", "sum"),
            mean_delivered_p_mw=("delivered_p_mw", "mean"),
        )
        .reset_index()
        .sort_values("total_reward")
    )
    by_agent = (
        trace.groupby(["agent_id", "vpp_id"], dropna=False)
        .agg(
            steps=("step", "count"),
            negative_steps=("negative_private_profit_reward", "sum"),
            absorption_steps=("net_absorption", "sum"),
            negative_revenue_steps=("negative_revenue", "sum"),
            cost_exceeds_revenue_steps=("cost_exceeds_revenue", "sum"),
            total_reward=("dispatch_private_profit_reward", "sum"),
            total_proxy=("private_profit_proxy", "sum"),
            total_revenue=("energy_market_revenue", "sum"),
            total_cost=("der_operation_cost", "sum"),
            total_visible_margin=("visible_energy_minus_operation_cost", "sum"),
            total_private_vs_visible_residual=("private_profit_vs_visible_energy_residual", "sum"),
            total_economic_surplus=("economic_operational_surplus", "sum"),
            total_market_margin=("market_energy_margin_total", "sum"),
            total_service_quality_penalty=("service_quality_penalty_total", "sum"),
            total_export_revenue=("export_revenue_total", "sum"),
            total_evcs_user_revenue=("evcs_user_revenue_total", "sum"),
            total_import_energy_cost=("import_energy_cost_total", "sum"),
            total_der_operating_cost=("der_operating_cost_total", "sum"),
            total_comfort_cost=("comfort_cost_total", "sum"),
            total_unserved_penalty=("unserved_penalty_total", "sum"),
            mean_delivered_p_mw=("delivered_p_mw", "mean"),
            min_delivered_p_mw=("delivered_p_mw", "min"),
            max_delivered_p_mw=("delivered_p_mw", "max"),
            total_service_payment=("service_payment", "sum"),
            total_availability_payment=("availability_payment", "sum"),
            total_contract_penalty=("contract_delivery_penalty", "sum"),
            total_projection_penalty=("dispatch_projection_penalty", "sum"),
            actual_action_effective_steps=("actual_action_effective", "sum"),
            accepted_action_nonzero_steps=("accepted_action_nonzero", "sum"),
            actual_delta_nonzero_steps=("actual_delta_nonzero_flag", "sum"),
            mean_action_landing_ratio=("action_landing_ratio", "mean"),
            total_raw_to_device_gap_mw=("raw_to_device_gap_mw", "sum"),
            total_device_to_ac_gap_mw=("device_to_ac_gap_mw", "sum"),
            total_ac_to_actual_gap_mw=("ac_to_actual_gap_mw", "sum"),
            total_accepted_to_actual_gap_mw=("accepted_to_actual_gap_mw", "sum"),
        )
        .reset_index()
    )
    by_agent["negative_rate"] = by_agent["negative_steps"] / by_agent["steps"].clip(lower=1)
    by_agent["absorption_rate"] = by_agent["absorption_steps"] / by_agent["steps"].clip(lower=1)
    by_agent["actual_action_effective_rate"] = by_agent["actual_action_effective_steps"] / by_agent["steps"].clip(lower=1)
    by_agent["accepted_action_nonzero_rate"] = by_agent["accepted_action_nonzero_steps"] / by_agent["steps"].clip(lower=1)
    by_agent["actual_delta_nonzero_rate"] = by_agent["actual_delta_nonzero_steps"] / by_agent["steps"].clip(lower=1)
    by_agent = by_agent.sort_values("total_reward")
    by_judgement = (
        trace.groupby("setting_vs_nn_judgement", dropna=False)
        .agg(
            rows=("step", "count"),
            agents=("agent_id", "nunique"),
            total_reward=("dispatch_private_profit_reward", "sum"),
            total_revenue=("energy_market_revenue", "sum"),
            total_cost=("der_operation_cost", "sum"),
            absorption_steps=("net_absorption", "sum"),
            actual_action_effective_steps=("actual_action_effective", "sum"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    by_settlement_driver = (
        trace.groupby("settlement_driver", dropna=False)
        .agg(
            rows=("step", "count"),
            agents=("agent_id", "nunique"),
            total_reward=("dispatch_private_profit_reward", "sum"),
            total_private_profit=("private_profit_proxy", "sum"),
            total_visible_margin=("visible_energy_minus_operation_cost", "sum"),
            total_private_vs_visible_residual=("private_profit_vs_visible_energy_residual", "sum"),
            total_economic_surplus=("economic_operational_surplus", "sum"),
            total_service_quality_penalty=("service_quality_penalty_total", "sum"),
        )
        .reset_index()
        .sort_values("total_reward")
    )
    by_action_landing = (
        trace.groupby("action_landing_drop_reason", dropna=False)
        .agg(
            rows=("step", "count"),
            agents=("agent_id", "nunique"),
            actual_delta_nonzero_steps=("actual_delta_nonzero_flag", "sum"),
            mean_action_landing_ratio=("action_landing_ratio", "mean"),
            total_raw_to_device_gap_mw=("raw_to_device_gap_mw", "sum"),
            total_device_to_ac_gap_mw=("device_to_ac_gap_mw", "sum"),
            total_ac_to_actual_gap_mw=("ac_to_actual_gap_mw", "sum"),
            total_accepted_to_actual_gap_mw=("accepted_to_actual_gap_mw", "sum"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    by_action_landing["actual_delta_nonzero_rate"] = (
        by_action_landing["actual_delta_nonzero_steps"] / by_action_landing["rows"].clip(lower=1)
    )

    worst_absorption = trace[trace["net_absorption"]].sort_values("dispatch_private_profit_reward").head(40)
    worst_cost = trace[
        (~trace["net_absorption"]) & trace["negative_private_profit_reward"]
    ].sort_values("dispatch_private_profit_reward").head(40)
    worst_residual = trace.reindex(
        trace["private_profit_vs_visible_energy_residual"].abs().sort_values(ascending=False).index
    ).head(40)
    trace_state_sentence = (
        "该 trace 已包含完整 settlement 分项，可以区分经济运行盈余和服务质量惩罚。"
        if settlement_trace_complete >= 1.0
        else "旧 trace 缺少完整 settlement 分项；若 private profit 与可见电费/运行成本残差很大，旧 operational_surplus 可能混入 raw comfort/unserved penalty。"
    )

    lines = [
        "# Dispatch 私有利润为负与净吸收功率根因报告",
        "",
        f"- 数据文件：`{trace_path}`",
        f"- 逐 step 明细：`{detail_path}`",
        f"- episode：`{int(trace['episode'].max()) if 'episode' in trace else 'unknown'}`",
        f"- 总行数：`{len(trace)}`",
        f"- settlement trace 完整性：`{settlement_trace_complete:.0f}`",
        f"- action landing trace 完整性：`{action_landing_trace_complete:.0f}`",
        f"- trace 口径判断：{trace_state_sentence}",
        "",
        "## 1. 结论先行",
        "",
        "当前负私有利润奖励不是单一原因。按 trace 证据，它至少分成两类：",
        "",
        "1. VPP 净吸收功率：`delivered_p_mw < 0`，导致 `energy_market_revenue = market_price * delivered_p_mw * dt_hours` 为负。",
        "2. 成本压过收入：`delivered_p_mw >= 0` 但 `der_operation_cost > energy_market_revenue`，即使 VPP 没有净吸收，私有利润代理仍为负。",
        "3. 旧口径残差：旧 trace 如果没有完整 settlement 分项，而 `private_profit_proxy - visible_energy_minus_operation_cost` 极大，说明当时私有利润不是简单电费减运行成本，必须回到结算口径检查。",
        "",
        "这说明不能只把负 reward 归咎于神经网络把 VPP 调成吸收功率；当前 reward/场景设定本身也会让正出力 VPP 因运行成本而得到负私有利润项。",
        "",
        "## 1.1 当前 trace 中使用的关键公式",
        "",
        "- 可见电费运行差额 = `energy_market_revenue - der_operation_cost`。",
        "- DER 逐项市场净收入 = `export_revenue_total + evcs_user_revenue_total - import_energy_cost_total`。",
        "- 经济运行盈余 = `export_revenue_total + evcs_user_revenue_total - import_energy_cost_total - der_operating_cost_total - battery_degradation_cost_total`。",
        "- 服务质量惩罚 = `comfort_cost_total + unserved_penalty_total`。",
        "- 质量调整盈余 = `economic_operational_surplus - service_quality_penalty_total`。",
        "- 当前主训练口径下的 dispatch 私有利润应使用经济运行盈余，不应再把 raw 服务质量惩罚直接混入私有利润。",
        "",
        "## 2. 按原因分类统计",
        "",
        _table(
            by_reason,
            [
                "negative_reason",
                "rows",
                "agents",
                "total_reward",
                "total_proxy",
                "total_revenue",
                "total_cost",
                "total_private_vs_visible_residual",
                "total_economic_surplus",
                "total_service_quality_penalty",
                "mean_delivered_p_mw",
            ],
        ),
        "",
        "## 3. 按 settlement driver 统计",
        "",
        _table(
            by_settlement_driver,
            [
                "settlement_driver",
                "rows",
                "agents",
                "total_reward",
                "total_private_profit",
                "total_visible_margin",
                "total_private_vs_visible_residual",
                "total_economic_surplus",
                "total_service_quality_penalty",
            ],
        ),
        "",
        "## 4. 动作落地审计",
        "",
        "本节用于判断 dispatch actor 的动作是否真实进入 DER/VPP 执行功率。核心指标：",
        "",
        "- `actual_delta_nonzero_rate`：实际执行功率相对 baseline 非零的比例。",
        "- `action_landing_ratio`：`abs(actual_delta_p_mw) / (abs(decoded_delta_p_mw) + epsilon)`。",
        "- `raw_to_device_gap_mw`：raw/decoded 目标被设备物理边界吃掉的量。",
        "- `device_to_ac_gap_mw`：设备可行目标到 AC-aware/DOE 目标之间的收缩量。",
        "- `ac_to_actual_gap_mw`：AC 投影/证书到真实执行结果之间的差距。",
        "- `accepted_to_actual_gap_mw`：接受目标和真实执行变化之间的差距。",
        "",
        _table(
            by_action_landing,
            [
                "action_landing_drop_reason",
                "rows",
                "agents",
                "actual_delta_nonzero_steps",
                "actual_delta_nonzero_rate",
                "mean_action_landing_ratio",
                "total_raw_to_device_gap_mw",
                "total_device_to_ac_gap_mw",
                "total_ac_to_actual_gap_mw",
                "total_accepted_to_actual_gap_mw",
            ],
        ),
        "",
        "## 5. 按 dispatch agent 统计",
        "",
        _table(
            by_agent,
            [
                "agent_id",
                "steps",
                "negative_steps",
                "negative_rate",
                "absorption_steps",
                "absorption_rate",
                "total_reward",
                "total_revenue",
                "total_cost",
                "total_visible_margin",
                "total_private_vs_visible_residual",
                "total_economic_surplus",
                "total_market_margin",
                "total_service_quality_penalty",
                "total_export_revenue",
                "total_evcs_user_revenue",
                "total_import_energy_cost",
                "total_der_operating_cost",
                "total_comfort_cost",
                "total_unserved_penalty",
                "mean_delivered_p_mw",
                "total_service_payment",
                "total_contract_penalty",
                "actual_action_effective_steps",
                "accepted_action_nonzero_steps",
                "actual_delta_nonzero_rate",
                "mean_action_landing_ratio",
                "total_raw_to_device_gap_mw",
                "total_device_to_ac_gap_mw",
                "total_ac_to_actual_gap_mw",
                "total_accepted_to_actual_gap_mw",
            ],
        ),
        "",
        "## 6. 按“神经网络动作 vs 设定/物理”归因统计",
        "",
        _table(
            by_judgement,
            [
                "setting_vs_nn_judgement",
                "rows",
                "agents",
                "total_reward",
                "total_revenue",
                "total_cost",
                "absorption_steps",
                "actual_action_effective_steps",
            ],
        ),
        "",
        "## 7. private profit 与可见电费成本残差最大的 step",
        "",
        _table(
            worst_residual,
            [
                "episode",
                "step",
                "agent_id",
                "market_price",
                "delivered_p_mw",
                "visible_energy_minus_operation_cost",
                "market_energy_margin_total",
                "economic_operational_surplus",
                "service_quality_penalty_total",
                "private_profit_proxy",
                "private_profit_vs_visible_energy_residual",
                "settlement_driver",
            ],
            max_rows=40,
        ),
        "",
        "## 8. 净吸收导致负 reward 的最严重 step",
        "",
        _table(
            worst_absorption,
            [
                "episode",
                "step",
                "agent_id",
                "market_price",
                "delivered_p_mw",
                "energy_market_revenue",
                "der_operation_cost",
                "visible_energy_minus_operation_cost",
                "economic_operational_surplus",
                "service_quality_penalty_total",
                "private_profit_proxy",
                "private_profit_vs_visible_energy_residual",
                "dispatch_private_profit_reward",
                "baseline_p_mw",
                "requested_delta_p_mw",
                "accepted_delta_p_mw",
                "actual_delta_p_mw",
                "requested_direction",
                "accepted_direction",
                "actual_direction",
                "setting_vs_nn_judgement",
            ],
            max_rows=40,
        ),
        "",
        "## 9. 没有净吸收但成本压过收入的最严重 step",
        "",
        _table(
            worst_cost,
            [
                "episode",
                "step",
                "agent_id",
                "market_price",
                "delivered_p_mw",
                "energy_market_revenue",
                "der_operation_cost",
                "visible_energy_minus_operation_cost",
                "economic_operational_surplus",
                "service_quality_penalty_total",
                "private_profit_proxy",
                "private_profit_vs_visible_energy_residual",
                "dispatch_private_profit_reward",
                "baseline_p_mw",
                "requested_delta_p_mw",
                "accepted_delta_p_mw",
                "actual_delta_p_mw",
                "requested_direction",
                "accepted_direction",
                "actual_direction",
                "setting_vs_nn_judgement",
            ],
            max_rows=40,
        ),
        "",
        "## 10. 判断口径",
        "",
        "- 若某 agent 的 `absorption_rate` 很高，并且 `total_revenue` 为负，优先怀疑场景/DOE/dispatch target 允许或鼓励净吸收。",
        "- 若 `absorption_rate` 不高但 `total_cost >> total_revenue`，优先怀疑成本函数、初始 DER 状态或 reward 经济口径不匹配。",
        "- 若 `private_profit_vs_visible_energy_residual` 极大，优先检查 settlement 口径，而不是直接归咎于电价或神经网络。",
        "- 若 trace 完整且 `service_quality_penalty_total` 很大，但 `economic_operational_surplus` 正常，说明服务质量风险应单独审计，不应再作为私有利润直接扣除。",
        "- 若 `requested_delta_p_mw` 与 `accepted_delta_p_mw` 长期把目标推向负功率，才更接近 DSO/策略动作导致吸收。",
        "- 若 `actual_delta_p_mw` 与目标相反，可能是 DER 物理约束、负荷/PV 外生曲线或 action projection 在主导。",
        "- 若 `action_landing_trace_complete = 1` 但 `actual_delta_nonzero_rate` 长期低于 10%，不建议进入 paper-long；应先修 action decoder/projection 链路。",
        "- 若 `device_to_ac_gap_mw` 或 `ac_to_actual_gap_mw` 占主导，说明安全外壳或 AC 证书正在大量接管策略；应降低不可行动作比例，而不是直接扩大网络。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path, detail_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, detail = analyze(args.trace_path, args.output_dir)
    print(f"report={report}")
    print(f"detail={detail}")


if __name__ == "__main__":
    main()
