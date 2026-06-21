from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import pandas as pd

try:
    from scripts.analyze_dispatch_absorption_rewards import (
        ACTION_LANDING_COLUMNS,
        REQUIRED_SETTLEMENT_TRACE_COLUMNS,
        REQUIRED_ACTION_LANDING_TRACE_COLUMNS,
        SETTLEMENT_BREAKDOWN_COLUMNS,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from analyze_dispatch_absorption_rewards import (  # type: ignore
        ACTION_LANDING_COLUMNS,
        REQUIRED_SETTLEMENT_TRACE_COLUMNS,
        REQUIRED_ACTION_LANDING_TRACE_COLUMNS,
        SETTLEMENT_BREAKDOWN_COLUMNS,
    )


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any, digits: int = 6) -> str:
    return f"{_float(value):.{digits}f}"


def _latest_trace(output_dir: Path) -> Path | None:
    traces = sorted(
        {
            *output_dir.glob("runs/*/train/*_dispatch_private_profit_trace_episode_*.csv"),
            *output_dir.glob("*_dispatch_private_profit_trace_episode_*.csv"),
        },
        key=lambda path: path.stat().st_mtime,
    )
    return traces[-1] if traces else None


def _algorithm_from_name(path: Path) -> str:
    name = path.name
    if name.startswith("happo_"):
        return "happo"
    if name.startswith("hatrpo_"):
        return "hatrpo"
    return name.split("_", 1)[0]


def _episode_from_frame(frame: pd.DataFrame) -> int:
    if frame.empty or "episode" not in frame:
        return -1
    return int(pd.to_numeric(frame["episode"], errors="coerce").dropna().max())


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    original_columns = set(clean.columns)
    for col in [
        "dispatch_private_profit_reward",
        "private_profit_proxy",
        "energy_market_revenue",
        "der_operation_cost",
        "visible_energy_minus_operation_cost",
        "market_energy_margin_total",
        "market_price",
        "delivered_p_mw",
        "dt_hours",
        "service_payment",
        "availability_payment",
        "contract_shortfall_mw",
        "contract_delivery_penalty",
        "dispatch_projection_penalty",
        "scaled_comfort_soc_penalty",
        "battery_degradation_cost",
        *SETTLEMENT_BREAKDOWN_COLUMNS,
        *ACTION_LANDING_COLUMNS,
    ]:
        clean[col] = _num(clean, col)
    if "action_landing_drop_reason" not in original_columns:
        clean["action_landing_drop_reason"] = "legacy_trace_missing_landing_audit"
    if "visible_energy_minus_operation_cost" not in original_columns:
        clean["visible_energy_minus_operation_cost"] = clean["energy_market_revenue"] - clean["der_operation_cost"]
    if "market_energy_margin_total" not in original_columns:
        if {"export_revenue_total", "evcs_user_revenue_total", "import_energy_cost_total"}.issubset(original_columns):
            clean["market_energy_margin_total"] = (
                clean["export_revenue_total"] + clean["evcs_user_revenue_total"] - clean["import_energy_cost_total"]
            )
        else:
            clean["market_energy_margin_total"] = clean["visible_energy_minus_operation_cost"]
    if "economic_operational_surplus" not in original_columns:
        clean["economic_operational_surplus"] = clean["private_profit_proxy"]
    if "quality_adjusted_operational_surplus" not in original_columns:
        clean["quality_adjusted_operational_surplus"] = clean["private_profit_proxy"]
    if "service_quality_penalty_total" not in original_columns:
        clean["service_quality_penalty_total"] = clean["comfort_cost_total"] + clean["unserved_penalty_total"]
    clean["settlement_trace_complete"] = float(REQUIRED_SETTLEMENT_TRACE_COLUMNS.issubset(original_columns))
    clean["action_landing_trace_complete"] = float(
        REQUIRED_ACTION_LANDING_TRACE_COLUMNS.issubset(original_columns)
    )
    clean["private_profit_vs_visible_residual"] = (
        clean["private_profit_proxy"] - clean["visible_energy_minus_operation_cost"]
    )
    return clean


def _agent_summary(frame: pd.DataFrame) -> pd.DataFrame:
    clean = _prepare_frame(frame)
    rows: list[dict[str, Any]] = []
    for (agent_id, vpp_id), group in clean.groupby(["agent_id", "vpp_id"], dropna=False):
        reward = group["dispatch_private_profit_reward"]
        proxy = group["private_profit_proxy"]
        rows.append(
            {
                "agent_id": agent_id,
                "vpp_id": vpp_id,
                "steps": int(len(group)),
                "negative_reward_steps": int((reward < 0.0).sum()),
                "negative_reward_rate": float((reward < 0.0).mean()) if len(group) else 0.0,
                "total_dispatch_private_profit_reward": float(reward.sum()),
                "mean_dispatch_private_profit_reward": float(reward.mean()),
                "min_dispatch_private_profit_reward": float(reward.min()),
                "max_dispatch_private_profit_reward": float(reward.max()),
                "total_private_profit_proxy": float(proxy.sum()),
                "mean_private_profit_proxy": float(proxy.mean()),
                "total_energy_market_revenue": float(group["energy_market_revenue"].sum()),
                "total_der_operation_cost": float(group["der_operation_cost"].sum()),
                "total_visible_energy_minus_operation_cost": float(
                    group["visible_energy_minus_operation_cost"].sum()
                ),
                "total_private_profit_vs_visible_residual": float(
                    group["private_profit_vs_visible_residual"].sum()
                ),
                "total_economic_operational_surplus": float(group["economic_operational_surplus"].sum()),
                "total_quality_adjusted_operational_surplus": float(
                    group["quality_adjusted_operational_surplus"].sum()
                ),
                "total_service_quality_penalty": float(group["service_quality_penalty_total"].sum()),
                "total_export_revenue": float(group["export_revenue_total"].sum()),
                "total_evcs_user_revenue": float(group["evcs_user_revenue_total"].sum()),
                "total_import_energy_cost": float(group["import_energy_cost_total"].sum()),
                "total_der_operating_cost": float(group["der_operating_cost_total"].sum()),
                "total_comfort_cost": float(group["comfort_cost_total"].sum()),
                "total_unserved_penalty": float(group["unserved_penalty_total"].sum()),
                "settlement_trace_complete": float(group["settlement_trace_complete"].mean()),
                "mean_market_price": float(group["market_price"].mean()),
                "mean_delivered_p_mw": float(group["delivered_p_mw"].mean()),
                "min_delivered_p_mw": float(group["delivered_p_mw"].min()),
                "max_delivered_p_mw": float(group["delivered_p_mw"].max()),
                "total_service_payment": float(group["service_payment"].sum()),
                "total_availability_payment": float(group["availability_payment"].sum()),
                "total_contract_shortfall_mw": float(group["contract_shortfall_mw"].sum()),
                "total_contract_delivery_penalty": float(group["contract_delivery_penalty"].sum()),
                "total_dispatch_projection_penalty": float(group["dispatch_projection_penalty"].sum()),
                "total_scaled_comfort_soc_penalty": float(group["scaled_comfort_soc_penalty"].sum()),
                "total_battery_degradation_cost": float(group["battery_degradation_cost"].sum()),
                "action_landing_trace_complete": float(group["action_landing_trace_complete"].mean()),
                "actual_delta_nonzero_rate": float(group["actual_delta_nonzero_flag"].mean()),
                "mean_action_landing_ratio": float(group["action_landing_ratio"].mean()),
                "total_raw_to_device_gap_mw": float(group["raw_to_device_gap_mw"].sum()),
                "total_device_to_ac_gap_mw": float(group["device_to_ac_gap_mw"].sum()),
                "total_ac_to_actual_gap_mw": float(group["ac_to_actual_gap_mw"].sum()),
                "total_accepted_to_actual_gap_mw": float(group["accepted_to_actual_gap_mw"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("total_dispatch_private_profit_reward")


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if frame.empty:
        return "无数据。"
    view = frame.loc[:, columns].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    return view.to_markdown(index=False)


def generate_report(trace_path: Path, output_dir: Path) -> Path:
    frame = _prepare_frame(pd.read_csv(trace_path))
    algorithm = _algorithm_from_name(trace_path)
    episode = _episode_from_frame(frame)
    summary = _agent_summary(frame)
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / f"dispatch_private_profit_agent_summary_{algorithm}_episode_{episode:04d}.csv"
    report_path = report_dir / f"dispatch_private_profit_report_{algorithm}_episode_{episode:04d}.md"
    summary.to_csv(summary_path, index=False)

    worst = frame.copy()
    for col in [
        "dispatch_private_profit_reward",
        "private_profit_proxy",
        "energy_market_revenue",
        "der_operation_cost",
        "market_price",
        "delivered_p_mw",
        "dt_hours",
    ]:
        if col in worst:
            worst[col] = pd.to_numeric(worst[col], errors="coerce").fillna(0.0)
    worst = worst.sort_values("dispatch_private_profit_reward").head(30)
    overall_reward = _float(frame.get("dispatch_private_profit_reward", pd.Series(dtype=float)).sum())
    overall_proxy = _float(frame.get("private_profit_proxy", pd.Series(dtype=float)).sum())
    overall_revenue = _float(frame.get("energy_market_revenue", pd.Series(dtype=float)).sum())
    overall_cost = _float(frame.get("der_operation_cost", pd.Series(dtype=float)).sum())
    overall_economic = _float(frame.get("economic_operational_surplus", pd.Series(dtype=float)).sum())
    overall_quality = _float(frame.get("service_quality_penalty_total", pd.Series(dtype=float)).sum())
    overall_residual = _float(frame.get("private_profit_vs_visible_residual", pd.Series(dtype=float)).sum())
    settlement_complete = _float(frame.get("settlement_trace_complete", pd.Series(dtype=float)).mean())
    action_landing_complete = _float(frame.get("action_landing_trace_complete", pd.Series(dtype=float)).mean())
    actual_delta_nonzero_rate = _float(frame.get("actual_delta_nonzero_flag", pd.Series(dtype=float)).mean())
    action_landing_ratio = _float(frame.get("action_landing_ratio", pd.Series(dtype=float)).mean())
    landing_by_reason = (
        frame.groupby("action_landing_drop_reason", dropna=False)
        .agg(
            rows=("step", "count"),
            actual_delta_nonzero_rate=("actual_delta_nonzero_flag", "mean"),
            mean_action_landing_ratio=("action_landing_ratio", "mean"),
            total_raw_to_device_gap_mw=("raw_to_device_gap_mw", "sum"),
            total_device_to_ac_gap_mw=("device_to_ac_gap_mw", "sum"),
            total_ac_to_actual_gap_mw=("ac_to_actual_gap_mw", "sum"),
            total_accepted_to_actual_gap_mw=("accepted_to_actual_gap_mw", "sum"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )

    lines = [
        "# 逐 Dispatch Agent 私有利润奖励报告",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 实验目录：`{output_dir}`",
        f"- 数据文件：`{trace_path}`",
        f"- 算法：`{algorithm}`",
        f"- episode：`{episode}`",
        f"- step 行数：`{len(frame)}`",
        f"- settlement trace 完整性：`{settlement_complete:.0f}`",
        f"- action landing trace 完整性：`{action_landing_complete:.0f}`",
        "",
        "## 1. 私有利润奖励公式",
        "",
        "对每个 dispatch agent、每个 step，当前记录按两套口径同时落盘：",
        "",
        "1. 可见电费运行差额 = 能量市场收入扣除聚合层 DER 运行成本。",
        "",
        "   能量市场收入 = 市场电价 × VPP 实际交付有功功率 × 单步时长。",
        "",
        "   说明：如果 `delivered_p_mw` 为负，表示该 VPP 在该 step 净吸收/购入功率，则这一项会是负数。",
        "",
        "2. 经济运行盈余 = DER 逐项出口收入 + EV 用户充电收入 - DER 逐项购电成本 - DER 运行成本 - 储能退化成本。",
        "",
        "   这是主训练口径下 dispatch 私有利润的核心来源。",
        "",
        "3. 服务质量惩罚 = HVAC 舒适度惩罚 + EV 未满足需求惩罚。",
        "",
        "   它用于质量审计或单独加权，不应再直接混入经济运行盈余。",
        "",
        "4. dispatch 私有利润奖励 = 私有利润权重 × 私有利润代理。",
        "",
        "   `dispatch_private_profit_reward = private_profit_weight * private_profit_proxy`",
        "",
        "所以私有利润奖励为负，要先看经济运行盈余，再看服务质量惩罚和旧 trace 残差，不能只看聚合净功率电费。",
        "",
        "## 2. 全 episode 总览",
        "",
        f"- 全体 dispatch agent 私有利润奖励合计：`{_fmt(overall_reward)}`",
        f"- 全体 private_profit_proxy 合计：`{_fmt(overall_proxy)}`",
        f"- 全体能量市场收入合计：`{_fmt(overall_revenue)}`",
        f"- 全体 DER 运行成本合计：`{_fmt(overall_cost)}`",
        f"- 全体经济运行盈余合计：`{_fmt(overall_economic)}`",
        f"- 全体服务质量惩罚合计：`{_fmt(overall_quality)}`",
        f"- private profit 与可见电费运行差额残差合计：`{_fmt(overall_residual)}`",
        f"- actual_delta_nonzero_rate：`{_fmt(actual_delta_nonzero_rate)}`",
        f"- mean_action_landing_ratio：`{_fmt(action_landing_ratio)}`",
        f"- agent 汇总 CSV：`{summary_path}`",
        "",
        "## 3. 每个 Dispatch Agent 汇总",
        "",
        _markdown_table(
            summary,
            [
                "agent_id",
                "vpp_id",
                "steps",
                "negative_reward_steps",
                "negative_reward_rate",
                "total_dispatch_private_profit_reward",
                "mean_dispatch_private_profit_reward",
                "total_energy_market_revenue",
                "total_der_operation_cost",
                "total_economic_operational_surplus",
                "total_service_quality_penalty",
                "total_private_profit_vs_visible_residual",
                "mean_delivered_p_mw",
                "actual_delta_nonzero_rate",
                "mean_action_landing_ratio",
            ],
        ),
        "",
        "## 4. 动作落地审计",
        "",
        "如果 `actual_delta_nonzero_rate` 很低，说明 actor 的解码动作没有稳定改变真实执行功率；如果 `action_landing_ratio` 很低，说明 decoded 动作大量被投影、AC shield 或实际执行链路吃掉。",
        "",
        _markdown_table(
            landing_by_reason,
            [
                "action_landing_drop_reason",
                "rows",
                "actual_delta_nonzero_rate",
                "mean_action_landing_ratio",
                "total_raw_to_device_gap_mw",
                "total_device_to_ac_gap_mw",
                "total_ac_to_actual_gap_mw",
                "total_accepted_to_actual_gap_mw",
            ],
        ),
        "",
        "## 5. 私有利润奖励最低的 step-agent 明细",
        "",
        _markdown_table(
            worst,
            [
                "episode",
                "step",
                "agent_id",
                "vpp_id",
                "market_price",
                "delivered_p_mw",
                "dt_hours",
                "energy_market_revenue",
                "der_operation_cost",
                "economic_operational_surplus",
                "service_quality_penalty_total",
                "private_profit_vs_visible_residual",
                "private_profit_proxy",
                "private_profit_weight",
                "dispatch_private_profit_reward",
                "decoded_delta_p_mw",
                "actual_delta_p_mw",
                "action_landing_ratio",
                "action_landing_drop_reason",
            ],
            max_rows=30,
        ),
        "",
        "## 6. 读数建议",
        "",
        "- 如果某个 agent 的 `negative_reward_rate` 接近 1，说明它几乎每步的私有利润项都在惩罚。",
        "- 如果 `total_energy_market_revenue` 本身为负，优先检查该 VPP 是否长期处于净吸收功率状态。",
        "- 如果 `energy_market_revenue` 为正但 `private_profit_proxy` 为负，优先检查 DER 运行成本是否过高。",
        "- 如果 `dispatch_private_profit_reward` 绝对值很小，不代表经济行为不重要，可能只是 `private_profit_weight` 较小。",
        "- 如果 `actual_delta_nonzero_rate < 0.10` 或 `mean_action_landing_ratio < 0.30`，不要直接进入 paper-long，应先查 action decoder、FR/DOE、AC shield 或 baseline override。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        trace = _latest_trace(args.output_dir)
        if trace is not None:
            report = generate_report(trace, args.output_dir)
            print(f"generated_report={report}", flush=True)
            return
        if args.once:
            raise SystemExit("No dispatch private-profit episode trace found yet.")
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    main()
