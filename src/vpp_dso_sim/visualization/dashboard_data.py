from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pandapower as pp

from vpp_dso_sim.entities.schemas import VPPPortfolio
from vpp_dso_sim.entities.vpp import VPPAggregator
from vpp_dso_sim.envs.observations import privacy_visibility_records
from vpp_dso_sim.learning.agent_roles import build_agent_role_map, build_encoder_role_map
from vpp_dso_sim.learning.rl_architecture import build_rl_architecture_frames
from vpp_dso_sim.optimization.feasibility_region import compute_static_feasible_region
from vpp_dso_sim.utils.io import ensure_dir
from vpp_dso_sim.visualization.dispatch_explanations import build_vpp_dispatch_explanations
from vpp_dso_sim.visualization.rl_algorithm_variants import build_rl_algorithm_variant_frame
from vpp_dso_sim.visualization.topology_layout import deterministic_feeder_layout


def frame_or_empty(frames: dict[str, pd.DataFrame], name: str) -> pd.DataFrame:
    return frames.get(name, pd.DataFrame()).copy()


DEEP_RL_FRAME_NAMES: tuple[str, ...] = (
    "deep_rl_training_summary",
    "deep_rl_episode_metrics",
    "deep_rl_step_metrics",
    "deep_rl_trajectory",
    "deep_rl_loss_metrics",
)


def _first_value(frame: pd.DataFrame, column: str, default: str = "n/a") -> str:
    if frame.empty or column not in frame.columns:
        return default
    value = frame.iloc[0].get(column, default)
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def model_update_summary_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize the currently exported learning stack for every UI surface.

    Static HTML pages and the Dash app consume this one table so algorithm
    changes made by training scripts are visible everywhere after report refresh.
    """

    overview = frame_or_empty(frames, "rl_algorithm_overview")
    deep_summary = frame_or_empty(frames, "deep_rl_training_summary")
    deep_losses = frame_or_empty(frames, "deep_rl_loss_metrics")
    algorithm = _first_value(
        overview,
        "algorithm_id",
        _first_value(deep_summary, "algorithm", "proto_ctde_interface_not_full_ctde"),
    )
    ctde_status = _first_value(overview, "ctde_status", "not_exported")
    target_ctde_status = _first_value(overview, "target_ctde_status", "not_exported")
    actor_scope = _first_value(overview, "actor_scope", "not_exported")
    critic_scope = _first_value(overview, "critic_scope", "not_exported")
    reward_formula = _first_value(overview, "reward_formula", "not_exported")
    loss_formula = _first_value(overview, "loss_formula", "not_exported")
    checkpoint = _first_value(deep_summary, "checkpoint", "outputs/deep_rl/<not_generated>")
    optimizer_steps = _first_value(deep_summary, "optimizer_steps", "0")
    best_reward = _first_value(deep_summary, "best_episode_reward", "n/a")
    final_reward = _first_value(deep_summary, "final_episode_reward", "n/a")
    privacy_flag = _first_value(deep_summary, "privacy_separated_execution", "n/a")
    shared_encoder_flag = _first_value(deep_summary, "dso_vpp_shared_encoder", "n/a")
    has_separate_losses = {
        "dso_policy_loss",
        "vpp_dispatch_policy_loss",
        "portfolio_policy_loss",
    }.issubset(set(deep_losses.columns))
    loss_export = (
        "separate_dso_vpp_dispatch_portfolio_losses"
        if has_separate_losses
        else "shared_policy_value_entropy_loss_columns"
    )

    rows = [
        {
            "display_order": 1,
            "update_area": "algorithm",
            "current_value": algorithm,
            "current_value_zh": f"当前训练算法为 {algorithm}",
            "explanation": "Main reports, architecture cards and first-person pages now read the same algorithm id.",
            "explanation_zh": "主报告、架构卡片和第一视角页面现在读取同一个算法标识，避免页面不同步。",
            "evidence_file": "outputs/deep_rl/deep_rl_training_summary.csv",
        },
        {
            "display_order": 2,
            "update_area": "privacy_and_ctde",
            "current_value": f"privacy_separated_execution={privacy_flag}; shared_encoder={shared_encoder_flag}; ctde={ctde_status}",
            "current_value_zh": f"隐私分离执行={privacy_flag}；是否共享编码器={shared_encoder_flag}；CTDE状态={ctde_status}",
            "explanation": "DSO and VPP policies are rendered as separate execution-side actors; centralized critic stays in training.",
            "explanation_zh": "DSO 与 VPP 策略在执行侧按独立智能体展示；集中 critic 只属于训练侧。",
            "evidence_file": "outputs/dashboard_data/rl_algorithm_overview.csv",
        },
        {
            "display_order": 3,
            "update_area": "actor_stack",
            "current_value": actor_scope,
            "current_value_zh": "DSO全局包络策略、VPP解聚合策略、VPP慢周期组合策略分层展示。",
            "explanation": "The UI should show which modules are trainable RL actors and which modules are rule/safety guards.",
            "explanation_zh": "界面会区分哪些模块是真正可训练的RL actor，哪些模块是规则或安全投影保护层。",
            "evidence_file": "outputs/dashboard_data/rl_agent_architecture.csv",
        },
        {
            "display_order": 4,
            "update_area": "critic_and_loss",
            "current_value": f"{critic_scope}; loss_export={loss_export}; loss_formula={loss_formula}",
            "current_value_zh": f"critic范围={critic_scope}；损失导出={loss_export}；损失函数={loss_formula}",
            "explanation": "Loss tables expose DSO, VPP-dispatch and portfolio losses when the privacy-separated trainer is used.",
            "explanation_zh": "使用隐私分离训练器时，损失表会展示 DSO、VPP解聚合、组合配置三个策略损失。",
            "evidence_file": "outputs/deep_rl/deep_rl_loss_metrics.csv",
        },
        {
            "display_order": 5,
            "update_area": "reward_and_checkpoint",
            "current_value": f"reward={reward_formula}; best={best_reward}; final={final_reward}; steps={optimizer_steps}",
            "current_value_zh": f"奖励={reward_formula}；最好回合={best_reward}；最终回合={final_reward}；优化步数={optimizer_steps}",
            "explanation": "Training progress and checkpoint path are surfaced beside the architecture, not hidden in terminal output.",
            "explanation_zh": "训练进度和checkpoint路径会在架构页面旁展示，不再只停留在终端输出里。",
            "evidence_file": checkpoint,
        },
        {
            "display_order": 6,
            "update_area": "ui_refresh_contract",
            "current_value": "interactive_report.html; rl_architecture.html; vpp_first_person/*.html; dashboard_data/model_update_summary.csv",
            "current_value_zh": "主交互报告、RL架构页、VPP第一视角页面、Dash数据表会同步刷新。",
            "explanation": "Run the simulation/report refresh or the training script without --skip-report-refresh after changing algorithms.",
            "explanation_zh": "算法更新后运行仿真报告刷新流程，或运行训练脚本且不要使用 --skip-report-refresh，即可同步所有静态HTML与Dash数据。",
            "evidence_file": "examples/10_train_deep_rl.py",
        },
        {
            "display_order": 7,
            "update_area": "algorithm_variants",
            "current_value": "HAPPO = DSO actor + per-VPP dispatch actors + per-VPP slow-loop portfolio actors + sequential update + importance correction; MATD3 = DSO twin Q + per-VPP twin Q heads + replay; HASAC = soft actor + entropy temperature + twin soft Q + replay",
            "current_value_zh": "HAPPO = DSO actor + 每个 VPP 独立调度 actor + 每个 VPP 独立慢周期组合配置 actor + 顺序更新 + importance correction；MATD3 = DSO 双 Q + 按 VPP 拆分的双 Q 头 + replay；HASAC = soft actor + 熵温度 + twin soft Q + replay",
            "explanation": "A dedicated comparison matrix is now exported so HTML pages can explain how HAPPO, MATD3 and HASAC differ instead of only showing the current primary trainer.",
            "explanation_zh": "现在会额外导出一张算法对照矩阵，让 HTML 页面不仅展示当前主训练器，也能明确解释 HAPPO、MATD3 和 HASAC 的架构差异。",
            "evidence_file": "outputs/dashboard_data/rl_algorithm_variants.csv",
        },
        {
            "display_order": 8,
            "update_area": "target_ctde_status",
            "current_value": target_ctde_status,
            "current_value_zh": f"目标CTDE实现状态：{target_ctde_status}",
            "explanation": "This row makes it explicit whether the displayed target architecture is already implemented or only a roadmap item.",
            "explanation_zh": "这一行明确说明当前显示的目标架构是已经实现，还是仍只是路线图。",
            "evidence_file": "outputs/dashboard_data/rl_target_ctde_architecture.csv",
        },
    ]
    return pd.DataFrame(rows)


def make_dashboard_summary(results: dict[str, pd.DataFrame]) -> dict[str, float | int | None]:
    bus = results.get("bus_voltage", pd.DataFrame())
    line = results.get("line_loading", pd.DataFrame())
    bus_values = bus.drop(columns=["step"], errors="ignore")
    line_values = line.drop(columns=["step"], errors="ignore")
    return {
        "steps": int(len(bus)),
        "min_voltage": float(bus_values.min().min()) if not bus_values.empty else None,
        "max_line_loading": float(line_values.max().max()) if not line_values.empty else None,
    }


def network_nodes_frame(net: pp.pandapowerNet, vpps: list[VPPAggregator]) -> pd.DataFrame:
    layout = deterministic_feeder_layout(net)
    slack_buses = set(int(bus) for bus in net.ext_grid["bus"].tolist()) if len(net.ext_grid) else set()
    pcc_by_bus: dict[int, list[str]] = {}
    asset_count_by_bus: dict[int, int] = {}
    for vpp in vpps:
        pcc_by_bus.setdefault(int(vpp.pcc_bus), []).append(vpp.id)
        for der in vpp.der_list:
            asset_count_by_bus[int(der.bus)] = asset_count_by_bus.get(int(der.bus), 0) + 1

    rows = []
    for bus_id, row in net.bus.iterrows():
        idx = int(bus_id)
        rows.append(
            {
                "bus_id": idx,
                "name": str(row.get("name", f"bus_{idx}")),
                "vn_kv": float(row.get("vn_kv", 0.0)),
                "is_slack": bool(idx in slack_buses),
                "is_pcc": bool(idx in pcc_by_bus),
                "vpp_ids": ",".join(pcc_by_bus.get(idx, [])),
                "asset_count": int(asset_count_by_bus.get(idx, 0)),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.merge(layout, on="bus_id", how="left")
    return frame


def _bus_vn_kv(net: pp.pandapowerNet, bus_id: int) -> float:
    if int(bus_id) not in net.bus.index:
        return 0.0
    return float(net.bus.at[int(bus_id), "vn_kv"])


def _voltage_level_label(from_vn_kv: float, to_vn_kv: float) -> str:
    if abs(float(from_vn_kv) - float(to_vn_kv)) < 1e-6:
        return f"{float(from_vn_kv):.2f} kV"
    return f"{float(from_vn_kv):.2f}/{float(to_vn_kv):.2f} kV"


def network_edges_frame(net: pp.pandapowerNet) -> pd.DataFrame:
    rows = []
    for line_idx, row in net.line.iterrows():
        from_bus = int(row["from_bus"])
        to_bus = int(row["to_bus"])
        from_vn_kv = _bus_vn_kv(net, from_bus)
        to_vn_kv = _bus_vn_kv(net, to_bus)
        rows.append(
            {
                "edge_id": f"line_{int(line_idx)}",
                "edge_type": "line",
                "pp_index": int(line_idx),
                "from_bus": from_bus,
                "to_bus": to_bus,
                "name": str(row.get("name", f"line_{int(line_idx)}")),
                "from_vn_kv": from_vn_kv,
                "to_vn_kv": to_vn_kv,
                "nominal_vn_kv": min(from_vn_kv, to_vn_kv) if from_vn_kv and to_vn_kv else max(from_vn_kv, to_vn_kv),
                "voltage_level_label": _voltage_level_label(from_vn_kv, to_vn_kv),
                "rating": float(row.get("max_i_ka", 0.0)),
                "rating_unit": "kA",
            }
        )
    for trafo_idx, row in net.trafo.iterrows():
        from_bus = int(row["hv_bus"])
        to_bus = int(row["lv_bus"])
        from_vn_kv = _bus_vn_kv(net, from_bus)
        to_vn_kv = _bus_vn_kv(net, to_bus)
        rows.append(
            {
                "edge_id": f"trafo_{int(trafo_idx)}",
                "edge_type": "trafo",
                "pp_index": int(trafo_idx),
                "from_bus": from_bus,
                "to_bus": to_bus,
                "name": str(row.get("name", f"trafo_{int(trafo_idx)}")),
                "from_vn_kv": from_vn_kv,
                "to_vn_kv": to_vn_kv,
                "nominal_vn_kv": min(from_vn_kv, to_vn_kv) if from_vn_kv and to_vn_kv else max(from_vn_kv, to_vn_kv),
                "voltage_level_label": _voltage_level_label(from_vn_kv, to_vn_kv),
                "rating": float(row.get("sn_mva", 0.0)),
                "rating_unit": "MVA",
            }
        )
    return pd.DataFrame(rows)


def asset_registry_frame(vpps: list[VPPAggregator]) -> pd.DataFrame:
    rows = []
    for vpp in vpps:
        for der in vpp.der_list:
            rows.append(
                {
                    "der_id": der.id,
                    "name": der.name,
                    "vpp_id": vpp.id,
                    "vpp_name": vpp.name,
                    "bus_id": int(der.bus),
                    "der_type": der.__class__.__name__,
                    "controllable": bool(der.controllable),
                    "pp_element_type": der.pp_element_type,
                    "pp_element_index": der.pp_element_index,
                    "p_min_mw": float(der.p_min_mw),
                    "p_max_mw": float(der.p_max_mw),
                    "q_min_mvar": float(der.q_min_mvar),
                    "q_max_mvar": float(der.q_max_mvar),
                    "cost_coefficients": json.dumps(der.cost_coefficients),
                    "metadata_json": json.dumps(der.metadata, default=str, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def vpp_portfolio_frame(vpps: list[VPPAggregator], t: int = 0) -> pd.DataFrame:
    rows = []
    for vpp in vpps:
        portfolio = VPPPortfolio.from_vpp(vpp, t)
        row = portfolio.to_dict()
        row["connection_buses"] = ",".join(str(bus) for bus in portfolio.connection_buses)
        row["zone_ids"] = ",".join(portfolio.zone_ids)
        row["der_ids"] = ",".join(portfolio.der_ids)
        rows.append(row)
    return pd.DataFrame(rows)


def vpp_portfolio_history_frame(
    results: dict[str, pd.DataFrame],
    vpps: list[VPPAggregator],
    steps: list[int],
) -> pd.DataFrame:
    recorded = results.get("vpp_portfolio_history", pd.DataFrame()).copy()
    if not recorded.empty:
        return recorded.sort_values(["step", "vpp_id"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for step in steps:
        for vpp in vpps:
            portfolio = VPPPortfolio.from_vpp(vpp, int(step))
            rows.append(
                {
                    "step": int(step),
                    "time_label": f"{float(step):05.2f}",
                    "vpp_id": portfolio.vpp_id,
                    "portfolio_version": portfolio.portfolio_version,
                    "physical_mode": portfolio.physical_mode,
                    "pcc_bus_id": portfolio.pcc_bus_id,
                    "connection_buses": ",".join(str(bus) for bus in portfolio.connection_buses),
                    "zone_ids": ",".join(portfolio.zone_ids),
                    "der_ids": ",".join(portfolio.der_ids),
                    "der_count": len(portfolio.der_ids),
                    "connection_bus_count": len(portfolio.connection_buses),
                    "max_import_mw": portfolio.max_import_mw,
                    "max_export_mw": portfolio.max_export_mw,
                }
            )
    return pd.DataFrame(rows)


def portfolio_change_log_frame(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = results.get("portfolio_change_log", pd.DataFrame()).copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "effective_step",
                "time_label",
                "reason",
                "der_id",
                "from_vpp_id",
                "to_vpp_id",
                "old_version",
                "new_version",
                "bus_id",
                "pp_element_type",
                "pp_element_index",
                "zone_id",
                "physical_bus_unchanged",
                "physical_element_unchanged",
            ]
        )
    return frame.sort_values(["effective_step", "event_id"]).reset_index(drop=True)


def feasible_region_frame(vpps: list[VPPAggregator], steps: list[int]) -> pd.DataFrame:
    rows = []
    for step in steps:
        for vpp in vpps:
            rows.extend(compute_static_feasible_region(vpp, int(step)).to_records())
    if not rows:
        return pd.DataFrame(
            columns=[
                "fr_id",
                "vpp_id",
                "time_index",
                "scope",
                "representation",
                "element_id",
                "p_min_mw",
                "p_max_mw",
                "q_min_mvar",
                "q_max_mvar",
                "source_method",
                "portfolio_version",
            ]
        )
    return pd.DataFrame(rows).sort_values(["time_index", "vpp_id", "element_id"]).reset_index(drop=True)


def fr_envelope_state_frame(
    results: dict[str, pd.DataFrame],
    vpps: list[VPPAggregator],
    steps: list[int],
) -> pd.DataFrame:
    recorded = results.get("fr_envelope_state", pd.DataFrame()).copy()
    if not recorded.empty:
        return recorded.sort_values(["step", "vpp_id", "scope_type", "scope_id", "variable"]).reset_index(drop=True)

    rows = []
    for step in steps:
        for vpp in vpps:
            fr = compute_static_feasible_region(vpp, int(step))
            for element_id, bounds in fr.bounds.items():
                scope_type, _, scope_id = element_id.partition("_")
                for variable, lower, upper, unit in [
                    ("p_mw", bounds.p_min_mw, bounds.p_max_mw, "MW"),
                    ("q_mvar", bounds.q_min_mvar, bounds.q_max_mvar, "MVAr"),
                ]:
                    rows.append(
                        {
                            "fr_id": fr.fr_id,
                            "step": int(step),
                            "time_label": f"{float(step):05.2f}",
                            "vpp_id": vpp.id,
                            "portfolio_version": fr.portfolio_version,
                            "physical_mode": fr.metadata.get("physical_mode", vpp.physical_mode()),
                            "representation": fr.representation,
                            "source_method": fr.source_method,
                            "scope_type": scope_type,
                            "scope_id": scope_id,
                            "bus_id": int(scope_id) if scope_type in {"bus", "pcc"} and scope_id.isdigit() else None,
                            "zone_id": scope_id if scope_type == "zone" else "",
                            "variable": variable,
                            "lower_bound": lower,
                            "upper_bound": upper,
                            "current_value": None,
                            "projected_value": None,
                            "unit": unit,
                            "safety_margin": fr.safety_margin_mw if variable == "p_mw" else fr.safety_margin_mvar,
                            "bound_basis": "local_der_bounds",
                            "is_binding": False,
                        }
                    )
    return pd.DataFrame(rows)


def projection_trace_frame(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = results.get("projection_trace", pd.DataFrame()).copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "trace_id",
                "step",
                "time_label",
                "vpp_id",
                "fr_id",
                "command_source",
                "stage_order",
                "stage_name",
                "scope_type",
                "scope_id",
                "p_mw",
                "q_mvar",
                "p_lower_mw",
                "p_upper_mw",
                "q_lower_mvar",
                "q_upper_mvar",
                "delta_p_mw",
                "delta_q_mvar",
                "was_projected",
                "active_constraint",
                "projection_reason",
                "pp_element_type",
                "pp_element_index",
            ]
        )
    return frame.sort_values(["step", "vpp_id", "stage_order", "scope_id"]).reset_index(drop=True)


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _step_window(profile: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    if profile.empty or "step" not in profile:
        return pd.DataFrame()
    return profile[(profile["step"].astype(int) >= int(start)) & (profile["step"].astype(int) <= int(end))].copy()


def _mean_or_default(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if frame.empty or column not in frame or frame[column].dropna().empty:
        return float(default)
    return float(frame[column].mean())


def _profile_summary_json(profile_window: pd.DataFrame) -> str:
    summary: dict[str, dict[str, float]] = {}
    for column in ("price", "load_scale", "pv_forecast_factor"):
        if column in profile_window and not profile_window[column].dropna().empty:
            series = profile_window[column].astype(float)
            summary[column] = {
                "min": round(float(series.min()), 4),
                "mean": round(float(series.mean()), 4),
                "max": round(float(series.max()), 4),
            }
    return _json_dumps(summary)


def _grid_need(price: float, load_scale: float, pv_factor: float) -> tuple[str, float, str]:
    if price >= 100.0:
        return "high_price_export_request", min(1.0, (price - 90.0) / 40.0), "export_up"
    if price <= 55.0:
        return "low_price_absorption_request", min(1.0, (60.0 - price) / 30.0), "absorb_down"
    if load_scale >= 1.08:
        return "high_load_voltage_support", min(1.0, (load_scale - 1.0) / 0.25), "export_up"
    if pv_factor >= 0.65:
        return "pv_rich_local_absorption", min(1.0, pv_factor), "absorb_down"
    return "normal_balancing", 0.25, "balance"


def _portfolio_for_window(
    history: pd.DataFrame,
    vpp_id: str,
    start: int,
    end: int,
) -> dict[str, object]:
    if history.empty:
        return {}
    selected = history[
        (history["vpp_id"].astype(str) == str(vpp_id))
        & (history["step"].astype(int) >= int(start))
        & (history["step"].astype(int) <= int(end))
    ]
    if selected.empty:
        selected = history[history["vpp_id"].astype(str) == str(vpp_id)].head(1)
    if selected.empty:
        return {}
    return selected.iloc[-1].to_dict()


def _window_fr_bounds_json(envelope: pd.DataFrame, vpp_id: str, start: int, end: int) -> tuple[str, str, str, str]:
    selected = envelope[
        (envelope["vpp_id"].astype(str) == str(vpp_id))
        & (envelope["step"].astype(int) >= int(start))
        & (envelope["step"].astype(int) <= int(end))
    ] if not envelope.empty and {"vpp_id", "step"}.issubset(envelope.columns) else pd.DataFrame()
    if selected.empty:
        return "{}", "", "", ""
    p_rows = selected[selected["variable"].astype(str) == "p_mw"] if "variable" in selected else selected
    compact: list[dict[str, object]] = []
    binding = p_rows.copy()
    if not p_rows.empty:
        for _, row in p_rows.head(12).iterrows():
            compact.append(
                {
                    "scope_type": row.get("scope_type", ""),
                    "scope_id": row.get("scope_id", ""),
                    "lower": round(float(row.get("lower_bound", 0.0)), 5),
                    "upper": round(float(row.get("upper_bound", 0.0)), 5),
                    "current": round(float(row.get("current_value", 0.0)), 5),
                }
            )
        if {"current_value", "lower_bound", "upper_bound"}.issubset(p_rows.columns):
            binding = p_rows.assign(
                margin=p_rows.apply(
                    lambda item: min(
                        abs(float(item.get("current_value", 0.0)) - float(item.get("lower_bound", 0.0))),
                        abs(float(item.get("upper_bound", 0.0)) - float(item.get("current_value", 0.0))),
                    ),
                    axis=1,
                )
            ).sort_values("margin")
    first = binding.iloc[0] if not binding.empty else selected.iloc[0]
    return (
        _json_dumps(compact),
        str(first.get("scope_type", "")),
        str(first.get("scope_id", "")),
        str(first.get("zone_id", "")),
    )


def _projection_summary_json(projection: pd.DataFrame, vpp_id: str, start: int, end: int) -> tuple[str, float, float, str, bool]:
    selected = projection[
        (projection["vpp_id"].astype(str) == str(vpp_id))
        & (projection["step"].astype(int) >= int(start))
        & (projection["step"].astype(int) <= int(end))
    ] if not projection.empty and {"vpp_id", "step"}.issubset(projection.columns) else pd.DataFrame()
    if selected.empty:
        return "{}", 0.0, 0.0, "", False
    raw = selected[selected["stage_name"].astype(str) == "raw_action"]
    projected = selected[selected["stage_name"].astype(str) == "fr_doe"]
    writes = selected[selected["stage_name"].astype(str) == "pandapower_write"]
    raw_p = _mean_or_default(raw, "p_mw", 0.0)
    projected_p = _mean_or_default(projected, "p_mw", raw_p)
    reason = ""
    if "projection_reason" in projected and not projected["projection_reason"].dropna().empty:
        reason = str(projected["projection_reason"].dropna().iloc[0])
    was_projected = bool(projected.get("was_projected", pd.Series(dtype=bool)).astype(bool).any()) if not projected.empty else False
    return (
        _json_dumps(
            {
                "raw_target_p_mw": round(raw_p, 5),
                "projected_target_p_mw": round(projected_p, 5),
                "der_write_count": int(len(writes)),
                "projected": was_projected,
            }
        ),
        raw_p,
        projected_p,
        reason,
        was_projected,
    )


def _der_summary_json(der_state: pd.DataFrame, vpp_id: str, start: int, end: int) -> str:
    selected = der_state[
        (der_state["vpp_id"].astype(str) == str(vpp_id))
        & (der_state["step"].astype(int) >= int(start))
        & (der_state["step"].astype(int) <= int(end))
    ] if not der_state.empty and {"vpp_id", "step"}.issubset(der_state.columns) else pd.DataFrame()
    if selected.empty:
        return "{}"
    rows: list[dict[str, object]] = []
    for der_type, group in selected.groupby("der_type", dropna=False):
        rows.append(
            {
                "der_type": str(der_type),
                "asset_count": int(group["der_id"].nunique()) if "der_id" in group else int(len(group)),
                "avg_p_mw": round(_mean_or_default(group, "p_mw", 0.0), 5),
                "avg_available_p_mw": round(_mean_or_default(group, "available_p_mw", 0.0), 5),
            }
        )
    return _json_dumps(rows)


def vpp_first_person_timeline_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    vpp_state = frame_or_empty(frames, "vpp_state")
    profile = frame_or_empty(frames, "profile_state")
    history = frame_or_empty(frames, "vpp_portfolio_history")
    envelope = frame_or_empty(frames, "fr_envelope_state")
    projection = frame_or_empty(frames, "projection_trace")
    der_state = frame_or_empty(frames, "der_state")
    if vpp_state.empty or "vpp_id" not in vpp_state:
        return pd.DataFrame()

    max_step = int(vpp_state["step"].max()) if "step" in vpp_state else 0
    rows: list[dict[str, object]] = []
    for vpp_id in sorted(vpp_state["vpp_id"].astype(str).unique()):
        windows: list[tuple[str, str, int, int]] = []
        day_end = min(max_step, 95)
        windows.append(("day_ahead", "day_ahead_0", 0, day_end))
        intraday_width = 24
        for start in range(0, max_step + 1, intraday_width):
            end = min(max_step, start + intraday_width - 1)
            windows.append(("intraday", f"intraday_{start:04d}_{end:04d}", start, end))

        for phase, window_id, start, end in windows:
            profile_window = _step_window(profile, start, end)
            vpp_window = vpp_state[
                (vpp_state["vpp_id"].astype(str) == vpp_id)
                & (vpp_state["step"].astype(int) >= start)
                & (vpp_state["step"].astype(int) <= end)
            ]
            if vpp_window.empty:
                continue
            portfolio = _portfolio_for_window(history, vpp_id, start, end)
            price = _mean_or_default(profile_window, "price", 0.0)
            load = _mean_or_default(profile_window, "load_scale", 1.0)
            pv = _mean_or_default(profile_window, "pv_forecast_factor", 0.0)
            need_label, need_score, direction = _grid_need(price, load, pv)
            fr_json, binding_scope_type, binding_scope_id, high_value_zone = _window_fr_bounds_json(
                envelope, vpp_id, start, end
            )
            dispatch_json, raw_p, projected_p, projection_reason, was_projected = _projection_summary_json(
                projection, vpp_id, start, end
            )
            avg_p = _mean_or_default(vpp_window, "p_mw", 0.0)
            p_min = _mean_or_default(vpp_window, "p_min_mw", 0.0)
            p_max = _mean_or_default(vpp_window, "p_max_mw", 0.0)
            delivery_risk = min(1.0, abs(raw_p - projected_p) / max(0.01, abs(raw_p) + 0.01))
            decision_type = "portfolio_forecast_plan" if phase == "day_ahead" else "real_time_dispatch"
            rows.append(
                {
                    "vpp_id": vpp_id,
                    "vpp_name": "",
                    "phase": phase,
                    "window_id": window_id,
                    "step_start": int(start),
                    "step_end": int(end),
                    "time_label_start": f"{float(start) * 0.25:05.2f} h",
                    "time_label_end": f"{float(end + 1) * 0.25:05.2f} h",
                    "portfolio_version": str(portfolio.get("portfolio_version", "v0")),
                    "physical_mode": str(portfolio.get("physical_mode", "")),
                    "pcc_bus_id": portfolio.get("pcc_bus_id", ""),
                    "connection_buses": str(portfolio.get("connection_buses", "")),
                    "zone_ids": str(portfolio.get("zone_ids", "")),
                    "der_ids": str(portfolio.get("der_ids", "")),
                    "seen_fr_id": f"fr_{vpp_id}_{start}",
                    "seen_fr_scope": "pcc" if str(portfolio.get("physical_mode", "")) == "single_pcc" else "bus_vector",
                    "seen_fr_bounds_json": fr_json,
                    "seen_target_constraint": need_label,
                    "seen_direction": direction,
                    "seen_price_profile": _profile_summary_json(profile_window),
                    "seen_load_scale": round(load, 5),
                    "seen_pv_forecast_factor": round(pv, 5),
                    "seen_own_state_json": _json_dumps(
                        {
                            "avg_p_mw": round(avg_p, 5),
                            "p_min_mw": round(p_min, 5),
                            "p_max_mw": round(p_max, 5),
                            "physical_mode": str(portfolio.get("physical_mode", "")),
                        }
                    ),
                    "visible_fields_json": _json_dumps(["own_portfolio", "own_fr_doe", "price", "load", "pv_forecast"]),
                    "inferred_grid_need_label": need_label,
                    "inferred_grid_need_score": round(float(need_score), 5),
                    "inferred_high_value_zone": high_value_zone,
                    "inferred_binding_scope_type": binding_scope_type,
                    "inferred_binding_scope_id": binding_scope_id,
                    "inferred_delivery_risk": round(delivery_risk, 5),
                    "inferred_profit_proxy": round(avg_p * price, 5),
                    "inferred_non_delivery_risk": round(max(delivery_risk, 0.2 if was_projected else 0.0), 5),
                    "inference_source": "rule_encoder_v0",
                    "encoder_version": "bidirectional_stat_encoder_v0",
                    "decision_type": decision_type,
                    "decision_summary": (
                        f"{phase}: {need_label}; target projected from {raw_p:.4f} MW to {projected_p:.4f} MW; "
                        f"actual average aggregate power {avg_p:.4f} MW."
                    ),
                    "bid_quantity_mw_or_mvar": round(projected_p, 5),
                    "target_p_mw": round(raw_p, 5),
                    "target_q_mvar": 0.0,
                    "projected_p_mw": round(projected_p, 5),
                    "projected_q_mvar": 0.0,
                    "dispatch_instruction_json": dispatch_json,
                    "der_dispatch_summary_json": _der_summary_json(der_state, vpp_id, start, end),
                    "decision_status": "projected" if was_projected else "accepted",
                    "projection_reason": projection_reason,
                    "private_cost_used": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["vpp_id", "phase", "step_start"]).reset_index(drop=True)


def vpp_first_person_scope_detail_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    envelope = frame_or_empty(frames, "fr_envelope_state")
    assets = frame_or_empty(frames, "asset_registry")
    if envelope.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, row in envelope.iterrows():
        vpp_id = str(row.get("vpp_id", ""))
        scope_type = str(row.get("scope_type", ""))
        scope_id = str(row.get("scope_id", ""))
        bus_id = row.get("bus_id", None)
        selected_assets = pd.DataFrame()
        if not assets.empty and "vpp_id" in assets:
            selected_assets = assets[assets["vpp_id"].astype(str) == vpp_id]
            if scope_type in {"bus", "pcc"} and pd.notna(bus_id):
                selected_assets = selected_assets[selected_assets["bus_id"].astype(str) == str(int(float(bus_id)))]
            elif scope_type == "zone" and "metadata_json" in selected_assets:
                selected_assets = selected_assets[
                    selected_assets["metadata_json"].astype(str).str.contains(scope_id, regex=False)
                ]
        rows.append(
            {
                "vpp_id": vpp_id,
                "phase": "intraday",
                "step": int(row.get("step", 0)),
                "time_label": row.get("time_label", ""),
                "scope_type": scope_type,
                "scope_id": scope_id,
                "bus_id": None if pd.isna(bus_id) else int(float(bus_id)),
                "zone_id": row.get("zone_id", ""),
                "is_pcc": scope_type == "pcc",
                "is_remote_bus": scope_type == "bus" and str(row.get("physical_mode", "")) == "multi_node",
                "asset_ids": ",".join(selected_assets["der_id"].astype(str).tolist()) if not selected_assets.empty else "",
                "der_count": int(len(selected_assets)) if not selected_assets.empty else 0,
                "variable": row.get("variable", ""),
                "p_lower_mw": row.get("lower_bound", None) if row.get("variable", "") == "p_mw" else None,
                "p_upper_mw": row.get("upper_bound", None) if row.get("variable", "") == "p_mw" else None,
                "q_lower_mvar": row.get("lower_bound", None) if row.get("variable", "") == "q_mvar" else None,
                "q_upper_mvar": row.get("upper_bound", None) if row.get("variable", "") == "q_mvar" else None,
                "current_p_mw": row.get("current_value", None) if row.get("variable", "") == "p_mw" else None,
                "projected_p_mw": row.get("projected_value", None) if row.get("variable", "") == "p_mw" else None,
                "is_binding": bool(row.get("is_binding", False)),
            }
        )
    return pd.DataFrame(rows).sort_values(["vpp_id", "step", "scope_type", "scope_id"]).reset_index(drop=True)


def _select_exact_step_vpp(frame: pd.DataFrame, step: int, vpp_id: str) -> pd.DataFrame:
    if frame.empty or not {"step", "vpp_id"}.issubset(frame.columns):
        return pd.DataFrame()
    return frame[
        (frame["step"].astype(int) == int(step))
        & (frame["vpp_id"].astype(str) == str(vpp_id))
    ].copy()


def _last_portfolio_at_step(history: pd.DataFrame, step: int, vpp_id: str) -> dict[str, object]:
    selected = _select_exact_step_vpp(history, step, vpp_id)
    if selected.empty and not history.empty and {"step", "vpp_id"}.issubset(history.columns):
        selected = history[
            (history["step"].astype(int) <= int(step))
            & (history["vpp_id"].astype(str) == str(vpp_id))
        ].tail(1)
    return selected.iloc[-1].to_dict() if not selected.empty else {}


def _projection_stage_row(projection: pd.DataFrame, step: int, vpp_id: str, stage_name: str) -> dict[str, object]:
    selected = _select_exact_step_vpp(projection, step, vpp_id)
    if selected.empty or "stage_name" not in selected:
        return {}
    selected = selected[selected["stage_name"].astype(str) == stage_name]
    return selected.iloc[-1].to_dict() if not selected.empty else {}


def _float_or_default(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bool_or_false(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value) if value is not None and not pd.isna(value) else False


def _der_response_summary(der_state: pd.DataFrame, step: int, vpp_id: str) -> tuple[str, str]:
    selected = _select_exact_step_vpp(der_state, step, vpp_id)
    if selected.empty:
        return "[]", "No DER state was recorded at this step."
    top = selected.assign(abs_p=selected["p_mw"].astype(float).abs()).sort_values("abs_p", ascending=False).head(5)
    records = []
    labels = []
    for _, row in top.iterrows():
        der_type = str(row.get("der_type", "DER")).replace("Model", "")
        der_id = str(row.get("der_id", ""))
        p_mw = _float_or_default(row.get("p_mw"))
        bus_id = row.get("bus_id", "")
        records.append(
            {
                "der_id": der_id,
                "der_type": der_type,
                "bus_id": "" if pd.isna(bus_id) else int(float(bus_id)),
                "p_mw": round(p_mw, 5),
                "state_label": str(row.get("state_label", "")),
            }
        )
        labels.append(f"{der_type} {der_id} at bus {bus_id}: P={p_mw:.4f} MW")
    return _json_dumps(records), "; ".join(labels)


def _action_label(projected_p: float) -> str:
    if projected_p > 0.005:
        return "increase_export_or_reduce_load"
    if projected_p < -0.005:
        return "absorb_power_or_charge"
    return "hold_near_neutral"


def _portfolio_change_text(changes: pd.DataFrame, step: int, vpp_id: str) -> tuple[str, str]:
    if changes.empty or "effective_step" not in changes:
        return "", ""
    selected = changes[
        (changes["effective_step"].astype(int) == int(step))
        & (
            (changes.get("from_vpp_id", pd.Series(dtype=str)).astype(str) == str(vpp_id))
            | (changes.get("to_vpp_id", pd.Series(dtype=str)).astype(str) == str(vpp_id))
        )
    ]
    if selected.empty:
        return "", ""
    row = selected.iloc[0]
    text = (
        f"{row.get('event_id', '')}: DER {row.get('der_id', '')} moved from "
        f"{row.get('from_vpp_id', '')} to {row.get('to_vpp_id', '')} because {row.get('reason', '')}"
    )
    return str(row.get("event_id", "")), text


def vpp_step_decision_summary_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    vpp_state = frame_or_empty(frames, "vpp_state")
    profile = frame_or_empty(frames, "profile_state")
    history = frame_or_empty(frames, "vpp_portfolio_history")
    projection = frame_or_empty(frames, "projection_trace")
    der_state = frame_or_empty(frames, "der_state")
    changes = frame_or_empty(frames, "portfolio_change_log")
    if vpp_state.empty or not {"step", "vpp_id"}.issubset(vpp_state.columns):
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    profile_by_step = profile.set_index("step") if not profile.empty and "step" in profile else pd.DataFrame()
    for _, state in vpp_state.sort_values(["step", "vpp_id"]).iterrows():
        step = int(state.get("step", 0))
        vpp_id = str(state.get("vpp_id", ""))
        prof = profile_by_step.loc[step].to_dict() if not profile_by_step.empty and step in profile_by_step.index else {}
        price = _float_or_default(prof.get("price"), 0.0)
        load = _float_or_default(prof.get("load_scale"), 1.0)
        pv = _float_or_default(prof.get("pv_forecast_factor"), 0.0)
        need_label, need_score, direction = _grid_need(price, load, pv)
        portfolio = _last_portfolio_at_step(history, step, vpp_id)
        raw = _projection_stage_row(projection, step, vpp_id, "raw_action")
        fr = _projection_stage_row(projection, step, vpp_id, "fr_doe")
        delivered = _projection_stage_row(projection, step, vpp_id, "powerflow_result")
        raw_p = _float_or_default(raw.get("p_mw"), _float_or_default(state.get("p_mw")))
        projected_p = _float_or_default(fr.get("p_mw"), raw_p)
        actual_p = _float_or_default(delivered.get("p_mw"), _float_or_default(state.get("p_mw")))
        p_min = _float_or_default(state.get("p_min_mw"))
        p_max = _float_or_default(state.get("p_max_mw"))
        tracking_error = actual_p - projected_p
        fr_margin = min(abs(projected_p - p_min), abs(p_max - projected_p)) if p_max >= p_min else 0.0
        was_projected = _bool_or_false(fr.get("was_projected"))
        projection_reason = str(fr.get("projection_reason", "")) or str(fr.get("active_constraint", ""))
        delivery_risk = min(1.0, abs(tracking_error) / max(0.01, abs(projected_p) + 0.01))
        event_id, event_text = _portfolio_change_text(changes, step, vpp_id)
        der_json, der_text = _der_response_summary(der_state, step, vpp_id)
        if was_projected:
            decision_status = "projected_to_fr_doe"
        elif delivery_risk > 0.35:
            decision_status = "tracking_gap"
        else:
            decision_status = "accepted_and_delivered"
        rows.append(
            {
                "step": step,
                "time_label": prof.get("time_label", f"{step * 0.25:05.2f} h"),
                "vpp_id": vpp_id,
                "portfolio_version": str(portfolio.get("portfolio_version", "")),
                "physical_mode": str(portfolio.get("physical_mode", "")),
                "connection_buses": str(portfolio.get("connection_buses", "")),
                "zone_ids": str(portfolio.get("zone_ids", "")),
                "command_seen": str(raw.get("command_source", "price_driven_baseline")),
                "need_label": need_label,
                "need_score": round(float(need_score), 5),
                "belief_label": f"{need_label}:{direction}",
                "direction": direction,
                "action_label": _action_label(projected_p),
                "raw_target_p_mw": round(raw_p, 5),
                "target_p_mw": round(raw_p, 5),
                "projected_p_mw": round(projected_p, 5),
                "actual_p_mw": round(actual_p, 5),
                "tracking_error_mw": round(tracking_error, 5),
                "p_min_mw": round(p_min, 5),
                "p_max_mw": round(p_max, 5),
                "fr_margin_min_mw": round(float(fr_margin), 5),
                "delivery_risk": round(float(delivery_risk), 5),
                "price": round(price, 5),
                "load_scale": round(load, 5),
                "pv_forecast_factor": round(pv, 5),
                "signed_energy_cashflow_proxy": round(actual_p * price, 5),
                "inferred_profit_proxy": round(actual_p * price, 5),
                "decision_status": decision_status,
                "projection_reason": projection_reason,
                "portfolio_event_id": event_id,
                "portfolio_change_text": event_text,
                "top_der_response_json": der_json,
                "der_response_summary": der_text,
                "long_horizon_judgment_label": "watch_projection_frequency" if was_projected else "operate_within_envelope",
                "visible_fields_json": _json_dumps(["own_fr_doe", "own_der_state", "price", "load_scale", "pv_forecast"]),
                "private_cost_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["step", "vpp_id"]).reset_index(drop=True)


def _label_zh(value: object) -> str:
    labels = {
        "price_driven_baseline": "电价驱动基线",
        "external_action": "外部/智能体指令",
        "low_price_absorption_request": "低电价吸收请求",
        "high_price_export_request": "高电价上调注入请求",
        "high_load_voltage_support": "高负荷电压支撑请求",
        "pv_rich_local_absorption": "光伏富余本地消纳请求",
        "normal_balancing": "常规平衡",
        "absorb_down": "降低净注入/增加吸收",
        "export_up": "提高净注入/降低负荷",
        "balance": "保持平衡",
        "increase_export_or_reduce_load": "提高注入或降低负荷",
        "absorb_power_or_charge": "吸收功率或充电",
        "hold_near_neutral": "保持接近中性功率",
        "accepted_and_delivered": "目标被接受且已交付",
        "projected_to_fr_doe": "目标被 FR/DOE 可行域修正",
        "tracking_gap": "实际交付存在跟踪偏差",
    }
    return labels.get(str(value), str(value))


def _p_direction_zh(p_mw: float) -> str:
    if p_mw > 0.005:
        return "向电网注入有功功率"
    if p_mw < -0.005:
        return "从电网吸收有功功率，通常对应充电或增加可控负荷"
    return "接近零功率交换"


def _p_direction_en(p_mw: float) -> str:
    if p_mw > 0.005:
        return "inject active power into the grid"
    if p_mw < -0.005:
        return "absorb active power from the grid, usually by charging storage or increasing flexible load"
    return "stay close to zero active-power exchange"


def _first_person_event_details(row: pd.Series, event_type: str) -> tuple[str, str, str]:
    raw_target = _float_or_default(row.get("raw_target_p_mw"))
    projected = _float_or_default(row.get("projected_p_mw"))
    actual = _float_or_default(row.get("actual_p_mw"))
    p_min = _float_or_default(row.get("p_min_mw"))
    p_max = _float_or_default(row.get("p_max_mw"))
    price = _float_or_default(row.get("price"))
    load_scale = _float_or_default(row.get("load_scale"), 1.0)
    pv_factor = _float_or_default(row.get("pv_forecast_factor"))
    need_score = _float_or_default(row.get("need_score"))
    delivery_risk = _float_or_default(row.get("delivery_risk"))
    tracking_error = _float_or_default(row.get("tracking_error_mw"))
    cashflow = _float_or_default(row.get("signed_energy_cashflow_proxy"))
    command = str(row.get("command_seen", ""))
    need = str(row.get("need_label", ""))
    direction = str(row.get("direction", ""))
    action = str(row.get("action_label", ""))
    status = str(row.get("decision_status", ""))
    reason = str(row.get("projection_reason", ""))
    margin = _float_or_default(row.get("fr_margin_min_mw"))

    if event_type == "received_instruction":
        en = (
            f"The VPP received {command}. The requested aggregate target is {raw_target:.3f} MW. "
            f"With the project sign convention, this means {_p_direction_en(raw_target)}. "
            f"The current feasible active-power range is [{p_min:.3f}, {p_max:.3f}] MW."
        )
        zh = (
            f"本步 VPP 收到“{_label_zh(command)}”。原始聚合目标为 {raw_target:.3f} MW。"
            f"按照项目符号约定，P>0 表示向电网注入，P<0 表示从电网吸收；所以该目标表示："
            f"{_p_direction_zh(raw_target)}。当前 FR/DOE 可行有功范围是 [{p_min:.3f}, {p_max:.3f}] MW，"
            f"即 VPP 在物理和设备约束下允许运行的聚合功率区间。"
        )
        raw = f"command={command}; raw_target={raw_target:.5f} MW; FR=[{p_min:.5f}, {p_max:.5f}] MW"
    elif event_type == "received_context":
        en = (
            f"The visible operating context is price={price:.2f}, load scale={load_scale:.3f}, "
            f"PV forecast factor={pv_factor:.3f}."
        )
        zh = (
            f"本步 VPP 可见的外部运行条件为：电价 {price:.2f}，负荷倍率 {load_scale:.3f}，"
            f"PV 预测因子 {pv_factor:.3f}。电价用于驱动吸收/注入倾向，负荷倍率反映台区负荷压力，"
            f"PV 预测因子反映光伏可用出力水平。"
        )
        raw = f"price={price:.5f}; load_scale={load_scale:.5f}; pv_forecast_factor={pv_factor:.5f}"
    elif event_type == "formed_belief":
        en = (
            f"The VPP inferred {need}:{direction}. Need score is {need_score:.2f}; delivery risk is "
            f"{delivery_risk:.2f}."
        )
        zh = (
            f"VPP 将当前场景判断为“{_label_zh(need)}”，调节方向为“{_label_zh(direction)}”。"
            f"need_score={need_score:.2f} 表示该需求信号强度，越接近 1 越强；"
            f"delivery_risk={delivery_risk:.2f} 表示目标执行风险，越接近 0 越容易按目标交付。"
        )
        raw = f"belief={need}:{direction}; need_score={need_score:.5f}; delivery_risk={delivery_risk:.5f}"
    elif event_type == "made_dispatch_decision":
        en = f"The VPP selected {action}; projected target is {projected:.3f} MW."
        zh = (
            f"VPP 采取动作“{_label_zh(action)}”。经过可行性处理后的目标功率为 {projected:.3f} MW，"
            f"含义是：{_p_direction_zh(projected)}。后续 DER 会围绕这个聚合目标进行分解。"
        )
        raw = f"action={action}; projected_target={projected:.5f} MW"
    elif event_type == "projection_result":
        if status == "projected_to_fr_doe":
            status_text = f"目标被修正，原因是 {reason or '触及可行域边界'}。"
        else:
            status_text = "目标位于可行域内，不需要额外修正。"
        en = f"Projection status is {status}. Remaining margin to the nearest bound is {margin:.3f} MW."
        zh = (
            f"FR/DOE 投影结果为“{_label_zh(status)}”。{status_text}"
            f"距离最近功率边界的裕度约为 {margin:.3f} MW；裕度越小，说明该 VPP 越接近设备或聚合能力边界。"
        )
        raw = f"status={status}; reason={reason}; margin={margin:.5f} MW"
    elif event_type == "der_dispatch":
        response = str(row.get("der_response_summary", ""))
        en = f"The aggregate target was mapped to DER setpoints. Top responses: {response}"
        zh = (
            "VPP 将聚合目标分解到内部 DER。下面列出响应最大的若干资源，"
            f"用于判断是 PV、储能、柔性负荷、EVCS、HVAC 还是燃机在承担主要调节：{response}"
        )
        raw = response
    else:
        en = (
            f"Delivered power is {actual:.3f} MW. Tracking error is {tracking_error:.3f} MW. "
            f"Signed cashflow proxy is {cashflow:.3f}."
        )
        zh = (
            f"潮流计算后的实际聚合功率为 {actual:.3f} MW，目标跟踪误差为 {tracking_error:.3f} MW。"
            f"signed_energy_cashflow_proxy={cashflow:.3f} 是带符号能量现金流代理，不是净利润；"
            "P<0 吸收功率乘以正电价时会表现为负数。"
        )
        raw = f"actual={actual:.5f} MW; tracking_error={tracking_error:.5f} MW; signed_cashflow_proxy={cashflow:.5f}"
    return en, zh, raw


def vpp_first_person_event_stream_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = frame_or_empty(frames, "vpp_step_decision_summary")
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    event_templates = [
        ("received_instruction", "Received DSO or baseline target"),
        ("received_context", "Read local price, load and PV forecast"),
        ("formed_belief", "Inferred grid need and delivery risk"),
        ("made_dispatch_decision", "Selected aggregate VPP response"),
        ("projection_result", "Checked FR/DOE envelope projection"),
        ("der_dispatch", "Mapped target to internal DER response"),
        ("delivery_feedback", "Observed delivered power and settlement proxy"),
    ]
    for _, row in summary.iterrows():
        for order, (event_type, title) in enumerate(event_templates, start=1):
            event_detail, event_detail_zh, event_detail_raw = _first_person_event_details(row, event_type)
            if event_type == "received_instruction":
                raw_detail = (
                    f"command={row.get('command_seen', '')}; raw_target={row.get('raw_target_p_mw', 0.0)} MW; "
                    f"FR=[{row.get('p_min_mw', '')}, {row.get('p_max_mw', '')}] MW"
                )
            elif event_type == "received_context":
                raw_detail = (
                    f"price={row.get('price', '')}; load_scale={row.get('load_scale', '')}; "
                    f"pv_forecast_factor={row.get('pv_forecast_factor', '')}"
                )
            elif event_type == "formed_belief":
                raw_detail = (
                    f"belief={row.get('belief_label', '')}; need_score={row.get('need_score', '')}; "
                    f"delivery_risk={row.get('delivery_risk', '')}"
                )
            elif event_type == "made_dispatch_decision":
                raw_detail = (
                    f"action={row.get('action_label', '')}; projected_target={row.get('projected_p_mw', '')} MW"
                )
            elif event_type == "projection_result":
                raw_detail = (
                    f"status={row.get('decision_status', '')}; reason={row.get('projection_reason', '')}; "
                    f"margin={row.get('fr_margin_min_mw', '')} MW"
                )
            elif event_type == "der_dispatch":
                raw_detail = str(row.get("der_response_summary", ""))
            else:
                raw_detail = (
                    f"actual={row.get('actual_p_mw', '')} MW; tracking_error={row.get('tracking_error_mw', '')} MW; "
                    f"signed_cashflow_proxy={row.get('signed_energy_cashflow_proxy', '')}"
                )
            event_detail_raw = event_detail_raw or raw_detail
            rows.append(
                {
                    "step": int(row.get("step", 0)),
                    "time_label": row.get("time_label", ""),
                    "vpp_id": row.get("vpp_id", ""),
                    "portfolio_version": row.get("portfolio_version", ""),
                    "physical_mode": row.get("physical_mode", ""),
                    "phase": "intraday",
                    "event_order": order,
                    "event_type": event_type,
                    "event_title": title,
                    "event_detail": event_detail,
                    "event_detail_zh": event_detail_zh,
                    "event_detail_raw": event_detail_raw,
                    "received_instruction_id": f"{row.get('vpp_id', '')}_{int(row.get('step', 0))}_target",
                    "target_constraint": row.get("need_label", ""),
                    "direction": row.get("direction", ""),
                    "price": row.get("price", ""),
                    "fr_scope": "pcc" if row.get("physical_mode", "") == "single_pcc" else "bus_vector",
                    "raw_target_p_mw": row.get("raw_target_p_mw", ""),
                    "projected_p_mw": row.get("projected_p_mw", ""),
                    "delivered_p_mw": row.get("actual_p_mw", ""),
                    "deviation_mw": row.get("tracking_error_mw", ""),
                    "decision_status": row.get("decision_status", ""),
                    "projection_reason": row.get("projection_reason", ""),
                    "top_der_response_json": row.get("top_der_response_json", ""),
                    "visible_fields_json": row.get("visible_fields_json", ""),
                    "private_cost_used": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["step", "vpp_id", "event_order"]).reset_index(drop=True)


def vpp_scope_step_summary_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    scope = frame_or_empty(frames, "vpp_first_person_scope_detail")
    if scope.empty:
        return pd.DataFrame()
    p_scope = scope[scope["variable"].astype(str) == "p_mw"].copy() if "variable" in scope else scope.copy()
    if p_scope.empty:
        return pd.DataFrame()
    p_scope["asset_count"] = p_scope.get("der_count", 0)
    return p_scope[
        [
            "step",
            "time_label",
            "vpp_id",
            "scope_type",
            "scope_id",
            "bus_id",
            "zone_id",
            "asset_count",
            "p_lower_mw",
            "p_upper_mw",
            "current_p_mw",
            "projected_p_mw",
            "is_binding",
            "is_remote_bus",
        ]
    ].sort_values(["step", "vpp_id", "scope_type", "scope_id"]).reset_index(drop=True)


def vpp_long_cycle_judgment_frame(frames: dict[str, pd.DataFrame], window_steps: int = 96) -> pd.DataFrame:
    summary = frame_or_empty(frames, "vpp_step_decision_summary")
    changes = frame_or_empty(frames, "portfolio_change_log")
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (vpp_id, window_id), group in summary.groupby(
        [summary["vpp_id"].astype(str), (summary["step"].astype(int) // int(window_steps))],
        dropna=False,
    ):
        start = int(group["step"].min())
        end = int(group["step"].max())
        projection_count = int((group["decision_status"].astype(str) == "projected_to_fr_doe").sum())
        non_delivery_rate = float((group["delivery_risk"].astype(float) > 0.35).mean())
        reliability = max(0.0, min(1.0, 1.0 - float(group["delivery_risk"].astype(float).mean())))
        dominant_need = str(group["need_label"].mode().iloc[0]) if not group["need_label"].mode().empty else ""
        action_mode = str(group["action_label"].mode().iloc[0]) if not group["action_label"].mode().empty else ""
        related_changes = pd.DataFrame()
        if not changes.empty and "effective_step" in changes:
            related_changes = changes[
                (changes["effective_step"].astype(int) >= start)
                & (changes["effective_step"].astype(int) <= end)
                & (
                    (changes.get("from_vpp_id", pd.Series(dtype=str)).astype(str) == str(vpp_id))
                    | (changes.get("to_vpp_id", pd.Series(dtype=str)).astype(str) == str(vpp_id))
                )
            ]
        if projection_count > max(2, 0.15 * len(group)):
            recommendation = "expand_or_reweight_flexibility_on_binding_scopes"
            reason = "Frequent FR/DOE projection indicates the requested aggregate target often hits local limits."
        elif non_delivery_rate > 0.25:
            recommendation = "tighten_dispatch_tracking_or_reserve_margin"
            reason = "Delivered power repeatedly deviates from the projected target."
        elif float(group["signed_energy_cashflow_proxy"].mean()) < 0 and dominant_need.endswith("absorption_request"):
            recommendation = "seek_absorption_service_payment"
            reason = "The VPP is absorbing energy during low-price periods; this is a service, not pure merchant profit."
        else:
            recommendation = "keep_current_portfolio"
            reason = "Delivery risk and projection frequency are acceptable for this window."
        risk_level = "high" if non_delivery_rate > 0.25 or reliability < 0.70 else "medium" if projection_count else "low"
        rows.append(
            {
                "vpp_id": vpp_id,
                "window_id": f"cycle_{int(window_id)}",
                "period_start_step": start,
                "period_end_step": end,
                "start_time": group["time_label"].iloc[0],
                "end_time": group["time_label"].iloc[-1],
                "portfolio_version": str(group["portfolio_version"].iloc[-1]),
                "dominant_grid_need": dominant_need,
                "dominant_action": action_mode,
                "reliability_score": round(float(reliability), 5),
                "non_delivery_rate": round(float(non_delivery_rate), 5),
                "projection_count": projection_count,
                "portfolio_change_count": int(len(related_changes)),
                "avg_price": round(float(group["price"].astype(float).mean()), 5),
                "avg_p_mw": round(float(group["actual_p_mw"].astype(float).mean()), 5),
                "avg_signed_energy_cashflow_proxy": round(float(group["signed_energy_cashflow_proxy"].astype(float).mean()), 5),
                "risk_level": risk_level,
                "grid_need_belief": f"{dominant_need} with action mode {action_mode}",
                "portfolio_recommendation": recommendation,
                "recommendation_reason": reason,
                "evidence_json": _json_dumps(
                    {
                        "projection_count": projection_count,
                        "non_delivery_rate": round(non_delivery_rate, 5),
                        "reliability_score": round(reliability, 5),
                        "portfolio_events": related_changes.get("event_id", pd.Series(dtype=str)).astype(str).tolist(),
                    }
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["vpp_id", "period_start_step"]).reset_index(drop=True)


def portfolio_adjustment_story_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    changes = frame_or_empty(frames, "portfolio_change_log")
    summary = frame_or_empty(frames, "vpp_step_decision_summary")
    if changes.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, change in changes.iterrows():
        step = int(change.get("effective_step", 0))
        from_vpp = str(change.get("from_vpp_id", ""))
        to_vpp = str(change.get("to_vpp_id", ""))

        def _window(vpp_id: str, start: int, end: int) -> pd.DataFrame:
            if summary.empty:
                return pd.DataFrame()
            return summary[
                (summary["vpp_id"].astype(str) == str(vpp_id))
                & (summary["step"].astype(int) >= int(start))
                & (summary["step"].astype(int) <= int(end))
            ]

        before = _window(from_vpp, max(0, step - 24), step - 1)
        after = _window(to_vpp, step, step + 24)
        before_risk = _mean_or_default(before, "delivery_risk", 0.0)
        after_risk = _mean_or_default(after, "delivery_risk", 0.0)
        before_cash = _mean_or_default(before, "signed_energy_cashflow_proxy", 0.0)
        after_cash = _mean_or_default(after, "signed_energy_cashflow_proxy", 0.0)
        rows.append(
            {
                "event_id": change.get("event_id", ""),
                "effective_step": step,
                "time_label": change.get("time_label", ""),
                "from_vpp_id": from_vpp,
                "to_vpp_id": to_vpp,
                "der_id": change.get("der_id", ""),
                "bus_id": change.get("bus_id", ""),
                "zone_id": change.get("zone_id", ""),
                "before_version": change.get("old_version", ""),
                "after_version": change.get("new_version", ""),
                "why_now": change.get("reason", ""),
                "evidence_window": f"steps {max(0, step - 24)}-{step + 24}",
                "expected_effect": "Move flexible capacity toward the VPP receiving more local service calls.",
                "post_change_effect": (
                    f"Before risk={before_risk:.3f}, after risk={after_risk:.3f}; "
                    f"signed cashflow proxy before={before_cash:.3f}, after={after_cash:.3f}."
                ),
                "physical_bus_unchanged": bool(change.get("physical_bus_unchanged", False)),
                "physical_element_unchanged": bool(change.get("physical_element_unchanged", False)),
            }
        )
    return pd.DataFrame(rows).sort_values(["effective_step", "event_id"]).reset_index(drop=True)


def economic_explanation_frame(frames: dict[str, pd.DataFrame], dt_hours: float = 0.25) -> pd.DataFrame:
    reward = frame_or_empty(frames, "reward_components")
    summary = frame_or_empty(frames, "vpp_step_decision_summary")
    rows: list[dict[str, object]] = []
    if not reward.empty:
        numeric = reward.select_dtypes(include="number")
        total_cost = float(numeric["total_cost"].sum()) if "total_cost" in numeric else 0.0
        for column in [
            "operation_cost",
            "target_tracking_error_penalty",
            "comfort_violation_penalty",
            "soc_violation_penalty",
            "voltage_violation_penalty",
            "line_overload_penalty",
            "transformer_overload_penalty",
            "powerflow_penalty",
            "total_cost",
            "raw_objective_reward",
            "feasibility_bonus",
            "tracking_bonus",
            "reward",
        ]:
            if column not in numeric:
                continue
            value = float(numeric[column].sum())
            share = value / total_cost if total_cost and column not in {"reward"} else None
            rows.append(
                {
                    "metric": column,
                    "value": round(value, 6),
                    "share_of_total_cost": None if share is None else round(float(share), 6),
                    "formula": "sum over simulation steps",
                    "interpretation": (
                        "Raw objective reward is negative by definition because raw_objective_reward = -total_cost."
                        if column == "raw_objective_reward"
                        else "Training reward is shaped: reward = -0.05 * total_cost + feasibility_bonus + tracking_bonus; total_cost includes action_projection_penalty."
                        if column == "reward"
                        else "Positive objective-cost or penalty contribution."
                    ),
                    "why_negative": (
                        "This is not profit. It is the unshaped control objective equal to negative system cost."
                        if column == "raw_objective_reward"
                        else "Reward can still be negative if penalties dominate the feasibility/tracking bonuses."
                        if column == "reward"
                        else ""
                    ),
                }
            )
    if not summary.empty:
        cash = float((summary["signed_energy_cashflow_proxy"].astype(float) * float(dt_hours)).sum())
        rows.append(
            {
                "metric": "signed_energy_cashflow_proxy",
                "value": round(cash, 6),
                "share_of_total_cost": None,
                "formula": "sum(actual_p_mw * price * dt_hours), with P>0 injection and P<0 absorption",
                "interpretation": "Signed energy cashflow proxy. Negative values usually mean the VPP absorbed power from the grid.",
                "why_negative": "It excludes flexibility-service payments, settlements, DER cost, imbalance penalties and overlapping-window effects; do not read it as net profit.",
            }
        )
    rows.append(
        {
            "metric": "profit_definition_status",
            "value": "",
            "share_of_total_cost": None,
            "formula": "net_profit_proxy is not yet part of the simulator objective",
            "interpretation": "The current simulator optimizes a system objective/reward, not a full market P&L ledger.",
            "why_negative": "A negative reward mainly means high modeled cost or penalty, especially HVAC comfort deviation in the current scenario.",
        }
    )
    return pd.DataFrame(rows)


def privacy_visibility_frame() -> pd.DataFrame:
    return pd.DataFrame(privacy_visibility_records())


def _wide_state_to_long(
    frame: pd.DataFrame,
    prefix: str,
    value_name: str,
    id_name: str,
) -> pd.DataFrame:
    if frame.empty or "step" not in frame:
        return pd.DataFrame(columns=["step", id_name, value_name])
    columns = [col for col in frame.columns if col.startswith(prefix)]
    if not columns:
        return pd.DataFrame(columns=["step", id_name, value_name])
    long = frame.melt(id_vars=["step"], value_vars=columns, var_name=id_name, value_name=value_name)
    long[id_name] = long[id_name].str.replace(prefix, "", regex=False).astype(int)
    return long.sort_values(["step", id_name]).reset_index(drop=True)


def bus_state_frame(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return _wide_state_to_long(
        results.get("bus_voltage", pd.DataFrame()),
        prefix="bus_",
        value_name="vm_pu",
        id_name="bus_id",
    )


def edge_state_frame(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    line = _wide_state_to_long(
        results.get("line_loading", pd.DataFrame()),
        prefix="line_",
        value_name="loading_percent",
        id_name="pp_index",
    )
    if not line.empty:
        line["edge_type"] = "line"
        line["edge_id"] = "line_" + line["pp_index"].astype(str)

    trafo = _wide_state_to_long(
        results.get("trafo_loading", pd.DataFrame()),
        prefix="trafo_",
        value_name="loading_percent",
        id_name="pp_index",
    )
    if not trafo.empty:
        trafo["edge_type"] = "trafo"
        trafo["edge_id"] = "trafo_" + trafo["pp_index"].astype(str)

    parts = [part for part in (line, trafo) if not part.empty]
    if not parts:
        return pd.DataFrame(
            columns=[
                "step",
                "edge_id",
                "edge_type",
                "pp_index",
                "loading_percent",
                "p_from_mw",
                "q_from_mvar",
                "p_to_mw",
                "q_to_mvar",
                "active_loss_mw",
                "reactive_loss_mvar",
                "flow_p_mw",
                "flow_direction",
                "flow_label",
            ]
        )
    combined = pd.concat(parts, ignore_index=True)
    power = results.get("edge_power_flow", pd.DataFrame()).copy()
    if not power.empty:
        combined = combined.merge(
            power[
                [
                    "step",
                    "edge_id",
                    "edge_type",
                    "pp_index",
                    "p_from_mw",
                    "q_from_mvar",
                    "p_to_mw",
                    "q_to_mvar",
                    "active_loss_mw",
                    "reactive_loss_mvar",
                ]
            ],
            on=["step", "edge_id", "edge_type", "pp_index"],
            how="left",
        )
    else:
        for column in [
            "p_from_mw",
            "q_from_mvar",
            "p_to_mw",
            "q_to_mvar",
            "active_loss_mw",
            "reactive_loss_mvar",
        ]:
            combined[column] = None
    combined["p_from_mw"] = combined["p_from_mw"].fillna(0.0)
    combined["q_from_mvar"] = combined["q_from_mvar"].fillna(0.0)
    combined["flow_p_mw"] = combined["p_from_mw"].abs()
    combined["flow_direction"] = combined["p_from_mw"].map(lambda value: "from_to" if value >= 0.0 else "to_from")
    combined["flow_label"] = combined.apply(
        lambda row: f"{abs(float(row['p_from_mw'])):.3f} MW / {abs(float(row['q_from_mvar'])):.3f} MVAr",
        axis=1,
    )
    return combined[
        [
            "step",
            "edge_id",
            "edge_type",
            "pp_index",
            "loading_percent",
            "p_from_mw",
            "q_from_mvar",
            "p_to_mw",
            "q_to_mvar",
            "active_loss_mw",
            "reactive_loss_mvar",
            "flow_p_mw",
            "flow_direction",
            "flow_label",
        ]
    ].sort_values(
        ["step", "edge_type", "pp_index"]
    )


def vpp_state_frame(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return results.get("vpp_power", pd.DataFrame()).copy()


def profile_state_frame(results: dict[str, pd.DataFrame], dt_hours: float = 0.25) -> pd.DataFrame:
    profile = results.get("profile_state", pd.DataFrame()).copy()
    if profile.empty:
        return pd.DataFrame(columns=["step", "time_hours", "time_label", "price", "load_scale", "pv_forecast_factor"])
    if "time_hours" not in profile:
        profile["time_hours"] = profile["step"].astype(float) * float(dt_hours)
    if "time_label" not in profile:
        profile["time_label"] = profile["time_hours"].map(lambda value: f"{float(value):05.2f} h")
    return profile.sort_values("step").reset_index(drop=True)


def der_state_frame(results: dict[str, pd.DataFrame], asset_registry: pd.DataFrame) -> pd.DataFrame:
    der = results.get("der_dispatch", pd.DataFrame()).copy()
    if der.empty:
        return pd.DataFrame(
            columns=[
                "step",
                "vpp_id",
                "der_id",
                "bus_id",
                "der_type",
                "p_mw",
                "q_mvar",
                "soc",
                "average_soc",
                "indoor_temp",
                "available_p_mw",
                "p_min_mw",
                "p_max_mw",
                "state_label",
            ]
        )
    if not asset_registry.empty:
        der = der.merge(
            asset_registry[["der_id", "bus_id", "der_type"]],
            on="der_id",
            how="left",
        )
    else:
        der["bus_id"] = None
        der["der_type"] = der.get("type", "")

    storage = results.get("storage_soc", pd.DataFrame())
    if not storage.empty:
        der = der.merge(storage[["step", "der_id", "soc"]], on=["step", "der_id"], how="left")
    else:
        der["soc"] = None

    evcs = results.get("evcs_soc", pd.DataFrame())
    if not evcs.empty:
        der = der.merge(evcs[["step", "der_id", "average_soc"]], on=["step", "der_id"], how="left")
    else:
        der["average_soc"] = None

    hvac = results.get("hvac_temperature", pd.DataFrame())
    if not hvac.empty:
        der = der.merge(hvac[["step", "der_id", "indoor_temp"]], on=["step", "der_id"], how="left")
    else:
        der["indoor_temp"] = None

    der["state_label"] = der.apply(_der_state_label, axis=1)
    return der.drop(columns=["type"], errors="ignore").sort_values(["step", "vpp_id", "der_id"])


def _fmt_optional(value: object, suffix: str) -> str:
    if value is None or pd.isna(value):
        return ""
    return f" {float(value):.2f}{suffix}"


def _der_state_label(row: pd.Series) -> str:
    der_type = str(row.get("der_type", "DER")).replace("Model", "")
    der_id = str(row.get("der_id", ""))
    p_mw = float(row.get("p_mw", 0.0))
    q_mvar = float(row.get("q_mvar", 0.0))
    parts = [f"{der_type} {der_id}", f"P={p_mw:.3f} MW", f"Q={q_mvar:.3f} MVAr"]
    if der_type == "PV" and "available_p_mw" in row and pd.notna(row.get("available_p_mw")):
        parts.append(f"Avail={float(row['available_p_mw']):.3f} MW")
    if "soc" in row and pd.notna(row.get("soc")):
        parts.append(f"SOC={float(row['soc']):.2f}")
    if "average_soc" in row and pd.notna(row.get("average_soc")):
        parts.append(f"EV SOC={float(row['average_soc']):.2f}")
    if "indoor_temp" in row and pd.notna(row.get("indoor_temp")):
        parts.append(f"T={float(row['indoor_temp']):.1f}C")
    return "\n".join(parts)


def _severity(kind: str, magnitude: float) -> str:
    if kind == "powerflow":
        return "critical"
    if kind.startswith("bus_voltage"):
        return "critical" if magnitude >= 0.03 else "warning"
    if kind.endswith("overload"):
        return "critical" if magnitude >= 20.0 else "warning"
    return "info"


def _element_type(kind: str) -> str:
    if kind.startswith("bus_voltage"):
        return "bus"
    if kind == "line_overload":
        return "line"
    if kind == "trafo_overload":
        return "trafo"
    if kind == "powerflow":
        return "net"
    return "unknown"


def _message(kind: str, element: str, value: float, limit: float, magnitude: float) -> str:
    if kind == "powerflow":
        return "Power flow did not converge."
    if kind == "bus_voltage_low":
        return f"Bus {element} voltage {value:.4f} pu is below lower limit {limit:.4f} pu."
    if kind == "bus_voltage_high":
        return f"Bus {element} voltage {value:.4f} pu is above upper limit {limit:.4f} pu."
    if kind == "line_overload":
        return f"Line {element} loading {value:.2f}% exceeds limit {limit:.2f}% by {magnitude:.2f}%."
    if kind == "trafo_overload":
        return f"Transformer {element} loading {value:.2f}% exceeds limit {limit:.2f}% by {magnitude:.2f}%."
    return f"{kind} on {element}: value={value:.4f}, limit={limit:.4f}."


def alert_event_frame(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    violations = results.get("constraint_violations", pd.DataFrame()).copy()
    if violations.empty:
        return pd.DataFrame(
            columns=[
                "step",
                "kind",
                "severity",
                "element_type",
                "element_id",
                "value",
                "limit",
                "magnitude",
                "message",
            ]
        )
    violations["magnitude"] = violations["magnitude"].astype(float)
    violations["value"] = violations["value"].astype(float)
    violations["limit"] = violations["limit"].astype(float)
    violations["severity"] = violations.apply(lambda row: _severity(str(row["kind"]), row["magnitude"]), axis=1)
    violations["element_type"] = violations["kind"].map(lambda kind: _element_type(str(kind)))
    violations["element_id"] = violations["element"].astype(str)
    violations["message"] = violations.apply(
        lambda row: _message(
            str(row["kind"]),
            str(row["element"]),
            float(row["value"]),
            float(row["limit"]),
            float(row["magnitude"]),
        ),
        axis=1,
    )
    return violations[
        [
            "step",
            "kind",
            "severity",
            "element_type",
            "element_id",
            "value",
            "limit",
            "magnitude",
            "message",
        ]
    ].sort_values(["step", "severity", "kind", "element_id"])


def step_summary_frame(
    results: dict[str, pd.DataFrame],
    dt_hours: float = 0.25,
) -> pd.DataFrame:
    bus_state = bus_state_frame(results)
    edge_state = edge_state_frame(results)
    reward = results.get("reward_components", pd.DataFrame()).copy()
    profile = profile_state_frame(results, dt_hours=dt_hours)

    if not bus_state.empty:
        summary = bus_state.groupby("step").agg(min_vm_pu=("vm_pu", "min"), max_vm_pu=("vm_pu", "max"))
    else:
        summary = pd.DataFrame(index=pd.Index([], name="step"))

    if not edge_state.empty:
        max_by_type = edge_state.pivot_table(
            index="step",
            columns="edge_type",
            values="loading_percent",
            aggfunc="max",
        )
        summary["max_line_loading_percent"] = max_by_type.get("line")
        summary["max_trafo_loading_percent"] = max_by_type.get("trafo")
    else:
        summary["max_line_loading_percent"] = None
        summary["max_trafo_loading_percent"] = None

    if not reward.empty:
        reward_cols = [col for col in ("total_cost", "reward") if col in reward.columns]
        summary = summary.merge(reward[["step", *reward_cols]].set_index("step"), left_index=True, right_index=True, how="outer")

    if not profile.empty:
        profile_cols = [col for col in ("price", "load_scale", "pv_forecast_factor") if col in profile.columns]
        summary = summary.merge(
            profile[["step", *profile_cols]].set_index("step"),
            left_index=True,
            right_index=True,
            how="outer",
        )

    summary = summary.sort_index().reset_index()
    if "step" not in summary:
        return pd.DataFrame(columns=["step", "time_hours", "time_label"])
    summary["time_hours"] = summary["step"].astype(float) * float(dt_hours)
    summary["time_label"] = summary["time_hours"].map(lambda value: f"{value:05.2f} h")
    return summary


def build_dashboard_frames(
    net: pp.pandapowerNet,
    vpps: list[VPPAggregator],
    results: dict[str, pd.DataFrame],
    dt_hours: float = 0.25,
    deep_summary: pd.DataFrame | None = None,
    deep_rl_frames: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    assets = asset_registry_frame(vpps)
    summary = step_summary_frame(results, dt_hours=dt_hours)
    steps = summary["step"].astype(int).to_list() if "step" in summary else []
    portfolio_history = vpp_portfolio_history_frame(results, vpps, steps)
    portfolio_changes = portfolio_change_log_frame(results)
    agent_role_map = pd.DataFrame([role.to_dict() for role in build_agent_role_map(vpps)])
    encoder_role_map = pd.DataFrame([role.to_dict() for role in build_encoder_role_map()])
    deep_frames = {name: frame.copy() for name, frame in (deep_rl_frames or {}).items()}
    if deep_summary is not None:
        deep_frames["deep_rl_training_summary"] = deep_summary.copy()
    frames = {
        "network_nodes": network_nodes_frame(net, vpps),
        "network_edges": network_edges_frame(net),
        "asset_registry": assets,
        "vpp_portfolio": vpp_portfolio_frame(vpps),
        "vpp_portfolio_history": portfolio_history,
        "portfolio_change_log": portfolio_changes,
        "vpp_day_ahead_bid": results.get("vpp_day_ahead_bid", pd.DataFrame()).copy(),
        "dso_operating_envelope": results.get("dso_operating_envelope", pd.DataFrame()).copy(),
        "feasible_region": feasible_region_frame(vpps, steps),
        "fr_envelope_state": fr_envelope_state_frame(results, vpps, steps),
        "projection_trace": projection_trace_frame(results),
        "vpp_rl_disaggregation": results.get("vpp_rl_disaggregation", pd.DataFrame()).copy(),
        "privacy_visibility": privacy_visibility_frame(),
        "step_summary": summary,
        "profile_state": profile_state_frame(results, dt_hours=dt_hours),
        "bus_state": bus_state_frame(results),
        "edge_state": edge_state_frame(results),
        "vpp_state": vpp_state_frame(results),
        "der_state": der_state_frame(results, assets),
        "alert_event": alert_event_frame(results),
        "reward_components": results.get("reward_components", pd.DataFrame()).copy(),
        "agent_role_map": agent_role_map,
        "encoder_role_map": encoder_role_map,
    }
    for name in DEEP_RL_FRAME_NAMES:
        frames[name] = deep_frames.get(name, pd.DataFrame()).copy()
    frames.update(
        build_rl_architecture_frames(
            agent_roles=agent_role_map,
            encoder_roles=encoder_role_map,
            asset_registry=assets,
            deep_summary=frame_or_empty(frames, "deep_rl_training_summary"),
        )
    )
    frames["rl_algorithm_variants"] = build_rl_algorithm_variant_frame(frames)
    frames["model_update_summary"] = model_update_summary_frame(frames)
    frames["vpp_dispatch_explanation"] = build_vpp_dispatch_explanations(frames, dt_hours=dt_hours)
    frames["vpp_first_person_timeline"] = vpp_first_person_timeline_frame(frames)
    frames["vpp_first_person_scope_detail"] = vpp_first_person_scope_detail_frame(frames)
    frames["vpp_step_decision_summary"] = vpp_step_decision_summary_frame(frames)
    frames["vpp_first_person_event_stream"] = vpp_first_person_event_stream_frame(frames)
    frames["vpp_scope_step_summary"] = vpp_scope_step_summary_frame(frames)
    frames["vpp_long_cycle_judgment"] = vpp_long_cycle_judgment_frame(frames)
    frames["portfolio_adjustment_story"] = portfolio_adjustment_story_frame(frames)
    frames["economic_explanation"] = economic_explanation_frame(frames, dt_hours=dt_hours)
    return frames


def export_dashboard_frames(
    frames: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    out = ensure_dir(output_dir)
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths
