from __future__ import annotations

from typing import Any

import pandas as pd

from vpp_dso_sim.visualization.icons import (
    DER_ICON_NAMES,
    DER_SHORT_LABELS,
    der_icon_data_uri,
    grid_source_icon_data_uri,
    pcc_switch_icon_data_uri,
)

DER_SYMBOLS = {
    "PVModel": "triangle-up",
    "MicroTurbineModel": "square",
    "StorageModel": "diamond",
    "FlexibleLoadModel": "triangle-down",
    "HVACModel": "x",
    "EVCSModel": "cross",
}

VPP_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
]


PLOT_TEXT_ZH = {
    "Bus": "母线",
    "Busbar voltage": "母线电压",
    "Bus<br>vm pu": "母线<br>电压pu",
    "Line/trafo loading <40%": "线路/变压器负载 <40%",
    "Line/trafo loading 40-75%": "线路/变压器负载 40-75%",
    "Line/trafo loading 75-100%": "线路/变压器负载 75-100%",
    "Line/trafo loading >100%": "线路/变压器负载 >100%",
    "Feeder switch/protection symbols": "馈线开关/保护符号",
    "Feeder voltage labels": "馈线电压等级标签",
    "DER": "DER 资源",
    "DER dispatch": "DER 调度",
    "DER P MW": "DER 有功 MW",
    "Flow labels": "潮流标签",
    "Interactive feeder topology replay": "馈线拓扑动态回放",
    "Topology level from substation (schematic)": "距变电站拓扑层级（示意）",
    "Radial branch lane (schematic)": "径向支路分道（示意）",
    "Play": "播放",
    "Pause": "暂停",
    "time ": "时间 ",
    "VPP aggregate active power": "VPP 聚合有功功率",
    "P MW, injection positive": "P MW，正值表示向电网注入",
    "Every branch: signed active power by time": "所有支路：按时间显示带符号有功潮流",
    "System branch-loading envelope": "系统支路负载包络",
    "Highest peak-loaded branches": "峰值负载最高的支路",
    "Every-line power-flow and loading summary": "全线路潮流与负载汇总",
    "time (h)": "时间 (h)",
    "branch | buses | nominal kV": "支路 | 母线 | 额定 kV",
    "loading (%)": "负载率 (%)",
    "peak loading (%)": "峰值负载率 (%)",
    "max loading %": "最大负载率 %",
    "p95 loading %": "P95 负载率 %",
    "thermal limit 100%": "热稳定限值 100%",
    "peak loading %": "峰值负载率 %",
    "Price": "电价",
    "Load scale": "负荷倍率",
    "PV forecast factor": "PV 预测因子",
    "Price, load, and PV forecast profiles": "电价、负荷与 PV 预测曲线",
    "price": "电价",
    "profile factor": "曲线因子",
    "Grid security envelope": "电网安全包络",
    "min vm_pu": "最低电压 vm_pu",
    "max vm_pu": "最高电压 vm_pu",
    "max line loading / 100": "最大线路负载率 / 100",
    "pu or per-unitized loading": "标幺值或归一化负载率",
}


PLOT_TEXT_ZH.update(
    {
        "Bus": "母线",
        "Busbar voltage": "母线电压",
        "Bus<br>vm pu": "母线<br>电压 pu",
        "Line/trafo loading <40%": "线路/变压器负载率 <40%",
        "Line/trafo loading 40-75%": "线路/变压器负载率 40-75%",
        "Line/trafo loading 75-100%": "线路/变压器负载率 75-100%",
        "Line/trafo loading >100%": "线路/变压器负载率 >100%",
        "Feeder switch/protection symbols": "馈线开关/保护符号",
        "Feeder voltage labels": "馈线电压等级标签",
        "DER": "DER 分布式资源",
        "DER dispatch": "DER 调度",
        "DER P MW": "DER 有功 MW",
        "Flow labels": "潮流标签",
        "Interactive feeder topology replay": "馈线拓扑动态回放",
        "Topology level from substation (schematic)": "距变电站拓扑层级（示意）",
        "Radial branch lane (schematic)": "径向支路通道（示意）",
        "Play": "播放",
        "Pause": "暂停",
        "time ": "时刻 ",
        "VPP aggregate active power": "VPP 聚合有功功率",
        "P MW, injection positive": "有功 P（MW，注入为正）",
        "Every branch: signed active power by time": "所有支路：按时间显示带符号有功潮流",
        "System branch-loading envelope": "系统支路负载包络",
        "Highest peak-loaded branches": "峰值负载率最高的支路",
        "Every-line power-flow and loading summary": "全线路潮流与负载总览",
        "time (h)": "时间 (h)",
        "branch | buses | nominal kV": "支路 | 母线 | 额定 kV",
        "loading (%)": "负载率 (%)",
        "peak loading (%)": "峰值负载率 (%)",
        "max loading %": "最大负载率 %",
        "p95 loading %": "P95 负载率 %",
        "thermal limit 100%": "热稳定限值 100%",
        "peak loading %": "峰值负载率 %",
        "Price": "电价",
        "Load scale": "负荷倍率",
        "PV forecast factor": "PV 预测因子",
        "Price, load, and PV forecast profiles": "电价、负荷与 PV 预测曲线",
        "price": "电价",
        "profile factor": "曲线因子",
        "Grid security envelope": "电网安全包络",
        "min vm_pu": "最低电压 vm_pu",
        "max vm_pu": "最高电压 vm_pu",
        "max line loading / 100": "最大线路负载率 / 100",
        "pu or per-unitized loading": "标幺值或归一化负载率",
        "PV": "PV 光伏",
        "ESS": "ESS 储能",
        "Flex": "柔性负荷",
        "MT": "MT 微型燃机",
        "EVCS": "EVCS 充电站",
        "HVAC": "HVAC 空调聚合",
    }
)


def zh_hover_text(text: object) -> object:
    """Translate common Plotly hover/tooltip labels while preserving numeric values."""

    if isinstance(text, (list, tuple)):
        return [zh_hover_text(item) for item in text]
    value = str(text)
    replacements = {
        "bus=": "母线=",
        "name=": "名称=",
        "nominal voltage=": "额定电压=",
        "vm=": "电压=",
        "voltage level=": "电压等级=",
        "voltage=": "电压等级=",
        "loading=": "负载率=",
        "P_from=": "首端有功 P_from=",
        "Q_from=": "首端无功 Q_from=",
        "P_to=": "末端有功 P_to=",
        "Q_to=": "末端无功 Q_to=",
        "active direction=": "有功方向=",
        "feeder switch": "馈线开关/保护符号",
        "schematic switching/protection point": "示意开关/保护点",
        "voltage level<br>": "电压等级<br>",
        "time=": "时间=",
        "step=": "时间步=",
        "price=": "电价=",
        "edge=": "支路=",
        "signed P=": "带符号有功 P=",
        "|Q|=": "|无功 Q|=",
        "max loading=": "最大负载率=",
        "p95 loading=": "P95 负载率=",
        "peak loading=": "峰值负载率=",
        "peak |P|=": "峰值 |P|=",
        "P=": "有功 P=",
        "Q=": "无功 Q=",
        "DER": "DER 分布式资源",
        "PCC": "PCC 并网点",
        "icon": "图标",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value


def zh_plot_text(text: object) -> str:
    value = str(text)
    if value in PLOT_TEXT_ZH:
        return PLOT_TEXT_ZH[value]
    if value.endswith(" PCC"):
        return value[:-4] + " PCC 并网点"
    if " | " in value and " icon | " in value:
        return (
            value.replace(" icon | ", " 图标 | ")
            .replace("PV", "PV 光伏")
            .replace("ESS", "ESS 储能")
            .replace("Flex", "柔性负荷")
            .replace("MT", "MT 微型燃机")
            .replace("EVCS", "EVCS 充电站")
            .replace("HVAC", "HVAC 空调聚合")
        )
    return value


def attach_plotly_i18n(
    fig: Any,
    layout_pairs: dict[str, tuple[str, str]] | None = None,
    trace_name_pairs: list[tuple[str, str]] | None = None,
) -> Any:
    """Store bilingual labels in figure metadata for client-side language switching."""

    if trace_name_pairs is None:
        trace_name_pairs = [
            (str(trace.name), zh_plot_text(trace.name))
            for trace in fig.data
        ]
    colorbar_pairs: list[dict[str, object]] = []
    hovertemplate_pairs: list[dict[str, object]] = []
    hovertext_pairs: list[dict[str, object]] = []
    for index, trace in enumerate(fig.data):
        trace_data = trace.to_plotly_json()
        hovertemplate = trace_data.get("hovertemplate")
        if hovertemplate:
            hovertemplate_pairs.append(
                {
                    "index": index,
                    "en": str(hovertemplate),
                    "zh": zh_hover_text(hovertemplate),
                }
            )
        hovertext = trace_data.get("hovertext")
        if hovertext:
            hovertext_pairs.append(
                {
                    "index": index,
                    "en": hovertext,
                    "zh": zh_hover_text(hovertext),
                }
            )
        marker_title = (
            trace_data.get("marker", {})
            .get("colorbar", {})
            .get("title", {})
            .get("text")
        )
        if marker_title:
            colorbar_pairs.append(
                {
                    "index": index,
                    "path": "marker.colorbar.title.text",
                    "en": str(marker_title),
                    "zh": zh_plot_text(marker_title),
                }
            )
        root_title = trace_data.get("colorbar", {}).get("title", {}).get("text")
        if root_title:
            colorbar_pairs.append(
                {
                    "index": index,
                    "path": "colorbar.title.text",
                    "en": str(root_title),
                    "zh": zh_plot_text(root_title),
                }
            )
    meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
    meta = dict(meta)
    meta["i18n"] = {
        "layout": {
            "en": {key: pair[0] for key, pair in (layout_pairs or {}).items()},
            "zh": {key: pair[1] for key, pair in (layout_pairs or {}).items()},
        },
        "trace_names": {
            "en": [pair[0] for pair in trace_name_pairs],
            "zh": [pair[1] for pair in trace_name_pairs],
        },
        "trace_colorbars": {
            "en": [{"index": item["index"], "path": item["path"], "text": item["en"]} for item in colorbar_pairs],
            "zh": [{"index": item["index"], "path": item["path"], "text": item["zh"]} for item in colorbar_pairs],
        },
        "trace_hovertemplates": {
            "en": [{"index": item["index"], "text": item["en"]} for item in hovertemplate_pairs],
            "zh": [{"index": item["index"], "text": item["zh"]} for item in hovertemplate_pairs],
        },
        "trace_hovertexts": {
            "en": [{"index": item["index"], "text": item["en"]} for item in hovertext_pairs],
            "zh": [{"index": item["index"], "text": item["zh"]} for item in hovertext_pairs],
        },
    }
    fig.update_layout(meta=meta)
    return fig


def require_plotly():
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as exc:  # pragma: no cover - depends on optional local env
        raise ImportError(
            "Plotly is required for interactive visualization. Install it with "
            '`pip install -e ".[viz]"` or `pip install plotly`.'
        ) from exc
    return go, pio


def frame_or_empty(frames: dict[str, pd.DataFrame], name: str) -> pd.DataFrame:
    return frames.get(name, pd.DataFrame()).copy()


def simulation_steps(frames: dict[str, pd.DataFrame]) -> list[int]:
    summary = frame_or_empty(frames, "step_summary")
    if not summary.empty and "step" in summary:
        return sorted(summary["step"].dropna().astype(int).unique().tolist())
    bus = frame_or_empty(frames, "bus_state")
    if not bus.empty and "step" in bus:
        return sorted(bus["step"].dropna().astype(int).unique().tolist())
    return [0]


def step_time_lookup(frames: dict[str, pd.DataFrame]) -> dict[int, str]:
    summary = frame_or_empty(frames, "step_summary")
    if not summary.empty and {"step", "time_label"}.issubset(summary.columns):
        return {
            int(row["step"]): str(row["time_label"])
            for _, row in summary[["step", "time_label"]].dropna().iterrows()
        }
    profile = frame_or_empty(frames, "profile_state")
    if not profile.empty and {"step", "time_label"}.issubset(profile.columns):
        return {
            int(row["step"]): str(row["time_label"])
            for _, row in profile[["step", "time_label"]].dropna().iterrows()
        }
    return {step: f"step {step}" for step in simulation_steps(frames)}


def with_time_axis(frame: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    data = frame.copy()
    if data.empty or "step" not in data:
        return data
    summary = frame_or_empty(frames, "step_summary")
    if not summary.empty and {"step", "time_hours", "time_label"}.issubset(summary.columns):
        data = data.merge(summary[["step", "time_hours", "time_label"]], on="step", how="left")
    if "time_hours" not in data:
        data["time_hours"] = data["step"].astype(float)
    if "time_label" not in data:
        lookup = step_time_lookup(frames)
        data["time_label"] = data["step"].map(lambda step: lookup.get(int(step), f"step {int(step)}"))
    return data


def positions_from_nodes(nodes: pd.DataFrame) -> dict[int, tuple[float, float]]:
    if nodes.empty:
        return {}
    clean = nodes.dropna(subset=["x", "y"])
    return {
        int(row["bus_id"]): (float(row["x"]), float(row["y"]))
        for _, row in clean.iterrows()
    }


def format_voltage_kv(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.2f} kV"


def edge_voltage_label(from_vn_kv: Any, to_vn_kv: Any) -> str:
    if pd.isna(from_vn_kv) and pd.isna(to_vn_kv):
        return ""
    if pd.isna(from_vn_kv):
        return format_voltage_kv(to_vn_kv)
    if pd.isna(to_vn_kv):
        return format_voltage_kv(from_vn_kv)
    if abs(float(from_vn_kv) - float(to_vn_kv)) < 1e-6:
        return format_voltage_kv(from_vn_kv)
    return f"{float(from_vn_kv):.2f}/{float(to_vn_kv):.2f} kV"


def edge_voltage_metadata(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()
    prepared = edges.copy()
    if not nodes.empty and {"bus_id", "vn_kv"}.issubset(nodes.columns):
        bus_levels = nodes.set_index("bus_id")["vn_kv"]
        if "from_vn_kv" not in prepared:
            prepared["from_vn_kv"] = prepared["from_bus"].map(bus_levels)
        else:
            prepared["from_vn_kv"] = prepared["from_vn_kv"].fillna(prepared["from_bus"].map(bus_levels))
        if "to_vn_kv" not in prepared:
            prepared["to_vn_kv"] = prepared["to_bus"].map(bus_levels)
        else:
            prepared["to_vn_kv"] = prepared["to_vn_kv"].fillna(prepared["to_bus"].map(bus_levels))
    if "voltage_level_label" not in prepared:
        prepared["voltage_level_label"] = prepared.apply(
            lambda row: edge_voltage_label(row.get("from_vn_kv"), row.get("to_vn_kv")),
            axis=1,
        )
    else:
        computed = prepared.apply(
            lambda row: edge_voltage_label(row.get("from_vn_kv"), row.get("to_vn_kv")),
            axis=1,
        )
        prepared["voltage_level_label"] = prepared["voltage_level_label"].fillna(computed)
        prepared.loc[prepared["voltage_level_label"].astype(str).str.strip() == "", "voltage_level_label"] = computed
    return prepared


def voltage_level_tables(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, pd.DataFrame]:
    bus_levels = pd.DataFrame(columns=["Nominal kV", "Bus count", "Bus IDs"])
    if not nodes.empty and {"bus_id", "vn_kv"}.issubset(nodes.columns):
        prepared = nodes.copy()
        prepared["Nominal kV"] = prepared["vn_kv"].map(format_voltage_kv)
        bus_levels = (
            prepared.groupby("Nominal kV", dropna=False)
            .agg(
                **{
                    "Bus count": ("bus_id", "count"),
                    "Bus IDs": ("bus_id", lambda values: ", ".join(str(int(v)) for v in sorted(set(values)))),
                }
            )
            .reset_index()
        )

    feeder_levels = pd.DataFrame(columns=["Nominal kV", "Branch count", "Branch IDs"])
    prepared_edges = edge_voltage_metadata(nodes, edges)
    if not prepared_edges.empty and {"edge_id", "voltage_level_label"}.issubset(prepared_edges.columns):
        feeder_levels = (
            prepared_edges.groupby("voltage_level_label", dropna=False)
            .agg(
                **{
                    "Branch count": ("edge_id", "count"),
                    "Branch IDs": ("edge_id", lambda values: ", ".join(str(v) for v in values)),
                }
            )
            .reset_index()
            .rename(columns={"voltage_level_label": "Nominal kV"})
        )

    return {"bus_levels": bus_levels, "feeder_levels": feeder_levels}


def vpp_color_map(nodes: pd.DataFrame, assets: pd.DataFrame) -> dict[str, str]:
    ids: set[str] = set()
    if not assets.empty and "vpp_id" in assets:
        ids.update(str(value) for value in assets["vpp_id"].dropna().unique().tolist())
    if not nodes.empty and "vpp_ids" in nodes:
        for value in nodes["vpp_ids"].dropna().tolist():
            ids.update(item for item in str(value).split(",") if item)
    return {vpp_id: VPP_COLORS[i % len(VPP_COLORS)] for i, vpp_id in enumerate(sorted(ids))}


def der_short_label(der_type: str) -> str:
    return DER_SHORT_LABELS.get(str(der_type), str(der_type).replace("Model", ""))


def loading_color(loading_percent: float) -> str:
    loading = max(0.0, min(float(loading_percent), 140.0))
    if loading < 40.0:
        return "#60a5fa"
    if loading < 75.0:
        return "#22c55e"
    if loading < 100.0:
        return "#f59e0b"
    return "#dc2626"


def truthy_mask(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def edge_segments(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[list[float], list[float]]:
    positions = positions_from_nodes(nodes)
    xs: list[float] = []
    ys: list[float] = []
    if edges.empty:
        return xs, ys
    for _, row in edges.iterrows():
        from_bus = int(row["from_bus"])
        to_bus = int(row["to_bus"])
        if from_bus not in positions or to_bus not in positions:
            continue
        xs.extend([positions[from_bus][0], positions[to_bus][0], None])
        ys.extend([positions[from_bus][1], positions[to_bus][1], None])
    return xs, ys


def bus_trace(go: Any, nodes: pd.DataFrame, bus_state: pd.DataFrame, step: int) -> Any:
    if nodes.empty:
        return go.Scatter(x=[], y=[], mode="markers", name="Bus")
    current = bus_state[bus_state["step"] == step] if not bus_state.empty else pd.DataFrame()
    state = current[["bus_id", "vm_pu"]] if not current.empty else pd.DataFrame(columns=["bus_id", "vm_pu"])
    merged = nodes.merge(state, on="bus_id", how="left")
    merged["vm_pu"] = merged["vm_pu"].fillna(1.0)
    hover = [
        f"bus={int(row.bus_id)}<br>name={row.name}<br>nominal voltage={format_voltage_kv(row.vn_kv)}<br>"
        f"vm={row.vm_pu:.4f} pu<br>VPP={row.vpp_ids or '-'}"
        for row in merged.itertuples()
    ]
    size = [
        34 if str(row.is_slack).strip().lower() in {"true", "1", "yes"} else 28 + 2 * min(int(row.asset_count), 3)
        for row in merged.itertuples()
    ]
    labels = [
        f"{int(row.bus_id)}<br>{float(row.vn_kv):.2f}kV"
        for row in merged.itertuples()
    ]
    return go.Scatter(
        x=merged["x"],
        y=merged["y"],
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont={"size": 12, "color": "#0f172a"},
        marker={
            "size": size,
            "symbol": "line-ns",
            "color": merged["vm_pu"],
            "colorscale": "RdYlGn",
            "cmin": 0.93,
            "cmax": 1.07,
            "line": {"color": "#111827", "width": 4},
            "colorbar": {
                "title": {"text": "Bus<br>vm pu"},
                "x": 1.16,
                "y": 0.60,
                "len": 0.48,
                "thickness": 14,
                "tickfont": {"size": 12},
                "outlinewidth": 0,
            },
        },
        hovertext=hover,
        hoverinfo="text",
        name="Busbar voltage",
    )


def edge_traces(go: Any, nodes: pd.DataFrame, edges: pd.DataFrame, edge_state: pd.DataFrame, step: int) -> list[Any]:
    positions = positions_from_nodes(nodes)
    current = edge_state[edge_state["step"] == step] if not edge_state.empty else pd.DataFrame()
    state_by_edge = (
        {str(row["edge_id"]): row for _, row in current.iterrows()} if not current.empty else {}
    )
    traces = [
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"color": "#60a5fa", "width": 4},
            name="Line/trafo loading <40%",
            legendgroup="edge_loading",
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"color": "#22c55e", "width": 4},
            name="Line/trafo loading 40-75%",
            legendgroup="edge_loading",
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"color": "#f59e0b", "width": 4},
            name="Line/trafo loading 75-100%",
            legendgroup="edge_loading",
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"color": "#dc2626", "width": 4},
            name="Line/trafo loading >100%",
            legendgroup="edge_loading",
            hoverinfo="skip",
        ),
    ]
    if edges.empty:
        return traces
    for _, row in edges.sort_values(["edge_type", "pp_index"]).iterrows():
        from_bus = int(row["from_bus"])
        to_bus = int(row["to_bus"])
        if from_bus not in positions or to_bus not in positions:
            continue
        state = state_by_edge.get(str(row["edge_id"]))
        loading = float(state.get("loading_percent", 0.0)) if state is not None else 0.0
        p_from = float(state.get("p_from_mw", 0.0)) if state is not None else 0.0
        q_from = float(state.get("q_from_mvar", 0.0)) if state is not None else 0.0
        p_to = float(state.get("p_to_mw", 0.0)) if state is not None else 0.0
        q_to = float(state.get("q_to_mvar", 0.0)) if state is not None else 0.0
        direction = f"{from_bus} -> {to_bus}" if p_from >= 0.0 else f"{to_bus} -> {from_bus}"
        voltage_label = str(row.get("voltage_level_label", ""))
        hover = (
            f"{row['edge_id']} ({row['edge_type']})<br>"
            f"{from_bus} -> {to_bus}<br>"
            f"voltage level={voltage_label}<br>"
            f"loading={loading:.2f}%<br>"
            f"P_from={p_from:.4f} MW, Q_from={q_from:.4f} MVAr<br>"
            f"P_to={p_to:.4f} MW, Q_to={q_to:.4f} MVAr<br>"
            f"active direction={direction}"
        )
        traces.append(
            go.Scatter(
                x=[positions[from_bus][0], positions[to_bus][0]],
                y=[positions[from_bus][1], positions[to_bus][1]],
                mode="lines",
                line={
                    "color": loading_color(loading),
                    "width": 1.8 + min(max(loading, 0.0), 140.0) / 32.0,
                },
                hovertext=hover,
                hoverinfo="text",
                showlegend=False,
                name=str(row["edge_id"]),
                legendgroup="edge_loading",
            )
        )
    return traces


def feeder_device_trace(go: Any, nodes: pd.DataFrame, edges: pd.DataFrame) -> Any:
    positions = positions_from_nodes(nodes)
    xs: list[float] = []
    ys: list[float] = []
    hover: list[str] = []
    if not edges.empty:
        for _, edge in edges.sort_values(["edge_type", "pp_index"]).iterrows():
            from_bus = int(edge["from_bus"])
            to_bus = int(edge["to_bus"])
            if from_bus not in positions or to_bus not in positions:
                continue
            x1, y1 = positions[from_bus]
            x2, y2 = positions[to_bus]
            xs.append(x1 + 0.16 * (x2 - x1))
            ys.append(y1 + 0.16 * (y2 - y1))
            hover.append(
                f"{edge['edge_id']} feeder switch<br>"
                f"{from_bus} -> {to_bus}<br>"
                f"voltage={edge.get('voltage_level_label', '')}<br>"
                "schematic switching/protection point"
            )
    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker={
            "symbol": "square-open",
            "size": 11,
            "color": "#0f172a",
            "line": {"color": "#0f172a", "width": 2},
        },
        hovertext=hover,
        hoverinfo="text",
        showlegend=True,
        name="Feeder switch/protection symbols",
        legendgroup="electrical_symbols",
    )


def voltage_level_label_trace(go: Any, nodes: pd.DataFrame, edges: pd.DataFrame) -> Any:
    positions = positions_from_nodes(nodes)
    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    hover: list[str] = []
    if not edges.empty:
        for _, edge in edges.sort_values(["edge_type", "pp_index"]).iterrows():
            from_bus = int(edge["from_bus"])
            to_bus = int(edge["to_bus"])
            if from_bus not in positions or to_bus not in positions:
                continue
            x1, y1 = positions[from_bus]
            x2, y2 = positions[to_bus]
            dx = x2 - x1
            dy = y2 - y1
            length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            offset = 0.18
            xs.append((x1 + x2) / 2.0 + dy / length * offset)
            ys.append((y1 + y2) / 2.0 - dx / length * offset)
            voltage_label = str(edge.get("voltage_level_label", ""))
            labels.append(voltage_label.replace(" ", ""))
            hover.append(
                f"{edge['edge_id']} voltage level<br>"
                f"{from_bus} -> {to_bus}<br>{voltage_label}"
            )
    return go.Scatter(
        x=xs,
        y=ys,
        mode="text",
        text=labels,
        textfont={"size": 11, "color": "#475569"},
        textposition="middle center",
        name="Feeder voltage labels",
        hovertext=hover,
        hoverinfo="text",
        visible=True,
    )


def asset_render_records(
    nodes: pd.DataFrame,
    assets: pd.DataFrame,
    der_state: pd.DataFrame,
    step: int,
) -> list[dict[str, Any]]:
    if assets.empty or nodes.empty:
        return []
    positions = positions_from_nodes(nodes)
    colors_by_vpp = vpp_color_map(nodes, assets)
    current = der_state[der_state["step"] == step] if not der_state.empty else pd.DataFrame()
    state = assets.merge(
        current[
            [
                "der_id",
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
        ]
        if not current.empty
        else pd.DataFrame(
            columns=[
                "der_id",
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
        ),
        on="der_id",
        how="left",
    )
    records: list[dict[str, Any]] = []
    for _, row in state.sort_values(["vpp_id", "bus_id", "der_id"]).iterrows():
        bus_id = int(row["bus_id"])
        if bus_id not in positions:
            continue
        x0, y0 = positions[bus_id]
        bus_assets = state[state["bus_id"] == bus_id].sort_values("der_id")["der_id"].tolist()
        pos = bus_assets.index(row["der_id"]) if row["der_id"] in bus_assets else 0
        offset = (pos - (len(bus_assets) - 1) / 2.0) * 0.22
        vpp_id = str(row["vpp_id"])
        der_type = str(row["der_type"])
        p_mw = float(row["p_mw"]) if pd.notna(row.get("p_mw")) else 0.0
        q_mvar = float(row["q_mvar"]) if pd.notna(row.get("q_mvar")) else 0.0
        p_min = float(row["p_min_mw"]) if pd.notna(row.get("p_min_mw")) else 0.0
        p_max = float(row["p_max_mw"]) if pd.notna(row.get("p_max_mw")) else 0.0
        der_label = der_short_label(der_type)
        state_label = (
            str(row["state_label"]).replace("\n", "<br>")
            if pd.notna(row.get("state_label", None))
            else f"{der_label} {row['der_id']}"
        )
        records.append(
            {
                "x": x0 + offset,
                "y": y0 - 0.34,
                "bus_id": bus_id,
                "vpp_id": vpp_id,
                "der_id": str(row["der_id"]),
                "der_type": der_type,
                "der_label": der_label,
                "icon_name": DER_ICON_NAMES.get(der_type, der_label),
                "color": colors_by_vpp.get(vpp_id, "#6b7280"),
                "p_mw": p_mw,
                "q_mvar": q_mvar,
                "p_min_mw": p_min,
                "p_max_mw": p_max,
                "state_label": state_label,
            }
        )
    return records


def asset_icon_layout_images(nodes: pd.DataFrame, assets: pd.DataFrame) -> list[dict[str, Any]]:
    images = []
    for item in asset_render_records(nodes, assets, pd.DataFrame(), 0):
        images.append(
            {
                "source": der_icon_data_uri(str(item["der_type"]), str(item["color"])),
                "xref": "x",
                "yref": "y",
                "x": float(item["x"]),
                "y": float(item["y"]),
                "sizex": 0.58,
                "sizey": 0.58,
                "xanchor": "center",
                "yanchor": "middle",
                "layer": "above",
                "opacity": 0.98,
            }
        )
    return images


def electrical_symbol_layout_images(nodes: pd.DataFrame, assets: pd.DataFrame) -> list[dict[str, Any]]:
    images = []
    if nodes.empty:
        return images
    colors_by_vpp = vpp_color_map(nodes, assets)
    for _, row in nodes.dropna(subset=["x", "y"]).iterrows():
        x = float(row["x"])
        y = float(row["y"])
        if str(row.get("is_slack", "")).strip().lower() in {"true", "1", "yes"}:
            images.append(
                {
                    "source": grid_source_icon_data_uri("#0f172a"),
                    "xref": "x",
                    "yref": "y",
                    "x": x - 0.46,
                    "y": y,
                    "sizex": 0.50,
                    "sizey": 0.50,
                    "xanchor": "center",
                    "yanchor": "middle",
                    "layer": "above",
                    "opacity": 0.96,
                }
            )
        if str(row.get("is_pcc", "")).strip().lower() in {"true", "1", "yes"}:
            vpp_ids = [item for item in str(row.get("vpp_ids", "")).split(",") if item]
            for i, vpp_id in enumerate(vpp_ids):
                images.append(
                    {
                        "source": pcc_switch_icon_data_uri(colors_by_vpp.get(vpp_id, "#2563eb")),
                        "xref": "x",
                        "yref": "y",
                        "x": x + 0.34,
                        "y": y + 0.24 + 0.14 * i,
                        "sizex": 0.42,
                        "sizey": 0.42,
                        "xanchor": "center",
                        "yanchor": "middle",
                        "layer": "above",
                        "opacity": 0.96,
                    }
                )
    return images


def asset_trace(go: Any, nodes: pd.DataFrame, assets: pd.DataFrame, der_state: pd.DataFrame, step: int) -> Any:
    if assets.empty or nodes.empty:
        return go.Scatter(x=[], y=[], mode="markers", name="DER")
    positions = positions_from_nodes(nodes)
    current = der_state[der_state["step"] == step] if not der_state.empty else pd.DataFrame()
    state = assets.merge(
        current[["der_id", "p_mw", "q_mvar", "soc", "average_soc", "indoor_temp"]]
        if not current.empty
        else pd.DataFrame(columns=["der_id", "p_mw", "q_mvar", "soc", "average_soc", "indoor_temp"]),
        on="der_id",
        how="left",
    )
    x: list[float] = []
    y: list[float] = []
    colors: list[float] = []
    text: list[str] = []
    for _, row in state.iterrows():
        bus_id = int(row["bus_id"])
        if bus_id not in positions:
            continue
        x0, y0 = positions[bus_id]
        bus_assets = state[state["bus_id"] == bus_id]["der_id"].tolist()
        pos = bus_assets.index(row["der_id"]) if row["der_id"] in bus_assets else 0
        offset = (pos - (len(bus_assets) - 1) / 2.0) * 0.12
        p_mw = float(row["p_mw"]) if pd.notna(row.get("p_mw")) else 0.0
        q_mvar = float(row["q_mvar"]) if pd.notna(row.get("q_mvar")) else 0.0
        x.append(x0 + offset)
        y.append(y0 - 0.28)
        colors.append(p_mw)
        if pd.notna(row.get("state_label", None)):
            text.append(str(row["state_label"]).replace("\n", "<br>"))
        else:
            text.append(
                f"DER={row['der_id']}<br>type={row['der_type']}<br>VPP={row['vpp_id']}<br>"
                f"bus={bus_id}<br>P={p_mw:.4f} MW<br>Q={q_mvar:.4f} MVAr"
            )
    return go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker={
            "size": 11,
            "symbol": "diamond",
            "color": colors,
            "colorscale": "RdBu",
            "cmid": 0.0,
            "line": {"color": "white", "width": 1},
            "colorbar": {"title": "DER P MW"},
        },
        hovertext=text,
        hoverinfo="text",
        name="DER dispatch",
    )


def asset_traces(
    go: Any,
    nodes: pd.DataFrame,
    assets: pd.DataFrame,
    der_state: pd.DataFrame,
    step: int,
    image_index_offset: int = 0,
) -> list[Any]:
    if assets.empty or nodes.empty:
        return [go.Scatter(x=[], y=[], mode="markers", name="DER")]
    traces = []
    for image_index, item in enumerate(asset_render_records(nodes, assets, der_state, step)):
        hover = (
            f"{item['state_label']}<br>"
            f"VPP={item['vpp_id']}<br>"
            f"bus={item['bus_id']}<br>"
            f"icon={item['icon_name']}<br>"
            f"P bounds=[{item['p_min_mw']:.3f}, {item['p_max_mw']:.3f}] MW<br>"
            f"Q={item['q_mvar']:.3f} MVAr"
        )
        traces.append(
            go.Scatter(
                x=[item["x"]],
                y=[item["y"]],
                mode="markers",
                marker={
                    "size": 30,
                    "symbol": "circle",
                    "color": "rgba(255,255,255,0.01)",
                    "line": {
                        "color": str(item["color"]),
                        "width": 1.8,
                    },
                },
                hovertext=hover,
                hoverinfo="text",
                name=f"{item['vpp_id']} | {item['der_label']} icon | {item['der_id']}",
                legendgroup=str(item["der_id"]),
                meta={"asset_image_index": image_index_offset + image_index},
            )
        )
    return traces or [go.Scatter(x=[], y=[], mode="markers", name="DER")]


def vpp_pcc_traces(go: Any, nodes: pd.DataFrame, assets: pd.DataFrame) -> list[Any]:
    if nodes.empty:
        return []
    positions = positions_from_nodes(nodes)
    colors_by_vpp = vpp_color_map(nodes, assets)
    traces = []
    pcc_nodes = nodes[truthy_mask(nodes["is_pcc"])] if "is_pcc" in nodes else pd.DataFrame()
    for _, row in pcc_nodes.sort_values("bus_id").iterrows():
        bus_id = int(row["bus_id"])
        if bus_id not in positions:
            continue
        vpp_ids = [item for item in str(row.get("vpp_ids", "")).split(",") if item]
        for offset, vpp_id in enumerate(vpp_ids):
            color = colors_by_vpp.get(vpp_id, "#6b7280")
            traces.append(
                go.Scatter(
                    x=[positions[bus_id][0]],
                    y=[positions[bus_id][1]],
                    mode="markers+text",
                    text=[vpp_id],
                    textposition="bottom right",
                    textfont={"size": 13, "color": color},
                    marker={
                        "size": 29 + 6 * offset,
                        "symbol": "circle-open",
                        "color": color,
                        "line": {"color": color, "width": 2.4},
                    },
                    hovertext=f"{vpp_id} PCC<br>bus={bus_id}",
                    hoverinfo="text",
                    name=f"{vpp_id} PCC",
                    legendgroup=vpp_id,
                )
            )
    return traces


def flow_label_trace(go: Any, nodes: pd.DataFrame, edges: pd.DataFrame, edge_state: pd.DataFrame, step: int) -> Any:
    positions = positions_from_nodes(nodes)
    current = edge_state[edge_state["step"] == step] if not edge_state.empty else pd.DataFrame()
    if current.empty or edges.empty:
        return go.Scatter(x=[], y=[], mode="text", name="Flow labels", visible=True)
    merged = edges.merge(
        current[["edge_id", "flow_label", "p_from_mw", "q_from_mvar", "loading_percent"]],
        on="edge_id",
        how="left",
    )
    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    hover: list[str] = []
    for _, row in merged.iterrows():
        from_bus = int(row["from_bus"])
        to_bus = int(row["to_bus"])
        if from_bus not in positions or to_bus not in positions or pd.isna(row.get("flow_label")):
            continue
        x1, y1 = positions[from_bus]
        x2, y2 = positions[to_bus]
        dx = x2 - x1
        dy = y2 - y1
        length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        offset = 0.10
        xs.append((x1 + x2) / 2.0 - dy / length * offset)
        ys.append((y1 + y2) / 2.0 + dx / length * offset)
        p_from = float(row.get("p_from_mw", 0.0))
        q_from = float(row.get("q_from_mvar", 0.0))
        direction = f"{from_bus}->{to_bus}" if p_from >= 0.0 else f"{to_bus}->{from_bus}"
        labels.append(f"{abs(p_from):.2f} MW")
        hover.append(
            f"{row['edge_id']}<br>{direction}<br>"
            f"P={abs(p_from):.4f} MW<br>Q={abs(q_from):.4f} MVAr<br>"
            f"loading={float(row.get('loading_percent', 0.0)):.2f}%"
        )
    return go.Scatter(
        x=xs,
        y=ys,
        mode="text",
        text=labels,
        textfont={"size": 12, "color": "#111827"},
        textposition="middle center",
        name="Flow labels",
        visible=True,
        hovertext=hover,
        hoverinfo="text",
    )


def topology_figure(go: Any, frames: dict[str, pd.DataFrame]) -> Any:
    nodes = frame_or_empty(frames, "network_nodes")
    edges = edge_voltage_metadata(nodes, frame_or_empty(frames, "network_edges"))
    bus_state = frame_or_empty(frames, "bus_state")
    edge_state = frame_or_empty(frames, "edge_state")
    assets = frame_or_empty(frames, "asset_registry")
    der_state = frame_or_empty(frames, "der_state")
    steps = simulation_steps(frames)
    time_lookup = step_time_lookup(frames)
    first = steps[0]
    electrical_images = electrical_symbol_layout_images(nodes, assets)
    asset_images = asset_icon_layout_images(nodes, assets)

    def traces_for_step(step: int) -> list[Any]:
        return [
            *edge_traces(go, nodes, edges, edge_state, step),
            feeder_device_trace(go, nodes, edges),
            bus_trace(go, nodes, bus_state, step),
            *vpp_pcc_traces(go, nodes, assets),
            voltage_level_label_trace(go, nodes, edges),
            flow_label_trace(go, nodes, edges, edge_state, step),
            *asset_traces(go, nodes, assets, der_state, step, image_index_offset=len(electrical_images)),
        ]

    fig = go.Figure(
        data=traces_for_step(first),
        frames=[
            go.Frame(
                name=str(step),
                data=traces_for_step(step),
            )
            for step in steps
        ],
    )
    mark_stride = max(1, len(steps) // 8)
    slider_steps = [
        {
            "args": [[str(step)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": time_lookup.get(step, str(step)) if i % mark_stride == 0 or i == len(steps) - 1 else "",
            "method": "animate",
        }
        for i, step in enumerate(steps)
    ]
    fig.update_layout(
        title={
            "text": f"Interactive feeder topology replay ({time_lookup.get(first, str(first))})",
            "x": 0.02,
            "xanchor": "left",
        },
        height=860,
        margin={"l": 45, "r": 300, "t": 96, "b": 110},
        xaxis={
            "title": "Topology level from substation (schematic)",
            "showgrid": True,
            "zeroline": False,
            "domain": [0.0, 0.79],
            "automargin": True,
        },
        yaxis={
            "title": "Radial branch lane (schematic)",
            "showgrid": True,
            "zeroline": False,
            "scaleanchor": "x",
            "automargin": True,
        },
        plot_bgcolor="#ffffff",
        images=[*electrical_images, *asset_images],
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "direction": "left",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.14,
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 350, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        legend={
            "orientation": "v",
            "x": 1.0,
            "y": 1.0,
            "xanchor": "left",
            "yanchor": "top",
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
            "bgcolor": "rgba(255,255,255,0.92)",
            "bordercolor": "#d6e0ea",
            "borderwidth": 1,
            "font": {"size": 12},
            "itemsizing": "constant",
            "tracegroupgap": 6,
        },
        sliders=[
            {
                "steps": slider_steps,
                "currentvalue": {"prefix": "time "},
                "pad": {"t": 52},
                "len": 0.76,
                "x": 0.0,
            }
        ],
    )
    return attach_plotly_i18n(
        fig,
        {
            "title.text": (
                f"Interactive feeder topology replay ({time_lookup.get(first, str(first))})",
                f"馈线拓扑动态回放（{time_lookup.get(first, str(first))}）",
            ),
            "xaxis.title.text": (
                "Topology level from substation (schematic)",
                "距变电站拓扑层级（示意）",
            ),
            "yaxis.title.text": (
                "Radial branch lane (schematic)",
                "径向支路分道（示意）",
            ),
            "sliders[0].currentvalue.prefix": ("time ", "时间 "),
            "updatemenus[0].buttons[0].label": ("Play", "播放"),
            "updatemenus[0].buttons[1].label": ("Pause", "暂停"),
        },
    )


def vpp_figure(go: Any, vpp_state: pd.DataFrame, frames: dict[str, pd.DataFrame] | None = None) -> Any:
    fig = go.Figure()
    if not vpp_state.empty:
        data = with_time_axis(vpp_state, frames or {})
        for vpp_id, group in vpp_state.groupby("vpp_id"):
            group = data[data["vpp_id"] == vpp_id]
            fig.add_trace(
                go.Scatter(
                    x=group["time_hours"],
                    y=group["p_mw"],
                    mode="lines",
                    name=str(vpp_id),
                    customdata=group[["time_label", "step"]],
                    hovertemplate="time=%{customdata[0]}<br>step=%{customdata[1]}<br>P=%{y:.4f} MW<extra></extra>",
                )
            )
    fig.update_layout(
        title="VPP aggregate active power",
        height=360,
        xaxis_title="time (h)",
        yaxis_title="P MW, injection positive",
        margin={"l": 50, "r": 20, "t": 60, "b": 45},
    )
    return attach_plotly_i18n(
        fig,
        {
            "title.text": ("VPP aggregate active power", "VPP 聚合有功功率"),
            "xaxis.title.text": ("time (h)", "时间 (h)"),
            "yaxis.title.text": ("P MW, injection positive", "P MW，正值表示向电网注入"),
        },
    )


def edge_flow_figure(go: Any, frames: dict[str, pd.DataFrame]) -> Any:
    from plotly.subplots import make_subplots

    edge_state = with_time_axis(frame_or_empty(frames, "edge_state"), frames)
    edges = edge_voltage_metadata(frame_or_empty(frames, "network_nodes"), frame_or_empty(frames, "network_edges"))
    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.50, 0.24, 0.26],
        vertical_spacing=0.14,
        subplot_titles=[
            "Every branch: signed active power by time",
            "System branch-loading envelope",
            "Highest peak-loaded branches",
        ],
    )
    if not edge_state.empty and {"edge_id", "time_hours", "flow_p_mw"}.issubset(edge_state.columns):
        if not edges.empty:
            metadata_cols = [
                column
                for column in (
                    "edge_id",
                    "from_bus",
                    "to_bus",
                    "voltage_level_label",
                    "edge_type",
                    "pp_index",
                )
                if column in edges
            ]
            edge_state = edge_state.merge(
                edges[metadata_cols].drop_duplicates("edge_id"),
                on="edge_id",
                how="left",
                suffixes=("", "_meta"),
            )
            for column in ("edge_type", "pp_index"):
                meta_column = f"{column}_meta"
                if meta_column in edge_state:
                    edge_state[column] = edge_state[column].combine_first(edge_state[meta_column])
                    edge_state = edge_state.drop(columns=[meta_column])
        for column, default in (("from_bus", -1), ("to_bus", -1), ("voltage_level_label", "")):
            if column not in edge_state:
                edge_state[column] = default
        edge_state["from_bus"] = edge_state["from_bus"].fillna(-1)
        edge_state["to_bus"] = edge_state["to_bus"].fillna(-1)
        edge_state["voltage_level_label"] = edge_state["voltage_level_label"].fillna("")
        edge_state["edge_label"] = edge_state.apply(
            lambda row: (
                f"{row['edge_id']} | {int(row['from_bus'])}->{int(row['to_bus'])} | "
                f"{str(row['voltage_level_label'])}"
            ),
            axis=1,
        )
        edge_state = edge_state.sort_values(["edge_type", "pp_index", "step"])
        flow = edge_state.pivot_table(
            index="edge_label",
            columns="time_hours",
            values="flow_p_mw",
            aggfunc="first",
        )
        loading = edge_state.pivot_table(
            index="edge_label",
            columns="time_hours",
            values="loading_percent",
            aggfunc="first",
        )
        q_flow = edge_state.pivot_table(
            index="edge_label",
            columns="time_hours",
            values="q_from_mvar",
            aggfunc="first",
        ).abs()
        custom = []
        for edge_label in flow.index:
            row = []
            for time_hour in flow.columns:
                row.append(
                    [
                        f"{float(time_hour):05.2f} h",
                        float(flow.loc[edge_label, time_hour]) if pd.notna(flow.loc[edge_label, time_hour]) else 0.0,
                        float(q_flow.loc[edge_label, time_hour]) if pd.notna(q_flow.loc[edge_label, time_hour]) else 0.0,
                        float(loading.loc[edge_label, time_hour]) if pd.notna(loading.loc[edge_label, time_hour]) else 0.0,
                    ]
                )
            custom.append(row)
        flow_values = pd.Series(flow.abs().to_numpy().ravel()).dropna()
        max_abs_flow = max(float(flow_values.max()) if not flow_values.empty else 0.0, 0.1)
        fig.add_trace(
            go.Heatmap(
                x=[float(value) for value in flow.columns],
                y=flow.index.tolist(),
                z=flow.to_numpy(),
                customdata=custom,
                colorscale="RdBu",
                zmid=0.0,
                zmin=-max_abs_flow,
                zmax=max_abs_flow,
                colorbar={
                    "title": {"text": "P<br>MW"},
                    "x": 1.02,
                    "y": 0.78,
                    "len": 0.34,
                    "thickness": 14,
                    "tickfont": {"size": 10},
                    "outlinewidth": 0,
                },
                xgap=1,
                ygap=1,
                hoverongaps=False,
                hovertemplate=(
                    "edge=%{y}<br>time=%{customdata[0]}<br>"
                    "signed P=%{customdata[1]:.4f} MW<br>"
                    "|Q|=%{customdata[2]:.4f} MVAr<br>"
                    "loading=%{customdata[3]:.2f}%<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        envelope = (
            edge_state.groupby("time_hours")
            .agg(
                max_loading_percent=("loading_percent", "max"),
                p95_loading_percent=("loading_percent", lambda values: float(values.quantile(0.95))),
                max_flow_p_mw=("flow_p_mw", "max"),
            )
            .reset_index()
        )
        fig.add_trace(
            go.Scatter(
                x=envelope["time_hours"],
                y=envelope["max_loading_percent"],
                mode="lines",
                name="max loading %",
                line={"color": "#dc2626", "width": 2.4},
                hovertemplate="time=%{x:.2f} h<br>max loading=%{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=envelope["time_hours"],
                y=envelope["p95_loading_percent"],
                mode="lines",
                name="p95 loading %",
                line={"color": "#f59e0b", "width": 1.8},
                hovertemplate="time=%{x:.2f} h<br>p95 loading=%{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=envelope["time_hours"],
                y=[100.0] * len(envelope),
                mode="lines",
                name="thermal limit 100%",
                line={"color": "#64748b", "width": 1.4, "dash": "dash"},
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
        peak = (
            edge_state.groupby("edge_label")
            .agg(
                peak_loading_percent=("loading_percent", "max"),
                peak_flow_p_mw=("flow_p_mw", "max"),
            )
            .sort_values("peak_loading_percent", ascending=False)
            .head(12)
            .sort_values("peak_loading_percent", ascending=True)
            .reset_index()
        )
        fig.add_trace(
            go.Bar(
                x=peak["peak_loading_percent"],
                y=peak["edge_label"],
                orientation="h",
                name="peak loading %",
                marker={"color": "#2563eb"},
                customdata=peak[["peak_flow_p_mw"]],
                hovertemplate="edge=%{y}<br>peak loading=%{x:.2f}%<br>peak |P|=%{customdata[0]:.4f} MW<extra></extra>",
            ),
            row=3,
            col=1,
        )
    fig.update_layout(
        title={"text": "Every-line power-flow and loading summary", "x": 0.02, "xanchor": "left"},
        height=1040,
        margin={"l": 220, "r": 145, "t": 118, "b": 92},
        legend={"orientation": "h", "y": -0.10, "font": {"size": 11}},
        bargap=0.25,
        plot_bgcolor="#ffffff",
        hoverlabel={"font": {"size": 12}},
    )
    fig.update_annotations(font={"size": 14, "color": "#17324c"})
    fig.update_xaxes(title_text="", tickfont={"size": 10}, row=1, col=1)
    fig.update_yaxes(title_text="branch | buses | nominal kV", tickfont={"size": 10}, automargin=True, row=1, col=1)
    fig.update_xaxes(title_text="time (h)", row=2, col=1)
    fig.update_yaxes(title_text="loading (%)", tickfont={"size": 10}, row=2, col=1, range=[0, 105])
    fig.update_xaxes(title_text="peak loading (%)", title_standoff=14, row=3, col=1)
    fig.update_yaxes(title_text="branch | buses | nominal kV", tickfont={"size": 10}, automargin=True, row=3, col=1)
    return attach_plotly_i18n(
        fig,
        {
            "title.text": ("Every-line power-flow and loading summary", "全线路潮流与负载汇总"),
            "annotations[0].text": (
                "Every branch: signed active power by time",
                "所有支路：按时间显示带符号有功潮流",
            ),
            "annotations[1].text": ("System branch-loading envelope", "系统支路负载包络"),
            "annotations[2].text": ("Highest peak-loaded branches", "峰值负载最高的支路"),
            "xaxis.title.text": ("", ""),
            "yaxis.title.text": ("branch | buses | nominal kV", "支路 | 母线 | 额定 kV"),
            "xaxis2.title.text": ("time (h)", "时间 (h)"),
            "yaxis2.title.text": ("loading (%)", "负载率 (%)"),
            "xaxis3.title.text": ("peak loading (%)", "峰值负载率 (%)"),
            "yaxis3.title.text": ("branch | buses | nominal kV", "支路 | 母线 | 额定 kV"),
        },
    )


def der_figure(go: Any, der_state: pd.DataFrame, frames: dict[str, pd.DataFrame] | None = None) -> Any:
    fig = go.Figure()
    if not der_state.empty:
        data = with_time_axis(der_state, frames or {})
        for der_id, group in data.groupby("der_id"):
            fig.add_trace(
                go.Scatter(
                    x=group["time_hours"],
                    y=group["p_mw"],
                    mode="lines",
                    name=str(der_id),
                    customdata=group[["time_label", "step", "state_label"]] if "state_label" in group else group[["time_label", "step"]],
                    hovertemplate="time=%{customdata[0]}<br>step=%{customdata[1]}<br>P=%{y:.4f} MW<extra></extra>",
                )
            )
    fig.update_layout(
        title="DER dispatch",
        height=420,
        xaxis_title="time (h)",
        yaxis_title="P MW, injection positive",
        margin={"l": 50, "r": 20, "t": 60, "b": 45},
        legend={"orientation": "h", "y": -0.25},
    )
    return attach_plotly_i18n(
        fig,
        {
            "title.text": ("DER dispatch", "DER 调度"),
            "xaxis.title.text": ("time (h)", "时间 (h)"),
            "yaxis.title.text": ("P MW, injection positive", "P MW，正值表示向电网注入"),
        },
    )


def profile_figure(go: Any, frames: dict[str, pd.DataFrame]) -> Any:
    profile = frame_or_empty(frames, "profile_state")
    if profile.empty:
        profile = frame_or_empty(frames, "step_summary")
    profile = with_time_axis(profile, frames)
    fig = go.Figure()
    if not profile.empty:
        if "price" in profile:
            fig.add_trace(
                go.Scatter(
                    x=profile["time_hours"],
                    y=profile["price"],
                    name="Price",
                    mode="lines",
                    yaxis="y",
                    customdata=profile[["time_label", "step"]],
                    hovertemplate="time=%{customdata[0]}<br>price=%{y:.2f}<extra></extra>",
                )
            )
        if "load_scale" in profile:
            fig.add_trace(
                go.Scatter(
                    x=profile["time_hours"],
                    y=profile["load_scale"],
                    name="Load scale",
                    mode="lines",
                    yaxis="y2",
                )
            )
        if "pv_forecast_factor" in profile:
            fig.add_trace(
                go.Scatter(
                    x=profile["time_hours"],
                    y=profile["pv_forecast_factor"],
                    name="PV forecast factor",
                    mode="lines",
                    yaxis="y2",
                )
            )
    fig.update_layout(
        title="Price, load, and PV forecast profiles",
        height=360,
        xaxis_title="time (h)",
        yaxis={"title": "price"},
        yaxis2={"title": "profile factor", "overlaying": "y", "side": "right", "range": [0, 1.5]},
        margin={"l": 50, "r": 55, "t": 60, "b": 45},
        legend={"orientation": "h", "y": -0.25},
    )
    return attach_plotly_i18n(
        fig,
        {
            "title.text": ("Price, load, and PV forecast profiles", "电价、负荷与 PV 预测曲线"),
            "xaxis.title.text": ("time (h)", "时间 (h)"),
            "yaxis.title.text": ("price", "电价"),
            "yaxis2.title.text": ("profile factor", "曲线因子"),
        },
    )
