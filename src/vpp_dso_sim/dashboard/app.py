from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from vpp_dso_sim.visualization.plotly_figures import (
    DER_ICON_NAMES,
    DER_SHORT_LABELS,
    der_figure,
    edge_flow_figure,
    frame_or_empty,
    profile_figure,
    require_plotly,
    step_time_lookup,
    topology_figure,
    truthy_mask,
    voltage_level_tables,
    vpp_figure,
    with_time_axis,
)


DASH_FRAME_NAMES = [
    "network_nodes",
    "network_edges",
    "asset_registry",
    "vpp_portfolio",
    "vpp_portfolio_history",
    "portfolio_change_log",
    "feasible_region",
    "fr_envelope_state",
    "projection_trace",
    "privacy_visibility",
    "step_summary",
    "profile_state",
    "bus_state",
    "edge_state",
    "vpp_state",
    "der_state",
    "alert_event",
    "vpp_dispatch_explanation",
    "vpp_first_person_timeline",
    "vpp_first_person_scope_detail",
    "vpp_step_decision_summary",
    "vpp_first_person_event_stream",
    "vpp_scope_step_summary",
    "vpp_long_cycle_judgment",
    "portfolio_adjustment_story",
    "economic_explanation",
    "reward_components",
    "rl_algorithm_overview",
    "rl_algorithm_variants",
    "rl_agent_groups",
    "rl_agent_architecture",
    "rl_neural_network_architecture",
    "rl_target_ctde_architecture",
    "rl_agent_relationships",
    "rl_step_workflow",
    "rl_reward_design",
    "rl_loss_components",
    "rl_ctde_assessment",
    "rl_implementation_gaps",
    "model_update_summary",
]

TRAINING_FRAME_NAMES = [
    "training_summary",
    "episode_metrics",
    "tuning_trials",
    "agent_role_map",
    "encoder_role_map",
    "step_metrics",
]

DEEP_RL_FRAME_NAMES = [
    "deep_rl_training_summary",
    "deep_rl_episode_metrics",
    "deep_rl_step_metrics",
    "deep_rl_trajectory",
    "deep_rl_loss_metrics",
]


TextPair = tuple[str, str]


TABLE_LABELS_ZH = {
    "vpp_id": "VPP 编号",
    "vpp_name": "VPP 名称",
    "pcc_bus": "PCC 母线",
    "asset_count": "资源数量",
    "der_types": "DER 类型",
    "buses": "接入母线",
    "asset_ids": "资源编号",
    "Nominal kV": "额定 kV",
    "Bus count": "母线数量",
    "Bus IDs": "母线编号",
    "Branch count": "支路数量",
    "Branch IDs": "支路编号",
    "Short label": "简称",
    "Device type": "设备类型",
    "Icon": "图标",
    "step": "时步",
    "time_label": "时间",
    "severity": "严重程度",
    "element_type": "元件类型",
    "element_id": "元件编号",
    "message": "说明",
    "magnitude": "幅值",
    "der_id": "DER 编号",
    "der_type": "DER 类型",
    "bus_id": "母线编号",
    "owner_vpp_id": "所属 VPP",
    "pp_element_type": "pandapower 元件",
    "pp_element_index": "pandapower 索引",
}


TABLE_VALUE_ZH = {
    "Residential VPP": "居民 VPP",
    "EVCS MT VPP": "充电站-微型燃机 VPP",
    "Mixed DER VPP": "混合 DER VPP",
    "PV": "光伏",
    "ESS": "储能",
    "Flex": "柔性负荷",
    "MT": "微型燃机",
    "EVCS": "充电站",
    "HVAC": "HVAC",
    "PVModel": "光伏模型",
    "StorageModel": "储能模型",
    "FlexibleLoadModel": "柔性负荷模型",
    "MicroTurbineModel": "微型燃机模型",
    "EVCSModel": "充电站模型",
    "HVACModel": "HVAC 模型",
    "solar panel": "光伏组件",
    "battery storage": "储能电池",
    "charging station": "充电站",
    "controllable load": "可控负荷",
    "fan coil": "HVAC 风机盘管",
    "microturbine": "微型燃机",
    "info": "提示",
    "warning": "警告",
    "critical": "严重",
    "bus": "母线",
    "line": "线路",
    "trafo": "变压器",
    "load": "负荷",
    "sgen": "静态电源",
    "storage": "储能",
}

LANG_EN = "en"
LANG_ZH = "zh"

COLUMN_LABELS: dict[str, TextPair] = {
    "step": ("Step", "时步"),
    "kind": ("Kind", "类型"),
    "severity": ("Severity", "严重度"),
    "element_type": ("Element Type", "对象类型"),
    "element_id": ("Element ID", "对象编号"),
    "value": ("Value", "数值"),
    "limit": ("Limit", "限值"),
    "magnitude": ("Magnitude", "超限幅度"),
    "message": ("Message", "说明"),
    "vpp_id": ("VPP ID", "VPP 标识"),
    "vpp_name": ("VPP Name", "VPP 名称"),
    "pcc_bus": ("PCC Bus", "PCC 母线"),
    "asset_count": ("Asset Count", "资产数量"),
    "der_types": ("DER Types", "DER 类型"),
    "buses": ("Bus IDs", "母线编号"),
    "asset_ids": ("Asset IDs", "资产编号"),
    "Nominal kV": ("Nominal kV", "额定 kV"),
    "Bus count": ("Bus Count", "母线数量"),
    "Bus IDs": ("Bus IDs", "母线编号"),
    "Branch count": ("Branch Count", "支路数量"),
    "Branch IDs": ("Branch IDs", "支路编号"),
    "Short label": ("Short Label", "缩写"),
    "Device type": ("Device Type", "设备类型"),
    "Icon": ("Icon", "图标"),
    "der_id": ("DER ID", "DER 标识"),
    "name": ("Name", "名称"),
    "bus_id": ("Bus ID", "母线编号"),
    "der_type": ("DER Type", "DER 类型"),
    "pp_element_type": ("pandapower Element", "pandapower 元件"),
    "pp_element_index": ("pandapower Index", "pandapower 索引"),
}

TRACE_LABELS: dict[str, TextPair] = {
    "min vm_pu": ("min vm_pu", "最小 vm_pu"),
    "max vm_pu": ("max vm_pu", "最大 vm_pu"),
    "max line loading / 100": ("max line loading / 100", "最大线路负载率 / 100"),
    "Price": ("Price", "电价"),
    "Load scale": ("Load scale", "负荷系数"),
    "PV forecast factor": ("PV forecast factor", "PV 预测系数"),
    "Busbar voltage": ("Busbar voltage", "母线电压"),
    "Line/trafo loading <40%": ("Line/trafo loading <40%", "线路/变压器负载率 <40%"),
    "Line/trafo loading 40-75%": ("Line/trafo loading 40-75%", "线路/变压器负载率 40-75%"),
    "Line/trafo loading 75-100%": ("Line/trafo loading 75-100%", "线路/变压器负载率 75-100%"),
    "Line/trafo loading >100%": ("Line/trafo loading >100%", "线路/变压器负载率 >100%"),
    "Feeder switch/protection symbols": ("Feeder switch/protection symbols", "馈线开关/保护符号"),
    "Feeder voltage labels": ("Feeder voltage labels", "馈线电压标签"),
    "Flow labels": ("Flow labels", "潮流标签"),
    "max loading %": ("max loading %", "最大负载率 %"),
    "p95 loading %": ("p95 loading %", "95 分位负载率 %"),
    "thermal limit 100%": ("thermal limit 100%", "热稳定限值 100%"),
    "peak loading %": ("peak loading %", "峰值负载率 %"),
}


def _require_dash():
    try:
        from dash import Dash, Input, Output, dcc, html
    except ImportError as exc:  # pragma: no cover - depends on optional local env
        raise ImportError(
            "Dash is required for the dashboard. Install visualization extras with "
            '`pip install -e ".[viz]"` or install `dash` directly.'
        ) from exc
    return Dash, Input, Output, dcc, html


def load_dashboard_frames(data_dir: str | Path = "outputs/dashboard_data") -> dict[str, pd.DataFrame]:
    """Load standardized dashboard frames written by dashboard_data.export_dashboard_frames."""

    root = Path(data_dir)
    frames: dict[str, pd.DataFrame] = {}

    def _read_frame(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    for name in DASH_FRAME_NAMES:
        path = root / f"{name}.csv"
        frames[name] = _read_frame(path)
    training_root = root if any((root / f"{name}.csv").exists() for name in TRAINING_FRAME_NAMES) else root.parent / "marl_baselines"
    for name in TRAINING_FRAME_NAMES:
        path = training_root / f"{name}.csv"
        frames[name] = _read_frame(path)
    deep_root = root if any((root / f"{name}.csv").exists() for name in DEEP_RL_FRAME_NAMES) else root.parent / "deep_rl"
    for name in DEEP_RL_FRAME_NAMES:
        path = deep_root / f"{name}.csv"
        frames[name] = _read_frame(path)
    return frames


def _kpi_value(frame: pd.DataFrame, column: str, agg: str, default: str = "n/a") -> str:
    if frame.empty or column not in frame:
        return default
    series = frame[column].dropna()
    if series.empty:
        return default
    if agg == "min":
        value = float(series.min())
    elif agg == "max":
        value = float(series.max())
    elif agg == "sum":
        value = float(series.sum())
    else:
        value = float(series.iloc[-1])
    return f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"


def _kpi_cards(html: Any, frames: dict[str, pd.DataFrame]) -> Any:
    summary = frame_or_empty(frames, "step_summary")
    alerts = frame_or_empty(frames, "alert_event")
    nodes = frame_or_empty(frames, "network_nodes")
    edges = frame_or_empty(frames, "network_edges")
    assets = frame_or_empty(frames, "asset_registry")
    items = [
        (("Steps", "步数"), str(len(summary))),
        (("Buses", "母线"), str(len(nodes))),
        (("Edges", "支路"), str(len(edges))),
        (("DER Assets", "DER 资源"), str(len(assets))),
        (("Min Voltage", "最低电压"), _kpi_value(summary, "min_vm_pu", "min")),
        (("Max Line Loading", "最大线路负载率"), _kpi_value(summary, "max_line_loading_percent", "max")),
        (("Alerts", "告警"), str(len(alerts))),
        (("Total Reward", "总奖励"), _kpi_value(summary, "reward", "sum")),
    ]
    return html.Div(
        [
            html.Div([html.Span(_lang_children(html, label)), html.Strong(value)], className="kpi-card")
            for label, value in items
        ],
        className="kpi-grid",
    )


def _summary_figure(go: Any, frames: dict[str, pd.DataFrame]) -> Any:
    summary = with_time_axis(frame_or_empty(frames, "step_summary"), frames)
    fig = go.Figure()
    if not summary.empty:
        if "min_vm_pu" in summary:
            fig.add_trace(go.Scatter(x=summary["time_hours"], y=summary["min_vm_pu"], name="min vm_pu"))
        if "max_vm_pu" in summary:
            fig.add_trace(go.Scatter(x=summary["time_hours"], y=summary["max_vm_pu"], name="max vm_pu"))
        if "max_line_loading_percent" in summary:
            fig.add_trace(
                go.Scatter(
                    x=summary["time_hours"],
                    y=summary["max_line_loading_percent"] / 100.0,
                    name="max line loading / 100",
                )
            )
    fig.update_layout(
        title="Grid security envelope",
        height=360,
        xaxis_title="time (h)",
        yaxis_title="pu or per-unitized loading",
        margin={"l": 45, "r": 20, "t": 55, "b": 40},
    )
    return fig


def _alerts_table_data(alerts: pd.DataFrame) -> list[dict[str, Any]]:
    if alerts.empty:
        return []
    return alerts.head(500).to_dict("records")


def _table_columns(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [{"name": str(col), "id": str(col)} for col in frame.columns]


def _empty_table_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _lang_children(html: Any, text: str | TextPair, *, block: bool = False) -> Any:
    if not isinstance(text, tuple):
        return text
    display_class = "lang-block" if block else "lang-inline"
    return [
        html.Span(text[0], className=f"lang-copy {display_class} lang-en"),
        html.Span(text[1], className=f"lang-copy {display_class} lang-zh"),
    ]


def _normalize_lang(lang: str | None) -> str:
    return LANG_ZH if lang == LANG_ZH else LANG_EN


def _lang_text(text: str | TextPair, lang: str) -> str:
    if not isinstance(text, tuple):
        return str(text)
    return text[1] if _normalize_lang(lang) == LANG_ZH else text[0]


def _column_label(column: object) -> str | TextPair:
    return COLUMN_LABELS.get(str(column), str(column))


def _match_bilingual(value: object, pair: TextPair) -> bool:
    return str(value) in pair


def _apply_trace_labels(fig: Any, lang: str) -> None:
    for trace in fig.data:
        name = getattr(trace, "name", None)
        if name is None:
            continue
        for pair in TRACE_LABELS.values():
            if _match_bilingual(name, pair):
                trace.name = _lang_text(pair, lang)
                break
        if _match_bilingual(name, TRACE_LABELS["Busbar voltage"]) and getattr(trace, "marker", None):
            trace.marker.colorbar.title.text = _lang_text(("Bus<br>vm pu", "母线<br>电压标幺值"), lang)
        if getattr(trace, "type", "") == "heatmap" and getattr(trace, "colorbar", None):
            trace.colorbar.title.text = _lang_text(("P<br>MW", "有功<br>MW"), lang)
        if _normalize_lang(lang) == LANG_ZH:
            current_name = str(getattr(trace, "name", ""))
            if current_name.endswith(" PCC"):
                trace.name = current_name[:-4] + " PCC 并网点"
            elif " icon | " in current_name:
                trace.name = (
                    current_name.replace(" icon | ", " 图标 | ")
                    .replace("PV", "光伏")
                    .replace("ESS", "储能")
                    .replace("Flex", "柔性负荷")
                    .replace("MT", "微型燃机")
                )


def _localize_figure(fig: Any, lang: str, figure_type: str, *, step_label: str | None = None) -> Any:
    lang = _normalize_lang(lang)
    if figure_type == "summary":
        fig.update_layout(
            title=_lang_text(("Grid security envelope", "电网安全包络"), lang),
            xaxis_title=_lang_text(("time (h)", "时间（h）"), lang),
            yaxis_title=_lang_text(("pu or per-unitized loading", "标幺值或归一化负载率"), lang),
        )
    elif figure_type == "profile":
        fig.update_layout(
            title=_lang_text(("Price, load, and PV forecast profiles", "电价、负荷与 PV 预测曲线"), lang),
            xaxis_title=_lang_text(("time (h)", "时间（h）"), lang),
            yaxis={"title": _lang_text(("price", "电价"), lang)},
            yaxis2={
                "title": _lang_text(("profile factor", "曲线系数"), lang),
                "overlaying": "y",
                "side": "right",
                "range": [0, 1.5],
            },
        )
    elif figure_type == "vpp":
        fig.update_layout(
            title=_lang_text(("VPP aggregate active power", "VPP 聚合有功功率"), lang),
            xaxis_title=_lang_text(("time (h)", "时间（h）"), lang),
            yaxis_title=_lang_text(("P MW, injection positive", "有功 P（MW，注入为正）"), lang),
        )
    elif figure_type == "der":
        fig.update_layout(
            title=_lang_text(("DER dispatch", "DER 调度"), lang),
            xaxis_title=_lang_text(("time (h)", "时间（h）"), lang),
            yaxis_title=_lang_text(("P MW, injection positive", "有功 P（MW，注入为正）"), lang),
        )
    elif figure_type == "edge_flow":
        fig.update_layout(title={"text": _lang_text(("Every-line power-flow and loading summary", "全线路潮流与负载总览"), lang), "x": 0.02, "xanchor": "left"})
        fig.update_xaxes(title_text="", row=1, col=1)
        fig.update_yaxes(title_text=_lang_text(("branch | buses | nominal kV", "支路 | 母线 | 额定 kV"), lang), row=1, col=1)
        fig.update_xaxes(title_text=_lang_text(("time (h)", "时间（h）"), lang), row=2, col=1)
        fig.update_yaxes(title_text=_lang_text(("loading (%)", "负载率（%）"), lang), row=2, col=1)
        fig.update_xaxes(title_text=_lang_text(("peak loading (%)", "峰值负载率（%）"), lang), title_standoff=14, row=3, col=1)
        fig.update_yaxes(title_text=_lang_text(("branch | buses | nominal kV", "支路 | 母线 | 额定 kV"), lang), row=3, col=1)
        subplot_titles = [
            ("Every branch: signed active power by time", "各支路分时带符号有功功率"),
            ("System branch-loading envelope", "系统支路负载包络"),
            ("Highest peak-loaded branches", "峰值负载率最高的支路"),
        ]
        if getattr(fig.layout, "annotations", None):
            for annotation, pair in zip(fig.layout.annotations, subplot_titles):
                annotation.text = _lang_text(pair, lang)
    elif figure_type == "topology":
        label = step_label or "step 0"
        fig.update_layout(
            title={"text": _lang_text((f"Network topology state, {label}", f"电网拓扑状态，{label}"), lang), "x": 0.02, "xanchor": "left"},
            xaxis={"title": _lang_text(("Topology level from substation (schematic)", "距变电站的拓扑层级（示意）"), lang)},
            yaxis={"title": _lang_text(("Radial branch lane (schematic)", "径向支路通道（示意）"), lang)},
        )
        if getattr(fig.layout, "sliders", None):
            fig.layout.sliders[0].currentvalue.prefix = _lang_text(("time ", "时刻 "), lang)
        if getattr(fig.layout, "updatemenus", None):
            fig.layout.updatemenus[0].buttons[0].label = _lang_text(("Play", "播放"), lang)
            fig.layout.updatemenus[0].buttons[1].label = _lang_text(("Pause", "暂停"), lang)
    _apply_trace_labels(fig, lang)
    return fig


def _text_value(value: object, default: str = "n/a") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _number_value(value: object, suffix: str = "", default: str = "n/a") -> str:
    if value is None or pd.isna(value):
        return default
    try:
        return f"{float(value):.2f}{suffix}"
    except (TypeError, ValueError):
        text = str(value).strip()
        return f"{text}{suffix}" if text else default


def _dispatch_copy(row: pd.Series, zh_column: str, en_column: str) -> TextPair:
    zh_text = _text_value(row.get(zh_column, ""), default="")
    en_text = _text_value(row.get(en_column, ""), default=zh_text) if en_column in row else zh_text
    if not zh_text and not en_text:
        return ("No explanation was generated.", "未生成说明。")
    return (en_text or zh_text, zh_text or en_text)


def _table_value(value: object) -> str | TextPair:
    en = "" if value is None or pd.isna(value) else str(value)
    if not en:
        return ""
    if en in TABLE_VALUE_ZH:
        return (en, TABLE_VALUE_ZH[en])
    if "," in en:
        parts = [part.strip() for part in en.split(",")]
        translated = [TABLE_VALUE_ZH.get(part, part) for part in parts]
        if translated != parts:
            return (en, ", ".join(translated))
    return en


def _html_table(html: Any, frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 200) -> Any:
    if frame.empty:
        frame = _empty_table_frame(columns or [])
    if columns is not None:
        for column in columns:
            if column not in frame:
                frame[column] = ""
        frame = frame[columns]
    display = frame.head(max_rows)
    return html.Div(
        html.Table(
            [
                html.Thead(html.Tr([html.Th(_lang_children(html, _column_label(col))) for col in display.columns])),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(_lang_children(html, rendered) if isinstance(rendered, tuple) else rendered)
                                for rendered in (_table_value(value) for value in row)
                            ]
                        )
                        for row in display.itertuples(index=False, name=None)
                    ]
                ),
            ],
            className="data-table",
        ),
        className="table-wrap",
    )


def _panel_header(html: Any, eyebrow: str | TextPair, title: str | TextPair, note: str | TextPair | None = None) -> Any:
    return html.Div(
        [
            html.Div(
                [
                    html.P(_lang_children(html, eyebrow), className="eyebrow"),
                    html.H2(_lang_children(html, title), className="panel-heading-title"),
                ],
                className="panel-title",
            ),
            html.P(_lang_children(html, note), className="panel-note") if note else None,
        ],
        className="panel-heading",
    )


def _vpp_summary_table(html: Any, frames: dict[str, pd.DataFrame]) -> Any:
    assets = frame_or_empty(frames, "asset_registry")
    nodes = frame_or_empty(frames, "network_nodes")
    if assets.empty:
        return html.P(_lang_children(html, ("No VPP assets were registered.", "未注册任何 VPP 资产。")))
    pcc_rows = []
    if not nodes.empty and {"is_pcc", "bus_id", "vpp_ids"}.issubset(nodes.columns):
        for _, row in nodes[truthy_mask(nodes["is_pcc"])].iterrows():
            for vpp_id in [item for item in str(row["vpp_ids"]).split(",") if item]:
                pcc_rows.append({"vpp_id": vpp_id, "pcc_bus": int(row["bus_id"])})
    grouped = (
        assets.groupby(["vpp_id", "vpp_name"], dropna=False)
        .agg(
            asset_count=("der_id", "count"),
            der_types=(
                "der_type",
                lambda values: ", ".join(sorted({DER_SHORT_LABELS.get(str(v), str(v).replace("Model", "")) for v in values})),
            ),
            buses=("bus_id", lambda values: ", ".join(str(int(v)) for v in sorted(set(values)))),
            asset_ids=("der_id", lambda values: ", ".join(str(v) for v in values)),
        )
        .reset_index()
    )
    pcc = pd.DataFrame(pcc_rows)
    grouped = grouped.merge(pcc, on="vpp_id", how="left") if not pcc.empty else grouped
    if "pcc_bus" not in grouped:
        grouped["pcc_bus"] = ""
    return _html_table(
        html,
        grouped[["vpp_id", "vpp_name", "pcc_bus", "asset_count", "der_types", "buses", "asset_ids"]],
        max_rows=50,
    )


def _dispatch_metric(html: Any, label: TextPair, value: str | TextPair) -> Any:
    return html.Div(
        [
            html.Span(_lang_children(html, label), className="dispatch-metric-label"),
            html.Strong(_lang_children(html, value) if isinstance(value, tuple) else value),
        ],
        className="dispatch-metric",
    )


def _dispatch_copy_section(html: Any, title: TextPair, body: TextPair) -> Any:
    return html.Section(
        [
            html.H3(_lang_children(html, title)),
            html.Div(_lang_children(html, body, block=True), className="dispatch-copy-body"),
        ],
        className="dispatch-copy-card",
    )


def _dispatch_instruction_table(html: Any, frames: dict[str, pd.DataFrame]) -> Any:
    dispatch = frame_or_empty(frames, "vpp_dispatch_explanation")
    if dispatch.empty:
        return html.P(_lang_children(html, ("No VPP dispatch explanation rows were generated.", "未生成 VPP 调度说明记录。")))
    dispatch = dispatch.copy()
    for column in [
        "vpp_id",
        "vpp_name",
        "start_time",
        "end_time",
        "command_type",
        "command_type_zh",
        "command_type_en",
        "avg_price",
        "avg_p_mw",
        "p_range_mw",
        "reason",
        "instruction",
        "asset_response",
        "reason_zh",
        "instruction_zh",
        "asset_response_zh",
        "reason_en",
        "instruction_en",
        "asset_response_en",
    ]:
        if column not in dispatch:
            dispatch[column] = ""

    groups = []
    for vpp_id, group in dispatch.groupby("vpp_id", sort=False, dropna=False):
        cards = []
        for _, row in group.iterrows():
            command = _dispatch_copy(row, "command_type_zh", "command_type_en")
            reason = _dispatch_copy(row, "reason_zh", "reason_en")
            instruction = _dispatch_copy(row, "instruction_zh", "instruction_en")
            asset_response = _dispatch_copy(row, "asset_response_zh", "asset_response_en")
            cards.append(
                html.Article(
                    [
                        html.Div(
                            [
                                _dispatch_metric(
                                    html,
                                    ("Time Window", "时间窗口"),
                                    f"{_text_value(row.get('start_time'))} - {_text_value(row.get('end_time'))}",
                                ),
                                _dispatch_metric(html, ("Command", "调度类型"), command),
                                _dispatch_metric(html, ("Avg Price", "平均电价"), _number_value(row.get("avg_price"))),
                                _dispatch_metric(html, ("Avg Power", "平均功率"), _number_value(row.get("avg_p_mw"), " MW")),
                                _dispatch_metric(html, ("Power Range", "功率区间"), _text_value(row.get("p_range_mw"))),
                            ],
                            className="dispatch-window-meta",
                        ),
                        html.Div(
                            [
                                _dispatch_copy_section(html, ("Reason", "触发原因"), reason),
                                _dispatch_copy_section(html, ("Instruction", "调度指令"), instruction),
                                _dispatch_copy_section(html, ("Asset Response", "资源响应"), asset_response),
                            ],
                            className="dispatch-window-body",
                        ),
                    ],
                    className="dispatch-window-card",
                )
            )

        raw_name = _text_value(group["vpp_name"].iloc[0], default="") if "vpp_name" in group else ""
        vpp_name = "" if raw_name == "n/a" else raw_name
        title = str(vpp_id) if not vpp_name else f"{vpp_id} · {vpp_name}"
        groups.append(
            html.Section(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.P(_lang_children(html, ("VPP Group", "VPP 分组")), className="eyebrow"),
                                    html.H3(title, className="dispatch-vpp-title"),
                                ]
                            ),
                            html.Span(
                                _lang_children(html, (f"{len(group)} windows", f"{len(group)} 个时间段")),
                                className="dispatch-count-pill",
                            ),
                        ],
                        className="dispatch-vpp-head",
                    ),
                    html.Div(cards, className="dispatch-vpp-list"),
                ],
                className="dispatch-vpp-card",
            )
        )
    return html.Div(groups, className="dispatch-vpp-shell")


def _voltage_levels_panel(html: Any, frames: dict[str, pd.DataFrame]) -> Any:
    tables = voltage_level_tables(frame_or_empty(frames, "network_nodes"), frame_or_empty(frames, "network_edges"))
    return html.Section(
        [
            _panel_header(
                html,
                ("Voltage Classes", "电压等级"),
                ("Bus / Feeder Voltage Levels", "母线 / 馈线电压等级"),
                (
                    "Bus nominal kV comes from network_nodes.vn_kv. Branch voltage classes come from edge endpoints, so transformers show their HV/LV transition explicitly.",
                    "母线额定 kV 来自 network_nodes.vn_kv。支路电压等级来自边两端节点，因此变压器会显式表现高低压过渡。",
                ),
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(_lang_children(html, ("Bus nominal voltage levels", "母线额定电压等级"))),
                            _html_table(html, tables["bus_levels"], columns=["Nominal kV", "Bus count", "Bus IDs"], max_rows=20),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.H3(_lang_children(html, ("Feeder branch voltage levels", "馈线支路电压等级"))),
                            _html_table(html, tables["feeder_levels"], columns=["Nominal kV", "Branch count", "Branch IDs"], max_rows=20),
                        ],
                        className="panel",
                    ),
                ],
                className="panel-grid panel-grid-two",
            ),
        ],
        className="panel",
    )


def _step_slider_marks(frames: dict[str, pd.DataFrame], min_step: int, max_step: int) -> dict[int, str]:
    lookup = step_time_lookup(frames)
    steps = sorted(step for step in lookup if min_step <= step <= max_step)
    if not steps:
        return {min_step: str(min_step)}
    stride = max(1, len(steps) // 8)
    return {
        step: lookup.get(step, str(step))
        for i, step in enumerate(steps)
        if i % stride == 0 or i == len(steps) - 1
    }


def _dashboard_brief_panel(html: Any) -> Any:
    cards = [
        (
            ("Single-line symbols", "单线图符号"),
            (
                "Bus markers represent electrical nodes, include nominal voltage labels, and follow deterministic one-line connectivity rather than a GIS map.",
                "母线标记表示电气节点，包含额定电压标签，并遵循确定性的单线图连接关系，而不是 GIS 地图。",
            ),
        ),
        (
            ("VPP colors", "VPP 颜色"),
            (
                "A VPP keeps the same accent color across PCC rings, legend items and DER icon borders so ownership stays consistent across tabs.",
                "同一 VPP 会在 PCC 环、图例项和 DER 图标边框上保持一致强调色，确保跨页签的归属关系一致。",
            ),
        ),
        (
            ("DER realistic icons", "DER 真实图标"),
            (
                "PV, ESS, EVCS, HVAC, microturbine and flexible-load assets use local SVG pictograms instead of generic dots or external icon fonts.",
                "PV、ESS、EVCS、HVAC、微型燃机和柔性负荷使用本地 SVG 图标，而不是通用圆点或外部图标字体。",
            ),
        ),
        (
            ("Flow labels", "潮流标签"),
            (
                "Topology flow labels show branch active power in MW; feeder labels show nominal kV. Use hover and the line-flow matrix when you need MVAr or loading context.",
                "拓扑潮流标签显示支路有功功率 MW；馈线标签显示额定 kV。需要 MVAr 或负载率语境时请结合悬停信息和线路潮流矩阵。",
            ),
        ),
    ]
    return html.Section(
        [
            _panel_header(
                html,
                ("Analysis Scope", "分析范围"),
                ("Read-only feeder simulation console", "只读馈线仿真控制台"),
                (
                    "Use Overview for system posture, Topology for spatial replay, VPP / DER for dispatch interpretation, and Alerts for event review.",
                    "先看总览把握系统态势，再看拓扑做空间回放，然后通过 VPP / DER 解读调度，最后在告警页复盘事件。",
                ),
            ),
            html.Div(
                [
                    html.Article(
                        [html.H3(_lang_children(html, title)), html.P(_lang_children(html, text))],
                        className="guide-card",
                    )
                    for title, text in cards
                ],
                className="guide-grid",
            ),
        ],
        className="mission-panel",
    )


def _decision_drivers_panel(html: Any, frames: dict[str, pd.DataFrame]) -> Any:
    profile = frame_or_empty(frames, "profile_state")
    dispatch = frame_or_empty(frames, "vpp_dispatch_explanation")
    if profile.empty or "price" not in profile:
        price_min = price_max = low_steps = high_steps = "n/a"
    else:
        price_min = _number_value(profile["price"].min())
        price_max = _number_value(profile["price"].max())
        low_steps = str(int((profile["price"] <= 55.0).sum()))
        high_steps = str(int((profile["price"] >= 100.0).sum()))
    segment_count = "0" if dispatch.empty else str(len(dispatch))
    cards = [
        (
            ("1. DSO envelope / baseline target", "1. DSO 包络 / 基线目标"),
            (
                "The DSO builds an operating envelope from VPP bids, local flexibility bounds and grid stress. When no RL action is supplied, the baseline target is the preferred point inside that envelope.",
                "DSO 根据 VPP 报量/报价、本地灵活性边界和电网压力生成运行包络。没有 RL 动作时，基线目标是包络内的推荐点。",
            ),
        ),
        (
            ("2. Feasible VPP flexibility band", "2. VPP 可行灵活性区间"),
            (
                "Every target is clipped to the aggregated [P_min, P_max] band from PV availability, ESS SOC, EV/HVAC/flexible-load limits and microturbine bounds.",
                "每个目标都会被裁剪到聚合 [P_min, P_max] 区间内，该区间来自 PV 可用功率、储能 SOC、EV/HVAC/柔性负荷限制和微型燃机边界。",
            ),
        ),
        (
            ("3. Learned DER-level disaggregation", "3. 学习型 DER 级解聚合"),
            (
                "When RL actions are supplied, the VPP dispatch policy proposes per-DER normalized actions. The safety layer clips bounds and repairs aggregate residuals; the cost-order allocator remains only as a reproducible fallback.",
                "接入 RL 动作时，VPP dispatch policy 会提出逐 DER 归一化动作。安全层负责边界裁剪和聚合残差修复；按成本顺序分配只作为可复现 fallback。",
            ),
        ),
        (
            ("Actual horizon summary", "当前仿真概览"),
            (
                f"Price range {price_min}-{price_max}; low-price steps {low_steps}; high-price steps {high_steps}; one-day instruction segments {segment_count}.",
                f"电价范围 {price_min}-{price_max}；低价时步 {low_steps}；高价时步 {high_steps}；单日调度片段 {segment_count}。",
            ),
        ),
    ]
    return html.Section(
        [
            _panel_header(
                html,
                ("Control Logic", "控制逻辑"),
                ("What Drives VPP Decisions", "VPP 决策由什么驱动"),
                (
                    "This dashboard shows the envelope baseline and, when training artifacts are present, the learned DER-level VPP dispatch path.",
                    "该仪表盘展示包络基线；当存在训练产物时，也展示学习型 DER 级 VPP 调度路径。",
                ),
            ),
            html.Div(
                [
                    html.Article(
                        [html.H3(_lang_children(html, title)), html.P(_lang_children(html, text))],
                        className="guide-card",
                    )
                    for title, text in cards
                ],
                className="guide-grid",
            ),
        ],
        className="mission-panel",
    )


def _fr_doe_panel(html: Any, frames: dict[str, pd.DataFrame]) -> Any:
    envelope = frame_or_empty(frames, "fr_envelope_state")
    projection = frame_or_empty(frames, "projection_trace")
    privacy = frame_or_empty(frames, "privacy_visibility")
    return html.Div(
        [
            html.Section(
                [
                    _panel_header(
                        html,
                        ("Operating Envelope", "Operating Envelope"),
                        ("FR/DOE Envelope", "FR/DOE Envelope"),
                        (
                            "Static v0 envelopes are exported by VPP and physical scope. Multi-node VPP rows remain bus/zone/DER scoped for auditability.",
                            "Static v0 envelopes are exported by VPP and physical scope. Multi-node VPP rows remain bus/zone/DER scoped for auditability.",
                        ),
                    ),
                    _html_table(
                        html,
                        envelope,
                        columns=[
                            "step",
                            "vpp_id",
                            "physical_mode",
                            "scope_type",
                            "scope_id",
                            "variable",
                            "lower_bound",
                            "upper_bound",
                            "current_value",
                            "is_binding",
                        ],
                        max_rows=80,
                    ),
                ],
                className="panel",
            ),
            html.Section(
                [
                    _panel_header(
                        html,
                        ("Projection Chain", "Projection Chain"),
                        ("Projection Audit", "Projection Audit"),
                        (
                            "Each command is traced through raw target, device bounds, FR/DOE projection, pandapower write and power-flow result.",
                            "Each command is traced through raw target, device bounds, FR/DOE projection, pandapower write and power-flow result.",
                        ),
                    ),
                    _html_table(
                        html,
                        projection,
                        columns=[
                            "step",
                            "vpp_id",
                            "stage_order",
                            "stage_name",
                            "scope_type",
                            "scope_id",
                            "p_mw",
                            "was_projected",
                            "active_constraint",
                            "projection_reason",
                        ],
                        max_rows=100,
                    ),
                ],
                className="panel",
            ),
            html.Section(
                [
                    _panel_header(
                        html,
                        ("Privacy Boundary", "Privacy Boundary"),
                        ("Privacy Visibility", "Privacy Visibility"),
                        (
                            "Visibility rows state which schema fields are visible to the DSO, owning VPP, other VPPs, or oracle-only baselines.",
                            "Visibility rows state which schema fields are visible to the DSO, owning VPP, other VPPs, or oracle-only baselines.",
                        ),
                    ),
                    _html_table(
                        html,
                        privacy,
                        columns=[
                            "schema",
                            "field",
                            "visible_to_dso",
                            "visible_to_vpp_i",
                            "visible_to_other_vpp",
                            "oracle_only",
                        ],
                        max_rows=100,
                    ),
                ],
                className="panel",
            ),
        ],
        className="tab-stack",
    )


def _first_person_panel(html: Any, frames: dict[str, pd.DataFrame]) -> Any:
    timeline = frame_or_empty(frames, "vpp_first_person_timeline")
    scope = frame_or_empty(frames, "vpp_first_person_scope_detail")
    step_summary = frame_or_empty(frames, "vpp_step_decision_summary")
    event_stream = frame_or_empty(frames, "vpp_first_person_event_stream")
    long_cycle = frame_or_empty(frames, "vpp_long_cycle_judgment")
    changes = frame_or_empty(frames, "portfolio_change_log")
    return html.Section(
        [
            _panel_header(
                html,
                ("VPP First View", "VPP 第一视角"),
                ("Saw / Inferred / Decided Replay", "看到 / 推断 / 决策回放"),
                (
                    "Each row is written from the VPP viewpoint: what information was visible, what grid need was inferred, and what dispatch or portfolio decision followed.",
                    "每一行都从 VPP 第一视角记录：看到了哪些信息、推断出什么电网友好需求，以及随后采取了什么调度或聚合配置动作。",
                ),
            ),
            html.H3("Step Decision Summary"),
            _html_table(
                html,
                step_summary,
                columns=[
                    "step",
                    "time_label",
                    "vpp_id",
                    "command_seen",
                    "belief_label",
                    "action_label",
                    "projected_p_mw",
                    "actual_p_mw",
                    "decision_status",
                ],
                max_rows=120,
            ),
            html.H3("Event Stream"),
            _html_table(
                html,
                event_stream,
                columns=["step", "time_label", "vpp_id", "event_order", "event_type", "event_detail", "decision_status"],
                max_rows=160,
            ),
            html.H3("Long-Cycle Judgment"),
            _html_table(
                html,
                long_cycle,
                columns=[
                    "vpp_id",
                    "window_id",
                    "dominant_grid_need",
                    "dominant_action",
                    "reliability_score",
                    "projection_count",
                    "risk_level",
                    "portfolio_recommendation",
                ],
                max_rows=80,
            ),
            html.H3("Timeline"),
            _html_table(
                html,
                timeline,
                columns=[
                    "vpp_id",
                    "phase",
                    "window_id",
                    "portfolio_version",
                    "physical_mode",
                    "connection_buses",
                    "seen_direction",
                    "inferred_grid_need_label",
                    "inferred_delivery_risk",
                    "decision_type",
                    "decision_status",
                    "private_cost_used",
                ],
                max_rows=80,
            ),
            html.H3("Scope Detail"),
            _html_table(
                html,
                scope,
                columns=[
                    "vpp_id",
                    "step",
                    "scope_type",
                    "scope_id",
                    "bus_id",
                    "asset_ids",
                    "variable",
                    "p_lower_mw",
                    "p_upper_mw",
                    "current_p_mw",
                    "is_binding",
                ],
                max_rows=120,
            ),
            html.H3("Portfolio Change Log"),
            _html_table(
                html,
                changes,
                columns=[
                    "event_id",
                    "effective_step",
                    "der_id",
                    "from_vpp_id",
                    "to_vpp_id",
                    "bus_id",
                    "zone_id",
                    "physical_bus_unchanged",
                ],
                max_rows=40,
            ),
        ],
        className="panel",
    )


def _training_status_panel(html: Any, frames: dict[str, pd.DataFrame]) -> Any:
    summary = frame_or_empty(frames, "training_summary")
    episodes = frame_or_empty(frames, "episode_metrics")
    trials = frame_or_empty(frames, "tuning_trials")
    roles = frame_or_empty(frames, "agent_role_map")
    deep_summary = frame_or_empty(frames, "deep_rl_training_summary")
    deep_episodes = frame_or_empty(frames, "deep_rl_episode_metrics")
    deep_losses = frame_or_empty(frames, "deep_rl_loss_metrics")
    variants = frame_or_empty(frames, "rl_algorithm_variants")
    model_updates = frame_or_empty(frames, "model_update_summary")
    if summary.empty:
        status = "not_available"
        best_algorithm = "n/a"
        message = "Run python examples/09_run_marl_baselines.py to generate training artifacts."
    else:
        first = summary.iloc[0]
        status = str(first.get("status", "unknown"))
        best_algorithm = str(first.get("best_algorithm", "n/a"))
        message = str(first.get("handoff_message", "")) or str(first.get("reason", ""))
    return html.Section(
        [
            _panel_header(
                html,
                ("Training", "训练"),
                ("Training Supervisor Status", "Training Supervisor Status"),
                (
                    "MARL baseline trials are read from outputs/marl_baselines. Non-converged trials explicitly request main-thread algorithm review.",
                    "MARL baseline trials are read from outputs/marl_baselines. Non-converged trials explicitly request main-thread algorithm review.",
                ),
            ),
            html.Div(
                [
                    html.Article([html.H3("Status"), html.P(status)], className="guide-card"),
                    html.Article([html.H3("Best Algorithm"), html.P(best_algorithm)], className="guide-card"),
                    html.Article([html.H3("Handoff"), html.P(message or "none")], className="guide-card"),
                ],
                className="guide-grid",
            ),
            html.H3("MARL Baselines"),
            _html_table(
                html,
                episodes,
                columns=["algorithm", "episode", "episode_reward", "episode_cost", "violation_count"],
                max_rows=40,
            ),
            html.H3("Tuning Trials"),
            _html_table(
                html,
                trials,
                columns=["trial_id", "algorithm", "action_scale", "exploration_noise", "mean_reward", "status"],
                max_rows=40,
            ),
            html.H3("Deep RL Actor-Critic"),
            _html_table(
                html,
                deep_summary,
                columns=[
                    "algorithm",
                    "status",
                    "is_deep_rl",
                    "optimizer_steps",
                    "param_delta_l2",
                    "dso_actor_trainable",
                    "vpp_dispatch_trainable",
                    "portfolio_trainable",
                    "best_episode_reward",
                    "final_episode_reward",
                ],
                max_rows=10,
            ),
            html.H3("RL Algorithm Variants"),
            _html_table(
                html,
                variants,
                columns=[
                    "algorithm_label",
                    "family",
                    "critic_style",
                    "update_core",
                    "repo_status",
                ],
                max_rows=10,
            ),
            html.H3("Model / Algorithm Update Summary"),
            _html_table(
                html,
                model_updates,
                columns=["update_area", "current_value", "current_value_zh", "explanation", "explanation_zh", "evidence_file"],
                max_rows=20,
            ),
            html.H3("Deep RL Episodes"),
            _html_table(
                html,
                deep_episodes,
                columns=["episode", "algorithm", "episode_reward", "episode_cost", "violation_count", "policy_loss", "value_loss", "entropy", "grad_norm"],
                max_rows=40,
            ),
            html.H3("Deep RL Loss Metrics"),
            _html_table(
                html,
                deep_losses,
                columns=["episode", "algorithm", "policy_loss", "value_loss", "entropy_loss", "total_loss", "grad_norm", "optimizer_step"],
                max_rows=40,
            ),
            html.H3("Agent Role Map"),
            _html_table(
                html,
                roles,
                columns=["agent_id", "role_type", "owner_id", "time_scale", "privacy_scope"],
                max_rows=30,
            ),
        ],
        className="panel",
    )


def _topology_legend_panel(html: Any) -> Any:
    legend = pd.DataFrame(
        [
            {
                "Short label": DER_SHORT_LABELS.get(der_type, der_type.replace("Model", "")),
                "Device type": der_type.replace("Model", ""),
                "Icon": icon_name,
            }
            for der_type, icon_name in DER_ICON_NAMES.items()
        ]
    )
    return html.Section(
        [
            _panel_header(
                html,
                ("Symbol Guide", "符号指南"),
                ("Topology decoding", "拓扑解读"),
                (
                    "Bus labels show bus ID and nominal kV. Feeder labels show nominal branch voltage. PCC rings label VPP coupling buses, and the axes describe schematic electrical layout rather than physical distance.",
                    "母线标签显示母线编号和额定 kV。馈线标签显示支路额定电压。PCC 圈标出 VPP 耦合母线，坐标轴描述的是示意化电气布局而不是物理距离。",
                ),
            ),
            html.Div(
                [
                    html.Article(
                        [
                            html.H3(_lang_children(html, ("Legend behavior", "图例行为"))),
                            html.P(
                                _lang_children(
                                    html,
                                    (
                                        "Hide a single DER trace, a PCC ring or flow labels from the topology legend when feeder density is more important than annotation.",
                                        "当馈线密度比注释更重要时，可从拓扑图例中隐藏单个 DER 曲线、PCC 圈或潮流标签。",
                                    ),
                                )
                            ),
                        ],
                        className="guide-card",
                    ),
                    html.Article(
                        [
                            html.H3(_lang_children(html, ("Hover detail", "悬停细节"))),
                            html.P(
                                _lang_children(
                                    html,
                                    (
                                        "Branch hover keeps `MW / MVAr / loading / voltage level` detail available even when the plot is decluttered for overview reading.",
                                        "即使为了总览而简化图面，支路悬停信息仍会保留 `MW / MVAr / loading / voltage level` 细节。",
                                    ),
                                )
                            ),
                        ],
                        className="guide-card",
                    ),
                ],
                className="guide-grid guide-grid-compact",
            ),
            _html_table(html, legend, columns=["Short label", "Device type", "Icon"], max_rows=20),
        ],
        className="panel panel-soft",
    )


def _metric_calculation_panel(html: Any) -> Any:
    return html.Section(
        [
            _panel_header(
                html,
                ("Methods Note", "方法说明"),
                ("Metric Calculations", "指标计算"),
                (
                    "These definitions explain how the dashboard KPIs and security traces are derived from pandapower result tables.",
                    "这些定义解释了 dashboard KPI 和安全曲线如何从 pandapower 结果表推导而来。",
                ),
            ),
            html.Ul(
                [
                    html.Li(_lang_children(html, ("Min voltage: minimum bus vm_pu from pandapower res_bus at each time point; KPI uses the full-horizon minimum.", "最低电压：每个时刻取 pandapower res_bus 中母线 vm_pu 的最小值；KPI 使用全时域最小值。"))),
                    html.Li(_lang_children(html, ("Max line loading: maximum line loading_percent over all branches at each time point.", "最大线路负载率：每个时刻取全部支路的最大 line loading_percent。"))),
                    html.Li(_lang_children(html, ("Line flow: p_from_mw/q_from_mvar are read from pandapower branch result tables; positive p_from_mw follows from_bus to to_bus.", "线路潮流：p_from_mw/q_from_mvar 来自 pandapower 支路结果表；正的 p_from_mw 表示从 from_bus 流向 to_bus。"))),
                    html.Li(_lang_children(html, ("VPP power: sum of internal DER active powers, injection positive and absorption negative.", "VPP 功率：内部 DER 有功功率求和，注入为正、吸收为负。"))),
                    html.Li(_lang_children(html, ("PV available power: p_max_mw multiplied by the configured PV forecast factor at the current time.", "PV 可用功率：p_max_mw 乘以当前时刻配置的 PV 预测因子。"))),
                    html.Li(_lang_children(html, ("Reward: role-specific general-sum rewards are now separated into r_dso, r_dispatch_i and r_portfolio_i; VPP portfolio receives localized DSO-alignment credit instead of raw global reward sharing.", "Reward：当前已拆分为 r_dso、r_dispatch_i、r_portfolio_i 三类角色专属 general-sum reward；VPP 组合配置接收局部化 DSO 对齐收益，而不是直接共享原始全局 reward。"))),
                ]
            ),
        ],
        className="panel",
    )


def create_dashboard_app(
    frames: dict[str, pd.DataFrame] | None = None,
    data_dir: str | Path = "outputs/dashboard_data",
    title: str = "pandapower VPP DSO Dashboard",
):
    """Create a read-only Dash dashboard for standardized simulation results."""

    Dash, Input, Output, dcc, html = _require_dash()
    go, _ = require_plotly()
    frames = load_dashboard_frames(data_dir) if frames is None else frames

    summary = frame_or_empty(frames, "step_summary")
    min_step = int(summary["step"].min()) if not summary.empty and "step" in summary else 0
    max_step = int(summary["step"].max()) if not summary.empty and "step" in summary else 0
    alerts = frame_or_empty(frames, "alert_event")
    assets = frame_or_empty(frames, "asset_registry")
    der_state = frame_or_empty(frames, "der_state")

    app = Dash(__name__, title=title)
    app.layout = html.Div(
        [
            html.Header(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.P(
                                        _lang_children(
                                            html,
                                            ("Feeder Simulation Analysis Console", "馈线仿真分析控制台"),
                                        ),
                                        className="eyebrow hero-eyebrow",
                                    ),
                                    html.Div(
                                        [
                                            html.Span(_lang_children(html, ("Language", "语言")), className="language-label"),
                                            html.Button(
                                                "EN",
                                                id="dashboard-lang-en",
                                                className="lang-button is-active",
                                                **{"data-lang-switch": "en", "aria-pressed": "true"},
                                            ),
                                            html.Button(
                                                "中文",
                                                id="dashboard-lang-zh",
                                                className="lang-button",
                                                **{"data-lang-switch": "zh", "aria-pressed": "false"},
                                            ),
                                        ],
                                        className="language-toolbar",
                                        role="group",
                                        **{"aria-label": "Language switch"},
                                    ),
                                ],
                                className="hero-topline",
                            ),
                            html.H1(title),
                            html.P(
                                _lang_children(
                                    html,
                                    (
                                        "Read-only dashboard for feeder-level DSO and multi-VPP simulation results built from standardized dashboard frames.",
                                        "基于标准化 dashboard 数据帧构建的只读页面，用于查看馈线级 DSO 与多 VPP 仿真结果。",
                                    ),
                                ),
                                className="hero-copy",
                            ),
                            html.P(
                                _lang_children(
                                    html,
                                    (
                                        "Tabs stay bilingual, while the toggle switches the surrounding explanation cards.",
                                        "页签保持双语标签，切换按钮用于切换周边说明卡片语言。",
                                    ),
                                ),
                                className="hero-language-note",
                            ),
                        ],
                        className="hero-copy-wrap",
                    ),
                    dcc.Store(id="dashboard-language", data=LANG_EN),
                    html.Div(
                        [
                            html.Div(_lang_children(html, ("Operator Brief", "操作员提示")), className="hero-brief-label"),
                            html.P(
                                _lang_children(
                                    html,
                                    (
                                        "Review grid security first, then replay topology states, then drill into VPP and DER dispatch before reading alert events.",
                                        "先检查电网安全，再回放拓扑状态，然后深入查看 VPP 与 DER 调度，最后阅读告警事件。",
                                    ),
                                ),
                            ),
                        ],
                        className="hero-brief",
                    ),
                ],
                className="app-header",
            ),
            html.Main(
                [
                    _dashboard_brief_panel(html),
                    _decision_drivers_panel(html, frames),
                    _kpi_cards(html, frames),
                    dcc.Tabs(
                        [
                            dcc.Tab(
                                label="Overview / 总览",
                                className="dashboard-tab",
                                selected_className="dashboard-tab dashboard-tab-selected",
                                children=[
                                    html.Div(
                                        [
                                            html.Section(
                                                [
                                                            _panel_header(
                                                                html,
                                                                ("Asset Inventory", "资产清单"),
                                                                ("VPP / PCC / Asset Map", "VPP / PCC / 资产映射"),
                                                                (
                                                                    "Cross-check each VPP identifier, PCC bus and registered DER fleet before interpreting step-by-step operating states.",
                                                                    "在解读逐步运行状态之前，先核对每个 VPP 标识、PCC 母线和已注册 DER 资源。",
                                                                ),
                                                            ),
                                                    _vpp_summary_table(html, frames),
                                                ],
                                                className="panel",
                                            ),
                                            _voltage_levels_panel(html, frames),
                                            _first_person_panel(html, frames),
                                            _fr_doe_panel(html, frames),
                                            _training_status_panel(html, frames),
                                            html.Div(
                                                [
                                                    html.Section(
                                                        [
                                                            _panel_header(
                                                                html,
                                                                ("Security Envelope", "安全包络"),
                                                                ("Grid Overview", "电网总览"),
                                                                (
                                                                    "Minimum voltage, maximum voltage and normalized line-loading envelopes summarize operating security over the full horizon.",
                                                                    "最低电压、最高电压与归一化线路负载率包络共同概括全时域运行安全状态。",
                                                                ),
                                                            ),
                                                            dcc.Graph(
                                                                id="dashboard-summary-graph",
                                                                figure=_localize_figure(_summary_figure(go, frames), LANG_EN, "summary"),
                                                                className="graph-shell",
                                                            ),
                                                        ],
                                                        className="panel",
                                                    ),
                                                    html.Section(
                                                        [
                                                            _panel_header(
                                                                html,
                                                                ("Driving Profiles", "驱动曲线"),
                                                                ("Price / Load / PV Forecast", "电价 / 负荷 / PV 预测"),
                                                                (
                                                                    "These exogenous profiles explain why congestion, charging and dispatch patterns evolve over time.",
                                                                    "这些外生曲线解释了拥塞、充电和调度模式为何会随时间演化。",
                                                                ),
                                                            ),
                                                            dcc.Graph(
                                                                id="dashboard-profile-graph",
                                                                figure=_localize_figure(profile_figure(go, frames), LANG_EN, "profile"),
                                                                className="graph-shell",
                                                            ),
                                                        ],
                                                        className="panel",
                                                    ),
                                                ],
                                                className="panel-grid panel-grid-two",
                                            ),
                                            html.Div(
                                                [
                                                    html.Section(
                                                        [
                                                            _panel_header(
                                                                html,
                                                                ("Branch Stress", "支路压力"),
                                                                ("Every-Line Power Flow", "全线路潮流"),
                                                                (
                                                                    "Read the matrix as signed active power, then use the loading envelope and peak-branch ranking to separate flow direction from thermal stress without crowding the topology replay.",
                                                                    "先将矩阵视作带符号有功功率，再结合负载包络和峰值支路排名，把潮流方向与热约束压力区分开来，而不挤占拓扑回放空间。",
                                                                ),
                                                            ),
                                                            dcc.Graph(
                                                                id="dashboard-overview-edge-flow-graph",
                                                                figure=_localize_figure(edge_flow_figure(go, frames), LANG_EN, "edge_flow"),
                                                                className="graph-shell",
                                                            ),
                                                        ],
                                                        className="panel",
                                                    ),
                                                    html.Section(
                                                        [
                                                            _panel_header(
                                                                html,
                                                                ("Aggregation", "聚合"),
                                                                ("VPP Power", "VPP 功率"),
                                                                (
                                                                    "Positive active power injects into the grid; negative active power absorbs from it under the project sign convention.",
                                                                    "按照项目符号约定，正有功表示向电网注入，负有功表示从电网吸收。",
                                                                ),
                                                            ),
                                                            dcc.Graph(
                                                                id="dashboard-vpp-graph",
                                                                figure=_localize_figure(
                                                                    vpp_figure(go, frame_or_empty(frames, "vpp_state"), frames),
                                                                    LANG_EN,
                                                                    "vpp",
                                                                ),
                                                                className="graph-shell",
                                                            ),
                                                        ],
                                                        className="panel",
                                                    ),
                                                ],
                                                className="panel-grid panel-grid-two",
                                            ),
                                            _metric_calculation_panel(html),
                                        ],
                                        className="tab-stack",
                                    )
                                ],
                            ),
                            dcc.Tab(
                                label="Topology / 拓扑",
                                className="dashboard-tab",
                                selected_className="dashboard-tab dashboard-tab-selected",
                                children=[
                                    html.Div(
                                        [
                                            _topology_legend_panel(html),
                                            html.Section(
                                                [
                                                    _panel_header(
                                                        html,
                                                        ("Spatial Replay", "空间回放"),
                                                        ("Topology Replay", "拓扑回放"),
                                                        (
                                                            "Move the slider to step through feeder states. Bus labels show nominal kV, feeder labels show branch voltage class, and the right-side legend/colorbar stay outside the schematic to avoid overlap.",
                                                            "移动滑块可逐步回放馈线状态。母线标签显示额定 kV，馈线标签显示支路电压等级，右侧图例和色条置于示意图外避免重叠。",
                                                        ),
                                                    ),
                                                    dcc.Slider(
                                                        id="dashboard-step-slider",
                                                        min=min_step,
                                                        max=max_step,
                                                        step=1,
                                                        value=min_step,
                                                        marks=_step_slider_marks(frames, min_step, max_step),
                                                    ),
                                                    dcc.Graph(id="dashboard-topology-graph", className="graph-shell"),
                                                ],
                                                className="panel",
                                            ),
                                            html.Section(
                                                [
                                                    _panel_header(
                                                        html,
                                                        ("Horizon Scan", "全时域扫描"),
                                                        ("Every-Line Power Flow", "全线路潮流"),
                                                        (
                                                            "The top matrix shows signed active power by branch, while the lower panels show the system loading envelope and the worst peak-loaded lines.",
                                                            "上方矩阵显示各支路带符号有功功率，下方图则显示系统负载包络和峰值负载最严重的线路。",
                                                        ),
                                                    ),
                                                    dcc.Graph(
                                                        id="dashboard-topology-edge-flow-graph",
                                                        figure=_localize_figure(edge_flow_figure(go, frames), LANG_EN, "edge_flow"),
                                                        className="graph-shell",
                                                    ),
                                                ],
                                                className="panel",
                                            ),
                                        ],
                                        className="tab-stack",
                                    )
                                ],
                            ),
                            dcc.Tab(
                                label="VPP / DER",
                                className="dashboard-tab",
                                selected_className="dashboard-tab dashboard-tab-selected",
                                children=[
                                    html.Div(
                                        [
                                            html.Section(
                                                [
                                                    _panel_header(
                                                        html,
                                                        ("Dispatch Narrative", "调度说明"),
                                                        ("One-Day VPP Dispatch Instructions", "单日 VPP 调度指令"),
                                                        (
                                                            "Generated from actual simulated VPP output, DER response, price, load and PV forecast profiles for the first 24-hour window.",
                                                            "基于首个 24 小时窗口内的模拟 VPP 出力、DER 响应、电价、负荷与 PV 预测曲线生成。",
                                                        ),
                                                    ),
                                                    _dispatch_instruction_table(html, frames),
                                                ],
                                                className="panel",
                                            ),
                                            html.Section(
                                                [
                                                    _panel_header(
                                                        html,
                                                        ("Dispatch Detail", "调度细节"),
                                                        ("DER Dispatch", "DER 调度"),
                                                        (
                                                            "Read individual resource trajectories here when VPP aggregation hides internal balancing between storage, generation and flexible demand.",
                                                            "当 VPP 聚合曲线掩盖储能、发电与柔性负荷之间的内部平衡时，可在这里查看单个资源轨迹。",
                                                        ),
                                                    ),
                                                    dcc.Graph(
                                                        id="dashboard-der-graph",
                                                        figure=_localize_figure(der_figure(go, der_state, frames), LANG_EN, "der"),
                                                        className="graph-shell",
                                                    ),
                                                ],
                                                className="panel",
                                            ),
                                            html.Section(
                                                [
                                                    _panel_header(
                                                        html,
                                                        ("Registry", "注册表"),
                                                        ("Asset Registry", "资产注册表"),
                                                        (
                                                            "This table keeps device identity, pandapower element mapping and VPP ownership visible next to the dispatch plot.",
                                                            "这张表在调度图旁保留设备身份、pandapower 元件映射和 VPP 归属信息。",
                                                        ),
                                                    ),
                                                    _html_table(
                                                        html,
                                                        assets,
                                                        columns=[
                                                            "der_id",
                                                            "name",
                                                            "vpp_id",
                                                            "vpp_name",
                                                            "bus_id",
                                                            "der_type",
                                                            "pp_element_type",
                                                            "pp_element_index",
                                                        ],
                                                        max_rows=500,
                                                    ),
                                                ],
                                                className="panel",
                                            ),
                                        ],
                                        className="tab-stack",
                                    )
                                ],
                            ),
                            dcc.Tab(
                                label="Alerts / 告警",
                                className="dashboard-tab",
                                selected_className="dashboard-tab dashboard-tab-selected",
                                children=[
                                    html.Div(
                                        [
                                            html.Section(
                                                [
                                                    _panel_header(
                                                        html,
                                                        ("Event Review", "事件复盘"),
                                                        ("Constraint Violations", "约束越限"),
                                                        (
                                                            "The first 500 alert rows are shown here for quick inspection. Correlate step IDs with the Topology tab when you need feeder location context.",
                                                            "这里展示前 500 条告警记录以便快速检查。需要馈线位置上下文时，可将 step ID 与拓扑页签联动查看。",
                                                        ),
                                                    ),
                                                    _html_table(
                                                        html,
                                                        alerts,
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
                                                        ],
                                                        max_rows=500,
                                                    ),
                                                ],
                                                className="panel",
                                            ),
                                        ],
                                        className="tab-stack",
                                    )
                                ],
                            ),
                        ]
                        ,
                        className="tabs-shell",
                        parent_className="tabs-parent",
                    ),
                ],
                className="app-main",
            ),
        ]
    )

    app.clientside_callback(
        """
        function(enClicks, zhClicks) {
          var storageKey = 'vpp-dso-dashboard-lang';
          var triggered = dash_clientside.callback_context.triggered;
          var lang = 'en';
          try {
            lang = window.localStorage.getItem(storageKey) || 'en';
          } catch (err) {
            lang = 'en';
          }
          if (triggered && triggered.length) {
            if (triggered[0].prop_id.indexOf('dashboard-lang-zh') === 0) {
              lang = 'zh';
            } else if (triggered[0].prop_id.indexOf('dashboard-lang-en') === 0) {
              lang = 'en';
            }
          }
          return lang;
        }
        """,
        Output("dashboard-language", "data"),
        Input("dashboard-lang-en", "n_clicks"),
        Input("dashboard-lang-zh", "n_clicks"),
    )

    app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      :root {
        --bg: #edf3f8;
        --panel: rgba(255, 255, 255, 0.95);
        --panel-soft: linear-gradient(180deg, rgba(244, 248, 252, 0.98), rgba(255, 255, 255, 0.98));
        --border: #d6e0ea;
        --text: #102033;
        --muted: #536779;
        --accent: #1177c3;
        --shadow: 0 14px 30px rgba(15, 23, 42, 0.08);
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        background:
          radial-gradient(circle at top left, rgba(17, 119, 195, 0.12), transparent 28%),
          linear-gradient(180deg, #dfeaf4 0%, var(--bg) 22%, #f6f9fc 100%);
        color: var(--text);
      }
      .lang-copy { display: none; }
      html[data-lang="en"] .lang-en.lang-inline,
      html[data-lang="zh"] .lang-zh.lang-inline { display: inline; }
      html[data-lang="en"] .lang-en.lang-block,
      html[data-lang="zh"] .lang-zh.lang-block { display: block; }
      .eyebrow {
        margin: 0;
        color: var(--accent);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }
      .hero-eyebrow { color: #8fd0ff; }
      .hero-topline {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 12px;
        align-items: center;
      }
      .language-toolbar {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.14);
      }
      .language-label {
        color: #dbe7f3;
        font-size: 12px;
        padding: 0 6px 0 8px;
      }
      .lang-button {
        border: 0;
        border-radius: 999px;
        padding: 7px 12px;
        background: transparent;
        color: #dbe7f3;
        font-size: 12px;
        font-weight: 700;
        cursor: pointer;
      }
      .lang-button.is-active {
        background: rgba(143, 208, 255, 0.2);
        color: white;
      }
      .lang-button:focus-visible {
        outline: 2px solid rgba(143, 208, 255, 0.5);
        outline-offset: 2px;
      }
      .app-header {
        display: grid;
        grid-template-columns: minmax(0, 2.6fr) minmax(250px, 1fr);
        gap: 18px;
        background:
          linear-gradient(135deg, rgba(7, 24, 42, 0.96), rgba(18, 52, 86, 0.96)),
          linear-gradient(90deg, rgba(74, 144, 226, 0.22), rgba(74, 144, 226, 0));
        color: white;
        padding: 28px 30px 22px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
      }
      .app-header h1 { margin: 4px 0 10px 0; font-size: clamp(28px, 4vw, 40px); line-height: 1.08; }
      .hero-copy { margin: 0; color: rgba(226, 232, 240, 0.96); font-size: 15px; line-height: 1.55; }
      .hero-language-note { margin: 10px 0 0 0; color: rgba(219, 231, 243, 0.84); font-size: 12px; line-height: 1.5; }
      .hero-copy-wrap { min-width: 0; }
      .hero-brief {
        align-self: end;
        padding: 16px 18px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(8px);
      }
      .hero-brief-label {
        color: #8fd0ff;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
      }
      .hero-brief p { margin: 0; color: #dbe7f3; font-size: 13px; line-height: 1.6; }
      .app-main { padding: 20px 28px 34px; }
      .mission-panel, .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
      }
      .mission-panel { margin-bottom: 18px; }
      .panel { margin-top: 14px; }
      .panel-soft { background: var(--panel-soft); }
      .panel-heading {
        display: flex;
        justify-content: space-between;
        align-items: end;
        gap: 16px;
        margin-bottom: 14px;
      }
      .panel-title { min-width: 0; }
      .panel h2 { margin: 4px 0 0 0; font-size: 22px; letter-spacing: -0.02em; }
      .panel h3, .mission-panel h3 { margin: 0 0 8px 0; font-size: 15px; }
      .panel-note {
        margin: 0;
        max-width: 520px;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.55;
      }
      .guide-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
      }
      .guide-grid-compact { margin-bottom: 12px; }
      .guide-card {
        padding: 14px 15px;
        border-radius: 14px;
        background: var(--panel-soft);
        border: 1px solid var(--border);
      }
      .guide-card p { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.55; }
      .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
        margin-bottom: 16px;
      }
      .kpi-card {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(245, 249, 252, 0.98));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
      }
      .kpi-card span {
        display: block;
        color: var(--muted);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
      }
      .kpi-card strong { font-size: 22px; letter-spacing: -0.02em; }
      .tabs-parent { margin-top: 6px; }
      .tabs-shell {
        background: transparent !important;
        border: 0 !important;
      }
      .dashboard-tab {
        display: inline-flex !important;
        align-items: center;
        justify-content: center;
        padding: 10px 14px !important;
        margin-right: 8px !important;
        border: 1px solid var(--border) !important;
        border-radius: 999px !important;
        background: rgba(255, 255, 255, 0.72) !important;
        color: var(--muted) !important;
        font-weight: 600;
      }
      .dashboard-tab-selected {
        background: rgba(17, 119, 195, 0.12) !important;
        border-color: rgba(17, 119, 195, 0.35) !important;
        color: #0f4f85 !important;
      }
      .dashboard-tab:focus, .dashboard-tab-selected:focus {
        outline: 2px solid rgba(17, 119, 195, 0.32) !important;
        outline-offset: 2px;
      }
      .tab-stack { margin-top: 12px; }
      .panel-grid {
        display: grid;
        gap: 14px;
      }
      .panel-grid-two {
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      }
      .graph-shell { min-height: 320px; }
      .table-wrap {
        width: 100%;
        overflow-x: auto;
        margin-bottom: 4px;
        border: 1px solid #e5edf4;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.96);
      }
      .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }
      .data-table th, .data-table td {
        border-bottom: 1px solid #e5edf4;
        padding: 9px 10px;
        text-align: left;
        white-space: normal;
        min-width: 110px;
        vertical-align: top;
        line-height: 1.45;
      }
      .data-table th {
        position: sticky;
        top: 0;
        z-index: 1;
        color: #314155;
        background: #f5f9fc;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }
      .data-table tbody tr:nth-child(even) td { background: rgba(247, 250, 252, 0.9); }
      .dispatch-vpp-shell { display: grid; gap: 16px; }
      .dispatch-vpp-card {
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px;
        background: var(--panel-soft);
      }
      .dispatch-vpp-head {
        display: flex;
        justify-content: space-between;
        gap: 14px;
        align-items: start;
        margin-bottom: 14px;
      }
      .dispatch-vpp-title { margin: 4px 0 0 0; font-size: 19px; letter-spacing: -0.02em; }
      .dispatch-count-pill {
        display: inline-flex;
        align-items: center;
        padding: 7px 10px;
        border-radius: 999px;
        background: rgba(17, 119, 195, 0.12);
        border: 1px solid rgba(17, 119, 195, 0.2);
        color: #0f4f85;
        font-size: 12px;
        font-weight: 700;
        white-space: nowrap;
      }
      .dispatch-vpp-list { display: grid; gap: 12px; }
      .dispatch-window-card {
        border-radius: 14px;
        border: 1px solid #dbe6f0;
        background: rgba(255, 255, 255, 0.96);
        padding: 14px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
      }
      .dispatch-window-meta {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 10px;
      }
      .dispatch-metric {
        padding: 10px 11px;
        border-radius: 12px;
        background: #f7fafc;
        border: 1px solid #e4edf5;
      }
      .dispatch-metric-label {
        display: block;
        color: var(--muted);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 6px;
      }
      .dispatch-metric strong { display: block; font-size: 14px; line-height: 1.45; color: #17324c; }
      .dispatch-window-body {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
        margin-top: 12px;
      }
      .dispatch-copy-card {
        padding: 13px 14px;
        border-radius: 12px;
        background: linear-gradient(180deg, rgba(245, 249, 252, 0.96), rgba(255, 255, 255, 1));
        border: 1px solid #e4edf5;
      }
      .dispatch-copy-card h3 { margin-bottom: 10px; }
      .dispatch-copy-body .lang-block {
        color: #304559;
        font-size: 13px;
        line-height: 1.62;
        white-space: pre-wrap;
      }
      .rc-slider-mark-text { color: var(--muted); font-size: 12px; }
      .rc-slider-track { background-color: rgba(17, 119, 195, 0.42); }
      .rc-slider-handle {
        border-color: #1177c3;
        box-shadow: 0 0 0 4px rgba(17, 119, 195, 0.12);
      }
      @media (max-width: 960px) {
        .app-header { grid-template-columns: 1fr; }
        .panel-heading { flex-direction: column; align-items: flex-start; }
        .dispatch-vpp-head { flex-direction: column; }
      }
      @media (max-width: 640px) {
        .app-main { padding: 18px 14px 28px; }
        .app-header { padding: 22px 16px 18px; }
        .mission-panel, .panel { padding: 15px 14px; border-radius: 16px; }
        .panel h2 { font-size: 19px; }
      }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
    <script>
      (function() {
        var storageKey = 'vpp-dso-dashboard-lang';
        function applyLang(lang) {
          var root = document.documentElement;
          root.setAttribute('data-lang', lang);
          root.setAttribute('lang', lang === 'zh' ? 'zh-CN' : 'en');
          document.querySelectorAll('[data-lang-switch]').forEach(function(button) {
            var active = button.getAttribute('data-lang-switch') === lang;
            button.classList.toggle('is-active', active);
            button.setAttribute('aria-pressed', active ? 'true' : 'false');
          });
        }
        function remember(lang) {
          try {
            window.localStorage.setItem(storageKey, lang);
          } catch (err) {
            void err;
          }
        }
        function initialLang() {
          try {
            return window.localStorage.getItem(storageKey) || 'en';
          } catch (err) {
            return 'en';
          }
        }
        document.addEventListener('click', function(event) {
          var button = event.target.closest('[data-lang-switch]');
          if (!button) {
            return;
          }
          var lang = button.getAttribute('data-lang-switch') || 'en';
          applyLang(lang);
          remember(lang);
        });
        var lang = initialLang();
        applyLang(lang);
        var observer = new MutationObserver(function() {
          if (!document.querySelector('[data-lang-switch]')) {
            return;
          }
          applyLang(initialLang());
          observer.disconnect();
        });
        observer.observe(document.documentElement, { childList: true, subtree: true });
      })();
    </script>
  </body>
</html>
"""

    @app.callback(
        Output("dashboard-summary-graph", "figure"),
        Output("dashboard-profile-graph", "figure"),
        Output("dashboard-overview-edge-flow-graph", "figure"),
        Output("dashboard-vpp-graph", "figure"),
        Output("dashboard-topology-edge-flow-graph", "figure"),
        Output("dashboard-der-graph", "figure"),
        Input("dashboard-language", "data"),
    )
    def _update_static_figures(lang: str | None):
        language = _normalize_lang(lang)
        return (
            _localize_figure(_summary_figure(go, frames), language, "summary"),
            _localize_figure(profile_figure(go, frames), language, "profile"),
            _localize_figure(edge_flow_figure(go, frames), language, "edge_flow"),
            _localize_figure(vpp_figure(go, frame_or_empty(frames, "vpp_state"), frames), language, "vpp"),
            _localize_figure(edge_flow_figure(go, frames), language, "edge_flow"),
            _localize_figure(der_figure(go, der_state, frames), language, "der"),
        )

    @app.callback(
        Output("dashboard-topology-graph", "figure"),
        Input("dashboard-step-slider", "value"),
        Input("dashboard-language", "data"),
    )
    def _update_topology(step: int, lang: str | None):
        topology = topology_figure(go, frames)
        lookup = step_time_lookup(frames)
        if topology.frames:
            for frame in topology.frames:
                if str(frame.name) == str(int(step)):
                    topology = go.Figure(data=frame.data, layout=topology.layout)
                    break
        label = lookup.get(int(step), f"step {int(step)}")
        return _localize_figure(topology, _normalize_lang(lang), "topology", step_label=label)

    return app


def run_dashboard(
    data_dir: str | Path = "outputs/dashboard_data",
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
) -> None:
    app = create_dashboard_app(data_dir=data_dir)
    app.run(host=host, port=port, debug=debug)
