from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib import patches
from matplotlib import colormaps
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from vpp_dso_sim.utils.io import ensure_dir


DER_MARKERS = {
    "PVModel": "^",
    "MicroTurbineModel": "s",
    "StorageModel": "D",
    "FlexibleLoadModel": "v",
    "HVACModel": "X",
    "EVCSModel": "P",
}

DER_SHORT_LABELS = {
    "PVModel": "PV",
    "MicroTurbineModel": "MT",
    "StorageModel": "ESS",
    "FlexibleLoadModel": "Flex",
    "HVACModel": "HVAC",
    "EVCSModel": "EVCS",
}

VPP_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
]


def _vpp_color_map(nodes: pd.DataFrame, assets: pd.DataFrame | None = None) -> dict[str, str]:
    ids: set[str] = set()
    if assets is not None and not assets.empty and "vpp_id" in assets:
        ids.update(str(value) for value in assets["vpp_id"].dropna().unique().tolist())
    if not nodes.empty and "vpp_ids" in nodes:
        for value in nodes["vpp_ids"].dropna().tolist():
            ids.update(item for item in str(value).split(",") if item)
    return {vpp_id: VPP_COLORS[i % len(VPP_COLORS)] for i, vpp_id in enumerate(sorted(ids))}


def _truthy_mask(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def _state_at_step(frame: pd.DataFrame, step: int, key: str, value: str) -> dict[Any, float]:
    if frame.empty:
        return {}
    selected = frame[frame["step"] == step]
    return {row[key]: float(row[value]) for _, row in selected.iterrows()}


def _node_positions(nodes: pd.DataFrame) -> dict[int, tuple[float, float]]:
    return {
        int(row["bus_id"]): (float(row["x"]), float(row["y"]))
        for _, row in nodes.dropna(subset=["x", "y"]).iterrows()
    }


def _edge_loading(edge_state: pd.DataFrame, step: int) -> dict[str, float]:
    if edge_state.empty:
        return {}
    selected = edge_state[edge_state["step"] == step]
    return {str(row["edge_id"]): float(row["loading_percent"]) for _, row in selected.iterrows()}


def _edge_state_at_step(edge_state: pd.DataFrame, step: int) -> dict[str, pd.Series]:
    if edge_state.empty:
        return {}
    selected = edge_state[edge_state["step"] == step]
    return {str(row["edge_id"]): row for _, row in selected.iterrows()}


def _asset_offsets(n_assets: int) -> list[tuple[float, float]]:
    if n_assets <= 0:
        return []
    radius = 0.23
    return [
        (radius * math.cos(2.0 * math.pi * i / n_assets), radius * math.sin(2.0 * math.pi * i / n_assets))
        for i in range(n_assets)
    ]


def _draw_edges(
    ax,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    edge_state: pd.DataFrame,
    step: int,
) -> None:
    positions = _node_positions(nodes)
    loading_by_edge = _edge_loading(edge_state, step)
    state_by_edge = _edge_state_at_step(edge_state, step)
    segments = []
    colors = []
    widths = []
    arrow_items = []
    for _, edge in edges.iterrows():
        from_bus = int(edge["from_bus"])
        to_bus = int(edge["to_bus"])
        if from_bus not in positions or to_bus not in positions:
            continue
        loading = loading_by_edge.get(str(edge["edge_id"]), 0.0)
        edge_id = str(edge["edge_id"])
        from_xy = positions[from_bus]
        to_xy = positions[to_bus]
        segments.append([from_xy, to_xy])
        colors.append(loading)
        widths.append(1.0 + min(max(loading, 0.0), 150.0) / 35.0)
        arrow_items.append((edge_id, from_xy, to_xy, state_by_edge.get(edge_id), edge))

    if not segments:
        return
    collection = LineCollection(
        segments,
        cmap="YlOrRd",
        norm=Normalize(vmin=0.0, vmax=120.0),
        linewidths=widths,
        alpha=0.9,
        zorder=1,
    )
    collection.set_array(pd.Series(colors).to_numpy())
    ax.add_collection(collection)
    cbar = plt.colorbar(collection, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Line/trafo loading (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    _draw_feeder_switches(ax, arrow_items)
    _draw_feeder_voltage_labels(ax, arrow_items)
    _draw_power_flow_annotations(ax, arrow_items)


def _draw_feeder_switches(ax, arrow_items: list[tuple[str, tuple[float, float], tuple[float, float], pd.Series | None, pd.Series]]) -> None:
    for _, from_xy, to_xy, _, _ in arrow_items:
        x = from_xy[0] + 0.16 * (to_xy[0] - from_xy[0])
        y = from_xy[1] + 0.16 * (to_xy[1] - from_xy[1])
        ax.add_patch(
            patches.Rectangle(
                (x - 0.055, y - 0.055),
                0.11,
                0.11,
                facecolor="white",
                edgecolor="#0f172a",
                linewidth=1.0,
                zorder=4,
            )
        )


def _draw_feeder_voltage_labels(ax, arrow_items: list[tuple[str, tuple[float, float], tuple[float, float], pd.Series | None, pd.Series]]) -> None:
    for _, from_xy, to_xy, _, edge in arrow_items:
        label = str(edge.get("voltage_level_label", ""))
        if not label:
            continue
        dx = to_xy[0] - from_xy[0]
        dy = to_xy[1] - from_xy[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        x = (from_xy[0] + to_xy[0]) / 2.0 + dy / length * 0.16
        y = (from_xy[1] + to_xy[1]) / 2.0 - dx / length * 0.16
        ax.text(
            x,
            y,
            label.replace(" ", ""),
            ha="center",
            va="center",
            fontsize=8.5,
            color="#475569",
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "#e5e7eb", "alpha": 0.82},
            zorder=5,
        )


def _draw_power_flow_annotations(ax, arrow_items: list[tuple[str, tuple[float, float], tuple[float, float], pd.Series | None, pd.Series]]) -> None:
    for edge_id, from_xy, to_xy, state, _ in arrow_items:
        if state is None:
            continue
        p_from = float(state.get("p_from_mw", 0.0))
        q_from = float(state.get("q_from_mvar", 0.0))
        flow_p = abs(p_from)
        if flow_p < 1e-4:
            continue
        start = from_xy if p_from >= 0.0 else to_xy
        end = to_xy if p_from >= 0.0 else from_xy
        sx, sy = start
        ex, ey = end
        mid_x = sx + 0.55 * (ex - sx)
        mid_y = sy + 0.55 * (ey - sy)
        ax.annotate(
            "",
            xy=(mid_x, mid_y),
            xytext=(sx + 0.35 * (ex - sx), sy + 0.35 * (ey - sy)),
            arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 0.9, "alpha": 0.75},
            zorder=3,
        )
        label_x = (from_xy[0] + to_xy[0]) / 2.0
        label_y = (from_xy[1] + to_xy[1]) / 2.0
        ax.text(
            label_x,
            label_y,
            f"{flow_p:.2f} MW\n{abs(q_from):.2f} MVAr",
            ha="center",
            va="center",
            fontsize=8.3,
            color="#111827",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.84},
            zorder=6,
        )


def _draw_buses(ax, nodes: pd.DataFrame, bus_state: pd.DataFrame, step: int) -> None:
    vm_by_bus = _state_at_step(bus_state, step, "bus_id", "vm_pu")
    cmap = colormaps["RdYlGn"]
    norm = Normalize(vmin=0.93, vmax=1.07)
    for _, row in nodes.iterrows():
        bus_id = int(row["bus_id"])
        if pd.isna(row["x"]) or pd.isna(row["y"]):
            continue
        x = float(row["x"])
        y = float(row["y"])
        vm = vm_by_bus.get(bus_id, 1.0)
        color = cmap(norm(vm))
        height = 0.52 if _truthy_value(row.get("is_slack", False)) else 0.40
        ax.plot(
            [x, x],
            [y - height / 2.0, y + height / 2.0],
            color=color,
            linewidth=7,
            solid_capstyle="butt",
            zorder=5,
        )
        ax.plot(
            [x, x],
            [y - height / 2.0, y + height / 2.0],
            color="#0f172a",
            linewidth=1.1,
            solid_capstyle="butt",
            zorder=6,
        )
        if _truthy_value(row.get("is_slack", False)):
            _draw_grid_source_symbol(ax, x - 0.38, y)

    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.025, pad=0.055)
    cbar.set_label("Bus voltage (pu)", fontsize=10)

    for _, row in nodes.iterrows():
        if pd.isna(row["x"]) or pd.isna(row["y"]):
            continue
        ax.text(
            float(row["x"]),
            float(row["y"]) + 0.16,
            f"{int(row['bus_id'])}\n{float(row.get('vn_kv', 0.0)):.2f}kV",
            ha="center",
            va="bottom",
            fontsize=8.6,
            color="#111827",
            zorder=5,
        )


def _truthy_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _draw_grid_source_symbol(ax, x: float, y: float) -> None:
    ax.add_patch(patches.Circle((x, y), 0.17, facecolor="#eff6ff", edgecolor="#0f172a", linewidth=1.2, zorder=7))
    wave_x = [x - 0.12, x - 0.06, x, x + 0.06, x + 0.12]
    wave_y = [y, y + 0.07, y, y - 0.07, y]
    ax.plot(wave_x, wave_y, color="#0f172a", linewidth=1.0, zorder=8)
    ax.plot([x, x], [y - 0.17, y - 0.30], color="#0f172a", linewidth=1.2, zorder=7)
    ax.plot([x - 0.12, x + 0.12], [y - 0.30, y - 0.30], color="#0f172a", linewidth=1.2, zorder=7)


def _draw_pcc_switch_symbol(ax, x: float, y: float, color: str) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x - 0.13, y - 0.12),
            0.26,
            0.24,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor="white",
            edgecolor=color,
            linewidth=1.4,
            zorder=7,
        )
    )
    ax.plot([x - 0.09, x + 0.09], [y + 0.06, y + 0.06], color=color, linewidth=1.4, zorder=8)
    ax.plot([x - 0.09, x + 0.09], [y - 0.06, y - 0.06], color=color, linewidth=1.4, zorder=8)
    ax.add_patch(patches.Circle((x - 0.04, y), 0.026, facecolor="white", edgecolor="#0f172a", linewidth=0.8, zorder=9))
    ax.add_patch(patches.Circle((x + 0.06, y), 0.026, facecolor="white", edgecolor="#0f172a", linewidth=0.8, zorder=9))
    ax.plot([x - 0.025, x + 0.045], [y, y + 0.055], color="#0f172a", linewidth=1.0, zorder=9)


def _draw_vpps(ax, nodes: pd.DataFrame, assets: pd.DataFrame) -> None:
    colors_by_vpp = _vpp_color_map(nodes, assets)
    pcc_nodes = nodes[_truthy_mask(nodes["is_pcc"])]
    for _, row in pcc_nodes.iterrows():
        if pd.isna(row["x"]) or pd.isna(row["y"]):
            continue
        vpp_ids = [item for item in str(row.get("vpp_ids", "")).split(",") if item]
        for i, vpp_id in enumerate(vpp_ids):
            color = colors_by_vpp.get(vpp_id, "#6b7280")
            ax.scatter(
                [float(row["x"])],
                [float(row["y"])],
                s=340 + 70 * i,
                facecolors="none",
                edgecolors=color,
                linewidths=1.7,
                zorder=3,
            )
            _draw_pcc_switch_symbol(ax, float(row["x"]) + 0.34, float(row["y"]) + 0.23 + 0.13 * i, color)
            ax.text(
                float(row["x"]) + 0.12,
                float(row["y"]) - 0.24 - 0.12 * i,
                vpp_id,
                fontsize=9,
                color=color,
                zorder=6,
            )


def _draw_asset_pictogram(
    ax,
    der_type: str,
    x: float,
    y: float,
    color: str,
    edge_color: str,
) -> None:
    size = 0.36
    left = x - size / 2
    bottom = y - size / 2
    ax.add_patch(
        patches.FancyBboxPatch(
            (left, bottom),
            size,
            size,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            facecolor="white",
            edgecolor=color,
            linewidth=1.6,
            zorder=7,
        )
    )
    if der_type == "PVModel":
        ax.add_patch(patches.Circle((x + 0.10, y + 0.09), 0.055, facecolor="#facc15", edgecolor="#ca8a04", linewidth=0.8, zorder=8))
        panel = patches.Polygon(
            [(x - 0.13, y - 0.05), (x + 0.07, y - 0.08), (x + 0.13, y - 0.18), (x - 0.09, y - 0.15)],
            closed=True,
            facecolor="#1d4ed8",
            edgecolor="#0f172a",
            linewidth=0.8,
            zorder=8,
        )
        ax.add_patch(panel)
        for shift in (-0.05, 0.02, 0.08):
            ax.plot([x + shift, x + shift + 0.04], [y - 0.06, y - 0.16], color="#93c5fd", linewidth=0.55, zorder=9)
        ax.plot([x - 0.11, x + 0.11], [y - 0.10, y - 0.13], color="#93c5fd", linewidth=0.55, zorder=9)
    elif der_type == "EVCSModel":
        ax.add_patch(patches.FancyBboxPatch((x - 0.11, y - 0.14), 0.17, 0.27, boxstyle="round,pad=0.01", facecolor="#e0f2fe", edgecolor="#0f172a", linewidth=0.8, zorder=8))
        ax.add_patch(patches.Rectangle((x - 0.075, y + 0.03), 0.09, 0.06, facecolor="#38bdf8", edgecolor="#0369a1", linewidth=0.6, zorder=9))
        ax.plot([x + 0.06, x + 0.17, x + 0.13], [y + 0.07, y + 0.01, y - 0.08], color=color, linewidth=1.6, zorder=9)
        ax.add_patch(patches.Rectangle((x + 0.10, y - 0.11), 0.06, 0.08, facecolor="#111827", edgecolor="#111827", linewidth=0.6, zorder=9))
        ax.text(x - 0.035, y - 0.06, "V", fontsize=5.5, fontweight="bold", color="#ca8a04", zorder=10)
    elif der_type == "StorageModel":
        ax.add_patch(patches.FancyBboxPatch((x - 0.14, y - 0.06), 0.25, 0.14, boxstyle="round,pad=0.01", facecolor="#dcfce7", edgecolor="#0f172a", linewidth=0.8, zorder=8))
        ax.add_patch(patches.Rectangle((x + 0.11, y - 0.02), 0.03, 0.06, facecolor="#0f172a", zorder=8))
        ax.add_patch(patches.Rectangle((x - 0.10, y - 0.025), 0.13, 0.07, facecolor=color, edgecolor=color, linewidth=0.6, zorder=9))
    elif der_type == "HVACModel":
        ax.add_patch(patches.Circle((x, y), 0.13, facecolor="#ecfeff", edgecolor="#0f172a", linewidth=0.8, zorder=8))
        ax.add_patch(patches.Circle((x, y), 0.028, facecolor=color, edgecolor="#0f172a", linewidth=0.5, zorder=10))
        blades = [
            [(x, y + 0.02), (x + 0.11, y + 0.08), (x + 0.08, y - 0.01)],
            [(x + 0.02, y - 0.02), (x + 0.01, y - 0.13), (x - 0.07, y - 0.07)],
            [(x - 0.02, y), (x - 0.11, y + 0.06), (x - 0.06, y + 0.10)],
        ]
        for blade in blades:
            ax.add_patch(patches.Polygon(blade, closed=True, facecolor="#67e8f9", edgecolor="#0891b2", linewidth=0.5, zorder=9))
    elif der_type == "MicroTurbineModel":
        ax.plot([x, x - 0.05], [y + 0.01, y - 0.14], color="#0f172a", linewidth=1.1, zorder=8)
        ax.plot([x, x + 0.05], [y + 0.01, y - 0.14], color="#0f172a", linewidth=1.1, zorder=8)
        ax.add_patch(patches.Circle((x, y + 0.05), 0.035, facecolor=color, edgecolor="#0f172a", linewidth=0.6, zorder=10))
        ax.plot([x, x + 0.12], [y + 0.05, y + 0.13], color="#4338ca", linewidth=1.2, zorder=9)
        ax.plot([x, x + 0.03], [y + 0.05, y - 0.08], color="#4338ca", linewidth=1.2, zorder=9)
        ax.plot([x, x - 0.12], [y + 0.05, y + 0.12], color="#4338ca", linewidth=1.2, zorder=9)
    elif der_type == "FlexibleLoadModel":
        house = patches.Polygon(
            [(x - 0.13, y - 0.12), (x - 0.13, y + 0.03), (x, y + 0.13), (x + 0.13, y + 0.03), (x + 0.13, y - 0.12)],
            closed=True,
            facecolor="#fef3c7",
            edgecolor="#0f172a",
            linewidth=0.8,
            zorder=8,
        )
        ax.add_patch(house)
        ax.plot([x - 0.08, x + 0.08], [y + 0.02, y + 0.02], color=color, linewidth=1.5, zorder=9)
        ax.add_patch(patches.Circle((x + 0.02, y + 0.02), 0.025, facecolor=color, zorder=10))
    else:
        ax.add_patch(patches.Circle((x, y), 0.12, facecolor="#f1f5f9", edgecolor=edge_color, linewidth=1.0, zorder=8))


def _draw_assets(
    ax,
    nodes: pd.DataFrame,
    assets: pd.DataFrame,
    der_state: pd.DataFrame,
    step: int,
) -> None:
    if assets.empty:
        return
    positions = _node_positions(nodes)
    colors_by_vpp = _vpp_color_map(nodes, assets)
    current = der_state[der_state["step"] == step] if not der_state.empty else pd.DataFrame()
    state_by_der = {str(row["der_id"]): row for _, row in current.iterrows()} if not current.empty else {}
    grouped = assets.groupby("bus_id", sort=True)
    for bus_id, group in grouped:
        bus_id_int = int(bus_id)
        if bus_id_int not in positions:
            continue
        offsets = _asset_offsets(len(group))
        for (_, asset), (dx, dy) in zip(group.iterrows(), offsets):
            der_type = str(asset["der_type"])
            state = state_by_der.get(str(asset["der_id"]))
            p_mw = float(state.get("p_mw", 0.0)) if state is not None else 0.0
            color = colors_by_vpp.get(str(asset.get("vpp_id", "")), "#6b7280")
            edge_color = "#111827" if p_mw >= 0 else "#f97316"
            x, y = positions[bus_id_int]
            asset_x = x + dx
            asset_y = y + dy
            _draw_asset_pictogram(ax, der_type, asset_x, asset_y, color, edge_color)
            ax.text(
                asset_x + 0.10,
                asset_y - 0.05,
                _compact_der_label(asset, state),
                ha="left",
                va="top",
                fontsize=8.1,
                color="#111827",
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.82},
                zorder=8,
            )


def _compact_der_label(asset: pd.Series, state: pd.Series | None) -> str:
    der_type = str(asset["der_type"])
    prefix = DER_SHORT_LABELS.get(der_type, der_type.replace("Model", ""))
    der_id = str(asset["der_id"])
    vpp_id = str(asset.get("vpp_id", ""))
    if state is None:
        return f"{vpp_id}\n{prefix} {der_id}"
    p_mw = float(state.get("p_mw", 0.0))
    parts = [f"{vpp_id}", f"{prefix} {der_id}", f"P={p_mw:.2f}MW"]
    if pd.notna(state.get("available_p_mw", None)):
        parts.append(f"Avail={float(state['available_p_mw']):.2f}")
    if pd.notna(state.get("soc", None)):
        parts.append(f"SOC={float(state['soc']):.2f}")
    if pd.notna(state.get("average_soc", None)):
        parts.append(f"EV={float(state['average_soc']):.2f}")
    if pd.notna(state.get("indoor_temp", None)):
        parts.append(f"T={float(state['indoor_temp']):.1f}C")
    return "\n".join(parts)


def _draw_alert_overlay(
    ax,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    alerts: pd.DataFrame,
    step: int,
) -> None:
    if alerts.empty:
        return
    positions = _node_positions(nodes)
    current = alerts[alerts["step"] == step]
    for _, alert in current.iterrows():
        element_type = str(alert["element_type"])
        element_id = str(alert["element_id"])
        if element_type == "bus":
            try:
                bus_id = int(element_id)
            except ValueError:
                continue
            if bus_id in positions:
                x, y = positions[bus_id]
                ax.scatter([x], [y], s=520, facecolors="none", edgecolors="#b91c1c", linewidths=2.2, zorder=8)
        elif element_type in {"line", "trafo"}:
            edge_id = f"{element_type}_{element_id}"
            row = edges[edges["edge_id"] == edge_id]
            if row.empty:
                continue
            edge = row.iloc[0]
            from_bus = int(edge["from_bus"])
            to_bus = int(edge["to_bus"])
            if from_bus in positions and to_bus in positions:
                xs = [positions[from_bus][0], positions[to_bus][0]]
                ys = [positions[from_bus][1], positions[to_bus][1]]
                ax.plot(xs, ys, color="#b91c1c", linewidth=4.5, alpha=0.75, zorder=2)


def _asset_legend_handles(nodes: pd.DataFrame, assets: pd.DataFrame) -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="w",
            label=DER_SHORT_LABELS.get(label, label.replace("Model", "")),
            markerfacecolor="#64748b",
            markeredgecolor="#111827",
            markersize=9,
        )
        for label, marker in DER_MARKERS.items()
    ]
    for vpp_id, color in _vpp_color_map(nodes, assets).items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"{vpp_id} PCC/assets",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=9,
            )
        )
    handles.append(Line2D([0], [0], color="#16a34a", linewidth=6, label="Busbar voltage symbol"))
    handles.append(Line2D([0], [0], marker="s", color="w", label="Feeder switch/protection", markerfacecolor="white", markeredgecolor="#0f172a", markersize=7))
    handles.append(Line2D([0], [0], color="#475569", linewidth=0, marker=None, label="Bus/feeder kV label"))
    handles.append(Line2D([0], [0], marker="o", color="w", label="VPP PCC ring", markerfacecolor="none", markeredgecolor="#2563eb", markersize=12))
    handles.append(Line2D([0], [0], marker="o", color="w", label="Alert", markerfacecolor="none", markeredgecolor="#b91c1c", markersize=12))
    return handles


def plot_topology_state(
    frames: dict[str, pd.DataFrame],
    step: int,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    nodes = frames.get("network_nodes", pd.DataFrame())
    edges = frames.get("network_edges", pd.DataFrame())
    assets = frames.get("asset_registry", pd.DataFrame())
    bus_state = frames.get("bus_state", pd.DataFrame())
    edge_state = frames.get("edge_state", pd.DataFrame())
    der_state = frames.get("der_state", pd.DataFrame())
    alerts = frames.get("alert_event", pd.DataFrame())

    if nodes.empty:
        raise ValueError("network_nodes frame is empty; cannot draw topology")

    fig, ax = plt.subplots(figsize=(15, 9))
    _draw_edges(ax, nodes, edges, edge_state, step)
    _draw_buses(ax, nodes, bus_state, step)
    _draw_vpps(ax, nodes, assets)
    _draw_assets(ax, nodes, assets, der_state, step)
    _draw_alert_overlay(ax, nodes, edges, alerts, step)

    ax.set_title(title or f"Network topology state, step {step}")
    ax.set_xlabel("Topology level from substation (schematic)")
    ax.set_ylabel("Radial branch lane (schematic)")
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(handles=_asset_legend_handles(nodes, assets), loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_alert_timeline(frames: dict[str, pd.DataFrame], output_path: str | Path) -> Path:
    alerts = frames.get("alert_event", pd.DataFrame())
    summary = frames.get("step_summary", pd.DataFrame())
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    if alerts.empty:
        max_step = int(summary["step"].max()) if not summary.empty else 0
        ax.plot([0, max_step], [0, 0], color="#16a34a", linewidth=2)
        ax.text(0.5, 0.55, "No constraint violations", transform=ax.transAxes, ha="center", va="center")
    else:
        severity_rank = {"info": 1, "warning": 2, "critical": 3}
        plot_data = alerts.copy()
        plot_data["severity_rank"] = plot_data["severity"].map(severity_rank).fillna(1)
        colors = plot_data["severity"].map({"info": "#2563eb", "warning": "#f59e0b", "critical": "#dc2626"}).fillna("#6b7280")
        ax.scatter(
            plot_data["step"],
            plot_data["severity_rank"],
            c=colors,
            s=70 + 45 * plot_data["magnitude"].clip(lower=0.0),
            alpha=0.85,
        )
        ax.set_yticks([1, 2, 3], ["info", "warning", "critical"])
    ax.set_xlabel("step")
    ax.set_ylabel("severity")
    ax.set_title("Constraint violation timeline")
    ax.grid(True, axis="x", color="#e5e7eb", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_topology_report(
    frames: dict[str, pd.DataFrame],
    output_dir: str | Path = "outputs/figures",
) -> dict[str, Path]:
    out = ensure_dir(output_dir)
    paths: dict[str, Path] = {}
    summary = frames.get("step_summary", pd.DataFrame())
    if summary.empty:
        steps = [0]
        peak_loading_step = 0
        min_voltage_step = 0
    else:
        steps = sorted(summary["step"].astype(int).unique().tolist())
        loading_col = "max_line_loading_percent"
        voltage_col = "min_vm_pu"
        peak_loading_step = int(summary.loc[summary[loading_col].fillna(0.0).idxmax(), "step"]) if loading_col in summary else steps[0]
        min_voltage_step = int(summary.loc[summary[voltage_col].fillna(1.0).idxmin(), "step"]) if voltage_col in summary else steps[0]

    first_step = steps[0]
    paths["topology_step_000"] = plot_topology_state(
        frames,
        first_step,
        out / "topology_step_000.png",
        title=f"Network topology state, step {first_step}",
    )
    paths["topology_step_peak_loading"] = plot_topology_state(
        frames,
        peak_loading_step,
        out / "topology_step_peak_loading.png",
        title=f"Network topology at peak loading, step {peak_loading_step}",
    )
    paths["topology_voltage_min_step"] = plot_topology_state(
        frames,
        min_voltage_step,
        out / "topology_voltage_min_step.png",
        title=f"Network topology at minimum voltage, step {min_voltage_step}",
    )
    paths["alert_timeline"] = plot_alert_timeline(frames, out / "alert_timeline.png")
    return paths
