from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DER_TYPE_LABELS_EN = {
    "PVModel": "PV",
    "MicroTurbineModel": "MT",
    "StorageModel": "ESS",
    "FlexibleLoadModel": "Flexible load",
    "HVACModel": "HVAC",
    "EVCSModel": "EVCS",
}

DER_TYPE_LABELS_ZH = {
    "PVModel": "光伏",
    "MicroTurbineModel": "微型燃机",
    "StorageModel": "储能",
    "FlexibleLoadModel": "柔性负荷",
    "HVACModel": "HVAC",
    "EVCSModel": "充电站",
}

COMMAND_LABELS_EN = {
    "high-price": "high-price dispatch",
    "low-price": "low-price absorption",
    "pv-rich": "PV-rich balancing",
    "high-load": "high-load support",
    "normal": "normal balancing",
}

COMMAND_LABELS_ZH = {
    "high-price": "高价上调",
    "low-price": "低价吸收",
    "pv-rich": "光伏富余平衡",
    "high-load": "高负荷支撑",
    "normal": "常规平衡",
}

REGION_LABELS_EN = {
    "lower-flex": "lower flexibility band",
    "mid-flex": "middle flexibility band",
    "upper-flex": "upper flexibility band",
}

REGION_LABELS_ZH = {
    "lower-flex": "灵活性下沿",
    "mid-flex": "灵活性中段",
    "upper-flex": "灵活性上沿",
}


def _frame(frames: dict[str, pd.DataFrame], name: str) -> pd.DataFrame:
    return frames.get(name, pd.DataFrame()).copy()


def _profile_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    profile = _frame(frames, "profile_state")
    if profile.empty:
        profile = _frame(frames, "step_summary")
    if profile.empty:
        return pd.DataFrame(
            columns=["step", "time_hours", "time_label", "price", "load_scale", "pv_forecast_factor"]
        )
    profile = profile.copy()
    if "time_hours" not in profile:
        profile["time_hours"] = profile["step"].astype(float)
    if "time_label" not in profile:
        profile["time_label"] = profile["time_hours"].map(lambda value: f"{float(value):05.2f} h")
    for column, default in (("price", 0.0), ("load_scale", 1.0), ("pv_forecast_factor", 0.0)):
        if column not in profile:
            profile[column] = default
    return profile[["step", "time_hours", "time_label", "price", "load_scale", "pv_forecast_factor"]]


def _day_slice(data: pd.DataFrame, day_index: int, day_hours: float) -> pd.DataFrame:
    if data.empty or "time_hours" not in data:
        return data.copy()
    start = float(day_index) * float(day_hours)
    end = start + float(day_hours)
    selected = data[(data["time_hours"] >= start) & (data["time_hours"] < end)].copy()
    if selected.empty and day_index == 0:
        selected = data.copy()
    return selected


def _dispatch_region(row: pd.Series) -> str:
    p_min = float(row.get("p_min_mw", 0.0))
    p_max = float(row.get("p_max_mw", 0.0))
    p = float(row.get("p_mw", 0.0))
    span = max(1e-9, p_max - p_min)
    ratio = (p - p_min) / span
    if ratio <= 0.35:
        return "lower-flex"
    if ratio >= 0.75:
        return "upper-flex"
    return "mid-flex"


def _command_category(row: pd.Series) -> str:
    price = float(row.get("price", 0.0))
    pv = float(row.get("pv_forecast_factor", 0.0))
    load = float(row.get("load_scale", 1.0))
    if price >= 100.0:
        return "high-price"
    if price <= 55.0:
        return "low-price"
    if pv >= 0.60:
        return "pv-rich"
    if load >= 1.10:
        return "high-load"
    return "normal"


def _command_type(row: pd.Series) -> str:
    return f"{_command_category(row)} {_dispatch_region(row)}"


def _command_label(command_type: str, lang: str) -> str:
    parts = str(command_type).split()
    category = parts[0] if parts else "normal"
    region = parts[1] if len(parts) > 1 else "mid-flex"
    if lang == "zh":
        return f"{COMMAND_LABELS_ZH.get(category, category)} / {REGION_LABELS_ZH.get(region, region)}"
    return f"{COMMAND_LABELS_EN.get(category, category)} / {REGION_LABELS_EN.get(region, region)}"


def _segment_stats(segment: pd.DataFrame) -> tuple[float, float, float, str]:
    price = float(segment["price"].mean()) if "price" in segment else 0.0
    pv = float(segment["pv_forecast_factor"].mean()) if "pv_forecast_factor" in segment else 0.0
    load = float(segment["load_scale"].mean()) if "load_scale" in segment else 1.0
    command = str(segment["command_type"].iloc[0])
    return price, pv, load, command


def _reason_for_segment(segment: pd.DataFrame, lang: str = "zh") -> str:
    price, pv, load, command = _segment_stats(segment)
    reasons: list[str] = []
    if lang == "zh":
        if "high-price" in command:
            reasons.append(f"电价均值 {price:.1f} 处于高价区间，DSO 包络倾向调用上调能力或减少可控负荷。")
        elif "low-price" in command:
            reasons.append(f"电价均值 {price:.1f} 处于低价区间，DSO 包络倾向降低净注入并给储能/EV 充电留空间。")
        else:
            reasons.append(f"电价均值 {price:.1f} 未触发极端价位，策略按灵活性区间进行平衡跟踪。")
        if pv >= 0.60:
            reasons.append(f"PV 预测因子均值 {pv:.2f} 较高，需要优先消纳本地光伏并观察反送。")
        if load >= 1.10:
            reasons.append(f"负荷倍率均值 {load:.2f} 偏高，VPP 出力对馈线压降和支路负载更敏感。")
        return " ".join(reasons)

    if "high-price" in command:
        reasons.append(
            f"The average price is {price:.1f}, inside the high-price band, so the DSO envelope favors upward flexibility or reduced controllable demand."
        )
    elif "low-price" in command:
        reasons.append(
            f"The average price is {price:.1f}, inside the low-price band, so the DSO envelope favors lower net injection and leaves room for ESS/EV charging."
        )
    else:
        reasons.append(
            f"The average price is {price:.1f}; no extreme price trigger is active, so the policy tracks the requested point inside the flexibility band."
        )
    if pv >= 0.60:
        reasons.append(
            f"The average PV forecast factor is {pv:.2f}, so local PV absorption and reverse-power behavior are important."
        )
    if load >= 1.10:
        reasons.append(
            f"The average load scale is {load:.2f}, making VPP dispatch more sensitive to feeder voltage drop and branch loading."
        )
    return " ".join(reasons)


def _instruction_for_segment(segment: pd.DataFrame, lang: str = "zh") -> str:
    p = float(segment["p_mw"].mean()) if "p_mw" in segment else 0.0
    p_min = float(segment["p_min_mw"].mean()) if "p_min_mw" in segment else 0.0
    p_max = float(segment["p_max_mw"].mean()) if "p_max_mw" in segment else 0.0
    region = _dispatch_region(pd.Series({"p_mw": p, "p_min_mw": p_min, "p_max_mw": p_max}))
    if lang == "zh":
        if region == "upper-flex":
            action = "向上调节：提高净注入或降低净吸收"
        elif region == "lower-flex":
            action = "向下调节：降低净注入或提高可控吸收"
        else:
            action = "中位跟踪：保持可调资源在可行区间中部"
        return f"{action}；该时段 VPP 平均聚合功率为 {p:.3f} MW，聚合可行区间约为 [{p_min:.3f}, {p_max:.3f}] MW。"

    if region == "upper-flex":
        action = "Upward regulation: increase net injection or reduce net absorption"
    elif region == "lower-flex":
        action = "Downward regulation: reduce net injection or increase controllable absorption"
    else:
        action = "Mid-band tracking: keep controllable resources near the middle of their feasible range"
    return (
        f"{action}. During this segment, average VPP aggregate power is {p:.3f} MW and "
        f"the aggregate feasible range is about [{p_min:.3f}, {p_max:.3f}] MW."
    )


def _asset_response_for_segment(
    der_state: pd.DataFrame,
    steps: set[int],
    vpp_id: str,
    dt_hours: float,
    lang: str = "zh",
) -> str:
    if der_state.empty or not steps:
        return "未记录 DER 明细。" if lang == "zh" else "No DER details were recorded."
    selected = der_state[
        (der_state["vpp_id"].astype(str) == str(vpp_id)) & (der_state["step"].astype(int).isin(steps))
    ]
    if selected.empty:
        return "未记录该 VPP 的 DER 明细。" if lang == "zh" else "No DER details were recorded for this VPP."

    labels = DER_TYPE_LABELS_ZH if lang == "zh" else DER_TYPE_LABELS_EN
    parts: list[str] = []
    for der_type, group in selected.groupby("der_type"):
        der_type = str(der_type)
        label = labels.get(der_type, der_type.replace("Model", ""))
        avg_p = float(group["p_mw"].mean()) if "p_mw" in group else 0.0
        energy = float(group["p_mw"].sum()) * float(dt_hours) if "p_mw" in group else 0.0
        if der_type == "PVModel":
            avail = (
                float(group["available_p_mw"].mean())
                if "available_p_mw" in group and group["available_p_mw"].notna().any()
                else 0.0
            )
            if lang == "zh":
                parts.append(f"{label} 平均出力 {avg_p:.3f} MW，可用功率均值 {avail:.3f} MW")
            else:
                parts.append(f"{label} averages {avg_p:.3f} MW with {avail:.3f} MW average available power")
        elif der_type == "StorageModel":
            if lang == "zh":
                direction = "放电支撑" if avg_p > 0 else "充电吸收" if avg_p < 0 else "待机"
                soc_text = ""
                if "soc" in group and group["soc"].notna().any():
                    soc_text = f"，SOC {float(group['soc'].min()):.2f}-{float(group['soc'].max()):.2f}"
                parts.append(f"{label} {direction}，平均 P={avg_p:.3f} MW，净能量 {energy:.3f} MWh{soc_text}")
            else:
                direction = "discharges for support" if avg_p > 0 else "charges for absorption" if avg_p < 0 else "stays idle"
                soc_text = ""
                if "soc" in group and group["soc"].notna().any():
                    soc_text = f", SOC {float(group['soc'].min()):.2f}-{float(group['soc'].max()):.2f}"
                parts.append(f"{label} {direction}, avg P={avg_p:.3f} MW, net energy={energy:.3f} MWh{soc_text}")
        elif der_type in {"FlexibleLoadModel", "HVACModel", "EVCSModel"}:
            if lang == "zh":
                parts.append(f"{label} 等效负荷平均 P={avg_p:.3f} MW，净能量 {energy:.3f} MWh")
            else:
                parts.append(f"{label} equivalent load averages P={avg_p:.3f} MW, net energy={energy:.3f} MWh")
        elif lang == "zh":
            parts.append(f"{label} 平均 P={avg_p:.3f} MW，净能量 {energy:.3f} MWh")
        else:
            parts.append(f"{label} averages P={avg_p:.3f} MW, net energy={energy:.3f} MWh")
    return ("；".join(parts) + "。") if lang == "zh" else ("; ".join(parts) + ".")


def _empty_explanation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "vpp_id",
            "vpp_name",
            "day_index",
            "start_time",
            "end_time",
            "step_start",
            "step_end",
            "command_type",
            "command_type_zh",
            "command_type_en",
            "avg_price",
            "avg_load_scale",
            "avg_pv_forecast_factor",
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
        ]
    )


def build_vpp_dispatch_explanations(
    frames: dict[str, pd.DataFrame],
    day_index: int = 0,
    day_hours: float = 24.0,
    dt_hours: float = 0.25,
) -> pd.DataFrame:
    """Create a one-day, per-VPP bilingual explanation table from actual simulated dispatch."""

    vpp_state = _frame(frames, "vpp_state")
    if vpp_state.empty:
        return _empty_explanation_frame()

    profile = _profile_frame(frames)
    data = vpp_state.merge(profile, on="step", how="left")
    data = _day_slice(data, day_index=day_index, day_hours=day_hours)
    if data.empty:
        return _empty_explanation_frame()

    assets = _frame(frames, "asset_registry")
    der_state = _frame(frames, "der_state")
    names = (
        assets.groupby("vpp_id")["vpp_name"].first().to_dict()
        if not assets.empty and {"vpp_id", "vpp_name"}.issubset(assets.columns)
        else {}
    )

    rows: list[dict[str, Any]] = []
    data["command_type"] = data.apply(_command_type, axis=1)
    for vpp_id, group in data.sort_values(["vpp_id", "step"]).groupby("vpp_id", sort=True):
        segment_id = (group["command_type"] != group["command_type"].shift()).cumsum()
        for _, segment in group.groupby(segment_id):
            steps = set(segment["step"].astype(int).tolist())
            start = float(segment["time_hours"].min())
            end = float(segment["time_hours"].max() + dt_hours)
            p_min = float(segment["p_min_mw"].mean()) if "p_min_mw" in segment else 0.0
            p_max = float(segment["p_max_mw"].mean()) if "p_max_mw" in segment else 0.0
            command_type = str(segment["command_type"].iloc[0])
            reason_zh = _reason_for_segment(segment, "zh")
            instruction_zh = _instruction_for_segment(segment, "zh")
            asset_response_zh = _asset_response_for_segment(der_state, steps, str(vpp_id), dt_hours, "zh")
            reason_en = _reason_for_segment(segment, "en")
            instruction_en = _instruction_for_segment(segment, "en")
            asset_response_en = _asset_response_for_segment(der_state, steps, str(vpp_id), dt_hours, "en")
            rows.append(
                {
                    "vpp_id": str(vpp_id),
                    "vpp_name": str(names.get(str(vpp_id), "")),
                    "day_index": int(day_index),
                    "start_time": f"{start:05.2f} h",
                    "end_time": f"{end:05.2f} h",
                    "step_start": int(segment["step"].min()),
                    "step_end": int(segment["step"].max()),
                    "command_type": command_type,
                    "command_type_zh": _command_label(command_type, "zh"),
                    "command_type_en": _command_label(command_type, "en"),
                    "avg_price": round(float(segment["price"].mean()), 4),
                    "avg_load_scale": round(float(segment["load_scale"].mean()), 4),
                    "avg_pv_forecast_factor": round(float(segment["pv_forecast_factor"].mean()), 4),
                    "avg_p_mw": round(float(segment["p_mw"].mean()), 5),
                    "p_range_mw": f"[{p_min:.3f}, {p_max:.3f}]",
                    "reason": reason_zh,
                    "instruction": instruction_zh,
                    "asset_response": asset_response_zh,
                    "reason_zh": reason_zh,
                    "instruction_zh": instruction_zh,
                    "asset_response_zh": asset_response_zh,
                    "reason_en": reason_en,
                    "instruction_en": instruction_en,
                    "asset_response_en": asset_response_en,
                }
            )
    return pd.DataFrame(rows).sort_values(["vpp_id", "step_start"]).reset_index(drop=True)


def _row_value(row: Any, name: str, default: str = "") -> str:
    return str(getattr(row, name, default))


def dispatch_explanations_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# VPP Daily Dispatch Instructions\n\nNo VPP dispatch explanations were generated.\n"
    lines = [
        "# VPP Daily Dispatch Instructions / VPP 一日调度指令说明",
        "",
        "Generated from actual simulated VPP aggregate power, DER dispatch, price, load, and PV forecast profiles. Positive P means injection to the grid; negative P means absorption.",
        "",
    ]
    for vpp_id, group in frame.groupby("vpp_id", sort=True):
        name = str(group["vpp_name"].iloc[0]) if "vpp_name" in group else ""
        segment_count = len(group)
        avg_p = float(group["avg_p_mw"].mean()) if "avg_p_mw" in group else 0.0
        lines.extend(
            [
                "---",
                "",
                f"## VPP: {vpp_id}" + (f" - {name}" if name else ""),
                "",
                f"- Segments / 调度片段: {segment_count}",
                f"- Average segment power / 平均片段功率: {avg_p:.3f} MW",
                "",
            ]
        )
        for index, row in enumerate(group.itertuples(index=False), start=1):
            label_zh = _row_value(row, "command_type_zh", _row_value(row, "command_type"))
            label_en = _row_value(row, "command_type_en", _row_value(row, "command_type"))
            lines.extend(
                [
                    f"### Segment {index}: {_row_value(row, 'start_time')} to {_row_value(row, 'end_time')}",
                    "",
                    f"**{label_zh}**  ",
                    f"**{label_en}**",
                    "",
                    "**中文**",
                    "",
                    f"- 原因：{_row_value(row, 'reason_zh', _row_value(row, 'reason'))}",
                    f"- 指令：{_row_value(row, 'instruction_zh', _row_value(row, 'instruction'))}",
                    f"- DER 响应：{_row_value(row, 'asset_response_zh', _row_value(row, 'asset_response'))}",
                    "",
                    "**English**",
                    "",
                    f"- Reason: {_row_value(row, 'reason_en')}",
                    f"- Instruction: {_row_value(row, 'instruction_en')}",
                    f"- DER response: {_row_value(row, 'asset_response_en')}",
                    "",
                    (
                        f"`avg_price={row.avg_price}, load_scale={row.avg_load_scale}, "
                        f"pv_forecast={row.avg_pv_forecast_factor}, avg_p_mw={row.avg_p_mw}, "
                        f"feasible_p={row.p_range_mw}`"
                    ),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def export_vpp_dispatch_instruction_report(frame: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dispatch_explanations_to_markdown(frame), encoding="utf-8")
    return path
