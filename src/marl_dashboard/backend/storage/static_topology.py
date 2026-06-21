from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from marl_dashboard.backend.storage.metadata_store import MetadataStore
from vpp_dso_sim.entities.schemas import DERSpec
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.visualization.dashboard_data import build_dashboard_frames


DER_TYPE_DESCRIPTIONS = {
    "PVModel": "光伏 / PV",
    "StorageModel": "储能 / Storage",
    "MicroTurbineModel": "微型燃机 / Microturbine",
    "FlexibleLoadModel": "柔性负荷 / Flexible load",
    "HVACModel": "空调聚合 / HVAC aggregator",
    "EVCSModel": "充电桩聚合 / EV charging station",
}

PRIVACY_MODE_DESCRIPTIONS = {
    "full_information": "完整信息 / Full information",
    "representative_data": "代表性聚合信息 / Representative aggregate data",
    "privacy_preserving_proxy": "隐私保护代理 / Privacy-preserving proxy",
}

PHYSICAL_MODE_DESCRIPTIONS = {
    "single_pcc": "单 PCC / Single-PCC",
    "multi_node": "多节点 / Multi-node",
}

SIGN_CONVENTIONS = {
    "load": "pandapower load p_mw > 0 means consumption; dashboard internal dispatch p_mw < 0 means import/load.",
    "sgen": "pandapower sgen p_mw > 0 means generation; dashboard internal dispatch p_mw > 0 means export.",
    "storage": "pandapower storage p_mw > 0 means charging; dashboard internal dispatch p_mw > 0 means export.",
    "internal_dispatch": "Dashboard VPP dispatch uses p_mw > 0 as net export to the grid and p_mw < 0 as net import from the grid.",
}

TOPOLOGY_ASSET_FIELDS = (
    "der_id",
    "name",
    "vpp_id",
    "vpp_name",
    "bus_id",
    "der_type",
    "controllable",
    "pp_element_type",
    "pp_element_index",
    "p_min_mw",
    "p_max_mw",
    "q_min_mvar",
    "q_max_mvar",
)


def _json_clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_clean(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_clean(value.item())
        except Exception:
            return str(value)
    if not isinstance(value, (list, tuple, dict, set)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    clean = frame.where(pd.notna(frame), None)
    return [_json_clean(row) for row in clean.to_dict(orient="records")]


def _records_with_fields(frame: pd.DataFrame, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    public_columns = [field for field in fields if field in frame.columns]
    return _records(frame.loc[:, public_columns])


def _candidate_score(path: Path) -> tuple[int, str]:
    parent = path.parent.name
    if parent.startswith("train_"):
        rank = 0
    elif "train" in parent:
        rank = 1
    elif "eval" in parent:
        rank = 2
    else:
        rank = 3
    return rank, path.as_posix()


class StaticTopologyService:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.metadata_store = MetadataStore(self.data_dir)

    def topology(self, run_id: str) -> dict[str, Any]:
        bundle = self._scenario_bundle(str(run_id))
        frames = bundle["frames"]
        scenario = bundle["scenario"]
        return {
            "run_id": str(run_id),
            "source_config_path": str(bundle["source_config_path"]),
            "network": {
                "name": str(getattr(scenario.net, "name", "") or scenario.config.get("name", "")),
                "bus_count": int(len(scenario.net.bus)),
                "line_count": int(len(scenario.net.line)),
                "trafo_count": int(len(scenario.net.trafo)),
                "vpp_count": int(len(scenario.vpps)),
                "dt_hours": float(scenario.dt_hours),
                "horizon_steps": int(scenario.horizon_steps),
            },
            "pandapower_tables": _pandapower_table_counts(scenario.net),
            "sign_conventions": SIGN_CONVENTIONS,
            "vpp_bus_map": _vpp_bus_map(scenario.vpps),
            "nodes": _records(frames.get("network_nodes", pd.DataFrame())),
            "edges": _records(frames.get("network_edges", pd.DataFrame())),
            "assets": _records_with_fields(frames.get("asset_registry", pd.DataFrame()), TOPOLOGY_ASSET_FIELDS),
            "vpp_portfolios": _records(frames.get("vpp_portfolio", pd.DataFrame())),
        }

    def vpp_config(self, run_id: str) -> dict[str, Any]:
        bundle = self._scenario_bundle(str(run_id))
        scenario = bundle["scenario"]
        vpps = [_vpp_detail(vpp) for vpp in scenario.vpps]
        asset_count = sum(int(vpp["der_count"]) for vpp in vpps)
        return {
            "run_id": str(run_id),
            "source_config_path": str(bundle["source_config_path"]),
            "summary": {
                "vpp_count": len(vpps),
                "asset_count": asset_count,
                "horizon_steps": int(scenario.horizon_steps),
                "dt_hours": float(scenario.dt_hours),
            },
            "vpps": vpps,
        }

    @lru_cache(maxsize=32)
    def _scenario_bundle(self, run_id: str) -> dict[str, Any]:
        source_config_path = self._source_config_path(run_id)
        scenario = load_scenario(source_config_path)
        frames = build_dashboard_frames(scenario.net, scenario.vpps, results={}, dt_hours=scenario.dt_hours)
        return {
            "source_config_path": source_config_path,
            "scenario": scenario,
            "frames": frames,
        }

    def _source_config_path(self, run_id: str) -> Path:
        metadata = self.metadata_store.metadata(run_id)
        config = self.metadata_store.config(run_id)
        candidates: list[Path] = []
        for payload in (config, metadata, metadata.get("metadata", {}) if isinstance(metadata.get("metadata"), dict) else {}):
            for key in ("scenario_config_path", "source_config_path", "config_path"):
                value = payload.get(key) if isinstance(payload, dict) else None
                if value:
                    candidates.append(Path(str(value)).expanduser())
        output_dir = None
        meta_payload = metadata.get("metadata", {}) if isinstance(metadata.get("metadata"), dict) else {}
        if meta_payload.get("output_dir"):
            output_dir = Path(str(meta_payload["output_dir"])).expanduser()
        elif config.get("output_dir"):
            output_dir = Path(str(config["output_dir"])).expanduser()
        if output_dir is not None:
            candidates.extend(sorted(output_dir.glob("profiles/**/scenario_config.yaml"), key=_candidate_score))
            candidates.append(output_dir / "scenario_config.yaml")
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(f"No scenario_config.yaml found for dashboard run {run_id}")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt_number(value: Any, digits: int = 3) -> str:
    number = _optional_float(value)
    if number is None:
        return "-"
    text = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _element_ref(der: Any) -> str:
    element_type = str(getattr(der, "pp_element_type", "") or "")
    element_index = _optional_int(getattr(der, "pp_element_index", None))
    if not element_type or element_index is None:
        return "unmapped"
    return f"{element_type}#{element_index}"


def _write_target(der: Any) -> str:
    element_ref = _element_ref(der)
    if element_ref == "unmapped":
        return f"未绑定 pandapower 元件 / no pandapower write target at bus {int(der.bus)}"
    return f"pandapower {element_ref} at bus {int(der.bus)}"


def _pandapower_table_counts(net: Any) -> dict[str, int]:
    table_names = (
        "bus",
        "line",
        "trafo",
        "load",
        "sgen",
        "storage",
        "ext_grid",
        "switch",
        "shunt",
    )
    return {f"{name}_count": int(len(getattr(net, name, []))) for name in table_names}


def _vpp_bus_map(vpps: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for vpp in vpps:
        assets = list(getattr(vpp, "der_list", []))
        rows.append(
            {
                "vpp_id": str(vpp.id),
                "display_name": str(vpp.name or vpp.id),
                "pcc_bus": int(vpp.pcc_bus),
                "physical_mode": str(vpp.physical_mode()),
                "asset_buses": sorted({int(der.bus) for der in assets}),
                "asset_ids": [str(der.id) for der in assets],
                "pp_elements": [_element_ref(der) for der in assets],
            }
        )
    return rows


def _asset_configuration_summary(details: dict[str, Any]) -> str:
    parts = [str(details["der_type_description"])]
    p_min = _fmt_number(details.get("p_min_mw"))
    p_max = _fmt_number(details.get("p_max_mw"))
    if p_min != "-" or p_max != "-":
        parts.append(f"P {p_min}..{p_max} MW")
    q_min = _fmt_number(details.get("q_min_mvar"))
    q_max = _fmt_number(details.get("q_max_mvar"))
    if q_min != "-" or q_max != "-":
        parts.append(f"Q {q_min}..{q_max} Mvar")
    if details.get("capacity_mwh") is not None:
        parts.append(f"capacity {_fmt_number(details.get('capacity_mwh'))} MWh")
    if details.get("soc") is not None or details.get("soc_min") is not None or details.get("soc_max") is not None:
        parts.append(
            f"SOC {_fmt_number(details.get('soc'))} ({_fmt_number(details.get('soc_min'))}..{_fmt_number(details.get('soc_max'))})"
        )
    if details.get("n_evs") is not None:
        parts.append(f"{details['n_evs']} EVs")
    if details.get("rated_power_mw") is not None:
        parts.append(f"rated {_fmt_number(details.get('rated_power_mw'))} MW")
    if details.get("baseline_p_mw") is not None:
        parts.append(f"baseline {_fmt_number(details.get('baseline_p_mw'))} MW")
    return ", ".join(parts)


def _asset_detail(der: Any) -> dict[str, Any]:
    spec = DERSpec.from_der(der, t=0, include_private_cost=False).to_dict()
    details: dict[str, Any] = {
        "der_id": str(der.id),
        "display_name": str(getattr(der, "name", der.id) or der.id),
        "der_type": der.__class__.__name__,
        "der_type_description": DER_TYPE_DESCRIPTIONS.get(der.__class__.__name__, der.__class__.__name__),
        "bus_id": int(der.bus),
        "controllable": bool(getattr(der, "controllable", True)),
        "p_min_mw": _optional_float(spec.get("p_min_mw")),
        "p_max_mw": _optional_float(spec.get("p_max_mw")),
        "q_min_mvar": _optional_float(spec.get("q_min_mvar")),
        "q_max_mvar": _optional_float(spec.get("q_max_mvar")),
        "soc": _optional_float(spec.get("soc")),
        "soc_min": _optional_float(spec.get("soc_min")),
        "soc_max": _optional_float(spec.get("soc_max")),
        "indoor_temp": _optional_float(spec.get("indoor_temp")),
        "temp_min": _optional_float(spec.get("temp_min")),
        "temp_max": _optional_float(spec.get("temp_max")),
        "pp_element_type": str(getattr(der, "pp_element_type", "") or ""),
        "pp_element_index": _optional_int(getattr(der, "pp_element_index", None)),
        "zone_id": str(getattr(der, "metadata", {}).get("zone_id", "")),
        "feeder_id": str(getattr(der, "metadata", {}).get("feeder_id", "")),
    }
    for name in (
        "capacity_mwh",
        "p_charge_max_mw",
        "p_discharge_max_mw",
        "n_evs",
        "rated_power_mw",
        "baseline_p_mw",
        "apparent_power_mva",
        "curtailment_rate",
        "ramp_up_mw_per_step",
        "ramp_down_mw_per_step",
    ):
        if hasattr(der, name):
            details[name] = _json_clean(getattr(der, name))
    details["write_target"] = _write_target(der)
    details["configuration_summary"] = _asset_configuration_summary(details)
    return details


def _vpp_detail(vpp: Any) -> dict[str, Any]:
    assets = [_asset_detail(der) for der in vpp.der_list]
    asset_counts = dict(Counter(asset["der_type"] for asset in assets))
    p_min, p_max, q_min, q_max = vpp.aggregate_flexibility(0)
    physical_mode = str(vpp.physical_mode())
    privacy_mode = str(vpp.privacy_mode)
    connection_buses = vpp.connection_buses()
    zone_ids = [str(item) for item in vpp.metadata.get("zone_ids", [])]
    description = (
        f"PCC 母线 {int(vpp.pcc_bus)}；物理模式为 {PHYSICAL_MODE_DESCRIPTIONS.get(physical_mode, physical_mode)}；"
        f"包含 {len(assets)} 个 DER assets，连接母线 {', '.join(str(bus) for bus in connection_buses)}。"
    )
    dispatch_capability = {
        "active_power_range_mw": [float(p_min), float(p_max)],
        "reactive_power_range_mvar": [float(q_min), float(q_max)],
        "max_import_mw": float(max(0.0, -p_min)),
        "max_export_mw": float(max(0.0, p_max)),
        "sign_convention": SIGN_CONVENTIONS["internal_dispatch"],
    }
    configuration_notes = [
        f"接入母线 / connection buses: {', '.join(str(bus) for bus in connection_buses)}；PCC bus: {int(vpp.pcc_bus)}。",
        f"物理模式 / physical mode: {PHYSICAL_MODE_DESCRIPTIONS.get(physical_mode, physical_mode)}；隐私模式 / privacy mode: {PRIVACY_MODE_DESCRIPTIONS.get(privacy_mode, privacy_mode)}。",
        (
            "调度能力 / dispatch capability: "
            f"P {float(p_min):.6g}..{float(p_max):.6g} MW, Q {float(q_min):.6g}..{float(q_max):.6g} Mvar。"
        ),
        f"pandapower 写入 / write target: {', '.join(str(asset['write_target']) for asset in assets) or '-'}。",
    ]
    return {
        "vpp_id": str(vpp.id),
        "display_name": str(vpp.name or vpp.id),
        "pcc_bus": int(vpp.pcc_bus),
        "physical_mode": physical_mode,
        "physical_mode_description": PHYSICAL_MODE_DESCRIPTIONS.get(physical_mode, physical_mode),
        "privacy_mode": privacy_mode,
        "privacy_mode_description": PRIVACY_MODE_DESCRIPTIONS.get(privacy_mode, privacy_mode),
        "portfolio_version": str(vpp.metadata.get("portfolio_version", "v0")),
        "connection_buses": [int(bus) for bus in connection_buses],
        "zone_ids": zone_ids,
        "der_count": len(assets),
        "asset_counts": asset_counts,
        "p_min_mw": float(p_min),
        "p_max_mw": float(p_max),
        "q_min_mvar": float(q_min),
        "q_max_mvar": float(q_max),
        "max_import_mw": float(max(0.0, -p_min)),
        "max_export_mw": float(max(0.0, p_max)),
        "dispatch_capability": dispatch_capability,
        "description": description,
        "configuration_notes": configuration_notes,
        "assets": assets,
    }
