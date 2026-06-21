from __future__ import annotations

from typing import Any

from vpp_dso_sim.der.evcs import EVCSModel
from vpp_dso_sim.der.flexible_load import FlexibleLoadModel
from vpp_dso_sim.der.hvac import HVACModel
from vpp_dso_sim.der.microturbine import MicroTurbineModel
from vpp_dso_sim.der.pv import PVModel
from vpp_dso_sim.der.storage import StorageModel


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _der_type(der: Any) -> str:
    if isinstance(der, PVModel):
        return "pv"
    if isinstance(der, MicroTurbineModel):
        return "microturbine"
    if isinstance(der, StorageModel):
        return "storage"
    if isinstance(der, EVCSModel):
        return "evcs"
    if isinstance(der, HVACModel):
        return "hvac"
    if isinstance(der, FlexibleLoadModel):
        return "flexible_load"
    return der.__class__.__name__.lower()


def _metadata_price(der: Any, key: str, fallback: float) -> float:
    metadata = getattr(der, "metadata", {}) or {}
    if key in metadata:
        return _safe_float(metadata.get(key), fallback)
    multiplier_key = f"{key}_multiplier"
    if multiplier_key in metadata:
        return float(fallback) * _safe_float(metadata.get(multiplier_key), 1.0)
    return float(fallback)


def _before_value(before_state: dict[str, dict[str, Any]], der_id: str, key: str, default: Any = None) -> Any:
    return dict(before_state.get(str(der_id), {})).get(key, default)


def build_settlement_audit(
    *,
    vpps,
    t: int,
    dt_hours: float,
    market_price: float,
    before_state: dict[str, dict[str, Any]] | None = None,
    settlement_power_balance_tolerance_mw: float = 1.0e-6,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Build per-DER and per-VPP settlement ledgers from executed DER powers.

    The audit intentionally uses the post-projection DER `p_mw` values. It does
    not infer revenues from aggregate VPP net power, because identical imports
    can mean EV charging, storage charging, HVAC load, or flexible load.
    """

    before_state = before_state or {}
    der_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, float]] = {}
    market_export_price = float(market_price)
    market_import_price = float(market_price)
    tolerance = max(0.0, float(settlement_power_balance_tolerance_mw))

    for vpp in vpps:
        summary = _empty_vpp_summary(vpp_id=str(vpp.id), step=int(t), dt_hours=float(dt_hours))
        reconstructed = 0.0
        for der in getattr(vpp, "der_list", []):
            row = _der_settlement_row(
                der=der,
                vpp_id=str(vpp.id),
                t=int(t),
                dt_hours=float(dt_hours),
                market_export_price=market_export_price,
                market_import_price=market_import_price,
                before_state=before_state,
            )
            reconstructed += _safe_float(row.get("p_mw_internal"))
            der_rows.append(row)
            _accumulate_vpp_summary(summary, row)

        delivered = float(vpp.current_power_mw())
        balance_error = reconstructed - delivered
        summary.update(
            {
                "vpp_delivered_p_mw": delivered,
                "audit_reconstructed_p_mw": float(reconstructed),
                "power_balance_error_mw": float(balance_error),
                "settlement_power_balance_ok": float(abs(balance_error) <= tolerance),
                "settlement_audit_complete": float(_settlement_complete(summary)),
                "settlement_power_balance_tolerance_mw": float(tolerance),
            }
        )
        economic_operational_surplus = float(
            summary["export_revenue_total"]
            + summary["evcs_user_revenue_total"]
            - summary["import_energy_cost_total"]
            - summary["der_operating_cost_total"]
            - summary["battery_degradation_cost_total"]
        )
        service_quality_penalty = float(summary["comfort_cost_total"] + summary["unserved_penalty_total"])
        quality_adjusted_operational_surplus = float(economic_operational_surplus - service_quality_penalty)
        summary["economic_operational_surplus"] = economic_operational_surplus
        summary["service_quality_penalty_total"] = service_quality_penalty
        summary["quality_adjusted_operational_surplus"] = quality_adjusted_operational_surplus
        summary["legacy_operational_surplus_with_service_quality"] = quality_adjusted_operational_surplus
        summary["operational_surplus"] = economic_operational_surplus
        summary["service_payment"] = 0.0
        summary["availability_payment"] = 0.0
        summary["contract_penalty"] = 0.0
        summary["dso_transfer_payment_cost"] = 0.0
        summary["private_profit"] = float(economic_operational_surplus)
        summaries[str(vpp.id)] = summary

    return der_rows, summaries


def _der_settlement_row(
    *,
    der: Any,
    vpp_id: str,
    t: int,
    dt_hours: float,
    market_export_price: float,
    market_import_price: float,
    before_state: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    der_type = _der_type(der)
    p_mw = float(getattr(der, "p_mw", 0.0))
    p_inject = max(0.0, p_mw)
    p_absorb = max(0.0, -p_mw)
    row: dict[str, Any] = {
        "step": int(t),
        "time_hours": float(t) * float(dt_hours),
        "dt_hours": float(dt_hours),
        "vpp_id": str(vpp_id),
        "der_id": str(getattr(der, "id", "")),
        "der_type": der_type,
        "bus": int(getattr(der, "bus", -1)),
        "pp_element_type": str(getattr(der, "pp_element_type", "")),
        "pp_element_index": getattr(der, "pp_element_index", None),
        "p_mw_internal": p_mw,
        "q_mvar": float(getattr(der, "q_mvar", 0.0)),
        "p_inject_mw": p_inject,
        "p_absorb_mw": p_absorb,
        "energy_inject_mwh": p_inject * float(dt_hours),
        "energy_absorb_mwh": p_absorb * float(dt_hours),
        "market_export_price": float(market_export_price),
        "market_import_price": float(market_import_price),
        "evcs_retail_price": 0.0,
        "pv_export_mwh": 0.0,
        "mt_export_mwh": 0.0,
        "storage_discharge_mwh": 0.0,
        "storage_charge_mwh": 0.0,
        "evcs_grid_charge_mwh": 0.0,
        "evcs_user_energy_mwh": 0.0,
        "hvac_load_mwh": 0.0,
        "flex_load_mwh": 0.0,
        "unclassified_export_mwh": 0.0,
        "unclassified_import_mwh": 0.0,
        "pv_export_revenue": 0.0,
        "mt_export_revenue": 0.0,
        "storage_discharge_revenue": 0.0,
        "evcs_user_revenue": 0.0,
        "evcs_wholesale_cost": 0.0,
        "storage_charge_cost": 0.0,
        "hvac_energy_cost": 0.0,
        "flex_energy_cost": 0.0,
        "unclassified_import_cost": 0.0,
        "der_operating_cost": float(der.operating_cost()) * float(dt_hours)
        if hasattr(der, "operating_cost")
        else 0.0,
        "battery_degradation_cost": 0.0,
        "comfort_cost": float(der.comfort_penalty(t)) if isinstance(der, HVACModel) else 0.0,
        "unserved_penalty": float(der.unmet_soc_penalty(t)) if isinstance(der, EVCSModel) else 0.0,
        "storage_soc_before": _before_value(before_state, str(getattr(der, "id", "")), "soc"),
        "storage_soc_after": getattr(der, "soc", None) if isinstance(der, StorageModel) else None,
        "evcs_avg_soc_before": _before_value(before_state, str(getattr(der, "id", "")), "average_soc"),
        "evcs_avg_soc_after": der.average_soc() if isinstance(der, EVCSModel) else None,
        "evcs_connected_count": len(der.connected_evs(t)) if isinstance(der, EVCSModel) else 0,
        "hvac_indoor_temp_before": _before_value(before_state, str(getattr(der, "id", "")), "indoor_temp"),
        "hvac_indoor_temp_after": getattr(der, "indoor_temp", None) if isinstance(der, HVACModel) else None,
    }

    if isinstance(der, PVModel):
        row["pv_export_mwh"] = row["energy_inject_mwh"]
        row["pv_export_revenue"] = row["pv_export_mwh"] * float(market_export_price)
    elif isinstance(der, MicroTurbineModel):
        row["mt_export_mwh"] = row["energy_inject_mwh"]
        row["mt_export_revenue"] = row["mt_export_mwh"] * float(market_export_price)
    elif isinstance(der, StorageModel):
        row["storage_discharge_mwh"] = row["energy_inject_mwh"]
        row["storage_charge_mwh"] = row["energy_absorb_mwh"]
        row["storage_discharge_revenue"] = row["storage_discharge_mwh"] * float(market_export_price)
        row["storage_charge_cost"] = row["storage_charge_mwh"] * float(market_import_price)
    elif isinstance(der, EVCSModel):
        metadata = getattr(der, "metadata", {}) or {}
        if "evcs_retail_price" in metadata:
            evcs_retail_price = _safe_float(metadata.get("evcs_retail_price"), 1.25 * float(market_import_price))
        elif "evcs_retail_price_multiplier" in metadata:
            evcs_retail_price = float(market_import_price) * _safe_float(
                metadata.get("evcs_retail_price_multiplier"),
                1.25,
            )
        else:
            evcs_retail_price = 1.25 * float(market_import_price)
        row["evcs_retail_price"] = float(evcs_retail_price)
        row["evcs_grid_charge_mwh"] = row["energy_absorb_mwh"]
        row["evcs_user_energy_mwh"] = row["energy_absorb_mwh"]
        row["evcs_user_revenue"] = row["evcs_user_energy_mwh"] * float(evcs_retail_price)
        row["evcs_wholesale_cost"] = row["evcs_grid_charge_mwh"] * float(market_import_price)
    elif isinstance(der, HVACModel):
        row["hvac_load_mwh"] = row["energy_absorb_mwh"]
        row["hvac_energy_cost"] = row["hvac_load_mwh"] * float(market_import_price)
    elif isinstance(der, FlexibleLoadModel):
        row["flex_load_mwh"] = row["energy_absorb_mwh"]
        row["flex_energy_cost"] = row["flex_load_mwh"] * float(market_import_price)
    else:
        row["unclassified_export_mwh"] = row["energy_inject_mwh"]
        row["unclassified_import_mwh"] = row["energy_absorb_mwh"]
        row["unclassified_import_cost"] = row["unclassified_import_mwh"] * float(market_import_price)

    return row


def _empty_vpp_summary(*, vpp_id: str, step: int, dt_hours: float) -> dict[str, float]:
    return {
        "step": float(step),
        "vpp_id": vpp_id,
        "dt_hours": float(dt_hours),
        "vpp_delivered_p_mw": 0.0,
        "audit_reconstructed_p_mw": 0.0,
        "power_balance_error_mw": 0.0,
        "settlement_power_balance_ok": 0.0,
        "settlement_audit_complete": 0.0,
        "export_revenue_total": 0.0,
        "pv_export_revenue_total": 0.0,
        "mt_export_revenue_total": 0.0,
        "storage_discharge_revenue_total": 0.0,
        "evcs_user_revenue_total": 0.0,
        "import_energy_cost_total": 0.0,
        "evcs_wholesale_cost_total": 0.0,
        "storage_charge_cost_total": 0.0,
        "hvac_energy_cost_total": 0.0,
        "flex_energy_cost_total": 0.0,
        "unclassified_import_cost_total": 0.0,
        "der_operating_cost_total": 0.0,
        "battery_degradation_cost_total": 0.0,
        "comfort_cost_total": 0.0,
        "unserved_penalty_total": 0.0,
        "economic_operational_surplus": 0.0,
        "service_quality_penalty_total": 0.0,
        "quality_adjusted_operational_surplus": 0.0,
        "legacy_operational_surplus_with_service_quality": 0.0,
        "operational_surplus": 0.0,
        "service_payment": 0.0,
        "availability_payment": 0.0,
        "contract_penalty": 0.0,
        "private_profit": 0.0,
    }


def _accumulate_vpp_summary(summary: dict[str, float], row: dict[str, Any]) -> None:
    summary["pv_export_revenue_total"] += _safe_float(row.get("pv_export_revenue"))
    summary["mt_export_revenue_total"] += _safe_float(row.get("mt_export_revenue"))
    summary["storage_discharge_revenue_total"] += _safe_float(row.get("storage_discharge_revenue"))
    summary["evcs_user_revenue_total"] += _safe_float(row.get("evcs_user_revenue"))
    summary["evcs_wholesale_cost_total"] += _safe_float(row.get("evcs_wholesale_cost"))
    summary["storage_charge_cost_total"] += _safe_float(row.get("storage_charge_cost"))
    summary["hvac_energy_cost_total"] += _safe_float(row.get("hvac_energy_cost"))
    summary["flex_energy_cost_total"] += _safe_float(row.get("flex_energy_cost"))
    summary["unclassified_import_cost_total"] += _safe_float(row.get("unclassified_import_cost"))
    summary["der_operating_cost_total"] += _safe_float(row.get("der_operating_cost"))
    summary["battery_degradation_cost_total"] += _safe_float(row.get("battery_degradation_cost"))
    summary["comfort_cost_total"] += _safe_float(row.get("comfort_cost"))
    summary["unserved_penalty_total"] += _safe_float(row.get("unserved_penalty"))
    summary["export_revenue_total"] = (
        summary["pv_export_revenue_total"]
        + summary["mt_export_revenue_total"]
        + summary["storage_discharge_revenue_total"]
    )
    summary["import_energy_cost_total"] = (
        summary["evcs_wholesale_cost_total"]
        + summary["storage_charge_cost_total"]
        + summary["hvac_energy_cost_total"]
        + summary["flex_energy_cost_total"]
        + summary["unclassified_import_cost_total"]
    )


def _settlement_complete(summary: dict[str, float]) -> bool:
    required = (
        "export_revenue_total",
        "evcs_user_revenue_total",
        "import_energy_cost_total",
        "der_operating_cost_total",
    )
    return all(key in summary for key in required)
