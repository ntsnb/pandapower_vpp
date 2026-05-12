from __future__ import annotations

from typing import Any


def encode_node_need(network_state: dict[str, float]) -> dict[str, float | str]:
    min_v = float(network_state.get("min_vm_pu", 1.0))
    max_v = float(network_state.get("max_vm_pu", 1.0))
    line = float(network_state.get("max_line_loading_percent", 0.0))
    trafo = float(network_state.get("max_trafo_loading_percent", 0.0))
    voltage_low = max(0.0, 0.95 - min_v) / 0.05
    voltage_high = max(0.0, max_v - 1.05) / 0.05
    congestion = max(0.0, max(line, trafo) - 100.0) / 20.0
    if voltage_low > max(voltage_high, congestion):
        label = "voltage_low_support"
    elif voltage_high > max(voltage_low, congestion):
        label = "voltage_high_mitigation"
    elif congestion > 0.0:
        label = "congestion_relief"
    else:
        label = "normal"
    return {
        "voltage_low_need": float(voltage_low),
        "voltage_high_need": float(voltage_high),
        "congestion_need": float(congestion),
        "dominant_need": label,
    }


def encode_vpp_capability(vpp, t: int) -> dict[str, Any]:
    p_min, p_max, q_min, q_max = vpp.aggregate_flexibility(t)
    return {
        "vpp_id": vpp.id,
        "export_headroom_mw": float(max(0.0, p_max - vpp.current_power_mw())),
        "import_headroom_mw": float(max(0.0, vpp.current_power_mw() - p_min)),
        "q_absorption_headroom_mvar": float(max(0.0, vpp.current_reactive_power_mvar() - q_min)),
        "q_injection_headroom_mvar": float(max(0.0, q_max - vpp.current_reactive_power_mvar())),
        "resource_count": int(len(vpp.der_list)),
        "physical_mode": vpp.physical_mode(),
    }


def encode_vpp_grid_need_belief(history: list[dict[str, Any]]) -> dict[str, float]:
    if not history:
        return {"service_call_rate": 0.0, "avg_awarded_quantity": 0.0, "avg_price": 0.0}
    calls = len(history)
    awarded = [float(row.get("awarded_quantity", 0.0)) for row in history]
    prices = [float(row.get("settlement_price", 0.0)) for row in history]
    return {
        "service_call_rate": float(calls),
        "avg_awarded_quantity": float(sum(awarded) / calls),
        "avg_price": float(sum(prices) / calls),
    }

