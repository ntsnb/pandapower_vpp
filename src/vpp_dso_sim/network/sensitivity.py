from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import pandas as pd
import pandapower as pp

from vpp_dso_sim.network.powerflow import run_powerflow


ControlledElement = tuple[str, int]


def _apply_internal_p_delta_to_net(net: pp.pandapowerNet, der, delta_p_mw: float) -> None:
    if der.pp_element_index is None:
        return
    if der.pp_element_type == "sgen":
        net.sgen.at[der.pp_element_index, "p_mw"] = (
            float(net.sgen.at[der.pp_element_index, "p_mw"]) + float(delta_p_mw)
        )
    elif der.pp_element_type == "load":
        net.load.at[der.pp_element_index, "p_mw"] = max(
            0.0,
            float(net.load.at[der.pp_element_index, "p_mw"]) - float(delta_p_mw),
        )
    elif der.pp_element_type == "storage":
        net.storage.at[der.pp_element_index, "p_mw"] = (
            float(net.storage.at[der.pp_element_index, "p_mw"]) - float(delta_p_mw)
        )


def _distributed_vpp_delta(vpp, t: int, requested_delta_p_mw: float) -> list[tuple[object, float]]:
    direction = 1.0 if requested_delta_p_mw >= 0.0 else -1.0
    headrooms: list[tuple[object, float]] = []
    for der in vpp.der_list:
        if der.pp_element_index is None or not getattr(der, "controllable", True):
            continue
        p_min, p_max, _, _ = der.get_bounds(t)
        headroom = float(p_max - der.p_mw) if direction > 0.0 else float(der.p_mw - p_min)
        if headroom > 1e-9:
            headrooms.append((der, headroom))
    total_headroom = sum(headroom for _, headroom in headrooms)
    if total_headroom <= 1e-9:
        return []
    applied_total = min(abs(float(requested_delta_p_mw)), total_headroom)
    return [
        (der, direction * applied_total * headroom / total_headroom)
        for der, headroom in headrooms
    ]


def compute_vpp_active_power_sensitivity(
    net: pp.pandapowerNet,
    vpp,
    t: int = 0,
    delta_p_mw: float = 0.02,
) -> dict[str, object]:
    """Finite-difference AC sensitivity of grid metrics to aggregate VPP active power.

    The perturbation follows the simulator's internal sign convention:
    positive aggregate P means more VPP injection or less load consumption.
    It is distributed across DERs with available headroom in the requested
    direction and written directly to a deep-copied pandapower net.
    """

    base = deepcopy(net)
    if not run_powerflow(base):
        return {"status": "base_powerflow_failed"}
    if not hasattr(base, "res_bus") or base.res_bus.empty:
        return {"status": "missing_bus_results"}

    min_bus = int(base.res_bus["vm_pu"].idxmin())
    max_bus = int(base.res_bus["vm_pu"].idxmax())
    connection_buses = [int(bus) for bus in getattr(vpp, "connection_buses", lambda: [vpp.pcc_bus])()]
    connection_buses = [bus for bus in connection_buses if bus in base.res_bus.index]
    pcc_bus = int(vpp.pcc_bus) if int(vpp.pcc_bus) in base.res_bus.index else None
    min_connection_bus = (
        int(base.res_bus.loc[connection_buses, "vm_pu"].idxmin()) if connection_buses else min_bus
    )
    max_connection_bus = (
        int(base.res_bus.loc[connection_buses, "vm_pu"].idxmax()) if connection_buses else max_bus
    )
    critical_line = None
    if hasattr(base, "res_line") and len(base.res_line):
        critical_line = int(base.res_line["loading_percent"].idxmax())

    results: dict[str, object] = {
        "status": "ok",
        "min_bus": min_bus,
        "max_bus": max_bus,
        "pcc_bus": pcc_bus,
        "min_connection_bus": min_connection_bus,
        "max_connection_bus": max_connection_bus,
        "critical_line": critical_line,
        "base_min_vm_pu": float(base.res_bus.at[min_bus, "vm_pu"]),
        "base_max_vm_pu": float(base.res_bus.at[max_bus, "vm_pu"]),
        "base_pcc_vm_pu": float(base.res_bus.at[pcc_bus, "vm_pu"]) if pcc_bus is not None else None,
        "base_min_connection_vm_pu": float(base.res_bus.at[min_connection_bus, "vm_pu"]),
        "base_max_connection_vm_pu": float(base.res_bus.at[max_connection_bus, "vm_pu"]),
        "base_max_line_loading_percent": (
            float(base.res_line.at[critical_line, "loading_percent"])
            if critical_line is not None
            else 0.0
        ),
    }

    for label, requested_delta in (
        ("increase", abs(float(delta_p_mw))),
        ("decrease", -abs(float(delta_p_mw))),
    ):
        distributed = _distributed_vpp_delta(vpp, t, requested_delta)
        applied_delta = float(sum(delta for _, delta in distributed))
        if abs(applied_delta) <= 1e-9:
            results[f"{label}_status"] = "no_directional_headroom"
            continue
        perturbed = deepcopy(net)
        for der, der_delta in distributed:
            _apply_internal_p_delta_to_net(perturbed, der, der_delta)
        if not run_powerflow(perturbed):
            results[f"{label}_status"] = "perturbed_powerflow_failed"
            continue
        results[f"{label}_status"] = "ok"
        results[f"{label}_applied_delta_p_mw"] = applied_delta
        results[f"{label}_min_bus_vm_pu_per_mw"] = float(
            (perturbed.res_bus.at[min_bus, "vm_pu"] - base.res_bus.at[min_bus, "vm_pu"])
            / applied_delta
        )
        results[f"{label}_max_bus_vm_pu_per_mw"] = float(
            (perturbed.res_bus.at[max_bus, "vm_pu"] - base.res_bus.at[max_bus, "vm_pu"])
            / applied_delta
        )
        if pcc_bus is not None:
            results[f"{label}_pcc_bus_vm_pu_per_mw"] = float(
                (perturbed.res_bus.at[pcc_bus, "vm_pu"] - base.res_bus.at[pcc_bus, "vm_pu"])
                / applied_delta
            )
        results[f"{label}_min_connection_bus_vm_pu_per_mw"] = float(
            (
                perturbed.res_bus.at[min_connection_bus, "vm_pu"]
                - base.res_bus.at[min_connection_bus, "vm_pu"]
            )
            / applied_delta
        )
        results[f"{label}_max_connection_bus_vm_pu_per_mw"] = float(
            (
                perturbed.res_bus.at[max_connection_bus, "vm_pu"]
                - base.res_bus.at[max_connection_bus, "vm_pu"]
            )
            / applied_delta
        )
        if critical_line is not None:
            results[f"{label}_critical_line_loading_percent_per_mw"] = float(
                (
                    perturbed.res_line.at[critical_line, "loading_percent"]
                    - base.res_line.at[critical_line, "loading_percent"]
                )
                / applied_delta
            )

    return results


def compute_voltage_sensitivity(
    net: pp.pandapowerNet,
    controlled_elements: Iterable[ControlledElement],
    delta_p: float = 0.01,
    delta_q: float = 0.005,
) -> dict[str, pd.DataFrame]:
    """Estimate voltage and line-loading sensitivities by finite differences.

    This uses direct pandapower table perturbations and is intended as a runnable
    representative-data seed, not a replacement for analytical distribution-load-flow
    sensitivities.
    """

    work_net = deepcopy(net)
    run_powerflow(work_net)
    base_vm = work_net.res_bus["vm_pu"].copy()
    base_line = work_net.res_line["loading_percent"].copy()
    voltage_rows: list[dict[str, float | str | int]] = []
    line_rows: list[dict[str, float | str | int]] = []

    for element_type, element_index in controlled_elements:
        for variable, delta in (("p_mw", delta_p), ("q_mvar", delta_q)):
            perturbed = deepcopy(net)
            table = getattr(perturbed, element_type)
            if variable not in table.columns:
                continue
            original = float(table.at[element_index, variable])
            table.at[element_index, variable] = original + delta
            run_powerflow(perturbed)
            for bus, vm in perturbed.res_bus["vm_pu"].items():
                voltage_rows.append(
                    {
                        "element_type": element_type,
                        "element_index": element_index,
                        "variable": variable,
                        "bus": int(bus),
                        "sensitivity": float((vm - base_vm.loc[bus]) / delta),
                    }
                )
            for line, loading in perturbed.res_line["loading_percent"].items():
                line_rows.append(
                    {
                        "element_type": element_type,
                        "element_index": element_index,
                        "variable": variable,
                        "line": int(line),
                        "sensitivity": float((loading - base_line.loc[line]) / delta),
                    }
                )

    return {
        "voltage": pd.DataFrame(voltage_rows),
        "line_loading": pd.DataFrame(line_rows),
    }


def fit_linear_sensitivity_from_samples(*args, **kwargs) -> dict[str, object]:
    # TODO(v0.2): fit linear models from randomized DER perturbation samples so
    # representative-data mode can avoid sharing feeder topology with VPPs.
    return {"status": "not_implemented", "reason": "reserved for sample-based fitting"}


def create_representative_data_for_vpp(dso, vpp, t: int) -> dict[str, object]:
    net = dso.net
    pcc_bus = vpp.pcc_bus
    voltage = None
    if hasattr(net, "res_bus") and len(net.res_bus):
        voltage = float(net.res_bus.at[pcc_bus, "vm_pu"])
    return {
        "time_step": t,
        "vpp_id": vpp.id,
        "pcc_bus": pcc_bus,
        "pcc_voltage_vm_pu": voltage,
        "voltage_limits": dso.voltage_limits,
        "line_loading_limit_percent": dso.line_loading_limit_percent,
        "trafo_loading_limit_percent": dso.trafo_loading_limit_percent,
        "sensitivity_status": "finite_difference_available_via_compute_voltage_sensitivity",
    }
