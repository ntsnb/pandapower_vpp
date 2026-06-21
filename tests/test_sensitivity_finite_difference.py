from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vpp_dso_sim.dso.sensitivity.finite_difference import (
    SENSITIVITY_CHANNELS,
    compute_finite_difference_sensitivity_tensor,
)
from vpp_dso_sim.dso.sensitivity.selectors import build_action_units, select_critical_network_objects
from vpp_dso_sim.network.powerflow import run_powerflow
from vpp_dso_sim.optimization.feasibility_region import compute_static_feasible_region
from vpp_dso_sim.simulation.scenario import load_scenario


def test_finite_difference_sensitivity_is_not_zero_initialized() -> None:
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    assert run_powerflow(scenario.net)
    vpp = scenario.vpps[0]
    fr = compute_static_feasible_region(vpp, 0, scope="bus_vector")
    units = build_action_units(vpp, fr, t=0, granularity="vpp_bus")[:1]
    objects = select_critical_network_objects(
        scenario.net,
        voltage_limits=scenario.dso.voltage_limits,
        line_loading_limit_percent=scenario.dso.line_loading_limit_percent,
        trafo_loading_limit_percent=scenario.dso.trafo_loading_limit_percent,
        topk_low_voltage_buses=1,
        topk_high_voltage_buses=0,
        topk_lines=1,
        topk_trafos=0,
    )

    tensor = compute_finite_difference_sensitivity_tensor(scenario.net, units, objects)

    p_channel_indices = [
        SENSITIVITY_CHANNELS.index("dV_dP"),
        SENSITIVITY_CHANNELS.index("dLineLoading_dP"),
        SENSITIVITY_CHANNELS.index("dTrafoLoading_dP"),
    ]
    p_values = tensor.values[:, :, p_channel_indices]
    assert tensor.edge_valid_mask.any()
    assert np.any(np.abs(p_values[tensor.edge_valid_mask]) > 1e-12)


def test_finite_difference_records_action_unit_allocation_weights() -> None:
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    assert run_powerflow(scenario.net)
    vpp = scenario.vpps[1]
    fr = compute_static_feasible_region(vpp, 0, scope="bus_vector")
    units = build_action_units(vpp, fr, t=0, granularity="vpp_pcc")[:1]
    objects = select_critical_network_objects(
        scenario.net,
        voltage_limits=scenario.dso.voltage_limits,
        line_loading_limit_percent=scenario.dso.line_loading_limit_percent,
        trafo_loading_limit_percent=scenario.dso.trafo_loading_limit_percent,
        topk_low_voltage_buses=1,
        topk_high_voltage_buses=0,
        topk_lines=0,
        topk_trafos=0,
    )

    tensor = compute_finite_difference_sensitivity_tensor(scenario.net, units, objects)

    weights_by_unit = tensor.metadata["sensitivity_allocation_weights"]
    weights = weights_by_unit[units[0].id.action_unit_id]
    assert weights
    assert sum(weights.values()) == pytest.approx(1.0)
    assert set(weights) == {f"{ref.element_type}:{ref.element_index}" for ref in units[0].id.pp_element_refs}
