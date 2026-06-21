from __future__ import annotations

from pathlib import Path

from vpp_dso_sim.dso.observation.legacy_flat import encode_dso_observation_legacy
from vpp_dso_sim.dso.observation.structured_bipartite import encode_dso_observation_structured
from vpp_dso_sim.dso.sensitivity.finite_difference import compute_finite_difference_sensitivity_tensor
from vpp_dso_sim.dso.sensitivity.selectors import build_action_units, select_critical_network_objects
from vpp_dso_sim.learning.deep_rl import encode_dso_observation
from vpp_dso_sim.network.powerflow import run_powerflow
from vpp_dso_sim.optimization.feasibility_region import compute_static_feasible_region
from vpp_dso_sim.simulation.scenario import load_scenario


def test_legacy_flat_dso_observation_still_matches_existing_encoder() -> None:
    obs = {
        "time_index": 0,
        "network_state": {"min_vm_pu": 0.99, "max_vm_pu": 1.01, "max_line_loading_percent": 12.0},
        "vpp_reports": {
            "vpp_a": {"p_mw": 0.1, "p_min_mw": -0.2, "p_max_mw": 0.5},
            "vpp_b": {"p_mw": 0.0, "p_min_mw": -0.1, "p_max_mw": 0.3},
            "vpp_c": {"p_mw": -0.1, "p_min_mw": -0.3, "p_max_mw": 0.2},
        },
    }
    vpp_ids = ["vpp_a", "vpp_b", "vpp_c"]

    legacy = encode_dso_observation_legacy(obs, vpp_ids, max_vpps=3)
    current = encode_dso_observation(obs, vpp_ids, max_vpps=3)

    assert legacy.shape == (26,)
    assert legacy.tolist() == current.tolist()


def test_structured_dso_observation_shapes_and_privacy() -> None:
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    assert run_powerflow(scenario.net)
    vpp = scenario.vpps[0]
    fr = compute_static_feasible_region(vpp, 0, scope="bus_vector")
    units = build_action_units(vpp, fr, t=0, granularity="vpp_bus")[:2]
    objects = select_critical_network_objects(
        scenario.net,
        voltage_limits=scenario.dso.voltage_limits,
        line_loading_limit_percent=scenario.dso.line_loading_limit_percent,
        trafo_loading_limit_percent=scenario.dso.trafo_loading_limit_percent,
        topk_low_voltage_buses=1,
        topk_high_voltage_buses=1,
        topk_lines=1,
        topk_trafos=1,
    )
    edges = compute_finite_difference_sensitivity_tensor(scenario.net, units, objects)

    structured = encode_dso_observation_structured(
        step=0,
        dt_hours=scenario.dt_hours,
        voltage_limits=scenario.dso.voltage_limits,
        line_loading_limit_percent=scenario.dso.line_loading_limit_percent,
        trafo_loading_limit_percent=scenario.dso.trafo_loading_limit_percent,
        action_units=units,
        network_objects=objects,
        sensitivity_edges=edges,
        max_action_units=4,
        max_network_objects=6,
    )

    assert structured.action_tokens.shape[0] == 4
    assert structured.object_tokens.shape[0] == 6
    assert structured.sensitivity_edges.shape[:2] == (6, 4)
    assert structured.action_mask[: len(units)].all()
    assert not structured.action_mask[len(units) :].any()
    assert structured.metadata["contains_private_true_cost"] is False
    assert "zone_id" not in structured.metadata["field_names"]
    assert "reliability" not in structured.metadata["field_names"]
