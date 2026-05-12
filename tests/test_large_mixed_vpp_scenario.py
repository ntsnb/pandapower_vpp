from __future__ import annotations

from pathlib import Path

from vpp_dso_sim.network.european_lv import build_european_lv_demo_network
from vpp_dso_sim.network.powerflow import run_powerflow
from vpp_dso_sim.optimization.feasibility_region import compute_static_feasible_region
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator
from vpp_dso_sim.visualization.dashboard_data import build_dashboard_frames


CONFIG = Path("configs") / "european_lv_mixed_vpp.yaml"


def test_european_lv_demo_network_builds_and_converges():
    net = build_european_lv_demo_network()

    assert len(net.bus) == 123
    assert len(net.line) == 121
    assert len(net.trafo) == 1
    assert run_powerflow(net)
    assert bool(net.converged)


def test_mixed_vpp_scenario_has_single_pcc_and_multi_node_vpps():
    scenario = load_scenario(CONFIG)
    modes = {vpp.physical_mode() for vpp in scenario.vpps}

    assert modes == {"single_pcc", "multi_node"}
    assert len(scenario.vpps) >= 6
    assert len(scenario.portfolio_events) == 2

    single = next(vpp for vpp in scenario.vpps if vpp.physical_mode() == "single_pcc")
    multi = next(vpp for vpp in scenario.vpps if vpp.physical_mode() == "multi_node")

    assert compute_static_feasible_region(single, 0).scope == "pcc"
    assert compute_static_feasible_region(multi, 0).scope == "bus_vector"


def test_portfolio_events_change_owner_without_moving_physical_element():
    scenario = load_scenario(CONFIG)
    target_der = next(
        der
        for vpp in scenario.vpps
        for der in vpp.der_list
        if der.id == "ess_f5_99"
    )
    original_bus = target_der.bus
    original_pp_type = target_der.pp_element_type
    original_pp_index = target_der.pp_element_index

    simulator = Simulator(scenario)
    results = simulator.run_timeseries(horizon_steps=100)
    frames = build_dashboard_frames(scenario.net, scenario.vpps, results, dt_hours=scenario.dt_hours)

    assert not frames["portfolio_change_log"].empty
    change = frames["portfolio_change_log"].iloc[0]
    assert change["der_id"] == "ess_f5_99"
    assert bool(change["physical_bus_unchanged"])
    assert target_der.owner_vpp_id == "vpp_commercial_multi"
    assert target_der.bus == original_bus
    assert target_der.pp_element_type == original_pp_type
    assert target_der.pp_element_index == original_pp_index
    assert {"single_pcc", "multi_node"}.issubset(set(frames["vpp_portfolio_history"]["physical_mode"]))


def test_first_person_frames_cover_day_ahead_intraday_and_multi_node_scope():
    scenario = load_scenario(CONFIG)
    simulator = Simulator(scenario)
    results = simulator.run_timeseries(horizon_steps=8)
    frames = build_dashboard_frames(scenario.net, scenario.vpps, results, dt_hours=scenario.dt_hours)

    timeline = frames["vpp_first_person_timeline"]
    scope = frames["vpp_first_person_scope_detail"]

    assert {"day_ahead", "intraday"}.issubset(set(timeline["phase"]))
    assert {"Saw", "Inferred", "Decided"}.isdisjoint(set(timeline.columns))
    assert {"seen_fr_bounds_json", "inferred_grid_need_label", "decision_summary", "private_cost_used"}.issubset(
        timeline.columns
    )
    assert not timeline["private_cost_used"].astype(bool).any()
    assert "cost_coefficients" not in ",".join(timeline.columns)

    multi_vpps = [vpp.id for vpp in scenario.vpps if vpp.physical_mode() == "multi_node"]
    multi_scope = scope[scope["vpp_id"].isin(multi_vpps)]
    assert not multi_scope.empty
    assert multi_scope["bus_id"].nunique() > 1
