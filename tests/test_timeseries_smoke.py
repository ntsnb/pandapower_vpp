from __future__ import annotations

from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator
from vpp_dso_sim.optimization.feasibility_region import compute_static_feasible_region, current_power_by_fr_scope


def test_timeseries_smoke_runs_and_outputs_fields():
    scenario = load_scenario()
    simulator = Simulator(scenario)
    results = simulator.run_timeseries(horizon_steps=4)
    assert len(results["bus_voltage"]) == 4
    assert len(results["line_loading"]) == 4
    assert not results["vpp_power"].empty
    assert not results["der_dispatch"].empty
    reward_columns = set(results["reward_components"].columns)
    assert {
        "reward",
        "raw_objective_reward",
        "dso_reward",
        "dso_total_cost",
        "feasibility_bonus",
        "tracking_bonus",
        "action_projection_gap_mw",
        "local_bounds_projection_gap_mw",
        "action_projection_penalty",
        "ac_certified_projection_gap_mw",
        "ac_certificate_failed_count",
        "post_ac_violation_count",
        "post_ac_voltage_violation_count",
        "post_ac_line_overload_count",
        "post_ac_violation_magnitude",
        "scaled_total_cost",
        "scaled_reward",
        "dso_reward_cost_basis",
    }.issubset(reward_columns)
    projection_columns = set(results["projection_trace"].columns)
    assert {
        "projection_gap_scope",
        "post_ac_violation_count",
        "post_ac_security_ok",
        "post_ac_powerflow_converged",
    }.issubset(projection_columns)
    stage_names = set(results["projection_trace"]["stage_name"])
    assert {"bus_vector_doe", "ac_pf_certificate", "pandapower_write", "powerflow_result"}.issubset(stage_names)
    certificate_rows = results["projection_trace"][
        results["projection_trace"]["stage_name"] == "ac_pf_certificate"
    ]
    assert not certificate_rows.empty
    assert "ac_certificate_status" in certificate_rows.columns


def test_simulator_reset_restores_dynamic_state():
    scenario = load_scenario()
    simulator = Simulator(scenario)
    first = simulator.run_timeseries(horizon_steps=2)["storage_soc"].copy()
    second = simulator.run_timeseries(horizon_steps=2)["storage_soc"].copy()
    assert first["soc"].round(8).to_list() == second["soc"].round(8).to_list()


def test_multi_node_vpp_accepts_explicit_scope_targets():
    scenario = load_scenario()
    simulator = Simulator(scenario)
    vpp = scenario.vpps[0]
    fr = compute_static_feasible_region(vpp, t=0)
    current = current_power_by_fr_scope(vpp, fr)
    scope_targets = {}
    for index, (key, value) in enumerate(current.items()):
        bounds = fr.bounds[key]
        delta = 0.01 if index == 0 else -0.01
        scope_targets[key] = max(bounds.p_min_mw, min(bounds.p_max_mw, value + delta))

    simulator.step(actions={vpp.id: {"selected_p_by_scope": scope_targets, "command_source": "scope_test"}})

    delivered_by_scope = {key: 0.0 for key in fr.bounds}
    for der in vpp.der_list:
        key = f"bus_{int(der.bus)}"
        delivered_by_scope[key] += float(der.p_mw)
    for key, target in scope_targets.items():
        assert abs(delivered_by_scope[key] - target) < 1e-6
    results = simulator.collect_results()
    assert "bus_vector_doe" in set(results["projection_trace"]["stage_name"])
    assert "vector_scope_target" in set(results["vpp_rl_disaggregation"]["action_mode"])
