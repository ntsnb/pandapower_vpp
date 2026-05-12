from __future__ import annotations

from pathlib import Path

from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator


def test_reward_component_weights_surface_raw_and_weighted_cost_terms() -> None:
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    simulator = Simulator(scenario)

    result = simulator.step()
    components = result["reward_components"]

    assert components["comfort_violation_penalty_weight"] == 0.02
    assert components["voltage_violation_penalty_weight"] == 20.0
    assert "raw_comfort_violation_penalty" in components
    assert "raw_voltage_violation_penalty" in components
    assert components["comfort_violation_penalty"] == (
        components["raw_comfort_violation_penalty"]
        * components["comfort_violation_penalty_weight"]
    )
    assert "scaled_comfort_violation_penalty" in components
    assert "scaled_total_cost" in components
    assert "scaled_reward" in components
    assert components["reward"] == components["scaled_reward"]
    assert components["dso_reward"] == components["scaled_reward"]
    assert components["scaled_comfort_violation_penalty"] <= components["comfort_violation_penalty"]
    assert components["total_cost"] >= components["operation_cost"]


def test_post_ac_security_metrics_are_part_of_scaled_reward_audit() -> None:
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    simulator = Simulator(scenario)

    result = simulator.step()
    components = result["reward_components"]

    assert "post_ac_violation_count" in components
    assert "post_ac_voltage_violation_count" in components
    assert "post_ac_line_overload_count" in components
    assert "post_ac_trafo_overload_count" in components
    assert "post_ac_powerflow_failed" in components
    assert "post_ac_violation_magnitude" in components
    assert "post_ac_security_penalty" in components
    assert "scaled_post_ac_security_penalty" in components
    assert "post_ac_violation_magnitude_penalty" in components
    assert "scaled_post_ac_violation_magnitude_penalty" in components
    assert components["constraint_violation_count"] == components["post_ac_violation_count"]
    assert components["dso_reward_cost_basis"] == components["scaled_total_cost"]
