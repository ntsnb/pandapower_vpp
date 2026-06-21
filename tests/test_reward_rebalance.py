from __future__ import annotations

from pathlib import Path

from vpp_dso_sim.envs.reward_design import vpp_dispatch_reward_components
from vpp_dso_sim.learning.reward_contracts import DISPATCH_PREFERRED_REGION_BONUS_WEIGHT
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator


class _RewardProbeVPP:
    def __init__(self, delivered_p_mw: float = 0.0) -> None:
        self._delivered_p_mw = float(delivered_p_mw)

    def current_power_mw(self) -> float:
        return self._delivered_p_mw

    def operating_cost(self) -> float:
        return 0.0

    def comfort_penalty(self, _t: int) -> float:
        return 0.0

    def soc_violation_penalty(self, _t: int) -> float:
        return 0.0


def test_vpp_preferred_range_bonus_is_gated_by_lambda_width_and_effectiveness() -> None:
    base_envelope = {
        "price": 0.0,
        "p_min_mw": -1.0,
        "p_max_mw": 1.0,
        "preferred_p_min_mw": -0.25,
        "preferred_p_max_mw": 0.25,
        "preferred_target_p_mw": 0.0,
        "guidance_strength_lambda": 0.8,
        "effective_response_score": 0.5,
    }
    components = vpp_dispatch_reward_components(
        vpp=_RewardProbeVPP(delivered_p_mw=0.0),
        envelope=base_envelope,
        audit={},
        dt_hours=0.25,
        t=0,
    )

    assert components["preferred_inside_range"] == 1.0
    assert components["preferred_bonus_lambda_gate"] == 0.8
    assert components["preferred_bonus_width_gate"] == 0.75
    assert components["preferred_bonus_effectiveness_gate"] == 0.5
    assert components["preferred_region_bonus"] == (
        DISPATCH_PREFERRED_REGION_BONUS_WEIGHT * 1.0 * 0.8 * 0.75 * 0.5
    )

    zero_lambda = dict(base_envelope, guidance_strength_lambda=0.0)
    zero_lambda_components = vpp_dispatch_reward_components(
        vpp=_RewardProbeVPP(delivered_p_mw=0.0),
        envelope=zero_lambda,
        audit={},
        dt_hours=0.25,
        t=0,
    )
    assert zero_lambda_components["preferred_region_bonus"] == 0.0

    full_width = dict(base_envelope, preferred_p_min_mw=-1.0, preferred_p_max_mw=1.0)
    full_width_components = vpp_dispatch_reward_components(
        vpp=_RewardProbeVPP(delivered_p_mw=0.0),
        envelope=full_width,
        audit={},
        dt_hours=0.25,
        t=0,
    )
    assert full_width_components["preferred_bonus_width_gate"] == 0.0
    assert full_width_components["preferred_region_bonus"] == 0.0

    ineffective = dict(base_envelope, effective_response_score=0.0)
    ineffective_components = vpp_dispatch_reward_components(
        vpp=_RewardProbeVPP(delivered_p_mw=0.0),
        envelope=ineffective,
        audit={},
        dt_hours=0.25,
        t=0,
    )
    assert ineffective_components["preferred_bonus_effectiveness_gate"] == 0.0
    assert ineffective_components["preferred_region_bonus"] == 0.0


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


def test_dso_envelope_guidance_reward_components_are_logged() -> None:
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    scenario.config["dso"] = {
        "envelope_policy": "sensitivity_attention_v1",
        "observation_mode": "structured_bipartite",
        "action_unit_granularity": "vpp_bus",
        "actor": {"d_model": 32, "num_heads": 4, "num_layers": 1},
    }
    scenario.config["selector"] = {
        "topk_low_voltage_buses": 1,
        "topk_high_voltage_buses": 1,
        "topk_lines": 1,
        "topk_trafos": 1,
    }
    simulator = Simulator(scenario)

    result = simulator.step()
    components = result["reward_components"]

    for key in (
        "envelope_width_penalty",
        "raw_envelope_width_penalty",
        "smoothness_penalty",
        "raw_smoothness_penalty",
        "effective_response_bonus",
        "mean_envelope_width_ratio",
        "mean_guidance_strength_lambda",
    ):
        assert key in components
    assert components["mean_envelope_width_ratio"] >= 0.0
    assert 0.0 <= components["effective_response_bonus"] <= 1.0


def test_nested_dso_reward_config_maps_to_component_weights() -> None:
    scenario = load_scenario(Path("configs") / "happo_sensitivity_attention_v1.yaml")

    assert scenario.dso.reward_component_weights["envelope_width_penalty"] == 0.1
    assert scenario.dso.reward_component_weights["smoothness_penalty"] == 0.05
    assert scenario.dso.reward_component_weights["action_projection_penalty"] == 5.0
