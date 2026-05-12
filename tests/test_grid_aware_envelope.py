from __future__ import annotations

from pathlib import Path

from vpp_dso_sim.envs.observations import build_actor_observation
from vpp_dso_sim.learning.ctde_networks import VPP_DISPATCH_CONTEXT_DIM, VPP_DISPATCH_TOKEN_DIM
from vpp_dso_sim.learning.deep_rl import encode_vpp_dispatch_observation
from vpp_dso_sim.optimization.feasibility_region import compute_static_feasible_region
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator


def test_grid_pressure_overrides_low_price_absorption_request() -> None:
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    simulator = Simulator(scenario)
    vpp = scenario.vpps[0]
    step = 0
    bid = vpp.day_ahead_bid(step, price_hint=35.0)
    fr = compute_static_feasible_region(vpp, step)
    original_p_min = max(fr.aggregate_bounds().p_min_mw, float(bid["p_min_mw"]))

    envelope = simulator._build_dso_operating_envelope(
        vpp,
        step,
        bid,
        fr,
        price=35.0,
        grid_state={
            "min_vm_pu": 0.90,
            "max_vm_pu": 1.00,
            "max_line_loading_percent": 0.0,
            "max_trafo_loading_percent": 0.0,
        },
    )

    assert envelope["grid_priority_over_price"] is True
    assert envelope["grid_pressure_mode"] == "low_voltage_support"
    assert envelope["ac_aware_grid_pressure_mode"] == "ac_aware_low_voltage_support"
    assert envelope["service_request"] == "export_or_reduce_load"
    assert envelope["ac_aware_enabled"] is True
    assert envelope["ac_aware_status"] == "tightened"
    assert envelope["p_min_mw"] > original_p_min
    assert envelope["ac_aware_shrink_lower_mw"] > 0.0
    assert "ac_aware_increase_pcc_bus_vm_pu_per_mw" in envelope
    assert envelope["preferred_target_p_mw"] > 0.5 * (
        envelope["p_min_mw"] + envelope["p_max_mw"]
    )


def test_grid_pressure_overrides_high_price_export_request() -> None:
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    simulator = Simulator(scenario)
    vpp = scenario.vpps[0]
    step = 0
    bid = vpp.day_ahead_bid(step, price_hint=130.0)
    fr = compute_static_feasible_region(vpp, step)
    original_p_max = min(fr.aggregate_bounds().p_max_mw, float(bid["p_max_mw"]))

    envelope = simulator._build_dso_operating_envelope(
        vpp,
        step,
        bid,
        fr,
        price=130.0,
        grid_state={
            "min_vm_pu": 1.00,
            "max_vm_pu": 1.08,
            "max_line_loading_percent": 0.0,
            "max_trafo_loading_percent": 0.0,
        },
    )

    assert envelope["grid_priority_over_price"] is True
    assert envelope["grid_pressure_mode"] == "high_voltage_absorption"
    assert envelope["ac_aware_grid_pressure_mode"] == "ac_aware_high_voltage_absorption"
    assert envelope["service_request"] == "absorb_or_charge"
    assert envelope["ac_aware_enabled"] is True
    assert envelope["ac_aware_status"] == "tightened"
    assert envelope["p_max_mw"] < original_p_max
    assert envelope["ac_aware_shrink_upper_mw"] > 0.0
    assert "ac_aware_decrease_pcc_bus_vm_pu_per_mw" in envelope
    assert envelope["preferred_target_p_mw"] < 0.5 * (
        envelope["p_min_mw"] + envelope["p_max_mw"]
    )


def test_ac_aware_envelope_falls_back_when_sensitivity_unavailable(monkeypatch) -> None:
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    simulator = Simulator(scenario)
    vpp = scenario.vpps[0]
    step = 0
    bid = vpp.day_ahead_bid(step, price_hint=35.0)
    fr = compute_static_feasible_region(vpp, step)
    original_p_min = max(fr.aggregate_bounds().p_min_mw, float(bid["p_min_mw"]))

    monkeypatch.setattr(
        "vpp_dso_sim.simulation.simulator.compute_vpp_active_power_sensitivity",
        lambda *args, **kwargs: {"status": "base_powerflow_failed"},
    )
    envelope = simulator._build_dso_operating_envelope(
        vpp,
        step,
        bid,
        fr,
        price=35.0,
        grid_state={
            "min_vm_pu": 0.90,
            "max_vm_pu": 1.00,
            "max_line_loading_percent": 0.0,
            "max_trafo_loading_percent": 0.0,
        },
    )

    assert envelope["ac_aware_enabled"] is False
    assert envelope["ac_aware_status"] == "base_powerflow_failed"
    assert envelope["p_min_mw"] == original_p_min


def test_vpp_dispatch_observation_exposes_dynamic_der_state_tokens() -> None:
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    vpp = scenario.vpps[0]
    obs = build_actor_observation(vpp, 0, include_private_cost=True)
    encoded = encode_vpp_dispatch_observation(obs, max_der_per_vpp=len(vpp.der_list))

    assert "current_p_mw" in obs["local_assets"][0]
    assert any("soc" in asset and asset["soc"] is not None for asset in obs["local_assets"])
    assert any("indoor_temp" in asset and asset["indoor_temp"] is not None for asset in obs["local_assets"])
    assert (len(encoded) - VPP_DISPATCH_CONTEXT_DIM) % VPP_DISPATCH_TOKEN_DIM == 0
