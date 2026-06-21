from __future__ import annotations

from pathlib import Path

from vpp_dso_sim.dso.envelope.rule_v0 import build_rule_v0_envelope
from vpp_dso_sim.optimization.feasibility_region import compute_static_feasible_region
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator


def test_rule_v0_adapter_matches_existing_simulator_envelope() -> None:
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    simulator = Simulator(scenario)
    vpp = scenario.vpps[0]
    bid = vpp.day_ahead_bid(0, price_hint=80.0)
    fr = compute_static_feasible_region(vpp, 0)
    grid_state = {"min_vm_pu": 0.99, "max_vm_pu": 1.01, "max_line_loading_percent": 0.0}

    expected = simulator._build_dso_operating_envelope(vpp, 0, bid, fr, 80.0, grid_state=grid_state)
    actual = build_rule_v0_envelope(simulator, vpp, 0, bid, fr, 80.0, grid_state=grid_state)

    assert actual["source_policy"] == "rule_v0"
    for key in ("p_min_mw", "p_max_mw", "preferred_p_min_mw", "preferred_p_max_mw", "preferred_target_p_mw"):
        assert actual[key] == expected[key]
