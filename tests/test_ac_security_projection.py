from __future__ import annotations

from pathlib import Path

from vpp_dso_sim.network.constraints import check_network_constraints
from vpp_dso_sim.optimization.ac_security_projection import certify_or_repair_dispatch
from vpp_dso_sim.optimization.oracle_baseline import build_ac_validated_search_actions
from vpp_dso_sim.simulation.scenario import load_scenario


def test_ac_certificate_dispatch_matches_final_apply_semantics():
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    reference = build_ac_validated_search_actions(
        scenario,
        step=0,
        price=float(scenario.price_profile[0]),
        max_candidates=4,
    )
    dispatch_by_vpp = {
        str(vpp_id): {str(der_id): float(p_mw) for der_id, p_mw in action["der_dispatch_p_mw"].items()}
        for vpp_id, action in reference.actions.items()
    }

    certificate = certify_or_repair_dispatch(
        base_net=scenario.net,
        vpps=scenario.vpps,
        candidate_dispatch_by_vpp=dispatch_by_vpp,
        t=0,
        voltage_limits=scenario.dso.voltage_limits,
        line_loading_limit_percent=scenario.dso.line_loading_limit_percent,
        trafo_loading_limit_percent=scenario.dso.trafo_loading_limit_percent,
    )

    assert certificate.ac_safe
    for vpp in scenario.vpps:
        vpp.apply_dispatch_to_net(scenario.net, certificate.dispatch_by_vpp[str(vpp.id)], 0)
    assert scenario.dso.run_powerflow()
    report = check_network_constraints(
        scenario.net,
        voltage_limits=scenario.dso.voltage_limits,
        line_loading_limit_percent=scenario.dso.line_loading_limit_percent,
        trafo_loading_limit_percent=scenario.dso.trafo_loading_limit_percent,
    )
    assert report.ok
