from __future__ import annotations

from vpp_dso_sim.entities.schemas import DERSpec, FRObject, VPPPortfolio, schema_visibility
from vpp_dso_sim.optimization.feasibility_region import compute_static_feasible_region
from vpp_dso_sim.simulation.scenario import load_scenario


def test_der_and_portfolio_schema_round_trip():
    scenario = load_scenario()
    vpp = scenario.vpps[0]
    der = vpp.der_list[0]

    der_spec = DERSpec.from_der(der, t=0)
    restored_der = DERSpec.from_dict(der_spec.to_dict())
    assert restored_der.der_id == der.id
    assert restored_der.bus_id == der.bus

    portfolio = VPPPortfolio.from_vpp(vpp, t=0)
    restored_portfolio = VPPPortfolio.from_dict(portfolio.to_dict())
    assert restored_portfolio.vpp_id == vpp.id
    assert restored_portfolio.physical_mode == "multi_node"
    assert set(restored_portfolio.connection_buses) == {der.bus for der in vpp.der_list}


def test_feasible_region_schema_round_trip():
    scenario = load_scenario()
    fr = compute_static_feasible_region(scenario.vpps[0], t=0)
    restored = FRObject.from_dict(fr.to_dict())

    assert restored.fr_id == fr.fr_id
    assert restored.scope == "bus_vector"
    assert set(restored.bounds) == {f"bus_{der.bus}" for der in scenario.vpps[0].der_list}
    assert restored.aggregate_bounds().p_max_mw >= restored.aggregate_bounds().p_min_mw


def test_schema_visibility_marks_private_der_costs():
    rows = schema_visibility(DERSpec)
    by_field = {row["field"]: row for row in rows}
    assert by_field["cost_coefficients"]["oracle_only"] is True
    assert by_field["cost_coefficients"]["visible_to_other_vpp"] is False

