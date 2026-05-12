from __future__ import annotations

from vpp_dso_sim.optimization.local_flex_market import (
    build_local_flex_needs_from_state,
    build_rule_based_vpp_bid,
    clear_local_flex_need,
    local_flex_price_from_need,
)
from vpp_dso_sim.simulation.scenario import load_scenario


def test_local_flex_need_is_generated_from_voltage_stress():
    state = {
        "min_vm_pu": 0.93,
        "max_vm_pu": 1.01,
        "max_line_loading_percent": 40.0,
        "max_trafo_loading_percent": 0.0,
    }

    needs = build_local_flex_needs_from_state(state, t=7)

    assert len(needs) == 1
    assert needs[0].target_constraint == "voltage_low"
    assert needs[0].direction == "inject_p"
    assert needs[0].severity > 0.0


def test_local_flex_price_is_not_energy_price_alias():
    need = build_local_flex_needs_from_state({"max_vm_pu": 1.08, "min_vm_pu": 1.0}, t=2)[0]
    price = local_flex_price_from_need(need, base_price=20.0, severity_adder=80.0)

    assert price.service_type == "voltage_high"
    assert price.direction == "absorb_p"
    assert price.price > 20.0
    assert price.source_method == "stress_driven_v0"


def test_rule_based_bid_and_clearing_produce_award_without_private_cost():
    scenario = load_scenario()
    vpp = scenario.vpps[0]
    need = build_local_flex_needs_from_state({"max_vm_pu": 1.08, "min_vm_pu": 1.0}, t=0)[0]
    price = local_flex_price_from_need(need)
    bid = build_rule_based_vpp_bid(vpp, need, t=0, local_flex_price=price)

    awards = clear_local_flex_need(need, [bid])

    assert bid.vpp_id == vpp.id
    assert bid.quantity_mw_or_mvar >= 0.0
    assert "cost_coefficients" not in bid.to_dict()
    if bid.quantity_mw_or_mvar > 0.0:
        assert len(awards) == 1
        assert awards[0].dispatch_instruction["direction"] == need.direction

