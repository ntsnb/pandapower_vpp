from __future__ import annotations

from vpp_dso_sim.simulation.scenario import load_scenario


def test_vpp_report_keeps_multi_node_connection_buses():
    scenario = load_scenario()
    vpp = scenario.vpps[0]
    report = vpp.report_to_dso(t=0)

    assert report["physical_mode"] == "multi_node"
    assert set(report["connection_buses"]) == {der.bus for der in vpp.der_list}
    assert vpp.pcc_bus in report["connection_buses"]
    assert any(bus != vpp.pcc_bus for bus in report["connection_buses"])


def test_der_dispatch_writes_to_true_physical_bus_not_fake_pcc():
    scenario = load_scenario()
    net = scenario.net
    vpp = scenario.vpps[0]
    remote_der = next(der for der in vpp.der_list if der.bus != vpp.pcc_bus)
    remote_der.metadata["current_t"] = 0

    target_p = remote_der.get_bounds(0)[1]
    remote_der.set_power(net, target_p, 0.0)

    assert remote_der.pp_element_index is not None
    table = getattr(net, remote_der.pp_element_type)
    assert int(table.at[remote_der.pp_element_index, "bus"]) == remote_der.bus
    assert int(table.at[remote_der.pp_element_index, "bus"]) != vpp.pcc_bus

