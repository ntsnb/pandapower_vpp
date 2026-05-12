from __future__ import annotations

import pandapower as pp

from vpp_dso_sim.der.flexible_load import FlexibleLoadModel
from vpp_dso_sim.der.pv import PVModel
from vpp_dso_sim.der.storage import (
    StorageModel,
    internal_to_pp_storage_p,
    pp_to_internal_storage_p,
)


def _simple_net():
    net = pp.create_empty_network()
    bus = pp.create_bus(net, vn_kv=10.0)
    pp.create_ext_grid(net, bus=bus)
    return net, bus


def test_pv_sgen_positive_p_is_generation():
    net, bus = _simple_net()
    pv = PVModel(
        id="pv",
        name="pv",
        bus=bus,
        p_max_mw=1.0,
        q_min_mvar=-0.5,
        q_max_mvar=0.5,
        forecast_profile=[1.0],
        apparent_power_mva=1.1,
    )
    pv.attach_to_net(net)
    pv.set_power(net, 0.8, 0.0)
    assert net.sgen.at[pv.pp_element_index, "p_mw"] == 0.8
    assert pv.p_mw == 0.8


def test_load_positive_p_is_consumption_and_internal_is_negative():
    net, bus = _simple_net()
    load = FlexibleLoadModel(
        id="load",
        name="load",
        bus=bus,
        baseline_p_mw=0.2,
        p_min_load_mw=0.1,
        p_max_load_mw=0.3,
    )
    load.attach_to_net(net)
    load.set_power(net, -0.25, 0.0)
    assert net.load.at[load.pp_element_index, "p_mw"] == 0.25
    assert load.p_mw == -0.25


def test_storage_internal_and_pandapower_sign_conversion():
    assert internal_to_pp_storage_p(0.2) == -0.2
    assert internal_to_pp_storage_p(-0.2) == 0.2
    assert pp_to_internal_storage_p(-0.2) == 0.2
    net, bus = _simple_net()
    storage = StorageModel(id="ess", name="ess", bus=bus, p_discharge_max_mw=0.5)
    storage.attach_to_net(net)
    storage.set_storage_power(net, 0.2, 0.0)
    assert net.storage.at[storage.pp_element_index, "p_mw"] == -0.2

