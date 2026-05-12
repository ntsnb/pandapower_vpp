from __future__ import annotations

from vpp_dso_sim.network.ieee33 import build_ieee33_network
from vpp_dso_sim.network.lv_taiqu import build_lv_taiqu_demo_network
from vpp_dso_sim.network.powerflow import run_powerflow


def test_ieee33_build_and_runpp_converges():
    net = build_ieee33_network()
    assert len(net.bus) == 33
    assert len(net.line) == 32
    assert run_powerflow(net)
    assert bool(net.converged)


def test_lv_taiqu_build_and_runpp_converges():
    net = build_lv_taiqu_demo_network()
    assert len(net.trafo) == 1
    assert len(net.load) >= 1
    assert run_powerflow(net)
    assert bool(net.converged)

