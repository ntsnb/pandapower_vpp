from __future__ import annotations

from vpp_dso_sim.der.ev import EVModel
from vpp_dso_sim.der.hvac import HVACModel
from vpp_dso_sim.der.microturbine import MicroTurbineModel
from vpp_dso_sim.der.pv import PVModel
from vpp_dso_sim.der.storage import StorageModel


def test_pv_does_not_exceed_forecast_power():
    pv = PVModel(id="pv", name="pv", bus=0, p_max_mw=1.0, forecast_profile=[0.35])
    _, p_max, _, _ = pv.get_bounds(0)
    assert p_max == 0.35


def test_microturbine_ramp_bounds_apply():
    mt = MicroTurbineModel(
        id="mt",
        name="mt",
        bus=0,
        p_min_mw=0.0,
        p_max_mw=1.0,
        previous_p_mw=0.2,
        ramp_up_mw_per_step=0.1,
        ramp_down_mw_per_step=0.05,
    )
    p_min, p_max = mt.get_ramp_limited_bounds(0)
    assert round(p_min, 6) == 0.15
    assert round(p_max, 6) == 0.3


def test_storage_soc_update_uses_internal_sign():
    ess = StorageModel(
        id="ess",
        name="ess",
        bus=0,
        capacity_mwh=1.0,
        soc=0.5,
        eta_charge=1.0,
        eta_discharge=1.0,
    )
    ess.update_soc(p_storage_mw=0.2, dt_hours=1.0)
    assert round(ess.soc, 6) == 0.3
    ess.update_soc(p_storage_mw=-0.1, dt_hours=1.0)
    assert round(ess.soc, 6) == 0.4


def test_ev_unmet_soc_penalty_at_departure():
    ev = EVModel(id="ev", arrival_time=0, departure_time=1, soc=0.2, target_soc=0.8)
    assert ev.unmet_soc_penalty(1) > 0.0


def test_hvac_temperature_update_runs():
    hvac = HVACModel(id="hvac", name="hvac", bus=0, rated_power_mw=0.2, indoor_temp=25.0)
    hvac.p_mw = -0.1
    hvac.update_temperature(t=0, dt_hours=0.25)
    assert isinstance(hvac.indoor_temp, float)
