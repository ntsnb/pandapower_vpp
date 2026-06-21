from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from vpp_dso_sim.dso.envelope.schemas import ActionUnitId, ActionUnitState
from vpp_dso_sim.dso.sensitivity.selectors import build_action_units
from vpp_dso_sim.optimization.feasibility_region import compute_static_feasible_region
from vpp_dso_sim.simulation.scenario import load_scenario


def test_action_unit_schema_has_no_zone_or_reliability_fields() -> None:
    names = {field.name for field in fields(ActionUnitId)} | {field.name for field in fields(ActionUnitState)}

    assert "zone_id" not in names
    assert "reliability" not in names


def test_build_vpp_bus_action_units_keep_physical_bus_mapping() -> None:
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    vpp = scenario.vpps[0]
    fr = compute_static_feasible_region(vpp, 0, scope="bus_vector")

    units = build_action_units(vpp, fr, t=0, granularity="vpp_bus")

    assert units
    assert {unit.id.bus_id for unit in units}.issubset(set(vpp.connection_buses()))
    assert all(unit.id.unit_type == "vpp_bus" for unit in units)
    assert all(unit.p_min_mw <= unit.p_cur_mw <= unit.p_max_mw for unit in units)
    assert all(unit.p_min_mw <= unit.p_max_mw for unit in units)
    assert all(unit.id.pp_element_refs for unit in units)
