from __future__ import annotations

from vpp_dso_sim.optimization.feasibility_region import (
    compute_static_feasible_region,
    current_power_by_fr_scope,
    project_scalar_target_to_feasible_region,
    project_vector_target_to_feasible_region,
    scalar_target_to_vector_targets,
)
from vpp_dso_sim.optimization.safety_projection import project_vpp_target
from vpp_dso_sim.simulation.scenario import load_scenario


def test_multi_node_vpp_static_fr_uses_bus_vector_scope():
    scenario = load_scenario()
    vpp = scenario.vpps[0]
    fr = compute_static_feasible_region(vpp, t=0)

    assert vpp.physical_mode() == "multi_node"
    assert fr.scope == "bus_vector"
    assert len(fr.bounds) > 1
    assert f"bus_{vpp.pcc_bus}" in fr.bounds
    assert any(key != f"bus_{vpp.pcc_bus}" for key in fr.bounds)


def test_feasible_region_scalar_projection_matches_aggregate_bounds():
    scenario = load_scenario()
    vpp = scenario.vpps[0]
    fr = compute_static_feasible_region(vpp, t=0)
    aggregate = fr.aggregate_bounds()

    p_high, q_high = project_scalar_target_to_feasible_region(fr, 999.0, 999.0)
    p_low, q_low = project_scalar_target_to_feasible_region(fr, -999.0, -999.0)

    assert p_high == aggregate.p_max_mw
    assert q_high == aggregate.q_max_mvar
    assert p_low == aggregate.p_min_mw
    assert q_low == aggregate.q_min_mvar
    assert project_vpp_target(vpp, 999.0, t=0, feasible_region=fr) == aggregate.p_max_mw


def test_feasible_region_vector_projection_preserves_bus_keys():
    scenario = load_scenario()
    fr = compute_static_feasible_region(scenario.vpps[0], t=0)
    targets = {key: (999.0, -999.0) for key in fr.bounds}
    projected = project_vector_target_to_feasible_region(fr, targets)

    assert set(projected) == set(fr.bounds)
    for key, (p_mw, q_mvar) in projected.items():
        bounds = fr.bounds[key]
        assert p_mw == bounds.p_max_mw
        assert q_mvar == bounds.q_min_mvar


def test_scalar_target_to_vector_targets_preserves_bus_locality():
    scenario = load_scenario()
    vpp = scenario.vpps[0]
    fr = compute_static_feasible_region(vpp, t=0)
    current = current_power_by_fr_scope(vpp, fr)
    aggregate = fr.aggregate_bounds()

    vector_targets = scalar_target_to_vector_targets(vpp, fr, target_p_mw=aggregate.p_max_mw)

    assert set(vector_targets) == set(fr.bounds)
    assert sum(vector_targets.values()) <= aggregate.p_max_mw + 1e-9
    assert any(abs(vector_targets[key] - current[key]) > 1e-9 for key in vector_targets)
    for key, target in vector_targets.items():
        assert fr.bounds[key].p_min_mw <= target <= fr.bounds[key].p_max_mw
