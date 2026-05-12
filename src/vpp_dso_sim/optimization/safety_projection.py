from __future__ import annotations

from vpp_dso_sim.optimization.feasibility_region import project_scalar_target_to_feasible_region


def clip_to_bounds(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def project_vpp_target(vpp, target_p_mw: float, t: int, feasible_region=None) -> float:
    if feasible_region is not None:
        projected_p, _ = project_scalar_target_to_feasible_region(feasible_region, target_p_mw)
        return projected_p
    p_min, p_max, _, _ = vpp.aggregate_flexibility(t)
    return clip_to_bounds(target_p_mw, p_min, p_max)
