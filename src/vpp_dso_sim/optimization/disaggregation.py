from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vpp_dso_sim.optimization.safety_projection import project_vpp_target

if TYPE_CHECKING:
    from vpp_dso_sim.entities.vpp import VPPAggregator


def disaggregate_target_by_rule(vpp: "VPPAggregator", target_p_mw: float, t: int) -> dict[str, float]:
    """Allocate aggregate target by local DER bounds and marginal-cost ordering."""

    target = project_vpp_target(vpp, target_p_mw, t)
    rows = []
    for der in vpp.der_list:
        der.metadata["current_t"] = t
        p_min, p_max, _, _ = der.get_bounds(t)
        rows.append(
            {
                "der": der,
                "p_min": p_min,
                "p_max": p_max,
                "capacity": max(0.0, p_max - p_min),
                "cost": der.marginal_cost(p_min),
            }
        )

    dispatch = {row["der"].id: row["p_min"] for row in rows}
    lower_sum = float(sum(row["p_min"] for row in rows))
    remaining = max(0.0, target - lower_sum)

    for row in sorted(rows, key=lambda item: item["cost"]):
        if remaining <= 1e-12:
            break
        addition = min(row["capacity"], remaining)
        dispatch[row["der"].id] += addition
        remaining -= addition

    return dispatch


def _normalized_value(raw: object, default: float = 0.0) -> float:
    try:
        value = float(raw)
        if not np.isfinite(value):
            return float(default)
        return float(np.clip(value, -1.0, 1.0))
    except (TypeError, ValueError):
        return float(default)


def _action_for_der(normalized_actions: dict[str, float] | list[float] | tuple[float, ...], der_id: str, index: int) -> float:
    if isinstance(normalized_actions, dict):
        return _normalized_value(normalized_actions.get(der_id, 0.0))
    if index < len(normalized_actions):
        return _normalized_value(normalized_actions[index])
    return 0.0


def _project_dispatch_sum_to_target(
    rows: list[dict[str, object]],
    dispatch: dict[str, float],
    target: float,
) -> dict[str, float]:
    """Adjust proposed DER setpoints so the aggregate tracks target inside bounds.

    This is a safety projection layer around learned DER actions. The policy is
    allowed to propose device setpoints, but the environment keeps the aggregate
    dispatch physically feasible by distributing residual power over available
    headroom. It is intentionally deterministic so the learning signal remains
    reproducible.
    """

    residual = float(target - sum(dispatch.values()))
    if abs(residual) <= 1e-10:
        return dispatch
    if residual > 0.0:
        capacities = [
            (str(row["der_id"]), max(0.0, float(row["p_max"]) - dispatch[str(row["der_id"])]))
            for row in rows
        ]
    else:
        capacities = [
            (str(row["der_id"]), max(0.0, dispatch[str(row["der_id"])] - float(row["p_min"])))
            for row in rows
        ]
    total_capacity = float(sum(cap for _, cap in capacities))
    if total_capacity <= 1e-12:
        return dispatch
    for der_id, capacity in capacities:
        delta = residual * capacity / total_capacity
        row = next(item for item in rows if str(item["der_id"]) == der_id)
        dispatch[der_id] = float(np.clip(dispatch[der_id] + delta, float(row["p_min"]), float(row["p_max"])))
    return dispatch


def disaggregate_target_by_learned_action(
    vpp: "VPPAggregator",
    target_p_mw: float,
    t: int,
    normalized_actions: dict[str, float] | list[float] | tuple[float, ...] | None,
) -> dict[str, float]:
    """Map VPP dispatch-agent actions to DER-level setpoints.

    The action is interpreted at DER level: each normalized value in ``[-1, 1]``
    is mapped to that DER's current physical active-power bounds. A projection
    layer then nudges the vector so the aggregate tracks the selected VPP target.

    This is the first learned-disaggregation interface. The neural policy now
    controls DER-level proposals; rule dispatch remains available as a fallback
    when no learned action is supplied.
    """

    if normalized_actions is None:
        return disaggregate_target_by_rule(vpp, target_p_mw, t)

    target = project_vpp_target(vpp, target_p_mw, t)
    rows: list[dict[str, object]] = []
    dispatch: dict[str, float] = {}
    for index, der in enumerate(vpp.der_list):
        der.metadata["current_t"] = t
        p_min, p_max, _, _ = der.get_bounds(t)
        action = _action_for_der(normalized_actions, der.id, index)
        p = 0.5 * (float(p_min) + float(p_max)) + 0.5 * action * max(0.0, float(p_max) - float(p_min))
        p = float(np.clip(p, float(p_min), float(p_max)))
        rows.append({"der_id": der.id, "p_min": float(p_min), "p_max": float(p_max), "raw_action": action})
        dispatch[der.id] = p

    return _project_dispatch_sum_to_target(rows, dispatch, target)


def disaggregate_target_by_qp(vpp: "VPPAggregator", target_p_mw: float, t: int) -> dict[str, float]:
    # TODO(v0.2): replace this fallback with an optional scipy/cvxpy QP:
    # minimize sum_i C_i(P_i), subject to bounds, ramp, SOC, and sum(P_i)=target.
    return disaggregate_target_by_rule(vpp, target_p_mw, t)
