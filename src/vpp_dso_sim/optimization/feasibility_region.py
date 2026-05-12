from __future__ import annotations

from collections import defaultdict

from vpp_dso_sim.entities.schemas import FRObject, PowerBounds, VPPPortfolio


def _safe_margin_bounds(
    p_min: float,
    p_max: float,
    q_min: float,
    q_max: float,
    safety_margin_mw: float,
    safety_margin_mvar: float,
) -> PowerBounds:
    adjusted_p_min = float(p_min + safety_margin_mw)
    adjusted_p_max = float(p_max - safety_margin_mw)
    adjusted_q_min = float(q_min + safety_margin_mvar)
    adjusted_q_max = float(q_max - safety_margin_mvar)
    if adjusted_p_min > adjusted_p_max:
        mid_p = 0.5 * (p_min + p_max)
        adjusted_p_min = adjusted_p_max = float(mid_p)
    if adjusted_q_min > adjusted_q_max:
        mid_q = 0.5 * (q_min + q_max)
        adjusted_q_min = adjusted_q_max = float(mid_q)
    return PowerBounds(adjusted_p_min, adjusted_p_max, adjusted_q_min, adjusted_q_max)


def _resolve_scope(vpp, scope: str, portfolio: VPPPortfolio) -> str:
    if scope != "auto":
        return scope
    return "pcc" if portfolio.physical_mode == "single_pcc" else "bus_vector"


def compute_static_feasible_region(
    vpp,
    t: int,
    scope: str = "auto",
    safety_margin_mw: float = 0.0,
    safety_margin_mvar: float = 0.0,
    valid_until_step: int | None = None,
) -> FRObject:
    """Build a v0 FR/DOE object directly from DER local bounds.

    This is intentionally conservative and local. It does not claim network OPF
    optimality. For multi-node VPPs the default output is bus-vector bounds, so
    downstream network security checks do not mistake a commercial VPP for a
    single fake physical injection.
    """

    portfolio = VPPPortfolio.from_vpp(vpp, t)
    resolved_scope = _resolve_scope(vpp, scope, portfolio)
    grouped: dict[str, list[PowerBounds]] = defaultdict(list)

    for der in vpp.der_list:
        p_min, p_max, q_min, q_max = der.get_bounds(t)
        bounds = _safe_margin_bounds(
            p_min,
            p_max,
            q_min,
            q_max,
            safety_margin_mw=safety_margin_mw,
            safety_margin_mvar=safety_margin_mvar,
        )
        if resolved_scope == "pcc":
            key = f"pcc_{int(vpp.pcc_bus)}"
        elif resolved_scope == "bus_vector":
            key = f"bus_{int(der.bus)}"
        elif resolved_scope == "zone_vector":
            zone_id = der.metadata.get("zone_id", f"bus_{int(der.bus)}")
            key = f"zone_{zone_id}"
        elif resolved_scope == "der_vector":
            key = str(der.id)
        else:
            raise ValueError(f"Unsupported feasible-region scope: {resolved_scope}")
        grouped[key].append(bounds)

    bounds_by_key = {key: PowerBounds.sum(items) for key, items in grouped.items()}
    return FRObject(
        fr_id=f"fr_{vpp.id}_{int(t)}_{resolved_scope}",
        vpp_id=str(vpp.id),
        time_index=int(t),
        scope=resolved_scope,
        representation="box",
        bounds=bounds_by_key,
        safety_margin_mw=float(safety_margin_mw),
        safety_margin_mvar=float(safety_margin_mvar),
        valid_until_step=valid_until_step,
        portfolio_version=portfolio.portfolio_version,
        metadata={
            "physical_mode": portfolio.physical_mode,
            "pcc_bus_id": portfolio.pcc_bus_id,
            "connection_buses": portfolio.connection_buses,
            "source_note": "v0 local DER bounds; not an OPF-certified network envelope",
        },
    )


def project_scalar_target_to_feasible_region(
    feasible_region: FRObject,
    target_p_mw: float,
    target_q_mvar: float = 0.0,
) -> tuple[float, float]:
    bounds = feasible_region.aggregate_bounds()
    return bounds.clipped(target_p_mw, target_q_mvar)


def project_vector_target_to_feasible_region(
    feasible_region: FRObject,
    targets: dict[str, tuple[float, float] | float],
) -> dict[str, tuple[float, float]]:
    projected: dict[str, tuple[float, float]] = {}
    for key, bounds in feasible_region.bounds.items():
        raw = targets.get(key, (0.0, 0.0))
        if isinstance(raw, tuple):
            p_raw, q_raw = raw
        else:
            p_raw, q_raw = float(raw), 0.0
        projected[key] = bounds.clipped(float(p_raw), float(q_raw))
    return projected


def current_power_by_fr_scope(vpp, feasible_region: FRObject) -> dict[str, float]:
    """Return current active power grouped by the FR/DOE scope keys."""

    grouped = {key: 0.0 for key in feasible_region.bounds}
    for der in vpp.der_list:
        if feasible_region.scope == "pcc":
            key = f"pcc_{int(vpp.pcc_bus)}"
        elif feasible_region.scope == "bus_vector":
            key = f"bus_{int(der.bus)}"
        elif feasible_region.scope == "zone_vector":
            key = f"zone_{der.metadata.get('zone_id', f'bus_{int(der.bus)}')}"
        elif feasible_region.scope == "der_vector":
            key = str(der.id)
        else:
            continue
        grouped[key] = grouped.get(key, 0.0) + float(der.p_mw)
    return grouped


def scalar_target_to_vector_targets(vpp, feasible_region: FRObject, target_p_mw: float) -> dict[str, float]:
    """Distribute an aggregate active-power request over FR scope keys.

    This keeps the legacy scalar DSO/RL action contract usable while ensuring a
    multi-node VPP is repaired and audited at bus/zone/DER-vector granularity.
    """

    current = current_power_by_fr_scope(vpp, feasible_region)
    current_total = float(sum(current.values()))
    target_total, _ = project_scalar_target_to_feasible_region(feasible_region, float(target_p_mw), 0.0)
    delta_total = float(target_total - current_total)
    targets = dict(current)
    if abs(delta_total) <= 1e-12:
        return targets

    if delta_total > 0.0:
        headroom = {
            key: max(0.0, float(bounds.p_max_mw) - float(current.get(key, 0.0)))
            for key, bounds in feasible_region.bounds.items()
        }
    else:
        headroom = {
            key: max(0.0, float(current.get(key, 0.0)) - float(bounds.p_min_mw))
            for key, bounds in feasible_region.bounds.items()
        }
    total_headroom = float(sum(headroom.values()))
    if total_headroom <= 1e-12:
        return targets

    applied_delta = min(abs(delta_total), total_headroom)
    direction = 1.0 if delta_total > 0.0 else -1.0
    for key, capacity in headroom.items():
        if capacity <= 0.0:
            continue
        targets[key] = float(current.get(key, 0.0) + direction * applied_delta * capacity / total_headroom)
    projected = project_vector_target_to_feasible_region(feasible_region, targets)
    return {key: float(value[0]) for key, value in projected.items()}
