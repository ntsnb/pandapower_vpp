from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from vpp_dso_sim.dso.envelope.schemas import (
    ActionUnitId,
    ActionUnitState,
    NetworkObjectId,
    NetworkObjectState,
    PPElementRef,
)
from vpp_dso_sim.entities.schemas import PowerBounds


def _pp_ref_for_der(der) -> PPElementRef | None:
    if der.pp_element_index is None or der.pp_element_type not in {"sgen", "load", "storage"}:
        return None
    return PPElementRef(element_type=str(der.pp_element_type), element_index=int(der.pp_element_index))


def _sum_bounds(items: Iterable[tuple[float, float, float, float]]) -> PowerBounds:
    bounds = [
        PowerBounds(float(p_min), float(p_max), float(q_min), float(q_max))
        for p_min, p_max, q_min, q_max in items
    ]
    return PowerBounds.sum(bounds) if bounds else PowerBounds(0.0, 0.0, 0.0, 0.0)


def _unit_from_group(
    *,
    vpp,
    unit_type: str,
    bus_id: int,
    pcc_id: str | None,
    ders: list[object],
    fr_bound: PowerBounds | None,
    t: int,
    bid: dict[str, object],
    projection_gap_hist_mw: float,
) -> ActionUnitState:
    refs = tuple(ref for der in ders if (ref := _pp_ref_for_der(der)) is not None)
    local_bounds = _sum_bounds(der.get_bounds(t) for der in ders)
    bounds = fr_bound if fr_bound is not None else local_bounds
    q_available = any(abs(float(getattr(der, "q_min_mvar", 0.0))) > 1e-12 or abs(float(getattr(der, "q_max_mvar", 0.0))) > 1e-12 for der in ders)
    suffix = pcc_id if unit_type == "vpp_pcc" else f"bus_{int(bus_id)}" if unit_type == "vpp_bus" else str(ders[0].id)
    unit_id = ActionUnitId(
        action_unit_id=f"{vpp.id}:{unit_type}:{suffix}",
        vpp_id=str(vpp.id),
        unit_type=unit_type,  # type: ignore[arg-type]
        pcc_id=pcc_id,
        bus_id=int(bus_id),
        pp_element_refs=refs,
    )
    return ActionUnitState(
        id=unit_id,
        p_cur_mw=float(sum(getattr(der, "p_mw", 0.0) for der in ders)),
        q_cur_mvar=float(sum(getattr(der, "q_mvar", 0.0) for der in ders)),
        p_min_mw=float(bounds.p_min_mw),
        p_max_mw=float(bounds.p_max_mw),
        q_min_mvar=float(bounds.q_min_mvar),
        q_max_mvar=float(bounds.q_max_mvar),
        bid_up=float(bid["bid_price_up"]) if "bid_price_up" in bid else None,
        bid_down=float(bid["bid_price_down"]) if "bid_price_down" in bid else None,
        projection_gap_hist_mw=float(projection_gap_hist_mw),
        q_control_available=bool(q_available),
    )


def build_action_units(
    vpp,
    feasible_region,
    *,
    t: int,
    granularity: str = "vpp_bus",
    projection_gap_hist_mw: float = 0.0,
    projection_gap_hist_by_scope: dict[str, float] | None = None,
) -> list[ActionUnitState]:
    """Build ActionUnit states without moving DER physical injection buses."""

    bid = vpp.day_ahead_bid(t)
    units: list[ActionUnitState] = []
    gap_by_scope = dict(projection_gap_hist_by_scope or {})
    if granularity in {"vpp", "vpp_pcc"}:
        bounds = feasible_region.aggregate_bounds()
        pcc_key = f"pcc_{int(vpp.pcc_bus)}"
        units.append(
            _unit_from_group(
                vpp=vpp,
                unit_type="vpp_pcc",
                bus_id=int(vpp.pcc_bus),
                pcc_id=pcc_key,
                ders=list(vpp.der_list),
                fr_bound=bounds,
                t=t,
                bid=bid,
                projection_gap_hist_mw=float(gap_by_scope.get(pcc_key, projection_gap_hist_mw)),
            )
        )
        return units

    if granularity == "vpp_bus":
        grouped: dict[int, list[object]] = defaultdict(list)
        for der in vpp.der_list:
            grouped[int(der.bus)].append(der)
        for bus_id in sorted(grouped):
            key = f"bus_{int(bus_id)}"
            units.append(
                _unit_from_group(
                    vpp=vpp,
                    unit_type="vpp_bus",
                    bus_id=bus_id,
                    pcc_id=f"pcc_{int(vpp.pcc_bus)}",
                    ders=grouped[bus_id],
                    fr_bound=feasible_region.bounds.get(key),
                    t=t,
                    bid=bid,
                    projection_gap_hist_mw=float(gap_by_scope.get(key, projection_gap_hist_mw)),
                )
            )
        return units

    if granularity == "der":
        for der in vpp.der_list:
            der_key = str(der.id)
            units.append(
                _unit_from_group(
                    vpp=vpp,
                    unit_type="der",
                    bus_id=int(der.bus),
                    pcc_id=f"pcc_{int(vpp.pcc_bus)}",
                    ders=[der],
                    fr_bound=feasible_region.bounds.get(str(der.id)),
                    t=t,
                    bid=bid,
                    projection_gap_hist_mw=float(gap_by_scope.get(der_key, projection_gap_hist_mw)),
                )
            )
        return units

    raise ValueError(f"Unsupported action_unit_granularity: {granularity}")


def _add_unique(target: list[NetworkObjectState], item: NetworkObjectState, seen: set[str]) -> None:
    if item.id.object_id in seen:
        return
    seen.add(item.id.object_id)
    target.append(item)


def select_critical_network_objects(
    net,
    *,
    voltage_limits: tuple[float, float],
    line_loading_limit_percent: float,
    trafo_loading_limit_percent: float,
    topk_low_voltage_buses: int = 5,
    topk_high_voltage_buses: int = 5,
    topk_lines: int = 5,
    topk_trafos: int = 3,
) -> list[NetworkObjectState]:
    """Select current critical bus/line/trafo objects from pandapower results."""

    objects: list[NetworkObjectState] = []
    seen: set[str] = set()
    vmin, vmax = float(voltage_limits[0]), float(voltage_limits[1])
    if hasattr(net, "res_bus") and len(net.res_bus):
        low = net.res_bus["vm_pu"].sort_values(ascending=True).head(max(0, int(topk_low_voltage_buses)))
        high = net.res_bus["vm_pu"].sort_values(ascending=False).head(max(0, int(topk_high_voltage_buses)))
        for bus, value in list(low.items()) + list(high.items()):
            bus_id = int(bus)
            _add_unique(
                objects,
                NetworkObjectState(
                    id=NetworkObjectId(f"bus_{bus_id}", "bus", bus_id, (bus_id,)),
                    value=float(value),
                    lower_limit=vmin,
                    upper_limit=vmax,
                    value_kind="vm_pu",
                ),
                seen,
            )

    if hasattr(net, "res_line") and len(net.res_line):
        lines = net.res_line["loading_percent"].sort_values(ascending=False).head(max(0, int(topk_lines)))
        for line, value in lines.items():
            line_id = int(line)
            endpoints = (
                int(net.line.at[line_id, "from_bus"]) if hasattr(net, "line") and line_id in net.line.index else -1,
                int(net.line.at[line_id, "to_bus"]) if hasattr(net, "line") and line_id in net.line.index else -1,
            )
            _add_unique(
                objects,
                NetworkObjectState(
                    id=NetworkObjectId(f"line_{line_id}", "line", line_id, endpoints),
                    value=float(value),
                    lower_limit=None,
                    upper_limit=float(line_loading_limit_percent),
                    value_kind="line_loading_percent",
                ),
                seen,
            )

    if hasattr(net, "res_trafo") and len(net.res_trafo):
        trafos = net.res_trafo["loading_percent"].sort_values(ascending=False).head(max(0, int(topk_trafos)))
        for trafo, value in trafos.items():
            trafo_id = int(trafo)
            endpoints = (
                int(net.trafo.at[trafo_id, "hv_bus"]) if hasattr(net, "trafo") and trafo_id in net.trafo.index else -1,
                int(net.trafo.at[trafo_id, "lv_bus"]) if hasattr(net, "trafo") and trafo_id in net.trafo.index else -1,
            )
            _add_unique(
                objects,
                NetworkObjectState(
                    id=NetworkObjectId(f"trafo_{trafo_id}", "trafo", trafo_id, endpoints),
                    value=float(value),
                    lower_limit=None,
                    upper_limit=float(trafo_loading_limit_percent),
                    value_kind="trafo_loading_percent",
                ),
                seen,
            )

    return objects
