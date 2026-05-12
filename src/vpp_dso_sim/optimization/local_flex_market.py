from __future__ import annotations

from dataclasses import dataclass

from vpp_dso_sim.entities.schemas import DispatchAward, LocalFlexNeed, VPPFlexBid


@dataclass(frozen=True)
class LocalFlexPrice:
    zone_id: str
    service_type: str
    direction: str
    price: float
    price_unit: str = "currency_per_mwh"
    severity: float = 0.0
    source_method: str = "stress_driven_v0"

    def to_dict(self) -> dict[str, float | str]:
        return {
            "zone_id": self.zone_id,
            "service_type": self.service_type,
            "direction": self.direction,
            "price": float(self.price),
            "price_unit": self.price_unit,
            "severity": float(self.severity),
            "source_method": self.source_method,
        }


def _severity(value: float, limit: float, deadband: float) -> float:
    if deadband <= 0.0:
        return 0.0
    return max(0.0, float(value - limit) / float(deadband))


def build_local_flex_needs_from_state(
    network_state: dict[str, float],
    t: int,
    voltage_limits: tuple[float, float] = (0.95, 1.05),
    line_loading_limit_percent: float = 100.0,
    trafo_loading_limit_percent: float = 100.0,
    zone_id: str = "system",
    duration_min: float = 15.0,
) -> list[LocalFlexNeed]:
    """Create v0 LocalFlexNeed records from network stress.

    This converts grid stress into service needs. It does not perform market
    clearing and it is not an LMP calculation.
    """

    needs: list[LocalFlexNeed] = []
    min_v = float(network_state.get("min_vm_pu", 1.0))
    max_v = float(network_state.get("max_vm_pu", 1.0))
    max_line = float(network_state.get("max_line_loading_percent", 0.0))
    max_trafo = float(network_state.get("max_trafo_loading_percent", 0.0))
    v_low, v_high = voltage_limits

    if min_v < v_low:
        severity = _severity(v_low, min_v, 0.05)
        needs.append(
            LocalFlexNeed(
                need_id=f"need_voltage_low_{t}_{zone_id}",
                zone_id=zone_id,
                target_constraint="voltage_low",
                direction="inject_p",
                required_effective_mw_or_mvar=0.05 + 0.20 * severity,
                start_time=int(t),
                duration_min=duration_min,
                response_time_min=duration_min,
                severity=severity,
            )
        )
    if max_v > v_high:
        severity = _severity(max_v, v_high, 0.05)
        needs.append(
            LocalFlexNeed(
                need_id=f"need_voltage_high_{t}_{zone_id}",
                zone_id=zone_id,
                target_constraint="voltage_high",
                direction="absorb_p",
                required_effective_mw_or_mvar=0.05 + 0.20 * severity,
                start_time=int(t),
                duration_min=duration_min,
                response_time_min=duration_min,
                severity=severity,
            )
        )
    if max_line > line_loading_limit_percent:
        severity = _severity(max_line, line_loading_limit_percent, 20.0)
        needs.append(
            LocalFlexNeed(
                need_id=f"need_line_overload_{t}_{zone_id}",
                zone_id=zone_id,
                target_constraint="line_overload",
                direction="reduce_flow",
                required_effective_mw_or_mvar=0.05 + 0.30 * severity,
                start_time=int(t),
                duration_min=duration_min,
                response_time_min=duration_min,
                severity=severity,
            )
        )
    if max_trafo > trafo_loading_limit_percent:
        severity = _severity(max_trafo, trafo_loading_limit_percent, 20.0)
        needs.append(
            LocalFlexNeed(
                need_id=f"need_trafo_overload_{t}_{zone_id}",
                zone_id=zone_id,
                target_constraint="transformer_overload",
                direction="reduce_flow",
                required_effective_mw_or_mvar=0.05 + 0.30 * severity,
                start_time=int(t),
                duration_min=duration_min,
                response_time_min=duration_min,
                severity=severity,
            )
        )
    return needs


def local_flex_price_from_need(
    need: LocalFlexNeed,
    base_price: float = 20.0,
    severity_adder: float = 80.0,
) -> LocalFlexPrice:
    return LocalFlexPrice(
        zone_id=need.zone_id,
        service_type=need.target_constraint,
        direction=need.direction,
        price=float(base_price + severity_adder * max(0.0, need.severity)),
        severity=float(need.severity),
    )


def build_rule_based_vpp_bid(
    vpp,
    need: LocalFlexNeed,
    t: int,
    local_flex_price: LocalFlexPrice,
    reliability: float = 1.0,
    location_effectiveness: float | None = None,
) -> VPPFlexBid:
    p_min, p_max, _, _ = vpp.aggregate_flexibility(t)
    current = vpp.current_power_mw()
    if need.direction in {"inject_p", "up"}:
        quantity = max(0.0, p_max - current)
    elif need.direction in {"absorb_p", "down"}:
        quantity = max(0.0, current - p_min)
    else:
        quantity = max(0.0, max(p_max - current, current - p_min))
    return VPPFlexBid(
        bid_id=f"bid_{need.need_id}_{vpp.id}",
        vpp_id=vpp.id,
        portfolio_version=str(vpp.metadata.get("portfolio_version", "v0")),
        zone_id=need.zone_id,
        direction=need.direction,
        quantity_mw_or_mvar=float(min(quantity, need.required_effective_mw_or_mvar)),
        duration_min=need.duration_min,
        response_time_min=need.response_time_min,
        price=local_flex_price.price,
        reliability=float(reliability),
        location_effectiveness=location_effectiveness,
    )


def clear_local_flex_need(
    need: LocalFlexNeed,
    bids: list[VPPFlexBid],
) -> list[DispatchAward]:
    """Clear one need by effective price until required quantity is met."""

    eligible = [bid for bid in bids if bid.direction == need.direction and bid.quantity_mw_or_mvar > 0.0]

    def effective_price(bid: VPPFlexBid) -> float:
        effectiveness = bid.location_effectiveness if bid.location_effectiveness is not None else 1.0
        score = max(1e-6, bid.reliability * effectiveness)
        return float(bid.price / score)

    awards: list[DispatchAward] = []
    remaining = float(need.required_effective_mw_or_mvar)
    for bid in sorted(eligible, key=effective_price):
        if remaining <= 1e-9:
            break
        awarded = min(remaining, float(bid.quantity_mw_or_mvar))
        awards.append(
            DispatchAward(
                award_id=f"award_{need.need_id}_{bid.vpp_id}_{len(awards)}",
                vpp_id=bid.vpp_id,
                need_id=need.need_id,
                awarded_quantity=awarded,
                settlement_price=float(bid.price),
                expected_effective_contribution=awarded
                * float(bid.location_effectiveness if bid.location_effectiveness is not None else 1.0),
                dispatch_instruction={
                    "direction": need.direction,
                    "quantity_mw_or_mvar": awarded,
                    "target_constraint": need.target_constraint,
                },
                start_time=need.start_time,
                end_time=int(need.start_time + max(1, round(need.duration_min / 15.0))),
            )
        )
        remaining -= awarded
    return awards

