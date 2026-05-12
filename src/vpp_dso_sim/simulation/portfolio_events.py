from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PortfolioEvent:
    event_id: str
    effective_step: int
    der_id: str
    from_vpp_id: str
    to_vpp_id: str
    reason: str = "slow_loop_portfolio_update"

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> "PortfolioEvent":
        return cls(
            event_id=str(data.get("event_id", f"portfolio_event_{data.get('effective_step', 0)}")),
            effective_step=int(data.get("effective_step", 0)),
            der_id=str(data["der_id"]),
            from_vpp_id=str(data["from_vpp_id"]),
            to_vpp_id=str(data["to_vpp_id"]),
            reason=str(data.get("reason", "slow_loop_portfolio_update")),
        )


def normalize_portfolio_events(config: dict[str, Any]) -> list[PortfolioEvent]:
    events = config.get("portfolio_events", [])
    return sorted(
        [PortfolioEvent.from_config(item) for item in events],
        key=lambda item: (item.effective_step, item.event_id),
    )


def _next_portfolio_version(current: str, step: int, event_id: str) -> str:
    root = str(current or "v0").split("+")[0]
    return f"{root}+{event_id}@{int(step)}"


def apply_portfolio_event(vpps, event: PortfolioEvent, step: int) -> dict[str, Any]:
    """Move one DER between VPP commercial portfolios without moving its grid bus.

    The physical pandapower element stays attached to the same bus and table row.
    Only VPP membership, `owner_vpp_id`, and portfolio-version metadata change.
    """

    by_id = {vpp.id: vpp for vpp in vpps}
    if event.from_vpp_id not in by_id:
        raise ValueError(f"Unknown source VPP in portfolio event: {event.from_vpp_id}")
    if event.to_vpp_id not in by_id:
        raise ValueError(f"Unknown target VPP in portfolio event: {event.to_vpp_id}")
    source = by_id[event.from_vpp_id]
    target = by_id[event.to_vpp_id]

    der = next((item for item in source.der_list if item.id == event.der_id), None)
    if der is None:
        raise ValueError(f"DER {event.der_id} is not owned by {event.from_vpp_id} at step {step}")

    source_old = str(source.metadata.get("portfolio_version", "v0"))
    target_old = str(target.metadata.get("portfolio_version", "v0"))
    source_new = _next_portfolio_version(source_old, step, event.event_id)
    target_new = _next_portfolio_version(target_old, step, event.event_id)

    source.der_list = [item for item in source.der_list if item.id != event.der_id]
    target.der_list.append(der)
    der.owner_vpp_id = target.id
    der.metadata["portfolio_event_id"] = event.event_id
    der.metadata["portfolio_owner_changed_step"] = int(step)
    source.metadata["portfolio_version"] = source_new
    target.metadata["portfolio_version"] = target_new

    return {
        "event_id": event.event_id,
        "effective_step": int(step),
        "reason": event.reason,
        "der_id": event.der_id,
        "from_vpp_id": event.from_vpp_id,
        "to_vpp_id": event.to_vpp_id,
        "source_old_version": source_old,
        "source_new_version": source_new,
        "target_old_version": target_old,
        "target_new_version": target_new,
        "old_version": target_old,
        "new_version": target_new,
        "bus_id": int(der.bus),
        "pp_element_type": der.pp_element_type,
        "pp_element_index": der.pp_element_index,
        "zone_id": str(der.metadata.get("zone_id", f"bus_{int(der.bus)}")),
        "physical_bus_unchanged": True,
        "physical_element_unchanged": True,
    }
