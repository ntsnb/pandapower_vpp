from __future__ import annotations

from dataclasses import dataclass, field

import pandapower as pp

from vpp_dso_sim.der.base import DERBase


@dataclass
class DRAggregator(DERBase):
    devices: list[DERBase] = field(default_factory=list)

    def aggregate_bounds(self, t: int) -> tuple[float, float]:
        if not self.devices:
            return (0.0, 0.0)
        lower = 0.0
        upper = 0.0
        for device in self.devices:
            p_min, p_max, _, _ = device.get_bounds(t)
            lower += p_min
            upper += p_max
        return (lower, upper)

    def disaggregate_target_power(self, target_p_mw: float, t: int) -> dict[str, float]:
        p_min, p_max = self.aggregate_bounds(t)
        target = max(p_min, min(p_max, target_p_mw))
        dispatch: dict[str, float] = {}
        remaining = target
        for device in self.devices:
            d_min, d_max, _, _ = device.get_bounds(t)
            value = max(d_min, min(d_max, remaining))
            dispatch[device.id] = value
            remaining -= value
        return dispatch

    def update_internal_devices(self, t: int, dt_hours: float) -> None:
        for device in self.devices:
            device.step(t, None, dt_hours)

    def attach_to_net(self, net: pp.pandapowerNet) -> None:
        for device in self.devices:
            device.attach_to_net(net)

    def set_aggregate_load(self, net: pp.pandapowerNet, p_mw: float, q_mvar: float = 0.0) -> None:
        dispatch = self.disaggregate_target_power(p_mw, 0)
        for device in self.devices:
            device.set_power(net, dispatch.get(device.id, device.p_mw), q_mvar)

