from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandapower as pp

from vpp_dso_sim.der.base import DERBase, PowerBounds
from vpp_dso_sim.der.ev import EVModel


@dataclass
class EVCSModel(DERBase):
    evs: list[EVModel] = field(default_factory=list)
    n_evs: int = 10
    p_charge_max_mw: float = 0.15
    power_factor: float = 0.99

    def __post_init__(self) -> None:
        if not self.evs:
            per_ev = self.p_charge_max_mw / max(1, self.n_evs)
            for i in range(self.n_evs):
                arrival = 60 + (i % 8)
                departure = 28 + (i % 8)
                self.evs.append(
                    EVModel(
                        id=f"{self.id}_ev_{i}",
                        arrival_time=arrival,
                        departure_time=departure,
                        p_charge_max_mw=per_ev,
                        soc=0.25 + 0.02 * (i % 10),
                    )
                )

    def attach_to_net(self, net: pp.pandapowerNet) -> None:
        self.pp_element_type = "load"
        self.p_mw = 0.0
        self.pp_element_index = pp.create_load(
            net,
            bus=self.bus,
            p_mw=0.0,
            q_mvar=0.0,
            name=self.name,
            controllable=True,
        )

    def connected_evs(self, t: int) -> list[EVModel]:
        return [ev for ev in self.evs if ev.is_connected(t)]

    def get_bounds(self, t: int) -> PowerBounds:
        max_charge = sum(ev.get_charging_bounds(t)[1] for ev in self.evs)
        max_charge = min(max_charge, self.p_charge_max_mw)
        return (-max_charge, 0.0, 0.0, 0.0)

    def set_power(self, net: pp.pandapowerNet, p_mw: float, q_mvar: float = 0.0) -> None:
        t = int(self.metadata.get("current_t", 0))
        p_min, p_max, _, _ = self.get_bounds(t)
        p_internal = float(np.clip(p_mw, p_min, p_max))
        load_p = max(0.0, -p_internal)
        if self.pp_element_index is None:
            raise RuntimeError(f"EVCS {self.id} is not attached")
        net.load.at[self.pp_element_index, "p_mw"] = load_p
        net.load.at[self.pp_element_index, "q_mvar"] = 0.0
        self.p_mw = -load_p
        self.q_mvar = 0.0

    def update_internal_devices(self, t: int, dt_hours: float) -> None:
        connected = self.connected_evs(t)
        if not connected:
            return
        total_charge = max(0.0, -self.p_mw)
        per_ev = total_charge / len(connected)
        for ev in connected:
            ev.update_soc(min(per_ev, ev.p_charge_max_mw), dt_hours)

    def average_soc(self) -> float:
        if not self.evs:
            return 0.0
        return float(np.mean([ev.soc for ev in self.evs]))

    def unmet_soc_penalty(self, t: int) -> float:
        return float(sum(ev.unmet_soc_penalty(t) for ev in self.evs))

    def get_state(self) -> dict:
        state = super().get_state()
        state.update({"average_soc": self.average_soc(), "connected_evs": len(self.connected_evs(0))})
        return state
