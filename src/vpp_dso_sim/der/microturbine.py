from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandapower as pp

from vpp_dso_sim.der.base import DERBase, PowerBounds


@dataclass
class MicroTurbineModel(DERBase):
    ramp_up_mw_per_step: float = 0.10
    ramp_down_mw_per_step: float = 0.10
    previous_p_mw: float = 0.0
    min_up_down_time: int | None = None

    def attach_to_net(self, net: pp.pandapowerNet) -> None:
        self.pp_element_type = "sgen"
        self.p_mw = max(self.p_min_mw, min(self.p_mw, self.p_max_mw))
        self.previous_p_mw = self.p_mw
        self.pp_element_index = pp.create_sgen(
            net,
            bus=self.bus,
            p_mw=self.p_mw,
            q_mvar=self.q_mvar,
            name=self.name,
            type="MT",
            controllable=True,
        )

    def get_ramp_limited_bounds(self, t: int) -> tuple[float, float]:
        lower = max(self.p_min_mw, self.previous_p_mw - self.ramp_down_mw_per_step)
        upper = min(self.p_max_mw, self.previous_p_mw + self.ramp_up_mw_per_step)
        return (lower, upper)

    def get_bounds(self, t: int) -> PowerBounds:
        p_min, p_max = self.get_ramp_limited_bounds(t)
        return (p_min, p_max, self.q_min_mvar, self.q_max_mvar)

    def set_power(self, net: pp.pandapowerNet, p_mw: float, q_mvar: float = 0.0) -> None:
        t = int(self.metadata.get("current_t", 0))
        p_min, p_max, q_min, q_max = self.get_bounds(t)
        p = float(np.clip(p_mw, p_min, p_max))
        q = float(np.clip(q_mvar, q_min, q_max))
        if self.pp_element_index is None:
            raise RuntimeError(f"Microturbine {self.id} is not attached")
        net.sgen.at[self.pp_element_index, "p_mw"] = p
        net.sgen.at[self.pp_element_index, "q_mvar"] = q
        self.p_mw = p
        self.previous_p_mw = p
        self.q_mvar = q

    def validate_ramp(self) -> bool:
        delta = self.p_mw - self.previous_p_mw
        return -self.ramp_down_mw_per_step - 1e-9 <= delta <= self.ramp_up_mw_per_step + 1e-9
