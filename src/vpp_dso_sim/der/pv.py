from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np
import pandapower as pp

from vpp_dso_sim.der.base import DERBase, PowerBounds


@dataclass
class PVModel(DERBase):
    forecast_profile: list[float] = field(default_factory=lambda: [1.0])
    curtailment_rate: float = 1.0
    apparent_power_mva: float = 0.0
    power_factor_min: float = 0.95

    def attach_to_net(self, net: pp.pandapowerNet) -> None:
        self.pp_element_type = "sgen"
        self.p_mw = min(self.available_power(0), self.p_max_mw)
        self.pp_element_index = pp.create_sgen(
            net,
            bus=self.bus,
            p_mw=self.p_mw,
            q_mvar=self.q_mvar,
            name=self.name,
            type="PV",
            controllable=self.controllable,
        )

    def available_power(self, t: int) -> float:
        if not self.forecast_profile:
            return self.p_max_mw
        factor = float(self.forecast_profile[t % len(self.forecast_profile)])
        return max(0.0, min(self.p_max_mw, self.p_max_mw * factor))

    def get_bounds(self, t: int) -> PowerBounds:
        available = self.available_power(t)
        p_min = max(0.0, (1.0 - self.curtailment_rate) * available)
        p_max = max(p_min, available)
        q_min, q_max = self.compute_q_bounds(self.p_mw)
        return (p_min, p_max, q_min, q_max)

    def set_curtailment_ratio(self, net: pp.pandapowerNet, ratio: float, t: int) -> None:
        clipped_ratio = float(np.clip(ratio, 0.0, self.curtailment_rate))
        p = self.available_power(t) * (1.0 - clipped_ratio)
        self.set_power(net, p, self.q_mvar)

    def compute_q_bounds(self, p_mw: float) -> tuple[float, float]:
        apparent = self.apparent_power_mva or max(self.p_max_mw, 1e-6)
        q_cap = sqrt(max(0.0, apparent * apparent - p_mw * p_mw))
        q_min = max(self.q_min_mvar, -q_cap)
        q_max = min(self.q_max_mvar if self.q_max_mvar else q_cap, q_cap)
        return (q_min, q_max)

    def set_power(self, net: pp.pandapowerNet, p_mw: float, q_mvar: float = 0.0) -> None:
        t = int(self.metadata.get("current_t", 0))
        p_min, p_max, _, _ = self.get_bounds(t)
        p = float(np.clip(p_mw, p_min, p_max))
        q_min, q_max = self.compute_q_bounds(p)
        q = float(np.clip(q_mvar, q_min, q_max))
        if self.pp_element_index is None:
            raise RuntimeError(f"PV {self.id} is not attached")
        net.sgen.at[self.pp_element_index, "p_mw"] = p
        net.sgen.at[self.pp_element_index, "q_mvar"] = q
        self.p_mw = p
        self.q_mvar = q
