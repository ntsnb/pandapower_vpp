from __future__ import annotations

from dataclasses import dataclass
from math import acos, tan

import numpy as np
import pandapower as pp

from vpp_dso_sim.der.base import DERBase, PowerBounds


@dataclass
class FlexibleLoadModel(DERBase):
    baseline_p_mw: float = 0.10
    p_min_load_mw: float = 0.05
    p_max_load_mw: float = 0.15
    power_factor: float = 0.98
    response_energy_mwh: float = 0.0

    def attach_to_net(self, net: pp.pandapowerNet) -> None:
        self.pp_element_type = "load"
        load_p = float(np.clip(self.baseline_p_mw, self.p_min_load_mw, self.p_max_load_mw))
        load_q = self._q_for_load(load_p)
        self.p_mw = -load_p
        self.q_mvar = -load_q
        self.pp_element_index = pp.create_load(
            net,
            bus=self.bus,
            p_mw=load_p,
            q_mvar=load_q,
            name=self.name,
            controllable=True,
        )

    def _q_for_load(self, p_load_mw: float) -> float:
        pf = float(np.clip(self.power_factor, 0.5, 1.0))
        if pf >= 0.999:
            return 0.0
        return p_load_mw * tan(acos(pf))

    def get_bounds(self, t: int) -> PowerBounds:
        return (-self.p_max_load_mw, -self.p_min_load_mw, -self.q_max_mvar, -self.q_min_mvar)

    def set_power(self, net: pp.pandapowerNet, p_mw: float, q_mvar: float = 0.0) -> None:
        t = int(self.metadata.get("current_t", 0))
        p_min, p_max, _, _ = self.get_bounds(t)
        p_internal = float(np.clip(p_mw, p_min, p_max))
        load_p = max(0.0, -p_internal)
        load_q = self._q_for_load(load_p)
        if self.pp_element_index is None:
            raise RuntimeError(f"Flexible load {self.id} is not attached")
        net.load.at[self.pp_element_index, "p_mw"] = load_p
        net.load.at[self.pp_element_index, "q_mvar"] = load_q
        self.p_mw = -load_p
        self.q_mvar = -load_q

    def step(self, t: int, action: float | dict | None, dt_hours: float) -> None:
        actual_load = -self.p_mw
        self.response_energy_mwh += (actual_load - self.baseline_p_mw) * dt_hours
