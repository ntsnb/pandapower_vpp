from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandapower as pp


PowerBounds = tuple[float, float, float, float]


@dataclass
class DERBase:
    id: str
    name: str
    bus: int
    owner_vpp_id: str = ""
    pp_element_type: str = ""
    pp_element_index: int | None = None
    p_mw: float = 0.0
    q_mvar: float = 0.0
    p_min_mw: float = 0.0
    p_max_mw: float = 0.0
    q_min_mvar: float = 0.0
    q_max_mvar: float = 0.0
    cost_coefficients: tuple[float, float, float] = (0.0, 0.0, 0.0)
    controllable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def attach_to_net(self, net: pp.pandapowerNet) -> None:
        raise NotImplementedError

    def get_bounds(self, t: int) -> PowerBounds:
        return (self.p_min_mw, self.p_max_mw, self.q_min_mvar, self.q_max_mvar)

    def get_state(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "bus": self.bus,
            "owner_vpp_id": self.owner_vpp_id,
            "pp_element_type": self.pp_element_type,
            "pp_element_index": self.pp_element_index,
            "p_mw": self.p_mw,
            "q_mvar": self.q_mvar,
            "controllable": self.controllable,
        }

    def set_power(self, net: pp.pandapowerNet, p_mw: float, q_mvar: float = 0.0) -> None:
        if self.pp_element_index is None:
            raise RuntimeError(f"DER {self.id} is not attached to a pandapower net")
        t = int(self.metadata.get("current_t", 0))
        p_min, p_max, q_min, q_max = self.get_bounds(t)
        p = float(np.clip(p_mw, p_min, p_max))
        q = float(np.clip(q_mvar, q_min, q_max))
        if self.pp_element_type == "sgen":
            net.sgen.at[self.pp_element_index, "p_mw"] = p
            net.sgen.at[self.pp_element_index, "q_mvar"] = q
        elif self.pp_element_type == "load":
            net.load.at[self.pp_element_index, "p_mw"] = max(0.0, -p)
            net.load.at[self.pp_element_index, "q_mvar"] = max(0.0, -q)
        elif self.pp_element_type == "storage":
            net.storage.at[self.pp_element_index, "p_mw"] = -p
            net.storage.at[self.pp_element_index, "q_mvar"] = q
        else:
            raise ValueError(f"Unsupported pp_element_type for set_power: {self.pp_element_type}")
        self.p_mw = p
        self.q_mvar = q

    def step(self, t: int, action: float | dict[str, Any] | None, dt_hours: float) -> None:
        return None

    def validate_constraints(self) -> bool:
        p_min, p_max, q_min, q_max = self.get_bounds(0)
        return p_min - 1e-9 <= self.p_mw <= p_max + 1e-9 and q_min - 1e-9 <= self.q_mvar <= q_max + 1e-9

    def marginal_cost(self, p_mw: float) -> float:
        a, b, _ = self.cost_coefficients
        return 2.0 * a * p_mw + b

    def operating_cost(self, p_mw: float | None = None) -> float:
        p = self.p_mw if p_mw is None else p_mw
        a, b, c = self.cost_coefficients
        return a * p * p + b * abs(p) + c
