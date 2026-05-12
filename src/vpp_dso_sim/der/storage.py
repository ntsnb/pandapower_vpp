from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandapower as pp

from vpp_dso_sim.der.base import DERBase, PowerBounds


def internal_to_pp_storage_p(p_internal_mw: float) -> float:
    return -float(p_internal_mw)


def pp_to_internal_storage_p(p_pp_mw: float) -> float:
    return -float(p_pp_mw)


@dataclass
class StorageModel(DERBase):
    capacity_mwh: float = 1.0
    soc: float = 0.5
    soc_min: float = 0.1
    soc_max: float = 0.9
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    p_charge_max_mw: float = 0.25
    p_discharge_max_mw: float = 0.25

    def attach_to_net(self, net: pp.pandapowerNet) -> None:
        self.pp_element_type = "storage"
        self.pp_element_index = pp.create_storage(
            net,
            bus=self.bus,
            p_mw=internal_to_pp_storage_p(self.p_mw),
            max_e_mwh=self.capacity_mwh,
            soc_percent=self.soc * 100.0,
            q_mvar=self.q_mvar,
            name=self.name,
            controllable=True,
        )

    def update_soc(self, p_storage_mw: float | None = None, dt_hours: float = 1.0) -> None:
        p = self.p_mw if p_storage_mw is None else float(p_storage_mw)
        if self.capacity_mwh <= 0:
            raise ValueError("Storage capacity_mwh must be positive")
        if p >= 0.0:
            delta_soc = -p * dt_hours / (self.eta_discharge * self.capacity_mwh)
        else:
            delta_soc = self.eta_charge * (-p) * dt_hours / self.capacity_mwh
        self.soc = float(np.clip(self.soc + delta_soc, 0.0, 1.0))

    def get_bounds(self, t: int) -> PowerBounds:
        p_min = -self.p_charge_max_mw
        p_max = self.p_discharge_max_mw
        if self.soc >= self.soc_max - 1e-9:
            p_min = 0.0
        if self.soc <= self.soc_min + 1e-9:
            p_max = 0.0
        return (p_min, p_max, self.q_min_mvar, self.q_max_mvar)

    def set_storage_power(self, net: pp.pandapowerNet, p_storage_mw: float, q_mvar: float = 0.0) -> None:
        self.set_power(net, p_storage_mw, q_mvar)

    def set_power(self, net: pp.pandapowerNet, p_mw: float, q_mvar: float = 0.0) -> None:
        t = int(self.metadata.get("current_t", 0))
        p_min, p_max, q_min, q_max = self.get_bounds(t)
        p = float(np.clip(p_mw, p_min, p_max))
        q = float(np.clip(q_mvar, q_min, q_max))
        if self.pp_element_index is None:
            raise RuntimeError(f"Storage {self.id} is not attached")
        net.storage.at[self.pp_element_index, "p_mw"] = internal_to_pp_storage_p(p)
        net.storage.at[self.pp_element_index, "q_mvar"] = q
        net.storage.at[self.pp_element_index, "soc_percent"] = self.soc * 100.0
        self.p_mw = p
        self.q_mvar = q

    def validate_soc(self) -> bool:
        return self.soc_min - 1e-9 <= self.soc <= self.soc_max + 1e-9

    def get_state(self) -> dict[str, float | str | int | None]:
        state = super().get_state()
        state.update({"soc": self.soc, "capacity_mwh": self.capacity_mwh})
        return state
