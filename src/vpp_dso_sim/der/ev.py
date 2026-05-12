from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EVModel:
    id: str
    arrival_time: int
    departure_time: int
    capacity_mwh: float = 0.06
    soc: float = 0.35
    target_soc: float = 0.80
    soc_min: float = 0.10
    soc_max: float = 1.00
    p_charge_max_mw: float = 0.007
    eta_charge: float = 0.92

    def is_connected(self, t: int) -> bool:
        tt = t % 96
        if self.arrival_time <= self.departure_time:
            return self.arrival_time <= tt < self.departure_time
        return tt >= self.arrival_time or tt < self.departure_time

    def get_charging_bounds(self, t: int) -> tuple[float, float]:
        if not self.is_connected(t) or self.soc >= self.soc_max - 1e-9:
            return (0.0, 0.0)
        return (0.0, self.p_charge_max_mw)

    def update_soc(self, p_charge_mw: float, dt_hours: float) -> None:
        delta = self.eta_charge * max(0.0, p_charge_mw) * dt_hours / self.capacity_mwh
        self.soc = float(np.clip(self.soc + delta, self.soc_min, self.soc_max))

    def unmet_soc_penalty(self, t: int) -> float:
        if (t % 96) != self.departure_time:
            return 0.0
        gap = max(0.0, self.target_soc - self.soc)
        return 500.0 * gap * gap

