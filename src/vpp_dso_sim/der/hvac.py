from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandapower as pp

from vpp_dso_sim.der.base import PowerBounds
from vpp_dso_sim.der.flexible_load import FlexibleLoadModel


@dataclass
class HVACModel(FlexibleLoadModel):
    rated_power_mw: float = 0.20
    indoor_temp: float = 24.0
    outdoor_temp_profile: list[float] = field(default_factory=lambda: [30.0])
    setpoint_profile: list[float] = field(default_factory=lambda: [24.0])
    temp_min: float = 22.0
    temp_max: float = 26.0
    alpha: float = 0.20
    beta: float = 8.0
    comfort_cost_coefficients: tuple[float, float] = (10.0, 0.0)

    def __post_init__(self) -> None:
        self.baseline_p_mw = self.baseline_p_mw or 0.5 * self.rated_power_mw
        self.p_min_load_mw = 0.0
        self.p_max_load_mw = self.rated_power_mw

    def get_flexible_power_bounds(self, t: int) -> tuple[float, float]:
        p_min, p_max, _, _ = self.get_bounds(t)
        return (p_min, p_max)

    def get_bounds(self, t: int) -> PowerBounds:
        if self.indoor_temp <= self.temp_min:
            return (0.0, 0.0, 0.0, 0.0)
        if self.indoor_temp >= self.temp_max:
            required = 0.4 * self.rated_power_mw
            return (-self.rated_power_mw, -required, 0.0, 0.0)
        return (-self.rated_power_mw, 0.0, 0.0, 0.0)

    def update_temperature(self, p_mw: float | None = None, t: int = 0, dt_hours: float = 0.25) -> None:
        load_p = -self.p_mw if p_mw is None else max(0.0, p_mw)
        outdoor = float(self.outdoor_temp_profile[t % len(self.outdoor_temp_profile)])
        drift = self.alpha * (outdoor - self.indoor_temp) * dt_hours
        cooling = self.beta * load_p * dt_hours
        self.indoor_temp = float(self.indoor_temp + drift - cooling)

    def comfort_penalty(self, t: int = 0) -> float:
        setpoint = float(self.setpoint_profile[t % len(self.setpoint_profile)])
        error = self.indoor_temp - setpoint
        a, b = self.comfort_cost_coefficients
        hard = max(0.0, self.temp_min - self.indoor_temp) + max(0.0, self.indoor_temp - self.temp_max)
        return a * error * error + b * abs(error) + 100.0 * hard * hard

    def step(self, t: int, action: float | dict | None, dt_hours: float) -> None:
        self.update_temperature(t=t, dt_hours=dt_hours)

    def get_state(self) -> dict:
        state = super().get_state()
        state.update({"indoor_temp": self.indoor_temp})
        return state

