from __future__ import annotations

from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator


def run_default_timeseries(horizon_steps: int | None = None):
    scenario = load_scenario()
    simulator = Simulator(scenario)
    return simulator.run_timeseries(horizon_steps=horizon_steps)

