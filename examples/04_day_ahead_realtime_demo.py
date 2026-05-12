from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator


def main() -> None:
    scenario = load_scenario(PROJECT_ROOT / "configs" / "ieee33_multi_vpp.yaml")
    simulator = Simulator(scenario)
    simulator.reset()
    day_ahead = scenario.dso.issue_day_ahead_targets(0)
    print("day_ahead_targets_mw=", {k: round(v, 4) for k, v in day_ahead.items()})
    for step in range(8):
        realtime = {
            vpp_id: target + (0.03 if step % 2 == 0 else -0.02)
            for vpp_id, target in day_ahead.items()
        }
        result = simulator.step(step, realtime)
        executed = {vpp.id: round(vpp.current_power_mw(), 4) for vpp in scenario.vpps}
        print(
            f"step={step} converged={result['converged']} "
            f"reward={result['reward_components']['reward']:.3f} executed={executed}"
        )


if __name__ == "__main__":
    main()

