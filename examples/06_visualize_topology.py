from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator
from vpp_dso_sim.visualization.dashboard_data import build_dashboard_frames, export_dashboard_frames
from vpp_dso_sim.visualization.topology_plots import plot_topology_report


def main() -> None:
    scenario = load_scenario(PROJECT_ROOT / "configs" / "ieee33_multi_vpp.yaml")
    simulator = Simulator(scenario)
    results = simulator.run_timeseries()

    output_dir = PROJECT_ROOT / "outputs"
    dashboard_dir = output_dir / "dashboard_data"
    figure_dir = output_dir / "figures"

    simulator.export_results(output_dir)
    frames = build_dashboard_frames(
        scenario.net,
        scenario.vpps,
        results,
        dt_hours=scenario.dt_hours,
    )
    data_paths = export_dashboard_frames(frames, dashboard_dir)
    figure_paths = plot_topology_report(frames, figure_dir)

    print(f"dashboard_tables={len(data_paths)}")
    print(f"topology_figures={len(figure_paths)}")
    print(f"dashboard_dir={dashboard_dir}")
    print(f"figure_dir={figure_dir}")


if __name__ == "__main__":
    main()

