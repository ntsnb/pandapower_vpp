from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator
from vpp_dso_sim.visualization.dashboard_data import build_dashboard_frames, export_dashboard_frames
from vpp_dso_sim.visualization.first_person_report import build_first_person_reports
from vpp_dso_sim.visualization.interactive_report import build_interactive_report_html
from vpp_dso_sim.visualization.report_refresh import load_deep_rl_frames


def main() -> None:
    scenario = load_scenario(PROJECT_ROOT / "configs" / "european_lv_mixed_vpp.yaml")
    simulator = Simulator(scenario)
    results = simulator.run_timeseries()

    output_dir = PROJECT_ROOT / "outputs"
    deep_rl_frames = load_deep_rl_frames(output_dir)
    frames = build_dashboard_frames(
        scenario.net,
        scenario.vpps,
        results,
        dt_hours=scenario.dt_hours,
        deep_summary=deep_rl_frames.get("deep_rl_training_summary"),
        deep_rl_frames=deep_rl_frames,
    )
    export_dashboard_frames(frames, output_dir / "dashboard_data")
    report_path = build_interactive_report_html(frames, output_dir / "interactive_report.html")
    first_person_paths = build_first_person_reports(frames, output_dir / "vpp_first_person")

    print(f"interactive_report={report_path}")
    print(f"vpp_first_person_report={first_person_paths['index']}")


if __name__ == "__main__":
    main()
