from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator
from vpp_dso_sim.visualization.dashboard_data import build_dashboard_frames, export_dashboard_frames
from vpp_dso_sim.visualization.dispatch_explanations import export_vpp_dispatch_instruction_report
from vpp_dso_sim.visualization.first_person_report import build_first_person_reports
from vpp_dso_sim.visualization.interactive_report import build_interactive_report_html
from vpp_dso_sim.visualization.plots import plot_timeseries_results
from vpp_dso_sim.visualization.report_refresh import load_deep_rl_frames
from vpp_dso_sim.visualization.rl_architecture_report import build_rl_architecture_report_html
from vpp_dso_sim.visualization.topology_plots import plot_topology_report


def main() -> None:
    scenario = load_scenario(PROJECT_ROOT / "configs" / "european_lv_mixed_vpp.yaml")
    simulator = Simulator(scenario)
    results = simulator.run_timeseries()
    output_dir = PROJECT_ROOT / "outputs"
    paths = simulator.export_results(output_dir)
    deep_rl_frames = load_deep_rl_frames(output_dir)
    dashboard_frames = build_dashboard_frames(
        scenario.net,
        scenario.vpps,
        results,
        dt_hours=scenario.dt_hours,
        deep_summary=deep_rl_frames.get("deep_rl_training_summary"),
        deep_rl_frames=deep_rl_frames,
    )
    dashboard_paths = export_dashboard_frames(dashboard_frames, output_dir / "dashboard_data")
    dispatch_report_path = export_vpp_dispatch_instruction_report(
        dashboard_frames["vpp_dispatch_explanation"],
        output_dir / "vpp_daily_dispatch_instructions.md",
    )
    figure_paths = plot_timeseries_results(results, output_dir / "figures")
    topology_paths = plot_topology_report(dashboard_frames, output_dir / "figures")
    interactive_report_path = build_interactive_report_html(
        dashboard_frames,
        output_dir / "interactive_report.html",
    )
    first_person_paths = build_first_person_reports(
        dashboard_frames,
        output_dir / "vpp_first_person",
    )
    rl_architecture_report_path = build_rl_architecture_report_html(
        dashboard_frames,
        output_dir / "rl_architecture.html",
    )
    summary = simulator._summary(results)
    print(f"steps={summary['steps']}")
    print(f"min_voltage_vm_pu={summary['min_voltage_vm_pu']:.4f}")
    print(f"max_line_loading_percent={summary['max_line_loading_percent']:.2f}")
    print(
        f"csv_outputs={len(paths)} dashboard_tables={len(dashboard_paths)} "
        f"figures={len(figure_paths) + len(topology_paths)}"
    )
    print(f"outputs_dir={output_dir}")
    print(f"interactive_report={interactive_report_path}")
    print(f"rl_architecture_report={rl_architecture_report_path}")
    print(f"vpp_first_person_report={first_person_paths['index']}")
    print(f"vpp_dispatch_report={dispatch_report_path}")


if __name__ == "__main__":
    main()
