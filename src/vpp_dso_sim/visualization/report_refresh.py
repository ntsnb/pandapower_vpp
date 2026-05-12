from __future__ import annotations

from pathlib import Path

import pandas as pd

from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator
from vpp_dso_sim.visualization.dashboard_data import (
    DEEP_RL_FRAME_NAMES,
    build_dashboard_frames,
    export_dashboard_frames,
)
from vpp_dso_sim.visualization.dispatch_explanations import export_vpp_dispatch_instruction_report
from vpp_dso_sim.visualization.first_person_report import build_first_person_reports
from vpp_dso_sim.visualization.interactive_report import build_interactive_report_html
from vpp_dso_sim.visualization.plots import plot_timeseries_results
from vpp_dso_sim.visualization.rl_architecture_report import build_rl_architecture_report_html
from vpp_dso_sim.visualization.topology_plots import plot_topology_report


def load_deep_rl_frames(output_dir: str | Path, deep_rl_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load deep-RL training artifacts for dashboard/report synchronization."""

    root = Path(deep_rl_dir) if deep_rl_dir is not None else Path(output_dir) / "deep_rl"
    frames: dict[str, pd.DataFrame] = {}
    for name in DEEP_RL_FRAME_NAMES:
        path = root / f"{name}.csv"
        frames[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return frames


def refresh_visualization_outputs(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    deep_rl_dir: str | Path | None = None,
) -> dict[str, Path | dict[str, Path]]:
    """Regenerate all static visualization artifacts from the latest simulation and learning outputs."""

    out = Path(output_dir)
    scenario = load_scenario(config_path)
    simulator = Simulator(scenario)
    results = simulator.run_timeseries()
    csv_paths = simulator.export_results(out)
    deep_rl_frames = load_deep_rl_frames(out, deep_rl_dir=deep_rl_dir)
    dashboard_frames = build_dashboard_frames(
        scenario.net,
        scenario.vpps,
        results,
        dt_hours=scenario.dt_hours,
        deep_summary=deep_rl_frames.get("deep_rl_training_summary"),
        deep_rl_frames=deep_rl_frames,
    )
    dashboard_paths = export_dashboard_frames(dashboard_frames, out / "dashboard_data")
    figure_paths = plot_timeseries_results(results, out / "figures")
    topology_paths = plot_topology_report(dashboard_frames, out / "figures")
    interactive_report_path = build_interactive_report_html(dashboard_frames, out / "interactive_report.html")
    first_person_paths = build_first_person_reports(dashboard_frames, out / "vpp_first_person")
    rl_architecture_report_path = build_rl_architecture_report_html(dashboard_frames, out / "rl_architecture.html")
    dispatch_report_path = export_vpp_dispatch_instruction_report(
        dashboard_frames["vpp_dispatch_explanation"],
        out / "vpp_daily_dispatch_instructions.md",
    )
    return {
        "outputs_dir": out,
        "csv_outputs": csv_paths,
        "dashboard_tables": dashboard_paths,
        "figures": {**figure_paths, **topology_paths},
        "interactive_report": interactive_report_path,
        "rl_architecture_report": rl_architecture_report_path,
        "first_person_reports": first_person_paths,
        "dispatch_report": dispatch_report_path,
        "model_update_summary": dashboard_paths["model_update_summary"],
    }
