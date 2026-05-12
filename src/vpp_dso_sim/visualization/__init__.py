"""Plotting, topology, and dashboard data helpers."""

from vpp_dso_sim.visualization.dashboard_data import build_dashboard_frames, export_dashboard_frames
from vpp_dso_sim.visualization.benchmark_report import export_benchmark_visualization_outputs
from vpp_dso_sim.visualization.dispatch_explanations import (
    build_vpp_dispatch_explanations,
    export_vpp_dispatch_instruction_report,
)
from vpp_dso_sim.visualization.first_person_report import build_first_person_reports
from vpp_dso_sim.visualization.high_severity_fix_explainer import build_high_severity_fix_explainer_html
from vpp_dso_sim.visualization.interactive_report import build_interactive_report_html
from vpp_dso_sim.visualization.rl_architecture_report import build_rl_architecture_report_html
from vpp_dso_sim.visualization.topology_plots import plot_topology_report, plot_topology_state

__all__ = [
    "export_algorithm_search_report",
    "build_interactive_report_html",
    "build_rl_architecture_report_html",
    "build_first_person_reports",
    "build_high_severity_fix_explainer_html",
    "export_benchmark_visualization_outputs",
    "build_dashboard_frames",
    "build_vpp_dispatch_explanations",
    "export_dashboard_frames",
    "export_vpp_dispatch_instruction_report",
    "plot_topology_report",
    "plot_topology_state",
]


def __getattr__(name: str):
    if name == "export_algorithm_search_report":
        from vpp_dso_sim.visualization.algorithm_search_report import export_algorithm_search_report

        return export_algorithm_search_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
