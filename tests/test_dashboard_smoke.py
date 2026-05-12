from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from vpp_dso_sim.dashboard.app import create_dashboard_app, load_dashboard_frames
from vpp_dso_sim.learning.tuning import TrainingSupervisor, TuningConfig
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator
from vpp_dso_sim.utils.io import ensure_dir
from vpp_dso_sim.visualization.dashboard_data import build_dashboard_frames, export_dashboard_frames


def test_load_dashboard_frames_from_exported_csv():
    scenario = load_scenario()
    simulator = Simulator(scenario)
    results = simulator.run_timeseries(horizon_steps=2)
    frames = build_dashboard_frames(scenario.net, scenario.vpps, results, dt_hours=scenario.dt_hours)
    output_dir = Path("outputs") / "test_dashboard_data"
    export_dashboard_frames(frames, output_dir)

    loaded = load_dashboard_frames(output_dir)

    assert len(loaded["network_nodes"]) == len(frames["network_nodes"])
    assert len(loaded["network_edges"]) == len(frames["network_edges"])
    assert len(loaded["bus_state"]) == len(frames["bus_state"])
    assert len(loaded["vpp_dispatch_explanation"]) == len(frames["vpp_dispatch_explanation"])
    assert len(loaded["vpp_first_person_timeline"]) == len(frames["vpp_first_person_timeline"])
    assert len(loaded["rl_target_ctde_architecture"]) == len(frames["rl_target_ctde_architecture"])
    assert len(loaded["rl_algorithm_variants"]) == len(frames["rl_algorithm_variants"])
    assert len(loaded["model_update_summary"]) == len(frames["model_update_summary"])
    assert "centralized_training_critic" in set(loaded["rl_target_ctde_architecture"]["component_id"])
    assert {"happo", "matd3", "hasac"}.issubset(set(loaded["rl_algorithm_variants"]["algorithm_id"]))


def test_load_dashboard_frames_reads_optional_training_outputs():
    output_dir = ensure_dir("outputs/test_dashboard_training")
    TrainingSupervisor(
        TuningConfig(
            algorithms=("ippo",),
            action_scales=(0.05,),
            exploration_noises=(0.0,),
            horizon_steps=2,
            episodes_per_trial=1,
        )
    ).run(output_dir=output_dir)

    loaded = load_dashboard_frames(output_dir)

    assert not loaded["training_summary"].empty
    assert not loaded["tuning_trials"].empty
    assert {"status", "best_algorithm", "needs_algorithm_review"}.issubset(loaded["training_summary"].columns)


def test_create_dashboard_app_optional_dependency_behavior():
    scenario = load_scenario()
    simulator = Simulator(scenario)
    results = simulator.run_timeseries(horizon_steps=1)
    frames = build_dashboard_frames(scenario.net, scenario.vpps, results, dt_hours=scenario.dt_hours)

    if importlib.util.find_spec("dash") is None:
        with pytest.raises(ImportError, match="Dash is required"):
            create_dashboard_app(frames=frames)
    else:
        app = create_dashboard_app(frames=frames)
        assert app.layout is not None
        layout_text = str(app.layout)
        assert "language-toolbar" in layout_text
        assert "data-lang-switch" in layout_text
        assert "dispatch-vpp-card" in layout_text
        assert "What Drives VPP Decisions" in layout_text
        assert "FR/DOE Envelope" in layout_text
        assert "Projection Audit" in layout_text
        assert "Privacy Visibility" in layout_text
        assert "Training Supervisor Status" in layout_text
        assert "Model / Algorithm Update Summary" in layout_text
        assert "RL Algorithm Variants" in layout_text
        assert "MARL Baselines" in layout_text
        assert "VPP 决策由什么驱动" in layout_text
        assert "Saw / Inferred / Decided Replay" in layout_text
        assert "lang-copy" in app.index_string
