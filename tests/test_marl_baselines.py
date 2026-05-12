from __future__ import annotations

from vpp_dso_sim.learning.agent_roles import build_agent_role_map, build_encoder_role_map
from vpp_dso_sim.learning.marl_baselines import run_marl_baselines
from vpp_dso_sim.learning.tuning import TrainingSupervisor, TuningConfig
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.utils.io import ensure_dir


def test_agent_role_map_contains_dso_dispatch_portfolio_and_supervisor():
    scenario = load_scenario()
    roles = build_agent_role_map(scenario.vpps)
    role_types = {role.role_type for role in roles}

    assert "dso_guidance_agent" in role_types
    assert "vpp_dispatch_agent" in role_types
    assert "vpp_portfolio_agent" in role_types
    assert "training_supervisor_agent" in role_types
    assert len([role for role in roles if role.role_type == "vpp_dispatch_agent"]) == len(scenario.vpps)
    assert len(build_encoder_role_map()) == 3


def test_marl_baseline_smoke_outputs_metrics():
    output_dir = ensure_dir("outputs/test_marl_baselines")
    result = run_marl_baselines(
        output_dir=output_dir,
        algorithms=("ippo", "mappo"),
        horizon_steps=2,
        episodes=1,
        action_scale=0.05,
        exploration_noise=0.0,
    )

    assert set(result["episode_metrics"]["algorithm"]) == {"ippo", "mappo"}
    assert not result["step_metrics"].empty
    assert (output_dir / "agent_role_map.csv").exists()
    assert (output_dir / "encoder_role_map.csv").exists()
    assert (output_dir / "baseline_summary.json").exists()


def test_training_supervisor_returns_convergence_or_review_status():
    output_dir = ensure_dir("outputs/test_training_supervisor")
    supervisor = TrainingSupervisor(
        TuningConfig(
            algorithms=("ippo",),
            action_scales=(0.05,),
            exploration_noises=(0.0,),
            horizon_steps=2,
            episodes_per_trial=1,
        )
    )

    result = supervisor.run(output_dir=output_dir)

    assert not result["tuning_trials"].empty
    assert result["summary"]["status"] in {"converged", "needs_algorithm_review"}
    assert "deep_learning_available" in result["summary"]
    assert (output_dir / "tuning_trials.csv").exists()
    assert (output_dir / "training_summary.csv").exists()
    assert (output_dir / "training_summary.json").exists()


def test_training_supervisor_non_convergence_handoff():
    output_dir = ensure_dir("outputs/test_training_supervisor_handoff")
    supervisor = TrainingSupervisor(
        TuningConfig(
            algorithms=("ippo",),
            action_scales=(0.05,),
            exploration_noises=(0.0,),
            horizon_steps=2,
            episodes_per_trial=1,
            max_violation_count=-1,
        )
    )

    result = supervisor.run(output_dir=output_dir)
    summary = result["summary"]

    assert summary["status"] == "needs_algorithm_review"
    assert summary["needs_algorithm_review"] is True
    assert summary["handoff_target"] == "main_thread"
    assert "algorithm agent" in summary["handoff_message"]
