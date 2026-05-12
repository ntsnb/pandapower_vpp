from __future__ import annotations

from pathlib import Path

from vpp_dso_sim.learning.marl_baselines import CLASSIC_BASELINES, run_marl_baselines
from vpp_dso_sim.learning.tuning import TrainingSupervisor, TuningConfig


def main() -> None:
    output_dir = Path("outputs") / "marl_baselines"
    baseline = run_marl_baselines(
        output_dir=output_dir,
        algorithms=CLASSIC_BASELINES,
        horizon_steps=8,
        episodes=2,
        action_scale=0.10,
        exploration_noise=0.02,
        seed=42,
    )
    supervisor = TrainingSupervisor(
        TuningConfig(
            algorithms=CLASSIC_BASELINES,
            action_scales=(0.05, 0.10),
            exploration_noises=(0.0, 0.02),
            horizon_steps=6,
            episodes_per_trial=1,
            seed=123,
        )
    )
    tuning = supervisor.run(output_dir=output_dir)
    print(f"output_dir={baseline['output_dir'].resolve()}")
    print(f"algorithms={','.join(CLASSIC_BASELINES)}")
    print(f"best_algorithm={baseline['summary']['best_algorithm']}")
    print(f"best_episode_reward={baseline['summary']['best_episode_reward']:.3f}")
    print(f"training_status={tuning['summary']['status']}")
    print(f"deep_learning_available={tuning['summary']['deep_learning_available']}")
    print("files=agent_role_map.csv,encoder_role_map.csv,episode_metrics.csv,step_metrics.csv,tuning_trials.csv,training_summary.csv,training_summary.json")


if __name__ == "__main__":
    main()
